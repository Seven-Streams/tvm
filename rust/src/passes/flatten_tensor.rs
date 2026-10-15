/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! `tirx.FlattenTensor` (`src/tirx/transform/flatten_tensor.cc`): flatten each n-d tensor into a
//! 1-d storage view, rewriting every access `buf[x]` into `buf'[f(x)]`.
//!
//! At each tensor definition (`alloc_tensor` / `decl_tensor` binding, or a function parameter)
//! the pass derives, once, the fold view (the original geometry with its expressions rewritten,
//! the source of `f`) and the flattened storage husk `buf'`. Every use looks the pair up.

use std::collections::{HashMap, HashSet};

use tvm_ffi::{
    dispatch, structural_mutate, Any, Array, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore,
    Result, INTERNAL_ERROR,
};

use super::analyzer::{is_const_int, AnalyzerBase, AnalyzerScope, AnalyzerState};
use super::flattened_tensor::{as_tensor_var, elem_offset, flattened_tensor, tensor_type};
use super::stmt_simplify::mark_global_non_neg_value;
use super::util::{function_pass, int_imm, prim_dtype, MutatorExt};
use crate::ir::{Bind, Call, CallObj, Expr, SeqStmt, Stmt, TensorLoad, TensorStore, Var};
use crate::ops::structural_equal;
use crate::sym::Analyzer;
use crate::tirx::op::{ALLOC_TENSOR, DECL_TENSOR, MASKED_LOAD, MASKED_STORE, TENSOR_DATA_PTR};
use crate::tirx::{Function, Iter, TensorType, TileLayout};
use crate::transform::Pass;

/// The pair derived at a tensor's definition.
struct FlatInfo {
    /// The original geometry with rewritten expressions; only folds indices, never emitted.
    fold_view: TensorType,
    /// The 1-d storage husk `buf'`.
    flattened: Var,
}

/// Where an expression of a tensor's geometry is rewritten: before the body, each in a traversal
/// of its own, or inside the body's traversal.
enum Site<'a> {
    Params,
    Body(&'a mut Mutator),
}

pub(crate) struct TensorFlattener {
    base: AnalyzerBase,
    flat: HashMap<ObjectIdentity, FlatInfo>,
    /// `VarRemapSet(buf, flattened)`: stray uses of a tensor variable follow its husk.
    remap: HashMap<ObjectIdentity, Var>,
    /// Tensors whose storage is a function parameter.
    extern_tensors: HashSet<ObjectIdentity>,
    /// Tensors accessed in the body.
    tensors_used: HashSet<ObjectIdentity>,
}

impl AnalyzerState for TensorFlattener {
    fn base(&mut self) -> &mut AnalyzerBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = AnalyzerScope)]
impl TensorFlattener {
    /// A use of a remapped tensor variable is its husk; a definition the default descent
    /// rebuilds is recorded as the engine records it, so later lookups see it too.
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        if let Some(mapped) = self.remap.get(&ObjectIdentity::of(&var)) {
            return Ok(mapped.clone().into());
        }
        let result = m.default_mutate(self, &var)?;
        if let Ok(rebuilt) = Var::try_from(result.clone()) {
            if !rebuilt.same_as(&var) {
                self.remap.insert(ObjectIdentity::of(&var), rebuilt);
            }
        }
        Ok(result)
    }

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if ALLOC_TENSOR.is(&call.op) {
                return self.mutate_alloc_tensor(op.clone(), op.value.clone().try_cast()?, m);
            }
            if DECL_TENSOR.is(&call.op) {
                return self.mutate_decl_tensor(op.clone(), op.value.clone().try_cast()?, m);
            }
        }
        m.default_mutate(self, &op)
    }

    /// The tensor and its indices are flattened together.
    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let original: Var = op.dest.clone().try_cast()?;
        let (flattened, indices) = self.access(&original, &indices)?;
        let store =
            TensorStore::from_complete_fields(op.loc.clone(), flattened.into(), indices, value);
        Ok(store.into())
    }

    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let original: Var = op.source.clone().try_cast()?;
        let (flattened, indices) = self.access(&original, &indices)?;
        Ok(TensorLoad::new_full(flattened, indices.iter(), op.loc.clone())?.into())
    }

    fn mutate_call(&mut self, call: Call, m: &mut Mutator) -> Result<Any> {
        let is_load = MASKED_LOAD.is(&call.op);
        if is_load || MASKED_STORE.is(&call.op) {
            let original: Var = call.args.get(0)?.try_cast()?;
            let count = call.args.len();
            let mut indices = Vec::new();
            for i in (if is_load { 1 } else { 2 })..count - 1 {
                indices.push(m.mutate_as(self, &call.args.get(i)?)?);
            }
            self.tensors_used.insert(ObjectIdentity::of(&original));
            let flattened = self.lookup(&original)?.flattened.clone();
            let mut args: Vec<Expr> = vec![flattened.into()];
            if !is_load {
                args.push(m.mutate_as(self, &call.args.get(1)?)?);
            }
            args.extend(self.fold_indices(&original, &Array::new(indices))?.iter());
            args.push(m.mutate_as(self, &call.args.get(count - 1)?)?);
            let call = Call::new_full(
                Some(call.ty.clone()),
                call.op.clone(),
                Array::new(args),
                call.attrs.clone(),
                call.ty_args.clone(),
                call.loc.clone(),
            )?;
            return Ok(call.into());
        }
        if TENSOR_DATA_PTR.is(&call.op) && call.args.len() == 1 {
            let arg = call.args.get(0)?;
            if let Ok(original) = arg.clone().try_cast::<Var>() {
                if tensor_type(&original).is_ok() {
                    self.tensors_used.insert(ObjectIdentity::of(&original));
                    let flattened = self.lookup(&original)?.flattened.clone();
                    return Ok(flattened.tensor_data()?.into());
                }
            }
        }
        m.default_mutate(self, &call)
    }
}

impl TensorFlattener {
    fn new(analyzer: Analyzer) -> Result<Self> {
        Ok(Self {
            base: AnalyzerBase::new(analyzer)?,
            flat: HashMap::new(),
            remap: HashMap::new(),
            extern_tensors: HashSet::new(),
            tensors_used: HashSet::new(),
        })
    }

    /// `Mutate(expr)` at `site`.
    fn mutate_expr(&mut self, site: &mut Site, expr: &Expr) -> Result<Expr> {
        match site {
            Site::Params => structural_mutate(expr.clone(), &mut *self)?.try_into(),
            Site::Body(m) => m.mutate_as(self, expr),
        }
    }

    /// `Define(buf)`: derive {fold view, husk} for `buf` at its definition; idempotent.
    fn define(&mut self, site: &mut Site, buf: &Var) -> Result<()> {
        let key = ObjectIdentity::of(buf);
        if self.flat.contains_key(&key) {
            return Ok(());
        }
        // The fold view: the geometry's expression leaves rewritten.
        let ty = tensor_type(buf)?;
        let shape = self.mutate_all(site, &ty.shape)?;
        let strides = self.mutate_all(site, &ty.strides)?;
        let elem_offset = self.mutate_expr(site, &ty.elem_offset)?;
        let mut layout = ty.layout.clone();
        if let Some(tile) = ty.layout.as_ref().and_then(|l| l.clone().try_cast::<TileLayout>().ok())
        {
            let (shard, shard_changed) = self.remap_iters(site, &tile.shard()?)?;
            let (replica, replica_changed) = self.remap_iters(site, &tile.replica()?)?;
            if shard_changed || replica_changed {
                layout = Some(TileLayout::new(&shard, &replica, tile.offset()?)?.into());
            }
        }
        let fold_view = TensorType::from_complete_fields(
            ty.loc.clone(),
            ty.dtype.clone(),
            ty.scope.clone(),
            shape,
            strides,
            elem_offset,
            ty.data_alignment,
            ty.offset_factor,
            layout,
        );
        let fold_var = Var::new_full(buf.name.clone(), fold_view.clone(), buf.loc.clone())?;

        // The husk: the linearized indices carry layout and elem_offset, so it keeps neither.
        let (_, flat_ty) = flattened_tensor(&fold_var, &fold_view)?;
        let mut shape = Vec::with_capacity(flat_ty.shape.len());
        for extent in flat_ty.shape.iter() {
            shape.push(self.base.analyzer.canonical_simplify(&extent)?);
        }
        let elem_offset = match is_const_int(&flat_ty.elem_offset, 0) {
            true => flat_ty.elem_offset.clone(),
            false => int_imm(prim_dtype(&flat_ty.elem_offset)?, 0)?.into(),
        };
        let husk = TensorType::from_complete_fields(
            flat_ty.loc.clone(),
            flat_ty.dtype.clone(),
            flat_ty.scope.clone(),
            Array::new(shape),
            flat_ty.strides.clone(),
            elem_offset,
            flat_ty.data_alignment,
            flat_ty.offset_factor,
            None,
        );
        // A body-local tensor keeps its identity when flattening changes nothing; a parameter
        // always rebuilds, as the epilogue aliases the husk onto the argument.
        let flattened = match !self.extern_tensors.contains(&key) && structural_equal(&husk, &ty)? {
            true => buf.clone(),
            false => Var::new_full(buf.name.clone(), husk, buf.loc.clone())?,
        };
        self.remap.insert(key.clone(), flattened.clone());
        self.flat.insert(key, FlatInfo { fold_view, flattened });
        Ok(())
    }

    fn mutate_all(&mut self, site: &mut Site, exprs: &Array<Expr>) -> Result<Array<Expr>> {
        let mut out = Vec::with_capacity(exprs.len());
        for expr in exprs.iter() {
            out.push(self.mutate_expr(site, &expr)?);
        }
        Ok(Array::new(out))
    }

    /// The layout iterators with their extents and strides rewritten; whether any changed.
    fn remap_iters(&mut self, site: &mut Site, iters: &Array<Iter>) -> Result<(Array<Iter>, bool)> {
        let mut out = Vec::with_capacity(iters.len());
        let mut changed = false;
        for iter in iters.iter() {
            let extent = self.mutate_expr(site, &iter.extent)?;
            let stride = self.mutate_expr(site, &iter.stride)?;
            if extent.same_as(&iter.extent) && stride.same_as(&iter.stride) {
                out.push(iter);
                continue;
            }
            changed = true;
            out.push(Iter::new(extent, stride, iter.axis.clone()));
        }
        Ok((Array::new(out), changed))
    }

    fn lookup(&self, buf: &Var) -> Result<&FlatInfo> {
        self.flat.get(&ObjectIdentity::of(buf)).ok_or_else(|| {
            error!(
                INTERNAL_ERROR,
                "Check failed: (it != flat_map_.end()) is false: Tensor {} is used before its \
                 definition (AllocTensor/DeclTensor/Function param)",
                buf.name.as_str()
            )
        })
    }

    /// `FoldIndices`: `f(indices)` from the fold view, simplified over the loop context.
    fn fold_indices(&self, original: &Var, indices: &Array<Expr>) -> Result<Array<Expr>> {
        let folded = elem_offset(&self.lookup(original)?.fold_view, indices, false)?;
        self.base.iter_map_simplify_with_context(&folded, false)
    }

    /// `VisitTensorAccess`: the husk of `original` and the folded indices.
    fn access(&mut self, original: &Var, indices: &Array<Expr>) -> Result<(Var, Array<Expr>)> {
        self.tensors_used.insert(ObjectIdentity::of(original));
        let flattened = self.lookup(original)?.flattened.clone();
        Ok((flattened, self.fold_indices(original, indices)?))
    }

    fn mutate_alloc_tensor(&mut self, op: Bind, call: Call, m: &mut Mutator) -> Result<Any> {
        self.define(&mut Site::Body(&mut *m), &op.var)?;
        // C++ `args.Set(0, ...)` copies the argument array, so a non-empty one is never the
        // same array again.
        let mut args: Vec<Expr> = call.args.iter().collect();
        if let Some(first) = args.first_mut() {
            *first = m.mutate_as(self, &*first)?;
        }
        let flattened = self.lookup(&op.var)?.flattened.clone();
        if flattened.same_as(&op.var) && args.is_empty() {
            return Ok(op.into());
        }
        self.rebind(&op, &call, flattened, args)
    }

    fn mutate_decl_tensor(&mut self, op: Bind, call: Call, m: &mut Mutator) -> Result<Any> {
        let source = call.args.get(0)?;
        let mut data = source.clone();
        if !self.is_extern_tensor_source(&source) {
            data = m.mutate_as(self, &source)?;
        }
        self.define(&mut Site::Body(&mut *m), &op.var)?;
        let flattened = self.lookup(&op.var)?.flattened.clone();
        if flattened.same_as(&op.var) && data.same_as(&source) {
            return Ok(op.into());
        }
        self.rebind(&op, &call, flattened, vec![data])
    }

    /// `tensor_data_ptr(p)` of a parameter tensor `p`: such a source is kept as it is.
    fn is_extern_tensor_source(&self, data: &Expr) -> bool {
        let Some(call) = data.as_node::<CallObj>() else { return false };
        if !TENSOR_DATA_PTR.is(&call.op) || call.args.len() != 1 {
            return false;
        }
        match call.args.get(0).ok().and_then(|arg| as_tensor_var(&arg)) {
            Some((tensor, _)) => self.extern_tensors.contains(&ObjectIdentity::of(&tensor)),
            None => false,
        }
    }

    /// `Bind(flattened, Call(ty, op, args, attrs, {ty}))` in place of `op`.
    fn rebind(&self, op: &Bind, call: &Call, flattened: Var, args: Vec<Expr>) -> Result<Any> {
        let ty = flattened.ty.clone();
        let value = Call::new_full(
            Some(ty.clone()),
            call.op.clone(),
            Array::new(args),
            call.attrs.clone(),
            Array::new(vec![ty]),
            call.loc.clone(),
        )?;
        Ok(Bind::new_full(flattened, value, op.loc.clone())?.into())
    }
}

/// C++ `TensorFlattener::Flatten(func)`.
fn flatten(func: Function) -> Result<Function> {
    let Some(body) = func.body.clone() else { return Ok(func) };
    let mut pass = TensorFlattener::new(Analyzer::new()?)?;
    // `MarkTensorParamShapes`: every shape of a tensor parameter is non-negative.
    for param in func.params.iter() {
        if let Ok(ty) = tensor_type(&param) {
            for extent in ty.shape.iter() {
                mark_global_non_neg_value(&pass.base.analyzer, &extent)?;
            }
        }
    }
    for param in func.params.iter() {
        if as_tensor_var(&param.clone().into()).is_some() {
            pass.extern_tensors.insert(ObjectIdentity::of(&param));
            pass.define(&mut Site::Params, &param)?;
        }
    }
    let result: Stmt = structural_mutate(body.clone(), &mut pass)?.try_into()?;
    let mut changed = !result.same_as(&body);
    let mut body = result;
    // Tensor parameters stay unflattened for argument validation; the body's husks alias them.
    let params: Vec<Var> = func.params.iter().collect();
    for param in params.iter().rev() {
        if as_tensor_var(&param.clone().into()).is_none()
            || !pass.tensors_used.contains(&ObjectIdentity::of(param))
        {
            continue;
        }
        let flattened = pass.lookup(param)?.flattened.clone();
        if flattened.same_as(param) {
            continue;
        }
        let ty = flattened.ty.clone();
        let alias = Call::new_full(
            Some(ty.clone()),
            DECL_TENSOR.get()?.into(),
            Array::new(vec![param.tensor_data()?]),
            None,
            Array::new(vec![ty]),
            crate::ops::unknown_loc()?,
        )?;
        body = SeqStmt::new(vec![Bind::new(flattened, alias)?.into(), body])?.into();
        changed = true;
    }
    if !changed {
        return Ok(func);
    }
    Ok(func.with_body(Some(SeqStmt::from_stmt(body)?)))
}

pub fn flatten_tensor() -> Result<Pass> {
    function_pass("tirx.FlattenTensor", 0, |func, _module, _context| Ok(Some(flatten(func)?)))
}
