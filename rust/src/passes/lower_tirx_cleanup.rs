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

//! `tirx.LowerTIRxCleanup` (`src/tirx/transform/lower_tirx_cleanup.cc`): the final stage of
//! TIRx lowering. Every tensor is flattened to its storage, an access through a layout becomes
//! the layout's element offset, and every `tensor_data_ptr` projects the storage root.

use std::collections::HashMap;

use tvm_ffi::{
    dispatch, structural_mutate, Any, Array, Map, Mutator, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, String,
};

use super::analyzer::{AnalyzerBase, AnalyzerScope, AnalyzerState};
use super::flattened_tensor::{
    as_tensor_var, elem_offset, flattened_tensor, structural_equal, tensor_data, tensor_type,
    tensor_var,
};
use super::util::{function_pass, get_attr, int_imm, prim_dtype, MutatorExt};
use crate::ir::{Bind, Call, CallObj, Expr, Op, SeqStmt, Stmt, TensorLoad, TensorStore, Type, Var};
use crate::ops::{int32, unknown_loc, Ex, PrimOps};
use crate::sym::Analyzer;
use crate::tirx::{Axis, ComposeLayout, Layout, TensorType, TileLayout};
use crate::transform::Pass;

/// Where an expression of a tensor's geometry is rewritten: before the body, each in a traversal
/// of its own, or inside the body's traversal.
enum Site<'a> {
    Params,
    Body(&'a mut Mutator),
}

struct LayoutApplier {
    base: AnalyzerBase,
    /// `VarRemapSet(buf, flattened)`, and every definition the default descent rebuilt.
    remap: HashMap<ObjectIdentity, Var>,
    /// The storage root each tensor aliases, recorded at its definition.
    tensor_aliases: HashMap<ObjectIdentity, Var>,
    alloc_tensor: Op,
    decl_tensor: Op,
    decl_tmem: Op,
    reinterpret: Op,
    tensor_data_ptr: Op,
}

impl AnalyzerState for LayoutApplier {
    fn base(&mut self) -> &mut AnalyzerBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = AnalyzerScope)]
impl LayoutApplier {
    /// A use of a remapped variable is its replacement; a definition the default descent rebuilds
    /// is recorded as the engine records it, so later lookups see it too.
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

    /// `tensor_data_ptr(t)` projects the storage root of `t`.
    fn mutate_call(&mut self, call: Call, m: &mut Mutator) -> Result<Any> {
        if call.op.same_as(&self.tensor_data_ptr) && call.args.len() == 1 {
            let arg = call.args.get(0)?;
            if let Ok(tensor) = arg.clone().try_cast::<Var>() {
                if tensor_type(&tensor).is_ok() {
                    let Some(root) = self.tensor_aliases.get(&ObjectIdentity::of(&tensor)) else {
                        let message = format!(
                            "Check failed: (root_opt.has_value()) is false: tensor_data_ptr \
                             projects {}, which has no visible definition \
                             (AllocTensor/DeclTensor/Function parameter) at this point",
                            tensor.name.as_str()
                        );
                        return Err(tvm_ffi::Error::new(tvm_ffi::INTERNAL_ERROR, &message, ""));
                    };
                    let mut root = root.clone();
                    if let Some(mapped) = self.remap.get(&ObjectIdentity::of(&root)) {
                        root = mapped.clone();
                    }
                    return Ok(tensor_data(&root)?.into());
                }
            }
        }
        m.default_mutate(self, &call)
    }

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        let var = op.var.clone();
        // `Bind(t, source)` of a tensor declares `t` over the source's data.
        if let Some((source, _)) = as_tensor_var(&op.value) {
            let declaration = self.declaration(&var, tensor_data(&source)?)?;
            let binding = Bind::new_full(var, declaration.clone(), op.loc.clone())?;
            let result = self.mutate_decl_tensor(&binding, &declaration, m)?;
            return Ok(result.unwrap_or(binding).into());
        }
        if op.value.as_node::<CallObj>().is_some() {
            let call: Call = op.value.clone().try_cast()?;
            if call.op.same_as(&self.alloc_tensor) {
                return self.mutate_alloc_tensor(&op, &call, m);
            }
            if call.op.same_as(&self.decl_tensor) {
                return Ok(self.mutate_decl_tensor(&op, &call, m)?.unwrap_or(op).into());
            }
            if call.op.same_as(&self.decl_tmem) {
                // TMEM is allocated by the pool; keep an explicit backing pointer so data
                // projections work for views and parameters too.
                let ty = tensor_type(&var)?;
                let pointer = Call::new(
                    data_ptr_type(&ty)?,
                    self.reinterpret.clone(),
                    vec![call.args.get(0)?],
                )?;
                let declaration = self.declaration(&var, pointer.into())?;
                let binding = Bind::new_full(var, declaration.clone(), op.loc.clone())?;
                let result = self.mutate_decl_tensor(&binding, &declaration, m)?;
                return Ok(result.unwrap_or(binding).into());
            }
        }
        m.default_mutate(self, &op)
    }

    /// The logical tensor stays until the access linearizes its indices.
    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let tensor: Var = op.dest.clone().try_cast()?;
        let indices = self.simplified_elem_offset(&tensor, &indices)?;
        let flattened = self.flattened(&mut Site::Body(m), &tensor, false)?;
        Ok(TensorStore::new(op.loc.clone(), flattened.into(), indices, value).into())
    }

    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let tensor: Var = op.source.clone().try_cast()?;
        let indices = self.simplified_elem_offset(&tensor, &indices)?;
        let flattened = self.flattened(&mut Site::Body(m), &tensor, false)?;
        Ok(TensorLoad::new_full(flattened, indices.iter(), op.loc.clone())?.into())
    }
}

/// C++ `TensorTypeNode::DataPtrType()`: `PtrType(dtype, scope)`.
fn data_ptr_type(ty: &TensorType) -> Result<Type> {
    tvm_ffi::cached_global_func!("ir.PtrType")
        .call_tuple((Type::from(ty.dtype.clone()), ty.scope.clone()))?
        .try_into()
}

fn axis_is(predicate: &str, axis: &Axis) -> Result<bool> {
    tvm_ffi::Function::get_global(predicate)?.call_tuple((axis.clone(),))?.try_into()
}

impl LayoutApplier {
    fn new(analyzer: Analyzer) -> Result<Self> {
        Ok(Self {
            base: AnalyzerBase::new(analyzer)?,
            remap: HashMap::new(),
            tensor_aliases: HashMap::new(),
            alloc_tensor: Op::get("tirx.alloc_tensor")?,
            decl_tensor: Op::get("tirx.decl_tensor")?,
            decl_tmem: Op::get("tirx.cuda.decl_tmem")?,
            reinterpret: Op::get("tirx.reinterpret")?,
            tensor_data_ptr: Op::get("tirx.tensor_data_ptr")?,
        })
    }

    /// `Call(t.ty, decl_tensor, {data}, {}, {t.ty})`.
    fn declaration(&self, tensor: &Var, data: Expr) -> Result<Call> {
        let ty = tensor.ty.clone();
        Call::new_full(
            Some(ty.clone()),
            self.decl_tensor.clone().into(),
            Array::new(vec![data]),
            None,
            Array::new(vec![ty]),
            unknown_loc()?,
        )
    }

    /// `Mutate(expr)` at `site`.
    fn mutate_expr(&mut self, site: &mut Site, expr: &Expr) -> Result<Expr> {
        match site {
            Site::Params => structural_mutate(expr.clone(), &mut *self)?.try_into(),
            Site::Body(m) => m.mutate_as(self, expr),
        }
    }

    fn mutate_alloc_tensor(&mut self, op: &Bind, call: &Call, m: &mut Mutator) -> Result<Any> {
        self.tensor_aliases.insert(ObjectIdentity::of(&op.var), op.var.clone());
        let tensor = self.flattened(&mut Site::Body(&mut *m), &op.var, true)?;
        let mut args: Vec<Expr> = call.args.iter().collect();
        if let Some(first) = args.first_mut() {
            *first = m.mutate_as(self, &*first)?;
        }
        let ty = tensor.ty.clone();
        let value = Call::new_full(
            Some(ty.clone()),
            self.alloc_tensor.clone().into(),
            Array::new(args),
            call.attrs.clone(),
            Array::new(vec![ty]),
            call.loc.clone(),
        )?;
        Ok(Bind::new_full(tensor, value, op.loc.clone())?.into())
    }

    /// The rebuilt binding, or `None` when nothing changed.
    fn mutate_decl_tensor(
        &mut self,
        op: &Bind,
        call: &Call,
        m: &mut Mutator,
    ) -> Result<Option<Bind>> {
        let source = call.args.get(0)?;
        self.register_alias(&op.var, &source)?;
        let data: Expr = m.mutate_as(self, &source)?;
        let tensor = self.flattened(&mut Site::Body(&mut *m), &op.var, false)?;
        if tensor.same_as(&op.var) && data.same_as(&source) {
            return Ok(None);
        }
        let ty = tensor.ty.clone();
        let value = Call::new_full(
            Some(ty.clone()),
            self.decl_tensor.clone().into(),
            Array::new(vec![data]),
            call.attrs.clone(),
            Array::new(vec![ty]),
            call.loc.clone(),
        )?;
        Ok(Some(Bind::new_full(tensor, value, op.loc.clone())?))
    }

    /// `RegisterTensorAlias`: a tensor declared over another tensor's data shares its root.
    fn register_alias(&mut self, tensor: &Var, data: &Expr) -> Result<()> {
        let mut root = tensor.clone();
        if let Some(call) = data.as_node::<CallObj>() {
            if call.op.same_as(&self.tensor_data_ptr) && call.args.len() == 1 {
                if let Ok(source) = call.args.get(0)?.try_cast::<Var>() {
                    if tensor_type(&source).is_ok() {
                        let Some(source_root) =
                            self.tensor_aliases.get(&ObjectIdentity::of(&source))
                        else {
                            let message = format!(
                                "Check failed: (source_root.has_value()) is false: Tensor alias \
                                 source {} must be registered before its DeclTensor alias",
                                source.name.as_str()
                            );
                            return Err(tvm_ffi::Error::new(tvm_ffi::INTERNAL_ERROR, &message, ""));
                        };
                        root = source_root.clone();
                    }
                }
            }
        }
        self.tensor_aliases.insert(ObjectIdentity::of(tensor), root);
        Ok(())
    }

    /// `GetFlattenedTensor(buf, is_alloc)`: the storage tensor of `buf`. An allocation through a
    /// layout covers the layout's memory span.
    fn flattened(&mut self, site: &mut Site, buf: &Var, is_alloc: bool) -> Result<Var> {
        if let Some(mapped) = self.remap.get(&ObjectIdentity::of(buf)) {
            return Ok(mapped.clone());
        }
        let buf_ty = tensor_type(buf)?;
        let layout = buf_ty.layout.clone();
        let tile = layout.as_ref().and_then(|l| l.clone().try_cast::<TileLayout>().ok());
        let compose = layout.as_ref().and_then(|l| l.clone().try_cast::<ComposeLayout>().ok());
        let (flattened, base_ty, shape, strides) = match (is_alloc, tile, compose) {
            (true, Some(tile), _) => {
                // The allocation covers the memory-axis span, gaps included.
                let analyzer = Analyzer::new()?;
                let mut span = Ex::from(int32(1));
                for iter in tile.shard()?.iter().chain(tile.replica()?.iter()) {
                    if axis_is("tirx.AxisIsMemoryAxis", &iter.axis)? {
                        span += (Ex::from(&iter.extent) - 1) * &iter.stride;
                    }
                }
                let offset: Map<Axis, Expr> = tile.offset()?;
                for (axis, offset) in offset.iter() {
                    if axis_is("tirx.AxisIsMemoryAxis", &axis)? {
                        span += &offset;
                    }
                }
                let span = analyzer.simplify(&span.expr()?, 2)?;
                (buf.clone(), buf_ty.clone(), Array::new(vec![span]), Array::new(vec![]))
            }
            (true, None, Some(compose)) => {
                // A swizzle permutes addresses: round the memory span to a block whose upper bits
                // the swizzle cannot change.
                let tile_layout = compose.tile_layout()?;
                let trivial: bool = tvm_ffi::cached_global_func!("tirx.TileLayoutIsTrivial")
                    .call_tuple((tile_layout.clone(),))?
                    .try_into()?;
                let mut span: Expr = if trivial {
                    let mut span = Ex::from(int32(1));
                    for extent in buf_ty.shape.iter() {
                        span *= &extent;
                    }
                    span.expr()?
                } else {
                    let layout: Layout = tile_layout.into();
                    tvm_ffi::cached_global_func!("tirx.LayoutGetSpan")
                        .call_tuple((layout, String::from("m")))?
                        .try_into()?
                };
                let swizzle_len = compose.swizzle_len()?;
                if swizzle_len != 0 {
                    let mut changed_bits = compose.per_element()? + swizzle_len;
                    if !compose.swizzle_inner()? {
                        changed_bits += compose.atom_len()?;
                    }
                    let block: Expr = int_imm(prim_dtype(&span)?, 1i128 << changed_bits)?.into();
                    span = ((Ex::from(&span) + &block - 1).floordiv(&block) * &block).expr()?;
                }
                let span = self.base.analyzer.simplify(&span, 2)?;
                (buf.clone(), buf_ty.clone(), Array::new(vec![span]), Array::new(vec![]))
            }
            _ => {
                let (flattened, flat_ty) = flattened_tensor(buf, &buf_ty)?;
                let (shape, strides) = (flat_ty.shape.clone(), flat_ty.strides.clone());
                (flattened, flat_ty, shape, strides)
            }
        };
        // Variables the pass already rebuilt are remapped (a shape may load from another local
        // tensor), then the shape is canonicalized.
        let mut new_shape = Vec::with_capacity(shape.len());
        for extent in shape.iter() {
            let extent = self.mutate_expr(site, &extent)?;
            new_shape.push(self.base.analyzer.canonical_simplify(&extent)?);
        }
        let mut new_strides = Vec::with_capacity(strides.len());
        for stride in strides.iter() {
            new_strides.push(self.mutate_expr(site, &stride)?);
        }
        let elem_offset = self.mutate_expr(site, &buf_ty.elem_offset)?;
        let ty = TensorType::new(
            base_ty.loc.clone(),
            base_ty.dtype.clone(),
            base_ty.scope.clone(),
            Array::new(new_shape),
            Array::new(new_strides),
            elem_offset,
            base_ty.data_alignment,
            base_ty.offset_factor,
            None,
        );
        if structural_equal(&buf_ty, &ty)? {
            return Ok(buf.clone());
        }
        let flattened = tensor_var(&flattened.name, ty, &flattened.loc)?;
        self.remap.insert(ObjectIdentity::of(buf), flattened.clone());
        Ok(flattened)
    }

    /// `GetSimplifiedElemOffset`: the element offset of an access, through the tensor's layout
    /// when it has one.
    fn simplified_elem_offset(&self, tensor: &Var, indices: &Array<Expr>) -> Result<Array<Expr>> {
        let ty = tensor_type(tensor)?;
        if let Some(layout) = ty.layout.clone() {
            if let Ok(tile) = layout.clone().try_cast::<TileLayout>() {
                for iter in tile.shard()?.iter() {
                    if axis_is("tirx.AxisIsThreadAxis", &iter.axis)? {
                        let message = "Cannot lower direct TensorLoad/TensorStore on a tensor \
                                       with thread-axis layout: unable to verify that the \
                                       coordinate matches the current thread. Use .view() + \
                                       .local() to decompose thread and memory axes.";
                        return Err(tvm_ffi::Error::new(tvm_ffi::INTERNAL_ERROR, message, ""));
                    }
                }
            }
            let canonical: Layout = tvm_ffi::cached_global_func!("tirx.LayoutCanonicalize")
                .call_tuple((layout,))?
                .try_into()?;
            let offsets: Map<String, Expr> =
                tvm_ffi::cached_global_func!("tirx.LayoutApplyWithShape")
                    .call_tuple((canonical, indices.clone(), ty.shape.clone()))?
                    .try_into()?;
            if offsets.len() != 1 {
                let message = format!(
                    "Check failed: res.size() == 1 ({} vs. 1) : Expected a single element offset",
                    offsets.len()
                );
                return Err(tvm_ffi::Error::new(tvm_ffi::INTERNAL_ERROR, &message, ""));
            }
            let (_, offset) = offsets.iter().next().expect("one offset");
            return Ok(Array::new(vec![self.base.analyzer.simplify(&offset, 2)?]));
        }
        let flattened = elem_offset(&ty, indices, true)?;
        Ok(Array::new(vec![self.base.analyzer.simplify(&flattened.get(0)?, 2)?]))
    }
}

/// C++ `LayoutApplier::Flatten(body, params, target)`: the flattened body and the parameters.
fn flatten(body: SeqStmt, params: &Array<Var>) -> Result<(Stmt, Array<Var>)> {
    let mut applier = LayoutApplier::new(Analyzer::new()?)?;
    let mut new_params = Vec::with_capacity(params.len());
    let mut aliased: Vec<(Var, Var)> = Vec::new();
    for param in params.iter() {
        let Some((tensor, ty)) = as_tensor_var(&param.clone().into()) else {
            new_params.push(param);
            continue;
        };
        applier.tensor_aliases.insert(ObjectIdentity::of(&tensor), tensor.clone());
        let flattened = applier.flattened(&mut Site::Params, &tensor, false)?;
        let mut source = tensor.clone();
        if ty.layout.is_some() {
            let without_layout = TensorType::new(
                ty.loc.clone(),
                ty.dtype.clone(),
                ty.scope.clone(),
                ty.shape.clone(),
                ty.strides.clone(),
                ty.elem_offset.clone(),
                ty.data_alignment,
                ty.offset_factor,
                None,
            );
            source = tensor_var(&tensor.name, without_layout, &tensor.loc)?;
        }
        if !flattened.same_as(&source) {
            aliased.push((flattened, source.clone()));
        }
        new_params.push(source);
    }
    let mut stmt: Stmt = structural_mutate(body, &mut applier)?.try_into()?;
    for (buf, source) in aliased {
        let declaration = applier.declaration(&buf, tensor_data(&source)?)?;
        stmt = SeqStmt::new(vec![Bind::new(buf, declaration)?.into(), stmt])?.into();
    }
    Ok((stmt, Array::new(new_params)))
}

pub fn lower_tirx_cleanup() -> Result<Pass> {
    function_pass("tirx.LowerTIRxCleanup", 0, |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        // `ResolveTarget`: the function's target, or the current one, which must exist.
        if get_attr(&func.attrs, "target")?.is_none() {
            tvm_ffi::cached_global_func!("target.TargetCurrent").call_tuple((false,))?;
        }
        let (body, params) = flatten(body, &func.params)?;
        let body = SeqStmt::from_stmt(body)?;
        Ok(Some(func.with(|node| {
            node.body = Some(body);
            node.params = params;
        })))
    })
}
