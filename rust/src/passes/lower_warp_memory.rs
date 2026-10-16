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

//! `tirx.LowerWarpMemory` (`src/tirx/transform/lower_warp_memory.cc`): stripe a warp-scoped
//! tensor across the threads of a warp in local memory, store to each thread's own part, and
//! read another thread's part with a warp shuffle.
//!
//! For an allocation `warp A[n * width * m]` written as `A[m * tx + (width * m) * y + x]` with
//! `x` in `[0, m)` and `tx` the `threadIdx.x` of extent `width`, the local tensor holds `n * m`
//! elements at `m * y + x`, and a load of `A[m * z + (width * m) * y + x]` becomes a shuffle of
//! the local element `m * y + x` from thread `z`.

use std::collections::HashMap;

use tvm_ffi::{
    bail, dispatch, structural_mutate, structural_walk, Any, AnyCompatible, AnyView, Array,
    DefRegionKind, Mutator, ObjectCore, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result,
    String, WalkOrder, WalkResult, INTERNAL_ERROR, TYPE_ERROR,
};

use super::side_effect::{side_effect, PURE};
use super::substitute_dtype::{is_tensor_var, substitute_with_dtype_legalization};
use super::util::{MutatorExt, function_pass, get_attr, int_imm, prim_type};
use crate::ir::{
    Bind, Call, CallObj, Expr, For, IntImmObj, PrimType, PrimTypeObj, PtrType, PtrTypeObj, Range,
    RegionStmt, SeqStmt, Stmt, StringImmObj, TensorLoad, TensorStore, Var, VarObj,
};
use crate::ops::pattern::{ramp, PVar, Pattern};
use crate::ops::{repr, PrimOps};
use crate::prim::Ramp;
use crate::runtime::MemoryScope;
use crate::sym::{detect_linear_equation, Analyzer};
use crate::target::Target;
use crate::tirx::op::{
    ALLOC_TENSOR, CUDA_MMA_FILL, CUDA_MMA_STORE, DECL_TENSOR, GPU_WARP_ACTIVEMASK,
    GPU_WARP_SHUFFLE, LAUNCH_THREAD, PTX_LEGACY_LDMATRIX, PTX_LEGACY_MMA, TENSOR_DATA_PTR,
};
use crate::tirx::{TensorType, TensorTypeObj};
use crate::transform::Pass;

/// `GetTensorVar(expr)`: the variable of a tensor, used directly or through `tensor_data_ptr`.
fn tensor_var_of(expr: &Expr) -> Option<ObjectIdentity> {
    if expr.as_node::<VarObj>().is_some() {
        return Some(ObjectIdentity::of(expr));
    }
    let call = expr.as_node::<CallObj>()?;
    if !TENSOR_DATA_PTR.is(&call.op) || call.args.len() != 1 {
        return None;
    }
    let arg = call.args.get(0).ok()?;
    arg.as_node::<VarObj>().is_some().then(|| ObjectIdentity::of(&arg))
}

/// `as_or_throw<TensorVar>()`.
fn as_tensor_var(expr: &Expr) -> Result<ObjectIdentity> {
    match is_tensor_var(expr) {
        true => Ok(ObjectIdentity::of(expr)),
        false => bail!(TYPE_ERROR, "expected a TensorVar"),
    }
}

/// `as_or_throw<PrimVar>()`: a `Var`, not a subclass, of a primitive type.
fn as_prim_var(expr: impl Into<Expr>) -> Result<Var> {
    let expr: Expr = expr.into();
    let is_prim_var = AnyView::from(&expr).type_index() == VarObj::type_index()
        && expr.ty.as_node::<PrimTypeObj>().is_some();
    match is_prim_var {
        true => Ok(expr.try_cast()?),
        false => bail!(TYPE_ERROR, "expected a PrimVar"),
    }
}

/// The lanes of a fixed-length type (`PrimType::lanes()`).
fn lanes(expr: &Expr) -> Result<i64> {
    let lanes = expr.prim_dtype()?.lanes as i16;
    match lanes < 0 {
        true => {
            bail!(INTERNAL_ERROR, "Can't fetch the lanes of a scalable vector at a compile time.")
        }
        false => Ok(lanes as i64),
    }
}

/// The hardware axis a lexical `launch_thread` defines.
#[derive(Clone)]
struct ThreadBinding {
    tag: String,
    extent: Expr,
}

type Bindings = HashMap<ObjectIdentity, ThreadBinding>;
type Aliases = HashMap<ObjectIdentity, Expr>;

/// `ExpandWarpIndexAliases`: each bound index variable replaced by its value.
fn expand_aliases(index: &Expr, aliases: &Aliases) -> Result<Expr> {
    substitute_with_dtype_legalization(index.clone(), |var| {
        Ok(aliases.get(&ObjectIdentity::of(var)).cloned())
    })
}

/// `NormalizeWarpIndex`: aliases expanded, and every other live `threadIdx.x` replaced by the
/// warp index, for index analysis only.
fn normalize_warp_index(
    index: &Expr,
    active: &Bindings,
    warp_index: &Var,
    aliases: &Aliases,
) -> Result<Expr> {
    let expanded = expand_aliases(index, aliases)?;
    substitute_with_dtype_legalization(expanded, |var| {
        let is_other_tx = active
            .get(&ObjectIdentity::of(var))
            .is_some_and(|binding| binding.tag.as_str() == "threadIdx.x")
            && !var.same_as(warp_index);
        if !is_other_tx {
            return Ok(None);
        }
        let warp_index = as_prim_var(warp_index)?;
        let ty: PrimType = var.ty.clone().try_cast()?;
        Ok(Some(warp_index.cast(ty).expr()?))
    })
}

/// The tag and extent of a `launch_thread` region.
fn launch_binding(op: &RegionStmt) -> Result<(Var, ThreadBinding)> {
    let var = as_prim_var(op.body_params.get(0)?)?;
    let tag = op.args.get(0)?;
    let tag = match tag.as_node::<StringImmObj>() {
        Some(tag) => tag.value.clone(),
        None => bail!(TYPE_ERROR, "expected a StringImm thread tag"),
    };
    Ok((var, ThreadBinding { tag, extent: op.args.get(1)? }))
}

// ---------------------------------------------------------------------------------------------
// WarpIndexFinder

/// Collects each lexical launch definition and checks that every `threadIdx.x` has one width, a
/// factor of the warp size.
struct WarpIndexFinder {
    warp_size: i64,
    width: i64,
    bindings: Bindings,
}

impl WarpIndexFinder {
    fn new(warp_size: i64, bindings: Bindings) -> Result<Self> {
        let mut finder = Self { warp_size, width: 0, bindings };
        let enclosing: Vec<ThreadBinding> = finder.bindings.values().cloned().collect();
        for binding in &enclosing {
            finder.check_width(binding)?;
        }
        Ok(finder)
    }

    fn find(mut self, stmt: &Stmt) -> Result<(Bindings, i64)> {
        structural_walk(stmt, &mut self, WalkOrder::PreOrder)?;
        check_cmp!(
            self.width,
            >,
            0,
            INTERNAL_ERROR,
            "width_ > 0",
            "Cannot find threadIdx.x within the scope of warp memory"
        );
        Ok((self.bindings, self.width))
    }

    fn check_width(&mut self, binding: &ThreadBinding) -> Result<()> {
        if binding.tag.as_str() != "threadIdx.x" {
            return Ok(());
        }
        let extent = binding.extent.as_node::<IntImmObj>().and_then(|imm| imm.value.to_i64());
        let valid = extent.is_some_and(|e| e > 0 && e <= self.warp_size && self.warp_size % e == 0);
        check!(
            valid,
            INTERNAL_ERROR,
            "extent && extent->value > 0 && extent->value <= warp_size_ && warp_size_ % \
             extent->value == 0",
            "Expect threadIdx.x size to be a positive factor of warp size ({}), but got {}",
            self.warp_size,
            repr(&binding.extent)
        );
        let width = extent.expect("checked above");
        check!(
            self.width == 0 || self.width == width,
            INTERNAL_ERROR,
            "width_ == 0 || width_ == width",
            "LowerWarpMemory requires compatible threadIdx.x extents across launch bindings"
        );
        self.width = width;
        Ok(())
    }
}

#[dispatch(walk)]
impl WarpIndexFinder {
    fn walk_region(&mut self, op: RegionStmt) -> Result<WalkResult> {
        if LAUNCH_THREAD.is(&op.op) {
            let (var, binding) = launch_binding(&op)?;
            self.check_width(&binding)?;
            self.bindings.insert(ObjectIdentity::of(&var), binding);
        }
        Ok(WalkResult::Advance)
    }
}

// ---------------------------------------------------------------------------------------------
// WarpStoreCoeffFinder

/// Finds the coefficient `m` of the warp index in the stores to the warp tensor.
struct WarpStoreCoeffFinder<'a> {
    tensor: ObjectIdentity,
    bindings: &'a Bindings,
    active: Bindings,
    warp_index: Option<Var>,
    warp_coeff: i64,
    analyzer: &'a Analyzer,
    aliases: &'a Aliases,
}

impl WarpStoreCoeffFinder<'_> {
    fn find(mut self, stmt: &Stmt) -> Result<i64> {
        structural_walk(stmt, &mut self, WalkOrder::PreOrder)?;
        Ok(self.warp_coeff)
    }

    fn warp_index(&self, what: &str) -> Result<Var> {
        check!(self.warp_index.is_some(), INTERNAL_ERROR, "warp_index_.defined()", "{}", what);
        Ok(self.warp_index.clone().expect("checked above"))
    }

    fn update_coefficient(&mut self, coefficient: i64) -> Result<()> {
        self.warp_index("Warp memory store must be inside a threadIdx.x launch")?;
        check_cmp!(coefficient, >, 0, INTERNAL_ERROR, "coefficient > 0");
        if self.warp_coeff != 0 {
            check_cmp!(
                self.warp_coeff,
                ==,
                coefficient,
                INTERNAL_ERROR,
                "warp_coeff_ == coefficient",
                "LowerWarpMemory requires compatible store coefficients across launch bindings"
            );
        } else {
            self.warp_coeff = coefficient;
        }
        Ok(())
    }

    fn update_pattern(&mut self, index: &Expr) -> Result<()> {
        let warp_index =
            self.warp_index("Warp memory store must be inside a threadIdx.x launch")?;
        let normalized = normalize_warp_index(index, &self.active, &warp_index, self.aliases)?;
        let vars = Array::new(vec![as_prim_var(&warp_index)?]);
        let m = detect_linear_equation(&normalized, &vars)?;
        check_cmp!(
            m.len(),
            ==,
            2,
            INTERNAL_ERROR,
            "m.size() == 2U",
            "LowerWarpMemory failed. Could not simplify the store index `{}` into the form ax + by \
             + cz + ... Warp memory is approximated by storing values in thread local registers \
             and shuffling values between these registers. Currently only linear equation indices \
             are supported.",
            repr(index)
        );
        let coeff = self.analyzer.canonical_simplify(&m.get(0)?)?;
        let value =
            coeff.as_node::<IntImmObj>().and_then(|imm| imm.value.to_i64()).filter(|v| *v > 0);
        check!(
            value.is_some(),
            INTERNAL_ERROR,
            "mcoeff_as_int && mcoeff_as_int->value > 0",
            "LowerWarpMemory failed due to store index={}, require positive constant coefficient \
             on warp index {} but get {}",
            repr(index),
            repr(&warp_index),
            repr(&coeff)
        );
        self.update_coefficient(value.expect("checked above"))
    }
}

#[dispatch(walk)]
impl WarpStoreCoeffFinder<'_> {
    /// The operands are visited before the launch binding comes into scope.
    fn walk_region(&mut self, op: RegionStmt) -> Result<WalkResult> {
        structural_walk(&op.args, &mut *self, WalkOrder::PreOrder)?;
        structural_walk(&op.attrs, &mut *self, WalkOrder::PreOrder)?;
        let (previous_index, previous_bindings) = (self.warp_index.clone(), self.active.clone());
        if LAUNCH_THREAD.is(&op.op) {
            let (var, _) = launch_binding(&op)?;
            let binding = self
                .bindings
                .get(&ObjectIdentity::of(&var))
                .cloned()
                .ok_or_else(|| error!(INTERNAL_ERROR, "unordered_map::at: key not found"))?;
            let is_tx = binding.tag.as_str() == "threadIdx.x";
            self.active.insert(ObjectIdentity::of(&var), binding);
            if is_tx {
                self.warp_index = Some(var);
            }
        }
        let result = structural_walk(&op.body, &mut *self, WalkOrder::PreOrder);
        self.active = previous_bindings;
        self.warp_index = previous_index;
        result?;
        Ok(WalkResult::Skip)
    }

    fn walk_call(&mut self, op: Call) -> Result<WalkResult> {
        if CUDA_MMA_FILL.is(&op.op)
            && tensor_var_of(&op.args.get(1)?).as_ref() == Some(&self.tensor)
        {
            let local_size = op.args.get(0)?;
            let local_size = local_size.as_node::<IntImmObj>();
            check!(
                local_size.is_some(),
                INTERNAL_ERROR,
                "local_size",
                "Integer expected for the first argument of mma_fill"
            );
            let value = local_size.expect("checked above").value.to_i64().unwrap_or(0);
            self.update_coefficient(value)?;
        } else if PTX_LEGACY_LDMATRIX.is(&op.op)
            && tensor_var_of(&op.args.get(3)?).as_ref() == Some(&self.tensor)
        {
            self.update_pattern(&op.args.get(4)?)?;
        }
        Ok(WalkResult::Advance)
    }

    fn walk_store(&mut self, op: TensorStore) -> Result<WalkResult> {
        if as_tensor_var(&op.dest)? != self.tensor {
            return Ok(WalkResult::Advance);
        }
        check_cmp!(
            op.indices.len(),
            ==,
            1,
            INTERNAL_ERROR,
            "op->indices.size() == 1",
            "Expected flat memory to use as warp memory.  Has FlattenTensor been run?"
        );
        let mut index = op.indices.get(0)?;
        let value_lanes = lanes(&op.value)?;
        if value_lanes != 1 {
            let base = PVar::expr();
            check!(
                ramp(&base, 1, value_lanes).matches(&index),
                INTERNAL_ERROR,
                "sym::ramp(base, 1, value_ty.lanes()).Match(index)",
                "LowerWarpMemory failed due to store index={}, can only handle continuous store",
                repr(&index)
            );
            self.update_pattern(&base.get())?;
            index = base.get();
        }
        self.update_pattern(&index)?;
        Ok(WalkResult::Skip)
    }
}

// ---------------------------------------------------------------------------------------------
// WarpAccessRewriter

/// Rewrites one warp allocation and its uses.
struct WarpAccessRewriter<'a> {
    /// The scope of the local tensor.
    local: MemoryScope,
    warp_size: i64,
    analyzer: &'a Analyzer,
    aliases: &'a Aliases,
    tensor: Option<ObjectIdentity>,
    new_tensor: Option<Var>,
    width: i64,
    warp_index: Option<Var>,
    warp_coeff: i64,
    warp_group: i64,
    bindings: Bindings,
    active: Bindings,
}

impl<'a> WarpAccessRewriter<'a> {
    fn new(
        local: MemoryScope,
        warp_size: i64,
        analyzer: &'a Analyzer,
        enclosing: Bindings,
        warp_index: Option<Var>,
        aliases: &'a Aliases,
    ) -> Self {
        Self {
            local,
            warp_size,
            analyzer,
            aliases,
            tensor: None,
            new_tensor: None,
            width: 0,
            warp_index,
            warp_coeff: 0,
            warp_group: 0,
            bindings: enclosing.clone(),
            active: enclosing,
        }
    }

    /// The local allocation replacing `alloc`, and `body` (its remaining siblings) rewritten.
    fn rewrite(mut self, alloc: &Bind, call: &CallObj, body: Stmt) -> Result<Stmt> {
        let alloc_ty = call.ty_args.get(0)?;
        let Some(alloc_ty) = alloc_ty.as_node::<TensorTypeObj>() else {
            bail!(TYPE_ERROR, "expected a TensorType");
        };
        let element_lanes = alloc_ty.dtype.dtype.lanes as i64;
        self.tensor = Some(ObjectIdentity::of(&alloc.var));
        let mut alloc_size: i128 = 1;
        for dim in alloc_ty.shape.iter() {
            alloc_size = match dim.as_node::<IntImmObj>() {
                Some(size) => (alloc_size * size.value.to_i128().unwrap_or(0)) as i64 as i128,
                None => 0,
            };
        }
        let mut alloc_size = alloc_size as i64;
        check_cmp!(
            alloc_size,
            >,
            0,
            INTERNAL_ERROR,
            "alloc_size > 0",
            "warp memory only support constant alloc size"
        );
        alloc_size *= element_lanes;
        let finder = WarpIndexFinder::new(self.warp_size, self.bindings.clone())?;
        let (bindings, width) = finder.find(&body)?;
        self.bindings = bindings;
        self.width = width;
        let tensor = self.tensor.clone().expect("set above");
        self.warp_coeff = WarpStoreCoeffFinder {
            tensor,
            bindings: &self.bindings,
            active: self.active.clone(),
            warp_index: self.warp_index.clone(),
            warp_coeff: 0,
            analyzer: self.analyzer,
            aliases: self.aliases,
        }
        .find(&body)?;

        // Round the local size up to a multiple of `width * warp_coeff`.
        let factor = (self.width * self.warp_coeff) as i32 as i64;
        check_cmp!(factor, !=, 0, INTERNAL_ERROR, "factor != 0", "Divide by zero");
        self.warp_group = (alloc_size + (factor - 1)) / factor;
        alloc_size = self.warp_group * factor;

        let var = &alloc.var;
        let var_ty = var.ty.as_node::<TensorTypeObj>().expect("a tensor variable");
        let elem_offset: crate::ir::Expr = int_imm(var_ty.elem_offset.prim_dtype()?, 0)?.into();
        let local_ty = TensorType::from_complete_fields(
            var_ty.loc.clone(),
            var_ty.dtype.clone(),
            self.local.clone(),
            Array::new(vec![super::util::int32(alloc_size / self.width)?.into()]),
            Array::new(vec![]),
            elem_offset,
            var_ty.data_alignment,
            var_ty.offset_factor,
            var_ty.layout.clone(),
        );
        let new_tensor = Var::new_full(var.name.clone(), local_ty, &var.loc)?;
        self.new_tensor = Some(new_tensor.clone());

        let rewritten_body: Stmt = structural_mutate(body, &mut self)?.try_into()?;
        let mut args: Vec<Expr> = call.args.iter().collect();
        if let Some(first) = args.first().cloned() {
            args[0] = structural_mutate(first, &mut self)?.try_into()?;
        }
        let new_ty = new_tensor.ty.clone();
        let new_call = Call::new_full(
            Some(new_ty.clone()),
            ALLOC_TENSOR.get()?.into(),
            Array::new(args),
            call.attrs.clone(),
            Array::new(vec![new_ty]),
            &call.loc,
        )?;
        let bind = Bind::new_full(new_tensor, new_call, &alloc.loc)?;
        Ok(SeqStmt::new(vec![bind.into(), rewritten_body])?.into())
    }

    fn warp_index(&self) -> Result<Var> {
        self.warp_index.clone().ok_or_else(|| {
            error!(
                INTERNAL_ERROR,
                "Check failed: (warp_index_.defined()) is false: Warp memory access must be inside \
                 a threadIdx.x launch"
            )
        })
    }

    fn new_tensor(&self) -> Var {
        self.new_tensor.clone().expect("set before rewriting")
    }

    fn is_tensor(&self, id: &ObjectIdentity) -> bool {
        self.tensor.as_ref() == Some(id)
    }

    /// `SplitIndexByGroup`: the index into the local tensor, and the thread whose part holds it.
    fn split_index_by_group(&self, original: &Expr) -> Result<(Expr, Expr)> {
        let warp_index = self.warp_index()?;
        let index = normalize_warp_index(original, &self.active, &warp_index, self.aliases)?;
        let index_dtype = index.prim_dtype()?;
        let index_lanes = lanes(&index)?;
        if index_lanes != 1 {
            let base = PVar::expr();
            check!(
                ramp(&base, 1, index_lanes).matches(&index),
                INTERNAL_ERROR,
                "sym::ramp(base, 1, index_ty.lanes()).Match(index)"
            );
            let (local_index, group) = self.split_index_by_group(&base.get())?;
            let one = int_imm(local_index.prim_dtype()?, 1)?;
            let local_index = Ramp::new(local_index, one, index_lanes as i32)?.into();
            return Ok((local_index, group));
        }
        let m: Expr = int_imm(index_dtype, self.warp_coeff as i128)?.into();
        let simplify =
            |e: crate::ops::Ex| -> Result<Expr> { self.analyzer.canonical_simplify(&e.expr()?) };
        if self.warp_group == 1 {
            let x = simplify(index.indexmod(&m))?;
            let z = simplify(index.indexdiv(&m))?;
            return Ok((x, z));
        }
        let span = self.warp_coeff * self.width;
        let x = simplify(index.indexmod(&m))?;
        let y = index.div(crate::ops::make_const(index_dtype, span)?);
        let y = y * &m + &x;
        let z = index.indexmod(int_imm(index_dtype, span as i128)?).indexdiv(&m);
        Ok((simplify(y)?, simplify(z)?))
    }

    /// `RewriteIndicesAt`: each `(tensor, index)` argument pair that names the warp tensor is
    /// pointed at the local tensor; the other arguments are left alone.
    fn rewrite_indices_at(&self, op: &Call, positions: &[usize]) -> Result<Any> {
        let mut args: Vec<Expr> = op.args.iter().collect();
        for &i in positions {
            if tensor_var_of(&op.args.get(i)?).is_some_and(|id| self.is_tensor(&id)) {
                let (local_index, _) = self.split_index_by_group(&op.args.get(i + 1)?)?;
                args[i] = self.new_tensor().tensor_data()?;
                args[i + 1] = local_index;
            }
        }
        let call = Call::new_full(
            Some(op.ty.clone()),
            op.op.clone(),
            Array::new(args),
            op.attrs.clone(),
            Array::new(vec![]),
            &op.loc,
        )?;
        Ok(call.into())
    }
}

#[dispatch(mutate)]
impl WarpAccessRewriter<'_> {
    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        if !LAUNCH_THREAD.is(&op.op) {
            return m.default_mutate(self, &op);
        }
        let extent: Expr = m.mutate_as(self, &op.args.get(1)?)?;
        let (var, _) = launch_binding(&op)?;
        let (previous_index, previous_bindings) = (self.warp_index.clone(), self.active.clone());
        let binding = self
            .bindings
            .get(&ObjectIdentity::of(&var))
            .cloned()
            .ok_or_else(|| error!(INTERNAL_ERROR, "unordered_map::at: key not found"))?;
        let is_tx = binding.tag.as_str() == "threadIdx.x";
        self.active.insert(ObjectIdentity::of(&var), binding);
        if is_tx {
            self.warp_index = Some(var);
        }
        let body = m.mutate(self, &op.body);
        self.active = previous_bindings;
        self.warp_index = previous_index;
        let body: Stmt = body?.try_into()?;
        let args = Array::new(vec![op.args.get(0)?, extent]);
        let region = RegionStmt::new_full(
            op.op.clone(),
            args,
            op.body_params.clone(),
            op.attrs.clone(),
            body,
            op.result_vars.clone(),
            &op.loc,
        )?;
        Ok(region.into())
    }

    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        if CUDA_MMA_STORE.is(&op.op) {
            return self.rewrite_indices_at(&op, &[3]);
        }
        if CUDA_MMA_FILL.is(&op.op) {
            return self.rewrite_indices_at(&op, &[1]);
        }
        // Legacy variants: (pointer, offset) pairs at their apache positions.
        if PTX_LEGACY_MMA.is(&op.op) {
            return self.rewrite_indices_at(&op, &[6, 8, 10]);
        }
        if PTX_LEGACY_LDMATRIX.is(&op.op) {
            return self.rewrite_indices_at(&op, &[3]);
        }
        m.default_mutate(self, &op)
    }

    /// The warp tensor may only be accessed through loads, stores and the rewritten calls.
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        let is_use = m.def_region_kind() == DefRegionKind::None;
        check!(
            !(is_use && self.is_tensor(&ObjectIdentity::of(&var))),
            INTERNAL_ERROR,
            "op != tensor_",
            "Cannot access address of warp memory directly"
        );
        m.default_mutate(self, &var)
    }

    /// A store to the warp tensor stores to the local element.
    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let mut store = match value.same_as(&op.value) && indices.same_as(&op.indices) {
            true => op,
            false => {
                TensorStore::from_complete_fields(op.loc.clone(), op.dest.clone(), indices, value)
            }
        };
        if as_tensor_var(&store.dest).is_ok_and(|id| self.is_tensor(&id)) {
            check_cmp!(
                store.indices.len(),
                ==,
                1,
                INTERNAL_ERROR,
                "store->indices.size() == 1",
                "Expected flat memory to use as warp memory.  Has FlattenTensor been run?"
            );
            let (local_index, _) = self.split_index_by_group(&store.indices.get(0)?)?;
            store = TensorStore::from_complete_fields(
                store.loc.clone(),
                self.new_tensor().into(),
                Array::new(vec![local_index]),
                store.value.clone(),
            );
        } else {
            as_tensor_var(&store.dest)?;
        }
        Ok(store.into())
    }

    /// A load of the warp tensor loads the local element, from another thread by a shuffle.
    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let mut load = op.clone();
        if !indices.same_as(&op.indices) {
            load = TensorLoad::from_complete_fields(
                op.loc.clone(),
                op.ty.clone(),
                op.source.clone(),
                indices,
            );
        }
        if !self.is_tensor(&as_tensor_var(&load.source)?) {
            return Ok(load.into());
        }
        check_cmp!(
            op.indices.len(),
            ==,
            1,
            INTERNAL_ERROR,
            "op->indices.size() == 1",
            "Expected flat memory to use as warp memory.  Has FlattenTensor been run?"
        );
        let (local_index, group) = self.split_index_by_group(&op.indices.get(0)?)?;
        // The local index must not depend on the warp index.
        let warp_index = self.warp_index()?;
        let mut finder = VarFinder { target: ObjectIdentity::of(&warp_index) };
        let uses_warp_index = structural_walk(&local_index, &mut finder, WalkOrder::PreOrder)?;
        check!(
            uses_warp_index.is_none(),
            INTERNAL_ERROR,
            "!ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(local_index, walkfn).has_value()",
            "LowerWarpMemory failed to rewrite load to shuffle for index {} local_index={}",
            repr(&op.indices.get(0).expect("checked")),
            repr(&local_index)
        );
        let local: Expr =
            TensorLoad::new_full(self.new_tensor(), vec![local_index], &load.loc)?.into();
        if self.analyzer.can_prove_equal(&group, &warp_index.into())? {
            return Ok(local.into());
        }
        let uint32 = prim_type(tvm_ffi::DLDataType { code: 1, bits: 32, lanes: 1 })?;
        let mask: Expr = Call::new(uint32, GPU_WARP_ACTIVEMASK.get()?, vec![])?.into();
        let args = vec![
            mask,
            local.clone(),
            group,
            super::util::int32(self.width)?.into(),
            super::util::int32(self.warp_size)?.into(),
        ];
        Ok(Call::new(&local.ty, GPU_WARP_SHUFFLE.get()?, args)?.into())
    }
}

struct VarFinder {
    target: ObjectIdentity,
}

#[dispatch(walk)]
impl VarFinder {
    fn walk_var(&mut self, var: Var) -> WalkResult {
        match ObjectIdentity::of(&var) == self.target {
            true => WalkResult::Interrupt,
            false => WalkResult::Advance,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// BindVarBoundInfo

/// Binds loop and thread ranges in the analyzer, and records pure integer bindings as aliases of
/// their (expanded) values. Only statements are visited.
struct BindVarBoundInfo<'a> {
    analyzer: &'a Analyzer,
    aliases: &'a mut Aliases,
}

#[dispatch(walk)]
impl BindVarBoundInfo<'_> {
    fn walk_bind(&mut self, op: Bind) -> Result<WalkResult> {
        let value = &op.value;
        let is_prim_var = AnyView::from(&op.var).type_index() == VarObj::type_index()
            && op.var.ty.as_node::<PrimTypeObj>().is_some();
        if let Some(ty) = value.ty.as_node::<PrimTypeObj>() {
            let dtype = ty.dtype;
            if is_prim_var && dtype.lanes == 1 && dtype.code <= 1 && side_effect(value)? <= PURE {
                let expanded = expand_aliases(value, self.aliases)?;
                self.aliases.insert(ObjectIdentity::of(&op.var), expanded);
            }
        }
        Ok(WalkResult::Advance)
    }

    fn walk_for(&mut self, op: For) -> Result<WalkResult> {
        self.analyzer.bind_range(&op.loop_var, &Range::from_min_extent(&op.min, &op.extent)?)?;
        Ok(WalkResult::Advance)
    }

    fn walk_region(&mut self, op: RegionStmt) -> Result<WalkResult> {
        if LAUNCH_THREAD.is(&op.op) {
            let var = as_prim_var(op.body_params.get(0)?)?;
            let extent = op.args.get(1)?;
            let zero = int_imm(extent.prim_dtype()?, 0)?;
            self.analyzer.bind_range(&var, &Range::from_min_extent(zero, extent)?)?;
        }
        Ok(WalkResult::Advance)
    }

    fn walk_expr(&mut self, _expr: Expr) -> WalkResult {
        WalkResult::Skip
    }
}

// ---------------------------------------------------------------------------------------------
// WarpMemoryRewriter

/// Finds each warp allocation in a sequence and rewrites it with the statements after it.
struct WarpMemoryRewriter {
    local: MemoryScope,
    warp: MemoryScope,
    warp_size: i64,
    analyzer: Analyzer,
    active: Bindings,
    warp_index: Option<Var>,
    aliases: Aliases,
    /// The rewritten warp variables, kept alive for `UpdatePointerStorageScope`.
    new_storage_scopes: Vec<(Var, MemoryScope)>,
}

impl WarpMemoryRewriter {
    fn rewrite(&mut self, stmt: Stmt) -> Result<Stmt> {
        if self.warp_size == 1 {
            return Ok(stmt);
        }
        let mut binder = BindVarBoundInfo { analyzer: &self.analyzer, aliases: &mut self.aliases };
        structural_walk(&stmt, &mut binder, WalkOrder::PreOrder)?;
        structural_mutate(stmt, self)?.try_into()
    }
}

#[dispatch(mutate)]
impl WarpMemoryRewriter {
    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        if !LAUNCH_THREAD.is(&op.op) {
            return m.default_mutate(self, &op);
        }
        let (var, binding) = launch_binding(&op)?;
        let (previous_bindings, previous_index) = (self.active.clone(), self.warp_index.clone());
        let is_tx = binding.tag.as_str() == "threadIdx.x";
        self.active.insert(ObjectIdentity::of(&var), binding);
        if is_tx {
            self.warp_index = Some(var);
        }
        let body = m.mutate(self, &op.body);
        self.active = previous_bindings;
        self.warp_index = previous_index;
        let body: Stmt = body?.try_into()?;
        let region = RegionStmt::new_full(
            op.op.clone(),
            op.args.clone(),
            op.body_params.clone(),
            op.attrs.clone(),
            body,
            op.result_vars.clone(),
            &op.loc,
        )?;
        Ok(region.into())
    }

    fn mutate_seq(&mut self, op: SeqStmt, m: &mut Mutator) -> Result<Any> {
        let mut new_seq: Vec<Stmt> = Vec::new();
        let mut changed = false;
        let stmts: Vec<Stmt> = op.seq.iter().collect();
        for (i, stmt) in stmts.iter().enumerate() {
            let warp_alloc = stmt.as_node::<crate::ir::BindObj>().and_then(|alloc| {
                let call = alloc.value.as_node::<CallObj>()?;
                let ty = call.ty_args.get(0).ok()?;
                let is_warp = ALLOC_TENSOR.is(&call.op)
                    && ty.as_node::<TensorTypeObj>()?.scope.same_as(&self.warp);
                is_warp.then_some(())
            });
            if warp_alloc.is_some() {
                let alloc: Bind = stmt.clone().try_cast()?;
                let call: Call = alloc.value.clone().try_cast()?;
                self.new_storage_scopes.push((alloc.var.clone(), self.local.clone()));
                let remaining: Vec<Stmt> = stmts[i + 1..].to_vec();
                let body: Stmt = SeqStmt::new(remaining)?.into();
                let rewriter = WarpAccessRewriter::new(
                    self.local.clone(),
                    self.warp_size,
                    &self.analyzer,
                    self.active.clone(),
                    self.warp_index.clone(),
                    &self.aliases,
                );
                let rewritten = rewriter.rewrite(&alloc, &call, body)?;
                // Continue through the remaining allocations in the same scope.
                new_seq.push(m.mutate_as(self, &rewritten)?);
                changed = true;
                break;
            }
            let result: Stmt = m.mutate_as(self, stmt)?;
            changed |= !result.same_as(stmt);
            new_seq.push(result);
        }
        if !changed {
            return Ok(op.into());
        }
        Ok(SeqStmt::new_full(new_seq, &op.loc)?.into())
    }

    /// Expressions are not visited.
    fn mutate_expr(&mut self, _expr: Expr) -> tvm_ffi::Unchanged {
        tvm_ffi::Unchanged
    }
}

// ---------------------------------------------------------------------------------------------
// UpdatePointerStorageScope

/// Retypes the rewritten warp variables to local memory wherever they remain.
struct UpdatePointerStorageScope {
    remap: HashMap<ObjectIdentity, Var>,
    tensor_scopes: HashMap<ObjectIdentity, MemoryScope>,
}

impl UpdatePointerStorageScope {
    fn new(new_storage_scopes: &[(Var, MemoryScope)]) -> Result<Self> {
        let mut remap = HashMap::new();
        for (var, scope) in new_storage_scopes {
            let replacement: Var = if let Some(ty) = var.ty.as_node::<TensorTypeObj>() {
                let ty = TensorType::from_complete_fields(
                    ty.loc.clone(),
                    ty.dtype.clone(),
                    scope.clone(),
                    ty.shape.clone(),
                    ty.strides.clone(),
                    ty.elem_offset.clone(),
                    ty.data_alignment,
                    ty.offset_factor,
                    ty.layout.clone(),
                );
                Var::new_full(var.name.clone(), ty, &var.loc)?
            } else {
                let ptr = var.ty.as_node::<PtrTypeObj>();
                check!(
                    ptr.is_some(),
                    INTERNAL_ERROR,
                    "ptr_type",
                    "The provided variable is not of pointer type"
                );
                let ptr = ptr.expect("checked above");
                let ty = PtrType::new(&ptr.element_type, scope.clone())?;
                Var::new_full(var.name.clone(), ty, &var.loc)?
            };
            remap.insert(ObjectIdentity::of(var), replacement);
        }
        Ok(Self { remap, tensor_scopes: HashMap::new() })
    }
}

/// `ReinferMutatedCallType`: the call, retyped by `Call::ReinferType` when the type changes.
fn reinfer_call_type(call: Expr) -> Result<Expr> {
    let call: Call = call.try_cast()?;
    let ty = call.reinfer_type()?;
    if call.ty.same_as(&ty) {
        return Ok(call.into());
    }
    let rebuilt = Call::from_complete_fields(
        call.loc.clone(),
        ty,
        call.op.clone(),
        call.args.clone(),
        call.attrs.clone(),
        call.ty_args.clone(),
    );
    Ok(rebuilt.into())
}

#[dispatch(mutate)]
impl UpdatePointerStorageScope {
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        match self.remap.get(&ObjectIdentity::of(&var)) {
            Some(replacement) => Ok(replacement.to_any()),
            None => m.default_mutate(self, &var),
        }
    }

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            let is_tensor_call = ALLOC_TENSOR.is(&call.op) || DECL_TENSOR.is(&call.op);
            if let Some(mapped) =
                self.remap.get(&ObjectIdentity::of(&op.var)).filter(|_| is_tensor_call)
            {
                let scope = match mapped.ty.as_node::<TensorTypeObj>() {
                    Some(ty) => ty.scope.clone(),
                    None => bail!(TYPE_ERROR, "expected a TensorVar"),
                };
                let key = ObjectIdentity::of(&op.value);
                self.tensor_scopes.insert(key.clone(), scope);
                let result = m.default_mutate(self, &op);
                self.tensor_scopes.remove(&key);
                return result;
            }
        }
        m.default_mutate(self, &op)
    }

    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        let is_data_ptr = TENSOR_DATA_PTR.is(&op.op);
        let original: Expr = op.into();
        let result: Expr = m.default_mutate_as(self, &original)?;
        if let Some(scope) = self.tensor_scopes.get(&ObjectIdentity::of(&original)).cloned() {
            let call = result.as_node::<CallObj>().expect("a call mutates to a call");
            let ty = call.ty_args.get(0)?;
            let Some(tensor_ty) = ty.as_node::<TensorTypeObj>() else {
                bail!(TYPE_ERROR, "expected a TensorType");
            };
            if !tensor_ty.scope.same_as(&scope) {
                let updated = TensorType::from_complete_fields(
                    tensor_ty.loc.clone(),
                    tensor_ty.dtype.clone(),
                    scope,
                    tensor_ty.shape.clone(),
                    tensor_ty.strides.clone(),
                    tensor_ty.elem_offset.clone(),
                    tensor_ty.data_alignment,
                    tensor_ty.offset_factor,
                    tensor_ty.layout.clone(),
                );
                let mut ty_args: Vec<crate::ir::Type> = call.ty_args.iter().collect();
                ty_args[0] = updated.into();
                let copy = Call::from_complete_fields(
                    call.loc.clone(),
                    call.ty.clone(),
                    call.op.clone(),
                    call.args.clone(),
                    call.attrs.clone(),
                    Array::new(ty_args),
                );
                return Ok(reinfer_call_type(copy.into())?.into());
            }
            return Ok(result.into());
        }
        if !is_data_ptr {
            return Ok(result.into());
        }
        Ok(reinfer_call_type(result)?.into())
    }
}

/// `target->GetAttr<int64_t>("thread_warp_size", 1)`.
fn thread_warp_size(target: &Target) -> Result<i64> {
    match target.attrs()?.get(&String::from("thread_warp_size"))? {
        Some(value) => i64::try_from(value),
        None => Ok(1),
    }
}

pub fn lower_warp_memory() -> Result<Pass> {
    function_pass("tirx.LowerWarpMemory", 0, |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let target = get_attr(&func.attrs, "target")?;
        check!(
            target.is_some(),
            INTERNAL_ERROR,
            "target.has_value()",
            "LowerWarpMemory: Require the target attribute"
        );
        let target: Target = target.expect("checked above").try_into()?;
        let mut rewriter = WarpMemoryRewriter {
            local: MemoryScope::get("local")?,
            warp: MemoryScope::get("warp")?,
            warp_size: thread_warp_size(&target)?,
            analyzer: Analyzer::new()?,
            active: HashMap::new(),
            warp_index: None,
            aliases: HashMap::new(),
            new_storage_scopes: Vec::new(),
        };
        let stmt = rewriter.rewrite(body.into())?;
        let mut updater = UpdatePointerStorageScope::new(&rewriter.new_storage_scopes)?;
        let stmt: Stmt = structural_mutate(stmt, &mut updater)?.try_into()?;
        Ok(Some(func.with_body(Some(SeqStmt::from_stmt(stmt)?))))
    })
}
