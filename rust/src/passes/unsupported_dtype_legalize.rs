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

//! `tirx.BF16ComputeLegalize`, `tirx.BF16StorageLegalize`, `tirx.FP8ComputeLegalize` and
//! `tirx.FP8StorageLegalize` (`src/tirx/transform/unsupported_dtype_legalize.cc`): on a target
//! without native bfloat16 / float8 support, compute in a wider float type, then store the
//! remaining narrow values as unsigned integers of the same width.

use crate::fast_hash::{HashMap, HashSet};

use tvm_ffi::{
    dispatch, structural_mutate, structural_walk, Any, Array, DLDataType, DLDataTypeCode,
    Function as FfiFunction, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result,
    WalkOrder, WalkResult, INTERNAL_ERROR, TYPE_ERROR,
};

use super::dtype_conversion::{dtype_conversion, storage_uint};
use super::util::{MutatorExt, function_pass, get_attr};
use crate::ir::{
    Bind, Call, CallObj, Expr, FloatImm, LambdaExpr, PrimType, PrimTypeObj, PtrType, PtrTypeObj,
    SeqStmt, Stmt, TensorLoad, TensorStore, Tuple, TupleObj, Type, Var,
};
use crate::ops::{float_imm, prim_type, repr, Ex, PrimOps};
use crate::prim::{
    Add, Broadcast, Cast, CastObj, Div, Let, Max, Min, Mul, Select, Shuffle, Sub, EQ, GE, GT, LE,
    LT, NE,
};
use crate::target::Target;
use crate::tirx::op::{
    ALLOC_TENSOR, DECL_TENSOR, GPU_THREAD_ALLREDUCE, MASKED_LOAD, MASKED_STORE, REINTERPRET,
    TENSOR_DATA_PTR,
};
use crate::tirx::{Function, TensorType, TensorTypeObj};
use crate::transform::Pass;

/// The narrow float types a pair of passes legalizes.
#[derive(Clone, Copy)]
enum Family {
    BFloat16,
    Float8,
}

impl Family {
    fn matches_dtype(self, dtype: DLDataType) -> bool {
        match self {
            // `MatchesElementType(kDLBfloat, 16)`.
            Family::BFloat16 => dtype.code == DLDataTypeCode::kDLBfloat as u8 && dtype.bits == 16,
            // kDLFloat8_e3m4 through kDLFloat8_e8m0fnu.
            Family::Float8 => (7..=14).contains(&dtype.code),
        }
    }

    /// `MatchType(type)`: a `PrimType` of this family.
    fn matches(self, ty: &Type) -> bool {
        ty.as_node::<PrimTypeObj>().is_some_and(|ty| self.matches_dtype(ty.dtype))
    }

    /// The `tvm.support.nvcc` function telling whether a compute version supports the family.
    fn support_function(self) -> &'static str {
        match self {
            Family::BFloat16 => "tvm.support.nvcc.supports_bf16",
            Family::Float8 => "tvm.support.nvcc.supports_fp8",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Shared helpers

/// `TensorVar::dtype()`: the element type of a tensor variable.
fn tensor_dtype(tensor: &Expr) -> Result<PrimType> {
    match tensor.ty.as_node::<TensorTypeObj>() {
        Some(ty) => Ok(ty.dtype.clone()),
        None => Err(error!(TYPE_ERROR, "expected a tensor variable")),
    }
}

/// `TensorLoad(tensor, indices).ty()`: what a load of `tensor` at `indices` yields.
fn load_type(tensor: &Expr, indices: &[Expr]) -> Result<PrimType> {
    let load = TensorLoad::new(tensor, indices.to_vec())?;
    load.ty.clone().try_cast()
}

/// `dtype.WithLanes(lanes)` of the dtype `other`.
fn with_lanes(dtype: DLDataType, other: DLDataType) -> DLDataType {
    DLDataType { lanes: other.lanes, ..dtype }
}

/// `var.CopyWithType(ty)`.
fn copy_with_type(var: &Var, ty: impl Into<Type>) -> Var {
    Var::from_complete_fields(var.loc.clone(), ty.into(), var.name.clone())
}

/// `make_object<TensorTypeNode>(*ty)` with `dtype` replaced.
fn with_tensor_dtype(ty: &TensorType, dtype: PrimType) -> TensorType {
    TensorType::from_complete_fields(
        ty.loc.clone(),
        dtype,
        ty.scope.clone(),
        ty.shape.clone(),
        ty.strides.clone(),
        ty.elem_offset.clone(),
        ty.data_alignment,
        ty.offset_factor,
        ty.layout.clone(),
    )
}

/// `call` with type argument 0 (a `TensorType`) rewritten by `rewrite`.
fn with_tensor_ty_arg(call: Call, rewrite: impl FnOnce(&TensorType) -> TensorType) -> Result<Call> {
    let ty: TensorType = call.ty_args.get(0)?.try_cast()?;
    let updated: Type = rewrite(&ty).into();
    let ty_args: Array<Type> =
        std::iter::once(updated).chain(call.ty_args.iter().skip(1)).collect();
    Ok(call.with_ty_args(ty_args))
}

/// C++ `ReinferMutatedCallType`: `call` with the result type its operator infers now.
fn reinfer(call: Call) -> Result<Call> {
    let ty = call.reinfer_type()?;
    if call.ty.same_as(&ty) {
        return Ok(call);
    }
    Ok(Call::from_complete_fields(
        call.loc.clone(),
        ty,
        call.op.clone(),
        call.args.clone(),
        call.attrs.clone(),
        call.ty_args.clone(),
    ))
}

/// `Call(ty, op, args, attrs, ty_args, loc)`.
fn call_like(
    original: &Call,
    ty: impl Into<Type>,
    args: Vec<Expr>,
    ty_args: Array<Type>,
) -> Result<Expr> {
    let call = Call::new_full(
        Some(ty.into()),
        original.op.clone(),
        Array::new(args),
        original.attrs.clone(),
        ty_args,
        &original.loc,
    )?;
    Ok(call.into())
}

fn void_type() -> Result<Type> {
    Ok(prim_type(DLDataType { code: DLDataTypeCode::kDLOpaqueHandle as u8, bits: 0, lanes: 0 })?
        .into())
}

fn same_elements(new: &[Expr], old: &Array<Expr>) -> bool {
    new.len() == old.len() && new.iter().zip(old.iter()).all(|(new, old)| new.same_as(&old))
}

// ---------------------------------------------------------------------------------------------
// ComputeLegalizePlanner

/// C++ `ComputeLegalizePlanner`: the allocations of the family's type whose storage nothing
/// reaches through a raw pointer.
struct Planner {
    family: Family,
    candidates: HashSet<ObjectIdentity>,
    opaque: HashSet<ObjectIdentity>,
}

#[dispatch(walk)]
impl Planner {
    fn walk_bind(&mut self, op: Bind) -> WalkResult {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if ALLOC_TENSOR.is(&call.op) {
                let dtype =
                    call.ty_args.get(0).ok().and_then(|ty| {
                        ty.as_node::<TensorTypeObj>().map(|tensor| tensor.dtype.dtype)
                    });
                if dtype.is_some_and(|dtype| self.family.matches_dtype(dtype)) {
                    self.candidates.insert(ObjectIdentity::of(&op.var));
                }
            }
        }
        WalkResult::Advance
    }

    fn walk_call(&mut self, op: Call) -> WalkResult {
        if TENSOR_DATA_PTR.is(&op.op) && op.args.len() == 1 {
            if let Ok(tensor) = op.args.get(0) {
                if tensor.as_node::<crate::ir::VarObj>().is_some() {
                    self.opaque.insert(ObjectIdentity::of(&tensor));
                }
            }
        }
        WalkResult::Advance
    }

    fn walk_var(&mut self, var: Var) -> WalkResult {
        if var.ty.as_node::<PtrTypeObj>().is_some() {
            self.opaque.insert(ObjectIdentity::of(&var));
        }
        WalkResult::Advance
    }
}

/// `ComputeLegalizePlanner::Plan(func)`: an opaque access anywhere vetoes a candidate.
fn plan(family: Family, body: &SeqStmt) -> Result<HashSet<ObjectIdentity>> {
    let mut planner =
        Planner { family, candidates: HashSet::default(), opaque: HashSet::default() };
    structural_walk(body, &mut planner, WalkOrder::PreOrder)?;
    let Planner { mut candidates, opaque, .. } = planner;
    candidates.retain(|var| !opaque.contains(var));
    Ok(candidates)
}

// ---------------------------------------------------------------------------------------------
// ComputeLegalizer

/// C++ `ComputeLegalizer`: computation on the family's type is done in `promote` instead; the
/// family's storage stays, except for the allocations the planner chose, which are promoted.
struct ComputeLegalizer {
    family: Family,
    /// The scalar type computation is promoted to.
    promote: DLDataType,
    promoted_tensors: HashSet<ObjectIdentity>,
    /// The value of the `Bind` of an allocation to promote, while it is visited.
    allocation_to_promote: Option<Expr>,
    /// `VarRemapSet` of the variables this pass retypes.
    remap: HashMap<ObjectIdentity, Var>,
}

#[dispatch(mutate)]
impl ComputeLegalizer {
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        if let Some(mapped) = self.remap.get(&ObjectIdentity::of(&var)) {
            return Ok(mapped.clone().into());
        }
        m.default_mutate(self, &var)
    }

    /// Every cast to the family's type becomes one to the promoted type.
    fn mutate_cast(&mut self, op: Cast, m: &mut Mutator) -> Result<Any> {
        let value = self.promoted(&op.value, m)?;
        let op_dtype = op.prim_dtype()?;
        if self.family.matches_dtype(op_dtype) {
            return Ok(Ex::from(value).cast(with_lanes(self.promote, op_dtype)).expr()?.into());
        }
        if value.same_as(&op.value) {
            return Ok(op.into());
        }
        Ok(Ex::from(value).cast(op_dtype).expr()?.into())
    }

    fn mutate_select(&mut self, op: Select, m: &mut Mutator) -> Result<Any> {
        let condition: Expr = m.mutate_as(self, &op.condition)?;
        let true_value = self.promoted(&op.true_value, m)?;
        let false_value = self.promoted(&op.false_value, m)?;
        if condition.same_as(&op.condition)
            && true_value.same_as(&op.true_value)
            && false_value.same_as(&op.false_value)
            && !self.family.matches(&op.ty)
        {
            return Ok(op.into());
        }
        Ok(Select::new(condition, true_value, false_value)?.into())
    }

    fn mutate_broadcast(&mut self, op: Broadcast, m: &mut Mutator) -> Result<Any> {
        let value = self.promoted(&op.value, m)?;
        if value.same_as(&op.value) && !self.family.matches(&op.ty) {
            return Ok(op.into());
        }
        Ok(Broadcast::new(value, &op.lanes)?.into())
    }

    fn mutate_shuffle(&mut self, op: Shuffle, m: &mut Mutator) -> Result<Any> {
        let mut vectors = Vec::with_capacity(op.vectors.len());
        for vector in op.vectors.iter() {
            vectors.push(self.promoted(&vector, m)?);
        }
        if same_elements(&vectors, &op.vectors) && !self.family.matches(&op.ty) {
            return Ok(op.into());
        }
        Ok(Shuffle::new(vectors, &op.indices)?.into())
    }

    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        if GPU_THREAD_ALLREDUCE.is(&op.op) {
            return Ok(self.legalize_thread_allreduce(&op, m)?.into());
        }
        if ALLOC_TENSOR.is(&op.op) || DECL_TENSOR.is(&op.op) {
            let mut call: Call = m.default_mutate_as(self, &op)?;
            let promoted =
                self.allocation_to_promote.as_ref().is_some_and(|value| value.same_as(&op));
            if promoted {
                let promote = self.promote;
                call = with_tensor_ty_arg(call, |ty| {
                    let dtype =
                        prim_type(with_lanes(promote, ty.dtype.dtype)).expect("a float type");
                    with_tensor_dtype(ty, dtype)
                })?;
            }
            return Ok(reinfer(call)?.into());
        }
        let is_load = MASKED_LOAD.is(&op.op);
        if is_load || MASKED_STORE.is(&op.op) {
            return Ok(self.legalize_masked(&op, is_load, m)?.into());
        }
        if op.ty.as_node::<PrimTypeObj>().is_none() || REINTERPRET.is(&op.op) {
            return m.default_mutate(self, &op);
        }
        let mut args = Vec::with_capacity(op.args.len());
        for arg in op.args.iter() {
            let is_prim = arg.ty.as_node::<PrimTypeObj>().is_some();
            let value: Expr = m.mutate_as(self, &arg)?;
            args.push(if is_prim { self.promote(value)? } else { value });
        }
        let op_dtype = op.prim_dtype()?;
        if self.family.matches_dtype(op_dtype) {
            let ty: Type = prim_type(with_lanes(self.promote, op_dtype))?.into();
            return Ok(call_like(&op, ty, args, Array::new(vec![]))?.into());
        }
        if same_elements(&args, &op.args) {
            return Ok(op.into());
        }
        Ok(call_like(&op, &op.ty, args, Array::new(vec![]))?.into())
    }

    fn mutate_float_imm(&mut self, op: FloatImm) -> Result<Any> {
        if self.family.matches(&op.ty) {
            return Ok(float_imm(self.promote, op.value)?.into());
        }
        Ok(op.into())
    }

    fn mutate_let(&mut self, op: Let, m: &mut Mutator) -> Result<Any> {
        let value = self.promoted(&op.value, m)?;
        let var = self.retyped(&op.var, &value, &op.value)?;
        let body: Expr = m.mutate_as(self, &op.body)?;
        if value.same_as(&op.value) && var.same_as(&op.var) && body.same_as(&op.body) {
            return Ok(op.into());
        }
        Ok(Let::new(var, value, body)?.into())
    }

    fn mutate_add(&mut self, op: Add, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a + b)
    }

    fn mutate_sub(&mut self, op: Sub, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a - b)
    }

    fn mutate_mul(&mut self, op: Mul, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a * b)
    }

    fn mutate_div(&mut self, op: Div, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a.div(b))
    }

    fn mutate_min(&mut self, op: Min, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a.min(b))
    }

    fn mutate_max(&mut self, op: Max, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a.max(b))
    }

    fn mutate_lt(&mut self, op: LT, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a.less(b))
    }

    fn mutate_le(&mut self, op: LE, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a.less_equal(b))
    }

    fn mutate_gt(&mut self, op: GT, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a.greater(b))
    }

    fn mutate_ge(&mut self, op: GE, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a.greater_equal(b))
    }

    fn mutate_eq(&mut self, op: EQ, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a.equal(b))
    }

    fn mutate_ne(&mut self, op: NE, m: &mut Mutator) -> Result<Any> {
        self.binary(&op, &op.a, &op.b, m, |a, b| a.not_equal(b))
    }

    /// A binding of a non-scalar value promotes the allocation the planner chose; a scalar
    /// binding is computed in the promoted type, and its variable retyped.
    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if op.value.ty.as_node::<PrimTypeObj>().is_none() {
            let chosen = self.promoted_tensors.contains(&ObjectIdentity::of(&op.var))
                && op.value.as_node::<CallObj>().is_some();
            let previous = std::mem::replace(
                &mut self.allocation_to_promote,
                chosen.then(|| op.value.clone()),
            );
            let result = m.default_mutate(self, &op);
            self.allocation_to_promote = previous;
            return result;
        }
        let value = self.promoted(&op.value, m)?;
        let var = self.retyped(&op.var, &value, &op.value)?;
        if value.same_as(&op.value) && var.same_as(&op.var) {
            return Ok(op.into());
        }
        Ok(Bind::new(var, value)?.into())
    }

    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let mut value: Expr = m.mutate_as(self, &op.value)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let tensor = self.remapped_tensor(&op.dest, m)?;
        if value.same_as(&op.value) && indices.same_as(&op.indices) && tensor.same_as(&op.dest) {
            return Ok(op.into());
        }
        let indices_vec: Vec<Expr> = indices.iter().collect();
        value = self.store_value(&tensor, &indices_vec, value)?;
        Ok(TensorStore::new(tensor, &indices, value)?.into())
    }

    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let tensor = self.remapped_tensor(&op.source, m)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        if tensor.same_as(&op.source) && indices.same_as(&op.indices) {
            return Ok(op.into());
        }
        Ok(TensorLoad::new_full(tensor, indices.iter(), &op.loc)?.into())
    }
}

impl ComputeLegalizer {
    /// `LegalizeWithPlanner(func)`.
    fn legalize(family: Family, promote: DLDataType, func: Function) -> Result<Function> {
        let Some(body) = func.body.clone() else { return Ok(func) };
        let mut legalizer = ComputeLegalizer {
            family,
            promote,
            promoted_tensors: plan(family, &body)?,
            allocation_to_promote: None,
            remap: HashMap::default(),
        };
        let body: Stmt = structural_mutate(body, &mut legalizer)?.try_into()?;
        Ok(func.with_body(Some(SeqStmt::from_stmt(body)?)))
    }

    /// `PromoteToTarget(Mutate(value))`.
    fn promoted(&mut self, value: &Expr, m: &mut Mutator) -> Result<Expr> {
        let value: Expr = m.mutate_as(self, value)?;
        self.promote(value)
    }

    /// C++ `PromoteToTarget`: a value of the family's type in the promoted type.
    fn promote(&self, value: Expr) -> Result<Expr> {
        let dtype = value.prim_dtype()?;
        if !self.family.matches_dtype(dtype) {
            return Ok(value);
        }
        let target = with_lanes(self.promote, dtype);
        if let Some(cast) = value.as_node::<CastObj>() {
            if cast.value.prim_dtype()? == target {
                return Ok(cast.value.clone());
            }
        }
        dtype_conversion(value, &prim_type(target)?)
    }

    /// C++ `CastTargetToDType`: a promoted float value back in `dtype`.
    fn cast_target_to_dtype(&self, value: Expr, dtype: &PrimType) -> Result<Expr> {
        let value_dtype = value.prim_dtype()?;
        if value_dtype.code != DLDataTypeCode::kDLFloat as u8 {
            return Ok(value);
        }
        tvm_ffi::ensure!(
            value_dtype == with_lanes(self.promote, value_dtype),
            INTERNAL_ERROR,
            "Check failed: value.ty() == this->promote_dtype_.WithLanes(value_ty.lanes())"
        );
        dtype_conversion(value, dtype)
    }

    /// The variable of a scalar binding: retyped, and remapped, when its value's type changed.
    fn retyped(&mut self, var: &Var, value: &Expr, original: &Expr) -> Result<Var> {
        if value.prim_dtype()? == original.prim_dtype()? {
            return Ok(var.clone());
        }
        let retyped = copy_with_type(var, &value.ty);
        self.remap.insert(ObjectIdentity::of(var), retyped.clone());
        Ok(retyped)
    }

    /// `GetRemappedTensor(buf)`: the variable the engine remapped `tensor` to, if any.
    fn remapped_tensor(&mut self, tensor: &Expr, m: &mut Mutator) -> Result<Expr> {
        m.mutate_as(self, tensor)
    }

    /// The value a store to `tensor` at `indices` writes: a promoted value cast back to the
    /// family's type, and a family value converted for a promoted tensor.
    fn store_value(&self, tensor: &Expr, indices: &[Expr], mut value: Expr) -> Result<Expr> {
        if self.family.matches_dtype(tensor_dtype(tensor)?.dtype) {
            value = self.cast_target_to_dtype(value, &load_type(tensor, indices)?)?;
        }
        let storage = load_type(tensor, indices)?;
        if value.prim_dtype()? != storage.dtype {
            check!(self.family.matches(&value.ty), INTERNAL_ERROR, "MatchType(value.ty())");
            value = dtype_conversion(value, &storage)?;
        }
        Ok(value)
    }

    fn binary(
        &mut self,
        original: impl Into<Expr>,
        a: &Expr,
        b: &Expr,
        m: &mut Mutator,
        combine: fn(Ex, Expr) -> Ex,
    ) -> Result<Any> {
        let original: Expr = original.into();
        let new_a = self.promoted(a, m)?;
        let new_b = self.promoted(b, m)?;
        if new_a.same_as(a) && new_b.same_as(b) && !self.family.matches(&original.ty) {
            return Ok(original.into());
        }
        Ok(combine(Ex::from(new_a), new_b).expr()?.into())
    }

    /// `masked_load(tensor, indices..., predicate)` / `masked_store(tensor, value, indices...,
    /// predicate)` on the remapped tensor, the stored value converted like a `TensorStore`'s.
    fn legalize_masked(&mut self, op: &Call, is_load: bool, m: &mut Mutator) -> Result<Expr> {
        let tensor = self.remapped_tensor(&op.args.get(0)?, m)?;
        let count = op.args.len();
        let value: Option<Expr> =
            if is_load { None } else { Some(m.mutate_as(self, &op.args.get(1)?)?) };
        let mut indices = Vec::new();
        for i in (if is_load { 1 } else { 2 })..count - 1 {
            indices.push(m.mutate_as(self, &op.args.get(i)?)?);
        }
        let predicate: Expr = m.mutate_as(self, &op.args.get(count - 1)?)?;
        let mut args = vec![tensor.clone()];
        if is_load {
            args.extend(indices.iter().cloned());
            args.push(predicate);
            let ty: Type = load_type(&tensor, &indices)?.into();
            return call_like(op, ty, args, op.ty_args.clone());
        }
        let value = self.store_value(&tensor, &indices, value.expect("a store has a value"))?;
        args.push(value);
        args.extend(indices);
        args.push(predicate);
        call_like(op, void_type()?, args, op.ty_args.clone())
    }

    /// C++ `LegalizeThreadAllreduce`: the reduction computed in the promoted type, its operands
    /// and combiner retyped, tuples kept as tuples.
    fn legalize_thread_allreduce(&mut self, op: &Call, m: &mut Mutator) -> Result<Expr> {
        let combine: LambdaExpr =
            op.args.get(0)?.try_cast().map_err(|_| error!(TYPE_ERROR, "expected a LambdaExpr"))?;
        let identity = self.map_operand(&op.args.get(1)?, &mut |this, value| {
            let value: Expr = m.mutate_as(this, value)?;
            this.promote(value)
        })?;
        let values = self.map_operand(&op.args.get(2)?, &mut |this, value| {
            let value: Expr = m.mutate_as(this, value)?;
            this.promote(value)
        })?;
        let value_fields: Vec<Expr> = match values.as_node::<TupleObj>() {
            Some(tuple) => tuple.fields.iter().collect(),
            None => vec![values.clone()],
        };
        let mut vars = Vec::new();
        for (i, var) in combine.vars.iter().enumerate() {
            let field_ty: PrimType = value_fields[i % value_fields.len()].ty.clone().try_cast()?;
            vars.push(copy_with_type(&var, field_ty));
        }
        let arguments: Array<Expr> = vars.iter().map(Expr::from).collect();
        let applied = combine.apply(arguments)?;
        let body = self.map_operand(&applied, &mut |this, value| {
            let value: Expr = m.mutate_as(this, value)?;
            this.promote(value)
        })?;
        let legalized_combine = LambdaExpr::new(vars, body)?.into();
        let predicate: Expr = m.mutate_as(self, &op.args.get(3)?)?;
        let destinations =
            self.map_operand(&op.args.get(4)?, &mut |this, value| m.mutate_as(this, value))?;
        let axes =
            self.map_operand(&op.args.get(5)?, &mut |this, value| m.mutate_as(this, value))?;
        let args = vec![legalized_combine, identity, values, predicate, destinations, axes];
        call_like(op, void_type()?, args, op.ty_args.clone())
    }

    /// `transform` applied to each field of a tuple operand, or to the operand itself.
    fn map_operand(
        &mut self,
        operand: &Expr,
        transform: &mut dyn FnMut(&mut Self, &Expr) -> Result<Expr>,
    ) -> Result<Expr> {
        let Some(tuple) = operand.as_node::<TupleObj>() else { return transform(self, operand) };
        let mut fields = Vec::with_capacity(tuple.fields.len());
        for field in tuple.fields.iter() {
            fields.push(transform(self, &field)?);
        }
        Ok(Tuple::new_full(fields, &operand.loc)?.into())
    }
}

// ---------------------------------------------------------------------------------------------
// StorageLegalizer

/// C++ `StorageLegalizer`: the family's remaining storage becomes unsigned integers of the same
/// width, and reinterprets between them are rewritten or dropped.
struct StorageLegalizer {
    family: Family,
    /// `VarRemapSet` of the pointer variables this pass retypes.
    remap: HashMap<ObjectIdentity, Var>,
}

#[dispatch(mutate)]
impl StorageLegalizer {
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        if let Some(mapped) = self.remap.get(&ObjectIdentity::of(&var)) {
            return Ok(mapped.clone().into());
        }
        m.default_mutate(self, &var)
    }

    fn mutate_let(&mut self, op: Let, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let var = self.remap_var_def(&op.var)?;
        let body: Expr = m.mutate_as(self, &op.body)?;
        if value.same_as(&op.value) && var.same_as(&op.var) && body.same_as(&op.body) {
            return Ok(op.into());
        }
        Ok(Let::new(var, value, body)?.into())
    }

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if op.value.ty.as_node::<TensorTypeObj>().is_some() {
            return m.default_mutate(self, &op);
        }
        let value: Expr = m.mutate_as(self, &op.value)?;
        let var = self.remap_var_def(&op.var)?;
        if value.same_as(&op.value) && var.same_as(&op.var) {
            return Ok(op.into());
        }
        Ok(Bind::new(var, value)?.into())
    }

    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let mutated: Expr = m.mutate_as(self, &op.value)?;
        let value = self.change_to_uint(mutated)?;
        let tensor = self.remapped_tensor(&op.dest, m)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        if tensor.same_as(&op.dest) && indices.same_as(&op.indices) && value.same_as(&op.value) {
            return Ok(op.into());
        }
        if self.family.matches(&op.value.ty) {
            self.check_uint_tensor(&tensor)?;
        }
        Ok(TensorStore::new(tensor, &indices, value)?.into())
    }

    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let tensor = self.remapped_tensor(&op.source, m)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        if tensor.same_as(&op.source) && indices.same_as(&op.indices) {
            return Ok(op.into());
        }
        Ok(TensorLoad::new_full(tensor, indices.iter(), &op.loc)?.into())
    }

    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        if ALLOC_TENSOR.is(&op.op) || DECL_TENSOR.is(&op.op) {
            let mut call: Call = m.default_mutate_as(self, &op)?;
            let ty: TensorType = call.ty_args.get(0)?.try_cast()?;
            if self.family.matches_dtype(ty.dtype.dtype) {
                call = with_tensor_ty_arg(call, |ty| {
                    let dtype = prim_type(storage_uint(ty.dtype.dtype)).expect("an integer type");
                    with_tensor_dtype(ty, dtype)
                })?;
            }
            return Ok(reinfer(call)?.into());
        }
        let is_load = MASKED_LOAD.is(&op.op);
        if is_load || MASKED_STORE.is(&op.op) {
            return Ok(self.legalize_masked(&op, is_load, m)?.into());
        }
        if let Some(pointer) = op.ty.as_node::<PtrTypeObj>() {
            let result: Expr = m.default_mutate_as(self, &Expr::from(&op))?;
            let element = pointer.element_type.as_node::<PrimTypeObj>().map(|ty| ty.dtype);
            let Some(element) = element.filter(|dtype| self.family.matches_dtype(*dtype)) else {
                return Ok(result.into());
            };
            let call: Call = result.try_cast()?;
            let ty = PtrType::new(prim_type(storage_uint(element))?, pointer.scope.clone())?;
            let rebuilt = Call::new_full(
                Some(ty.into()),
                call.op.clone(),
                call.args.clone(),
                call.attrs.clone(),
                call.ty_args.clone(),
                &call.loc,
            )?;
            return Ok(rebuilt.into());
        }
        if op.ty.as_node::<PrimTypeObj>().is_none() || !REINTERPRET.is(&op.op) {
            return m.default_mutate(self, &op);
        }
        // A reinterpret between the family and its storage type is dropped or retargeted.
        let source = op.args.get(0)?;
        if source.ty.as_node::<PrimTypeObj>().is_none() {
            return m.default_mutate(self, &op);
        }
        let value: Expr = m.mutate_as(self, &source)?;
        let op_ty = op.prim_ty()?;
        if value.prim_dtype()? == op_ty.dtype {
            return Ok(value.into());
        }
        if self.family.matches_dtype(op_ty.dtype) {
            let storage = prim_type(storage_uint(op_ty.dtype))?;
            return Ok(Ex::from(value).reinterpret(storage).expr()?.into());
        }
        if source.same_as(&value) {
            return Ok(op.into());
        }
        Ok(Ex::from(value).reinterpret(op_ty).expr()?.into())
    }
}

impl StorageLegalizer {
    /// `StorageLegalizer::Legalize(func)`: parameters and body, after MakePackedAPI.
    fn legalize(family: Family, func: Function) -> Result<Function> {
        check!(
            !func.params.iter().any(|param| param.ty.as_node::<TensorTypeObj>().is_some()),
            INTERNAL_ERROR,
            "!param->ty.as<TensorTypeNode>()",
            "This pass must be called after MakePackedAPI"
        );
        let mut legalizer = StorageLegalizer { family, remap: HashMap::default() };
        let mut params = Vec::with_capacity(func.params.len());
        for param in func.params.iter() {
            params.push(legalizer.remap_var_def(&param)?);
        }
        let body = match func.body.clone() {
            Some(body) => {
                let body: Stmt = structural_mutate(body, &mut legalizer)?.try_into()?;
                Some(SeqStmt::from_stmt(body)?)
            }
            None => None,
        };
        // C++ `CopyOnWrite()`: the function type is kept, not derived again.
        Ok(Function::from_complete_fields(
            func.loc.clone(),
            func.ty.clone(),
            func.attrs.clone(),
            Array::new(params),
            func.ret_type.clone(),
            body,
        ))
    }

    /// C++ `ChangeToUInt`: a reinterpret to the family's type becomes one to its storage type.
    fn change_to_uint(&self, value: Expr) -> Result<Expr> {
        let dtype = value.prim_dtype()?;
        if !self.family.matches_dtype(dtype) {
            return Ok(value);
        }
        match value.as_node::<CallObj>() {
            Some(call) if REINTERPRET.is(&call.op) => {
                let storage = prim_type(storage_uint(dtype))?;
                Ex::from(call.args.get(0)?).reinterpret(storage).expr()
            }
            _ => Ok(value),
        }
    }

    /// C++ `RemapVarDef`: a pointer to the family's type becomes a pointer to its storage type.
    fn remap_var_def(&mut self, var: &Var) -> Result<Var> {
        let Some(pointer) = var.ty.as_node::<PtrTypeObj>() else { return Ok(var.clone()) };
        let Some(element) = pointer.element_type.as_node::<PrimTypeObj>() else {
            return Ok(var.clone());
        };
        if !self.family.matches_dtype(element.dtype) {
            return Ok(var.clone());
        }
        let ty = PtrType::new(prim_type(storage_uint(element.dtype))?, pointer.scope.clone())?;
        let remapped = Var::new(var.name.clone(), ty)?;
        self.remap.insert(ObjectIdentity::of(var), remapped.clone());
        Ok(remapped)
    }

    /// `GetRemappedTensor(buf)`: a tensor of the family's type must have been remapped.
    fn remapped_tensor(&mut self, tensor: &Expr, m: &mut Mutator) -> Result<Expr> {
        let mapped: Expr = m.mutate_as(self, tensor)?;
        if mapped.same_as(tensor) && self.family.matches_dtype(tensor_dtype(tensor)?.dtype) {
            tvm_ffi::bail!(INTERNAL_ERROR, "Cannot find var remap for {}", repr(tensor));
        }
        Ok(mapped)
    }

    fn check_uint_tensor(&self, tensor: &Expr) -> Result<()> {
        check!(
            tensor_dtype(tensor)?.dtype.code == DLDataTypeCode::kDLUInt as u8,
            INTERNAL_ERROR,
            "new_buf.dtype().MatchesCode(DLDataTypeCode::kDLUInt)"
        );
        Ok(())
    }

    fn legalize_masked(&mut self, op: &Call, is_load: bool, m: &mut Mutator) -> Result<Expr> {
        let tensor = self.remapped_tensor(&op.args.get(0)?, m)?;
        let mut args = vec![tensor.clone()];
        if !is_load {
            let original = op.args.get(1)?;
            let mutated: Expr = m.mutate_as(self, &original)?;
            let value = self.change_to_uint(mutated)?;
            if self.family.matches(&original.ty) {
                self.check_uint_tensor(&tensor)?;
            }
            args.push(value);
        }
        let count = op.args.len();
        let mut indices = Vec::new();
        for i in (if is_load { 1 } else { 2 })..count - 1 {
            let index: Expr = m.mutate_as(self, &op.args.get(i)?)?;
            indices.push(index.clone());
            args.push(index);
        }
        args.push(m.mutate_as(self, &op.args.get(count - 1)?)?);
        let ty: Type = match is_load {
            true => load_type(&tensor, &indices)?.into(),
            false => void_type()?,
        };
        call_like(op, ty, args, op.ty_args.clone())
    }
}

// ---------------------------------------------------------------------------------------------
// Passes

/// C++ `CheckDataTypeSupport`: a CUDA target whose compute version supports the family.
fn has_native_support(target: &Target, family: Family) -> Result<bool> {
    if target.kind()?.name()?.as_str() != "cuda" {
        return Ok(false);
    }
    let Some(version) = target.cuda_compute_version()? else { return Ok(false) };
    let Ok(supports) = FfiFunction::get_global(family.support_function()) else {
        return Ok(false);
    };
    supports.call_tuple((version,))?.try_into()
}

/// Whether `func` runs on a target that supports the family natively, so is left as it is.
fn natively_supported(func: &Function, family: Family) -> Result<bool> {
    match get_attr(&func.attrs, "target")? {
        Some(target) => has_native_support(&Target::try_from(target)?, family),
        None => Ok(false),
    }
}

fn compute_legalize(name: &str, family: Family, promote: DLDataType) -> Result<Pass> {
    function_pass(name, 0, move |func, _module, _context| {
        if natively_supported(&func, family)? {
            return Ok(Some(func));
        }
        ComputeLegalizer::legalize(family, promote, func).map(Some)
    })
}

fn storage_legalize(name: &str, family: Family) -> Result<Pass> {
    function_pass(name, 0, move |func, _module, _context| {
        if natively_supported(&func, family)? {
            return Ok(Some(func));
        }
        StorageLegalizer::legalize(family, func).map(Some)
    })
}

pub fn bf16_compute_legalize() -> Result<Pass> {
    let float32 = DLDataType { code: DLDataTypeCode::kDLFloat as u8, bits: 32, lanes: 1 };
    compute_legalize("tirx.BF16ComputeLegalize", Family::BFloat16, float32)
}

pub fn bf16_storage_legalize() -> Result<Pass> {
    storage_legalize("tirx.BF16StorageLegalize", Family::BFloat16)
}

pub fn fp8_compute_legalize(promote_dtype: &str) -> Result<Pass> {
    let promote = <DLDataType as tvm_ffi::DLDataTypeExt>::try_from_str(promote_dtype)?;
    compute_legalize("tirx.FP8ComputeLegalize", Family::Float8, promote)
}

pub fn fp8_storage_legalize() -> Result<Pass> {
    storage_legalize("tirx.FP8StorageLegalize", Family::Float8)
}
