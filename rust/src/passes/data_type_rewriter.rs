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

//! C++ `DataTypeLegalizer` and `IndexDataTypeRewriter` (`src/tirx/ir/data_type_rewriter.cc`) as
//! a mutation policy.
//!
//! A pass keeps a [`DataTypeRewriterBase`], implements [`DataTypeRewriterState`], and declares
//! `#[dispatch(mutate, policy = DataTypeRewrite)]`. Its handlers are the C++ overrides of the
//! derived class; `m.default_mutate(self, &op)` is the C++ `Parent::Mutate_(op)`, and every value
//! no handler takes descends through the policy.
//!
//! - The legalizer keeps types consistent after operands changed type: a loop's bounds take the
//!   type of its variable, the operands of a binary operation are matched by its C++ operator, a
//!   `Bind` or `Let` variable takes the type of its value, and so on.
//! - The index rewriter (`index`) also tracks whether the current expression is a tensor index
//!   (`is_enabled`) or a condition (`is_condition`), for the derived classes to act on.
//!
//! `VarRemapSet` entries are kept in [`DataTypeRewriterBase::remap`]; the engine keeps the rest,
//! the variables whose type it rewrote at their definition.

use crate::fast_hash::HashMap;

use tvm_ffi::{
    bail, match_any, Any, AnyView, Array, MutContextPolicy, MutateContext, MutateValue,
    ObjectIdentity, ObjectRefCore, Result, StructuralView, UnchangedOr, INTERNAL_ERROR, TYPE_ERROR,
};

use super::util::ContextExt;
use crate::ir::{
    Bind, Call, Expr, For, If, PrimType, PrimTypeObj, RegionStmt, SeqStmt, TensorLoad, TensorStore,
    Type, Var,
};
use crate::ops::{make_const, Ex, PrimOps};
use crate::prim::op::{CLZ, IF_THEN_ELSE, POW};
use crate::prim::{
    Add, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, Broadcast, Div, FloorDiv, FloorMod, LShift,
    Let, Max, Min, Mod, Mul, RShift, Ramp, Select, Shuffle, Sub, EQ, GE, GT, LE, LT, NE,
};
use crate::tirx::op::LAUNCH_THREAD;
use crate::tirx::{TensorType, TensorTypeObj};
use tvm_ffi::{DLDataType, DLDataTypeExt};

const DL_INT: u8 = 0;

/// The members C++ `DataTypeLegalizer` and `IndexDataTypeRewriter` add.
pub struct DataTypeRewriterBase {
    /// `IndexDataTypeRewriter`; `false` for `DataTypeLegalizer` alone.
    pub index: bool,
    /// The current expression is a tensor index (`is_enabled_`).
    pub is_enabled: bool,
    /// The current expression is a condition (`is_condition_`).
    pub is_condition: bool,
    /// `ShouldClampShiftAmounts()`.
    pub clamp_shift_amounts: bool,
    /// The `VarRemapSet` substitutions, by the replaced variable.
    pub remap: HashMap<ObjectIdentity, Var>,
}

impl DataTypeRewriterBase {
    /// The members of `IndexDataTypeRewriter` (`index`) or `DataTypeLegalizer`.
    pub fn new(index: bool) -> Self {
        Self {
            index,
            is_enabled: false,
            is_condition: false,
            clamp_shift_amounts: false,
            remap: HashMap::default(),
        }
    }

    /// `VarRemapSet(var, replacement)`.
    pub fn remap_set(&mut self, var: &Var, replacement: Var) {
        self.remap.insert(ObjectIdentity::of(var), replacement);
    }

    /// `VarRemapGet(var)`, for the substitutions set through [`Self::remap_set`].
    pub fn remap_get(&self, var: &Var) -> Option<&Var> {
        self.remap.get(&ObjectIdentity::of(var))
    }
}

/// A mutator that descends through [`DataTypeRewrite`].
pub trait DataTypeRewriterState {
    fn base(&mut self) -> &mut DataTypeRewriterBase;
}

/// The default descent of `IndexDataTypeRewriter` (or `DataTypeLegalizer`).
pub struct DataTypeRewrite;

impl<S: DataTypeRewriterState> MutContextPolicy<S> for DataTypeRewrite {
    fn default_mutate<D: tvm_ffi::MutateDriver<S> + ?Sized>(
        &self,
        value: MutateValue<'_>,
        ctx: &mut MutateContext<'_, S, D>,
    ) -> Result<UnchangedOr<Any>> {
        match rewrite(ctx, value.as_value())? {
            Some(result) => Ok(result),
            None => ctx.default_maybe_inplace_mutate_result(value),
        }
    }
}

type Ctx<'a, 'b, S, D> = &'a mut MutateContext<'b, S, D>;
type Rewritten = Result<Option<UnchangedOr<Any>>>;

fn changed(value: impl Into<Any>) -> Rewritten {
    Ok(Some(UnchangedOr::changed(value.into())))
}

fn unchanged() -> Rewritten {
    Ok(Some(UnchangedOr::unchanged()))
}

/// The nodes the base classes rewrite; `None` for every other value.
fn rewrite<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    view: &StructuralView,
) -> Rewritten {
    let index = ctx.state_mut().base().index;
    // `Var` is not a final type and the index-only arms carry guards; both would keep the
    // `match_any!` below off its exact-leaf lookup table, so they are tested on their own first.
    if let Some(var) = view.cast::<Var>() {
        return match ctx.state_mut().base().remap_get(&var) {
            Some(replacement) => changed(replacement.clone()),
            None => Ok(None),
        };
    }
    let value = AnyView::from(view);
    if index {
        let indexed = match_any! {
            value {
                TensorType(ty) => {
                    let base = ctx.state_mut().base();
                    let enabled = std::mem::replace(&mut base.is_enabled, true);
                    let result = ctx.default_mutate(&ty);
                    ctx.state_mut().base().is_enabled = enabled;
                    Some(changed(result?))
                },
                TensorStore(op) => Some(index_store(ctx, op)),
                TensorLoad(op) => Some(index_load(ctx, op)),
                If(op) => Some(index_if(ctx, op)),
                _ => None,
            }
        };
        if let Some(result) = indexed {
            return result;
        }
    }
    match_any! {
        value {
            RegionStmt(op) => {
                let base = ctx.state_mut().base();
                let enabled = base.is_enabled;
                if index && LAUNCH_THREAD.is(&op.op) {
                    base.is_enabled = true;
                }
                let result = legalize_region(ctx, op);
                ctx.state_mut().base().is_enabled = enabled;
                result
            },
            For(op) => if index { index_for(ctx, op) } else { legalize_for(ctx, op) },
            Bind(op) => if index { index_bind(ctx, op) } else { legalize_bind(ctx, op) },
            Call(op) => {
                if index && IF_THEN_ELSE.is(&op.op) {
                    index_if_then_else(ctx, op)
                } else {
                    legalize_call(ctx, op)
                }
            },
            Select(op) => select(ctx, op, index),
            EQ(op) => comparison(ctx, &op.a, &op.b, index, |a, b| a.equal(b)),
            NE(op) => comparison(ctx, &op.a, &op.b, index, |a, b| a.not_equal(b)),
            LT(op) => comparison(ctx, &op.a, &op.b, index, |a, b| a.less(b)),
            LE(op) => comparison(ctx, &op.a, &op.b, index, |a, b| a.less_equal(b)),
            GT(op) => comparison(ctx, &op.a, &op.b, index, |a, b| a.greater(b)),
            GE(op) => comparison(ctx, &op.a, &op.b, index, |a, b| a.greater_equal(b)),
            Let(op) => legalize_let(ctx, op),
            Ramp(op) => legalize_ramp(ctx, op),
            Broadcast(op) => legalize_broadcast(ctx, op),
            Shuffle(op) => legalize_shuffle(ctx, op),
            LShift(op) => shift(ctx, &op.a, &op.b, |a, b| a << b),
            RShift(op) => shift(ctx, &op.a, &op.b, |a, b| a >> b),
            BitwiseNot(op) => legalize_bitwise_not(ctx, op),
            Add(op) => binary(ctx, &op.a, &op.b, |a, b| a + b),
            Sub(op) => binary(ctx, &op.a, &op.b, |a, b| a - b),
            Mul(op) => binary(ctx, &op.a, &op.b, |a, b| a * b),
            Div(op) => binary(ctx, &op.a, &op.b, |a, b| a.div(b)),
            Mod(op) => binary(ctx, &op.a, &op.b, |a, b| a.truncmod(b)),
            FloorDiv(op) => binary(ctx, &op.a, &op.b, |a, b| a.floordiv(b)),
            FloorMod(op) => binary(ctx, &op.a, &op.b, |a, b| a.floormod(b)),
            Min(op) => binary(ctx, &op.a, &op.b, |a, b| a.min(b)),
            Max(op) => binary(ctx, &op.a, &op.b, |a, b| a.max(b)),
            BitwiseAnd(op) => binary(ctx, &op.a, &op.b, |a, b| a & b),
            BitwiseOr(op) => binary(ctx, &op.a, &op.b, |a, b| a | b),
            BitwiseXor(op) => binary(ctx, &op.a, &op.b, |a, b| a ^ b),
            _ => Ok(None),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Types

/// `PrimType::WithBits(bits)`.
pub fn with_bits(ty: &PrimType, bits: u8) -> Result<PrimType> {
    crate::ops::prim_type(DLDataType { bits, ..ty.dtype })
}

/// `var.CopyWithType(ty)`: the same variable, name and location with another type.
pub fn copy_with_type(var: &Var, ty: impl Into<Type>) -> Var {
    Var::from_complete_fields(var.loc.clone(), ty.into(), var.name.clone())
}

/// `tensor.dtype()` of a tensor variable.
pub fn tensor_dtype(tensor: &Expr) -> Result<PrimType> {
    match tensor.ty.as_node::<TensorTypeObj>() {
        Some(ty) => Ok(ty.dtype.clone()),
        None => bail!(TYPE_ERROR, "expected a tensor variable"),
    }
}

fn cast(value: Expr, ty: &PrimType) -> Result<Expr> {
    value.cast(ty).expr()
}

// ---------------------------------------------------------------------------------------------
// DataTypeLegalizer

/// A launched thread's extent takes the type of its thread variable.
fn legalize_region<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: RegionStmt,
) -> Rewritten {
    let result = ctx.default_mutate(&op)?;
    let Ok(region) = RegionStmt::try_from(result.clone()) else { return changed(result) };
    if !LAUNCH_THREAD.is(&region.op) {
        return changed(result);
    }
    let var: Expr = region.body_params.get(0)?.into();
    let extent = region.args.get(1)?;
    let var_ty = var.prim_ty()?;
    if var_ty.dtype == extent.prim_dtype()? {
        return changed(result);
    }
    let mut args: Vec<Expr> = region.args.iter().collect();
    args[1] = cast(extent, &var_ty)?;
    changed(region.with_args(Array::new(args)))
}

/// A loop's bounds and step take the type of its variable.
fn legalize_for<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: For,
) -> Rewritten {
    let result: For = ctx.default_mutate(&op)?.try_into()?;
    let var: Expr = ctx.mutate_as(&result.loop_var)?.into();
    let var_ty = var.prim_ty()?;
    let min = cast(result.min.clone(), &var_ty)?;
    let extent = cast(result.extent.clone(), &var_ty)?;
    let step = result.step.clone().map(|step| cast(step, &var_ty)).transpose()?;
    changed(result.with(|node| {
        node.min = min;
        node.extent = extent;
        node.step = step;
    }))
}

/// A bound variable takes the type of its value.
fn legalize_bind<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: Bind,
) -> Rewritten {
    if op.value.ty.as_node::<PrimTypeObj>().is_none() {
        return changed(ctx.default_mutate(&op)?);
    }
    let value: Expr = ctx.mutate_as(&op.value)?;
    let mut var = op.var.clone();
    if let Ok(value_ty) = value.prim_ty() {
        if value_ty.dtype != op.var.prim_dtype()? {
            var = copy_with_type(&op.var, value_ty);
            ctx.state_mut().base().remap_set(&op.var, var.clone());
        }
    }
    if value.same_as(&op.value) && var.same_as(&op.var) {
        return unchanged();
    }
    changed(Bind::new_full(var, value, &op.loc)?)
}

/// A let variable takes the type of its value, or the value the type of a replaced variable.
fn legalize_let<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: Let,
) -> Rewritten {
    let mut value: Expr = ctx.mutate_as(&op.value)?;
    let mut value_unchanged = value.same_as(&op.value);
    let mut var: Var = ctx.mutate_as(&op.var)?;
    let var_ty = var.prim_ty()?;
    if value.prim_dtype()? != var_ty.dtype {
        if var.same_as(&op.var) {
            var = copy_with_type(&op.var, value.prim_ty()?);
            ctx.state_mut().base().remap_set(&op.var, var.clone());
        } else {
            value = cast(value, &var_ty)?;
            value_unchanged = false;
        }
    }
    let body: Expr = ctx.mutate_as(&op.body)?;
    if value_unchanged && body.same_as(&op.body) && var.same_as(&op.var) {
        return unchanged();
    }
    changed(Let::new_full(var, value, body, &op.loc)?)
}

/// The two arms of a select take the wider of their types.
fn select<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: Select,
    index: bool,
) -> Rewritten {
    let base = ctx.state_mut().base();
    let flag = index || base.is_condition;
    let condition_flag = std::mem::replace(&mut base.is_condition, flag);
    let condition = ctx.mutate_as(&op.condition);
    ctx.state_mut().base().is_condition = condition_flag;
    let condition: Expr = condition?;
    let true_value: Expr = ctx.mutate_as(&op.true_value)?;
    let false_value: Expr = ctx.mutate_as(&op.false_value)?;
    let (true_ty, false_ty) = (true_value.prim_ty()?, false_value.prim_ty()?);
    if condition.same_as(&op.condition)
        && true_value.same_as(&op.true_value)
        && false_value.same_as(&op.false_value)
        && true_ty.dtype == false_ty.dtype
    {
        return unchanged();
    }
    let (true_value, false_value) = widen(true_value, false_value)?;
    changed(Select::new(condition, true_value, false_value)?)
}

/// Both values cast to the first one's type with the wider of their bit widths.
fn widen(lhs: Expr, rhs: Expr) -> Result<(Expr, Expr)> {
    let (lhs_ty, rhs_ty) = (lhs.prim_ty()?, rhs.prim_ty()?);
    let ty = with_bits(&lhs_ty, lhs_ty.dtype.bits.max(rhs_ty.dtype.bits))?;
    let lhs = if lhs_ty.dtype != ty.dtype { cast(lhs, &ty)? } else { lhs };
    let rhs = if rhs_ty.dtype != ty.dtype { cast(rhs, &ty)? } else { rhs };
    Ok((lhs, rhs))
}

fn legalize_ramp<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: Ramp,
) -> Rewritten {
    let base: Expr = ctx.mutate_as(&op.base_)?;
    let stride: Expr = ctx.mutate_as(&op.stride)?;
    let (base_ty, stride_ty) = (base.prim_ty()?, stride.prim_ty()?);
    if base.same_as(&op.base_) && stride.same_as(&op.stride) && base_ty.dtype == stride_ty.dtype {
        return unchanged();
    }
    check!(
        base_ty.dtype.code == DL_INT && stride_ty.dtype.code == DL_INT,
        INTERNAL_ERROR,
        "base_dtype.MatchesCode(DLDataTypeCode::kDLInt) && \
         stride_dtype.MatchesCode(DLDataTypeCode::kDLInt)"
    );
    let (base, stride) = widen(base, stride)?;
    changed(Ramp::new(base, stride, &op.lanes)?)
}

/// Rebuilt by its constructor, which infers the type again.
fn legalize_broadcast<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: Broadcast,
) -> Rewritten {
    let value: Expr = ctx.mutate_as(&op.value)?;
    let lanes: Expr = ctx.mutate_as(&op.lanes)?;
    if value.same_as(&op.value) && lanes.same_as(&op.lanes) {
        return unchanged();
    }
    changed(Broadcast::new(value, lanes)?)
}

/// Rebuilt by its constructor, which infers the type again.
fn legalize_shuffle<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: Shuffle,
) -> Rewritten {
    let vectors: Array<Expr> = ctx.mutate_as(&op.vectors)?;
    let indices: Array<Expr> = ctx.mutate_as(&op.indices)?;
    if vectors.same_as(&op.vectors) && indices.same_as(&op.indices) {
        return unchanged();
    }
    changed(Shuffle::new(&vectors, &indices)?)
}

/// A shift by the C++ operator; with `clamp_shift_amounts`, a shift amount of a narrowed signed
/// value is clamped below its new width.
fn shift<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    a: &Expr,
    b: &Expr,
    combine: fn(&Expr, &Expr) -> Ex,
) -> Rewritten {
    let before = a.prim_dtype()?;
    let lhs: Expr = ctx.mutate_as(a)?;
    let mut rhs: Expr = ctx.mutate_as(b)?;
    let after = lhs.prim_dtype()?;
    let clamp = ctx.state_mut().base().clamp_shift_amounts;
    if clamp && before.code == DL_INT && after.code == DL_INT && before.bits > after.bits {
        let limit = make_const(rhs.prim_dtype()?, after.bits as i64 - 1)?;
        rhs = rhs.min(&limit).expr()?;
    }
    if lhs.same_as(a) && rhs.same_as(b) && after == rhs.prim_dtype()? {
        return unchanged();
    }
    changed(combine(&lhs, &rhs).expr()?)
}

fn legalize_bitwise_not<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: BitwiseNot,
) -> Rewritten {
    let a: Expr = ctx.mutate_as(&op.a)?;
    if a.same_as(&op.a) {
        return unchanged();
    }
    changed(BitwiseNot::new_full(a, &op.loc)?)
}

/// A binary operation rebuilt by its C++ operator, which matches the operand types.
fn binary<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    a: &Expr,
    b: &Expr,
    combine: fn(&Expr, &Expr) -> Ex,
) -> Rewritten {
    let new_a: Expr = ctx.mutate_as(a)?;
    let new_b: Expr = ctx.mutate_as(b)?;
    if new_a.same_as(a) && new_b.same_as(b) && new_a.prim_dtype()? == new_b.prim_dtype()? {
        return unchanged();
    }
    changed(combine(&new_a, &new_b).expr()?)
}

/// A comparison; under the index rewriter, an integer comparison in a condition is an index.
fn comparison<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    a: &Expr,
    b: &Expr,
    index: bool,
    combine: fn(&Expr, &Expr) -> Ex,
) -> Rewritten {
    if !index {
        return binary(ctx, a, b, combine);
    }
    let is_int = |e: &Expr| e.prim_dtype().map(|d| d.code == DL_INT);
    let enable = ctx.state_mut().base().is_condition && is_int(a)? && is_int(b)?;
    let enabled = std::mem::replace(&mut ctx.state_mut().base().is_enabled, enable);
    let result = binary(ctx, a, b, combine);
    ctx.state_mut().base().is_enabled = enabled;
    result
}

/// `pow`, `if_then_else` and `clz` calls are rebuilt for their new argument types.
fn legalize_call<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    before: Call,
) -> Rewritten {
    let result = ctx.default_mutate(&before)?;
    let call: Call = result.clone().try_into()?;
    if call.ty.as_node::<PrimTypeObj>().is_none() {
        return changed(result);
    }
    if POW.is(&call.op) {
        return changed(call.args.get(0)?.pow(call.args.get(1)?).expr()?);
    }
    if IF_THEN_ELSE.is(&call.op) {
        let ty = call.prim_ty()?;
        let args = Array::new(vec![call.args.get(0)?, call.args.get(1)?, call.args.get(2)?]);
        let rebuilt = Call::new_full(
            Some(ty.into()),
            call.op.clone(),
            args,
            call.attrs.clone(),
            Array::new(vec![]),
            &call.loc,
        )?;
        return changed(rebuilt);
    }
    if CLZ.is(&call.op) {
        let before_dtype = before.args.get(0)?.prim_dtype()?;
        let after_dtype = call.args.get(0)?.prim_dtype()?;
        for (when, d) in [("before", before_dtype), ("after", after_dtype)] {
            if !(matches!(d.code, 0 | 1) && matches!(d.bits, 32 | 64)) {
                bail!(
                    INTERNAL_ERROR,
                    "clz only supports 32 or 64 bit integer types, but get type {when} legalizing: {}",
                    d.to_string().as_str()
                );
            }
        }
        let e: Expr = call.into();
        return changed((&e - after_dtype.bits as i32 + before_dtype.bits as i32).expr()?);
    }
    changed(result)
}

// ---------------------------------------------------------------------------------------------
// IndexDataTypeRewriter

/// Mutate `indices` as tensor indices.
fn visit_indices<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    indices: &Array<Expr>,
) -> Result<Array<Expr>> {
    let enabled = std::mem::replace(&mut ctx.state_mut().base().is_enabled, true);
    let result = ctx.mutate_as(indices);
    ctx.state_mut().base().is_enabled = enabled;
    result
}

/// A stored scalar takes the element type of its (possibly retyped) tensor.
fn index_store<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: TensorStore,
) -> Rewritten {
    let dest: Expr = ctx.mutate_as(&op.dest)?;
    let mut value: Expr = ctx.mutate_as(&op.value)?;
    let mut value_unchanged = value.same_as(&op.value);
    let element = tensor_dtype(&dest)?;
    let value_dtype = value.prim_dtype()?;
    if element.dtype != value_dtype && value_dtype.lanes == 1 {
        value = cast(value, &element)?;
        value_unchanged = false;
    }
    let indices = visit_indices(ctx, &op.indices)?;
    if dest.same_as(&op.dest) && value_unchanged && indices.same_as(&op.indices) {
        return changed(op);
    }
    changed(TensorStore::from_complete_fields(op.loc.clone(), dest, indices, value))
}

fn index_load<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: TensorLoad,
) -> Rewritten {
    let source: Expr = ctx.mutate_as(&op.source)?;
    let indices = visit_indices(ctx, &op.indices)?;
    if source.same_as(&op.source) && indices.same_as(&op.indices) {
        return changed(op);
    }
    changed(TensorLoad::new_full(source, indices.iter(), &op.loc)?)
}

fn index_if<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: If,
) -> Rewritten {
    let condition_flag = std::mem::replace(&mut ctx.state_mut().base().is_condition, true);
    let condition = ctx.mutate_as(&op.condition);
    ctx.state_mut().base().is_condition = condition_flag;
    let condition: Expr = condition?;
    let then_case: SeqStmt = ctx.mutate_as(&op.then_case)?;
    let else_case: Option<SeqStmt> = op.else_case.as_ref().map(|e| ctx.mutate_as(e)).transpose()?;
    let else_same = match (&else_case, &op.else_case) {
        (Some(new), Some(old)) => new.same_as(old),
        (None, None) => true,
        _ => false,
    };
    if condition.same_as(&op.condition) && then_case.same_as(&op.then_case) && else_same {
        return unchanged();
    }
    changed(op.with(|node| {
        node.condition = condition;
        node.then_case = then_case;
        node.else_case = else_case;
    }))
}

/// The loop variable and bounds are indices; the bounds take the variable's type.
fn index_for<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: For,
) -> Rewritten {
    let enabled = std::mem::replace(&mut ctx.state_mut().base().is_enabled, true);
    let header = (|| -> Result<(Var, Expr, Expr)> {
        Ok((ctx.mutate_as(&op.loop_var)?, ctx.mutate_as(&op.min)?, ctx.mutate_as(&op.extent)?))
    })();
    ctx.state_mut().base().is_enabled = enabled;
    let (loop_var, min, extent) = header?;
    let body: SeqStmt = ctx.mutate_as(&op.body)?;
    if loop_var.same_as(&op.loop_var)
        && min.same_as(&op.min)
        && extent.same_as(&op.extent)
        && body.same_as(&op.body)
    {
        return unchanged();
    }
    let var_ty = loop_var.prim_ty()?;
    let min = cast(min, &var_ty)?;
    let extent = cast(extent, &var_ty)?;
    changed(op.with(|node| {
        node.loop_var = loop_var;
        node.min = min;
        node.extent = extent;
        node.body = body;
    }))
}

/// A bound variable replaced before its definition takes the value, cast to its type.
fn index_bind<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: Bind,
) -> Rewritten {
    if op.value.ty.as_node::<PrimTypeObj>().is_none() {
        return changed(ctx.default_mutate(&op)?);
    }
    let Some(var) = ctx.state_mut().base().remap_get(&op.var).cloned() else {
        return legalize_bind(ctx, op);
    };
    let enabled = std::mem::replace(&mut ctx.state_mut().base().is_enabled, true);
    let value = ctx.mutate_as(&op.value);
    ctx.state_mut().base().is_enabled = enabled;
    let value = cast(value?, &var.prim_ty()?)?;
    changed(Bind::new_full(var, value, &op.loc)?)
}

/// `if_then_else`: the condition is a condition, and the arms take the wider type.
fn index_if_then_else<S: DataTypeRewriterState, D: tvm_ffi::MutateDriver<S> + ?Sized>(
    ctx: Ctx<'_, '_, S, D>,
    op: Call,
) -> Rewritten {
    let condition_flag = std::mem::replace(&mut ctx.state_mut().base().is_condition, true);
    let condition = ctx.mutate_as(&op.args.get(0)?);
    ctx.state_mut().base().is_condition = condition_flag;
    let condition: Expr = condition?;
    let true_value: Expr = ctx.mutate_as(&op.args.get(1)?)?;
    let false_value: Expr = ctx.mutate_as(&op.args.get(2)?)?;
    let (true_value, false_value) = widen(true_value, false_value)?;
    let ty = true_value.prim_ty()?;
    let call = Call::new_full(
        Some(ty.into()),
        op.op.clone(),
        Array::new(vec![condition, true_value, false_value]),
        op.attrs.clone(),
        Array::new(vec![]),
        &op.loc,
    )?;
    changed(call)
}
