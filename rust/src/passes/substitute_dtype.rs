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

//! `SubstituteWithDataTypeLegalization(expr, vmap)` (`src/tirx/ir/stmt_functor.cc`) for an
//! expression: replace variables, then rebuild each expression above a replacement the way the C++
//! `DataTypeLegalizer` (`src/tirx/ir/data_type_rewriter.cc`) does, so the types match again. A
//! rebuilt binary operation goes through its operator, which also folds constants.
//!
//! Only the expression handlers of `DataTypeLegalizer` are here; the statement ones (`For`,
//! `RegionStmt`, `Bind`) never apply to an expression.

use std::collections::HashMap;

use tvm_ffi::{
    dispatch, structural_mutate, structural_walk, Any, AnyView, Array, Error, Mutator,
    DLDataTypeExt, ObjectCore, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, WalkOrder,
    WalkResult, INTERNAL_ERROR,
};

use super::util::{prim_dtype, prim_type, MutatorExt};
use crate::ir::{Call, Expr, Op, PrimTypeObj, TensorLoad, TensorLoadObj, Var, VarObj};
use crate::ops::{ex, Ex, PrimOps};
use crate::prim::{
    Add, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, Broadcast, Div, FloorDiv, FloorMod, LShift,
    Let, Max, Min, Mod, Mul, Ramp, RShift, Select, Shuffle, Sub, EQ, GE, GT, LE, LT, NE,
};
use crate::tirx::TensorTypeObj;

/// `SubstituteWithDataTypeLegalization(expr, vmap)`: `vmap` is asked once for each variable in
/// `expr`, in pre-order.
pub fn substitute_with_dtype_legalization(
    expr: Expr,
    mut vmap: impl FnMut(&Var) -> Result<Option<Expr>>,
) -> Result<Expr> {
    let mut collector = VarCollector { seen: HashMap::new(), order: Vec::new() };
    structural_walk(&expr, &mut collector, WalkOrder::PreOrder)?;
    let mut remap = HashMap::new();
    for var in collector.order {
        if let Some(replacement) = vmap(&var)? {
            remap.insert(ObjectIdentity::of(&var), replacement);
        }
    }
    let mut legalizer = Legalizer {
        remap,
        pow: Op::get("prim.pow")?,
        if_then_else: Op::get("prim.if_then_else")?,
        clz: Op::get("prim.clz")?,
    };
    structural_mutate(expr, &mut legalizer)?.try_into()
}

struct VarCollector {
    seen: HashMap<ObjectIdentity, ()>,
    order: Vec<Var>,
}

#[dispatch(walk)]
impl VarCollector {
    fn walk_var(&mut self, var: Var) -> WalkResult {
        if self.seen.insert(ObjectIdentity::of(&var), ()).is_some() {
            return WalkResult::Skip;
        }
        self.order.push(var);
        WalkResult::Advance
    }
}

struct Legalizer {
    /// `VarRemapSet`: the replacements, and the retyped variables of `Let`.
    remap: HashMap<ObjectIdentity, Expr>,
    pow: Op,
    if_then_else: Op,
    clz: Op,
}

/// C++ `as<TensorVar>()`: a `Var`, not a subclass, whose type is a `TensorType`.
pub(crate) fn is_tensor_var(expr: &Expr) -> bool {
    AnyView::from(expr).type_index() == VarObj::type_index()
        && expr.ty.as_node::<TensorTypeObj>().is_some()
}

fn same_type(a: &Expr, b: &Expr) -> Result<bool> {
    Ok(prim_dtype(a)? == prim_dtype(b)?)
}

impl Legalizer {
    /// `TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH`: unchanged operands of one type keep the
    /// node; otherwise the operator rebuilds it.
    fn binary(
        &mut self,
        m: &mut Mutator,
        node: Expr,
        a: &Expr,
        b: &Expr,
        op: impl FnOnce(Ex, Expr) -> Ex,
    ) -> Result<Any> {
        let new_a: Expr = m.mutate_as(self, a)?;
        let new_b: Expr = m.mutate_as(self, b)?;
        if new_a.same_as(a) && new_b.same_as(b) && same_type(&new_a, &new_b)? {
            return Ok(node.into());
        }
        Ok(op(ex(new_a), new_b).expr()?.into())
    }

    /// The left and right shifts; the amounts are not clamped (`ShouldClampShiftAmounts`).
    fn shift(
        &mut self,
        m: &mut Mutator,
        node: Expr,
        a: &Expr,
        b: &Expr,
        op: impl FnOnce(Ex, Expr) -> Ex,
    ) -> Result<Any> {
        let lhs: Expr = m.mutate_as(self, a)?;
        let rhs: Expr = m.mutate_as(self, b)?;
        if lhs.same_as(a) && rhs.same_as(b) && same_type(&lhs, &rhs)? {
            return Ok(node.into());
        }
        Ok(op(ex(lhs), rhs).expr()?.into())
    }
}

#[dispatch(mutate)]
impl Legalizer {
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        match self.remap.get(&ObjectIdentity::of(&var)) {
            Some(replacement) => Ok(replacement.clone().into()),
            None => m.default_mutate(self, &var),
        }
    }

    fn mutate_add(&mut self, op: Add, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a + b)
    }
    fn mutate_sub(&mut self, op: Sub, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a - b)
    }
    fn mutate_mul(&mut self, op: Mul, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a * b)
    }
    fn mutate_div(&mut self, op: Div, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.div(b))
    }
    fn mutate_mod(&mut self, op: Mod, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.truncmod(b))
    }
    fn mutate_floordiv(&mut self, op: FloorDiv, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.floordiv(b))
    }
    fn mutate_floormod(&mut self, op: FloorMod, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.floormod(b))
    }
    fn mutate_min(&mut self, op: Min, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.min(b))
    }
    fn mutate_max(&mut self, op: Max, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.max(b))
    }
    fn mutate_eq(&mut self, op: EQ, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.equal(b))
    }
    fn mutate_ne(&mut self, op: NE, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.not_equal(b))
    }
    fn mutate_le(&mut self, op: LE, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.less_equal(b))
    }
    fn mutate_lt(&mut self, op: LT, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.less(b))
    }
    fn mutate_gt(&mut self, op: GT, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.greater(b))
    }
    fn mutate_ge(&mut self, op: GE, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a.greater_equal(b))
    }
    fn mutate_bitwise_and(&mut self, op: BitwiseAnd, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a & b)
    }
    fn mutate_bitwise_or(&mut self, op: BitwiseOr, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a | b)
    }
    fn mutate_bitwise_xor(&mut self, op: BitwiseXor, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.binary(m, op.into(), &a, &b, |a, b| a ^ b)
    }
    fn mutate_left_shift(&mut self, op: LShift, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.shift(m, op.into(), &a, &b, |a, b| a << b)
    }
    fn mutate_right_shift(&mut self, op: RShift, m: &mut Mutator) -> Result<Any> {
        let (a, b) = (op.a.clone(), op.b.clone());
        self.shift(m, op.into(), &a, &b, |a, b| a >> b)
    }

    fn mutate_bitwise_not(&mut self, op: BitwiseNot, m: &mut Mutator) -> Result<Any> {
        let a: Expr = m.mutate_as(self, &op.a)?;
        if a.same_as(&op.a) {
            return Ok(op.into());
        }
        Ok(BitwiseNot::new_full(a, op.loc.clone())?.into())
    }

    /// Both values are cast to the wider of their types.
    fn mutate_select(&mut self, op: Select, m: &mut Mutator) -> Result<Any> {
        let condition: Expr = m.mutate_as(self, &op.condition)?;
        let true_value: Expr = m.mutate_as(self, &op.true_value)?;
        let false_value: Expr = m.mutate_as(self, &op.false_value)?;
        if condition.same_as(&op.condition)
            && true_value.same_as(&op.true_value)
            && false_value.same_as(&op.false_value)
            && same_type(&true_value, &false_value)?
        {
            return Ok(op.into());
        }
        let (true_dtype, false_dtype) = (prim_dtype(&true_value)?, prim_dtype(&false_value)?);
        let dtype =
            tvm_ffi::DLDataType { bits: true_dtype.bits.max(false_dtype.bits), ..true_dtype };
        let true_value = match true_dtype == dtype {
            true => true_value,
            false => true_value.cast(dtype).expr()?,
        };
        let false_value = match false_dtype == dtype {
            true => false_value,
            false => false_value.cast(dtype).expr()?,
        };
        Ok(Select::new(condition, true_value, false_value)?.into())
    }

    /// The base and the stride are cast to the wider of their integer types.
    fn mutate_ramp(&mut self, op: Ramp, m: &mut Mutator) -> Result<Any> {
        let base: Expr = m.mutate_as(self, &op.base_)?;
        let stride: Expr = m.mutate_as(self, &op.stride)?;
        if base.same_as(&op.base_) && stride.same_as(&op.stride) && same_type(&base, &stride)? {
            return Ok(op.into());
        }
        let (base_dtype, stride_dtype) = (prim_dtype(&base)?, prim_dtype(&stride)?);
        let int = tvm_ffi::DLDataTypeCode::kDLInt as u8;
        if base_dtype.code != int || stride_dtype.code != int {
            let condition = "base_dtype.MatchesCode(DLDataTypeCode::kDLInt) && \
                             stride_dtype.MatchesCode(DLDataTypeCode::kDLInt)";
            return Err(Error::new(
                INTERNAL_ERROR,
                &format!("Check failed: ({condition}) is false: "),
                "",
            ));
        }
        let dtype =
            tvm_ffi::DLDataType { bits: base_dtype.bits.max(stride_dtype.bits), ..base_dtype };
        let base = match base_dtype == dtype {
            true => base,
            false => base.cast(dtype).expr()?,
        };
        let stride = match stride_dtype == dtype {
            true => stride,
            false => stride.cast(dtype).expr()?,
        };
        Ok(Ramp::new(base, stride, op.lanes.clone())?.into())
    }

    /// Rebuilt by the constructor, which infers the type.
    fn mutate_broadcast(&mut self, op: Broadcast, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let lanes: Expr = m.mutate_as(self, &op.lanes)?;
        if value.same_as(&op.value) && lanes.same_as(&op.lanes) {
            return Ok(op.into());
        }
        Ok(Broadcast::new(value, lanes)?.into())
    }

    fn mutate_shuffle(&mut self, op: Shuffle, m: &mut Mutator) -> Result<Any> {
        let vectors: Array<Expr> = m.mutate_as(self, &op.vectors)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        if vectors.same_as(&op.vectors) && indices.same_as(&op.indices) {
            return Ok(op.into());
        }
        Ok(Shuffle::new(&vectors, &indices)?.into())
    }

    /// A value of another type retypes the variable, or is cast to a variable already replaced.
    fn mutate_let(&mut self, op: Let, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let mut value_unchanged = value.same_as(&op.value);
        let mut value = value;
        let var: Var = m.mutate(self, &op.var)?.try_into()?;
        let var_dtype = var.ty.as_node::<PrimTypeObj>().map(|ty| ty.dtype);
        let mut var = var;
        if Some(prim_dtype(&value)?) != var_dtype {
            if var.same_as(&op.var) {
                let ty = prim_type(prim_dtype(&value)?)?;
                let retyped = Var::new_full(op.var.name.clone(), ty, op.var.loc.clone())?;
                self.remap.insert(ObjectIdentity::of(&op.var), retyped.clone().into());
                var = retyped;
            } else {
                let var_ty = var.ty.clone().try_cast::<crate::ir::PrimType>()?;
                value = value.cast(var_ty).expr()?;
                value_unchanged = false;
            }
        }
        let body: Expr = m.mutate_as(self, &op.body)?;
        if value_unchanged && body.same_as(&op.body) && var.same_as(&op.var) {
            return Ok(op.into());
        }
        Ok(Let::new_full(var, value, body, op.loc.clone())?.into())
    }

    /// `pow`, `if_then_else` and `clz` are rebuilt for the new argument types.
    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        let before = op.clone();
        let e: Expr = m.default_mutate_as(self, &Expr::from(op))?;
        let Some(call) = e.as_node::<crate::ir::CallObj>() else {
            return Err(Error::new(INTERNAL_ERROR, "Expected type to be CallNode", ""));
        };
        if call.ty.as_node::<PrimTypeObj>().is_none() {
            return Ok(e.into());
        }
        if call.op.same_as(&self.pow) {
            return Ok(call.args.get(0)?.pow(call.args.get(1)?).expr()?.into());
        }
        if call.op.same_as(&self.if_then_else) {
            let args = Array::new(vec![call.args.get(0)?, call.args.get(1)?, call.args.get(2)?]);
            let rebuilt = Call::new_full(
                Some(call.ty.clone()),
                call.op.clone(),
                args,
                call.attrs.clone(),
                Array::new(vec![]),
                call.loc.clone(),
            )?;
            return Ok(rebuilt.into());
        }
        if call.op.same_as(&self.clz) {
            let (before_dtype, after_dtype) =
                (prim_dtype(&before.args.get(0)?)?, prim_dtype(&call.args.get(0)?)?);
            let integer = |d: tvm_ffi::DLDataType| d.code <= 1 && (d.bits == 32 || d.bits == 64);
            for (when, dtype) in [("before", before_dtype), ("after", after_dtype)] {
                if !integer(dtype) {
                    let message = format!(
                        "clz only supports 32 or 64 bit integer types, but get type {when} legalizing: {}",
                        dtype.to_string().as_str()
                    );
                    return Err(Error::new(INTERNAL_ERROR, &message, ""));
                }
            }
            let corrected = ex(e) - after_dtype.bits as i64 + before_dtype.bits as i64;
            return Ok(corrected.expr()?.into());
        }
        Ok(e.into())
    }

    /// A changed load of a tensor variable is rebuilt by the constructor, which infers its type.
    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let original: Expr = op.clone().into();
        let result: Expr = m.default_mutate_as(self, &original)?;
        if result.same_as(&original) {
            return Ok(op.into());
        }
        let load = result.as_node::<TensorLoadObj>().expect("a load mutates to a load");
        if is_tensor_var(&load.source) {
            let indices: Vec<Expr> = load.indices.iter().collect();
            return Ok(TensorLoad::new_full(load.source.clone(), indices, load.loc.clone())?.into());
        }
        Ok(result.into())
    }
}
