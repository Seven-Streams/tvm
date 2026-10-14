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

//! `tirx.LowerIntrin` (`src/tirx/transform/lower_intrin.cc`): apply the target's intrinsic
//! lowering rules (`<target>.FLowerIntrinsic` / `FLegalize` op attributes), lower floor
//! division and modulo to truncating forms, and fuse multiply-add.

use tvm_ffi::{
    dispatch, Mutator, Any, DLDataType, Function as FfiFunction, ObjectRefCast, ObjectRefCore,
    Result, String,
};

use super::analyzer::{binary, AnalyzerBase, AnalyzerScope, AnalyzerState};
use super::util::{
    function_pass, get_attr, int_imm, make_const, prim_dtype, prim_type, seq_stmt_from,
    unknown_loc, with_body, MutatorExt,
};
use crate::ir::{Call, Expr, IntImmObj, Op, OpAttrMap, PrimTypeObj, Stmt, Type, Var};
use crate::prim::{
    Add, BroadcastObj, CastObj, FloorDiv, FloorDivObj, FloorMod, FloorModObj, Max, MulObj, EQ, NE,
};
use crate::sym::Analyzer;
use crate::target::Target;
use crate::transform::Pass;

const DL_INT: u8 = 0;
const DL_UINT: u8 = 1;
const DL_FLOAT: u8 = 2;

/// The op-attribute columns holding the target's lowering rules, in lookup order.
struct LoweringRules {
    columns: Vec<OpAttrMap<FfiFunction>>,
    fma_op: Op,
    /// The first rule for `prim.fma`, as C++ `fma_`.
    fma: Option<FfiFunction>,
}

impl LoweringRules {
    fn for_target(target: &Target, fast_math: bool) -> Result<Self> {
        let kind = target.kind()?.name()?;
        let kind = kind.as_str();
        let mtriple = match target.attrs()?.get(&String::from("mtriple"))? {
            Some(value) => String::try_from(value)?,
            None => String::default(),
        };
        let mut patterns = Vec::new();
        if fast_math {
            patterns.extend([
                format!("{kind}.fastmath.FLowerIntrinsic"),
                format!("{kind}.fastmath.FLegalize"),
            ]);
        }
        patterns.extend([format!("{kind}.FLowerIntrinsic"), format!("{kind}.FLegalize")]);
        if mtriple.as_str().contains("aarch64") {
            patterns.extend([
                format!("{kind}.aarch64.FLowerIntrinsic"),
                format!("{kind}.aarch64.FLegalize"),
            ]);
        }
        patterns.extend(["default.FLowerIntrinsic".to_owned(), "default.FLegalize".to_owned()]);

        let mut columns = Vec::new();
        for pattern in patterns {
            if let Some(column) = Op::attr_map(&pattern)? {
                columns.push(column);
            }
        }
        let fma_op = Op::get("prim.fma")?;
        let mut fma = None;
        for column in &columns {
            if let Some(rule) = column.get(&fma_op)? {
                fma = Some(rule);
                break;
            }
        }
        Ok(Self { columns, fma_op, fma })
    }

    /// The first rule that rewrites `call`, if any.
    fn apply(&self, call: &Call) -> Result<Option<Expr>> {
        let Ok(op) = call.op.clone().try_cast::<Op>() else { return Ok(None) };
        if call.ty.as_node::<PrimTypeObj>().is_none() {
            return Ok(None);
        }
        let expr: Expr = call.clone().into();
        for column in &self.columns {
            let Some(rule) = column.get(&op)? else { continue };
            let lowered: Expr = rule.call_tuple((expr.clone(),))?.try_into()?;
            if !lowered.same_as(&expr) {
                return Ok(Some(lowered));
            }
        }
        Ok(None)
    }
}

struct IntrinInjecter {
    base: AnalyzerBase,
    rules: LoweringRules,
    support_bitwise_op: bool,
}

impl AnalyzerState for IntrinInjecter {
    fn base(&mut self) -> &mut AnalyzerBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = AnalyzerScope)]
impl IntrinInjecter {
    /// A call the target has a rule for becomes the rule's result, itself lowered.
    fn mutate_call(&mut self, call: Call, m: &mut Mutator) -> Result<Any> {
        match self.rules.apply(&call)? {
            Some(lowered) => m.mutate(self, &lowered),
            None => m.default_mutate(self, &call),
        }
    }

    /// `a * b + c` becomes a fused multiply-add.
    fn mutate_add(&mut self, op: Add, m: &mut Mutator) -> Result<Any> {
        let (mul, c) = match (op.b.as_node::<MulObj>(), op.a.as_node::<MulObj>()) {
            (Some(mul), _) => (mul, &op.a),
            (None, Some(mul)) => (mul, &op.b),
            (None, None) => return m.default_mutate(self, &op),
        };
        let (a, b, c) = (mul.a.clone(), mul.b.clone(), c.clone());
        self.make_fma(op, a, b, c, m)
    }

    fn mutate_floordiv(&mut self, op: FloorDiv, m: &mut Mutator) -> Result<Any> {
        let original = Expr::from(op);
        let lowered: Expr = m.default_mutate_as(self, &original)?;
        if lowered.as_node::<FloorDivObj>().is_none() {
            return Ok(lowered.into());
        }
        self.lower_floordiv(&lowered, &original, m).map(Any::from)
    }

    fn mutate_floormod(&mut self, op: FloorMod, m: &mut Mutator) -> Result<Any> {
        let lowered: Expr = m.default_mutate_as(self, &Expr::from(op))?;
        if lowered.as_node::<FloorModObj>().is_none() {
            return Ok(lowered.into());
        }
        self.lower_floormod(&lowered, m).map(Any::from)
    }

    /// `max(floordiv(x, y), c)` with `c >= 0` and `y >= 0` is `max(truncdiv(x, y), c)`.
    fn mutate_max(&mut self, op: Max, m: &mut Mutator) -> Result<Any> {
        if let (Some(div), Some(c)) = (op.a.as_node::<FloorDivObj>(), op.b.as_node::<IntImmObj>()) {
            if c.value >= tvm_ffi::BigInt::from(0i64)
                && self.analyzer().can_prove_greater_equal(&div.b, 0)?
            {
                let truncated = truncdiv(&div.a, &div.b)?;
                let lowered: Expr = m.mutate_as(self, &truncated)?;
                return binary("prim._OpMax", &lowered, &op.b).map(Any::from);
            }
        }
        m.default_mutate(self, &op)
    }

    /// `floormod(x, y) == 0` is `truncmod(x, y) == 0`.
    fn mutate_eq(&mut self, op: EQ, m: &mut Mutator) -> Result<Any> {
        match divisibility(&op.a, &op.b, "prim._OpEQ")? {
            Some(test) => m.mutate(self, &test),
            None => m.default_mutate(self, &op),
        }
    }

    /// `floormod(x, y) != 0` is `truncmod(x, y) != 0`.
    fn mutate_ne(&mut self, op: NE, m: &mut Mutator) -> Result<Any> {
        match divisibility(&op.a, &op.b, "prim._OpNE")? {
            Some(test) => m.mutate(self, &test),
            None => m.default_mutate(self, &op),
        }
    }
}

impl IntrinInjecter {
    fn analyzer(&self) -> &Analyzer {
        &self.base.analyzer
    }

    fn make_fma(&mut self, op: Add, a: Expr, b: Expr, c: Expr, m: &mut Mutator) -> Result<Any> {
        let lhs = swap_broadcast_cast(&a)?;
        let rhs = swap_broadcast_cast(&b)?;
        match (&self.rules.fma, prim_dtype(&op.clone().into())?.code == DL_FLOAT) {
            (Some(fma), true) => {
                let call = call(&op.ty, &self.rules.fma_op, vec![lhs, rhs, c])?;
                let fused = fma.call_tuple((call,))?;
                if fused.type_index() != 0 {
                    return m.mutate(self, &Expr::try_from(fused)?);
                }
            }
            _ if !lhs.same_as(&a) || !rhs.same_as(&b) => {
                let product: Expr = m.mutate_as(self, &raw_binary("prim.Mul", &lhs, &rhs)?)?;
                let c: Expr = m.mutate_as(self, &c)?;
                return raw_binary("prim.Add", &product, &c).map(Any::from);
            }
            _ => {}
        }
        m.default_mutate(self, &op)
    }

    /// `TryFindShiftCoefficientForPositiveRange`: a `c` with `a + b * c` non-negative and
    /// without overflow.
    fn shift_coefficient(&self, a: &Expr, b: i64) -> Result<Option<i64>> {
        if b <= 0 {
            return Ok(None);
        }
        let bound = self.analyzer().const_int_bound(a)?;
        if bound.min_value >= 0 {
            return Ok(None);
        }
        let dtype = DLDataType { lanes: 1, ..prim_dtype(a)? };
        let max_value: Expr = tvm_ffi::cached_global_func!("prim.max_value")
            .call_tuple((prim_type(dtype)?, unknown_loc()?))?
            .try_into()?;
        let Some(dtype_max) = int_value(&max_value) else { return Ok(None) };
        let (b, min, max, dtype_max) =
            (b as i128, bound.min_value as i128, bound.max_value as i128, dtype_max as i128);
        if b - 1 > dtype_max + min {
            return Ok(None);
        }
        let c = ((b - 1) - min) / b;
        if c <= 0 {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::INTERNAL_ERROR,
                "Check failed: c_value > 0",
                "",
            ));
        }
        if c > dtype_max / b || max > dtype_max - b * c {
            return Ok(None);
        }
        Ok(Some(c as i64))
    }

    /// `floordiv(a, b)`, its operands already lowered, in truncating arithmetic.
    /// `original` is the node before its operands were lowered: C++ asks whether it is
    /// non-negative, and its default descent copies rather than mutating in place.
    fn lower_floordiv(&mut self, e: &Expr, original: &Expr, m: &mut Mutator) -> Result<Expr> {
        let div = e.as_node::<FloorDivObj>().expect("checked by the caller");
        let (a, b) = (&div.a, &div.b);
        let dtype = prim_dtype(e)?;
        if let Some(shift) = power_of_two_shift(b).filter(|_| self.support_bitwise_op) {
            return right_shift(a, &imm(dtype, shift as i128)?);
        }
        if !self.analyzer().can_prove_greater_equal(b, 0)? {
            if dtype.code == DL_FLOAT {
                let floor = call_global("prim.floor", &[&binary("prim._OpDiv", a, b)?])?;
                return m.mutate_as(self, &floor);
            }
            // Neither sign known: correct the truncating quotient when the signs differ.
            let rmod = prim_var("rmod", dtype)?;
            let rdiv = prim_var("rdiv", dtype)?;
            let (rmod_e, rdiv_e): (Expr, Expr) = (rmod.clone().into(), rdiv.clone().into());
            let cond = signs_agree(b, &rmod_e, dtype)?;
            let corrected = select(&cond, &rdiv_e, &sub(&rdiv_e, &make_const(dtype, 1)?)?)?;
            return let_(&rmod, &truncmod(a, b)?, &let_(&rdiv, &truncdiv(a, b)?, &corrected)?);
        }
        if self.analyzer().can_prove_greater_equal(a, 0)?
            || self.analyzer().can_prove_greater_equal(original, 0)?
        {
            return truncdiv(a, b);
        }
        if let Some(b_value) = int_value(b) {
            if let Some(c) = self.shift_coefficient(a, b_value)? {
                // Shift `a` into the non-negative range by a multiple of `b`, then back.
                let shifted = add(a, &imm(dtype, b_value as i128 * c as i128)?)?;
                return sub(&truncdiv(&shifted, b)?, &imm(dtype, c as i128)?);
            }
        }
        let rdiv = truncdiv(a, b)?;
        let rmod = truncmod(a, b)?;
        if is_int32_or_int64(dtype) && self.support_bitwise_op {
            return add(&rdiv, &right_shift(&rmod, &imm(dtype, dtype.bits as i128 - 1)?)?);
        }
        select(&ge(&rmod, &make_const(dtype, 0)?)?, &rdiv, &sub(&rdiv, &make_const(dtype, 1)?)?)
    }

    /// `floormod(a, b)`, its operands already lowered, in truncating arithmetic.
    fn lower_floormod(&mut self, e: &Expr, m: &mut Mutator) -> Result<Expr> {
        let modulo = e.as_node::<FloorModObj>().expect("checked by the caller");
        let (a, b) = (&modulo.a, &modulo.b);
        let dtype = prim_dtype(e)?;
        if let Some(shift) = power_of_two_shift(b).filter(|_| self.support_bitwise_op) {
            return bitwise_and(a, &imm(dtype, (1i128 << shift) - 1)?);
        }
        if !self.analyzer().can_prove_greater_equal(b, 0)? {
            if dtype.code == DL_FLOAT {
                let floor = call_global("prim.floor", &[&binary("prim._OpDiv", a, b)?])?;
                let floor: Expr = m.mutate_as(self, &floor)?;
                return sub(a, &binary("prim._OpMul", &floor, b)?);
            }
            let rmod = prim_var("rmod", dtype)?;
            let rmod_e: Expr = rmod.clone().into();
            let cond = signs_agree(b, &rmod_e, dtype)?;
            return let_(&rmod, &truncmod(a, b)?, &select(&cond, &rmod_e, &add(&rmod_e, b)?)?);
        }
        if self.analyzer().can_prove_greater_equal(a, 0)? {
            return truncmod(a, b);
        }
        if let Some(b_value) = int_value(b) {
            if let Some(c) = self.shift_coefficient(a, b_value)? {
                return truncmod(&add(a, &imm(dtype, b_value as i128 * c as i128)?)?, b);
            }
        }
        let rmod = truncmod(a, b)?;
        if is_int32_or_int64(dtype) && self.support_bitwise_op {
            let sign = right_shift(&rmod, &imm(dtype, dtype.bits as i128 - 1)?)?;
            return add(&rmod, &bitwise_and(b, &sign)?);
        }
        select(&ge(&rmod, &make_const(dtype, 0)?)?, &rmod, &add(&rmod, b)?)
    }
}

/// `floormod(x, y) <compare> 0` as `truncmod(x, y) <compare> 0`; `None` for anything else.
fn divisibility(lhs: &Expr, rhs: &Expr, compare: &str) -> Result<Option<Expr>> {
    match (lhs.as_node::<FloorModObj>(), int_value(rhs)) {
        (Some(modulo), Some(0)) => {
            let truncated = truncmod(&modulo.a, &modulo.b)?;
            let zero = make_const(prim_dtype(&truncated)?, 0)?;
            binary(compare, &truncated, &zero).map(Some)
        }
        _ => Ok(None),
    }
}

/// `(b >= 0 && r >= 0) || (b < 0 && r <= 0)`: the remainder `r` has the sign of `b`.
fn signs_agree(b: &Expr, r: &Expr, dtype: DLDataType) -> Result<Expr> {
    let zero = make_const(dtype, 0)?;
    let nonneg = binary("prim._OpAnd", &ge(b, &zero)?, &ge(r, &zero)?)?;
    let negative =
        binary("prim._OpAnd", &binary("prim._OpLT", b, &zero)?, &binary("prim._OpLE", r, &zero)?)?;
    binary("prim._OpOr", &nonneg, &negative)
}

/// `SwapBroadcastCast`: `broadcast(cast(x))` to `cast(broadcast(x))` when the cast widens.
fn swap_broadcast_cast(expr: &Expr) -> Result<Expr> {
    let Some(bcast) = expr.as_node::<BroadcastObj>() else { return Ok(expr.clone()) };
    let Some(cast) = bcast.value.as_node::<CastObj>() else { return Ok(expr.clone()) };
    let to = cast.ty.as_node::<PrimTypeObj>().expect("a cast has a PrimType").dtype;
    let from = prim_dtype(&cast.value)?;
    let integral = |code| code == DL_INT || code == DL_UINT;
    let widens = to.bits as u32 == from.bits as u32 * 2
        || (integral(to.code) && integral(from.code) && to.bits > from.bits);
    if !widens {
        return Ok(expr.clone());
    }
    let broadcast = call_global("prim.Broadcast", &[&cast.value, &bcast.lanes])?;
    tvm_ffi::cached_global_func!("prim.Cast")
        .call_tuple((broadcast, bcast.ty.clone(), unknown_loc()?))?
        .try_into()
}

/// `Call(ty, op, args)`.
fn call(ty: &Type, op: &Op, args: Vec<Expr>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("ir.Call")
        .call_tuple((
            ty.clone(),
            op.clone(),
            tvm_ffi::Array::new(args),
            Option::<crate::ir::Attrs>::None,
            tvm_ffi::Array::<Type>::new(vec![]),
            unknown_loc()?,
        ))?
        .try_into()
}

/// A global taking expressions and a trailing location.
fn call_global(name: &str, args: &[&Expr]) -> Result<Expr> {
    let loc = unknown_loc()?;
    let mut packed: Vec<tvm_ffi::AnyView> =
        args.iter().map(|arg| tvm_ffi::AnyView::from(*arg)).collect();
    packed.push(tvm_ffi::AnyView::from(&loc));
    FfiFunction::get_global(name)?.call_packed(&packed)?.try_into()
}

/// A raw binary constructor (`prim.Add`, `prim.Mul`): no folding or type matching.
fn raw_binary(name: &str, a: &Expr, b: &Expr) -> Result<Expr> {
    call_global(name, &[a, b])
}

fn add(a: &Expr, b: &Expr) -> Result<Expr> {
    binary("prim._OpAdd", a, b)
}
fn sub(a: &Expr, b: &Expr) -> Result<Expr> {
    binary("prim._OpSub", a, b)
}
fn ge(a: &Expr, b: &Expr) -> Result<Expr> {
    binary("prim._OpGE", a, b)
}
fn truncdiv(a: &Expr, b: &Expr) -> Result<Expr> {
    binary("prim._OpTruncDiv", a, b)
}
fn truncmod(a: &Expr, b: &Expr) -> Result<Expr> {
    binary("prim._OpTruncMod", a, b)
}
fn right_shift(a: &Expr, b: &Expr) -> Result<Expr> {
    call_global("prim.right_shift", &[a, b])
}
fn bitwise_and(a: &Expr, b: &Expr) -> Result<Expr> {
    call_global("prim.bitwise_and", &[a, b])
}
/// `prim::Select(c, t, f)`, the raw constructor.
fn select(c: &Expr, t: &Expr, f: &Expr) -> Result<Expr> {
    call_global("prim.Select", &[c, t, f])
}
/// `prim::Let(var, value, body)`, the raw constructor.
fn let_(var: &Var, value: &Expr, body: &Expr) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.Let")
        .call_tuple((var.clone(), value.clone(), body.clone(), unknown_loc()?))?
        .try_into()
}
fn imm(dtype: DLDataType, value: i128) -> Result<Expr> {
    Ok(int_imm(dtype, value)?.into())
}
/// `PrimVar(name, dtype)`.
fn prim_var(name: &str, dtype: DLDataType) -> Result<Var> {
    tvm_ffi::cached_global_func!("ir.Var")
        .call_tuple((String::from(name), prim_type(dtype)?, unknown_loc()?))?
        .try_into()
}

fn int_value(expr: &Expr) -> Option<i64> {
    expr.as_node::<IntImmObj>().and_then(|imm| imm.value.to_i64())
}

/// `IsPowerOfTwoInt(x, &shift)`.
fn power_of_two_shift(expr: &Expr) -> Option<u32> {
    let value = int_value(expr)?;
    (value > 0 && value & (value - 1) == 0).then(|| value.trailing_zeros())
}

fn is_int32_or_int64(dtype: DLDataType) -> bool {
    dtype.code == DL_INT && dtype.lanes == 1 && (dtype.bits == 32 || dtype.bits == 64)
}

/// `ctx->GetConfig<bool>(key)`; C++ `TypeTraits<bool>` also accepts an `int`.
fn config_flag(value: Option<Any>) -> Result<bool> {
    let Some(value) = value else { return Ok(false) };
    if let Some(flag) = value.try_as::<bool>() {
        return Ok(flag);
    }
    Ok(i64::try_from(value)? != 0)
}

pub fn lower_intrin() -> Result<Pass> {
    function_pass("tirx.LowerIntrin", 0, |func, _module, context| {
        let Some(target) = get_attr(&func.attrs, "target")? else {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::INTERNAL_ERROR,
                "Check failed: target.has_value() is false: LowerIntrin: Require the target attribute",
                "",
            ));
        };
        let target: Target = target.try_into()?;
        let fast_math = config_flag(context.config.get(&String::from("tirx.enable_fast_math"))?)?;
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let mut injecter = IntrinInjecter {
            base: AnalyzerBase::new(Analyzer::new()?)?,
            rules: LoweringRules::for_target(&target, fast_math)?,
            support_bitwise_op: true,
        };
        let body: Stmt = tvm_ffi::structural_mutate(body, &mut injecter)?.try_into()?;
        Ok(Some(with_body(&func, Some(seq_stmt_from(body)?))))
    })
}
