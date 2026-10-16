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

//! The operators: `Ex`, literals, the operations of `src/ir/prim/op.cc` and `PrimOps`.

use std::cell::OnceCell;
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
};

use tvm_ffi::{BigInt, DLDataType, DLDataTypeCode, ObjectRefCore, Result, INTERNAL_ERROR};

use super::build::{
    big_int_imm, float_imm, int32, int_imm_at, make_const, scalar_dtype, unknown_loc, IntoExpr,
    IntoPrimType,
};
use super::native::repr;
use super::{fold, math};
use crate::ir::{
    Call, CallObj, Expr, FloatImm, FloatImmObj, IntImm, IntImmObj, Location, TensorLoad, Type, Var,
};
use crate::prim;
use crate::prim::op::{IF_THEN_ELSE, VSCALE};

// ---------------------------------------------------------------------------------------------
// `Ex`

/// An expression being built by operators: either the expression, or the first error any step
/// raised. Later steps pass an error through untouched, so `?` is needed once, on `.expr()`.
#[derive(Clone)]
#[must_use = "an `Ex` holds the result; take it with `.expr()`"]
pub struct Ex(Result<Expr>);

impl Ex {
    /// The built expression, or the first error.
    pub fn expr(self) -> Result<Expr> {
        self.0
    }

    /// The dtype a literal next to this expression takes (`MakeConst(this.ty(), v)`).
    fn dtype(&self) -> Result<DLDataType> {
        match &self.0 {
            Ok(expr) => expr.prim_dtype(),
            Err(error) => Err(error.clone()),
        }
    }

    fn map(self, f: impl FnOnce(Expr) -> Result<Expr>) -> Ex {
        Ex(self.0.and_then(f))
    }
}

/// `value` as an [`Ex`], to start an expression from something that is not one yet.
pub fn ex(value: impl Into<Ex>) -> Ex {
    value.into()
}

impl<T: Into<Expr>> From<Result<T>> for Ex {
    fn from(value: Result<T>) -> Self {
        Ex(value.map(Into::into))
    }
}

impl From<&Ex> for Ex {
    fn from(value: &Ex) -> Self {
        value.clone()
    }
}

macro_rules! ex_from_expr_like {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl From<$ty> for Ex {
                fn from(value: $ty) -> Self {
                    Ex(Ok(value.into()))
                }
            }
            impl From<&$ty> for Ex {
                fn from(value: &$ty) -> Self {
                    Ex(Ok(value.clone().into()))
                }
            }
        )+
    };
}

// Any expression node converts, as any C++ `PrimExpr` subclass does.
ex_from_expr_like!(Expr, Var, IntImm, FloatImm, Call, TensorLoad);
ex_from_expr_like!(
    prim::Add,
    prim::And,
    prim::BitwiseAnd,
    prim::BitwiseNot,
    prim::BitwiseOr,
    prim::BitwiseXor,
    prim::Broadcast,
    prim::Cast,
    prim::Div,
    prim::EQ,
    prim::FloorDiv,
    prim::FloorMod,
    prim::GE,
    prim::GT,
    prim::LE,
    prim::LShift,
    prim::LT,
    prim::Let,
    prim::Max,
    prim::Min,
    prim::Mod,
    prim::Mul,
    prim::NE,
    prim::Not,
    prim::Or,
    prim::RShift,
    prim::Ramp,
    prim::Select,
    prim::Shuffle,
    prim::Sub,
);

// ---------------------------------------------------------------------------------------------
// Operands and literals

/// The other side of a binary operation: an expression, or a literal typed after the expression
/// it meets (C++ `TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD` and friends).
pub trait Operand {
    fn operand(self, peer: &Ex) -> Ex;
}

impl<T: Into<Ex>> Operand for T {
    fn operand(self, _peer: &Ex) -> Ex {
        self.into()
    }
}

/// C++ `int`: `MakeConst(peer.ty(), v)`. `i32` is what an unsuffixed literal falls back to.
impl Operand for i32 {
    fn operand(self, peer: &Ex) -> Ex {
        (self as i64).operand(peer)
    }
}

impl Operand for i64 {
    fn operand(self, peer: &Ex) -> Ex {
        Ex(peer.dtype().and_then(|dtype| make_const(dtype, self)))
    }
}

/// C++ `float`: a float32 `FloatImm`, whatever the peer.
impl Operand for f32 {
    fn operand(self, _peer: &Ex) -> Ex {
        float_imm(scalar_dtype(DLDataTypeCode::kDLFloat, 32), self as f64).into()
    }
}

/// C++ `double`: a float64 `FloatImm`, whatever the peer.
impl Operand for f64 {
    fn operand(self, _peer: &Ex) -> Ex {
        float_imm(scalar_dtype(DLDataTypeCode::kDLFloat, 64), self).into()
    }
}

// ---------------------------------------------------------------------------------------------
// The operations (`src/ir/prim/op.cc`)

/// The binary operations that match the operand types (`BinaryOpMatchTypes`), fold constants
/// ([`fold`]) and otherwise build the raw node.
macro_rules! folded_binary {
    ($($name:ident: $cpp:literal, $fold:path => $node:ident),+ $(,)?) => {
        $(
            #[doc = concat!("C++ `", $cpp, "(a, b, loc)`.")]
            fn $name(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
                let (a, b) = math::match_types_at(a, b, &loc)?;
                match $fold(&a, &b)? {
                    Some(folded) => Ok(folded),
                    None => Ok(prim::$node::new_full(a, b, loc)?.into()),
                }
            }
        )+
    };
}

folded_binary!(
    op_add: "add", fold::add => Add,
    op_sub: "sub", fold::sub => Sub,
    op_mul: "mul", fold::mul => Mul,
    op_div: "div", fold::div => Div,
    op_truncmod: "truncmod", fold::truncmod => Mod,
    op_gt: "greater", fold_gt => GT,
    op_ge: "greater_equal", fold_ge => GE,
    op_lt: "less", fold_lt => LT,
    op_le: "less_equal", fold_le => LE,
    op_ne: "not_equal", fold_ne => NE,
);

fn fold_gt(a: &Expr, b: &Expr) -> Result<Option<Expr>> {
    fold::compare(a, b, |x, y| x > y, |x, y| x > y)
}

fn fold_ge(a: &Expr, b: &Expr) -> Result<Option<Expr>> {
    fold::compare(a, b, |x, y| x >= y, |x, y| x >= y)
}

fn fold_lt(a: &Expr, b: &Expr) -> Result<Option<Expr>> {
    fold::compare(a, b, |x, y| x < y, |x, y| x < y)
}

fn fold_le(a: &Expr, b: &Expr) -> Result<Option<Expr>> {
    fold::compare(a, b, |x, y| x <= y, |x, y| x <= y)
}

fn fold_ne(a: &Expr, b: &Expr) -> Result<Option<Expr>> {
    fold::compare(a, b, |x, y| x != y, |x, y| x != y)
}

/// The `TVM_FFI_ICHECK` an operation makes of an operand's type: `codes` are the C++ type codes
/// the condition spells out, `message` what it reports with the operand and its type.
fn check_operand(
    operand: &Expr,
    side: &str,
    accepted: &[u8],
    codes: &str,
    message: impl FnOnce() -> String,
) -> Result<()> {
    if accepted.contains(&operand.prim_dtype()?.code) {
        return Ok(());
    }
    let condition = format!("{side}.ty().MatchesCode({codes})");
    Err(math::check_failed(INTERNAL_ERROR, &condition, &message()))
}

const INTEGER: [u8; 2] = [math::INT, math::UINT];
const INTEGER_CODES: &str = "DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt";
const INTEGER_OR_BOOL: [u8; 3] = [math::INT, math::UINT, math::BOOL];
const INTEGER_OR_BOOL_CODES: &str =
    "DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLBool";

/// `TVM_FFI_ICHECK(x.ty().MatchesCode(kDLInt, kDLUInt)) << x` of the integer divisions.
fn check_integer(a: &Expr, b: &Expr) -> Result<()> {
    check_operand(a, "a", &INTEGER, INTEGER_CODES, || repr(a))?;
    check_operand(b, "b", &INTEGER, INTEGER_CODES, || repr(b))
}

/// The operand checks of a bitwise operation `symbol` (`what` names it): integer operands, or
/// also booleans.
fn check_bitwise(
    a: &Expr,
    b: &Expr,
    symbol: &str,
    what: &str,
    (accepted, codes): (&[u8], &str),
) -> Result<()> {
    for (operand, side, which) in [(a, "a", "LHS"), (b, "b", "RHS")] {
        check_operand(operand, side, accepted, codes, || {
            format!(
                "Expected integer argument as {which} of {symbol} operator ({what}), but received \
                 {} of type {}",
                repr(operand),
                repr(&operand.ty)
            )
        })?;
    }
    Ok(())
}

/// The operand checks of `&&` and `||`: boolean operands.
fn check_logical(a: &Expr, b: &Expr, symbol: &str, what: &str) -> Result<()> {
    for (operand, side, which) in [(a, "a", "LHS"), (b, "b", "RHS")] {
        check_operand(operand, side, &[math::BOOL], "DLDataTypeCode::kDLBool", || {
            format!(
                "Expected boolean argument as {which} of {symbol} operator ({what}), but received \
                 {} of type {}",
                repr(operand),
                repr(&operand.ty)
            )
        })?;
    }
    Ok(())
}

/// C++ `truncdiv(a, b, loc)`: `div` of integers.
fn op_truncdiv(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    check_integer(&a, &b)?;
    op_div(a, b, loc)
}

/// C++ `floordiv(a, b, loc)`.
fn op_floordiv(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    check_integer(&a, &b)?;
    let (a, b) = math::match_types_at(a, b, &loc)?;
    match fold::floordiv(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::FloorDiv::new_full(a, b, loc)?.into()),
    }
}

/// C++ `floormod(a, b, loc)`.
fn op_floormod(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    check_integer(&a, &b)?;
    let (a, b) = math::match_types_at(a, b, &loc)?;
    match fold::floormod(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::FloorMod::new_full(a, b, loc)?.into()),
    }
}

/// C++ `indexdiv(a, b, loc)`, which is `floordiv`.
fn op_indexdiv(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    op_floordiv(a, b, loc)
}

/// C++ `indexmod(a, b, loc)`, which is `floormod`.
fn op_indexmod(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    op_floormod(a, b, loc)
}

/// C++ `ceildiv(a, b, loc)`: `floordiv(a + b - 1, b)`.
fn op_ceildiv(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    check_integer(&a, &b)?;
    let (a, b) = math::match_types_at(a, b, &loc)?;
    let sum = op_add(a, b.clone(), unknown_loc()?)?;
    let one = make_const(sum.prim_dtype()?, 1)?;
    let numerator = op_sub(sum, one, unknown_loc()?)?;
    match fold::floordiv(&numerator, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::FloorDiv::new_full(numerator, b, loc)?.into()),
    }
}

thread_local! {
    static INFINITIES: OnceCell<(Expr, Expr)> = const { OnceCell::new() };
}

/// `SymbolicLimits::pos_inf_` and `neg_inf_`, the opaque infinities of integer analysis, which
/// `min` and `max` recognize by identity.
fn infinities() -> Result<(Expr, Expr)> {
    INFINITIES.with(|cell| {
        if let Some(infinities) = cell.get() {
            return Ok(infinities.clone());
        }
        let infinities = (crate::sym::pos_inf()?, crate::sym::neg_inf()?);
        let _ = cell.set(infinities.clone());
        Ok(infinities)
    })
}

/// C++ `min(a, b, loc)`: an infinity on either side decides it.
fn op_min(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    let (pos_inf, neg_inf) = infinities()?;
    if a.same_as(&pos_inf) || b.same_as(&neg_inf) {
        return Ok(b);
    }
    if a.same_as(&neg_inf) || b.same_as(&pos_inf) {
        return Ok(a);
    }
    let (a, b) = math::match_types_at(a, b, &loc)?;
    match fold::min(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::Min::new_full(a, b, loc)?.into()),
    }
}

/// C++ `max(a, b, loc)`: an infinity on either side decides it.
fn op_max(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    let (pos_inf, neg_inf) = infinities()?;
    if a.same_as(&pos_inf) || b.same_as(&neg_inf) {
        return Ok(a);
    }
    if a.same_as(&neg_inf) || b.same_as(&pos_inf) {
        return Ok(b);
    }
    let (a, b) = math::match_types_at(a, b, &loc)?;
    match fold::max(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::Max::new_full(a, b, loc)?.into()),
    }
}

/// Whether `expr` is a call to `prim.vscale`.
fn is_vscale(expr: &Expr) -> bool {
    expr.as_node::<CallObj>().is_some_and(|call| VSCALE.is(&call.op))
}

/// C++ `equal(a, b, loc)`: two `vscale()` calls are equal, as the C++ `PrimExpr(true)`, an int32
/// one.
fn op_eq(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    let (a, b) = math::match_types_at(a, b, &loc)?;
    if let Some(folded) = fold::compare(&a, &b, |x, y| x == y, |x, y| x == y)? {
        return Ok(folded);
    }
    if is_vscale(&a) && is_vscale(&b) {
        return Ok(int32(1)?.into());
    }
    Ok(prim::EQ::new_full(a, b, loc)?.into())
}

/// C++ `logical_and(a, b, loc)`: boolean operands, not matched.
fn op_and(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    check_logical(&a, &b, "&&", "logical AND")?;
    match fold::and(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::And::new_full(a, b, loc)?.into()),
    }
}

/// C++ `logical_or(a, b, loc)`: boolean operands, not matched.
fn op_or(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    check_logical(&a, &b, "||", "logical OR")?;
    match fold::or(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::Or::new_full(a, b, loc)?.into()),
    }
}

/// The folding of a shift of index types: the amount is checked against the width, and two
/// constants fold, with a shift by zero the operand itself.
fn fold_shift(
    a: &Expr,
    b: &Expr,
    shift: impl FnOnce(&BigInt, &BigInt) -> Result<BigInt>,
    loc: &Location,
) -> Result<Option<Expr>> {
    let dtype = a.prim_dtype()?;
    if !fold::is_index_type(dtype) || !fold::is_index_type(b.prim_dtype()?) {
        return Ok(None);
    }
    let (pa, pb) = (int_value(a), int_value(b));
    if let Some(y) = pb {
        let in_range = !y.is_negative() && y < &BigInt::from(dtype.bits as i64);
        if !in_range {
            let condition = "pb->value >= 0 && pb->value < result_ty.bits()";
            let message = format!(
                "Shift amount must be non-negative and less than {} for type {}",
                dtype.bits,
                repr(&a.ty)
            );
            return Err(math::check_failed(INTERNAL_ERROR, condition, &message));
        }
    }
    if let (Some(x), Some(y)) = (pa, pb) {
        return Ok(Some(int_imm_at(dtype, shift(x, y)?, loc)?.into()));
    }
    if pb.is_some_and(BigInt::is_zero) {
        return Ok(Some(a.clone()));
    }
    Ok(None)
}

/// C++ `right_shift(a, b, loc)` (`>>`), sign-extending.
fn op_right_shift(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    check_bitwise(&a, &b, ">>", "right shift", (&INTEGER, INTEGER_CODES))?;
    let (a, b) = math::match_types_at(a, b, &loc)?;
    match fold_shift(&a, &b, |x, y| x.try_shr(y.clone()), &loc)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::RShift::new_full(a, b, loc)?.into()),
    }
}

/// C++ `left_shift(a, b, loc)` (`<<`): a 64-bit result wraps, a narrower one is checked by
/// `IntImm`.
fn op_left_shift(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
    check_bitwise(&a, &b, "<<", "left shift", (&INTEGER, INTEGER_CODES))?;
    let (a, b) = math::match_types_at(a, b, &loc)?;
    let dtype = a.prim_dtype()?;
    let shift = |x: &BigInt, y: &BigInt| {
        let value = x.try_shl(y.clone())?;
        Ok(if dtype.bits >= 64 { fold::fold_result(value, dtype) } else { value })
    };
    match fold_shift(&a, &b, shift, &loc)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::LShift::new_full(a, b, loc)?.into()),
    }
}

/// The bitwise operations, which fold two constants of index types.
macro_rules! bitwise {
    ($($name:ident: $cpp:literal, $symbol:literal, $what:literal, $op:tt => $node:ident),+ $(,)?) => {
        $(
            #[doc = concat!("C++ `", $cpp, "(a, b, loc)` (`", $symbol, "`).")]
            fn $name(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
                check_bitwise(&a, &b, $symbol, $what, (&INTEGER_OR_BOOL, INTEGER_OR_BOOL_CODES))?;
                let (a, b) = math::match_types_at(a, b, &loc)?;
                let dtype = a.prim_dtype()?;
                let index = fold::is_index_type(dtype) && fold::is_index_type(b.prim_dtype()?);
                if let (true, Some(x), Some(y)) = (index, int_value(&a), int_value(&b)) {
                    return Ok(int_imm_at(dtype, x $op y, &loc)?.into());
                }
                Ok(prim::$node::new_full(a, b, loc)?.into())
            }
        )+
    };
}

bitwise!(
    op_bitwise_and: "bitwise_and", "&", "bitwise AND", & => BitwiseAnd,
    op_bitwise_or: "bitwise_or", "|", "bitwise OR", | => BitwiseOr,
    op_bitwise_xor: "bitwise_xor", "^", "bitwise XOR", ^ => BitwiseXor,
);

/// C++ `bitwise_neg(a, loc)` (`~`): an integer or boolean operand, never folded.
fn op_bitwise_not(a: Expr, loc: Location) -> Result<Expr> {
    check_operand(&a, "a", &INTEGER_OR_BOOL, INTEGER_OR_BOOL_CODES, || {
        format!(
            "Expected integer or boolean argument for ~ operator (bitwise NOT), but received {} of \
             type {}",
            repr(&a),
            repr(&a.ty)
        )
    })?;
    Ok(prim::BitwiseNot::new_full(a, loc)?.into())
}

/// The value of an integer constant.
fn int_value(expr: &Expr) -> Option<&BigInt> {
    expr.as_node::<IntImmObj>().map(|imm| &imm.value)
}

type BinaryOp = fn(Expr, Expr, Location) -> Result<Expr>;
type UnaryOp = fn(Expr, Location) -> Result<Expr>;

/// `lhs <op> rhs`; a literal `rhs` is typed after `lhs`.
fn binary(lhs: Ex, rhs: impl Operand, op: BinaryOp) -> Ex {
    let rhs = rhs.operand(&lhs);
    let (a, b) = match (lhs.0, rhs.0) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(error), _) | (_, Err(error)) => return Ex(Err(error)),
    };
    Ex(unknown_loc().and_then(|loc| op(a, b, loc)))
}

/// `lhs <op> rhs` for a literal `lhs`, typed after `rhs`.
fn binary_literal(lhs: impl Operand, rhs: Ex, op: BinaryOp) -> Ex {
    let lhs = lhs.operand(&rhs);
    binary(lhs, rhs, op)
}

/// A function of [`math`] on one expression.
fn math_unary(value: Ex, f: fn(Expr, &Location) -> Result<Expr>) -> Ex {
    value.map(|a| f(a, &unknown_loc()?))
}

/// A function of [`math`] on two expressions; a literal `rhs` converts as C++ converts it
/// implicitly.
fn math_binary(value: Ex, rhs: impl IntoExpr, f: fn(Expr, Expr, &Location) -> Result<Expr>) -> Ex {
    let rhs = rhs.into_expr();
    value.map(|a| f(a, rhs?, &unknown_loc()?))
}

fn unary(value: Ex, op: UnaryOp) -> Ex {
    value.map(|a| op(a, unknown_loc()?))
}

/// C++ `neg` (unary `-`): a constant is negated in place (a signed 64-bit one wraps), anything
/// else becomes `MakeConst(a.ty(), 0) - a`.
fn neg(a: Expr) -> Result<Expr> {
    let dtype = a.prim_dtype()?;
    if let Some(imm) = a.as_node::<IntImmObj>() {
        let mut value = -&imm.value;
        if dtype.code == math::INT && dtype.bits >= 64 {
            value = fold::fold_result(value, dtype);
        }
        return Ok(big_int_imm(dtype, value)?.into());
    }
    if let Some(imm) = a.as_node::<FloatImmObj>() {
        return Ok(float_imm(dtype, -imm.value)?.into());
    }
    op_sub(make_const(dtype, 0)?, a, unknown_loc()?)
}

/// C++ `logical_not` (`!a`): a boolean operand, a constant folded, else the raw `Not`.
fn logical_not(a: Expr) -> Result<Expr> {
    check_operand(&a, "a", &[math::BOOL], "DLDataTypeCode::kDLBool", || {
        format!(
            "Expected boolean argument for ! operator (logical NOT), but received {} of type {}",
            repr(&a),
            repr(&a.ty)
        )
    })?;
    match fold::not(&a)? {
        Some(folded) => Ok(folded),
        None => Ok(prim::Not::new(a)?.into()),
    }
}

// ---------------------------------------------------------------------------------------------
// Operators

macro_rules! binary_operators {
    ($($Trait:ident::$method:ident => $op:ident),+ $(,)?) => {
        $(
            impl<R: Operand> $Trait<R> for Ex {
                type Output = Ex;
                fn $method(self, rhs: R) -> Ex {
                    binary(self, rhs, $op)
                }
            }
            binary_operators!(@lhs $Trait::$method => $op; &Ex, Expr, &Expr, Var, &Var);
            binary_operators!(@literal $Trait::$method => $op; i32, i64, f32, f64);
        )+
    };
    (@lhs $Trait:ident::$method:ident => $op:ident; $($lhs:ty),+) => {
        $(
            impl<R: Operand> $Trait<R> for $lhs {
                type Output = Ex;
                fn $method(self, rhs: R) -> Ex {
                    binary(self.into(), rhs, $op)
                }
            }
        )+
    };
    (@literal $Trait:ident::$method:ident => $op:ident; $($lit:ty),+) => {
        $(
            binary_operators!(@literal_rhs $Trait::$method => $op; $lit; Ex, &Ex, Expr, &Expr, Var, &Var);
        )+
    };
    (@literal_rhs $Trait:ident::$method:ident => $op:ident; $lit:ty; $($rhs:ty),+) => {
        $(
            impl $Trait<$rhs> for $lit {
                type Output = Ex;
                fn $method(self, rhs: $rhs) -> Ex {
                    binary_literal(self, rhs.into(), $op)
                }
            }
        )+
    };
}

binary_operators!(
    Add::add => op_add,
    Sub::sub => op_sub,
    Mul::mul => op_mul,
    BitAnd::bitand => op_bitwise_and,
    BitOr::bitor => op_bitwise_or,
    BitXor::bitxor => op_bitwise_xor,
    Shl::shl => op_left_shift,
    Shr::shr => op_right_shift,
);

macro_rules! assign_operators {
    ($($Trait:ident::$method:ident => $op:ident),+ $(,)?) => {
        $(
            impl<R: Operand> $Trait<R> for Ex {
                fn $method(&mut self, rhs: R) {
                    *self = binary(self.clone(), rhs, $op);
                }
            }
        )+
    };
}

assign_operators!(AddAssign::add_assign => op_add, SubAssign::sub_assign => op_sub, MulAssign::mul_assign => op_mul);

macro_rules! unary_operators {
    ($($ty:ty),+) => {
        $(
            /// C++ unary `-` (`neg`).
            impl Neg for $ty {
                type Output = Ex;
                fn neg(self) -> Ex {
                    Ex::from(self).map(neg)
                }
            }
            /// C++ `!` (`logical_not`).
            impl Not for $ty {
                type Output = Ex;
                fn not(self) -> Ex {
                    Ex::from(self).map(logical_not)
                }
            }
        )+
    };
}

unary_operators!(Ex, &Ex, Expr, &Expr, Var, &Var);

// ---------------------------------------------------------------------------------------------
// Named operations

/// [`PrimOps`] methods for unary intrinsics; `true` marks those that take only floats.
macro_rules! intrinsics {
    ($($name:ident => $op:literal, $float_only:literal;)+) => {
        $(
            #[doc = concat!("C++ `", stringify!($name), "(a)` (`", $op, "`).")]
            fn $name(&self) -> Ex {
                self.to_ex().map(|a| math::intrinsic($op, a, $float_only, &unknown_loc()?))
            }
        )+
    };
}

/// [`PrimOps`] methods for binary intrinsics, typed as the first argument.
macro_rules! binary_intrinsics {
    ($($name:ident => $op:literal;)+) => {
        $(
            #[doc = concat!("C++ `", stringify!($name), "(a, b)` (`", $op, "`).")]
            fn $name(&self, rhs: impl IntoExpr) -> Ex {
                let rhs = rhs.into_expr();
                self.to_ex().map(|a| math::binary_intrinsic($op, a, rhs?, &unknown_loc()?))
            }
        )+
    };
}

/// The C++ `PrimExpr` operations that are functions rather than operators. Each takes an
/// expression or a literal on the right, typed as for the operators, and leaves `self` usable.
pub trait PrimOps {
    /// This expression as an [`Ex`] (a reference-count increment).
    fn to_ex(&self) -> Ex;

    /// C++ `a < b` (`less`).
    fn less(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_lt)
    }
    /// C++ `a <= b` (`less_equal`).
    fn less_equal(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_le)
    }
    /// C++ `a > b` (`greater`).
    fn greater(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_gt)
    }
    /// C++ `a >= b` (`greater_equal`).
    fn greater_equal(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_ge)
    }
    /// C++ `a == b` (`equal`).
    fn equal(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_eq)
    }
    /// C++ `a != b` (`not_equal`).
    fn not_equal(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_ne)
    }
    /// C++ `a && b` (`logical_and`).
    fn logical_and(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_and)
    }
    /// C++ `a || b` (`logical_or`).
    fn logical_or(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_or)
    }
    /// C++ `div`: division for float types.
    fn div(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_div)
    }
    fn truncdiv(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_truncdiv)
    }
    fn truncmod(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_truncmod)
    }
    fn floordiv(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_floordiv)
    }
    fn floormod(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_floormod)
    }
    fn indexdiv(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_indexdiv)
    }
    fn indexmod(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_indexmod)
    }
    fn ceildiv(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_ceildiv)
    }
    fn min(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_min)
    }
    fn max(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_max)
    }
    /// C++ `~a`.
    fn bitwise_not(&self) -> Ex {
        unary(self.to_ex(), op_bitwise_not)
    }
    /// C++ `cast(a, t)`: `a` as type `t`. A value already of type `t` is itself, a constant is
    /// folded to `t`, a scalar cast to a vector type is broadcast, and a broadcast or an integer
    /// ramp is cast element by element; anything else is a `Cast` node, which
    /// [`Cast::new`](crate::prim::Cast::new) builds in every case.
    fn cast(&self, ty: impl IntoPrimType) -> Ex {
        let ty = ty.into_prim_type();
        self.to_ex().map(|a| math::cast(a, ty?, &unknown_loc()?))
    }
    /// C++ `tirx::reinterpret(t, a)`: the bits of `a` as type `t`, a `PrimType` of the same size
    /// or a pointer.
    fn reinterpret(&self, ty: impl Into<Type>) -> Ex {
        let ty = ty.into();
        self.to_ex().map(|a| math::reinterpret(ty, a, &unknown_loc()?))
    }

    // Math functions (`src/ir/prim/math.cc`), written in Rust in [`math`].

    /// C++ `floor(a)`: an integer is itself and a float constant is folded.
    fn floor(&self) -> Ex {
        math_unary(self.to_ex(), math::floor)
    }
    /// C++ `ceil(a)`, folded like `floor`.
    fn ceil(&self) -> Ex {
        math_unary(self.to_ex(), math::ceil)
    }
    /// C++ `round(a)`, folded like `floor`, a tie to even.
    fn round(&self) -> Ex {
        math_unary(self.to_ex(), math::round)
    }
    /// C++ `nearbyint(a)`, folded like `round`.
    fn nearbyint(&self) -> Ex {
        math_unary(self.to_ex(), math::nearbyint)
    }
    /// C++ `trunc(a)`, folded like `floor`.
    fn trunc(&self) -> Ex {
        math_unary(self.to_ex(), math::trunc)
    }
    /// C++ `abs(a)`: `select(a >= 0, a, -a)` for a signed integer, a call for a float, and an
    /// unsigned integer itself; a constant is folded.
    fn abs(&self) -> Ex {
        math_unary(self.to_ex(), math::abs)
    }
    /// C++ `isnan(a)`.
    fn isnan(&self) -> Ex {
        math_unary(self.to_ex(), math::isnan)
    }
    /// C++ `isinf(a)`: false for an integer, `abs(a) == inf && !isnan(a)` for a float.
    fn isinf(&self) -> Ex {
        math_unary(self.to_ex(), math::isinf)
    }
    /// C++ `isfinite(a)`: `!isinf(a) && !isnan(a)`.
    fn isfinite(&self) -> Ex {
        math_unary(self.to_ex(), math::isfinite)
    }
    /// C++ `likely(a)`: an integer constant is itself.
    fn likely(&self) -> Ex {
        math_unary(self.to_ex(), math::likely)
    }
    /// C++ `assume(a)`.
    fn assume(&self) -> Ex {
        math_unary(self.to_ex(), math::assume)
    }
    /// C++ `pow(a, b)`: both sides matched, a float result. A literal `b` converts as in C++
    /// (`PrimExpr(int)`).
    fn pow(&self, rhs: impl IntoExpr) -> Ex {
        math_binary(self.to_ex(), rhs, math::pow)
    }
    /// C++ `fmod(a, b)`, matched like `pow`.
    fn fmod(&self, rhs: impl IntoExpr) -> Ex {
        math_binary(self.to_ex(), rhs, math::fmod)
    }
    /// C++ `tirx::logaddexp(a, b)`: `log(exp(a) + exp(b))` of two matched floats.
    fn logaddexp(&self, rhs: impl IntoExpr) -> Ex {
        math_binary(self.to_ex(), rhs, math::logaddexp)
    }

    // The intrinsics (`TVM_DECLARE_INTRIN_UNARY` and friends): a call typed as the argument, a
    // bfloat16 argument computed in float32.
    intrinsics! {
        exp => "prim.exp", false;
        exp2 => "prim.exp2", false;
        exp10 => "prim.exp10", false;
        erf => "prim.erf", false;
        sigmoid => "prim.sigmoid", false;
        sqrt => "prim.sqrt", false;
        rsqrt => "prim.rsqrt", false;
        log => "prim.log", false;
        log2 => "prim.log2", false;
        log10 => "prim.log10", false;
        log1p => "prim.log1p", false;
        popcount => "prim.popcount", false;
        tanh => "prim.tanh", true;
        tan => "prim.tan", true;
        cos => "prim.cos", true;
        cosh => "prim.cosh", true;
        sin => "prim.sin", true;
        sinh => "prim.sinh", true;
        asin => "prim.asin", true;
        acos => "prim.acos", true;
        atan => "prim.atan", true;
        acosh => "prim.acosh", true;
        asinh => "prim.asinh", true;
        atanh => "prim.atanh", true;
    }
    binary_intrinsics! {
        atan2 => "prim.atan2";
        nextafter => "prim.nextafter";
        copysign => "prim.copysign";
        hypot => "prim.hypot";
        ldexp => "prim.ldexp";
    }
}

impl PrimOps for Ex {
    fn to_ex(&self) -> Ex {
        self.clone()
    }
}

impl PrimOps for Expr {
    fn to_ex(&self) -> Ex {
        self.into()
    }
}

impl PrimOps for Var {
    fn to_ex(&self) -> Ex {
        self.into()
    }
}

/// C++ `if_then_else(condition, t, f)`: a boolean condition and matched values, folded when the
/// condition is constant, else a `tirx.if_then_else` call typed as the values.
pub fn if_then_else(condition: impl Into<Ex>, t: impl Into<Ex>, f: impl Into<Ex>) -> Ex {
    let (c, t, f) = (condition.into(), t.into(), f.into());
    match (c.0, t.0, f.0) {
        (Ok(c), Ok(t), Ok(f)) => Ex(unknown_loc().and_then(|loc| if_then_else_at(c, t, f, loc))),
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => Ex(Err(error)),
    }
}

fn if_then_else_at(condition: Expr, t: Expr, f: Expr, loc: Location) -> Result<Expr> {
    if condition.prim_dtype()?.code != math::BOOL {
        let check = "cond.ty().MatchesCode(DLDataTypeCode::kDLBool)";
        let message = "if_then_else only accept the condition to be boolean type.";
        return Err(math::check_failed(INTERNAL_ERROR, check, message));
    }
    let (t, f) = math::match_types_at(t, f, &loc)?;
    if let Some(value) = int_value(&condition) {
        return Ok(if value.is_zero() { f } else { t });
    }
    let ty = t.ty.clone();
    let call = Call::new_full(
        Some(ty),
        IF_THEN_ELSE.get()?.into(),
        [condition, t, f].into_iter().collect(),
        None,
        tvm_ffi::Array::new(vec![]),
        loc,
    )?;
    Ok(call.into())
}

/// The raw `Select(condition, t, f)` (C++ `prim::Select`), from operands under construction.
pub fn select(condition: impl Into<Ex>, t: impl Into<Ex>, f: impl Into<Ex>) -> Ex {
    let (c, t, f) = (condition.into(), t.into(), f.into());
    match (c.0, t.0, f.0) {
        (Ok(c), Ok(t), Ok(f)) => Ex(crate::prim::Select::new(c, t, f).map(Into::into)),
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => Ex(Err(error)),
    }
}

/// The raw `Let(var, value, body)` (C++ `prim::Let`), from operands under construction.
pub fn let_(var: &Var, value: impl Into<Ex>, body: impl Into<Ex>) -> Ex {
    match (value.into().0, body.into().0) {
        (Ok(value), Ok(body)) => {
            Ex(crate::prim::Let::new(var.clone(), value, body).map(Into::into))
        }
        (Err(error), _) | (_, Err(error)) => Ex(Err(error)),
    }
}
