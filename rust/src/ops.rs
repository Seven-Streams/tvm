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

//! Expression arithmetic with the semantics of the C++ `PrimExpr` operators
//! (`include/tvm/ir/prim/op.h`).
//!
//! `a + b` calls the global behind C++ `a + b` (`prim._OpAdd`), so operand type matching and
//! constant folding are the native ones. A literal on either side is typed the way C++ types it:
//! an integer becomes `MakeConst(other.ty(), v)`, an `f32` a float32 `FloatImm`, an `f64` a
//! float64 `FloatImm`, a `bool` a bool `IntImm`. As in C++, `/` and `%` are not defined: name the
//! division (`truncdiv`, `floordiv`, `div`, ...). Comparisons and `&&` / `||` are [`PrimOps`]
//! methods named after the C++ functions, because Rust's comparison traits return `bool`.
//!
//! Operators build an [`Ex`], which keeps the first error instead of panicking; `.expr()` turns
//! it back into a `Result`:
//!
//! ```ignore
//! select(rmod.greater_equal(0), &rdiv, &rdiv - 1).expr()?
//! ```
//!
//! The raw constructors (C++ `Add(a, b)`: type checks, no folding) are `prim::Add::new` and so on.

use std::cell::OnceCell;
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
};

use tvm_ffi::{BigInt, DLDataType, DLDataTypeCode, ObjectRefCore, Result};

use crate::ir::{
    Call, Expr, FloatImm, FloatImmObj, IntImm, IntImmObj, Location, PrimType, PrimTypeObj,
    UnknownLoc, Var,
};

// ---------------------------------------------------------------------------------------------
// Constants and types

thread_local! {
    static UNKNOWN_LOC: OnceCell<Location> = const { OnceCell::new() };
}

/// The process-wide `UnknownLoc()` singleton, as a `Location`, fetched once per thread.
pub fn unknown_loc() -> Result<Location> {
    UNKNOWN_LOC.with(|cell| {
        if let Some(loc) = cell.get() {
            return Ok(loc.clone());
        }
        let loc: Location = UnknownLoc::new()?.into();
        let _ = cell.set(loc.clone());
        Ok(loc)
    })
}

pub fn int32_dtype() -> DLDataType {
    DLDataType { code: DLDataTypeCode::kDLInt as u8, bits: 32, lanes: 1 }
}

fn scalar_dtype(code: DLDataTypeCode, bits: u8) -> DLDataType {
    DLDataType { code: code as u8, bits, lanes: 1 }
}

/// `PrimType(dtype)` through the native constructor.
pub fn prim_type(dtype: DLDataType) -> Result<PrimType> {
    tvm_ffi::cached_global_func!("ir.PrimType").call_tuple((dtype,))?.try_into()
}

/// The dtype of a PrimExpr (`expr.ty()`); an error for any other expression.
pub fn prim_dtype(expr: &Expr) -> Result<DLDataType> {
    match expr.ty.as_node::<PrimTypeObj>() {
        Some(ty) => Ok(ty.dtype),
        None => Err(tvm_ffi::Error::new(tvm_ffi::TYPE_ERROR, "expected a PrimType expression", "")),
    }
}

/// `IntImm::Int32(value)`, the C++ meaning of a bare integer argument such as `Evaluate(0)`.
pub fn int32(value: i64) -> Result<IntImm> {
    int_imm(int32_dtype(), value as i128)
}

/// `IntImm(dtype, value)` through the native constructor.
pub fn int_imm(dtype: DLDataType, value: i128) -> Result<IntImm> {
    big_int_imm(dtype, BigInt::from(value))
}

fn big_int_imm(dtype: DLDataType, value: BigInt) -> Result<IntImm> {
    tvm_ffi::cached_global_func!("ir.IntImm").call_tuple((dtype, value, unknown_loc()?))?.try_into()
}

/// `FloatImm(dtype, value)` through the native constructor.
pub fn float_imm(dtype: DLDataType, value: f64) -> Result<FloatImm> {
    tvm_ffi::cached_global_func!("ir.FloatImm")
        .call_tuple((dtype, value, unknown_loc()?))?
        .try_into()
}

/// `prim::MakeConst(dtype, value)`, which has no global: an `IntImm` or `FloatImm` by type code,
/// broadcast for a fixed-length vector type. An integer literal next to an expression is typed so.
pub fn make_const(dtype: DLDataType, value: i64) -> Result<Expr> {
    let scalar = DLDataType { lanes: 1, ..dtype };
    let element: Expr = match dtype.code {
        // kDLInt, kDLUInt, kDLBool
        0 | 1 | 6 => int_imm(scalar, value as i128)?.into(),
        // kDLFloat, kDLBfloat, and the float8/6/4 codes
        2 | 4 | 7..=17 => float_imm(scalar, value as f64)?.into(),
        _ => {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::INTERNAL_ERROR,
                "cannot make const for this type",
                "",
            ))
        }
    };
    if dtype.lanes == 1 {
        return Ok(element);
    }
    let lanes: Expr = int32(dtype.lanes as i64)?.into();
    Ok(crate::prim::Broadcast::new(element, lanes, unknown_loc()?)?.into())
}

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
            Ok(expr) => prim_dtype(expr),
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

ex_from_expr_like!(Expr, Var, IntImm, FloatImm, Call);

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

/// C++ `bool`: a bool `IntImm`.
impl Operand for bool {
    fn operand(self, _peer: &Ex) -> Ex {
        int_imm(scalar_dtype(DLDataTypeCode::kDLBool, 8), self as i128).into()
    }
}

// ---------------------------------------------------------------------------------------------
// The native operations

macro_rules! native_binary {
    ($($name:ident => $global:literal),+ $(,)?) => {
        $(
            fn $name(a: Expr, b: Expr, loc: Location) -> Result<Expr> {
                tvm_ffi::cached_global_func!($global).call_tuple((a, b, loc))?.try_into()
            }
        )+
    };
}

native_binary!(
    op_add => "prim._OpAdd",
    op_sub => "prim._OpSub",
    op_mul => "prim._OpMul",
    op_div => "prim._OpDiv",
    op_truncdiv => "prim._OpTruncDiv",
    op_truncmod => "prim._OpTruncMod",
    op_floordiv => "prim._OpFloorDiv",
    op_floormod => "prim._OpFloorMod",
    op_indexdiv => "prim._OpIndexDiv",
    op_indexmod => "prim._OpIndexMod",
    op_ceildiv => "prim._OpCeilDiv",
    op_pow => "prim._OpPow",
    op_min => "prim._OpMin",
    op_max => "prim._OpMax",
    op_eq => "prim._OpEQ",
    op_ne => "prim._OpNE",
    op_lt => "prim._OpLT",
    op_le => "prim._OpLE",
    op_gt => "prim._OpGT",
    op_ge => "prim._OpGE",
    op_and => "prim._OpAnd",
    op_or => "prim._OpOr",
    op_bitwise_and => "prim.bitwise_and",
    op_bitwise_or => "prim.bitwise_or",
    op_bitwise_xor => "prim.bitwise_xor",
    op_left_shift => "prim.left_shift",
    op_right_shift => "prim.right_shift",
);

macro_rules! native_unary {
    ($($name:ident => $global:literal),+ $(,)?) => {
        $(
            fn $name(a: Expr, loc: Location) -> Result<Expr> {
                tvm_ffi::cached_global_func!($global).call_tuple((a, loc))?.try_into()
            }
        )+
    };
}

native_unary!(
    op_abs => "prim.abs",
    op_floor => "prim.floor",
    op_likely => "prim.likely",
    op_bitwise_not => "prim.bitwise_not",
);

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

fn unary(value: Ex, op: UnaryOp) -> Ex {
    value.map(|a| op(a, unknown_loc()?))
}

/// C++ `neg` (unary `-`), which has no global: a constant is negated in place, anything else
/// becomes `MakeConst(a.ty(), 0) - a`.
fn neg(a: Expr) -> Result<Expr> {
    let dtype = prim_dtype(&a)?;
    if let Some(imm) = a.as_node::<IntImmObj>() {
        let mut value = -&imm.value;
        if dtype.code == DLDataTypeCode::kDLInt as u8 && dtype.bits >= 64 {
            value = wrap_signed(value, dtype.bits)?;
        }
        return Ok(big_int_imm(dtype, value)?.into());
    }
    if let Some(imm) = a.as_node::<FloatImmObj>() {
        return Ok(float_imm(dtype, -imm.value)?.into());
    }
    op_sub(make_const(dtype, 0)?, a, unknown_loc()?)
}

/// `GetFoldResult` for a signed type of `bits >= 64`: the value modulo `2^bits`, as signed.
fn wrap_signed(value: BigInt, bits: u8) -> Result<BigInt> {
    if value.to_i64().is_some() {
        return Ok(value);
    }
    let Some(value) = value.to_i128().filter(|_| bits < 128) else {
        return Err(tvm_ffi::Error::new(
            tvm_ffi::INTERNAL_ERROR,
            "negating a constant wider than 127 bits is not supported",
            "",
        ));
    };
    let modulus = 1i128 << bits;
    let wrapped = value.rem_euclid(modulus);
    Ok(BigInt::from(if wrapped >= modulus >> 1 { wrapped - modulus } else { wrapped }))
}

/// C++ `logical_not` (`!a`), which has no global: folds a constant, else the raw `Not`.
fn logical_not(a: Expr) -> Result<Expr> {
    match a.as_node::<IntImmObj>() {
        Some(imm) => Ok(int_imm(prim_dtype(&a)?, imm.value.is_zero() as i128)?.into()),
        None => Ok(crate::prim::Not::new(a, unknown_loc()?)?.into()),
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
    fn pow(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_pow)
    }
    fn min(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_min)
    }
    fn max(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, op_max)
    }
    fn abs(&self) -> Ex {
        unary(self.to_ex(), op_abs)
    }
    fn floor(&self) -> Ex {
        unary(self.to_ex(), op_floor)
    }
    fn likely(&self) -> Ex {
        unary(self.to_ex(), op_likely)
    }
    /// C++ `~a`.
    fn bitwise_not(&self) -> Ex {
        unary(self.to_ex(), op_bitwise_not)
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

/// C++ `if_then_else(condition, t, f)`, folded when the condition is constant.
pub fn if_then_else(condition: impl Into<Ex>, t: impl Into<Ex>, f: impl Into<Ex>) -> Ex {
    let (c, t, f) = (condition.into(), t.into(), f.into());
    match (c.0, t.0, f.0) {
        (Ok(c), Ok(t), Ok(f)) => Ex(unknown_loc().and_then(|loc| {
            tvm_ffi::cached_global_func!("prim._OpIfThenElse")
                .call_tuple((c, t, f, loc))?
                .try_into()
        })),
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => Ex(Err(error)),
    }
}

/// The raw `Select(condition, t, f)` (C++ `prim::Select`), from operands under construction.
pub fn select(condition: impl Into<Ex>, t: impl Into<Ex>, f: impl Into<Ex>) -> Ex {
    let (c, t, f) = (condition.into(), t.into(), f.into());
    match (c.0, t.0, f.0) {
        (Ok(c), Ok(t), Ok(f)) => {
            Ex(unknown_loc().and_then(|loc| Ok(crate::prim::Select::new(c, t, f, loc)?.into())))
        }
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => Ex(Err(error)),
    }
}

/// The raw `Let(var, value, body)` (C++ `prim::Let`), from operands under construction.
pub fn let_(var: &Var, value: impl Into<Ex>, body: impl Into<Ex>) -> Ex {
    match (value.into().0, body.into().0) {
        (Ok(value), Ok(body)) => Ex(unknown_loc()
            .and_then(|loc| Ok(crate::prim::Let::new(var.clone(), value, body, loc)?.into()))),
        (Err(error), _) | (_, Err(error)) => Ex(Err(error)),
    }
}
