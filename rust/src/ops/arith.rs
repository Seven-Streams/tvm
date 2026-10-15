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

//! The operators: `Ex`, literals, the native operations and `PrimOps`.

use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
};

use tvm_ffi::{BigInt, DLDataType, DLDataTypeCode, ObjectRefCore, Result};

use super::build::{
    big_int_imm, float_imm, int_imm, make_const, prim_dtype, scalar_dtype, unknown_loc, IntoExpr,
    IntoPrimType,
};
use super::math;
use crate::ir::{Call, Expr, FloatImm, FloatImmObj, IntImm, IntImmObj, Location, TensorLoad, Type, Var};
use crate::prim;

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

native_unary!(op_bitwise_not => "prim.bitwise_not");

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
pub(super) fn wrap_signed(value: BigInt, bits: u8) -> Result<BigInt> {
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
        None => Ok(crate::prim::Not::new(a)?.into()),
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
