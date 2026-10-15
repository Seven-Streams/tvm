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

//! The math functions on `PrimExpr` (`include/tvm/ir/prim/op.h`, `src/ir/prim/math.cc`,
//! `src/tirx/op/math.cc`, `src/tirx/op/memory.cc`), and the `MakeConst`, `cast` and
//! `BinaryOpMatchTypes` they build on, written in Rust.
//!
//! Each follows its C++ definition: what it folds, what it checks, the node it builds and the
//! error it raises. Like the C++ functions, they make nodes with the native constructors and
//! combine expressions with the operators of [`super::arith`].

use std::cell::RefCell;
use std::collections::HashMap;

use tvm_ffi::{
    AnyView, Array, BigInt, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, ErrorKind,
    ObjectRefCast, ObjectRefCore, Result, INTERNAL_ERROR, TYPE_ERROR,
};

use super::arith::{wrap_signed, Ex, PrimOps};
use super::build::{
    float_imm, float_imm_at, int32, int32_dtype, int_imm_at, prim_dtype, prim_type, IntoExpr,
    IntoPrimType,
};
use crate::ir::{
    Call, Expr, FloatImmObj, IntImmObj, Location, Op, PrimType, PrimTypeObj, PtrTypeObj,
    StringImmObj, Type,
};
use crate::prim::{Broadcast, BroadcastObj, Cast, Mul, Ramp, RampObj, Select};

pub(super) const INT: u8 = DLDataTypeCode::kDLInt as u8;
pub(super) const UINT: u8 = DLDataTypeCode::kDLUInt as u8;
const FLOAT: u8 = DLDataTypeCode::kDLFloat as u8;
pub(super) const HANDLE: u8 = DLDataTypeCode::kDLOpaqueHandle as u8;
const BFLOAT: u8 = DLDataTypeCode::kDLBfloat as u8;
pub(super) const BOOL: u8 = DLDataTypeCode::kDLBool as u8;
const FLOAT8: std::ops::RangeInclusive<u8> =
    DLDataTypeCode::kDLFloat8_e3m4 as u8..=DLDataTypeCode::kDLFloat8_e8m0fnu as u8;
const FLOAT6: std::ops::RangeInclusive<u8> =
    DLDataTypeCode::kDLFloat6_e2m3fn as u8..=DLDataTypeCode::kDLFloat6_e3m2fn as u8;
const FLOAT4: u8 = DLDataTypeCode::kDLFloat4_e2m1fn as u8;

// ---------------------------------------------------------------------------------------------
// Types

/// The lanes as C++ `PrimType` reads them: a scalable vector stores minus its vscale factor.
fn encoded_lanes(dtype: DLDataType) -> i16 {
    dtype.lanes as i16
}

fn is_scalar(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) == 1
}

fn is_scalable(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) < -1
}

fn is_fixed_vector(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) > 1
}

/// The void sentinel, `handle(0, 0)`.
fn is_void(dtype: DLDataType) -> bool {
    dtype.code == HANDLE && dtype.bits == 0 && encoded_lanes(dtype) == 0
}

fn vscale_factor(dtype: DLDataType) -> i32 {
    -(encoded_lanes(dtype) as i32)
}

/// C++ `PrimType::lanes()`, which a scalable vector has none of at compile time.
pub(super) fn lanes(dtype: DLDataType) -> Result<i32> {
    match encoded_lanes(dtype) {
        lanes if lanes < 0 => {
            Err(internal("Can't fetch the lanes of a scalable vector at a compile time."))
        }
        lanes => Ok(lanes as i32),
    }
}

/// The scalar element type, `WithLanes(1)`.
pub(super) fn element(dtype: DLDataType) -> DLDataType {
    DLDataType { lanes: 1, ..dtype }
}

fn is_bfloat16(dtype: DLDataType) -> bool {
    dtype.code == BFLOAT && dtype.bits == 16
}

fn storage_bytes(dtype: DLDataType) -> u64 {
    (dtype.bits as u64 * dtype.lanes as u64).div_ceil(8)
}

// ---------------------------------------------------------------------------------------------
// Shared pieces

pub(super) fn internal(message: &str) -> Error {
    Error::new(INTERNAL_ERROR, message, "")
}

/// The error of a failed C++ `TVM_FFI_CHECK` or `TVM_FFI_ICHECK`.
pub(super) fn check_failed(kind: ErrorKind, condition: &str, message: &str) -> Error {
    Error::new(kind, &format!("Check failed: ({condition}) is false: {message}"), "")
}

/// How C++ prints a node in an error message (`ffi.ReprPrint`), such as `T.float32`.
fn repr<T>(value: &T) -> String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    tvm_ffi::cached_global_func!("ffi.ReprPrint")
        .call_packed(&[AnyView::from(value)])
        .and_then(tvm_ffi::String::try_from)
        .map(|text| text.as_str().to_string())
        .unwrap_or_default()
}

thread_local! {
    static OPS: RefCell<HashMap<&'static str, Op>> = RefCell::new(HashMap::new());
}

/// `Op::Get(name)`, fetched once per thread as the C++ functions keep theirs in `static`s.
pub(super) fn op(name: &'static str) -> Result<Op> {
    if let Some(op) = OPS.with(|ops| ops.borrow().get(name).cloned()) {
        return Ok(op);
    }
    let op = Op::get(name)?;
    OPS.with(|ops| ops.borrow_mut().insert(name, op.clone()));
    Ok(op)
}

/// `Call(ty, Op::Get(name), args, {}, {}, loc)`; without `ty` the constructor infers it.
fn call(ty: Option<Type>, name: &'static str, args: Vec<Expr>, loc: &Location) -> Result<Expr> {
    let (op, args) = (op(name)?.into(), Array::new(args));
    Ok(Call::new_full(ty, op, args, None, Array::new(vec![]), loc.clone())?.into())
}

/// `Call(PrimType::Int(32), prim::vscale_op(), {})`.
fn vscale() -> Result<Expr> {
    call(Some(prim_type(int32_dtype())?.into()), "prim.vscale", vec![], &super::unknown_loc()?)
}

/// `x.ty()` of a `PrimExpr`.
fn prim_ty(x: &Expr) -> Result<PrimType> {
    prim_dtype(x)?;
    Ok(x.ty.clone().try_cast().expect("checked to be a PrimType"))
}

// ---------------------------------------------------------------------------------------------
// MakeConst and cast

/// The value of a constant, as `MakeConst` takes it.
#[derive(Clone, Debug)]
pub(crate) enum Value {
    Int(BigInt),
    Float(f64),
}

impl Value {
    /// C++ `BigInt(value)`.
    fn to_big_int(&self) -> Result<BigInt> {
        match self {
            Value::Int(value) => Ok(value.clone()),
            Value::Float(value) => BigInt::try_from(*value),
        }
    }

    /// C++ `static_cast<double>(value)`.
    fn to_f64(&self) -> Result<f64> {
        match self {
            Value::Int(value) => value.to_f64(),
            Value::Float(value) => Ok(*value),
        }
    }

    fn is_nonnegative(&self) -> bool {
        match self {
            Value::Int(value) => !value.is_negative(),
            Value::Float(value) => *value >= 0.0,
        }
    }

    fn describe(&self) -> String {
        match self {
            Value::Int(value) => value.to_string(),
            Value::Float(value) => value.to_string(),
        }
    }
}

/// C++ `MakeConstScalar(ty, value, loc)`.
fn make_const_scalar(dtype: DLDataType, value: &Value, loc: &Location) -> Result<Expr> {
    match dtype.code {
        INT | BOOL => Ok(int_imm_at(dtype, value.to_big_int()?, loc)?.into()),
        UINT if !value.is_nonnegative() => {
            let message = format!("cannot make uint from negative value {}", value.describe());
            Err(check_failed(INTERNAL_ERROR, "value >= 0", &message))
        }
        UINT => Ok(int_imm_at(dtype, value.to_big_int()?, loc)?.into()),
        code if code == FLOAT
            || code == FLOAT4
            || FLOAT8.contains(&code)
            || FLOAT6.contains(&code) =>
        {
            Ok(float_imm_at(dtype, value.to_f64()?, loc)?.into())
        }
        _ if is_bfloat16(dtype) => Ok(float_imm_at(dtype, value.to_f64()?, loc)?.into()),
        _ => Err(internal(&format!("cannot make const for type {}", repr(&prim_type(dtype)?)))),
    }
}

/// C++ `MakeConst(ty, value, loc)`: the scalar, broadcast for a vector type.
pub(crate) fn make_const(dtype: DLDataType, value: Value, loc: &Location) -> Result<Expr> {
    if !is_scalable(dtype) && !is_fixed_vector(dtype) {
        return make_const_scalar(dtype, &value, loc);
    }
    let scalar = make_const_scalar(element(dtype), &value, loc)?;
    let lanes: Expr = match is_fixed_vector(dtype) {
        true => int32(encoded_lanes(dtype) as i64)?.into(),
        false => Mul::new(vscale()?, vscale_factor(dtype))?.into(),
    };
    Ok(Broadcast::new_full(scalar, lanes, loc.clone())?.into())
}

/// A constant of another type folded to `ty`, keeping its location; anything else a `Cast`.
fn fold_or_cast(value: Expr, ty: PrimType, loc: &Location) -> Result<Expr> {
    if let Some(imm) = value.as_node::<IntImmObj>() {
        return make_const(ty.dtype, Value::Int(imm.value.clone()), &imm.loc);
    }
    if let Some(imm) = value.as_node::<FloatImmObj>() {
        return make_const(ty.dtype, Value::Float(imm.value), &imm.loc);
    }
    Ok(Cast::new_full(value, ty, loc.clone())?.into())
}

/// C++ `cast(value, t, loc)` (`src/ir/prim/op.cc`): a value already of type `t` is itself, a
/// constant is folded, a scalar cast to a vector type is broadcast, and a broadcast or an
/// integer ramp is cast element by element; anything else is a `Cast` node.
pub(crate) fn cast(value: Expr, ty: PrimType, loc: &Location) -> Result<Expr> {
    let (from, to) = (prim_dtype(&value)?, ty.dtype);
    if from == to {
        return Ok(value);
    }
    if is_void(from) {
        let message = "Cannot cast an expression with the void sentinel type";
        return Err(check_failed(TYPE_ERROR, "!value.ty().IsVoid()", message));
    }
    if is_scalar(to) {
        return fold_or_cast(value, ty, loc);
    }
    let element_ty = prim_type(element(to))?;
    if !is_scalable(from) && !is_fixed_vector(from) {
        let value = match from == element(to) {
            true => value,
            false => fold_or_cast(value, element_ty, loc)?,
        };
        let lanes: Expr = match is_scalable(to) {
            true => Mul::new(vscale_factor(to), vscale()?)?.into(),
            false => int32(encoded_lanes(to) as i64)?.into(),
        };
        return Ok(Broadcast::new_full(value, lanes, loc.clone())?.into());
    }
    if is_scalable(from) != is_scalable(to) {
        let condition = "value.ty().IsScalableVector() == dtype.IsScalableVector()";
        return Err(check_failed(INTERNAL_ERROR, condition, ""));
    }
    let lanes_match = match is_scalable(from) {
        true => vscale_factor(from) == vscale_factor(to),
        false => encoded_lanes(from) == encoded_lanes(to),
    };
    if !lanes_match {
        return Err(check_failed(INTERNAL_ERROR, "lanes_match", ""));
    }
    if let Some(broadcast) = value.as_node::<BroadcastObj>() {
        let element = cast(broadcast.value.clone(), element_ty, loc)?;
        return Ok(Broadcast::new_full(element, broadcast.lanes.clone(), loc.clone())?.into());
    }
    if let Some(ramp) = value.as_node::<RampObj>().filter(|_| matches!(to.code, INT | UINT)) {
        let base = cast(ramp.base_.clone(), element_ty.clone(), loc)?;
        let stride = cast(ramp.stride.clone(), element_ty, loc)?;
        return Ok(Ramp::new_full(base, stride, ramp.lanes.clone(), loc.clone())?.into());
    }
    Ok(Cast::new_full(value, ty, loc.clone())?.into())
}

// ---------------------------------------------------------------------------------------------
// BinaryOpMatchTypes

/// C++ `BroadcastToMatchLanes`: a scalar next to a vector of type `other` is broadcast to its
/// lanes.
fn broadcast_to_lanes(value: Expr, other: DLDataType) -> Result<Expr> {
    let dtype = prim_dtype(&value)?;
    if is_scalable(dtype)
        || is_fixed_vector(dtype)
        || !(is_scalable(other) || is_fixed_vector(other))
    {
        return Ok(value);
    }
    let lanes: Expr = match is_scalable(other) {
        true => Mul::new(vscale_factor(other), vscale()?)?.into(),
        false => int32(encoded_lanes(other) as i64)?.into(),
    };
    Ok(Broadcast::new(value, lanes)?.into())
}

/// C++ `PromoteBinaryOpType`: the type both sides of a binary operation take.
fn promote(lhs: DLDataType, rhs: DLDataType) -> Result<DLDataType> {
    if lhs == rhs {
        return Ok(lhs);
    }
    let wider = |lhs: DLDataType, rhs: DLDataType| if lhs.bits < rhs.bits { rhs } else { lhs };
    // The narrow float families take precedence over every other type, in this order.
    let families: [fn(DLDataType) -> bool; 5] = [
        |t| t.code == FLOAT,
        |t| t.code == BFLOAT,
        |t| FLOAT8.contains(&t.code),
        |t| FLOAT6.contains(&t.code),
        |t| t.code == FLOAT4,
    ];
    if lhs.code == FLOAT && rhs.code == FLOAT {
        return Ok(wider(lhs, rhs));
    }
    for family in families {
        match (family(lhs), family(rhs)) {
            (false, true) => return Ok(rhs),
            (true, false) => return Ok(lhs),
            _ => {}
        }
    }
    let integer = |t: DLDataType| matches!(t.code, INT | UINT);
    Ok(match (lhs.code, rhs.code) {
        (BOOL, _) if integer(rhs) => rhs,
        (_, BOOL) if integer(lhs) => lhs,
        (INT, INT) | (UINT, UINT) => wider(lhs, rhs),
        (INT, UINT) | (UINT, INT) if lhs.bits != rhs.bits => wider(lhs, rhs),
        (INT, UINT) | (UINT, INT) => DLDataType { code: UINT, ..lhs },
        _ => {
            let (lhs, rhs) = (lhs.to_string(), rhs.to_string());
            return Err(internal(&format!(
                "Cannot match type {} vs {}",
                lhs.as_str(),
                rhs.as_str()
            )));
        }
    })
}

/// C++ `BinaryOpMatchTypes(lhs, rhs)`: a scalar next to a vector is broadcast to its lanes, then
/// both sides are cast to their promoted type.
pub fn match_types(lhs: impl IntoExpr, rhs: impl IntoExpr) -> Result<(Expr, Expr)> {
    match_types_at(lhs.into_expr()?, rhs.into_expr()?, &super::unknown_loc()?)
}

pub(crate) fn match_types_at(lhs: Expr, rhs: Expr, loc: &Location) -> Result<(Expr, Expr)> {
    if prim_dtype(&lhs)? == prim_dtype(&rhs)? {
        return Ok((lhs, rhs));
    }
    let lhs = broadcast_to_lanes(lhs, prim_dtype(&rhs)?)?;
    let rhs = broadcast_to_lanes(rhs, prim_dtype(&lhs)?)?;
    let (lhs_ty, rhs_ty) = (prim_dtype(&lhs)?, prim_dtype(&rhs)?);
    if is_scalable(lhs_ty) != is_scalable(rhs_ty) {
        let condition = "lhs_ty.IsScalableVector() == rhs_ty.IsScalableVector()";
        return Err(check_failed(
            INTERNAL_ERROR,
            condition,
            "Can't match scalable and fixed length vectors",
        ));
    }
    let lanes_match = match is_scalable(lhs_ty) {
        true => vscale_factor(lhs_ty) == vscale_factor(rhs_ty),
        false => encoded_lanes(lhs_ty) == encoded_lanes(rhs_ty),
    };
    if !lanes_match {
        let message = format!(
            "Cannot match type {} vs {}",
            lhs_ty.to_string().as_str(),
            rhs_ty.to_string().as_str()
        );
        return Err(check_failed(INTERNAL_ERROR, "lanes_match", &message));
    }
    let promoted = promote(lhs_ty, rhs_ty)?;
    let lhs = match lhs_ty == promoted {
        true => lhs,
        false => cast(lhs, prim_type(promoted)?, loc)?,
    };
    let rhs = match rhs_ty == promoted {
        true => rhs,
        false => cast(rhs, prim_type(promoted)?, loc)?,
    };
    Ok((lhs, rhs))
}

// ---------------------------------------------------------------------------------------------
// Rounding, predicates and constants (`src/ir/prim/math.cc`)

/// `floor`, `ceil`, `round`, `nearbyint` and `trunc`: an integer or a bool is itself, a float
/// constant is folded with `fold`, keeping its location, and anything else is a call.
fn rounding(x: Expr, name: &'static str, fold: fn(f64) -> f64, loc: &Location) -> Result<Expr> {
    let dtype = prim_dtype(&x)?;
    if matches!(dtype.code, INT | UINT | BOOL) {
        return Ok(x);
    }
    if let Some(imm) = x.as_node::<FloatImmObj>() {
        return Ok(float_imm_at(dtype, fold(imm.value), &imm.loc)?.into());
    }
    call(Some(x.ty.clone()), name, vec![x], loc)
}

pub(crate) fn floor(x: Expr, loc: &Location) -> Result<Expr> {
    rounding(x, "prim.floor", f64::floor, loc)
}

pub(crate) fn ceil(x: Expr, loc: &Location) -> Result<Expr> {
    rounding(x, "prim.ceil", f64::ceil, loc)
}

/// C++ folds `round` with `std::nearbyint`, which rounds a tie to even.
pub(crate) fn round(x: Expr, loc: &Location) -> Result<Expr> {
    rounding(x, "prim.round", f64::round_ties_even, loc)
}

pub(crate) fn nearbyint(x: Expr, loc: &Location) -> Result<Expr> {
    rounding(x, "prim.nearbyint", f64::round_ties_even, loc)
}

pub(crate) fn trunc(x: Expr, loc: &Location) -> Result<Expr> {
    rounding(x, "prim.trunc", |v| if v < 0.0 { v.ceil() } else { v.floor() }, loc)
}

/// The call is typed by its `FInferType`.
pub(crate) fn isnan(x: Expr, loc: &Location) -> Result<Expr> {
    call(None, "prim.isnan", vec![x], loc)
}

pub(crate) fn isinf(x: Expr, loc: &Location) -> Result<Expr> {
    let dtype = prim_dtype(&x)?;
    let result = DLDataType { code: BOOL, bits: 8, lanes: lanes(dtype)? as u16 };
    match dtype.code {
        INT | UINT => make_const(result, Value::Int(BigInt::from(0i64)), loc),
        FLOAT => {
            let infinity = infinity_at(prim_ty(&x)?, loc)?;
            let finite = Ex::from(abs(x.clone(), loc)?).equal(infinity);
            finite.logical_and(!Ex::from(isnan(x, loc)?)).expr()
        }
        _ => {
            let message = format!(
                "Data type {} not supported for finiteness ops. Skipping it...",
                repr(&x.ty)
            );
            Err(internal(&message))
        }
    }
}

pub(crate) fn isfinite(x: Expr, loc: &Location) -> Result<Expr> {
    let infinite = Ex::from(isinf(x.clone(), loc)?);
    (!infinite).logical_and(!Ex::from(isnan(x, loc)?)).expr()
}

/// C++ `infinity(ty)`: positive infinity of a scalar float type.
pub fn infinity(ty: impl IntoPrimType) -> Ex {
    let ty = ty.into_prim_type();
    Ex::from(super::unknown_loc().and_then(|loc| infinity_at(ty?, &loc)))
}

fn infinity_at(ty: PrimType, loc: &Location) -> Result<Expr> {
    let dtype = ty.dtype;
    let lanes = lanes(dtype)?;
    if lanes != 1 {
        // The text of a failed `TVM_FFI_ICHECK_EQ`.
        let message = format!("Check failed: dtype.lanes() == 1 ({lanes} vs. 1) : ");
        return Err(internal(&message));
    }
    match (dtype.code, dtype.bits) {
        (FLOAT, 64) => Ok(float_imm_at(dtype, f64::INFINITY, loc)?.into()),
        (FLOAT, 32 | 16) => Ok(float_imm_at(dtype, f32::INFINITY as f64, loc)?.into()),
        _ => Err(internal(&format!("Cannot decide infinity for type {}", repr(&ty)))),
    }
}

pub(crate) fn abs(x: Expr, loc: &Location) -> Result<Expr> {
    let dtype = prim_dtype(&x)?;
    match dtype.code {
        INT => integer_abs(x, loc),
        FLOAT | BFLOAT => match x.as_node::<FloatImmObj>() {
            Some(imm) => Ok(float_imm_at(dtype, imm.value.abs(), &imm.loc)?.into()),
            None => call(Some(x.ty.clone()), "prim.fabs", vec![x], loc),
        },
        UINT => Ok(x),
        _ => {
            let message = format!(
                "Data type {} not supported for absolute op. Skipping absolute op...",
                repr(&x.ty)
            );
            Err(internal(&message))
        }
    }
}

/// C++ `IntegerAbs(x, loc)`: a constant folded, wrapping at 64 bits; `select(x >= 0, x, -x)`
/// otherwise.
fn integer_abs(x: Expr, loc: &Location) -> Result<Expr> {
    let dtype = prim_dtype(&x)?;
    if let Some(imm) = x.as_node::<IntImmObj>() {
        let mut value = if imm.value.is_negative() { -&imm.value } else { imm.value.clone() };
        if dtype.bits >= 64 {
            value = wrap_signed(value, dtype.bits)?;
        }
        return Ok(int_imm_at(dtype, value, &imm.loc)?.into());
    }
    let zero = make_const(dtype, Value::Int(BigInt::from(0i64)), &super::unknown_loc()?)?;
    let nonnegative = x.greater_equal(zero).expr()?;
    let negated = (-Ex::from(&x)).expr()?;
    Ok(Select::new_full(nonnegative, x, negated, loc.clone())?.into())
}

pub(crate) fn assume(condition: Expr, loc: &Location) -> Result<Expr> {
    let bool_ty = prim_type(DLDataType { code: BOOL, bits: 8, lanes: 1 })?;
    call(Some(bool_ty.into()), "prim.assume", vec![condition], loc)
}

/// An integer constant is its own `likely`.
pub(crate) fn likely(condition: Expr, loc: &Location) -> Result<Expr> {
    if condition.as_node::<IntImmObj>().is_some() {
        return Ok(condition);
    }
    call(Some(condition.ty.clone()), "prim.likely", vec![condition], loc)
}

// ---------------------------------------------------------------------------------------------
// Intrinsics (`TVM_DECLARE_INTRIN_UNARY` and friends in `include/tvm/ir/prim/op.h`)

/// A unary intrinsic `name`, typed as its argument. A bfloat16 argument is computed in float32
/// and cast back; with `float_only`, any argument but a float or a bfloat16 is a `TypeError`.
pub(crate) fn intrinsic(
    name: &'static str,
    x: Expr,
    float_only: bool,
    loc: &Location,
) -> Result<Expr> {
    let dtype = prim_dtype(&x)?;
    if float_only && dtype.code != FLOAT && !is_bfloat16(dtype) {
        let condition = "dtype.code() == DLDataTypeCode::kDLFloat || \
                         dtype.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)";
        let message =
            format!("{name} only supports floating-point inputs, but got {}", repr(&x.ty));
        return Err(check_failed(TYPE_ERROR, condition, &message));
    }
    if !is_bfloat16(dtype) {
        return call(Some(x.ty.clone()), name, vec![x], loc);
    }
    let f32_ty = prim_type(DLDataType { code: FLOAT, bits: 32, lanes: dtype.lanes })?;
    let x_f32: Expr = Cast::new_full(x, f32_ty.clone(), loc.clone())?.into();
    let result = call(Some(f32_ty.into()), name, vec![x_f32], loc)?;
    Ok(Cast::new_full(result, prim_type(dtype)?, loc.clone())?.into())
}

/// A binary intrinsic `name`, typed as its first argument; the arguments are not matched.
pub(crate) fn binary_intrinsic(
    name: &'static str,
    x: Expr,
    y: Expr,
    loc: &Location,
) -> Result<Expr> {
    call(Some(x.ty.clone()), name, vec![x, y], loc)
}

// ---------------------------------------------------------------------------------------------
// Functions of two matched values

/// `pow` and `fmod`: both sides matched, and a float result.
fn float_binary(x: Expr, y: Expr, name: &'static str, what: &str, loc: &Location) -> Result<Expr> {
    let (x, y) = match_types_at(x, y, loc)?;
    if prim_dtype(&x)?.code != FLOAT {
        let condition = "x.ty().MatchesCode(DLDataTypeCode::kDLFloat)";
        return Err(check_failed(
            INTERNAL_ERROR,
            condition,
            &format!("{what} only applies to float"),
        ));
    }
    call(Some(x.ty.clone()), name, vec![x, y], loc)
}

/// C++ also logs a warning for a constant exponent of 3 or more, which is not repeated here.
pub(crate) fn pow(x: Expr, y: Expr, loc: &Location) -> Result<Expr> {
    float_binary(x, y, "prim.pow", "power", loc)
}

pub(crate) fn fmod(x: Expr, y: Expr, loc: &Location) -> Result<Expr> {
    float_binary(x, y, "prim.fmod", "fmod", loc)
}

/// C++ `tirx::logaddexp(a, b, loc)`: `log(exp(a) + exp(b))` of two matched floats.
pub(crate) fn logaddexp(a: Expr, b: Expr, loc: &Location) -> Result<Expr> {
    for (side, value) in [("a", &a), ("b", &b)] {
        if prim_dtype(value)?.code != FLOAT {
            let condition = format!("{side}.ty().MatchesCode(DLDataTypeCode::kDLFloat)");
            return Err(check_failed(INTERNAL_ERROR, &condition, &repr(value)));
        }
    }
    let (a, b) = match_types_at(a, b, loc)?;
    let no_loc = super::unknown_loc()?;
    let sum = (Ex::from(intrinsic("prim.exp", a, false, &no_loc)?)
        + Ex::from(intrinsic("prim.exp", b, false, &no_loc)?))
    .expr()?;
    intrinsic("prim.log", sum, false, &no_loc)
}

/// C++ `tirx::fast_erf_float_expr(arg, bits)`: a rational approximation of `erf` on `[-4, 4]`,
/// in `bits`-bit floats.
pub fn fast_erf_float_expr(arg: impl Into<Ex>, bits: u8) -> Ex {
    let dtype = DLDataType { code: FLOAT, bits, lanes: 1 };
    // The C++ coefficients are `float` literals.
    let constant = |value: f32| -> Ex { float_imm(dtype, value as f64).into() };
    // alpha_1, alpha_3, ..., alpha_13 and beta_0, beta_2, ..., beta_8, verbatim.
    #[allow(clippy::excessive_precision)]
    let alpha: [f32; 7] = [
        -1.60960333262415e-02,
        -2.95459980854025e-03,
        -7.34990630326855e-04,
        -5.69250639462346e-05,
        -2.10102402082508e-06,
        2.77068142495902e-08,
        -2.72614225801306e-10,
    ];
    #[allow(clippy::excessive_precision)]
    let beta: [f32; 5] = [
        -1.42647390514189e-02,
        -7.37332916720468e-03,
        -1.68282697438203e-03,
        -2.13374055278905e-04,
        -1.45660718464996e-05,
    ];
    let x = arg.into().min(constant(4.0)).max(constant(-4.0));
    let x2 = &x * &x;
    // Horner's rule from the highest coefficient down, as C++ writes it out.
    let mut p = &x2 * constant(alpha[6]) + constant(alpha[5]);
    for coefficient in alpha[..5].iter().rev() {
        p = &x2 * p + constant(*coefficient);
    }
    p = &x * p;
    let mut q = &x2 * constant(beta[4]) + constant(beta[3]);
    for coefficient in beta[..3].iter().rev() {
        q = &x2 * q + constant(*coefficient);
    }
    p.div(q)
}

// ---------------------------------------------------------------------------------------------
// reinterpret (`src/tirx/op/memory.cc`)

fn is_scalar_64bit_integer(dtype: DLDataType) -> bool {
    is_scalar(dtype) && dtype.bits == 64 && matches!(dtype.code, INT | UINT)
}

/// C++ `tirx::reinterpret(target_ty, value, loc)`: the bits of `value` as `target_ty`.
pub(crate) fn reinterpret(target: Type, value: Expr, loc: &Location) -> Result<Expr> {
    let target_is_ptr = target.as_node::<PtrTypeObj>().is_some();
    if value.as_node::<StringImmObj>().is_some() {
        if !target_is_ptr {
            let message =
                format!("String reinterpret requires a pointer target, but got {}", repr(&target));
            return Err(check_failed(TYPE_ERROR, "target_ty.as<PtrTypeNode>()", &message));
        }
    } else if let Some(target_prim) = target.as_node::<PrimTypeObj>() {
        let target_dtype = target_prim.dtype;
        match value.ty.as_node::<PrimTypeObj>() {
            Some(source) if source.dtype == target_dtype => return Ok(value),
            Some(source) => {
                let source_dtype = source.dtype;
                if !is_scalable(target_dtype) && !is_scalable(source_dtype) {
                    let bits =
                        |d: DLDataType| -> Result<i64> { Ok(d.bits as i64 * lanes(d)? as i64) };
                    let float4 = source_dtype.code == FLOAT4 || target_dtype.code == FLOAT4;
                    let same_size = bits(source_dtype)? == bits(target_dtype)?
                        || (float4 && storage_bytes(source_dtype) == storage_bytes(target_dtype));
                    if !same_size {
                        let condition = "value_bits == target_bits || \
                                         ((value_dtype.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn) || \
                                         target_prim.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn)) && \
                                         value_dtype.StorageBytes() == target_prim.StorageBytes())";
                        let message = format!(
                            "Reinterpret requires size match {} vs {}",
                            repr(&target),
                            repr(&value.ty)
                        );
                        return Err(check_failed(INTERNAL_ERROR, condition, &message));
                    }
                }
            }
            None => {
                if value.ty.as_node::<PtrTypeObj>().is_none() {
                    let message = format!(
                        "Reinterpret source must be PrimType or PtrType, but got {}",
                        repr(&value.ty)
                    );
                    return Err(check_failed(TYPE_ERROR, "value->ty.as<PtrTypeNode>()", &message));
                }
                if !is_scalar_64bit_integer(target_dtype) {
                    let condition = "target_dtype.value().IsScalar() && target_dtype.value().bits() == 64 && \
                                     target_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)";
                    let message = format!(
                        "Pointer reinterpret requires a scalar 64-bit integer target, but got {}",
                        repr(&target)
                    );
                    return Err(check_failed(TYPE_ERROR, condition, &message));
                }
            }
        }
    } else {
        if !target_is_ptr {
            let message = format!(
                "Reinterpret target must be PrimType or PtrType, but got {}",
                repr(&target)
            );
            return Err(check_failed(TYPE_ERROR, "target_ty.as<PtrTypeNode>()", &message));
        }
        match value.ty.as_node::<PrimTypeObj>() {
            Some(source) if !is_scalar_64bit_integer(source.dtype) => {
                let condition = "source_dtype.value().IsScalar() && source_dtype.value().bits() == 64 && \
                                 source_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)";
                let message = format!(
                    "Pointer reinterpret requires a scalar 64-bit integer source, but got {}",
                    repr(&value.ty)
                );
                return Err(check_failed(TYPE_ERROR, condition, &message));
            }
            Some(_) => {}
            None if value.ty.as_node::<PtrTypeObj>().is_none() => {
                let message = format!(
                    "Reinterpret source must be PrimType or PtrType, but got {}",
                    repr(&value.ty)
                );
                return Err(check_failed(TYPE_ERROR, "value->ty.as<PtrTypeNode>()", &message));
            }
            None => {}
        }
    }
    call(Some(target), "tirx.reinterpret", vec![value], loc)
}
