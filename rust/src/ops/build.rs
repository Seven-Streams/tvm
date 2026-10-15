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

//! Constructors shaped like the C++ ones, and the constants they build on (see the module doc).

use std::cell::{OnceCell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fmt::Display;
use std::sync::OnceLock;
use std::thread::LocalKey;

use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use tvm_ffi::{
    Any, AnyView, Array, BigInt, DLDataType, DLDataTypeCode, DLDataTypeExt,
    Function as FfiFunction, Map, ObjectArc, ObjectCore, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, String, TypeAttrColumn, INTERNAL_ERROR, TYPE_ERROR, VALUE_ERROR,
};

use super::arith::Ex;
use super::math::{check_failed, element, lanes, op, BOOL, HANDLE, INT, UINT};
use super::native::{repr, structural_equal};
use crate::ir::{
    AssertStmt, Attrs, Bind, Call, CallObj, DictAttrs, Evaluate, EvaluateObj, Expr, FloatImm, For,
    FuncType, FuncTypeObj, GlobalVar, If, IntImm, IntImmObj, LambdaExpr, Location, MissingType,
    MissingTypeObj, Op, OpObj, PrimType, PrimTypeObj, PtrType, Range, RegionStmt, Return, SeqStmt,
    SeqStmtObj, Stmt, StringImm, StringType, TensorLoad, TensorStore, Tuple, TupleType, Type,
    UniqueNameSupply, UnknownLoc, Var,
};
use crate::prim::{
    Add, And, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, Broadcast, Cast, Div, FloorDiv,
    FloorMod, LShift, Let, Max, Min, Mod, Mul, MulObj, Not, Or, RShift, Ramp, Select, Shuffle, Sub,
    EQ, GE, GT, LE, LT, NE,
};
use crate::runtime::{MemoryScope, MemoryScopeObj};
use crate::target::Target;
use crate::tirx::{
    Axis, DispatchContext, ExecScope, Function, Iter, Layout, TensorType, TensorTypeObj, TileLayout,
};

// ---------------------------------------------------------------------------------------------
// Constants and types

thread_local! {
    static UNKNOWN_LOC: OnceCell<Location> = const { OnceCell::new() };
    static MISSING_TYPE: OnceCell<MissingType> = const { OnceCell::new() };
    static GLOBAL_SCOPE: OnceCell<MemoryScope> = const { OnceCell::new() };
}

/// A process-wide singleton, fetched once per thread.
fn singleton<T: Clone>(
    cell: &'static LocalKey<OnceCell<T>>,
    fetch: impl FnOnce() -> Result<T>,
) -> Result<T> {
    cell.with(|cell| {
        if let Some(value) = cell.get() {
            return Ok(value.clone());
        }
        let value = fetch()?;
        let _ = cell.set(value.clone());
        Ok(value)
    })
}

/// The process-wide `UnknownLoc()` singleton, as a `Location`, fetched once per thread.
pub fn unknown_loc() -> Result<Location> {
    singleton(&UNKNOWN_LOC, || Ok(UnknownLoc::new()?.into()))
}

/// The registered type key of a type index (C++ `TypeIndexToTypeKey`).
pub(crate) fn type_key(type_index: i32) -> std::string::String {
    // SAFETY: every value's type index is registered.
    unsafe {
        let info = tvm_ffi::tvm_ffi_sys::TVMFFIGetTypeInfo(type_index);
        (*info).type_key.as_str().to_string()
    }
}

pub fn int32_dtype() -> DLDataType {
    DLDataType { code: DLDataTypeCode::kDLInt as u8, bits: 32, lanes: 1 }
}

pub(crate) fn scalar_dtype(code: DLDataTypeCode, bits: u8) -> DLDataType {
    DLDataType { code: code as u8, bits, lanes: 1 }
}

/// `PrimType(dtype)`. C++ keeps one node per dtype and compares types by identity, so the node
/// comes from the native cache, fetched once per thread and dtype.
pub fn prim_type(dtype: DLDataType) -> Result<PrimType> {
    thread_local! {
        static PRIM_TYPES: RefCell<HashMap<(u8, u8, u16), PrimType>> = RefCell::new(HashMap::new());
    }
    let key = (dtype.code, dtype.bits, dtype.lanes);
    if let Some(ty) = PRIM_TYPES.with(|types| types.borrow().get(&key).cloned()) {
        return Ok(ty);
    }
    let ty: PrimType =
        tvm_ffi::cached_global_func!("ir.PrimType").call_tuple((dtype,))?.try_into()?;
    PRIM_TYPES.with(|types| types.borrow_mut().insert(key, ty.clone()));
    Ok(ty)
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

/// C++ `IntImm(dtype, value)`: a scalar integer or boolean constant; the value must fit the type.
pub fn int_imm(dtype: DLDataType, value: i128) -> Result<IntImm> {
    big_int_imm(dtype, BigInt::from(value))
}

pub(crate) fn big_int_imm(dtype: DLDataType, value: BigInt) -> Result<IntImm> {
    int_imm_at(dtype, value, &unknown_loc()?)
}

/// C++ `IntImm(dtype, value, loc)`.
pub(crate) fn int_imm_at(dtype: DLDataType, value: BigInt, loc: &Location) -> Result<IntImm> {
    let ty = prim_type(dtype)?;
    check_int_imm(dtype, &value)?;
    Ok(IntImm::new(loc.clone(), ty.into(), value))
}

/// The checks of C++ `IntImm(ty, value)`: a scalar integer type whose range holds `value`.
fn check_int_imm(dtype: DLDataType, value: &BigInt) -> Result<()> {
    let name = dtype.to_string();
    let name = name.as_str();
    if !is_scalar_type(dtype) {
        let condition = "!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector()";
        let message = format!("IntImm can only take scalar, but {name} was supplied.");
        return Err(check_failed(VALUE_ERROR, condition, &message));
    }
    if !matches!(dtype.code, INT | UINT | BOOL) {
        let condition = "value_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt, \
                         DLDataTypeCode::kDLBool)";
        let message =
            format!("IntImm supports only int or uint or bool type, but {name} was supplied.");
        return Err(check_failed(VALUE_ERROR, condition, &message));
    }
    let bits = dtype.bits as u32;
    if bits == 0 {
        let message = "IntImm requires a positive integer width";
        return Err(compare_failed(VALUE_ERROR, "bits > 0", bits, 0, message));
    }
    let zero = BigInt::from(0i64);
    if dtype.code == UINT {
        if *value < zero {
            let message =
                format!("Literal value {value} is negative for unsigned integer type {name}");
            return Err(compare_failed(VALUE_ERROR, "value >= 0", value, 0, &message));
        }
        let message = format!("Literal value {value} exceeds maximum of {name}");
        if bits <= 64 {
            let fits = value.to_u64().is_some_and(|small| bits == 64 || small < 1u64 << bits);
            if !fits {
                let condition =
                    "small.has_value() && (bits == 64 || *small < (uint64_t{1} << bits))";
                return Err(check_failed(VALUE_ERROR, condition, &message));
            }
        } else {
            let limit = BigInt::from(1i64).try_shl(bits as i64)?;
            if *value >= limit {
                let condition = "value < ffi::BigInt(1) << bits";
                return Err(compare_failed(VALUE_ERROR, condition, value, &limit, &message));
            }
        }
    } else if bits == 1 || dtype.code == BOOL {
        if *value != zero && *value != BigInt::from(1i64) {
            let message = format!("{value} exceeds range of {name}");
            return Err(check_failed(VALUE_ERROR, "value == 0 || value == 1", &message));
        }
    } else {
        let message = format!("Literal value {value} exceeds range of {name}");
        if bits <= 64 {
            let half = 1i128 << (bits - 1);
            let fits = value.to_i64().is_some_and(|small| {
                bits == 64 || (small as i128 >= -half && (small as i128) < half)
            });
            if !fits {
                let condition = "small.has_value() && (bits == 64 || (*small >= -(int64_t{1} << \
                                 (bits - 1)) && *small < (int64_t{1} << (bits - 1))))";
                return Err(check_failed(VALUE_ERROR, condition, &message));
            }
        } else {
            let limit = BigInt::from(1i64).try_shl(bits as i64 - 1)?;
            if *value < -limit.clone() || *value >= limit {
                return Err(check_failed(
                    VALUE_ERROR,
                    "value >= -limit && value < limit",
                    &message,
                ));
            }
        }
    }
    Ok(())
}

/// C++ `FloatImm(dtype, value)`: a scalar floating-point constant; a finite value must be in the
/// range of the type.
pub fn float_imm(dtype: DLDataType, value: f64) -> Result<FloatImm> {
    float_imm_at(dtype, value, &unknown_loc()?)
}

/// C++ `FloatImm(dtype, value, loc)`.
pub(crate) fn float_imm_at(dtype: DLDataType, value: f64, loc: &Location) -> Result<FloatImm> {
    let ty = prim_type(dtype)?;
    check_float_imm(dtype, value)?;
    Ok(FloatImm::new(loc.clone(), ty.into(), value))
}

/// The checks of C++ `FloatImm(ty, value)`: a scalar floating-point type whose range holds a
/// finite `value` (`support::kMax*` in `src/support/limits.h`).
fn check_float_imm(dtype: DLDataType, value: f64) -> Result<()> {
    const FLOAT: u8 = DLDataTypeCode::kDLFloat as u8;
    const BFLOAT: u8 = DLDataTypeCode::kDLBfloat as u8;
    const FLOAT4_E2M1FN: u8 = DLDataTypeCode::kDLFloat4_e2m1fn as u8;
    /// `ffi::DLExtDataTypeCode::kDLExtCustomBegin`.
    const CUSTOM_BEGIN: u8 = 129;
    let name = dtype.to_string();
    let name = name.as_str();
    if !is_scalar_type(dtype) {
        let condition = "!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector()";
        return Err(check_failed(VALUE_ERROR, condition, "FloatImm can only take scalar."));
    }
    let (code, bits) = (dtype.code, dtype.bits);
    let float8 = float8_bound(code);
    let float6 = float6_bound(code);
    let supported = code == FLOAT
        || float8.is_some()
        || float6.is_some()
        || (code == BFLOAT && bits == 16)
        || (code == FLOAT4_E2M1FN && bits == 4)
        || code >= CUSTOM_BEGIN;
    if !supported {
        let condition = "value_ty.MatchesCode(DLDataTypeCode::kDLFloat, \
                         DLDataTypeCode::kDLFloat8_e3m4, DLDataTypeCode::kDLFloat8_e4m3, \
                         DLDataTypeCode::kDLFloat8_e4m3b11fnuz, DLDataTypeCode::kDLFloat8_e4m3fn, \
                         DLDataTypeCode::kDLFloat8_e4m3fnuz, DLDataTypeCode::kDLFloat8_e5m2, \
                         DLDataTypeCode::kDLFloat8_e5m2fnuz, DLDataTypeCode::kDLFloat8_e8m0fnu, \
                         DLDataTypeCode::kDLFloat6_e2m3fn, DLDataTypeCode::kDLFloat6_e3m2fn) || \
                         value_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16) || \
                         value_ty.MatchesElementType(DLDataTypeCode::kDLFloat4_e2m1fn, 4) || \
                         static_cast<int>(code) >= \
                         static_cast<int>(ffi::DLExtDataTypeCode::kDLExtCustomBegin)";
        let message = format!("FloatImm supports only float, but {name} was supplied.");
        return Err(check_failed(VALUE_ERROR, condition, &message));
    }
    if value.is_infinite() || value.is_nan() {
        return Ok(());
    }
    let below = |condition: &str, bound: f64, what: &str| {
        let message = format!("Literal value {} {what} {name}", CppDouble(value));
        compare_failed(VALUE_ERROR, condition, CppDouble(value), CppDouble(bound), &message)
    };
    // The bound checked from below, how its message says so, and the bound from above.
    let (low, high): ((&str, f64, &str), (&str, f64)) = if bits == 32 {
        let lowest = (f32::MIN as f64, f32::MAX as f64);
        (
            ("value >= std::numeric_limits<float>::lowest()", lowest.0, "exceeds minimum of"),
            ("value <= std::numeric_limits<float>::max()", lowest.1),
        )
    } else if code == FLOAT && bits == 16 {
        (
            ("value >= -support::kMaxFloat16", -65504.0, "exceeds minimum of"),
            ("value <= support::kMaxFloat16", 65504.0),
        )
    } else if code == BFLOAT && bits == 16 {
        const MAX_BFLOAT16: f64 = 3.895313892515354759047080037148786688e38;
        (
            ("value >= -support::kMaxBFloat16", -MAX_BFLOAT16, "exceeds minimum of"),
            ("value <= support::kMaxBFloat16", MAX_BFLOAT16),
        )
    } else if let Some((bound, nonneg)) = float8 {
        let low = match nonneg {
            true => ("value >= 0", 0.0, "below zero for unsigned"),
            false => ("value >= -bound", -bound, "below minimum of"),
        };
        (low, ("value <= bound", bound))
    } else if let Some(bound) = float6 {
        (("value >= -bound", -bound, "below minimum of"), ("value <= bound", bound))
    } else if code == FLOAT4_E2M1FN {
        (("value >= -bound", -6.0, "below minimum of"), ("value <= bound", 6.0))
    } else {
        return Ok(());
    };
    if value < low.1 {
        return Err(below(low.0, low.1, low.2));
    }
    if value > high.1 {
        return Err(below(high.0, high.1, "exceeds maximum of"));
    }
    Ok(())
}

/// `support::kMaxE8M0FNU`, 2^128.
const MAX_E8M0FNU: f64 = 3.402_823_669_209_385e38;

/// The largest value of a float8 type code, and whether the type is unsigned.
fn float8_bound(code: u8) -> Option<(f64, bool)> {
    use DLDataTypeCode::*;
    let bound = match code {
        c if c == kDLFloat8_e3m4 as u8 => (31.0, false),
        c if c == kDLFloat8_e4m3 as u8 => (448.0, false),
        c if c == kDLFloat8_e4m3b11fnuz as u8 => (30.0, true),
        c if c == kDLFloat8_e4m3fn as u8 => (448.0, false),
        c if c == kDLFloat8_e4m3fnuz as u8 => (448.0, true),
        c if c == kDLFloat8_e5m2 as u8 => (57344.0, false),
        c if c == kDLFloat8_e5m2fnuz as u8 => (57344.0, true),
        c if c == kDLFloat8_e8m0fnu as u8 => (MAX_E8M0FNU, true),
        _ => return None,
    };
    Some(bound)
}

/// The largest value of a float6 type code.
fn float6_bound(code: u8) -> Option<f64> {
    match code {
        c if c == DLDataTypeCode::kDLFloat6_e2m3fn as u8 => Some(7.5),
        c if c == DLDataTypeCode::kDLFloat6_e3m2fn as u8 => Some(28.0),
        _ => None,
    }
}

/// Neither a fixed-length nor a scalable vector (C++ `!IsFixedLengthVector() &&
/// !IsScalableVector()`).
fn is_scalar_type(dtype: DLDataType) -> bool {
    (-1..=1).contains(&(dtype.lanes as i16))
}

/// A double as C++ `std::ostream` prints it by default: `%g`, six significant digits.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct CppDouble(f64);

impl Display for CppDouble {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.0;
        if value.is_nan() {
            return f.write_str(if value.is_sign_negative() { "-nan" } else { "nan" });
        }
        if value.is_infinite() {
            return f.write_str(if value < 0.0 { "-inf" } else { "inf" });
        }
        if value == 0.0 {
            return f.write_str(if value.is_sign_negative() { "-0" } else { "0" });
        }
        let trim = |text: &str| -> std::string::String {
            match text.contains('.') {
                true => text.trim_end_matches('0').trim_end_matches('.').to_string(),
                false => text.to_string(),
            }
        };
        let scientific = format!("{value:.5e}");
        let (mantissa, exponent) = scientific.split_once('e').expect("an exponent");
        let exponent: i32 = exponent.parse().expect("an integer exponent");
        if !(-4..6).contains(&exponent) {
            let sign = if exponent < 0 { '-' } else { '+' };
            return write!(f, "{}e{sign}{:02}", trim(mantissa), exponent.abs());
        }
        let decimals = (5 - exponent) as usize;
        f.write_str(&trim(&format!("{value:.decimals$}")))
    }
}

/// `prim::MakeConst(dtype, value)`, which has no global: an `IntImm` or `FloatImm` by type code,
/// broadcast for a vector type. An integer literal next to an expression is typed so.
pub fn make_const(dtype: DLDataType, value: i64) -> Result<Expr> {
    super::math::make_const(dtype, super::math::Value::Int(BigInt::from(value)), &unknown_loc()?)
}

// ---------------------------------------------------------------------------------------------
// Implicit conversions

/// What C++ converts to a `PrimExpr` argument implicitly: an expression as it is, an `i32` to
/// `IntImm::Int32` (`PrimExpr(int32_t)`), an `f32` to a float32 `FloatImm` (`PrimExpr(float)`).
pub trait IntoExpr {
    fn into_expr(self) -> Result<Expr>;
}

impl<T: Into<Ex>> IntoExpr for T {
    fn into_expr(self) -> Result<Expr> {
        self.into().expr()
    }
}

impl IntoExpr for i32 {
    fn into_expr(self) -> Result<Expr> {
        Ok(int32(self as i64)?.into())
    }
}

impl IntoExpr for f32 {
    fn into_expr(self) -> Result<Expr> {
        Ok(float_imm(scalar_dtype(DLDataTypeCode::kDLFloat, 32), self as f64)?.into())
    }
}

/// What converts to a statement body: a `SeqStmt` as it is, any other statement by the C++
/// implicit `Stmt -> SeqStmt` conversion ([`SeqStmt::from_stmt`]).
pub trait IntoBody {
    fn into_body(self) -> Result<SeqStmt>;
}

impl<T: Into<Stmt>> IntoBody for T {
    fn into_body(self) -> Result<SeqStmt> {
        SeqStmt::from_stmt(self)
    }
}

/// A primitive type argument: a `PrimType`, or a `DLDataType` for the C++ overloads that take
/// one (`cast(value, DLDataType)`).
pub trait IntoPrimType {
    fn into_prim_type(self) -> Result<PrimType>;
}

impl IntoPrimType for PrimType {
    fn into_prim_type(self) -> Result<PrimType> {
        Ok(self)
    }
}

impl IntoPrimType for &PrimType {
    fn into_prim_type(self) -> Result<PrimType> {
        Ok(self.clone())
    }
}

impl IntoPrimType for DLDataType {
    fn into_prim_type(self) -> Result<PrimType> {
        prim_type(self)
    }
}

// ---------------------------------------------------------------------------------------------
// Locations, constants and names

impl UnknownLoc {
    /// C++ `UnknownLoc()`: the process-wide singleton, which C++ owns, so it is fetched from there.
    pub fn new() -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.UnknownLoc").call_packed(&[])?.try_into()
    }
}

impl StringImm {
    /// C++ `StringImm(value)`, typed `StringType()`.
    pub fn new(value: &str) -> Result<Self> {
        let loc = unknown_loc()?;
        Ok(Self::from_complete_fields(
            loc.clone(),
            StringType::new(loc).into(),
            String::from(value),
        ))
    }
}

/// C++ `max_value(dtype)`: the largest value of a scalar type.
pub fn max_value(dtype: DLDataType) -> Result<Expr> {
    limit_value(dtype, true)
}

/// C++ `min_value(dtype)`: the smallest value of a scalar type.
pub fn min_value(dtype: DLDataType) -> Result<Expr> {
    limit_value(dtype, false)
}

/// C++ `max_value(dtype)` (`max`) or `min_value(dtype)`; the float8 and narrower limits follow
/// <https://arxiv.org/pdf/2209.05433.pdf>.
fn limit_value(dtype: DLDataType, max: bool) -> Result<Expr> {
    use DLDataTypeCode::*;
    let (loc, ty) = (unknown_loc()?, prim_type(dtype)?);
    let lanes = lanes(dtype)?;
    if lanes != 1 {
        return Err(compare_failed(INTERNAL_ERROR, "dtype.lanes() == 1", lanes, 1, ""));
    }
    let (code, bits) = (dtype.code, dtype.bits as u32);
    let int =
        |value: i128| -> Result<Expr> { Ok(int_imm_at(dtype, BigInt::from(value), &loc)?.into()) };
    let float = |value: f64| -> Result<Expr> { Ok(float_imm_at(dtype, value, &loc)?.into()) };
    let pick = |max_value: f64, min_value: f64| float(if max { max_value } else { min_value });
    match code {
        INT if bits <= 64 => {
            let half = 1i128 << (bits - 1);
            int(if max { half - 1 } else { -half })
        }
        UINT if !max => int(0),
        UINT if bits < 64 => int((1i128 << bits) - 1),
        UINT if bits == 64 => {
            let value = super::math::Value::Int(BigInt::from(u64::MAX));
            super::math::make_const(dtype, value, &loc)
        }
        c if c == kDLFloat as u8 && bits == 64 => pick(f64::MAX, f64::MIN),
        c if c == kDLFloat as u8 && bits == 32 => pick(f32::MAX as f64, f32::MIN as f64),
        c if c == kDLFloat as u8 && bits == 16 => pick(65504.0, -65504.0),
        c if c == kDLBfloat as u8 => pick(f32::MAX as f64, f32::MIN as f64),
        c if c == kDLFloat8_e5m2 as u8 => pick(57344.0, -57344.0),
        c if c == kDLFloat8_e5m2fnuz as u8 => pick(57344.0, 0.0),
        c if c == kDLFloat8_e4m3fn as u8 || c == kDLFloat8_e4m3 as u8 => pick(448.0, -448.0),
        c if c == kDLFloat8_e4m3fnuz as u8 => pick(448.0, 0.0),
        c if c == kDLFloat8_e4m3b11fnuz as u8 => pick(30.0, 0.0),
        c if c == kDLFloat8_e3m4 as u8 => pick(31.0, -31.0),
        c if c == kDLFloat8_e8m0fnu as u8 => pick(MAX_E8M0FNU, 0.0),
        c if c == kDLFloat6_e2m3fn as u8 => pick(7.5, -7.5),
        c if c == kDLFloat6_e3m2fn as u8 => pick(28.0, -28.0),
        c if c == kDLFloat4_e2m1fn as u8 => pick(6.0, -6.0),
        _ => {
            let which = if max { "max_value" } else { "min_value" };
            let message = format!("Cannot decide {which} for type{}", repr(&ty));
            Err(tvm_ffi::Error::new(INTERNAL_ERROR, &message, ""))
        }
    }
}

impl FuncType {
    /// C++ `FuncType(arg_types, ret_type)`.
    pub fn new(
        arg_types: impl IntoIterator<Item = Type>,
        ret_type: impl Into<Type>,
    ) -> Result<Self> {
        let arg_types: Array<Type> = arg_types.into_iter().collect();
        Ok(Self::from_complete_fields(unknown_loc()?, arg_types, ret_type.into()))
    }
}

impl Range {
    /// C++ `Range(begin, end)`: the extent is `end - begin`, or `end` from a constant 0.
    pub fn new(begin: impl IntoExpr, end: impl IntoExpr) -> Result<Self> {
        let (begin, end) = (begin.into_expr()?, end.into_expr()?);
        prim_dtype(&begin)?;
        prim_dtype(&end)?;
        let extent = match is_const_int(&begin, 0) {
            true => end,
            false => (Ex::from(end) - Ex::from(begin.clone())).expr()?,
        };
        Ok(Self::from_complete_fields(begin, extent, unknown_loc()?))
    }

    /// C++ `Range::FromMinExtent(min, extent)`.
    pub fn from_min_extent(min: impl IntoExpr, extent: impl IntoExpr) -> Result<Self> {
        let (min, extent) = (min.into_expr()?, extent.into_expr()?);
        prim_dtype(&min)?;
        prim_dtype(&extent)?;
        Ok(Self::from_complete_fields(min, extent, unknown_loc()?))
    }
}

/// C++ `IsConstInt(expr, value)`: an `IntImm` of that value.
fn is_const_int(expr: &Expr, value: i64) -> bool {
    expr.as_node::<IntImmObj>().is_some_and(|imm| imm.value == BigInt::from(value))
}

// ---------------------------------------------------------------------------------------------
// Types and variables

impl MissingType {
    /// C++ `MissingType()`: the process-wide singleton, the type of an expression whose type is
    /// not known yet. C++ owns it, so it is fetched from there once per thread.
    pub fn new() -> Result<Self> {
        singleton(&MISSING_TYPE, || {
            tvm_ffi::cached_global_func!("ir.MissingType").call_packed(&[])?.try_into()
        })
    }
}

impl MemoryScope {
    /// C++ `MemoryScope::Get(name)`: the registered scope of that name; an error for an
    /// unregistered one. The enum's registry lives in C++; its JSON decoder is this lookup.
    pub fn get(name: &str) -> Result<Self> {
        let decode = tvm_ffi::get_type_attr(MemoryScopeObj::type_index(), "__data_from_json__")
            .ok_or_else(|| super::math::internal("runtime.MemoryScope has no JSON decoder"))?;
        FfiFunction::try_from(decode)?.call_tuple((String::from(name),))?.try_into()
    }

    /// C++ `MemoryScope::Global()`, the default scope.
    pub fn global() -> Result<Self> {
        singleton(&GLOBAL_SCOPE, || Self::get("global"))
    }
}

impl PtrType {
    /// C++ `PtrType(element_type, scope)`: a pointer to `element_type` values in memory `scope`.
    /// The element type cannot be `MissingType`.
    pub fn new(element_type: impl Into<Type>, scope: MemoryScope) -> Result<Self> {
        let element_type = element_type.into();
        if element_type.as_node::<MissingTypeObj>().is_some() {
            let condition = "!element_type.as<MissingType>().has_value()";
            let message = "PtrType element_type cannot be Type::Missing()";
            return Err(check_failed(INTERNAL_ERROR, condition, message));
        }
        Ok(Self::from_complete_fields(unknown_loc()?, element_type, scope))
    }

    /// C++ `PtrType::VoidPtrType()`: an untyped pointer to global memory.
    pub fn void() -> Result<Self> {
        let void = DLDataType { code: HANDLE, bits: 0, lanes: 0 };
        Self::new(prim_type(void)?, MemoryScope::global()?)
    }
}

impl TupleType {
    /// C++ `TupleType(fields)`.
    pub fn new(fields: impl IntoIterator<Item = Type>) -> Result<Self> {
        Self::new_full(fields, unknown_loc()?)
    }

    pub fn new_full(fields: impl IntoIterator<Item = Type>, loc: Location) -> Result<Self> {
        Ok(Self::from_complete_fields(loc, fields.into_iter().collect()))
    }
}

impl TensorType {
    /// C++ `TensorType(shape, dtype, scope)`: compact (no strides), at offset 0, with the default
    /// alignment and offset factor, and no layout.
    pub fn new(
        shape: impl IntoIterator<Item = Expr>,
        dtype: impl IntoPrimType,
        scope: MemoryScope,
    ) -> Result<Self> {
        Self::new_full(shape, dtype, scope, [], None, 0, 0, None, unknown_loc()?)
    }

    /// C++ `TensorType(shape, dtype, scope, strides, elem_offset, data_alignment, offset_factor,
    /// layout, loc)`. A missing `elem_offset` is an index-typed 0, a `data_alignment` of 0 or less
    /// is `kAllocAlignment` and an `offset_factor` of 0 is 1.
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        shape: impl IntoIterator<Item = Expr>,
        dtype: impl IntoPrimType,
        scope: MemoryScope,
        strides: impl IntoIterator<Item = Expr>,
        elem_offset: Option<Expr>,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<Layout>,
        loc: Location,
    ) -> Result<Self> {
        /// `runtime::kAllocAlignment`.
        const ALLOC_ALIGNMENT: i32 = 64;
        let elem_offset = match elem_offset {
            Some(offset) => offset,
            // `DefaultIndexType()`: int64, as TVM is built with `INDEX_DEFAULT_I64`.
            None => int_imm(scalar_dtype(DLDataTypeCode::kDLInt, 64), 0)?.into(),
        };
        Ok(Self::from_complete_fields(
            loc,
            dtype.into_prim_type()?,
            scope,
            shape.into_iter().collect(),
            strides.into_iter().collect(),
            elem_offset,
            if data_alignment <= 0 { ALLOC_ALIGNMENT } else { data_alignment },
            if offset_factor == 0 { 1 } else { offset_factor },
            layout,
        ))
    }
}

impl TensorTypeObj {
    /// C++ `TensorTypeNode::DataPtrType()`: the type of the tensor's data pointer.
    pub fn data_ptr_type(&self) -> Result<PtrType> {
        PtrType::new(self.dtype.clone(), self.scope.clone())
    }
}

impl Var {
    /// C++ `Var(name, ty)`. A `TensorType` makes a tensor variable (C++ `TensorVar(name, ty)`).
    pub fn new(name: impl Into<String>, ty: impl Into<Type>) -> Result<Self> {
        Self::new_full(name, ty, unknown_loc()?)
    }

    pub fn new_full(name: impl Into<String>, ty: impl Into<Type>, loc: Location) -> Result<Self> {
        Ok(Self::from_complete_fields(loc, ty.into(), name.into()))
    }

    /// C++ `PrimVar(name, dtype)`.
    pub fn prim(name: &str, dtype: DLDataType) -> Result<Self> {
        Self::new(name, prim_type(dtype)?)
    }

    /// C++ `TensorVar::data()`: `tensor_data_ptr(var)`, the data pointer of a tensor variable.
    pub fn tensor_data(&self) -> Result<Expr> {
        let ty: TensorType = self.ty.clone().try_cast()?;
        let this: Expr = self.clone().into();
        Ok(Call::new(ty.data_ptr_type()?, op("tirx.tensor_data_ptr")?, [this])?.into())
    }
}

impl GlobalVar {
    /// C++ `GlobalVar(name_hint)`.
    pub fn new(name_hint: impl Into<String>) -> Result<Self> {
        Self::new_full(name_hint, unknown_loc()?)
    }

    pub fn new_full(name_hint: impl Into<String>, loc: Location) -> Result<Self> {
        Ok(Self::from_complete_fields(loc, MissingType::new()?.into(), name_hint.into()))
    }
}

// ---------------------------------------------------------------------------------------------
// Expressions

// ---------------------------------------------------------------------------------------------
// Native hooks

/// A C++ `NativeFunctionView` hook of a registry, as the constructors call it: the C ABI adapter
/// (`TVMFFIAny(const Node*)`) its opaque pointer names, or a packed function a frontend
/// registered in its place.
enum NativeHook {
    Adapter(extern "C" fn(*const c_void) -> TVMFFIAny),
    Packed(FfiFunction),
}

impl NativeHook {
    /// The hook `value` holds.
    fn from_any(value: Any) -> Result<Self> {
        if value.type_index() != TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
            return Ok(Self::Packed(value.try_into()?));
        }
        let address: *mut c_void = value.try_into()?;
        // SAFETY: the opaque pointer of a `NativeFunctionView` is its C ABI adapter.
        let adapter = unsafe {
            std::mem::transmute::<*mut c_void, extern "C" fn(*const c_void) -> TVMFFIAny>(address)
        };
        Ok(Self::Adapter(adapter))
    }

    /// The hook `op` registers in the attribute column `column`, if any.
    fn of_op(op: &OpObj, column: &'static str) -> Result<Option<Self>> {
        op.attr_value(column)?.map(Self::from_any).transpose()
    }

    /// C++ `CallExpected(node)`: the hook's result on `node`, or the error it reports.
    fn call_any<T: ObjectRefCore>(&self, node: &T) -> Result<Any>
    where
        for<'a> AnyView<'a>: From<&'a T>,
    {
        let result = match self {
            // SAFETY: `node` keeps the object alive for the call, and the adapter returns an
            // owned value.
            Self::Adapter(adapter) => unsafe {
                let node = ObjectArc::as_raw(ObjectRefCore::data(node)) as *const c_void;
                Any::from_raw_ffi_any(adapter(node))
            },
            Self::Packed(function) => function.call_packed(&[AnyView::from(node)])?,
        };
        if result.type_index() == TVMFFITypeIndex::kTVMFFIError as i32 {
            return Err(result.try_into()?);
        }
        Ok(result)
    }

    /// The hook's result on `node` as an `R`; a packed hook's result of another type is the C++
    /// error.
    fn call<T: ObjectRefCore, R: TryFrom<Any>>(&self, node: &T) -> Result<R>
    where
        for<'a> AnyView<'a>: From<&'a T>,
    {
        R::try_from(self.call_any(node)?).map_err(|_| hook_result_mismatch())
    }

    /// The hook, which returns nothing, on `node`.
    fn call_void<T: ObjectRefCore>(&self, node: &T) -> Result<()>
    where
        for<'a> AnyView<'a>: From<&'a T>,
    {
        let result = self.call_any(node)?;
        match result.type_index() == TVMFFITypeIndex::kTVMFFINone as i32 {
            true => Ok(()),
            false => Err(hook_result_mismatch()),
        }
    }
}

fn hook_result_mismatch() -> tvm_ffi::Error {
    tvm_ffi::Error::new(TYPE_ERROR, "Packed hook result type mismatch", "")
}

/// C++ `op.Validate(call)`: the operator's validator, if it has one, on `call`.
pub(crate) fn validate_call(op: &OpObj, call: &Call) -> Result<()> {
    let validator = op.validator();
    if validator.type_index() == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(());
    }
    NativeHook::from_any(validator.clone())?.call_void(call)
}

/// C++ `Call::ReinferType(call)`: the operator's `TFixedReturnType`, or else what its
/// `FInferType` infers; the return type of a callee of a function type; otherwise `MissingType`.
pub(crate) fn reinfer_call_type(call: &Call) -> Result<Type> {
    let Some(op) = call.op.as_node::<OpObj>() else {
        return match call.op.ty.as_node::<FuncTypeObj>() {
            Some(func_type) => Ok(func_type.ret_type.clone()),
            None => Ok(MissingType::new()?.into()),
        };
    };
    if let Some(fixed) = op.attr_value("TFixedReturnType")? {
        return fixed.try_into();
    }
    match NativeHook::of_op(op, "FInferType")? {
        Some(infer) => infer.call(call),
        None => Ok(MissingType::new()?.into()),
    }
}

impl Call {
    /// C++ `Call(ret_ty, op, args)`: no attrs and no type arguments.
    pub fn new(
        ret_ty: impl Into<Type>,
        op: impl Into<Expr>,
        args: impl IntoIterator<Item = Expr>,
    ) -> Result<Self> {
        let args = args.into_iter().collect();
        Self::new_full(
            Some(ret_ty.into()),
            op.into(),
            args,
            None,
            Array::new(vec![]),
            unknown_loc()?,
        )
    }

    /// C++ `Call(ret_ty, op, args, attrs, ty_args, loc)`: infers the result type
    /// ([`Call::reinfer_type`]) when `ret_ty` is absent. It does not validate the call.
    pub fn new_full(
        ret_ty: Option<Type>,
        op: Expr,
        args: Array<Expr>,
        attrs: Option<Attrs>,
        ty_args: Array<Type>,
        loc: Location,
    ) -> Result<Self> {
        if let Some(ty) = ret_ty {
            return Ok(Self::from_complete_fields(loc, ty, op, args, attrs, ty_args));
        }
        let missing = MissingType::new()?.into();
        let call = Self::from_complete_fields(loc.clone(), missing, op, args, attrs, ty_args);
        let ty = reinfer_call_type(&call)?;
        let (op, args) = (call.op.clone(), call.args.clone());
        Ok(Self::from_complete_fields(loc, ty, op, args, call.attrs.clone(), call.ty_args.clone()))
    }
}

impl TensorLoad {
    /// C++ `TensorLoad(source, indices)`: infers the loaded type.
    pub fn new(source: impl Into<Expr>, indices: impl IntoIterator<Item = Expr>) -> Result<Self> {
        Self::new_full(source, indices, unknown_loc()?)
    }

    /// C++ `TensorLoad(source, indices, loc)`: the type the source's type infers
    /// (`__tvm_tensor_load_infer_type__`), such as a tensor's element type.
    pub fn new_full(
        source: impl Into<Expr>,
        indices: impl IntoIterator<Item = Expr>,
        loc: Location,
    ) -> Result<Self> {
        static INFER: OnceLock<Option<TypeAttrColumn>> = OnceLock::new();
        let source = source.into();
        let indices: Array<Expr> = indices.into_iter().collect();
        for index in &indices {
            prim_dtype(&index)?;
        }
        let ty_index = AnyView::from(&source.ty).type_index();
        let column = *INFER.get_or_init(|| TypeAttrColumn::new("__tvm_tensor_load_infer_type__"));
        let Some(infer) = column.and_then(|column| column.get(ty_index)) else {
            let message = format!("Type {} does not support TensorLoad", type_key(ty_index));
            return Err(check_failed(TYPE_ERROR, "infer != nullptr", &message));
        };
        let load =
            Self::from_complete_fields(loc.clone(), MissingType::new()?.into(), source, indices);
        let ty: PrimType = NativeHook::from_any(infer)?.call(&load)?;
        Ok(Self::from_complete_fields(loc, ty.into(), load.source.clone(), load.indices.clone()))
    }
}

/// What a raw binary constructor checks, and the type it gives the node
/// (`TVM_DEFINE_BINOP_CONSTRUCTOR` and its siblings in `src/ir/prim/expr.cc`).
#[derive(Clone, Copy)]
enum Operands {
    /// `Add`, `Min`, ...: operands of one type, the node's.
    Arith,
    /// `LShift`, `BitwiseAnd`, ...: as `Arith`, of an integer type, or boolean if allowed.
    Bitwise { allow_bool: bool },
    /// `EQ`, `LT`, ...: operands of one type; the node is boolean.
    Compare,
    /// `And`, `Or`: boolean operands of one type; the node is boolean.
    Logical,
}

fn bool_type(lanes: u16) -> Result<Type> {
    Ok(prim_type(DLDataType { code: BOOL, bits: 8, lanes })?.into())
}

fn binary_type(node: &str, operands: Operands, a: &Expr, b: &Expr) -> Result<Type> {
    let (a_ty, b_ty) = (prim_dtype(a)?, prim_dtype(b)?);
    if let Operands::Logical = operands {
        for (ty, condition) in [(a_ty, "a_ty"), (b_ty, "b_ty")] {
            if ty.code != BOOL {
                let condition = format!("{condition}.MatchesCode(DLDataTypeCode::kDLBool)");
                return Err(check_failed(INTERNAL_ERROR, &condition, ""));
            }
        }
        if a_ty != b_ty {
            return Err(check_failed(TYPE_ERROR, "a_ty == b_ty", "mismatched types"));
        }
        return bool_type(a_ty.lanes);
    }
    if a_ty != b_ty {
        let (a_name, b_name) = (a_ty.to_string(), b_ty.to_string());
        let message = format!("mismatched types. {} vs. {}\n", a_name.as_str(), b_name.as_str());
        return Err(check_failed(TYPE_ERROR, "a_ty->dtype == b_ty->dtype", &message));
    }
    match operands {
        Operands::Bitwise { allow_bool } => {
            if matches!(a_ty.code, INT | UINT) || (allow_bool && a_ty.code == BOOL) {
                return Ok(a.ty.clone());
            }
            let condition = format!(
                "a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) || ({allow_bool} \
                 && a.ty().MatchesCode(DLDataTypeCode::kDLBool))"
            );
            let operands = if allow_bool { "integer or boolean" } else { "integer" };
            let message = format!("{node} requires {operands} operands");
            Err(check_failed(TYPE_ERROR, &condition, &message))
        }
        Operands::Compare => bool_type(a_ty.lanes),
        Operands::Arith | Operands::Logical => Ok(a.ty.clone()),
    }
}

/// `$node::new(a, b)` / `new_full(a, b, loc)`: the raw binary constructors, with the C++ checks.
macro_rules! raw_binary {
    ($($node:ident: $operands:expr),+ $(,)?) => {
        $(impl $node {
            #[doc = concat!("C++ `", stringify!($node), "(a, b)`: the raw constructor.")]
            pub fn new(a: impl IntoExpr, b: impl IntoExpr) -> Result<Self> {
                Self::new_full(a, b, unknown_loc()?)
            }

            #[doc = concat!("C++ `", stringify!($node), "(a, b, loc)`.")]
            pub fn new_full(a: impl IntoExpr, b: impl IntoExpr, loc: Location) -> Result<Self> {
                let (a, b) = (a.into_expr()?, b.into_expr()?);
                let ty = binary_type(stringify!($node), $operands, &a, &b)?;
                Ok(Self::from_complete_fields(loc, ty, a, b))
            }
        })+
    };
}

raw_binary!(
    Add: Operands::Arith,
    Sub: Operands::Arith,
    Mul: Operands::Arith,
    Div: Operands::Arith,
    Mod: Operands::Arith,
    FloorDiv: Operands::Arith,
    FloorMod: Operands::Arith,
    Min: Operands::Arith,
    Max: Operands::Arith,
    LShift: Operands::Bitwise { allow_bool: false },
    RShift: Operands::Bitwise { allow_bool: false },
    BitwiseAnd: Operands::Bitwise { allow_bool: true },
    BitwiseOr: Operands::Bitwise { allow_bool: true },
    BitwiseXor: Operands::Bitwise { allow_bool: true },
    EQ: Operands::Compare,
    NE: Operands::Compare,
    LT: Operands::Compare,
    LE: Operands::Compare,
    GT: Operands::Compare,
    GE: Operands::Compare,
    And: Operands::Logical,
    Or: Operands::Logical,
);

impl Not {
    /// C++ `Not(a)`: the raw constructor; `a` is boolean.
    pub fn new(a: impl IntoExpr) -> Result<Self> {
        Self::new_full(a, unknown_loc()?)
    }

    pub fn new_full(a: impl IntoExpr, loc: Location) -> Result<Self> {
        let a = a.into_expr()?;
        let a_ty = prim_dtype(&a)?;
        if a_ty.code != BOOL {
            let condition = "a_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(check_failed(INTERNAL_ERROR, condition, ""));
        }
        Ok(Self::from_complete_fields(loc, bool_type(a_ty.lanes)?, a))
    }
}

impl BitwiseNot {
    /// C++ `BitwiseNot(a)`: the raw constructor; `a` is an integer or boolean.
    pub fn new(a: impl IntoExpr) -> Result<Self> {
        Self::new_full(a, unknown_loc()?)
    }

    pub fn new_full(a: impl IntoExpr, loc: Location) -> Result<Self> {
        let a = a.into_expr()?;
        if !matches!(prim_dtype(&a)?.code, INT | UINT | BOOL) {
            let condition = "a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) || \
                             a_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            let message = "BitwiseNot requires an integer or boolean operand";
            return Err(check_failed(TYPE_ERROR, condition, message));
        }
        Ok(Self::from_complete_fields(loc, a.ty.clone(), a))
    }
}

/// The error of a failed C++ `TVM_FFI_CHECK_EQ(lhs, rhs, Kind)` / `_LE` / ... (`TVM_FFI_ICHECK_*`
/// with `INTERNAL_ERROR`), which prints both sides.
fn compare_failed(
    kind: tvm_ffi::ErrorKind,
    condition: &str,
    lhs: impl Display,
    rhs: impl Display,
    message: &str,
) -> tvm_ffi::Error {
    let message = format!("Check failed: {condition} ({lhs} vs. {rhs}) : {message}");
    tvm_ffi::Error::new(kind, &message, "")
}

impl Shuffle {
    /// C++ `Shuffle(vectors, indices)`: lane `i` of the result is lane `indices[i]` of the
    /// concatenated `vectors`, which share an element type.
    pub fn new(
        vectors: impl IntoIterator<Item = Expr>,
        indices: impl IntoIterator<Item = Expr>,
    ) -> Result<Self> {
        Self::new_full(vectors, indices, unknown_loc()?)
    }

    pub fn new_full(
        vectors: impl IntoIterator<Item = Expr>,
        indices: impl IntoIterator<Item = Expr>,
        loc: Location,
    ) -> Result<Self> {
        let (vectors, indices): (Array<Expr>, Array<Expr>) =
            (vectors.into_iter().collect(), indices.into_iter().collect());
        let Some(first) = vectors.iter().next() else {
            return Err(compare_failed(INTERNAL_ERROR, "vectors.size() != 0U", 0, 0, ""));
        };
        if indices.is_empty() {
            return Err(compare_failed(INTERNAL_ERROR, "indices.size() != 0U", 0, 0, ""));
        }
        let base = element(prim_dtype(&first)?);
        let mut total_lanes = 0;
        for vector in &vectors {
            let dtype = prim_dtype(&vector)?;
            if element(dtype) != base {
                return Err(check_failed(
                    INTERNAL_ERROR,
                    "val_ty.WithLanes(1)->dtype == base_type->dtype",
                    "",
                ));
            }
            total_lanes += lanes(dtype)?;
        }
        if indices.len() > total_lanes as usize {
            let condition = "indices.size() <= static_cast<size_t>(total_lanes)";
            return Err(compare_failed(INTERNAL_ERROR, condition, indices.len(), total_lanes, ""));
        }
        let ty = prim_type(DLDataType { lanes: indices.len() as u16, ..base })?;
        Ok(Self::from_complete_fields(loc, ty.into(), vectors, indices))
    }

    /// C++ `Shuffle::Concat(vectors)`: the lanes of the vectors in order; one vector is itself.
    pub fn concat(vectors: impl IntoIterator<Item = Expr>) -> Result<Expr> {
        let mut vectors: Vec<Expr> = vectors.into_iter().collect();
        if vectors.len() <= 1 {
            return vectors
                .pop()
                .ok_or_else(|| compare_failed(INTERNAL_ERROR, "vectors.size() != 0", 0, 0, ""));
        }
        let mut indices = Vec::new();
        for vector in &vectors {
            for _ in 0..lanes(prim_dtype(vector)?)? {
                indices.push(int32(indices.len() as i64)?.into());
            }
        }
        Ok(Self::new(vectors, indices)?.into())
    }

    /// C++ `Shuffle::ExtractElement(vector, index)`: lane `index` of `vector`.
    pub fn extract_element(vector: impl IntoExpr, index: i32) -> Result<Self> {
        Self::new([vector.into_expr()?], [int32(index as i64)?.into()])
    }
}

impl Tuple {
    /// C++ `Tuple(fields)`: typed `TupleType` of the field types once every field has a type.
    pub fn new(fields: impl IntoIterator<Item = Expr>) -> Result<Self> {
        Self::new_full(fields, unknown_loc()?)
    }

    pub fn new_full(fields: impl IntoIterator<Item = Expr>, loc: Location) -> Result<Self> {
        let fields: Array<Expr> = fields.into_iter().collect();
        let types: Vec<Type> = fields.iter().map(|field| field.ty.clone()).collect();
        let ty: Type = match types.iter().any(|ty| ty.as_node::<MissingTypeObj>().is_some()) {
            true => MissingType::new()?.into(),
            false => TupleType::new(types)?.into(),
        };
        Ok(Self::from_complete_fields(loc, ty, fields))
    }
}

impl LambdaExpr {
    /// C++ `LambdaExpr(vars, body)`: typed as a function from the variables' types to the body's.
    pub fn new(vars: impl IntoIterator<Item = Var>, body: impl Into<Expr>) -> Result<Self> {
        let (vars, body): (Array<Var>, Expr) = (vars.into_iter().collect(), body.into());
        let ty = FuncType::new(vars.iter().map(|var| var.ty.clone()), body.ty.clone())?;
        Ok(Self::from_complete_fields(unknown_loc()?, ty.into(), vars, body))
    }
}

impl Select {
    /// C++ `Select(condition, true_value, false_value)`: the raw constructor. The condition is
    /// boolean, a scalar or of the values' lanes, and the values are of one type.
    pub fn new(
        condition: impl IntoExpr,
        true_value: impl IntoExpr,
        false_value: impl IntoExpr,
    ) -> Result<Self> {
        Self::new_full(condition, true_value, false_value, unknown_loc()?)
    }

    pub fn new_full(
        condition: impl IntoExpr,
        true_value: impl IntoExpr,
        false_value: impl IntoExpr,
        loc: Location,
    ) -> Result<Self> {
        let (c, t, f) = (condition.into_expr()?, true_value.into_expr()?, false_value.into_expr()?);
        let (condition_ty, true_ty, false_ty) = (prim_dtype(&c)?, prim_dtype(&t)?, prim_dtype(&f)?);
        if condition_ty.code != BOOL {
            let condition = "condition_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(check_failed(INTERNAL_ERROR, condition, ""));
        }
        let lanes_match = lanes_or_vscale_factor(condition_ty)? == lanes_or_vscale_factor(true_ty)?;
        if !lanes_match && condition_ty.lanes != 1 {
            let condition =
                "GetLanesOrVScaleFactor(condition_ty) == GetLanesOrVScaleFactor(true_ty) \
                             || condition_ty.IsScalar()";
            return Err(check_failed(INTERNAL_ERROR, condition, ""));
        }
        if false_ty != true_ty {
            let (false_name, true_name) = (false_ty.to_string(), true_ty.to_string());
            let message = format!(
                "mismatched types. False type: {}; True type: {}",
                false_name.as_str(),
                true_name.as_str()
            );
            return Err(check_failed(TYPE_ERROR, "false_ty == true_ty", &message));
        }
        Ok(Self::from_complete_fields(loc, t.ty.clone(), c, t, f))
    }
}

/// C++ `GetLanesOrVScaleFactor(ty)`: the vscale factor of a scalable vector, else the lanes.
fn lanes_or_vscale_factor(dtype: DLDataType) -> Result<i32> {
    match (dtype.lanes as i16) < -1 {
        true => Ok(-(dtype.lanes as i16) as i32),
        false => lanes(dtype),
    }
}

impl Let {
    /// C++ `Let(var, value, body)`: the raw constructor; the value has the variable's type.
    pub fn new(var: Var, value: impl IntoExpr, body: impl IntoExpr) -> Result<Self> {
        Self::new_full(var, value, body, unknown_loc()?)
    }

    pub fn new_full(
        var: Var,
        value: impl IntoExpr,
        body: impl IntoExpr,
        loc: Location,
    ) -> Result<Self> {
        let (value, body) = (value.into_expr()?, body.into_expr()?);
        let (value_ty, var_ty) = (prim_dtype(&value)?, prim_dtype(&var.clone().into())?);
        prim_dtype(&body)?;
        if value_ty != var_ty {
            let condition = "value.ty() == var->ty.as_or_throw<PrimType>()";
            return Err(check_failed(INTERNAL_ERROR, condition, ""));
        }
        Ok(Self::from_complete_fields(loc, body.ty.clone(), var, value, body))
    }
}

impl Cast {
    /// C++ `Cast(value, ty)`: the raw constructor; the value has the lanes of `ty`.
    pub fn new(value: impl IntoExpr, ty: PrimType) -> Result<Self> {
        Self::new_full(value, ty, unknown_loc()?)
    }

    pub fn new_full(value: impl IntoExpr, ty: PrimType, loc: Location) -> Result<Self> {
        let value = value.into_expr()?;
        let value_lanes = prim_dtype(&value)?.lanes;
        if ty.dtype.lanes != value_lanes {
            let condition = "value_ty->dtype.lanes == value_expr_ty->dtype.lanes";
            return Err(compare_failed(INTERNAL_ERROR, condition, ty.dtype.lanes, value_lanes, ""));
        }
        Ok(Self::from_complete_fields(loc, ty.into(), value))
    }
}

impl Broadcast {
    /// C++ `Broadcast(value, lanes)`: the raw constructor of a scalar `value` repeated `lanes`
    /// times. Constant lanes are an int32 `IntImm`, scalable ones `vscale() * factor`.
    pub fn new(value: impl IntoExpr, lanes: impl IntoExpr) -> Result<Self> {
        Self::new_full(value, lanes, unknown_loc()?)
    }

    pub fn new_full(value: impl IntoExpr, lanes: impl IntoExpr, loc: Location) -> Result<Self> {
        let (value, lanes) = (value.into_expr()?, lanes.into_expr()?);
        let value_ty = prim_dtype(&value)?;
        if value_ty.lanes != 1 {
            return Err(check_failed(INTERNAL_ERROR, "value_ty.IsScalar()", ""));
        }
        let (ty, lanes) = vector_of(value_ty, &lanes)?;
        Ok(Self::from_complete_fields(loc, ty, value, lanes))
    }
}

impl Ramp {
    /// C++ `Ramp(base, stride, lanes)`: the raw constructor; a stride of another type is cast to
    /// the base's. Lanes are as for [`Broadcast::new`].
    pub fn new(base: impl IntoExpr, stride: impl IntoExpr, lanes: impl IntoExpr) -> Result<Self> {
        Self::new_full(base, stride, lanes, unknown_loc()?)
    }

    pub fn new_full(
        base: impl IntoExpr,
        stride: impl IntoExpr,
        lanes: impl IntoExpr,
        loc: Location,
    ) -> Result<Self> {
        let (base, stride, lanes) = (base.into_expr()?, stride.into_expr()?, lanes.into_expr()?);
        let (base_ty, stride_ty) = (prim_dtype(&base)?, prim_dtype(&stride)?);
        if base_ty.lanes != 1 {
            return Err(check_failed(INTERNAL_ERROR, "base_ty.IsScalar()", ""));
        }
        if stride_ty.lanes != 1 {
            return Err(check_failed(INTERNAL_ERROR, "stride_ty.IsScalar()", ""));
        }
        let stride = match stride_ty == base_ty {
            true => stride,
            false => super::math::cast(stride, base.ty.clone().try_cast()?, &unknown_loc()?)?,
        };
        let (ty, lanes) = vector_of(base_ty, &lanes)?;
        Ok(Self::from_complete_fields(loc, ty, base, stride, lanes))
    }
}

/// The type and the lanes of a `Broadcast` or `Ramp` of `element` over `lanes`: constant lanes
/// as an int32 `IntImm` above 1, scalable lanes (`factor * vscale()` either way round) as
/// `vscale() * factor`.
fn vector_of(element: DLDataType, lanes: &Expr) -> Result<(Type, Expr)> {
    if let Some(imm) = lanes.as_node::<IntImmObj>() {
        let Some(count) = imm.value.to_i64().and_then(|value| i32::try_from(value).ok()) else {
            return Err(tvm_ffi::Error::new(INTERNAL_ERROR, "bad optional access", ""));
        };
        if count <= 1 {
            return Err(compare_failed(INTERNAL_ERROR, "lanes > 1", count, 1, ""));
        }
        let ty = prim_type(DLDataType { lanes: count as u16, ..element })?;
        return Ok((ty.into(), int32(count as i64)?.into()));
    }
    let Some(factor) = vscale_factor_of(lanes) else {
        let message = format!("Invalid expression for scalable lanes {}", repr(lanes));
        return Err(check_failed(INTERNAL_ERROR, "vscale_factor", &message));
    };
    let ty = prim_type(DLDataType { lanes: -(factor as i16) as u16, ..element })?;
    let vscale = Call::new(prim_type(int32_dtype())?, op("prim.vscale")?, [])?;
    Ok((ty.into(), Mul::new(vscale, factor)?.into()))
}

/// C++ `ExtractVscaleFactor(lanes)`: `factor` of `factor * vscale()` or `vscale() * factor`.
fn vscale_factor_of(lanes: &Expr) -> Option<i32> {
    let mul = lanes.as_node::<MulObj>()?;
    let is_vscale = |expr: &Expr| {
        expr.as_node::<CallObj>()
            .is_some_and(|call| op("prim.vscale").is_ok_and(|vscale| call.op.same_as(&vscale)))
    };
    let factor = |imm: &Expr, other: &Expr| match is_vscale(other) {
        true => imm.as_node::<IntImmObj>().map(|imm| imm.value.to_i64()),
        false => None,
    };
    let value = factor(&mul.a, &mul.b).or_else(|| factor(&mul.b, &mul.a))??;
    i32::try_from(value).ok()
}

// ---------------------------------------------------------------------------------------------
// Statements

impl SeqStmt {
    /// C++ `SeqStmt(seq)`: flattens nested sequences and drops no-op statements.
    pub fn new(seq: impl IntoIterator<Item = Stmt>) -> Result<Self> {
        Ok(seq_stmt(seq.into_iter().collect(), unknown_loc()?))
    }

    /// C++ `SeqStmt(seq, loc)`.
    pub fn new_full(seq: impl IntoIterator<Item = Stmt>, loc: Location) -> Result<Self> {
        Ok(seq_stmt(seq.into_iter().collect(), loc))
    }

    /// The C++ implicit `Stmt -> SeqStmt` conversion (`SeqStmt(stmt)`): a `SeqStmt` passes
    /// through, any other statement is wrapped with its own location.
    pub fn from_stmt(stmt: impl Into<Stmt>) -> Result<Self> {
        let stmt = stmt.into();
        if stmt.as_node::<SeqStmtObj>().is_some() {
            return stmt.try_cast();
        }
        let loc = stmt.loc.clone();
        Ok(seq_stmt(Array::new(vec![stmt]), loc))
    }
}

/// C++ `SeqStmt(seq, loc)`: nested sequences are spliced in and `Evaluate(0)` no-ops dropped.
fn seq_stmt(seq: Array<Stmt>, loc: Location) -> SeqStmt {
    fn append(stmt: Stmt, flattened: &mut Vec<Stmt>) {
        match stmt.as_node::<SeqStmtObj>() {
            Some(nested) => nested.seq.iter().for_each(|child| append(child, flattened)),
            None if is_no_op(&stmt) => {}
            None => flattened.push(stmt),
        }
    }
    let flatten = seq.iter().any(|stmt| stmt.as_node::<SeqStmtObj>().is_some() || is_no_op(&stmt));
    if !flatten {
        return SeqStmt::from_complete_fields(loc, seq);
    }
    let mut flattened = Vec::with_capacity(seq.len());
    seq.iter().for_each(|stmt| append(stmt, &mut flattened));
    SeqStmt::from_complete_fields(loc, Array::new(flattened))
}

/// C++ `detail::IsSeqStmtNoOp`: `Evaluate` of an integer 0.
fn is_no_op(stmt: &Stmt) -> bool {
    let value = stmt.as_node::<EvaluateObj>().map(|evaluate| &evaluate.value);
    value.is_some_and(|value| is_const_int(value, 0))
}

impl Evaluate {
    /// C++ `Evaluate(value)`; an `i32` is `IntImm::Int32`, as in C++ `Evaluate(0)`. Runs the
    /// validation registered for the value's type (`__tvm_evaluate_validate__`).
    pub fn new(value: impl IntoExpr) -> Result<Self> {
        Self::new_full(value, unknown_loc()?)
    }

    pub fn new_full(value: impl IntoExpr, loc: Location) -> Result<Self> {
        static VALIDATE: OnceLock<Option<TypeAttrColumn>> = OnceLock::new();
        let value = value.into_expr()?;
        let ty_index = AnyView::from(&value.ty).type_index();
        let column = *VALIDATE.get_or_init(|| TypeAttrColumn::new("__tvm_evaluate_validate__"));
        if let Some(validate) = column.and_then(|column| column.get(ty_index)) {
            FfiFunction::try_from(validate)?.call_tuple((value.clone(),))?;
        }
        Ok(Self::from_complete_fields(loc, value))
    }
}

impl Return {
    /// C++ `Return(value)`; an `i32` is `IntImm::Int32`, as in C++ `Return(0)`.
    pub fn new(value: impl IntoExpr) -> Result<Self> {
        Self::new_full(value, unknown_loc()?)
    }

    pub fn new_full(value: impl IntoExpr, loc: Location) -> Result<Self> {
        Ok(Self::from_complete_fields(loc, value.into_expr()?))
    }
}

impl TensorStore {
    /// C++ `TensorStore(dest, indices, value)`: runs the validation registered for the
    /// destination's type (`__tvm_tensor_store_validate__`), such as the tensor rank and element
    /// type checks of a `TensorType`.
    pub fn new(
        dest: impl Into<Expr>,
        indices: impl IntoIterator<Item = Expr>,
        value: impl IntoExpr,
    ) -> Result<Self> {
        Self::new_full(dest, indices, value, unknown_loc()?)
    }

    pub fn new_full(
        dest: impl Into<Expr>,
        indices: impl IntoIterator<Item = Expr>,
        value: impl IntoExpr,
        loc: Location,
    ) -> Result<Self> {
        static VALIDATE: OnceLock<Option<TypeAttrColumn>> = OnceLock::new();
        let (dest, value) = (dest.into(), value.into_expr()?);
        let indices: Array<Expr> = indices.into_iter().collect();
        prim_dtype(&value)?;
        for index in &indices {
            prim_dtype(&index)?;
        }
        let ty_index = AnyView::from(&dest.ty).type_index();
        let column = *VALIDATE.get_or_init(|| TypeAttrColumn::new("__tvm_tensor_store_validate__"));
        let Some(validate) = column.and_then(|column| column.get(ty_index)) else {
            let message = format!("Type {} does not support TensorStore", type_key(ty_index));
            return Err(check_failed(TYPE_ERROR, "validate != nullptr", &message));
        };
        FfiFunction::try_from(validate)?.call_tuple((
            dest.clone(),
            indices.clone(),
            value.clone(),
        ))?;
        Ok(Self::from_complete_fields(loc, dest, indices, value))
    }
}

impl Bind {
    /// C++ `Bind(var, value)`: checks that the value has the variable's type.
    pub fn new(var: Var, value: impl IntoExpr) -> Result<Self> {
        Self::new_full(var, value, unknown_loc()?)
    }

    pub fn new_full(var: Var, value: impl IntoExpr, loc: Location) -> Result<Self> {
        let value = value.into_expr()?;
        if !structural_equal(&value.ty, &var.ty)? {
            let condition = "ffi::StructuralEqual()(value->ty, var->ty)";
            return Err(check_failed(INTERNAL_ERROR, condition, ""));
        }
        Ok(Self::from_complete_fields(loc, var, value))
    }
}

impl AssertStmt {
    /// C++ `AssertStmt(condition, error_kind, message_parts)`: checks for a boolean condition.
    pub fn new(
        condition: impl IntoExpr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Result<Self> {
        Self::new_full(condition, error_kind, message_parts, unknown_loc()?)
    }

    pub fn new_full(
        condition: impl IntoExpr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
        loc: Location,
    ) -> Result<Self> {
        let condition = condition.into_expr()?;
        if prim_dtype(&condition)?.code != BOOL {
            let message = format!(
                "AssertStmt should have boolean condition, but received {} with dtype {}",
                repr(&condition),
                repr(&condition.ty)
            );
            let check = "condition_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(check_failed(INTERNAL_ERROR, check, &message));
        }
        Ok(Self::from_complete_fields(loc, condition, error_kind, message_parts))
    }
}

impl If {
    /// C++ `If(condition, then_case, else_case)`; either arm may be any statement.
    pub fn new(
        condition: impl IntoExpr,
        then_case: impl IntoBody,
        else_case: Option<SeqStmt>,
    ) -> Result<Self> {
        Self::new_full(condition, then_case, else_case, unknown_loc()?)
    }

    pub fn new_full(
        condition: impl IntoExpr,
        then_case: impl IntoBody,
        else_case: Option<SeqStmt>,
        loc: Location,
    ) -> Result<Self> {
        let condition = condition.into_expr()?;
        prim_dtype(&condition)?;
        Ok(Self::from_complete_fields(loc, condition, then_case.into_body()?, else_case))
    }
}

impl For {
    /// C++ `For(loop_var, min, extent, kind, body)`: no annotations and no step. Checks for
    /// integer bounds and promotes an `IntImm` bound to the loop variable's dtype.
    pub fn new(
        loop_var: Var,
        min: impl IntoExpr,
        extent: impl IntoExpr,
        kind: i32,
        body: impl IntoBody,
    ) -> Result<Self> {
        Self::new_full(loop_var, min, extent, kind, body, Map::new(), None, unknown_loc()?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        loop_var: Var,
        min: impl IntoExpr,
        extent: impl IntoExpr,
        kind: i32,
        body: impl IntoBody,
        annotations: Map<String, Any>,
        step: Option<Expr>,
        loc: Location,
    ) -> Result<Self> {
        let (min, extent, body) = (min.into_expr()?, extent.into_expr()?, body.into_body()?);
        let loop_expr: Expr = loop_var.clone().into();
        require_scalar_int(&loop_expr, "loop_var")?;
        require_scalar_int(&min, "min")?;
        require_scalar_int(&extent, "extent")?;
        let loop_ty = &loop_var.ty;
        let min = promote_to_loop_type(min, loop_ty)?;
        let extent = promote_to_loop_type(extent, loop_ty)?;
        let same_type = |expr: &Expr, condition: &str| -> Result<()> {
            if prim_dtype(expr)? != prim_dtype(&loop_expr)? {
                let message = format!("{} vs {}", repr(loop_ty), repr(&expr.ty));
                return Err(check_failed(INTERNAL_ERROR, condition, &message));
            }
            Ok(())
        };
        same_type(&min, "loop_var.ty() == min.ty()")?;
        same_type(&extent, "loop_var.ty() == extent.ty()")?;
        let step = match step {
            Some(step) => {
                require_scalar_int(&step, "step")?;
                let step = promote_to_loop_type(step, loop_ty)?;
                same_type(&step, "loop_var.ty() == step.value().ty()")?;
                Some(step)
            }
            None => None,
        };
        Ok(Self::from_complete_fields(loc, loop_var, min, extent, kind, body, annotations, step))
    }
}

/// The check of a `For` field (`field`) for a scalar integer.
fn require_scalar_int(expr: &Expr, field: &str) -> Result<()> {
    let dtype = prim_dtype(expr)?;
    if dtype.lanes == 1 && matches!(dtype.code, INT | UINT) {
        return Ok(());
    }
    let condition = "dtype.IsScalar() && (dtype.MatchesCode(DLDataTypeCode::kDLUInt, \
                     DLDataTypeCode::kDLInt))";
    let message = format!(
        "TIR For nodes require a scalar integer as the {field}, but received {} with dtype {}",
        repr(expr),
        repr(&expr.ty)
    );
    Err(check_failed(INTERNAL_ERROR, condition, &message))
}

/// A `For` bound or step of another type than the loop variable (`loop_ty`): a constant is
/// rebuilt in that type if its value fits, anything else must be no wider.
fn promote_to_loop_type(expr: Expr, loop_ty: &Type) -> Result<Expr> {
    let Some(loop_dtype) = loop_ty.as_node::<PrimTypeObj>().map(|ty| ty.dtype) else {
        return Err(tvm_ffi::Error::new(TYPE_ERROR, "expected a PrimType expression", ""));
    };
    let dtype = prim_dtype(&expr)?;
    if dtype == loop_dtype {
        return Ok(expr);
    }
    if let Some(imm) = expr.as_node::<IntImmObj>() {
        if !int_imm_value_fits(&imm.value, loop_dtype)? {
            let message = format!(
                "Literal value {} is not representable in the loop variable's dtype ({})",
                imm.value,
                repr(loop_ty)
            );
            return Err(check_failed(
                INTERNAL_ERROR,
                "IntImmValueFits(a->value, loop_var_ty)",
                &message,
            ));
        }
        check_int_imm(loop_dtype, &imm.value)?;
        return Ok(IntImm::new(unknown_loc()?, loop_ty.clone(), imm.value.clone()).into());
    }
    if dtype.bits > loop_dtype.bits {
        let message = format!(
            " Loop variable's dtype ({}) is narrower than that of `min` or `extent` ({})",
            repr(loop_ty),
            repr(&expr.ty)
        );
        return Err(check_failed(INTERNAL_ERROR, "e_ty.bits() <= loop_var_ty.bits()", &message));
    }
    Ok(expr)
}

/// C++ `IntImmValueFits(value, ty)`: `value` is in the range of the integer type.
fn int_imm_value_fits(value: &BigInt, dtype: DLDataType) -> Result<bool> {
    let bits = dtype.bits as i64;
    let one = BigInt::from(1i64);
    Ok(match dtype.code == UINT {
        true => *value >= BigInt::from(0i64) && *value < one.try_shl(bits)?,
        false => {
            let limit = one.try_shl(bits - 1)?;
            *value >= -limit.clone() && *value < limit
        }
    })
}

impl RegionStmt {
    /// C++ `RegionStmt(op, args, body_params, attrs, body)`: no result variables. Checks the body
    /// parameters against the operation's contract and runs its `FRegionValidate`.
    pub fn new(
        op: Op,
        args: Array<Expr>,
        body_params: Array<Var>,
        attrs: DictAttrs,
        body: impl IntoBody,
    ) -> Result<Self> {
        Self::new_full(op, args, body_params, attrs, body, Array::new(vec![]), unknown_loc()?)
    }

    /// C++ `RegionStmt(op, args, body_params, attrs, body, result_vars, loc)`: the body
    /// parameters are as many as, and typed as, those the operation's `FRegionGetBodyParams`
    /// gives; they and the results are distinct definitions.
    pub fn new_full(
        op: Op,
        args: Array<Expr>,
        body_params: Array<Var>,
        attrs: DictAttrs,
        body: impl IntoBody,
        result_vars: Array<Var>,
        loc: Location,
    ) -> Result<Self> {
        let body = body.into_body()?;
        let name = op.name.as_str().to_string();
        let expected = region_body_params(&op, &args, &attrs)?;
        if body_params.len() != expected.len() {
            let message = format!("{name} expects {} body parameters", expected.len());
            let condition = "body_params.size() == expected_params.size()";
            let (actual, expected) = (body_params.len(), expected.len());
            return Err(compare_failed(VALUE_ERROR, condition, actual, expected, &message));
        }
        for (i, (param, contract)) in body_params.iter().zip(expected.iter()).enumerate() {
            if !structural_equal(&param.ty, &contract.ty)? {
                let condition =
                    "ffi::StructuralEqual()(body_params[i]->ty, expected_params[i]->ty)";
                let message =
                    format!("{name} body parameter {i} has a type inconsistent with its contract");
                return Err(check_failed(VALUE_ERROR, condition, &message));
            }
        }
        let mut definitions = HashSet::new();
        if !body_params
            .iter()
            .chain(result_vars.iter())
            .all(|var| definitions.insert(ObjectIdentity::of(&var)))
        {
            let message = "RegionStmt parameters and results must be distinct definitions";
            return Err(check_failed(VALUE_ERROR, "definitions.insert(var.get()).second", message));
        }
        let region =
            Self::from_complete_fields(loc, op, args, body_params, attrs, body, result_vars);
        if let Some(validate) = NativeHook::of_op(&region.op, "FRegionValidate")? {
            validate.call_void(&region)?;
        }
        Ok(region)
    }
}

/// C++ `GetRegionBodyParams(op, args, attrs)`: the variables `FRegionGetBodyParams` gives for a
/// call to `op` of `args` and `attrs`, which the operator's validator checks first.
fn region_body_params(op: &Op, args: &Array<Expr>, attrs: &DictAttrs) -> Result<Array<Var>> {
    let name = op.name.as_str();
    let Some(get_body_params) = NativeHook::of_op(op, "FRegionGetBodyParams")? else {
        let message = format!(
            "{name} does not support region construction: FRegionGetBodyParams is required"
        );
        return Err(check_failed(VALUE_ERROR, "IsRegionOp(op)", &message));
    };
    let attrs: Attrs = attrs.clone().into();
    let signature = Call::from_complete_fields(
        unknown_loc()?,
        MissingType::new()?.into(),
        op.clone().into(),
        args.clone(),
        Some(attrs),
        Array::new(vec![]),
    );
    validate_call(op, &signature)?;
    let params: Array<Var> = get_body_params.call(&signature)?;
    let mut definitions = HashSet::new();
    for param in &params {
        if param.ty.as_node::<MissingTypeObj>().is_some() {
            let message = format!("FRegionGetBodyParams for {name} must return typed variables");
            let condition = "!param->ty.as<MissingType>().has_value()";
            return Err(check_failed(VALUE_ERROR, condition, &message));
        }
        if !definitions.insert(ObjectIdentity::of(&param)) {
            let message =
                format!("FRegionGetBodyParams for {name} must return distinct definitions");
            return Err(check_failed(
                VALUE_ERROR,
                "definitions.insert(param.get()).second",
                &message,
            ));
        }
    }
    Ok(params)
}

impl Function {
    /// C++ `Function(params, body)`: a void function without attributes. Derives `ty` from the
    /// signature.
    pub fn new(params: Array<Var>, body: Option<SeqStmt>) -> Result<Self> {
        let attrs = DictAttrs::new(Map::new());
        Self::new_full(params, body, MissingType::new()?.into(), attrs, unknown_loc()?)
    }

    /// C++ `Function(params, body, ret_type, attrs, loc)`; a missing return type becomes void
    /// (the empty tuple type), and `ty` is the function type of the parameters' types.
    pub fn new_full(
        params: Array<Var>,
        body: Option<SeqStmt>,
        ret_type: Type,
        attrs: DictAttrs,
        loc: Location,
    ) -> Result<Self> {
        let ret_type = match ret_type.as_node::<MissingTypeObj>() {
            Some(_) => TupleType::new([])?.into(),
            None => ret_type,
        };
        let ty = FuncType::new(params.iter().map(|param| param.ty.clone()), ret_type.clone())?;
        Ok(Self::from_complete_fields(loc, ty.into(), attrs, params, ret_type, body))
    }
}

// ---------------------------------------------------------------------------------------------
// Objects whose state Rust cannot see: these call the native constructor.

impl ExecScope {
    /// C++ `ExecScope(backend, name)`: an error unless `backend.name` is a registered execution
    /// scope (a C++ registry).
    pub fn new(backend: impl Into<String>, name: impl Into<String>) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.ExecScope")
            .call_tuple((backend.into(), name.into()))?
            .try_into()
    }
}

impl TileLayout {
    /// C++ `TileLayout(shard, replica, offset)`. `TileLayout` is opaque to Rust.
    pub fn new(
        shard: impl IntoIterator<Item = Iter>,
        replica: impl IntoIterator<Item = Iter>,
        offset: Map<Axis, Expr>,
    ) -> Result<Self> {
        let (shard, replica): (Array<Iter>, Array<Iter>) =
            (shard.into_iter().collect(), replica.into_iter().collect());
        tvm_ffi::cached_global_func!("tirx.TileLayout")
            .call_tuple((shard, replica, offset))?
            .try_into()
    }
}

impl DispatchContext {
    /// C++ `DispatchContext(target, exec_scope)`: no launch parameters, ranges, callbacks or
    /// state. `DispatchContext` is opaque to Rust.
    pub fn new(target: Target, exec_scope: ExecScope) -> Result<Self> {
        let (launch_params, var_range_map) = (Map::new(), Map::new());
        let (callbacks, shared_state, inter, intra) =
            (Map::new(), Map::new(), Map::new(), Map::new());
        Self::new_full(
            target,
            exec_scope,
            launch_params,
            var_range_map,
            false,
            callbacks,
            shared_state,
            inter,
            intra,
            "",
        )
    }

    /// C++ `DispatchContext(target, exec_scope, launch_params, var_range_map, alloc_only,
    /// callbacks, shared_state, inter, intra, scope_kind)`. A launch parameter is the pair
    /// `[var, extent]`; its extent is a signed integer of the variable's type.
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        target: Target,
        exec_scope: ExecScope,
        launch_params: Map<String, Array<Expr>>,
        var_range_map: Map<Var, Range>,
        alloc_only: bool,
        callbacks: Map<String, ObjectRef>,
        shared_state: Map<String, ObjectRef>,
        inter: Map<String, Array<Expr>>,
        intra: Map<String, Array<Expr>>,
        scope_kind: impl Into<String>,
    ) -> Result<Self> {
        let args = [
            Any::from(target),
            Any::from(exec_scope),
            Any::from(launch_params),
            Any::from(var_range_map),
            Any::from(alloc_only),
            Any::from(callbacks),
            Any::from(shared_state),
            Any::from(inter),
            Any::from(intra),
            Any::from(scope_kind.into()),
        ];
        let args: Vec<AnyView> = args.iter().map(AnyView::from).collect();
        tvm_ffi::cached_global_func!("tirx.DispatchContext").call_packed(&args)?.try_into()
    }
}

impl UniqueNameSupply {
    /// C++ `UniqueNameSupply()`: a supply of fresh names without a prefix. `UniqueNameSupply` is
    /// opaque to Rust.
    pub fn new() -> Result<Self> {
        Self::new_full("")
    }

    /// C++ `UniqueNameSupply(prefix)`.
    pub fn new_full(prefix: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupply")
            .call_tuple((String::from(prefix),))?
            .try_into()
    }
}
