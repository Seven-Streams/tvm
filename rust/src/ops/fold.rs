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

//! Constant folding of the operators, after `TryConstFold<Op>` (`src/ir/prim/const_fold.h`).
//!
//! Each function takes operands of one type, as `BinaryOpMatchTypes` leaves them, and returns the
//! folded expression, or `None` when there is nothing to fold. Integers fold exactly and wrap to
//! the width of their type (`GetFoldResult`); floats fold for 32 and 64 bits, a float32 in single
//! precision; a constant operand also folds the identities (`x + 0`, `x * 1`, ...). Division by a
//! constant zero is an error, as in C++.

use std::fmt::Display;

use tvm_ffi::{BigInt, DLDataType, Error, ObjectRefCore, Result, INTERNAL_ERROR};

use super::build::{big_int_imm, float_imm};
use super::math::{check_failed, BOOL, INT, UINT};
use crate::ir::{Expr, FloatImmObj, IntImmObj};

type Folded = Result<Option<Expr>>;

// ---------------------------------------------------------------------------------------------
// Helpers

/// `IsIndexType`: a scalar `int32` or `int64`, which index arithmetic folds eagerly.
pub(super) fn is_index_type(dtype: DLDataType) -> bool {
    dtype.code == INT && (dtype.bits == 32 || dtype.bits == 64) && dtype.lanes == 1
}

/// `GetFoldResult(value, dtype)`: exact integer arithmetic brought to the width of `dtype`,
/// wrapping as two's complement (as unsigned for an unsigned type).
pub(super) fn fold_result(value: BigInt, dtype: DLDataType) -> BigInt {
    let signed = dtype.code == INT;
    let bits = dtype.bits as u32;
    if let Some(small) = value.to_i64() {
        if bits >= 64 && (signed || small >= 0) {
            return value;
        }
        if bits < 64 {
            let mut result = ((small as u64) & ((1u64 << bits) - 1)) as i64;
            if signed {
                let sign_bit = 1i64 << (bits - 1);
                result = (result ^ sign_bit) - sign_bit;
            }
            return BigInt::from(result);
        }
    }
    let modulus = BigInt::from(1) << bits as i64;
    let mut value = value & (&modulus - 1);
    if signed && value >= (&modulus >> 1) {
        value -= modulus;
    }
    value
}

/// The value of an integer constant.
fn int(expr: &Expr) -> Option<&BigInt> {
    expr.as_node::<IntImmObj>().map(|imm| &imm.value)
}

/// The value of a float constant.
fn float(expr: &Expr) -> Option<f64> {
    expr.as_node::<FloatImmObj>().map(|imm| imm.value)
}

fn int_result(dtype: DLDataType, value: BigInt) -> Folded {
    Ok(Some(big_int_imm(dtype, value)?.into()))
}

/// `IntImm(result_ty, GetFoldResult(value, result_ty))`.
fn wrapped(dtype: DLDataType, value: BigInt) -> Folded {
    int_result(dtype, fold_result(value, dtype))
}

fn float_result(dtype: DLDataType, value: f64) -> Folded {
    Ok(Some(float_imm(dtype, value)?.into()))
}

/// `IntImm::Bool(value)`.
fn bool_result(value: bool) -> Folded {
    int_result(DLDataType { code: BOOL, bits: 8, lanes: 1 }, BigInt::from(value))
}

/// A float32 operation folded in single precision, as C++ computes `float` arithmetic
/// (`GetFoldResultDoubleRepr`, whose overflow warning is not reproduced).
fn single(x: f64, y: f64, op: fn(f32, f32) -> f32) -> f64 {
    op(x as f32, y as f32) as f64
}

/// The error of `TVM_FFI_ICHECK_NE(<operand>, 0) << "Divide by zero"`.
fn divide_by_zero(operand: &str, value: impl Display) -> Error {
    let message = format!("Check failed: {operand} != 0 ({value} vs. 0) : Divide by zero");
    Error::new(INTERNAL_ERROR, &message, "")
}

fn check_int_divisor(divisor: &BigInt) -> Result<()> {
    match divisor.is_zero() {
        true => Err(divide_by_zero("pb->value", divisor)),
        false => Ok(()),
    }
}

fn check_float_divisor(divisor: f64) -> Result<()> {
    match divisor == 0.0 {
        true => Err(divide_by_zero("fb->value", divisor)),
        false => Ok(()),
    }
}

fn is_one(value: &BigInt) -> bool {
    value.to_i64() == Some(1)
}

/// `a` itself.
fn keep(expr: &Expr) -> Folded {
    Ok(Some(expr.clone()))
}

// ---------------------------------------------------------------------------------------------
// Arithmetic

/// `TryConstFold<Add>`.
pub(super) fn add(a: &Expr, b: &Expr) -> Folded {
    let dtype = a.prim_dtype()?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        return wrapped(dtype, x + y);
    }
    if pa.is_some_and(BigInt::is_zero) {
        return keep(b);
    }
    if pb.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        match dtype.bits {
            32 => return float_result(dtype, single(x, y, |x, y| x + y)),
            64 => return float_result(dtype, x + y),
            _ => {}
        }
    }
    if fa == Some(0.0) {
        return keep(b);
    }
    if fb == Some(0.0) {
        return keep(a);
    }
    Ok(None)
}

/// `TryConstFold<Sub>`: an unsigned `0 - n` is an error, as it would be negative.
pub(super) fn sub(a: &Expr, b: &Expr) -> Folded {
    let dtype = a.prim_dtype()?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    let unsigned_zero = pa.is_some_and(BigInt::is_zero) && dtype.code == UINT;
    if unsigned_zero && pb.is_some_and(|y| !y.is_zero() && !y.is_negative()) && dtype.code == UINT {
        let condition =
            "!((pa && pa->ty.as_or_throw<PrimType>().MatchesCode(DLDataTypeCode::kDLUInt) \
                         && pa->value == 0U) && (pb && \
                         pb->ty.as_or_throw<PrimType>().MatchesCode(DLDataTypeCode::kDLUInt) && \
                         pb->value > 0U))";
        let message = "Checked failed. Minuend 's value is 0U and it's dtype is uint while \
                       Subtrahend's dtype is uint; which will cause a negative uint";
        return Err(check_failed(INTERNAL_ERROR, condition, message));
    }
    if let (Some(x), Some(y)) = (pa, pb) {
        return wrapped(dtype, x - y);
    }
    if pb.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        match dtype.bits {
            32 => return float_result(dtype, single(x, y, |x, y| x - y)),
            64 => return float_result(dtype, x - y),
            _ => {}
        }
    }
    if fb == Some(0.0) {
        return keep(a);
    }
    Ok(None)
}

/// `TryConstFold<Mul>`.
pub(super) fn mul(a: &Expr, b: &Expr) -> Folded {
    let dtype = a.prim_dtype()?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        return wrapped(dtype, x * y);
    }
    if let Some(x) = pa {
        if is_one(x) {
            return keep(b);
        }
        if x.is_zero() {
            return keep(a);
        }
    }
    if let Some(y) = pb {
        if is_one(y) {
            return keep(a);
        }
        if y.is_zero() {
            return keep(b);
        }
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        match dtype.bits {
            32 => return float_result(dtype, single(x, y, |x, y| x * y)),
            64 => return float_result(dtype, x * y),
            _ => {}
        }
    }
    if let Some(x) = fa {
        if x == 1.0 {
            return keep(b);
        }
        if x == 0.0 {
            return keep(a);
        }
    }
    if let Some(y) = fb {
        if y == 1.0 {
            return keep(a);
        }
        if y == 0.0 {
            return keep(b);
        }
    }
    Ok(None)
}

/// `TryConstFold<Div>`: integers divide truncating.
pub(super) fn div(a: &Expr, b: &Expr) -> Folded {
    let dtype = a.prim_dtype()?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        check_int_divisor(y)?;
        return wrapped(dtype, x.try_div(y)?);
    }
    if pa.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let Some(y) = pb {
        if is_one(y) {
            return keep(a);
        }
        check_int_divisor(y)?;
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        check_float_divisor(y)?;
        match dtype.bits {
            32 => return float_result(dtype, single(x, y, |x, y| x / y)),
            64 => return float_result(dtype, x / y),
            _ => {}
        }
    }
    if fa == Some(0.0) {
        return keep(a);
    }
    if let Some(y) = fb {
        if y == 1.0 {
            return keep(a);
        }
        check_float_divisor(y)?;
    }
    Ok(None)
}

/// `TryConstFold<Mod>`, the truncating remainder: index types fold with the identities, wider
/// integers fold only two constants.
pub(super) fn truncmod(a: &Expr, b: &Expr) -> Folded {
    let dtype = a.prim_dtype()?;
    let (pa, pb) = (int(a), int(b));
    if is_index_type(dtype) && is_index_type(b.prim_dtype()?) {
        if let (Some(x), Some(y)) = (pa, pb) {
            check_int_divisor(y)?;
            return wrapped(dtype, x.try_rem(y)?);
        }
        if pa.is_some_and(BigInt::is_zero) {
            return keep(a);
        }
        if let Some(y) = pb {
            if is_one(y) {
                return int_result(dtype, BigInt::from(0));
            }
            check_int_divisor(y)?;
        }
    }
    if let (Some(x), Some(y)) = (pa, pb) {
        if dtype.bits >= 64 {
            check_int_divisor(y)?;
            return wrapped(dtype, x.try_rem(y)?);
        }
    }
    Ok(None)
}

/// `TryConstFold<FloorDiv>`.
pub(super) fn floordiv(a: &Expr, b: &Expr) -> Folded {
    let dtype = a.prim_dtype()?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        check_int_divisor(y)?;
        return wrapped(dtype, x.floor_div(y)?);
    }
    if pa.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let Some(y) = pb {
        if is_one(y) {
            return keep(a);
        }
        check_int_divisor(y)?;
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        if y != 0.0 {
            return match dtype.bits {
                32 => float_result(dtype, single(x, y, |x, y| (x / y).floor())),
                64 => float_result(dtype, (x / y).floor()),
                _ => Ok(None),
            };
        }
    }
    if fa == Some(0.0) {
        return keep(a);
    }
    if let Some(y) = fb {
        if y == 1.0 {
            return keep(a);
        }
        check_float_divisor(y)?;
    }
    Ok(None)
}

/// `TryConstFold<FloorMod>`: index types fold with the identities, wider or unsigned integers
/// fold only two constants.
pub(super) fn floormod(a: &Expr, b: &Expr) -> Folded {
    let dtype = a.prim_dtype()?;
    let (pa, pb) = (int(a), int(b));
    if is_index_type(dtype) && is_index_type(b.prim_dtype()?) {
        if let (Some(x), Some(y)) = (pa, pb) {
            check_int_divisor(y)?;
            return wrapped(dtype, x.floor_mod(y)?);
        }
        if pa.is_some_and(BigInt::is_zero) {
            return keep(a);
        }
        if let Some(y) = pb {
            if is_one(y) {
                return int_result(dtype, BigInt::from(0));
            }
            check_int_divisor(y)?;
        }
    }
    if let (Some(x), Some(y)) = (pa, pb) {
        if dtype.bits >= 64 || dtype.code == UINT {
            check_int_divisor(y)?;
            return wrapped(dtype, x.floor_mod(y)?);
        }
    }
    Ok(None)
}

/// `TryConstFold<Min>`: two constants, or an operand with itself. Floats compare as
/// `std::min`, which keeps the first of two unordered values.
pub(super) fn min(a: &Expr, b: &Expr) -> Folded {
    let dtype = a.prim_dtype()?;
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        return int_result(dtype, x.min(y).clone());
    }
    if let (Some(x), Some(y)) = (float(a), float(b)) {
        return float_result(dtype, if y < x { y } else { x });
    }
    if a.same_as(b) {
        return keep(a);
    }
    Ok(None)
}

/// `TryConstFold<Max>`, as [`min`] (`std::max` keeps the first of two unordered values).
pub(super) fn max(a: &Expr, b: &Expr) -> Folded {
    let dtype = a.prim_dtype()?;
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        return int_result(dtype, x.max(y).clone());
    }
    if let (Some(x), Some(y)) = (float(a), float(b)) {
        return float_result(dtype, if x < y { y } else { x });
    }
    if a.same_as(b) {
        return keep(a);
    }
    Ok(None)
}

// ---------------------------------------------------------------------------------------------
// Comparisons and logic

/// `TryConstFold` of a comparison: two constants compare to a boolean constant.
pub(super) fn compare(
    a: &Expr,
    b: &Expr,
    ints: fn(&BigInt, &BigInt) -> bool,
    floats: fn(&f64, &f64) -> bool,
) -> Folded {
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        return bool_result(ints(x, y));
    }
    if let (Some(x), Some(y)) = (float(a), float(b)) {
        return bool_result(floats(&x, &y));
    }
    Ok(None)
}

/// `TryConstFold<And>`: a true constant leaves the other side, a false one is the result.
pub(super) fn and(a: &Expr, b: &Expr) -> Folded {
    if let Some(x) = int(a) {
        return keep(if x.is_zero() { a } else { b });
    }
    if let Some(y) = int(b) {
        return keep(if y.is_zero() { b } else { a });
    }
    Ok(None)
}

/// `TryConstFold<Or>`: a true constant is the result, a false one leaves the other side.
pub(super) fn or(a: &Expr, b: &Expr) -> Folded {
    if let Some(x) = int(a) {
        return keep(if x.is_zero() { b } else { a });
    }
    if let Some(y) = int(b) {
        return keep(if y.is_zero() { a } else { b });
    }
    Ok(None)
}

/// `TryConstFold<Not>`.
pub(super) fn not(a: &Expr) -> Folded {
    match int(a) {
        Some(x) => bool_result(x.is_zero()),
        None => Ok(None),
    }
}
