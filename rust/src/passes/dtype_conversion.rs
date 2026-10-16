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

//! `src/tirx/transform/dtype_conversion.{h,cc}`: convert a value between floating-point types
//! bit by bit, for targets without a native conversion.

use tvm_ffi::{DLDataType, DLDataTypeCode, Result, INTERNAL_ERROR};

use crate::ir::{Expr, PrimType};
use crate::ops::{if_then_else, make_const, prim_type, Ex, PrimOps};

const FLOAT: u8 = DLDataTypeCode::kDLFloat as u8;
const BFLOAT: u8 = DLDataTypeCode::kDLBfloat as u8;
const UINT: u8 = DLDataTypeCode::kDLUInt as u8;

/// The float8, float6 and float4 codes, `kDLFloat8_e3m4` through `kDLFloat4_e2m1fn`.
const NARROW_FLOATS: std::ops::RangeInclusive<u8> =
    DLDataTypeCode::kDLFloat8_e3m4 as u8..=DLDataTypeCode::kDLFloat4_e2m1fn as u8;

/// C++ `FloatConfig`: the exponent and mantissa widths and the exponent bias of a float format.
#[derive(Clone, Copy)]
struct FloatConfig {
    exponent: i32,
    mantissa: i32,
    bias: i32,
}

impl FloatConfig {
    /// C++ `FloatConfig::FromDataType`, for a type `is_floating_point` accepts.
    fn of(dtype: DLDataType) -> Self {
        let (exponent, mantissa, bias) = match (dtype.code, dtype.bits) {
            (FLOAT, 16) => (5, 10, 15),
            (FLOAT, 32) => (8, 23, 127),
            (FLOAT, _) => (11, 52, 1023),
            (BFLOAT, _) => (8, 7, 127),
            // kDLFloat8_e3m4
            (7, _) => (3, 4, 3),
            // kDLFloat8_e4m3, _e4m3b11fnuz, _e4m3fn, _e4m3fnuz
            (8..=11, _) => (4, 3, 7),
            // kDLFloat8_e5m2, _e5m2fnuz
            (12 | 13, _) => (5, 2, 15),
            // kDLFloat8_e8m0fnu
            (14, _) => (8, 0, 127),
            // kDLFloat6_e2m3fn
            (15, _) => (2, 3, 1),
            // kDLFloat6_e3m2fn
            (16, _) => (3, 2, 3),
            // kDLFloat4_e2m1fn
            _ => (2, 1, 1),
        };
        Self { exponent, mantissa, bias }
    }
}

fn is_floating_point(dtype: DLDataType) -> bool {
    dtype.code == FLOAT || dtype.code == BFLOAT || NARROW_FLOATS.contains(&dtype.code)
}

/// C++ `GetStorageUIntDType`: the unsigned integer type of the same width and lanes.
pub(crate) fn storage_uint(dtype: DLDataType) -> DLDataType {
    DLDataType { code: UINT, ..dtype }
}

/// C++ `DTypeConversion(src_value, tgt_dtype)` with `RoundingMode::kHalfToEven`.
pub(crate) fn dtype_conversion(src_value: Expr, tgt: &PrimType) -> Result<Expr> {
    let (src_dtype, tgt_dtype) = (src_value.prim_dtype()?, tgt.dtype);
    check_cmp!(
        src_dtype.lanes,
        ==,
        tgt_dtype.lanes,
        INTERNAL_ERROR,
        "src_dtype->dtype.lanes == tgt_dtype->dtype.lanes",
        "The lanes for data type for source value must matches the target datatype."
    );
    check!(
        is_floating_point(src_dtype) && is_floating_point(tgt_dtype),
        INTERNAL_ERROR,
        "is_floating_point(src_dtype) && is_floating_point(tgt_dtype)"
    );
    let (src_fp, tgt_fp) = (FloatConfig::of(src_dtype), FloatConfig::of(tgt_dtype));
    let exponent_delta = tgt_fp.exponent - src_fp.exponent;
    let bias_delta = tgt_fp.bias - src_fp.bias;
    let mantissa_delta = tgt_fp.mantissa - src_fp.mantissa;
    let src_uint = storage_uint(src_dtype);
    let tgt_uint_dtype = storage_uint(tgt_dtype);
    let tgt_uint = prim_type(tgt_uint_dtype)?;
    // `ReinterpretAsUInt(src_value)`.
    let mut value = Ex::from(src_value).reinterpret(prim_type(src_uint)?);
    if mantissa_delta < 0 {
        // Round half to even.
        let half_minus_one = (1i64 << (-mantissa_delta - 1)) - 1;
        let rounding_bias =
            ((&value >> -mantissa_delta) & 1) + make_const(src_uint, half_minus_one)?;
        value = &value + rounding_bias;
    }
    let shift_left = |value: &Ex| match mantissa_delta >= 0 {
        true => value.cast(&tgt_uint) << mantissa_delta,
        false => (value >> -mantissa_delta).cast(&tgt_uint),
    };
    if exponent_delta == 0 {
        let mut ret = shift_left(&value);
        if bias_delta > 0 {
            ret += Ex::from(make_const(tgt_uint_dtype, bias_delta as i64)?) << tgt_fp.mantissa;
        } else if bias_delta < 0 {
            ret -= Ex::from(make_const(tgt_uint_dtype, -bias_delta as i64)?) << tgt_fp.mantissa;
        }
        return ret.reinterpret(tgt).expr();
    }
    let mantissa_mask = make_const(tgt_uint_dtype, (1i64 << tgt_fp.mantissa) - 1)?;
    let ret_mantissa = shift_left(&value) & mantissa_mask;
    let exponent_before_delta = (&value << 1) >> (src_fp.mantissa + 1);
    let ret_sign = (&value >> (src_fp.mantissa + src_fp.exponent)).cast(&tgt_uint)
        << (tgt_fp.mantissa + tgt_fp.exponent);
    if bias_delta >= 0 {
        let ret_exponent = match bias_delta > 0 {
            true => (&exponent_before_delta + bias_delta).cast(&tgt_uint) << tgt_fp.mantissa,
            false => exponent_before_delta.cast(&tgt_uint) << tgt_fp.mantissa,
        };
        return (ret_mantissa | ret_exponent | ret_sign).reinterpret(tgt).expr();
    }
    let round_to_zero = exponent_before_delta.less(-bias_delta);
    let ret_exponent = (&exponent_before_delta - -bias_delta).cast(&tgt_uint) << tgt_fp.mantissa;
    let zero = make_const(tgt_uint_dtype, 0)?;
    if_then_else(round_to_zero, zero, ret_mantissa | ret_exponent | ret_sign)
        .reinterpret(tgt)
        .expr()
}
