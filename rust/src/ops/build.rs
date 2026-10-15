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

use std::cell::OnceCell;

use tvm_ffi::{
    Any, Array, BigInt, DLDataType, DLDataTypeCode, Map, ObjectRefCast, ObjectRefCore, Result,
    String,
};

use super::arith::Ex;
use crate::ir::{
    AssertStmt, Attrs, Bind, Call, DictAttrs, Evaluate, Expr, FloatImm, For, FuncType, If, IntImm,
    Location, Op, PrimType, PrimTypeObj, Range, RegionStmt, SeqStmt, SeqStmtObj, Stmt, StringImm,
    TensorLoad, Type, UnknownLoc, Var,
};
use crate::prim::{Add, Broadcast, Cast, Let, Mul, Not, Select, Sub};
use crate::tirx::Function;

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

pub(crate) fn scalar_dtype(code: DLDataTypeCode, bits: u8) -> DLDataType {
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

pub(crate) fn big_int_imm(dtype: DLDataType, value: BigInt) -> Result<IntImm> {
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
    Ok(Broadcast::new(element, lanes)?.into())
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
        SeqStmt::from_stmt(self.into())
    }
}

// ---------------------------------------------------------------------------------------------
// Locations, constants and names

impl UnknownLoc {
    /// C++ `UnknownLoc()`: the process-wide singleton.
    pub fn new() -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.UnknownLoc").call_packed(&[])?.try_into()
    }
}

impl StringImm {
    /// C++ `StringImm(value)`.
    pub fn new(value: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.StringImm")
            .call_tuple((String::from(value), unknown_loc()?))?
            .try_into()
    }
}

impl Var {
    /// C++ `PrimVar(name, dtype)`.
    pub fn prim(name: &str, dtype: DLDataType) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.Var")
            .call_tuple((String::from(name), prim_type(dtype)?, unknown_loc()?))?
            .try_into()
    }
}

/// C++ `max_value(dtype)`: the largest value of a scalar type.
pub fn max_value(dtype: DLDataType) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.max_value")
        .call_tuple((prim_type(dtype)?, unknown_loc()?))?
        .try_into()
}

/// C++ `min_value(dtype)`: the smallest value of a scalar type.
pub fn min_value(dtype: DLDataType) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.min_value")
        .call_tuple((prim_type(dtype)?, unknown_loc()?))?
        .try_into()
}

impl FuncType {
    /// C++ `FuncType(arg_types, ret_type)`.
    pub fn new(
        arg_types: impl IntoIterator<Item = Type>,
        ret_type: impl Into<Type>,
    ) -> Result<Self> {
        let arg_types: Array<Type> = arg_types.into_iter().collect();
        tvm_ffi::cached_global_func!("ir.FuncType")
            .call_tuple((arg_types, ret_type.into()))?
            .try_into()
    }
}

impl Range {
    /// C++ `Range(begin, end)`: the extent is `end - begin`.
    pub fn new(begin: impl IntoExpr, end: impl IntoExpr) -> Result<Self> {
        let (begin, end) = (begin.into_expr()?, end.into_expr()?);
        tvm_ffi::cached_global_func!("ir.Range")
            .call_tuple((begin, Some(end), unknown_loc()?))?
            .try_into()
    }

    /// C++ `Range::FromMinExtent(min, extent)`.
    pub fn from_min_extent(min: impl IntoExpr, extent: impl IntoExpr) -> Result<Self> {
        let (min, extent) = (min.into_expr()?, extent.into_expr()?);
        tvm_ffi::cached_global_func!("ir.Range_from_min_extent")
            .call_tuple((min, extent, unknown_loc()?))?
            .try_into()
    }
}

// ---------------------------------------------------------------------------------------------
// Expressions

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

    /// C++ `Call(ret_ty, op, args, attrs, ty_args, loc)`: infers the result type when `ret_ty` is
    /// absent.
    pub fn new_full(
        ret_ty: Option<Type>,
        op: Expr,
        args: Array<Expr>,
        attrs: Option<Attrs>,
        ty_args: Array<Type>,
        loc: Location,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.Call")
            .call_tuple((ret_ty, op, args, attrs, ty_args, loc))?
            .try_into()
    }
}

impl TensorLoad {
    /// C++ `TensorLoad(source, indices)`: infers the loaded type.
    pub fn new(source: impl Into<Expr>, indices: impl IntoIterator<Item = Expr>) -> Result<Self> {
        Self::new_full(source, indices, unknown_loc()?)
    }

    pub fn new_full(
        source: impl Into<Expr>,
        indices: impl IntoIterator<Item = Expr>,
        loc: Location,
    ) -> Result<Self> {
        let indices: Array<Expr> = indices.into_iter().collect();
        tvm_ffi::cached_global_func!("ir.TensorLoad")
            .call_tuple((source.into(), indices, loc))?
            .try_into()
    }
}

/// `$node::new(a, b)` / `new_full(a, b, loc)` for a raw binary constructor.
macro_rules! raw_binary {
    ($($node:ident => $global:literal),+ $(,)?) => {
        $(impl $node {
            #[doc = concat!("C++ `", stringify!($node), "(a, b)`: the raw constructor.")]
            pub fn new(a: impl IntoExpr, b: impl IntoExpr) -> Result<Self> {
                Self::new_full(a, b, unknown_loc()?)
            }

            #[doc = concat!("C++ `", stringify!($node), "(a, b, loc)`.")]
            pub fn new_full(a: impl IntoExpr, b: impl IntoExpr, loc: Location) -> Result<Self> {
                let (a, b) = (a.into_expr()?, b.into_expr()?);
                tvm_ffi::cached_global_func!($global).call_tuple((a, b, loc))?.try_into()
            }
        })+
    };
}

raw_binary!(Add => "prim.Add", Sub => "prim.Sub", Mul => "prim.Mul");

impl Select {
    /// C++ `Select(condition, true_value, false_value)`: the raw constructor.
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
        tvm_ffi::cached_global_func!("prim.Select").call_tuple((c, t, f, loc))?.try_into()
    }
}

impl Let {
    /// C++ `Let(var, value, body)`: the raw constructor.
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
        tvm_ffi::cached_global_func!("prim.Let").call_tuple((var, value, body, loc))?.try_into()
    }
}

impl Cast {
    /// C++ `Cast(value, ty)`: the raw constructor.
    pub fn new(value: impl IntoExpr, ty: PrimType) -> Result<Self> {
        Self::new_full(value, ty, unknown_loc()?)
    }

    pub fn new_full(value: impl IntoExpr, ty: PrimType, loc: Location) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Cast")
            .call_tuple((value.into_expr()?, ty, loc))?
            .try_into()
    }
}

impl Broadcast {
    /// C++ `Broadcast(value, lanes)`: the raw constructor.
    pub fn new(value: impl IntoExpr, lanes: impl IntoExpr) -> Result<Self> {
        Self::new_full(value, lanes, unknown_loc()?)
    }

    pub fn new_full(value: impl IntoExpr, lanes: impl IntoExpr, loc: Location) -> Result<Self> {
        let (value, lanes) = (value.into_expr()?, lanes.into_expr()?);
        tvm_ffi::cached_global_func!("prim.Broadcast").call_tuple((value, lanes, loc))?.try_into()
    }
}

impl Not {
    /// C++ `Not(a)`: the raw constructor.
    pub fn new(a: impl IntoExpr) -> Result<Self> {
        Self::new_full(a, unknown_loc()?)
    }

    pub fn new_full(a: impl IntoExpr, loc: Location) -> Result<Self> {
        tvm_ffi::cached_global_func!("prim.Not").call_tuple((a.into_expr()?, loc))?.try_into()
    }
}

// ---------------------------------------------------------------------------------------------
// Statements

impl SeqStmt {
    /// C++ `SeqStmt(seq)`: flattens nested sequences and drops no-op statements.
    pub fn new(seq: impl IntoIterator<Item = Stmt>) -> Result<Self> {
        seq_stmt(seq.into_iter().collect(), None)
    }

    /// C++ `SeqStmt(seq, loc)`.
    pub fn new_full(seq: impl IntoIterator<Item = Stmt>, loc: Location) -> Result<Self> {
        seq_stmt(seq.into_iter().collect(), Some(loc))
    }

    /// The C++ implicit `Stmt -> SeqStmt` conversion (`SeqStmt(stmt)`): a `SeqStmt` passes
    /// through, any other statement is wrapped with its own location.
    pub fn from_stmt(stmt: Stmt) -> Result<Self> {
        if stmt.as_node::<SeqStmtObj>().is_some() {
            return stmt.try_cast();
        }
        tvm_ffi::cached_global_func!("ir.SeqStmt")
            .call_tuple((stmt, Option::<Location>::None))?
            .try_into()
    }
}

fn seq_stmt(seq: Array<Stmt>, loc: Option<Location>) -> Result<SeqStmt> {
    tvm_ffi::cached_global_func!("ir.SeqStmt").call_tuple((seq, loc))?.try_into()
}

impl Evaluate {
    /// C++ `Evaluate(value)`; an `i32` is `IntImm::Int32`, as in C++ `Evaluate(0)`. Runs the
    /// validation registered for the value's type.
    pub fn new(value: impl IntoExpr) -> Result<Self> {
        Self::new_full(value, unknown_loc()?)
    }

    pub fn new_full(value: impl IntoExpr, loc: Location) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.Evaluate")
            .call_tuple((value.into_expr()?, loc))?
            .try_into()
    }
}

impl Bind {
    /// C++ `Bind(var, value)`: checks that the value has the variable's type.
    pub fn new(var: Var, value: impl IntoExpr) -> Result<Self> {
        Self::new_full(var, value, unknown_loc()?)
    }

    pub fn new_full(var: Var, value: impl IntoExpr, loc: Location) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.Bind")
            .call_tuple((var, value.into_expr()?, loc))?
            .try_into()
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
        tvm_ffi::cached_global_func!("ir.AssertStmt")
            .call_tuple((condition.into_expr()?, error_kind, message_parts, loc))?
            .try_into()
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
        tvm_ffi::cached_global_func!("ir.If")
            .call_tuple((condition.into_expr()?, then_case.into_body()?, else_case, loc))?
            .try_into()
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
        tvm_ffi::cached_global_func!("ir.For")
            .call_tuple((loop_var, min, extent, kind, body, annotations, step, loc))?
            .try_into()
    }
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

    pub fn new_full(
        op: Op,
        args: Array<Expr>,
        body_params: Array<Var>,
        attrs: DictAttrs,
        body: impl IntoBody,
        result_vars: Array<Var>,
        loc: Location,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.RegionStmt")
            .call_tuple((op, args, body_params, attrs, body.into_body()?, result_vars, loc))?
            .try_into()
    }
}

impl Function {
    /// C++ `Function(params, body)`: a void function without attributes. Derives `ty` from the
    /// signature.
    pub fn new(params: Array<Var>, body: Option<SeqStmt>) -> Result<Self> {
        let void: Type =
            tvm_ffi::cached_global_func!("ir.MissingType").call_packed(&[])?.try_into()?;
        let attrs = DictAttrs::new(Map::new());
        Self::new_full(params, body, void, attrs, unknown_loc()?)
    }

    /// C++ `Function(params, body, ret_type, attrs, loc)`; a missing return type becomes void.
    pub fn new_full(
        params: Array<Var>,
        body: Option<SeqStmt>,
        ret_type: Type,
        attrs: DictAttrs,
        loc: Location,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.Function")
            .call_tuple((params, body, ret_type, attrs, loc))?
            .try_into()
    }
}
