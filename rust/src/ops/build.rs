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
use std::fmt::Display;
use std::sync::OnceLock;
use std::thread::LocalKey;

use tvm_ffi::object::ObjectRef;
use tvm_ffi::{
    Any, AnyView, Array, BigInt, DLDataType, DLDataTypeCode, DLDataTypeExt,
    Function as FfiFunction, Map, ObjectCore, ObjectRefCast, ObjectRefCore, Result, String,
    TypeAttrColumn, INTERNAL_ERROR, TYPE_ERROR,
};

use super::arith::Ex;
use super::math::{check_failed, element, lanes, op, BOOL, HANDLE, INT, UINT};
use crate::ir::{
    AssertStmt, Attrs, Bind, Call, DictAttrs, Evaluate, Expr, FloatImm, For, FuncType, GlobalVar,
    If, IntImm, LambdaExpr, Location, MissingType, MissingTypeObj, Op, PrimType, PrimTypeObj,
    PtrType, Range, RegionStmt, Return, SeqStmt, SeqStmtObj, Stmt, StringImm, TensorLoad,
    TensorStore, Tuple, TupleType, Type, UniqueNameSupply, UnknownLoc, Var,
};
use crate::prim::{
    Add, And, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, Broadcast, Cast, Div, FloorDiv,
    FloorMod, LShift, Let, Max, Min, Mod, Mul, Not, Or, RShift, Ramp, Select, Shuffle, Sub, EQ, GE,
    GT, LE, LT, NE,
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
    int_imm_at(dtype, value, &unknown_loc()?)
}

/// `IntImm(dtype, value, loc)` through the native constructor.
pub(crate) fn int_imm_at(dtype: DLDataType, value: BigInt, loc: &Location) -> Result<IntImm> {
    tvm_ffi::cached_global_func!("ir.IntImm").call_tuple((dtype, value, loc.clone()))?.try_into()
}

/// `FloatImm(dtype, value)` through the native constructor.
pub fn float_imm(dtype: DLDataType, value: f64) -> Result<FloatImm> {
    float_imm_at(dtype, value, &unknown_loc()?)
}

/// `FloatImm(dtype, value, loc)` through the native constructor.
pub(crate) fn float_imm_at(dtype: DLDataType, value: f64, loc: &Location) -> Result<FloatImm> {
    tvm_ffi::cached_global_func!("ir.FloatImm").call_tuple((dtype, value, loc.clone()))?.try_into()
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
// Types and variables

impl MissingType {
    /// C++ `MissingType()`: the process-wide singleton, the type of an expression whose type is
    /// not known yet.
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

/// The error of a failed C++ `TVM_FFI_ICHECK_NE` / `_LE` / ..., which prints both sides.
fn compare_failed(condition: &str, lhs: impl Display, rhs: impl Display) -> tvm_ffi::Error {
    let message = format!("Check failed: {condition} ({lhs} vs. {rhs}) : ");
    tvm_ffi::Error::new(INTERNAL_ERROR, &message, "")
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
            return Err(compare_failed("vectors.size() != 0U", 0, 0));
        };
        if indices.is_empty() {
            return Err(compare_failed("indices.size() != 0U", 0, 0));
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
            return Err(compare_failed(condition, indices.len(), total_lanes));
        }
        let ty = prim_type(DLDataType { lanes: indices.len() as u16, ..base })?;
        Ok(Self::from_complete_fields(loc, ty.into(), vectors, indices))
    }

    /// C++ `Shuffle::Concat(vectors)`: the lanes of the vectors in order; one vector is itself.
    pub fn concat(vectors: impl IntoIterator<Item = Expr>) -> Result<Expr> {
        let mut vectors: Vec<Expr> = vectors.into_iter().collect();
        if vectors.len() <= 1 {
            return vectors.pop().ok_or_else(|| compare_failed("vectors.size() != 0", 0, 0));
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

impl Ramp {
    /// C++ `Ramp(base, stride, lanes)`: the raw constructor.
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
        tvm_ffi::cached_global_func!("prim.Ramp").call_tuple((base, stride, lanes, loc))?.try_into()
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
    pub fn from_stmt(stmt: impl Into<Stmt>) -> Result<Self> {
        let stmt = stmt.into();
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
        let attrs = DictAttrs::new(Map::new());
        Self::new_full(params, body, MissingType::new()?.into(), attrs, unknown_loc()?)
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
