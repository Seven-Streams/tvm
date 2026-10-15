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

//! `TVMFFIABIBuilder` (`src/tirx/transform/tvm_ffi_binder.{h,cc}`): the statements that decode,
//! check and bind the packed arguments of a function for `MakePackedAPI`. It has no FFI, so it is
//! written in Rust after the C++ one.
//!
//! The statements go to three sequences, so that a symbolic shape variable defined by a later
//! tensor is in scope wherever an earlier one checks it: `init_nest` (the bindings and the
//! per-field checks that guard later loads), `asserts` (the checks of bound values) and
//! `decl_tensors` (the tensor declarations of the parameters).

use std::collections::HashMap;

use tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi::{
    Array, BigInt, DLDataType, DLDataTypeExt, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result,
    INTERNAL_ERROR, TYPE_ERROR,
};

use super::analyzer::is_const_int;
use crate::ir::{
    AssertStmt, Bind, Call, Evaluate, Expr, FloatImmObj, If, IntImmObj, OpHandle, PrimType,
    PrimTypeObj, PtrType, PtrTypeObj, SeqStmt, Stmt, StringImm, Type, Var, VarObj,
};
use crate::ops::{
    if_then_else, int32, int_imm, make_const, prim_dtype, prim_type, repr, structural_equal, Ex,
    PrimOps,
};
use crate::prim::op::IF_THEN_ELSE;
use crate::prim::{
    AddObj, CastObj, DivObj, FloorDivObj, FloorModObj, MaxObj, MinObj, ModObj, MulObj, Not, Select,
    SubObj,
};
use crate::runtime::MemoryScope;
use crate::sym::Analyzer;
use crate::tirx::op::{
    ABI_FIELD_GET, ASSUME_ALIGNED, DECL_TENSOR, HANDLE_ADD_BYTE_OFFSET, ISNULLPTR, REINTERPRET,
};
use crate::tirx::{TensorType, TensorTypeObj};

// `TVMStructFieldKind` (`include/tvm/tirx/op/abi.h`).
const DL_TENSOR_DATA: i32 = 1;
const DL_TENSOR_SHAPE: i32 = 2;
const DL_TENSOR_STRIDES: i32 = 3;
const DL_TENSOR_NDIM: i32 = 4;
const DL_TENSOR_TYPE_CODE: i32 = 5;
const DL_TENSOR_TYPE_BITS: i32 = 6;
const DL_TENSOR_TYPE_LANES: i32 = 7;
const DL_TENSOR_BYTE_OFFSET: i32 = 8;
const DL_TENSOR_DEVICE_ID: i32 = 9;
const DL_TENSOR_DEVICE_TYPE: i32 = 10;
pub(crate) const FFI_ANY_TYPE_INDEX: i32 = 13;
pub(crate) const FFI_ANY_ZERO_PADDING: i32 = 14;
pub(crate) const FFI_ANY_UNION_VALUE: i32 = 15;
const INT64_ARRAY_ELEM: i32 = 17;

// `DLDeviceType` (`dlpack.h`).
const DL_CUDA_HOST: i32 = 3;
const DL_ROCM: i32 = 10;
const DL_ROCM_HOST: i32 = 11;
const DL_CUDA_MANAGED: i32 = 13;

/// `sizeof(TVMFFIObject)`: where an object's cell starts after its header.
const OBJECT_CELL_OFFSET: i32 = 24;

const INT: u8 = 0;
const UINT: u8 = 1;
const FLOAT: u8 = 2;
const BFLOAT: u8 = 4;
const BOOL: u8 = 6;

fn dtype(code: u8, bits: u8) -> DLDataType {
    DLDataType { code, bits, lanes: 1 }
}

/// `PrimType::{Int, UInt, Float, Bool}(bits)`.
pub(crate) fn scalar(code: u8, bits: u8) -> Result<PrimType> {
    prim_type(dtype(code, bits))
}

/// `PrimType::Void()`, `handle(0, 0)`.
fn void() -> Result<PrimType> {
    prim_type(DLDataType { code: 3, bits: 0, lanes: 0 })
}

/// `IntImm::Int32(value)` as an expression.
pub(crate) fn i32_imm(value: i64) -> Result<Expr> {
    Ok(int32(value)?.into())
}

pub(crate) fn text(value: &str) -> Result<StringImm> {
    StringImm::new(value)
}

fn dtype_str(dtype: DLDataType) -> std::string::String {
    dtype.to_string().as_str().to_string()
}

fn storage_bytes(dtype: DLDataType) -> i64 {
    (dtype.bits as i64 * dtype.lanes as i16 as i64 + 7) / 8
}

/// `runtime::DLDeviceType2Str`.
fn device_name(device_type: i64) -> Result<&'static str> {
    Ok(match device_type {
        1 => "cpu",
        2 => "cuda",
        3 => "cuda_host",
        13 => "cuda_managed",
        4 => "opencl",
        7 => "vulkan",
        8 => "metal",
        9 => "vpi",
        10 => "rocm",
        11 => "rocm_host",
        12 => "ext_dev",
        14 => "oneapi",
        15 => "webgpu",
        16 => "hexagon",
        18 => "trn",
        other => tvm_ffi::bail!(INTERNAL_ERROR, "unknown type = {other}"),
    })
}

/// `APIType(t)` (`ir_utils.h`): the type a packed argument of type `t` travels as.
fn api_type(ty: &PrimType) -> Result<PrimType> {
    let dtype = ty.dtype;
    let is_void = dtype.code == 3 && dtype.bits == 0 && dtype.lanes == 0;
    check!(!is_void, INTERNAL_ERROR, "!t.IsVoid()", "Cannot pass void type through packed API.");
    check_cmp!(
        dtype.lanes as i16,
        ==,
        1,
        INTERNAL_ERROR,
        "t.lanes() == 1",
        "Cannot pass vector type through packed API."
    );
    match dtype.code {
        BOOL | UINT | INT => scalar(INT, 64),
        FLOAT | BFLOAT => scalar(FLOAT, 64),
        _ => Err(error!(
            INTERNAL_ERROR,
            "Check failed: (t.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat)) is \
             false: "
        )),
    }
}

/// `Call(ty, op, args)`.
fn call(ty: impl Into<Type>, op: &OpHandle, args: Vec<Expr>) -> Result<Expr> {
    Ok(Call::new(ty, op.get()?, args)?.into())
}

/// One step of an `ffi::reflection::AccessPath`, from the root parameter array down.
#[derive(Clone)]
enum Step {
    Item(i64),
    Attr(&'static str),
    Name(std::string::String),
}

type Path = Vec<Step>;

fn extend(path: &Path, step: Step) -> Path {
    let mut path = path.clone();
    path.push(step);
    path
}

/// A defined variable: the value it stands for and where it was first bound.
pub(crate) struct VarDef {
    value: Expr,
    first_def_path: Path,
}

/// What `Finalize` returns.
pub(crate) struct Binding {
    pub var_defs: HashMap<ObjectIdentity, VarDef>,
    pub init_nest: Vec<Stmt>,
    pub asserts: Vec<Stmt>,
    pub decl_tensors: Vec<Stmt>,
}

/// A check of a bound constant expression, rendered when the binder finishes.
struct PendingAssert {
    condition: Expr,
    path: Path,
    expected: Expr,
}

/// `TVMFFIABIBuilder`.
pub(crate) struct AbiBinder {
    var_defs: HashMap<ObjectIdentity, VarDef>,
    init_nest: Vec<Stmt>,
    asserts: Vec<Stmt>,
    decl_tensors: Vec<Stmt>,
    pending: Vec<PendingAssert>,
    analyzer: Analyzer,
    func_name: std::string::String,
    params: Vec<Var>,
    tensor_handles: HashMap<ObjectIdentity, Var>,
    packed_args: Var,
    device_type: Expr,
    device_id: Expr,
    param_names: HashMap<i64, std::string::String>,
    when_calling: StringImm,
    signature: StringImm,
}

/// The tensor type of a tensor-typed variable (C++ `param.as<TensorVar>()`).
fn tensor_type(var: &Var) -> Option<TensorType> {
    var.ty.as_node::<TensorTypeObj>()?;
    var.ty.clone().try_cast().ok()
}

/// `TensorTypeNode::DefaultIndexType()`: the type of the first extent, or int64.
fn index_type(ty: &TensorTypeObj) -> Result<PrimType> {
    match ty.shape.len() {
        0 => scalar(INT, 64),
        _ => Ok(ty.shape.get(0)?.ty.clone().try_cast()?),
    }
}

impl AbiBinder {
    /// The constructor: the signature for messages, and the argument-count and null-pointer
    /// checks.
    pub(crate) fn new(
        func_name: &str,
        params: &Array<Var>,
        packed_args: Var,
        num_packed_args: &Var,
        device_type: Expr,
        device_id: Expr,
    ) -> Result<Self> {
        let mut param_names = HashMap::new();
        let mut entries = Vec::new();
        for (i, param) in params.iter().enumerate() {
            let name = param.name.as_str().to_string();
            match tensor_type(&param) {
                Some(tensor) => {
                    let mut dims = Vec::new();
                    for dim in tensor.shape.iter() {
                        dims.push(match prim_var_name(&dim) {
                            Some(name) if name.is_empty() => "v".to_string(),
                            Some(name) => name,
                            None => repr(&dim),
                        });
                    }
                    let dtype = dtype_str(tensor.dtype.dtype);
                    entries.push(format!("{name}: Tensor([{}], {dtype})", dims.join(", ")));
                }
                None => {
                    let ty = match param.ty.as_node::<PrimTypeObj>() {
                        Some(prim) => dtype_str(prim.dtype),
                        None => repr(&param.ty),
                    };
                    entries.push(format!("{name}: {ty}"));
                }
            }
            param_names.insert(i as i64, name);
        }
        let signature = format!("{func_name}({})", entries.join(", "));
        let mut binder = Self {
            var_defs: HashMap::new(),
            init_nest: Vec::new(),
            asserts: Vec::new(),
            decl_tensors: Vec::new(),
            pending: Vec::new(),
            analyzer: Analyzer::new()?,
            func_name: func_name.to_string(),
            params: params.iter().collect(),
            tensor_handles: HashMap::new(),
            packed_args,
            device_type,
            device_id,
            param_names,
            when_calling: text(" when calling:\n  `")?,
            signature: text(&signature)?,
        };
        let num_args = params.len() as i64;
        let count = num_packed_args.equal(num_args).expr()?;
        let parts = binder.message(&["Expected ", &num_args.to_string(), " arguments"], &["`"])?;
        binder.emit_assert(count, "TypeError", parts)?;
        if num_args > 0 {
            let null = call(scalar(BOOL, 8)?, &ISNULLPTR, vec![binder.packed_args.clone().into()])?;
            let parts = binder.message(&["args pointer is NULL"], &["`"])?;
            binder.emit_assert((!Ex::from(null)).expr()?, "TypeError", parts)?;
        }
        Ok(binder)
    }

    /// `head`, then " when calling:\n  `<signature>", then `tail`, as message parts.
    fn message(&self, head: &[&str], tail: &[&str]) -> Result<Array<StringImm>> {
        let mut parts = Vec::new();
        for part in head {
            parts.push(text(part)?);
        }
        parts.push(self.when_calling.clone());
        parts.push(self.signature.clone());
        for part in tail {
            parts.push(text(part)?);
        }
        Ok(Array::new(parts))
    }

    /// `EmitAssert`: a check in `init_nest`, ahead of the loads it guards.
    fn emit_assert(&mut self, condition: Expr, kind: &str, parts: Array<StringImm>) -> Result<()> {
        self.init_nest.push(AssertStmt::new(condition, text(kind)?, parts)?.into());
        Ok(())
    }

    fn emit_type_index_check(&mut self, index: i64, condition: Expr, expected: &str) -> Result<()> {
        let head = ["Mismatched type on argument #", &index.to_string()];
        let parts = self.message(&head, &["`,\n  expected ", expected])?;
        self.emit_assert(condition, "TypeError", parts)
    }

    /// `RenderAccessPath`: `A.shape[0]` for a tensor field, the name for a scalar parameter.
    fn render(&self, path: &Path) -> std::string::String {
        let mut out = std::string::String::new();
        let mut first_printed = false;
        let mut consumed_root = false;
        for (i, step) in path.iter().enumerate() {
            match step {
                Step::Attr(name) => attr(&mut out, &mut first_printed, name),
                Step::Name(name) => attr(&mut out, &mut first_printed, name),
                Step::Item(index) if !first_printed && !consumed_root => {
                    consumed_root = true;
                    if matches!(path.get(i + 1), Some(Step::Attr(_) | Step::Name(_))) {
                        continue;
                    }
                    if let Some(name) = self.param_names.get(index) {
                        out.push_str(name);
                        first_printed = true;
                    }
                }
                Step::Item(index) => out.push_str(&format!("[{index}]")),
            }
        }
        out
    }

    /// `GetParamIndex`: the parameter a path starts at, or -1.
    fn param_index(path: &Path) -> i64 {
        match path.first() {
            Some(Step::Item(index)) => *index,
            _ => -1,
        }
    }

    fn simplify(&self, expr: Ex) -> Result<Expr> {
        self.analyzer.simplify(&expr.expr()?, 2)
    }

    /// `BindScalar`: define a variable on its first bind, else check the value against the
    /// first; a constant expression is checked when the binder finishes. Whether this defined
    /// the variable.
    fn bind_scalar(
        &mut self,
        arg: &Expr,
        value: &Expr,
        path: &Path,
        with_lets: bool,
    ) -> Result<bool> {
        check!(prim_dtype(arg)? == prim_dtype(value)?, INTERNAL_ERROR, "arg.ty() == value.ty()");
        if arg.as_node::<VarObj>().is_none() {
            let condition = self.simplify(arg.equal(value))?;
            if is_const_int(&condition, 0) {
                tvm_ffi::bail!(
                    INTERNAL_ERROR,
                    "Bind have an unmet assertion: {} == {} at {}",
                    repr(arg),
                    repr(value),
                    self.render(path)
                );
            }
            if !is_const_int(&condition, 1) {
                self.pending.push(PendingAssert {
                    condition,
                    path: path.clone(),
                    expected: arg.clone(),
                });
            }
            return Ok(false);
        }
        let var: Var = arg.clone().try_cast()?;
        let key = ObjectIdentity::of(&var);
        let Some(def) = self.var_defs.get(&key) else {
            let bound = if with_lets { arg.clone() } else { value.clone() };
            self.var_defs.insert(key, VarDef { value: bound, first_def_path: path.clone() });
            if with_lets {
                self.init_nest.push(Bind::new(var, value.clone())?.into());
            }
            return Ok(true);
        };
        let (previous, first_path) = (def.value.clone(), def.first_def_path.clone());
        let condition = self.simplify(previous.equal(value))?;
        if is_const_int(&condition, 0) {
            tvm_ffi::bail!(
                INTERNAL_ERROR,
                "Bind have an unmet assertion: {} == {} at {}",
                repr(&previous),
                repr(value),
                self.render(path)
            );
        }
        if !is_const_int(&condition, 1) {
            let (current, first) = (self.render(path), self.render(&first_path));
            let index = Self::param_index(path);
            let mut head = vec!["Mismatched ".to_string(), current.clone()];
            if index >= 0 {
                head.push(" on argument #".to_string());
                head.push(index.to_string());
            }
            let tail = if !first.is_empty() && first != current {
                vec!["`,\n  expected to match ".to_string(), first]
            } else {
                vec!["`,\n  expected matching value".to_string()]
            };
            let head: Vec<&str> = head.iter().map(|s| s.as_str()).collect();
            let tail: Vec<&str> = tail.iter().map(|s| s.as_str()).collect();
            let parts = self.message(&head, &tail)?;
            self.asserts.push(AssertStmt::new(condition, text("ValueError")?, parts)?.into());
        }
        Ok(false)
    }

    /// `BindPointer`: define a pointer variable, or check it against its first binding.
    fn bind_pointer(
        &mut self,
        arg: &Var,
        value: Expr,
        path: &Path,
        with_lets: bool,
    ) -> Result<bool> {
        check!(
            structural_equal(&arg.ty, &value.ty)?,
            INTERNAL_ERROR,
            "ffi::StructuralEqual()(arg->ty, value->ty)"
        );
        let key = ObjectIdentity::of(arg);
        let Some(def) = self.var_defs.get(&key) else {
            self.var_defs
                .insert(key, VarDef { value: value.clone(), first_def_path: path.clone() });
            if with_lets {
                self.init_nest.push(Bind::new(arg.clone(), value)?.into());
            }
            return Ok(true);
        };
        let (previous, first_path) = (def.value.clone(), def.first_def_path.clone());
        let uint64 = scalar(UINT, 64)?;
        let previous = call(uint64.clone(), &REINTERPRET, vec![previous])?;
        let current = call(uint64, &REINTERPRET, vec![value])?;
        let condition = self.simplify(previous.equal(&current))?;
        if is_const_int(&condition, 0) {
            let at = self.render(path);
            tvm_ffi::bail!(INTERNAL_ERROR, "Bind has an unmet pointer assertion at {at}");
        }
        if !is_const_int(&condition, 1) {
            let (current, first) = (self.render(path), self.render(&first_path));
            let mut parts = vec![text("Mismatched ")?, text(&current)?];
            if !first.is_empty() && first != current {
                parts.push(text("`,\n  expected to match ")?);
                parts.push(text(&first)?);
            }
            self.asserts
                .push(AssertStmt::new(condition, text("ValueError")?, Array::new(parts))?.into());
        }
        Ok(false)
    }

    /// `ExprPathRenderer`: an expression with each bound variable written as the path that
    /// defined it.
    fn render_expr(&self, expr: &Expr) -> std::string::String {
        if let Some(var) = expr.as_node::<VarObj>() {
            let key = ObjectIdentity::of(expr);
            if let Some(def) = self.var_defs.get(&key) {
                let path = self.render(&def.first_def_path);
                if !path.is_empty() {
                    return path;
                }
            }
            return var.name.as_str().to_string();
        }
        if let Some(imm) = expr.as_node::<IntImmObj>() {
            return imm.value.to_string();
        }
        if let Some(imm) = expr.as_node::<FloatImmObj>() {
            return format_double(imm.value);
        }
        if let Some(cast) = expr.as_node::<CastObj>() {
            return self.render_expr(&cast.value);
        }
        macro_rules! binary {
            ($($node:ty => $op:literal),*) => {$(
                if let Some(node) = expr.as_node::<$node>() {
                    return format!("{}{}{}", self.render_expr(&node.a), $op, self.render_expr(&node.b));
                }
            )*};
        }
        binary!(AddObj => " + ", SubObj => " - ", MulObj => " * ", DivObj => " / ", ModObj => " % ");
        macro_rules! function {
            ($($node:ty => $name:literal),*) => {$(
                if let Some(node) = expr.as_node::<$node>() {
                    return format!("{}({}, {})", $name, self.render_expr(&node.a), self.render_expr(&node.b));
                }
            )*};
        }
        function!(FloorDivObj => "floordiv", FloorModObj => "floormod", MinObj => "min", MaxObj => "max");
        repr(expr)
    }

    /// `RenderPendingAsserts`: the deferred constant-expression checks, with readable names.
    fn render_pending(&mut self) -> Result<()> {
        for pending in std::mem::take(&mut self.pending) {
            let display = self.render_expr(&pending.expected);
            let path = self.render(&pending.path);
            let index = Self::param_index(&pending.path);
            let mut head = vec!["Invalid ".to_string(), path];
            if index >= 0 {
                head.push(" on argument #".to_string());
                head.push(index.to_string());
            }
            let head: Vec<&str> = head.iter().map(|s| s.as_str()).collect();
            let parts = self.message(&head, &["`,\n  expected ", &display])?;
            self.asserts
                .push(AssertStmt::new(pending.condition, text("ValueError")?, parts)?.into());
        }
        Ok(())
    }

    /// `Finalize`.
    pub(crate) fn finalize(mut self) -> Result<Binding> {
        self.render_pending()?;
        Ok(Binding {
            var_defs: self.var_defs,
            init_nest: self.init_nest,
            asserts: self.asserts,
            decl_tensors: self.decl_tensors,
        })
    }

    /// `LoadTVMFFIAnyUnionValue`: the union value of argument `index` as `ty`.
    fn load_union_value(&self, index: i64, ty: &Type) -> Result<Expr> {
        let args = vec![
            self.packed_args.clone().into(),
            i32_imm(index)?,
            i32_imm(FFI_ANY_UNION_VALUE as i64)?,
        ];
        if let Ok(prim) = ty.clone().try_cast::<PrimType>() {
            let api = api_type(&prim)?;
            let value = call(api.clone(), &ABI_FIELD_GET, args)?;
            if api.dtype != prim.dtype {
                return Ok(crate::prim::Cast::new(value, prim)?.into());
            }
            return Ok(value);
        }
        check!(
            ty.as_node::<PtrTypeObj>().is_some(),
            TYPE_ERROR,
            "arg_type.as<PtrTypeNode>()",
            "Packed union values must have primitive or pointer type, but got {}",
            repr(ty)
        );
        call(ty.clone(), &ABI_FIELD_GET, args)
    }

    fn load_prim(&self, index: i64, ty: &PrimType) -> Result<Expr> {
        self.load_union_value(index, &ty.clone().into())
    }

    /// `DecodeParamOpaqueHandle`: a tensor or a pointer, skipping the object header of a tensor.
    fn decode_opaque_handle(&mut self, index: i64, type_index: &Expr) -> Result<Expr> {
        let param = &self.params[index as usize];
        let expected =
            if param.ty.as_node::<TensorTypeObj>().is_some() { "Tensor" } else { "pointer" };
        let accepted = type_index
            .equal(TypeIndex::kTVMFFINone as i32)
            .logical_or(type_index.equal(TypeIndex::kTVMFFIOpaquePtr as i32))
            .logical_or(type_index.equal(TypeIndex::kTVMFFIDLTensorPtr as i32))
            .logical_or(type_index.greater_equal(TypeIndex::kTVMFFIStaticObjectBegin as i32))
            .expr()?;
        self.emit_type_index_check(index, accepted, expected)?;
        let value = self.load_union_value(index, &PtrType::void()?.into())?;
        let from_tensor = call(
            PtrType::void()?,
            &HANDLE_ADD_BYTE_OFFSET,
            vec![value.clone(), i32_imm(OBJECT_CELL_OFFSET as i64)?],
        )?;
        let is_tensor = type_index.equal(TypeIndex::kTVMFFITensor as i32).expr()?;
        call(PtrType::void()?, &IF_THEN_ELSE, vec![is_tensor, from_tensor, value])
    }

    fn decode_bool(&mut self, index: i64, type_index: &Expr) -> Result<Expr> {
        let accepted = type_index
            .equal(TypeIndex::kTVMFFIBool as i32)
            .logical_or(type_index.equal(TypeIndex::kTVMFFIInt as i32))
            .expr()?;
        self.emit_type_index_check(index, accepted, "boolean")?;
        let value = self.load_prim(index, &scalar(INT, 64)?)?;
        Ok(crate::prim::Cast::new(value, scalar(BOOL, 8)?)?.into())
    }

    fn decode_int(&mut self, index: i64, type_index: &Expr, ty: &PrimType) -> Result<Expr> {
        let accepted = type_index
            .equal(TypeIndex::kTVMFFIInt as i32)
            .logical_or(type_index.equal(TypeIndex::kTVMFFIBool as i32))
            .expr()?;
        self.emit_type_index_check(index, accepted, "int")?;
        self.load_prim(index, ty)
    }

    fn decode_float(&mut self, index: i64, type_index: &Expr, ty: &PrimType) -> Result<Expr> {
        let accepted = type_index
            .equal(TypeIndex::kTVMFFIFloat as i32)
            .logical_or(type_index.equal(TypeIndex::kTVMFFIInt as i32))
            .logical_or(type_index.equal(TypeIndex::kTVMFFIBool as i32))
            .expr()?;
        self.emit_type_index_check(index, accepted, "float")?;
        let is_float = type_index.equal(TypeIndex::kTVMFFIFloat as i32).expr()?;
        let float_value = self.load_prim(index, ty)?;
        let int_value = self.load_prim(index, &scalar(INT, 64)?)?;
        let converted = crate::prim::Cast::new(int_value, ty.clone())?;
        Ok(Select::new(is_float, float_value, converted)?.into())
    }

    /// `DecodeParam`: check the type index of argument `index`, load it, and bind it.
    fn decode_param(&mut self, index: i64) -> Result<()> {
        let param = self.params[index as usize].clone();
        let name = param.name.as_str().to_string();
        let type_index: Expr = Var::new(format!("{name}.type_index"), scalar(INT, 32)?)?.into();
        let args = vec![
            self.packed_args.clone().into(),
            i32_imm(index)?,
            i32_imm(FFI_ANY_TYPE_INDEX as i64)?,
        ];
        let load = call(scalar(INT, 32)?, &ABI_FIELD_GET, args)?;
        self.init_nest.push(Bind::new(type_index.clone().try_cast()?, load)?.into());
        let path = vec![Step::Item(index)];
        if param.ty.as_node::<TensorTypeObj>().is_some() {
            let handle = Var::new(format!("{name}.handle"), PtrType::void()?)?;
            let value = self.decode_opaque_handle(index, &type_index)?;
            self.bind_pointer(&handle, value, &path, true)?;
            self.tensor_handles.insert(ObjectIdentity::of(&param), handle);
            return Ok(());
        }
        if param.ty.as_node::<PtrTypeObj>().is_some() {
            let value = self.decode_opaque_handle(index, &type_index)?;
            let pointer = call(param.ty.clone(), &REINTERPRET, vec![value])?;
            self.bind_pointer(&param, pointer, &path, true)?;
            return Ok(());
        }
        let ty: PrimType = param.ty.clone().try_cast()?;
        let value = match ty.dtype.code {
            BOOL => self.decode_bool(index, &type_index)?,
            INT | UINT => self.decode_int(index, &type_index, &ty)?,
            FLOAT | BFLOAT => self.decode_float(index, &type_index, &ty)?,
            _ => tvm_ffi::bail!(
                INTERNAL_ERROR,
                "Check failed: (dtype.MatchesCode(DLDataTypeCode::kDLFloat, \
                 DLDataTypeCode::kDLBfloat)) is false: "
            ),
        };
        self.bind_scalar(&param.into(), &value, &path, true)?;
        Ok(())
    }

    /// `DecodeAllParams`: decode every argument, then bind the DLTensor fields of the tensors.
    pub(crate) fn decode_all_params(&mut self) -> Result<()> {
        for index in 0..self.params.len() as i64 {
            self.decode_param(index)?;
        }
        for index in 0..self.params.len() {
            let param = self.params[index].clone();
            let Some(tensor) = tensor_type(&param) else { continue };
            let handle = self.tensor_handles[&ObjectIdentity::of(&param)].clone();
            let name = param.name.as_str().to_string();
            let path = vec![Step::Item(index as i64), Step::Name(name.clone())];
            let arg_name = format!("{}.{name}", self.func_name);
            let data = self.decode_dl_tensor(&param, &tensor, &handle, &arg_name, &path)?;
            let ty: Type = tensor.clone().into();
            let declared = Call::new_full(
                Some(ty.clone()),
                DECL_TENSOR.get()?.into(),
                Array::new(vec![data]),
                None,
                Array::new(vec![ty]),
                crate::ops::unknown_loc()?,
            )?;
            self.decl_tensors.push(Bind::new(param.clone(), Expr::from(declared))?.into());
            let mut size = Ex::from(int_imm(index_type(&tensor)?.dtype, 1));
            for extent in tensor.shape.iter() {
                size *= extent;
            }
            let alignment_args = vec![param.clone().into(), i32_imm(tensor.data_alignment as i64)?];
            let alignment: Stmt =
                Evaluate::new(call(void()?, &ASSUME_ALIGNED, alignment_args)?)?.into();
            let aligned_storage =
                self.simplify(size.not_equal(0).logical_and(self.direct_address()?))?;
            if is_const_int(&aligned_storage, 1) {
                self.decl_tensors.push(alignment);
            } else if !is_const_int(&aligned_storage, 0) {
                self.decl_tensors.push(If::new(aligned_storage, alignment, None)?.into());
            }
        }
        Ok(())
    }

    /// Whether the device exposes data addresses directly (`ffi::IsDirectAddressDevice`).
    fn direct_address(&self) -> Result<Expr> {
        let device_type = &self.device_type;
        device_type
            .less_equal(i32_imm(DL_CUDA_HOST as i64)?)
            .logical_or(device_type.equal(i32_imm(DL_CUDA_MANAGED as i64)?))
            .logical_or(device_type.equal(i32_imm(DL_ROCM as i64)?))
            .logical_or(device_type.equal(i32_imm(DL_ROCM_HOST as i64)?))
            .expr()
    }

    /// `DLTensorGetFieldPtr`: bind the shape or strides pointer of a DLTensor.
    fn field_ptr(&mut self, handle: &Var, field: i32, name: &str) -> Result<Var> {
        let pointer_type = PtrType::new(scalar(INT, 64)?, MemoryScope::global()?)?;
        let ptr = Var::new(name, pointer_type.clone())?;
        let args = vec![handle.clone().into(), i32_imm(0)?, i32_imm(field as i64)?];
        let load = call(pointer_type, &ABI_FIELD_GET, args)?;
        self.init_nest.push(Bind::new(ptr.clone(), load)?.into());
        Ok(ptr)
    }

    /// `LoadInt64ArrayElem`: `ptr[index]` of a shape or strides array.
    fn load_element(&self, ptr: &Var, index: i64) -> Result<Expr> {
        let args = vec![ptr.clone().into(), i32_imm(index)?, i32_imm(INT64_ARRAY_ELEM as i64)?];
        call(scalar(INT, 64)?, &ABI_FIELD_GET, args)
    }

    /// `BindCompactStrides`: without declared strides, given strides must be compact.
    fn bind_compact_strides(
        &mut self,
        tensor: &TensorType,
        name: &str,
        strides: &Var,
        strides_is_null: &Expr,
        path: &Path,
    ) -> Result<()> {
        let stype = index_type(tensor)?;
        let mut expect_stride = Ex::from(int_imm(stype.dtype, 1));
        let mut conditions = Vec::new();
        for k in (0..tensor.shape.len()).rev() {
            let extent = tensor.shape.get(k)?;
            let stride = self.load_element(strides, k as i64)?.cast(&stype);
            conditions.push(extent.equal(1).logical_or(expect_stride.equal(stride)));
            expect_stride *= &extent;
        }
        if conditions.is_empty() {
            return Ok(());
        }
        let mut all = Ex::from(int_imm(dtype(BOOL, 8), 1));
        for condition in conditions {
            all = all.logical_and(condition);
        }
        let index = Self::param_index(path).to_string();
        let parts = self.message(
            &["Mismatched ", name, ".strides on argument #", &index],
            &["`,\n  expected to be compact array"],
        )?;
        let check = AssertStmt::new(all.expr()?, text("ValueError")?, parts)?;
        let not_null = Not::new(strides_is_null)?;
        let check = If::new(not_null, check, None)?;
        self.asserts.push(SeqStmt::new(vec![check.into()])?.into());
        Ok(())
    }

    /// `BindRegularStrides`: declared strides bound to the given ones, or to the compact ones
    /// the shape implies when none are given.
    fn bind_regular_strides(
        &mut self,
        tensor: &TensorType,
        strides: &Var,
        shape: &Var,
        strides_is_null: &Expr,
        path: &Path,
    ) -> Result<()> {
        let mut stride_from_shape = Ex::from(int32(1));
        for k in (0..tensor.strides.len()).rev() {
            let extent_ty: PrimType = tensor.shape.get(k)?.ty.clone().try_cast()?;
            let explicit = self.load_element(strides, k as i64)?.cast(&extent_ty);
            let stride_path = extend(&extend(path, Step::Attr("strides")), Step::Item(k as i64));
            let value = if_then_else(strides_is_null, &stride_from_shape, explicit).expr()?;
            self.bind_scalar(&tensor.strides.get(k)?, &value, &stride_path, true)?;
            let extent = self.load_element(shape, k as i64)?.cast(&extent_ty);
            stride_from_shape *= extent;
        }
        Ok(())
    }

    /// `DecodeParamDLTensor`: check and bind the fields of a DLTensor argument; its typed data
    /// pointer.
    fn decode_dl_tensor(
        &mut self,
        param: &Var,
        tensor: &TensorType,
        handle: &Var,
        arg_name: &str,
        path: &Path,
    ) -> Result<Expr> {
        let buf_name = param.name.as_str().to_string();
        let index = Self::param_index(path);
        let index_str = index.to_string();
        let field = |kind: i32| -> Result<Vec<Expr>> {
            Ok(vec![handle.clone().into(), i32_imm(0)?, i32_imm(kind as i64)?])
        };

        // Null pointer.
        let null = call(scalar(BOOL, 8)?, &ISNULLPTR, vec![handle.clone().into()])?;
        self.emit_type_index_check(index, (!Ex::from(null)).expr()?, "Tensor")?;

        // ndim.
        let int32_ty = scalar(INT, 32)?;
        let ndim = call(int32_ty.clone(), &ABI_FIELD_GET, field(DL_TENSOR_NDIM)?)?;
        let rank = tensor.shape.len() as i64;
        let expected_ndim = Ex::from(int_imm(int32_ty.dtype, rank as i128)).equal(&ndim).expr()?;
        let parts = self.message(
            &["Mismatched ", &buf_name, ".ndim on argument #", &index_str],
            &["`,\n  expected ", &rank.to_string()],
        )?;
        self.emit_assert(expected_ndim, "ValueError", parts)?;

        // dtype.
        let element = tensor.dtype.dtype;
        let (uint8, uint16) = (scalar(UINT, 8)?, scalar(UINT, 16)?);
        let code = call(uint8.clone(), &ABI_FIELD_GET, field(DL_TENSOR_TYPE_CODE)?)?
            .equal(int_imm(uint8.dtype, element.code as i128)?);
        let bits = call(uint8.clone(), &ABI_FIELD_GET, field(DL_TENSOR_TYPE_BITS)?)?
            .equal(int_imm(uint8.dtype, element.bits as i128)?);
        let lanes = call(uint16.clone(), &ABI_FIELD_GET, field(DL_TENSOR_TYPE_LANES)?)?
            .equal(int_imm(uint16.dtype, lanes_of(element)? as i128)?);
        let matches = code.logical_and(bits).logical_and(lanes).expr()?;
        let sub_byte =
            element == dtype(INT, 1) || element == dtype(INT, 4) || element == dtype(UINT, 4);
        if !sub_byte {
            let parts = self.message(
                &["Mismatched ", &buf_name, ".dtype on argument #", &index_str],
                &["`,\n  expected ", &dtype_str(element)],
            )?;
            self.emit_assert(matches, "TypeError", parts)?;
        }

        // shape.
        let shape_ptr = self.field_ptr(handle, DL_TENSOR_SHAPE, &format!("{arg_name}_shape"))?;
        if !sub_byte {
            for k in 0..tensor.shape.len() {
                let extent = tensor.shape.get(k)?;
                let extent_ty: PrimType = extent.ty.clone().try_cast()?;
                let value = self.load_element(&shape_ptr, k as i64)?.cast(&extent_ty).expr()?;
                let shape_path = extend(&extend(path, Step::Attr("shape")), Step::Item(k as i64));
                self.bind_scalar(&extent, &value, &shape_path, true)?;
            }
        }

        // strides.
        let strides_ptr =
            self.field_ptr(handle, DL_TENSOR_STRIDES, &format!("{arg_name}_strides"))?;
        let strides_is_null = call(scalar(BOOL, 8)?, &ISNULLPTR, vec![strides_ptr.clone().into()])?;
        if tensor.strides.is_empty() {
            self.bind_compact_strides(tensor, &buf_name, &strides_ptr, &strides_is_null, path)?;
        } else {
            self.bind_regular_strides(tensor, &strides_ptr, &shape_ptr, &strides_is_null, path)?;
        }

        // byte_offset.
        let data_bytes = storage_bytes(element);
        let uint64 = scalar(UINT, 64)?;
        let byte_offset_path = extend(path, Step::Attr("byte_offset"));
        if let Some(offset) = tensor.elem_offset.as_node::<IntImmObj>() {
            let expected: Expr =
                crate::ops::big_int_imm(uint64.dtype, &offset.value * &BigInt::from(data_bytes))?
                    .into();
            let actual = call(uint64.clone(), &ABI_FIELD_GET, field(DL_TENSOR_BYTE_OFFSET)?)?;
            self.bind_scalar(&expected, &actual, &byte_offset_path, true)?;
        } else {
            let size = make_const(uint64.dtype, data_bytes)?;
            let divisible = call(uint64.clone(), &ABI_FIELD_GET, field(DL_TENSOR_BYTE_OFFSET)?)?
                .truncmod(&size)
                .equal(make_const(uint64.dtype, 0)?)
                .expr()?;
            let parts =
                Array::new(vec![text("Tensor byte offset must be divisible by its element size")?]);
            self.emit_assert(divisible, "ValueError", parts)?;
            let offset_ty: PrimType = tensor.elem_offset.ty.clone().try_cast()?;
            let value = call(uint64.clone(), &ABI_FIELD_GET, field(DL_TENSOR_BYTE_OFFSET)?)?
                .div(&size)
                .cast(&offset_ty)
                .expr()?;
            if self.bind_scalar(&tensor.elem_offset, &value, &byte_offset_path, true)?
                && tensor.offset_factor > 1
            {
                let offset = &tensor.elem_offset;
                let factor = int_imm(offset_ty.dtype, tensor.offset_factor as i128)?;
                let zero = int_imm(offset_ty.dtype, 0)?;
                let condition = self.simplify(offset.truncmod(&factor).equal(&zero))?;
                if is_const_int(&condition, 0) {
                    let at = self.render(&byte_offset_path);
                    tvm_ffi::bail!(INTERNAL_ERROR, "Bind have an unmet assertion at {at}");
                }
                if !is_const_int(&condition, 1) {
                    let alignment = (tensor.offset_factor as i64 * data_bytes).to_string();
                    let parts = self.message(
                        &["Misaligned Tensor data on argument #", &index_str],
                        &["`,\n  expected data alignment=", &alignment, " bytes"],
                    )?;
                    self.emit_assert(condition, "ValueError", parts)?;
                }
            }
        }

        // device.
        let actual_device_type =
            call(int32_ty.clone(), &ABI_FIELD_GET, field(DL_TENSOR_DEVICE_TYPE)?)?;
        if let Some(device) = self.device_type.as_node::<IntImmObj>() {
            let device = device.value.to_i64().unwrap_or(0);
            let condition = self.simplify(Ex::from(i32_imm(device)?).equal(&actual_device_type))?;
            if !is_const_int(&condition, 1) {
                let parts = self.message(
                    &["Mismatched ", &buf_name, ".device_type on argument #", &index_str],
                    &["`,\n  expected ", device_name(device)?],
                )?;
                self.emit_assert(condition, "ValueError", parts)?;
            }
        } else {
            let device_type = self.device_type.clone();
            self.bind_scalar(
                &device_type,
                &actual_device_type,
                &extend(path, Step::Attr("device_type")),
                true,
            )?;
        }
        let device_id = self.device_id.clone();
        let actual_device_id = call(int32_ty, &ABI_FIELD_GET, field(DL_TENSOR_DEVICE_ID)?)?;
        self.bind_scalar(
            &device_id,
            &actual_device_id,
            &extend(path, Step::Attr("device_id")),
            true,
        )?;

        // data pointer.
        let raw_data = call(PtrType::void()?, &ABI_FIELD_GET, field(DL_TENSOR_DATA)?)?;
        let data_ptr_type = tensor.data_ptr_type()?;
        let typed_data = call(data_ptr_type, &REINTERPRET, vec![raw_data])?;
        let mut alloc_size = Ex::from(int_imm(index_type(tensor)?.dtype, 1));
        for extent in tensor.shape.iter() {
            alloc_size *= extent;
        }
        let empty_alloc = alloc_size.equal(0).cast(scalar(BOOL, 8)?).expr()?;
        let null = call(scalar(BOOL, 8)?, &ISNULLPTR, vec![typed_data.clone()])?;
        let non_null = Ex::from(&empty_alloc).logical_or(!Ex::from(null)).expr()?;
        let parts = self.message(
            &[&buf_name, " data pointer is NULL on argument #", &index_str],
            &["`,\n  expected non-NULL data pointer"],
        )?;
        self.asserts.push(AssertStmt::new(non_null, text("ValueError")?, parts)?.into());
        if tensor.data_alignment > 1 {
            let direct_address = self.direct_address()?;
            let as_handle = call(PtrType::void()?, &REINTERPRET, vec![typed_data.clone()])?;
            let as_int = call(uint64.clone(), &REINTERPRET, vec![as_handle])?;
            let alignment = int_imm(uint64.dtype, tensor.data_alignment as i128)?;
            let aligned = as_int.truncmod(alignment).equal(int_imm(uint64.dtype, 0)?);
            let condition = Ex::from(&empty_alloc)
                .logical_or(!Ex::from(direct_address))
                .logical_or(aligned)
                .expr()?;
            let parts = self.message(
                &["Misaligned Tensor data on argument #", &index_str],
                &["`,\n  expected data alignment=", &tensor.data_alignment.to_string(), " bytes"],
            )?;
            self.asserts.push(AssertStmt::new(condition, text("ValueError")?, parts)?.into());
        }
        Ok(typed_data)
    }
}

fn attr(out: &mut std::string::String, first_printed: &mut bool, name: &str) {
    if !*first_printed {
        out.push_str(name);
        *first_printed = true;
    } else {
        out.push('.');
        out.push_str(name);
    }
}

/// The name of a variable of a `PrimType` (C++ `as<PrimVar>()`), if `expr` is one.
fn prim_var_name(expr: &Expr) -> Option<std::string::String> {
    let var = expr.as_node::<VarObj>()?;
    var.ty.as_node::<PrimTypeObj>()?;
    Some(var.name.as_str().to_string())
}

/// `PrimType::lanes()`, which a scalable vector has none of.
fn lanes_of(dtype: DLDataType) -> Result<i64> {
    match dtype.lanes as i16 {
        lanes if lanes < 0 => Err(error!(
            INTERNAL_ERROR,
            "Can't fetch the lanes of a scalable vector at a compile time."
        )),
        lanes => Ok(lanes as i64),
    }
}

/// `std::ostream << double` with the default format (`%g`, six significant digits).
fn format_double(value: f64) -> std::string::String {
    if value.is_nan() {
        return if value.is_sign_negative() { "-nan" } else { "nan" }.to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let precision = 6i32;
    let exponent = format!("{:.*e}", (precision - 1) as usize, value);
    let (mantissa, exp) = exponent.split_once('e').expect("an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    if exp < -4 || exp >= precision {
        let mantissa = trim_zeros(mantissa);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        let decimals = (precision - 1 - exp).max(0) as usize;
        trim_zeros(&format!("{value:.decimals$}")).to_string()
    }
}

fn trim_zeros(text: &str) -> &str {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text
    }
}
