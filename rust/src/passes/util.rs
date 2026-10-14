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

//! Pass plumbing and small IR helpers shared by the Rust passes.

use tvm_ffi::{
    Any, AnyView, Array, DLDataType, DLDataTypeCode, Function as FfiFunction, Map, MutateContext,
    MutateDispatch, Mutator, ObjectRefCore, RValueRef, Result, String,
};

use crate::ir::{BaseFunc, DictAttrs, GlobalVar, IRModule, IntImm, Location, PrimType, SeqStmt, Stmt};
use crate::tirx::Function;
use crate::transform::{Pass, PassContext, PassInfo};

/// A `tirx.FunctionPass` that runs `rewrite` on every `tirx.Function` of a module.
///
/// Returning `None` removes the function, as a C++ pass function returning `nullopt` does.
pub fn function_pass<F>(name: &str, opt_level: i32, rewrite: F) -> Result<Pass>
where
    F: Fn(Function, &IRModule, &PassContext) -> Result<Option<Function>> + 'static,
{
    let callback = FfiFunction::from_packed(move |args: &[AnyView]| -> Result<Any> {
        let func = RValueRef::<Function>::try_from(args[0])?.into_inner();
        let module = IRModule::try_from(args[1])?;
        let context = PassContext::try_from(args[2])?;
        Ok(match rewrite(func, &module, &context)? {
            Some(func) => Any::from(func),
            None => Any::new(),
        })
    });
    tvm_ffi::cached_global_func!("tirx.transform.CreateFunctionPass")
        .call_tuple((callback, PassInfo::new(opt_level, String::from(name))))?
        .try_into()
}

/// A `transform.ModulePass` that runs `rewrite` on the whole module.
pub fn module_pass<F>(name: &str, opt_level: i32, rewrite: F) -> Result<Pass>
where
    F: Fn(IRModule, &PassContext) -> Result<IRModule> + 'static,
{
    let callback = FfiFunction::from_packed(move |args: &[AnyView]| -> Result<Any> {
        let module = RValueRef::<IRModule>::try_from(args[0])?.into_inner();
        let context = PassContext::try_from(args[1])?;
        Ok(Any::from(rewrite(module, &context)?))
    });
    tvm_ffi::cached_global_func!("transform.MakeModulePass")
        .call_tuple((callback, PassInfo::new(opt_level, String::from(name))))?
        .try_into()
}

/// `transform.Sequential(passes, opt_level=0, name)`.
pub fn sequential(passes: Vec<Pass>, name: &str) -> Result<Pass> {
    tvm_ffi::cached_global_func!("transform.Sequential")
        .call_tuple((Array::new(passes), 0i32, String::from(name)))?
        .try_into()
}

/// The C++ pass a `tirx.transform.<Name>()` factory returns.
pub fn native_pass(factory: &str) -> Result<Pass> {
    FfiFunction::get_global(factory)?.call_packed(&[])?.try_into()
}

/// `WithAttr(func, key, value)`.
pub fn with_attr(func: &Function, key: &str, value: Any) -> Function {
    func.with_attrs(dict_with(&func.attrs, key, value))
}

/// The process-wide `UnknownLoc()` singleton, as a `Location`.
pub fn unknown_loc() -> Result<Location> {
    Ok(crate::ir::UnknownLoc::new()?.into())
}

pub fn int32_dtype() -> DLDataType {
    DLDataType { code: DLDataTypeCode::kDLInt as u8, bits: 32, lanes: 1 }
}

/// `PrimType(dtype)` through the native constructor.
pub fn prim_type(dtype: DLDataType) -> Result<PrimType> {
    tvm_ffi::cached_global_func!("ir.PrimType").call_tuple((dtype,))?.try_into()
}

/// `IntImm::Int32(value)`, the C++ meaning of a bare integer literal such as `Evaluate(0)`.
pub fn int32(value: i64) -> Result<IntImm> {
    tvm_ffi::cached_global_func!("ir.IntImm")
        .call_tuple((int32_dtype(), tvm_ffi::BigInt::from(value), unknown_loc()?))?
        .try_into()
}

/// `IntImm(dtype, value)` through the native constructor.
pub fn int_imm(dtype: DLDataType, value: i128) -> Result<IntImm> {
    tvm_ffi::cached_global_func!("ir.IntImm")
        .call_tuple((dtype, tvm_ffi::BigInt::from(value), unknown_loc()?))?
        .try_into()
}

/// `prim::MakeConst(dtype, value)`, which has no global: an `IntImm` or `FloatImm` by type code,
/// broadcast for a fixed-length vector type. C++ `a >= 0` and `a - 1` build their constant so.
pub fn make_const(dtype: DLDataType, value: i64) -> Result<crate::ir::Expr> {
    let scalar = DLDataType { lanes: 1, ..dtype };
    let element: crate::ir::Expr = match dtype.code {
        // kDLInt, kDLUInt, kDLBool
        0 | 1 | 6 => int_imm(scalar, value as i128)?.into(),
        // kDLFloat, kDLBfloat, and the float8/6/4 codes
        2 | 4 | 7..=17 => tvm_ffi::cached_global_func!("ir.FloatImm")
            .call_tuple((scalar, value as f64, unknown_loc()?))?
            .try_into()?,
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
    let lanes: crate::ir::Expr = int32(dtype.lanes as i64)?.into();
    tvm_ffi::cached_global_func!("prim.Broadcast")
        .call_tuple((element, lanes, unknown_loc()?))?
        .try_into()
}

/// The dtype of a PrimExpr (`expr.ty()`); an error for any other expression.
pub fn prim_dtype(expr: &crate::ir::Expr) -> Result<DLDataType> {
    match expr.ty.as_node::<crate::ir::PrimTypeObj>() {
        Some(ty) => Ok(ty.dtype),
        None => Err(tvm_ffi::Error::new(tvm_ffi::TYPE_ERROR, "expected a PrimType expression", "")),
    }
}

/// Typed re-entry from a `#[dispatch(mutate)]` handler: `m.mutate_as(self, &op.body)?` gives the
/// mutated field back as its own type.
pub trait MutatorExt {
    fn mutate_as<D, T>(&mut self, dispatch: &mut D, value: &T) -> Result<T>
    where
        D: MutateDispatch,
        T: TryFrom<Any, Error = tvm_ffi::Error>,
        for<'x> AnyView<'x>: From<&'x T>;

    /// Default descent of `value`, the C++ `Parent::Mutate_(op)`, as its own type.
    fn default_mutate_as<D, T>(&mut self, dispatch: &mut D, value: &T) -> Result<T>
    where
        D: MutateDispatch,
        T: TryFrom<Any, Error = tvm_ffi::Error>,
        for<'x> AnyView<'x>: From<&'x T>;
}

impl MutatorExt for Mutator {
    fn mutate_as<D, T>(&mut self, dispatch: &mut D, value: &T) -> Result<T>
    where
        D: MutateDispatch,
        T: TryFrom<Any, Error = tvm_ffi::Error>,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.mutate(dispatch, value)?.try_into()
    }

    fn default_mutate_as<D, T>(&mut self, dispatch: &mut D, value: &T) -> Result<T>
    where
        D: MutateDispatch,
        T: TryFrom<Any, Error = tvm_ffi::Error>,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.default_mutate(dispatch, value)?.try_into()
    }
}

/// Typed re-entry from a mutation policy: `ctx.mutate_as(&op.body)?`.
pub trait ContextExt {
    fn mutate_as<T>(&mut self, value: &T) -> Result<T>
    where
        T: TryFrom<Any, Error = tvm_ffi::Error>,
        for<'x> AnyView<'x>: From<&'x T>;
}

impl<S> ContextExt for MutateContext<'_, S> {
    fn mutate_as<T>(&mut self, value: &T) -> Result<T>
    where
        T: TryFrom<Any, Error = tvm_ffi::Error>,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.mutate(value)?.try_into()
    }
}

/// The C++ implicit `Stmt -> SeqStmt` conversion: a SeqStmt passes through, anything else is
/// wrapped (with flattening and no-op removal) by the native constructor.
pub fn seq_stmt_from(stmt: Stmt) -> Result<SeqStmt> {
    tvm_ffi::cached_global_func!("ir.SeqStmt")
        .call_packed(&[AnyView::from(&stmt), AnyView::new()])?
        .try_into()
}

/// `attrs.GetAttr<Any>(key)`.
pub fn get_attr(attrs: &DictAttrs, key: &str) -> Result<Option<Any>> {
    attrs.dict.get(&String::from(key))
}

/// `attrs.HasNonzeroAttr(key)`: the attribute, read as an integer, is non-zero.
pub fn has_nonzero_attr(attrs: &DictAttrs, key: &str) -> Result<bool> {
    Ok(match get_attr(attrs, key)? {
        Some(value) => i64::try_from(value)? != 0,
        None => false,
    })
}

/// `attrs` with `key` set to `value` (C++ `WithAttr`); an existing key keeps its position.
pub fn dict_with(attrs: &DictAttrs, key: &str, value: Any) -> DictAttrs {
    let mut entries: Vec<(String, Any)> = Vec::new();
    let mut replaced = false;
    for (k, v) in attrs.dict.iter() {
        if k.as_str() == key {
            entries.push((k, value.clone()));
            replaced = true;
        } else {
            entries.push((k, v));
        }
    }
    if !replaced {
        entries.push((String::from(key), value));
    }
    DictAttrs::new(entries.into_iter().collect::<Map<String, Any>>())
}

/// `mod->Update(gvar, func)`: the entry for `gvar` is replaced in place.
pub fn module_with_function(module: &IRModule, gvar: &GlobalVar, func: BaseFunc) -> IRModule {
    let functions: Map<GlobalVar, BaseFunc> = module
        .functions
        .iter()
        .map(|(key, value)| if key.same_as(gvar) { (key, func.clone()) } else { (key, value) })
        .collect();
    IRModule::new(
        functions,
        module.source_map.clone(),
        module.attrs.clone(),
        module.global_infos.clone(),
        module.global_var_map.clone(),
    )
}

/// `attrs` without `key` (C++ `WithoutAttr`).
pub fn dict_without(attrs: &DictAttrs, key: &str) -> DictAttrs {
    DictAttrs::new(
        attrs.dict.iter().filter(|(k, _)| k.as_str() != key).collect::<Map<String, Any>>(),
    )
}
