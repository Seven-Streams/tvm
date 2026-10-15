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
    Any, AnyView, Array, Function as FfiFunction, Map, MutateContext, MutateDispatch, Mutator,
    ObjectRefCore, RValueRef, Result, String,
};

use crate::ir::{BaseFunc, DictAttrs, GlobalVar, IRModule};
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
pub fn with_attr(func: &Function, key: &str, value: impl Into<Any>) -> Function {
    func.with_attrs(dict_with(&func.attrs, key, value.into()))
}

pub use crate::ops::{int32, int32_dtype, int_imm, make_const, prim_dtype, prim_type, unknown_loc};

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
pub fn module_with_function(
    module: &IRModule,
    gvar: &GlobalVar,
    func: impl Into<BaseFunc>,
) -> IRModule {
    let func = func.into();
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
