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

//! Typed wrappers of the native functions Rust calls rather than writes again: printing,
//! structural comparison, the target context, layouts, analyses. Each calls the global C++ (or,
//! where noted, Python) registers for it.

use tvm_ffi::object::ObjectRef;
use tvm_ffi::{Any, AnyView, Array, FieldGetter, Map, ObjectCore, ObjectRefCore, Result, String};

use crate::ir::{Call, Expr, LambdaExpr, OpObj, Stmt, Type, UniqueNameSupply, Var};
use crate::runtime::{MemoryScope, MemoryScopeObj};
use crate::target::Target;
use crate::tirx::{Axis, Function, Layout, TileLayout};

// ---------------------------------------------------------------------------------------------
// Printing and structural comparison

/// How C++ prints a node in an error message (`ffi.ReprPrint`), such as `T.float32`.
pub fn repr<T>(value: &T) -> std::string::String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    tvm_ffi::cached_global_func!("ffi.ReprPrint")
        .call_packed(&[AnyView::from(value)])
        .and_then(String::try_from)
        .map(|text| text.as_str().to_string())
        .unwrap_or_default()
}

/// C++ `ffi::StructuralEqual()(lhs, rhs)`, which does not map free variables.
pub fn structural_equal<L, R>(lhs: &L, rhs: &R) -> Result<bool>
where
    for<'a> AnyView<'a>: From<&'a L> + From<&'a R>,
{
    let (map_free_vars, skip_tensor_content) = (false, false);
    tvm_ffi::cached_global_func!("ffi.StructuralEqual")
        .call_packed(&[
            AnyView::from(lhs),
            AnyView::from(rhs),
            AnyView::from(&map_free_vars),
            AnyView::from(&skip_tensor_content),
        ])?
        .try_into()
}

/// C++ `ffi::StructuralHash()(value)`.
pub fn structural_hash<T>(value: &T) -> Result<i64>
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let flag = <AnyView as From<&bool>>::from;
    let (map_free_vars, skip_tensor_content) = (false, false);
    tvm_ffi::cached_global_func!("ffi.StructuralHash")
        .call_packed(&[AnyView::from(value), flag(&map_free_vars), flag(&skip_tensor_content)])?
        .try_into()
}

/// C++ `prim::ExprDeepEqual()(lhs, rhs)`: equal expressions, variables compared by identity.
pub fn deep_equal(lhs: &Expr, rhs: &Expr) -> Result<bool> {
    tvm_ffi::cached_global_func!("prim.expr_deep_equal")
        .call_tuple((lhs.clone(), rhs.clone()))?
        .try_into()
}

// ---------------------------------------------------------------------------------------------
// Expressions and names

impl Call {
    /// C++ `Call::Validate()`: the operator's validator, if it has one, on this call.
    pub fn validate(&self) -> Result<()> {
        match self.op.as_node::<OpObj>() {
            Some(op) => super::build::validate_call(op, self),
            None => Ok(()),
        }
    }

    /// C++ `Call::ReinferType()`: the type the operator gives this call now (its
    /// `TFixedReturnType`, else what its `FInferType` infers), the return type of a callee of a
    /// function type, otherwise `MissingType`.
    pub fn reinfer_type(&self) -> Result<Type> {
        super::build::reinfer_call_type(self)
    }
}

impl LambdaExpr {
    /// C++ `LambdaExpr::Apply(args)`: the body with the variables replaced by `args`.
    pub fn apply(&self, args: Array<Expr>) -> Result<Expr> {
        tvm_ffi::cached_global_func!("ir.LambdaExprApply")
            .call_tuple((self.clone(), args))?
            .try_into()
    }
}

impl UniqueNameSupply {
    /// C++ `ReserveName(name, add_prefix)`: `name` is taken from now on.
    pub fn reserve_name(&self, name: &str, add_prefix: bool) -> Result<String> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupply_ReserveName")
            .call_tuple((self.clone(), String::from(name), add_prefix))?
            .try_into()
    }

    /// C++ `FreshName(name, add_prefix, add_underscore)`: `name`, or `name` with a number, that
    /// is not taken yet.
    pub fn fresh_name(&self, name: &str, add_prefix: bool, add_underscore: bool) -> Result<String> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupply_FreshName")
            .call_tuple((self.clone(), String::from(name), add_prefix, add_underscore))?
            .try_into()
    }
}

// ---------------------------------------------------------------------------------------------
// Targets

impl Target {
    /// C++ `Target(tag_or_config)`, such as `Target("llvm")`.
    pub fn new(tag_or_config: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("target.Target")
            .call_tuple((String::from(tag_or_config),))?
            .try_into()
    }

    /// C++ `Target::Current(allow_not_defined)`: the target of the innermost target scope. With
    /// none, an error unless `allow_not_defined`.
    pub fn current(allow_not_defined: bool) -> Result<Option<Self>> {
        tvm_ffi::cached_global_func!("target.TargetCurrent")
            .call_tuple((allow_not_defined,))?
            .try_into()
    }

    /// C++ `target->GetTargetDeviceType()`.
    pub fn device_type(&self) -> Result<i64> {
        tvm_ffi::cached_global_func!("target.TargetGetDeviceType")
            .call_tuple((self.clone(),))?
            .try_into()
    }

    /// C++ `target->str()`.
    pub fn str(&self) -> Result<String> {
        tvm_ffi::cached_global_func!("target.TargetAsJSON").call_tuple((self.clone(),))?.try_into()
    }

    /// C++ `Target::WithHost(target, host)`: this target with its host replaced, or removed.
    pub fn with_host(&self, host: Option<&Target>) -> Result<Self> {
        let host = host.map(AnyView::from).unwrap_or_default();
        tvm_ffi::cached_global_func!("target.WithHost")
            .call_packed(&[AnyView::from(self), host])?
            .try_into()
    }

    /// C++ `With<Target>`: this target is the current one until the scope is dropped.
    pub fn enter_scope(&self) -> Result<TargetScope> {
        tvm_ffi::cached_global_func!("target.TargetEnterScope").call_tuple((self.clone(),))?;
        Ok(TargetScope(self.clone()))
    }

    /// `codegen.target_has_feature(feature, target)`, an LLVM target query; false in a build
    /// without LLVM.
    pub fn has_llvm_feature(&self, feature: &str) -> Result<bool> {
        let Ok(has_feature) = tvm_ffi::Function::get_global("codegen.target_has_feature") else {
            return Ok(false);
        };
        has_feature.call_tuple((String::from(feature), self.clone()))?.try_into()
    }

    /// `tvm.support.nvcc.get_compute_version(target)`, which Python registers: the CUDA compute
    /// version such as `"9.0"`, or `None` when Python has not registered it.
    pub fn cuda_compute_version(&self) -> Result<Option<String>> {
        let Ok(version) = tvm_ffi::Function::get_global("tvm.support.nvcc.get_compute_version")
        else {
            return Ok(None);
        };
        Ok(Some(version.call_tuple((self.clone(),))?.try_into()?))
    }
}

/// The scope of [`Target::enter_scope`]: dropping it restores the enclosing target.
pub struct TargetScope(Target);

impl Drop for TargetScope {
    fn drop(&mut self) {
        let _ =
            tvm_ffi::cached_global_func!("target.TargetExitScope").call_tuple((self.0.clone(),));
    }
}

// ---------------------------------------------------------------------------------------------
// Memory scopes, tensors and functions

impl MemoryScope {
    /// The value of the enum attribute column `column` for this scope (C++
    /// `MemoryScope::GetAttr`), `None` when the column has none for it.
    pub fn attr(&self, column: &str) -> Result<Option<Any>> {
        let Some(state) = tvm_ffi::get_type_attr(MemoryScopeObj::type_index(), "__ffi_enum__")
        else {
            return Ok(None);
        };
        let getter = FieldGetter::new(state.type_index(), "attrs")?;
        let state: ObjectRef = state.try_into()?;
        let columns = getter.get_any(&**ObjectRefCore::data(&state))?;
        let (count, get) = (
            tvm_ffi::cached_global_func!("ffi.DictCount"),
            tvm_ffi::cached_global_func!("ffi.DictGetItem"),
        );
        let column = String::from(column);
        if i64::try_from(count.call_tuple((columns.clone(), column.clone()))?)? == 0 {
            return Ok(None);
        }
        let values = get.call_tuple((columns, column))?;
        if i64::try_from(count.call_tuple((values.clone(), self.clone()))?)? == 0 {
            return Ok(None);
        }
        Ok(Some(get.call_tuple((values, self.clone()))?))
    }
}

impl Var {
    /// C++ `TensorVar::IsScalar(alloc_or_decl)`: whether this tensor variable supports scalar
    /// tensor syntax.
    pub fn tensor_is_scalar(&self, alloc_or_decl: bool) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.TensorIsScalar")
            .call_tuple((self.clone(), alloc_or_decl))?
            .try_into()
    }
}

impl Function {
    /// C++ `RenewDefs(func)`: a copy whose every definition is a fresh variable.
    pub fn renew_defs(&self) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.RenewDef").call_tuple((self.clone(),))?.try_into()
    }
}

/// C++ `tirx::UndefinedVars(stmt, defs)`: the variables `stmt` uses without defining them, other
/// than `defs`.
pub fn undefined_vars(stmt: &Stmt, defs: &Array<Var>) -> Result<Array<Var>> {
    tvm_ffi::cached_global_func!("tirx.analysis.UndefinedVars")
        .call_tuple((stmt.clone(), defs.clone()))?
        .try_into()
}

// ---------------------------------------------------------------------------------------------
// Layouts

impl Axis {
    /// C++ `axis->IsMemoryAxis()`.
    pub fn is_memory_axis(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.AxisIsMemoryAxis")
            .call_tuple((self.clone(),))?
            .try_into()
    }

    /// C++ `axis->IsThreadAxis()`.
    pub fn is_thread_axis(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.AxisIsThreadAxis")
            .call_tuple((self.clone(),))?
            .try_into()
    }
}

impl Layout {
    /// C++ `layout->GetSpan(axis)`: the extent of memory the layout covers along `axis`.
    pub fn span(&self, axis: &str) -> Result<Expr> {
        tvm_ffi::cached_global_func!("tirx.LayoutGetSpan")
            .call_tuple((self.clone(), String::from(axis)))?
            .try_into()
    }

    /// C++ `layout->Canonicalize()`.
    pub fn canonicalize(&self) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.LayoutCanonicalize")
            .call_tuple((self.clone(),))?
            .try_into()
    }

    /// C++ `layout->Apply(coord, shape)`: the offset along each axis of the element at `coord`
    /// of a tensor of `shape`.
    pub fn apply_with_shape(
        &self,
        coord: &Array<Expr>,
        shape: &Array<Expr>,
    ) -> Result<Map<String, Expr>> {
        tvm_ffi::cached_global_func!("tirx.LayoutApplyWithShape")
            .call_tuple((self.clone(), coord.clone(), shape.clone()))?
            .try_into()
    }
}

impl TileLayout {
    /// C++ `layout->IsTrivial()`.
    pub fn is_trivial(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.TileLayoutIsTrivial")
            .call_tuple((self.clone(),))?
            .try_into()
    }
}
