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

use std::path::PathBuf;

use tvm_ffi::{Any, AnyView, DLDataType, DLDataTypeExt, Function, Module, Result};
use tvm_tirx_bindings::analysis::{
    collect_structural_vars, contains_var_in_def_region, contains_var_outside_def_region,
};
use tvm_tirx_bindings::generated::ir::PrimType;
use tvm_tirx_bindings::generated::tirx::{Let, Var};

fn compiler_library() -> PathBuf {
    std::env::var_os("TVM_COMPILER_LIB")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../build/lib")
                .join(format!(
                    "{}tvm_compiler{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                ))
        })
}

fn make_var(name: &str, ty: &PrimType) -> Result<Var> {
    let name = tvm_ffi::String::from(name);
    let none = Any::default();
    Function::get_global("tirx.Var")?
        .call_packed(&[
            AnyView::from(&name),
            AnyView::from(ty),
            AnyView::from(&none),
        ])?
        .try_into()
}

#[test]
fn variable_analyses_follow_structural_definition_regions() -> Result<()> {
    let compiler_library = compiler_library();
    assert!(
        compiler_library.is_file(),
        "set TVM_COMPILER_LIB to a built compiler library (missing {})",
        compiler_library.display()
    );
    let _compiler = Module::load_from_file(compiler_library.to_string_lossy())?;

    let none = Any::default();
    let dtype = DLDataType::try_from_str("int32")?;
    let ty: PrimType = Function::get_global("ir.PrimType")?
        .call_packed(&[AnyView::from(&dtype)])?
        .try_into()?;
    let x = make_var("v", &ty)?;
    let y = make_var("v", &ty)?;
    let expr: Let = Function::get_global("tirx.Let")?
        .call_packed(&[
            AnyView::from(&x),
            AnyView::from(&y),
            AnyView::from(&y),
            AnyView::from(&none),
        ])?
        .try_into()?;

    assert!(!x.same_as(&y));
    assert!(expr.var.same_as(&x));
    let vars = collect_structural_vars(&expr)?;
    assert_eq!(vars.len(), 2);
    assert!(vars.iter().any(|var| var.same_as(&x)));
    assert!(vars.iter().any(|var| var.same_as(&y)));
    assert!(contains_var_in_def_region(&expr, &x)?);
    assert!(!contains_var_outside_def_region(&expr, &x)?);
    assert!(!contains_var_in_def_region(&expr, &y)?);
    assert!(contains_var_outside_def_region(&expr, &y)?);
    Ok(())
}
