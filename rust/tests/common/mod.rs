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

//! Helpers shared by the binding smoke tests.

#![allow(dead_code)]

use std::mem::size_of;
use std::path::PathBuf;
use std::sync::OnceLock;

use tvm::tvm_ffi::tvm_ffi_sys::{TVMFFIGetTypeInfo, TVMFFITypeInfo};
use tvm::tvm_ffi::{AnyCompatible, AnyView, Function, Module, ObjectCore};

static TVM_COMPILER: OnceLock<Module> = OnceLock::new();

/// Load `libtvm_compiler` so its type keys are in the FFI type registry.
pub fn load_tvm_compiler() {
    TVM_COMPILER.get_or_init(|| {
        let default = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("build")
            .join("lib")
            .join("libtvm_compiler.so");
        let library =
            std::env::var_os("TVM_COMPILER_LIBRARY").map(PathBuf::from).unwrap_or(default);
        Module::load_from_file(library.to_string_lossy()).unwrap()
    });
}

pub fn global(name: &str) -> Function {
    Function::get_global(name).unwrap()
}

pub fn structural_equal<L: AnyCompatible, R: AnyCompatible>(lhs: &L, rhs: &R) -> bool {
    let equal = global("ffi.StructuralEqual")
        .call_packed(&[
            AnyView::from(lhs),
            AnyView::from(rhs),
            AnyView::from(&false),
            AnyView::from(&false),
        ])
        .unwrap();
    bool::try_from(equal).unwrap()
}

pub fn runtime_type_info<N: ObjectCore>() -> &'static TVMFFITypeInfo {
    let pointer = unsafe { TVMFFIGetTypeInfo(N::type_index()) };
    assert!(!pointer.is_null(), "missing type info for {}", N::TYPE_KEY);
    unsafe { &*pointer }
}

/// The Rust struct agrees with the registry on key, depth, parent, and (when complete) size.
pub fn assert_type_contract<N: ObjectCore, P: ObjectCore>(complete: bool) {
    let info = runtime_type_info::<N>();
    assert_eq!(info.type_key.as_str(), N::TYPE_KEY);
    assert_eq!(info.type_depth, N::TYPE_DEPTH, "depth of {}", N::TYPE_KEY);
    assert_eq!(N::TYPE_DEPTH, P::TYPE_DEPTH + 1, "parent of {}", N::TYPE_KEY);
    let parent = unsafe { *info.type_acenstors.add(P::TYPE_DEPTH as usize) };
    assert_eq!(unsafe { (*parent).type_index }, P::type_index(), "parent of {}", N::TYPE_KEY);
    if complete {
        let metadata = unsafe { info.metadata.as_ref() }
            .unwrap_or_else(|| panic!("{} has no metadata", N::TYPE_KEY));
        assert_eq!(metadata.total_size as usize, size_of::<N>(), "size of {}", N::TYPE_KEY);
    }
}

#[allow(unused_macros)]
macro_rules! assert_type_contracts {
    ($complete:expr; $($node:ty : $parent:ty),* $(,)?) => {
        $($crate::common::assert_type_contract::<$node, $parent>($complete);)*
    };
}
#[allow(unused_imports)]
pub(crate) use assert_type_contracts;
