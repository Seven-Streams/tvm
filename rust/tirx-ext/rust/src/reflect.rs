// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Shared reflected-field iteration — the single home of the ancestor-chain
//! walk and of the zero-field guard (`fields == NULL` on e.g. `ffi.Object`,
//! which `slice::from_raw_parts` forbids even for len 0).

use std::ops::ControlFlow;

use tvm_ffi::tvm_ffi_sys::{TVMFFIFieldInfo, TVMFFIGetTypeInfo};

/// Visit every reflected field of `type_index` and then of its ancestors
/// (single-inheritance chain, derived type first). The callback's
/// `ControlFlow::Break` value short-circuits and is returned.
///
/// # Safety
/// `type_index` must be a registered type index.
pub(crate) unsafe fn for_each_field<B>(
    type_index: i32,
    mut f: impl FnMut(&'static TVMFFIFieldInfo) -> ControlFlow<B>,
) -> Option<B> {
    let mut info = TVMFFIGetTypeInfo(type_index);
    while !info.is_null() {
        // Zero-field types have fields == NULL; skip the level safely.
        if !(*info).fields.is_null() {
            let fields =
                std::slice::from_raw_parts((*info).fields, (*info).num_fields as usize);
            for field in fields {
                // The C reflection tables are immortal once registered.
                let field: &'static TVMFFIFieldInfo = &*(field as *const TVMFFIFieldInfo);
                if let ControlFlow::Break(b) = f(field) {
                    return Some(b);
                }
            }
        }
        let depth = (*info).type_depth;
        if depth == 0 {
            break;
        }
        info = *(*info).type_acenstors.offset((depth - 1) as isize);
    }
    None
}
