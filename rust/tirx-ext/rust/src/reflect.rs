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

/// Visit every reflected field of `type_index` and its ancestors in the same
/// parent-to-child order as C++ `ForEachFieldInfoWithEarlyStop`. The callback's
/// `ControlFlow::Break` value short-circuits and is returned.
///
/// # Safety
/// `type_index` must be a registered type index.
pub(crate) unsafe fn for_each_field<B>(
    type_index: i32,
    mut f: impl FnMut(&'static TVMFFIFieldInfo) -> ControlFlow<B>,
) -> Option<B> {
    let info = TVMFFIGetTypeInfo(type_index);
    if info.is_null() {
        return None;
    }

    // Ancestor slot 0 is the root Object.  C++ starts at slot 1, walks toward
    // the immediate parent, then visits the concrete type's own fields.
    for depth in 1..(*info).type_depth {
        let ancestor = *(*info).type_acenstors.offset(depth as isize);
        if let Some(b) = visit_level(ancestor, &mut f) {
            return Some(b);
        }
    }
    visit_level(info, &mut f)
}

unsafe fn visit_level<B>(
    info: *const tvm_ffi::tvm_ffi_sys::TVMFFITypeInfo,
    f: &mut impl FnMut(&'static TVMFFIFieldInfo) -> ControlFlow<B>,
) -> Option<B> {
    if info.is_null() || (*info).fields.is_null() {
        return None;
    }
    let fields = std::slice::from_raw_parts((*info).fields, (*info).num_fields as usize);
    for field in fields {
        // The C reflection tables are immortal once registered.
        let field: &'static TVMFFIFieldInfo = &*(field as *const TVMFFIFieldInfo);
        if let ControlFlow::Break(b) = f(field) {
            return Some(b);
        }
    }
    None
}
