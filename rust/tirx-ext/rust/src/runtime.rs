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

//! Thin FFI call layer.
//!
//! Unlike a stand-alone Rust host, this cdylib never bootstraps the process:
//! the Python host `import tvm`s first, which loads `libtvm_compiler.so` and
//! registers every `tirx.*` / `ir.*` type and global this crate looks up. The
//! panics below therefore indicate a load-order bug in the host, and they are
//! converted to `ffi.Error`s by the `catch_unwind` in the exported entry points.

use tvm_ffi::any::{Any, AnyView};
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeAttrColumn,
    TVMFFITypeKeyToIndex,
};

extern "C" {
    // Present in libtvm_ffi but not yet declared by tvm-ffi-sys.
    fn TVMFFIGetTypeAttrColumn(attr_name: *const TVMFFIByteArray) -> *const TVMFFITypeAttrColumn;
}

/// Layout prefix shared by C++ `ArrayObj` and `ListObj`.
#[repr(C)]
pub(crate) struct SeqPrefix {
    _header: TVMFFIObject,
    pub(crate) data: *const TVMFFIAny,
    pub(crate) size: i64,
}

const _: () = {
    assert!(std::mem::offset_of!(SeqPrefix, data) == 24);
    assert!(std::mem::offset_of!(SeqPrefix, size) == 32);
};

/// Resolve a C++ type key (e.g. `"tirx.For"`) to its runtime `type_index`.
pub(crate) fn lookup_type_index(type_key: &str) -> i32 {
    unsafe {
        let arg = TVMFFIByteArray::from_str(type_key);
        let mut idx: i32 = 0;
        let ret = TVMFFITypeKeyToIndex(&arg, &mut idx);
        if ret != 0 {
            panic!(
                "tirx_ext: type key `{}` is not registered — \
                 `import tvm` must run before loading libtvm_tirx.so",
                type_key
            );
        }
        idx
    }
}

pub(crate) fn type_key_of(type_index: i32) -> String {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            format!("<type_index {type_index}>")
        } else {
            (*info).type_key.as_str().to_string()
        }
    }
}

pub(crate) fn type_attr_column(attr_name: &str) -> *const TVMFFITypeAttrColumn {
    unsafe {
        let attr_name = TVMFFIByteArray::from_str(attr_name);
        TVMFFIGetTypeAttrColumn(&attr_name)
    }
}

pub(crate) fn raw_of(view: AnyView) -> TVMFFIAny {
    unsafe { std::ptr::read(&view as *const AnyView as *const TVMFFIAny) }
}

pub(crate) fn raw_of_owned(any: &mut Any) -> TVMFFIAny {
    unsafe { *Any::as_data_ptr(any) }
}

pub(crate) unsafe fn view_of(raw: &TVMFFIAny) -> AnyView<'_> {
    std::ptr::read(raw as *const TVMFFIAny as *const AnyView)
}
