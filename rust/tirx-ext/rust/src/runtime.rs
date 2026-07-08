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
use tvm_ffi::function::Function;

/// Resolve a C++ type key (e.g. `"tirx.For"`) to its runtime `type_index`.
pub(crate) fn lookup_type_index(type_key: &str) -> i32 {
    use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeKeyToIndex};
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

/// Call a registered global function with pre-built argument views.
pub(crate) fn call_packed_global(name: &str, args: &[AnyView]) -> Any {
    let f = Function::get_global(name)
        .unwrap_or_else(|e| panic!("tirx_ext: global `{}` not found: {:?}", name, e));
    f.call_packed(args)
        .unwrap_or_else(|e| panic!("tirx_ext: call to `{}` failed: {:?}", name, e))
}
