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

//! Linking for `libtvm_tirx.so`.
//!
//! The transitive `tvm-ffi-sys` build script links `libtvm_ffi.so` from whatever
//! `tvm-ffi-config --libdir` reports (`tools/tvm-ffi-config` shims it to the same
//! directory `TIRX_LIB_DIR` points at). Here we only re-emit `-ltvm_ffi` after the
//! rlibs (the toolchain's default `--as-needed` can otherwise drop it) and bake an
//! rpath so the cdylib also resolves outside a Python process that pre-loaded tvm.

use std::path::PathBuf;

fn tir_build_lib() -> String {
    if let Ok(p) = std::env::var("TIRX_LIB_DIR") {
        return p;
    }
    // crate dir = <repo>/rust/tirx-ext/rust  ->  <repo>/build/lib
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest
        .join("../../../build/lib")
        .canonicalize()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "/usr/local/lib".to_string())
}

fn main() {
    let lib_dir = tir_build_lib();
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltvm_ffi");
    println!("cargo:rustc-link-arg=-Wl,--as-needed");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
    println!("cargo:rerun-if-env-changed=TIRX_LIB_DIR");
}
