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

//! Smoke tests for the stubgen-generated `target` bindings against `libtvm_compiler`.

use tvm::target::{Target, TargetKind, TargetKindObj, TargetObj};
use tvm::tvm_ffi::{ObjectRefCast, String as FfiString};

mod common;
use common::{assert_type_contracts, global, load_tvm_compiler};

fn target(name: &str) -> Target {
    global("target.Target").call_tuple((FfiString::from(name),)).unwrap().try_into().unwrap()
}

#[test]
fn generated_types_match_the_registry() {
    load_tvm_compiler();
    type Object = tvm::tvm_ffi::Object;
    // Both hold members reflection does not describe, so both bindings are opaque.
    assert_type_contracts!(false; TargetObj: Object, TargetKindObj: Object);
}

#[test]
fn native_targets_are_read_through_accessors() {
    load_tvm_compiler();
    let llvm = target("llvm");
    let kind: TargetKind = llvm.kind().unwrap();
    assert_eq!(kind.name().unwrap().as_str(), "llvm");
    assert!(kind.default_keys().unwrap().iter().any(|key| key.as_str() == "cpu"));
    assert!(llvm.keys().unwrap().iter().any(|key| key.as_str() == "cpu"));
    assert!(llvm.host().unwrap().is_none());

    // The kind's reflected device type is what the native query reports.
    let cuda = target("cuda");
    assert_eq!(cuda.kind().unwrap().name().unwrap().as_str(), "cuda");
    let device_type: i64 = global("target.TargetGetDeviceType")
        .call_tuple((cuda.clone(),))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(cuda.kind().unwrap().default_device_type().unwrap(), device_type);
    assert_ne!(device_type, kind.default_device_type().unwrap());
    cuda.attrs().unwrap();

    // `host` is declared as an `ObjectRef`; it holds a `Target`.
    let with_host: Target = global("target.WithHost")
        .call_tuple((cuda.clone(), llvm.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    let host: Target = with_host.host().unwrap().unwrap().try_cast().unwrap();
    assert_eq!(host.kind().unwrap().name().unwrap().as_str(), "llvm");
    assert!(with_host.host().unwrap().unwrap().try_cast::<TargetKind>().is_err());
}
