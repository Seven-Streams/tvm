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

//! Layout self-check: every Rust field offset must equal the C++ side's
//! reflected field offset — the ground-truth proof that the hand-mirrored
//! `#[repr(C)]` structs in node.rs are byte-compatible with the C++ nodes.
//! Exposed as the `tirx_ext._check_layouts` global and run on every Python
//! import, so a tirx header change that shifts a field fails loudly at load
//! time instead of reading the wrong pointer slot at count time.

use std::mem::offset_of;
use std::ops::ControlFlow;

use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeKeyToIndex};

use crate::node::{
    AddNode, ExprNode, ForNode, IfThenElseNode, IntImmNode, VarNode, WhileNode,
};
use crate::reflect::for_each_field;

/// The reflected absolute byte offset of `field` in the C++ type `type_key`,
/// searching the type and then its ancestors (single-inheritance chain).
fn reflect_offset(type_key: &str, field: &str) -> Result<i64, String> {
    unsafe {
        let key = TVMFFIByteArray::from_str(type_key);
        let mut idx: i32 = 0;
        if TVMFFITypeKeyToIndex(&key, &mut idx) != 0 {
            return Err(format!("type key `{type_key}` is not registered"));
        }
        for_each_field(idx, |f| {
            if f.name.as_str() == field {
                ControlFlow::Break(f.offset)
            } else {
                ControlFlow::Continue(())
            }
        })
        .ok_or_else(|| format!("field `{field}` not found in `{type_key}` or its ancestors"))
    }
}

/// Verify every field of every node this crate mirrors. Returns the full list
/// of mismatches (empty = layouts proven identical).
pub(crate) fn check_layouts() -> Result<(), String> {
    // (rust offset, C++ type key, C++ reflected field name, label)
    macro_rules! entry {
        ($node:ty, $key:literal, $field:ident) => {
            entry!($node, $key, $field, stringify!($field))
        };
        ($node:ty, $key:literal, $field:ident, $cxx:expr) => {
            (
                offset_of!($node, $field) as i64,
                $key,
                $cxx,
                concat!(stringify!($node), ".", stringify!($field)),
            )
        };
    }
    let table = [
        // Expr base: span @24, ty @32 (the reflected `ir.Expr`).
        entry!(ExprNode, "ir.Expr", span),
        entry!(ExprNode, "ir.Expr", ty),
        entry!(VarNode, "tirx.Var", name_hint, "name"),
        entry!(IntImmNode, "ir.IntImm", value),
        entry!(AddNode, "tirx.Add", a),
        entry!(AddNode, "tirx.Add", b),
        entry!(ForNode, "tirx.For", loop_var),
        entry!(ForNode, "tirx.For", min),
        entry!(ForNode, "tirx.For", extent),
        entry!(ForNode, "tirx.For", kind),
        entry!(ForNode, "tirx.For", body),
        entry!(ForNode, "tirx.For", thread_binding),
        entry!(ForNode, "tirx.For", annotations),
        entry!(ForNode, "tirx.For", step),
        entry!(IfThenElseNode, "tirx.IfThenElse", condition),
        entry!(IfThenElseNode, "tirx.IfThenElse", then_case),
        entry!(IfThenElseNode, "tirx.IfThenElse", else_case),
        entry!(WhileNode, "tirx.While", condition),
        entry!(WhileNode, "tirx.While", body),
    ];

    let mut mismatches = Vec::new();
    for (rust, key, cxx, label) in table {
        match reflect_offset(key, cxx) {
            Ok(reflected) if reflected == rust => {}
            Ok(reflected) => mismatches.push(format!(
                "{label} @ {rust} (Rust) vs {key}.{cxx} @ {reflected} (C++)"
            )),
            Err(e) => mismatches.push(format!("{label}: {e}")),
        }
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "tirx_ext: node layout mismatch against the loaded libtvm_compiler.so — \
             rebuild the extension against the current tir headers: {}",
            mismatches.join("; ")
        ))
    }
}
