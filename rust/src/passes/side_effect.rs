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

//! `SideEffect` (`src/ir/side_effect.cc`): the strongest `CallEffectKind` an expression evaluates.

use std::cell::OnceCell;

use tvm_ffi::{dispatch, structural_walk, ObjectRefCast, Result, WalkOrder, WalkResult};

use crate::ir::{Call, Op, OpAttrMap, TensorLoad, Type};

// `CallEffectKind` (include/tvm/ir/expr.h).
pub const PURE: i64 = 1;
pub const READ_STATE: i64 = 2;
pub const UPDATE_STATE: i64 = 3;
pub const OPAQUE: i64 = UPDATE_STATE;

thread_local! {
    /// `Op::GetAttrMap(op_attr::kCallEffectKind)`, kept for the process like the C++ `static`.
    static CALL_EFFECT_KIND: OnceCell<Option<OpAttrMap<i64>>> = const { OnceCell::new() };
}

/// The registered `TCallEffectKind` of `op`, if any.
fn call_effect_kind(op: &Op) -> Result<Option<i64>> {
    CALL_EFFECT_KIND.with(|cell| {
        if cell.get().is_none() {
            let _ = cell.set(Op::attr_map("TCallEffectKind")?);
        }
        match cell.get() {
            Some(Some(column)) => column.get(op),
            _ => Ok(None),
        }
    })
}

/// The strongest effect seen so far; the walk stops at the first state update.
struct EffectWalker {
    kind: i64,
}

#[dispatch(walk)]
impl EffectWalker {
    fn walk_call(&mut self, call: Call) -> Result<WalkResult> {
        let effect = match call.op.clone().try_cast::<Op>() {
            Ok(op) => call_effect_kind(&op)?.unwrap_or(OPAQUE),
            Err(_) => OPAQUE,
        };
        self.kind = self.kind.max(effect.min(UPDATE_STATE));
        Ok(if self.kind == UPDATE_STATE { WalkResult::Interrupt } else { WalkResult::Advance })
    }

    fn walk_load(&mut self, _load: TensorLoad) -> WalkResult {
        self.kind = self.kind.max(READ_STATE);
        WalkResult::Advance
    }

    /// Effects describe evaluated expressions, not expressions embedded in types.
    fn walk_type(&mut self, _ty: Type) -> WalkResult {
        WalkResult::Skip
    }
}

pub fn side_effect<T>(expr: &T) -> Result<i64>
where
    for<'x> tvm_ffi::AnyView<'x>: From<&'x T>,
{
    let mut walker = EffectWalker { kind: PURE };
    structural_walk(expr, &mut walker, WalkOrder::PreOrder)?;
    Ok(walker.kind)
}
