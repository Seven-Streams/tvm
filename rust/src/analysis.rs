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

//! Queries over the reflection-driven structural graph.
//!
//! Definition regions here are structural equality/hash metadata, not TIRx
//! lexical use/definition semantics.  A variable may occur both inside and
//! outside such regions.  Roots must expose an acyclic structural graph because
//! [`structural_walk`] does not detect cycles.

use std::collections::HashSet;

use tvm_ffi::{structural_walk, AnyView, DefRegionKind, Result, WalkOrder, WalkResult};

use crate::generated::tirx::Var;

fn identity(value: &Var) -> usize {
    &**value as *const _ as usize
}

/// Collect each reachable [`Var`] once by object identity.
///
/// Results follow first pre-order occurrence.  Fields ignored by
/// [`structural_walk`] are not included.
pub fn collect_structural_vars<R>(root: &R) -> Result<Vec<Var>>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let mut seen = HashSet::new();
    let mut vars = Vec::new();
    structural_walk(
        root,
        |var: Var| {
            if seen.insert(identity(&var)) {
                vars.push(var);
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )?;
    Ok(vars)
}

fn contains_var_where<R>(
    root: &R,
    target: &Var,
    accept: impl Fn(DefRegionKind) -> bool,
) -> Result<bool>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let target = identity(target);
    let mut found = false;
    structural_walk(
        root,
        |var: Var, kind: DefRegionKind| {
            if accept(kind) && identity(&var) == target {
                found = true;
                WalkResult::Interrupt
            } else {
                WalkResult::Advance
            }
        },
        WalkOrder::PreOrder,
    )?;
    Ok(found)
}

/// Return whether `target` occurs outside a structural definition region.
pub fn contains_var_outside_def_region<R>(root: &R, target: &Var) -> Result<bool>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    contains_var_where(root, target, |kind| kind == DefRegionKind::None)
}

/// Return whether `target` occurs inside a structural definition region.
pub fn contains_var_in_def_region<R>(root: &R, target: &Var) -> Result<bool>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    contains_var_where(root, target, |kind| kind != DefRegionKind::None)
}
