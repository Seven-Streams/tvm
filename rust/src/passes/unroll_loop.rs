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

//! `tirx.UnrollLoop` (`src/tirx/transform/unroll_loop.cc`): unroll loops marked for unrolling and
//! the small constant loops the `tirx.UnrollLoop` config selects, then restore SSA form.

use std::collections::HashSet;

use tvm_ffi::{
    dispatch, structural_map, structural_mutate, structural_visit, Any, Array, DefRegionKind, Map,
    Mutator, ObjectIdentity, ObjectRefCore, Result, String, Unchanged, UnchangedOr, WalkOrder,
};

use super::convert_ssa::convert_ssa_stmt;
use super::util::{function_pass, int_imm, prim_type, MutatorExt};
use crate::ir::{
    Evaluate, Expr, For, IntImm, IntImmObj, PrimTypeObj, PtrType, SeqStmt, Stmt, TensorLoad,
    TensorStore, Var,
};
use crate::runtime::MemoryScope;
use crate::sym::Analyzer;
use crate::tirx::transform::UnrollLoopConfig;
use crate::tirx::TensorTypeObj;
use crate::transform::{Pass, PassContext};

const AUTO_UNROLL_MAX_STEP: &str = "auto_unroll_max_step";
const UNROLL_EXPLICIT: &str = "unroll_explicit";
const FOR_DEFAULT: i32 = 0;
const FOR_UNROLLED: i32 = 3;

/// The unrolling budget a loop inherits and its annotations may override.
#[derive(Clone, Copy)]
struct UnrollPolicy {
    auto_unroll_max_step: i64,
    unroll_explicit: bool,
}

/// What the statements seen so far in the current sequence element add up to.
#[derive(Clone, Copy, Default)]
struct Counters {
    steps: i64,
    unroll_depth: i64,
    normal_loop_depth: i64,
}

impl Counters {
    /// The counters after a sequence element that counted `next` on its own.
    fn then(self, next: Counters) -> Counters {
        Counters {
            steps: self.steps + next.steps,
            unroll_depth: self.unroll_depth.max(next.unroll_depth),
            normal_loop_depth: self.normal_loop_depth.max(next.normal_loop_depth),
        }
    }
}

struct LoopUnroller {
    policy: UnrollPolicy,
    counters: Counters,
    auto_max_depth: i64,
    auto_max_extent: i64,
    unroll_local_access: bool,
    /// Loop variables that index `local` or `warp` memory; unrolling them keeps those accesses in
    /// registers.
    indexes_local: HashSet<ObjectIdentity>,
    analyzer: Analyzer,
    local: MemoryScope,
    warp: MemoryScope,
}

#[dispatch(mutate)]
impl LoopUnroller {
    /// A loop runs under its parent's policy, overridden by its own annotations.
    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Stmt> {
        let parent = self.policy;
        if let Some(step) = annotation(&op, AUTO_UNROLL_MAX_STEP)? {
            self.policy.auto_unroll_max_step = step;
        }
        if let Some(explicit) = annotation(&op, UNROLL_EXPLICIT)? {
            self.policy.unroll_explicit = explicit != 0;
        }
        let result = self.rewrite_loop(op, m);
        self.policy = parent;
        result
    }

    fn mutate_load(&mut self, op: TensorLoad) -> Result<Unchanged> {
        self.note_local_access(&op.source, &op.indices)?;
        Ok(Unchanged)
    }

    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        self.counters.steps += 1;
        self.note_local_access(&op.dest, &op.indices)?;
        m.default_mutate(self, &op)
    }

    fn mutate_evaluate(&mut self, op: Evaluate, m: &mut Mutator) -> Result<Any> {
        self.counters.steps += 1;
        m.default_mutate(self, &op)
    }

    /// Each element is counted on its own, then added to what precedes it.
    fn mutate_seq(&mut self, op: SeqStmt, m: &mut Mutator) -> Result<Any> {
        let mut seq = Vec::with_capacity(op.seq.len());
        let mut changed = false;
        for child in op.seq.iter() {
            let before = std::mem::take(&mut self.counters);
            let result: Stmt = m.mutate_as(self, &child)?;
            self.counters = before.then(self.counters);
            changed |= !result.same_as(&child);
            seq.push(result);
        }
        if !changed {
            return Ok(op.into());
        }
        Ok(SeqStmt::new_full(seq, op.loc.clone())?.into())
    }
}

impl LoopUnroller {
    fn rewrite_loop(&mut self, op: For, m: &mut Mutator) -> Result<Stmt> {
        let mut op: For = m.default_mutate_as(self, &op)?;
        if [AUTO_UNROLL_MAX_STEP, UNROLL_EXPLICIT]
            .iter()
            .any(|key| op.annotations.contains_key(&String::from(*key)))
        {
            let annotations: Map<String, Any> = op
                .annotations
                .iter()
                .filter(|(key, _)| {
                    key.as_str() != AUTO_UNROLL_MAX_STEP && key.as_str() != UNROLL_EXPLICIT
                })
                .collect();
            op = op.with_annotations(annotations);
        }
        let extent = self.constant_extent(&op)?;
        let policy = self.policy;
        let fits = extent * self.counters.steps <= policy.auto_unroll_max_step
            || extent <= self.auto_max_extent;
        let mut auto_unroll = op.kind == FOR_DEFAULT
            && extent >= 0
            && self.counters.normal_loop_depth == 0
            && self.counters.unroll_depth <= self.auto_max_depth
            && fits;
        if op.kind == FOR_UNROLLED {
            if extent < 0 {
                return Err(tvm_ffi::Error::new(
                    tvm_ffi::INTERNAL_ERROR,
                    "Cannot unroll non-constant loop",
                    "",
                ));
            }
            auto_unroll = true;
        }
        if self.unroll_local_access
            && extent > 0
            && self.indexes_local.contains(&ObjectIdentity::of(&op.loop_var))
        {
            auto_unroll = true;
        }
        if auto_unroll {
            self.counters.steps *= extent;
            self.counters.unroll_depth += 1;
        } else {
            self.counters.normal_loop_depth += 1;
        }
        if (auto_unroll && policy.unroll_explicit)
            || (0..=1).contains(&extent) && self.auto_max_extent == 1
        {
            return unroll(&op, extent);
        }
        if auto_unroll && op.kind != FOR_UNROLLED {
            return Ok(op.with_kind(FOR_UNROLLED).into());
        }
        Ok(op.into())
    }

    /// `GetExtent`: the simplified extent as an `int`, or -1.
    fn constant_extent(&self, op: &For) -> Result<i64> {
        let extent = self.analyzer.simplify(&op.extent, 2)?;
        Ok(extent
            .as_node::<IntImmObj>()
            .and_then(|imm| imm.value.to_i64())
            .filter(|value| i32::try_from(*value).is_ok())
            .unwrap_or(-1))
    }

    /// Record the variables indexing `tensor` when it lives in `local` or `warp` memory.
    fn note_local_access(&mut self, tensor: &Expr, indices: &Array<Expr>) -> Result<()> {
        if !self.unroll_local_access {
            return Ok(());
        }
        let scope = &tensor.ty.as_node::<TensorTypeObj>().expect("an accessed tensor").scope;
        if scope.same_as(&self.local) || scope.same_as(&self.warp) {
            let mut collector = VarCollector { vars: &mut self.indexes_local };
            for index in indices.iter() {
                structural_visit(&index, &mut collector)?;
            }
        }
        Ok(())
    }
}

struct VarCollector<'a> {
    vars: &'a mut HashSet<ObjectIdentity>,
}

#[dispatch(visit)]
impl VarCollector<'_> {
    fn visit_var(&mut self, var: Var) {
        self.vars.insert(ObjectIdentity::of(&var));
    }
}

/// An integer annotation, read as C++ `value.cast<IntImm>()->value`; `None` when absent or null.
fn annotation(op: &For, key: &str) -> Result<Option<i64>> {
    let Some(value) = op.annotations.get(&String::from(key))? else { return Ok(None) };
    if value.type_index() == 0 {
        return Ok(None);
    }
    // C++ `TypeTraits<IntImm>` also accepts a plain `int` or `bool` (`ConvertFallbackValue`);
    // Rust `try_into` has no such fallback.
    if let Some(value) = value.try_as::<i64>() {
        return Ok(Some(value));
    }
    if let Some(value) = value.try_as::<bool>() {
        return Ok(Some(value as i64));
    }
    let imm: IntImm = value.try_into()?;
    Ok(Some(imm.value.to_i64().unwrap_or(0)))
}

/// The loop body repeated `extent` times, the loop variable replaced by each value in turn.
fn unroll(op: &For, extent: i64) -> Result<Stmt> {
    if extent == 0 {
        return Ok(Evaluate::new(0)?.into());
    }
    let dtype =
        op.loop_var.ty.as_node::<PrimTypeObj>().expect("a loop variable is a PrimVar").dtype;
    let mut unrolled: Vec<Stmt> = Vec::new();
    for i in 0..extent {
        // C++ `op->min + IntImm(op->loop_var.ty(), i)`.
        let value = (&op.min + int_imm(dtype, i as i128)).expr()?;
        let loop_var = &op.loop_var;
        let step = structural_map(
            op.body.clone(),
            |var: Var, kind: DefRegionKind| -> UnchangedOr<Expr> {
                if kind == DefRegionKind::None && var.same_as(loop_var) {
                    UnchangedOr::changed(value.clone())
                } else {
                    UnchangedOr::unchanged()
                }
            },
            WalkOrder::PreOrder,
        )?;
        unrolled.push(step.try_into()?);
    }
    Ok(SeqStmt::new(unrolled)?.into())
}

/// `runtime::MemoryScope::Get(name)`. Rust has no lookup of a registered enum value by name, so
/// this lets a native `PtrType` constructor convert the name.
fn memory_scope(name: &str) -> Result<MemoryScope> {
    let ptr: PtrType = tvm_ffi::cached_global_func!("ir.PtrType")
        .call_tuple((prim_type(super::util::int32_dtype())?, String::from(name)))?
        .try_into()?;
    Ok(ptr.scope.clone())
}

/// `ctx->GetConfigOrDefault<UnrollLoopConfig>("tirx.UnrollLoop")`.
fn config(context: &PassContext) -> Result<(i64, i64, i64, bool, bool)> {
    match context.config.get(&String::from("tirx.UnrollLoop"))? {
        Some(value) => {
            let cfg: UnrollLoopConfig = value.try_into()?;
            Ok((
                cfg.auto_max_step as i64,
                cfg.auto_max_depth as i64,
                cfg.auto_max_extent as i64,
                cfg.explicit_unroll != 0,
                cfg.unroll_local_access != 0,
            ))
        }
        // The reflection defaults of UnrollLoopConfig.
        None => Ok((0, 8, 0, true, false)),
    }
}

pub fn unroll_loop() -> Result<Pass> {
    function_pass("tirx.UnrollLoop", 0, |func, _module, context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let (auto_max_step, auto_max_depth, auto_max_extent, explicit_unroll, unroll_local_access) =
            config(context)?;
        let mut unroller = LoopUnroller {
            policy: UnrollPolicy {
                auto_unroll_max_step: auto_max_step,
                unroll_explicit: explicit_unroll,
            },
            counters: Counters::default(),
            auto_max_depth,
            auto_max_extent,
            unroll_local_access,
            indexes_local: HashSet::new(),
            analyzer: Analyzer::new()?,
            local: memory_scope("local")?,
            warp: memory_scope("warp")?,
        };
        let result: Stmt = structural_mutate(body.clone(), &mut unroller)?.try_into()?;
        let body = if result.same_as(&body) {
            body
        } else {
            convert_ssa_stmt(SeqStmt::from_stmt(result)?)?
        };
        Ok(Some(func.with_body(Some(body))))
    })
}
