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

//! `tirx.RemoveNoOp` (`src/tirx/transform/remove_no_op.cc`): drop statements without effect,
//! empty loops and branches, and stores of the value already in place.

use std::collections::HashMap;

use tvm_ffi::{
    dispatch, Any, Array, Map, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result,
    String,
};

use super::analyzer::{binary, is_const_int, not, AnalyzerBase, AnalyzerScope, AnalyzerState};
use super::side_effect::{side_effect, READ_STATE};
use super::util::{
    function_pass, int32, int_imm, make_const, prim_dtype, seq_stmt_from, unknown_loc, with_body,
    MutatorExt,
};
use crate::ir::{
    CallObj, Evaluate, EvaluateObj, Expr, For, If, IntImmObj, IntSet, Op, PrimTypeObj, SeqStmt,
    SeqStmtObj, Stmt, TensorLoadObj, TensorStore, Var,
};
use crate::sym::{neg_inf, pos_inf, Analyzer};
use crate::tirx::transform::RemoveNoOpConfig;
use crate::tirx::TensorTypeObj;
use crate::transform::Pass;

struct NoOpRemover {
    base: AnalyzerBase,
    /// The range of each enclosing loop's variable (`var_range_map_`).
    loop_ranges: HashMap<ObjectIdentity, (Var, IntSet)>,
    /// Profiler calls, which count as no-ops when `ignore_profiler_call` is set.
    profiler_ops: Vec<Op>,
}

impl AnalyzerState for NoOpRemover {
    fn base(&mut self) -> &mut AnalyzerBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = AnalyzerScope)]
impl NoOpRemover {
    /// A branch whose arms do nothing; an empty arm is dropped, negating the condition if needed.
    fn mutate_if(&mut self, op: If, m: &mut Mutator) -> Result<Stmt> {
        let stmt: Stmt = m.default_mutate_as(self, &Stmt::from(op))?;
        let Ok(op) = stmt.clone().try_cast::<If>() else { return Ok(stmt) };
        let then_is_no_op = is_no_op(&op.then_case);
        let else_is_no_op = op.else_case.as_ref().is_none_or(is_no_op);
        Ok(match (then_is_no_op, else_is_no_op, &op.else_case) {
            (true, true, _) => evaluate(&op.condition)?,
            (false, true, Some(_)) => {
                If::new(unknown_loc()?, op.condition.clone(), op.then_case.clone(), None).into()
            }
            (true, false, Some(else_case)) => {
                If::new(unknown_loc()?, logical_not(&op.condition)?, else_case.clone(), None).into()
            }
            _ => stmt,
        })
    }

    /// A loop that provably runs zero times, or whose body does nothing.
    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Stmt> {
        let max = self.extent_range(&op.extent)?.max()?;
        if !max.same_as(&pos_inf()?) && !max.same_as(&neg_inf()?) {
            let empty = binary("prim._OpLE", &max, &make_const(prim_dtype(&max)?, 0)?)?;
            if self.base.analyzer.can_prove(&empty, 0)? {
                return evaluate_zero();
            }
        }
        let key = ObjectIdentity::of(&op.loop_var);
        self.loop_ranges.insert(key.clone(), (op.loop_var.clone(), loop_range(&op)?));
        let stmt = m.default_mutate_as(self, &Stmt::from(op));
        self.loop_ranges.remove(&key);
        let stmt: Stmt = stmt?;
        let result: For = stmt.clone().try_cast()?;
        if is_const_int(&result.extent, 0) {
            return evaluate_zero();
        }
        if is_no_op(&result.body) {
            return evaluate_effects(&[result.min.clone(), result.extent.clone()]);
        }
        Ok(stmt)
    }

    /// An evaluation whose value has no effect.
    fn mutate_evaluate(&mut self, op: Evaluate) -> Result<Stmt> {
        if self.has_side_effect(&op.value)? {
            return Ok(op.into());
        }
        evaluate_zero()
    }

    /// A store of the value already in place keeps only the effects of its operands.
    fn mutate_store(&mut self, store: TensorStore, m: &mut Mutator) -> Result<Any> {
        if !self.stores_existing_value(&store)? {
            return Ok(store.into());
        }
        let mut effects = vec![evaluate(&store.value)?];
        for index in store.indices.iter() {
            effects.push(evaluate(&index)?);
        }
        m.mutate(self, &seq_stmt(effects)?)
    }
}

impl NoOpRemover {
    /// `sym::EvalSet(extent, var_range_map_)`.
    fn extent_range(&self, extent: &Expr) -> Result<IntSet> {
        let dom_map: Map<Var, IntSet> = self.loop_ranges.values().cloned().collect();
        Analyzer::new()?.int_set(extent, &dom_map)
    }

    fn has_side_effect(&self, value: &Expr) -> Result<bool> {
        if value.ty.as_node::<PrimTypeObj>().is_none() {
            return Ok(value.as_node::<CallObj>().is_some());
        }
        if let Some(call) = value.as_node::<CallObj>() {
            if self.profiler_ops.iter().any(|op| call.op.same_as(op)) {
                return Ok(false);
            }
        }
        Ok(side_effect(value)? > READ_STATE)
    }

    fn stores_existing_value(&self, store: &TensorStore) -> Result<bool> {
        let dest: Var = store.dest.clone().try_cast()?;
        let load: Expr = tvm_ffi::cached_global_func!("ir.TensorLoad")
            .call_tuple((dest.clone(), store.indices.clone(), unknown_loc()?))?
            .try_into()?;
        let diff = binary("prim._OpSub", &store.value, &load)?;
        let unchanged = binary("prim._OpEQ", &diff, &make_const(prim_dtype(&diff)?, 0)?)?;
        if is_const_int(&self.base.analyzer.simplify(&unchanged, 2)?, 1) {
            return Ok(true);
        }
        let Some(load) = store.value.as_node::<TensorLoadObj>() else { return Ok(false) };
        if !load.source.same_as(&dest) {
            return Ok(false);
        }
        let tensor = dest.ty.as_node::<TensorTypeObj>().expect("a store's destination is a tensor");
        // The same tensor on both sides: its layout trivially matches; compare the indices.
        Ok(self.base.analyzer.can_prove_equal(&tensor.elem_offset, &tensor.elem_offset)?
            && self.all_equal(&tensor.shape, &tensor.shape)?
            && self.all_equal(&tensor.strides, &tensor.strides)?
            && self.all_equal(&load.indices, &store.indices)?)
    }

    fn all_equal(&self, a: &Array<Expr>, b: &Array<Expr>) -> Result<bool> {
        if a.len() != b.len() {
            return Ok(false);
        }
        for (x, y) in a.iter().zip(b.iter()) {
            if !self.base.analyzer.can_prove_equal(&x, &y)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// `IntSet::FromMinExtent(min, extent)` for a loop.
fn loop_range(op: &For) -> Result<IntSet> {
    if is_const_int(&op.extent, 1) {
        return IntSet::single_point(&op.min);
    }
    let one = make_const(prim_dtype(&op.extent)?, 1)?;
    let max = binary("prim._OpSub", &binary("prim._OpAdd", &op.extent, &op.min)?, &one)?;
    IntSet::interval(&op.min, &max)
}

/// `tirx::IsNoOp`: an integer `Evaluate` or an empty sequence.
fn is_no_op<S: ObjectRefCore>(stmt: &S) -> bool {
    if let Some(eval) = stmt.as_node::<EvaluateObj>() {
        return eval.value.as_node::<IntImmObj>().is_some();
    }
    matches!(stmt.as_node::<SeqStmtObj>(), Some(seq) if seq.seq.is_empty())
}

fn evaluate_zero() -> Result<Stmt> {
    Ok(Evaluate::new(unknown_loc()?, int32(0)?.into()).into())
}

/// `MakeEvaluate(value)`: the evaluation if it has an effect, else a no-op.
fn evaluate(value: &Expr) -> Result<Stmt> {
    if side_effect(value)? > READ_STATE {
        return Ok(Evaluate::new(unknown_loc()?, value.clone()).into());
    }
    evaluate_zero()
}

/// `MakeEvaluate(values)`: the evaluations that have an effect, in sequence.
fn evaluate_effects(values: &[Expr]) -> Result<Stmt> {
    let mut effects = Vec::new();
    for value in values {
        if side_effect(value)? > READ_STATE {
            effects.push(Evaluate::new(unknown_loc()?, value.clone()).into());
        }
    }
    Ok(seq_stmt(effects)?.into())
}

/// The native `SeqStmt(stmts)` constructor, which flattens and drops no-ops.
fn seq_stmt(stmts: Vec<Stmt>) -> Result<SeqStmt> {
    tvm_ffi::cached_global_func!("ir.SeqStmt")
        .call_tuple((Array::new(stmts), Option::<crate::ir::Location>::None))?
        .try_into()
}

/// C++ `!a` (`logical_not`), which has no global: folds a constant, else the raw `Not`.
fn logical_not(a: &Expr) -> Result<Expr> {
    match a.as_node::<IntImmObj>() {
        Some(imm) => {
            Ok(int_imm(prim_dtype(a)?, (imm.value == tvm_ffi::BigInt::from(0i64)) as i128)?.into())
        }
        None => not(a),
    }
}

pub fn remove_no_op() -> Result<Pass> {
    function_pass("tirx.RemoveNoOp", 0, |func, _module, context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        // `GetConfigOrDefault<RemoveNoOpConfig>`: the reflection defaults are 0 and false.
        let (max_steps, ignore_profiler_call) =
            match context.config.get(&String::from("tirx.RemoveNoOp"))? {
                Some(value) => {
                    let config: RemoveNoOpConfig = value.try_into()?;
                    (config.max_simplification_steps, config.ignore_profiler_call)
                }
                None => (0, false),
            };
        let analyzer = Analyzer::new()?;
        analyzer.set_maximum_rewrite_steps(max_steps)?;
        let profiler_ops = if ignore_profiler_call {
            [
                "tirx.cuda.timer_init",
                "tirx.cuda.timer_start",
                "tirx.cuda.timer_end",
                "tirx.cuda.timer_finalize",
            ]
            .into_iter()
            .map(Op::get)
            .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let mut remover = NoOpRemover {
            base: AnalyzerBase::new(analyzer)?,
            loop_ranges: HashMap::new(),
            profiler_ops,
        };
        let body: Stmt = tvm_ffi::structural_mutate(body, &mut remover)?.try_into()?;
        Ok(Some(with_body(&func, Some(seq_stmt_from(body)?))))
    })
}
