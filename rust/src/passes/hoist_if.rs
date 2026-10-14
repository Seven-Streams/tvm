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

//! `tirx.HoistIf` (`src/tirx/transform/hoist_if.cc`): wrap a loop in the loop-invariant `if`
//! conditions found inside it (`InsertHoistedIf`), then simplify (`StmtSimplify`, `RemoveNoOp`).

use std::collections::{HashMap, HashSet};

use tvm_ffi::{
    dispatch, structural_mutate, structural_walk, Any, Mutator, ObjectIdentity, ObjectRefCore,
    Result, Unchanged, WalkOrder, WalkResult,
};

use super::convert_ssa::convert_ssa_stmt;
use super::side_effect::{side_effect, READ_STATE};
use super::util::{function_pass, seq_stmt_from, sequential, unknown_loc};
use crate::ir::{
    Call, Expr, For, If, IntImmObj, Op, PrimTypeObj, RegionStmt, SeqStmt, Stmt, TensorLoad, Type,
    Var,
};
use crate::prim::{Cast, Div, FloorDiv, FloorMod, LShift, Mod, RShift};
use crate::transform::Pass;

/// A loop being rewritten: the conditions to hoist above it, outermost first.
struct LoopState {
    else_depth_at_entry: i32,
    conditions: Vec<(Expr, bool)>,
}

/// The loops visible from the current statement. A statement that is not a loop, a branch, or a
/// sequence starts a fresh scope: conditions inside it are not hoisted past it.
#[derive(Default)]
struct Scope {
    /// Indices into `IfHoister::loops`, outermost first.
    loops: Vec<usize>,
    loop_depths: HashMap<ObjectIdentity, usize>,
}

struct IfHoister {
    scopes: Vec<Scope>,
    loops: Vec<LoopState>,
    else_depth: i32,
    active_loops: i32,
    split_in_nest: bool,
    launch_thread: Op,
    likely: Op,
}

#[dispatch(mutate)]
impl IfHoister {
    /// Expressions are never rewritten.
    fn mutate_expr(&mut self, _expr: Expr) -> Unchanged {
        Unchanged
    }

    /// A loop collects the invariant conditions found in its body and is wrapped in them.
    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        let effectful = |expr: &Expr| -> Result<bool> { Ok(side_effect(expr)? > READ_STATE) };
        let step_effectful = op.step.as_ref().map_or(Ok(false), effectful)?;
        if op.kind != 0 || effectful(&op.min)? || effectful(&op.extent)? || step_effectful {
            return self.scoped(|this| m.default_mutate(this, &op));
        }
        if self.active_loops == 0 {
            self.split_in_nest = false;
        }
        self.active_loops += 1;
        let index = self.loops.len();
        self.loops.push(LoopState { else_depth_at_entry: self.else_depth, conditions: vec![] });
        let var = ObjectIdentity::of(&op.loop_var);
        let scope = self.scope();
        scope.loop_depths.insert(var.clone(), scope.loops.len());
        scope.loops.push(index);
        let result = m.default_mutate(self, &op);
        let scope = self.scope();
        scope.loops.pop();
        scope.loop_depths.remove(&var);
        self.active_loops -= 1;
        let mut stmt: Stmt = result?.try_into()?;
        for (condition, preserve_else) in
            std::mem::take(&mut self.loops[index].conditions).into_iter().rev()
        {
            let else_case = if preserve_else { Some(seq_stmt_from(stmt.clone())?) } else { None };
            stmt = If::new(unknown_loc()?, condition, seq_stmt_from(stmt)?, else_case).into();
        }
        Ok(stmt.into())
    }

    /// A condition invariant in an enclosing loop is recorded on the outermost such loop.
    fn mutate_if(&mut self, op: If, m: &mut Mutator) -> Result<Any> {
        let in_loop = !self.scope().loops.is_empty();
        if in_loop && side_effect(&op.condition)? > READ_STATE {
            return self.scoped(|this| m.default_mutate(this, &op));
        }
        let has_else = op.else_case.is_some();
        if in_loop {
            let destination = self.lift_destination(&op.condition)?;
            if let Some(&loop_index) = self.scope().loops.get(destination) {
                let preserve_else =
                    has_else || self.else_depth > self.loops[loop_index].else_depth_at_entry;
                if !preserve_else || !self.split_in_nest {
                    self.loops[loop_index].conditions.push((op.condition.clone(), preserve_else));
                    self.split_in_nest |= preserve_else;
                }
            }
        }
        self.else_depth += has_else as i32;
        let result = m.default_mutate(self, &op);
        self.else_depth -= has_else as i32;
        result
    }

    fn mutate_seq(&mut self, op: SeqStmt, m: &mut Mutator) -> Result<Any> {
        if op.seq.len() == 1 {
            return m.default_mutate(self, &op);
        }
        self.scoped(|this| m.default_mutate(this, &op))
    }

    /// A launched thread is not entered.
    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        if op.op.same_as(&self.launch_thread) {
            return Ok(op.into());
        }
        self.scoped(|this| m.default_mutate(this, &op))
    }

    fn mutate_stmt(&mut self, op: Stmt, m: &mut Mutator) -> Result<Any> {
        self.scoped(|this| m.default_mutate(this, &op))
    }
}

impl IfHoister {
    fn scope(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("the bottom scope is never left")
    }

    fn scoped<T>(&mut self, body: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.scopes.push(Scope::default());
        let result = body(self);
        self.scopes.pop();
        result
    }

    /// `FindLiftDestination`: the outermost loop `condition` can be hoisted above, or the
    /// number of loops when it cannot be hoisted.
    fn lift_destination(&self, condition: &Expr) -> Result<usize> {
        let scope = self.scopes.last().expect("the bottom scope is never left");
        let mut walker = LiftWalker {
            loop_depths: &scope.loop_depths,
            loops: scope.loops.len(),
            destination: 0,
            visited: HashSet::new(),
            likely: &self.likely,
        };
        let blocked = structural_walk(condition, &mut walker, WalkOrder::PreOrder)?;
        Ok(if blocked.is_some() { scope.loops.len() } else { walker.destination })
    }
}

/// Walks a condition for the loops it depends on, interrupting at anything that cannot be
/// evaluated ahead of the loop.
struct LiftWalker<'a> {
    loop_depths: &'a HashMap<ObjectIdentity, usize>,
    loops: usize,
    destination: usize,
    visited: HashSet<ObjectIdentity>,
    likely: &'a Op,
}

#[dispatch(walk)]
impl LiftWalker<'_> {
    fn walk_var(&mut self, var: Var) -> WalkResult {
        if let Some(depth) = self.loop_depths.get(&ObjectIdentity::of(&var)) {
            self.destination = self.destination.max(depth + 1);
        }
        if self.destination == self.loops {
            WalkResult::Interrupt
        } else {
            WalkResult::Skip
        }
    }

    fn walk_type(&mut self, _ty: Type) -> WalkResult {
        WalkResult::Skip
    }

    fn walk_load(&mut self, _op: TensorLoad) -> WalkResult {
        WalkResult::Interrupt
    }

    fn walk_lshift(&mut self, _op: LShift) -> WalkResult {
        WalkResult::Interrupt
    }

    fn walk_rshift(&mut self, _op: RShift) -> WalkResult {
        WalkResult::Interrupt
    }

    fn walk_div(&mut self, op: Div) -> WalkResult {
        let divisor = op.b.clone();
        self.division(op.into(), &divisor)
    }

    fn walk_mod(&mut self, op: Mod) -> WalkResult {
        let divisor = op.b.clone();
        self.division(op.into(), &divisor)
    }

    fn walk_floordiv(&mut self, op: FloorDiv) -> WalkResult {
        let divisor = op.b.clone();
        self.division(op.into(), &divisor)
    }

    fn walk_floormod(&mut self, op: FloorMod) -> WalkResult {
        let divisor = op.b.clone();
        self.division(op.into(), &divisor)
    }

    /// A float-to-integer cast cannot be hoisted.
    fn walk_cast(&mut self, op: Cast) -> WalkResult {
        if is_integer(&op.ty) && !is_integer(&op.value.ty) {
            return WalkResult::Interrupt;
        }
        self.advance(op.into())
    }

    /// Only `likely` hints can be hoisted among calls.
    fn walk_call(&mut self, op: Call) -> WalkResult {
        if !op.op.same_as(self.likely) {
            return WalkResult::Interrupt;
        }
        self.advance(op.into())
    }

    fn walk_expr(&mut self, expr: Expr) -> WalkResult {
        self.advance(expr)
    }
}

impl LiftWalker<'_> {
    /// Each shared subexpression is walked once.
    fn advance(&mut self, expr: Expr) -> WalkResult {
        if self.visited.insert(ObjectIdentity::of(&expr)) {
            WalkResult::Advance
        } else {
            WalkResult::Skip
        }
    }

    /// Division by anything but a positive constant may trap or depend on the loop.
    fn division(&mut self, expr: Expr, divisor: &Expr) -> WalkResult {
        match divisor.as_node::<IntImmObj>() {
            Some(value) if value.value > tvm_ffi::BigInt::from(0i64) => self.advance(expr),
            _ => WalkResult::Interrupt,
        }
    }
}

/// `MatchesCode(kDLInt, kDLUInt)`.
fn is_integer(ty: &Type) -> bool {
    ty.as_node::<PrimTypeObj>().is_some_and(|ty| ty.dtype.code <= 1)
}

pub fn insert_hoisted_if() -> Result<Pass> {
    let launch_thread = Op::get("tirx.launch_thread")?;
    let likely = Op::get("prim.likely")?;
    function_pass("tirx.InsertHoistedIf", 0, move |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let mut hoister = IfHoister {
            scopes: vec![Scope::default()],
            loops: vec![],
            else_depth: 0,
            active_loops: 0,
            split_in_nest: false,
            launch_thread: launch_thread.clone(),
            likely: likely.clone(),
        };
        let body: Stmt = structural_mutate(body, &mut hoister)?.try_into()?;
        let body = convert_ssa_stmt(seq_stmt_from(body)?)?;
        Ok(Some(func.with_body(Some(body))))
    })
}

pub fn hoist_if() -> Result<Pass> {
    sequential(
        vec![
            insert_hoisted_if()?,
            super::stmt_simplify::stmt_simplify()?,
            super::remove_no_op::remove_no_op()?,
        ],
        "tirx.HoistIf",
    )
}
