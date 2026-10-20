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

//! C++ `StmtExprMutatorWithAnalyzer` (`src/tirx/ir/ir_mutator_with_analyzer.cc`) as a mutation
//! policy.
//!
//! A pass keeps an [`AnalyzerBase`] field, implements [`AnalyzerState`], and declares
//! `#[dispatch(mutate, policy = AnalyzerScope)]`. Its handlers are the C++ `Mutate_` overrides;
//! every value no handler takes, and every `m.default_mutate(self, &op)` a handler makes (the C++
//! `Parent::Mutate_(op)`), descends through [`AnalyzerScope`]: a loop binds its variable for its
//! body, a branch assumes its condition for its arm, and so on. Children re-enter the pass's
//! handlers.

use crate::fast_hash::HashSet;

use tvm_ffi::{
    Any, AnyView, Array, Map, match_any, MutateContext, MutateValue, MutContextPolicy,
    ObjectIdentity, ObjectRefCore, Result, String, structural_walk, StructuralView, UnchangedOr,
    WalkOrder, WalkResult,
};

use super::side_effect::{side_effect, PURE};
use super::util::{int32_dtype, int_imm, prim_type, ContextExt};
use crate::ir::{
    AssertStmt, Bind, Call, Evaluate, Expr, For, If, OpHandle, PrimType, Range, RegionStmt,
    SeqStmt, StringImm, Var,
};
use crate::ops::PrimOps;
use crate::prim::op::{IF_THEN_ELSE, LIKELY};
use crate::prim::{And, BitwiseAnd, FloorDivObj, Let, Not, Select, EQ, GE, GT, LE, LT};
use crate::sym::{Analyzer, ConstraintContext};
use crate::tirx::op::{CUDA_BLOCK_IDX, CUDA_THREAD_IDX, LAUNCH_THREAD};

/// The members C++ `StmtExprMutatorWithAnalyzer` adds: the analyzer and the stack of scoped
/// constraints (`ScopeStack<WithGroup<ConstraintContext>>`).
pub struct AnalyzerBase {
    pub analyzer: Analyzer,
    /// Each scope leaves its constraints, newest first, when it ends. The bottom scope lives as
    /// long as the mutator, as in C++.
    scopes: Vec<Vec<ConstraintContext>>,
    /// `iter_vars_`: each loop variable and launched thread index with its range, in the order
    /// first entered. Never removed, as in C++.
    iter_vars: Vec<(Var, Range)>,
    /// `iter_predicates_`: the conditions over loop variables of the enclosing branches.
    iter_predicates: Vec<Expr>,
}

impl AnalyzerBase {
    pub fn new(analyzer: Analyzer) -> Result<Self> {
        Ok(Self {
            analyzer,
            scopes: vec![Vec::new()],
            iter_vars: Vec::new(),
            iter_predicates: Vec::new(),
        })
    }

    /// Assume `constraint` until the current scope ends (`Current().Emplace(...)`).
    pub fn assume(&mut self, constraint: &Expr) -> Result<()> {
        let context = self.analyzer.assume(constraint)?;
        self.scopes.last_mut().expect("the bottom scope is never left").push(context);
        Ok(())
    }

    /// `sym::detail::EnterConstraintFacts`: `condition`, plus the ranges it implies for any
    /// `floordiv` it compares with a constant.
    pub fn assume_facts(&mut self, condition: &Expr) -> Result<()> {
        self.assume(condition)?;
        let mut derived = Vec::new();
        collect_derived_facts(condition, &mut derived)?;
        derived.iter().try_for_each(|fact| self.assume(fact))
    }

    /// `iter_vars_.Set(var, dom)`.
    fn record_iter_var(&mut self, var: &Var, dom: &Range) {
        match self.iter_vars.iter_mut().find(|(known, _)| known.same_as(var)) {
            Some(entry) => entry.1 = dom.clone(),
            None => self.iter_vars.push((var.clone(), dom.clone())),
        }
    }

    /// The heuristic of C++ `WithRecordIterPredicate`: whether `condition` mentions a recorded
    /// loop variable.
    fn mentions_iter_var(&self, condition: &Expr) -> Result<bool> {
        let vars: HashSet<ObjectIdentity> =
            self.iter_vars.iter().map(|(var, _)| ObjectIdentity::of(var)).collect();
        let walker = |var: Var| match vars.contains(&ObjectIdentity::of(&var)) {
            true => WalkResult::Interrupt,
            false => WalkResult::Advance,
        };
        Ok(structural_walk(condition, walker, WalkOrder::PreOrder)?.is_some())
    }

    /// C++ `IterMapSimplifyWithContext(indices, non_trivial_only)`: `sym::IterMapSimplify` over
    /// the recorded loop variables, under the conjunction of the recorded branch conditions.
    pub fn iter_map_simplify_with_context(
        &self,
        indices: &Array<Expr>,
        non_trivial_only: bool,
    ) -> Result<Array<Expr>> {
        let bool_dtype = tvm_ffi::DLDataType { code: 6, bits: 8, lanes: 1 };
        let mut predicate = crate::ops::Ex::from(int_imm(bool_dtype, 1));
        for condition in &self.iter_predicates {
            predicate = predicate.logical_and(condition);
        }
        let iters: Map<Var, Range> = self.iter_vars.iter().cloned().collect();
        // `IterMapLevel::Surjective`, with trivial iterators simplified.
        let simplified =
            self.analyzer.iter_map_simplify(indices, iters, &predicate.expr()?, 1, true)?;
        if !non_trivial_only {
            return Ok(simplified);
        }
        let mut kept = Vec::with_capacity(indices.len());
        for (index, value) in indices.iter().zip(simplified.iter()) {
            let is_prim_var =
                index.as_node::<crate::ir::VarObj>().is_some() && is_prim_expr(&index);
            kept.push(match value.as_node::<crate::ir::IntImmObj>().is_some() && is_prim_var {
                true => index,
                false => value,
            });
        }
        Ok(Array::new(kept))
    }

    fn enter_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    fn leave_scope(&mut self) {
        if let Some(mut scope) = self.scopes.pop() {
            while scope.pop().is_some() {}
        }
    }
}

impl Drop for AnalyzerBase {
    fn drop(&mut self) {
        while !self.scopes.is_empty() {
            self.leave_scope();
        }
    }
}

/// A mutator that descends through [`AnalyzerScope`].
pub trait AnalyzerState {
    fn base(&mut self) -> &mut AnalyzerBase;
}

/// The default descent of `StmtExprMutatorWithAnalyzer`.
pub struct AnalyzerScope;

impl<S: AnalyzerState> MutContextPolicy<S> for AnalyzerScope {
    fn default_mutate(
        &self,
        value: MutateValue<'_>,
        ctx: &mut MutateContext<'_, S>,
    ) -> Result<UnchangedOr<Any>> {
        match descend(ctx, value.as_value())? {
            Some(result) => Ok(UnchangedOr::changed(result)),
            None => ctx.default_maybe_inplace_mutate_result(value),
        }
    }
}

/// Scoped analyzer access for the policy's own descent.
trait Scoped {
    fn base(&mut self) -> &mut AnalyzerBase;

    /// `constraint_scope_.WithNewScope(body)`: the constraints `body` adds end with it, on error
    /// too.
    fn scoped<T>(&mut self, body: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.base().enter_scope();
        let result = body(self);
        self.base().leave_scope();
        result
    }
}

impl<S: AnalyzerState> Scoped for MutateContext<'_, S> {
    fn base(&mut self) -> &mut AnalyzerBase {
        self.state_mut().base()
    }
}

/// The nodes whose descent the analyzer scopes; `None` for every other value.
fn descend<S: AnalyzerState>(
    ctx: &mut MutateContext<'_, S>,
    view: &StructuralView,
) -> Result<Option<Any>> {
    // Match on the borrowed view: only a matching arm takes a reference on the node, so the
    // nodes this policy does not handle (most of them) cost one type test and no refcount
    // traffic.
    let value = AnyView::from(view);
    let result: Any = match_any! {
        value {
            For(op) => descend_for(ctx, op)?.into(),
            Bind(op) if op.value.ty.as_node::<crate::ir::PrimTypeObj>().is_some() => {
                descend_bind(ctx, op)?.into()
            },
            If(op) => descend_if(ctx, op)?,
            RegionStmt(op) => descend_region(ctx, op)?,
            AssertStmt(op) => descend_assert(ctx, op)?.into(),
            Call(op) if IF_THEN_ELSE.is(&op.op) => descend_if_then_else(ctx, op)?,
            Let(op) => descend_let(ctx, op)?,
            Select(op) => descend_select(ctx, op)?,
            _ => return Ok(None),
        }
    };
    Ok(Some(result))
}

/// The loop variable ranges over the loop for all of it; the body may also assume the loop runs.
fn descend_for<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: For) -> Result<For> {
    ctx.scoped(|ctx| {
        let range = Range::from_min_extent(&op.min, &op.extent)?;
        ctx.base().analyzer.bind_range(&op.loop_var, &range)?;
        ctx.base().record_iter_var(&op.loop_var, &range);
        let min = ctx.mutate_as(&op.min)?;
        let extent = ctx.mutate_as(&op.extent)?;
        let step = ctx.mutate_as(&op.step)?;
        let body: SeqStmt = ctx.scoped(|ctx| {
            // C++ `extent > IntImm(extent.ty(), 0)`.
            let positive = extent.greater(int_imm(extent.prim_dtype()?, 0)).expr()?;
            ctx.base().assume_facts(&positive)?;
            ctx.mutate_as(&op.body)
        })?;
        Ok(op.update(|node| {
            node.min = min;
            node.extent = extent;
            node.body = body;
            node.step = step;
        }))
    })
}

/// A pure bound value is known for the rest of the scope.
fn descend_bind<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: Bind) -> Result<Bind> {
    let value: Expr = ctx.mutate_as(&op.value)?;
    if is_prim_expr(&value) && side_effect(&value)? <= PURE {
        ctx.base().analyzer.bind(&op.var, &value)?;
    }
    Ok(op.update(|node| node.value = value))
}

/// Each arm assumes the condition it runs under; a constant condition selects its arm.
fn descend_if<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: If) -> Result<Any> {
    ctx.scoped(|ctx| {
        let condition: Expr = ctx.mutate_as(&op.condition)?;
        let real_condition = match condition.as_node::<crate::ir::CallObj>() {
            Some(call) if LIKELY.is(&call.op) => call.args.get(0)?,
            _ => condition.clone(),
        };
        let pure = side_effect(&real_condition)? <= PURE;
        let then_case: SeqStmt = ctx.scoped(|ctx| {
            if pure {
                ctx.base().assume_facts(&real_condition)?;
            }
            // C++ records `IntImm::Bool(true)`, which mentions no loop variable, when impure.
            with_iter_predicate(ctx, pure.then_some(&real_condition), |ctx| {
                ctx.mutate_as(&op.then_case)
            })
        })?;
        let else_case: Option<SeqStmt> = match &op.else_case {
            Some(else_case) => {
                let negated =
                    ctx.base().analyzer.rewrite_simplify(&Not::new(&real_condition)?.into())?;
                Some(ctx.scoped(|ctx| {
                    if pure {
                        ctx.base().assume(&negated)?;
                    }
                    ctx.mutate_as(else_case)
                })?)
            }
            None => None,
        };
        if is_const_int(&real_condition, 1) {
            return Ok(then_case.into());
        }
        if is_const_int(&real_condition, 0) {
            return Ok(match else_case {
                Some(else_case) => else_case.into(),
                None => Evaluate::new(0)?.into(),
            });
        }
        Ok(op
            .update(|node| {
                node.condition = condition;
                node.then_case = then_case;
                node.else_case = else_case;
            })
            .into())
    })
}

/// A launched thread index ranges over its extent, and the launch fields bound the CUDA block and
/// thread indices, for the region's body.
fn descend_region<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: RegionStmt) -> Result<Any> {
    ctx.scoped(|ctx| {
        if LAUNCH_THREAD.is(&op.op) {
            let var = op.body_params.get(0)?;
            let extent = op.args.get(1)?;
            let zero = int_imm(extent.prim_dtype()?, 0)?;
            let dom = Range::from_min_extent(zero, extent)?;
            ctx.base().analyzer.bind_range(&var, &dom)?;
            ctx.base().record_iter_var(&var, &dom);
        }
        if let Some(fields) = op.attrs.dict.get(&String::from("cuda.launch_fields"))? {
            let fields: tvm_ffi::Array<String> = fields.try_into()?;
            for (i, field) in fields.iter().enumerate() {
                let intrinsic = match field.as_str() {
                    f if f.starts_with("grid.") => &CUDA_BLOCK_IDX,
                    f if f.starts_with("block.") => &CUDA_THREAD_IDX,
                    _ => continue,
                };
                let index = launch_index(intrinsic, &field.as_str()[field.len() - 1..])?;
                ctx.base().assume(&index.greater_equal(0).expr()?)?;
                ctx.base().assume(&index.less(op.args.get(i)?).expr()?)?;
            }
        }
        ctx.default_mutate(&op)
    })
}

/// C++ `WithRecordIterPredicate(condition, body)`: `condition` is recorded for `body` when it
/// mentions a loop variable.
fn with_iter_predicate<S: AnalyzerState, T>(
    ctx: &mut MutateContext<'_, S>,
    condition: Option<&Expr>,
    body: impl FnOnce(&mut MutateContext<'_, S>) -> Result<T>,
) -> Result<T> {
    let record = match condition {
        Some(condition) => ctx.base().mentions_iter_var(condition)?,
        None => false,
    };
    if !record {
        return body(ctx);
    }
    ctx.base().iter_predicates.push(condition.expect("recorded").clone());
    let result = body(ctx);
    ctx.base().iter_predicates.pop();
    result
}

/// `Call(int32, intrinsic, {StringImm(axis)})`.
fn launch_index(intrinsic: &OpHandle, axis: &str) -> Result<Expr> {
    let axis: Expr = StringImm::new(axis)?.into();
    Ok(Call::new(prim_type(int32_dtype())?, intrinsic.get()?, vec![axis])?.into())
}

/// A pure asserted condition holds for the rest of the scope.
fn descend_assert<S: AnalyzerState>(
    ctx: &mut MutateContext<'_, S>,
    op: AssertStmt,
) -> Result<AssertStmt> {
    let condition: Expr = ctx.mutate_as(&op.condition)?;
    if side_effect(&condition)? <= PURE {
        ctx.base().assume(&condition)?;
    }
    Ok(op.update(|node| node.condition = condition))
}

/// `if_then_else(c, t, f)`: `t` under `c`, `f` under `!c`; a constant `c` selects its value.
fn descend_if_then_else<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: Call) -> Result<Any> {
    let (cond, true_value, false_value) = (op.args.get(0)?, op.args.get(1)?, op.args.get(2)?);
    let cond_new: Expr = ctx.mutate_as(&cond)?;
    let true_new: Expr = ctx.scoped(|ctx| {
        ctx.base().assume_facts(&cond_new)?;
        with_iter_predicate(ctx, Some(&cond_new), |ctx| ctx.mutate_as(&true_value))
    })?;
    let not_cond: Expr = Not::new(&cond_new)?.into();
    let false_new: Expr = ctx.scoped(|ctx| {
        ctx.base().assume(&not_cond)?;
        with_iter_predicate(ctx, Some(&not_cond), |ctx| ctx.mutate_as(&false_value))
    })?;
    if is_const_int(&cond_new, 0) {
        return Ok(false_new.into());
    }
    if is_const_int(&cond_new, 1) {
        return Ok(true_new.into());
    }
    if cond_new.same_as(&cond) && true_new.same_as(&true_value) && false_new.same_as(&false_value) {
        return Ok(op.into());
    }
    let call = Call::new_full(
        Some(op.ty.clone()),
        op.op.clone(),
        tvm_ffi::Array::new(vec![cond_new, true_new, false_new]),
        op.attrs.clone(),
        tvm_ffi::Array::new(vec![]),
        &op.loc,
    )?;
    Ok(call.into())
}

/// A pure let value is known inside the body.
fn descend_let<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: Let) -> Result<Any> {
    let value: Expr = ctx.mutate_as(&op.value)?;
    if side_effect(&value)? <= PURE {
        ctx.base().analyzer.bind(&op.var, &value)?;
    }
    let body: Expr = ctx.mutate_as(&op.body)?;
    if value.same_as(&op.value) && body.same_as(&op.body) {
        return Ok(op.into());
    }
    Ok(Let::new(op.var.clone(), value, body)?.into())
}

/// `select(c, t, f)`: `t` under `c`, `f` under `!c`; a constant `c` selects its value.
fn descend_select<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: Select) -> Result<Any> {
    let cond: Expr = ctx.mutate_as(&op.condition)?;
    let true_value: Expr = ctx.scoped(|ctx| {
        ctx.base().assume_facts(&cond)?;
        ctx.mutate_as(&op.true_value)
    })?;
    let negated = ctx.base().analyzer.rewrite_simplify(&Not::new(&cond)?.into())?;
    let false_value: Expr = ctx.scoped(|ctx| {
        ctx.base().assume(&negated)?;
        ctx.mutate_as(&op.false_value)
    })?;
    if is_const_int(&cond, 0) {
        return Ok(false_value.into());
    }
    if is_const_int(&cond, 1) {
        return Ok(true_value.into());
    }
    if cond.same_as(&op.condition)
        && true_value.same_as(&op.true_value)
        && false_value.same_as(&op.false_value)
    {
        return Ok(op.into());
    }
    Ok(Select::new(cond, true_value, false_value)?.into())
}

fn is_prim_expr(expr: &Expr) -> bool {
    expr.ty.as_node::<crate::ir::PrimTypeObj>().is_some()
}

/// A comparison of `floordiv(a, divisor)` with the constant `k`, as the bound on `a` it implies.
#[derive(Clone, Copy)]
enum Compare {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Compare {
    /// The comparison with its operands swapped: `k < floordiv(..)` is `floordiv(..) > k`.
    fn flip(self) -> Self {
        match self {
            Compare::Eq => Compare::Eq,
            Compare::Lt => Compare::Gt,
            Compare::Le => Compare::Ge,
            Compare::Gt => Compare::Lt,
            Compare::Ge => Compare::Le,
        }
    }
}

const DL_INT: u8 = 0;
const DL_BOOL: u8 = 6;

fn floordiv_facts(
    div: &FloorDivObj,
    k: &crate::ir::IntImmObj,
    kind: Compare,
    out: &mut Vec<Expr>,
) -> Result<()> {
    let Some(divisor) = div.b.as_node::<crate::ir::IntImmObj>() else { return Ok(()) };
    // `ffi::BigInt` arithmetic in C++; i128 covers every value an `int64` IntImm can hold.
    let (Some(divisor), Some(k)) = (divisor.value.to_i64(), k.value.to_i64()) else {
        return Ok(());
    };
    if divisor <= 0 {
        return Ok(());
    }
    let dtype = div.a.prim_dtype()?;
    let (lo, hi) = (k as i128 * divisor as i128, (k as i128 + 1) * divisor as i128);
    let signed = dtype.code == DL_INT;
    let limit: i128 = 1i128 << (dtype.bits as u32 - signed as u32);
    let minimum = if signed { -limit } else { 0 };
    if ![lo, hi].iter().all(|v| (minimum..limit).contains(v)) {
        return Ok(());
    }
    let (lo, hi) = (int_imm(dtype, lo)?, int_imm(dtype, hi)?);
    let a = &div.a;
    match kind {
        Compare::Eq => {
            out.push(a.greater_equal(&lo).expr()?);
            out.push(a.less(&hi).expr()?);
        }
        Compare::Lt => out.push(a.less(&lo).expr()?),
        Compare::Le => out.push(a.less(&hi).expr()?),
        Compare::Gt => out.push(a.greater_equal(&hi).expr()?),
        Compare::Ge => out.push(a.greater_equal(&lo).expr()?),
    }
    Ok(())
}

fn compare_facts(lhs: &Expr, rhs: &Expr, kind: Compare, out: &mut Vec<Expr>) -> Result<()> {
    for (div, k, kind) in [(lhs, rhs, kind), (rhs, lhs, kind.flip())] {
        if let (Some(div), Some(k)) =
            (div.as_node::<FloorDivObj>(), k.as_node::<crate::ir::IntImmObj>())
        {
            floordiv_facts(div, k, kind, out)?;
        }
    }
    Ok(())
}

fn is_bool(expr: &Expr) -> bool {
    matches!(expr.prim_dtype(), Ok(dtype) if dtype.code == DL_BOOL && dtype.bits == 8 && dtype.lanes == 1)
}

fn collect_derived_facts(condition: &Expr, out: &mut Vec<Expr>) -> Result<()> {
    match_any! {
        condition.clone() {
            And(node) => {
                collect_derived_facts(&node.a, out)?;
                collect_derived_facts(&node.b, out)
            },
            BitwiseAnd(node) if is_bool(&node.a) && is_bool(&node.b) => {
                collect_derived_facts(&node.a, out)?;
                collect_derived_facts(&node.b, out)
            },
            EQ(node) => compare_facts(&node.a, &node.b, Compare::Eq, out),
            LT(node) => compare_facts(&node.a, &node.b, Compare::Lt, out),
            LE(node) => compare_facts(&node.a, &node.b, Compare::Le, out),
            GT(node) => compare_facts(&node.a, &node.b, Compare::Gt, out),
            GE(node) => compare_facts(&node.a, &node.b, Compare::Ge, out),
            _ => Ok(()),
        }
    }
}

/// `IsOne` / `IsZero`: an `IntImm` with exactly this value.
pub fn is_const_int(expr: &Expr, value: i64) -> bool {
    matches!(expr.as_node::<crate::ir::IntImmObj>(), Some(imm) if imm.value == tvm_ffi::BigInt::from(value))
}

/// `analyzer->Bind(var, Range::FromMinExtent(min, extent))`.
pub fn bind_range(analyzer: &Analyzer, var: &Var, min: &Expr, extent: &Expr) -> Result<()> {
    analyzer.bind_range(var, &Range::from_min_extent(min, extent)?)
}

#[allow(dead_code)]
fn _assert_prim_type_is_bound(_: PrimType) {}
