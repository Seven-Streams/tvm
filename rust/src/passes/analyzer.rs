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

use tvm_ffi::{
    match_any, Any, MutContextPolicy, MutateContext, MutateValue, ObjectRefCast, ObjectRefCore,
    Result, String, UnchangedOr,
};

use super::side_effect::{side_effect, PURE};
use super::util::{int32_dtype, int_imm, prim_dtype, prim_type, same_option, unknown_loc, ContextExt};
use crate::ir::{
    AssertStmt, Bind, Call, Evaluate, Expr, For, If, IntImm, Op, PrimType, Range, RegionStmt,
    SeqStmt, StringImm, Type, Var,
};
use crate::prim::{And, BitwiseAnd, FloorDiv, Let, Select, EQ, GE, GT, LE, LT};
use crate::sym::{Analyzer, ConstraintContext};

/// The members C++ `StmtExprMutatorWithAnalyzer` adds: the analyzer and the stack of scoped
/// constraints (`ScopeStack<WithGroup<ConstraintContext>>`).
pub struct AnalyzerBase {
    pub analyzer: Analyzer,
    /// Each scope leaves its constraints, newest first, when it ends. The bottom scope lives as
    /// long as the mutator, as in C++.
    scopes: Vec<Vec<ConstraintContext>>,
    likely: Op,
    if_then_else: Op,
    launch_thread: Op,
}

impl AnalyzerBase {
    pub fn new(analyzer: Analyzer) -> Result<Self> {
        Ok(Self {
            analyzer,
            scopes: vec![Vec::new()],
            likely: Op::get("prim.likely")?,
            if_then_else: Op::get("prim.if_then_else")?,
            launch_thread: Op::get("tirx.launch_thread")?,
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
        match descend(ctx, value.to_owned())? {
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
fn descend<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, value: Any) -> Result<Option<Any>> {
    let result: Any = match_any! {
        value {
            For(op) => descend_for(ctx, op)?.into(),
            Bind(op) if op.value.ty.as_node::<crate::ir::PrimTypeObj>().is_some() => {
                descend_bind(ctx, op)?.into()
            },
            If(op) => descend_if(ctx, op)?,
            RegionStmt(op) => descend_region(ctx, op)?,
            AssertStmt(op) => descend_assert(ctx, op)?.into(),
            Call(op) if op.op.same_as(&ctx.base().if_then_else) => descend_if_then_else(ctx, op)?,
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
        let range = Range::new(op.min.clone(), op.extent.clone(), unknown_loc()?);
        ctx.base().analyzer.bind_range(&op.loop_var, &range)?;
        let min = ctx.mutate_as(&op.min)?;
        let extent = ctx.mutate_as(&op.extent)?;
        let step = ctx.mutate_as(&op.step)?;
        let body: SeqStmt = ctx.scoped(|ctx| {
            let zero: Expr = int_imm(prim_dtype(&extent)?, 0)?.into();
            ctx.base().assume_facts(&binary("prim._OpGT", &extent, &zero)?)?;
            ctx.mutate_as(&op.body)
        })?;
        if min.same_as(&op.min)
            && extent.same_as(&op.extent)
            && same_option(&step, &op.step)
            && body.same_as(&op.body)
        {
            return Ok(op);
        }
        Ok(op.with(|node| {
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
    if value.same_as(&op.value) {
        return Ok(op);
    }
    Ok(op.with_value(value))
}

/// Each arm assumes the condition it runs under; a constant condition selects its arm.
fn descend_if<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: If) -> Result<Any> {
    ctx.scoped(|ctx| {
        let condition: Expr = ctx.mutate_as(&op.condition)?;
        let likely = ctx.base().likely.clone();
        let real_condition = match condition.as_node::<crate::ir::CallObj>() {
            Some(call) if call.op.same_as(&likely) => call.args.get(0)?,
            _ => condition.clone(),
        };
        let pure = side_effect(&real_condition)? <= PURE;
        let then_case: SeqStmt = ctx.scoped(|ctx| {
            if pure {
                ctx.base().assume_facts(&real_condition)?;
            }
            ctx.mutate_as(&op.then_case)
        })?;
        let else_case: Option<SeqStmt> = match &op.else_case {
            Some(else_case) => {
                let negated = ctx.base().analyzer.rewrite_simplify(&not(&real_condition)?)?;
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
                None => Evaluate::new(unknown_loc()?, int_imm(int32_dtype(), 0)?.into()).into(),
            });
        }
        if condition.same_as(&op.condition)
            && then_case.same_as(&op.then_case)
            && same_option(&else_case, &op.else_case)
        {
            return Ok(op.into());
        }
        Ok(op
            .with(|node| {
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
        if op.op.same_as(&ctx.base().launch_thread) {
            let var = op.body_params.get(0)?;
            let extent = op.args.get(1)?;
            let zero: Expr = int_imm(prim_dtype(&extent)?, 0)?.into();
            ctx.base().analyzer.bind_range(&var, &Range::new(zero, extent, unknown_loc()?))?;
        }
        if let Some(fields) = op.attrs.dict.get(&String::from("cuda.launch_fields"))? {
            let fields: tvm_ffi::Array<String> = fields.try_into()?;
            for (i, field) in fields.iter().enumerate() {
                let intrinsic = match field.as_str() {
                    f if f.starts_with("grid.") => "tirx.cuda.block_idx",
                    f if f.starts_with("block.") => "tirx.cuda.thread_idx",
                    _ => continue,
                };
                let index = launch_index(intrinsic, &field.as_str()[field.len() - 1..])?;
                let zero: Expr = int_imm(int32_dtype(), 0)?.into();
                ctx.base().assume(&binary("prim._OpGE", &index, &zero)?)?;
                ctx.base().assume(&binary("prim._OpLT", &index, &op.args.get(i)?)?)?;
            }
        }
        ctx.default_mutate(&op)
    })
}

/// `Call(int32, Op::Get(intrinsic), {StringImm(axis)})`.
fn launch_index(intrinsic: &str, axis: &str) -> Result<Expr> {
    let axis: StringImm = tvm_ffi::cached_global_func!("ir.StringImm")
        .call_tuple((String::from(axis), unknown_loc()?))?
        .try_into()?;
    let args: tvm_ffi::Array<Expr> = tvm_ffi::Array::new(vec![axis.into()]);
    tvm_ffi::cached_global_func!("ir.Call")
        .call_tuple((
            prim_type(int32_dtype())?,
            Op::get(intrinsic)?,
            args,
            Option::<crate::ir::Attrs>::None,
            tvm_ffi::Array::<Type>::new(vec![]),
            unknown_loc()?,
        ))?
        .try_into()
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
    if condition.same_as(&op.condition) {
        return Ok(op);
    }
    Ok(op.with_condition(condition))
}

/// `if_then_else(c, t, f)`: `t` under `c`, `f` under `!c`; a constant `c` selects its value.
fn descend_if_then_else<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: Call) -> Result<Any> {
    let (cond, true_value, false_value) = (op.args.get(0)?, op.args.get(1)?, op.args.get(2)?);
    let cond_new: Expr = ctx.mutate_as(&cond)?;
    let true_new: Expr = ctx.scoped(|ctx| {
        ctx.base().assume_facts(&cond_new)?;
        ctx.mutate_as(&true_value)
    })?;
    let not_cond = not(&cond_new)?;
    let false_new: Expr = ctx.scoped(|ctx| {
        ctx.base().assume(&not_cond)?;
        ctx.mutate_as(&false_value)
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
    let call: Call = tvm_ffi::cached_global_func!("ir.Call")
        .call_tuple((
            op.ty.clone(),
            op.op.clone(),
            tvm_ffi::Array::new(vec![cond_new, true_new, false_new]),
            op.attrs.clone(),
            tvm_ffi::Array::<Type>::new(vec![]),
            op.loc.clone(),
        ))?
        .try_into()?;
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
    let result: Expr = tvm_ffi::cached_global_func!("prim.Let")
        .call_tuple((op.var.clone(), value, body, unknown_loc()?))?
        .try_into()?;
    Ok(result.into())
}

/// `select(c, t, f)`: `t` under `c`, `f` under `!c`; a constant `c` selects its value.
fn descend_select<S: AnalyzerState>(ctx: &mut MutateContext<'_, S>, op: Select) -> Result<Any> {
    let cond: Expr = ctx.mutate_as(&op.condition)?;
    let true_value: Expr = ctx.scoped(|ctx| {
        ctx.base().assume_facts(&cond)?;
        ctx.mutate_as(&op.true_value)
    })?;
    let negated = ctx.base().analyzer.rewrite_simplify(&not(&cond)?)?;
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
    let result: Expr = tvm_ffi::cached_global_func!("prim.Select")
        .call_tuple((cond, true_value, false_value, unknown_loc()?))?
        .try_into()?;
    Ok(result.into())
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

fn floordiv_facts(div: &FloorDiv, k: &IntImm, kind: Compare, out: &mut Vec<Expr>) -> Result<()> {
    let Some(divisor) = div.b.as_node::<crate::ir::IntImmObj>() else { return Ok(()) };
    // `ffi::BigInt` arithmetic in C++; i128 covers every value an `int64` IntImm can hold.
    let (Some(divisor), Some(k)) = (divisor.value.to_i64(), k.value.to_i64()) else {
        return Ok(());
    };
    if divisor <= 0 {
        return Ok(());
    }
    let dtype = prim_dtype(&div.a)?;
    let (lo, hi) = (k as i128 * divisor as i128, (k as i128 + 1) * divisor as i128);
    let signed = dtype.code == DL_INT;
    let limit: i128 = 1i128 << (dtype.bits as u32 - signed as u32);
    let minimum = if signed { -limit } else { 0 };
    if ![lo, hi].iter().all(|v| (minimum..limit).contains(v)) {
        return Ok(());
    }
    let (lo, hi): (Expr, Expr) = (int_imm(dtype, lo)?.into(), int_imm(dtype, hi)?.into());
    let a = &div.a;
    match kind {
        Compare::Eq => {
            out.push(binary("prim._OpGE", a, &lo)?);
            out.push(binary("prim._OpLT", a, &hi)?);
        }
        Compare::Lt => out.push(binary("prim._OpLT", a, &lo)?),
        Compare::Le => out.push(binary("prim._OpLT", a, &hi)?),
        Compare::Gt => out.push(binary("prim._OpGE", a, &hi)?),
        Compare::Ge => out.push(binary("prim._OpGE", a, &lo)?),
    }
    Ok(())
}

fn compare_facts(lhs: &Expr, rhs: &Expr, kind: Compare, out: &mut Vec<Expr>) -> Result<()> {
    for (div, k, kind) in [(lhs, rhs, kind), (rhs, lhs, kind.flip())] {
        if let (Ok(div), Ok(k)) =
            (div.clone().try_cast::<FloorDiv>(), k.clone().try_cast::<IntImm>())
        {
            floordiv_facts(&div, &k, kind, out)?;
        }
    }
    Ok(())
}

fn is_bool(expr: &Expr) -> bool {
    matches!(prim_dtype(expr), Ok(dtype) if dtype.code == DL_BOOL && dtype.bits == 8 && dtype.lanes == 1)
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

/// A C++ binary operator on PrimExprs (`prim._OpAdd`, `prim._OpGE`, ...): type matching and
/// constant folding, as `a + b` or `a >= b` in C++.
pub fn binary(global: &str, a: &Expr, b: &Expr) -> Result<Expr> {
    tvm_ffi::Function::get_global(global)?
        .call_tuple((a.clone(), b.clone(), unknown_loc()?))?
        .try_into()
}

/// `prim::Not(a)`, the raw constructor.
pub fn not(a: &Expr) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.Not").call_tuple((a.clone(), unknown_loc()?))?.try_into()
}

/// `IsOne` / `IsZero`: an `IntImm` with exactly this value.
pub fn is_const_int(expr: &Expr, value: i64) -> bool {
    matches!(expr.as_node::<crate::ir::IntImmObj>(), Some(imm) if imm.value == tvm_ffi::BigInt::from(value))
}

/// `analyzer->Bind(var, Range::FromMinExtent(min, extent))`.
pub fn bind_range(analyzer: &Analyzer, var: &Var, min: &Expr, extent: &Expr) -> Result<()> {
    analyzer.bind_range(var, &Range::new(min.clone(), extent.clone(), unknown_loc()?))
}

#[allow(dead_code)]
fn _assert_prim_type_is_bound(_: PrimType) {}
