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

//! `tirx.NarrowDataType` (`src/tirx/transform/narrow_datatype.cc`): narrow the integer variables,
//! constants and casts of index expressions to `target_bits` bits when every expression they
//! occur in provably fits.
//!
//! A visitor first decides the narrowed type of each `Var`, `IntImm` and `Cast`: the widest
//! of the types the enclosing expressions allow. The rewriter then replaces the variables and
//! retypes the constants and casts inside indices; the [`DataTypeRewrite`] policy keeps the
//! surrounding expressions consistent.

use crate::fast_hash::{HashMap, HashSet};

use tvm_ffi::{
    dispatch, structural_mutate, structural_visit, Any, DLDataType, DefRegionKind, Mutator,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, StructuralView, UnchangedOr,
    VisitCallbacks, VisitContext, VisitInterrupt,
};

use super::data_type_rewriter::{DataTypeRewrite, DataTypeRewriterBase, DataTypeRewriterState};
use super::util::{function_pass, int_imm, prim_type, MutatorExt};
use crate::ir::{
    AssertStmt, Bind, Call, CallObj, Expr, FloatImmObj, For, IntImm, IntImmObj, OpaqueExprObj,
    PrimTypeObj, Range, RegionStmt, SeqStmt, Stmt, TensorLoad, TensorLoadObj, TensorStore, Tuple,
    TupleGetItem, Var, VarObj,
};
use crate::ops::{Ex, PrimOps};
use crate::prim::op::{CEIL, LOG2};
use crate::prim::{
    Add, BitwiseAnd, Broadcast, Cast, Div, FloorDiv, FloorMod, LShift, Let, Max, Min, Mod, Mul,
    RShift, Ramp, Select, Shuffle, Sub, EQ, GE, GT, LE, LT, NE,
};
use crate::sym::Analyzer;
use crate::tirx::op::LAUNCH_THREAD;
use crate::transform::Pass;

const DL_INT: u8 = 0;
/// The width of an expression that does not fit (`max_bits_`).
const MAX_BITS: i32 = 64;

/// `DataTypeVisitor`: the narrowed type of every `Var`, `IntImm` and `Cast`.
struct DataTypeVisitor {
    analyzer: Analyzer,
    /// The widest type the current expression's enclosing expressions allow (`bits_`).
    bits: i32,
    target_bits: i32,
    /// The type of the extent of each loop and launched thread variable.
    vextent: HashMap<ObjectIdentity, DLDataType>,
    /// The bound of each integer expression, computed once (`bound_`).
    bounds: HashMap<ObjectIdentity, (i64, i64)>,
    /// The `Let`s whose bodies have their bounds recorded.
    memoized_lets: HashSet<ObjectIdentity>,
    /// The `Let` variables bound in the analyzer only while their bodies were analyzed.
    released_lets: HashSet<ObjectIdentity>,
    /// The narrowed type of each node, with the node.
    vmap: HashMap<ObjectIdentity, (Expr, DLDataType)>,
}

impl DataTypeVisitor {
    fn new(target_bits: i32) -> Result<Self> {
        Ok(Self {
            analyzer: Analyzer::new()?,
            bits: target_bits,
            target_bits,
            vextent: HashMap::default(),
            bounds: HashMap::default(),
            memoized_lets: HashSet::default(),
            released_lets: HashSet::default(),
            vmap: HashMap::default(),
        })
    }

    /// Record the bound of `expr` and of every subexpression `ConstIntBoundAnalyzer` reaches
    /// from it, as C++ `const_int_bound(e, &bound_)` does.
    ///
    /// Only the body of a `Let` is analyzed differently from on its own, with the variable
    /// bound to the bound of the value, so this is needed only below a `Let`.
    fn memoize(&mut self, expr: &Expr) -> Result<()> {
        let bound = self.analyzer.const_int_bound(expr)?;
        self.bounds.insert(ObjectIdentity::of(expr), (bound.min_value, bound.max_value));
        macro_rules! operands {
            ($($ty:ty),*) => {$(
                if let Ok(op) = expr.clone().try_cast::<$ty>() {
                    self.memoize(&op.a)?;
                    return self.memoize(&op.b);
                }
            )*};
        }
        operands!(
            Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max, LShift, RShift, BitwiseAnd
        );
        if let Ok(op) = expr.clone().try_cast::<Let>() {
            return self.memoize_let(&op);
        }
        if let Ok(op) = expr.clone().try_cast::<Select>() {
            self.memoize(&op.true_value)?;
            return self.memoize(&op.false_value);
        }
        if let Ok(op) = expr.clone().try_cast::<Broadcast>() {
            return self.memoize(&op.value);
        }
        if let Ok(op) = expr.clone().try_cast::<Ramp>() {
            // The last lane, `base + (lanes - 1) * stride`, reaches a stride that is not folded.
            self.memoize(&op.base_)?;
            if op.stride.as_node::<IntImmObj>().is_none() {
                self.memoize(&op.stride)?;
            }
            return Ok(());
        }
        if let Ok(op) = expr.clone().try_cast::<Cast>() {
            return match self.ceil_log2_arg(&op) {
                Some(arg) if arg.as_node::<FloatImmObj>().is_none() => self.memoize(&arg),
                Some(_) => Ok(()),
                None => self.memoize(&op.value),
            };
        }
        Ok(())
    }

    /// The analyzer binds the variable of a `Let` to the bound of its value while it analyzes
    /// the body, unless the variable is bound already.
    fn memoize_let(&mut self, op: &Let) -> Result<()> {
        self.memoized_lets.insert(ObjectIdentity::of(op));
        self.memoize(&op.value)?;
        let key = ObjectIdentity::of(&op.var);
        if self.analyzer.const_int_bound_is_bound(&op.var)? && !self.released_lets.contains(&key) {
            return self.memoize(&op.body);
        }
        let value = self.analyzer.const_int_bound(&op.value)?;
        let unbound = self.analyzer.const_int_bound(&Expr::from(&op.var))?;
        self.analyzer.update_const_int_bound(&op.var, &value, true)?;
        self.released_lets.remove(&key);
        let result = self.memoize(&op.body);
        // Bound to everything, the variable analyzes as if it were not bound.
        self.analyzer.update_const_int_bound(&op.var, &unbound, true)?;
        self.released_lets.insert(key);
        result
    }

    /// `FindCeilLog2Arg`: `arg` of an integer `cast(ceil(log2(arg)))`.
    fn ceil_log2_arg(&self, op: &Cast) -> Option<Expr> {
        let ty = op.ty.as_node::<PrimTypeObj>()?;
        if ty.dtype.code != DL_INT {
            return None;
        }
        let ceil = op.value.as_node::<CallObj>().filter(|call| CEIL.is(&call.op))?;
        let log2 = ceil.args.get(0).ok()?;
        let log2 = log2.as_node::<CallObj>().filter(|call| LOG2.is(&call.op))?;
        log2.args.get(0).ok()
    }

    /// `target_bits` if `expr` is that narrow or its bound fits a signed integer of that
    /// width; the widest width otherwise.
    fn fitting_bits(&mut self, expr: &Expr) -> Result<i32> {
        let key = ObjectIdentity::of(expr);
        let (min, max) = match self.bounds.get(&key) {
            Some(bound) => *bound,
            None => {
                let bound = self.analyzer.const_int_bound(expr)?;
                let bound = (bound.min_value, bound.max_value);
                self.bounds.insert(key, bound);
                bound
            }
        };
        let limit = 1i128 << (self.target_bits - 1);
        let fits = expr.prim_dtype()?.bits as i32 <= self.target_bits
            || ((max as i128) < limit && min as i128 >= -limit);
        Ok(if fits { self.target_bits } else { MAX_BITS })
    }

    /// Record that `expr` may be narrowed to `bits`; the widest requirement wins.
    fn narrow(&mut self, expr: Expr, bits: i32) -> Result<()> {
        let dtype = expr.prim_dtype()?;
        let key = ObjectIdentity::of(&expr);
        let bits = match self.vmap.get(&key) {
            Some((_, found)) => (found.bits as i32).max(bits),
            None => bits,
        };
        self.vmap.insert(key, (expr, DLDataType { bits: bits as u8, ..dtype }));
        Ok(())
    }
}

/// C++ `DataTypeVisitor::Visit`: inside an integer expression that fits `target_bits`, its
/// operands may narrow to it.
fn visit(
    value: &StructuralView,
    ctx: &mut VisitContext<'_, DataTypeVisitor>,
) -> Result<Option<VisitInterrupt>> {
    let integer = value
        .cast::<Expr>()
        .filter(|e| e.ty.as_node::<PrimTypeObj>().is_some_and(|t| t.dtype.code == DL_INT));
    let Some(expr) = integer else { return visit_node(value, ctx) };
    let state = ctx.state_mut();
    let bits = state.fitting_bits(&expr)?;
    let outer = state.bits;
    state.bits = bits.max(outer);
    let result = visit_node(value, ctx);
    ctx.state_mut().bits = outer;
    result
}

/// Run each visit in order, returning from the caller at the first interrupt.
macro_rules! visit_seq {
    ($($visit:expr),* $(,)?) => {{
        $(if let Some(interrupt) = $visit? {
            return Ok(Some(interrupt));
        })*
        Ok(None)
    }};
}

/// The `Visit_` overrides of `DataTypeVisitor`.
fn visit_node(
    value: &StructuralView,
    ctx: &mut VisitContext<'_, DataTypeVisitor>,
) -> Result<Option<VisitInterrupt>> {
    // The indices of a load are indices whatever the load's own type.
    if value.as_node::<TensorLoadObj>().is_some() {
        let state = ctx.state_mut();
        let outer = std::mem::replace(&mut state.bits, state.target_bits);
        let result = visit_operands(value, ctx);
        ctx.state_mut().bits = outer;
        return result;
    }
    if let Some(op) = value.cast::<For>() {
        let state = ctx.state_mut();
        state.analyzer.bind_range(&op.loop_var, &Range::from_min_extent(&op.min, &op.extent)?)?;
        state.vextent.insert(ObjectIdentity::of(&op.loop_var), op.extent.prim_dtype()?);
    }
    if let Some(op) = value.cast::<Let>() {
        // The analysis of an integer `Let` (here or as part of an enclosing expression) recorded
        // the bounds of its body with the variable bound.
        let state = ctx.state_mut();
        let integer = op.ty.as_node::<PrimTypeObj>().is_some_and(|t| t.dtype.code == DL_INT);
        if integer && !state.memoized_lets.contains(&ObjectIdentity::of(&op)) {
            state.memoize_let(&op)?;
        }
    }
    if let Some(op) = value.cast::<RegionStmt>() {
        let state = ctx.state_mut();
        if LAUNCH_THREAD.is(&op.op) {
            let var = op.body_params.get(0)?;
            let extent = op.args.get(1)?;
            let zero = int_imm(extent.prim_dtype()?, 0)?;
            state.analyzer.bind_range(&var, &Range::from_min_extent(zero, &extent)?)?;
            state.vextent.insert(ObjectIdentity::of(&var), extent.prim_dtype()?);
        }
    }
    if value.as_node::<VarObj>().is_some() {
        let var: Expr = value.cast().expect("checked to be a Var");
        let state = ctx.state_mut();
        if let Some(extent) = state.vextent.get(&ObjectIdentity::of(&var)).copied() {
            let bits = (extent.bits as i32).min(state.bits);
            state.narrow(var, bits)?;
        }
    }
    if let Some(imm) = value.cast::<IntImm>() {
        let expr: Expr = imm.into();
        let state = ctx.state_mut();
        let dtype = expr.prim_dtype()?;
        if dtype.code == DL_INT {
            let bits = (dtype.bits as i32).min(state.bits);
            state.narrow(expr, bits)?;
        }
        return Ok(None);
    }
    if let Some(cast) = value.cast::<Cast>() {
        let expr: Expr = cast.into();
        let state = ctx.state_mut();
        let dtype = expr.prim_dtype()?;
        if dtype.code == DL_INT {
            let bits = (dtype.bits as i32).min(state.bits);
            state.narrow(expr, bits)?;
        }
    }
    visit_operands(value, ctx)
}

/// The children `StmtExprVisitor` descends into where they differ from the structural defaults:
/// it skips the types of variable uses and of most expressions, loop and `Let` variables, and
/// vector lanes.
fn visit_operands(
    value: &StructuralView,
    ctx: &mut VisitContext<'_, DataTypeVisitor>,
) -> Result<Option<VisitInterrupt>> {
    if value.as_node::<VarObj>().is_some() {
        if ctx.def_region_kind() == DefRegionKind::None {
            return Ok(None);
        }
        return ctx.visit_children();
    }
    if value.as_node::<OpaqueExprObj>().is_some() {
        return Ok(None);
    }
    if let Some(op) = value.cast::<TensorLoad>() {
        return visit_seq!(ctx.visit(&op.source), ctx.visit(&op.indices));
    }
    if let Some(op) = value.cast::<Tuple>() {
        return ctx.visit(&op.fields);
    }
    if let Some(op) = value.cast::<TupleGetItem>() {
        return ctx.visit(&op.tuple_value);
    }
    if let Some(op) = value.cast::<Call>() {
        // An opaque callee has nothing to visit; an operator is interned.
        return visit_seq!(ctx.visit(&op.args), ctx.visit(&op.ty_args));
    }
    if let Some(op) = value.cast::<Let>() {
        return visit_seq!(ctx.visit(&op.value), ctx.visit(&op.body));
    }
    if let Some(op) = value.cast::<Ramp>() {
        return visit_seq!(ctx.visit(&op.base_), ctx.visit(&op.stride));
    }
    if let Some(op) = value.cast::<Broadcast>() {
        return ctx.visit(&op.value);
    }
    if let Some(op) = value.cast::<Shuffle>() {
        return visit_seq!(ctx.visit(&op.indices), ctx.visit(&op.vectors));
    }
    if let Some(op) = value.cast::<Bind>() {
        return visit_seq!(ctx.visit(&op.value), ctx.visit_with(&op.var, DefRegionKind::Simple));
    }
    if let Some(op) = value.cast::<RegionStmt>() {
        return visit_seq!(
            ctx.visit(&op.attrs),
            ctx.visit(&op.args),
            ctx.visit_with(&op.body_params, DefRegionKind::Simple),
            ctx.visit(&op.body),
            ctx.visit_with(&op.result_vars, DefRegionKind::Simple),
        );
    }
    if let Some(op) = value.cast::<For>() {
        if let Some(step) = &op.step {
            return visit_seq!(
                ctx.visit(&op.min),
                ctx.visit(&op.extent),
                ctx.visit(step),
                ctx.visit(&op.body)
            );
        }
        return visit_seq!(ctx.visit(&op.min), ctx.visit(&op.extent), ctx.visit(&op.body));
    }
    if let Some(op) = value.cast::<TensorStore>() {
        return visit_seq!(ctx.visit(&op.dest), ctx.visit(&op.indices), ctx.visit(&op.value));
    }
    if let Some(op) = value.cast::<AssertStmt>() {
        return visit_seq!(ctx.visit(&op.condition), ctx.visit(&op.error_kind));
    }
    ctx.visit_children()
}

/// `NarrowDataTypeRewriter`.
struct NarrowRewriter {
    base: DataTypeRewriterBase,
    /// The narrowed type of each constant and cast that changes type.
    vmap: HashMap<ObjectIdentity, DLDataType>,
}

impl DataTypeRewriterState for NarrowRewriter {
    fn base(&mut self) -> &mut DataTypeRewriterBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = DataTypeRewrite)]
impl NarrowRewriter {
    /// A narrowed constant in an index takes its narrowed type.
    fn mutate_int_imm(&mut self, op: IntImm, m: &mut Mutator) -> Result<Any> {
        if self.base.is_enabled {
            if let Some(dtype) = self.vmap.get(&ObjectIdentity::of(&op)).copied() {
                return Ok(crate::ops::big_int_imm(dtype, op.value.clone())?.into());
            }
        }
        m.default_mutate(self, &op)
    }

    /// A narrowed cast in an index casts its value to the narrowed type, or is the value itself.
    fn mutate_cast(&mut self, op: Cast, m: &mut Mutator) -> Result<Any> {
        match self.vmap.get(&ObjectIdentity::of(&op)).copied() {
            Some(dtype) if self.base.is_enabled => {
                let cast: Cast = m.default_mutate_as(self, &op)?;
                let value = cast.value.clone();
                if value.prim_dtype()? == dtype {
                    return Ok(value.into());
                }
                Ok(Cast::new(value, prim_type(dtype)?)?.into())
            }
            _ => m.default_mutate(self, &op),
        }
    }

    fn mutate_add(&mut self, op: Add, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a + b)
    }
    fn mutate_sub(&mut self, op: Sub, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a - b)
    }
    fn mutate_mul(&mut self, op: Mul, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a * b)
    }
    fn mutate_div(&mut self, op: Div, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.div(b))
    }
    fn mutate_mod(&mut self, op: Mod, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.truncmod(b))
    }
    fn mutate_floordiv(&mut self, op: FloorDiv, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.floordiv(b))
    }
    fn mutate_floormod(&mut self, op: FloorMod, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.floormod(b))
    }
    fn mutate_min(&mut self, op: Min, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.min(b))
    }
    fn mutate_max(&mut self, op: Max, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.max(b))
    }
    fn mutate_eq(&mut self, op: EQ, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.equal(b))
    }
    fn mutate_ne(&mut self, op: NE, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.not_equal(b))
    }
    fn mutate_le(&mut self, op: LE, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.less_equal(b))
    }
    fn mutate_lt(&mut self, op: LT, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.less(b))
    }
    fn mutate_gt(&mut self, op: GT, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.greater(b))
    }
    fn mutate_ge(&mut self, op: GE, m: &mut Mutator) -> Result<Any> {
        self.binary(&op.a, &op.b, m, |a, b| a.greater_equal(b))
    }
}

impl NarrowRewriter {
    /// A binary operation whose operands changed: when their types now differ, both are mutated
    /// again as indices, so narrowed constants and casts on either side match.
    fn binary(
        &mut self,
        a: &Expr,
        b: &Expr,
        m: &mut Mutator,
        combine: fn(&Expr, &Expr) -> Ex,
    ) -> Result<Any> {
        let new_a: Expr = m.mutate_as(self, a)?;
        let new_b: Expr = m.mutate_as(self, b)?;
        let types_match = new_a.prim_dtype()? == new_b.prim_dtype()?;
        if new_a.same_as(a) && new_b.same_as(b) && types_match {
            return Ok(UnchangedOr::<Any>::unchanged().into());
        }
        if types_match {
            return Ok(combine(&new_a, &new_b).expr()?.into());
        }
        let enabled = std::mem::replace(&mut self.base.is_enabled, true);
        let operands = m.mutate_as(self, a).and_then(|lhs| Ok((lhs, m.mutate_as(self, b)?)));
        self.base.is_enabled = enabled;
        let (lhs, rhs): (Expr, Expr) = operands?;
        Ok(combine(&lhs, &rhs).expr()?.into())
    }
}

/// C++ `NarrowDataTypeRewriter::Rewrite(stmt)`.
fn narrow_body(body: SeqStmt, target_bits: i32) -> Result<SeqStmt> {
    let mut visitor = VisitCallbacks::new(DataTypeVisitor::new(target_bits)?, visit);
    structural_visit(&body, &mut visitor)?;
    let mut rewriter =
        NarrowRewriter { base: DataTypeRewriterBase::new(true), vmap: HashMap::default() };
    for (key, (expr, dtype)) in visitor.into_state().vmap {
        if expr.prim_dtype()? == dtype {
            continue;
        }
        if expr.as_node::<VarObj>().is_some() {
            let var: Var = expr.try_cast()?;
            let narrowed = Var::prim(var.name.as_str(), dtype)?;
            rewriter.base.remap_set(&var, narrowed);
        }
        rewriter.vmap.insert(key, dtype);
    }
    let body: Stmt = structural_mutate(body, &mut rewriter)?.try_into()?;
    SeqStmt::from_stmt(body)
}

pub fn narrow_datatype(target_bits: i32) -> Result<Pass> {
    function_pass("tirx.NarrowDataType", 0, move |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        Ok(Some(func.with_body(Some(narrow_body(body, target_bits)?))))
    })
}
