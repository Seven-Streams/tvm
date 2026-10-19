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

//! `tirx.CommonSubexprElim` (`src/tirx/transform/common_subexpr_elim.cc`): bind each pure,
//! non-bool expression that occurs at least twice to a `cse_v<N>` variable, placed before the
//! narrowest statement dominating its uses, and replace the occurrences.
//!
//! As in C++, a planner scans the body and a rewriter applies its plan. The planner visits nodes
//! in the order C++ `StmtExprVisitor` does, not in reflected field order: the order expressions
//! are first seen numbers the variables.

use crate::fast_hash::{HashMap, HashSet};

use tvm_ffi::{
    dispatch, structural_map, structural_mutate, structural_walk, Any, Array, DefRegionKind,
    Mutator, ObjectIdentity, ObjectRefCore, Result, Unchanged, UnchangedOr, WalkOrder, WalkResult,
};

use super::util::{function_pass, MutatorExt};
use crate::ir::{
    AssertStmt, Bind, Break, Call, Continue, DictAttrs, Evaluate, Expr, FloatImmObj, For, If,
    IntImmObj, OpaqueExpr, OpaqueExprObj, PrimTypeObj, RegionStmt, Return, SeqStmt, Stmt,
    StringImmObj, TensorLoad, TensorLoadObj, TensorRegion, TensorStore, Tuple, TupleGetItem, Type,
    Var, VarObj, While,
};
use crate::prim::{
    AddObj, AndObj, BitwiseAndObj, BitwiseNotObj, BitwiseOrObj, BitwiseXorObj, Broadcast,
    BroadcastObj, CastObj, DivObj, FloorDivObj, FloorModObj, LShiftObj, Let, MaxObj, MinObj,
    ModObj, MulObj, NotObj, OrObj, RShiftObj, Ramp, RampObj, SelectObj, Shuffle, SubObj, EQObj,
    GEObj, GTObj, LEObj, LTObj, NEObj,
};
use crate::ops::structural_hash;
use crate::tirx::op::LAUNCH_THREAD;
use crate::transform::Pass;

const DL_BOOL: u8 = 6;

fn is_prim(expr: &Expr) -> bool {
    expr.ty.as_node::<PrimTypeObj>().is_some()
}

/// C++ `prim::ExprDeepEqual`.
fn deep_equal(lhs: &Expr, rhs: &Expr) -> Result<bool> {
    // Nodes of different kinds are never equal; this spares most calls across the FFI.
    if lhs.same_as(rhs) {
        return Ok(true);
    }
    if tvm_ffi::AnyView::from(lhs).type_index() != tvm_ffi::AnyView::from(rhs).type_index() {
        return Ok(false);
    }
    crate::ops::deep_equal(lhs, rhs)
}

/// A map keyed by expression under `ExprDeepEqual`, iterated in insertion order: C++
/// `support::OrderedMap` (and, for lookups, `std::unordered_map`) with `StructuralHash` and
/// `ExprDeepEqual`.
struct ExprTable<T> {
    entries: Vec<(Expr, T)>,
    buckets: HashMap<i64, Vec<usize>>,
}

impl<T> ExprTable<T> {
    fn new() -> Self {
        Self { entries: Vec::new(), buckets: HashMap::default() }
    }

    /// The hash of `key` and the index of its entry, if any.
    fn locate(&self, key: &Expr) -> Result<(i64, Option<usize>)> {
        let hash = structural_hash(key)?;
        if let Some(bucket) = self.buckets.get(&hash) {
            for &index in bucket {
                if deep_equal(&self.entries[index].0, key)? {
                    return Ok((hash, Some(index)));
                }
            }
        }
        Ok((hash, None))
    }

    fn find(&self, key: &Expr) -> Result<Option<usize>> {
        Ok(self.locate(key)?.1)
    }

    fn push(&mut self, hash: i64, key: Expr, value: T) -> usize {
        let index = self.entries.len();
        self.entries.push((key, value));
        self.buckets.entry(hash).or_default().push(index);
        index
    }
}

// ---------------------------------------------------------------------------------------------
// Phase 1: the planner

/// A node of the scope tree that For, If (each arm), While and launch-thread regions make.
struct Scope {
    parent: usize,
    depth: usize,
    /// The statement that made the scope; none for the root.
    creator: Option<Stmt>,
}

/// An eligible expression, by first occurrence.
struct Entry {
    count: usize,
    /// 1 + the deepest child's, 1 without children.
    expr_depth: usize,
    /// The expression, with shallower expressions already replaced by their variables.
    repr: Expr,
    lca_scope: usize,
    first_use_scope: usize,
    first_use_stmt: Stmt,
    /// The eligible table entries among the expression's operands, with multiplicity.
    children: Vec<(Expr, usize)>,
    /// Occurrences that the bindings of parent expressions absorb.
    consumed: usize,
}

/// Where to insert the bindings (by statement identity) and what to replace.
struct Plan {
    insert_before: HashMap<ObjectIdentity, Vec<Bind>>,
    remap: ExprTable<Var>,
}

/// C++ `CSEPlanner`, which visits as `StmtExprVisitor` does and records eligible expressions
/// bottom-up.
struct Planner {
    scopes: Vec<Scope>,
    table: ExprTable<Entry>,
    current_scope: usize,
    /// The statement most recently entered (C++ sets it on every statement visit).
    current_stmt: Option<Stmt>,
    let_depth: usize,
}

/// The operands of a node the planner records, in the order C++ visits them.
fn recorded_operands(expr: &Expr) -> Option<Vec<Expr>> {
    macro_rules! binary {
        ($($node:ty),+) => {
            $(if let Some(node) = expr.as_node::<$node>() {
                return Some(vec![node.a.clone(), node.b.clone()]);
            })+
        };
    }
    binary!(
        AddObj,
        SubObj,
        MulObj,
        DivObj,
        ModObj,
        FloorDivObj,
        FloorModObj,
        MinObj,
        MaxObj,
        EQObj,
        NEObj,
        LTObj,
        LEObj,
        GTObj,
        GEObj,
        AndObj,
        OrObj,
        LShiftObj,
        RShiftObj,
        BitwiseAndObj,
        BitwiseOrObj,
        BitwiseXorObj
    );
    if let Some(node) = expr.as_node::<BitwiseNotObj>() {
        return Some(vec![node.a.clone()]);
    }
    if let Some(node) = expr.as_node::<NotObj>() {
        return Some(vec![node.a.clone()]);
    }
    if let Some(node) = expr.as_node::<CastObj>() {
        return Some(vec![node.value.clone()]);
    }
    if let Some(node) = expr.as_node::<SelectObj>() {
        return Some(vec![
            node.condition.clone(),
            node.true_value.clone(),
            node.false_value.clone(),
        ]);
    }
    None
}

/// A `Call` or a `TensorLoad` of a primitive type: possibly effectful or memory dependent.
fn is_forbidden(expr: &Expr) -> bool {
    is_prim(expr)
        && (expr.as_node::<crate::ir::CallObj>().is_some()
            || expr.as_node::<TensorLoadObj>().is_some())
}

/// C++ `ForbiddenNodeFinder`: whether a forbidden node occurs under an expression.
struct ForbiddenFinder;

#[dispatch(walk)]
impl ForbiddenFinder {
    /// A variable's type belongs to its definition, which C++ does not visit from a use.
    fn walk_var(&mut self, _var: Var) -> WalkResult {
        WalkResult::Skip
    }

    fn walk_type(&mut self, _ty: Type) -> WalkResult {
        WalkResult::Skip
    }

    fn walk_expr(&mut self, expr: Expr) -> WalkResult {
        if is_forbidden(&expr) {
            WalkResult::Interrupt
        } else {
            WalkResult::Advance
        }
    }
}

/// C++ `CSEPlanner::IsEligible`: a non-leaf, non-vector-constructor, non-bool expression with
/// no call or tensor load under it.
fn is_eligible(expr: &Expr) -> Result<bool> {
    let leaf = expr.as_node::<IntImmObj>().is_some()
        || expr.as_node::<FloatImmObj>().is_some()
        || expr.as_node::<StringImmObj>().is_some()
        || expr.as_node::<VarObj>().is_some();
    if leaf || is_forbidden(expr) {
        return Ok(false);
    }
    if expr.as_node::<RampObj>().is_some() || expr.as_node::<BroadcastObj>().is_some() {
        return Ok(false);
    }
    if expr.prim_dtype()?.code == DL_BOOL {
        return Ok(false);
    }
    Ok(structural_walk(expr, &mut ForbiddenFinder, WalkOrder::PreOrder)?.is_none())
}

impl Planner {
    fn plan(body: &SeqStmt) -> Result<Plan> {
        let mut planner = Planner {
            scopes: vec![Scope { parent: usize::MAX, depth: 0, creator: None }],
            table: ExprTable::new(),
            current_scope: 0,
            current_stmt: None,
            let_depth: 0,
        };
        structural_mutate(body.clone(), &mut planner)?;
        planner.compute_plan()
    }

    fn enter(&mut self, stmt: impl Into<Stmt>) {
        self.current_stmt = Some(stmt.into());
    }

    fn alloc_scope(&mut self, parent: usize, creator: Stmt) -> usize {
        let depth = self.scopes[parent].depth + 1;
        self.scopes.push(Scope { parent, depth, creator: Some(creator) });
        self.scopes.len() - 1
    }

    /// Visit `body` in a new scope made by `creator`.
    fn scoped(&mut self, creator: impl Into<Stmt>, body: &SeqStmt, m: &mut Mutator) -> Result<()> {
        let saved = self.current_scope;
        self.current_scope = self.alloc_scope(saved, creator.into());
        let result = m.mutate(self, body);
        self.current_scope = saved;
        result.map(|_| ())
    }

    fn lca(&self, mut a: usize, mut b: usize) -> usize {
        while self.scopes[a].depth > self.scopes[b].depth {
            a = self.scopes[a].parent;
        }
        while self.scopes[b].depth > self.scopes[a].depth {
            b = self.scopes[b].parent;
        }
        while a != b {
            a = self.scopes[a].parent;
            b = self.scopes[b].parent;
        }
        a
    }

    /// C++ `RecordExpr`: count an occurrence of `expr`, whose operands are already recorded.
    fn record(&mut self, expr: &Expr, operands: &[Expr]) -> Result<()> {
        if !is_eligible(expr)? || self.let_depth > 0 {
            return Ok(());
        }
        let (hash, found) = self.table.locate(expr)?;
        if let Some(index) = found {
            let lca = self.lca(self.table.entries[index].1.lca_scope, self.current_scope);
            let entry = &mut self.table.entries[index].1;
            entry.lca_scope = lca;
            entry.count += 1;
            return Ok(());
        }
        // `CollectChildren`: operands that are table entries, with multiplicity.
        let mut children: Vec<(Expr, usize)> = Vec::new();
        let mut max_child_depth = 0;
        for operand in operands {
            let Some(child) = self.table.find(operand)? else { continue };
            max_child_depth = max_child_depth.max(self.table.entries[child].1.expr_depth);
            let mut recorded = false;
            for (existing, multiplicity) in children.iter_mut() {
                if deep_equal(existing, operand)? {
                    *multiplicity += 1;
                    recorded = true;
                    break;
                }
            }
            if !recorded {
                children.push((operand.clone(), 1));
            }
        }
        let entry = Entry {
            count: 1,
            expr_depth: 1 + max_child_depth,
            repr: expr.clone(),
            lca_scope: self.current_scope,
            first_use_scope: self.current_scope,
            first_use_stmt: self.current_stmt.clone().expect("expressions are inside a statement"),
            children,
            consumed: 0,
        };
        self.table.push(hash, expr.clone(), entry);
        Ok(())
    }

    /// C++ `FindInsertionStmt`.
    fn insertion_stmt(&self, entry: &Entry) -> Stmt {
        if entry.first_use_scope == entry.lca_scope {
            return entry.first_use_stmt.clone();
        }
        let mut scope = entry.first_use_scope;
        while self.scopes[scope].parent != entry.lca_scope {
            scope = self.scopes[scope].parent;
        }
        self.scopes[scope].creator.clone().expect("a non-root scope has a creator")
    }

    /// C++ `ComputePlan`: shallower expressions first, each binding propagated into the deeper
    /// expressions that contain it.
    fn compute_plan(mut self) -> Result<Plan> {
        let mut order: Vec<usize> = (0..self.table.entries.len()).collect();
        order.sort_by_key(|&index| self.table.entries[index].1.expr_depth);

        for &index in &order {
            let entry = &self.table.entries[index].1;
            if entry.count < 2 {
                continue;
            }
            let (parent_count, children) = (entry.count, entry.children.clone());
            for (child, multiplicity) in children {
                if let Some(child) = self.table.find(&child)? {
                    self.table.entries[child].1.consumed += (parent_count - 1) * multiplicity;
                }
            }
        }

        let mut plan = Plan { insert_before: HashMap::default(), remap: ExprTable::new() };
        let mut counter = 0;
        for (position, &index) in order.iter().enumerate() {
            let entry = &self.table.entries[index].1;
            if entry.count < entry.consumed + 2 {
                continue;
            }
            let insert_at = self.insertion_stmt(entry);
            counter += 1;
            let repr = entry.repr.clone();
            let var = Var::prim(&format!("cse_v{counter}"), repr.prim_dtype()?)?;
            let bind = Bind::new(var.clone(), &repr)?;
            plan.insert_before.entry(ObjectIdentity::of(&insert_at)).or_default().push(bind);
            let key = self.table.entries[index].0.clone();
            let (hash, found) = plan.remap.locate(&key)?;
            match found {
                Some(existing) => plan.remap.entries[existing].1 = var.clone(),
                None => {
                    plan.remap.push(hash, key, var.clone());
                }
            }
            // Deeper entries come later in `order`; replace this expression in theirs.
            let depth = entry.expr_depth;
            for &other in &order[position + 1..] {
                if self.table.entries[other].1.expr_depth <= depth {
                    continue;
                }
                let substituted = substitute(&self.table.entries[other].1.repr, &repr, &var)?;
                self.table.entries[other].1.repr = substituted;
            }
        }
        Ok(plan)
    }
}

#[dispatch(mutate)]
impl Planner {
    // Statements: each sets the current statement, as the C++ `Visit` override does.

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.mutate(self, &op.value)?;
        m.mutate_with(self, &op.var, DefRegionKind::Simple)?;
        Ok(Unchanged)
    }

    /// Operands in the current scope; the body of a launch-thread region in a new one, and of
    /// any other region not at all.
    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.mutate(self, &op.args)?;
        m.mutate(self, &op.attrs)?;
        if LAUNCH_THREAD.is(&op.op) {
            self.scoped(&op, &op.body, m)?;
        }
        Ok(Unchanged)
    }

    fn mutate_if(&mut self, op: If, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.mutate(self, &op.condition)?;
        self.scoped(&op, &op.then_case, m)?;
        if let Some(else_case) = &op.else_case {
            self.scoped(&op, else_case, m)?;
        }
        Ok(Unchanged)
    }

    /// The bounds in the current scope, the body in a new one; the step is not visited.
    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.mutate(self, &op.min)?;
        m.mutate(self, &op.extent)?;
        self.scoped(&op, &op.body, m)?;
        Ok(Unchanged)
    }

    fn mutate_while(&mut self, op: While, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.mutate(self, &op.condition)?;
        self.scoped(&op, &op.body, m)?;
        Ok(Unchanged)
    }

    fn mutate_return(&mut self, op: Return, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.mutate(self, &op.value)?;
        Ok(Unchanged)
    }

    fn mutate_break(&mut self, op: Break) -> Unchanged {
        self.enter(op);
        Unchanged
    }

    fn mutate_continue(&mut self, op: Continue) -> Unchanged {
        self.enter(op);
        Unchanged
    }

    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.mutate(self, &op.dest)?;
        for index in op.indices.iter() {
            m.mutate(self, &index)?;
        }
        m.mutate(self, &op.value)?;
        Ok(Unchanged)
    }

    /// The message parts are constant and not visited.
    fn mutate_assert(&mut self, op: AssertStmt, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.mutate(self, &op.condition)?;
        m.mutate(self, &op.error_kind)?;
        Ok(Unchanged)
    }

    fn mutate_seq(&mut self, op: SeqStmt, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        for stmt in op.seq.iter() {
            m.mutate(self, &stmt)?;
        }
        Ok(Unchanged)
    }

    fn mutate_evaluate(&mut self, op: Evaluate, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.mutate(self, &op.value)?;
        Ok(Unchanged)
    }

    fn mutate_stmt(&mut self, op: Stmt, m: &mut Mutator) -> Result<Unchanged> {
        self.enter(&op);
        m.default_mutate(self, &op)?;
        Ok(Unchanged)
    }

    // Expressions, as `StmtExprVisitor` visits them.

    /// A use is not visited; a definition visits a non-primitive type.
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Unchanged> {
        let kind = m.def_region_kind();
        if kind == DefRegionKind::None || var.ty.as_node::<PrimTypeObj>().is_some() {
            return Ok(Unchanged);
        }
        match kind {
            DefRegionKind::Simple => m.mutate_with(self, &var.ty, DefRegionKind::None)?,
            _ => m.mutate(self, &var.ty)?,
        };
        Ok(Unchanged)
    }

    fn mutate_opaque(&mut self, _op: OpaqueExpr) -> Unchanged {
        Unchanged
    }

    fn mutate_tuple(&mut self, op: Tuple, m: &mut Mutator) -> Result<Unchanged> {
        for field in op.fields.iter() {
            m.mutate(self, &field)?;
        }
        Ok(Unchanged)
    }

    fn mutate_tuple_get_item(&mut self, op: TupleGetItem, m: &mut Mutator) -> Result<Unchanged> {
        m.mutate(self, &op.tuple_value)?;
        Ok(Unchanged)
    }

    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Unchanged> {
        m.mutate(self, &op.source)?;
        for index in op.indices.iter() {
            m.mutate(self, &index)?;
        }
        Ok(Unchanged)
    }

    /// `ExprVisitor`'s: the type, the source and the region.
    fn mutate_region_expr(&mut self, op: TensorRegion, m: &mut Mutator) -> Result<Unchanged> {
        m.mutate(self, &op.ty)?;
        m.mutate(self, &op.source)?;
        m.mutate(self, &op.region)?;
        Ok(Unchanged)
    }

    /// The operator only when it is an opaque expression, then the arguments and type arguments.
    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Unchanged> {
        if op.op.as_node::<OpaqueExprObj>().is_some() {
            m.mutate(self, &op.op)?;
        }
        for arg in op.args.iter() {
            m.mutate(self, &arg)?;
        }
        for ty in op.ty_args.iter() {
            m.mutate(self, &ty)?;
        }
        Ok(Unchanged)
    }

    /// The body may use the let-bound variable, which is undefined at statement level, so
    /// nothing in it is recorded.
    fn mutate_let(&mut self, op: Let, m: &mut Mutator) -> Result<Unchanged> {
        m.mutate(self, &op.value)?;
        self.let_depth += 1;
        let result = m.mutate(self, &op.body);
        self.let_depth -= 1;
        result?;
        Ok(Unchanged)
    }

    fn mutate_ramp(&mut self, op: Ramp, m: &mut Mutator) -> Result<Unchanged> {
        m.mutate(self, &op.base_)?;
        m.mutate(self, &op.stride)?;
        Ok(Unchanged)
    }

    fn mutate_broadcast(&mut self, op: Broadcast, m: &mut Mutator) -> Result<Unchanged> {
        m.mutate(self, &op.value)?;
        Ok(Unchanged)
    }

    fn mutate_shuffle(&mut self, op: Shuffle, m: &mut Mutator) -> Result<Unchanged> {
        for index in op.indices.iter() {
            m.mutate(self, &index)?;
        }
        for vector in op.vectors.iter() {
            m.mutate(self, &vector)?;
        }
        Ok(Unchanged)
    }

    /// Operators, casts and selects record themselves after their operands.
    fn mutate_expr(&mut self, expr: Expr, m: &mut Mutator) -> Result<Unchanged> {
        let Some(operands) = recorded_operands(&expr) else {
            m.default_mutate(self, &expr)?;
            return Ok(Unchanged);
        };
        for operand in &operands {
            m.mutate(self, operand)?;
        }
        self.record(&expr, &operands)?;
        Ok(Unchanged)
    }
}

/// C++ `SubstituteSubexpr`: `body` with every expression equal to `target` replaced, not looking
/// inside a replaced one.
fn substitute(body: &Expr, target: &Expr, replacement: impl Into<Expr>) -> Result<Expr> {
    let mut replacer = Replacer { target: target.clone(), replacement: replacement.into() };
    structural_mutate(body.clone(), &mut replacer)?.try_into()
}

struct Replacer {
    target: Expr,
    replacement: Expr,
}

#[dispatch(mutate)]
impl Replacer {
    fn mutate_expr(&mut self, expr: Expr, m: &mut Mutator) -> Result<Any> {
        if is_prim(&expr) && deep_equal(&expr, &self.target)? {
            return Ok(self.replacement.clone().into());
        }
        m.default_mutate(self, &expr)
    }
}

// ---------------------------------------------------------------------------------------------
// Phase 2: the rewriter

/// C++ `CSERewriter`: replace the planned expressions and insert the planned bindings.
struct Rewriter {
    plan: Plan,
    /// Insertion targets whose bindings were already emitted once.
    materialized: HashSet<ObjectIdentity>,
    /// Inside the body of a region other than a launched thread, which the plan does not cover.
    opaque: bool,
}

impl Rewriter {
    /// Prepend the bindings planned before `original` to its rewritten `visited` form. The same
    /// statement object can occur more than once (unrolling shares subtrees); after the first,
    /// the bindings get fresh variables, so the result stays in SSA form.
    fn insert(&mut self, original: &Stmt, visited: Any) -> Result<Any> {
        let key = ObjectIdentity::of(original);
        let Some(binds) = self.plan.insert_before.get(&key) else { return Ok(visited) };
        let mut visited: Stmt = visited.try_into()?;
        let mut stmts: Vec<Stmt> = Vec::with_capacity(binds.len() + 1);
        if self.materialized.insert(key) {
            stmts.extend(binds.iter().map(Stmt::from));
        } else {
            let mut fresh_vars: HashMap<ObjectIdentity, Var> = HashMap::default();
            for bind in binds {
                let value: Expr = rename(&bind.value, &fresh_vars)?;
                let ty = bind.var.ty.as_node::<PrimTypeObj>().expect("a binding of a PrimExpr");
                let fresh = Var::prim(bind.var.name.as_str(), ty.dtype)?;
                fresh_vars.insert(ObjectIdentity::of(&bind.var), fresh.clone());
                stmts.push(Bind::new(fresh, value)?.into());
            }
            visited = rename(&visited, &fresh_vars)?;
        }
        stmts.push(visited);
        Ok(SeqStmt::new(stmts)?.into())
    }
}

/// `value` with uses of the variables in `fresh_vars` replaced.
fn rename<T>(value: &T, fresh_vars: &HashMap<ObjectIdentity, Var>) -> Result<T>
where
    T: Clone + Into<Any> + TryFrom<Any, Error = tvm_ffi::Error>,
{
    let map = |var: Var, kind: DefRegionKind| -> UnchangedOr<Var> {
        match fresh_vars.get(&ObjectIdentity::of(&var)) {
            Some(fresh) if kind == DefRegionKind::None => UnchangedOr::changed(fresh.clone()),
            _ => UnchangedOr::unchanged(),
        }
    };
    structural_map(value.clone().into(), map, WalkOrder::PreOrder)?.try_into()
}

#[dispatch(mutate)]
impl Rewriter {
    /// A launched thread is rewritten as a whole. Any other region has only its operands
    /// rewritten; its body is visited without the plan.
    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        if self.opaque {
            return m.default_mutate(self, &op);
        }
        let visited = if LAUNCH_THREAD.is(&op.op) {
            m.default_mutate(self, &op)?
        } else {
            let args: Array<Expr> = m.mutate_as(self, &op.args)?;
            let attrs: DictAttrs = m.mutate_as(self, &op.attrs)?;
            let region = op.update(|node| {
                node.args = args;
                node.attrs = attrs;
            });
            self.opaque = true;
            let visited = m.default_mutate(self, &region);
            self.opaque = false;
            visited?
        };
        self.insert(&op.into(), visited)
    }

    fn mutate_stmt(&mut self, op: Stmt, m: &mut Mutator) -> Result<Any> {
        let visited = m.default_mutate(self, &op)?;
        if self.opaque {
            return Ok(visited);
        }
        self.insert(&op, visited)
    }

    /// A planned expression becomes its variable, before its operands are visited.
    fn mutate_expr(&mut self, expr: Expr, m: &mut Mutator) -> Result<Any> {
        if !self.opaque && is_prim(&expr) {
            if let Some(index) = self.plan.remap.find(&expr)? {
                return Ok(self.plan.remap.entries[index].1.clone().into());
            }
        }
        m.default_mutate(self, &expr)
    }
}

pub fn common_subexpr_elim() -> Result<Pass> {
    function_pass("tirx.CommonSubexprElim", 0, |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let plan = Planner::plan(&body)?;
        if plan.insert_before.is_empty() {
            return Ok(Some(func));
        }
        let mut rewriter = Rewriter { plan, materialized: HashSet::default(), opaque: false };
        let body: Stmt = structural_mutate(body, &mut rewriter)?.try_into()?;
        Ok(Some(func.with_body(Some(SeqStmt::from_stmt(body)?))))
    })
}
