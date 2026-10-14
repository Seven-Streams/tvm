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

//! `tirx.StmtSimplify` (`src/tirx/transform/stmt_simplify.cc`): simplify every PrimExpr under
//! the facts of its enclosing loops, bindings, and branches, and drop branches and self-stores
//! the analyzer proves dead.

use std::collections::HashMap;

use tvm_ffi::{
    dispatch, Mutator, structural_map, Any, Array, DefRegionKind, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, Unchanged, UnchangedOr, WalkOrder,
};

use super::analyzer::{binary, bind_range, AnalyzerBase, AnalyzerScope, AnalyzerState};
use super::side_effect::{side_effect, PURE};
use super::util::{
    function_pass, int32, make_const, prim_dtype, seq_stmt_from, unknown_loc, MutatorExt,
};
use crate::ir::{
    Bind, Call, CallObj, Evaluate, Expr, For, If, IntImmObj, Op, PrimTypeObj, SeqStmt, Stmt,
    StringImm, TensorLoadObj, TensorStore, Var, VarObj,
};
use crate::prim::{AddObj, MulObj, SubObj};
use crate::sym::{Analyzer, ConstIntBound};
use crate::tirx::{Function, TensorType, TensorTypeObj};
use crate::transform::Pass;

const POS_INF: i64 = i64::MAX;
const NEG_INF: i64 = -POS_INF;

struct StmtSimplifier {
    base: AnalyzerBase,
    /// Pure `Bind` values, substituted when proving a branch condition.
    non_inlined_bindings: HashMap<ObjectIdentity, Expr>,
    alloc_tensor: Op,
    decl_tensor: Op,
}

impl AnalyzerState for StmtSimplifier {
    fn base(&mut self) -> &mut AnalyzerBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = AnalyzerScope)]
impl StmtSimplifier {
    /// Tensor types are left as written.
    fn mutate_tensor_type(&mut self, _ty: TensorType) -> Unchanged {
        Unchanged
    }

    /// The loop variable lies in `[min, min + extent)` throughout the loop.
    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        let analyzer = self.base.analyzer.clone();
        bind_range(&analyzer, &op.loop_var, &op.min, &op.extent)?;
        let loop_var: Expr = op.loop_var.clone().into();
        let end = binary("prim._OpAdd", &op.min, &op.extent)?;
        let _lower = analyzer.assume(&binary("prim._OpGE", &loop_var, &op.min)?)?;
        let _upper = analyzer.assume(&binary("prim._OpLT", &loop_var, &end)?)?;
        m.default_mutate(self, &op)
    }

    /// A pure bound value is known to the analyzer and to later branch conditions.
    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if call.op.same_as(&self.alloc_tensor) {
                return Ok(op.into());
            }
            if call.op.same_as(&self.decl_tensor) {
                let data = call.args.get(0)?;
                let simplified: Expr = m.mutate_as(self, &data)?;
                if simplified.same_as(&data) {
                    return Ok(op.into());
                }
                let args: Array<Expr> =
                    std::iter::once(simplified).chain(call.args.iter().skip(1)).collect();
                let value = Call::new(
                    call.loc.clone(),
                    call.ty.clone(),
                    call.op.clone(),
                    args,
                    call.attrs.clone(),
                    call.ty_args.clone(),
                );
                return Ok(op.with_value(value.into()).into());
            }
        }
        if op.value.ty.as_node::<PrimTypeObj>().is_none() {
            return m.default_mutate(self, &op);
        }
        let value: Expr = m.mutate_as(self, &op.value)?;
        if side_effect(&value)? <= PURE {
            self.base.analyzer.bind(&op.var, &value)?;
            self.non_inlined_bindings.insert(ObjectIdentity::of(&op.var), value.clone());
        }
        Ok(op.update(|node| node.value = value).into())
    }

    /// A branch whose condition is provable keeps only the arm that runs.
    fn mutate_if(&mut self, op: If, m: &mut Mutator) -> Result<Any> {
        match (self.prove(&op.condition)?, &op.else_case) {
            (Some(true), _) => m.mutate(self, &op.then_case),
            (Some(false), Some(else_case)) => m.mutate(self, else_case),
            (Some(false), None) => Ok(Evaluate::new(unknown_loc()?, int32(0)?.into()).into()),
            (None, _) => m.default_mutate(self, &op),
        }
    }

    /// A store of the value just loaded from the same element is dropped.
    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let store: TensorStore = m.default_mutate_as(self, &op)?;
        if stores_loaded_value(&store)? {
            return Ok(Evaluate::new(unknown_loc()?, int32(0)?.into()).into());
        }
        Ok(store.into())
    }

    /// Every other PrimExpr is simplified as a whole; a variable definition keeps the default.
    fn mutate_expr(&mut self, expr: Expr, m: &mut Mutator) -> Result<Any> {
        let defines_var =
            m.def_region_kind() != DefRegionKind::None && expr.as_node::<VarObj>().is_some();
        if defines_var || expr.ty.as_node::<PrimTypeObj>().is_none() {
            return m.default_mutate(self, &expr);
        }
        let simplified = self.base.analyzer.simplify(&expr, 2)?;
        Ok(if simplified.same_as(&expr) { expr } else { simplified }.into())
    }
}

impl StmtSimplifier {
    /// `ProveCondition`: substitute the known bindings, simplify, and read a constant.
    fn prove(&self, condition: &Expr) -> Result<Option<bool>> {
        let bindings = &self.non_inlined_bindings;
        let substituted: Expr = structural_map(
            condition.clone(),
            |var: Var| match bindings.get(&ObjectIdentity::of(&var)) {
                Some(value) => UnchangedOr::changed(Any::from(value.clone())),
                None => UnchangedOr::unchanged(),
            },
            WalkOrder::PreOrder,
        )?
        .try_into()?;
        let simplified = self.base.analyzer.simplify(&substituted, 2)?;
        Ok(simplified.as_node::<IntImmObj>().map(|imm| imm.value != tvm_ffi::BigInt::from(0i64)))
    }

    /// `MarkTensorParamShapes`: every shape of a tensor parameter is non-negative.
    fn mark_tensor_param_shapes(&self, func: &Function) -> Result<()> {
        for param in func.params.iter() {
            if let Some(tensor) = param.ty.as_node::<TensorTypeObj>() {
                for shape in tensor.shape.iter() {
                    mark_global_non_neg_value(&self.base.analyzer, &shape)?;
                }
            }
        }
        Ok(())
    }
}

/// `prim::ExprDeepEqual` over arrays.
fn all_deep_equal(lhs: &Array<Expr>, rhs: &Array<Expr>) -> Result<bool> {
    if lhs.len() != rhs.len() {
        return Ok(false);
    }
    for (x, y) in lhs.iter().zip(rhs.iter()) {
        if !deep_equal(&x, &y)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn deep_equal(lhs: &Expr, rhs: &Expr) -> Result<bool> {
    tvm_ffi::cached_global_func!("prim.expr_deep_equal")
        .call_tuple((lhs.clone(), rhs.clone()))?
        .try_into()
}

fn stores_loaded_value(store: &TensorStore) -> Result<bool> {
    let Some(load) = store.value.as_node::<TensorLoadObj>() else { return Ok(false) };
    if !load.source.same_as(&store.dest) {
        return Ok(false);
    }
    let tensor =
        store.dest.ty.as_node::<TensorTypeObj>().expect("a store's destination is a tensor");
    // The same tensor on both sides, so its layout matches; `ExprDeepEqual` of each field with
    // itself is what C++ checks.
    Ok(all_deep_equal(&load.indices, &store.indices)?
        && deep_equal(&tensor.elem_offset, &tensor.elem_offset)?
        && all_deep_equal(&tensor.shape, &tensor.shape)?
        && all_deep_equal(&tensor.strides, &tensor.strides)?)
}

/// `AnalyzerObj::MarkGlobalNonNegValue`, which has no global: for `value = symbol * scale +
/// offset` with a positive constant scale, bound `symbol >= -offset`.
fn mark_global_non_neg_value(analyzer: &Analyzer, value: &Expr) -> Result<()> {
    let dtype = prim_dtype(value)?;
    let constant = |expr: &Expr| {
        expr.as_node::<IntImmObj>().map(|imm| imm.value.to_i64().unwrap_or(0) as i128)
    };
    let mut offset: i128 = 0;
    let mut symbol_scale = make_const(dtype, 0)?;
    for (term, sign) in sum_terms(value, 1) {
        match constant(&term) {
            Some(c) => offset += c * sign as i128,
            None if sign > 0 => symbol_scale = binary("prim._OpAdd", &symbol_scale, &term)?,
            None => symbol_scale = binary("prim._OpSub", &symbol_scale, &term)?,
        }
    }
    let mut scale: i128 = 1;
    let mut symbol = make_const(dtype, 1)?;
    for factor in product_factors(&symbol_scale) {
        match constant(&factor) {
            Some(c) => scale = scale.saturating_mul(c),
            None => symbol = binary("prim._OpMul", &symbol, &factor)?,
        }
    }
    if scale <= 0 {
        return Ok(());
    }
    let Some(var_ty) = symbol.as_node::<VarObj>().and_then(|var| var.ty.as_node::<PrimTypeObj>())
    else {
        return Ok(());
    };
    let index_type =
        var_ty.dtype.code == 0 && matches!(var_ty.dtype.bits, 32 | 64) && var_ty.dtype.lanes == 1;
    if !index_type || -offset <= NEG_INF as i128 || -offset >= POS_INF as i128 {
        return Ok(());
    }
    let var: Var = symbol.try_cast().expect("checked above");
    analyzer.update_const_int_bound(&var, &ConstIntBound::new(-offset as i64, POS_INF), true)
}

/// `UnpackSum`: the terms of a tree of `Add` and `Sub`, with their signs.
fn sum_terms(value: &Expr, sign: i32) -> Vec<(Expr, i32)> {
    if let Some(node) = value.as_node::<AddObj>() {
        return [sum_terms(&node.a, sign), sum_terms(&node.b, sign)].concat();
    }
    if let Some(node) = value.as_node::<SubObj>() {
        return [sum_terms(&node.a, sign), sum_terms(&node.b, -sign)].concat();
    }
    vec![(value.clone(), sign)]
}

/// `UnpackReduction<MulNode>`: the factors of a tree of `Mul`.
fn product_factors(value: &Expr) -> Vec<Expr> {
    match value.as_node::<MulObj>() {
        Some(node) => [product_factors(&node.a), product_factors(&node.b)].concat(),
        None => vec![value.clone()],
    }
}

/// `Run`'s first step: a `Bind` of a string is removed and its uses take the string itself.
#[derive(Default)]
struct StringInliner {
    strings: HashMap<ObjectIdentity, StringImm>,
}

#[dispatch(map)]
impl StringInliner {
    fn map_bind(&mut self, bind: Bind) -> Result<UnchangedOr<Any>> {
        let Ok(value) = bind.value.clone().try_cast::<StringImm>() else {
            return Ok(UnchangedOr::unchanged());
        };
        self.strings.insert(ObjectIdentity::of(&bind.var), value);
        Ok(UnchangedOr::changed(Evaluate::new(unknown_loc()?, int32(0)?.into()).into()))
    }

    fn map_var(&mut self, var: Var) -> UnchangedOr<Any> {
        match self.strings.get(&ObjectIdentity::of(&var)) {
            Some(value) => UnchangedOr::changed(value.clone().into()),
            None => UnchangedOr::unchanged(),
        }
    }
}

fn inline_string_bindings(body: SeqStmt) -> Result<SeqStmt> {
    structural_map(body, &mut StringInliner::default(), WalkOrder::PostOrder)?.try_into()
}

pub fn stmt_simplify() -> Result<Pass> {
    function_pass("tirx.StmtSimplify", 0, |func, _module, _context| {
        let mut simplifier = StmtSimplifier {
            base: AnalyzerBase::new(Analyzer::new()?)?,
            non_inlined_bindings: HashMap::new(),
            alloc_tensor: Op::get("tirx.alloc_tensor")?,
            decl_tensor: Op::get("tirx.decl_tensor")?,
        };
        simplifier.mark_tensor_param_shapes(&func)?;
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let body = inline_string_bindings(body)?;
        let body: Stmt = tvm_ffi::structural_mutate(body, &mut simplifier)?.try_into()?;
        Ok(Some(func.with_body(Some(seq_stmt_from(body)?))))
    })
}
