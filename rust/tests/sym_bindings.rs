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

//! Smoke tests for the stubgen-generated `sym` bindings against `libtvm_compiler`.

use tvm::ir::{Expr, ExprObj, IntImm, IntSet, IntSetObj, Location, PrimType, Range, Var};
use tvm::sym::*;
use tvm::tvm_ffi::{
    Array, BigInt, DLDataType, DLDataTypeExt, Map, Object, ObjectRefCast, String as FfiString,
};

mod common;
use common::{assert_type_contracts, global, load_tvm_compiler};

#[test]
fn generated_types_match_the_registry() {
    load_tvm_compiler();
    assert_type_contracts!(true;
        ConstIntBoundObj: Object, IterMapResultObj: Object, IterMarkObj: Object,
        ModularSetObj: Object, RewriteSimplifierStatsObj: Object,
    );
    // The analyzer holds its sub-analyzers unreflected; the expression hierarchies and
    // `ir.IntSet` register no layout of their own, so their subclasses are opaque too.
    assert_type_contracts!(false;
        AnalyzerObj: Object, IntervalSetObj: IntSetObj,
        CanonicalExprObj: ExprObj, SplitExprObj: CanonicalExprObj, SumExprObj: CanonicalExprObj,
        IterMapExprObj: ExprObj, IterSplitExprObj: IterMapExprObj, IterSumExprObj: IterMapExprObj,
    );
}

struct Fixture {
    loc: Location,
    analyzer: Analyzer,
    x: Var,
}

fn int32(value: i64) -> Expr {
    let dtype = DLDataType::try_from_str("int32").unwrap();
    let imm: IntImm = global("ir.IntImm")
        .call_tuple((dtype, BigInt::from(value), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    imm.into()
}

fn loc() -> Location {
    global("ir.UnknownLoc").call_packed(&[]).unwrap().try_into().unwrap()
}

fn op(name: &str, a: &Expr, b: &Expr) -> Expr {
    global(name).call_tuple((a.clone(), b.clone(), loc())).unwrap().try_into().unwrap()
}

/// An analyzer with `x` bound to `[0, 16)`.
fn fixture() -> Fixture {
    load_tvm_compiler();
    let loc = loc();
    let int32_ty: PrimType = global("ir.PrimType")
        .call_tuple((DLDataType::try_from_str("int32").unwrap(),))
        .unwrap()
        .try_into()
        .unwrap();
    let x: Var = global("ir.Var")
        .call_tuple((FfiString::from("x"), int32_ty, loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    // A typed `Analyzer` handle goes through `call_tuple` like any other object.
    let analyzer: Analyzer = global("sym.Analyzer").call_tuple(()).unwrap().try_into().unwrap();
    let range = Range::new(int32(0), int32(16), loc.clone());
    global("sym.AnalyzerBind").call_tuple((analyzer.clone(), x.clone(), range)).unwrap();
    Fixture { loc, analyzer, x }
}

#[test]
fn analyzer_results_are_read_as_fields() {
    let Fixture { analyzer, x, .. } = fixture();
    let x_expr: Expr = x.clone().into();

    let bound: ConstIntBound = global("sym.AnalyzerConstIntBound")
        .call_tuple((analyzer.clone(), op("prim._OpAdd", &x_expr, &int32(1))))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!((bound.min_value, bound.max_value), (1, 16));

    let modular: ModularSet = global("sym.AnalyzerModularSet")
        .call_tuple((
            analyzer.clone(),
            op("prim._OpAdd", &op("prim._OpMul", &x_expr, &int32(4)), &int32(2)),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!((modular.coeff, modular.base_), (4, 2));

    let stats: RewriteSimplifierStats = global("sym.AnalyzerGetRewriteSimplifyStats")
        .call_tuple((analyzer,))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(stats.nodes_visited >= 0);
}

#[test]
fn rust_allocated_bounds_reach_the_native_analyzer() {
    let Fixture { analyzer, x, .. } = fixture();
    // A complete binding allocates on the Rust side; the native analyzer reads its fields.
    let bound = ConstIntBound::new(3, 7);
    global("sym.AnalyzerConstIntBoundUpdate")
        .call_tuple((analyzer.clone(), x.clone(), bound, true))
        .unwrap();
    let read: ConstIntBound = global("sym.AnalyzerConstIntBound")
        .call_tuple((analyzer, Expr::from(x)))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!((read.min_value, read.max_value), (3, 7));
}

#[test]
fn interval_sets_are_read_through_accessors() {
    let Fixture { analyzer, x, loc } = fixture();
    let y: Var = global("ir.Var")
        .call_tuple((FfiString::from("y"), x.ty.clone(), loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    // With an explicit domain map the evaluation ignores the analyzer's own bindings.
    let interval = |min: i64, max: i64| -> IntSet {
        global("sym.IntervalSet").call_tuple((int32(min), int32(max))).unwrap().try_into().unwrap()
    };
    let dom_map: Map<Var, IntSet> =
        [(x.clone(), interval(0, 15)), (y.clone(), interval(2, 5))].into_iter().collect();
    let set: IntSet = global("sym.AnalyzerIntSet")
        .call_tuple((analyzer, op("prim._OpAdd", &x.into(), &y.into()), dom_map))
        .unwrap()
        .try_into()
        .unwrap();
    let interval: IntervalSet = set.try_cast().unwrap();
    let as_int = |expr: Expr| expr.try_cast::<IntImm>().unwrap().value.to_i64().unwrap();
    assert_eq!(as_int(interval.min_value().unwrap()), 2);
    assert_eq!(as_int(interval.max_value().unwrap()), 20);
}

fn mark_is(mark: &IterMark, var: &Var, extent: i64) -> bool {
    let source: Var = mark.source.clone().try_cast().unwrap();
    let extent_imm: IntImm = mark.extent.clone().try_cast().unwrap();
    tvm::tvm_ffi::ObjectIdentity::of(&source) == tvm::tvm_ffi::ObjectIdentity::of(var)
        && extent_imm.value.to_i64() == Some(extent)
}

#[test]
fn iter_maps_decompose_into_marks_and_splits() {
    let Fixture { analyzer, x, loc } = fixture();
    let x_expr: Expr = x.clone().into();
    let indices = Array::new(vec![
        op("prim._OpFloorDiv", &x_expr, &int32(4)),
        op("prim._OpFloorMod", &x_expr, &int32(4)),
    ]);
    let iters: Map<Var, Range> =
        [(x.clone(), Range::new(int32(0), int32(16), loc))].into_iter().collect();
    let result: IterMapResult = global("sym.DetectIterMap")
        .call_tuple((indices, iters, int32(1), 0i32, false, analyzer))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(
        result.errors.is_empty(),
        "{:?}",
        result.errors.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );
    assert_eq!(result.indices.len(), 2);
    let as_int = |expr: Expr| expr.try_cast::<IntImm>().unwrap().value.to_i64().unwrap();
    // Each index is one split of a mark of extent 4, whose source is itself a sum over the
    // mark of x on [0, 16): x // 4 takes lower factor 4 of it, x % 4 lower factor 1.
    let only_split = |sum: &IterSumExpr| -> IterSplitExpr {
        let splits = sum.args().unwrap();
        assert_eq!(splits.len(), 1);
        splits.get(0).unwrap()
    };
    let mut lower_factors = Vec::new();
    for sum in result.indices.iter() {
        assert_eq!(as_int(sum.base_().unwrap()), 0);
        let outer = only_split(&sum);
        let mark: IterMark = outer.source().unwrap();
        assert_eq!(as_int(mark.extent.clone()), 4);
        let inner_sum: IterSumExpr = mark.source.clone().try_cast().unwrap();
        let inner = only_split(&inner_sum);
        let x_mark: IterMark = inner.source().unwrap();
        assert!(mark_is(&x_mark, &x, 16));
        assert_eq!(as_int(inner.extent().unwrap()), 4);
        lower_factors.push(as_int(inner.lower_factor().unwrap()));
    }
    assert_eq!(lower_factors, vec![4, 1]);
}
