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

//! The `ops` operators against the native operations they stand for: each case is built once
//! with operators and once by calling the C++ operator's global with the operand C++ would make.

use tvm::ir::{Expr, Location, PrimType, Var};
use tvm::ops::{ex, float_imm, int_imm, let_, make_const, select, PrimOps};
use tvm::prim::{Not, Sub};
use tvm::tvm_ffi::{DLDataType, DLDataTypeExt, ObjectRefCore, String as FfiString};

mod common;
use common::{global, load_tvm_compiler, structural_equal};

fn dtype(name: &str) -> DLDataType {
    DLDataType::try_from_str(name).unwrap()
}

fn loc() -> Location {
    tvm::ops::unknown_loc().unwrap()
}

fn var(name: &str, ty: &str) -> Var {
    let ty: PrimType = global("ir.PrimType").call_tuple((dtype(ty),)).unwrap().try_into().unwrap();
    global("ir.Var").call_tuple((FfiString::from(name), ty, loc())).unwrap().try_into().unwrap()
}

/// The C++ operator's global applied to two expressions.
fn native(op: &str, a: impl Into<Expr>, b: impl Into<Expr>) -> Expr {
    global(op).call_tuple((a.into(), b.into(), loc())).unwrap().try_into().unwrap()
}

fn imm(ty: &str, value: i64) -> Expr {
    int_imm(dtype(ty), value as i128).unwrap().into()
}

fn fimm(ty: &str, value: f64) -> Expr {
    float_imm(dtype(ty), value).unwrap().into()
}

fn same(built: tvm::ops::Ex, expected: &Expr) -> bool {
    structural_equal(&built.expr().unwrap(), expected)
}

#[test]
fn literals_are_typed_as_in_cpp() {
    load_tvm_compiler();
    let x = var("x", "int32");
    let w = var("w", "int64");
    let v = var("v", "int32x4");
    let f = var("f", "float32");

    // An integer is `MakeConst(other.ty(), v)`, on either side.
    assert!(same(&x + 1, &native("prim._OpAdd", x.clone(), imm("int32", 1))));
    assert!(same(1 + &x, &native("prim._OpAdd", imm("int32", 1), x.clone())));
    assert!(same(&w - 1, &native("prim._OpSub", w.clone(), imm("int64", 1))));
    assert!(same(
        &v * 2,
        &native("prim._OpMul", v.clone(), make_const(dtype("int32x4"), 2).unwrap())
    ));
    assert!(same(&f + 1, &native("prim._OpAdd", f.clone(), fimm("float32", 1.0))));
    // A `double` is a float64 `FloatImm`, a `float` a float32 one, whatever the other side.
    assert!(same(&f + 1.5, &native("prim._OpAdd", f.clone(), fimm("float64", 1.5))));
    assert!(same(&f + 1.5_f32, &native("prim._OpAdd", f.clone(), fimm("float32", 1.5))));
    // Comparisons and the named operations type a literal the same way.
    assert!(same(x.greater_equal(0), &native("prim._OpGE", x.clone(), imm("int32", 0))));
    assert!(same(w.floordiv(8), &native("prim._OpFloorDiv", w.clone(), imm("int64", 8))));
    assert!(same(&x >> 3, &native("prim.right_shift", x.clone(), imm("int32", 3))));
}

#[test]
fn constants_fold_natively() {
    load_tvm_compiler();
    let x = var("x", "int32");
    assert!(same(ex(imm("int32", 2)) + 3, &imm("int32", 5)));
    assert!(same(&x * 1, &Expr::from(x.clone())));
    assert!(same(ex(imm("int32", 7)).truncmod(4), &imm("int32", 3)));
    // Chained: `(x + 0) - 1` folds the first step, then builds a Sub.
    assert!(same(&x + 0 - 1, &native("prim._OpSub", x.clone(), imm("int32", 1))));
    let mut acc = ex(&x);
    acc += 2;
    acc *= 3;
    assert!(same(
        acc,
        &native("prim._OpMul", native("prim._OpAdd", x, imm("int32", 2)), imm("int32", 3))
    ));
}

#[test]
fn unary_operators_follow_neg_and_logical_not() {
    load_tvm_compiler();
    let x = var("x", "int32");
    let p = var("p", "bool");
    // `-x` is `MakeConst(x.ty(), 0) - x`; a constant is negated in place, wrapping at 64 bits.
    assert!(same(-&x, &native("prim._OpSub", imm("int32", 0), x.clone())));
    assert!(same(-ex(imm("int32", 5)), &imm("int32", -5)));
    assert!(same(-ex(imm("int64", i64::MIN)), &imm("int64", i64::MIN)));
    assert!(same(-ex(fimm("float32", 0.0)), &fimm("float32", -0.0)));
    // `!p` is the raw `Not`; a constant folds.
    let not_p = Not::new(p.clone().into(), loc()).unwrap();
    assert!(same(!&p, &not_p.into()));
    assert!(same(!ex(imm("bool", 1)), &imm("bool", 0)));
}

#[test]
fn raw_constructors_and_free_functions() {
    load_tvm_compiler();
    let x = var("x", "int32");
    let r = var("r", "int32");
    // `select` and `let_` are the raw constructors, built from operands under construction.
    let built = let_(&r, x.truncmod(4), select(r.greater_equal(0), &r, &r - 1)).expr().unwrap();
    let cond = native("prim._OpGE", r.clone(), imm("int32", 0));
    let minus = native("prim._OpSub", r.clone(), imm("int32", 1));
    let select: Expr = global("prim.Select")
        .call_tuple((cond, r.clone(), minus, loc()))
        .unwrap()
        .try_into()
        .unwrap();
    let modulo = native("prim._OpTruncMod", x.clone(), imm("int32", 4));
    let expected: Expr =
        global("prim.Let").call_tuple((r, modulo, select, loc())).unwrap().try_into().unwrap();
    assert!(structural_equal(&built, &expected));
    // The raw `Sub` does not fold; the operator does.
    let raw = Sub::new(imm("int32", 3), imm("int32", 1), loc()).unwrap();
    assert!(!structural_equal(&Expr::from(raw), &imm("int32", 2)));
    assert!(same(ex(imm("int32", 3)) - 1, &imm("int32", 2)));
}

#[test]
fn the_first_error_is_kept() {
    load_tvm_compiler();
    let x = var("x", "int32");
    let f = var("f", "float32");
    // Integer division of a float raises natively; later steps pass the error through.
    let Err(error) = (f.truncdiv(&x) + 1).less(&x).expr() else { panic!("expected an error") };
    let direct = global("prim._OpTruncDiv").call_tuple((Expr::from(f), Expr::from(x), loc()));
    let direct = direct.err().expect("the native operation raises");
    assert_eq!(error.kind(), direct.kind());
    assert_eq!(error.message(), direct.message());
    // A literal next to a non-PrimType expression has no dtype to take: an error, not a panic.
    let string: Expr = global("ir.StringImm")
        .call_tuple((FfiString::from("s"), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!((ex(&string) + 1).expr().is_err());
}

#[test]
fn identity_is_kept_when_nothing_folds_away() {
    load_tvm_compiler();
    let x = var("x", "int32");
    // `x + 0` folds back to `x` itself, as in C++.
    assert!((&x + 0).expr().unwrap().same_as(&Expr::from(x)));
}
