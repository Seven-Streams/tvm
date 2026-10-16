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

//! Expression patterns match and evaluate as C++ `sym::Pattern` does: variables bind on their
//! first occurrence and compare on the next ones, integer literals take the other operand's type,
//! and evaluation folds constants or builds the raw node.

use tvm::ir::{Expr, Var};
use tvm::ops::pattern::{broadcast, ramp, PVar, Pattern, PatternOps};
use tvm::ops::{int_imm, PrimOps};
use tvm::prim::{Add, Broadcast, FloorDiv, FloorMod, Mul, Not, Ramp, EQ, LT};
use tvm::tvm_ffi::{DLDataType, DLDataTypeExt, ObjectRefCore};

mod common;
use common::{load_tvm_compiler, structural_equal};

fn dtype(name: &str) -> DLDataType {
    DLDataType::try_from_str(name).unwrap()
}

fn var(name: &str, ty: &str) -> Expr {
    Var::prim(name, dtype(ty)).unwrap().into()
}

fn imm(ty: &str, value: i64) -> Expr {
    int_imm(dtype(ty), value as i128).unwrap().into()
}

fn expr(node: impl Into<Expr>) -> Expr {
    node.into()
}

#[test]
fn variables_bind_once_and_compare_after() {
    load_tvm_compiler();
    let (a, b) = (var("a", "int32"), var("b", "int32"));
    let (x, y) = (PVar::expr(), PVar::expr());

    let pattern = &x + &y;
    assert!(pattern.matches(&expr(Add::new(&a, &b).unwrap())));
    assert!(x.get().same_as(&a) && y.get().same_as(&b));

    // A variable that occurs twice matches equal parts: the same object, or deep-equal ones.
    let twice = &x + &x;
    assert!(twice.matches(&expr(Add::new(&a, &a).unwrap())));
    assert!(!twice.matches(&expr(Add::new(&a, &b).unwrap())));
    let (left, right) = ((&a + 1).expr().unwrap(), (&a + 1).expr().unwrap());
    assert!(!left.same_as(&right));
    assert!(twice.matches(&expr(Add::new(&left, &right).unwrap())));

    // Each match starts unbound.
    assert!(pattern.matches(&expr(Add::new(&b, &a).unwrap())));
    assert!(x.get().same_as(&b) && y.get().same_as(&a));

    // The node type must match.
    assert!(!pattern.matches(&expr(Mul::new(&a, &b).unwrap())));
    assert!(!pattern.matches(&a));
}

#[test]
fn integer_variables_match_constants() {
    load_tvm_compiler();
    let a = var("a", "int32");
    let (x, c) = (PVar::expr(), PVar::int());
    let scaled = &x * &c;
    assert!(scaled.matches(&expr(Mul::new(&a, 4).unwrap())));
    assert_eq!(c.get().value.to_i64(), Some(4));
    assert!(!scaled.matches(&expr(Mul::new(&a, &a).unwrap())));

    // A constant variable twice compares the values.
    let same = &c + &c;
    assert!(same.matches(&expr(Add::new(imm("int32", 3), imm("int32", 3)).unwrap())));
    assert!(!same.matches(&expr(Add::new(imm("int32", 3), imm("int32", 4)).unwrap())));
}

#[test]
fn literals_match_by_value_and_take_the_operand_type() {
    load_tvm_compiler();
    let (a, b) = (var("a", "int64"), var("b", "int64"));
    let (x, y) = (PVar::expr(), PVar::expr());

    let is_zero_mod = x.floormod(&y).equal(0);
    let zero = imm("int64", 0);
    let modulo = expr(FloorMod::new(&a, &b).unwrap());
    assert!(is_zero_mod.matches(&expr(EQ::new(&modulo, &zero).unwrap())));
    assert!(!is_zero_mod.matches(&expr(EQ::new(&modulo, imm("int64", 1)).unwrap())));

    // `x * 2` evaluates with an int64 constant, the type of `x`.
    assert!(x.matches(&a));
    let doubled = (&x * 2).eval().unwrap();
    assert!(structural_equal(&doubled, &expr(Mul::new(&a, imm("int64", 2)).unwrap())));
    let less = x.floordiv(&y).less(1);
    assert!(is_zero_mod.matches(&expr(EQ::new(&modulo, &zero).unwrap())));
    let expected = LT::new(FloorDiv::new(&a, &b).unwrap(), imm("int64", 1)).unwrap();
    assert!(structural_equal(&less.eval().unwrap(), &expr(expected)));
}

#[test]
fn evaluation_folds_constants_or_builds_the_raw_node() {
    load_tvm_compiler();
    let (a, b) = (var("a", "int32"), var("b", "int32"));
    let (x, y, c1, c2) = (PVar::expr(), PVar::expr(), PVar::int(), PVar::int());

    // floordiv(floormod(x, c1), c2) with constants that fold.
    let pattern = x.floormod(&c1).floordiv(&c2);
    let input = FloorDiv::new(FloorMod::new(&a, 12).unwrap(), 4).unwrap();
    assert!(pattern.matches(&expr(input)));
    let folded = c1.floordiv(&c2).eval().unwrap();
    assert!(structural_equal(&folded, &imm("int32", 3)));
    let rewritten = x.floordiv(&c2).floormod(c1.floordiv(&c2)).eval().unwrap();
    let expected = FloorMod::new(FloorDiv::new(&a, 4).unwrap(), 3).unwrap();
    assert!(structural_equal(&rewritten, &expr(expected)));

    // Identities fold as `TryConstFold` folds them; anything else is the raw node.
    assert!((&x + &y).matches(&expr(Add::new(&a, &b).unwrap())));
    assert!((&x * 1).eval().unwrap().same_as(&a));
    let raw = x.floordiv(&y).eval().unwrap();
    assert!(structural_equal(&raw, &expr(FloorDiv::new(&a, &b).unwrap())));
    let compared = c1.less(&c2).eval().unwrap();
    assert!(structural_equal(&compared, &imm("bool", 0)));
}

#[test]
fn not_broadcast_and_ramp() {
    load_tvm_compiler();
    let (a, b) = (var("a", "int32"), var("b", "int32"));
    let (x, y, lanes) = (PVar::expr(), PVar::expr(), PVar::expr());

    let not_less = !x.less(&y);
    let input = Not::new(LT::new(&a, &b).unwrap()).unwrap();
    assert!(not_less.matches(&expr(input)));
    assert!(structural_equal(&y.less_equal(&x).eval().unwrap(), &b.less_equal(&a).expr().unwrap()));
    let less = expr(LT::new(&a, &b).unwrap());
    let double_not = expr(Not::new(Not::new(&less).unwrap()).unwrap());
    assert!((!(!&x)).matches(&double_not));
    assert!(x.get().same_as(&less));
    assert!(structural_equal(&(!(!&x)).eval().unwrap(), &double_not));

    // `x * broadcast(c, lanes)` with a constant `c`.
    let c = PVar::int();
    let scaled = &x * broadcast(&c, &lanes);
    let vector = expr(Broadcast::new(&a, 4).unwrap());
    let input = Mul::new(&vector, Broadcast::new(3, 4).unwrap()).unwrap();
    assert!(scaled.matches(&expr(input)));
    assert!(x.get().same_as(&vector));
    assert_eq!(c.get().value.to_i64(), Some(3));
    assert!(structural_equal(&lanes.get(), &imm("int32", 4)));
    let input = Mul::new(&vector, Broadcast::new(&a, 4).unwrap()).unwrap();
    assert!(!scaled.matches(&expr(input)), "the broadcast value is not a constant");

    // A ramp with literal stride and lanes, typed like the base.
    let base = PVar::expr();
    let unit = ramp(&base, 1, 4);
    assert!(unit.matches(&expr(Ramp::new(&a, 1, 4).unwrap())));
    assert!(!unit.matches(&expr(Ramp::new(&a, 2, 4).unwrap())));
    assert!(structural_equal(&unit.eval().unwrap(), &expr(Ramp::new(&a, 1, 4).unwrap())));
    assert!(base.get().same_as(&a));
}
