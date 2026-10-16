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

//! The operators of `ops` (`+`, `floordiv`, `<`, `&&`, `<<`, `if_then_else`, ...), written in
//! Rust, against the native operators they follow (`prim._OpAdd` and the like): every operator
//! on every pair of a grid of operands, which must give the same expression or an error of the
//! same kind with the same message.

use tvm::ir::{Call, Expr, Location, PrimType, Type};
use tvm::ops::{ex, float_imm, if_then_else, make_const, unknown_loc, Ex, PrimOps};
use tvm::prim::{Broadcast, Ramp};
use tvm::tvm_ffi::{Any, AnyView, BigInt, DLDataType, DLDataTypeExt, Result, String as FfiString};

mod common;
use common::{global, load_tvm_compiler, structural_equal};

fn dtype(name: &str) -> DLDataType {
    DLDataType::try_from_str(name).unwrap()
}

fn loc() -> Location {
    unknown_loc().unwrap()
}

fn prim(name: &str) -> PrimType {
    global("ir.PrimType").call_tuple((dtype(name),)).unwrap().try_into().unwrap()
}

fn var(name: &str, ty: impl Into<Type>) -> Expr {
    let ty: Type = ty.into();
    global("ir.Var").call_tuple((FfiString::from(name), ty, loc())).unwrap().try_into().unwrap()
}

fn prim_var(ty: &str) -> Expr {
    var(&format!("v_{}", ty.replace('x', "_")), prim(ty))
}

fn imm(ty: &str, value: impl Into<BigInt>) -> Expr {
    let value: BigInt = value.into();
    global("ir.IntImm").call_tuple((dtype(ty), value, loc())).unwrap().try_into().unwrap()
}

fn fimm(ty: &str, value: f64) -> Expr {
    float_imm(dtype(ty), value).unwrap().into()
}

fn repr<T>(value: &T) -> std::string::String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let text: FfiString =
        global("ffi.ReprPrint").call_packed(&[AnyView::from(value)]).unwrap().try_into().unwrap();
    text.as_str().to_string()
}

/// The cases where the Rust operator and the native one disagree, and how many were compared.
#[derive(Default)]
struct Outcome {
    compared: usize,
    folded_or_built: usize,
    errors: usize,
    mismatches: Vec<std::string::String>,
}

impl Outcome {
    /// The Rust result and the native one agree: the same expression, or an error of the same
    /// kind with the same message.
    fn check(
        &mut self,
        what: impl FnOnce() -> std::string::String,
        rust: Result<Expr>,
        native: Result<Any>,
    ) {
        self.compared += 1;
        let mismatch = match (rust, native) {
            (Ok(rust), Ok(native)) => {
                self.folded_or_built += 1;
                let native: Expr = native.try_into().unwrap();
                match structural_equal(&rust, &native) {
                    true => None,
                    false => Some(format!("{} vs {}", repr(&rust), repr(&native))),
                }
            }
            (Err(rust), Err(native)) => {
                self.errors += 1;
                match (rust.kind() == native.kind(), rust.message() == native.message()) {
                    (true, true) => None,
                    _ => Some(format!(
                        "{}: {:?} vs {}: {:?}",
                        rust.kind(),
                        rust.message(),
                        native.kind(),
                        native.message()
                    )),
                }
            }
            (Ok(rust), Err(native)) => {
                Some(format!("Rust gave {}, C++ failed with {}", repr(&rust), native.message()))
            }
            (Err(rust), Ok(native)) => {
                let native: Expr = native.try_into().unwrap();
                Some(format!("Rust failed with {}, C++ gave {}", rust.message(), repr(&native)))
            }
        };
        if let Some(mismatch) = mismatch {
            self.mismatches.push(format!("{}: {mismatch}", what()));
        }
    }

    fn assert_agrees(&self, what: &str) {
        let shown: Vec<&std::string::String> = self.mismatches.iter().take(20).collect();
        assert!(
            self.mismatches.is_empty(),
            "{what}: {} of {} cases disagree:\n{}",
            self.mismatches.len(),
            self.compared,
            shown.iter().map(|m| m.as_str()).collect::<Vec<_>>().join("\n")
        );
        println!(
            "{what}: {} cases ({} expressions, {} errors) agree",
            self.compared, self.folded_or_built, self.errors
        );
    }
}

/// Operands of every type the operators treat differently: variables, constants including 0,
/// 1, -1, the limits of each type and values that overflow, vectors, broadcasts, a ramp, the
/// symbolic infinities and `vscale()` calls.
fn operands() -> Vec<Expr> {
    let mut operands: Vec<Expr> = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint32",
        "uint64",
        "bool",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int32x4",
        "int64x4",
        "float32x4",
        "boolx4",
        "int32x8",
    ]
    .into_iter()
    .map(prim_var)
    .collect();
    operands.push(var("w", prim("int32")));
    for (ty, values) in [
        ("int8", vec![0i64, 1, -1, 127, -128, 5]),
        ("int16", vec![0, 1, 32767, -3]),
        ("int32", vec![0, 1, -1, 2, 3, 7, -5, 31, 32, i32::MAX as i64, i32::MIN as i64]),
        ("int64", vec![0, 1, -1, 3, 63, 64, 1 << 40, i64::MAX, i64::MIN]),
        ("uint8", vec![0, 1, 7, 255]),
        ("uint32", vec![0, 1, 7, u32::MAX as i64]),
        ("bool", vec![0, 1]),
    ] {
        for value in values {
            operands.push(imm(ty, value));
        }
    }
    operands.push(imm("uint64", 0i64));
    operands.push(imm("uint64", 5i64));
    operands.push(imm("uint64", BigInt::from(u64::MAX)));
    for (ty, values) in [
        ("float16", vec![0.0, -0.0, 1.0, 2.5, 65504.0]),
        (
            "float32",
            vec![0.0, -0.0, 1.0, 2.5, -2.5, 3.0e38, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
        ),
        ("float64", vec![0.0, 1.0, -4.5, 1e300, f64::INFINITY, f64::NAN]),
        ("bfloat16", vec![0.0, 1.5]),
    ] {
        for value in values {
            operands.push(fimm(ty, value));
        }
    }
    operands.push(Broadcast::new(imm("int32", 7i64), 4).unwrap().into());
    operands.push(Broadcast::new(imm("int32", 0i64), 4).unwrap().into());
    operands.push(Broadcast::new(fimm("float32", 2.5), 4).unwrap().into());
    operands.push(Ramp::new(imm("int32", 0i64), imm("int32", 1i64), 4).unwrap().into());
    operands.push(tvm::sym::pos_inf().unwrap());
    operands.push(tvm::sym::neg_inf().unwrap());
    let vscale = || -> Expr {
        let op = tvm::prim::op::VSCALE.get().unwrap();
        Call::new(prim("int32"), op, Vec::<Expr>::new()).unwrap().into()
    };
    operands.push(vscale());
    operands.push(vscale());
    operands
}

type Binary = fn(&Expr, &Expr) -> Ex;

#[test]
fn binary_operators_match_cpp() {
    load_tvm_compiler();
    let operators: Vec<(&str, Binary)> = vec![
        ("prim._OpAdd", |a, b| a + b),
        ("prim._OpSub", |a, b| a - b),
        ("prim._OpMul", |a, b| a * b),
        ("prim._OpDiv", |a, b| a.div(b)),
        ("prim._OpTruncDiv", |a, b| a.truncdiv(b)),
        ("prim._OpTruncMod", |a, b| a.truncmod(b)),
        ("prim._OpFloorDiv", |a, b| a.floordiv(b)),
        ("prim._OpFloorMod", |a, b| a.floormod(b)),
        ("prim._OpIndexDiv", |a, b| a.indexdiv(b)),
        ("prim._OpIndexMod", |a, b| a.indexmod(b)),
        ("prim._OpCeilDiv", |a, b| a.ceildiv(b)),
        ("prim._OpMin", |a, b| a.min(b)),
        ("prim._OpMax", |a, b| a.max(b)),
        ("prim._OpEQ", |a, b| a.equal(b)),
        ("prim._OpNE", |a, b| a.not_equal(b)),
        ("prim._OpLT", |a, b| a.less(b)),
        ("prim._OpLE", |a, b| a.less_equal(b)),
        ("prim._OpGT", |a, b| a.greater(b)),
        ("prim._OpGE", |a, b| a.greater_equal(b)),
        ("prim._OpAnd", |a, b| a.logical_and(b)),
        ("prim._OpOr", |a, b| a.logical_or(b)),
        ("prim.bitwise_and", |a, b| a & b),
        ("prim.bitwise_or", |a, b| a | b),
        ("prim.bitwise_xor", |a, b| a ^ b),
        ("prim.left_shift", |a, b| a << b),
        ("prim.right_shift", |a, b| a >> b),
    ];
    let operands = operands();
    let mut total = Outcome::default();
    for (name, operator) in operators {
        let native_op = global(name);
        let mut outcome = Outcome::default();
        for a in &operands {
            for b in &operands {
                let native = native_op.call_tuple((a.clone(), b.clone(), loc()));
                let what = || format!("{name}({}, {})", repr(a), repr(b));
                outcome.check(what, operator(a, b).expr(), native);
            }
        }
        outcome.assert_agrees(name);
        total.compared += outcome.compared;
        total.folded_or_built += outcome.folded_or_built;
        total.errors += outcome.errors;
    }
    total.assert_agrees("all binary operators");
}

#[test]
fn unary_and_ternary_operators_match_cpp() {
    load_tvm_compiler();
    let operands = operands();
    let mut outcome = Outcome::default();
    let bitwise_not = global("prim.bitwise_not");
    for a in &operands {
        let native = bitwise_not.call_tuple((a.clone(), loc()));
        outcome.check(|| format!("~{}", repr(a)), a.bitwise_not().expr(), native);
    }
    outcome.assert_agrees("prim.bitwise_not");

    let mut outcome = Outcome::default();
    let conditions: Vec<Expr> = vec![
        prim_var("bool"),
        imm("bool", 0i64),
        imm("bool", 1i64),
        prim_var("boolx4"),
        prim_var("int32"),
        imm("int32", 1i64),
    ];
    let values: Vec<Expr> = operands.iter().step_by(3).cloned().collect();
    let if_then_else_op = global("prim._OpIfThenElse");
    for c in &conditions {
        for t in &values {
            for f in &values {
                let native = if_then_else_op.call_tuple((c.clone(), t.clone(), f.clone(), loc()));
                let what = || format!("if_then_else({}, {}, {})", repr(c), repr(t), repr(f));
                outcome.check(what, if_then_else(c, t, f).expr(), native);
            }
        }
    }
    outcome.assert_agrees("prim._OpIfThenElse");
}

#[test]
fn literals_and_compound_expressions_match_cpp() {
    load_tvm_compiler();
    // A literal is typed after the other side, then the operator folds as for two constants.
    let x = prim_var("int64");
    let add = global("prim._OpAdd");
    let mut outcome = Outcome::default();
    outcome.check(
        || "x + 3".into(),
        (ex(&x) + 3).expr(),
        add.call_tuple((x.clone(), imm("int64", 3i64), loc())),
    );
    outcome.check(
        || "3 + x".into(),
        (3i32 + &x).expr(),
        add.call_tuple((imm("int64", 3i64), x.clone(), loc())),
    );
    let floordiv = global("prim._OpFloorDiv");
    let rust = ((ex(&x) + 0) * 1).floordiv(1);
    outcome.check(
        || "((x + 0) * 1) // 1".into(),
        rust.expr(),
        floordiv.call_tuple((x.clone(), imm("int64", 1i64), loc())),
    );
    // Unary minus folds a constant (a 64-bit one wraps) and is `0 - x` otherwise.
    let sub = global("prim._OpSub");
    for value in
        [imm("int64", i64::MIN), imm("int32", 5i64), fimm("float32", 2.5), prim_var("int32")]
    {
        let zero = make_const(value.prim_dtype().unwrap(), 0).unwrap();
        let native = sub.call_tuple((zero, value.clone(), loc()));
        outcome.check(|| format!("-{}", repr(&value)), (-&value).expr(), native);
    }
    outcome.assert_agrees("literals");
}
