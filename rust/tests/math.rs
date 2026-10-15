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

//! The math functions, `cast` and `match_types` of `ops`, written in Rust, against the native
//! functions they follow: each case is built by both, which must give the same expression or the
//! same error. The intrinsics have no native global; their nodes are checked directly, and `exp`
//! and `log` also through the native `logaddexp`.

use tvm::ir::{Call, Expr, FuncType, Location, Op, PrimType, StringImm, Type};
use tvm::ops::{
    fast_erf_float_expr, float_imm, infinity, int_imm, match_types, unknown_loc, Ex, PrimOps,
};
use tvm::prim::{Add, Broadcast, Cast};
use tvm::tvm_ffi::{
    Any, AnyView, DLDataType, DLDataTypeExt, Function, Result, String as FfiString, TYPE_ERROR,
};

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

fn imm(ty: &str, value: i64) -> Expr {
    int_imm(dtype(ty), value as i128).unwrap().into()
}

fn fimm(ty: &str, value: f64) -> Expr {
    float_imm(dtype(ty), value).unwrap().into()
}

fn ptr(element: &str) -> Type {
    global("ir.PtrType")
        .call_tuple((Type::from(prim(element)), FfiString::from("global")))
        .unwrap()
        .try_into()
        .unwrap()
}

fn repr<T>(value: &T) -> std::string::String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let text: FfiString =
        global("ffi.ReprPrint").call_packed(&[AnyView::from(value)]).unwrap().try_into().unwrap();
    text.as_str().to_string()
}

/// The Rust function and the native one agree: the same expression, or an error of the same
/// kind with the same message.
fn agree(what: &str, rust: Result<Expr>, native: Result<Any>) {
    match (rust, native) {
        (Ok(rust), Ok(native)) => {
            let native: Expr = native.try_into().unwrap();
            assert!(
                structural_equal(&rust, &native),
                "{what}: {} vs {}",
                repr(&rust),
                repr(&native)
            );
        }
        (Err(rust), Err(native)) => {
            assert_eq!(rust.kind(), native.kind(), "{what}: {}", native.message());
            assert_eq!(rust.message(), native.message(), "{what}");
        }
        (Ok(rust), Err(native)) => {
            panic!("{what}: Rust gave {}, C++ {}", repr(&rust), native.message())
        }
        (Err(rust), Ok(native)) => {
            let native: Expr = native.try_into().unwrap();
            panic!("{what}: Rust failed with {}, C++ gave {}", rust.message(), repr(&native))
        }
    }
}

/// Variables and constants of the types the functions treat differently.
fn values() -> Vec<Expr> {
    let mut values: Vec<Expr> = [
        "int32",
        "int64",
        "uint8",
        "uint32",
        "bool",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int32x4",
        "float32x4",
        "bfloat16x4",
    ]
    .into_iter()
    .map(prim_var)
    .collect();
    values.extend([
        imm("int32", -5),
        imm("int32", 7),
        imm("int64", i64::MIN),
        imm("uint32", 7),
        int_imm(dtype("bool"), 1).unwrap().into(),
        fimm("float32", 2.5),
        fimm("float32", -2.5),
        fimm("float32", 3.5),
        fimm("float32", -0.5),
        fimm("float64", 1e300),
        fimm("float16", -1.25),
        fimm("bfloat16", 1.5),
    ]);
    values
}

#[test]
fn unary_functions_match_cpp() {
    load_tvm_compiler();
    type Unary = fn(&Expr) -> Ex;
    let functions: [(&str, Unary); 11] = [
        ("prim.floor", |x| x.floor()),
        ("prim.ceil", |x| x.ceil()),
        ("prim.round", |x| x.round()),
        ("prim.nearbyint", |x| x.nearbyint()),
        ("prim.trunc", |x| x.trunc()),
        ("prim.abs", |x| x.abs()),
        ("prim.isnan", |x| x.isnan()),
        ("prim.isinf", |x| x.isinf()),
        ("prim.isfinite", |x| x.isfinite()),
        ("prim.likely", |x| x.likely()),
        ("prim.assume", |x| x.assume()),
    ];
    for (name, function) in functions {
        for x in values() {
            let native = global(name).call_tuple((x.clone(), loc()));
            agree(&format!("{name}({})", repr(&x)), function(&x).expr(), native);
        }
    }
    // `round` folds a tie to even, as `std::nearbyint` does.
    assert!(structural_equal(&fimm("float32", 2.5).round().expr().unwrap(), &fimm("float32", 2.0)));
}

#[test]
fn infinity_matches_cpp() {
    load_tvm_compiler();
    for ty in ["float16", "float32", "float64", "bfloat16", "int32", "float32x4"] {
        let native = global("prim.infinity").call_tuple((prim(ty), loc()));
        agree(&format!("infinity({ty})"), infinity(dtype(ty)).expr(), native);
    }
}

#[test]
fn cast_matches_cpp() {
    load_tvm_compiler();
    let mut values = values();
    values.push(Broadcast::new(prim_var("float32"), 4).unwrap().into());
    let ramp: Expr = global("prim.Ramp")
        .call_tuple((prim_var("int32"), imm("int32", 1), imm("int32", 4), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    values.push(ramp);
    values.push(prim_var("void"));
    let targets = [
        "int32",
        "int64",
        "uint8",
        "uint32",
        "bool",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int32x4",
        "float32x4",
        "int64x4",
        "float16x4",
        "int32x8",
    ];
    for x in &values {
        for ty in targets {
            let native = global("prim._cast").call_tuple((x.clone(), prim(ty), loc()));
            agree(&format!("cast({}, {ty})", repr(x)), x.cast(dtype(ty)).expr(), native);
        }
    }
}

#[test]
fn type_matching_matches_the_native_operators() {
    load_tvm_compiler();
    let pairs = [
        ("int32", "int64"),
        ("int64", "int32"),
        ("int32", "uint32"),
        ("uint32", "int32"),
        ("uint8", "int32"),
        ("uint32", "int64"),
        ("bool", "int32"),
        ("int32", "bool"),
        ("float16", "float32"),
        ("int32", "float32"),
        ("float32", "int64"),
        ("bfloat16", "float32"),
        ("bfloat16", "int32"),
        ("bool", "float16"),
        ("int32x4", "int32"),
        ("float32", "float32x4"),
        ("int32", "float32x4"),
        ("float32x4", "int64"),
        ("int32x4", "int32x8"),
    ];
    for (a, b) in pairs {
        let (x, y) = (var("x", prim(a)), var("y", prim(b)));
        // The raw `Add` of the matched sides is what C++ `x + y` builds when nothing folds.
        let rust = match_types(x.clone(), y.clone()).and_then(|(x, y)| Ok(Add::new(x, y)?.into()));
        let native = global("prim._OpAdd").call_tuple((x, y, loc()));
        agree(&format!("{a} + {b}"), rust, native);
    }
    // A constant side is cast by folding.
    let (x, three) = (prim_var("float32"), imm("int32", 3));
    let rust = match_types(x.clone(), three.clone()).and_then(|(x, y)| Ok(Add::new(x, y)?.into()));
    agree("float32 + 3", rust, global("prim._OpAdd").call_tuple((x, three, loc())));
}

#[test]
fn pow_fmod_and_logaddexp_match_cpp() {
    load_tvm_compiler();
    let pairs: Vec<(Expr, Expr)> = vec![
        (prim_var("float32"), var("w", prim("float32"))),
        (prim_var("float16"), prim_var("float32")),
        (prim_var("float32"), imm("int32", 2)),
        (prim_var("float32x4"), prim_var("float32")),
        (prim_var("float32"), fimm("float16", 0.5)),
        (prim_var("int32"), var("j", prim("int32"))),
    ];
    for (x, y) in &pairs {
        let what = format!("({}, {})", repr(x), repr(y));
        let native = global("prim._OpPow").call_tuple((x.clone(), y.clone(), loc()));
        agree(&format!("pow{what}"), x.pow(y).expr(), native);
        let native = global("tirx._OpLogAddExp").call_tuple((x.clone(), y.clone(), loc()));
        agree(&format!("logaddexp{what}"), x.logaddexp(y).expr(), native);
    }
    // `fmod` has no global: the sides are matched and the call is typed as them.
    let (h, f) = (prim_var("float16"), prim_var("float32"));
    let widened: Expr = Cast::new(&h, prim("float32")).unwrap().into();
    let expected =
        Call::new(prim("float32"), Op::get("prim.fmod").unwrap(), vec![widened, f.clone()]);
    assert!(structural_equal(&h.fmod(&f).expr().unwrap(), &expected.unwrap()));
    let error = prim_var("int32").fmod(imm("int32", 2)).expr().err().unwrap();
    assert!(error.message().contains("fmod only applies to float"), "{}", error.message());
}

#[test]
fn reinterpret_matches_cpp() {
    load_tvm_compiler();
    let text: Expr = StringImm::new("data").unwrap().into();
    let pointer = var("p", ptr("float32"));
    let function: Type = FuncType::new(vec![], prim("int32")).unwrap().into();
    let cases: Vec<(Type, Expr)> = vec![
        (prim("int32").into(), prim_var("float32")),
        (prim("float32").into(), prim_var("float32")),
        (prim("int64").into(), prim_var("float32")),
        (prim("float32x4").into(), prim_var("int32x4")),
        (prim("int16x2").into(), prim_var("float32")),
        (ptr("int8"), prim_var("int64")),
        (ptr("int8"), prim_var("int32")),
        (prim("int64").into(), pointer.clone()),
        (prim("int32").into(), pointer),
        (ptr("int8"), text.clone()),
        (prim("int32").into(), text),
        (function, prim_var("int32")),
    ];
    for (ty, x) in cases {
        let native = global("tirx.reinterpret").call_tuple((ty.clone(), x.clone(), loc()));
        let what = format!("reinterpret({}, {})", repr(&ty), repr(&x));
        agree(&what, x.reinterpret(ty).expr(), native);
    }
}

#[test]
fn intrinsics_are_typed_calls() {
    load_tvm_compiler();
    let call = |ty: &str, op: &str, args: Vec<Expr>| -> Expr {
        Call::new(prim(ty), Op::get(op).unwrap(), args).unwrap().into()
    };
    let cast = |value: Expr, ty: &str| -> Expr { Cast::new(value, prim(ty)).unwrap().into() };
    let same = |built: Ex, expected: &Expr| structural_equal(&built.expr().unwrap(), expected);

    let x = prim_var("float32");
    assert!(same(x.exp(), &call("float32", "prim.exp", vec![x.clone()])));
    assert!(same(x.tan(), &call("float32", "prim.tan", vec![x.clone()])));
    // Only the trigonometric and hyperbolic ones reject an integer.
    let i = prim_var("int32");
    assert!(same(i.exp(), &call("int32", "prim.exp", vec![i.clone()])));
    let error = i.tan().expr().err().unwrap();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert!(error
        .message()
        .ends_with("prim.tan only supports floating-point inputs, but got T.int32"));
    // A bfloat16 argument is computed in float32 and cast back, keeping its lanes.
    for (ty, wide) in [("bfloat16", "float32"), ("bfloat16x4", "float32x4")] {
        let b = prim_var(ty);
        for (built, op) in
            [(b.sqrt(), "prim.sqrt"), (b.log2(), "prim.log2"), (b.tanh(), "prim.tanh")]
        {
            let expected = cast(call(wide, op, vec![cast(b.clone(), wide)]), ty);
            assert!(same(built, &expected), "{op}({ty})");
        }
    }
    // A binary intrinsic is typed as its first argument, its arguments unmatched.
    let h = prim_var("float16");
    assert!(same(x.atan2(&h), &call("float32", "prim.atan2", vec![x.clone(), h.clone()])));
    assert!(same(x.copysign(&h), &call("float32", "prim.copysign", vec![x.clone(), h])));
}

#[test]
fn fast_erf_matches_the_native_lowering() {
    load_tvm_compiler();
    // Metal lowers `erf` to C++ `fast_erf_float_expr`.
    let erf = Op::get("prim.erf").unwrap();
    let rules = Op::attr_map::<Function>("metal.FLowerIntrinsic").unwrap().unwrap();
    let rule = rules.get(&erf).unwrap().unwrap();
    for (ty, bits) in [("float32", 32), ("float16", 16)] {
        let x = prim_var(ty);
        let call: Expr = Call::new(prim(ty), erf.clone(), vec![x.clone()]).unwrap().into();
        let native: Expr = rule.call_tuple((call,)).unwrap().try_into().unwrap();
        let rust = fast_erf_float_expr(&x, bits).expr().unwrap();
        assert!(structural_equal(&rust, &native), "{ty}: {} vs {}", repr(&rust), repr(&native));
    }
}
