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

//! The constructors of `ops` written in Rust, against the native constructors they follow: each
//! case is built by both, which must give the same node or an error of the same kind with the
//! same message.

use std::sync::{Mutex, MutexGuard};

use tvm::ir::{
    AssertStmt, Attrs, Bind, Call, DictAttrs, Evaluate, Expr, FloatImm, For, FuncType, GlobalVar,
    If, IntImm, LambdaExpr, Location, MissingType, Op, OpDef, PrimType, PtrType, Range, RegionStmt,
    Return, SeqStmt, Stmt, StringImm, TensorLoad, TensorStore, Tuple, TupleType, Type,
    UniqueNameSupply, Var,
};
use tvm::ops::{float_imm, int_imm, max_value, min_value, unknown_loc};
use tvm::prim::{
    Add, And, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, Broadcast, Cast, Div, FloorDiv,
    FloorMod, LShift, Let, Max, Min, Mod, Mul, Not, Or, RShift, Ramp, Select, Shuffle, Sub, EQ, GE,
    GT, LE, LT, NE,
};
use tvm::runtime::MemoryScope;
use tvm::tirx::{ExecScope, Function, TensorType};
use tvm::tvm_ffi::{
    Any, AnyCompatible, AnyView, Array, BigInt, DLDataType, DLDataTypeExt, Function as FfiFunction,
    Map, ObjectRefCore, Result, String as FfiString,
};

mod common;
use common::{global, load_tvm_compiler};

/// One test at a time: `packed_hooks` registers operators, which the registry does not
/// synchronize with the readers in the other tests.
fn setup() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    let guard = SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    load_tvm_compiler();
    guard
}

fn dtype(name: &str) -> DLDataType {
    DLDataType::try_from_str(name).unwrap()
}

fn loc() -> Location {
    unknown_loc().unwrap()
}

fn prim(name: &str) -> PrimType {
    global("ir.PrimType").call_tuple((dtype(name),)).unwrap().try_into().unwrap()
}

fn native_var(name: &str, ty: impl Into<Type>) -> Var {
    let ty: Type = ty.into();
    global("ir.Var").call_tuple((FfiString::from(name), ty, loc())).unwrap().try_into().unwrap()
}

fn prim_var(ty: &str) -> Expr {
    native_var(&format!("v_{}", ty.replace('x', "_")), prim(ty)).into()
}

fn imm(ty: &str, value: i64) -> Expr {
    int_imm(dtype(ty), value as i128).unwrap().into()
}

fn scope(name: &str) -> MemoryScope {
    MemoryScope::get(name).unwrap()
}

fn native_tensor_type(shape: &[i64], element: &str) -> TensorType {
    let shape: Array<Expr> = shape.iter().map(|&extent| imm("int32", extent)).collect();
    let (strides, layout) = (Array::<Expr>::new(vec![]), None::<tvm::tirx::Layout>);
    global("tirx.TensorType")
        .call_packed(&[
            AnyView::from(&shape),
            AnyView::from(&prim(element)),
            AnyView::from(&scope("global")),
            AnyView::from(&strides),
            AnyView::new(),
            AnyView::from(&0i32),
            AnyView::from(&0i32),
            AnyView::from(&layout),
            AnyView::from(&loc()),
        ])
        .unwrap()
        .try_into()
        .unwrap()
}

/// `ffi.StructuralEqual` with free variables mapped, so fresh variables of one shape are equal.
fn equal<L: AnyCompatible, R: AnyCompatible>(lhs: &L, rhs: &R) -> bool {
    let equal = global("ffi.StructuralEqual")
        .call_packed(&[
            AnyView::from(lhs),
            AnyView::from(rhs),
            AnyView::from(&true),
            AnyView::from(&false),
        ])
        .unwrap();
    bool::try_from(equal).unwrap()
}

fn repr<T>(value: &T) -> std::string::String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let text: FfiString =
        global("ffi.ReprPrint").call_packed(&[AnyView::from(value)]).unwrap().try_into().unwrap();
    text.as_str().to_string()
}

/// The Rust constructor and the native one agree: the same node, or an error of the same kind
/// with the same message.
fn agree<T>(what: &str, rust: Result<T>, native: Result<Any>)
where
    T: AnyCompatible + TryFrom<Any>,
    for<'a> AnyView<'a>: From<&'a T>,
    <T as TryFrom<Any>>::Error: std::fmt::Debug,
{
    match (rust, native) {
        (Ok(rust), Ok(native)) => {
            let native: T = native.try_into().unwrap();
            assert!(equal(&rust, &native), "{what}: {} vs {}", repr(&rust), repr(&native));
        }
        (Err(rust), Err(native)) => {
            assert_eq!(rust.kind(), native.kind(), "{what}: {}", native.message());
            assert_eq!(rust.message(), native.message(), "{what}");
        }
        (Ok(rust), Err(native)) => {
            panic!("{what}: Rust gave {}, C++ failed with {}", repr(&rust), native.message())
        }
        (Err(rust), Ok(native)) => {
            panic!("{what}: Rust failed with {}, C++ gave {}", rust.message(), repr::<Any>(&native))
        }
    }
}

/// Operands of the types the binary constructors treat differently.
fn operands() -> Vec<Expr> {
    let mut operands: Vec<Expr> =
        ["int32", "int64", "uint8", "bool", "float32", "int32x4", "boolx4"]
            .into_iter()
            .map(prim_var)
            .collect();
    operands.push(imm("int32", 3));
    operands.push(float_imm(dtype("float32"), 2.5).unwrap().into());
    operands
}

macro_rules! check_binary {
    ($($node:ident),* $(,)?) => {$({
        let name = concat!("prim.", stringify!($node));
        let operands = operands();
        for a in &operands {
            for b in &operands {
                let what = format!("{name}({}, {})", repr(a), repr(b));
                let rust = $node::new_full(a.clone(), b.clone(), loc());
                let native = global(name).call_tuple((a.clone(), b.clone(), loc()));
                agree(&what, rust, native);
            }
        }
    })*};
}

#[test]
fn binary_nodes() {
    let _serial = setup();
    check_binary!(
        Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max, EQ, NE, LT, LE, GT, GE, LShift,
        RShift, BitwiseAnd, BitwiseOr, BitwiseXor, And, Or,
    );
}

#[test]
fn unary_nodes() {
    let _serial = setup();
    for a in &operands() {
        agree(
            &format!("Not({})", repr(a)),
            Not::new_full(a.clone(), loc()),
            global("prim.Not").call_tuple((a.clone(), loc())),
        );
        agree(
            &format!("BitwiseNot({})", repr(a)),
            BitwiseNot::new_full(a.clone(), loc()),
            global("prim.BitwiseNot").call_tuple((a.clone(), loc())),
        );
    }
}

#[test]
fn shuffle() {
    let _serial = setup();
    let (v4, w4, f4) = (prim_var("int32x4"), prim_var("int32x4"), prim_var("float32x4"));
    let scalar = prim_var("int32");
    let index = |i: i64| imm("int32", i);
    let cases: Vec<(&str, Vec<Expr>, Vec<Expr>)> = vec![
        ("two vectors", vec![v4.clone(), w4.clone()], vec![index(0), index(5), index(7)]),
        ("one lane", vec![v4.clone()], vec![index(2)]),
        ("a scalar", vec![scalar.clone(), v4.clone()], vec![index(0), index(4)]),
        ("all lanes", vec![v4.clone(), scalar], (0..5).map(index).collect()),
        ("too many lanes", vec![v4.clone()], (0..5).map(index).collect()),
        ("mixed elements", vec![v4.clone(), f4], vec![index(0)]),
        ("no vectors", vec![], vec![index(0)]),
        ("no indices", vec![v4.clone()], vec![]),
    ];
    for (what, vectors, indices) in cases {
        let rust = Shuffle::new_full(vectors.clone(), indices.clone(), loc());
        let (vectors, indices): (Array<Expr>, Array<Expr>) =
            (vectors.into_iter().collect(), indices.into_iter().collect());
        let native = global("prim.Shuffle").call_tuple((vectors, indices, loc()));
        agree(what, rust, native);
    }

    let native = |vectors: Vec<Expr>, indices: Vec<i64>| -> Expr {
        let vectors: Array<Expr> = vectors.into_iter().collect();
        let indices: Array<Expr> = indices.into_iter().map(index).collect();
        global("prim.Shuffle").call_tuple((vectors, indices, loc())).unwrap().try_into().unwrap()
    };
    let concat = Shuffle::concat([v4.clone(), w4.clone()]).unwrap();
    assert!(equal(&concat, &native(vec![v4.clone(), w4.clone()], (0..8).collect())));
    assert!(Shuffle::concat([v4.clone()]).unwrap().same_as(&v4), "one vector is itself");
    let element = Shuffle::extract_element(v4.clone(), 3).unwrap();
    assert!(equal(&element, &native(vec![v4], vec![3])));
}

#[test]
fn variables() {
    let _serial = setup();
    let int32 = prim("int32");
    for (name, ty) in
        [("x", Type::from(int32.clone())), ("t", native_tensor_type(&[4], "float32").into())]
    {
        let rust = Var::new(name, ty.clone()).unwrap();
        let native = native_var(name, ty.clone());
        assert_eq!(rust.name.as_str(), native.name.as_str());
        assert!(rust.ty.same_as(&native.ty));
        assert!(equal(&rust, &native));
    }
    let prim_var = Var::prim("p", dtype("int64")).unwrap();
    assert!(equal(&prim_var, &native_var("p", prim("int64"))));

    let global_var = GlobalVar::new("main").unwrap();
    let native: GlobalVar =
        global("ir.GlobalVar").call_tuple((FfiString::from("main"),)).unwrap().try_into().unwrap();
    assert_eq!(global_var.name_hint.as_str(), native.name_hint.as_str());
    assert!(global_var.ty.same_as(&native.ty), "both are typed MissingType()");

    let missing: Type = global("ir.MissingType").call_packed(&[]).unwrap().try_into().unwrap();
    assert!(MissingType::new().unwrap().same_as(&missing), "MissingType() is a singleton");
}

#[test]
fn types() {
    let _serial = setup();
    let missing: Type = MissingType::new().unwrap().into();
    for (element, scope_name) in [
        (Type::from(prim("float32")), "global"),
        (prim("int8").into(), "shared"),
        (native_tensor_type(&[2], "int32").into(), "local"),
        (missing, "global"),
    ] {
        let what = format!("PtrType({}, {scope_name})", repr(&element));
        let rust = PtrType::new(element.clone(), scope(scope_name));
        let native = global("ir.PtrType").call_tuple((element, FfiString::from(scope_name)));
        agree(&what, rust, native);
    }
    let void: PtrType = global("ir.PtrType")
        .call_tuple((prim("void"), FfiString::from("global")))
        .unwrap()
        .try_into()
        .unwrap();
    let rust_void = PtrType::void().unwrap();
    assert!(equal(&rust_void, &void));
    assert!(rust_void.scope.same_as(&MemoryScope::global().unwrap()));
    assert!(MemoryScope::get("no.such.scope").is_err());

    let fields: Vec<Type> = vec![prim("int32").into(), PtrType::void().unwrap().into()];
    for fields in [fields, vec![]] {
        let array: Array<Type> = fields.iter().cloned().collect();
        agree(
            "TupleType",
            TupleType::new(fields),
            global("ir.TupleType").call_tuple((array, loc())),
        );
    }
}

#[test]
fn tensor_types() {
    let _serial = setup();
    let shape = || vec![imm("int32", 4), imm("int32", 8)];
    let rust = TensorType::new(shape(), dtype("float16"), scope("shared")).unwrap();
    let native: TensorType = global("tirx.TensorType")
        .call_packed(&[
            AnyView::from(&shape().into_iter().collect::<Array<Expr>>()),
            AnyView::from(&prim("float16")),
            AnyView::from(&scope("shared")),
            AnyView::from(&Array::<Expr>::new(vec![])),
            AnyView::new(),
            AnyView::from(&0i32),
            AnyView::from(&0i32),
            AnyView::new(),
            AnyView::from(&loc()),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert!(equal(&rust, &native), "{} vs {}", repr(&rust), repr(&native));
    assert_eq!((rust.data_alignment, rust.offset_factor), (64, 1));

    let (strides, offset) = (vec![imm("int32", 8), imm("int32", 1)], prim_var("int32"));
    for (data_alignment, offset_factor) in [(16, 4), (-1, 0), (0, 2)] {
        let rust = TensorType::new_full(
            shape(),
            prim("int8"),
            scope("local"),
            strides.clone(),
            Some(offset.clone()),
            data_alignment,
            offset_factor,
            None,
            loc(),
        )
        .unwrap();
        let native: TensorType = global("tirx.TensorType")
            .call_packed(&[
                AnyView::from(&shape().into_iter().collect::<Array<Expr>>()),
                AnyView::from(&prim("int8")),
                AnyView::from(&scope("local")),
                AnyView::from(&strides.iter().cloned().collect::<Array<Expr>>()),
                AnyView::from(&offset),
                AnyView::from(&data_alignment),
                AnyView::from(&offset_factor),
                AnyView::new(),
                AnyView::from(&loc()),
            ])
            .unwrap()
            .try_into()
            .unwrap();
        assert!(equal(&rust, &native), "({data_alignment}, {offset_factor})");
    }

    let tensor = native_var("buf", rust.clone());
    let native_ptr: Type = global("ir.PtrType")
        .call_tuple((Type::from(rust.dtype.clone()), rust.scope.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(equal(&rust.data_ptr_type().unwrap(), &native_ptr));
    let native_data: Expr =
        global("tirx.TensorData").call_tuple((tensor.clone(),)).unwrap().try_into().unwrap();
    assert!(equal(&tensor.tensor_data().unwrap(), &native_data));
    assert!(native_var("x", prim("int32")).tensor_data().is_err());
}

#[test]
fn expressions_and_statements() {
    let _serial = setup();
    let (x, y) = (prim_var("int32"), prim_var("float32"));
    let untyped: Expr = GlobalVar::new("f").unwrap().into();
    for fields in [vec![x.clone(), y.clone()], vec![x.clone(), untyped], vec![]] {
        let array: Array<Expr> = fields.iter().cloned().collect();
        agree(
            "Tuple",
            Tuple::new_full(fields, loc()),
            global("ir.Tuple").call_tuple((array, loc())),
        );
    }

    let (a, b) = (native_var("a", prim("int32")), native_var("b", prim("int32")));
    let body: Expr = Add::new(a.clone(), b.clone()).unwrap().into();
    let vars = Array::new(vec![a.clone(), b.clone()]);
    agree(
        "LambdaExpr",
        LambdaExpr::new(&vars, body.clone()),
        global("ir.LambdaExpr").call_tuple((vars, body)),
    );

    agree("Return(0)", Return::new(0), global("ir.Return").call_tuple((imm("int32", 0), loc())));
    agree("Return(y)", Return::new(y.clone()), global("ir.Return").call_tuple((y.clone(), loc())));

    let tensor: Expr = native_var("buf", native_tensor_type(&[16], "int32")).into();
    let i = imm("int32", 3);
    let cases: Vec<(&str, Expr, Vec<Expr>, Expr)> = vec![
        ("a store", tensor.clone(), vec![i.clone()], x.clone()),
        ("a float value", tensor.clone(), vec![i.clone()], y.clone()),
        ("two indices", tensor.clone(), vec![i.clone(), i.clone()], x.clone()),
        ("a scalar destination", x.clone(), vec![i.clone()], x.clone()),
    ];
    for (what, dest, indices, value) in cases {
        let rust = TensorStore::new_full(dest.clone(), indices.clone(), value.clone(), loc());
        let indices: Array<Expr> = indices.into_iter().collect();
        let native = global("ir.TensorStore").call_tuple((dest, indices, value, loc()));
        agree(what, rust, native);
    }
}

#[test]
fn native_objects() {
    let _serial = setup();
    let what = |name: &str| format!("ExecScope(cuda, {name})");
    for name in ["kernel", "warp", "no_such_scope"] {
        let rust = ExecScope::new("cuda", name);
        let native =
            global("tirx.ExecScope").call_tuple((FfiString::from("cuda"), FfiString::from(name)));
        match (rust, native) {
            (Ok(rust), Ok(native)) => {
                let native: ExecScope = native.try_into().unwrap();
                assert_eq!(rust.name.as_str(), native.name.as_str(), "{}", what(name));
            }
            (Err(rust), Err(native)) => assert_eq!(rust.message(), native.message()),
            _ => panic!("{}: one side failed", what(name)),
        }
    }
    UniqueNameSupply::new().unwrap();
}

/// Dtypes the immediates, limits and casts treat differently; names this build cannot parse are
/// left out.
fn dtypes(names: &[&str]) -> Vec<DLDataType> {
    names.iter().filter_map(|name| DLDataType::try_from_str(name).ok()).collect()
}

#[test]
fn immediates() {
    let _serial = setup();
    let int_types = dtypes(&[
        "int8", "int16", "int32", "int64", "int1", "bool", "uint1", "uint8", "uint32", "uint64",
        "int128", "uint128", "float32", "int32x4", "handle",
    ]);
    let values: Vec<i128> = vec![
        0,
        1,
        -1,
        2,
        127,
        128,
        -128,
        -129,
        255,
        256,
        65535,
        (1 << 31) - 1,
        1 << 31,
        -(1 << 31) - 1,
        i64::MAX as i128,
        i64::MAX as i128 + 1,
        i64::MIN as i128,
        i64::MIN as i128 - 1,
        u64::MAX as i128,
        u64::MAX as i128 + 1,
        i128::MAX,
    ];
    for &dtype in &int_types {
        for &value in &values {
            let what = format!("IntImm({}, {value})", dtype.to_string().as_str());
            let native = global("ir.IntImm").call_tuple((dtype, BigInt::from(value), loc()));
            agree::<IntImm>(&what, int_imm(dtype, value), native);
        }
    }

    let float_types = dtypes(&[
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float6_e2m3fn",
        "float6_e3m2fn",
        "float4_e2m1fn",
        "int32",
        "float32x4",
    ]);
    let values = [
        0.0,
        -0.0,
        0.1,
        1.5,
        -1.5,
        6.0,
        7.0,
        -7.0,
        7.5,
        8.0,
        28.0,
        29.0,
        -29.0,
        30.0,
        31.0,
        32.0,
        -32.0,
        448.0,
        449.0,
        -449.0,
        57344.0,
        60000.0,
        65504.0,
        65505.0,
        -65505.0,
        1234567.0,
        1e-7,
        1e38,
        3.5e38,
        4e38,
        -4e38,
        1e300,
        -1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    for &dtype in &float_types {
        for value in values {
            let what = format!("FloatImm({}, {value})", dtype.to_string().as_str());
            let native = global("ir.FloatImm").call_tuple((dtype, value, loc()));
            match (float_imm(dtype, value), native) {
                (Ok(rust), Ok(native)) if value.is_nan() => {
                    let native: FloatImm = native.try_into().unwrap();
                    assert!(rust.value.is_nan() && native.value.is_nan(), "{what}");
                    assert!(equal(&rust.ty, &native.ty), "{what}");
                }
                (rust, native) => agree(&what, rust, native),
            }
        }
    }

    let native = global("ir.StringImm").call_tuple((FfiString::from("text"), loc()));
    agree("StringImm", StringImm::new("text"), native);
}

#[test]
fn limits() {
    let _serial = setup();
    let types = dtypes(&[
        "int8",
        "int16",
        "int32",
        "int64",
        "int128",
        "uint1",
        "uint8",
        "uint32",
        "uint64",
        "uint128",
        "bool",
        "float16",
        "float32",
        "float64",
        "float8",
        "bfloat16",
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float6_e2m3fn",
        "float6_e3m2fn",
        "float4_e2m1fn",
        "handle",
        "int32x4",
        "int32xvscalex4",
    ]);
    for dtype in types {
        let name = dtype.to_string();
        for (which, rust, global_name) in [
            ("max_value", max_value(dtype), "prim.max_value"),
            ("min_value", min_value(dtype), "prim.min_value"),
        ] {
            let what = format!("{which}({})", name.as_str());
            let native = global("ir.PrimType")
                .call_tuple((dtype,))
                .and_then(|ty| global(global_name).call_tuple((PrimType::try_from(ty)?, loc())));
            agree(&what, rust, native);
        }
    }
}

#[test]
fn types_and_ranges() {
    let _serial = setup();
    let args: Vec<Type> = vec![prim("int32").into(), prim("float32").into()];
    let native = global("ir.FuncType")
        .call_tuple((args.iter().cloned().collect::<Array<Type>>(), Type::from(prim("bool"))));
    agree("FuncType", FuncType::new(args, prim("bool")), native);

    let (i, n) = (prim_var("int32"), prim_var("int64"));
    let cases: Vec<(&str, Expr, Expr)> = vec![
        ("from 0", imm("int32", 0), i.clone()),
        ("from an int64 0", imm("int64", 0), n.clone()),
        ("from a constant", imm("int32", 2), imm("int32", 10)),
        ("from a variable", i.clone(), imm("int32", 10)),
        ("of two types", imm("int32", 2), n.clone()),
    ];
    for (what, begin, end) in cases {
        let rust = Range::new(begin.clone(), end.clone());
        let native = global("ir.Range").call_tuple((begin.clone(), Some(end.clone()), loc()));
        agree(&format!("Range {what}"), rust, native);
        let rust = Range::from_min_extent(begin.clone(), end.clone());
        let native = global("ir.Range_from_min_extent").call_tuple((begin, end, loc()));
        agree(&format!("Range::FromMinExtent {what}"), rust, native);
    }
}

/// `Call(None, op, args, attrs, ty_args)` by both, and both type re-inferences and validations.
fn check_call(what: &str, op: Expr, args: Vec<Expr>, attrs: Option<Attrs>, ty_args: Vec<Type>) {
    let (args, ty_args): (Array<Expr>, Array<Type>) =
        (args.into_iter().collect(), ty_args.into_iter().collect());
    let rust =
        Call::new_full(None, op.clone(), args.clone(), attrs.clone(), ty_args.clone(), loc());
    let native = global("ir.Call").call_tuple((
        None::<Type>,
        op.clone(),
        args.clone(),
        attrs.clone(),
        ty_args.clone(),
        loc(),
    ));
    agree(what, rust, native);
    let unchecked = Call::new_full(Some(prim("int8").into()), op, args, attrs, ty_args, loc());
    let call = unchecked.unwrap();
    let native = global("ir.reinfer_type").call_tuple((call.clone(),));
    agree(&format!("{what}: ReinferType"), call.reinfer_type(), native);
    let rust = call.validate();
    let native = global("ir.CallValidate").call_tuple((call.clone(),));
    match (rust, native) {
        (Ok(()), Ok(_)) => {}
        (Err(rust), Err(native)) => {
            assert_eq!(rust.kind(), native.kind(), "{what}: Validate");
            assert_eq!(rust.message(), native.message(), "{what}: Validate");
        }
        (rust, native) => panic!("{what}: Validate {:?} vs {:?}", rust.is_ok(), native.is_ok()),
    }
}

#[test]
fn calls() {
    let _serial = setup();
    let op = |name: &str| -> Expr { Op::get(name).unwrap().into() };
    let (x, i) = (prim_var("float32"), prim_var("int32"));
    let tensor: Expr = native_var("buf", native_tensor_type(&[4], "float32")).into();
    let tensor_ty: Type = native_tensor_type(&[8], "int8").into();
    check_call(
        "a fixed return type",
        op("tirx.ignore_loop_partition"),
        vec![i.clone()],
        None,
        vec![],
    );
    check_call("a fixed void type", op("tirx.kernel_replace_point"), vec![], None, vec![]);
    check_call("an inferred pointer", op("tirx.address_of"), vec![x.clone()], None, vec![]);
    check_call(
        "an inferred tensor",
        op("tirx.decl_tensor"),
        vec![i.clone()],
        None,
        vec![tensor_ty.clone()],
    );
    check_call("an inference that fails", op("tirx.decl_tensor"), vec![i.clone()], None, vec![]);
    check_call("a tensor's data", op("tirx.tensor_data_ptr"), vec![tensor.clone()], None, vec![]);
    check_call("no inference", op("tirx.call_extern"), vec![x.clone()], None, vec![]);
    check_call("an invalid call", op("tirx.get_active_lane_mask"), vec![x.clone()], None, vec![]);
    let attrs: Attrs = DictAttrs::new(Map::new()).into();
    check_call("unexpected attrs", op("tirx.address_of"), vec![x.clone()], Some(attrs), vec![]);

    let func_type = FuncType::new(vec![prim("int32").into()], prim("int64")).unwrap();
    let callee: Expr = GlobalVar::from_complete_fields(loc(), func_type.into(), "f".into()).into();
    check_call("a function", callee, vec![i.clone()], None, vec![]);
    check_call("a variable", i.clone(), vec![], None, vec![]);

    let rust = Call::new(prim("int16"), op("tirx.address_of"), [x.clone()]);
    let native = global("ir.Call").call_tuple((
        Some(Type::from(prim("int16"))),
        op("tirx.address_of"),
        Array::new(vec![x]),
        None::<Attrs>,
        Array::<Type>::new(vec![]),
        loc(),
    ));
    agree("a given type", rust, native);

    for (what, source, indices) in
        [("a tensor", tensor.clone(), vec![i.clone()]), ("a scalar", i.clone(), vec![i.clone()])]
    {
        let rust = TensorLoad::new_full(source.clone(), indices.clone(), loc());
        let indices: Array<Expr> = indices.into_iter().collect();
        let native = global("ir.TensorLoad").call_tuple((source, indices, loc()));
        agree(&format!("TensorLoad of {what}"), rust, native);
    }
}

/// `vscale() * factor`, the lanes of a scalable vector.
fn scalable_lanes(factor: i64) -> Expr {
    let vscale = Call::new(prim("int32"), Op::get("prim.vscale").unwrap(), []).unwrap();
    Mul::new(vscale, imm("int32", factor)).unwrap().into()
}

#[test]
fn vectors_selects_and_casts() {
    let _serial = setup();
    let (b, i, f) = (prim_var("bool"), prim_var("int32"), prim_var("float32"));
    let (b4, i4, i8x) = (prim_var("boolx4"), prim_var("int32x4"), prim_var("int8"));
    let selects: Vec<(&str, Expr, Expr, Expr)> = vec![
        ("a select", b.clone(), i.clone(), i.clone()),
        ("an integer condition", i.clone(), i.clone(), i.clone()),
        ("a vector condition", b4.clone(), i4.clone(), i4.clone()),
        ("a vector condition on scalars", b4.clone(), i.clone(), i.clone()),
        ("a scalar condition on vectors", b.clone(), i4.clone(), i4.clone()),
        ("values of two types", b.clone(), i.clone(), f.clone()),
    ];
    for (what, c, t, e) in selects {
        let rust = Select::new_full(c.clone(), t.clone(), e.clone(), loc());
        agree(what, rust, global("prim.Select").call_tuple((c, t, e, loc())));
    }

    let var = native_var("v", prim("int32"));
    for (what, value) in [("a let", i.clone()), ("a value of another type", f.clone())] {
        let rust = Let::new_full(var.clone(), value.clone(), f.clone(), loc());
        let native = global("prim.Let").call_tuple((var.clone(), value, f.clone(), loc()));
        agree(what, rust, native);
    }

    for (what, value, ty) in [
        ("a cast", i.clone(), prim("float32")),
        ("a vector cast", i4.clone(), prim("float32x4")),
        ("a cast of other lanes", i.clone(), prim("float32x4")),
    ] {
        let rust = Cast::new_full(value.clone(), ty.clone(), loc());
        agree(what, rust, global("prim.Cast").call_tuple((value, ty, loc())));
    }

    let lanes_cases: Vec<(&str, Expr)> = vec![
        ("4 lanes", imm("int32", 4)),
        ("int64 lanes", imm("int64", 8)),
        ("1 lane", imm("int32", 1)),
        ("0 lanes", imm("int32", 0)),
        ("scalable lanes", scalable_lanes(4)),
        ("lanes of a variable", i.clone()),
    ];
    for (what, lanes) in &lanes_cases {
        for (value_what, value) in [("", i.clone()), (" of a vector", i4.clone())] {
            let rust = Broadcast::new_full(value.clone(), lanes.clone(), loc());
            let native = global("prim.Broadcast").call_tuple((value, lanes.clone(), loc()));
            agree(&format!("Broadcast over {what}{value_what}"), rust, native);
        }
        for (base_what, base, stride) in [
            ("", i.clone(), imm("int32", 2)),
            (" with a narrower stride", i.clone(), i8x.clone()),
            (" with a float stride", i.clone(), f.clone()),
            (" of a vector", i4.clone(), imm("int32", 1)),
        ] {
            let rust = Ramp::new_full(base.clone(), stride.clone(), lanes.clone(), loc());
            let native = global("prim.Ramp").call_tuple((base, stride, lanes.clone(), loc()));
            agree(&format!("Ramp over {what}{base_what}"), rust, native);
        }
    }
}

#[test]
fn sequences() {
    let _serial = setup();
    let evaluate = |value: Expr| -> Stmt { Evaluate::new(value).unwrap().into() };
    let (a, b) = (evaluate(prim_var("int32")), evaluate(imm("int32", 1)));
    let nested: Stmt = SeqStmt::new_full(vec![a.clone(), b.clone()], loc()).unwrap().into();
    let cases: Vec<(&str, Vec<Stmt>)> = vec![
        ("a sequence", vec![a.clone(), b.clone()]),
        ("a nested sequence", vec![a.clone(), nested.clone(), b.clone()]),
        ("no-ops", vec![evaluate(imm("int32", 0)), a.clone(), evaluate(imm("int64", 0))]),
        ("only a no-op", vec![evaluate(imm("int8", 0))]),
        ("a false condition", vec![evaluate(imm("bool", 0))]),
        ("nothing", vec![]),
    ];
    for (what, seq) in cases {
        let rust = SeqStmt::new_full(seq.clone(), loc());
        let array: Array<Stmt> = seq.into_iter().collect();
        let native = global("ir.SeqStmt").call_tuple((array, Some(loc())));
        agree(what, rust, native);
    }
    for (what, stmt) in [("a statement", a.clone()), ("a no-op", evaluate(imm("int32", 0)))] {
        let rust = SeqStmt::from_stmt(stmt.clone());
        let native = global("ir.SeqStmt").call_tuple((stmt, None::<Location>));
        agree(&format!("SeqStmt of {what}"), rust, native);
    }
    let sequence = SeqStmt::from_stmt(nested.clone()).unwrap();
    assert!(sequence.same_as(&nested), "a sequence is itself");
}

#[test]
fn statements() {
    let _serial = setup();
    let (b, i, f) = (prim_var("bool"), prim_var("int32"), prim_var("float32"));
    let tensor: Expr = native_var("buf", native_tensor_type(&[4], "float32")).into();
    for (what, value) in [("a value", i.clone()), ("a tensor variable", tensor.clone())] {
        let rust = Evaluate::new_full(value.clone(), loc());
        agree(&format!("Evaluate {what}"), rust, global("ir.Evaluate").call_tuple((value, loc())));
    }

    let var = native_var("v", prim("int32"));
    for (what, value) in [("a binding", i.clone()), ("a value of another type", f.clone())] {
        let rust = Bind::new_full(var.clone(), value.clone(), loc());
        agree(what, rust, global("ir.Bind").call_tuple((var.clone(), value, loc())));
    }

    let kind = StringImm::new("RuntimeError").unwrap();
    let parts = Array::new(vec![StringImm::new("message").unwrap()]);
    for (what, condition) in [("an assertion", b.clone()), ("an integer assertion", i.clone())] {
        let rust = AssertStmt::new_full(condition.clone(), kind.clone(), parts.clone(), loc());
        let native =
            global("ir.AssertStmt").call_tuple((condition, kind.clone(), parts.clone(), loc()));
        agree(what, rust, native);
    }

    let body = SeqStmt::from_stmt(Evaluate::new(i.clone()).unwrap()).unwrap();
    let rust = If::new_full(b.clone(), body.clone(), Some(body.clone()), loc());
    let native = global("ir.If").call_tuple((b.clone(), body.clone(), Some(body.clone()), loc()));
    agree("an if", rust, native);

    let (loop32, loop8, loopu8) = (
        native_var("i", prim("int32")),
        native_var("j", prim("int8")),
        native_var("k", prim("uint8")),
    );
    let fors: Vec<(&str, Var, Expr, Expr, Option<Expr>)> = vec![
        ("a loop", loop32.clone(), imm("int32", 0), i.clone(), None),
        ("int64 constant bounds", loop32.clone(), imm("int64", 0), imm("int64", 16), None),
        ("uint8 constant bounds", loop32.clone(), imm("uint8", 1), imm("uint8", 200), None),
        ("a bound too large", loop8.clone(), imm("int32", 0), imm("int32", 300), None),
        ("a negative unsigned bound", loopu8.clone(), imm("int32", -1), imm("int32", 3), None),
        ("a wider variable bound", loop8.clone(), imm("int8", 0), i.clone(), None),
        ("a narrower variable bound", loop32.clone(), imm("int32", 0), prim_var("int8"), None),
        ("a float bound", loop32.clone(), f.clone(), i.clone(), None),
        ("a vector bound", loop32.clone(), imm("int32", 0), prim_var("int32x4"), None),
        ("a step", loop32.clone(), imm("int32", 0), i.clone(), Some(imm("int64", 2))),
        ("a float step", loop32.clone(), imm("int32", 0), i.clone(), Some(f.clone())),
        ("a float loop variable", native_var("x", prim("float32")), f.clone(), f.clone(), None),
    ];
    let annotations: Map<FfiString, Any> =
        [(FfiString::from("key"), Any::from(1i64))].into_iter().collect();
    for (what, loop_var, min, extent, step) in fors {
        for kind in [0, 1] {
            let rust = For::new_full(
                loop_var.clone(),
                min.clone(),
                extent.clone(),
                kind,
                body.clone(),
                annotations.clone(),
                step.clone(),
                loc(),
            );
            let native = global("ir.For").call_tuple((
                loop_var.clone(),
                min.clone(),
                extent.clone(),
                kind,
                body.clone(),
                Some(annotations.clone()),
                step.clone(),
                loc(),
            ));
            agree(&format!("For with {what} (kind {kind})"), rust, native);
        }
    }
}

/// `RegionStmt(op, args, body_params, attrs, body, result_vars)` by both.
fn check_region(
    what: &str,
    op: &Op,
    args: Vec<Expr>,
    params: Vec<Var>,
    attrs: Vec<(&str, i64)>,
    results: Vec<Var>,
) {
    let body = SeqStmt::from_stmt(Evaluate::new(1).unwrap()).unwrap();
    let attrs: Map<FfiString, Any> =
        attrs.into_iter().map(|(key, value)| (FfiString::from(key), Any::from(value))).collect();
    let (args, params, results): (Array<Expr>, Array<Var>, Array<Var>) =
        (args.into_iter().collect(), params.into_iter().collect(), results.into_iter().collect());
    let attrs = DictAttrs::new(attrs);
    let rust = RegionStmt::new_full(
        op.clone(),
        args.clone(),
        params.clone(),
        attrs.clone(),
        body.clone(),
        results.clone(),
        loc(),
    );
    let native =
        global("ir.RegionStmt").call_tuple((op.clone(), args, params, attrs, body, results, loc()));
    agree(what, rust, native);
}

#[test]
fn regions_and_functions() {
    let _serial = setup();
    let launch = Op::get("tirx.launch_thread").unwrap();
    let tag: Expr = StringImm::new("threadIdx.x").unwrap().into();
    let empty: Expr = StringImm::new("").unwrap().into();
    let (i, wide) = (native_var("i", prim("int32")), native_var("w", prim("int64")));
    let extent = imm("int32", 8);
    check_region(
        "a region",
        &launch,
        vec![tag.clone(), extent.clone()],
        vec![i.clone()],
        vec![],
        vec![],
    );
    check_region(
        "no parameter",
        &launch,
        vec![tag.clone(), extent.clone()],
        vec![],
        vec![],
        vec![],
    );
    check_region(
        "a parameter of another type",
        &launch,
        vec![tag.clone(), extent.clone()],
        vec![wide.clone()],
        vec![],
        vec![],
    );
    check_region(
        "an empty tag",
        &launch,
        vec![empty, extent.clone()],
        vec![i.clone()],
        vec![],
        vec![],
    );
    check_region("an invalid call", &launch, vec![extent.clone()], vec![i.clone()], vec![], vec![]);
    check_region(
        "an attribute",
        &launch,
        vec![tag.clone(), extent.clone()],
        vec![i.clone()],
        vec![("k", 1)],
        vec![],
    );
    check_region(
        "a repeated variable",
        &launch,
        vec![tag.clone(), extent.clone()],
        vec![i.clone()],
        vec![],
        vec![i.clone()],
    );
    let extern_op = Op::get("tirx.call_extern").unwrap();
    check_region("an operator without regions", &extern_op, vec![tag], vec![], vec![], vec![]);

    let params: Array<Var> = Array::new(vec![
        native_var("a", prim("int32")),
        native_var("t", native_tensor_type(&[4], "float32")),
    ]);
    let body = Some(SeqStmt::from_stmt(Evaluate::new(1).unwrap()).unwrap());
    let attrs = DictAttrs::new(Map::new());
    let missing: Type = MissingType::new().unwrap().into();
    for (what, ret_type) in [("a void function", missing), ("a function", prim("int32").into())] {
        let rust = Function::new_full(
            params.clone(),
            body.clone(),
            ret_type.clone(),
            attrs.clone(),
            loc(),
        );
        let native = global("tirx.Function").call_tuple((
            params.clone(),
            body.clone(),
            ret_type,
            attrs.clone(),
            loc(),
        ));
        agree(what, rust, native);
    }
}

#[test]
fn packed_hooks() {
    let _serial = setup();
    let i = prim_var("int32");
    let returns = |value: Any| {
        FfiFunction::from_packed(move |_args: &[AnyView]| -> Result<Any> { Ok(value.clone()) })
    };
    let typed = OpDef::new("rust_test.packed_infer", "")
        .set_attr("FInferType", returns(Any::from(Type::from(prim("int16")))))
        .op()
        .unwrap();
    let untyped = OpDef::new("rust_test.packed_mistyped", "")
        .set_attr("FInferType", returns(Any::from(5i64)))
        .op()
        .unwrap();
    check_call("a packed inference", typed.into(), vec![i.clone()], None, vec![]);
    check_call("a packed inference of no type", untyped.into(), vec![i.clone()], None, vec![]);

    let param = Var::prim("p", DLDataType::try_from_str("int32").unwrap()).unwrap();
    let params = Array::new(vec![param.clone()]);
    let region = OpDef::new("rust_test.packed_region", "")
        .set_attr("FRegionGetBodyParams", returns(Any::from(params)))
        .set_attr("FRegionValidate", returns(Any::from(1i64)))
        .op()
        .unwrap();
    let fresh = native_var("q", prim("int32"));
    check_region("a packed region contract", &region, vec![], vec![fresh], vec![], vec![]);
}
