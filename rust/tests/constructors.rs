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

use tvm::ir::{
    Expr, GlobalVar, LambdaExpr, Location, MissingType, PrimType, PtrType, Return, TensorStore,
    Tuple, TupleType, Type, UniqueNameSupply, Var,
};
use tvm::ops::{float_imm, int_imm, unknown_loc};
use tvm::prim::{
    Add, And, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, Div, FloorDiv, FloorMod, LShift, Max,
    Min, Mod, Mul, Not, Or, RShift, Shuffle, Sub, EQ, GE, GT, LE, LT, NE,
};
use tvm::runtime::MemoryScope;
use tvm::tirx::{ExecScope, TensorType};
use tvm::tvm_ffi::{
    Any, AnyCompatible, AnyView, Array, DLDataType, DLDataTypeExt, ObjectRefCore, Result,
    String as FfiString,
};

mod common;
use common::{global, load_tvm_compiler};

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
    load_tvm_compiler();
    check_binary!(
        Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max, EQ, NE, LT, LE, GT, GE, LShift,
        RShift, BitwiseAnd, BitwiseOr, BitwiseXor, And, Or,
    );
}

#[test]
fn unary_nodes() {
    load_tvm_compiler();
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
    load_tvm_compiler();
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
    load_tvm_compiler();
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
    load_tvm_compiler();
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
    load_tvm_compiler();
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
    load_tvm_compiler();
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
    load_tvm_compiler();
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
