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

//! Smoke tests for the stubgen-generated `prim` bindings against `libtvm_compiler`.

use tvm::ir::{Expr, ExprObj, IntImm, Location, PrimType, Var};
use tvm::prim::*;
use tvm::tvm_ffi::{Array, BigInt, DLDataType, DLDataTypeExt, ObjectRefCast, String as FfiString};

mod common;
use common::{assert_type_contracts, global, load_tvm_compiler, structural_equal};

#[test]
fn generated_types_match_the_registry() {
    load_tvm_compiler();
    // Every `prim` node is complete and derives directly from `ir.Expr`.
    assert_type_contracts!(true;
        AddObj: ExprObj, AndObj: ExprObj, BitwiseAndObj: ExprObj, BitwiseNotObj: ExprObj,
        BitwiseOrObj: ExprObj, BitwiseXorObj: ExprObj, BroadcastObj: ExprObj, CastObj: ExprObj,
        DivObj: ExprObj, EQObj: ExprObj, FloorDivObj: ExprObj, FloorModObj: ExprObj,
        GEObj: ExprObj, GTObj: ExprObj, LEObj: ExprObj, LShiftObj: ExprObj, LTObj: ExprObj,
        LetObj: ExprObj, MaxObj: ExprObj, MinObj: ExprObj, ModObj: ExprObj, MulObj: ExprObj,
        NEObj: ExprObj, NotObj: ExprObj, OrObj: ExprObj, RShiftObj: ExprObj, RampObj: ExprObj,
        SelectObj: ExprObj, ShuffleObj: ExprObj, SubObj: ExprObj,
    );
}

/// Operands shared by the tests: two `int32` variables and two boolean expressions over them.
struct Operands {
    loc: Location,
    int32: PrimType,
    x_var: Var,
    x: Expr,
    y: Expr,
    p: Expr,
    q: Expr,
}

fn prim_type(dtype: &str) -> PrimType {
    global("ir.PrimType")
        .call_tuple((DLDataType::try_from_str(dtype).unwrap(),))
        .unwrap()
        .try_into()
        .unwrap()
}

fn operands() -> Operands {
    let loc: Location = global("ir.UnknownLoc").call_packed(&[]).unwrap().try_into().unwrap();
    let int32 = prim_type("int32");
    let var = |name: &str| -> Var {
        global("ir.Var")
            .call_tuple((FfiString::from(name), int32.clone(), loc.clone()))
            .unwrap()
            .try_into()
            .unwrap()
    };
    let x_var = var("x");
    let (x, y): (Expr, Expr) = (x_var.clone().into(), var("y").into());
    let compare = |name: &str| -> Expr {
        global(name).call_tuple((x.clone(), y.clone(), loc.clone())).unwrap().try_into().unwrap()
    };
    let (p, q) = (compare("prim.LT"), compare("prim.GT"));
    Operands { loc, int32, x_var, x, y, p, q }
}

fn int_imm(o: &Operands, value: i64) -> Expr {
    let imm: IntImm = global("ir.IntImm")
        .call_tuple((o.int32.dtype, BigInt::from(value), o.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    imm.into()
}

/// A native binary node reads back its operands, and the Rust-allocated node equals it.
macro_rules! check_binary {
    ($o:expr; $($node:ident($lhs:expr, $rhs:expr)),* $(,)?) => {$({
        let name = concat!("prim.", stringify!($node));
        let native: $node = global(name)
            .call_tuple(($lhs.clone(), $rhs.clone(), $o.loc.clone()))
            .unwrap()
            .try_into()
            .unwrap();
        assert!(structural_equal(&native.a, &$lhs), "{name}: a");
        assert!(structural_equal(&native.b, &$rhs), "{name}: b");
        let rust = $node::new($o.loc.clone(), native.ty.clone(), $lhs.clone(), $rhs.clone());
        assert!(structural_equal(&rust, &native), "{name}: Rust-allocated node differs");
        let swapped = $node::new($o.loc.clone(), native.ty.clone(), $rhs.clone(), $lhs.clone());
        assert!(!structural_equal(&swapped, &native), "{name}: operands are not distinguished");
    })*};
}

#[test]
fn binary_nodes_round_trip() {
    load_tvm_compiler();
    let o = operands();
    check_binary!(o;
        Add(o.x, o.y), Sub(o.x, o.y), Mul(o.x, o.y), Div(o.x, o.y), Mod(o.x, o.y),
        FloorDiv(o.x, o.y), FloorMod(o.x, o.y), Min(o.x, o.y), Max(o.x, o.y),
        EQ(o.x, o.y), NE(o.x, o.y), LT(o.x, o.y), LE(o.x, o.y), GT(o.x, o.y), GE(o.x, o.y),
        BitwiseAnd(o.x, o.y), BitwiseOr(o.x, o.y), BitwiseXor(o.x, o.y),
        LShift(o.x, o.y), RShift(o.x, o.y),
        And(o.p, o.q), Or(o.p, o.q),
    );
}

#[test]
fn other_nodes_round_trip() {
    load_tvm_compiler();
    let o = operands();
    let loc = || o.loc.clone();

    let not: Not = global("prim.Not").call_tuple((o.p.clone(), loc())).unwrap().try_into().unwrap();
    assert!(structural_equal(&Not::new(loc(), not.ty.clone(), o.p.clone()), &not));
    assert!(!structural_equal(&Not::new(loc(), not.ty.clone(), o.q.clone()), &not));

    let bitwise_not: BitwiseNot =
        global("prim.BitwiseNot").call_tuple((o.x.clone(), loc())).unwrap().try_into().unwrap();
    let rust = BitwiseNot::new(loc(), bitwise_not.ty.clone(), o.x.clone());
    assert!(structural_equal(&rust, &bitwise_not));

    // `Cast` keeps its target type in the inherited `ty`.
    let cast: Cast = global("prim.Cast")
        .call_tuple((prim_type("float32"), o.x.clone(), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    let target: PrimType = cast.ty.clone().try_cast().unwrap();
    assert_eq!(target.dtype, DLDataType::try_from_str("float32").unwrap());
    assert!(structural_equal(&Cast::new(loc(), cast.ty.clone(), o.x.clone()), &cast));

    let select: Select = global("prim.Select")
        .call_tuple((o.p.clone(), o.x.clone(), o.y.clone(), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(structural_equal(&select.true_value, &o.x));
    assert!(structural_equal(&select.false_value, &o.y));
    let rust = Select::new(loc(), select.ty.clone(), o.p.clone(), o.x.clone(), o.y.clone());
    assert!(structural_equal(&rust, &select));
    let swapped = Select::new(loc(), select.ty.clone(), o.p.clone(), o.y.clone(), o.x.clone());
    assert!(!structural_equal(&swapped, &select));

    let let_: Let = global("prim.Let")
        .call_tuple((o.x_var.clone(), int_imm(&o, 1), o.x.clone(), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(structural_equal(&let_.body, &o.x));
    let rust = Let::new(loc(), let_.ty.clone(), o.x_var.clone(), int_imm(&o, 1), o.x.clone());
    assert!(structural_equal(&rust, &let_));

    // The reflected field `base` is spelled `base_`: `base` is the parent slot.
    let ramp: Ramp = global("prim.Ramp")
        .call_tuple((o.x.clone(), int_imm(&o, 2), int_imm(&o, 4), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(structural_equal(&ramp.base_, &o.x));
    let rust = Ramp::new(loc(), ramp.ty.clone(), o.x.clone(), int_imm(&o, 2), int_imm(&o, 4));
    assert!(structural_equal(&rust, &ramp));

    let broadcast: Broadcast = global("prim.Broadcast")
        .call_tuple((o.y.clone(), int_imm(&o, 4), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    let rust = Broadcast::new(loc(), broadcast.ty.clone(), o.y.clone(), int_imm(&o, 4));
    assert!(structural_equal(&rust, &broadcast));

    let vectors = Array::<Expr>::new(vec![ramp.clone().into(), broadcast.clone().into()]);
    let indices = Array::<Expr>::new(vec![int_imm(&o, 0), int_imm(&o, 5)]);
    let shuffle: Shuffle = global("prim.Shuffle")
        .call_tuple((vectors.clone(), indices.clone(), loc()))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(shuffle.vectors.len(), 2);
    assert_eq!(shuffle.indices.len(), 2);
    let rust = Shuffle::new(loc(), shuffle.ty.clone(), vectors, indices);
    assert!(structural_equal(&rust, &shuffle));
}

#[test]
fn expressions_are_matched_by_checked_casts() {
    load_tvm_compiler();
    let o = operands();
    // (x + y) * x, built natively and taken apart in Rust.
    let sum: Expr = global("prim.Add")
        .call_tuple((o.x.clone(), o.y.clone(), o.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    let product: Expr = global("prim.Mul")
        .call_tuple((sum, o.x.clone(), o.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();

    assert!(product.clone().try_cast::<Add>().is_err());
    let mul: Mul = product.try_cast().unwrap();
    assert!(mul.a.clone().try_cast::<Sub>().is_err());
    let add: Add = mul.a.clone().try_cast().unwrap();
    assert!(structural_equal(&add.b, &o.y));
    let var: Var = mul.b.clone().try_cast().unwrap();
    assert_eq!(var.name.as_str(), "x");
}
