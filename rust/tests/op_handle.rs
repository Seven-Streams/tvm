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

//! Operator handles register a Rust-defined operator once, on first use or by `register_ops`,
//! and the region hooks of a Rust-defined operator are native and used by the `RegionStmt`
//! constructor as C++ ones are.
//!
//! The registry does not synchronize registration, so every registration is in one test.

use std::ffi::c_void;

use tvm::ir::sig::IntExpr;
use tvm::ir::{
    register_ops, Call, CallObj, DictAttrs, Evaluate, Expr, Op, OpHandle, RegionStmt,
    RegionStmtObj, Signature, StringImm, Var,
};
use tvm::ops::{int32, int_imm, unknown_loc};
use tvm::tvm_ffi::{
    Any, Array, DLDataType, DLDataTypeExt, Error, ErrorKind, Map, ObjectRefCast, ObjectRefCore,
    Result, String as FfiString, ATTRIBUTE_ERROR, INTERNAL_ERROR, VALUE_ERROR,
};

mod common;
use common::load_tvm_compiler;

fn first_arg_type(call: &CallObj) -> Result<tvm::ir::Type> {
    Ok(call.args.get(0)?.ty.clone())
}

fn accept(_call: &CallObj) -> Result<()> {
    Ok(())
}

static IDENTITY: OpHandle = OpHandle::define("rust_test.identity", "Return the value.", |def| {
    def.signature(Signature::new().arg::<Expr>("value", "The value."))
        .set_attr("TCallEffectKind", 0i64)
        .set_infer_type(first_arg_type)
});
static LAUNCH_THREAD: OpHandle = OpHandle::registered("tirx.launch_thread");
static UNREGISTERED: OpHandle = OpHandle::registered("rust_test.unregistered");
/// A name C++ registers: Rust does not take it over.
static TAKEN: OpHandle = OpHandle::define("tirx.launch_thread", "", |def| def);
/// A definition that fails part way: a second custom validator.
static BROKEN: OpHandle =
    OpHandle::define("rust_test.broken", "", |def| def.set_validator(accept).set_validator(accept));

/// `tirx.launch_thread`'s region contract (`src/tirx/op/region.cc`), written in Rust: one thread
/// index of the extent's type.
static LAUNCH: OpHandle =
    OpHandle::define("rust_test.launch", "Bind a thread index below an extent.", |def| {
        def.signature(Signature::new().arg::<StringImm>("tag", "").arg::<IntExpr>("extent", ""))
            .set_region_body_params(launch_body_params)
            .set_region_validate(no_results_or_attributes)
    });
static PANICKING: OpHandle = OpHandle::define("rust_test.panicking_region", "", |def| {
    def.set_region_body_params(|_: &CallObj| -> Result<Array<Var>> { panic!("hook bug") })
});

fn launch_body_params(call: &CallObj) -> Result<Array<Var>> {
    let tag = call.args.get(0)?.try_cast::<StringImm>()?;
    let dtype = call.args.get(1)?.prim_dtype()?;
    if tag.value.as_str().is_empty() {
        let message = "Check failed: (!tag->value.empty()) is false: launch_thread expects a \
                       nonempty StringImm thread tag";
        return Err(Error::new(VALUE_ERROR, message, ""));
    }
    if dtype.bits <= 1 {
        let message = format!(
            "Check failed: dtype.bits() > 1 ({} vs. 1) : launch_thread extent must have a scalar \
             integer type wider than one bit",
            dtype.bits
        );
        return Err(Error::new(VALUE_ERROR, &message, ""));
    }
    Ok(Array::new(vec![Var::prim("", dtype)?]))
}

fn no_results_or_attributes(region: &RegionStmtObj) -> Result<()> {
    if !region.result_vars.is_empty() || !region.attrs.dict.is_empty() {
        let message = format!(
            "Check failed: (region->result_vars.empty() && region->attrs->dict.empty()) is false: \
             {} expects no results or attributes",
            region.op.name.as_str()
        );
        return Err(Error::new(VALUE_ERROR, &message, ""));
    }
    Ok(())
}

fn assert_error<T>(result: Result<T>, kind: ErrorKind, message: &str) {
    match result {
        Ok(_) => panic!("expected {kind}: {message}"),
        Err(error) => {
            assert_eq!(error.kind(), kind, "{}", error.message());
            assert!(error.message().contains(message), "{} lacks {message}", error.message());
        }
    }
}

fn int_var(name: &str, dtype: &str) -> Var {
    Var::prim(name, DLDataType::try_from_str(dtype).unwrap()).unwrap()
}

/// `RegionStmt(op, {tag, extent}, body_params, attrs, Evaluate(0), result_vars)`, its error text
/// with the operator's name taken out.
fn region(
    op: &Op,
    tag: &str,
    extent: Expr,
    body_params: Vec<Var>,
    attrs: Vec<(&str, i64)>,
    result_vars: Vec<Var>,
) -> std::result::Result<(), (std::string::String, std::string::String)> {
    let args = Array::new(vec![StringImm::new(tag).unwrap().into(), extent]);
    let attrs: Map<FfiString, Any> =
        attrs.into_iter().map(|(key, value)| (FfiString::from(key), Any::from(value))).collect();
    let result_vars = Array::new(result_vars);
    let body = Evaluate::new(0).unwrap();
    let built = RegionStmt::new_full(
        op.clone(),
        args,
        Array::new(body_params),
        DictAttrs::new(attrs),
        body,
        result_vars,
        unknown_loc().unwrap(),
    );
    built.map(|_| ()).map_err(|error| {
        let message = error.message().replace(op.name.as_str(), "<op>");
        (error.kind().to_string(), message)
    })
}

#[test]
fn rust_defined_operators() {
    load_tvm_compiler();

    // A handle registers its operator on first use, once; registering it again does nothing.
    assert!(Op::get("rust_test.identity").is_err(), "not registered before first use");
    let identity = IDENTITY.get().unwrap();
    assert!(identity.same_as(&Op::get("rust_test.identity").unwrap()));
    assert_eq!(identity.doc.as_str(), "Return the value.");
    assert!(IDENTITY.get().unwrap().same_as(&identity));
    register_ops(&[&IDENTITY, &LAUNCH_THREAD]).unwrap();
    assert_eq!(tvm::tvm_rust_register_ops(), 0);
    assert_eq!(tvm::tvm_rust_register_ops(), 0);
    let x = int_var("x", "int32");
    let call = Call::new_full(
        None,
        IDENTITY.get().unwrap().into(),
        Array::new(vec![x.clone().into()]),
        None,
        Array::new(vec![]),
        unknown_loc().unwrap(),
    )
    .unwrap();
    assert!(call.ty.same_as(&x.ty), "the Rust FInferType types the Call");

    // Another thread sees the same operator.
    let name = std::thread::spawn(|| IDENTITY.get().unwrap().name.as_str().to_string());
    assert_eq!(name.join().unwrap(), "rust_test.identity");

    // A handle to an operator registered elsewhere looks it up.
    assert!(LAUNCH_THREAD.get().unwrap().same_as(&Op::get("tirx.launch_thread").unwrap()));
    assert!(UNREGISTERED.get().is_err());

    // A name registered elsewhere is not taken over; the first caller gets the registry's error,
    // later ones an error that names the operator.
    assert_error(TAKEN.register(), ATTRIBUTE_ERROR, "is registered before");
    assert_error(TAKEN.get(), INTERNAL_ERROR, "Operator tirx.launch_thread failed to register");
    assert_error(BROKEN.get(), VALUE_ERROR, "Validator of rust_test.broken is already registered");
    assert_error(
        register_ops(&[&IDENTITY, &BROKEN]),
        INTERNAL_ERROR,
        "Operator rust_test.broken failed to register: ValueError",
    );

    // The region hooks are native function pointers.
    let launch = LAUNCH.get().unwrap();
    for key in ["FRegionGetBodyParams", "FRegionValidate"] {
        let hooks = Op::attr_map::<*mut c_void>(key).unwrap().unwrap();
        assert!(hooks.get(&launch).unwrap().is_some(), "{key}");
    }

    // The `RegionStmt` constructor checks a region of the Rust operator as it checks one of
    // `tirx.launch_thread`: the same outcome, and the same error but for the operator's name.
    let native = LAUNCH_THREAD.get().unwrap();
    let extent: Expr = int32(8).unwrap().into();
    let wide: Expr = int_imm(DLDataType::try_from_str("int64").unwrap(), 8).unwrap().into();
    let flag: Expr = int_imm(DLDataType::try_from_str("bool").unwrap(), 1).unwrap().into();
    let i = int_var("i", "int32");
    let cases = vec![
        ("a well-formed region", "threadIdx.x", extent.clone(), vec![i.clone()], vec![], vec![]),
        ("no body parameter", "threadIdx.x", extent.clone(), vec![], vec![], vec![]),
        ("a body parameter of another type", "threadIdx.x", wide, vec![i.clone()], vec![], vec![]),
        ("an empty tag", "", extent.clone(), vec![i.clone()], vec![], vec![]),
        ("a one-bit extent", "threadIdx.x", flag, vec![int_var("b", "bool")], vec![], vec![]),
        ("an attribute", "threadIdx.x", extent.clone(), vec![i.clone()], vec![("k", 1)], vec![]),
        (
            "a result",
            "threadIdx.x",
            extent.clone(),
            vec![i.clone()],
            vec![],
            vec![int_var("r", "int32")],
        ),
        ("a repeated variable", "threadIdx.x", extent, vec![i.clone()], vec![], vec![i]),
    ];
    for (what, tag, extent, params, attrs, results) in cases {
        let rust =
            region(&launch, tag, extent.clone(), params.clone(), attrs.clone(), results.clone());
        let cpp = region(&native, tag, extent, params, attrs, results);
        assert_eq!(rust, cpp, "{what}");
    }

    // A panic in a hook becomes an error instead of unwinding into C++.
    let panicking = PANICKING.get().unwrap();
    let body = Evaluate::new(0).unwrap();
    let empty = || DictAttrs::new(Map::new());
    let built = RegionStmt::new(panicking, Array::new(vec![]), Array::new(vec![]), empty(), body);
    assert!(built.is_err(), "a panicking hook fails the constructor");
}
