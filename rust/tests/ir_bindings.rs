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

//! Smoke tests for the stubgen-generated `ir` bindings against `libtvm_compiler`.

use tvm::ir::*;
use tvm::runtime::{MemoryScope, MemoryScopeObj};
use tvm::tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeKeyToIndex};
use tvm::tvm_ffi::{
    Any, AnyView, Array, BigInt, DLDataType, DLDataTypeExt, FieldGetter, ObjectCore, ObjectRefCast,
    ObjectRefCore, String as FfiString, TypeIndex,
};

mod common;
use common::{assert_type_contracts, global, load_tvm_compiler, runtime_type_info, structural_equal};

#[test]
fn generated_types_match_the_registry() {
    load_tvm_compiler();
    type Object = tvm::tvm_ffi::Object;
    assert_type_contracts!(true;
        ArgumentInfoObj: Object, AttrsObj: Object, DictAttrsObj: AttrsObj, EnvFuncObj: Object,
        ExprObj: Object, BaseFuncObj: ExprObj, CallObj: ExprObj, ConstantObj: ExprObj,
        DataTypeImmObj: ConstantObj, FloatImmObj: ConstantObj, GenericConstObj: ConstantObj,
        GlobalInfoObj: Object, GlobalVarObj: ExprObj, IRModuleObj: Object, IntImmObj: ConstantObj,
        LocationObj: Object, CallSiteLocObj: LocationObj, OpObj: ExprObj, OpaqueExprObj: ExprObj, RangeObj: Object,
        SourceLocObj: LocationObj, SourceMapObj: Object, SourceNameObj: Object,
        StagingExprObj: ExprObj, LambdaExprObj: StagingExprObj, StmtObj: Object,
        AssertStmtObj: StmtObj, BindObj: StmtObj, BreakObj: StmtObj, ContinueObj: StmtObj,
        EvaluateObj: StmtObj, ForObj: StmtObj, IfObj: StmtObj, RegionStmtObj: StmtObj,
        ReturnObj: StmtObj, SeqStmtObj: StmtObj, StringImmObj: ConstantObj,
        TensorLoadObj: ExprObj, TensorRegionObj: ExprObj, TensorStoreObj: StmtObj,
        TupleObj: ExprObj, TupleGetItemObj: ExprObj, TypeObj: Object, AnyTypeObj: TypeObj,
        FuncTypeObj: TypeObj, MissingTypeObj: TypeObj, OpaqueTypeObj: TypeObj,
        PrimTypeObj: TypeObj, PtrTypeObj: TypeObj, StringTypeObj: TypeObj,
        TensorRegionTypeObj: TypeObj, TupleTypeObj: TypeObj, UnknownLocObj: LocationObj,
        VarObj: ExprObj, WhileObj: StmtObj,
    );
    // Opaque bindings embed only their parent; the native object is larger.
    assert_type_contracts!(false;
        IntSetObj: Object, SourceObj: Object, UniqueNameSupplyObj: Object,
    );
    assert_eq!(MemoryScopeObj::TYPE_KEY, "runtime.MemoryScope");
    assert_eq!(runtime_type_info::<MemoryScopeObj>().type_depth, MemoryScopeObj::TYPE_DEPTH);
}

#[test]
fn native_objects_are_read_through_generated_fields() {
    load_tvm_compiler();
    let int32 = DLDataType::try_from_str("int32").unwrap();
    let loc: Location = global("ir.UnknownLoc").call_packed(&[]).unwrap().try_into().unwrap();

    let value: IntImm = global("ir.IntImm")
        .call_tuple((int32, BigInt::from(42i64), loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(value.value.to_i64(), Some(42));
    // `ty` lives on `ir.Expr`, two `Deref`s up from `IntImmObj`.
    let ty: PrimType = value.ty.clone().try_cast().unwrap();
    assert_eq!(ty.dtype, int32);

    // `PtrType.scope` is the one field whose type lives outside `ir`.
    let pointer: PtrType = global("ir.PtrType")
        .call_tuple((ty.clone(), tvm::tvm_ffi::String::from("global")))
        .unwrap()
        .try_into()
        .unwrap();
    let _scope: &MemoryScope = &pointer.scope;
    assert!(structural_equal(&pointer.element_type, &ty));
}

/// Read a reflected field of `op` through the C ABI getter, bypassing the Rust layout.
fn reflected<T>(op: &Op, field: &str) -> T
where
    T: TryFrom<Any>,
    T::Error: std::fmt::Debug,
{
    let value = FieldGetter::new(OpObj::type_index(), field).unwrap().get_any(&**op).unwrap();
    T::try_from(value).unwrap()
}

fn argument_names(arguments: &Array<ArgumentInfo>) -> Vec<std::string::String> {
    arguments.iter().map(|argument| argument.name.as_str().to_owned()).collect()
}

#[test]
fn hand_written_op_layout_matches_native() {
    load_tvm_compiler();
    let names: Array<FfiString> =
        global("ir.ListOpNames").call_packed(&[]).unwrap().try_into().unwrap();
    assert!(names.len() > 0);

    let (mut with_attrs, mut with_validator) = (0, 0);
    for name in names.iter() {
        let op = Op::get(name.as_str()).unwrap();

        // Reflected members: the direct read agrees with the native getter.
        assert_eq!(op.name.as_str(), name.as_str());
        assert_eq!(i64::from(op.index), reflected::<i64>(&op, "index"), "{name}: index");
        assert_eq!(op.name.as_str(), reflected::<FfiString>(&op, "name").as_str());
        assert_eq!(op.doc.as_str(), reflected::<FfiString>(&op, "doc").as_str(), "{name}: doc");
        assert_eq!(
            op.attrs_type_key.as_str(),
            reflected::<FfiString>(&op, "attrs_type_key").as_str(),
            "{name}: attrs_type_key"
        );
        for (field, direct) in [("args_info", &op.args_info), ("ty_args_info", &op.ty_args_info)] {
            let native = reflected::<Array<ArgumentInfo>>(&op, field);
            assert_eq!(argument_names(direct), argument_names(&native), "{name}: {field}");
        }
        for (field, direct) in
            [("var_args_info", &op.var_args_info), ("var_ty_args_info", &op.var_ty_args_info)]
        {
            let native = reflected::<Option<ArgumentInfo>>(&op, field);
            assert_eq!(
                direct.as_ref().map(|argument| argument.name.as_str().to_owned()),
                native.as_ref().map(|argument| argument.name.as_str().to_owned()),
                "{name}: {field}"
            );
        }

        // Unreflected members: nothing native reads them back, so check their invariants.
        if op.attrs_type_key.as_str().is_empty() {
            assert_eq!(op.attrs_type_index(), 0, "{name}: attrs_type_index");
        } else {
            let key = unsafe { TVMFFIByteArray::from_str(op.attrs_type_key.as_str()) };
            let mut expected = 0;
            assert_eq!(unsafe { TVMFFITypeKeyToIndex(&key, &mut expected) }, 0);
            // Native registration leaves the index at 0 for an operator defined before its
            // attrs type's index is initialized, so only a non-zero index is comparable.
            if op.attrs_type_index() != 0 {
                with_attrs += 1;
                assert_eq!(op.attrs_type_index() as i32, expected, "{name}: attrs_type_index");
            }
        }
        let validator = op.validator().type_index();
        if validator == TypeIndex::kTVMFFINone as i32 {
            assert!(!op.validator_is_custom(), "{name}: custom validator without a validator");
        } else {
            // A native adapter is stored as its function address, a packed one as a Function.
            with_validator += 1;
            assert!(
                validator == TypeIndex::kTVMFFIOpaquePtr as i32
                    || validator == TypeIndex::kTVMFFIFunction as i32,
                "{name}: validator is neither a native nor a packed function \
                 (type index {validator})"
            );
        }
    }
    // Both unreflected members were exercised with a non-default value.
    assert!(with_attrs > 0, "no operator has an attrs type index");
    assert!(with_validator > 0, "no operator has a validator");

    // The registry hands out one instance per name.
    let first = names.get(0).unwrap();
    let (lhs, rhs) = (Op::get(first.as_str()).unwrap(), Op::get(first.as_str()).unwrap());
    assert!(std::ptr::eq::<OpObj>(&*lhs, &*rhs));
    assert!(Op::get("ir.no_such_operator").is_err());
}

#[test]
fn op_attr_map_reads_the_live_registry_column() {
    load_tvm_compiler();
    assert!(Op::has_attr_map("TCallEffectKind").unwrap());
    assert!(!Op::has_attr_map("test.rust.MissingColumn").unwrap());
    assert!(Op::attr_map::<i64>("test.rust.MissingColumn").unwrap().is_none());

    // Every lookup agrees with the reflected per-Op `get_attr`.
    let effect = Op::attr_map::<i64>("TCallEffectKind").unwrap().unwrap();
    let get_attr =
        tvm::tvm_ffi::Function::from_type_method(OpObj::type_index(), "get_attr").unwrap();
    let names: Array<FfiString> =
        global("ir.ListOpNames").call_packed(&[]).unwrap().try_into().unwrap();
    let mut with_effect = 0;
    for name in names.iter() {
        let op = Op::get(name.as_str()).unwrap();
        let native: Any =
            get_attr.call_tuple((op.clone(), FfiString::from("TCallEffectKind"))).unwrap();
        let native = (native.type_index() != TypeIndex::kTVMFFINone as i32)
            .then(|| i64::try_from(native).unwrap());
        assert_eq!(effect.get(&op).unwrap(), native, "{name}");
        with_effect += usize::from(native.is_some());
    }
    assert!(with_effect > 0, "no operator has a TCallEffectKind");

    // A map taken earlier sees later registrations, including the column's growth.
    let key = FfiString::from("test.rust.LiveColumn");
    let register = |name: &str, value: i64| {
        global("ir.RegisterOp").call_tuple((FfiString::from(name), FfiString::from(""))).unwrap();
        global("ir.RegisterOpAttr")
            .call_tuple((FfiString::from(name), key.clone(), value, false))
            .unwrap();
        Op::get(name).unwrap()
    };
    let first = register("test.rust.live_column.first", 7);
    let column = Op::attr_map::<i64>(key.as_str()).unwrap().unwrap();
    assert_eq!(column.get(&first).unwrap(), Some(7));
    let second = register("test.rust.live_column.second", 8);
    assert_eq!(column.get(&second).unwrap(), Some(8));
    assert_eq!(column.get(&Op::get("prim.exp").unwrap()).unwrap(), None);

    // A value of another type is an error, not a silent miss.
    let as_string = Op::attr_map::<FfiString>(key.as_str()).unwrap().unwrap();
    assert!(as_string.get(&first).is_err());
}

#[test]
fn rust_allocated_nodes_equal_native_ones() {
    load_tvm_compiler();
    let int32 = DLDataType::try_from_str("int32").unwrap();
    let loc: Location = global("ir.UnknownLoc").call_packed(&[]).unwrap().try_into().unwrap();
    let int_imm = |value: i64| -> IntImm {
        global("ir.IntImm")
            .call_tuple((int32, BigInt::from(value), loc.clone()))
            .unwrap()
            .try_into()
            .unwrap()
    };
    let native = int_imm(42);

    let rust = IntImm::new(loc.clone(), native.ty.clone(), BigInt::from(42i64));
    assert!(structural_equal(&rust, &native));
    assert!(!structural_equal(&rust, &int_imm(7)));

    // Upcasts along the ancestor chain, and back down with a checked cast.
    let expr: Expr = rust.clone().into();
    let back: IntImm = expr.clone().try_cast().unwrap();
    assert_eq!(back.value.to_i64(), Some(42));
    assert!(expr.clone().try_cast::<Var>().is_err());

    let native_range: Range = global("ir.Range_from_min_extent")
        .call_tuple((int_imm(0), native.clone(), loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    let rust_range = Range::new(int_imm(0).into(), expr.clone(), loc.clone());
    assert!(structural_equal(&rust_range, &native_range));

    let native_stmt: Evaluate = global("ir.Evaluate")
        .call_tuple((native.clone(), loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    let rust_stmt = Evaluate::new(loc.clone(), expr);
    assert!(structural_equal(&rust_stmt, &native_stmt));

    // A Rust-allocated node is printed by the C++ repr machinery.
    let repr: Any = global("ffi.ReprPrint").call_packed(&[AnyView::from(&rust_stmt)]).unwrap();
    let repr = tvm::tvm_ffi::String::try_from(repr).unwrap();
    assert!(repr.as_str().contains("42"), "unexpected repr: {repr}");
}

#[test]
fn with_fields_copy_the_node_and_keep_the_rest() {
    load_tvm_compiler();
    let int32 = DLDataType::try_from_str("int32").unwrap();
    let loc: Location = global("ir.UnknownLoc").call_packed(&[]).unwrap().try_into().unwrap();
    let int_imm = |value: i64| -> Expr {
        let imm: IntImm = global("ir.IntImm")
            .call_tuple((int32, BigInt::from(value), loc.clone()))
            .unwrap()
            .try_into()
            .unwrap();
        imm.into()
    };
    let ty: Type =
        PrimType::try_from(global("ir.PrimType").call_tuple((int32,)).unwrap()).unwrap().into();
    let x: Var = global("ir.Var")
        .call_tuple((FfiString::from("x"), ty.clone(), loc.clone()))
        .unwrap()
        .try_into()
        .unwrap();

    // One field replaced in a new node; the original is untouched and the rest is shared.
    let bind = Bind::new(loc.clone(), x.clone(), int_imm(1));
    let rebound = bind.with_value(int_imm(2));
    assert!(!rebound.same_as(&bind));
    assert!(rebound.var.same_as(&bind.var) && rebound.loc.same_as(&bind.loc));
    assert!(structural_equal(&bind.value, &int_imm(1)));
    assert!(structural_equal(&rebound, &Bind::new(loc.clone(), x.clone(), int_imm(2))));

    // `with` edits several fields in one copy, like C++ `CopyOnWrite()`.
    let exp: Expr = Op::get("prim.exp").unwrap().into();
    let call = Call::new(
        loc.clone(),
        ty.clone(),
        exp,
        Array::new(vec![x.clone().into()]),
        None,
        Array::new(vec![]),
    );
    let edited = call.with(|node| {
        node.args = Array::new(vec![int_imm(3)]);
        node.ty_args = Array::new(vec![ty.clone()]);
    });
    assert!(edited.op.same_as(&call.op) && edited.ty.same_as(&call.ty));
    assert!(structural_equal(&edited.args, &Array::<Expr>::new(vec![int_imm(3)])));
    assert_eq!((edited.ty_args.len(), call.ty_args.len()), (1, 0));
    assert!(call.args.get(0).unwrap().same_as(&x));
}
