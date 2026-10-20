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

//! An `ir.Type` defined in Rust with `#[reflect]` fields, seen through the FFI type table, the C++
//! engines (structural equality and hashing, deep copy, `repr`, TVMScript) and the Rust engines.

use std::mem::offset_of;
use std::ops::Deref;

use tvm::ir::{Location, Type, TypeObj};
use tvm::node::type_base;
use tvm::ops::{int32_dtype, prim_type, repr, structural_hash, unknown_loc};
use tvm::tvm_ffi::derive::{Object, ObjectRef};
use tvm::tvm_ffi::reflection::ShallowCopy;
use tvm::tvm_ffi::{
    structural_map, Any, AnyView, DLDataType, DLDataTypeCode, DefRegionKind, FieldGetter,
    MapDispatch, ObjectArc, ObjectCore, ObjectRefCore, Result, String as FfiString, StructuralView,
    WalkOrder,
};

mod common;
use common::{assert_type_contract, global, load_tvm_compiler, runtime_type_info, structural_equal};

/// A pair of types with a tag, for the tests.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.PairType"]
#[type_register]
#[type_final]
pub struct PairTypeObj {
    base: TypeObj,
    /// The first element.
    #[reflect]
    pub lhs: Type,
    /// The second element.
    #[reflect]
    pub rhs: Type,
    #[reflect]
    pub tag: i64,
    /// Not part of structural equality or hashing.
    #[reflect(seq_hash_ignore)]
    pub note: FfiString,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PairType {
    base: ObjectArc<PairTypeObj>,
}

impl Deref for PairType {
    type Target = PairTypeObj;
    fn deref(&self) -> &PairTypeObj {
        &self.base
    }
}

impl Deref for PairTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

tvm::tvm_ffi::impl_object_upcast!(PairType => Type);

impl PairType {
    fn new(base: TypeObj, lhs: Type, rhs: Type, tag: i64, note: FfiString) -> Self {
        Self { base: ObjectArc::new(PairTypeObj { base, lhs, rhs, tag, note }) }
    }

    /// A copy with the fields `edit` assigns replaced (C++ `CopyOnWrite()`).
    fn with(&self, edit: impl FnOnce(&mut PairTypeObj)) -> Self {
        let mut node = self.base.shallow_copy();
        edit(&mut node);
        Self { base: ObjectArc::new(node) }
    }
}

fn setup() {
    load_tvm_compiler();
    // The first use registers the type and its reflection.
    PairTypeObj::type_index();
}

fn float32() -> DLDataType {
    DLDataType { code: DLDataTypeCode::kDLFloat as u8, bits: 32, lanes: 1 }
}

fn pair(lhs: Type, rhs: Type, tag: i64, note: &str) -> PairType {
    PairType::new(type_base(unknown_loc().unwrap()), lhs, rhs, tag, FfiString::from(note))
}

fn int_float_pair(tag: i64, note: &str) -> PairType {
    let int32: Type = prim_type(int32_dtype()).unwrap().into();
    let float32: Type = prim_type(float32()).unwrap().into();
    pair(int32, float32, tag, note)
}

#[test]
fn registers_under_ir_type_with_its_fields_reflected() {
    setup();
    assert_type_contract::<PairTypeObj, TypeObj>(true);
    assert_eq!(PairTypeObj::TYPE_DEPTH, TypeObj::TYPE_DEPTH + 1);

    let info = runtime_type_info::<PairTypeObj>();
    let metadata = unsafe { &*info.metadata };
    assert_eq!(metadata.structural_eq_hash_kind, 1, "TreeNode, as every ir.Type");
    let fields: Vec<(String, i64, i64)> = (0..info.num_fields as usize)
        .map(|index| {
            let field = unsafe { &*info.fields.add(index) };
            (field.name.as_str().to_string(), field.offset, field.flags)
        })
        .collect();
    assert_eq!(
        fields,
        vec![
            ("lhs".to_string(), offset_of!(PairTypeObj, lhs) as i64, 0),
            ("rhs".to_string(), offset_of!(PairTypeObj, rhs) as i64, 0),
            ("tag".to_string(), offset_of!(PairTypeObj, tag) as i64, 0),
            ("note".to_string(), offset_of!(PairTypeObj, note) as i64, 1 << 3),
        ]
    );
}

#[test]
fn reflection_getters_read_own_and_inherited_fields() {
    setup();
    let value = int_float_pair(7, "seven");
    let get = |field: &str| -> Any {
        FieldGetter::new(PairTypeObj::type_index(), field).unwrap().get_any(&*value).unwrap()
    };
    assert!(Type::try_from(get("lhs")).unwrap().same_as(&value.lhs));
    assert!(Type::try_from(get("rhs")).unwrap().same_as(&value.rhs));
    assert_eq!(i64::try_from(get("tag")).unwrap(), 7);
    assert_eq!(FfiString::try_from(get("note")).unwrap().as_str(), "seven");
    // `loc` belongs to the parent `ir.Type`.
    assert!(Location::try_from(get("loc")).unwrap().same_as(&value.loc));
}

#[test]
fn cpp_structural_equal_and_hash_walk_the_reflected_fields() {
    setup();
    let a = int_float_pair(7, "a");
    let b = int_float_pair(7, "b");
    let c = int_float_pair(8, "a");
    assert!(structural_equal(&a, &b), "the note is ignored");
    assert!(!structural_equal(&a, &c));
    assert_eq!(structural_hash(&a).unwrap(), structural_hash(&b).unwrap());
    assert_ne!(structural_hash(&a).unwrap(), structural_hash(&c).unwrap());

    // Nested in itself, as any type.
    let int32: Type = prim_type(int32_dtype()).unwrap().into();
    let outer_a = pair(a.clone().into(), int32.clone(), 0, "");
    let outer_b = pair(b.into(), int32.clone(), 0, "");
    let outer_c = pair(c.into(), int32, 0, "");
    assert!(structural_equal(&outer_a, &outer_b));
    assert!(!structural_equal(&outer_a, &outer_c));
    assert_eq!(structural_hash(&outer_a).unwrap(), structural_hash(&outer_b).unwrap());
}

#[test]
fn cpp_repr_and_tvmscript_print_the_node() {
    setup();
    let value = int_float_pair(7, "seven");
    let text = repr(&value);
    assert!(text.contains("testing.rust.PairType"), "{text}");
    assert!(text.contains("tag=7"), "{text}");
    assert!(text.contains("seven"), "{text}");

    let script = global("script.printer.Script")
        .call_packed(&[AnyView::from(&value), AnyView::new()])
        .unwrap();
    let script = FfiString::try_from(script).unwrap();
    assert!(script.as_str().contains("testing.rust.Pair("), "{}", script.as_str());
    assert!(script.as_str().contains("\"seven\""), "{}", script.as_str());
}

#[test]
fn cpp_deep_copy_goes_through_the_shallow_copy_and_the_setters() {
    setup();
    let value = int_float_pair(7, "seven");
    let copy = global("ffi.DeepCopy").call_packed(&[AnyView::from(&value)]).unwrap();
    let copy = PairType::try_from(copy).unwrap();
    assert!(!copy.same_as(&value));
    assert!(structural_equal(&copy, &value));
    assert_eq!(copy.tag, 7);
    assert_eq!(copy.note.as_str(), "seven");
}

/// Adds one to every integer leaf.
struct IncrementIntegers;

impl MapDispatch for IncrementIntegers {
    fn dispatch_map(&mut self, value: &StructuralView, _: DefRegionKind) -> Option<Result<Any>> {
        value.cast::<i64>().map(|integer| Ok(Any::from(integer + 1)))
    }
}

#[test]
fn rust_engines_descend_through_the_reflected_fields() {
    setup();
    let value = int_float_pair(7, "seven");
    // Shared with `value`, so the engine copies the node through `__ffi_shallow_copy__`.
    let mapped =
        structural_map(value.clone(), &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    let mapped = PairType::try_from(mapped).unwrap();
    assert!(!mapped.same_as(&value));
    assert_eq!(mapped.tag, 8);
    assert_eq!(mapped.note.as_str(), "seven");
    assert!(mapped.lhs.same_as(&value.lhs));
    assert_eq!(value.tag, 7);
}

#[test]
fn with_is_copy_on_write() {
    setup();
    let value = int_float_pair(7, "seven");
    let edited = value.with(|node| node.tag = 9);
    assert!(!edited.same_as(&value));
    assert_eq!(edited.tag, 9);
    assert_eq!(value.tag, 7);
    assert!(edited.lhs.same_as(&value.lhs));
    assert!(edited.rhs.same_as(&value.rhs));
    assert!(edited.loc.same_as(&value.loc));
    assert_eq!(edited.note.as_str(), "seven");
}

#[test]
fn converts_through_any_and_the_type_hierarchy() {
    setup();
    let value = int_float_pair(1, "");
    let as_type: Type = value.clone().into();
    assert!(as_type.as_node::<PairTypeObj>().is_some());
    let any = Any::from(as_type);
    assert_eq!(any.type_index(), PairTypeObj::type_index());
    let back = PairType::try_from(any).unwrap();
    assert!(back.same_as(&value));
}
