// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Hand-rolled `ObjectRefCore` + `AnyCompatible` for our ref types.
//!
//! We cannot use tvm-ffi's `#[derive(ObjectRef)]`: its expansion references the
//! crate-private `object::unsafe_` module (ref-count helpers), so it only works
//! *inside* the tvm-ffi crate. This module reimplements the same behaviour with
//! the public API only, and additionally:
//!
//! * tags the FFI `Any` with the object's **runtime** `type_index` (read from
//!   the header) rather than the static container type — so an upcast base ref
//!   (`PrimExpr` holding an `AddNode`) is passed to C++ with the correct type;
//! * accepts **subtypes** in `check`/`try_cast` (via an O(1) ancestor test), so
//!   a `Stmt` argument legitimately arrives as a `tirx.For`/`tirx.SeqStmt`/….

use tvm_ffi::tvm_ffi_sys::TVMFFIGetTypeInfo;

/// O(1) "is `obj_ti` the same as, or a descendant of, `base_ti`?" using the
/// type-info ancestor table (single-inheritance tree, so `ancestors[depth(base)]`
/// is `base` iff `obj` is a descendant).
pub(crate) fn is_instance(obj_ti: i32, base_ti: i32) -> bool {
    if obj_ti == base_ti {
        return true;
    }
    unsafe {
        let info = TVMFFIGetTypeInfo(obj_ti);
        let base_info = TVMFFIGetTypeInfo(base_ti);
        if info.is_null() || base_info.is_null() {
            return false;
        }
        let base_depth = (*base_info).type_depth;
        if (*info).type_depth <= base_depth {
            return false;
        }
        let ancestors = (*info).type_acenstors;
        if ancestors.is_null() {
            return false;
        }
        let anc = *ancestors.offset(base_depth as isize);
        !anc.is_null() && (*anc).type_index == base_ti
    }
}

/// Implement `ObjectRefCore` + `AnyCompatible` + the arg/try-from glue for a ref
/// newtype `$ref { data: ObjectArc<$node> }`. Invoke it in the module that
/// defines `$ref` (the private `data` field is accessed).
macro_rules! impl_object_ref {
    ($ref:ty, $node:ty) => {
        impl $ref {
            /// C++ `ObjectRef::same_as`: pointer identity of the underlying
            /// object — the mutator's "unchanged" test (any ref class of any
            /// node family compares against any other).
            #[inline]
            pub fn same_as<O: tvm_ffi::object::ObjectRefCore>(&self, other: &O) -> bool {
                // Safety: both as_raw calls only read the pointer for an
                // identity comparison; no dereference happens.
                unsafe {
                    tvm_ffi::ObjectArc::as_raw(&self.data) as *const u8
                        == tvm_ffi::ObjectArc::as_raw(
                            <O as tvm_ffi::object::ObjectRefCore>::data(other),
                        ) as *const u8
                }
            }
        }

        unsafe impl tvm_ffi::object::ObjectRefCore for $ref {
            type ContainerType = $node;
            #[inline]
            fn data(this: &Self) -> &tvm_ffi::ObjectArc<$node> {
                &this.data
            }
            #[inline]
            fn into_data(this: Self) -> tvm_ffi::ObjectArc<$node> {
                this.data
            }
            #[inline]
            fn from_data(data: tvm_ffi::ObjectArc<$node>) -> Self {
                Self { data }
            }
        }

        unsafe impl tvm_ffi::type_traits::AnyCompatible for $ref {
            fn type_str() -> std::string::String {
                <$node as tvm_ffi::object::ObjectCore>::TYPE_KEY.to_string()
            }

            unsafe fn copy_to_any_view(src: &Self, data: &mut tvm_ffi::tvm_ffi_sys::TVMFFIAny) {
                let raw = tvm_ffi::ObjectArc::as_raw(&src.data);
                // Tag with the *runtime* type index read from the object header.
                data.type_index = (*(raw as *const tvm_ffi::tvm_ffi_sys::TVMFFIObject)).type_index;
                data.small_str_len = 0;
                data.data_union.v_obj = raw as *mut tvm_ffi::tvm_ffi_sys::TVMFFIObject;
            }

            unsafe fn move_to_any(src: Self, data: &mut tvm_ffi::tvm_ffi_sys::TVMFFIAny) {
                let raw = tvm_ffi::ObjectArc::into_raw(src.data);
                data.type_index = (*(raw as *const tvm_ffi::tvm_ffi_sys::TVMFFIObject)).type_index;
                data.small_str_len = 0;
                data.data_union.v_obj = raw as *mut tvm_ffi::tvm_ffi_sys::TVMFFIObject;
            }

            unsafe fn check_any_strict(data: &tvm_ffi::tvm_ffi_sys::TVMFFIAny) -> bool {
                $crate::object_ref::is_instance(
                    data.type_index,
                    <$node as tvm_ffi::object::ObjectCore>::type_index(),
                )
            }

            unsafe fn copy_from_any_view_after_check(
                data: &tvm_ffi::tvm_ffi_sys::TVMFFIAny,
            ) -> Self {
                // Bump the strong count without touching crate-private helpers:
                // `from_raw` adopts the pointer (no inc), `clone` does the inc,
                // and `forget` prevents the adopted handle's dec on drop.
                let ptr = data.data_union.v_obj as *const $node;
                let borrowed = tvm_ffi::ObjectArc::from_raw(ptr);
                let owned = borrowed.clone();
                std::mem::forget(borrowed);
                Self { data: owned }
            }

            unsafe fn move_from_any_after_check(
                data: &mut tvm_ffi::tvm_ffi_sys::TVMFFIAny,
            ) -> Self {
                let ptr = data.data_union.v_obj as *const $node;
                let obj = Self {
                    data: tvm_ffi::ObjectArc::from_raw(ptr),
                };
                data.type_index = tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex::kTVMFFINone as i32;
                data.data_union.v_int64 = 0;
                obj
            }

            unsafe fn try_cast_from_any_view(
                data: &tvm_ffi::tvm_ffi_sys::TVMFFIAny,
            ) -> std::result::Result<Self, ()> {
                if Self::check_any_strict(data) {
                    Ok(Self::copy_from_any_view_after_check(data))
                } else {
                    Err(())
                }
            }
        }

        tvm_ffi::impl_try_from_any!($ref);
        tvm_ffi::impl_arg_into_ref!($ref);
        tvm_ffi::impl_into_arg_holder_default!($ref);
    };
}

pub(crate) use impl_object_ref;
