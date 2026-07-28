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

//! Native Rust structural visiting.
//!
//! This module separates the two jobs involved in a visit:
//!
//! * [`VisitValue`] provides borrowed matching for generated Rust dispatch.
//! * `NativeWalker` owns recursion through containers and reflected fields.
//!
//! TVM's runtime object registry is open, so the walker still uses the stable
//! tvm-ffi reflection ABI for arbitrary registered node types. That ABI is
//! only the object-description boundary: traversal, control flow, typed
//! dispatch, visitor state, and definition-region propagation remain in Rust.
//! A Rust handler may override a type's children by visiting them through
//! [`VisitCtx`] and returning [`WalkResult::Skip`]. No `ffi.StructuralVisitor`
//! is constructed and no C++ `DefaultVisit` function is called. A non-container
//! type with a foreign `__s_visit__` hook must be handled this way; advancing
//! into its default children is rejected instead of silently substituting
//! reflection with potentially different semantics.

use std::ops::ControlFlow;
use std::os::raw::c_void;

use tvm_ffi::any::{Any, AnyView};
use tvm_ffi::error::{Error, Result, RUNTIME_ERROR};
use tvm_ffi::function::Function;
use tvm_ffi::object::ObjectCore;
use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFIFieldInfo, TVMFFIGetTypeInfo, TVMFFITypeIndex};

use crate::object_ref::is_instance;
use crate::reflect::{
    for_each_field, FLAG_SEQ_HASH_DEF_NON_RECURSIVE, FLAG_SEQ_HASH_DEF_RECURSIVE,
    FLAG_SEQ_HASH_IGNORE,
};
use crate::runtime::{raw_of, raw_of_owned, type_attr_column, type_key_of, view_of, SeqPrefix};

#[cfg(test)]
use tvm_ffi::tvm_ffi_sys::TVMFFIByteArray;

// Static type indices added after the pinned tvm-ffi-sys bindings were
// generated.  They are stable ABI constants in tvm/ffi/c_api.h.
const TYPE_LIST: i32 = 75;
const TYPE_DICT: i32 = 76;

const STRUCTURAL_VISIT_ATTR: &str = "__s_visit__";

/// What a callback asks the Rust walker to do with the current value.
pub enum WalkResult {
    /// Continue and visit this value's children.
    Advance,
    /// Continue without visiting this value's children or firing its exit hook.
    Skip,
    /// Halt the entire traversal.
    Interrupt,
    /// Halt the entire traversal and return a payload to the caller.
    InterruptWith(Any),
}

impl WalkResult {
    /// Halt traversal with an FFI-compatible payload.
    pub fn interrupt_with<T: Into<Any>>(payload: T) -> Self {
        Self::InterruptWith(payload.into())
    }
}

/// Convert either an infallible or fallible typed handler result.
///
/// This keeps simple handlers terse while allowing a handler to return
/// `tvm_ffi::error::Result<WalkResult>` and use `?`.
pub trait IntoVisitResult {
    fn into_visit_result(self) -> Result<WalkResult>;
}

impl IntoVisitResult for WalkResult {
    fn into_visit_result(self) -> Result<WalkResult> {
        Ok(self)
    }
}

impl IntoVisitResult for Result<WalkResult> {
    fn into_visit_result(self) -> Result<WalkResult> {
        self
    }
}

/// Whether a callback runs before or after a value's children.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Before the value's children.
    Enter,
    /// After the value's children.
    Exit,
}

/// Placement of generated typed handlers relative to child traversal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VisitOrder {
    /// Run the typed handler before the current value's children.
    #[default]
    PreOrder,
    /// Run the typed handler after the current value's children.
    PostOrder,
}

/// Definition-region state active at the current value.
///
/// Reflected fields marked `SEqHashDefRecursive` or
/// `SEqHashDefNonRecursive` override the inherited state for that field's
/// complete recursive visit.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DefRegionKind {
    /// The value is outside a definition region.
    #[default]
    None = 0,
    /// Definitions apply recursively through the visited value.
    Recursive = 1,
    /// Definitions apply to the visited value using non-recursive semantics.
    NonRecursive = 2,
}

/// Result of a completed Rust walk.
///
/// `Continue(())` means the whole graph was visited. `Break(payload)` means a
/// handler interrupted it; a payload-less interrupt carries `ffi::None`.
pub type VisitOutcome = ControlFlow<Any>;

/// Fallible result returned by generated typed dispatch.
#[doc(hidden)]
pub type VisitResult = Result<WalkResult>;

/// A borrowed view of a raw tvm-ffi value.
///
/// Generated visitors match this value without taking ownership: borrowed
/// object-node handlers use [`VisitValue::as_node`], while POD or object-ref
/// value handlers use [`VisitValue::cast`].
#[repr(transparent)]
pub struct VisitValue(TVMFFIAny);

impl VisitValue {
    /// Wrap a raw borrowed FFI value.
    #[inline]
    pub(crate) fn from_raw(raw: TVMFFIAny) -> Self {
        VisitValue(raw)
    }

    /// Convert the value into an owned typed handle.
    #[inline]
    pub fn cast<R: tvm_ffi::type_traits::AnyCompatible>(&self) -> Option<R> {
        unsafe {
            if R::check_any_strict(&self.0) {
                Some(R::copy_from_any_view_after_check(&self.0))
            } else {
                None
            }
        }
    }

    /// Runtime type index stored in this value.
    #[inline]
    pub fn type_index(&self) -> i32 {
        self.0.type_index
    }

    /// Borrow the value as node type `N` if it is one of that type's instances.
    #[inline]
    pub fn as_node<N: ObjectCore>(&self) -> Option<&N> {
        if self.0.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            return None;
        }
        if !is_instance(self.0.type_index, N::type_index()) {
            return None;
        }
        Some(unsafe { &*(self.0.data_union.v_obj as *const N) })
    }
}

enum NativeHalt {
    Interrupt(Any),
    Error(Error),
}

impl From<Error> for NativeHalt {
    fn from(error: Error) -> Self {
        NativeHalt::Error(error)
    }
}

type NativeResult = std::result::Result<(), NativeHalt>;

/// Typed dispatch implemented by the visitor object itself.
///
/// The dispatch macro tests the impl's `visit_*` methods in source order.
/// Borrowed node arguments use refcount-free subtype checks, owned
/// FFI-compatible arguments use exact value casts, and `&VisitValue` is a
/// catch-all. `None` asks the Rust walker to continue normally.
pub trait VisitDispatch: Sized {
    fn dispatch_visit(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>)
        -> Option<VisitResult>;
}

/// Recursive traversal access passed to a typed handler.
///
/// The context contains the walker, not the visitor.  A handler lends its
/// current `&mut self` back to [`VisitCtx::visit`], so nested traversal is an
/// ordinary checked Rust reborrow and needs no raw visitor pointer.
pub struct VisitCtx<'a> {
    walker: &'a NativeWalker,
    order: VisitOrder,
    def_region_kind: DefRegionKind,
    halted: Option<NativeHalt>,
}

impl VisitCtx<'_> {
    /// Return the definition-region state active at the current node.
    pub fn def_region_kind(&self) -> DefRegionKind {
        self.def_region_kind
    }

    /// Visit `child` immediately with the same typed dispatcher.
    pub fn visit<V, T>(&mut self, visitor: &mut V, child: &T) -> bool
    where
        V: VisitDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.visit_with_def_region(visitor, child, self.def_region_kind)
    }

    /// Visit `child` under an explicitly selected definition-region state.
    ///
    /// The override is scoped to this recursive call. The current context is
    /// unchanged after success, error, or interruption.
    pub fn visit_with_def_region<V, T>(
        &mut self,
        visitor: &mut V,
        child: &T,
        def_region_kind: DefRegionKind,
    ) -> bool
    where
        V: VisitDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        if self.halted.is_some() {
            return false;
        }
        let mut dispatch = DispatchVisitor {
            visitor,
            order: self.order,
        };
        let result =
            self.walker
                .visit_raw(raw_of(AnyView::from(child)), &mut dispatch, def_region_kind);
        self.absorb(result)
    }

    fn absorb(&mut self, result: NativeResult) -> bool {
        match result {
            Ok(()) => true,
            Err(halt) => {
                self.halted = Some(halt);
                false
            }
        }
    }
}

trait NativeVisit {
    fn order(&self) -> VisitOrder {
        VisitOrder::PreOrder
    }

    fn enter(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult>;

    fn exit(&mut self, _value: &VisitValue, _ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        Ok(WalkResult::Advance)
    }
}

struct DispatchVisitor<'a, V> {
    visitor: &'a mut V,
    order: VisitOrder,
}

impl<V: VisitDispatch> NativeVisit for DispatchVisitor<'_, V> {
    fn order(&self) -> VisitOrder {
        self.order
    }

    fn enter(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        match self.order {
            VisitOrder::PreOrder => self
                .visitor
                .dispatch_visit(value, ctx)
                .unwrap_or(Ok(WalkResult::Advance)),
            VisitOrder::PostOrder => Ok(WalkResult::Advance),
        }
    }

    fn exit(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        match self.order {
            VisitOrder::PreOrder => Ok(WalkResult::Advance),
            VisitOrder::PostOrder => self
                .visitor
                .dispatch_visit(value, ctx)
                .unwrap_or(Ok(WalkResult::Advance)),
        }
    }
}

struct CallbackVisitor<F>(F);

impl<F, O> NativeVisit for CallbackVisitor<F>
where
    F: FnMut(&VisitValue, Phase, DefRegionKind) -> O,
    O: IntoVisitResult,
{
    fn enter(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        (self.0)(value, Phase::Enter, ctx.def_region_kind()).into_visit_result()
    }

    fn exit(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        (self.0)(value, Phase::Exit, ctx.def_region_kind()).into_visit_result()
    }
}

/// Stateless Rust recursion engine.
struct NativeWalker;

impl NativeWalker {
    fn visit_raw<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        if value.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(());
        }

        let visit_value = VisitValue::from_raw(value);
        let mut ctx = VisitCtx {
            walker: self,
            order: visitor.order(),
            def_region_kind,
            halted: None,
        };
        let enter = match visitor.enter(&visit_value, &mut ctx) {
            Ok(flow) => flow,
            Err(error) => return Err(self.with_value_context(error.into(), value)),
        };
        if let Some(halt) = ctx.halted.take() {
            return Err(self.with_value_context(halt, value));
        }
        match enter {
            WalkResult::Advance => {}
            WalkResult::Skip => return Ok(()),
            WalkResult::Interrupt => return Err(NativeHalt::Interrupt(Any::new())),
            WalkResult::InterruptWith(payload) => return Err(NativeHalt::Interrupt(payload)),
        }

        if let Err(halt) = self.visit_children_raw(value, visitor, def_region_kind) {
            return Err(self.with_value_context(halt, value));
        }

        let exit = match visitor.exit(&visit_value, &mut ctx) {
            Ok(flow) => flow,
            Err(error) => return Err(self.with_value_context(error.into(), value)),
        };
        if let Some(halt) = ctx.halted.take() {
            return Err(self.with_value_context(halt, value));
        }
        match exit {
            WalkResult::Interrupt => Err(NativeHalt::Interrupt(Any::new())),
            WalkResult::InterruptWith(payload) => Err(NativeHalt::Interrupt(payload)),
            WalkResult::Advance | WalkResult::Skip => Ok(()),
        }
    }

    fn visit_children_raw<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        match value.type_index {
            x if x == TVMFFITypeIndex::kTVMFFIArray as i32 || x == TYPE_LIST => {
                self.visit_sequence(value, visitor, def_region_kind)
            }
            x if x == TVMFFITypeIndex::kTVMFFIMap as i32 || x == TYPE_DICT => {
                self.visit_map(value, visitor, def_region_kind)
            }
            x if x < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 => Ok(()),
            _ => self.visit_reflected_fields(value, visitor, def_region_kind),
        }
    }

    fn visit_sequence<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        let seq = unsafe { &*(value.data_union.v_obj as *const SeqPrefix) };
        if seq.size < 0 {
            return Err(runtime_error("native visitor: sequence reports a negative size").into());
        }
        if seq.data.is_null() && seq.size != 0 {
            return Err(runtime_error(
                "native visitor: non-empty sequence has a null data pointer",
            )
            .into());
        }
        let size = usize::try_from(seq.size)
            .map_err(|_| runtime_error("native visitor: sequence size does not fit usize"))?;
        if size == 0 {
            return Ok(());
        }

        if value.type_index == TYPE_LIST {
            // List storage may be invalidated by a re-entrant callback.  Own a
            // snapshot before running the first callback.
            let children: Vec<Any> = {
                let cells = unsafe { std::slice::from_raw_parts(seq.data, size) };
                cells
                    .iter()
                    .map(|cell| Any::from(unsafe { view_of(cell) }))
                    .collect()
            };
            for (index, mut child) in children.into_iter().enumerate() {
                let raw = raw_of_owned(&mut child);
                self.visit_raw(raw, visitor, def_region_kind)
                    .map_err(|halt| {
                        with_error_context(halt, &format!("sequence item [{index}]"))
                    })?;
            }
            return Ok(());
        }

        // Array is immutable, so its element cells remain stable throughout
        // recursive callbacks and need no refcounted snapshot.
        let cells = unsafe { std::slice::from_raw_parts(seq.data, size) };
        for (index, child) in cells.iter().enumerate() {
            self.visit_raw(*child, visitor, def_region_kind)
                .map_err(|halt| with_error_context(halt, &format!("sequence item [{index}]")))?;
        }
        Ok(())
    }

    fn visit_map<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        // Map storage is private C++.  The Rust binding itself uses these
        // public iterator functors; using them here does not invoke structural
        // visiting or transfer traversal control out of Rust.
        let is_dict = value.type_index == TYPE_DICT;
        let (size_name, iter_name) = if is_dict {
            ("ffi.DictSize", "ffi.DictForwardIterFunctor")
        } else {
            ("ffi.MapSize", "ffi.MapForwardIterFunctor")
        };
        let size = Function::get_global(size_name)?
            .call_packed(&[unsafe { view_of(&value) }])
            .and_then(i64::try_from)?;
        if size < 0 {
            return Err(runtime_error("native visitor: map reports a negative size").into());
        }
        let size = usize::try_from(size)
            .map_err(|_| runtime_error("native visitor: map size does not fit usize"))?;
        if size == 0 {
            return Ok(());
        }

        let iter_any =
            Function::get_global(iter_name)?.call_packed(&[unsafe { view_of(&value) }])?;
        let iter = Function::try_from(iter_any)?;

        if is_dict {
            // Dict mutation invalidates its iterator, so snapshot all entries
            // before dispatching to user code.
            let mut entries = Vec::with_capacity(size);
            for index in 0..size {
                let key = iter.call_packed(&[AnyView::from(&0i64)])?;
                let map_value = iter.call_packed(&[AnyView::from(&1i64)])?;
                entries.push((key, map_value));
                if index + 1 != size {
                    iter.call_packed(&[AnyView::from(&2i64)])?;
                }
            }

            for (index, (mut key, mut map_value)) in entries.into_iter().enumerate() {
                let key_raw = raw_of_owned(&mut key);
                self.visit_raw(key_raw, visitor, def_region_kind)
                    .map_err(|halt| with_error_context(halt, &format!("dict key [{index}]")))?;
                let value_raw = raw_of_owned(&mut map_value);
                self.visit_raw(value_raw, visitor, def_region_kind)
                    .map_err(|halt| with_error_context(halt, &format!("dict value [{index}]")))?;
            }
            return Ok(());
        }

        // Map is immutable.  Retain only the current owned key/value pair.
        for index in 0..size {
            let mut key = iter.call_packed(&[AnyView::from(&0i64)])?;
            let mut map_value = iter.call_packed(&[AnyView::from(&1i64)])?;
            let key_raw = raw_of_owned(&mut key);
            self.visit_raw(key_raw, visitor, def_region_kind)
                .map_err(|halt| with_error_context(halt, &format!("map key [{index}]")))?;
            let value_raw = raw_of_owned(&mut map_value);
            self.visit_raw(value_raw, visitor, def_region_kind)
                .map_err(|halt| with_error_context(halt, &format!("map value [{index}]")))?;
            if index + 1 != size {
                iter.call_packed(&[AnyView::from(&2i64)])?;
            }
        }
        Ok(())
    }

    fn visit_reflected_fields<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        if unsafe { TVMFFIGetTypeInfo(value.type_index) }.is_null() {
            return Err(runtime_error(&format!(
                "native visitor: unregistered type index {}",
                value.type_index
            ))
            .into());
        }
        if has_foreign_structural_visit(value.type_index) {
            return Err(runtime_error(&format!(
                "native visitor: type `{}` registers foreign `{STRUCTURAL_VISIT_ATTR}`; \
                 use a matching pre-order Rust handler, visit its children through `VisitCtx`, \
                 and return `WalkResult::Skip`",
                type_key_of(value.type_index)
            ))
            .into());
        }

        let object = unsafe { value.data_union.v_obj } as *mut u8;
        let halted = unsafe {
            for_each_field(value.type_index, |field| {
                if field.flags & FLAG_SEQ_HASH_IGNORE != 0 {
                    return ControlFlow::Continue(());
                }

                let Some(getter) = field.getter else {
                    return ControlFlow::Break(NativeHalt::Error(runtime_error(&format!(
                        "native visitor: reflected field `{}` has no getter",
                        field.name.as_str()
                    ))));
                };
                let address = object.offset(field.offset as isize) as *mut c_void;
                let mut child_raw = TVMFFIAny::new();
                if getter(address, &mut child_raw) != 0 {
                    return ControlFlow::Break(with_error_context(
                        NativeHalt::Error(Error::from_raised()),
                        &format!("field `{}`", field.name.as_str()),
                    ));
                }

                // A reflection getter returns an owned Any.  Keep it alive
                // while the recursive walk borrows its raw cell.
                let mut child = Any::from_raw_ffi_any(child_raw);
                let borrowed = raw_of_owned(&mut child);
                let child_region = field_def_region(field, def_region_kind);
                match self.visit_raw(borrowed, visitor, child_region) {
                    Ok(()) => ControlFlow::Continue(()),
                    Err(halt) => ControlFlow::Break(with_error_context(
                        halt,
                        &format!("field `{}`", field.name.as_str()),
                    )),
                }
            })
        };
        halted.map_or(Ok(()), Err)
    }

    fn with_value_context(&self, halt: NativeHalt, value: TVMFFIAny) -> NativeHalt {
        if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            halt
        } else {
            with_error_context(halt, &format!("object `{}`", type_key_of(value.type_index)))
        }
    }
}

/// Visit `root` in pre-order with typed handlers stored in `visitor`.
pub fn structural_visit<R, V>(root: &R, visitor: &mut V) -> Result<VisitOutcome>
where
    V: VisitDispatch,
    for<'x> AnyView<'x>: From<&'x R>,
{
    structural_visit_ordered(root, visitor, VisitOrder::PreOrder)
}

/// Visit `root` with typed handlers placed according to `order`.
pub fn structural_visit_ordered<R, V>(
    root: &R,
    visitor: &mut V,
    order: VisitOrder,
) -> Result<VisitOutcome>
where
    V: VisitDispatch,
    for<'x> AnyView<'x>: From<&'x R>,
{
    let walker = NativeWalker;
    let mut dispatch = DispatchVisitor { visitor, order };
    finish(walker.visit_raw(
        raw_of(AnyView::from(root)),
        &mut dispatch,
        DefRegionKind::None,
    ))
}

/// Native pre/post walk used by analyses that need to observe every raw value.
pub fn walk<R, F, O>(root: &R, mut callback: F) -> Result<VisitOutcome>
where
    for<'x> AnyView<'x>: From<&'x R>,
    F: FnMut(&VisitValue, Phase) -> O,
    O: IntoVisitResult,
{
    walk_with_context(root, move |value, phase, _def_region_kind| {
        callback(value, phase)
    })
}

/// Native pre/post walk whose callback also receives definition-region state.
pub fn walk_with_context<R, F, O>(root: &R, callback: F) -> Result<VisitOutcome>
where
    for<'x> AnyView<'x>: From<&'x R>,
    F: FnMut(&VisitValue, Phase, DefRegionKind) -> O,
    O: IntoVisitResult,
{
    let walker = NativeWalker;
    let mut callback = CallbackVisitor(callback);
    finish(walker.visit_raw(
        raw_of(AnyView::from(root)),
        &mut callback,
        DefRegionKind::None,
    ))
}

fn finish(result: NativeResult) -> Result<VisitOutcome> {
    match result {
        Ok(()) => Ok(ControlFlow::Continue(())),
        Err(NativeHalt::Error(error)) => Err(error),
        Err(NativeHalt::Interrupt(payload)) => Ok(ControlFlow::Break(payload)),
    }
}

fn field_def_region(field: &TVMFFIFieldInfo, inherited: DefRegionKind) -> DefRegionKind {
    if field.flags & FLAG_SEQ_HASH_DEF_NON_RECURSIVE != 0 {
        DefRegionKind::NonRecursive
    } else if field.flags & FLAG_SEQ_HASH_DEF_RECURSIVE != 0 {
        DefRegionKind::Recursive
    } else {
        inherited
    }
}

fn has_foreign_structural_visit(type_index: i32) -> bool {
    unsafe {
        // Resolve the column for each query because later type registrations
        // may reallocate its backing array.
        let column = type_attr_column(STRUCTURAL_VISIT_ATTR);
        if column.is_null() || (*column).data.is_null() {
            return false;
        }
        let index = type_index - (*column).begin_index;
        if index < 0 || index >= (*column).size {
            return false;
        }
        let attr = *(*column).data.offset(index as isize);
        attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32
    }
}

fn with_error_context(halt: NativeHalt, frame: &str) -> NativeHalt {
    match halt {
        NativeHalt::Error(error) => NativeHalt::Error(Error::with_appended_backtrace(
            error,
            &format!("[native structural visit] {frame}\n"),
        )),
        interrupt => interrupt,
    }
}

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tvm_ffi::{Array, Map, Shape, String as FfiString};

    extern "C" {
        fn TVMFFITypeRegisterAttr(
            type_index: i32,
            attr_name: *const TVMFFIByteArray,
            attr_value: *const TVMFFIAny,
        ) -> i32;
    }

    struct RegionProbe(Vec<DefRegionKind>);

    impl NativeVisit for RegionProbe {
        fn enter(&mut self, _value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
            self.0.push(ctx.def_region_kind());
            Ok(WalkResult::Advance)
        }
    }

    #[test]
    fn def_region_is_inherited_through_containers() {
        let root = Array::new(vec![1i64, 2]);
        let walker = NativeWalker;
        let mut probe = RegionProbe(Vec::new());
        assert!(walker
            .visit_raw(
                raw_of(AnyView::from(&root)),
                &mut probe,
                DefRegionKind::Recursive,
            )
            .is_ok());
        assert_eq!(probe.0, vec![DefRegionKind::Recursive; 3]);
    }

    #[test]
    fn reflected_field_flags_override_or_inherit_def_region() {
        let mut field: TVMFFIFieldInfo = unsafe { std::mem::zeroed() };
        assert_eq!(
            field_def_region(&field, DefRegionKind::Recursive),
            DefRegionKind::Recursive
        );

        field.flags = FLAG_SEQ_HASH_DEF_NON_RECURSIVE;
        assert_eq!(
            field_def_region(&field, DefRegionKind::Recursive),
            DefRegionKind::NonRecursive
        );

        field.flags = FLAG_SEQ_HASH_DEF_RECURSIVE;
        assert_eq!(
            field_def_region(&field, DefRegionKind::NonRecursive),
            DefRegionKind::Recursive
        );
    }

    #[test]
    fn plain_walk_uses_native_sequence_fallback() {
        let root = Array::new(vec![1i64, 2, 3]);
        let mut integers = 0;
        assert!(walk(&root, |value, phase| {
            if phase == Phase::Enter && value.cast::<i64>().is_some() {
                integers += 1;
            }
            WalkResult::Advance
        })
        .unwrap()
        .is_continue());
        assert_eq!(integers, 3);
    }

    #[test]
    fn plain_walk_uses_native_map_fallback() {
        let root: Map<FfiString, i64> =
            [(FfiString::from("a"), 1i64), (FfiString::from("b"), 2i64)]
                .into_iter()
                .collect();
        let mut integers = 0;
        assert!(walk(&root, |value, phase| {
            if phase == Phase::Enter && value.cast::<i64>().is_some() {
                integers += 1;
            }
            WalkResult::Advance
        })
        .unwrap()
        .is_continue());
        assert_eq!(integers, 2);
    }

    struct SkipForeignShape;

    #[crate::dispatch(visit)]
    impl SkipForeignShape {
        fn visit_shape(&mut self, _shape: Shape, _ctx: &mut VisitCtx<'_>) -> WalkResult {
            WalkResult::Skip
        }
    }

    #[test]
    fn foreign_structural_visit_requires_explicit_rust_override() {
        let hook = Function::get_global("ffi.ArraySize").unwrap();
        let attr_name = unsafe { TVMFFIByteArray::from_str(STRUCTURAL_VISIT_ATTR) };
        let attr_value = raw_of(AnyView::from(&hook));
        assert_eq!(
            unsafe {
                TVMFFITypeRegisterAttr(
                    TVMFFITypeIndex::kTVMFFIShape as i32,
                    &attr_name,
                    &attr_value,
                )
            },
            0
        );

        let root = Shape::from([2i64, 3]);
        let error = match walk(&root, |_value, _phase| WalkResult::Advance) {
            Err(error) => error,
            Ok(_) => panic!("foreign structural visit unexpectedly used reflection"),
        };
        assert!(error.message().contains("registers foreign `__s_visit__`"));
        assert!(error.message().contains("return `WalkResult::Skip`"));

        assert!(structural_visit(&root, &mut SkipForeignShape)
            .unwrap()
            .is_continue());
    }

    #[test]
    fn mutable_list_is_snapshotted_before_callbacks() {
        let root = Function::get_global("ffi.List")
            .unwrap()
            .call_packed(&[AnyView::from(&1i64), AnyView::from(&2i64)])
            .unwrap();
        let captured = root.clone();
        let append = Function::get_global("ffi.ListAppend").unwrap();
        let mut appended = false;
        let mut integers = Vec::new();

        assert!(walk(&root, |value, phase| {
            if phase == Phase::Enter {
                if let Some(integer) = value.cast::<i64>() {
                    integers.push(integer);
                    if !appended {
                        append
                            .call_packed(&[AnyView::from(&captured), AnyView::from(&3i64)])
                            .unwrap();
                        appended = true;
                    }
                }
            }
            WalkResult::Advance
        })
        .unwrap()
        .is_continue());

        assert_eq!(integers, vec![1, 2]);
        let size = Function::get_global("ffi.ListSize")
            .unwrap()
            .call_packed(&[AnyView::from(&root)])
            .and_then(i64::try_from)
            .unwrap();
        assert_eq!(size, 3);
    }

    #[test]
    fn mutable_dict_is_snapshotted_before_callbacks() {
        let root = Function::get_global("ffi.Dict")
            .unwrap()
            .call_packed(&[
                AnyView::from(&FfiString::from("a")),
                AnyView::from(&1i64),
                AnyView::from(&FfiString::from("b")),
                AnyView::from(&2i64),
            ])
            .unwrap();
        let captured = root.clone();
        let set_item = Function::get_global("ffi.DictSetItem").unwrap();
        let mut inserted = false;
        let mut integers = Vec::new();

        assert!(walk(&root, |value, phase| {
            if phase == Phase::Enter {
                if let Some(integer) = value.cast::<i64>() {
                    integers.push(integer);
                    if !inserted {
                        set_item
                            .call_packed(&[
                                AnyView::from(&captured),
                                AnyView::from(&FfiString::from("c")),
                                AnyView::from(&3i64),
                            ])
                            .unwrap();
                        inserted = true;
                    }
                }
            }
            WalkResult::Advance
        })
        .unwrap()
        .is_continue());

        integers.sort_unstable();
        assert_eq!(integers, vec![1, 2]);
        let size = Function::get_global("ffi.DictSize")
            .unwrap()
            .call_packed(&[AnyView::from(&root)])
            .and_then(i64::try_from)
            .unwrap();
        assert_eq!(size, 3);
    }

    #[test]
    fn interrupt_stops_without_running_remaining_callbacks() {
        let root = Array::new(vec![1i64, 2, 3]);
        let mut integers = 0;
        let outcome = walk(&root, |value, phase| {
            if phase == Phase::Enter && value.cast::<i64>().is_some() {
                integers += 1;
                return WalkResult::Interrupt;
            }
            WalkResult::Advance
        })
        .unwrap();
        assert!(outcome.is_break());
        assert_eq!(integers, 1);
    }

    #[derive(Default)]
    struct ManualRegionProbe {
        seen: Vec<DefRegionKind>,
    }

    #[crate::dispatch(visit)]
    impl ManualRegionProbe {
        fn visit_array(&mut self, array: Array<i64>, ctx: &mut VisitCtx<'_>) -> WalkResult {
            let overridden = array.get(0).unwrap();
            if !ctx.visit_with_def_region(self, &overridden, DefRegionKind::NonRecursive) {
                return WalkResult::Interrupt;
            }
            let inherited = array.get(1).unwrap();
            if !ctx.visit(self, &inherited) {
                return WalkResult::Interrupt;
            }
            WalkResult::Skip
        }

        fn visit_integer(&mut self, _value: i64, ctx: &mut VisitCtx<'_>) -> WalkResult {
            self.seen.push(ctx.def_region_kind());
            WalkResult::Advance
        }
    }

    #[test]
    fn manual_child_visit_can_override_def_region() {
        let root = Array::new(vec![7i64, 8]);
        let mut probe = ManualRegionProbe::default();
        assert!(structural_visit(&root, &mut probe).unwrap().is_continue());
        assert_eq!(
            probe.seen,
            vec![DefRegionKind::NonRecursive, DefRegionKind::None]
        );
    }

    #[derive(Default)]
    struct GenericDispatchProbe {
        integers: Vec<i64>,
        objects: usize,
        catch_all: usize,
    }

    #[crate::dispatch(visit)]
    impl GenericDispatchProbe {
        fn visit_integer(&mut self, value: i64, _ctx: &mut VisitCtx<'_>) -> WalkResult {
            self.integers.push(value);
            WalkResult::Advance
        }

        fn visit_object(
            &mut self,
            _value: &tvm_ffi::Object,
            _ctx: &mut VisitCtx<'_>,
        ) -> WalkResult {
            self.objects += 1;
            WalkResult::Advance
        }

        fn visit_any(&mut self, _value: &VisitValue, _ctx: &mut VisitCtx<'_>) -> WalkResult {
            self.catch_all += 1;
            WalkResult::Advance
        }
    }

    #[test]
    fn generated_dispatch_supports_pod_and_ordered_catch_all() {
        let root = Array::new(vec![1i64, 2]);
        let mut probe = GenericDispatchProbe::default();
        assert!(structural_visit(&root, &mut probe).unwrap().is_continue());
        assert_eq!(probe.integers, vec![1, 2]);
        assert_eq!(probe.objects, 1);

        let floats = Array::new(vec![1.0f64, 2.0]);
        assert!(structural_visit(&floats, &mut probe).unwrap().is_continue());
        assert_eq!(probe.objects, 2);
        assert_eq!(probe.catch_all, 2);
    }

    #[derive(Default)]
    struct OrderProbe {
        events: Vec<String>,
    }

    #[crate::dispatch(visit)]
    impl OrderProbe {
        fn visit_array(&mut self, _array: Array<i64>, _ctx: &mut VisitCtx<'_>) -> WalkResult {
            self.events.push("array".to_string());
            WalkResult::Advance
        }

        fn visit_integer(&mut self, value: i64, _ctx: &mut VisitCtx<'_>) -> WalkResult {
            self.events.push(format!("int:{value}"));
            WalkResult::Advance
        }
    }

    #[test]
    fn typed_dispatch_supports_post_order() {
        let root = Array::new(vec![1i64, 2]);
        let mut probe = OrderProbe::default();
        assert!(
            structural_visit_ordered(&root, &mut probe, VisitOrder::PostOrder)
                .unwrap()
                .is_continue()
        );
        assert_eq!(probe.events, vec!["int:1", "int:2", "array"]);
    }

    #[test]
    fn interrupt_payload_is_returned_to_the_caller() {
        let root = Array::new(vec![1i64, 2]);
        let outcome = walk(&root, |value, phase| {
            if phase == Phase::Enter && value.cast::<i64>() == Some(1) {
                return WalkResult::interrupt_with(42i64);
            }
            WalkResult::Advance
        })
        .unwrap();
        let ControlFlow::Break(payload) = outcome else {
            panic!("walk unexpectedly completed");
        };
        assert_eq!(i64::try_from(payload).unwrap(), 42);
    }

    #[test]
    fn handler_errors_include_native_visit_path() {
        let root = Array::new(vec![1i64]);
        let error = match walk(&root, |value, phase| {
            if phase == Phase::Enter && value.cast::<i64>().is_some() {
                Err(runtime_error("handler failed"))
            } else {
                Ok(WalkResult::Advance)
            }
        }) {
            Err(error) => error,
            Ok(_) => panic!("handler unexpectedly succeeded"),
        };
        assert_eq!(error.message(), "handler failed");
        assert!(error.backtrace().contains("sequence item [0]"));
        assert!(error.backtrace().contains("object `ffi.Array`"));
    }

    #[test]
    fn raw_walk_context_receives_def_region() {
        let root = Array::new(vec![1i64]);
        let mut regions = Vec::new();
        assert!(walk_with_context(&root, |_value, phase, region| {
            if phase == Phase::Enter {
                regions.push(region);
            }
            WalkResult::Advance
        })
        .unwrap()
        .is_continue());
        assert_eq!(regions, vec![DefRegionKind::None; 2]);
    }
}
