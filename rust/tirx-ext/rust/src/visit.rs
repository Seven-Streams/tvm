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

//! Rust binding of tvm-ffi's **structural visit** — a visitor + function table.
//!
//! C++ (`tvm/ffi/extra/structural_visit.h`) drives traversal through an ABI
//! pair: a `StructuralVisitorObj` (an FFI object whose first own field is a
//! pointer to a `StructuralVisitorVTable`) and that vtable's single `visit`
//! function pointer, called for every value in the graph. Subclassing =
//! swapping the vtable. The *default* vtable entry implements the actual
//! recursion: it dispatches on the registered `kStructuralVisit` type
//! attribute (arrays, maps) or walks the reflected fields of an object, and
//! recurses into children through `visitor->vtable_->visit` again.
//!
//! This module builds a visitor subclass **in Rust** the same way the C++
//! `StructuralWalkCallbackVisitorObj` template does:
//!
//! * [`StructuralVisitorVTable`] / [`FStructuralVisit`] mirror the C ABI;
//! * a heap-backed `#[repr(C)]` `CallbackVisitor` extends the C++ object
//!   layout (`Object` header, `vtable_`, `def_region_mode_`) with an opaque
//!   pointer to the active Rust callback frame;
//! * its vtable entry `callback_visit` runs the `pre` closure, then chains
//!   to the **default** visit function — harvested at runtime from a
//!   C++-constructed `ffi.StructuralVisitor` — for the reflected recursion
//!   into children (which re-enters our vtable), then runs `post`.
//!
//! So all reflection/traversal logic stays in libtvm_ffi; Rust only supplies
//! the per-node callback through the same function table C++ subclasses use.

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::ops::ControlFlow;
use std::os::raw::c_void;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use tvm_ffi::any::{Any, AnyView};
use tvm_ffi::error::Error;
use tvm_ffi::function::Function;
use tvm_ffi::string::String as FfiString;
use tvm_ffi::tvm_ffi_sys::TVMFFIObjectDeleterFlagBitMask::{
    kTVMFFIObjectDeleterFlagBitMaskStrong, kTVMFFIObjectDeleterFlagBitMaskWeak,
};
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIErrorCell, TVMFFIFieldGetter, TVMFFIObject, TVMFFITypeIndex,
    COMBINED_REF_COUNT_BOTH_ONE, COMBINED_REF_COUNT_MASK_U32,
};

use crate::object_ref::is_instance;
use crate::runtime::{call_packed_global, lookup_type_index};

// ---------------------------------------------------------------------------
// ABI mirrors
// ---------------------------------------------------------------------------

/// C++ `tvm::ffi::FStructuralVisit`: `TVMFFIAny (*)(StructuralVisitorObj*,
/// AnyView) noexcept`. Both `AnyView` (a single trivially-copyable
/// `TVMFFIAny`) and the returned `TVMFFIAny` are 16-byte PODs, so the Itanium
/// C++ call matches this `extern "C"` signature. The returned `TVMFFIAny`
/// encodes `Expected<Optional<VisitInterrupt>>`: FFI None to continue, an
/// owned `ffi.VisitInterrupt` to halt, an owned `ffi.Error` on failure.
pub type FStructuralVisit =
    unsafe extern "C" fn(visitor: *mut c_void, value: TVMFFIAny) -> TVMFFIAny;

/// C++ `tvm::ffi::StructuralVisitorVTable` — the function table: one slot.
#[repr(C)]
pub struct StructuralVisitorVTable {
    /// Visit callback (nullable in the ABI, never null on a live visitor).
    pub visit: Option<FStructuralVisit>,
}

/// Layout prefix of C++ `StructuralVisitorObj`:
/// `{ Object (24B); const StructuralVisitorVTable* vtable_ (@24);
///    TVMFFIDefRegionKind def_region_mode_ (@32) }`.
/// `def_region_mode_` is written by C++ (`WithDefRegionKind`) during the
/// reflected walk, hence the `Cell`.
#[repr(C)]
struct VisitorPrefix {
    header: TVMFFIObject,
    vtable: *const StructuralVisitorVTable,
    def_region_mode: Cell<i32>,
}

/// Full C ABI prefix of `ffi.Error`.
///
/// The vendored Rust `TVMFFIErrorCell` binding still describes the original
/// prefix ending at `update_backtrace`, while the pinned C ABI appends the two
/// owned object handles `cause_chain` and `extra_context`.  Structural-visit
/// breadcrumbs live in the latter, so mirror only those ABI-mandated trailing
/// slots here.  No Rust reference is ever formed to the C++ `ErrorObj`.
#[repr(C)]
struct ErrorObjectWithContext {
    header: TVMFFIObject,
    cell_prefix: TVMFFIErrorCell,
    cause_chain: *mut c_void,
    extra_context: *mut c_void,
}

const _: () = {
    assert!(std::mem::size_of::<TVMFFIObject>() == 24);
    assert!(std::mem::offset_of!(VisitorPrefix, vtable) == 24);
    assert!(std::mem::offset_of!(VisitorPrefix, def_region_mode) == 32);
    assert!(std::mem::size_of::<TVMFFIAny>() == 16);
    assert!(std::mem::size_of::<AnyView<'_>>() == std::mem::size_of::<TVMFFIAny>());
    assert!(std::mem::align_of::<AnyView<'_>>() == std::mem::align_of::<TVMFFIAny>());
    assert!(
        std::mem::offset_of!(ErrorObjectWithContext, cause_chain)
            == std::mem::size_of::<TVMFFIObject>() + std::mem::size_of::<TVMFFIErrorCell>()
    );
    assert!(
        std::mem::offset_of!(ErrorObjectWithContext, extra_context)
            == std::mem::offset_of!(ErrorObjectWithContext, cause_chain)
                + std::mem::size_of::<*mut c_void>()
    );
    // This module already targets the pinned 64-bit StructuralVisitor ABI.
    // Absolute offsets make a future tvm-ffi-sys update that extends its
    // TVMFFIErrorCell binding fail loudly instead of duplicating these slots.
    assert!(std::mem::offset_of!(ErrorObjectWithContext, cause_chain) == 80);
    assert!(std::mem::offset_of!(ErrorObjectWithContext, extra_context) == 88);
};

// ---------------------------------------------------------------------------
// Harvesting the default (reflected-recursion) visit function
// ---------------------------------------------------------------------------

/// The default `visit` implementation (C++ `StructuralVisitorObj::DispatchVisit`),
/// read out of the vtable of a C++-constructed `ffi.StructuralVisitor`. This is
/// the function that owns all recursion logic — `kStructuralVisit` container
/// hooks, reflected-field walking, def-region scoping — and calls back into
/// `visitor->vtable_->visit` (i.e. *our* table) for every child.
///
/// OnceLock (not LazyLock) so a panicking harvest — libtvm_ffi not fully set
/// up yet — leaves the cell uninitialized and retries on the next walk instead
/// of poisoning every future walk.
static DEFAULT_VISIT: OnceLock<FStructuralVisit> = OnceLock::new();

fn default_visit_fn() -> FStructuralVisit {
    *DEFAULT_VISIT.get_or_init(|| unsafe { harvest_default_visit() })
}

unsafe fn harvest_default_visit() -> FStructuralVisit {
    let sv_index = lookup_type_index("ffi.StructuralVisitor");
    let probe = || -> (Any, *const VisitorPrefix) {
        let mut v = call_packed_global("ffi.StructuralVisitor", &[]);
        assert_eq!(
            v.type_index(),
            sv_index,
            "ffi.StructuralVisitor() returned an unexpected type"
        );
        let raw = Any::as_data_ptr(&mut v);
        let obj = (*raw).data_union.v_obj as *const VisitorPrefix;
        (v, obj)
    };
    // Two independent instances must agree on the (static, immortal) vtable —
    // a cheap runtime proof that `vtable_` really lives at offset 24.
    let (_g1, p1) = probe();
    let (_g2, p2) = probe();
    let (vt1, vt2) = ((*p1).vtable, (*p2).vtable);
    assert!(
        !vt1.is_null() && vt1 == vt2,
        "StructuralVisitorObj layout probe failed: vtable_ not at offset 24"
    );
    assert_eq!(
        (*p1).def_region_mode.get(),
        0,
        "StructuralVisitorObj layout probe failed: def_region_mode_ not at offset 32"
    );
    (*vt1)
        .visit
        .expect("default StructuralVisitorVTable has a null visit slot")
}

// ---------------------------------------------------------------------------
// The Rust callback visitor (a StructuralVisitorObj "subclass")
// ---------------------------------------------------------------------------

/// What the callback tells the traversal to do next, mirroring the C++
/// `WalkResult` actions.
pub enum WalkResult {
    /// Continue: on [`Phase::Enter`], visit this node's children.
    Advance,
    /// On [`Phase::Enter`], skip this node's children (its [`Phase::Exit`]
    /// call is skipped too). On [`Phase::Exit`] this is the same as `Advance`.
    Skip,
    /// Halt the entire traversal with a payload-less `ffi.VisitInterrupt`
    /// (C++ `WalkResult::Interrupt()`; the payload channel is not bound yet).
    Interrupt,
}

/// Whether the callback fires before or after a value's children — pairing
/// `Enter`/`Exit` of the same node gives scope tracking (e.g. a running
/// product of enclosing loop extents) that a bare pre-order walk cannot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Before the value's children (the C++ pre-order callback position).
    Enter,
    /// After the value's children (the C++ post-order callback position).
    Exit,
}

/// A borrowed view of the value currently being visited (any field/element the
/// reflected walk reaches: objects, ints, strings, FFI None, ...).
#[repr(transparent)]
pub struct VisitValue(TVMFFIAny);

impl VisitValue {
    /// Wrap a raw (borrowed) FFI value — lets the mutator engine reuse this
    /// module's typed `cast` for its own function-table dispatch.
    #[inline]
    pub(crate) fn from_raw(raw: TVMFFIAny) -> Self {
        VisitValue(raw)
    }

    /// The FFI type index of the value (`0` == FFI None).
    #[inline]
    pub fn type_index(&self) -> i32 {
        self.0.type_index
    }

    /// Convert the value into an owned reference-class handle `R` (`For`,
    /// `IfThenElse`, `Stmt`, … — also PODs like `i64`): the typed counterpart
    /// of C++ `AnyView::as<T>`. Subtypes match; FFI None never matches (the
    /// underlying check is pointer-style). Costs one refcount increment on
    /// success, in exchange for a handle that can be stored and cloned.
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

    /// Borrow the value as node type `N` if it is one (or a subtype). Returns
    /// `None` for FFI None and non-objects — the pointer-style type test, so
    /// unset `Optional` fields can never match (unlike a nullable-ref test).
    #[inline]
    pub fn as_node<N: tvm_ffi::object::ObjectCore>(&self) -> Option<&N> {
        if self.0.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            return None;
        }
        if !is_instance(self.0.type_index, N::type_index()) {
            return None;
        }
        Some(unsafe { &*(self.0.data_union.v_obj as *const N) })
    }
}

/// Hands the low-level FFI visitor to a [`structural_visit_manual`] callback,
/// so it can drive child visits explicitly — the Rust face of calling
/// `visitor->Visit(child)` / `DefaultVisit(value)` inside a C++
/// `StructuralVisitorObj` subclass.
pub struct ManualVisitCtx {
    visitor: *mut c_void,
    /// The value the callback is currently visiting.
    current: TVMFFIAny,
    /// A halt (owned `ffi.VisitInterrupt` or `ffi.Error`) raised by a manual
    /// child visit, pending propagation by the trampoline.
    halted: Option<PendingRawHalt>,
}

/// An owned raw halt plus whether the callback wrapper still needs to append
/// its current node to an Error's VisitErrorContext.
struct PendingRawHalt {
    value: TVMFFIAny,
    needs_current_error_context: bool,
}

impl ManualVisitCtx {
    /// Visit `child` right now, with full dispatch (the callback fires for the
    /// whole subtree). Returns `false` when the walk has been halted (a
    /// deeper callback returned [`WalkResult::Interrupt`], or the FFI side
    /// failed) — stop doing work and return; the halt propagates automatically.
    pub fn visit<T>(&mut self, child: &T) -> bool
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        if self.halted.is_some() {
            return false;
        }
        let view = [AnyView::from(child)];
        let raw = unsafe { std::ptr::read(view.as_ptr() as *const TVMFFIAny) };
        self.dispatch(raw)
    }

    /// Run the **default reflected recursion** over the current value's
    /// children, right now — the equivalent of a C++ subclass calling
    /// `DefaultVisit(value)` mid-body. Each child re-enters the callback.
    /// Returns `false` when the walk has been halted.
    pub fn visit_children(&mut self) -> bool {
        if self.halted.is_some() {
            return false;
        }
        let me = self.visitor as *const CallbackVisitor;
        let state = unsafe {
            active_callback_state(me).expect("manual visit context used outside its callback")
        };
        let ret = unsafe { ((*state).default_visit)(self.visitor, self.current) };
        // The harvested default dispatch already appends `current` when an
        // Error comes back from one of its reflected children.
        self.absorb(ret, false)
    }

    /// The active FFI definition-region mode.  A higher-level visitor that
    /// opens a nested traversal frame carries this value into that frame so
    /// manual recursion preserves the surrounding reflection context.
    fn def_region_mode(&self) -> i32 {
        let me = self.visitor as *const CallbackVisitor;
        unsafe { (*me).prefix.def_region_mode.get() }
    }

    /// Dispatch `raw` through the visitor's function table.
    fn dispatch(&mut self, raw: TVMFFIAny) -> bool {
        let ret = unsafe { callback_visit(self.visitor, raw) };
        // This callback frame is outside the explicitly visited child, so it
        // must append its own current node while forwarding an Error.
        self.absorb(ret, true)
    }

    /// Record a halt returned by a child visit; `true` == keep going.
    fn absorb(&mut self, ret: TVMFFIAny, needs_current_error_context: bool) -> bool {
        if ret.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            true
        } else {
            self.halted = Some(PendingRawHalt {
                value: ret,
                needs_current_error_context,
            });
            false
        }
    }
}

/// The uniform callback shape both entry points lower to. `Fn` (not `FnMut`)
/// because [`ManualVisitCtx::visit`] re-enters the callback while an outer
/// call is still on the stack — re-entrancy needs a shared-ref callable; mutable
/// callback state goes in `Cell`s (≈ a C++ `const` method with `mutable`
/// members).
type DynCallback<'a> = &'a dyn Fn(&VisitValue, Phase, &mut ManualVisitCtx) -> WalkResult;

/// The borrowed state of one active driver call.  It deliberately lives on
/// the Rust stack, outside the ref-counted FFI object, and is reachable only
/// while [`CallbackVisitor::active_state`] is non-null.
struct ActiveCallbackState<'a> {
    callback: DynCallback<'a>,
    default_visit: FStructuralVisit,
    error_context: &'a VisitErrorContextResources,
    panic: Option<Box<dyn std::any::Any + Send + 'static>>,
}

/// Our visitor object: the C++-visible [`VisitorPrefix`] followed by a pointer
/// to the currently active Rust frame and one owned interrupt prototype.
///
/// Unlike the callback frame, this object is heap allocated and uses the
/// ordinary tvm-ffi strong/weak reference-count protocol.  A structural hook
/// may therefore retain the visitor handle without leaving a dangling pointer.
/// The driver clears `active_state` before its borrowed frame goes away; a
/// retained visitor is then inert and every later visit returns an interrupt.
#[repr(C)]
struct CallbackVisitor {
    prefix: VisitorPrefix,
    active_state: Cell<*mut c_void>,
    /// A pre-built payload-less `ffi.VisitInterrupt`, constructed in a caught
    /// driver setup region so the extern "C" paths below only ever *clone* it
    /// — a refcount increment that cannot fail — instead of calling into the
    /// FFI registry from a frame where a panic would abort the host process.
    /// `ManuallyDrop` permits the strong and weak halves of the tvm-ffi
    /// deleter protocol to run in two separate calls.
    interrupt_proto: ManuallyDrop<Any>,
}

const _: () = assert!(std::mem::offset_of!(CallbackVisitor, prefix) == 0);

/// Release the heap visitor according to tvm-ffi's two-phase object lifetime.
/// Strong destruction releases owned Rust fields; weak destruction frees the
/// allocation that still contains the object header.
unsafe extern "C" fn callback_visitor_deleter(self_ptr: *mut c_void, flags: i32) {
    let me = self_ptr as *mut CallbackVisitor;
    if flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0 {
        ManuallyDrop::drop(&mut (*me).interrupt_proto);
    }
    if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
        drop(Box::from_raw(me));
    }
}

/// Load the type-erased active callback frame.  No Rust reference is formed:
/// recursive visits may re-enter while an outer callback is still live.
unsafe fn active_callback_state<'a>(
    me: *const CallbackVisitor,
) -> Option<*mut ActiveCallbackState<'a>> {
    let state = (*me).active_state.get();
    (!state.is_null()).then_some(state.cast())
}

/// The single function-table entry of the Rust visitor. Re-entered for every
/// child (by the default chain and by [`ManualVisitCtx::visit`]); each frame
/// only takes short-lived reborrows through the raw `me` pointer.
unsafe extern "C" fn callback_visit(visitor: *mut c_void, value: TVMFFIAny) -> TVMFFIAny {
    let me = visitor as *mut CallbackVisitor;
    let Some(state) = active_callback_state(me) else {
        return make_interrupt_raw(me);
    };
    // The C++ walk visitor short-circuits FFI None without invoking callbacks.
    if value.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return TVMFFIAny::new();
    }
    let view = VisitValue(value);
    let mut ctx = ManualVisitCtx {
        visitor,
        current: value,
        halted: None,
    };

    match catch_unwind(AssertUnwindSafe(|| {
        ((*state).callback)(&view, Phase::Enter, &mut ctx)
    })) {
        Ok(flow) => {
            // A halt raised by a manual child visit outranks the flow verdict.
            if let Some(halt) = ctx.halted.take() {
                return propagate_pending_halt(state, value, halt);
            }
            match flow {
                WalkResult::Advance => {}
                WalkResult::Skip => return TVMFFIAny::new(),
                WalkResult::Interrupt => return make_interrupt_raw(me),
            }
        }
        Err(p) => {
            stash_first_panic(state, p);
            release_pending_halt_opt(ctx.halted.take());
            return make_interrupt_raw(me);
        }
    }

    // Chain to tvm-ffi's default visit: reflected recursion into children.
    let ret = ((*state).default_visit)(visitor, value);
    if ret.type_index != TVMFFITypeIndex::kTVMFFINone as i32 {
        return ret; // owned VisitInterrupt or Error — forward as-is
    }

    match catch_unwind(AssertUnwindSafe(|| {
        ((*state).callback)(&view, Phase::Exit, &mut ctx)
    })) {
        Ok(flow) => {
            if let Some(halt) = ctx.halted.take() {
                return propagate_pending_halt(state, value, halt);
            }
            match flow {
                WalkResult::Interrupt => make_interrupt_raw(me),
                _ => TVMFFIAny::new(),
            }
        }
        Err(p) => {
            stash_first_panic(state, p);
            release_pending_halt_opt(ctx.halted.take());
            make_interrupt_raw(me)
        }
    }
}

/// Preserve the first panic raised during one low-level visitor frame.
///
/// Manual recursion re-enters [`callback_visit`] before the outer callback has
/// returned.  If the outer callback panics after an inner panic was already
/// stashed, assigning into the occupied slot would drop the first arbitrary
/// panic payload inside this `extern "C"` frame.  Keep the first payload for
/// [`drive_raw`] to resume on the Rust side of the ABI boundary.  A later
/// payload is destroyed inside another caught Rust region.  If that destructor
/// itself panics, only the replacement panic payload is forgotten so no unwind
/// can cross the C ABI boundary.
unsafe fn stash_first_panic(
    state: *mut ActiveCallbackState<'_>,
    panic: Box<dyn std::any::Any + Send + 'static>,
) {
    if (*state).panic.is_none() {
        (*state).panic = Some(panic);
    } else {
        if let Err(destructor_panic) = catch_unwind(AssertUnwindSafe(|| drop(panic))) {
            std::mem::forget(destructor_panic);
        }
    }
}

/// A payload-less `ffi.VisitInterrupt` as an owned raw `TVMFFIAny`: a clone
/// (refcount inc) of the prototype the driver built up front — infallible, as
/// required inside the extern "C" callback.
unsafe fn make_interrupt_raw(me: *const CallbackVisitor) -> TVMFFIAny {
    Any::into_raw_ffi_any((&*(*me).interrupt_proto).clone())
}

/// Append `value` to an Error returned by a callback-controlled child visit.
///
/// C++'s default dispatch adds this breadcrumb when its own recursive call
/// returns an Error.  A manual `ctx.visit(...)` returns through the callback
/// before default dispatch runs at the current level, so it needs the same
/// update here.  Context enrichment is diagnostic only: failure or panic in
/// the helper must preserve and return the original owned Error unchanged,
/// and must never unwind through the `extern "C"` boundary.
unsafe fn propagate_pending_halt(
    state: *mut ActiveCallbackState<'_>,
    value: TVMFFIAny,
    halt: PendingRawHalt,
) -> TVMFFIAny {
    if halt.needs_current_error_context
        && halt.value.type_index == TVMFFITypeIndex::kTVMFFIError as i32
        && value.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32
    {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            // Any secondary ffi.Error is dropped here.  It must not replace
            // the traversal Error whose breadcrumb we are trying to enrich.
            let _ = (*state).error_context.append_to_error(&halt.value, &value);
        }));
    }
    halt.value
}

/// Reinterpret one borrowed ABI cell as tvm-ffi's transparent `AnyView`.
/// The size/alignment equality is asserted above; the returned view never
/// owns or outlives the raw value it references.
unsafe fn borrowed_any_view(raw: &TVMFFIAny) -> AnyView<'_> {
    std::ptr::read(raw as *const TVMFFIAny as *const AnyView<'_>)
}

/// Make a non-owning raw Any cell for an object handle owned elsewhere.
unsafe fn borrowed_object_raw(handle: *mut c_void) -> TVMFFIAny {
    let object = handle as *mut TVMFFIObject;
    let mut raw = TVMFFIAny::new();
    raw.type_index = (*object).type_index;
    raw.data_union.v_obj = object;
    raw
}

/// Release one object-handle ownership slot after it has been replaced.
unsafe fn release_object_handle(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Any::from_raw_ffi_any(borrowed_object_raw(handle)));
    }
}

/// Load the low 32-bit strong-reference count from an object header.
unsafe fn object_handle_strong_count(handle: *mut c_void) -> u64 {
    (*(handle as *mut TVMFFIObject))
        .combined_ref_count
        .load(Ordering::Relaxed)
        & COMBINED_REF_COUNT_MASK_U32
}

fn error_strong_count(error: &Error) -> u64 {
    let raw = borrowed_raw(error);
    debug_assert_eq!(raw.type_index, TVMFFITypeIndex::kTVMFFIError as i32);
    unsafe { object_handle_strong_count(raw.data_union.v_obj as *mut c_void) }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Structurally visit every value reachable from `root`, in the same order and
/// with the same coverage as C++ `StructuralWalk` (reflected fields, arrays,
/// maps; fields marked SEqHash-ignored are skipped).
///
/// The callback runs twice per value — [`Phase::Enter`] before its children
/// and [`Phase::Exit`] after them (`Exit` is skipped when `Enter` returned
/// [`WalkResult::Skip`]) — and steers the walk via [`WalkResult`].
///
/// Returns `Ok(None)` when the walk completed, `Ok(Some(interrupt))` when the
/// callback returned [`WalkResult::Interrupt`] (the owned `ffi.VisitInterrupt`),
/// or the FFI error if the C++ side failed. A panic in the callback halts the
/// walk and resumes unwinding here.
pub fn walk<R>(
    root: &R,
    mut callback: impl FnMut(&VisitValue, Phase) -> WalkResult,
) -> Result<Option<Any>, tvm_ffi::error::Error>
where
    for<'x> AnyView<'x>: From<&'x R>,
{
    // Plain walks never re-enter the callback (they don't use the ctx), so an
    // `FnMut` is fine — adapt it to the re-entrant `Fn` shape via `RefCell`.
    let resources = DriveResources::new()?;
    drive_non_reentrant_raw(
        borrowed_raw(root),
        0,
        RootAction::Dispatch,
        &resources,
        move |v, phase, _ctx| callback(v, phase),
    )
}

/// Low-level manual callback API, distinct from the stateful [`Visitor`].  The
/// callback receives a [`ManualVisitCtx`] and can take full control of a
/// node's visit order by visiting chosen children explicitly and returning
/// [`WalkResult::Skip`].
///
/// This entry point directly re-enters the same callback, so it must be `Fn`;
/// code that owns mutable pass state should use [`Visitor`] instead, whose
/// handlers receive the pass object as `&mut V` and recurse with safe
/// reborrows.  An FFI structural hook may retain the low-level visitor handle
/// safely, but the handle is traversal-scoped: after this function returns it
/// is inert and any later visit through it yields `ffi.VisitInterrupt` without
/// accessing the expired callback frame.  The upstream contract still forbids
/// overlapping traversals through the same visitor handle.
pub fn structural_visit_manual<R>(
    root: &R,
    callback: impl Fn(&VisitValue, Phase, &mut ManualVisitCtx) -> WalkResult,
) -> Result<Option<Any>, tvm_ffi::error::Error>
where
    for<'x> AnyView<'x>: From<&'x R>,
{
    drive(root, &callback)
}

/// Shared driver: create one ref-counted visitor and a stack-local callback
/// frame, run the walk from `root`, then translate the ABI result (None /
/// interrupt / error / stashed panic).
fn drive<R>(root: &R, callback: DynCallback) -> Result<Option<Any>, tvm_ffi::error::Error>
where
    for<'x> AnyView<'x>: From<&'x R>,
{
    let resources = DriveResources::new()?;
    drive_raw(
        borrowed_raw(root),
        callback,
        0,
        RootAction::Dispatch,
        &resources,
    )
}

/// Pieces shared by all nested frames of one high-level visitor run.  In
/// particular, explicitly visiting a child must not look up and construct a
/// new interrupt prototype for every edge in the tree.
struct DriveResources {
    default_visit: FStructuralVisit,
    interrupt_proto: Any,
    error_context: VisitErrorContextResources,
}

impl DriveResources {
    fn new() -> Result<Self, Error> {
        Ok(Self {
            default_visit: default_visit_fn(),
            interrupt_proto: Function::get_global("ffi.VisitInterrupt")?
                .call_packed(&[AnyView::new()])?,
            error_context: VisitErrorContextResources::new()?,
        })
    }
}

/// Pre-resolved operations needed to mirror C++
/// `details::UpdateVisitErrorContext` without making fallible registry lookups
/// from the `extern "C"` callback.
struct VisitErrorContextResources {
    type_index: i32,
    reverse_visit_pattern_offset: usize,
    reverse_visit_pattern_getter: TVMFFIFieldGetter,
    list: Function,
    list_append: Function,
    make_object: Function,
    reverse_visit_pattern_name: FfiString,
    prev_error_context_name: FfiString,
}

impl VisitErrorContextResources {
    fn new() -> Result<Self, Error> {
        let type_index = lookup_type_index("ffi.VisitErrorContext");
        let (offset, getter) = unsafe {
            crate::reflect::for_each_field(type_index, |field| {
                if field.name.as_str() == "reverse_visit_pattern" {
                    ControlFlow::Break((
                        usize::try_from(field.offset)
                            .expect("negative ffi.VisitErrorContext field offset"),
                        field
                            .getter
                            .expect("ffi.VisitErrorContext field has no getter"),
                    ))
                } else {
                    ControlFlow::Continue(())
                }
            })
        }
        .expect("ffi.VisitErrorContext.reverse_visit_pattern is not reflected");

        Ok(Self {
            type_index,
            reverse_visit_pattern_offset: offset,
            reverse_visit_pattern_getter: getter,
            list: Function::get_global("ffi.List")?,
            list_append: Function::get_global("ffi.ListAppend")?,
            make_object: Function::get_global("ffi.MakeObjectFromPackedArgs")?,
            reverse_visit_pattern_name: FfiString::from("reverse_visit_pattern"),
            prev_error_context_name: FfiString::from("prev_error_context"),
        })
    }

    /// Apply the same in-place update as C++ `UpdateVisitErrorContext`.
    /// `error_raw` owns the Error, while `node_raw` is borrowed for the active
    /// callback frame.  This method neither consumes nor clones the Error.
    unsafe fn append_to_error(
        &self,
        error_raw: &TVMFFIAny,
        node_raw: &TVMFFIAny,
    ) -> Result<(), Error> {
        debug_assert_eq!(error_raw.type_index, TVMFFITypeIndex::kTVMFFIError as i32);
        debug_assert!(node_raw.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32);

        let error_obj = error_raw.data_union.v_obj as *mut ErrorObjectWithContext;
        let strong_count = object_handle_strong_count(error_obj as *mut c_void);
        if strong_count != 1 {
            // C++ UpdateVisitErrorContext mutates ErrorObj in place under the
            // explicit assumption that the propagating Error is single-owned.
            // A caller may retain a clone before `ctx.fail(error)`; in that
            // case enriching this Error would also mutate the retained clone.
            // Preserve the shared Error exactly and omit this best-effort
            // breadcrumb instead.
            return Ok(());
        }
        let previous = (*error_obj).extra_context;

        if !previous.is_null() && (*(previous as *mut TVMFFIObject)).type_index == self.type_index {
            if object_handle_strong_count(previous) != 1 {
                // VisitErrorContext is an explicitly mutable FFI object, but
                // this Rust-managed callback avoids mutating a separately
                // retained handle.  Native reflected ancestors may still
                // append under VisitErrorContext's documented shared-mutable
                // C++ semantics.  Breadcrumbs here are best-effort.
                return Ok(());
            }
            // Existing VisitErrorContext: retrieve its owned List field using
            // reflection, append the current node, then drop our temporary
            // list handle.  The context retains its own list ownership.
            let field_addr = (previous as *mut u8).add(self.reverse_visit_pattern_offset);
            let mut pattern = Any::new();
            let getter_result = (self.reverse_visit_pattern_getter)(
                field_addr as *mut c_void,
                Any::as_data_ptr(&mut pattern),
            );
            if getter_result != 0 {
                return Err(Error::from_raised());
            }
            self.list_append
                .call_packed(&[AnyView::from(&pattern), borrowed_any_view(node_raw)])?;
            return Ok(());
        }

        // No VisitErrorContext yet: seed one with this node and preserve an
        // unrelated pre-existing extra_context in `prev_error_context`.
        let list = self.list.call_packed(&[borrowed_any_view(node_raw)])?;
        let previous_raw = if previous.is_null() {
            TVMFFIAny::new()
        } else {
            borrowed_object_raw(previous)
        };
        let context = self.make_object.call_packed(&[
            AnyView::from(&self.type_index),
            AnyView::from(&self.reverse_visit_pattern_name),
            AnyView::from(&list),
            AnyView::from(&self.prev_error_context_name),
            borrowed_any_view(&previous_raw),
        ])?;
        assert_eq!(
            context.type_index(),
            self.type_index,
            "ffi.MakeObjectFromPackedArgs returned the wrong context type"
        );

        // Transfer the new context's one owned handle into ErrorObj, then
        // release the ownership previously held by ErrorObj.  If `previous`
        // was non-null, the new context has already retained it through its
        // `prev_error_context` field.
        let context_raw = Any::into_raw_ffi_any(context);
        let new_context = context_raw.data_union.v_obj as *mut c_void;
        (*error_obj).extra_context = new_context;
        release_object_handle(previous);
        Ok(())
    }
}

/// Whether a new traversal frame dispatches its root or only runs the default
/// reflected recursion over the root's children.
#[derive(Clone, Copy)]
enum RootAction {
    Dispatch,
    Children,
}

/// Copy the borrowed FFI cell represented by `value`.  The result never owns
/// the referenced object; its lifetime is bounded by the caller's `&value`.
fn borrowed_raw<R>(value: &R) -> TVMFFIAny
where
    for<'x> AnyView<'x>: From<&'x R>,
{
    let view = [AnyView::from(value)];
    unsafe { std::ptr::read(view.as_ptr() as *const TVMFFIAny) }
}

/// Run a callback that is temporally non-reentrant.  The FFI engine still
/// requires an `Fn`, so this adapter retains the low-level callback `RefCell`;
/// high-level visitor state is *not* stored in it.  Manual high-level recursion
/// opens a distinct frame and explicitly reborrows its `&mut visitor`.
fn drive_non_reentrant_raw(
    root_raw: TVMFFIAny,
    initial_def_region_mode: i32,
    root_action: RootAction,
    resources: &DriveResources,
    callback: impl FnMut(&VisitValue, Phase, &mut ManualVisitCtx) -> WalkResult,
) -> Result<Option<Any>, Error> {
    let callback = RefCell::new(callback);
    drive_raw(
        root_raw,
        &move |value, phase, ctx| (callback.borrow_mut())(value, phase, ctx),
        initial_def_region_mode,
        root_action,
        resources,
    )
}

/// Allocate one ref-counted FFI visitor and translate its ABI result.  Nested
/// high-level visits reuse `resources` but get their own callback frame and
/// FFI handle.
fn drive_raw<'a>(
    root_raw: TVMFFIAny,
    callback: DynCallback<'a>,
    initial_def_region_mode: i32,
    root_action: RootAction,
    resources: &'a DriveResources,
) -> Result<Option<Any>, Error> {
    static VTABLE: StructuralVisitorVTable = StructuralVisitorVTable {
        visit: Some(callback_visit),
    };

    let mut state = ActiveCallbackState {
        callback,
        default_visit: resources.default_visit,
        error_context: &resources.error_context,
        panic: None,
    };
    let vis = Box::new(CallbackVisitor {
        prefix: VisitorPrefix {
            header: TVMFFIObject {
                combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                type_index: lookup_type_index("ffi.StructuralVisitor"),
                __padding: 0,
                deleter: Some(callback_visitor_deleter),
            },
            vtable: &VTABLE,
            def_region_mode: Cell::new(initial_def_region_mode),
        },
        active_state: Cell::new((&mut state as *mut ActiveCallbackState<'a>).cast()),
        interrupt_proto: ManuallyDrop::new(resources.interrupt_proto.clone()),
    });

    let default_visit = state.default_visit;
    let visitor = Box::into_raw(vis);
    // Consume the initial strong reference.  If an FFI hook retains another
    // one, dropping this owner leaves the heap allocation alive but inert.
    let visitor_owner = unsafe { Any::from_raw_ffi_any(borrowed_object_raw(visitor.cast())) };
    let ret = unsafe {
        match root_action {
            RootAction::Dispatch => callback_visit(visitor.cast(), root_raw),
            RootAction::Children => default_visit(visitor.cast(), root_raw),
        }
    };
    // Invalidate every externally retained handle before any borrowed field
    // in `state` can be dropped or unwound through.
    unsafe { (*visitor).active_state.set(std::ptr::null_mut()) };

    if let Some(p) = state.panic.take() {
        release_raw_any(ret);
        drop(visitor_owner);
        resume_unwind(p);
    }
    drop(visitor_owner);
    if ret.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(None);
    }
    let any = unsafe { Any::from_raw_ffi_any(ret) };
    if any.type_index() == TVMFFITypeIndex::kTVMFFIError as i32 {
        Err(tvm_ffi::error::Error::try_from(any).expect("ffi.Error cast cannot fail"))
    } else {
        Ok(Some(any))
    }
}

/// Drop an owned raw `TVMFFIAny` (releases the strong ref if it holds one).
fn release_raw_any(raw: TVMFFIAny) {
    if raw.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        drop(unsafe { Any::from_raw_ffi_any(raw) });
    }
}

fn release_pending_halt_opt(halt: Option<PendingRawHalt>) {
    if let Some(halt) = halt {
        release_raw_any(halt.value);
    }
}

// ---------------------------------------------------------------------------
// The stateful Visitor agent
// ---------------------------------------------------------------------------

/// The compiled dispatcher: `None` for values outside the claimed set.
///
/// A handler receives the visitor object itself through `&mut V`.  Method
/// items such as `Counter::visit_for` therefore fit this function pointer
/// directly; a free function with the same signature remains valid too.
pub type Dispatch<V> =
    for<'ctx> fn(&mut V, &VisitValue, &mut VisitCtx<'ctx, V>) -> Option<WalkResult>;

/// The function table: one compiled `fn` pointer — nothing built at run
/// time, nothing boxed.
pub struct FunctionTable<V>(pub Dispatch<V>);

/// Typed dispatch implemented by a visitor object.
///
/// [`crate::dispatch`] generates this implementation from the
/// `visit_*` methods in one annotated inherent `impl`.  The method only
/// selects a handler for the current value.  Recursion remains the job of the
/// structural walker and [`VisitCtx`].
pub trait VisitDispatch: Sized {
    /// Return `Some` when this visitor claims `value`; `None` asks the
    /// structural walker to recurse through the value's reflected children.
    fn dispatch_visit(
        &mut self,
        value: &VisitValue,
        ctx: &mut VisitCtx<'_, Self>,
    ) -> Option<WalkResult>;
}

/// An owned halt raised by a fresh nested traversal frame.  It is moved back
/// into the current low-level FFI return channel so C++ sees the exact Error or
/// VisitInterrupt while unwinding its structural recursion.
enum VisitorHaltValue {
    Interrupt(Any),
    Error(Error),
}

struct VisitorHalt {
    value: VisitorHaltValue,
    needs_current_error_context: bool,
}

impl VisitorHalt {
    /// Move the owned halt back into the FFI return channel.  In particular,
    /// an Error must remain an Error while unwinding through C++ reflected
    /// frames so they can append structural-visit context.
    fn into_pending_raw(self) -> PendingRawHalt {
        let value = match self.value {
            VisitorHaltValue::Interrupt(interrupt) => interrupt,
            VisitorHaltValue::Error(error) => Any::from(error),
        };
        PendingRawHalt {
            value: unsafe { Any::into_raw_ffi_any(value) },
            needs_current_error_context: self.needs_current_error_context,
        }
    }
}

/// Per-callback traversal context for a stateful [`Visitor`].
///
/// It deliberately does not retain a reference capable of accessing the
/// visitor object.  A non-dereferenced raw address is kept only to reject a
/// different `V` instance.  A handler must still pass its own `&mut self`
/// reborrow to [`VisitCtx::visit`], so Rust statically rejects holding a
/// mutable borrow of visitor state across child recursion.
pub struct VisitCtx<'a, V> {
    function_table: &'a FunctionTable<V>,
    resources: &'a DriveResources,
    /// Address of the pass object for this callback frame.  It is used only
    /// for identity comparison; the raw pointer is never dereferenced.
    visitor_identity: *mut V,
    current: TVMFFIAny,
    def_region_mode: i32,
    halted: Option<VisitorHalt>,
}

impl<V> VisitCtx<'_, V> {
    /// Visit `child` immediately using the same function table and a fresh
    /// stack-local FFI driver.  Passing `visitor` explicitly is an ordinary
    /// safe Rust mutable reborrow; no user state is hidden behind a raw
    /// pointer or a `RefCell`.  The fresh low-level visitor has a distinct
    /// address but inherits the same table, definition-region mode, and shared
    /// resources; custom structural hooks must not rely on visitor identity
    /// across an explicit edge.  Passing a different `V` instance is a logic
    /// error and panics instead of silently splitting pass state.
    pub fn visit<T>(&mut self, visitor: &mut V, child: &T) -> bool
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.assert_active_visitor(visitor);
        if self.halted.is_some() {
            return false;
        }
        let result = drive_visitor_raw(
            borrowed_raw(child),
            visitor,
            self.function_table,
            self.def_region_mode,
            RootAction::Dispatch,
            self.resources,
        );
        self.absorb(result, true)
    }

    /// Run default reflected recursion over the current value's children.
    /// Each child dispatches through the same function table.  `visitor` must
    /// be the same pass object that received this context.
    pub fn visit_children(&mut self, visitor: &mut V) -> bool {
        self.assert_active_visitor(visitor);
        if self.halted.is_some() {
            return false;
        }
        let result = drive_visitor_raw(
            self.current,
            visitor,
            self.function_table,
            self.def_region_mode,
            RootAction::Children,
            self.resources,
        );
        // RootAction::Children enters the harvested default dispatch, which
        // already records `current` before returning an Error.
        self.absorb(result, false)
    }

    /// Halt traversal with `error` and return the handler's flow value.
    ///
    /// This is the stateful-visitor counterpart of returning an
    /// `Expected<WalkResult>` error from a C++ structural-walk callback:
    ///
    /// ```ignore
    /// return ctx.fail(Error::new(VALUE_ERROR, "invalid node", ""));
    /// ```
    ///
    /// `error` must be uniquely owned.  Retaining a clone and passing the
    /// other handle here panics before traversal propagation starts, because
    /// C++ structural parents enrich Errors in place and would otherwise
    /// mutate the retained clone as well.
    ///
    /// The first pending halt wins.  As the Error propagates, both explicit
    /// `ctx.visit(...)` frames and ordinary reflected recursion append their
    /// current object to `ffi.VisitErrorContext`.
    pub fn fail(&mut self, error: Error) -> WalkResult {
        if self.halted.is_none() {
            assert_eq!(
                error_strong_count(&error),
                1,
                "VisitCtx::fail requires a uniquely owned Error; do not retain a clone"
            );
            self.halted = Some(VisitorHalt {
                value: VisitorHaltValue::Error(error),
                needs_current_error_context: true,
            });
        }
        WalkResult::Interrupt
    }

    fn assert_active_visitor(&self, visitor: &mut V) {
        assert!(
            std::ptr::eq(self.visitor_identity as *const V, visitor as *const V),
            "VisitCtx must recurse with the visitor instance that owns the active handler"
        );
    }

    fn absorb(
        &mut self,
        result: Result<Option<Any>, Error>,
        needs_current_error_context: bool,
    ) -> bool {
        match result {
            Ok(None) => true,
            Ok(Some(interrupt)) => {
                self.halted = Some(VisitorHalt {
                    value: VisitorHaltValue::Interrupt(interrupt),
                    needs_current_error_context,
                });
                false
            }
            Err(error) => {
                self.halted = Some(VisitorHalt {
                    value: VisitorHaltValue::Error(error),
                    needs_current_error_context,
                });
                false
            }
        }
    }
}

/// Run one stateful traversal frame.  Explicit child visits recursively call
/// this function with a normal `&mut V` reborrow, creating a different
/// low-level callback frame from the one whose handler is currently active.
fn drive_visitor_raw<V>(
    root_raw: TVMFFIAny,
    visitor: &mut V,
    function_table: &FunctionTable<V>,
    initial_def_region_mode: i32,
    root_action: RootAction,
    resources: &DriveResources,
) -> Result<Option<Any>, Error> {
    drive_non_reentrant_raw(
        root_raw,
        initial_def_region_mode,
        root_action,
        resources,
        |value, phase, manual_ctx| match phase {
            Phase::Enter => {
                let mut ctx = VisitCtx {
                    function_table,
                    resources,
                    visitor_identity: visitor as *mut V,
                    current: value.0,
                    def_region_mode: manual_ctx.def_region_mode(),
                    halted: None,
                };
                let flow =
                    (function_table.0)(visitor, value, &mut ctx).unwrap_or(WalkResult::Advance);
                if let Some(halt) = ctx.halted.take() {
                    // Return the exact owned Error/VisitInterrupt through the
                    // current FFI frame.  Error values must not be replaced by
                    // a sentinel interrupt: C++ enriches Errors with ancestor
                    // structural-visit context while they propagate.
                    manual_ctx.halted = Some(halt.into_pending_raw());
                    WalkResult::Interrupt
                } else {
                    flow
                }
            }
            // Claimed nodes return Skip on Enter, so their Exit never fires.
            Phase::Exit => WalkResult::Advance,
        },
    )
}

/// Visit `root` with a generated-dispatch visitor borrowed from the caller.
///
/// The visitor object is both the typed handler set and its persistent mutable
/// state.  This is the direct `structural_visit(node, visitor)` form; use
/// [`Visitor`] when an owning driver and [`Visitor::into_inner`] are more
/// convenient.
pub fn structural_visit<R, V>(root: &R, visitor: &mut V) -> Result<Option<Any>, Error>
where
    V: VisitDispatch,
    for<'x> AnyView<'x>: From<&'x R>,
{
    let table = FunctionTable(V::dispatch_visit);
    let resources = DriveResources::new()?;
    drive_visitor_raw(
        borrowed_raw(root),
        visitor,
        &table,
        0,
        RootAction::Dispatch,
        &resources,
    )
}

/// A stateful visitor agent.  `V` is the actual pass object: it is owned here,
/// handed to handlers as `&mut V`, and returned with [`Visitor::into_inner`].
/// [`Visitor::new`] obtains its dispatch from [`VisitDispatch`];
/// [`Visitor::with_function_table`] remains available for explicit/manual
/// dispatchers.
pub struct Visitor<V> {
    inner: V,
    function_table: FunctionTable<V>,
}

impl<V> Visitor<V> {
    /// Wrap a pass object with an explicitly constructed dispatch table.
    pub fn with_function_table(inner: V, function_table: FunctionTable<V>) -> Self {
        Self {
            inner,
            function_table,
        }
    }

    /// Borrow the pass object.
    pub fn inner(&self) -> &V {
        &self.inner
    }

    /// Mutably borrow the pass object between traversals.
    pub fn inner_mut(&mut self) -> &mut V {
        &mut self.inner
    }

    /// Consume the driver and return the pass object.
    pub fn into_inner(self) -> V {
        self.inner
    }

    /// Replace the function table, for callers that need a manually composed
    /// dispatcher instead of the generated [`VisitDispatch`] implementation.
    pub fn function_table(mut self, function_table: FunctionTable<V>) -> Self {
        self.function_table = function_table;
        self
    }

    /// Drive this visitor over the tree — C++ `visitor->Visit(root)`.
    /// Claimed nodes go to the compiled dispatcher; everything else gets
    /// the default reflected recursion.
    pub fn visit<Root>(&mut self, root: &Root) -> Result<Option<Any>, Error>
    where
        for<'x> AnyView<'x>: From<&'x Root>,
    {
        let resources = DriveResources::new()?;
        drive_visitor_raw(
            borrowed_raw(root),
            &mut self.inner,
            &self.function_table,
            0,
            RootAction::Dispatch,
            &resources,
        )
    }
}

impl<V: VisitDispatch> Visitor<V> {
    /// Wrap a generated-dispatch visitor.  Persistent analysis state and all
    /// typed handlers live together in `inner`; no separate state argument is
    /// attached to the driver.
    pub fn new(inner: V) -> Self {
        Self::with_function_table(inner, FunctionTable(V::dispatch_visit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tvm_ffi::error::VALUE_ERROR;

    fn object_ptr(value: &Any) -> *mut TVMFFIObject {
        let raw = borrowed_raw(value);
        assert!(raw.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32);
        unsafe { raw.data_union.v_obj }
    }

    fn error_object(error: &Error) -> *mut ErrorObjectWithContext {
        let raw = borrowed_raw(error);
        assert_eq!(raw.type_index, TVMFFITypeIndex::kTVMFFIError as i32);
        unsafe { raw.data_union.v_obj as *mut ErrorObjectWithContext }
    }

    fn error_extra_context(error: &Error) -> *mut c_void {
        unsafe { (*error_object(error)).extra_context }
    }

    fn visit_error_pattern(error: &Error) -> Vec<*mut TVMFFIObject> {
        let context = error_extra_context(error);
        assert!(!context.is_null());

        let resources = VisitErrorContextResources::new().unwrap();
        assert_eq!(
            unsafe { (*(context as *mut TVMFFIObject)).type_index },
            resources.type_index
        );
        let mut pattern = Any::new();
        let getter_result = unsafe {
            (resources.reverse_visit_pattern_getter)(
                (context as *mut u8).add(resources.reverse_visit_pattern_offset) as *mut c_void,
                Any::as_data_ptr(&mut pattern),
            )
        };
        assert_eq!(getter_result, 0);

        let list_size = Function::get_global("ffi.ListSize").unwrap();
        let size =
            i64::try_from(list_size.call_packed(&[AnyView::from(&pattern)]).unwrap()).unwrap();
        let list_get_item = Function::get_global("ffi.ListGetItem").unwrap();
        (0..size)
            .map(|index| {
                let item = list_get_item
                    .call_packed(&[AnyView::from(&pattern), AnyView::from(&index)])
                    .unwrap();
                object_ptr(&item)
            })
            .collect()
    }

    #[test]
    fn manual_reentrant_first_panic_wins() {
        let array = Function::get_global("ffi.Array").unwrap();
        let child = array.call_packed(&[]).unwrap();
        let root = array.call_packed(&[AnyView::from(&child)]).unwrap();
        let enters = Cell::new(0usize);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = structural_visit_manual(&root, |_value, phase, ctx| {
                if phase != Phase::Enter {
                    return WalkResult::Advance;
                }

                let enter = enters.get();
                enters.set(enter + 1);
                if enter == 0 {
                    assert!(!ctx.visit(&child));
                    panic!("outer panic must not replace the inner panic");
                }
                panic!("inner panic");
            });
        }));

        let payload = panic.expect_err("manual recursion unexpectedly succeeded");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("non-string panic");
        assert_eq!(message, "inner panic");
        assert_eq!(enters.get(), 2);
    }

    #[test]
    fn manual_reentrant_later_panic_payload_is_dropped_safely() {
        struct DropMarker(std::sync::Arc<std::sync::atomic::AtomicUsize>);

        impl Drop for DropMarker {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let array = Function::get_global("ffi.Array").unwrap();
        let child = array.call_packed(&[]).unwrap();
        let root = array.call_packed(&[AnyView::from(&child)]).unwrap();
        let enters = Cell::new(0usize);
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = structural_visit_manual(&root, |_value, phase, ctx| {
                if phase != Phase::Enter {
                    return WalkResult::Advance;
                }
                let enter = enters.get();
                enters.set(enter + 1);
                if enter == 0 {
                    assert!(!ctx.visit(&child));
                }
                std::panic::panic_any(DropMarker(drops.clone()));
            });
        }))
        .expect_err("manual recursion unexpectedly succeeded");

        // The outer (later) payload was disposed of inside the trampoline;
        // the inner (first) payload is still owned by this caught panic.
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        drop(panic);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn retained_low_level_visitor_is_heap_backed_and_inert_after_drive() {
        let array = Function::get_global("ffi.Array").unwrap();
        let root = array.call_packed(&[]).unwrap();
        let interrupt_type_index = lookup_type_index("ffi.VisitInterrupt");
        let retained = RefCell::new(None::<Any>);

        let result = structural_visit_manual(&root, |_value, phase, ctx| {
            if phase == Phase::Enter {
                assert!(retained.borrow().is_none());
                unsafe {
                    let object = ctx.visitor as *mut TVMFFIObject;
                    (*object).combined_ref_count.fetch_add(1, Ordering::Relaxed);
                    *retained.borrow_mut() =
                        Some(Any::from_raw_ffi_any(borrowed_object_raw(ctx.visitor)));
                }
                WalkResult::Skip
            } else {
                WalkResult::Advance
            }
        });
        assert!(result.unwrap().is_none());

        let retained = retained
            .into_inner()
            .expect("callback did not retain visitor");
        assert_eq!(retained.debug_strong_count(), Some(1));
        let visitor = object_ptr(&retained).cast::<CallbackVisitor>();
        let visit = unsafe {
            (*(*visitor).prefix.vtable)
                .visit
                .expect("retained visitor has a null visit slot")
        };
        let halt = unsafe { visit(visitor.cast(), borrowed_raw(&root)) };
        assert_eq!(halt.type_index, interrupt_type_index);
        release_raw_any(halt);
        drop(retained);
    }

    struct NestedFailure {
        root: *mut TVMFFIObject,
        child: *mut TVMFFIObject,
        child_value: Option<Any>,
    }

    impl VisitDispatch for NestedFailure {
        fn dispatch_visit(
            &mut self,
            value: &VisitValue,
            ctx: &mut VisitCtx<'_, Self>,
        ) -> Option<WalkResult> {
            let object = unsafe { value.0.data_union.v_obj };
            if object == self.root {
                // Move the child handle out temporarily so the ordinary
                // `&mut self` reborrow can be passed to ctx.visit.
                let child = self.child_value.take().unwrap();
                let completed = ctx.visit(self, &child);
                self.child_value = Some(child);
                assert!(!completed);
                Some(WalkResult::Skip)
            } else if object == self.child {
                Some(ctx.fail(Error::new(VALUE_ERROR, "nested visit failed", "")))
            } else {
                None
            }
        }
    }

    #[test]
    fn explicit_child_error_records_child_and_current_node() {
        let array = Function::get_global("ffi.Array").unwrap();
        let child = array.call_packed(&[]).unwrap();
        let root = array.call_packed(&[AnyView::from(&child)]).unwrap();
        let child_ptr = object_ptr(&child);
        let root_ptr = object_ptr(&root);
        let mut visitor = NestedFailure {
            root: root_ptr,
            child: child_ptr,
            child_value: Some(child),
        };

        let error = match structural_visit(&root, &mut visitor) {
            Err(error) => error,
            Ok(_) => panic!("nested visit unexpectedly succeeded"),
        };
        assert_eq!(error.kind(), VALUE_ERROR);
        assert_eq!(error.message(), "nested visit failed");

        assert_eq!(visit_error_pattern(&error), vec![child_ptr, root_ptr]);
        let child_count_with_context = visitor
            .child_value
            .as_ref()
            .unwrap()
            .debug_strong_count()
            .unwrap();
        let root_count_with_context = root.debug_strong_count().unwrap();
        drop(error);
        assert_eq!(
            child_count_with_context,
            visitor
                .child_value
                .as_ref()
                .unwrap()
                .debug_strong_count()
                .unwrap()
                + 1
        );
        assert_eq!(
            root_count_with_context,
            root.debug_strong_count().unwrap() + 1
        );
    }

    struct SharedFailure {
        child: *mut TVMFFIObject,
        error: Option<Error>,
    }

    impl VisitDispatch for SharedFailure {
        fn dispatch_visit(
            &mut self,
            value: &VisitValue,
            ctx: &mut VisitCtx<'_, Self>,
        ) -> Option<WalkResult> {
            let object = unsafe { value.0.data_union.v_obj };
            if object == self.child {
                Some(ctx.fail(self.error.take().unwrap()))
            } else {
                // In particular, leave the root unclaimed so C++ reflected
                // recursion is the parent frame around the failing child.
                None
            }
        }
    }

    #[test]
    fn shared_error_is_rejected_before_reflected_parent_propagation() {
        let array = Function::get_global("ffi.Array").unwrap();
        let child = array.call_packed(&[]).unwrap();
        let root = array.call_packed(&[AnyView::from(&child)]).unwrap();
        let child_ptr = object_ptr(&child);
        let shared = Error::new(VALUE_ERROR, "shared nested failure", "");
        let retained = shared.clone();
        assert_eq!(error_strong_count(&retained), 2);
        assert!(error_extra_context(&retained).is_null());

        let mut visitor = SharedFailure {
            child: child_ptr,
            error: Some(shared),
        };
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = structural_visit(&root, &mut visitor);
        }));
        let payload = panic.expect_err("a cloned Error must be rejected");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("non-string panic");
        assert!(message.contains("requires a uniquely owned Error"));

        assert_eq!(retained.message(), "shared nested failure");
        assert!(error_extra_context(&retained).is_null());
        assert!(visitor.error.is_none());
        assert_eq!(error_strong_count(&retained), 1);
    }

    struct DoubleFailure {
        child: *mut TVMFFIObject,
        first: Option<Error>,
        second: Option<Error>,
    }

    impl VisitDispatch for DoubleFailure {
        fn dispatch_visit(
            &mut self,
            value: &VisitValue,
            ctx: &mut VisitCtx<'_, Self>,
        ) -> Option<WalkResult> {
            if unsafe { value.0.data_union.v_obj } == self.child {
                let flow = ctx.fail(self.first.take().unwrap());
                // The first halt wins, so this shared Error is ignored and
                // dropped without entering the C++ propagation chain.
                let _ = ctx.fail(self.second.take().unwrap());
                Some(flow)
            } else {
                None
            }
        }
    }

    #[test]
    fn first_pending_error_wins_without_validating_an_ignored_second_error() {
        let array = Function::get_global("ffi.Array").unwrap();
        let child = array.call_packed(&[]).unwrap();
        let root = array.call_packed(&[AnyView::from(&child)]).unwrap();
        let second = Error::new(VALUE_ERROR, "ignored shared failure", "");
        let retained_second = second.clone();
        let mut visitor = DoubleFailure {
            child: object_ptr(&child),
            first: Some(Error::new(VALUE_ERROR, "first failure", "")),
            second: Some(second),
        };

        let returned = match structural_visit(&root, &mut visitor) {
            Err(error) => error,
            Ok(_) => panic!("first failure unexpectedly succeeded"),
        };
        assert_eq!(returned.message(), "first failure");
        assert_eq!(
            visit_error_pattern(&returned),
            vec![object_ptr(&child), object_ptr(&root)]
        );
        assert_eq!(retained_second.message(), "ignored shared failure");
        assert_eq!(error_strong_count(&retained_second), 1);
        assert!(error_extra_context(&retained_second).is_null());
    }

    struct ChildrenFailure {
        root: *mut TVMFFIObject,
        child: *mut TVMFFIObject,
    }

    impl VisitDispatch for ChildrenFailure {
        fn dispatch_visit(
            &mut self,
            value: &VisitValue,
            ctx: &mut VisitCtx<'_, Self>,
        ) -> Option<WalkResult> {
            let object = unsafe { value.0.data_union.v_obj };
            if object == self.root {
                assert!(!ctx.visit_children(self));
                Some(WalkResult::Skip)
            } else if object == self.child {
                Some(ctx.fail(Error::new(VALUE_ERROR, "child visit failed", "")))
            } else {
                None
            }
        }
    }

    #[test]
    fn explicit_children_error_records_current_node_once() {
        let array = Function::get_global("ffi.Array").unwrap();
        let child = array.call_packed(&[]).unwrap();
        let root = array.call_packed(&[AnyView::from(&child)]).unwrap();
        let child_ptr = object_ptr(&child);
        let root_ptr = object_ptr(&root);
        let mut visitor = ChildrenFailure {
            root: root_ptr,
            child: child_ptr,
        };

        let error = match structural_visit(&root, &mut visitor) {
            Err(error) => error,
            Ok(_) => panic!("child visit unexpectedly succeeded"),
        };
        assert_eq!(visit_error_pattern(&error), vec![child_ptr, root_ptr]);
    }

    struct WrongVisitor {
        root: *mut TVMFFIObject,
        child: Option<Any>,
        other: Option<Box<WrongVisitor>>,
    }

    impl VisitDispatch for WrongVisitor {
        fn dispatch_visit(
            &mut self,
            value: &VisitValue,
            ctx: &mut VisitCtx<'_, Self>,
        ) -> Option<WalkResult> {
            if unsafe { value.0.data_union.v_obj } == self.root {
                let child = self.child.take().unwrap();
                let mut other = self.other.take().unwrap();
                ctx.visit(&mut other, &child);
                unreachable!("a different visitor instance must be rejected")
            }
            None
        }
    }

    #[test]
    fn visit_rejects_a_different_visitor_instance() {
        let array = Function::get_global("ffi.Array").unwrap();
        let child = array.call_packed(&[]).unwrap();
        let root = array.call_packed(&[AnyView::from(&child)]).unwrap();
        let root_ptr = object_ptr(&root);
        let mut visitor = WrongVisitor {
            root: root_ptr,
            child: Some(child),
            other: Some(Box::new(WrongVisitor {
                root: std::ptr::null_mut(),
                child: None,
                other: None,
            })),
        };

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = structural_visit(&root, &mut visitor);
        }));
        assert!(panic.is_err());
    }
}
