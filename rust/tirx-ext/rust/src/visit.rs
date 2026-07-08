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
//! * a `#[repr(C)]` [`CallbackVisitor`] extends the C++ object layout
//!   (`Object` header, `vtable_`, `def_region_mode_`) with Rust-side state
//!   (the user's pre/post closures);
//! * its vtable entry [`callback_visit`] runs the `pre` closure, then chains
//!   to the **default** visit function — harvested at runtime from a
//!   C++-constructed `ffi.StructuralVisitor` — for the reflected recursion
//!   into children (which re-enters our vtable), then runs `post`.
//!
//! So all reflection/traversal logic stays in libtvm_ffi; Rust only supplies
//! the per-node callback through the same function table C++ subclasses use.

use std::cell::{Cell, RefCell};
use std::os::raw::c_void;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::AtomicU64;
use std::sync::OnceLock;

use tvm_ffi::any::{Any, AnyView};
use tvm_ffi::function::Function;
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIObject, TVMFFITypeIndex, COMBINED_REF_COUNT_BOTH_ONE,
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

const _: () = {
    assert!(std::mem::size_of::<TVMFFIObject>() == 24);
    assert!(std::mem::offset_of!(VisitorPrefix, vtable) == 24);
    assert!(std::mem::offset_of!(VisitorPrefix, def_region_mode) == 32);
    assert!(std::mem::size_of::<TVMFFIAny>() == 16);
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

/// Hands the *visitor itself* to a [`structural_visit_manual`] callback, so it
/// can drive child visits explicitly — the Rust face of calling
/// `visitor->Visit(child)` / `DefaultVisit(value)` inside a C++
/// `StructuralVisitorObj` subclass.
pub struct VisitCtx {
    visitor: *mut c_void,
    /// The value the callback is currently visiting.
    current: TVMFFIAny,
    /// A halt (owned `ffi.VisitInterrupt` or `ffi.Error`) raised by a manual
    /// child visit, pending propagation by the trampoline.
    halted: Option<TVMFFIAny>,
}

impl VisitCtx {
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
        let ret = unsafe { ((*me).default_visit)(self.visitor, self.current) };
        self.absorb(ret)
    }

    /// Dispatch `raw` through the visitor's function table.
    fn dispatch(&mut self, raw: TVMFFIAny) -> bool {
        let ret = unsafe { callback_visit(self.visitor, raw) };
        self.absorb(ret)
    }

    /// Record a halt returned by a child visit; `true` == keep going.
    fn absorb(&mut self, ret: TVMFFIAny) -> bool {
        if ret.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            true
        } else {
            self.halted = Some(ret);
            false
        }
    }
}

/// The uniform callback shape both entry points lower to. `Fn` (not `FnMut`)
/// because [`VisitCtx::visit`] re-enters the callback while an outer call is
/// still on the stack — re-entrancy needs a shared-ref callable; mutable
/// callback state goes in `Cell`s (≈ a C++ `const` method with `mutable`
/// members).
type DynCallback<'a> = &'a dyn Fn(&VisitValue, Phase, &mut VisitCtx) -> WalkResult;

/// Our visitor object: the C++-visible [`VisitorPrefix`] followed by Rust-only
/// state C++ never touches. Lives on the stack of the driver; the header
/// carries the real `ffi.StructuralVisitor` type index and a normal refcount
/// so it is indistinguishable from a C++ subclass instance for the
/// (borrow-only) traversal paths.
#[repr(C)]
struct CallbackVisitor<'a> {
    prefix: VisitorPrefix,
    callback: DynCallback<'a>,
    default_visit: FStructuralVisit,
    /// A pre-built payload-less `ffi.VisitInterrupt`, constructed in [`drive`]
    /// (a caught region) so the extern "C" paths below only ever *clone* it —
    /// a refcount increment that cannot fail — instead of calling into the FFI
    /// registry from a frame where a panic would abort the host process.
    interrupt_proto: Any,
    panic: Option<Box<dyn std::any::Any + Send + 'static>>,
}

/// The single function-table entry of the Rust visitor. Re-entered for every
/// child (by the default chain and by [`VisitCtx::visit`]); each frame only
/// takes short-lived reborrows through the raw `me` pointer.
unsafe extern "C" fn callback_visit(visitor: *mut c_void, value: TVMFFIAny) -> TVMFFIAny {
    let me = visitor as *mut CallbackVisitor;
    // The C++ walk visitor short-circuits FFI None without invoking callbacks.
    if value.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return TVMFFIAny::new();
    }
    let view = VisitValue(value);
    let mut ctx = VisitCtx {
        visitor,
        current: value,
        halted: None,
    };

    match catch_unwind(AssertUnwindSafe(|| ((*me).callback)(&view, Phase::Enter, &mut ctx))) {
        Ok(flow) => {
            // A halt raised by a manual child visit outranks the flow verdict.
            if let Some(halt) = ctx.halted.take() {
                return halt;
            }
            match flow {
                WalkResult::Advance => {}
                WalkResult::Skip => return TVMFFIAny::new(),
                WalkResult::Interrupt => return make_interrupt_raw(me),
            }
        }
        Err(p) => {
            (*me).panic = Some(p);
            release_raw_any_opt(ctx.halted.take());
            return make_interrupt_raw(me);
        }
    }

    // Chain to tvm-ffi's default visit: reflected recursion into children.
    let ret = ((*me).default_visit)(visitor, value);
    if ret.type_index != TVMFFITypeIndex::kTVMFFINone as i32 {
        return ret; // owned VisitInterrupt or Error — forward as-is
    }

    match catch_unwind(AssertUnwindSafe(|| ((*me).callback)(&view, Phase::Exit, &mut ctx))) {
        Ok(flow) => {
            if let Some(halt) = ctx.halted.take() {
                return halt;
            }
            match flow {
                WalkResult::Interrupt => make_interrupt_raw(me),
                _ => TVMFFIAny::new(),
            }
        }
        Err(p) => {
            (*me).panic = Some(p);
            release_raw_any_opt(ctx.halted.take());
            make_interrupt_raw(me)
        }
    }
}

/// A payload-less `ffi.VisitInterrupt` as an owned raw `TVMFFIAny`: a clone
/// (refcount inc) of the prototype [`drive`] built up front — infallible, as
/// required inside the extern "C" callback.
unsafe fn make_interrupt_raw(me: *const CallbackVisitor) -> TVMFFIAny {
    Any::into_raw_ffi_any((*me).interrupt_proto.clone())
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
    callback: impl FnMut(&VisitValue, Phase) -> WalkResult,
) -> Result<Option<Any>, tvm_ffi::error::Error>
where
    for<'x> AnyView<'x>: From<&'x R>,
{
    // Plain walks never re-enter the callback (they don't use the ctx), so an
    // `FnMut` is fine — adapt it to the re-entrant `Fn` shape via `RefCell`.
    let callback = RefCell::new(callback);
    drive(root, &move |v: &VisitValue, phase: Phase, _ctx: &mut VisitCtx| {
        (callback.borrow_mut())(v, phase)
    })
}

/// The *visitor* variant of [`walk`]: the callback also receives a
/// [`VisitCtx`] and can take **full control of a node's visit order** by
/// visiting chosen children explicitly and returning [`WalkResult::Skip`].
/// This mirrors subclassing C++ `StructuralVisitorObj` and calling
/// `visitor->Visit(child)` from the overridden visit. Because `ctx.visit`
/// **re-enters** the callback while an outer call is still running, the
/// callback must be `Fn`; keep its mutable state in `Cell`s/`RefCell`s.
pub fn structural_visit_manual<R>(
    root: &R,
    callback: impl Fn(&VisitValue, Phase, &mut VisitCtx) -> WalkResult,
) -> Result<Option<Any>, tvm_ffi::error::Error>
where
    for<'x> AnyView<'x>: From<&'x R>,
{
    drive(root, &callback)
}

/// Shared driver: stack-allocate the visitor, run the walk from `root`, and
/// translate the ABI result (None / interrupt / error / stashed panic).
fn drive<R>(root: &R, callback: DynCallback) -> Result<Option<Any>, tvm_ffi::error::Error>
where
    for<'x> AnyView<'x>: From<&'x R>,
{
    static VTABLE: StructuralVisitorVTable = StructuralVisitorVTable {
        visit: Some(callback_visit),
    };
    let default_visit = default_visit_fn();
    // Pre-build the interrupt prototype here, where an ffi failure is still a
    // clean `Err` — the callback clones it instead of calling the registry.
    let interrupt_proto = Function::get_global("ffi.VisitInterrupt")?.call_packed(&[AnyView::new()])?;

    let mut vis = CallbackVisitor {
        prefix: VisitorPrefix {
            header: TVMFFIObject {
                combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                type_index: lookup_type_index("ffi.StructuralVisitor"),
                __padding: 0,
                deleter: None,
            },
            vtable: &VTABLE,
            def_region_mode: Cell::new(0),
        },
        callback,
        default_visit,
        interrupt_proto,
        panic: None,
    };

    // Root as a borrowed AnyView cell (AnyView is layout-identical to
    // TVMFFIAny — the same cast `Function::call_packed` relies on).
    let root_view = [AnyView::from(root)];
    let root_raw = unsafe { std::ptr::read(root_view.as_ptr() as *const TVMFFIAny) };

    let ret = unsafe { callback_visit(&mut vis as *mut _ as *mut c_void, root_raw) };

    if let Some(p) = vis.panic.take() {
        release_raw_any(ret);
        resume_unwind(p);
    }
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

fn release_raw_any_opt(raw: Option<TVMFFIAny>) {
    if let Some(raw) = raw {
        release_raw_any(raw);
    }
}

// ---------------------------------------------------------------------------
// The Visitor agent (mirrored by `mutate::Mapper`)
// ---------------------------------------------------------------------------

/// The compiled dispatcher: `None` for values outside the claimed set.
pub type Dispatch<S> = fn(&RefCell<S>, &VisitValue, &mut VisitCtx) -> Option<WalkResult>;

/// The function table: one compiled `fn` pointer — nothing built at run
/// time, nothing boxed.
pub struct FunctionTable<S>(pub Dispatch<S>);

/// The visitor agent: pass state + function table, then drive with
/// [`Visitor::visit`] (entry point and in-handler recursion share the
/// name, exactly as in C++). The rewriting counterpart is `mutate::Mapper`,
/// which mirrors this builder shape. Built by lib.rs's `function_table!` macro
/// and driven over a tree; the low-level engine it wraps is
/// [`structural_visit_manual`].
pub struct Visitor<'s, S> {
    state: Option<&'s RefCell<S>>,
    function_table: Option<FunctionTable<S>>,
}

// Manual impl: a `#[derive(Default)]` would demand `S: Default` even
// though no `S` value is ever defaulted — the classic derive-bound trap.
impl<S> Default for Visitor<'_, S> {
    fn default() -> Self {
        Self { state: None, function_table: None }
    }
}

impl<'s, S> Visitor<'s, S> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the extra content the handlers carry along: the pass's
    /// mutable state.
    pub fn visit_with_extra_content(mut self, state: &'s RefCell<S>) -> Self {
        self.state = Some(state);
        self
    }

    /// Attach the function table: the per-type handlers.
    pub fn function_table(mut self, function_table: FunctionTable<S>) -> Self {
        self.function_table = Some(function_table);
        self
    }

    /// Drive this visitor over the tree — C++ `visitor->Visit(root)`.
    /// Claimed nodes go to the compiled dispatcher; everything else gets
    /// the default reflected recursion.
    pub fn visit<Root>(&self, root: &Root) -> Result<Option<Any>, tvm_ffi::error::Error>
    where
        for<'x> AnyView<'x>: From<&'x Root>,
    {
        let state = self.state.expect("Visitor: call visit_with_extra_content first");
        let table = self.function_table.as_ref().expect("Visitor: call function_table first");
        structural_visit_manual(root, |v, phase, visitor| match phase {
            Phase::Enter => (table.0)(state, v, visitor).unwrap_or(WalkResult::Advance),
            // Claimed nodes return Skip on Enter, so their Exit never fires.
            Phase::Exit => WalkResult::Advance,
        })
    }
}
