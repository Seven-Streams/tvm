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

//! Structural **mapper**: the rewriting counterpart of `visit.rs`, API-shaped
//! after tvm-ffi's `StructuralMapper` (apache/tvm-ffi#649) per
//! `docs/mutator-redesign.md` (history: `docs/mutator-proposal.md`, hardening:
//! `docs/mutator-review.md`).
//!
//! The vendored tvm-ffi has no mapper engine yet (#649 is WIP), so recursion
//! is pure Rust — but every *rebuild* primitive is a tvm-ffi-blessed
//! mechanism, mirroring FromJSONGraph deserialization
//! (`src/ffi/extra/serialization.cc`) and, since #649, the upstream engine
//! itself:
//!
//! * take a unique copy via the `__ffi_shallow_copy__` type attr (the
//!   reflected C++ copy constructor — the same `make_object<T>(*node)` that
//!   C++ `StmtMutator::CopyOnWrite` uses on its non-unique path);
//! * overwrite changed child fields via the reflected per-field `setter`
//!   (registered even for read-only fields, exactly for this kind of use).
//!
//! Dispatch layers per node (the shape of #649's `DefaultMap`):
//!
//! 1. the pass's **function table** (`mutation_table!`) — a claimed node is
//!    fully the handler's business (C++ `VisitStmt_` override semantics);
//! 2. **`TYPE_HOOKS`** — per-type structural rules, the Rust face of the
//!    `__s_map__` type attr (route A's incremental home; the table ships
//!    empty — see `docs/mutator-redesign.md` §3.3);
//! 3. the **default rebuild** over value-position children.
//!
//! Handlers and hooks follow the C++ `StmtMutator` / #649 hook contract
//! literally: they **return the node** — the original reference for
//! "unchanged" (detected by pointer identity, the `same_as` check C++
//! handlers rely on) or a replacement — and recurse into children through
//! the [`MapCtx`] they receive: `mapper.map(&child)` is C++
//! `mapper->Map(child)` / `this->VisitStmt(child)`, `mapper.map_fields(&op)`
//! is the `VisitStmt_` default body. Errors are plain `Err` (the
//! `Expected<Any>` channel — no side-channel).
//!
//! Recursion coverage (P1 route B, decided 2026-07-06): the default rebuild
//! is **guaranteed to match C++ `StmtMutator` only on the control-flow
//! skeleton** (`SUPPORTED_STMT_KEYS`). Value-position `Stmt`/`PrimExpr`
//! children (and Arrays of them) are recursed; `Map` fields (annotations),
//! def-position fields (either `SEqHashDef*` flavor, e.g. `loop_var`) and
//! SEqHash-ignored fields (span) are left untouched. A node **outside the
//! skeleton whose children changed errors loudly** instead of silently
//! approximating (SBlock/AllocBuffer/TilePrimitiveCall internals need C++'s
//! per-type rules — the `TYPE_HOOKS` slot exists to add them one type at a
//! time); untouched subtrees of any type pass through unchanged, where no
//! divergence is possible. COW throughout: an untouched subtree is returned
//! pointer-identically; Python's original tree is never written to.

use std::cell::RefCell;
use std::ops::ControlFlow;
use std::os::raw::c_void;
use std::sync::OnceLock;

use tvm_ffi::any::{Any, AnyView};
use tvm_ffi::error::{Error, Result, RUNTIME_ERROR};
use tvm_ffi::function::Function;
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIErrorMoveFromRaised, TVMFFIFieldInfo, TVMFFIFieldSetter,
    TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeAttrColumn, TVMFFITypeIndex,
};

use crate::node::{ExprNode, Stmt, StmtNode};
use crate::object_ref::is_instance;
use crate::reflect::for_each_field;
use crate::runtime::lookup_type_index;
use crate::visit::VisitValue;

// Field flag bits (include/tvm/ffi/c_api.h, TVMFFIFieldFlagBitMask*).
const FLAG_SEQ_HASH_IGNORE: i64 = 1 << 3;
const FLAG_SEQ_HASH_DEF_RECURSIVE: i64 = 1 << 4;
const FLAG_SETTER_IS_FUNCTION_OBJ: i64 = 1 << 11;
// tirx headers carry TODOs migrating def fields (Bind.var, AllocBuffer.buffer)
// from DefRecursive to this flag — treat both as def-position from day one.
const FLAG_SEQ_HASH_DEF_NON_RECURSIVE: i64 = 1 << 12;

/// Fields the default rebuild must not descend into: span-likes
/// (SEqHashIgnore) and def-position fields (either def-region flavor) —
/// the C++ StmtMutator skip set.
const FLAG_SKIP_FIELD: i64 =
    FLAG_SEQ_HASH_IGNORE | FLAG_SEQ_HASH_DEF_RECURSIVE | FLAG_SEQ_HASH_DEF_NON_RECURSIVE;

const SHALLOW_COPY_ATTR: &str = "__ffi_shallow_copy__";

/// P1 route B: statement types whose default rebuild provably matches the
/// C++ `StmtMutator` field rules (verified against stmt_functor.cc — value
/// positions only, no def-site machinery, no normalization step). A CHANGED
/// statement outside this set errors instead of silently diverging — unless
/// a [`TYPE_HOOKS`] entry takes it over first.
const SUPPORTED_STMT_KEYS: &[&str] = &[
    "tirx.For",
    "tirx.IfThenElse",
    "tirx.While",
    "tirx.SeqStmt",
    "tirx.Evaluate",
    "tirx.BufferStore",
    "tirx.Break",
];

/// Expression rebuilds are operand-position by construction and match C++
/// `ExprMutator` generically — except these known-divergent corners.
const UNSUPPORTED_EXPR_KEYS: &[&str] = &["tirx.Reduce"];

/// Recursion guard: ~3 stack frames per IR level would otherwise turn deep
/// (generated) IR into an uncatchable stack-overflow SIGSEGV of the host.
const MAX_DEPTH: u32 = 10_000;

extern "C" {
    // Present in libtvm_ffi but not yet declared by tvm-ffi-sys.
    fn TVMFFIGetTypeAttrColumn(attr_name: *const TVMFFIByteArray) -> *const TVMFFITypeAttrColumn;
}

// ---------------------------------------------------------------------------
// Raw-cell helpers (single home for the AnyView <-> TVMFFIAny reinterpret)
// ---------------------------------------------------------------------------

/// Reinterpret a borrowed `AnyView` as its raw cell.
fn raw_of(view: AnyView) -> TVMFFIAny {
    unsafe { std::ptr::read(&view as *const AnyView as *const TVMFFIAny) }
}

/// Reinterpret a borrowed raw cell as an `AnyView` (inverse of [`raw_of`]).
unsafe fn view_of(raw: &TVMFFIAny) -> AnyView<'_> {
    std::ptr::read(raw as *const TVMFFIAny as *const AnyView)
}

/// The raw cell of an owned `Any` (a POD copy; ownership stays with `any`).
fn raw_of_owned(any: &mut Any) -> TVMFFIAny {
    unsafe { *Any::as_data_ptr(any) }
}

fn type_key_of(type_index: i32) -> String {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            format!("<type_index {type_index}>")
        } else {
            (*info).type_key.as_str().to_string()
        }
    }
}

/// The detailed error the C side stored in the thread's raised-error slot for
/// the last failing safe-call (getter/setter), or `fallback` if none — taking
/// it also clears the slot, so it cannot resurface on an unrelated failure.
fn take_raised_error(fallback: &str) -> Error {
    unsafe {
        let mut handle: *mut c_void = std::ptr::null_mut();
        TVMFFIErrorMoveFromRaised(&mut handle);
        if !handle.is_null() {
            let header = handle as *mut TVMFFIObject;
            let mut cell = TVMFFIAny::new();
            cell.type_index = (*header).type_index;
            cell.data_union.v_obj = header;
            let any = Any::from_raw_ffi_any(cell);
            if let Ok(e) = Error::try_from(any) {
                return e;
            }
        }
        Error::new(RUNTIME_ERROR, fallback, "")
    }
}

// ---------------------------------------------------------------------------
// Rebuild primitives (tvm-ffi blessed: shallow copy + field setter)
// ---------------------------------------------------------------------------

/// Clone the node behind `raw` via its `__ffi_shallow_copy__` type attr:
/// a refcount==1 copy sharing every field with the original.
fn shallow_copy(raw: &TVMFFIAny) -> Result<Any> {
    unsafe {
        let attr_name = TVMFFIByteArray::from_str(SHALLOW_COPY_ATTR);
        // Looked up per call on purpose: the column's data array can be
        // reallocated when later registrations grow it.
        let col = TVMFFIGetTypeAttrColumn(&attr_name);
        let cell: Option<&TVMFFIAny> = if col.is_null() {
            None
        } else {
            let c: &TVMFFITypeAttrColumn = &*col;
            let idx = raw.type_index - c.begin_index;
            if idx >= 0 && idx < c.size {
                Some(&*c.data.offset(idx as isize))
            } else {
                None
            }
        };
        let copy_fn = cell
            .and_then(|c| view_of(c).try_as::<Function>())
            .ok_or_else(|| {
                Error::new(
                    RUNTIME_ERROR,
                    &format!(
                        "tirx_ext: type `{}` registers no {SHALLOW_COPY_ATTR} — cannot rebuild",
                        type_key_of(raw.type_index)
                    ),
                    "",
                )
            })?;
        let mut copy = copy_fn.call_packed(&[view_of(raw)])?;
        // Trust but verify: the attr registry is open, a custom hook could
        // return anything — a wild write through a non-object would follow.
        let copy_raw = raw_of_owned(&mut copy);
        if copy_raw.type_index != raw.type_index {
            return Err(Error::new(
                RUNTIME_ERROR,
                &format!(
                    "tirx_ext: {SHALLOW_COPY_ATTR} of `{}` returned a `{}`",
                    type_key_of(raw.type_index),
                    type_key_of(copy_raw.type_index)
                ),
                "",
            ));
        }
        Ok(copy)
    }
}

/// Write `value` into the field described by `field` on the (uniquely owned)
/// object at `obj` — the `CallFieldSetter` recipe of accessor.h.
unsafe fn set_field(obj: *mut u8, field: &TVMFFIFieldInfo, value: &TVMFFIAny) -> Result<()> {
    if field.flags & FLAG_SETTER_IS_FUNCTION_OBJ != 0 {
        return Err(Error::new(
            RUNTIME_ERROR,
            "tirx_ext: FunctionObj-form field setters are not supported",
            "",
        ));
    }
    if field.setter.is_null() {
        return Err(Error::new(RUNTIME_ERROR, "tirx_ext: field has no setter", ""));
    }
    let setter: TVMFFIFieldSetter = std::mem::transmute(field.setter);
    let addr = obj.offset(field.offset as isize) as *mut c_void;
    if setter(addr, value) != 0 {
        return Err(take_raised_error(&format!(
            "tirx_ext: setter rejected value for field `{}`",
            field.name.as_str()
        )));
    }
    Ok(())
}

/// The raw object payload pointer of an owned `Any` (which must hold an object).
fn obj_ptr_of(any: &mut Any) -> *mut u8 {
    unsafe { (*Any::as_data_ptr(any)).data_union.v_obj as *mut u8 }
}

// ---------------------------------------------------------------------------
// Handler vocabulary
// ---------------------------------------------------------------------------

/// The compiled dispatcher: `None` for values outside the claimed set
/// (mirror of `visit::Dispatch`). A handler returns `Ok` with the node
/// itself — the original for "unchanged" (COW by pointer identity), a new
/// one for a replacement — type-erased to `Any` by the `mutation_table!`
/// macro; `Err` aborts the whole map.
pub type Dispatch<S> = fn(&RefCell<S>, &VisitValue, &mut MapCtx) -> Option<Result<Any>>;

/// The function table: one compiled `fn` pointer (mirror of
/// `visit::FunctionTable`).
pub struct FunctionTable<S>(pub Dispatch<S>);

/// A per-type structural-map rule — the Rust face of #649's `__s_map__`
/// type attr. Unlike a pass handler it never sees pass state: it defines how
/// its *type* maps structurally (recursing children through the ctx), for
/// every pass alike. Same return contract as a handler.
pub type TypeHook = fn(&VisitValue, &mut MapCtx) -> Result<Any>;

/// The registered per-type rules (route A's incremental home): a type listed
/// here is rebuilt by its hook instead of the default reflected rebuild, and
/// thereby leaves the [`SUPPORTED_STMT_KEYS`] error surface. Ships empty —
/// each entry must be hand-verified against the type's stmt_functor.cc
/// rules before it is added (docs/mutator-redesign.md §3.3 / 待拍板 #3).
const TYPE_HOOKS: &[(&str, TypeHook)] = &[];

/// Resolve a `(type_key, hook)` table to `(type_index, hook)` via the FFI
/// type registry (types must already be registered — host imported tvm).
pub(crate) fn resolve_hooks(table: &[(&str, TypeHook)]) -> Vec<(i32, TypeHook)> {
    table.iter().map(|(key, hook)| (lookup_type_index(key), *hook)).collect()
}

fn default_hooks() -> &'static [(i32, TypeHook)] {
    // OnceLock (not LazyLock): a panicking lookup — host not fully loaded —
    // leaves the cell empty and retries on the next map.
    static HOOKS: OnceLock<Vec<(i32, TypeHook)>> = OnceLock::new();
    HOOKS.get_or_init(|| resolve_hooks(TYPE_HOOKS))
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

/// Layout prefix shared by `ArrayObj`/`ListObj`:
/// `Object (24B) + TVMFFISeqCell { data, size, capacity, deleter }`
/// (container/seq_base.h) — asserted here the same way visit.rs pins the
/// `StructuralVisitorObj` prefix.
#[repr(C)]
struct SeqPrefix {
    header: TVMFFIObject,
    data: *const TVMFFIAny,
    size: i64,
}

const _: () = {
    assert!(std::mem::offset_of!(SeqPrefix, data) == 24);
    assert!(std::mem::offset_of!(SeqPrefix, size) == 32);
};

/// The pass dispatch with its state captured — `S` is erased at the agent
/// boundary so the engine (and [`TypeHook`]s, which must not see `S`) stay
/// non-generic. Elided lifetimes in the `Fn` sugar are higher-ranked: the
/// callee accepts any `MapCtx` frame.
type ErasedDispatch<'e> = &'e dyn Fn(&VisitValue, &mut MapCtx) -> Option<Result<Any>>;

/// Adapt a compiled [`Dispatch`] + its state to the erased engine shape.
/// (A named fn, not an inline closure, so the returned closure is checked
/// against the higher-ranked `Fn` bound and coerces to [`ErasedDispatch`].)
fn erase_dispatch<'s, S>(
    state: &'s RefCell<S>,
    dispatch: Dispatch<S>,
) -> impl Fn(&VisitValue, &mut MapCtx) -> Option<Result<Any>> + 's {
    move |v, ctx| dispatch(state, v, ctx)
}

struct Engine<'e> {
    dispatch: ErasedDispatch<'e>,
    hooks: &'e [(i32, TypeHook)],
    stmt_ti: i32,
    expr_ti: i32,
}

/// The mapper handle a handler/hook receives — #649's
/// `StructuralMapperObj* mapper` parameter, and the mutator-side mirror of
/// `visit::VisitCtx`. Both methods **re-enter** the dispatch (other handlers
/// may run while this frame is live), so no `state` borrow may be held
/// across a call — the same discipline as `VisitCtx::visit`.
pub struct MapCtx<'a, 'e> {
    engine: &'a Engine<'e>,
    depth: u32,
}

impl MapCtx<'_, '_> {
    /// Map `child` through the full dispatch chain (pass table → type hooks
    /// → default rebuild) — C++ `mapper->Map(child)` / `this->VisitStmt(s)`.
    /// COW: the result is `same_as` `child` iff nothing beneath changed.
    pub fn map<T>(&mut self, child: &T) -> Result<T>
    where
        T: Clone + TryFrom<Any, Error = Error>,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let raw = raw_of(AnyView::from(child));
        match self.engine.map_raw(&raw, self.depth + 1)? {
            Some(any) => T::try_from(any),
            None => Ok(child.clone()),
        }
    }

    /// Run the **default reflected rebuild** over `node`'s value-position
    /// children, skipping the pass table for `node` itself (else the calling
    /// handler would recurse into itself) — the C++ `VisitStmt_(op)` default
    /// body, #649's `DefaultMap` sans hook consult. Children re-enter the
    /// full chain. Subject to the skeleton whitelist like any rebuild.
    pub fn map_fields<T>(&mut self, node: &T) -> Result<T>
    where
        T: Clone + TryFrom<Any, Error = Error>,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let raw = raw_of(AnyView::from(node));
        match self.engine.rebuild_default(&raw, self.depth)? {
            Some(any) => T::try_from(any),
            None => Ok(node.clone()),
        }
    }
}

/// COW verdict on a handler/hook return value: pointer-identical to the
/// input means "unchanged" — the same check C++ handlers express via
/// `same_as`. `Ok(None)` == unchanged.
fn cow_verdict(raw: &TVMFFIAny, mut returned: Any) -> Result<Option<Any>> {
    let same = unsafe { raw_of_owned(&mut returned).data_union.v_obj == raw.data_union.v_obj };
    Ok(if same { None } else { Some(returned) })
}

impl Engine<'_> {
    /// Is this a value-position IR child (`Stmt` or `PrimExpr` family)?
    fn is_ir_family(&self, type_index: i32) -> bool {
        is_instance(type_index, self.stmt_ti) || is_instance(type_index, self.expr_ti)
    }

    /// P1 route B gate, consulted only on the changed path (an untouched
    /// subtree of any type is returned as-is — no rebuild, no divergence).
    fn check_rebuild_supported(&self, type_index: i32) -> Result<()> {
        let key = type_key_of(type_index);
        let supported = if is_instance(type_index, self.stmt_ti) {
            SUPPORTED_STMT_KEYS.contains(&key.as_str())
        } else {
            !UNSUPPORTED_EXPR_KEYS.contains(&key.as_str())
        };
        if supported {
            Ok(())
        } else {
            Err(Error::new(
                RUNTIME_ERROR,
                &format!(
                    "tirx_ext: cannot rebuild `{key}` — its children changed but the type \
                     is outside the mapper's supported control-flow skeleton \
                     (docs/mutator-review.md P1, route B)"
                ),
                "",
            ))
        }
    }

    /// Map the subtree behind a borrowed raw value through the dispatch
    /// chain. `Ok(None)` == unchanged.
    fn map_raw(&self, raw: &TVMFFIAny, depth: u32) -> Result<Option<Any>> {
        if depth > MAX_DEPTH {
            return Err(Error::new(
                RUNTIME_ERROR,
                &format!("tirx_ext: IR nesting exceeds {MAX_DEPTH} levels"),
                "",
            ));
        }
        if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            return Ok(None);
        }
        let vv = VisitValue::from_raw(*raw);
        // 1) Pass table: a claimed node is fully the handler's business.
        {
            let mut ctx = MapCtx { engine: self, depth };
            if let Some(outcome) = (self.dispatch)(&vv, &mut ctx) {
                return cow_verdict(raw, outcome?);
            }
        }
        // 2) Per-type rules (`__s_map__` layer).
        if let Some(hook) =
            self.hooks.iter().find(|(ti, _)| *ti == raw.type_index).map(|(_, h)| *h)
        {
            let mut ctx = MapCtx { engine: self, depth };
            return cow_verdict(raw, hook(&vv, &mut ctx)?);
        }
        // 3) Unclaimed: default rebuild over value-position children.
        self.rebuild_default(raw, depth)
    }

    /// Default rebuild (C++ StmtMutator's VisitStmt_ default bodies,
    /// generically): recurse into Stmt/PrimExpr-family fields and Arrays
    /// thereof; on any change, shallow-copy and set the changed fields.
    fn rebuild_default(&self, raw: &TVMFFIAny, depth: u32) -> Result<Option<Any>> {
        let obj = unsafe { raw.data_union.v_obj } as *const u8;
        let mut changes: Vec<(&'static TVMFFIFieldInfo, Any)> = Vec::new();

        let failed: Option<Error> = unsafe {
            for_each_field(raw.type_index, |field| {
                if field.flags & FLAG_SKIP_FIELD != 0 {
                    return ControlFlow::Continue(()); // span / def-position
                }
                let Some(getter) = field.getter else {
                    return ControlFlow::Continue(());
                };
                let addr = obj.offset(field.offset as isize) as *mut c_void;
                let mut out = TVMFFIAny::new();
                if getter(addr, &mut out) != 0 {
                    return ControlFlow::Break(take_raised_error(&format!(
                        "tirx_ext: getter failed for field `{}`",
                        field.name.as_str()
                    )));
                }
                let val_ti = out.type_index;
                let mut owned = Any::from_raw_ffi_any(out); // owns the getter's ref

                let outcome = if val_ti == TVMFFITypeIndex::kTVMFFIArray as i32 {
                    self.rebuild_array(&raw_of_owned(&mut owned), depth + 1)
                } else if val_ti >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32
                    && self.is_ir_family(val_ti)
                {
                    self.map_raw(&raw_of_owned(&mut owned), depth + 1)
                } else {
                    // Map (annotations), PODs, strings, None, foreign
                    // objects: left untouched, per StmtMutator semantics.
                    Ok(None)
                };
                match outcome {
                    Ok(Some(new_child)) => {
                        changes.push((field, new_child));
                        ControlFlow::Continue(())
                    }
                    Ok(None) => ControlFlow::Continue(()),
                    Err(e) => ControlFlow::Break(e),
                }
            })
        };
        if let Some(e) = failed {
            return Err(e);
        }

        if changes.is_empty() {
            return Ok(None);
        }
        // P1 route B: only skeleton types get a default rebuild; error loudly
        // rather than approximate C++ semantics for anything else.
        self.check_rebuild_supported(raw.type_index)?;
        let mut copy = shallow_copy(raw)?;
        let copy_obj = obj_ptr_of(&mut copy);
        for (field, new_value) in changes.iter_mut() {
            let value_raw = raw_of_owned(new_value);
            unsafe { set_field(copy_obj, field, &value_raw)? };
        }
        Ok(Some(copy))
    }

    /// Rebuild an `ffi.Array` field: map each IR-family element; a new
    /// Array (same element order, via the registered `ffi.Array` packed
    /// constructor — container.cc) only when something changed.
    fn rebuild_array(&self, raw: &TVMFFIAny, depth: u32) -> Result<Option<Any>> {
        unsafe {
            let seq = &*(raw.data_union.v_obj as *const SeqPrefix);
            if seq.size < 0 {
                return Err(Error::new(
                    RUNTIME_ERROR,
                    "tirx_ext: ffi.Array reports a negative size",
                    "",
                ));
            }
            let cells: &[TVMFFIAny] = if seq.data.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(seq.data, seq.size as usize)
            };

            let mut changed = false;
            let mut elems: Vec<Any> = Vec::with_capacity(cells.len());
            for cell in cells {
                let is_obj =
                    cell.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32;
                if is_obj && self.is_ir_family(cell.type_index) {
                    if let Some(mut new_elem) = self.map_raw(cell, depth + 1)? {
                        // Route B: an element replaced BY a SeqStmt would need
                        // C++'s SeqStmt::Flatten splicing — unsupported, error
                        // loudly instead of leaving a nested SeqStmt behind.
                        let ne_ti = raw_of_owned(&mut new_elem).type_index;
                        if type_key_of(ne_ti) == "tirx.SeqStmt" {
                            return Err(Error::new(
                                RUNTIME_ERROR,
                                "tirx_ext: a handler replaced an array element with a \
                                 tirx.SeqStmt — this needs SeqStmt::Flatten normalization, \
                                 which is outside the supported skeleton \
                                 (docs/mutator-review.md P1, route B)",
                                "",
                            ));
                        }
                        changed = true;
                        elems.push(new_elem);
                        continue;
                    }
                }
                elems.push(Any::from(view_of(cell))); // owning copy of the original
            }
            if !changed {
                return Ok(None);
            }
            let views: Vec<AnyView> = elems.iter().map(AnyView::from).collect();
            Function::get_global("ffi.Array")?.call_packed(&views).map(Some)
        }
    }
}

// ---------------------------------------------------------------------------
// The Mapper agent
// ---------------------------------------------------------------------------

/// The mapper agent: attach pass state and a function table, then drive with
/// [`Mapper::map`].  Unlike the stateful visitor, this legacy mapper adapter
/// still borrows its state through a `RefCell`.
pub struct Mapper<'s, S> {
    state: Option<&'s RefCell<S>>,
    function_table: Option<FunctionTable<S>>,
}

impl<S> Default for Mapper<'_, S> {
    fn default() -> Self {
        Self { state: None, function_table: None }
    }
}

impl<'s, S> Mapper<'s, S> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the pass's mutable state.
    pub fn visit_with_extra_content(mut self, state: &'s RefCell<S>) -> Self {
        self.state = Some(state);
        self
    }

    /// Attach the function table: the per-type handlers.
    pub fn function_table(mut self, function_table: FunctionTable<S>) -> Self {
        self.function_table = Some(function_table);
        self
    }

    /// Drive the map over `root`. COW: when nothing changes, the result is
    /// pointer-identical to `root`.
    pub fn map(&self, root: &Stmt) -> Result<Stmt> {
        self.map_with_hooks(root, default_hooks())
    }

    /// [`Mapper::map`] with an explicit (pre-resolved) per-type hook table —
    /// crate-only: the production hook set is the compiled [`TYPE_HOOKS`];
    /// this variant exists so lib.rs test globals can pin the dispatch order.
    pub(crate) fn map_with_hooks(
        &self,
        root: &Stmt,
        hooks: &[(i32, TypeHook)],
    ) -> Result<Stmt> {
        let state = self.state.expect("Mapper: call visit_with_extra_content first");
        let table = self.function_table.as_ref().expect("Mapper: call function_table first");
        let dispatch = erase_dispatch(state, table.0);
        let engine = Engine {
            dispatch: &dispatch,
            hooks,
            stmt_ti: <StmtNode as tvm_ffi::object::ObjectCore>::type_index(),
            expr_ti: <ExprNode as tvm_ffi::object::ObjectCore>::type_index(),
        };
        let raw = raw_of(AnyView::from(root));
        match engine.map_raw(&raw, 0)? {
            Some(any) => Stmt::try_from(any),
            None => Ok(root.clone()),
        }
    }
}
