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

//! Layout-identical bindings for the slice of the tirx C++ IR the passes
//! touch: `#[repr(C)]` node structs for field reads, ref newtypes for FFI
//! argument passing. Expression nodes are read-only (only ever *received*
//! from the host); `Stmt`-family nodes ([`stmt_node!`]) additionally carry a
//! **native positional constructor** — the mutator rebuilds claimed nodes in
//! pure Rust, no FFI constructor calls (unclaimed types go through the
//! engine's generic shallow-copy + setter path instead).
//!
//! Layout mirrors C++ exactly (`A { B base; own... }`):
//! ```text
//! ffi::Object   -> tvm_ffi::Object                     (24B header @0)
//!   ExprNode    -> { Object base; Span span; Type ty } (span @24, ty @32; size 40)
//!   StmtNode    -> { Object base; Span span }          (span @24; size 32)
//! ```
//!
//! Note: the former `ir.BaseExpr`/`ir.PrimExpr` split collapsed into a single
//! reflected `ir.Expr`, and the old `DataType dtype` slot is now the `Type ty`
//! field. Nullable slots that are `ffi::Optional<T>` on the C++ side use the
//! uniform 16-byte [`tvm_ffi::Optional`] cell (the "stable Optional layout");
//! a bare `Span` field stays an 8-byte nullable object pointer.

use tvm_ffi::object::ObjectRef;
use tvm_ffi::{Map, ObjectArc, Optional, String};

use crate::object_ref::impl_object_ref;

/// A source span: a nullable `ffi::ObjectRef` (C++ `tvm::Span`). `None` == null.
pub type Span = Option<ObjectRef>;

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

/// Implement `ObjectCore` for a `#[repr(C)]` node whose first field is `base:
/// $basety`. `type_index` is resolved from the C++ registry by type key and
/// cached per-type. (We deliberately do *not* use tvm-ffi's `#[derive(Object)]`:
/// its auto-`type_index` path expands a `proc_macro_error::abort!` into runtime
/// code, which does not resolve in a normal crate.)
macro_rules! impl_object_core {
    ($node:ty, $key:literal, $basety:ty) => {
        unsafe impl tvm_ffi::object::ObjectCore for $node {
            const TYPE_KEY: &'static str = $key;
            #[inline]
            fn type_index() -> i32 {
                // OnceLock, not LazyLock: a panicking initializer (type not yet
                // registered — host called us before `import tvm`) leaves the
                // cell uninitialized, so the lookup retries once the host fixes
                // its load order instead of staying poisoned forever.
                static IDX: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
                *IDX.get_or_init(|| $crate::runtime::lookup_type_index($key))
            }
            #[inline]
            unsafe fn object_header_mut(
                this: &mut Self,
            ) -> &mut tvm_ffi::tvm_ffi_sys::TVMFFIObject {
                <$basety as tvm_ffi::object::ObjectCore>::object_header_mut(&mut this.base)
            }
        }
    };
}

/// Define a node struct (`$basety` prefix at offset 0, own fields after) plus
/// its ref newtype: `ObjectCore` + `AnyCompatible` glue and `Deref` so
/// `r.field` reads node fields (the Rust spelling of the C++ ref-class
/// `operator->`).
macro_rules! node {
    ($refname:ident, $nodename:ident, $key:literal, base = $basety:ty,
        { $($fname:ident : $fty:ty),* $(,)? }) => {
        #[repr(C)]
        pub struct $nodename {
            pub(crate) base: $basety,
            $(pub $fname: $fty,)*
        }
        impl_object_core!($nodename, $key, $basety);

        #[derive(Clone)]
        pub struct $refname {
            data: ObjectArc<$nodename>,
        }
        impl_object_ref!($refname, $nodename);

        impl ::core::ops::Deref for $refname {
            type Target = $nodename;
            #[inline]
            fn deref(&self) -> &$nodename {
                unsafe { &*ObjectArc::as_raw(&self.data) }
            }
        }
    };
}

/// A [`node!`] in the `Stmt` family, plus **native construction**: a
/// positional all-fields constructor (`span` last) building the `#[repr(C)]`
/// struct directly — `ObjectArc::new` stamps the runtime type index and a
/// Rust deleter (object.rs:299), so C++ owns/frees it like any of its own —
/// and the `From<Ref> for Stmt` upcast (offset-0 prefix retype, the draft's
/// `impl_upcast!`). No FFI constructor call anywhere on this path.
macro_rules! stmt_node {
    ($refname:ident, $nodename:ident, $key:literal,
        { $($fname:ident : $fty:ty),* $(,)? }) => {
        node!($refname, $nodename, $key, base = StmtNode, { $($fname : $fty),* });

        impl $refname {
            /// Construct natively from every field value (`span` last).
            #[allow(clippy::new_without_default, clippy::too_many_arguments)]
            pub fn new($($fname: $fty,)* span: Span) -> $refname {
                let node = $nodename {
                    base: StmtNode { base: tvm_ffi::Object::new(), span },
                    $($fname,)*
                };
                $refname { data: ObjectArc::new(node) }
            }
        }

        // Upcast: sound because the node embeds `StmtNode` as its offset-0
        // prefix, so the object pointer is identical; only the arc is retyped
        // (ownership moves, no refcount change).
        impl From<$refname> for Stmt {
            #[inline]
            fn from(x: $refname) -> Stmt {
                use tvm_ffi::object::ObjectRefCore;
                let arc = <$refname as ObjectRefCore>::into_data(x);
                let up: ObjectArc<StmtNode> =
                    unsafe { ObjectArc::from_raw(ObjectArc::into_raw(arc) as *const StmtNode) };
                <Stmt as ObjectRefCore>::from_data(up)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Base chain (abstract; never constructed)
// ---------------------------------------------------------------------------

/// `ir.Expr` — `{ Object; Span span; Type ty }` (span @24, ty @32; size 40).
/// The common base of every tirx expression. Replaces the former
/// `ir.BaseExpr`/`ir.PrimExpr` split; the old `dtype` slot is now the reflected
/// `ty` field.
#[repr(C)]
pub struct ExprNode {
    pub(crate) base: tvm_ffi::Object,
    pub span: Span,
    // C++ `Type ty`: a non-null object ref. We never dereference it (only the
    // 8-byte slot matters for layout), so mirror it as an opaque nullable ref.
    pub ty: Option<ObjectRef>,
}
impl_object_core!(ExprNode, "ir.Expr", tvm_ffi::Object);

/// `tirx.Stmt` — `{ Object; Span span }`.
#[repr(C)]
pub struct StmtNode {
    pub(crate) base: tvm_ffi::Object,
    pub span: Span,
}
impl_object_core!(StmtNode, "tirx.Stmt", tvm_ffi::Object);

/// Managed ref to any [`ExprNode`] (a tirx expression).
#[derive(Clone)]
pub struct PrimExpr {
    data: ObjectArc<ExprNode>,
}
impl_object_ref!(PrimExpr, ExprNode);

/// Managed ref to any [`StmtNode`].
#[derive(Clone)]
pub struct Stmt {
    data: ObjectArc<StmtNode>,
}
impl_object_ref!(Stmt, StmtNode);

impl PrimExpr {
    /// Checked downcast to a concrete node `N` by comparing the object header's
    /// runtime `type_index` with `N`'s. Returns `None` on mismatch.
    #[inline]
    pub fn downcast<N: tvm_ffi::object::ObjectCore>(&self) -> Option<&N> {
        downcast_ref::<PrimExpr, N>(self)
    }
}

impl Stmt {
    /// Checked downcast to a concrete statement node `N` (see [`PrimExpr::downcast`]).
    #[inline]
    pub fn downcast<N: tvm_ffi::object::ObjectCore>(&self) -> Option<&N> {
        downcast_ref::<Stmt, N>(self)
    }
}

/// Shared downcast helper: reinterpret the object pointer as `*const N` iff the
/// header's runtime `type_index` equals `N::type_index()`.
#[inline]
fn downcast_ref<R, N>(r: &R) -> Option<&N>
where
    R: tvm_ffi::object::ObjectRefCore,
    N: tvm_ffi::object::ObjectCore,
{
    unsafe {
        let raw = ObjectArc::as_raw(<R as tvm_ffi::object::ObjectRefCore>::data(r)) as *const N;
        let header = raw as *const tvm_ffi::tvm_ffi_sys::TVMFFIObject;
        if (*header).type_index == <N as tvm_ffi::object::ObjectCore>::type_index() {
            Some(&*raw)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Concrete nodes (field order mirrors the C++ headers exactly)
// ---------------------------------------------------------------------------

// `tirx.Var` — { name_hint: String } (reflected as `name`). Base is `ir.Expr`;
// the inherited `ty` carries the variable's type, so there is no own
// `type_annotation` field in the current tirx IR.
node!(Var, VarNode, "tirx.Var", base = ExprNode, {
    name_hint: String,
});

// `tirx.For` — { loop_var, min, extent, kind, body, thread_binding, annotations, step }.
stmt_node!(For, ForNode, "tirx.For", {
    loop_var: Var,
    min: PrimExpr,
    extent: PrimExpr,
    kind: i32,
    body: Stmt,
    // C++ `Optional<IterVar>`: the uniform 16-byte `ffi::Optional<T>` cell
    // (stable Optional layout), read as an opaque object ref.
    thread_binding: Optional<ObjectRef>,
    // C++ type is Map<String, Any>; annotation *values* in TIR are objects
    // (IntImm/StringImm/Array/…), and `Map` is phantom-typed (pointer-only
    // layout), so `ObjectRef` values are layout-identical.
    annotations: Map<String, ObjectRef>,
    step: Optional<PrimExpr>,
});

// `tirx.IfThenElse` — { condition; then_case; else_case (nullable) }.
stmt_node!(IfThenElse, IfThenElseNode, "tirx.IfThenElse", {
    condition: PrimExpr,
    then_case: Stmt,
    else_case: Optional<Stmt>,
});

// `tirx.While` — { condition; body }. Trip count is statically unknowable, so
// the counting passes exclude the whole subtree (see lib.rs).
stmt_node!(While, WhileNode, "tirx.While", {
    condition: PrimExpr,
    body: Stmt,
});

// `tirx.Break` — no own fields (just the inherited span); `Break::new(None)`
// is pure allocation + header stamp.
stmt_node!(Break, BreakNode, "tirx.Break", {});

/// `ir.IntImm` — `{ value: int64 }` (base-IR layer, `ir.` prefix). Only ever
/// downcast to, so no ref newtype is generated.
#[repr(C)]
pub struct IntImmNode {
    pub(crate) base: ExprNode,
    pub value: i64,
}
impl_object_core!(IntImmNode, "ir.IntImm", ExprNode);

/// `tirx.Add` — `{ a: PrimExpr; b: PrimExpr }`. Only ever type-tested, so no
/// ref newtype is generated.
#[repr(C)]
pub struct AddNode {
    pub(crate) base: ExprNode,
    pub a: PrimExpr,
    pub b: PrimExpr,
}
impl_object_core!(AddNode, "tirx.Add", ExprNode);
