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

//! # tirx_ext — `count_loops_v10` packaged as a tvm-ffi extension
//!
//! The pip package's Rust side: `rust-draft/tirx-ver3/examples/count_loops_v10.rs`
//! (the loop-counting pass), turned
//! from a demo binary into two global functions a Python host calls
//! in-process — the IR crosses as a borrowed `AnyView`, zero serialization:
//!
//! * `tirx_ext.count_loops(stmt)` → `Map<String, i64>` with keys `loops`,
//!   `total_iters`, `ifs`, `branch_execs`, `innermost`;
//! * `tirx_ext.count_adds(stmt)` → `Map<String, i64>` with keys `adds`,
//!   `add_execs`;
//! * `tirx_ext.break_for_bodies(stmt)` → the rewritten `Stmt` (the demo
//!   mutation pass, engine in `mutate.rs` — COW, original tree untouched);
//! * `tirx_ext.break_innermost_for_bodies(stmt)` → the rewritten `Stmt`
//!   (the handler-re-entrancy demo: only innermost `For` bodies break).
//!
//! Differences from the example, required by the general-tool contract:
//! the demo's `assert!(innermost == 1)` is dropped (the count is returned
//! instead); every `.expect("constant extent")` panic became a flagged
//! [`WalkResult::Interrupt`] surfaced as a proper `ffi.Error`; trip counts
//! honor `ForNode.step` (`ceildiv(extent, step)`, the `s_tir.CanonicalizeLoop`
//! normalization; non-constant or non-positive steps error); `tirx.While`
//! subtrees are excluded from every counter (unknowable trip count); and the
//! iteration arithmetic is overflow-checked (an `ffi.Error` instead of a
//! silently wrapped total).
//!
//! The host must `import tvm` (or otherwise load `libtvm_compiler.so`) before
//! loading this cdylib and calling [`tirx_ext_init`]: the `tirx.*` types this
//! crate looks up are registered by that library's static initializers, and
//! the dynamic linker's soname dedup then binds both sides to one
//! `libtvm_ffi.so` — a single global function registry in the process.

// The proc macro uses the public crate path both here and in downstream crates.
extern crate self as tvm_tirx;

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};

use tvm_ffi::any::{Any, AnyView};
use tvm_ffi::error::{Error, Result, RUNTIME_ERROR, TYPE_ERROR};
use tvm_ffi::function::Function;
use tvm_ffi::{Map, String as FfiString};

mod layout;
pub mod mutate;
mod node;
mod object_ref;
mod reflect;
mod runtime;
pub mod visit;

pub use node::{
    AddNode, Break, BreakNode, ExprNode, For, ForNode, IfThenElse, IfThenElseNode, IntImmNode,
    PrimExpr, Span, Stmt, StmtNode, Var, VarNode, While, WhileNode,
};
pub use tvm_tirx_macros::dispatch;
pub use visit::{structural_visit, VisitCtx, VisitDispatch, WalkResult};

use mutate::MapCtx;
use visit::{walk, Phase, VisitValue};

/// The mapper's `function_table!`: the same first-match-wins chain. Each
/// handler returns `Result` of its node (`Stmt`/`PrimExpr` — original ==
/// unchanged, per C++ StmtMutator; `Err` aborts the map), type-erased to
/// `Any` so one table can mix node families.
macro_rules! mutation_table {
    ( $($ty:ty => $handler:expr),+ $(,)? ) => {
        mutate::FunctionTable(|state, v, mapper| {
            None$( .or_else(|| v.cast::<$ty>().map(|x| $handler(state, x, mapper).map(Any::from))) )+
        })
    };
}

// ---------------------------------------------------------------------------
// The counting passes.
// ---------------------------------------------------------------------------

// Pass-level error conditions, stored as the message and mapped to an
// ffi.Error at the entry points.
const ERR_EXTENT: &str = "tirx_ext: loop extent is not a constant ir.IntImm";
const ERR_STEP: &str = "tirx_ext: loop step is not a constant ir.IntImm";
const ERR_STEP_POSITIVE: &str = "tirx_ext: loop step must be a positive constant";
const ERR_OVERFLOW: &str = "tirx_ext: iteration count overflows i64";

fn pass_error(msg: &'static str) -> Error {
    let kind = if msg == ERR_OVERFLOW { RUNTIME_ERROR } else { TYPE_ERROR };
    Error::new(kind, msg, "")
}

/// Constant trip count of a `For`: `ceildiv(extent, step)`, step defaulting to
/// one — the same normalization C++ `s_tir.CanonicalizeLoop` applies (which
/// also rejects non-positive steps).
fn for_trip_count(op: &ForNode) -> Result<i64, &'static str> {
    let extent = op.extent.downcast::<IntImmNode>().ok_or(ERR_EXTENT)?.value;
    let step = match op.step.get() {
        None => 1,
        Some(s) => s.downcast::<IntImmNode>().ok_or(ERR_STEP)?.value,
    };
    if step < 1 {
        return Err(ERR_STEP_POSITIVE);
    }
    // Overflow-free ceildiv for step >= 1.
    Ok(extent.div_euclid(step) + i64::from(extent.rem_euclid(step) != 0))
}

/// Gathered by the generated typed visitor (For / If, custom order).
#[derive(Debug, Default)]
struct Counter {
    loops: i64,
    total_iters: i64,
    ifs: i64,
    branch_execs: i64,
    innermost: i64,
    outer_iter_num: i64,
    is_for_in_body: bool,
    error: Option<&'static str>,
}

#[dispatch(visit)]
impl Counter {
    fn enter_for(&mut self, trips: i64) -> Result<i64, &'static str> {
        self.loops += 1;
        let saved = self.outer_iter_num;
        let inner = saved.checked_mul(trips).ok_or(ERR_OVERFLOW)?;
        self.total_iters = self.total_iters.checked_add(inner).ok_or(ERR_OVERFLOW)?;
        self.outer_iter_num = inner;
        self.is_for_in_body = false;
        Ok(saved)
    }

    fn exit_for(&mut self, saved: i64) {
        if !self.is_for_in_body {
            self.innermost += 1;
        }
        self.outer_iter_num = saved;
        self.is_for_in_body = true;
    }

    fn count_branch(&mut self) -> Result<(), &'static str> {
        self.ifs += 1;
        self.branch_execs = self
            .branch_execs
            .checked_add(self.outer_iter_num)
            .ok_or(ERR_OVERFLOW)?;
        Ok(())
    }

    fn visit_for(&mut self, op: For, ctx: &mut VisitCtx<Self>) -> WalkResult {
        if !ctx.visit(self, &op.extent) {
            return WalkResult::Interrupt;
        }
        if let Some(step) = op.step.get() {
            if !ctx.visit(self, &step) {
                return WalkResult::Interrupt;
            }
        }

        let saved = match for_trip_count(&op).and_then(|trips| self.enter_for(trips)) {
            Ok(saved) => saved,
            Err(msg) => {
                self.error = Some(msg);
                return WalkResult::Interrupt;
            }
        };
        if !ctx.visit(self, &op.body) {
            self.exit_for(saved);
            return WalkResult::Interrupt;
        }
        self.exit_for(saved);
        WalkResult::Skip
    }

    fn visit_if(&mut self, op: IfThenElse, ctx: &mut VisitCtx<Self>) -> WalkResult {
        if !ctx.visit(self, &op.condition) {
            return WalkResult::Interrupt;
        }
        if let Err(msg) = self.count_branch() {
            self.error = Some(msg);
            return WalkResult::Interrupt;
        }
        if !ctx.visit(self, &op.then_case) {
            return WalkResult::Interrupt;
        }
        if let Some(else_case) = op.else_case.get() {
            if !ctx.visit(self, &else_case) {
                return WalkResult::Interrupt;
            }
        }
        WalkResult::Skip
    }

    fn visit_while(&mut self, _op: While, _ctx: &mut VisitCtx<Self>) -> WalkResult {
        WalkResult::Skip
    }
}

/// Count For/If statements and their executions, `count_loops_v10` style: the
/// handlers drive their children explicitly (extent → state update → body)
/// through the typed visitor. Trip counts honor `step` (`ceildiv(extent,
/// step)`); `While` subtrees are excluded entirely (statically unknowable trip
/// count — counting them would fabricate numbers).
fn count_loops(root: &Stmt) -> Result<Counter> {
    let mut counter = Counter {
        outer_iter_num: 1,
        ..Default::default()
    };
    structural_visit(root, &mut counter)?;

    if let Some(msg) = counter.error {
        return Err(pass_error(msg));
    }
    Ok(counter)
}

/// Gathered by the plain walk (Add is a pure observation). Type tests are
/// borrow-only (`as_node`) and the cheap phase check runs first.
#[derive(Debug, Default)]
struct AddStats {
    adds: i64,
    add_execs: i64,
    prod_stack: Vec<i64>,
    error: Option<&'static str>,
}

impl AddStats {
    fn observe(&mut self, v: &VisitValue, phase: Phase) -> WalkResult {
        if let Some(op) = v.as_node::<ForNode>() {
            match phase {
                Phase::Enter => {
                    let pushed = for_trip_count(op)
                        .and_then(|trips| self.prod().checked_mul(trips).ok_or(ERR_OVERFLOW));
                    match pushed {
                        Ok(total) => self.prod_stack.push(total),
                        Err(msg) => {
                            self.error = Some(msg);
                            return WalkResult::Interrupt;
                        }
                    }
                }
                Phase::Exit => {
                    self.prod_stack.pop();
                }
            }
        } else if v.as_node::<WhileNode>().is_some() {
            // Unknowable trip count: exclude the whole While subtree.
            return WalkResult::Skip;
        } else if phase == Phase::Enter && v.as_node::<AddNode>().is_some() {
            self.adds += 1;
            match self.add_execs.checked_add(self.prod()) {
                Some(total) => self.add_execs = total,
                None => {
                    self.error = Some(ERR_OVERFLOW);
                    return WalkResult::Interrupt;
                }
            }
        }
        WalkResult::Advance
    }

    fn prod(&self) -> i64 {
        *self.prod_stack.last().expect("prod_stack is seeded with 1")
    }
}

/// Count `tirx.Add` nodes and their executions with the plain walk engine.
/// Same trip-count semantics as [`count_loops`]: `step`-aware, `While`
/// subtrees excluded.
fn count_adds(root: &Stmt) -> Result<AddStats> {
    let mut stats = AddStats {
        prod_stack: vec![1],
        ..Default::default()
    };
    walk(root, |v, phase| stats.observe(v, phase))?;
    if let Some(msg) = stats.error {
        return Err(pass_error(msg));
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// The demo mutation pass.
// ---------------------------------------------------------------------------

/// Rebuild `op` positionally with a new body, every other field cloned (a
/// refcount inc each: the subtrees stay shared, COW). Fully native — no FFI
/// on this path.
fn for_with_body(op: &ForNode, body: Stmt) -> Stmt {
    For::new(
        op.loop_var.clone(),
        op.min.clone(),
        op.extent.clone(),
        op.kind,
        body,
        op.thread_binding.clone(),
        op.annotations.clone(),
        op.step.clone(),
        op.base.span.clone(),
    )
    .into()
}

/// Replace every `For` body with a `tirx.Break` statement — the demo mapper
/// pass (`docs/mutator-proposal.md`). The handler takes full control of its
/// node, C++ `VisitStmt_`-override style: it does *not* recurse into the
/// original body, so nested loops vanish with it; the min/extent/step subtrees
/// are shared untouched (COW), and a tree containing no `For` comes back
/// pointer-identical.
fn break_for_bodies(root: &Stmt) -> Result<Stmt> {
    fn rewrite_for(_state: &RefCell<()>, op: For, _mapper: &mut MapCtx) -> Result<Stmt> {
        Ok(for_with_body(&op, Break::new(None).into()))
    }

    let state = RefCell::new(());
    let table = mutation_table! {
        For => rewrite_for,
    };
    mutate::Mapper::new()
        .visit_with_extra_content(&state)
        .function_table(table)
        .map(root)
}

/// Replace the body of every **innermost** `For` (no `For` anywhere in its
/// body's value positions) with `tirx.Break`, keeping the loop structure
/// above it — the re-entrancy demo (`docs/mutator-redesign.md` §5): the
/// handler decides from the result of driving its own child through the
/// engine, which the pre-redesign API could not express.
fn break_innermost_for_bodies(root: &Stmt) -> Result<Stmt> {
    fn rewrite_for(_state: &RefCell<()>, op: For, mapper: &mut MapCtx) -> Result<Stmt> {
        let body = mapper.map(&op.body)?;
        if !body.same_as(&op.body) {
            // Deeper Fors were rewritten — not innermost: keep this loop,
            // adopt the mapped body (min/extent/step stay shared, COW).
            return Ok(for_with_body(&op, body));
        }
        // Unchanged body ⟺ no For in its value positions, because every For
        // this handler reaches returns a fresh node (both branches rebuild —
        // even an already-Break body gets a new Break, keeping the
        // equivalence sound for nested pre-broken loops).
        Ok(for_with_body(&op, Break::new(None).into()))
    }

    let state = RefCell::new(());
    let table = mutation_table! {
        For => rewrite_for,
    };
    mutate::Mapper::new()
        .visit_with_extra_content(&state)
        .function_table(table)
        .map(root)
}

// ---------------------------------------------------------------------------
// Test-only passes (underscore-registered, driven by tests/test_mutate.py) —
// they pin engine behavior that the demo passes cannot reach from Python:
// the TYPE_HOOKS dispatch order, the map_fields idiom, the error channel.
// ---------------------------------------------------------------------------

/// A [`mutate::TypeHook`] replacing every `tirx.While` with `Break` — a
/// deliberately loud stand-in rule (TYPE_HOOKS proper ships empty).
fn hook_while_to_break(_v: &VisitValue, _mapper: &mut MapCtx) -> Result<Any> {
    Ok(Any::from(Stmt::from(Break::new(None))))
}

const TEST_HOOKS: &[(&str, mutate::TypeHook)] = &[("tirx.While", hook_while_to_break)];

/// Pass table misses → the While hook must fire (hook > default rebuild).
fn map_test_hook_dispatch(root: &Stmt) -> Result<Stmt> {
    fn keep_if(_state: &RefCell<()>, op: IfThenElse, _mapper: &mut MapCtx) -> Result<Stmt> {
        Ok(op.into())
    }
    let state = RefCell::new(());
    let table = mutation_table! {
        IfThenElse => keep_if,
    };
    mutate::Mapper::new()
        .visit_with_extra_content(&state)
        .function_table(table)
        .map_with_hooks(root, &mutate::resolve_hooks(TEST_HOOKS))
}

/// Pass table claims While (identity) → the hook must NOT fire (table > hook):
/// the whole tree comes back pointer-identical.
fn map_test_table_wins(root: &Stmt) -> Result<Stmt> {
    fn keep_while(_state: &RefCell<()>, op: While, _mapper: &mut MapCtx) -> Result<Stmt> {
        Ok(op.into())
    }
    let state = RefCell::new(());
    let table = mutation_table! {
        While => keep_while,
    };
    mutate::Mapper::new()
        .visit_with_extra_content(&state)
        .function_table(table)
        .map_with_hooks(root, &mutate::resolve_hooks(TEST_HOOKS))
}

/// The `map_fields` idiom: the For handler defers to the default rebuild of
/// its own fields (children re-enter the table, where While → Break).
fn map_test_map_fields(root: &Stmt) -> Result<Stmt> {
    fn for_via_map_fields(_state: &RefCell<()>, op: For, mapper: &mut MapCtx) -> Result<Stmt> {
        Ok(mapper.map_fields(&op)?.into())
    }
    fn break_while(_state: &RefCell<()>, _op: While, _mapper: &mut MapCtx) -> Result<Stmt> {
        Ok(Break::new(None).into())
    }
    let state = RefCell::new(());
    let table = mutation_table! {
        For => for_via_map_fields,
        While => break_while,
    };
    mutate::Mapper::new()
        .visit_with_extra_content(&state)
        .function_table(table)
        .map(root)
}

/// The single error channel: a handler `Err` surfaces as one `ffi.Error`.
fn map_test_handler_error(root: &Stmt) -> Result<Stmt> {
    fn fail_for(_state: &RefCell<()>, _op: For, _mapper: &mut MapCtx) -> Result<Stmt> {
        Err(Error::new(RUNTIME_ERROR, "tirx_ext: test handler failure", ""))
    }
    let state = RefCell::new(());
    let table = mutation_table! {
        For => fail_for,
    };
    mutate::Mapper::new()
        .visit_with_extra_content(&state)
        .function_table(table)
        .map(root)
}

// ---------------------------------------------------------------------------
// FFI exports.
// ---------------------------------------------------------------------------

fn map_of(pairs: &[(&str, i64)]) -> Map<FfiString, i64> {
    pairs.iter().map(|(k, v)| (FfiString::from(*k), *v)).collect()
}

/// Best-effort text of a caught panic payload (panic! with a literal gives
/// `&str`, with a format string gives `String`).
fn panic_message(p: &(dyn std::any::Any + Send)) -> &str {
    p.downcast_ref::<&'static str>()
        .copied()
        .or_else(|| p.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

/// Register one packed-function global (idempotent: `register_global` uses
/// can_override=0, so probe first). The body is `catch_unwind`-wrapped — the
/// ffi trampoline does not catch panics — and a caught panic surfaces its
/// message in the `ffi.Error` instead of unwinding into the C++ caller.
fn register_packed(
    name: &'static str,
    f: impl Fn(&[AnyView]) -> Result<Any> + 'static,
) -> Result<()> {
    if Function::get_global(name).is_ok() {
        return Ok(());
    }
    let func = Function::from_packed(move |args: &[AnyView]| -> Result<Any> {
        catch_unwind(AssertUnwindSafe(|| f(args))).unwrap_or_else(|p| {
            Err(Error::new(
                RUNTIME_ERROR,
                &format!("{name} panicked: {}", panic_message(p.as_ref())),
                "",
            ))
        })
    });
    Function::register_global(name, func)
}

/// Register one `Stmt -> R` global (counting passes return a Map, mutation
/// passes return the rewritten Stmt).
fn register<R: tvm_ffi::type_traits::AnyCompatible + 'static>(
    name: &'static str,
    f: fn(&Stmt) -> Result<R>,
) -> Result<()> {
    register_packed(name, move |args: &[AnyView]| -> Result<Any> {
        if args.len() != 1 {
            return Err(Error::new(
                TYPE_ERROR,
                "tirx_ext functions expect exactly one argument (a tirx Stmt)",
                "",
            ));
        }
        let root = Stmt::try_from(args[0])?;
        Ok(Any::from(f(&root)?))
    })
}

/// Register this crate's globals (idempotent):
///
/// * `tirx_ext.count_loops(stmt)` → `{loops, total_iters, ifs, branch_execs, innermost}`
/// * `tirx_ext.count_adds(stmt)`  → `{adds, add_execs}`
/// * `tirx_ext._check_layouts()`  → `true`, or an `ffi.Error` naming every
///   Rust-vs-C++ field-offset mismatch (run by python/tirx_ext on import).
pub fn register_globals() -> Result<()> {
    register("tirx_ext.count_loops", |root| {
        let c = count_loops(root)?;
        Ok(map_of(&[
            ("loops", c.loops),
            ("total_iters", c.total_iters),
            ("ifs", c.ifs),
            ("branch_execs", c.branch_execs),
            ("innermost", c.innermost),
        ]))
    })?;
    register("tirx_ext.count_adds", |root| {
        let s = count_adds(root)?;
        Ok(map_of(&[("adds", s.adds), ("add_execs", s.add_execs)]))
    })?;
    register("tirx_ext.break_for_bodies", break_for_bodies)?;
    register("tirx_ext.break_innermost_for_bodies", break_innermost_for_bodies)?;
    register("tirx_ext._map_test_hook_dispatch", map_test_hook_dispatch)?;
    register("tirx_ext._map_test_table_wins", map_test_table_wins)?;
    register("tirx_ext._map_test_map_fields", map_test_map_fields)?;
    register("tirx_ext._map_test_handler_error", map_test_handler_error)?;
    register_packed("tirx_ext._check_layouts", |_args| {
        layout::check_layouts().map_err(|msg| Error::new(RUNTIME_ERROR, &msg, ""))?;
        Ok(Any::from(true))
    })
}

/// C entry point for hosts that `dlopen` this cdylib (python/tirx_ext does).
/// Returns 0 on success. Safe to call more than once.
///
/// Registration is an explicit call rather than an `.init_array` constructor
/// on purpose: failure inside `dlopen`-time init cannot cross the FFI boundary
/// as an error (a C++ exception there aborts a Rust host, a Rust panic there
/// aborts a C host) — an explicit init returns a status the host can turn
/// into a clean exception. The failure detail goes to stderr — a plain i32 is
/// all the C ABI offers here, and silence would leave nothing to debug with.
#[no_mangle]
pub extern "C" fn tirx_ext_init() -> i32 {
    match catch_unwind(register_globals) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            eprintln!("tirx_ext_init failed: {e:?}");
            -1
        }
        Err(p) => {
            eprintln!("tirx_ext_init panicked: {}", panic_message(p.as_ref()));
            -1
        }
    }
}
