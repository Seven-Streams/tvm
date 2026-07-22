<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->
<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# tirx-ext

`count_loops_v10` (`rust-draft/tirx-ver3/examples/`) packaged as a pip-installable
tvm-ffi extension — the downstream-project shape:

```text
tirx-ext/                  # pip install -e <this dir>
├── pyproject.toml         # scikit-build-core backend
├── CMakeLists.txt         # invokes cargo, installs libtvm_tirx.so
├── python/tirx_ext/       # the Python package (loads the lib, exposes the API)
├── rust/                  # Rust rlib/cdylib + local typed-dispatch proc macro
└── tests/                 # pytest — IR built in Python, counted in Rust
```

`pip install -e .` → scikit-build-core → CMake → `cargo build` →
`libtvm_tirx.so`, which registers `tirx_ext.count_loops` / `tirx_ext.count_adds`
in the shared tvm-ffi global function registry.

## Usage

```python
import tvm
from tvm import tirx
import tirx_ext

stats = tirx_ext.count_loops(stmt)     # stmt: a tirx Stmt built in Python
stats["loops"], stats["total_iters"], stats["ifs"]
adds = tirx_ext.count_adds(stmt)
```

## Writing a Rust visitor

The pass author keeps mutable state and typed handlers in one object.  The
attribute macro reads the `visit_*` methods in declaration order and generates
the runtime type dispatcher:

```rust
use tvm_tirx::{dispatch, structural_visit, For, IfThenElse, VisitCtx, WalkResult};

#[derive(Default)]
struct Counter {
    loops: usize,
    ifs: usize,
}

#[dispatch(visit)]
impl Counter {
    fn visit_for(&mut self, op: For, ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
        self.loops += 1;
        if !ctx.visit(self, &op.body) {
            return WalkResult::Interrupt;
        }
        WalkResult::Skip
    }

    fn visit_if(&mut self, _op: IfThenElse, _ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
        self.ifs += 1;
        WalkResult::Advance
    }
}

let mut counter = Counter::default();
let interrupt = structural_visit(&root, &mut counter)?;
if interrupt.is_some() {
    // The walk stopped early: `counter` is partial, not a whole-tree result.
}
```

`ctx.visit(self, child)` explicitly reborrows the same visitor for a chosen
child; no separate `RefCell<Counter>` state is passed to every handler.
`Advance` asks the generic walker to visit the current node's children.
`Skip` suppresses that default recursion, which avoids visiting `op.body`
twice after the handler has already visited it explicitly.  `Interrupt` stops
the entire traversal, including all enclosing recursive visits.

The top-level return distinguishes three outcomes: `Ok(None)` means the whole
tree was visited, `Ok(Some(ffi.VisitInterrupt))` means a hook or handler stopped
successfully but early, and `Err(ffi.Error)` means traversal failed.  The `?`
operator propagates only `Err`; it does not reject `Ok(Some(...))`.  A pass that
publishes whole-tree statistics must therefore handle the `Some` case instead
of returning its partial state.  `tirx_ext.count_loops` and
`tirx_ext.count_adds` both turn an unexpected interrupt into an error.

When `ctx.visit(self, child)` returns `false`, an interrupt or error is already
pending.  Restore any scoped state (such as an enclosing-loop multiplier) and
return promptly; the traversal driver will propagate the original halt.

Each explicit `ctx.visit` edge uses a fresh low-level FFI traversal frame while
preserving the definition-region mode.  Custom structural hooks therefore must
not use the address of the low-level `StructuralVisitorObj` as persistent
identity across that edge.

The visitor handle used by the low-level `structural_visit_manual` API is
heap-backed and may be retained safely by an FFI hook, but its callback frame
is scoped to one traversal.  Once that traversal returns, a retained handle is
inert: a later visit returns `ffi.VisitInterrupt` and never re-enters expired
Rust callback state.  As with the upstream structural visitor, the same handle
must not be used by overlapping traversals.

The package builds an `rlib` for Rust pass crates as well as the Python-facing
`cdylib`, and re-exports `dispatch` and the high-level visitor types above.  If
the dependency is renamed in `Cargo.toml`, select that path explicitly, for
example `#[dispatch(visit, runtime = my_tvm_tirx)]`.

This generated stateful API currently covers the `visit` family.  The mapper
module still exposes its older explicit state/function-table API while the
corresponding generated `mutate` family is developed.

## Development (in-repo)

```sh
# build tir itself (provides build/lib/libtvm_ffi.so + libtvm_compiler.so)
# force a full recompile from a clean build/:
rm -rf build && mkdir -p build && cp cmake/config.cmake build/config.cmake
cmake -S . -B build && cmake --build build -j"$(nproc)"

# then, inside the repo venv:
pip install -e rust/tirx-ext --no-build-isolation
pytest rust/tirx-ext/tests

# fast Rust iteration without re-running pip:
PATH="$PWD/rust/tirx-ext/tools:$PATH" cargo build --release \
    --manifest-path rust/tirx-ext/rust/Cargo.toml
```

The crate links against tir's own `build/lib/libtvm_ffi.so` (resolution order:
`TIRX_LIB_DIR` → repo `build/lib` → `tvm-ffi-config --libdir`), and the Python
package `import tvm`s before loading the cdylib — one libtvm_ffi, one registry.
