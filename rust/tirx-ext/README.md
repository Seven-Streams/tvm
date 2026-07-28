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
├── rust/                  # Rust library plus its typed-dispatch proc macro
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

## Rust visitors

`#[dispatch(visit)]` turns the `visit_*` methods into a typed dispatcher.  The
visitor struct owns its mutable state, so handlers use ordinary `&mut self`
instead of a separate `RefCell` and hand-written function table:

```rust
use tvm_tirx::{dispatch, structural_visit, ForNode, VisitCtx, WalkResult};

#[derive(Default)]
struct Counter {
    loops: usize,
}

#[dispatch(visit)]
impl Counter {
    fn visit_for(&mut self, op: &ForNode, ctx: &mut VisitCtx<'_>) -> WalkResult {
        self.loops += 1;
        ctx.visit(self, &op.body);
        WalkResult::Skip
    }
}

let mut counter = Counter::default();
let _ = structural_visit(&root, &mut counter)?;
```

Typed handlers are tested in source order. Borrowed node arguments such as
`&ForNode` and `&StmtNode` use refcount-free runtime subtype checks; owned
FFI-compatible arguments handle POD values or object references, and a final
`&VisitValue` handler is a catch-all. A handler may return either `WalkResult`
or `tvm_ffi::error::Result<WalkResult>`.

`Advance` asks the native Rust walker to recurse through the current node.
`ctx.visit(self, child)` visits a selected child immediately; return `Skip`
afterwards to prevent the generic walker from visiting that child again.
`ctx.visit_with_def_region(...)` does the same under an explicit, scoped
definition region. `Interrupt` stops the whole traversal, while
`WalkResult::interrupt_with(value)` returns a payload in
`ControlFlow::Break(value)`.

The default traversal is implemented by a separate Rust walker: it reads
reflected fields and container contents through the stable tvm-ffi ABI and
re-enters the Rust dispatcher for every child. It does not construct an
`ffi.StructuralVisitor` or call C++ `DefaultVisit`. Unknown registered object
types remain walkable through reflection and can have a Rust-native override
when the pass supplies a borrowed `ObjectCore` binding and manually visits the
desired children before returning `Skip`.

Definition-region state is native too. `ctx.def_region_kind()` returns
`None`, `Recursive`, or `NonRecursive`; reflected field flags override the
inherited value for that field's subtree, while containers and
`ctx.visit(self, child)` preserve the current value. `walk_with_context`
exposes the same region to raw callbacks.

Array/List/Map/Dict have native Rust traversal paths. Foreign `__s_visit__`
functions are never invoked because their ABI requires a C++
`ffi.StructuralVisitor`. A matching pre-order Rust handler must visit the
intended children through `VisitCtx` and return `Skip`; otherwise traversal
returns an error instead of silently substituting reflected fields with
potentially different semantics.

`structural_visit_ordered` supports both pre-order and post-order typed
callbacks. Traversal errors accumulate native object/field/container path
frames. The generated API currently covers visitors; the mapper keeps its
existing explicit state/function-table API. The macro expects the dependency
name `tvm_tirx` and projects `cfg`/`cfg_attr` from handlers.

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
