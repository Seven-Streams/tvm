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
use tvm_tirx::{dispatch, structural_visit, For, VisitCtx, WalkResult};

#[derive(Default)]
struct Counter {
    loops: usize,
}

#[dispatch(visit)]
impl Counter {
    fn visit_for(&mut self, op: For, ctx: &mut VisitCtx<Self>) -> WalkResult {
        self.loops += 1;
        ctx.visit(self, &op.body);
        WalkResult::Skip
    }
}

let mut counter = Counter::default();
structural_visit(&root, &mut counter)?;
```

Handlers are tried in source order, and the first matching node type wins.
`Advance` asks the generic walker to recurse through the current node.
`ctx.visit(self, child)` visits a selected child immediately; return `Skip`
afterwards to prevent the generic walker from visiting that child again.
`Interrupt` stops the whole traversal.

The generated API currently covers visitors.  The mapper keeps its existing
explicit state/function-table API.  This minimal macro expects the dependency
name `tvm_tirx` and does not project `cfg` attributes from handlers.

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
