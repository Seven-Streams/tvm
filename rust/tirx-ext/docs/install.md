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

# tirx-ext installation

From nothing to `import tirx_ext` and a running pass. Build/install only — to
*use* or *write* passes, see `usage.md`.

Paths are relative to the **repo root**; run commands there unless noted.

```sh
PKG=rust/tirx-ext
PY=.venv/bin/python
```

---

## 0. Prerequisites

tirx-ext has no C++ of its own — install is just scikit-build-core → CMake →
`cargo build` → `libtvm_tirx.so`. That library must link tir's **own**
`libtvm_ffi.so` (one process, one registry — see the note below), so every
prerequisite amounts to "tir's dev environment is in place".

| # | Dependency | Check | If missing |
|---|---|---|---|
| 1 | tir built (both .so) | `ls build/lib/libtvm_ffi.so build/lib/libtvm_compiler.so` | Build tir — §0.1 |
| 2 | venv has `tvm_ffi` | `$PY -c "import tvm_ffi"` | Install a tvm-ffi checkout: `uv pip install --python $PY -e <tvm-ffi>/python` |
| 3 | `tvm` source tree | `PYTHONPATH=python $PY -c "import tvm"` | Not installed by design — always run with `PYTHONPATH=python` (§2) |
| 4 | Rust toolchain | `cargo --version` | Any recent stable |
| 5 | CMake ≥ 3.18 | `cmake --version` | — |
| 6 | scikit-build-core in venv | `$PY -c "import scikit_build_core"` | `uv pip install --python $PY scikit-build-core` |
| 7 | tvm-ffi submodule | `ls 3rdparty/tvm-ffi/rust/tvm-ffi/Cargo.toml` | `git submodule update --init 3rdparty/tvm-ffi` |

> **One libtvm_ffi, one registry.** `libtvm_compiler.so` (it registers the
> `tirx.*` types) was built against `build/lib/libtvm_ffi.so`; the crate must
> link that **same** .so — otherwise `tirx_ext.*` registers into one registry
> while Python looks in another and the functions vanish. The
> `tools/tvm-ffi-config` shim points the build's `--libdir` query at `build/lib`
> for exactly this reason.

### 0.1 Build tir

The only heavy step. Force a clean recompile from the repo root; it produces
both `.so` under `build/lib/`:

```sh
rm -rf build && mkdir -p build && cp cmake/config.cmake build/config.cmake
cmake -S . -B build
cmake --build build -j"$(nproc)"          # ~minutes; whole tree
```

`tvm` itself is never installed — it's imported from `python/` via
`PYTHONPATH=python` at runtime (§2).

---

## 1. Install (editable)

```sh
uv pip install --python $PY -e $PKG --no-build-isolation
```

- `-e`: Python edits take effect immediately; the build writes `libtvm_tirx.so`
  to `$PKG/build-wheel/cargo/{release,debug}/`.
- `--no-build-isolation`: reuse the venv's scikit-build-core (prereq #6) and let
  the build see the tir environment.
- Release profile by default (`pyproject.toml`: `cmake.build-type = "Release"`).

`libtvm_ffi.so` is resolved in order: `-DTIRX_LIB_DIR=` / env `TIRX_LIB_DIR` →
`build/lib` (in-repo default) → `tvm-ffi-config --libdir` (external wheel).
In-repo needs no configuration. Success ends with
`Installed 1 package … tirx-ext==0.1.0`.

---

## 2. Runtime environment (mandatory)

Two conditions, or `import tirx_ext` fails:

1. **`PYTHONPATH=python`** — `import tirx_ext` internally does `import tvm`,
   which lives only in the source tree. Without it: `ModuleNotFoundError: No
   module named 'tvm'`.
2. **`import tvm` before `import tirx_ext`** — so the cdylib binds to the
   already-loaded libtvm_ffi. `_ffi_api.py` enforces this at module scope; in
   your own scripts, just keep tvm first.

```python
import tvm                      # first
from tvm import tirx
import tirx_ext                 # _check_layouts() runs here; errors on field-offset drift
```

---

## 3. Verify

```sh
PYTHONPATH=python $PY - <<'EOF'
import tvm
from tvm import tirx
import tirx_ext
print("loaded:", tirx_ext.__file__)

i = tirx.Var("i", "int32"); j = tirx.Var("j", "int32")
root = tirx.For(i, 0, 8, tirx.ForKind.SERIAL,
                tirx.For(j, 0, 4, tirx.ForKind.SERIAL, tirx.Evaluate(0)))

print("count_loops:", dict(tirx_ext.count_loops(root)))     # loops=2, total_iters=40
out = tirx_ext.break_innermost_for_bodies(root)             # rewrite pass
print("original tree unchanged (COW):", dict(tirx_ext.count_loops(root)))
print("out is a new object:", not out.same_as(root))        # True
EOF
```

Full test suite — run **from the package dir** (the repo-root `conftest.py`
demands a codegen target; the package's `pyproject.toml` re-anchors pytest's
rootdir here). A subshell keeps the relative paths simple:

```sh
( cd $PKG && \
  PYTHONPATH=../../python \
  TIRX_EXT_LIB=build-wheel/cargo/release/libtvm_tirx.so \
    ../../$PY -m pytest tests -q )
# 27 passed
```

---

## 4. Fast Rust iteration (no reinstall)

After editing `rust/src/*.rs`, rebuild with cargo directly:

```sh
PATH="$PWD/$PKG/tools:$PATH" \
CARGO_TARGET_DIR="$PWD/$PKG/build-wheel/cargo" \
TIRX_LIB_DIR="$PWD/build/lib" \
    cargo build --release --manifest-path "$PKG/rust/Cargo.toml"
```

The `tools/` shim provides the `tvm-ffi-config` the build queries. Paths are
absolute (`$PWD/...`) because cargo runs from the manifest's directory.

### ⚠ Stale `.so`

`_find_lib()` order: env `TIRX_EXT_LIB` → a .so **inside the package dir** →
`build-wheel/cargo/{release,debug}/`. An editable install may leave a copy in
the package dir that wins over fresh cargo output, so tests load the old lib.
Fix: pin `TIRX_EXT_LIB=$PKG/build-wheel/cargo/release/libtvm_tirx.so` (as §3
does), or copy the new .so over the package-dir one. Debug build: `release` →
`debug`, drop `--release`.

---

## 5. Uninstall

```sh
uv pip uninstall --python $PY tirx-ext
```

Leaves `build-wheel/` behind; `rm -rf $PKG/build-wheel` clears the build cache.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'tvm'` (at import) | Missing `PYTHONPATH=python` (§2) |
| `cannot locate libtvm_ffi.so` (at install) | tir not built (§0.1), or set `TIRX_LIB_DIR=` / pass `-DTIRX_LIB_DIR=` |
| `No module named 'scikit_build_core'` (at install) | prereq #6: `uv pip install --python $PY scikit-build-core` |
| `cannot locate libtvm_tirx.so` (at import) | Not installed / never cargo-built; point `TIRX_EXT_LIB=` at a built .so |
| Edited Rust, behavior unchanged | Stale .so (§4) — use `TIRX_EXT_LIB=` or overwrite the package-dir copy |
| Field-offset mismatch at import | `_check_layouts` found the Rust mirror drifted from libtvm_compiler.so — sync field defs in `node.rs` |
| pytest asks for a codegen target | Ran without `cd $PKG` — run from the package dir (§3) |

---

## Appendix: external-project shape (not in-repo)

The above is the **in-repo** path (resolving to `build/lib`). To install
tirx-ext as a genuine downstream package elsewhere, two things change:

- `pip install .` (or `-e .`) with an apache-tvm-ffi wheel; CMake's third
  fallback (`tvm-ffi-config --libdir`) then finds the wheel's libtvm_ffi.
- The `tirx.*` types are still registered by `libtvm_compiler.so`, so the host
  process must be able to `import tvm`. This pure-external shape is not verified
  in this repo.
