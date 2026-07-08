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

This follows TVM's **officially recommended conda + LLVM** toolchain (see
`docs/install/from_source.rst`): one conda environment supplies LLVM, CMake, a
matched C/C++ compiler, and Rust, so every artifact — `libtvm_ffi.so`,
`libtvm_compiler.so`, and `libtvm_tirx.so` — is built with one consistent
`libstdc++`.

Paths are relative to the **repo root**; run commands there unless noted.

```sh
PKG=rust/tirx-ext
ENV=tvm-build-venv        # the conda env built in §0.1
```

---

## 0. Prerequisites

tirx-ext has no C++ of its own — install is just scikit-build-core → CMake →
`cargo build` → `libtvm_tirx.so`. That library must link tir's **own**
`libtvm_ffi.so` (one process, one registry — see the note below), so every
prerequisite amounts to "tir's conda dev environment is in place".

| # | Dependency | Check | If missing |
|---|---|---|---|
| 1 | conda | `conda --version` | Install [Miniforge](https://github.com/conda-forge/miniforge) |
| 2 | `tvm-build-venv` env (LLVM, CMake ≥ 3.24, Rust, compiler) | `conda run -n $ENV llvm-config --version` | Create it — §0.1 |
| 3 | tir built (both .so) | `ls build/lib/libtvm_ffi.so build/lib/libtvm_compiler.so` | Build tir — §0.2 |
| 4 | env has `tvm_ffi` | `conda run -n $ENV python -c "import tvm_ffi"` | `conda run -n $ENV pip install -e 3rdparty/tvm-ffi` (§0.2) |
| 5 | `tvm` source tree | `PYTHONPATH=python conda run -n $ENV python -c "import tvm"` | Not installed by design — always run with `PYTHONPATH=python` (§2) |
| 6 | tvm-ffi submodule | `ls 3rdparty/tvm-ffi/rust/tvm-ffi/Cargo.toml` | `git submodule update --init --recursive 3rdparty/tvm-ffi` |

> **One libtvm_ffi, one registry.** `libtvm_compiler.so` (it registers the
> `tirx.*` types) was built against `build/lib/libtvm_ffi.so`; the crate must
> link that **same** .so — otherwise `tirx_ext.*` registers into one registry
> while Python looks in another and the functions vanish. The
> `tools/tvm-ffi-config` shim points the build's `--libdir` query at `build/lib`
> for exactly this reason.

### 0.1 Create the conda environment

The env carries the whole build toolchain, mirroring the official TVM guide and
adding Rust for the crate. `cxx-compiler` pins a conda-forge gcc/g++ so tir and
the LLVM static libs share one `libstdc++`. `zlib` and `zstd` are the **dev**
packages (headers + unversioned `.so`) that static LLVM's `find_package(ZLIB)`
and link step need — `llvmdev` only pulls the runtime `libzlib`, so without
`zlib` here CMake fails with `Could NOT find ZLIB (missing: ZLIB_LIBRARY)`.

```sh
conda create -n tvm-build-venv -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    git \
    python=3.11 \
    ninja \
    rust \
    cxx-compiler \
    zlib \
    zstd
conda activate tvm-build-venv
# Python build/runtime deps (tvm-ffi + tvm + scikit-build-core stack)
pip install scikit-build-core cython setuptools-scm numpy ml_dtypes typing_extensions pytest
```

> **libxml2 link fix (one-time).** Static LLVM's `llvm-config --system-libs`
> emits `-lxml2`, but conda-forge `libxml2` ≥ 2.15 ships only the versioned
> `libxml2.so.N` — the unversioned link name is missing, so the link of
> `libtvm_compiler.so` fails with `library not found: xml2`. With the env
> **activated** (so `$CONDA_PREFIX` points at it, not the system `/lib`), create
> the link once, targeting whatever soname is actually installed:
>
> ```sh
> ( cd "$CONDA_PREFIX/lib" && ln -sf "$(ls libxml2.so.* | sort -V | tail -1)" libxml2.so )
> ```

All build commands below are shown as `conda run -n $ENV …` so they work from
any shell; equivalently, `conda activate tvm-build-venv` once and drop the
prefix.

### 0.2 Build tir (with LLVM)

The only heavy step. From the repo root, seed `config.cmake` and append the
official LLVM flags, then build — it produces both `.so` under `build/lib/`:

```sh
rm -rf build && mkdir build && cp cmake/config.cmake build/config.cmake
{
  echo 'set(CMAKE_BUILD_TYPE RelWithDebInfo)'
  # LLVM is a must dependency for the compiler end (official recommendation)
  echo 'set(USE_LLVM "llvm-config --ignore-libllvm --link-static")'
  echo 'set(HIDE_PRIVATE_SYMBOLS ON)'
} >> build/config.cmake

conda run -n $ENV cmake -S . -B build -G Ninja
conda run -n $ENV cmake --build build -j"$(nproc)"

# tir's Python FFI runtime (provides `import tvm_ffi`)
conda run -n $ENV pip install -e 3rdparty/tvm-ffi
```

`tvm` itself is never installed — it's imported from `python/` via
`PYTHONPATH=python` at runtime (§2).

---

## 1. Install (editable)

```sh
conda run -n $ENV pip install -e $PKG --no-build-isolation
```

- `-e`: Python edits take effect immediately; the build writes `libtvm_tirx.so`
  to `$PKG/build-wheel/cargo/{release,debug}/` and copies it into the installed
  package dir.
- `--no-build-isolation`: reuse the env's scikit-build-core (§0.1) and let the
  build see the tir environment.
- Release profile by default (`pyproject.toml`: `cmake.build-type = "Release"`).

`libtvm_ffi.so` is resolved in order: `-DTIRX_LIB_DIR=` / env `TIRX_LIB_DIR` →
`build/lib` (in-repo default) → `tvm-ffi-config --libdir` (external wheel).
In-repo needs no configuration. Success ends with
`Successfully installed tirx-ext-0.1.0`.

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

`conda run` does not forward a heredoc on stdin, so activate the env for the
inline snippet (or drop it into a `.py` file and `conda run … python file.py`):

```sh
conda activate tvm-build-venv
PYTHONPATH=python python - <<'EOF'
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
rootdir here). A subshell keeps the relative paths simple (no stdin, so
`conda run` is fine here):

```sh
( cd $PKG && \
  PYTHONPATH=../../python \
    conda run -n tvm-build-venv python -m pytest tests -q )
# 27 passed
```

---

## 4. Fast Rust iteration (no reinstall)

After editing `rust/src/*.rs`, rebuild with cargo directly. Wrap it in
`conda run … bash -c '…'` so `$PATH` — and thus `cargo` — is the env's; a bare
`env PATH="…:$PATH"` would splice in the *outer* shell's PATH and drop the env's
cargo (breaking a conda-only shell):

```sh
conda run -n $ENV bash -c '
  PATH="$PWD/rust/tirx-ext/tools:$PATH" \
  CARGO_TARGET_DIR="$PWD/rust/tirx-ext/build-wheel/cargo" \
  TIRX_LIB_DIR="$PWD/build/lib" \
  cargo build --release --manifest-path rust/tirx-ext/rust/Cargo.toml
'
```

Run it from the repo root: the `tools/` shim provides the `tvm-ffi-config` the
build queries, and the `$PWD/...` paths are absolute because cargo runs from the
manifest's directory.

### ⚠ Stale `.so`

`_find_lib()` order: env `TIRX_EXT_LIB` → a .so **inside the package dir** →
`build-wheel/cargo/{release,debug}/`. An editable install leaves a copy in the
installed package dir that wins over fresh cargo output, so tests load the old
lib. Fix: pin `TIRX_EXT_LIB=$PWD/$PKG/build-wheel/cargo/release/libtvm_tirx.so`,
copy the new .so over the package-dir one, or just re-run the editable install
(§1) — it rebuilds via cargo and refreshes the installed copy. Debug build:
`release` → `debug`, drop `--release`.

---

## 5. Uninstall

```sh
conda run -n $ENV pip uninstall -y tirx-ext
```

`-y` is required: `conda run` gives the child no stdin, so pip's interactive
`Proceed (Y/n)?` prompt would otherwise abort on EOF.

Leaves `build-wheel/` behind; `rm -rf $PKG/build-wheel` clears the build cache.
To discard the whole toolchain: `conda env remove -n tvm-build-venv`.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'tvm'` (at import) | Missing `PYTHONPATH=python` (§2) |
| `cannot locate libtvm_ffi.so` (at install) | tir not built (§0.2), or set `TIRX_LIB_DIR=` / pass `-DTIRX_LIB_DIR=` |
| `cannot locate libtvm_tirx.so` (at import) | Not installed / never cargo-built; point `TIRX_EXT_LIB=` at a built .so |
| Edited Rust, behavior unchanged | Stale .so (§4) — use `TIRX_EXT_LIB=` or reinstall |
| `node layout mismatch … rebuild the extension` at import | `_check_layouts` found the Rust mirror drifted from libtvm_compiler.so — sync the field defs in `rust/src/node.rs`/`layout.rs` |
| pytest asks for a codegen target | Ran without `cd $PKG` — run from the package dir (§3) |
| `llvm-config` not found at configure | Not inside the env — prefix with `conda run -n $ENV` or `conda activate` first (§0.1) |

---

## Appendix: external-project shape (not in-repo)

The above is the **in-repo** path (resolving to `build/lib`). To install
tirx-ext as a genuine downstream package elsewhere, two things change:

- `pip install .` (or `-e .`) with an apache-tvm-ffi wheel; CMake's third
  fallback (`tvm-ffi-config --libdir`) then finds the wheel's libtvm_ffi.
- The `tirx.*` types are still registered by `libtvm_compiler.so`, so the host
  process must be able to `import tvm`. This pure-external shape is not verified
  in this repo.
