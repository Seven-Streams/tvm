# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations.
"""FFI API bindings for tirx_ext.

Load order matters and is enforced here: ``import tvm`` first (it loads
libtvm_compiler.so, registering every ``tirx.*`` type and its libtvm_ffi.so),
*then* dlopen libtvm_tirx.so — the dynamic linker's soname dedup binds the
Rust library to the already-loaded libtvm_ffi.so, so both sides share one
global function registry.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import tvm_ffi

import tvm  # noqa: F401  -- must precede loading libtvm_tirx.so (see module docstring)

_LIB_NAME = "libtvm_tirx.so"


def _find_lib() -> Path:
    """Locate libtvm_tirx.so: env override, package dir(s), then the editable
    install's in-tree cargo artifacts."""
    override = os.environ.get("TIRX_EXT_LIB")
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise OSError(f"TIRX_EXT_LIB={override} does not exist")

    here = Path(__file__).resolve().parent
    candidates = [here / _LIB_NAME]
    pkg = sys.modules.get(__package__)
    for entry in getattr(pkg, "__path__", []):
        candidates.append(Path(entry) / _LIB_NAME)
    # Editable install: scikit-build-core builds under <root>/build-wheel.
    root = here.parent.parent
    for profile in ("release", "debug"):
        candidates.append(root / "build-wheel" / "cargo" / profile / _LIB_NAME)

    for c in candidates:
        if c.is_file():
            return c
    raise OSError(
        f"tirx_ext: cannot locate {_LIB_NAME}; searched: " + ", ".join(str(c) for c in candidates)
    )


_LIB_PATH = _find_lib()
# RTLD_GLOBAL for parity with C++ tvm-ffi extension loading; the registry
# sharing itself comes from soname dedup of the library's libtvm_ffi.so dep.
_LIB = ctypes.CDLL(str(_LIB_PATH), mode=ctypes.RTLD_GLOBAL)
if _LIB.tirx_ext_init() != 0:
    raise RuntimeError(f"tirx_ext: initialization failed in {_LIB_PATH} (detail on stderr)")

# Attach every `tirx_ext.*` global function registered by the Rust library
# (count_loops, count_adds, _check_layouts) as an attribute of this module.
tvm_ffi.init_ffi_api("tirx_ext", __name__)

# Fail fast, at import, if the Rust-mirrored node layouts drifted from the
# loaded libtvm_compiler.so — misreads at count time would be far worse.
_check_layouts()  # noqa: F821  -- injected by init_ffi_api above
