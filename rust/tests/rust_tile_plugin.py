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
# specific language governing permissions and limitations
# under the License.
"""Pytest plugin that routes `tvm.tirx.transform.TileDispatch` and `LowerTIRx` through Rust.

Load it with ``-p rust_tile_plugin`` (this directory on ``PYTHONPATH``) before any test module
imports the factories. ``TVM_RUST_TILE_MODE`` picks what the factories return:

* ``rust``: the Rust pass, so the tests run on Rust alone;
* ``compare``: a pass that runs the native and the Rust pass on the same module, appends the
  outcome to ``TVM_RUST_TILE_LOG`` (JSON lines), and returns the native result, so the tests
  see native behavior while every module they lower is compared.
"""

import ctypes
import json
import os
from pathlib import Path

import tvm_ffi

import tvm

LIBRARY = Path(
    os.environ.get(
        "TVM_RUST_PASSES", Path(__file__).resolve().parent.parent / "target" / "debug" / "libtvm.so"
    )
)
MODE = os.environ.get("TVM_RUST_TILE_MODE", "rust")
LOG = os.environ.get("TVM_RUST_TILE_LOG")

_LIB = ctypes.CDLL(str(LIBRARY))
assert _LIB.tvm_rust_register_passes() == 0, "registering the Rust passes failed"

_current_test = [""]


def _log(name, outcome, detail=""):
    if LOG:
        with open(LOG, "a") as out:
            entry = dict(test=_current_test[0], name=name, outcome=outcome, detail=detail[:2000])
            out.write(json.dumps(entry) + "\n")


def _compared(name, native, rust):
    @tvm.transform.module_pass(opt_level=0, name=f"compare.{name}")
    def compare(mod, _ctx):
        # The Rust pass runs first: it leaves its input alone, while a native pass may update a
        # uniquely owned module in place.
        try:
            actual = rust(mod)
        except Exception as rust_error:
            actual = rust_error
        try:
            expected = native(mod)
        except Exception as native_error:
            outcome = "both_failed" if isinstance(actual, Exception) else "rust_ok_native_error"
            _log(name, outcome, str(native_error))
            raise
        if isinstance(actual, Exception):
            _log(name, "rust_error_native_ok", str(actual))
        elif tvm_ffi.structural_equal(expected, actual, map_free_vars=True):
            _log(name, "equal")
        else:
            mismatch = tvm_ffi.get_first_structural_mismatch(expected, actual, map_free_vars=True)
            _log(name, "mismatch", str(mismatch))
        return expected

    return compare


def _install(name):
    native_factory = getattr(tvm.tirx.transform, name)
    rust_factory = tvm_ffi.get_global_func(f"rust.tirx.transform.{name}")

    def factory():
        if MODE == "rust":
            return rust_factory()
        return _compared(name, native_factory(), rust_factory())

    factory.__name__ = name
    factory.__doc__ = native_factory.__doc__
    setattr(tvm.tirx.transform, name, factory)


for _name in ("TileDispatch", "LowerTIRx"):
    _install(_name)


def pytest_runtest_setup(item):
    _current_test[0] = item.nodeid
