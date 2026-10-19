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
"""Run one side (cpp or rust) of one parity case ``iters`` times on its largest runnable
record, for profiling under callgrind::

    valgrind --tool=callgrind --collect-atstart=no \\
        "--toggle-collect=*FunctionPassObj::operator()*" \\
        --callgrind-out-file=cg.rust.SkipAssert.out \\
        python rust/tests/bench/profile_one.py rust SkipAssert 10

Module passes (VerifyMemory, BindTarget, ...) use ``*ModulePassObj::operator()*`` instead.
The first run on the chosen record is the warm-up; the record is reported on stdout so the
cpp and rust runs can be checked to have used the same input.

usage: profile_one.py <cpp|rust> <case> <iters> [record.json]
"""

import contextlib
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load_parity_module():
    os.environ.setdefault(
        "TVM_RUST_PASSES", str(REPO / "rust" / "target" / "release" / "libtvm.so")
    )
    path = HERE.parent / "test_pass_parity.py"
    spec = importlib.util.spec_from_file_location("test_pass_parity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    side, case, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
    explicit = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    parity = load_parity_module()
    import tvm_ffi

    import tvm

    spec = parity.CASES[case]
    factory = spec.get("factory", case)
    args = spec.get("args", ())
    args = args() if callable(args) else args
    if side == "cpp":
        if "native" in spec:
            pass_ = spec["native"]()
        else:
            pass_ = tvm_ffi.get_global_func(f"tirx.transform.{factory}")(*args)
    else:
        pass_ = tvm_ffi.get_global_func(f"rust.tirx.transform.{factory}")(*args)
    target = spec.get("target")

    @contextlib.contextmanager
    def context(config):
        with contextlib.ExitStack() as stack:
            if target is not None:
                stack.enter_context(tvm.target.Target(target))
            stack.enter_context(tvm.transform.PassContext(config=config))
            yield

    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = sorted(parity._records(spec["sources"]), key=lambda p: -p.stat().st_size)
    for path in candidates:
        text = path.read_text()
        try:
            config = dict(tvm.ir.load_json(text)["config"])
            mod = tvm.ir.load_json(text)["mod"]
            with context(config):
                pass_(mod)
        except Exception as error:  # the parity test skips records that fail on both sides
            print(f"skip {path.name}: {str(error).splitlines()[0][:100]}", file=sys.stderr)
            continue
        break
    else:
        raise SystemExit("no runnable record")

    for _ in range(iters):
        mod = tvm.ir.load_json(text)["mod"]
        with context(config):
            pass_(mod)
    print(f"{side} {case} {path.parent.name}/{path.name} {path.stat().st_size} bytes x{iters}")


if __name__ == "__main__":
    main()
