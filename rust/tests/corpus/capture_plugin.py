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
"""pytest plugin: record the input module of every tirx pass a test runs.

Each record is ``<CORPUS_DIR>/<pass name>/<source>-<hash>.json``: the module and the
active PassContext config, serialized with ``tvm.ir.save_json``. ``source`` is the
test file, which separates the two passes that share the name ``tirx.NarrowDataType``.

Environment: ``CORPUS_DIR`` (required), ``CORPUS_LIMIT`` (records per pass and source).
"""

import os
import time
from pathlib import Path

import tvm_ffi

import tvm
from tvm.ir import transform as T

CORPUS = Path(os.environ["CORPUS_DIR"])
LIMIT = int(os.environ.get("CORPUS_LIMIT", "12"))
STATE = {"source": "unknown", "seen": set(), "count": {}, "skipped": {}}
MAX_BYTES = 2_000_000


def _record(name, mod):
    if not name.startswith("tirx."):
        return
    key = (name, STATE["source"])
    if STATE["count"].get(key, 0) >= LIMIT:
        return
    try:
        digest = tvm_ffi.structural_hash(mod)
        config = T.PassContext.current().config
        if (name, digest) in STATE["seen"]:
            return
        text = tvm.ir.save_json({"mod": mod, "config": dict(config)})
        if len(text) > MAX_BYTES:
            raise ValueError(f"{len(text)} bytes")
        tvm.ir.load_json(text)  # only keep records that round-trip
    except Exception as error:
        STATE["skipped"][name] = STATE["skipped"].get(name, 0) + 1
        STATE.setdefault("errors", {}).setdefault(name, f"{type(error).__name__}: {error}"[:200])
        return
    STATE["seen"].add((name, digest))
    STATE["count"][key] = STATE["count"].get(key, 0) + 1
    out = CORPUS / name / f"{STATE['source']}-{digest & 0xFFFFFFFF:08x}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


@T.pass_instrument
class _Capture:
    def run_before_pass(self, mod, info):
        _record(info.name, mod)


# Every PassContext carries the capture instrument, including the ones tests create.
_init = T.PassContext.__init__


def _init_with_capture(self, *args, **kwargs):
    instruments = list(kwargs.pop("instruments", None) or [])
    instruments.append(_Capture())
    _init(self, *args, instruments=instruments, **kwargs)


T.PassContext.__init__ = _init_with_capture
_GLOBAL = T.PassContext()
_GLOBAL.__enter__()

# Passes applied directly from Python (`pass_(mod)`) go through PassContext too, but the
# instrument only sees passes that the C++ PassNode::operator() dispatches; that covers both.


def pytest_runtest_setup(item):
    STATE["source"] = Path(str(item.fspath)).stem


def pytest_sessionfinish(session):
    counts = {}
    for (name, _source), count in STATE["count"].items():
        counts[name] = counts.get(name, 0) + count
    CORPUS.mkdir(parents=True, exist_ok=True)
    summary = CORPUS / f"summary-{int(time.time())}-{os.getpid()}.txt"
    lines = [f"{name} {count}" for name, count in sorted(counts.items())]
    lines += [
        f"SKIPPED {name} {n} {STATE['errors'].get(name, '')}"
        for name, n in sorted(STATE["skipped"].items())
    ]
    summary.write_text("\n".join(lines) + "\n")
