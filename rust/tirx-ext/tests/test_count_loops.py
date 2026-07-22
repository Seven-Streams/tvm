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
"""tirx_ext tests: IR is built *in Python* with tvm.tirx and handed to the
Rust passes as a live object (borrowed AnyView — no serialization)."""

from __future__ import annotations

import pytest
import tirx_ext
from tvm_ffi.structural import WalkResult, structural_walk

import tvm  # noqa: F401  -- loads libtvm_compiler.so, registering tirx.* types
from tvm import tirx
from tvm.ir import PointerType, PrimType


def make_buffer(name: str, dims: list[int]) -> tirx.Buffer:
    data = tirx.Var(f"{name}_data", PointerType(PrimType("float32"), "global"))
    return tirx.decl_buffer(
        shape=[tirx.IntImm("int32", d) for d in dims],
        dtype="float32",
        name=name,
        data=data,
        elem_offset=tirx.IntImm("int32", 0),
        scope="global",
        data_alignment=64,
        offset_factor=1,
        layout=None,
    )


def build_matmul(m: int, n: int, k: int, guard: bool = False) -> tirx.For:
    """A perfect i/j/k nest computing C[i,j] += A[i,k] * B[k,j]; with
    ``guard=True`` the innermost store is wrapped in ``if k < j`` (same shape
    as the count_loops_v10 native demo tree)."""
    a = make_buffer("A", [m, k])
    b = make_buffer("B", [k, n])
    c = make_buffer("C", [m, n])

    vi = tirx.Var("i", "int32")
    vj = tirx.Var("j", "int32")
    vk = tirx.Var("k", "int32")

    a_ik = tirx.BufferLoad(a, [vi, vk])
    b_kj = tirx.BufferLoad(b, [vk, vj])
    c_ij = tirx.BufferLoad(c, [vi, vj])
    body = tirx.BufferStore(c, tirx.Add(c_ij, tirx.Mul(a_ik, b_kj)), [vi, vj])
    if guard:
        body = tirx.IfThenElse(tirx.LT(vk, vj), body, None)

    zero = tirx.IntImm("int32", 0)
    ext = lambda v: tirx.IntImm("int32", v)  # noqa: E731
    for_k = tirx.For(vk, zero, ext(k), tirx.ForKind.SERIAL, body)
    for_j = tirx.For(vj, zero, ext(n), tirx.ForKind.SERIAL, for_k)
    return tirx.For(vi, zero, ext(m), tirx.ForKind.SERIAL, for_j)


def loop_stats_reference(root) -> tuple[int, int]:
    """Pure-Python reference: (For count, total loop-body executions) via
    tvm_ffi's structural_walk."""
    n_loops = 0
    total_iters = 0
    prod_stack = [1]

    def pre(node: tirx.For):
        nonlocal n_loops, total_iters
        n_loops += 1
        this_total = prod_stack[-1] * int(node.extent)
        total_iters += this_total
        prod_stack.append(this_total)
        structural_walk(node.body, (tirx.For, pre))
        prod_stack.pop()
        return WalkResult.SKIP

    structural_walk(root, (tirx.For, pre))
    return n_loops, total_iters


def test_count_loops_matmul():
    r = tirx_ext.count_loops(build_matmul(64, 64, 64))
    assert r["loops"] == 3
    assert r["total_iters"] == 64 + 64**2 + 64**3
    assert r["ifs"] == 0
    assert r["branch_execs"] == 0
    assert r["innermost"] == 1


def test_count_loops_guarded_matmul():
    r = tirx_ext.count_loops(build_matmul(64, 64, 64, guard=True))
    assert r["loops"] == 3
    assert r["total_iters"] == 64 + 64**2 + 64**3
    assert r["ifs"] == 1
    assert r["branch_execs"] == 64**3  # the predicate is judged on every iteration
    assert r["innermost"] == 1


def test_count_adds():
    r = tirx_ext.count_adds(build_matmul(64, 64, 64))
    assert r["adds"] == 1
    assert r["add_execs"] == 64**3  # executed once per innermost iteration


def test_cross_check_with_python_walk():
    root = build_matmul(3, 5, 7)
    r = tirx_ext.count_loops(root)
    assert (r["loops"], r["total_iters"]) == loop_stats_reference(root)


def test_non_constant_extent_raises():
    n = tirx.Var("n", "int32")
    vi = tirx.Var("i", "int32")
    body = tirx.Evaluate(tirx.IntImm("int32", 0))
    loop = tirx.For(vi, tirx.IntImm("int32", 0), n, tirx.ForKind.SERIAL, body)
    with pytest.raises(Exception, match="constant"):
        tirx_ext.count_loops(loop)
    with pytest.raises(Exception, match="constant"):
        tirx_ext.count_adds(loop)


def _i32(v: int) -> tirx.IntImm:
    return tirx.IntImm("int32", v)


def _add_loop(extent: int, step: int | None = None, body=None) -> tirx.For:
    """One loop over an Evaluate(Add(1, 2)) body (or the given body)."""
    vi = tirx.Var("i", "int32")
    if body is None:
        body = tirx.Evaluate(tirx.Add(_i32(1), _i32(2)))
    return tirx.For(
        vi,
        _i32(0),
        _i32(extent),
        tirx.ForKind.SERIAL,
        body,
        step=None if step is None else _i32(step),
    )


def test_step_trip_count():
    # 0..64 step 2 -> 32 iterations; uneven 0..65 step 2 -> ceildiv = 33.
    for extent, step, trips in [(64, 2, 32), (65, 2, 33), (64, None, 64), (64, 1, 64)]:
        loop = _add_loop(extent, step)
        r = tirx_ext.count_loops(loop)
        assert (r["loops"], r["total_iters"]) == (1, trips), (extent, step)
        a = tirx_ext.count_adds(loop)
        assert (a["adds"], a["add_execs"]) == (1, trips), (extent, step)


@pytest.mark.parametrize("extent", [0, -1, -3])
def test_non_positive_extent_has_zero_trips(extent):
    # A TIR For covers [min, min + extent) with a positive step.  Empty or
    # reversed ranges therefore have zero executions, never a negative count.
    loop = _add_loop(extent, step=2)
    r = tirx_ext.count_loops(loop)
    assert (r["loops"], r["total_iters"]) == (1, 0)

    a = tirx_ext.count_adds(loop)
    assert (a["adds"], a["add_execs"]) == (1, 0)


def test_for_control_adds_use_enclosing_multiplicity():
    # The min and annotation are evaluated/configured outside the repeated
    # body.  Only body_add is charged once per loop trip.
    vi = tirx.Var("i", "int32")
    min_add = tirx.Add(_i32(1), _i32(2))
    annotation_add = tirx.Add(_i32(3), _i32(4))
    body_add = tirx.Add(_i32(5), _i32(6))
    loop = tirx.For(
        vi,
        min_add,
        _i32(4),
        tirx.ForKind.SERIAL,
        tirx.Evaluate(body_add),
        annotations={"test.value": annotation_add},
    )

    a = tirx_ext.count_adds(loop)
    assert a["adds"] == 3
    assert a["add_execs"] == 1 + 1 + 4

    # In a nested loop, the inner min is evaluated once per *outer* trip,
    # while the inner body is evaluated once per outer * inner trip.
    inner_var = tirx.Var("j", "int32")
    inner = tirx.For(
        inner_var,
        tirx.Add(_i32(7), _i32(8)),
        _i32(3),
        tirx.ForKind.SERIAL,
        tirx.Evaluate(tirx.Add(_i32(9), _i32(10))),
    )
    nested = _add_loop(4, body=inner)
    nested_stats = tirx_ext.count_adds(nested)
    assert (nested_stats["adds"], nested_stats["add_execs"]) == (2, 4 + 4 * 3)


def test_step_nested():
    # outer 0..64 step 2 (32 trips) around inner 0..8 (8 trips).
    inner = _add_loop(8)
    outer = _add_loop(64, step=2, body=inner)
    r = tirx_ext.count_loops(outer)
    assert r["loops"] == 2
    assert r["total_iters"] == 32 + 32 * 8


def test_unclaimed_seq_and_sibling_state_restore():
    # SeqStmt itself is not claimed by Counter's generated dispatcher, so the
    # default walker must reach all three For nodes.  The final sibling is not
    # inside the first outer loop: its five trips must not be multiplied by 2.
    nested = _add_loop(2, body=_add_loop(3))
    sibling = _add_loop(5)
    root = tirx.SeqStmt([nested, sibling])
    r = tirx_ext.count_loops(root)
    assert r["loops"] == 3
    assert r["total_iters"] == 2 + 2 * 3 + 5
    assert r["innermost"] == 2

    a = tirx_ext.count_adds(root)
    assert (a["adds"], a["add_execs"]) == (2, 2 * 3 + 5)


def test_generated_dispatch_order_advance_and_state_lifecycle():
    from tirx_ext import _ffi_api as ffi

    # One tree has two For statements and one Evaluate statement.  The
    # concrete-first visitor runs twice, which also verifies that one Visitor
    # retains its state across calls.  Both handlers return Advance, so the
    # nested For and the Evaluate must be reached by default recursion.
    root = _add_loop(2, body=_add_loop(3))
    r = ffi._visit_test_dispatch_order(root)
    assert r["concrete_first_for"] == 4
    assert r["concrete_first_stmt"] == 2

    # With the base Stmt method declared first, it claims For as well; the
    # later concrete method is therefore never called.
    assert r["base_first_for"] == 0
    assert r["base_first_stmt"] == 3


def test_generated_dispatch_visit_children_then_skip():
    from tirx_ext import _ffi_api as ffi

    r = ffi._visit_test_children(_add_loop(2, body=_add_loop(3)))
    assert r["pre"] == 2
    assert r["post"] == 2
    assert r["active"] == 0
    assert r["max_active"] == 2


def test_bad_step_raises():
    vi = tirx.Var("i", "int32")
    non_constant = tirx.For(
        vi,
        _i32(0),
        _i32(64),
        tirx.ForKind.SERIAL,
        tirx.Evaluate(_i32(0)),
        step=tirx.Var("s", "int32"),
    )
    for analyze in (tirx_ext.count_loops, tirx_ext.count_adds):
        with pytest.raises(Exception, match="constant"):
            analyze(non_constant)
        with pytest.raises(Exception, match="positive"):
            analyze(_add_loop(64, step=0))


def test_while_subtree_not_counted():
    # For(4) { While(i < 10) { Evaluate(Add); If(...) } }: nothing under the
    # While — loops, ifs, adds — may contribute to any counter.
    vi = tirx.Var("i", "int32")
    guarded = tirx.IfThenElse(tirx.LT(vi, _i32(3)), tirx.Evaluate(_i32(0)), None)
    w_body = tirx.SeqStmt([tirx.Evaluate(tirx.Add(_i32(1), _i32(2))), guarded])
    w = tirx.While(tirx.LT(vi, _i32(10)), w_body)
    root = _add_loop(4, body=w)
    r = tirx_ext.count_loops(root)
    assert (r["loops"], r["total_iters"]) == (1, 4)
    assert (r["ifs"], r["branch_execs"]) == (0, 0)
    a = tirx_ext.count_adds(root)
    assert (a["adds"], a["add_execs"]) == (0, 0)


def test_overflow_raises():
    # 3_000_000^3 = 2.7e19 > i64::MAX: must error, not silently wrap.
    root = _add_loop(3_000_000, body=_add_loop(3_000_000, body=_add_loop(3_000_000)))
    with pytest.raises(Exception, match="overflow"):
        tirx_ext.count_loops(root)
    with pytest.raises(Exception, match="overflow"):
        tirx_ext.count_adds(root)


def test_analysis_rejects_custom_visit_interrupt():
    import tvm_ffi
    from tvm_ffi.dataclasses import py_class

    @py_class("tirx_ext.testing.InterruptStmt", structural_eq=None)
    class InterruptStmt(tirx.Stmt):
        @staticmethod
        def __s_visit__(_visitor, _value):
            return tvm_ffi.VisitInterrupt("stop before traversal is complete")

    # Put the hook under a loop so this also proves that an explicitly
    # recursive handler does not publish the state accumulated before halt.
    root = _add_loop(4, body=InterruptStmt(None))
    for analyze in (tirx_ext.count_loops, tirx_ext.count_adds):
        with pytest.raises(Exception, match="interrupted"):
            analyze(root)


def test_layout_self_check():
    # Also runs implicitly on every import; here it documents the contract.
    from tirx_ext import _ffi_api

    assert _ffi_api._check_layouts()


_EARLY_CALL_CHILD = """
import ctypes, sys
import tvm_ffi

lib = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
assert lib.tirx_ext_init() == 0
f = tvm_ffi.get_global_func("tirx_ext.count_loops")

# Called before `import tvm`: tirx.* types are unregistered. Must raise a
# Python error naming the type key -- not abort, not a bare "panicked".
try:
    f(1)
    raise SystemExit("expected an error before import tvm")
except Exception as e:  # noqa: BLE001
    assert "tirx.Stmt" in str(e), str(e)

# After the host fixes its load order the same process must recover (the
# type-index caches may not stay poisoned by the earlier failure).
import tvm
from tvm import tirx

vi = tirx.Var("i", "int32")
loop = tirx.For(vi, tirx.IntImm("int32", 0), tirx.IntImm("int32", 8),
                tirx.ForKind.SERIAL, tirx.Evaluate(tirx.IntImm("int32", 0)))
r = f(loop)
assert r["loops"] == 1 and r["total_iters"] == 8, dict(r.items())
print("RECOVERED")
"""


def test_early_call_recovers_after_import_tvm():
    import os
    import subprocess
    import sys

    from tirx_ext import _ffi_api

    proc = subprocess.run(
        [sys.executable, "-c", _EARLY_CALL_CHILD, str(_ffi_api._LIB_PATH)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RECOVERED" in proc.stdout
