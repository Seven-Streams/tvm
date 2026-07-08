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
"""Mutator tests: the break_for_bodies demo pass (docs/mutator-proposal.md).

COW contract throughout: the original tree is never modified; unchanged
subtrees are shared pointer-identically into the result.
"""

from __future__ import annotations

import tirx_ext
from test_count_loops import _add_loop, _i32, build_matmul

import tvm  # noqa: F401
from tvm import tirx
from tvm.tirx.stmt import Break


def test_break_matmul():
    root = build_matmul(64, 64, 64)
    out = tirx_ext.break_for_bodies(root)
    # The outer loop's body became Break; the inner loops vanished with it.
    assert isinstance(out, tirx.For)
    assert isinstance(out.body, Break)
    r = tirx_ext.count_loops(out)
    assert (r["loops"], r["total_iters"]) == (1, 64)
    # Path-external subtrees are shared, not copied.
    assert not out.same_as(root)
    assert out.min.same_as(root.min)
    assert out.extent.same_as(root.extent)
    assert out.loop_var.same_as(root.loop_var)
    # The original tree is untouched.
    r0 = tirx_ext.count_loops(root)
    assert r0["total_iters"] == 64 + 64**2 + 64**3
    assert isinstance(root.body, tirx.For)


def test_break_cow_pointer_identity_without_for():
    ev = tirx.Evaluate(_i32(0))
    out = tirx_ext.break_for_bodies(ev)
    assert out.same_as(ev)  # nothing claimed, nothing changed -> same object


def test_break_seq_stmt_via_generic_array_path():
    # SeqStmt is unclaimed by the table: the engine's generic rebuild must
    # rebuild its Array<Stmt> field element-wise.
    f1, f2 = _add_loop(4), _add_loop(8)
    seq = tirx.SeqStmt([f1, f2])
    out = tirx_ext.break_for_bodies(seq)
    assert isinstance(out, tirx.SeqStmt) and not out.same_as(seq)
    assert isinstance(out.seq[0].body, Break)
    assert isinstance(out.seq[1].body, Break)
    # Each rewritten For gets its own fresh Break node — no sharing.
    assert not out.seq[0].body.same_as(out.seq[1].body)
    assert out.seq[0].extent.same_as(f1.extent)
    r = tirx_ext.count_loops(out)
    assert (r["loops"], r["total_iters"]) == (2, 4 + 8)
    # Originals untouched.
    assert isinstance(f1.body, tirx.Evaluate)


def test_break_under_if_via_generic_object_path():
    # IfThenElse is unclaimed: generic rebuild recurses its Stmt fields.
    vc = tirx.Var("c", "int32")
    ite = tirx.IfThenElse(tirx.LT(vc, _i32(3)), _add_loop(4), None)
    out = tirx_ext.break_for_bodies(ite)
    assert isinstance(out, tirx.IfThenElse) and not out.same_as(ite)
    assert isinstance(out.then_case.body, Break)
    assert out.condition.same_as(ite.condition)


def test_break_under_while_like_cxx_stmt_mutator():
    # C++ StmtMutator recurses While bodies (stmt_functor.cc:334); so do we.
    vc = tirx.Var("c", "int32")
    w = tirx.While(tirx.LT(vc, _i32(3)), _add_loop(4))
    out = tirx_ext.break_for_bodies(w)
    assert isinstance(out, tirx.While) and not out.same_as(w)
    assert isinstance(out.body.body, Break)
    assert out.condition.same_as(w.condition)


def test_unsupported_stmt_rebuild_raises():
    # AttrStmt is outside the supported control-flow skeleton (route B,
    # docs/mutator-review.md P1): when its children change, the engine must
    # error loudly instead of silently approximating C++ StmtMutator.
    import pytest

    vi = tirx.Var("i", "int32")
    attr = tirx.AttrStmt(vi, "pragma_test", _i32(1), _add_loop(4))
    with pytest.raises(Exception, match="skeleton"):
        tirx_ext.break_for_bodies(attr)


def test_unsupported_stmt_unchanged_passes_through():
    # An untouched subtree of ANY type is safe: no rebuild, no divergence.
    vi = tirx.Var("i", "int32")
    attr = tirx.AttrStmt(vi, "pragma_test", _i32(1), tirx.Evaluate(_i32(0)))
    out = tirx_ext.break_for_bodies(attr)
    assert out.same_as(attr)


# ---------------------------------------------------------------------------
# break_innermost_for_bodies: the handler-re-entrancy demo — the handler
# drives its own child through the engine (mapper.map) and decides from the
# result, which the pre-redesign API could not express
# (docs/mutator-redesign.md §5).
# ---------------------------------------------------------------------------


def test_break_innermost_matmul():
    root = build_matmul(64, 64, 64)
    out = tirx_ext.break_innermost_for_bodies(root)
    # Only the innermost (k) loop's body became Break; i/j structure is kept.
    assert isinstance(out, tirx.For)
    assert isinstance(out.body, tirx.For)
    assert isinstance(out.body.body, tirx.For)
    assert isinstance(out.body.body.body, Break)
    r = tirx_ext.count_loops(out)
    assert (r["loops"], r["total_iters"], r["innermost"]) == (3, 64 + 64**2 + 64**3, 1)
    # Path-external sharing + original untouched.
    assert out.min.same_as(root.min)
    assert out.body.extent.same_as(root.body.extent)
    assert not isinstance(root.body.body.body, Break)


def test_break_innermost_single_loop():
    f = _add_loop(4)
    out = tirx_ext.break_innermost_for_bodies(f)
    assert isinstance(out.body, Break)
    assert out.extent.same_as(f.extent)


def test_break_innermost_through_unclaimed_while():
    # An unclaimed While between two Fors: the deeper For's rewrite must
    # propagate up through the While's default rebuild, so the outer For is
    # NOT treated as innermost.
    vc = tirx.Var("c", "int32")
    w = tirx.While(tirx.LT(vc, _i32(3)), _add_loop(8))
    outer = _add_loop(4, body=w)
    out = tirx_ext.break_innermost_for_bodies(outer)
    assert isinstance(out.body, tirx.While)  # outer loop kept its While body
    assert isinstance(out.body.body.body, Break)  # the inner loop broke
    assert out.body.condition.same_as(w.condition)


# ---------------------------------------------------------------------------
# Engine dispatch order + handler idioms, pinned through the underscore test
# globals of lib.rs (pass table > TYPE_HOOKS > default rebuild; map_fields;
# the single Result error channel).
# ---------------------------------------------------------------------------


def _while_tree():
    vc = tirx.Var("c", "int32")
    return tirx.While(tirx.LT(vc, _i32(3)), tirx.Evaluate(_i32(0)))


def test_type_hook_fires_when_table_misses():
    from tirx_ext import _ffi_api as ffi

    out = ffi._map_test_hook_dispatch(_while_tree())
    assert isinstance(out, Break)  # the While hook replaced the root
    # ... and a hook replacement participates in the default rebuild above it.
    loop = _add_loop(4, body=_while_tree())
    out2 = ffi._map_test_hook_dispatch(loop)
    assert isinstance(out2, tirx.For) and isinstance(out2.body, Break)
    assert out2.extent.same_as(loop.extent)


def test_pass_table_outranks_type_hook():
    from tirx_ext import _ffi_api as ffi

    w = _while_tree()
    out = ffi._map_test_table_wins(w)
    assert out.same_as(w)  # the identity handler won; the hook never fired


def test_map_fields_idiom():
    from tirx_ext import _ffi_api as ffi

    loop = _add_loop(4, body=_while_tree())
    out = ffi._map_test_map_fields(loop)
    # The For handler deferred to the default rebuild of its own fields; the
    # While child re-entered the table and broke.
    assert isinstance(out, tirx.For) and isinstance(out.body, Break)
    assert out.min.same_as(loop.min)
    assert out.extent.same_as(loop.extent)
    assert not out.same_as(loop)
    # COW through map_fields: no changed children -> pointer-identical result.
    plain = _add_loop(4)
    assert ffi._map_test_map_fields(plain).same_as(plain)


def test_handler_error_single_channel():
    import pytest
    from tirx_ext import _ffi_api as ffi

    with pytest.raises(Exception, match="test handler failure"):
        ffi._map_test_handler_error(_add_loop(4))


def test_break_annotations_map_not_descended():
    # A For stashed inside annotations must NOT be rewritten (Map values are
    # metadata — C++ StmtVisitor/StmtMutator never descend them).
    stowaway = _add_loop(2)
    vi = tirx.Var("i", "int32")
    loop = tirx.For(
        vi,
        _i32(0),
        _i32(4),
        tirx.ForKind.SERIAL,
        tirx.Evaluate(_i32(0)),
        annotations={"x": stowaway},
    )
    out = tirx_ext.break_for_bodies(loop)
    assert isinstance(out.body, Break)
    assert isinstance(out.annotations["x"], tirx.For)
    assert isinstance(out.annotations["x"].body, tirx.Evaluate)  # untouched
