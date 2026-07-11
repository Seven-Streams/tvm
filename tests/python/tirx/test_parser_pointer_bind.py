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
"""Pointer-typed bindings in the script parser and the legacy dtype shims."""

import pytest
import tvm_ffi

import tvm
import tvm.script
import tvm.testing
from tvm.ir import PointerType, PrimType
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.op import handle_add_byte_offset


def from_source(code):
    return tvm.script.from_source(code)


def test_pointer_plain_assign_binds_immutably():
    """A plain assign of a pointer-typed expression becomes a Bind var."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        smem = T.alloc_shared([128], "float16")
        p = handle_add_byte_offset(smem.data, 8)
        T.evaluate(p)
        # fmt: on

    binds = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body, lambda s: binds.append(s) if isinstance(s, tvm.tirx.Bind) else None
    )
    ptr_binds = [b for b in binds if isinstance(b.var.ty, PointerType)]
    assert ptr_binds, "expected a pointer-typed Bind"
    assert any(b.var.name == "p" for b in ptr_binds)


def test_pointer_reassign_is_rejected():
    """Rebinding a pointer name must error, not silently shadow (Bind vars
    are immutable and frame-scoped; a rebind inside for/if would be dropped
    on block exit while the same code with an int updates correctly)."""

    src_loop = """
from tvm.script import tirx as T
from tvm.tirx.op import handle_add_byte_offset

@T.prim_func
def func():
    T.device_entry()
    smem = T.alloc_shared([128], "float16")
    p = handle_add_byte_offset(smem.data, 8)
    for i in T.serial(4):
        p = handle_add_byte_offset(smem.data, 16)
    T.evaluate(p)
"""
    with pytest.raises(tvm.error.DiagnosticError):
        from_source(src_loop)

    src_same_scope = """
from tvm.script import tirx as T
from tvm.tirx.op import handle_add_byte_offset

@T.prim_func
def func():
    T.device_entry()
    smem = T.alloc_shared([128], "float16")
    p = handle_add_byte_offset(smem.data, 8)
    p = handle_add_byte_offset(smem.data, 16)
    T.evaluate(p)
"""
    with pytest.raises(tvm.error.DiagnosticError):
        from_source(src_same_scope)


def test_ann_assign_void_ptr_same_scope_coerces():
    """T.let binding a void* value to a typed pointer var of the same
    storage scope inserts the reinterpret coercion."""

    # fmt: off
    @T.prim_func
    def func(a: T.handle):
        A = T.match_buffer(a, (16,), "uint64")
        T.device_entry()
        p: T.let[T.Var(name="p", dtype=PointerType(PrimType("uint64"), "global"))] = (
            handle_add_byte_offset(A.data, 8)
        )
        T.evaluate(p)
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code


def test_ann_assign_cross_pointee_is_rejected():
    """A typed-pointer value bound to a differently-typed pointer var must
    keep failing loudly instead of being silently reinterpreted."""

    src = """
from tvm.script import tirx as T
from tvm.ir import PointerType, PrimType

@T.prim_func
def func(a: T.handle):
    A = T.match_buffer(a, (16,), "float16")
    T.device_entry()
    p: T.let[T.Var(name="p", dtype=PointerType(PrimType("uint64"), "global"))] = A.data
    T.evaluate(p)
"""
    with pytest.raises(tvm.error.DiagnosticError):
        from_source(src)


def test_ann_assign_cross_scope_void_ptr_is_rejected():
    """A void* value whose storage scope differs from the annotated var's
    scope must not be silently coerced."""

    src = """
from tvm.script import tirx as T
from tvm.ir import PointerType, PrimType
from tvm.tirx.op import handle_add_byte_offset

@T.prim_func
def func(a: T.handle):
    A = T.match_buffer(a, (16,), "uint64")
    T.device_entry()
    p: T.let[T.Var(name="p", dtype=PointerType(PrimType("uint64"), "shared"))] = (
        handle_add_byte_offset(A.data, 8)
    )
    T.evaluate(p)
"""
    with pytest.raises(tvm.error.DiagnosticError):
        from_source(src)


def test_legacy_dtype_shims():
    """Legacy accessors: Expr.dtype, StringImm.dtype, PrimType.bits/.lanes,
    and the PrimExpr alias."""

    # PrimType-valued expression: dtype forwards to ty.dtype
    x = tvm.tirx.const(1, "int32")
    assert str(x.dtype) == "int32"

    # Pointer-typed expression: no scalar dtype, keep AttributeError so
    # hasattr()/getattr() duck-typing guards behave as before the shim.
    v = tvm.tirx.Var("p", PointerType(PrimType("float16")))
    with pytest.raises(AttributeError):
        _ = v.dtype
    assert not hasattr(v, "dtype")

    # StringImm: legacy dtype was "handle", not void
    assert tvm.tirx.StringImm("x").dtype == tvm_ffi.dtype("handle")

    # PrimType legacy accessors
    pt = PrimType("float16x2")
    assert pt.bits == 16
    assert pt.lanes == 2

    # PrimExpr alias: importable from tvm.ir and re-exported by tvm.tirx
    assert tvm.ir.PrimExpr is tvm.ir.Expr
    assert tvm.tirx.PrimExpr is tvm.ir.PrimExpr


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_mbarrier_remote_view_compiles():
    """End-to-end compile of a cluster kernel arriving through
    MBarrier.remote_view — the only consumer of the uint64* reinterpret
    bind and of the explicit pointer cast in C codegen."""
    from tvm.tirx.lang.pipeline import MBarrier

    CLUSTER_N = 2

    # fmt: off
    @T.prim_func
    def remote_arrive(a: T.handle) -> None:
        A = T.match_buffer(a, (1,), "float32")
        T.device_entry()
        cbx = T.cta_id_in_cluster([CLUSTER_N])
        T.cta_id([CLUSTER_N])
        tid = T.thread_id([1])
        pool = T.SMEMPool()
        mbar = MBarrier(pool, 1)
        pool.commit()

        mbar.init(1)
        T.ptx.fence.mbarrier_init()
        T.cuda.cluster_sync()

        if tid == 0:
            remote = mbar.remote_view(1 - cbx)
            remote.arrive(0)
            mbar.wait(0, 0)
            if cbx == 0:
                A[0] = T.float32(1)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": remote_arrive})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    cuda_src = mod.mod.imports[0].inspect_source()
    assert "mapa.u64" in cuda_src
    # The remote pointer bind must carry an explicit cast: the mapped value
    # prints as a plain address expression while the var is uint64-typed.
    assert "uint64_t* remote_mbar_ptr = (uint64_t*)" in cuda_src


if __name__ == "__main__":
    tvm.testing.main()
