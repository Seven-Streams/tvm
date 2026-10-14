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
"""Parity of the Rust tirx passes with their C++ counterparts on the recorded corpus.

Build the crate first (``cargo build`` in ``rust/``), record the corpus
(``rust/tests/corpus/generate.sh``), then run this file with pytest from a Python where
``tvm`` is importable. Each case runs ``tirx.transform.<Name>`` and the Rust
``rust.tirx.transform.<Name>`` on every record, inside a PassContext carrying the recorded
config, and requires structurally equal results or a failure on both sides.

Environment: ``TVM_RUST_PASSES`` (the built library), ``TVM_PASS_CORPUS`` (the corpus).
"""

import ctypes
import os
from pathlib import Path

import pytest
import tvm_ffi

import tvm

HERE = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("TVM_PASS_CORPUS", HERE / "corpus" / "data"))
LIBRARY = Path(os.environ.get("TVM_RUST_PASSES", HERE.parent / "target" / "debug" / "libtvm.so"))

_LIB = ctypes.CDLL(str(LIBRARY))
assert _LIB.tvm_rust_register_passes() == 0, "registering the Rust passes failed"


def _is_host_function(func):
    """The host/device split `tvm.tirx.compile` makes with `tirx.transform.Filter`."""
    target = func.attrs.get("target", tvm.target.Target("llvm"))
    return target.kind.name in ["llvm", "c", "cuda_host"]


# Case -> the factory it compares, the corpus directories (pass names) whose records it runs
# on, and the factory arguments.
CASES = {
    # No test runs SkipAssert itself; host modules around MakePackedAPI carry the asserts.
    "SkipAssert": dict(sources=["tirx.MakePackedAPI", "tirx.LowerTVMBuiltin"]),
    "StmtSimplify": dict(sources=["tirx.StmtSimplify", "tirx.RemoveNoOp", "tirx.LowerIntrin"]),
    "RemoveNoOp": dict(sources=["tirx.RemoveNoOp", "tirx.StmtSimplify", "tirx.LowerIntrin"]),
    "ConvertSSA": dict(sources=["tirx.ConvertSSA", "tirx.StmtSimplify"]),
    "DecorateDeviceScope": dict(sources=["tirx.DecorateDeviceScope", "tirx.StmtSimplify"]),
    "AnnotateEntryFunc": dict(
        sources=["tirx.AnnotateEntryFunc", "tirx.SplitHostDevice", "tirx.MakePackedAPI"]
    ),
    # The recorded inputs do not carry the target BindTarget ran with; try three shapes.
    "BindTarget[cuda+llvm]": dict(
        factory="BindTarget",
        sources=["tirx.BindTarget"],
        args=lambda: (tvm.target.Target("cuda", host="llvm"),),
    ),
    "BindTarget[llvm]": dict(
        factory="BindTarget", sources=["tirx.BindTarget"], args=lambda: (tvm.target.Target("llvm"),)
    ),
    "BindTarget[cuda]": dict(
        factory="BindTarget", sources=["tirx.BindTarget"], args=lambda: (tvm.target.Target("cuda"),)
    ),
    "VerifyMemory": dict(sources=["tirx.VerifyMemory", "tirx.SplitHostDevice", "tirx.BindTarget"]),
    "HoistIf": dict(sources=["tirx.HoistIf"]),
    "UnrollLoop": dict(sources=["tirx.UnrollLoop", "tirx.VectorizeLoop"]),
    # Inputs of the passes around LowerIntrin in the pipeline carry the `target` attribute too.
    "LowerIntrin": dict(
        sources=[
            "tirx.LowerIntrin",
            "tirx.MakePackedAPI",
            "tirx.LowerTVMBuiltin",
            "tirx.SplitHostDevice",
        ]
    ),
    # InsertHoistedIf has no factory of its own; it is the first pass of the native HoistIf.
    "InsertHoistedIf": dict(
        sources=["tirx.InsertHoistedIf", "tirx.StmtSimplify"],
        native=lambda: tvm.tirx.transform.HoistIf().passes[0],
    ),
    "Filter[host]": dict(factory="Filter", sources=["tirx.Filter"], args=(_is_host_function,)),
    "Filter[device]": dict(
        factory="Filter", sources=["tirx.Filter"], args=(lambda f: not _is_host_function(f),)
    ),
}


def _records(sources):
    for source in sources:
        yield from sorted((CORPUS / source).glob("*.json"))


def _run(pass_, text):
    """Run `pass_` on a fresh copy of the record: module passes may update their input."""
    record = tvm.ir.load_json(text)
    with tvm.transform.PassContext(config=dict(record["config"])):
        return pass_(record["mod"])


@pytest.mark.parametrize("case", sorted(CASES))
def test_parity(case):
    spec = CASES[case]
    factory = spec.get("factory", case)
    args = spec.get("args", ())
    args = args() if callable(args) else args
    if "native" in spec:
        native = spec["native"]()
    else:
        native = tvm_ffi.get_global_func(f"tirx.transform.{factory}")(*args)
    rust = tvm_ffi.get_global_func(f"rust.tirx.transform.{factory}")(*args)
    compared = both_failed = unloadable = changed = 0
    failures = []
    for path in _records(spec["sources"]):
        text = path.read_text()
        try:
            original = tvm.ir.load_json(text)["mod"]
        except Exception:
            unloadable += 1
            continue
        outcome = []
        for pass_ in (native, rust):
            try:
                outcome.append(("ok", _run(pass_, text)))
            except Exception as error:
                outcome.append(("error", error))
        compared += 1
        (native_kind, expected), (rust_kind, actual) = outcome
        if native_kind == rust_kind == "error":
            both_failed += 1
        elif native_kind != rust_kind:
            detail = expected if native_kind == "error" else actual
            failures.append(f"{path.name}: native {native_kind}, rust {rust_kind}: {detail!s:.300}")
        else:
            changed += not tvm_ffi.structural_equal(original, expected, map_free_vars=True)
            if not tvm_ffi.structural_equal(expected, actual, map_free_vars=True):
                mismatch = tvm_ffi.get_first_structural_mismatch(
                    expected, actual, map_free_vars=True
                )
                failures.append(f"{path.name}: results differ at {mismatch}")
    print(
        f"{case}: {compared} compared ({changed} changed by the pass), {both_failed} failed "
        f"on both sides, {unloadable} unloadable, {len(failures)} disagree"
    )
    assert compared > 0, f"{case}: no loadable records"
    assert not failures, "\n".join(failures[:20])


def _verify_memory_modules():
    """A store to a CUDA tensor argument on the host side, and the same store inside a thread."""
    from tvm.script import tirx as T

    @T.function
    def host_access(A: T.Tensor((16,), "float32")):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "host_access"})
        A[0] = T.float32(1)

    @T.function
    def thread_access(A: T.Tensor((16,), "float32")):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "thread_access"})
        with T.launch_thread("threadIdx.x", 16) as tx:
            A[tx] = T.float32(1)

    return tvm.IRModule({"main": host_access}), tvm.IRModule({"main": thread_access})


def test_verify_memory_rejects_host_access():
    rejected, accepted = _verify_memory_modules()
    for factory in ("tirx.transform.VerifyMemory", "rust.tirx.transform.VerifyMemory"):
        pass_ = tvm_ffi.get_global_func(factory)()
        with pytest.raises(Exception, match="directly accessed by host memory"):
            pass_(rejected)
        pass_(accepted)


def _lower_intrin_module():
    """Every LowerIntrin rewrite: floordiv/floormod of unknown, constrained, and shiftable sign,
    power-of-two divisors, the max/eq patterns, CUDA and LLVM intrinsic rules, and fma fusion."""
    from tvm.script import tirx as T

    @T.function
    def cuda(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "float32"), x: T.int32, y: T.int32):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "cuda"})
        with T.launch_thread("threadIdx.x", 16) as tx:
            A[0] = T.floordiv(x, y)
            A[1] = T.floormod(x, y)
            A[2] = T.floordiv(x, 8)
            A[3] = T.floormod(x, 8)
            A[4] = T.Select(y >= 0, T.floordiv(x, y), 0)
            A[5] = T.floordiv(tx - 4, 5)
            A[6] = T.Select(y >= 0, T.max(T.floordiv(x, y), 0), 0)
            A[7] = T.Select(T.floormod(x, y) == 0, 1, 0)
            A[8] = T.if_then_else(y > 0, T.floordiv(x, y), 0)
            A[9] = T.floormod(tx - 4, 5)
            B[tx] = B[tx] * B[0] + B[1]
            B[1] = T.exp(B[2])
            if y > 0:
                A[10] = T.floordiv(x, y)
            else:
                A[11] = T.floormod(x, y)

    @T.function
    def llvm(
        C: T.Tensor((16,), "float32"),
        D: T.Tensor((16,), "float16"),
        E: T.Tensor((4,), "float32x4"),
        n: T.int32,
    ):
        T.func_attr({"target": T.target("llvm"), "global_symbol": "llvm"})
        for i in range(n):
            C[i] = C[i] * C[0] + C[1]
            C[1] = T.exp(C[2]) + T.floor(C[3])
            E[0] = T.Broadcast(T.Cast(D[0], "float32"), 4) * E[1] + E[2]
            C[2] = T.Cast(T.floordiv(i - 3, 4), "float32")

    return tvm.IRModule({"cuda": cuda, "llvm": llvm})


def test_lower_intrin_branches():
    native = tvm.tirx.transform.LowerIntrin()(_lower_intrin_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.LowerIntrin")()(_lower_intrin_module())
    assert not tvm_ffi.structural_equal(native, _lower_intrin_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def test_call_attrs_are_left_alone():
    """The engine's default recursion follows the native structural hooks, which skip Call.attrs
    like C++ StmtExprMutator: a floordiv in attrs stays while the one in args is lowered."""
    tvm.ir._ffi_api.RegisterOp("test.parity.call_attrs", "")
    op = tvm.ir.Op.get("test.parity.call_attrs")

    def module():
        x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")
        div = tvm.tirx.floordiv(x, y)
        call = tvm.ir.Call(op, [div], attrs={"k": div}, ty=tvm.ir.PrimType("int32"))
        func = tvm.tirx.Function([x, y], tvm.ir.Evaluate(call)).with_attr(
            {"target": tvm.target.Target("cuda"), "global_symbol": "main"}
        )
        return tvm.IRModule({"main": func})

    native = tvm.tirx.transform.LowerIntrin()(module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.LowerIntrin")()(module())
    call = rust["main"].body.seq[0].value
    assert not isinstance(call.args[0], tvm.tirx.FloorDiv)
    assert isinstance(call.attrs["k"], tvm.tirx.FloorDiv)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _remove_no_op_module():
    """Every RemoveNoOp rewrite: empty and provably empty loops, no-op branches on either side,
    pure evaluates, and stores of the value already in place."""
    from tvm.script import tirx as T

    @T.function
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32"), n: T.int32, m: T.int32):
        T.func_attr({"global_symbol": "main"})
        for i in range(n):
            for j in range(i - n):
                A[j] = T.float32(1)
            for k in range(m):
                T.evaluate(0)
            if n > 4:
                A[i] = A[i]
            else:
                B[i] = T.float32(2)
            if m > 2:
                T.evaluate(n + 1)
            else:
                B[0] = A[0] + T.float32(0)
            if n > m:
                B[1] = T.float32(3)
            else:
                T.evaluate(0)
            T.evaluate(n * m)
        for z in range(0):
            A[0] = T.float32(4)

    return tvm.IRModule({"main": main})


def test_remove_no_op_branches():
    native = tvm.tirx.transform.RemoveNoOp()(_remove_no_op_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.RemoveNoOp")()(_remove_no_op_module())
    assert not tvm_ffi.structural_equal(native, _remove_no_op_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _stmt_simplify_module():
    """StmtSimplify facts: tensor shapes are non-negative, loop ranges, pure bindings substituted
    into branch conditions, branch constraints, and a store of the loaded value."""
    from tvm.script import tirx as T

    n = T.dynamic("n")

    @T.function(check_well_formed=False)
    def main(A: T.Tensor((n * 32 + 1 - 2,), "float32"), B: T.Tensor((16,), "float32"), m: T.int32):
        T.func_attr({"global_symbol": "main"})
        A[T.min(T.int64(1), n)] = T.float32(0)
        for i in range(16):
            if i < 16:
                B[i] = B[i] + T.float32(1)
            k = T.bind(i * 2)
            if k < 32:
                B[i] = B[i]
            if m > 4:
                if m > 2:
                    B[i] = T.Select(m > 1, T.float32(2), T.float32(3))
            B[T.floordiv(i, 16)] = T.float32(4)

    return tvm.IRModule({"main": main})


def test_stmt_simplify_facts():
    native = tvm.tirx.transform.StmtSimplify()(_stmt_simplify_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.StmtSimplify")()(_stmt_simplify_module())
    assert not tvm_ffi.structural_equal(native, _stmt_simplify_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _string_binding_module():
    from tvm.script import tirx as T

    @T.function
    def main(A: T.Tensor((4,), "int32")):
        s = T.bind(T.StringImm("hello"))
        T.evaluate(T.call_extern("int32", "printf", s))
        A[0] = 1

    return tvm.IRModule({"main": main})


def test_stmt_simplify_inlines_string_bindings():
    native = tvm.tirx.transform.StmtSimplify()(_string_binding_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.StmtSimplify")()(_string_binding_module())
    assert not tvm_ffi.structural_equal(native, _string_binding_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)
