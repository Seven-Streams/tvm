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

import ast
import contextlib
import ctypes
import importlib.util
import itertools
import os
import sys
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


REPO = HERE.parent.parent

# Upstream test suites run again with the C++ pass replaced by a run of both passes
# (`test_upstream_suite_parity`), by the factory they exercise.
SUITES = {
    "StorageRewrite": "tests/python/tirx/transform/test_tirx_transform_storage_rewrite.py",
    "PointerValueTypeRewrite": (
        "tests/python/tirx/transform/test_tirx_transform_pointer_value_type_rewrite.py"
    ),
}
_SUITE_MODULES = {}


def _suite(factory):
    """The upstream test module of `factory`, imported once."""
    if factory not in _SUITE_MODULES:
        path = REPO / SUITES[factory]
        spec = importlib.util.spec_from_file_location(f"_parity_suite_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        # TVMScript finds the source of a parsed class through `sys.modules`.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SUITE_MODULES[factory] = module
    return _SUITE_MODULES[factory]


# Importing the StorageRewrite suite registers the memory scopes `local.L0A` and `local.L0A2`
# (single allocation), which part of the recorded StorageRewrite corpus uses.
_suite("StorageRewrite")


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
    # The inputs of other passes give CSE many more expressions to share.
    "CommonSubexprElim": dict(
        sources=[
            "tirx.CommonSubexprElim",
            "tirx.NarrowDataType",
            "tirx.StmtSimplify",
            "tirx.FlattenTensor",
            "tirx.UnrollLoop",
            "tirx.VectorizeLoop",
            "tirx.LowerWarpMemory",
        ]
    ),
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
    "LowerTVMBuiltin": dict(sources=["tirx.LowerTVMBuiltin", "tirx.MakePackedAPI"]),
    # InsertHoistedIf has no factory of its own; it is the first pass of the native HoistIf.
    "InsertHoistedIf": dict(
        sources=["tirx.InsertHoistedIf", "tirx.StmtSimplify"],
        native=lambda: tvm.tirx.transform.HoistIf().passes[0],
    ),
    # The native pass rewrites none of its recorded inputs; see test_lower_thread_allreduce_paths.
    "LowerThreadAllreduce": dict(sources=["tirx.LowerThreadAllreduce", "tirx.LowerWarpMemory"]),
    "Filter[host]": dict(factory="Filter", sources=["tirx.Filter"], args=(_is_host_function,)),
    "Filter[device]": dict(
        factory="Filter", sources=["tirx.Filter"], args=(lambda f: not _is_host_function(f),)
    ),
    # The legalizers leave functions alone on a target with native bfloat16 / float8 support, so
    # the records of every pass around them in the host pipeline are inputs too.
    "BF16ComputeLegalize": dict(
        sources=["tirx.BF16ComputeLegalize", "tirx.BF16StorageLegalize", "tirx.FP8ComputeLegalize"]
    ),
    "BF16StorageLegalize": dict(
        sources=["tirx.BF16StorageLegalize", "tirx.BF16ComputeLegalize", "tirx.FP8StorageLegalize"]
    ),
    "FP8ComputeLegalize[float16]": dict(
        factory="FP8ComputeLegalize",
        sources=["tirx.FP8ComputeLegalize", "tirx.BF16ComputeLegalize"],
        args=("float16",),
    ),
    "FP8ComputeLegalize[float32]": dict(
        factory="FP8ComputeLegalize",
        sources=["tirx.FP8ComputeLegalize", "tirx.BF16ComputeLegalize"],
        args=("float32",),
    ),
    "FP8StorageLegalize": dict(
        sources=["tirx.FP8StorageLegalize", "tirx.FP8ComputeLegalize", "tirx.BF16StorageLegalize"]
    ),
    "MakePackedAPI": dict(sources=["tirx.MakePackedAPI"]),
    "SplitHostDevice": dict(
        sources=["tirx.SplitHostDevice", "tirx.BindTarget", "tirx.AnnotateEntryFunc", "tirx.Filter"]
    ),
    "FlattenTensor": dict(sources=["tirx.FlattenTensor"]),
    # The pass needs a current target when a function has none; any target gives the same result.
    "LowerTIRxCleanup": dict(sources=["tirx.LowerTIRxCleanup"], target="cuda"),
    "VectorizeLoop[on]": dict(
        factory="VectorizeLoop", sources=["tirx.VectorizeLoop"], args=(True,)
    ),
    "VectorizeLoop[off]": dict(
        factory="VectorizeLoop", sources=["tirx.VectorizeLoop"], args=(False,)
    ),
    # Few inputs come to RenormalizeSplitPattern itself; the split patterns appear in the inputs of
    # many passes after it.
    "RenormalizeSplitPattern": dict(
        sources=[
            "tirx.RenormalizeSplitPattern",
            "tirx.StmtSimplify",
            "tirx.LowerTVMBuiltin",
            "tirx.LowerIntrin",
            "tirx.LowerTIRx",
            "tirx.LowerTIRxCleanup",
            "tirx.TileDispatch",
            "tirx.VectorizeLoop",
            "tirx.NarrowDataType",
            "tirx.MakePackedAPI",
            "tirx.LowerWarpMemory",
        ]
    ),
    # No recorded module allocates warp memory: the device functions around LowerWarpMemory in the
    # pipeline carry a target and stay unchanged, the others fail on both sides without one.
    "LowerWarpMemory": dict(
        sources=[
            "tirx.LowerWarpMemory",
            "tirx.LowerThreadAllreduce",
            "tirx.LowerIntrin",
            "tirx.UnrollLoop",
            "tirx.StmtSimplify",
        ]
    ),
    "NarrowDataType": dict(
        sources=[
            "tirx.NarrowDataType",
            "tirx.FlattenTensor",
            "tirx.VectorizeLoop",
            "tirx.UnrollLoop",
            "tirx.LowerThreadAllreduce",
            "tirx.LowerWarpMemory",
        ],
        args=(32,),
    ),
    # A narrower target narrows the int32 indices, shapes and thread extents of the corpus too.
    "NarrowDataType[16]": dict(
        factory="NarrowDataType",
        sources=[
            "tirx.NarrowDataType",
            "tirx.BF16ComputeLegalize",
            "tirx.FlattenTensor",
            "tirx.VectorizeLoop",
            "tirx.LowerThreadAllreduce",
        ],
        args=(16,),
    ),
    # ForceNarrowIndexToInt32 is not in the default pipeline; it runs on NarrowDataType's inputs
    # and on the lowered modules around it.
    "ForceNarrowIndexToInt32": dict(
        sources=[
            "tirx.NarrowDataType",
            "tirx.FlattenTensor",
            "tirx.LowerTIRxCleanup",
            "tirx.MakePackedAPI",
            "tirx.VectorizeLoop",
        ]
    ),
    # StorageRewrite runs after VectorizeLoop; the passes after it see its results.
    "StorageRewrite": dict(
        sources=[
            "tirx.StorageRewrite",
            "tirx.VectorizeLoop",
            "tirx.HoistIf",
            "tirx.UnrollLoop",
            "tirx.FlattenTensor",
        ]
    ),
    "PointerValueTypeRewrite": dict(
        sources=[
            "tirx.PointerValueTypeRewrite",
            "tirx.StorageRewrite",
            "tirx.HoistIf",
            "tirx.UnrollLoop",
            "tirx.LowerTVMBuiltin",
        ]
    ),
    # The recorded TileDispatch inputs leave the target to the context, as `tvm.compile` sets it.
    "TileDispatch": dict(sources=["tirx.TileDispatch", "tirx.LowerTIRx"], target="cuda"),
    "LowerTIRx": dict(sources=["tirx.LowerTIRx", "tirx.TileDispatch"], target="cuda"),
}


def _records(sources):
    for source in sources:
        yield from sorted((CORPUS / source).glob("*.json"))


def _run(pass_, text, target=None):
    """Run `pass_` on a fresh copy of the record: module passes may update their input.

    `target`, when given, is the current target around the run (`Target::Current`); the corpus
    does not record the one the pass ran under.
    """
    record = tvm.ir.load_json(text)
    with contextlib.ExitStack() as stack:
        if target is not None:
            stack.enter_context(tvm.target.Target(target))
        stack.enter_context(tvm.transform.PassContext(config=dict(record["config"])))
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
                outcome.append(("ok", _run(pass_, text, spec.get("target"))))
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


def test_common_subexpr_elim_unit_inputs(monkeypatch):
    """Every module the C++ CSE unit tests run the pass on: scopes of If, For and While, Let
    bodies, bool predicates, calls and loads, and a statement shared by unrolling."""
    import importlib.util

    path = (
        HERE.parents[1] / "tests/python/tirx/transform/test_tirx_transform_common_subexpr_elim.py"
    )
    spec = importlib.util.spec_from_file_location("cse_unit_tests", path)
    tests = importlib.util.module_from_spec(spec)
    # TVMScript reads the source of the classes it parses through the module.
    monkeypatch.setitem(sys.modules, spec.name, tests)
    spec.loader.exec_module(tests)
    native_factory = tvm.tirx.transform.CommonSubexprElim
    rust_factory = tvm_ffi.get_global_func("rust.tirx.transform.CommonSubexprElim")
    compared = []

    def both():
        native, rust = native_factory(), rust_factory()

        def run(mod):
            # The Rust pass leaves its input alone; the native one may update it in place.
            actual = rust(mod)
            expected = native(mod)
            assert tvm_ffi.structural_equal(expected, actual, map_free_vars=True)
            compared.append(expected)
            return expected

        return run

    monkeypatch.setattr(tvm.tirx.transform, "CommonSubexprElim", both)
    for name in sorted(dir(tests)):
        if name.startswith("test_"):
            getattr(tests, name)()
    assert len(compared) >= 20


def _allreduce_targets():
    return {
        "cuda": tvm.target.Target("cuda", host="llvm"),
        "rocm": tvm.target.Target("rocm", host="llvm"),
        "metal": tvm.target.Target(
            {"kind": "metal", "max_threads_per_block": 1024, "thread_warp_size": 32, "host": "llvm"}
        ),
        "webgpu": tvm.target.Target({"kind": "webgpu", "supports_subgroups": True, "host": "llvm"}),
        "webgpu1024": tvm.target.Target(
            {"kind": "webgpu", "max_num_threads": 1024, "supports_subgroups": True, "host": "llvm"}
        ),
        "llvm": tvm.target.Target("llvm"),
    }


def _allreduce_sum_1d(target, n, dtype="float32"):
    """A sum over `n` threads, in a loop over 4 rows."""
    from tvm.script import ir as I
    from tvm.script import s_tir as Ts
    from tvm.script import tirx as T

    zero = tvm.tirx.const(0, dtype)

    @I.ir_module
    class Module:
        @Ts.function(private=True)
        def main(A: T.Tensor((4, n), dtype), B: T.Tensor((4,), dtype)):
            T.func_attr({"target": target})
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(4 * n, dtype, data=A.data)])
            for i in range(4):
                threadIdx_x = T.launch_thread("threadIdx.x", n)
                reduce = T.alloc_tensor(ty_args=[T.Tensor((1,), dtype, scope="local")])
                T.gpu_thread_allreduce(
                    T.Lambda([dtype, dtype], lambda x, y: (x + y,)),
                    (zero,),
                    (A_flat[i * n + threadIdx_x],),
                    T.bool(True),
                    (reduce[0],),
                    (threadIdx_x,),
                )
                if threadIdx_x == 0:
                    B[i] = reduce[0]

    return Module


def _allreduce_predicated(target, n):
    """A sum whose condition is not constant, so the inputs are selected against the inits."""
    from tvm.script import ir as I
    from tvm.script import s_tir as Ts
    from tvm.script import tirx as T

    @I.ir_module
    class Module:
        @Ts.function(private=True)
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((1,), "float32")):
            T.func_attr({"target": target})
            threadIdx_x = T.launch_thread("threadIdx.x", n)
            reduce = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                (T.float32(0),),
                (A[threadIdx_x],),
                threadIdx_x < n - 3,
                (reduce[0],),
                (threadIdx_x,),
            )
            if threadIdx_x == 0:
                B[0] = reduce[0]

    return Module


def _allreduce_argmax_2d(target, ny, nx, dtype="float32"):
    """Two values (a max and its index) reduced over threadIdx.x, in groups along threadIdx.y."""
    from tvm.script import ir as I
    from tvm.script import s_tir as Ts
    from tvm.script import tirx as T

    @I.ir_module
    class Module:
        @Ts.function(private=True)
        def main(
            A: T.Tensor((ny * nx,), dtype), B: T.Tensor((ny,), dtype), C: T.Tensor((ny,), "int32")
        ):
            T.func_attr({"target": target})
            threadIdx_y = T.launch_thread("threadIdx.y", ny)
            red_v = T.alloc_tensor(ty_args=[T.Tensor((1,), dtype, scope="local")])
            red_i = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32", scope="local")])
            threadIdx_x = T.launch_thread("threadIdx.x", nx)
            T.gpu_thread_allreduce(
                T.Lambda(
                    [dtype, "int32", dtype, "int32"],
                    lambda v0, i0, v1, i1: (T.max(v0, v1), T.Select(v0 >= v1, i0, i1)),
                ),
                (T.min_value(dtype), T.int32(-1)),
                (A[threadIdx_y * nx + threadIdx_x], threadIdx_x),
                T.bool(True),
                (red_v[0], red_i[0]),
                (threadIdx_x,),
            )
            if threadIdx_x == 0:
                B[threadIdx_y] = red_v[0]
                C[threadIdx_y] = red_i[0]

    return Module


def _allreduce_over_y(target, ny, nx):
    """A sum over threadIdx.y, which is not contiguous in the flattened thread index."""
    from tvm.script import ir as I
    from tvm.script import s_tir as Ts
    from tvm.script import tirx as T

    @I.ir_module
    class Module:
        @Ts.function(private=True)
        def main(A: T.Tensor((ny * nx,), "float32"), B: T.Tensor((nx,), "float32")):
            T.func_attr({"target": target})
            threadIdx_y = T.launch_thread("threadIdx.y", ny)
            reduce = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            threadIdx_x = T.launch_thread("threadIdx.x", nx)
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                (T.float32(0),),
                (A[threadIdx_y * nx + threadIdx_x],),
                T.bool(True),
                (reduce[0],),
                (threadIdx_y,),
            )
            if threadIdx_y == 0:
                B[threadIdx_x] = reduce[0]

    return Module


def _allreduce_over_xy(target, ny, nx):
    """A sum over both thread axes."""
    from tvm.script import ir as I
    from tvm.script import s_tir as Ts
    from tvm.script import tirx as T

    @I.ir_module
    class Module:
        @Ts.function(private=True)
        def main(A: T.Tensor((ny * nx,), "float32"), B: T.Tensor((1,), "float32")):
            T.func_attr({"target": target})
            threadIdx_y = T.launch_thread("threadIdx.y", ny)
            threadIdx_x = T.launch_thread("threadIdx.x", nx)
            reduce = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                (T.float32(0),),
                (A[threadIdx_y * nx + threadIdx_x],),
                T.bool(True),
                (reduce[0],),
                (threadIdx_x, threadIdx_y),
            )
            if threadIdx_x == 0:
                B[0] = reduce[0]

    return Module


def _allreduce_aliases(target, alias_after):
    """The result consumed through a declaration over the destination's data, declared before or
    after the allreduce."""
    from tvm.script import ir as I
    from tvm.script import s_tir as Ts
    from tvm.script import tirx as T

    if alias_after:

        @I.ir_module
        class After:
            @Ts.function(private=True)
            def main(A: T.Tensor((4, 128), "float32"), B: T.Tensor((4,), "float32")):
                T.func_attr({"target": target})
                threadIdx_y = T.launch_thread("threadIdx.y", 4)
                cross = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                threadIdx_x = T.launch_thread("threadIdx.x", 128)
                A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor((512,), "float32", data=A.data)])
                T.gpu_thread_allreduce(
                    T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                    (T.float32(0),),
                    (A_flat[threadIdx_y * 128 + threadIdx_x],),
                    T.bool(True),
                    (cross[0],),
                    (threadIdx_x,),
                )
                cross_alias = T.decl_tensor(
                    cross.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
                )
                cross[0] = cross_alias[0]
                if threadIdx_x == 0:
                    B_flat = T.decl_tensor(B.data, ty_args=[T.Tensor((4,), "float32", data=B.data)])
                    B_flat[threadIdx_y] = cross_alias[0]

        return After

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((128, 32), "float32"), B: T.Tensor(128, "float32")):
            T.func_attr({"target": target})
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(4096, "float32", data=A.data)])
            for i in range(128):
                threadIdx_x = T.launch_thread("threadIdx.x", 32)
                reduce_data = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                reduce = T.decl_tensor(
                    reduce_data.data, ty_args=[T.Tensor(1, "float32", scope="local")]
                )
                reduce_alias = T.decl_tensor(
                    reduce.data, ty_args=[T.Tensor(1, "float32", scope="local")]
                )
                T.gpu_thread_allreduce(
                    T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                    (T.float32(0),),
                    (A_flat[0],),
                    T.bool(True),
                    (reduce_alias[0],),
                    (threadIdx_x,),
                )
                if threadIdx_x == 0:
                    B[i] = reduce_alias[0]

    return Before


def _allreduce_modules():
    targets = _allreduce_targets()
    cases = {}
    # Within a warp, across warps, and through shared memory (an extent that is no multiple of
    # the warp size, a narrow type, a target without shuffles), and the extent 1 that needs none.
    for target in ["cuda", "rocm", "metal", "webgpu", "webgpu1024", "llvm"]:
        for n in [1, 16, 32, 48, 64, 128, 1024]:
            cases[f"sum_1d[{target},{n}]"] = _allreduce_sum_1d(targets[target], n)
    for dtype in ["int8", "float16", "int64", "float64", "int32"]:
        for target in ["cuda", "rocm"]:
            cases[f"sum_1d[{target},64,{dtype}]"] = _allreduce_sum_1d(targets[target], 64, dtype)
    for target in ["cuda", "metal", "llvm"]:
        for n in [32, 128]:
            cases[f"predicated[{target},{n}]"] = _allreduce_predicated(targets[target], n)
    for target in ["cuda", "rocm", "webgpu1024", "llvm"]:
        for ny, nx in [(4, 16), (2, 64), (4, 128), (32, 32), (3, 40)]:
            cases[f"argmax_2d[{target},{ny}x{nx}]"] = _allreduce_argmax_2d(targets[target], ny, nx)
            cases[f"over_y[{target},{ny}x{nx}]"] = _allreduce_over_y(targets[target], ny, nx)
            cases[f"over_xy[{target},{ny}x{nx}]"] = _allreduce_over_xy(targets[target], ny, nx)
    for target in ["cuda", "webgpu"]:
        for after in [False, True]:
            cases[f"aliases[{target},after={after}]"] = _allreduce_aliases(targets[target], after)
    return cases


def test_lower_thread_allreduce_paths():
    """Inputs the native pass rewrites, as the corpus has none."""
    prepare = tvm.transform.Sequential(
        [tvm.s_tir.transform.LowerOpaqueBlock(), tvm.s_tir.transform.LowerThreadBinding()]
    )
    native = tvm.tirx.transform.LowerThreadAllreduce()
    rust = tvm_ffi.get_global_func("rust.tirx.transform.LowerThreadAllreduce")()
    changed = both_failed = 0
    failures = []
    for name, module in _allreduce_modules().items():
        module = prepare(module)
        outcome = []
        for pass_ in (native, rust):
            try:
                outcome.append(("ok", pass_(module)))
            except Exception as error:
                outcome.append(("error", error))
        (native_kind, expected), (rust_kind, actual) = outcome
        if native_kind == rust_kind == "error":
            both_failed += 1
        elif native_kind != rust_kind:
            detail = expected if native_kind == "error" else actual
            failures.append(f"{name}: native {native_kind}, rust {rust_kind}: {detail!s:.300}")
        else:
            changed += not tvm_ffi.structural_equal(module, expected, map_free_vars=True)
            if not tvm_ffi.structural_equal(expected, actual, map_free_vars=True):
                mismatch = tvm_ffi.get_first_structural_mismatch(
                    expected, actual, map_free_vars=True
                )
                failures.append(f"{name}: results differ at {mismatch}")
    print(f"LowerThreadAllreduce paths: {changed} changed, {both_failed} failed on both sides")
    assert not failures, "\n".join(failures[:20])


def _lower_tvm_builtin_modules():
    """LowerTVMBuiltin branches the corpus barely reaches: packed calls with every argument kind
    (int, float, string, handle, bool, tensor, a nested packed call), `call_cpacked`, shape stacks
    with and without extents, a parallel loop with its own stacks, a device allocation freed per
    branch and at the end of the device region, the disable annotation, CPU stack and workspace
    allocations, and an OpenCL image allocation."""
    from tvm.script import tirx as T

    float_ptr = tvm.ir.PtrType(tvm.ir.PrimType("float32"))
    image = tvm.ir.Call(
        "tirx.opencl.nd_mem_alloc_with_scope",
        [tvm.ir.StringImm("global.texture"), 2, tvm.tirx.const(16, "int64")],
        ty=float_ptr,
    )

    @T.function
    def gpu(A: T.Tensor((16,), "float32"), n: T.int32, x: T.float32, h: T.handle, flag: T.bool):
        T.func_attr(
            {
                "target": T.target("cuda", host="llvm"),
                "global_symbol": "gpu",
                "tirx.is_host_func": True,
            }
        )
        with T.device_context(2, 0):
            T.evaluate(T.call_packed("f", n, x, "text", h, flag, A, T.call_packed("g", n)))
            T.evaluate(T.call_cpacked("c", n, T.int64(3)))
            T.evaluate(T.call_packed("s", T.stack_make_shape(n, 4), T.stack_make_shape()))
            for i in T.parallel(4):
                T.evaluate(T.call_packed("p", i, A))
            if n > 0:
                B = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                T.evaluate(T.call_packed("b", B.data))
            else:
                C = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32")])
                T.evaluate(T.call_packed("c", C.data))
            D = T.alloc_tensor(
                annotations={"disable_lower_builtin": 1}, ty_args=[T.Tensor((8,), "float32")]
            )
            E = T.alloc_tensor(ty_args=[T.Tensor((8,), "float32")])
            T.evaluate(T.call_packed("e", D.data, E.data))

    @T.function
    def cpu(n: T.int32):
        T.func_attr({"target": T.target("llvm"), "global_symbol": "cpu"})
        with T.device_context(1, 0):
            small = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            large = T.alloc_tensor(ty_args=[T.Tensor((1024,), "float32")])
            dynamic = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
            T.evaluate(T.call_packed("f", small.data, large.data, dynamic.data))

    @T.function
    def opencl(n: T.int32):
        T.func_attr(
            {
                "target": T.target("opencl", host="llvm"),
                "global_symbol": "opencl",
                "tirx.is_host_func": True,
            }
        )
        with T.device_context(4, 0):
            p = T.bind(image)
            T.evaluate(T.call_packed("use", p))

    @T.function
    def no_device(n: T.int32):
        T.func_attr({"target": T.target("cuda", host="llvm"), "tirx.is_host_func": True})
        B = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
        T.evaluate(T.call_packed("b", B.data))

    return {
        name: (lambda func=func: tvm.IRModule({name: func}))
        for name, func in [("gpu", gpu), ("cpu", cpu), ("opencl", opencl), ("no_device", no_device)]
    }


def test_lower_tvm_builtin_branches():
    native_pass = tvm.tirx.transform.LowerTVMBuiltin()
    rust_pass = tvm_ffi.get_global_func("rust.tirx.transform.LowerTVMBuiltin")()
    for name, module in _lower_tvm_builtin_modules().items():
        if name == "no_device":
            # A device allocation outside a device region has no device id: both reject it.
            with pytest.raises(Exception) as native_error:
                native_pass(module())
            with pytest.raises(Exception) as rust_error:
                rust_pass(module())
            assert type(native_error.value) is type(rust_error.value), name
            continue
        native = native_pass(module())
        rust = rust_pass(module())
        assert not tvm_ffi.structural_equal(native, module(), map_free_vars=True), name
        assert tvm_ffi.structural_equal(native, rust, map_free_vars=True), (
            name,
            tvm_ffi.get_first_structural_mismatch(native, rust, map_free_vars=True),
        )


_DTYPE_LEGALIZE_PASSES = [
    "BF16ComputeLegalize",
    "BF16StorageLegalize",
    "FP8ComputeLegalize",
    "FP8StorageLegalize",
]


def _comparing_factory(name, compared):
    """A stand-in for `tvm.tirx.transform.<name>` whose passes run the native and the Rust pass
    on copies of the input, require the same result (or the same error), and return the native
    one."""
    native_factory = getattr(tvm.tirx.transform, name)
    rust_factory = tvm_ffi.get_global_func(f"rust.tirx.transform.{name}")

    def factory(*args):
        native, rust = native_factory(*args), rust_factory(*args)

        def run(mod):
            text = tvm.ir.save_json(mod)
            try:
                expected = native(tvm.ir.load_json(text))
            except Exception as error:
                with pytest.raises(type(error)):
                    rust(tvm.ir.load_json(text))
                raise
            actual = rust(tvm.ir.load_json(text))
            tvm.ir.assert_structural_equal(actual, expected, map_free_vars=True)
            compared.append(name)
            return expected

        return run

    return factory


@pytest.mark.parametrize("family", ["bf16", "fp8"])
def test_dtype_legalize_unit_inputs(family, monkeypatch):
    """Every input of the native bf16 / fp8 legalize tests, legalized by both passes."""
    import importlib.util
    import inspect

    path = (
        HERE.parent.parent
        / "tests/python/tirx/transform"
        / (f"test_tirx_transform_{family}_legalize.py")
    )
    spec = importlib.util.spec_from_file_location(f"_native_{family}_legalize_tests", path)
    module = importlib.util.module_from_spec(spec)
    # TVMScript reads the source of the module a parsed function belongs to.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    compared = []
    for name in _DTYPE_LEGALIZE_PASSES:
        monkeypatch.setattr(tvm.tirx.transform, name, _comparing_factory(name, compared))
    for test_name in sorted(dir(module)):
        test = getattr(module, test_name)
        if not test_name.startswith("test_") or not callable(test):
            continue
        if not inspect.signature(test).parameters:
            test()
            continue
        for dtype in ("float8_e4m3fn", "float8_e5m2"):
            for promote_dtype in ("float16", "float32"):
                test(dtype=dtype, promote_dtype=promote_dtype)
    print(f"{family}: {len(compared)} pass runs compared")
    assert compared


@pytest.mark.parametrize(
    "dtype",
    [
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    ],
)
@pytest.mark.parametrize("promote_dtype", ["float16", "float32", "float64", "bfloat16"])
def test_fp8_legalize_formats(dtype, promote_dtype, monkeypatch):
    """Compute then storage legalization of every float8 format, through each promoted type: the
    rounding, exponent and bias branches of the bit-level conversion."""
    import importlib.util

    path = HERE.parent.parent / "tests/python/tirx/transform/test_tirx_transform_fp8_legalize.py"
    spec = importlib.util.spec_from_file_location("_native_fp8_formats", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    compared = []
    compute = _comparing_factory("FP8ComputeLegalize", compared)
    storage = _comparing_factory("FP8StorageLegalize", compared)
    target = tvm.target.Target("llvm")
    before = tvm.tirx.transform.BindTarget(target)(module.get_before(dtype))
    try:
        after = storage()(compute(promote_dtype)(before))
    except Exception:  # pylint: disable=broad-except
        # Both sides failed with the same kind of error (checked by the stand-in factory).
        return
    assert compared == ["FP8ComputeLegalize", "FP8StorageLegalize"]
    assert not tvm_ffi.structural_equal(after, before)


def _legalize_module(dtype):
    """Scalar and vector computation on `dtype`: bindings, Let, Select, min / max, comparisons,
    casts, constants, broadcasts and shuffles; one allocation is reached through a pointer, so
    it keeps its type, and one is not, so compute legalization promotes it."""
    from tvm.script import ir as I
    from tvm.script import tirx as T

    z = T.dynamic("z", dtype)

    @I.ir_module
    class Module:
        @T.function
        def scalars(Aptr: T.handle(dtype), Bptr: T.handle("float32")):
            T.func_attr({"global_symbol": "scalars"})
            A = T.decl_tensor(Aptr, ty_args=[T.Tensor((8,), dtype, data=Aptr)])
            B = T.decl_tensor(Bptr, ty_args=[T.Tensor((8,), "float32", data=Bptr)])
            C = T.alloc_tensor(ty_args=[T.Tensor((8,), dtype)])
            for i in range(8):
                x = T.bind(A[i] * T.FloatImm(dtype, 2.0))
                y = T.bind(T.max(x, T.min(A[i], T.FloatImm(dtype, 1.5))))
                C[i] = T.Select(x < y, x, T.Cast(B[i], dtype))
                B[i] = T.Cast(C[i], "float32") + T.Cast(T.Let(z + y, where={z: A[i]}), "float32")
            T.evaluate(T.call_extern("consume", C.data, ty="void"))

        @T.function
        def vectors(Aptr: T.handle(dtype), Bptr: T.handle(dtype)):
            T.func_attr({"global_symbol": "vectors"})
            A = T.decl_tensor(Aptr, ty_args=[T.Tensor((8,), dtype, data=Aptr)])
            B = T.decl_tensor(Bptr, ty_args=[T.Tensor((8,), dtype, data=Bptr)])
            D = T.alloc_tensor(ty_args=[T.Tensor((8,), dtype)])
            D[T.Ramp(0, 1, 4)] = A[T.Ramp(0, 1, 4)] + T.Broadcast(T.FloatImm(dtype, 1.0), 4)
            D[T.Ramp(4, 1, 4)] = T.Shuffle([A[T.Ramp(4, 1, 4)]], [3, 2, 1, 0])
            B[T.Ramp(0, 1, 4)] = T.max(D[T.Ramp(0, 1, 4)], D[T.Ramp(4, 1, 4)])
            for i in range(8):
                if A[i] >= B[i]:
                    B[i] = T.exp(A[i]) / B[i] - A[i]
                B[i] = T.Cast(T.Cast(A[i], "float32") * T.float32(0.5), dtype)

    return Module


@pytest.mark.parametrize(
    "dtype, compute, compute_args, storage",
    [
        ("bfloat16", "BF16ComputeLegalize", (), "BF16StorageLegalize"),
        ("float8_e4m3fn", "FP8ComputeLegalize", ("float16",), "FP8StorageLegalize"),
        ("float8_e5m2", "FP8ComputeLegalize", ("float32",), "FP8StorageLegalize"),
    ],
)
@pytest.mark.parametrize("target", [None, "llvm", "nvidia/geforce-rtx-2080-ti", "cuda"])
def test_dtype_legalize_branches(dtype, compute, compute_args, storage, target):
    compared = []
    module = _legalize_module(dtype)
    if target is not None:
        module = tvm.tirx.transform.BindTarget(tvm.target.Target(target))(module)
    after_compute = _comparing_factory(compute, compared)(*compute_args)(module)
    _comparing_factory(storage, compared)()(after_compute)
    assert compared == [compute, storage]


def _make_packed_api_modules():
    """MakePackedAPI inputs beyond the corpus: every scalar parameter and return kind, symbolic
    shapes bound twice or only through later parameters, explicit strides, element offsets, sub-byte
    and vector dtypes, CUDA/Vulkan/OpenCL devices, subroutine calls, and the skipped functions."""
    from tvm.script import ir as I
    from tvm.script import tirx as T

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int32")
    s = T.dynamic("s", "int32")
    off = T.dynamic("off", "int32")

    @I.ir_module
    class Scalars:
        @T.function
        def ret_int(a: T.int32, u: T.uint8) -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            if a > 0:
                return a
            return T.int32(u)

        @T.function
        def ret_bool(b: T.bool) -> T.bool:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return b

        @T.function
        def ret_float(x: T.float32, h: T.bfloat16, d: T.float64) -> T.float32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return x

        @T.function
        def ret_ptr(p: T.handle) -> T.handle:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return p

        @T.function
        def typed_ptr(p: T.handle("float32"), q: T.handle("int8")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            T.evaluate(p)

        @T.function
        def zero_args() -> T.int64:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            for i in T.parallel(1):
                T.evaluate(i)
            return T.int64(42)

    @I.ir_module
    class Symbolic:
        @T.function
        def main(
            A: T.Tensor((n + 1,), "int32"),
            B: T.Tensor((n,), "int32"),
            C: T.Tensor((m, 4), "float32", strides=(s, 1)),
            D: T.Tensor((m * 2,), "float16", elem_offset=off, offset_factor=4),
            E: T.Tensor((8,), "int4"),
            F: T.Tensor((4,), "float32", align=1),
            k: T.int64,
        ):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            for i in range(n):
                B[i] = A[i] + A[i + 1]

    @I.ir_module
    class Tensors:
        @T.function
        def main(
            A: T.Tensor((8,), "int1"),
            B: T.Tensor((8,), "uint4"),
            C: T.Tensor((), "float32"),
            D: T.Tensor((4,), "float32", elem_offset=4),
            E: T.Tensor((4,), "bool"),
            F: T.Tensor((4,), "float32x4"),
            G: T.Tensor((off,), "float16", elem_offset=off * 2, offset_factor=2),
            k: T.int64,
            h: T.float16,
        ):
            T.func_attr({"target": T.target("opencl", host="llvm"), "calling_conv": 0})
            T.evaluate(0)

    @I.ir_module
    class Cuda:
        @T.function
        def main(A: T.Tensor((m,), "float32"), B: T.Tensor((m,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.evaluate(0)

    @I.ir_module
    class Vulkan:
        @T.function
        def main(A: T.Tensor((m,), "float32"), B: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("vulkan", host="llvm")})
            T.evaluate(0)

    @I.ir_module
    class Calls:
        @T.function
        def main(A: T.handle("float32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            Calls.subroutine(A)
            Calls.internal(A)

        @T.function
        def subroutine(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            T.evaluate(A_data)

        @T.function(private=True)
        def internal(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A_data)

        @T.function
        def lowered(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm", host="llvm"), "calling_conv": 2})
            T.evaluate(A_data)

        @T.function
        def no_host(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A_data)

    x = tvm.tirx.Var("x", "int32")
    void_return = tvm.tirx.Function([x], tvm.ir.Return(tvm.tirx.call_extern("void", "f", x)))
    void_return = void_return.with_attr(
        {"target": tvm.target.Target("llvm", host="llvm"), "global_symbol": "void_return"}
    )
    return [Scalars, Symbolic, Tensors, Cuda, Vulkan, Calls, tvm.IRModule({"main": void_return})]


def test_make_packed_api_branches():
    native = tvm.tirx.transform.MakePackedAPI()
    rust = tvm_ffi.get_global_func("rust.tirx.transform.MakePackedAPI")()
    for before, expected, actual in zip(
        _make_packed_api_modules(),
        map(native, _make_packed_api_modules()),
        map(rust, _make_packed_api_modules()),
    ):
        assert not tvm_ffi.structural_equal(before, expected, map_free_vars=True)
        assert tvm_ffi.structural_equal(expected, actual, map_free_vars=True)


def _make_packed_api_rejected():
    """MakePackedAPI inputs both sides reject, with the message each must carry."""
    from tvm.script import ir as I
    from tvm.script import tirx as T

    @I.ir_module
    class NoTarget:
        @T.function
        def main(A: T.handle("float32")):
            T.evaluate(A)

    @I.ir_module
    class ParallelReturn:
        @T.function
        def main() -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            for i in T.parallel(2):
                return i

    @I.ir_module
    class Bfloat16Return:
        @T.function
        def main(x: T.bfloat16) -> T.bfloat16:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return x

    @I.ir_module
    class Float8Param:
        @T.function
        def main(x: T.float8_e4m3fn):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            T.evaluate(x)

    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")
    undefined = tvm.tirx.Function([x], tvm.ir.Evaluate(x + y)).with_attr(
        {"target": tvm.target.Target("llvm", host="llvm"), "global_symbol": "main"}
    )
    return [
        (NoTarget, "required the function to be annotated with tvm::attr::kTarget"),
        (ParallelReturn, "Return cannot be used in parallel scope"),
        (Bfloat16Return, "data type bfloat16 not supported yet"),
        (Float8Param, "kDLFloat, DLDataTypeCode::kDLBfloat"),
        (tvm.IRModule({"main": undefined}), r"variables \(y,\) are used, but are not passed"),
    ]


def test_make_packed_api_rejects():
    for factory in ("tirx.transform.MakePackedAPI", "rust.tirx.transform.MakePackedAPI"):
        pass_ = tvm_ffi.get_global_func(factory)()
        for mod, message in _make_packed_api_rejected():
            with pytest.raises(tvm.error.InternalError, match=message):
                pass_(mod)


class _Disagreement(list):
    """The inputs on which a native pass and its Rust port disagree."""


def _compare_on(native, rust, mod, compared):
    """Run both passes on separate copies of `mod`: the native one may update its input."""
    text = tvm.ir.save_json(mod)
    outcome = []
    for pass_ in (native, rust):
        try:
            outcome.append(("ok", pass_(tvm.ir.load_json(text))))
        except Exception as error:  # pylint: disable=broad-except
            outcome.append(("error", error))
    (native_kind, expected), (rust_kind, actual) = outcome
    if native_kind == rust_kind == "error":
        agree = type(expected) is type(actual)
    elif native_kind != rust_kind:
        agree = False
    else:
        agree = tvm_ffi.structural_equal(expected, actual, map_free_vars=True)
    compared.append((agree, native_kind, mod, expected, actual))
    return native_kind, expected


def _pipeline_comparing_factory(name, compared, stop=False):
    """A stand-in for `tvm.tirx.transform.<name>` that runs the native pass and the Rust one on
    each input and continues with the native result, or stops the pipeline after comparing."""
    native_factory = getattr(tvm.tirx.transform, name)
    rust_factory = tvm_ffi.get_global_func(f"rust.tirx.transform.{name}")

    def factory(*args):
        native, rust = native_factory(*args), rust_factory(*args)

        def apply(mod, _ctx):
            kind, result = _compare_on(native, rust, mod, compared)
            if stop:
                raise _StopPipeline()
            if kind == "error":
                raise result
            return result

        return tvm.transform.module_pass(apply, opt_level=0, name=f"tirx.{name}")

    return factory


class _StopPipeline(Exception):
    pass


def _assert_all_agree(compared, minimum):
    failures = []
    for agree, kind, _, expected, actual in compared:
        if agree:
            continue
        if kind == "ok" and not isinstance(actual, Exception):
            detail = tvm_ffi.get_first_structural_mismatch(expected, actual, map_free_vars=True)
        else:
            detail = (expected, actual)
        failures.append(f"native {kind}: {detail!s:.600}")
    assert not failures, "\n".join(failures)
    assert len(compared) >= minimum, f"only {len(compared)} inputs compared"


def _load_upstream(relative):
    import importlib.util  # pylint: disable=import-outside-toplevel

    path = HERE.parent.parent / relative
    spec = importlib.util.spec_from_file_location(f"upstream_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    # TVMScript finds the source of a class through its module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_host_device_upstream_inputs(monkeypatch):
    """Every input of the upstream SplitHostDevice tests, which still pass on the native results."""
    compared = []
    monkeypatch.setattr(
        tvm.tirx.transform,
        "SplitHostDevice",
        _pipeline_comparing_factory("SplitHostDevice", compared),
    )
    tests = [
        ("tests/python/tirx/transform/test_tirx_transform_split_host_device.py", None),
        (
            "tests/python/tirx/test_cuda_launch.py",
            ["test_launch_region_roundtrip_and_host_only_operands"],
        ),
        (
            "tests/python/tirx/test_cuda_backend_config.py",
            ["test_device_helpers_are_cloned_per_configuration"],
        ),
    ]
    for path, names in tests:
        module = _load_upstream(path)
        for name in names or [n for n in vars(module) if n.startswith("test_")]:
            getattr(module, name)()
    _assert_all_agree(compared, 20)


def _split_host_device_kernels():
    """Device kernels whose SplitHostDevice input the compile pipeline produces."""
    from tvm.script import tirx as T  # pylint: disable=import-outside-toplevel

    @T.function
    def dynamic_bytes(A: T.Tensor((1,), "int32"), n: T.int32, bytes_: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=bytes_))
        scratch = T.alloc_tensor(ty_args=[T.Tensor((n,), "int32", scope="shared.dyn")])
        if T.cuda.thread_idx("x") == 0:
            scratch[n - 1] = n
            A[0] = scratch[n - 1]

    @T.function
    def too_few_bytes():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=64))
        scratch = T.alloc_tensor(ty_args=[T.Tensor((128,), "int32", scope="shared.dyn")])
        scratch[0] = T.cuda.thread_idx("x")

    @T.function
    def enough_bytes():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=1024))
        scratch = T.alloc_tensor(ty_args=[T.Tensor((128,), "int32", scope="shared.dyn")])
        scratch[0] = T.cuda.thread_idx("x")

    @T.function
    def undeclared_bytes(A: T.Tensor((64,), "float32"), n: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(n, 2), block=(32, 2)))
        scratch = T.alloc_tensor(ty_args=[T.Tensor((n, 4), "float32", scope="shared.dyn")])
        scratch[0, 0] = A[T.cuda.thread_idx("x")]
        A[0] = scratch[0, 0]

    return [dynamic_bytes, too_few_bytes, enough_bytes, undeclared_bytes]


def test_split_host_device_cuda_launches(monkeypatch):
    """CUDA launches sizing their dynamic shared memory, as the compile pipeline hands them over."""
    compared = []
    monkeypatch.setattr(
        tvm.tirx.transform,
        "SplitHostDevice",
        _pipeline_comparing_factory("SplitHostDevice", compared, stop=True),
    )
    target = {"kind": "cuda", "backend_config": {"cuda": {"arch": "sm_80"}}}
    for kernel in _split_host_device_kernels():
        with pytest.raises(Exception):
            tvm.tirx.compile(kernel, target=target)
    _assert_all_agree(compared, 4)


def _split_host_device_modules():
    from tvm.script import ir as I  # pylint: disable=import-outside-toplevel
    from tvm.script import tirx as T  # pylint: disable=import-outside-toplevel

    @I.ir_module
    class Trainium:
        @T.function
        def main(B: T.Tensor((16,), "float32"), A: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("trn", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("trn")})
            A[0] = B[0]

    @I.ir_module
    class UntargetedScope:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("llvm")})
            T.region("tirx.device_scope", [])
            A[0] = T.float32(1)

    @I.ir_module
    class MetalReturn:
        @T.function
        def main() -> T.int32:
            T.func_attr({"global_symbol": "main", "target": T.target("metal", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("metal")})
            T.launch_thread("threadIdx.x", 16)
            return 0

    @I.ir_module
    class CudaReturn:
        @T.function
        def main() -> T.int32:
            T.func_attr({"global_symbol": "main", "target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            T.launch_thread("threadIdx.x", 16)
            return 0

    @I.ir_module
    class LegacySharedDyn:
        @T.function
        def main(A: T.Tensor((16,), "float32"), n: T.int32):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            tx = T.launch_thread("threadIdx.x", 16)
            m: T.int32 = n * 2
            scratch = T.alloc_tensor(ty_args=[T.Tensor((m,), "float32", scope="shared.dyn")])
            scratch[tx] = A[tx]
            A[tx] = scratch[tx]

    @I.ir_module
    class Placeholder:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            tx = T.launch_thread("threadIdx.x", 16)
            scratch = T.alloc_tensor(ty_args=[T.Tensor((0,), "float32", scope="shared.dyn")])
            A[tx] = scratch[tx]

    @I.ir_module
    class TwoSharedDyn:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            tx = T.launch_thread("threadIdx.x", 16)
            first = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32", scope="shared.dyn")])
            second = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32", scope="shared.dyn")])
            A[tx] = first[tx] + second[tx]

    @I.ir_module
    class IncompatibleExtents:
        @T.function
        def main(A: T.Tensor((64,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.region("tirx.device_scope", [], attrs={"target": T.target("cuda")})
            tx = T.launch_thread("threadIdx.x", 16)
            with T.launch_thread("threadIdx.x", 32) as ty:
                A[tx + ty] = T.float32(1)

    def pool(launch_fields, values, required, alloc):
        @I.ir_module
        class Pool:
            @T.function
            def main(A: T.Tensor((16,), "int32")):
                T.func_attr({"target": T.target("cuda", host="llvm")})
                with T.region(
                    "tirx.device_scope",
                    values,
                    attrs={
                        "target": T.target("cuda"),
                        "cuda.launch_fields": launch_fields,
                        "cuda.kernel_attrs": {},
                        "cuda.smem_required": required,
                    },
                ):
                    if alloc:
                        scratch = T.alloc_tensor(
                            ty_args=[T.Tensor((16,), "int32", scope="shared.dyn")]
                        )
                        scratch[0] = A[0]
                        A[1] = scratch[0]
                    else:
                        A[1] = A[0]

        return Pool

    grid = ["grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z"]
    pools = [
        pool(grid, [1, 1, 1, 32, 1, 1], 256, True),
        pool(grid, [1, 1, 1, 32, 1, 1], 16, True),
        pool(grid, [1, 1, 1, 32, 1, 1], 128, False),
        pool([*grid, "dynamic_smem_bytes"], [1, 1, 1, 32, 1, 1, 32], 128, False),
        pool([*grid, "dynamic_smem_bytes"], [1, 1, 1, 32, 1, 1, 512], 128, True),
        pool([*grid, "dynamic_smem_bytes"], [1, 1, 1, 32, 1, 1, 0], 0, False),
    ]
    return [
        Trainium,
        UntargetedScope,
        MetalReturn,
        CudaReturn,
        LegacySharedDyn,
        Placeholder,
        TwoSharedDyn,
        IncompatibleExtents,
        *pools,
    ]


def test_split_host_device_branches():
    """Trainium parameter order, untargeted scopes, kernel returns, legacy dynamic shared memory,
    shared memory pools and their errors."""
    native = tvm.tirx.transform.SplitHostDevice()
    rust = tvm_ffi.get_global_func("rust.tirx.transform.SplitHostDevice")()
    compared = []
    for mod in _split_host_device_modules():
        _compare_on(native, rust, mod, compared)
    _assert_all_agree(compared, 14)
    changed = sum(
        kind == "ok" and not tvm_ffi.structural_equal(mod, expected, map_free_vars=True)
        for _, kind, mod, expected, _ in compared
    )
    failed = sum(kind == "error" for _, kind, *_ in compared)
    assert changed >= 8 and failed >= 3, (changed, failed)


def _flatten_tensor_module():
    """FlattenTensor rewrites the corpus does not reach: masked loads and stores through plain and
    strided parameters, a local tensor, and a view whose layout iterators load a local tensor."""
    from tvm.script import tirx as T

    tile = T.TileLayout.from_iters

    @T.function
    def main(
        A: T.Tensor((4, 8), "float32"),
        B: T.Tensor((4, 8), "float32", strides=(16, 1)),
        k: T.int32,
    ):
        T.func_attr({"global_symbol": "main"})
        mask = T.meta_var(T.Broadcast(T.bool(True), 4))
        n = T.alloc_local([1], "int32")
        n[0] = 4
        data = T.alloc_shared([64], "float32")
        view = T.decl_tensor(
            data.data,
            ty_args=[
                T.Tensor(
                    (n[0], 8),
                    "float32",
                    scope="shared",
                    layout=tile(shard=[T.Iter(n[0], 8, "m"), T.Iter(8, 1, "m")], replica=[]),
                )
            ],
        )
        view[1, k] = A[1, 2]
        loaded = T.meta_var(T.masked_load(A, k, T.Ramp(0, 1, 4), mask, ty="float32x4"))
        T.masked_store(B, loaded, 2, T.Ramp(4, 1, 4), mask)
        T.masked_store(
            data,
            T.masked_load(view, 1, T.Ramp(0, 1, 4), mask, ty="float32x4"),
            T.Ramp(k, 1, 4),
            mask,
        )

    return tvm.IRModule({"main": main})


def test_flatten_tensor_masked_accesses():
    native = tvm.tirx.transform.FlattenTensor()(_flatten_tensor_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.FlattenTensor")()(_flatten_tensor_module())
    assert not tvm_ffi.structural_equal(native, _flatten_tensor_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _lower_tirx_cleanup_module():
    """LowerTIRxCleanup rewrites the corpus does not reach: a tensor bound to a parameter with a
    layout, allocations spanning a tile layout's replica and offset and an unaligned swizzle over
    a gapped tile, and a declaration flattening leaves alone."""
    from tvm.script import tirx as T

    tile = T.TileLayout.from_iters

    @T.function
    def main(
        A: T.Tensor(
            (8, 16), "float32", layout=tile(shard=[T.Iter(8, 32, "m"), T.Iter(16, 1, "m")])
        ),
        B: T.Tensor((64,), "float32"),
        n: T.int32,
    ):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "main"})
        V: T.let[
            T.Tensor(
                (8, 16), "float32", layout=tile(shard=[T.Iter(8, 32, "m"), T.Iter(16, 1, "m")])
            )
        ] = A
        S = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (4, 4),
                    "float32",
                    scope="shared",
                    layout=tile(
                        shard=[T.Iter(4, 8, "m"), T.Iter(4, 1, "m")],
                        replica=[T.Iter(2, 64, "m")],
                        offset={"m": 3},
                    ),
                )
            ]
        )
        W = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (8, 64),
                    "float16",
                    scope="shared",
                    layout=T.ComposeLayout(
                        3,
                        3,
                        3,
                        tile(shard=[T.Iter(8, 128, "m"), T.Iter(64, 1, "m")]),
                        swizzle_inner=False,
                    ),
                )
            ]
        )
        ptr: T.let[T.Ptr(T.float32, "shared")] = T.reinterpret(B[0], ty=T.Ptr(T.float32, "shared"))
        D = T.decl_tensor(ptr, ty_args=[T.Tensor((64,), "float32", scope="shared", layout=None)])
        S[1, 2] = A[n, 3]
        V[n, 2] = T.float32(1)
        W[2, 5] = T.float16(1)
        D[n] = B[n] + S[3, 3]
        T.evaluate(T.call_extern("int32", "use", V.data))

    return tvm.IRModule({"main": main})


def test_lower_tirx_cleanup_layouts():
    native = tvm_ffi.get_global_func("tirx.transform.LowerTIRxCleanup")()(
        _lower_tirx_cleanup_module()
    )
    rust = tvm_ffi.get_global_func("rust.tirx.transform.LowerTIRxCleanup")()(
        _lower_tirx_cleanup_module()
    )
    assert not tvm_ffi.structural_equal(native, _lower_tirx_cleanup_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _rejected_tensor_modules():
    """A direct access through a thread-axis layout, a projection of an undefined tensor, and a
    DeclTensor alias over one."""
    from tvm.script import tirx as T

    @T.function
    def thread_axis(n: T.int32):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "thread_axis"})
        S = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (32, 4),
                    "float32",
                    scope="local",
                    layout=T.TileLayout.from_iters(
                        shard=[T.Iter(32, 1, "laneid"), T.Iter(4, 1, "m")]
                    ),
                )
            ]
        )
        S[0, n] = T.float32(1)

    undefined = tvm.tirx.decl_tensor((4,), "float32", "W")
    target = tvm.target.Target("cuda")

    def function(stmt):
        func = tvm.tirx.Function([], stmt)
        return tvm.IRModule({"main": func.with_attr({"target": target, "global_symbol": "main"})})

    projection = function(tvm.ir.Evaluate(tvm.tirx.call_extern("int32", "use", undefined.data)))
    alias = tvm.tirx.decl_tensor((4,), "float32", "V")
    declaration = tvm.ir.Call(
        tvm.ir.Op.get("tirx.decl_tensor"), [undefined.data], ty=alias.ty, ty_args=[alias.ty]
    )
    aliased = function(
        tvm.ir.SeqStmt(
            [
                tvm.ir.Bind(alias, declaration),
                tvm.ir.Evaluate(tvm.tirx.call_extern("int32", "use", alias.data)),
            ]
        )
    )
    return tvm.IRModule({"main": thread_axis}), projection, aliased


@pytest.mark.parametrize("name", ["FlattenTensor", "LowerTIRxCleanup"])
def test_rejected_tensor_accesses(name):
    thread_axis, projection, aliased = _rejected_tensor_modules()
    if name == "FlattenTensor":
        cases = [
            (projection, "Tensor W is used before its definition"),
            (aliased, "Tensor W is used before its definition"),
        ]
    else:
        cases = [
            (
                thread_axis,
                "Cannot lower direct TensorLoad/TensorStore on a tensor with thread-axis",
            ),
            (projection, "tensor_data_ptr projects W, which has no visible definition"),
            (aliased, "Tensor alias source W must be registered before its DeclTensor alias"),
        ]
    for prefix in ("", "rust."):
        pass_ = tvm_ffi.get_global_func(f"{prefix}tirx.transform.{name}")()
        for module, message in cases:
            with pytest.raises(Exception, match=message):
                pass_(module)


def test_lower_tirx_cleanup_requires_a_target():
    from tvm.script import tirx as T

    @T.function
    def main(A: T.Tensor((4, 4), "float32")):
        A[1, 2] = T.float32(1)

    for prefix in ("", "rust."):
        pass_ = tvm_ffi.get_global_func(f"{prefix}tirx.transform.LowerTIRxCleanup")()
        with pytest.raises(Exception, match="Target context required"):
            pass_(tvm.IRModule({"main": main}))
        with tvm.target.Target("cuda"):
            pass_(tvm.IRModule({"main": main}))


def _vectorize_loop_functions():
    """VectorizeLoop rewrites the corpus misses: every widened operator, scalar operands that
    change (always-rebuilt calls), scalarized statements, merged and concatenated ramps, `Let`,
    `Shuffle`, nested and serial loops, branches, unroll policies, RVV, tensor-level predication,
    OpenCL texture accesses, and the checks that reject a loop."""
    from tvm.script import tirx as T

    x86 = tvm.target.Target({"kind": "llvm", "mtriple": "x86_64-linux-gnu"})
    rvv = tvm.target.Target(
        {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu", "mattr": ["+v"]}
    )
    sve = tvm.target.Target(
        {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+v8.2a", "+sve"]}
    )

    @T.function
    def arith(
        A: T.Tensor((64,), "float32"),
        B: T.Tensor((64,), "int32"),
        C: T.Tensor((64, 64), "int32"),
        n: T.int32,
    ):
        for i in T.vectorized(4):
            B[i] = T.truncdiv(i, 2) + T.truncmod(i, 3) + i // 2 + i % 3 + T.min(i, n) + T.max(i, 1)
            B[i + 4] = T.shift_left(i, 1) | (i & 3) ^ T.shift_right(T.bitwise_not(i), 1)
            B[i + 8] = T.Select(T.And(i < n, T.Or(i > 1, i == n)), i, n)
            B[i + 12] = (
                T.Cast(T.Not(i != n), "int32") + T.Cast(i <= n, "int32") + T.Cast(i >= n, "int32")
            )
            B[i * 3 + 16] = B[3 * i] + B[i * n] + B[n * i] + B[i * 0] + B[i * -1 + 8]
            B[i + 32] = n * 2 + i - n - (n - i)
            A[i] = T.reinterpret(B[i], ty="float32") + T.exp(A[i]) + T.Cast(B[i], "float32")
            B[i + 36] = T.call_extern("g", n, ty="int32") * n + i
            B[i + 40] = T.call_extern("g", n, ty="int32") + n + i
            B[i + 44] = T.truncdiv(T.call_extern("g", n, ty="int32"), n) + i
            B[T.Ramp(T.call_extern("g", n, ty="int32"), 1, 4)] = T.Broadcast(i, 4)
            C[i, n] = n
            C[n, i] = n
            C[i, i] = n

    @T.function
    def scalarize(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "int32"), n: T.int32):
        for i in T.vectorized(4):
            A[i] = T.call_extern("f", A[i], ty="float32")
            T.evaluate(T.call_extern("g", n, ty="int32"))
            A[T.Ramp(i * 4, 1, 4)] = T.Broadcast(A[i], 4)
            A[T.Ramp(32, 1, 4)] = T.Broadcast(A[0], 4)
            v: T.let = T.call_extern("f", A[i], ty="float32")
            A[i + 16] = v
            w: T.let = n * 2
            B[i] = w + i
            u: T.let = n + 1
            B[u] = n

    @T.function
    def ramps(A: T.Tensor((128,), "float32"), n: T.int32):
        for i in T.vectorized(4):
            A[T.Ramp(i * 4, 1, 4)] = T.Broadcast(T.float32(1), 4)
            A[T.Ramp(i * 8, 1, 4)] = T.Broadcast(T.float32(2), 4)
            A[T.Ramp(64, i, 2)] = T.Broadcast(T.float32(3), 2)
            A[T.Ramp(i, n, 2)] = T.Broadcast(T.float32(4), 2)

    @T.function
    def lets(B: T.Tensor((64,), "int32"), n: T.int32):
        for i in T.vectorized(4):
            x = T.int32()
            y = T.int32()
            B[i] = T.Let(x * 2 + x, where={x: i + n})
            B[i + 4] = T.Let(y + i, where={y: n * 3})

    @T.function(check_well_formed=False)
    def lets_rebound(B: T.Tensor((64,), "int32"), n: T.int32):
        for i in T.vectorized(4):
            y = T.int32()
            B[i + 4] = T.Let(y + i, where={y: n * 3})
            B[i + 8] = T.Let(y + 1, where={y: n * 3})

    @T.function
    def loops(A: T.Tensor((256,), "float32"), n: T.int32):
        for i in T.vectorized(4):
            for j in range(n):
                A[i * 16 + j] = A[j] + T.float32(1)
            for k in range(i):
                A[i + k] = T.float32(2)
            for k2 in range(n):
                A[k2] = T.float32(3)
        for i2 in T.vectorized(4):
            for j2 in T.vectorized(4):
                A[i2 * 4 + j2] = T.float32(4)
        for i3 in T.vectorized(4):
            A[n] = T.float32(5)

    @T.function
    def branches(A: T.Tensor((64,), "float32"), n: T.int32):
        for i in T.vectorized(4):
            if n > 2:
                A[i] = T.float32(1)
            if i < 2:
                A[i] = T.float32(2)
            else:
                A[i] = T.float32(3)
            if T.call_extern("p", i, ty="bool"):
                A[i + 4] = T.float32(4)
            if n > 3:
                A[n] = T.float32(5)
            else:
                A[i] = T.float32(6)

    @T.function
    def unroll_policy(A: T.Tensor((256,), "float32")):
        for i in T.vectorized(4, annotations={"auto_unroll_max_step": 16, "unroll_explicit": 1}):
            for j in range(4):
                for j2 in range(4):
                    A[i * 64 + j * 4 + j2] = T.float32(1)
            for k in T.serial(4, annotations={"auto_unroll_max_step": 8}):
                A[i * 64 + k + 32] = T.float32(2)
            A[i] = T.float32(3)
        for i2 in T.vectorized(4, annotations={"unroll_explicit": 0, "pragma_other": 2}):
            for j3 in range(4):
                A[i2 * 4 + j3] = T.float32(1)

    @T.function
    def shuffles(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        for i in T.vectorized(4):
            B[i] = T.Shuffle([A[T.Ramp(0, 1, 4)]], [i])
            B[i + 8] = T.Shuffle([A[T.Ramp(i * 0, 1, 4)]], [i])
            v: T.let = T.Shuffle([A[T.Ramp(i * 0 + 8, 1, 2)]], [i % 2])
            B[0] = v

    @T.function
    def shuffle_lanes(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        for i in T.vectorized(4):
            B[i] = T.Shuffle([A[T.Ramp(i * 0 + 8, 1, 2)]], [i % 2])

    @T.function
    def shuffle_vectors(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        for i in T.vectorized(4):
            B[i] = T.Shuffle([A[T.Ramp(0, 1, 2)], A[T.Ramp(2, 1, 2)]], [0])

    @T.function
    def shuffle_index(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        for i in T.vectorized(4):
            B[i] = T.Shuffle([A[T.Ramp(0, 1, 4)]], [i + 1])

    @T.function
    def targets(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        T.func_attr({"target": rvv})
        for i in T.vectorized(14, annotations={"unroll_explicit": 1}):
            B[i] = A[i] + T.float32(1)
        for j in T.vectorized(8):
            B[j] = T.exp(A[j])
        for k in T.vectorized(T.int64(10)):
            A[k] = T.float32(1)

    @T.function
    def cuda(A: T.Tensor((64,), "float32x2")):
        T.func_attr({"target": T.target("cuda")})
        for i in T.vectorized(4):
            A[i] = T.Broadcast(T.float32(1), 2)

    @T.function
    def sve_ops(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "int32"), n: T.int32):
        T.func_attr({"target": sve})
        for i in T.vectorized(4 * T.vscale()):
            B[i] = i * 2 + n
            A[i] = T.Select(i < n, A[i], T.float32(0))
            B[i + 1] = T.if_then_else(n > 0, B[i], 0)

    @T.function
    def predicated(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32"), n: T.int32):
        for i in T.vectorized(4):
            if i < n:
                B[i] = A[i] + T.float32(1)
        for j in T.vectorized(4):
            if j < 7:
                B[j + 8] = A[j]
        for k in T.vectorized(4):
            if k < n:
                B[k] = T.float32(2)
                B[k * 2] = T.float32(3)
        for m in T.vectorized(4):
            if m < n:
                T.evaluate(0)
        for q in T.vectorized(4):
            if q + n < 9:
                B[q + n] = A[q + n]

    @T.function
    def predicated_rvv(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        T.func_attr({"target": rvv})
        for i in range(4):
            for j in T.vectorized(4):
                if i * 4 + j < 14:
                    B[i * 4 + j] = A[i * 4 + j] + T.float32(1)

    @T.function
    def nonzero_min(A: T.Tensor((64,), "float32")):
        for i in T.vectorized(1, 5):
            A[i] = T.float32(1)

    @T.function
    def dynamic_extent(A: T.Tensor((64,), "float32"), n: T.int32):
        for i in T.vectorized(n):
            A[i] = T.float32(1)

    @T.function
    def dynamic_extent_x86(A: T.Tensor((64,), "float32"), n: T.int32):
        T.func_attr({"target": x86})
        for i in T.vectorized(n):
            A[i] = T.float32(1)

    @T.function
    def unit_extent(A: T.Tensor((64,), "float32")):
        for i in T.vectorized(1):
            A[i] = T.float32(1)

    def texture(load_bits, store_bits):
        @T.function
        def main(
            a: T.handle("float32", "global.texture"), c: T.handle("float32", "global.texture")
        ):
            tx = T.launch_thread("threadIdx.x", 128)
            for e in T.vectorized(4):
                load = T.call_intrin(
                    "tirx.opencl.texture2d_load", a, tx, 0, 0, load_bits, e, ty="float32"
                )
                T.evaluate(
                    T.call_intrin(
                        "tirx.opencl.texture2d_store",
                        c,
                        tx,
                        0,
                        0,
                        store_bits,
                        load + T.float32(1),
                        ty="int32",
                    )
                )

        return main

    predicate = {"tirx.enable_tensor_level_predication": True}
    return {
        "texture": (texture(128, 128), {}),
        "texture_load_bits": (texture(64, 128), {}),
        "texture_store_bits": (texture(128, 64), {}),
        "arith": (arith, {}),
        "scalarize": (scalarize, {}),
        "ramps": (ramps, {}),
        "lets": (lets, {}),
        "lets_rebound": (lets_rebound, {}),
        "loops": (loops, {}),
        "branches": (branches, {}),
        "unroll_policy": (unroll_policy, {}),
        "shuffles": (shuffles, {}),
        "shuffle_lanes": (shuffle_lanes, {}),
        "shuffle_vectors": (shuffle_vectors, {}),
        "shuffle_index": (shuffle_index, {}),
        "targets": (targets, {}),
        "cuda": (cuda, {}),
        "sve_ops": (sve_ops, {}),
        "predicated": (predicated, predicate),
        "predicated_unset": (predicated, {}),
        "predicated_int": (predicated, {"tirx.enable_tensor_level_predication": 1}),
        "predicated_rvv": (predicated_rvv, predicate),
        "nonzero_min": (nonzero_min, {}),
        "dynamic_extent": (dynamic_extent, {}),
        "dynamic_extent_x86": (dynamic_extent_x86, {}),
        "unit_extent": (unit_extent, {}),
    }


def _run_vectorize_loop(factory, enable, func, config):
    try:
        with tvm.transform.PassContext(config=config):
            return "ok", factory(enable)(tvm.IRModule({"main": func}))
    except Exception as error:
        return type(error).__name__, str(error).strip().splitlines()[-1]


def test_vectorize_loop_branches():
    native = tvm_ffi.get_global_func("tirx.transform.VectorizeLoop")
    rust = tvm_ffi.get_global_func("rust.tirx.transform.VectorizeLoop")
    failures = []
    for name, (func, config) in _vectorize_loop_functions().items():
        for enable in (True, False):
            expected = _run_vectorize_loop(native, enable, func, config)
            actual = _run_vectorize_loop(rust, enable, func, config)
            if expected[0] != "ok" or actual[0] != "ok":
                if expected != actual:
                    failures.append(f"{name}[{enable}]: native {expected}, rust {actual}")
            elif not tvm_ffi.structural_equal(expected[1], actual[1], map_free_vars=True):
                failures.append(f"{name}[{enable}]: results differ")
    assert not failures, "\n".join(failures)


def _renormalize_split_pattern_module():
    """Every RenormalizeSplitPattern rule, matching and not: floordiv of floormod (scalar and
    broadcast, divisible or not), unsigned types, the factoring rules with two and three terms
    (scalar and broadcast, positive or not, with or without a common factor), `x < c`, the
    comparison rewrites, and the negation rules, inside and outside a loop."""
    t = tvm.tirx
    x, y, z = t.Var("x", "int32"), t.Var("y", "int32"), t.Var("z", "int32")
    u, w = t.Var("u", "uint32"), t.Var("w", "uint32")
    v, q = t.Var("v", "int32x4"), t.Var("q", "int32x4")
    i = t.Var("i", "int32")

    def c(value, dtype="int32"):
        return tvm.tirx.IntImm(dtype, value)

    def bc(value, lanes=4):
        return t.Broadcast(c(value), lanes)

    exprs = [
        t.FloorDiv(t.FloorMod(x, c(16)), c(4)),
        t.FloorDiv(t.FloorMod(x, c(15)), c(4)),
        t.FloorDiv(t.FloorMod(v, bc(16)), bc(4)),
        t.FloorDiv(t.FloorMod(v, bc(15)), bc(4)),
        t.FloorDiv(t.FloorMod(u, c(16, "uint32")), c(4, "uint32")),
        t.FloorDiv(t.Add(t.Mul(u, c(6, "uint32")), w), c(4, "uint32")),
        t.FloorDiv(t.Add(t.Mul(x, c(6)), y), c(4)),
        t.FloorDiv(t.Add(t.Mul(x, c(3)), y), c(4)),
        t.FloorDiv(t.Add(t.Mul(x, c(-6)), y), c(4)),
        t.FloorDiv(t.Add(t.Mul(x, c(6)), y), c(-4)),
        t.FloorDiv(t.Add(t.Mul(v, bc(6)), q), bc(4)),
        t.FloorDiv(t.Add(t.Mul(v, bc(3)), q), bc(4)),
        t.FloorDiv(t.Add(t.Add(t.Mul(x, c(6)), y), z), c(4)),
        t.FloorDiv(t.Add(t.Add(t.Mul(x, c(5)), y), z), c(4)),
        t.FloorDiv(t.Add(t.Add(t.Mul(v, bc(6)), q), v), bc(4)),
        t.FloorDiv(t.Add(t.Mul(x, c(12)), y), c(8)),
        t.FloorDiv(t.Add(t.Mul(x, c(64)), y), c(32)),
        t.FloorDiv(t.FloorMod(t.Add(t.Mul(x, c(8)), y), c(64)), c(4)),
        t.FloorDiv(x, y),
        t.FloorDiv(c(17), c(4)),
        t.LT(t.Add(x, c(1)), c(8)),
        t.LT(x, c(8)),
        t.LT(t.Mul(x, c(4)), c(16)),
        t.LT(t.Add(x, c(1)), c(0)),
        t.LT(t.Add(x, c(1)), y),
        t.LE(x, t.Add(y, c(1))),
        t.GT(x, t.Mul(y, c(2))),
        t.GE(t.Add(x, c(3)), c(8)),
        t.Not(t.Not(t.LT(x, y))),
        t.Not(t.LE(x, y)),
        t.Not(t.GE(x, y)),
        t.Not(t.LT(x, y)),
        t.Not(t.GT(x, y)),
        t.Not(t.LT(t.Add(x, c(2)), c(6))),
        t.Select(t.LT(x, c(4)), t.FloorDiv(t.FloorMod(x, c(16)), c(4)), y),
    ]
    body = [tvm.ir.Evaluate(e) for e in exprs]
    loop_body = tvm.ir.SeqStmt(
        [
            tvm.ir.Evaluate(t.FloorDiv(t.FloorMod(t.Add(t.Mul(i, c(4)), x), c(32)), c(4))),
            tvm.ir.Evaluate(t.LT(t.Add(t.Mul(i, c(2)), c(1)), c(10))),
        ]
    )
    body.append(tvm.ir.For(i, c(0), c(8), tvm.ir.ForKind.DEFAULT, loop_body))
    func = t.Function([x, y, z, u, w, v, q], tvm.ir.SeqStmt(body)).with_attr(
        {"global_symbol": "main"}
    )
    return tvm.IRModule({"main": func})


def test_renormalize_split_pattern_rules():
    native = tvm.tirx.transform.RenormalizeSplitPattern()(_renormalize_split_pattern_module())
    rust = tvm_ffi.get_global_func("rust.tirx.transform.RenormalizeSplitPattern")()(
        _renormalize_split_pattern_module()
    )
    assert not tvm_ffi.structural_equal(
        native, _renormalize_split_pattern_module(), map_free_vars=True
    )
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True), (
        tvm_ffi.get_first_structural_mismatch(native, rust, map_free_vars=True)
    )


def _lower_warp_memory_module(name):
    """LowerWarpMemory: shuffles across a warp of 16 threads with the local size rounded up to
    whole groups, a vector store and load, an index alias, stores from nested and sequential
    threadIdx.x launches, warp memory inside a loop, the mma intrinsics, int64 indices, and a
    target whose warp size is one."""
    from tvm.script import tirx as T

    @T.function
    def shuffle(
        A: T.Tensor((96,), "float16"), B: T.Tensor((128,), "float32"), C: T.Tensor((96,), "float16")
    ):
        T.func_attr({"target": T.target("cuda")})
        with T.launch_thread("blockIdx.x", 2) as bx:  # noqa: F841
            with T.launch_thread("threadIdx.y", 2) as ty:  # noqa: F841
                with T.launch_thread("threadIdx.x", 16) as tx:
                    W = T.alloc_tensor(ty_args=[T.Tensor((90,), "float16", scope="warp")])
                    V = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32", scope="warp")])
                    k = T.bind(tx * 2)
                    for i in range(3):
                        W[i * 32 + k] = A[i * 32 + tx * 2]
                    V[T.Ramp(tx * 4, 1, 4)] = B[T.Ramp(tx * 4, 1, 4)]
                    C[tx] = W[((tx + 4) % 16) * 2 + 1] + W[tx * 2 + 32]
                    B[T.Ramp(tx * 4, 1, 4)] = (
                        V[T.Ramp(((tx + 1) % 16) * 4, 1, 4)] + V[T.Ramp(tx * 4, 1, 4)]
                    )

    @T.function
    def launches(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        T.func_attr({"target": T.target("cuda")})
        with T.launch_thread("blockIdx.x", 2) as bx:
            W = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32", scope="warp")])
            with T.launch_thread("threadIdx.x", 32) as tx:
                W[tx * 2] = A[bx * 32 + tx]
                with T.launch_thread("threadIdx.x", 32) as tx2:
                    W[tx2 * 2 + 1] = A[bx * 32 + tx2]
            with T.launch_thread("threadIdx.x", 32) as tx3:
                B[bx * 32 + tx3] = W[((tx3 + 1) % 32) * 2] + W[tx3 * 2 + 1]
                for i in range(2):
                    L = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="warp")])
                    L[tx3] = B[i * 32 + tx3]
                    B[i * 32 + tx3] = L[(tx3 + i) % 32]

    @T.function
    def mma(S: T.Tensor((512,), "float16"), D: T.Tensor((256,), "float32")):
        T.func_attr({"target": T.target("cuda")})
        with T.launch_thread("threadIdx.x", 32) as tx:
            A = T.alloc_tensor(ty_args=[T.Tensor((256,), "float16", scope="warp")])
            B = T.alloc_tensor(ty_args=[T.Tensor((256,), "float16", scope="warp")])
            C = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32", scope="warp")])
            T.evaluate(
                T.ptx_legacy.ldmatrix(
                    False, 4, ".b16", A.data, tx * 8, S.ptr_to([0]), tx * 8, ty="float16"
                )
            )
            T.evaluate(
                T.ptx_legacy.ldmatrix(
                    True, 4, ".b16", B.data, tx * 8, S.ptr_to([256]), tx * 8, ty="float16"
                )
            )
            T.evaluate(T.cuda.mma_fill(8, C.data, 0, ty="float32"))
            T.evaluate(
                T.ptx_legacy.mma(
                    "m16n8k16",
                    "row",
                    "col",
                    "fp16",
                    "fp16",
                    "fp32",
                    A.data,
                    tx * 8,
                    B.data,
                    tx * 8 + 4,
                    C.data,
                    tx * 8,
                    False,
                    ty="float32",
                )
            )
            T.evaluate(T.cuda.mma_store(16, 16, D.ptr_to([0]), C.data, 0, 16, ty="float32"))

    @T.function
    def int64(A: T.Tensor((T.int64(64),), "float32")):
        T.func_attr({"target": T.target("cuda")})
        with T.launch_thread("threadIdx.x", T.int64(32)) as tx:
            W = T.alloc_tensor(ty_args=[T.Tensor((T.int64(64),), "float32", scope="warp")])
            for j in range(T.int64(2)):
                W[tx * T.int64(2) + j] = A[tx]
            A[tx] = W[((tx + T.int64(1)) % T.int64(32)) * T.int64(2) + T.int64(1)]

    @T.function
    def llvm(A: T.Tensor((64,), "float32")):
        T.func_attr({"target": T.target("llvm")})
        with T.launch_thread("threadIdx.x", 32) as tx:
            W = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="warp")])
            W[tx] = A[tx]
            A[tx] = W[(tx + 1) % 32]

    funcs = dict(shuffle=shuffle, launches=launches, mma=mma, int64=int64, llvm=llvm)
    return tvm.IRModule({"main": funcs[name]})


@pytest.mark.parametrize("name", ["shuffle", "launches", "mma", "int64", "llvm"])
def test_lower_warp_memory(name):
    native = tvm.tirx.transform.LowerWarpMemory()(_lower_warp_memory_module(name))
    rust = tvm_ffi.get_global_func("rust.tirx.transform.LowerWarpMemory")()(
        _lower_warp_memory_module(name)
    )
    unchanged = tvm_ffi.structural_equal(
        native, _lower_warp_memory_module(name), map_free_vars=True
    )
    assert unchanged == (name == "llvm")
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


# Name -> target, threadIdx.x extent, store index, and use of the warp tensor W.
_WARP_MEMORY_ERRORS = {
    "no_target": (None, 32, lambda tx: tx, lambda W, tx: W[tx]),
    "width": ("cuda", 24, lambda tx: tx, lambda W, tx: W[tx]),
    "nonlinear": ("cuda", 32, lambda tx: tx * tx % 32, lambda W, tx: W[tx]),
    "coefficient": ("cuda", 32, lambda tx: 31 - tx, lambda W, tx: W[tx]),
    "address": ("cuda", 32, lambda tx: tx, lambda W, tx: W.data),
}


@pytest.mark.parametrize("name", sorted(_WARP_MEMORY_ERRORS))
def test_lower_warp_memory_errors(name):
    from tvm.script import tirx as T

    target, extent, store_index, use = _WARP_MEMORY_ERRORS[name]

    @T.function
    def main(A: T.Tensor((64,), "float32")):
        with T.launch_thread("threadIdx.x", extent) as tx:
            W = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="warp")])
            W[store_index(tx)] = A[tx]
            T.evaluate(T.call_extern("int32", "use", use(W, tx)))

    if target is not None:
        main = main.with_attr("target", tvm.target.Target(target))
    messages = []
    for factory in ("tirx.transform.LowerWarpMemory", "rust.tirx.transform.LowerWarpMemory"):
        with pytest.raises(Exception) as error:
            tvm_ffi.get_global_func(factory)()(tvm.IRModule({"main": main}))
        messages.append(str(error.value).strip().splitlines()[-1])
    assert messages[0] == messages[1]


def _narrow_datatype_module():
    """NarrowDataType branches: casts, Lets (nested, with bodies analyzed under the variable's
    bound), Select/if_then_else, arithmetic and comparisons, clz, shifts and bitwise operations,
    vector lanes, unused loop variables, symbolic and overflowing extents, and launched threads."""
    from tvm.script import tirx as T

    x, y, z, w = (tvm.tirx.Var(name, "int64") for name in "xyzw")

    @T.function
    def exprs(A: T.Tensor((T.int64(64),), "float32"), B: T.Tensor((T.int64(64),), "int64")):
        T.func_attr({"global_symbol": "exprs"})
        for i in range(T.int64(8)):
            A[i * T.int64(8)] = A[T.Cast(i, "int64")] + T.float32(1)
        for i in range(T.int64(8)):
            A[T.Let(x * T.int64(2), where={x: i})] = T.float32(2)
        for i in range(T.int64(8)):
            A[i] = A[T.Let(y * T.int64(2), where={y: T.int64(3)})]
        for i in range(T.int64(8)):
            A[T.Let(T.Let(z + w, where={w: z * T.int64(2)}), where={z: i})] = T.float32(2)
        for i, j in T.grid(T.int64(8), T.int64(8)):
            A[T.Select(i < j, i, j)] = A[T.if_then_else(i >= T.int64(1), i - T.int64(1), j)]
        for i, j in T.grid(T.int64(8), T.int64(8)):
            A[T.min(i, j) + T.max(i, j) + T.floordiv(i, T.int64(2)) + T.floormod(j, T.int64(3))] = (
                A[T.truncdiv(i, T.int64(2)) + T.truncmod(j, T.int64(3))]
            )
        for i in range(T.int64(8)):
            if (
                (i * T.int64(8) != T.int64(5) and i <= T.int64(6))
                or i > T.int64(7)
                or i == T.int64(1)
            ):
                A[i] = T.float32(3)
        for i in range(T.int64(8)):
            B[i] = T.bitwise_not(i) + T.Cast(T.clz(i), "int64")

    @T.function
    def shifts(A: T.Tensor((T.int64(64),), "float32")):
        T.func_attr({"global_symbol": "shifts"})
        for i in range(T.int64(8)):
            A[T.shift_right(i, T.int64(1)) + T.shift_left(i, T.int64(2))] = T.float32(3)
        for i in range(T.int64(8)):
            A[T.bitwise_and(i, T.int64(7)) + T.bitwise_or(i, T.int64(1))] = T.float32(4)

    @T.function
    def vectors(A: T.Tensor((T.int64(64),), "float32"), C: T.Tensor((T.int64(64),), "float32")):
        T.func_attr({"global_symbol": "vectors"})
        for i in range(T.int64(8)):
            A[T.Ramp(i * T.int64(4), T.int64(1), 4)] = T.Broadcast(T.float32(4), 4)
            A[T.Broadcast(i, 4)] = T.Broadcast(T.float32(5), 4)
            C[i] = T.pow(C[i], T.float32(2))
        for i in range(T.int64(16)):
            A[i] = A[T.Shuffle([T.Ramp(i, T.int64(1), 2)], [T.int64(1)])]

    @T.function
    def extents(
        A: T.Tensor((T.int64(4096),), "float32"), B: T.Tensor((T.int64(4096),), "int64"), n: T.int64
    ):
        T.func_attr({"global_symbol": "extents"})
        for k in range(T.int64(8)):
            T.evaluate(0)
        for m in range(n):
            A[m] = T.float32(5)
        for v in range(T.int64(2**33)):
            B[T.floormod(v, T.int64(4096))] = v

    @T.function
    def threads(A: T.Tensor((T.int64(2048),), "float32")):
        T.func_attr({"global_symbol": "threads"})
        bx = T.launch_thread("blockIdx.x", T.int64(16))
        tx = T.launch_thread("threadIdx.x", T.int64(128))
        A[bx * T.int64(128) + tx] = A[tx] + T.float32(1)

    @T.function
    def small(A: T.Tensor((256,), "float32"), B: T.Tensor((70000,), "float32")):
        T.func_attr({"global_symbol": "small"})
        for i in range(16):
            for j in range(16):
                A[i * 16 + j] = A[j] * T.float32(2)
        for u in range(4):
            T.evaluate(0)
        for v in range(70000):
            B[v] = T.float32(1)

    functions = [exprs, shifts, vectors, extents, threads, small]
    return tvm.IRModule({f.attrs["global_symbol"]: f for f in functions})


@pytest.mark.parametrize("target_bits", [32, 16, 8])
def test_narrow_datatype_branches(target_bits):
    native = tvm.tirx.transform.NarrowDataType(target_bits)(_narrow_datatype_module())
    rust_pass = tvm_ffi.get_global_func("rust.tirx.transform.NarrowDataType")(target_bits)
    rust = rust_pass(_narrow_datatype_module())
    assert not tvm_ffi.structural_equal(native, _narrow_datatype_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _force_narrow_module():
    """ForceNarrowIndexToInt32 branches: casts of launched threads, scalar parameters, Select and
    if_then_else (with an int64 local scalar on one side), clz, clamped shift amounts (scalar,
    dynamic and vector), Let, bitwise not, narrow tensors, and a symbolic shape."""
    from tvm.script import tirx as T

    n = T.dynamic("n", "int64")
    x = tvm.tirx.Var("x", "int64")

    @T.function
    def threads(A: T.Tensor((T.int64(64),), "float32"), B: T.Tensor((T.int64(64),), "float32")):
        T.func_attr({"global_symbol": "threads"})
        bx = T.launch_thread("blockIdx.x", T.int64(2))
        tx = T.launch_thread("threadIdx.x", T.int64(32))
        B[T.Cast(bx, "int64") * T.int64(32) + T.Cast(tx, "int64")] = A[
            T.Cast(bx, "int64") * T.int64(32) + T.Cast(tx, "int64")
        ] + T.float32(1)

    @T.function
    def exprs(
        A: T.Tensor((T.int64(16),), "float32"),
        B: T.Tensor((16,), "float32"),
        C: T.Tensor((T.int64(4),), "int32"),
        D: T.Tensor((16,), "int16"),
        m: T.int64,
        s: T.int64,
    ):
        T.func_attr({"global_symbol": "exprs"})
        for i in T.serial(T.int64(4)):
            B[i] = T.Select(T.int64(1) <= i, A[i + m], T.Cast(i, "float32"))
            B[i + T.int64(4)] = A[T.if_then_else(m < T.int64(0), m + T.int64(1), m)]
            C[i] = T.clz(i)
            B[i + T.int64(8)] = A[
                T.shift_right(T.truncmod(m - T.int64(8), T.int64(6)), T.int64(63))
            ]
            B[T.int64(12)] = A[T.shift_left(T.truncmod(m, T.int64(1)) + i, s)]
            B[T.Ramp(T.int64(1), T.int64(1), 4)] = A[
                T.shift_right(
                    T.Broadcast(T.truncmod(m - T.int64(8), T.int64(6)), 4),
                    T.Ramp(T.int64(30), T.int64(1), 4),
                )
            ]
            B[T.Let(x + T.int64(1), where={x: i})] = A[T.bitwise_not(i) + T.int64(8)]
            D[i] = D[i] + T.int16(1)

    @T.function
    def scalars(A: T.Tensor((T.int64(4),), "float32"), B: T.Tensor((4,), "float32"), m: T.int64):
        T.func_attr({"global_symbol": "scalars"})
        opaque_index: T.int64 = T.call_extern("opaque_index", m, ty="int64")
        B[0] = A[T.if_then_else(m < T.int64(0), opaque_index, m)]
        B[1] = A[T.Select(m < T.int64(0), m, opaque_index)]

    @T.function
    def dynamic(Buf: T.Tensor([n], "int32")):
        T.func_attr({"global_symbol": "dynamic"})
        ceil_log2: T.int64 = T.Cast(T.ceil(T.log2(T.Cast(n, "float32"))), "int64")
        for i in T.serial(ceil_log2):
            Buf[i] = 0

    functions = [threads, exprs, scalars, dynamic]
    return tvm.IRModule({f.attrs["global_symbol"]: f for f in functions})


def test_force_narrow_index_to_i32_branches():
    native = tvm.tirx.transform.ForceNarrowIndexToInt32()(_force_narrow_module())
    rust_pass = tvm_ffi.get_global_func("rust.tirx.transform.ForceNarrowIndexToInt32")()
    rust = rust_pass(_force_narrow_module())
    assert not tvm_ffi.structural_equal(native, _force_narrow_module(), map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _force_narrow_rejected():
    """An int64 tensor parameter, an int64 tensor allocated in the body, and a constant that
    does not fit in int32."""
    from tvm.script import tirx as T

    @T.function
    def param(A: T.Tensor((16,), "int64")):
        for i in range(16):
            A[i] = A[i] + T.int64(1)

    @T.function
    def internal(A: T.Tensor((16,), "int32")):
        C = T.alloc_tensor(ty_args=[T.Tensor((16,), "int64")])
        for i in range(16):
            C[i] = T.Cast(A[i], "int64") + T.int64(1)
            A[i] = T.Cast(C[i], "int32")

    @T.function
    def large(A: T.Tensor((16,), "int32")):
        A[0] = T.Cast(T.int64(2**40), "int32")

    return [tvm.IRModule({"main": f}) for f in (param, internal, large)]


def test_force_narrow_index_to_i32_rejects_wide_integers():
    for factory in (
        "tirx.transform.ForceNarrowIndexToInt32",
        "rust.tirx.transform.ForceNarrowIndexToInt32",
    ):
        pass_ = tvm_ffi.get_global_func(factory)()
        for module in _force_narrow_rejected():
            with pytest.raises(tvm.error.InternalError):
                pass_(module)


def _storage_rewrite_threads():
    """Thread regions: the hardware axes share the outer one's allocation scope, where shared
    storage is reused; a vthread opens a scope of its own."""
    from tvm.script import tirx as T

    @T.function
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        T.func_attr({"target": T.target("cuda"), "global_symbol": "main"})
        with T.launch_thread("blockIdx.x", 2) as bx:
            with T.launch_thread("threadIdx.x", 32) as tx:
                S = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="shared")])
                S[tx] = A[bx * 32 + tx]
                R = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="shared")])
                R[tx] = S[tx] + T.float32(1)
                B[bx * 32 + tx] = R[tx]
                Q = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32", scope="shared")])
                Q[tx] = B[tx] * T.float32(2)
                A[tx] = Q[tx]
        with T.launch_thread("vthread", 2) as vt:
            V = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            for i in range(16):
                V[i] = A[vt * 16 + i]
            W = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            for i in range(16):
                W[i] = V[i] * T.float32(3)
            for i in range(16):
                B[vt * 16 + i] = W[i]

    return tvm.IRModule({"main": main})


def _storage_rewrite_inplace():
    """In-place updates: `Y[i] = f(X[i])` writes over X; an indirect load, a shifted read, a
    reduction, and a volatile allocation do not."""
    from tvm.script import tirx as T

    @T.function
    def main(
        A: T.Tensor((64,), "float32"), Idx: T.Tensor((64,), "int32"), B: T.Tensor((64,), "float32")
    ):
        T.func_attr({"global_symbol": "main"})
        X = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            X[i] = A[i]
        Y = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            Y[i] = X[i] + T.float32(1)
        Z = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            Z[i] = Y[Idx[i]] * T.float32(2)
        U = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(63):
            U[i] = Z[i + 1]
        V = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            V[i] = U[i]
            V[i] = V[i] + U[i]
        W = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        for i in range(64):
            if i < 32:
                W[i] = V[i] - T.float32(1)
        for i in range(64):
            B[i] = W[i]
        P = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32")])
        P[0] = B[0]
        Q = T.alloc_tensor(
            ty_args=[T.Tensor((64,), "float32")], annotations={"tirx.volatile": True}
        )
        Q[0] = P[0]
        B[1] = Q[0]

    return tvm.IRModule({"main": main})


def _storage_rewrite_reuse():
    """Reuse: symbolic sizes round robin, a larger buffer takes a smaller freed one (a merged
    size), a large extent widens to int64, and branches and asserts as scopes."""
    from tvm.script import tirx as T

    @T.function
    def main(n: T.int32, A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
        T.func_attr({"global_symbol": "main"})
        S1 = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
        for i in range(n):
            S1[i] = A[i]
        B[0] = S1[0]
        S2 = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
        for i in range(n):
            S2[i] = A[i] + T.float32(1)
        B[1] = S2[0]
        X = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32")])
        for i in range(256):
            X[i] = A[i]
        B[2] = X[255]
        Y = T.alloc_tensor(ty_args=[T.Tensor((1024,), "float32")])
        for i in range(1024):
            Y[i] = A[i] * T.float32(2)
        B[3] = Y[1023]
        L = T.alloc_tensor(ty_args=[T.Tensor((16777216,), "float32")])
        L[0] = A[0]
        B[4] = L[0]
        M = T.alloc_tensor(ty_args=[T.Tensor((268435456,), "float32")])
        M[0] = A[1]
        B[5] = M[0]
        if n > 4:
            C = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
            C[0] = A[0]
            B[6] = C[0]
        else:
            D = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
            D[0] = A[1]
            B[6] = D[0]
        assert n > 0, "positive"
        E = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
        E[0] = A[2]
        B[7] = E[0]

    return tvm.IRModule({"main": main})


def _storage_rewrite_dtypes(target):
    """Reuse across dtypes (Vulkan and WebGPU reuse only identical ones), and pointers into
    packed single-allocation storage (`local.L0A`, registered by the upstream suite)."""
    from tvm.script import tirx as T

    @T.function
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "int32")):
        T.func_attr({"target": T.target(target), "global_symbol": "main"})
        X = T.alloc_tensor(ty_args=[T.Tensor((256,), "int32")])
        for i in range(256):
            X[i] = B[i]
        B[0] = X[255]
        Y = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32")])
        for i in range(256):
            Y[i] = A[i]
        A[0] = Y[255]
        P = T.alloc_tensor(ty_args=[T.Tensor((64,), "float16", "local.L0A")])
        Q = T.alloc_tensor(ty_args=[T.Tensor((64,), "int32", "local.L0A")])
        for i in range(64):
            P[i] = T.Cast(A[i], "float16")
            Q[i] = B[i]
        T.evaluate(T.call_extern("int32", "consume", P.data, Q.data))
        A[2] = T.Cast(P[0], "float32") + T.Cast(Q[0], "float32")

    return tvm.IRModule({"main": main})


def _storage_rewrite_vector_merge():
    """A vector allocation reusing a scalar one is sized by the widest element type; the scalar
    accesses then fail the PointerValueTypeRewrite that StorageRewrite ends with."""
    from tvm.script import tirx as T

    @T.function
    def main(A: T.Tensor((256,), "float32")):
        T.func_attr({"global_symbol": "main"})
        Y = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32")])
        for i in range(256):
            Y[i] = A[i]
        A[0] = Y[255]
        V = T.alloc_tensor(ty_args=[T.Tensor((64,), "float32x4")])
        for i in range(64):
            V[i] = A[T.Ramp(i * 4, 1, 4)]
        A[T.Ramp(0, 1, 4)] = V[63]

    return tvm.IRModule({"main": main})


def _storage_rewrite_masked():
    """Masked accesses to planned and to retyped tensors."""
    from tvm.script import tirx as T

    @T.function
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main"})
        mask = T.Broadcast(T.bool(True), 4)
        X = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(4):
            T.evaluate(
                T.call_intrin(
                    "tirx.masked_store",
                    X,
                    T.call_intrin("tirx.masked_load", A, T.Ramp(i * 4, 1, 4), mask, ty="float32x4"),
                    T.Ramp(i * 4, 1, 4),
                    mask,
                    ty="void",
                )
            )
        B[0] = X[0]
        Y = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(4):
            Y[T.Ramp(i * 4, 1, 4)] = T.call_intrin(
                "tirx.masked_load", A, T.Ramp(i * 4, 1, 4), mask, ty="float32x4"
            )
        for i in range(4):
            T.evaluate(
                T.call_intrin(
                    "tirx.masked_store",
                    B,
                    T.call_intrin("tirx.masked_load", Y, T.Ramp(i * 4, 1, 4), mask, ty="float32x4"),
                    T.Ramp(i * 4, 1, 4),
                    mask,
                    ty="void",
                )
            )

    return tvm.IRModule({"main": main})


def _storage_rewrite_vectors():
    """Retyping: an allocation accessed as float32x4 whose data pointer is bound to a scalar
    pointer and passed on, a declaration over a pointer parameter, and an untyped pointer."""
    from tvm.script import tirx as T

    @T.function
    def main(
        A: T.Tensor((32,), "float32"),
        Bptr: T.handle("float32"),
        C: T.Tensor((32,), "float32"),
        D: T.handle(),
    ):
        T.func_attr({"global_symbol": "main"})
        B = T.decl_tensor(Bptr, ty_args=[T.Tensor((32,), "float32")])
        X = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32")])
        for i in range(8):
            X[T.Ramp(i * 4, 1, 4)] = A[T.Ramp(i * 4, 1, 4)]
        for i in range(8):
            B[T.Ramp(i * 4, 1, 4)] = X[T.Ramp(i * 4, 1, 4)]
        p = T.bind(X.data)
        T.evaluate(T.call_extern("int32", "consume", p, X.data, D))
        for i in range(8):
            C[T.Ramp(i * 4, 1, 4)] = A[T.Ramp(i * 4, 1, 4)] + B[T.Ramp(i * 4, 1, 4)]

    return tvm.IRModule({"main": main})


def _storage_rewrite_two_dimensional():
    """A two-dimensional allocation cannot be planned: both sides reject it."""
    from tvm.script import tirx as T

    @T.function
    def main(A: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main"})
        X = T.alloc_tensor(ty_args=[T.Tensor((4, 4), "float32")])
        for i in range(4):
            X[i, 0] = A[i]
        A[0] = X[3, 0]

    return tvm.IRModule({"main": main})


_STORAGE_REWRITE_MODULES = {
    "threads": _storage_rewrite_threads,
    "inplace": _storage_rewrite_inplace,
    "reuse": _storage_rewrite_reuse,
    "dtypes[llvm]": lambda: _storage_rewrite_dtypes("llvm"),
    "dtypes[vulkan]": lambda: _storage_rewrite_dtypes("vulkan"),
    "dtypes[webgpu]": lambda: _storage_rewrite_dtypes("webgpu"),
    "vector_merge": _storage_rewrite_vector_merge,
    "masked": _storage_rewrite_masked,
    "vectors": _storage_rewrite_vectors,
    "two_dimensional": _storage_rewrite_two_dimensional,
}


@pytest.mark.parametrize("name", list(_STORAGE_REWRITE_MODULES))
@pytest.mark.parametrize(
    "factory,config",
    [
        ("StorageRewrite", {}),
        # MergeSharedMemoryAllocations takes over reuse.
        ("StorageRewrite", {"tirx.merge_static_smem": True}),
        ("PointerValueTypeRewrite", {}),
    ],
    ids=["StorageRewrite", "StorageRewrite[merge_static_smem]", "PointerValueTypeRewrite"],
)
def test_storage_rewrite_branches(name, factory, config):
    outcome = []
    for prefix in ("", "rust."):
        pass_ = tvm_ffi.get_global_func(f"{prefix}tirx.transform.{factory}")()
        try:
            with tvm.transform.PassContext(config=config):
                outcome.append(("ok", pass_(_STORAGE_REWRITE_MODULES[name]())))
        except Exception as error:
            outcome.append(("error", error))
    (native_kind, expected), (rust_kind, actual) = outcome
    assert native_kind == rust_kind, f"native {native_kind}, rust {rust_kind}: {expected or actual}"
    if native_kind == "ok":
        assert tvm_ffi.structural_equal(expected, actual, map_free_vars=True), (
            tvm_ffi.get_first_structural_mismatch(expected, actual, map_free_vars=True)
        )
    if factory == "StorageRewrite" and native_kind == "ok":
        # Every module has allocations to plan.
        original = _STORAGE_REWRITE_MODULES[name]()
        assert not tvm_ffi.structural_equal(original, expected, map_free_vars=True)


def _parametrizations(test):
    """Keyword arguments of every case of a test's `pytest.mark.parametrize` marks."""
    axes = []
    for mark in getattr(test, "pytestmark", []):
        if mark.name != "parametrize":
            continue
        names, values = mark.args[0], mark.args[1]
        if isinstance(names, str):
            names = [name.strip() for name in names.split(",")]
        axes.append([dict(zip(names, value if len(names) > 1 else (value,))) for value in values])
    for combination in itertools.product(*axes):
        yield {key: value for case in combination for key, value in case.items()}


def _upstream_tests():
    for factory, path in SUITES.items():
        for node in ast.parse((REPO / path).read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                yield pytest.param(factory, node.name, id=f"{factory}::{node.name}")


def _dual_run(factory, runs):
    """A factory for `tirx.transform.<factory>` that runs the C++ pass and, on a copy of the
    module, the Rust one, requires them to agree, and returns the C++ result."""
    native_factory = getattr(tvm.tirx.transform, factory)
    rust_factory = tvm_ffi.get_global_func(f"rust.tirx.transform.{factory}")

    def make():
        native, rust = native_factory(), rust_factory()

        def run(mod, _ctx):
            copy = tvm.ir.load_json(tvm.ir.save_json(mod))
            try:
                expected = native(mod)
            except Exception:
                with pytest.raises(Exception):
                    rust(copy)
                runs.append("both failed")
                raise
            actual = rust(copy)
            assert tvm_ffi.structural_equal(expected, actual, map_free_vars=True), (
                tvm_ffi.get_first_structural_mismatch(expected, actual, map_free_vars=True)
            )
            runs.append("compared")
            return expected

        return tvm.transform.module_pass(run, opt_level=0, name=f"parity.{factory}")

    return make


@pytest.mark.parametrize("factory,test_name", list(_upstream_tests()))
def test_upstream_suite_parity(monkeypatch, factory, test_name):
    test = getattr(_suite(factory), test_name)
    runs = []
    monkeypatch.setattr(tvm.tirx.transform, factory, _dual_run(factory, runs))
    for kwargs in _parametrizations(test):
        test(**kwargs)
    assert runs, f"{test_name} never ran {factory}"


def _tile_callbacks_module():
    """A tile instruction whose dispatcher uses every callback of its context: a private
    allocation, a device and a host init statement, and a statement after the definition of the
    view it writes. The view is a `decl_tensor` over a parameter, so the host statement's
    `tensor_data_ptr` of the view moves onto the parameter."""
    from tvm.script import ir as I
    from tvm.script import tirx as T
    from tvm.tirx.tile.instruction import Instruction, Operand

    name = "__rust_parity_callbacks__"

    @T.function(private=True)
    def implementation():
        T.evaluate(0)

    def lower(view, sctx):
        tensor = view.call.args[0].source
        sctx.add_alloc_tensor(T.Var("scratch", T.Tensor((4,), "float32", scope="local")))

        @T.function(check_well_formed=False)
        def device_init():
            T.evaluate(T.int32(11))
            T.kernel_replace_point()

        @T.function(check_well_formed=False)
        def host_init():
            T.evaluate(tensor.data)
            T.kernel_replace_point()

        @T.function(check_well_formed=False)
        def after_definition():
            T.evaluate(tensor.data)
            T.kernel_replace_point()

        sctx.add_init_stmt(device_init.body)
        sctx.add_init_stmt(host_init.body, host=True)
        sctx.add_post_tensor_def_stmt(tensor, after_definition.body)
        return implementation

    spec = Instruction(
        "tirx.cuda.tile." + name,
        "copy",
        (Operand("dst", "region"), Operand("src", "region")),
        scopes=("cluster", "cta", "warpgroup", "warp", "thread"),
        lower=lower,
    )
    setattr(T.cuda.tile, name, spec.register())

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((64,), "float32", layout="default")):
            T.func_attr({"global_symbol": "main"})
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=(128,)))
            B = T.decl_tensor(
                A.data, ty_args=[T.Tensor((64,), "float32", layout=None, data=A.data)]
            )
            if T.cuda.warp_id() == 0:
                getattr(T.cuda.tile, name)(B[0:1], B[1:2])

    return Module


@pytest.mark.parametrize("factory", ["TileDispatch", "LowerTIRx"])
def test_tile_dispatch_callbacks(factory):
    module = _tile_callbacks_module()
    with tvm.target.Target("cuda"):
        native = tvm_ffi.get_global_func(f"tirx.transform.{factory}")()(module)
        rust = tvm_ffi.get_global_func(f"rust.tirx.transform.{factory}")()(module)
    assert not tvm_ffi.structural_equal(native, module, map_free_vars=True)
    assert tvm_ffi.structural_equal(native, rust, map_free_vars=True)


def _nested_launch_module():
    from tvm.script import ir as I
    from tvm.script import tirx as T

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((64,), "float32", layout="default")):
            T.func_attr({"global_symbol": "main"})
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=(128,)))
            with T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=(128,))):
                A[0] = T.float32(1)

    return Module


def test_tile_dispatch_errors():
    """A nested CUDA launch, and a function without a target outside a target context."""
    for factory in ("tirx.transform.TileDispatch", "rust.tirx.transform.TileDispatch"):
        with tvm.target.Target("cuda"):
            with pytest.raises(ValueError, match="Nested CUDA device_entry"):
                tvm_ffi.get_global_func(factory)()(_nested_launch_module())
        with pytest.raises(Exception, match="Target context required"):
            tvm_ffi.get_global_func(factory)()(_nested_launch_module())


def test_upstream_lower_tirx_suite():
    """`tests/python/tirx/transform/test_tirx_transform_lower_tirx.py` with `TileDispatch` and
    `LowerTIRx` routed to Rust (`rust_tile_plugin.py`). Its probes also check the execution
    context each dispatcher receives."""
    import subprocess
    import sys

    suite = HERE.parent.parent / "tests/python/tirx/transform/test_tirx_transform_lower_tirx.py"
    if not suite.exists():
        pytest.skip(f"{suite} not found")
    path = os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, PYTHONPATH=path, TVM_RUST_PASSES=str(LIBRARY), TVM_RUST_TILE_MODE="rust")
    command = [sys.executable, "-m", "pytest", str(suite), "-q", "-p", "no:cacheprovider"]
    command += ["-p", "rust_tile_plugin"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
