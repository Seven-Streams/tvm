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
"""Time every Rust tirx pass against its C++ counterpart on the recorded corpus.

Reuses the case table of ``rust/tests/test_pass_parity.py`` (same factories, arguments,
corpus directories and current target), so the timed runs are exactly the parity runs.

For each record both passes run once to check that they agree and that neither fails, then
``--repeat`` timed runs per side, interleaved (C++, Rust, Rust, C++, ...), each on a freshly
loaded copy of the module, with the PassContext entered outside the timer. The per-record
statistic is the median of the timed runs; per case the script reports the sum of those
medians on each side and the distribution of per-record ratios.

Build the crate with ``cargo build --release`` first and pin the process to one core::

    taskset -c 8 python rust/tests/bench/bench_passes.py --out bench.csv

Environment: ``TVM_RUST_PASSES`` (defaults to the release ``libtvm.so``), ``TVM_PASS_CORPUS``
(defaults to ``rust/tests/corpus/data``). See ``rust/PERFORMANCE.md``.
"""

import argparse
import contextlib
import csv
import gc
import importlib.util
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load_parity_module():
    # The parity test loads the Rust library named by TVM_RUST_PASSES at import time; the
    # benchmark wants the release build, not the test's debug default.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=20, help="timed runs per side and record")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="records per case (0 = all)")
    parser.add_argument("--cases", nargs="*", default=None, help="substring filters on case names")
    parser.add_argument("--out", default="bench_results.csv")
    args = parser.parse_args()

    parity = load_parity_module()
    import tvm_ffi

    import tvm

    CASES, _records = parity.CASES, parity._records

    @contextlib.contextmanager
    def context(target, config):
        with contextlib.ExitStack() as stack:
            if target is not None:
                stack.enter_context(tvm.target.Target(target))
            stack.enter_context(tvm.transform.PassContext(config=config))
            yield

    rows = []
    summaries = []
    cases = sorted(CASES)
    if args.cases:
        cases = [c for c in cases if any(f in c for f in args.cases)]

    for case in cases:
        spec = CASES[case]
        factory = spec.get("factory", case)
        fargs = spec.get("args", ())
        fargs = fargs() if callable(fargs) else fargs
        if "native" in spec:
            native = spec["native"]()
        else:
            native = tvm_ffi.get_global_func(f"tirx.transform.{factory}")(*fargs)
        rust = tvm_ffi.get_global_func(f"rust.tirx.transform.{factory}")(*fargs)
        target = spec.get("target")

        records = list(_records(spec["sources"]))
        if args.limit:
            records = records[: args.limit]
        timed = errors = unloadable = mismatches = 0
        error_sides = {}
        one_sided = []
        case_rows = []
        for path in records:
            text = path.read_text()
            try:
                record = tvm.ir.load_json(text)
            except Exception:
                unloadable += 1
                continue
            config = dict(record["config"])
            n_funcs = len(record["mod"].functions)
            del record

            def run_once(pass_):
                mod = tvm.ir.load_json(text)["mod"]
                with context(target, config):
                    return pass_(mod)

            outcome = {}
            for side, pass_ in (("cpp", native), ("rust", rust)):
                try:
                    outcome[side] = ("ok", run_once(pass_))
                except Exception as error:
                    outcome[side] = ("error", f"{type(error).__name__}: {error}"[:160])
            failed = [side for side, (kind, _) in outcome.items() if kind == "error"]
            if failed:
                errors += 1
                key = "both" if len(failed) == 2 else failed[0]
                error_sides[key] = error_sides.get(key, 0) + 1
                if key != "both" and len(one_sided) < 3:
                    one_sided.append(f"{path.name}: only {key} failed: {outcome[key][1]}")
                continue
            expected, actual = outcome["cpp"][1], outcome["rust"][1]
            agree = bool(tvm_ffi.structural_equal(expected, actual, map_free_vars=True))
            mismatches += not agree
            del expected, actual

            times = {"cpp": [], "rust": []}
            gc.collect()
            gc.disable()
            try:
                for i in range(args.warmup + args.repeat):
                    order = (("cpp", native), ("rust", rust))
                    if i % 2:
                        order = order[::-1]
                    for name, pass_ in order:
                        mod = tvm.ir.load_json(text)["mod"]
                        with context(target, config):
                            t0 = time.perf_counter_ns()
                            pass_(mod)
                            t1 = time.perf_counter_ns()
                        if i >= args.warmup:
                            times[name].append(t1 - t0)
                        del mod
            finally:
                gc.enable()
            timed += 1
            cpp_med = statistics.median(times["cpp"]) / 1e3
            rust_med = statistics.median(times["rust"]) / 1e3
            row = dict(
                case=case,
                record=f"{path.parent.name}/{path.name}",
                json_bytes=len(text),
                n_funcs=n_funcs,
                cpp_us=round(cpp_med, 2),
                rust_us=round(rust_med, 2),
                cpp_min_us=round(min(times["cpp"]) / 1e3, 2),
                rust_min_us=round(min(times["rust"]) / 1e3, 2),
                ratio=round(rust_med / cpp_med, 3) if cpp_med else float("nan"),
                agree=agree,
            )
            case_rows.append(row)
            rows.append(row)

        error_text = ";".join(f"{k}={v}" for k, v in sorted(error_sides.items()))
        if case_rows:
            cpp_total = sum(r["cpp_us"] for r in case_rows)
            rust_total = sum(r["rust_us"] for r in case_rows)
            ratios = [r["ratio"] for r in case_rows if r["cpp_us"] > 0]
            largest = max(case_rows, key=lambda r: r["json_bytes"])
            summary = dict(
                case=case,
                records=timed,
                errors=errors,
                error_sides=error_text,
                unloadable=unloadable,
                mismatches=mismatches,
                cpp_ms=round(cpp_total / 1e3, 3),
                rust_ms=round(rust_total / 1e3, 3),
                ratio_total=round(rust_total / cpp_total, 2) if cpp_total else float("nan"),
                ratio_median=round(statistics.median(ratios), 2) if ratios else float("nan"),
                ratio_min=round(min(ratios), 2) if ratios else float("nan"),
                ratio_max=round(max(ratios), 2) if ratios else float("nan"),
                largest_cpp_us=largest["cpp_us"],
                largest_rust_us=largest["rust_us"],
                largest_ratio=largest["ratio"],
            )
        else:
            summary = dict(
                case=case,
                records=0,
                errors=errors,
                error_sides=error_text,
                unloadable=unloadable,
                mismatches=0,
            )
        summaries.append(summary)
        for line in one_sided:
            print(f"  !! {case}: {line}", flush=True)
        print(
            f"{case:32s} n={summary['records']:4d} err={errors:3d}({error_text}) "
            f"unload={unloadable:2d} mismatch={mismatches:2d} "
            f"cpp={summary.get('cpp_ms', 0):9.3f}ms rust={summary.get('rust_ms', 0):9.3f}ms "
            f"ratio(total)={summary.get('ratio_total', float('nan')):5.2f} "
            f"ratio(median)={summary.get('ratio_median', float('nan')):5.2f} "
            f"[{summary.get('ratio_min', float('nan')):.2f}, "
            f"{summary.get('ratio_max', float('nan')):.2f}] "
            f"largest={summary.get('largest_ratio', float('nan'))}",
            flush=True,
        )

    out = Path(args.out)
    if rows:
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with out.with_suffix(".summary.csv").open("w", newline="") as f:
        keys = max((list(s.keys()) for s in summaries), key=len)
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    cpp_all = sum(r["cpp_us"] for r in rows)
    rust_all = sum(r["rust_us"] for r in rows)
    ratios = [s["ratio_total"] for s in summaries if s.get("records")]
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else float("nan")
    print(
        f"\nALL: {len(rows)} records timed, cpp={cpp_all / 1e3:.1f}ms rust={rust_all / 1e3:.1f}ms "
        f"ratio(total)={rust_all / cpp_all if cpp_all else float('nan'):.2f} "
        f"geomean(per-case ratio_total)={geo:.2f} "
        f"mismatches={sum(s['mismatches'] for s in summaries)}"
    )
    print(f"rows -> {out}\nsummary -> {out.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()
