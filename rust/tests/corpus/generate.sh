#!/usr/bin/env bash
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

# Record the input of every tirx pass the selected TVM tests run, for the pass parity tests.
#
# Usage (from the repository root, with the tvm and tvm_ffi Python packages importable):
#   rust/tests/corpus/generate.sh [output directory, default rust/tests/corpus/data]
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
out=${1:-$root/rust/tests/corpus/data}
python=${PYTHON:-python}
mkdir -p "$out"
export PYTHONPATH="$root/rust/tests/corpus${PYTHONPATH:+:$PYTHONPATH}"

run() {  # run <records per pass and test file> <pytest paths...>
  local limit=$1
  shift
  CORPUS_DIR=$out CORPUS_LIMIT=$limit "$python" -m pytest -p capture_plugin "$@" \
    -q --tb=no -p no:cacheprovider | tail -1 || true
}

# Unit tests of the passes: small inputs that cover their edge cases.
run 40 "$root/tests/python/tirx/transform"
# The CUDA compile pipeline on real kernels: elementwise ops, and tcgen05 GEMMs on sm_100.
run 6 "$root/tests/python/tirx/tile/cuda/test_ewise.py" \
  "$root/tests/python/tirx/tile/cuda/test_tcgen05_mma.py"

echo "corpus written to $out"
