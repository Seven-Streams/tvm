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
"""tirx_ext: Rust-implemented tirx IR analysis tools.

>>> import tvm
>>> from tvm import tirx
>>> import tirx_ext
>>> stats = tirx_ext.count_loops(stmt)   # a tirx Stmt built in Python
>>> stats["loops"], stats["total_iters"]
"""

from ._ffi_api import (
    break_for_bodies,
    break_innermost_for_bodies,
    count_adds,
    count_loops,
)

__all__ = [
    "break_for_bodies",
    "break_innermost_for_bodies",
    "count_adds",
    "count_loops",
]
