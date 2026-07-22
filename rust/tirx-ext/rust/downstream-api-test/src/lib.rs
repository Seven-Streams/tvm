// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Compile-only fixture for a pass crate that renames its `tvm-tirx`
//! dependency.  Keeping this as a real workspace member catches packaging and
//! visibility regressions that a proc-macro token test cannot see.

use tirx_runtime::{dispatch, structural_visit, For, Stmt, VisitCtx, WalkResult};

#[derive(Default)]
pub struct Counter {
    pub loops: usize,
}

#[dispatch(visit, runtime = tirx_runtime)]
impl Counter {
    #[cfg_attr(all(), inline)]
    fn visit_for(&mut self, _op: For, _ctx: &mut VisitCtx<Self>) -> WalkResult {
        self.loops += 1;
        WalkResult::Advance
    }
}

pub fn count(root: &Stmt) -> Counter {
    let mut counter = Counter::default();
    assert!(
        structural_visit(root, &mut counter)
            .expect("structural visit failed")
            .is_none(),
        "structural visit stopped before the whole tree was analyzed"
    );
    counter
}
