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

//! Compile the generated visitor exactly as a downstream crate would.  This
//! catches runtime-crate path and attribute-projection bugs that token-string
//! tests in the proc-macro crate cannot detect.

use tvm_tirx::{dispatch, For, VisitCtx, VisitDispatch, WalkResult};

#[derive(Default)]
struct ExternalCounter {
    loops: usize,
}

#[dispatch(visit)]
impl ExternalCounter {
    // `inline` is valid on this method, but would be invalid if cfg_attr were
    // copied blindly onto the generated dispatch expression.
    #[cfg_attr(all(), inline)]
    fn visit_for(&mut self, _op: For, _ctx: &mut VisitCtx<Self>) -> WalkResult {
        self.loops += 1;
        WalkResult::Advance
    }
}

struct ExplicitRuntimePath;

#[dispatch(visit, runtime = tvm_tirx)]
impl ExplicitRuntimePath {
    fn visit_for(&mut self, _op: For, _ctx: &mut VisitCtx<Self>) -> WalkResult {
        WalkResult::Skip
    }
}

fn assert_generated_dispatch<T: VisitDispatch>() {}

#[test]
fn dispatch_is_a_downstream_api() {
    assert_generated_dispatch::<ExternalCounter>();
    assert_generated_dispatch::<ExplicitRuntimePath>();
    assert_eq!(ExternalCounter::default().loops, 0);
}
