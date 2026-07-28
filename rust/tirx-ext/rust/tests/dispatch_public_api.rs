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

use tvm_tirx::{
    dispatch, DefRegionKind, ForNode, StmtNode, VisitCtx, VisitDispatch, VisitValue, WalkResult,
};

#[derive(Default)]
struct ExternalCounter {
    loops: usize,
}

#[dispatch(visit)]
impl ExternalCounter {
    #[cfg(any(unix, windows))]
    fn visit_for(&mut self, _op: &ForNode, ctx: &mut VisitCtx<'_>) -> WalkResult {
        let _: DefRegionKind = ctx.def_region_kind();
        self.loops += 1;
        WalkResult::Advance
    }

    fn visit_stmt(&mut self, _op: &StmtNode, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        WalkResult::Advance
    }

    fn visit_integer(
        &mut self,
        _value: i64,
        _ctx: &mut VisitCtx<'_>,
    ) -> tvm_ffi::Result<WalkResult> {
        Ok(WalkResult::Advance)
    }

    fn visit_any(&mut self, _value: &VisitValue, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        WalkResult::Advance
    }
}

fn assert_visit_dispatch<T: VisitDispatch>() {}

#[test]
fn generated_dispatch_is_public() {
    assert_visit_dispatch::<ExternalCounter>();
}
