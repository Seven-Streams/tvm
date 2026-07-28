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

//! This must compile as a downstream crate so the generated public paths are
//! checked outside `tvm_tirx` itself.

use tvm_tirx::{dispatch, ForNode, VisitCtx, VisitDispatch, WalkResult};

struct ExternalCounter {
    loops: usize,
}

#[dispatch(visit)]
impl ExternalCounter {
    #[cfg(any(unix, windows))]
    #[cfg_attr(all(), inline)]
    fn visit_for(&mut self, _op: &ForNode, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        self.loops += 1;
        WalkResult::Advance
    }
}

fn assert_visit_dispatch<T: VisitDispatch>() {}

const _: fn() = assert_visit_dispatch::<ExternalCounter>;

struct CfgAttrCounter;

#[dispatch(visit)]
impl CfgAttrCounter {
    #[cfg_attr(all(), cfg(any()))]
    fn visit_disabled(&mut self, _op: &ForNode, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        WalkResult::Advance
    }

    fn visit_for(&mut self, _op: &ForNode, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        WalkResult::Advance
    }
}

const _: fn() = assert_visit_dispatch::<CfgAttrCounter>;

struct DisabledCounter;
const _: usize = std::mem::size_of::<DisabledCounter>();

#[dispatch(visit)]
#[cfg(any())]
impl DisabledCounter {
    fn visit_for(&mut self, _op: &ForNode, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        WalkResult::Advance
    }
}

struct CfgAttrDisabledCounter;
const _: usize = std::mem::size_of::<CfgAttrDisabledCounter>();

#[dispatch(visit)]
#[cfg_attr(all(), cfg(any()))]
impl CfgAttrDisabledCounter {
    fn visit_for(&mut self, _op: &ForNode, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        WalkResult::Advance
    }
}
