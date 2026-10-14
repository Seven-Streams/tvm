/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! `tirx.DecorateDeviceScope` (`src/tirx/transform/decorate_device_scope.cc`): wrap the body in a
//! `tirx.device_scope` region.

use tvm_ffi::{Any, Array, Map, Result, String};

use super::util::{function_pass, seq_stmt_from, unknown_loc};
use crate::ir::{DictAttrs, Expr, Op, RegionStmt, Var};
use crate::transform::Pass;

pub fn decorate_device_scope() -> Result<Pass> {
    let device_scope = Op::get("tirx.device_scope")?;
    function_pass("tirx.DecorateDeviceScope", 0, move |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let region = RegionStmt::new(
            device_scope.clone(),
            Array::<Expr>::new(vec![]),
            Array::<Var>::new(vec![]),
            DictAttrs::new(Map::<String, Any>::new()),
            body,
            Array::<Var>::new(vec![]),
            unknown_loc()?,
        )?;
        Ok(Some(func.with_body(Some(seq_stmt_from(region.into())?))))
    })
}
