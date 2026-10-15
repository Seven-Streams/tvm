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

//! `tirx.SkipAssert` (`src/tirx/transform/skip_assert.cc`): replace every `AssertStmt` with
//! `Evaluate(0)`. Expressions are not descended into.

use tvm_ffi::{dispatch, structural_mutate, Result, Unchanged};

use super::util::{function_pass};
use crate::ir::{AssertStmt, Evaluate, Expr, SeqStmt};
use crate::transform::Pass;

struct AssertSkipper;

#[dispatch(mutate)]
impl AssertSkipper {
    fn mutate_assert(&mut self, _assert: AssertStmt) -> Result<Evaluate> {
        Evaluate::new(0)
    }

    fn mutate_expr(&mut self, _expr: Expr) -> Unchanged {
        Unchanged
    }
}

pub fn skip_assert() -> Result<Pass> {
    function_pass("tirx.SkipAssert", 0, |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let body: SeqStmt = structural_mutate(body, &mut AssertSkipper)?.try_into()?;
        Ok(Some(func.with_body(Some(body))))
    })
}
