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

//! `tirx.AnnotateEntryFunc` and `tirx.Filter` (`src/tirx/transform/function_utils.cc`).

use tvm_ffi::{Function as FfiFunction, ObjectRefCast, Result, String};

use super::util::{
    function_pass, get_attr, has_nonzero_attr, module_pass, module_with_function, with_attr,
};
use crate::ir::{BaseFunc, GlobalVar};
use crate::tirx::Function;
use crate::transform::Pass;

const IS_ENTRY_FUNC: &str = "tirx.is_entry_func";
const GLOBAL_SYMBOL: &str = "global_symbol";

/// Mark the module's entry function: its only function, or else its only externally visible one.
pub fn annotate_entry_func() -> Result<Pass> {
    module_pass("tirx.AnnotateEntryFunc", 0, |module, _context| {
        let entries: Vec<(GlobalVar, BaseFunc)> = module.functions.iter().collect();
        if entries.len() == 1 {
            let (gvar, base_func) = &entries[0];
            if !has_nonzero_attr(&base_func.attrs, IS_ENTRY_FUNC)? {
                if let Ok(func) = base_func.clone().try_cast::<Function>() {
                    let marked = with_attr(&func, IS_ENTRY_FUNC, true);
                    return Ok(module_with_function(&module, gvar, marked));
                }
            }
            return Ok(module);
        }
        let mut external: Vec<(GlobalVar, Function)> = Vec::new();
        let mut external_non_functions = false;
        for (gvar, base_func) in &entries {
            let symbol = get_attr(&base_func.attrs, GLOBAL_SYMBOL)?;
            if symbol.map(|symbol| String::try_from(symbol).is_ok()).unwrap_or(false) {
                match base_func.clone().try_cast::<Function>() {
                    Ok(func) => external.push((gvar.clone(), func)),
                    Err(_) => external_non_functions = true,
                }
            }
        }
        if external.len() == 1 && !external_non_functions {
            let (gvar, func) = &external[0];
            let marked = with_attr(func, IS_ENTRY_FUNC, true);
            return Ok(module_with_function(&module, gvar, marked));
        }
        Ok(module)
    })
}

/// Keep the functions for which `condition(func)` holds.
pub fn filter(condition: FfiFunction) -> Result<Pass> {
    function_pass("tirx.Filter", 0, move |func, _module, _context| {
        let keep: bool = condition.call_tuple((func.clone(),))?.try_into()?;
        Ok(keep.then_some(func))
    })
}
