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

//! `tirx.VerifyMemory` (`src/tirx/analysis/verify_memory.cc`): on a GPU target, reject a function
//! that accesses a tensor from its arguments outside any thread environment.

use std::collections::HashMap;

use tvm_ffi::{
    dispatch, structural_walk, ContextPolicy, Error, ObjectIdentity, ObjectRefCast, ObjectRefCore,
    Result, StructuralView, VisitContext, VisitInterrupt, WalkOrder, WalkResult,
    WalkWithContextPolicy, RUNTIME_ERROR, TYPE_ERROR,
};

use super::util::{get_attr, module_pass};
use crate::ir::{Bind, Call, CallObj, Expr, Op, RegionStmt, StringImm, TensorLoad, TensorStore, Var};
use crate::target::Target;
use crate::tirx::{Function, TensorTypeObj};
use crate::transform::Pass;

// DLDeviceType values of the GPU devices the C++ pass checks.
const GPU_DEVICES: [i64; 5] = [
    2,  /* CUDA */
    4,  /* OpenCL */
    7,  /* Vulkan */
    8,  /* Metal */
    10, /* ROCm */
];

struct Ops {
    device_scope: Op,
    launch_thread: Op,
    masked_load: Op,
    masked_store: Op,
    abi_field_get: Op,
}

struct MemoryAccessVerifier<'a> {
    func: &'a Function,
    ops: &'a Ops,
    in_thread_env: bool,
    errors: Vec<std::string::String>,
    defs: HashMap<ObjectIdentity, Expr>,
}

/// `as_or_throw<TensorVar>().var()`: the expression must be a Var of tensor type.
fn tensor_var(expr: &Expr) -> Result<Var> {
    match expr.clone().try_cast::<Var>() {
        Ok(var) if var.ty.as_node::<TensorTypeObj>().is_some() => Ok(var),
        _ => Err(Error::new(TYPE_ERROR, "expected a tensor variable", "")),
    }
}

impl MemoryAccessVerifier<'_> {
    fn is_from_function_args(&self, var: &Var) -> bool {
        let params: Vec<Var> = self.func.params.iter().collect();
        if params
            .iter()
            .any(|param| param.ty.as_node::<TensorTypeObj>().is_some() && param.same_as(var))
        {
            return true;
        }
        let mut current = var.clone();
        loop {
            if !params.is_empty() && params[0].same_as(&current) {
                return true;
            }
            let Some(value) = self.defs.get(&ObjectIdentity::of(&current)) else { return false };
            let Some(call) = value.as_node::<CallObj>() else { return false };
            if !call.op.same_as(&self.ops.abi_field_get) {
                return false;
            }
            match call.args.get(0).ok().and_then(|arg| arg.try_cast::<Var>().ok()) {
                Some(next) => current = next,
                // C++ `as<VarNode>()` yields null, which matches neither a parameter nor a def.
                None => return false,
            }
        }
    }

    fn handle_access(&mut self, var: &Var) {
        if self.in_thread_env || !self.is_from_function_args(var) {
            return;
        }
        self.errors.push(format!(
            "Variable `{}` is directly accessed by host memory (it is not contained in a thread \
             environment or in the function arguments.",
            var.name.as_str()
        ));
    }
}

#[dispatch(walk)]
impl MemoryAccessVerifier<'_> {
    fn walk_bind(&mut self, bind: Bind) -> WalkResult {
        self.defs.insert(ObjectIdentity::of(&bind.var), bind.value.clone());
        WalkResult::Advance
    }

    fn walk_load(&mut self, load: TensorLoad) -> Result<WalkResult> {
        self.handle_access(&tensor_var(&load.source)?);
        Ok(WalkResult::Advance)
    }

    fn walk_store(&mut self, store: TensorStore) -> Result<WalkResult> {
        self.handle_access(&tensor_var(&store.dest)?);
        Ok(WalkResult::Advance)
    }

    fn walk_call(&mut self, call: Call) -> Result<WalkResult> {
        let masked =
            call.op.same_as(&self.ops.masked_load) || call.op.same_as(&self.ops.masked_store);
        if masked && !call.args.is_empty() {
            let var: Var = call.args.get(0)?.try_cast()?;
            self.handle_access(&var);
        }
        Ok(WalkResult::Advance)
    }
}

/// The body of a device scope or of a launched (non-virtual) thread is inside a thread
/// environment; its arguments and attributes are not.
struct ThreadScope;

impl<'a> ContextPolicy<MemoryAccessVerifier<'a>> for ThreadScope {
    fn default_visit(
        &self,
        value: &StructuralView,
        ctx: &mut VisitContext<'_, MemoryAccessVerifier<'a>>,
    ) -> Result<Option<VisitInterrupt>> {
        let Some(region) = value.cast::<RegionStmt>() else { return ctx.visit_children() };
        if ctx.state().in_thread_env || !ctx.state().opens_thread_env(&region)? {
            return ctx.visit_children();
        }
        if let Some(interrupt) = ctx.visit(&region.args)? {
            return Ok(Some(interrupt));
        }
        if let Some(interrupt) = ctx.visit(&region.attrs)? {
            return Ok(Some(interrupt));
        }
        ctx.state_mut().in_thread_env = true;
        let result = ctx.visit(&region.body);
        ctx.state_mut().in_thread_env = false;
        result
    }
}

impl MemoryAccessVerifier<'_> {
    fn opens_thread_env(&self, region: &RegionStmt) -> Result<bool> {
        if region.op.same_as(&self.ops.device_scope) {
            return Ok(true);
        }
        if !region.op.same_as(&self.ops.launch_thread) {
            return Ok(false);
        }
        let tag: StringImm = region.args.get(0)?.try_cast()?;
        Ok(!tag.value.as_str().starts_with("vthread"))
    }
}

fn device_type(target: &Target) -> Result<i64> {
    tvm_ffi::cached_global_func!("target.TargetGetDeviceType")
        .call_tuple((target.clone(),))?
        .try_into()
}

fn verify(func: &Function, ops: &Ops) -> Result<Vec<std::string::String>> {
    let Some(target) = get_attr(&func.attrs, "target")? else { return Ok(vec![]) };
    let target: Target = target.try_into()?;
    let calling_conv = match get_attr(&func.attrs, "calling_conv")? {
        Some(value) => i64::try_from(value)?,
        None => 0,
    };
    if calling_conv != 0 {
        return Ok(vec![]);
    }
    if !GPU_DEVICES.contains(&device_type(&target)?) {
        return Ok(vec![]);
    }
    let verifier = MemoryAccessVerifier {
        func,
        ops,
        in_thread_env: false,
        errors: vec![],
        defs: HashMap::new(),
    };
    let mut walk = WalkWithContextPolicy::new(verifier, ThreadScope);
    structural_walk(&func.body, &mut walk, WalkOrder::PreOrder)?;
    Ok(walk.into_state().errors)
}

pub fn verify_memory() -> Result<Pass> {
    let ops = Ops {
        device_scope: Op::get("tirx.device_scope")?,
        launch_thread: Op::get("tirx.launch_thread")?,
        masked_load: Op::get("tirx.masked_load")?,
        masked_store: Op::get("tirx.masked_store")?,
        abi_field_get: Op::get("tirx.abi_field_get")?,
    };
    module_pass("tirx.VerifyMemory", 0, move |module, _context| {
        for (_gvar, func) in module.functions.iter() {
            let Ok(func) = func.try_cast::<Function>() else { continue };
            let errors = verify(&func, &ops)?;
            if !errors.is_empty() {
                let listed: std::string::String =
                    errors.iter().map(|e| format!("    {e}\n")).collect();
                return Err(Error::new(
                    RUNTIME_ERROR,
                    &format!("Memory verification failed with the following errors:\n{listed}  Did you forget to bind?"),
                    "",
                ));
            }
        }
        Ok(module)
    })
}

#[allow(dead_code)]
fn unused(_: String) {}
