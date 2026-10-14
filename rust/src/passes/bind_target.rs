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

//! `tirx.BindTarget` (`src/tirx/transform/bind_target.cc`): bind a host or device target to every
//! function according to where it is called from, duplicating functions called from both.

use std::collections::{HashMap, HashSet};

use tvm_ffi::{
    dispatch, structural_mutate, structural_walk, Any, AnyView, Array, ContextPolicy, Map, Mutator,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String, StructuralView, VisitContext,
    VisitInterrupt, WalkOrder, WalkResult, WalkWithContextPolicy,
};

use super::util::{dict_without, get_attr, has_nonzero_attr, module_pass, with_attr, with_attrs};
use crate::ir::{BaseFunc, Call, Expr, GlobalVar, IRModule, Op, RegionStmt, SeqStmt};
use crate::target::Target;
use crate::tirx::Function;
use crate::transform::Pass;

const TARGET: &str = "target";
const GLOBAL_SYMBOL: &str = "global_symbol";
const IS_HOST_FUNC: &str = "tirx.is_host_func";

struct LaunchOps {
    launch_thread: Op,
    device_entry: Op,
}

impl LaunchOps {
    fn new() -> Result<Self> {
        Ok(Self {
            launch_thread: Op::get("tirx.launch_thread")?,
            device_entry: Op::get("tirx.device_entry")?,
        })
    }

    fn opens_device_scope(&self, region: &RegionStmt) -> bool {
        region.op.same_as(&self.launch_thread) || region.op.same_as(&self.device_entry)
    }
}

/// `FunctionClassifierVisitor`: which global functions are called from host and from device code.
struct Classifier<'a> {
    ops: &'a LaunchOps,
    under_device_scope: bool,
    host_called: HashSet<ObjectIdentity>,
    device_called: HashSet<ObjectIdentity>,
}

#[dispatch(walk)]
impl Classifier<'_> {
    fn walk_call(&mut self, call: Call) -> WalkResult {
        if let Ok(callee) = call.op.clone().try_cast::<GlobalVar>() {
            let callers = if self.under_device_scope {
                &mut self.device_called
            } else {
                &mut self.host_called
            };
            callers.insert(ObjectIdentity::of(&callee));
        }
        WalkResult::Advance
    }
}

/// Calls in the body of a launched thread or device entry are device calls; its arguments are
/// still evaluated on the host.
struct DeviceScope;

impl<'a> ContextPolicy<Classifier<'a>> for DeviceScope {
    fn default_visit(
        &self,
        value: &StructuralView,
        ctx: &mut VisitContext<'_, Classifier<'a>>,
    ) -> Result<Option<VisitInterrupt>> {
        match value.cast::<RegionStmt>() {
            Some(region) if ctx.state().ops.opens_device_scope(&region) => {
                if let Some(interrupt) = ctx.visit(&region.args)? {
                    return Ok(Some(interrupt));
                }
                let outer = std::mem::replace(&mut ctx.state_mut().under_device_scope, true);
                let result = ctx.visit(&region.body);
                ctx.state_mut().under_device_scope = outer;
                result
            }
            _ => ctx.visit_children(),
        }
    }
}

/// `CallSubstitutor`: redirect host-side calls to the host duplicates.
struct CallSubstitutor<'a> {
    ops: &'a LaunchOps,
    under_device_scope: bool,
    replacements: &'a HashMap<ObjectIdentity, GlobalVar>,
}

#[dispatch(mutate)]
impl CallSubstitutor<'_> {
    fn mutate_call(&mut self, call: Call, ctx: &mut Mutator) -> Result<Any> {
        let call: Call = ctx.default_mutate(self, &call)?.try_into()?;
        if !self.under_device_scope {
            if let Ok(old) = call.op.clone().try_cast::<GlobalVar>() {
                if let Some(new) = self.replacements.get(&ObjectIdentity::of(&old)) {
                    return Ok(Any::from(Call::new(
                        call.loc.clone(),
                        call.ty.clone(),
                        new.clone().into(),
                        call.args.clone(),
                        call.attrs.clone(),
                        call.ty_args.clone(),
                    )));
                }
            }
        }
        Ok(Any::from(call))
    }

    fn mutate_region(&mut self, region: RegionStmt, ctx: &mut Mutator) -> Result<Any> {
        if !self.ops.opens_device_scope(&region) {
            return ctx.default_mutate(self, &region);
        }
        let args: Array<Expr> = ctx.mutate(self, &region.args)?.try_into()?;
        let previous = std::mem::replace(&mut self.under_device_scope, true);
        let body = ctx.mutate(self, &region.body);
        self.under_device_scope = previous;
        let body: SeqStmt = body?.try_into()?;
        Ok(Any::from(RegionStmt::new(
            region.loc.clone(),
            region.op.clone(),
            args,
            region.body_params.clone(),
            region.attrs.clone(),
            body,
            region.result_vars.clone(),
        )))
    }
}

fn is_externally_exposed(func: &BaseFunc) -> Result<bool> {
    Ok(get_attr(&func.attrs, GLOBAL_SYMBOL)?.map(|s| String::try_from(s).is_ok()).unwrap_or(false))
}

fn host_of(target: &Target) -> Result<Option<Target>> {
    target.host()?.map(|host| host.try_cast::<Target>()).transpose()
}

fn with_host(target: &Target, host: Option<&Target>) -> Result<Target> {
    let host = host.map(AnyView::from).unwrap_or_default();
    tvm_ffi::cached_global_func!("target.WithHost")
        .call_packed(&[AnyView::from(target), host])?
        .try_into()
}

/// `Target::WithoutHost`; `target.WithHost(target, None)` copies the node with the host cleared.
fn without_host(target: &Target) -> Result<Target> {
    if host_of(target)?.is_none() {
        return Ok(target.clone());
    }
    with_host(target, None)
}

fn new_target(name: &str) -> Result<Target> {
    tvm_ffi::cached_global_func!("target.Target").call_tuple((String::from(name),))?.try_into()
}

/// The module being rebuilt: an ordered function table, as `IRModuleNode::Update`/`Add` keep it.
struct ModuleState {
    functions: Vec<(GlobalVar, BaseFunc)>,
    global_var_map: Vec<(String, GlobalVar)>,
}

impl ModuleState {
    fn update(&mut self, gvar: &GlobalVar, func: BaseFunc) {
        match self.functions.iter_mut().find(|(key, _)| key.same_as(gvar)) {
            Some(entry) => entry.1 = func,
            None => self.add(gvar.clone(), func),
        }
    }

    fn add(&mut self, gvar: GlobalVar, func: BaseFunc) {
        if !self.global_var_map.iter().any(|(name, _)| name.as_str() == gvar.name_hint.as_str()) {
            self.global_var_map.push((gvar.name_hint.clone(), gvar.clone()));
        }
        self.functions.push((gvar, func));
    }
}

fn bind(module: IRModule, target: &Target) -> Result<IRModule> {
    let target_host = match host_of(target)? {
        Some(host) => host,
        None => new_target("llvm")?,
    };
    let target_without_host = without_host(target)?;
    // C++ Step 3 iterates `mod->functions`, which `mod.CopyOnWrite()` aliases to the module
    // being rebuilt only when the input is uniquely owned; with a shared input it re-reads the
    // original functions and drops the targets Step 2 bound. A pass always receives a uniquely
    // owned module in the pipeline, but the reference count seen across the FFI boundary is not
    // the one C++ sees, so this port always follows the uniquely owned behavior.

    // Step 1: who calls whom, from the bodies of the externally exposed functions.
    let ops = LaunchOps::new()?;
    let classifier = Classifier {
        ops: &ops,
        under_device_scope: false,
        host_called: HashSet::new(),
        device_called: HashSet::new(),
    };
    let mut walk = WalkWithContextPolicy::new(classifier, DeviceScope);
    for (_gvar, func) in module.functions.iter() {
        if is_externally_exposed(&func)? {
            if let Ok(function) = func.clone().try_cast::<Function>() {
                structural_walk(&function.body, &mut walk, WalkOrder::PreOrder)?;
            }
        }
    }
    let classifier = walk.into_state();

    // Step 2: bind a target to every function.
    let mut state = ModuleState {
        functions: module.functions.iter().collect(),
        global_var_map: module.global_var_map.iter().collect(),
    };
    let names = UniqueNames::new(state.functions.iter().map(|(gvar, _)| gvar.name_hint.clone()))?;
    let mut replacements: HashMap<ObjectIdentity, GlobalVar> = HashMap::new();
    let target_any = Any::from(target.clone());
    for (gvar, func) in module.functions.iter() {
        let Ok(function) = func.clone().try_cast::<Function>() else { continue };
        let exposed = is_externally_exposed(&func)?;
        if let Some(func_target) = get_attr(&function.attrs, TARGET)? {
            let func_target: Target = func_target.try_into()?;
            if let Some(host) = host_of(target)? {
                if host_of(&func_target)?.is_none() && exposed {
                    let bound = with_host(&func_target, Some(&host))?;
                    state.update(&gvar, with_attr(&function, TARGET, Any::from(bound)).into());
                }
            }
            continue;
        }
        if has_nonzero_attr(&function.attrs, IS_HOST_FUNC)? {
            let host = with_host(&target_host, Some(&target_host))?;
            let bound = with_attr(&function, TARGET, Any::from(host));
            let attrs = dict_without(&bound.attrs, IS_HOST_FUNC);
            state.update(&gvar, with_attrs(&bound, attrs).into());
            continue;
        }
        if exposed {
            state.update(&gvar, with_attr(&function, TARGET, target_any.clone()).into());
            continue;
        }
        let id = ObjectIdentity::of(&gvar);
        let (by_host, by_device) =
            (classifier.host_called.contains(&id), classifier.device_called.contains(&id));
        if by_host && by_device {
            let host_func: Function = tvm_ffi::cached_global_func!("tirx.RenewDef")
                .call_tuple((function.clone(),))?
                .try_into()?;
            state.update(
                &gvar,
                with_attr(&function, TARGET, Any::from(target_without_host.clone())).into(),
            );
            let host_func = with_attr(&host_func, TARGET, Any::from(target_host.clone()));
            let name = names.fresh(&format!("{}_host", gvar.name_hint.as_str()))?;
            let host_gvar: GlobalVar =
                tvm_ffi::cached_global_func!("ir.GlobalVar").call_tuple((name,))?.try_into()?;
            state.add(host_gvar.clone(), host_func.into());
            replacements.insert(id, host_gvar);
        } else if by_host {
            state
                .update(&gvar, with_attr(&function, TARGET, Any::from(target_host.clone())).into());
        } else {
            // Called from device only, or not called at all: both bind the device target.
            state.update(
                &gvar,
                with_attr(&function, TARGET, Any::from(target_without_host.clone())).into(),
            );
        }
    }

    // Step 3: point host-side calls in externally exposed functions at the host duplicates.
    if !replacements.is_empty() {
        for (gvar, func) in state.functions.clone() {
            let Ok(function) = func.clone().try_cast::<Function>() else { continue };
            if !is_externally_exposed(&func)? {
                continue;
            }
            let mut substitutor = CallSubstitutor {
                ops: &ops,
                under_device_scope: false,
                replacements: &replacements,
            };
            let body: Option<SeqStmt> = match function.body.clone() {
                Some(body) => Some(structural_mutate(body, &mut substitutor)?.try_into()?),
                None => None,
            };
            let unchanged = match (&body, &function.body) {
                (Some(new), Some(old)) => new.same_as(old),
                (None, None) => true,
                _ => false,
            };
            let updated =
                if unchanged { function } else { super::util::with_body(&function, body) };
            state.update(&gvar, updated.into());
        }
    }

    Ok(IRModule::new(
        state.functions.into_iter().collect::<Map<GlobalVar, BaseFunc>>(),
        module.source_map.clone(),
        module.attrs.clone(),
        module.global_infos.clone(),
        state.global_var_map.into_iter().collect::<Map<String, GlobalVar>>(),
    ))
}

/// `UniqueNameSupply(begin, end, f)`: a supply with every existing name reserved.
struct UniqueNames {
    supply: crate::ir::UniqueNameSupply,
}

impl UniqueNames {
    fn new(names: impl Iterator<Item = String>) -> Result<Self> {
        let supply: crate::ir::UniqueNameSupply =
            tvm_ffi::cached_global_func!("ir.UniqueNameSupply")
                .call_tuple((String::from(""),))?
                .try_into()?;
        for name in names {
            tvm_ffi::cached_global_func!("ir.UniqueNameSupply_ReserveName").call_tuple((
                supply.clone(),
                name,
                false,
            ))?;
        }
        Ok(Self { supply })
    }

    /// `FreshName(name, add_prefix=false)`.
    fn fresh(&self, name: &str) -> Result<String> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupply_FreshName")
            .call_tuple((self.supply.clone(), String::from(name), false, true))?
            .try_into()
    }
}

pub fn bind_target(target: Target) -> Result<Pass> {
    module_pass("tirx.BindTarget", 0, move |module, _context| bind(module, &target))
}
