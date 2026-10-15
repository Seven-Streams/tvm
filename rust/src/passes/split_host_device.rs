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

//! `tirx.SplitHostDevice` (`src/tirx/transform/split_host_device.cc`): mark the device regions of
//! each host function with the device target, move every marked region into a kernel function
//! of its own, and lower the calls to kernels into launches (`call_ffi_kernel`) or external calls.

use std::collections::{HashMap, HashSet};

use tvm_ffi::{
    bail, dispatch, structural_map, structural_mutate, structural_walk, Any, Array, ContextPolicy,
    Error, Map, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String,
    StructuralView, Unchanged, UnchangedOr, VisitContext, VisitInterrupt, WalkOrder, WalkResult,
    WalkWithContextPolicy, INTERNAL_ERROR, TYPE_ERROR, VALUE_ERROR,
};

use super::analyzer::is_const_int;
use super::bind_target::{host_of, without_host, UniqueNames};
use super::convert_ssa::convert_ssa_module;
use super::util::{MutatorExt, dict_with, get_attr, module_pass, with_attr};
use crate::ir::{
    AssertStmt, Attrs, BaseFunc, Bind, Call, CallObj, DictAttrs, Evaluate, Expr, GlobalVar,
    IRModule, IntImmObj, PrimType, PrimTypeObj, PtrTypeObj, RegionStmt, Return, SeqStmt, Stmt,
    StringImm, TupleType, TupleTypeObj, Type, Var,
};
use crate::ops::{int32, int32_dtype, int_imm, prim_type, undefined_vars, unknown_loc, Ex, PrimOps};
use crate::runtime::MemoryScope;
use crate::sym::Analyzer;
use crate::target::Target;
use crate::tirx::op::{
    ALLOC_TENSOR, CALL_EXTERN, CALL_FFI_KERNEL, DECL_TENSOR, DEVICE_SCOPE, LAUNCH_THREAD,
};
use crate::tirx::{CallFFIKernelAttr, Function, TensorType, TensorTypeObj};
use crate::transform::Pass;

const TARGET: &str = "target";
const GLOBAL_SYMBOL: &str = "global_symbol";
const CALLING_CONV: &str = "calling_conv";
const NUM_INPUTS: &str = "num_inputs";
const SCRIPT_NAMESPACE: &str = "script.namespace";
const S_TIR: &str = "s_tir";
const BACKEND_CONFIG: &str = "backend_config";
const NO_ALIAS: &str = "tirx.noalias";
const IS_GLOBAL_FUNC: &str = "tirx.is_global_func";
const KERNEL_LAUNCH_PARAMS: &str = "tirx.kernel_launch_params";
const LAUNCH_FIELDS: &str = "cuda.launch_fields";
const KERNEL_ATTRS: &str = "cuda.kernel_attrs";
const SMEM_REQUIRED: &str = "cuda.smem_required";
const LAUNCH_DIMENSIONS: &str = "cuda.launch_dimensions";
/// `runtime::launch_param` tags (`src/runtime/metadata.h`).
const USE_DYN_SHARED_MEMORY: &str = "tirx.use_dyn_shared_memory";
const USE_PDL: &str = "tirx.use_programtic_dependent_launch";
const USE_COOPERATIVE_LAUNCH: &str = "tirx.use_cooperative_launch";
const USE_REQUIRED_BLOCK_DIMENSION: &str = "tirx.use_required_block_dimension";
/// `CallingConv::kDeviceKernelLaunch`.
const DEVICE_KERNEL_LAUNCH: i64 = 2;
/// The device types whose kernels can return an error code: `kDLCPU`, `kDLExtDev`, `kDLHexagon`.
const ERROR_PROPAGATING_DEVICES: [i64; 3] = [1, 12, 16];

fn has_attr(attrs: &DictAttrs, key: &str) -> Result<bool> {
    Ok(get_attr(attrs, key)?.is_some())
}

/// `func->GetAttr<T>(key)`: absent is `None`, a value of another type an error.
fn typed_attr<T: TryFrom<Any, Error = Error>>(attrs: &DictAttrs, key: &str) -> Result<Option<T>> {
    get_attr(attrs, key)?.map(T::try_from).transpose()
}

/// `func->GetAttr<Target>(kTarget).value()`.
fn function_target(func: &Function) -> Result<Target> {
    typed_attr(&func.attrs, TARGET)?
        .ok_or_else(|| error!(INTERNAL_ERROR, "Bad optional access: the function has no target"))
}

fn kind_name(target: &Target) -> Result<String> {
    target.kind()?.name()
}

/// `VoidType()`, the empty tuple type.
fn void_type() -> Result<Type> {
    Ok(TupleType::new([])?.into())
}

/// `IsVoidType(ty)`.
fn is_void_type(ty: &Type) -> bool {
    ty.as_node::<TupleTypeObj>().is_some_and(|tuple| tuple.fields.is_empty())
}

/// `PrimType::Void()`.
fn prim_void() -> Result<PrimType> {
    prim_type(tvm_ffi::DLDataType { code: 3, bits: 0, lanes: 0 })
}

fn call(ty: Type, op: impl Into<Expr>, args: Vec<Expr>, attrs: Option<Attrs>) -> Result<Call> {
    let (args, ty_args) = (Array::new(args), Array::new(vec![]));
    Call::new_full(Some(ty), op.into(), args, attrs, ty_args, unknown_loc()?)
}

/// `value` with each bound variable replaced by its value.
fn substitute(bindings: &HashMap<ObjectIdentity, Expr>, value: &Expr) -> Result<Expr> {
    structural_map(
        value.clone(),
        |var: Var| -> UnchangedOr<Expr> {
            match bindings.get(&ObjectIdentity::of(&var)) {
                Some(replacement) => UnchangedOr::changed(replacement.clone()),
                None => UnchangedOr::unchanged(),
            }
        },
        WalkOrder::PreOrder,
    )?
    .try_into()
}

fn is_prim_expr(expr: &Expr) -> bool {
    expr.ty.as_node::<PrimTypeObj>().is_some()
}

// ---------------------------------------------------------------------------------------------
// Device-region annotation

/// `DeviceRegionAnnotater`: a device scope without a target, and a launched thread outside any
/// device scope, run on the device target.
struct DeviceRegionAnnotater {
    device_target: Target,
}

#[dispatch(mutate)]
impl DeviceRegionAnnotater {
    fn mutate_expr(&mut self, _expr: Expr) -> Unchanged {
        Unchanged
    }

    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        if DEVICE_SCOPE.is(&op.op) {
            if has_attr(&op.attrs, TARGET)? {
                return Ok(op.into());
            }
            let attrs = dict_with(&op.attrs, TARGET, Any::from(self.device_target.clone()));
            let region = RegionStmt::new_full(
                op.op.clone(),
                op.args.clone(),
                op.body_params.clone(),
                attrs,
                op.body.clone(),
                op.result_vars.clone(),
                op.loc.clone(),
            )?;
            return Ok(region.into());
        }
        if LAUNCH_THREAD.is(&op.op) {
            let attrs: Map<String, Any> =
                [(String::from(TARGET), Any::from(self.device_target.clone()))]
                    .into_iter()
                    .collect();
            let region = RegionStmt::new(
                DEVICE_SCOPE.get()?,
                Array::new(vec![]),
                Array::new(vec![]),
                DictAttrs::new(attrs),
                op,
            )?;
            return Ok(region.into());
        }
        m.default_mutate(self, &op)
    }
}

/// `AnnotateDeviceRegionsForSplit`: only a function whose target has a host is annotated.
fn annotate_device_regions(func: Function) -> Result<Function> {
    let Some(target) = typed_attr::<Target>(&func.attrs, TARGET)? else {
        bail!(
            INTERNAL_ERROR,
            "Check failed: (opt_target) is false: SplitHostDevice: Require the target attribute"
        );
    };
    if host_of(&target)?.is_none() {
        return Ok(func);
    }
    let Some(body) = func.body.clone() else { return Ok(func) };
    let mut annotater = DeviceRegionAnnotater { device_target: without_host(&target)? };
    let annotated: SeqStmt = structural_mutate(body.clone(), &mut annotater)?.try_into()?;
    if annotated.same_as(&body) {
        return Ok(func);
    }
    Ok(func.with_body(Some(annotated)))
}

// ---------------------------------------------------------------------------------------------
// Host/device function extraction

/// `KernelBodyRewriter`: the body of a kernel, nested device scopes flattened and each captured
/// tensor replaced by the kernel's own tensor variable.
struct KernelBodyRewriter {
    tensors: HashMap<ObjectIdentity, Var>,
}

#[dispatch(mutate)]
impl KernelBodyRewriter {
    fn mutate_var(&mut self, var: Var) -> Var {
        self.tensors.get(&ObjectIdentity::of(&var)).cloned().unwrap_or(var)
    }

    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        if DEVICE_SCOPE.is(&op.op) {
            return m.mutate(self, &op.body);
        }
        m.default_mutate(self, &op)
    }
}

/// `HostDeviceSplitter`: each targeted device scope of one function becomes a kernel launch.
struct HostDeviceSplitter<'a> {
    names: &'a UniqueNames,
    kernel_name: std::string::String,
    cur_func: Function,
    device_funcs: Vec<(GlobalVar, Function)>,
}

#[dispatch(mutate)]
impl HostDeviceSplitter<'_> {
    fn mutate_expr(&mut self, _expr: Expr) -> Unchanged {
        Unchanged
    }

    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        if !DEVICE_SCOPE.is(&op.op) {
            return m.default_mutate(self, &op);
        }
        match get_attr(&op.attrs, TARGET)? {
            None => m.mutate(self, &op.body),
            Some(target) => {
                let target: Target = target.try_into()?;
                Ok(self.split_device_func(op.body.clone(), without_host(&target)?, &op)?.into())
            }
        }
    }
}

impl HostDeviceSplitter<'_> {
    /// The kernel parameters, in the order C++ sorts them: pointers and tensors first, then by
    /// name; for Trainium, in the order of the function's parameters. Ties keep the order of the
    /// free variables (C++ `std::sort` is not stable).
    fn kernel_params(&self, undefined: &[Var], device_target: &Target) -> Result<Vec<Var>> {
        let mut params = undefined.to_vec();
        if kind_name(device_target)?.as_str() != "trn" {
            let key = |var: &Var| {
                let is_handle = var.ty.as_node::<PtrTypeObj>().is_some()
                    || var.ty.as_node::<TensorTypeObj>().is_some();
                (!is_handle, var.name.as_str().to_string())
            };
            params.sort_by_key(key);
        } else {
            let mut order = HashMap::new();
            for (index, param) in self.cur_func.params.iter().enumerate() {
                if param.ty.as_node::<TensorTypeObj>().is_none() {
                    bail!(TYPE_ERROR, "Function parameter is not a tirx.TensorVar");
                }
                order.insert(ObjectIdentity::of(&param), index);
            }
            params.sort_by_key(|var| order.get(&ObjectIdentity::of(var)).copied().unwrap_or(0));
        }
        Ok(params)
    }

    fn split_device_func(
        &mut self,
        body: SeqStmt,
        device_target: Target,
        region: &RegionStmt,
    ) -> Result<Stmt> {
        let undefined: Vec<Var> =
            undefined_vars(&body.clone().into(), &Array::new(vec![]))?.iter().collect();
        let tensors: Vec<Var> = undefined
            .iter()
            .filter(|var| var.ty.as_node::<TensorTypeObj>().is_some())
            .cloned()
            .collect();
        let params = self.kernel_params(&undefined, &device_target)?;

        // A tensor variable is a compiler-side value: the kernel takes its data pointer and
        // declares a tensor of its own over it.
        let mut kernel_params = Vec::new();
        let mut call_args: Vec<Expr> = Vec::new();
        let mut data_params: HashMap<ObjectIdentity, Var> = HashMap::new();
        let mut rewriter = KernelBodyRewriter { tensors: HashMap::new() };
        for param in &params {
            match param.ty.clone().try_cast::<TensorType>() {
                Ok(ty) => {
                    let kernel_tensor =
                        Var::new_full(param.name.clone(), ty.clone(), param.loc.clone())?;
                    let data_ptr = ty.data_ptr_type()?;
                    let data_param = Var::new(format!("{}_ptr", param.name.as_str()), data_ptr)?;
                    let data = param.tensor_data()?;
                    kernel_params.push(data_param.clone());
                    call_args.push(data);
                    data_params.insert(ObjectIdentity::of(param), data_param);
                    rewriter.tensors.insert(ObjectIdentity::of(param), kernel_tensor);
                }
                Err(_) => {
                    kernel_params.push(param.clone());
                    call_args.push(param.clone().into());
                }
            }
        }
        let mut body: Stmt = structural_mutate(body, &mut rewriter)?.try_into()?;

        // Codegen for a CPU-like device (e.g. `ext_dev`) expects an int32 status code.
        let can_propagate_errors =
            ERROR_PROPAGATING_DEVICES.contains(&device_target.device_type()?);
        let int32_ty: Type = prim_type(int32_dtype())?.into();
        let success: Expr = int32(0)?.into();
        let kernel_ret_type = if can_propagate_errors {
            body = SeqStmt::new(vec![body, Return::new(success.clone())?.into()])?.into();
            int32_ty.clone()
        } else {
            void_type()?
        };

        for tensor in &tensors {
            let id = ObjectIdentity::of(tensor);
            let data_param = data_params.get(&id).cloned().ok_or_else(|| {
                error!(
                    INTERNAL_ERROR,
                    "Undefined tensor {} was not captured as a kernel parameter",
                    tensor.name.as_str()
                )
            })?;
            let kernel_tensor = rewriter.tensors[&id].clone();
            let ty = kernel_tensor.ty.clone();
            let declare = Call::new_full(
                Some(ty.clone()),
                DECL_TENSOR.get()?.into(),
                Array::new(vec![data_param.into()]),
                None,
                Array::new(vec![ty]),
                unknown_loc()?,
            )?;
            let bind: Stmt = Bind::new(kernel_tensor, declare)?.into();
            body = SeqStmt::new(vec![bind, body])?.into();
        }

        let void_attrs = DictAttrs::new(Map::new());
        let mut device_func = Function::new_full(
            Array::new(kernel_params),
            Some(SeqStmt::from_stmt(body)?),
            kernel_ret_type.clone(),
            void_attrs,
            unknown_loc()?,
        )?;
        device_func = with_attr(&device_func, TARGET, device_target.clone());
        device_func = with_attr(&device_func, NO_ALIAS, true);
        device_func = with_attr(&device_func, IS_GLOBAL_FUNC, true);
        let cur_attrs = self.cur_func.attrs.clone();
        if let Some(config) = get_attr(&region.attrs, BACKEND_CONFIG)? {
            device_func = with_attr(&device_func, BACKEND_CONFIG, config);
        } else if let Some(config) = typed_attr::<String>(&cur_attrs, BACKEND_CONFIG)? {
            device_func = with_attr(&device_func, BACKEND_CONFIG, config);
        }
        if let Some(namespace) = typed_attr::<String>(&cur_attrs, SCRIPT_NAMESPACE)? {
            device_func = with_attr(&device_func, SCRIPT_NAMESPACE, namespace);
        }
        if has_attr(&cur_attrs, S_TIR)? {
            device_func = with_attr(&device_func, S_TIR, true);
        }
        if let Some(params) = typed_attr::<Array<String>>(&cur_attrs, KERNEL_LAUNCH_PARAMS)? {
            device_func = with_attr(&device_func, KERNEL_LAUNCH_PARAMS, params);
        }
        if let Some(inputs) = typed_attr::<i64>(&cur_attrs, NUM_INPUTS)? {
            device_func = with_attr(&device_func, NUM_INPUTS, inputs);
        }

        let symbol = GlobalVar::new(self.names.fresh(&self.kernel_name)?)?;
        if has_attr(&region.attrs, LAUNCH_FIELDS)? {
            let launch = make_cuda_kernel_launch(&symbol, &mut device_func, call_args, region)?;
            self.device_funcs.push((symbol, device_func));
            return Ok(launch);
        }
        self.device_funcs.push((symbol.clone(), device_func));
        if can_propagate_errors {
            let code = Var::new("kernel_error_code", int32_ty.clone())?;
            let kernel_call = call(int32_ty, symbol, call_args, None)?;
            let succeeded = code.equal(&success).expr()?;
            let assert_success: Stmt = AssertStmt::new(
                succeeded,
                StringImm::new("RuntimeError")?,
                Array::new(vec![StringImm::new("Error executing compute kernel")?]),
            )?
            .into();
            let bind: Stmt = Bind::new(code, kernel_call)?.into();
            return Ok(SeqStmt::new(vec![bind, assert_success])?.into());
        }
        let kernel_call = call(kernel_ret_type, symbol, call_args, None)?;
        Ok(Evaluate::new(kernel_call)?.into())
    }
}

/// `SplitHostDevice(func, device_mod, var_supply)` for one function.
fn split_function(
    func: Function,
    names: &UniqueNames,
    kernel_name: std::string::String,
    device_funcs: &mut Vec<(GlobalVar, Function)>,
) -> Result<Function> {
    let Some(body) = func.body.clone() else { return Ok(func) };
    let mut splitter =
        HostDeviceSplitter { names, kernel_name, cur_func: func.clone(), device_funcs: vec![] };
    let split: SeqStmt = structural_mutate(body.clone(), &mut splitter)?.try_into()?;
    device_funcs.append(&mut splitter.device_funcs);
    if split.same_as(&body) {
        return Ok(func);
    }
    Ok(func.with_body(Some(split)))
}

// ---------------------------------------------------------------------------------------------
// Device kernel launch lowering

/// `KernelInfo`: what a kernel needs from its launch.
#[derive(Clone)]
struct KernelInfo {
    target: Target,
    global_symbol: String,
    params: Array<Var>,
    launch_params: Vec<String>,
    launch_args: Vec<Expr>,
    dynamic_smem_requirement: Option<Expr>,
}

/// `DeviceInfoCollector`: the launch extents and dynamic shared memory of a kernel.
struct DeviceInfoCollector {
    /// The `shared.dyn` memory scope.
    shared_dyn: MemoryScope,
    info: KernelInfo,
    /// Thread tag -> extent, in the order the tags were found.
    thread_extent: Vec<(String, Expr)>,
    saw_dyn_shared_alloc: bool,
    inferred_shmem_size: Option<Expr>,
    use_pdl: bool,
    use_cooperative_launch: bool,
    /// Bind definitions, inlined into extents and sizes so they refer to function parameters.
    bind_map: HashMap<ObjectIdentity, Expr>,
}

#[dispatch(walk)]
impl DeviceInfoCollector {
    fn walk_expr(&mut self, _expr: Expr) -> WalkResult {
        WalkResult::Skip
    }

    fn walk_bind(&mut self, op: Bind) -> Result<WalkResult> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if ALLOC_TENSOR.is(&call.op) {
                self.dispatch_alloc_tensor(call)?;
                return Ok(WalkResult::Advance);
            }
        }
        if !is_prim_expr(&op.value) {
            return Ok(WalkResult::Advance);
        }
        let value = match self.bind_map.is_empty() {
            true => op.value.clone(),
            false => substitute(&self.bind_map, &op.value)?,
        };
        self.bind_map.insert(ObjectIdentity::of(&op.var), value);
        Ok(WalkResult::Advance)
    }
}

impl DeviceInfoCollector {
    fn thread_extent(&self, tag: &str) -> Option<&Expr> {
        self.thread_extent.iter().find(|(name, _)| name.as_str() == tag).map(|(_, extent)| extent)
    }

    /// A launched thread defines the extent of its axis, which every launch of it must agree on.
    fn enter_region(&mut self, op: &RegionStmt) -> Result<()> {
        if !LAUNCH_THREAD.is(&op.op) {
            return Ok(());
        }
        let tag: StringImm = op.args.get(0)?.try_cast()?;
        if tag.value.as_str().starts_with("vthread") {
            return Ok(());
        }
        let mut value = op.args.get(1)?;
        if !self.bind_map.is_empty() {
            value = substitute(&self.bind_map, &value)?;
        }
        match self.thread_extent(tag.value.as_str()) {
            Some(previous) => {
                check!(
                    Analyzer::new()?.can_prove_equal(previous, &value)?,
                    VALUE_ERROR,
                    "analyzer->CanProveEqual(previous.value(), value)",
                    "Incompatible launch extents for {}",
                    tag.value.as_str()
                );
            }
            None => {
                self.info.launch_params.push(tag.value.clone());
                self.thread_extent.push((tag.value.clone(), value));
            }
        }
        Ok(())
    }

    /// A `shared.dyn` allocation sizes the dynamic shared memory of the launch.
    fn dispatch_alloc_tensor(&mut self, call: &CallObj) -> Result<()> {
        let tensor: TensorType = call.ty_args.get(0)?.try_cast()?;
        if !tensor.scope.same_as(&self.shared_dyn) {
            return Ok(());
        }
        check!(
            !self.saw_dyn_shared_alloc,
            INTERNAL_ERROR,
            "!saw_dyn_shared_alloc_",
            "Only one dynamic shared memory allocation is allowed.",
        );
        self.saw_dyn_shared_alloc = true;
        check_cmp!(tensor.shape.len(), >, 0, INTERNAL_ERROR, "shape->fields.size() > 0");
        let mut size = Ex::from(int32(1)?);
        for extent in tensor.shape.iter() {
            size *= extent;
        }
        let dtype = tensor.dtype.dtype;
        let bytes = (dtype.bits as i128 * dtype.lanes as i128 + 7) / 8;
        let element_bytes = int_imm(tvm_ffi::DLDataType { code: 0, bits: 64, lanes: 1 }, bytes)?;
        let mut size = (size * element_bytes).expr()?;
        if !self.bind_map.is_empty() {
            size = substitute(&self.bind_map, &size)?;
        }
        self.inferred_shmem_size = Some(size);
        Ok(())
    }

    fn argument(&self, launch_param: &str, dyn_shmem_size: &Option<Expr>) -> Result<Expr> {
        if launch_param == USE_DYN_SHARED_MEMORY {
            return dyn_shmem_size.clone().ok_or_else(|| {
                error!(
                    INTERNAL_ERROR,
                    "Compute kernel requires launch parameter \"{launch_param}\", but Function \
                     has no dynamic shared memory requirement."
                )
            });
        }
        self.thread_extent(launch_param).cloned().ok_or_else(|| {
            error!(
                INTERNAL_ERROR,
                "Compute kernel requires launch parameter \"{launch_param}\", but Function does \
                 not contain a launch region defining this axis"
            )
        })
    }
}

/// The scope of a region: the bindings made inside it do not reach past it.
struct RegionScope;

impl ContextPolicy<DeviceInfoCollector> for RegionScope {
    fn default_visit(
        &self,
        value: &StructuralView,
        ctx: &mut VisitContext<'_, DeviceInfoCollector>,
    ) -> Result<Option<VisitInterrupt>> {
        let Some(region) = value.cast::<RegionStmt>() else { return ctx.visit_children() };
        ctx.state_mut().enter_region(&region)?;
        let outer = ctx.state().bind_map.clone();
        let result = ctx.visit_children();
        ctx.state_mut().bind_map = outer;
        result
    }
}

/// `DeviceInfoCollector::Collect(gvar, func, allow_placeholder)`.
fn collect_kernel_info(
    gvar: &GlobalVar,
    func: &Function,
    allow_placeholder: bool,
) -> Result<KernelInfo> {
    let mut info = KernelInfo {
        target: without_host(&function_target(func)?)?,
        global_symbol: String::from(""),
        params: func.params.clone(),
        launch_params: Vec::new(),
        launch_args: Vec::new(),
        dynamic_smem_requirement: None,
    };
    let global_symbol =
        typed_attr::<String>(&func.attrs, GLOBAL_SYMBOL)?.unwrap_or_else(|| gvar.name_hint.clone());
    if typed_attr::<Array<String>>(&func.attrs, LAUNCH_FIELDS)?.is_some() {
        info.global_symbol = global_symbol;
        return Ok(info);
    }
    let mut collector = DeviceInfoCollector {
        shared_dyn: MemoryScope::get("shared.dyn")?,
        info,
        thread_extent: Vec::new(),
        saw_dyn_shared_alloc: false,
        inferred_shmem_size: None,
        use_pdl: false,
        use_cooperative_launch: false,
        bind_map: HashMap::new(),
    };
    if let Some(requested) = typed_attr::<Array<String>>(&func.attrs, KERNEL_LAUNCH_PARAMS)? {
        for tag in requested.iter() {
            match tag.as_str() {
                USE_PDL => collector.use_pdl = true,
                USE_COOPERATIVE_LAUNCH => collector.use_cooperative_launch = true,
                _ => {}
            }
        }
    }
    let mut walk = WalkWithContextPolicy::new(collector, RegionScope);
    if let Some(body) = &func.body {
        structural_walk(body, &mut walk, WalkOrder::PreOrder)?;
    }
    let mut collector = walk.into_state();
    if collector.use_pdl {
        collector.info.launch_params.push(String::from(USE_PDL));
    }
    if collector.use_cooperative_launch {
        collector.info.launch_params.push(String::from(USE_COOPERATIVE_LAUNCH));
    }
    // Dynamic shared memory is the last legacy launch operand.
    let mut dyn_shmem_size = None;
    if let Some(inferred) = collector.inferred_shmem_size.clone() {
        check!(
            allow_placeholder || !is_const_int(&inferred, 0),
            INTERNAL_ERROR,
            "allow_placeholder || !(inferred && inferred->value == 0)",
            "Function {} allocates dynamic shared memory with a placeholder extent but does not \
             declare its size; use LaunchConfig.dynamic_smem_bytes or SMEMPool.commit().",
            gvar.name_hint.as_str()
        );
        dyn_shmem_size = Some(inferred);
    }
    if dyn_shmem_size.is_some() {
        collector.info.launch_params.push(String::from(USE_DYN_SHARED_MEMORY));
    }
    collector.info.global_symbol = global_symbol;
    for param in collector.info.launch_params.clone() {
        if [USE_PDL, USE_COOPERATIVE_LAUNCH, USE_REQUIRED_BLOCK_DIMENSION].contains(&param.as_str())
        {
            continue;
        }
        let argument = collector.argument(param.as_str(), &dyn_shmem_size)?;
        collector.info.launch_args.push(argument);
    }
    collector.info.dynamic_smem_requirement = dyn_shmem_size;
    Ok(collector.info)
}

/// `MakeCudaKernelLaunch`: a kernel with CUDA launch fields is launched by `call_ffi_kernel`
/// with the launch values of its region, and records its static launch geometry.
fn make_cuda_kernel_launch(
    symbol: &GlobalVar,
    func: &mut Function,
    args: Vec<Expr>,
    region: &RegionStmt,
) -> Result<Stmt> {
    let missing = |key: &str| error!(INTERNAL_ERROR, "key {key} is not in the map");
    let mut fields: Vec<String> = Array::<String>::try_from(
        get_attr(&region.attrs, LAUNCH_FIELDS)?.ok_or_else(|| missing(LAUNCH_FIELDS))?,
    )?
    .iter()
    .collect();
    let mut values: Vec<Expr> = region.args.iter().collect();
    let kernel_attrs: Map<String, i64> =
        get_attr(&region.attrs, KERNEL_ATTRS)?.ok_or_else(|| missing(KERNEL_ATTRS))?.try_into()?;
    let info = collect_kernel_info(symbol, func, true)?;
    let mut host_stmts: Vec<Stmt> = Vec::new();
    let mut required_bytes = info.dynamic_smem_requirement.clone();
    let has_pool = has_attr(&region.attrs, SMEM_REQUIRED)?;
    if let Some(pool_bytes) = get_attr(&region.attrs, SMEM_REQUIRED)? {
        let int64 = tvm_ffi::DLDataType { code: 0, bits: 64, lanes: 1 };
        let pool: Expr = int_imm(int64, i64::try_from(pool_bytes)? as i128)?.into();
        required_bytes = Some(match required_bytes {
            Some(required) => crate::prim::Max::new(required, pool)?.into(),
            None => pool,
        });
    }
    if let Some(required) = required_bytes {
        let analyzer = Analyzer::new()?;
        let bytes = analyzer.simplify(&required, 2)?;
        match fields.iter().rposition(|field| field.as_str() == "dynamic_smem_bytes") {
            None => {
                check!(
                    !is_const_int(&bytes, 0) || has_pool,
                    VALUE_ERROR,
                    "!prim::IsZero(bytes) || \
                     region->attrs->dict.count(tvm::backend::cuda::attr::kSmemRequired)",
                    "A shared.dyn placeholder requires LaunchConfig.dynamic_smem_bytes or \
                     SMEMPool.commit()",
                );
                fields.push(String::from("dynamic_smem_bytes"));
                values.push(bytes);
            }
            Some(index) => {
                let available = values[index].greater_equal(&bytes).expr()?;
                let condition = analyzer.simplify(&available, 2)?;
                let message =
                    "LaunchConfig.dynamic_smem_bytes is smaller than the kernel allocation";
                check!(
                    !is_const_int(&condition, 0),
                    VALUE_ERROR,
                    "!prim::IsZero(condition)",
                    "{}",
                    message
                );
                if !is_const_int(&condition, 1) {
                    let assert: Stmt = AssertStmt::new(
                        condition,
                        StringImm::new("ValueError")?,
                        Array::new(vec![StringImm::new(message)?]),
                    )?
                    .into();
                    host_stmts.push(assert);
                }
            }
        }
    }
    // Only constants needed by code generation enter the kernel's metadata; zero marks a dynamic
    // dimension.
    let mut dimensions: Vec<Expr> = Vec::new();
    for prefix in ["block.", "cluster."] {
        for axis in ['x', 'y', 'z'] {
            let name = format!("{prefix}{axis}");
            let mut dimension: Expr = int32(1)?.into();
            for (field, value) in fields.iter().zip(&values) {
                if field.as_str() == name {
                    let value = Analyzer::new()?.simplify(value, 2)?;
                    dimension = match value.as_node::<IntImmObj>() {
                        Some(_) => value,
                        None => int32(0)?.into(),
                    };
                }
            }
            dimensions.push(dimension);
        }
    }
    let fields = Array::new(fields);
    *func = with_attr(func, CALLING_CONV, DEVICE_KERNEL_LAUNCH);
    *func = with_attr(func, GLOBAL_SYMBOL, symbol.name_hint.clone());
    *func = with_attr(func, LAUNCH_FIELDS, fields.clone());
    *func = with_attr(func, KERNEL_ATTRS, kernel_attrs.clone());
    *func = with_attr(func, LAUNCH_DIMENSIONS, Array::new(dimensions));
    let num_kernel_args = args.len() as i64;
    let attrs = CallFFIKernelAttr::new(Array::new(vec![]), fields, num_kernel_args, kernel_attrs);
    let mut call_args: Vec<Expr> = vec![StringImm::new(symbol.name_hint.as_str())?.into()];
    call_args.extend(args);
    call_args.extend(values);
    let int32_ty: Type = prim_type(int32_dtype())?.into();
    let launch = call(int32_ty, CALL_FFI_KERNEL.get()?, call_args, Some(attrs.into()))?;
    host_stmts.push(Evaluate::new(launch)?.into());
    Ok(SeqStmt::new(host_stmts)?.into())
}

/// `ReturnRemover`: a kernel may only return 0, which is dropped unless kept.
struct ReturnRemover {
    remove: bool,
}

#[dispatch(mutate)]
impl ReturnRemover {
    fn mutate_return(&mut self, op: Return) -> Result<Any> {
        check!(
            is_const_int(&op.value, 0),
            INTERNAL_ERROR,
            "as_int && as_int->value == 0",
            "Device kernel may only contain a successful return, return 0",
        );
        if !self.remove {
            return Ok(op.into());
        }
        Ok(Evaluate::new(0)?.into())
    }
}

/// `GlobalVarCallCollector`: the functions called anywhere in the module.
struct CalledFunctions {
    called: HashSet<ObjectIdentity>,
}

#[dispatch(walk)]
impl CalledFunctions {
    fn walk_call(&mut self, call: Call) -> WalkResult {
        if let Ok(gvar) = call.op.clone().try_cast::<GlobalVar>() {
            self.called.insert(ObjectIdentity::of(&gvar));
        }
        WalkResult::Advance
    }
}

/// `DeviceKernelMutator`: calls to kernels become launches, and calls to another target on the
/// same device become external calls.
struct DeviceKernelMutator {
    current_target: Option<Target>,
    /// The host target of the caller being rewritten, when the caller is a host function.
    caller_host_target: Option<Target>,
    device_info: HashMap<ObjectIdentity, KernelInfo>,
    kernel_launches: HashSet<ObjectIdentity>,
    extern_calls: HashSet<ObjectIdentity>,
}

#[dispatch(mutate)]
impl DeviceKernelMutator {
    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        let node: Call = m.default_mutate_as(self, &op)?;
        let Ok(gvar) = op.op.clone().try_cast::<GlobalVar>() else { return Ok(node.into()) };
        let name = gvar.name_hint.as_str();
        let Some(info) = self.device_info.get(&ObjectIdentity::of(&gvar)).cloned() else {
            bail!(
                INTERNAL_ERROR,
                "Check failed: (it != device_info_map_.end()) is false: CallNode attempted \
                 subroutine call to {name}, but {name} did not appear within the IRModule"
            );
        };
        let callee_target = info.target.clone();
        let callee_is_kernel = !info.launch_params.is_empty();
        let caller_is_host = self.caller_host_target.is_some();
        let caller_target = match &self.caller_host_target {
            Some(host) => host.clone(),
            None => self.current_target.clone().expect("set for the function being rewritten"),
        };
        // A host caller always launches a real device kernel.
        let force_kernel_launch = callee_is_kernel && caller_is_host;
        // CUDA helpers stay direct calls; they are specialized per kernel after splitting.
        if !caller_is_host
            && !callee_is_kernel
            && kind_name(&caller_target)?.as_str() == "cuda"
            && kind_name(&callee_target)?.as_str() == "cuda"
        {
            return Ok(node.into());
        }
        if !force_kernel_launch {
            if caller_target.str()?.as_str() == callee_target.str()?.as_str() {
                return Ok(node.into());
            }
            if caller_target.device_type()? == callee_target.device_type()? {
                self.extern_calls.insert(ObjectIdentity::of(&gvar));
                let mut args: Vec<Expr> = vec![StringImm::new(name)?.into()];
                args.extend(node.args.iter());
                let ret_ty: Type = match is_void_type(&node.ty) {
                    true => prim_void()?.into(),
                    false => node.ty.clone(),
                };
                return Ok(call(ret_ty, CALL_EXTERN.get()?, args, None)?.into());
            }
        }

        // The launch arguments of the kernel are in terms of its parameters; substitute the
        // arguments of this call.
        let args: Vec<Expr> = node.args.iter().collect();
        check_cmp!(
            args.len(),
            ==,
            info.params.len(),
            INTERNAL_ERROR,
            "args.size() == dev_info.params.size()",
            "Function {} accepts {} arguments as input, but is called using {} arguments",
            name,
            info.params.len(),
            args.len()
        );
        let mut param_map = HashMap::new();
        for (param, arg) in info.params.iter().zip(&args) {
            if is_prim_expr(arg) {
                param_map.insert(ObjectIdentity::of(&param), arg.clone());
            }
        }
        self.kernel_launches.insert(ObjectIdentity::of(&gvar));
        let mut call_args: Vec<Expr> = vec![StringImm::new(info.global_symbol.as_str())?.into()];
        call_args.extend(args);
        for launch_arg in &info.launch_args {
            call_args.push(substitute(&param_map, launch_arg)?);
        }
        let node_ty: PrimType = match is_void_type(&node.ty) {
            true => prim_void()?,
            false => node.ty.clone().try_cast()?,
        };
        let is_void = node_ty.dtype == prim_void()?.dtype;
        let ret_ty: Type = match is_void {
            true => prim_type(int32_dtype())?.into(),
            false => node_ty.into(),
        };
        let attrs = CallFFIKernelAttr::new(
            Array::new(info.launch_params.clone()),
            Array::new(vec![]),
            -1,
            Map::new(),
        );
        Ok(call(ret_ty, CALL_FFI_KERNEL.get()?, call_args, Some(attrs.into()))?.into())
    }
}

impl DeviceKernelMutator {
    fn rewrite_kernel_launch_site(&mut self, func: Function) -> Result<Function> {
        let full_target = function_target(&func)?;
        self.current_target = Some(without_host(&full_target)?);
        self.caller_host_target = host_of(&full_target)?;
        let result = match func.body.clone() {
            Some(body) => structural_mutate(body.clone(), &mut *self)
                .and_then(SeqStmt::try_from)
                .map(|new_body| match new_body.same_as(&body) {
                    true => func.clone(),
                    false => func.with_body(Some(new_body)),
                }),
            None => Ok(func.clone()),
        };
        self.current_target = None;
        self.caller_host_target = None;
        result
    }

    fn update_kernel_attributes(&self, gvar: &GlobalVar, func: Function) -> Result<Function> {
        let id = ObjectIdentity::of(gvar);
        let is_kernel_launch = self.kernel_launches.contains(&id);
        let is_call_extern = self.extern_calls.contains(&id);
        check!(
            !is_kernel_launch || !is_call_extern,
            INTERNAL_ERROR,
            "!is_kernel_launch || !is_call_extern",
            "Function {} has multiple callees, and would need to be lowered into a call_extern at \
             some call sites, and a device kernel launch at others.  This case is not yet \
             supported.",
            gvar.name_hint.as_str()
        );
        let mut func = func;
        if is_kernel_launch || is_call_extern {
            func = with_attr(&func, IS_GLOBAL_FUNC, true);
        }
        if is_kernel_launch {
            let info = &self.device_info[&id];
            // A kernel launch returns the launch status; the kernel itself returns nothing.
            let target = function_target(&func)?;
            let preserve_early_returns = kind_name(&target)?.as_str() == "cuda";
            let body =
                func.body.clone().ok_or_else(|| error!(INTERNAL_ERROR, "Bad optional access"))?;
            let mut remover = ReturnRemover { remove: !preserve_early_returns };
            let body: Stmt = structural_mutate(body, &mut remover)?.try_into()?;
            let body = SeqStmt::from_stmt(body)?;
            let void = void_type()?;
            func = func.with(|node| {
                node.ret_type = void;
                node.body = Some(body);
            });
            func = with_attr(&func, CALLING_CONV, DEVICE_KERNEL_LAUNCH);
            func = with_attr(&func, KERNEL_LAUNCH_PARAMS, Array::new(info.launch_params.clone()));
            func = with_attr(&func, GLOBAL_SYMBOL, info.global_symbol.clone());
        } else if is_call_extern && get_attr(&func.attrs, GLOBAL_SYMBOL)?.is_none() {
            func = with_attr(&func, GLOBAL_SYMBOL, gvar.name_hint.clone());
        }
        Ok(func)
    }
}

/// The module's functions as an ordered table, updated in place as `IRModuleNode::Update` does.
struct ModuleFunctions {
    functions: Vec<(GlobalVar, BaseFunc)>,
    global_var_map: Vec<(String, GlobalVar)>,
}

impl ModuleFunctions {
    fn of(module: &IRModule) -> Self {
        Self {
            functions: module.functions.iter().collect(),
            global_var_map: module.global_var_map.iter().collect(),
        }
    }

    /// `IRModuleNode::AddUnchecked(gvar, func)`.
    fn update(&mut self, gvar: &GlobalVar, func: impl Into<BaseFunc>) {
        let func = func.into();
        match self.functions.iter_mut().find(|(key, _)| key.same_as(gvar)) {
            Some(entry) => entry.1 = func,
            None => self.functions.push((gvar.clone(), func)),
        }
        match self
            .global_var_map
            .iter_mut()
            .find(|(name, _)| name.as_str() == gvar.name_hint.as_str())
        {
            Some(entry) => entry.1 = gvar.clone(),
            None => self.global_var_map.push((gvar.name_hint.clone(), gvar.clone())),
        }
    }

    fn into_module(self, module: &IRModule) -> IRModule {
        IRModule::new(
            self.functions.into_iter().collect::<Map<GlobalVar, BaseFunc>>(),
            module.source_map.clone(),
            module.attrs.clone(),
            module.global_infos.clone(),
            self.global_var_map.into_iter().collect::<Map<String, GlobalVar>>(),
        )
    }
}

/// `LowerDeviceKernelLaunches(mod)`.
fn lower_device_kernel_launches(module: IRModule) -> Result<IRModule> {
    let mut collector = CalledFunctions { called: HashSet::new() };
    for (_gvar, base_func) in module.functions.iter() {
        if let Ok(func) = base_func.try_cast::<Function>() {
            structural_walk(&func.body, &mut collector, WalkOrder::PreOrder)?;
        }
    }
    let mut device_info = HashMap::new();
    for (gvar, base_func) in module.functions.iter() {
        if !collector.called.contains(&ObjectIdentity::of(&gvar)) {
            continue;
        }
        if let Ok(func) = base_func.try_cast::<Function>() {
            device_info
                .insert(ObjectIdentity::of(&gvar), collect_kernel_info(&gvar, &func, false)?);
        }
    }
    let mut mutator = DeviceKernelMutator {
        current_target: None,
        caller_host_target: None,
        device_info,
        kernel_launches: HashSet::new(),
        extern_calls: HashSet::new(),
    };

    let mut table = ModuleFunctions::of(&module);
    let mut updates = Vec::new();
    for (gvar, base_func) in module.functions.iter() {
        if let Ok(func) = base_func.clone().try_cast::<Function>() {
            let rewritten = mutator.rewrite_kernel_launch_site(func)?;
            if !rewritten.same_as(&base_func) {
                updates.push((gvar, rewritten));
            }
        }
    }
    for (gvar, func) in &updates {
        table.update(gvar, func.clone());
    }
    let module = if updates.is_empty() { module } else { table.into_module(&module) };

    let mut table = ModuleFunctions::of(&module);
    let mut updates = Vec::new();
    for (gvar, base_func) in module.functions.iter() {
        if let Ok(func) = base_func.clone().try_cast::<Function>() {
            let updated = mutator.update_kernel_attributes(&gvar, func)?;
            if !updated.same_as(&base_func) {
                updates.push((gvar, updated));
            }
        }
    }
    for (gvar, func) in &updates {
        table.update(gvar, func.clone());
    }
    Ok(if updates.is_empty() { module } else { table.into_module(&module) })
}

pub fn split_host_device() -> Result<Pass> {
    module_pass("tirx.SplitHostDevice", 0, move |module, _context| {
        let names =
            UniqueNames::new(module.functions.iter().map(|(gvar, _)| gvar.name_hint.clone()))?;
        let mut device_funcs = Vec::new();
        let mut updates = Vec::new();
        for (gvar, base_func) in module.functions.iter() {
            let Ok(func) = base_func.clone().try_cast::<Function>() else { continue };
            let func = annotate_device_regions(func)?;
            let prefix = typed_attr::<String>(&func.attrs, GLOBAL_SYMBOL)?
                .unwrap_or_else(|| gvar.name_hint.clone());
            let kernel_name = format!("{}_kernel", prefix.as_str());
            let func = split_function(func, &names, kernel_name, &mut device_funcs)?;
            if !func.same_as(&base_func) {
                updates.push((gvar, func));
            }
        }
        let mut table = ModuleFunctions::of(&module);
        for (gvar, func) in updates.into_iter().chain(device_funcs) {
            table.update(&gvar, func);
        }
        let module = table.into_module(&module);
        let module = convert_ssa_module(module)?;
        lower_device_kernel_launches(module)
    })
}
