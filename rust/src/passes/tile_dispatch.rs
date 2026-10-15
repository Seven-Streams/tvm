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

//! `tirx.TileDispatch` (`src/tirx/tile/lower.cc`): replace every tile instruction by the
//! implementation its `FTileOpDispatch` returns for the execution context it runs in, and
//! `tirx.LowerTIRx` (`src/tirx/transform/lower_tirx.cc`), which follows it with
//! `LowerTIRxCleanup`.
//!
//! Inside a CUDA kernel the context comes from [`ExecAnalysis`]; a kernel launched through
//! `cuda.launch_fields` also gets its CUDA index calls lowered and is wrapped in
//! `tirx.device_scope`. The statements the dispatchers ask for through their callbacks land at
//! the kernel's replace point, after the tensor they follow, or on the host.

use std::collections::HashMap;

use tvm_ffi::{
    dispatch, structural_mutate, structural_walk, Any, AnyView, Array, Error, FieldGetter,
    Function as FfiFunction, Map, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result,
    String, Unchanged, WalkOrder, WalkResult,
};

use super::cuda_exec_analysis::{lower_indices, ExecAnalysis, LaunchParams};
use super::util::{MutatorExt, function_pass, get_attr, internal, sequential};
use crate::ir::{
    Bind, Call, CallObj, Evaluate, Expr, For, If, IntImmObj, Op, OpObj, OpAttrMap, Range,
    RegionStmt, SeqStmt, Stmt, Var,
};
use crate::ops::{int32_dtype, prim_dtype, unknown_loc};
use crate::sym::Analyzer;
use crate::target::Target;
use crate::tirx::{DispatchContext, ExecScope, Function, TensorTypeObj};
use crate::transform::Pass;

/// A failed C++ `TVM_FFI_CHECK(condition, ValueError) << message`.
fn value_error(condition: &str, message: &str) -> Error {
    let message = format!("Check failed: ({condition}) is false: {message}");
    Error::new(tvm_ffi::VALUE_ERROR, &message, "")
}

/// The operators the pass recognizes.
struct TileOps {
    device_entry: Op,
    device_scope: Op,
    alloc_tensor: Op,
    decl_tensor: Op,
    decl_tmem: Op,
    tensor_data_ptr: Op,
    kernel_replace_point: Op,
    /// The `FTileOpDispatch` column, or `None` when no operator registered one.
    dispatchers: Option<OpAttrMap<FfiFunction>>,
}

impl TileOps {
    fn new() -> Result<Self> {
        Ok(TileOps {
            device_entry: Op::get("tirx.device_entry")?,
            device_scope: Op::get("tirx.device_scope")?,
            alloc_tensor: Op::get("tirx.alloc_tensor")?,
            decl_tensor: Op::get("tirx.decl_tensor")?,
            decl_tmem: Op::get("tirx.cuda.decl_tmem")?,
            tensor_data_ptr: Op::get("tirx.tensor_data_ptr")?,
            kernel_replace_point: Op::get("tirx.kernel_replace_point")?,
            dispatchers: Op::attr_map("FTileOpDispatch")?,
        })
    }

    /// The dispatcher of a call to a tile instruction.
    fn dispatcher(&self, call: &CallObj) -> Result<Option<FfiFunction>> {
        match (call.op.as_node::<OpObj>(), &self.dispatchers) {
            (Some(op), Some(dispatchers)) => dispatchers.get(op),
            _ => Ok(None),
        }
    }

    /// A call that defines a tensor: `alloc_tensor`, `decl_tensor` or `cuda.decl_tmem`.
    fn defines_tensor(&self, value: &Expr) -> bool {
        value.as_node::<CallObj>().is_some_and(|call| {
            call.op.same_as(&self.alloc_tensor)
                || call.op.same_as(&self.decl_tensor)
                || call.op.same_as(&self.decl_tmem)
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Helper rewriters

/// C++ `KernelReplacePointSearcher::Seek`: `stmt` with each `kernel_replace_point()` evaluation
/// replaced by `body`.
fn seek(stmt: &Stmt, body: &Stmt, ops: &TileOps) -> Result<Stmt> {
    struct Searcher<'a> {
        body: &'a Stmt,
        replace_point: &'a Op,
    }

    #[dispatch(mutate)]
    impl Searcher<'_> {
        fn mutate_evaluate(&mut self, op: Evaluate, m: &mut Mutator) -> Result<Any> {
            if op.value.as_node::<CallObj>().is_some_and(|call| call.op.same_as(self.replace_point))
            {
                return Ok(self.body.clone().into());
            }
            m.default_mutate(self, &op)
        }
    }

    let mut searcher = Searcher { body, replace_point: &ops.kernel_replace_point };
    structural_mutate(stmt.clone(), &mut searcher)?.try_into()
}

/// C++ `TensorRefRewriter::Rewrite`: `stmt` with the tensor `src` replaced by `dst`.
fn rewrite_tensor_ref(stmt: &Stmt, src: &Var, dst: &Var) -> Result<Stmt> {
    struct Rewriter<'a> {
        src: &'a Var,
        dst: &'a Var,
    }

    #[dispatch(mutate)]
    impl Rewriter<'_> {
        fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
            if var.same_as(self.src) {
                return Ok(self.dst.clone().into());
            }
            m.default_mutate(self, &var)
        }
    }

    if src.same_as(dst) {
        return Ok(stmt.clone());
    }
    structural_mutate(stmt.clone(), &mut Rewriter { src, dst })?.try_into()
}

/// A variable whose type is a tirx `TensorType`.
fn tensor_var(expr: &Expr) -> Option<Var> {
    let var = expr.clone().try_cast::<Var>().ok()?;
    var.ty.as_node::<TensorTypeObj>().is_some().then_some(var)
}

/// C++ `StorageRootResolver::Apply`: each `tensor_data_ptr(view)` as the data pointer of the
/// view's storage root.
fn resolve_storage_roots(
    stmt: &Stmt,
    roots: &HashMap<ObjectIdentity, Var>,
    ops: &TileOps,
) -> Result<Stmt> {
    struct Resolver<'a> {
        roots: &'a HashMap<ObjectIdentity, Var>,
        tensor_data_ptr: &'a Op,
    }

    #[dispatch(mutate)]
    impl Resolver<'_> {
        fn mutate_call(&mut self, call: Call, m: &mut Mutator) -> Result<Any> {
            if call.op.same_as(self.tensor_data_ptr) && call.args.len() == 1 {
                if let Some(var) = tensor_var(&call.args.get(0)?) {
                    if let Some(root) = self.roots.get(&ObjectIdentity::of(&var)) {
                        if !root.same_as(&var) {
                            let data: Expr = tvm_ffi::cached_global_func!("tirx.TensorData")
                                .call_tuple((root.clone(),))?
                                .try_into()?;
                            return Ok(data.into());
                        }
                    }
                }
            }
            m.default_mutate(self, &call)
        }
    }

    let mut resolver = Resolver { roots, tensor_data_ptr: &ops.tensor_data_ptr };
    structural_mutate(stmt.clone(), &mut resolver)?.try_into()
}

/// C++ `NoOpCallVerifier::Verify`: no tile instruction is left. (C++ also rejects a statement
/// its visitor cannot dispatch; every statement the parser makes can be dispatched.)
fn no_tile_calls_left(stmt: &Stmt, ops: &TileOps) -> Result<bool> {
    let mut error = None;
    let interrupted = structural_walk(
        stmt,
        |call: Call| match ops.dispatcher(&call) {
            Ok(Some(_)) => WalkResult::Interrupt,
            Ok(None) => WalkResult::Advance,
            Err(e) => {
                error = Some(e);
                WalkResult::Interrupt
            }
        },
        WalkOrder::PreOrder,
    )?;
    if let Some(error) = error {
        return Err(error);
    }
    Ok(interrupted.is_none())
}

// ---------------------------------------------------------------------------------------------
// The dispatcher

/// Restores the enclosing target context when dropped (C++ `With<Target>`).
struct TargetScope(Target);

impl TargetScope {
    fn enter(target: &Target) -> Result<Self> {
        tvm_ffi::cached_global_func!("target.TargetEnterScope").call_tuple((target.clone(),))?;
        Ok(TargetScope(target.clone()))
    }
}

impl Drop for TargetScope {
    fn drop(&mut self) {
        let _ =
            tvm_ffi::cached_global_func!("target.TargetExitScope").call_tuple((self.0.clone(),));
    }
}

struct TileDispatcher {
    ops: TileOps,
    target: Target,
    native_launch: bool,
    variable_cluster: bool,
    cluster_bounds: Vec<(std::string::String, Expr)>,
    /// `var_range_map_`, in insertion order, with the position of each variable.
    var_ranges: Vec<(Var, Range)>,
    var_index: HashMap<ObjectIdentity, usize>,
    analyzer: Analyzer,
    analysis: Option<ExecAnalysis>,
    launch_params: LaunchParams,
    alloc_tensors: Vec<Bind>,
    device_init_stmts: Vec<Stmt>,
    host_init_stmts: Vec<Stmt>,
    /// The storage root of each tensor variable defined in the body.
    tensor_root: HashMap<ObjectIdentity, Var>,
    post_tensor_def_stmts: HashMap<ObjectIdentity, (Var, Vec<Stmt>)>,
    shared_state: Map<String, tvm_ffi::object::ObjectRef>,
    is_first_block: bool,
}

#[dispatch(mutate)]
impl TileDispatcher {
    /// Expressions hold no statements and are left as they are.
    fn mutate_expr(&mut self, _expr: Expr) -> Unchanged {
        Unchanged
    }

    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        if op.op.same_as(&self.ops.device_entry) {
            return Ok(self.process_device_entry(&op, m)?.into());
        }
        m.default_mutate(self, &op)
    }

    /// The statements registered to follow a tensor definition come right after it.
    fn mutate_seq(&mut self, op: SeqStmt, m: &mut Mutator) -> Result<Any> {
        let stmt: Stmt = m.default_mutate_as(self, &Stmt::from(op))?;
        if self.post_tensor_def_stmts.is_empty() {
            return Ok(stmt.into());
        }
        let seq: SeqStmt = stmt.clone().try_cast()?;
        let mut rebuilt = Vec::with_capacity(seq.seq.len());
        let mut changed = false;
        for s in seq.seq.iter() {
            rebuilt.push(s.clone());
            if let Ok(bind) = s.try_cast::<Bind>() {
                if self.ops.defines_tensor(&bind.value) {
                    changed |=
                        self.append_post_tensor_def_stmts(&mut rebuilt, &bind.var, &bind.var)?;
                }
            }
        }
        if !changed {
            return Ok(stmt.into());
        }
        Ok(Stmt::from(SeqStmt::new_full(rebuilt, seq.loc.clone())?).into())
    }

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if call.op.same_as(&self.ops.alloc_tensor) {
                return Ok(self.mutate_tensor_def(op, false, m)?.into());
            }
            if call.op.same_as(&self.ops.decl_tensor) || call.op.same_as(&self.ops.decl_tmem) {
                return Ok(self.mutate_tensor_def(op, true, m)?.into());
            }
        }
        let stmt: Stmt = m.default_mutate_as(self, &Stmt::from(op))?;
        let bind: Bind = stmt.clone().try_cast().map_err(|_| internal("Check failed: bind"))?;
        if prim_dtype(&bind.value).is_ok() {
            if let Some(analysis) = &mut self.analysis {
                analysis.bind(&bind.var, &bind.value)?;
            }
            // Bind is flat: under SSA a stale entry is never consulted, so none is removed.
            self.set_var_range(&bind.var, Range::from_min_extent(bind.value.clone(), 1)?);
        }
        Ok(stmt.into())
    }

    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        if self.var_index.contains_key(&ObjectIdentity::of(&op.loop_var)) {
            return Err(internal(
                "Check failed: (!var_range_map_.count(loop_var)) is false: Internal Error: \
                 Duplicate loop variable",
            ));
        }
        self.set_var_range(
            &op.loop_var,
            Range::from_min_extent(op.min.clone(), op.extent.clone())?,
        );
        m.default_mutate(self, &op)
    }

    /// The then-branch runs in the context narrowed by the condition, which loses its thread
    /// filters; the condition itself is not visited.
    fn mutate_if(&mut self, op: If, m: &mut Mutator) -> Result<Any> {
        let pushed = match &mut self.analysis {
            Some(analysis) => analysis.push_predicate(&op.condition)?,
            None => 0,
        };
        let condition = match &self.analysis {
            Some(analysis) => analysis.rewrite_predicate(&op.condition)?,
            None => op.condition.clone(),
        };
        let then_case: Stmt = m.mutate_as(self, &Stmt::from(op.then_case.clone()))?;
        if let Some(analysis) = &mut self.analysis {
            analysis.pop(pushed);
        }
        let else_case = match &op.else_case {
            Some(else_case) => {
                Some(SeqStmt::from_stmt(m.mutate_as::<_, Stmt>(self, &else_case.clone().into())?)?)
            }
            None => None,
        };
        let else_unchanged = match (&op.else_case, &else_case) {
            (Some(old), Some(new)) => new.same_as(old),
            (None, None) => true,
            _ => false,
        };
        if condition.same_as(&op.condition) && then_case.same_as(&op.then_case) && else_unchanged {
            return Ok(op.into());
        }
        Ok(Stmt::from(If::new(condition, then_case, else_case)?).into())
    }

    /// A tile instruction is replaced by its implementation, itself dispatched.
    fn mutate_evaluate(&mut self, stmt: Evaluate, m: &mut Mutator) -> Result<Any> {
        let Some(call) = stmt.value.as_node::<CallObj>() else {
            return m.default_mutate(self, &stmt);
        };
        let Some(dispatcher) = self.ops.dispatcher(call)? else {
            return m.default_mutate(self, &stmt);
        };
        let call: Call = stmt.value.clone().try_cast()?;
        tvm_ffi::cached_global_func!("ir.CallValidate").call_tuple((call.clone(),))?;
        let Some(attrs) = call.attrs.clone() else {
            return Err(value_error(
                "op->attrs.has_value()",
                "A tile instruction must have scope attributes",
            ));
        };
        let attrs_type = Any::from(attrs.clone()).type_index();
        let scope_name: String = FieldGetter::new(attrs_type, "scope")?.get(&*attrs)?;
        let kind = self.target.kind()?.name()?;
        let scope: ExecScope = tvm_ffi::cached_global_func!("tirx.ExecScope")
            .call_tuple((kind, scope_name))?
            .try_into()?;
        let (inter, intra) = match &self.analysis {
            Some(analysis) => analysis.split(&scope)?,
            None => (Map::new(), Map::new()),
        };
        let launch_params: Map<String, Array<Expr>> = self
            .launch_params
            .iter()
            .map(|(tag, var, extent)| {
                (String::from(tag.as_str()), Array::new(vec![var.clone().into(), extent.clone()]))
            })
            .collect();
        let var_range_map: Map<Var, Range> = self.var_ranges.iter().cloned().collect();
        let callbacks: Map<String, tvm_ffi::object::ObjectRef> = Map::new();
        let args = [
            Any::from(self.target.clone()),
            Any::from(scope.clone()),
            Any::from(launch_params),
            Any::from(var_range_map),
            Any::from(false),
            Any::from(callbacks),
            Any::from(self.shared_state.clone()),
            Any::from(inter),
            Any::from(intra),
            Any::from(scope.name.clone()),
        ];
        let args: Vec<AnyView> = args.iter().map(AnyView::from).collect();
        let context: DispatchContext =
            tvm_ffi::cached_global_func!("tirx.DispatchContext").call_packed(&args)?.try_into()?;
        let implementation = dispatcher.call_tuple((call, context.clone()))?;
        if implementation.type_index() == tvm_ffi::TypeIndex::kTVMFFINone as i32 {
            return Err(internal(
                "Check failed: implementation.defined() is false: TIRx dispatcher did not return \
                 a Function",
            ));
        }
        let implementation: Function = implementation.try_into()?;
        self.collect_callbacks(&context.callbacks()?)?;
        self.shared_state = context.shared_state()?;
        let Some(body) = implementation.body.clone() else {
            return Err(value_error(
                "res->body.has_value()",
                "A tile op implementation must have a body",
            ));
        };
        m.mutate(self, &body)
    }
}

impl TileDispatcher {
    fn new(target: Target) -> Result<Self> {
        Ok(TileDispatcher {
            ops: TileOps::new()?,
            target,
            native_launch: false,
            variable_cluster: false,
            cluster_bounds: Vec::new(),
            var_ranges: Vec::new(),
            var_index: HashMap::new(),
            analyzer: Analyzer::new()?,
            analysis: None,
            launch_params: Vec::new(),
            alloc_tensors: Vec::new(),
            device_init_stmts: Vec::new(),
            host_init_stmts: Vec::new(),
            tensor_root: HashMap::new(),
            post_tensor_def_stmts: HashMap::new(),
            shared_state: Map::new(),
            is_first_block: true,
        })
    }

    /// `var_range_map_.Set(var, range)`.
    fn set_var_range(&mut self, var: &Var, range: Range) {
        let key = ObjectIdentity::of(var);
        match self.var_index.get(&key) {
            Some(&index) => self.var_ranges[index].1 = range,
            None => {
                self.var_index.insert(key, self.var_ranges.len());
                self.var_ranges.push((var.clone(), range));
            }
        }
    }

    /// The statements the dispatcher asked for through the context's callbacks.
    fn collect_callbacks(
        &mut self,
        callbacks: &Map<String, tvm_ffi::object::ObjectRef>,
    ) -> Result<()> {
        let get = |key: &str| callbacks.get(&String::from(key));
        if let Some(allocs) = get("private_alloc")? {
            let allocs: Array<Bind> = Any::from(allocs).try_into()?;
            self.alloc_tensors.extend(allocs.iter());
        }
        if let Some(stmts) = get("device_init_stmt")? {
            let stmts: Array<Stmt> = Any::from(stmts).try_into()?;
            self.device_init_stmts.extend(stmts.iter());
        }
        if let Some(stmts) = get("host_init_stmt")? {
            let stmts: Array<Stmt> = Any::from(stmts).try_into()?;
            self.host_init_stmts.extend(stmts.iter());
        }
        if let Some(mapping) = get("post_tensor_def_stmt")? {
            let mapping: Map<Var, Array<Stmt>> = Any::from(mapping).try_into()?;
            for (tensor, stmts) in mapping.iter() {
                let entry = self
                    .post_tensor_def_stmts
                    .entry(ObjectIdentity::of(&tensor))
                    .or_insert_with(|| (tensor.clone(), Vec::new()));
                entry.1.extend(stmts.iter());
            }
        }
        Ok(())
    }

    /// C++ `MutateAllocTensor` / `MutateDeclTensor`: the definition, then the statements
    /// registered to follow the tensor.
    fn mutate_tensor_def(&mut self, op: Bind, declared: bool, m: &mut Mutator) -> Result<Stmt> {
        let old_tensor = op.var.clone();
        let stmt: Stmt = m.default_mutate_as(self, &Stmt::from(op))?;
        let bind: Bind = stmt.clone().try_cast().map_err(|_| internal("Check failed: op"))?;
        let data = if declared {
            let call = bind
                .value
                .as_node::<CallObj>()
                .ok_or_else(|| internal("Check failed: tensor_call"))?;
            Some(call.args.get(0)?)
        } else {
            None
        };
        self.register_storage_root(&old_tensor, &bind.var, data.as_ref());
        let mut seq = vec![stmt];
        self.append_post_tensor_def_stmts(&mut seq, &old_tensor, &bind.var)?;
        Ok(SeqStmt::new(seq)?.into())
    }

    /// A `decl_tensor` over `tensor_data_ptr(src)` shares `src`'s root; anything else owns its
    /// storage. Function parameters are absent and are their own root.
    fn register_storage_root(&mut self, old_var: &Var, new_var: &Var, data: Option<&Expr>) {
        let mut root = new_var.clone();
        if let Some(call) = data.and_then(|data| data.as_node::<CallObj>()) {
            if call.op.same_as(&self.ops.tensor_data_ptr) && call.args.len() == 1 {
                if let Some(src) = call.args.get(0).ok().as_ref().and_then(tensor_var) {
                    root = self.storage_root_of(&src);
                }
            }
        }
        self.tensor_root.insert(ObjectIdentity::of(new_var), root.clone());
        if !old_var.same_as(new_var) {
            self.tensor_root.insert(ObjectIdentity::of(old_var), root);
        }
    }

    fn storage_root_of(&self, var: &Var) -> Var {
        self.tensor_root.get(&ObjectIdentity::of(var)).cloned().unwrap_or_else(|| var.clone())
    }

    /// Appends the statements registered for `old_tensor` and `new_tensor`, rewritten onto
    /// `new_tensor`, and forgets them.
    fn append_post_tensor_def_stmts(
        &mut self,
        seq: &mut Vec<Stmt>,
        old_tensor: &Var,
        new_tensor: &Var,
    ) -> Result<bool> {
        let mut keys = vec![old_tensor];
        if !new_tensor.same_as(old_tensor) {
            keys.push(new_tensor);
        }
        let mut changed = false;
        for key in keys {
            let Some((src, stmts)) = self.post_tensor_def_stmts.remove(&ObjectIdentity::of(key))
            else {
                continue;
            };
            let no_op: Stmt = Evaluate::new(0)?.into();
            for stmt in stmts {
                let remapped = rewrite_tensor_ref(&stmt, &src, new_tensor)?;
                seq.push(seek(&remapped, &no_op, &self.ops)?);
            }
            changed = true;
        }
        Ok(changed)
    }

    /// C++ `ProcessDeviceEntry`.
    fn process_device_entry(&mut self, entry: &RegionStmt, m: &mut Mutator) -> Result<Stmt> {
        let saved_target = self.target.clone();
        if let Some(target) = get_attr(&entry.attrs, "target")? {
            self.target = target.try_into()?;
        }
        let _target_scope = TargetScope::enter(&self.target)?;
        let is_first_block = std::mem::replace(&mut self.is_first_block, false);

        self.launch_params.clear();
        self.cluster_bounds.clear();
        self.variable_cluster = false;
        self.native_launch = get_attr(&entry.attrs, "cuda.launch_fields")?.is_some();
        if !is_first_block && self.native_launch {
            return Err(value_error(
                "is_first_block || !native_launch_",
                "Nested CUDA device_entry regions are not supported",
            ));
        }
        if self.native_launch {
            self.prepare_cuda_launch_params(entry)?;
        }
        let saved_analysis = self.analysis.take();
        if self.target.kind()?.name()?.as_str() == "cuda" {
            self.analysis = Some(ExecAnalysis::new(
                &self.launch_params,
                &self.cluster_bounds,
                self.variable_cluster,
            )?);
        }

        let body_input: Stmt = entry.body.clone().into();
        let body: Stmt = m.mutate_as(self, &body_input)?;
        let body_unchanged = body.same_as(&body_input);

        let restore = |this: &mut Self| {
            this.analysis = saved_analysis;
            this.target = saved_target;
        };

        if !is_first_block {
            self.is_first_block = is_first_block;
            restore(self);
            if body_unchanged {
                return Ok(entry.clone().into());
            }
            let region = RegionStmt::new_full(
                entry.op.clone(),
                entry.args.clone(),
                entry.body_params.clone(),
                entry.attrs.clone(),
                body,
                entry.result_vars.clone(),
                entry.loc.clone(),
            )?;
            return Ok(region.into());
        }

        let mut body = body;
        for init in self.device_init_stmts.clone().iter().rev() {
            body = seek(init, &body, &self.ops)?;
        }
        if !self.alloc_tensors.is_empty() {
            let mut seq: Vec<Stmt> = self.alloc_tensors.iter().cloned().map(Stmt::from).collect();
            seq.push(body);
            body = SeqStmt::new(seq)?.into();
        }
        self.alloc_tensors.clear();

        let mut res = body;
        if self.native_launch {
            res = lower_indices(&res, &self.launch_params)?;
        }
        if self.native_launch || get_attr(&entry.attrs, "backend_config")?.is_some() {
            res = RegionStmt::new(
                self.ops.device_scope.clone(),
                entry.args.clone(),
                Array::new(vec![]),
                entry.attrs.clone(),
                res,
            )?
            .into();
        }
        // Host init statements leave the kernel, where a `tensor_data_ptr` of a device-local
        // view cannot be resolved: each one is moved onto the view's storage root.
        for stmt in std::mem::take(&mut self.host_init_stmts) {
            let resolved = resolve_storage_roots(&stmt, &self.tensor_root, &self.ops)?;
            res = seek(&resolved, &res, &self.ops)?;
        }
        self.is_first_block = is_first_block;
        restore(self);
        Ok(res)
    }

    /// C++ `PrepareCudaLaunchParams`: the launch variables and extents of the kernel, and the
    /// bounds of its cluster.
    fn prepare_cuda_launch_params(&mut self, entry: &RegionStmt) -> Result<()> {
        if self.target.kind()?.name()?.as_str() != "cuda" {
            return Err(value_error(
                "target_->kind->name == \"cuda\"",
                "CUDA LaunchConfig requires a CUDA target",
            ));
        }
        let fields: Array<String> = get_attr(&entry.attrs, "cuda.launch_fields")?
            .ok_or_else(|| internal("cuda.launch_fields"))?
            .try_into()?;
        if fields.len() != entry.args.len() {
            return Err(internal(&format!(
                "Check failed: fields.size() == entry->args.size() ({} vs. {}) : ",
                fields.len(),
                entry.args.len()
            )));
        }
        for (field, arg) in fields.iter().zip(entry.args.iter()) {
            let field = field.as_str();
            let tag = if let Some(axis) = field.strip_prefix("grid.") {
                format!("blockIdx.{axis}")
            } else if let Some(axis) = field.strip_prefix("block.") {
                format!("threadIdx.{axis}")
            } else if let Some(axis) = field.strip_prefix("cluster.") {
                format!("clusterCtaIdx.{axis}")
            } else {
                continue;
            };
            prim_dtype(&arg)?;
            let extent = self.analyzer.simplify(&arg, 2)?;
            let var = Var::prim(&tag, int32_dtype())?;
            if !self.launch_params.iter().any(|(name, _, _)| *name == tag) {
                self.launch_params.push((tag, var, extent));
            }
        }
        let has = |tag: &str| self.launch_params.iter().any(|(name, _, _)| name == tag);
        if !has("blockIdx.x") || !has("threadIdx.x") {
            return Err(value_error(
                "launch_params_.count(\"blockIdx.x\") && launch_params_.count(\"threadIdx.x\")",
                "LaunchConfig requires grid and block dimensions",
            ));
        }
        self.variable_cluster = false;
        self.cluster_bounds.clear();
        for axis in ["x", "y", "z"] {
            let tag = format!("clusterCtaIdx.{axis}");
            let Some((_, _, extent)) = self.launch_params.iter().find(|(name, _, _)| *name == tag)
            else {
                continue;
            };
            let mut extent = extent.clone();
            self.variable_cluster |= extent.as_node::<IntImmObj>().is_none();
            let preferred_field = format!("preferred_cluster.{axis}");
            for (field, arg) in fields.iter().zip(entry.args.iter()) {
                if field.as_str() == preferred_field {
                    prim_dtype(&arg)?;
                    self.variable_cluster |= !self.analyzer.can_prove_equal(&extent, &arg)?;
                    let max: Expr = tvm_ffi::cached_global_func!("prim.Max")
                        .call_tuple((extent.clone(), arg.clone(), unknown_loc()?))?
                        .try_into()?;
                    extent = self.analyzer.simplify(&max, 2)?;
                }
            }
            self.cluster_bounds.push((tag, extent));
        }
        Ok(())
    }
}

/// `TileDispatcher::LowerOpCalls`.
fn lower_op_calls(body: SeqStmt, target: Target) -> Result<Stmt> {
    let mut dispatcher = TileDispatcher::new(target)?;
    structural_mutate(Stmt::from(body), &mut dispatcher)?.try_into()
}

/// How C++ prints a node into an error message.
fn repr<T: Into<Any> + Clone>(value: &T) -> Result<std::string::String> {
    let text: String = tvm_ffi::cached_global_func!("ffi.ReprPrint")
        .call_tuple((value.clone().into(),))?
        .try_into()?;
    Ok(text.as_str().to_string())
}

pub fn tile_dispatch() -> Result<Pass> {
    function_pass("tirx.TileDispatch", 0, move |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let target: Target = match get_attr(&func.attrs, "target")? {
            Some(target) => target.try_into()?,
            None => tvm_ffi::cached_global_func!("target.TargetCurrent")
                .call_tuple((false,))?
                .try_into()?,
        };
        let body = lower_op_calls(body, target)?;
        let ops = TileOps::new()?;
        let func = func.with_body(Some(SeqStmt::from_stmt(body.clone())?));
        if !no_tile_calls_left(&body, &ops)? {
            return Err(internal(&format!("Failed to lower the TIRx program: {}", repr(&func)?)));
        }
        Ok(Some(func))
    })
}

pub fn lower_tirx() -> Result<Pass> {
    let mut passes = vec![tile_dispatch()?];
    if std::env::var_os("TVM_PRINT_AFTER_TIRX_DISPATCH_OPS").is_some() {
        let print: Pass = tvm_ffi::cached_global_func!("transform.PrintIR")
            .call_tuple((String::from(""),))?
            .try_into()?;
        passes.push(print);
    }
    passes.push(super::lower_tirx_cleanup::lower_tirx_cleanup()?);
    sequential(passes, "tirx.LowerTIRx")
}
