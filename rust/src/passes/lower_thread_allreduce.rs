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

//! `tirx.LowerThreadAllreduce` (`src/tirx/transform/lower_thread_allreduce.cc`): lower each
//! `tirx.gpu_thread_allreduce` to warp shuffles, or to a reduction through shared memory, over the
//! threads of its enclosing `launch_thread` regions.

use std::collections::{BTreeMap, HashMap, HashSet};

use tvm_ffi::{
    bail, dispatch, structural_map, structural_mutate, Any, AnyView, Array, DLDataType, Map,
    Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String, UnchangedOr, WalkOrder,
    INTERNAL_ERROR, TYPE_ERROR, VALUE_ERROR,
};

use super::analyzer::is_const_int;
use super::util::{MutatorExt, dict_with, function_pass, get_attr, int_imm, prim_dtype};
use crate::ir::{
    Bind, Call, CallObj, DictAttrs, Evaluate, Expr, If, LambdaExpr, LambdaExprObj, OpHandle,
    PrimType, PrimTypeObj, RegionStmt, SeqStmt, Stmt, StringImm, StringImmObj, TensorLoad,
    TensorLoadObj, TensorStore, TupleObj, Var, VarObj,
};
use crate::ops::{prim_type, PrimOps};
use crate::prim::{CastObj, Select};
use crate::runtime::MemoryScope;
use crate::sym::Analyzer;
use crate::target::Target;
use crate::tirx::op::{
    ALLOC_TENSOR, DECL_TENSOR, GPU_STORAGE_SYNC, GPU_THREAD_ALLREDUCE, GPU_WARP_ACTIVEMASK,
    GPU_WARP_SHUFFLE, GPU_WARP_SHUFFLE_DOWN, LAUNCH_THREAD, TENSOR_DATA_PTR,
};
use crate::tirx::{TensorType, TensorTypeObj};
use crate::transform::Pass;

/// `TensorStore(dest, indices, value)`, as a statement.
fn tensor_store(dest: &Var, indices: Vec<Expr>, value: Expr) -> Result<Stmt> {
    Ok(TensorStore::new(dest.clone(), indices, value)?.into())
}

fn tensor_load(source: &Var, indices: Vec<Expr>) -> Result<Expr> {
    Ok(TensorLoad::new(source.clone(), indices)?.into())
}

/// The tensor type of a `TensorVar`.
fn tensor_ty(var: &Var) -> &TensorTypeObj {
    var.ty.as_node::<TensorTypeObj>().expect("a TensorVar has a TensorType")
}

/// `expr.as_or_throw<TensorVar>()`: exactly a `Var` whose type is a `TensorType`.
fn as_tensor_var(expr: &Expr) -> Result<Var> {
    let var = expr.as_node::<VarObj>().filter(|_| is_exact_var(expr));
    match var {
        Some(var) if var.ty.as_node::<TensorTypeObj>().is_some() => Ok(expr.clone().try_cast()?),
        _ => bail!(TYPE_ERROR, "Cannot convert to `tirx.TensorVar`"),
    }
}

/// `expr.as<PrimVar>()`: exactly a `Var` whose type is a `PrimType`.
fn as_prim_var(expr: &Expr) -> Option<Var> {
    let var = expr.as_node::<VarObj>().filter(|_| is_exact_var(expr))?;
    var.ty.as_node::<PrimTypeObj>()?;
    expr.clone().try_cast().ok()
}

fn is_exact_var(expr: &Expr) -> bool {
    AnyView::from(expr).type_index() == <VarObj as tvm_ffi::ObjectCore>::type_index()
}

/// `expr.as_or_throw<PrimExpr>()`.
fn as_prim_expr(expr: &Expr) -> Result<Expr> {
    match expr.ty.as_node::<PrimTypeObj>() {
        Some(_) => Ok(expr.clone()),
        None => bail!(TYPE_ERROR, "Cannot convert to `ir.PrimExpr`"),
    }
}

fn prim_ty(expr: &Expr) -> Result<PrimType> {
    as_prim_expr(expr)?;
    expr.ty.clone().try_cast()
}

/// `PrimType::lanes()`, which a scalable vector has none of at compile time.
fn lanes(dtype: DLDataType) -> Result<i32> {
    match dtype.lanes as i16 {
        lanes if lanes < 0 => {
            bail!(INTERNAL_ERROR, "Can't fetch the lanes of a scalable vector at a compile time.")
        }
        lanes => Ok(lanes as i32),
    }
}

/// A `Tuple` argument's fields, or the argument itself.
fn fields(value: &Expr) -> Vec<Expr> {
    match value.as_node::<TupleObj>() {
        Some(tuple) => tuple.fields.iter().collect(),
        None => vec![value.clone()],
    }
}

/// `support::GPUThreadTag::Parse(tag)`: the rank (block, thread, cluster CTA, preferred cluster
/// CTA) and dimension of a thread tag.
fn parse_thread_tag(tag: &str) -> Result<(i32, i32)> {
    let dim = |prefix: &str| (tag.as_bytes()[prefix.len()] as i32) - ('x' as i32);
    for (rank, prefix) in
        [(0, "blockIdx."), (1, "threadIdx."), (2, "clusterCtaIdx."), (3, "preferredClusterCtaIdx.")]
    {
        if tag.starts_with(prefix) {
            return Ok((rank, dim(prefix)));
        }
    }
    bail!(INTERNAL_ERROR, "Unknown threadscope {tag}")
}

/// `LambdaExpr::Apply`: the body with the arguments substituted for the parameters.
fn apply_lambda(lambda: &LambdaExprObj, arguments: Vec<Expr>) -> Result<Expr> {
    if arguments.len() != lambda.vars.len() {
        bail!(VALUE_ERROR, "LambdaExpr Apply arity mismatch");
    }
    let substitutions: HashMap<ObjectIdentity, Expr> =
        lambda.vars.iter().map(|var| ObjectIdentity::of(&var)).zip(arguments).collect();
    structural_map(
        lambda.body.clone(),
        |var: Var| match substitutions.get(&ObjectIdentity::of(&var)) {
            Some(value) => UnchangedOr::changed(value.clone()),
            None => UnchangedOr::unchanged(),
        },
        WalkOrder::PostOrder,
    )?
    .try_into()
}

/// `ApplyCombiner(combiner, lhs, rhs)`: the combined values, one per reduction.
fn apply_combiner(combiner: &LambdaExpr, lhs: &[Expr], rhs: &[Expr]) -> Result<Vec<Expr>> {
    let arguments = lhs.iter().chain(rhs).cloned().collect();
    let result = apply_lambda(combiner, arguments)?;
    fields(&result).iter().map(as_prim_expr).collect()
}

/// `(GetBufferDataVar(data))`: the variable a tensor declaration's data refers to.
fn buffer_data_var(data: &Expr) -> Option<Var> {
    if let Ok(var) = data.clone().try_cast::<Var>() {
        return Some(var);
    }
    let call = data.as_node::<CallObj>()?;
    if TENSOR_DATA_PTR.is(&call.op) && call.args.len() == 1 {
        return call.args.get(0).ok()?.try_cast::<Var>().ok();
    }
    None
}

/// A thread axis of the enclosing launches.
#[derive(Clone)]
struct ThreadEntry {
    rank: i32,
    dim_index: i32,
    var: Var,
    extent: i32,
}

/// The remappings an allreduce leaves behind, shared by the builder and `DeferredRemapper`.
#[derive(Default)]
struct Remaps {
    /// Allocation root -> the tensor that replaces its allocation.
    alloc: HashMap<ObjectIdentity, Var>,
    /// Allocation root or destination tensor -> the tensor that replaces it.
    var: HashMap<ObjectIdentity, Var>,
    /// Allocations visited before their allreduce (flat IR), remapped afterwards.
    pending: Vec<ObjectIdentity>,
    /// The physical root of each tensor alias.
    aliases: HashMap<ObjectIdentity, Var>,
}

impl Remaps {
    fn root(&self, var: &Var) -> Var {
        self.aliases.get(&ObjectIdentity::of(var)).cloned().unwrap_or_else(|| var.clone())
    }

    /// `GetAllocationKey`: the allocation a tensor's accesses go to.
    fn allocation_key(&self, tensor: &Var) -> ObjectIdentity {
        match tensor.ty.as_node::<TensorTypeObj>() {
            Some(_) => ObjectIdentity::of(&self.root(tensor)),
            None => ObjectIdentity::of(tensor),
        }
    }

    /// `GetRemappedTensor`.
    fn remapped_tensor(&self, tensor: &Var) -> Option<Var> {
        self.var.get(&ObjectIdentity::of(&self.root(tensor))).cloned()
    }

    /// `RemapAllocTensor`: the allocation of `node` made for `replacement`, volatile in shared
    /// memory.
    fn remap_alloc(&self, node: &Bind, replacement: &Var, shared: &MemoryScope) -> Result<Stmt> {
        let call = node.value.as_node::<CallObj>().expect("checked to be an alloc_tensor call");
        let Some(attrs) = &call.attrs else {
            bail!(INTERNAL_ERROR, "alloc_tensor without attributes");
        };
        let mut annotations: DictAttrs = attrs.clone().try_cast()?;
        if tensor_ty(replacement).scope.same_as(shared) {
            annotations = dict_with(&annotations, "tirx.volatile", Any::from(true));
        }
        let ty = replacement.ty.clone();
        let value = Call::new_full(
            Some(ty.clone()),
            ALLOC_TENSOR.get()?.into(),
            call.args.clone(),
            Some(annotations.into()),
            Array::new(vec![ty]),
            call.loc.clone(),
        )?;
        Ok(Bind::new_full(replacement.clone(), value, node.loc.clone())?.into())
    }
}

struct ThreadAllreduceBuilder {
    shared: MemoryScope,
    local: MemoryScope,
    target_kind: std::string::String,
    warp_size: i64,
    max_num_threads: i64,
    need_warp_shuffle_mask: bool,
    /// The enclosing `launch_thread` regions, outermost first.
    thread_extents: Vec<RegionStmt>,
    /// Allocation root -> the expression its loads become.
    load_remap: HashMap<ObjectIdentity, Expr>,
    analyzer: Analyzer,
    remaps: Remaps,
}

#[dispatch(mutate)]
impl ThreadAllreduceBuilder {
    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        if !LAUNCH_THREAD.is(&op.op) || thread_tag(&op)?.starts_with("vthread") {
            return m.default_mutate(self, &op);
        }
        self.thread_extents.push(op.clone());
        let result = m.default_mutate(self, &op);
        self.thread_extents.pop();
        result
    }

    fn mutate_evaluate(&mut self, op: Evaluate, m: &mut Mutator) -> Result<Any> {
        let stmt: Evaluate = m.default_mutate_as(self, &op)?;
        match stmt.value.as_node::<CallObj>() {
            Some(call) if GPU_THREAD_ALLREDUCE.is(&call.op) => {
                Ok(self.make_allreduce(call)?.into())
            }
            _ => Ok(stmt.into()),
        }
    }

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if ALLOC_TENSOR.is(&call.op) {
                return self.mutate_alloc_tensor(op, m);
            }
            if DECL_TENSOR.is(&call.op) {
                let data = call.args.get(0)?;
                self.register_tensor_alias(&as_tensor_var(&op.var.clone().into())?, &data)?;
                // Declarations are remapped by `DeferredRemapper`, once every allreduce is known.
                return m.default_mutate(self, &op);
            }
        }
        m.default_mutate(self, &op)
    }

    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let allocation = self.remaps.allocation_key(&as_tensor_var(&op.source)?);
        if let Some(remapped) = self.load_remap.get(&allocation) {
            check_zero_indices(op.indices.iter())?;
            return Ok(remapped.clone().into());
        }
        let load: TensorLoad = m.default_mutate_as(self, &op)?;
        match self.remaps.remapped_tensor(&as_tensor_var(&load.source)?) {
            Some(tensor) => {
                let indices: Vec<Expr> = load.indices.iter().collect();
                Ok(TensorLoad::new_full(tensor, indices, load.loc.clone())?.into())
            }
            None => Ok(load.into()),
        }
    }

    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let allocation = self.remaps.allocation_key(&as_tensor_var(&op.dest)?);
        let store: TensorStore = m.default_mutate_as(self, &op)?;
        if let Some(remapped) = self.load_remap.get(&allocation) {
            let replacement = remapped.as_node::<TensorLoadObj>();
            check!(replacement.is_some(), INTERNAL_ERROR, "replacement");
            let replacement = replacement.expect("checked above");
            check_zero_indices(store.indices.iter())?;
            let (dest, indices) = (replacement.source.clone(), replacement.indices.clone());
            let value = store.value.clone();
            return Ok(
                TensorStore::from_complete_fields(store.loc.clone(), dest, indices, value).into()
            );
        }
        match self.remaps.remapped_tensor(&as_tensor_var(&store.dest)?) {
            Some(tensor) => {
                let (loc, indices, value) =
                    (store.loc.clone(), store.indices.clone(), store.value.clone());
                Ok(TensorStore::from_complete_fields(loc, tensor.into(), indices, value).into())
            }
            None => Ok(store.into()),
        }
    }
}

fn thread_tag(region: &RegionStmt) -> Result<std::string::String> {
    let tag = region.args.get(0)?;
    match tag.as_node::<StringImmObj>() {
        Some(tag) => Ok(tag.value.as_str().to_string()),
        None => bail!(TYPE_ERROR, "Cannot convert to `ir.StringImm`"),
    }
}

fn check_zero_indices(indices: impl Iterator<Item = Expr>) -> Result<()> {
    for index in indices {
        check!(is_const_int(&index, 0), INTERNAL_ERROR, "IsZero(index)");
    }
    Ok(())
}

impl ThreadAllreduceBuilder {
    fn new(target: &Target, params: &Array<Var>) -> Result<Self> {
        let attrs = target.attrs()?;
        let attr = |key: &str, default: i64| -> Result<i64> {
            match attrs.get(&String::from(key))? {
                Some(value) => i64::try_from(value),
                None => Ok(default),
            }
        };
        let mut remaps = Remaps::default();
        for param in params.iter() {
            if param.ty.as_node::<TensorTypeObj>().is_some() {
                remaps.aliases.insert(ObjectIdentity::of(&param), param.clone());
            }
        }
        Ok(Self {
            shared: MemoryScope::get("shared")?,
            local: MemoryScope::get("local")?,
            target_kind: target.kind()?.name()?.as_str().to_string(),
            warp_size: attr("thread_warp_size", 1)?,
            max_num_threads: attr("max_num_threads", -1)?,
            need_warp_shuffle_mask: false,
            thread_extents: Vec::new(),
            load_remap: HashMap::new(),
            analyzer: Analyzer::new()?,
            remaps,
        })
    }

    fn mutate_alloc_tensor(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        let key = ObjectIdentity::of(&op.var);
        self.remaps.aliases.insert(key.clone(), op.var.clone());
        let node: Bind = m.default_mutate_as(self, &op)?;
        if let Some(replacement) = self.remaps.alloc.get(&key) {
            return Ok(self.remaps.remap_alloc(&node, replacement, &self.shared)?.into());
        }
        self.remaps.pending.push(key);
        Ok(node.into())
    }

    /// `RegisterTensorAlias`: a declaration over another tensor's data shares its root.
    fn register_tensor_alias(&mut self, tensor: &Var, data: &Expr) -> Result<()> {
        let mut root = tensor.clone();
        if let Some(source) = buffer_data_var(data) {
            if source.ty.as_node::<TensorTypeObj>().is_some() {
                match self.remaps.aliases.get(&ObjectIdentity::of(&source)) {
                    Some(source_root) => root = source_root.clone(),
                    None => bail!(
                        INTERNAL_ERROR,
                        "Tensor alias source {} must be registered before its DeclTensor alias",
                        source.name.as_str()
                    ),
                }
            }
        }
        self.remaps.aliases.insert(ObjectIdentity::of(tensor), root);
        Ok(())
    }

    /// `SyncThread(sync)`.
    fn sync_thread(&self, sync: &str) -> Result<Stmt> {
        let ty = prim_type(super::util::int32_dtype())?;
        let call = Call::new(ty, GPU_STORAGE_SYNC.get()?, vec![StringImm::new(sync)?.into()])?;
        Ok(Evaluate::new(call)?.into())
    }

    /// `WarpShuffle(op, mask_tensor, val, delta_or_lane)`.
    fn warp_shuffle(
        &self,
        op: &OpHandle,
        mask_tensor: Option<&Var>,
        val: Expr,
        delta_or_lane: Expr,
    ) -> Result<Expr> {
        let mask: Expr = match mask_tensor.filter(|_| self.need_warp_shuffle_mask) {
            Some(mask) => tensor_load(mask, vec![crate::ops::int32(0)?.into()])?,
            None => crate::ops::int32(0)?.into(),
        };
        let width: Expr = crate::ops::int32(self.warp_size)?.into();
        let ty = val.ty.clone();
        Ok(Call::new(ty, op.get()?, vec![mask, val, delta_or_lane, width.clone(), width])?.into())
    }

    /// `FlattenThread`: the flattened index of `threads` and their total extent.
    fn flatten_thread(threads: &[ThreadEntry]) -> Result<(Expr, i32)> {
        let Some(first) = threads.first() else {
            return Ok((crate::ops::int32(0)?.into(), 1));
        };
        let mut index: Expr = first.var.clone().into();
        let mut total = first.extent;
        for thread in &threads[1..] {
            index = (&index + (&thread.var * total)).expr()?;
            total *= thread.extent;
        }
        Ok((index, total))
    }

    /// `BufIndex`: the shared buffer slot of a reduction lane in a group.
    fn buf_index(
        &self,
        reduce_index: &Expr,
        group_index: &Expr,
        reduce_extent: i32,
    ) -> Result<Expr> {
        if is_const_int(group_index, 0) {
            return Ok(reduce_index.clone());
        }
        let index = (group_index * reduce_extent + reduce_index).expr()?;
        self.analyzer.simplify(&index, 2)
    }

    /// `IsWarpReduction`: whether shuffles within warps can do the reduction.
    fn is_warp_reduction(
        &mut self,
        dtypes: &[DLDataType],
        group_extent: i32,
        reduce_extent: i32,
        contiguous_reduce_extent: i32,
    ) -> Result<bool> {
        let kind = self.target_kind.as_str();
        if !matches!(kind, "cuda" | "rocm" | "metal" | "webgpu") {
            return Ok(false);
        }
        self.need_warp_shuffle_mask = kind != "metal" && kind != "webgpu";
        // ROCm only shuffles 32-bit operands.
        if kind == "rocm" {
            for dtype in dtypes {
                let lanes = lanes(*dtype)?;
                let unsupported = match lanes > 1 {
                    true => dtype.bits as i32 * lanes != 32,
                    false => dtype.bits != 32,
                };
                if unsupported {
                    return Ok(false);
                }
            }
        }
        // {u}int, {u}long, {u}long long, float, double, half and half2.
        for dtype in dtypes {
            let lanes = lanes(*dtype)?;
            let unsupported = if dtype.code == 2 && dtype.bits == 16 {
                lanes > 2
            } else if lanes > 1 {
                true
            } else {
                let bytes = (dtype.bits as i32 * lanes + 7) / 8;
                !(4..=8).contains(&bytes)
            };
            if unsupported {
                return Ok(false);
            }
        }
        if self.thread_extents.is_empty() || contiguous_reduce_extent != reduce_extent {
            return Ok(false);
        }
        let (warp, reduce) = (self.warp_size, reduce_extent as i64);
        if kind == "rocm" {
            return Ok(reduce == warp);
        }
        if reduce == 1 {
            return Ok(false);
        }
        let subwarp = warp % reduce == 0;
        let multiwarp =
            self.max_num_threads != -1 && self.max_num_threads <= warp * warp && reduce % warp == 0;
        Ok(subwarp || multiwarp || (group_extent == 1 && reduce <= warp))
    }

    /// `MakeAllreduce(call)`.
    fn make_allreduce(&mut self, call: &CallObj) -> Result<Stmt> {
        let combiner: LambdaExpr = call.args.get(0)?.try_cast()?;
        let inits = fields(&call.args.get(1)?);
        let inputs = fields(&call.args.get(2)?);
        let destinations = fields(&call.args.get(4)?);
        let thread_axes = fields(&call.args.get(5)?);
        let size = inputs.len();
        let cond = as_prim_expr(&call.args.get(3)?)?;
        let mut values = Vec::with_capacity(size);
        let mut dtypes = Vec::with_capacity(size);
        for (idx, input) in inputs.iter().enumerate() {
            let mut value = as_prim_expr(input)?;
            if !is_const_int(&cond, 1) {
                let init = as_prim_expr(&inits[idx])?;
                value = Select::new(&cond, value, init)?.into();
            }
            dtypes.push(prim_dtype(&value)?);
            values.push(value);
        }
        let mut tensors = Vec::with_capacity(size);
        for destination in destinations.iter().take(size) {
            let mut arg = as_prim_expr(destination)?;
            // Loads from boolean tensors may have cast nodes inserted by earlier passes.
            if let Some(cast) = arg.as_node::<CastObj>() {
                arg = cast.value.clone();
            }
            let Some(load) = arg.as_node::<TensorLoadObj>() else {
                bail!(TYPE_ERROR, "Cannot convert to `ir.TensorLoad`");
            };
            tensors.push(as_tensor_var(&load.source)?);
        }

        let mut reduce_set = HashSet::new();
        for axis in &thread_axes {
            match as_prim_var(axis) {
                Some(var) => {
                    reduce_set.insert(ObjectIdentity::of(&var));
                }
                None if is_const_int(axis, 0) => {}
                None => {
                    bail!(
                        INTERNAL_ERROR,
                        "Reduction thread axis should be a VarNode or zero IntImmNode"
                    )
                }
            }
        }

        let mut nmatch = 0usize;
        let mut thread_axes_by_dim: BTreeMap<i32, (ThreadEntry, bool)> = BTreeMap::new();
        for launch in &self.thread_extents {
            let var = as_prim_var(&launch.body_params.get(0)?.into())
                .ok_or_else(|| error!(TYPE_ERROR, "Cannot convert to `ir.PrimVar`"))?;
            let tag = thread_tag(launch)?;
            if tag.starts_with("vthread") {
                bail!(INTERNAL_ERROR, "vthread do not work with cross thread reduction");
            }
            let (rank, dim_index) = parse_thread_tag(&tag)?;
            check_cmp!(rank, <=, 1, INTERNAL_ERROR, "e.scope.rank <= 1");
            if dim_index < 0 {
                bail!(INTERNAL_ERROR, "vthread do not work with cross thread reduction");
            }
            if rank != 1 {
                continue;
            }
            let extent = as_prim_expr(&launch.args.get(1)?)?;
            let Some(extent) = extent.as_node::<crate::ir::IntImmObj>() else {
                bail!(INTERNAL_ERROR, "Need constant extent for reduce set {}", var.name.as_str());
            };
            let extent = extent.value.to_i64().and_then(|v| i32::try_from(v).ok());
            let extent =
                extent.ok_or_else(|| error!(INTERNAL_ERROR, "thread extent out of int range"))?;
            let entry = ThreadEntry { rank, dim_index, var: var.clone(), extent };
            let is_reduce = reduce_set.contains(&ObjectIdentity::of(&var));
            nmatch += is_reduce as usize;
            match thread_axes_by_dim.get_mut(&dim_index) {
                None => {
                    thread_axes_by_dim.insert(dim_index, (entry, is_reduce));
                }
                Some((existing, reduce)) => {
                    if existing.extent != entry.extent {
                        bail!(INTERNAL_ERROR, "Incompatible extents for nested bindings of {tag}");
                    }
                    // Fresh bindings of one hardware axis are aliases: the innermost one makes
                    // the indexes, and the axis is reduced when any of them is.
                    *existing = entry;
                    *reduce |= is_reduce;
                }
            }
        }
        let (mut vred, mut vpar) = (Vec::new(), Vec::new());
        for (entry, is_reduce) in thread_axes_by_dim.into_values() {
            if entry.extent != 1 {
                if is_reduce {
                    vred.push(entry)
                } else {
                    vpar.push(entry)
                }
            }
        }
        if nmatch != reduce_set.len() {
            bail!(INTERNAL_ERROR, "Not all reduce index are presented in the context");
        }
        let (reduce_index, reduce_extent) = Self::flatten_thread(&vred)?;
        let (group_index, group_extent) = Self::flatten_thread(&vpar)?;

        // The longest contiguous reduce extent after flattening, over threadIdx in dim order.
        let mut block_threads: Vec<(i32, i32, bool)> = vred
            .iter()
            .filter(|t| t.rank == 1)
            .map(|t| (t.dim_index, t.extent, true))
            .chain(vpar.iter().filter(|t| t.rank == 1).map(|t| (t.dim_index, t.extent, false)))
            .collect();
        block_threads.sort();
        let mut contiguous_reduce_extent = 1;
        for (_, extent, is_reduce) in block_threads {
            if !is_reduce {
                break;
            }
            contiguous_reduce_extent *= extent;
        }

        let mut seq: Vec<Stmt> = Vec::new();
        let mut new_alloc_bufs: Vec<Var> = Vec::new();
        let reduce_dtype = prim_dtype(&reduce_index)?;
        let zero_index: Expr = int_imm(reduce_dtype, 0)?.into();
        if self.is_warp_reduction(&dtypes, group_extent, reduce_extent, contiguous_reduce_extent)? {
            let uint32 = prim_type(DLDataType { code: 1, bits: 32, lanes: 1 })?;
            let mask: Expr = Call::new(uint32, GPU_WARP_ACTIVEMASK.get()?, vec![])?.into();
            let mut reduce_results;
            if reduce_extent as i64 <= self.warp_size {
                let local_bufs;
                (reduce_results, local_bufs) = self.make_warp_allreduce(
                    values,
                    &dtypes,
                    &combiner,
                    &reduce_index,
                    reduce_extent,
                    &group_index,
                    &mask,
                    None,
                    &mut seq,
                )?;
                new_alloc_bufs = local_bufs;
                // Broadcast the result from lane 0 to every lane, so every thread stores the same
                // value without a predicate.
                for (i, result) in reduce_results.iter().enumerate() {
                    let buf = load_source(result)?;
                    let val = tensor_load(&buf, vec![zero_index.clone()])?;
                    check_dtype(&val, dtypes[i])?;
                    let lane =
                        (crate::ops::make_const(prim_dtype(&group_index)?, reduce_extent as i64)?
                            * &group_index)
                            .expr()?;
                    let splat =
                        self.warp_shuffle(&GPU_WARP_SHUFFLE, new_alloc_bufs.last(), val, lane)?;
                    seq.push(tensor_store(&buf, vec![zero_index.clone()], splat)?);
                }
            } else {
                let n_warps = (reduce_extent as i64 / self.warp_size) as i32;
                // 1. The staging tensors in shared memory.
                let mut staging = Vec::with_capacity(size);
                for tensor in &tensors {
                    let shape =
                        vec![int_imm(reduce_dtype, (n_warps * group_extent) as i128)?.into()];
                    let ty = TensorType::new(shape, &tensor_ty(tensor).dtype, self.shared.clone())?;
                    let buf = Var::new("red_buf_staging", ty)?;
                    new_alloc_bufs.push(buf.clone());
                    staging.push(buf);
                }
                // 2. The first round, within each warp.
                let warp_size = self.warp_size as i32;
                let local_bufs;
                (reduce_results, local_bufs) = self.make_warp_allreduce(
                    values,
                    &dtypes,
                    &combiner,
                    &reduce_index,
                    warp_size,
                    &group_index,
                    &mask,
                    None,
                    &mut seq,
                )?;
                new_alloc_bufs.extend(local_bufs);
                // 3. Each warp's result into the staging tensor.
                let mut write_staging = Vec::with_capacity(size);
                for (i, result) in reduce_results.iter().enumerate() {
                    new_alloc_bufs.push(load_source(result)?);
                    let index =
                        (&group_index * n_warps + reduce_index.floordiv(warp_size)).expr()?;
                    write_staging.push(tensor_store(&staging[i], vec![index], result.clone())?);
                }
                let first_lane = reduce_index.floormod(warp_size).equal(&zero_index).expr()?;
                seq.push(If::new(first_lane, SeqStmt::new(write_staging)?, None)?.into());
                seq.push(self.sync_thread("shared")?);
                // 4. The second round, over the warps' results.
                let mut staged = Vec::with_capacity(size);
                for buf in &staging {
                    let index = (&group_index * n_warps + &reduce_index).expr()?;
                    staged.push(tensor_load(buf, vec![index])?);
                }
                let n_warps_imm: Expr = int_imm(reduce_dtype, n_warps as i128)?.into();
                let predicate = reduce_index.less(n_warps_imm).expr()?;
                let local_bufs;
                (reduce_results, local_bufs) = self.make_warp_allreduce(
                    staged,
                    &dtypes,
                    &combiner,
                    &reduce_index,
                    n_warps,
                    &group_index,
                    &mask,
                    Some(predicate),
                    &mut seq,
                )?;
                new_alloc_bufs.extend(local_bufs);
                // 5. The results into shared memory, `group_extent` of them, for every thread.
                let mut write_result = Vec::with_capacity(size);
                for (i, result) in reduce_results.iter_mut().enumerate() {
                    new_alloc_bufs.push(load_source(result)?);
                    let shape = vec![int_imm(reduce_dtype, group_extent as i128)?.into()];
                    let ty =
                        TensorType::new(shape, &tensor_ty(&tensors[i]).dtype, self.shared.clone())?;
                    let broadcast = Var::new("red_result", ty)?;
                    write_result.push(tensor_store(
                        &broadcast,
                        vec![group_index.clone()],
                        result.clone(),
                    )?);
                    *result = tensor_load(&broadcast, vec![group_index.clone()])?;
                }
                let lane_zero = reduce_index.equal(&zero_index).expr()?;
                seq.push(If::new(lane_zero, SeqStmt::new(write_result)?, None)?.into());
                seq.push(self.sync_thread("shared")?);
            }
            // The results replace the destinations' allocations and loads.
            for (i, result) in reduce_results.into_iter().enumerate() {
                let key = self.remaps.allocation_key(&tensors[i]);
                check!(
                    !self.load_remap.contains_key(&key),
                    INTERNAL_ERROR,
                    "!load_remap_.count(alloc_key)"
                );
                let buf = load_source(&result)?;
                check_dtype(&result, dtypes[i])?;
                self.load_remap.insert(key.clone(), result);
                self.remaps.alloc.insert(key.clone(), buf.clone());
                self.remaps.var.insert(key, buf.clone());
                self.remaps.var.insert(ObjectIdentity::of(&tensors[i]), buf);
            }
        } else {
            if reduce_extent == 1 {
                // No reduction is needed.
                let mut stores = Vec::with_capacity(size);
                for (tensor, value) in tensors.iter().zip(values) {
                    stores.push(tensor_store(tensor, vec![crate::ops::int32(0)?.into()], value)?);
                }
                return Ok(SeqStmt::new(stores)?.into());
            }
            // A previous iteration may still be reading the same tensor.
            seq.push(self.sync_thread("shared")?);
            let group_dtype = prim_dtype(&group_index)?;
            let mut shared_bufs = Vec::with_capacity(size);
            for (idx, value) in values.into_iter().enumerate() {
                let shape =
                    vec![int_imm(group_dtype, (group_extent * reduce_extent) as i128)?.into()];
                let ty = TensorType::new(shape, dtypes[idx], self.shared.clone())?;
                let buf = Var::new(format!("red_buf{idx}"), ty)?;
                let index = self.buf_index(&reduce_index, &group_index, reduce_extent)?;
                seq.push(tensor_store(&buf, vec![index], value)?);
                shared_bufs.push(buf);
            }
            seq.push(self.sync_thread("shared")?);
            seq.push(self.make_buf_allreduce(
                &combiner,
                &dtypes,
                &shared_bufs,
                &reduce_index,
                &group_index,
                reduce_extent,
                group_extent,
                contiguous_reduce_extent,
            )?);
            for (idx, buf) in shared_bufs.iter().enumerate() {
                let key = self.remaps.allocation_key(&tensors[idx]);
                check!(
                    !self.load_remap.contains_key(&key),
                    INTERNAL_ERROR,
                    "!load_remap_.count(alloc_key)"
                );
                // C++ also makes an unused `MakeConst(Bool(lanes), true)`, which only fails for a
                // scalable vector.
                lanes(dtypes[idx])?;
                let zero: Expr = int_imm(reduce_dtype, 0)?.into();
                let index = self.buf_index(&zero, &group_index, reduce_extent)?;
                let load = tensor_load(buf, vec![index])?;
                check_dtype(&load, dtypes[idx])?;
                self.load_remap.insert(key.clone(), load);
                self.remaps.alloc.insert(key.clone(), buf.clone());
                self.remaps.var.insert(key, buf.clone());
                self.remaps.var.insert(ObjectIdentity::of(&tensors[idx]), buf.clone());
            }
        }

        // The new local allocations, ahead of the statements.
        let mut stmts: Vec<Stmt> = Vec::with_capacity(new_alloc_bufs.len() + seq.len());
        for buf in &new_alloc_bufs {
            let ty = buf.ty.clone();
            let no_attrs = DictAttrs::new(Map::new());
            let value = Call::new_full(
                Some(ty.clone()),
                ALLOC_TENSOR.get()?.into(),
                Array::new(vec![]),
                Some(no_attrs.into()),
                Array::new(vec![ty]),
                crate::ops::unknown_loc()?,
            )?;
            stmts.push(Bind::new(buf.clone(), value)?.into());
        }
        stmts.extend(seq);
        Ok(SeqStmt::new(stmts)?.into())
    }

    /// `MakeWarpAllreduce`: the reduction of `values` within warps by shuffles down; the loads of
    /// the results, and the local tensors to allocate.
    #[allow(clippy::too_many_arguments)]
    fn make_warp_allreduce(
        &self,
        values: Vec<Expr>,
        dtypes: &[DLDataType],
        combiner: &LambdaExpr,
        reduce_index: &Expr,
        reduce_extent: i32,
        _group_index: &Expr,
        mask: &Expr,
        predicate: Option<Expr>,
        seq: &mut Vec<Stmt>,
    ) -> Result<(Vec<Expr>, Vec<Var>)> {
        let zero = || -> Result<Vec<Expr>> { Ok(vec![crate::ops::int32(0)?.into()]) };
        let shape = || -> Result<Vec<Expr>> { Ok(vec![crate::ops::int32(1)?.into()]) };
        let mut red_bufs = Vec::with_capacity(values.len());
        let mut local_bufs = Vec::with_capacity(values.len());
        let mut load_values = Vec::with_capacity(values.len());
        for (idx, value) in values.into_iter().enumerate() {
            let dtype = prim_type(dtypes[idx])?;
            let red_buf = Var::new(
                format!("red_buf{idx}"),
                TensorType::new(shape()?, &dtype, self.local.clone())?,
            )?;
            load_values.push(tensor_store(&red_buf, zero()?, value)?);
            red_bufs.push(red_buf);
            // The shuffled value, in a local tensor of its own.
            local_bufs.push(Var::new(
                format!("t{idx}"),
                TensorType::new(shape()?, dtype, self.local.clone())?,
            )?);
        }
        match predicate {
            Some(predicate) => {
                seq.push(If::new(predicate, SeqStmt::new(load_values)?, None)?.into())
            }
            None => seq.extend(load_values),
        }
        // The active lanes, cached as the reduction may sit in divergent control flow.
        let mut mask_tensor = None;
        if self.need_warp_shuffle_mask {
            let ty = TensorType::new(shape()?, prim_ty(mask)?, self.local.clone())?;
            let tensor = Var::new("mask", ty)?;
            seq.push(tensor_store(&tensor, zero()?, mask.clone())?);
            local_bufs.push(tensor.clone());
            mask_tensor = Some(tensor);
        }
        let mut start_offset = 1;
        while start_offset * 2 < reduce_extent {
            start_offset *= 2;
        }
        let mut offset = start_offset;
        while offset > 0 {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for (i, red_buf) in red_bufs.iter().enumerate() {
                let val = tensor_load(red_buf, zero()?)?;
                check_dtype(&val, dtypes[i])?;
                a.push(val.clone());
                // The shuffle goes to a local first: a shuffle in an if_then_else would diverge.
                let delta: Expr = crate::ops::int32(offset as i64)?.into();
                let other =
                    self.warp_shuffle(&GPU_WARP_SHUFFLE_DOWN, mask_tensor.as_ref(), val, delta)?;
                seq.push(tensor_store(&local_bufs[i], zero()?, other)?);
                let load = tensor_load(&local_bufs[i], zero()?)?;
                check_dtype(&load, dtypes[i])?;
                b.push(load);
            }
            let combined = apply_combiner(combiner, &a, &b)?;
            let mut stores = Vec::with_capacity(red_bufs.len());
            for (red_buf, value) in red_bufs.iter().zip(combined) {
                stores.push(tensor_store(red_buf, zero()?, value)?);
            }
            // Lanes past the extent hold undefined values: guard them for max or prod.
            if offset * 2 > reduce_extent {
                let in_range = (reduce_index + offset).less(reduce_extent).expr()?;
                seq.push(If::new(in_range, SeqStmt::new(stores)?, None)?.into());
            } else {
                seq.push(SeqStmt::new(stores)?.into());
            }
            offset /= 2;
        }
        let results =
            red_bufs.iter().map(|buf| tensor_load(buf, zero()?)).collect::<Result<_>>()?;
        Ok((results, local_bufs))
    }

    /// `MakeBufAllreduce`: the reduction through shared memory, halving the active lanes.
    #[allow(clippy::too_many_arguments)]
    fn make_buf_allreduce(
        &self,
        combiner: &LambdaExpr,
        dtypes: &[DLDataType],
        shared_bufs: &[Var],
        reduce_index: &Expr,
        group_index: &Expr,
        reduce_extent: i32,
        group_extent: i32,
        contiguous_reduce_extent: i32,
    ) -> Result<Stmt> {
        let mut reduce_align = 1;
        while reduce_extent > reduce_align {
            reduce_align <<= 1;
        }
        check_cmp!(reduce_align, >, 1, INTERNAL_ERROR, "reduce_align > 1");
        let mut seq: Vec<Stmt> = Vec::new();
        let buf_index = self.buf_index(reduce_index, group_index, reduce_extent)?;
        let fload = |offset: i32| -> Result<Vec<Expr>> {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for (i, buf) in shared_bufs.iter().enumerate() {
                let shifted = (reduce_index + offset).expr()?;
                let other = self.buf_index(&shifted, group_index, reduce_extent)?;
                let b_load = tensor_load(buf, vec![other])?;
                check_dtype(&b_load, dtypes[i])?;
                b.push(b_load);
                let a_load = tensor_load(buf, vec![buf_index.clone()])?;
                check_dtype(&a_load, dtypes[i])?;
                a.push(a_load);
            }
            apply_combiner(combiner, &a, &b)
        };
        let fstore = |values: Vec<Expr>| -> Result<Stmt> {
            let mut stores = Vec::with_capacity(values.len());
            for (buf, value) in shared_bufs.iter().zip(values) {
                stores.push(tensor_store(buf, vec![buf_index.clone()], value)?);
            }
            Ok(SeqStmt::new(stores)?.into())
        };
        let freduce = |offset: i32| -> Result<Stmt> { fstore(fload(offset)?) };
        // The lanes past the last power of two below the extent.
        if reduce_align > reduce_extent {
            reduce_align >>= 1;
            let cond = reduce_index.less(reduce_extent - reduce_align).expr()?;
            seq.push(If::new(cond, freduce(reduce_align)?, None)?.into());
            seq.push(self.sync_thread("shared")?);
        }
        // Halving with block synchronization.
        let warp_size = self.warp_size as i32;
        let warp_align = group_extent == 1 || contiguous_reduce_extent % warp_size == 0;
        while reduce_align > contiguous_reduce_extent || reduce_align > warp_size || !warp_align {
            if reduce_align == 1 {
                break;
            }
            reduce_align >>= 1;
            let cond = reduce_index.less(reduce_align).expr()?;
            seq.push(If::new(cond, freduce(reduce_align)?, None)?.into());
            seq.push(self.sync_thread("shared")?);
        }
        // Halving within a warp: the loads are bound before a warp barrier and the stores, as a
        // lane may read the slot another lane writes.
        if reduce_align > 1 {
            let in_warp = reduce_index.less(reduce_align >> 1).expr()?;
            let mut in_warp_seq: Vec<Stmt> = Vec::new();
            while reduce_align > 1 {
                reduce_align >>= 1;
                let loads = fload(reduce_align)?;
                let mut vars = Vec::with_capacity(loads.len());
                for (k, load) in loads.iter().enumerate() {
                    vars.push(Var::new(format!("w_{reduce_align}_{k}"), load.ty.clone())?);
                }
                let mut body: Vec<Stmt> = Vec::with_capacity(loads.len() + 3);
                for (var, load) in vars.iter().zip(loads) {
                    body.push(Bind::new(var.clone(), load)?.into());
                }
                body.push(self.sync_thread("warp")?);
                body.push(fstore(vars.into_iter().map(Expr::from).collect())?);
                body.push(self.sync_thread("warp")?);
                in_warp_seq.push(SeqStmt::new(body)?.into());
            }
            seq.push(If::new(in_warp, SeqStmt::new(in_warp_seq)?, None)?.into());
            seq.push(self.sync_thread("shared")?);
        }
        Ok(SeqStmt::new(seq)?.into())
    }
}

/// The tensor a `TensorLoad` reads.
fn load_source(load: &Expr) -> Result<Var> {
    match load.as_node::<TensorLoadObj>() {
        Some(load) => as_tensor_var(&load.source),
        None => bail!(TYPE_ERROR, "Cannot convert to `ir.TensorLoad`"),
    }
}

fn check_dtype(expr: &Expr, dtype: DLDataType) -> Result<()> {
    if prim_dtype(expr)? != dtype {
        bail!(INTERNAL_ERROR, "Check failed: the reduction keeps its value types");
    }
    Ok(())
}

/// `DeferredRemapper`: in flat IR an allocation precedes the allreduce that remaps it, so the
/// allocations and the declarations over them are rewritten afterwards.
struct DeferredRemapper<'a> {
    remaps: &'a Remaps,
    shared: &'a MemoryScope,
    pending: HashSet<ObjectIdentity>,
}

#[dispatch(mutate)]
impl DeferredRemapper<'_> {
    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        let Some(call) = op.value.as_node::<CallObj>() else { return m.default_mutate(self, &op) };
        if ALLOC_TENSOR.is(&call.op) {
            let key = ObjectIdentity::of(&op.var);
            let node: Bind = m.default_mutate_as(self, &op)?;
            if self.pending.contains(&key) {
                if let Some(replacement) = self.remaps.alloc.get(&key) {
                    return Ok(self.remaps.remap_alloc(&node, replacement, self.shared)?.into());
                }
            }
            return Ok(node.into());
        }
        if DECL_TENSOR.is(&call.op) {
            let root = ObjectIdentity::of(&self.remaps.root(&op.var));
            if self.pending.contains(&root) && self.remaps.alloc.contains_key(&root) {
                return Ok(Evaluate::new(0)?.into());
            }
            let node: Bind = m.default_mutate_as(self, &op)?;
            let Some(tensor) =
                self.remaps.remapped_tensor(&as_tensor_var(&node.var.clone().into())?)
            else {
                return Ok(node.into());
            };
            let call = node.value.as_node::<CallObj>().expect("checked to be a decl_tensor call");
            let ty = tensor.ty.clone();
            let value = Call::new_full(
                Some(ty.clone()),
                DECL_TENSOR.get()?.into(),
                Array::new(vec![call.args.get(0)?]),
                call.attrs.clone(),
                Array::new(vec![ty]),
                call.loc.clone(),
            )?;
            return Ok(Bind::new_full(tensor, value, node.loc.clone())?.into());
        }
        m.default_mutate(self, &op)
    }
}

pub fn lower_thread_allreduce() -> Result<Pass> {
    function_pass("tirx.LowerThreadAllreduce", 0, |func, _module, _context| {
        let target = get_attr(&func.attrs, "target")?;
        check!(
            target.is_some(),
            INTERNAL_ERROR,
            "target.has_value()",
            "LowerThreadAllreduce: Require the target attribute"
        );
        let target: Target = target.expect("checked above").try_into()?;
        let mut builder = ThreadAllreduceBuilder::new(&target, &func.params)?;
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let body: Stmt = structural_mutate(body, &mut builder)?.try_into()?;
        let mut body = SeqStmt::from_stmt(body)?;
        let pending: HashSet<ObjectIdentity> = builder.remaps.pending.iter().cloned().collect();
        if pending.iter().any(|key| builder.remaps.alloc.contains_key(key)) {
            let mut remapper =
                DeferredRemapper { remaps: &builder.remaps, shared: &builder.shared, pending };
            let remapped: Stmt = structural_mutate(body, &mut remapper)?.try_into()?;
            body = SeqStmt::from_stmt(remapped)?;
        }
        Ok(Some(func.with_body(Some(body))))
    })
}
