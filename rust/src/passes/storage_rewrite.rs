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

//! `tirx.StorageRewrite` and `tirx.PointerValueTypeRewrite`
//! (`src/tirx/transform/storage_rewrite.cc`).
//!
//! StorageRewrite plans tensor allocations by liveness. [`LinearAccessPatternFinder`] flattens
//! the body into a sequence of statements, scopes as a begin and an end entry, each recording
//! the allocations it touches; the first and last touches are where an allocation comes alive
//! and dies. [`StoragePlanRewriter`] walks that sequence, reusing freed storage of a compatible
//! size, folding element-wise updates into their source, and packing single-allocation scopes
//! into one buffer, then rewrites the body onto the planned backing tensors.
//!
//! PointerValueTypeRewrite retypes a tensor or pointer that is only ever accessed as one vector
//! type to that vector element type, dividing its indices. StorageRewrite ends with it.

use std::collections::HashMap;

use tvm_ffi::object::ObjectRef;
use tvm_ffi::{
    dispatch, structural_mutate, structural_walk, Any, Array, ContextPolicy, DLDataType,
    DLDataTypeCode, DefRegionKind, Error, FieldGetter, Map, Mutator, ObjectCore, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, String, StructuralView, VisitContext, VisitInterrupt,
    WalkOrder, WalkResult, WalkWithContextPolicy, INTERNAL_ERROR, TYPE_ERROR, VALUE_ERROR,
};

use super::util::{function_pass, get_attr, MutatorExt};
use crate::ir::{
    AssertStmtObj, Attrs, Bind, BindObj, Call, CallObj, DictAttrs, Evaluate, EvaluateObj, Expr,
    For, ForObj, IfObj, IntImmObj, Op, PrimType, PrimTypeObj, PtrType, PtrTypeObj, RegionStmt,
    RegionStmtObj, ReturnObj, SeqStmt, Stmt, StringImm, TensorLoad, TensorLoadObj, TensorStore,
    TensorStoreObj, Type, Var, WhileObj,
};
use crate::ops::{ex, int32, int_imm, make_const, prim_dtype, prim_type, unknown_loc, PrimOps};
use crate::prim::{Let, Ramp, RampObj, Shuffle};
use crate::runtime::{MemoryScope, MemoryScopeObj};
use crate::sym::Analyzer;
use crate::target::Target;
use crate::tirx::{Function, Layout, TensorType, TensorTypeObj};
use crate::transform::Pass;
use crate::SameField;

/// `tvm::tirx::attr::kVolatile`.
const VOLATILE: &str = "tirx.volatile";

const FOR_KIND_PARALLEL: i32 = 1;
const FOR_KIND_VECTORIZED: i32 = 2;

fn internal_error(message: impl AsRef<str>) -> Error {
    Error::new(INTERNAL_ERROR, message.as_ref(), "")
}

/// C++ `TVM_FFI_ICHECK(cond) << message`.
macro_rules! check {
    ($cond:expr, $($message:tt)+) => {
        if !($cond) {
            return Err(internal_error(format!($($message)+)));
        }
    };
}

#[derive(Clone)]
struct Ops {
    alloc_tensor: Op,
    decl_tensor: Op,
    tensor_data_ptr: Op,
    ptr_byte_offset: Op,
    masked_load: Op,
    masked_store: Op,
    address_of: Op,
    launch_thread: Op,
}

impl Ops {
    fn get() -> Result<Self> {
        Ok(Self {
            alloc_tensor: Op::get("tirx.alloc_tensor")?,
            decl_tensor: Op::get("tirx.decl_tensor")?,
            tensor_data_ptr: Op::get("tirx.tensor_data_ptr")?,
            ptr_byte_offset: Op::get("tirx.ptr_byte_offset")?,
            masked_load: Op::get("tirx.masked_load")?,
            masked_store: Op::get("tirx.masked_store")?,
            address_of: Op::get("tirx.address_of")?,
            launch_thread: Op::get("tirx.launch_thread")?,
        })
    }

    /// `expr` as a call to `alloc_tensor` or `decl_tensor`.
    fn tensor_binding<'e>(&self, expr: &'e Expr) -> Option<&'e CallObj> {
        expr.as_node::<CallObj>().filter(|call| {
            call.op.same_as(&self.alloc_tensor) || call.op.same_as(&self.decl_tensor)
        })
    }

    /// C++ `GetBufferDataVar`: the variable `data` is, or the tensor of a `tensor_data_ptr`.
    fn buffer_data_var(&self, data: &Expr) -> Option<Var> {
        if let Ok(var) = data.clone().try_cast::<Var>() {
            return Some(var);
        }
        let call = data.as_node::<CallObj>()?;
        if !call.op.same_as(&self.tensor_data_ptr) || call.args.len() != 1 {
            return None;
        }
        call.args.get(0).ok()?.try_cast::<Var>().ok()
    }
}

// ---------------------------------------------------------------------------------------------
// Types

fn encoded_lanes(dtype: DLDataType) -> i16 {
    dtype.lanes as i16
}

/// `PrimType::lanes()`: the fixed lane count; an error for a scalable vector.
fn lanes(dtype: DLDataType) -> Result<i32> {
    match encoded_lanes(dtype) {
        lanes if lanes < 0 => {
            Err(internal_error("Can't fetch the lanes of a scalable vector at a compile time."))
        }
        lanes => Ok(lanes as i32),
    }
}

fn is_scalable_vector(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) < -1
}

/// `PrimType::WithLanes(lanes)`.
fn with_lanes(dtype: DLDataType, lanes: i32) -> DLDataType {
    DLDataType { code: dtype.code, bits: dtype.bits, lanes: lanes as u16 }
}

fn is_void(dtype: DLDataType) -> bool {
    dtype.code == DLDataTypeCode::kDLOpaqueHandle as u8 && dtype.bits == 0 && dtype.lanes == 0
}

/// A bool element type is stored as `int8`.
fn storage_dtype(dtype: DLDataType) -> Result<DLDataType> {
    if dtype.code != DLDataTypeCode::kDLBool as u8 {
        return Ok(dtype);
    }
    Ok(DLDataType { code: DLDataTypeCode::kDLInt as u8, bits: 8, lanes: lanes(dtype)? as u16 })
}

/// `bits() * lanes()` of a fixed-length type.
fn element_bits(dtype: DLDataType) -> Result<u64> {
    Ok(dtype.bits as u64 * lanes(dtype)? as u64)
}

fn void_dtype() -> DLDataType {
    DLDataType { code: DLDataTypeCode::kDLOpaqueHandle as u8, bits: 0, lanes: 0 }
}

/// `as_or_throw<TensorVar>()`: a `Var` whose type is a tirx `TensorType`.
fn tensor_var(expr: &Expr) -> Result<Var> {
    match expr.clone().try_cast::<Var>() {
        Ok(var) if is_tensor(&var) => Ok(var),
        _ => Err(Error::new(TYPE_ERROR, "expected a tensor variable", "")),
    }
}

fn is_tensor(var: &Var) -> bool {
    var.ty.as_node::<TensorTypeObj>().is_some()
}

fn tensor_type(var: &Var) -> Result<TensorType> {
    var.ty.clone().try_cast()
}

/// The `TensorType` an `alloc_tensor` or `decl_tensor` call declares.
fn declared_type(call: &CallObj) -> Result<TensorType> {
    call.ty_args.get(0)?.try_cast()
}

fn dtype_of(ty: &TensorType) -> DLDataType {
    ty.dtype.dtype
}

/// A copy of `ty` with the dtype, shape and layout replaced (C++ `make_object<TensorTypeNode>(*ty)`
/// and edits).
fn retyped(
    ty: &TensorType,
    dtype: PrimType,
    shape: Array<Expr>,
    layout: Option<Layout>,
) -> TensorType {
    TensorType::from_complete_fields(
        ty.loc.clone(),
        dtype,
        ty.scope.clone(),
        shape,
        ty.strides.clone(),
        ty.elem_offset.clone(),
        ty.data_alignment,
        ty.offset_factor,
        layout,
    )
}

/// `TensorType({extent}, dtype, scope, {}, std::nullopt, data_alignment, offset_factor)`.
fn flat_tensor_type(
    extent: Expr,
    dtype: DLDataType,
    scope: &MemoryScope,
    data_alignment: i32,
    offset_factor: i32,
) -> Result<TensorType> {
    let (scope, loc) = (scope.clone(), unknown_loc()?);
    TensorType::new_full([extent], dtype, scope, [], None, data_alignment, offset_factor, None, loc)
}

/// `TensorTypeNode::ConstantAllocationSize()`: the element count when every extent is constant.
fn constant_allocation_size(ty: &TensorType) -> Option<i64> {
    ty.shape.iter().try_fold(1i64, |size, extent| {
        size.checked_mul(extent.as_node::<IntImmObj>()?.value.to_i64()?)
    })
}

/// `IsConstInt(expr, value)`.
fn is_const_int(expr: &Expr, value: i64) -> bool {
    expr.as_node::<IntImmObj>().is_some_and(|imm| imm.value.to_i64() == Some(value))
}

/// An `IntImm` lane count as C++ `value.as<int>().value()`.
fn int_value(expr: &Expr) -> Result<i32> {
    let imm = expr.as_node::<IntImmObj>().ok_or_else(|| internal_error("expected an IntImm"))?;
    imm.value
        .to_i64()
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| internal_error("an IntImm out of the int range"))
}

/// `ffi::StructuralEqual()(lhs, rhs)`.
fn structural_equal(lhs: &Type, rhs: &Type) -> Result<bool> {
    tvm_ffi::cached_global_func!("ffi.StructuralEqual")
        .call_tuple((lhs.clone(), rhs.clone(), false, false))?
        .try_into()
}

/// `prim::ExprDeepEqual()(lhs, rhs)`.
fn expr_deep_equal(lhs: &Expr, rhs: &Expr) -> Result<bool> {
    tvm_ffi::cached_global_func!("prim.expr_deep_equal")
        .call_tuple((lhs.clone(), rhs.clone()))?
        .try_into()
}

/// `call.attrs` holds the volatile flag.
fn is_volatile(call: &CallObj) -> Result<bool> {
    let Some(attrs) = &call.attrs else { return Ok(false) };
    let attrs: DictAttrs = attrs.clone().try_cast()?;
    Ok(get_attr(&attrs, VOLATILE)?.is_some())
}

/// `ctx->GetConfig<bool>(key)`; C++ `TypeTraits<bool>` also accepts an `int`.
fn config_flag(value: Option<Any>) -> Result<bool> {
    let Some(value) = value else { return Ok(false) };
    if let Some(flag) = value.try_as::<bool>() {
        return Ok(flag);
    }
    Ok(i64::try_from(value)? != 0)
}

/// `runtime::MemoryAllocProperty`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AllocProperty {
    Normal,
    SingleAllocation,
    SeparateAllocations,
}

/// `MemoryScope::alloc_property()`: the scope's `runtime.memory_alloc_property` column in the
/// enum state of `runtime.MemoryScope`, `Normal` when unset.
fn alloc_property(scope: &MemoryScope) -> Result<AllocProperty> {
    let state = tvm_ffi::get_type_attr(MemoryScopeObj::type_index(), "__ffi_enum__")
        .ok_or_else(|| internal_error("runtime.MemoryScope has no enum state"))?;
    let getter = FieldGetter::new(state.type_index(), "attrs")?;
    let state: ObjectRef = state.try_into()?;
    let columns = getter.get_any(&**ObjectRefCore::data(&state))?;
    let count = tvm_ffi::cached_global_func!("ffi.DictCount");
    let get = tvm_ffi::cached_global_func!("ffi.DictGetItem");
    let key = String::from("runtime.memory_alloc_property");
    if i64::try_from(count.call_tuple((columns.clone(), key.clone()))?)? == 0 {
        return Ok(AllocProperty::Normal);
    }
    let column = get.call_tuple((columns, key))?;
    if i64::try_from(count.call_tuple((column.clone(), scope.clone()))?)? == 0 {
        return Ok(AllocProperty::Normal);
    }
    match i64::try_from(get.call_tuple((column, scope.clone()))?)? {
        0 => Ok(AllocProperty::Normal),
        1 => Ok(AllocProperty::SingleAllocation),
        2 => Ok(AllocProperty::SeparateAllocations),
        _ => Err(Error::new(VALUE_ERROR, "Invalid memory allocation property", "")),
    }
}

/// `ffi::Map<Var, Var>` from each tensor to its physical root, flattened as aliases are declared.
#[derive(Default)]
struct TensorAliases(HashMap<ObjectIdentity, (Var, Var)>);

impl TensorAliases {
    fn set(&mut self, var: &Var, root: &Var) {
        self.0.insert(ObjectIdentity::of(var), (var.clone(), root.clone()));
    }

    fn get(&self, var: &Var) -> Option<Var> {
        self.0.get(&ObjectIdentity::of(var)).map(|(_, root)| root.clone())
    }

    /// `Get(var).value_or(var)`.
    fn root(&self, var: &Var) -> Var {
        self.get(var).unwrap_or_else(|| var.clone())
    }

    /// The C++ `RegisterTensorAlias`: `tensor` shares the root of the tensor whose data it
    /// declares, or is a root itself.
    fn register(&mut self, ops: &Ops, tensor: &Var, data: &Expr) -> Result<()> {
        let mut root = tensor.clone();
        if let Some(source) = ops.buffer_data_var(data).filter(is_tensor) {
            let Some(source_root) = self.get(&source) else {
                return Err(internal_error(format!(
                    "Tensor alias source {} must be registered before its DeclTensor alias",
                    source.name.as_str()
                )));
            };
            root = source_root;
        }
        self.set(tensor, &root);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Linear access pattern

/// One point of the linear access sequence: a statement, or the begin or end of a scope.
struct StmtEntry {
    stmt: Stmt,
    /// Non-zero for a scope: positive at its begin entry, negative at its end entry, the
    /// distance to the other entry.
    scope_pair_offset: i64,
    /// The allocations touched, recorded at a leaf or at the end of a scope.
    touched: Vec<ObjectIdentity>,
}

struct AllocEntry {
    num_physical_dimensions: usize,
    /// Scope depth of the allocation.
    level: usize,
    /// The binding that allocates.
    alloc: Bind,
}

/// Linearizes the body: composite statements (loops, branches, regions) become a begin and an
/// end entry, and an access to an allocation counts at the entry of the scope it was allocated
/// in, so its lifetime never ends inside a nested scope.
struct LinearAccessPatternFinder {
    ops: Ops,
    linear_seq: Vec<StmtEntry>,
    alloc_info: HashMap<ObjectIdentity, AllocEntry>,
    aliases: TensorAliases,
    in_thread_env: bool,
    /// Touched allocations of each open scope.
    scope: Vec<Vec<ObjectIdentity>>,
}

impl LinearAccessPatternFinder {
    fn new(ops: Ops, params: &Array<Var>) -> Self {
        let mut aliases = TensorAliases::default();
        for param in params.iter().filter(is_tensor) {
            aliases.set(&param, &param);
        }
        Self {
            ops,
            linear_seq: Vec::new(),
            alloc_info: HashMap::new(),
            aliases,
            in_thread_env: false,
            scope: Vec::new(),
        }
    }

    /// Touch the allocation `buf`, if it is one, at the scope it was allocated in.
    fn touch(&mut self, buf: ObjectIdentity, name: &str) -> Result<Option<usize>> {
        let Some(entry) = self.alloc_info.get(&buf) else { return Ok(None) };
        check!(entry.level < self.scope.len(), " buf={name}");
        let ndim = entry.num_physical_dimensions;
        self.scope[entry.level].push(buf);
        Ok(Some(ndim))
    }

    /// `RecordAccess`: a load from or store to `tensor`.
    fn record_access(&mut self, tensor: &Expr) -> Result<()> {
        let tensor = tensor_var(tensor)?;
        let root = self.aliases.root(&tensor);
        if let Some(ndim) = self.touch(ObjectIdentity::of(&root), tensor.name.as_str())? {
            check!(
                ndim == 1,
                "TensorVar {} is allocated with {ndim} physical dimensions, but is accessed as \
                 having 1 physical dimension",
                tensor.name.as_str()
            );
        }
        Ok(())
    }

    /// A statement whose expressions touched something becomes a leaf entry.
    fn close_statement(&mut self, stmt: Stmt) {
        let touched = self.scope.pop().expect("a statement scope is open");
        if !touched.is_empty() {
            self.linear_seq.push(StmtEntry { stmt, scope_pair_offset: 0, touched });
        }
    }
}

#[dispatch(walk)]
impl LinearAccessPatternFinder {
    /// A direct reference to a tensor counts as a read of its root allocation.
    fn walk_var(&mut self, var: Var, kind: DefRegionKind) -> Result<WalkResult> {
        if kind != DefRegionKind::None {
            return Ok(WalkResult::Advance);
        }
        let root = if is_tensor(&var) { self.aliases.root(&var) } else { var.clone() };
        self.touch(ObjectIdentity::of(&root), var.name.as_str())?;
        Ok(WalkResult::Skip)
    }
}

/// The statements that open a scope or record accesses.
struct AccessScopes;

impl AccessScopes {
    /// `VisitNewScope`: a begin entry, the body, and an end entry carrying the body's touches.
    fn visit_new_scope(
        ctx: &mut VisitContext<'_, LinearAccessPatternFinder>,
        stmt: Stmt,
    ) -> Result<Option<VisitInterrupt>> {
        let state = ctx.state_mut();
        state.scope.push(Vec::new());
        let begin = state.linear_seq.len();
        state.linear_seq.push(StmtEntry {
            stmt: stmt.clone(),
            scope_pair_offset: 0,
            touched: Vec::new(),
        });
        ctx.visit_children()?;
        let state = ctx.state_mut();
        let touched = state.scope.pop().expect("the scope is open");
        let end = state.linear_seq.len();
        check!(end > begin, "a scope's end entry follows its begin entry");
        let offset = (end - begin) as i64;
        state.linear_seq.push(StmtEntry { stmt, scope_pair_offset: -offset, touched });
        state.linear_seq[begin].scope_pair_offset = offset;
        Ok(None)
    }

    /// `VisitBindingValue`: a binding is a leaf; retained metadata of a tensor declaration can be
    /// the only use of an earlier allocation.
    fn visit_binding(
        ctx: &mut VisitContext<'_, LinearAccessPatternFinder>,
        op: Bind,
    ) -> Result<Option<VisitInterrupt>> {
        ctx.state_mut().scope.push(Vec::new());
        if let Some(call) = ctx.state().ops.tensor_binding(&op.value) {
            let attrs = call.attrs.clone();
            ctx.visit(&attrs)?;
        }
        ctx.visit_children()?;
        ctx.state_mut().close_statement(op.into());
        Ok(None)
    }
}

impl ContextPolicy<LinearAccessPatternFinder> for AccessScopes {
    fn default_visit(
        &self,
        value: &StructuralView,
        ctx: &mut VisitContext<'_, LinearAccessPatternFinder>,
    ) -> Result<Option<VisitInterrupt>> {
        // The value first, then the indices; the destination is an access, not a read.
        if let Some(op) = value.cast::<TensorStore>() {
            ctx.state_mut().scope.push(Vec::new());
            ctx.visit(&op.value)?;
            ctx.visit(&op.indices)?;
            ctx.state_mut().record_access(&op.dest)?;
            ctx.state_mut().close_statement(op.into());
            return Ok(None);
        }
        if let Some(op) = value.as_node::<TensorLoadObj>() {
            let (indices, source) = (op.indices.clone(), op.source.clone());
            ctx.visit(&indices)?;
            ctx.state_mut().record_access(&source)?;
            return Ok(None);
        }
        if value.as_node::<EvaluateObj>().is_some() || value.as_node::<ReturnObj>().is_some() {
            let stmt = value.cast::<Stmt>().expect("a statement");
            ctx.state_mut().scope.push(Vec::new());
            ctx.visit_children()?;
            ctx.state_mut().close_statement(stmt);
            return Ok(None);
        }
        // Hardware thread axes share the allocation scope of the outermost one; virtual threads
        // and other regions open their own.
        if let Some(region) = value.cast::<RegionStmt>() {
            if region.op.same_as(&ctx.state().ops.launch_thread) {
                let tag: StringImm = region.args.get(0)?.try_cast()?;
                if !tag.value.as_str().starts_with("vthread") {
                    if ctx.state().in_thread_env {
                        return ctx.visit_children();
                    }
                    ctx.state_mut().in_thread_env = true;
                    let result = Self::visit_new_scope(ctx, region.into());
                    ctx.state_mut().in_thread_env = false;
                    return result;
                }
            }
            return Self::visit_new_scope(ctx, region.into());
        }
        if value.as_node::<IfObj>().is_some()
            || value.as_node::<ForObj>().is_some()
            || value.as_node::<WhileObj>().is_some()
            || value.as_node::<AssertStmtObj>().is_some()
        {
            return Self::visit_new_scope(ctx, value.cast::<Stmt>().expect("a statement"));
        }
        if let Some(op) = value.cast::<Bind>() {
            let ops = ctx.state().ops.clone();
            if let Some(call) = op.value.as_node::<CallObj>() {
                if call.op.same_as(&ops.alloc_tensor) {
                    let ndim = declared_type(call)?.shape.len();
                    let state = ctx.state_mut();
                    let level = state.scope.len();
                    state.aliases.set(&op.var, &op.var);
                    state.alloc_info.insert(
                        ObjectIdentity::of(&op.var),
                        AllocEntry { num_physical_dimensions: ndim, level, alloc: op.clone() },
                    );
                } else if call.op.same_as(&ops.decl_tensor) {
                    let tensor = tensor_var(&op.var.clone().into())?;
                    let data = call.args.get(0)?;
                    ctx.state_mut().aliases.register(&ops, &tensor, &data)?;
                }
            }
            return Self::visit_binding(ctx, op);
        }
        ctx.visit_children()
    }
}

// ---------------------------------------------------------------------------------------------
// In-place detection

/// Whether a statement can run in place, writing `dst` over `src`: it has the form
/// `dst[index] = f(src[index])`, with no other access to either and no indirect loads.
///
/// Like the C++ check, this cannot see a location written more than once
/// (`A[0] = B[0] + 1; A[0] = B[0] + 1`); the code generators producing the IR rule that out.
struct InplaceOpVerifier {
    ops: Ops,
    dst: ObjectIdentity,
    src: ObjectIdentity,
    result: bool,
    /// Depth of loads inside indices; an indirect load such as `A[B[i]]` is not in place.
    mem_nest: i32,
    /// The store to `dst` being inspected.
    store: Option<TensorStore>,
}

impl InplaceOpVerifier {
    /// Run the check on `stmt`, which must be a loop, a branch, or a store.
    fn check(ops: &Ops, stmt: &Stmt, dst: &ObjectIdentity, src: &ObjectIdentity) -> Result<bool> {
        let checked = stmt.as_node::<ForObj>().is_some()
            || stmt.as_node::<IfObj>().is_some()
            || stmt.as_node::<WhileObj>().is_some()
            || stmt.as_node::<TensorStoreObj>().is_some();
        if !checked {
            return Ok(false);
        }
        let verifier = Self {
            ops: ops.clone(),
            dst: dst.clone(),
            src: src.clone(),
            result: true,
            mem_nest: 0,
            store: None,
        };
        let mut walk = WalkWithContextPolicy::new(verifier, InplaceAccesses);
        structural_walk(stmt, &mut walk, WalkOrder::PreOrder)?;
        Ok(walk.into_state().result)
    }

    fn fail(&mut self) -> Option<VisitInterrupt> {
        self.result = false;
        Some(VisitInterrupt::with(Any::new()))
    }
}

#[dispatch(walk)]
impl InplaceOpVerifier {
    /// Any opaque reference to either buffer is unsafe.
    fn walk_var(&mut self, var: Var, kind: DefRegionKind) -> WalkResult {
        if kind != DefRegionKind::None {
            return WalkResult::Advance;
        }
        let id = ObjectIdentity::of(&var);
        if id == self.dst || id == self.src {
            self.result = false;
            return WalkResult::Interrupt;
        }
        WalkResult::Skip
    }
}

struct InplaceAccesses;

impl ContextPolicy<InplaceOpVerifier> for InplaceAccesses {
    fn default_visit(
        &self,
        value: &StructuralView,
        ctx: &mut VisitContext<'_, InplaceOpVerifier>,
    ) -> Result<Option<VisitInterrupt>> {
        if let Some(op) = value.cast::<TensorStore>() {
            ctx.state_mut().mem_nest += 1;
            let interrupt = ctx.visit(&op.indices)?;
            ctx.state_mut().mem_nest -= 1;
            if interrupt.is_some() {
                return Ok(interrupt);
            }
            let dest = tensor_var(&op.dest)?;
            if ObjectIdentity::of(&dest) == ctx.state().dst {
                ctx.state_mut().store = Some(op.clone());
                let interrupt = ctx.visit(&op.value)?;
                ctx.state_mut().store = None;
                return Ok(interrupt);
            }
            return ctx.visit(&op.value);
        }
        if let Some(op) = value.as_node::<BindObj>() {
            let alloc_tensor = &ctx.state().ops.alloc_tensor;
            let volatile = match op.value.as_node::<CallObj>() {
                Some(call) => call.op.same_as(alloc_tensor) && is_volatile(call)?,
                None => false,
            };
            // A volatile allocation is never in place.
            if volatile {
                return Ok(ctx.state_mut().fail());
            }
            return ctx.visit_children();
        }
        if let Some(op) = value.cast::<TensorLoad>() {
            let buf = ObjectIdentity::of(&tensor_var(&op.source)?);
            let state = ctx.state_mut();
            // No reduction into dst, and no indirect load.
            if buf == state.dst || state.mem_nest != 0 {
                return Ok(state.fail());
            }
            if buf == state.src {
                let Some(store) = state.store.clone() else { return Ok(state.fail()) };
                if prim_dtype(&store.value)? != prim_dtype(&op.clone().into())? {
                    return Ok(state.fail());
                }
                check!(
                    store.indices.len() == op.indices.len(),
                    "Store/Load occur to the same buffer with differing number of indices"
                );
                for (stored, loaded) in store.indices.iter().zip(op.indices.iter()) {
                    if !expr_deep_equal(&stored, &loaded)? {
                        return Ok(state.fail());
                    }
                }
            }
            ctx.state_mut().mem_nest += 1;
            let interrupt = ctx.visit(&op.indices)?;
            ctx.state_mut().mem_nest -= 1;
            return Ok(interrupt);
        }
        ctx.visit_children()
    }
}

// ---------------------------------------------------------------------------------------------
// Storage planning

/// A planned backing allocation, shared by the allocations assigned to it.
struct StorageEntry {
    /// The scope this allocation is placed at the start of: a thread or parallel region, or
    /// `None` for the function body.
    attach_scope: Option<ObjectIdentity>,
    /// The size in bits when constant, else 0.
    const_nbits: u64,
    scope: MemoryScope,
    allocs: Vec<Bind>,
    /// Entries packed into this one (single-allocation scopes).
    merged_children: Vec<usize>,
    /// The replacement allocation, placed at the attach scope.
    alloc_nest: Vec<Stmt>,
    /// The backing tensor.
    alloc_var: Option<Var>,
    /// The scalar element type.
    elem_type: DLDataType,
    is_volatile: bool,
    /// Explicit placement or opaque metadata ties this to its own allocation.
    requires_exact_allocation: bool,
    /// Offset in bits into the backing tensor of the entry this one is packed into.
    bits_offset: u64,
    packing_alignment_bits: u64,
}

#[derive(Default)]
struct EventEntry {
    gen: Vec<ObjectIdentity>,
    kill: Vec<ObjectIdentity>,
}

/// Rewrites and merges allocations: tensors with disjoint lifetimes share storage, in-place
/// updates write over their source, and single-allocation scopes pack into one buffer that the
/// code generator casts from.
struct StoragePlanRewriter {
    ops: Ops,
    detect_inplace: bool,
    thread_scope: Option<ObjectIdentity>,
    thread_scope_stack: Vec<Option<ObjectIdentity>>,
    event_map: HashMap<ObjectIdentity, EventEntry>,
    /// Freed constant-size entries by size; equal sizes in the order freed (`std::multimap`).
    const_free_map: Vec<(u64, usize)>,
    /// Freed symbolic-size entries.
    sym_free_list: Vec<usize>,
    attach_map: HashMap<Option<ObjectIdentity>, Vec<usize>>,
    /// The entry of each allocated tensor.
    alloc_map: HashMap<ObjectIdentity, (Var, usize)>,
    alloc_vec: Vec<StorageEntry>,
    /// The backing tensor each remapped tensor view uses.
    remapped_backing: HashMap<ObjectIdentity, Var>,
    /// Tensor views onto the planned storage, replacing the original tensors everywhere.
    remap: HashMap<ObjectIdentity, Var>,
    aliases: TensorAliases,
    analyzer: Analyzer,
}

/// `RequiresExactAllocation`: placement and opaque metadata that merged or resized storage
/// cannot carry, or a physical view that flat packing and reuse cannot reconstruct.
fn requires_exact_allocation(call: &CallObj) -> Result<bool> {
    let ty = declared_type(call)?;
    if alloc_property(&ty.scope)? == AllocProperty::SeparateAllocations {
        return Ok(true);
    }
    if !is_simple_flat(&ty) {
        return Ok(true);
    }
    if !call.args.is_empty() {
        return Ok(true);
    }
    let attrs = call.attrs.clone().ok_or_else(|| internal_error("bad optional access"))?;
    let attrs: DictAttrs = attrs.try_cast()?;
    Ok(attrs.dict.iter().any(|(key, _)| key.as_str() != VOLATILE))
}

/// One extent, no layout, no element offset, and compact strides.
fn is_simple_flat(ty: &TensorType) -> bool {
    ty.shape.len() == 1
        && ty.layout.is_none()
        && is_const_int(&ty.elem_offset, 0)
        && (ty.strides.is_empty()
            || (ty.strides.len() == 1
                && ty.strides.get(0).is_ok_and(|stride| is_const_int(&stride, 1))))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl StoragePlanRewriter {
    fn new(ops: Ops) -> Result<Self> {
        Ok(Self {
            ops,
            detect_inplace: false,
            thread_scope: None,
            thread_scope_stack: Vec::new(),
            event_map: HashMap::new(),
            const_free_map: Vec::new(),
            sym_free_list: Vec::new(),
            attach_map: HashMap::new(),
            alloc_map: HashMap::new(),
            alloc_vec: Vec::new(),
            remapped_backing: HashMap::new(),
            remap: HashMap::new(),
            aliases: TensorAliases::default(),
            analyzer: Analyzer::new()?,
        })
    }

    fn rewrite(
        mut self,
        body: SeqStmt,
        params: &Array<Var>,
        detect_inplace: bool,
        enable_reuse: bool,
        reuse_require_exact_matched_dtype: bool,
    ) -> Result<SeqStmt> {
        self.detect_inplace = detect_inplace;
        let finder = LinearAccessPatternFinder::new(self.ops.clone(), params);
        let mut walk = WalkWithContextPolicy::new(finder, AccessScopes);
        structural_walk(&body, &mut walk, WalkOrder::PreOrder)?;
        let finder = walk.into_state();
        self.liveness_analysis(&finder.linear_seq);
        self.plan_memory(
            &finder.linear_seq,
            &finder.alloc_info,
            enable_reuse,
            reuse_require_exact_matched_dtype,
        )?;
        self.aliases = finder.aliases;
        self.prepare_new_alloc()?;
        let allocated: Vec<(Var, usize)> = self.alloc_map.values().cloned().collect();
        for (var, e) in allocated {
            let backing = self.alloc_var(e)?;
            self.remap_tensor(&var, &backing)?;
        }
        let aliased: Vec<(Var, Var)> = self.aliases.0.values().cloned().collect();
        for (var, root) in aliased {
            if let Some(&(_, e)) = self.alloc_map.get(&ObjectIdentity::of(&root)) {
                let backing = self.alloc_var(e)?;
                self.remap_tensor(&tensor_var(&var.into())?, &backing)?;
            }
        }
        let body: SeqStmt = structural_mutate(body, &mut self)?.try_into()?;
        match self.attach_map.get(&None) {
            Some(entries) => self.make_attach(entries, body),
            None => Ok(body),
        }
    }

    fn alloc_var(&self, e: usize) -> Result<Var> {
        self.alloc_vec[e].alloc_var.clone().ok_or_else(|| internal_error("an unplanned allocation"))
    }

    /// The allocations of `entries` placed before `body`, outermost first.
    fn make_attach(&self, entries: &[usize], body: SeqStmt) -> Result<SeqStmt> {
        let mut body = body;
        for &e in entries.iter().rev() {
            let nest = &self.alloc_vec[e].alloc_nest;
            if !nest.is_empty() {
                body = SeqStmt::new(nest.iter().cloned().chain([body.into()]))?;
            }
        }
        Ok(body)
    }

    /// A view of `buf` onto `backing`: `buf` itself when it is the backing tensor, else a tensor
    /// of `buf`'s type named after the backing one. Every use of `buf` becomes the view.
    fn remap_tensor(&mut self, buf: &Var, backing: &Var) -> Result<Var> {
        if let Some(mapped) = self.remap.get(&ObjectIdentity::of(buf)) {
            let previous = &self.remapped_backing[&ObjectIdentity::of(mapped)];
            check!(
                previous.same_as(backing),
                "Cannot remap tensor {} to use backing array {}, previously used backing array {}",
                buf.name.as_str(),
                backing.name.as_str(),
                previous.name.as_str()
            );
            return Ok(mapped.clone());
        }
        tensor_type(backing)?;
        let remapped = if buf.same_as(backing) {
            buf.clone()
        } else {
            let ty = tensor_type(buf)?;
            let copy = retyped(&ty, ty.dtype.clone(), ty.shape.clone(), ty.layout.clone());
            Var::new_full(backing.name.clone(), copy, buf.loc.clone())?
        };
        self.remap.insert(ObjectIdentity::of(buf), remapped.clone());
        self.remapped_backing.insert(ObjectIdentity::of(&remapped), backing.clone());
        Ok(remapped)
    }

    /// The last index shifted by the entry's offset into its backing tensor.
    fn remap_indices(&self, tensor: &Var, indices: &Array<Expr>, e: usize) -> Result<Array<Expr>> {
        let bits_offset = self.alloc_vec[e].bits_offset;
        let mut indices: Vec<Expr> = indices.iter().collect();
        let last = indices.pop().ok_or_else(|| internal_error("an access without indices"))?;
        if bits_offset == 0 {
            indices.push(last);
            return Ok(Array::new(indices));
        }
        let elem_bits = dtype_of(&tensor_type(tensor)?).bits as u64;
        check!(bits_offset.is_multiple_of(elem_bits), "the offset is a whole number of elements");
        let offset = make_const(prim_dtype(&last)?, (bits_offset / elem_bits) as i64)?;
        indices.push((ex(offset) + &last).expr()?);
        Ok(Array::new(indices))
    }

    /// `VisitTensorAccess` for a store: onto the backing tensor, at the entry's offset.
    fn remap_store(&mut self, node: TensorStore) -> Result<TensorStore> {
        let dest = tensor_var(&node.dest)?;
        let root = self.aliases.root(&dest);
        let Some(&(_, e)) = self.alloc_map.get(&ObjectIdentity::of(&root)) else {
            return Ok(node);
        };
        let backing = self.alloc_var(e)?;
        let buf = self.remap_tensor(&dest, &backing)?;
        let indices = self.remap_indices(&dest, &node.indices, e)?;
        let (loc, value) = (node.loc.clone(), node.value.clone());
        Ok(TensorStore::from_complete_fields(loc, buf.into(), indices, value))
    }

    /// `VisitTensorAccess` for a load.
    fn remap_load(&mut self, node: TensorLoad) -> Result<TensorLoad> {
        let tensor = tensor_var(&node.source)?;
        let root = self.aliases.root(&tensor);
        let Some(&(_, e)) = self.alloc_map.get(&ObjectIdentity::of(&root)) else {
            return Ok(node);
        };
        let backing = self.alloc_var(e)?;
        let remapped = self.remap_tensor(&tensor, &backing)?;
        let indices = self.remap_indices(&tensor, &node.indices, e)?;
        TensorLoad::new_full(remapped, indices.iter(), node.loc.clone())
    }

    /// `alloc_tensor` binding of `buf`, volatile if asked.
    fn alloc_binding(&self, buf: &Var, is_volatile: bool) -> Result<Stmt> {
        let ty = buf.ty.clone();
        let mut annotations: Vec<(String, Any)> = Vec::new();
        if is_volatile {
            annotations.push((String::from(VOLATILE), Any::from(true)));
        }
        let attrs: Attrs =
            DictAttrs::new(annotations.into_iter().collect::<Map<String, Any>>()).into();
        let call = Call::new_full(
            Some(ty.clone()),
            self.ops.alloc_tensor.clone().into(),
            Array::new(vec![]),
            Some(attrs),
            Array::new(vec![ty]),
            unknown_loc()?,
        )?;
        Ok(Bind::new(buf.clone(), call)?.into())
    }

    // -- Liveness -------------------------------------------------------------------------------

    /// The gen point of each allocation is its first touch, the kill point its last.
    fn liveness_analysis(&mut self, seq: &[StmtEntry]) {
        let mut touched: std::collections::HashSet<ObjectIdentity> = Default::default();
        for s in seq.iter().rev() {
            for buffer in &s.touched {
                if touched.insert(buffer.clone()) {
                    self.event_map
                        .entry(ObjectIdentity::of(&s.stmt))
                        .or_default()
                        .kill
                        .push(buffer.clone());
                }
            }
        }
        touched.clear();
        for (i, entry) in seq.iter().enumerate() {
            if entry.scope_pair_offset < 0 {
                continue;
            }
            // A scope's touches are recorded at its end entry, its gen point is its begin.
            let s = &seq[(i as i64 + entry.scope_pair_offset) as usize];
            for buffer in &s.touched {
                if touched.insert(buffer.clone()) {
                    self.event_map
                        .entry(ObjectIdentity::of(&s.stmt))
                        .or_default()
                        .gen
                        .push(buffer.clone());
                }
            }
        }
    }

    fn plan_new_scope(&mut self, op: ObjectIdentity) -> Result<()> {
        if self.thread_scope.as_ref() == Some(&op) {
            // Storage attached to the closing scope cannot be reused outside it.
            let alloc_vec = &self.alloc_vec;
            let attached = |e: &usize| alloc_vec[*e].attach_scope.as_ref() == Some(&op);
            self.const_free_map.retain(|(_, e)| !attached(e));
            self.sym_free_list.retain(|e| !attached(e));
            check!(!self.thread_scope_stack.is_empty(), "a thread scope is open");
            self.thread_scope = self.thread_scope_stack.pop().expect("checked");
        } else {
            self.thread_scope_stack.push(self.thread_scope.take());
            self.thread_scope = Some(op);
        }
        Ok(())
    }

    fn plan_memory(
        &mut self,
        seq: &[StmtEntry],
        alloc_info: &HashMap<ObjectIdentity, AllocEntry>,
        enable_reuse: bool,
        reuse_require_exact_matched_dtype: bool,
    ) -> Result<()> {
        let mut inplace_flag: std::collections::HashSet<ObjectIdentity> = Default::default();
        for s in seq {
            let stmt_id = ObjectIdentity::of(&s.stmt);
            let (gen, kill) = match self.event_map.get(&stmt_id) {
                Some(event) => (Some(event.gen.clone()), event.kill.clone()),
                None => (None, Vec::new()),
            };
            // A leaf or the begin of a scope: allocations come alive.
            if let (Some(gen), true) = (&gen, s.scope_pair_offset >= 0) {
                let detect_inplace = self.detect_inplace && gen.len() <= 2;
                for var in gen {
                    let entry = alloc_info
                        .get(var)
                        .ok_or_else(|| internal_error("a touched tensor has an allocation"))?;
                    let alloc = &entry.alloc;
                    let call = alloc.value.as_node::<CallObj>().expect("an alloc_tensor call");
                    let ty = declared_type(call)?;
                    let scope = ty.scope.clone();
                    let mut dst_entry = None;
                    let var_size = constant_allocation_size(&tensor_type(&alloc.var)?).unwrap_or(0);
                    if detect_inplace
                        && !requires_exact_allocation(call)?
                        && (alloc_property(&scope)? != AllocProperty::SingleAllocation
                            || var_size > 0)
                    {
                        let element_type = dtype_of(&ty);
                        let mut inplace_found = false;
                        for src in &kill {
                            if inplace_flag.contains(src) {
                                continue;
                            }
                            let Some(&(_, src_entry)) = self.alloc_map.get(src) else { continue };
                            let candidate = &self.alloc_vec[src_entry];
                            if !candidate.requires_exact_allocation
                                && self.has_compatible_alignment(src_entry, call)?
                                && candidate.scope.same_as(&scope)
                                && candidate.attach_scope == self.thread_scope
                                && !is_scalable_vector(element_type)
                                && candidate.elem_type == with_lanes(element_type, 1)
                                && InplaceOpVerifier::check(&self.ops, &s.stmt, var, src)?
                            {
                                let const_nbits = (var_size as u64)
                                    .wrapping_mul(element_type.bits as u64)
                                    .wrapping_mul(lanes(element_type)? as u64);
                                if self.alloc_vec[src_entry].const_nbits == const_nbits
                                    && !inplace_found
                                {
                                    dst_entry = Some(src_entry);
                                    inplace_flag.insert(src.clone());
                                    inplace_found = true;
                                }
                            }
                        }
                    }
                    let dst_entry = match dst_entry {
                        Some(e) => e,
                        None => self.find_alloc(
                            alloc,
                            self.thread_scope.clone(),
                            &scope,
                            entry.num_physical_dimensions,
                            enable_reuse,
                            reuse_require_exact_matched_dtype,
                        )?,
                    };
                    self.alloc_vec[dst_entry].allocs.push(alloc.clone());
                    let attrs =
                        call.attrs.clone().ok_or_else(|| internal_error("bad optional access"))?;
                    let attrs: DictAttrs = attrs.try_cast()?;
                    if get_attr(&attrs, VOLATILE)?.is_some() {
                        self.alloc_vec[dst_entry].is_volatile = true;
                    }
                    self.alloc_map.insert(var.clone(), (alloc.var.clone(), dst_entry));
                }
            }
            // Enter or exit a thread or parallel scope.
            if s.stmt.as_node::<RegionStmtObj>().is_some() {
                self.plan_new_scope(stmt_id.clone())?;
            } else if let Some(op) = s.stmt.as_node::<ForObj>() {
                if op.kind == FOR_KIND_PARALLEL
                    && (self.thread_scope.is_none() || self.thread_scope.as_ref() == Some(&stmt_id))
                {
                    self.plan_new_scope(stmt_id.clone())?;
                }
            }
            // A leaf or the end of a scope: allocations die, unless reused in place.
            if gen.is_some() && s.scope_pair_offset <= 0 {
                for var in &kill {
                    if !inplace_flag.contains(var) {
                        self.free(var)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Lifetime reuse keeps the first allocation's alignment contract.
    fn has_compatible_alignment(&self, e: usize, call: &CallObj) -> Result<bool> {
        let first_call = self.alloc_vec[e].allocs[0].value.as_node::<CallObj>().expect("a call");
        let first = declared_type(first_call)?;
        let ty = declared_type(call)?;
        Ok(first.data_alignment == ty.data_alignment && first.offset_factor == ty.offset_factor)
    }

    fn new_alloc(
        &mut self,
        op: &Bind,
        attach_scope: Option<ObjectIdentity>,
        scope: &MemoryScope,
        const_nbits: u64,
    ) -> Result<usize> {
        let call = op.value.as_node::<CallObj>().expect("an alloc_tensor call");
        let element_type = dtype_of(&declared_type(call)?);
        self.alloc_vec.push(StorageEntry {
            attach_scope,
            const_nbits,
            scope: scope.clone(),
            allocs: Vec::new(),
            merged_children: Vec::new(),
            alloc_nest: Vec::new(),
            alloc_var: None,
            elem_type: with_lanes(element_type, 1),
            is_volatile: false,
            requires_exact_allocation: requires_exact_allocation(call)?,
            bits_offset: 0,
            packing_alignment_bits: 32,
        });
        Ok(self.alloc_vec.len() - 1)
    }

    /// `std::multimap::lower_bound`.
    fn free_lower_bound(&self, nbits: u64) -> usize {
        self.const_free_map.partition_point(|(size, _)| *size < nbits)
    }

    /// `std::multimap::upper_bound`.
    fn free_upper_bound(&self, nbits: u64) -> usize {
        self.const_free_map.partition_point(|(size, _)| *size <= nbits)
    }

    /// Reuse a freed entry of the same scope and a close size, or allocate a new one.
    fn find_alloc(
        &mut self,
        op: &Bind,
        attach_scope: Option<ObjectIdentity>,
        scope: &MemoryScope,
        num_physical_dimensions: usize,
        enable_reuse: bool,
        reuse_require_exact_matched_dtype: bool,
    ) -> Result<usize> {
        let call = op.value.as_node::<CallObj>().expect("an alloc_tensor call");
        let element_type = dtype_of(&declared_type(call)?);
        const MATCH_RANGE: u64 = 16;
        let is_scalable = is_scalable_vector(element_type);
        let op_elem_bits = if is_scalable { 0 } else { element_bits(element_type)? };
        let const_size = constant_allocation_size(&tensor_type(&op.var)?).unwrap_or(0);
        let const_nbits =
            if is_scalable { 0 } else { (const_size as u64).wrapping_mul(op_elem_bits) };
        // A size unknown at compile time needs its own runtime-sized allocation.
        let is_known_size = const_nbits != 0;
        // Only flat memory can be reused; N-d spaces need more planning.
        let is_flat_memory_space = num_physical_dimensions == 1;
        let property = alloc_property(scope)?;
        // Small arrays of ordinary memory are left to register allocation.
        let is_small_array =
            property == AllocProperty::Normal && is_known_size && const_nbits <= 32;
        if requires_exact_allocation(call)?
            || is_scalable
            || !enable_reuse
            || is_small_array
            || !is_flat_memory_space
            || (property == AllocProperty::SingleAllocation && !is_known_size)
        {
            return self.new_alloc(op, attach_scope, scope, const_nbits);
        }
        if is_known_size {
            let begin = self.free_lower_bound(const_nbits / MATCH_RANGE);
            let mid = self.free_lower_bound(const_nbits);
            let end = self.free_upper_bound(const_nbits.wrapping_mul(MATCH_RANGE));
            // Larger freed buffers first.
            for it in mid..end {
                let e = self.const_free_map[it].1;
                let entry = &self.alloc_vec[e];
                if entry.attach_scope != attach_scope
                    || !entry.scope.same_as(scope)
                    || !self.has_compatible_alignment(e, call)?
                    // A buffer of float3 cannot host float4.
                    || !entry.bits_offset.is_multiple_of(op_elem_bits)
                    || (reuse_require_exact_matched_dtype && entry.elem_type != element_type)
                {
                    continue;
                }
                let entry = &mut self.alloc_vec[e];
                entry.const_nbits = entry.const_nbits.max(const_nbits);
                self.const_free_map.remove(it);
                return Ok(e);
            }
            // Then smaller ones, nearest first.
            for it in (begin..mid).rev() {
                let e = self.const_free_map[it].1;
                let entry = &self.alloc_vec[e];
                if entry.attach_scope != attach_scope
                    || !entry.scope.same_as(scope)
                    || !self.has_compatible_alignment(e, call)?
                    || entry.elem_type != with_lanes(element_type, 1)
                    || (reuse_require_exact_matched_dtype && entry.elem_type != element_type)
                {
                    continue;
                }
                let entry = &mut self.alloc_vec[e];
                entry.const_nbits = entry.const_nbits.max(const_nbits);
                self.const_free_map.remove(it);
                return Ok(e);
            }
        } else {
            // Symbolic sizes: round robin.
            for it in 0..self.sym_free_list.len() {
                let e = self.sym_free_list[it];
                let entry = &self.alloc_vec[e];
                if entry.attach_scope != attach_scope
                    || !entry.scope.same_as(scope)
                    || !self.has_compatible_alignment(e, call)?
                    || entry.elem_type != with_lanes(element_type, 1)
                {
                    continue;
                }
                self.sym_free_list.remove(it);
                return Ok(e);
            }
        }
        self.new_alloc(op, attach_scope, scope, const_nbits)
    }

    /// Simulated free: the entry becomes available for reuse.
    fn free(&mut self, var: &ObjectIdentity) -> Result<()> {
        let &(_, e) = self.alloc_map.get(var).ok_or_else(|| internal_error("a freed tensor"))?;
        let entry = &self.alloc_vec[e];
        check!(!entry.allocs.is_empty(), "a freed entry has allocations");
        if entry.requires_exact_allocation {
            return Ok(());
        }
        let call = entry.allocs[0].value.as_node::<CallObj>().expect("an alloc_tensor call");
        if is_scalable_vector(dtype_of(&declared_type(call)?)) {
            return Ok(());
        }
        let property = alloc_property(&entry.scope)?;
        if property == AllocProperty::Normal && entry.const_nbits > 0 && entry.const_nbits <= 32 {
            return Ok(());
        }
        if property == AllocProperty::SingleAllocation && entry.const_nbits == 0 {
            return Ok(());
        }
        if entry.const_nbits != 0 {
            let at = self.free_upper_bound(entry.const_nbits);
            self.const_free_map.insert(at, (entry.const_nbits, e));
        } else {
            self.sym_free_list.push(e);
        }
        Ok(())
    }

    // -- Packing and new allocations ------------------------------------------------------------

    /// Packing needs constant flat layouts for every member of the scope group.
    fn can_pack_allocation(&self, e: usize) -> Result<bool> {
        let entry = &self.alloc_vec[e];
        if entry.requires_exact_allocation
            || entry.const_nbits == 0
            || entry.const_nbits > i64::MAX as u64
        {
            return Ok(false);
        }
        for alloc in &entry.allocs {
            let ty = declared_type(alloc.value.as_node::<CallObj>().expect("a call"))?;
            if is_scalable_vector(dtype_of(&ty)) || !is_simple_flat(&ty) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The alignment, in bits, every member of the `scope` group keeps when packed, if they can
    /// be packed and fit.
    fn packing_alignment(&self, entries: &[usize], scope: &MemoryScope) -> Result<Option<u64>> {
        let mut alignment: u64 = 32;
        let mut include = |value: u64| {
            let factor = value / gcd(alignment, value);
            // TensorType records alignment in bytes as an int.
            let limit = i32::MAX as u64 * 8;
            if factor > limit / alignment {
                return false;
            }
            alignment *= factor;
            true
        };
        for &e in entries {
            if !self.alloc_vec[e].scope.same_as(scope) {
                continue;
            }
            if !self.can_pack_allocation(e)? {
                return Ok(None);
            }
            for alloc in &self.alloc_vec[e].allocs {
                let ty = declared_type(alloc.value.as_node::<CallObj>().expect("a call"))?;
                let bits = element_bits(dtype_of(&ty))?;
                if !include(bits)
                    || !include(bits.wrapping_mul(ty.offset_factor.max(1) as u64))
                    || !include(ty.data_alignment.max(1) as u64 * 8)
                {
                    return Ok(None);
                }
            }
        }
        let mut total_bits: u64 = 0;
        let limit = i64::MAX as u64;
        for &e in entries {
            let entry = &self.alloc_vec[e];
            if !entry.scope.same_as(scope) {
                continue;
            }
            let padding = (alignment - entry.const_nbits % alignment) % alignment;
            if entry.const_nbits > limit - padding
                || total_bits > limit - entry.const_nbits - padding
            {
                return Ok(None);
            }
            total_bits += entry.const_nbits + padding;
        }
        Ok(Some(alignment))
    }

    fn prepare_new_alloc(&mut self) -> Result<()> {
        for e in 0..self.alloc_vec.len() {
            let attach = self.alloc_vec[e].attach_scope.clone();
            self.attach_map.entry(attach).or_default().push(e);
        }
        let groups: Vec<Vec<usize>> = self.attach_map.values().cloned().collect();
        for vec in groups {
            // Single-allocation scopes pack into the first entry of their scope.
            for (i, &e) in vec.iter().enumerate() {
                let scope = self.alloc_vec[e].scope.clone();
                if alloc_property(&scope)? != AllocProperty::SingleAllocation {
                    continue;
                }
                let Some(alignment) = self.packing_alignment(&vec, &scope)? else { continue };
                self.alloc_vec[e].packing_alignment_bits = alignment;
                if let Some(&parent) =
                    vec[..i].iter().find(|&&j| self.alloc_vec[j].scope.same_as(&scope))
                {
                    self.alloc_vec[parent].merged_children.push(e);
                }
            }
            for &e in &vec {
                self.allocate(e)?;
            }
        }
        Ok(())
    }

    /// Plan the backing allocation of entry `e`.
    fn allocate(&mut self, e: usize) -> Result<()> {
        let entry = &self.alloc_vec[e];
        if entry.requires_exact_allocation {
            check!(entry.allocs.len() == 1, "an exact allocation is not shared");
            check!(entry.merged_children.is_empty(), "an exact allocation is not packed");
            // The original binding stays in place, metadata included.
            self.alloc_vec[e].alloc_var = Some(entry.allocs[0].var.clone());
            return Ok(());
        }
        // Already packed into another entry.
        if entry.bits_offset != 0 {
            return Ok(());
        }
        if !entry.merged_children.is_empty() {
            return self.new_alloc_merged(e);
        }
        let first = entry.allocs[0].clone();
        let first_call = first.value.as_node::<CallObj>().expect("an alloc_tensor call");
        let first_type = declared_type(first_call)?;
        let first_dtype = dtype_of(&first_type);
        let is_volatile = entry.is_volatile;
        if entry.allocs.len() == 1 && is_scalable_vector(first_dtype) {
            // Runtime-dependent lanes: keep the allocation exact.
            self.alloc_vec[e].alloc_var = Some(first.var.clone());
            let buf = self.remap_tensor(&first.var, &first.var)?;
            let binding = self.alloc_binding(&buf, is_volatile)?;
            self.alloc_vec[e].alloc_nest.push(binding);
            return Ok(());
        }
        let allocs = entry.allocs.clone();
        let mut alloc_type = first_dtype;
        let mut types: Vec<TensorType> = Vec::new();
        for op in &allocs {
            let ty = declared_type(op.value.as_node::<CallObj>().expect("a call"))?;
            if lanes(dtype_of(&ty))? > lanes(alloc_type)? {
                alloc_type = dtype_of(&ty);
            }
            types.push(ty);
        }
        self.alloc_vec[e].alloc_var = Some(first.var.clone());
        let mut all_identical = true;
        for ty in &types[1..] {
            if dtype_of(ty) != first_dtype || ty.shape.len() != first_type.shape.len() {
                all_identical = false;
                break;
            }
            let mut same_shape = true;
            for (extent, first_extent) in ty.shape.iter().zip(first_type.shape.iter()) {
                if !expr_deep_equal(&extent, &first_extent)? {
                    same_shape = false;
                    break;
                }
            }
            if !same_shape {
                all_identical = false;
                break;
            }
        }
        if all_identical {
            // The first allocation, hoisted to the attach scope.
            let buf = self.remap_tensor(&first.var, &first.var)?;
            let binding = self.alloc_binding(&buf, is_volatile)?;
            self.alloc_vec[e].alloc_nest.push(binding);
            return Ok(());
        }
        // One allocation large enough for every member, in units of the widest element.
        let mut max_size: Option<Expr> = None;
        for (op, ty) in allocs.iter().zip(&types) {
            check!(
                ty.shape.len() == 1,
                "TensorVar var {} was identified as a re-usable allocation, but has {} physical \
                 dimensions.  Currently, only flat 1-d memory spaces should be identified as \
                 re-usable allocations.",
                op.var.name.as_str(),
                ty.shape.len()
            );
            let mut size = ty.shape.get(0)?;
            let dtype = dtype_of(ty);
            let nbits = dtype.bits as i32 * lanes(dtype)?;
            if let Some(imm) = size.as_node::<IntImmObj>() {
                if imm.value > tvm_ffi::BigInt::from((i32::MAX / nbits) as i64) {
                    let int64 =
                        DLDataType { code: DLDataTypeCode::kDLInt as u8, bits: 64, lanes: 1 };
                    size = int_imm(int64, imm.value.to_i128().unwrap_or(i128::MAX))?.into();
                }
            }
            let size_nbits = (ex(&size) * nbits).expr()?;
            max_size = Some(match max_size {
                Some(max_size) => max_size.max(&size_nbits).expr()?,
                None => size_nbits,
            });
        }
        let combo_size = max_size.expect("an entry has allocations");
        let type_bits = alloc_type.bits as i32 * lanes(alloc_type)?;
        let divided =
            self.analyzer.can_prove(&combo_size.indexmod(type_bits).equal(0).expr()?, 0)?;
        let mut combo_size = combo_size.indexdiv(type_bits);
        if !divided {
            combo_size += int32(1)?;
        }
        let combo_size = self.analyzer.simplify(&combo_size.expr()?, 2)?;
        let entry = &self.alloc_vec[e];
        let ty = flat_tensor_type(
            combo_size,
            alloc_type,
            &entry.scope,
            first_type.data_alignment,
            first_type.offset_factor,
        )?;
        let buf = Var::new(first.var.name.clone(), ty)?;
        let binding = self.alloc_binding(&buf, is_volatile)?;
        let entry = &mut self.alloc_vec[e];
        entry.alloc_var = Some(buf);
        entry.alloc_nest.push(binding);
        Ok(())
    }

    /// `NewAllocMerged`: one allocation holding the entry and its packed children, each at an
    /// offset aligned to the group's packing alignment.
    fn new_alloc_merged(&mut self, e: usize) -> Result<()> {
        let entry = &self.alloc_vec[e];
        check!(
            alloc_property(&entry.scope)? == AllocProperty::SingleAllocation,
            "only single-allocation scopes are packed"
        );
        check!(entry.const_nbits != 0, "a packed entry has a constant size");
        let align = entry.packing_alignment_bits;
        let round_up = |bits: u64| {
            if bits.is_multiple_of(align) {
                bits
            } else {
                bits + (align - bits % align)
            }
        };
        let mut total_bits = round_up(entry.const_nbits);
        let first = entry.allocs[0].clone();
        let children = entry.merged_children.clone();
        for &child in &children {
            let child = &mut self.alloc_vec[child];
            check!(child.const_nbits != 0, "a packed child has a constant size");
            check!(total_bits != 0, "a packed child follows its parent");
            child.bits_offset = total_bits;
            child.alloc_var = Some(first.var.clone());
            total_bits = round_up(total_bits + child.const_nbits);
        }
        let entry = &self.alloc_vec[e];
        let type_bits = element_bits(entry.elem_type)?;
        let first_type = declared_type(first.value.as_node::<CallObj>().expect("a call"))?;
        let alloc_elements = total_bits / type_bits;
        let mut shape_type = prim_dtype(&first_type.shape.get(0)?)?;
        if shape_type.bits < 64 && alloc_elements > i32::MAX as u64 {
            shape_type = DLDataType { code: DLDataTypeCode::kDLInt as u8, bits: 64, lanes: 1 };
        }
        let alloc_size = make_const(shape_type, alloc_elements as i64)?;
        let ty =
            flat_tensor_type(alloc_size, entry.elem_type, &entry.scope, (align / 8) as i32, 0)?;
        let buf = Var::new(first.var.name.clone(), ty)?;
        let mut any_volatile = entry.is_volatile;
        for &child in &children {
            self.alloc_vec[child].alloc_var = Some(buf.clone());
            any_volatile |= self.alloc_vec[child].is_volatile;
        }
        let binding = self.alloc_binding(&buf, any_volatile)?;
        let entry = &mut self.alloc_vec[e];
        entry.alloc_var = Some(buf);
        entry.alloc_nest.push(binding);
        Ok(())
    }

    // -- Rewriting ------------------------------------------------------------------------------

    /// An allocation the plan replaced: removed, or a view onto the planned backing tensor.
    fn mutate_alloc_tensor(&mut self, op: &Bind, call: &CallObj, m: &mut Mutator) -> Result<Any> {
        // Placement and opaque metadata stay at the original definition.
        if requires_exact_allocation(call)? {
            return m.default_mutate(self, op);
        }
        let Some(&(_, e)) = self.alloc_map.get(&ObjectIdentity::of(&op.var)) else {
            // Never accessed.
            return Ok(Evaluate::new(0)?.into());
        };
        let backing = self.alloc_var(e)?;
        if backing.same_as(&op.var) {
            // The winner: its allocation is already hoisted.
            return Ok(Evaluate::new(0)?.into());
        }
        let buf = self.remap_tensor(&op.var, &backing)?;
        self.declared_view(&buf, &backing, None, call, op)
    }

    /// A declaration whose root was planned: a view onto the planned backing tensor.
    fn mutate_decl_tensor(&mut self, op: &Bind, call: &CallObj, m: &mut Mutator) -> Result<Any> {
        let root = self.aliases.root(&op.var);
        let Some(&(_, e)) = self.alloc_map.get(&ObjectIdentity::of(&root)) else {
            return m.default_mutate(self, op);
        };
        let backing = self.alloc_var(e)?;
        let tensor = self.remap_tensor(&tensor_var(&op.var.clone().into())?, &backing)?;
        let attrs: Option<Attrs> = m.mutate_as(self, &call.attrs)?;
        self.declared_view(&tensor, &backing, attrs, call, op)
    }

    /// `tensor = decl_tensor(backing.data)`.
    fn declared_view(
        &self,
        tensor: &Var,
        backing: &Var,
        attrs: Option<Attrs>,
        call: &CallObj,
        op: &Bind,
    ) -> Result<Any> {
        let ty = tensor.ty.clone();
        let view = Call::new_full(
            Some(ty.clone()),
            self.ops.decl_tensor.clone().into(),
            Array::new(vec![backing.tensor_data()?]),
            attrs,
            Array::new(vec![ty]),
            call.loc.clone(),
        )?;
        Ok(Bind::new_full(tensor.clone(), view, op.loc.clone())?.into())
    }

    /// A masked load or store accesses its tensor like a load or store.
    fn mutate_masked(&mut self, op: &Call, m: &mut Mutator) -> Result<Any> {
        let is_load = op.op.same_as(&self.ops.masked_load);
        let tensor = tensor_var(&op.args.get(0)?)?;
        let args: Vec<Expr> = op.args.iter().collect();
        let value: Option<Expr> = if is_load { None } else { Some(m.mutate_as(self, &args[1])?) };
        let first_index = if is_load { 1 } else { 2 };
        let mut indices: Vec<Expr> = Vec::new();
        for index in &args[first_index..args.len() - 1] {
            indices.push(m.mutate_as(self, index)?);
        }
        let mask = &args[args.len() - 1];
        match value {
            None => {
                let access = TensorLoad::new_full(tensor, indices, op.loc.clone())?;
                let access = self.remap_load(access)?;
                let mut args = vec![access.source.clone()];
                args.extend(access.indices.iter());
                args.push(m.mutate_as(self, mask)?);
                let call = Call::new_full(
                    Some(access.ty.clone()),
                    op.op.clone(),
                    Array::new(args),
                    op.attrs.clone(),
                    op.ty_args.clone(),
                    op.loc.clone(),
                )?;
                Ok(call.into())
            }
            Some(value) => {
                let access = TensorStore::new_full(tensor, indices, value, op.loc.clone())?;
                let access = self.remap_store(access)?;
                let mut args = vec![access.dest.clone(), access.value.clone()];
                args.extend(access.indices.iter());
                args.push(m.mutate_as(self, mask)?);
                let call = Call::new_full(
                    Some(prim_type(void_dtype())?.into()),
                    op.op.clone(),
                    Array::new(args),
                    op.attrs.clone(),
                    op.ty_args.clone(),
                    op.loc.clone(),
                )?;
                Ok(call.into())
            }
        }
    }
}

#[dispatch(mutate)]
impl StoragePlanRewriter {
    /// Every use and definition of a remapped tensor becomes its view.
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        match self.remap.get(&ObjectIdentity::of(&var)) {
            Some(mapped) => Ok(mapped.clone().into()),
            None => m.default_mutate(self, &var),
        }
    }

    fn mutate_tensor_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let node = if value.same_as(&op.value) && indices.same_as(&op.indices) {
            op
        } else {
            TensorStore::new_full(op.dest.clone(), &indices, value, op.loc.clone())?
        };
        Ok(self.remap_store(node)?.into())
    }

    fn mutate_tensor_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let node = if indices.same_as(&op.indices) {
            op
        } else {
            TensorLoad::new_full(op.source.clone(), indices.iter(), op.loc.clone())?
        };
        Ok(self.remap_load(node)?.into())
    }

    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        if op.op.same_as(&self.ops.alloc_tensor) || op.op.same_as(&self.ops.decl_tensor) {
            // Call attrs are opaque to the default descent, but retained storage metadata can
            // refer to remapped bindings.
            let attrs: Option<Attrs> = m.mutate_as(self, &op.attrs)?;
            let call: Call = m.default_mutate_as(self, &op)?;
            if attrs.same_field(&op.attrs) {
                return Ok(call.into());
            }
            return Ok(call.with_attrs(attrs).into());
        }
        if op.op.same_as(&self.ops.tensor_data_ptr) && op.args.len() == 1 {
            if let Ok(var) = op.args.get(0)?.try_cast::<Var>() {
                let root = self.aliases.root(&var);
                if let Some(&(_, e)) = self.alloc_map.get(&ObjectIdentity::of(&root)) {
                    let bits_offset = self.alloc_vec[e].bits_offset;
                    check!(
                        bits_offset.is_multiple_of(8),
                        "a packed offset is a whole number of bytes"
                    );
                    let data = self.alloc_var(e)?.tensor_data()?;
                    if bits_offset != 0 || !structural_equal(&data.ty, &op.ty)? {
                        let int64 =
                            DLDataType { code: DLDataTypeCode::kDLInt as u8, bits: 64, lanes: 1 };
                        let offset = int_imm(int64, (bits_offset / 8) as i128)?;
                        let call = Call::new_full(
                            Some(op.ty.clone()),
                            self.ops.ptr_byte_offset.clone().into(),
                            Array::new(vec![data, offset.into()]),
                            None,
                            Array::new(vec![]),
                            op.loc.clone(),
                        )?;
                        return Ok(call.into());
                    }
                    return Ok(data.into());
                }
            }
        }
        if op.op.same_as(&self.ops.masked_load) || op.op.same_as(&self.ops.masked_store) {
            return self.mutate_masked(&op, m);
        }
        m.default_mutate(self, &op)
    }

    /// Allocations attached to a thread region open its body.
    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Any> {
        let attached = self.attach_map.get(&Some(ObjectIdentity::of(&op))).cloned();
        let region: RegionStmt = m.default_mutate_as(self, &op)?;
        match attached {
            Some(entries) => {
                let body = self.make_attach(&entries, region.body.clone())?;
                Ok(region.with_body(body).into())
            }
            None => Ok(region.into()),
        }
    }

    /// Allocations attached to a parallel loop open its body.
    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        check!(op.kind != FOR_KIND_VECTORIZED, "VectorizeLoop before LiftStorageAlloc");
        let Some(entries) = self.attach_map.get(&Some(ObjectIdentity::of(&op))).cloned() else {
            return m.default_mutate(self, &op);
        };
        let op: For = m.default_mutate_as(self, &op)?;
        let body = self.make_attach(&entries, op.body.clone())?;
        let stmt = For::new_full(
            op.loop_var.clone(),
            op.min.clone(),
            op.extent.clone(),
            op.kind,
            body,
            op.annotations.clone(),
            op.step.clone(),
            unknown_loc()?,
        )?;
        Ok(stmt.into())
    }

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if call.op.same_as(&self.ops.alloc_tensor) {
                return self.mutate_alloc_tensor(&op, call, m);
            }
            if call.op.same_as(&self.ops.decl_tensor) {
                return self.mutate_decl_tensor(&op, call, m);
            }
        }
        m.default_mutate(self, &op)
    }
}

// ---------------------------------------------------------------------------------------------
// Pointer value types

const FUNCTION_TENSOR_PARAM: u32 = 1 << 0;
const FUNCTION_POINTER_PARAM: u32 = 1 << 1;
const ALLOC_TENSOR_CALL: u32 = 1 << 2;
const LET_NODE: u32 = 1 << 3;
const DECL_TENSOR_CALL: u32 = 1 << 4;

/// How a buffer is declared and accessed.
struct BufferVarInfo {
    var: Var,
    element_dtype: DLDataType,
    /// The extent of the last dimension; zero when unknown.
    extent: Expr,
    declaration_location: u32,
    /// Element types of the accesses: another base type, or a vector of the element type.
    access_dtype: Vec<DLDataType>,
    /// Vector types a scalar read can be shuffled out of.
    scalar_read_dtype: Vec<DLDataType>,
}

fn insert_dtype(set: &mut Vec<DLDataType>, dtype: DLDataType) {
    if !set.contains(&dtype) {
        set.push(dtype);
    }
}

impl BufferVarInfo {
    /// The vector type every access agrees on and the extent allows, else the declared type.
    fn preferred_dtype(&self) -> Result<DLDataType> {
        let mut base_access_dtype: Vec<DLDataType> = Vec::new();
        for &dtype in self.access_dtype.iter().chain(&self.scalar_read_dtype) {
            insert_dtype(&mut base_access_dtype, with_lanes(dtype, 1));
        }
        // Several base types: a type cast at each access is all the code generator needs.
        if base_access_dtype.len() != 1 {
            return Ok(self.element_dtype);
        }
        let preferred_base_type = base_access_dtype[0];
        // Scalar reads alone (no writes) leave the type as declared.
        let mut preferred_lanes = lanes(self.element_dtype)?;
        if preferred_lanes == 1 && self.access_dtype.len() == 1 {
            let lanes_used = lanes(self.access_dtype[0])?;
            for &dtype in &self.scalar_read_dtype {
                if lanes(dtype)? % lanes_used != 0 {
                    return Ok(self.element_dtype);
                }
            }
            let me = Analyzer::new()?.modular_set(&self.extent)?;
            let lanes_used64 = lanes_used as i64;
            if me.coeff % lanes_used64 == 0 && me.base_ % lanes_used64 == 0 {
                preferred_lanes = lanes_used;
            }
        }
        Ok(with_lanes(preferred_base_type, preferred_lanes))
    }
}

/// Records how each buffer and pointer is declared and accessed.
struct VectorTypeAccessChecker {
    ops: Ops,
    info_map: HashMap<ObjectIdentity, BufferVarInfo>,
    aliases: TensorAliases,
    /// A pointer without an element type takes the type it is accessed as.
    allow_untyped_pointers: bool,
    detect_scalar_read_patterns: bool,
    analyzer: Analyzer,
}

impl VectorTypeAccessChecker {
    fn new(
        ops: Ops,
        params: &Array<Var>,
        allow_untyped_pointers: bool,
        detect_scalar_read_patterns: bool,
    ) -> Result<Self> {
        let mut checker = Self {
            ops,
            info_map: HashMap::new(),
            aliases: TensorAliases::default(),
            allow_untyped_pointers,
            detect_scalar_read_patterns,
            analyzer: Analyzer::new()?,
        };
        for param in params.iter() {
            if let Ok(ty) = tensor_type(&param) {
                checker.aliases.set(&param, &param);
                let extent = match ty.shape.len() {
                    0 => int32(0)?.into(),
                    n => ty.shape.get(n - 1)?,
                };
                checker.declare(&param, dtype_of(&ty), extent, FUNCTION_TENSOR_PARAM)?;
                continue;
            }
            checker.declare_pointer(&param, FUNCTION_POINTER_PARAM)?;
        }
        Ok(checker)
    }

    /// A pointer variable of a known element type, or of none when untyped pointers are allowed.
    fn declare_pointer(&mut self, var: &Var, location: u32) -> Result<()> {
        let Some(pointer) = var.ty.as_node::<PtrTypeObj>() else { return Ok(()) };
        let Some(element) = pointer.element_type.as_node::<PrimTypeObj>() else { return Ok(()) };
        let dtype = element.dtype;
        if !is_void(dtype) {
            self.declare(var, dtype, int32(0)?.into(), location)?;
        } else if self.allow_untyped_pointers {
            self.declare(var, void_dtype(), int32(0)?.into(), location)?;
        }
        Ok(())
    }

    /// `OnArrayDeclaration`.
    fn declare(
        &mut self,
        buffer: &Var,
        element_dtype: DLDataType,
        extent: Expr,
        declaration_location: u32,
    ) -> Result<()> {
        let id = ObjectIdentity::of(buffer);
        check!(
            !self.info_map.contains_key(&id),
            "Array declaration of {} occurred multiple times.",
            buffer.name.as_str()
        );
        self.info_map.insert(
            id,
            BufferVarInfo {
                var: buffer.clone(),
                element_dtype: storage_dtype(element_dtype)?,
                extent,
                declaration_location,
                access_dtype: Vec::new(),
                scalar_read_dtype: Vec::new(),
            },
        );
        Ok(())
    }

    /// A tensor declared by `alloc_tensor` or `decl_tensor`, of the declared type's last extent.
    fn declare_tensor(&mut self, var: &Var, call: &CallObj, location: u32) -> Result<()> {
        let ty = declared_type(call)?;
        let extent = match ty.shape.len() {
            0 => int32(0)?.into(),
            n => ty.shape.get(n - 1)?,
        };
        self.declare(var, dtype_of(&ty), extent, location)
    }

    /// `OnArrayAccess`: `buffer` accessed as `value_dtype` at `indices`.
    fn on_array_access(
        &mut self,
        value_dtype: DLDataType,
        buffer: &Expr,
        indices: &Array<Expr>,
        is_tensor_load: bool,
    ) -> Result<()> {
        let buffer = tensor_var(buffer)?;
        let root = self.aliases.root(&buffer);
        let mut id = ObjectIdentity::of(&buffer);
        if self.info_map.contains_key(&ObjectIdentity::of(&root)) {
            id = ObjectIdentity::of(&root);
        }
        let allow_untyped_pointers = self.allow_untyped_pointers;
        let Some(var_info) = self.info_map.get_mut(&id) else {
            return Err(internal_error(format!(
                "Load/Store of buffer {} occurred before its declaration.",
                buffer.name.as_str()
            )));
        };
        // Scalable accesses are neither checked nor rewritten.
        if is_scalable_vector(value_dtype) {
            return Ok(());
        }
        let value_dtype = storage_dtype(value_dtype)?;
        if is_void(var_info.element_dtype) {
            check!(
                allow_untyped_pointers,
                "Variable {} was missing a type annotation in its declaration",
                buffer.name.as_str()
            );
            var_info.element_dtype = with_lanes(value_dtype, 1);
        }
        let indices: Vec<Expr> = indices.iter().collect();
        for index in indices.iter().take(indices.len().saturating_sub(1)) {
            check!(
                encoded_lanes(prim_dtype(index)?) == 1,
                "Only the last index of a buffer access may be a vector type."
            );
        }
        let index_lanes = match indices.last() {
            Some(index) => lanes(prim_dtype(index)?)?,
            None => 1,
        };
        let access_dtype = value_dtype;
        let mut lanes_used = lanes(var_info.element_dtype)?;
        // A previous rewrite without index rewriting leaves a vector element type accessed by
        // vector indices.
        if index_lanes * lanes(var_info.element_dtype)? != lanes(value_dtype)? {
            check!(index_lanes == lanes(value_dtype)?, "the index lanes match the value lanes");
            lanes_used = 1;
            var_info.element_dtype = with_lanes(var_info.element_dtype, 1);
        }
        // A unit-stride ramp at a lane-aligned base is a vector access.
        if let Some(ramp) = indices.last().and_then(|index| index.as_node::<RampObj>()) {
            if is_const_int(&ramp.stride, 1) && ramp.lanes.as_node::<IntImmObj>().is_some() {
                let lanes = int_value(&ramp.lanes)? as i64;
                let me = self.analyzer.modular_set(&ramp.base_)?;
                if me.coeff % lanes == 0 && me.base_ % lanes == 0 {
                    lanes_used = lanes as i32;
                }
            }
        }
        if self.detect_scalar_read_patterns && is_tensor_load {
            if let Some(last) = indices.last() {
                if encoded_lanes(prim_dtype(last)?) == 1 {
                    let me = self.analyzer.modular_set(last)?;
                    // Fixed lane counts use 15 bits.
                    if (0..=32767).contains(&me.coeff) {
                        let var_info = self.info_map.get_mut(&id).expect("found above");
                        insert_dtype(
                            &mut var_info.scalar_read_dtype,
                            with_lanes(access_dtype, me.coeff as i32),
                        );
                        return Ok(());
                    }
                }
            }
        }
        let var_info = self.info_map.get_mut(&id).expect("found above");
        insert_dtype(&mut var_info.access_dtype, with_lanes(access_dtype, lanes_used));
        Ok(())
    }

    /// A load's value type and indices: `masked_load`/`masked_store` take them as arguments.
    fn masked_access(&self, call: &Call, is_load: bool) -> Result<(DLDataType, Array<Expr>)> {
        let args: Vec<Expr> = call.args.iter().collect();
        let dtype = if is_load { prim_dtype(&call.clone().into())? } else { prim_dtype(&args[1])? };
        let first_index = if is_load { 1 } else { 2 };
        Ok((dtype, args[first_index..args.len() - 1].iter().cloned().collect()))
    }
}

#[dispatch(walk)]
impl VectorTypeAccessChecker {
    fn walk_tensor_load(&mut self, op: TensorLoad) -> Result<WalkResult> {
        let dtype = prim_dtype(&op.clone().into())?;
        self.on_array_access(dtype, &op.source, &op.indices, true)?;
        Ok(WalkResult::Advance)
    }

    fn walk_tensor_store(&mut self, op: TensorStore) -> Result<WalkResult> {
        self.on_array_access(prim_dtype(&op.value)?, &op.dest, &op.indices, false)?;
        Ok(WalkResult::Advance)
    }

    /// Masked accesses, and the address of a load (an access, though not a read).
    fn walk_call(&mut self, op: Call) -> Result<WalkResult> {
        let is_load = op.op.same_as(&self.ops.masked_load);
        if is_load || op.op.same_as(&self.ops.masked_store) {
            let (dtype, indices) = self.masked_access(&op, is_load)?;
            self.on_array_access(dtype, &op.args.get(0)?, &indices, is_load)?;
        } else if op.op.same_as(&self.ops.address_of) {
            if let Ok(load) = op.args.get(0)?.try_cast::<TensorLoad>() {
                let dtype = prim_dtype(&load.clone().into())?;
                self.on_array_access(dtype, &load.source, &load.indices, false)?;
            }
        }
        Ok(WalkResult::Advance)
    }

    fn walk_let(&mut self, op: Let) -> Result<WalkResult> {
        self.declare_pointer(&op.var, LET_NODE)?;
        Ok(WalkResult::Advance)
    }

    fn walk_bind(&mut self, op: Bind) -> Result<WalkResult> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if call.op.same_as(&self.ops.alloc_tensor) {
                self.aliases.set(&op.var, &op.var);
                self.declare_tensor(&op.var, call, ALLOC_TENSOR_CALL)?;
                return Ok(WalkResult::Advance);
            }
            if call.op.same_as(&self.ops.decl_tensor) {
                let tensor = tensor_var(&op.var.clone().into())?;
                let ops = self.ops.clone();
                self.aliases.register(&ops, &tensor, &call.args.get(0)?)?;
                self.declare_tensor(&op.var, call, DECL_TENSOR_CALL)?;
                return Ok(WalkResult::Advance);
            }
        }
        self.declare_pointer(&op.var, LET_NODE)?;
        Ok(WalkResult::Advance)
    }
}

/// The operand of `address_of` is an address, not a value: its load is not visited as one.
struct AddressOfOperand;

impl ContextPolicy<VectorTypeAccessChecker> for AddressOfOperand {
    fn default_visit(
        &self,
        value: &StructuralView,
        ctx: &mut VisitContext<'_, VectorTypeAccessChecker>,
    ) -> Result<Option<VisitInterrupt>> {
        if let Some(call) = value.as_node::<CallObj>() {
            if call.op.same_as(&ctx.state().ops.address_of) {
                if let Ok(load) = call.args.get(0)?.try_cast::<TensorLoad>() {
                    let kind = ctx.def_region_kind();
                    return ctx.default_visit_children(&load, kind);
                }
            }
        }
        ctx.visit_children()
    }
}

/// A buffer variable to retype.
struct RewriteInfo {
    old_tensor_var: Var,
    new_tensor_var: Var,
    old_element_dtype: DLDataType,
    new_element_dtype: DLDataType,
}

impl RewriteInfo {
    /// Scalar elements per new element.
    fn factor(&self) -> Result<i32> {
        let old_lanes = lanes(self.old_element_dtype)?;
        let new_lanes = lanes(self.new_element_dtype)?;
        check!(new_lanes % old_lanes == 0, "the new lanes are a multiple of the old");
        Ok(new_lanes / old_lanes)
    }
}

/// Which declarations to retype, and whether to rewrite the accesses.
#[derive(Clone, Copy)]
struct RewriteOptions {
    allow_untyped_pointers: bool,
    rewrite_tensor_params: bool,
    rewrite_pointer_params: bool,
    rewrite_alloc_tensor_node: bool,
    rewrite_indices: bool,
    rewrite_let_node: bool,
    rewrite_scalar_read_to_vector_shuffle: bool,
}

impl RewriteOptions {
    /// The `PointerValueTypeRewrite` pass defaults.
    const PASS: Self = Self {
        allow_untyped_pointers: false,
        rewrite_tensor_params: true,
        rewrite_pointer_params: true,
        rewrite_alloc_tensor_node: true,
        rewrite_indices: true,
        rewrite_let_node: true,
        rewrite_scalar_read_to_vector_shuffle: true,
    };

    /// After StorageRewrite: parameters keep their types; internal allocations may change.
    const AFTER_STORAGE_REWRITE: Self = Self {
        allow_untyped_pointers: true,
        rewrite_tensor_params: false,
        rewrite_pointer_params: false,
        rewrite_alloc_tensor_node: true,
        rewrite_indices: true,
        rewrite_let_node: true,
        rewrite_scalar_read_to_vector_shuffle: false,
    };

    fn rewrite_mask(&self) -> u32 {
        let mut mask = 0;
        if self.rewrite_tensor_params {
            mask |= FUNCTION_TENSOR_PARAM;
        }
        if self.rewrite_pointer_params {
            mask |= FUNCTION_POINTER_PARAM;
        }
        if self.rewrite_alloc_tensor_node {
            mask |= ALLOC_TENSOR_CALL;
        }
        if self.rewrite_let_node {
            mask |= LET_NODE;
        }
        mask
    }
}

/// Retypes buffers whose accesses all agree on a vector type, dividing their indices: with
/// `float16* a` retyped to `float16x4* a`, `a[Ramp(i, 1, 4)]` becomes `a[i / 4]`, and a scalar
/// read `a[i]` becomes a shuffle out of `a[i / 4]`.
struct VectorTypeRewriter {
    ops: Ops,
    rewrite_indices: bool,
    rewrite_map: HashMap<ObjectIdentity, RewriteInfo>,
    aliases: TensorAliases,
    /// Retyped tensors, replacing the originals everywhere.
    remap: HashMap<ObjectIdentity, Var>,
    analyzer: Analyzer,
}

/// `last_dim / factor` as the last extent of a retyped tensor type.
fn divided_shape(shape: &Array<Expr>, factor: i32) -> Result<Array<Expr>> {
    let mut extents: Vec<Expr> = shape.iter().collect();
    if let Some(last) = extents.pop() {
        let factor = make_const(prim_dtype(&last)?, factor as i64)?;
        extents.push(last.div(factor).expr()?);
    }
    Ok(Array::new(extents))
}

impl VectorTypeRewriter {
    fn new(
        ops: Ops,
        info_map: &HashMap<ObjectIdentity, BufferVarInfo>,
        aliases: TensorAliases,
        options: RewriteOptions,
    ) -> Result<Self> {
        let rewrite_mask = options.rewrite_mask();
        let mut rewrite_map = HashMap::new();
        for (id, var_info) in info_map {
            let preferred = var_info.preferred_dtype()?;
            if preferred == var_info.element_dtype
                || rewrite_mask & var_info.declaration_location == 0
            {
                continue;
            }
            let old = &var_info.var;
            let new = match tensor_type(old) {
                Ok(ty) => {
                    let factor = lanes(preferred)? / lanes(var_info.element_dtype)?;
                    let shape = divided_shape(&ty.shape, factor)?;
                    let ty = retyped(&ty, prim_type(preferred)?, shape, None);
                    Var::new_full(old.name.clone(), ty, old.loc.clone())?
                }
                Err(_) => {
                    let pointer = old
                        .ty
                        .as_node::<PtrTypeObj>()
                        .ok_or_else(|| Error::new(TYPE_ERROR, "expected a pointer variable", ""))?;
                    let ty = PtrType::new(prim_type(preferred)?, pointer.scope.clone())?;
                    Var::new_full(old.name.clone(), ty, old.loc.clone())?
                }
            };
            rewrite_map.insert(
                id.clone(),
                RewriteInfo {
                    old_tensor_var: old.clone(),
                    new_tensor_var: new,
                    old_element_dtype: var_info.element_dtype,
                    new_element_dtype: preferred,
                },
            );
        }
        Ok(Self {
            ops,
            rewrite_indices: options.rewrite_indices,
            rewrite_map,
            aliases,
            remap: HashMap::new(),
            analyzer: Analyzer::new()?,
        })
    }

    /// The access's last index in units of the new element type; for a scalar index, also the
    /// lane it reads (`VisitTensorAccess`). `None` when the tensor is not retyped.
    fn rewrite_access(
        &self,
        tensor: &Var,
        indices: &Array<Expr>,
    ) -> Result<Option<(Array<Expr>, i64)>> {
        if !self.rewrite_indices {
            return Ok(None);
        }
        let root = self.aliases.root(tensor);
        let Some(info) = self.rewrite_map.get(&ObjectIdentity::of(&root)) else {
            return Ok(None);
        };
        let mut indices: Vec<Expr> = indices.iter().collect();
        let last_dim_index = indices.last().cloned().ok_or_else(|| internal_error("no indices"))?;
        let index_dtype = prim_dtype(&last_dim_index)?;
        // Scalable accesses are neither checked nor rewritten.
        if is_scalable_vector(dtype_of(&tensor_type(tensor)?)) || is_scalable_vector(index_dtype) {
            return Ok(None);
        }
        let mut shuffle_index = -1;
        let ramp = last_dim_index.as_node::<RampObj>();
        if let Some(ramp) = ramp.filter(|ramp| {
            is_const_int(&ramp.stride, 1) && ramp.lanes.as_node::<IntImmObj>().is_some()
        }) {
            let lanes = int_value(&ramp.lanes)?;
            let factor = info.factor()?;
            let base_lanes = make_const(prim_dtype(&ramp.base_)?, lanes as i64)?;
            let mut new_index = ramp.base_.div(base_lanes).expr()?;
            if lanes != factor {
                check!(factor != 0 && lanes % factor == 0, "the ramp covers whole elements");
                let new_lanes = lanes / factor;
                new_index = Ramp::new_full(
                    ex(&new_index) * new_lanes,
                    ramp.stride.clone(),
                    new_lanes,
                    ramp.loc.clone(),
                )?
                .into();
            }
            *indices.last_mut().expect("non-empty") = new_index;
        } else if lanes(index_dtype)? == 1 && info.factor()? > 1 {
            let factor = info.factor()? as i64;
            let me = self.analyzer.modular_set(&last_dim_index)?;
            check!(me.coeff == 0 || factor % me.coeff == 0, "the index reads within one element");
            let divisor = make_const(index_dtype, factor)?;
            *indices.last_mut().expect("non-empty") = last_dim_index.div(divisor).expr()?;
            shuffle_index = me.base_ % factor;
        }
        Ok(Some((Array::new(indices), shuffle_index)))
    }

    fn visit_store(&mut self, node: TensorStore) -> Result<(TensorStore, i64)> {
        let dest = tensor_var(&node.dest)?;
        let Some((indices, shuffle_index)) = self.rewrite_access(&dest, &node.indices)? else {
            return Ok((node, -1));
        };
        let dest = self.remap_tensor(&dest)?;
        let (loc, value) = (node.loc.clone(), node.value.clone());
        Ok((TensorStore::from_complete_fields(loc, dest.into(), indices, value), shuffle_index))
    }

    fn visit_load(&mut self, node: TensorLoad) -> Result<(TensorLoad, i64)> {
        let tensor = tensor_var(&node.source)?;
        let Some((indices, shuffle_index)) = self.rewrite_access(&tensor, &node.indices)? else {
            return Ok((node, -1));
        };
        let tensor = self.remap_tensor(&tensor)?;
        Ok((TensorLoad::new_full(tensor, indices.iter(), node.loc.clone())?, shuffle_index))
    }

    /// The retyped tensor `buf` becomes: the new root, or an alias retyped like its root.
    fn remap_tensor(&mut self, buf: &Var) -> Result<Var> {
        let id = ObjectIdentity::of(buf);
        if let Some(mapped) = self.remap.get(&id) {
            return Ok(mapped.clone());
        }
        let root = self.aliases.root(buf);
        let mut remapped = buf.clone();
        if let Some(info) = self.rewrite_map.get(&ObjectIdentity::of(&root)) {
            if root.same_as(buf) {
                remapped = info.new_tensor_var.clone();
            } else {
                let ty = tensor_type(buf)?;
                let shape = divided_shape(&ty.shape, info.factor()?)?;
                let ty = retyped(&ty, prim_type(info.new_element_dtype)?, shape, None);
                remapped = Var::new_full(buf.name.clone(), ty, buf.loc.clone())?;
            }
        }
        self.remap.insert(id, remapped.clone());
        Ok(remapped)
    }

    /// Masked accesses rewrite like loads and stores; neither can take a scalar shuffle.
    fn rewrite_masked_call(&mut self, op: &Call, m: &mut Mutator) -> Result<Option<Any>> {
        let is_load = op.op.same_as(&self.ops.masked_load);
        if !is_load && !op.op.same_as(&self.ops.masked_store) {
            return Ok(None);
        }
        let tensor = tensor_var(&op.args.get(0)?)?;
        let args: Vec<Expr> = op.args.iter().collect();
        let value: Option<Expr> = if is_load { None } else { Some(m.mutate_as(self, &args[1])?) };
        let first_index = if is_load { 1 } else { 2 };
        let mut indices: Vec<Expr> = Vec::new();
        for index in &args[first_index..args.len() - 1] {
            prim_dtype(index)?;
            indices.push(m.mutate_as(self, index)?);
        }
        let mask = &args[args.len() - 1];
        let call = match value {
            None => {
                let access = TensorLoad::new_full(tensor, indices, op.loc.clone())?;
                let (modified, shuffle_index) = self.visit_load(access)?;
                check!(
                    shuffle_index < 0,
                    "A masked vector load cannot be rewritten into a scalar shuffle."
                );
                let mut args = vec![modified.source.clone()];
                args.extend(modified.indices.iter());
                args.push(m.mutate_as(self, mask)?);
                Call::new_full(
                    Some(modified.ty.clone()),
                    op.op.clone(),
                    Array::new(args),
                    op.attrs.clone(),
                    op.ty_args.clone(),
                    op.loc.clone(),
                )?
            }
            Some(value) => {
                let access = TensorStore::new_full(tensor, indices, value, op.loc.clone())?;
                let (modified, shuffle_index) = self.visit_store(access)?;
                check!(
                    shuffle_index < 0,
                    "A masked vector store cannot be rewritten into a scalar shuffle."
                );
                let mut args = vec![modified.dest.clone(), modified.value.clone()];
                args.extend(modified.indices.iter());
                args.push(m.mutate_as(self, mask)?);
                Call::new_full(
                    Some(prim_type(void_dtype())?.into()),
                    op.op.clone(),
                    Array::new(args),
                    op.attrs.clone(),
                    op.ty_args.clone(),
                    op.loc.clone(),
                )?
            }
        };
        Ok(Some(call.into()))
    }

    /// `alloc_tensor` of a retyped tensor: the binding of the new tensor.
    fn mutate_alloc_tensor(&mut self, op: &Bind, call: &CallObj, m: &mut Mutator) -> Result<Any> {
        let new_buf = self.remap_tensor(&tensor_var(&op.var.clone().into())?)?;
        let mut args: Vec<Expr> = call.args.iter().collect();
        // C++ `args.Set(0, ...)` copies the shared array, so a call with arguments is rebuilt.
        let args_same = args.is_empty();
        if let Some(first) = args.first_mut() {
            *first = m.mutate_as(self, &*first)?;
        }
        let attrs: Option<Attrs> = m.mutate_as(self, &call.attrs)?;
        if new_buf.same_as(&op.var) && args_same && attrs.same_field(&call.attrs) {
            return Ok(op.clone().into());
        }
        let ty = new_buf.ty.clone();
        let call = Call::new_full(
            Some(ty.clone()),
            self.ops.alloc_tensor.clone().into(),
            Array::new(args),
            attrs,
            Array::new(vec![ty]),
            call.loc.clone(),
        )?;
        Ok(Bind::new_full(new_buf, call, op.loc.clone())?.into())
    }

    /// `decl_tensor` of a retyped tensor or of a retyped tensor's data.
    fn mutate_decl_tensor(&mut self, op: &Bind, call: &CallObj, m: &mut Mutator) -> Result<Any> {
        let original = call.args.get(0)?;
        let data: Expr = m.mutate_as(self, &original)?;
        let tensor = self.remap_tensor(&tensor_var(&op.var.clone().into())?)?;
        let attrs: Option<Attrs> = m.mutate_as(self, &call.attrs)?;
        if tensor.same_as(&op.var) && data.same_as(&original) && attrs.same_field(&call.attrs) {
            return Ok(op.clone().into());
        }
        let ty = tensor.ty.clone();
        let call = Call::new_full(
            Some(ty.clone()),
            self.ops.decl_tensor.clone().into(),
            Array::new(vec![data]),
            attrs,
            Array::new(vec![ty]),
            call.loc.clone(),
        )?;
        Ok(Bind::new_full(tensor, call, op.loc.clone())?.into())
    }
}

#[dispatch(mutate)]
impl VectorTypeRewriter {
    /// Every use of a remapped tensor becomes the retyped one.
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        match self.remap.get(&ObjectIdentity::of(&var)) {
            Some(mapped) => Ok(mapped.clone().into()),
            None => m.default_mutate(self, &var),
        }
    }

    /// A rewritten scalar read takes its lane out of the vector element.
    fn mutate_tensor_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let node = if indices.same_as(&op.indices) {
            op
        } else {
            TensorLoad::new_full(op.source.clone(), indices.iter(), op.loc.clone())?
        };
        let (modified, shuffle_index) = self.visit_load(node.clone())?;
        if node.same_as(&modified) {
            return Ok(node.into());
        }
        if shuffle_index >= 0 {
            let element = Shuffle::extract_element(modified, shuffle_index as i32)?;
            return Ok(element.into());
        }
        Ok(modified.into())
    }

    fn mutate_tensor_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let indices: Array<Expr> = m.mutate_as(self, &op.indices)?;
        let node = if value.same_as(&op.value) && indices.same_as(&op.indices) {
            op
        } else {
            TensorStore::new_full(op.dest.clone(), &indices, value, op.loc.clone())?
        };
        let (modified, shuffle_index) = self.visit_store(node)?;
        check!(shuffle_index < 0, "a store is not rewritten into a shuffle");
        Ok(modified.into())
    }

    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        if let Some(rewritten) = self.rewrite_masked_call(&op, m)? {
            return Ok(rewritten);
        }
        if op.op.same_as(&self.ops.tensor_data_ptr) && op.args.len() == 1 {
            if let Ok(var) = op.args.get(0)?.try_cast::<Var>() {
                if is_tensor(&var) {
                    let tensor = self.remap_tensor(&var)?;
                    return Ok(tensor.tensor_data()?.into());
                }
            }
        }
        m.default_mutate(self, &op)
    }

    /// A retyped pointer binding takes the new variable; its value's type follows.
    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if call.op.same_as(&self.ops.alloc_tensor) {
                return self.mutate_alloc_tensor(&op, call, m);
            }
            if call.op.same_as(&self.ops.decl_tensor) {
                return self.mutate_decl_tensor(&op, call, m);
            }
        }
        let var = match self.rewrite_map.get(&ObjectIdentity::of(&op.var)) {
            Some(info) => info.new_tensor_var.clone(),
            None => op.var.clone(),
        };
        let mut value: Expr = m.mutate_as(self, &op.value)?;
        let mut value_unchanged = value.same_as(&op.value);
        if !structural_equal(&value.ty, &var.ty)? {
            let call: Call = value.try_cast()?;
            value = Call::new_full(
                Some(var.ty.clone()),
                call.op.clone(),
                call.args.clone(),
                call.attrs.clone(),
                call.ty_args.clone(),
                call.loc.clone(),
            )?
            .into();
            value_unchanged = false;
        }
        if var.same_as(&op.var) && value_unchanged {
            return Ok(op.into());
        }
        Ok(Bind::new(var, value)?.into())
    }
}

/// `Finalize`: the remaining references to retyped variables, and the pointer types of their
/// `tensor_data_ptr` projections.
struct PointerVarSubstituter {
    remap: HashMap<ObjectIdentity, Var>,
    tensor_data_ptr: Op,
}

#[dispatch(mutate)]
impl PointerVarSubstituter {
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        match self.remap.get(&ObjectIdentity::of(&var)) {
            Some(mapped) => Ok(mapped.clone().into()),
            None => m.default_mutate(self, &var),
        }
    }

    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        let call: Call = m.default_mutate_as(self, &op)?;
        if !call.op.same_as(&self.tensor_data_ptr) || call.args.len() != 1 {
            return Ok(call.into());
        }
        let ty: Type = tensor_type(&tensor_var(&call.args.get(0)?)?)?.data_ptr_type()?.into();
        if structural_equal(&call.ty, &ty)? {
            return Ok(call.into());
        }
        let call = Call::new_full(
            Some(ty),
            call.op.clone(),
            call.args.clone(),
            call.attrs.clone(),
            call.ty_args.clone(),
            call.loc.clone(),
        )?;
        Ok(call.into())
    }
}

/// `PointerValueTypeRewrite(f, ...)`: retype the buffers of `func` whose accesses agree on a
/// vector type.
fn rewrite_pointer_value_types(
    func: Function,
    ops: &Ops,
    options: RewriteOptions,
) -> Result<Function> {
    let checker = VectorTypeAccessChecker::new(
        ops.clone(),
        &func.params,
        options.allow_untyped_pointers,
        options.rewrite_scalar_read_to_vector_shuffle,
    )?;
    let mut walk = WalkWithContextPolicy::new(checker, AddressOfOperand);
    structural_walk(&func.body, &mut walk, WalkOrder::PreOrder)?;
    let checker = walk.into_state();
    let mut rewriter =
        VectorTypeRewriter::new(ops.clone(), &checker.info_map, checker.aliases, options)?;
    let body: Option<SeqStmt> = match &func.body {
        Some(body) => Some(structural_mutate(body.clone(), &mut rewriter)?.try_into()?),
        None => None,
    };
    let mut substituter = PointerVarSubstituter {
        remap: rewriter
            .rewrite_map
            .values()
            .map(|info| (ObjectIdentity::of(&info.old_tensor_var), info.new_tensor_var.clone()))
            .collect(),
        tensor_data_ptr: ops.tensor_data_ptr.clone(),
    };
    let body: Option<SeqStmt> = match body {
        Some(body) => Some(structural_mutate(body, &mut substituter)?.try_into()?),
        None => None,
    };
    let params: Array<Var> = func
        .params
        .iter()
        .map(|param| match rewriter.rewrite_map.get(&ObjectIdentity::of(&param)) {
            Some(info) => info.new_tensor_var.clone(),
            None => param,
        })
        .collect();
    Ok(func.with(|node| {
        node.params = params;
        node.body = body;
    }))
}

pub fn storage_rewrite() -> Result<Pass> {
    let ops = Ops::get()?;
    function_pass("tirx.StorageRewrite", 0, move |func, _module, context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        // With merged static shared memory, MergeSharedMemoryAllocations does the reuse.
        let merge_static_smem =
            config_flag(context.config.get(&String::from("tirx.merge_static_smem"))?)?;
        let enable_reuse = !merge_static_smem;
        // Vulkan and WebGPU reuse storage only between identical element types.
        let reuse_require_exact_matched_dtype = match get_attr(&func.attrs, "target")? {
            Some(target) => {
                let kind = Target::try_from(target)?.kind()?.name()?;
                kind.as_str() == "vulkan" || kind.as_str() == "webgpu"
            }
            None => false,
        };
        let body = StoragePlanRewriter::new(ops.clone())?.rewrite(
            body,
            &func.params,
            true,
            enable_reuse,
            reuse_require_exact_matched_dtype,
        )?;
        let func = func.with_body(Some(body));
        rewrite_pointer_value_types(func, &ops, RewriteOptions::AFTER_STORAGE_REWRITE).map(Some)
    })
}

pub fn pointer_value_type_rewrite() -> Result<Pass> {
    let ops = Ops::get()?;
    function_pass("tirx.PointerValueTypeRewrite", 0, move |func, _module, _context| {
        if func.body.is_none() {
            return Ok(Some(func));
        }
        rewrite_pointer_value_types(func, &ops, RewriteOptions::PASS).map(Some)
    })
}
