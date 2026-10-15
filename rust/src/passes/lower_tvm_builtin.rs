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

//! `tirx.LowerTVMBuiltin` (`src/tirx/transform/lower_tvm_builtin.cc`): in a host function, lower
//! packed calls onto an argument stack, `stack_make_shape` / `stack_make_dltensor` onto shape and
//! array stacks, and device tensor allocations onto workspace calls.
//!
//! The stacks of a function body, and of each parallel loop body, are sized by a first pass of
//! the same rewrite over a copy (`GetMaxStack`), then allocated ahead of the body. The statements
//! a call needs prepared go before the statement holding it.

use tvm_ffi::{
    dispatch, structural_mutate, Any, Array, BigInt, DLDataType, DLDataTypeExt, Mutator,
    ObjectRefCast, ObjectRefCore, Result, String, INTERNAL_ERROR, TYPE_ERROR, VALUE_ERROR,
};

use super::util::{MutatorExt, function_pass, get_attr, has_nonzero_attr, prim_type};
use crate::ir::{
    Bind, BindObj, Call, CallObj, DictAttrs, Evaluate, Expr, For, ForObj, If, IfObj, IntImmObj, Op,
    PrimType, PrimTypeObj, PtrType, PtrTypeObj, RegionStmt, RegionStmtObj, SeqStmt, Stmt,
    StringImm, StringImmObj, TensorLoad, TensorStore, Var,
};
use crate::ops::{int32, int_imm, repr, unknown_loc, Ex, PrimOps};
use crate::prim::Cast;
use crate::runtime::MemoryScope;
use crate::target::Target;
use crate::tirx::cuda::TensorMapEncodeTiledAttrObj;
use crate::tirx::op::{
    ABI_FIELD_GET, ABI_FIELD_SET, ADDRESS_OF, ALLOC_TENSOR, ALLOC_WORKSPACE, CALL_CPACKED,
    CALL_CPACKED_LOWERED, CALL_FFI_KERNEL, CALL_PACKED, CALL_PACKED_LOWERED,
    CUDA_TENSORMAP_ENCODE_TILED, DECL_TENSOR, DEVICE_CONTEXT, FREE_WORKSPACE, ISNULLPTR,
    OPENCL_ND_MEM_ALLOC_WITH_SCOPE, STACK_ALLOCA, STACK_MAKE_DLTENSOR, STACK_MAKE_SHAPE,
    TENSOR_DATA_PTR, THROW_LAST_ERROR,
};
use crate::tirx::{Function, TensorType, TensorTypeObj};
use crate::transform::Pass;
use crate::SameField;

const TARGET: &str = "target";
const IS_HOST_FUNC: &str = "tirx.is_host_func";
/// `transform::kDisableLowerTVMBuiltin`.
const DISABLE_LOWER_BUILTIN: &str = "disable_lower_builtin";
/// `runtime::kMaxStackAlloca`.
const MAX_STACK_ALLOCA: u64 = 1024;
const DL_CPU: i64 = 1;
/// `ForKind::kParallel`.
const FOR_PARALLEL: i32 = 1;

// `TVMStructFieldKind` (`include/tvm/tirx/op/abi.h`).
const DLTENSOR_ADDR: i32 = 0;
const DLTENSOR_DATA: i32 = 1;
const DLTENSOR_SHAPE: i32 = 2;
const DLTENSOR_STRIDES: i32 = 3;
const DLTENSOR_NDIM: i32 = 4;
const DLTENSOR_TYPE_CODE: i32 = 5;
const DLTENSOR_TYPE_BITS: i32 = 6;
const DLTENSOR_TYPE_LANES: i32 = 7;
const DLTENSOR_BYTE_OFFSET: i32 = 8;
const DLTENSOR_DEVICE_ID: i32 = 9;
const DLTENSOR_DEVICE_TYPE: i32 = 10;
const FFI_ANY_TYPE_INDEX: i32 = 13;
const FFI_ANY_ZERO_PADDING: i32 = 14;
const FFI_ANY_UNION_VALUE: i32 = 15;

// `TVMFFITypeIndex`.
const FFI_NONE: i32 = 0;
const FFI_INT: i32 = 1;
const FFI_BOOL: i32 = 2;
const FFI_FLOAT: i32 = 3;
const FFI_OPAQUE_PTR: i32 = 4;
const FFI_DLTENSOR_PTR: i32 = 7;
const FFI_RAW_STR: i32 = 8;

const INT: u8 = 0;
const UINT: u8 = 1;
const FLOAT: u8 = 2;
const HANDLE: u8 = 3;
const BFLOAT: u8 = 4;
const BOOL: u8 = 6;

fn dtype(code: u8, bits: u8, lanes: u16) -> DLDataType {
    DLDataType { code, bits, lanes }
}

fn ty(code: u8, bits: u8) -> Result<PrimType> {
    prim_type(dtype(code, bits, 1))
}

/// `PrimType::lanes()`, which a scalable vector has none of at compile time.
fn lanes(d: DLDataType) -> Result<i64> {
    match d.lanes as i16 {
        lanes if lanes < 0 => Err(error!(
            INTERNAL_ERROR,
            "Can't fetch the lanes of a scalable vector at a compile time."
        )),
        lanes => Ok(lanes as i64),
    }
}

fn is_scalable(d: DLDataType) -> bool {
    (d.lanes as i16) < -1
}

/// `PrimType::StorageBytes()`.
fn storage_bytes(d: DLDataType) -> Result<i64> {
    let lanes = lanes(d)?;
    Ok((d.bits as i64 * lanes + 7) / 8)
}

/// `TensorVar(name, TensorType({extent}, Int(64)))`.
fn int64_tensor_var(name: &str, extent: i64) -> Result<Var> {
    let int64 = dtype(INT, 64, 1);
    let shape = [int_imm(int64, extent as i128)?.into()];
    Var::new(name, TensorType::new(shape, int64, MemoryScope::global()?)?)
}

/// The `TensorType` of a `TensorVar`.
fn tensor_type_of(var: &Var) -> Result<TensorType> {
    var.ty
        .clone()
        .try_cast::<TensorType>()
        .map_err(|_| error!(TYPE_ERROR, "expected a TensorVar, a variable of a TensorType"))
}

/// `TensorTypeNode::ConstantAllocationSize()`.
fn constant_allocation_size(tensor: &TensorTypeObj) -> Option<i64> {
    let mut result: i64 = 1;
    for extent in tensor.shape.iter() {
        let size = extent.as_node::<IntImmObj>()?.value.to_i64()?;
        result = result.checked_mul(size)?;
    }
    Some(result)
}

/// `IsZero(x)`: an `IntImm` 0.
fn is_zero(x: &Expr) -> bool {
    matches!(x.as_node::<IntImmObj>(), Some(imm) if imm.value == BigInt::from(0i64))
}

/// `x.as<PrimExpr>()`: an expression of a primitive type.
fn is_prim_expr(x: &Expr) -> bool {
    x.ty.as_node::<PrimTypeObj>().is_some()
}

fn prim_dtype_of(x: &Expr) -> Result<DLDataType> {
    x.ty.as_node::<PrimTypeObj>()
        .map(|t| t.dtype)
        .ok_or_else(|| error!(TYPE_ERROR, "expected a PrimExpr"))
}

/// `runtime::DLDeviceType2Str`.
fn device_type_name(device_type: i64) -> Result<&'static str> {
    Ok(match device_type {
        1 => "cpu",
        2 => "cuda",
        3 => "cuda_host",
        13 => "cuda_managed",
        4 => "opencl",
        7 => "vulkan",
        8 => "metal",
        9 => "vpi",
        10 => "rocm",
        11 => "rocm_host",
        12 => "ext_dev",
        14 => "oneapi",
        15 => "webgpu",
        16 => "hexagon",
        18 => "trn",
        other => tvm_ffi::bail!(INTERNAL_ERROR, "unknown type = {other}"),
    })
}

/// The stack slots in use, or the most a scope needs. A shape stack of `-1` is no shape at all;
/// `0` is shapes of no extents.
#[derive(Clone, Copy)]
struct StackSizes {
    shape: i64,
    array: u64,
    arg: u64,
}

impl Default for StackSizes {
    fn default() -> Self {
        Self { shape: -1, array: 0, arg: 0 }
    }
}

/// The stacks of a function body or parallel loop body (`AllocaScope`).
struct AllocaScope {
    stack_shape: Option<Var>,
    stack_array: Var,
    stack_ffi_any: Var,
    max: StackSizes,
    run: StackSizes,
}

impl AllocaScope {
    fn new() -> Result<Self> {
        Ok(Self {
            stack_shape: None,
            stack_array: Var::new("stack_array", PtrType::void()?)?,
            stack_ffi_any: Var::new("stack_ffi_any", PtrType::void()?)?,
            max: StackSizes::default(),
            run: StackSizes::default(),
        })
    }

    fn update_max(&mut self) {
        self.max.shape = self.max.shape.max(self.run.shape);
        self.max.array = self.max.array.max(self.run.array);
        self.max.arg = self.max.arg.max(self.run.arg);
    }

    fn assert_max_is_valid(&self) -> Result<()> {
        check!(
            self.max.shape >= self.run.shape
                || self.max.array >= self.run.array
                || self.max.arg >= self.run.arg,
            INTERNAL_ERROR,
            "(max_sizes.shape_stack >= run_sizes.shape_stack) || (max_sizes.array_stack >= \
             run_sizes.array_stack) || (max_sizes.arg_stack >= run_sizes.arg_stack)"
        );
        Ok(())
    }
}

struct BuiltinLower {
    /// The statements to emit before the statement being lowered, one list per statement.
    prep_seq: Vec<Vec<Stmt>>,
    device_type: Option<Expr>,
    device_id: Option<Expr>,
    /// CUDA host codegen consumes explicit kernel launches after this lowering.
    preserve_ffi_kernel: bool,
    /// The sizing run of `GetMaxStack`, which neither allocates nor enters parallel loops.
    is_precheck: bool,
    alloca_scope: Vec<AllocaScope>,
    /// The frees of `alloc_tensor` / `nd_mem_alloc_with_scope` to emit when each scope ends
    /// (`ScopeStack<ScopeLevel>`, which starts with one level).
    pending_frees: Vec<Vec<Stmt>>,
}

impl BuiltinLower {
    fn new(device_type: Option<Expr>, preserve_ffi_kernel: bool) -> Self {
        Self {
            prep_seq: Vec::new(),
            device_type,
            device_id: None,
            preserve_ffi_kernel,
            is_precheck: false,
            alloca_scope: Vec::new(),
            pending_frees: vec![Vec::new()],
        }
    }

    fn scope(&mut self) -> Result<&mut AllocaScope> {
        let scope = self.alloca_scope.last_mut();
        scope.ok_or_else(|| {
            error!(INTERNAL_ERROR, "Check failed: (!alloca_scope_.empty()) is false: ")
        })
    }

    fn prep(&mut self) -> &mut Vec<Stmt> {
        self.prep_seq.last_mut().expect("a call is lowered within a statement")
    }

    /// `scope_.WithNewScope(body)`.
    fn with_new_scope<T>(&mut self, body: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.pending_frees.push(Vec::new());
        let result = body(self);
        self.pending_frees.pop();
        result
    }

    /// `AppendPendingFrees(body)`: the frees of the current scope after `body`, newest first.
    fn append_pending_frees(&mut self, body: Stmt) -> Result<Stmt> {
        let frees = std::mem::take(self.pending_frees.last_mut().expect("a scope is active"));
        if frees.is_empty() {
            return Ok(body);
        }
        let stmts = std::iter::once(body).chain(frees.into_iter().rev());
        Ok(SeqStmt::new(stmts)?.into())
    }

    fn require_device(&self) -> Result<(Expr, Expr)> {
        let Some(device_type) = self.device_type.clone() else {
            tvm_ffi::bail!(
                INTERNAL_ERROR,
                "Check failed: (device_type_) is false: Unknown device type in current IR"
            );
        };
        let Some(device_id) = self.device_id.clone() else {
            tvm_ffi::bail!(
                INTERNAL_ERROR,
                "Check failed: (device_id_) is false: Unknown device id in current IR"
            );
        };
        Ok((device_type, device_id))
    }

    // -----------------------------------------------------------------------------------------
    // Stack allocation (`VisitBodyAndRealizeAlloca`)

    /// `GetMaxStack(stmt)`: the stack sizes `stmt` needs, from a sizing run over it.
    fn max_stack(&self, stmt: &Stmt) -> Result<StackSizes> {
        // A default-constructed precheck: it lowers `call_ffi_kernel` whatever this one does.
        let mut precheck = BuiltinLower::new(self.device_type.clone(), false);
        precheck.is_precheck = true;
        precheck.device_id = self.device_id.clone();
        let mut scope = AllocaScope::new()?;
        scope.stack_shape = Some(int64_tensor_var("stack_shape", 0)?);
        precheck.alloca_scope.push(scope);
        structural_mutate(stmt.clone(), &mut precheck)?;
        check_cmp!(
            precheck.alloca_scope.len(),
            ==,
            1,
            INTERNAL_ERROR,
            "precheck->alloca_scope_.size() == 1"
        );
        Ok(precheck.alloca_scope[0].max)
    }

    /// Open the stack scope of `stmt` and put its stack allocations ahead of it.
    fn enter_alloca_scope(&mut self, stmt: Stmt) -> Result<Stmt> {
        let mut scope = AllocaScope::new()?;
        scope.max = self.max_stack(&stmt)?;
        let mut allocations: Vec<Stmt> = Vec::new();
        if scope.max.arg != 0 {
            allocations.push(self.stack_alloca(
                &scope.stack_ffi_any,
                "tvm_ffi_any",
                scope.max.arg,
            )?);
        }
        if scope.max.array != 0 {
            allocations.push(self.stack_alloca(&scope.stack_array, "array", scope.max.array)?);
        }
        let mut stmt = stmt;
        if scope.max.shape != -1 {
            let stack_shape = int64_tensor_var("stack_shape", scope.max.shape)?;
            let tensor = tensor_type_of(&stack_shape)?;
            let data = Call::new(
                tensor.data_ptr_type()?,
                STACK_ALLOCA.get()?,
                vec![StringImm::new("shape")?.into(), int32(scope.max.shape)?.into()],
            )?;
            let decl = Call::new_full(
                Some(tensor.clone().into()),
                DECL_TENSOR.get()?.into(),
                Array::new(vec![data.into()]),
                None,
                Array::new(vec![tensor.into()]),
                unknown_loc()?,
            )?;
            let bind: Stmt = Bind::new(stack_shape.clone(), decl)?.into();
            stmt = SeqStmt::new(vec![bind, stmt])?.into();
            scope.stack_shape = Some(stack_shape);
        }
        if !allocations.is_empty() {
            allocations.push(stmt);
            stmt = SeqStmt::new(allocations)?.into();
        }
        self.alloca_scope.push(scope);
        Ok(stmt)
    }

    /// `Bind(var, stack_alloca(kind, size))`.
    fn stack_alloca(&self, var: &Var, kind: &str, size: u64) -> Result<Stmt> {
        let args = vec![StringImm::new(kind)?.into(), int32(size as i64)?.into()];
        let call = Call::new(var.ty.clone(), STACK_ALLOCA.get()?, args)?;
        Ok(Bind::new(var.clone(), call)?.into())
    }

    /// `VisitBodyAndRealizeAlloca` for the function body, the root of the rewrite.
    fn realize_body(&mut self, stmt: Stmt) -> Result<Stmt> {
        let stmt = self.enter_alloca_scope(stmt)?;
        let result = self.with_new_scope(|this| {
            let visited: Stmt = structural_mutate(stmt, &mut *this)?.try_into()?;
            this.append_pending_frees(visited)
        });
        self.alloca_scope.pop();
        result
    }

    /// `VisitBodyAndRealizeAlloca` for a parallel loop body, reached during the rewrite.
    fn realize_loop_body(&mut self, stmt: Stmt, m: &mut Mutator) -> Result<Stmt> {
        if self.is_precheck {
            return Ok(stmt);
        }
        let stmt = self.enter_alloca_scope(stmt)?;
        let result = self.with_new_scope(|this| {
            let visited: Stmt = m.mutate_as(this, &stmt)?;
            this.append_pending_frees(visited)
        });
        self.alloca_scope.pop();
        result
    }

    /// `Mutate(stmt)` in a fresh free scope, with the scope's frees after it.
    fn visit_scoped(&mut self, stmt: Stmt, m: &mut Mutator) -> Result<Stmt> {
        self.with_new_scope(|this| {
            let visited: Stmt = m.mutate_as(this, &stmt)?;
            this.append_pending_frees(visited)
        })
    }

    // -----------------------------------------------------------------------------------------
    // Statements

    /// The `Mutate_` overloads for statements; any other statement takes the default descent.
    fn lower_stmt(&mut self, stmt: Stmt, m: &mut Mutator) -> Result<Stmt> {
        if stmt.as_node::<BindObj>().is_some() {
            return self.lower_bind(stmt.try_cast()?, m);
        }
        if stmt.as_node::<RegionStmtObj>().is_some() {
            return self.lower_region(stmt.try_cast()?, m);
        }
        if stmt.as_node::<ForObj>().is_some() {
            return self.lower_for(stmt.try_cast()?, m);
        }
        if stmt.as_node::<IfObj>().is_some() {
            return self.lower_if(stmt.try_cast()?, m);
        }
        m.default_mutate(self, &stmt)?.try_into()
    }

    fn lower_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Stmt> {
        if let Some(call) = op.value.as_node::<CallObj>() {
            if ALLOC_TENSOR.is(&call.op) {
                return self.lower_alloc_tensor(op, m);
            }
            if OPENCL_ND_MEM_ALLOC_WITH_SCOPE.is(&call.op) {
                return self.lower_nd_mem_alloc(op, m);
            }
        }
        m.default_mutate(self, &op)?.try_into()
    }

    /// `MutateAllocTensor`: a device tensor, or one too large for the CPU stack, is allocated
    /// from the workspace and freed when its scope ends.
    fn lower_alloc_tensor(&mut self, op: Bind, m: &mut Mutator) -> Result<Stmt> {
        let stmt: Stmt = m.default_mutate(self, &op)?.try_into()?;
        let bind: Bind = stmt.clone().try_cast()?;
        let tensor_call: Call = bind.value.clone().try_cast()?;
        let alloc_type: TensorType = tensor_call.ty_args.get(0)?.try_cast()?;
        let element = alloc_type.dtype.dtype;
        let Some(attrs) = tensor_call.attrs.clone() else {
            tvm_ffi::bail!(INTERNAL_ERROR, "bad optional access: alloc_tensor has no attrs");
        };
        let annotations: DictAttrs = attrs.try_cast()?;
        if let Some(disable) = annotations.dict.get(&String::from(DISABLE_LOWER_BUILTIN))? {
            if int_attr(disable)? != BigInt::from(0i64) {
                return Ok(stmt);
            }
        }
        if is_scalable(element) {
            return Ok(stmt);
        }
        let nbytes = storage_bytes(element)?;
        let on_cpu = matches!(
            self.device_type.as_ref().and_then(|t| t.as_node::<IntImmObj>()),
            Some(imm) if imm.value == BigInt::from(DL_CPU)
        );
        let var_type = tensor_type_of(&bind.var)?;
        if on_cpu && alloc_type.scope.same_as(&MemoryScope::global()?) {
            if let Some(size) = constant_allocation_size(&var_type) {
                if size > 0 && (size as u64).wrapping_mul(nbytes as u64) < MAX_STACK_ALLOCA {
                    return Ok(stmt);
                }
            }
        }
        let mut total_bytes = Ex::from(int_imm(dtype(UINT, 64, 1), nbytes as i128));
        for extent in alloc_type.shape.iter() {
            total_bytes *= extent;
        }
        let total_bytes = total_bytes.expr()?;
        let (device_type, device_id) = self.require_device()?;
        let int32_ty = ty(INT, 32)?;
        let throw_last_error: Stmt =
            Evaluate::new(Call::new(int32_ty.clone(), THROW_LAST_ERROR.get()?, vec![])?)?.into();
        let data: Expr = Call::new(
            var_type.data_ptr_type()?,
            TENSOR_DATA_PTR.get()?,
            vec![bind.var.clone().into()],
        )?
        .into();
        let is_null = Call::new(ty(BOOL, 8)?, ISNULLPTR.get()?, vec![data.clone()])?;
        let alloc_nullptr_check: Stmt = If::new(is_null, throw_last_error.clone(), None)?.into();

        let device_type = device_type.cast(int32_ty.clone()).expr()?;
        let device_id = device_id.cast(int32_ty.clone()).expr()?;
        let free = Call::new(
            int32_ty.clone(),
            FREE_WORKSPACE.get()?,
            vec![device_type.clone(), device_id.clone(), data],
        )?;
        let failed = Ex::from(free).not_equal(int32(0)?).expr()?;
        let free_stmt: Stmt = If::new(failed, throw_last_error, None)?.into();
        self.pending_frees.last_mut().expect("a scope is active").push(free_stmt);

        let alloc = Call::new(
            var_type.data_ptr_type()?,
            ALLOC_WORKSPACE.get()?,
            vec![
                device_type,
                device_id,
                total_bytes,
                int32(element.code as i64)?.into(),
                int32(element.bits as i64)?.into(),
            ],
        )?;
        let decl = Call::new_full(
            Some(var_type.clone().into()),
            DECL_TENSOR.get()?.into(),
            Array::new(vec![alloc.into()]),
            None,
            Array::new(vec![var_type.into()]),
            tensor_call.loc.clone(),
        )?;
        let alloc_bind: Stmt = Bind::new_full(bind.var.clone(), decl, bind.loc.clone())?.into();
        Ok(SeqStmt::new(vec![alloc_bind, alloc_nullptr_check])?.into())
    }

    /// `MakeNdMemAllocWithScope`: an OpenCL image allocation through the device API, freed when
    /// its scope ends.
    fn lower_nd_mem_alloc(&mut self, op: Bind, m: &mut Mutator) -> Result<Stmt> {
        let call: Call = op.value.clone().try_cast()?;
        self.require_device()?;
        let (device_type, device_id) = self.require_device()?;
        let int32_ty = ty(INT, 32)?;
        let throw_last_error: Stmt =
            Evaluate::new(Call::new(int32_ty.clone(), THROW_LAST_ERROR.get()?, vec![])?)?.into();
        let element = op
            .var
            .ty
            .as_node::<PtrTypeObj>()
            .and_then(|ptr| ptr.element_type.as_node::<PrimTypeObj>())
            .map(|element| element.dtype)
            .ok_or_else(|| error!(INTERNAL_ERROR, "Check failed: (dtype_node) is false: "))?;
        let mut args: Vec<Expr> = vec![
            self.device_method_name("alloc_nd")?.into(),
            device_type.clone(),
            device_id.clone(),
            int32(element.code as i64)?.into(),
            int32(element.bits as i64)?.into(),
        ];
        args.extend(call.args.iter());
        let call_packed = Call::new(op.var.ty.clone(), CALL_PACKED.get()?, args)?;
        let is_null = Call::new(ty(BOOL, 8)?, ISNULLPTR.get()?, vec![op.var.clone().into()])?;
        let null_check: Stmt = If::new(is_null, throw_last_error.clone(), None)?.into();

        let free = Call::new(
            int32_ty,
            CALL_PACKED.get()?,
            vec![
                self.device_method_name("free_nd")?.into(),
                device_type,
                device_id,
                call.args.get(0)?,
                op.var.clone().into(),
            ],
        )?;
        let failed = Ex::from(free).not_equal(int32(0)?).expr()?;
        let free_stmt: Stmt = If::new(failed, throw_last_error, None)?.into();
        // `StmtExprMutator::Mutate`, the base one: the free's preparation joins this statement's.
        let free_stmt = self.lower_stmt(free_stmt, m)?;
        self.pending_frees.last_mut().expect("a scope is active").push(free_stmt);

        let bind: Stmt = Bind::new(op.var.clone(), call_packed)?.into();
        let input: Stmt = SeqStmt::new(vec![bind, null_check])?.into();
        self.lower_stmt(input, m)
    }

    /// `GetDeviceMethodName(method)`: `device_api.<device>.<method>`.
    fn device_method_name(&self, method: &str) -> Result<StringImm> {
        let Some(device_type) = &self.device_type else {
            tvm_ffi::bail!(
                INTERNAL_ERROR,
                "Check failed: (device_type_) is false: Method {method} requires the device type, \
                 but occurred outside of a \"device_context\" region"
            );
        };
        let Some(imm) = device_type.as_node::<IntImmObj>() else {
            tvm_ffi::bail!(
                INTERNAL_ERROR,
                "Check failed: (as_int) is false: Method {method} requires the device type to be a \
                 DLDeviceType enum value"
            );
        };
        let device = imm.value.to_i64().unwrap_or(-1);
        StringImm::new(&format!("device_api.{}.{method}", device_type_name(device)?))
    }

    fn lower_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<Stmt> {
        if DEVICE_CONTEXT.is(&op.op) {
            let device_type: Expr = m.mutate_as(self, &op.args.get(0)?)?;
            let device_id: Expr = m.mutate_as(self, &op.args.get(1)?)?;
            let saved_type = self.device_type.replace(device_type);
            let saved_id = self.device_id.replace(device_id);
            let body = self.visit_scoped(op.body.clone().into(), m);
            self.device_type = saved_type;
            self.device_id = saved_id;
            return body;
        }
        self.with_new_scope(|this| {
            let region: RegionStmt = m.default_mutate(this, &op)?.try_into()?;
            if this.pending_frees.last().is_some_and(|frees| frees.is_empty()) {
                return Ok(region.into());
            }
            let body = this.append_pending_frees(region.body.clone().into())?;
            Ok(region.with_body(SeqStmt::from_stmt(body)?).into())
        })
    }

    fn lower_for(&mut self, op: For, m: &mut Mutator) -> Result<Stmt> {
        let min: Expr = m.mutate_as(self, &op.min)?;
        let extent: Expr = m.mutate_as(self, &op.extent)?;
        let body: Stmt = if op.kind == FOR_PARALLEL {
            self.realize_loop_body(op.body.clone().into(), m)?
        } else {
            self.visit_scoped(op.body.clone().into(), m)?
        };
        if min.same_as(&op.min) && extent.same_as(&op.extent) && body.same_as(&op.body) {
            return Ok(op.into());
        }
        let body = SeqStmt::from_stmt(body)?;
        Ok(op
            .with(|node| {
                node.min = min;
                node.extent = extent;
                node.body = body;
            })
            .into())
    }

    /// Each arm frees what it allocates.
    fn lower_if(&mut self, op: If, m: &mut Mutator) -> Result<Stmt> {
        let condition: Expr = m.mutate_as(self, &op.condition)?;
        let then_case = self.visit_scoped(op.then_case.clone().into(), m)?;
        let else_case = match &op.else_case {
            Some(else_case) => {
                Some(SeqStmt::from_stmt(self.visit_scoped(else_case.clone().into(), m)?)?)
            }
            None => None,
        };
        if condition.same_as(&op.condition)
            && then_case.same_as(&op.then_case)
            && else_case.same_field(&op.else_case)
        {
            return Ok(op.into());
        }
        Ok(If::new_full(condition, then_case, else_case, op.loc.clone())?.into())
    }

    // -----------------------------------------------------------------------------------------
    // Calls

    fn lower_call(&mut self, call: Call, m: &mut Mutator) -> Result<Expr> {
        let op = &call.op;
        if CUDA_TENSORMAP_ENCODE_TILED.is(op) && !self.preserve_ffi_kernel {
            let packed = self.tensormap_packed_call(&call)?;
            return self.make_call_packed(packed, CALL_PACKED_LOWERED.get()?, m);
        }
        if CALL_PACKED.is(op) || (CALL_FFI_KERNEL.is(op) && !self.preserve_ffi_kernel) {
            return self.make_call_packed(call, CALL_PACKED_LOWERED.get()?, m);
        }
        if CALL_CPACKED.is(op) {
            return self.make_call_packed(call, CALL_CPACKED_LOWERED.get()?, m);
        }
        if STACK_MAKE_SHAPE.is(op) {
            return self.make_shape(call, m);
        }
        if STACK_MAKE_DLTENSOR.is(op) {
            return self.make_array(call, m);
        }
        m.default_mutate(self, &call)?.try_into()
    }

    /// `tensormap_encode_tiled(...)` as the packed call of `runtime.cuTensorMapEncodeTiled`.
    fn tensormap_packed_call(&self, call: &Call) -> Result<Call> {
        let attr =
            call.attrs.as_ref().and_then(|attrs| attrs.as_node::<TensorMapEncodeTiledAttrObj>());
        let Some(attr) = attr.filter(|attr| {
            (1..=5).contains(&attr.rank) && call.args.len() as i64 == 4 * attr.rank + 1
        }) else {
            tvm_ffi::bail!(
                VALUE_ERROR,
                "Check failed: (attr && attr->rank >= 1 && attr->rank <= 5 && op->args.size() == \
                 static_cast<size_t>(4 * attr->rank + 1)) is false: Invalid tensormap_encode_tiled \
                 attributes or operands"
            );
        };
        let int32_imm = |value: i64| -> Result<Expr> { Ok(int32(value)?.into()) };
        let mut args: Vec<Expr> = vec![
            StringImm::new("runtime.cuTensorMapEncodeTiled")?.into(),
            call.args.get(0)?,
            StringImm::new(attr.descriptor_dtype.to_string().as_str())?.into(),
            int32_imm(attr.rank)?,
            call.args.get(1)?,
        ];
        args.extend(call.args.iter().skip(2));
        for value in [attr.interleave, attr.swizzle, attr.l2_promotion, attr.oob_fill] {
            args.push(int32_imm(value)?);
        }
        if attr.force_cu_dtype != -1 {
            args.push(int32_imm(attr.force_cu_dtype)?);
        }
        Call::new(call.ty.clone(), CALL_PACKED.get()?, args)
    }

    /// `MakeShape`: the extents stored on the shape stack, passed as the address of the first.
    fn make_shape(&mut self, call: Call, m: &mut Mutator) -> Result<Expr> {
        let begin = {
            let scope = self.scope()?;
            if scope.run.shape == -1 {
                scope.run.shape = 0;
            }
            let begin = scope.run.shape;
            scope.run.shape += call.args.len() as i64;
            begin
        };
        let call: Call = m.default_mutate(self, &call)?.try_into()?;
        let Some(stack_shape) = self.scope()?.stack_shape.clone() else {
            tvm_ffi::bail!(INTERNAL_ERROR, "bad optional access: no shape stack");
        };
        let int64 = ty(INT, 64)?;
        for (i, arg) in call.args.iter().enumerate() {
            let index: Expr = int32(begin + i as i64)?.into();
            let value = arg.cast(int64.clone()).expr()?;
            let store = TensorStore::new(stack_shape.clone(), [index], value)?;
            self.prep().push(store.into());
        }
        let load = TensorLoad::new(stack_shape.clone(), vec![int32(begin)?.into()])?;
        let tensor = tensor_type_of(&stack_shape)?;
        Ok(Call::new(tensor.data_ptr_type()?, ADDRESS_OF.get()?, vec![load.into()])?.into())
    }

    /// `abi_field_set(stack, index, field, value)` as a statement.
    fn set_field(&self, stack: &Var, index: i64, field: i32, value: Expr) -> Result<Stmt> {
        let args =
            vec![stack.clone().into(), int32(index)?.into(), int32(field as i64)?.into(), value];
        let call = Call::new(ty(INT, 32)?, ABI_FIELD_SET.get()?, args)?;
        Ok(Evaluate::new(call)?.into())
    }

    /// `MakeArray`: a `DLTensor` filled in on the array stack, passed by its address.
    fn make_array(&mut self, call: Call, m: &mut Mutator) -> Result<Expr> {
        let index = {
            let scope = self.scope()?;
            let index = scope.run.array;
            scope.run.array += 1;
            index as i64
        };
        let call: Call = m.default_mutate(self, &call)?.try_into()?;
        let stack = self.scope()?.stack_array.clone();
        let arg = |i: usize| call.args.get(i);
        let mut prep: Vec<Stmt> = vec![
            self.set_field(&stack, index, DLTENSOR_DATA, arg(0)?)?,
            self.set_field(&stack, index, DLTENSOR_SHAPE, arg(1)?)?,
        ];
        let mut strides = arg(2)?;
        if is_prim_expr(&strides) && is_zero(&strides) {
            // `ConstHandle(0)`.
            let zero: Expr = int_imm(dtype(UINT, 64, 1), 0)?.into();
            strides = zero.reinterpret(PtrType::void()?).expr()?;
        }
        prep.push(self.set_field(&stack, index, DLTENSOR_STRIDES, strides)?);
        let ndim = arg(3)?;
        prim_dtype_of(&ndim)?;
        prep.push(self.set_field(&stack, index, DLTENSOR_NDIM, ndim)?);
        let element = prim_dtype_of(&arg(4)?)?;
        let byte = |code: u8, bits: u8, value: i64| -> Result<Expr> {
            Ok(int_imm(dtype(code, bits, 1), value as i128)?.into())
        };
        prep.push(self.set_field(
            &stack,
            index,
            DLTENSOR_TYPE_CODE,
            byte(UINT, 8, element.code as i64)?,
        )?);
        prep.push(self.set_field(
            &stack,
            index,
            DLTENSOR_TYPE_BITS,
            byte(UINT, 8, element.bits as i64)?,
        )?);
        prep.push(self.set_field(
            &stack,
            index,
            DLTENSOR_TYPE_LANES,
            byte(UINT, 16, lanes(element)?)?,
        )?);
        let data_bytes = storage_bytes(element)?;
        let elem_offset = arg(5)?;
        let offset_dtype = prim_dtype_of(&elem_offset)?;
        let byte_offset = match is_zero(&elem_offset) {
            true => elem_offset,
            false => {
                let size: Expr = int_imm(offset_dtype, data_bytes as i128)?.into();
                (Ex::from(elem_offset) * size).expr()?
            }
        };
        let byte_offset = byte_offset.cast(dtype(UINT, 64, 1)).expr()?;
        prep.push(self.set_field(&stack, index, DLTENSOR_BYTE_OFFSET, byte_offset)?);
        let (device_type, device_id) = self.require_device()?;
        let device_id = device_id.cast(dtype(INT, 32, 1)).expr()?;
        prep.push(self.set_field(&stack, index, DLTENSOR_DEVICE_ID, device_id)?);
        let device_type = device_type.cast(dtype(INT, 32, 1)).expr()?;
        prep.push(self.set_field(&stack, index, DLTENSOR_DEVICE_TYPE, device_type)?);
        self.prep().extend(prep);
        let args = vec![stack.into(), int32(index)?.into(), int32(DLTENSOR_ADDR as i64)?.into()];
        Ok(Call::new(PtrType::void()?, ABI_FIELD_GET.get()?, args)?.into())
    }

    /// `IsArrayHandle`: the address of a `DLTensor` made by `stack_make_dltensor`.
    fn is_array_handle(&self, arg: &Expr) -> bool {
        let Some(call) = arg.as_node::<CallObj>() else { return false };
        ABI_FIELD_GET.is(&call.op)
            && call.args.get(2).ok().is_some_and(|field| {
                matches!(field.as_node::<IntImmObj>(), Some(imm) if imm.value == BigInt::from(DLTENSOR_ADDR as i64))
            })
    }

    /// `SetPackedArg`: argument `offset` of the argument stack set to `arg`.
    fn set_packed_arg(&mut self, arg: Expr, stack: &Var, offset: i64) -> Result<()> {
        let mut arg = arg;
        let type_index = if arg.as_node::<StringImmObj>().is_some() {
            arg = arg.reinterpret(PtrType::void()?).expr()?;
            FFI_RAW_STR
        } else if arg.ty.as_node::<PtrTypeObj>().is_some() {
            if self.is_array_handle(&arg) {
                FFI_DLTENSOR_PTR
            } else {
                FFI_OPAQUE_PTR
            }
        } else {
            let arg_type = prim_dtype_of(&arg)?;
            let api_type = api_type(arg_type)?;
            if arg_type != api_type {
                arg = Cast::new(arg, prim_type(api_type)?)?.into();
            }
            match api_type.code {
                BOOL => FFI_BOOL,
                INT | UINT => FFI_INT,
                FLOAT => FFI_FLOAT,
                _ => {
                    let api = prim_type(api_type)?;
                    tvm_ffi::bail!(INTERNAL_ERROR, "Unsupported type: {}", repr(&api));
                }
            }
        };
        let type_slot = |value: i32| -> Result<Stmt> {
            self.set_field(stack, offset, FFI_ANY_TYPE_INDEX, int32(value as i64)?.into())
        };
        let kind: Stmt = if type_index == FFI_OPAQUE_PTR {
            // An opaque handle that is null is passed as None.
            let is_null = Call::new(ty(BOOL, 8)?, ISNULLPTR.get()?, vec![arg.clone()])?;
            let opaque = SeqStmt::from_stmt(type_slot(FFI_OPAQUE_PTR)?)?;
            If::new(is_null, type_slot(FFI_NONE)?, Some(opaque))?.into()
        } else {
            type_slot(type_index)?
        };
        let padding = self.set_field(stack, offset, FFI_ANY_ZERO_PADDING, int32(0)?.into())?;
        let value = self.set_field(stack, offset, FFI_ANY_UNION_VALUE, arg)?;
        self.prep().extend([kind, padding, value]);
        Ok(())
    }

    /// `MakeCallPackedGeneric(op, 0, lowered)`: the arguments of a packed call set on the argument
    /// stack, and the call made to `lowered` with the stack and the slot range.
    fn make_call_packed(&mut self, call: Call, lowered: Op, m: &mut Mutator) -> Result<Expr> {
        let (restore_shape, restore_array, arg_begin) = {
            let scope = self.scope()?;
            (scope.run.shape, scope.run.array, scope.run.arg)
        };
        let num_args = call.args.len().wrapping_sub(1) as u64;
        // The extra slot holds the return value.
        self.scope()?.run.arg += num_args + 1;
        let call: Call = m.default_mutate(self, &call)?.try_into()?;
        let stack = self.scope()?.stack_ffi_any.clone();
        for i in 0..num_args {
            let arg = call.args.get(1 + i as usize)?;
            check!(
                is_prim_expr(&arg)
                    || arg.ty.as_node::<PtrTypeObj>().is_some()
                    || arg.as_node::<StringImmObj>().is_some(),
                TYPE_ERROR,
                "arg.as<PrimExpr>() || arg->ty.as<PtrTypeNode>() || arg.as<StringImmNode>()",
                "Packed call argument must have a primitive, pointer, or string type, but got {}",
                repr(&arg.ty)
            );
            self.set_packed_arg(arg, &stack, (arg_begin + i) as i64)?;
        }
        // The return value starts as None.
        let ret = num_args as i64;
        let prep = vec![
            self.set_field(&stack, ret, FFI_ANY_TYPE_INDEX, int32(FFI_NONE as i64)?.into())?,
            self.set_field(&stack, ret, FFI_ANY_ZERO_PADDING, int32(0)?.into())?,
            self.set_field(
                &stack,
                ret,
                FFI_ANY_UNION_VALUE,
                int_imm(dtype(INT, 64, 1), 0)?.into(),
            )?,
        ];
        self.prep().extend(prep);
        let is_precheck = self.is_precheck;
        let scope = self.scope()?;
        if is_precheck {
            scope.update_max();
        } else {
            scope.assert_max_is_valid()?;
        }
        scope.run.shape = restore_shape;
        scope.run.array = restore_array;
        scope.run.arg = arg_begin;
        let args = vec![
            call.args.get(0)?,
            stack.into(),
            int32(arg_begin as i64)?.into(),
            int32((arg_begin + num_args) as i64)?.into(),
        ];
        Ok(Call::new(call.ty.clone(), lowered, args)?.into())
    }
}

#[dispatch(mutate)]
impl BuiltinLower {
    /// C++ `Mutate` for a statement: the preparation its calls need goes before it.
    fn mutate_stmt(&mut self, op: Stmt, m: &mut Mutator) -> Result<Any> {
        self.prep_seq.push(Vec::new());
        let scope_size = self.alloca_scope.len();
        let stmt = self.lower_stmt(op, m)?;
        tvm_ffi::ensure!(
            self.alloca_scope.len() == scope_size,
            INTERNAL_ERROR,
            "alloca_scope_ length is different before and after recursion"
        );
        let scope = self.scope()?;
        tvm_ffi::ensure!(
            scope.run.shape == -1,
            INTERNAL_ERROR,
            "Expect no stack_make_shape outside of CallNodes"
        );
        tvm_ffi::ensure!(
            scope.run.array == 0,
            INTERNAL_ERROR,
            "Expect no stack_make_dltensor outside of CallNodes"
        );
        let mut prep = self.prep_seq.pop().expect("pushed above");
        if prep.is_empty() {
            return Ok(stmt.into());
        }
        prep.push(stmt);
        Ok(SeqStmt::new(prep)?.into())
    }

    fn mutate_call(&mut self, call: Call, m: &mut Mutator) -> Result<Any> {
        Ok(self.lower_call(call, m)?.into())
    }
}

/// `APIType(t)`: the type a value of type `t` is passed through the packed API as.
fn api_type(t: DLDataType) -> Result<DLDataType> {
    let is_void = t.code == HANDLE && t.bits == 0 && t.lanes as i16 == 0;
    check!(!is_void, INTERNAL_ERROR, "!t.IsVoid()", "Cannot pass void type through packed API.");
    check_cmp!(
        lanes(t)?,
        ==,
        1,
        INTERNAL_ERROR,
        "t.lanes() == 1",
        "Cannot pass vector type through packed API."
    );
    if matches!(t.code, BOOL | UINT | INT) {
        return Ok(dtype(INT, 64, 1));
    }
    check!(
        matches!(t.code, FLOAT | BFLOAT),
        INTERNAL_ERROR,
        "t.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat)"
    );
    Ok(dtype(FLOAT, 64, 1))
}

/// An integer annotation, read as C++ `value.as_or_throw<IntImm>()->value`.
fn int_attr(value: Any) -> Result<BigInt> {
    if let Some(value) = value.try_as::<i64>() {
        return Ok(BigInt::from(value));
    }
    if let Some(value) = value.try_as::<bool>() {
        return Ok(BigInt::from(value as i64));
    }
    let imm: crate::ir::IntImm = value.try_into()?;
    Ok(imm.value.clone())
}

/// `target->GetHost()`.
fn host_of(target: &Target) -> Result<Option<Target>> {
    target.host()?.map(|host| host.try_cast::<Target>()).transpose()
}

/// `BuiltinLower::Build(func)`.
fn build(func: Function) -> Result<Function> {
    let Some(body) = func.body.clone() else { return Ok(func) };
    let (device_type, preserve_ffi_kernel) = match get_attr(&func.attrs, TARGET)? {
        Some(target) => {
            let target: Target = target.try_into()?;
            let device_type: Expr = int32(target.kind()?.default_device_type()?)?.into();
            let host = host_of(&target)?.unwrap_or(target);
            (Some(device_type), host.kind()?.name()?.as_str() == "cuda_host")
        }
        None => (None, false),
    };
    let mut lower = BuiltinLower::new(device_type, preserve_ffi_kernel);
    let body = lower.realize_body(body.into())?;
    Ok(func.with_body(Some(SeqStmt::from_stmt(body)?)))
}

pub fn lower_tvm_builtin() -> Result<Pass> {
    function_pass("tirx.LowerTVMBuiltin", 0, move |func, _module, _context| {
        let mut is_host = has_nonzero_attr(&func.attrs, IS_HOST_FUNC)?;
        if !is_host {
            if let Some(target) = get_attr(&func.attrs, TARGET)? {
                let target: Target = target.try_into()?;
                is_host = target.keys()?.iter().any(|key| key.as_str() == "cpu");
            }
        }
        if !is_host {
            return Ok(Some(func));
        }
        build(func).map(Some)
    })
}
