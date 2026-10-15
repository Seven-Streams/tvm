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

//! `tirx.ForceNarrowIndexToInt32` (`src/tirx/transform/force_narrow_index_to_i32.cc`): make
//! every index and every `int64` constant `int32`, rejecting integer tensors wider than that.
//!
//! The C++ `IndexDataTypeNormalizer::Rewrite` first collects the variables used in indices with
//! a separate rewriter that only observes, then replaces each of them (and every integer scalar
//! parameter) with an `int32` copy before rewriting the function.

use tvm_ffi::{
    dispatch, structural_mutate, Any, Array, DefRegionKind, Mutator, ObjectRefCast, ObjectRefCore,
    Result, UnchangedOr, INTERNAL_ERROR,
};

use super::data_type_rewriter::{
    copy_with_type, tensor_dtype, DataTypeRewrite, DataTypeRewriterBase, DataTypeRewriterState,
};
use super::util::{MutatorExt, function_pass, int32_dtype, prim_dtype, prim_type};
use crate::ir::{Bind, CallObj, Expr, IntImm, PrimType, SeqStmt, Stmt, Var};
use crate::ops::repr;
use crate::prim::Cast;
use crate::runtime::MemoryScope;
use crate::tirx::op::ALLOC_TENSOR;
use crate::tirx::{Function, TensorTypeObj};
use crate::transform::Pass;

const DL_INT: u8 = 0;

/// `IndexDataTypeNormalizer::CanRewriteDType`: a signed integer of 32 bits or more.
fn can_rewrite(ty: &PrimType) -> bool {
    ty.dtype.code == DL_INT && ty.dtype.bits >= 32
}

/// `param.as<TensorVar>()`: a variable whose type is a tensor type.
fn is_tensor_var(var: &Var) -> bool {
    var.ty.as_node::<TensorTypeObj>().is_some()
}

/// `IndexVarCollector`: the variables a function uses in indices, in the order C++ seeds them.
struct IndexVarCollector {
    base: DataTypeRewriterBase,
    seeds: Vec<Var>,
}

impl DataTypeRewriterState for IndexVarCollector {
    fn base(&mut self) -> &mut DataTypeRewriterBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = DataTypeRewrite)]
impl IndexVarCollector {
    /// A use of a variable in an index.
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        if m.def_region_kind() == DefRegionKind::None && self.base.is_enabled {
            self.seeds.push(var.clone());
        }
        m.default_mutate(self, &var)
    }

    /// The body, then each parameter: a tensor's shape and strides are visited, any other
    /// parameter is seeded directly. Nothing is rewritten.
    fn mutate_function(&mut self, func: Function, m: &mut Mutator) -> Result<Any> {
        if let Some(body) = &func.body {
            m.mutate(self, body)?;
        }
        for param in func.params.iter() {
            if is_tensor_var(&param) {
                m.mutate_with(self, &param, DefRegionKind::Simple)?;
            } else {
                self.seeds.push(param);
            }
        }
        Ok(UnchangedOr::<Any>::unchanged().into())
    }
}

/// `Int32DTypeNarrower`, an `IndexDataTypeNormalizer` to `int32`.
struct Int32Narrower {
    base: DataTypeRewriterBase,
    target: PrimType,
    local: MemoryScope,
    /// The function before rewriting, for error messages.
    func: Function,
}

impl DataTypeRewriterState for Int32Narrower {
    fn base(&mut self) -> &mut DataTypeRewriterBase {
        &mut self.base
    }
}

#[dispatch(mutate, policy = DataTypeRewrite)]
impl Int32Narrower {
    /// Every `int64` constant becomes `int32`, index or not.
    fn mutate_int_imm(&mut self, op: IntImm) -> Result<Any> {
        let dtype = prim_dtype(&op.clone().into())?;
        if !(dtype.code == DL_INT && dtype.bits == 64 && dtype.lanes == 1) {
            return Ok(UnchangedOr::<Any>::unchanged().into());
        }
        let Some(value) = op.value.to_i64().filter(|v| *v <= i32::MAX as i64) else {
            tvm_ffi::bail!(
                INTERNAL_ERROR,
                "Check failed: op->value <= \
                 prim::max_value(target_data_type_).as_or_throw<IntImm>()->value ({} vs. {}) : ",
                op.value,
                i32::MAX
            );
        };
        Ok(super::util::int32(value)?.into())
    }

    /// `IndexDataTypeNormalizer`: a wide integer cast in an index becomes a cast to `int32`, or
    /// its value if that is `int32` already.
    fn mutate_cast(&mut self, op: Cast, m: &mut Mutator) -> Result<Any> {
        let ty: PrimType = op.ty.clone().try_cast()?;
        if !(self.base.is_enabled && can_rewrite(&ty)) {
            return m.default_mutate(self, &op);
        }
        let value: Expr = m.mutate_as(self, &op.value)?;
        if prim_dtype(&value)? == self.target.dtype {
            return Ok(value.into());
        }
        Ok(Cast::new(value, self.target.clone())?.into())
    }

    /// An allocated tensor may not have an integer type wider than `int32`, unless it is the
    /// local scalar storage of a TVMScript assignment.
    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        let allocates = op.value.as_node::<CallObj>().is_some_and(|call| ALLOC_TENSOR.is(&call.op));
        let result = m.default_mutate(self, &op)?;
        if allocates {
            let alloc: Bind = result.clone().try_into()?;
            self.check_allocated_tensor(&alloc.var)?;
        }
        Ok(result)
    }

    /// `IndexDataTypeNormalizer::Rewrite`'s second half: the parameters as definitions, then the
    /// body; the function is copied without running its constructor.
    fn mutate_function(&mut self, func: Function, m: &mut Mutator) -> Result<Any> {
        let mut params = Vec::with_capacity(func.params.len());
        for param in func.params.iter() {
            params.push(m.mutate_with(self, &param, DefRegionKind::Simple)?.try_into()?);
        }
        let body = match &func.body {
            Some(body) => Some(SeqStmt::from_stmt(Stmt::try_from(m.mutate(self, body)?)?)?),
            None => None,
        };
        Ok(func
            .with(|node| {
                node.params = Array::new(params);
                node.body = body;
            })
            .into())
    }
}

impl Int32Narrower {
    fn check_allocated_tensor(&self, tensor: &Var) -> Result<()> {
        let Some(ty) = tensor.ty.as_node::<TensorTypeObj>() else { return Ok(()) };
        let is_scalar = tensor.tensor_is_scalar(true)?;
        let is_local_scalar = ty.scope.same_as(&self.local) && is_scalar;
        let dtype = tensor_dtype(&tensor.clone().into())?;
        if !is_local_scalar && dtype.dtype.code == DL_INT && dtype.dtype.bits > 32 {
            tvm_ffi::bail!(
                INTERNAL_ERROR,
                "The tensor {} allocated in the function has dtype {}. The function is {}",
                repr(tensor),
                repr(&dtype),
                repr(&self.func)
            );
        }
        Ok(())
    }
}

/// `CheckTensorParams`: no integer tensor parameter may be wider than `int32`.
fn check_tensor_params(func: &Function) -> Result<()> {
    for param in func.params.iter() {
        if !is_tensor_var(&param) {
            continue;
        }
        let dtype = tensor_dtype(&param.clone().into())?;
        if dtype.dtype.code == DL_INT && dtype.dtype.bits > 32 {
            tvm_ffi::bail!(
                INTERNAL_ERROR,
                "The tensor parameter {} has dtype {}. The function is {}",
                repr(&param),
                repr(&dtype),
                repr(func)
            );
        }
    }
    Ok(())
}

/// C++ `ForceNarrowIndexToInt32(func)`.
fn rewrite(func: Function) -> Result<Function> {
    check_tensor_params(&func)?;
    let mut collector = IndexVarCollector { base: DataTypeRewriterBase::new(true), seeds: vec![] };
    structural_mutate(func.clone(), &mut collector)?;
    let mut narrower = Int32Narrower {
        base: DataTypeRewriterBase::new(true),
        target: prim_type(int32_dtype())?,
        local: MemoryScope::get("local")?,
        func: func.clone(),
    };
    narrower.base.clamp_shift_amounts = true;
    for var in collector.seeds {
        let Ok(ty) = var.ty.clone().try_cast::<PrimType>() else { continue };
        if can_rewrite(&ty)
            && ty.dtype != narrower.target.dtype
            && narrower.base.remap_get(&var).is_none()
        {
            let narrowed = copy_with_type(&var, narrower.target.clone());
            narrower.base.remap_set(&var, narrowed);
        }
    }
    structural_mutate(func, &mut narrower)?.try_into()
}

pub fn force_narrow_index_to_i32() -> Result<Pass> {
    function_pass("tirx.NarrowDataType", 0, |func, _module, _context| Ok(Some(rewrite(func)?)))
}
