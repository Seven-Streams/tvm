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

//! `tirx.VectorizeLoop` (`src/tirx/transform/vectorize_loop.cc`): replace each vectorized loop by
//! its body over a `Ramp` of the loop variable, widening the expressions it touches; a statement
//! that cannot be widened becomes a serial loop again ("scalarized").
//!
//! The C++ `VecAllocAccess` and `SkipVectorize` are not used by the pass and are not ported.

use std::cell::RefCell;
use std::collections::HashMap;

use tvm_ffi::{
    dispatch, structural_map, structural_mutate, structural_walk, Any, AnyView, Array, BigInt,
    DLDataType, DLDataTypeCode, DLDataTypeExt, DefRegionKind, Error, Function as FfiFunction, Map,
    Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String, UnchangedOr, WalkOrder,
    WalkResult,
};

use super::util::{MutatorExt, function_pass, get_attr, internal};
use crate::ir::{
    Bind, Call, Evaluate, Expr, For, GlobalVar, If, IntImmObj, Op, OpAttrMap, OpaqueExpr, PrimType,
    PrimTypeObj, SeqStmt, Stmt, TensorLoad, TensorRegion, TensorStore, Tuple, TupleGetItem, Var,
    While,
};
use crate::ops::{int32, int_imm, make_const, prim_dtype, prim_type, unknown_loc, Ex, PrimOps};
use crate::prim::{
    Broadcast, BroadcastObj, Cast, FloorModObj, Let, Mul, Ramp, RampObj, Select, Shuffle, LTObj,
};
use crate::sym::Analyzer;
use crate::target::Target;
use crate::tirx::TensorTypeObj;
use crate::transform::{Pass, PassContext};

const FOR_DEFAULT: i32 = 0;
const FOR_VECTORIZED: i32 = 2;
const AUTO_UNROLL_MAX_STEP: &str = "auto_unroll_max_step";
const UNROLL_EXPLICIT: &str = "unroll_explicit";
const UINT: u8 = DLDataTypeCode::kDLUInt as u8;
const FLOAT4_E2M1FN: u8 = DLDataTypeCode::kDLFloat4_e2m1fn as u8;

// ---------------------------------------------------------------------------------------------
// Types and shared helpers

/// The error of a failed C++ `TVM_FFI_ICHECK`.
fn check_failed(condition: &str, message: &str) -> Error {
    internal(&format!("Check failed: ({condition}) is false: {message}"))
}

/// The lanes as C++ `PrimType` stores them: a scalable vector stores minus its vscale factor.
fn encoded_lanes(dtype: DLDataType) -> i16 {
    dtype.lanes as i16
}

fn is_scalable(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) < -1
}

fn is_fixed_vector(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) > 1
}

fn is_vector(dtype: DLDataType) -> bool {
    is_scalable(dtype) || is_fixed_vector(dtype)
}

fn is_scalar(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) == 1
}

/// C++ `PrimType::lanes()`, which a scalable vector has none of at compile time.
fn lanes(dtype: DLDataType) -> Result<i32> {
    match encoded_lanes(dtype) {
        lanes if lanes < 0 => {
            Err(internal("Can't fetch the lanes of a scalable vector at a compile time."))
        }
        lanes => Ok(lanes as i32),
    }
}

/// `GetLanesOrVScaleFactor`.
fn lanes_or_factor(dtype: DLDataType) -> Result<i32> {
    match is_scalable(dtype) {
        true => Ok(-(encoded_lanes(dtype) as i32)),
        false => lanes(dtype),
    }
}

/// `PrimType(code, bits, lanes)`, or the scalable vector of that vscale factor.
fn with_lanes(dtype: DLDataType, lanes: i32, scalable: bool) -> DLDataType {
    let encoded = if scalable { -lanes } else { lanes };
    DLDataType { lanes: encoded as i16 as u16, ..dtype }
}

thread_local! {
    static OPS: RefCell<HashMap<&'static str, Op>> = RefCell::new(HashMap::new());
    static GLOBALS: RefCell<HashMap<&'static str, FfiFunction>> = RefCell::new(HashMap::new());
}

/// `Op::Get(name)`, fetched once per thread as C++ keeps it in a `static`.
fn op(name: &'static str) -> Result<Op> {
    if let Some(op) = OPS.with(|ops| ops.borrow().get(name).cloned()) {
        return Ok(op);
    }
    let op = Op::get(name)?;
    OPS.with(|ops| ops.borrow_mut().insert(name, op.clone()));
    Ok(op)
}

/// A global function, fetched once per thread.
fn global(name: &'static str) -> Result<FfiFunction> {
    if let Some(function) = GLOBALS.with(|globals| globals.borrow().get(name).cloned()) {
        return Ok(function);
    }
    let function = FfiFunction::get_global(name)?;
    GLOBALS.with(|globals| globals.borrow_mut().insert(name, function.clone()));
    Ok(function)
}

/// The raw C++ constructor `prim::<Node>(a, b, loc)`, through its global `prim.<Node>`.
fn raw_binary(name: &'static str, a: Expr, b: Expr, loc: &crate::ir::Location) -> Result<Expr> {
    global(name)?.call_tuple((a, b, loc.clone()))?.try_into()
}

/// `Call(ty, op, args, attrs, ty_args, loc)`.
fn call(
    ty: impl Into<crate::ir::Type>,
    op: Expr,
    args: Vec<Expr>,
    attrs: Option<crate::ir::Attrs>,
    ty_args: Array<crate::ir::Type>,
    loc: crate::ir::Location,
) -> Result<Expr> {
    let args = Array::new(args);
    Ok(Call::new_full(Some(ty.into()), op, args, attrs, ty_args, loc)?.into())
}

/// `ffi::StructuralEqual()(lhs, rhs)`.
fn structural_equal(lhs: &Expr, rhs: &Expr) -> Result<bool> {
    global("ffi.StructuralEqual")?
        .call_packed(&[
            AnyView::from(lhs),
            AnyView::from(rhs),
            AnyView::from(&false),
            AnyView::from(&false),
        ])?
        .try_into()
}

/// `IsConstInt(x, value)`.
fn is_const_int(x: &Expr, value: i64) -> bool {
    x.as_node::<IntImmObj>().is_some_and(|imm| imm.value == BigInt::from(value))
}

fn int_value(x: &Expr) -> Result<i64> {
    let imm = x.as_node::<IntImmObj>().ok_or_else(|| internal("expected an IntImm"))?;
    imm.value.to_i64().ok_or_else(|| internal("IntImm out of range"))
}

/// `CreateNewLanes`: `vscale() * n` for a scalable vector, `n` otherwise.
fn create_new_lanes(scalable: bool, lanes: i32) -> Result<Expr> {
    if scalable {
        let vscale = call(
            prim_type(int32_dtype())?,
            op("prim.vscale")?.into(),
            vec![],
            None,
            Array::new(vec![]),
            unknown_loc()?,
        )?;
        return Ok(Mul::new(vscale, lanes)?.into());
    }
    Ok(int32(lanes as i64)?.into())
}

fn int32_dtype() -> DLDataType {
    crate::ops::int32_dtype()
}

/// `BroadcastTo(e, lanes, is_scalable)`.
fn broadcast_to(e: Expr, lanes: i32, scalable: bool) -> Result<Expr> {
    let dtype = prim_dtype(&e)?;
    if lanes_or_factor(dtype)? == lanes && is_scalable(dtype) == scalable {
        return Ok(e);
    }
    if let Some(broadcast) = e.as_node::<BroadcastObj>() {
        let broadcast_dtype = prim_dtype(&e)?;
        if is_scalable(broadcast_dtype) != scalable {
            return Err(check_failed(
                "op->ty.as_or_throw<PrimType>().IsScalableVector() == is_scalable",
                "Can't broadcast between scalable and fixed length vectors.",
            ));
        }
        let e_lanes = lanes_or_factor(broadcast_dtype)?;
        if lanes % e_lanes == 0 {
            return Ok(Broadcast::new(
                broadcast.value.clone(),
                create_new_lanes(scalable, lanes)?,
            )?
            .into());
        }
    }
    if !is_scalar(dtype) {
        let message = format!(
            "Cannot broadcast lanes={} is_scalable={} to {lanes}",
            lanes_or_factor(dtype)?,
            is_scalable(dtype) as i32
        );
        return Err(check_failed("e.ty().IsScalar()", &message));
    }
    Ok(Broadcast::new(e, create_new_lanes(scalable, lanes)?)?.into())
}

/// `TargetHasRVV`: the target has the RISC-V vector extension.
fn target_has_rvv(target: &Option<Target>) -> Result<bool> {
    let Some(target) = target else { return Ok(false) };
    let Ok(has_feature) = FfiFunction::get_global("codegen.target_has_feature") else {
        return Ok(false);
    };
    has_feature.call_tuple((String::from("v"), target.clone()))?.try_into()
}

/// `TargetHasVLA`: AArch64 SVE or RISC-V V.
fn target_has_vla(target: &Option<Target>) -> Result<bool> {
    let Some(value) = target else { return Ok(false) };
    let attrs = value.attrs()?;
    let mut has_vla = match attrs.get(&String::from("feature.has_sve"))? {
        Some(flag) => bool_value(flag)?,
        None => false,
    };
    if !has_vla {
        if let Some(mattr) = attrs.get(&String::from("mattr"))? {
            let mattr: Array<String> = mattr.try_into()?;
            has_vla = mattr.iter().any(|attr| attr.as_str() == "+sve");
        }
    }
    Ok(has_vla || target_has_rvv(target)?)
}

/// C++ `TypeTraits<bool>`, which also accepts an integer.
fn bool_value(value: Any) -> Result<bool> {
    if let Some(flag) = value.try_as::<bool>() {
        return Ok(flag);
    }
    Ok(i64::try_from(value)? != 0)
}

/// `EnableTensorLevelPredication`: the pass config, else on for VLA targets.
fn enable_tensor_level_predication(context: &PassContext, target: &Option<Target>) -> Result<bool> {
    match context.config.get(&String::from("tirx.enable_tensor_level_predication"))? {
        Some(value) => bool_value(value),
        None => target_has_vla(target),
    }
}

/// `ContainsCallNode`: a primitive call anywhere in `stmt`.
fn contains_call(stmt: &Stmt) -> Result<bool> {
    let found = structural_walk(
        stmt,
        |call: Call| match call.ty.as_node::<PrimTypeObj>() {
            Some(_) => WalkResult::Interrupt,
            None => WalkResult::Advance,
        },
        WalkOrder::PreOrder,
    )?;
    Ok(found.is_some())
}

/// `IsVScaleCall` anywhere in `expr`.
fn contains_vscale(expr: &Expr) -> Result<bool> {
    let vscale = op("prim.vscale")?;
    let found = structural_walk(
        expr,
        |call: Call| match call.op.same_as(&vscale) && call.ty.as_node::<PrimTypeObj>().is_some() {
            true => WalkResult::Interrupt,
            false => WalkResult::Advance,
        },
        WalkOrder::PreOrder,
    )?;
    Ok(found.is_some())
}

/// `stmt` with every use of `var` replaced by `value`.
fn substitute(stmt: Stmt, var: &Var, value: &Expr) -> Result<Stmt> {
    structural_map(
        stmt,
        |candidate: Var, kind: DefRegionKind| -> UnchangedOr<Any> {
            if kind == DefRegionKind::None && candidate.same_as(var) {
                UnchangedOr::changed(Any::from(value.clone()))
            } else {
                UnchangedOr::unchanged()
            }
        },
        WalkOrder::PreOrder,
    )?
    .try_into()
}

/// `PrimVar(name, ty)`.
fn prim_var(name: &str, dtype: DLDataType) -> Result<Var> {
    Var::prim(name, dtype)
}

/// `Shuffle::ExtractElement(vector, index)`.
fn extract_element(vector: Expr, index: i64) -> Result<Expr> {
    shuffle(vec![vector], vec![int32(index)?.into()])
}

/// `Shuffle::Concat(vectors)`.
fn concat(vectors: Vec<Expr>) -> Result<Expr> {
    if vectors.len() == 1 {
        return Ok(vectors.into_iter().next().expect("one vector"));
    }
    let mut indices = Vec::new();
    let mut index = 0;
    for vector in &vectors {
        for _ in 0..lanes(prim_dtype(vector)?)? {
            indices.push(int32(index)?.into());
            index += 1;
        }
    }
    shuffle(vectors, indices)
}

fn shuffle(vectors: Vec<Expr>, indices: Vec<Expr>) -> Result<Expr> {
    global("prim.Shuffle")?
        .call_tuple((Array::new(vectors), Array::new(indices), unknown_loc()?))?
        .try_into()
}

// ---------------------------------------------------------------------------------------------
// TryPredicateTensorAccesses

/// Rewrites the tensor accesses of a statement under a block-level predicate `ramp < limit` into
/// masked loads and stores.
struct TryPredicateTensorAccesses {
    allow_offset_predication: bool,
    base: Option<Expr>,
    stride: Option<Expr>,
    lanes: Option<Expr>,
    limit: Option<Expr>,
    analyzed: usize,
    rewritten: usize,
}

#[dispatch(mutate)]
impl TryPredicateTensorAccesses {
    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let load: TensorLoad = m.default_mutate_as(self, &op)?;
        self.predicate_load(load)
    }

    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let store: TensorStore = m.default_mutate_as(self, &op)?;
        self.predicate_store(store)
    }

    fn mutate_call(&mut self, original: Call, m: &mut Mutator) -> Result<Any> {
        let call: Call = m.default_mutate_as(self, &original)?;
        let (masked_load, masked_store) = (op("tirx.masked_load")?, op("tirx.masked_store")?);
        if !call.op.same_as(&masked_load) && !call.op.same_as(&masked_store) {
            return Ok(call.into());
        }
        let is_load = call.op.same_as(&masked_load);
        let args: Vec<Expr> = call.args.iter().collect();
        let first = if is_load { 1 } else { 2 };
        let indices: Vec<Expr> = args.iter().take(args.len() - 1).skip(first).cloned().collect();
        let Some(mask) = self.lane_mask(&indices)? else { return Ok(call.into()) };
        let predicate = args.last().expect("a masked access has a predicate").clone();
        let predicate = match self.allow_offset_predication {
            true => (&predicate & mask).expr()?,
            false => mask,
        };
        let mut args = args;
        let last = args.len() - 1;
        args[last] = predicate;
        let call = Call::new_full(
            Some(call.ty.clone()),
            call.op.clone(),
            Array::new(args),
            call.attrs.clone(),
            call.ty_args.clone(),
            call.loc.clone(),
        )?;
        Ok(call.into())
    }
}

impl TryPredicateTensorAccesses {
    fn new(allow_offset_predication: bool) -> Self {
        Self {
            allow_offset_predication,
            base: None,
            stride: None,
            lanes: None,
            limit: None,
            analyzed: 0,
            rewritten: 0,
        }
    }

    /// `Run(stmt, condition)`: whether every access was predicated, and the rewritten statement.
    fn run(mut self, stmt: Stmt, condition: &Expr) -> Result<(bool, Stmt)> {
        let Some(lt) = condition.as_node::<LTObj>() else { return Ok((false, stmt)) };
        let (Some(ramp), Some(broadcast)) =
            (lt.a.as_node::<RampObj>(), lt.b.as_node::<BroadcastObj>())
        else {
            return Ok((false, stmt));
        };
        self.base = Some(ramp.base_.clone());
        self.stride = Some(ramp.stride.clone());
        self.lanes = Some(ramp.lanes.clone());
        self.limit = Some(broadcast.value.clone());
        let predicated: Stmt = structural_mutate(stmt.clone(), &mut self)?.try_into()?;
        if self.analyzed > 0 && self.analyzed == self.rewritten {
            return Ok((true, predicated));
        }
        Ok((false, stmt))
    }

    /// `GetLaneMask(indices)`.
    fn lane_mask(&mut self, indices: &[Expr]) -> Result<Option<Expr>> {
        self.analyzed += 1;
        let Some(ramp) = indices.first().and_then(|index| index.as_node::<RampObj>()) else {
            return Ok(None);
        };
        let (base, stride, lanes, limit) = (
            self.base.clone().expect("set by run"),
            self.stride.clone().expect("set by run"),
            self.lanes.clone().expect("set by run"),
            self.limit.clone().expect("set by run"),
        );
        if !structural_equal(&ramp.stride, &stride)? || !structural_equal(&ramp.lanes, &lanes)? {
            return Ok(None);
        }
        if !structural_equal(&ramp.base_, &base)? && !self.allow_offset_predication {
            return Ok(None);
        }
        let ramp_dtype = prim_dtype(&indices[0])?;
        let mask_dtype = with_lanes(
            DLDataType { code: UINT, bits: 1, lanes: 1 },
            lanes_or_factor(ramp_dtype)?,
            is_scalable(ramp_dtype),
        );
        let mask = call(
            prim_type(mask_dtype)?,
            op("tirx.get_active_lane_mask")?.into(),
            vec![base, limit],
            None,
            Array::new(vec![]),
            unknown_loc()?,
        )?;
        self.rewritten += 1;
        Ok(Some(mask))
    }

    fn predicate_load(&mut self, load: TensorLoad) -> Result<Any> {
        let indices: Vec<Expr> = load.indices.iter().collect();
        let Some(mask) = self.lane_mask(&indices)? else { return Ok(load.into()) };
        let mut args = vec![load.source.clone()];
        args.extend(indices);
        args.push(mask);
        let masked = call(
            load.ty.clone(),
            op("tirx.masked_load")?.into(),
            args,
            None,
            Array::new(vec![]),
            load.loc.clone(),
        )?;
        Ok(masked.into())
    }

    fn predicate_store(&mut self, store: TensorStore) -> Result<Any> {
        let indices: Vec<Expr> = store.indices.iter().collect();
        let Some(mask) = self.lane_mask(&indices)? else { return Ok(store.into()) };
        let mut args = vec![store.dest.clone(), store.value.clone()];
        args.extend(indices);
        args.push(mask);
        let void = prim_type(DLDataType::try_from_str("void")?)?;
        let masked = call(
            void,
            op("tirx.masked_store")?.into(),
            args,
            None,
            Array::new(vec![]),
            store.loc.clone(),
        )?;
        Ok(Evaluate::new_full(masked, store.loc.clone())?.into())
    }
}

// ---------------------------------------------------------------------------------------------
// Vectorizer

/// Rewrites the body of a vectorized loop over `var` with `var_lanes` lanes.
///
/// Every statement visit goes through [`Vectorizer::statement`], which scalarizes the statement
/// when an expression in it asked for that (C++ overrides `Mutate` for statements).
struct Vectorizer {
    var: Var,
    var_lanes: Expr,
    /// C++ `VarRemapGet` / `VarRemapSet`: the loop variable's ramp and the rebound variables.
    remap: HashMap<ObjectIdentity, Expr>,
    need_scalarize: bool,
    analyzer: Analyzer,
    vectorizable: OpAttrMap<bool>,
    target: Option<Target>,
    context: PassContext,
}

#[dispatch(mutate)]
impl Vectorizer {
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Any> {
        if let Some(mapped) = self.remap.get(&ObjectIdentity::of(&var)) {
            return Ok(mapped.clone().into());
        }
        m.default_mutate(self, &var)
    }

    fn mutate_tuple(&mut self, op: Tuple) -> Result<Any> {
        Err(no_default(&op))
    }

    fn mutate_tuple_get_item(&mut self, op: TupleGetItem) -> Result<Any> {
        Err(no_default(&op))
    }

    fn mutate_tensor_region(&mut self, op: TensorRegion) -> Result<Any> {
        Err(no_default(&op))
    }

    fn mutate_global_var(&mut self, op: GlobalVar) -> Result<Any> {
        Err(no_default(&op))
    }

    fn mutate_op(&mut self, op: Op) -> Result<Any> {
        Err(no_default(&op))
    }

    /// Only `OpaqueExpr` itself; C++ traverses a subclass structurally.
    fn mutate_opaque(&mut self, op: OpaqueExpr, m: &mut Mutator) -> Result<Any> {
        if AnyView::from(&op).type_index()
            == <crate::ir::OpaqueExprObj as tvm_ffi::ObjectCore>::type_index()
        {
            return Err(no_default(&op));
        }
        m.default_mutate(self, &op)
    }

    fn mutate_add(&mut self, op: crate::prim::Add, m: &mut Mutator) -> Result<Any> {
        self.add_sub(op.clone().into(), op.a.clone(), op.b.clone(), true, m)
    }

    fn mutate_sub(&mut self, op: crate::prim::Sub, m: &mut Mutator) -> Result<Any> {
        self.add_sub(op.clone().into(), op.a.clone(), op.b.clone(), false, m)
    }

    fn mutate_mul(&mut self, op: crate::prim::Mul, m: &mut Mutator) -> Result<Any> {
        let a: Expr = m.mutate_as(self, &op.a)?;
        let b: Expr = m.mutate_as(self, &op.b)?;
        if a.same_as(&op.a) && b.same_as(&op.b) {
            return Ok(op.into());
        }
        let (a_dtype, b_dtype) = (prim_dtype(&a)?, prim_dtype(&b)?);
        let (vector_a, vector_b) = (is_vector(a_dtype), is_vector(b_dtype));
        if vector_a && vector_b && is_scalable(a_dtype) != is_scalable(b_dtype) {
            return Err(check_failed(
                "a.ty().IsScalableVector() == b.ty().IsScalableVector()",
                "Fixed length and scalable vectors can't be mixed in multiplication.",
            ));
        }
        if vector_a || vector_b {
            if let Some(ramp) = a.as_node::<RampObj>() {
                if is_scalar(b_dtype) && self.can_prove(b.greater(0))? {
                    let base = (&ramp.base_ * &b).expr()?;
                    let stride = (&ramp.stride * &b).expr()?;
                    return Ok(Ramp::new(base, stride, ramp.lanes.clone())?.into());
                }
            }
            if let Some(ramp) = b.as_node::<RampObj>() {
                if is_scalar(a_dtype) && self.can_prove(a.greater(0))? {
                    let base = (&ramp.base_ * &a).expr()?;
                    let stride = (&ramp.stride * &a).expr()?;
                    return Ok(Ramp::new(base, stride, ramp.lanes.clone())?.into());
                }
            }
            let lanes = lanes_or_factor(a_dtype)?.max(lanes_or_factor(b_dtype)?);
            let scalable = is_scalable(a_dtype) || is_scalable(b_dtype);
            let (a, b) = (broadcast_to(a, lanes, scalable)?, broadcast_to(b, lanes, scalable)?);
            return Ok(Mul::new(a, b)?.into());
        }
        // C++ falls back to `BinaryVec<Mul>`, which visits both operands again.
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.Mul", m)
    }

    fn mutate_div(&mut self, op: crate::prim::Div, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.Div", m)
    }

    fn mutate_mod(&mut self, op: crate::prim::Mod, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.Mod", m)
    }

    fn mutate_floordiv(&mut self, op: crate::prim::FloorDiv, m: &mut Mutator) -> Result<Any> {
        let (a, b, loc) = (op.a.clone(), op.b.clone(), op.loc.clone());
        self.binary(op.into(), a, b, loc, "prim.FloorDiv", m)
    }

    fn mutate_floormod(&mut self, op: crate::prim::FloorMod, m: &mut Mutator) -> Result<Any> {
        let (a, b, loc) = (op.a.clone(), op.b.clone(), op.loc.clone());
        self.binary(op.into(), a, b, loc, "prim.FloorMod", m)
    }

    fn mutate_min(&mut self, op: crate::prim::Min, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.Min", m)
    }

    fn mutate_max(&mut self, op: crate::prim::Max, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.Max", m)
    }

    fn mutate_eq(&mut self, op: crate::prim::EQ, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.EQ", m)
    }

    fn mutate_ne(&mut self, op: crate::prim::NE, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.NE", m)
    }

    fn mutate_lt(&mut self, op: crate::prim::LT, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.LT", m)
    }

    fn mutate_le(&mut self, op: crate::prim::LE, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.LE", m)
    }

    fn mutate_gt(&mut self, op: crate::prim::GT, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.GT", m)
    }

    fn mutate_ge(&mut self, op: crate::prim::GE, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.GE", m)
    }

    fn mutate_and(&mut self, op: crate::prim::And, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.And", m)
    }

    fn mutate_or(&mut self, op: crate::prim::Or, m: &mut Mutator) -> Result<Any> {
        self.binary(op.clone().into(), op.a.clone(), op.b.clone(), op.loc.clone(), "prim.Or", m)
    }

    fn mutate_lshift(&mut self, op: crate::prim::LShift, m: &mut Mutator) -> Result<Any> {
        let (a, b, loc) = (op.a.clone(), op.b.clone(), op.loc.clone());
        self.binary(op.into(), a, b, loc, "prim.LShift", m)
    }

    fn mutate_rshift(&mut self, op: crate::prim::RShift, m: &mut Mutator) -> Result<Any> {
        let (a, b, loc) = (op.a.clone(), op.b.clone(), op.loc.clone());
        self.binary(op.into(), a, b, loc, "prim.RShift", m)
    }

    fn mutate_bitwise_and(&mut self, op: crate::prim::BitwiseAnd, m: &mut Mutator) -> Result<Any> {
        let (a, b, loc) = (op.a.clone(), op.b.clone(), op.loc.clone());
        self.binary(op.into(), a, b, loc, "prim.BitwiseAnd", m)
    }

    fn mutate_bitwise_or(&mut self, op: crate::prim::BitwiseOr, m: &mut Mutator) -> Result<Any> {
        let (a, b, loc) = (op.a.clone(), op.b.clone(), op.loc.clone());
        self.binary(op.into(), a, b, loc, "prim.BitwiseOr", m)
    }

    fn mutate_bitwise_xor(&mut self, op: crate::prim::BitwiseXor, m: &mut Mutator) -> Result<Any> {
        let (a, b, loc) = (op.a.clone(), op.b.clone(), op.loc.clone());
        self.binary(op.into(), a, b, loc, "prim.BitwiseXor", m)
    }

    fn mutate_bitwise_not(&mut self, op: crate::prim::BitwiseNot, m: &mut Mutator) -> Result<Any> {
        let a: Expr = m.mutate_as(self, &op.a)?;
        if a.same_as(&op.a) {
            return Ok(op.into());
        }
        let not: Expr = global("prim.BitwiseNot")?.call_tuple((a, op.loc.clone()))?.try_into()?;
        Ok(not.into())
    }

    fn mutate_not(&mut self, op: crate::prim::Not, m: &mut Mutator) -> Result<Any> {
        let a: Expr = m.mutate_as(self, &op.a)?;
        if a.same_as(&op.a) {
            return Ok(op.into());
        }
        Ok((!Ex::from(a)).expr()?.into())
    }

    /// A ramp over a vector base becomes a longer ramp or a concatenation of ramps; C++ rebuilds
    /// every ramp, so a scalar one comes back with its operands wrapped in `Shuffle`s.
    fn mutate_ramp(&mut self, op: crate::prim::Ramp, m: &mut Mutator) -> Result<Any> {
        let base: Expr = m.mutate_as(self, &op.base_)?;
        let stride: Expr = m.mutate_as(self, &op.stride)?;
        let (base_dtype, stride_dtype) = (prim_dtype(&base)?, prim_dtype(&stride)?);
        if is_scalable(base_dtype) {
            return Err(check_failed(
                "!base.ty().IsScalableVector()",
                "Creating scalable vectors from existing vectors is not supported.",
            ));
        }
        if is_scalable(stride_dtype) {
            return Err(check_failed(
                "!stride.ty().IsScalableVector()",
                "Ramp stride with scalable dtype is not supported",
            ));
        }
        if is_fixed_vector(base_dtype) && is_scalar(stride_dtype) {
            if op.lanes.as_node::<IntImmObj>().is_none() {
                return Err(check_failed(
                    "op->lanes->IsInstance<IntImmNode>()",
                    "Vectorizing over existing scalable vectors is not supported.",
                ));
            }
            let base_ramp = base
                .as_node::<RampObj>()
                .ok_or_else(|| internal("a vector ramp base must be a Ramp"))?;
            let op_lanes = int_value(&op.lanes)?;
            let base_ramp_lanes = int_value(&base_ramp.lanes)?;
            let step = make_const(stride_dtype, base_ramp_lanes)?;
            let same = base_ramp.stride.equal(&stride * step).expr()?;
            if self.analyzer.can_prove(&same, 0)? {
                let lanes = (op_lanes * base_ramp_lanes) as i32;
                return Ok(Ramp::new(base_ramp.base_.clone(), stride, lanes)?.into());
            }
        }
        let lanes = lanes(base_dtype)?.max(lanes(stride_dtype)?);
        let base = broadcast_to(base, lanes, false)?;
        let stride = broadcast_to(stride, lanes, false)?;
        let mut elems = Vec::with_capacity(lanes as usize);
        for i in 0..lanes as i64 {
            let (b, s) = (extract_element(base.clone(), i)?, extract_element(stride.clone(), i)?);
            elems.push(Ramp::new(b, s, op.lanes.clone())?.into());
        }
        Ok(concat(elems)?.into())
    }

    /// A broadcast of a value that became a vector cannot be widened; C++ rebuilds a changed one
    /// from its original value.
    fn mutate_broadcast(&mut self, op: crate::prim::Broadcast, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        if is_vector(prim_dtype(&value)?) {
            self.need_scalarize = true;
            return Ok(op.into());
        }
        if value.same_as(&op.value) {
            return Ok(op.into());
        }
        Ok(Broadcast::new(op.value.clone(), op.lanes.clone())?.into())
    }

    fn mutate_select(&mut self, op: crate::prim::Select, m: &mut Mutator) -> Result<Any> {
        let cond: Expr = m.mutate_as(self, &op.condition)?;
        let t: Expr = m.mutate_as(self, &op.true_value)?;
        let f: Expr = m.mutate_as(self, &op.false_value)?;
        if cond.same_as(&op.condition) && t.same_as(&op.true_value) && f.same_as(&op.false_value) {
            return Ok(op.into());
        }
        let dtypes = [prim_dtype(&cond)?, prim_dtype(&t)?, prim_dtype(&f)?];
        let mut lanes = 0;
        for dtype in dtypes {
            lanes = lanes.max(lanes_or_factor(dtype)?);
        }
        let scalable = dtypes.into_iter().any(is_scalable);
        let cond = broadcast_to(cond, lanes, scalable)?;
        let t = broadcast_to(t, lanes, scalable)?;
        let f = broadcast_to(f, lanes, scalable)?;
        Ok(Select::new(cond, t, f)?.into())
    }

    fn mutate_cast(&mut self, op: crate::prim::Cast, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        if value.same_as(&op.value) {
            return Ok(op.into());
        }
        let (value_dtype, target) = (prim_dtype(&value)?, prim_dtype(&Expr::from(op.clone()))?);
        let ty = match is_scalable(value_dtype) {
            true => with_lanes(target, lanes_or_factor(value_dtype)?, true),
            false => with_lanes(target, lanes(value_dtype)?, false),
        };
        Ok(Cast::new(value, prim_type(ty)?)?.into())
    }

    fn mutate_call(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        let Some(ret) = op.ty.as_node::<PrimTypeObj>().map(|ty| ty.dtype) else {
            // Non-primitive calls are not vectorized; a vector operand scalarizes the statement.
            let mut args = Vec::with_capacity(op.args.len());
            for arg in op.args.iter() {
                let new_arg: Expr = m.mutate_as(self, &arg)?;
                if is_prim_vector(&new_arg) {
                    self.need_scalarize = true;
                    return Ok(op.into());
                }
                args.push(new_arg);
            }
            // C++ compares the rebuilt array with the original, so it always rebuilds.
            let call = call(
                op.ty.clone(),
                op.op.clone(),
                args,
                op.attrs.clone(),
                op.ty_args.clone(),
                op.loc.clone(),
            )?;
            return Ok(call.into());
        };
        if is_op(&op.op, &op_lookup("prim.if_then_else")?) {
            return self.if_then_else(op, m);
        }
        if is_op(&op.op, &op_lookup("tirx.opencl.texture2d_load")?) {
            return self.texture_load(op, ret, m);
        }
        if is_op(&op.op, &op_lookup("tirx.opencl.texture2d_store")?) {
            return self.texture_store(op, m);
        }
        if is_op(&op.op, &op_lookup("tirx.reinterpret")?) {
            return self.reinterpret(op, m);
        }
        let vectorizable = match op.op.as_node::<crate::ir::OpObj>() {
            Some(callee) => self.vectorizable.get(callee)?.unwrap_or(false) && !is_scalable(ret),
            None => false,
        };
        let ret_ty: PrimType = op.ty.clone().try_cast()?;
        if !vectorizable {
            let mut args = Vec::with_capacity(op.args.len());
            for arg in op.args.iter() {
                let new_arg: Expr = m.mutate_as(self, &arg)?;
                if is_prim_vector(&new_arg) {
                    self.need_scalarize = true;
                    return Ok(op.into());
                }
                args.push(new_arg);
            }
            // As above, C++ always rebuilds, dropping the type arguments.
            let call = call(
                ret_ty,
                op.op.clone(),
                args,
                op.attrs.clone(),
                Array::new(vec![]),
                op.loc.clone(),
            )?;
            return Ok(call.into());
        }
        let mut lanes = 0;
        let args: Vec<Expr> = op.args.iter().collect();
        let new_args = if is_op(&op.op, &op_lookup("tirx.call_llvm_pure_intrin")?) {
            let (updated, _) = self.mutate_call_args(&args[1..], &mut lanes, m)?;
            std::iter::once(args[0].clone()).chain(updated).collect()
        } else {
            let (updated, changed) = self.mutate_call_args(&args, &mut lanes, m)?;
            if !changed {
                return Ok(op.into());
            }
            updated
        };
        let ty = prim_type(with_lanes(ret, lanes, false))?;
        let call = call(
            ty,
            op.op.clone(),
            new_args,
            op.attrs.clone(),
            Array::new(vec![]),
            op.loc.clone(),
        )?;
        Ok(call.into())
    }

    fn mutate_load(&mut self, op: TensorLoad, m: &mut Mutator) -> Result<Any> {
        let (indices, changed) = self.mutate_each(&op.indices, m)?;
        if !changed {
            return Ok(op.into());
        }
        Ok(TensorLoad::new_full(op.source.clone(), indices, op.loc.clone())?.into())
    }

    fn mutate_let(&mut self, op: Let, m: &mut Mutator) -> Result<Any> {
        let value: Expr = m.mutate_as(self, &op.value)?;
        let key = ObjectIdentity::of(&op.var);
        if let Some(mapped) = self.remap.get(&key) {
            if !deep_equal(mapped, &value)? {
                return Err(check_failed(
                    "deep_equal_(mapped.as_or_throw<PrimExpr>(), value)",
                    "Let cannot bind the same var to two different values",
                ));
            }
        }
        if lanes_or_factor(prim_dtype(&value)?)? != lanes_or_factor(prim_dtype(&op.value)?)? {
            let new_var = new_var(op.var.name.as_str(), value.ty.clone())?;
            self.remap.insert(key, new_var.clone().into());
            let body: Expr = m.mutate_as(self, &op.body)?;
            return Ok(Let::new(new_var, value, body)?.into());
        }
        self.remap.insert(key, op.var.clone().into());
        let body: Expr = m.mutate_as(self, &op.body)?;
        if value.same_as(&op.value) && body.same_as(&op.body) {
            return Ok(op.into());
        }
        Ok(Let::new(op.var.clone(), value, body)?.into())
    }

    fn mutate_shuffle(&mut self, op: Shuffle, m: &mut Mutator) -> Result<Any> {
        if op.vectors.len() != 1 || op.indices.len() != 1 {
            let message = format!(
                "Cannot vectorize ShuffleNode with multiple vectors or indices: the vector size is \
                 {} and the index size is {}",
                op.vectors.len(),
                op.indices.len()
            );
            return Err(check_failed(
                "op->vectors.size() == 1 && op->indices.size() == 1",
                &message,
            ));
        }
        let vectors: Vec<Expr> = op.vectors.iter().collect();
        let indices: Vec<Expr> = op.indices.iter().collect();
        let (_, vectors_changed) = self.mutate_array(&vectors, &mut 0, m)?;
        let (new_indices, indices_changed) = self.mutate_array(&indices, &mut 0, m)?;
        if !vectors_changed && !indices_changed {
            return Ok(op.into());
        }
        let var_lanes = int_value(&self.var_lanes)?;
        let vector_lanes = lanes(prim_dtype(&vectors[0])?)? as i64;
        let new_vec_length = var_lanes / vector_lanes;
        if !shuffle_index_matches(&new_indices[0], var_lanes, vector_lanes) {
            return Err(check_failed("f_check_index(updated_index)", ""));
        }
        if new_vec_length == 1 {
            let zero: Expr = int_imm(prim_dtype(&Expr::from(self.var.clone()))?, 0)?.into();
            let var = self.var.clone();
            let substituted = structural_map(
                vectors[0].clone(),
                move |candidate: Var| -> UnchangedOr<Any> {
                    match candidate.same_as(&var) {
                        true => UnchangedOr::changed(Any::from(zero.clone())),
                        false => UnchangedOr::unchanged(),
                    }
                },
                WalkOrder::PreOrder,
            )?;
            return Ok(substituted);
        }
        let previous = self.var_lanes.clone();
        self.var_lanes = int_imm(prim_dtype(&previous)?, new_vec_length as i128)?.into();
        let result = self.mutate_array(&vectors, &mut 0, m);
        self.var_lanes = previous;
        let (vectors, _) = result?;
        Ok(vectors[0].clone().into())
    }

    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        let original: Stmt = op.clone().into();
        self.statement(original, |this| this.vectorize_for(op, m))
    }

    fn mutate_if(&mut self, op: If, m: &mut Mutator) -> Result<Any> {
        let original: Stmt = op.clone().into();
        self.statement(original, |this| this.vectorize_if(op, m))
    }

    fn mutate_while(&mut self, op: While) -> Result<Any> {
        let original: Stmt = op.into();
        self.statement(original, |_| {
            Err(internal("A while loop inside a vectorized loop not supported."))
        })
    }

    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        let original: Stmt = op.clone().into();
        self.statement(original, |this| this.vectorize_bind(op, m))
    }

    fn mutate_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let original: Stmt = op.clone().into();
        self.statement(original, |this| this.vectorize_store(op, m))
    }

    fn mutate_stmt(&mut self, op: Stmt, m: &mut Mutator) -> Result<Any> {
        let original = op.clone();
        self.statement(original, |this| m.default_mutate(this, &op))
    }
}

/// How C++ prints a node in an error message (`ffi.ReprPrint`).
fn repr<T>(value: &T) -> std::string::String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    tvm_ffi::cached_global_func!("ffi.ReprPrint")
        .call_packed(&[AnyView::from(value)])
        .and_then(String::try_from)
        .map(|text| text.as_str().to_string())
        .unwrap_or_default()
}

/// The error a C++ `Vectorizer` raises for a node it has no rule for.
fn no_default<T>(node: &T) -> Error
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let index = AnyView::from(node).type_index();
    // SAFETY: every value's type index is registered.
    let key =
        unsafe { (*tvm_ffi::tvm_ffi_sys::TVMFFIGetTypeInfo(index)).type_key.as_str().to_string() };
    internal(&format!("Do not have a default for {key}"))
}

/// An `Op` that may not be registered in this build (the OpenCL texture ops); `None` matches
/// no callee.
fn op_lookup(name: &'static str) -> Result<Option<Op>> {
    Ok(op(name).ok())
}

/// Whether `callee` is the operator `op`.
fn is_op(callee: &Expr, op: &Option<Op>) -> bool {
    op.as_ref().is_some_and(|op| callee.same_as(op))
}

fn is_prim_vector(expr: &Expr) -> bool {
    expr.ty.as_node::<PrimTypeObj>().is_some_and(|ty| is_vector(ty.dtype))
}

/// `ExprDeepEqual()(lhs, rhs)`.
fn deep_equal(lhs: &Expr, rhs: &Expr) -> Result<bool> {
    global("prim.expr_deep_equal")?.call_tuple((lhs.clone(), rhs.clone()))?.try_into()
}

/// `Var(name, ty)`.
fn new_var(name: &str, ty: crate::ir::Type) -> Result<Var> {
    global("ir.Var")?.call_tuple((String::from(name), ty, unknown_loc()?))?.try_into()
}

/// The index patterns C++ accepts for a vectorized `Shuffle`: `Ramp(0, 1, lanes)`, or
/// `FloorMod(Ramp(0, 1, lanes), Broadcast(vector_lanes, lanes))`.
fn shuffle_index_matches(index: &Expr, var_lanes: i64, vector_lanes: i64) -> bool {
    let unit_ramp = |expr: &Expr| {
        expr.as_node::<RampObj>().is_some_and(|ramp| {
            is_const_int(&ramp.base_, 0)
                && is_const_int(&ramp.stride, 1)
                && is_const_int(&ramp.lanes, var_lanes)
        })
    };
    if unit_ramp(index) {
        return true;
    }
    index.as_node::<FloorModObj>().is_some_and(|floormod| {
        unit_ramp(&floormod.a)
            && floormod.b.as_node::<BroadcastObj>().is_some_and(|broadcast| {
                is_const_int(&broadcast.value, vector_lanes)
                    && is_const_int(&broadcast.lanes, var_lanes)
            })
    })
}

impl Vectorizer {
    fn new(
        var: Var,
        var_lanes: Expr,
        target: Option<Target>,
        context: PassContext,
    ) -> Result<Self> {
        let vectorizable = Op::attr_map::<bool>("TVectorizable")?
            .ok_or_else(|| internal("Attribute 'TVectorizable' is not registered"))?;
        let dtype = prim_dtype(&Expr::from(var.clone()))?;
        let ramp: Expr =
            Ramp::new(int_imm(dtype, 0)?, int_imm(dtype, 1)?, var_lanes.clone())?.into();
        let mut remap = HashMap::new();
        remap.insert(ObjectIdentity::of(&var), ramp);
        Ok(Self {
            var,
            var_lanes,
            remap,
            need_scalarize: false,
            analyzer: Analyzer::new()?,
            vectorizable,
            target,
            context,
        })
    }

    /// C++ `Vectorizer::Mutate` for a statement: rewrite it, and scalarize it instead if an
    /// expression in it asked for that.
    fn statement(
        &mut self,
        original: Stmt,
        rewrite: impl FnOnce(&mut Self) -> Result<Any>,
    ) -> Result<Any> {
        if self.need_scalarize {
            return Err(check_failed("!need_scalarize_", ""));
        }
        let result = rewrite(self)?;
        if self.need_scalarize {
            self.need_scalarize = false;
            return Ok(self.scalarize(original)?.into());
        }
        Ok(result)
    }

    /// `Scalarize(stmt)`: a serial loop over the lanes, the vectorized variable renamed.
    fn scalarize(&self, stmt: Stmt) -> Result<Stmt> {
        let dtype = prim_dtype(&Expr::from(self.var.clone()))?;
        let index = prim_var(&format!("{}.s", self.var.name.as_str()), dtype)?;
        let stmt = substitute(stmt, &self.var, &index.clone().into())?;
        Ok(For::new(index, int_imm(dtype, 0)?, self.var_lanes.clone(), FOR_DEFAULT, stmt)?.into())
    }

    fn can_prove(&self, condition: Ex) -> Result<bool> {
        self.analyzer.can_prove(&condition.expr()?, 0)
    }

    /// `AddSubVec`: a scalar added to a ramp moves into the ramp's base.
    fn add_sub(
        &mut self,
        original: Expr,
        a: Expr,
        b: Expr,
        add: bool,
        m: &mut Mutator,
    ) -> Result<Any> {
        let new_a: Expr = m.mutate_as(self, &a)?;
        let new_b: Expr = m.mutate_as(self, &b)?;
        if new_a.same_as(&a) && new_b.same_as(&b) {
            return Ok(original.into());
        }
        let compute = |x: Expr, y: Expr| -> Result<Expr> {
            match add {
                true => (Ex::from(x) + y).expr(),
                false => (Ex::from(x) - y).expr(),
            }
        };
        let (a_dtype, b_dtype) = (prim_dtype(&new_a)?, prim_dtype(&new_b)?);
        let lanes = lanes_or_factor(a_dtype)?.max(lanes_or_factor(b_dtype)?);
        if lanes != 1 {
            if let Some(ramp) = new_b.as_node::<RampObj>().filter(|_| is_scalar(a_dtype)) {
                let zero: Expr = int_imm(prim_dtype(&ramp.stride)?, 0)?.into();
                let base = compute(new_a.clone(), ramp.base_.clone())?;
                let stride = compute(zero, ramp.stride.clone())?;
                return Ok(Ramp::new(base, stride, ramp.lanes.clone())?.into());
            }
            if let Some(ramp) = new_a.as_node::<RampObj>().filter(|_| is_scalar(b_dtype)) {
                let base = compute(ramp.base_.clone(), new_b.clone())?;
                return Ok(Ramp::new(base, ramp.stride.clone(), ramp.lanes.clone())?.into());
            }
        }
        let scalable = is_scalable(a_dtype) || is_scalable(b_dtype);
        let (a, b) = (broadcast_to(new_a, lanes, scalable)?, broadcast_to(new_b, lanes, scalable)?);
        Ok(compute(a, b)?.into())
    }

    /// `BinaryVec<Node>`: both operands broadcast to the wider lanes, rebuilt by the raw
    /// constructor `global`.
    fn binary(
        &mut self,
        original: Expr,
        a: Expr,
        b: Expr,
        loc: crate::ir::Location,
        global: &'static str,
        m: &mut Mutator,
    ) -> Result<Any> {
        let new_a: Expr = m.mutate_as(self, &a)?;
        let new_b: Expr = m.mutate_as(self, &b)?;
        if new_a.same_as(&a) && new_b.same_as(&b) {
            return Ok(original.into());
        }
        let (a_dtype, b_dtype) = (prim_dtype(&new_a)?, prim_dtype(&new_b)?);
        let lanes = lanes_or_factor(a_dtype)?.max(lanes_or_factor(b_dtype)?);
        let scalable = is_scalable(a_dtype) || is_scalable(b_dtype);
        let (a, b) = (broadcast_to(new_a, lanes, scalable)?, broadcast_to(new_b, lanes, scalable)?);
        Ok(raw_binary(global, a, b, &loc)?.into())
    }

    /// `indices.Map(mutate)`: the mutated elements, and whether any changed.
    fn mutate_each(&mut self, values: &Array<Expr>, m: &mut Mutator) -> Result<(Vec<Expr>, bool)> {
        let mut changed = false;
        let mut result = Vec::with_capacity(values.len());
        for value in values.iter() {
            let new_value: Expr = m.mutate_as(self, &value)?;
            changed |= !new_value.same_as(&value);
            result.push(new_value);
        }
        Ok((result, changed))
    }

    /// `MutateArray`: the elements mutated and broadcast to the most lanes among them.
    fn mutate_array(
        &mut self,
        values: &[Expr],
        lanes: &mut i32,
        m: &mut Mutator,
    ) -> Result<(Vec<Expr>, bool)> {
        if values.is_empty() {
            return Ok((Vec::new(), false));
        }
        let mut changed = false;
        let mut result = Vec::with_capacity(values.len());
        for value in values {
            let new_value: Expr = m.mutate_as(self, value)?;
            changed |= !new_value.same_as(value);
            *lanes = (*lanes).max(self::lanes(prim_dtype(&new_value)?)?);
            result.push(new_value);
        }
        for value in result.iter_mut() {
            if self::lanes(prim_dtype(value)?)? != *lanes {
                *value = broadcast_to(value.clone(), *lanes, false)?;
                changed = true;
            }
        }
        Ok((result, changed))
    }

    /// `MutateCallArgs`: like [`Vectorizer::mutate_array`], leaving non-primitive operands as
    /// they are.
    fn mutate_call_args(
        &mut self,
        values: &[Expr],
        lanes: &mut i32,
        m: &mut Mutator,
    ) -> Result<(Vec<Expr>, bool)> {
        if values.is_empty() {
            return Ok((Vec::new(), false));
        }
        let mut changed = false;
        let mut result = Vec::with_capacity(values.len());
        for value in values {
            let new_value: Expr = m.mutate_as(self, value)?;
            changed |= !new_value.same_as(value);
            if let Some(ty) = new_value.ty.as_node::<PrimTypeObj>() {
                *lanes = (*lanes).max(self::lanes(ty.dtype)?);
            }
            result.push(new_value);
        }
        for value in result.iter_mut() {
            let Some(dtype) = value.ty.as_node::<PrimTypeObj>().map(|ty| ty.dtype) else {
                continue;
            };
            if self::lanes(dtype)? != *lanes {
                *value = broadcast_to(value.clone(), *lanes, false)?;
                changed = true;
            }
        }
        Ok((result, changed))
    }

    fn if_then_else(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        let args: Vec<Expr> = op.args.iter().collect();
        let cond: Expr = m.mutate_as(self, &args[0])?;
        if is_vector(prim_dtype(&cond)?) {
            self.need_scalarize = true;
            return Ok(op.into());
        }
        let t: Expr = m.mutate_as(self, &args[1])?;
        let f: Expr = m.mutate_as(self, &args[2])?;
        if cond.same_as(&args[0]) && t.same_as(&args[1]) && f.same_as(&args[2]) {
            return Ok(op.into());
        }
        let (t_dtype, f_dtype) = (prim_dtype(&t)?, prim_dtype(&f)?);
        let lanes = lanes_or_factor(t_dtype)?.max(lanes_or_factor(f_dtype)?);
        let scalable = is_scalable(t_dtype) || is_scalable(f_dtype);
        let t = broadcast_to(t, lanes, scalable)?;
        let f = broadcast_to(f, lanes, scalable)?;
        let ret = prim_dtype(&Expr::from(op.clone()))?;
        let ty = prim_type(with_lanes(ret, lanes, scalable))?;
        let call = call(
            ty,
            op.op.clone(),
            vec![cond, t, f],
            op.attrs.clone(),
            Array::new(vec![]),
            op.loc.clone(),
        )?;
        Ok(call.into())
    }

    fn reinterpret(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        let input = op.args.get(0)?;
        let value: Expr = m.mutate_as(self, &input)?;
        if value.same_as(&input) {
            return Ok(op.into());
        }
        let ret = prim_dtype(&Expr::from(op.clone()))?;
        let value_dtype = prim_dtype(&value)?;
        let ty = if is_scalable(value_dtype) {
            with_lanes(ret, lanes_or_factor(value_dtype)?, true)
        } else {
            let input_dtype = prim_dtype(&input)?;
            let new_lanes = if ret.code != FLOAT4_E2M1FN && input_dtype.code != FLOAT4_E2M1FN {
                (value_dtype.bits as i32 * lanes(value_dtype)?) / ret.bits as i32
            } else {
                lanes(value_dtype)?
            };
            with_lanes(ret, new_lanes, false)
        };
        let call = call(
            prim_type(ty)?,
            op.op.clone(),
            vec![value],
            op.attrs.clone(),
            Array::new(vec![]),
            op.loc.clone(),
        )?;
        Ok(call.into())
    }

    /// `GetTextureElementType`: the element type of a texture's pointer type.
    fn texture_bits(texture: &Expr) -> Result<i64> {
        let pointer = texture
            .ty
            .as_node::<crate::ir::PtrTypeObj>()
            .ok_or_else(|| check_failed("pointer_type", "Texture arguments must have PtrType"))?;
        let element: PrimType = pointer.element_type.clone().try_cast()?;
        Ok(element.dtype.bits as i64)
    }

    fn texture_load(&mut self, op: Call, ret: DLDataType, m: &mut Mutator) -> Result<Any> {
        let args: Vec<Expr> = op.args.iter().collect();
        let mut lane = 0;
        let (fcd, _) = self.mutate_array(&args[args.len() - 1..], &mut lane, m)?;
        let bits = Self::texture_bits(&args[0])?;
        if lane as i64 * bits > int_value(&args[4])? {
            return Err(check_failed(
                "lane * dtype.bits() <= op->args[4].as<IntImmNode>()->value",
                "Expected Data to be Read is lesser than or equal to Texture Load length",
            ));
        }
        let mut new_args = args[..args.len() - 1].to_vec();
        new_args.push(fcd[0].clone());
        let ty = prim_type(with_lanes(ret, lane, false))?;
        Ok(call(ty, op.op.clone(), new_args, op.attrs.clone(), Array::new(vec![]), op.loc.clone())?
            .into())
    }

    fn texture_store(&mut self, op: Call, m: &mut Mutator) -> Result<Any> {
        let mut args: Vec<Expr> = op.args.iter().collect();
        let mut lane = 0;
        let (value, _) = self.mutate_array(&args[args.len() - 1..], &mut lane, m)?;
        let bits = Self::texture_bits(&args[0])?;
        if lane as i64 * bits != int_value(&args[4])? {
            return Err(check_failed(
                "lane * dtype.bits() == op->args[4].as<IntImmNode>()->value",
                "Expected Data to be Written equal to Texture Store length",
            ));
        }
        let last = args.len() - 1;
        args[last] = value[0].clone();
        Ok(call(
            op.ty.clone(),
            op.op.clone(),
            args,
            op.attrs.clone(),
            Array::new(vec![]),
            op.loc.clone(),
        )?
        .into())
    }

    fn vectorize_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        // C++ also logs a warning for a vectorized loop nested in one; it keeps the loop.
        if !is_const_int(&op.min, 0) {
            return Err(check_failed("IsZero(op->min)", ""));
        }
        if is_vector(prim_dtype(&op.extent)?) {
            return Err(check_failed(
                "!op->extent.ty().IsScalableVector() && !op->extent.ty().IsFixedLengthVector()",
                "",
            ));
        }
        let extent: Expr = m.mutate_as(self, &op.extent)?;
        if is_vector(prim_dtype(&extent)?) {
            return Ok(self.scalarize(op.into())?.into());
        }
        let body = self.mutate_body(&op.body, m)?;
        if extent.same_as(&op.extent) && body.same_as(&op.body) {
            return Ok(op.into());
        }
        Ok(op
            .with(|node| {
                node.extent = extent;
                node.body = body;
            })
            .into())
    }

    /// C++ `Mutate(const SeqStmt&)`: a changed statement is wrapped into a sequence.
    fn mutate_body(&mut self, body: &SeqStmt, m: &mut Mutator) -> Result<SeqStmt> {
        let result: Stmt = m.mutate_as(self, &Stmt::from(body.clone()))?;
        if result.same_as(body) {
            return Ok(body.clone());
        }
        SeqStmt::from_stmt(result)
    }

    fn vectorize_if(&mut self, op: If, m: &mut Mutator) -> Result<Any> {
        if is_vector(prim_dtype(&op.condition)?) {
            return Err(check_failed(
                "!op->condition.ty().IsScalableVector() && !op->condition.ty().IsFixedLengthVector()",
                "",
            ));
        }
        let condition: Expr = m.mutate_as(self, &op.condition)?;
        let cond_need_scalarize = std::mem::replace(&mut self.need_scalarize, false);
        let then_case = self.mutate_body(&op.then_case, m)?;
        let else_case = match &op.else_case {
            Some(else_case) => Some(self.mutate_body(else_case, m)?),
            None => None,
        };
        let condition_is_vector = is_vector(prim_dtype(&condition)?);
        let predication = enable_tensor_level_predication(&self.context, &self.target)?;
        if predication && condition_is_vector && else_case.is_none() {
            let predicator = TryPredicateTensorAccesses::new(target_has_rvv(&self.target)?);
            let (success, stmt) = predicator.run(then_case.clone().into(), &condition)?;
            if success {
                return Ok(stmt.into());
            }
        }
        if cond_need_scalarize || condition_is_vector {
            return Ok(self.scalarize(op.into())?.into());
        }
        let else_same = match (&else_case, &op.else_case) {
            (Some(new), Some(old)) => new.same_as(old),
            (None, None) => true,
            _ => false,
        };
        if condition.same_as(&op.condition) && then_case.same_as(&op.then_case) && else_same {
            return Ok(op.into());
        }
        Ok(If::new(condition, then_case, else_case)?.into())
    }

    fn vectorize_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Any> {
        if op.value.ty.as_node::<PrimTypeObj>().is_none() {
            return m.default_mutate(self, &op);
        }
        let value: Expr = m.mutate_as(self, &op.value)?;
        if self.need_scalarize {
            self.need_scalarize = false;
            return Ok(self.scalarize(op.into())?.into());
        }
        let key = ObjectIdentity::of(&op.var);
        if self.remap.contains_key(&key) {
            return Err(check_failed(
                "VarRemapGet(op->var) == nullptr",
                "SSA violation, a single var is binded twice",
            ));
        }
        if lanes_or_factor(prim_dtype(&value)?)? != lanes_or_factor(prim_dtype(&op.value)?)? {
            let new_var = new_var(op.var.name.as_str(), value.ty.clone())?;
            self.remap.insert(key, new_var.clone().into());
            return Ok(Bind::new(new_var, value)?.into());
        }
        self.remap.insert(key, op.var.clone().into());
        if value.same_as(&op.value) {
            return Ok(op.into());
        }
        Ok(Bind::new_full(op.var.clone(), value, op.loc.clone())?.into())
    }

    fn vectorize_store(&mut self, op: TensorStore, m: &mut Mutator) -> Result<Any> {
        let (mut indices, indices_changed) = self.mutate_each(&op.indices, m)?;
        let value: Expr = m.mutate_as(self, &op.value)?;
        if !indices_changed && value.same_as(&op.value) {
            return Ok(op.into());
        }
        let tensor = op
            .dest
            .ty
            .as_node::<TensorTypeObj>()
            .ok_or_else(|| internal("a store's destination is a tensor"))?;
        let element = tensor.dtype.dtype;
        if is_scalable(element) {
            return Err(check_failed(
                "!op->dest.as_or_throw<TensorVar>().dtype().IsScalableVector()",
                "Vectorizing over scalable tensor elements is not supported in vectorizer.",
            ));
        }
        let mut other_index_lanes = lanes(element)?;
        for index in &indices[..indices.len() - 1] {
            let dtype = prim_dtype(index)?;
            other_index_lanes *= lanes(dtype)?;
            if is_scalable(dtype) {
                return Err(check_failed(
                    "!indices[i].ty().IsScalableVector()",
                    "Only the last index can be scalable.",
                ));
            }
        }
        let last = indices.len() - 1;
        let last_dtype = prim_dtype(&indices[last])?;
        let index_lanes = other_index_lanes * lanes_or_factor(last_dtype)?;
        let value_lanes = lanes_or_factor(prim_dtype(&value)?)?;
        let last_scalable = is_scalable(last_dtype);
        let total_lanes = index_lanes.max(value_lanes);
        if total_lanes % other_index_lanes != 0 {
            let name =
                op.dest.as_node::<crate::ir::VarObj>().map(|var| var.name.as_str().to_string());
            return Err(internal(&format!(
                "Check failed: total_lanes % other_index_lanes == 0 ({} vs. 0) : When storing to \
                 tensor {}, cannot produce {total_lanes} lanes of storage location by changing \
                 the last index.",
                total_lanes % other_index_lanes,
                name.unwrap_or_default()
            )));
        }
        let last_index_lanes = total_lanes / other_index_lanes;
        indices[last] = broadcast_to(indices[last].clone(), last_index_lanes, last_scalable)?;
        let value = broadcast_to(value, total_lanes, last_scalable)?;
        let store = TensorStore::new(op.loc.clone(), op.dest.clone(), Array::new(indices), value);
        Ok(store.into())
    }
}

// ---------------------------------------------------------------------------------------------
// The loop driver

/// Copies the unrolling policy of a vectorized loop onto the first loops in its body, which
/// survive vectorization (`PreserveUnrollPolicy`).
struct PolicyTransfer {
    policy: Vec<(String, Any)>,
}

#[dispatch(mutate)]
impl PolicyTransfer {
    fn mutate_expr(&mut self, _expr: Expr) -> tvm_ffi::Unchanged {
        tvm_ffi::Unchanged
    }

    /// A key the loop leaves unset or null takes the policy's value; the body is not entered.
    fn mutate_for(&mut self, op: For) -> Result<Any> {
        let mut entries: Vec<(String, Any)> = op.annotations.iter().collect();
        let mut changed = false;
        for (key, value) in &self.policy {
            match entries.iter_mut().find(|(existing, _)| existing.as_str() == key.as_str()) {
                Some(entry) if entry.1.type_index() == 0 => {
                    entry.1 = value.clone();
                    changed = true;
                }
                Some(_) => {}
                None => {
                    entries.push((key.clone(), value.clone()));
                    changed = true;
                }
            }
        }
        if !changed {
            return Ok(op.into());
        }
        Ok(op.with_annotations(entries.into_iter().collect::<Map<String, Any>>()).into())
    }
}

/// `PreserveUnrollPolicy(body, annotations)`.
fn preserve_unroll_policy(body: SeqStmt, annotations: &Map<String, Any>) -> Result<Stmt> {
    let mut policy = Vec::new();
    for key in [AUTO_UNROLL_MAX_STEP, UNROLL_EXPLICIT] {
        let key = String::from(key);
        if let Some(value) = annotations.get(&key)? {
            if value.type_index() != 0 {
                policy.push((key, value));
            }
        }
    }
    if policy.is_empty() {
        return Ok(body.into());
    }
    structural_mutate(body, &mut PolicyTransfer { policy })?.try_into()
}

/// Vectorizes each vectorized loop of a function body.
struct LoopVectorizer {
    target: Option<Target>,
    context: PassContext,
}

#[dispatch(mutate)]
impl LoopVectorizer {
    fn mutate_expr(&mut self, _expr: Expr) -> tvm_ffi::Unchanged {
        tvm_ffi::Unchanged
    }

    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        if op.kind != FOR_VECTORIZED {
            return m.default_mutate(self, &op);
        }
        if !is_const_int(&op.min, 0) {
            return Err(check_failed("IsZero(op->min)", ""));
        }
        let extent = op.extent.as_node::<IntImmObj>().map(|imm| imm.value.clone());
        if let Some(value) = &extent {
            if *value > BigInt::from(1i64)
                && target_has_rvv(&self.target)?
                && !contains_call(&op.body.clone().into())?
            {
                let value = value.to_i64().ok_or_else(|| internal("extent out of range"))?;
                return self.vectorize_for_rvv(op, value, m);
            }
        }
        if extent.as_ref().is_none_or(|value| *value < BigInt::from(1i64)) {
            let scalable = contains_vscale(&op.extent)?;
            if !(scalable && target_has_vla(&self.target)?) {
                let target = match &self.target {
                    Some(target) => repr(target),
                    None => repr(&Any::default()),
                };
                let message = format!(
                    "Failed to vectorize loop with extent {} for target {target}",
                    repr(&op.extent)
                );
                return Err(check_failed("is_scalable_expr && TargetHasVLA(target_)", &message));
            }
        }
        let body = preserve_unroll_policy(op.body.clone(), &op.annotations)?;
        let mut vectorizer = Vectorizer::new(
            op.loop_var.clone(),
            op.extent.clone(),
            self.target.clone(),
            self.context.clone(),
        )?;
        structural_mutate(body, &mut vectorizer)
    }
}

impl LoopVectorizer {
    /// `VectorizeFixedLoopForRVV`: a constant-extent loop as a serial loop over chunks of a
    /// scalable vector length, the tail guarded.
    fn vectorize_for_rvv(&mut self, op: For, extent: i64, m: &mut Mutator) -> Result<Any> {
        const DEFAULT_VSCALE_FACTOR: i32 = 4;
        let index_dtype = prim_dtype(&Expr::from(op.loop_var.clone()))?;
        let zero = int_imm(index_dtype, 0)?;
        let fixed_extent: Expr = int_imm(index_dtype, extent as i128)?.into();
        let scalable_lanes = create_new_lanes(true, DEFAULT_VSCALE_FACTOR)?;
        let lane_dtype = prim_dtype(&scalable_lanes)?;
        let lanes_index: Expr = match lane_dtype == index_dtype {
            true => scalable_lanes.clone(),
            false => Cast::new(scalable_lanes.clone(), prim_type(index_dtype)?)?.into(),
        };
        let num_chunks = fixed_extent.ceildiv(&lanes_index).expr()?;
        let name = op.loop_var.name.as_str();
        let outer = prim_var(&format!("{name}.vla.o"), index_dtype)?;
        let inner = prim_var(&format!("{name}.vla.i"), lane_dtype)?;
        let inner_index: Expr = match lane_dtype == index_dtype {
            true => inner.clone().into(),
            false => Cast::new(inner.clone(), prim_type(index_dtype)?)?.into(),
        };
        let index = (&outer * &lanes_index + inner_index).expr()?;
        let body = substitute(op.body.clone().into(), &op.loop_var, &index)?;
        let guarded = If::new_full(index.less(&fixed_extent), body, None, op.loc.clone())?;
        let vector_loop = For::new_full(
            inner,
            int_imm(lane_dtype, 0)?,
            scalable_lanes,
            FOR_VECTORIZED,
            guarded,
            op.annotations.clone(),
            None,
            op.loc.clone(),
        )?;
        let outer_loop = For::new_full(
            outer,
            zero,
            num_chunks,
            FOR_DEFAULT,
            vector_loop,
            Map::new(),
            None,
            op.loc.clone(),
        )?;
        m.mutate(self, &outer_loop)
    }
}

/// With vectorization off, a vectorized loop runs serially.
struct VectorizeSkipper;

#[dispatch(mutate)]
impl VectorizeSkipper {
    fn mutate_expr(&mut self, _expr: Expr) -> tvm_ffi::Unchanged {
        tvm_ffi::Unchanged
    }

    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        let stmt: For = m.default_mutate_as(self, &op)?;
        if stmt.kind == FOR_VECTORIZED {
            return Ok(stmt.with_kind(FOR_DEFAULT).into());
        }
        Ok(stmt.into())
    }
}

pub fn vectorize_loop(enable_vectorize: bool) -> Result<Pass> {
    function_pass("tirx.VectorizeLoop", 0, move |func, _module, context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let result: Stmt = if enable_vectorize {
            let target = get_attr(&func.attrs, "target")?.map(Target::try_from).transpose()?;
            let mut vectorizer = LoopVectorizer { target, context: context.clone() };
            structural_mutate(body.clone(), &mut vectorizer)?.try_into()?
        } else {
            structural_mutate(body.clone(), &mut VectorizeSkipper)?.try_into()?
        };
        if result.same_as(&body) {
            return Ok(Some(func));
        }
        Ok(Some(func.with_body(Some(SeqStmt::from_stmt(result)?))))
    })
}
