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

//! Rust ports of the C++ structural hooks of the IR node kinds the passes traverse most.
//!
//! The Rust structural engines reach a node's registered C++ `__s_visit__` / `__s_mutate__`
//! hook through the ABI vtable, and the hook reaches each child the same way back: two
//! foreign calls per child. The hooks here are registered with the engines' Rust hook
//! registry ([`tvm_ffi::register_structural_hooks`]) and run in the engine's own frame.
//!
//! Each function mirrors the C++ hook of the same node in `src/ir/expr.cc`, `src/ir/stmt.cc`,
//! `src/ir/type.cc`, `src/ir/op.cc`, `src/ir/prim/expr.cc`, `src/ir/prim/vector_expr.cc` and
//! `src/tirx/ir/type.cc`: the same children in the same
//! order, the same skipped fields, the same definition regions, and the same copy / in-place
//! protocol. One Rust function serves both the `XxxMutate` and the `XxxMaybeInplaceMutate` C++
//! hook: [`MutateChildren::is_inplace`] tells which one the engine would have chosen.

use std::sync::OnceLock;

use tvm_ffi::{
    register_structural_hooks, Any, AnyView, Array, DefRegionKind, Error, InplaceMode,
    MutateChildren, ObjectArc, ObjectCore, ObjectRefCore, Result, StructuralHooks, StructuralView,
    TypeIndex, UnchangedOr, VisitChildren, VisitHalt, TYPE_ERROR,
};

use crate::ir::{
    AnyTypeObj, AssertStmt, AssertStmtObj, Bind, BindObj, BreakObj, Call, CallObj, ContinueObj,
    DataTypeImm, DataTypeImmObj, DictAttrs, Evaluate, EvaluateObj, Expr, ExprObj, FloatImmObj, For,
    ForObj, FuncType, FuncTypeObj, GenericConst, GenericConstObj, GlobalVarObj, If, IfObj,
    IntImmObj, MissingTypeObj, OpObj, OpaqueExpr, OpaqueExprObj, OpaqueTypeObj, PrimTypeObj,
    PtrType, PtrTypeObj, Range, RangeObj, RegionStmt, RegionStmtObj, Return, ReturnObj, SeqStmt,
    SeqStmtObj, Stmt, StringImm, StringImmObj, StringTypeObj, TensorLoad, TensorLoadObj,
    TensorRegion, TensorRegionObj, TensorRegionTypeObj, TensorStore, TensorStoreObj, Tuple,
    TupleGetItem, TupleGetItemObj, TupleObj, TupleType, TupleTypeObj, Type, TypeObj, Var, VarObj,
    While, WhileObj,
};
use crate::prim::{
    Add, AddObj, And, AndObj, BitwiseAnd, BitwiseAndObj, BitwiseNot, BitwiseNotObj, BitwiseOr,
    BitwiseOrObj, BitwiseXor, BitwiseXorObj, Broadcast, BroadcastObj, Cast, CastObj, Div, DivObj,
    FloorDiv, FloorDivObj, FloorMod, FloorModObj, LShift, LShiftObj, Let, LetObj, Max, MaxObj, Min,
    MinObj, Mod, ModObj, Mul, MulObj, Not, NotObj, Or, OrObj, RShift, RShiftObj, Ramp, RampObj,
    Select, SelectObj, Shuffle, ShuffleObj, Sub, SubObj, EQ, EQObj, GE, GEObj, GT, GTObj, LE,
    LEObj, LT, LTObj, NE, NEObj,
};
use crate::tirx::{TensorMapTypeObj, TensorType, TensorTypeObj};

type VisitResult = std::result::Result<(), VisitHalt>;
type MutateResult = Result<UnchangedOr<Any>>;

/// Register the hooks once per process. Later calls do nothing.
pub fn register() -> Result<()> {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    if REGISTERED.get().is_some() {
        return Ok(());
    }
    let leaf = StructuralHooks { visit: Some(leaf_visit), mutate: Some(leaf_mutate) };
    // Leaves: constants, registry atoms and field-less types.
    register_structural_hooks::<IntImmObj>(leaf)?;
    register_structural_hooks::<FloatImmObj>(leaf)?;
    register_structural_hooks::<OpObj>(leaf)?;
    register_structural_hooks::<GlobalVarObj>(leaf)?;
    register_structural_hooks::<BreakObj>(leaf)?;
    register_structural_hooks::<ContinueObj>(leaf)?;
    register_structural_hooks::<PrimTypeObj>(leaf)?;
    register_structural_hooks::<TypeObj>(leaf)?;
    register_structural_hooks::<AnyTypeObj>(leaf)?;
    register_structural_hooks::<MissingTypeObj>(leaf)?;
    register_structural_hooks::<OpaqueTypeObj>(leaf)?;
    register_structural_hooks::<StringTypeObj>(leaf)?;
    register_structural_hooks::<TensorRegionTypeObj>(leaf)?;
    register_structural_hooks::<TensorMapTypeObj>(leaf)?;
    // Expressions.
    register_structural_hooks::<VarObj>(hooks(var_visit, var_mutate))?;
    register_structural_hooks::<CallObj>(hooks(call_visit, call_mutate))?;
    register_structural_hooks::<TensorLoadObj>(hooks(tensor_load_visit, tensor_load_mutate))?;
    register_structural_hooks::<TensorRegionObj>(hooks(tensor_region_visit, tensor_region_mutate))?;
    register_structural_hooks::<TupleObj>(hooks(tuple_visit, tuple_mutate))?;
    register_structural_hooks::<TupleGetItemObj>(hooks(
        tuple_get_item_visit,
        tuple_get_item_mutate,
    ))?;
    register_structural_hooks::<RangeObj>(hooks(range_visit, range_mutate))?;
    register_structural_hooks::<DataTypeImmObj>(hooks(data_type_imm_visit, data_type_imm_mutate))?;
    register_structural_hooks::<OpaqueExprObj>(hooks(ty_only_visit, opaque_expr_mutate))?;
    register_structural_hooks::<StringImmObj>(hooks(ty_only_visit, string_imm_mutate))?;
    register_structural_hooks::<GenericConstObj>(hooks(ty_only_visit, generic_const_mutate))?;
    // Statements.
    register_structural_hooks::<BindObj>(hooks(bind_visit, bind_mutate))?;
    register_structural_hooks::<EvaluateObj>(hooks(evaluate_visit, evaluate_mutate))?;
    register_structural_hooks::<AssertStmtObj>(hooks(assert_stmt_visit, assert_stmt_mutate))?;
    register_structural_hooks::<ForObj>(hooks(for_visit, for_mutate))?;
    register_structural_hooks::<WhileObj>(hooks(while_visit, while_mutate))?;
    register_structural_hooks::<IfObj>(hooks(if_visit, if_mutate))?;
    register_structural_hooks::<SeqStmtObj>(hooks(seq_stmt_visit, seq_stmt_mutate))?;
    register_structural_hooks::<RegionStmtObj>(hooks(region_stmt_visit, region_stmt_mutate))?;
    register_structural_hooks::<ReturnObj>(hooks(return_visit, return_mutate))?;
    register_structural_hooks::<TensorStoreObj>(hooks(tensor_store_visit, tensor_store_mutate))?;
    // Types.
    register_structural_hooks::<PtrTypeObj>(hooks(ptr_type_visit, ptr_type_mutate))?;
    register_structural_hooks::<FuncTypeObj>(hooks(func_type_visit, func_type_mutate))?;
    register_structural_hooks::<TupleTypeObj>(hooks(tuple_type_visit, tuple_type_mutate))?;
    register_structural_hooks::<TensorTypeObj>(hooks(tensor_type_visit, tensor_type_mutate))?;
    // Primitive expressions.
    register_binary_hooks::<AddObj>()?;
    register_binary_hooks::<SubObj>()?;
    register_binary_hooks::<MulObj>()?;
    register_binary_hooks::<DivObj>()?;
    register_binary_hooks::<ModObj>()?;
    register_binary_hooks::<FloorDivObj>()?;
    register_binary_hooks::<FloorModObj>()?;
    register_binary_hooks::<MinObj>()?;
    register_binary_hooks::<MaxObj>()?;
    register_binary_hooks::<EQObj>()?;
    register_binary_hooks::<NEObj>()?;
    register_binary_hooks::<LTObj>()?;
    register_binary_hooks::<LEObj>()?;
    register_binary_hooks::<GTObj>()?;
    register_binary_hooks::<GEObj>()?;
    register_binary_hooks::<AndObj>()?;
    register_binary_hooks::<OrObj>()?;
    register_binary_hooks::<BitwiseAndObj>()?;
    register_binary_hooks::<BitwiseOrObj>()?;
    register_binary_hooks::<BitwiseXorObj>()?;
    register_binary_hooks::<LShiftObj>()?;
    register_binary_hooks::<RShiftObj>()?;
    register_unary_hooks::<CastObj>()?;
    register_unary_hooks::<NotObj>()?;
    register_unary_hooks::<BitwiseNotObj>()?;
    register_structural_hooks::<SelectObj>(hooks(select_visit, select_mutate))?;
    register_structural_hooks::<LetObj>(hooks(let_visit, let_mutate))?;
    register_structural_hooks::<RampObj>(hooks(ramp_visit, ramp_mutate))?;
    register_structural_hooks::<BroadcastObj>(hooks(broadcast_visit, broadcast_mutate))?;
    register_structural_hooks::<ShuffleObj>(hooks(shuffle_visit, shuffle_mutate))?;
    let _ = REGISTERED.set(());
    Ok(())
}

fn hooks(
    visit: fn(&StructuralView, &mut VisitChildren<'_>) -> VisitResult,
    mutate: fn(&StructuralView, &mut MutateChildren<'_>) -> MutateResult,
) -> StructuralHooks {
    StructuralHooks { visit: Some(visit), mutate: Some(mutate) }
}

// ---- Shared helpers ------------------------------------------------------------------------

/// The hook's node. Hooks are registered for their own node type, so the check never fails.
#[inline]
fn node<N: ObjectCore>(value: &StructuralView) -> &N {
    value.as_node::<N>().expect("structural hook registered for its own node type")
}

/// A reference object from its freshly built node, as an `Any`.
#[inline]
fn wrap<R: ObjectRefCore + Into<Any>>(obj: R::ContainerType) -> Any {
    // SAFETY: `obj` is a complete node of exactly the type `R` refers to.
    unsafe { R::from_data(ObjectArc::new(obj)) }.into()
}

/// Whether `mapped` leaves the field holding `original` as it is.
#[inline]
fn same<O>(mapped: &UnchangedOr<Any>, original: &O) -> bool
where
    for<'x> AnyView<'x>: From<&'x O>,
{
    mapped.unchanged_or_same_as_value(original)
}

/// The replacement a mutation produced, typed, or `None` when it left the child unchanged.
/// A replacement of the wrong type is an error, as in the C++ `TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN`.
#[inline]
fn changed<T: TryFrom<Any, Error = Error>>(mapped: UnchangedOr<Any>) -> Result<Option<T>> {
    mapped.into_option().map(T::try_from).transpose()
}

/// The mutated field value: the typed replacement, or a copy of the original.
#[inline]
fn value_or<T: TryFrom<Any, Error = Error> + Clone>(
    mapped: UnchangedOr<Any>,
    original: &T,
) -> Result<T> {
    Ok(changed::<T>(mapped)?.unwrap_or_else(|| original.clone()))
}

/// [`changed`] for an optional field: `Some(None)` is a replacement by nothing.
#[inline]
fn changed_optional<T: TryFrom<Any, Error = Error>>(
    mapped: UnchangedOr<Any>,
) -> Result<Option<Option<T>>> {
    match mapped.into_option() {
        None => Ok(None),
        Some(any) if any.type_index() == TypeIndex::kTVMFFINone as i32 => Ok(Some(None)),
        Some(any) => Ok(Some(Some(T::try_from(any)?))),
    }
}

/// A statement mapped into a `SeqStmt` field: C++ assigns through `SeqStmt(Stmt)`, which
/// reuses a sequence and wraps (and normalizes) anything else.
#[inline]
fn changed_seq(mapped: UnchangedOr<Any>) -> Result<Option<SeqStmt>> {
    changed::<Stmt>(mapped)?.map(SeqStmt::from_stmt).transpose()
}

/// Set the `ty` of an expression node in place. `ExprObj` is the `#[repr(C)]` prefix of every
/// expression node (directly, or through `ConstantObj`), so the cast lands on its `ty` field.
///
/// # Safety
///
/// `N` must be an expression node type.
#[inline]
unsafe fn set_expr_ty<N: ObjectCore>(node: &mut N, ty: Type) {
    (*std::ptr::from_mut(node).cast::<ExprObj>()).ty = ty;
}

fn prim_type_error() -> Error {
    Error::new(TYPE_ERROR, "TensorLoad requires a concrete PrimType", "")
}

// ---- Leaves --------------------------------------------------------------------------------

fn leaf_visit(_value: &StructuralView, _children: &mut VisitChildren<'_>) -> VisitResult {
    Ok(())
}

fn leaf_mutate(_value: &StructuralView, _children: &mut MutateChildren<'_>) -> MutateResult {
    Ok(UnchangedOr::unchanged())
}

// ---- Expressions ---------------------------------------------------------------------------

/// C++ `VarVisit`. skips: name.
fn var_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<VarObj>(value);
    // A PrimType carries only a dtype, so it has nothing to visit. Broad callbacks do not see
    // this skipped field; dynamically typed Vars still descend through the Type value.
    if node.ty.as_node::<PrimTypeObj>().is_none() {
        // Only Simple is clamped: Pattern co-introduces type fields such as TensorType shape
        // variables, so that ambient region must continue through the dynamic type.
        if children.def_region_kind() == DefRegionKind::Simple {
            children.visit_with(&node.ty, DefRegionKind::None)?;
        } else {
            children.visit(&node.ty)?;
        }
    }
    Ok(())
}

/// C++ `VarMutate` / `VarMaybeInplaceMutate`. skips: name.
fn var_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    if let Some(mapped) = children.var_remap_get(value)? {
        return Ok(mapped);
    }
    let kind = children.def_region_kind();
    if kind == DefRegionKind::None {
        return Ok(UnchangedOr::unchanged());
    }
    let node = node::<VarObj>(value);
    let mut result = UnchangedOr::<Any>::unchanged();
    // A PrimType carries only a dtype, so it has nothing to substitute.
    if node.ty.as_node::<PrimTypeObj>().is_none() {
        // Pattern co-introduces type fields such as TensorType shape variables; Simple does not.
        let mapped_ty = if kind == DefRegionKind::Simple {
            children.mutate_with(&node.ty, DefRegionKind::None)?
        } else {
            children.mutate(&node.ty)?
        };
        if !same(&mapped_ty, &node.ty) {
            let ty = Type::try_from(mapped_ty.into_option().expect("changed"))?;
            let loc = node.loc.clone();
            let name = node.name.clone();
            // SAFETY: `node` is not used after the write.
            result = if let Some(this) = unsafe { children.inplace_node::<VarObj>() } {
                unsafe { set_expr_ty(this, ty) };
                UnchangedOr::changed(value.to_owned())
            } else {
                UnchangedOr::changed(wrap::<Var>(VarObj::new(loc, ty, name)))
            };
        }
    }
    if !result.is_unchanged() || kind == DefRegionKind::Pattern {
        children.var_remap_set(value, &result)?;
    }
    Ok(result)
}

/// C++ `CallVisit`. skips: attrs, constant metadata left untouched like the classic functors.
fn call_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<CallObj>(value);
    if node.ty.as_node::<PrimTypeObj>().is_none() {
        children.visit(&node.ty)?;
    }
    // An Op is an interned registry singleton, so it has nothing to visit.
    if node.op.as_node::<OpObj>().is_none() {
        children.visit(&node.op)?;
    }
    children.visit(&node.args)?;
    // An empty ty_args has no element to traverse.
    if !node.ty_args.is_empty() {
        children.visit(&node.ty_args)?;
    }
    Ok(())
}

/// C++ `CallMutate` / `CallMaybeInplaceMutate`. skips: attrs.
fn call_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<CallObj>(value);
    let mapped_ty = if node.ty.as_node::<PrimTypeObj>().is_none() {
        children.mutate(&node.ty)?
    } else {
        UnchangedOr::unchanged()
    };
    let mapped_op = if node.op.as_node::<OpObj>().is_none() {
        children.mutate(&node.op)?
    } else {
        UnchangedOr::unchanged()
    };
    let mapped_args = children.mutate(&node.args)?;
    let mapped_ty_args = if !node.ty_args.is_empty() {
        children.mutate(&node.ty_args)?
    } else {
        UnchangedOr::unchanged()
    };
    if same(&mapped_ty, &node.ty)
        && same(&mapped_op, &node.op)
        && same(&mapped_args, &node.args)
        && same(&mapped_ty_args, &node.ty_args)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let ty = changed::<Type>(mapped_ty)?;
    let op = changed::<Expr>(mapped_op)?;
    let args = changed::<Array<Expr>>(mapped_args)?;
    let ty_args = changed::<Array<Type>>(mapped_ty_args)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<CallObj>() } {
        if let Some(ty) = ty {
            unsafe { set_expr_ty(this, ty) };
        }
        if let Some(op) = op {
            this.op = op;
        }
        if let Some(args) = args {
            this.args = args;
        }
        if let Some(ty_args) = ty_args {
            this.ty_args = ty_args;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Call>(CallObj::new(
        node.loc.clone(),
        ty.unwrap_or_else(|| node.ty.clone()),
        op.unwrap_or_else(|| node.op.clone()),
        args.unwrap_or_else(|| node.args.clone()),
        node.attrs.clone(),
        ty_args.unwrap_or_else(|| node.ty_args.clone()),
    ))))
}

/// C++ `TensorLoadVisit`.
fn tensor_load_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<TensorLoadObj>(value);
    children.visit(&node.ty)?;
    children.visit(&node.source)?;
    children.visit(&node.indices)
}

/// C++ `TensorLoadMutate` / `TensorLoadMaybeInplaceMutate`.
fn tensor_load_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<TensorLoadObj>(value);
    let result_ty = value_or::<Type>(children.mutate(&node.ty)?, &node.ty)?;
    if result_ty.as_node::<PrimTypeObj>().is_none() {
        return Err(prim_type_error());
    }
    let mapped_source = children.mutate(&node.source)?;
    let mapped_indices = children.mutate(&node.indices)?;
    if result_ty.same_as(&node.ty)
        && same(&mapped_source, &node.source)
        && same(&mapped_indices, &node.indices)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let source = changed::<Expr>(mapped_source)?;
    let indices = changed::<Array<Expr>>(mapped_indices)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<TensorLoadObj>() } {
        unsafe { set_expr_ty(this, result_ty) };
        if let Some(source) = source {
            this.source = source;
        }
        if let Some(indices) = indices {
            this.indices = indices;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<TensorLoad>(TensorLoadObj::new(
        node.loc.clone(),
        result_ty,
        source.unwrap_or_else(|| node.source.clone()),
        indices.unwrap_or_else(|| node.indices.clone()),
    ))))
}

/// C++ `TensorRegionVisit`.
fn tensor_region_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<TensorRegionObj>(value);
    children.visit(&node.ty)?;
    children.visit(&node.source)?;
    children.visit(&node.region)
}

/// C++ `TensorRegionMutate` / `TensorRegionMaybeInplaceMutate`.
fn tensor_region_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<TensorRegionObj>(value);
    let mapped_ty = children.mutate(&node.ty)?;
    let mapped_source = children.mutate(&node.source)?;
    let mapped_region = children.mutate(&node.region)?;
    if same(&mapped_ty, &node.ty)
        && same(&mapped_source, &node.source)
        && same(&mapped_region, &node.region)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let ty = changed::<Type>(mapped_ty)?;
    let source = changed::<Expr>(mapped_source)?;
    let region = changed::<Array<Range>>(mapped_region)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<TensorRegionObj>() } {
        if let Some(ty) = ty {
            unsafe { set_expr_ty(this, ty) };
        }
        if let Some(source) = source {
            this.source = source;
        }
        if let Some(region) = region {
            this.region = region;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<TensorRegion>(TensorRegionObj::new(
        node.loc.clone(),
        ty.unwrap_or_else(|| node.ty.clone()),
        source.unwrap_or_else(|| node.source.clone()),
        region.unwrap_or_else(|| node.region.clone()),
    ))))
}

/// C++ `TupleVisit`.
fn tuple_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<TupleObj>(value);
    children.visit(&node.ty)?;
    children.visit(&node.fields)
}

/// C++ `TupleMutate` / `TupleMaybeInplaceMutate`.
fn tuple_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<TupleObj>(value);
    let mapped_ty = children.mutate(&node.ty)?;
    let mapped_fields = children.mutate(&node.fields)?;
    if same(&mapped_ty, &node.ty) && same(&mapped_fields, &node.fields) {
        return Ok(UnchangedOr::unchanged());
    }
    let ty = changed::<Type>(mapped_ty)?;
    let fields = changed::<Array<Expr>>(mapped_fields)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<TupleObj>() } {
        if let Some(ty) = ty {
            unsafe { set_expr_ty(this, ty) };
        }
        if let Some(fields) = fields {
            this.fields = fields;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Tuple>(TupleObj::new(
        node.loc.clone(),
        ty.unwrap_or_else(|| node.ty.clone()),
        fields.unwrap_or_else(|| node.fields.clone()),
    ))))
}

/// C++ `TupleGetItemVisit`. skips: index.
fn tuple_get_item_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<TupleGetItemObj>(value);
    children.visit(&node.ty)?;
    children.visit(&node.tuple_value)
}

/// C++ `TupleGetItemMutate` / `TupleGetItemMaybeInplaceMutate`. skips: index.
fn tuple_get_item_mutate(
    value: &StructuralView,
    children: &mut MutateChildren<'_>,
) -> MutateResult {
    let node = node::<TupleGetItemObj>(value);
    let mapped_ty = children.mutate(&node.ty)?;
    let mapped_tuple = children.mutate(&node.tuple_value)?;
    if same(&mapped_ty, &node.ty) && same(&mapped_tuple, &node.tuple_value) {
        return Ok(UnchangedOr::unchanged());
    }
    let ty = changed::<Type>(mapped_ty)?;
    let tuple = changed::<Expr>(mapped_tuple)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<TupleGetItemObj>() } {
        if let Some(ty) = ty {
            unsafe { set_expr_ty(this, ty) };
        }
        if let Some(tuple) = tuple {
            this.tuple_value = tuple;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<TupleGetItem>(TupleGetItemObj::new(
        node.loc.clone(),
        ty.unwrap_or_else(|| node.ty.clone()),
        tuple.unwrap_or_else(|| node.tuple_value.clone()),
        node.index,
    ))))
}

/// C++ `RangeVisit`.
fn range_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<RangeObj>(value);
    children.visit(&node.min)?;
    children.visit(&node.extent)
}

/// C++ `RangeMutate` / `RangeMaybeInplaceMutate`.
fn range_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<RangeObj>(value);
    let mapped_min = children.mutate(&node.min)?;
    let mapped_extent = children.mutate(&node.extent)?;
    if same(&mapped_min, &node.min) && same(&mapped_extent, &node.extent) {
        return Ok(UnchangedOr::unchanged());
    }
    let min = changed::<Expr>(mapped_min)?;
    let extent = changed::<Expr>(mapped_extent)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<RangeObj>() } {
        if let Some(min) = min {
            this.min = min;
        }
        if let Some(extent) = extent {
            this.extent = extent;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Range>(RangeObj::new(
        min.unwrap_or_else(|| node.min.clone()),
        extent.unwrap_or_else(|| node.extent.clone()),
        node.loc.clone(),
    ))))
}

/// C++ `DataTypeImmVisit`.
fn data_type_imm_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<DataTypeImmObj>(value);
    children.visit(&node.ty)?;
    children.visit(&node.value)
}

/// C++ `DataTypeImmMutate` / `DataTypeImmMaybeInplaceMutate`.
fn data_type_imm_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<DataTypeImmObj>(value);
    let mapped_ty = children.mutate(&node.ty)?;
    let mapped_value = children.mutate(&node.value)?;
    let ty = changed::<Type>(mapped_ty)?;
    let dtype = changed::<tvm_ffi::DLDataType>(mapped_value)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<DataTypeImmObj>() } {
        // The C++ in-place hook writes every replacement without an unchanged check.
        if let Some(ty) = ty {
            unsafe { set_expr_ty(this, ty) };
        }
        if let Some(dtype) = dtype {
            this.value = dtype;
        }
        return Ok(UnchangedOr::unchanged());
    }
    let ty_same = ty.as_ref().is_none_or(|ty| ty.same_as(&node.ty));
    let value_same = dtype.is_none_or(|dtype| dtype == node.value);
    if ty_same && value_same {
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<DataTypeImm>(DataTypeImmObj::new(
        node.loc.clone(),
        ty.unwrap_or_else(|| node.ty.clone()),
        dtype.unwrap_or(node.value),
    ))))
}

/// C++ `ConstantVisit` and `OpaqueExprVisit`: the type only.
fn ty_only_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<ExprObj>(value);
    children.visit(&node.ty)
}

/// C++ `OpaqueExprMutate` / `OpaqueExprMaybeInplaceMutate`.
fn opaque_expr_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<OpaqueExprObj>(value);
    let mapped_ty = children.mutate(&node.ty)?;
    if same(&mapped_ty, &node.ty) {
        return Ok(UnchangedOr::unchanged());
    }
    let ty = Type::try_from(mapped_ty.into_option().expect("changed"))?;
    // SAFETY: `node` is not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<OpaqueExprObj>() } {
        unsafe { set_expr_ty(this, ty) };
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<OpaqueExpr>(OpaqueExprObj::new(node.loc.clone(), ty))))
}

/// C++ `ConstantMutate<StringImmNode>` / `ConstantMaybeInplaceMutate<StringImmNode>`.
fn string_imm_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<StringImmObj>(value);
    let mapped_ty = children.mutate(&node.ty)?;
    if same(&mapped_ty, &node.ty) {
        return Ok(UnchangedOr::unchanged());
    }
    let ty = Type::try_from(mapped_ty.into_option().expect("changed"))?;
    // SAFETY: `node` is not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<StringImmObj>() } {
        unsafe { set_expr_ty(this, ty) };
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<StringImm>(StringImmObj::new(
        node.loc.clone(),
        ty,
        node.value.clone(),
    ))))
}

/// C++ `ConstantMutate<GenericConstNode>` / `ConstantMaybeInplaceMutate<GenericConstNode>`.
fn generic_const_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<GenericConstObj>(value);
    let mapped_ty = children.mutate(&node.ty)?;
    if same(&mapped_ty, &node.ty) {
        return Ok(UnchangedOr::unchanged());
    }
    let ty = Type::try_from(mapped_ty.into_option().expect("changed"))?;
    // SAFETY: `node` is not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<GenericConstObj>() } {
        unsafe { set_expr_ty(this, ty) };
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<GenericConst>(GenericConstObj::new(
        node.loc.clone(),
        ty,
        node.value.clone(),
    ))))
}

// ---- Statements ----------------------------------------------------------------------------

/// C++ `BindVisit`.
fn bind_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<BindObj>(value);
    children.visit_with(&node.var, DefRegionKind::Simple)?;
    children.visit(&node.value)
}

/// C++ `Var::CopyWithType`.
fn copy_with_type(var: &Var, ty: Type) -> Result<Var> {
    if AnyView::from(var).type_index() != VarObj::type_index() {
        return Err(Error::new(
            TYPE_ERROR,
            "Cannot copy a Var runtime subtype as an ordinary Var",
            "",
        ));
    }
    Ok(Var::from_complete_fields(var.loc.clone(), ty, var.name.clone()))
}

/// C++ `BindMutate` / `BindMaybeInplaceMutate`: the value, then the variable as a simple
/// definition; the variable follows the value's type and later uses resolve through the remap.
fn bind_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<BindObj>(value);
    let old_value_type = node.value.ty.clone();
    let mapped_value = children.mutate(&node.value)?;
    let mut value_changed = !same(&mapped_value, &node.value);
    let new_value = value_or::<Expr>(mapped_value, &node.value)?;
    value_changed |= !old_value_type.same_as(&new_value.ty);
    let old_var_type = node.var.ty.clone();
    let mapped_var = children.mutate_with(&node.var, DefRegionKind::Simple)?;
    let mut var_changed = !same(&mapped_var, &node.var);
    let mut new_var = value_or::<Var>(mapped_var, &node.var)?;
    var_changed |= !old_var_type.same_as(&new_var.ty);
    if (value_changed || var_changed) && !new_var.ty.same_as(&new_value.ty) {
        new_var = copy_with_type(&new_var, new_value.ty.clone())?;
    }
    if !new_var.same_as(&node.var) {
        children.var_remap_set(&node.var, &UnchangedOr::changed(new_var.clone().into()))?;
    }
    if new_var.same_as(&node.var) && new_value.same_as(&node.value) {
        return Ok(UnchangedOr::unchanged());
    }
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<BindObj>() } {
        this.var = new_var;
        this.value = new_value;
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Bind>(BindObj::new(node.loc.clone(), new_var, new_value))))
}

/// C++ `EvaluateVisit`.
fn evaluate_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    children.visit(&node::<EvaluateObj>(value).value)
}

/// C++ `EvaluateMutate` / `EvaluateMaybeInplaceMutate`.
fn evaluate_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<EvaluateObj>(value);
    let mapped_value = children.mutate(&node.value)?;
    if same(&mapped_value, &node.value) {
        return Ok(UnchangedOr::unchanged());
    }
    let new_value = Expr::try_from(mapped_value.into_option().expect("changed"))?;
    // SAFETY: `node` is not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<EvaluateObj>() } {
        this.value = new_value;
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Evaluate>(EvaluateObj::new(node.loc.clone(), new_value))))
}

/// C++ `AssertStmtVisit`. skips: error_kind and message_parts, constant assertion metadata.
fn assert_stmt_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    children.visit(&node::<AssertStmtObj>(value).condition)
}

/// C++ `AssertStmtMutate` / `AssertStmtMaybeInplaceMutate`.
fn assert_stmt_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<AssertStmtObj>(value);
    let mapped_condition = children.mutate(&node.condition)?;
    if same(&mapped_condition, &node.condition) {
        return Ok(UnchangedOr::unchanged());
    }
    let condition = Expr::try_from(mapped_condition.into_option().expect("changed"))?;
    // SAFETY: `node` is not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<AssertStmtObj>() } {
        this.condition = condition;
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<AssertStmt>(AssertStmtObj::new(
        node.loc.clone(),
        condition,
        node.error_kind.clone(),
        node.message_parts.clone(),
    ))))
}

/// C++ `ForVisit`. skips: kind and constant annotations.
fn for_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<ForObj>(value);
    children.visit_with(&node.loop_var, DefRegionKind::Simple)?;
    children.visit(&node.min)?;
    children.visit(&node.extent)?;
    children.visit(&node.body)?;
    children.visit(&node.step)
}

/// C++ `ForMutate` / `ForMaybeInplaceMutate`. skips: kind and constant annotations.
fn for_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<ForObj>(value);
    let mapped_loop_var = children.mutate_with(&node.loop_var, DefRegionKind::Simple)?;
    let mapped_min = children.mutate(&node.min)?;
    let mapped_extent = children.mutate(&node.extent)?;
    let mapped_body = children.mutate(&node.body)?;
    let mapped_step = children.mutate(&node.step)?;
    if same(&mapped_loop_var, &node.loop_var)
        && same(&mapped_min, &node.min)
        && same(&mapped_extent, &node.extent)
        && same(&mapped_body, &node.body)
        && same(&mapped_step, &node.step)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let loop_var = changed::<Var>(mapped_loop_var)?;
    let min = changed::<Expr>(mapped_min)?;
    let extent = changed::<Expr>(mapped_extent)?;
    let body = changed_seq(mapped_body)?;
    let step = changed_optional::<Expr>(mapped_step)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<ForObj>() } {
        if let Some(loop_var) = loop_var {
            this.loop_var = loop_var;
        }
        if let Some(min) = min {
            this.min = min;
        }
        if let Some(extent) = extent {
            this.extent = extent;
        }
        if let Some(body) = body {
            this.body = body;
        }
        if let Some(step) = step {
            this.step = step;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<For>(ForObj::new(
        node.loc.clone(),
        loop_var.unwrap_or_else(|| node.loop_var.clone()),
        min.unwrap_or_else(|| node.min.clone()),
        extent.unwrap_or_else(|| node.extent.clone()),
        node.kind,
        body.unwrap_or_else(|| node.body.clone()),
        node.annotations.clone(),
        step.unwrap_or_else(|| node.step.clone()),
    ))))
}

/// C++ `WhileVisit`.
fn while_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<WhileObj>(value);
    children.visit(&node.condition)?;
    children.visit(&node.body)
}

/// C++ `WhileMutate` / `WhileMaybeInplaceMutate`.
fn while_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<WhileObj>(value);
    let mapped_condition = children.mutate(&node.condition)?;
    let mapped_body = children.mutate(&node.body)?;
    if same(&mapped_condition, &node.condition) && same(&mapped_body, &node.body) {
        return Ok(UnchangedOr::unchanged());
    }
    let condition = changed::<Expr>(mapped_condition)?;
    let body = changed_seq(mapped_body)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<WhileObj>() } {
        if let Some(condition) = condition {
            this.condition = condition;
        }
        if let Some(body) = body {
            this.body = body;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<While>(WhileObj::new(
        node.loc.clone(),
        condition.unwrap_or_else(|| node.condition.clone()),
        body.unwrap_or_else(|| node.body.clone()),
    ))))
}

/// C++ `IfVisit`.
fn if_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<IfObj>(value);
    children.visit(&node.condition)?;
    children.visit(&node.then_case)?;
    children.visit(&node.else_case)
}

/// C++ `IfMutate` / `IfMaybeInplaceMutate`.
fn if_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<IfObj>(value);
    let mapped_condition = children.mutate(&node.condition)?;
    let mapped_then = children.mutate(&node.then_case)?;
    let mapped_else = children.mutate(&node.else_case)?;
    if same(&mapped_condition, &node.condition)
        && same(&mapped_then, &node.then_case)
        && same(&mapped_else, &node.else_case)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let condition = changed::<Expr>(mapped_condition)?;
    let then_case = changed_seq(mapped_then)?;
    let else_case = match changed_optional::<Stmt>(mapped_else)? {
        None => None,
        Some(None) => Some(None),
        Some(Some(stmt)) => Some(Some(SeqStmt::from_stmt(stmt)?)),
    };
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<IfObj>() } {
        if let Some(condition) = condition {
            this.condition = condition;
        }
        if let Some(then_case) = then_case {
            this.then_case = then_case;
        }
        if let Some(else_case) = else_case {
            this.else_case = else_case;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<If>(IfObj::new(
        node.loc.clone(),
        condition.unwrap_or_else(|| node.condition.clone()),
        then_case.unwrap_or_else(|| node.then_case.clone()),
        else_case.unwrap_or_else(|| node.else_case.clone()),
    ))))
}

/// C++ `SeqStmtVisit`.
fn seq_stmt_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    children.visit(&node::<SeqStmtObj>(value).seq)
}

/// C++ `detail::IsSeqStmtNoOp`: `Evaluate` of an integer 0.
#[inline]
fn is_seq_stmt_no_op(evaluate: Option<&EvaluateObj>) -> bool {
    evaluate.is_some_and(|evaluate| {
        evaluate.value.as_node::<IntImmObj>().is_some_and(|imm| imm.value.is_zero())
    })
}

/// Whether a final element makes the sequence splice: a no-op is dropped and a nested
/// sequence is expanded (one level, as the C++ `MutateSeqStmtChanged` and
/// `SeqStmtExpandNested` do).
#[inline]
fn seq_element_splices(evaluate: Option<&EvaluateObj>, nested: Option<&SeqStmtObj>) -> bool {
    nested.is_some() || is_seq_stmt_no_op(evaluate)
}

#[inline]
fn splice_seq_element(rebuilt: &mut Vec<Stmt>, stmt: Stmt) {
    if is_seq_stmt_no_op(stmt.as_node::<EvaluateObj>()) {
        return;
    }
    match stmt.as_node::<SeqStmtObj>() {
        Some(nested) => rebuilt.extend(nested.seq.iter()),
        None => rebuilt.push(stmt),
    }
}

/// C++ `SeqStmtMutate` / `SeqStmtMaybeInplaceMutate` (`detail::MutateSeqStmt`): every element,
/// then the sequence is rebuilt when an element became a no-op or is itself a sequence, and
/// otherwise replacements are written back.
///
/// Elements may be updated in place only when the node is and its `seq` array is not shared:
/// the engine establishes ownership of the SeqStmt, but `seq` is a field and needs its own
/// check. Replacements that keep the length go into a unique array in place; a rebuilt array
/// replaces `seq` in place on an in-place node.
fn seq_stmt_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<SeqStmtObj>(value);
    let kind = children.def_region_kind();
    let seq_unique = ObjectArc::strong_count(<Array<Stmt> as ObjectRefCore>::data(&node.seq)) == 1;
    let inplace = children.is_inplace() && seq_unique;
    let mode = if inplace { InplaceMode::Allow } else { InplaceMode::Disallow };
    let views = node.seq.element_views();
    let mut replaced: Vec<(usize, Stmt)> = Vec::new();
    let mut splices = false;
    for (index, item) in views.iter().enumerate() {
        let mapped = children.mutate_with_mode(item, kind, mode)?;
        if same(&mapped, item) {
            splices |=
                seq_element_splices(item.as_node::<EvaluateObj>(), item.as_node::<SeqStmtObj>());
        } else {
            let stmt = Stmt::try_from(mapped.into_option().expect("changed"))?;
            splices |=
                seq_element_splices(stmt.as_node::<EvaluateObj>(), stmt.as_node::<SeqStmtObj>());
            replaced.push((index, stmt));
        }
    }
    if replaced.is_empty() && !splices {
        return Ok(UnchangedOr::unchanged());
    }
    if inplace && !splices {
        // Same length: write the replacements into the unique array, as the C++ in-place hook
        // writes its slots.
        // SAFETY: `node` and `views` are not used after this point.
        let this = unsafe { children.inplace_node::<SeqStmtObj>() }.expect("in-place node");
        for (index, stmt) in replaced {
            let written = this.seq.set_if_unique(index, stmt)?;
            debug_assert!(written, "the sequence array was checked to be unique");
        }
        return Ok(UnchangedOr::unchanged());
    }
    let mut rebuilt = Vec::with_capacity(views.len());
    let mut replaced = replaced.into_iter().peekable();
    for (index, item) in views.iter().enumerate() {
        let stmt = match replaced.peek() {
            Some((at, _)) if *at == index => replaced.next().expect("peeked").1,
            _ => item.cast::<Stmt>().expect("a sequence element is a Stmt"),
        };
        splice_seq_element(&mut rebuilt, stmt);
    }
    let seq = Array::new(rebuilt);
    // SAFETY: `node` and `views` are not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<SeqStmtObj>() } {
        this.seq = seq;
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(SeqStmt::from_complete_fields(node.loc.clone(), seq).into()))
}

/// C++ `RegionStmtVisit`.
fn region_stmt_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<RegionStmtObj>(value);
    children.visit(&node.attrs)?;
    children.visit(&node.args)?;
    children.visit_with(&node.body_params, DefRegionKind::Simple)?;
    children.visit(&node.body)?;
    children.visit_with(&node.result_vars, DefRegionKind::Simple)
}

/// C++ `RegionStmtMutate` / `RegionStmtMaybeInplaceMutate`.
fn region_stmt_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<RegionStmtObj>(value);
    let mapped_attrs = children.mutate(&node.attrs)?;
    let mapped_args = children.mutate(&node.args)?;
    let mapped_body_params = children.mutate_with(&node.body_params, DefRegionKind::Simple)?;
    let mapped_body = children.mutate(&node.body)?;
    let mapped_result_vars = children.mutate_with(&node.result_vars, DefRegionKind::Simple)?;
    if same(&mapped_args, &node.args)
        && same(&mapped_attrs, &node.attrs)
        && same(&mapped_body_params, &node.body_params)
        && same(&mapped_body, &node.body)
        && same(&mapped_result_vars, &node.result_vars)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let attrs = changed::<DictAttrs>(mapped_attrs)?;
    let args = changed::<Array<Expr>>(mapped_args)?;
    let body_params = changed::<Array<Var>>(mapped_body_params)?;
    let body = changed_seq(mapped_body)?;
    let result_vars = changed::<Array<Var>>(mapped_result_vars)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<RegionStmtObj>() } {
        if let Some(args) = args {
            this.args = args;
        }
        if let Some(attrs) = attrs {
            this.attrs = attrs;
        }
        if let Some(body_params) = body_params {
            this.body_params = body_params;
        }
        if let Some(body) = body {
            this.body = body;
        }
        if let Some(result_vars) = result_vars {
            this.result_vars = result_vars;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<RegionStmt>(RegionStmtObj::new(
        node.loc.clone(),
        node.op.clone(),
        args.unwrap_or_else(|| node.args.clone()),
        body_params.unwrap_or_else(|| node.body_params.clone()),
        attrs.unwrap_or_else(|| node.attrs.clone()),
        body.unwrap_or_else(|| node.body.clone()),
        result_vars.unwrap_or_else(|| node.result_vars.clone()),
    ))))
}

/// C++ `ReturnVisit`.
fn return_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    children.visit(&node::<ReturnObj>(value).value)
}

/// C++ `ReturnMutate` / `ReturnMaybeInplaceMutate`.
fn return_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<ReturnObj>(value);
    let mapped_value = children.mutate(&node.value)?;
    if same(&mapped_value, &node.value) {
        return Ok(UnchangedOr::unchanged());
    }
    let new_value = Expr::try_from(mapped_value.into_option().expect("changed"))?;
    // SAFETY: `node` is not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<ReturnObj>() } {
        this.value = new_value;
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Return>(ReturnObj::new(node.loc.clone(), new_value))))
}

/// C++ `TensorStoreVisit`.
fn tensor_store_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<TensorStoreObj>(value);
    children.visit(&node.dest)?;
    children.visit(&node.indices)?;
    children.visit(&node.value)
}

/// C++ `TensorStoreMutate` / `TensorStoreMaybeInplaceMutate`.
fn tensor_store_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<TensorStoreObj>(value);
    let mapped_dest = children.mutate(&node.dest)?;
    let mapped_indices = children.mutate(&node.indices)?;
    let mapped_value = children.mutate(&node.value)?;
    if same(&mapped_dest, &node.dest)
        && same(&mapped_value, &node.value)
        && same(&mapped_indices, &node.indices)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let dest = changed::<Expr>(mapped_dest)?;
    let indices = changed::<Array<Expr>>(mapped_indices)?;
    let new_value = changed::<Expr>(mapped_value)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<TensorStoreObj>() } {
        if let Some(dest) = dest {
            this.dest = dest;
        }
        if let Some(new_value) = new_value {
            this.value = new_value;
        }
        if let Some(indices) = indices {
            this.indices = indices;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<TensorStore>(TensorStoreObj::new(
        node.loc.clone(),
        dest.unwrap_or_else(|| node.dest.clone()),
        indices.unwrap_or_else(|| node.indices.clone()),
        new_value.unwrap_or_else(|| node.value.clone()),
    ))))
}

// ---- Types ---------------------------------------------------------------------------------

/// C++ `PtrTypeVisit`. skips: scope (canonical enum identity).
fn ptr_type_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    children.visit(&node::<PtrTypeObj>(value).element_type)
}

/// C++ `PtrTypeMutate` / `PtrTypeMaybeInplaceMutate`.
fn ptr_type_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<PtrTypeObj>(value);
    let mapped = children.mutate(&node.element_type)?;
    if same(&mapped, &node.element_type) {
        return Ok(UnchangedOr::unchanged());
    }
    let element_type = Type::try_from(mapped.into_option().expect("changed"))?;
    // SAFETY: `node` is not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<PtrTypeObj>() } {
        this.element_type = element_type;
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<PtrType>(PtrTypeObj::new(
        node.loc.clone(),
        element_type,
        node.scope.clone(),
    ))))
}

/// C++ `FuncTypeVisit`.
fn func_type_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<FuncTypeObj>(value);
    children.visit(&node.arg_types)?;
    children.visit(&node.ret_type)
}

/// C++ `FuncTypeMutate` / `FuncTypeMaybeInplaceMutate`.
fn func_type_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<FuncTypeObj>(value);
    let mapped_arg_types = children.mutate(&node.arg_types)?;
    let mapped_ret_type = children.mutate(&node.ret_type)?;
    if same(&mapped_arg_types, &node.arg_types) && same(&mapped_ret_type, &node.ret_type) {
        return Ok(UnchangedOr::unchanged());
    }
    let arg_types = changed::<Array<Type>>(mapped_arg_types)?;
    let ret_type = changed::<Type>(mapped_ret_type)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<FuncTypeObj>() } {
        if let Some(arg_types) = arg_types {
            this.arg_types = arg_types;
        }
        if let Some(ret_type) = ret_type {
            this.ret_type = ret_type;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<FuncType>(FuncTypeObj::new(
        node.loc.clone(),
        arg_types.unwrap_or_else(|| node.arg_types.clone()),
        ret_type.unwrap_or_else(|| node.ret_type.clone()),
    ))))
}

/// C++ `TupleTypeVisit`.
fn tuple_type_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    children.visit(&node::<TupleTypeObj>(value).fields)
}

/// C++ `TupleTypeMutate` / `TupleTypeMaybeInplaceMutate`.
fn tuple_type_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<TupleTypeObj>(value);
    let mapped = children.mutate(&node.fields)?;
    if same(&mapped, &node.fields) {
        return Ok(UnchangedOr::unchanged());
    }
    let fields = Array::<Type>::try_from(mapped.into_option().expect("changed"))?;
    // SAFETY: `node` is not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<TupleTypeObj>() } {
        this.fields = fields;
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<TupleType>(TupleTypeObj::new(node.loc.clone(), fields))))
}

/// C++ `TensorTypeVisit` (tirx). skips: scope, data_alignment, offset_factor.
fn tensor_type_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<TensorTypeObj>(value);
    children.visit(&node.dtype)?;
    children.visit(&node.shape)?;
    // Empty strides denote the common compact layout.
    if !node.strides.is_empty() {
        children.visit(&node.strides)?;
    }
    children.visit(&node.elem_offset)?;
    children.visit(&node.layout)
}

/// C++ `TensorTypeMutate` / `TensorTypeMaybeInplaceMutate` (tirx).
fn tensor_type_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<TensorTypeObj>(value);
    let mapped_dtype = children.mutate(&node.dtype)?;
    let mapped_shape = children.mutate(&node.shape)?;
    let mapped_strides = if !node.strides.is_empty() {
        children.mutate(&node.strides)?
    } else {
        UnchangedOr::unchanged()
    };
    let mapped_elem_offset = children.mutate(&node.elem_offset)?;
    let mapped_layout = children.mutate(&node.layout)?;
    if same(&mapped_dtype, &node.dtype)
        && same(&mapped_shape, &node.shape)
        && same(&mapped_strides, &node.strides)
        && same(&mapped_elem_offset, &node.elem_offset)
        && same(&mapped_layout, &node.layout)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let dtype = changed::<crate::ir::PrimType>(mapped_dtype)?;
    let shape = changed::<Array<Expr>>(mapped_shape)?;
    let strides = changed::<Array<Expr>>(mapped_strides)?;
    let elem_offset = changed::<Expr>(mapped_elem_offset)?;
    let layout = changed_optional::<crate::tirx::Layout>(mapped_layout)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<TensorTypeObj>() } {
        if let Some(dtype) = dtype {
            this.dtype = dtype;
        }
        if let Some(shape) = shape {
            this.shape = shape;
        }
        if let Some(strides) = strides {
            this.strides = strides;
        }
        if let Some(elem_offset) = elem_offset {
            this.elem_offset = elem_offset;
        }
        if let Some(layout) = layout {
            this.layout = layout;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<TensorType>(TensorTypeObj::new(
        node.loc.clone(),
        dtype.unwrap_or_else(|| node.dtype.clone()),
        node.scope.clone(),
        shape.unwrap_or_else(|| node.shape.clone()),
        strides.unwrap_or_else(|| node.strides.clone()),
        elem_offset.unwrap_or_else(|| node.elem_offset.clone()),
        node.data_alignment,
        node.offset_factor,
        layout.unwrap_or_else(|| node.layout.clone()),
    ))))
}

// ---- Primitive expressions -----------------------------------------------------------------
//
// A primitive expression's type is always a PrimType and stays unchanged in normal mutation,
// so none of these hooks visits `ty`; `Let` recomputes it from its body.

/// A `prim` node with operands `a` and `b`, served by the C++ `BinaryVisit` / `BinaryMutate` /
/// `BinaryMaybeInplaceMutate` templates.
trait BinaryNode: ObjectCore {
    fn a(&self) -> &Expr;
    fn b(&self) -> &Expr;
    fn set_a(&mut self, a: Expr);
    fn set_b(&mut self, b: Expr);
    /// A copy with the operands replaced and every other field kept.
    fn rebuild(&self, a: Expr, b: Expr) -> Any;
}

macro_rules! impl_binary_node {
    ($($node:ident / $obj:ident),+ $(,)?) => {
        $(impl BinaryNode for $obj {
            #[inline]
            fn a(&self) -> &Expr {
                &self.a
            }
            #[inline]
            fn b(&self) -> &Expr {
                &self.b
            }
            #[inline]
            fn set_a(&mut self, a: Expr) {
                self.a = a;
            }
            #[inline]
            fn set_b(&mut self, b: Expr) {
                self.b = b;
            }
            #[inline]
            fn rebuild(&self, a: Expr, b: Expr) -> Any {
                wrap::<$node>($obj::new(self.loc.clone(), self.ty.clone(), a, b))
            }
        })+
    };
}

impl_binary_node!(
    Add / AddObj,
    Sub / SubObj,
    Mul / MulObj,
    Div / DivObj,
    Mod / ModObj,
    FloorDiv / FloorDivObj,
    FloorMod / FloorModObj,
    Min / MinObj,
    Max / MaxObj,
    EQ / EQObj,
    NE / NEObj,
    LT / LTObj,
    LE / LEObj,
    GT / GTObj,
    GE / GEObj,
    And / AndObj,
    Or / OrObj,
    BitwiseAnd / BitwiseAndObj,
    BitwiseOr / BitwiseOrObj,
    BitwiseXor / BitwiseXorObj,
    LShift / LShiftObj,
    RShift / RShiftObj,
);

fn register_binary_hooks<N: BinaryNode>() -> Result<()> {
    register_structural_hooks::<N>(hooks(binary_visit::<N>, binary_mutate::<N>))
}

/// C++ `BinaryVisit<TNode>`.
fn binary_visit<N: BinaryNode>(
    value: &StructuralView,
    children: &mut VisitChildren<'_>,
) -> VisitResult {
    let node = node::<N>(value);
    children.visit(node.a())?;
    children.visit(node.b())
}

/// C++ `BinaryMutate<TNode>` / `BinaryMaybeInplaceMutate<TNode>`.
fn binary_mutate<N: BinaryNode>(
    value: &StructuralView,
    children: &mut MutateChildren<'_>,
) -> MutateResult {
    let node = node::<N>(value);
    let mapped_a = children.mutate(node.a())?;
    let mapped_b = children.mutate(node.b())?;
    if same(&mapped_a, node.a()) && same(&mapped_b, node.b()) {
        return Ok(UnchangedOr::unchanged());
    }
    let a = changed::<Expr>(mapped_a)?;
    let b = changed::<Expr>(mapped_b)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<N>() } {
        if let Some(a) = a {
            this.set_a(a);
        }
        if let Some(b) = b {
            this.set_b(b);
        }
        return Ok(UnchangedOr::unchanged());
    }
    // StructuralMap preserves node types. A rewrite that changes operand dtypes must keep the
    // operands compatible and set the result type itself.
    Ok(UnchangedOr::changed(
        node.rebuild(a.unwrap_or_else(|| node.a().clone()), b.unwrap_or_else(|| node.b().clone())),
    ))
}

/// A `prim` node with one operand: `Cast` (`value`), `Not` and `BitwiseNot` (`a`).
trait UnaryNode: ObjectCore {
    fn operand(&self) -> &Expr;
    fn set_operand(&mut self, operand: Expr);
    fn rebuild(&self, operand: Expr) -> Any;
}

macro_rules! impl_unary_node {
    ($($node:ident / $obj:ident . $field:ident),+ $(,)?) => {
        $(impl UnaryNode for $obj {
            #[inline]
            fn operand(&self) -> &Expr {
                &self.$field
            }
            #[inline]
            fn set_operand(&mut self, operand: Expr) {
                self.$field = operand;
            }
            #[inline]
            fn rebuild(&self, operand: Expr) -> Any {
                wrap::<$node>($obj::new(self.loc.clone(), self.ty.clone(), operand))
            }
        })+
    };
}

impl_unary_node!(Cast / CastObj.value, Not / NotObj.a, BitwiseNot / BitwiseNotObj.a);

fn register_unary_hooks<N: UnaryNode>() -> Result<()> {
    register_structural_hooks::<N>(hooks(unary_visit::<N>, unary_mutate::<N>))
}

/// C++ `CastVisit`, `NotVisit`, `BitwiseNotVisit`.
fn unary_visit<N: UnaryNode>(
    value: &StructuralView,
    children: &mut VisitChildren<'_>,
) -> VisitResult {
    children.visit(node::<N>(value).operand())
}

/// C++ `CastMutate`, `NotMutate`, `BitwiseNotMutate` and their in-place variants.
fn unary_mutate<N: UnaryNode>(
    value: &StructuralView,
    children: &mut MutateChildren<'_>,
) -> MutateResult {
    let node = node::<N>(value);
    let mapped = children.mutate(node.operand())?;
    if same(&mapped, node.operand()) {
        return Ok(UnchangedOr::unchanged());
    }
    let operand = Expr::try_from(mapped.into_option().expect("changed"))?;
    // SAFETY: `node` is not used after the write.
    if let Some(this) = unsafe { children.inplace_node::<N>() } {
        this.set_operand(operand);
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(node.rebuild(operand)))
}

/// C++ `SelectVisit`.
fn select_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<SelectObj>(value);
    children.visit(&node.condition)?;
    children.visit(&node.true_value)?;
    children.visit(&node.false_value)
}

/// C++ `SelectMutate` / `SelectMaybeInplaceMutate`.
fn select_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<SelectObj>(value);
    let mapped_condition = children.mutate(&node.condition)?;
    let mapped_true = children.mutate(&node.true_value)?;
    let mapped_false = children.mutate(&node.false_value)?;
    if same(&mapped_condition, &node.condition)
        && same(&mapped_true, &node.true_value)
        && same(&mapped_false, &node.false_value)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let condition = changed::<Expr>(mapped_condition)?;
    let true_value = changed::<Expr>(mapped_true)?;
    let false_value = changed::<Expr>(mapped_false)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<SelectObj>() } {
        if let Some(condition) = condition {
            this.condition = condition;
        }
        if let Some(true_value) = true_value {
            this.true_value = true_value;
        }
        if let Some(false_value) = false_value {
            this.false_value = false_value;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Select>(SelectObj::new(
        node.loc.clone(),
        node.ty.clone(),
        condition.unwrap_or_else(|| node.condition.clone()),
        true_value.unwrap_or_else(|| node.true_value.clone()),
        false_value.unwrap_or_else(|| node.false_value.clone()),
    ))))
}

/// C++ `LetVisit`.
fn let_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<LetObj>(value);
    children.visit_with(&node.var, DefRegionKind::Simple)?;
    children.visit(&node.value)?;
    children.visit(&node.body)
}

/// C++ `LetMutate` / `LetMaybeInplaceMutate`: like `Bind`, then the body; the node's type
/// follows the body's.
fn let_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<LetObj>(value);
    let old_value_type = node.value.ty.clone();
    let mapped_value = children.mutate(&node.value)?;
    let mut value_changed = !same(&mapped_value, &node.value);
    let new_value = value_or::<Expr>(mapped_value, &node.value)?;
    value_changed |= !old_value_type.same_as(&new_value.ty);
    let old_var_type = node.var.ty.clone();
    let mapped_var = children.mutate_with(&node.var, DefRegionKind::Simple)?;
    let mut var_changed = !same(&mapped_var, &node.var);
    let mut new_var = value_or::<Var>(mapped_var, &node.var)?;
    var_changed |= !old_var_type.same_as(&new_var.ty);
    if (value_changed || var_changed) && !new_var.ty.same_as(&new_value.ty) {
        new_var = copy_with_type(&new_var, new_value.ty.clone())?;
    }
    if !new_var.same_as(&node.var) {
        children.var_remap_set(&node.var, &UnchangedOr::changed(new_var.clone().into()))?;
    }
    let mapped_body = children.mutate(&node.body)?;
    let new_body = value_or::<Expr>(mapped_body, &node.body)?;
    if new_var.same_as(&node.var)
        && new_value.same_as(&node.value)
        && new_body.same_as(&node.body)
        && new_body.ty.same_as(&node.ty)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let body_ty = new_body.ty.clone();
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<LetObj>() } {
        this.var = new_var;
        this.value = new_value;
        this.body = new_body;
        unsafe { set_expr_ty(this, body_ty) };
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Let>(LetObj::new(
        node.loc.clone(),
        body_ty,
        new_var,
        new_value,
        new_body,
    ))))
}

/// C++ `RampVisit`.
fn ramp_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<RampObj>(value);
    children.visit(&node.base_)?;
    children.visit(&node.stride)?;
    children.visit(&node.lanes)
}

/// C++ `RampMutate` / `RampMaybeInplaceMutate`.
fn ramp_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<RampObj>(value);
    let mapped_base = children.mutate(&node.base_)?;
    let mapped_stride = children.mutate(&node.stride)?;
    let mapped_lanes = children.mutate(&node.lanes)?;
    if same(&mapped_base, &node.base_)
        && same(&mapped_stride, &node.stride)
        && same(&mapped_lanes, &node.lanes)
    {
        return Ok(UnchangedOr::unchanged());
    }
    let base = changed::<Expr>(mapped_base)?;
    let stride = changed::<Expr>(mapped_stride)?;
    let lanes = changed::<Expr>(mapped_lanes)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<RampObj>() } {
        if let Some(base) = base {
            this.base_ = base;
        }
        if let Some(stride) = stride {
            this.stride = stride;
        }
        if let Some(lanes) = lanes {
            this.lanes = lanes;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Ramp>(RampObj::new(
        node.loc.clone(),
        node.ty.clone(),
        base.unwrap_or_else(|| node.base_.clone()),
        stride.unwrap_or_else(|| node.stride.clone()),
        lanes.unwrap_or_else(|| node.lanes.clone()),
    ))))
}

/// C++ `BroadcastVisit`.
fn broadcast_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<BroadcastObj>(value);
    children.visit(&node.value)?;
    children.visit(&node.lanes)
}

/// C++ `BroadcastMutate` / `BroadcastMaybeInplaceMutate`.
fn broadcast_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<BroadcastObj>(value);
    let mapped_value = children.mutate(&node.value)?;
    let mapped_lanes = children.mutate(&node.lanes)?;
    if same(&mapped_value, &node.value) && same(&mapped_lanes, &node.lanes) {
        return Ok(UnchangedOr::unchanged());
    }
    let new_value = changed::<Expr>(mapped_value)?;
    let lanes = changed::<Expr>(mapped_lanes)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<BroadcastObj>() } {
        if let Some(new_value) = new_value {
            this.value = new_value;
        }
        if let Some(lanes) = lanes {
            this.lanes = lanes;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Broadcast>(BroadcastObj::new(
        node.loc.clone(),
        node.ty.clone(),
        new_value.unwrap_or_else(|| node.value.clone()),
        lanes.unwrap_or_else(|| node.lanes.clone()),
    ))))
}

/// C++ `ShuffleVisit`.
fn shuffle_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> VisitResult {
    let node = node::<ShuffleObj>(value);
    children.visit(&node.vectors)?;
    children.visit(&node.indices)
}

/// C++ `ShuffleMutate` / `ShuffleMaybeInplaceMutate`.
fn shuffle_mutate(value: &StructuralView, children: &mut MutateChildren<'_>) -> MutateResult {
    let node = node::<ShuffleObj>(value);
    let mapped_vectors = children.mutate(&node.vectors)?;
    let mapped_indices = children.mutate(&node.indices)?;
    if same(&mapped_vectors, &node.vectors) && same(&mapped_indices, &node.indices) {
        return Ok(UnchangedOr::unchanged());
    }
    let vectors = changed::<Array<Expr>>(mapped_vectors)?;
    let indices = changed::<Array<Expr>>(mapped_indices)?;
    // SAFETY: `node` is not used after the first write.
    if let Some(this) = unsafe { children.inplace_node::<ShuffleObj>() } {
        if let Some(vectors) = vectors {
            this.vectors = vectors;
        }
        if let Some(indices) = indices {
            this.indices = indices;
        }
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(wrap::<Shuffle>(ShuffleObj::new(
        node.loc.clone(),
        node.ty.clone(),
        vectors.unwrap_or_else(|| node.vectors.clone()),
        indices.unwrap_or_else(|| node.indices.clone()),
    ))))
}
