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

//! The tensor geometry the flattening passes share: `TensorTypeNode::ElemOffset` and its index
//! simplification (`src/tirx/ir/tensor.cc`), `FlattenedTensor`
//! (`src/tirx/transform/flattened_tensor.h`), and small `TensorVar` helpers.

use tvm_ffi::{AnyView, Array, ObjectCore, ObjectRefCast, ObjectRefCore, Result, INTERNAL_ERROR};

use super::analyzer::is_const_int;
use crate::ir::{Expr, Var, VarObj};
use crate::ops::{deep_equal, int32, unknown_loc, Ex, PrimOps};
use crate::prim::{AddObj, FloorDivObj, FloorModObj, Max, MulObj};
use crate::sym::Analyzer;
use crate::tirx::TensorType;

/// A `TensorVar`: a `Var`, not a subclass, whose type is a `TensorType` (C++ `as<TensorVar>()`).
pub(crate) fn as_tensor_var(expr: &Expr) -> Option<(Var, TensorType)> {
    if expr.as_node::<VarObj>().is_none()
        || AnyView::from(expr).type_index() != VarObj::type_index()
    {
        return None;
    }
    let ty: TensorType = expr.ty.clone().try_cast().ok()?;
    Some((expr.clone().try_cast().ok()?, ty))
}

/// The `TensorType` of a variable known to be a tensor.
pub(crate) fn tensor_type(var: &Var) -> Result<TensorType> {
    var.ty.clone().try_cast()
}

/// C++ `FlattenedTensor(tensor)`: the 1-d storage view of `tensor`, which is `tensor` itself
/// when it is 1-d without strides.
pub(crate) fn flattened_tensor(tensor: &Var, ty: &TensorType) -> Result<(Var, TensorType)> {
    let (shape, strides) = (&ty.shape, &ty.strides);
    let extent: Ex = if !strides.is_empty() {
        check_cmp!(
            shape.len(), ==, strides.len(), INTERNAL_ERROR,
            "tensor.shape().size() == tensor.strides().size()"
        );
        // Cover every addressed element: permuted views can have their largest stride inside.
        let mut span = Ex::from(int32(1));
        for (extent, stride) in shape.iter().zip(strides.iter()) {
            span += (Ex::from(&extent) - 1) * &stride;
        }
        let outer = (Ex::from(&strides.get(0)?) * &shape.get(0)?).expr()?;
        Ex::from(Max::new(outer, span)?)
    } else {
        let mut extent = Ex::from(int32(1));
        for dim in shape.iter() {
            extent *= &dim;
        }
        extent
    };
    if shape.len() == 1 && strides.is_empty() {
        return Ok((tensor.clone(), ty.clone()));
    }
    // The original layout describes the old rank; reset it for the flattened shape.
    let flat_ty = TensorType::new_full(
        [extent.expr()?],
        &ty.dtype,
        ty.scope.clone(),
        [],
        Some(ty.elem_offset.clone()),
        ty.data_alignment,
        ty.offset_factor,
        None,
        unknown_loc()?,
    )?;
    Ok((Var::new_full(tensor.name.clone(), flat_ty.clone(), tensor.loc.clone())?, flat_ty))
}

/// C++ `TensorTypeNode::ElemOffset(indices, inner)`: the offset of an element in elements,
/// simplified with a fresh analyzer. `inner` leaves out `elem_offset`.
pub(crate) fn elem_offset(
    ty: &TensorType,
    indices: &Array<Expr>,
    inner: bool,
) -> Result<Array<Expr>> {
    let (shape, strides) = (&ty.shape, &ty.strides);
    check_cmp!(
        shape.len(), ==, indices.len(), INTERNAL_ERROR, "shape.size() == input_indices.size()",
        "TensorType is {}-dimensional, cannot be indexed with the {}-dimensional indices \
         provided.",
        shape.len(),
        indices.len(),
    );
    if !strides.is_empty() {
        check_cmp!(
            strides.len(), ==, indices.len(), INTERNAL_ERROR,
            "this->strides.size() == input_indices.size()",
            "If strides are defined, the index's dimensionality must match the dimensionality of \
             the index given.",
        );
    }
    let analyzer = Analyzer::new()?;
    let mut output = int32(0)?.into();
    for (i, index) in indices.iter().enumerate() {
        let next = if strides.is_empty() {
            Ex::from(&output) * &shape.get(i)? + &index
        } else {
            Ex::from(&output) + Ex::from(&index) * &strides.get(i)?
        };
        output = next.expr()?;
        if i > 0 {
            output = merge_mul_mod(&analyzer, output)?;
        }
    }
    if !is_const_int(&ty.elem_offset, 0) && !inner {
        output = (Ex::from(&output) + &ty.elem_offset).expr()?;
    }
    Ok(Array::new(vec![analyzer.simplify(&output, 2)?]))
}

/// C++ `ExprSplitAddition`: the terms of a tree of `Add`, left to right.
fn split_addition(expr: &Expr) -> Vec<Expr> {
    let mut terms = Vec::new();
    let mut stack = vec![expr.clone()];
    while let Some(top) = stack.pop() {
        match top.as_node::<AddObj>() {
            Some(add) => {
                stack.push(add.b.clone());
                stack.push(add.a.clone());
            }
            None => terms.push(top),
        }
    }
    terms
}

/// `lhs + rhs`, or `rhs` when there is no `lhs` yet.
fn sum(lhs: Option<Expr>, rhs: &Expr) -> Result<Expr> {
    match lhs {
        Some(lhs) => (Ex::from(&lhs) + rhs).expr(),
        None => Ok(rhs.clone()),
    }
}

/// C++ `MergeMulModInner`: `(a1 + ... + c / k * k1 ...) * kt ...` merged with `c % k` into
/// `(a1 + ...) * kt ... + c`, when the factors match and `c` equals the modulus operand mod `k`.
fn merge_mul_mod_inner(
    analyzer: &Analyzer,
    mult_expr: &Expr,
    mod_lhs: &Expr,
    mod_rhs: &Expr,
) -> Result<Option<Expr>> {
    let Some(mult) = mult_expr.as_node::<MulObj>() else { return Ok(None) };
    // 1. The outer multiplier.
    let mut mult_outer = mult.b.clone();
    let mut inner = mult.a.clone();
    while let Some(mult) = inner.as_node::<MulObj>() {
        mult_outer = (Ex::from(&mult.b) * &mult_outer).expr()?;
        inner = mult.a.clone();
    }
    // 2. Search `c / (...) * (...) + c % (...)`.
    let mut search = inner;
    let mut mult_inner: Option<Expr> = None;
    let mut no_opt_sum: Option<Expr> = None;
    loop {
        if let Some(div) = search.as_node::<FloorDivObj>() {
            let overall = match &mult_inner {
                Some(mult_inner) => (Ex::from(mult_inner) * &mult_outer).expr()?,
                None => mult_outer.clone(),
            };
            let remainder = (Ex::from(&div.a) - mod_lhs).floormod(mod_rhs).expr()?;
            if deep_equal(&overall, &div.b)?
                && deep_equal(&overall, mod_rhs)?
                && analyzer.can_prove_equal(&remainder, &int32(0)?.into())?
            {
                return Ok(Some(match no_opt_sum {
                    Some(sum) => (Ex::from(&sum) * &mult_outer + &div.a).expr()?,
                    None => div.a.clone(),
                }));
            }
            return Ok(None);
        } else if let Some(mult) = search.as_node::<MulObj>() {
            mult_inner = Some(match &mult_inner {
                Some(mult_inner) => (Ex::from(&mult.b) * mult_inner).expr()?,
                None => mult.b.clone(),
            });
            search = mult.a.clone();
        } else if let Some(add) = search.as_node::<AddObj>() {
            if mult_inner.is_some() {
                return Ok(None);
            }
            no_opt_sum = Some(sum(no_opt_sum, &add.a)?);
            search = add.b.clone();
        } else {
            return Ok(None);
        }
    }
}

/// The terms of a sum, sorted as C++ `MergeMulModInsertElements` sorts them.
#[derive(Default)]
struct Terms {
    mults: Vec<Expr>,
    mods: Vec<(Expr, Expr)>,
    no_opt_sum: Option<Expr>,
}

impl Terms {
    /// Returns whether a product and whether a modulus were among `terms`.
    fn insert(&mut self, terms: Vec<Expr>) -> Result<(bool, bool)> {
        let (mut has_mult, mut has_mod) = (false, false);
        for term in terms {
            if let Some(modulo) = term.as_node::<FloorModObj>() {
                has_mod = true;
                self.mods.push((modulo.a.clone(), modulo.b.clone()));
            } else if term.as_node::<MulObj>().is_some() {
                has_mult = true;
                self.mults.push(term);
            } else {
                self.no_opt_sum = Some(sum(self.no_opt_sum.take(), &term)?);
            }
        }
        Ok((has_mult, has_mod))
    }
}

/// C++ `MergeMulMod`: `x // y * y + x % y` becomes `x`, and every product that pairs with a
/// modulus as `MergeMulModInner` describes is merged, repeatedly.
fn merge_mul_mod(analyzer: &Analyzer, base: Expr) -> Result<Expr> {
    let mut simplified = base;
    if let Some(x) = div_mul_plus_mod(&simplified)? {
        simplified = x;
    }
    let simplified = analyzer.simplify(&simplified, 2)?;
    let mut terms = Terms::default();
    terms.insert(split_addition(&simplified))?;
    let mut find_opt = false;
    // The C++ list iterator over `mods`, as an index; `mods.len()` is its end.
    let mut search = 0;
    while search < terms.mods.len() {
        let mut inner_find_opt = false;
        let mut mult = 0;
        while mult < terms.mults.len() {
            let (mod_lhs, mod_rhs) = terms.mods[search].clone();
            match merge_mul_mod_inner(analyzer, &terms.mults[mult], &mod_lhs, &mod_rhs)? {
                Some(merged) => {
                    inner_find_opt = true;
                    // Erasing the current modulus leaves the iterator on its successor.
                    terms.mods.remove(search);
                    terms.mults.remove(mult);
                    let was_end = search == terms.mods.len();
                    let (has_mult, has_mod) = terms.insert(split_addition(&merged))?;
                    if has_mult {
                        search = 0;
                    } else if has_mod && was_end {
                        search = terms.mods.len() - 1;
                    }
                    break;
                }
                None => mult += 1,
            }
        }
        find_opt = find_opt || inner_find_opt;
        if !inner_find_opt {
            search += 1;
        }
    }
    if !find_opt {
        return Ok(simplified);
    }
    let mut total = terms.no_opt_sum;
    for mult in &terms.mults {
        total = Some(sum(total, mult)?);
    }
    for (lhs, rhs) in &terms.mods {
        total = Some(sum(total, &lhs.indexmod(rhs).expr()?)?);
    }
    Ok(total.expect("a merge leaves a term"))
}

/// The pattern `floordiv(x, y) * y + floormod(x, y)`, matched as `sym::PVar` matches it: the
/// first occurrence of `x` and `y` binds them, later ones must be deeply equal.
fn div_mul_plus_mod(expr: &Expr) -> Result<Option<Expr>> {
    let Some(add) = expr.as_node::<AddObj>() else { return Ok(None) };
    let Some(mult) = add.a.as_node::<MulObj>() else { return Ok(None) };
    let Some(div) = mult.a.as_node::<FloorDivObj>() else { return Ok(None) };
    let (x, y) = (&div.a, &div.b);
    if !same_or_deep_equal(&mult.b, y)? {
        return Ok(None);
    }
    let Some(modulo) = add.b.as_node::<FloorModObj>() else { return Ok(None) };
    if same_or_deep_equal(&modulo.a, x)? && same_or_deep_equal(&modulo.b, y)? {
        return Ok(Some(x.clone()));
    }
    Ok(None)
}

/// `PEqualChecker<PrimExpr>`: the same object, or deeply equal.
fn same_or_deep_equal(value: &Expr, bound: &Expr) -> Result<bool> {
    Ok(value.same_as(bound) || deep_equal(bound, value)?)
}
