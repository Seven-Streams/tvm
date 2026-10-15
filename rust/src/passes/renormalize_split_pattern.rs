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

//! `tirx.RenormalizeSplitPattern` (`src/tirx/transform/renormalize_split_pattern.cc`): rewrite
//! `floordiv(floormod(x, c1 * c2), c2)` to `floormod(floordiv(x, c2), c1)`, factor common
//! constants out of `floordiv(x * c1 + y, c2)`, and put comparisons in `<` form.
//!
//! The C++ rules are `PVar` patterns; here each is explicit matching in the same order with the
//! same conditions. A pattern's `Eval()` folds constants and otherwise builds the raw node, which
//! is what the `ops` operators do on operands of one type, as a matched node's operands are.

use tvm_ffi::{dispatch, Any, BigInt, Mutator, ObjectRefCore, Result};

use super::analyzer::{bind_range, AnalyzerBase, AnalyzerScope, AnalyzerState};
use super::util::{function_pass, int_imm, prim_dtype, MutatorExt};
use crate::ir::{Expr, For, IntImmObj, SeqStmt, Stmt, VarObj};
use crate::ops::{ex, PrimOps};
use crate::prim::{
    AddObj, Broadcast, BroadcastObj, FloorDiv, FloorDivObj, FloorModObj, GEObj, GTObj, LEObj,
    LTObj, MulObj, Not, NotObj, GE, GT, LE, LT,
};
use crate::sym::Analyzer;
use crate::transform::Pass;

/// The C++ limit on nested recursive rewrites.
const MAX_RECUR_DEPTH: usize = 5;

struct SplitPatternRenormalizer {
    base: AnalyzerBase,
    recur_depth: usize,
}

impl AnalyzerState for SplitPatternRenormalizer {
    fn base(&mut self) -> &mut AnalyzerBase {
        &mut self.base
    }
}

/// A `PVar<IntImm>` match: the constant `expr` holds, if it is one.
fn int_const(expr: &Expr) -> Option<&IntImmObj> {
    expr.as_node::<IntImmObj>()
}

/// A `broadcast(c, lanes)` match with an integer constant `c`: the constant and the lanes.
fn int_broadcast(expr: &Expr) -> Option<(&Expr, &Expr)> {
    let broadcast = expr.as_node::<BroadcastObj>()?;
    int_const(&broadcast.value)?;
    Some((&broadcast.value, &broadcast.lanes))
}

/// The value of an integer constant matched by `int_const` or `int_broadcast`.
fn value(constant: &Expr) -> &BigInt {
    &int_const(constant).expect("matched as an integer constant").value
}

/// A `PVar<PrimExpr>` bound twice: the same object, or deep-equal (`PEqualChecker<PrimExpr>`).
fn same_value(lhs: &Expr, rhs: &Expr) -> Result<bool> {
    if lhs.same_as(rhs) {
        return Ok(true);
    }
    tvm_ffi::cached_global_func!("prim.expr_deep_equal")
        .call_tuple((lhs.clone(), rhs.clone()))?
        .try_into()
}

/// C++ `ffi::BigInt` `%` by a nonzero divisor, compared with zero.
fn divides(divisor: &BigInt, value: &BigInt) -> Result<bool> {
    Ok(value.try_rem(divisor)?.is_zero())
}

/// `ZeroAwareGCD` of two positive constants.
fn gcd(lhs: i128, rhs: i128) -> i128 {
    let (mut a, mut b) = (lhs.abs(), rhs.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn positive(value: &BigInt) -> Option<i128> {
    value.to_i128().filter(|v| *v > 0)
}

/// `x * c1` with an integer constant (or, with `broadcast`, a broadcast integer constant) `c1`:
/// `x`, `c1`, and the lanes of the broadcast.
fn scaled(expr: &Expr, broadcast: bool) -> Option<(&Expr, &Expr, Option<&Expr>)> {
    let mul = expr.as_node::<MulObj>()?;
    if broadcast {
        let (c1, lanes) = int_broadcast(&mul.b)?;
        Some((&mul.a, c1, Some(lanes)))
    } else {
        int_const(&mul.b)?;
        Some((&mul.a, &mul.b, None))
    }
}

impl SplitPatternRenormalizer {
    /// `RecursiveRewrite(x)`: mutate `x` again, at most `MAX_RECUR_DEPTH` levels deep.
    fn recursive_rewrite(&mut self, x: Expr, m: &mut Mutator) -> Result<Expr> {
        if self.recur_depth >= MAX_RECUR_DEPTH {
            return Ok(x);
        }
        self.recur_depth += 1;
        let result = m.mutate_as(self, &x);
        self.recur_depth -= 1;
        result
    }

    /// The `floordiv` rules, in the C++ order, on `ret = floordiv(a, b)`.
    fn rewrite_floordiv(&mut self, ret: Expr, m: &mut Mutator) -> Result<Expr> {
        let Some(div) = ret.as_node::<FloorDivObj>() else { return Ok(ret) };
        // floordiv(floormod(x, c3), c2) = floormod(floordiv(x, c2), floordiv(c3, c2)) if c2 | c3.
        if let (Some(modulo), Some(c2)) = (div.a.as_node::<FloorModObj>(), int_const(&div.b)) {
            if let Some(c3) = int_const(&modulo.b) {
                if divides(&c2.value, &c3.value)? {
                    let x = &modulo.a;
                    let rewritten = x.floordiv(&div.b).floormod(ex(&modulo.b).floordiv(&div.b));
                    return self.recursive_rewrite(rewritten.expr()?, m);
                }
            }
        }
        // The same with broadcast constants of one lane count.
        if let (Some(modulo), Some((c2, lanes2))) =
            (div.a.as_node::<FloorModObj>(), int_broadcast(&div.b))
        {
            if let Some((c3, lanes)) = int_broadcast(&modulo.b) {
                if same_value(lanes, lanes2)? && divides(value(c2), value(c3))? {
                    let x = &modulo.a;
                    let divisor: Expr = Broadcast::new(c2, lanes)?.into();
                    let quotient = ex(c3).floordiv(c2).expr()?;
                    let modulus: Expr = Broadcast::new(quotient, lanes)?.into();
                    let rewritten = x.floordiv(&divisor).floormod(&modulus);
                    return self.recursive_rewrite(rewritten.expr()?, m);
                }
            }
        }

        // Factoring a product requires signed, non-wrapping index arithmetic.
        if prim_dtype(&ret)?.code == tvm_ffi::DLDataTypeCode::kDLUInt as u8 {
            return Ok(ret);
        }

        // floordiv(x*c1*c3 + y, c2*c3) = floordiv(x*c1 + floordiv(y, c3), c2), and the same with
        // a third term: floordiv(x*c1*c3 + y + z, c2*c3) = floordiv(x*c1 + floordiv(y + z, c3), c2).
        for (three_terms, broadcast) in [(false, false), (false, true), (true, false), (true, true)]
        {
            if let Some(rewritten) = self.factor(div, three_terms, broadcast)? {
                return self.recursive_rewrite(rewritten, m);
            }
        }
        Ok(ret)
    }

    /// The factoring rule on `floordiv(x * c1 + y [+ z], c2)`, `None` when it does not apply.
    fn factor(
        &self,
        div: &FloorDivObj,
        three_terms: bool,
        broadcast: bool,
    ) -> Result<Option<Expr>> {
        let Some(sum) = div.a.as_node::<AddObj>() else { return Ok(None) };
        // `x * c1 + y` is the whole numerator, or the left operand of `... + z`.
        let (product_sum, z) = if three_terms {
            let Some(inner) = sum.a.as_node::<AddObj>() else { return Ok(None) };
            (inner, Some(&sum.b))
        } else {
            (sum, None)
        };
        let Some((x, c1, lanes)) = scaled(&product_sum.a, broadcast) else { return Ok(None) };
        let y = &product_sum.b;
        let c2 = match (broadcast, lanes) {
            (true, Some(lanes)) => match int_broadcast(&div.b) {
                Some((c2, lanes2)) if same_value(lanes, lanes2)? => c2,
                _ => return Ok(None),
            },
            _ => match int_const(&div.b) {
                Some(_) => &div.b,
                None => return Ok(None),
            },
        };
        let (Some(c1_val), Some(c2_val)) = (positive(value(c1)), positive(value(c2))) else {
            return Ok(None);
        };
        let c3 = gcd(c1_val, c2_val);
        if c3 <= 1 {
            return Ok(None);
        }
        let (c1_ty, c2_ty) = (prim_dtype(c1)?, prim_dtype(c2)?);
        let lift = |value: Expr| -> Result<Expr> {
            match lanes {
                Some(lanes) => Ok(Broadcast::new(value, lanes)?.into()),
                None => Ok(value),
            }
        };
        let c1_div = lift(int_imm(c1_ty, c1_val / c3)?.into())?;
        let c2_div = lift(int_imm(c2_ty, c2_val / c3)?.into())?;
        let c3 = lift(int_imm(c1_ty, c3)?.into())?;
        let remainder = match z {
            Some(z) => (ex(y) + z).floordiv(&c3),
            None => ex(y).floordiv(&c3),
        };
        Ok(Some((ex(x) * &c1_div + remainder).floordiv(&c2_div).expr()?))
    }
}

#[dispatch(mutate, policy = AnalyzerScope)]
impl SplitPatternRenormalizer {
    fn mutate_floordiv(&mut self, op: FloorDiv, m: &mut Mutator) -> Result<Any> {
        let a: Expr = m.mutate_as(self, &op.a)?;
        let b: Expr = m.mutate_as(self, &op.b)?;
        let ret = a.floordiv(&b).expr()?;
        Ok(self.rewrite_floordiv(ret, m)?.into())
    }

    /// `a <= b` is `!(b < a)`.
    fn mutate_le(&mut self, op: LE, m: &mut Mutator) -> Result<Any> {
        let rewritten: Expr = Not::new(op.b.less(&op.a))?.into();
        m.mutate(self, &rewritten)
    }

    /// `a > b` is `b < a`.
    fn mutate_gt(&mut self, op: GT, m: &mut Mutator) -> Result<Any> {
        let rewritten = op.b.less(&op.a).expr()?;
        m.mutate(self, &rewritten)
    }

    /// `a >= b` is `!(a < b)`.
    fn mutate_ge(&mut self, op: GE, m: &mut Mutator) -> Result<Any> {
        let rewritten: Expr = Not::new(op.a.less(&op.b))?.into();
        m.mutate(self, &rewritten)
    }

    /// `x < c` with a positive constant `c` and `x` not a variable is `floordiv(x, c) < 1`.
    fn mutate_lt(&mut self, op: LT, m: &mut Mutator) -> Result<Any> {
        let a: Expr = m.mutate_as(self, &op.a)?;
        let b: Expr = m.mutate_as(self, &op.b)?;
        let ret = LT::new(a.clone(), b.clone())?;
        if let Some(c2) = int_const(&b) {
            if c2.value > BigInt::from(0i64) && a.as_node::<VarObj>().is_none() {
                let rewritten = a.floordiv(&b).less(1).expr()?;
                return Ok(self.recursive_rewrite(rewritten, m)?.into());
            }
        }
        Ok(ret.into())
    }

    /// Double negation and negated comparisons.
    fn mutate_not(&mut self, op: Not, m: &mut Mutator) -> Result<Any> {
        let ret: Expr = m.default_mutate_as(self, &Expr::from(op))?;
        let Some(not) = ret.as_node::<NotObj>() else { return Ok(ret.into()) };
        let inner = &not.a;
        if let Some(inner) = inner.as_node::<NotObj>() {
            return Ok(inner.a.clone().into());
        }
        if let Some(le) = inner.as_node::<LEObj>() {
            return Ok(le.b.less(&le.a).expr()?.into());
        }
        if let Some(ge) = inner.as_node::<GEObj>() {
            return Ok(ge.a.less(&ge.b).expr()?.into());
        }
        if let Some(lt) = inner.as_node::<LTObj>() {
            return Ok(lt.b.less_equal(&lt.a).expr()?.into());
        }
        if let Some(gt) = inner.as_node::<GTObj>() {
            return Ok(gt.a.less_equal(&gt.b).expr()?.into());
        }
        Ok(ret.into())
    }

    /// The loop variable lies in `[min, min + extent)` throughout the loop.
    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        let analyzer = self.base.analyzer.clone();
        bind_range(&analyzer, &op.loop_var, &op.min, &op.extent)?;
        let _lower = analyzer.assume(&op.loop_var.greater_equal(&op.min).expr()?)?;
        let _upper = analyzer.assume(&op.loop_var.less(&op.min + &op.extent).expr()?)?;
        m.default_mutate(self, &op)
    }
}

pub fn renormalize_split_pattern() -> Result<Pass> {
    function_pass("tirx.RenormalizeSplitPattern", 0, |func, _module, _context| {
        let Some(body) = func.body.clone() else { return Ok(Some(func)) };
        let mut renormalizer =
            SplitPatternRenormalizer { base: AnalyzerBase::new(Analyzer::new()?)?, recur_depth: 0 };
        let body: Stmt = tvm_ffi::structural_mutate(body, &mut renormalizer)?.try_into()?;
        Ok(Some(func.with_body(Some(SeqStmt::from_stmt(body)?))))
    })
}
