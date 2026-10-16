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
//! The rules are the C++ `PVar` patterns, in the same order with the same conditions
//! ([`crate::ops::pattern`]).

use tvm_ffi::{dispatch, Any, BigInt, Mutator, ObjectRefCore, Result};

use super::analyzer::{bind_range, AnalyzerBase, AnalyzerScope, AnalyzerState};
use super::util::{function_pass, int_imm, MutatorExt};
use crate::ir::{Expr, For, IntImm, SeqStmt, Stmt, VarObj};
use crate::ops::pattern::{broadcast, PVar, Pattern, PatternOps};
use crate::ops::{ex, Ex, PrimOps};
use crate::prim::{Broadcast, FloorDiv, Not, GE, GT, LE, LT};
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

/// C++ `ffi::BigInt` `%` by a nonzero divisor, compared with zero: whether `c2` divides `c3`.
fn divides(c2: &PVar<IntImm>, c3: &PVar<IntImm>) -> Result<bool> {
    Ok(c3.get().value.try_rem(&c2.get().value)?.is_zero())
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

/// `floordiv(x * c1 + y, c2)` with the common factor `c3` of `c1` and `c2` taken out,
/// `floordiv(x * (c1 / c3) + floordiv(y, c3), c2 / c3)`, each new constant broadcast to `lanes`;
/// `None` unless `c1` and `c2` are positive with a common factor.
fn factored(
    x: &PVar<Expr>,
    c1: &PVar<IntImm>,
    c2: &PVar<IntImm>,
    y: Ex,
    lanes: Option<&PVar<Expr>>,
) -> Result<Option<Expr>> {
    let (c1, c2) = (c1.get(), c2.get());
    let (Some(c1_val), Some(c2_val)) = (positive(&c1.value), positive(&c2.value)) else {
        return Ok(None);
    };
    let c3 = gcd(c1_val, c2_val);
    if c3 <= 1 {
        return Ok(None);
    }
    let lift = |value: IntImm| -> Result<Expr> {
        match lanes {
            Some(lanes) => Ok(Broadcast::new(value, lanes.get())?.into()),
            None => Ok(value.into()),
        }
    };
    let (c1_ty, c2_ty) = (c1.prim_dtype()?, c2.prim_dtype()?);
    let c1_div = lift(int_imm(c1_ty, c1_val / c3)?)?;
    let c2_div = lift(int_imm(c2_ty, c2_val / c3)?)?;
    let c3 = lift(int_imm(c1_ty, c3)?)?;
    Ok(Some((ex(x.get()) * &c1_div + y.floordiv(&c3)).floordiv(&c2_div).expr()?))
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
        let (x, y, z, lanes) = (PVar::expr(), PVar::expr(), PVar::expr(), PVar::expr());
        let (c1, c2, c3) = (PVar::int(), PVar::int(), PVar::int());

        // floordiv(floormod(x, c1 * c2), c2) = floormod(floordiv(x, c2), c1)
        if x.floormod(&c3).floordiv(&c2).matches(&ret) && divides(&c2, &c3)? {
            let rewritten = x.floordiv(&c2).floormod(c3.floordiv(&c2));
            return self.recursive_rewrite(rewritten.eval()?, m);
        }
        let pattern = x.floormod(broadcast(&c3, &lanes)).floordiv(broadcast(&c2, &lanes));
        if pattern.matches(&ret) && divides(&c2, &c3)? {
            let rewritten =
                x.floordiv(broadcast(&c2, &lanes)).floormod(broadcast(c3.floordiv(&c2), &lanes));
            return self.recursive_rewrite(rewritten.eval()?, m);
        }

        // Factoring a product requires signed, non-wrapping index arithmetic.
        if ret.prim_dtype()?.code == tvm_ffi::DLDataTypeCode::kDLUInt as u8 {
            return Ok(ret);
        }

        // floordiv(x*c1*c3 + y, c2*c3) = floordiv(x*c1 + floordiv(y, c3), c2)
        if (&x * &c1 + &y).floordiv(&c2).matches(&ret) {
            if let Some(rewritten) = factored(&x, &c1, &c2, ex(y.get()), None)? {
                return self.recursive_rewrite(rewritten, m);
            }
        }
        if (&x * broadcast(&c1, &lanes) + &y).floordiv(broadcast(&c2, &lanes)).matches(&ret) {
            if let Some(rewritten) = factored(&x, &c1, &c2, ex(y.get()), Some(&lanes))? {
                return self.recursive_rewrite(rewritten, m);
            }
        }

        // floordiv(x*c1*c3 + y + z, c2*c3) = floordiv(x*c1 + floordiv(y + z, c3), c2)
        if (&x * &c1 + &y + &z).floordiv(&c2).matches(&ret) {
            if let Some(rewritten) = factored(&x, &c1, &c2, ex(y.get()) + z.get(), None)? {
                return self.recursive_rewrite(rewritten, m);
            }
        }
        let pattern = (&x * broadcast(&c1, &lanes) + &y + &z).floordiv(broadcast(&c2, &lanes));
        if pattern.matches(&ret) {
            if let Some(rewritten) = factored(&x, &c1, &c2, ex(y.get()) + z.get(), Some(&lanes))? {
                return self.recursive_rewrite(rewritten, m);
            }
        }
        Ok(ret)
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
        let ret: Expr = LT::new(a, b)?.into();
        let (x, c2) = (PVar::expr(), PVar::int());
        if x.less(&c2).matches(&ret)
            && c2.get().value > BigInt::from(0i64)
            && x.get().as_node::<VarObj>().is_none()
        {
            let rewritten = x.floordiv(&c2).less(1).eval()?;
            return Ok(self.recursive_rewrite(rewritten, m)?.into());
        }
        Ok(ret.into())
    }

    /// Double negation and negated comparisons.
    fn mutate_not(&mut self, op: Not, m: &mut Mutator) -> Result<Any> {
        let ret: Expr = m.default_mutate_as(self, &Expr::from(op))?;
        let (x, y) = (PVar::expr(), PVar::expr());
        let rewritten = if (!!&x).matches(&ret) {
            x.get()
        } else if (!x.less_equal(&y)).matches(&ret) {
            y.less(&x).eval()?
        } else if (!x.greater_equal(&y)).matches(&ret) {
            x.less(&y).eval()?
        } else if (!x.less(&y)).matches(&ret) {
            y.less_equal(&x).eval()?
        } else if (!x.greater(&y)).matches(&ret) {
            x.less_equal(&y).eval()?
        } else {
            ret
        };
        Ok(rewritten.into())
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
