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

//! Expression patterns, after `sym::Pattern` (`src/sym/pattern_match.h`).
//!
//! A pattern is built like an expression, from pattern variables ([`PVar`]), integer literals,
//! the operators `+`, `-`, `*`, `!` and the methods of [`PatternOps`] (`floordiv`, `less`, ...),
//! and [`broadcast`] / [`ramp`]. [`Pattern::matches`] tells whether an expression has its shape
//! and binds the variables to the parts they stand for; a variable that occurs twice matches equal
//! parts (`ExprDeepEqual`, or the same value for a constant). [`Pattern::eval`] builds the pattern
//! with the variables' values as C++ `Eval()` does: an arithmetic or comparison node folds
//! constants (`TryConstFold`) and is otherwise the raw node; `!`, `broadcast` and `ramp` are the
//! raw nodes. An integer literal is a constant of the other operand's type.
//!
//! ```ignore
//! let (x, c1, c2) = (PVar::expr(), PVar::int(), PVar::int());
//! // floordiv(floormod(x, c1), c2)
//! if x.floormod(&c1).floordiv(&c2).matches(&expr) && divides(c2.get(), c1.get()) {
//!     return x.floordiv(&c2).floormod(c1.floordiv(&c2)).eval();
//! }
//! ```

use std::cell::RefCell;
use std::marker::PhantomData;

use tvm_ffi::{ObjectRefCast, ObjectRefCore, Result, INTERNAL_ERROR};

use super::build::{make_const};
use super::fold;
use super::native::deep_equal;
use crate::ir::{Expr, IntImm};
use crate::prim::{
    Add, And, Broadcast, BroadcastObj, Div, FloorDiv, FloorMod, Max, Min, Mod, Mul, Not, NotObj,
    Or, Ramp, RampObj, Sub, EQ, GE, GT, LE, LT, NE,
};

/// An expression pattern (C++ `sym::Pattern`).
pub trait Pattern: Clone {
    /// Whether `expr` has this pattern's shape, with the variables bound to the parts they stand
    /// for (C++ `Match`). The variables keep those values until the next match.
    fn matches(&self, expr: &Expr) -> bool {
        self.reset();
        self.match_node(expr)
    }

    /// The pattern built with the variables' values (C++ `Eval()`).
    fn eval(&self) -> Result<Expr>;

    /// C++ `InitMatch_`: unbind the variables.
    #[doc(hidden)]
    fn reset(&self);

    /// C++ `Match_`: match `expr`, binding the variables that are not bound yet.
    #[doc(hidden)]
    fn match_node(&self, expr: &Expr) -> bool;

    /// `eval`, with an integer literal typed like `like` (C++ `PConstWithTypeLike`).
    #[doc(hidden)]
    fn eval_like(&self, _like: &Expr) -> Result<Expr> {
        self.eval()
    }

    /// Whether this is an integer literal, which takes its type from the other operand.
    #[doc(hidden)]
    const LITERAL: bool = false;
}

/// What converts to a pattern: a pattern, a reference to a [`PVar`], or an integer literal.
pub trait IntoPattern {
    type Pattern: Pattern;
    fn into_pattern(self) -> Self::Pattern;
}

// ---------------------------------------------------------------------------------------------
// Variables and literals

/// A pattern variable (C++ `PVar<T>`): `PVar<Expr>` matches any expression, `PVar<IntImm>` an
/// integer constant. Patterns hold variables by reference.
pub struct PVar<T> {
    value: RefCell<Option<T>>,
}

impl PVar<Expr> {
    /// A variable that matches any expression (C++ `PVar<PrimExpr>`).
    pub fn expr() -> Self {
        Self { value: RefCell::new(None) }
    }
}

impl PVar<IntImm> {
    /// A variable that matches an integer constant (C++ `PVar<IntImm>`).
    pub fn int() -> Self {
        Self { value: RefCell::new(None) }
    }
}

impl<T: Clone> PVar<T> {
    /// The value the last match bound (C++ `Eval()` of a variable). Reading a variable no match
    /// bound is a bug in the caller, and panics.
    pub fn get(&self) -> T {
        self.value.borrow().clone().expect("a pattern variable is read after a match binds it")
    }
}

impl<T: Bindable> PVar<T> {
    /// [`Pattern::matches`] of the variable alone: whether it can stand for `expr`, binding it.
    pub fn matches(&self, expr: &Expr) -> bool {
        Pattern::matches(&self, expr)
    }
}

/// What a [`PVar`] can stand for, and when two of them are equal (C++ `PEqualChecker`).
pub trait Bindable: Clone {
    fn bind(expr: &Expr) -> Option<Self>;
    fn same(&self, other: &Self) -> bool;
    fn to_expr(&self) -> Expr;
}

impl Bindable for Expr {
    fn bind(expr: &Expr) -> Option<Self> {
        Some(expr.clone())
    }

    /// The same object, or deep-equal; a comparison that fails is no match.
    fn same(&self, other: &Self) -> bool {
        self.same_as(other) || deep_equal(self, other).unwrap_or(false)
    }

    fn to_expr(&self) -> Expr {
        self.clone()
    }
}

impl Bindable for IntImm {
    fn bind(expr: &Expr) -> Option<Self> {
        expr.clone().try_cast().ok()
    }

    fn same(&self, other: &Self) -> bool {
        self.value == other.value
    }

    fn to_expr(&self) -> Expr {
        self.clone().into()
    }
}

impl<T: Bindable> Pattern for &PVar<T> {
    fn eval(&self) -> Result<Expr> {
        Ok(self.get().to_expr())
    }

    fn reset(&self) {
        self.value.replace(None);
    }

    fn match_node(&self, expr: &Expr) -> bool {
        let Some(value) = T::bind(expr) else { return false };
        let mut bound = self.value.borrow_mut();
        match bound.as_ref() {
            Some(previous) => previous.same(&value),
            None => {
                *bound = Some(value);
                true
            }
        }
    }
}

impl<'a, T: Bindable> IntoPattern for &'a PVar<T> {
    type Pattern = &'a PVar<T>;
    fn into_pattern(self) -> Self::Pattern {
        self
    }
}

/// An integer literal in a pattern (C++ `PConstWithTypeLike`): it matches an integer constant of
/// that value, and evaluates to a constant of the other operand's type.
#[derive(Clone, Copy)]
pub struct PInt(i64);

impl Pattern for PInt {
    fn eval(&self) -> Result<Expr> {
        tvm_ffi::bail!(INTERNAL_ERROR, "an integer literal pattern takes its type from an operand")
    }

    fn reset(&self) {}

    fn match_node(&self, expr: &Expr) -> bool {
        IntImm::bind(expr).is_some_and(|imm| imm.value.to_i64() == Some(self.0))
    }

    fn eval_like(&self, like: &Expr) -> Result<Expr> {
        make_const(like.prim_dtype()?, self.0)
    }

    const LITERAL: bool = true;
}

impl IntoPattern for i64 {
    type Pattern = PInt;
    fn into_pattern(self) -> PInt {
        PInt(self)
    }
}

impl IntoPattern for i32 {
    type Pattern = PInt;
    fn into_pattern(self) -> PInt {
        PInt(self as i64)
    }
}

// ---------------------------------------------------------------------------------------------
// Composite patterns

/// A binary node of type `N` (C++ `PBinaryExpr`).
pub struct PBinary<N, A, B> {
    a: A,
    b: B,
    node: PhantomData<N>,
}

impl<N, A: Clone, B: Clone> Clone for PBinary<N, A, B> {
    fn clone(&self) -> Self {
        Self { a: self.a.clone(), b: self.b.clone(), node: PhantomData }
    }
}

impl<N, A, B> PBinary<N, A, B> {
    fn new(a: A, b: B) -> Self {
        Self { a, b, node: PhantomData }
    }
}

/// A binary node type a pattern matches and builds.
pub trait BinaryNode {
    /// The operands of `expr`, if it is this node.
    fn operands(expr: &Expr) -> Option<(&Expr, &Expr)>;
    /// `TryConstFold` of this node.
    fn fold(a: &Expr, b: &Expr) -> Result<Option<Expr>>;
    /// The raw node.
    fn build(a: Expr, b: Expr) -> Result<Expr>;
}

/// `BinaryNode` for each node, with its `TryConstFold`.
macro_rules! binary_nodes {
    ($($node:ident: $fold:expr;)*) => {$(
        impl BinaryNode for $node {
            fn operands(expr: &Expr) -> Option<(&Expr, &Expr)> {
                let node = expr.as_node::<<$node as ObjectRefCore>::ContainerType>()?;
                Some((&node.a, &node.b))
            }

            fn fold(a: &Expr, b: &Expr) -> Result<Option<Expr>> {
                $fold(a, b)
            }

            fn build(a: Expr, b: Expr) -> Result<Expr> {
                Ok($node::new(a, b)?.into())
            }
        }
    )*};
}

binary_nodes! {
    Add: fold::add;
    Sub: fold::sub;
    Mul: fold::mul;
    Div: fold::div;
    Mod: fold::truncmod;
    FloorDiv: fold::floordiv;
    FloorMod: fold::floormod;
    Min: fold::min;
    Max: fold::max;
    EQ: |a, b| fold::compare(a, b, |x, y| x == y, |x, y| x == y);
    NE: |a, b| fold::compare(a, b, |x, y| x != y, |x, y| x != y);
    LT: |a, b| fold::compare(a, b, |x, y| x < y, |x, y| x < y);
    LE: |a, b| fold::compare(a, b, |x, y| x <= y, |x, y| x <= y);
    GT: |a, b| fold::compare(a, b, |x, y| x > y, |x, y| x > y);
    GE: |a, b| fold::compare(a, b, |x, y| x >= y, |x, y| x >= y);
    And: fold::and;
    Or: fold::or;
}

impl<N: BinaryNode, A: Pattern, B: Pattern> Pattern for PBinary<N, A, B> {
    fn eval(&self) -> Result<Expr> {
        let (a, b) = if A::LITERAL {
            let b = self.b.eval()?;
            (self.a.eval_like(&b)?, b)
        } else {
            let a = self.a.eval()?;
            let b = self.b.eval_like(&a)?;
            (a, b)
        };
        match N::fold(&a, &b)? {
            Some(folded) => Ok(folded),
            None => N::build(a, b),
        }
    }

    fn reset(&self) {
        self.a.reset();
        self.b.reset();
    }

    fn match_node(&self, expr: &Expr) -> bool {
        N::operands(expr).is_some_and(|(a, b)| self.a.match_node(a) && self.b.match_node(b))
    }
}

/// `!a` (C++ `PNotExpr`); it evaluates to the raw `Not`.
#[derive(Clone)]
pub struct PNot<A>(A);

impl<A: Pattern> Pattern for PNot<A> {
    fn eval(&self) -> Result<Expr> {
        Ok(Not::new(self.0.eval()?)?.into())
    }

    fn reset(&self) {
        self.0.reset();
    }

    fn match_node(&self, expr: &Expr) -> bool {
        expr.as_node::<NotObj>().is_some_and(|not| self.0.match_node(&not.a))
    }
}

/// `broadcast(value, lanes)` (C++ `PBroadcastExpr`); it evaluates to the raw `Broadcast`.
#[derive(Clone)]
pub struct PBroadcast<A, L> {
    value: A,
    lanes: L,
}

impl<A: Pattern, L: Pattern> Pattern for PBroadcast<A, L> {
    fn eval(&self) -> Result<Expr> {
        Ok(Broadcast::new(self.value.eval()?, self.lanes.eval()?)?.into())
    }

    fn reset(&self) {
        self.value.reset();
        self.lanes.reset();
    }

    fn match_node(&self, expr: &Expr) -> bool {
        expr.as_node::<BroadcastObj>().is_some_and(|broadcast| {
            self.value.match_node(&broadcast.value) && self.lanes.match_node(&broadcast.lanes)
        })
    }
}

/// `ramp(base, stride, lanes)` (C++ `PRampExpr`); integer literals are typed like `base`, and it
/// evaluates to the raw `Ramp`.
#[derive(Clone)]
pub struct PRamp<B, S, L> {
    base: B,
    stride: S,
    lanes: L,
}

impl<B: Pattern, S: Pattern, L: Pattern> Pattern for PRamp<B, S, L> {
    fn eval(&self) -> Result<Expr> {
        let base = self.base.eval()?;
        let (stride, lanes) = (self.stride.eval_like(&base)?, self.lanes.eval_like(&base)?);
        Ok(Ramp::new(base, stride, lanes)?.into())
    }

    fn reset(&self) {
        self.base.reset();
        self.stride.reset();
        self.lanes.reset();
    }

    fn match_node(&self, expr: &Expr) -> bool {
        expr.as_node::<RampObj>().is_some_and(|ramp| {
            self.base.match_node(&ramp.base_)
                && self.stride.match_node(&ramp.stride)
                && self.lanes.match_node(&ramp.lanes)
        })
    }
}

/// `broadcast(value, lanes)`.
pub fn broadcast<A: IntoPattern, L: IntoPattern>(
    value: A,
    lanes: L,
) -> PBroadcast<A::Pattern, L::Pattern> {
    PBroadcast { value: value.into_pattern(), lanes: lanes.into_pattern() }
}

/// `ramp(base, stride, lanes)`.
pub fn ramp<B: IntoPattern, S: IntoPattern, L: IntoPattern>(
    base: B,
    stride: S,
    lanes: L,
) -> PRamp<B::Pattern, S::Pattern, L::Pattern> {
    PRamp { base: base.into_pattern(), stride: stride.into_pattern(), lanes: lanes.into_pattern() }
}

// ---------------------------------------------------------------------------------------------
// Building patterns

/// The binary patterns as methods, named after the C++ functions as [`super::PrimOps`] names
/// them: `x.floordiv(&c)` is C++ `floordiv(x, c)`.
pub trait PatternOps: Pattern + Sized {
    fn floordiv<R: IntoPattern>(self, rhs: R) -> PBinary<FloorDiv, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn floormod<R: IntoPattern>(self, rhs: R) -> PBinary<FloorMod, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn truncdiv<R: IntoPattern>(self, rhs: R) -> PBinary<Div, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn truncmod<R: IntoPattern>(self, rhs: R) -> PBinary<Mod, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn min<R: IntoPattern>(self, rhs: R) -> PBinary<Min, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn max<R: IntoPattern>(self, rhs: R) -> PBinary<Max, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn equal<R: IntoPattern>(self, rhs: R) -> PBinary<EQ, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn not_equal<R: IntoPattern>(self, rhs: R) -> PBinary<NE, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn less<R: IntoPattern>(self, rhs: R) -> PBinary<LT, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn less_equal<R: IntoPattern>(self, rhs: R) -> PBinary<LE, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn greater<R: IntoPattern>(self, rhs: R) -> PBinary<GT, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn greater_equal<R: IntoPattern>(self, rhs: R) -> PBinary<GE, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn logical_and<R: IntoPattern>(self, rhs: R) -> PBinary<And, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
    fn logical_or<R: IntoPattern>(self, rhs: R) -> PBinary<Or, Self, R::Pattern> {
        PBinary::new(self, rhs.into_pattern())
    }
}

impl<P: Pattern> PatternOps for P {}

/// `IntoPattern` for the composite patterns themselves, and the operators `+`, `-`, `*` and `!`
/// on every pattern.
macro_rules! composite_patterns {
    ($([$($generics:tt)*] $pattern:ty;)*) => {$(
        impl<$($generics)*> IntoPattern for $pattern where $pattern: Pattern {
            type Pattern = Self;
            fn into_pattern(self) -> Self {
                self
            }
        }

        impl<$($generics)* R: IntoPattern> std::ops::Add<R> for $pattern {
            type Output = PBinary<Add, Self, R::Pattern>;
            fn add(self, rhs: R) -> Self::Output {
                PBinary::new(self, rhs.into_pattern())
            }
        }

        impl<$($generics)* R: IntoPattern> std::ops::Sub<R> for $pattern {
            type Output = PBinary<Sub, Self, R::Pattern>;
            fn sub(self, rhs: R) -> Self::Output {
                PBinary::new(self, rhs.into_pattern())
            }
        }

        impl<$($generics)* R: IntoPattern> std::ops::Mul<R> for $pattern {
            type Output = PBinary<Mul, Self, R::Pattern>;
            fn mul(self, rhs: R) -> Self::Output {
                PBinary::new(self, rhs.into_pattern())
            }
        }

        impl<$($generics)*> std::ops::Not for $pattern {
            type Output = PNot<Self>;
            fn not(self) -> PNot<Self> {
                PNot(self)
            }
        }
    )*};
}

composite_patterns! {
    [N, A, B,] PBinary<N, A, B>;
    [A,] PNot<A>;
    [A, L,] PBroadcast<A, L>;
    [B, S, L,] PRamp<B, S, L>;
}

/// The operators on a reference to a variable, which `IntoPattern` already covers.
macro_rules! variable_operators {
    ($($trait:ident :: $method:ident => $node:ident),* $(,)?) => {$(
        impl<'a, T: Bindable, R: IntoPattern> std::ops::$trait<R> for &'a PVar<T> {
            type Output = PBinary<$node, Self, R::Pattern>;
            fn $method(self, rhs: R) -> Self::Output {
                PBinary::new(self, rhs.into_pattern())
            }
        }
    )*};
}

variable_operators!(Add::add => Add, Sub::sub => Sub, Mul::mul => Mul);

impl<T: Bindable> std::ops::Not for &PVar<T> {
    type Output = PNot<Self>;
    fn not(self) -> PNot<Self> {
        PNot(self)
    }
}
