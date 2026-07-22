// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Compile-pass coverage for the actual output of the procedural macro.

#![deny(warnings)]

extern crate self as tvm_tirx;

use std::marker::PhantomData;
use tvm_tirx_macros::dispatch;

pub mod visit {
    use super::PhantomData;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum WalkResult {
        Advance,
        Skip,
    }

    pub struct VisitValue;

    impl VisitValue {
        pub fn cast<T>(&self) -> Option<T> {
            None
        }
    }

    pub struct VisitCtx<'ctx, V> {
        marker: PhantomData<&'ctx mut V>,
    }

    impl<V> VisitCtx<'_, V> {
        pub(crate) fn new() -> Self {
            Self {
                marker: PhantomData,
            }
        }
    }

    pub trait VisitDispatch: Sized {
        fn dispatch_visit(
            &mut self,
            value: &VisitValue,
            ctx: &mut VisitCtx<'_, Self>,
        ) -> Option<WalkResult>;
    }
}

pub mod renamed_runtime {
    pub mod visit {
        pub use crate::visit::*;
    }
}

struct For;
struct Stmt;

#[deprecated]
struct DeprecatedNode;

struct DefaultRuntimeAnalyzer;

#[dispatch(visit)]
impl DefaultRuntimeAnalyzer {
    // `inline` is valid on the method, but invalid if copied to the generated
    // `if` expression.  This is the regression that token-only tests missed.
    #[cfg_attr(all(), inline)]
    fn visit_for(&mut self, _node: For, _ctx: &mut visit::VisitCtx<'_, Self>) -> visit::WalkResult {
        visit::WalkResult::Advance
    }

    // The generated dispatch arm must be disabled under exactly the same
    // condition as this method.  MissingNode deliberately does not exist.
    #[cfg_attr(all(), cfg(any()), inline)]
    fn visit_disabled(
        &mut self,
        _node: MissingNode,
        _ctx: &mut visit::VisitCtx<'_, Self>,
    ) -> visit::WalkResult {
        visit::WalkResult::Advance
    }
}

struct RenamedRuntimeAnalyzer;

#[dispatch(visit, runtime = crate::renamed_runtime)]
impl RenamedRuntimeAnalyzer {
    fn visit_stmt(
        &mut self,
        _node: Stmt,
        _ctx: &mut visit::VisitCtx<'_, Self>,
    ) -> visit::WalkResult {
        visit::WalkResult::Skip
    }
}

type HandlerContext<'borrow, 'ctx, V> = &'borrow mut visit::VisitCtx<'ctx, V>;
type HandlerResult = visit::WalkResult;

struct AliasSignatureAnalyzer;

#[dispatch(visit)]
impl AliasSignatureAnalyzer {
    fn visit_stmt(&mut self, _node: Stmt, _ctx: HandlerContext<'_, '_, Self>) -> HandlerResult {
        visit::WalkResult::Advance
    }
}

struct GenericAnalyzer<'a, T, const N: usize> {
    marker: PhantomData<&'a T>,
}

#[dispatch(visit)]
impl<'a, T, const N: usize> GenericAnalyzer<'a, T, N>
where
    T: Clone + 'a,
{
    fn visit_stmt(
        &mut self,
        _node: Stmt,
        _ctx: &mut visit::VisitCtx<'_, Self>,
    ) -> visit::WalkResult {
        visit::WalkResult::Advance
    }
}

struct TrailingCommaAnalyzer;

#[dispatch(visit)]
impl TrailingCommaAnalyzer {
    fn visit_stmt(
        &mut self,
        _node: Stmt,
        _ctx: &mut visit::VisitCtx<'_, Self>,
    ) -> visit::WalkResult {
        visit::WalkResult::Advance
    }
}

// Signature errors belonging to a disabled method must be disabled too.  This
// method deliberately violates every important part of the handler contract.
struct DisabledInvalidAnalyzer;

#[dispatch(visit)]
impl DisabledInvalidAnalyzer {
    #[cfg(any())]
    fn visit_missing<T>(&self, _node: MissingNode, _ctx: ())
    where
        T: Clone,
    {
    }
}

// An impl-level allow applies to the copied cfg predicate and its gated
// signature diagnostic.  The feature is deliberately absent from Cargo.toml,
// so losing the allow makes this fail under the crate-level `deny(warnings)`.
struct UnknownCfgAnalyzer;

#[dispatch(visit)]
#[allow(unexpected_cfgs)]
impl UnknownCfgAnalyzer {
    #[cfg(feature = "dispatch-undeclared-feature")]
    fn visit_missing<T>(&self, _node: MissingNode, _ctx: ())
    where
        T: Clone,
    {
    }
}

// Lint scopes can live on the handler itself or on its enclosing impl.  The
// generated arm repeats DeprecatedNode, so both scopes are needed there too;
// `inline` remains intentionally absent from the generated expression.
struct MethodLintAnalyzer;

#[dispatch(visit)]
impl MethodLintAnalyzer {
    #[cfg_attr(all(), allow(deprecated), inline)]
    fn visit_deprecated(
        &mut self,
        _node: DeprecatedNode,
        _ctx: &mut visit::VisitCtx<'_, Self>,
    ) -> visit::WalkResult {
        visit::WalkResult::Advance
    }
}

struct ImplLintAnalyzer;

#[dispatch(visit)]
#[allow(deprecated)]
impl ImplLintAnalyzer {
    fn visit_deprecated(
        &mut self,
        _node: DeprecatedNode,
        _ctx: &mut visit::VisitCtx<'_, Self>,
    ) -> visit::WalkResult {
        visit::WalkResult::Advance
    }
}

// The source expectation must remain a real expectation, while generated code
// inherits it as `allow` so it does not create a second unfulfilled one.
struct ExpectLintAnalyzer;

#[dispatch(visit)]
impl ExpectLintAnalyzer {
    #[expect(unused_variables)]
    fn visit_stmt(&mut self, node: Stmt, ctx: &mut visit::VisitCtx<'_, Self>) -> visit::WalkResult {
        visit::WalkResult::Advance
    }
}

// The generated trait impl must carry the same impl-level configuration as
// the inherent impl.  MissingNode deliberately does not exist.
struct ImplDisabledAnalyzer;

#[dispatch(visit)]
#[cfg(any())]
impl ImplDisabledAnalyzer {
    fn visit_missing(
        &mut self,
        _node: MissingNode,
        _ctx: &mut visit::VisitCtx<'_, Self>,
    ) -> visit::WalkResult {
        visit::WalkResult::Advance
    }
}

#[test]
fn default_runtime_path_compiles() {
    let mut analyzer = DefaultRuntimeAnalyzer;
    let mut ctx = visit::VisitCtx::new();
    assert_eq!(
        visit::VisitDispatch::dispatch_visit(&mut analyzer, &visit::VisitValue, &mut ctx),
        None
    );
}

#[test]
fn runtime_path_override_compiles() {
    let mut analyzer = RenamedRuntimeAnalyzer;
    let mut ctx = visit::VisitCtx::new();
    assert_eq!(
        visit::VisitDispatch::dispatch_visit(&mut analyzer, &visit::VisitValue, &mut ctx),
        None
    );
}

#[test]
fn signature_aliases_compile() {
    let mut analyzer = AliasSignatureAnalyzer;
    let mut ctx = visit::VisitCtx::new();
    assert_eq!(
        visit::VisitDispatch::dispatch_visit(&mut analyzer, &visit::VisitValue, &mut ctx),
        None
    );
    let _impl_disabled = ImplDisabledAnalyzer;
    assert_generated_dispatch::<TrailingCommaAnalyzer>();
    assert_generated_dispatch::<DisabledInvalidAnalyzer>();
    assert_generated_dispatch::<UnknownCfgAnalyzer>();
    assert_generated_dispatch::<MethodLintAnalyzer>();
    assert_generated_dispatch::<ImplLintAnalyzer>();
    assert_generated_dispatch::<ExpectLintAnalyzer>();
    assert_generated_dispatch::<GenericAnalyzer<'static, String, 3>>();
    let generic = GenericAnalyzer::<String, 3> {
        marker: PhantomData,
    };
    let _ = generic.marker;
}

fn assert_generated_dispatch<T: visit::VisitDispatch>() {}
