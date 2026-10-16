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

//! Building IR with the semantics of the C++ API: the `PrimExpr` operators
//! (`include/tvm/ir/prim/op.h`) and the node constructors.
//!
//! Operators (`arith`). `a + b` is C++ `a + b` (`add` in `src/ir/prim/op.cc`), written in Rust:
//! the operand types are matched (`BinaryOpMatchTypes`), constants are folded as
//! `TryConstFold` folds them (`fold`, after `src/ir/prim/const_fold.h`), with the same checks and
//! errors, and anything else is the raw node. A literal on either side is typed the
//! way C++ types it: an integer becomes `MakeConst(other.ty(), v)`, an `f32` a float32 `FloatImm`,
//! an `f64` a float64 `FloatImm`. C++ has no `PrimExpr(bool)`, so there is no `bool` literal. As in
//! C++, `/` and `%` are not defined: name the division (`truncdiv`, `floordiv`, `div`, ...).
//! Comparisons and `&&` / `||` are [`PrimOps`] methods named after the C++ functions, because
//! Rust's comparison traits return `bool`; so is `cast`, the folding C++ `cast()` (`Cast::new` is
//! the raw node).
//!
//! Math (`math`). `cast`, the math functions (`floor`, `abs`, `isinf`, `pow`, `exp`, ...),
//! `reinterpret`, and the type matching of binary operations ([`match_types`]) are written in
//! Rust after their C++ definitions, with the same folding, checks and errors. Operators build an
//! [`Ex`], which keeps the first error instead of panicking; `.expr()` turns it back into a
//! `Result`:
//!
//! ```ignore
//! select(rmod.greater_equal(0), &rdiv, &rdiv - 1).expr()?
//! ```
//!
//! Constructors (`build`). `X::new` takes what the C++ constructor has no default for: the
//! location is `UnknownLoc()` and trailing optional fields take their C++ defaults. `X::new_full`
//! takes every argument. Arguments convert the way C++ converts them implicitly: an `i32` is an
//! `IntImm::Int32` ([`IntoExpr`], so `Evaluate::new(0)`), a statement is a body ([`IntoBody`]).
//! Each is written in Rust after the C++ constructor, with its checks, normalization, inferred
//! type and errors; the raw expression constructors (C++ `Add(a, b)`: type checks, no folding)
//! are `prim::Add::new` and so on. A constructor that dispatches through a registry (`Call`'s
//! `FInferType`, `Evaluate`'s and `TensorStore`'s validation, `RegionStmt`'s contract) reads it
//! and calls the registered hook. What C++ keeps as state stays C++'s: the `UnknownLoc()` and
//! `MissingType()` singletons and the `PrimType` node of each dtype are fetched once per thread,
//! and `ExecScope`, `TileLayout`, `DispatchContext` and `UniqueNameSupply` call the native
//! constructor. The generated lossless allocations are `from_complete_fields`: use them for what
//! C++ builds by `CopyOnWrite` or `make_object<T>(*node)`.
//!
//! Native functions (`native`). Printing, structural comparison, the target context, layouts and
//! analyses that Rust calls rather than writes again, as typed functions and methods.

mod arith;
mod build;
mod fold;
mod math;
mod native;
pub mod pattern;

pub use arith::*;
pub use build::*;
pub use math::{fast_erf_float_expr, infinity, match_types};
pub use native::{deep_equal, repr, structural_equal, structural_hash, undefined_vars, TargetScope};
