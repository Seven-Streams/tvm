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

//! What a node type defined in Rust needs from this crate to extend a C++ IR node.
//!
//! tvm-ffi's `#[derive(Object)]` registers the reflection of a `#[type_register]` type with
//! `#[reflect]` fields (C++ `refl::ObjectDef<T>`), so its fields reach the C++ structural
//! engines, the TVMScript printer and Python. It copies the parent through
//! [`tvm_ffi::reflection::ShallowCopy`], which tvm-ffi implements only for `Object`; this module
//! implements it for the C++ bases a Rust node extends, and provides constructors for those
//! bases, whose `new` is crate-private generated code.
//!
//! ```ignore
//! #[repr(C)]
//! #[derive(tvm_ffi::derive::Object)]
//! #[type_key = "my.PairType"]
//! #[type_register]
//! #[type_final]
//! pub struct PairTypeObj {
//!     base: TypeObj, // kind: inherited from `ir.Type`, TreeNode
//!     #[reflect]
//!     pub lhs: Type,
//!     #[reflect(seq_hash_ignore)]
//!     pub note: tvm_ffi::String,
//! }
//! ```

use tvm_ffi::reflection::ShallowCopy;

use crate::ir::{AttrsObj, ExprObj, Location, StmtObj, Type, TypeObj};

impl ShallowCopy for TypeObj {
    fn shallow_copy(&self) -> Self {
        TypeObj::new(self.loc.clone())
    }
}

impl ShallowCopy for ExprObj {
    fn shallow_copy(&self) -> Self {
        ExprObj::new(self.loc.clone(), self.ty.clone())
    }
}

impl ShallowCopy for StmtObj {
    fn shallow_copy(&self) -> Self {
        StmtObj::new(self.loc.clone())
    }
}

impl ShallowCopy for AttrsObj {
    fn shallow_copy(&self) -> Self {
        AttrsObj::new()
    }
}

/// The `ir.Type` base of a Rust-defined type, at `loc` (the C++ `TypeNode` constructor).
pub fn type_base(loc: Location) -> TypeObj {
    TypeObj::new(loc)
}

/// The `ir.Expr` base of a Rust-defined expression, at `loc` and of type `ty`.
pub fn expr_base(loc: Location, ty: Type) -> ExprObj {
    ExprObj::new(loc, ty)
}

/// The `ir.Stmt` base of a Rust-defined statement, at `loc`.
pub fn stmt_base(loc: Location) -> StmtObj {
    StmtObj::new(loc)
}

/// The `ir.Attrs` base of a Rust-defined attribute node.
pub fn attrs_base() -> AttrsObj {
    AttrsObj::new()
}
