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

//! `tirx.ConvertSSA` (`src/tirx/transform/ir_utils.cc`): give every variable definition in the
//! module a distinct `Var`, renaming redefinitions and their uses.
//!
//! Every occurrence of a variable reaches `mutate_var`, which tells definitions from uses by the
//! definition region the engine reports. Scopes that end a definition's reach (`Let`, loops,
//! branch arms, region bodies) push a lexical scope of renames.

use std::collections::{HashMap, HashSet};

use tvm_ffi::{
    dispatch, structural_mutate, structural_walk, Any, Array, DefRegionKind, Mutator,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, WalkOrder, WalkResult,
};

use super::util::{module_pass, same_option, unknown_loc, MutatorExt};
use crate::ir::{
    Bind, BaseFunc, Expr, For, FuncType, GlobalVar, If, IRModule, RegionStmt, SeqStmt, Type, Var,
    While,
};
use crate::prim::Let;
use crate::tirx::Function;
use crate::transform::Pass;

pub(crate) struct ConvertSsa {
    /// Lexical renames, innermost scope last.
    scopes: Vec<HashMap<ObjectIdentity, Var>>,
    /// Definitions seen anywhere in the module; a second definition gets a fresh `Var`.
    defined: HashSet<ObjectIdentity>,
}

#[dispatch(mutate)]
impl ConvertSsa {
    /// A use takes the innermost rename; a definition is renamed if the variable was defined
    /// before. A pattern definition of a variable already in scope is a use.
    fn mutate_var(&mut self, var: Var, m: &mut Mutator) -> Result<Var> {
        match (m.def_region_kind(), self.renamed(&var)) {
            (DefRegionKind::None, renamed) => Ok(renamed.unwrap_or(var)),
            (DefRegionKind::Pattern, Some(renamed)) => Ok(renamed),
            _ => {
                let ty: Type = m.mutate_with(self, &var.ty, DefRegionKind::None)?.try_into()?;
                Ok(self.define(&var, ty))
            }
        }
    }

    fn mutate_let(&mut self, op: Let, m: &mut Mutator) -> Result<Any> {
        self.scoped(|this| m.default_mutate(this, &op))
    }

    fn mutate_for(&mut self, op: For, m: &mut Mutator) -> Result<Any> {
        self.scoped(|this| m.default_mutate(this, &op))
    }

    fn mutate_while(&mut self, op: While, m: &mut Mutator) -> Result<Any> {
        self.scoped(|this| m.default_mutate(this, &op))
    }

    /// A bound variable stays defined for the rest of the enclosing scope.
    fn mutate_bind(&mut self, op: Bind, m: &mut Mutator) -> Result<Bind> {
        let stmt: Bind = m.default_mutate_as(self, &op)?;
        if !self.renamed(&op.var).is_some_and(|renamed| renamed.same_as(&stmt.var)) {
            self.rename(&op.var, stmt.var.clone());
        }
        Ok(stmt)
    }

    /// Each arm is a scope of its own.
    fn mutate_if(&mut self, op: If, m: &mut Mutator) -> Result<If> {
        let condition: Expr = m.mutate_as(self, &op.condition)?;
        let then_case: SeqStmt = self.scoped(|this| m.mutate_as(this, &op.then_case))?;
        let else_case: Option<SeqStmt> = op
            .else_case
            .as_ref()
            .map(|arm| self.scoped(|this| m.mutate_as(this, arm)))
            .transpose()?;
        if condition.same_as(&op.condition)
            && then_case.same_as(&op.then_case)
            && same_option(&else_case, &op.else_case)
        {
            return Ok(op);
        }
        Ok(If::new(unknown_loc()?, condition, then_case, else_case))
    }

    /// Body parameters are defined for the body; results after the region.
    fn mutate_region(&mut self, op: RegionStmt, m: &mut Mutator) -> Result<RegionStmt> {
        let attrs = m.mutate_as(self, &op.attrs)?;
        let args: Array<Expr> = m.mutate_as(self, &op.args)?;
        let (params, body) = self.scoped(|this| {
            let params: Array<Var> =
                m.mutate_with(this, &op.body_params, DefRegionKind::Simple)?.try_into()?;
            let body: SeqStmt = m.mutate_as(this, &op.body)?;
            Ok((params, body))
        })?;
        let results: Array<Var> =
            m.mutate_with(self, &op.result_vars, DefRegionKind::Simple)?.try_into()?;
        Ok(op.with(|node| {
            node.args = args;
            node.body_params = params;
            node.attrs = attrs;
            node.body = body;
            node.result_vars = results;
        }))
    }
}

impl ConvertSsa {
    /// A converter with the bottom scope C++ `ScopeStack` starts with.
    pub(crate) fn new() -> Self {
        Self { scopes: vec![HashMap::new()], defined: HashSet::new() }
    }

    fn scoped<T>(&mut self, body: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.scopes.push(HashMap::new());
        let result = body(self);
        self.scopes.pop();
        result
    }

    fn renamed(&self, var: &Var) -> Option<Var> {
        let id = ObjectIdentity::of(var);
        self.scopes.iter().rev().find_map(|scope| scope.get(&id).cloned())
    }

    fn rename(&mut self, old: &Var, new: Var) {
        self.scopes.last_mut().expect("inside a scope").insert(ObjectIdentity::of(old), new);
    }

    /// Define `var` with its type mutated to `ty`: a fresh `Var` if it was defined before.
    fn define(&mut self, var: &Var, ty: Type) -> Var {
        let first = self.defined.insert(ObjectIdentity::of(var));
        let defined = match (first, ty.same_as(&var.ty)) {
            (true, true) => var.clone(),
            _ => Var::new(var.loc.clone(), ty, var.name.clone()),
        };
        self.rename(var, defined.clone());
        defined
    }

    /// `VisitFunction`: parameters, then the symbolic shapes in their types, are definitions;
    /// then the body.
    fn visit_function(&mut self, func: Function) -> Result<Function> {
        self.scoped(|this| {
            let mut defined: HashSet<ObjectIdentity> = HashSet::new();
            for param in func.params.iter() {
                let ty: Type = structural_mutate(param.ty.clone(), &mut *this)?.try_into()?;
                this.define(&param, ty);
                defined.insert(ObjectIdentity::of(&param));
            }
            for param in func.params.iter() {
                let mut symbols: Vec<Var> = Vec::new();
                structural_walk(
                    &param.ty,
                    |var: Var| {
                        symbols.push(var);
                        WalkResult::Advance
                    },
                    WalkOrder::PostOrder,
                )?;
                for var in symbols {
                    if defined.insert(ObjectIdentity::of(&var)) {
                        let ty: Type = structural_mutate(var.ty.clone(), &mut *this)?.try_into()?;
                        this.define(&var, ty);
                    }
                }
            }
            let mut params: Vec<Var> = Vec::new();
            for var in func.params.iter() {
                let mut mapped = this.renamed(&var).unwrap_or_else(|| var.clone());
                let ty: Type = structural_mutate(var.ty.clone(), &mut *this)?.try_into()?;
                if !ty.same_as(&mapped.ty) {
                    mapped = Var::new(mapped.loc.clone(), ty, mapped.name.clone());
                    this.rename(&var, mapped.clone());
                }
                params.push(mapped);
            }
            let params_same =
                params.iter().zip(func.params.iter()).all(|(new, old)| new.same_as(&old));
            let body: Option<SeqStmt> = match &func.body {
                Some(body) => Some(structural_mutate(body.clone(), &mut *this)?.try_into()?),
                None => None,
            };
            if params_same && same_option(&body, &func.body) {
                return Ok(func);
            }
            let params = Array::new(params);
            let ty = FuncType::new(
                unknown_loc()?,
                params.iter().map(|param| param.ty.clone()).collect::<Array<Type>>(),
                func.ret_type.clone(),
            );
            Ok(Function::new(
                func.loc.clone(),
                ty.into(),
                func.attrs.clone(),
                params,
                func.ret_type.clone(),
                body,
            ))
        })
    }
}

pub fn convert_ssa() -> Result<Pass> {
    module_pass("tirx.ConvertSSA", 0, |module, _context| {
        let mut converter = ConvertSsa::new();
        let mut made_change = false;
        let mut functions: Vec<(GlobalVar, BaseFunc)> = Vec::new();
        for (gvar, base_func) in module.functions.iter() {
            let mut base_func = base_func;
            if let Ok(func) = base_func.clone().try_cast::<Function>() {
                let updated = converter.visit_function(func)?;
                if !updated.same_as(&base_func) {
                    made_change = true;
                    base_func = updated.into();
                }
            }
            functions.push((gvar, base_func));
        }
        if !made_change {
            return Ok(module);
        }
        Ok(IRModule::new(
            functions.into_iter().collect(),
            module.source_map.clone(),
            module.attrs.clone(),
            module.global_infos.clone(),
            module.global_var_map.clone(),
        ))
    })
}

/// `tirx::ConvertSSA(Stmt)`: a fresh converter run over one statement.
pub(crate) fn convert_ssa_stmt(stmt: SeqStmt) -> Result<SeqStmt> {
    structural_mutate(stmt, &mut ConvertSsa::new())?.try_into()
}
