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

//! Code generation for the typed dispatch layer of the tirx-ext prototype.
//!
//! The first deliberately narrow family is `#[dispatch(visit)]`.  It scans
//! the annotated inherent `impl`, in source order, and turns methods named
//! `visit_*` into a `VisitDispatch` implementation.  It does not
//! generate recursion: the runtime walker and `VisitCtx` remain responsible
//! for visiting children.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Attribute, FnArg, Ident, ImplItem, ImplItemMethod, ItemImpl, Meta,
    NestedMeta, Path, ReturnType, Token, Type,
};

struct DispatchArgs {
    mode: Ident,
    runtime: Path,
}

impl Parse for DispatchArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mode = input.parse()?;
        let mut runtime = syn::parse_quote!(::tvm_tirx);

        if !input.is_empty() {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                return Ok(Self { mode, runtime });
            }
            let option: Ident = input.parse()?;
            if option != "runtime" {
                return Err(syn::Error::new(
                    option.span(),
                    "unknown dispatch option; expected `runtime = path`",
                ));
            }
            input.parse::<Token![=]>()?;
            runtime = input.parse()?;
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        if !input.is_empty() {
            return Err(input.error("unexpected tokens after the dispatch options"));
        }

        Ok(Self { mode, runtime })
    }
}

/// Generate a typed dispatch implementation from one inherent `impl` block.
///
/// The supported form is:
///
/// ```ignore
/// #[dispatch(visit)]
/// impl Analyzer {
///     fn visit_for(
///         &mut self,
///         node: For,
///         ctx: &mut VisitCtx<'_, Self>,
///     ) -> WalkResult {
///         // ...
///     }
/// }
/// ```
///
/// Methods are tried in declaration order, so the first successful runtime
/// cast wins.  Only the annotated block is visible to the macro.  Generated
/// paths use `::tvm_tirx` by default.  A renamed dependency can be selected
/// with `#[dispatch(visit, runtime = my_tvm_tirx)]`.
#[proc_macro_attribute]
pub fn dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as DispatchArgs);
    let item_impl = parse_macro_input!(item as ItemImpl);

    match expand_dispatch(&args.mode, &args.runtime, &item_impl) {
        Ok(generated) => quote!(#item_impl #generated).into(),
        Err(error) => {
            let error = error.to_compile_error();
            quote!(#item_impl #error).into()
        }
    }
}

fn expand_dispatch(
    mode: &Ident,
    runtime: &Path,
    item_impl: &ItemImpl,
) -> syn::Result<TokenStream2> {
    if mode != "visit" {
        return Err(syn::Error::new(
            mode.span(),
            "unsupported dispatch family; the prototype currently supports only `visit`",
        ));
    }
    if item_impl.trait_.is_some() {
        return Err(syn::Error::new_spanned(
            item_impl,
            "#[dispatch(visit)] must be attached to an inherent impl",
        ));
    }
    let impl_generated_attrs =
        project_attributes(&item_impl.attrs, AttributeProjection::Generated)?;

    let mut handlers = Vec::new();
    let mut gated_diagnostics = Vec::new();
    let mut found_handler = false;
    for item in &item_impl.items {
        let ImplItem::Method(method) = item else {
            continue;
        };
        if method.sig.ident.to_string().starts_with("visit_") {
            found_handler = true;
            let condition_attrs =
                project_attributes(&method.attrs, AttributeProjection::ConditionsOnly)?;
            let lint_attrs = project_attributes(&method.attrs, AttributeProjection::LintsOnly)?;
            match parse_visit_method(method) {
                Ok(mut handler) => {
                    handler.condition_attrs = condition_attrs;
                    handler.lint_attrs = lint_attrs;
                    handlers.push(handler);
                }
                Err(error) => {
                    if condition_attrs.is_empty() {
                        return Err(error);
                    }
                    let error = error.to_compile_error();
                    let impl_attrs = &impl_generated_attrs;
                    gated_diagnostics.push(quote! {
                        #(#impl_attrs)*
                        const _: () = {
                            #(#lint_attrs)*
                            const _: () = {
                                #(#condition_attrs)*
                                const _: () = {
                                    #error
                                };
                            };
                        };
                    });
                }
            }
        }
    }
    if !found_handler {
        return Err(syn::Error::new_spanned(
            item_impl,
            "#[dispatch(visit)] found no methods whose names start with `visit_`",
        ));
    }

    let dispatch_arms = handlers.iter().map(|handler| {
        let method = &handler.method;
        let node_type = &handler.node_type;
        let condition_attrs = &handler.condition_attrs;
        let lint_attrs = &handler.lint_attrs;
        let dispatch = quote! {
            #(#condition_attrs)*
            if let Some(node) = value.cast::<#node_type>() {
                let result: #runtime::visit::WalkResult = self.#method(node, ctx);
                return Some(result);
            }
        };
        if lint_attrs.is_empty() {
            dispatch
        } else {
            quote! {
                #(#lint_attrs)*
                {
                    #dispatch
                }
            }
        }
    });
    let self_type = &item_impl.self_ty;
    let (impl_generics, _, where_clause) = item_impl.generics.split_for_impl();

    Ok(quote! {
        #(#gated_diagnostics)*
        #(#impl_generated_attrs)*
        impl #impl_generics #runtime::visit::VisitDispatch for #self_type #where_clause {
            fn dispatch_visit(
                &mut self,
                value: &#runtime::visit::VisitValue,
                ctx: &mut #runtime::visit::VisitCtx<'_, Self>,
            ) -> Option<#runtime::visit::WalkResult> {
                #(#dispatch_arms)*
                None
            }
        }
    })
}

struct VisitHandler {
    method: Ident,
    node_type: Type,
    condition_attrs: Vec<TokenStream2>,
    lint_attrs: Vec<TokenStream2>,
}

fn parse_visit_method(method: &ImplItemMethod) -> syn::Result<VisitHandler> {
    let sig = &method.sig;
    if sig.constness.is_some()
        || sig.asyncness.is_some()
        || sig.unsafety.is_some()
        || sig.abi.is_some()
        || sig.variadic.is_some()
        || !sig.generics.params.is_empty()
        || sig.generics.where_clause.is_some()
    {
        return Err(syn::Error::new_spanned(
            sig,
            "visit handlers must be ordinary, non-generic Rust methods",
        ));
    }

    let mut inputs = sig.inputs.iter();
    match inputs.next() {
        Some(FnArg::Receiver(receiver))
            if receiver.reference.is_some() && receiver.mutability.is_some() => {}
        _ => {
            return Err(syn::Error::new_spanned(
                sig,
                "visit handlers must start with `&mut self`",
            ));
        }
    }

    let node_type = match inputs.next() {
        Some(FnArg::Typed(node)) => (*node.ty).clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                sig,
                "visit handlers need a typed node argument after `&mut self`",
            ));
        }
    };
    if type_contains_impl_trait(&node_type) {
        return Err(syn::Error::new_spanned(
            &node_type,
            "visit handlers cannot use `impl Trait`; use a concrete node wrapper type",
        ));
    }
    if matches!(node_type, Type::Reference(_)) {
        return Err(syn::Error::new_spanned(
            &node_type,
            "the visit node argument must be an owned TVM object-reference wrapper",
        ));
    }

    let ctx_type = match (inputs.next(), inputs.next()) {
        (Some(FnArg::Typed(ctx)), None) => &*ctx.ty,
        _ => {
            return Err(syn::Error::new_spanned(
                sig,
                "visit handlers must have exactly `&mut self`, node, and VisitCtx arguments",
            ));
        }
    };
    validate_visit_ctx(ctx_type)?;
    validate_walk_result(&sig.output)?;

    Ok(VisitHandler {
        method: sig.ident.clone(),
        node_type,
        condition_attrs: Vec::new(),
        lint_attrs: Vec::new(),
    })
}

fn type_contains_impl_trait(ty: &Type) -> bool {
    struct ImplTraitFinder {
        found: bool,
    }

    impl<'ast> syn::visit::Visit<'ast> for ImplTraitFinder {
        fn visit_type_impl_trait(&mut self, _node: &'ast syn::TypeImplTrait) {
            self.found = true;
        }
    }

    let mut finder = ImplTraitFinder { found: false };
    syn::visit::Visit::visit_type(&mut finder, ty);
    finder.found
}

/// Conservatively diagnose signatures that are unambiguously not a mutable
/// VisitCtx reference.  An opaque path is accepted because it may be a type
/// alias for `&mut VisitCtx`; the generated impl remains the final type check.
fn validate_visit_ctx(ctx_type: &Type) -> syn::Result<()> {
    if type_contains_impl_trait(ctx_type) {
        return Err(syn::Error::new_spanned(
            ctx_type,
            "the VisitCtx argument cannot use `impl Trait`; use `&mut VisitCtx<...>`",
        ));
    }
    match ctx_type {
        Type::Paren(parenthesized) => validate_visit_ctx(&parenthesized.elem),
        Type::Group(group) => validate_visit_ctx(&group.elem),
        Type::Reference(reference) if reference.mutability.is_none() => {
            Err(syn::Error::new_spanned(
                ctx_type,
                "the VisitCtx argument must be a mutable reference",
            ))
        }
        Type::Path(path) if matches!(path.path.segments.last(), Some(segment) if segment.ident == "VisitCtx") => {
            Err(syn::Error::new_spanned(
                ctx_type,
                "the VisitCtx argument must be passed as `&mut VisitCtx<...>`",
            ))
        }
        _ => Ok(()),
    }
}

/// Keep aliases and qualified paths valid, while producing an early error for
/// the most common accidental omission (`-> ()` or no return annotation).
fn validate_walk_result(output: &ReturnType) -> syn::Result<()> {
    let ReturnType::Type(_, return_type) = output else {
        return Err(syn::Error::new_spanned(
            output,
            "visit handlers must return WalkResult",
        ));
    };

    if type_contains_impl_trait(return_type) {
        return Err(syn::Error::new_spanned(
            return_type,
            "the visit result cannot use `impl Trait`; return WalkResult directly",
        ));
    }

    if matches!(&**return_type, Type::Tuple(tuple) if tuple.elems.is_empty()) {
        return Err(syn::Error::new_spanned(
            return_type,
            "visit handlers must return WalkResult, not `()`",
        ));
    }
    Ok(())
}

/// Attributes that are meaningful on generated dispatch code.  Conditions
/// keep an arm in lock-step with its method; lint levels preserve the source
/// scope around names and cfg predicates repeated by the expansion.  Other
/// attributes (notably `inline`) must not be copied onto an `if` expression.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AttributeProjection {
    ConditionsOnly,
    LintsOnly,
    Generated,
}

fn project_attributes(
    attrs: &[Attribute],
    projection: AttributeProjection,
) -> syn::Result<Vec<TokenStream2>> {
    attrs
        .iter()
        .filter_map(|attr| match project_attribute(attr, projection) {
            Ok(projected) => projected.map(Ok),
            Err(error) => Some(Err(error)),
        })
        .collect()
}

fn project_attribute(
    attr: &Attribute,
    projection: AttributeProjection,
) -> syn::Result<Option<TokenStream2>> {
    let supported = attr.path.is_ident("cfg")
        || attr.path.is_ident("cfg_attr")
        || (projection != AttributeProjection::ConditionsOnly && is_lint_level(&attr.path));
    if !supported {
        // Attribute macros may use token syntax that is not `syn::Meta`.
        // Ignore unrelated attributes without trying to parse their payload.
        return Ok(None);
    }
    project_meta(&attr.parse_meta()?, projection).map(|meta| meta.map(|meta| quote!(#[#meta])))
}

fn project_meta(meta: &Meta, projection: AttributeProjection) -> syn::Result<Option<TokenStream2>> {
    if meta.path().is_ident("cfg") {
        return Ok((projection != AttributeProjection::LintsOnly).then(|| quote!(#meta)));
    }
    if projection != AttributeProjection::ConditionsOnly && is_lint_level(meta.path()) {
        // `expect` on the source method remains responsible for checking that
        // its lint actually fires.  Repeating it would create a second,
        // usually-unfulfilled expectation on macro-generated code, so inherit
        // its suppression there as `allow` instead.
        if meta.path().is_ident("expect") {
            let mut inherited = meta.clone();
            match &mut inherited {
                Meta::Path(path) => *path = syn::parse_quote!(allow),
                Meta::List(list) => list.path = syn::parse_quote!(allow),
                Meta::NameValue(name_value) => name_value.path = syn::parse_quote!(allow),
            }
            return Ok(Some(quote!(#inherited)));
        }
        return Ok(Some(quote!(#meta)));
    }
    if !meta.path().is_ident("cfg_attr") {
        return Ok(None);
    }

    let Meta::List(list) = meta else {
        return Err(syn::Error::new_spanned(
            meta,
            "cfg_attr must contain a predicate and at least one attribute",
        ));
    };
    let mut nested = list.nested.iter();
    let Some(predicate) = nested.next() else {
        return Err(syn::Error::new_spanned(
            meta,
            "cfg_attr must contain a predicate and at least one attribute",
        ));
    };
    if list.nested.len() == 1 {
        return Err(syn::Error::new_spanned(
            meta,
            "cfg_attr must contain a predicate and at least one attribute",
        ));
    }

    let mut conditional = Vec::new();
    for attribute in nested {
        let NestedMeta::Meta(attribute) = attribute else {
            return Err(syn::Error::new_spanned(
                attribute,
                "cfg_attr entries must be attributes",
            ));
        };
        if let Some(projected) = project_meta(attribute, projection)? {
            conditional.push(projected);
        }
    }

    if conditional.is_empty() {
        Ok(None)
    } else {
        Ok(Some(quote!(cfg_attr(#predicate, #(#conditional),*))))
    }
}

fn is_lint_level(path: &Path) -> bool {
    path.is_ident("allow")
        || path.is_ident("warn")
        || path.is_ident("deny")
        || path.is_ident("forbid")
        || path.is_ident("expect")
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse_quote;

    fn expand(mode: &Ident, item: &ItemImpl) -> syn::Result<TokenStream2> {
        expand_dispatch(mode, &parse_quote!(::tvm_tirx), item)
    }

    #[test]
    fn accepts_a_trailing_comma_after_the_mode() {
        let args: DispatchArgs = syn::parse2(quote!(visit,)).unwrap();
        assert_eq!(args.mode, "visit");
        let runtime = &args.runtime;
        assert_eq!(quote!(#runtime).to_string(), quote!(::tvm_tirx).to_string());
    }

    #[test]
    fn preserves_handler_source_order() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                fn visit_for(&mut self, node: For, ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
                    todo!()
                }

                fn visit_stmt(&mut self, node: Stmt, ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
                    todo!()
                }
            }
        };
        let generated = expand(&parse_quote!(visit), &item).unwrap().to_string();
        let for_call = generated.find("visit_for").unwrap();
        let stmt_call = generated.find("visit_stmt").unwrap();
        assert!(for_call < stmt_call, "{generated}");
    }

    #[test]
    fn rejects_non_mutable_receiver() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                fn visit_for(&self, node: For, ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
                    todo!()
                }
            }
        };
        let error = expand(&parse_quote!(visit), &item).unwrap_err();
        assert!(error.to_string().contains("&mut self"));
    }

    #[test]
    fn gates_signature_diagnostics_with_the_handler_configuration() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                #[cfg(any())]
                fn visit_for<T>(&self, node: MissingNode, ctx: ()) where T: Clone {
                    todo!()
                }
            }
        };
        let generated = expand(&parse_quote!(visit), &item).unwrap().to_string();
        assert!(generated.contains("cfg (any ())"), "{generated}");
        assert!(generated.contains("compile_error"), "{generated}");
        assert!(
            generated.contains("non-generic Rust methods"),
            "{generated}"
        );
        assert!(generated.contains("VisitDispatch"), "{generated}");
    }

    #[test]
    fn rejects_an_empty_dispatch_impl() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                fn helper(&mut self) {}
            }
        };
        let error = expand(&parse_quote!(visit), &item).unwrap_err();
        assert!(error.to_string().contains("found no methods"));
    }

    #[test]
    fn rejects_unimplemented_family() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                fn visit_for(&mut self, node: For, ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
                    todo!()
                }
            }
        };
        let error = expand(&parse_quote!(map), &item).unwrap_err();
        assert!(error.to_string().contains("only `visit`"));
    }

    #[test]
    fn generated_impl_uses_runtime_trait() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                fn visit_for(&mut self, node: For, ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
                    todo!()
                }
            }
        };
        let generated = expand(&parse_quote!(visit), &item).unwrap();
        let expected_fragment = quote!(::tvm_tirx::visit::VisitDispatch).to_string();
        assert!(generated.to_string().contains(&expected_fragment));
    }

    #[test]
    fn copies_conditional_compilation_to_dispatch_arm() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                #[cfg(feature = "visit-for")]
                fn visit_for(&mut self, node: For, ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
                    todo!()
                }
            }
        };
        let generated = expand(&parse_quote!(visit), &item).unwrap().to_string();
        assert!(
            generated.contains("cfg (feature = \"visit-for\")"),
            "{generated}"
        );
    }

    #[test]
    fn cfg_attr_projects_only_conditional_compilation() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                #[cfg_attr(feature = "fast", inline, cfg(feature = "visit-for"))]
                fn visit_for(&mut self, node: For, ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
                    todo!()
                }
            }
        };
        let generated = expand(&parse_quote!(visit), &item).unwrap().to_string();

        assert!(
            generated.contains("cfg_attr (feature = \"fast\" , cfg (feature = \"visit-for\"))"),
            "{generated}"
        );
        assert!(!generated.contains("inline"), "{generated}");
    }

    #[test]
    fn projects_lint_levels_but_not_arbitrary_cfg_attr_entries() {
        let attr: Attribute = parse_quote!(
            #[cfg_attr(
                all(),
                allow(deprecated),
                warn(unused),
                deny(missing_docs),
                forbid(unsafe_code),
                expect(unused_variables),
                inline,
                cfg(any())
            )]
        );
        let generated = project_attribute(&attr, AttributeProjection::Generated)
            .unwrap()
            .unwrap()
            .to_string();
        assert!(generated.contains("allow (deprecated)"), "{generated}");
        assert!(generated.contains("warn (unused)"), "{generated}");
        assert!(generated.contains("deny (missing_docs)"), "{generated}");
        assert!(generated.contains("forbid (unsafe_code)"), "{generated}");
        assert!(
            generated.contains("allow (unused_variables)"),
            "{generated}"
        );
        assert!(generated.contains("cfg (any ())"), "{generated}");
        assert!(!generated.contains("expect"), "{generated}");
        assert!(!generated.contains("inline"), "{generated}");

        let conditions = project_attribute(&attr, AttributeProjection::ConditionsOnly)
            .unwrap()
            .unwrap()
            .to_string();
        assert!(conditions.contains("cfg (any ())"), "{conditions}");
        assert!(!conditions.contains("allow"), "{conditions}");
        assert!(!conditions.contains("warn"), "{conditions}");
        assert!(!conditions.contains("deny"), "{conditions}");
        assert!(!conditions.contains("forbid"), "{conditions}");
        assert!(!conditions.contains("expect"), "{conditions}");
        assert!(!conditions.contains("inline"), "{conditions}");
    }

    #[test]
    fn ignores_unrelated_attributes_without_parsing_their_payload() {
        let attr: Attribute = parse_quote!(#[custom_dispatch(key => value)]);
        assert!(project_attribute(&attr, AttributeProjection::Generated)
            .unwrap()
            .is_none());
    }

    #[test]
    fn copies_only_conditional_attributes_from_the_impl() {
        let item: ItemImpl = parse_quote! {
            #[cfg_attr(feature = "fast", inline, cfg(feature = "visitor"))]
            impl Analyzer {
                fn visit_for(&mut self, node: For, ctx: &mut VisitCtx<'_, Self>) -> WalkResult {
                    todo!()
                }
            }
        };
        let generated = expand(&parse_quote!(visit), &item).unwrap().to_string();

        assert!(
            generated.contains("cfg_attr (feature = \"fast\" , cfg (feature = \"visitor\"))"),
            "{generated}"
        );
        assert!(!generated.contains("inline"), "{generated}");
    }

    #[test]
    fn accepts_aliases_for_context_and_result() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                fn visit_for(&mut self, node: For, ctx: HandlerContext<'_>) -> HandlerResult {
                    todo!()
                }
            }
        };
        expand(&parse_quote!(visit), &item).unwrap();
    }

    #[test]
    fn rejects_method_level_where_clauses() {
        let item: ItemImpl = parse_quote! {
            impl<T> Analyzer<T> {
                fn visit_for(
                    &mut self,
                    node: For,
                    ctx: &mut VisitCtx<'_, Self>,
                ) -> WalkResult
                where
                    T: Clone,
                {
                    todo!()
                }
            }
        };
        let error = expand(&parse_quote!(visit), &item).unwrap_err();
        assert!(error.to_string().contains("non-generic"));
    }

    #[test]
    fn rejects_argument_position_impl_trait() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                fn visit_for(
                    &mut self,
                    node: impl NodeWrapper,
                    ctx: &mut VisitCtx<'_, Self>,
                ) -> WalkResult {
                    todo!()
                }
            }
        };
        let error = expand(&parse_quote!(visit), &item).unwrap_err();
        assert!(error.to_string().contains("cannot use `impl Trait`"));
    }

    #[test]
    fn rejects_impl_trait_in_context_and_result_types() {
        let ctx: Type = parse_quote!(&mut impl HandlerContext);
        assert!(validate_visit_ctx(&ctx)
            .unwrap_err()
            .to_string()
            .contains("cannot use `impl Trait`"));

        let output: ReturnType = parse_quote!(-> impl HandlerResult);
        assert!(validate_walk_result(&output)
            .unwrap_err()
            .to_string()
            .contains("cannot use `impl Trait`"));
    }

    #[test]
    fn rejects_obviously_immutable_context() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                fn visit_for(&mut self, node: For, ctx: &VisitCtx<'_, Self>) -> WalkResult {
                    todo!()
                }
            }
        };
        let error = expand(&parse_quote!(visit), &item).unwrap_err();
        assert!(error.to_string().contains("mutable reference"));
    }

    #[test]
    fn rejects_unit_result() {
        let item: ItemImpl = parse_quote! {
            impl Analyzer {
                fn visit_for(&mut self, node: For, ctx: &mut VisitCtx<'_, Self>) -> () {
                    todo!()
                }
            }
        };
        let error = expand(&parse_quote!(visit), &item).unwrap_err();
        assert!(error.to_string().contains("not `()`"));
    }
}
