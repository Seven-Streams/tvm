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

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Attribute, FnArg, ImplItem, ImplItemMethod, ItemImpl, Type};

/// Generate `VisitDispatch` from the `visit_*` methods in an inherent impl.
#[proc_macro_attribute]
pub fn dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mode = parse_macro_input!(attr as syn::Ident);
    let item_impl = parse_macro_input!(item as ItemImpl);

    match expand(&mode, &item_impl) {
        Ok(generated) => quote!(#item_impl #generated).into(),
        Err(error) => {
            let error = error.to_compile_error();
            quote!(#item_impl #error).into()
        }
    }
}

fn expand(mode: &syn::Ident, item_impl: &ItemImpl) -> syn::Result<TokenStream2> {
    if mode != "visit" {
        return Err(syn::Error::new(mode.span(), "expected `dispatch(visit)`"));
    }
    if item_impl.trait_.is_some() {
        return Err(syn::Error::new_spanned(
            item_impl,
            "`dispatch(visit)` requires an inherent impl",
        ));
    }

    let handlers = item_impl
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Method(method) if method.sig.ident.to_string().starts_with("visit_") => {
                Some(parse_handler(method))
            }
            _ => None,
        })
        .collect::<syn::Result<Vec<_>>>()?;

    if handlers.is_empty() {
        return Err(syn::Error::new_spanned(
            item_impl,
            "`dispatch(visit)` found no `visit_*` methods",
        ));
    }
    if let Some((index, handler)) = handlers
        .iter()
        .enumerate()
        .find(|(_, handler)| matches!(handler.argument, HandlerArgument::Value))
    {
        if index + 1 != handlers.len() {
            return Err(syn::Error::new_spanned(
                &handler.method,
                "the `&VisitValue` catch-all handler must be last",
            ));
        }
    }

    let has_catch_all = handlers
        .last()
        .is_some_and(|handler| matches!(handler.argument, HandlerArgument::Value));
    let typed_handler_count = handlers.len() - usize::from(has_catch_all);
    let links = handlers[..typed_handler_count].iter().map(|handler| {
        let method = &handler.method;
        let attrs = &handler.cfg_attrs;
        let invoke = match &handler.argument {
            HandlerArgument::Value => unreachable!("catch-all is emitted as the tail expression"),
            HandlerArgument::BorrowedNode(node_type) => quote! {
                if let Some(node) = value.as_node::<#node_type>() {
                    return Some(
                        ::tvm_tirx::visit::IntoVisitResult::into_visit_result(
                            self.#method(node, ctx)
                        )
                    );
                }
            },
            HandlerArgument::Owned(value_type) => quote! {
                if let Some(node) = value.cast::<#value_type>() {
                    return Some(
                        ::tvm_tirx::visit::IntoVisitResult::into_visit_result(
                            self.#method(node, ctx)
                        )
                    );
                }
            },
        };
        quote! {
            #(#attrs)*
            {
                #invoke
            }
        }
    });
    let tail = if has_catch_all {
        let handler = handlers.last().unwrap();
        let method = &handler.method;
        let attrs = &handler.cfg_attrs;
        if attrs.is_empty() {
            quote! {
                Some(
                    ::tvm_tirx::visit::IntoVisitResult::into_visit_result(
                        self.#method(value, ctx)
                    )
                )
            }
        } else {
            quote! {
                #(#attrs)*
                {
                    return Some(
                        ::tvm_tirx::visit::IntoVisitResult::into_visit_result(
                            self.#method(value, ctx)
                        )
                    );
                }
                None
            }
        }
    } else {
        quote!(None)
    };
    let self_type = &item_impl.self_ty;
    let (impl_generics, _, where_clause) = item_impl.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics ::tvm_tirx::visit::VisitDispatch for #self_type #where_clause {
            #[allow(unreachable_code)]
            fn dispatch_visit(
                &mut self,
                value: &::tvm_tirx::visit::VisitValue,
                ctx: &mut ::tvm_tirx::visit::VisitCtx<'_>,
            ) -> Option<::tvm_tirx::visit::VisitResult> {
                #(#links)*
                #tail
            }
        }
    })
}

struct Handler {
    method: syn::Ident,
    argument: HandlerArgument,
    cfg_attrs: Vec<Attribute>,
}

enum HandlerArgument {
    Value,
    BorrowedNode(Type),
    Owned(Type),
}

fn parse_handler(method: &ImplItemMethod) -> syn::Result<Handler> {
    let inputs = &method.sig.inputs;
    let receiver_is_mut = matches!(
        inputs.first(),
        Some(FnArg::Receiver(receiver))
            if receiver.reference.is_some() && receiver.mutability.is_some()
    );
    if !receiver_is_mut || inputs.len() != 3 {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "visit handlers must take `&mut self`, a node, and a context",
        ));
    }

    let node_type = match inputs.iter().nth(1) {
        Some(FnArg::Typed(node)) => (*node.ty).clone(),
        _ => unreachable!("the second argument cannot be a receiver"),
    };
    let argument = match &node_type {
        Type::Reference(reference) if reference.mutability.is_none() => {
            if is_visit_value(reference.elem.as_ref()) {
                HandlerArgument::Value
            } else {
                HandlerArgument::BorrowedNode((*reference.elem).clone())
            }
        }
        Type::Reference(_) => {
            return Err(syn::Error::new_spanned(
                node_type,
                "visit handler values cannot be mutable references",
            ));
        }
        _ => HandlerArgument::Owned(node_type),
    };
    let cfg_attrs = method
        .attrs
        .iter()
        .filter(|attr| attr.path.is_ident("cfg") || attr.path.is_ident("cfg_attr"))
        .cloned()
        .collect();
    Ok(Handler {
        method: method.sig.ident.clone(),
        argument,
        cfg_attrs,
    })
}

fn is_visit_value(value_type: &Type) -> bool {
    let Type::Path(path) = value_type else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "VisitValue")
}
