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
use syn::{parse_macro_input, FnArg, ImplItem, ImplItemMethod, ItemImpl, Type};

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

    let arms = handlers.iter().map(|(method, node_type)| {
        quote! {
            if let Some(node) = value.cast::<#node_type>() {
                return Some(self.#method(node, ctx));
            }
        }
    });
    let self_type = &item_impl.self_ty;
    let (impl_generics, _, where_clause) = item_impl.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics ::tvm_tirx::visit::VisitDispatch for #self_type #where_clause {
            fn dispatch_visit(
                &mut self,
                value: &::tvm_tirx::visit::VisitValue,
                ctx: &mut ::tvm_tirx::visit::VisitCtx<'_, Self>,
            ) -> Option<::tvm_tirx::visit::WalkResult> {
                #(#arms)*
                None
            }
        }
    })
}

fn parse_handler(method: &ImplItemMethod) -> syn::Result<(syn::Ident, Type)> {
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
    Ok((method.sig.ident.clone(), node_type))
}
