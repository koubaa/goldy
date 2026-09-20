//! Shared `GpuType` metadata generation and the `#[goldy::gpu]` attribute.

use proc_macro2::TokenStream;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{parse2, Attribute, Data, DeriveInput, Error, Expr, Fields, Meta, Token, Type};

const BUNDLED_DERIVES: &[&str] = &["Clone", "Copy", "Pod", "Zeroable", "GpuType"];

pub fn expand_gpu_type_derive(input: &DeriveInput) -> syn::Result<TokenStream> {
    if !has_repr_c(&input.attrs) {
        return Err(Error::new_spanned(input, "GpuType requires #[repr(C)]"));
    }
    gpu_type_impls(input)
}

pub fn expand_gpu_attr(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    if !attr.is_empty() {
        return Err(Error::new_spanned(attr, "#[goldy::gpu] does not take arguments"));
    }
    let input: DeriveInput = parse2(item)?;
    validate_gpu_attr_item(&input)?;
    let impls = gpu_type_impls(&input)?;
    let pod = pod_impls(&input)?;
    let vis = &input.vis;
    let name = &input.ident;
    let generics = &input.generics;
    let attrs = attrs_except_repr_and_bundled_derives(&input.attrs);
    let Data::Struct(data) = &input.data else {
        unreachable!("validated named struct");
    };
    let Fields::Named(named) = &data.fields else {
        unreachable!("validated named struct");
    };
    let fields = &named.named;

    Ok(quote! {
        #(#attrs)*
        #[repr(C)]
        #[derive(Clone, Copy)]
        #vis struct #name #generics {
            #fields
        }

        #impls
        #pod
    })
}

fn validate_gpu_attr_item(input: &DeriveInput) -> syn::Result<()> {
    if let Some(repr) = input.attrs.iter().find(|attr| attr.path().is_ident("repr")) {
        return Err(Error::new_spanned(
            repr,
            "#[goldy::gpu] supplies #[repr(C)]; omit explicit repr attributes",
        ));
    }
    if let Some(attr) = first_bundled_derive(&input.attrs) {
        return Err(Error::new_spanned(
            attr,
            "#[goldy::gpu] already provides Clone, Copy, Pod, Zeroable, and GpuType; omit those derives",
        ));
    }
    Ok(())
}

fn gpu_type_impls(input: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(Error::new_spanned(
            name,
            "GpuType cannot be derived for generic structs",
        ));
    }
    let fields = named_fields(input)?;
    let mut entries = Vec::with_capacity(fields.len());
    for field in fields {
        let ident = field.ident.as_ref().expect("named field");
        let field_name = ident.to_string();
        if field_name.starts_with("__goldy_pad") {
            return Err(Error::new_spanned(
                ident,
                "field names beginning with `__goldy_pad` are reserved for generated Slang padding",
            ));
        }
        let (variant, _) = gpu_field_info(&field.ty)?;
        let ty = &field.ty;
        entries.push(quote! {
            ::goldy::GpuField {
                name: #field_name,
                offset: ::std::mem::offset_of!(#name, #ident),
                size: ::std::mem::size_of::<#ty>(),
                ty: ::goldy::GpuFieldType::#variant,
            }
        });
    }
    let count = entries.len();
    let type_name = name.to_string();

    Ok(quote! {
        impl #name {
            /// Rust-authored GPU type descriptor used to generate the matching Slang struct.
            pub const GPU_TYPE: ::goldy::GpuType<'static> = ::goldy::GpuType {
                type_name: #type_name,
                rust_size: ::std::mem::size_of::<#name>(),
                fields: {
                    const FIELDS: [::goldy::GpuField<'static>; #count] = [
                        #(#entries),*
                    ];
                    &FIELDS
                },
            };
        }

        impl ::goldy::StructuredBufferElement for #name {
            fn gpu_element_stride() -> usize {
                #name::GPU_TYPE
                    .storage_stride()
                    .expect("GpuType storage stride")
            }

            fn gpu_encode_slice(items: &[Self]) -> ::std::borrow::Cow<'_, [u8]> {
                if items.is_empty() {
                    return ::std::borrow::Cow::Borrowed(&[]);
                }
                match #name::GPU_TYPE.encode_pod_slice(items) {
                    Ok(bytes) => ::std::borrow::Cow::Owned(bytes),
                    Err(err) => panic!("{err}"),
                }
            }
        }
    })
}

fn pod_impls(input: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let fields = named_fields(input)?;
    let mut packed_size = 0usize;
    for field in fields {
        let (_, size) = gpu_field_info(&field.ty)?;
        packed_size += size;
    }

    Ok(quote! {
        const _: () = {
            ::core::assert!(
                ::core::mem::size_of::<#name>() == #packed_size,
                "#[goldy::gpu] host layout has padding; use only supported 4-byte-aligned fields"
            );
        };

        unsafe impl ::goldy::__private::Zeroable for #name {}
        unsafe impl ::goldy::__private::Pod for #name {}
    })
}

fn named_fields(input: &DeriveInput) -> syn::Result<&Punctuated<syn::Field, Token![,]>> {
    match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => Ok(&named.named),
            _ => Err(Error::new_spanned(input, "GpuType requires named fields")),
        },
        _ => Err(Error::new_spanned(input, "GpuType can only be derived on structs")),
    }
}

fn has_repr_c(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("repr") {
            return false;
        }
        attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            .map(|items| items.iter().any(|item| item.path().is_ident("C")))
            .unwrap_or(false)
    })
}

fn first_bundled_derive(attrs: &[Attribute]) -> Option<&Attribute> {
    attrs.iter().find(|attr| {
        if !attr.path().is_ident("derive") {
            return false;
        }
        let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) else {
            return false;
        };
        nested.iter().any(|meta| {
            let ident = meta
                .path()
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            BUNDLED_DERIVES.contains(&ident.as_str())
        })
    })
}

fn attrs_except_repr_and_bundled_derives(attrs: &[Attribute]) -> Vec<&Attribute> {
    attrs
        .iter()
        .filter(|attr| {
            if attr.path().is_ident("repr") {
                return false;
            }
            if first_bundled_derive(std::slice::from_ref(*attr)).is_some() {
                return false;
            }
            true
        })
        .collect()
}

fn gpu_field_info(ty: &Type) -> syn::Result<(syn::Ident, usize)> {
    if let Type::Path(path) = ty {
        if path.qself.is_none() && path.path.segments.len() == 1 {
            let ident = &path.path.segments[0].ident;
            let (variant, size) = match ident.to_string().as_str() {
                "f32" => ("F32", 4),
                "u32" => ("U32", 4),
                "i32" => ("I32", 4),
                _ => {
                    return Err(Error::new_spanned(
                        ty,
                        "unsupported GpuType field; supported scalars are f32, u32, and i32",
                    ))
                }
            };
            return Ok((syn::Ident::new(variant, ident.span()), size));
        }
    }

    let Type::Array(outer) = ty else {
        return Err(Error::new_spanned(
            ty,
            "unsupported GpuType field; use a supported scalar, [T; 2..4] vector, or square f32 matrix",
        ));
    };
    let outer_len = array_len(&outer.len)?;

    if let Type::Array(inner) = outer.elem.as_ref() {
        let inner_len = array_len(&inner.len)?;
        if inner_len != outer_len || !(2..=4).contains(&outer_len) || !is_scalar(inner.elem.as_ref(), "f32") {
            return Err(Error::new_spanned(
                ty,
                "GpuType matrices must be square [[f32; N]; N] with N in 2..=4",
            ));
        }
        let size = match outer_len {
            2 => 16,
            3 => 36,
            4 => 64,
            _ => unreachable!(),
        };
        return Ok((
            syn::Ident::new(&format!("F32x{outer_len}x{outer_len}"), proc_macro2::Span::call_site()),
            size,
        ));
    }

    if !(2..=4).contains(&outer_len) {
        return Err(Error::new_spanned(ty, "GpuType vectors must have 2, 3, or 4 elements"));
    }
    let prefix = if is_scalar(outer.elem.as_ref(), "f32") {
        "F32"
    } else if is_scalar(outer.elem.as_ref(), "u32") {
        "U32"
    } else if is_scalar(outer.elem.as_ref(), "i32") {
        "I32"
    } else {
        return Err(Error::new_spanned(
            ty,
            "GpuType vectors support only f32, u32, and i32 elements",
        ));
    };
    let size = 4 * outer_len;
    Ok((
        syn::Ident::new(&format!("{prefix}x{outer_len}"), proc_macro2::Span::call_site()),
        size,
    ))
}

fn array_len(expr: &Expr) -> syn::Result<usize> {
    let Expr::Lit(lit) = expr else {
        return Err(Error::new_spanned(
            expr,
            "GpuType array lengths must be integer literals",
        ));
    };
    let syn::Lit::Int(value) = &lit.lit else {
        return Err(Error::new_spanned(
            expr,
            "GpuType array lengths must be integer literals",
        ));
    };
    value.base10_parse()
}

fn is_scalar(ty: &Type, expected: &str) -> bool {
    matches!(
        ty,
        Type::Path(path)
            if path.qself.is_none()
                && path.path.segments.len() == 1
                && path.path.segments[0].ident == expected
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_portable_scalars_vectors_and_matrices() {
        let input: DeriveInput = syn::parse_quote! {
            #[repr(C)]
            struct Example {
                scalar: f32,
                vector: [u32; 3],
                matrix: [[f32; 4]; 4],
            }
        };
        let expanded = expand_gpu_type_derive(&input).unwrap().to_string();
        assert!(expanded.contains("GpuFieldType :: F32"));
        assert!(expanded.contains("GpuFieldType :: U32x3"));
        assert!(expanded.contains("GpuFieldType :: F32x4x4"));
    }

    #[test]
    fn rejects_reserved_names() {
        let input: DeriveInput = syn::parse_quote! {
            #[repr(C)]
            struct Example {
                __goldy_pad0: u32,
            }
        };
        let err = expand_gpu_type_derive(&input).unwrap_err().to_string();
        assert!(err.contains("reserved"));
    }

    #[test]
    fn rejects_unsupported_field_types() {
        let input: DeriveInput = syn::parse_quote! {
            #[repr(C)]
            struct Example {
                value: u64,
            }
        };
        let err = expand_gpu_type_derive(&input).unwrap_err().to_string();
        assert!(err.contains("unsupported"));
    }

    #[test]
    fn all_named_fields_are_logical() {
        let input: DeriveInput = syn::parse_quote! {
            #[repr(C)]
            struct Example {
                value: [f32; 3],
                uv: [f32; 2],
            }
        };
        let expanded = expand_gpu_type_derive(&input).unwrap().to_string();
        assert!(expanded.contains("\"value\""));
        assert!(expanded.contains("\"uv\""));
        assert!(expanded.contains("StructuredBufferElement"));
    }

    #[test]
    fn accepts_repr_c_with_explicit_alignment() {
        let input: DeriveInput = syn::parse_quote! {
            #[repr(C, align(16))]
            struct Example {
                values: [u32; 3],
            }
        };
        expand_gpu_type_derive(&input).unwrap();
    }

    #[test]
    fn gpu_attr_supplies_repr_clone_copy_and_pod() {
        let item = quote! {
            struct Uniforms {
                width: u32,
                height: u32,
                time: f32,
            }
        };
        let expanded = expand_gpu_attr(TokenStream::new(), item).unwrap().to_string();
        assert!(expanded.contains("repr (C)"));
        assert!(expanded.contains("Clone"));
        assert!(expanded.contains("Copy"));
        assert!(expanded.contains("__private :: Zeroable"));
        assert!(expanded.contains("__private :: Pod"));
        assert!(expanded.contains("GPU_TYPE"));
        assert!(expanded.contains("size_of :: < Uniforms > () == 12usize"));
    }

    #[test]
    fn gpu_attr_rejects_arguments() {
        let err = expand_gpu_attr(quote! { packed }, quote! { struct Uniforms { x: f32 } })
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not take arguments"));
    }

    #[test]
    fn gpu_attr_rejects_explicit_repr() {
        let err = expand_gpu_attr(
            TokenStream::new(),
            quote! {
                #[repr(C)]
                struct Uniforms { x: f32 }
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("omit explicit repr"));
    }

    #[test]
    fn gpu_attr_rejects_duplicate_derives() {
        let err = expand_gpu_attr(
            TokenStream::new(),
            quote! {
                #[derive(Clone, Copy, bytemuck::Pod)]
                struct Uniforms { x: f32 }
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("already provides"));
    }

    #[test]
    fn gpu_attr_rejects_unsupported_fields() {
        let err = expand_gpu_attr(
            TokenStream::new(),
            quote! {
                struct Uniforms { value: u64 }
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("unsupported"));
    }
}
