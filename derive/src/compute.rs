//! `#[goldy_derive::compute]` — compile-time Rust GPU dialect → Slang + KernelAbi.

use crate::ir_tokens;
use goldy_shader_ir::{
    emit_canonical_compute_source, BinOp, BuiltinFn, BuiltinMask, ElementType, Expr, KernelParam, ParamCategory,
    ScalarType, ShaderKernel, SourceMap, Stmt, TensorDimSpec, TensorShapeSpec, UnaryOp, WorkgroupReduceOp,
    TENSOR_SHAPE_SPEC_MAX_RANK,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{
    parse2, Attribute, BinOp as SynBinOp, Error, Expr as SynExpr, ExprBinary, ExprCall, ExprField, ExprIndex, ExprLit,
    ExprMethodCall, ExprPath, ExprUnary, FnArg, GenericArgument, ItemFn, Lit, Meta, Pat, PatType, PathArguments,
    ReturnType, Stmt as SynStmt, Type, UnOp,
};

mod kw {
    syn::custom_keyword!(workgroup_size);
}

pub struct ComputeArgs {
    pub workgroup_size: [u32; 3],
}

impl Parse for ComputeArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut workgroup_size = [64, 1, 1];
        while !input.is_empty() {
            if input.peek(kw::workgroup_size) {
                let _ = input.parse::<kw::workgroup_size>()?;
                input.parse::<syn::Token![=]>()?;
                let content;
                syn::bracketed!(content in input);
                let x: syn::LitInt = content.parse()?;
                content.parse::<syn::Token![,]>()?;
                let y: syn::LitInt = content.parse()?;
                content.parse::<syn::Token![,]>()?;
                let z: syn::LitInt = content.parse()?;
                workgroup_size = [x.base10_parse()?, y.base10_parse()?, z.base10_parse()?];
            } else {
                return Err(input.error("expected `workgroup_size = [x, y, z]`"));
            }
            if input.peek(syn::Token![,]) {
                let _ = input.parse::<syn::Token![,]>()?;
            }
        }
        Ok(Self { workgroup_size })
    }
}

pub fn expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    let args: ComputeArgs = if attr.is_empty() {
        ComputeArgs {
            workgroup_size: [64, 1, 1],
        }
    } else {
        parse2(attr)?
    };
    let func: ItemFn = parse2(item)?;
    expand_fn(args, func)
}

fn expand_fn(args: ComputeArgs, func: ItemFn) -> Result<TokenStream, Error> {
    if !matches!(func.sig.output, ReturnType::Default) {
        return Err(Error::new(func.sig.output.span(), "#[compute] kernels must return ()"));
    }
    if !func.sig.generics.params.is_empty() {
        return Err(Error::new(
            func.sig.generics.span(),
            "#[compute] kernels cannot be generic",
        ));
    }
    if func.sig.asyncness.is_some() {
        return Err(Error::new(
            func.sig.asyncness.span(),
            "#[compute] kernels cannot be async",
        ));
    }

    let fn_name = &func.sig.ident;
    let mod_name = fn_name.clone();
    let mut builtins = BuiltinMask::NONE;
    let mut params = Vec::new();
    let mut record_args = Vec::new();
    let mut bind_stmts = Vec::new();
    let mut validate_stmts = Vec::new();
    let mut gpu_type_idents: Vec<syn::Ident> = Vec::new();
    let mut type_env: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut has_tensors = false;

    for input in &func.sig.inputs {
        let FnArg::Typed(PatType { attrs, pat, ty, .. }) = input else {
            return Err(Error::new(input.span(), "#[compute] does not support `self`"));
        };
        let Pat::Ident(name) = pat.as_ref() else {
            return Err(Error::new(pat.span(), "kernel parameters must be plain identifiers"));
        };
        let pname = name.ident.to_string();
        let pident = &name.ident;
        let shape_spec = take_tensor_shape_spec(attrs)?;
        let classified = classify_param_type(ty)?;
        if shape_spec.is_some()
            && !matches!(
                classified,
                ClassifiedParam::TensorRead(_) | ClassifiedParam::TensorReadWrite(_) | ClassifiedParam::TensorWrite(_)
            )
        {
            return Err(Error::new(
                ty.span(),
                "#[tensor(shape = ...)] is only valid on gpu::Tensor / TensorMut / TensorWrite parameters",
            ));
        }
        match classified {
            ClassifiedParam::BufferRead(elem) => {
                params.push(KernelParam::buffer_read(&pname, elem));
                type_env.insert(pname.clone(), format!("BufRO<{}>", elem.slang_name()));
                record_args.push(quote! { #pident: &impl ::goldy::kernel::KernelBindable });
                bind_stmts.push(quote! {
                    start = ::goldy::kernel::KernelBindable::__goldy_bind_kernel(
                        #pident,
                        start,
                        ::goldy::NodeAccess::Read,
                    );
                });
            }
            ClassifiedParam::BufferReadNamed(type_name) => {
                let ty_ident = syn::Ident::new(&type_name, pident.span());
                if !gpu_type_idents.iter().any(|id| id == &ty_ident) {
                    gpu_type_idents.push(ty_ident);
                }
                params.push(KernelParam::buffer_read_named(&pname, &type_name));
                type_env.insert(pname.clone(), format!("BufRO<{type_name}>"));
                record_args.push(quote! { #pident: &impl ::goldy::kernel::KernelBindable });
                bind_stmts.push(quote! {
                    start = ::goldy::kernel::KernelBindable::__goldy_bind_kernel(
                        #pident,
                        start,
                        ::goldy::NodeAccess::Read,
                    );
                });
            }
            ClassifiedParam::BufferReadWrite(elem) => {
                params.push(KernelParam::buffer_read_write(&pname, elem));
                type_env.insert(pname.clone(), format!("Scattered<{}>", elem.slang_name()));
                record_args.push(quote! { #pident: &impl ::goldy::kernel::KernelBindable });
                bind_stmts.push(quote! {
                    start = ::goldy::kernel::KernelBindable::__goldy_bind_kernel(
                        #pident,
                        start,
                        ::goldy::NodeAccess::ReadWrite,
                    );
                });
            }
            ClassifiedParam::BufferWrite(elem) => {
                params.push(KernelParam::buffer_write(&pname, elem));
                type_env.insert(pname.clone(), format!("Scattered<{}>", elem.slang_name()));
                record_args.push(quote! { #pident: &impl ::goldy::kernel::KernelBindable });
                bind_stmts.push(quote! {
                    start = ::goldy::kernel::KernelBindable::__goldy_bind_kernel(
                        #pident,
                        start,
                        ::goldy::NodeAccess::Write,
                    );
                });
            }
            ClassifiedParam::TensorRead(elem) => {
                has_tensors = true;
                let mut p = KernelParam::tensor_read(&pname, elem);
                p.shape_spec = shape_spec;
                params.push(p);
                type_env.insert(pname.clone(), format!("Tensor<{}>", elem.slang_name()));
                let dtype = tensor_dtype_tokens(elem);
                record_args.push(quote! { #pident: ::goldy::TensorView<'_> });
                validate_stmts.push(quote! {
                    start.check_tensor_view(
                        #pident,
                        #pname,
                        ::goldy::NodeAccess::Read,
                        #dtype,
                        &mut tensor_shape_env,
                    )?;
                });
                bind_stmts.push(quote! {
                    start = start.bind_tensor_view(#pident, ::goldy::NodeAccess::Read, #dtype)?;
                });
            }
            ClassifiedParam::TensorReadWrite(elem) => {
                has_tensors = true;
                let mut p = KernelParam::tensor_read_write(&pname, elem);
                p.shape_spec = shape_spec;
                params.push(p);
                type_env.insert(pname.clone(), format!("TensorMut<{}>", elem.slang_name()));
                let dtype = tensor_dtype_tokens(elem);
                record_args.push(quote! { #pident: ::goldy::TensorView<'_> });
                validate_stmts.push(quote! {
                    start.check_tensor_view(
                        #pident,
                        #pname,
                        ::goldy::NodeAccess::ReadWrite,
                        #dtype,
                        &mut tensor_shape_env,
                    )?;
                });
                bind_stmts.push(quote! {
                    start = start.bind_tensor_view(#pident, ::goldy::NodeAccess::ReadWrite, #dtype)?;
                });
            }
            ClassifiedParam::TensorWrite(elem) => {
                has_tensors = true;
                let mut p = KernelParam::tensor_write(&pname, elem);
                p.shape_spec = shape_spec;
                params.push(p);
                type_env.insert(pname.clone(), format!("TensorWrite<{}>", elem.slang_name()));
                let dtype = tensor_dtype_tokens(elem);
                record_args.push(quote! { #pident: ::goldy::TensorView<'_> });
                validate_stmts.push(quote! {
                    start.check_tensor_view(
                        #pident,
                        #pname,
                        ::goldy::NodeAccess::Write,
                        #dtype,
                        &mut tensor_shape_env,
                    )?;
                });
                bind_stmts.push(quote! {
                    start = start.bind_tensor_view(#pident, ::goldy::NodeAccess::Write, #dtype)?;
                });
            }
            ClassifiedParam::Uniform(type_name) => {
                let ty_ident = syn::Ident::new(&type_name, pident.span());
                if !gpu_type_idents.iter().any(|id| id == &ty_ident) {
                    gpu_type_idents.push(ty_ident);
                }
                params.push(KernelParam {
                    name: pname.clone(),
                    category: ParamCategory::Uniform,
                    access: Some(goldy_shader_ir::AccessKind::Read),
                    scalar: None,
                    slang_type: type_name.clone(),
                    stride_bytes: None,
                    is_tensor: false,
                    shape_spec: None,
                });
                type_env.insert(pname, type_name);
                record_args.push(quote! { #pident: &impl ::goldy::kernel::KernelBindable });
                bind_stmts.push(quote! {
                    start = ::goldy::kernel::KernelBindable::__goldy_bind_kernel(
                        #pident,
                        start,
                        ::goldy::NodeAccess::Read,
                    );
                });
            }
            ClassifiedParam::Scalar(st) => {
                params.push(KernelParam::scalar_param(&pname, st));
                type_env.insert(pname.clone(), st.slang_name().to_string());
                match st {
                    ScalarType::U32 => {
                        bind_stmts.push(quote! { start = start.bind_u32(#pident); });
                        record_args.push(quote! { #pident: u32 });
                    }
                    ScalarType::I32 => {
                        bind_stmts.push(quote! { start = start.bind_i32(#pident); });
                        record_args.push(quote! { #pident: i32 });
                    }
                    ScalarType::F32 => {
                        bind_stmts.push(quote! { start = start.bind_f32(#pident); });
                        record_args.push(quote! { #pident: f32 });
                    }
                    ScalarType::Bool => {
                        bind_stmts.push(quote! { start = start.bind_bool(#pident); });
                        record_args.push(quote! { #pident: bool });
                    }
                }
            }
            ClassifiedParam::StorageImage(elem) => {
                params.push(KernelParam::storage_image(&pname, &elem));
                type_env.insert(pname.clone(), format!("DirectSpatial<{elem}>"));
                record_args.push(quote! { #pident: &impl ::goldy::kernel::KernelBindable });
                bind_stmts.push(quote! {
                    start = ::goldy::kernel::KernelBindable::__goldy_bind_kernel(
                        #pident,
                        start,
                        ::goldy::NodeAccess::Write,
                    );
                });
            }
        }
    }

    let body_stmts = lower_block(&func.block.stmts, &mut builtins, &mut type_env, args.workgroup_size[0])?;
    if workgroup_array_in_nested_scope(&body_stmts) {
        return Err(Error::new(
            func.sig.ident.span(),
            "gpu::workgroup_array must be declared at kernel top level",
        ));
    }

    let kernel = ShaderKernel {
        name: fn_name.to_string(),
        workgroup_size: args.workgroup_size,
        params: params.clone(),
        builtins,
        body: body_stmts,
        source_map: SourceMap {
            rust_file: "<goldy-compute>".into(),
            rust_line: 0,
        },
        type_decls: Vec::new(),
    };

    let def = emit_canonical_compute_source(&kernel);
    let slang = &def.source.canonical_slang;
    let entry = &def.entry;
    let [wx, wy, wz] = def.workgroup_size;
    let abi_version = def.abi_version;
    let rust_file = &def.source_map.rust_file;
    let rust_line = def.source_map.rust_line;

    let param_tokens: Vec<_> = def.params.iter().map(ir_tokens::param).collect();
    let builtins_tokens = ir_tokens::builtins(builtins);
    let definition_tokens = ir_tokens::kernel(
        &kernel,
        quote! { ::std::vec![#(#gpu_type_idents::GPU_TYPE.to_slang_source()?),*] },
    );

    let record_method = if has_tensors {
        quote! {
            /// Record a dispatch into `scheme`, binding tensor views and packing their layouts.
            pub fn record<'a>(
                &'a self,
                scheme: &'a mut ::goldy::Scheme,
                label: impl Into<::goldy::SchemeLabel>,
                #(#record_args),*
            ) -> ::core::result::Result<::goldy::kernel::DispatchBuilder<'a>, ::goldy::GoldyError> {
                let mut start = self.prepared.begin_record(scheme, label);
                let mut tensor_shape_env = ::goldy::kernel::TensorShapeEnv::new();
                #(#validate_stmts)*
                #(#bind_stmts)*
                start.finish_with_tensor_meta()
            }
        }
    } else {
        quote! {
            /// Record a dispatch into `scheme`, binding arguments in declaration order.
            pub fn record<'a>(
                &'a self,
                scheme: &'a mut ::goldy::Scheme,
                label: impl Into<::goldy::SchemeLabel>,
                #(#record_args),*
            ) -> ::goldy::kernel::DispatchBuilder<'a> {
                let mut start = self.prepared.begin_record(scheme, label);
                #(#bind_stmts)*
                start.finish()
            }
        }
    };
    let kernel_struct = format_ident!("Kernel");
    let docs = format!(
        "Prepared handle for the `{fn_name}` compute kernel (workgroup {:?}).",
        args.workgroup_size
    );

    // Keep original attributes except our own compute attr if re-exported.
    let attrs: Vec<&Attribute> = func.attrs.iter().filter(|a| !is_compute_attr(a)).collect();

    Ok(quote! {
        #(#attrs)*
        #[allow(non_snake_case)]
        pub mod #mod_name {
            use super::*;

            /// Canonical `[goldy_compute]` Slang produced by `#[goldy::compute]`.
            ///
            /// Excludes the declarations of `#[goldy::gpu]` parameter types, which
            /// `prepare` places ahead of it.
            pub const CANONICAL_SOURCE: &str = #slang;

            /// Structured definition this kernel was lowered from, including the
            /// declarations of its `#[goldy::gpu]` parameter types.
            pub fn definition() -> ::core::result::Result<::goldy::kernel::ShaderKernel, ::goldy::GoldyError> {
                Ok(#definition_tokens)
            }

            #[doc = #docs]
            pub struct #kernel_struct {
                prepared: ::goldy::kernel::PreparedKernel,
            }

            impl #kernel_struct {
                /// Compile (or hit the shader cache) and create a device-scoped pipeline.
                pub fn prepare(device: &::goldy::Runtime) -> ::core::result::Result<Self, ::goldy::GoldyError> {
                    let definition = definition()?;
                    let mut canonical_slang = ::std::string::String::new();
                    for decl in &definition.type_decls {
                        canonical_slang.push_str(decl);
                        canonical_slang.push('\n');
                    }
                    canonical_slang.push_str(CANONICAL_SOURCE);
                    let def = ::goldy::kernel::KernelDef {
                        source: ::goldy::kernel::KernelSource {
                            canonical_slang,
                        },
                        entry: #entry.to_string(),
                        workgroup_size: [#wx, #wy, #wz],
                        params: vec![#(#param_tokens),*],
                        builtins: #builtins_tokens,
                        source_map: ::goldy::kernel::SourceMap {
                            rust_file: #rust_file.to_string(),
                            rust_line: #rust_line,
                        },
                        abi_version: #abi_version,
                        definition: Some(definition),
                    };
                    let prepared = ::goldy::kernel::prepare_kernel(device, def).map_err(::goldy::GoldyError::Backend)?;
                    Ok(Self { prepared })
                }

                #record_method

                pub fn workgroup_size(&self) -> [u32; 3] {
                    self.prepared.workgroup_size()
                }

                /// Device pipeline used by this prepared kernel.
                pub fn pipeline(&self) -> &::goldy::ComputePipeline {
                    self.prepared.pipeline()
                }

                /// Shared device pipeline. Cloning the `Arc` does not destroy the PSO.
                pub fn pipeline_arc(&self) -> ::std::sync::Arc<::goldy::ComputePipeline> {
                    self.prepared.pipeline_arc()
                }

                /// Structured ABI used to prepare this kernel.
                pub fn def(&self) -> &::goldy::kernel::KernelDef {
                    self.prepared.def()
                }
            }
        }
    })
}

fn is_compute_attr(attr: &Attribute) -> bool {
    match &attr.meta {
        Meta::Path(p)
        | Meta::List(syn::MetaList { path: p, .. })
        | Meta::NameValue(syn::MetaNameValue { path: p, .. }) => {
            p.is_ident("compute") || p.segments.last().is_some_and(|s| s.ident == "compute")
        }
    }
}

fn take_tensor_shape_spec(attrs: &[Attribute]) -> Result<Option<TensorShapeSpec>, Error> {
    let mut found = None;
    for attr in attrs {
        if !attr.path().is_ident("tensor") {
            continue;
        }
        if found.is_some() {
            return Err(Error::new(attr.span(), "duplicate #[tensor(...)] attribute"));
        }
        found = Some(parse_tensor_attr(attr)?);
    }
    Ok(found)
}

fn parse_tensor_attr(attr: &Attribute) -> Result<TensorShapeSpec, Error> {
    match &attr.meta {
        Meta::List(_) => attr.parse_args_with(parse_tensor_attr_args),
        _ => Err(Error::new(attr.span(), "expected #[tensor(shape = [...])]")),
    }
}

fn parse_tensor_attr_args(input: ParseStream) -> Result<TensorShapeSpec, Error> {
    let ident: syn::Ident = input.parse()?;
    if ident != "shape" {
        return Err(Error::new(ident.span(), "expected `shape = [...]` in #[tensor(...)]"));
    }
    input.parse::<syn::Token![=]>()?;
    let content;
    syn::bracketed!(content in input);
    let mut dims = Vec::new();
    while !content.is_empty() {
        dims.push(parse_tensor_dim(&content)?);
        if content.peek(syn::Token![,]) {
            let _ = content.parse::<syn::Token![,]>()?;
        } else if !content.is_empty() {
            return Err(content.error("expected comma or closing `]` in tensor shape"));
        }
    }
    if dims.len() > TENSOR_SHAPE_SPEC_MAX_RANK {
        return Err(Error::new(
            ident.span(),
            format!("tensor shape contracts support at most {TENSOR_SHAPE_SPEC_MAX_RANK} dimensions"),
        ));
    }
    if !input.is_empty() {
        return Err(input.error("unexpected tokens after tensor shape"));
    }
    Ok(TensorShapeSpec { dims })
}

fn parse_tensor_dim(input: ParseStream) -> Result<TensorDimSpec, Error> {
    if input.peek(syn::Token![_]) {
        let _: syn::Token![_] = input.parse()?;
        return Ok(TensorDimSpec::Any);
    }
    if input.peek(syn::LitInt) {
        let n: syn::LitInt = input.parse()?;
        let v: u32 = n.base10_parse()?;
        return Ok(TensorDimSpec::Exact(v));
    }
    if input.peek(syn::Ident) {
        let id: syn::Ident = input.parse()?;
        return Ok(TensorDimSpec::Symbol(id.to_string()));
    }
    Err(input.error("tensor dim must be `_`, an integer literal, or an identifier"))
}

enum ClassifiedParam {
    BufferRead(ElementType),
    BufferReadNamed(String),
    BufferReadWrite(ElementType),
    BufferWrite(ElementType),
    TensorRead(ElementType),
    TensorReadWrite(ElementType),
    TensorWrite(ElementType),
    Uniform(String),
    StorageImage(String),
    Scalar(ScalarType),
}

fn classify_param_type(ty: &Type) -> Result<ClassifiedParam, Error> {
    // Resource names match shaders/goldy_exp/access.slang.
    if let Some(inner) = match_gpu_generic(ty, "Scattered") {
        let elem = element_from_type(inner)?;
        return Ok(ClassifiedParam::BufferWrite(elem));
    }
    if let Some(inner) = match_gpu_generic(ty, "Tensor") {
        return Ok(ClassifiedParam::TensorRead(tensor_element_from_type(inner)?));
    }
    if let Some(inner) = match_gpu_generic(ty, "TensorMut") {
        return Ok(ClassifiedParam::TensorReadWrite(tensor_element_from_type(inner)?));
    }
    if let Some(inner) = match_gpu_generic(ty, "TensorWrite") {
        return Ok(ClassifiedParam::TensorWrite(tensor_element_from_type(inner)?));
    }
    if let Some(inner) = match_gpu_generic(ty, "BufRO") {
        return match buffer_element(inner)? {
            BufferElem::Primitive(elem) => Ok(ClassifiedParam::BufferRead(elem)),
            BufferElem::Named(name) => Ok(ClassifiedParam::BufferReadNamed(name)),
        };
    }
    if let Some(inner) = match_gpu_generic(ty, "Uniform") {
        let name = type_to_slang_name(inner)?;
        return Ok(ClassifiedParam::Uniform(name));
    }
    if match_gpu(ty, "DirectSpatial") {
        let elem = match match_gpu_generic(ty, "DirectSpatial") {
            Some(inner) => texel_element_slang(inner)?,
            None => "float4".into(),
        };
        return Ok(ClassifiedParam::StorageImage(elem));
    }
    if match_gpu(ty, "Interpolated") {
        return Err(Error::new(
            ty.span(),
            "gpu::Interpolated is not yet supported in #[compute] kernels",
        ));
    }
    if match_gpu(ty, "ByteAddress") {
        return Err(Error::new(
            ty.span(),
            "gpu::ByteAddress is not yet supported in #[compute] kernels",
        ));
    }
    if match_gpu(ty, "Filter") {
        return Err(Error::new(
            ty.span(),
            "gpu::Filter is not yet supported in #[compute] kernels",
        ));
    }
    if match_gpu(ty, "Accel") {
        return Err(Error::new(
            ty.span(),
            "gpu::Accel is not yet supported in #[compute] kernels",
        ));
    }

    match ty {
        Type::Reference(r) => match buffer_element(&r.elem)? {
            BufferElem::Primitive(elem) => {
                if r.mutability.is_some() {
                    Ok(ClassifiedParam::BufferReadWrite(elem))
                } else {
                    Ok(ClassifiedParam::BufferRead(elem))
                }
            }
            BufferElem::Named(name) => {
                if r.mutability.is_some() {
                    Err(Error::new(
                        ty.span(),
                        "named struct buffers are read-only in the MVP (`&[T]` / gpu::BufRO); use gpu::DirectSpatial for surfaces",
                    ))
                } else {
                    Ok(ClassifiedParam::BufferReadNamed(name))
                }
            }
        },
        Type::Path(p) if p.qself.is_none() => {
            let name = p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            match name.as_str() {
                "u32" => Ok(ClassifiedParam::Scalar(ScalarType::U32)),
                "i32" => Ok(ClassifiedParam::Scalar(ScalarType::I32)),
                "f32" => Ok(ClassifiedParam::Scalar(ScalarType::F32)),
                "bool" => Ok(ClassifiedParam::Scalar(ScalarType::Bool)),
                "usize" | "isize" => Err(Error::new(
                    ty.span(),
                    "usize/isize are not supported in #[compute] kernels; use u32/i32",
                )),
                other => Err(Error::new(
                    ty.span(),
                    format!("unsupported kernel parameter type `{other}`"),
                )),
            }
        }
        _ => Err(Error::new(
            ty.span(),
            "unsupported kernel parameter type; expected &[T], &mut [T], gpu::BufRO<T>, gpu::Scattered<T>, gpu::Tensor<T>, gpu::TensorMut<T>, gpu::TensorWrite<T>, gpu::Uniform<T>, gpu::DirectSpatial<T>, or u32/i32/f32/bool",
        )),
    }
}

fn path_matches(ty: &Type, segs: &[&str]) -> bool {
    let Type::Path(p) = ty else {
        return false;
    };
    if p.qself.is_some() || p.path.segments.len() != segs.len() {
        return false;
    }
    p.path
        .segments
        .iter()
        .zip(segs.iter())
        .all(|(seg, expect)| seg.ident == expect)
}

fn match_gpu(ty: &Type, name: &str) -> bool {
    path_matches(ty, &["gpu", name]) || path_matches(ty, &["goldy", "gpu", name])
}

fn match_gpu_generic<'a>(ty: &'a Type, name: &str) -> Option<&'a Type> {
    match_path_generic(ty, &["gpu", name]).or_else(|| match_path_generic(ty, &["goldy", "gpu", name]))
}

fn match_path_generic<'a>(ty: &'a Type, segs: &[&str]) -> Option<&'a Type> {
    let Type::Path(p) = ty else {
        return None;
    };
    if p.qself.is_some() || p.path.segments.len() != segs.len() {
        return None;
    }
    for (seg, expect) in p.path.segments.iter().zip(segs.iter()) {
        if seg.ident != expect {
            return None;
        }
    }
    let last = p.path.segments.last()?;
    match &last.arguments {
        syn::PathArguments::AngleBracketed(a) if a.args.len() == 1 => match a.args.first()? {
            syn::GenericArgument::Type(t) => Some(t),
            _ => None,
        },
        _ => None,
    }
}

fn buffer_element(ty: &Type) -> Result<BufferElem, Error> {
    match ty {
        Type::Slice(s) => buffer_element(&s.elem),
        Type::Path(p) if p.qself.is_none() => {
            let name = p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
            match name.as_str() {
                "u32" => Ok(BufferElem::Primitive(ElementType::U32)),
                "i32" => Ok(BufferElem::Primitive(ElementType::I32)),
                "f32" => Ok(BufferElem::Primitive(ElementType::F32)),
                "bool" => Ok(BufferElem::Primitive(ElementType::Bool)),
                other => Ok(BufferElem::Named(other.to_string())),
            }
        }
        _ => Err(Error::new(
            ty.span(),
            "unsupported buffer element type; use u32/i32/f32/bool or a #[goldy::gpu] struct",
        )),
    }
}

enum BufferElem {
    Primitive(ElementType),
    Named(String),
}

fn texel_element_slang(ty: &Type) -> Result<String, Error> {
    let Type::Path(p) = ty else {
        return Err(Error::new(ty.span(), "gpu::DirectSpatial element must be a path type"));
    };
    let name = p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
    Ok(match name.as_str() {
        "Float4" | "float4" => "float4".into(),
        "Float3" | "float3" => "float3".into(),
        "Float2" | "float2" => "float2".into(),
        "f32" => "float".into(),
        other => {
            return Err(Error::new(
                ty.span(),
                format!("unsupported gpu::DirectSpatial element `{other}`; use gpu::Float4"),
            ))
        }
    })
}

fn tensor_dtype_tokens(elem: ElementType) -> TokenStream {
    match elem {
        ElementType::F32 => quote! { ::goldy::TensorDType::F32 },
        ElementType::U32 => quote! { ::goldy::TensorDType::U32 },
        ElementType::I32 => quote! { ::goldy::TensorDType::I32 },
        ElementType::Bool => quote! { ::goldy::TensorDType::F32 },
    }
}

fn tensor_element_from_type(ty: &Type) -> Result<ElementType, Error> {
    let elem = element_from_type(ty)?;
    if matches!(elem, ElementType::Bool) {
        return Err(Error::new(
            ty.span(),
            "gpu::Tensor parameters support f32, u32, and i32 (not bool)",
        ));
    }
    Ok(elem)
}

fn is_tensor_env_ty(ty: &str) -> bool {
    ty.starts_with("Tensor<") || ty.starts_with("TensorMut<") || ty.starts_with("TensorWrite<")
}

fn element_from_type(ty: &Type) -> Result<ElementType, Error> {
    match ty {
        Type::Slice(s) => element_from_type(&s.elem),
        Type::Path(p) if p.qself.is_none() => {
            let name = p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
            match name.as_str() {
                "u32" => Ok(ElementType::U32),
                "i32" => Ok(ElementType::I32),
                "f32" => Ok(ElementType::F32),
                "bool" => Ok(ElementType::Bool),
                other => Err(Error::new(
                    ty.span(),
                    format!("MVP #[compute] buffers only support u32/i32/f32/bool elements (got `{other}`)"),
                )),
            }
        }
        _ => Err(Error::new(
            ty.span(),
            "unsupported buffer element type; MVP supports u32/i32/f32/bool",
        )),
    }
}

fn type_to_slang_name(ty: &Type) -> Result<String, Error> {
    match ty {
        Type::Path(p) if p.qself.is_none() => {
            let name = p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_else(|| "Unknown".into());
            Ok(match name.as_str() {
                "f32" => "float".into(),
                "u32" => "uint".into(),
                "i32" => "int".into(),
                "bool" => "bool".into(),
                "Float2" | "float2" => "float2".into(),
                "Float3" | "float3" => "float3".into(),
                "Float4" | "float4" => "float4".into(),
                other => other.to_string(),
            })
        }
        _ => Err(Error::new(ty.span(), "unsupported type in #[compute] kernel")),
    }
}

fn lower_block(
    stmts: &[SynStmt],
    builtins: &mut BuiltinMask,
    env: &mut std::collections::HashMap<String, String>,
    wg_x: u32,
) -> Result<Vec<Stmt>, Error> {
    let mut out = Vec::new();
    for s in stmts {
        out.extend(lower_stmt(s, builtins, env, wg_x)?);
    }
    Ok(out)
}

fn lower_stmt(
    stmt: &SynStmt,
    builtins: &mut BuiltinMask,
    env: &mut std::collections::HashMap<String, String>,
    wg_x: u32,
) -> Result<Vec<Stmt>, Error> {
    match stmt {
        SynStmt::Local(local) => {
            let (name, ascribed) = match &local.pat {
                Pat::Ident(name) => (name, None),
                Pat::Type(pt) => {
                    let Pat::Ident(name) = pt.pat.as_ref() else {
                        return Err(Error::new(
                            local.pat.span(),
                            "only simple `let` bindings are supported in #[compute] kernels",
                        ));
                    };
                    (name, Some(type_to_slang_name(&pt.ty)?))
                }
                _ => {
                    return Err(Error::new(
                        local.pat.span(),
                        "only simple `let` bindings are supported in #[compute] kernels",
                    ))
                }
            };
            let init = local
                .init
                .as_ref()
                .ok_or_else(|| Error::new(local.span(), "let without initializer is unsupported"))?;
            if let Some((elem, len)) = parse_workgroup_array_init(&init.expr)? {
                env.insert(name.ident.to_string(), format!("groupshared<{elem}>"));
                return Ok(vec![Stmt::WorkgroupArray {
                    name: name.ident.to_string(),
                    elem,
                    len,
                }]);
            }
            if let Some(collective) = try_parse_collective(&init.expr, builtins, wg_x, env)? {
                let dest_name = name.ident.to_string();
                return match collective {
                    Collective::Reduce { op, n, val, scratch } => {
                        env.insert(dest_name.clone(), "float".into());
                        Ok(vec![
                            Stmt::Let {
                                name: dest_name.clone(),
                                mutable: true,
                                ty: Some("float".into()),
                                init: val,
                            },
                            Stmt::WorkgroupReduce {
                                op,
                                n,
                                val: Expr::Var(dest_name.clone()),
                                scratch,
                                dest: Expr::Var(dest_name),
                            },
                        ])
                    }
                    Collective::Softmax { .. } => Err(Error::new(
                        init.expr.span(),
                        "gpu::workgroup_softmax_in_place is a statement, not a value",
                    )),
                };
            }
            let expr = lower_expr(&init.expr, builtins, env)?;
            let ty = ascribed.or_else(|| infer_slang_ty(&expr, env));
            if let Some(ref ty) = ty {
                env.insert(name.ident.to_string(), ty.clone());
            }
            Ok(vec![Stmt::Let {
                name: name.ident.to_string(),
                mutable: name.mutability.is_some(),
                ty,
                init: expr,
            }])
        }
        SynStmt::Expr(expr, semi) => {
            if semi.is_none() {
                // trailing expr — treat as expression statement (or assign)
            }
            match expr {
                SynExpr::Assign(a) => {
                    if let Some(collective) = try_parse_collective(&a.right, builtins, wg_x, env)? {
                        return Ok(vec![lower_collective_assign(
                            lower_expr(&a.left, builtins, env)?,
                            collective,
                            a.right.span(),
                        )?]);
                    }
                    Ok(vec![Stmt::Assign {
                        target: lower_expr(&a.left, builtins, env)?,
                        value: lower_expr(&a.right, builtins, env)?,
                    }])
                }
                SynExpr::Binary(ExprBinary { left, op, right, .. }) if assign_arith(op).is_some() => {
                    if try_parse_collective(right, builtins, wg_x, env)?.is_some() {
                        return Err(Error::new(
                            right.span(),
                            "workgroup collectives must be used as a `let` or simple assignment, not `+=`",
                        ));
                    }
                    let arith = assign_arith(op).unwrap();
                    Ok(vec![Stmt::Assign {
                        target: lower_expr(left, builtins, env)?,
                        value: Expr::Binary {
                            op: arith,
                            left: Box::new(lower_expr(left, builtins, env)?),
                            right: Box::new(lower_expr(right, builtins, env)?),
                        },
                    }])
                }
                SynExpr::If(i) => {
                    let cond = lower_expr(&i.cond, builtins, env)?;
                    let then_body = lower_block_from_expr_block(&i.then_branch, builtins, env, wg_x)?;
                    let else_body = match &i.else_branch {
                        Some((_, else_e)) => match else_e.as_ref() {
                            SynExpr::Block(b) => Some(lower_block(&b.block.stmts, builtins, env, wg_x)?),
                            SynExpr::If(_) => Some(lower_stmt(
                                &SynStmt::Expr(else_e.as_ref().clone(), None),
                                builtins,
                                env,
                                wg_x,
                            )?),
                            other => return Err(Error::new(other.span(), "unsupported else branch in #[compute]")),
                        },
                        None => None,
                    };
                    Ok(vec![Stmt::If {
                        cond,
                        then_body,
                        else_body,
                    }])
                }
                SynExpr::While(w) => Ok(vec![Stmt::While {
                    cond: lower_expr(&w.cond, builtins, env)?,
                    body: lower_block_from_expr_block(&w.body, builtins, env, wg_x)?,
                }]),
                SynExpr::ForLoop(f) => {
                    let Pat::Ident(var) = f.pat.as_ref() else {
                        return Err(Error::new(f.pat.span(), "for-loop variable must be a plain identifier"));
                    };
                    let SynExpr::Range(range) = f.expr.as_ref() else {
                        return Err(Error::new(
                            f.expr.span(),
                            "only `for i in start..end` ranges are supported",
                        ));
                    };
                    if !matches!(range.limits, syn::RangeLimits::HalfOpen(_)) {
                        return Err(Error::new(
                            f.expr.span(),
                            "only half-open `start..end` ranges are supported",
                        ));
                    }
                    let start = range
                        .start
                        .as_ref()
                        .ok_or_else(|| Error::new(f.expr.span(), "range start required"))?;
                    let end = range
                        .end
                        .as_ref()
                        .ok_or_else(|| Error::new(f.expr.span(), "range end required"))?;
                    env.insert(var.ident.to_string(), "uint".into());
                    Ok(vec![Stmt::ForRange {
                        var: var.ident.to_string(),
                        start: lower_expr(start, builtins, env)?,
                        end: lower_expr(end, builtins, env)?,
                        body: lower_block_from_expr_block(&f.body, builtins, env, wg_x)?,
                    }])
                }
                SynExpr::Return(r) => Ok(vec![Stmt::Return {
                    value: r.expr.as_ref().map(|e| lower_expr(e, builtins, env)).transpose()?,
                }]),
                other => {
                    if let Some(collective) = try_parse_collective(other, builtins, wg_x, env)? {
                        return match collective {
                            Collective::Softmax {
                                n,
                                buf,
                                base,
                                count,
                                scratch,
                            } => Ok(vec![Stmt::WorkgroupSoftmax {
                                n,
                                buf,
                                base,
                                count,
                                scratch,
                            }]),
                            Collective::Reduce { .. } => Err(Error::new(
                                other.span(),
                                "gpu::workgroup_sum/workgroup_max must be used as a `let` or assignment",
                            )),
                        };
                    }
                    Ok(vec![Stmt::Expr(lower_expr(other, builtins, env)?)])
                }
            }
        }
        SynStmt::Item(item) => Err(Error::new(
            item.span(),
            "nested items are not supported inside #[compute] kernels",
        )),
        SynStmt::Macro(m) => Err(Error::new(
            m.span(),
            "macros are not supported inside #[compute] kernels",
        )),
    }
}

fn lower_block_from_expr_block(
    block: &syn::Block,
    builtins: &mut BuiltinMask,
    env: &mut std::collections::HashMap<String, String>,
    wg_x: u32,
) -> Result<Vec<Stmt>, Error> {
    lower_block(&block.stmts, builtins, env, wg_x)
}

enum Collective {
    Reduce {
        op: WorkgroupReduceOp,
        n: u32,
        val: Expr,
        scratch: String,
    },
    Softmax {
        n: u32,
        buf: String,
        base: Expr,
        count: Expr,
        scratch: String,
    },
}

fn lower_collective_assign(dest: Expr, collective: Collective, span: proc_macro2::Span) -> Result<Stmt, Error> {
    match collective {
        Collective::Reduce { op, n, val, scratch } => Ok(Stmt::WorkgroupReduce {
            op,
            n,
            val,
            scratch,
            dest,
        }),
        Collective::Softmax { .. } => Err(Error::new(
            span,
            "gpu::workgroup_softmax_in_place is a statement, not a value",
        )),
    }
}

fn peel_ref(expr: &SynExpr) -> &SynExpr {
    match expr {
        SynExpr::Reference(r) => peel_ref(&r.expr),
        SynExpr::Paren(p) => peel_ref(&p.expr),
        other => other,
    }
}

fn ident_from_expr(expr: &SynExpr, what: &str) -> Result<String, Error> {
    match peel_ref(expr) {
        SynExpr::Path(p) if p.path.get_ident().is_some() => Ok(p.path.get_ident().unwrap().to_string()),
        other => Err(Error::new(other.span(), format!("{what} must be a simple identifier"))),
    }
}

fn path_const_n(path: &syn::Path) -> Result<Option<u32>, Error> {
    let last = path.segments.last().unwrap();
    match &last.arguments {
        PathArguments::None => Ok(None),
        PathArguments::AngleBracketed(ab) if ab.args.len() == 1 => match &ab.args[0] {
            GenericArgument::Const(SynExpr::Lit(ExprLit { lit: Lit::Int(i), .. })) => {
                let n: u32 = i.base10_parse()?;
                if n == 0 || !n.is_power_of_two() {
                    return Err(Error::new(
                        i.span(),
                        "workgroup collective N must be a power of two greater than zero",
                    ));
                }
                Ok(Some(n))
            }
            other => Err(Error::new(
                other.span(),
                "workgroup collective N must be a power-of-two integer literal",
            )),
        },
        PathArguments::AngleBracketed(ab) => Err(Error::new(
            ab.span(),
            "workgroup collectives take a single const generic `::<N>`",
        )),
        PathArguments::Parenthesized(p) => Err(Error::new(p.span(), "unexpected path arguments")),
    }
}

fn resolve_collective_n(n: Option<u32>, wg_x: u32, span: proc_macro2::Span) -> Result<u32, Error> {
    let n = n.unwrap_or(wg_x);
    if n == 0 || !n.is_power_of_two() {
        return Err(Error::new(
            span,
            "workgroup collective N must be a power of two (use turbofish `::<N>` or a power-of-two workgroup_size.x)",
        ));
    }
    Ok(n)
}

fn require_tensor_method(
    receiver: &SynExpr,
    env: &std::collections::HashMap<String, String>,
    method: &str,
) -> Result<(), Error> {
    let peeled = peel_ref(receiver);
    if let SynExpr::Path(p) = peeled {
        if let Some(id) = p.path.get_ident() {
            if env.get(&id.to_string()).is_some_and(|ty| is_tensor_env_ty(ty)) {
                return Ok(());
            }
        }
    }
    Err(Error::new(
        receiver.span(),
        format!(".{method}() is only valid on gpu::Tensor / TensorMut / TensorWrite parameters"),
    ))
}

fn try_parse_collective(
    expr: &SynExpr,
    builtins: &mut BuiltinMask,
    wg_x: u32,
    env: &std::collections::HashMap<String, String>,
) -> Result<Option<Collective>, Error> {
    let SynExpr::Call(ExprCall { func, args, .. }) = expr else {
        return Ok(None);
    };
    let SynExpr::Path(p) = func.as_ref() else {
        return Ok(None);
    };
    let segs: Vec<String> = p.path.segments.iter().map(|s| s.ident.to_string()).collect();
    let segs_str: Vec<&str> = segs.iter().map(String::as_str).collect();
    let kind = match segs_str.as_slice() {
        ["workgroup_sum"] | ["gpu", "workgroup_sum"] | ["goldy", "gpu", "workgroup_sum"] => {
            Some(WorkgroupReduceOp::Sum)
        }
        ["workgroup_max"] | ["gpu", "workgroup_max"] | ["goldy", "gpu", "workgroup_max"] => {
            Some(WorkgroupReduceOp::Max)
        }
        ["workgroup_softmax_in_place"]
        | ["gpu", "workgroup_softmax_in_place"]
        | ["goldy", "gpu", "workgroup_softmax_in_place"] => None,
        _ => return Ok(None),
    };
    builtins.local_id = true;
    let n = resolve_collective_n(path_const_n(&p.path)?, wg_x, p.path.span())?;
    if segs_str.last() == Some(&"workgroup_softmax_in_place") {
        if args.len() != 4 {
            return Err(Error::new(
                expr.span(),
                "gpu::workgroup_softmax_in_place::<N>(buf, base, count, scratch)",
            ));
        }
        let buf = ident_from_expr(&args[0], "softmax buffer")?;
        let scratch = ident_from_expr(&args[3], "softmax scratch")?;
        return Ok(Some(Collective::Softmax {
            n,
            buf,
            base: lower_expr(&args[1], builtins, env)?,
            count: lower_expr(&args[2], builtins, env)?,
            scratch,
        }));
    }
    if args.len() != 2 {
        return Err(Error::new(
            expr.span(),
            "gpu::workgroup_sum/workgroup_max::<N>(val, scratch)",
        ));
    }
    let op = kind.expect("reduce op");
    Ok(Some(Collective::Reduce {
        op,
        n,
        val: lower_expr(&args[0], builtins, env)?,
        scratch: ident_from_expr(&args[1], "workgroup reduce scratch")?,
    }))
}

fn lower_expr(
    expr: &SynExpr,
    builtins: &mut BuiltinMask,
    env: &std::collections::HashMap<String, String>,
) -> Result<Expr, Error> {
    match expr {
        SynExpr::Lit(ExprLit { lit, .. }) => match lit {
            Lit::Int(i) => {
                if i.suffix() == "i32" {
                    Ok(Expr::LitI32(i.base10_parse()?))
                } else if i.suffix() == "f32" {
                    Ok(Expr::LitF32(i.base10_parse::<f32>()?))
                } else {
                    Ok(Expr::LitU32(i.base10_parse()?))
                }
            }
            Lit::Float(f) => Ok(Expr::LitF32(f.base10_parse()?)),
            Lit::Bool(b) => Ok(Expr::LitBool(b.value())),
            _ => Err(Error::new(lit.span(), "unsupported literal")),
        },
        SynExpr::Path(ExprPath { path, .. }) if path.get_ident().is_some() => {
            Ok(Expr::Var(path.get_ident().unwrap().to_string()))
        }
        SynExpr::Field(ExprField { base, member, .. }) => {
            let field = match member {
                syn::Member::Named(id) => id.to_string(),
                syn::Member::Unnamed(i) => i.index.to_string(),
            };
            Ok(Expr::Field {
                base: Box::new(lower_expr(base, builtins, env)?),
                field,
            })
        }
        SynExpr::Index(ExprIndex { expr, index, .. }) => Ok(Expr::Index {
            base: Box::new(lower_expr(expr, builtins, env)?),
            index: Box::new(lower_expr(index, builtins, env)?),
        }),
        SynExpr::Binary(ExprBinary { left, op, right, .. }) => Ok(Expr::Binary {
            op: map_binop(op)?,
            left: Box::new(lower_expr(left, builtins, env)?),
            right: Box::new(lower_expr(right, builtins, env)?),
        }),
        SynExpr::Unary(ExprUnary { op, expr, .. }) => Ok(Expr::Unary {
            op: map_unary(op)?,
            expr: Box::new(lower_expr(expr, builtins, env)?),
        }),
        SynExpr::Paren(p) => lower_expr(&p.expr, builtins, env),
        SynExpr::Group(g) => lower_expr(&g.expr, builtins, env),
        SynExpr::Cast(c) => Ok(Expr::Cast {
            expr: Box::new(lower_expr(&c.expr, builtins, env)?),
            ty: type_to_slang_name(&c.ty)?,
        }),
        SynExpr::Call(ExprCall { func, args, .. }) => lower_call(func, args, builtins, env),
        SynExpr::MethodCall(ExprMethodCall {
            receiver, method, args, ..
        }) => {
            if method == "len" && args.is_empty() {
                Ok(Expr::Len {
                    base: Box::new(lower_expr(receiver, builtins, env)?),
                })
            } else if method == "dim" && args.len() == 1 {
                require_tensor_method(receiver, env, "dim")?;
                Ok(Expr::Dim {
                    base: Box::new(lower_expr(receiver, builtins, env)?),
                    axis: Box::new(lower_expr(&args[0], builtins, env)?),
                })
            } else if method == "rank" && args.is_empty() {
                require_tensor_method(receiver, env, "rank")?;
                Ok(Expr::Rank {
                    base: Box::new(lower_expr(receiver, builtins, env)?),
                })
            } else {
                Err(Error::new(
                    method.span(),
                    format!("unsupported method `{method}` in #[compute] kernel"),
                ))
            }
        }
        SynExpr::Reference(_) => Err(Error::new(
            expr.span(),
            "references are only allowed on kernel resource parameters",
        )),
        SynExpr::Closure(_) => Err(Error::new(
            expr.span(),
            "closures are not supported in #[compute] kernels",
        )),
        SynExpr::Try(_) | SynExpr::Async(_) | SynExpr::Await(_) => Err(Error::new(
            expr.span(),
            "async/try/await are not supported in #[compute] kernels",
        )),
        SynExpr::Macro(_) => Err(Error::new(
            expr.span(),
            "macros are not supported in #[compute] kernels",
        )),
        other => Err(Error::new(
            other.span(),
            "unsupported expression in #[compute] kernel GPU dialect",
        )),
    }
}

fn lower_call(
    func: &SynExpr,
    args: &syn::punctuated::Punctuated<SynExpr, syn::Token![,]>,
    builtins: &mut BuiltinMask,
    env: &std::collections::HashMap<String, String>,
) -> Result<Expr, Error> {
    let path = match func {
        SynExpr::Path(p) => &p.path,
        _ => return Err(Error::new(func.span(), "only simple function calls are supported")),
    };
    let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    let segs_str: Vec<&str> = segs.iter().map(String::as_str).collect();
    let builtin =
        match segs_str.as_slice() {
            ["gpu", "global_id"] | ["goldy", "gpu", "global_id"] => {
                builtins.global_id = true;
                BuiltinFn::GlobalId
            }
            ["gpu", "local_id"] | ["goldy", "gpu", "local_id"] => {
                builtins.local_id = true;
                BuiltinFn::LocalId
            }
            ["gpu", "workgroup_id"] | ["goldy", "gpu", "workgroup_id"] => {
                builtins.workgroup_id = true;
                BuiltinFn::WorkgroupId
            }
            ["gpu", "workgroup_size"] | ["goldy", "gpu", "workgroup_size"] => BuiltinFn::WorkgroupSize,
            ["abs"] | ["gpu", "abs"] | ["goldy", "gpu", "abs"] => BuiltinFn::Abs,
            ["min"] | ["gpu", "min"] | ["goldy", "gpu", "min"] => BuiltinFn::Min,
            ["max"] | ["gpu", "max"] | ["goldy", "gpu", "max"] => BuiltinFn::Max,
            ["floor"] | ["gpu", "floor"] | ["goldy", "gpu", "floor"] => BuiltinFn::Floor,
            ["ceil"] | ["gpu", "ceil"] | ["goldy", "gpu", "ceil"] => BuiltinFn::Ceil,
            ["sqrt"] | ["gpu", "sqrt"] | ["goldy", "gpu", "sqrt"] => BuiltinFn::Sqrt,
            ["sin"] | ["gpu", "sin"] | ["goldy", "gpu", "sin"] => BuiltinFn::Sin,
            ["cos"] | ["gpu", "cos"] | ["goldy", "gpu", "cos"] => BuiltinFn::Cos,
            ["exp"] | ["gpu", "exp"] | ["goldy", "gpu", "exp"] => BuiltinFn::Exp,
            ["log"] | ["gpu", "log"] | ["goldy", "gpu", "log"] => BuiltinFn::Log,
            ["pow"] | ["gpu", "pow"] | ["goldy", "gpu", "pow"] => BuiltinFn::Pow,
            ["workgroup_barrier"] | ["gpu", "workgroup_barrier"] | ["goldy", "gpu", "workgroup_barrier"] => {
                BuiltinFn::WorkgroupBarrier
            }
            ["workgroup_array"] | ["gpu", "workgroup_array"] | ["goldy", "gpu", "workgroup_array"] => {
                return Err(Error::new(
                    path.span(),
                    "gpu::workgroup_array must be bound as `let mut name = gpu::workgroup_array::<T, N>()`",
                ))
            }
            ["workgroup_sum"]
            | ["gpu", "workgroup_sum"]
            | ["goldy", "gpu", "workgroup_sum"]
            | ["workgroup_max"]
            | ["gpu", "workgroup_max"]
            | ["goldy", "gpu", "workgroup_max"] => return Err(Error::new(
                path.span(),
                "gpu::workgroup_sum/workgroup_max must be used as a `let` or assignment, not nested in an expression",
            )),
            ["workgroup_softmax_in_place"]
            | ["gpu", "workgroup_softmax_in_place"]
            | ["goldy", "gpu", "workgroup_softmax_in_place"] => {
                return Err(Error::new(
                    path.span(),
                    "gpu::workgroup_softmax_in_place is a statement, not an expression",
                ))
            }
            ["length"] | ["gpu", "length"] | ["goldy", "gpu", "length"] => BuiltinFn::Length,
            ["float2"] | ["gpu", "float2"] | ["goldy", "gpu", "float2"] => BuiltinFn::Float2,
            ["float3"] | ["gpu", "float3"] | ["goldy", "gpu", "float3"] => BuiltinFn::Float3,
            ["float4"] | ["gpu", "float4"] | ["goldy", "gpu", "float4"] => BuiltinFn::Float4,
            ["uint2"] | ["gpu", "uint2"] | ["goldy", "gpu", "uint2"] => BuiltinFn::Uint2,
            other => {
                return Err(Error::new(
                    path.span(),
                    format!(
                        "unsupported call `{}` in #[compute] kernel; use gpu::* builtins or selected math intrinsics",
                        other.join("::")
                    ),
                ))
            }
        };
    let mut lowered_args = Vec::new();
    for a in args {
        lowered_args.push(lower_expr(a, builtins, env)?);
    }
    Ok(Expr::Call {
        func: builtin,
        args: lowered_args,
    })
}

fn assign_arith(op: &SynBinOp) -> Option<BinOp> {
    match op {
        SynBinOp::AddAssign(_) => Some(BinOp::Add),
        SynBinOp::SubAssign(_) => Some(BinOp::Sub),
        SynBinOp::MulAssign(_) => Some(BinOp::Mul),
        SynBinOp::DivAssign(_) => Some(BinOp::Div),
        SynBinOp::RemAssign(_) => Some(BinOp::Rem),
        _ => None,
    }
}

fn map_binop(op: &SynBinOp) -> Result<BinOp, Error> {
    Ok(match op {
        SynBinOp::Add(_) => BinOp::Add,
        SynBinOp::Sub(_) => BinOp::Sub,
        SynBinOp::Mul(_) => BinOp::Mul,
        SynBinOp::Div(_) => BinOp::Div,
        SynBinOp::Rem(_) => BinOp::Rem,
        SynBinOp::Eq(_) => BinOp::Eq,
        SynBinOp::Ne(_) => BinOp::Ne,
        SynBinOp::Lt(_) => BinOp::Lt,
        SynBinOp::Le(_) => BinOp::Le,
        SynBinOp::Gt(_) => BinOp::Gt,
        SynBinOp::Ge(_) => BinOp::Ge,
        SynBinOp::And(_) => BinOp::And,
        SynBinOp::Or(_) => BinOp::Or,
        SynBinOp::BitAnd(_) => BinOp::BitAnd,
        SynBinOp::BitOr(_) => BinOp::BitOr,
        SynBinOp::BitXor(_) => BinOp::BitXor,
        SynBinOp::Shl(_) => BinOp::Shl,
        SynBinOp::Shr(_) => BinOp::Shr,
        other => {
            return Err(Error::new(
                other.span(),
                "unsupported binary operator in #[compute] kernel",
            ))
        }
    })
}

fn map_unary(op: &UnOp) -> Result<UnaryOp, Error> {
    Ok(match op {
        UnOp::Neg(_) => UnaryOp::Neg,
        UnOp::Not(_) => UnaryOp::Not,
        other => {
            return Err(Error::new(
                other.span(),
                "unsupported unary operator in #[compute] kernel",
            ))
        }
    })
}

fn infer_slang_ty(expr: &Expr, env: &std::collections::HashMap<String, String>) -> Option<String> {
    match expr {
        Expr::Field { base, field }
            if (field == "x" || field == "y" || field == "z")
                && matches!(
                    base.as_ref(),
                    Expr::Call {
                        func: BuiltinFn::GlobalId | BuiltinFn::LocalId | BuiltinFn::WorkgroupId,
                        ..
                    }
                ) =>
        {
            Some("uint".into())
        }
        Expr::Field { base, field } if field == "xy" || field == "zw" => {
            let base_ty = infer_slang_ty(base, env)?;
            if base_ty == "uint3"
                || base_ty == "uint4"
                || base_ty == "ThreadId"
                || base_ty == "GroupThreadId"
                || base_ty == "GroupId"
            {
                Some("uint2".into())
            } else if base_ty.starts_with("float") {
                Some("float2".into())
            } else {
                None
            }
        }
        Expr::Field { base, field } if field == "x" || field == "y" || field == "z" || field == "w" => {
            let base_ty = infer_slang_ty(base, env)?;
            if base_ty.starts_with("uint")
                || base_ty == "ThreadId"
                || base_ty == "GroupThreadId"
                || base_ty == "GroupId"
            {
                Some("uint".into())
            } else if base_ty.starts_with("float") {
                Some("float".into())
            } else {
                None
            }
        }
        Expr::Call {
            func: BuiltinFn::GlobalId,
            ..
        } => Some("ThreadId".into()),
        Expr::Call {
            func: BuiltinFn::LocalId,
            ..
        } => Some("GroupThreadId".into()),
        Expr::Call {
            func: BuiltinFn::WorkgroupId,
            ..
        } => Some("GroupId".into()),
        Expr::Call {
            func: BuiltinFn::Float2,
            ..
        } => Some("float2".into()),
        Expr::Call {
            func: BuiltinFn::Float3,
            ..
        } => Some("float3".into()),
        Expr::Call {
            func: BuiltinFn::Float4,
            ..
        } => Some("float4".into()),
        Expr::Call {
            func: BuiltinFn::Uint2, ..
        } => Some("uint2".into()),
        Expr::Call {
            func:
                BuiltinFn::Sin
                | BuiltinFn::Cos
                | BuiltinFn::Exp
                | BuiltinFn::Log
                | BuiltinFn::Pow
                | BuiltinFn::Length
                | BuiltinFn::Abs
                | BuiltinFn::Min
                | BuiltinFn::Max
                | BuiltinFn::Floor
                | BuiltinFn::Ceil
                | BuiltinFn::Sqrt,
            ..
        } => Some("float".into()),
        Expr::LitU32(_) => Some("uint".into()),
        Expr::LitI32(_) => Some("int".into()),
        Expr::LitF32(_) => Some("float".into()),
        Expr::LitBool(_) => Some("bool".into()),
        Expr::Len { .. } | Expr::Dim { .. } | Expr::Rank { .. } => Some("uint".into()),
        Expr::Var(name) => env.get(name).cloned(),
        Expr::Index { base, .. } => {
            let base_ty = infer_slang_ty(base, env)?;
            unwrap_generic(&base_ty, "BufRO<")
                .or_else(|| unwrap_generic(&base_ty, "Scattered<"))
                .or_else(|| unwrap_generic(&base_ty, "Tensor<"))
                .or_else(|| unwrap_generic(&base_ty, "TensorMut<"))
                .or_else(|| unwrap_generic(&base_ty, "TensorWrite<"))
                .or_else(|| unwrap_generic(&base_ty, "DirectSpatial<"))
                .or_else(|| unwrap_generic(&base_ty, "groupshared<"))
                .map(str::to_string)
        }
        Expr::Cast { ty, .. } => Some(ty.clone()),
        Expr::Binary { left, right, .. } => {
            let l = infer_slang_ty(left, env);
            let r = infer_slang_ty(right, env);
            match (l.as_deref(), r.as_deref()) {
                (Some(a), _) if is_vector_ty(a) => Some(a.to_string()),
                (_, Some(b)) if is_vector_ty(b) => Some(b.to_string()),
                (Some("float"), _) | (_, Some("float")) => Some("float".into()),
                (Some("uint"), Some("uint")) => Some("uint".into()),
                (Some(a), _) => Some(a.to_string()),
                (_, Some(b)) => Some(b.to_string()),
                _ => None,
            }
        }
        Expr::Unary { expr, .. } => infer_slang_ty(expr, env),
        _ => None,
    }
}

fn is_vector_ty(ty: &str) -> bool {
    matches!(
        ty,
        "float2" | "float3" | "float4" | "uint2" | "uint3" | "uint4" | "int2" | "int3" | "int4"
    )
}

fn unwrap_generic<'a>(ty: &'a str, prefix: &str) -> Option<&'a str> {
    if ty.starts_with(prefix) && ty.ends_with('>') {
        Some(&ty[prefix.len()..ty.len() - 1])
    } else {
        None
    }
}

fn parse_workgroup_array_init(expr: &SynExpr) -> Result<Option<(String, u32)>, Error> {
    let SynExpr::Call(ExprCall { func, args, .. }) = expr else {
        return Ok(None);
    };
    let SynExpr::Path(p) = func.as_ref() else {
        return Ok(None);
    };
    let segs: Vec<String> = p.path.segments.iter().map(|s| s.ident.to_string()).collect();
    let segs_str: Vec<&str> = segs.iter().map(String::as_str).collect();
    if !matches!(
        segs_str.as_slice(),
        ["workgroup_array"] | ["gpu", "workgroup_array"] | ["goldy", "gpu", "workgroup_array"]
    ) {
        return Ok(None);
    }
    if !args.is_empty() {
        return Err(Error::new(
            expr.span(),
            "gpu::workgroup_array takes no runtime arguments; use turbofish `<T, N>`",
        ));
    }
    let last = p.path.segments.last().unwrap();
    let PathArguments::AngleBracketed(ab) = &last.arguments else {
        return Err(Error::new(
            p.path.span(),
            "gpu::workgroup_array requires turbofish `<T, N>` (for example `::<f32, 256>`)",
        ));
    };
    if ab.args.len() != 2 {
        return Err(Error::new(
            p.path.span(),
            "gpu::workgroup_array requires exactly two generic arguments `<T, N>`",
        ));
    }
    let elem = match &ab.args[0] {
        GenericArgument::Type(ty) => type_to_slang_name(ty)?,
        other => {
            return Err(Error::new(
                other.span(),
                "workgroup array element type must be f32, u32, i32, or bool",
            ))
        }
    };
    if !matches!(elem.as_str(), "float" | "uint" | "int" | "bool") {
        return Err(Error::new(
            ab.args[0].span(),
            "workgroup array element type must be f32, u32, i32, or bool",
        ));
    }
    let len = match &ab.args[1] {
        GenericArgument::Const(SynExpr::Lit(ExprLit { lit: Lit::Int(i), .. })) => {
            let n: u32 = i.base10_parse()?;
            if n == 0 {
                return Err(Error::new(i.span(), "workgroup array length must be > 0"));
            }
            n
        }
        other => {
            return Err(Error::new(
                other.span(),
                "workgroup array length must be an integer literal",
            ))
        }
    };
    Ok(Some((elem, len)))
}

fn workgroup_array_in_nested_scope(stmts: &[Stmt]) -> bool {
    fn nested(stmts: &[Stmt]) -> bool {
        stmts.iter().any(|s| match s {
            Stmt::WorkgroupArray { .. } => true,
            Stmt::If {
                then_body, else_body, ..
            } => nested(then_body) || else_body.as_ref().is_some_and(|e| nested(e)),
            Stmt::While { body, .. } | Stmt::ForRange { body, .. } => nested(body),
            _ => false,
        })
    }
    stmts.iter().any(|s| match s {
        Stmt::WorkgroupArray { .. } => false,
        Stmt::If {
            then_body, else_body, ..
        } => nested(then_body) || else_body.as_ref().is_some_and(|e| nested(e)),
        Stmt::While { body, .. } | Stmt::ForRange { body, .. } => nested(body),
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse_quote;

    fn parse_attr(tokens: Attribute) -> Result<TensorShapeSpec, Error> {
        parse_tensor_attr(&tokens)
    }

    #[test]
    fn parse_symbolic_and_wildcard_and_exact() {
        let spec = parse_attr(parse_quote! { #[tensor(shape = [vocab, dim])] }).unwrap();
        assert_eq!(
            spec.dims,
            vec![
                TensorDimSpec::Symbol("vocab".into()),
                TensorDimSpec::Symbol("dim".into()),
            ]
        );
        let spec = parse_attr(parse_quote! { #[tensor(shape = [_, 4])] }).unwrap();
        assert_eq!(spec.dims, vec![TensorDimSpec::Any, TensorDimSpec::Exact(4)]);
        let spec = parse_attr(parse_quote! { #[tensor(shape = [])] }).unwrap();
        assert!(spec.dims.is_empty());
    }

    #[test]
    fn reject_rank_above_four() {
        let err = parse_attr(parse_quote! { #[tensor(shape = [a, b, c, d, e])] }).unwrap_err();
        assert!(err.to_string().contains("at most 4"), "{err}");
    }

    #[test]
    fn reject_malformed_dim() {
        let err = parse_attr(parse_quote! { #[tensor(shape = [true])] }).unwrap_err();
        assert!(err.to_string().contains("tensor dim must be"), "{err}");
    }

    #[test]
    fn reject_shape_on_buffer_param() {
        let err = expand(
            quote! { workgroup_size = [64, 1, 1] },
            quote! {
                fn k(#[tensor(shape = [n])] x: &[f32]) {}
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("only valid on gpu::Tensor"), "{err}");
    }

    #[test]
    fn generated_abi_embeds_shape_spec() {
        let tokens = expand(
            quote! { workgroup_size = [64, 1, 1] },
            quote! {
                fn rmsnorm(
                    #[tensor(shape = [dim])] x: gpu::Tensor<f32>,
                    #[tensor(shape = [dim])] weight: gpu::Tensor<f32>,
                    #[tensor(shape = [dim])] out: gpu::TensorWrite<f32>,
                ) {
                    let i = gpu::global_id().x;
                    if i < x.len() {
                        out[i] = x[i] * weight[i];
                    }
                }
            },
        )
        .unwrap();
        let text = tokens.to_string();
        assert!(text.contains("TensorShapeSpec"), "{text}");
        assert!(text.contains("TensorDimSpec :: Symbol"), "{text}");
        assert!(text.contains("check_tensor_view"), "{text}");
    }

    #[test]
    fn generated_module_retains_definition() {
        let tokens = expand(
            quote! { workgroup_size = [64, 1, 1] },
            quote! {
                fn scale(input: &[f32], output: gpu::Scattered<f32>, factor: f32) {
                    let i = gpu::global_id().x;
                    output[i] = input[i] * factor + 1e40;
                }
            },
        )
        .unwrap();
        let text = tokens.to_string();
        assert!(text.contains("pub fn definition ()"), "{text}");
        assert!(text.contains("definition : Some (definition)"), "{text}");
        assert!(text.contains("ir :: Stmt :: Let"), "{text}");
        assert!(text.contains("ir :: BinOp :: Mul"), "{text}");
        let inf_bits = f32::INFINITY.to_bits();
        assert!(text.contains(&format!("f32 :: from_bits ({inf_bits}u32)")), "{text}");
    }

    #[test]
    fn unannotated_tensor_has_no_shape_spec() {
        let tokens = expand(
            quote! { workgroup_size = [64, 1, 1] },
            quote! {
                fn copy_view(src: gpu::Tensor<f32>, dst: gpu::TensorWrite<f32>) {
                    let i = gpu::global_id().x;
                    if i < dst.len() {
                        dst[i] = src[i];
                    }
                }
            },
        )
        .unwrap();
        let text = tokens.to_string();
        assert!(text.contains("shape_spec : None"), "{text}");
        assert!(!text.contains("TensorDimSpec"), "{text}");
    }
}
