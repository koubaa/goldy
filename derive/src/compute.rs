//! `#[goldy_derive::compute]` — compile-time Rust GPU dialect → Slang + KernelAbi.

use goldy_shader_ir::{
    emit_canonical_compute_source, BinOp, BuiltinFn, BuiltinMask, ElementType, Expr, KernelParam, ParamCategory,
    ScalarType, ShaderKernel, SourceMap, Stmt, UnaryOp,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{
    parse2, Attribute, BinOp as SynBinOp, Error, Expr as SynExpr, ExprBinary, ExprCall, ExprField, ExprIndex, ExprLit,
    ExprMethodCall, ExprPath, ExprUnary, FnArg, ItemFn, Lit, Meta, Pat, PatType, ReturnType, Stmt as SynStmt, Type,
    UnOp,
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

    for input in &func.sig.inputs {
        let FnArg::Typed(PatType { pat, ty, .. }) = input else {
            return Err(Error::new(input.span(), "#[compute] does not support `self`"));
        };
        let Pat::Ident(name) = pat.as_ref() else {
            return Err(Error::new(pat.span(), "kernel parameters must be plain identifiers"));
        };
        let pname = name.ident.to_string();
        let pident = &name.ident;
        match classify_param_type(ty)? {
            ClassifiedParam::BufferRead(elem) => {
                params.push(KernelParam::buffer_read(&pname, elem));
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
                record_args.push(quote! { #pident: &impl ::goldy::kernel::KernelBindable });
                bind_stmts.push(quote! {
                    start = ::goldy::kernel::KernelBindable::__goldy_bind_kernel(
                        #pident,
                        start,
                        ::goldy::NodeAccess::Write,
                    );
                });
            }
            ClassifiedParam::Uniform(type_name) => {
                params.push(KernelParam {
                    name: pname,
                    category: ParamCategory::Uniform,
                    access: Some(goldy_shader_ir::AccessKind::Read),
                    scalar: None,
                    slang_type: type_name,
                    stride_bytes: None,
                });
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
        }
    }

    let body_stmts = lower_block(&func.block.stmts, &mut builtins)?;

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
    };

    let def = emit_canonical_compute_source(&kernel);
    let slang = &def.source.canonical_slang;
    let entry = &def.entry;
    let [wx, wy, wz] = def.workgroup_size;
    let abi_version = def.abi_version;
    let rust_file = &def.source_map.rust_file;
    let rust_line = def.source_map.rust_line;

    let param_tokens: Vec<_> = def
        .params
        .iter()
        .map(|p| {
            let name = &p.name;
            let slang_type = &p.slang_type;
            let category = match p.category {
                ParamCategory::BufferRead => quote! { ::goldy::kernel::ParamCategory::BufferRead },
                ParamCategory::BufferReadWrite => {
                    quote! { ::goldy::kernel::ParamCategory::BufferReadWrite }
                }
                ParamCategory::BufferWrite => quote! { ::goldy::kernel::ParamCategory::BufferWrite },
                ParamCategory::Uniform => quote! { ::goldy::kernel::ParamCategory::Uniform },
                ParamCategory::Scalar => quote! { ::goldy::kernel::ParamCategory::Scalar },
            };
            let access = match p.access {
                Some(goldy_shader_ir::AccessKind::Read) => {
                    quote! { Some(::goldy::kernel::AccessKind::Read) }
                }
                Some(goldy_shader_ir::AccessKind::Write) => {
                    quote! { Some(::goldy::kernel::AccessKind::Write) }
                }
                Some(goldy_shader_ir::AccessKind::ReadWrite) => {
                    quote! { Some(::goldy::kernel::AccessKind::ReadWrite) }
                }
                None => quote! { None },
            };
            let scalar = match p.scalar {
                Some(ScalarType::U32) => quote! { Some(::goldy::kernel::ScalarType::U32) },
                Some(ScalarType::I32) => quote! { Some(::goldy::kernel::ScalarType::I32) },
                Some(ScalarType::F32) => quote! { Some(::goldy::kernel::ScalarType::F32) },
                Some(ScalarType::Bool) => quote! { Some(::goldy::kernel::ScalarType::Bool) },
                None => quote! { None },
            };
            let stride = match p.stride_bytes {
                Some(s) => quote! { Some(#s) },
                None => quote! { None },
            };
            quote! {
                ::goldy::kernel::KernelParam {
                    name: #name.to_string(),
                    category: #category,
                    access: #access,
                    scalar: #scalar,
                    slang_type: #slang_type.to_string(),
                    stride_bytes: #stride,
                }
            }
        })
        .collect();

    let builtins_tokens = {
        let g = builtins.global_id;
        let l = builtins.local_id;
        let w = builtins.workgroup_id;
        quote! {
            ::goldy::kernel::BuiltinMask {
                global_id: #g,
                local_id: #l,
                workgroup_id: #w,
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
            pub const CANONICAL_SOURCE: &str = #slang;

            #[doc = #docs]
            pub struct #kernel_struct {
                prepared: ::goldy::kernel::PreparedKernel,
            }

            impl #kernel_struct {
                /// Compile (or hit the shader cache) and create a device-scoped pipeline.
                pub fn prepare(device: &::goldy::Device) -> ::core::result::Result<Self, ::goldy::GoldyError> {
                    let def = ::goldy::kernel::KernelDef {
                        source: ::goldy::kernel::KernelSource {
                            canonical_slang: CANONICAL_SOURCE.to_string(),
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
                    };
                    let prepared = ::goldy::kernel::prepare_kernel(device, def).map_err(::goldy::GoldyError::Backend)?;
                    Ok(Self { prepared })
                }

                /// Record a dispatch into `scheme`, binding arguments in declaration order.
                pub fn record<'a>(
                    &'a self,
                    scheme: &'a mut ::goldy::Scheme,
                    label: &'static str,
                    #(#record_args),*
                ) -> ::goldy::kernel::DispatchBuilder<'a> {
                    let mut start = self.prepared.begin_record(scheme, label);
                    #(#bind_stmts)*
                    start.finish()
                }

                pub fn workgroup_size(&self) -> [u32; 3] {
                    self.prepared.workgroup_size()
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

enum ClassifiedParam {
    BufferRead(ElementType),
    BufferReadWrite(ElementType),
    BufferWrite(ElementType),
    Uniform(String),
    Scalar(ScalarType),
}

fn classify_param_type(ty: &Type) -> Result<ClassifiedParam, Error> {
    // gpu::Out<T> / goldy::gpu::Out<T>
    if let Some(inner) =
        match_path_generic(ty, &["gpu", "Out"]).or_else(|| match_path_generic(ty, &["goldy", "gpu", "Out"]))
    {
        let elem = element_from_type(inner)?;
        return Ok(ClassifiedParam::BufferWrite(elem));
    }
    if let Some(inner) =
        match_path_generic(ty, &["gpu", "Uniform"]).or_else(|| match_path_generic(ty, &["goldy", "gpu", "Uniform"]))
    {
        let name = type_to_slang_name(inner)?;
        return Ok(ClassifiedParam::Uniform(name));
    }

    match ty {
        Type::Reference(r) => {
            let elem = element_from_type(&r.elem)?;
            if r.mutability.is_some() {
                Ok(ClassifiedParam::BufferReadWrite(elem))
            } else {
                Ok(ClassifiedParam::BufferRead(elem))
            }
        }
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
            "unsupported kernel parameter type; expected &[T], &mut [T], gpu::Out<T>, gpu::Uniform<T>, or u32/i32/f32/bool",
        )),
    }
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
        Type::Path(p) if p.qself.is_none() => Ok(p
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_else(|| "Unknown".into())),
        _ => Err(Error::new(ty.span(), "unsupported Uniform type")),
    }
}

fn lower_block(stmts: &[SynStmt], builtins: &mut BuiltinMask) -> Result<Vec<Stmt>, Error> {
    let mut out = Vec::new();
    for s in stmts {
        out.push(lower_stmt(s, builtins)?);
    }
    Ok(out)
}

fn lower_stmt(stmt: &SynStmt, builtins: &mut BuiltinMask) -> Result<Stmt, Error> {
    match stmt {
        SynStmt::Local(local) => {
            let Pat::Ident(name) = &local.pat else {
                return Err(Error::new(
                    local.pat.span(),
                    "only simple `let` bindings are supported in #[compute] kernels",
                ));
            };
            let init = local
                .init
                .as_ref()
                .ok_or_else(|| Error::new(local.span(), "let without initializer is unsupported"))?;
            let expr = lower_expr(&init.expr, builtins)?;
            let ty = infer_slang_ty(&expr);
            Ok(Stmt::Let {
                name: name.ident.to_string(),
                mutable: name.mutability.is_some(),
                ty,
                init: expr,
            })
        }
        SynStmt::Expr(expr, semi) => {
            if semi.is_none() {
                // trailing expr — treat as expression statement (or assign)
            }
            match expr {
                SynExpr::Assign(a) => Ok(Stmt::Assign {
                    target: lower_expr(&a.left, builtins)?,
                    value: lower_expr(&a.right, builtins)?,
                }),
                SynExpr::If(i) => {
                    let cond = lower_expr(&i.cond, builtins)?;
                    let then_body = lower_block_from_expr_block(&i.then_branch, builtins)?;
                    let else_body = match &i.else_branch {
                        Some((_, else_e)) => match else_e.as_ref() {
                            SynExpr::Block(b) => Some(lower_block(&b.block.stmts, builtins)?),
                            SynExpr::If(_) => {
                                // else if → wrap as single-stmt else body
                                Some(vec![lower_stmt(
                                    &SynStmt::Expr(else_e.as_ref().clone(), None),
                                    builtins,
                                )?])
                            }
                            other => return Err(Error::new(other.span(), "unsupported else branch in #[compute]")),
                        },
                        None => None,
                    };
                    Ok(Stmt::If {
                        cond,
                        then_body,
                        else_body,
                    })
                }
                SynExpr::While(w) => Ok(Stmt::While {
                    cond: lower_expr(&w.cond, builtins)?,
                    body: lower_block_from_expr_block(&w.body, builtins)?,
                }),
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
                    Ok(Stmt::ForRange {
                        var: var.ident.to_string(),
                        start: lower_expr(start, builtins)?,
                        end: lower_expr(end, builtins)?,
                        body: lower_block_from_expr_block(&f.body, builtins)?,
                    })
                }
                SynExpr::Return(r) => Ok(Stmt::Return {
                    value: r.expr.as_ref().map(|e| lower_expr(e, builtins)).transpose()?,
                }),
                other => Ok(Stmt::Expr(lower_expr(other, builtins)?)),
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

fn lower_block_from_expr_block(block: &syn::Block, builtins: &mut BuiltinMask) -> Result<Vec<Stmt>, Error> {
    lower_block(&block.stmts, builtins)
}

fn lower_expr(expr: &SynExpr, builtins: &mut BuiltinMask) -> Result<Expr, Error> {
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
                base: Box::new(lower_expr(base, builtins)?),
                field,
            })
        }
        SynExpr::Index(ExprIndex { expr, index, .. }) => Ok(Expr::Index {
            base: Box::new(lower_expr(expr, builtins)?),
            index: Box::new(lower_expr(index, builtins)?),
        }),
        SynExpr::Binary(ExprBinary { left, op, right, .. }) => Ok(Expr::Binary {
            op: map_binop(op)?,
            left: Box::new(lower_expr(left, builtins)?),
            right: Box::new(lower_expr(right, builtins)?),
        }),
        SynExpr::Unary(ExprUnary { op, expr, .. }) => Ok(Expr::Unary {
            op: map_unary(op)?,
            expr: Box::new(lower_expr(expr, builtins)?),
        }),
        SynExpr::Paren(p) => lower_expr(&p.expr, builtins),
        SynExpr::Group(g) => lower_expr(&g.expr, builtins),
        SynExpr::Cast(c) => Ok(Expr::Cast {
            expr: Box::new(lower_expr(&c.expr, builtins)?),
            ty: type_to_slang_name(&c.ty)?,
        }),
        SynExpr::Call(ExprCall { func, args, .. }) => lower_call(func, args, builtins),
        SynExpr::MethodCall(ExprMethodCall {
            receiver, method, args, ..
        }) => {
            if method == "len" && args.is_empty() {
                Ok(Expr::Len {
                    base: Box::new(lower_expr(receiver, builtins)?),
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
) -> Result<Expr, Error> {
    let path = match func {
        SynExpr::Path(p) => &p.path,
        _ => return Err(Error::new(func.span(), "only simple function calls are supported")),
    };
    let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    let segs_str: Vec<&str> = segs.iter().map(String::as_str).collect();
    let builtin = match segs_str.as_slice() {
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
        ["abs"] | ["gpu", "abs"] => BuiltinFn::Abs,
        ["min"] | ["gpu", "min"] => BuiltinFn::Min,
        ["max"] | ["gpu", "max"] => BuiltinFn::Max,
        ["floor"] | ["gpu", "floor"] => BuiltinFn::Floor,
        ["ceil"] | ["gpu", "ceil"] => BuiltinFn::Ceil,
        ["sqrt"] | ["gpu", "sqrt"] => BuiltinFn::Sqrt,
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
        lowered_args.push(lower_expr(a, builtins)?);
    }
    Ok(Expr::Call {
        func: builtin,
        args: lowered_args,
    })
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

fn infer_slang_ty(expr: &Expr) -> Option<String> {
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
        Expr::Call {
            func: BuiltinFn::GlobalId | BuiltinFn::LocalId | BuiltinFn::WorkgroupId,
            ..
        } => Some("uint3".into()),
        Expr::LitU32(_) => Some("uint".into()),
        Expr::LitI32(_) => Some("int".into()),
        Expr::LitF32(_) => Some("float".into()),
        Expr::LitBool(_) => Some("bool".into()),
        Expr::Len { .. } => Some("uint".into()),
        _ => None,
    }
}
