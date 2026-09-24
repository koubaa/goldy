//! Rust constructor tokens for shader IR and ABI values, so generated kernels
//! retain their structured definition at runtime.

use goldy_shader_ir::{
    AccessKind, BinOp, BuiltinFn, BuiltinMask, Expr, KernelParam, ParamCategory, ScalarType, ShaderKernel, Stmt,
    TensorDimSpec, TensorShapeSpec, UnaryOp, WorkgroupReduceOp,
};
use proc_macro2::TokenStream;
use quote::quote;

/// `ShaderKernel` literal. `type_decls` is an expression evaluated in the generated code.
pub fn kernel(kernel: &ShaderKernel, type_decls: TokenStream) -> TokenStream {
    let name = &kernel.name;
    let [wx, wy, wz] = kernel.workgroup_size;
    let params = kernel.params.iter().map(param);
    let builtins = builtins(kernel.builtins);
    let body = stmts(&kernel.body);
    let rust_file = &kernel.source_map.rust_file;
    let rust_line = kernel.source_map.rust_line;
    quote! {
        ::goldy::kernel::ShaderKernel {
            name: ::std::string::String::from(#name),
            workgroup_size: [#wx, #wy, #wz],
            params: ::std::vec![#(#params),*],
            builtins: #builtins,
            body: #body,
            source_map: ::goldy::kernel::SourceMap {
                rust_file: ::std::string::String::from(#rust_file),
                rust_line: #rust_line,
            },
            type_decls: #type_decls,
        }
    }
}

pub fn param(p: &KernelParam) -> TokenStream {
    let name = &p.name;
    let slang_type = &p.slang_type;
    let category = match p.category {
        ParamCategory::BufferRead => quote! { ::goldy::kernel::ParamCategory::BufferRead },
        ParamCategory::BufferReadWrite => quote! { ::goldy::kernel::ParamCategory::BufferReadWrite },
        ParamCategory::BufferWrite => quote! { ::goldy::kernel::ParamCategory::BufferWrite },
        ParamCategory::Uniform => quote! { ::goldy::kernel::ParamCategory::Uniform },
        ParamCategory::StorageImage => quote! { ::goldy::kernel::ParamCategory::StorageImage },
        ParamCategory::Scalar => quote! { ::goldy::kernel::ParamCategory::Scalar },
    };
    let access = match p.access {
        Some(AccessKind::Read) => quote! { Some(::goldy::kernel::AccessKind::Read) },
        Some(AccessKind::Write) => quote! { Some(::goldy::kernel::AccessKind::Write) },
        Some(AccessKind::ReadWrite) => quote! { Some(::goldy::kernel::AccessKind::ReadWrite) },
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
    let is_tensor = p.is_tensor;
    let shape_spec = shape_spec(&p.shape_spec);
    quote! {
        ::goldy::kernel::KernelParam {
            name: #name.to_string(),
            category: #category,
            access: #access,
            scalar: #scalar,
            slang_type: #slang_type.to_string(),
            stride_bytes: #stride,
            is_tensor: #is_tensor,
            shape_spec: #shape_spec,
        }
    }
}

pub fn builtins(mask: BuiltinMask) -> TokenStream {
    let g = mask.global_id;
    let l = mask.local_id;
    let w = mask.workgroup_id;
    quote! {
        ::goldy::kernel::BuiltinMask {
            global_id: #g,
            local_id: #l,
            workgroup_id: #w,
        }
    }
}

pub fn shape_spec(spec: &Option<TensorShapeSpec>) -> TokenStream {
    match spec {
        None => quote! { None },
        Some(spec) => {
            let dims = spec.dims.iter().map(|d| match d {
                TensorDimSpec::Any => quote! { ::goldy::kernel::TensorDimSpec::Any },
                TensorDimSpec::Exact(n) => quote! { ::goldy::kernel::TensorDimSpec::Exact(#n) },
                TensorDimSpec::Symbol(name) => {
                    quote! { ::goldy::kernel::TensorDimSpec::Symbol(#name.to_string()) }
                }
            });
            quote! {
                Some(::goldy::kernel::TensorShapeSpec {
                    dims: vec![#(#dims),*],
                })
            }
        }
    }
}

fn string(s: &str) -> TokenStream {
    quote! { ::std::string::String::from(#s) }
}

fn stmts(body: &[Stmt]) -> TokenStream {
    let items = body.iter().map(stmt);
    quote! { ::std::vec![#(#items),*] }
}

fn boxed(e: &Expr) -> TokenStream {
    let e = expr(e);
    quote! { ::std::boxed::Box::new(#e) }
}

fn stmt(s: &Stmt) -> TokenStream {
    match s {
        Stmt::Let {
            name,
            mutable,
            ty,
            init,
        } => {
            let name = string(name);
            let ty = match ty {
                Some(t) => {
                    let t = string(t);
                    quote! { Some(#t) }
                }
                None => quote! { None },
            };
            let init = expr(init);
            quote! { ::goldy::kernel::ir::Stmt::Let { name: #name, mutable: #mutable, ty: #ty, init: #init } }
        }
        Stmt::Assign { target, value } => {
            let target = expr(target);
            let value = expr(value);
            quote! { ::goldy::kernel::ir::Stmt::Assign { target: #target, value: #value } }
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            let cond = expr(cond);
            let then_body = stmts(then_body);
            let else_body = match else_body {
                Some(b) => {
                    let b = stmts(b);
                    quote! { Some(#b) }
                }
                None => quote! { None },
            };
            quote! {
                ::goldy::kernel::ir::Stmt::If { cond: #cond, then_body: #then_body, else_body: #else_body }
            }
        }
        Stmt::While { cond, body } => {
            let cond = expr(cond);
            let body = stmts(body);
            quote! { ::goldy::kernel::ir::Stmt::While { cond: #cond, body: #body } }
        }
        Stmt::ForRange { var, start, end, body } => {
            let var = string(var);
            let start = expr(start);
            let end = expr(end);
            let body = stmts(body);
            quote! {
                ::goldy::kernel::ir::Stmt::ForRange { var: #var, start: #start, end: #end, body: #body }
            }
        }
        Stmt::Return { value } => {
            let value = match value {
                Some(v) => {
                    let v = expr(v);
                    quote! { Some(#v) }
                }
                None => quote! { None },
            };
            quote! { ::goldy::kernel::ir::Stmt::Return { value: #value } }
        }
        Stmt::WorkgroupArray { name, elem, len } => {
            let name = string(name);
            let elem = string(elem);
            quote! { ::goldy::kernel::ir::Stmt::WorkgroupArray { name: #name, elem: #elem, len: #len } }
        }
        Stmt::WorkgroupReduce {
            op,
            n,
            val,
            scratch,
            dest,
        } => {
            let op = match op {
                WorkgroupReduceOp::Sum => quote! { ::goldy::kernel::ir::WorkgroupReduceOp::Sum },
                WorkgroupReduceOp::Max => quote! { ::goldy::kernel::ir::WorkgroupReduceOp::Max },
            };
            let val = expr(val);
            let scratch = string(scratch);
            let dest = expr(dest);
            quote! {
                ::goldy::kernel::ir::Stmt::WorkgroupReduce { op: #op, n: #n, val: #val, scratch: #scratch, dest: #dest }
            }
        }
        Stmt::WorkgroupSoftmax {
            n,
            buf,
            base,
            count,
            scratch,
        } => {
            let buf = string(buf);
            let base = expr(base);
            let count = expr(count);
            let scratch = string(scratch);
            quote! {
                ::goldy::kernel::ir::Stmt::WorkgroupSoftmax {
                    n: #n, buf: #buf, base: #base, count: #count, scratch: #scratch
                }
            }
        }
        Stmt::Expr(e) => {
            let e = expr(e);
            quote! { ::goldy::kernel::ir::Stmt::Expr(#e) }
        }
    }
}

fn expr(e: &Expr) -> TokenStream {
    match e {
        Expr::LitU32(v) => quote! { ::goldy::kernel::ir::Expr::LitU32(#v) },
        Expr::LitI32(v) => quote! { ::goldy::kernel::ir::Expr::LitI32(#v) },
        Expr::LitF32(v) => {
            // Bit pattern: exact, and valid for literals that parsed to infinity.
            let bits = v.to_bits();
            quote! { ::goldy::kernel::ir::Expr::LitF32(::core::primitive::f32::from_bits(#bits)) }
        }
        Expr::LitBool(v) => quote! { ::goldy::kernel::ir::Expr::LitBool(#v) },
        Expr::Var(name) => {
            let name = string(name);
            quote! { ::goldy::kernel::ir::Expr::Var(#name) }
        }
        Expr::Field { base, field } => {
            let base = boxed(base);
            let field = string(field);
            quote! { ::goldy::kernel::ir::Expr::Field { base: #base, field: #field } }
        }
        Expr::Index { base, index } => {
            let base = boxed(base);
            let index = boxed(index);
            quote! { ::goldy::kernel::ir::Expr::Index { base: #base, index: #index } }
        }
        Expr::Len { base } => {
            let base = boxed(base);
            quote! { ::goldy::kernel::ir::Expr::Len { base: #base } }
        }
        Expr::Dim { base, axis } => {
            let base = boxed(base);
            let axis = boxed(axis);
            quote! { ::goldy::kernel::ir::Expr::Dim { base: #base, axis: #axis } }
        }
        Expr::Rank { base } => {
            let base = boxed(base);
            quote! { ::goldy::kernel::ir::Expr::Rank { base: #base } }
        }
        Expr::Binary { op, left, right } => {
            let op = bin_op(*op);
            let left = boxed(left);
            let right = boxed(right);
            quote! { ::goldy::kernel::ir::Expr::Binary { op: #op, left: #left, right: #right } }
        }
        Expr::Unary { op, expr: inner } => {
            let op = match op {
                UnaryOp::Neg => quote! { ::goldy::kernel::ir::UnaryOp::Neg },
                UnaryOp::Not => quote! { ::goldy::kernel::ir::UnaryOp::Not },
                UnaryOp::BitNot => quote! { ::goldy::kernel::ir::UnaryOp::BitNot },
            };
            let inner = boxed(inner);
            quote! { ::goldy::kernel::ir::Expr::Unary { op: #op, expr: #inner } }
        }
        Expr::Call { func, args } => {
            let func = builtin_fn(*func);
            let args = args.iter().map(expr);
            quote! { ::goldy::kernel::ir::Expr::Call { func: #func, args: ::std::vec![#(#args),*] } }
        }
        Expr::Cast { expr: inner, ty } => {
            let inner = boxed(inner);
            let ty = string(ty);
            quote! { ::goldy::kernel::ir::Expr::Cast { expr: #inner, ty: #ty } }
        }
    }
}

fn bin_op(op: BinOp) -> TokenStream {
    let variant = match op {
        BinOp::Add => quote! { Add },
        BinOp::Sub => quote! { Sub },
        BinOp::Mul => quote! { Mul },
        BinOp::Div => quote! { Div },
        BinOp::Rem => quote! { Rem },
        BinOp::Eq => quote! { Eq },
        BinOp::Ne => quote! { Ne },
        BinOp::Lt => quote! { Lt },
        BinOp::Le => quote! { Le },
        BinOp::Gt => quote! { Gt },
        BinOp::Ge => quote! { Ge },
        BinOp::And => quote! { And },
        BinOp::Or => quote! { Or },
        BinOp::BitAnd => quote! { BitAnd },
        BinOp::BitOr => quote! { BitOr },
        BinOp::BitXor => quote! { BitXor },
        BinOp::Shl => quote! { Shl },
        BinOp::Shr => quote! { Shr },
    };
    quote! { ::goldy::kernel::ir::BinOp::#variant }
}

fn builtin_fn(func: BuiltinFn) -> TokenStream {
    let variant = match func {
        BuiltinFn::GlobalId => quote! { GlobalId },
        BuiltinFn::LocalId => quote! { LocalId },
        BuiltinFn::WorkgroupId => quote! { WorkgroupId },
        BuiltinFn::WorkgroupSize => quote! { WorkgroupSize },
        BuiltinFn::Abs => quote! { Abs },
        BuiltinFn::Min => quote! { Min },
        BuiltinFn::Max => quote! { Max },
        BuiltinFn::Floor => quote! { Floor },
        BuiltinFn::Ceil => quote! { Ceil },
        BuiltinFn::Sqrt => quote! { Sqrt },
        BuiltinFn::Sin => quote! { Sin },
        BuiltinFn::Cos => quote! { Cos },
        BuiltinFn::Exp => quote! { Exp },
        BuiltinFn::Log => quote! { Log },
        BuiltinFn::Pow => quote! { Pow },
        BuiltinFn::Length => quote! { Length },
        BuiltinFn::Float2 => quote! { Float2 },
        BuiltinFn::Float3 => quote! { Float3 },
        BuiltinFn::Float4 => quote! { Float4 },
        BuiltinFn::Uint2 => quote! { Uint2 },
        BuiltinFn::WorkgroupBarrier => quote! { WorkgroupBarrier },
    };
    quote! { ::goldy::kernel::ir::BuiltinFn::#variant }
}
