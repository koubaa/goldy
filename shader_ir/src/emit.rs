//! Emit canonical `[goldy_compute]` Slang from a lowered [`ShaderKernel`].

use crate::{BinOp, BuiltinFn, BuiltinMask, Expr, KernelDef, ShaderKernel, Stmt, UnaryOp, WorkgroupReduceOp};

/// Emit the portable canonical compute source (still marked `[goldy_compute]`).
///
/// Backend-specific virtual-main transforms remain responsible for PushLayout /
/// frame-table / CUDA / WebGPU plumbing.
pub fn emit_canonical_compute_source(kernel: &ShaderKernel) -> KernelDef {
    let entry = "cs_main";
    let mut sig_parts: Vec<String> = kernel
        .params
        .iter()
        .map(|p| format!("{} {}", p.slang_param_type(), p.name))
        .collect();
    // Hidden builtins appended in stable order.
    if kernel.builtins.global_id {
        sig_parts.push("ThreadId _goldy_gid".to_string());
    }
    if kernel.builtins.local_id {
        sig_parts.push("GroupThreadId _goldy_lid".to_string());
    }
    if kernel.builtins.workgroup_id {
        sig_parts.push("GroupId _goldy_wid".to_string());
    }

    let [wx, wy, wz] = kernel.workgroup_size;
    let body = emit_user_helper_body(&kernel.body, &kernel.builtins);
    let shared = emit_workgroup_decls(&kernel.body);
    let sig = sig_parts.join(", ");
    let canonical = format!(
        "import goldy_exp;\n\n\
         {shared}\
         [goldy_compute]\n\
         [numthreads({wx}, {wy}, {wz})]\n\
         void {entry}({sig}) {{\n{body}}}\n"
    );

    KernelDef::new(
        canonical,
        entry,
        kernel.workgroup_size,
        kernel.params.clone(),
        kernel.builtins,
        kernel.source_map.clone(),
    )
}

/// Emit the indented Slang body for a list of statements.
pub fn emit_user_helper_body(body: &[Stmt], builtins: &BuiltinMask) -> String {
    let mut out = String::new();
    for stmt in body {
        emit_stmt(&mut out, stmt, 1, builtins);
    }
    out
}

fn emit_workgroup_decls(body: &[Stmt]) -> String {
    let mut out = String::new();
    for stmt in body {
        if let Stmt::WorkgroupArray { name, elem, len } = stmt {
            out.push_str(&format!("groupshared {elem} {name}[{len}];\n"));
        }
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

fn indent(level: usize) -> String {
    "    ".repeat(level)
}

fn emit_stmt(out: &mut String, stmt: &Stmt, level: usize, builtins: &BuiltinMask) {
    let pad = indent(level);
    match stmt {
        Stmt::Let {
            name,
            mutable: _,
            ty,
            init,
        } => {
            // Slang requires typed locals; default to uint when the frontend
            // could not infer a more precise type.
            let ty_s = ty.as_deref().unwrap_or("uint");
            out.push_str(&format!("{pad}{ty_s} {name} = {};\n", emit_expr(init, builtins)));
        }
        Stmt::Assign { target, value } => {
            out.push_str(&format!(
                "{pad}{} = {};\n",
                emit_expr(target, builtins),
                emit_expr(value, builtins)
            ));
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            out.push_str(&format!("{pad}if ({}) {{\n", emit_expr(cond, builtins)));
            for s in then_body {
                emit_stmt(out, s, level + 1, builtins);
            }
            if let Some(else_body) = else_body {
                out.push_str(&format!("{pad}}} else {{\n"));
                for s in else_body {
                    emit_stmt(out, s, level + 1, builtins);
                }
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::While { cond, body } => {
            out.push_str(&format!("{pad}while ({}) {{\n", emit_expr(cond, builtins)));
            for s in body {
                emit_stmt(out, s, level + 1, builtins);
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::ForRange { var, start, end, body } => {
            out.push_str(&format!(
                "{pad}for (uint {var} = {}; {var} < {}; ++{var}) {{\n",
                emit_expr(start, builtins),
                emit_expr(end, builtins)
            ));
            for s in body {
                emit_stmt(out, s, level + 1, builtins);
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::Return { value } => {
            if let Some(v) = value {
                out.push_str(&format!("{pad}return {};\n", emit_expr(v, builtins)));
            } else {
                out.push_str(&format!("{pad}return;\n"));
            }
        }
        Stmt::WorkgroupArray { .. } => {}
        Stmt::WorkgroupReduce {
            op,
            n,
            val,
            scratch,
            dest,
        } => emit_workgroup_reduce(out, level, *op, *n, val, scratch, dest, builtins),
        Stmt::WorkgroupSoftmax {
            n,
            buf,
            base,
            count,
            scratch,
        } => emit_workgroup_softmax(out, level, *n, buf, base, count, scratch, builtins),
        Stmt::Expr(expr) => {
            out.push_str(&format!("{pad}{};\n", emit_expr(expr, builtins)));
        }
    }
}

fn reduce_steps(n: u32) -> u32 {
    n.trailing_zeros()
}

fn emit_workgroup_reduce(
    out: &mut String,
    level: usize,
    op: WorkgroupReduceOp,
    n: u32,
    val: &Expr,
    scratch: &str,
    dest: &Expr,
    builtins: &BuiltinMask,
) {
    let pad = indent(level);
    let inner = indent(level + 1);
    let loop_pad = indent(level + 2);
    let steps = reduce_steps(n);
    out.push_str(&format!("{pad}{{\n"));
    out.push_str(&format!("{inner}float _goldy_red = {};\n", emit_expr(val, builtins)));
    out.push_str(&format!("{inner}{scratch}[_goldy_lid.x] = _goldy_red;\n"));
    out.push_str(&format!(
        "{inner}for (uint _goldy_s = 0u; _goldy_s < {steps}u; ++_goldy_s) {{\n"
    ));
    out.push_str(&format!("{loop_pad}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{loop_pad}if (_goldy_lid.x + (1u << _goldy_s) < {n}u)\n"));
    match op {
        WorkgroupReduceOp::Sum => out.push_str(&format!(
            "{loop_pad}    _goldy_red = _goldy_red + {scratch}[_goldy_lid.x + (1u << _goldy_s)];\n"
        )),
        WorkgroupReduceOp::Max => out.push_str(&format!(
            "{loop_pad}    _goldy_red = max(_goldy_red, {scratch}[_goldy_lid.x + (1u << _goldy_s)]);\n"
        )),
    }
    out.push_str(&format!("{loop_pad}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{loop_pad}{scratch}[_goldy_lid.x] = _goldy_red;\n"));
    out.push_str(&format!("{inner}}}\n"));
    out.push_str(&format!("{inner}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{inner}{} = {scratch}[0];\n", emit_expr(dest, builtins)));
    out.push_str(&format!("{pad}}}\n"));
}

fn emit_workgroup_softmax(
    out: &mut String,
    level: usize,
    n: u32,
    buf: &str,
    base: &Expr,
    count: &Expr,
    scratch: &str,
    builtins: &BuiltinMask,
) {
    let pad = indent(level);
    let inner = indent(level + 1);
    let loop_pad = indent(level + 2);
    let base_s = emit_expr(base, builtins);
    let count_s = emit_expr(count, builtins);
    out.push_str(&format!("{pad}{{\n"));
    out.push_str(&format!("{inner}float _goldy_sm_max = -1e30;\n"));
    out.push_str(&format!("{inner}uint _goldy_sm_t = _goldy_lid.x;\n"));
    out.push_str(&format!("{inner}while (_goldy_sm_t < {count_s}) {{\n"));
    out.push_str(&format!(
        "{loop_pad}float _goldy_sm_s = {buf}[({base_s}) + _goldy_sm_t];\n"
    ));
    out.push_str(&format!("{loop_pad}if (_goldy_sm_s > _goldy_sm_max) {{\n"));
    out.push_str(&format!("{loop_pad}    _goldy_sm_max = _goldy_sm_s;\n"));
    out.push_str(&format!("{loop_pad}}}\n"));
    out.push_str(&format!("{loop_pad}_goldy_sm_t = _goldy_sm_t + {n}u;\n"));
    out.push_str(&format!("{inner}}}\n"));
    emit_workgroup_reduce(
        out,
        level + 1,
        WorkgroupReduceOp::Max,
        n,
        &Expr::Var("_goldy_sm_max".into()),
        scratch,
        &Expr::Var("_goldy_sm_max".into()),
        builtins,
    );
    out.push_str(&format!("{inner}float _goldy_sm_sum = 0.0;\n"));
    out.push_str(&format!("{inner}_goldy_sm_t = _goldy_lid.x;\n"));
    out.push_str(&format!("{inner}while (_goldy_sm_t < {count_s}) {{\n"));
    out.push_str(&format!(
        "{loop_pad}float _goldy_sm_e = exp({buf}[({base_s}) + _goldy_sm_t] - _goldy_sm_max);\n"
    ));
    out.push_str(&format!("{loop_pad}{buf}[({base_s}) + _goldy_sm_t] = _goldy_sm_e;\n"));
    out.push_str(&format!("{loop_pad}_goldy_sm_sum = _goldy_sm_sum + _goldy_sm_e;\n"));
    out.push_str(&format!("{loop_pad}_goldy_sm_t = _goldy_sm_t + {n}u;\n"));
    out.push_str(&format!("{inner}}}\n"));
    emit_workgroup_reduce(
        out,
        level + 1,
        WorkgroupReduceOp::Sum,
        n,
        &Expr::Var("_goldy_sm_sum".into()),
        scratch,
        &Expr::Var("_goldy_sm_sum".into()),
        builtins,
    );
    out.push_str(&format!("{inner}_goldy_sm_t = _goldy_lid.x;\n"));
    out.push_str(&format!("{inner}while (_goldy_sm_t < {count_s}) {{\n"));
    out.push_str(&format!(
        "{loop_pad}{buf}[({base_s}) + _goldy_sm_t] = {buf}[({base_s}) + _goldy_sm_t] / _goldy_sm_sum;\n"
    ));
    out.push_str(&format!("{loop_pad}_goldy_sm_t = _goldy_sm_t + {n}u;\n"));
    out.push_str(&format!("{inner}}}\n"));
    out.push_str(&format!("{inner}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{pad}}}\n"));
}

fn emit_expr(expr: &Expr, builtins: &BuiltinMask) -> String {
    match expr {
        Expr::LitU32(v) => format!("{v}u"),
        Expr::LitI32(v) => format!("{v}"),
        Expr::LitF32(v) => {
            let mut s = format!("{v}");
            if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                s.push('.');
                s.push('0');
            }
            s
        }
        Expr::LitBool(v) => if *v { "true" } else { "false" }.to_string(),
        Expr::Var(name) => name.clone(),
        Expr::Field { base, field } => format!("{}.{}", emit_expr(base, builtins), field),
        Expr::Index { base, index } => {
            format!("{}[{}]", emit_expr(base, builtins), emit_expr(index, builtins))
        }
        Expr::Len { base } => format!("goldy_buf_len({})", emit_expr(base, builtins)),
        Expr::Binary { op, left, right } => format!(
            "({} {} {})",
            emit_expr(left, builtins),
            bin_op_slang(*op),
            emit_expr(right, builtins)
        ),
        Expr::Unary { op, expr } => format!("({}{})", unary_op_slang(*op), emit_expr(expr, builtins)),
        Expr::Call { func, args } => emit_call(*func, args, builtins),
        Expr::Cast { expr, ty } => format!("(({}){})", ty, emit_expr(expr, builtins)),
    }
}

fn emit_call(func: BuiltinFn, args: &[Expr], builtins: &BuiltinMask) -> String {
    match func {
        BuiltinFn::GlobalId => {
            assert!(builtins.global_id, "global_id used without builtin mask");
            assert!(args.is_empty());
            "_goldy_gid".to_string()
        }
        BuiltinFn::LocalId => {
            assert!(builtins.local_id);
            assert!(args.is_empty());
            "_goldy_lid".to_string()
        }
        BuiltinFn::WorkgroupId => {
            assert!(builtins.workgroup_id);
            assert!(args.is_empty());
            "_goldy_wid".to_string()
        }
        BuiltinFn::WorkgroupSize => {
            // Compile-time constant — callers should prefer the KernelDef field;
            // as an expression we emit a uint3 literal only if args carry the size.
            if let [Expr::LitU32(x), Expr::LitU32(y), Expr::LitU32(z)] = args {
                format!("uint3({x}u, {y}u, {z}u)")
            } else {
                "uint3(0u, 0u, 0u)".to_string()
            }
        }
        BuiltinFn::Abs => format!("abs({})", join_args(args, builtins)),
        BuiltinFn::Min => format!("min({})", join_args(args, builtins)),
        BuiltinFn::Max => format!("max({})", join_args(args, builtins)),
        BuiltinFn::Floor => format!("floor({})", join_args(args, builtins)),
        BuiltinFn::Ceil => format!("ceil({})", join_args(args, builtins)),
        BuiltinFn::Sqrt => format!("sqrt({})", join_args(args, builtins)),
        BuiltinFn::Sin => format!("sin({})", join_args(args, builtins)),
        BuiltinFn::Cos => format!("cos({})", join_args(args, builtins)),
        BuiltinFn::Exp => format!("exp({})", join_args(args, builtins)),
        BuiltinFn::Log => format!("log({})", join_args(args, builtins)),
        BuiltinFn::Pow => format!("pow({})", join_args(args, builtins)),
        BuiltinFn::Length => format!("length({})", join_args(args, builtins)),
        BuiltinFn::WorkgroupBarrier => {
            assert!(args.is_empty());
            "GroupMemoryBarrierWithGroupSync()".to_string()
        }
        BuiltinFn::Float2 => format!("float2({})", join_args(args, builtins)),
        BuiltinFn::Float3 => format!("float3({})", join_args(args, builtins)),
        BuiltinFn::Float4 => format!("float4({})", join_args(args, builtins)),
        BuiltinFn::Uint2 => format!("uint2({})", join_args(args, builtins)),
    }
}

fn join_args(args: &[Expr], builtins: &BuiltinMask) -> String {
    args.iter()
        .map(|a| emit_expr(a, builtins))
        .collect::<Vec<_>>()
        .join(", ")
}

fn bin_op_slang(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
    }
}

fn unary_op_slang(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "-",
        UnaryOp::Not => "!",
        UnaryOp::BitNot => "~",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementType, KernelParam, ScalarType, SourceMap};

    #[test]
    fn emits_saxpy_shaped_kernel() {
        let kernel = ShaderKernel {
            name: "saxpy".into(),
            workgroup_size: [256, 1, 1],
            params: vec![
                KernelParam::buffer_read("x", ElementType::F32),
                KernelParam::buffer_read_write("y", ElementType::F32),
                KernelParam::scalar_param("a", ScalarType::F32),
            ],
            builtins: BuiltinMask {
                global_id: true,
                ..BuiltinMask::NONE
            },
            body: vec![
                Stmt::Let {
                    name: "i".into(),
                    mutable: false,
                    ty: Some("uint".into()),
                    init: Expr::Field {
                        base: Box::new(Expr::Call {
                            func: BuiltinFn::GlobalId,
                            args: vec![],
                        }),
                        field: "x".into(),
                    },
                },
                Stmt::If {
                    cond: Expr::Binary {
                        op: BinOp::Lt,
                        left: Box::new(Expr::Var("i".into())),
                        right: Box::new(Expr::Len {
                            base: Box::new(Expr::Var("y".into())),
                        }),
                    },
                    then_body: vec![Stmt::Assign {
                        target: Expr::Index {
                            base: Box::new(Expr::Var("y".into())),
                            index: Box::new(Expr::Var("i".into())),
                        },
                        value: Expr::Binary {
                            op: BinOp::Add,
                            left: Box::new(Expr::Binary {
                                op: BinOp::Mul,
                                left: Box::new(Expr::Var("a".into())),
                                right: Box::new(Expr::Index {
                                    base: Box::new(Expr::Var("x".into())),
                                    index: Box::new(Expr::Var("i".into())),
                                }),
                            }),
                            right: Box::new(Expr::Index {
                                base: Box::new(Expr::Var("y".into())),
                                index: Box::new(Expr::Var("i".into())),
                            }),
                        },
                    }],
                    else_body: None,
                },
            ],
            source_map: SourceMap {
                rust_file: "saxpy.rs".into(),
                rust_line: 10,
            },
        };

        let def = emit_canonical_compute_source(&kernel);
        assert!(def.source.canonical_slang.contains("[goldy_compute]"));
        assert!(def.source.canonical_slang.contains("BufRO<float> x"));
        assert!(def.source.canonical_slang.contains("Scattered<float> y"));
        assert!(def.source.canonical_slang.contains("float a"));
        assert!(def.source.canonical_slang.contains("ThreadId _goldy_gid"));
        assert!(def.source.canonical_slang.contains("goldy_buf_len(y)"));
        assert_eq!(def.entry, "cs_main");
        assert_eq!(def.workgroup_size, [256, 1, 1]);
    }

    #[test]
    fn emits_groupshared_and_barrier() {
        let kernel = ShaderKernel {
            name: "reduce".into(),
            workgroup_size: [256, 1, 1],
            params: vec![KernelParam::buffer_read_write("data", ElementType::F32)],
            builtins: BuiltinMask {
                local_id: true,
                ..BuiltinMask::NONE
            },
            body: vec![
                Stmt::WorkgroupArray {
                    name: "scratch".into(),
                    elem: "float".into(),
                    len: 256,
                },
                Stmt::Assign {
                    target: Expr::Index {
                        base: Box::new(Expr::Var("scratch".into())),
                        index: Box::new(Expr::Field {
                            base: Box::new(Expr::Call {
                                func: BuiltinFn::LocalId,
                                args: vec![],
                            }),
                            field: "x".into(),
                        }),
                    },
                    value: Expr::LitF32(1.0),
                },
                Stmt::Expr(Expr::Call {
                    func: BuiltinFn::WorkgroupBarrier,
                    args: vec![],
                }),
            ],
            source_map: SourceMap {
                rust_file: "reduce.rs".into(),
                rust_line: 1,
            },
        };
        let slang = emit_canonical_compute_source(&kernel).source.canonical_slang;
        assert!(slang.contains("groupshared float scratch[256];"));
        assert!(slang.contains("GroupMemoryBarrierWithGroupSync()"));
        assert!(!slang.contains("float scratch ="));
    }

    #[test]
    fn emits_workgroup_sum_and_softmax() {
        let kernel = ShaderKernel {
            name: "collectives".into(),
            workgroup_size: [256, 1, 1],
            params: vec![KernelParam::buffer_read_write("att", ElementType::F32)],
            builtins: BuiltinMask {
                local_id: true,
                ..BuiltinMask::NONE
            },
            body: vec![
                Stmt::WorkgroupArray {
                    name: "scratch".into(),
                    elem: "float".into(),
                    len: 256,
                },
                Stmt::Let {
                    name: "ss".into(),
                    mutable: true,
                    ty: Some("float".into()),
                    init: Expr::LitF32(1.0),
                },
                Stmt::WorkgroupReduce {
                    op: WorkgroupReduceOp::Sum,
                    n: 256,
                    val: Expr::Var("ss".into()),
                    scratch: "scratch".into(),
                    dest: Expr::Var("ss".into()),
                },
                Stmt::WorkgroupSoftmax {
                    n: 256,
                    buf: "att".into(),
                    base: Expr::LitU32(0),
                    count: Expr::LitU32(4),
                    scratch: "scratch".into(),
                },
            ],
            source_map: SourceMap {
                rust_file: "collectives.rs".into(),
                rust_line: 1,
            },
        };
        let slang = emit_canonical_compute_source(&kernel).source.canonical_slang;
        assert!(slang.contains("GroupThreadId _goldy_lid"));
        assert!(slang.contains("_goldy_red = _goldy_red + scratch[_goldy_lid.x + (1u << _goldy_s)]"));
        assert!(slang.contains("ss = scratch[0];"));
        assert!(slang.contains("exp(att[(0u) + _goldy_sm_t] - _goldy_sm_max)"));
        assert!(slang.contains("_goldy_red = max(_goldy_red, scratch[_goldy_lid.x + (1u << _goldy_s)])"));
    }
}
