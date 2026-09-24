//! Symbol renaming over a retained [`ShaderKernel`].
//!
//! Composition places several definitions in one virtual entry, so every name a
//! definition binds must be made collision-free and every formal parameter mapped
//! to the composed entry's parameter.

use crate::{Expr, MatrixOp, ShaderKernel, Stmt};
use std::collections::HashMap;

/// What a name bound by a kernel definition denotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    /// Formal parameter of the virtual entry.
    Param,
    /// `let` binding or `for` induction variable.
    Local,
    /// Workgroup-shared array, emitted at module scope.
    WorkgroupArray,
}

impl ShaderKernel {
    /// Copy with every bound symbol renamed by `rename`, following Rust scoping.
    ///
    /// `rename` is called once per binding. Each reference resolves to its innermost
    /// binding; names the definition does not bind (Slang globals) are unchanged.
    /// Keeping locals and workgroup arrays distinct after renaming is the caller's
    /// responsibility; several parameters may deliberately map to one name.
    pub fn rename_symbols(&self, mut rename: impl FnMut(&str, SymbolKind) -> String) -> ShaderKernel {
        let mut renamer = Renamer {
            scopes: vec![HashMap::new()],
            rename: &mut rename,
        };
        let params = self
            .params
            .iter()
            .map(|p| {
                let mut p = p.clone();
                p.name = renamer.bind(&p.name, SymbolKind::Param);
                p
            })
            .collect();
        let body = renamer.block(&self.body);
        ShaderKernel {
            name: self.name.clone(),
            workgroup_size: self.workgroup_size,
            params,
            builtins: self.builtins,
            body,
            source_map: self.source_map.clone(),
            type_decls: self.type_decls.clone(),
        }
    }

    /// Copy with every local and workgroup array prefixed by `{namespace}_`.
    ///
    /// Parameters keep their names; mapping them to a composed entry is a separate
    /// [`rename_symbols`](Self::rename_symbols) step.
    pub fn namespaced(&self, namespace: &str) -> ShaderKernel {
        self.rename_symbols(|name, kind| match kind {
            SymbolKind::Param => name.to_string(),
            SymbolKind::Local | SymbolKind::WorkgroupArray => format!("{namespace}_{name}"),
        })
    }
}

struct Renamer<'a> {
    scopes: Vec<HashMap<String, String>>,
    rename: &'a mut dyn FnMut(&str, SymbolKind) -> String,
}

impl Renamer<'_> {
    fn bind(&mut self, name: &str, kind: SymbolKind) -> String {
        let renamed = (self.rename)(name, kind);
        self.scopes
            .last_mut()
            .expect("renamer always has a scope")
            .insert(name.to_string(), renamed.clone());
        renamed
    }

    fn resolve(&self, name: &str) -> String {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    fn block(&mut self, stmts: &[Stmt]) -> Vec<Stmt> {
        self.scopes.push(HashMap::new());
        let out = stmts.iter().map(|s| self.stmt(s)).collect();
        self.scopes.pop();
        out
    }

    fn stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let {
                name,
                mutable,
                ty,
                init,
            } => {
                // The initializer cannot see the binding it introduces.
                let init = self.expr(init);
                Stmt::Let {
                    name: self.bind(name, SymbolKind::Local),
                    mutable: *mutable,
                    ty: ty.clone(),
                    init,
                }
            }
            Stmt::Assign { target, value } => Stmt::Assign {
                target: self.expr(target),
                value: self.expr(value),
            },
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => Stmt::If {
                cond: self.expr(cond),
                then_body: self.block(then_body),
                else_body: else_body.as_ref().map(|b| self.block(b)),
            },
            Stmt::While { cond, body } => Stmt::While {
                cond: self.expr(cond),
                body: self.block(body),
            },
            Stmt::ForRange { var, start, end, body } => {
                let start = self.expr(start);
                let end = self.expr(end);
                self.scopes.push(HashMap::new());
                let var = self.bind(var, SymbolKind::Local);
                let body = self.block(body);
                self.scopes.pop();
                Stmt::ForRange { var, start, end, body }
            }
            Stmt::Return { value } => Stmt::Return {
                value: value.as_ref().map(|v| self.expr(v)),
            },
            Stmt::WorkgroupArray { name, elem, len } => Stmt::WorkgroupArray {
                name: self.bind(name, SymbolKind::WorkgroupArray),
                elem: elem.clone(),
                len: *len,
            },
            Stmt::WorkgroupReduce {
                op,
                n,
                val,
                scratch,
                dest,
            } => Stmt::WorkgroupReduce {
                op: *op,
                n: *n,
                val: self.expr(val),
                scratch: self.resolve(scratch),
                dest: self.expr(dest),
            },
            Stmt::WorkgroupSoftmax {
                n,
                buf,
                base,
                count,
                scratch,
            } => Stmt::WorkgroupSoftmax {
                n: *n,
                buf: self.resolve(buf),
                base: self.expr(base),
                count: self.expr(count),
                scratch: self.resolve(scratch),
            },
            Stmt::Matrix(op) => Stmt::Matrix(match op {
                MatrixOp::Accumulator { name } => MatrixOp::Accumulator {
                    name: self.bind(name, SymbolKind::Local),
                },
                MatrixOp::MulAdd { acc, a, b } => MatrixOp::MulAdd {
                    acc: self.resolve(acc),
                    a: self.resolve(a),
                    b: self.resolve(b),
                },
                MatrixOp::Store { acc, dest } => MatrixOp::Store {
                    acc: self.resolve(acc),
                    dest: self.resolve(dest),
                },
            }),
            Stmt::Expr(expr) => Stmt::Expr(self.expr(expr)),
        }
    }

    fn expr(&self, expr: &Expr) -> Expr {
        let sub = |e: &Expr| Box::new(self.expr(e));
        match expr {
            Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_) => expr.clone(),
            Expr::Var(name) => Expr::Var(self.resolve(name)),
            Expr::Field { base, field } => Expr::Field {
                base: sub(base),
                field: field.clone(),
            },
            Expr::Index { base, index } => Expr::Index {
                base: sub(base),
                index: sub(index),
            },
            Expr::Len { base } => Expr::Len { base: sub(base) },
            Expr::Dim { base, axis } => Expr::Dim {
                base: sub(base),
                axis: sub(axis),
            },
            Expr::Rank { base } => Expr::Rank { base: sub(base) },
            Expr::Binary { op, left, right } => Expr::Binary {
                op: *op,
                left: sub(left),
                right: sub(right),
            },
            Expr::Unary { op, expr } => Expr::Unary {
                op: *op,
                expr: sub(expr),
            },
            Expr::Call { func, args } => Expr::Call {
                func: *func,
                args: args.iter().map(|a| self.expr(a)).collect(),
            },
            Expr::Cast { expr, ty } => Expr::Cast {
                expr: sub(expr),
                ty: ty.clone(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        emit_canonical_compute_source, BinOp, BuiltinFn, BuiltinMask, ElementType, KernelParam, SourceMap,
        WorkgroupReduceOp,
    };

    fn var(name: &str) -> Expr {
        Expr::Var(name.into())
    }

    fn kernel(params: Vec<KernelParam>, body: Vec<Stmt>) -> ShaderKernel {
        ShaderKernel {
            name: "k".into(),
            workgroup_size: [64, 1, 1],
            params,
            builtins: BuiltinMask {
                global_id: true,
                local_id: true,
                ..BuiltinMask::NONE
            },
            body,
            source_map: SourceMap::default(),
            type_decls: Vec::new(),
        }
    }

    fn global_x() -> Expr {
        Expr::Field {
            base: Box::new(Expr::Call {
                func: BuiltinFn::GlobalId,
                args: vec![],
            }),
            field: "x".into(),
        }
    }

    #[test]
    fn namespaced_prefixes_locals_and_keeps_params_fields_and_globals() {
        let k = kernel(
            vec![KernelParam::buffer_read_write("data", ElementType::F32)],
            vec![
                Stmt::Let {
                    name: "i".into(),
                    mutable: false,
                    ty: Some("uint".into()),
                    init: global_x(),
                },
                Stmt::Assign {
                    target: Expr::Index {
                        base: Box::new(var("data")),
                        index: Box::new(var("i")),
                    },
                    value: Expr::Cast {
                        expr: Box::new(var("SOME_GLOBAL")),
                        ty: "float".into(),
                    },
                },
            ],
        );
        let ns = k.namespaced("_goldy_k0");
        assert_eq!(ns.params[0].name, "data");
        let slang = emit_canonical_compute_source(&ns).source.canonical_slang;
        assert!(slang.contains("uint _goldy_k0_i = _goldy_gid.x;"), "{slang}");
        assert!(slang.contains("data[_goldy_k0_i] = ((float)SOME_GLOBAL);"), "{slang}");
    }

    #[test]
    fn rename_follows_rust_scoping() {
        // fn k(x: &mut [f32]) { let v = 1.0; if .. { let v = v + 2.0; x[0] = v; } x[1] = v; }
        let k = kernel(
            vec![KernelParam::buffer_read_write("x", ElementType::F32)],
            vec![
                Stmt::Let {
                    name: "v".into(),
                    mutable: false,
                    ty: Some("float".into()),
                    init: Expr::LitF32(1.0),
                },
                Stmt::If {
                    cond: Expr::LitBool(true),
                    then_body: vec![
                        Stmt::Let {
                            name: "v".into(),
                            mutable: false,
                            ty: Some("float".into()),
                            init: Expr::Binary {
                                op: BinOp::Add,
                                left: Box::new(var("v")),
                                right: Box::new(Expr::LitF32(2.0)),
                            },
                        },
                        Stmt::Assign {
                            target: Expr::Index {
                                base: Box::new(var("x")),
                                index: Box::new(Expr::LitU32(0)),
                            },
                            value: var("v"),
                        },
                    ],
                    else_body: None,
                },
                Stmt::Assign {
                    target: Expr::Index {
                        base: Box::new(var("x")),
                        index: Box::new(Expr::LitU32(1)),
                    },
                    value: var("v"),
                },
            ],
        );
        let mut depth = 0;
        let renamed = k.rename_symbols(|name, kind| match kind {
            SymbolKind::Param => format!("fused_{name}"),
            _ => {
                depth += 1;
                format!("{name}{depth}")
            }
        });
        let slang = emit_canonical_compute_source(&renamed).source.canonical_slang;
        assert!(slang.contains("Scattered<float> fused_x"), "{slang}");
        assert!(slang.contains("float v1 = 1.0;"), "{slang}");
        assert!(slang.contains("float v2 = (v1 + 2.0);"), "{slang}");
        assert!(slang.contains("fused_x[0u] = v2;"), "{slang}");
        assert!(slang.contains("fused_x[1u] = v1;"), "{slang}");
    }

    #[test]
    fn for_variable_is_scoped_to_its_loop() {
        let k = kernel(
            vec![KernelParam::buffer_read_write("x", ElementType::U32)],
            vec![
                Stmt::ForRange {
                    var: "x".into(),
                    start: Expr::LitU32(0),
                    end: Expr::Len {
                        base: Box::new(var("x")),
                    },
                    body: vec![Stmt::Expr(var("x"))],
                },
                Stmt::Expr(var("x")),
            ],
        );
        let renamed = k.namespaced("n");
        let Stmt::ForRange { var: v, end, body, .. } = &renamed.body[0] else {
            panic!("expected loop");
        };
        assert_eq!(v, "n_x");
        assert_eq!(
            end,
            &Expr::Len {
                base: Box::new(var("x"))
            }
        );
        assert_eq!(body[0], Stmt::Expr(var("n_x")));
        assert_eq!(renamed.body[1], Stmt::Expr(var("x")));
    }

    #[test]
    fn workgroup_arrays_and_collective_operands_are_renamed() {
        let k = kernel(
            vec![KernelParam::buffer_read_write("att", ElementType::F32)],
            vec![
                Stmt::WorkgroupArray {
                    name: "scratch".into(),
                    elem: "float".into(),
                    len: 64,
                },
                Stmt::Let {
                    name: "total".into(),
                    mutable: true,
                    ty: Some("float".into()),
                    init: Expr::LitF32(1.0),
                },
                Stmt::WorkgroupReduce {
                    op: WorkgroupReduceOp::Sum,
                    n: 64,
                    val: var("total"),
                    scratch: "scratch".into(),
                    dest: var("total"),
                },
                Stmt::WorkgroupSoftmax {
                    n: 64,
                    buf: "att".into(),
                    base: Expr::LitU32(0),
                    count: Expr::LitU32(4),
                    scratch: "scratch".into(),
                },
            ],
        );
        let slang = emit_canonical_compute_source(&k.namespaced("_goldy_k1"))
            .source
            .canonical_slang;
        assert!(slang.contains("groupshared float _goldy_k1_scratch[64];"), "{slang}");
        assert!(!slang.contains(" scratch["), "{slang}");
        assert!(slang.contains("_goldy_k1_total = _goldy_k1_scratch[0];"), "{slang}");
        assert!(
            slang.contains("exp(att[(0u) + _goldy_sm_t] - _goldy_sm_max)"),
            "{slang}"
        );
    }

    #[test]
    fn renamed_tensor_params_keep_their_metadata_slots() {
        let k = kernel(
            vec![
                KernelParam::tensor_read("src", ElementType::F32),
                KernelParam::tensor_write("dst", ElementType::F32),
            ],
            vec![Stmt::Assign {
                target: Expr::Index {
                    base: Box::new(var("dst")),
                    index: Box::new(Expr::LitU32(0)),
                },
                value: Expr::Len {
                    base: Box::new(var("src")),
                },
            }],
        );
        let renamed = k.rename_symbols(|name, _| format!("p_{name}"));
        let slang = emit_canonical_compute_source(&renamed).source.canonical_slang;
        assert!(
            slang.contains("p_dst[goldy_tensor_offset(_goldy_tensor_meta[1u], 0u)] = _goldy_tensor_meta[0u].numel;"),
            "{slang}"
        );
    }
}
