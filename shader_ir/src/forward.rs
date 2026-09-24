//! Value forwarding between the stages of a fused entry.
//!
//! A forwarded parameter's element at the invoking thread's own index is cached in a
//! pair of locals the fused entry owns and passes `inout` to every stage: `value` and
//! `ok`. The rewrite keeps every store, so `ok` implies `value` equals the parcel
//! element. A load that follows a store (or an earlier load) reads `value` instead of
//! reloading. A load left untouched still reads memory and is still correct, which is
//! what makes skipping awkward positions (short-circuit operands, `while` conditions)
//! safe.
//!
//! Only parameters whose every access in every stage names the thread's own index are
//! forwarded; admission establishes that before a parameter is listed.
//!
//! An elided parameter has no parcel to fall back on, so [`elide_body`] routes every
//! access through its register, including the positions forwarding leaves on memory.

use crate::{BinOp, Expr, Stmt, UnaryOp};
use std::collections::{HashMap, HashSet};

/// The locals that cache one forwarded parameter's element inside a stage function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ForwardedLocal {
    pub value: String,
    pub ok: String,
}

/// Rewrite `body` so accesses to the formals in `locals` go through their cached element.
pub(crate) fn forward_body(body: &[Stmt], locals: &HashMap<String, ForwardedLocal>) -> Vec<Stmt> {
    let mut rewriter = Rewriter {
        locals,
        shadowed: vec![HashSet::new()],
    };
    rewriter.block(body)
}

/// Rewrite `body` so every element access to a formal in `locals` reads or writes the
/// named local instead.
///
/// Every access to such a formal must name the thread's own element, and the formal
/// must not be used as a whole resource; afterwards the body no longer mentions it.
pub(crate) fn elide_body(body: &[Stmt], locals: &HashMap<String, String>) -> Vec<Stmt> {
    let mut eliminator = Eliminator {
        locals,
        shadowed: vec![HashSet::new()],
    };
    eliminator.block(body)
}

/// Zero literal of a forwardable element type, or `None` when the type is not forwardable.
pub(crate) fn zero_literal(slang_type: &str) -> Option<&'static str> {
    match slang_type {
        "float" => Some("0.0"),
        "uint" => Some("0u"),
        "int" => Some("0"),
        "bool" => Some("false"),
        _ => None,
    }
}

struct Rewriter<'a> {
    locals: &'a HashMap<String, ForwardedLocal>,
    /// Names bound by `let`, `for` or workgroup arrays, per scope; they hide formals.
    shadowed: Vec<HashSet<String>>,
}

impl<'a> Rewriter<'a> {
    fn forwarded<'e>(&self, expr: &'e Expr) -> Option<(&'e str, &'a ForwardedLocal)> {
        let Expr::Var(name) = expr else {
            return None;
        };
        if self.shadowed.iter().any(|s| s.contains(name)) {
            return None;
        }
        self.locals.get(name).map(|l| (name.as_str(), l))
    }

    fn bind(&mut self, name: &str) {
        self.shadowed
            .last_mut()
            .expect("rewriter always has a scope")
            .insert(name.to_string());
    }

    fn block(&mut self, stmts: &[Stmt]) -> Vec<Stmt> {
        self.shadowed.push(HashSet::new());
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts {
            self.stmt(s, &mut out);
        }
        self.shadowed.pop();
        out
    }

    fn stmt(&mut self, stmt: &Stmt, out: &mut Vec<Stmt>) {
        match stmt {
            Stmt::Let {
                name,
                mutable,
                ty,
                init,
            } => {
                self.materialize(&[init], out);
                out.push(Stmt::Let {
                    name: name.clone(),
                    mutable: *mutable,
                    ty: ty.clone(),
                    init: self.replace(init),
                });
                self.bind(name);
            }
            Stmt::Assign { target, value } => {
                let mut reads = vec![value];
                if let Some(index) = target_index(target) {
                    reads.push(index);
                }
                self.materialize(&reads, out);
                let value = self.replace(value);
                self.store(target, value, out, |target, value| Stmt::Assign { target, value });
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.materialize(&[cond], out);
                out.push(Stmt::If {
                    cond: self.replace(cond),
                    then_body: self.block(then_body),
                    else_body: else_body.as_ref().map(|b| self.block(b)),
                });
            }
            // The condition is re-evaluated after the body; leave its loads on memory.
            Stmt::While { cond, body } => out.push(Stmt::While {
                cond: cond.clone(),
                body: self.block(body),
            }),
            Stmt::ForRange { var, start, end, body } => {
                self.materialize(&[start, end], out);
                let (start, end) = (self.replace(start), self.replace(end));
                self.shadowed.push(HashSet::from([var.clone()]));
                let body = self.block(body);
                self.shadowed.pop();
                out.push(Stmt::ForRange {
                    var: var.clone(),
                    start,
                    end,
                    body,
                });
            }
            Stmt::Return { value } => {
                if let Some(v) = value {
                    self.materialize(&[v], out);
                }
                out.push(Stmt::Return {
                    value: value.as_ref().map(|v| self.replace(v)),
                });
            }
            Stmt::WorkgroupArray { name, .. } => {
                out.push(stmt.clone());
                self.bind(name);
            }
            Stmt::WorkgroupReduce {
                op,
                n,
                val,
                scratch,
                dest,
            } => {
                let mut reads = vec![val];
                if let Some(index) = target_index(dest) {
                    reads.push(index);
                }
                self.materialize(&reads, out);
                let val = self.replace(val);
                let (op, n, scratch) = (*op, *n, scratch.clone());
                self.store(dest, Expr::LitBool(false), out, |dest, _| Stmt::WorkgroupReduce {
                    op,
                    n,
                    val: val.clone(),
                    scratch: scratch.clone(),
                    dest,
                });
            }
            Stmt::WorkgroupSoftmax {
                n,
                buf,
                base,
                count,
                scratch,
            } => {
                self.materialize(&[base, count], out);
                out.push(Stmt::WorkgroupSoftmax {
                    n: *n,
                    buf: buf.clone(),
                    base: self.replace(base),
                    count: self.replace(count),
                    scratch: scratch.clone(),
                });
            }
            Stmt::Expr(e) => {
                self.materialize(&[e], out);
                out.push(Stmt::Expr(self.replace(e)));
            }
        }
    }

    /// Emit `make(target, value)` for a store, routing a store to a forwarded element
    /// through its cached value. `make` receives the rewritten target.
    fn store(&self, target: &Expr, value: Expr, out: &mut Vec<Stmt>, make: impl Fn(Expr, Expr) -> Stmt) {
        if let Expr::Index { base, index } = target {
            if let Some((_, local)) = self.forwarded(base) {
                let element = Expr::Index {
                    base: base.clone(),
                    index: Box::new(self.replace(index)),
                };
                out.push(make(Expr::Var(local.value.clone()), value));
                out.push(assign(&local.ok, Expr::LitBool(true)));
                out.push(Stmt::Assign {
                    target: element,
                    value: Expr::Var(local.value.clone()),
                });
                return;
            }
        }
        let partial = self.partial_store(target);
        out.push(make(self.replace_target(target), value));
        if let Some(local) = partial {
            out.push(assign(&local.ok, Expr::LitBool(false)));
        }
    }

    /// A store to part of a forwarded element (`buf[i].x = ..`) leaves `value` stale.
    fn partial_store(&self, target: &Expr) -> Option<&'a ForwardedLocal> {
        match target {
            Expr::Field { base, .. } => match base.as_ref() {
                Expr::Index { base, .. } => self.forwarded(base).map(|(_, l)| l),
                other => self.partial_store(other),
            },
            _ => None,
        }
    }

    /// Load each forwarded element `exprs` read into its cache, once, ahead of the statement.
    fn materialize(&self, exprs: &[&Expr], out: &mut Vec<Stmt>) {
        let mut loads: Vec<(&str, &'a ForwardedLocal, &Expr)> = Vec::new();
        for e in exprs {
            self.collect_loads(e, &mut loads);
        }
        for (formal, local, index) in loads {
            out.push(Stmt::If {
                cond: Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(Expr::Var(local.ok.clone())),
                },
                then_body: vec![
                    assign(
                        &local.value,
                        Expr::Index {
                            base: Box::new(Expr::Var(formal.to_string())),
                            index: Box::new(index.clone()),
                        },
                    ),
                    assign(&local.ok, Expr::LitBool(true)),
                ],
                else_body: None,
            });
        }
    }

    fn collect_loads<'e>(&self, expr: &'e Expr, loads: &mut Vec<(&'e str, &'a ForwardedLocal, &'e Expr)>) {
        match expr {
            Expr::Index { base, index } => {
                self.collect_loads(index, loads);
                match self.forwarded(base) {
                    Some((formal, local)) => {
                        if !loads.iter().any(|(f, ..)| *f == formal) {
                            loads.push((formal, local, index));
                        }
                    }
                    None => self.collect_loads(base, loads),
                }
            }
            // Only the left operand is evaluated unconditionally.
            Expr::Binary {
                op: BinOp::And | BinOp::Or,
                left,
                ..
            } => self.collect_loads(left, loads),
            Expr::Binary { left, right, .. } => {
                self.collect_loads(left, loads);
                self.collect_loads(right, loads);
            }
            Expr::Field { base, .. } | Expr::Len { base } | Expr::Rank { base } => self.collect_loads(base, loads),
            Expr::Dim { base, axis } => {
                self.collect_loads(base, loads);
                self.collect_loads(axis, loads);
            }
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => self.collect_loads(expr, loads),
            Expr::Call { args, .. } => {
                for a in args {
                    self.collect_loads(a, loads);
                }
            }
            Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_) | Expr::Var(_) => {}
        }
    }

    /// `expr` with every materialized load read from its cache.
    fn replace(&self, expr: &Expr) -> Expr {
        let boxed = |e: &Expr| Box::new(self.replace(e));
        match expr {
            Expr::Index { base, index } => match self.forwarded(base) {
                Some((_, local)) => Expr::Var(local.value.clone()),
                None => Expr::Index {
                    base: boxed(base),
                    index: boxed(index),
                },
            },
            Expr::Binary {
                op: op @ (BinOp::And | BinOp::Or),
                left,
                right,
            } => Expr::Binary {
                op: *op,
                left: boxed(left),
                right: right.clone(),
            },
            Expr::Binary { op, left, right } => Expr::Binary {
                op: *op,
                left: boxed(left),
                right: boxed(right),
            },
            Expr::Field { base, field } => Expr::Field {
                base: boxed(base),
                field: field.clone(),
            },
            Expr::Len { base } => Expr::Len { base: base.clone() },
            Expr::Rank { base } => Expr::Rank { base: base.clone() },
            Expr::Dim { base, axis } => Expr::Dim {
                base: base.clone(),
                axis: boxed(axis),
            },
            Expr::Unary { op, expr } => Expr::Unary {
                op: *op,
                expr: boxed(expr),
            },
            Expr::Cast { expr, ty } => Expr::Cast {
                expr: boxed(expr),
                ty: ty.clone(),
            },
            Expr::Call { func, args } => Expr::Call {
                func: *func,
                args: args.iter().map(|a| self.replace(a)).collect(),
            },
            Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_) | Expr::Var(_) => expr.clone(),
        }
    }

    /// A store target with its index expressions rewritten; the stored element stays in memory.
    fn replace_target(&self, target: &Expr) -> Expr {
        match target {
            Expr::Index { base, index } => Expr::Index {
                base: Box::new(self.replace_target(base)),
                index: Box::new(self.replace(index)),
            },
            Expr::Field { base, field } => Expr::Field {
                base: Box::new(self.replace_target(base)),
                field: field.clone(),
            },
            other => other.clone(),
        }
    }
}

struct Eliminator<'a> {
    locals: &'a HashMap<String, String>,
    /// Names bound by `let`, `for` or workgroup arrays, per scope; they hide formals.
    shadowed: Vec<HashSet<String>>,
}

impl Eliminator<'_> {
    fn local(&self, base: &Expr) -> Option<&str> {
        let Expr::Var(name) = base else {
            return None;
        };
        if self.shadowed.iter().any(|s| s.contains(name)) {
            return None;
        }
        self.locals.get(name).map(String::as_str)
    }

    fn bind(&mut self, name: &str) {
        self.shadowed
            .last_mut()
            .expect("eliminator always has a scope")
            .insert(name.to_string());
    }

    fn block(&mut self, stmts: &[Stmt]) -> Vec<Stmt> {
        self.shadowed.push(HashSet::new());
        let out = stmts.iter().map(|s| self.stmt(s)).collect();
        self.shadowed.pop();
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
                let init = self.expr(init);
                self.bind(name);
                Stmt::Let {
                    name: name.clone(),
                    mutable: *mutable,
                    ty: ty.clone(),
                    init,
                }
            }
            Stmt::Assign { target, value } => Stmt::Assign {
                value: self.expr(value),
                target: self.target(target),
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
                let (start, end) = (self.expr(start), self.expr(end));
                self.shadowed.push(HashSet::from([var.clone()]));
                let body = self.block(body);
                self.shadowed.pop();
                Stmt::ForRange {
                    var: var.clone(),
                    start,
                    end,
                    body,
                }
            }
            Stmt::Return { value } => Stmt::Return {
                value: value.as_ref().map(|v| self.expr(v)),
            },
            Stmt::WorkgroupArray { name, .. } => {
                self.bind(name);
                stmt.clone()
            }
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
                scratch: scratch.clone(),
                dest: self.target(dest),
            },
            Stmt::WorkgroupSoftmax {
                n,
                buf,
                base,
                count,
                scratch,
            } => Stmt::WorkgroupSoftmax {
                n: *n,
                buf: buf.clone(),
                base: self.expr(base),
                count: self.expr(count),
                scratch: scratch.clone(),
            },
            Stmt::Expr(e) => Stmt::Expr(self.expr(e)),
        }
    }

    fn target(&self, target: &Expr) -> Expr {
        match target {
            Expr::Index { base, index } => match self.local(base) {
                Some(local) => Expr::Var(local.to_string()),
                None => Expr::Index {
                    base: Box::new(self.target(base)),
                    index: Box::new(self.expr(index)),
                },
            },
            Expr::Field { base, field } => Expr::Field {
                base: Box::new(self.target(base)),
                field: field.clone(),
            },
            other => self.expr(other),
        }
    }

    fn expr(&self, expr: &Expr) -> Expr {
        let boxed = |e: &Expr| Box::new(self.expr(e));
        match expr {
            Expr::Index { base, index } => match self.local(base) {
                Some(local) => Expr::Var(local.to_string()),
                None => Expr::Index {
                    base: boxed(base),
                    index: boxed(index),
                },
            },
            Expr::Binary { op, left, right } => Expr::Binary {
                op: *op,
                left: boxed(left),
                right: boxed(right),
            },
            Expr::Field { base, field } => Expr::Field {
                base: boxed(base),
                field: field.clone(),
            },
            Expr::Len { base } => Expr::Len { base: boxed(base) },
            Expr::Rank { base } => Expr::Rank { base: boxed(base) },
            Expr::Dim { base, axis } => Expr::Dim {
                base: boxed(base),
                axis: boxed(axis),
            },
            Expr::Unary { op, expr } => Expr::Unary {
                op: *op,
                expr: boxed(expr),
            },
            Expr::Cast { expr, ty } => Expr::Cast {
                expr: boxed(expr),
                ty: ty.clone(),
            },
            Expr::Call { func, args } => Expr::Call {
                func: *func,
                args: args.iter().map(|a| self.expr(a)).collect(),
            },
            Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_) | Expr::Var(_) => expr.clone(),
        }
    }
}

/// Index expression of a store target, evaluated before the store.
fn target_index(target: &Expr) -> Option<&Expr> {
    match target {
        Expr::Index { index, .. } => Some(index),
        Expr::Field { base, .. } => target_index(base),
        _ => None,
    }
}

fn assign(name: &str, value: Expr) -> Stmt {
    Stmt::Assign {
        target: Expr::Var(name.to_string()),
        value,
    }
}
