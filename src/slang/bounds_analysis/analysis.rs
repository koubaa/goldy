//! Interval analysis over Slang IR functions.
//!
//! See the module documentation in `mod.rs` for the overall model. This file holds the
//! abstract domain, the per-function fixpoint, path-sensitive refinement, the
//! interprocedural driver, and the access checks.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::ir::{op, Inst, IntShape, IntTy, Module, Types};
use super::{BoundsDiagnostic, BoundsReport};

/// Vulkan guarantees `SubgroupSize` in `[1, 128]`.
const MAX_SUBGROUP_SIZE: i128 = 128;
/// Depth limit for the recursive path-sensitive evaluation.
const REFINE_DEPTH_LIMIT: u32 = 24;
/// Passes a value may keep growing before it is widened.
const WIDEN_AFTER: u32 = 3;
const ASCEND_PASSES: u32 = 64;
const ASCEND_FORCE_TOP_AT: u32 = 48;
const NARROW_PASSES: u32 = 8;
/// Interprocedural limits: call depth, distinct contexts per function, total analyses.
const MAX_CALL_DEPTH: u32 = 12;
const MAX_CONTEXTS_PER_FUNCTION: usize = 24;
const MAX_FUNCTION_ANALYSES: u32 = 4000;

// ============================================================================
// Abstract domain
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Interval {
    pub lo: i128,
    pub hi: i128,
}

impl Interval {
    pub fn new(lo: i128, hi: i128) -> Interval {
        Interval { lo, hi }
    }
    fn point(v: i128) -> Interval {
        Interval { lo: v, hi: v }
    }
    fn join(self, o: Interval) -> Interval {
        Interval::new(self.lo.min(o.lo), self.hi.max(o.hi))
    }
    fn meet(self, o: Interval) -> Option<Interval> {
        let lo = self.lo.max(o.lo);
        let hi = self.hi.min(o.hi);
        (lo <= hi).then_some(Interval::new(lo, hi))
    }
    fn contains(self, o: Interval) -> bool {
        self.lo <= o.lo && o.hi <= self.hi
    }
    fn is_nonneg(self) -> bool {
        self.lo >= 0
    }
}

fn ty_range(ty: IntTy) -> Interval {
    let (lo, hi) = ty.range();
    Interval::new(lo, hi)
}

/// Struct field identity: the `key` instruction (linked, so imported keys resolve to their
/// definition).
type FieldKey = u32;

/// Abstract value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Abs {
    /// One interval per lane of an integer scalar/vector.
    Ints(Vec<Interval>),
    /// Struct with per-field values; absent fields are unknown.
    Struct(BTreeMap<FieldKey, Abs>),
    /// Anything the analysis does not track (floats, resources, unknown structs, ...).
    Opaque,
}

impl Abs {
    fn scalar(i: Interval) -> Abs {
        Abs::Ints(vec![i])
    }
    fn top(shape: IntShape) -> Abs {
        Abs::Ints(vec![ty_range(shape.ty); shape.lanes.max(1)])
    }
    fn lane(&self, k: usize) -> Option<Interval> {
        match self {
            Abs::Ints(v) if v.len() == 1 => v.first().copied(),
            Abs::Ints(v) => v.get(k).copied(),
            _ => None,
        }
    }
    fn as_scalar(&self) -> Option<Interval> {
        match self {
            Abs::Ints(v) if v.len() == 1 => Some(v[0]),
            Abs::Ints(v) => v.iter().copied().reduce(Interval::join),
            _ => None,
        }
    }
    fn join(&self, o: &Abs) -> Abs {
        match (self, o) {
            (Abs::Ints(a), Abs::Ints(b)) => {
                let n = a.len().max(b.len());
                Abs::Ints(
                    (0..n)
                        .map(|k| {
                            let x = a.get(k).or(a.first()).copied().unwrap();
                            let y = b.get(k).or(b.first()).copied().unwrap();
                            x.join(y)
                        })
                        .collect(),
                )
            }
            (Abs::Struct(a), Abs::Struct(b)) => Abs::Struct(
                a.iter()
                    .filter_map(|(k, v)| b.get(k).map(|w| (*k, v.join(w))))
                    .collect(),
            ),
            _ => Abs::Opaque,
        }
    }
    /// Lane-wise refinement: every lane is intersected with the corresponding lane of `o`.
    fn meet(&self, o: &Abs) -> Abs {
        match (self, o) {
            (Abs::Ints(a), Abs::Ints(b)) => Abs::Ints(
                a.iter()
                    .enumerate()
                    .map(|(k, x)| {
                        let y = b.get(k).or(b.first()).copied().unwrap_or(*x);
                        x.meet(y).unwrap_or(*x)
                    })
                    .collect(),
            ),
            (Abs::Ints(_), _) => self.clone(),
            (Abs::Struct(a), Abs::Struct(b)) => {
                let mut out = a.clone();
                for (k, w) in b {
                    match out.get_mut(k) {
                        Some(v) => *v = v.meet(w),
                        None => {
                            out.insert(*k, w.clone());
                        }
                    }
                }
                Abs::Struct(out)
            }
            (Abs::Struct(_), _) => self.clone(),
            _ => o.clone(),
        }
    }
    fn map(&self, f: impl Fn(Interval) -> Interval) -> Abs {
        match self {
            Abs::Ints(v) => Abs::Ints(v.iter().map(|x| f(*x)).collect()),
            _ => Abs::Opaque,
        }
    }
    fn zip(&self, o: &Abs, f: impl Fn(Interval, Interval) -> Interval) -> Abs {
        match (self, o) {
            (Abs::Ints(a), Abs::Ints(b)) => {
                let n = a.len().max(b.len());
                Abs::Ints(
                    (0..n)
                        .map(|k| {
                            let x = a.get(k).or(a.first()).copied().unwrap();
                            let y = b.get(k).or(b.first()).copied().unwrap();
                            f(x, y)
                        })
                        .collect(),
                )
            }
            _ => Abs::Opaque,
        }
    }
    fn field(&self, key: FieldKey) -> Option<&Abs> {
        match self {
            Abs::Struct(m) => m.get(&key),
            _ => None,
        }
    }
    fn is_top_for(&self, shape: Option<IntShape>) -> bool {
        match (self, shape) {
            (Abs::Ints(v), Some(s)) => v.iter().all(|i| *i == ty_range(s.ty)),
            (Abs::Opaque, _) => true,
            _ => false,
        }
    }
}

/// Values that fit the type keep their interval; anything that may wrap collapses to the type range.
pub(super) fn wrap(i: Interval, ty: IntTy) -> Interval {
    let r = ty_range(ty);
    if r.contains(i) {
        i
    } else {
        r
    }
}

/// Reinterpret an interval expressed in `from`'s signedness as `to` (same bit width).
pub(super) fn reinterpret(i: Interval, from: IntTy, to: IntTy) -> Interval {
    if from.signed == to.signed {
        return wrap(i, to);
    }
    let half = 1i128 << (from.bits - 1);
    let full = 1i128 << from.bits;
    let shared = Interval::new(0, half - 1);
    if shared.contains(i) {
        return i;
    }
    if from.signed {
        if i.hi < 0 {
            return Interval::new(i.lo + full, i.hi + full);
        }
    } else if i.lo >= half {
        return Interval::new(i.lo - full, i.hi - full);
    }
    ty_range(to)
}

/// Integer conversion between widths/signedness with C semantics (sign/zero extension,
/// truncation modulo 2^bits).
fn convert(i: Interval, from: IntTy, to: IntTy) -> Interval {
    if to.bits == from.bits {
        return reinterpret(i, from, to);
    }
    if to.bits > from.bits {
        // Extension preserves the value in `from`'s signedness.
        let wide = IntTy {
            bits: to.bits,
            signed: from.signed,
        };
        return reinterpret(i, wide, to);
    }
    // Truncation: exact when the value already fits the destination.
    if ty_range(to).contains(i) {
        i
    } else {
        ty_range(to)
    }
}

fn next_pow2_minus1(v: i128) -> i128 {
    if v <= 0 {
        return 0;
    }
    let mut p = 1i128;
    while p <= v {
        p <<= 1;
    }
    p - 1
}

fn shift_amounts(s: Interval, bits: u32) -> Option<(u32, u32)> {
    if s.lo < 0 || s.hi >= i128::from(bits) {
        return None;
    }
    Some((s.lo as u32, s.hi as u32))
}

fn div_trunc_hull(a: Interval, b: Interval) -> Interval {
    let mut divisors = Vec::new();
    if b.lo <= -1 {
        divisors.push(b.lo);
        divisors.push(b.hi.min(-1));
    }
    if b.hi >= 1 {
        divisors.push(b.lo.max(1));
        divisors.push(b.hi);
    }
    if divisors.is_empty() {
        return Interval::new(0, 0);
    }
    let mut out: Option<Interval> = None;
    for x in [a.lo, a.hi] {
        for &d in &divisors {
            let q = x / d;
            out = Some(out.map_or(Interval::point(q), |o| o.join(Interval::point(q))));
        }
    }
    out.unwrap()
}

// ============================================================================
// Facts (dominating branch conditions)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Rel {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Rel {
    fn negate(self) -> Rel {
        match self {
            Rel::Lt => Rel::Ge,
            Rel::Le => Rel::Gt,
            Rel::Gt => Rel::Le,
            Rel::Ge => Rel::Lt,
            Rel::Eq => Rel::Ne,
            Rel::Ne => Rel::Eq,
        }
    }
    fn flip(self) -> Rel {
        match self {
            Rel::Lt => Rel::Gt,
            Rel::Le => Rel::Ge,
            Rel::Gt => Rel::Lt,
            Rel::Ge => Rel::Le,
            Rel::Eq => Rel::Eq,
            Rel::Ne => Rel::Ne,
        }
    }
}

/// `a rel b`, compared with the given signedness (that of the operand type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Fact {
    a: u32,
    rel: Rel,
    b: u32,
    signed: bool,
}

fn compare_rel(opcode: u32) -> Option<Rel> {
    Some(match opcode {
        op::CMP_EQ => Rel::Eq,
        op::CMP_NE => Rel::Ne,
        op::CMP_LT => Rel::Lt,
        op::CMP_LE => Rel::Le,
        op::CMP_GT => Rel::Gt,
        op::CMP_GE => Rel::Ge,
        _ => return None,
    })
}

/// Scratch state for one refined evaluation query.
#[derive(Default)]
struct EvalCtx {
    memo: HashMap<(u32, Vec<Fact>), Abs>,
    visiting: Vec<u32>,
}

// ============================================================================
// Interprocedural driver
// ============================================================================

/// Generic substitution: `(generic parameter, argument)` pairs, outermost generic first.
type Subst = Vec<(u32, u32)>;

/// Calling context: function definition plus argument abstractions and generic
/// substitution. For pointer (`out`/`inout`/`ref`) parameters the argument abstraction is
/// the pointee's value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CallKey {
    f: u32,
    args: Vec<Abs>,
    subst: Subst,
}

/// What a function does to its caller in one context.
#[derive(Debug, Clone)]
struct Summary {
    ret: Abs,
    /// Value stored through each pointer parameter (`None` for value parameters).
    outs: Vec<Option<Abs>>,
}

impl Summary {
    fn unknown(ret: Abs, nparams: usize) -> Summary {
        Summary {
            ret,
            outs: vec![None; nparams],
        }
    }
}

#[derive(Debug)]
struct MergedDiagnostic {
    diag: BoundsDiagnostic,
}

/// Reporting context for a function analyzed on a path from an entry point.
struct Report {
    /// Call path from the entry point, this function last.
    path: Vec<String>,
    /// Per parameter, what the caller's argument depends on that the analysis cannot bound
    /// (so a callee can say "SV_DispatchThreadID" rather than "parameter `id`").
    param_sources: Vec<Vec<String>>,
}

/// Shared state for one analysis run over a linked module.
pub(super) struct Analyzer<'m> {
    m: &'m Module,
    summaries: RefCell<HashMap<CallKey, Summary>>,
    in_progress: RefCell<Vec<u32>>,
    contexts_per_fn: RefCell<HashMap<u32, usize>>,
    reported: RefCell<HashSet<CallKey>>,
    analyses: Cell<u32>,
    diagnostics: RefCell<BTreeMap<u32, MergedDiagnostic>>,
    checked: RefCell<BTreeSet<u32>>,
}

impl<'m> Analyzer<'m> {
    pub fn new(m: &'m Module) -> Analyzer<'m> {
        Analyzer {
            m,
            summaries: RefCell::new(HashMap::new()),
            in_progress: RefCell::new(Vec::new()),
            contexts_per_fn: RefCell::new(HashMap::new()),
            reported: RefCell::new(HashSet::new()),
            analyses: Cell::new(0),
            diagnostics: RefCell::new(BTreeMap::new()),
            checked: RefCell::new(BTreeSet::new()),
        }
    }

    /// Analyze every entry point of the translation unit; linked library modules are only
    /// analyzed as callees. A translation unit without entry points is treated as a library:
    /// each of its non-generic functions is analyzed with unconstrained parameters.
    pub fn run(self) -> BoundsReport {
        let tu = self.m;
        let entry_points: Vec<u32> = tu
            .function_defs()
            .filter(|&f| tu.in_translation_unit(f) && tu.is_entry_point(f))
            .collect();
        let roots: Vec<u32> = if entry_points.is_empty() {
            tu.function_defs()
                .filter(|&f| tu.in_translation_unit(f))
                .filter(|&f| tu.inst(f).parent.is_some_and(|p| tu.inst(p).op == op::MODULE_INST))
                .collect()
        } else {
            entry_points
        };
        for f in roots {
            let name = tu.name_hint(f).unwrap_or("<anonymous>").to_string();
            let args = self.entry_param_abs(tu, f);
            let key = CallKey {
                f,
                args,
                subst: Vec::new(),
            };
            self.analyze(
                &key,
                Some(Report {
                    path: vec![name],
                    param_sources: Vec::new(),
                }),
            );
        }
        let diagnostics: Vec<BoundsDiagnostic> = self.diagnostics.into_inner().into_values().map(|d| d.diag).collect();
        let checked_accesses = self.checked.into_inner().len();
        BoundsReport {
            proven_safe: checked_accesses - diagnostics.len(),
            checked_accesses,
            diagnostics,
        }
    }

    /// Parameter abstractions for an entry point: system-value semantics give ranges,
    /// everything else is unknown.
    fn entry_param_abs(&self, m: &Module, func: u32) -> Vec<Abs> {
        let Some(entry_block) = m.body(func).find(|&c| m.inst(c).op == op::BLOCK) else {
            return Vec::new();
        };
        let num_threads = m.num_threads(func);
        m.body(entry_block)
            .filter(|&c| m.inst(c).op == op::PARAM)
            .map(|p| {
                let shape = m.type_of(p).and_then(|t| m.int_shape(t));
                match (m.semantic(p), shape, num_threads) {
                    (Some(sem), Some(shape), Some(nt)) if sem.eq_ignore_ascii_case("SV_GroupThreadID") => Abs::Ints(
                        (0..shape.lanes)
                            .map(|k| Interval::new(0, i128::from(nt[k.min(2)]) - 1))
                            .collect(),
                    ),
                    (Some(sem), Some(_), Some(nt)) if sem.eq_ignore_ascii_case("SV_GroupIndex") => {
                        Abs::scalar(Interval::new(0, i128::from(nt[0] * nt[1] * nt[2]) - 1))
                    }
                    (_, Some(shape), _) => Abs::top(shape),
                    _ => Abs::Opaque,
                }
            })
            .collect()
    }

    /// The `func` a call's callee operand names, with the generic substitution accumulated on
    /// the way: through `specialize` (generic arguments, resolved in the caller's own
    /// substitution so nested generics chain) and `lookupWitness` (the entry of the — possibly
    /// substituted — witness table for the requirement key). `None` for anything else, such
    /// as a witness lookup on a table the analysis does not have.
    fn callee_func(&self, callee: u32, resolve: &dyn Fn(u32) -> u32) -> Option<(u32, Subst)> {
        let m = self.m;
        let mut subst = Vec::new();
        let mut cur = resolve(callee);
        for _ in 0..8 {
            let inst = m.inst(cur);
            match inst.op {
                op::SPECIALIZE => {
                    let generic = resolve(inst.operand(0)?);
                    if m.inst(generic).op != op::GENERIC {
                        return None;
                    }
                    let block = m.body(generic).find(|&c| m.inst(c).op == op::BLOCK)?;
                    let params: Vec<u32> = m.body(block).filter(|&c| m.inst(c).op == op::PARAM).collect();
                    for (k, p) in params.iter().enumerate() {
                        if let Some(a) = inst.operand(1 + k) {
                            subst.push((*p, resolve(a)));
                        }
                    }
                    let t = m.inst(m.body(block).last()?);
                    if t.op != op::RETURN_VAL {
                        return None;
                    }
                    cur = resolve(t.operand(0)?);
                }
                op::LOOKUP_WITNESS => {
                    let table = resolve(inst.operand(0)?);
                    let key = resolve(inst.operand(1)?);
                    if m.inst(table).op != op::WITNESS_TABLE {
                        return None;
                    }
                    let entry = m.body(table).find(|&e| {
                        let e = m.inst(e);
                        e.op == op::WITNESS_TABLE_ENTRY && e.operand(0).map(resolve) == Some(key)
                    })?;
                    cur = resolve(m.inst(entry).operand(1)?);
                }
                op::FUNC => return Some((cur, subst)),
                _ => return None,
            }
        }
        None
    }

    /// Resolve the callee of a `call` to a function definition (possibly in a linked
    /// library, possibly the body of a generic) together with the generic substitution, or
    /// `None` when the callee has no analyzable body.
    fn resolve_callee(&self, callee: u32, resolve: &dyn Fn(u32) -> u32) -> Option<(u32, Subst)> {
        let (f, subst) = self.callee_func(callee, resolve)?;
        self.m
            .body(f)
            .any(|c| self.m.inst(c).op == op::BLOCK)
            .then_some((f, subst))
    }

    /// Summary of `key`'s function in that context. When `report` is given, the callee's
    /// accesses are checked and the callee's own calls are reported.
    fn analyze(&self, key: &CallKey, report: Option<Report>) -> Summary {
        let m = self.m;
        let name = m.name_hint(key.f).unwrap_or("<anonymous>").to_string();
        let mut key = key.clone();
        let summarized = self.summaries.borrow().get(&key).cloned();
        let already_reported = report.is_none() || self.reported.borrow().contains(&key);
        if let (Some(s), true) = (&summarized, already_reported) {
            return s.clone();
        }
        let subst_map: HashMap<u32, u32> = key.subst.iter().copied().collect();
        let types = Types {
            m,
            subst: Some(&subst_map),
        };
        let ret_shape = m
            .type_of(key.f)
            .and_then(|t| m.inst(t).operand(0))
            .and_then(|r| types.int_shape(r));
        let nparams = key.args.len();
        let top = move || Summary::unknown(ret_shape.map_or(Abs::Opaque, Abs::top), nparams);
        {
            let mut per_fn = self.contexts_per_fn.borrow_mut();
            let n = per_fn.entry(key.f).or_insert(0);
            if *n >= MAX_CONTEXTS_PER_FUNCTION && summarized.is_none() {
                // Too many distinct contexts: fall back to one unconstrained analysis.
                key.args = key.args.iter().map(|_| Abs::Opaque).collect();
                if let Some(s) = self.summaries.borrow().get(&key) {
                    if report.is_none() || self.reported.borrow().contains(&key) {
                        return s.clone();
                    }
                }
            } else if summarized.is_none() {
                *n += 1;
            }
        }
        if self.in_progress.borrow().iter().any(|f| *f == key.f)
            || self.in_progress.borrow().len() as u32 >= MAX_CALL_DEPTH
            || self.analyses.get() >= MAX_FUNCTION_ANALYSES
        {
            return summarized.unwrap_or_else(top);
        }
        self.analyses.set(self.analyses.get() + 1);
        self.in_progress.borrow_mut().push(key.f);
        if report.is_some() {
            self.reported.borrow_mut().insert(key.clone());
        }
        let mut fa = FunctionAnalysis::new(self, m, key.f, &key.args, subst_map, name);
        fa.fixpoint();
        let summary = Summary {
            ret: fa.return_range(),
            outs: fa.out_param_values(),
        };
        if let Some(report) = report {
            fa.param_sources = report.param_sources;
            fa.check_accesses(&report.path);
            fa.check_calls(&report.path);
        }
        self.in_progress.borrow_mut().pop();
        self.summaries.borrow_mut().insert(key, summary.clone());
        summary
    }

    fn record_check(&self, inst: u32) {
        self.checked.borrow_mut().insert(inst);
    }

    fn record_diagnostic(&self, inst: u32, diag: BoundsDiagnostic) {
        let mut all = self.diagnostics.borrow_mut();
        match all.get_mut(&inst) {
            None => {
                all.insert(inst, MergedDiagnostic { diag });
            }
            Some(existing) => {
                let d = &mut existing.diag;
                d.index_range = match (d.index_range, diag.index_range) {
                    (Some((a, b)), Some((c, e))) => Some((a.min(c), b.max(e))),
                    _ => None,
                };
                for s in diag.depends_on {
                    if !d.depends_on.contains(&s) {
                        d.depends_on.push(s);
                    }
                }
                if d.call_path.len() > diag.call_path.len() {
                    d.call_path = diag.call_path;
                }
            }
        }
    }
}

// ============================================================================
// Per-function analysis
// ============================================================================

struct Block {
    params: Vec<u32>,
    /// Non-parameter instructions in order; the last one is the terminator.
    body: Vec<u32>,
    preds: Vec<usize>,
    succs: Vec<usize>,
}

impl Block {
    fn terminator(&self) -> Option<u32> {
        self.body.last().copied()
    }
}

/// Where a pointer points: a local `var` (or a pointer-typed parameter of this function)
/// plus the field path taken from it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VarPath {
    var: u32,
    fields: Vec<FieldKey>,
    /// A dynamic element index was taken somewhere along the path.
    indexed: bool,
}

impl VarPath {
    fn root(var: u32) -> VarPath {
        VarPath {
            var,
            fields: Vec::new(),
            indexed: false,
        }
    }
}

/// Something written through a pointer into a tracked variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stored {
    /// `store(ptr, value)`.
    Value(u32),
    /// The variable's address was passed as argument `arg` of `call`; the callee's summary
    /// tells what it wrote.
    CallOut { call: u32, arg: usize },
}

struct FunctionAnalysis<'a, 'm> {
    ctx: &'a Analyzer<'m>,
    m: &'m Module,
    func: u32,
    name: String,
    blocks: Vec<Block>,
    block_of: HashMap<u32, usize>,
    /// Block index of each instruction defined in this function (params included).
    inst_block: HashMap<u32, usize>,
    /// Entry-block parameters, positionally.
    params: Vec<u32>,
    param_abs: Vec<Abs>,
    /// Incoming `(pred block, value)` pairs for every non-entry block parameter.
    phis: HashMap<u32, Vec<(usize, u32)>>,
    subst: HashMap<u32, u32>,
    idom: Vec<Option<usize>>,
    rpo: Vec<usize>,
    /// Stores into non-escaping locals, keyed by variable.
    var_stores: HashMap<u32, Vec<(VarPath, Stored)>>,
    escaping_vars: HashSet<u32>,
    /// Structurally equal pure instructions map to one representative (Slang's front-end IR
    /// has no CSE, so `i + 1` in a guard and in the index are different instructions).
    value_number: HashMap<u32, u32>,
    /// `call` instructions whose out-parameter summary is being computed (cycle guard).
    active_call_outs: RefCell<Vec<u32>>,
    ranges: HashMap<u32, Abs>,
    widened: HashSet<u32>,
    block_facts: RefCell<HashMap<usize, Vec<Fact>>>,
    /// See [`Report::param_sources`]; empty unless this function is being reported.
    param_sources: Vec<Vec<String>>,
}

impl<'a, 'm> FunctionAnalysis<'a, 'm> {
    fn new(
        ctx: &'a Analyzer<'m>,
        m: &'m Module,
        func: u32,
        args: &[Abs],
        subst: HashMap<u32, u32>,
        name: String,
    ) -> Self {
        let mut blocks: Vec<Block> = Vec::new();
        let mut block_of = HashMap::new();
        for b in m.body(func).filter(|&c| m.inst(c).op == op::BLOCK) {
            let mut params = Vec::new();
            let mut body = Vec::new();
            for c in m.body(b) {
                if m.inst(c).op == op::PARAM {
                    params.push(c);
                } else {
                    body.push(c);
                }
            }
            block_of.insert(b, blocks.len());
            blocks.push(Block {
                params,
                body,
                preds: Vec::new(),
                succs: Vec::new(),
            });
        }
        let mut inst_block = HashMap::new();
        for (bi, b) in blocks.iter().enumerate() {
            for &p in b.params.iter().chain(b.body.iter()) {
                inst_block.insert(p, bi);
            }
        }
        // CFG edges and block-parameter incomings.
        let mut phis: HashMap<u32, Vec<(usize, u32)>> = HashMap::new();
        let mut edges: Vec<(usize, usize, Vec<u32>)> = Vec::new();
        for (bi, b) in blocks.iter().enumerate() {
            let Some(term) = b.terminator() else { continue };
            let t = m.inst(term);
            let target = |k: usize| t.operand(k).and_then(|x| block_of.get(&x).copied());
            match t.op {
                op::UNCONDITIONAL_BRANCH => {
                    if let Some(s) = target(0) {
                        edges.push((bi, s, t.operands[1..].iter().flatten().copied().collect()));
                    }
                }
                op::LOOP => {
                    if let Some(s) = target(0) {
                        edges.push((
                            bi,
                            s,
                            t.operands[3.min(t.operands.len())..]
                                .iter()
                                .flatten()
                                .copied()
                                .collect(),
                        ));
                    }
                }
                op::CONDITIONAL_BRANCH | op::IF_ELSE => {
                    for k in 1..=2 {
                        if let Some(s) = target(k) {
                            edges.push((bi, s, Vec::new()));
                        }
                    }
                }
                op::SWITCH => {
                    if let Some(s) = target(2) {
                        edges.push((bi, s, Vec::new()));
                    }
                    let mut k = 4;
                    while k < t.operands.len() {
                        if let Some(s) = target(k) {
                            edges.push((bi, s, Vec::new()));
                        }
                        k += 2;
                    }
                }
                _ => {}
            }
        }
        for (from, to, args) in edges {
            if !blocks[from].succs.contains(&to) {
                blocks[from].succs.push(to);
            }
            blocks[to].preds.push(from);
            for (k, &p) in blocks[to].params.iter().enumerate() {
                if let Some(&a) = args.get(k) {
                    phis.entry(p).or_default().push((from, a));
                }
            }
        }
        let (idom, rpo) = dominators(&blocks);
        let params = blocks.first().map(|b| b.params.clone()).unwrap_or_default();
        let types = Types { m, subst: Some(&subst) };
        // For pointer parameters this is the abstraction of the pointee at the call site.
        let param_abs: Vec<Abs> = params
            .iter()
            .enumerate()
            .map(|(k, &p)| {
                let value_ty = m.type_of(p).map(|t| types.pointee(t).unwrap_or(t));
                let shape = value_ty.and_then(|t| types.int_shape(t));
                match args.get(k) {
                    Some(Abs::Ints(v)) if shape.is_some() => Abs::Ints(v.clone()),
                    Some(a @ Abs::Struct(_)) => a.clone(),
                    _ => shape.map_or(Abs::Opaque, Abs::top),
                }
            })
            .collect();
        let mut fa = FunctionAnalysis {
            ctx,
            m,
            func,
            name,
            blocks,
            block_of,
            inst_block,
            params,
            param_abs,
            phis,
            subst,
            idom,
            rpo,
            var_stores: HashMap::new(),
            escaping_vars: HashSet::new(),
            value_number: HashMap::new(),
            active_call_outs: RefCell::new(Vec::new()),
            ranges: HashMap::new(),
            widened: HashSet::new(),
            block_facts: RefCell::new(HashMap::new()),
            param_sources: Vec::new(),
        };
        fa.compute_value_numbers();
        fa.collect_var_stores();
        fa
    }

    /// Representative of `id`'s value-number class (after generic substitution).
    fn canon(&self, id: u32) -> u32 {
        let id = self.resolve(id);
        self.value_number.get(&id).copied().unwrap_or(id)
    }

    fn compute_value_numbers(&mut self) {
        let mut table: HashMap<(u32, Option<u32>, Vec<Option<u32>>), u32> = HashMap::new();
        let mut numbers = HashMap::new();
        for &bi in &self.rpo {
            for &i in &self.blocks[bi].body {
                let inst = self.inst(i);
                if !is_pure(inst.op) {
                    continue;
                }
                let key = (
                    inst.op,
                    inst.ty.map(|t| self.resolve(t)),
                    inst.operands
                        .iter()
                        .map(|o| {
                            o.map(|x| {
                                let x = self.resolve(x);
                                numbers.get(&x).copied().unwrap_or(x)
                            })
                        })
                        .collect(),
                );
                let rep = *table.entry(key).or_insert(i);
                if rep != i {
                    numbers.insert(i, rep);
                }
            }
        }
        self.value_number = numbers;
    }

    /// Whether `p` is a pointer-typed (`out`/`inout`/`ref`) parameter of this function.
    fn is_pointer_param(&self, p: u32) -> bool {
        self.params.contains(&p) && self.type_of(p).is_some_and(|t| self.types().is_pointer_type(t))
    }

    /// Whether the pointer parameter carries a value in from the caller (`inout`/`ref`, not `out`).
    fn param_has_incoming(&self, p: u32) -> bool {
        self.type_of(p)
            .is_some_and(|t| self.inst(self.types().unqualified(t)).op != op::TYPE_OUT_PARAM)
    }

    fn inst(&self, id: u32) -> &'m Inst {
        self.m.inst(id)
    }

    fn resolve(&self, id: u32) -> u32 {
        self.subst.get(&id).copied().unwrap_or(id)
    }

    /// Type queries under this function's generic substitution.
    fn types(&self) -> Types<'_> {
        Types {
            m: self.m,
            subst: Some(&self.subst),
        }
    }

    fn type_of(&self, id: u32) -> Option<u32> {
        self.m.type_of(self.resolve(id)).map(|t| self.resolve(t))
    }

    fn int_shape_of(&self, id: u32) -> Option<IntShape> {
        self.type_of(id).and_then(|t| self.types().int_shape(t))
    }

    fn resolve_callee(&self, callee: u32) -> Option<(u32, Subst)> {
        self.ctx.resolve_callee(callee, &|x| self.resolve(x))
    }

    fn is_local(&self, id: u32) -> bool {
        self.inst_block.contains_key(&id)
    }

    // ------------------------------------------------------------------
    // Local memory
    // ------------------------------------------------------------------

    /// Root variable and field path of a pointer built from a local `var`.
    fn var_path(&self, mut ptr: u32) -> Option<VarPath> {
        let mut fields = Vec::new();
        let mut indexed = false;
        for _ in 0..32 {
            let inst = self.inst(ptr);
            match inst.op {
                op::VAR if self.is_local(ptr) => {
                    fields.reverse();
                    return Some(VarPath {
                        var: ptr,
                        fields,
                        indexed,
                    });
                }
                op::PARAM if self.is_pointer_param(ptr) => {
                    fields.reverse();
                    return Some(VarPath {
                        var: ptr,
                        fields,
                        indexed,
                    });
                }
                op::GET_FIELD_ADDR => {
                    fields.push(self.resolve(inst.operand(1)?));
                    ptr = inst.operand(0)?;
                }
                op::GET_ELEMENT_PTR => {
                    indexed = true;
                    ptr = inst.operand(0)?;
                }
                _ => return None,
            }
        }
        None
    }

    fn collect_var_stores(&mut self) {
        let mut stores: HashMap<u32, Vec<(VarPath, Stored)>> = HashMap::new();
        let mut escaping = HashSet::new();
        for b in &self.blocks {
            for &i in &b.body {
                let inst = self.inst(i);
                let uses_ptr = |k: usize| inst.operand(k).and_then(|p| self.var_path(p));
                match inst.op {
                    op::LOAD | op::GET_FIELD_ADDR | op::GET_ELEMENT_PTR => {}
                    op::STORE => {
                        if let (Some(path), Some(v)) = (uses_ptr(0), inst.operand(1)) {
                            stores.entry(path.var).or_default().push((path, Stored::Value(v)));
                        }
                        // The stored *value* may itself be a pointer to a local.
                        if let Some(p) = uses_ptr(1) {
                            escaping.insert(p.var);
                        }
                    }
                    op::CALL => {
                        // Passing the address to a function with a body is modeled through
                        // its summary; anything else (intrinsics, no body) makes it escape.
                        let modeled = inst
                            .operand(0)
                            .is_some_and(|c| self.intrinsic_name(c).is_none() && self.resolve_callee(c).is_some());
                        for k in 1..inst.operands.len() {
                            let Some(path) = uses_ptr(k) else { continue };
                            if modeled {
                                stores
                                    .entry(path.var)
                                    .or_default()
                                    .push((path, Stored::CallOut { call: i, arg: k - 1 }));
                            } else {
                                escaping.insert(path.var);
                            }
                        }
                    }
                    _ => {
                        // Any other use of the address (or a derived address) is an escape.
                        for k in 0..inst.operands.len() {
                            if let Some(p) = uses_ptr(k) {
                                escaping.insert(p.var);
                            }
                        }
                    }
                }
            }
        }
        self.var_stores = stores;
        self.escaping_vars = escaping;
    }

    /// Value read through `path` from a non-escaping local (or pointer parameter): the join of
    /// everything stored into it, flow-insensitively. `skip_call` leaves out what one call
    /// wrote, to compute the value that call received.
    fn load_local(
        &self,
        path: &VarPath,
        get: &mut dyn FnMut(u32) -> Option<Abs>,
        skip_call: Option<u32>,
    ) -> Option<Abs> {
        if path.indexed || self.escaping_vars.contains(&path.var) {
            return None;
        }
        let mut out: Option<Abs> = None;
        let mut any = false;
        let mut whole: Vec<Abs> = Vec::new();
        if let Some(k) = self.params.iter().position(|&p| p == path.var) {
            if self.param_has_incoming(path.var) {
                whole.push(self.param_abs[k].clone());
            }
        }
        let stores = self.var_stores.get(&path.var).map(Vec::as_slice).unwrap_or(&[]);
        let mut resolved: Vec<(&VarPath, Abs)> = Vec::new();
        for (sp, stored) in stores {
            if sp.indexed {
                return None;
            }
            let v = match *stored {
                Stored::Value(v) => get(v)?,
                Stored::CallOut { call, .. } if Some(call) == skip_call => continue,
                Stored::CallOut { call, arg } => self.call_out(call, arg, get)?,
            };
            resolved.push((sp, v));
        }
        let root = VarPath::root(path.var);
        for w in whole {
            resolved.push((&root, w));
        }
        for (sp, v) in resolved {
            let val = if sp.fields == path.fields {
                v
            } else if sp.fields.len() < path.fields.len() && path.fields.starts_with(&sp.fields) {
                // Whole aggregate stored, field loaded: project.
                let mut cur = v;
                for k in &path.fields[sp.fields.len()..] {
                    cur = cur.field(*k).cloned().unwrap_or(Abs::Opaque);
                }
                cur
            } else if sp.fields.len() > path.fields.len() && sp.fields.starts_with(&path.fields) {
                // Field stored, aggregate loaded: contribute one field of a struct.
                let mut cur = v;
                for k in sp.fields[path.fields.len()..].iter().rev() {
                    cur = Abs::Struct(BTreeMap::from([(*k, cur)]));
                }
                // Merge field-wise rather than join (join would drop disjoint fields).
                any = true;
                out = Some(match out {
                    None => cur,
                    Some(o) => o.meet(&cur),
                });
                continue;
            } else {
                continue;
            };
            any = true;
            out = Some(match out {
                None => val,
                Some(o) => o.join(&val),
            });
        }
        any.then_some(out.unwrap_or(Abs::Opaque))
    }

    /// Argument abstractions of `call`: values as evaluated by `get`, pointers to tracked
    /// variables as the pointee's value before the call.
    fn call_args(&self, call: u32, get: &mut dyn FnMut(u32) -> Option<Abs>) -> Option<Vec<Abs>> {
        let inst = self.inst(call);
        let mut out = Vec::with_capacity(inst.operands.len().saturating_sub(1));
        for a in inst.operands.iter().skip(1).flatten() {
            let a = self.resolve(*a);
            let v = match self.var_path(a) {
                Some(path) => self.load_local(&path, get, Some(call)).unwrap_or(Abs::Opaque),
                None => get(a)?,
            };
            out.push(v);
        }
        Some(out)
    }

    /// What `call` wrote through its `arg`-th argument, from the callee's summary.
    fn call_out(&self, call: u32, arg: usize, get: &mut dyn FnMut(u32) -> Option<Abs>) -> Option<Abs> {
        if self.active_call_outs.borrow().contains(&call) {
            return None;
        }
        let callee = self.inst(call).operand(0)?;
        let (target, subst) = self.resolve_callee(callee)?;
        self.active_call_outs.borrow_mut().push(call);
        let args = self.call_args(call, get);
        self.active_call_outs.borrow_mut().pop();
        let key = CallKey {
            f: target,
            args: args?,
            subst,
        };
        self.ctx.analyze(&key, None).outs.get(arg).cloned().flatten()
    }

    /// Final value of every pointer parameter (what the caller observes after the call).
    fn out_param_values(&self) -> Vec<Option<Abs>> {
        self.params
            .iter()
            .map(|&p| {
                if !self.is_pointer_param(p) {
                    return None;
                }
                let mut get = |o: u32| self.lookup(o);
                Some(
                    self.load_local(&VarPath::root(p), &mut get, None)
                        .unwrap_or(Abs::Opaque),
                )
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Transfer function
    // ------------------------------------------------------------------

    fn constant_abs(&self, id: u32) -> Option<Abs> {
        let id = self.resolve(id);
        let inst = self.inst(id);
        match inst.op {
            op::INT_LIT => {
                let v = self.m.int_lit(id)?;
                let ty = inst.ty.and_then(|t| self.m.int_ty(t));
                let v = match ty {
                    Some(t) if !t.signed && t.bits < 64 => i128::from(v) & ((1i128 << t.bits) - 1),
                    Some(t) if !t.signed => i128::from(v as u64),
                    Some(t) if t.bits < 64 => {
                        let m = 1i128 << t.bits;
                        let x = i128::from(v).rem_euclid(m);
                        if x >= m / 2 {
                            x - m
                        } else {
                            x
                        }
                    }
                    _ => i128::from(v),
                };
                Some(Abs::scalar(Interval::point(v)))
            }
            op::BOOL_LIT => Some(Abs::Opaque),
            _ => None,
        }
    }

    /// Name of a core-module intrinsic when `callee` is one (`min`, `WaveGetLaneCount`, ...).
    fn intrinsic_name(&self, callee: u32) -> Option<&'m str> {
        let (f, _) = self.ctx.callee_func(callee, &|x| self.resolve(x))?;
        if self.m.body(f).any(|c| self.inst(c).op == op::BLOCK) {
            return None;
        }
        let imported_from_core = self.m.import_name(f).is_some_and(|n| n.contains("4core"));
        let intrinsic = self.m.decoration(f, op::DECORATION_TARGET_INTRINSIC).is_some();
        (imported_from_core || intrinsic).then(|| self.m.name_hint(f)).flatten()
    }

    fn intrinsic_call(
        &self,
        name: &str,
        ty: IntTy,
        args: &[u32],
        get: &mut dyn FnMut(u32) -> Option<Abs>,
    ) -> Option<Abs> {
        let r = ty_range(ty);
        let minmax = |get: &mut dyn FnMut(u32) -> Option<Abs>, is_max: bool| {
            let a = get(*args.first()?)?;
            let b = get(*args.get(1)?)?;
            Some(a.zip(&b, |x, y| {
                if is_max {
                    Interval::new(x.lo.max(y.lo), x.hi.max(y.hi))
                } else {
                    Interval::new(x.lo.min(y.lo), x.hi.min(y.hi))
                }
            }))
        };
        Some(match name {
            "min" => minmax(get, false)?,
            "max" => minmax(get, true)?,
            "clamp" if args.len() >= 3 => {
                let x = get(args[0])?;
                let l = get(args[1])?;
                let h = get(args[2])?;
                let lower = x.zip(&l, |x, l| Interval::new(x.lo.max(l.lo), x.hi.max(l.hi)));
                lower.zip(&h, |p, h| Interval::new(p.lo.min(h.lo), p.hi.min(h.hi)))
            }
            "abs" if !args.is_empty() => get(args[0])?.map(|x| {
                let m = x.lo.abs().max(x.hi.abs());
                let lo = if x.lo <= 0 && x.hi >= 0 {
                    0
                } else {
                    x.lo.abs().min(x.hi.abs())
                };
                wrap(Interval::new(lo, m), ty)
            }),
            "sign" => Abs::scalar(Interval::new(-1, 1)),
            "WaveGetLaneCount" => Abs::scalar(Interval::new(1, MAX_SUBGROUP_SIZE)),
            "WaveGetLaneIndex" => Abs::scalar(Interval::new(0, MAX_SUBGROUP_SIZE - 1)),
            "WaveActiveCountBits" | "WavePrefixCountBits" => Abs::scalar(Interval::new(0, MAX_SUBGROUP_SIZE)),
            "countbits" | "firstbithigh" | "firstbitlow" => {
                // Exact on a constant (`firstbitlow(N)` is how generic code spells log2(N)).
                let arg = args.first().and_then(|a| get(*a)).and_then(|a| a.as_scalar());
                let bits = i128::from(ty.bits);
                match arg {
                    Some(c) if c.lo == c.hi && c.lo >= 0 => {
                        let v = c.lo as u128;
                        let exact = match name {
                            "countbits" => i128::from(v.count_ones()),
                            "firstbitlow" if v == 0 => -1,
                            "firstbitlow" => i128::from(v.trailing_zeros()),
                            _ if v == 0 => -1,
                            _ => 127 - i128::from(v.leading_zeros()),
                        };
                        Abs::scalar(Interval::point(if exact < 0 { r.hi.max(-1) } else { exact }))
                    }
                    _ if name == "countbits" => Abs::scalar(Interval::new(0, bits)),
                    _ => Abs::scalar(Interval::new(r.lo.max(-1), bits - 1)),
                }
            }
            _ => return None,
        })
    }

    fn transfer(&self, id: u32, get: &mut dyn FnMut(u32) -> Option<Abs>) -> Option<Abs> {
        let inst = self.inst(id);
        let shape = self.int_shape_of(id);
        let ops: Vec<Option<u32>> = inst.operands.iter().map(|o| o.map(|x| self.resolve(x))).collect();
        let opnd = |k: usize| ops.get(k).copied().flatten();

        // Struct-valued and other non-integer instructions.
        let Some(shape) = shape else {
            return Some(match inst.op {
                op::PARAM => self.param_or_phi(id, get).unwrap_or(Abs::Opaque),
                op::CALL => self.call_result(id, &ops, get).unwrap_or(Abs::Opaque),
                op::LOAD => self.load(opnd(0), get).unwrap_or(Abs::Opaque),
                op::GET_FIELD => {
                    let base = get(opnd(0)?)?;
                    base.field(opnd(1)?).cloned().unwrap_or(Abs::Opaque)
                }
                op::SELECT if ops.len() >= 3 => {
                    let a = get(opnd(1)?)?;
                    let b = get(opnd(2)?)?;
                    a.join(&b)
                }
                _ if self.type_of(id).is_some_and(|t| self.inst(t).op == op::TYPE_STRUCT) => {
                    self.make_struct(id, &ops, get).unwrap_or(Abs::Opaque)
                }
                _ => Abs::Opaque,
            });
        };

        let ty = shape.ty;
        let top = Abs::top(shape);
        let bin = |get: &mut dyn FnMut(u32) -> Option<Abs>, f: &dyn Fn(Interval, Interval) -> Interval| {
            let a = get(opnd(0)?)?;
            let b = get(opnd(1)?)?;
            Some(a.zip(&b, f))
        };
        let fix = |v: Abs| if matches!(v, Abs::Ints(_)) { v } else { Abs::top(shape) };
        match inst.op {
            op::PARAM => self.param_or_phi(id, get).map(fix),
            op::LOAD => Some(self.load(opnd(0), get).map_or(top, fix)),
            op::CALL => Some(self.call_result(id, &ops, get).map_or(top, fix)),
            op::GET_FIELD => {
                let base = get(opnd(0)?)?;
                Some(base.field(opnd(1)?).cloned().map_or(top, fix))
            }
            op::SELECT if ops.len() >= 3 => {
                let a = get(opnd(1)?)?;
                let b = get(opnd(2)?)?;
                Some(fix(a.join(&b)))
            }
            op::SWIZZLE => {
                let base = get(opnd(0)?)?;
                let lanes: Option<Vec<Interval>> = ops[1..]
                    .iter()
                    .map(|c| {
                        let c = self.m.int_lit((*c)?)?;
                        base.lane(usize::try_from(c).ok()?)
                    })
                    .collect();
                Some(lanes.map_or(top, Abs::Ints))
            }
            op::GET_ELEMENT => {
                let base = get(opnd(0)?)?;
                let idx = opnd(1).and_then(|i| self.m.int_lit(i));
                let base_is_vector = opnd(0)
                    .and_then(|b| self.type_of(b))
                    .is_some_and(|t| self.inst(self.types().unqualified(t)).op == op::TYPE_VEC);
                match (idx, base_is_vector) {
                    (Some(c), true) => Some(base.lane(usize::try_from(c).ok()?).map_or(top, Abs::scalar)),
                    _ => Some(top),
                }
            }
            op::MAKE_VECTOR => {
                let mut lanes = Vec::new();
                for o in ops.iter().flatten() {
                    match get(*o)? {
                        Abs::Ints(v) => lanes.extend(v),
                        _ => return Some(top),
                    }
                }
                Some(Abs::Ints(lanes))
            }
            op::INT_CAST | op::BIT_CAST => {
                let src = opnd(0)?;
                let a = get(src)?;
                let Some(from) = self.int_shape_of(src).map(|s| s.ty) else {
                    return Some(top);
                };
                Some(a.map(|i| {
                    if inst.op == op::BIT_CAST {
                        if from.bits == ty.bits {
                            reinterpret(i, from, ty)
                        } else {
                            ty_range(ty)
                        }
                    } else {
                        convert(i, from, ty)
                    }
                }))
            }
            op::NEG => Some(get(opnd(0)?)?.map(|i| wrap(Interval::new(-i.hi, -i.lo), ty))),
            op::ADD => bin(get, &|a, b| wrap(Interval::new(a.lo + b.lo, a.hi + b.hi), ty)),
            op::SUB => bin(get, &|a, b| wrap(Interval::new(a.lo - b.hi, a.hi - b.lo), ty)),
            op::MUL => bin(get, &|a, b| {
                let c = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
                wrap(Interval::new(*c.iter().min().unwrap(), *c.iter().max().unwrap()), ty)
            }),
            op::DIV if !ty.signed => bin(get, &|a, b| {
                if b.hi < 1 {
                    return ty_range(ty);
                }
                Interval::new(a.lo / b.hi, a.hi / b.lo.max(1))
            }),
            op::DIV => bin(get, &|a, b| wrap(div_trunc_hull(a, b), ty)),
            op::IREM if !ty.signed => bin(get, &|a, b| {
                if b.hi < 1 {
                    return ty_range(ty);
                }
                if a.hi < b.lo.max(1) {
                    a
                } else {
                    Interval::new(0, (b.hi - 1).min(a.hi))
                }
            }),
            op::IREM => bin(get, &|a, b| {
                let m = b.lo.abs().max(b.hi.abs());
                if m == 0 {
                    return ty_range(ty);
                }
                let lim = m - 1;
                let lo = if a.lo >= 0 { 0 } else { (-lim).max(a.lo) };
                let hi = if a.hi <= 0 { 0 } else { lim.min(a.hi) };
                Interval::new(lo, hi)
            }),
            op::SHL => bin(get, &|a, s| match shift_amounts(s, ty.bits) {
                Some((lo_s, hi_s)) if a.is_nonneg() => wrap(Interval::new(a.lo << lo_s, a.hi << hi_s), ty),
                _ => ty_range(ty),
            }),
            op::SHR if !ty.signed => bin(get, &|a, s| match shift_amounts(s, ty.bits) {
                Some((lo_s, hi_s)) => Interval::new(a.lo >> hi_s, a.hi >> lo_s),
                None => ty_range(ty),
            }),
            op::SHR => bin(get, &|a, s| match shift_amounts(s, ty.bits) {
                Some((lo_s, hi_s)) => {
                    let c = [a.lo >> lo_s, a.lo >> hi_s, a.hi >> lo_s, a.hi >> hi_s];
                    wrap(Interval::new(*c.iter().min().unwrap(), *c.iter().max().unwrap()), ty)
                }
                None => ty_range(ty),
            }),
            op::AND => bin(get, &|a, b| match (a.is_nonneg(), b.is_nonneg()) {
                (true, true) => Interval::new(0, a.hi.min(b.hi)),
                (true, false) => Interval::new(0, a.hi),
                (false, true) => Interval::new(0, b.hi),
                (false, false) => ty_range(ty),
            }),
            op::OR => bin(get, &|a, b| {
                if a.is_nonneg() && b.is_nonneg() {
                    wrap(Interval::new(a.lo.max(b.lo), next_pow2_minus1(a.hi.max(b.hi))), ty)
                } else {
                    ty_range(ty)
                }
            }),
            op::XOR => bin(get, &|a, b| {
                if a.is_nonneg() && b.is_nonneg() {
                    wrap(Interval::new(0, next_pow2_minus1(a.hi.max(b.hi))), ty)
                } else {
                    ty_range(ty)
                }
            }),
            _ => Some(top),
        }
    }

    /// Entry-block parameter (from the calling context) or block parameter (phi join).
    fn param_or_phi(&self, id: u32, get: &mut dyn FnMut(u32) -> Option<Abs>) -> Option<Abs> {
        if let Some(k) = self.params.iter().position(|&p| p == id) {
            if self.is_pointer_param(id) {
                return Some(Abs::Opaque);
            }
            return Some(self.param_abs[k].clone());
        }
        let incoming = self.phis.get(&id)?;
        let mut out: Option<Abs> = None;
        for &(_, v) in incoming {
            if let Some(a) = get(v) {
                out = Some(out.map_or(a.clone(), |o| o.join(&a)));
            }
        }
        out
    }

    fn load(&self, ptr: Option<u32>, get: &mut dyn FnMut(u32) -> Option<Abs>) -> Option<Abs> {
        let path = self.var_path(ptr?)?;
        self.load_local(&path, get, None)
    }

    fn make_struct(&self, id: u32, ops: &[Option<u32>], get: &mut dyn FnMut(u32) -> Option<Abs>) -> Option<Abs> {
        let inst = self.inst(id);
        // `makeStruct(fields...)` in declaration order of the struct type's `field` children.
        if self.m.op_name(id) != "makeStruct" {
            return None;
        }
        let ty = self.type_of(id)?;
        let keys: Vec<u32> = self
            .m
            .body(ty)
            .filter(|&c| self.inst(c).op == op::FIELD)
            .filter_map(|c| self.inst(c).operand(0))
            .collect();
        if keys.len() != inst.operands.len() {
            return None;
        }
        let mut fields = BTreeMap::new();
        for (k, o) in keys.iter().zip(ops) {
            if let Some(v) = get((*o)?) {
                fields.insert(self.resolve(*k), v);
            }
        }
        Some(Abs::Struct(fields))
    }

    /// Result of a `call`: core intrinsics by name, user functions through the
    /// interprocedural summary, anything else unknown.
    fn call_result(&self, id: u32, ops: &[Option<u32>], get: &mut dyn FnMut(u32) -> Option<Abs>) -> Option<Abs> {
        let callee = ops.first().copied().flatten()?;
        let args: Vec<u32> = ops[1..].iter().flatten().copied().collect();
        if let Some(name) = self.intrinsic_name(callee) {
            if let Some(ty) = self.int_shape_of(id).map(|s| s.ty) {
                return self.intrinsic_call(name, ty, &args, get);
            }
            return None;
        }
        let (target, subst) = self.resolve_callee(callee)?;
        let key = CallKey {
            f: target,
            args: self.call_args(id, get)?,
            subst,
        };
        Some(self.ctx.analyze(&key, None).ret)
    }

    // ------------------------------------------------------------------
    // Global fixpoint
    // ------------------------------------------------------------------

    fn fixpoint(&mut self) {
        let mut all: Vec<u32> = Vec::new();
        for &bi in &self.rpo {
            let b = &self.blocks[bi];
            all.extend(b.params.iter().copied());
            all.extend(b.body.iter().copied());
        }
        let mut grow_count: HashMap<u32, u32> = HashMap::new();
        self.ranges = HashMap::new();

        for pass in 0..ASCEND_PASSES {
            let mut changed = false;
            let force_top = pass >= ASCEND_FORCE_TOP_AT;
            for &id in &all {
                let Some(mut new) = self.recompute(id, force_top) else {
                    continue;
                };
                if let Some(old) = self.ranges.get(&id) {
                    if *old == new {
                        continue;
                    }
                    let n = grow_count.entry(id).or_insert(0);
                    *n += 1;
                    if *n > WIDEN_AFTER {
                        new = self.widen(id, old, &new);
                        self.widened.insert(id);
                    }
                    new = old.join(&new);
                    if *old == new {
                        continue;
                    }
                }
                self.ranges.insert(id, new);
                changed = true;
            }
            if !changed {
                break;
            }
        }

        for _ in 0..NARROW_PASSES {
            let mut changed = false;
            for &id in &all {
                let Some(new) = self.recompute(id, false) else { continue };
                let Some(old) = self.ranges.get(&id) else { continue };
                let narrowed = new.meet(old);
                if narrowed != *old {
                    self.ranges.insert(id, narrowed);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// One transfer step for `id` from the current `self.ranges`; `None` is bottom.
    fn recompute(&self, id: u32, force_top: bool) -> Option<Abs> {
        if let Some(incoming) = self.phis.get(&id) {
            let shape = self.int_shape_of(id);
            if force_top {
                return Some(shape.map_or(Abs::Opaque, Abs::top));
            }
            let block = self.inst_block[&id];
            return self.phi_incoming_join(id, block, incoming);
        }
        let mut get = |o: u32| self.lookup(o);
        self.transfer(id, &mut get)
    }

    fn widen(&self, id: u32, old: &Abs, new: &Abs) -> Abs {
        let (Abs::Ints(o), Abs::Ints(nv)) = (old, new) else {
            return new.clone();
        };
        let Some(ty) = self.int_shape_of(id).map(|s| s.ty) else {
            return new.clone();
        };
        let r = ty_range(ty);
        Abs::Ints(
            nv.iter()
                .enumerate()
                .map(|(k, x)| {
                    let ox = o.get(k).copied().unwrap_or(*x);
                    Interval::new(
                        if x.lo < ox.lo { r.lo } else { x.lo },
                        if x.hi > ox.hi { r.hi } else { x.hi },
                    )
                })
                .collect(),
        )
    }

    /// Join of a block parameter's incoming values, each evaluated under the facts that hold
    /// on its edge, so a counted loop's back edge contributes `[1, n]` rather than a wrapped
    /// increment.
    fn phi_incoming_join(&self, id: u32, block: usize, incoming: &[(usize, u32)]) -> Option<Abs> {
        let mut out: Option<Abs> = None;
        for &(pred, v) in incoming {
            let Some(flat) = self.lookup(v) else {
                continue;
            };
            let val = if matches!(flat, Abs::Ints(_)) && self.ranges.contains_key(&id) {
                let mut edge = Vec::new();
                self.edge_facts(pred, block, &mut edge);
                let mut ctx = EvalCtx {
                    visiting: vec![id],
                    ..Default::default()
                };
                self.eval(v, &edge, 0, &mut ctx).meet(&flat)
            } else {
                flat
            };
            out = Some(out.map_or(val.clone(), |o| o.join(&val)));
        }
        out
    }

    /// Current abstraction of any value: constants, locals from `ranges` (bottom when not yet
    /// computed), everything else unknown.
    fn lookup(&self, id: u32) -> Option<Abs> {
        let id = self.resolve(id);
        if let Some(c) = self.constant_abs(id) {
            return Some(c);
        }
        if let Some(r) = self.ranges.get(&id) {
            return Some(r.clone());
        }
        if self.is_local(id) {
            return None;
        }
        Some(self.int_shape_of(id).map_or(Abs::Opaque, Abs::top))
    }

    fn global(&self, id: u32) -> Abs {
        self.lookup(id)
            .unwrap_or_else(|| self.int_shape_of(id).map_or(Abs::Opaque, Abs::top))
    }

    // ------------------------------------------------------------------
    // Dominating facts
    // ------------------------------------------------------------------

    fn facts_from_condition(&self, cond: u32, polarity: bool, out: &mut Vec<Fact>, depth: u32) {
        if depth > 8 {
            return;
        }
        let cond = self.resolve(cond);
        if let Some(incoming) = self.phis.get(&cond) {
            // `a && b` is lowered to control flow with a bool block parameter. It is true only
            // via an incoming edge whose value can be true; with exactly one such edge, that
            // edge's condition and value both hold.
            let block = self.inst_block[&cond];
            let mut candidates = Vec::new();
            for &(pred, v) in incoming {
                let v = self.resolve(v);
                if self.m.bool_lit(v) == Some(!polarity) {
                    continue;
                }
                candidates.push((pred, v));
            }
            if let [(pred, v)] = candidates[..] {
                self.edge_facts(pred, block, out);
                if self.m.bool_lit(v) != Some(polarity) {
                    self.facts_from_condition(v, polarity, out, depth + 1);
                }
            }
            return;
        }
        let inst = self.inst(cond);
        if let Some(rel) = compare_rel(inst.op) {
            if let (Some(a), Some(b)) = (inst.operand(0), inst.operand(1)) {
                let rel = if polarity { rel } else { rel.negate() };
                let signed = self.int_shape_of(a).is_some_and(|s| s.ty.signed);
                out.push(Fact {
                    a: self.canon(a),
                    rel,
                    b: self.canon(b),
                    signed,
                });
            }
            return;
        }
        match inst.op {
            op::AND | op::LOGICAL_AND if polarity => {
                for o in inst.operands.iter().take(2).flatten() {
                    self.facts_from_condition(*o, true, out, depth + 1);
                }
            }
            op::OR | op::LOGICAL_OR if !polarity => {
                for o in inst.operands.iter().take(2).flatten() {
                    self.facts_from_condition(*o, false, out, depth + 1);
                }
            }
            op::NOT => {
                if let Some(o) = inst.operand(0) {
                    self.facts_from_condition(o, !polarity, out, depth + 1);
                }
            }
            op::SELECT => {
                if let (Some(c), Some(x), Some(y)) = (inst.operand(0), inst.operand(1), inst.operand(2)) {
                    let bconst = |id: u32| self.m.bool_lit(self.resolve(id));
                    if polarity && bconst(y) == Some(false) {
                        self.facts_from_condition(c, true, out, depth + 1);
                        self.facts_from_condition(x, true, out, depth + 1);
                    } else if !polarity && bconst(x) == Some(true) {
                        self.facts_from_condition(c, false, out, depth + 1);
                        self.facts_from_condition(y, false, out, depth + 1);
                    }
                }
            }
            _ => {}
        }
    }

    /// Facts from `pred`'s terminator for the edge `pred -> succ`.
    fn branch_facts(&self, pred: usize, succ: usize, out: &mut Vec<Fact>) {
        let Some(term) = self.blocks[pred].terminator() else {
            return;
        };
        let t = self.inst(term);
        let target = |k: usize| t.operand(k).and_then(|x| self.block_of.get(&x).copied());
        match t.op {
            op::CONDITIONAL_BRANCH | op::IF_ELSE => {
                let (Some(c), Some(tb), Some(fb)) = (t.operand(0), target(1), target(2)) else {
                    return;
                };
                if tb == fb {
                    return;
                }
                if tb == succ {
                    self.facts_from_condition(c, true, out, 0);
                } else if fb == succ {
                    self.facts_from_condition(c, false, out, 0);
                }
            }
            op::SWITCH => {
                let Some(c) = t.operand(0) else { return };
                if target(2) == Some(succ) {
                    return;
                }
                let mut values = Vec::new();
                let mut k = 3;
                while k + 1 < t.operands.len() {
                    if target(k + 1) == Some(succ) {
                        values.push(t.operand(k));
                    }
                    k += 2;
                }
                if let [Some(v)] = values[..] {
                    let signed = self.int_shape_of(c).is_some_and(|s| s.ty.signed);
                    out.push(Fact {
                        a: self.canon(c),
                        rel: Rel::Eq,
                        b: self.canon(v),
                        signed,
                    });
                }
            }
            _ => {}
        }
    }

    /// Facts that hold on entry to `block`: conditions on every dominating single-predecessor edge.
    fn facts_for_block(&self, block: usize) -> Vec<Fact> {
        if let Some(f) = self.block_facts.borrow().get(&block) {
            return f.clone();
        }
        let facts = self.compute_block_facts(block);
        self.block_facts.borrow_mut().insert(block, facts.clone());
        facts
    }

    fn compute_block_facts(&self, block: usize) -> Vec<Fact> {
        let mut facts = Vec::new();
        let mut cur = Some(block);
        while let Some(b) = cur {
            if let [p] = self.blocks[b].preds[..] {
                self.branch_facts(p, b, &mut facts);
            }
            cur = self.idom[b];
        }
        facts
    }

    /// Facts along the CFG edge `pred -> succ`: everything dominating `pred` plus `pred`'s
    /// own branch condition.
    fn edge_facts(&self, pred: usize, succ: usize, out: &mut Vec<Fact>) {
        for f in self.facts_for_block(pred) {
            if !out.contains(&f) {
                out.push(f);
            }
        }
        self.branch_facts(pred, succ, out);
    }

    // ------------------------------------------------------------------
    // Refined (path-sensitive) evaluation
    // ------------------------------------------------------------------

    fn refine_with_fact(
        &self,
        r: Interval,
        ty: IntTy,
        rel: Rel,
        signed: bool,
        other: Interval,
        other_ty: IntTy,
    ) -> Interval {
        let view = ty.with_signed(signed);
        let rv = reinterpret(r, ty, view);
        let ov = reinterpret(other, other_ty, other_ty.with_signed(signed));
        let bound = ty_range(view);
        let refined = match rel {
            Rel::Lt => Interval::new(rv.lo, rv.hi.min(ov.hi - 1)),
            Rel::Le => Interval::new(rv.lo, rv.hi.min(ov.hi)),
            Rel::Gt => Interval::new(rv.lo.max(ov.lo + 1), rv.hi),
            Rel::Ge => Interval::new(rv.lo.max(ov.lo), rv.hi),
            Rel::Eq => Interval::new(rv.lo.max(ov.lo), rv.hi.min(ov.hi)),
            Rel::Ne => {
                if ov.lo == ov.hi && rv.hi > rv.lo {
                    if rv.lo == ov.lo {
                        Interval::new(rv.lo + 1, rv.hi)
                    } else if rv.hi == ov.lo {
                        Interval::new(rv.lo, rv.hi - 1)
                    } else {
                        rv
                    }
                } else {
                    rv
                }
            }
        };
        let Some(refined) = refined.meet(bound) else {
            return r; // infeasible path: keep the unrefined range
        };
        let back = reinterpret(refined, view, ty);
        back.meet(r).unwrap_or(r)
    }

    fn eval(&self, id: u32, facts: &[Fact], depth: u32, ctx: &mut EvalCtx) -> Abs {
        let id = self.resolve(id);
        if let Some(c) = self.constant_abs(id) {
            return c;
        }
        let global = self.global(id);
        let Abs::Ints(_) = global else {
            return global;
        };
        let Some(shape) = self.int_shape_of(id) else {
            return global;
        };
        if depth > REFINE_DEPTH_LIMIT || ctx.visiting.contains(&id) || !self.is_local(id) {
            return self.apply_facts(id, global, shape, facts, depth, ctx);
        }
        let memo_key = (id, facts.to_vec());
        if let Some(v) = ctx.memo.get(&memo_key) {
            return v.clone();
        }
        ctx.visiting.push(id);
        let mut base = global.clone();
        let inst = self.inst(id);
        if let Some(incoming) = self.phis.get(&id) {
            let block = self.inst_block[&id];
            let mut out: Option<Abs> = None;
            for &(pred, v) in incoming {
                let mut edge = facts.to_vec();
                self.edge_facts(pred, block, &mut edge);
                let val = self.eval(v, &edge, depth + 1, ctx);
                out = Some(out.map_or(val.clone(), |o| o.join(&val)));
            }
            if let Some(o) = out {
                base = o.meet(&global);
            }
        } else {
            match inst.op {
                op::SELECT if inst.operands.len() >= 3 => {
                    if let (Some(c), Some(x), Some(y)) = (inst.operand(0), inst.operand(1), inst.operand(2)) {
                        let mut ft = facts.to_vec();
                        self.facts_from_condition(c, true, &mut ft, 0);
                        let mut ff = facts.to_vec();
                        self.facts_from_condition(c, false, &mut ff, 0);
                        let a = self.eval(x, &ft, depth + 1, ctx);
                        let b = self.eval(y, &ff, depth + 1, ctx);
                        base = a.join(&b).meet(&global);
                    }
                }
                op::SUB if inst.operands.len() >= 2 => {
                    if let (Some(a_id), Some(b_id)) = (inst.operand(0), inst.operand(1)) {
                        let a_id = self.canon(a_id);
                        let b_id = self.canon(b_id);
                        let a = self.eval(a_id, facts, depth + 1, ctx);
                        let b = self.eval(b_id, facts, depth + 1, ctx);
                        // Relational rule: a >= b makes a - b non-negative and bounded by
                        // hi(a) - lo(b), so the subtraction cannot wrap. Trusted when the
                        // comparison's signedness matches the type, or both operands are
                        // known non-negative.
                        let mut rel_facts: Vec<(i128, bool)> = Vec::new();
                        for f in facts {
                            let rel = if f.a == a_id && f.b == b_id {
                                f.rel
                            } else if f.a == b_id && f.b == a_id {
                                f.rel.flip()
                            } else {
                                continue;
                            };
                            match rel {
                                Rel::Ge | Rel::Eq => rel_facts.push((0, f.signed)),
                                Rel::Gt => rel_facts.push((1, f.signed)),
                                _ => {}
                            }
                        }
                        let ty = shape.ty;
                        base = a
                            .zip(&b, |x, y| {
                                let raw = Interval::new(x.lo - y.hi, x.hi - y.lo);
                                let min_diff = rel_facts
                                    .iter()
                                    .filter(|(_, signed)| *signed == ty.signed || (x.is_nonneg() && y.is_nonneg()))
                                    .map(|(m, _)| *m)
                                    .max();
                                match min_diff {
                                    Some(m) if raw.hi >= m => wrap(Interval::new(raw.lo.max(m), raw.hi), ty),
                                    _ => wrap(raw, ty),
                                }
                            })
                            .meet(&global);
                    }
                }
                _ => {
                    let mut get = |o: u32| Some(self.eval(o, facts, depth + 1, ctx));
                    if let Some(v) = self.transfer(id, &mut get) {
                        base = v.meet(&global);
                    }
                }
            }
        }
        ctx.visiting.pop();
        let out = self.apply_facts(id, base, shape, facts, depth, ctx);
        ctx.memo.insert(memo_key, out.clone());
        out
    }

    fn apply_facts(
        &self,
        id: u32,
        mut value: Abs,
        shape: IntShape,
        facts: &[Fact],
        depth: u32,
        ctx: &mut EvalCtx,
    ) -> Abs {
        if shape.lanes != 1 || depth > REFINE_DEPTH_LIMIT {
            return value;
        }
        let id = self.canon(id);
        for f in facts {
            let (other, rel) = if f.a == id {
                (f.b, f.rel)
            } else if f.b == id {
                (f.a, f.rel.flip())
            } else {
                continue;
            };
            if other == id {
                continue;
            }
            let Some(other_shape) = self.int_shape_of(other) else {
                continue;
            };
            let other_val = self.eval(other, facts, depth + 1, ctx);
            let Some(ov) = other_val.as_scalar() else { continue };
            let Some(cur) = value.as_scalar() else { continue };
            let refined = self.refine_with_fact(cur, shape.ty, rel, f.signed, ov, other_shape.ty);
            value = Abs::scalar(refined);
        }
        value
    }

    // ------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------

    /// Join of every `return_val` operand, evaluated under the facts of its block.
    fn return_range(&self) -> Abs {
        let mut out: Option<Abs> = None;
        for (bi, b) in self.blocks.iter().enumerate() {
            let Some(term) = b.terminator() else { continue };
            let t = self.inst(term);
            if t.op != op::RETURN_VAL {
                continue;
            }
            let Some(v) = t.operand(0) else { continue };
            let facts = self.facts_for_block(bi);
            let mut ctx = EvalCtx::default();
            let val = self.eval(v, &facts, 0, &mut ctx);
            out = Some(out.map_or(val.clone(), |o| o.join(&val)));
        }
        out.unwrap_or(Abs::Opaque)
    }

    /// Re-analyze every callee in the argument ranges that hold at its call site, reporting
    /// the callee's accesses under `path + callee`.
    fn check_calls(&self, path: &[String]) {
        for &bi in &self.rpo {
            for &i in &self.blocks[bi].body {
                let inst = self.inst(i);
                if inst.op != op::CALL {
                    continue;
                }
                let Some(callee) = inst.operand(0) else { continue };
                if self.intrinsic_name(callee).is_some() {
                    continue;
                }
                let Some((target, subst)) = self.resolve_callee(callee) else {
                    continue;
                };
                let facts = self.facts_for_block(bi);
                let mut ctx = EvalCtx::default();
                let mut get = |a: u32| Some(self.eval(a, &facts, 0, &mut ctx));
                let Some(args) = self.call_args(i, &mut get) else {
                    continue;
                };
                let param_sources = inst
                    .operands
                    .iter()
                    .skip(1)
                    .map(|a| a.map(|a| self.unknown_sources(a)).unwrap_or_default())
                    .collect();
                let key = CallKey { f: target, args, subst };
                let callee_name = self.m.name_hint(target).unwrap_or("<anonymous>").to_string();
                let mut new_path = path.to_vec();
                new_path.push(callee_name);
                self.ctx.analyze(
                    &key,
                    Some(Report {
                        path: new_path,
                        param_sources,
                    }),
                );
            }
        }
    }

    /// Root of an address / aggregate chain and the field path taken from it.
    fn aggregate_root(&self, mut base: u32) -> (u32, Vec<String>) {
        let mut path = Vec::new();
        for _ in 0..32 {
            let inst = self.inst(base);
            match inst.op {
                op::GET_FIELD_ADDR | op::GET_FIELD => {
                    if let Some(key) = inst.operand(1) {
                        path.push(self.m.name_hint(key).unwrap_or("?").to_string());
                    }
                    match inst.operand(0) {
                        Some(b) => base = b,
                        None => break,
                    }
                }
                op::GET_ELEMENT_PTR | op::GET_ELEMENT => {
                    path.push("[]".to_string());
                    match inst.operand(0) {
                        Some(b) => base = b,
                        None => break,
                    }
                }
                op::LOAD => match inst.operand(0) {
                    Some(b) => base = b,
                    None => break,
                },
                // Slang copies by-value aggregate parameters into an unnamed local; name the
                // access after what was copied in when that is unambiguous.
                op::VAR if self.m.name_hint(base).is_none() => {
                    let mut whole = self
                        .var_stores
                        .get(&base)
                        .into_iter()
                        .flatten()
                        .filter(|(p, _)| p.fields.is_empty() && !p.indexed)
                        .filter_map(|(_, st)| match st {
                            Stored::Value(v) => Some(*v),
                            Stored::CallOut { .. } => None,
                        });
                    match (whole.next(), whole.next()) {
                        (Some(v), None) => base = v,
                        _ => break,
                    }
                }
                _ => break,
            }
        }
        path.reverse();
        (base, path)
    }

    fn array_name(&self, base: u32, ty_fallback: &str) -> String {
        let (root, path) = self.aggregate_root(base);
        let mut name = self
            .m
            .name_hint(root)
            .map(str::to_string)
            .unwrap_or_else(|| format!("<unnamed {ty_fallback}>"));
        for p in path {
            if p == "[]" {
                name.push_str("[]");
            } else {
                name.push('.');
                name.push_str(&p);
            }
        }
        name
    }

    fn describe_param(&self, p: u32) -> String {
        let name = self.m.name_hint(p).map(|n| format!(" `{n}`")).unwrap_or_default();
        if let Some(sem) = self.m.semantic(p) {
            return sem.to_string();
        }
        if self.m.is_entry_point(self.func) {
            return format!("an entry-point parameter{name}");
        }
        format!("parameter{name} of `{}`", self.name)
    }

    /// Values the index ultimately depends on that the analysis treats as (nearly) unknown.
    fn unknown_sources(&self, idx: u32) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut seen: HashSet<u32> = HashSet::new();
        let mut work = vec![self.resolve(idx)];
        let mut budget = 256;
        let push = |s: String, out: &mut Vec<String>| {
            if !out.contains(&s) {
                out.push(s);
            }
        };
        while let Some(id) = work.pop() {
            budget -= 1;
            if budget == 0 || out.len() >= 4 {
                break;
            }
            let id = self.resolve(id);
            if !seen.insert(id) || self.constant_abs(id).is_some() {
                continue;
            }
            let inst = self.inst(id);
            if let Some(incoming) = self.phis.get(&id) {
                if self.is_widened(id) {
                    push("a loop-carried value the analysis could not bound".into(), &mut out);
                }
                work.extend(incoming.iter().map(|&(_, v)| v));
                continue;
            }
            match inst.op {
                op::PARAM => {
                    if let Some(k) = self.params.iter().position(|&p| p == id) {
                        if self.param_abs[k].is_top_for(self.int_shape_of(id)) {
                            self.push_param_sources(k, id, None, &mut out);
                        }
                    }
                }
                op::CALL => {
                    let callee = inst.operand(0);
                    let name = callee.and_then(|c| self.callee_display_name(c)).unwrap_or_default();
                    if let Some(name) = callee.and_then(|c| self.intrinsic_name(c)) {
                        push(format!("the result of `{name}()`"), &mut out);
                    } else if callee.is_some_and(|c| self.resolve_callee(c).is_some()) {
                        // Analyzed in context: only worth naming when that gave nothing. An
                        // aggregate result (a constructor, typically) depends on its arguments.
                        match self.global(id) {
                            Abs::Ints(_) if self.global(id).is_top_for(self.int_shape_of(id)) => {
                                push(format!("the result of calling `{name}`"), &mut out);
                            }
                            Abs::Ints(_) => {}
                            _ => work.extend(inst.operands.iter().skip(1).flatten().copied()),
                        }
                    } else {
                        push(format!("the result of calling `{name}` (no body available)"), &mut out);
                    }
                }
                op::CAST_FLOAT_TO_INT => push("a float-to-int conversion".into(), &mut out),
                op::LOAD => {
                    let Some(ptr) = inst.operand(0) else { continue };
                    let (root, _) = self.aggregate_root(ptr);
                    match self.params.iter().position(|&p| p == root) {
                        Some(k) if self.param_sources.get(k).is_some_and(|s| !s.is_empty()) => {
                            self.push_param_sources(k, root, None, &mut out);
                        }
                        _ => push(self.describe_memory(root), &mut out),
                    }
                }
                op::GET_FIELD => {
                    if let Some(b) = inst.operand(0) {
                        let root = self.aggregate_root(b).0;
                        if let Some(pos) = self.params.iter().position(|&p| p == root) {
                            let key = inst.operand(1).map(|k| self.resolve(k));
                            let bounded = key
                                .and_then(|k| self.param_abs[pos].field(k))
                                .is_some_and(|a| !a.is_top_for(self.int_shape_of(id)));
                            if !bounded {
                                let fname = inst.operand(1).and_then(|k| self.m.name_hint(k)).unwrap_or("?");
                                self.push_param_sources(pos, root, Some(fname), &mut out);
                            }
                            continue;
                        }
                        work.push(b);
                    }
                }
                op::SELECT => work.extend(inst.operands.iter().skip(1).flatten().copied()),
                op::SWIZZLE | op::GET_ELEMENT => work.extend(inst.operand(0)),
                _ => {
                    let name = self.m.op_name(id).to_ascii_lowercase();
                    if name.contains("buffer") {
                        push("a buffer load".into(), &mut out);
                    } else if name.contains("texture") || name.contains("image") || name.contains("sample") {
                        push("a texture read".into(), &mut out);
                    } else if name.starts_with("wave") {
                        push("a wave intrinsic result".into(), &mut out);
                    } else {
                        work.extend(inst.operands.iter().flatten().copied());
                    }
                }
            }
        }
        out
    }

    /// Name what parameter `k` (or its field `field`) depends on: the caller's sources when
    /// this function is reported on a call path, the parameter itself otherwise.
    fn push_param_sources(&self, k: usize, param: u32, field: Option<&str>, out: &mut Vec<String>) {
        let from_caller = self.param_sources.get(k).filter(|s| !s.is_empty());
        let sources: Vec<String> = match (from_caller, field) {
            (Some(s), _) => s.clone(),
            (None, Some(f)) => vec![format!("field `{f}` of {}", self.describe_param(param))],
            (None, None) => vec![self.describe_param(param)],
        };
        for s in sources {
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }

    fn callee_display_name(&self, callee: u32) -> Option<String> {
        let (f, _) = self.ctx.callee_func(callee, &|x| self.resolve(x))?;
        self.m.name_hint(f).map(str::to_string)
    }

    fn describe_memory(&self, root: u32) -> String {
        let inst = self.inst(root);
        let name = self.m.name_hint(root).map(|n| format!(" `{n}`")).unwrap_or_default();
        match inst.op {
            op::GLOBAL_VAR => {
                if inst.ty.is_some_and(|t| self.m.is_group_shared(t)) {
                    format!("groupshared memory{name}")
                } else {
                    format!("a global{name}")
                }
            }
            op::GLOBAL_PARAM => format!("a shader parameter{name}"),
            op::VAR => format!("an untracked local{name}"),
            op::PARAM => format!("memory reached through {}", self.describe_param(root)),
            op::RW_STRUCTURED_BUFFER_GET_ELEMENT_PTR => "a buffer load".into(),
            _ => {
                let n = self.m.op_name(root).to_ascii_lowercase();
                if n.contains("buffer") || n.contains("texture") || n.contains("image") {
                    "a buffer load".into()
                } else {
                    "an untracked memory load".into()
                }
            }
        }
    }

    /// Static length of an indexable type, including array lengths that are constant
    /// expressions over generic arguments (`uint prefix_sums[1 << LG_N]`).
    fn aggregate_len(&self, ty: u32) -> Option<u64> {
        if let Some((_, len)) = self.types().indexable(ty) {
            return Some(len);
        }
        let t = self.inst(self.types().unqualified(ty));
        if t.op != op::TYPE_ARRAY {
            return None;
        }
        u64::try_from(self.const_int(t.operand(1)?)?).ok()
    }

    /// Value of a constant expression over literals and generic arguments, when it folds to
    /// one integer.
    fn const_int(&self, id: u32) -> Option<i128> {
        let point = |a: Abs| a.as_scalar().filter(|i| i.lo == i.hi).map(|i| i.lo);
        let id = self.resolve(id);
        if let Some(c) = self.constant_abs(id) {
            return point(c);
        }
        if self.is_local(id) || !is_pure(self.inst(id).op) {
            return None;
        }
        let mut get = |o: u32| self.const_int(o).map(|v| Abs::scalar(Interval::point(v)));
        point(self.transfer(id, &mut get)?)
    }

    /// Whether an address is rooted at a `DebugVar`: with debug info Slang mirrors stores into
    /// by-value aggregate parameters onto a debug variable, which is not a real access.
    fn is_debug_only(&self, base: u32) -> bool {
        self.inst(self.aggregate_root(base).0).op == op::DEBUG_VAR
    }

    fn is_widened(&self, id: u32) -> bool {
        if !self.widened.contains(&id) {
            return false;
        }
        match (self.global(id).as_scalar(), self.int_shape_of(id)) {
            (Some(r), Some(s)) => {
                let t = ty_range(s.ty);
                r.hi == t.hi || (s.ty.signed && r.lo == t.lo)
            }
            _ => false,
        }
    }

    fn check_accesses(&self, path: &[String]) {
        for &bi in &self.rpo {
            for &i in &self.blocks[bi].body {
                let inst = self.inst(i);
                let (base, idx) = match inst.op {
                    op::GET_ELEMENT_PTR | op::GET_ELEMENT => match (inst.operand(0), inst.operand(1)) {
                        (Some(b), Some(x)) => (b, x),
                        _ => continue,
                    },
                    _ => continue,
                };
                let idx = self.resolve(idx);
                if self.constant_abs(idx).is_some() || self.is_debug_only(base) {
                    continue;
                }
                let Some(base_ty) = self.type_of(base) else { continue };
                let agg_ty = if inst.op == op::GET_ELEMENT_PTR {
                    match self.types().pointee(base_ty) {
                        Some(t) => t,
                        None => continue,
                    }
                } else {
                    base_ty
                };
                let Some(len) = self.aggregate_len(agg_ty) else {
                    continue;
                };

                self.ctx.record_check(i);
                let facts = self.facts_for_block(bi);
                let mut ctx = EvalCtx::default();
                let value = self.eval(idx, &facts, 0, &mut ctx);
                let shape = self.int_shape_of(idx);
                let valid = Interval::new(0, i128::from(len) - 1);
                let range = value.as_scalar();
                let is_top = match (range, shape) {
                    (Some(r), Some(s)) => r == ty_range(s.ty),
                    _ => true,
                };
                if let Some(r) = range {
                    if valid.contains(r) {
                        continue;
                    }
                }
                let type_name = self.types().type_name(agg_ty);
                let mut call_path = path.to_vec();
                call_path.pop();
                self.ctx.record_diagnostic(
                    i,
                    BoundsDiagnostic {
                        function: self.name.clone(),
                        call_path,
                        array: self.array_name(base, &type_name),
                        array_length: len,
                        index_range: if is_top { None } else { range.map(|r| (r.lo, r.hi)) },
                        location: self.m.location(i),
                        depends_on: self.unknown_sources(idx),
                    },
                );
            }
        }
    }
}

/// Side-effect-free instructions whose value is a function of their operands alone.
fn is_pure(opcode: u32) -> bool {
    matches!(
        opcode,
        op::ADD
            | op::SUB
            | op::MUL
            | op::DIV
            | op::IREM
            | op::SHL
            | op::SHR
            | op::AND
            | op::OR
            | op::XOR
            | op::NEG
            | op::NOT
            | op::BIT_NOT
            | op::CMP_EQ
            | op::CMP_NE
            | op::CMP_GT
            | op::CMP_LT
            | op::CMP_GE
            | op::CMP_LE
            | op::LOGICAL_AND
            | op::LOGICAL_OR
            | op::SELECT
            | op::SWIZZLE
            | op::GET_FIELD
            | op::GET_ELEMENT
            | op::MAKE_VECTOR
            | op::INT_CAST
            | op::BIT_CAST
            | op::CAST_FLOAT_TO_INT
    )
}

// ============================================================================
// Dominators (Cooper, Harvey & Kennedy)
// ============================================================================

fn dominators(blocks: &[Block]) -> (Vec<Option<usize>>, Vec<usize>) {
    let n = blocks.len();
    let mut idom: Vec<Option<usize>> = vec![None; n];
    if n == 0 {
        return (idom, Vec::new());
    }
    let mut post: Vec<usize> = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    visited[0] = true;
    while let Some(top) = stack.last_mut() {
        let b = top.0;
        let succs = &blocks[b].succs;
        if top.1 < succs.len() {
            let s = succs[top.1];
            top.1 += 1;
            if !visited[s] {
                visited[s] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(b);
            stack.pop();
        }
    }
    let rpo: Vec<usize> = post.iter().rev().copied().collect();
    let mut rpo_index = vec![usize::MAX; n];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_index[b] = i;
    }
    idom[0] = Some(0);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new_idom: Option<usize> = None;
            for &p in &blocks[b].preds {
                if idom[p].is_none() {
                    continue;
                }
                new_idom = Some(match new_idom {
                    None => p,
                    Some(cur) => intersect(&idom, &rpo_index, p, cur),
                });
            }
            if new_idom.is_some() && idom[b] != new_idom {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }
    idom[0] = None;
    (idom, rpo)
}

fn intersect(idom: &[Option<usize>], rpo_index: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while rpo_index[a] > rpo_index[b] {
            a = idom[a].unwrap_or(0);
        }
        while rpo_index[b] > rpo_index[a] {
            b = idom[b].unwrap_or(0);
        }
    }
    a
}
