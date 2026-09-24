//! Conservative composition of retained definitions into one fused virtual entry.
//!
//! Each constituent is lowered to a module-scope Slang function whose formals keep their
//! own names, and the fused entry calls those functions in order with its own
//! parameters as actual arguments. A constituent's `return` therefore exits only its own
//! body and its locals stay in its own scope. Only workgroup arrays, which Slang places
//! at module scope, are namespaced.
//!
//! Every parcel load and store of every constituent is preserved. Admission proves
//! that removing the dispatch boundary between constituents cannot change any
//! observable parcel state: every parameter written by one constituent and accessed by
//! another is accessed only at the invoking thread's own global id.

use crate::abi::StableHasher;
use crate::{
    assemble_virtual_entry, emit_canonical_compute_source, lower_body, AccessKind, BodyEnv, BuiltinFn, BuiltinMask,
    Expr, KernelDef, KernelId, KernelParam, LoweredBody, ParamCategory, ShaderKernel, SourceMap, Stmt, SymbolKind,
    VirtualEntrySignature, KERNEL_ABI_VERSION,
};
use std::collections::HashMap;
use std::fmt;

/// Workgroup-shared bytes every Goldy backend provides (the WebGPU default limit).
pub const PORTABLE_WORKGROUP_BYTES: u32 = 16 * 1024;

/// Bump when [`FusedDefinition::lower`] changes the program it emits for the same
/// constituents, so [`FusedDefinition::id`] stops matching earlier fused programs.
pub const FUSION_ABI_VERSION: u32 = 1;

/// One constituent of a composition, in execution order.
#[derive(Debug, Clone, Copy)]
pub struct FusionStage<'a> {
    pub definition: &'a ShaderKernel,
    /// Fused parameter index of each formal, in declaration order.
    ///
    /// Formals bound to one actual parcel must share an index: the fused entry binds
    /// each parcel once so a constituent's stores are visible to later loads.
    pub args: &'a [usize],
}

/// Virtual-entry ABI limits a fused entry must respect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionLimits {
    pub resources: usize,
    pub scalars: usize,
    pub workgroup_bytes: u32,
}

/// A composed definition: constituents in order and the fused formals they bind.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedDefinition {
    pub name: String,
    pub workgroup_size: [u32; 3],
    /// Fused formals, in order of first use.
    pub params: Vec<KernelParam>,
    /// Union of the constituents' builtins.
    pub builtins: BuiltinMask,
    /// Constituents in execution order, with workgroup arrays namespaced.
    pub stages: Vec<FusedStage>,
    /// Deduplicated `#[goldy::gpu]` type declarations of every constituent.
    pub type_decls: Vec<String>,
    /// Number of global-id axes that cross-stage parcel indices name (`gid.x` → 1,
    /// `gid.xy` → 2). The dispatch grid must be one thread deep beyond this rank for
    /// those indices to be distinct per thread. `None` when no parcel crosses stages.
    pub dependence_rank: Option<u8>,
}

/// One constituent of a [`FusedDefinition`].
#[derive(Debug, Clone, PartialEq)]
pub struct FusedStage {
    pub definition: ShaderKernel,
    /// [`KernelDef::id`] of the constituent lowered on its own, before namespacing.
    pub kernel: KernelId,
    /// Fused parameter index of each formal.
    pub args: Vec<usize>,
}

/// The constituent scalar a fused scalar parameter binds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarOrigin {
    pub stage: usize,
    pub kernel: String,
    pub formal: String,
    /// Scalar slot of `formal` in the constituent's own entry.
    pub slot: usize,
}

impl fmt::Display for ScalarOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}.{}", self.stage, self.kernel, self.formal)
    }
}

/// Why a set of dispatch invocations cannot be lowered to one physical dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionRejection {
    /// No constituents were supplied.
    Empty,
    /// The constituent is hand-authored Slang with no retained definition.
    OpaqueDefinition {
        stage: usize,
        kernel: String,
    },
    /// The constituent uses a parameter kind composition does not support yet.
    Unsupported {
        stage: usize,
        kernel: String,
        reason: String,
    },
    /// The argument map does not match the constituent formals.
    InvalidArgumentMap(String),
    /// Constituents declare different workgroup sizes.
    WorkgroupSize {
        stage: usize,
        expected: [u32; 3],
        got: [u32; 3],
    },
    /// Constituents dispatch different grids.
    Grid {
        stage: usize,
        expected: [u32; 3],
        got: [u32; 3],
    },
    /// Cross-stage parcel indices name fewer global-id axes than the grid spans.
    GridRank {
        rank: u8,
        threads: [u32; 3],
    },
    /// Formals bound to one fused parameter cannot share a binding.
    IncompatibleBinding {
        param: String,
        reason: String,
    },
    /// An actual argument's parcel identity is unknown, so aliasing cannot be checked.
    UnknownIdentity {
        stage: usize,
        formal: String,
    },
    /// Distinct actual arguments overlap and at least one of them is written.
    PartialAlias {
        stage: usize,
        formal: String,
    },
    /// A parcel written by one constituent is accessed by another at an index other
    /// than the invoking thread's global id (neighbour or cross-workgroup dependence).
    NonLocalDependence {
        stage: usize,
        kernel: String,
        formal: String,
    },
    /// Two constituents declare different `#[goldy::gpu]` types with one name.
    TypeDeclConflict {
        name: String,
    },
    ResourceLimit {
        count: usize,
        max: usize,
    },
    ScalarLimit {
        count: usize,
        max: usize,
    },
    WorkgroupMemory {
        bytes: u32,
        max: u32,
    },
}

impl fmt::Display for FusionRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "no constituents to fuse"),
            Self::OpaqueDefinition { stage, kernel } => write!(
                f,
                "stage {stage} (`{kernel}`) has no retained definition; hand-authored Slang is opaque to fusion"
            ),
            Self::Unsupported { stage, kernel, reason } => {
                write!(f, "stage {stage} (`{kernel}`): {reason} are not composable yet")
            }
            Self::InvalidArgumentMap(reason) => write!(f, "invalid argument map: {reason}"),
            Self::WorkgroupSize { stage, expected, got } => write!(
                f,
                "stage {stage} workgroup size {got:?} differs from {expected:?}"
            ),
            Self::Grid { stage, expected, got } => {
                write!(f, "stage {stage} dispatches {got:?} groups, not {expected:?}")
            }
            Self::GridRank { rank, threads } => write!(
                f,
                "cross-stage parcels are indexed by {rank} global-id axes, but the grid spans {threads:?} threads"
            ),
            Self::IncompatibleBinding { param, reason } => write!(f, "fused parameter `{param}`: {reason}"),
            Self::UnknownIdentity { stage, formal } => {
                write!(f, "stage {stage} `{formal}`: argument has no parcel identity")
            }
            Self::PartialAlias { stage, formal } => write!(
                f,
                "stage {stage} `{formal}` partially overlaps another argument that is written"
            ),
            Self::NonLocalDependence { stage, kernel, formal } => write!(
                f,
                "stage {stage} (`{kernel}`) accesses `{formal}`, which crosses stages, at an index other than its own global id"
            ),
            Self::TypeDeclConflict { name } => write!(f, "constituents declare different GPU types named `{name}`"),
            Self::ResourceLimit { count, max } => write!(f, "fused entry needs {count} resources, limit is {max}"),
            Self::ScalarLimit { count, max } => write!(f, "fused entry needs {count} scalars, limit is {max}"),
            Self::WorkgroupMemory { bytes, max } => {
                write!(f, "fused entry needs {bytes} workgroup-shared bytes, limit is {max}")
            }
        }
    }
}

impl std::error::Error for FusionRejection {}

/// Compose `stages` into one fused definition, or explain why they are inadmissible.
///
/// Parcel identity and dispatch grids are the caller's to check; `stages[..].args`
/// must already give formals bound to one parcel a shared index.
pub fn compose(
    name: &str,
    stages: &[FusionStage<'_>],
    limits: &FusionLimits,
) -> Result<FusedDefinition, FusionRejection> {
    let first = stages.first().ok_or(FusionRejection::Empty)?;
    let workgroup_size = first.definition.workgroup_size;
    let mut origins: Vec<Vec<(usize, usize)>> = Vec::new();
    for (k, stage) in stages.iter().enumerate() {
        let def = stage.definition;
        if def.workgroup_size != workgroup_size {
            return Err(FusionRejection::WorkgroupSize {
                stage: k,
                expected: workgroup_size,
                got: def.workgroup_size,
            });
        }
        if def.params.iter().any(|p| p.is_tensor) {
            return Err(FusionRejection::Unsupported {
                stage: k,
                kernel: def.name.clone(),
                reason: "tensor parameters".into(),
            });
        }
        if stage.args.len() != def.params.len() {
            return Err(FusionRejection::InvalidArgumentMap(format!(
                "stage {k} (`{}`) has {} formals but {} arguments",
                def.name,
                def.params.len(),
                stage.args.len()
            )));
        }
        for (i, &j) in stage.args.iter().enumerate() {
            if origins.len() <= j {
                origins.resize(j + 1, Vec::new());
            }
            origins[j].push((k, i));
        }
    }
    if let Some(j) = origins.iter().position(Vec::is_empty) {
        return Err(FusionRejection::InvalidArgumentMap(format!(
            "fused parameter {j} is bound by no formal"
        )));
    }

    let params = origins
        .iter()
        .map(|o| merge_formals(stages, o))
        .collect::<Result<Vec<_>, _>>()?;
    let resources = params.iter().filter(|p| p.category.is_resource()).count();
    if resources > limits.resources {
        return Err(FusionRejection::ResourceLimit {
            count: resources,
            max: limits.resources,
        });
    }
    let scalars = params.len() - resources;
    if scalars > limits.scalars {
        return Err(FusionRejection::ScalarLimit {
            count: scalars,
            max: limits.scalars,
        });
    }
    let workgroup_bytes: u32 = stages.iter().map(|s| workgroup_array_bytes(s.definition)).sum();
    if workgroup_bytes > limits.workgroup_bytes {
        return Err(FusionRejection::WorkgroupMemory {
            bytes: workgroup_bytes,
            max: limits.workgroup_bytes,
        });
    }
    let type_decls = merge_type_decls(stages)?;
    let dependence_rank = dependence_rank(stages, &params)?;

    let mut builtins = BuiltinMask::NONE;
    for s in stages {
        builtins.global_id |= s.definition.builtins.global_id;
        builtins.local_id |= s.definition.builtins.local_id;
        builtins.workgroup_id |= s.definition.builtins.workgroup_id;
    }
    let stages = stages
        .iter()
        .enumerate()
        .map(|(k, s)| FusedStage {
            definition: s.definition.rename_symbols(|name, kind| match kind {
                SymbolKind::WorkgroupArray => format!("_goldy_k{k}_{name}"),
                SymbolKind::Param | SymbolKind::Local => name.to_string(),
            }),
            kernel: emit_canonical_compute_source(s.definition).id(),
            args: s.args.to_vec(),
        })
        .collect();
    Ok(FusedDefinition {
        name: name.to_string(),
        workgroup_size,
        params,
        builtins,
        stages,
        type_decls,
        dependence_rank,
    })
}

impl FusedDefinition {
    /// Identity of the fused program: the constituent kernel ids, the argument map and
    /// the workgroup size, under the kernel and fusion ABI versions.
    ///
    /// Every other field is derived from these, so definitions with one id lower to the
    /// same program up to source-location comments.
    pub fn id(&self) -> KernelId {
        let mut h = StableHasher::new();
        h.u32(KERNEL_ABI_VERSION).u32(FUSION_ABI_VERSION);
        for &axis in &self.workgroup_size {
            h.u32(axis);
        }
        h.u64(self.stages.len() as u64);
        for stage in &self.stages {
            h.u64(stage.kernel.0).u64(stage.args.len() as u64);
            for &j in &stage.args {
                h.u64(j as u64);
            }
        }
        h.finish()
    }

    /// Origin of every fused scalar, indexed by fused scalar slot.
    pub fn scalar_origins(&self) -> Vec<ScalarOrigin> {
        self.params
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.category.is_resource())
            .map(|(j, _)| {
                let (stage, i) = self
                    .stages
                    .iter()
                    .enumerate()
                    .find_map(|(k, s)| s.args.iter().position(|&a| a == j).map(|i| (k, i)))
                    .expect("every fused parameter is bound by a formal");
                let def = &self.stages[stage].definition;
                ScalarOrigin {
                    stage,
                    kernel: def.name.clone(),
                    formal: def.params[i].name.clone(),
                    slot: def.params[..i].iter().filter(|p| !p.category.is_resource()).count(),
                }
            })
            .collect()
    }

    /// Fused scalar slot that scalar `formal` of stage `stage` binds.
    pub fn scalar_slot(&self, stage: usize, formal: &str) -> Option<usize> {
        self.scalar_origins()
            .iter()
            .position(|o| o.stage == stage && o.formal == formal)
    }

    /// Lower to one canonical `[goldy_compute]` source unit.
    ///
    /// The returned [`KernelDef`] carries no retained definition.
    pub fn lower(&self) -> KernelDef {
        let mut body = LoweredBody::default();
        for (k, stage) in self.stages.iter().enumerate() {
            let def = &stage.definition;
            let function = stage_function_name(k, &def.name);
            let lowered = lower_body(
                &def.body,
                1,
                &BodyEnv {
                    builtins: def.builtins,
                    tensor_slots: HashMap::new(),
                },
            );
            body.workgroup_decls.push_str(&lowered.workgroup_decls);

            let mut formals: Vec<String> = def
                .params
                .iter()
                .zip(&stage.args)
                .map(|(formal, &j)| format!("{} {}", self.params[j].slang_param_type(), formal.name))
                .collect();
            let mut actuals: Vec<String> = stage.args.iter().map(|&j| self.params[j].name.clone()).collect();
            for (ty, name) in builtin_params(def.builtins) {
                formals.push(format!("{ty} {name}"));
                actuals.push(name.to_string());
            }
            let SourceMap { rust_file, rust_line } = &def.source_map;
            body.functions.push_str(&format!(
                "// stage {k}: `{}` ({rust_file}:{rust_line})\nvoid {function}({}) {{\n{}}}\n\n",
                def.name,
                formals.join(", "),
                lowered.stmts
            ));
            body.stmts
                .push_str(&format!("    {function}({});\n", actuals.join(", ")));
        }
        let sig = VirtualEntrySignature {
            workgroup_size: self.workgroup_size,
            params: self.params.clone(),
            builtins: self.builtins,
            type_decls: self.type_decls.clone(),
            source_map: self
                .stages
                .first()
                .map(|s| s.definition.source_map.clone())
                .unwrap_or_default(),
        };
        assemble_virtual_entry(&sig, &body)
    }
}

/// Module-scope name of stage `k`'s function.
///
/// Not `_goldy_`-prefixed: CUDA's packed `DirectSpatial` overload injection skips
/// functions with that prefix, and these helpers must receive the overloads.
fn stage_function_name(k: usize, kernel: &str) -> String {
    format!("goldy_fused_{k}_{kernel}")
}

/// Hidden builtin parameters in the order [`assemble_virtual_entry`] declares them.
fn builtin_params(mask: BuiltinMask) -> impl Iterator<Item = (&'static str, &'static str)> {
    [
        (mask.global_id, "ThreadId", "_goldy_gid"),
        (mask.local_id, "GroupThreadId", "_goldy_lid"),
        (mask.workgroup_id, "GroupId", "_goldy_wid"),
    ]
    .into_iter()
    .filter_map(|(used, ty, name)| used.then_some((ty, name)))
}

fn merge_formals(stages: &[FusionStage<'_>], origins: &[(usize, usize)]) -> Result<KernelParam, FusionRejection> {
    let (k0, i0) = origins[0];
    let formal = |&(k, i): &(usize, usize)| &stages[k].definition.params[i];
    let first = formal(&origins[0]);
    let mut param = first.clone();
    param.name = format!("k{k0}_{}", first.name);
    let incompatible = |reason: String| FusionRejection::IncompatibleBinding {
        param: format!("k{k0}_{}", stages[k0].definition.params[i0].name),
        reason,
    };
    for o in &origins[1..] {
        let other = formal(o);
        if other.slang_type != first.slang_type {
            return Err(incompatible(format!(
                "bound as `{}` and as `{}`",
                first.slang_type, other.slang_type
            )));
        }
        if other.category.is_resource() != first.category.is_resource() {
            return Err(incompatible("bound as both a resource and a scalar".into()));
        }
    }
    if !first.category.is_resource() {
        return Ok(param);
    }

    let categories: Vec<ParamCategory> = origins.iter().map(|o| formal(o).category).collect();
    let buffer = |c: &ParamCategory| {
        matches!(
            c,
            ParamCategory::BufferRead | ParamCategory::BufferReadWrite | ParamCategory::BufferWrite
        )
    };
    param.category = if categories.iter().all(buffer) {
        if categories.iter().all(|c| *c == ParamCategory::BufferRead) {
            ParamCategory::BufferRead
        } else if categories.iter().all(|c| *c == ParamCategory::BufferWrite) {
            ParamCategory::BufferWrite
        } else {
            ParamCategory::BufferReadWrite
        }
    } else if categories.iter().all(|c| *c == first.category) {
        first.category
    } else {
        return Err(incompatible(format!("bound as {categories:?}")));
    };
    param.access = AccessKind::for_category(param.category);
    Ok(param)
}

fn workgroup_array_bytes(def: &ShaderKernel) -> u32 {
    def.body
        .iter()
        .map(|s| match s {
            Stmt::WorkgroupArray { len, .. } => len * 4,
            _ => 0,
        })
        .sum()
}

fn merge_type_decls(stages: &[FusionStage<'_>]) -> Result<Vec<String>, FusionRejection> {
    let mut decls: Vec<String> = Vec::new();
    for decl in stages.iter().flat_map(|s| &s.definition.type_decls) {
        if decls.contains(decl) {
            continue;
        }
        if let Some(name) = declared_struct(decl) {
            if decls.iter().any(|d| declared_struct(d) == Some(name)) {
                return Err(FusionRejection::TypeDeclConflict { name: name.to_string() });
            }
        }
        decls.push(decl.clone());
    }
    Ok(decls)
}

fn declared_struct(decl: &str) -> Option<&str> {
    let rest = decl.lines().find_map(|l| l.trim_start().strip_prefix("struct "))?;
    let end = rest
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

/// How one constituent indexes one of its formals' parcel data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Use {
    reads: bool,
    writes: bool,
    index: Index,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Index {
    /// No element accessed.
    #[default]
    None,
    /// Every element access is at the invoking thread's global id over this many axes.
    Own(u8),
    /// Some element access may touch another thread's element.
    Other,
}

impl Index {
    fn join(self, other: Index) -> Index {
        match (self, other) {
            (Index::None, x) | (x, Index::None) => x,
            (Index::Own(a), Index::Own(b)) if a == b => Index::Own(a),
            _ => Index::Other,
        }
    }
}

/// Reject cross-stage dependences that are not invocation-local; return their rank.
///
/// A fused parameter carries a dependence when two constituents access its elements
/// and at least one writes them. Inside one dispatch, only a thread's own prior
/// stores are guaranteed visible and only its own later stores are guaranteed
/// ordered after its loads, so every such access must name the thread's global id.
fn dependence_rank(stages: &[FusionStage<'_>], params: &[KernelParam]) -> Result<Option<u8>, FusionRejection> {
    let uses: Vec<Vec<Use>> = stages.iter().map(|s| formal_uses(s.definition)).collect();
    let mut rank: Option<u8> = None;
    for j in 0..params.len() {
        let touching: Vec<(usize, usize, Use)> = stages
            .iter()
            .enumerate()
            .flat_map(|(k, s)| {
                let uses = &uses[k];
                s.args
                    .iter()
                    .enumerate()
                    .filter(move |&(_, &a)| a == j)
                    .map(move |(i, _)| (k, i, uses[i]))
            })
            .filter(|(_, _, u)| u.index != Index::None)
            .collect();
        let stage_count = {
            let mut ks: Vec<usize> = touching.iter().map(|t| t.0).collect();
            ks.dedup();
            ks.len()
        };
        if stage_count < 2 || !touching.iter().any(|t| t.2.writes) {
            continue;
        }
        let joined = touching.iter().fold(Index::None, |acc, t| acc.join(t.2.index));
        let Index::Own(r) = joined else {
            let &(k, i, _) = touching
                .iter()
                .find(|t| !matches!(t.2.index, Index::Own(_)))
                .unwrap_or(&touching[0]);
            let def = stages[k].definition;
            return Err(FusionRejection::NonLocalDependence {
                stage: k,
                kernel: def.name.clone(),
                formal: def.params[i].name.clone(),
            });
        };
        rank = Some(rank.map_or(r, |prev| prev.min(r)));
    }
    Ok(rank)
}

fn formal_uses(def: &ShaderKernel) -> Vec<Use> {
    let mut walker = UseWalker {
        def,
        uses: vec![Use::default(); def.params.len()],
        scopes: vec![HashMap::new()],
    };
    walker.block(&def.body);
    walker.uses
}

/// What an in-scope local is known to hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Value {
    /// `gpu::global_id()`.
    GlobalId,
    /// The first `n` global-id axes (`gid.x`, `gid.xy`, `gid.xyz`).
    OwnIndex(u8),
    Unknown,
}

struct UseWalker<'a> {
    def: &'a ShaderKernel,
    uses: Vec<Use>,
    scopes: Vec<HashMap<String, Value>>,
}

impl UseWalker<'_> {
    fn local(&self, name: &str) -> Option<Value> {
        self.scopes.iter().rev().find_map(|s| s.get(name)).copied()
    }

    fn bind(&mut self, name: &str, value: Value) {
        self.scopes
            .last_mut()
            .expect("walker always has a scope")
            .insert(name.to_string(), value);
    }

    /// Resource formal `name` refers to, unless a local shadows it.
    fn resource(&self, name: &str) -> Option<usize> {
        if self.local(name).is_some() {
            return None;
        }
        self.def
            .params
            .iter()
            .position(|p| p.name == name && p.category.is_resource())
    }

    fn resource_expr(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::Var(name) => self.resource(name),
            _ => None,
        }
    }

    fn value(&self, expr: &Expr) -> Value {
        match expr {
            Expr::Call {
                func: BuiltinFn::GlobalId,
                ..
            } => Value::GlobalId,
            Expr::Var(name) => self.local(name).unwrap_or(Value::Unknown),
            Expr::Field { base, field } if self.value(base) == Value::GlobalId => match field.as_str() {
                "x" => Value::OwnIndex(1),
                "xy" => Value::OwnIndex(2),
                "xyz" => Value::OwnIndex(3),
                _ => Value::Unknown,
            },
            Expr::Cast { expr, ty } if ty == "uint" => match self.value(expr) {
                v @ Value::OwnIndex(_) => v,
                _ => Value::Unknown,
            },
            _ => Value::Unknown,
        }
    }

    fn access(&mut self, formal: usize, write: bool, index: &Expr) {
        let at = match self.value(index) {
            Value::OwnIndex(r) => Index::Own(r),
            _ => Index::Other,
        };
        let u = &mut self.uses[formal];
        u.reads |= !write;
        u.writes |= write;
        u.index = u.index.join(at);
    }

    /// The formal is used whole (not through `[index]`): treat as any element, read and written.
    fn escape(&mut self, formal: usize) {
        self.uses[formal] = Use {
            reads: true,
            writes: true,
            index: Index::Other,
        };
    }

    fn block(&mut self, stmts: &[Stmt]) {
        self.scopes.push(HashMap::new());
        for s in stmts {
            self.stmt(s);
        }
        self.scopes.pop();
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                name, mutable, init, ..
            } => {
                self.expr(init);
                let value = if *mutable { Value::Unknown } else { self.value(init) };
                self.bind(name, value);
            }
            Stmt::Assign { target, value } => {
                self.expr(value);
                self.target(target);
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.expr(cond);
                self.block(then_body);
                if let Some(b) = else_body {
                    self.block(b);
                }
            }
            Stmt::While { cond, body } => {
                self.expr(cond);
                self.block(body);
            }
            Stmt::ForRange { var, start, end, body } => {
                self.expr(start);
                self.expr(end);
                self.scopes.push(HashMap::new());
                self.bind(var, Value::Unknown);
                self.block(body);
                self.scopes.pop();
            }
            Stmt::Return { value } => {
                if let Some(v) = value {
                    self.expr(v);
                }
            }
            Stmt::WorkgroupArray { name, .. } => self.bind(name, Value::Unknown),
            Stmt::WorkgroupReduce { val, dest, .. } => {
                self.expr(val);
                self.target(dest);
            }
            Stmt::WorkgroupSoftmax { buf, base, count, .. } => {
                self.expr(base);
                self.expr(count);
                if let Some(f) = self.resource(buf) {
                    self.escape(f);
                }
            }
            Stmt::Expr(e) => self.expr(e),
        }
    }

    fn target(&mut self, target: &Expr) {
        match target {
            Expr::Index { base, index } => {
                self.expr(index);
                match self.resource_expr(base) {
                    Some(f) => self.access(f, true, index),
                    None => self.target(base),
                }
            }
            Expr::Field { base, .. } => self.target(base),
            other => self.expr(other),
        }
    }

    fn expr(&mut self, expr: &Expr) {
        match expr {
            Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_) => {}
            Expr::Var(name) => {
                if let Some(f) = self.resource(name) {
                    self.escape(f);
                }
            }
            Expr::Index { base, index } => {
                self.expr(index);
                match self.resource_expr(base) {
                    Some(f) => self.access(f, false, index),
                    None => self.expr(base),
                }
            }
            Expr::Len { base } | Expr::Rank { base } => {
                if self.resource_expr(base).is_none() {
                    self.expr(base);
                }
            }
            Expr::Dim { base, axis } => {
                self.expr(axis);
                if self.resource_expr(base).is_none() {
                    self.expr(base);
                }
            }
            Expr::Field { base, .. } => self.expr(base),
            Expr::Binary { left, right, .. } => {
                self.expr(left);
                self.expr(right);
            }
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => self.expr(expr),
            Expr::Call { args, .. } => {
                for a in args {
                    self.expr(a);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinOp, ElementType, ScalarType};

    const LIMITS: FusionLimits = FusionLimits {
        resources: 16,
        scalars: 8,
        workgroup_bytes: PORTABLE_WORKGROUP_BYTES,
    };

    fn var(n: &str) -> Expr {
        Expr::Var(n.into())
    }

    fn at(base: &str, index: Expr) -> Expr {
        Expr::Index {
            base: Box::new(var(base)),
            index: Box::new(index),
        }
    }

    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    fn gid_x() -> Expr {
        Expr::Field {
            base: Box::new(Expr::Call {
                func: BuiltinFn::GlobalId,
                args: vec![],
            }),
            field: "x".into(),
        }
    }

    /// `let i = gid.x; if i < count { <dst>[<index(i)>] = <src>[i] <op> k; } return;`
    fn map(name: &str, src: KernelParam, dst: KernelParam, op: BinOp, k: f32, index: Expr) -> ShaderKernel {
        let (s, d) = (src.name.clone(), dst.name.clone());
        ShaderKernel {
            name: name.into(),
            workgroup_size: [64, 1, 1],
            params: vec![src, dst, KernelParam::scalar_param("count", ScalarType::U32)],
            builtins: BuiltinMask {
                global_id: true,
                ..BuiltinMask::NONE
            },
            body: vec![
                Stmt::Let {
                    name: "i".into(),
                    mutable: false,
                    ty: Some("uint".into()),
                    init: gid_x(),
                },
                Stmt::If {
                    cond: bin(BinOp::Ge, var("i"), var("count")),
                    then_body: vec![Stmt::Return { value: None }],
                    else_body: None,
                },
                Stmt::Assign {
                    target: at(&d, index),
                    value: bin(op, at(&s, var("i")), Expr::LitF32(k)),
                },
            ],
            source_map: SourceMap {
                rust_file: format!("{name}.rs"),
                rust_line: 1,
            },
            type_decls: Vec::new(),
        }
    }

    fn scale() -> ShaderKernel {
        map(
            "scale",
            KernelParam::buffer_read("input", ElementType::F32),
            KernelParam::buffer_write("output", ElementType::F32),
            BinOp::Mul,
            2.0,
            var("i"),
        )
    }

    fn bias() -> ShaderKernel {
        map(
            "bias",
            KernelParam::buffer_read("input", ElementType::F32),
            KernelParam::buffer_write("output", ElementType::F32),
            BinOp::Add,
            1.0,
            var("i"),
        )
    }

    fn stage<'a>(definition: &'a ShaderKernel, args: &'a [usize]) -> FusionStage<'a> {
        FusionStage { definition, args }
    }

    #[test]
    fn pointwise_chain_lowers_to_one_entry_with_shared_intermediate() {
        let (a, b) = (scale(), bias());
        let fused = compose("scale+bias", &[stage(&a, &[0, 1, 2]), stage(&b, &[1, 3, 4])], &LIMITS).unwrap();
        let names: Vec<&str> = fused.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["k0_input", "k0_output", "k0_count", "k1_output", "k1_count"]);
        assert_eq!(fused.params[1].category, ParamCategory::BufferReadWrite);
        assert_eq!(fused.params[1].access, Some(AccessKind::ReadWrite));
        assert_eq!(fused.dependence_rank, Some(1));

        let slang = fused.lower().source.canonical_slang;
        assert_eq!(slang.matches("[goldy_compute]").count(), 1, "{slang}");
        assert!(
            slang.contains("void goldy_fused_0_scale(BufRO<float> input, Scattered<float> output, uint count, ThreadId _goldy_gid) {\n    uint i = _goldy_gid.x;\n    if ((i >= count)) {\n        return;\n    }\n"),
            "{slang}"
        );
        assert!(
            slang.contains("void goldy_fused_1_bias(Scattered<float> input, Scattered<float> output, uint count, ThreadId _goldy_gid)"),
            "{slang}"
        );
        assert!(
            slang.contains("void cs_main(BufRO<float> k0_input, Scattered<float> k0_output, uint k0_count, Scattered<float> k1_output, uint k1_count, ThreadId _goldy_gid) {\n    goldy_fused_0_scale(k0_input, k0_output, k0_count, _goldy_gid);\n    goldy_fused_1_bias(k0_output, k1_output, k1_count, _goldy_gid);\n}\n"),
            "{slang}"
        );
    }

    #[test]
    fn fused_identity_follows_constituents_and_argument_map() {
        let (a, b) = (scale(), bias());
        let chain = |args: &[usize]| compose("x", &[stage(&a, &[0, 1, 2]), stage(&b, args)], &LIMITS).unwrap();
        let fused = chain(&[1, 3, 4]);
        assert_eq!(fused.stages[0].kernel, emit_canonical_compute_source(&a).id());
        assert_eq!(fused.stages[1].kernel, emit_canonical_compute_source(&b).id());
        assert_eq!(fused.id(), chain(&[1, 3, 4]).id(), "deterministic");
        assert_eq!(
            fused.id(),
            compose("renamed", &[stage(&a, &[0, 1, 2]), stage(&b, &[1, 3, 4])], &LIMITS)
                .unwrap()
                .id(),
            "the fused name is a label, not identity"
        );
        assert_ne!(
            fused.id(),
            chain(&[3, 4, 5]).id(),
            "sharing a parcel changes the program"
        );
        let swapped = compose("x", &[stage(&b, &[0, 1, 2]), stage(&a, &[1, 3, 4])], &LIMITS).unwrap();
        assert_ne!(fused.id(), swapped.id(), "stage order is identity");
        let mut wide = (scale(), bias());
        wide.0.workgroup_size = [128, 1, 1];
        wide.1.workgroup_size = [128, 1, 1];
        let wide = compose("x", &[stage(&wide.0, &[0, 1, 2]), stage(&wide.1, &[1, 3, 4])], &LIMITS).unwrap();
        assert_ne!(fused.id(), wide.id());
        assert_eq!(format!("{}", fused.id()).len(), 16);
    }

    #[test]
    fn scalar_origins_map_fused_slots_to_constituent_scalars() {
        let (a, b) = (scale(), bias());
        let fused = compose("x", &[stage(&a, &[0, 1, 2]), stage(&b, &[1, 3, 4])], &LIMITS).unwrap();
        let origins = fused.scalar_origins();
        assert_eq!(
            origins,
            [
                ScalarOrigin {
                    stage: 0,
                    kernel: "scale".into(),
                    formal: "count".into(),
                    slot: 0,
                },
                ScalarOrigin {
                    stage: 1,
                    kernel: "bias".into(),
                    formal: "count".into(),
                    slot: 0,
                },
            ]
        );
        assert_eq!(origins[1].to_string(), "1:bias.count");
        assert_eq!(fused.scalar_slot(1, "count"), Some(1));
        assert_eq!(fused.scalar_slot(0, "count"), Some(0));
        assert_eq!(fused.scalar_slot(1, "input"), None, "resources have no scalar slot");
        assert_eq!(fused.scalar_slot(2, "count"), None);
    }

    #[test]
    fn workgroup_arrays_are_namespaced_and_type_decls_deduplicated() {
        let decl = "struct P { uint a; };".to_string();
        let mut a = scale();
        let mut b = bias();
        for k in [&mut a, &mut b] {
            k.body.insert(
                0,
                Stmt::WorkgroupArray {
                    name: "scratch".into(),
                    elem: "float".into(),
                    len: 64,
                },
            );
            k.type_decls = vec![decl.clone()];
        }
        let fused = compose("ab", &[stage(&a, &[0, 1, 2]), stage(&b, &[3, 4, 5])], &LIMITS).unwrap();
        assert_eq!(fused.type_decls, vec![decl]);
        assert_eq!(fused.dependence_rank, None);
        let slang = fused.lower().source.canonical_slang;
        assert!(slang.starts_with("struct P { uint a; };\nimport goldy_exp;"), "{slang}");
        assert!(
            slang.contains(
                "groupshared float _goldy_k0_scratch[64];\ngroupshared float _goldy_k1_scratch[64];\n\n// stage 0"
            ),
            "{slang}"
        );

        b.type_decls = vec!["struct P { float a; };".into()];
        let err = compose("ab", &[stage(&a, &[0, 1, 2]), stage(&b, &[3, 4, 5])], &LIMITS).unwrap_err();
        assert_eq!(err, FusionRejection::TypeDeclConflict { name: "P".into() });
    }

    #[test]
    fn neighbour_read_of_an_intermediate_is_rejected() {
        let a = scale();
        let b = map(
            "shift",
            KernelParam::buffer_read("input", ElementType::F32),
            KernelParam::buffer_write("output", ElementType::F32),
            BinOp::Add,
            0.0,
            var("i"),
        );
        let mut b = b;
        let Stmt::Assign { value, .. } = &mut b.body[2] else {
            unreachable!()
        };
        *value = at("input", bin(BinOp::Add, var("i"), Expr::LitU32(1)));
        let err = compose("a+b", &[stage(&a, &[0, 1, 2]), stage(&b, &[1, 3, 4])], &LIMITS).unwrap_err();
        assert!(
            matches!(&err, FusionRejection::NonLocalDependence { stage: 1, formal, .. } if formal == "input"),
            "{err}"
        );
    }

    #[test]
    fn cross_workgroup_read_of_an_intermediate_is_rejected() {
        let a = scale();
        let mut b = bias();
        let Stmt::Assign { value, .. } = &mut b.body[2] else {
            unreachable!()
        };
        *value = at("input", Expr::LitU32(0));
        let err = compose("a+b", &[stage(&a, &[0, 1, 2]), stage(&b, &[1, 3, 4])], &LIMITS).unwrap_err();
        assert!(
            matches!(err, FusionRejection::NonLocalDependence { stage: 1, .. }),
            "{err}"
        );
    }

    #[test]
    fn shared_read_only_input_needs_no_locality() {
        let a = scale();
        let mut b = bias();
        let Stmt::Assign { value, .. } = &mut b.body[2] else {
            unreachable!()
        };
        *value = at("input", Expr::LitU32(0));
        let fused = compose("a+b", &[stage(&a, &[0, 1, 2]), stage(&b, &[0, 3, 4])], &LIMITS).unwrap();
        assert_eq!(fused.params[0].category, ParamCategory::BufferRead);
        assert_eq!(fused.dependence_rank, None);
    }

    #[test]
    fn mutable_index_is_not_local() {
        let a = scale();
        let mut b = bias();
        b.body[0] = Stmt::Let {
            name: "i".into(),
            mutable: true,
            ty: Some("uint".into()),
            init: gid_x(),
        };
        let err = compose("a+b", &[stage(&a, &[0, 1, 2]), stage(&b, &[1, 3, 4])], &LIMITS).unwrap_err();
        assert!(matches!(err, FusionRejection::NonLocalDependence { .. }), "{err}");
    }

    #[test]
    fn softmax_over_an_intermediate_is_rejected() {
        let a = scale();
        let b = ShaderKernel {
            name: "sm".into(),
            workgroup_size: [64, 1, 1],
            params: vec![KernelParam::buffer_read_write("scores", ElementType::F32)],
            builtins: BuiltinMask {
                local_id: true,
                ..BuiltinMask::NONE
            },
            body: vec![
                Stmt::WorkgroupArray {
                    name: "scratch".into(),
                    elem: "float".into(),
                    len: 64,
                },
                Stmt::WorkgroupSoftmax {
                    n: 64,
                    buf: "scores".into(),
                    base: Expr::LitU32(0),
                    count: Expr::LitU32(4),
                    scratch: "scratch".into(),
                },
            ],
            source_map: SourceMap::default(),
            type_decls: Vec::new(),
        };
        let err = compose("a+sm", &[stage(&a, &[0, 1, 2]), stage(&b, &[1])], &LIMITS).unwrap_err();
        assert!(
            matches!(err, FusionRejection::NonLocalDependence { stage: 1, .. }),
            "{err}"
        );
    }

    #[test]
    fn inadmissible_signatures() {
        let a = scale();
        let mut wide = bias();
        wide.workgroup_size = [128, 1, 1];
        assert!(matches!(
            compose("x", &[stage(&a, &[0, 1, 2]), stage(&wide, &[1, 3, 4])], &LIMITS),
            Err(FusionRejection::WorkgroupSize { stage: 1, .. })
        ));
        assert!(matches!(
            compose("x", &[stage(&a, &[0, 1])], &LIMITS),
            Err(FusionRejection::InvalidArgumentMap(_))
        ));
        assert!(matches!(
            compose("x", &[stage(&a, &[0, 2, 3])], &LIMITS),
            Err(FusionRejection::InvalidArgumentMap(_))
        ));
        assert!(matches!(
            compose("x", &[stage(&a, &[0, 1, 1])], &LIMITS),
            Err(FusionRejection::IncompatibleBinding { .. })
        ));
        let aliased = compose("x", &[stage(&a, &[0, 0, 1])], &LIMITS).unwrap();
        assert_eq!(aliased.params[0].category, ParamCategory::BufferReadWrite);
        let tight = FusionLimits { scalars: 1, ..LIMITS };
        assert_eq!(
            compose("x", &[stage(&a, &[0, 1, 2]), stage(&a, &[1, 3, 4])], &tight),
            Err(FusionRejection::ScalarLimit { count: 2, max: 1 })
        );
        let mut t = scale();
        t.params[0].is_tensor = true;
        assert!(matches!(
            compose("x", &[stage(&t, &[0, 1, 2])], &LIMITS),
            Err(FusionRejection::Unsupported { .. })
        ));
    }
}
