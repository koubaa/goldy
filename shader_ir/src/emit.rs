//! Emit canonical `[goldy_compute]` Slang from a lowered [`ShaderKernel`].

use crate::{
    tensor_fact_macro, tensor_offset_macro, BinOp, BuiltinFn, BuiltinMask, Expr, KernelDef, KernelParam, MatrixOp,
    ShaderKernel, SourceMap, Stmt, UnaryOp, WorkgroupReduceOp, MATRIX_TILE, TENSOR_FACTS, TENSOR_LAYOUT_SLANG,
    TENSOR_META_PARAM,
};
use std::collections::HashMap;

/// Packed layout + logical-index helpers prepended when the kernel has tensor parameters.
pub const TENSOR_LAYOUT_SLANG_PREAMBLE: &str = r#"struct GoldyTensorLayout {
    uint off;
    uint rank;
    uint numel;
    uint d0;
    uint d1;
    uint d2;
    uint d3;
    uint s0;
    uint s1;
    uint s2;
    uint s3;
    uint flags;
};

uint goldy_tensor_offset(GoldyTensorLayout L, uint i) {
    if ((L.flags & 1u) != 0u)
        return L.off + i;
    uint rest = i;
    uint i3 = rest % L.d3;
    rest = rest / L.d3;
    uint i2 = rest % L.d2;
    rest = rest / L.d2;
    uint i1 = rest % L.d1;
    rest = rest / L.d1;
    uint i0 = rest % L.d0;
    return L.off + i0 * L.s0 + i1 * L.s1 + i2 * L.s2 + i3 * L.s3;
}

uint goldy_tensor_offset1(GoldyTensorLayout L, uint i) {
    return L.off + i * L.s0;
}

uint goldy_tensor_offset2(GoldyTensorLayout L, uint i) {
    if ((L.flags & 1u) != 0u)
        return L.off + i;
    return L.off + (i / L.d1) * L.s0 + (i % L.d1) * L.s1;
}

uint goldy_tensor_offset3(GoldyTensorLayout L, uint i) {
    if ((L.flags & 1u) != 0u)
        return L.off + i;
    uint rest = i / L.d2;
    return L.off + (rest / L.d1) * L.s0 + (rest % L.d1) * L.s1 + (i % L.d2) * L.s2;
}

uint goldy_tensor_dim(GoldyTensorLayout L, uint axis) {
    if (axis == 0) return L.d0;
    if (axis == 1) return L.d1;
    if (axis == 2) return L.d2;
    return L.d3;
}

GoldyTensorLayout goldy_tensor_layout(uint off, uint rank, uint numel, uint d0, uint d1, uint d2, uint d3,
                                      uint s0, uint s1, uint s2, uint s3, uint flags) {
    GoldyTensorLayout L;
    L.off = off;
    L.rank = rank;
    L.numel = numel;
    L.d0 = d0;
    L.d1 = d1;
    L.d2 = d2;
    L.d3 = d3;
    L.s0 = s0;
    L.s1 = s1;
    L.s2 = s2;
    L.s3 = s3;
    L.flags = flags;
    return L;
}

"#;

/// Entry-point name of every generated virtual compute entry.
pub const VIRTUAL_ENTRY_NAME: &str = "cs_main";

/// Preprocessor macro holding the device's fixed subgroup width, when it has one.
///
/// Every subgroup of a one-dimensional workgroup then holds that many consecutive
/// local invocations. Emitted source only tests it with `defined(...)` first, so a
/// compile without it takes the portable form.
pub const SUBGROUP_WIDTH_DEFINE: &str = "GOLDY_SUBGROUP_WIDTH";

/// A tensor parameter's place in the entry's packed [`TENSOR_META_PARAM`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorSlot {
    pub slot: u32,
    /// Rank fixed by the parameter's shape contract, which `record` enforces.
    pub rank: Option<usize>,
}

impl TensorSlot {
    /// Entry-scope local holding the slot's layout, declared by [`assemble_virtual_entry`].
    fn layout(self) -> String {
        format!("_goldy_t{}", self.slot)
    }

    /// Parent-buffer element of logical index `index`.
    ///
    /// A contracted rank below four delinearizes only its own axes. Rank 1 needs no
    /// division, and no contiguity test, because its one stride covers both cases.
    fn offset(self, index: &str) -> String {
        let helper = match self.rank {
            Some(1) => "goldy_tensor_offset1",
            Some(2) => "goldy_tensor_offset2",
            Some(3) => "goldy_tensor_offset3",
            _ => "goldy_tensor_offset",
        };
        format!("{helper}({}, {index})", self.layout())
    }
}

/// Tensor parameter name → its [`TensorSlot`].
pub type TensorSlots = HashMap<String, TensorSlot>;

/// Slot of each tensor parameter in the entry's packed [`TENSOR_META_PARAM`], in declaration order.
pub fn tensor_slot_map(params: &[KernelParam]) -> TensorSlots {
    let mut map = HashMap::new();
    let mut slot = 0u32;
    for p in params {
        if p.is_tensor {
            let rank = p.shape_spec.as_ref().map(|s| s.rank());
            map.insert(p.name.clone(), TensorSlot { slot, rank });
            slot += 1;
        }
    }
    map
}

fn tensor_slot(expr: &Expr, slots: &TensorSlots) -> Option<TensorSlot> {
    match expr {
        Expr::Var(name) => slots.get(name).copied(),
        _ => None,
    }
}

/// Entry-level facts a definition's statements are lowered against.
///
/// A standalone entry uses the definition's own builtins and tensor slots. A composed
/// entry supplies its union of builtins and the tensor slots of its own parameter list,
/// keyed by the names the constituent bodies reference after renaming.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BodyEnv {
    /// Builtins declared by the enclosing entry signature.
    pub builtins: BuiltinMask,
    /// Tensor parameter name → slot in the enclosing entry's tensor metadata.
    pub tensor_slots: TensorSlots,
}

impl BodyEnv {
    pub fn standalone(kernel: &ShaderKernel) -> Self {
        Self {
            builtins: kernel.builtins,
            tensor_slots: tensor_slot_map(&kernel.params),
        }
    }
}

/// A definition's statements lowered to Slang, not yet placed in an entry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoweredBody {
    /// Module-scope `groupshared` declarations, one per line.
    pub workgroup_decls: String,
    /// Module-scope functions the statements call, emitted after `workgroup_decls`.
    pub functions: String,
    /// Statements, indented for their position in the entry.
    pub stmts: String,
}

/// Lower a definition's statements at indentation `level` (1 = directly in the entry body).
pub fn lower_body(body: &[Stmt], level: usize, env: &BodyEnv) -> LoweredBody {
    let mut workgroup_decls = String::new();
    let mut stmts = String::new();
    for stmt in body {
        if let Stmt::WorkgroupArray { name, elem, len } = stmt {
            workgroup_decls.push_str(&format!("groupshared {elem} {name}[{len}];\n"));
        }
        emit_stmt(&mut stmts, stmt, level, &env.builtins, &env.tensor_slots);
    }
    LoweredBody {
        workgroup_decls,
        functions: String::new(),
        stmts,
    }
}

/// Everything a virtual compute entry declares except its body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualEntrySignature {
    pub workgroup_size: [u32; 3],
    /// Declared parameters. [`TENSOR_META_PARAM`] is appended when any is a tensor.
    pub params: Vec<KernelParam>,
    pub builtins: BuiltinMask,
    /// Module-scope type declarations, emitted in order ahead of the import.
    pub type_decls: Vec<String>,
    pub source_map: SourceMap,
}

impl VirtualEntrySignature {
    pub fn standalone(kernel: &ShaderKernel) -> Self {
        Self {
            workgroup_size: kernel.workgroup_size,
            params: kernel.params.clone(),
            builtins: kernel.builtins,
            type_decls: kernel.type_decls.clone(),
            source_map: kernel.source_map.clone(),
        }
    }
}

/// Assemble one `[goldy_compute]` source unit around an already-lowered body.
///
/// The body must have been lowered against [`tensor_slot_map`] of `sig.params`.
/// The returned [`KernelDef`] carries no retained definition.
pub fn assemble_virtual_entry(sig: &VirtualEntrySignature, body: &LoweredBody) -> KernelDef {
    let entry = VIRTUAL_ENTRY_NAME;
    let has_tensors = sig.params.iter().any(|p| p.is_tensor);
    let mut params = sig.params.clone();
    if has_tensors {
        params.push(KernelParam::tensor_meta());
    }
    let mut sig_parts: Vec<String> = params
        .iter()
        .map(|p| format!("{} {}", p.slang_param_type(), p.name))
        .collect();
    // Hidden builtins appended in stable order.
    if sig.builtins.global_id {
        sig_parts.push("ThreadId _goldy_gid".to_string());
    }
    if sig.builtins.local_id {
        sig_parts.push("GroupThreadId _goldy_lid".to_string());
    }
    if sig.builtins.workgroup_id {
        sig_parts.push("GroupId _goldy_wid".to_string());
    }

    let mut types = String::new();
    for decl in &sig.type_decls {
        types.push_str(decl);
        types.push('\n');
    }
    let tensor_count = sig.params.iter().filter(|p| p.is_tensor).count() as u32;
    let mut layout = String::new();
    let mut stmts = String::new();
    if has_tensors {
        layout.push_str(TENSOR_LAYOUT_SLANG_PREAMBLE);
        layout.push_str(&tensor_layout_macros(entry, tensor_count));
        stmts.push_str(&tensor_layout_locals(entry, tensor_count));
    }
    stmts.push_str(&body.stmts);
    let mut shared = body.workgroup_decls.clone();
    if !shared.is_empty() {
        shared.push('\n');
    }
    let functions = &body.functions;
    let [wx, wy, wz] = sig.workgroup_size;
    let sig_text = sig_parts.join(", ");
    let canonical = format!(
        "{types}\
         import goldy_exp;\n\n\
         {layout}\
         {shared}\
         {functions}\
         [goldy_compute]\n\
         [numthreads({wx}, {wy}, {wz})]\n\
         void {entry}({sig_text}) {{\n{stmts}}}\n"
    );
    debug_assert!(!has_tensors || canonical.contains(TENSOR_LAYOUT_SLANG) && canonical.contains(TENSOR_META_PARAM));

    KernelDef::new(
        canonical,
        entry,
        sig.workgroup_size,
        params,
        sig.builtins,
        sig.source_map.clone(),
    )
}

/// Default every layout field of `count` tensor slots to the metadata parcel.
///
/// A virtual-main wrapper may define the offset macros first, and a specialized
/// variant defines the fact macros to literals.
fn tensor_layout_macros(entry: &str, count: u32) -> String {
    let mut out = String::new();
    let mut default = |name: String, field: &str, slot: u32| {
        out.push_str(&format!(
            "#ifndef {name}\n#define {name} {TENSOR_META_PARAM}[{slot}u].{field}\n#endif\n"
        ));
    };
    for slot in 0..count {
        default(tensor_offset_macro(slot), "off", slot);
        for (fact, field) in TENSOR_FACTS.iter().enumerate() {
            default(tensor_fact_macro(entry, slot, fact), field, slot);
        }
    }
    out.push('\n');
    out
}

/// Entry-scope layout locals that [`TensorSlot`] indexing reads.
fn tensor_layout_locals(entry: &str, count: u32) -> String {
    let mut out = String::new();
    for slot in 0..count {
        let mut fields = vec![tensor_offset_macro(slot)];
        fields.extend((0..TENSOR_FACTS.len()).map(|fact| tensor_fact_macro(entry, slot, fact)));
        out.push_str(&format!(
            "    {TENSOR_LAYOUT_SLANG} {} = goldy_tensor_layout({});\n",
            TensorSlot { slot, rank: None }.layout(),
            fields.join(", ")
        ));
    }
    out
}

/// Lower one definition to its standalone canonical compute source (still marked `[goldy_compute]`).
///
/// Backend-specific virtual-main transforms remain responsible for PushLayout /
/// frame-table / CUDA / WebGPU plumbing. The returned [`KernelDef`] retains `kernel`.
pub fn emit_canonical_compute_source(kernel: &ShaderKernel) -> KernelDef {
    let body = lower_body(&kernel.body, 1, &BodyEnv::standalone(kernel));
    let mut def = assemble_virtual_entry(&VirtualEntrySignature::standalone(kernel), &body);
    def.definition = Some(kernel.clone());
    def
}

/// Emit the indented Slang body for a list of statements.
pub fn emit_user_helper_body(body: &[Stmt], builtins: &BuiltinMask) -> String {
    let env = BodyEnv {
        builtins: *builtins,
        tensor_slots: HashMap::new(),
    };
    lower_body(body, 1, &env).stmts
}

fn indent(level: usize) -> String {
    "    ".repeat(level)
}

fn emit_stmt(out: &mut String, stmt: &Stmt, level: usize, builtins: &BuiltinMask, tensor_slots: &TensorSlots) {
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
            out.push_str(&format!(
                "{pad}{ty_s} {name} = {};\n",
                emit_expr(init, builtins, tensor_slots)
            ));
        }
        Stmt::Assign { target, value } => {
            out.push_str(&format!(
                "{pad}{} = {};\n",
                emit_expr(target, builtins, tensor_slots),
                emit_expr(value, builtins, tensor_slots)
            ));
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            out.push_str(&format!("{pad}if ({}) {{\n", emit_expr(cond, builtins, tensor_slots)));
            for s in then_body {
                emit_stmt(out, s, level + 1, builtins, tensor_slots);
            }
            if let Some(else_body) = else_body {
                out.push_str(&format!("{pad}}} else {{\n"));
                for s in else_body {
                    emit_stmt(out, s, level + 1, builtins, tensor_slots);
                }
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::While { cond, body } => {
            out.push_str(&format!(
                "{pad}while ({}) {{\n",
                emit_expr(cond, builtins, tensor_slots)
            ));
            for s in body {
                emit_stmt(out, s, level + 1, builtins, tensor_slots);
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::ForRange { var, start, end, body } => {
            out.push_str(&format!(
                "{pad}for (uint {var} = {}; {var} < {}; ++{var}) {{\n",
                emit_expr(start, builtins, tensor_slots),
                emit_expr(end, builtins, tensor_slots)
            ));
            for s in body {
                emit_stmt(out, s, level + 1, builtins, tensor_slots);
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::Return { value } => {
            if let Some(v) = value {
                out.push_str(&format!("{pad}return {};\n", emit_expr(v, builtins, tensor_slots)));
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
        } => emit_workgroup_reduce(out, level, *op, *n, val, scratch, dest, builtins, tensor_slots),
        Stmt::WorkgroupSoftmax {
            n,
            buf,
            base,
            count,
            scratch,
        } => emit_workgroup_softmax(out, level, *n, buf, base, count, scratch, builtins, tensor_slots),
        Stmt::Matrix(op) => emit_matrix(out, level, op),
        Stmt::Expr(expr) => {
            out.push_str(&format!("{pad}{};\n", emit_expr(expr, builtins, tensor_slots)));
        }
    }
}

fn reduce_steps(n: u32) -> u32 {
    n.trailing_zeros()
}

fn reduce_combine(op: WorkgroupReduceOp, a: &str, b: &str) -> String {
    match op {
        WorkgroupReduceOp::Sum => format!("{a} + {b}"),
        WorkgroupReduceOp::Max => format!("max({a}, {b})"),
    }
}

/// Pairwise tree over blocks of `extent` consecutive lanes of `_goldy_red`, in which
/// every lane of a block ends with the block's value.
///
/// Each step combines the lower half of a block with the upper half, in that operand
/// order on every lane, so all lanes hold the bits lane 0 of the block would.
fn emit_subgroup_tree(out: &mut String, level: usize, op: WorkgroupReduceOp, extent: &str) {
    let pad = indent(level);
    let inner = indent(level + 1);
    let combined = reduce_combine(
        op,
        "(_goldy_upper ? _goldy_other : _goldy_red)",
        "(_goldy_upper ? _goldy_red : _goldy_other)",
    );
    out.push_str(&format!(
        "{pad}for (uint _goldy_s = 1u; _goldy_s < {extent}; _goldy_s <<= 1u) {{\n"
    ));
    // Slang lowers `WaveReadLaneAt` on CUDA by synthesizing an active mask, which puts a
    // warp vote on every branch of the calling function. Every lane of every subgroup
    // runs this loop, so the full mask is exact, and `__shfl_sync` waits for all of them.
    out.push_str(&format!(
        "{inner}float _goldy_other = WaveMaskReadLaneAt(0xFFFFFFFFu, _goldy_red, int(_goldy_lane ^ _goldy_s));\n"
    ));
    out.push_str(&format!("{inner}bool _goldy_upper = (_goldy_lane & _goldy_s) != 0u;\n"));
    out.push_str(&format!("{inner}_goldy_red = {combined};\n"));
    out.push_str(&format!("{pad}}}\n"));
}

/// Both forms combine lane `l` with lane `l + 2^s` for `s = 0, 1, …`, which is the
/// pairwise tree over adjacent lanes, so they produce the same bits. With a
/// [`SUBGROUP_WIDTH_DEFINE`] `w` where `w ≤ n ≤ w²`, the steps below `w` run as subgroup
/// reads, and the `n / w` subgroup partials are exchanged through `scratch` and reduced
/// by the same subgroup tree: two workgroup barriers instead of `2·log2(n) + 1`.
#[allow(clippy::too_many_arguments)]
fn emit_workgroup_reduce(
    out: &mut String,
    level: usize,
    op: WorkgroupReduceOp,
    n: u32,
    val: &Expr,
    scratch: &str,
    dest: &Expr,
    builtins: &BuiltinMask,
    tensor_slots: &TensorSlots,
) {
    let pad = indent(level);
    let inner = indent(level + 1);
    let body = indent(level + 2);
    let nested = indent(level + 3);
    let width = SUBGROUP_WIDTH_DEFINE;
    let steps = reduce_steps(n);
    out.push_str(&format!("{pad}{{\n"));
    out.push_str(&format!(
        "{inner}float _goldy_red = {};\n",
        emit_expr(val, builtins, tensor_slots)
    ));

    out.push_str(&format!("#if defined({width})\n"));
    out.push_str(&format!("{inner}if ({width} <= {n} && {n} <= {width} * {width}) {{\n"));
    out.push_str(&format!("{body}uint _goldy_lane = WaveGetLaneIndex();\n"));
    out.push_str(&format!("{body}uint _goldy_parts = {n}u / uint({width});\n"));
    emit_subgroup_tree(out, level + 2, op, &format!("uint({width})"));
    out.push_str(&format!("{body}if (_goldy_parts > 1u) {{\n"));
    out.push_str(&format!("{nested}if (_goldy_lane == 0u)\n"));
    out.push_str(&format!(
        "{nested}    {scratch}[_goldy_lid.x / uint({width})] = _goldy_red;\n"
    ));
    out.push_str(&format!("{nested}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!(
        "{nested}_goldy_red = {scratch}[_goldy_lane & (_goldy_parts - 1u)];\n"
    ));
    emit_subgroup_tree(out, level + 3, op, "_goldy_parts");
    out.push_str(&format!("{body}}}\n"));
    out.push_str(&format!("{body}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{inner}}} else\n"));
    out.push_str("#endif\n");

    let loop_pad = indent(level + 3);
    out.push_str(&format!("{inner}{{\n"));
    out.push_str(&format!("{body}{scratch}[_goldy_lid.x] = _goldy_red;\n"));
    out.push_str(&format!(
        "{body}for (uint _goldy_s = 0u; _goldy_s < {steps}u; ++_goldy_s) {{\n"
    ));
    out.push_str(&format!("{loop_pad}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{loop_pad}if (_goldy_lid.x + (1u << _goldy_s) < {n}u)\n"));
    out.push_str(&format!(
        "{loop_pad}    _goldy_red = {};\n",
        reduce_combine(op, "_goldy_red", &format!("{scratch}[_goldy_lid.x + (1u << _goldy_s)]"))
    ));
    out.push_str(&format!("{loop_pad}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{loop_pad}{scratch}[_goldy_lid.x] = _goldy_red;\n"));
    out.push_str(&format!("{body}}}\n"));
    out.push_str(&format!("{body}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{body}_goldy_red = {scratch}[0];\n"));
    out.push_str(&format!("{body}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{inner}}}\n"));

    out.push_str(&format!(
        "{inner}{} = _goldy_red;\n",
        emit_expr(dest, builtins, tensor_slots)
    ));
    out.push_str(&format!("{pad}}}\n"));
}

fn tensor_index_expr(buf: &str, index: &str, tensor_slots: &TensorSlots) -> String {
    match tensor_slots.get(buf) {
        Some(slot) => format!("{buf}[{}]", slot.offset(index)),
        None => format!("{buf}[{index}]"),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_workgroup_softmax(
    out: &mut String,
    level: usize,
    n: u32,
    buf: &str,
    base: &Expr,
    count: &Expr,
    scratch: &str,
    builtins: &BuiltinMask,
    tensor_slots: &TensorSlots,
) {
    let pad = indent(level);
    let inner = indent(level + 1);
    let loop_pad = indent(level + 2);
    let base_s = emit_expr(base, builtins, tensor_slots);
    let count_s = emit_expr(count, builtins, tensor_slots);
    let idx = format!("({base_s}) + _goldy_sm_t");
    let at = tensor_index_expr(buf, &idx, tensor_slots);
    out.push_str(&format!("{pad}{{\n"));
    out.push_str(&format!("{inner}float _goldy_sm_max = -1e30;\n"));
    out.push_str(&format!("{inner}uint _goldy_sm_t = _goldy_lid.x;\n"));
    out.push_str(&format!("{inner}while (_goldy_sm_t < {count_s}) {{\n"));
    out.push_str(&format!("{loop_pad}float _goldy_sm_s = {at};\n"));
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
        tensor_slots,
    );
    out.push_str(&format!("{inner}float _goldy_sm_sum = 0.0;\n"));
    out.push_str(&format!("{inner}_goldy_sm_t = _goldy_lid.x;\n"));
    out.push_str(&format!("{inner}while (_goldy_sm_t < {count_s}) {{\n"));
    out.push_str(&format!("{loop_pad}float _goldy_sm_e = exp({at} - _goldy_sm_max);\n"));
    out.push_str(&format!("{loop_pad}{at} = _goldy_sm_e;\n"));
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
        tensor_slots,
    );
    out.push_str(&format!("{inner}_goldy_sm_t = _goldy_lid.x;\n"));
    out.push_str(&format!("{inner}while (_goldy_sm_t < {count_s}) {{\n"));
    out.push_str(&format!("{loop_pad}{at} = {at} / _goldy_sm_sum;\n"));
    out.push_str(&format!("{loop_pad}_goldy_sm_t = _goldy_sm_t + {n}u;\n"));
    out.push_str(&format!("{inner}}}\n"));
    out.push_str(&format!("{inner}GroupMemoryBarrierWithGroupSync();\n"));
    out.push_str(&format!("{pad}}}\n"));
}

fn emit_expr(expr: &Expr, builtins: &BuiltinMask, tensor_slots: &TensorSlots) -> String {
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
        Expr::Field { base, field } => format!("{}.{}", emit_expr(base, builtins, tensor_slots), field),
        Expr::Index { base, index } => {
            let index_s = emit_expr(index, builtins, tensor_slots);
            if let Some(slot) = tensor_slot(base, tensor_slots) {
                let buf = emit_expr(base, builtins, tensor_slots);
                format!("{buf}[{}]", slot.offset(&index_s))
            } else {
                format!("{}[{}]", emit_expr(base, builtins, tensor_slots), index_s)
            }
        }
        Expr::Len { base } => {
            if let Some(slot) = tensor_slot(base, tensor_slots) {
                format!("{}.numel", slot.layout())
            } else {
                format!("goldy_buf_len({})", emit_expr(base, builtins, tensor_slots))
            }
        }
        Expr::Dim { base, axis } => {
            if let Some(slot) = tensor_slot(base, tensor_slots) {
                format!(
                    "goldy_tensor_dim({}, {})",
                    slot.layout(),
                    emit_expr(axis, builtins, tensor_slots)
                )
            } else {
                format!(
                    "goldy_tensor_dim({}[0], {})",
                    emit_expr(base, builtins, tensor_slots),
                    emit_expr(axis, builtins, tensor_slots)
                )
            }
        }
        Expr::Rank { base } => {
            if let Some(slot) = tensor_slot(base, tensor_slots) {
                format!("{}.rank", slot.layout())
            } else {
                "0u".to_string()
            }
        }
        Expr::Binary { op, left, right } => format!(
            "({} {} {})",
            emit_expr(left, builtins, tensor_slots),
            bin_op_slang(*op),
            emit_expr(right, builtins, tensor_slots)
        ),
        Expr::Unary { op, expr } => format!("({}{})", unary_op_slang(*op), emit_expr(expr, builtins, tensor_slots)),
        Expr::Call { func, args } => emit_call(*func, args, builtins, tensor_slots),
        Expr::Cast { expr, ty } => format!("(({}){})", ty, emit_expr(expr, builtins, tensor_slots)),
    }
}

fn emit_call(func: BuiltinFn, args: &[Expr], builtins: &BuiltinMask, tensor_slots: &TensorSlots) -> String {
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
        BuiltinFn::Abs => format!("abs({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Min => format!("min({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Max => format!("max({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Floor => format!("floor({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Ceil => format!("ceil({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Sqrt => format!("sqrt({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Sin => format!("sin({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Cos => format!("cos({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Exp => format!("exp({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Log => format!("log({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Pow => format!("pow({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Length => format!("length({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::WorkgroupBarrier => {
            assert!(args.is_empty());
            "GroupMemoryBarrierWithGroupSync()".to_string()
        }
        BuiltinFn::Float2 => format!("float2({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Float3 => format!("float3({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Float4 => format!("float4({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::Uint2 => format!("uint2({})", join_args(args, builtins, tensor_slots)),
        BuiltinFn::SubgroupLane => {
            assert!(args.is_empty());
            "WaveGetLaneIndex()".to_string()
        }
        BuiltinFn::SubgroupRead => {
            assert_eq!(args.len(), 2);
            format!("WaveReadLaneAt({})", join_args(args, builtins, tensor_slots))
        }
    }
}

fn matrix_type(elem: &str, usage: &str) -> String {
    format!(
        "linalg.CoopMat<{elem}, MemoryScope.Subgroup, {MATRIX_TILE}, {MATRIX_TILE}, linalg.CoopMatMatrixUse.{usage}>"
    )
}

fn emit_matrix(out: &mut String, level: usize, op: &MatrixOp) {
    let pad = indent(level);
    let accumulator = matrix_type("float", "MatrixAccumulator");
    let row_major = "linalg.CoopMatMatrixLayout.RowMajor";
    match op {
        MatrixOp::Accumulator { name } => out.push_str(&format!("{pad}{accumulator} {name} = {accumulator}(0.0);\n")),
        MatrixOp::MulAdd { acc, a, b } => {
            let a = format!(
                "{}.Load<{row_major}>({a}, 0, {MATRIX_TILE})",
                matrix_type("half", "MatrixA")
            );
            let b = format!(
                "{}.Load<{row_major}>({b}, 0, {MATRIX_TILE})",
                matrix_type("half", "MatrixB")
            );
            out.push_str(&format!(
                "{pad}{acc} = linalg.coopMatMulAdd<float, false>({a}, {b}, {acc});\n"
            ));
        }
        MatrixOp::Store { acc, dest } => {
            out.push_str(&format!("{pad}{acc}.Store<{row_major}>({dest}, 0, {MATRIX_TILE});\n"));
        }
    }
}

fn join_args(args: &[Expr], builtins: &BuiltinMask, tensor_slots: &TensorSlots) -> String {
    args.iter()
        .map(|a| emit_expr(a, builtins, tensor_slots))
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
            type_decls: Vec::new(),
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
            type_decls: Vec::new(),
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
            type_decls: Vec::new(),
        };
        let slang = emit_canonical_compute_source(&kernel).source.canonical_slang;
        assert!(slang.contains("GroupThreadId _goldy_lid"));
        assert!(slang.contains("_goldy_red = _goldy_red + scratch[_goldy_lid.x + (1u << _goldy_s)]"));
        assert!(slang.contains("_goldy_red = scratch[0];\n"));
        assert!(slang.contains("ss = _goldy_red;\n"));
        assert!(slang.contains("exp(att[(0u) + _goldy_sm_t] - _goldy_sm_max)"));
        assert!(slang.contains("_goldy_red = max(_goldy_red, scratch[_goldy_lid.x + (1u << _goldy_s)])"));
    }

    #[test]
    fn workgroup_reduce_has_a_subgroup_form_behind_the_width_define() {
        let reduce = |op| {
            let mut out = String::new();
            emit_stmt(
                &mut out,
                &Stmt::WorkgroupReduce {
                    op,
                    n: 256,
                    val: Expr::Var("v".into()),
                    scratch: "scratch".into(),
                    dest: Expr::Var("v".into()),
                },
                0,
                &BuiltinMask {
                    local_id: true,
                    ..BuiltinMask::NONE
                },
                &HashMap::new(),
            );
            out
        };
        let sum = reduce(WorkgroupReduceOp::Sum);
        let hierarchical = sum
            .split("#if defined(GOLDY_SUBGROUP_WIDTH)\n")
            .nth(1)
            .and_then(|rest| rest.split("#endif\n").next())
            .expect("guarded subgroup form");
        assert!(hierarchical.starts_with(
            "    if (GOLDY_SUBGROUP_WIDTH <= 256 && 256 <= GOLDY_SUBGROUP_WIDTH * GOLDY_SUBGROUP_WIDTH) {\n"
        ));
        assert!(hierarchical.ends_with("    } else\n"));
        assert_eq!(hierarchical.matches("GroupMemoryBarrierWithGroupSync();").count(), 2);
        assert_eq!(
            hierarchical
                .matches("WaveMaskReadLaneAt(0xFFFFFFFFu, _goldy_red, int(_goldy_lane ^ _goldy_s))")
                .count(),
            2
        );
        assert!(!hierarchical.contains("WaveReadLaneAt"));
        assert!(hierarchical.contains("scratch[_goldy_lid.x / uint(GOLDY_SUBGROUP_WIDTH)] = _goldy_red;"));
        assert!(hierarchical.contains("_goldy_red = scratch[_goldy_lane & (_goldy_parts - 1u)];"));
        assert!(hierarchical.contains(
            "_goldy_red = (_goldy_upper ? _goldy_other : _goldy_red) + (_goldy_upper ? _goldy_red : _goldy_other);"
        ));
        let portable = sum.split("#endif\n").nth(1).expect("portable form");
        assert!(!portable.contains("Wave"));
        assert_eq!(portable.matches("GroupMemoryBarrierWithGroupSync();").count(), 4);
        assert!(portable.ends_with(
            "        _goldy_red = scratch[0];\n        GroupMemoryBarrierWithGroupSync();\n    }\n    v = _goldy_red;\n}\n"
        ));

        assert!(reduce(WorkgroupReduceOp::Max).contains(
            "_goldy_red = max((_goldy_upper ? _goldy_other : _goldy_red), (_goldy_upper ? _goldy_red : _goldy_other));"
        ));
    }

    #[test]
    fn emits_tensor_view_index_and_metadata_resource() {
        let kernel = ShaderKernel {
            name: "scale_view".into(),
            workgroup_size: [64, 1, 1],
            params: vec![
                KernelParam::tensor_read("x", ElementType::F32),
                KernelParam::tensor_write("y", ElementType::F32),
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
                        value: Expr::Index {
                            base: Box::new(Expr::Var("x".into())),
                            index: Box::new(Expr::Binary {
                                op: BinOp::Add,
                                left: Box::new(Expr::Var("i".into())),
                                right: Box::new(Expr::Dim {
                                    base: Box::new(Expr::Var("x".into())),
                                    axis: Box::new(Expr::LitU32(1)),
                                }),
                            }),
                        },
                    }],
                    else_body: None,
                },
            ],
            source_map: SourceMap {
                rust_file: "view.rs".into(),
                rust_line: 1,
            },
            type_decls: Vec::new(),
        };
        let def = emit_canonical_compute_source(&kernel);
        let slang = &def.source.canonical_slang;
        assert!(slang.contains("struct GoldyTensorLayout"));
        assert!(slang.contains("BufRO<GoldyTensorLayout> _goldy_tensor_meta"));
        assert!(slang
            .contains("#ifndef _GOLDY_TENSOR_OFF1\n#define _GOLDY_TENSOR_OFF1 _goldy_tensor_meta[1u].off\n#endif\n"));
        assert!(slang.contains(
            "#ifndef _GOLDY_SPEC_CS_MAIN_T0_D1\n#define _GOLDY_SPEC_CS_MAIN_T0_D1 _goldy_tensor_meta[0u].d1\n#endif\n"
        ));
        assert!(slang.contains(
            "    GoldyTensorLayout _goldy_t0 = goldy_tensor_layout(_GOLDY_TENSOR_OFF0, _GOLDY_SPEC_CS_MAIN_T0_RANK, \
             _GOLDY_SPEC_CS_MAIN_T0_NUMEL, _GOLDY_SPEC_CS_MAIN_T0_D0, _GOLDY_SPEC_CS_MAIN_T0_D1, \
             _GOLDY_SPEC_CS_MAIN_T0_D2, _GOLDY_SPEC_CS_MAIN_T0_D3, _GOLDY_SPEC_CS_MAIN_T0_S0, \
             _GOLDY_SPEC_CS_MAIN_T0_S1, _GOLDY_SPEC_CS_MAIN_T0_S2, _GOLDY_SPEC_CS_MAIN_T0_S3, \
             _GOLDY_SPEC_CS_MAIN_T0_FLAGS);\n    GoldyTensorLayout _goldy_t1 = "
        ));
        assert!(slang.contains("goldy_tensor_offset(_goldy_t0, "));
        assert!(slang.contains("y[goldy_tensor_offset(_goldy_t1, i)]"));
        assert!(slang.contains("(i < _goldy_t1.numel)"));
        assert!(slang.contains("goldy_tensor_dim(_goldy_t0, 1u)"));
        let body = slang.split("void cs_main(").nth(1).expect("entry");
        assert!(
            !body.contains("_goldy_tensor_meta["),
            "the body reads the parcel only through the macros"
        );
        assert_eq!(def.params.len(), 3);
        assert!(def.params[0].is_tensor);
        assert!(def.params[1].is_tensor);
        assert!(!def.params[2].is_tensor);
        assert_eq!(def.params[2].name, "_goldy_tensor_meta");
        assert_eq!(def.abi_version, crate::KERNEL_ABI_VERSION);

        let mut contracted = kernel;
        let spec = |rank| crate::TensorShapeSpec {
            dims: vec![crate::TensorDimSpec::Any; rank],
        };
        contracted.params[0].shape_spec = Some(spec(2));
        contracted.params[1].shape_spec = Some(spec(1));
        let slang = emit_canonical_compute_source(&contracted).source.canonical_slang;
        assert!(slang.contains("x[goldy_tensor_offset2(_goldy_t0, "));
        assert!(slang.contains("y[goldy_tensor_offset1(_goldy_t1, i)]"));
        assert!(!slang.contains("goldy_tensor_offset(_goldy_t"));
    }

    /// `fn name(input: &[f32], output: Scattered<f32>, count: u32) { let i = gid.x; if i < count { output[i] = input[i] <op> k; } }`
    fn pointwise(name: &str, op: BinOp, k: f32, scratch: Option<&str>) -> ShaderKernel {
        let var = |n: &str| Expr::Var(n.into());
        let mut body = Vec::new();
        if let Some(s) = scratch {
            body.push(Stmt::WorkgroupArray {
                name: s.into(),
                elem: "float".into(),
                len: 64,
            });
        }
        body.push(Stmt::Let {
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
        });
        body.push(Stmt::If {
            cond: Expr::Binary {
                op: BinOp::Lt,
                left: Box::new(var("i")),
                right: Box::new(var("count")),
            },
            then_body: vec![Stmt::Assign {
                target: Expr::Index {
                    base: Box::new(var("output")),
                    index: Box::new(var("i")),
                },
                value: Expr::Binary {
                    op,
                    left: Box::new(Expr::Index {
                        base: Box::new(var("input")),
                        index: Box::new(var("i")),
                    }),
                    right: Box::new(Expr::LitF32(k)),
                },
            }],
            else_body: None,
        });
        ShaderKernel {
            name: name.into(),
            workgroup_size: [64, 1, 1],
            params: vec![
                KernelParam::buffer_read("input", ElementType::F32),
                KernelParam::buffer_write("output", ElementType::F32),
                KernelParam::scalar_param("count", ScalarType::U32),
            ],
            builtins: BuiltinMask {
                global_id: true,
                ..BuiltinMask::NONE
            },
            body,
            source_map: SourceMap::default(),
            type_decls: Vec::new(),
        }
    }

    #[test]
    fn standalone_source_is_byte_stable() {
        let mut kernel = pointwise("scale", BinOp::Mul, 2.0, Some("scratch"));
        kernel.type_decls = vec!["struct A { uint a; };".into(), "struct B { float b; };".into()];
        let def = emit_canonical_compute_source(&kernel);
        assert_eq!(
            def.source.canonical_slang,
            "struct A { uint a; };\n\
             struct B { float b; };\n\
             import goldy_exp;\n\
             \n\
             groupshared float scratch[64];\n\
             \n\
             [goldy_compute]\n\
             [numthreads(64, 1, 1)]\n\
             void cs_main(BufRO<float> input, Scattered<float> output, uint count, ThreadId _goldy_gid) {\n    \
                 uint i = _goldy_gid.x;\n    \
                 if ((i < count)) {\n        \
                     output[i] = (input[i] * 2.0);\n    \
                 }\n\
             }\n"
        );
        assert_eq!(def.entry, VIRTUAL_ENTRY_NAME);
        assert_eq!(def.definition.as_ref(), Some(&kernel));
    }

    #[test]
    fn assembled_entry_carries_no_definition() {
        let kernel = pointwise("scale", BinOp::Mul, 2.0, None);
        let body = lower_body(&kernel.body, 1, &BodyEnv::standalone(&kernel));
        let def = assemble_virtual_entry(&VirtualEntrySignature::standalone(&kernel), &body);
        assert!(def.definition.is_none());
        assert_eq!(
            def.source.canonical_slang,
            emit_canonical_compute_source(&kernel).source.canonical_slang
        );
    }

    #[test]
    fn namespaced_bodies_lower_into_one_entry() {
        let produce = pointwise("produce", BinOp::Mul, 2.0, Some("scratch")).namespaced("_goldy_k0");
        let consume = pointwise("consume", BinOp::Add, 1.0, Some("scratch"))
            .namespaced("_goldy_k1")
            .rename_symbols(|name, kind| match (name, kind) {
                ("input", crate::SymbolKind::Param) => "output".into(),
                ("output", crate::SymbolKind::Param) => "result".into(),
                _ => name.into(),
            });
        let mut params = produce.params.clone();
        params.push(consume.params[1].clone());
        let sig = VirtualEntrySignature {
            workgroup_size: [64, 1, 1],
            params,
            builtins: produce.builtins,
            type_decls: Vec::new(),
            source_map: SourceMap::default(),
        };
        let env = BodyEnv {
            builtins: sig.builtins,
            tensor_slots: tensor_slot_map(&sig.params),
        };
        let mut body = LoweredBody::default();
        for k in [&produce, &consume] {
            let lowered = lower_body(&k.body, 2, &env);
            body.workgroup_decls.push_str(&lowered.workgroup_decls);
            body.stmts.push_str(&format!("    {{\n{}    }}\n", lowered.stmts));
        }
        let slang = assemble_virtual_entry(&sig, &body).source.canonical_slang;

        assert_eq!(slang.matches("[goldy_compute]").count(), 1);
        assert!(slang.contains(
            "groupshared float _goldy_k0_scratch[64];\ngroupshared float _goldy_k1_scratch[64];\n\n[goldy_compute]"
        ));
        assert!(slang.contains(
            "(BufRO<float> input, Scattered<float> output, uint count, Scattered<float> result, ThreadId _goldy_gid)"
        ));
        assert!(slang.contains("        uint _goldy_k0_i = _goldy_gid.x;\n"));
        assert!(slang.contains("            output[_goldy_k0_i] = (input[_goldy_k0_i] * 2.0);\n"));
        assert!(slang.contains("            result[_goldy_k1_i] = (output[_goldy_k1_i] + 1.0);\n"));
    }
}
