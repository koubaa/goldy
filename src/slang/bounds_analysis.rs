//! Static bounds analysis over Slang-generated SPIR-V (prototype).
//!
//! Slang already rejects *constant* out-of-bounds indices, but a dynamic index such as
//! `links[link]` is only checked at runtime — and on most GPUs "checked" means an undefined
//! read, a hang, or a device loss. This module proves, conservatively, that every dynamic
//! index into a statically sized array (`groupshared`, `Private`, `Function` locals, struct
//! members, vectors) stays inside `0 <= index < length`, and reports every access it could
//! *not* prove with the Slang source location Slang embedded in the SPIR-V.
//!
//! The analysis is opt-in (`GOLDY_VALIDATION=bounds`) and never fails a compile: findings are
//! warnings. See `docs/src/design/shader-bounds-analysis.md` for the integration decision
//! (SPIR-V vs Slang IR), the analysis model, and known false positives.
//!
//! # Analysis model
//!
//! 1. **SSA reconstruction.** Slang emits locals as `OpVariable` + `OpLoad`/`OpStore`
//!    (not `OpPhi`). Non-escaping scalar integer locals are promoted to SSA (phis at the
//!    iterated dominance frontier) so that a guard on one load refines every other load of
//!    the same variable in the guarded region.
//! 2. **Interval propagation.** Every integer SSA value gets a flow-insensitive interval via a
//!    fixpoint with widening. Built-ins such as `SV_GroupThreadID` use `LocalSize`, wave
//!    built-ins use their Vulkan minimum/maximum, everything else starts at the type range.
//!    Arithmetic that can wrap collapses to the type range (sound, not precise).
//! 3. **Path-sensitive refinement.** For each dynamic index the analysis walks the dominator
//!    tree collecting conditions from `OpBranchConditional` edges that dominate the access
//!    (`index >= 0`, `index < N`, `a >= b`, `LogicalAnd` conjuncts, ...). The index expression
//!    is re-evaluated under those facts, including the relational rule
//!    `a >= b  ==>  a - b in [0, hi(a) - lo(b)]` that workgroup scans rely on. `OpSelect` and
//!    `OpPhi` operands are evaluated under their own edge conditions, so
//!    `if (x >= 0) idx = x; else idx = 0;` is understood.
//! 4. **Check.** Each `OpAccessChain` index into a known-length composite must satisfy
//!    `0 <= index <= length - 1`. Anything else is a [`BoundsDiagnostic`].

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

// ============================================================================
// Public API
// ============================================================================

/// Source location recovered from `NonSemantic.Shader.DebugInfo.100` `DebugLine` or `OpLine`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// One dynamic array access the analysis could not prove in bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundsDiagnostic {
    /// Name of the function containing the access (`OpName`, or `fn%<id>`).
    pub function: String,
    /// Name of the indexed array (variable name plus struct member path when available).
    pub array: String,
    /// Statically known number of elements.
    pub array_length: u64,
    /// Interval the index was proven to lie in (in the index type's signedness), or `None`
    /// when nothing narrower than the full type range is known.
    pub index_range: Option<(i128, i128)>,
    /// Slang source location, when the module carries debug info.
    pub location: Option<SourceLocation>,
    /// What the index ultimately depends on that the analysis cannot bound (system values,
    /// buffer loads, float conversions, uninlined calls, ...). Empty when the range is known
    /// but simply too wide. Deduplicated, at most a few entries.
    pub depends_on: Vec<String>,
}

impl fmt::Display for BoundsDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "possible out-of-bounds index into `{}[{}]`",
            self.array, self.array_length
        )?;
        match self.index_range {
            Some((lo, hi)) => {
                write!(f, ": index range [{lo}, {hi}]")?;
                if lo < 0 {
                    write!(f, " (may be negative)")?;
                }
                if hi >= self.array_length as i128 {
                    write!(f, " (may exceed {})", self.array_length as i128 - 1)?;
                }
            }
            None => write!(f, ": index range unknown")?,
        }
        if !self.depends_on.is_empty() {
            write!(f, " (depends on {})", self.depends_on.join(", "))?;
        }
        if let Some(loc) = &self.location {
            write!(f, " at {loc}")?;
        }
        write!(f, " in `{}`", self.function)
    }
}

/// Result of [`analyze_spirv`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundsReport {
    /// Accesses that could not be proven safe.
    pub diagnostics: Vec<BoundsDiagnostic>,
    /// Number of dynamic (non-constant) indices into known-length composites that were checked.
    pub checked_accesses: usize,
    /// Number of those proven to be in bounds.
    pub proven_safe: usize,
}

impl BoundsReport {
    /// `true` when every checked access was proven in bounds.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

/// Analysis failure (malformed SPIR-V). Never raised for shaders the analysis merely
/// cannot reason about — those produce diagnostics or are skipped.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BoundsAnalysisError {
    #[error("SPIR-V blob is not a multiple of 4 bytes")]
    Misaligned,
    #[error("SPIR-V magic number mismatch (got {0:#x})")]
    BadMagic(u32),
    #[error("SPIR-V instruction stream truncated at word {0}")]
    Truncated(usize),
}

/// Analyze a SPIR-V module given as bytes (as returned by [`super::SlangCompiler`]).
pub fn analyze_spirv_bytes(bytes: &[u8]) -> Result<BoundsReport, BoundsAnalysisError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(BoundsAnalysisError::Misaligned);
    }
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    analyze_spirv(&words)
}

/// Analyze a SPIR-V module given as words.
pub fn analyze_spirv(words: &[u32]) -> Result<BoundsReport, BoundsAnalysisError> {
    let module = Module::parse(words)?;
    let mut report = BoundsReport::default();
    for func in &module.functions {
        FunctionAnalysis::new(&module, func).run(&mut report);
    }
    Ok(report)
}

// ============================================================================
// SPIR-V opcodes (only those the analysis interprets)
// ============================================================================

mod op {
    pub const UNDEF: u16 = 1;
    pub const NAME: u16 = 5;
    pub const MEMBER_NAME: u16 = 6;
    pub const STRING: u16 = 7;
    pub const LINE: u16 = 8;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const EXECUTION_MODE: u16 = 16;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_MATRIX: u16 = 24;
    pub const TYPE_IMAGE: u16 = 25;
    pub const TYPE_SAMPLER: u16 = 26;
    pub const TYPE_SAMPLED_IMAGE: u16 = 27;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_OPAQUE: u16 = 31;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const CONSTANT_COMPOSITE: u16 = 44;
    pub const CONSTANT_NULL: u16 = 46;
    pub const SPEC_CONSTANT_TRUE: u16 = 48;
    pub const SPEC_CONSTANT_FALSE: u16 = 49;
    pub const SPEC_CONSTANT: u16 = 50;
    pub const SPEC_CONSTANT_COMPOSITE: u16 = 51;
    pub const SPEC_CONSTANT_OP: u16 = 52;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_PARAMETER: u16 = 55;
    pub const FUNCTION_END: u16 = 56;
    pub const FUNCTION_CALL: u16 = 57;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const IN_BOUNDS_ACCESS_CHAIN: u16 = 66;
    pub const PTR_ACCESS_CHAIN: u16 = 67;
    pub const ARRAY_LENGTH: u16 = 68;
    pub const IN_BOUNDS_PTR_ACCESS_CHAIN: u16 = 70;
    pub const DECORATE: u16 = 71;
    pub const VECTOR_SHUFFLE: u16 = 79;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const COMPOSITE_INSERT: u16 = 82;
    pub const COPY_OBJECT: u16 = 83;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const U_CONVERT: u16 = 113;
    pub const S_CONVERT: u16 = 114;
    pub const BITCAST: u16 = 124;
    pub const S_NEGATE: u16 = 126;
    pub const I_ADD: u16 = 128;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const U_DIV: u16 = 134;
    pub const S_DIV: u16 = 135;
    pub const F_DIV: u16 = 136;
    pub const U_MOD: u16 = 137;
    pub const S_REM: u16 = 138;
    pub const S_MOD: u16 = 139;
    pub const LOGICAL_EQUAL: u16 = 164;
    pub const LOGICAL_NOT_EQUAL: u16 = 165;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    /// `OpGroupNonUniformElect` .. `OpGroupNonUniformQuadSwap`: the subgroup (wave) ops.
    pub const GROUP_NON_UNIFORM_FIRST: u16 = 333;
    pub const GROUP_NON_UNIFORM_LAST: u16 = 366;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const PHI: u16 = 245;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const SWITCH: u16 = 251;
    pub const NO_LINE: u16 = 317;

    // GLSL.std.450
    pub const GLSL_S_ABS: u32 = 5;
    pub const GLSL_U_MIN: u32 = 38;
    pub const GLSL_S_MIN: u32 = 39;
    pub const GLSL_U_MAX: u32 = 41;
    pub const GLSL_S_MAX: u32 = 42;
    pub const GLSL_U_CLAMP: u32 = 44;
    pub const GLSL_S_CLAMP: u32 = 45;

    // NonSemantic.Shader.DebugInfo.100
    pub const DEBUG_LOCAL_VARIABLE: u32 = 26;
    pub const DEBUG_DECLARE: u32 = 28;
    pub const DEBUG_VALUE: u32 = 29;
    pub const DEBUG_SOURCE: u32 = 35;
    pub const DEBUG_LINE: u32 = 103;
    pub const DEBUG_NO_LINE: u32 = 104;

    // Storage classes
    pub const SC_UNIFORM_CONSTANT: u32 = 0;
    pub const SC_INPUT: u32 = 1;
    pub const SC_UNIFORM: u32 = 2;
    pub const SC_WORKGROUP: u32 = 4;
    pub const SC_PRIVATE: u32 = 6;
    pub const SC_FUNCTION: u32 = 7;
    pub const SC_PUSH_CONSTANT: u32 = 9;
    pub const SC_STORAGE_BUFFER: u32 = 12;
    pub const SC_PHYSICAL_STORAGE_BUFFER: u32 = 5349;

    // Decorations / built-ins / execution modes
    pub const DECORATION_BUILTIN: u32 = 11;
    pub const BUILTIN_PRIMITIVE_ID: u32 = 7;
    pub const BUILTIN_NUM_WORKGROUPS: u32 = 24;
    pub const BUILTIN_WORKGROUP_SIZE: u32 = 25;
    pub const BUILTIN_WORKGROUP_ID: u32 = 26;
    pub const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
    pub const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
    pub const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;
    pub const BUILTIN_SUBGROUP_SIZE: u32 = 36;
    pub const BUILTIN_NUM_SUBGROUPS: u32 = 38;
    pub const BUILTIN_SUBGROUP_ID: u32 = 40;
    pub const BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
    pub const BUILTIN_VERTEX_INDEX: u32 = 42;
    pub const BUILTIN_INSTANCE_INDEX: u32 = 43;
    pub const BUILTIN_BASE_VERTEX: u32 = 4424;
    pub const BUILTIN_BASE_INSTANCE: u32 = 4425;
    pub const BUILTIN_DRAW_INDEX: u32 = 4426;
    pub const EXEC_MODE_LOCAL_SIZE: u32 = 17;
    pub const EXEC_MODE_LOCAL_SIZE_ID: u32 = 38;
}

/// Vulkan guarantees `subgroupSize` in `[1, 128]`.
const MAX_SUBGROUP_SIZE: i128 = 128;

/// Recursion budget for on-demand refined evaluation of an index expression.
const REFINE_DEPTH_LIMIT: u32 = 24;

/// Number of times a phi may grow before its moving bound is widened to the type bound.
const WIDEN_AFTER: u32 = 3;
/// Cap on ascending fixpoint passes; from [`ASCEND_FORCE_TOP_AT`] every phi is forced to top so
/// the loop is guaranteed to converge.
const ASCEND_PASSES: u32 = 64;
const ASCEND_FORCE_TOP_AT: u32 = 48;
/// Cap on narrowing passes after the ascending phase converges.
const NARROW_PASSES: u32 = 8;

// ============================================================================
// Parsing
// ============================================================================

#[derive(Debug, Clone)]
struct Inst {
    opcode: u16,
    result_type: Option<u32>,
    result_id: Option<u32>,
    /// Remaining words after `result_type` / `result_id`.
    operands: Vec<u32>,
    /// All words of the instruction after the opcode word (for conservative escape scans).
    all_words: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    TypeAndId,
    IdOnly,
    NoResult,
}

fn layout(opcode: u16) -> Layout {
    use op::*;
    match opcode {
        UNDEF
        | EXT_INST
        | CONSTANT_TRUE
        | CONSTANT_FALSE
        | CONSTANT
        | CONSTANT_COMPOSITE
        | CONSTANT_NULL
        | SPEC_CONSTANT_TRUE
        | SPEC_CONSTANT_FALSE
        | SPEC_CONSTANT
        | SPEC_CONSTANT_COMPOSITE
        | SPEC_CONSTANT_OP
        | FUNCTION
        | FUNCTION_PARAMETER
        | FUNCTION_CALL
        | VARIABLE
        | LOAD
        | ACCESS_CHAIN
        | IN_BOUNDS_ACCESS_CHAIN
        | PTR_ACCESS_CHAIN
        | ARRAY_LENGTH
        | IN_BOUNDS_PTR_ACCESS_CHAIN
        | VECTOR_SHUFFLE
        | COMPOSITE_CONSTRUCT
        | COMPOSITE_EXTRACT
        | COMPOSITE_INSERT
        | COPY_OBJECT
        | CONVERT_F_TO_U
        | CONVERT_F_TO_S
        | CONVERT_S_TO_F
        | CONVERT_U_TO_F
        | U_CONVERT
        | S_CONVERT
        | BITCAST
        | S_NEGATE
        | I_ADD
        | F_ADD
        | I_SUB
        | F_SUB
        | I_MUL
        | F_MUL
        | U_DIV
        | S_DIV
        | F_DIV
        | U_MOD
        | S_REM
        | S_MOD
        | LOGICAL_EQUAL
        | LOGICAL_NOT_EQUAL
        | LOGICAL_OR
        | LOGICAL_AND
        | LOGICAL_NOT
        | SELECT
        | I_EQUAL
        | I_NOT_EQUAL
        | U_GREATER_THAN
        | S_GREATER_THAN
        | U_GREATER_THAN_EQUAL
        | S_GREATER_THAN_EQUAL
        | U_LESS_THAN
        | S_LESS_THAN
        | U_LESS_THAN_EQUAL
        | S_LESS_THAN_EQUAL
        | SHIFT_RIGHT_LOGICAL
        | SHIFT_RIGHT_ARITHMETIC
        | SHIFT_LEFT_LOGICAL
        | BITWISE_OR
        | BITWISE_XOR
        | BITWISE_AND
        | NOT
        | PHI => Layout::TypeAndId,
        STRING | EXT_INST_IMPORT | TYPE_VOID | TYPE_BOOL | TYPE_INT | TYPE_FLOAT | TYPE_VECTOR | TYPE_MATRIX
        | TYPE_IMAGE | TYPE_SAMPLER | TYPE_SAMPLED_IMAGE | TYPE_ARRAY | TYPE_RUNTIME_ARRAY | TYPE_STRUCT
        | TYPE_OPAQUE | TYPE_POINTER | TYPE_FUNCTION | LABEL => Layout::IdOnly,
        _ => Layout::NoResult,
    }
}

fn decode_string(words: &[u32]) -> (String, usize) {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    let mut used = 0;
    'outer: for w in words {
        used += 1;
        for b in w.to_le_bytes() {
            if b == 0 {
                break 'outer;
            }
            bytes.push(b);
        }
    }
    (String::from_utf8_lossy(&bytes).into_owned(), used)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct IntTy {
    bits: u32,
    signed: bool,
}

impl IntTy {
    fn range(self) -> Interval {
        if self.signed {
            Interval::new(-(1i128 << (self.bits - 1)), (1i128 << (self.bits - 1)) - 1)
        } else {
            Interval::new(0, (1i128 << self.bits) - 1)
        }
    }
    fn with_signed(self, signed: bool) -> IntTy {
        IntTy {
            bits: self.bits,
            signed,
        }
    }
}

#[derive(Debug, Clone)]
enum Ty {
    Bool,
    Int(IntTy),
    Float,
    Vector { elem: u32, count: u32 },
    Matrix { column: u32, count: u32 },
    Array { elem: u32, length: Option<u64> },
    RuntimeArray { elem: u32 },
    Struct { members: Vec<u32> },
    Pointer { storage_class: u32, pointee: u32 },
    Other,
}

/// Integer shape of a type: scalar or vector of integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IntShape {
    ty: IntTy,
    lanes: usize,
}

#[derive(Debug, Default)]
struct Module {
    types: HashMap<u32, Ty>,
    int_consts: HashMap<u32, i128>,
    bool_consts: HashMap<u32, bool>,
    composite_consts: HashMap<u32, Vec<u32>>,
    /// Result type of every module-level constant.
    const_types: HashMap<u32, u32>,
    names: HashMap<u32, String>,
    member_names: HashMap<(u32, u32), String>,
    strings: HashMap<u32, String>,
    builtins: HashMap<u32, u32>,
    global_vars: HashMap<u32, (u32, u32)>, // id -> (storage class, pointer type)
    debug_ext: Option<u32>,
    glsl_ext: Option<u32>,
    debug_sources: HashMap<u32, u32>,
    /// `DebugLocalVariable` id -> name string id.
    debug_local_vars: HashMap<u32, u32>,
    /// Join of `LocalSize` over all entry points, if any declares one.
    local_size: Option<[i128; 3]>,
    functions: Vec<Function>,
}

#[derive(Debug, Default)]
struct Block {
    label: u32,
    /// Inclusive start / exclusive end into `Function::insts`.
    start: usize,
    end: usize,
    preds: Vec<usize>,
    succs: Vec<usize>,
}

#[derive(Debug, Default)]
struct Function {
    id: u32,
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    label_to_block: HashMap<u32, usize>,
}

impl Module {
    fn parse(words: &[u32]) -> Result<Module, BoundsAnalysisError> {
        if words.len() < 5 {
            return Err(BoundsAnalysisError::Truncated(words.len()));
        }
        if words[0] != 0x0723_0203 {
            return Err(BoundsAnalysisError::BadMagic(words[0]));
        }
        let mut m = Module::default();
        let mut local_size_ids: Vec<[u32; 3]> = Vec::new();
        let mut current: Option<Function> = None;
        let mut current_block: Option<Block> = None;

        let mut i = 5;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            let opcode = (words[i] & 0xffff) as u16;
            if count == 0 || i + count > words.len() {
                return Err(BoundsAnalysisError::Truncated(i));
            }
            let body = &words[i + 1..i + count];
            let (result_type, result_id, operands) = match layout(opcode) {
                Layout::TypeAndId if body.len() >= 2 => (Some(body[0]), Some(body[1]), body[2..].to_vec()),
                Layout::IdOnly if !body.is_empty() => (None, Some(body[0]), body[1..].to_vec()),
                _ => (None, None, body.to_vec()),
            };
            let inst = Inst {
                opcode,
                result_type,
                result_id,
                operands,
                all_words: body.to_vec(),
            };
            i += count;
            if current.is_none()
                && matches!(
                    opcode,
                    op::CONSTANT
                        | op::CONSTANT_TRUE
                        | op::CONSTANT_FALSE
                        | op::CONSTANT_COMPOSITE
                        | op::CONSTANT_NULL
                        | op::SPEC_CONSTANT
                        | op::SPEC_CONSTANT_TRUE
                        | op::SPEC_CONSTANT_FALSE
                        | op::SPEC_CONSTANT_COMPOSITE
                        | op::SPEC_CONSTANT_OP
                        | op::UNDEF
                )
            {
                if let (Some(ty), Some(id)) = (result_type, result_id) {
                    m.const_types.insert(id, ty);
                }
            }

            if let Some(func) = current.as_mut() {
                // `DebugDeclare %local_var %variable %expr` names locals that have no OpName
                // (Slang only emits OpName for some of them).
                if opcode == op::EXT_INST
                    && inst.operands.len() >= 4
                    && Some(inst.operands[0]) == m.debug_ext
                    && inst.operands[1] == op::DEBUG_DECLARE
                {
                    if let Some(name) = m
                        .debug_local_vars
                        .get(&inst.operands[2])
                        .and_then(|s| m.strings.get(s))
                        .cloned()
                    {
                        m.names.entry(inst.operands[3]).or_insert(name);
                    }
                }
                match opcode {
                    op::LABEL => {
                        if let Some(b) = current_block.take() {
                            func.blocks.push(b);
                        }
                        current_block = Some(Block {
                            label: result_id.unwrap_or(0),
                            start: func.insts.len(),
                            end: func.insts.len(),
                            ..Default::default()
                        });
                        func.insts.push(inst);
                    }
                    op::FUNCTION_END => {
                        if let Some(b) = current_block.take() {
                            func.blocks.push(b);
                        }
                        let mut f = current.take().unwrap();
                        for (idx, b) in f.blocks.iter().enumerate() {
                            f.label_to_block.insert(b.label, idx);
                        }
                        m.name_locals_from_debug_values(&f);
                        m.functions.push(f);
                    }
                    _ => {
                        func.insts.push(inst);
                        if let Some(b) = current_block.as_mut() {
                            b.end = func.insts.len();
                        }
                    }
                }
                continue;
            }

            match opcode {
                op::FUNCTION => {
                    current = Some(Function {
                        id: result_id.unwrap_or(0),
                        ..Default::default()
                    });
                }
                op::NAME => {
                    if let Some((&target, rest)) = inst.operands.split_first() {
                        m.names.insert(target, decode_string(rest).0);
                    }
                }
                op::MEMBER_NAME => {
                    if inst.operands.len() >= 2 {
                        m.member_names.insert(
                            (inst.operands[0], inst.operands[1]),
                            decode_string(&inst.operands[2..]).0,
                        );
                    }
                }
                op::STRING => {
                    if let Some(id) = result_id {
                        m.strings.insert(id, decode_string(&inst.operands).0);
                    }
                }
                op::EXT_INST_IMPORT => {
                    let name = decode_string(&inst.operands).0;
                    match name.as_str() {
                        "NonSemantic.Shader.DebugInfo.100" => m.debug_ext = result_id,
                        "GLSL.std.450" => m.glsl_ext = result_id,
                        _ => {}
                    }
                }
                op::EXT_INST => {
                    // Module-level debug info: DebugSource ties a file string to a source id;
                    // DebugLocalVariable carries the source name of a local.
                    if inst.operands.len() >= 3 && Some(inst.operands[0]) == m.debug_ext {
                        if let Some(id) = result_id {
                            match inst.operands[1] {
                                op::DEBUG_SOURCE => {
                                    m.debug_sources.insert(id, inst.operands[2]);
                                }
                                op::DEBUG_LOCAL_VARIABLE => {
                                    m.debug_local_vars.insert(id, inst.operands[2]);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                op::EXECUTION_MODE => {
                    if inst.operands.len() >= 5 && inst.operands[1] == op::EXEC_MODE_LOCAL_SIZE {
                        let ls = [
                            inst.operands[2] as i128,
                            inst.operands[3] as i128,
                            inst.operands[4] as i128,
                        ];
                        m.local_size = Some(match m.local_size {
                            Some(prev) => [prev[0].max(ls[0]), prev[1].max(ls[1]), prev[2].max(ls[2])],
                            None => ls,
                        });
                    } else if inst.operands.len() >= 5 && inst.operands[1] == op::EXEC_MODE_LOCAL_SIZE_ID {
                        local_size_ids.push([inst.operands[2], inst.operands[3], inst.operands[4]]);
                    }
                }
                op::DECORATE => {
                    if inst.operands.len() >= 3 && inst.operands[1] == op::DECORATION_BUILTIN {
                        m.builtins.insert(inst.operands[0], inst.operands[2]);
                    }
                }
                op::TYPE_BOOL => {
                    m.types.insert(result_id.unwrap(), Ty::Bool);
                }
                op::TYPE_INT => {
                    if inst.operands.len() >= 2 {
                        let bits = inst.operands[0].clamp(1, 64);
                        m.types.insert(
                            result_id.unwrap(),
                            Ty::Int(IntTy {
                                bits,
                                signed: inst.operands[1] != 0,
                            }),
                        );
                    }
                }
                op::TYPE_FLOAT => {
                    m.types.insert(result_id.unwrap(), Ty::Float);
                }
                op::TYPE_VECTOR => {
                    if inst.operands.len() >= 2 {
                        m.types.insert(
                            result_id.unwrap(),
                            Ty::Vector {
                                elem: inst.operands[0],
                                count: inst.operands[1],
                            },
                        );
                    }
                }
                op::TYPE_MATRIX => {
                    if inst.operands.len() >= 2 {
                        m.types.insert(
                            result_id.unwrap(),
                            Ty::Matrix {
                                column: inst.operands[0],
                                count: inst.operands[1],
                            },
                        );
                    }
                }
                op::TYPE_ARRAY => {
                    if inst.operands.len() >= 2 {
                        let length = m.int_consts.get(&inst.operands[1]).and_then(|&v| u64::try_from(v).ok());
                        m.types.insert(
                            result_id.unwrap(),
                            Ty::Array {
                                elem: inst.operands[0],
                                length,
                            },
                        );
                    }
                }
                op::TYPE_RUNTIME_ARRAY => {
                    if !inst.operands.is_empty() {
                        m.types
                            .insert(result_id.unwrap(), Ty::RuntimeArray { elem: inst.operands[0] });
                    }
                }
                op::TYPE_STRUCT => {
                    m.types.insert(
                        result_id.unwrap(),
                        Ty::Struct {
                            members: inst.operands.clone(),
                        },
                    );
                }
                op::TYPE_POINTER => {
                    if inst.operands.len() >= 2 {
                        m.types.insert(
                            result_id.unwrap(),
                            Ty::Pointer {
                                storage_class: inst.operands[0],
                                pointee: inst.operands[1],
                            },
                        );
                    }
                }
                op::TYPE_VOID
                | op::TYPE_IMAGE
                | op::TYPE_SAMPLER
                | op::TYPE_SAMPLED_IMAGE
                | op::TYPE_OPAQUE
                | op::TYPE_FUNCTION => {
                    m.types.insert(result_id.unwrap(), Ty::Other);
                }
                op::CONSTANT => {
                    if let (Some(ty), Some(id)) = (result_type, result_id) {
                        if let Some(Ty::Int(it)) = m.types.get(&ty) {
                            let raw: u64 = match inst.operands.len() {
                                1 => u64::from(inst.operands[0]),
                                _ if inst.operands.len() >= 2 => {
                                    u64::from(inst.operands[0]) | (u64::from(inst.operands[1]) << 32)
                                }
                                _ => 0,
                            };
                            let value = if it.signed {
                                // Sign-extend from `bits`.
                                let shift = 64 - it.bits;
                                (((raw << shift) as i64) >> shift) as i128
                            } else {
                                (raw & (u64::MAX >> (64 - it.bits))) as i128
                            };
                            m.int_consts.insert(id, value);
                        }
                    }
                }
                op::CONSTANT_TRUE | op::SPEC_CONSTANT_TRUE => {
                    m.bool_consts.insert(result_id.unwrap(), true);
                }
                op::CONSTANT_FALSE | op::SPEC_CONSTANT_FALSE => {
                    m.bool_consts.insert(result_id.unwrap(), false);
                }
                op::CONSTANT_NULL => {
                    if let (Some(ty), Some(id)) = (result_type, result_id) {
                        if matches!(m.types.get(&ty), Some(Ty::Int(_))) {
                            m.int_consts.insert(id, 0);
                        }
                    }
                }
                op::CONSTANT_COMPOSITE => {
                    if let Some(id) = result_id {
                        m.composite_consts.insert(id, inst.operands.clone());
                    }
                }
                op::VARIABLE => {
                    if let (Some(ty), Some(id)) = (result_type, result_id) {
                        if let Some(&sc) = inst.operands.first() {
                            m.global_vars.insert(id, (sc, ty));
                        }
                    }
                }
                _ => {}
            }
        }

        for ids in local_size_ids {
            if let (Some(&x), Some(&y), Some(&z)) = (
                m.int_consts.get(&ids[0]),
                m.int_consts.get(&ids[1]),
                m.int_consts.get(&ids[2]),
            ) {
                m.local_size = Some(match m.local_size {
                    Some(prev) => [prev[0].max(x), prev[1].max(y), prev[2].max(z)],
                    None => [x, y, z],
                });
            }
        }

        for f in &mut m.functions {
            build_cfg(f);
        }
        Ok(m)
    }

    fn int_shape(&self, ty: u32) -> Option<IntShape> {
        match self.types.get(&ty)? {
            Ty::Int(it) => Some(IntShape { ty: *it, lanes: 1 }),
            Ty::Vector { elem, count } => match self.types.get(elem)? {
                Ty::Int(it) => Some(IntShape {
                    ty: *it,
                    lanes: *count as usize,
                }),
                _ => None,
            },
            _ => None,
        }
    }

    /// HLSL-style spelling of a type for diagnostics (`float2`, `uint[4]`, `float4x4`, ...).
    fn type_name(&self, ty: u32) -> String {
        match self.types.get(&ty) {
            Some(Ty::Bool) => "bool".into(),
            Some(Ty::Int(t)) => match (t.signed, t.bits) {
                (true, 32) => "int".into(),
                (false, 32) => "uint".into(),
                (true, b) => format!("int{b}_t"),
                (false, b) => format!("uint{b}_t"),
            },
            Some(Ty::Float) => "float".into(),
            Some(Ty::Vector { elem, count }) => format!("{}{count}", self.type_name(*elem)),
            Some(Ty::Matrix { column, count }) => match self.types.get(column) {
                Some(Ty::Vector { elem, count: rows }) => format!("{}{count}x{rows}", self.type_name(*elem)),
                _ => format!("{}[{count}]", self.type_name(*column)),
            },
            Some(Ty::Array { elem, length: Some(n) }) => format!("{}[{n}]", self.type_name(*elem)),
            Some(Ty::Array { elem, length: None }) | Some(Ty::RuntimeArray { elem }) => {
                format!("{}[]", self.type_name(*elem))
            }
            Some(Ty::Struct { .. }) => self.names.get(&ty).cloned().unwrap_or_else(|| "struct".into()),
            Some(Ty::Pointer { pointee, .. }) => format!("{}*", self.type_name(*pointee)),
            Some(Ty::Other) | None => "?".into(),
        }
    }

    fn pointee(&self, ptr_ty: u32) -> Option<(u32, u32)> {
        match self.types.get(&ptr_ty)? {
            Ty::Pointer { storage_class, pointee } => Some((*storage_class, *pointee)),
            _ => None,
        }
    }

    /// Name still-anonymous function-scope `OpVariable`s from debug info.
    ///
    /// For an aggregate local such as `uint table[4] = {...}` Slang emits
    /// `DebugValue %table %init` followed by `OpStore %var %init`, with neither `OpName`
    /// nor `DebugDeclare` on `%var`. Every value that `DebugValue` attributes to exactly
    /// one source variable and that is stored into exactly one unnamed local names that local.
    fn name_locals_from_debug_values(&mut self, f: &Function) {
        let Some(ext) = self.debug_ext else { return };
        // value id -> name string id; `None` marks a value attributed to several variables.
        let mut value_names: HashMap<u32, Option<u32>> = HashMap::new();
        for inst in &f.insts {
            if inst.opcode == op::EXT_INST
                && inst.operands.len() >= 4
                && inst.operands[0] == ext
                && inst.operands[1] == op::DEBUG_VALUE
            {
                if let Some(&name) = self.debug_local_vars.get(&inst.operands[2]) {
                    value_names
                        .entry(inst.operands[3])
                        .and_modify(|n| {
                            if *n != Some(name) {
                                *n = None;
                            }
                        })
                        .or_insert(Some(name));
                }
            }
        }
        if value_names.is_empty() {
            return;
        }
        let locals: HashSet<u32> = f
            .insts
            .iter()
            .filter(|i| i.opcode == op::VARIABLE)
            .filter_map(|i| i.result_id)
            .collect();
        let mut candidates: HashMap<u32, Option<u32>> = HashMap::new();
        for inst in &f.insts {
            if inst.opcode != op::STORE || inst.operands.len() < 2 {
                continue;
            }
            let (ptr, val) = (inst.operands[0], inst.operands[1]);
            if !locals.contains(&ptr) || self.names.contains_key(&ptr) {
                continue;
            }
            let Some(&Some(name)) = value_names.get(&val) else {
                continue;
            };
            candidates
                .entry(ptr)
                .and_modify(|n| {
                    if *n != Some(name) {
                        *n = None;
                    }
                })
                .or_insert(Some(name));
        }
        for (ptr, name) in candidates {
            if let Some(s) = name.and_then(|n| self.strings.get(&n)).cloned() {
                self.names.insert(ptr, s);
            }
        }
    }
}

fn build_cfg(f: &mut Function) {
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        if b.end == b.start {
            continue;
        }
        let term = &f.insts[b.end - 1];
        let targets: Vec<u32> = match term.opcode {
            op::BRANCH => term.operands.first().copied().into_iter().collect(),
            op::BRANCH_CONDITIONAL => term.operands.iter().skip(1).take(2).copied().collect(),
            op::SWITCH => {
                // Literal width depends on the selector type; Slang emits 32-bit selectors.
                // Targets are every other word after (selector, default).
                let mut t = Vec::new();
                if let Some(&default) = term.operands.get(1) {
                    t.push(default);
                }
                let mut k = 3;
                while k < term.operands.len() {
                    t.push(term.operands[k]);
                    k += 2;
                }
                t
            }
            _ => Vec::new(),
        };
        for t in targets {
            if let Some(&ti) = f.label_to_block.get(&t) {
                edges.push((bi, ti));
            }
        }
    }
    for (from, to) in edges {
        if !f.blocks[from].succs.contains(&to) {
            f.blocks[from].succs.push(to);
        }
        if !f.blocks[to].preds.contains(&from) {
            f.blocks[to].preds.push(from);
        }
    }
}

// ============================================================================
// Intervals
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interval {
    lo: i128,
    hi: i128,
}

impl Interval {
    fn new(lo: i128, hi: i128) -> Interval {
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

/// Abstract value: one interval per lane for integer scalars/vectors, or opaque.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Abs {
    Ints(Vec<Interval>),
    Opaque,
}

impl Abs {
    fn scalar(i: Interval) -> Abs {
        Abs::Ints(vec![i])
    }
    fn top(shape: IntShape) -> Abs {
        Abs::Ints(vec![shape.ty.range(); shape.lanes.max(1)])
    }
    fn lane(&self, k: usize) -> Option<Interval> {
        match self {
            Abs::Ints(v) if v.len() == 1 => v.first().copied(),
            Abs::Ints(v) => v.get(k).copied(),
            Abs::Opaque => None,
        }
    }
    fn as_scalar(&self) -> Option<Interval> {
        match self {
            Abs::Ints(v) if v.len() == 1 => Some(v[0]),
            Abs::Ints(v) => v.iter().copied().reduce(Interval::join),
            Abs::Opaque => None,
        }
    }
    fn join(&self, o: &Abs) -> Abs {
        match (self, o) {
            (Abs::Ints(a), Abs::Ints(b)) if a.len() == b.len() => {
                Abs::Ints(a.iter().zip(b).map(|(x, y)| x.join(*y)).collect())
            }
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
            (Abs::Ints(_), Abs::Opaque) => self.clone(),
            _ => o.clone(),
        }
    }
    fn map(&self, f: impl Fn(Interval) -> Interval) -> Abs {
        match self {
            Abs::Ints(v) => Abs::Ints(v.iter().map(|x| f(*x)).collect()),
            Abs::Opaque => Abs::Opaque,
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
}

/// Values that fit the type keep their interval; anything that may wrap collapses to the type range.
fn wrap(i: Interval, ty: IntTy) -> Interval {
    let r = ty.range();
    if r.contains(i) {
        i
    } else {
        r
    }
}

/// Reinterpret an interval expressed in `from`'s signedness as `to` (same bit width).
fn reinterpret(i: Interval, from: IntTy, to: IntTy) -> Interval {
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
        // signed -> unsigned: negative values map to the upper half.
        if i.hi < 0 {
            return Interval::new(i.lo + full, i.hi + full);
        }
    } else if i.lo >= half {
        // unsigned -> signed: upper half maps to negatives.
        return Interval::new(i.lo - full, i.hi - full);
    }
    to.range()
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
    // Exclude zero divisors (undefined behavior, so any result is acceptable).
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

/// Scratch state for one refined evaluation query.
#[derive(Default)]
struct EvalCtx {
    memo: HashMap<(u32, Vec<Fact>), Abs>,
    visiting: Vec<u32>,
}

/// `a rel b`, compared with the given signedness. Ids are resolved SSA values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Fact {
    a: u32,
    rel: Rel,
    b: u32,
    signed: bool,
}

fn compare_rel(opcode: u16) -> Option<(Rel, bool)> {
    Some(match opcode {
        op::I_EQUAL => (Rel::Eq, false),
        op::I_NOT_EQUAL => (Rel::Ne, false),
        op::U_LESS_THAN => (Rel::Lt, false),
        op::S_LESS_THAN => (Rel::Lt, true),
        op::U_LESS_THAN_EQUAL => (Rel::Le, false),
        op::S_LESS_THAN_EQUAL => (Rel::Le, true),
        op::U_GREATER_THAN => (Rel::Gt, false),
        op::S_GREATER_THAN => (Rel::Gt, true),
        op::U_GREATER_THAN_EQUAL => (Rel::Ge, false),
        op::S_GREATER_THAN_EQUAL => (Rel::Ge, true),
        _ => return None,
    })
}

// ============================================================================
// Per-function analysis
// ============================================================================

/// A synthetic phi introduced by SSA reconstruction of a promoted local.
#[derive(Debug, Clone)]
struct Phi {
    block: usize,
    /// Result type of the promoted variable's pointee.
    ty: u32,
    incoming: Vec<(usize, u32)>,
}

struct FunctionAnalysis<'m> {
    m: &'m Module,
    f: &'m Function,
    name: String,
    /// Immediate dominator per block (`None` for the entry / unreachable blocks).
    idom: Vec<Option<usize>>,
    /// Reverse postorder of reachable blocks.
    rpo: Vec<usize>,
    /// Definition site of every original result id in this function.
    def: HashMap<u32, usize>,
    /// Block of every instruction index.
    inst_block: Vec<usize>,
    /// Load / phi aliases produced by SSA reconstruction (chased by `resolve`).
    alias: HashMap<u32, u32>,
    phis: HashMap<u32, Phi>,
    /// Result type of synthetic values (phis, undefs).
    synth_ty: HashMap<u32, u32>,
    /// Ids that read uninitialized memory; treated as unknown.
    undefs: HashSet<u32>,
    next_id: u32,
    /// Value numbers (structural equivalence) for pure computations.
    vn: HashMap<u32, u32>,
    /// Global (flow-insensitive) ranges after the fixpoint.
    ranges: HashMap<u32, Abs>,
    /// Cached dominating facts per block.
    block_facts: HashMap<usize, Vec<Fact>>,
    /// Values whose range was widened during the ascending fixpoint.
    widened: HashSet<u32>,
}

impl<'m> FunctionAnalysis<'m> {
    fn new(m: &'m Module, f: &'m Function) -> Self {
        let name = m.names.get(&f.id).cloned().unwrap_or_else(|| format!("fn%{}", f.id));
        let mut def = HashMap::new();
        let mut inst_block = vec![0; f.insts.len()];
        // `OpFunctionParameter`s precede the first label; attribute them to the entry block so
        // their types resolve (an untyped parameter would otherwise be treated as opaque).
        let first_block_start = f.blocks.first().map_or(f.insts.len(), |b| b.start);
        for (k, inst) in f.insts.iter().enumerate().take(first_block_start) {
            if let Some(id) = inst.result_id {
                def.insert(id, k);
            }
        }
        for (bi, b) in f.blocks.iter().enumerate() {
            for (k, (slot, inst)) in inst_block
                .iter_mut()
                .zip(&f.insts)
                .enumerate()
                .take(b.end)
                .skip(b.start)
            {
                *slot = bi;
                if let Some(id) = inst.result_id {
                    def.insert(id, k);
                }
            }
        }
        let bound = f
            .insts
            .iter()
            .filter_map(|i| i.result_id)
            .max()
            .unwrap_or(0)
            .max(m.types.keys().copied().max().unwrap_or(0))
            .max(m.int_consts.keys().copied().max().unwrap_or(0))
            .max(m.global_vars.keys().copied().max().unwrap_or(0));
        let (idom, rpo) = dominators(f);
        Self {
            m,
            f,
            name,
            idom,
            rpo,
            def,
            inst_block,
            alias: HashMap::new(),
            phis: HashMap::new(),
            synth_ty: HashMap::new(),
            undefs: HashSet::new(),
            next_id: bound + 1,
            vn: HashMap::new(),
            ranges: HashMap::new(),
            block_facts: HashMap::new(),
            widened: HashSet::new(),
        }
    }

    fn run(mut self, report: &mut BoundsReport) {
        if self.f.blocks.is_empty() {
            return;
        }
        self.promote_locals();
        self.value_number();
        self.fixpoint();
        self.check_accesses(report);
    }

    fn fresh_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn resolve(&self, mut id: u32) -> u32 {
        let mut guard = 0;
        while let Some(&next) = self.alias.get(&id) {
            id = next;
            guard += 1;
            if guard > 64 {
                break;
            }
        }
        id
    }

    fn inst_of(&self, id: u32) -> Option<&'m Inst> {
        let f: &'m Function = self.f;
        self.def.get(&id).map(move |&k| &f.insts[k])
    }

    fn type_of(&self, id: u32) -> Option<u32> {
        if let Some(&t) = self.synth_ty.get(&id) {
            return Some(t);
        }
        if let Some(inst) = self.inst_of(id) {
            return inst.result_type;
        }
        if let Some(&(_, ptr_ty)) = self.m.global_vars.get(&id) {
            return Some(ptr_ty);
        }
        self.m.const_types.get(&id).copied()
    }

    fn int_shape_of(&self, id: u32) -> Option<IntShape> {
        self.type_of(id).and_then(|t| self.m.int_shape(t))
    }

    // ------------------------------------------------------------------
    // SSA reconstruction for non-escaping scalar integer locals
    // ------------------------------------------------------------------

    fn promote_locals(&mut self) {
        let f = self.f;
        let entry = &f.blocks[0];
        let mut candidates: Vec<(u32, u32)> = Vec::new(); // (var id, pointee type)
        for k in entry.start..entry.end {
            let inst = &f.insts[k];
            if inst.opcode != op::VARIABLE {
                continue;
            }
            let (Some(id), Some(ptr_ty)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            let Some((sc, pointee)) = self.m.pointee(ptr_ty) else {
                continue;
            };
            if sc != op::SC_FUNCTION || inst.operands.len() > 1 {
                // Only uninitialized Function-storage variables (Slang never emits initializers).
                continue;
            }
            if !matches!(self.m.types.get(&pointee), Some(Ty::Int(_)) | Some(Ty::Bool)) {
                continue;
            }
            candidates.push((id, pointee));
        }
        if candidates.is_empty() {
            return;
        }

        // Escape analysis: a variable is promotable when every mention is a direct
        // OpLoad / OpStore pointer operand (debug ext-insts are ignored).
        let mut promotable: Vec<(u32, u32)> = Vec::new();
        'vars: for (var, pointee) in candidates {
            for inst in &f.insts {
                match inst.opcode {
                    op::VARIABLE | op::EXT_INST => continue,
                    op::LOAD if inst.operands.first() == Some(&var) => continue,
                    op::STORE if inst.operands.first() == Some(&var) => {
                        if inst.operands.get(1) == Some(&var) {
                            continue 'vars;
                        }
                        continue;
                    }
                    _ => {}
                }
                if inst.all_words.contains(&var) {
                    continue 'vars;
                }
            }
            promotable.push((var, pointee));
        }

        let df = dominance_frontiers(f, &self.idom);
        let nblocks = f.blocks.len();
        for (var, pointee) in promotable {
            // Blocks that store to the variable.
            let mut def_blocks: Vec<usize> = Vec::new();
            for (bi, b) in f.blocks.iter().enumerate() {
                if (b.start..b.end).any(|k| {
                    let i = &f.insts[k];
                    i.opcode == op::STORE && i.operands.first() == Some(&var)
                }) {
                    def_blocks.push(bi);
                }
            }
            // Iterated dominance frontier -> phi blocks.
            let mut phi_blocks: HashSet<usize> = HashSet::new();
            let mut work: Vec<usize> = def_blocks.clone();
            work.push(0);
            let mut seen = vec![false; nblocks];
            while let Some(b) = work.pop() {
                for &d in &df[b] {
                    if phi_blocks.insert(d) && !seen[d] {
                        seen[d] = true;
                        work.push(d);
                    }
                }
            }
            let mut phi_at: HashMap<usize, u32> = HashMap::new();
            for &b in &phi_blocks {
                let id = self.fresh_id();
                phi_at.insert(b, id);
                self.synth_ty.insert(id, pointee);
                self.phis.insert(
                    id,
                    Phi {
                        block: b,
                        ty: pointee,
                        incoming: Vec::new(),
                    },
                );
            }
            let undef = self.fresh_id();
            self.synth_ty.insert(undef, pointee);
            self.undefs.insert(undef);

            // Renaming over the dominator tree.
            let mut children: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
            for (b, d) in self.idom.iter().enumerate() {
                if let Some(d) = d {
                    children[*d].push(b);
                }
            }
            let mut stack: Vec<u32> = vec![undef];
            let mut incoming: Vec<(u32, usize, u32)> = Vec::new(); // (phi, pred, value)
            self.rename(0, var, &phi_at, &children, &mut stack, &mut incoming);
            for (phi, pred, value) in incoming {
                if let Some(p) = self.phis.get_mut(&phi) {
                    p.incoming.push((pred, value));
                }
            }
        }
        self.simplify_phis();
    }

    fn rename(
        &mut self,
        block: usize,
        var: u32,
        phi_at: &HashMap<usize, u32>,
        children: &[Vec<usize>],
        stack: &mut Vec<u32>,
        incoming: &mut Vec<(u32, usize, u32)>,
    ) {
        let depth = stack.len();
        if let Some(&phi) = phi_at.get(&block) {
            stack.push(phi);
        }
        let f: &'m Function = self.f;
        let b = &f.blocks[block];
        for k in b.start..b.end {
            let inst = &f.insts[k];
            match inst.opcode {
                op::LOAD if inst.operands.first() == Some(&var) => {
                    if let Some(id) = inst.result_id {
                        self.alias.insert(id, *stack.last().unwrap());
                    }
                }
                op::STORE if inst.operands.first() == Some(&var) => {
                    if let Some(&v) = inst.operands.get(1) {
                        stack.push(v);
                    }
                }
                _ => {}
            }
        }
        for &s in &b.succs {
            if let Some(&phi) = phi_at.get(&s) {
                incoming.push((phi, block, *stack.last().unwrap()));
            }
        }
        for &c in &children[block] {
            self.rename(c, var, phi_at, children, stack, incoming);
        }
        stack.truncate(depth);
    }

    /// Phis whose incoming values all resolve to one value (or themselves) become aliases.
    fn simplify_phis(&mut self) {
        loop {
            let mut changed = false;
            let mut ids: Vec<u32> = self.phis.keys().copied().collect();
            ids.sort_unstable();
            for id in ids {
                let Some(phi) = self.phis.get(&id) else { continue };
                let mut unique: Option<u32> = None;
                let mut trivial = true;
                for &(_, v) in &phi.incoming {
                    let v = self.resolve(v);
                    if v == id {
                        continue;
                    }
                    match unique {
                        None => unique = Some(v),
                        Some(u) if u == v => {}
                        Some(_) => {
                            trivial = false;
                            break;
                        }
                    }
                }
                if trivial {
                    if let Some(u) = unique {
                        self.alias.insert(id, u);
                        self.phis.remove(&id);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // Value numbering
    // ------------------------------------------------------------------

    fn pointer_base(&self, mut ptr: u32) -> Option<u32> {
        for _ in 0..32 {
            ptr = self.resolve(ptr);
            if self.m.global_vars.contains_key(&ptr) {
                return Some(ptr);
            }
            let inst = self.inst_of(ptr)?;
            match inst.opcode {
                op::ACCESS_CHAIN
                | op::IN_BOUNDS_ACCESS_CHAIN
                | op::PTR_ACCESS_CHAIN
                | op::IN_BOUNDS_PTR_ACCESS_CHAIN
                | op::COPY_OBJECT => ptr = *inst.operands.first()?,
                op::VARIABLE | op::FUNCTION_PARAMETER => return Some(ptr),
                _ => return None,
            }
        }
        None
    }

    fn is_pure_load(&self, inst: &Inst) -> bool {
        let Some(&ptr) = inst.operands.first() else {
            return false;
        };
        let Some(base) = self.pointer_base(ptr) else {
            return false;
        };
        matches!(
            self.m.global_vars.get(&base).map(|v| v.0),
            Some(op::SC_INPUT) | Some(op::SC_UNIFORM) | Some(op::SC_PUSH_CONSTANT) | Some(op::SC_UNIFORM_CONSTANT)
        )
    }

    fn value_number(&mut self) {
        let mut table: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut next_vn = 1u32;
        let pure = |opcode: u16| {
            matches!(
                opcode,
                op::ACCESS_CHAIN
                    | op::IN_BOUNDS_ACCESS_CHAIN
                    | op::VECTOR_SHUFFLE
                    | op::COMPOSITE_CONSTRUCT
                    | op::COMPOSITE_EXTRACT
                    | op::COPY_OBJECT
                    | op::U_CONVERT
                    | op::S_CONVERT
                    | op::BITCAST
                    | op::S_NEGATE
                    | op::I_ADD
                    | op::I_SUB
                    | op::I_MUL
                    | op::U_DIV
                    | op::S_DIV
                    | op::U_MOD
                    | op::S_REM
                    | op::S_MOD
                    | op::LOGICAL_EQUAL
                    | op::LOGICAL_NOT_EQUAL
                    | op::LOGICAL_OR
                    | op::LOGICAL_AND
                    | op::LOGICAL_NOT
                    | op::SELECT
                    | op::I_EQUAL
                    | op::I_NOT_EQUAL
                    | op::U_GREATER_THAN
                    | op::S_GREATER_THAN
                    | op::U_GREATER_THAN_EQUAL
                    | op::S_GREATER_THAN_EQUAL
                    | op::U_LESS_THAN
                    | op::S_LESS_THAN
                    | op::U_LESS_THAN_EQUAL
                    | op::S_LESS_THAN_EQUAL
                    | op::SHIFT_RIGHT_LOGICAL
                    | op::SHIFT_RIGHT_ARITHMETIC
                    | op::SHIFT_LEFT_LOGICAL
                    | op::BITWISE_OR
                    | op::BITWISE_XOR
                    | op::BITWISE_AND
                    | op::NOT
                    | op::EXT_INST
            )
        };
        // Function parameters live before the first block; give them unique numbers first.
        for inst in &self.f.insts {
            if inst.opcode == op::FUNCTION_PARAMETER {
                if let Some(id) = inst.result_id {
                    self.vn.insert(id, next_vn);
                    next_vn += 1;
                }
            }
        }
        for &bi in &self.rpo {
            let b = &self.f.blocks[bi];
            for k in b.start..b.end {
                let inst = &self.f.insts[k];
                let Some(id) = inst.result_id else { continue };
                if self.alias.contains_key(&id) {
                    continue;
                }
                let key: Option<Vec<u32>> = if pure(inst.opcode) {
                    let mut key = vec![u32::from(inst.opcode), inst.result_type.unwrap_or(0)];
                    let mut ok = true;
                    for &o in &inst.operands {
                        // Literal operands (component indices, ext-inst opcodes) hash as-is;
                        // ids hash by value number. Mixing them is harmless because literals
                        // and ids sit at fixed operand positions per opcode.
                        let r = self.resolve(o);
                        match self.vn.get(&r) {
                            Some(&v) => key.push(v | 0x8000_0000),
                            None if self.def.contains_key(&r) || self.phis.contains_key(&r) => {
                                ok = false;
                                break;
                            }
                            None => key.push(r),
                        }
                    }
                    ok.then_some(key)
                } else if inst.opcode == op::LOAD && self.is_pure_load(inst) {
                    let ptr = self.resolve(inst.operands[0]);
                    let ptr_vn = self.vn.get(&ptr).copied().unwrap_or(ptr | 0x4000_0000);
                    Some(vec![u32::from(op::LOAD), inst.result_type.unwrap_or(0), ptr_vn])
                } else {
                    None
                };
                let v = match key {
                    Some(key) => *table.entry(key).or_insert_with(|| {
                        let v = next_vn;
                        next_vn += 1;
                        v
                    }),
                    None => {
                        let v = next_vn;
                        next_vn += 1;
                        v
                    }
                };
                self.vn.insert(id, v);
            }
        }
    }

    fn same_value(&self, a: u32, b: u32) -> bool {
        let a = self.resolve(a);
        let b = self.resolve(b);
        if a == b {
            return true;
        }
        match (self.vn.get(&a), self.vn.get(&b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }

    // ------------------------------------------------------------------
    // Transfer functions
    // ------------------------------------------------------------------

    fn builtin_range(&self, var: u32, shape: IntShape) -> Abs {
        let top = Abs::top(shape);
        let Some(&builtin) = self.m.builtins.get(&var) else {
            return top;
        };
        match builtin {
            op::BUILTIN_LOCAL_INVOCATION_ID => match self.m.local_size {
                Some(ls) => Abs::Ints(ls.iter().map(|&n| Interval::new(0, (n - 1).max(0))).collect()),
                None => top,
            },
            op::BUILTIN_LOCAL_INVOCATION_INDEX => match self.m.local_size {
                Some(ls) => Abs::scalar(Interval::new(0, (ls[0] * ls[1] * ls[2] - 1).max(0))),
                None => top,
            },
            op::BUILTIN_WORKGROUP_SIZE => match self.m.local_size {
                Some(ls) => Abs::Ints(ls.iter().map(|&n| Interval::point(n)).collect()),
                None => top,
            },
            op::BUILTIN_SUBGROUP_SIZE => Abs::scalar(Interval::new(1, MAX_SUBGROUP_SIZE)),
            op::BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID => Abs::scalar(Interval::new(0, MAX_SUBGROUP_SIZE - 1)),
            op::BUILTIN_NUM_SUBGROUPS | op::BUILTIN_SUBGROUP_ID | op::BUILTIN_WORKGROUP_ID => top,
            _ => top,
        }
    }

    fn constant_abs(&self, id: u32) -> Option<Abs> {
        if let Some(&v) = self.m.int_consts.get(&id) {
            return Some(Abs::scalar(Interval::point(v)));
        }
        if let Some(parts) = self.m.composite_consts.get(&id) {
            let lanes: Option<Vec<Interval>> = parts
                .iter()
                .map(|p| self.m.int_consts.get(p).map(|&v| Interval::point(v)))
                .collect();
            return lanes.map(Abs::Ints);
        }
        None
    }

    /// Compute the abstract value of `inst` from operand values supplied by `get`.
    /// Returns `None` when an operand is still bottom (fixpoint not yet reached).
    fn transfer(&self, inst: &Inst, get: &mut dyn FnMut(u32) -> Option<Abs>) -> Option<Abs> {
        let shape = inst.result_type.and_then(|t| self.m.int_shape(t));
        let Some(shape) = shape else {
            return Some(Abs::Opaque);
        };
        let ty = shape.ty;
        let top = Abs::top(shape);
        let ops = &inst.operands;
        let bin = |get: &mut dyn FnMut(u32) -> Option<Abs>, f: &dyn Fn(Interval, Interval) -> Interval| {
            let a = get(*ops.first()?)?;
            let b = get(*ops.get(1)?)?;
            Some(a.zip(&b, f))
        };
        match inst.opcode {
            op::LOAD => {
                let ptr = ops.first().copied()?;
                let resolved = self.resolve(ptr);
                if let Some(base) = self.pointer_base(ptr) {
                    if self.m.builtins.contains_key(&base) {
                        if resolved == base {
                            return Some(self.builtin_range(base, shape));
                        }
                        // Component load: OpAccessChain %builtin %const.
                        if let Some(chain) = self.inst_of(resolved) {
                            if matches!(chain.opcode, op::ACCESS_CHAIN | op::IN_BOUNDS_ACCESS_CHAIN)
                                && chain.operands.len() == 2
                            {
                                let vec_shape = self
                                    .m
                                    .global_vars
                                    .get(&base)
                                    .and_then(|&(_, pt)| self.m.pointee(pt))
                                    .and_then(|(_, t)| self.m.int_shape(t));
                                if let (Some(vs), Some(&c)) =
                                    (vec_shape, self.m.int_consts.get(&self.resolve(chain.operands[1])))
                                {
                                    if let Some(lane) = self.builtin_range(base, vs).lane(c.max(0) as usize) {
                                        return Some(Abs::scalar(lane));
                                    }
                                }
                            }
                        }
                    }
                }
                Some(top)
            }
            op::COPY_OBJECT => get(*ops.first()?),
            op::PHI => {
                let mut out: Option<Abs> = None;
                let mut k = 0;
                while k + 1 < ops.len() {
                    if let Some(v) = get(ops[k]) {
                        out = Some(out.map_or(v.clone(), |o| o.join(&v)));
                    }
                    k += 2;
                }
                out
            }
            op::SELECT if ops.len() >= 3 => {
                let a = get(ops[1])?;
                let b = get(ops[2])?;
                Some(a.join(&b))
            }
            op::COMPOSITE_EXTRACT if !ops.is_empty() => {
                let v = get(ops[0])?;
                if ops.len() == 2 {
                    return Some(v.lane(ops[1] as usize).map_or(top, Abs::scalar));
                }
                Some(top)
            }
            op::COMPOSITE_CONSTRUCT => {
                let mut lanes = Vec::new();
                for &o in ops {
                    match get(o)? {
                        Abs::Ints(v) => lanes.extend(v),
                        Abs::Opaque => return Some(top),
                    }
                }
                Some(Abs::Ints(lanes))
            }
            op::VECTOR_SHUFFLE if ops.len() >= 2 => {
                let a = get(ops[0])?;
                let b = get(ops[1])?;
                let (Abs::Ints(a), Abs::Ints(b)) = (a, b) else {
                    return Some(top);
                };
                let lanes = ops[2..]
                    .iter()
                    .map(|&c| {
                        let c = c as usize;
                        if c < a.len() {
                            a[c]
                        } else if c - a.len() < b.len() {
                            b[c - a.len()]
                        } else {
                            ty.range()
                        }
                    })
                    .collect();
                Some(Abs::Ints(lanes))
            }
            op::BITCAST | op::U_CONVERT | op::S_CONVERT if !ops.is_empty() => {
                let a = get(ops[0])?;
                let Some(src_shape) = self.int_shape_of(self.resolve(ops[0])) else {
                    return Some(top);
                };
                let from = src_shape.ty;
                Some(a.map(|i| {
                    if inst.opcode == op::BITCAST {
                        if from.bits == ty.bits {
                            reinterpret(i, from, ty)
                        } else {
                            ty.range()
                        }
                    } else {
                        let view = from.with_signed(inst.opcode == op::S_CONVERT);
                        let i = reinterpret(i, from, view);
                        // Widening in the view's signedness preserves the value; then move to
                        // the destination signedness and clamp to what fits.
                        let widened = IntTy {
                            bits: ty.bits,
                            signed: view.signed,
                        };
                        if ty.bits >= from.bits {
                            reinterpret(i, widened, ty)
                        } else {
                            wrap(i, ty)
                        }
                    }
                }))
            }
            op::S_NEGATE => Some(get(*ops.first()?)?.map(|i| wrap(Interval::new(-i.hi, -i.lo), ty))),
            op::I_ADD => bin(get, &|a, b| wrap(Interval::new(a.lo + b.lo, a.hi + b.hi), ty)),
            op::I_SUB => bin(get, &|a, b| wrap(Interval::new(a.lo - b.hi, a.hi - b.lo), ty)),
            op::I_MUL => bin(get, &|a, b| {
                let c = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
                wrap(Interval::new(*c.iter().min().unwrap(), *c.iter().max().unwrap()), ty)
            }),
            op::U_DIV => bin(get, &|a, b| {
                let u = ty.with_signed(false);
                let a = reinterpret(a, ty, u);
                let b = reinterpret(b, ty, u);
                if b.hi < 1 {
                    return ty.range();
                }
                let lo_b = b.lo.max(1);
                reinterpret(Interval::new(a.lo / b.hi, a.hi / lo_b), u, ty)
            }),
            op::S_DIV => bin(get, &|a, b| wrap(div_trunc_hull(a, b), ty)),
            op::U_MOD => bin(get, &|a, b| {
                let u = ty.with_signed(false);
                let a = reinterpret(a, ty, u);
                let b = reinterpret(b, ty, u);
                if b.hi < 1 {
                    return ty.range();
                }
                let r = if a.hi < b.lo.max(1) {
                    a
                } else {
                    Interval::new(0, (b.hi - 1).min(a.hi))
                };
                reinterpret(r, u, ty)
            }),
            op::S_REM => bin(get, &|a, b| {
                let m = b.lo.abs().max(b.hi.abs());
                if m == 0 {
                    return ty.range();
                }
                let lim = m - 1;
                let lo = if a.lo >= 0 { 0 } else { (-lim).max(a.lo) };
                let hi = if a.hi <= 0 { 0 } else { lim.min(a.hi) };
                Interval::new(lo, hi)
            }),
            op::S_MOD => bin(get, &|a, b| {
                if b.lo > 0 {
                    Interval::new(0, if a.is_nonneg() { (b.hi - 1).min(a.hi) } else { b.hi - 1 })
                } else if b.hi < 0 {
                    Interval::new(b.lo + 1, 0)
                } else {
                    let m = b.lo.abs().max(b.hi.abs());
                    if m == 0 {
                        ty.range()
                    } else {
                        Interval::new(-(m - 1), m - 1)
                    }
                }
            }),
            op::SHIFT_LEFT_LOGICAL => bin(get, &|a, s| match shift_amounts(s, ty.bits) {
                Some((lo_s, hi_s)) if a.is_nonneg() => wrap(Interval::new(a.lo << lo_s, a.hi << hi_s), ty),
                _ => ty.range(),
            }),
            op::SHIFT_RIGHT_LOGICAL => bin(get, &|a, s| match shift_amounts(s, ty.bits) {
                Some((lo_s, hi_s)) => {
                    let u = ty.with_signed(false);
                    let a = reinterpret(a, ty, u);
                    reinterpret(Interval::new(a.lo >> hi_s, a.hi >> lo_s), u, ty)
                }
                None => ty.range(),
            }),
            op::SHIFT_RIGHT_ARITHMETIC => bin(get, &|a, s| match shift_amounts(s, ty.bits) {
                Some((lo_s, hi_s)) => {
                    let c = [a.lo >> lo_s, a.lo >> hi_s, a.hi >> lo_s, a.hi >> hi_s];
                    wrap(Interval::new(*c.iter().min().unwrap(), *c.iter().max().unwrap()), ty)
                }
                None => ty.range(),
            }),
            op::BITWISE_AND => bin(get, &|a, b| {
                // x & m with m >= 0 never exceeds m, whatever x is.
                match (a.is_nonneg(), b.is_nonneg()) {
                    (true, true) => Interval::new(0, a.hi.min(b.hi)),
                    (true, false) => Interval::new(0, a.hi),
                    (false, true) => Interval::new(0, b.hi),
                    (false, false) => ty.range(),
                }
            }),
            op::BITWISE_OR => bin(get, &|a, b| {
                if a.is_nonneg() && b.is_nonneg() {
                    wrap(Interval::new(a.lo.max(b.lo), next_pow2_minus1(a.hi.max(b.hi))), ty)
                } else {
                    ty.range()
                }
            }),
            op::BITWISE_XOR => bin(get, &|a, b| {
                if a.is_nonneg() && b.is_nonneg() {
                    wrap(Interval::new(0, next_pow2_minus1(a.hi.max(b.hi))), ty)
                } else {
                    ty.range()
                }
            }),
            op::EXT_INST => {
                if ops.len() < 2 || Some(ops[0]) != self.m.glsl_ext {
                    return Some(top);
                }
                let args = &ops[2..];
                let minmax = |get: &mut dyn FnMut(u32) -> Option<Abs>, signed: bool, is_max: bool| {
                    let a = get(args[0])?;
                    let b = get(args[1])?;
                    let view = ty.with_signed(signed);
                    Some(a.zip(&b, |x, y| {
                        let x = reinterpret(x, ty, view);
                        let y = reinterpret(y, ty, view);
                        let r = if is_max {
                            Interval::new(x.lo.max(y.lo), x.hi.max(y.hi))
                        } else {
                            Interval::new(x.lo.min(y.lo), x.hi.min(y.hi))
                        };
                        reinterpret(r, view, ty)
                    }))
                };
                match ops[1] {
                    op::GLSL_U_MIN if args.len() >= 2 => minmax(get, false, false),
                    op::GLSL_S_MIN if args.len() >= 2 => minmax(get, true, false),
                    op::GLSL_U_MAX if args.len() >= 2 => minmax(get, false, true),
                    op::GLSL_S_MAX if args.len() >= 2 => minmax(get, true, true),
                    op::GLSL_U_CLAMP | op::GLSL_S_CLAMP if args.len() >= 3 => {
                        let signed = ops[1] == op::GLSL_S_CLAMP;
                        let view = ty.with_signed(signed);
                        let x = get(args[0])?;
                        let l = get(args[1])?;
                        let h = get(args[2])?;
                        let lower = x.zip(&l, |x, l| {
                            let x = reinterpret(x, ty, view);
                            let l = reinterpret(l, ty, view);
                            Interval::new(x.lo.max(l.lo), x.hi.max(l.hi))
                        });
                        Some(lower.zip(&h, |p, h| {
                            let h = reinterpret(h, ty, view);
                            reinterpret(Interval::new(p.lo.min(h.lo), p.hi.min(h.hi)), view, ty)
                        }))
                    }
                    op::GLSL_S_ABS if !args.is_empty() => Some(get(args[0])?.map(|x| {
                        let m = x.lo.abs().max(x.hi.abs());
                        let lo = if x.lo <= 0 && x.hi >= 0 {
                            0
                        } else {
                            x.lo.abs().min(x.hi.abs())
                        };
                        wrap(Interval::new(lo, m), ty)
                    })),
                    _ => Some(top),
                }
            }
            _ => Some(top),
        }
    }

    // ------------------------------------------------------------------
    // Global fixpoint
    // ------------------------------------------------------------------

    fn fixpoint(&mut self) {
        let mut order: Vec<u32> = Vec::new();
        for &bi in &self.rpo {
            let b = &self.f.blocks[bi];
            for k in b.start..b.end {
                if let Some(id) = self.f.insts[k].result_id {
                    if !self.alias.contains_key(&id) {
                        order.push(id);
                    }
                }
            }
        }
        let mut phi_ids: Vec<u32> = self.phis.keys().copied().collect();
        phi_ids.sort_unstable(); // deterministic widening behaviour
        let all: Vec<u32> = phi_ids.iter().chain(order.iter()).copied().collect();
        let mut grow_count: HashMap<u32, u32> = HashMap::new();
        // Computed in place so `eval` can consult the in-progress ranges.
        self.ranges = HashMap::new();

        // Ascending phase: join with the previous value, widening anything that keeps
        // growing, until nothing changes (or everything is forced to top).
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

        // Descending (narrowing) phase: from the post-fixpoint, recompute each value and keep
        // the intersection. Recovers the precision widening gave away, e.g. a counted loop's
        // `[0, n]` after the header was widened to `[0, MAX]`. Every input is a sound
        // over-approximation, so each recomputed value (and its meet with the old one) is too.
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
        if let Some(phi) = self.phis.get(&id) {
            let Some(shape) = self.m.int_shape(phi.ty) else {
                return Some(Abs::Opaque);
            };
            if force_top {
                return Some(Abs::top(shape));
            }
            return self.phi_incoming_join(id, self.f.blocks[phi.block].label, &phi.incoming);
        }
        let inst = self.inst_of(id)?;
        if inst.opcode == op::PHI {
            // Native phi (Slang emits these at higher optimization levels): same edge-refined
            // join as the synthetic ones.
            if let Some(shape) = self.int_shape_of(id) {
                if force_top {
                    return Some(Abs::top(shape));
                }
                let mut incoming = Vec::new();
                let mut k = 0;
                while k + 1 < inst.operands.len() {
                    if let Some(&pred) = self.f.label_to_block.get(&inst.operands[k + 1]) {
                        incoming.push((pred, inst.operands[k]));
                    }
                    k += 2;
                }
                let block_label = self.f.blocks[self.inst_block[self.def[&id]]].label;
                return self.phi_incoming_join(id, block_label, &incoming);
            }
        }
        let mut get = |o: u32| self.lookup(&self.ranges, o);
        self.transfer(inst, &mut get)
    }

    /// Standard interval widening: a bound that moved since the previous pass jumps to the
    /// type's limit in that direction.
    fn widen(&self, id: u32, old: &Abs, new: &Abs) -> Abs {
        let (Abs::Ints(o), Abs::Ints(nv)) = (old, new) else {
            return new.clone();
        };
        let Some(ty) = self.int_shape_of(id).map(|s| s.ty) else {
            return new.clone();
        };
        let r = ty.range();
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

    /// Join of a phi's incoming values, each evaluated under the facts that hold on its edge.
    ///
    /// For a counted loop `for (i = 0; i < n; i++)` the back-edge operand `i + 1` is only
    /// reached when `i < n`, so evaluating it under that fact keeps the header value at
    /// `[0, n]` instead of widening to the type's range and then wrapping on the increment.
    /// The phi itself is marked as visiting so the recursion bottoms out at its current
    /// (fact-refined) range. Incoming values not yet computed in this pass are bottom and
    /// contribute nothing.
    fn phi_incoming_join(&self, id: u32, block_label: u32, incoming: &[(usize, u32)]) -> Option<Abs> {
        let mut out: Option<Abs> = None;
        for &(pred, v) in incoming {
            let Some(flat) = self.lookup(&self.ranges, v) else {
                continue;
            };
            let val = if matches!(flat, Abs::Ints(_)) && !self.ranges.contains_key(&id) {
                // First visit: nothing to refine against yet.
                flat
            } else if let Abs::Ints(_) = flat {
                let mut edge = Vec::new();
                self.edge_facts(pred, block_label, &mut edge);
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

    fn lookup(&self, ranges: &HashMap<u32, Abs>, id: u32) -> Option<Abs> {
        let id = self.resolve(id);
        if let Some(c) = self.constant_abs(id) {
            return Some(c);
        }
        if let Some(r) = ranges.get(&id) {
            return Some(r.clone());
        }
        if self.undefs.contains(&id) {
            return Some(self.int_shape_of(id).map_or(Abs::Opaque, Abs::top));
        }
        if self.inst_of(id).is_some_and(|i| i.opcode == op::FUNCTION_PARAMETER) {
            return Some(self.int_shape_of(id).map_or(Abs::Opaque, Abs::top));
        }
        if self.def.contains_key(&id) || self.phis.contains_key(&id) {
            // Defined in this function but not yet computed (bottom).
            return None;
        }
        // Function parameters, globals, values from other scopes: unknown.
        Some(self.int_shape_of(id).map_or(Abs::Opaque, Abs::top))
    }

    fn global(&self, id: u32) -> Abs {
        let id = self.resolve(id);
        self.lookup(&self.ranges, id)
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
        if let Some(phi) = self.phis.get(&cond) {
            // Slang lowers `a && b` to `bool t; if (a) t = b; else t = false;` (and `||`
            // symmetrically). A bool phi is true only via an incoming edge whose value can be
            // true; when exactly one such edge exists, its edge condition and value both hold.
            let block_label = self.f.blocks[phi.block].label;
            let mut candidates = Vec::new();
            for &(pred, v) in &phi.incoming {
                let v = self.resolve(v);
                if self.m.bool_consts.get(&v) == Some(&!polarity) {
                    continue; // cannot produce the wanted polarity
                }
                candidates.push((pred, v));
            }
            if let [(pred, v)] = candidates[..] {
                self.edge_facts(pred, block_label, out);
                if self.m.bool_consts.get(&v) != Some(&polarity) {
                    self.facts_from_condition(v, polarity, out, depth + 1);
                }
            }
            return;
        }
        let Some(inst) = self.inst_of(cond) else { return };
        if let Some((rel, signed)) = compare_rel(inst.opcode) {
            if inst.operands.len() >= 2 {
                let rel = if polarity { rel } else { rel.negate() };
                out.push(Fact {
                    a: self.resolve(inst.operands[0]),
                    rel,
                    b: self.resolve(inst.operands[1]),
                    signed,
                });
            }
            return;
        }
        match inst.opcode {
            op::LOGICAL_AND if polarity => {
                for &o in inst.operands.iter().take(2) {
                    self.facts_from_condition(o, true, out, depth + 1);
                }
            }
            op::LOGICAL_OR if !polarity => {
                for &o in inst.operands.iter().take(2) {
                    self.facts_from_condition(o, false, out, depth + 1);
                }
            }
            op::LOGICAL_NOT => {
                if let Some(&o) = inst.operands.first() {
                    self.facts_from_condition(o, !polarity, out, depth + 1);
                }
            }
            op::SELECT if inst.operands.len() >= 3 => {
                // select(c, x, false) == c && x ; select(c, true, x) == c || x
                let (c, x, y) = (inst.operands[0], inst.operands[1], inst.operands[2]);
                let bconst = |id: u32| self.m.bool_consts.get(&self.resolve(id)).copied();
                if polarity && bconst(y) == Some(false) {
                    self.facts_from_condition(c, true, out, depth + 1);
                    self.facts_from_condition(x, true, out, depth + 1);
                } else if !polarity && bconst(x) == Some(true) {
                    self.facts_from_condition(c, false, out, depth + 1);
                    self.facts_from_condition(y, false, out, depth + 1);
                }
            }
            _ => {}
        }
    }

    /// Facts that hold on entry to `block`: conditions on every dominating single-predecessor edge.
    fn facts_for_block(&mut self, block: usize) -> Vec<Fact> {
        if let Some(f) = self.block_facts.get(&block) {
            return f.clone();
        }
        let facts = self.compute_block_facts(block);
        self.block_facts.insert(block, facts.clone());
        facts
    }

    fn compute_block_facts(&self, block: usize) -> Vec<Fact> {
        let mut facts = Vec::new();
        let mut cur = Some(block);
        while let Some(b) = cur {
            let blk = &self.f.blocks[b];
            if blk.preds.len() == 1 {
                let p = blk.preds[0];
                let pb = &self.f.blocks[p];
                if pb.end > pb.start {
                    let term = &self.f.insts[pb.end - 1];
                    if term.opcode == op::BRANCH_CONDITIONAL && term.operands.len() >= 3 {
                        let (c, t, f) = (term.operands[0], term.operands[1], term.operands[2]);
                        if t != f {
                            if t == blk.label {
                                self.facts_from_condition(c, true, &mut facts, 0);
                            } else if f == blk.label {
                                self.facts_from_condition(c, false, &mut facts, 0);
                            }
                        }
                    }
                }
            }
            cur = self.idom[b];
        }
        facts
    }

    /// Facts that hold along the CFG edge `pred -> succ` (used for phi / select operands):
    /// everything that dominates `pred`, plus `pred`'s own branch condition when it is
    /// conditional. These are the path-specific facts the merge block itself has lost.
    fn edge_facts(&self, pred: usize, succ_label: u32, out: &mut Vec<Fact>) {
        for f in self.compute_block_facts(pred) {
            if !out.contains(&f) {
                out.push(f);
            }
        }
        let pb = &self.f.blocks[pred];
        if pb.end == pb.start {
            return;
        }
        let term = &self.f.insts[pb.end - 1];
        if term.opcode == op::BRANCH_CONDITIONAL && term.operands.len() >= 3 {
            let (c, t, f) = (term.operands[0], term.operands[1], term.operands[2]);
            if t != f {
                if t == succ_label {
                    self.facts_from_condition(c, true, out, 0);
                } else if f == succ_label {
                    self.facts_from_condition(c, false, out, 0);
                }
            }
        }
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
        let bound = view.range();
        let refined = match rel {
            Rel::Lt => Interval::new(rv.lo, rv.hi.min(ov.hi - 1)),
            Rel::Le => Interval::new(rv.lo, rv.hi.min(ov.hi)),
            Rel::Gt => Interval::new(rv.lo.max(ov.lo + 1), rv.hi),
            Rel::Ge => Interval::new(rv.lo.max(ov.lo), rv.hi),
            Rel::Eq => Interval::new(rv.lo.max(ov.lo), rv.hi.min(ov.hi)),
            Rel::Ne => {
                if ov.lo == ov.hi {
                    if rv.lo == ov.lo && rv.hi > rv.lo {
                        Interval::new(rv.lo + 1, rv.hi)
                    } else if rv.hi == ov.lo && rv.hi > rv.lo {
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
            // Infeasible path: keep the unrefined range rather than claiming emptiness.
            return r;
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
        if depth > REFINE_DEPTH_LIMIT || ctx.visiting.contains(&id) {
            return self.apply_facts(id, global, shape, facts, depth, ctx);
        }
        let memo_key = (id, facts.to_vec());
        if let Some(v) = ctx.memo.get(&memo_key) {
            return v.clone();
        }
        ctx.visiting.push(id);
        let mut base = global.clone();
        if let Some(phi) = self.phis.get(&id) {
            let mut out: Option<Abs> = None;
            for &(pred, v) in &phi.incoming {
                let mut edge = facts.to_vec();
                self.edge_facts(pred, self.f.blocks[phi.block].label, &mut edge);
                let val = self.eval(v, &edge, depth + 1, ctx);
                out = Some(out.map_or(val.clone(), |o| o.join(&val)));
            }
            if let Some(o) = out {
                base = o.meet(&global);
            }
        } else if let Some(inst) = self.inst_of(id) {
            match inst.opcode {
                op::SELECT if inst.operands.len() >= 3 => {
                    let mut ft = facts.to_vec();
                    self.facts_from_condition(inst.operands[0], true, &mut ft, 0);
                    let mut ff = facts.to_vec();
                    self.facts_from_condition(inst.operands[0], false, &mut ff, 0);
                    let a = self.eval(inst.operands[1], &ft, depth + 1, ctx);
                    let b = self.eval(inst.operands[2], &ff, depth + 1, ctx);
                    base = a.join(&b).meet(&global);
                }
                op::PHI => {
                    let mut out: Option<Abs> = None;
                    let mut k = 0;
                    let block_label = self.f.blocks[self.inst_block[self.def[&id]]].label;
                    while k + 1 < inst.operands.len() {
                        let mut edge = facts.to_vec();
                        if let Some(&pred) = self.f.label_to_block.get(&inst.operands[k + 1]) {
                            self.edge_facts(pred, block_label, &mut edge);
                        }
                        let val = self.eval(inst.operands[k], &edge, depth + 1, ctx);
                        out = Some(out.map_or(val.clone(), |o| o.join(&val)));
                        k += 2;
                    }
                    if let Some(o) = out {
                        base = o.meet(&global);
                    }
                }
                op::I_SUB if inst.operands.len() >= 2 => {
                    let a_id = self.resolve(inst.operands[0]);
                    let b_id = self.resolve(inst.operands[1]);
                    let a = self.eval(a_id, facts, depth + 1, ctx);
                    let b = self.eval(b_id, facts, depth + 1, ctx);
                    // Relational rule: a >= b makes a - b non-negative and bounded by
                    // hi(a) - lo(b), so the subtraction cannot wrap. Only trusted when the
                    // comparison's signedness matches the type, or both operands are known
                    // non-negative (where signed and unsigned orderings agree).
                    let mut rel_facts: Vec<(i128, bool)> = Vec::new(); // (min diff, compare signedness)
                    for f in facts {
                        let rel = if self.same_value(f.a, a_id) && self.same_value(f.b, b_id) {
                            f.rel
                        } else if self.same_value(f.a, b_id) && self.same_value(f.b, a_id) {
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
                _ => {
                    let mut get = |o: u32| Some(self.eval(o, facts, depth + 1, ctx));
                    if let Some(v) = self.transfer(inst, &mut get) {
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
        for f in facts {
            let (other, rel) = if self.same_value(f.a, id) {
                (f.b, f.rel)
            } else if self.same_value(f.b, id) {
                (f.a, f.rel.flip())
            } else {
                continue;
            };
            let other = self.resolve(other);
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
    // Access checks
    // ------------------------------------------------------------------

    fn location_for(&self, inst_idx: usize) -> Option<SourceLocation> {
        let mut block = Some(self.inst_block[inst_idx]);
        let mut upper = inst_idx;
        while let Some(bi) = block {
            let b = &self.f.blocks[bi];
            let mut k = upper;
            while k > b.start {
                k -= 1;
                let inst = &self.f.insts[k];
                match inst.opcode {
                    op::LINE if inst.operands.len() >= 3 => {
                        return Some(SourceLocation {
                            file: self.m.strings.get(&inst.operands[0]).cloned().unwrap_or_default(),
                            line: inst.operands[1],
                            column: inst.operands[2],
                        });
                    }
                    op::NO_LINE => return None,
                    op::EXT_INST if Some(inst.operands[0]) == self.m.debug_ext => match inst.operands.get(1) {
                        Some(&op::DEBUG_LINE) if inst.operands.len() >= 7 => {
                            let src = inst.operands[2];
                            let file = self
                                .m
                                .debug_sources
                                .get(&src)
                                .and_then(|s| self.m.strings.get(s))
                                .cloned()
                                .unwrap_or_default();
                            let line = self.m.int_consts.get(&inst.operands[3]).copied().unwrap_or(0);
                            let col = self.m.int_consts.get(&inst.operands[5]).copied().unwrap_or(0);
                            return Some(SourceLocation {
                                file,
                                line: line as u32,
                                column: col as u32,
                            });
                        }
                        Some(&op::DEBUG_NO_LINE) => return None,
                        _ => {}
                    },
                    _ => {}
                }
            }
            block = self.idom[bi];
            if let Some(nb) = block {
                upper = self.f.blocks[nb].end;
            }
        }
        None
    }

    /// Values the index expression ultimately depends on that the analysis treats as (nearly)
    /// unknown, spelled for a shader author. Bounded backwards walk over the SSA graph.
    fn unknown_sources(&self, idx: u32) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut seen: HashSet<u32> = HashSet::new();
        let mut work = vec![self.resolve(idx)];
        let mut budget = 256;
        while let Some(id) = work.pop() {
            budget -= 1;
            if budget == 0 || out.len() >= 4 {
                break;
            }
            let id = self.resolve(id);
            if !seen.insert(id) || self.constant_abs(id).is_some() || self.m.const_types.contains_key(&id) {
                continue;
            }
            let push = |s: String, out: &mut Vec<String>| {
                if !out.contains(&s) {
                    out.push(s);
                }
            };
            if let Some(phi) = self.phis.get(&id) {
                if self.is_widened(id) {
                    push("a loop-carried value the analysis could not bound".into(), &mut out);
                }
                work.extend(phi.incoming.iter().map(|&(_, v)| v));
                continue;
            }
            if self.undefs.contains(&id) {
                push("an undefined value".into(), &mut out);
                continue;
            }
            let Some(inst) = self.inst_of(id) else {
                if self.m.global_vars.contains_key(&id) {
                    push(self.describe_global(id), &mut out);
                } else {
                    push("a value from another scope".into(), &mut out);
                }
                continue;
            };
            match inst.opcode {
                op::FUNCTION_PARAMETER => push("a function parameter".into(), &mut out),
                op::FUNCTION_CALL => {
                    let callee = inst
                        .operands
                        .first()
                        .and_then(|c| self.m.names.get(c))
                        .map_or("a function".to_string(), |n| format!("`{n}`"));
                    push(format!("the result of calling {callee} (not inlined)"), &mut out);
                }
                op::CONVERT_F_TO_S | op::CONVERT_F_TO_U => {
                    push("a float-to-int conversion".into(), &mut out);
                }
                op::LOAD => {
                    let Some(&ptr) = inst.operands.first() else { continue };
                    match self.pointer_base(ptr) {
                        Some(base) if self.m.global_vars.contains_key(&base) => {
                            push(self.describe_global(base), &mut out);
                        }
                        Some(local) => {
                            let name = self.m.names.get(&local).map(|n| format!(" `{n}`")).unwrap_or_default();
                            push(format!("an untracked local variable{name}"), &mut out);
                        }
                        None => push("an untracked memory load".into(), &mut out),
                    }
                }
                op::PHI => {
                    if self.is_widened(id) {
                        push("a loop-carried value the analysis could not bound".into(), &mut out);
                    }
                    let mut k = 0;
                    while k + 1 < inst.operands.len() {
                        work.push(inst.operands[k]);
                        k += 2;
                    }
                }
                op::SELECT => work.extend(inst.operands.iter().skip(1).copied()),
                // Trailing operands are literal component indices, not ids.
                op::COMPOSITE_EXTRACT => work.extend(inst.operands.first().copied()),
                op::COMPOSITE_INSERT | op::VECTOR_SHUFFLE => work.extend(inst.operands.iter().take(2).copied()),
                op::EXT_INST => {
                    if Some(inst.operands[0]) == self.m.glsl_ext {
                        work.extend(inst.operands.iter().skip(2).copied());
                    } else {
                        push("an extended-instruction result".into(), &mut out);
                    }
                }
                _ if (op::GROUP_NON_UNIFORM_FIRST..=op::GROUP_NON_UNIFORM_LAST).contains(&inst.opcode) => {
                    push("a wave intrinsic result".into(), &mut out);
                }
                _ => work.extend(inst.operands.iter().copied()),
            }
        }
        out
    }

    /// A phi that was widened and that narrowing did not pull back from the type bound.
    fn is_widened(&self, phi: u32) -> bool {
        if !self.widened.contains(&phi) {
            return false;
        }
        match (self.global(phi).as_scalar(), self.int_shape_of(phi)) {
            (Some(r), Some(s)) => {
                let t = s.ty.range();
                r.hi == t.hi || (s.ty.signed && r.lo == t.lo)
            }
            _ => false,
        }
    }

    fn describe_global(&self, var: u32) -> String {
        if let Some(&b) = self.m.builtins.get(&var) {
            return match b {
                // Slang materializes `SV_VertexID` as `VertexIndex - BaseVertex` (likewise for
                // instances), so both halves are reported under the HLSL name.
                op::BUILTIN_VERTEX_INDEX | op::BUILTIN_BASE_VERTEX => "SV_VertexID".into(),
                op::BUILTIN_INSTANCE_INDEX | op::BUILTIN_BASE_INSTANCE => "SV_InstanceID".into(),
                op::BUILTIN_DRAW_INDEX => "the draw index".into(),
                op::BUILTIN_PRIMITIVE_ID => "SV_PrimitiveID".into(),
                op::BUILTIN_GLOBAL_INVOCATION_ID => "SV_DispatchThreadID".into(),
                op::BUILTIN_WORKGROUP_ID => "SV_GroupID".into(),
                op::BUILTIN_NUM_WORKGROUPS => "the dispatch size".into(),
                op::BUILTIN_LOCAL_INVOCATION_ID => "SV_GroupThreadID".into(),
                op::BUILTIN_LOCAL_INVOCATION_INDEX => "SV_GroupIndex".into(),
                op::BUILTIN_SUBGROUP_SIZE => "WaveGetLaneCount()".into(),
                op::BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID => "WaveGetLaneIndex()".into(),
                op::BUILTIN_SUBGROUP_ID | op::BUILTIN_NUM_SUBGROUPS => "the wave index/count".into(),
                other => format!("built-in #{other}"),
            };
        }
        let name = self.m.names.get(&var).map(|n| format!(" `{n}`")).unwrap_or_default();
        match self.m.global_vars.get(&var).map(|&(sc, _)| sc) {
            Some(op::SC_STORAGE_BUFFER) | Some(op::SC_PHYSICAL_STORAGE_BUFFER) => format!("a buffer load{name}"),
            Some(op::SC_UNIFORM) | Some(op::SC_PUSH_CONSTANT) | Some(op::SC_UNIFORM_CONSTANT) => {
                format!("a uniform{name}")
            }
            Some(op::SC_WORKGROUP) => format!("groupshared memory{name}"),
            Some(op::SC_INPUT) => format!("a stage input{name}"),
            Some(op::SC_PRIVATE) => format!("a global{name}"),
            _ => format!("a global{name}"),
        }
    }

    /// Source-level name of the indexed aggregate, or `<unnamed T>` when no name survived
    /// lowering (Slang folds `static const` tables into anonymous temporaries).
    fn array_name(&self, base: u32, path: &[String], fallback: &str) -> String {
        let base = self.resolve(base);
        let mut name = self
            .pointer_base(base)
            .and_then(|b| self.m.names.get(&b).cloned())
            .unwrap_or_else(|| format!("<unnamed {fallback}>"));
        for p in path {
            name.push('.');
            name.push_str(p);
        }
        name
    }

    fn check_accesses(&mut self, report: &mut BoundsReport) {
        // (inst idx, index id, len, base, path, fallback description of the indexed aggregate)
        let mut checks: Vec<(usize, u32, u64, u32, Vec<String>, String)> = Vec::new();
        for &bi in &self.rpo {
            let b = &self.f.blocks[bi];
            for k in b.start..b.end {
                let inst = &self.f.insts[k];
                let skip_first = match inst.opcode {
                    op::ACCESS_CHAIN | op::IN_BOUNDS_ACCESS_CHAIN => 0,
                    op::PTR_ACCESS_CHAIN | op::IN_BOUNDS_PTR_ACCESS_CHAIN => 1,
                    _ => continue,
                };
                let Some(&base) = inst.operands.first() else { continue };
                let Some(base_ty) = self.type_of(self.resolve(base)) else {
                    continue;
                };
                let Some((_, mut cur)) = self.m.pointee(base_ty) else {
                    continue;
                };
                let mut path: Vec<String> = Vec::new();
                for &idx in inst.operands.iter().skip(1 + skip_first) {
                    let Some(ty) = self.m.types.get(&cur) else { break };
                    match ty {
                        Ty::Array { elem, length } => {
                            if let Some(len) = length {
                                if !self.m.int_consts.contains_key(&self.resolve(idx)) {
                                    let desc = format!("{} array", self.m.type_name(*elem));
                                    checks.push((k, idx, *len, base, path.clone(), desc));
                                }
                            }
                            cur = *elem;
                        }
                        Ty::RuntimeArray { elem } => cur = *elem,
                        Ty::Vector { elem, count } => {
                            if !self.m.int_consts.contains_key(&self.resolve(idx)) {
                                checks.push((k, idx, u64::from(*count), base, path.clone(), self.m.type_name(cur)));
                            }
                            cur = *elem;
                        }
                        Ty::Matrix { column, count } => {
                            if !self.m.int_consts.contains_key(&self.resolve(idx)) {
                                checks.push((k, idx, u64::from(*count), base, path.clone(), self.m.type_name(cur)));
                            }
                            cur = *column;
                        }
                        Ty::Struct { members } => {
                            let Some(&m) = self.m.int_consts.get(&self.resolve(idx)) else {
                                break;
                            };
                            let Ok(mi) = usize::try_from(m) else { break };
                            let Some(&member_ty) = members.get(mi) else { break };
                            path.push(
                                self.m
                                    .member_names
                                    .get(&(cur, mi as u32))
                                    .cloned()
                                    .unwrap_or_else(|| format!("[{mi}]")),
                            );
                            cur = member_ty;
                        }
                        _ => break,
                    }
                }
            }
        }

        for (k, idx, len, base, path, desc) in checks {
            report.checked_accesses += 1;
            let block = self.inst_block[k];
            let facts = self.facts_for_block(block);
            let mut ctx = EvalCtx::default();
            let value = self.eval(idx, &facts, 0, &mut ctx);
            let shape = self.int_shape_of(self.resolve(idx));
            let valid = Interval::new(0, i128::from(len) - 1);
            let range = value.as_scalar();
            let is_top = match (range, shape) {
                (Some(r), Some(s)) => r == s.ty.range(),
                _ => true,
            };
            match range {
                Some(r) if valid.contains(r) => report.proven_safe += 1,
                _ => report.diagnostics.push(BoundsDiagnostic {
                    function: self.name.clone(),
                    array: self.array_name(base, &path, &desc),
                    array_length: len,
                    index_range: if is_top { None } else { range.map(|r| (r.lo, r.hi)) },
                    location: self.location_for(k),
                    depends_on: self.unknown_sources(idx),
                }),
            }
        }
    }
}

// ============================================================================
// Dominators (Cooper, Harvey & Kennedy)
// ============================================================================

fn dominators(f: &Function) -> (Vec<Option<usize>>, Vec<usize>) {
    let n = f.blocks.len();
    let mut idom: Vec<Option<usize>> = vec![None; n];
    if n == 0 {
        return (idom, Vec::new());
    }
    // Reverse postorder via iterative DFS from the entry block.
    let mut post: Vec<usize> = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    visited[0] = true;
    while let Some(top) = stack.last_mut() {
        let b = top.0;
        let succs = &f.blocks[b].succs;
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
            for &p in &f.blocks[b].preds {
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

/// While the dominator fixpoint runs, `idom[entry] == Some(entry)` and every processed
/// block's chain ends at the entry (RPO index 0), so both inner loops terminate.
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

fn dominance_frontiers(f: &Function, idom: &[Option<usize>]) -> Vec<Vec<usize>> {
    let n = f.blocks.len();
    let mut df: Vec<BTreeMap<usize, ()>> = vec![BTreeMap::new(); n];
    for b in 0..n {
        if f.blocks[b].preds.len() < 2 || (b != 0 && idom[b].is_none()) {
            continue;
        }
        for &p in &f.blocks[b].preds {
            if p != 0 && idom[p].is_none() {
                continue; // unreachable predecessor
            }
            let mut runner = p;
            while Some(runner) != idom[b] {
                df[runner].insert(b, ());
                match idom[runner] {
                    Some(next) => runner = next,
                    None => break,
                }
            }
        }
    }
    df.into_iter().map(|m| m.into_keys().collect()).collect()
}

#[cfg(test)]
mod tests;
