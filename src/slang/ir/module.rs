//! In-memory model of a serialized Slang IR module.
//!
//! Slang writes a module's IR as a flat table (`FlatInstTable` in
//! `source/slang/slang-serialize-ir.cpp`): every instruction in pre-order with its opcode
//! (as a stable name), operand count, child count, source location, operand indices, and
//! literal payloads. This module rebuilds the instruction tree and offers the typed queries
//! the analysis needs (decorations, name hints, integer shapes, array lengths).

use std::collections::HashMap;
use std::fmt::Write as _;

use super::fossil::{Fossil, Kind};
use super::source_loc::DebugInfo;
use super::stable_names::opcode_name;
use super::{IrError, SourceLocation};

/// Slang IR opcodes the analysis interprets, by stable name
/// (`source/slang/slang-ir-insts-stable-names.lua`).
pub(crate) mod op {
    // Types
    pub const TYPE_VOID: u32 = 2;
    pub const TYPE_BOOL: u32 = 3;
    pub const TYPE_INT8: u32 = 4;
    pub const TYPE_INT16: u32 = 5;
    pub const TYPE_INT: u32 = 6;
    pub const TYPE_INT64: u32 = 7;
    pub const TYPE_UINT8: u32 = 8;
    pub const TYPE_UINT16: u32 = 9;
    pub const TYPE_UINT: u32 = 10;
    pub const TYPE_UINT64: u32 = 11;
    pub const TYPE_HALF: u32 = 12;
    pub const TYPE_FLOAT: u32 = 13;
    pub const TYPE_DOUBLE: u32 = 14;
    pub const TYPE_INTPTR: u32 = 16;
    pub const TYPE_UINTPTR: u32 = 17;
    pub const TYPE_ARRAY: u32 = 27;
    pub const TYPE_UNSIZED_ARRAY: u32 = 28;
    pub const TYPE_VEC: u32 = 31;
    pub const TYPE_MAT: u32 = 32;
    pub const TYPE_ATTRIBUTED: u32 = 34;
    pub const TYPE_RATE_GROUP_SHARED: u32 = 50;
    pub const TYPE_RATE_QUALIFIED: u32 = 52;
    pub const TYPE_PTR: u32 = 57;
    pub const TYPE_REF_PARAM: u32 = 58;
    pub const TYPE_BORROW_IN_PARAM: u32 = 59;
    pub const TYPE_PSEUDO_PTR: u32 = 60;
    pub const TYPE_OUT_PARAM: u32 = 61;
    pub const TYPE_BORROW_IN_OUT_PARAM: u32 = 62;
    pub const TYPE_STRUCT: u32 = 117;
    // Global values and structure
    pub const FUNC: u32 = 132;
    pub const GENERIC: u32 = 133;
    pub const GLOBAL_VAR: u32 = 134;
    pub const GLOBAL_PARAM: u32 = 135;
    pub const MODULE_INST: u32 = 144;
    pub const BLOCK: u32 = 145;
    pub const BOOL_LIT: u32 = 146;
    pub const INT_LIT: u32 = 147;
    pub const FLOAT_LIT: u32 = 148;
    pub const PTR_LIT: u32 = 149;
    pub const STRING_LIT: u32 = 151;
    pub const BLOB_LIT: u32 = 152;
    pub const WITNESS_TABLE: u32 = 139;
    pub const SPECIALIZE: u32 = 166;
    pub const LOOKUP_WITNESS: u32 = 167;
    pub const MAKE_VECTOR: u32 = 173;
    pub const CALL: u32 = 205;
    pub const WITNESS_TABLE_ENTRY: u32 = 214;
    pub const PARAM: u32 = 218;
    pub const FIELD: u32 = 219;
    pub const VAR: u32 = 220;
    pub const LOAD: u32 = 221;
    pub const STORE: u32 = 222;
    pub const GET_FIELD: u32 = 241;
    pub const GET_FIELD_ADDR: u32 = 242;
    pub const GET_ELEMENT: u32 = 243;
    pub const GET_ELEMENT_PTR: u32 = 244;
    pub const RW_STRUCTURED_BUFFER_GET_ELEMENT_PTR: u32 = 264;
    pub const SWIZZLE: u32 = 277;
    // Terminators
    pub const RETURN_VAL: u32 = 280;
    pub const UNCONDITIONAL_BRANCH: u32 = 282;
    pub const LOOP: u32 = 283;
    pub const CONDITIONAL_BRANCH: u32 = 284;
    pub const IF_ELSE: u32 = 285;
    pub const SWITCH: u32 = 288;
    // Arithmetic and logic
    pub const ADD: u32 = 302;
    pub const SUB: u32 = 303;
    pub const MUL: u32 = 304;
    pub const DIV: u32 = 305;
    pub const IREM: u32 = 306;
    pub const SHL: u32 = 308;
    pub const SHR: u32 = 309;
    pub const CMP_EQ: u32 = 310;
    pub const CMP_NE: u32 = 311;
    pub const CMP_GT: u32 = 312;
    pub const CMP_LT: u32 = 313;
    pub const CMP_GE: u32 = 314;
    pub const CMP_LE: u32 = 315;
    pub const AND: u32 = 316;
    pub const XOR: u32 = 317;
    pub const OR: u32 = 318;
    pub const LOGICAL_AND: u32 = 319;
    pub const LOGICAL_OR: u32 = 320;
    pub const NEG: u32 = 321;
    pub const NOT: u32 = 322;
    pub const BIT_NOT: u32 = 323;
    pub const SELECT: u32 = 324;
    pub const BIT_CAST: u32 = 556;
    pub const INT_CAST: u32 = 561;
    pub const CAST_FLOAT_TO_INT: u32 = 564;
    // Debug info
    pub const DEBUG_VAR: u32 = 640;
    // Decorations
    pub const DECORATION_NAME_HINT: u32 = 375;
    pub const DECORATION_NUM_THREADS: u32 = 411;
    pub const DECORATION_ENTRY_POINT: u32 = 422;
    pub const DECORATION_IMPORT: u32 = 442;
    pub const DECORATION_EXPORT: u32 = 443;
    pub const DECORATION_SEMANTIC: u32 = 492;
    pub const DECORATION_TARGET_INTRINSIC: u32 = 370;

    /// Fold Slang's `constexpr*` opcode variants (used for compile-time expressions such as
    /// array lengths over generic arguments, `uint a[1 << LG_N]`) onto the ordinary
    /// arithmetic opcodes they mirror, so the analysis evaluates them the same way.
    pub fn normalize(opcode: u32) -> u32 {
        let Some(suffix) = super::opcode_name(opcode).and_then(|n| n.strip_prefix("constexpr")) else {
            return opcode;
        };
        match suffix {
            "Add" => ADD,
            "Sub" => SUB,
            "Mul" => MUL,
            "Div" => DIV,
            "IRem" => IREM,
            "Neg" => NEG,
            "Shl" => SHL,
            "Shr" => SHR,
            "BitAnd" => AND,
            "BitOr" => OR,
            "BitXor" => XOR,
            "BitNot" => BIT_NOT,
            "Not" => NOT,
            "And" => LOGICAL_AND,
            "Or" => LOGICAL_OR,
            "Eql" => CMP_EQ,
            "Neq" => CMP_NE,
            "Greater" => CMP_GT,
            "Less" => CMP_LT,
            "Geq" => CMP_GE,
            "Leq" => CMP_LE,
            "Select" => SELECT,
            "IntCast" => INT_CAST,
            "CastFloatToInt" => CAST_FLOAT_TO_INT,
            _ => opcode,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Payload {
    None,
    Int(i64),
    Float(f64),
    Str(String),
}

#[derive(Debug, Clone)]
pub(crate) struct Inst {
    /// Opcode stable name.
    pub op: u32,
    /// Instruction index of the result type (`None` for untyped instructions such as decorations).
    pub ty: Option<u32>,
    /// Operand instruction indices; `None` is a null operand.
    pub operands: Vec<Option<u32>>,
    pub parent: Option<u32>,
    /// Decorations first, then ordinary children, in module order.
    pub children: Vec<u32>,
    /// Serial source location (0 = none).
    pub loc: u32,
    pub payload: Payload,
    pub is_decoration: bool,
}

impl Inst {
    pub fn operand(&self, i: usize) -> Option<u32> {
        self.operands.get(i).copied().flatten()
    }
}

/// Integer type: bit width and signedness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct IntTy {
    pub bits: u32,
    pub signed: bool,
}

impl IntTy {
    pub fn range(self) -> (i128, i128) {
        if self.signed {
            (-(1i128 << (self.bits - 1)), (1i128 << (self.bits - 1)) - 1)
        } else {
            (0, (1i128 << self.bits) - 1)
        }
    }
    pub fn with_signed(self, signed: bool) -> IntTy {
        IntTy { signed, ..self }
    }
}

/// Integer scalar or vector shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct IntShape {
    pub ty: IntTy,
    pub lanes: usize,
}

/// A parsed Slang IR module plus its debug information — or several modules linked into one
/// instruction space by [`Module::link`].
pub(crate) struct Module {
    pub name: String,
    pub insts: Vec<Inst>,
    /// Debug information per source module: `(first instruction id, debug info)`, ascending.
    debug: Vec<(u32, Option<DebugInfo>)>,
    /// Instructions below this id belong to the translation unit; the rest were linked in
    /// from library modules.
    pub tu_end: u32,
}

impl Module {
    /// Parse a `.slang-module` container. All modules in the container are read; Slang
    /// puts one module per translation unit there.
    pub fn parse_container(bytes: &[u8]) -> Result<Vec<Module>, IrError> {
        let root = super::riff::parse(bytes)?;
        if &root.kind != b"SLmc" {
            return Err(IrError::Malformed("not a Slang module container"));
        }
        let debug_chunk = root.find_list(b"Sdeb");
        let module_list = root
            .find_list(b"SLml")
            .ok_or(IrError::Malformed("container without module list"))?;
        let mut out = Vec::new();
        for m in module_list.lists(b"smod") {
            let Some(ir) = m.lists(b"ir  ").next().and_then(|l| l.data_child(b"ir  ")) else {
                continue; // AST-only module
            };
            let mut module = Module::parse_ir(ir.data)?;
            // The debug chunk is shared by every module in the container.
            module.debug = vec![(0, debug_chunk.map(DebugInfo::parse).transpose()?)];
            out.push(module);
        }
        Ok(out)
    }

    /// Link modules into one instruction space. The first module is the translation unit.
    ///
    /// A Slang module refers to another module's declarations through `import`-decorated
    /// stubs (a `func` without blocks, a `witness_table` without entries, a struct `key`, an
    /// `interface`, ...) that carry the definition's mangled name. Linking redirects every
    /// operand and type reference from such a stub to the `export`-decorated definition when
    /// one of the modules provides it, so calls, generic specializations, witness lookups
    /// and struct field keys resolve across module boundaries exactly as they do inside one
    /// module. Stubs whose definition is not available stay as they are.
    pub fn link(modules: Vec<Module>) -> Module {
        let mut linked = Module {
            name: modules.first().map(|m| m.name.clone()).unwrap_or_default(),
            insts: Vec::with_capacity(modules.iter().map(|m| m.insts.len()).sum()),
            debug: Vec::with_capacity(modules.len()),
            tu_end: modules.first().map_or(0, |m| m.insts.len() as u32),
        };
        for module in modules {
            let base = linked.insts.len() as u32;
            for (offset, debug) in module.debug {
                linked.debug.push((base + offset, debug));
            }
            let shift = |r: &mut Option<u32>| {
                if let Some(x) = r {
                    *x += base;
                }
            };
            for mut inst in module.insts {
                shift(&mut inst.ty);
                inst.operands.iter_mut().for_each(shift);
                shift(&mut inst.parent);
                inst.children.iter_mut().for_each(|c| *c += base);
                linked.insts.push(inst);
            }
        }

        let n = linked.insts.len() as u32;
        let mut exports: HashMap<&str, u32> = HashMap::new();
        for i in 0..n {
            if let Some(name) = linked.export_name(i) {
                if linked.is_definition(i) {
                    exports.entry(name).or_insert(i);
                }
            }
        }
        let alias: Vec<u32> = (0..n)
            .map(|i| {
                linked
                    .import_name(i)
                    .and_then(|name| exports.get(name).copied())
                    .filter(|&d| d != i)
                    .unwrap_or(i)
            })
            .collect();
        for inst in &mut linked.insts {
            if let Some(t) = &mut inst.ty {
                *t = alias[*t as usize];
            }
            for o in inst.operands.iter_mut().flatten() {
                *o = alias[*o as usize];
            }
        }
        linked
    }

    /// Whether an exported instruction is a definition worth linking to (a function needs a
    /// body; everything else — types, keys, witness tables, generics — is its own definition).
    fn is_definition(&self, i: u32) -> bool {
        match self.inst(i).op {
            op::FUNC => self.body(i).any(|c| self.inst(c).op == op::BLOCK),
            op::GENERIC => {
                let inner = self
                    .body(i)
                    .find(|&c| self.inst(c).op == op::BLOCK)
                    .and_then(|b| self.body(b).last())
                    .and_then(|t| self.inst(t).operand(0));
                inner.is_none_or(|f| self.inst(f).op != op::FUNC || self.is_definition(f))
            }
            _ => true,
        }
    }

    /// Parse one fossilized `IRModuleInfo` blob.
    fn parse_ir(bytes: &[u8]) -> Result<Module, IrError> {
        let fossil = Fossil::new(bytes)?;
        let (layout, root_off) = fossil.root()?;
        let info = fossil.val(root_off, &layout);
        // IRModuleInfo { serializationVersion, fullVersion, module }
        if info.kind() != Kind::Struct || info.field_count() < 3 {
            return Err(IrError::Malformed("IRModuleInfo layout"));
        }
        let module = info
            .field(2)?
            .deref()?
            .ok_or(IrError::Malformed("IRModuleInfo without module"))?;
        // IRModule { m_name, m_version, m_moduleInst: FlatInstTable }
        if module.kind() != Kind::Struct || module.field_count() < 3 {
            return Err(IrError::Malformed("IRModule layout"));
        }
        let name = String::from_utf8_lossy(module.field(0)?.deref()?.map_or(Ok(&[][..]), |v| v.string())?).into_owned();
        let flat = module.field(2)?;
        if flat.kind() != Kind::Struct || flat.field_count() < 7 {
            return Err(IrError::Malformed("FlatInstTable layout"));
        }
        let array_field = |i: usize| -> Result<_, IrError> {
            let f = flat.field(i)?;
            match f.kind() {
                Kind::Ptr => f
                    .deref()?
                    .ok_or(IrError::Malformed("FlatInstTable array missing"))?
                    .array(),
                Kind::ArrayObj => f.array(),
                _ => Err(IrError::Malformed("FlatInstTable field is not an array")),
            }
        };
        let alloc = array_field(0)?;
        let child_counts = array_field(1)?;
        let source_locs = array_field(2)?;
        let operand_indices = array_field(3)?;
        let string_lengths = array_field(4)?;
        let string_chars = array_field(5)?.bytes()?;
        let literals = array_field(6)?;

        let n = alloc.len();
        if child_counts.len() != n || source_locs.len() != n {
            return Err(IrError::Malformed("FlatInstTable table lengths disagree"));
        }
        let mut insts: Vec<Inst> = Vec::with_capacity(n);
        for i in 0..n {
            let a = alloc.get(i)?;
            let op = op::normalize(a.field(0)?.u32()?);
            let operand_count = a.field(1)?.u32()? as usize;
            let loc = match source_locs.get(i)?.deref()? {
                Some(v) => v.u32()?,
                None => 0,
            };
            let name = opcode_name(op);
            insts.push(Inst {
                op,
                ty: None,
                operands: vec![None; operand_count],
                parent: None,
                children: Vec::with_capacity(child_counts.get(i)?.i64()?.max(0) as usize),
                loc,
                payload: Payload::None,
                is_decoration: name.is_some_and(|n| n.starts_with("Decoration.")),
            });
        }

        // Operands, literals and strings are consumed in instruction order.
        let mut operand_at = 0usize;
        let mut lit_at = 0usize;
        let mut str_at = 0usize;
        let mut chars_at = 0usize;
        let read_ref = |operand_at: &mut usize| -> Result<Option<u32>, IrError> {
            let v = operand_indices.get(*operand_at)?.i64()?;
            *operand_at += 1;
            match v {
                -1 => Ok(None),
                x if x >= 0 && (x as usize) < n => Ok(Some(x as u32)),
                _ => Err(IrError::Malformed("operand index out of range")),
            }
        };
        for inst in insts.iter_mut().take(n) {
            let ty = read_ref(&mut operand_at)?;
            let count = inst.operands.len();
            let mut operands = Vec::with_capacity(count);
            for _ in 0..count {
                operands.push(read_ref(&mut operand_at)?);
            }
            let payload = match inst.op {
                op::BOOL_LIT | op::INT_LIT => {
                    let bits = literals.get(lit_at)?.u64()?;
                    lit_at += 1;
                    Payload::Int(bits as i64)
                }
                op::FLOAT_LIT => {
                    let bits = literals.get(lit_at)?.u64()?;
                    lit_at += 1;
                    Payload::Float(f64::from_bits(bits))
                }
                op::PTR_LIT => {
                    lit_at += 1;
                    Payload::None
                }
                op::STRING_LIT | op::BLOB_LIT => {
                    let len = string_lengths.get(str_at)?.i64()?.max(0) as usize;
                    str_at += 1;
                    let s = string_chars
                        .get(chars_at..chars_at + len)
                        .ok_or(IrError::Malformed("string literal exceeds table"))?;
                    chars_at += len;
                    Payload::Str(String::from_utf8_lossy(s).into_owned())
                }
                _ => Payload::None,
            };
            inst.ty = ty;
            inst.operands = operands;
            inst.payload = payload;
        }

        // Pre-order tree: each instruction is followed by its children.
        let mut stack: Vec<(u32, usize)> = Vec::new(); // (inst, remaining children)
        for i in 0..n {
            let remaining = child_counts.get(i)?.i64()?.max(0) as usize;
            while let Some(&(_, 0)) = stack.last() {
                stack.pop();
            }
            if let Some(top) = stack.last_mut() {
                top.1 -= 1;
                let parent = top.0;
                insts[i].parent = Some(parent);
                insts[parent as usize].children.push(i as u32);
            } else if i != 0 {
                return Err(IrError::Malformed("instruction tree has several roots"));
            }
            stack.push((i as u32, remaining));
        }

        Ok(Module {
            name,
            tu_end: insts.len() as u32,
            insts,
            debug: Vec::new(),
        })
    }

    // ------------------------------------------------------------------
    // Structural queries
    // ------------------------------------------------------------------

    pub fn inst(&self, i: u32) -> &Inst {
        &self.insts[i as usize]
    }

    pub fn op_name(&self, i: u32) -> String {
        let op = self.inst(i).op;
        opcode_name(op).map_or_else(|| format!("op#{op}"), |n| n.rsplit('.').next().unwrap_or(n).to_string())
    }

    pub fn decorations(&self, i: u32) -> impl Iterator<Item = &Inst> + '_ {
        self.inst(i)
            .children
            .iter()
            .map(|&c| self.inst(c))
            .filter(|c| c.is_decoration)
    }

    pub fn decoration(&self, i: u32, op: u32) -> Option<&Inst> {
        self.decorations(i).find(|d| d.op == op)
    }

    /// Non-decoration children (blocks of a function, params + body of a block, ...).
    pub fn body(&self, i: u32) -> impl Iterator<Item = u32> + '_ {
        self.inst(i)
            .children
            .iter()
            .copied()
            .filter(move |&c| !self.inst(c).is_decoration)
    }

    pub fn int_lit(&self, i: u32) -> Option<i64> {
        let inst = self.inst(i);
        match (inst.op, &inst.payload) {
            (op::INT_LIT, Payload::Int(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn bool_lit(&self, i: u32) -> Option<bool> {
        let inst = self.inst(i);
        match (inst.op, &inst.payload) {
            (op::BOOL_LIT, Payload::Int(v)) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn string_lit(&self, i: u32) -> Option<&str> {
        match &self.inst(i).payload {
            Payload::Str(s) if self.inst(i).op == op::STRING_LIT => Some(s),
            _ => None,
        }
    }

    pub fn name_hint(&self, i: u32) -> Option<&str> {
        self.decoration(i, op::DECORATION_NAME_HINT)
            .and_then(|d| d.operand(0))
            .and_then(|s| self.string_lit(s))
    }

    /// Mangled name from an `import` decoration.
    pub fn import_name(&self, i: u32) -> Option<&str> {
        self.decoration(i, op::DECORATION_IMPORT)
            .and_then(|d| d.operand(0))
            .and_then(|s| self.string_lit(s))
    }

    pub fn export_name(&self, i: u32) -> Option<&str> {
        self.decoration(i, op::DECORATION_EXPORT)
            .and_then(|d| d.operand(0))
            .and_then(|s| self.string_lit(s))
    }

    /// `SV_*` / user semantic name of an entry-point parameter.
    pub fn semantic(&self, i: u32) -> Option<&str> {
        self.decoration(i, op::DECORATION_SEMANTIC)
            .and_then(|d| d.operand(0))
            .and_then(|s| self.string_lit(s))
    }

    /// `[numthreads(x, y, z)]` when all three are literals.
    pub fn num_threads(&self, func: u32) -> Option<[u64; 3]> {
        let d = self.decoration(func, op::DECORATION_NUM_THREADS)?;
        let mut out = [1u64; 3];
        for (k, slot) in out.iter_mut().enumerate() {
            *slot = u64::try_from(self.int_lit(d.operand(k)?)?).ok()?;
        }
        Some(out)
    }

    pub fn is_entry_point(&self, func: u32) -> bool {
        self.decoration(func, op::DECORATION_ENTRY_POINT).is_some()
    }

    /// Function definitions: `func` instructions with at least one block.
    pub fn function_defs(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.insts.len() as u32)
            .filter(move |&i| self.inst(i).op == op::FUNC && self.body(i).any(|c| self.inst(c).op == op::BLOCK))
    }

    /// Whether `i` belongs to the translation unit rather than a linked library module.
    pub fn in_translation_unit(&self, i: u32) -> bool {
        i < self.tu_end
    }

    /// Text dump of the instruction tree (one line per instruction, children indented), for
    /// debugging the analysis against what Slang actually emitted.
    #[cfg(test)]
    pub fn dump(&self) -> String {
        fn depth(m: &Module, mut i: u32) -> usize {
            let mut d = 0;
            while let Some(p) = m.inst(i).parent {
                d += 1;
                i = p;
            }
            d
        }
        let mut out = String::new();
        for i in 0..self.insts.len() as u32 {
            let inst = self.inst(i);
            if inst.is_decoration {
                continue;
            }
            let _ = write!(out, "{:indent$}%{i}", "", indent = depth(self, i) * 2);
            if let Some(n) = self.name_hint(i) {
                let _ = write!(out, " `{n}`");
            }
            let _ = write!(out, " = {}", self.op_name(i));
            match &inst.payload {
                Payload::Int(v) => {
                    let _ = write!(out, " {v}");
                }
                Payload::Float(v) => {
                    let _ = write!(out, " {v}");
                }
                Payload::Str(s) => {
                    let _ = write!(out, " {s:?}");
                }
                Payload::None => {}
            }
            let operands: Vec<String> = inst
                .operands
                .iter()
                .map(|o| o.map_or_else(|| "_".to_string(), |o| format!("%{o}")))
                .collect();
            if !operands.is_empty() {
                let _ = write!(out, "({})", operands.join(", "));
            }
            if let Some(t) = inst.ty {
                let _ = write!(out, " : {}", self.types().type_name(t));
            }
            if let Some(loc) = self.location(i) {
                let _ = write!(out, "  @{loc}");
            }
            out.push('\n');
        }
        out
    }

    /// Source location of an instruction, falling back to the nearest earlier sibling and
    /// then to ancestors, so decorations and hoisted values still get a usable line.
    pub fn location(&self, i: u32) -> Option<SourceLocation> {
        let segment = self.debug.partition_point(|(base, _)| *base <= i).checked_sub(1)?;
        let debug = self.debug[segment].1.as_ref()?;
        let mut cur = Some(i);
        while let Some(c) = cur {
            let inst = self.inst(c);
            if let Some(l) = debug.lookup(inst.loc) {
                return Some(l);
            }
            if let Some(p) = inst.parent {
                let siblings = &self.inst(p).children;
                if let Some(pos) = siblings.iter().position(|&s| s == c) {
                    for &s in siblings[..pos].iter().rev() {
                        if let Some(l) = debug.lookup(self.inst(s).loc) {
                            return Some(l);
                        }
                    }
                }
            }
            cur = inst.parent;
        }
        None
    }

    // ------------------------------------------------------------------
    // Types
    // ------------------------------------------------------------------

    pub fn type_of(&self, i: u32) -> Option<u32> {
        self.inst(i).ty
    }

    /// Type queries without a generic substitution.
    pub fn types(&self) -> Types<'_> {
        Types { m: self, subst: None }
    }

    pub fn is_group_shared(&self, ty: u32) -> bool {
        let t = self.inst(ty);
        t.op == op::TYPE_RATE_QUALIFIED
            && t.operand(0)
                .is_some_and(|r| self.inst(r).op == op::TYPE_RATE_GROUP_SHARED)
    }

    pub fn int_ty(&self, ty: u32) -> Option<IntTy> {
        self.types().int_ty(ty)
    }

    pub fn int_shape(&self, ty: u32) -> Option<IntShape> {
        self.types().int_shape(ty)
    }
}

/// Type queries under a generic substitution: every type reference is first mapped through
/// `subst` (generic parameter -> argument), so the body of `T foo<T, let N : int>(T x[N])`
/// specialized with `<uint, 64>` sees `x` as `uint[64]`.
#[derive(Clone, Copy)]
pub(crate) struct Types<'a> {
    pub m: &'a Module,
    pub subst: Option<&'a HashMap<u32, u32>>,
}

impl<'a> Types<'a> {
    fn r(&self, id: u32) -> u32 {
        self.subst.and_then(|s| s.get(&id).copied()).unwrap_or(id)
    }

    fn inst(&self, id: u32) -> &'a Inst {
        self.m.inst(self.r(id))
    }

    fn int_lit(&self, id: u32) -> Option<i64> {
        self.m.int_lit(self.r(id))
    }

    /// Strip rate qualifiers and attributes from a type.
    pub fn unqualified(&self, mut ty: u32) -> u32 {
        loop {
            ty = self.r(ty);
            let t = self.m.inst(ty);
            match t.op {
                op::TYPE_RATE_QUALIFIED => match t.operand(1) {
                    Some(inner) => ty = inner,
                    None => return ty,
                },
                op::TYPE_ATTRIBUTED => match t.operand(0) {
                    Some(inner) => ty = inner,
                    None => return ty,
                },
                _ => return ty,
            }
        }
    }

    pub fn is_pointer_type(&self, ty: u32) -> bool {
        matches!(
            self.inst(self.unqualified(ty)).op,
            op::TYPE_PTR
                | op::TYPE_REF_PARAM
                | op::TYPE_BORROW_IN_PARAM
                | op::TYPE_PSEUDO_PTR
                | op::TYPE_OUT_PARAM
                | op::TYPE_BORROW_IN_OUT_PARAM
        )
    }

    /// Pointee of a pointer-like type.
    pub fn pointee(&self, ty: u32) -> Option<u32> {
        let ty = self.unqualified(ty);
        self.is_pointer_type(ty)
            .then(|| self.inst(ty).operand(0))
            .flatten()
            .map(|p| self.r(p))
    }

    pub fn int_ty(&self, ty: u32) -> Option<IntTy> {
        Some(match self.inst(self.unqualified(ty)).op {
            op::TYPE_INT8 => IntTy { bits: 8, signed: true },
            op::TYPE_INT16 => IntTy { bits: 16, signed: true },
            op::TYPE_INT => IntTy { bits: 32, signed: true },
            op::TYPE_INT64 | op::TYPE_INTPTR => IntTy { bits: 64, signed: true },
            op::TYPE_UINT8 => IntTy { bits: 8, signed: false },
            op::TYPE_UINT16 => IntTy {
                bits: 16,
                signed: false,
            },
            op::TYPE_UINT => IntTy {
                bits: 32,
                signed: false,
            },
            op::TYPE_UINT64 | op::TYPE_UINTPTR => IntTy {
                bits: 64,
                signed: false,
            },
            _ => return None,
        })
    }

    /// Integer scalar/vector shape of a type.
    pub fn int_shape(&self, ty: u32) -> Option<IntShape> {
        let ty = self.unqualified(ty);
        if let Some(t) = self.int_ty(ty) {
            return Some(IntShape { ty: t, lanes: 1 });
        }
        let inst = self.inst(ty);
        if inst.op == op::TYPE_VEC {
            let elem = self.int_ty(inst.operand(0)?)?;
            let lanes = usize::try_from(self.int_lit(inst.operand(1)?)?).ok()?;
            return Some(IntShape { ty: elem, lanes });
        }
        None
    }

    /// Statically sized indexable type: `(element type, length)` for arrays, vectors and
    /// matrices (rows). `None` for unsized arrays, buffers, unsubstituted generic lengths, ...
    pub fn indexable(&self, ty: u32) -> Option<(u32, u64)> {
        let inst = self.inst(self.unqualified(ty));
        let (elem, count) = match inst.op {
            op::TYPE_ARRAY | op::TYPE_VEC => (inst.operand(0)?, inst.operand(1)?),
            op::TYPE_MAT => {
                // Mat(elem, rows, cols, layout): indexing yields a row vector.
                let elem = self.r(inst.operand(0)?);
                let cols = self.r(inst.operand(2)?);
                let rows = inst.operand(1)?;
                let row_ty = (0..self.m.insts.len() as u32).find(|&i| {
                    let t = self.m.inst(i);
                    t.op == op::TYPE_VEC && t.operand(0) == Some(elem) && t.operand(1) == Some(cols)
                });
                return Some((row_ty.unwrap_or(elem), u64::try_from(self.int_lit(rows)?).ok()?));
            }
            _ => return None,
        };
        Some((self.r(elem), u64::try_from(self.int_lit(count)?).ok()?))
    }

    /// Human-readable type name (Slang spelling where practical).
    pub fn type_name(&self, ty: u32) -> String {
        let mut out = String::new();
        self.write_type_name(ty, &mut out, 0);
        out
    }

    fn write_type_name(&self, ty: u32, out: &mut String, depth: u32) {
        if depth > 8 {
            out.push_str("...");
            return;
        }
        let ty = self.r(ty);
        let inst = self.m.inst(ty);
        let scalar = |o: u32| -> Option<&str> {
            Some(match o {
                op::TYPE_VOID => "void",
                op::TYPE_BOOL => "bool",
                op::TYPE_INT8 => "int8_t",
                op::TYPE_INT16 => "int16_t",
                op::TYPE_INT => "int",
                op::TYPE_INT64 => "int64_t",
                op::TYPE_UINT8 => "uint8_t",
                op::TYPE_UINT16 => "uint16_t",
                op::TYPE_UINT => "uint",
                op::TYPE_UINT64 => "uint64_t",
                op::TYPE_HALF => "half",
                op::TYPE_FLOAT => "float",
                op::TYPE_DOUBLE => "double",
                op::TYPE_INTPTR => "intptr_t",
                op::TYPE_UINTPTR => "uintptr_t",
                _ => return None,
            })
        };
        if let Some(s) = scalar(inst.op) {
            out.push_str(s);
            return;
        }
        let lit = |i: Option<u32>| i.and_then(|i| self.int_lit(i));
        match inst.op {
            op::TYPE_VEC => {
                if let Some(e) = inst.operand(0) {
                    self.write_type_name(e, out, depth + 1);
                }
                if let Some(n) = lit(inst.operand(1)) {
                    let _ = write!(out, "{n}");
                }
            }
            op::TYPE_MAT => {
                if let Some(e) = inst.operand(0) {
                    self.write_type_name(e, out, depth + 1);
                }
                if let (Some(r), Some(c)) = (lit(inst.operand(1)), lit(inst.operand(2))) {
                    let _ = write!(out, "{r}x{c}");
                }
            }
            op::TYPE_ARRAY => {
                if let Some(e) = inst.operand(0) {
                    self.write_type_name(e, out, depth + 1);
                }
                match lit(inst.operand(1)) {
                    Some(n) => {
                        let _ = write!(out, "[{n}]");
                    }
                    None => out.push_str("[N]"),
                }
            }
            op::TYPE_UNSIZED_ARRAY => {
                if let Some(e) = inst.operand(0) {
                    self.write_type_name(e, out, depth + 1);
                }
                out.push_str("[]");
            }
            op::TYPE_STRUCT => match self.m.name_hint(ty) {
                Some(n) => out.push_str(n),
                None => out.push_str("struct"),
            },
            op::TYPE_RATE_QUALIFIED | op::TYPE_ATTRIBUTED => {
                self.write_type_name(self.unqualified(ty), out, depth + 1);
            }
            _ if self.is_pointer_type(ty) => {
                if let Some(p) = self.pointee(ty) {
                    self.write_type_name(p, out, depth + 1);
                }
            }
            _ => match self.m.name_hint(ty) {
                Some(n) => out.push_str(n),
                None => out.push_str(&self.m.op_name(ty)),
            },
        }
    }
}
