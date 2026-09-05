//! Yielding scripts: source-to-source lowering of `[goldy_petition]`,
//! `[goldy_resume]`, and `$yield(...)`.
//!
//! A yielding script is an ordinary `[goldy_compute]` entry point (`cs_main`, the
//! *prologue*) plus one or more `[goldy_resume]` functions (*continuations*). A
//! lane suspends by calling `$yield(continuation, payload, state)` and returning;
//! the runtime services the payload through a handler bound on the host with
//! [`crate::SchemeNodeBuilder::yield_point`] and resumes the lane in the named
//! continuation with the handler's `Resolved<T>` result and the saved state.
//!
//! This module runs *before* the regular virtual-main transform
//! ([`super::virtual_main`]) and only rewrites the parts of the source that
//! those attributes introduce:
//!
//! - `[goldy_petition(Result = BufRO<E>)] struct P { .. }` — attribute stripped,
//!   struct kept; records the payload type and its promised element type `E`.
//! - `$yield(c, payload, state)` — becomes a block that materialises the payload
//!   and state into typed locals (`P { .. }` / `{ .. }` initializers or plain
//!   expressions) and calls a generated `__goldy_yield_<c>` helper, which appends
//!   them to continuation `c`'s mailbox with an atomic bump on a per-dispatch
//!   count buffer.
//! - Every function that yields gains *hidden* trailing parameters carrying the
//!   mailbox buffers (`Scattered<P> __gy_pay_c`, `Scattered<S> __gy_st_c`,
//!   `Scattered<Interlocked<uint>> __gy_cnt`) and a `uint __gy_cap_c` scalar per
//!   target. The prologue also gains `uint __gy_base`, added to `ThreadId.x`, so
//!   the host can launch it in chunks (`Backpressure::Stall`).
//! - `[goldy_resume] void c(program params.., Resolved<E> r, S s, ThreadId tid)`
//!   — attributes stripped; the function stays as the continuation *body*.
//!
//! The lowering produces one translation unit per entry point: [`lower`] with
//! `variant = None` yields the prologue module, and `variant = Some(c)` a module
//! whose `cs_main` is a generated `[goldy_compute]` wrapper that reads record
//! `ThreadId.x` from continuation `c`'s mailbox, builds the `Resolved<E>` view
//! from the resolution table and result arena, and calls the continuation body.
//! Non-selected entries stay in the unit as plain (dead) functions so every
//! variant type-checks the whole script.
//!
//! [`reflect`] returns the host-facing description ([`YieldReflection`]) the
//! scheme uses to validate handlers and size the mailboxes.
//!
//! # v0 restrictions
//!
//! - Payload and state structs may only hold `uint` / `int` / `float` fields,
//!   fixed arrays of those, and nested structs obeying the same rule. Their byte
//!   layout is then identical under every buffer layout rule, which is what lets
//!   the host size the mailboxes without a reflection round-trip.
//! - Program parameters of yielding entry points are buffers (`Scattered<T>`,
//!   `BufRO<T>`, broadcast structs) and scalars. Textures, samplers, and
//!   preprocessor-conditional parameter lists are rejected.
//! - A continuation's program parameters must match prologue parameters by
//!   name and type; that is how the host re-binds them for the resume dispatch.

use std::collections::BTreeSet;
use std::ops::Range;

use super::virtual_main::{
    self, classify_type, find_all_entries, find_attr_group_start, find_matching_close, find_numthreads_in_range,
    find_substr, is_in_line_comment, match_brace, parse_params, scan_bracket_block, scan_identifier,
    skip_whitespace_and_comments, Param, ParamItem, ParamKind, Stage, SvKind,
};

/// Prefix shared by every hidden identifier this pass introduces.
pub(crate) const HIDDEN_PREFIX: &str = "__gy_";

/// Sentinel written by handlers for a rejected petition (matches `GOLDY_RESOLVED_NULL`).
pub(crate) const RESOLVED_NULL: u32 = u32::MAX;

/// Default `[numthreads]` for a continuation that declares none.
const DEFAULT_NUMTHREADS: (u32, u32, u32) = (64, 1, 1);

/// One `[goldy_petition(Result = BufRO<E>)] struct P { .. }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PetitionDecl {
    /// Slang struct name (`P`).
    pub name: String,
    /// Element type of the promised buffer (`E`).
    pub result_elem: String,
    /// Byte size of the payload struct (natural layout; see module docs).
    pub payload_bytes: u32,
    attr_range: Range<usize>,
}

/// One program-visible parameter of a yielding entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramParam {
    pub name: String,
    pub ty: String,
    /// `true` for a `with_param` scalar, `false` for a `with_parcel` buffer slot.
    pub is_scalar: bool,
}

/// One `[goldy_resume]` continuation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuationDecl {
    /// Function name; also the yield point's name on the host.
    pub fn_name: String,
    /// Payload struct this continuation is resumed for.
    pub petition: String,
    /// Promised element type (`Resolved<E>`).
    pub result_elem: String,
    /// Saved-state struct type.
    pub state_ty: String,
    /// Byte size of the state struct.
    pub state_bytes: u32,
    /// Workgroup size of the resume dispatch.
    pub numthreads: (u32, u32, u32),
    /// Program parameters in declaration order (each matches a prologue parameter).
    pub program_params: Vec<ProgramParam>,
    /// Continuations this body yields to (sorted, unique).
    pub yields_to: Vec<String>,
    /// Whether the body declares a `ThreadId` parameter.
    pub has_thread_id: bool,
    attr_range: Range<usize>,
    numthreads_range: Option<Range<usize>>,
    fn_name_range: Range<usize>,
    params_close: usize,
    body: Range<usize>,
    /// Position of the `Resolved<E>` parameter within the user parameter list.
    resolved_index: usize,
    state_index: usize,
    thread_id_index: Option<usize>,
}

/// Host-facing description of a yielding script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YieldReflection {
    /// Prologue program parameters in declaration order.
    pub prologue_params: Vec<ProgramParam>,
    /// Continuations the prologue yields to (sorted, unique).
    pub prologue_yields_to: Vec<String>,
    /// Whether the prologue declares a `ThreadId` parameter (chunked launches need it).
    pub prologue_has_thread_id: bool,
    /// Whether the prologue declares `GroupId` / `GroupThreadId` (chunked launches break them).
    pub prologue_uses_group_ids: bool,
    /// Prologue workgroup size.
    pub prologue_numthreads: (u32, u32, u32),
    /// Continuations in declaration order; the index is the mailbox slot.
    pub continuations: Vec<ContinuationDecl>,
    pub petitions: Vec<PetitionDecl>,
}

impl YieldReflection {
    pub fn continuation(&self, name: &str) -> Option<&ContinuationDecl> {
        self.continuations.iter().find(|c| c.fn_name == name)
    }

    pub fn continuation_index(&self, name: &str) -> Option<usize> {
        self.continuations.iter().position(|c| c.fn_name == name)
    }

    pub fn petition(&self, name: &str) -> Option<&PetitionDecl> {
        self.petitions.iter().find(|p| p.name == name)
    }

    /// Byte size of `ty` under the v0 layout rule, when `ty` is a petition/state-style type.
    pub fn scalar_struct_bytes(&self, source: &str, ty: &str) -> Option<u32> {
        struct_bytes(source, ty, &mut Vec::new()).ok()
    }
}

/// `true` when `source` uses any yielding-script construct.
pub fn has_yield_constructs(source: &str) -> bool {
    source.contains("$yield") || source.contains("[goldy_resume") || source.contains("[goldy_petition")
}

/// Reflect the yielding-script structure of `source`, or `Ok(None)` when it has none.
pub fn reflect(source: &str) -> Result<Option<YieldReflection>, String> {
    if !has_yield_constructs(source) {
        return Ok(None);
    }
    analyze(source).map(|a| Some(a.reflection))
}

/// Lower `source` into the translation unit for `variant` (`None` = prologue,
/// `Some(name)` = continuation `name` as `cs_main`).
///
/// Sources without yielding constructs are returned unchanged.
pub fn lower(source: &str, variant: Option<&str>) -> Result<String, String> {
    if !has_yield_constructs(source) {
        return Ok(source.to_string());
    }
    let analysis = analyze(source)?;
    emit(source, &analysis, variant)
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

struct YieldSite {
    range: Range<usize>,
    target: String,
    payload_expr: String,
    state_expr: String,
}

struct Prologue {
    fn_name: String,
    attr_range: Range<usize>,
    numthreads_range: Option<Range<usize>>,
    fn_name_range: Range<usize>,
    params_close: usize,
    body: Range<usize>,
    thread_id_name: Option<String>,
}

struct Analysis {
    reflection: YieldReflection,
    prologue: Prologue,
    sites: Vec<YieldSite>,
}

fn err<T>(msg: impl Into<String>) -> Result<T, String> {
    Err(format!("yielding script: {}", msg.into()))
}

fn analyze(source: &str) -> Result<Analysis, String> {
    let petitions = find_petitions(source)?;
    let mut continuations = find_continuations(source, &petitions)?;
    let prologue = find_prologue(source)?;
    let sites = find_yield_sites(source)?;

    // Assign yield sites to their enclosing yielding function.
    let mut prologue_targets = BTreeSet::new();
    let mut cont_targets: Vec<BTreeSet<String>> = vec![BTreeSet::new(); continuations.len()];
    for site in &sites {
        if continuations.iter().all(|c| c.fn_name != site.target) {
            return err(format!(
                "$yield targets `{}`, which is not a [goldy_resume] function",
                site.target
            ));
        }
        if prologue.body.contains(&site.range.start) {
            prologue_targets.insert(site.target.clone());
        } else if let Some(idx) = continuations.iter().position(|c| c.body.contains(&site.range.start)) {
            cont_targets[idx].insert(site.target.clone());
        } else {
            return err(format!(
                "$yield(`{}`, ..) must appear directly inside the [goldy_compute] prologue or a [goldy_resume] body",
                site.target
            ));
        }
    }
    for (c, targets) in continuations.iter_mut().zip(cont_targets) {
        c.yields_to = targets.into_iter().collect();
    }

    // Infer petition types for `[goldy_resume]` without an explicit argument.
    for c in continuations.iter_mut() {
        if !c.petition.is_empty() {
            continue;
        }
        let mut inferred: Option<String> = None;
        for site in sites.iter().filter(|s| s.target == c.fn_name) {
            let Some(name) = leading_identifier(&site.payload_expr) else {
                continue;
            };
            if petitions.iter().any(|p| p.name == name) {
                match &inferred {
                    Some(prev) if prev != &name => {
                        return err(format!(
                            "continuation `{}` is yielded to with both `{prev}` and `{name}` payloads",
                            c.fn_name
                        ))
                    }
                    _ => inferred = Some(name),
                }
            }
        }
        let Some(name) = inferred else {
            return err(format!(
                "cannot infer the petition type of continuation `{}`; write `[goldy_resume({{Petition}})]` \
                 or yield with a `Petition {{ .. }}` literal",
                c.fn_name
            ));
        };
        c.petition = name;
    }
    for c in continuations.iter_mut() {
        let Some(p) = petitions.iter().find(|p| p.name == c.petition) else {
            return err(format!(
                "continuation `{}` names petition `{}`, which has no [goldy_petition] declaration",
                c.fn_name, c.petition
            ));
        };
        if c.result_elem != p.result_elem {
            return err(format!(
                "continuation `{}` takes Resolved<{}> but petition `{}` promises BufRO<{}>",
                c.fn_name, c.result_elem, p.name, p.result_elem
            ));
        }
    }

    // Prologue program parameters.
    let entries = find_all_entries(source);
    let entry = entries
        .iter()
        .find(|e| e.stage == Stage::Compute && e.fn_name == prologue.fn_name)
        .ok_or_else(|| "yielding script: prologue entry vanished".to_string())?;
    let mut prologue_params = Vec::new();
    let mut prologue_uses_group_ids = false;
    for item in &entry.params {
        let ParamItem::Single(p) = item else {
            return err("preprocessor-conditional parameter lists are not supported in yielding scripts");
        };
        match &p.kind {
            ParamKind::SystemValue(SvKind::DispatchThreadId) => {}
            ParamKind::SystemValue(SvKind::GroupId | SvKind::GroupThreadId) => prologue_uses_group_ids = true,
            ParamKind::SystemValue(other) => {
                return err(format!(
                    "system value {other:?} is not supported in a yielding prologue"
                ))
            }
            _ => prologue_params.push(program_param(p, &prologue.fn_name)?),
        }
    }
    if prologue_params.iter().any(|p| p.name.starts_with(HIDDEN_PREFIX)) {
        return err(format!("parameter names starting with `{HIDDEN_PREFIX}` are reserved"));
    }

    // Continuation parameters must exist on the prologue with the same type.
    for c in &continuations {
        for p in &c.program_params {
            match prologue_params.iter().find(|q| q.name == p.name) {
                Some(q) if q.ty == p.ty => {}
                Some(q) => {
                    return err(format!(
                        "continuation `{}` parameter `{}` has type `{}` but the prologue declares `{}`",
                        c.fn_name, p.name, p.ty, q.ty
                    ))
                }
                None => {
                    return err(format!(
                        "continuation `{}` parameter `{}` does not name a prologue parameter (continuations \
                         are re-bound by parameter name)",
                        c.fn_name, p.name
                    ))
                }
            }
        }
    }

    Ok(Analysis {
        reflection: YieldReflection {
            prologue_params,
            prologue_yields_to: prologue_targets.into_iter().collect(),
            prologue_has_thread_id: prologue.thread_id_name.is_some(),
            prologue_uses_group_ids,
            prologue_numthreads: entry.numthreads.unwrap_or(DEFAULT_NUMTHREADS),
            continuations,
            petitions,
        },
        prologue,
        sites,
    })
}

fn program_param(p: &Param, owner: &str) -> Result<ProgramParam, String> {
    let ty = p.ty.trim();
    match &p.kind {
        ParamKind::Scalar => Ok(ProgramParam {
            name: p.name.clone(),
            ty: ty.to_string(),
            is_scalar: true,
        }),
        ParamKind::Resource if ty.starts_with("Scattered<") || ty.starts_with("BufRO<") || ty == "ByteAddress" => {
            Ok(ProgramParam {
                name: p.name.clone(),
                ty: ty.to_string(),
                is_scalar: false,
            })
        }
        ParamKind::Broadcast | ParamKind::PassThrough => Ok(ProgramParam {
            name: p.name.clone(),
            ty: ty.to_string(),
            is_scalar: false,
        }),
        ParamKind::Resource => err(format!(
            "`{owner}` parameter `{}` of type `{ty}`: yielding scripts bind buffers only in v0",
            p.name
        )),
        ParamKind::SystemValue(_) => err(format!("`{owner}` parameter `{}` is a system value", p.name)),
    }
}

fn leading_identifier(expr: &str) -> Option<String> {
    let s = expr.trim_start();
    scan_identifier(s, 0).map(|(id, _)| id)
}

// ---- petitions -------------------------------------------------------------

fn find_petitions(source: &str) -> Result<Vec<PetitionDecl>, String> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(pos) = find_substr(source, from, "[goldy_petition") {
        from = pos + 1;
        if is_in_line_comment(source, pos) {
            continue;
        }
        let (attr, attr_end) = scan_bracket_block(source, pos).ok_or("unterminated [goldy_petition")?;
        let result_elem = parse_petition_result(attr.trim())?;
        let mut p = skip_whitespace_and_comments(source, attr_end);
        // Skip further attributes and the optional `public`.
        while source[p..].starts_with('[') {
            let (_, e) = scan_bracket_block(source, p).ok_or("unterminated attribute")?;
            p = skip_whitespace_and_comments(source, e);
        }
        let (mut kw, mut q) = scan_identifier(source, p).ok_or("[goldy_petition] must precede a struct")?;
        if kw == "public" {
            (kw, q) = scan_identifier(source, skip_whitespace_and_comments(source, q))
                .ok_or("[goldy_petition] must precede a struct")?;
        }
        if kw != "struct" {
            return err("[goldy_petition] must precede a `struct` declaration");
        }
        let (name, _) = scan_identifier(source, skip_whitespace_and_comments(source, q))
            .ok_or("[goldy_petition] struct has no name")?;
        let payload_bytes = struct_bytes(source, &name, &mut Vec::new())?;
        out.push(PetitionDecl {
            name,
            result_elem,
            payload_bytes,
            attr_range: pos..attr_end,
        });
    }
    Ok(out)
}

/// `goldy_petition(Result = BufRO<E>)` / `goldy_petition(Result = E)` → `E`.
fn parse_petition_result(attr: &str) -> Result<String, String> {
    let rest = attr
        .strip_prefix("goldy_petition")
        .ok_or("malformed [goldy_petition]")?
        .trim();
    let inner = rest
        .strip_prefix('(')
        .and_then(|r| r.strip_suffix(')'))
        .ok_or("[goldy_petition] requires `(Result = BufRO<E>)`")?
        .trim();
    let value = inner
        .strip_prefix("Result")
        .map(|r| r.trim_start())
        .and_then(|r| r.strip_prefix('='))
        .ok_or("[goldy_petition] requires `Result = BufRO<E>`")?
        .trim();
    let elem = value
        .strip_prefix("BufRO<")
        .and_then(|r| r.strip_suffix('>'))
        .unwrap_or(value)
        .trim();
    if elem.is_empty() || elem.contains(',') {
        return err(format!(
            "[goldy_petition] result `{value}` must be `BufRO<E>` for one element type E"
        ));
    }
    Ok(elem.to_string())
}

// ---- struct layout ---------------------------------------------------------

/// Byte size of struct `name` under the v0 rule (4-byte scalars, arrays, nested structs).
fn struct_bytes(source: &str, name: &str, stack: &mut Vec<String>) -> Result<u32, String> {
    if let Some(b) = scalar_bytes(name) {
        return Ok(b);
    }
    if stack.iter().any(|s| s == name) {
        return err(format!("struct `{name}` is recursive"));
    }
    let Some(body) = find_struct_body(source, name) else {
        return err(format!(
            "type `{name}` is not a 4-byte scalar or a struct declared in this source (v0 petition/state \
             types must be built from uint/int/float)"
        ));
    };
    stack.push(name.to_string());
    let mut total = 0u32;
    for decl in strip_comments(&source[body]).split(';') {
        let decl = decl.trim();
        if decl.is_empty() {
            continue;
        }
        let decl = decl.strip_prefix("public").map(str::trim_start).unwrap_or(decl);
        // `type name` or `type name[N]`; multiple declarators are not supported.
        let (ty, rest) = decl
            .split_once(char::is_whitespace)
            .ok_or_else(|| format!("yielding script: cannot parse field `{decl}` of struct `{name}`"))?;
        let rest = rest.trim();
        if rest.contains(',') {
            return err(format!("struct `{name}`: one declarator per field (`{decl}`)"));
        }
        let count = match rest.find('[') {
            Some(i) => {
                let n = rest[i + 1..].trim_end_matches(']').trim().parse::<u32>().map_err(|_| {
                    format!("yielding script: struct `{name}` array field `{decl}` needs a literal length")
                })?;
                n
            }
            None => 1,
        };
        let field_bytes =
            struct_bytes(source, ty.trim(), stack).map_err(|e| format!("{e} (field `{rest}` of struct `{name}`)"))?;
        total = total
            .checked_add(field_bytes.checked_mul(count).ok_or("struct too large")?)
            .ok_or("struct too large")?;
    }
    stack.pop();
    if total == 0 {
        return err(format!("struct `{name}` has no fields"));
    }
    Ok(total)
}

fn scalar_bytes(ty: &str) -> Option<u32> {
    match ty {
        "uint" | "int" | "float" => Some(4),
        _ => None,
    }
}

/// Byte range of the `{ .. }` body of `struct name`.
fn find_struct_body(source: &str, name: &str) -> Option<Range<usize>> {
    let mut from = 0;
    while let Some(pos) = find_substr(source, from, "struct") {
        from = pos + 6;
        if is_in_line_comment(source, pos) {
            continue;
        }
        if pos > 0 && is_ident_byte(source.as_bytes()[pos - 1]) {
            continue;
        }
        let p = skip_whitespace_and_comments(source, pos + 6);
        let Some((id, after)) = scan_identifier(source, p) else {
            continue;
        };
        if id != name {
            continue;
        }
        let mut p = skip_whitespace_and_comments(source, after);
        // Skip an inheritance / conformance clause `: IFoo`.
        if source[p..].starts_with(':') {
            p = source[p..].find('{').map(|i| p + i)?;
        }
        if !source[p..].starts_with('{') {
            continue;
        }
        let end = match_brace(source, p)?;
        return Some(p + 1..end - 1);
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

// ---- continuations ---------------------------------------------------------

fn find_continuations(source: &str, petitions: &[PetitionDecl]) -> Result<Vec<ContinuationDecl>, String> {
    let mut out: Vec<ContinuationDecl> = Vec::new();
    let mut from = 0;
    while let Some(pos) = find_substr(source, from, "[goldy_resume") {
        from = pos + 1;
        if is_in_line_comment(source, pos) {
            continue;
        }
        let (attr, attr_end) = scan_bracket_block(source, pos).ok_or("unterminated [goldy_resume")?;
        let explicit_petition = parse_resume_attr(attr.trim())?;

        // Attributes around the function: `[numthreads]` may precede or follow.
        let group_start = find_attr_group_start(source, pos);
        let mut numthreads: Option<((u32, u32, u32), Range<usize>)> = None;
        let mut p = skip_whitespace_and_comments(source, attr_end);
        while source[p..].starts_with('[') {
            let (inner, e) = scan_bracket_block(source, p).ok_or("unterminated attribute")?;
            if inner.trim().starts_with("numthreads") {
                if let Some(nt) = virtual_main::parse_numthreads(inner.trim()) {
                    numthreads = Some((nt, p..e));
                }
            }
            p = skip_whitespace_and_comments(source, e);
        }
        if numthreads.is_none() {
            numthreads = find_numthreads_in_range(source, group_start, pos);
        }

        let (ret, q) = scan_identifier(source, p).ok_or("[goldy_resume] must precede a function")?;
        if ret != "void" {
            return err("[goldy_resume] functions must return void");
        }
        let fn_name_start = skip_whitespace_and_comments(source, q);
        let (fn_name, fn_name_end) =
            scan_identifier(source, fn_name_start).ok_or("[goldy_resume] function has no name")?;
        let params_open = skip_whitespace_and_comments(source, fn_name_end);
        if !source[params_open..].starts_with('(') {
            return err(format!("[goldy_resume] `{fn_name}` is not a function declaration"));
        }
        let params_close = find_matching_close(source, params_open, '(', ')').ok_or("unterminated parameter list")?;
        let params_str = &source[params_open + 1..params_close];
        if params_str.contains("#if") {
            return err("preprocessor-conditional parameter lists are not supported in yielding scripts");
        }
        let params = parse_params(params_str);
        let body_open = skip_whitespace_and_comments(source, params_close + 1);
        if !source[body_open..].starts_with('{') {
            return err(format!("[goldy_resume] `{fn_name}` must have a body"));
        }
        let body_end = match_brace(source, body_open).ok_or("unterminated function body")?;

        // Signature: program params.., Resolved<E> r, State s, [ThreadId tid] (any order for tid).
        let resolved_index = params
            .iter()
            .position(|p| p.ty.trim().starts_with("Resolved<"))
            .ok_or_else(|| format!("yielding script: continuation `{fn_name}` needs a `Resolved<E>` parameter"))?;
        if params.iter().filter(|p| p.ty.trim().starts_with("Resolved<")).count() > 1 {
            return err(format!(
                "continuation `{fn_name}` may take only one `Resolved<E>` parameter"
            ));
        }
        let result_elem = params[resolved_index]
            .ty
            .trim()
            .strip_prefix("Resolved<")
            .and_then(|r| r.strip_suffix('>'))
            .map(|s| s.trim().to_string())
            .ok_or_else(|| format!("yielding script: malformed Resolved<> on `{fn_name}`"))?;
        let state_index = resolved_index + 1;
        let Some(state) = params.get(state_index) else {
            return err(format!(
                "continuation `{fn_name}`: the state parameter must directly follow `Resolved<E>`"
            ));
        };
        let state_ty = state.ty.trim().to_string();
        if !matches!(classify_type(&state_ty), ParamKind::PassThrough | ParamKind::Broadcast) {
            return err(format!(
                "continuation `{fn_name}`: parameter `{}` after `Resolved<E>` must be the saved-state struct, got `{state_ty}`",
                state.name
            ));
        }
        let state_bytes = struct_bytes(source, &state_ty, &mut Vec::new())?;
        let thread_id_index = params.iter().position(|p| p.ty.trim() == "ThreadId");

        let mut program_params = Vec::new();
        for (i, p) in params.iter().enumerate() {
            if i == resolved_index || i == state_index || Some(i) == thread_id_index {
                continue;
            }
            if matches!(p.kind, ParamKind::SystemValue(_)) {
                return err(format!(
                    "continuation `{fn_name}` parameter `{}`: only ThreadId is available in a continuation",
                    p.name
                ));
            }
            program_params.push(program_param(p, &fn_name)?);
        }
        if program_params.iter().any(|p| p.name.starts_with(HIDDEN_PREFIX)) {
            return err(format!("parameter names starting with `{HIDDEN_PREFIX}` are reserved"));
        }

        if let Some(name) = &explicit_petition {
            if !petitions.iter().any(|p| &p.name == name) {
                return err(format!(
                    "[goldy_resume({name})] on `{fn_name}`: `{name}` has no [goldy_petition] declaration"
                ));
            }
        }
        if out.iter().any(|c| c.fn_name == fn_name) {
            return err(format!("continuation `{fn_name}` is declared twice"));
        }

        out.push(ContinuationDecl {
            fn_name,
            petition: explicit_petition.unwrap_or_default(),
            result_elem,
            state_ty,
            state_bytes,
            numthreads: numthreads.as_ref().map(|(nt, _)| *nt).unwrap_or(DEFAULT_NUMTHREADS),
            program_params,
            yields_to: Vec::new(),
            has_thread_id: thread_id_index.is_some(),
            attr_range: pos..attr_end,
            numthreads_range: numthreads.map(|(_, r)| r),
            fn_name_range: fn_name_start..fn_name_end,
            params_close,
            body: body_open..body_end,
            resolved_index,
            state_index,
            thread_id_index,
        });
    }
    Ok(out)
}

/// `goldy_resume` → `None`; `goldy_resume(P)` → `Some("P")`.
fn parse_resume_attr(attr: &str) -> Result<Option<String>, String> {
    let rest = attr
        .strip_prefix("goldy_resume")
        .ok_or("malformed [goldy_resume]")?
        .trim();
    if rest.is_empty() {
        return Ok(None);
    }
    let inner = rest
        .strip_prefix('(')
        .and_then(|r| r.strip_suffix(')'))
        .ok_or("[goldy_resume] takes an optional `(Petition)` argument")?
        .trim();
    if inner.is_empty() {
        return Ok(None);
    }
    Ok(Some(inner.to_string()))
}

// ---- prologue --------------------------------------------------------------

fn find_prologue(source: &str) -> Result<Prologue, String> {
    let entries = find_all_entries(source);
    let mut compute = entries.iter().filter(|e| e.stage == Stage::Compute);
    let Some(entry) = compute.next() else {
        return err("a yielding script needs exactly one [goldy_compute] entry point");
    };
    if compute.next().is_some() || entries.iter().any(|e| e.stage != Stage::Compute) {
        return err("a yielding script may declare only one [goldy_compute] entry point and no other stages");
    }
    let params_open = skip_whitespace_and_comments(source, entry.fn_name_range.end);
    let params_close = find_matching_close(source, params_open, '(', ')').ok_or("unterminated parameter list")?;
    let body_open = skip_whitespace_and_comments(source, params_close + 1);
    if !source[body_open..].starts_with('{') {
        return err("the [goldy_compute] prologue must have a body");
    }
    let body_end = match_brace(source, body_open).ok_or("unterminated prologue body")?;
    let thread_id_name = entry.params.iter().find_map(|item| match item {
        ParamItem::Single(p) if p.ty.trim() == "ThreadId" => Some(p.name.clone()),
        _ => None,
    });
    Ok(Prologue {
        fn_name: entry.fn_name.clone(),
        attr_range: entry.goldy_attr_range.clone(),
        numthreads_range: entry.numthreads_attr_range.clone(),
        fn_name_range: entry.fn_name_range.clone(),
        params_close,
        body: body_open..body_end,
        thread_id_name,
    })
}

// ---- yield sites -----------------------------------------------------------

fn find_yield_sites(source: &str) -> Result<Vec<YieldSite>, String> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(pos) = find_substr(source, from, "$yield") {
        from = pos + 1;
        if is_in_line_comment(source, pos) {
            continue;
        }
        let open = skip_whitespace_and_comments(source, pos + "$yield".len());
        if !source[open..].starts_with('(') {
            return err("`$yield` must be called as `$yield(continuation, payload, state)`");
        }
        let close = find_matching_close(source, open, '(', ')').ok_or("unterminated $yield(")?;
        let args = split_top_level_args(&source[open + 1..close]);
        if args.len() != 3 {
            return err(format!(
                "`$yield` takes (continuation, payload, state); got {} argument(s)",
                args.len()
            ));
        }
        let target = args[0].trim();
        if scan_identifier(target, 0).map(|(_, e)| e) != Some(target.len()) || target.is_empty() {
            return err(format!(
                "`$yield` first argument must be a [goldy_resume] function name, got `{target}`"
            ));
        }
        out.push(YieldSite {
            range: pos..close + 1,
            target: target.to_string(),
            payload_expr: args[1].trim().to_string(),
            state_expr: args[2].trim().to_string(),
        });
    }
    Ok(out)
}

fn split_top_level_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                cur.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => out.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() || !out.is_empty() {
        out.push(cur);
    }
    out
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

/// Hidden parameter names (shared with the host-side driver through the ABI order
/// documented on [`hidden_yield_params`] / [`continuation_entry_params`]).
pub(crate) fn pay_param(cont: &str) -> String {
    format!("{HIDDEN_PREFIX}pay_{cont}")
}
pub(crate) fn state_param(cont: &str) -> String {
    format!("{HIDDEN_PREFIX}st_{cont}")
}
pub(crate) fn cap_param(cont: &str) -> String {
    format!("{HIDDEN_PREFIX}cap_{cont}")
}
pub(crate) const CNT_PARAM: &str = "__gy_cnt";
pub(crate) const BASE_PARAM: &str = "__gy_base";
pub(crate) const COUNT_PARAM: &str = "__gy_count";

/// Hidden trailing parameters appended to a function that yields to `targets`.
///
/// Resources (in bindless-slot order): `Scattered<P_c> pay_c, Scattered<S_c> st_c`
/// per target, then `Scattered<Interlocked<uint>> __gy_cnt`. Scalars (in user-word
/// order): `uint cap_c` per target.
fn hidden_yield_params(reflection: &YieldReflection, targets: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for t in targets {
        let c = reflection.continuation(t).expect("target validated");
        out.push(format!("Scattered<{}> {}", c.petition, pay_param(t)));
        out.push(format!("Scattered<{}> {}", c.state_ty, state_param(t)));
    }
    if !targets.is_empty() {
        out.push(format!("Scattered<Interlocked<uint>> {CNT_PARAM}"));
    }
    for t in targets {
        out.push(format!("uint {}", cap_param(t)));
    }
    out
}

/// `P { a, b }` → `{ a, b }`; anything else unchanged.
fn initializer(expr: &str) -> &str {
    let trimmed = expr.trim();
    if let Some((_, after)) = scan_identifier(trimmed, 0) {
        let rest = trimmed[after..].trim_start();
        if rest.starts_with('{') && trimmed.ends_with('}') {
            return rest;
        }
    }
    trimmed
}

fn emit(source: &str, analysis: &Analysis, variant: Option<&str>) -> Result<String, String> {
    let refl = &analysis.reflection;
    let variant_idx = match variant {
        None => None,
        Some(name) => Some(
            refl.continuation_index(name)
                .ok_or_else(|| format!("yielding script: no continuation named `{name}`"))?,
        ),
    };

    let mut edits: Vec<(Range<usize>, String)> = Vec::new();

    // Petition attributes: strip.
    for p in &refl.petitions {
        edits.push((p.attr_range.clone(), String::new()));
    }

    // Continuations: strip attributes, append hidden yield params to bodies that yield.
    for c in &refl.continuations {
        edits.push((c.attr_range.clone(), String::new()));
        if let Some(r) = &c.numthreads_range {
            edits.push((r.clone(), String::new()));
        }
        let hidden = hidden_yield_params(refl, &c.yields_to);
        if !hidden.is_empty() {
            edits.push((c.params_close..c.params_close, format!(", {}", hidden.join(", "))));
        }
    }

    // Prologue: hidden yield params + chunk base; demote to a plain function in
    // continuation variants.
    {
        let pro = &analysis.prologue;
        let mut hidden = hidden_yield_params(refl, &refl.prologue_yields_to);
        hidden.push(format!("uint {BASE_PARAM}"));
        edits.push((pro.params_close..pro.params_close, format!(", {}", hidden.join(", "))));
        if let Some(tid) = &pro.thread_id_name {
            let insert_at = pro.body.start + 1;
            edits.push((insert_at..insert_at, format!(" {tid}.x += {BASE_PARAM};")));
        }
        if variant_idx.is_some() {
            edits.push((pro.attr_range.clone(), String::new()));
            if let Some(r) = &pro.numthreads_range {
                edits.push((r.clone(), String::new()));
            }
            edits.push((pro.fn_name_range.clone(), format!("__goldy_prologue_{}", pro.fn_name)));
        }
    }

    // Yield sites. Slang has no `Type { .. }` expression, so a `P { .. }` / `{ .. }`
    // argument becomes an initializer for a typed local; other expressions pass through.
    for site in &analysis.sites {
        let t = &site.target;
        let c = refl.continuation(t).expect("target validated");
        edits.push((
            site.range.clone(),
            format!(
                "{{ {p} {HIDDEN_PREFIX}p = {}; {s} {HIDDEN_PREFIX}s = {}; __goldy_yield_{t}({CNT_PARAM}, {}, {}, {}, {HIDDEN_PREFIX}p, {HIDDEN_PREFIX}s); }}",
                initializer(&site.payload_expr),
                initializer(&site.state_expr),
                pay_param(t),
                state_param(t),
                cap_param(t),
                p = c.petition,
                s = c.state_ty,
            ),
        ));
    }

    // Apply edits back to front. Insertions at the same offset keep relative order.
    edits.sort_by(|a, b| b.0.start.cmp(&a.0.start).then(b.0.end.cmp(&a.0.end)));
    let mut out = source.to_string();
    for (range, text) in edits {
        out.replace_range(range, &text);
    }

    // Generated tail: yield helpers and, for continuation variants, the entry wrapper.
    out.push_str("\n\n// [generated by goldy yielding — do not edit]\n");
    for (k, c) in refl.continuations.iter().enumerate() {
        out.push_str(&format!(
            "void __goldy_yield_{name}(Scattered<Interlocked<uint>> cnt, Scattered<{p}> pay, Scattered<{s}> st, uint cap, {p} payload, {s} state) {{\n\
             \x20   uint idx = InterlockedAdd(cnt[{k}u], 1u);\n\
             \x20   if (idx < cap) {{ pay[idx] = payload; st[idx] = state; }}\n\
             }}\n",
            name = c.fn_name,
            p = c.petition,
            s = c.state_ty,
        ));
    }
    if let Some(idx) = variant_idx {
        out.push_str(&continuation_entry(refl, &refl.continuations[idx]));
    }
    Ok(out)
}

/// Entry-point parameter list of a continuation variant, in declaration order:
/// program params, `Scattered<P> __gy_pay`, `Scattered<S> __gy_st`,
/// `Scattered<uint2> __gy_res`, `Scattered<E> __gy_arena`, hidden yield params,
/// `ThreadId __gy_tid`, `uint __gy_count`, hidden caps.
fn continuation_entry(refl: &YieldReflection, c: &ContinuationDecl) -> String {
    let (x, y, z) = c.numthreads;
    let mut sig: Vec<String> = c
        .program_params
        .iter()
        .map(|p| format!("{} {}", p.ty, p.name))
        .collect();
    sig.push(format!("Scattered<{}> {HIDDEN_PREFIX}pay", c.petition));
    sig.push(format!("Scattered<{}> {HIDDEN_PREFIX}st", c.state_ty));
    sig.push(format!("Scattered<uint2> {HIDDEN_PREFIX}res"));
    sig.push(format!("Scattered<{}> {HIDDEN_PREFIX}arena", c.result_elem));
    let hidden = hidden_yield_params(refl, &c.yields_to);
    let (hidden_res, hidden_scalars): (Vec<&String>, Vec<&String>) =
        hidden.iter().partition(|h| !h.starts_with("uint "));
    sig.extend(hidden_res.iter().map(|s| s.to_string()));
    sig.push(format!("ThreadId {HIDDEN_PREFIX}tid"));
    sig.push(format!("uint {COUNT_PARAM}"));
    sig.extend(hidden_scalars.iter().map(|s| s.to_string()));

    // Call the continuation body with its declared parameter order.
    let mut call: Vec<String> = Vec::new();
    let total = c.program_params.len() + 2 + usize::from(c.thread_id_index.is_some());
    let mut prog = c.program_params.iter();
    for i in 0..total {
        if i == c.resolved_index {
            call.push(format!("{HIDDEN_PREFIX}rv"));
        } else if i == c.state_index {
            call.push(format!("{HIDDEN_PREFIX}s"));
        } else if Some(i) == c.thread_id_index {
            call.push(format!("{HIDDEN_PREFIX}tid"));
        } else {
            call.push(prog.next().expect("program param count").name.clone());
        }
    }
    for t in &c.yields_to {
        call.push(pay_param(t));
        call.push(state_param(t));
    }
    if !c.yields_to.is_empty() {
        call.push(CNT_PARAM.to_string());
    }
    for t in &c.yields_to {
        call.push(cap_param(t));
    }

    format!(
        "\n[goldy_compute]\n[numthreads({x}, {y}, {z})]\nvoid cs_main({sig}) {{\n\
         \x20   if ({HIDDEN_PREFIX}tid.x >= {COUNT_PARAM}) return;\n\
         \x20   {state} {HIDDEN_PREFIX}s = {HIDDEN_PREFIX}st[{HIDDEN_PREFIX}tid.x];\n\
         \x20   uint2 {HIDDEN_PREFIX}r = {HIDDEN_PREFIX}res[{HIDDEN_PREFIX}tid.x];\n\
         \x20   Resolved<{elem}> {HIDDEN_PREFIX}rv = Resolved<{elem}>({HIDDEN_PREFIX}arena, {HIDDEN_PREFIX}r.x, {HIDDEN_PREFIX}r.y);\n\
         \x20   {body}({call});\n\
         }}\n",
        sig = sig.join(", "),
        state = c.state_ty,
        elem = c.result_elem,
        body = c.fn_name,
        call = call.join(", "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"
import goldy_exp;

[goldy_petition(Result = BufRO<uint>)]
struct Fetch { uint key; };

struct St { uint lane; uint acc; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, uint scale, ThreadId tid) {
    uint v = data[tid.x];
    if (v % 2u == 1u) {
        $yield(cs_resume, Fetch { v }, St { tid.x, v * scale });
        return;
    }
    data[tid.x] = v * 2u;
}

[goldy_resume]
[numthreads(32, 1, 1)]
void cs_resume(Scattered<uint> data, Resolved<uint> r, St s, ThreadId tid) {
    data[s.lane] = r.is_null() ? 0xFFFFFFFFu : r[0] + s.acc;
}
"#;

    #[test]
    fn reflects_prologue_and_continuation() {
        let r = reflect(SRC).unwrap().unwrap();
        assert_eq!(r.prologue_yields_to, vec!["cs_resume".to_string()]);
        assert!(r.prologue_has_thread_id);
        assert_eq!(r.prologue_numthreads, (64, 1, 1));
        assert_eq!(r.prologue_params.len(), 2);
        assert_eq!(r.prologue_params[1].name, "scale");
        assert!(r.prologue_params[1].is_scalar);
        assert_eq!(r.petitions.len(), 1);
        assert_eq!(r.petitions[0].payload_bytes, 4);
        assert_eq!(r.petitions[0].result_elem, "uint");
        let c = &r.continuations[0];
        assert_eq!(c.fn_name, "cs_resume");
        assert_eq!(c.petition, "Fetch");
        assert_eq!(c.state_ty, "St");
        assert_eq!(c.state_bytes, 8);
        assert_eq!(c.numthreads, (32, 1, 1));
        assert_eq!(c.program_params.len(), 1);
        assert!(c.yields_to.is_empty());
        assert!(c.has_thread_id);
    }

    #[test]
    fn plain_sources_pass_through() {
        let src = "[goldy_compute] void cs_main(ThreadId tid) {}";
        assert!(reflect(src).unwrap().is_none());
        assert_eq!(lower(src, None).unwrap(), src);
    }

    #[test]
    fn lowers_prologue_variant() {
        let out = lower(SRC, None).unwrap();
        assert!(!out.contains("$yield"));
        assert!(!out.contains("[goldy_petition"));
        assert!(!out.contains("[goldy_resume"));
        assert!(out.contains(
            "void cs_main(Scattered<uint> data, uint scale, ThreadId tid, Scattered<Fetch> __gy_pay_cs_resume, \
             Scattered<St> __gy_st_cs_resume, Scattered<Interlocked<uint>> __gy_cnt, uint __gy_cap_cs_resume, uint __gy_base) { tid.x += __gy_base;"
        ));
        assert!(out.contains(
            "{ Fetch __gy_p = { v }; St __gy_s = { tid.x, v * scale }; __goldy_yield_cs_resume(__gy_cnt, \
             __gy_pay_cs_resume, __gy_st_cs_resume, __gy_cap_cs_resume, __gy_p, __gy_s); };"
        ));
        assert!(out.contains("void cs_resume(Scattered<uint> data, Resolved<uint> r, St s, ThreadId tid) {"));
        assert!(out.contains("uint idx = InterlockedAdd(cnt[0u], 1u);"));
        // Exactly one compute entry survives, and it is still on the same source line.
        assert_eq!(out.matches("[goldy_compute]").count(), 1);
        let line_of = |needle: &str| out[..out.find(needle).unwrap()].matches('\n').count();
        assert_eq!(
            line_of("void cs_main("),
            SRC[..SRC.find("void cs_main(").unwrap()].matches('\n').count()
        );
    }

    #[test]
    fn lowers_continuation_variant() {
        let out = lower(SRC, Some("cs_resume")).unwrap();
        assert_eq!(out.matches("[goldy_compute]").count(), 1);
        assert!(out.contains("void __goldy_prologue_cs_main("));
        assert!(out.contains("[numthreads(32, 1, 1)]\nvoid cs_main(Scattered<uint> data, Scattered<Fetch> __gy_pay, Scattered<St> __gy_st, Scattered<uint2> __gy_res, Scattered<uint> __gy_arena, ThreadId __gy_tid, uint __gy_count) {"));
        assert!(out.contains("cs_resume(data, __gy_rv, __gy_s, __gy_tid);"));
        assert!(lower(SRC, Some("nope")).is_err());
    }

    #[test]
    fn continuation_can_yield_again() {
        let src = r#"
[goldy_petition(Result = BufRO<uint>)] struct A { uint k; };
[goldy_petition(Result = BufRO<float>)] struct B { uint k; uint j; };
struct S1 { uint x; };
struct S2 { uint x; uint y[3]; };
[goldy_compute] [numthreads(8,1,1)]
void cs_main(Scattered<uint> d, ThreadId t) { $yield(ra, A{1}, S1{t.x}); }
[goldy_resume(A)] void ra(Scattered<uint> d, Resolved<uint> r, S1 s) { $yield(rb, B{1,2}, S2{s.x, {1,2,3}}); }
[goldy_resume(B)] void rb(Resolved<float> r, S2 s, ThreadId t) { d[0] = 1; }
"#;
        let r = reflect(src).unwrap().unwrap();
        assert_eq!(r.continuations[0].yields_to, vec!["rb".to_string()]);
        assert_eq!(r.continuations[1].state_bytes, 16);
        assert!(r.continuations[1].program_params.is_empty());
        assert!(!r.continuations[0].has_thread_id);
        let out = lower(src, Some("ra")).unwrap();
        assert!(out.contains("void ra(Scattered<uint> d, Resolved<uint> r, S1 s, Scattered<B> __gy_pay_rb, Scattered<S2> __gy_st_rb, Scattered<Interlocked<uint>> __gy_cnt, uint __gy_cap_rb)"));
        assert!(out.contains("ra(d, __gy_rv, __gy_s, __gy_pay_rb, __gy_st_rb, __gy_cnt, __gy_cap_rb);"));
        assert!(out.contains("uint idx = InterlockedAdd(cnt[1u], 1u);"));
    }

    #[test]
    fn rejects_bad_shapes() {
        // Unknown target.
        let e = reflect(
            "[goldy_petition(Result = BufRO<uint>)] struct A { uint k; }; struct S { uint x; };\n\
             [goldy_compute] void cs_main(ThreadId t) { $yield(nope, A{1}, S{1}); }\n\
             [goldy_resume] void ra(Resolved<uint> r, S s) {}",
        )
        .unwrap_err();
        assert!(e.contains("not a [goldy_resume]"), "{e}");
        // Vector fields are outside the v0 layout rule.
        let e = reflect(
            "[goldy_petition(Result = BufRO<uint>)] struct A { float3 p; }; struct S { uint x; };\n\
             [goldy_compute] void cs_main(ThreadId t) { $yield(ra, A{1}, S{1}); }\n\
             [goldy_resume] void ra(Resolved<uint> r, S s) {}",
        )
        .unwrap_err();
        assert!(e.contains("float3"), "{e}");
        // Result element mismatch.
        let e = reflect(
            "[goldy_petition(Result = BufRO<uint>)] struct A { uint k; }; struct S { uint x; };\n\
             [goldy_compute] void cs_main(ThreadId t) { $yield(ra, A{1}, S{1}); }\n\
             [goldy_resume] void ra(Resolved<float> r, S s) {}",
        )
        .unwrap_err();
        assert!(e.contains("Resolved<float>"), "{e}");
        // Continuation param not on the prologue.
        let e = reflect(
            "[goldy_petition(Result = BufRO<uint>)] struct A { uint k; }; struct S { uint x; };\n\
             [goldy_compute] void cs_main(Scattered<uint> d, ThreadId t) { $yield(ra, A{1}, S{1}); }\n\
             [goldy_resume] void ra(Scattered<uint> other, Resolved<uint> r, S s) {}",
        )
        .unwrap_err();
        assert!(e.contains("does not name a prologue parameter"), "{e}");
        // Yield outside a yielding function.
        let e = reflect(
            "[goldy_petition(Result = BufRO<uint>)] struct A { uint k; }; struct S { uint x; };\n\
             void helper() { $yield(ra, A{1}, S{1}); }\n\
             [goldy_compute] void cs_main(ThreadId t) { helper(); }\n\
             [goldy_resume] void ra(Resolved<uint> r, S s) {}",
        )
        .unwrap_err();
        assert!(e.contains("must appear directly inside"), "{e}");
    }
}
