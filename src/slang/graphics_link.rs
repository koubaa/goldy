//! Automatic reflected linking for Goldy graphics pipelines.
//!
//! A graphics pipeline is a typed connection:
//! vertex input → vertex/mesh stage → interpolated payload → fragment stage.
//! Payloads are shader-owned; Rust never repeats the varying struct.
//!
//! Virtual-main bindless parameters are merged by name into one pipeline-wide
//! resource contract. Each stage may declare only the resources it uses; wrappers
//! remap local names onto the shared push-constant slot order.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};

use super::virtual_main::{
    find_all_entries, flatten_bindless_params, mesh_payload_type, mesh_vertices_type, stage_input_payload_type,
    EntryDef, Param, ParamKind, Stage,
};
use crate::backend::shared::MAX_BINDLESS_SLOTS;
use crate::types::{ResourceAccess, ResourceCategory, VertexBufferLayout, VertexFormat};

/// Interpolation mode on a stage I/O field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum InterpolationMode {
    /// Default perspective-correct interpolation (float types).
    Perspective,
    /// `linear` / centroid-style; treated as compatible with [`Self::Perspective`].
    Linear,
    /// `noperspective`.
    NoPerspective,
    /// `nointerpolation`.
    NoInterpolation,
}

impl InterpolationMode {
    fn is_compatible_with(self, other: Self) -> bool {
        match (self, other) {
            (Self::NoInterpolation, Self::NoInterpolation) => true,
            (Self::NoPerspective, Self::NoPerspective) => true,
            (Self::NoInterpolation, _) | (_, Self::NoInterpolation) => false,
            (Self::NoPerspective, _) | (_, Self::NoPerspective) => false,
            _ => true,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Perspective => "perspective",
            Self::Linear => "linear",
            Self::NoPerspective => "noperspective",
            Self::NoInterpolation => "nointerpolation",
        }
    }
}

/// One field of a stage input or output payload.
///
/// Structural identity is `(semantic, semantic_index, scalar_type, vector_size, interpolation)`.
/// `field_name` and `struct_name` are diagnostic labels only.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageIoField {
    pub field_name: String,
    pub struct_name: String,
    pub semantic: String,
    pub semantic_index: u32,
    pub scalar_type: String,
    pub vector_size: u32,
    pub interpolation: InterpolationMode,
}

impl StageIoField {
    fn match_key(&self) -> (String, u32) {
        (self.semantic.to_ascii_uppercase(), self.semantic_index)
    }

    fn shape_compatible(&self, other: &Self) -> bool {
        self.scalar_type.eq_ignore_ascii_case(&other.scalar_type) && self.vector_size == other.vector_size
    }

    fn describe(&self) -> String {
        let ty = if self.vector_size <= 1 {
            self.scalar_type.clone()
        } else {
            format!("{}{}", self.scalar_type, self.vector_size)
        };
        format!(
            "{}.{} : {}{} ({ty}, {})",
            self.struct_name,
            self.field_name,
            self.semantic,
            if self.semantic_index == 0 && !self.semantic.chars().last().is_some_and(|c| c.is_ascii_digit()) {
                String::new()
            } else {
                self.semantic_index.to_string()
            },
            self.interpolation.name()
        )
    }
}

/// Reflected I/O for one compiled (or source-parsed) graphics stage.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageInterface {
    pub stage: String,
    pub entry_name: String,
    pub vertex_inputs: Vec<StageIoField>,
    pub payload_inputs: Vec<StageIoField>,
    pub payload_outputs: Vec<StageIoField>,
    pub fragment_outputs: Vec<StageIoField>,
}

/// One named bindless parameter in the merged pipeline contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineResource {
    pub name: String,
    pub ty: String,
    pub category: ResourceCategory,
    pub access: ResourceAccess,
    pub element_type: Option<String>,
    pub declaring_stages: Vec<String>,
}

impl PipelineResource {
    pub fn slot_access(&self) -> Option<ResourceAccess> {
        match self.category {
            ResourceCategory::Scattered | ResourceCategory::StorageImage => Some(self.access),
            ResourceCategory::Broadcast
            | ResourceCategory::Texture
            | ResourceCategory::Sampler
            | ResourceCategory::Accel => Some(ResourceAccess::Read),
        }
    }

    #[cfg(all(feature = "dx12", target_os = "windows"))]
    pub(crate) fn slot_kind(&self) -> Option<crate::types::BindlessSlotKind> {
        use crate::types::BindlessSlotKind;
        if self.ty.starts_with("BufRO<")
            || (self.category == ResourceCategory::Scattered && self.access == ResourceAccess::Read)
        {
            Some(BindlessSlotKind::ReadOnlySrv)
        } else if matches!(
            self.category,
            ResourceCategory::Scattered | ResourceCategory::StorageImage
        ) {
            Some(BindlessSlotKind::StorageUav)
        } else if self.category == ResourceCategory::Accel {
            Some(BindlessSlotKind::ReadOnlySrv)
        } else if self.category == ResourceCategory::Broadcast {
            Some(BindlessSlotKind::UniformCbv)
        } else {
            None
        }
    }
}

/// Merged per-pipeline bindless layout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PipelineResourceContract {
    pub resources: Vec<PipelineResource>,
}

impl PipelineResourceContract {
    pub fn slot_access(&self) -> Vec<Option<ResourceAccess>> {
        self.resources.iter().map(PipelineResource::slot_access).collect()
    }

    pub fn categories(&self) -> Vec<Option<ResourceCategory>> {
        self.resources.iter().map(|r| Some(r.category)).collect()
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.resources.iter().position(|r| r.name == name)
    }

    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

/// Read-only pipeline interface for diagnostics and named binding.
#[derive(Debug, Clone, Default)]
pub struct GraphicsPipelineInterface {
    pub vertex_input: Vec<StageIoField>,
    pub payload_links: Vec<(StageIoField, StageIoField)>,
    pub fragment_outputs: Vec<StageIoField>,
    pub resources: PipelineResourceContract,
}

/// Per-stage local-name → pipeline-slot map.
pub type SlotRemap = HashMap<String, u32>;

/// Result of linking a raster pipeline.
#[derive(Debug, Clone)]
pub struct LinkedRasterPipeline {
    pub interface: GraphicsPipelineInterface,
    pub vs_remap: SlotRemap,
    pub fs_remap: SlotRemap,
}

/// Result of linking a mesh pipeline.
#[derive(Debug, Clone)]
pub struct LinkedMeshPipeline {
    pub interface: GraphicsPipelineInterface,
    pub mesh_remap: SlotRemap,
    pub fs_remap: SlotRemap,
    pub amp_remap: SlotRemap,
}

/// Stable fingerprint of a slot remap for in-memory bytecode caches.
pub fn slot_remap_fingerprint(remap: &SlotRemap) -> u64 {
    if remap.is_empty() {
        return 0;
    }
    let mut pairs: Vec<(&str, u32)> = remap.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    let mut h = 0xcbf29ce484222325u64;
    for (name, slot) in pairs {
        for b in name.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= u64::from(slot);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// True when `remap` assigns each locally declared name to its sequential slot.
pub fn remap_is_identity(local_names: &[String], remap: &SlotRemap) -> bool {
    if remap.is_empty() {
        return true;
    }
    local_names
        .iter()
        .enumerate()
        .all(|(i, name)| remap.get(name).copied() == Some(i as u32))
        && remap.len() == local_names.len()
}

// ---------------------------------------------------------------------------
// Source-level payload / resource extraction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct NamedResource {
    name: String,
    ty: String,
    category: ResourceCategory,
    access: ResourceAccess,
    element_type: Option<String>,
}

fn category_and_access(param: &Param) -> Option<(ResourceCategory, ResourceAccess, Option<String>)> {
    match &param.kind {
        ParamKind::Resource => {
            let ty = param.ty.trim();
            if ty.starts_with("Scattered<") {
                Some((
                    ResourceCategory::Scattered,
                    ResourceAccess::ReadWrite,
                    inner_type(ty, "Scattered<"),
                ))
            } else if ty.starts_with("BufRO<") {
                Some((
                    ResourceCategory::Scattered,
                    ResourceAccess::Read,
                    inner_type(ty, "BufRO<"),
                ))
            } else if ty == "ByteAddress" {
                Some((ResourceCategory::Scattered, ResourceAccess::ReadWrite, None))
            } else if ty.starts_with("Interpolated<") {
                Some((
                    ResourceCategory::Texture,
                    ResourceAccess::Read,
                    inner_type(ty, "Interpolated<"),
                ))
            } else if ty.starts_with("DirectSpatial<") {
                Some((
                    ResourceCategory::StorageImage,
                    ResourceAccess::ReadWrite,
                    inner_type(ty, "DirectSpatial<"),
                ))
            } else if ty == "Filter" {
                Some((ResourceCategory::Sampler, ResourceAccess::Read, None))
            } else if ty == "Accel" {
                Some((ResourceCategory::Accel, ResourceAccess::Read, None))
            } else {
                None
            }
        }
        ParamKind::Broadcast => Some((
            ResourceCategory::Broadcast,
            ResourceAccess::Read,
            Some(param.ty.clone()),
        )),
        _ => None,
    }
}

fn inner_type(ty: &str, prefix: &str) -> Option<String> {
    if ty.starts_with(prefix) && ty.ends_with('>') {
        Some(ty[prefix.len()..ty.len() - 1].to_string())
    } else {
        None
    }
}

fn named_resources_of(entry: &EntryDef) -> Vec<NamedResource> {
    flatten_bindless_params(&entry.params)
        .into_iter()
        .filter_map(|p| {
            let (category, access, element_type) = category_and_access(&p)?;
            Some(NamedResource {
                name: p.name,
                ty: p.ty,
                category,
                access,
                element_type,
            })
        })
        .collect()
}

fn types_compatible(a: &str, b: &str) -> bool {
    a.trim() == b.trim()
}

fn merge_named_resources(
    stages: &[(Stage, &str, Vec<NamedResource>)],
    primary: Stage,
) -> Result<(PipelineResourceContract, HashMap<Stage, SlotRemap>)> {
    let mut by_name: HashMap<String, PipelineResource> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    let mut ordered_stages: Vec<&(Stage, &str, Vec<NamedResource>)> = stages.iter().collect();
    ordered_stages.sort_by_key(|(stage, _, _)| if *stage == primary { 0 } else { stage_order(*stage) });

    for (stage, label, resources) in &ordered_stages {
        for res in resources {
            if let Some(existing) = by_name.get_mut(&res.name) {
                if existing.category != res.category
                    || !types_compatible(&existing.ty, &res.ty)
                    || existing.access != res.access
                {
                    bail!(
                        "graphics pipeline resource `{}` is declared incompatibly by {} (`{}` {:?}/{:?}) \
                         and {} (`{}` {:?}/{:?}). Shared names must use the same type, category, and access.",
                        res.name,
                        existing.declaring_stages.join("/"),
                        existing.ty,
                        existing.category,
                        existing.access,
                        label,
                        res.ty,
                        res.category,
                        res.access
                    );
                }
                if !existing.declaring_stages.iter().any(|s| s == label) {
                    existing.declaring_stages.push((*label).to_string());
                }
            } else {
                order.push(res.name.clone());
                by_name.insert(
                    res.name.clone(),
                    PipelineResource {
                        name: res.name.clone(),
                        ty: res.ty.clone(),
                        category: res.category,
                        access: res.access,
                        element_type: res.element_type.clone(),
                        declaring_stages: vec![(*label).to_string()],
                    },
                );
            }
        }
        let _ = stage;
    }

    if order.len() > MAX_BINDLESS_SLOTS {
        let contributors: Vec<String> = ordered_stages
            .iter()
            .map(|(_, label, res)| format!("{label} ({})", res.len()))
            .collect();
        bail!(
            "graphics pipeline bindless contract has {} unique resources (limit is {MAX_BINDLESS_SLOTS}). \
             Contributing stages: {}. Split unused resources out of virtual-main, or share names across stages.",
            order.len(),
            contributors.join(", ")
        );
    }

    let resources: Vec<PipelineResource> = order
        .iter()
        .map(|n| by_name.remove(n).expect("resource recorded in order"))
        .collect();

    let mut remaps: HashMap<Stage, SlotRemap> = HashMap::new();
    for (stage, _, stage_res) in stages {
        let mut map = SlotRemap::new();
        for res in stage_res {
            let slot = resources
                .iter()
                .position(|r| r.name == res.name)
                .expect("merged resource present") as u32;
            map.insert(res.name.clone(), slot);
        }
        remaps.insert(*stage, map);
    }

    Ok((PipelineResourceContract { resources }, remaps))
}

fn stage_order(stage: Stage) -> u8 {
    match stage {
        Stage::Fragment => 1,
        Stage::Vertex => 2,
        Stage::Mesh => 2,
        Stage::Amplification => 3,
        _ => 9,
    }
}

// ---------------------------------------------------------------------------
// Struct-field parsing (source-level, for tests and same-file payloads)
// ---------------------------------------------------------------------------

pub fn parse_struct_fields(source: &str, struct_name: &str) -> Option<Vec<StageIoField>> {
    let needle = format!("struct {struct_name}");
    let mut search = 0;
    let brace = loop {
        let pos = source[search..].find(&needle)? + search;
        if is_in_line_comment(source, pos) {
            search = pos + needle.len();
            continue;
        }
        let after = pos + needle.len();
        let rest = source[after..].trim_start();
        let rel = rest.find('{')?;
        break after + (source[after..].len() - rest.len()) + rel;
    };
    let close = find_matching_brace(source, brace)?;
    let body = &source[brace + 1..close];
    Some(parse_field_list(body, struct_name))
}

fn is_in_line_comment(source: &str, pos: usize) -> bool {
    source[..pos]
        .rfind('\n')
        .map(|n| source[n + 1..pos].contains("//"))
        .unwrap_or(source[..pos].contains("//"))
}

fn find_matching_brace(source: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in source[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_field_list(body: &str, struct_name: &str) -> Vec<StageIoField> {
    let mut fields = Vec::new();
    for raw_line in body.split(';') {
        let line = strip_comments(raw_line).trim().to_string();
        if line.is_empty() {
            continue;
        }
        if let Some(field) = parse_one_field(&line, struct_name) {
            fields.push(field);
        }
    }
    fields
}

fn strip_comments(s: &str) -> String {
    let mut out = String::new();
    for line in s.lines() {
        if let Some(pos) = line.find("//") {
            out.push_str(&line[..pos]);
        } else {
            out.push_str(line);
        }
        out.push(' ');
    }
    out
}

fn parse_one_field(line: &str, struct_name: &str) -> Option<StageIoField> {
    let mut rest = line.trim();
    let mut interpolation = InterpolationMode::Perspective;
    loop {
        if let Some(next) = rest.strip_prefix("nointerpolation") {
            interpolation = InterpolationMode::NoInterpolation;
            rest = next.trim_start();
        } else if let Some(next) = rest.strip_prefix("noperspective") {
            interpolation = InterpolationMode::NoPerspective;
            rest = next.trim_start();
        } else if let Some(next) = rest.strip_prefix("linear") {
            interpolation = InterpolationMode::Linear;
            rest = next.trim_start();
        } else {
            break;
        }
    }
    let (ty, after_ty) = split_ident(rest)?;
    let after_ty = after_ty.trim_start();
    let (name, after_name) = split_ident(after_ty)?;
    let after_name = after_name.trim_start();
    let (semantic, semantic_index) = if let Some(sem) = after_name.strip_prefix(':') {
        parse_semantic(sem.trim())
    } else {
        (name.to_ascii_uppercase(), 0)
    };
    let (scalar_type, vector_size) = parse_value_shape(ty);
    Some(StageIoField {
        field_name: name.to_string(),
        struct_name: struct_name.to_string(),
        semantic,
        semantic_index,
        scalar_type,
        vector_size,
        interpolation,
    })
}

fn split_ident(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return None;
    }
    let mut chars = s.char_indices();
    let (start, c0) = chars.next()?;
    if !c0.is_ascii_alphabetic() && c0 != '_' {
        return None;
    }
    let mut end = start + c0.len_utf8();
    for (i, c) in chars {
        if c.is_ascii_alphanumeric() || c == '_' {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    Some((&s[..end], &s[end..]))
}

pub fn parse_semantic(raw: &str) -> (String, u32) {
    let token = raw.split_whitespace().next().unwrap_or(raw);
    let token = token.trim_end_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
    let bytes = token.as_bytes();
    let mut split = bytes.len();
    while split > 0 && bytes[split - 1].is_ascii_digit() {
        split -= 1;
    }
    if split == 0 || split == bytes.len() {
        return (token.to_ascii_uppercase(), 0);
    }
    let name = token[..split].to_ascii_uppercase();
    let index = token[split..].parse().unwrap_or(0);
    (name, index)
}

pub fn parse_value_shape(ty: &str) -> (String, u32) {
    let ty = ty.trim();
    let prefixes = ["float", "uint", "int", "bool", "half", "double"];
    for prefix in prefixes {
        if ty == prefix {
            return (prefix.to_string(), 1);
        }
        if let Some(rest) = ty.strip_prefix(prefix) {
            if rest.len() == 1 && rest.as_bytes()[0].is_ascii_digit() {
                let n = u32::from(rest.as_bytes()[0] - b'0');
                if (1..=4).contains(&n) {
                    return (prefix.to_string(), n);
                }
            }
        }
    }
    (ty.to_string(), 1)
}

fn is_ia_system_value(field: &StageIoField) -> bool {
    matches!(
        field.semantic.as_str(),
        "SV_VERTEXID" | "SV_INSTANCEID" | "SV_ISFRONTFACE" | "SV_PRIMITIVEID"
    )
}

fn fields_for_type(sources: &[&str], type_name: &str) -> Vec<StageIoField> {
    if type_name.is_empty() || type_name == "void" {
        return Vec::new();
    }
    for src in sources {
        if let Some(fields) = parse_struct_fields(src, type_name) {
            return fields;
        }
    }
    Vec::new()
}

// ---------------------------------------------------------------------------
// Structural payload linking
// ---------------------------------------------------------------------------

pub fn link_payload_fields(
    producer_stage: &str,
    consumer_stage: &str,
    producer: &[StageIoField],
    consumer: &[StageIoField],
) -> Result<Vec<(StageIoField, StageIoField)>> {
    if producer.is_empty() && consumer.is_empty() {
        return Ok(Vec::new());
    }
    let mut by_sem: HashMap<(String, u32), &StageIoField> = HashMap::new();
    for field in producer {
        by_sem.entry(field.match_key()).or_insert(field);
    }
    let mut links = Vec::new();
    let mut errors = Vec::new();
    for cons in consumer {
        match by_sem.get(&cons.match_key()) {
            None => errors.push(format!(
                "{consumer_stage} input {} has no matching {producer_stage} output semantic {}{}",
                cons.describe(),
                cons.semantic,
                cons.semantic_index
            )),
            Some(prod) => {
                if !prod.shape_compatible(cons) {
                    let prod_ty = format_shape(prod);
                    let cons_ty = format_shape(cons);
                    errors.push(format!(
                        "{consumer_stage} input {} expected {cons_ty} from {producer_stage} output {}, but found {prod_ty}",
                        cons.describe(),
                        prod.describe()
                    ));
                } else if !prod.interpolation.is_compatible_with(cons.interpolation) {
                    errors.push(format!(
                        "{consumer_stage} input {} interpolation `{}` does not match {producer_stage} output {} (`{}`)",
                        cons.describe(),
                        cons.interpolation.name(),
                        prod.describe(),
                        prod.interpolation.name()
                    ));
                } else {
                    links.push(((*prod).clone(), cons.clone()));
                }
            }
        }
    }
    if !errors.is_empty() {
        bail!(
            "graphics pipeline payload link failed ({producer_stage} → {consumer_stage}):\n  {}",
            errors.join("\n  ")
        );
    }
    Ok(links)
}

fn format_shape(field: &StageIoField) -> String {
    if field.vector_size <= 1 {
        field.scalar_type.clone()
    } else {
        format!("{}{}", field.scalar_type, field.vector_size)
    }
}

// ---------------------------------------------------------------------------
// IA validation
// ---------------------------------------------------------------------------

fn vertex_format_compatible(format: VertexFormat, field: &StageIoField) -> bool {
    let scalar = field.scalar_type.as_str();
    match (format, field.vector_size) {
        (VertexFormat::Float32, 1) => scalar == "float",
        (VertexFormat::Float32x2, 2) => scalar == "float",
        (VertexFormat::Float32x3, 3) => scalar == "float",
        (VertexFormat::Float32x4, 4) => scalar == "float",
        (VertexFormat::Uint32, 1) => scalar == "uint",
        (VertexFormat::Sint32, 1) => scalar == "int",
        (VertexFormat::Uint8x4 | VertexFormat::Unorm8x4, 4) => scalar == "float" || scalar == "uint" || scalar == "int",
        _ => false,
    }
}

pub fn validate_vertex_layout(layout: &VertexBufferLayout, vertex_inputs: &[StageIoField]) -> Result<()> {
    let required: Vec<&StageIoField> = vertex_inputs.iter().filter(|f| !is_ia_system_value(f)).collect();
    if required.is_empty() {
        return Ok(());
    }
    if layout.attributes.is_empty() {
        let names: Vec<String> = required.iter().map(|f| f.describe()).collect();
        bail!(
            "vertex stage expects input attributes [{}] but RenderPipelineDesc.vertex_layout is empty. \
             Provide a VertexBufferLayout (or GpuType::vertex_buffer_layout) that matches those semantics.",
            names.join(", ")
        );
    }
    let mut attrs: Vec<_> = layout.attributes.iter().collect();
    attrs.sort_by_key(|a| a.location);
    let mut errors = Vec::new();
    for (i, field) in required.iter().enumerate() {
        let Some(attr) = attrs.get(i) else {
            errors.push(format!(
                "vertex input {} has no matching vertex-layout attribute (need location {i})",
                field.describe()
            ));
            continue;
        };
        if !vertex_format_compatible(attr.format, field) {
            errors.push(format!(
                "vertex input {} expected a {}{} layout format, but location {} is {:?}",
                field.describe(),
                field.scalar_type,
                if field.vector_size <= 1 {
                    String::new()
                } else {
                    field.vector_size.to_string()
                },
                attr.location,
                attr.format
            ));
        }
    }
    if !errors.is_empty() {
        bail!(
            "vertex input layout does not match the vertex stage:\n  {}",
            errors.join("\n  ")
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public link entry points
// ---------------------------------------------------------------------------

fn entry_for_stage(source: &str, stage: Stage) -> Option<EntryDef> {
    find_all_entries(source).into_iter().find(|e| e.stage == stage)
}

fn payload_outputs_for(entry: &EntryDef, sources: &[&str]) -> Vec<StageIoField> {
    match entry.stage {
        Stage::Vertex => {
            let ty = entry.return_type.trim();
            fields_for_type(sources, ty)
        }
        Stage::Mesh => mesh_vertices_type(entry)
            .map(|ty| fields_for_type(sources, &ty))
            .unwrap_or_default(),
        Stage::Amplification => mesh_payload_type(entry, true)
            .map(|ty| fields_for_type(sources, &ty))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn payload_inputs_for(entry: &EntryDef, sources: &[&str]) -> Vec<StageIoField> {
    match entry.stage {
        Stage::Fragment | Stage::Vertex => stage_input_payload_type(entry)
            .map(|ty| fields_for_type(sources, &ty))
            .unwrap_or_default(),
        Stage::Mesh => mesh_payload_type(entry, false)
            .map(|ty| fields_for_type(sources, &ty))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn fragment_outputs_for(entry: &EntryDef) -> Vec<StageIoField> {
    if entry.stage != Stage::Fragment {
        return Vec::new();
    }
    let ty = entry.return_type.trim();
    if ty.is_empty() || ty == "void" {
        return Vec::new();
    }
    let (scalar_type, vector_size) = parse_value_shape(ty);
    let (semantic, semantic_index) = entry
        .return_semantic
        .as_deref()
        .map(parse_semantic)
        .unwrap_or_else(|| ("SV_TARGET".to_string(), 0));
    vec![StageIoField {
        field_name: "color".into(),
        struct_name: ty.to_string(),
        semantic,
        semantic_index,
        scalar_type,
        vector_size,
        interpolation: InterpolationMode::Perspective,
    }]
}

fn link_or_same_type(
    producer_stage: &str,
    consumer_stage: &str,
    producer_ty: Option<&str>,
    consumer_ty: Option<&str>,
    producer_fields: &[StageIoField],
    consumer_fields: &[StageIoField],
) -> Result<Vec<(StageIoField, StageIoField)>> {
    if !producer_fields.is_empty() || !consumer_fields.is_empty() {
        return link_payload_fields(producer_stage, consumer_stage, producer_fields, consumer_fields);
    }
    match (producer_ty, consumer_ty) {
        (Some(a), Some(b)) if a == b => Ok(Vec::new()),
        (Some(a), Some(b)) => {
            bail!(
                "graphics pipeline payload link failed ({producer_stage} → {consumer_stage}): \
                 producer writes `{a}` but consumer reads `{b}`, and neither type's fields were \
                 visible in the shader source. Define the payload struct once (for example in a \
                 shared module) or give both structs matching semantics / shapes."
            )
        }
        (None, Some(b)) => bail!(
            "graphics pipeline payload link failed ({producer_stage} → {consumer_stage}): \
             {consumer_stage} expects `{b}` but {producer_stage} has no payload output"
        ),
        (Some(_), None) => Ok(Vec::new()),
        (None, None) => Ok(Vec::new()),
    }
}

/// Link a vertex + fragment pipeline from authored Slang sources.
pub fn link_raster_pipeline(
    vs_source: &str,
    fs_source: &str,
    vertex_layout: Option<&VertexBufferLayout>,
) -> Result<Option<LinkedRasterPipeline>> {
    let vs = entry_for_stage(vs_source, Stage::Vertex);
    let fs = entry_for_stage(fs_source, Stage::Fragment);
    if vs.is_none() && fs.is_none() {
        return Ok(None);
    }
    let vs = vs.context("raster pipeline is missing a [goldy_vertex] entry point")?;
    let fs = fs.context("raster pipeline is missing a [goldy_fragment] entry point")?;

    let sources = [vs_source, fs_source];
    let vs_out = payload_outputs_for(&vs, &sources);
    let fs_in = payload_inputs_for(&fs, &sources);
    let vs_in = payload_inputs_for(&vs, &sources);
    let fs_out = fragment_outputs_for(&fs);

    let vs_out_ty = Some(vs.return_type.trim());
    let fs_in_ty = stage_input_payload_type(&fs);
    let payload_links = link_or_same_type("vertex", "fragment", vs_out_ty, fs_in_ty.as_deref(), &vs_out, &fs_in)?;

    if let Some(layout) = vertex_layout {
        validate_vertex_layout(layout, &vs_in)?;
    }

    let stages = [
        (Stage::Fragment, "fragment", named_resources_of(&fs)),
        (Stage::Vertex, "vertex", named_resources_of(&vs)),
    ];
    let (resources, remaps) = merge_named_resources(&stages, Stage::Fragment)?;

    Ok(Some(LinkedRasterPipeline {
        interface: GraphicsPipelineInterface {
            vertex_input: vs_in,
            payload_links,
            fragment_outputs: fs_out,
            resources,
        },
        vs_remap: remaps.get(&Stage::Vertex).cloned().unwrap_or_default(),
        fs_remap: remaps.get(&Stage::Fragment).cloned().unwrap_or_default(),
    }))
}

/// Link a mesh (+ optional amplification) + fragment pipeline from authored sources.
pub fn link_mesh_pipeline(
    mesh_source: &str,
    fs_source: &str,
    amp_source: Option<&str>,
) -> Result<Option<LinkedMeshPipeline>> {
    let mesh = entry_for_stage(mesh_source, Stage::Mesh);
    let fs = entry_for_stage(fs_source, Stage::Fragment);
    if mesh.is_none() && fs.is_none() {
        return Ok(None);
    }
    let mesh = mesh.context("mesh pipeline is missing a [goldy_mesh] entry point")?;
    let fs = fs.context("mesh pipeline is missing a [goldy_fragment] entry point")?;
    let amp = amp_source.and_then(|src| entry_for_stage(src, Stage::Amplification));

    let mut sources = vec![mesh_source, fs_source];
    if let Some(src) = amp_source {
        sources.push(src);
    }
    let src_refs: Vec<&str> = sources.iter().copied().collect();

    let mesh_out = payload_outputs_for(&mesh, &src_refs);
    let fs_in = payload_inputs_for(&fs, &src_refs);
    let fs_out = fragment_outputs_for(&fs);
    let mesh_out_ty = mesh_vertices_type(&mesh);
    let fs_in_ty = stage_input_payload_type(&fs);
    let payload_links = link_or_same_type(
        "mesh",
        "fragment",
        mesh_out_ty.as_deref(),
        fs_in_ty.as_deref(),
        &mesh_out,
        &fs_in,
    )?;

    if let Some(ref amp_entry) = amp {
        let amp_out = payload_outputs_for(amp_entry, &src_refs);
        let mesh_in = payload_inputs_for(&mesh, &src_refs);
        let amp_ty = mesh_payload_type(amp_entry, true);
        let mesh_ty = mesh_payload_type(&mesh, false);
        link_or_same_type(
            "amplification",
            "mesh",
            amp_ty.as_deref(),
            mesh_ty.as_deref(),
            &amp_out,
            &mesh_in,
        )?;
    }

    let mut stages = vec![
        (Stage::Mesh, "mesh", named_resources_of(&mesh)),
        (Stage::Fragment, "fragment", named_resources_of(&fs)),
    ];
    if let Some(ref amp_entry) = amp {
        stages.push((Stage::Amplification, "amplification", named_resources_of(amp_entry)));
    }
    let (resources, remaps) = merge_named_resources(&stages, Stage::Mesh)?;

    Ok(Some(LinkedMeshPipeline {
        interface: GraphicsPipelineInterface {
            vertex_input: Vec::new(),
            payload_links,
            fragment_outputs: fs_out,
            resources,
        },
        mesh_remap: remaps.get(&Stage::Mesh).cloned().unwrap_or_default(),
        fs_remap: remaps.get(&Stage::Fragment).cloned().unwrap_or_default(),
        amp_remap: remaps.get(&Stage::Amplification).cloned().unwrap_or_default(),
    }))
}

/// Refine a source-parsed payload link with compiled Slang reflection.
pub fn refine_payload_link(
    producer_stage: &str,
    consumer_stage: &str,
    producer: &StageInterface,
    consumer: &StageInterface,
) -> Result<Vec<(StageIoField, StageIoField)>> {
    let prod = if producer.payload_outputs.is_empty() {
        &producer.fragment_outputs
    } else {
        &producer.payload_outputs
    };
    let cons = if consumer.payload_inputs.is_empty() {
        &consumer.vertex_inputs
    } else {
        &consumer.payload_inputs
    };
    if prod.is_empty() && cons.is_empty() {
        return Ok(Vec::new());
    }
    link_payload_fields(producer_stage, consumer_stage, prod, cons)
}

/// Names in the merged contract that a pass-level named binding set is missing.
pub fn missing_required_bindings(contract: &PipelineResourceContract, provided: &HashSet<&str>) -> Vec<String> {
    contract
        .resources
        .iter()
        .filter(|r| !provided.contains(r.name.as_str()))
        .map(|r| r.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VertexAttribute;

    const VS_FS: &str = r#"
import goldy_exp;

struct VertIn {
    float3 pos : POSITION;
    float2 uv  : TEXCOORD0;
};

struct Varying {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float  extra    : TEXCOORD1;
};

[goldy_vertex]
Varying vs_main(SceneUniforms scene, VertIn input) {
    Varying o;
    o.position = float4(input.pos, 1);
    o.uv = input.uv;
    o.extra = 1.0;
    return o;
}

[goldy_fragment]
float4 fs_main(Interpolated<float4> tex, Filter smp, Varying input) : SV_Target {
    return tex.Sample(smp, input.uv);
}
"#;

    #[test]
    fn same_struct_name_links() {
        let linked = link_raster_pipeline(VS_FS, VS_FS, None).unwrap().unwrap();
        assert_eq!(linked.interface.payload_links.len(), 3);
        assert_eq!(linked.interface.resources.resources.len(), 3);
        // Fragment-first contract: tex, smp, then unique VS `scene`.
        assert_eq!(linked.interface.resources.resources[0].name, "tex");
        assert_eq!(linked.interface.resources.resources[1].name, "smp");
        assert_eq!(linked.interface.resources.resources[2].name, "scene");
        assert_eq!(linked.fs_remap.get("tex").copied(), Some(0));
        assert_eq!(linked.fs_remap.get("smp").copied(), Some(1));
        assert_eq!(linked.vs_remap.get("scene").copied(), Some(2));
        assert!(!linked.vs_remap.contains_key("tex"));
    }

    #[test]
    fn different_struct_names_matching_semantics() {
        let vs = r#"
struct VertIn { float2 pos : POSITION; };
struct VsOut { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[goldy_vertex]
VsOut vs_main(VertIn input) { VsOut o; return o; }
"#;
        let fs = r#"
struct FsIn { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[goldy_fragment]
float4 fs_main(FsIn input) : SV_Target { return float4(input.uv, 0, 1); }
"#;
        let linked = link_raster_pipeline(vs, fs, None).unwrap().unwrap();
        assert_eq!(linked.interface.payload_links.len(), 2);
    }

    #[test]
    fn producer_superset_is_ok() {
        let vs = r#"
struct VertIn { float2 pos : POSITION; };
struct VsOut { float4 position : SV_Position; float2 uv : TEXCOORD0; float extra : TEXCOORD1; };
[goldy_vertex]
VsOut vs_main(VertIn input) { VsOut o; return o; }
"#;
        let fs = r#"
struct FsIn { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[goldy_fragment]
float4 fs_main(FsIn input) : SV_Target { return float4(1,0,0,1); }
"#;
        link_raster_pipeline(vs, fs, None).unwrap().unwrap();
    }

    #[test]
    fn missing_semantic_errors() {
        let vs = r#"
struct VertIn { float2 pos : POSITION; };
struct VsOut { float4 position : SV_Position; };
[goldy_vertex]
VsOut vs_main(VertIn input) { VsOut o; return o; }
"#;
        let fs = r#"
struct FsIn { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[goldy_fragment]
float4 fs_main(FsIn input) : SV_Target { return float4(1,0,0,1); }
"#;
        let err = link_raster_pipeline(vs, fs, None).unwrap_err().to_string();
        assert!(err.contains("TEXCOORD"), "{err}");
        assert!(err.contains("vertex"), "{err}");
        assert!(err.contains("fragment"), "{err}");
    }

    #[test]
    fn scalar_vector_mismatch_errors() {
        let vs = r#"
struct VertIn { float2 pos : POSITION; };
struct VsOut { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[goldy_vertex]
VsOut vs_main(VertIn input) { VsOut o; return o; }
"#;
        let fs = r#"
struct FsIn { float4 position : SV_Position; float uv : TEXCOORD0; };
[goldy_fragment]
float4 fs_main(FsIn input) : SV_Target { return float4(1,0,0,1); }
"#;
        let err = link_raster_pipeline(vs, fs, None).unwrap_err().to_string();
        assert!(err.contains("float2") && err.contains("float"), "{err}");
    }

    #[test]
    fn interpolation_mismatch_errors() {
        let vs = r#"
struct VertIn { float2 pos : POSITION; };
struct VsOut { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[goldy_vertex]
VsOut vs_main(VertIn input) { VsOut o; return o; }
"#;
        let fs = r#"
struct FsIn { float4 position : SV_Position; nointerpolation float2 uv : TEXCOORD0; };
[goldy_fragment]
float4 fs_main(FsIn input) : SV_Target { return float4(1,0,0,1); }
"#;
        let err = link_raster_pipeline(vs, fs, None).unwrap_err().to_string();
        assert!(
            err.contains("nointerpolation") || err.contains("interpolation"),
            "{err}"
        );
    }

    #[test]
    fn shared_name_conflict_errors() {
        let vs = r#"
struct VertIn { float2 pos : POSITION; };
struct Varying { float4 position : SV_Position; };
[goldy_vertex]
Varying vs_main(SceneUniforms scene, VertIn input) { Varying o; return o; }
"#;
        let fs = r#"
struct Varying { float4 position : SV_Position; };
[goldy_fragment]
float4 fs_main(BufRO<uint> scene, Varying input) : SV_Target { return float4(1,0,0,1); }
"#;
        let err = link_raster_pipeline(vs, fs, None).unwrap_err().to_string();
        assert!(err.contains("scene"), "{err}");
        assert!(err.contains("incompatibly") || err.contains("incompatible"), "{err}");
    }

    #[test]
    fn mesh_fragment_link() {
        let src = r#"
struct MeshOut { float4 pos : SV_Position; float4 color : COLOR; };
struct FsIn { float4 pos : SV_Position; float4 color : COLOR; };
[goldy_mesh]
[numthreads(1,1,1)]
[outputtopology("triangle")]
void mesh_main(out vertices MeshOut verts[3], out indices uint3 tris[1]) {}
[goldy_fragment]
float4 fs_main(FsIn input) : SV_Target { return input.color; }
"#;
        let linked = link_mesh_pipeline(src, src, None).unwrap().unwrap();
        assert_eq!(linked.interface.payload_links.len(), 2);
    }

    #[test]
    fn amplification_payload_mismatch() {
        let amp = r#"
struct AmpOut { uint count; };
[goldy_amplification]
[numthreads(1,1,1)]
void amp_main(out payload AmpOut p) { p.count = 1; }
"#;
        let mesh = r#"
struct Other { float x; };
struct MeshOut { float4 pos : SV_Position; };
[goldy_mesh]
[numthreads(1,1,1)]
[outputtopology("triangle")]
void mesh_main(in payload Other p, out vertices MeshOut verts[1], out indices uint3 tris[1]) {}
"#;
        let fs = r#"
struct MeshOut { float4 pos : SV_Position; };
[goldy_fragment]
float4 fs_main(MeshOut input) : SV_Target { return float4(1,0,0,1); }
"#;
        let err = link_mesh_pipeline(mesh, fs, Some(amp)).unwrap_err().to_string();
        assert!(
            err.contains("amplification") || err.contains("payload") || err.contains("AmpOut"),
            "{err}"
        );
    }

    #[test]
    fn ia_layout_mismatch() {
        let src = VS_FS;
        let layout = VertexBufferLayout {
            stride: 8,
            attributes: vec![VertexAttribute {
                location: 0,
                format: VertexFormat::Float32x2,
                offset: 0,
            }],
        };
        let err = link_raster_pipeline(src, src, Some(&layout)).unwrap_err().to_string();
        assert!(err.contains("vertex input") || err.contains("POSITION"), "{err}");
    }

    #[test]
    fn parse_semantic_splits_index() {
        assert_eq!(parse_semantic("TEXCOORD0"), ("TEXCOORD".into(), 0));
        assert_eq!(parse_semantic("TEXCOORD1"), ("TEXCOORD".into(), 1));
        assert_eq!(parse_semantic("SV_Position"), ("SV_POSITION".into(), 0));
        assert_eq!(parse_semantic("COLOR"), ("COLOR".into(), 0));
    }

    #[test]
    fn named_missing_and_extra_bindings() {
        let linked = link_raster_pipeline(VS_FS, VS_FS, None).unwrap().unwrap();
        let contract = &linked.interface.resources;
        let provided: HashSet<&str> = ["tex", "smp", "scene", "unused_extra"].into_iter().collect();
        assert!(missing_required_bindings(contract, &provided).is_empty());
        let missing = missing_required_bindings(contract, &["tex"].into_iter().collect());
        assert!(missing.iter().any(|n| n == "smp" || n == "scene"), "{missing:?}");
    }

    #[test]
    fn positional_fragment_first_then_unique_vertex() {
        let linked = link_raster_pipeline(VS_FS, VS_FS, None).unwrap().unwrap();
        let names: Vec<&str> = linked
            .interface
            .resources
            .resources
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(names, vec!["tex", "smp", "scene"]);
    }

    #[test]
    fn remap_identity_when_fragment_declares_all() {
        let src = r#"
struct VertIn { float2 pos : POSITION; };
struct Varying { float4 position : SV_Position; };
[goldy_vertex]
Varying vs_main(SceneUniforms scene, VertIn input) { Varying o; return o; }
[goldy_fragment]
float4 fs_main(SceneUniforms scene, Varying input) : SV_Target { return float4(1,0,0,1); }
"#;
        let linked = link_raster_pipeline(src, src, None).unwrap().unwrap();
        assert_eq!(linked.vs_remap.get("scene").copied(), Some(0));
        assert_eq!(linked.fs_remap.get("scene").copied(), Some(0));
        assert!(remap_is_identity(&["scene".into()], &linked.fs_remap));
        assert!(remap_is_identity(&["scene".into()], &linked.vs_remap));
    }

    #[test]
    fn slot_remap_fingerprint_separates_maps() {
        let mut a = SlotRemap::new();
        a.insert("scene".into(), 0);
        let mut b = SlotRemap::new();
        b.insert("scene".into(), 2);
        assert_ne!(slot_remap_fingerprint(&a), slot_remap_fingerprint(&b));
    }
}
