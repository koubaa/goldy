//! Structured virtual-kernel ABI metadata.
//!
//! This is the central descriptor shared by Rust proc-macro codegen and
//! virtual-main wrapper emission. It carries enough information that Goldy
//! does not need to re-parse generated Slang to bind Scheme parameters.

use std::collections::HashMap;
use std::fmt;

/// Bump when the wire layout or parameter classification changes.
pub const KERNEL_ABI_VERSION: u32 = 3;

/// Hidden structured-buffer parameter that packs every tensor layout for one dispatch.
pub const TENSOR_META_PARAM: &str = "_goldy_tensor_meta";

/// Slang struct name for [`TENSOR_META_PARAM`] elements.
pub const TENSOR_LAYOUT_SLANG: &str = "GoldyTensorLayout";

/// Host/device stride of [`TENSOR_LAYOUT_SLANG`] (`12` `uint`s, 16-byte aligned).
pub const TENSOR_LAYOUT_STRIDE_BYTES: u32 = 48;

/// Bitflags for hidden builtins injected into the generated Slang signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BuiltinMask {
    pub global_id: bool,
    pub local_id: bool,
    pub workgroup_id: bool,
}

impl BuiltinMask {
    pub const NONE: Self = Self {
        global_id: false,
        local_id: false,
        workgroup_id: false,
    };

    pub fn is_empty(self) -> bool {
        !self.global_id && !self.local_id && !self.workgroup_id
    }
}

/// Logical scalar types supported on the push-constant scalar ABI (MVP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    U32,
    I32,
    F32,
    Bool,
}

impl ScalarType {
    pub fn slang_name(self) -> &'static str {
        match self {
            Self::U32 => "uint",
            Self::I32 => "int",
            Self::F32 => "float",
            Self::Bool => "bool",
        }
    }

    pub fn rust_name(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
        }
    }

    /// Number of `u32` push words occupied (MVP: always 1).
    pub fn word_count(self) -> u32 {
        1
    }
}

/// Element type for buffer slice parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    U32,
    I32,
    F32,
    Bool,
}

impl ElementType {
    pub fn slang_name(self) -> &'static str {
        match self {
            Self::U32 => "uint",
            Self::I32 => "int",
            Self::F32 => "float",
            Self::Bool => "bool",
        }
    }

    pub fn rust_name(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
        }
    }

    pub fn stride_bytes(self) -> u32 {
        4
    }
}

/// Resource / scalar category for one logical kernel parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamCategory {
    /// `BufRO<T>` / read-only structured buffer.
    BufferRead,
    /// `Scattered<T>` with read+write access (from `&mut [T]`).
    BufferReadWrite,
    /// `Scattered<T>` write-only (`gpu::Scattered<T>`).
    BufferWrite,
    /// Explicit `gpu::Uniform<T>` broadcast resource.
    Uniform,
    /// `DirectSpatial<T>` (`gpu::DirectSpatial<T>`).
    StorageImage,
    /// Typed scalar push word.
    Scalar,
}

impl ParamCategory {
    pub fn is_resource(self) -> bool {
        !matches!(self, Self::Scalar)
    }

    /// Slang resource type string for this category (without element).
    pub fn slang_resource_wrapper(self, element_slang: &str) -> String {
        match self {
            Self::BufferRead => format!("BufRO<{element_slang}>"),
            Self::BufferReadWrite | Self::BufferWrite => format!("Scattered<{element_slang}>"),
            Self::Uniform => element_slang.to_string(),
            Self::StorageImage => format!("DirectSpatial<{element_slang}>"),
            Self::Scalar => unreachable!("scalar params are not resource wrappers"),
        }
    }
}

/// Scheme graph access implied by a resource parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
    ReadWrite,
}

impl AccessKind {
    pub fn for_category(category: ParamCategory) -> Option<Self> {
        match category {
            ParamCategory::BufferRead | ParamCategory::Uniform => Some(Self::Read),
            ParamCategory::BufferWrite | ParamCategory::StorageImage => Some(Self::Write),
            ParamCategory::BufferReadWrite => Some(Self::ReadWrite),
            ParamCategory::Scalar => None,
        }
    }
}

/// Maximum rank a [`TensorShapeSpec`] may name. Matches Goldy's dense tensor rank.
pub const TENSOR_SHAPE_SPEC_MAX_RANK: usize = 4;

/// One axis in a host-only tensor shape contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorDimSpec {
    /// `_`: any extent at this axis.
    Any,
    /// Exact extent, e.g. `4`.
    Exact(u32),
    /// Symbolic name; repeated names in one `record` call must match.
    Symbol(String),
}

impl fmt::Display for TensorDimSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => f.write_str("_"),
            Self::Exact(n) => write!(f, "{n}"),
            Self::Symbol(name) => f.write_str(name),
        }
    }
}

/// Rank-fixing tensor shape contract (`[vocab, dim]`, `[4, _]`, …).
///
/// Host-only: shader parameter order and the 48-byte GPU layout are unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorShapeSpec {
    pub dims: Vec<TensorDimSpec>,
}

impl TensorShapeSpec {
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Bind `actual` against this spec, unifying symbolic names into `env`.
    pub fn check(
        &self,
        kernel: &str,
        param: &str,
        actual: &[u32],
        env: &mut HashMap<String, BoundTensorDim>,
    ) -> Result<(), String> {
        if actual.len() != self.dims.len() {
            return Err(format!(
                "kernel `{kernel}` parameter `{param}`: expected rank {} {}, got rank {} [{}]",
                self.dims.len(),
                self,
                actual.len(),
                format_extents(actual),
            ));
        }
        for (axis, (spec, &got)) in self.dims.iter().zip(actual.iter()).enumerate() {
            match spec {
                TensorDimSpec::Any => {}
                TensorDimSpec::Exact(n) => {
                    if got != *n {
                        return Err(format!(
                            "kernel `{kernel}` parameter `{param}` axis {axis}: expected {n}, got {got} (shape [{}])",
                            format_extents(actual),
                        ));
                    }
                }
                TensorDimSpec::Symbol(name) => {
                    if let Some(bound) = env.get(name) {
                        if bound.extent != got {
                            return Err(format!(
                                "kernel `{kernel}` parameter `{param}` axis {axis}: expected `{name}`={} (from `{}` axis {}), got {got} (shape [{}])",
                                bound.extent,
                                bound.from_param,
                                bound.axis,
                                format_extents(actual),
                            ));
                        }
                    } else {
                        env.insert(
                            name.clone(),
                            BoundTensorDim {
                                extent: got,
                                from_param: param.to_string(),
                                axis,
                            },
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for TensorShapeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}

fn format_extents(dims: &[u32]) -> String {
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// First binding of a symbolic contract dimension within one `record` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTensorDim {
    pub extent: u32,
    pub from_param: String,
    pub axis: usize,
}

/// One logical kernel parameter (declaration order; builtins are separate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelParam {
    pub name: String,
    pub category: ParamCategory,
    pub access: Option<AccessKind>,
    pub scalar: Option<ScalarType>,
    /// Slang element / broadcast type name (`uint`, `float`, `Particle`, …).
    pub slang_type: String,
    /// Expected structured-buffer stride in bytes, when applicable.
    pub stride_bytes: Option<u32>,
    /// Logical tensor view: indexing is view-relative; a packed metadata parcel follows user resources.
    pub is_tensor: bool,
    /// Optional host-only rank/extent contract. `None` keeps any-shape recording.
    pub shape_spec: Option<TensorShapeSpec>,
}

impl KernelParam {
    pub fn buffer_read(name: impl Into<String>, element: ElementType) -> Self {
        Self {
            name: name.into(),
            category: ParamCategory::BufferRead,
            access: Some(AccessKind::Read),
            scalar: None,
            slang_type: element.slang_name().to_string(),
            stride_bytes: Some(element.stride_bytes()),
            is_tensor: false,
            shape_spec: None,
        }
    }

    pub fn buffer_read_write(name: impl Into<String>, element: ElementType) -> Self {
        Self {
            name: name.into(),
            category: ParamCategory::BufferReadWrite,
            access: Some(AccessKind::ReadWrite),
            scalar: None,
            slang_type: element.slang_name().to_string(),
            stride_bytes: Some(element.stride_bytes()),
            is_tensor: false,
            shape_spec: None,
        }
    }

    pub fn buffer_read_named(name: impl Into<String>, slang_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            category: ParamCategory::BufferRead,
            access: Some(AccessKind::Read),
            scalar: None,
            slang_type: slang_type.into(),
            stride_bytes: None,
            is_tensor: false,
            shape_spec: None,
        }
    }

    pub fn storage_image(name: impl Into<String>, slang_element: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            category: ParamCategory::StorageImage,
            access: Some(AccessKind::Write),
            scalar: None,
            slang_type: slang_element.into(),
            stride_bytes: None,
            is_tensor: false,
            shape_spec: None,
        }
    }

    pub fn buffer_write(name: impl Into<String>, element: ElementType) -> Self {
        Self {
            name: name.into(),
            category: ParamCategory::BufferWrite,
            access: Some(AccessKind::Write),
            scalar: None,
            slang_type: element.slang_name().to_string(),
            stride_bytes: Some(element.stride_bytes()),
            is_tensor: false,
            shape_spec: None,
        }
    }

    pub fn tensor_read(name: impl Into<String>, element: ElementType) -> Self {
        let mut p = Self::buffer_read(name, element);
        p.is_tensor = true;
        p
    }

    pub fn tensor_read_write(name: impl Into<String>, element: ElementType) -> Self {
        let mut p = Self::buffer_read_write(name, element);
        p.is_tensor = true;
        p
    }

    pub fn tensor_write(name: impl Into<String>, element: ElementType) -> Self {
        let mut p = Self::buffer_write(name, element);
        p.is_tensor = true;
        p
    }

    pub fn tensor_meta() -> Self {
        Self {
            name: TENSOR_META_PARAM.to_string(),
            category: ParamCategory::BufferRead,
            access: Some(AccessKind::Read),
            scalar: None,
            slang_type: TENSOR_LAYOUT_SLANG.to_string(),
            stride_bytes: Some(TENSOR_LAYOUT_STRIDE_BYTES),
            is_tensor: false,
            shape_spec: None,
        }
    }

    pub fn scalar_param(name: impl Into<String>, ty: ScalarType) -> Self {
        Self {
            name: name.into(),
            category: ParamCategory::Scalar,
            access: None,
            scalar: Some(ty),
            slang_type: ty.slang_name().to_string(),
            stride_bytes: None,
            is_tensor: false,
            shape_spec: None,
        }
    }

    pub fn slang_param_type(&self) -> String {
        match self.category {
            ParamCategory::Scalar => self.slang_type.clone(),
            other => other.slang_resource_wrapper(&self.slang_type),
        }
    }
}

/// Maps generated Slang lines back to the originating Rust source.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceMap {
    pub rust_file: String,
    pub rust_line: u32,
}

/// Canonical shader text plus structured ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSource {
    /// Portable `[goldy_compute]` Slang (before backend-specific virtual-main lowering).
    pub canonical_slang: String,
}

/// Full prepare-time kernel descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelDef {
    pub source: KernelSource,
    pub entry: String,
    pub workgroup_size: [u32; 3],
    pub params: Vec<KernelParam>,
    pub builtins: BuiltinMask,
    pub source_map: SourceMap,
    pub abi_version: u32,
}

impl KernelDef {
    pub fn new(
        canonical_slang: impl Into<String>,
        entry: impl Into<String>,
        workgroup_size: [u32; 3],
        params: Vec<KernelParam>,
        builtins: BuiltinMask,
        source_map: SourceMap,
    ) -> Self {
        Self {
            source: KernelSource {
                canonical_slang: canonical_slang.into(),
            },
            entry: entry.into(),
            workgroup_size,
            params,
            builtins,
            source_map,
            abi_version: KERNEL_ABI_VERSION,
        }
    }

    pub fn resource_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.params.iter().filter(|p| p.category.is_resource())
    }

    pub fn scalar_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.params
            .iter()
            .filter(|p| matches!(p.category, ParamCategory::Scalar))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(dims: Vec<TensorDimSpec>) -> TensorShapeSpec {
        TensorShapeSpec { dims }
    }

    #[test]
    fn wildcard_accepts_any_extent() {
        let mut env = HashMap::new();
        spec(vec![TensorDimSpec::Any, TensorDimSpec::Exact(4)])
            .check("k", "x", &[99, 4], &mut env)
            .unwrap();
        assert!(env.is_empty());
    }

    #[test]
    fn exact_extent_mismatch() {
        let mut env = HashMap::new();
        let err = spec(vec![TensorDimSpec::Exact(4)])
            .check("k", "x", &[8], &mut env)
            .unwrap_err();
        assert!(err.contains("parameter `x` axis 0: expected 4, got 8"), "{err}");
    }

    #[test]
    fn wrong_rank() {
        let mut env = HashMap::new();
        let err = spec(vec![TensorDimSpec::Symbol("dim".into())])
            .check("rmsnorm", "x", &[2, 4], &mut env)
            .unwrap_err();
        assert!(err.contains("expected rank 1 [dim], got rank 2 [2, 4]"), "{err}");
    }

    #[test]
    fn symbolic_equality_across_params() {
        let mut env = HashMap::new();
        let dim = spec(vec![TensorDimSpec::Symbol("dim".into())]);
        dim.check("rms", "x", &[4], &mut env).unwrap();
        dim.check("rms", "weight", &[4], &mut env).unwrap();
        let err = dim.check("rms", "o", &[8], &mut env).unwrap_err();
        assert!(
            err.contains("expected `dim`=4 (from `x` axis 0), got 8"),
            "{err}"
        );
    }

    #[test]
    fn empty_spec_is_rank_zero() {
        let mut env = HashMap::new();
        spec(vec![]).check("k", "s", &[], &mut env).unwrap();
        let err = spec(vec![]).check("k", "s", &[1], &mut env).unwrap_err();
        assert!(err.contains("expected rank 0 []"), "{err}");
    }
}
