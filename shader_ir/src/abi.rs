//! Structured virtual-kernel ABI metadata.
//!
//! This is the central descriptor shared by Rust proc-macro codegen and
//! virtual-main wrapper emission. It carries enough information that Goldy
//! does not need to re-parse generated Slang to bind Scheme parameters.

/// Bump when the wire layout or parameter classification changes.
pub const KERNEL_ABI_VERSION: u32 = 1;

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
    /// `Scattered<T>` write-only (`gpu::Out<T>`).
    BufferWrite,
    /// Explicit `gpu::Uniform<T>` broadcast resource.
    Uniform,
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
            ParamCategory::BufferWrite => Some(Self::Write),
            ParamCategory::BufferReadWrite => Some(Self::ReadWrite),
            ParamCategory::Scalar => None,
        }
    }
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
