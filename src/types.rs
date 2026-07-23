//! Common types used throughout Goldy.

use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};

/// RGBA color with floating point components (0.0 - 1.0).
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// Color-target load declared on [`crate::Scheme::render_pass`], not in the command list.
///
/// - [`Self::Load`] — preserve prior contents (`NodeAccess::Write` on the RT).
/// - [`Self::Clear`] — clear to a color, then draw (private-inaugural / `Overwrite`).
/// - [`Self::Discard`] — prior contents irrelevant; fully overwritten by draws (`Overwrite`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetLoad {
    /// Depend on prior color contents.
    Load,
    /// Clear the color target to this value at pass begin.
    Clear(Color),
    /// Discard prior color contents (private-inaugural).
    Discard,
}

impl TargetLoad {
    /// True when the pass does not depend on prior color contents.
    pub fn overwrites(self) -> bool {
        matches!(self, Self::Clear(_) | Self::Discard)
    }
}

impl Color {
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const RED: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const GREEN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    pub const BLUE: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    pub const CORNFLOWER_BLUE: Color = Color {
        r: 0.392,
        g: 0.584,
        b: 0.929,
        a: 1.0,
    };

    /// Create a color from RGB values (0-255).
    pub const fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: 1.0,
        }
    }

    /// Create a color from RGBA values (0-255).
    pub const fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    /// Convert to RGBA u8 array.
    pub fn to_rgba8(&self) -> [u8; 4] {
        [
            (self.r * 255.0) as u8,
            (self.g * 255.0) as u8,
            (self.b * 255.0) as u8,
            (self.a * 255.0) as u8,
        ]
    }
}

/// Texture format for render targets and textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureFormat {
    /// Single-channel 8-bit unsigned normalized (red only)
    R8Unorm,
    /// Two-channel 8-bit unsigned normalized (red + green)
    Rg8Unorm,
    /// 8-bit RGBA, sRGB color space
    Rgba8UnormSrgb,
    /// 8-bit RGBA, linear color space
    #[default]
    Rgba8Unorm,
    /// 8-bit BGRA, sRGB color space
    Bgra8UnormSrgb,
    /// 8-bit BGRA, linear color space
    Bgra8Unorm,
    /// 16-bit RGBA float
    Rgba16Float,
    /// 32-bit RGBA float
    Rgba32Float,
}

impl TextureFormat {
    /// Get the number of bytes per pixel.
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            TextureFormat::R8Unorm => 1,
            TextureFormat::Rg8Unorm => 2,
            TextureFormat::Rgba8UnormSrgb => 4,
            TextureFormat::Rgba8Unorm => 4,
            TextureFormat::Bgra8UnormSrgb => 4,
            TextureFormat::Bgra8Unorm => 4,
            TextureFormat::Rgba16Float => 8,
            TextureFormat::Rgba32Float => 16,
        }
    }
}

// ============================================================================
// Dispatch Types
// ============================================================================

/// Workgroup-count triple for compute dispatch (direct or indirect).
///
/// Matches `goldy_exp.types.DispatchShape` in shaders. When held in a device
/// parcel, the first 12 bytes at the parcel's byte offset are consumed by
/// indirect dispatch commands (`vkCmdDispatchIndirect` / DX12 `ExecuteIndirect`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Pod, Zeroable)]
pub struct DispatchShape {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl DispatchShape {
    /// Create a workgroup-count triple.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

impl From<(u32, u32, u32)> for DispatchShape {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self::new(x, y, z)
    }
}

// ============================================================================
// Access Pattern Types
// ============================================================================

/// Use-time access direction for a resource descriptor slot.
///
/// Passed to [`crate::Parcel::handle`], [`crate::Texture::handle`], and related
/// accessors to select the correct descriptor pool entry for how the resource
/// will be used in the current dispatch — read-only, write-only, or read-write.
///
/// This is distinct from [`crate::task_graph::NodeAccess`], which describes
/// scheduling / SWMR hazards between graph nodes rather than shader binding slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ResourceAccess {
    /// Read-only access (SRV, CBV, sampled texture, etc.).
    #[default]
    Read,
    /// Write-only access (UAV, storage image, etc.).
    Write,
    /// Read-write access (UAV / storage where both directions are valid).
    ReadWrite,
}

/// DX12 bindless descriptor view kind for a push-constant slot.
///
/// Scattered buffers get separate SRV and UAV heap indices; binding the wrong
/// one compiles but reads zeros on WARP (and may misbehave on hardware).
#[cfg(all(feature = "dx12", target_os = "windows"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BindlessSlotKind {
    /// `StorageBuffer<T>` / `RWStructuredBuffer` (UAV).
    StorageUav,
    /// `BufRO<T>` / read-only `StructuredBuffer` (SRV).
    ReadOnlySrv,
    /// Broadcast uniform / CBV.
    UniformCbv,
}

#[cfg(all(feature = "dx12", target_os = "windows"))]
impl BindlessSlotKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::StorageUav => "storage UAV",
            Self::ReadOnlySrv => "read-only SRV",
            Self::UniformCbv => "uniform CBV",
        }
    }
}

/// Buffer kind / data access pattern for buffers.
///
/// This describes how threads will access the buffer, which determines
/// hardware optimization strategies:
///
/// - `Scattered`: Any thread can access any address. No coherence assumptions.
/// - `Broadcast`: All threads read the same address. Hardware can broadcast
///   a single fetch to the entire wave (32-64 threads).
///
/// When creating buffers with [`crate::RetainedPool::acquire_buffer_with_data`], the inferred
/// element stride must match what the shader expects. Passing `&[u8]` (e.g. from
/// `bytemuck::bytes_of`) sets stride to 1 byte; for structured data use a typed slice or
/// [`crate::RetainedPool::acquire_buffer`] with an explicit element stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BufferKind {
    /// Any thread, any address, read/write. No coherence assumptions.
    ///
    /// Structured-buffer views use the buffer's recorded **element stride** (from
    /// [`crate::RetainedPool::acquire_buffer_with_data`], [`crate::RetainedPool::acquire_buffer`], etc.).
    ///
    /// Maps to storage buffers (StructuredBuffer, RWStructuredBuffer in shaders).
    #[default]
    Scattered,
    /// All threads read the same address. Hardware can broadcast a single
    /// fetch to the entire wave. Maps to uniform buffers (Vulkan/Metal) or
    /// constant buffers (DX12/HLSL) depending on the backend.
    Broadcast,
}

/// How costly in-place buffer resize (`resize_to`) is on this device.
///
/// Phase 1 uses [`Self::Copy`] on all backends. Later phases may report
/// [`Self::Constant`] (oversized + demand paging) or [`Self::PageBind`] (sparse).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BufferResizeCost {
    /// Handle is stable; growth within reserved capacity is metadata-only.
    Constant,
    /// Physical pages bound/unbound (Vulkan sparse, DX12 tiled).
    PageBind,
    /// Reallocation plus GPU copy of existing contents.
    #[default]
    Copy,
}

/// Category of a resource descriptor slot.
///
/// Goldy's bindless argument buffers / descriptor heaps are organized into
/// separate pools per access pattern. A resource index is only meaningful
/// relative to its category — e.g. a `Scattered` slot #3 and a `Broadcast`
/// slot #3 refer to different physical entries on Metal (`storageBuffers[3]`
/// vs `uniformBuffers[3]`) even though the `u32` indices are identical.
///
/// Capturing the category alongside the index lets the CPU API and the
/// shader-side `goldy_exp` access functions be type-checked against each
/// other — see [`ResourceHandle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ResourceCategory {
    /// Storage-buffer slot. Used with `goldy_scattered<T>` / `goldy_buf_ro<T>`
    /// (both shader functions index the same pool).
    Scattered,
    /// Uniform / constant-buffer slot. Used with `goldy_broadcast<T>`.
    Broadcast,
    /// Storage-image (writable texture) slot. Used with `goldy_direct_spatial<T>`.
    StorageImage,
    /// Sampled-texture slot. Used with `goldy_interpolated<T>`.
    Texture,
    /// Sampler slot. Used with `goldy_filter`.
    Sampler,
}

impl ResourceCategory {
    /// Short human-readable name for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            ResourceCategory::Scattered => "scattered",
            ResourceCategory::Broadcast => "broadcast",
            ResourceCategory::StorageImage => "storage_image",
            ResourceCategory::Texture => "texture",
            ResourceCategory::Sampler => "sampler",
        }
    }

    /// True if this handle can satisfy a shader slot declared as `expected`.
    ///
    /// `Scattered` and `Broadcast` are strictly distinct (different pools on Metal,
    /// different descriptor types on Vulkan/DX12). `StorageImage`, `Texture`, and
    /// `Sampler` are likewise non-interchangeable.
    pub fn is_compatible_with(self, expected: ResourceCategory) -> bool {
        self == expected
    }
}

impl From<BufferKind> for ResourceCategory {
    fn from(access: BufferKind) -> Self {
        match access {
            BufferKind::Scattered => ResourceCategory::Scattered,
            BufferKind::Broadcast => ResourceCategory::Broadcast,
        }
    }
}

impl From<TextureKind> for ResourceCategory {
    fn from(access: TextureKind) -> Self {
        match access {
            TextureKind::Interpolated => ResourceCategory::Texture,
            TextureKind::Direct => ResourceCategory::StorageImage,
            // Primary slot for DirectInterpolated is the storage (UAV) handle.
            // The secondary sampled (SRV) handle is obtained via
            // `Texture::handle(ResourceAccess::Read)`.
            TextureKind::DirectInterpolated => ResourceCategory::StorageImage,
        }
    }
}

/// Opaque typed resource descriptor identity: `(category, index)`.
///
/// Goldy's resources (`Buffer`, `Texture`, `Sampler`) expose `handle(access)`
/// which returns one of these. Equality and hashing are stable for a live
/// descriptor identity, so callers may compare handles (for example retained-
/// scheme staleness checks) without observing the underlying heap slot.
///
/// The raw descriptor index is crate-private (`index`) so backends may
/// reinterpret slots without a public heap-index contract.
/// Push-constant / bind paths that accept `ResourceHandle` validate the
/// handle's [`ResourceCategory`] against shader reflection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle {
    category: ResourceCategory,
    index: u32,
}

impl ResourceHandle {
    /// Build a handle from a category and raw index.
    pub(crate) const fn new(category: ResourceCategory, index: u32) -> Self {
        Self { category, index }
    }

    /// Raw descriptor index for this handle (crate-internal bindless plumbing).
    pub(crate) const fn index(self) -> u32 {
        self.index
    }

    /// Category this handle was tagged with at creation.
    pub const fn category(self) -> ResourceCategory {
        self.category
    }
}

/// Presentation mode controlling how frames are displayed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PresentMode {
    /// Vsync: wait for display refresh. No tearing, capped at display Hz.
    /// Maps to Metal `displaySyncEnabled=YES`, Vulkan `FIFO`, DX12 `Present(1)`.
    Fifo,
    /// Triple-buffered: latest frame queued, older frames dropped. Low latency + no tearing.
    /// Maps to Vulkan `MAILBOX`. Falls back to Fifo on Metal and some DX12 configurations.
    Mailbox,
    /// No sync: present immediately. May tear. Maximum throughput for benchmarks.
    /// Maps to Metal `displaySyncEnabled=NO`, Vulkan `IMMEDIATE`, DX12 `Present(0)`.
    Immediate,
    /// Let Goldy choose (Mailbox if available, then Fifo).
    #[default]
    Auto,
}

/// Configuration for surface creation.
#[derive(Debug, Clone, Default)]
pub struct SurfaceConfig {
    /// Presentation mode (vsync strategy).
    pub present_mode: PresentMode,
    /// Optional depth buffer for 3D rendering.
    pub depth_format: Option<DepthFormat>,
}

/// Texture kind / spatial access pattern for textures and images.
///
/// This describes how the texture will be accessed:
///
/// - `Interpolated`: Hardware filtering between neighbors (texture units).
/// - `Direct`: Direct 2D/3D indexing without filtering, read/write.
/// - `DirectInterpolated`: Both storage (UAV) and sampled (SRV) access on the same texture.
///   Suitable for filter layers that are written by one pass and read with hardware bilinear
///   by the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureKind {
    /// Hardware filtering between neighbors (texture units).
    /// Maps to sampled images (Texture2D with sampler in shaders).
    #[default]
    Interpolated,
    /// Direct 2D/3D indexing, no filtering, read/write.
    /// Maps to storage images (RWTexture2D in shaders).
    Direct,
    /// Both UAV (storage/write via `DirectSpatial`) and SRV (sampled/read via `Interpolated`)
    /// access on the same underlying texture resource.
    ///
    /// The primary resource handle (returned by
    /// [`crate::Texture::handle`](ResourceAccess::Write)) is the storage slot;
    /// the secondary sampled handle is returned by
    /// [`crate::Texture::handle`](ResourceAccess::Read).
    DirectInterpolated,
}

bitflags! {
    /// Additional buffer flags for copy operations and CPU readback.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferFlags: u32 {
        /// Can be used as a copy source.
        const COPY_SRC = 1 << 0;
        /// Can be used as a copy destination.
        const COPY_DST = 1 << 1;
        /// Medium hint: prefer host-visible storage when the runtime chooses to map
        /// withdraw staging onto the same allocation (UMA backends). Discrete backends may
        /// still keep GPU-local storage and blit into withdraw staging at claim time.
        ///
        /// Prefer [`crate::MemoryExchange::bind_withdraw`] for CPU observation — this flag
        /// is not a public readback API.
        ///
        /// Query [`crate::device::DeviceCapabilities::has_zero_copy_storage_readback`] to
        /// distinguish zero-copy vs staged withdraw behavior.
        const CPU_READABLE = 1 << 2;
        /// GPU-local storage (Metal: [`MTLStorageMode::Private`]), no CPU mapping.
        ///
        /// Intended for purely device-side buffers (e.g. frame scratch pools filled via blits /
        /// compute only). Cannot be combined with [`Self::CPU_READABLE`].
        ///
        /// **Backends**: implemented on Metal. Other backends treat this flag as unused for now.
        const GPU_ONLY = 1 << 3;
        /// Host-visible storage buffer for per-frame CPU writes (deposit / staging uploads).
        ///
        /// On Vulkan/Metal the buffer is persistently mapped for [`crate::Buffer::write`].
        /// On Direct3D 12, storage buffers use a paired UPLOAD heap for CPU writes and
        /// a DEFAULT heap UAV for GPU access; scheme [`CopyBuffer`](crate::task_graph::NodeKind::CopyBuffer)
        /// nodes (including deposits) copy staging → device each submission.
        ///
        /// **Write contract:** [`crate::Buffer::write`] on a `CPU_WRITABLE` buffer does
        /// **not** queue-order the host write behind in-flight GPU readers. Callers must
        /// only write when the buffer is **settled** (host-observed GPU progress past its
        /// last use) or **fresh** (never GPU-referenced). [`crate::MemoryExchange`] deposits
        /// and [`crate::Parcel::is_settled`] enforce this; arbitrary reuse with live GPU
        /// readers is a race (Vulkan/Metal host-visible semantics; DX12 applies at copy time).
        ///
        /// Prefer deposits for application uploads. Only valid for [`BufferKind::Scattered`].
        /// Cannot be combined with [`Self::CPU_READABLE`] or [`Self::GPU_ONLY`].
        const CPU_WRITABLE = 1 << 4;
    }
}

/// Vertex attribute format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Sint32,
    Uint8x4,
    Unorm8x4,
}

impl VertexFormat {
    /// Get the size in bytes of this format.
    pub fn size(&self) -> u32 {
        match self {
            VertexFormat::Float32 => 4,
            VertexFormat::Float32x2 => 8,
            VertexFormat::Float32x3 => 12,
            VertexFormat::Float32x4 => 16,
            VertexFormat::Uint32 => 4,
            VertexFormat::Sint32 => 4,
            VertexFormat::Uint8x4 => 4,
            VertexFormat::Unorm8x4 => 4,
        }
    }
}

/// Vertex attribute description.
#[derive(Debug, Clone)]
pub struct VertexAttribute {
    /// Shader location (e.g., `layout(location = 0)`).
    pub location: u32,
    /// Data format.
    pub format: VertexFormat,
    /// Byte offset within the vertex.
    pub offset: u32,
}

/// Vertex buffer layout.
#[derive(Debug, Clone)]
pub struct VertexBufferLayout {
    /// Stride in bytes between vertices.
    pub stride: u32,
    /// Attribute descriptions.
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    /// Create an empty vertex layout.
    ///
    /// Use this when the shader doesn't read from vertex buffers
    /// (e.g., when using only `SV_VertexID` and storage buffers).
    pub fn empty() -> Self {
        Self {
            stride: 0,
            attributes: Vec::new(),
        }
    }

    /// Build a layout from a list of formats, inferring offsets and locations.
    ///
    /// Locations are assigned sequentially (0, 1, 2, ...). Offsets are computed
    /// by accumulating each format's byte size. The stride is taken from
    /// `size_of::<T>()`, and the method panics if the summed format sizes don't
    /// match — catching field-list mismatches immediately rather than producing
    /// silent GPU corruption.
    ///
    /// # Example
    ///
    /// ```rust
    /// use goldy::types::{VertexBufferLayout, VertexFormat};
    ///
    /// #[repr(C)]
    /// struct MyVertex {
    ///     pos: [f32; 3],
    ///     uv: [f32; 2],
    ///     color: u32,
    /// }
    ///
    /// let layout = VertexBufferLayout::from_formats::<MyVertex>(&[
    ///     VertexFormat::Float32x3, // pos
    ///     VertexFormat::Float32x2, // uv
    ///     VertexFormat::Uint32,    // color
    /// ]);
    /// assert_eq!(layout.stride, 24);
    /// assert_eq!(layout.attributes.len(), 3);
    /// ```
    pub fn from_formats<T>(formats: &[VertexFormat]) -> Self {
        let mut offset = 0u32;
        let attributes: Vec<VertexAttribute> = formats
            .iter()
            .enumerate()
            .map(|(i, fmt)| {
                let attr = VertexAttribute {
                    location: i as u32,
                    offset,
                    format: *fmt,
                };
                offset += fmt.size();
                attr
            })
            .collect();
        let expected_stride = std::mem::size_of::<T>() as u32;
        assert_eq!(
            offset,
            expected_stride,
            "VertexBufferLayout::from_formats: sum of format sizes ({offset}) != \
             size_of::<{}>() ({expected_stride}). Check field order and padding.",
            std::any::type_name::<T>(),
        );
        Self {
            stride: expected_stride,
            attributes,
        }
    }
}

/// Primitive topology for drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    #[default]
    TriangleList,
    TriangleStrip,
}

/// Index buffer format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IndexFormat {
    /// 16-bit unsigned indices (0-65535).
    #[default]
    Uint16,
    /// 32-bit unsigned indices (0-4 billion).
    Uint32,
}

impl IndexFormat {
    /// Get the size in bytes of one index.
    pub fn size(&self) -> u32 {
        match self {
            IndexFormat::Uint16 => 2,
            IndexFormat::Uint32 => 4,
        }
    }
}

/// Type of GPU device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Discrete GPU (dedicated graphics card).
    DiscreteGpu,
    /// Integrated GPU (part of CPU).
    IntegratedGpu,
    /// Software renderer (CPU).
    Cpu,
    /// Other/unknown.
    Other,
}

/// Shader compiler optimization level.
///
/// Controls how aggressively the Slang compiler optimizes generated SPIR-V / DXIL / Metal IR.
/// Use `None` to work around driver bugs in software renderers (e.g. lavapipe SSA corruption
/// across barriers).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub enum OptimizationLevel {
    /// No optimization — preserves all loads and barriers exactly as written.
    None,
    /// Default balanced optimization.
    #[default]
    Default,
    /// Aggressive optimization.
    High,
    /// Maximum optimization (may be very slow to compile).
    Maximal,
}

/// Graphics backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    Vulkan,
    Metal,
    Dx12,
    /// Portable WebGPU backend (currently compute-only).
    WebGpu,
}

/// A simple 2D vertex with position and color.
/// Use for colored primitives (triangle, particles, etc.)
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex2D {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex2D {
    pub const fn new(x: f32, y: f32, color: Color) -> Self {
        Self {
            position: [x, y],
            color: [color.r, color.g, color.b, color.a],
        }
    }

    /// Get the vertex buffer layout for this vertex type.
    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout::from_formats::<Self>(&[
            VertexFormat::Float32x2, // position
            VertexFormat::Float32x4, // color
        ])
    }
}

/// A 2D vertex with position and UV coordinates.
/// Use for textured/shader effects (plasma, gradient, etc.)
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex2DUv {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

impl Vertex2DUv {
    pub const fn new(x: f32, y: f32, u: f32, v: f32) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
        }
    }

    /// Get the vertex buffer layout for this vertex type.
    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout::from_formats::<Self>(&[
            VertexFormat::Float32x2, // position
            VertexFormat::Float32x2, // uv
        ])
    }
}

// StructuredBufferElement impls for public vertex types
use crate::buffer::StructuredBufferElement;
impl StructuredBufferElement for Vertex2D {}
impl StructuredBufferElement for Vertex2DUv {}

// ============================================================================
// Depth Buffer Types
// ============================================================================

/// Depth/stencil texture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthFormat {
    /// 16-bit depth, no stencil
    Depth16Unorm,
    /// 24-bit depth, no stencil (platform may use 32-bit internally)
    Depth24Plus,
    /// 24-bit depth + 8-bit stencil
    Depth24PlusStencil8,
    /// 32-bit floating point depth, no stencil
    Depth32Float,
    /// 32-bit floating point depth + 8-bit stencil
    Depth32FloatStencil8,
}

impl DepthFormat {
    /// Returns true if this format includes a stencil component.
    pub fn has_stencil(&self) -> bool {
        matches!(
            self,
            DepthFormat::Depth24PlusStencil8 | DepthFormat::Depth32FloatStencil8
        )
    }
}

/// Depth comparison function for depth testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompareFunction {
    /// Never passes.
    Never,
    /// Passes if new < current.
    #[default]
    Less,
    /// Passes if new == current.
    Equal,
    /// Passes if new <= current.
    LessEqual,
    /// Passes if new > current.
    Greater,
    /// Passes if new != current.
    NotEqual,
    /// Passes if new >= current.
    GreaterEqual,
    /// Always passes.
    Always,
}

/// Depth/stencil state for render pipelines.
#[derive(Debug, Clone)]
pub struct DepthStencilState {
    /// Depth format to use. Must match the render target's depth format.
    pub format: DepthFormat,
    /// Whether to write depth values.
    pub depth_write_enabled: bool,
    /// Comparison function for depth test.
    pub depth_compare: CompareFunction,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            format: DepthFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
        }
    }
}

// ============================================================================
// Texture Types
// ============================================================================

bitflags! {
    /// Additional texture flags for copy and render operations.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TextureFlags: u32 {
        /// Can be used as a copy source.
        const COPY_SRC = 1 << 0;
        /// Can be used as a copy destination.
        const COPY_DST = 1 << 1;
        /// Can be used as a render attachment.
        const RENDER_TARGET = 1 << 2;
    }
}

/// Texture addressing mode for coordinates outside [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AddressMode {
    /// Clamp to edge color.
    #[default]
    ClampToEdge,
    /// Repeat the texture.
    Repeat,
    /// Mirror and repeat the texture.
    MirrorRepeat,
}

/// Texture filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FilterMode {
    /// Nearest-neighbor sampling (blocky).
    #[default]
    Nearest,
    /// Linear interpolation (smooth).
    Linear,
}

/// Sampler descriptor for texture sampling.
#[derive(Debug, Clone)]
pub struct SamplerDesc {
    /// Addressing mode for U (horizontal) coordinate.
    pub address_mode_u: AddressMode,
    /// Addressing mode for V (vertical) coordinate.
    pub address_mode_v: AddressMode,
    /// Addressing mode for W (depth) coordinate.
    pub address_mode_w: AddressMode,
    /// Magnification filter (when texture is enlarged).
    pub mag_filter: FilterMode,
    /// Minification filter (when texture is shrunk).
    pub min_filter: FilterMode,
    /// Mipmap filter mode.
    pub mipmap_filter: FilterMode,
    /// Maximum anisotropic filtering level (1.0 = disabled).
    pub max_anisotropy: f32,
    /// Compare function for depth textures (None = no comparison).
    pub compare: Option<CompareFunction>,
    /// Minimum LOD clamp.
    pub lod_min_clamp: f32,
    /// Maximum LOD clamp.
    pub lod_max_clamp: f32,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            max_anisotropy: 1.0,
            compare: None,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_rgb() {
        let color = Color::from_rgb(255, 128, 0);
        assert!((color.r - 1.0).abs() < 0.01);
        assert!((color.g - 0.502).abs() < 0.01);
        assert!((color.b - 0.0).abs() < 0.01);
        assert!((color.a - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_color_to_rgba8() {
        let color = Color {
            r: 1.0,
            g: 0.5,
            b: 0.0,
            a: 1.0,
        };
        let rgba = color.to_rgba8();
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[1], 127);
        assert_eq!(rgba[2], 0);
        assert_eq!(rgba[3], 255);
    }

    #[test]
    fn test_color_constants() {
        assert_eq!(Color::BLACK.r, 0.0);
        assert_eq!(Color::WHITE.r, 1.0);
        assert_eq!(Color::RED.r, 1.0);
        assert_eq!(Color::RED.g, 0.0);
    }

    #[test]
    fn test_texture_format_bytes_per_pixel() {
        assert_eq!(TextureFormat::R8Unorm.bytes_per_pixel(), 1);
        assert_eq!(TextureFormat::Rg8Unorm.bytes_per_pixel(), 2);
        assert_eq!(TextureFormat::Rgba8Unorm.bytes_per_pixel(), 4);
        assert_eq!(TextureFormat::Rgba16Float.bytes_per_pixel(), 8);
        assert_eq!(TextureFormat::Rgba32Float.bytes_per_pixel(), 16);
    }

    #[test]
    fn test_vertex_format_size() {
        assert_eq!(VertexFormat::Float32.size(), 4);
        assert_eq!(VertexFormat::Float32x2.size(), 8);
        assert_eq!(VertexFormat::Float32x3.size(), 12);
        assert_eq!(VertexFormat::Float32x4.size(), 16);
    }

    #[test]
    fn test_vertex2d_layout() {
        let layout = Vertex2D::layout();
        assert_eq!(layout.stride, 24); // 2 floats + 4 floats = 6 * 4 = 24
        assert_eq!(layout.attributes.len(), 2);
        assert_eq!(layout.attributes[0].location, 0);
        assert_eq!(layout.attributes[1].location, 1);
    }

    #[test]
    fn test_vertex2duv_layout() {
        let layout = Vertex2DUv::layout();
        assert_eq!(layout.stride, 16); // 2 floats + 2 floats = 4 * 4 = 16
        assert_eq!(layout.attributes.len(), 2);
    }

    #[test]
    fn test_buffer_flags() {
        let flags = BufferFlags::COPY_SRC | BufferFlags::COPY_DST;
        assert!(flags.contains(BufferFlags::COPY_SRC));
        assert!(flags.contains(BufferFlags::COPY_DST));
    }

    #[test]
    fn test_buffer_kind_default() {
        assert_eq!(BufferKind::default(), BufferKind::Scattered);
    }

    #[test]
    fn test_texture_kind_default() {
        assert_eq!(TextureKind::default(), TextureKind::Interpolated);
    }

    #[test]
    fn test_index_format_size() {
        assert_eq!(IndexFormat::Uint16.size(), 2);
        assert_eq!(IndexFormat::Uint32.size(), 4);
    }

    #[test]
    fn test_index_format_default() {
        assert_eq!(IndexFormat::default(), IndexFormat::Uint16);
    }

    // Depth buffer tests
    #[test]
    fn test_depth_format_has_stencil() {
        assert!(!DepthFormat::Depth16Unorm.has_stencil());
        assert!(!DepthFormat::Depth24Plus.has_stencil());
        assert!(DepthFormat::Depth24PlusStencil8.has_stencil());
        assert!(!DepthFormat::Depth32Float.has_stencil());
        assert!(DepthFormat::Depth32FloatStencil8.has_stencil());
    }

    #[test]
    fn test_compare_function_default() {
        assert_eq!(CompareFunction::default(), CompareFunction::Less);
    }

    #[test]
    fn test_depth_stencil_state_default() {
        let state = DepthStencilState::default();
        assert_eq!(state.format, DepthFormat::Depth24Plus);
        assert!(state.depth_write_enabled);
        assert_eq!(state.depth_compare, CompareFunction::Less);
    }

    // Texture types tests
    #[test]
    fn test_dispatch_shape_size() {
        assert_eq!(std::mem::size_of::<DispatchShape>(), 12);
    }

    #[test]
    fn test_texture_flags() {
        let flags = TextureFlags::COPY_SRC | TextureFlags::COPY_DST;
        assert!(flags.contains(TextureFlags::COPY_SRC));
        assert!(flags.contains(TextureFlags::COPY_DST));
        assert!(!flags.contains(TextureFlags::RENDER_TARGET));
    }

    #[test]
    fn test_address_mode_default() {
        assert_eq!(AddressMode::default(), AddressMode::ClampToEdge);
    }

    #[test]
    fn test_filter_mode_default() {
        assert_eq!(FilterMode::default(), FilterMode::Nearest);
    }

    #[test]
    fn test_sampler_desc_default() {
        let desc = SamplerDesc::default();
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
        assert_eq!(desc.max_anisotropy, 1.0);
        assert!(desc.compare.is_none());
        assert_eq!(desc.lod_min_clamp, 0.0);
        assert_eq!(desc.lod_max_clamp, 32.0);
    }
}
