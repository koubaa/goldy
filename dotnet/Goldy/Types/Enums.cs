namespace Goldy;

/// <summary>
/// GPU device type.
/// </summary>
public enum DeviceType
{
    /// <summary>Discrete GPU (dedicated graphics card).</summary>
    DiscreteGpu = 0,
    /// <summary>Integrated GPU (part of CPU).</summary>
    IntegratedGpu = 1,
    /// <summary>Software renderer (CPU).</summary>
    Cpu = 2,
    /// <summary>Other/unknown.</summary>
    Other = 3,
}

/// <summary>
/// Graphics backend type.
/// </summary>
public enum BackendType
{
    Vulkan = 0,
    Metal = 1,
    Dx12 = 2,
    WebGpu = 3,
}

/// <summary>
/// Texture format for render targets.
/// </summary>
public enum TextureFormat
{
    /// <summary>8-bit RGBA, sRGB color space.</summary>
    Rgba8UnormSrgb = 0,
    /// <summary>8-bit RGBA, linear color space.</summary>
    Rgba8Unorm = 1,
    /// <summary>8-bit BGRA, sRGB color space.</summary>
    Bgra8UnormSrgb = 2,
    /// <summary>8-bit BGRA, linear color space.</summary>
    Bgra8Unorm = 3,
    /// <summary>16-bit RGBA float.</summary>
    Rgba16Float = 4,
    /// <summary>32-bit RGBA float.</summary>
    Rgba32Float = 5,
    /// <summary>Single-channel 8-bit unsigned normalized.</summary>
    R8Unorm = 6,
    /// <summary>Two-channel 8-bit unsigned normalized.</summary>
    Rg8Unorm = 7,
}

/// <summary>
/// Data access pattern for buffers.
/// Describes how threads will access the buffer, which determines hardware optimization strategies.
/// </summary>
public enum BufferKind
{
    /// <summary>
    /// Any thread, any address, read/write. No coherence assumptions.
    /// Maps to storage buffers (StructuredBuffer, RWStructuredBuffer in shaders).
    /// </summary>
    Scattered = 0,
    /// <summary>
    /// All threads read same address. Hardware broadcast optimization.
    /// Maps to uniform/constant buffers (ConstantBuffer in shaders).
    /// </summary>
    Broadcast = 1,
}

/// <summary>
/// Spatial access pattern for textures.
/// Describes how the texture will be accessed, which determines hardware optimization strategies.
/// </summary>
public enum TextureKind
{
    /// <summary>
    /// Hardware filtering between neighbors (texture units).
    /// Maps to sampled images (Texture2D with sampler in shaders).
    /// </summary>
    Interpolated = 0,
    /// <summary>
    /// Direct 2D/3D indexing, no filtering, read/write.
    /// Maps to storage images (RWTexture2D in shaders).
    /// </summary>
    Direct = 1,
}

/// <summary>
/// Vertex attribute format.
/// </summary>
public enum VertexFormat
{
    Float32 = 0,
    Float32x2 = 1,
    Float32x3 = 2,
    Float32x4 = 3,
    Uint32 = 4,
    Sint32 = 5,
    Uint8x4 = 6,
    Unorm8x4 = 7,
}

/// <summary>
/// Primitive topology for drawing.
/// </summary>
public enum PrimitiveTopology
{
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    TriangleList = 3,
    TriangleStrip = 4,
}

/// <summary>
/// Index buffer format.
/// </summary>
public enum IndexFormat
{
    /// <summary>16-bit unsigned indices (0-65535).</summary>
    Uint16 = 0,
    /// <summary>32-bit unsigned indices (0-4 billion).</summary>
    Uint32 = 1,
}

/// <summary>
/// Depth/stencil texture format.
/// </summary>
public enum DepthFormat
{
    /// <summary>16-bit depth, no stencil.</summary>
    Depth16Unorm = 0,
    /// <summary>24-bit depth, no stencil (platform may use 32-bit internally).</summary>
    Depth24Plus = 1,
    /// <summary>24-bit depth + 8-bit stencil.</summary>
    Depth24PlusStencil8 = 2,
    /// <summary>32-bit floating point depth, no stencil.</summary>
    Depth32Float = 3,
    /// <summary>32-bit floating point depth + 8-bit stencil.</summary>
    Depth32FloatStencil8 = 4,
}

/// <summary>
/// Depth comparison function for depth testing.
/// </summary>
public enum CompareFunction
{
    /// <summary>Never passes.</summary>
    Never = 0,
    /// <summary>Passes if new &lt; current.</summary>
    Less = 1,
    /// <summary>Passes if new == current.</summary>
    Equal = 2,
    /// <summary>Passes if new &lt;= current.</summary>
    LessEqual = 3,
    /// <summary>Passes if new &gt; current.</summary>
    Greater = 4,
    /// <summary>Passes if new != current.</summary>
    NotEqual = 5,
    /// <summary>Passes if new &gt;= current.</summary>
    GreaterEqual = 6,
    /// <summary>Always passes.</summary>
    Always = 7,
}

/// <summary>
/// Texture flags for copy and render operations.
/// </summary>
[Flags]
public enum TextureFlags : uint
{
    /// <summary>No flags.</summary>
    None = 0,
    /// <summary>Can be used as a copy source.</summary>
    CopySrc = 1 << 0,
    /// <summary>Can be used as a copy destination.</summary>
    CopyDst = 1 << 1,
    /// <summary>Can be used as a render attachment.</summary>
    RenderTarget = 1 << 2,
}

/// <summary>
/// Texture addressing mode for coordinates outside [0, 1].
/// </summary>
public enum AddressMode
{
    /// <summary>Clamp to edge color.</summary>
    ClampToEdge = 0,
    /// <summary>Repeat the texture.</summary>
    Repeat = 1,
    /// <summary>Mirror and repeat the texture.</summary>
    MirrorRepeat = 2,
}

/// <summary>
/// Texture filtering mode.
/// </summary>
public enum FilterMode
{
    /// <summary>Nearest-neighbor sampling (blocky).</summary>
    Nearest = 0,
    /// <summary>Linear interpolation (smooth).</summary>
    Linear = 1,
}

/// <summary>
/// Per-node resource access for task graph bindings.
/// </summary>
public enum NodeAccess
{
    Read = 0,
    Write = 1,
    ReadWrite = 2,
}

/// <summary>
/// Shader resource binding access (SRV vs UAV).
/// </summary>
public enum ResourceAccess
{
    Read = 0,
    Write = 1,
    ReadWrite = 2,
}

/// <summary>
/// Shader stages flags.
/// </summary>
[Flags]
public enum ShaderStages : uint
{
    Vertex = 1,
    Fragment = 2,
    Compute = 4,
    All = 7,
}
