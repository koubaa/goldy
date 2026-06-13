using System.Runtime.InteropServices;

namespace Goldy.Native;

/// <summary>
/// Native adapter info structure (fixed-size for FFI).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct AdapterInfoNative
{
    public uint Id;
    public DeviceType DeviceType;
    public fixed byte Name[256];
    public fixed byte Vendor[64];

    public string GetName()
    {
        fixed (byte* ptr = Name)
        {
            return Marshal.PtrToStringUTF8((nint)ptr) ?? "";
        }
    }

    public string GetVendor()
    {
        fixed (byte* ptr = Vendor)
        {
            return Marshal.PtrToStringUTF8((nint)ptr) ?? "";
        }
    }
}

/// <summary>
/// Native render pipeline descriptor.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct RenderPipelineDescNative
{
    public nint VertexAttributes;
    public uint VertexAttributeCount;
    public uint VertexStride;
    public PrimitiveTopology Topology;
    public TextureFormat TargetFormat;
    [MarshalAs(UnmanagedType.U1)]
    public bool DepthEnabled;
    public DepthFormat DepthFormat;
    [MarshalAs(UnmanagedType.U1)]
    public bool DepthWriteEnabled;
    public CompareFunction DepthCompare;
}

/// <summary>
/// Native vertex attribute descriptor.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct VertexAttributeNative
{
    public uint Location;
    public VertexFormat Format;
    public uint Offset;
}

/// <summary>
/// Native sampler descriptor.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct SamplerDescNative
{
    public AddressMode AddressModeU;
    public AddressMode AddressModeV;
    public AddressMode AddressModeW;
    public FilterMode MagFilter;
    public FilterMode MinFilter;
    public FilterMode MipmapFilter;
    public float MaxAnisotropy;
    public float LodMinClamp;
    public float LodMaxClamp;

    public static SamplerDescNative Default => new()
    {
        AddressModeU = AddressMode.ClampToEdge,
        AddressModeV = AddressMode.ClampToEdge,
        AddressModeW = AddressMode.ClampToEdge,
        MagFilter = FilterMode.Nearest,
        MinFilter = FilterMode.Nearest,
        MipmapFilter = FilterMode.Nearest,
        MaxAnisotropy = 1.0f,
        LodMinClamp = 0.0f,
        LodMaxClamp = 32.0f,
    };
}

/// <summary>
/// Per-submission identity returned by scheme submit (mirrors C ABI).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct GoldySchemeFrameNative
{
    public ulong TimelineValue;
}
