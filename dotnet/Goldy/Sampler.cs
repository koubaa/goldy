using Goldy.Native;

namespace Goldy;

/// <summary>
/// A texture sampler.
/// </summary>
public sealed class Sampler : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    /// <summary>
    /// Create a new sampler with the given descriptor.
    /// </summary>
    public Sampler(Device device, SamplerDesc desc)
    {
        device.ThrowIfDisposed();
        
        var nativeDesc = new SamplerDescNative
        {
            AddressModeU = desc.AddressModeU,
            AddressModeV = desc.AddressModeV,
            AddressModeW = desc.AddressModeW,
            MagFilter = desc.MagFilter,
            MinFilter = desc.MinFilter,
            MipmapFilter = desc.MipmapFilter,
            MaxAnisotropy = desc.MaxAnisotropy,
            LodMinClamp = desc.LodMinClamp,
            LodMaxClamp = desc.LodMaxClamp,
        };
        
        Handle = NativeMethods.SamplerCreate(device.Handle, in nativeDesc);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("Sampler creation");
    }

    /// <summary>
    /// Create a sampler with default settings.
    /// </summary>
    public Sampler(Device device)
    {
        device.ThrowIfDisposed();
        
        Handle = NativeMethods.SamplerCreateDefault(device.Handle);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("Sampler creation");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.SamplerDestroy(Handle);
            _disposed = true;
        }
    }
}

/// <summary>
/// Sampler descriptor.
/// </summary>
public record struct SamplerDesc
{
    public AddressMode AddressModeU { get; set; }
    public AddressMode AddressModeV { get; set; }
    public AddressMode AddressModeW { get; set; }
    public FilterMode MagFilter { get; set; }
    public FilterMode MinFilter { get; set; }
    public FilterMode MipmapFilter { get; set; }
    public float MaxAnisotropy { get; set; }
    public float LodMinClamp { get; set; }
    public float LodMaxClamp { get; set; }

    public static SamplerDesc Default => new()
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

    public static SamplerDesc Linear => new()
    {
        AddressModeU = AddressMode.ClampToEdge,
        AddressModeV = AddressMode.ClampToEdge,
        AddressModeW = AddressMode.ClampToEdge,
        MagFilter = FilterMode.Linear,
        MinFilter = FilterMode.Linear,
        MipmapFilter = FilterMode.Linear,
        MaxAnisotropy = 1.0f,
        LodMinClamp = 0.0f,
        LodMaxClamp = 32.0f,
    };
}

