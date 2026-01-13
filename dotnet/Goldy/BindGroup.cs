using Goldy.Native;

namespace Goldy;

/// <summary>
/// A bind group layout defines the structure of a bind group.
/// </summary>
public sealed class BindGroupLayout : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    /// <summary>
    /// Create a new bind group layout.
    /// </summary>
    public BindGroupLayout(Device device, params BindGroupLayoutBinding[] bindings)
    {
        device.ThrowIfDisposed();
        
        unsafe
        {
            var nativeBindings = new BindGroupLayoutBindingNative[bindings.Length];
            for (int i = 0; i < bindings.Length; i++)
            {
                nativeBindings[i] = new BindGroupLayoutBindingNative
                {
                    Binding = bindings[i].Binding,
                    Visibility = bindings[i].Visibility,
                    BindingType = bindings[i].BindingType,
                };
            }
            
            fixed (BindGroupLayoutBindingNative* ptr = nativeBindings)
            {
                Handle = NativeMethods.BindGroupLayoutCreate(device.Handle, (nint)ptr, (uint)bindings.Length);
            }
        }
        
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("BindGroupLayout creation");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.BindGroupLayoutDestroy(Handle);
            _disposed = true;
        }
    }
}

/// <summary>
/// Description of a binding in a bind group layout.
/// </summary>
public readonly record struct BindGroupLayoutBinding(uint Binding, ShaderStages Visibility, BindingType BindingType)
{
    /// <summary>Create a uniform buffer binding visible to all graphics stages.</summary>
    public static BindGroupLayoutBinding Uniform(uint binding) => 
        new(binding, ShaderStages.All, BindingType.UniformBuffer);

    /// <summary>Create a storage buffer binding.</summary>
    public static BindGroupLayoutBinding Storage(uint binding, bool readOnly = false) => 
        new(binding, ShaderStages.All, readOnly ? BindingType.StorageBufferReadOnly : BindingType.StorageBufferReadWrite);

    /// <summary>Create a sampled texture binding.</summary>
    public static BindGroupLayoutBinding Texture(uint binding) => 
        new(binding, ShaderStages.Fragment, BindingType.Texture);

    /// <summary>Create a sampler binding.</summary>
    public static BindGroupLayoutBinding Sampler(uint binding) => 
        new(binding, ShaderStages.Fragment, BindingType.Sampler);
}

/// <summary>
/// A bind group contains actual resource bindings matching a layout.
/// </summary>
public sealed class BindGroup : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    /// <summary>
    /// Create a new bind group from a layout and buffer bindings.
    /// </summary>
    public BindGroup(Device device, BindGroupLayout layout, params BufferBinding[] bufferBindings)
    {
        device.ThrowIfDisposed();
        
        unsafe
        {
            var nativeBindings = new BufferBindingNative[bufferBindings.Length];
            for (int i = 0; i < bufferBindings.Length; i++)
            {
                nativeBindings[i] = new BufferBindingNative
                {
                    Binding = bufferBindings[i].Binding,
                    Buffer = bufferBindings[i].Buffer.Handle,
                    Offset = bufferBindings[i].Offset,
                    Size = bufferBindings[i].Size,
                };
            }
            
            fixed (BufferBindingNative* ptr = nativeBindings)
            {
                Handle = NativeMethods.BindGroupCreate(device.Handle, layout.Handle, (nint)ptr, (uint)bufferBindings.Length);
            }
        }
        
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("BindGroup creation");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.BindGroupDestroy(Handle);
            _disposed = true;
        }
    }
}

/// <summary>
/// Description of a buffer binding in a bind group.
/// </summary>
public readonly record struct BufferBinding(uint Binding, Buffer Buffer, ulong Offset = 0, ulong Size = 0)
{
    /// <summary>Create a buffer binding for the entire buffer.</summary>
    public static BufferBinding Whole(uint binding, Buffer buffer) => new(binding, buffer, 0, 0);
}

