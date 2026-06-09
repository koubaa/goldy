using Goldy.Native;

namespace Goldy;

/// <summary>
/// A compute pipeline.
/// </summary>
public sealed class ComputePipeline : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    /// <summary>
    /// Create a new compute pipeline.
    /// </summary>
    public ComputePipeline(Device device, ShaderModule computeShader)
    {
        device.ThrowIfDisposed();
        
        Handle = NativeMethods.ComputePipelineCreate(device.Handle, computeShader.Handle);
        
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("ComputePipeline creation");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.ComputePipelineDestroy(Handle);
            _disposed = true;
        }
    }
}
