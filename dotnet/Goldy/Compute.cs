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
    public ComputePipeline(Device device, ShaderModule computeShader, params BindGroupLayout[] bindGroupLayouts)
    {
        device.ThrowIfDisposed();
        
        unsafe
        {
            var layoutHandles = new nint[bindGroupLayouts.Length];
            for (int i = 0; i < bindGroupLayouts.Length; i++)
            {
                layoutHandles[i] = bindGroupLayouts[i].Handle;
            }
            
            fixed (nint* ptr = layoutHandles)
            {
                Handle = NativeMethods.ComputePipelineCreate(
                    device.Handle, 
                    computeShader.Handle, 
                    (nint)ptr, 
                    (uint)bindGroupLayouts.Length);
            }
        }
        
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

/// <summary>
/// Command encoder for compute operations.
/// </summary>
public sealed class ComputeEncoder
{
    private readonly nint _handle;
    private bool _executed;

    /// <summary>
    /// Create a new compute encoder.
    /// </summary>
    public ComputeEncoder()
    {
        _handle = NativeMethods.ComputeEncoderCreate();
        if (_handle == nint.Zero)
            throw new GoldyException("Failed to create compute encoder");
    }

    /// <summary>
    /// Set the active compute pipeline.
    /// </summary>
    public void SetPipeline(ComputePipeline pipeline)
    {
        EnsureNotExecuted();
        NativeMethods.ComputeEncoderSetPipeline(_handle, pipeline.Handle);
    }

    /// <summary>
    /// Set a bind group for shader resources.
    /// </summary>
    public void SetBindGroup(uint index, BindGroup bindGroup)
    {
        EnsureNotExecuted();
        NativeMethods.ComputeEncoderSetBindGroup(_handle, index, bindGroup.Handle);
    }

    /// <summary>
    /// Dispatch compute workgroups.
    /// </summary>
    public void Dispatch(uint workgroupsX, uint workgroupsY = 1, uint workgroupsZ = 1)
    {
        EnsureNotExecuted();
        NativeMethods.ComputeEncoderDispatch(_handle, workgroupsX, workgroupsY, workgroupsZ);
    }

    /// <summary>
    /// Execute the recorded compute commands on the device.
    /// </summary>
    public void Execute(Device device)
    {
        EnsureNotExecuted();
        _executed = true;
        
        var result = NativeMethods.ComputeEncoderExecute(_handle, device.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Compute execution");
    }

    private void EnsureNotExecuted()
    {
        if (_executed)
            throw new InvalidOperationException("ComputeEncoder has already been executed");
    }
}

