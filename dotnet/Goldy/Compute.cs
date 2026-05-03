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
    /// Bind resource slots for compute.
    /// The indices are bound in order, so buffers[0] becomes slot 0,
    /// buffers[1] becomes slot 1, etc.
    /// </summary>
    /// <param name="buffers">Buffers to bind to shader resource slots.</param>
    public void BindResources(params Buffer[] buffers)
    {
        EnsureNotExecuted();
        if (buffers.Length == 0)
            return;

        // Collect buffer handles into an array
        Span<nint> handles = stackalloc nint[buffers.Length];
        for (int i = 0; i < buffers.Length; i++)
            handles[i] = buffers[i].Handle;

        unsafe
        {
            fixed (nint* ptr = handles)
            {
                NativeMethods.ComputeEncoderBindResources(_handle, (nint)ptr, (uint)buffers.Length);
            }
        }
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

