using Goldy.Native;

namespace Goldy;

/// <summary>
/// Retained compute scheme bound to one <see cref="Context"/>.
/// </summary>
public sealed class Scheme : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    public Scheme(Context ctx)
    {
        ctx.ThrowIfDisposed();
        Handle = NativeMethods.SchemeCreate(ctx.Handle);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("Scheme creation");
    }

    /// <summary>
    /// Begin recording a compute dispatch node. Finish with <see cref="SchemeComputeNodeScope.Dispose"/>.
    /// </summary>
    public SchemeComputeNodeScope ComputeNode(string label, ComputePipeline pipeline)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.SchemeComputeNodeBegin(Handle, label, pipeline.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme compute_node_begin");
        return new SchemeComputeNodeScope(this);
    }

    /// <summary>
    /// Submit the scheme and block until GPU completion.
    /// </summary>
    public void Submit()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.SchemeSubmit(Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme submit");
    }

    internal void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.SchemeDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}

/// <summary>
/// Active scheme compute-node recording scope. Call <see cref="Dispose"/> to dispatch and finish the node.
/// </summary>
public sealed class SchemeComputeNodeScope : IDisposable
{
    private readonly Scheme _scheme;
    private bool _finished;

    internal SchemeComputeNodeScope(Scheme scheme) => _scheme = scheme;

    public SchemeComputeNodeScope DeclareParcel(Parcel parcel, NodeAccess nodeAccess, ResourceAccess resourceAccess)
    {
        EnsureOpen();
        var result = NativeMethods.SchemeComputeNodeDeclareParcel(
            _scheme.Handle, parcel.Handle, nodeAccess, resourceAccess);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme compute_node_declare_parcel");
        return this;
    }

    public void Dispatch(uint workgroupsX = 1, uint workgroupsY = 1, uint workgroupsZ = 1)
    {
        EnsureOpen();
        _finished = true;
        var result = NativeMethods.SchemeComputeNodeDispatch(
            _scheme.Handle, workgroupsX, workgroupsY, workgroupsZ);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme compute_node_dispatch");
    }

    public void Dispose()
    {
        if (!_finished)
            Dispatch();
    }

    private void EnsureOpen()
    {
        if (_finished)
            throw new InvalidOperationException("Scheme compute node already finished");
    }
}
