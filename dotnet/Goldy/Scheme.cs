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
    /// Record a read easement over a buffer parcel (once per scheme).
    /// </summary>
    public ReadGrant GrantRead(Parcel parcel)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(parcel);

        var grant = NativeMethods.SchemeGrantRead(Handle, parcel.Handle);
        if (grant == nint.Zero)
            throw GoldyException.FromLastError("Scheme grant_read");
        return new ReadGrant(grant);
    }

    /// <summary>
    /// Record a read easement over a texture parcel (once per scheme).
    /// </summary>
    public ReadGrant GrantReadTexture(Parcel parcel)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(parcel);

        var grant = NativeMethods.SchemeGrantRead(Handle, parcel.Handle);
        if (grant == nint.Zero)
            throw GoldyException.FromLastError("Scheme grant_read_texture");
        return new ReadGrant(grant);
    }

    /// <summary>
    /// Submit the scheme and return a per-submission frame token.
    /// </summary>
    public SchemeFrame Submit()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.SchemeSubmit(Handle, out var frame);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme submit");
        return new SchemeFrame(frame);
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
