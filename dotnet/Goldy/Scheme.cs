using Goldy.Native;

namespace Goldy;

/// <summary>
/// Retained scheme bound to one <see cref="Context"/>.
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
    /// Declare an offscreen render-target lease on this scheme.
    /// </summary>
    public SchemeRenderTargetLease LeaseRenderTarget(
        uint width,
        uint height,
        TextureFormat format,
        DepthFormat? depthFormat = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var hasDepth = depthFormat.HasValue;
        var depth = depthFormat ?? default;
        var lease = NativeMethods.SchemeLeaseRenderTarget(Handle, width, height, format, hasDepth, depth);
        if (lease == nint.Zero)
            throw GoldyException.FromLastError("Scheme lease_render_target");
        return new SchemeRenderTargetLease(lease);
    }

    /// <summary>
    /// Begin recording an offscreen render pass. Finish with <see cref="SchemeRenderPassScope.Dispose"/>.
    /// </summary>
    public SchemeRenderPassScope RenderPass(string label, SchemeRenderTargetLease lease, TargetLoadKind load, Color clearColor = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(lease);
        var result = NativeMethods.SchemeRenderPassBegin(Handle, label, lease.Handle, load, clearColor);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme render_pass_begin");
        return new SchemeRenderPassScope(this);
    }

    /// <summary>
    /// Begin a render pass that clears the color target.
    /// </summary>
    public SchemeRenderPassScope RenderPassClear(string label, SchemeRenderTargetLease lease, Color clearColor) =>
        RenderPass(label, lease, TargetLoadKind.Clear, clearColor);

    /// <summary>
    /// Begin a render pass that discards prior color contents.
    /// </summary>
    public SchemeRenderPassScope RenderPassDiscard(string label, SchemeRenderTargetLease lease) =>
        RenderPass(label, lease, TargetLoadKind.Discard);

    /// <summary>
    /// Copy a scheme-held render target into a texture parcel (for grant readback).
    /// </summary>
    public void CopyToTexture(SchemeRenderTargetLease src, Texture dst)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(src);
        ArgumentNullException.ThrowIfNull(dst);
        var result = NativeMethods.SchemeCopyToTexture(Handle, src.Handle, dst.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme copy_to_texture");
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
    public ReadGrant GrantReadTexture(Texture texture)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(texture);

        var grant = NativeMethods.SchemeGrantReadTexture(Handle, texture.Handle);
        if (grant == nint.Zero)
            throw GoldyException.FromLastError("Scheme grant_read_texture");
        return new ReadGrant(grant);
    }

    /// <summary>
    /// Submit the scheme and return a per-submission token.
    /// </summary>
    public SchemeSubmission Submit()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.SchemeSubmit(Handle, out var submission);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme submit");
        return new SchemeSubmission(submission);
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

    public SchemeComputeNodeScope WithParcel(Parcel parcel, NodeAccess nodeAccess)
    {
        EnsureOpen();
        var result = NativeMethods.SchemeComputeNodeWithParcel(
            _scheme.Handle, parcel.Handle, nodeAccess);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme compute_node_with_parcel");
        return this;
    }

    public SchemeComputeNodeScope WithTexture(Texture texture, NodeAccess nodeAccess)
    {
        EnsureOpen();
        ArgumentNullException.ThrowIfNull(texture);
        var result = NativeMethods.SchemeComputeNodeWithTexture(
            _scheme.Handle, texture.Handle, nodeAccess);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme compute_node_with_texture");
        return this;
    }

    public SchemeComputeNodeScope WithField(
        Buffer buffer,
        uint unit,
        NodeAccess nodeAccess)
    {
        EnsureOpen();
        ArgumentNullException.ThrowIfNull(buffer);
        var result = NativeMethods.SchemeComputeNodeWithField(
            _scheme.Handle, buffer.Handle, unit, nodeAccess);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme compute_node_with_field");
        return this;
    }

    public SchemeComputeNodeScope WithBufferUnit(
        Buffer buffer,
        uint unit,
        NodeAccess nodeAccess) =>
        WithField(buffer, unit, nodeAccess);

    public SchemeComputeNodeScope WithParam(uint value)
    {
        EnsureOpen();
        var result = NativeMethods.SchemeComputeNodeWithParam(_scheme.Handle, value);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme compute_node_with_param");
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

/// <summary>
/// Active scheme render-pass recording scope. Call <see cref="Dispose"/> to finish the pass.
/// </summary>
public sealed class SchemeRenderPassScope : IDisposable
{
    private readonly Scheme _scheme;
    private bool _finished;

    internal SchemeRenderPassScope(Scheme scheme) => _scheme = scheme;

    public SchemeRenderPassScope WithParcel(Parcel parcel, NodeAccess access)
    {
        EnsureOpen();
        var result = NativeMethods.SchemeRenderPassWithParcel(_scheme.Handle, parcel.Handle, access);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme render_pass_with_parcel");
        return this;
    }

    public SchemeRenderPassScope WithField(Buffer buffer, uint unit, NodeAccess access)
    {
        EnsureOpen();
        ArgumentNullException.ThrowIfNull(buffer);
        var result = NativeMethods.SchemeRenderPassWithField(_scheme.Handle, buffer.Handle, unit, access);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme render_pass_with_field");
        return this;
    }

    public SchemeRenderPassScope WithBufferUnit(Buffer buffer, uint unit, NodeAccess access) =>
        WithField(buffer, unit, access);

    public SchemeRenderPassScope SetPipeline(RenderPipeline pipeline)
    {
        EnsureOpen();
        var result = NativeMethods.SchemeRenderPassSetPipeline(_scheme.Handle, pipeline.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme render_pass_set_pipeline");
        return this;
    }

    public SchemeRenderPassScope SetVertexBuffer(uint slot, Parcel parcel)
    {
        EnsureOpen();
        var result = NativeMethods.SchemeRenderPassSetVertexBufferParcel(_scheme.Handle, slot, parcel.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme render_pass_set_vertex_buffer_parcel");
        return this;
    }

    public SchemeRenderPassScope Draw(uint vertexCount, uint instanceCount = 1, uint firstVertex = 0, uint firstInstance = 0)
    {
        EnsureOpen();
        var result = NativeMethods.SchemeRenderPassDraw(
            _scheme.Handle, firstVertex, vertexCount, firstInstance, instanceCount);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme render_pass_draw");
        return this;
    }

    public SchemeRenderPassScope DrawFullscreen()
    {
        EnsureOpen();
        var result = NativeMethods.SchemeRenderPassDrawFullscreen(_scheme.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme render_pass_draw_fullscreen");
        return this;
    }

    public void Dispose()
    {
        if (_finished)
            return;
        _finished = true;
        var result = NativeMethods.SchemeRenderPassFinish(_scheme.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme render_pass_finish");
    }

    private void EnsureOpen()
    {
        if (_finished)
            throw new InvalidOperationException("Scheme render pass already finished");
    }
}
