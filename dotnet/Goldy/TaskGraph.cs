using Goldy.Native;

namespace Goldy;

/// <summary>
/// Task graph for explicit GPU scheduling with automatic barrier insertion.
/// </summary>
public sealed class TaskGraph : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    public TaskGraph()
    {
        Handle = NativeMethods.TaskGraphCreate();
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("TaskGraph creation");
    }

    /// <summary>
    /// Reset the graph to empty while retaining internal capacity.
    /// </summary>
    public void Clear()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.TaskGraphClear(Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph clear");
    }

    /// <summary>
    /// Analyze the graph, submit GPU work, and block until complete (headless path).
    /// </summary>
    public void Dispatch(Device device)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        device.ThrowIfDisposed();
        var result = NativeMethods.TaskGraphDispatch(Handle, device.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph dispatch");
    }

    /// <summary>
    /// Declare that this graph will blit to the swapchain at surface submit time.
    /// </summary>
    public SwapchainOutput DeclareSwapchainOutput()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var token = NativeMethods.TaskGraphDeclareSwapchainOutput(Handle);
        if (token == nint.Zero)
            throw GoldyException.FromLastError("TaskGraph declare_swapchain_output");
        return new SwapchainOutput(token);
    }

    /// <summary>
    /// Add a render-target to swapchain blit node.
    /// </summary>
    public void CopyRenderTargetToSwapchain(RenderTarget src, SwapchainOutput swapchain)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.TaskGraphCopyRenderTargetToSwapchain(Handle, src.Handle, swapchain.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph copy_render_target_to_swapchain");
    }

    /// <summary>
    /// Begin recording an offscreen render pass. Finish with <see cref="RenderPassScope.Dispose"/>.
    /// </summary>
    public RenderPassScope RenderPass(string label, RenderTarget target)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.TaskGraphRenderPassBegin(Handle, label, target.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_begin");
        return new RenderPassScope(this);
    }

    internal void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.TaskGraphDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}

/// <summary>
/// Non-owning swapchain output token from <see cref="TaskGraph.DeclareSwapchainOutput"/>.
/// </summary>
public readonly struct SwapchainOutput
{
    internal nint Handle { get; }
    internal SwapchainOutput(nint handle) => Handle = handle;
}

/// <summary>
/// Active render-pass recording scope. Call <see cref="Dispose"/> to finish the pass.
/// </summary>
public sealed class RenderPassScope : IDisposable
{
    private readonly TaskGraph _graph;
    private bool _finished;

    internal RenderPassScope(TaskGraph graph) => _graph = graph;

    public RenderPassScope BindBuffer(Buffer buffer, NodeAccess access)
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassBindBuffer(_graph.Handle, buffer.Handle, access);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_bind_buffer");
        return this;
    }

    public RenderPassScope BindResources(ReadOnlySpan<Buffer> buffers)
    {
        EnsureOpen();
        unsafe
        {
            if (buffers.IsEmpty)
                return this;

            Span<nint> ptrs = stackalloc nint[buffers.Length];
            for (var i = 0; i < buffers.Length; i++)
                ptrs[i] = buffers[i].Handle;

            fixed (nint* p = ptrs)
            {
                var result = NativeMethods.TaskGraphRenderPassBindResources(
                    _graph.Handle, (nint)p, (uint)buffers.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("TaskGraph render_pass_bind_resources");
            }
        }
        return this;
    }

    public RenderPassScope Clear(Color color)
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassClear(_graph.Handle, color);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_clear");
        return this;
    }

    public RenderPassScope ClearDepth(float depth = 1.0f)
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassClearDepth(_graph.Handle, depth);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_clear_depth");
        return this;
    }

    public RenderPassScope SetPipeline(RenderPipeline pipeline)
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassSetPipeline(_graph.Handle, pipeline.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_set_pipeline");
        return this;
    }

    public RenderPassScope SetVertexBuffer(uint slot, Buffer buffer)
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassSetVertexBuffer(_graph.Handle, slot, buffer.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_set_vertex_buffer");
        return this;
    }

    public RenderPassScope SetVertexBuffer(uint slot, Buffer buffer, ulong offset)
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassSetVertexBufferOffset(
            _graph.Handle, slot, buffer.Handle, offset);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_set_vertex_buffer_offset");
        return this;
    }

    public RenderPassScope SetIndexBuffer(Buffer buffer, IndexFormat format)
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassSetIndexBuffer(_graph.Handle, buffer.Handle, format);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_set_index_buffer");
        return this;
    }

    public RenderPassScope Draw(uint vertexCount, uint instanceCount = 1, uint firstVertex = 0, uint firstInstance = 0)
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassDraw(
            _graph.Handle, firstVertex, vertexCount, firstInstance, instanceCount);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_draw");
        return this;
    }

    public RenderPassScope DrawIndexed(
        uint indexCount,
        uint instanceCount = 1,
        uint firstIndex = 0,
        int baseVertex = 0,
        uint firstInstance = 0)
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassDrawIndexed(
            _graph.Handle, firstIndex, indexCount, baseVertex, firstInstance, instanceCount);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_draw_indexed");
        return this;
    }

    public RenderPassScope DrawFullscreen()
    {
        EnsureOpen();
        var result = NativeMethods.TaskGraphRenderPassDrawFullscreen(_graph.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_draw_fullscreen");
        return this;
    }

    public void Dispose()
    {
        if (_finished)
            return;
        _finished = true;
        var result = NativeMethods.TaskGraphRenderPassFinish(_graph.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("TaskGraph render_pass_finish");
    }

    private void EnsureOpen()
    {
        if (_finished)
            throw new InvalidOperationException("Render pass already finished");
    }
}
