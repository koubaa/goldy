using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// Command encoder for recording GPU commands.
/// CommandEncoder is completely lock-free and does not interact with the GPU backend
/// until submitted via RenderTarget.Render() or Surface.Present().
/// </summary>
public sealed class CommandEncoder
{
    private nint _handle;

    /// <summary>
    /// Create a new command encoder.
    /// </summary>
    public CommandEncoder()
    {
        _handle = NativeMethods.EncoderCreate();
        if (_handle == nint.Zero)
            throw new GoldyException("Failed to create command encoder");
    }

    /// <summary>
    /// Clear the color render target to a color.
    /// </summary>
    public void Clear(Color color)
    {
        EnsureNotConsumed();
        NativeMethods.EncoderClear(_handle, color);
    }

    /// <summary>
    /// Clear the depth buffer to a value.
    /// The default depth clear value is 1.0 (far plane).
    /// </summary>
    public void ClearDepth(float depth = 1.0f)
    {
        EnsureNotConsumed();
        NativeMethods.EncoderClearDepth(_handle, depth);
    }

    /// <summary>
    /// Set the active render pipeline.
    /// </summary>
    public void SetPipeline(RenderPipeline pipeline)
    {
        EnsureNotConsumed();
        NativeMethods.EncoderSetPipeline(_handle, pipeline.Handle);
    }

    /// <summary>
    /// Set a vertex buffer.
    /// </summary>
    public void SetVertexBuffer(uint slot, Buffer buffer)
    {
        EnsureNotConsumed();
        NativeMethods.EncoderSetVertexBuffer(_handle, slot, buffer.Handle);
    }

    /// <summary>
    /// Set a vertex buffer with an offset.
    /// </summary>
    public void SetVertexBuffer(uint slot, Buffer buffer, ulong offset)
    {
        EnsureNotConsumed();
        NativeMethods.EncoderSetVertexBufferOffset(_handle, slot, buffer.Handle, offset);
    }

    /// <summary>
    /// Bind resource slots for rendering.
    /// The indices are bound in order, so buffers[0] becomes slot 0,
    /// buffers[1] becomes slot 1, etc.
    /// </summary>
    /// <param name="buffers">Buffers to bind to shader resource slots.</param>
    public void BindResources(params Buffer[] buffers)
    {
        EnsureNotConsumed();
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
                NativeMethods.EncoderBindResources(_handle, (nint)ptr, (uint)buffers.Length);
            }
        }
    }

    /// <summary>
    /// Set an index buffer.
    /// </summary>
    public void SetIndexBuffer(Buffer buffer, IndexFormat format)
    {
        EnsureNotConsumed();
        NativeMethods.EncoderSetIndexBuffer(_handle, buffer.Handle, format);
    }

    /// <summary>
    /// Draw primitives.
    /// </summary>
    public void Draw(uint vertexCount, uint instanceCount = 1, uint firstVertex = 0, uint firstInstance = 0)
    {
        EnsureNotConsumed();
        NativeMethods.EncoderDraw(_handle, firstVertex, vertexCount, firstInstance, instanceCount);
    }

    /// <summary>
    /// Draw indexed primitives.
    /// </summary>
    public void DrawIndexed(uint indexCount, uint instanceCount = 1, uint firstIndex = 0, int baseVertex = 0, uint firstInstance = 0)
    {
        EnsureNotConsumed();
        NativeMethods.EncoderDrawIndexed(_handle, firstIndex, indexCount, baseVertex, firstInstance, instanceCount);
    }

    /// <summary>
    /// Take ownership of the native handle (for submission).
    /// After calling this, the encoder cannot be used.
    /// </summary>
    internal nint TakeHandle()
    {
        EnsureNotConsumed();
        var handle = _handle;
        _handle = nint.Zero;
        return handle;
    }

    private void EnsureNotConsumed()
    {
        if (_handle == nint.Zero)
            throw new InvalidOperationException("CommandEncoder has already been submitted");
    }
}

