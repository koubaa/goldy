using Goldy.Native;

namespace Goldy;

/// <summary>
/// A GPU surface for zero-copy presentation to a window.
/// 
/// Unlike RenderTarget, a Surface presents directly to the display
/// without any CPU-side copies. This is the optimal path for windowed rendering.
/// </summary>
/// <example>
/// <code>
/// // Create surface from a window handle (platform-specific)
/// var surface = Surface.CreateWin32(device, windowHandle);
/// 
/// // Render loop
/// while (running)
/// {
///     // Acquire next frame from swapchain
///     var frame = surface.Acquire();
///     
///     // Build render commands
///     var encoder = new CommandEncoder();
///     encoder.Clear(Color.CornflowerBlue);
///     encoder.SetPipeline(pipeline);
///     encoder.SetVertexBuffer(0, vertexBuffer);
///     encoder.Draw(3);
///     
///     // Render to swapchain image (zero-copy!)
///     frame.Render(encoder);
///     
///     // Present to screen
///     surface.Present(frame);
/// }
/// </code>
/// </example>
public sealed class Surface : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    private Surface(nint handle)
    {
        Handle = handle;
    }

    /// <summary>
    /// Create a surface from a Win32 window handle (HWND).
    /// </summary>
    /// <param name="device">The GPU device to use for rendering.</param>
    /// <param name="hwnd">The Win32 window handle (HWND).</param>
    /// <returns>A new Surface for the window.</returns>
    /// <exception cref="GoldyException">Thrown if surface creation fails.</exception>
    /// <remarks>
    /// The window must remain valid for the lifetime of the surface.
    /// Currently only Vulkan backend is supported for C# windowed rendering.
    /// </remarks>
    public static Surface CreateWin32(Device device, nint hwnd)
    {
        device.ThrowIfDisposed();
        
        var handle = NativeMethods.SurfaceCreateWin32(device.Handle, hwnd);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("Surface creation");
        
        return new Surface(handle);
    }

    /// <summary>
    /// Get the surface width in pixels.
    /// </summary>
    public uint Width => NativeMethods.SurfaceWidth(Handle);

    /// <summary>
    /// Get the surface height in pixels.
    /// </summary>
    public uint Height => NativeMethods.SurfaceHeight(Handle);

    /// <summary>
    /// Get the swapchain texture format.
    /// Use this to set RenderPipelineDesc.TargetFormat when rendering to this surface.
    /// </summary>
    public TextureFormat Format => NativeMethods.SurfaceFormat(Handle);

    /// <summary>
    /// Resize the surface.
    /// Call this when the window is resized. This recreates the swapchain.
    /// </summary>
    /// <param name="width">New width in pixels.</param>
    /// <param name="height">New height in pixels.</param>
    /// <exception cref="GoldyException">Thrown if resize fails.</exception>
    public void Resize(uint width, uint height)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        // Ignore zero-sized resize (minimized window)
        if (width == 0 || height == 0)
            return;
        
        var result = NativeMethods.SurfaceResize(Handle, width, height);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Surface resize");
    }

    /// <summary>
    /// Acquire the next frame to render to.
    /// This blocks until a frame is available from the swapchain.
    /// </summary>
    /// <returns>A SurfaceFrame that can be rendered to and presented.</returns>
    /// <exception cref="GoldyException">Thrown if frame acquisition fails.</exception>
    public SurfaceFrame Acquire()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        var frameHandle = NativeMethods.SurfaceAcquire(Handle);
        if (frameHandle == nint.Zero)
            throw GoldyException.FromLastError("Surface acquire");
        
        return new SurfaceFrame(frameHandle, this);
    }

    /// <summary>
    /// Present a rendered frame to the screen.
    /// This submits the frame to be displayed and returns immediately.
    /// </summary>
    /// <param name="frame">The frame to present. This consumes the frame.</param>
    /// <exception cref="GoldyException">Thrown if presentation fails.</exception>
    public void Present(SurfaceFrame frame)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        var frameHandle = frame.TakeHandle();
        var result = NativeMethods.SurfacePresent(Handle, frameHandle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Surface present");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.SurfaceDestroy(Handle);
            _disposed = true;
        }
    }
}

/// <summary>
/// A frame acquired from a surface, ready for rendering.
/// After rendering, pass to Surface.Present() to display it.
/// </summary>
public sealed class SurfaceFrame
{
    private nint _handle;
    private readonly Surface _surface;

    internal SurfaceFrame(nint handle, Surface surface)
    {
        _handle = handle;
        _surface = surface;
    }

    /// <summary>
    /// Get the frame width in pixels.
    /// </summary>
    public uint Width => NativeMethods.SurfaceFrameWidth(_handle);

    /// <summary>
    /// Get the frame height in pixels.
    /// </summary>
    public uint Height => NativeMethods.SurfaceFrameHeight(_handle);

    /// <summary>
    /// Render commands to this frame.
    /// This consumes the encoder.
    /// </summary>
    /// <param name="encoder">The command encoder with recorded commands.</param>
    /// <exception cref="GoldyException">Thrown if rendering fails.</exception>
    public void Render(CommandEncoder encoder)
    {
        if (_handle == nint.Zero)
            throw new InvalidOperationException("Frame has already been consumed");
        
        var encoderHandle = encoder.TakeHandle();
        var result = NativeMethods.SurfaceFrameRender(_handle, encoderHandle);
        // Note: encoder handle is now owned by native code
        
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("SurfaceFrame render");
    }

    /// <summary>
    /// Take ownership of the native handle (for internal use).
    /// </summary>
    internal nint TakeHandle()
    {
        var handle = _handle;
        _handle = nint.Zero;
        return handle;
    }
}
