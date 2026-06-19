using Goldy.Native;

namespace Goldy;

/// <summary>
/// A GPU surface for zero-copy presentation to a window.
/// 
/// Unlike RenderTarget, a Surface presents directly to the display
/// without any CPU-side copies. Windowed rendering uses
/// <see cref="SwapchainPool"/> + present-on-scheme.
/// </summary>
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
    public static Surface CreateWin32(Device device, nint hwnd)
    {
        device.ThrowIfDisposed();
        
        var handle = NativeMethods.SurfaceCreateWin32(device.Handle, hwnd);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("Surface creation");
        
        return new Surface(handle);
    }

    /// <summary>
    /// Create a surface from an AppKit NSView pointer (macOS).
    /// </summary>
    public static Surface CreateAppKit(Device device, nint nsView)
    {
        device.ThrowIfDisposed();

        var handle = NativeMethods.SurfaceCreateAppKit(device.Handle, nsView);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("Surface creation");

        return new Surface(handle);
    }

    /// <summary>
    /// Create a surface from Wayland wl_display and wl_surface pointers (Linux).
    /// </summary>
    public static Surface CreateWayland(Device device, nint display, nint surface)
    {
        device.ThrowIfDisposed();

        var handle = NativeMethods.SurfaceCreateWayland(device.Handle, display, surface);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("Surface creation");

        return new Surface(handle);
    }

    public uint Width => NativeMethods.SurfaceWidth(Handle);
    public uint Height => NativeMethods.SurfaceHeight(Handle);
    public TextureFormat Format => NativeMethods.SurfaceFormat(Handle);

    public void Resize(uint width, uint height)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (width == 0 || height == 0)
            return;
        
        var result = NativeMethods.SurfaceResize(Handle, width, height);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Surface resize");
    }

    public SurfaceFrame Acquire()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        var frameHandle = NativeMethods.SurfaceAcquire(Handle);
        if (frameHandle == nint.Zero)
            throw GoldyException.FromLastError("Surface acquire");
        
        return new SurfaceFrame(frameHandle, this);
    }

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

    public uint Width => NativeMethods.SurfaceFrameWidth(_handle);
    public uint Height => NativeMethods.SurfaceFrameHeight(_handle);

    internal nint NativeHandle => _handle;

    internal nint TakeHandle()
    {
        var handle = _handle;
        _handle = nint.Zero;
        return handle;
    }
}
