using Goldy.Native;

namespace Goldy;

/// <summary>
/// A GPU device - used to create resources and render.
/// The Device is the primary interface for GPU operations.
/// </summary>
public sealed class Device : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    internal Device(nint handle)
    {
        Handle = handle;
    }

    /// <summary>
    /// Get the adapter ID this device was created on.
    /// </summary>
    public uint AdapterId => NativeMethods.DeviceAdapterId(Handle);

    /// <summary>
    /// Check if the device is still valid.
    /// </summary>
    public bool IsValid => NativeMethods.DeviceIsValid(Handle);

    /// <summary>
    /// Check if a shader library is registered.
    /// </summary>
    public bool HasLibrary(string name) => NativeMethods.DeviceHasLibrary(Handle, name);

    internal void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.DeviceDestroy(Handle);
            _disposed = true;
        }
    }
}

