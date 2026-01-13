using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// A GPU buffer.
/// </summary>
public sealed class Buffer : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    /// <summary>
    /// Create a new buffer.
    /// </summary>
    public Buffer(Device device, ulong size, BufferUsage usage)
    {
        device.ThrowIfDisposed();
        
        Handle = NativeMethods.BufferCreate(device.Handle, size, usage);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("Buffer creation");
        
        Size = size;
        Usage = usage;
    }

    /// <summary>
    /// Create a buffer initialized with data.
    /// </summary>
    public Buffer(Device device, ReadOnlySpan<byte> data, BufferUsage usage)
    {
        device.ThrowIfDisposed();
        
        unsafe
        {
            fixed (byte* ptr = data)
            {
                Handle = NativeMethods.BufferCreateWithData(device.Handle, (nint)ptr, (nuint)data.Length, usage);
            }
        }
        
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("Buffer creation");
        
        Size = (ulong)data.Length;
        Usage = usage;
    }

    /// <summary>
    /// Create a buffer initialized with typed data.
    /// </summary>
    public static Buffer WithData<T>(Device device, ReadOnlySpan<T> data, BufferUsage usage) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(data);
        return new Buffer(device, bytes, usage);
    }

    /// <summary>
    /// Get the buffer size in bytes.
    /// </summary>
    public ulong Size { get; }

    /// <summary>
    /// Get the buffer usage flags.
    /// </summary>
    public BufferUsage Usage { get; }

    /// <summary>
    /// Write data to the buffer.
    /// </summary>
    public void Write(ulong offset, ReadOnlySpan<byte> data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        unsafe
        {
            fixed (byte* ptr = data)
            {
                var result = NativeMethods.BufferWrite(Handle, offset, (nint)ptr, (nuint)data.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("Buffer write");
            }
        }
    }

    /// <summary>
    /// Write typed data to the buffer.
    /// </summary>
    public void Write<T>(ulong offset, ReadOnlySpan<T> data) where T : unmanaged
    {
        Write(offset, MemoryMarshal.AsBytes(data));
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.BufferDestroy(Handle);
            _disposed = true;
        }
    }
}

