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
    /// Create a new buffer with the specified access pattern.
    /// </summary>
    /// <param name="device">The GPU device to create the buffer on.</param>
    /// <param name="size">Size in bytes.</param>
    /// <param name="access">Access pattern (Scattered for general data, Broadcast for uniforms).</param>
    public Buffer(Device device, ulong size, BufferKind access)
    {
        device.ThrowIfDisposed();
        
        Handle = NativeMethods.BufferCreate(device.Handle, size, access);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("Buffer creation");
        
        Size = size;
        Access = access;
    }

    /// <summary>
    /// Create a buffer initialized with data.
    /// </summary>
    /// <param name="device">The GPU device to create the buffer on.</param>
    /// <param name="data">Initial data to upload.</param>
    /// <param name="access">Access pattern (Scattered for general data, Broadcast for uniforms).</param>
    public Buffer(Device device, ReadOnlySpan<byte> data, BufferKind access)
    {
        device.ThrowIfDisposed();
        
        unsafe
        {
            fixed (byte* ptr = data)
            {
                Handle = NativeMethods.BufferCreateWithData(device.Handle, (nint)ptr, (nuint)data.Length, access);
            }
        }
        
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("Buffer creation");
        
        Size = (ulong)data.Length;
        Access = access;
    }

    /// <summary>
    /// Create a buffer initialized with typed data.
    /// </summary>
    public static Buffer WithData<T>(Device device, ReadOnlySpan<T> data, BufferKind access) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(data);
        var stride = (uint)Marshal.SizeOf<T>();
        return WithDataStride(device, bytes, access, stride);
    }

    /// <summary>
    /// Create a buffer initialized with raw bytes and an explicit element stride.
    /// </summary>
    public static Buffer WithDataStride(Device device, ReadOnlySpan<byte> data, BufferKind access, uint elementStride)
    {
        device.ThrowIfDisposed();

        nint handle;
        unsafe
        {
            fixed (byte* ptr = data)
            {
                handle = NativeMethods.BufferCreateWithDataStride(
                    device.Handle, (nint)ptr, (nuint)data.Length, access, elementStride);
            }
        }

        if (handle == nint.Zero)
            throw GoldyException.FromLastError("Buffer creation");

        var buffer = new Buffer(handle, (ulong)data.Length, access);
        return buffer;
    }

    private Buffer(nint handle, ulong size, BufferKind access)
    {
        Handle = handle;
        Size = size;
        Access = access;
    }

    /// <summary>
    /// Get the buffer size in bytes.
    /// </summary>
    public ulong Size { get; }

    /// <summary>
    /// Get the buffer's access pattern.
    /// </summary>
    public BufferKind Access { get; }

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

    /// <summary>
    /// Bindless resource slot index for shader binding.
    /// </summary>
    public uint ResourceIndex(ResourceAccess access)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var idx = NativeMethods.BufferResourceIndex(Handle, access);
        if (idx == uint.MaxValue)
            throw GoldyException.FromLastError("Buffer resource_index");
        return idx;
    }

    /// <summary>
    /// Read buffer contents back to CPU memory from offset 0.
    /// </summary>
    public byte[] ReadToCpu(Device device)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        device.ThrowIfDisposed();

        var output = new byte[Size];
        unsafe
        {
            fixed (byte* p = output)
            {
                var result = NativeMethods.BufferReadToCpu(Handle, device.Handle, (nint)p, (nuint)output.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("Buffer read_to_cpu");
            }
        }
        return output;
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

