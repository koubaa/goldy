using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// Deed-governed pool for retained GPU parcels.
/// </summary>
public sealed class RetainedPool : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    public RetainedPool(Device device)
    {
        device.ThrowIfDisposed();
        Handle = NativeMethods.RetainedPoolCreate(device.Handle);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("RetainedPool creation");
    }

    /// <summary>
    /// Begin building a partitioned record buffer.
    /// </summary>
    public RecordBuilder Record() => new();

    /// <summary>
    /// Acquire an uninitialized retained buffer.
    /// </summary>
    public Buffer AcquireBuffer(ulong size, BufferKind access, uint elementStride = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            var buffer = NativeMethods.RetainedPoolAcquireBuffer(
                Handle, size, access, elementStride, nint.Zero, 0);
            if (buffer == nint.Zero)
                throw GoldyException.FromLastError("RetainedPool acquire_buffer");
            return new Buffer(buffer);
        }
    }

    /// <summary>
    /// Acquire a retained buffer initialized with raw bytes.
    /// </summary>
    public Buffer AcquireBuffer(ReadOnlySpan<byte> data, BufferKind access, uint elementStride = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            fixed (byte* ptr = data)
            {
                var buffer = NativeMethods.RetainedPoolAcquireBuffer(
                    Handle, (ulong)data.Length, access, elementStride, (nint)ptr, (nuint)data.Length);
                if (buffer == nint.Zero)
                    throw GoldyException.FromLastError("RetainedPool acquire_buffer");
                return new Buffer(buffer);
            }
        }
    }

    /// <summary>
    /// Acquire a retained buffer initialized with typed data.
    /// </summary>
    public Buffer AcquireBuffer<T>(ReadOnlySpan<T> data, BufferKind access) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(data);
        var stride = (uint)Marshal.SizeOf<T>();
        return AcquireBuffer(bytes, access, stride);
    }

    /// <summary>
    /// Acquire an uninitialized retained texture parcel.
    /// </summary>
    public Texture AcquireTexture(
        uint width,
        uint height,
        TextureFormat format,
        TextureKind access,
        TextureFlags flags = TextureFlags.None)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var texture = NativeMethods.RetainedPoolAcquireTexture(
            Handle, width, height, format, access, flags, nint.Zero, 0);
        if (texture == nint.Zero)
            throw GoldyException.FromLastError("RetainedPool acquire_texture");
        return new Texture(texture);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.RetainedPoolDestroy(Handle);
            _disposed = true;
        }
    }
}
