using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// Builder for a retained record buffer (one backing buffer, multiple sub-views).
/// </summary>
public sealed class RecordBuilder : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    public RecordBuilder()
    {
        Handle = NativeMethods.RecordBuilderCreate();
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("RecordBuilder creation");
    }

    /// <summary>
    /// Upload typed data into the next ordinal field.
    /// </summary>
    public uint Emplace<T>(ReadOnlySpan<T> data) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var bytes = MemoryMarshal.AsBytes(data);
        var stride = (uint)Marshal.SizeOf<T>();
        var count = (ulong)(bytes.Length / stride);
        return EmplaceBytes(null, bytes, count, stride);
    }

    /// <summary>
    /// Upload raw bytes into the next ordinal field.
    /// </summary>
    public uint Emplace(ReadOnlySpan<byte> data, ulong elementCount, uint elementStride) =>
        EmplaceBytes(null, data, elementCount, elementStride);

    /// <summary>
    /// Define a named field and upload typed data.
    /// </summary>
    public uint EmplaceField<T>(string name, ReadOnlySpan<T> data) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var bytes = MemoryMarshal.AsBytes(data);
        var stride = (uint)Marshal.SizeOf<T>();
        var count = (ulong)(bytes.Length / stride);
        return EmplaceBytes(name, bytes, count, stride);
    }

    /// <summary>
    /// Reserve the next ordinal field without uploading data.
    /// </summary>
    public uint Reserve(ulong elementCount, uint elementStride)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var slot = NativeMethods.RecordBuilderReserve(Handle, null, elementCount, elementStride);
        if (slot == uint.MaxValue)
            throw GoldyException.FromLastError("RecordBuilder reserve");
        return slot;
    }

    /// <summary>
    /// Allocate the backing buffer and return the partitioned record.
    /// </summary>
    public Buffer Build(RetainedPool pool)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(pool);
        var buffer = NativeMethods.RecordBuilderBuild(Handle, pool.Handle);
        if (buffer == nint.Zero)
            throw GoldyException.FromLastError("RecordBuilder build");
        _disposed = true;
        return new Buffer(buffer);
    }

    private uint EmplaceBytes(string? name, ReadOnlySpan<byte> data, ulong elementCount, uint elementStride)
    {
        unsafe
        {
            fixed (byte* p = data)
            {
                var slot = NativeMethods.RecordBuilderEmplace(
                    Handle, name, (nint)p, (nuint)data.Length, elementCount, elementStride);
                if (slot == uint.MaxValue)
                    throw GoldyException.FromLastError("RecordBuilder emplace");
                return slot;
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.RecordBuilderDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}
