using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// Cloneable device-scoped machine root. Owns retained parcels, shaders,
/// pipelines, capabilities, and diagnostics.
/// </summary>
public sealed class Runtime : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    internal Runtime(nint handle)
    {
        Handle = handle;
    }

    /// <summary>
    /// Get the adapter ID this runtime was created on.
    /// </summary>
    public uint AdapterId => NativeMethods.RuntimeAdapterId(Handle);

    /// <summary>
    /// Check if the runtime is still valid.
    /// </summary>
    public bool IsValid => NativeMethods.RuntimeIsValid(Handle);

    /// <summary>
    /// Check if a shader library is registered.
    /// </summary>
    public bool HasLibrary(string name) => NativeMethods.RuntimeHasLibrary(Handle, name);

    internal void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// Create a GPU submission context for retained schemes.
    /// </summary>
    public Context CreateContext() => Context.Create(this);

    /// <summary>
    /// Begin building a partitioned record buffer.
    /// </summary>
    public RecordBuilder Record() => new();

    /// <summary>
    /// Acquire an uninitialized retained buffer.
    /// </summary>
    public Buffer AcquireBuffer(ulong size, BufferKind access, uint elementStride = 0)
    {
        ThrowIfDisposed();
        unsafe
        {
            var buffer = NativeMethods.RuntimeAcquireBuffer(
                Handle, size, access, elementStride, nint.Zero, 0);
            if (buffer == nint.Zero)
                throw GoldyException.FromLastError("Runtime acquire_buffer");
            return new Buffer(buffer);
        }
    }

    /// <summary>
    /// Acquire a retained buffer initialized with raw bytes.
    /// </summary>
    public Buffer AcquireBuffer(ReadOnlySpan<byte> data, BufferKind access, uint elementStride = 0)
    {
        ThrowIfDisposed();
        unsafe
        {
            fixed (byte* ptr = data)
            {
                var buffer = NativeMethods.RuntimeAcquireBuffer(
                    Handle, (ulong)data.Length, access, elementStride, (nint)ptr, (nuint)data.Length);
                if (buffer == nint.Zero)
                    throw GoldyException.FromLastError("Runtime acquire_buffer");
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
        ThrowIfDisposed();
        var texture = NativeMethods.RuntimeAcquireTexture(
            Handle, width, height, format, access, flags, nint.Zero, 0);
        if (texture == nint.Zero)
            throw GoldyException.FromLastError("Runtime acquire_texture");
        return new Texture(texture);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.RuntimeDestroy(Handle);
            _disposed = true;
        }
    }
}

