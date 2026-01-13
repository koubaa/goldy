using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// A compiled shader module.
/// </summary>
public sealed class ShaderModule : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    /// <summary>
    /// Create a shader module from Slang source.
    /// </summary>
    public ShaderModule(Device device, string source)
    {
        device.ThrowIfDisposed();
        
        Handle = NativeMethods.ShaderCreate(device.Handle, source);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("Shader compilation");
    }

    /// <summary>
    /// Get the built-in vertex color 2D shader source.
    /// </summary>
    public static string BuiltinVertexColor2D
    {
        get
        {
            var ptr = NativeMethods.ShaderBuiltinVertexColor2D();
            return Marshal.PtrToStringUTF8(ptr) ?? "";
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.ShaderDestroy(Handle);
            _disposed = true;
        }
    }
}

