using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// GPU instance - entry point for Goldy.
/// Create an instance to enumerate adapters and create devices.
/// </summary>
public sealed class Instance : IDisposable
{
    private readonly nint _handle;
    private bool _disposed;

    /// <summary>
    /// Create a new Goldy instance.
    /// </summary>
    public Instance()
    {
        _handle = NativeMethods.InstanceCreate();
        if (_handle == nint.Zero)
            throw GoldyException.FromLastError("Instance creation");
    }

    /// <summary>
    /// Get the backend type (Vulkan, Metal, DX12).
    /// </summary>
    public BackendType BackendType => NativeMethods.InstanceBackendType(_handle);

    /// <summary>
    /// Enumerate available GPU adapters.
    /// </summary>
    public AdapterInfo[] EnumerateAdapters()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        uint count = NativeMethods.InstanceAdapterCount(_handle);
        var adapters = new AdapterInfo[count];
        
        for (uint i = 0; i < count; i++)
        {
            var result = NativeMethods.InstanceGetAdapter(_handle, i, out var info);
            if (result != GoldyResult.Ok)
                throw GoldyException.FromLastError("Enumerate adapters");
            
            adapters[i] = new AdapterInfo(info.Id, info.DeviceType, info.GetName(), info.GetVendor());
        }
        
        return adapters;
    }

    /// <summary>
    /// Create a device on the first adapter matching the given type.
    /// </summary>
    public Device CreateDevice(DeviceType preferredType = DeviceType.DiscreteGpu)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        var handle = NativeMethods.InstanceCreateDevice(_handle, preferredType);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("Device creation");
        
        return new Device(handle);
    }

    /// <summary>
    /// Create a device on a specific adapter by ID.
    /// </summary>
    public Device CreateDeviceForAdapter(uint adapterId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        var handle = NativeMethods.InstanceCreateDeviceForAdapter(_handle, adapterId);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("Device creation");
        
        return new Device(handle);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.InstanceDestroy(_handle);
            _disposed = true;
        }
    }
}

/// <summary>
/// Information about a GPU adapter.
/// </summary>
public readonly record struct AdapterInfo(uint Id, DeviceType DeviceType, string Name, string Vendor);

