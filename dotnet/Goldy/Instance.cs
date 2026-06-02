using System.Linq;
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
    /// Request the best available GPU adapter (highest performance by default).
    /// </summary>
    public Adapter RequestAdapter()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var adapters = EnumerateAdapters();
        if (adapters.Length == 0)
            throw GoldyException.FromLastError("Request adapter");

        var adapter = adapters.FirstOrDefault(a => a.DeviceType == DeviceType.DiscreteGpu);
        if (string.IsNullOrEmpty(adapter.Name))
            adapter = adapters[0];

        return new Adapter(this, adapter);
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

/// <summary>
/// A selected GPU adapter used to create devices.
/// </summary>
public sealed class Adapter
{
    private readonly Instance _instance;
    private readonly AdapterInfo _info;

    internal Adapter(Instance instance, AdapterInfo info)
    {
        _instance = instance;
        _info = info;
    }

    public uint Id => _info.Id;
    public DeviceType DeviceType => _info.DeviceType;
    public string Name => _info.Name;
    public string Vendor => _info.Vendor;

    public Device RequestDevice() => _instance.CreateDeviceForAdapter(_info.Id);
}

