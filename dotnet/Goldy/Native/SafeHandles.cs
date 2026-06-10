using System.Runtime.InteropServices;

namespace Goldy.Native;

/// <summary>
/// SafeHandle for Goldy Instance.
/// </summary>
internal sealed class InstanceHandle : SafeHandle
{
    public InstanceHandle() : base(nint.Zero, true) { }
    
    public override bool IsInvalid => handle == nint.Zero;
    
    protected override bool ReleaseHandle()
    {
        NativeMethods.InstanceDestroy(handle);
        return true;
    }
}

/// <summary>
/// SafeHandle for Goldy Device.
/// </summary>
internal sealed class DeviceHandle : SafeHandle
{
    public DeviceHandle() : base(nint.Zero, true) { }
    
    public override bool IsInvalid => handle == nint.Zero;
    
    protected override bool ReleaseHandle()
    {
        NativeMethods.DeviceDestroy(handle);
        return true;
    }
}

/// <summary>
/// SafeHandle for Goldy RenderTarget.
/// </summary>
internal sealed class RenderTargetHandle : SafeHandle
{
    public RenderTargetHandle() : base(nint.Zero, true) { }
    
    public override bool IsInvalid => handle == nint.Zero;
    
    protected override bool ReleaseHandle()
    {
        NativeMethods.RenderTargetDestroy(handle);
        return true;
    }
}

/// <summary>
/// SafeHandle for Goldy ShaderModule.
/// </summary>
internal sealed class ShaderModuleHandle : SafeHandle
{
    public ShaderModuleHandle() : base(nint.Zero, true) { }
    
    public override bool IsInvalid => handle == nint.Zero;
    
    protected override bool ReleaseHandle()
    {
        NativeMethods.ShaderDestroy(handle);
        return true;
    }
}

/// <summary>
/// SafeHandle for Goldy RenderPipeline.
/// </summary>
internal sealed class RenderPipelineHandle : SafeHandle
{
    public RenderPipelineHandle() : base(nint.Zero, true) { }
    
    public override bool IsInvalid => handle == nint.Zero;
    
    protected override bool ReleaseHandle()
    {
        NativeMethods.RenderPipelineDestroy(handle);
        return true;
    }
}

/// <summary>
/// SafeHandle for Goldy ComputePipeline.
/// </summary>
internal sealed class ComputePipelineHandle : SafeHandle
{
    public ComputePipelineHandle() : base(nint.Zero, true) { }
    
    public override bool IsInvalid => handle == nint.Zero;
    
    protected override bool ReleaseHandle()
    {
        NativeMethods.ComputePipelineDestroy(handle);
        return true;
    }
}

/// <summary>
/// SafeHandle for Goldy Sampler.
/// </summary>
internal sealed class SamplerHandle : SafeHandle
{
    public SamplerHandle() : base(nint.Zero, true) { }
    
    public override bool IsInvalid => handle == nint.Zero;
    
    protected override bool ReleaseHandle()
    {
        NativeMethods.SamplerDestroy(handle);
        return true;
    }
}

/// <summary>
/// SafeHandle for Goldy Surface.
/// </summary>
internal sealed class SurfaceHandle : SafeHandle
{
    public SurfaceHandle() : base(nint.Zero, true) { }
    
    public override bool IsInvalid => handle == nint.Zero;
    
    protected override bool ReleaseHandle()
    {
        NativeMethods.SurfaceDestroy(handle);
        return true;
    }
}

