using System.Runtime.InteropServices;

namespace Goldy.Native;

/// <summary>
/// Native FFI methods for the Goldy GPU library.
/// Uses .NET 7+ source-generated P/Invoke for optimal performance.
/// </summary>
internal static partial class NativeMethods
{
    private const string LibName = "goldy_ffi";

    // ========================================================================
    // Error handling
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_get_last_error")]
    internal static partial nint GetLastError();

    [LibraryImport(LibName, EntryPoint = "goldy_clear_error")]
    internal static partial void ClearError();

    internal static string? GetLastErrorString()
    {
        var ptr = GetLastError();
        return ptr == nint.Zero ? null : Marshal.PtrToStringUTF8(ptr);
    }

    // ========================================================================
    // Instance
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_instance_create")]
    internal static partial nint InstanceCreate();

    [LibraryImport(LibName, EntryPoint = "goldy_instance_destroy")]
    internal static partial void InstanceDestroy(nint instance);

    [LibraryImport(LibName, EntryPoint = "goldy_instance_backend_type")]
    internal static partial BackendType InstanceBackendType(nint instance);

    [LibraryImport(LibName, EntryPoint = "goldy_instance_adapter_count")]
    internal static partial uint InstanceAdapterCount(nint instance);

    [LibraryImport(LibName, EntryPoint = "goldy_instance_get_adapter")]
    internal static partial GoldyResult InstanceGetAdapter(nint instance, uint index, out AdapterInfoNative info);

    [LibraryImport(LibName, EntryPoint = "goldy_instance_create_device")]
    internal static partial nint InstanceCreateDevice(nint instance, DeviceType preferredType);

    [LibraryImport(LibName, EntryPoint = "goldy_instance_create_device_for_adapter")]
    internal static partial nint InstanceCreateDeviceForAdapter(nint instance, uint adapterId);

    // ========================================================================
    // Device
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_device_destroy")]
    internal static partial void DeviceDestroy(nint device);

    [LibraryImport(LibName, EntryPoint = "goldy_device_adapter_id")]
    internal static partial uint DeviceAdapterId(nint device);

    [LibraryImport(LibName, EntryPoint = "goldy_device_is_valid")]
    [return: MarshalAs(UnmanagedType.U1)]
    internal static partial bool DeviceIsValid(nint device);

    [LibraryImport(LibName, EntryPoint = "goldy_device_has_library", StringMarshalling = StringMarshalling.Utf8)]
    [return: MarshalAs(UnmanagedType.U1)]
    internal static partial bool DeviceHasLibrary(nint device, string name);

    // ========================================================================
    // Buffer
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_create")]
    internal static partial nint BufferCreate(nint device, ulong size, DataAccess access);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_create_with_data")]
    internal static partial nint BufferCreateWithData(nint device, nint data, nuint size, DataAccess access);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_destroy")]
    internal static partial void BufferDestroy(nint buffer);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_write")]
    internal static partial GoldyResult BufferWrite(nint buffer, ulong offset, nint data, nuint size);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_size")]
    internal static partial ulong BufferSize(nint buffer);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_access")]
    internal static partial DataAccess BufferAccess(nint buffer);

    // ========================================================================
    // RenderTarget
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_create")]
    internal static partial nint RenderTargetCreate(nint device, uint width, uint height, TextureFormat format);

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_create_with_depth")]
    internal static partial nint RenderTargetCreateWithDepth(nint device, uint width, uint height, TextureFormat colorFormat, DepthFormat depthFormat);

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_destroy")]
    internal static partial void RenderTargetDestroy(nint target);

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_width")]
    internal static partial uint RenderTargetWidth(nint target);

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_height")]
    internal static partial uint RenderTargetHeight(nint target);

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_format")]
    internal static partial TextureFormat RenderTargetFormat(nint target);

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_has_depth")]
    [return: MarshalAs(UnmanagedType.U1)]
    internal static partial bool RenderTargetHasDepth(nint target);

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_buffer_size")]
    internal static partial nuint RenderTargetBufferSize(nint target);

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_render")]
    internal static partial GoldyResult RenderTargetRender(nint target, nint encoder);

    [LibraryImport(LibName, EntryPoint = "goldy_render_target_read_to_buffer")]
    internal static partial GoldyResult RenderTargetReadToBuffer(nint target, nint output, nuint outputSize);

    // ========================================================================
    // Shader
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_shader_create", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint ShaderCreate(nint device, string source);

    [LibraryImport(LibName, EntryPoint = "goldy_shader_destroy")]
    internal static partial void ShaderDestroy(nint shader);

    [LibraryImport(LibName, EntryPoint = "goldy_shader_builtin_vertex_color_2d")]
    internal static partial nint ShaderBuiltinVertexColor2D();

    // ========================================================================
    // Pipeline
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_render_pipeline_create")]
    internal static partial nint RenderPipelineCreate(nint device, nint vertexShader, nint fragmentShader, in RenderPipelineDescNative desc);

    [LibraryImport(LibName, EntryPoint = "goldy_render_pipeline_destroy")]
    internal static partial void RenderPipelineDestroy(nint pipeline);

    // ========================================================================
    // CommandEncoder
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_create")]
    internal static partial nint EncoderCreate();

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_destroy")]
    internal static partial void EncoderDestroy(nint encoder);

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_clear")]
    internal static partial void EncoderClear(nint encoder, Color color);

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_clear_depth")]
    internal static partial void EncoderClearDepth(nint encoder, float depth);

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_set_pipeline")]
    internal static partial void EncoderSetPipeline(nint encoder, nint pipeline);

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_set_vertex_buffer")]
    internal static partial void EncoderSetVertexBuffer(nint encoder, uint slot, nint buffer);

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_set_vertex_buffer_offset")]
    internal static partial void EncoderSetVertexBufferOffset(nint encoder, uint slot, nint buffer, ulong offset);

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_set_index_buffer")]
    internal static partial void EncoderSetIndexBuffer(nint encoder, nint buffer, IndexFormat format);

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_draw")]
    internal static partial void EncoderDraw(nint encoder, uint vertexStart, uint vertexCount, uint instanceStart, uint instanceCount);

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_draw_indexed")]
    internal static partial void EncoderDrawIndexed(nint encoder, uint indexStart, uint indexCount, int baseVertex, uint instanceStart, uint instanceCount);

    [LibraryImport(LibName, EntryPoint = "goldy_encoder_bind_resources")]
    internal static partial void EncoderBindResources(nint encoder, nint buffers, uint bufferCount);

    // ========================================================================
    // Compute
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_compute_pipeline_create")]
    internal static partial nint ComputePipelineCreate(nint device, nint computeShader);

    [LibraryImport(LibName, EntryPoint = "goldy_compute_pipeline_destroy")]
    internal static partial void ComputePipelineDestroy(nint pipeline);

    [LibraryImport(LibName, EntryPoint = "goldy_compute_encoder_create")]
    internal static partial nint ComputeEncoderCreate();

    [LibraryImport(LibName, EntryPoint = "goldy_compute_encoder_destroy")]
    internal static partial void ComputeEncoderDestroy(nint encoder);

    [LibraryImport(LibName, EntryPoint = "goldy_compute_encoder_set_pipeline")]
    internal static partial void ComputeEncoderSetPipeline(nint encoder, nint pipeline);

    [LibraryImport(LibName, EntryPoint = "goldy_compute_encoder_bind_resources")]
    internal static partial void ComputeEncoderBindResources(nint encoder, nint buffers, uint bufferCount);

    [LibraryImport(LibName, EntryPoint = "goldy_compute_encoder_dispatch")]
    internal static partial void ComputeEncoderDispatch(nint encoder, uint workgroupsX, uint workgroupsY, uint workgroupsZ);

    [LibraryImport(LibName, EntryPoint = "goldy_compute_encoder_execute")]
    internal static partial GoldyResult ComputeEncoderExecute(nint encoder, nint device);

    // ========================================================================
    // Texture
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_texture_create")]
    internal static partial nint TextureCreate(nint device, uint width, uint height, TextureFormat format, SpatialAccess access, TextureFlags flags);

    [LibraryImport(LibName, EntryPoint = "goldy_texture_destroy")]
    internal static partial void TextureDestroy(nint texture);

    [LibraryImport(LibName, EntryPoint = "goldy_texture_width")]
    internal static partial uint TextureWidth(nint texture);

    [LibraryImport(LibName, EntryPoint = "goldy_texture_height")]
    internal static partial uint TextureHeight(nint texture);

    [LibraryImport(LibName, EntryPoint = "goldy_texture_format")]
    internal static partial TextureFormat TextureFormat(nint texture);

    // ========================================================================
    // Sampler
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_sampler_create")]
    internal static partial nint SamplerCreate(nint device, in SamplerDescNative desc);

    [LibraryImport(LibName, EntryPoint = "goldy_sampler_create_default")]
    internal static partial nint SamplerCreateDefault(nint device);

    [LibraryImport(LibName, EntryPoint = "goldy_sampler_destroy")]
    internal static partial void SamplerDestroy(nint sampler);

    // ========================================================================
    // Surface
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_surface_destroy")]
    internal static partial void SurfaceDestroy(nint surface);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_width")]
    internal static partial uint SurfaceWidth(nint surface);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_height")]
    internal static partial uint SurfaceHeight(nint surface);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_format")]
    internal static partial TextureFormat SurfaceFormat(nint surface);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_resize")]
    internal static partial GoldyResult SurfaceResize(nint surface, uint width, uint height);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_acquire")]
    internal static partial nint SurfaceAcquire(nint surface);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_present")]
    internal static partial GoldyResult SurfacePresent(nint surface, nint frame);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_frame_render")]
    internal static partial GoldyResult SurfaceFrameRender(nint frame, nint encoder);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_frame_width")]
    internal static partial uint SurfaceFrameWidth(nint frame);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_frame_height")]
    internal static partial uint SurfaceFrameHeight(nint frame);

    // ========================================================================
    // Surface - Platform-specific creation
    // ========================================================================

    /// <summary>
    /// Create a surface from a Win32 HWND.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "goldy_surface_create_win32")]
    internal static partial nint SurfaceCreateWin32(nint device, nint hwnd);
}

/// <summary>
/// Result codes from native FFI operations.
/// </summary>
internal enum GoldyResult
{
    Ok = 0,
    InvalidArgument = 1,
    NullPointer = 2,
    GpuError = 3,
    ShaderError = 4,
    ResourceError = 5,
    InternalError = 6,
}

