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
    // Compute
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_compute_pipeline_create")]
    internal static partial nint ComputePipelineCreate(nint device, nint computeShader);

    [LibraryImport(LibName, EntryPoint = "goldy_compute_pipeline_destroy")]
    internal static partial void ComputePipelineDestroy(nint pipeline);

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

    [LibraryImport(LibName, EntryPoint = "goldy_surface_frame_width")]
    internal static partial uint SurfaceFrameWidth(nint frame);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_frame_height")]
    internal static partial uint SurfaceFrameHeight(nint frame);

    // ========================================================================
    // Context
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_context_create")]
    internal static partial nint ContextCreate(nint device);

    [LibraryImport(LibName, EntryPoint = "goldy_context_destroy")]
    internal static partial void ContextDestroy(nint ctx);

    // ========================================================================
    // Scheme
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_create")]
    internal static partial nint SchemeCreate(nint ctx);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_destroy")]
    internal static partial void SchemeDestroy(nint scheme);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_compute_node_begin", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial GoldyResult SchemeComputeNodeBegin(nint scheme, string label, nint pipeline);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_compute_node_with_parcel")]
    internal static partial GoldyResult SchemeComputeNodeWithParcel(
        nint scheme, nint parcel, NodeAccess nodeAccess);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_compute_node_with_texture")]
    internal static partial GoldyResult SchemeComputeNodeWithTexture(
        nint scheme, nint texture, NodeAccess nodeAccess);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_compute_node_with_param")]
    internal static partial GoldyResult SchemeComputeNodeWithParam(nint scheme, uint value);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_compute_node_dispatch")]
    internal static partial GoldyResult SchemeComputeNodeDispatch(
        nint scheme, uint workgroupsX, uint workgroupsY, uint workgroupsZ);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_submit")]
    internal static partial GoldyResult SchemeSubmit(nint scheme, out nint outSubmission);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_submission_destroy")]
    internal static partial void SchemeSubmissionDestroy(nint submission);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_submission_timeline_value")]
    internal static partial ulong SchemeSubmissionTimelineValue(nint submission);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_submission_wait")]
    internal static partial GoldyResult SchemeSubmissionWait(nint ctx, nint submission);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_grant_read")]
    internal static partial nint SchemeGrantRead(nint scheme, nint parcel);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_grant_read_texture")]
    internal static partial nint SchemeGrantReadTexture(nint scheme, nint texture);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_lease_render_target")]
    internal static partial nint SchemeLeaseRenderTarget(
        nint scheme, uint width, uint height, TextureFormat format, [MarshalAs(UnmanagedType.U1)] bool hasDepth, DepthFormat depthFormat);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_target_lease_destroy")]
    internal static partial void SchemeRenderTargetLeaseDestroy(nint lease);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_copy_to_texture")]
    internal static partial GoldyResult SchemeCopyToTexture(nint scheme, nint srcLease, nint dstTexture);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_copy_to_present")]
    internal static partial GoldyResult SchemeCopyToPresent(nint scheme, nint srcLease, nint dstLease);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_grant_present")]
    internal static partial nint SchemeGrantPresent(nint scheme, nint presentLease);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_begin", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial GoldyResult SchemeRenderPassBegin(nint scheme, string label, nint lease);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_with_parcel")]
    internal static partial GoldyResult SchemeRenderPassWithParcel(nint scheme, nint parcel, NodeAccess access);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_with_field")]
    internal static partial GoldyResult SchemeRenderPassWithField(
        nint scheme, nint buffer, uint unit, NodeAccess access);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_with_buffer_unit")]
    internal static partial GoldyResult SchemeRenderPassWithBufferUnit(
        nint scheme, nint buffer, uint unit, NodeAccess access);


    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_clear")]
    internal static partial GoldyResult SchemeRenderPassClear(nint scheme, Color color);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_set_pipeline")]
    internal static partial GoldyResult SchemeRenderPassSetPipeline(nint scheme, nint pipeline);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_set_vertex_buffer_parcel")]
    internal static partial GoldyResult SchemeRenderPassSetVertexBufferParcel(nint scheme, uint slot, nint parcel);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_draw")]
    internal static partial GoldyResult SchemeRenderPassDraw(
        nint scheme, uint firstVertex, uint vertexCount, uint firstInstance, uint instanceCount);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_draw_fullscreen")]
    internal static partial GoldyResult SchemeRenderPassDrawFullscreen(nint scheme);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_render_pass_finish")]
    internal static partial GoldyResult SchemeRenderPassFinish(nint scheme);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_compute_node_with_field")]
    internal static partial GoldyResult SchemeComputeNodeWithField(
        nint scheme, nint buffer, uint unit, NodeAccess nodeAccess);

    [LibraryImport(LibName, EntryPoint = "goldy_scheme_compute_node_with_buffer_unit")]
    internal static partial GoldyResult SchemeComputeNodeWithBufferUnit(
        nint scheme, nint buffer, uint unit, NodeAccess nodeAccess);

    [LibraryImport(LibName, EntryPoint = "goldy_present_grant_consume")]
    internal static partial GoldyResult PresentGrantConsume(nint grant, nint submission);

    [LibraryImport(LibName, EntryPoint = "goldy_present_grant_destroy")]
    internal static partial void PresentGrantDestroy(nint grant);

    [LibraryImport(LibName, EntryPoint = "goldy_present_lease_destroy")]
    internal static partial void PresentLeaseDestroy(nint lease);

    [LibraryImport(LibName, EntryPoint = "goldy_read_grant_destroy")]
    internal static partial void ReadGrantDestroy(nint grant);

    [LibraryImport(LibName, EntryPoint = "goldy_read_grant_byte_size")]
    internal static partial ulong ReadGrantByteSize(nint grant);

    [LibraryImport(LibName, EntryPoint = "goldy_read_grant_consume")]
    internal static partial GoldyResult ReadGrantConsume(
        nint grant, nint submission, nint output, nuint outputSize);

    // ========================================================================
    // RetainedPool / Parcel
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_retained_pool_create")]
    internal static partial nint RetainedPoolCreate(nint device);

    [LibraryImport(LibName, EntryPoint = "goldy_retained_pool_destroy")]
    internal static partial void RetainedPoolDestroy(nint pool);

    [LibraryImport(LibName, EntryPoint = "goldy_retained_pool_acquire_buffer")]
    internal static partial nint RetainedPoolAcquireBuffer(
        nint pool, ulong size, BufferKind access, uint elementStride, nint data, nuint dataSize);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_destroy")]
    internal static partial void BufferDestroy(nint buffer);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_byte_size")]
    internal static partial ulong BufferByteSize(nint buffer);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_field")]
    internal static partial nint BufferField(nint buffer, uint unit);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_unit_count")]
    internal static partial uint BufferUnitCount(nint buffer);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_unit_byte_size")]
    internal static partial ulong BufferUnitByteSize(nint buffer, uint unit);

    [LibraryImport(LibName, EntryPoint = "goldy_buffer_unit_read_to_cpu")]
    internal static partial GoldyResult BufferUnitReadToCpu(
        nint buffer, uint unit, nint device, nint output, nuint outputSize);

    [LibraryImport(LibName, EntryPoint = "goldy_record_builder_create")]
    internal static partial nint RecordBuilderCreate();

    [LibraryImport(LibName, EntryPoint = "goldy_record_builder_destroy")]
    internal static partial void RecordBuilderDestroy(nint builder);

    [LibraryImport(LibName, EntryPoint = "goldy_record_builder_emplace", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial uint RecordBuilderEmplace(
        nint builder,
        string? name,
        nint data,
        nuint dataSize,
        ulong elementCount,
        uint elementStride);

    [LibraryImport(LibName, EntryPoint = "goldy_record_builder_reserve", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial uint RecordBuilderReserve(
        nint builder, string? name, ulong elementCount, uint elementStride);

    [LibraryImport(LibName, EntryPoint = "goldy_record_builder_build")]
    internal static partial nint RecordBuilderBuild(nint builder, nint pool);

    [LibraryImport(LibName, EntryPoint = "goldy_retained_pool_acquire_texture")]
    internal static partial nint RetainedPoolAcquireTexture(
        nint pool,
        uint width,
        uint height,
        TextureFormat format,
        TextureKind access,
        TextureFlags flags,
        nint data,
        nuint dataSize);

    [LibraryImport(LibName, EntryPoint = "goldy_texture_destroy")]
    internal static partial void TextureDestroy(nint texture);

    [LibraryImport(LibName, EntryPoint = "goldy_texture_byte_size")]
    internal static partial ulong TextureByteSize(nint texture);

    [LibraryImport(LibName, EntryPoint = "goldy_parcel_destroy")]
    internal static partial void ParcelDestroy(nint parcel);

    [LibraryImport(LibName, EntryPoint = "goldy_parcel_byte_size")]
    internal static partial ulong ParcelByteSize(nint parcel);

    // ========================================================================
    // Surface - Platform-specific creation
    // ========================================================================

    /// <summary>
    /// Create a surface from a Win32 HWND.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "goldy_surface_create_win32")]
    internal static partial nint SurfaceCreateWin32(nint device, nint hwnd);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_create_appkit")]
    internal static partial nint SurfaceCreateAppKit(nint device, nint nsView);

    [LibraryImport(LibName, EntryPoint = "goldy_surface_create_wayland")]
    internal static partial nint SurfaceCreateWayland(nint device, nint display, nint surface);

    // ========================================================================
    // SwapchainPool (present-on-scheme)
    // ========================================================================

    [LibraryImport(LibName, EntryPoint = "goldy_swapchain_pool_create_win32")]
    internal static partial nint SwapchainPoolCreateWin32(nint ctx, nint hwnd, uint depth);

    [LibraryImport(LibName, EntryPoint = "goldy_swapchain_pool_create_appkit")]
    internal static partial nint SwapchainPoolCreateAppKit(nint ctx, nint nsView, uint depth);

    [LibraryImport(LibName, EntryPoint = "goldy_swapchain_pool_create_wayland")]
    internal static partial nint SwapchainPoolCreateWayland(nint ctx, nint display, nint surface, uint depth);

    [LibraryImport(LibName, EntryPoint = "goldy_swapchain_pool_destroy")]
    internal static partial void SwapchainPoolDestroy(nint pool);

    [LibraryImport(LibName, EntryPoint = "goldy_swapchain_pool_width")]
    internal static partial uint SwapchainPoolWidth(nint pool);

    [LibraryImport(LibName, EntryPoint = "goldy_swapchain_pool_height")]
    internal static partial uint SwapchainPoolHeight(nint pool);

    [LibraryImport(LibName, EntryPoint = "goldy_swapchain_pool_format")]
    internal static partial TextureFormat SwapchainPoolFormat(nint pool);

    [LibraryImport(LibName, EntryPoint = "goldy_swapchain_pool_resize")]
    internal static partial GoldyResult SwapchainPoolResize(nint pool, uint width, uint height);

    [LibraryImport(LibName, EntryPoint = "goldy_swapchain_pool_lease")]
    internal static partial nint SwapchainPoolLease(nint pool);
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

