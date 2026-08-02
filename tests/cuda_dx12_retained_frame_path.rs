//! Structural proof for the steady-state CUDA -> DX12 retained frame path.

#![cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]

use goldy::types::BackendType;
use goldy::{
    BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, PresentMode, PrimitiveTopology, RenderPipeline,
    RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange,
    TargetLoad, TextureFormat, Vertex2D,
};
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    Win32WindowHandle, WindowHandle, WindowsDisplayHandle,
};
use std::num::NonZeroIsize;
use std::sync::Arc;
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, CW_USEDEFAULT, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

const TRIANGLE_SHADER: &str = r#"
struct VertexInput { float2 position : POSITION; float4 color : COLOR; };
struct VertexOutput { float4 position : SV_Position; float4 color : COLOR; };
[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}
[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target { return input.color; }
"#;

struct TestWindow(HWND);

impl TestWindow {
    fn create() -> windows::core::Result<Self> {
        unsafe {
            CreateWindowExW(
                Default::default(),
                w!("STATIC"),
                w!("goldy retained frame path"),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                128,
                128,
                None,
                None,
                None,
                None,
            )
            .map(Self)
        }
    }
}

impl Drop for TestWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.0);
        }
    }
}

impl HasWindowHandle for TestWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let handle = Win32WindowHandle::new(NonZeroIsize::new(self.0 .0 as isize).ok_or(HandleError::Unavailable)?);
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
    }
}

impl HasDisplayHandle for TestWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(unsafe { DisplayHandle::borrow_raw(RawDisplayHandle::Windows(WindowsDisplayHandle::new())) })
    }
}

fn try_cuda_instance() -> Option<Instance> {
    unsafe { std::env::set_var("GOLDY_BACKEND", "cuda") };
    Instance::new().ok().filter(|instance| {
        instance
            .enumerate_adapters()
            .into_iter()
            .any(|adapter| adapter.get_info().backend == BackendType::Cuda)
    })
}

#[test]
fn cuda_raster_direct_retained_steady_state() {
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("CUDA adapter");
    let device = Arc::new(
        adapter
            .request_device(&DeviceDescriptor::default())
            .expect("CUDA/DX12 companion"),
    );
    let ctx = device.create_context().expect("context");
    let Ok(window) = TestWindow::create() else {
        eprintln!("skip: no interactive Win32 session");
        return;
    };
    let surface = SurfaceExchange::new_with_depth(
        &ctx,
        &window,
        2,
        SurfaceConfig {
            present_mode: PresentMode::Immediate,
            depth_format: None,
        },
    )
    .expect("surface");
    let shader = ShaderModule::from_slang(&device, TRIANGLE_SHADER).expect("shader");
    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba32Float,
            topology: PrimitiveTopology::TriangleList,
            depth_stencil: None,
        },
    )
    .expect("pipeline");
    let mut pool = RetainedPool::new(Arc::clone(&device));
    let vertices = [
        Vertex2D::new(0.0, 0.5, Color::RED),
        Vertex2D::new(-0.5, -0.5, Color::RED),
        Vertex2D::new(0.5, -0.5, Color::RED),
    ];
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vertex buffer");

    let mut scheme = Scheme::new(&ctx);
    let (lease, present) = surface.bind_destination(&mut scheme).expect("bind destination");
    let (width, height) = surface.size();
    let target = scheme
        .lease_render_target(width, height, TextureFormat::Rgba32Float, None)
        .expect("render target");
    {
        let mut pass = scheme.render_pass("retained triangle", &target, TargetLoad::Clear(Color::BLACK));
        pass.with_parcel(&vertex_buffer, goldy::NodeAccess::Read);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..3, 0..1);
        pass.finish();
    }
    scheme.copy_to_present(&target, &lease);

    for frame_i in 0..9 {
        let mut submission = scheme.submit().expect("submit frame");
        present
            .claim(&mut submission)
            .expect("claim present")
            .consume()
            .unwrap_or_else(|error| panic!("frame {frame_i}: present failed: {error:#}"));
    }

    let stats = device.cuda_path_stats_for_test().expect("CUDA stats must be available");
    assert_eq!(stats.present_handoffs, 0, "direct raster present needs no CUDA handoff");
    assert_eq!(stats.worker_flushes, 0, "direct raster present needs no worker flush");
    assert_eq!(
        stats.present_completion_events, 0,
        "direct raster present must not allocate CUDA present-completion events"
    );
    assert_eq!(
        stats.completion_events, 0,
        "direct raster frames must allocate zero CUDA completion events"
    );
    assert_eq!(
        stats.rematerialize_fallbacks, 0,
        "retained replay must not rematerialize"
    );
    assert_eq!(stats.dtoh_calls, 0, "shared vertex buffers must not DtoH");
    assert!(
        stats.shared_vb_binds >= 1,
        "shared vertex buffer should bind at least once during warmup, got {}",
        stats.shared_vb_binds
    );
    assert!(
        stats.raster_list_records <= 3,
        "warmup should record at most one retained list per raster slot, got {}",
        stats.raster_list_records
    );
    assert!(
        stats.present_list_records <= 6,
        "aligned backbuffer slots should retain present lists after COMMON→PRESENT warmup, got {}",
        stats.present_list_records
    );
}

const FILL_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(DirectSpatial<float4> output, ThreadId tid) {
    output[tid.xy] = float4(0.1, 0.2, 0.3, 1.0);
}
"#;

#[test]
fn cuda_compute_to_present_retained_steady_state() {
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("CUDA adapter");
    let device = Arc::new(
        adapter
            .request_device(&DeviceDescriptor::default())
            .expect("CUDA/DX12 companion"),
    );
    let ctx = device.create_context().expect("context");
    let Ok(window) = TestWindow::create() else {
        eprintln!("skip: no interactive Win32 session");
        return;
    };
    let surface = SurfaceExchange::new_with_depth(
        &ctx,
        &window,
        2,
        SurfaceConfig {
            present_mode: PresentMode::Immediate,
            depth_format: None,
        },
    )
    .expect("surface");
    let shader = ShaderModule::from_slang(&device, FILL_SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let mut scheme = Scheme::new(&ctx);
    let (lease, present) = surface.bind_destination(&mut scheme).expect("bind destination");
    let (width, height) = surface.size();
    scheme
        .node("fill", &pipeline)
        .with_present(&lease)
        .dispatch(width.div_ceil(8), height.div_ceil(8), 1);

    for frame_i in 0..8 {
        let mut submission = scheme.submit().expect("submit frame");
        present
            .claim(&mut submission)
            .expect("claim present")
            .consume()
            .unwrap_or_else(|error| panic!("frame {frame_i}: present failed: {error:#}"));
    }

    let stats = device.cuda_path_stats_for_test().expect("CUDA stats must be available");
    // Present-bound launches are rewritten onto CUDA-owned staging and captured; the
    // imported scratch export stays in the fixed GraphWithTail (CopyTexture + fence).
    assert_eq!(stats.rematerialize_fallbacks, 0, "retained replay must not rematerialize");
    assert_eq!(stats.present_handoffs, 0, "scratch present must use submit-tail signal");
    assert_eq!(stats.worker_flushes, 0, "scratch present must not flush the worker");
    assert!(
        stats.captures >= 1,
        "compute-to-present must capture the staging launch core, got {}",
        stats.captures
    );
    assert!(
        stats.launches >= 1,
        "compute-to-present must relaunch the retained graph, got {}",
        stats.launches
    );
    assert_eq!(
        stats.fallbacks, 0,
        "compute-to-present must not fall back to Ops replay after staging rewrite"
    );
    assert_eq!(
        stats.shared_vb_binds, 0,
        "compute-to-present has no vertex binds"
    );
}
