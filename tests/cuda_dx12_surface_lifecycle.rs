//! CUDA + DX12 surface lifecycle hardening (Windows).
//!
//! Covers resize during present, minimize/restore, destroy-with-in-flight work,
//! same-size early-out, and a small resize latency microbench. Offscreen RT
//! recreate stress lives here too so interop teardown stays covered without a
//! window when CreateWindowExW is unavailable.

#![cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]

use goldy::types::BackendType;
use goldy::{
    BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, MemoryExchange, PresentMode,
    PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool,
    Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, TextureFlags, TextureFormat,
    TextureKind, Vertex2D,
};
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    Win32WindowHandle, WindowHandle, WindowsDisplayHandle,
};
use std::num::NonZeroIsize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, CW_USEDEFAULT, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

fn try_cuda_instance() -> Option<Instance> {
    // SAFETY: test process; GOLDY_BACKEND is read during Instance::new.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cuda") };
    Instance::new().ok().filter(|i| {
        i.enumerate_adapters()
            .into_iter()
            .any(|a| a.get_info().backend == BackendType::Cuda)
    })
}

struct TestWindow {
    hwnd: HWND,
}

impl TestWindow {
    fn create(width: i32, height: i32) -> windows::core::Result<Self> {
        let hwnd = unsafe {
            CreateWindowExW(
                Default::default(),
                w!("STATIC"),
                w!("goldy cuda surface lifecycle"),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                width,
                height,
                None,
                None,
                None,
                None,
            )
        }?;
        Ok(Self { hwnd })
    }
}

impl Drop for TestWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.hwnd);
        }
    }
}

impl HasWindowHandle for TestWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let mut handle = Win32WindowHandle::new(
            NonZeroIsize::new(self.hwnd.0 as isize).ok_or(HandleError::Unavailable)?,
        );
        handle.hinstance = None;
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
    }
}

impl HasDisplayHandle for TestWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(unsafe {
            DisplayHandle::borrow_raw(RawDisplayHandle::Windows(WindowsDisplayHandle::new()))
        })
    }
}

const FILL_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(DirectSpatial<float4> output, ThreadId tid) {
    output[tid.xy] = float4(0.1, 0.2, 0.3, 1.0);
}
"#;

const TRIANGLE_SHADER: &str = r#"
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
"#;

fn red_triangle_vertices() -> [Vertex2D; 3] {
    [
        Vertex2D::new(0.0, 0.5, Color::RED),
        Vertex2D::new(-0.5, -0.5, Color::RED),
        Vertex2D::new(0.5, -0.5, Color::RED),
    ]
}

fn present_one_fill(surface: &SurfaceExchange, pipeline: &ComputePipeline, ctx: &goldy::Context) {
    let mut scheme = Scheme::new(ctx);
    let (lease, present_tx) = surface.bind_destination(&mut scheme).expect("bind");
    let (w, h) = surface.size();
    scheme
        .node("fill", pipeline)
        .with_present(&lease)
        .dispatch(w.div_ceil(8), h.div_ceil(8), 1);
    let mut submission = scheme.submit().expect("submit");
    let compute_tv = goldy::test_support::submission_epoch(&submission);
    let claim = present_tx.claim(&mut submission).expect("claim");
    claim.consume().expect("present");
    goldy::test_support::wait_until(ctx, compute_tv).expect("wait_until");
}

#[test]
fn cuda_surface_resize_during_present_loop() {
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
            .expect("DX12 companion must attach"),
    );
    let ctx = device.create_context().expect("context");
    let Ok(window) = TestWindow::create(128, 128) else {
        eprintln!("skip: CreateWindowExW failed (headless / no interactive session)");
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
    .expect("surface create");
    let shader = ShaderModule::from_slang(&device, FILL_SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    present_one_fill(&surface, &pipeline, &ctx);
    surface.resize(256, 192).expect("resize 256x192");
    assert_eq!(surface.size(), (256, 192));
    present_one_fill(&surface, &pipeline, &ctx);
    surface.resize(160, 160).expect("resize 160x160");
    assert_eq!(surface.size(), (160, 160));
    present_one_fill(&surface, &pipeline, &ctx);
}

#[test]
fn cuda_surface_minimize_restore() {
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
            .expect("DX12 companion must attach"),
    );
    let ctx = device.create_context().expect("context");
    let Ok(window) = TestWindow::create(128, 128) else {
        eprintln!("skip: CreateWindowExW failed");
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
    .expect("surface create");
    let shader = ShaderModule::from_slang(&device, FILL_SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    present_one_fill(&surface, &pipeline, &ctx);
    let (w0, h0) = surface.size();
    surface.resize(0, 0).expect("minimize no-op");
    assert_eq!(surface.size(), (w0, h0));
    // Skip frames while minimized (app pattern).
    surface.resize(192, 144).expect("restore");
    assert_eq!(surface.size(), (192, 144));
    present_one_fill(&surface, &pipeline, &ctx);
}

#[test]
fn cuda_surface_destroy_with_inflight_submit() {
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
            .expect("DX12 companion must attach"),
    );
    let ctx = device.create_context().expect("context");
    let Ok(window) = TestWindow::create(96, 96) else {
        eprintln!("skip: CreateWindowExW failed");
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
    .expect("surface create");
    let shader = ShaderModule::from_slang(&device, FILL_SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let mut scheme = Scheme::new(&ctx);
    let (lease, present_tx) = surface.bind_destination(&mut scheme).expect("bind");
    let (w, h) = surface.size();
    scheme
        .node("fill", &pipeline)
        .with_present(&lease)
        .dispatch(w.div_ceil(8), h.div_ceil(8), 1);
    let mut submission = scheme.submit().expect("submit");
    let _claim = present_tx.claim(&mut submission).expect("claim");
    // Drop surface (and claim) without consuming present — teardown must idle safely.
    drop(surface);
    drop(window);
    // Context/device drop after idle must also succeed.
    drop(submission);
    drop(ctx);
    drop(device);
}

#[test]
fn cuda_surface_same_size_resize_is_cheap() {
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
            .expect("DX12 companion must attach"),
    );
    let ctx = device.create_context().expect("context");
    let Ok(window) = TestWindow::create(128, 128) else {
        eprintln!("skip: CreateWindowExW failed");
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
    .expect("surface create");
    let shader = ShaderModule::from_slang(&device, FILL_SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
    present_one_fill(&surface, &pipeline, &ctx);

    // Deferred resize: recording a new extent is near-free; apply happens on present.
    let record = Instant::now();
    surface.resize(256, 256).expect("record structural");
    let record_dt = record.elapsed();
    assert_eq!(surface.size(), (256, 256));

    let apply = Instant::now();
    present_one_fill(&surface, &pipeline, &ctx);
    let apply_dt = apply.elapsed();

    let mut same_total = Duration::ZERO;
    for _ in 0..32 {
        let t0 = Instant::now();
        surface.resize(256, 256).expect("same-size");
        same_total += t0.elapsed();
    }
    let same_avg = same_total / 32;
    eprintln!(
        "cuda resize bench: record {:?} apply+present {:?} same-size avg {:?}",
        record_dt, apply_dt, same_avg
    );
    assert!(
        record_dt < Duration::from_millis(1),
        "deferred resize record too slow: {record_dt:?}"
    );
    assert!(
        same_avg < Duration::from_millis(1),
        "same-size resize too slow: avg {same_avg:?}"
    );
}

#[test]
fn cuda_surface_resize_latency_microbench() {
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
            .expect("DX12 companion must attach"),
    );
    let ctx = device.create_context().expect("context");
    let Ok(window) = TestWindow::create(128, 128) else {
        eprintln!("skip: CreateWindowExW failed");
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
    .expect("surface create");
    let shader = ShaderModule::from_slang(&device, FILL_SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let sizes = [(160u32, 120u32), (192, 144), (224, 168), (256, 192), (128, 128)];
    let mut times = Vec::new();
    for &(w, h) in &sizes {
        present_one_fill(&surface, &pipeline, &ctx);
        let t0 = Instant::now();
        surface.resize(w, h).expect("resize");
        let record_dt = t0.elapsed();
        let t1 = Instant::now();
        present_one_fill(&surface, &pipeline, &ctx);
        let apply_dt = t1.elapsed();
        times.push((w, h, record_dt, apply_dt));
    }
    for (w, h, record_dt, apply_dt) in &times {
        eprintln!("cuda resize {w}x{h}: record {record_dt:?} apply+present {apply_dt:?}");
    }

    // Window-size storm: many resize() calls, one present applies the last extent.
    present_one_fill(&surface, &pipeline, &ctx);
    let storm = Instant::now();
    for i in 0..30u32 {
        let w = 160 + i * 2;
        let h = 120 + i;
        surface.resize(w, h).expect("storm");
    }
    let storm_record = storm.elapsed();
    assert_eq!(surface.size(), (160 + 29 * 2, 120 + 29));
    let storm_apply = Instant::now();
    present_one_fill(&surface, &pipeline, &ctx);
    let storm_apply_dt = storm_apply.elapsed();
    eprintln!(
        "cuda resize storm: 30 records in {:?}, apply+present {:?}",
        storm_record, storm_apply_dt
    );
    assert!(
        storm_record < Duration::from_millis(5),
        "resize storm record too slow: {storm_record:?}"
    );

    let oscillate = [(192u32, 144u32), (256, 192), (192, 144), (256, 192), (192, 144)];
    let mut osc_times = Vec::new();
    for &(w, h) in &oscillate {
        surface.resize(w, h).expect("oscillate");
        let t0 = Instant::now();
        present_one_fill(&surface, &pipeline, &ctx);
        osc_times.push(t0.elapsed());
    }
    eprintln!("cuda oscillate apply+present: {osc_times:?}");

    let max_apply = times
        .iter()
        .map(|(_, _, _, d)| *d)
        .chain(osc_times.iter().copied())
        .chain(std::iter::once(storm_apply_dt))
        .max()
        .unwrap();
    assert!(
        max_apply < Duration::from_secs(2),
        "apply+present exceeded soft budget: {max_apply:?}"
    );
}

#[test]
fn cuda_offscreen_rt_recreate_stress() {
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
            .expect("DX12 companion must attach"),
    );
    let ctx = device.create_context().expect("context");
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
    .expect("graphics pipeline");
    let mut pool = RetainedPool::new(Arc::clone(&device));
    let vertices = red_triangle_vertices();
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vertex buffer");

    let sizes = [32u32, 48, 64, 96, 64, 32];
    let mut total = Duration::ZERO;
    for &dim in &sizes {
        let readback = pool
            .acquire_texture(
                dim,
                dim,
                TextureFormat::Rgba32Float,
                TextureKind::Direct,
                TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
                None,
            )
            .expect("readback");
        let t0 = Instant::now();
        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(dim, dim, TextureFormat::Rgba32Float, None)
            .expect("rt");
        {
            let mut pass = scheme.render_pass("tri", &rt, TargetLoad::Clear(Color::BLACK));
            pass.with_parcel(&vertex_buffer, goldy::NodeAccess::Read);
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..3, 0..1);
            pass.finish();
        }
        scheme.copy_to_texture(&rt, &readback).expect("copy");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &readback)
            .expect("withdraw");
        let mut submission = scheme.submit().expect("submit");
        let _pixels = grant
            .claim(&mut submission)
            .expect("claim")
            .consume()
            .expect("consume");
        total += t0.elapsed();
    }
    eprintln!(
        "cuda offscreen RT recreate stress: {} sizes in {:?}",
        sizes.len(),
        total
    );
}

#[test]
fn cuda_surface_rejects_depth_config() {
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
            .expect("DX12 companion must attach"),
    );
    let ctx = device.create_context().expect("context");
    let Ok(window) = TestWindow::create(64, 64) else {
        eprintln!("skip: CreateWindowExW failed");
        return;
    };
    let err = match SurfaceExchange::new_with_depth(
        &ctx,
        &window,
        2,
        SurfaceConfig {
            present_mode: PresentMode::Immediate,
            depth_format: Some(goldy::types::DepthFormat::Depth32Float),
        },
    ) {
        Ok(_) => panic!("depth on CUDA surfaces must fail in first slice"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("depth") || msg.contains("not supported"),
        "unexpected error: {msg}"
    );
}
