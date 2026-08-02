//! CUDA + DX12 first-slice raster tests (Windows).
//!
//! Covers offscreen TriangleList draws into shared `Rgba32Float` render targets,
//! CopyRenderTarget → CUDA texture readback, and render → present.

#![cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]

use goldy::types::BackendType;
use goldy::{
    BufferKind, Color, DeviceDescriptor, Instance, MemoryExchange, PresentMode, PrimitiveTopology,
    RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule,
    SurfaceConfig, SurfaceExchange, TargetLoad, TextureFlags, TextureFormat, TextureKind, Vertex2D,
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
                w!("goldy cuda raster test"),
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

#[test]
fn cuda_raster_rejects_depth_and_wrong_format() {
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
    let shader = ShaderModule::from_slang(&device, TRIANGLE_SHADER).expect("shader");

    let err = match RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            topology: PrimitiveTopology::TriangleList,
            depth_stencil: None,
        },
    ) {
        Ok(_) => panic!("Rgba8Unorm must be rejected in first raster slice"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Rgba32Float") || msg.contains("only"),
        "unexpected error: {msg}"
    );
}

#[test]
fn cuda_raster_triangle_readback() {
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
    assert_eq!(device.backend_type(), BackendType::Cuda);
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
    let readback = pool
        .acquire_texture(
            64,
            64,
            TextureFormat::Rgba32Float,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
            None,
        )
        .expect("readback texture");

    let mut scheme = Scheme::new(&ctx);
    let rt = scheme
        .lease_render_target(64, 64, TextureFormat::Rgba32Float, None)
        .expect("render target");
    {
        let mut pass = scheme.render_pass("tri", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_parcel(&vertex_buffer, goldy::NodeAccess::Read);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..3, 0..1);
        pass.finish();
    }
    scheme.copy_to_texture(&rt, &readback).expect("copy_to_texture");
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &readback)
        .expect("withdraw");
    let mut submission = scheme.submit().expect("submit");
    let pixels = grant
        .claim(&mut submission)
        .expect("claim")
        .consume()
        .expect("consume")
        .to_vec();

    assert_eq!(pixels.len(), 64 * 64 * 16);
    // Sample near the centroid of the NDC triangle (slightly above center).
    let x = 32usize;
    let y = 28usize;
    let offset = (y * 64 + x) * 16;
    let r = f32::from_le_bytes(pixels[offset..offset + 4].try_into().unwrap());
    let g = f32::from_le_bytes(pixels[offset + 4..offset + 8].try_into().unwrap());
    let b = f32::from_le_bytes(pixels[offset + 8..offset + 12].try_into().unwrap());
    assert!(
        r > 0.5 && g < 0.25 && b < 0.25,
        "expected red triangle pixel at ({x},{y}), got ({r},{g},{b})"
    );
}

#[test]
fn cuda_raster_to_present_multi_frame() {
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

    for frame_i in 0..3u32 {
        let mut scheme = Scheme::new(&ctx);
        let (lease, present_tx) = surface.bind_destination(&mut scheme).expect("bind");
        let (w, h) = surface.size();
        let rt = scheme
            .lease_render_target(w, h, TextureFormat::Rgba32Float, None)
            .expect("render target");
        {
            let mut pass = scheme.render_pass("tri", &rt, TargetLoad::Clear(Color::BLACK));
            pass.with_parcel(&vertex_buffer, goldy::NodeAccess::Read);
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..3, 0..1);
            pass.finish();
        }
        scheme.copy_to_present(&rt, &lease);
        let mut submission = scheme.submit().expect("submit");
        let epoch = goldy::test_support::submission_epoch(&submission);
        assert!(epoch > 0, "frame {frame_i}: submit must advance timeline");

        let claim = present_tx.claim(&mut submission).expect("claim");
        claim.consume().expect("present");

        goldy::test_support::wait_until(&ctx, epoch).unwrap_or_else(|e| {
            panic!("frame {frame_i}: wait_until({epoch}) failed: {e:#}")
        });
    }
}
