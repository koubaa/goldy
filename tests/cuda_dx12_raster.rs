//! CUDA + DX12 first-slice raster tests (Windows).
//!
//! Covers offscreen TriangleList draws into shared `Rgba32Float` render targets,
//! CopyRenderTarget → CUDA texture readback, and render → present.

#![cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]

use goldy::types::BackendType;
use goldy::{
    BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, MemoryExchange, NodeAccess, PresentMode,
    PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Sampler,
    SamplerDesc, Scheme, ShaderModule, ShaderResourceSlot, SurfaceConfig, SurfaceExchange, TargetLoad,
    TextureFlags, TextureFormat, TextureKind, Vertex2D,
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
    CreateWindowExW, DestroyWindow, CW_USEDEFAULT, WS_OVERLAPPEDWINDOW,
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
                // Hidden: unit tests must not pop visible windows/dialogs.
                WS_OVERLAPPEDWINDOW,
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
        let mut handle =
            Win32WindowHandle::new(NonZeroIsize::new(self.hwnd.0 as isize).ok_or(HandleError::Unavailable)?);
        handle.hinstance = None;
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
    }
}

impl HasDisplayHandle for TestWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(unsafe { DisplayHandle::borrow_raw(RawDisplayHandle::Windows(WindowsDisplayHandle::new())) })
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
        // DX12-direct raster presentation has no CUDA body to submit. Its raster
        // fence and the present timeline provide completion, so epoch may be zero.

        let claim = present_tx.claim(&mut submission).expect("claim");
        claim.consume().expect("present");

        if epoch > 0 {
            goldy::test_support::wait_until(&ctx, epoch)
                .unwrap_or_else(|e| panic!("frame {frame_i}: wait_until({epoch}) failed: {e:#}"));
        }
    }
    let stats = device.cuda_path_stats_for_test().expect("CUDA stats must be available");
    assert_eq!(stats.dtoh_calls, 0, "shared vertex data must not DtoH");
    assert!(
        stats.shared_vb_binds >= 1,
        "expected shared VB binds, got {}",
        stats.shared_vb_binds
    );
}

const FILL_VERTS_SHADER: &str = r#"
import goldy_exp;

// Pack 3×Vertex2D as 18 floats (Rust Vertex2D is 24 bytes / 6 floats, tightly packed).
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<float> verts, ThreadId id) {
    // v0
    verts[0] = 0.0;  verts[1] = 0.5;
    verts[2] = 1.0;  verts[3] = 0.0; verts[4] = 0.0; verts[5] = 1.0;
    // v1
    verts[6] = -0.5; verts[7] = -0.5;
    verts[8] = 1.0;  verts[9] = 0.0; verts[10] = 0.0; verts[11] = 1.0;
    // v2
    verts[12] = 0.5; verts[13] = -0.5;
    verts[14] = 1.0; verts[15] = 0.0; verts[16] = 0.0; verts[17] = 1.0;
}
"#;

#[test]
fn cuda_compute_generated_vertices_raster_no_dtoh() {
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

    let vs_fs = ShaderModule::from_slang(&device, TRIANGLE_SHADER).expect("graphics shader");
    let pipeline = RenderPipeline::new(
        &device,
        &vs_fs,
        &vs_fs,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba32Float,
            topology: PrimitiveTopology::TriangleList,
            depth_stencil: None,
        },
    )
    .expect("graphics pipeline");

    let cs = ShaderModule::from_slang(&device, FILL_VERTS_SHADER).expect("compute shader");
    let compute = ComputePipeline::new(&device, &cs).expect("compute pipeline");

    let mut pool = RetainedPool::new(Arc::clone(&device));
    // Empty shared VB — compute fills it each frame.
    let vertex_buffer = pool
        .acquire_buffer_sized::<f32>(18, BufferKind::Scattered, goldy::BufferFlags::empty())
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

    let before = device.cuda_path_stats_for_test().expect("stats");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("gen_verts", &compute)
        .with_parcel(&vertex_buffer, goldy::NodeAccess::Write)
        .dispatch(1, 1, 1);
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
    let x = 32usize;
    let y = 28usize;
    let offset = (y * 64 + x) * 16;
    let r = f32::from_le_bytes(pixels[offset..offset + 4].try_into().unwrap());
    let g = f32::from_le_bytes(pixels[offset + 4..offset + 8].try_into().unwrap());
    let b = f32::from_le_bytes(pixels[offset + 8..offset + 12].try_into().unwrap());
    assert!(
        r > 0.5 && g < 0.25 && b < 0.25,
        "expected red triangle from compute-generated verts at ({x},{y}), got ({r},{g},{b})"
    );

    let after = device.cuda_path_stats_for_test().expect("stats");
    // Texture withdraw uses DtoH for pixel readback; vertex path must not.
    assert!(
        after.shared_vb_binds > before.shared_vb_binds,
        "expected shared VB refresh/bind"
    );
}

#[test]
fn cuda_deposit_refreshes_shared_vb_each_frame() {
    // Spinning-cube style: CPU deposit into a Shared-primary VB every frame.
    // Retained upload ops replay without rematerializing; CUDA→DX12 handoff is the
    // companion fence on the imported buffer (no twin DtoD).
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
    let vertex_buffer = pool
        .acquire_buffer_sized::<Vertex2D>(3, BufferKind::Scattered, goldy::BufferFlags::empty())
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

    let mut upload = Scheme::new(&ctx);
    let deposit = MemoryExchange::new(&ctx)
        .bind_deposit_buffer(&mut upload, &vertex_buffer, vertex_buffer.byte_size())
        .expect("deposit");

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

    let frames: [(Color, fn(f32, f32, f32) -> bool); 2] = [
        (
            Color {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            |r, g, b| r > 0.5 && g < 0.25 && b < 0.25,
        ),
        (
            Color {
                r: 0.0,
                g: 1.0,
                b: 0.0,
                a: 1.0,
            },
            |r, g, b| g > 0.5 && r < 0.25 && b < 0.25,
        ),
    ];

    // Warm retention on upload + render so color frames exercise the resubmit path.
    for _ in 0..2 {
        let verts = [
            Vertex2D::new(0.0, 0.5, Color::BLACK),
            Vertex2D::new(-0.5, -0.5, Color::BLACK),
            Vertex2D::new(0.5, -0.5, Color::BLACK),
        ];
        deposit
            .write(&mut upload, 0, bytemuck::cast_slice(&verts))
            .expect("warmup deposit");
        upload.submit().expect("warmup upload");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &readback)
            .expect("warmup withdraw");
        let mut submission = scheme.submit().expect("warmup submit");
        let _ = grant.claim(&mut submission).expect("warmup claim").consume();
    }

    let mut prev_binds = device
        .cuda_path_stats_for_test()
        .expect("stats")
        .shared_vb_binds;

    for (frame_i, (color, expect_pixel)) in frames.iter().enumerate() {
        let verts = [
            Vertex2D::new(0.0, 0.5, *color),
            Vertex2D::new(-0.5, -0.5, *color),
            Vertex2D::new(0.5, -0.5, *color),
        ];
        deposit
            .write(&mut upload, 0, bytemuck::cast_slice(&verts))
            .expect("deposit write");
        upload.submit().expect("upload submit");

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

        let binds = device
            .cuda_path_stats_for_test()
            .expect("stats")
            .shared_vb_binds;
        assert!(
            binds > prev_binds,
            "frame {frame_i}: expected shared VB refresh after retain (binds {binds} <= prev {prev_binds})"
        );
        prev_binds = binds;

        let x = 32usize;
        let y = 28usize;
        let offset = (y * 64 + x) * 16;
        let r = f32::from_le_bytes(pixels[offset..offset + 4].try_into().unwrap());
        let g = f32::from_le_bytes(pixels[offset + 4..offset + 8].try_into().unwrap());
        let b = f32::from_le_bytes(pixels[offset + 8..offset + 12].try_into().unwrap());
        assert!(
            expect_pixel(r, g, b),
            "frame {frame_i}: unexpected pixel ({r},{g},{b}) — twin likely stale after retain"
        );
    }
}

#[test]
fn cuda_raster_goldy_vertex_color_2d() {
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

    let shader = ShaderModule::from_slang(&device, goldy::shaders::VERTEX_COLOR_2D).expect("shader");
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
        let mut pass = scheme.render_pass("goldy_tri", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_parcel(&vertex_buffer, NodeAccess::Read);
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

    let x = 32usize;
    let y = 28usize;
    let offset = (y * 64 + x) * 16;
    let r = f32::from_le_bytes(pixels[offset..offset + 4].try_into().unwrap());
    let g = f32::from_le_bytes(pixels[offset + 4..offset + 8].try_into().unwrap());
    let b = f32::from_le_bytes(pixels[offset + 8..offset + 12].try_into().unwrap());
    assert!(
        r > 0.5 && g < 0.25 && b < 0.25,
        "expected red goldy_ triangle pixel at ({x},{y}), got ({r},{g},{b})"
    );
}

const BINDLESS_TINT_SHADER: &str = r#"
import goldy_exp;

struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[goldy_vertex]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[goldy_fragment]
float4 fs_main(BufRO<float4> tint, VertexOutput input) : SV_Target {
    return input.color * tint[0];
}
"#;

const BINDLESS_TEXTURE_SHADER: &str = r#"
import goldy_exp;

struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[goldy_vertex]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    // Map NDC triangle into [0,1] uv roughly around center.
    output.uv = input.position * 0.5 + 0.5;
    return output;
}

[goldy_fragment]
float4 fs_main(Interpolated<float4> tex, Filter smp, VertexOutput input) : SV_Target {
    return tex.Sample(smp, input.uv);
}
"#;

#[test]
fn cuda_raster_bindless_buffer_tint() {
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

    let shader = ShaderModule::from_slang(&device, BINDLESS_TINT_SHADER).expect("shader");
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
    // White vertices × green tint → green. (Component-wise red×green is black.)
    let vertices = [
        Vertex2D::new(0.0, 0.5, Color::WHITE),
        Vertex2D::new(-0.5, -0.5, Color::WHITE),
        Vertex2D::new(0.5, -0.5, Color::WHITE),
    ];
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vertex buffer");
    // Tint multiplies vertex white by green → green output.
    let tint = pool
        .acquire_buffer_with_data(&[[0.0f32, 1.0, 0.0, 1.0]], BufferKind::Scattered)
        .expect("tint buffer");
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
        let mut pass = scheme.render_pass("tint", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_shader_resources(&[ShaderResourceSlot::Parcel {
            parcel: &tint,
            access: NodeAccess::Read,
        }]);
        pass.with_parcel(&vertex_buffer, NodeAccess::Read);
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

    let x = 32usize;
    let y = 28usize;
    let offset = (y * 64 + x) * 16;
    let r = f32::from_le_bytes(pixels[offset..offset + 4].try_into().unwrap());
    let g = f32::from_le_bytes(pixels[offset + 4..offset + 8].try_into().unwrap());
    let b = f32::from_le_bytes(pixels[offset + 8..offset + 12].try_into().unwrap());
    assert!(
        r < 0.25 && g > 0.5 && b < 0.25,
        "expected green tinted pixel at ({x},{y}), got ({r},{g},{b})"
    );

    // Retained replay with unchanged bindings must succeed a second submit.
    let _ = scheme.submit().expect("resubmit");
}

#[test]
fn cuda_raster_bindless_tint_change_rerecords() {
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

    let shader = ShaderModule::from_slang(&device, BINDLESS_TINT_SHADER).expect("shader");
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
    let vertices = [
        Vertex2D::new(0.0, 0.5, Color::WHITE),
        Vertex2D::new(-0.5, -0.5, Color::WHITE),
        Vertex2D::new(0.5, -0.5, Color::WHITE),
    ];
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vertex buffer");
    let green = pool
        .acquire_buffer_with_data(&[[0.0f32, 1.0, 0.0, 1.0]], BufferKind::Scattered)
        .expect("green tint");
    let blue = pool
        .acquire_buffer_with_data(&[[0.0f32, 0.0, 1.0, 1.0]], BufferKind::Scattered)
        .expect("blue tint");
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

    let sample = |pixels: &[u8]| {
        let offset = (28usize * 64 + 32) * 16;
        (
            f32::from_le_bytes(pixels[offset..offset + 4].try_into().unwrap()),
            f32::from_le_bytes(pixels[offset + 4..offset + 8].try_into().unwrap()),
            f32::from_le_bytes(pixels[offset + 8..offset + 12].try_into().unwrap()),
        )
    };

    // Frame 1: green tint.
    let mut scheme = Scheme::new(&ctx);
    let rt = scheme
        .lease_render_target(64, 64, TextureFormat::Rgba32Float, None)
        .expect("render target");
    {
        let mut pass = scheme.render_pass("green", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_shader_resources(&[ShaderResourceSlot::Parcel {
            parcel: &green,
            access: NodeAccess::Read,
        }]);
        pass.with_parcel(&vertex_buffer, NodeAccess::Read);
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
    let pixels = grant.claim(&mut submission).expect("claim").consume().expect("consume").to_vec();
    let (r, g, b) = sample(&pixels);
    assert!(r < 0.25 && g > 0.5 && b < 0.25, "frame1 expected green, got ({r},{g},{b})");

    // Frame 2: blue tint — changed bindings must not replay the green list.
    let mut scheme = Scheme::new(&ctx);
    let rt = scheme
        .lease_render_target(64, 64, TextureFormat::Rgba32Float, None)
        .expect("render target");
    {
        let mut pass = scheme.render_pass("blue", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_shader_resources(&[ShaderResourceSlot::Parcel {
            parcel: &blue,
            access: NodeAccess::Read,
        }]);
        pass.with_parcel(&vertex_buffer, NodeAccess::Read);
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
    let pixels = grant.claim(&mut submission).expect("claim").consume().expect("consume").to_vec();
    let (r, g, b) = sample(&pixels);
    assert!(r < 0.25 && g < 0.25 && b > 0.5, "frame2 expected blue, got ({r},{g},{b})");
}

#[test]
fn cuda_raster_bindless_sampled_texture() {
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

    let shader = ShaderModule::from_slang(&device, BINDLESS_TEXTURE_SHADER).expect("shader");
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

    // Solid blue Rgba32Float 4x4 texture.
    let mut tex_data = vec![0u8; 4 * 4 * 16];
    for pixel in tex_data.chunks_exact_mut(16) {
        pixel[0..4].copy_from_slice(&0.0f32.to_le_bytes());
        pixel[4..8].copy_from_slice(&0.0f32.to_le_bytes());
        pixel[8..12].copy_from_slice(&1.0f32.to_le_bytes());
        pixel[12..16].copy_from_slice(&1.0f32.to_le_bytes());
    }
    let texture = pool
        .acquire_texture(
            4,
            4,
            TextureFormat::Rgba32Float,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
            Some(&tex_data),
        )
        .expect("texture");
    let sampler = Sampler::new(
        &device,
        &SamplerDesc {
            mag_filter: goldy::types::FilterMode::Nearest,
            min_filter: goldy::types::FilterMode::Nearest,
            mipmap_filter: goldy::types::FilterMode::Nearest,
            ..Default::default()
        },
    )
    .expect("sampler");
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
        let mut pass = scheme.render_pass("tex", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_shader_resources(&[
            ShaderResourceSlot::Parcel {
                parcel: &texture,
                access: NodeAccess::Read,
            },
            ShaderResourceSlot::Sampler(&sampler),
        ]);
        pass.with_parcel(&vertex_buffer, NodeAccess::Read);
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

    let x = 32usize;
    let y = 28usize;
    let offset = (y * 64 + x) * 16;
    let r = f32::from_le_bytes(pixels[offset..offset + 4].try_into().unwrap());
    let g = f32::from_le_bytes(pixels[offset + 4..offset + 8].try_into().unwrap());
    let b = f32::from_le_bytes(pixels[offset + 8..offset + 12].try_into().unwrap());
    assert!(
        r < 0.25 && g < 0.25 && b > 0.5,
        "expected blue sampled pixel at ({x},{y}), got ({r},{g},{b})"
    );
}
