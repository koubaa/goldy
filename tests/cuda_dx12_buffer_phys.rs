//! Corner cases for CUDA+DX12 late buffer physicalization.
//!
//! Pool acquire is logical-only; scheme usage chooses Shared / Native / NativeAndTwin.
//! These tests stress scheme deletion, multi-scheme reuse of one retained buffer, and
//! promotions that must not invalidate client schemes.

#![cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]

use goldy::types::BackendType;
use goldy::{
    test_support, BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, MemoryExchange,
    PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule, TargetLoad, TextureFlags, TextureFormat, TextureKind, Vertex2D,
};
use std::sync::Arc;

fn try_cuda_instance() -> Option<Instance> {
    // SAFETY: test process; GOLDY_BACKEND is read during Instance::new.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cuda") };
    Instance::new().ok().filter(|i| {
        i.enumerate_adapters()
            .into_iter()
            .any(|a| a.get_info().backend == BackendType::Cuda)
    })
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

const FILL_VERTS_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<float> verts, ThreadId id) {
    verts[0] = 0.0;  verts[1] = 0.5;
    verts[2] = 1.0;  verts[3] = 0.0; verts[4] = 0.0; verts[5] = 1.0;
    verts[6] = -0.5; verts[7] = -0.5;
    verts[8] = 1.0;  verts[9] = 0.0; verts[10] = 0.0; verts[11] = 1.0;
    verts[12] = 0.5; verts[13] = -0.5;
    verts[14] = 1.0; verts[15] = 0.0; verts[16] = 0.0; verts[17] = 1.0;
}
"#;

const RECOLOR_VERTS_SHADER: &str = r#"
import goldy_exp;

// Flip RGB of each Vertex2D color (floats 2..4 of each 6-float vertex).
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<float> verts, ThreadId id) {
    for (int v = 0; v < 3; v++) {
        int base = v * 6;
        float r = verts[base + 2];
        float g = verts[base + 3];
        float b = verts[base + 4];
        verts[base + 2] = g;
        verts[base + 3] = b;
        verts[base + 4] = r;
    }
}
"#;

fn red_triangle() -> [Vertex2D; 3] {
    [
        Vertex2D::new(0.0, 0.5, Color::RED),
        Vertex2D::new(-0.5, -0.5, Color::RED),
        Vertex2D::new(0.5, -0.5, Color::RED),
    ]
}

fn sample_centroid(pixels: &[u8]) -> (f32, f32, f32) {
    let x = 32usize;
    let y = 28usize;
    let offset = (y * 64 + x) * 16;
    (
        f32::from_le_bytes(pixels[offset..offset + 4].try_into().unwrap()),
        f32::from_le_bytes(pixels[offset + 4..offset + 8].try_into().unwrap()),
        f32::from_le_bytes(pixels[offset + 8..offset + 12].try_into().unwrap()),
    )
}

fn draw_and_readback(
    ctx: &goldy::Context,
    device: &Arc<goldy::Device>,
    pipeline: &RenderPipeline,
    vertex_buffer: &goldy::Parcel,
    readback: &goldy::Parcel,
) -> Vec<u8> {
    let _ = device;
    let mut scheme = Scheme::new(ctx);
    let rt = scheme
        .lease_render_target(64, 64, TextureFormat::Rgba32Float, None)
        .expect("render target");
    {
        let mut pass = scheme.render_pass("tri", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_parcel(vertex_buffer, goldy::NodeAccess::Read);
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_buffer);
        pass.draw(0..3, 0..1);
        pass.finish();
    }
    scheme.copy_to_texture(&rt, readback).expect("copy_to_texture");
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, readback)
        .expect("withdraw");
    let mut submission = scheme.submit().expect("submit");
    grant
        .claim(&mut submission)
        .expect("claim")
        .consume()
        .expect("consume")
        .to_vec()
}

#[test]
fn acquire_with_data_raster_lands_shared() {
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter");
    let device = Arc::new(
        adapter
            .request_device(&DeviceDescriptor::default())
            .expect("device"),
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
    .expect("pipeline");

    let mut pool = RetainedPool::new(Arc::clone(&device));
    let vertex_buffer = pool
        .acquire_buffer_with_data(&red_triangle(), BufferKind::Scattered)
        .expect("vb");
    assert_eq!(
        test_support::cuda_buffer_phys_kind(&device, &vertex_buffer),
        Some("deferred"),
        "acquire must defer physical backing"
    );

    let readback = pool
        .acquire_texture(
            64,
            64,
            TextureFormat::Rgba32Float,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
            None,
        )
        .expect("readback");

    let before = device.cuda_path_stats_for_test().expect("stats");
    let pixels = draw_and_readback(&ctx, &device, &pipeline, &vertex_buffer, &readback);
    let (r, g, b) = sample_centroid(&pixels);
    assert!(r > 0.5 && g < 0.25 && b < 0.25, "got ({r},{g},{b})");
    assert_eq!(
        test_support::cuda_buffer_phys_kind(&device, &vertex_buffer),
        Some("shared"),
        "host+vertex without kernel must land Shared"
    );
    let after = device.cuda_path_stats_for_test().expect("stats");
    assert!(
        after.buffer_materializations > before.buffer_materializations,
        "expected materialization"
    );
}

#[test]
fn separate_deposit_then_draw_promotes_to_shared() {
    // Spinning-cube shape: deposit scheme first (no VERTEX), then draw scheme.
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter");
    let device = Arc::new(
        adapter
            .request_device(&DeviceDescriptor::default())
            .expect("device"),
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
    .expect("pipeline");

    let mut pool = RetainedPool::new(Arc::clone(&device));
    let vertex_buffer = pool
        .acquire_buffer_sized::<Vertex2D>(3, BufferKind::Scattered, goldy::BufferFlags::empty())
        .expect("vb");
    let readback = pool
        .acquire_texture(
            64,
            64,
            TextureFormat::Rgba32Float,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
            None,
        )
        .expect("readback");

    let mut upload = Scheme::new(&ctx);
    let deposit = MemoryExchange::new(&ctx)
        .bind_deposit_buffer(&mut upload, &vertex_buffer, vertex_buffer.byte_size())
        .expect("deposit");
    deposit
        .write(&mut upload, 0, bytemuck::cast_slice(&red_triangle()))
        .expect("write");
    upload.submit().expect("upload");
    // Deposit alone may provisional-Native the buffer.
    let kind_after_deposit = test_support::cuda_buffer_phys_kind(&device, &vertex_buffer);
    assert!(
        matches!(kind_after_deposit, Some("native") | Some("deferred") | Some("shared")),
        "unexpected kind after deposit: {kind_after_deposit:?}"
    );

    let before = device.cuda_path_stats_for_test().expect("stats");
    let pixels = draw_and_readback(&ctx, &device, &pipeline, &vertex_buffer, &readback);
    let (r, g, b) = sample_centroid(&pixels);
    assert!(r > 0.5 && g < 0.25 && b < 0.25, "got ({r},{g},{b})");
    assert_eq!(
        test_support::cuda_buffer_phys_kind(&device, &vertex_buffer),
        Some("shared")
    );
    let after = device.cuda_path_stats_for_test().expect("stats");
    if kind_after_deposit == Some("native") {
        assert!(
            after.buffer_promotions > before.buffer_promotions,
            "Native→Shared promotion expected when deposit preceded VERTEX"
        );
    }
}

#[test]
fn compute_then_raster_lands_native_and_twin() {
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter");
    let device = Arc::new(
        adapter
            .request_device(&DeviceDescriptor::default())
            .expect("device"),
    );
    let ctx = device.create_context().expect("context");
    let vs_fs = ShaderModule::from_slang(&device, TRIANGLE_SHADER).expect("gfx");
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
    .expect("pipeline");
    let cs = ShaderModule::from_slang(&device, FILL_VERTS_SHADER).expect("cs");
    let compute = ComputePipeline::new(&device, &cs).expect("compute");

    let mut pool = RetainedPool::new(Arc::clone(&device));
    let vertex_buffer = pool
        .acquire_buffer_sized::<f32>(18, BufferKind::Scattered, goldy::BufferFlags::empty())
        .expect("vb");
    let readback = pool
        .acquire_texture(
            64,
            64,
            TextureFormat::Rgba32Float,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
            None,
        )
        .expect("readback");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("gen", &compute)
        .with_parcel(&vertex_buffer, goldy::NodeAccess::Write)
        .dispatch(1, 1, 1);
    let rt = scheme
        .lease_render_target(64, 64, TextureFormat::Rgba32Float, None)
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
    let pixels = grant.claim(&mut submission).expect("claim").consume().expect("consume");
    let (r, g, b) = sample_centroid(&pixels);
    assert!(r > 0.5 && g < 0.25 && b < 0.25, "got ({r},{g},{b})");
    assert_eq!(
        test_support::cuda_buffer_phys_kind(&device, &vertex_buffer),
        Some("native_and_twin")
    );
}

#[test]
fn shared_then_kernel_promotes_without_invalidating_schemes() {
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter");
    let device = Arc::new(
        adapter
            .request_device(&DeviceDescriptor::default())
            .expect("device"),
    );
    let ctx = device.create_context().expect("context");
    let vs_fs = ShaderModule::from_slang(&device, TRIANGLE_SHADER).expect("gfx");
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
    .expect("pipeline");
    let cs = ShaderModule::from_slang(&device, RECOLOR_VERTS_SHADER).expect("cs");
    let recolor = ComputePipeline::new(&device, &cs).expect("compute");

    let mut pool = RetainedPool::new(Arc::clone(&device));
    let vertex_buffer = pool
        .acquire_buffer_with_data(&red_triangle(), BufferKind::Scattered)
        .expect("vb");
    let readback = pool
        .acquire_texture(
            64,
            64,
            TextureFormat::Rgba32Float,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
            None,
        )
        .expect("readback");

    // Land Shared via raster.
    let pixels = draw_and_readback(&ctx, &device, &pipeline, &vertex_buffer, &readback);
    let (r, g, b) = sample_centroid(&pixels);
    assert!(r > 0.5 && g < 0.25 && b < 0.25, "pre-promote ({r},{g},{b})");
    assert_eq!(
        test_support::cuda_buffer_phys_kind(&device, &vertex_buffer),
        Some("shared")
    );

    let before = device.cuda_path_stats_for_test().expect("stats");

    // Same retained buffer: kernel + draw in one scheme → promote Shared→NativeAndTwin.
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("recolor", &recolor)
        .with_parcel(&vertex_buffer, goldy::NodeAccess::ReadWrite)
        .dispatch(1, 1, 1);
    let rt = scheme
        .lease_render_target(64, 64, TextureFormat::Rgba32Float, None)
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
    let mut submission = scheme.submit().expect("submit after promote");
    let pixels = grant.claim(&mut submission).expect("claim").consume().expect("consume");
    // Recolor rotates R→G channel into R slot from previous G(=0) — expect dark/greenish.
    // Original red (1,0,0) → (0,0,1) blue after one rotate of (r,g,b)->(g,b,r).
    let (r, g, b) = sample_centroid(&pixels);
    assert!(
        b > 0.5 && r < 0.25 && g < 0.25,
        "expected blue after recolor, got ({r},{g},{b})"
    );
    assert_eq!(
        test_support::cuda_buffer_phys_kind(&device, &vertex_buffer),
        Some("native_and_twin")
    );
    let after = device.cuda_path_stats_for_test().expect("stats");
    assert!(
        after.buffer_promotions > before.buffer_promotions,
        "expected Shared→NativeAndTwin promotion"
    );
}

#[test]
fn scheme_delete_and_multi_scheme_same_retained_buffer() {
    // Drop the deposit scheme; rebuild a new one against the same pool buffer.
    // Render scheme retains across frames and must keep seeing updated Shared bytes.
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter");
    let device = Arc::new(
        adapter
            .request_device(&DeviceDescriptor::default())
            .expect("device"),
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
    .expect("pipeline");

    let mut pool = RetainedPool::new(Arc::clone(&device));
    let vertex_buffer = pool
        .acquire_buffer_sized::<Vertex2D>(3, BufferKind::Scattered, goldy::BufferFlags::empty())
        .expect("vb");
    let readback = pool
        .acquire_texture(
            64,
            64,
            TextureFormat::Rgba32Float,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
            None,
        )
        .expect("readback");

    // Warm render scheme retention with black verts via a throwaway deposit scheme.
    {
        let mut upload = Scheme::new(&ctx);
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit_buffer(&mut upload, &vertex_buffer, vertex_buffer.byte_size())
            .expect("deposit");
        let black = [
            Vertex2D::new(0.0, 0.5, Color::BLACK),
            Vertex2D::new(-0.5, -0.5, Color::BLACK),
            Vertex2D::new(0.5, -0.5, Color::BLACK),
        ];
        deposit
            .write(&mut upload, 0, bytemuck::cast_slice(&black))
            .expect("write");
        upload.submit().expect("upload");
    } // upload scheme dropped — physical buffer identity must remain valid

    let mut draw = Scheme::new(&ctx);
    let rt = draw
        .lease_render_target(64, 64, TextureFormat::Rgba32Float, None)
        .expect("rt");
    {
        let mut pass = draw.render_pass("tri", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_parcel(&vertex_buffer, goldy::NodeAccess::Read);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..3, 0..1);
        pass.finish();
    }
    draw.copy_to_texture(&rt, &readback).expect("copy");
    {
        let grant = MemoryExchange::new(draw.context())
            .bind_withdraw(&mut draw, &readback)
            .expect("withdraw");
        let mut submission = draw.submit().expect("warm draw");
        let _ = grant.claim(&mut submission).expect("claim").consume();
    }
    assert_eq!(
        test_support::cuda_buffer_phys_kind(&device, &vertex_buffer),
        Some("shared")
    );

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

    for (frame_i, (color, expect)) in frames.iter().enumerate() {
        // Brand-new deposit scheme each frame (delete/recreate stress).
        let mut upload = Scheme::new(&ctx);
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit_buffer(&mut upload, &vertex_buffer, vertex_buffer.byte_size())
            .expect("deposit");
        let verts = [
            Vertex2D::new(0.0, 0.5, *color),
            Vertex2D::new(-0.5, -0.5, *color),
            Vertex2D::new(0.5, -0.5, *color),
        ];
        deposit
            .write(&mut upload, 0, bytemuck::cast_slice(&verts))
            .expect("write");
        upload.submit().expect("upload");
        drop(upload);

        let grant = MemoryExchange::new(draw.context())
            .bind_withdraw(&mut draw, &readback)
            .expect("withdraw");
        let mut submission = draw.submit().expect("draw resubmit");
        let pixels = grant
            .claim(&mut submission)
            .expect("claim")
            .consume()
            .expect("consume")
            .to_vec();
        let (r, g, b) = sample_centroid(&pixels);
        assert!(
            expect(r, g, b),
            "frame {frame_i}: stale Shared content ({r},{g},{b})"
        );
        assert_eq!(
            test_support::cuda_buffer_phys_kind(&device, &vertex_buffer),
            Some("shared"),
            "phys kind must stay Shared across scheme delete/recreate"
        );
    }
}
