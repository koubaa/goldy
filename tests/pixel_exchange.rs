//! Pixel exchange on the CPU backend (`GOLDY_BACKEND=cpu`).
//!
//! Isolated crate so the env override cannot race other GPU tests. Fine writes a
//! buffer pixmap; [`goldy::PixelExchange`] withdraws it into a [`goldy::PixelSink`].
//! With a graphics feature, a second test copies the same pixmap through a
//! foreign image that is **not** a Goldy device.

use goldy::{
    BufferKind, DeviceDescriptor, HostPixelSink, Instance, NodeAccess, PixelExchange, PixmapLayout,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, TextureFormat,
};
use std::sync::Arc;

fn select_cpu() {
    // SAFETY: this integration test is its own process.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") };
}

fn cpu_device() -> goldy::Device {
    let instance = Instance::new().expect("instance");
    assert_eq!(instance.backend_type(), goldy::BackendType::Cpu);
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device")
}

fn fill_pixmap(device: &goldy::Device, n: u32) -> (goldy::Buffer, goldy::Context, RetainedPool) {
    let ctx = device.create_context().expect("ctx");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let zeros = vec![0u32; n as usize];
    let buf = pool
        .acquire_buffer_with_data(&zeros, BufferKind::Scattered)
        .expect("buffer");
    let src = r#"
        import goldy_exp;
        [goldy_compute]
        [numthreads(64, 1, 1)]
        void cs_main(Scattered<uint> pixels, ThreadId id) {
            if (id.x < goldy_buf_len(pixels)) {
                pixels[id.x] = 0xFF000000u | id.x;
            }
        }
    "#;
    let shader = ShaderModule::from_slang(device, src).expect("compile");
    let pipeline = goldy::ComputePipeline::new(device, &shader).expect("pipeline");
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fill", &pipeline)
        .with_parcel(buf.whole(), NodeAccess::ReadWrite)
        .dispatch(n.div_ceil(64), 1, 1);
    scheme.submit().expect("submit").wait_until_settled().expect("wait");
    (buf, ctx, pool)
}

fn expected_words() -> Vec<u32> {
    vec![0xFF000000, 0xFF000001, 0xFF000002, 0xFF000003]
}

fn consume_into_sink(ctx: &goldy::Context, buf: &goldy::Buffer, sink: Arc<dyn goldy::PixelSink>) {
    let layout = PixmapLayout::tight(2, 2, TextureFormat::Rgba8Unorm);
    let exchange = PixelExchange::new(ctx, sink);
    let mut scheme = Scheme::new(ctx);
    let tx = exchange.bind_source(&mut scheme, buf.whole(), layout).unwrap();
    let mut submission = scheme.submit().unwrap();
    tx.claim(&mut submission).unwrap().consume().unwrap();
}

#[test]
fn cpu_compute_blits_to_host_sink() {
    select_cpu();
    let device = cpu_device();
    let (buf, ctx, _pool) = fill_pixmap(&device, 4);
    let sink = Arc::new(HostPixelSink::new(2, 2, TextureFormat::Rgba8Unorm).unwrap());
    consume_into_sink(&ctx, &buf, sink.clone());
    let words: Vec<u32> = bytemuck::cast_slice(&sink.pixels()).to_vec();
    assert_eq!(words, expected_words());
}

#[cfg(feature = "vulkan")]
#[test]
fn cpu_compute_blits_to_foreign_vulkan() {
    select_cpu();
    let Some(adapter) = goldy::foreign::vulkan::try_adapter() else {
        eprintln!("skipping: foreign Vulkan adapter unavailable");
        return;
    };
    let device = cpu_device();
    let (buf, ctx, _pool) = fill_pixmap(&device, 4);
    let layout = PixmapLayout::tight(2, 2, TextureFormat::Rgba8Unorm);
    let surface = adapter.offscreen(2, 2, TextureFormat::Rgba8Unorm).expect("offscreen");
    consume_into_sink(&ctx, &buf, Arc::new(surface.clone()));
    let words: Vec<u32> = bytemuck::cast_slice(&surface.snapshot(layout).expect("snapshot")).to_vec();
    assert_eq!(words, expected_words());
}

#[cfg(all(feature = "dx12", target_os = "windows"))]
#[test]
fn cpu_compute_blits_to_foreign_dx12() {
    select_cpu();
    let adapter = goldy::foreign::dx12::try_adapter().expect("foreign DX12 adapter (WARP or hardware)");
    let device = cpu_device();
    let (buf, ctx, _pool) = fill_pixmap(&device, 4);
    let layout = PixmapLayout::tight(2, 2, TextureFormat::Rgba8Unorm);
    let surface = adapter.offscreen(2, 2, TextureFormat::Rgba8Unorm).expect("offscreen");
    consume_into_sink(&ctx, &buf, Arc::new(surface.clone()));
    let words: Vec<u32> = bytemuck::cast_slice(&surface.snapshot(layout).expect("snapshot")).to_vec();
    assert_eq!(words, expected_words());
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn cpu_compute_blits_to_foreign_metal() {
    select_cpu();
    let adapter = goldy::foreign::metal::try_adapter().expect("foreign Metal adapter");
    let device = cpu_device();
    let (buf, ctx, _pool) = fill_pixmap(&device, 4);
    let layout = PixmapLayout::tight(2, 2, TextureFormat::Rgba8Unorm);
    let surface = adapter.offscreen(2, 2, TextureFormat::Rgba8Unorm).expect("offscreen");
    consume_into_sink(&ctx, &buf, Arc::new(surface.clone()));
    let words: Vec<u32> = bytemuck::cast_slice(&surface.snapshot(layout).expect("snapshot")).to_vec();
    assert_eq!(words, expected_words());
}
