//! Headless compute example using goldy-ffi-client.
//!
//! Run from `goldy/ffi-client`: `cargo run --example compute_simple`

use goldy_ffi_client::{
    BufferKind, ComputePipeline, Context, DeviceDescriptor, Instance, MemoryExchange, NodeAccess,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule,
};

const COMPUTE_SRC: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<float> data, ThreadId id) {
    uint idx = id.x;
    if (idx < 64u) {
        data[idx] = float(idx) * 2.0;
    }
}
"#;

fn main() -> goldy_ffi_client::Result<()> {
    println!("Goldy compute_simple (ffi-client)\n");

    let instance = Instance::new()?;
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())?
        .request_device(&DeviceDescriptor::default())?;

    let data = [0f32; 64];
    let mut retained_pool = RetainedPool::new(&device)?;
    let buffer = retained_pool.acquire_buffer_with_data(&data, BufferKind::Scattered)?;
    let _retained_pool = retained_pool;

    let shader = ShaderModule::from_slang(&device, COMPUTE_SRC)?;
    let pipeline = ComputePipeline::new(&device, &shader)?;

    let ctx = Context::new(&device)?;
    let mut scheme = Scheme::new(&ctx)?;
    let mut node = scheme.compute_node("double", &pipeline);
    node.with_buffer(&buffer, NodeAccess::ReadWrite);
    node.dispatch(1, 1, 1);
    let memory = MemoryExchange::new(&ctx)?;
    let parcel = buffer.field(0)?;
    let withdraw = memory.bind_withdraw(&mut scheme, &parcel)?;
    let mut submission = scheme.submit()?;
    let claim = withdraw.claim(&mut submission)?;
    let bytes = claim.consume()?;
    let values: &[f32] = bytemuck::cast_slice(&bytes);
    for (i, &v) in values.iter().enumerate().take(64) {
        let expected = i as f32 * 2.0;
        assert!((v - expected).abs() < 1e-4, "index {i}: expected {expected}, got {v}");
    }

    println!("Compute dispatch verified: data[i] == i * 2 for 64 elements.");
    Ok(())
}
