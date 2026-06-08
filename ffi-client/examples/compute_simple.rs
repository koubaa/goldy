//! Headless compute example using goldy-ffi-client.
//!
//! Run from `goldy/ffi-client`: `cargo run --example compute_simple`

use goldy_ffi_client::{
    BufferKind, ComputeEncoder, ComputePipeline, DeviceDescriptor, Instance, RequestAdapterOptions, ShaderModule,
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
    let buffer = device.alloc_buffer_with_data(&data, BufferKind::Scattered)?;

    let shader = ShaderModule::from_slang(&device, COMPUTE_SRC)?;
    let pipeline = ComputePipeline::new(&device, &shader)?;

    let mut encoder = ComputeEncoder::new();
    encoder.set_pipeline(&pipeline);
    encoder.bind_resources(&[&buffer]);
    encoder.dispatch(1, 1, 1);
    encoder.execute(&device)?;

    println!("Compute dispatch completed successfully.");
    Ok(())
}
