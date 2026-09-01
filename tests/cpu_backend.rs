//! Compute-only CPU backend via `GOLDY_BACKEND=cpu` (issue #292).
//!
//! Isolated crate so the env override cannot race other GPU tests.

use goldy::{
    BufferKind, DeviceDescriptor, Instance, MemoryExchange, NodeAccess, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule,
};
use std::sync::Arc;

#[test]
fn scheme_double_u32() {
    // SAFETY: this integration test is its own process.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") };

    let instance = Instance::new().expect("instance");
    assert_eq!(instance.backend_type(), goldy::BackendType::Cpu);
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device");
    assert_eq!(device.backend_type(), goldy::BackendType::Cpu);

    let ctx = device.create_context().expect("ctx");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let n = 64usize;
    let input: Vec<u32> = (0..n as u32).collect();
    let data = pool
        .acquire_buffer_with_data(&input, BufferKind::Scattered)
        .expect("buffer");

    let src = r#"
        import goldy_exp;
        [goldy_compute]
        [numthreads(64, 1, 1)]
        void cs_main(Scattered<uint> data, ThreadId id) {
            if (id.x < goldy_buf_len(data)) {
                data[id.x] = data[id.x] * 2u;
            }
        }
    "#;
    let shader = ShaderModule::from_slang(&device, src).expect("compile");
    let pipeline = goldy::ComputePipeline::new(&device, &shader).expect("pipeline");
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("double", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .dispatch((n as u32).div_ceil(64), 1, 1);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &data)
        .expect("withdraw");
    let mut frame = scheme.submit().expect("submit");
    let bytes = grant.claim(&mut frame).expect("claim").consume().expect("consume");
    let out: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
    assert_eq!(out.len(), n);
    for i in 0..n {
        assert_eq!(out[i], (i as u32) * 2, "index {i}");
    }
}
