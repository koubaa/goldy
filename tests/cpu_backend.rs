//! Compute-only CPU backend via `GOLDY_BACKEND=cpu` (issue #292).
//!
//! Isolated crate so the env override cannot race other GPU tests.

use goldy::{
    BufferKind, DeviceDescriptor, Instance, MemoryExchange, NodeAccess, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule,
};
use std::sync::Arc;

fn cpu_device() -> goldy::Device {
    let instance = Instance::new().expect("instance");
    assert_eq!(instance.backend_type(), goldy::BackendType::Cpu);
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device")
}

fn run_scheme_double_u32() {
    let device = cpu_device();
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

#[test]
fn scheme_double_u32() {
    // SAFETY: this integration test is its own process.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") };
    run_scheme_double_u32();
}

#[test]
fn scheme_double_u32_host_access() {
    // SAFETY: this integration test is its own process.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") };
    let _protect = goldy::test_support::HostAccessOverride::force_enabled();
    run_scheme_double_u32();
}

fn run_workgroup_kernel(src: &str, n: usize, check: impl Fn(&[u32])) {
    let device = cpu_device();
    let ctx = device.create_context().expect("ctx");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let zeros = vec![0u32; n];
    let out = pool
        .acquire_buffer_with_data(&zeros, BufferKind::Scattered)
        .expect("out");
    let shader = ShaderModule::from_slang(&device, src).expect("compile");
    let pipeline = goldy::ComputePipeline::new(&device, &shader).expect("pipeline");
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("wg", &pipeline)
        .with_parcel(&out, NodeAccess::ReadWrite)
        .dispatch(1, 1, 1);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &out)
        .expect("withdraw");
    let mut frame = scheme.submit().expect("submit");
    let bytes = grant.claim(&mut frame).expect("claim").consume().expect("consume");
    check(bytemuck::cast_slice(&bytes));
}

const REDUCE_64: &str = r#"
    import goldy_exp;
    groupshared uint sh_scratch[64];
    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
        uint ix  = local_id.x;
        uint val = 1u;
        sh_scratch[ix] = val;
        for (uint i = 0; i < 6; i++) {
            GroupMemoryBarrierWithGroupSync();
            if (ix + (1u << i) < 64u)
                val = val + sh_scratch[ix + (1u << i)];
            GroupMemoryBarrierWithGroupSync();
            sh_scratch[ix] = val;
        }
        OUT[ix] = val;
    }
"#;

const INCLUSIVE_SCAN_64: &str = r#"
    import goldy_exp;
    groupshared uint sh_scratch[64];
    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
        uint ix  = local_id.x;
        uint val = 1u;
        sh_scratch[ix] = val;
        for (uint i = 0; i < 6; i++) {
            GroupMemoryBarrierWithGroupSync();
            if (ix >= (1u << i))
                val = sh_scratch[ix - (1u << i)] + val;
            GroupMemoryBarrierWithGroupSync();
            sh_scratch[ix] = val;
        }
        OUT[ix] = val;
    }
"#;

#[test]
#[ignore = "CPU workgroups are serial loops; barrier reduce is not implemented"]
fn scheme_workgroup_reduce_uint() {
    // SAFETY: this integration test is its own process.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") };
    run_workgroup_kernel(REDUCE_64, 64, |out| {
        assert_eq!(out[0], 64, "thread 0 holds the workgroup sum");
    });
}

#[test]
#[ignore = "CPU workgroups are serial loops; barrier scan is not implemented"]
fn scheme_workgroup_inclusive_scan_uint() {
    // SAFETY: this integration test is its own process.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") };
    run_workgroup_kernel(INCLUSIVE_SCAN_64, 64, |out| {
        for i in 0..64u32 {
            assert_eq!(out[i as usize], i + 1, "inclusive scan[{i}]");
        }
    });
}
