#![allow(deprecated)]

#[path = "common/submission.rs"]
mod submission;

#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
mod imp {
    //! Compute pipeline integration tests (skip bucket).
    //!
    //! Migrated dispatch/readback tests live in `scheme_compute_integration.rs`.
    //! This file retains allocation, lifecycle, shader-math, validation, and pool tests
    //! that do not depend on `TaskGraph`.

    use crate::submission::submission_context;
    use goldy::{
        types::{BackendType, BufferFlags},
        Buffer, BufferKind, ComputePipeline, Device, DeviceDescriptor, Instance, MemoryExchange, NodeAccess,
        RequestAdapterOptions, RetainedPool, Scheme, ShaderModule,
    };
    use std::sync::Arc;

    fn request_default_device(instance: &Instance) -> Device {
        instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("Failed to request adapter")
            .request_device(&DeviceDescriptor::default())
            .expect("Failed to create device")
    }

    fn make_device() -> Device {
        request_default_device(&Instance::new().expect("Failed to create instance"))
    }

    fn test_alloc_buffer(
        device: &Device,
        size: u64,
        kind: BufferKind,
        stride: Option<u32>,
        flags: BufferFlags,
    ) -> Buffer {
        RetainedPool::new(Arc::new(device.clone()))
            .acquire_buffer(size, kind, stride, flags, None)
            .expect("acquire_buffer")
    }

    fn test_alloc_buffer_with_data<T: goldy::StructuredBufferElement>(
        device: &Device,
        data: &[T],
        kind: BufferKind,
    ) -> Buffer {
        RetainedPool::new(Arc::new(device.clone()))
            .acquire_buffer_with_data(data, kind)
            .expect("acquire_buffer_with_data")
    }

    /// Advance the context timeline with an empty scheme submission.
    fn scheme_submit_empty(ctx: &goldy::Context) -> u64 {
        let mut scheme = Scheme::new(ctx);
        goldy::test_support::submission_epoch(&scheme.submit().expect("submit empty"))
    }

    fn test_compute_pipeline_creation(device: &Device) {
        const DOUBLE_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        data[id.x] = data[id.x] * 2;
    }
    "#;

        let shader = ShaderModule::from_slang(device, DOUBLE_SHADER).expect("Failed to compile shader");
        let pipeline = ComputePipeline::new(device, &shader);
        assert!(
            pipeline.is_ok(),
            "Failed to create compute pipeline: {:?}",
            pipeline.err()
        );
    }

    fn test_compute_pipeline_no_bindings(device: &Device) {
        const MINIMAL_SHADER: &str = r#"
    [shader("compute")]
    [numthreads(1, 1, 1)]
    void cs_main(uint3 id : SV_DispatchThreadID) {
    }
    "#;

        let shader = ShaderModule::from_slang(device, MINIMAL_SHADER).expect("Failed to compile shader");
        let pipeline = ComputePipeline::new(device, &shader);
        assert!(
            pipeline.is_ok(),
            "Failed to create minimal compute pipeline: {:?}",
            pipeline.err()
        );
    }

    #[cfg(feature = "vulkan")]
    const MINIMAL_COMPUTE_FOR_VK_VALIDATION: &str = r#"
    [shader("compute")]
    [numthreads(1, 1, 1)]
    void cs_main(uint3 id : SV_DispatchThreadID) {
    }
    "#;

    #[cfg(feature = "vulkan")]
    fn vk_api_validation_active_backend_is_vulkan() -> bool {
        let Ok(instance) = Instance::new() else {
            return false;
        };
        instance.backend_type() == BackendType::Vulkan
    }

    #[cfg(feature = "vulkan")]
    fn run_in_subprocess_with_vk_validation(test_name: &str) {
        let exe = std::env::current_exe().expect("current_exe for subprocess");
        let output = std::process::Command::new(exe)
            .args([test_name, "--exact", "--nocapture"])
            .env("GOLDY_SUBPROC", "1")
            .env("GOLDY_VALIDATION", "api")
            .env("GOLDY_BACKEND", "vk")
            .env_remove("VK_LAYER_PATH")
            .output()
            .unwrap_or_else(|e| panic!("spawn subprocess for {test_name}: {e}"));

        assert!(
            output.status.success(),
            "Vulkan validation subprocess for `{test_name}` failed (exit {:?})\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    #[cfg(feature = "vulkan")]
    fn vk_api_validation_timeline_semaphore() {
        if std::env::var("GOLDY_SUBPROC").is_err() {
            if !vk_api_validation_active_backend_is_vulkan() {
                return;
            }
            run_in_subprocess_with_vk_validation("vk_api_validation_timeline_semaphore");
            return;
        }

        let instance = Instance::new().expect("instance");
        let device = request_default_device(&instance);
        let ctx = submission_context(&device);
        let shader = ShaderModule::from_slang(&device, MINIMAL_COMPUTE_FOR_VK_VALIDATION).expect("shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

        let mut scheme = Scheme::new(&ctx);
        scheme.node("n0", &pipeline).dispatch(1, 1, 1);
        let submission = scheme.submit().expect("submit");

        let buf = test_alloc_buffer(&device, 256, BufferKind::Scattered, None, BufferFlags::empty());
        drop(buf);

        submission.wait_until_settled().expect("wait_until_settled");
    }

    #[cfg(feature = "vulkan")]
    fn vk_api_validation_two_device_teardown() {
        if std::env::var("GOLDY_SUBPROC").is_err() {
            if !vk_api_validation_active_backend_is_vulkan() {
                return;
            }
            run_in_subprocess_with_vk_validation("vk_api_validation_two_device_teardown");
            return;
        }

        let submit_minimal = |device: &Device| {
            let ctx = submission_context(device);
            let shader = ShaderModule::from_slang(device, MINIMAL_COMPUTE_FOR_VK_VALIDATION).expect("shader");
            let pipeline = ComputePipeline::new(device, &shader).expect("pipeline");
            let mut scheme = Scheme::new(&ctx);
            scheme.node("minimal", &pipeline).dispatch(1, 1, 1);
            scheme.submit().expect("submit");
        };

        let i1 = Instance::new().expect("i1");
        let d1 = i1
            .request_adapter(&RequestAdapterOptions::default())
            .expect("adapter d1")
            .request_device(&DeviceDescriptor::default())
            .expect("d1");
        let _b1 = test_alloc_buffer(&d1, 256, BufferKind::Scattered, None, BufferFlags::empty());
        submit_minimal(&d1);

        let i2 = Instance::new().expect("i2");
        let d2 = i2
            .request_adapter(&RequestAdapterOptions::default())
            .expect("adapter d2")
            .request_device(&DeviceDescriptor::default())
            .expect("d2");
        let _b2 = test_alloc_buffer(&d2, 256, BufferKind::Scattered, None, BufferFlags::empty());
        submit_minimal(&d2);

        drop(d1);
        drop(i1);
        drop(d2);
        drop(i2);
    }

    fn test_positive_mod_correctness(device: &Device) {
        const SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(1, 1, 1)]
    void cs_main(Scattered<float> out, ThreadId id) {
        out[0] = positive_mod(-1.0, 3.0);
        out[1] = positive_mod(-3.0, 3.0);
        out[2] = positive_mod(-0.5, 1.0);
        out[3] = positive_mod(2.5, 3.0);
        out[4] = positive_mod(0.0, 1.0);
        float2 r = positive_mod(float2(-1.0, -0.5), float2(3.0, 1.0));
        out[5] = r.x;
        out[6] = r.y;
    }
    "#;

        let ctx = submission_context(device);
        let shader = ShaderModule::from_slang(device, SHADER).expect("compile positive_mod shader");
        let pipeline = ComputePipeline::new(device, &shader).expect("create pipeline");
        let buf = test_alloc_buffer_with_data(device, &[0.0f32; 7], BufferKind::Scattered);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n0", &pipeline)
            .with_parcel(&buf, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, buf.whole())
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let loan = grant
            .claim(&mut frame)
            .expect("claim")
            .consume()
            .expect("grant consume");
        let result: &[f32] = bytemuck::cast_slice(&loan);

        let eps = 1e-5f32;
        let cases: &[(usize, f32, &str)] = &[
            (0, 2.0, "positive_mod(-1, 3)"),
            (1, 0.0, "positive_mod(-3, 3)"),
            (2, 0.5, "positive_mod(-0.5, 1)"),
            (3, 2.5, "positive_mod(2.5, 3)"),
            (4, 0.0, "positive_mod(0, 1)"),
            (5, 2.0, "float2 positive_mod x"),
            (6, 0.5, "float2 positive_mod y"),
        ];
        for &(i, expected, label) in cases {
            assert!(
                (result[i] - expected).abs() < eps,
                "{}: expected {}, got {}",
                label,
                expected,
                result[i]
            );
        }
    }

    fn test_billboard_math(device: &Device) {
        const SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(1, 1, 1)]
    void cs_main(Scattered<float> out, ThreadId id) {
        float4x4 m = float4x4(
            1, 0, 0, 0,
            5, 1, 0, 0,
            9, 0, 1, 0,
            0, 0, 0, 1
        );
        float3 r = modelview_right(m);
        out[0] = r.x;
        out[1] = r.y;
        out[2] = r.z;

        float3 off = billboard_cylindrical_offset(
            float3(1.0, 2.0, 3.0),
            float3(1.0, 0.0, 0.0),
            5.0
        );
        out[3] = off.x;
        out[4] = off.y;
        out[5] = off.z;

        float4x4 ident = float4x4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );
        float3 ident_right = modelview_right(ident);
        out[6] = ident_right.x;
        out[7] = ident_right.y;
        out[8] = ident_right.z;
    }
    "#;

        let ctx = submission_context(device);
        let shader = ShaderModule::from_slang(device, SHADER).expect("compile billboard shader");
        let pipeline = ComputePipeline::new(device, &shader).expect("create pipeline");
        let buf = test_alloc_buffer_with_data(device, &[0.0f32; 9], BufferKind::Scattered);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n0", &pipeline)
            .with_parcel(&buf, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, buf.whole())
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let loan = grant
            .claim(&mut frame)
            .expect("claim")
            .consume()
            .expect("grant consume");
        let result: &[f32] = bytemuck::cast_slice(&loan);

        let eps = 1e-5f32;
        let cases: &[(usize, f32, &str)] = &[
            (0, 1.0, "modelview_right col0.x"),
            (1, 5.0, "modelview_right col0.y"),
            (2, 9.0, "modelview_right col0.z"),
            (3, 6.0, "cylindrical offset x"),
            (4, 2.0, "cylindrical offset y (unchanged)"),
            (5, 3.0, "cylindrical offset z (unchanged)"),
            (6, 1.0, "identity right.x"),
            (7, 0.0, "identity right.y"),
            (8, 0.0, "identity right.z"),
        ];
        for &(i, expected, label) in cases {
            assert!(
                (result[i] - expected).abs() < eps,
                "{}: expected {}, got {}",
                label,
                expected,
                result[i]
            );
        }
    }

    fn test_heap_overflow_allocation(device: &Device) {
        const LARGE_COPY_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
        uint idx = id.x;
        if (idx >= 2097152) return;
        output[idx] = input[idx];
    }
    "#;

        let ctx = submission_context(device);
        let shader = ShaderModule::from_slang(device, LARGE_COPY_SHADER).expect("compile large copy shader");
        let pipeline = ComputePipeline::new(device, &shader).expect("create pipeline");

        const BUF_SIZE: u64 = 8 * 1024 * 1024;
        const NUM_BUFFERS: usize = 10;
        const ELEM_COUNT: usize = (BUF_SIZE / 4) as usize;

        let mut buffers = Vec::with_capacity(NUM_BUFFERS);
        for i in 0..NUM_BUFFERS {
            let data: Vec<u32> = if i == 0 {
                (0..ELEM_COUNT as u32).collect()
            } else {
                vec![0u32; ELEM_COUNT]
            };
            buffers.push(test_alloc_buffer_with_data(device, &data, BufferKind::Scattered));
        }

        let workgroups = (ELEM_COUNT as u32).div_ceil(64);
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n0", &pipeline)
            .with_parcel(&buffers[0], NodeAccess::Read)
            .with_parcel(&buffers[NUM_BUFFERS - 1], NodeAccess::Write)
            .dispatch(workgroups, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, buffers[NUM_BUFFERS - 1].whole())
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let loan = grant
            .claim(&mut frame)
            .expect("claim")
            .consume()
            .expect("grant consume");
        let result: &[u32] = bytemuck::cast_slice(&loan);
        for i in (0..ELEM_COUNT).step_by(1024) {
            assert_eq!(
                result[i], i as u32,
                "element {} expected {} got {} — overflow heap copy failed",
                i, i, result[i]
            );
        }
    }

    fn flush_deferred_deletions_reclaims_slots_after_gpu_idle(device: &Device) {
        let ctx = submission_context(device);
        let buf = test_alloc_buffer(device, 256, BufferKind::Scattered, None, BufferFlags::empty());
        let tv = scheme_submit_empty(&ctx);
        goldy::test_support::wait_until(&ctx, tv).expect("wait");

        drop(buf);
        ctx.flush_deferred_deletions();

        assert_eq!(
            ctx.deferred_deletion_pending_count(),
            0,
            "flush_deferred_deletions must reclaim all slots when GPU is idle"
        );
    }

    fn flush_deferred_deletions_respects_gpu_progress(device: &Device) {
        let ctx = submission_context(device);
        let tv = scheme_submit_empty(&ctx);

        let buf = test_alloc_buffer(device, 256, BufferKind::Scattered, None, BufferFlags::empty());
        drop(buf);

        ctx.flush_deferred_deletions();
        goldy::test_support::wait_until(&ctx, tv).expect("wait");
        ctx.flush_deferred_deletions();

        assert_eq!(
            ctx.deferred_deletion_pending_count(),
            0,
            "pending slots must be zero after wait_until + flush_deferred_deletions"
        );
    }

    fn flush_deferred_deletions_noop_on_idle_device(device: &Device) {
        let ctx = submission_context(device);
        ctx.flush_deferred_deletions();
        assert_eq!(ctx.deferred_deletion_pending_count(), 0);
    }

    pub fn run() {
        let device = make_device();

        let mut trials = Vec::new();
        macro_rules! trial {
            ($f:ident) => {{
                let device = device.clone();
                trials.push(libtest_mimic::Trial::test(stringify!($f), move || {
                    $f(&device);
                    Ok(())
                }));
            }};
        }
        macro_rules! trial0 {
            ($f:ident) => {
                trials.push(libtest_mimic::Trial::test(stringify!($f), || {
                    $f();
                    Ok(())
                }));
            };
        }

        trial!(test_compute_pipeline_creation);
        trial!(test_compute_pipeline_no_bindings);
        trial!(test_positive_mod_correctness);
        trial!(test_billboard_math);
        trial!(test_heap_overflow_allocation);
        trial!(flush_deferred_deletions_reclaims_slots_after_gpu_idle);
        trial!(flush_deferred_deletions_respects_gpu_progress);
        trial!(flush_deferred_deletions_noop_on_idle_device);
        #[cfg(feature = "vulkan")]
        trial0!(vk_api_validation_timeline_semaphore);
        #[cfg(feature = "vulkan")]
        trial0!(vk_api_validation_two_device_teardown);

        let mut args = libtest_mimic::Arguments::from_args();
        crate::submission::clamp_test_threads(&mut args, &device);
        let conclusion = libtest_mimic::run(&args, trials);

        drop(device);

        conclusion.exit_if_failed();
    }
}

fn main() {
    #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
    imp::run();
}
