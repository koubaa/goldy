#![allow(deprecated)]

#[path = "common/submission.rs"]
mod submission;
#[path = "common/upload.rs"]
mod upload;

#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
mod imp {
    use crate::submission::submission_context;
    use crate::upload::write_to_parcel;
    use goldy::{
        types::{BufferFlags, DispatchShape, TextureFlags, TextureFormat, TextureKind},
        BufferKind, ComputePipeline, Device, DeviceDescriptor, Instance, MemoryExchange, NodeAccess, Parcel,
        RequestAdapterOptions, RetainedPool, Sampler, Scheme, ShaderModule, StructuredBufferElement, Submission,
        WithdrawTransaction,
    };
    use std::sync::Arc;

    fn make_device() -> Device {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        if std::env::var("GOLDY_DX12_ALLOW_WARP").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")) {
            let instance = Instance::new().expect("Failed to create instance");
            if let Ok(adapter) = instance.request_adapter(&RequestAdapterOptions {
                power_preference: goldy::PowerPreference::None,
                force_fallback_adapter: true,
            }) {
                if let Ok(dev) = adapter.request_device(&DeviceDescriptor::default()) {
                    return dev;
                }
            }
        }

        let instance = Instance::new().expect("Failed to create instance");
        instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("adapter")
            .request_device(&DeviceDescriptor::default())
            .expect("device")
    }

    fn test_alloc_texture(
        device: &Device,
        data: &[u8],
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> goldy::Texture {
        RetainedPool::new(Arc::new(device.clone()))
            .acquire_texture(width, height, format, access, flags, Some(data))
            .expect("acquire_texture")
    }

    /// Read a scheme-tracked texture via MemoryExchange withdraw.
    fn read_texture_via_scheme_copy(ctx: &goldy::Context, texture: &goldy::Texture) -> Vec<u8> {
        let mut scheme = Scheme::new(ctx);
        let grant = MemoryExchange::new(ctx)
            .bind_withdraw(&mut scheme, texture)
            .expect("withdraw texture");
        let mut frame = scheme.submit().expect("submit");
        grant
            .claim(&mut frame)
            .expect("claim")
            .consume()
            .expect("consume")
            .to_vec()
    }

    fn read_grant_u32(grant: &WithdrawTransaction, submission: &mut Submission, count: usize) -> Vec<u32> {
        let loan = grant
            .claim(submission)
            .expect("claim")
            .consume()
            .expect("withdraw consume");
        assert_eq!(loan.len(), count * 4, "grant readback size");
        bytemuck::cast_slice(&loan).to_vec()
    }

    /// Read parcel bytes after an upload micro-scheme (grant-only verification scheme).
    fn read_uploaded_parcel_u32(ctx: &goldy::Context, parcel: &Parcel, count: usize) -> Vec<u32> {
        let mut scheme = Scheme::new(ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, parcel)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        read_grant_u32(&grant, &mut frame, count)
    }

    fn dispatch_u32_write_and_read(ctx: &goldy::Context, shader_src: &str, out: &Parcel, count: usize) -> Vec<u32> {
        let device = ctx.device();
        let shader = ShaderModule::from_slang(device, shader_src).expect("compile shader");
        let pipeline = ComputePipeline::new(device, &shader).expect("create pipeline");

        let mut scheme = Scheme::new(ctx);
        scheme
            .node("n0", &pipeline)
            .with_parcel(out, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, out)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        read_grant_u32(&grant, &mut frame, count)
    }

    fn write_zeros_to_parcel(ctx: &goldy::Context, parcel: &Parcel, byte_len: usize) {
        // Upload micro-scheme: separate from worker Scheme.
        write_to_parcel(ctx, parcel, &vec![0u8; byte_len]).expect("write_to_parcel zeros");
    }

    const DOUBLE_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
        output[id.x] = input[id.x] * 2;
    }
    "#;

    const IN_PLACE_DOUBLE_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        data[id.x] = data[id.x] * 2;
    }
    "#;

    const ADD_TEN_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        data[id.x] = data[id.x] + 10;
    }
    "#;

    const FILL_42_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        data[id.x] = 42;
    }
    "#;

    const FILL_99_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        data[id.x] = 99;
    }
    "#;

    const SUM_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> a, Scattered<uint> b, Scattered<uint> out, ThreadId id) {
        out[id.x] = a[id.x] + b[id.x];
    }
    "#;

    const COPY_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
        output[id.x] = input[id.x];
    }
    "#;

    const INCREMENT_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        data[id.x] = data[id.x] + 1;
    }
    "#;

    const FILL_SRC_WITH_INDEX_SHADER: &str = r#"
    import goldy_exp;
    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        data[id.x] = id.x;
    }
    "#;

    const WRITE_IOTA_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        if (id.x < 64) data[id.x] = id.x + 1;
    }
    "#;

    const SIX_SLOT_SUM_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> a, Scattered<uint> b, Scattered<uint> c,
                 Scattered<uint> d, Scattered<uint> e, Scattered<uint> out,
                 ThreadId id) {
        uint idx = id.x;
        if (idx >= 16) return;
        out[idx] = a[idx] + b[idx] + c[idx] + d[idx] + e[idx];
    }
    "#;

    const MINIMAL_SHADER: &str = r#"
    [shader("compute")]
    [numthreads(1, 1, 1)]
    void cs_main(uint3 id : SV_DispatchThreadID) {
    }
    "#;

    // ---------------------------------------------------------------------------
    // Migrated from task_graph_integration.rs
    // ---------------------------------------------------------------------------

    fn scheme_graph_linear_chain(device: &Device) {
        let ctx = submission_context(&device);

        let double_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap()).unwrap();
        let add_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap()).unwrap();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let src = pool
            .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
            .unwrap();
        let dst = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("double", &double_pipe)
            .with_parcel(&src, NodeAccess::Read)
            .with_parcel(&dst, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("add_ten", &add_pipe)
            .with_parcel(&dst, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &dst)
            .expect("withdraw");
        scheme.submit().unwrap();
        let mut frame = scheme.submit().unwrap();
        assert_eq!(scheme.replay_stats().records, 1, "linear chain records once");
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            1,
            "second submit must resubmit without re-record"
        );

        let result = read_grant_u32(&grant, &mut frame, 64);
        for (i, &val) in result.iter().enumerate() {
            let expected = (i as u32) * 2 + 10;
            assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
        }
    }

    fn scheme_graph_independent_dispatches(device: &Device) {
        let ctx = submission_context(&device);

        let pipe_42 =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap()).unwrap();
        let pipe_99 =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, FILL_99_SHADER).unwrap()).unwrap();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf_a = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        let buf_b = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("fill_a", &pipe_42)
            .with_parcel(&buf_a, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("fill_b", &pipe_99)
            .with_parcel(&buf_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let grant_a = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buf_a)
            .expect("withdraw");
        let grant_b = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buf_b)
            .expect("withdraw");
        let mut frame = scheme.submit().unwrap();

        for &v in &read_grant_u32(&grant_a, &mut frame, 64) {
            assert_eq!(v, 42);
        }
        for &v in &read_grant_u32(&grant_b, &mut frame, 64) {
            assert_eq!(v, 99);
        }
    }

    fn scheme_graph_diamond_dependency(device: &Device) {
        let ctx = submission_context(&device);

        let fill_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, FILL_SRC_WITH_INDEX_SHADER).unwrap(),
        )
        .unwrap();
        let double_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap()).unwrap();
        let sum_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, SUM_SHADER).unwrap()).unwrap();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let src = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        let y = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        let z = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        let out = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("fill_src", &fill_pipe)
            .with_parcel(&src, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("double_to_y", &double_pipe)
            .with_parcel(&src, NodeAccess::Read)
            .with_parcel(&y, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("double_to_z", &double_pipe)
            .with_parcel(&src, NodeAccess::Read)
            .with_parcel(&z, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("sum_yz", &sum_pipe)
            .with_parcel(&y, NodeAccess::Read)
            .with_parcel(&z, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("withdraw");
        let mut frame = scheme.submit().unwrap();
        assert_eq!(scheme.replay_stats().records, 1, "diamond records once");

        let result = read_grant_u32(&grant, &mut frame, 64);
        for (i, &val) in result.iter().enumerate() {
            let expected = (i as u32) * 4;
            assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
        }
    }

    fn scheme_graph_fill_readback(device: &Device) {
        let ctx = submission_context(&device);

        let pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap()).unwrap();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("fill", &pipe)
            .with_parcel(&buf, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buf)
            .expect("withdraw");
        let mut frame = scheme.submit().unwrap();

        for &v in &read_grant_u32(&grant, &mut frame, 64) {
            assert_eq!(v, 42);
        }
    }

    fn scheme_zeros_then_dispatch_reads_zeros(device: &Device) {
        let ctx = submission_context(&device);

        let copy_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer_with_data(&(1..=64).collect::<Vec<u32>>(), BufferKind::Scattered)
            .unwrap();
        let out = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        write_zeros_to_parcel(&ctx, &buf, 64 * 4);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("copy", &copy_pipe)
            .with_parcel(&buf, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("withdraw");
        let mut frame = scheme.submit().unwrap();

        for (i, &val) in read_grant_u32(&grant, &mut frame, 64).iter().enumerate() {
            assert_eq!(val, 0, "element {i}: expected 0 after zero write, got {val}");
        }
    }

    fn scheme_write_then_dispatch_reads_uploaded_data(device: &Device) {
        let ctx = submission_context(&device);

        let copy_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        let out = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        let known_data: Vec<u32> = (100..164).collect();
        write_to_parcel(&ctx, &buf, bytemuck::cast_slice(&known_data)).expect("write_to_parcel");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("copy", &copy_pipe)
            .with_parcel(&buf, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("withdraw");
        let mut frame = scheme.submit().unwrap();

        for (i, &val) in read_grant_u32(&grant, &mut frame, 64).iter().enumerate() {
            assert_eq!(val, known_data[i], "element {i}");
        }
    }

    fn scheme_stress_zeros_then_dispatch_large(device: &Device) {
        let ctx = submission_context(&device);

        let copy_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

        const N: usize = 16384;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer_with_data(&(1..=N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
            .unwrap();
        let out = pool
            .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        write_zeros_to_parcel(&ctx, &buf, N * 4);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("copy", &copy_pipe)
            .with_parcel(&buf, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .dispatch((N / 64) as u32, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("withdraw");
        let mut frame = scheme.submit().unwrap();

        let nonzero_count = read_grant_u32(&grant, &mut frame, N)
            .iter()
            .filter(|&&v| v != 0)
            .count();
        assert_eq!(nonzero_count, 0, "expected all zeros after zero write");
    }

    fn scheme_stress_many_zero_writes_many_dispatches(device: &Device) {
        let ctx = submission_context(&device);

        let copy_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

        const N: usize = 1024;
        const NUM_BUFS: usize = 8;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));

        let mut srcs = Vec::new();
        let mut outs = Vec::new();
        for _ in 0..NUM_BUFS {
            srcs.push(
                pool.acquire_buffer_with_data(&(1..=N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
                    .unwrap(),
            );
            outs.push(
                pool.acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
                    .unwrap(),
            );
        }

        for src in &srcs {
            write_zeros_to_parcel(&ctx, src, N * 4);
        }

        let mut scheme = Scheme::new(&ctx);
        for (src, out) in srcs.iter().zip(outs.iter()) {
            scheme
                .node("copy", &copy_pipe)
                .with_parcel(src, NodeAccess::Read)
                .with_parcel(out, NodeAccess::Write)
                .dispatch((N / 64) as u32, 1, 1);
        }

        let mut grants = Vec::new();
        for out in &outs {
            grants.push(
                MemoryExchange::new(scheme.context())
                    .bind_withdraw(&mut scheme, out)
                    .expect("withdraw"),
            );
        }
        let mut frame = scheme.submit().unwrap();

        for (i, grant) in grants.iter().enumerate() {
            let nonzero_count = read_grant_u32(grant, &mut frame, N).iter().filter(|&&v| v != 0).count();
            assert_eq!(nonzero_count, 0, "buffer {i}: expected all zeros after zero write");
        }
    }

    fn scheme_stress_write_then_dispatch_chain(device: &Device) {
        let ctx = submission_context(&device);

        let copy_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

        const N: usize = 1024;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer_with_data(&(1..=N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
            .unwrap();
        let out = pool
            .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        let known_data: Vec<u32> = (0..N as u32).map(|i| i * 7 + 42).collect();
        write_to_parcel(&ctx, &buf, bytemuck::cast_slice(&known_data)).expect("write_to_parcel");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("copy", &copy_pipe)
            .with_parcel(&buf, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .dispatch((N / 64) as u32, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("withdraw");
        let mut frame = scheme.submit().unwrap();

        for (i, &val) in read_grant_u32(&grant, &mut frame, N).iter().enumerate() {
            assert_eq!(val, known_data[i], "element {i}");
        }
    }

    fn scheme_stress_two_phase_submission(device: &Device) {
        let ctx = submission_context(&device);

        let double_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap()).unwrap();
        let add_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap()).unwrap();

        const N: usize = 4096;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer_with_data(&(0..N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
            .unwrap();
        let tmp = pool
            .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        {
            let mut scheme = Scheme::new(&ctx);
            scheme
                .node("double", &double_pipe)
                .with_parcel(&buf, NodeAccess::Read)
                .with_parcel(&tmp, NodeAccess::Write)
                .dispatch((N / 64) as u32, 1, 1);
            scheme.submit().unwrap();
        }

        let result = {
            let mut scheme = Scheme::new(&ctx);
            scheme
                .node("add_ten", &add_pipe)
                .with_parcel(&tmp, NodeAccess::ReadWrite)
                .dispatch((N / 64) as u32, 1, 1);
            let grant = MemoryExchange::new(scheme.context())
                .bind_withdraw(&mut scheme, &tmp)
                .expect("withdraw");
            let mut frame = scheme.submit().unwrap();
            read_grant_u32(&grant, &mut frame, N)
        };

        for (i, &val) in result.iter().enumerate() {
            let expected = (i as u32) * 2 + 10;
            assert_eq!(val, expected, "element {i}");
        }
    }

    fn scheme_stress_rapid_submissions(device: &Device) {
        let ctx = submission_context(&device);

        let add_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap()).unwrap();

        const N: usize = 256;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer_with_data(&vec![0u32; N], BufferKind::Scattered)
            .unwrap();

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("add_ten", &add_pipe)
            .with_parcel(&buf, NodeAccess::ReadWrite)
            .dispatch((N / 64) as u32, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buf)
            .expect("withdraw");
        const ROUNDS: u32 = 20;
        let mut last_frame = None;
        for _ in 0..ROUNDS {
            last_frame = Some(scheme.submit().unwrap());
        }
        let mut frame = last_frame.expect("submit");

        assert_eq!(scheme.replay_stats().records, 1, "rapid submissions record once");
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            u64::from(ROUNDS) - 1,
            "remaining submits are retention hits"
        );

        let expected = ROUNDS * 10;
        for (i, &val) in read_grant_u32(&grant, &mut frame, N).iter().enumerate() {
            assert_eq!(val, expected, "element {i}");
        }
    }

    // ---------------------------------------------------------------------------
    // Migrated from compute_integration.rs
    // ---------------------------------------------------------------------------

    fn scheme_compute_dispatch_empty(device: &Device) {
        let ctx = submission_context(&device);

        let pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, MINIMAL_SHADER).expect("shader"),
        )
        .expect("pipeline");

        let mut scheme = Scheme::new(&ctx);
        scheme.node("n0", &pipe).dispatch(1, 1, 1);
        scheme.submit().expect("submit");
    }

    fn scheme_compute_write_and_readback(device: &Device) {
        let ctx = submission_context(&device);

        let pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, IN_PLACE_DOUBLE_SHADER).expect("shader"),
        )
        .expect("pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buffer = pool
            .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("buffer");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("double", &pipe)
            .with_parcel(&buffer, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buffer)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        for (i, &val) in read_grant_u32(&grant, &mut frame, 64).iter().enumerate() {
            assert_eq!(val, (i as u32) * 2, "element {i}");
        }
    }

    fn scheme_compute_with_uav_parcel(device: &Device) {
        let ctx = submission_context(&device);

        let pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, IN_PLACE_DOUBLE_SHADER).expect("shader"),
        )
        .expect("pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buffer = pool
            .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("buffer");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("double", &pipe)
            .with_parcel(&buffer, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buffer)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        for (i, &val) in read_grant_u32(&grant, &mut frame, 64).iter().enumerate() {
            assert_eq!(val, (i as u32) * 2, "element {i}");
        }
    }

    fn scheme_compute_with_srv_and_uav_parcels(device: &Device) {
        let ctx = submission_context(&device);

        let pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
        )
        .expect("pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let input = pool
            .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("input");
        let mut output = pool
            .acquire_buffer_with_data(&vec![0u32; 64], BufferKind::Scattered)
            .expect("output");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("copy", &pipe)
            .with_parcel(&input, NodeAccess::Read)
            .with_parcel(&output, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &output)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let output_vals = read_grant_u32(&grant, &mut frame, 64);
        let input_vals = read_uploaded_parcel_u32(&ctx, &input, 64);
        assert_eq!(output_vals, input_vals, "copy must reproduce input in output");
    }

    fn scheme_parcel_write_zeros_full(device: &Device) {
        let ctx = submission_context(&device);

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let parcel = pool
            .acquire_buffer_with_data(&vec![0xDEAD_BEEFu32; 64], BufferKind::Scattered)
            .expect("parcel");

        let token = write_to_parcel(&ctx, &parcel, &vec![0u8; 64 * 4]).expect("write_to_parcel zeros");
        token.wait_until_settled().expect("wait");

        for (i, &val) in read_uploaded_parcel_u32(&ctx, &parcel, 64).iter().enumerate() {
            assert_eq!(val, 0, "element {i} should be 0 after zero write");
        }
    }

    fn scheme_parcel_write_zeros_partial(device: &Device) {
        let ctx = submission_context(&device);

        const SENTINEL: u32 = 0xDEAD_BEEF;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let parcel = pool
            .acquire_buffer_with_data(&vec![SENTINEL; 64], BufferKind::Scattered)
            .expect("parcel");

        let mut data = vec![SENTINEL; 64];
        for slot in data.iter_mut().take(64).skip(16).take(16) {
            *slot = 0;
        }
        let token = write_to_parcel(&ctx, &parcel, bytemuck::cast_slice(&data)).expect("write_to_parcel");
        token.wait_until_settled().expect("wait");

        for (i, &val) in read_uploaded_parcel_u32(&ctx, &parcel, 64).iter().enumerate() {
            let expected = if (16..32).contains(&i) { 0 } else { SENTINEL };
            assert_eq!(val, expected, "element {i}");
        }
    }

    fn scheme_parcel_write_zeros_to_end(device: &Device) {
        let ctx = submission_context(&device);

        const SENTINEL: u32 = 0xCAFE_BABE;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let parcel = pool
            .acquire_buffer_with_data(&vec![SENTINEL; 64], BufferKind::Scattered)
            .expect("parcel");

        let mut data = vec![SENTINEL; 64];
        for slot in data.iter_mut().skip(32) {
            *slot = 0;
        }
        let token = write_to_parcel(&ctx, &parcel, bytemuck::cast_slice(&data)).expect("write_to_parcel");
        token.wait_until_settled().expect("wait");

        for (i, &val) in read_uploaded_parcel_u32(&ctx, &parcel, 64).iter().enumerate() {
            let expected = if i < 32 { SENTINEL } else { 0 };
            assert_eq!(val, expected, "element {i}");
        }
    }

    fn scheme_zeros_before_copy_dispatch(device: &Device) {
        let ctx = submission_context(&device);

        let pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
        )
        .expect("pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let input = pool
            .acquire_buffer_with_data(&vec![0xDEAD_BEEFu32; 64], BufferKind::Scattered)
            .expect("input");
        let mut output = pool
            .acquire_buffer_with_data(&vec![0xFFFF_FFFFu32; 64], BufferKind::Scattered)
            .expect("output");

        write_zeros_to_parcel(&ctx, &input, 64 * 4);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("copy", &pipe)
            .with_parcel(&input, NodeAccess::Read)
            .with_parcel(&output, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &output)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        for (i, &val) in read_grant_u32(&grant, &mut frame, 64).iter().enumerate() {
            assert_eq!(val, 0, "output[{i}] should be 0 (copied from zeroed input)");
        }
    }

    /// GPU ordering: copy scheme writes 42s → `write_to_parcel` zeros output → increment scheme.
    /// Correct result is 1 (0 + 1). Cross-scheme serialization replaces in-graph `clear_buffer`.
    fn scheme_write_to_parcel_zeros_between_submissions(device: &Device) {
        let ctx = submission_context(&device);

        let copy_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
        )
        .expect("copy pipeline");
        let inc_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, INCREMENT_SHADER).expect("shader"),
        )
        .expect("inc pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let input = pool
            .acquire_buffer_with_data(&vec![42u32; 64], BufferKind::Scattered)
            .expect("input");
        let mut output = pool
            .acquire_buffer_with_data(&vec![0u32; 64], BufferKind::Scattered)
            .expect("output");

        {
            let mut copy_scheme = Scheme::new(&ctx);
            copy_scheme
                .node("copy", &copy_pipe)
                .with_parcel(&input, NodeAccess::Read)
                .with_parcel(&output, NodeAccess::Write)
                .dispatch(1, 1, 1);
            copy_scheme.submit().expect("copy submit 0");
            copy_scheme.submit().expect("copy submit 1");
            assert_eq!(copy_scheme.replay_stats().records, 1, "copy scheme records once");
            #[cfg(not(feature = "metal"))]
            assert_eq!(
                copy_scheme.replay_stats().resubmit_hits,
                1,
                "second copy submit must resubmit without re-record"
            );
        }

        // No wait here: zero write must serialize after copy via queue order alone.
        write_zeros_to_parcel(&ctx, &output, 64 * 4);

        let result = {
            let mut inc_scheme = Scheme::new(&ctx);
            inc_scheme
                .node("inc", &inc_pipe)
                .with_parcel(&output, NodeAccess::ReadWrite)
                .dispatch(1, 1, 1);
            let grant = MemoryExchange::new(inc_scheme.context())
                .bind_withdraw(&mut inc_scheme, &output)
                .expect("withdraw");
            let mut frame = inc_scheme.submit().expect("inc submit");
            read_grant_u32(&grant, &mut frame, 64)
        };

        for (i, &val) in result.iter().enumerate() {
            assert_eq!(
                val, 1,
                "output[{i}]: expected 1 (write_to_parcel zeroed before increment), got {val}"
            );
        }
    }

    /// Cross-scheme ordering: an upload micro-scheme may return its [`Submission`] without
    /// waiting; the next worker [`Scheme::submit`] on the same context still sees the upload.
    fn scheme_upload_frame_unwaited_serializes_before_worker_submit(device: &Device) {
        let ctx = submission_context(&device);

        let copy_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
        )
        .expect("pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let input = pool
            .acquire_buffer_with_data(&vec![0xDEAD_BEEFu32; 64], BufferKind::Scattered)
            .expect("input");
        let mut output = pool
            .acquire_buffer_with_data(&vec![0u32; 64], BufferKind::Scattered)
            .expect("output");

        const PATTERN: u32 = 0xCAFE_BABE;
        let upload_data = vec![PATTERN; 64];
        let _upload_frame = write_to_parcel(&ctx, &input, bytemuck::cast_slice(&upload_data)).expect("write_to_parcel");
        // Deliberately no wait on upload frame — queue order must serialize before worker submit.

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("copy", &copy_pipe)
            .with_parcel(&input, NodeAccess::Read)
            .with_parcel(&output, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &output)
            .expect("withdraw");
        let mut worker_frame = scheme.submit().expect("worker submit");

        for (i, &val) in read_grant_u32(&grant, &mut worker_frame, 64).iter().enumerate() {
            assert_eq!(
                val, PATTERN,
                "output[{i}]: upload must be visible without waiting on upload frame"
            );
        }
    }

    fn scheme_compute_many_resource_slots(device: &Device) {
        let ctx = submission_context(&device);

        let pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, SIX_SLOT_SUM_SHADER).expect("shader"),
        )
        .expect("pipeline");

        const N: usize = 16;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let a = pool
            .acquire_buffer_with_data(&[1u32; N], BufferKind::Scattered)
            .expect("a");
        let b = pool
            .acquire_buffer_with_data(&[2u32; N], BufferKind::Scattered)
            .expect("b");
        let c = pool
            .acquire_buffer_with_data(&[3u32; N], BufferKind::Scattered)
            .expect("c");
        let d = pool
            .acquire_buffer_with_data(&[4u32; N], BufferKind::Scattered)
            .expect("d");
        let e = pool
            .acquire_buffer_with_data(&[5u32; N], BufferKind::Scattered)
            .expect("e");
        let out = pool
            .acquire_buffer_with_data(&[0u32; N], BufferKind::Scattered)
            .expect("out");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("sum", &pipe)
            .with_parcel(&a, NodeAccess::Read)
            .with_parcel(&b, NodeAccess::Read)
            .with_parcel(&c, NodeAccess::Read)
            .with_parcel(&d, NodeAccess::Read)
            .with_parcel(&e, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        for (i, &val) in read_grant_u32(&grant, &mut frame, N).iter().enumerate() {
            assert_eq!(val, 15, "out[{i}] expected 15, got {val}");
        }
    }

    fn scheme_regular_buffer_write_then_copy(device: &Device) {
        let ctx = submission_context(&device);

        let write_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, WRITE_IOTA_SHADER).expect("shader"),
        )
        .expect("write pipeline");
        let copy_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
        )
        .expect("copy pipeline");

        const N: usize = 64;
        let byte_size = N * 4;

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let scratch = pool
            .acquire_buffer(
                byte_size as u64,
                BufferKind::Scattered,
                None,
                BufferFlags::empty(),
                None,
            )
            .expect("scratch");
        let mut output = pool
            .acquire_buffer(
                byte_size as u64,
                BufferKind::Scattered,
                None,
                BufferFlags::empty(),
                None,
            )
            .expect("output");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_iota", &write_pipe)
            .with_parcel(&scratch, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("copy_out", &copy_pipe)
            .with_parcel(&scratch, NodeAccess::Read)
            .with_parcel(&output, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &output)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let expected: Vec<u32> = (1..=N as u32).collect();
        assert_eq!(read_grant_u32(&grant, &mut frame, N), expected);
    }

    fn scheme_transient_buffer_write_then_copy(device: &Device) {
        let ctx = submission_context(&device);

        let write_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, WRITE_IOTA_SHADER).expect("shader"),
        )
        .expect("write pipeline");
        let copy_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
        )
        .expect("copy pipeline");

        const N: usize = 64;
        let byte_size = (N * 4) as u64;

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let mut output = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("output");

        let mut scheme = Scheme::new(&ctx);
        let scratch = scheme.lease_buffer(byte_size).expect("lease scratch");

        scheme
            .node("write_iota", &write_pipe)
            .with_parcel(&scratch, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("copy_out", &copy_pipe)
            .with_parcel(&scratch, NodeAccess::Read)
            .with_parcel(&output, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &output)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let expected: Vec<u32> = (1..=N as u32).collect();
        assert_eq!(read_grant_u32(&grant, &mut frame, N), expected);
    }

    const WRITE_SCALE_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        if (id.x < 64) data[id.x] = (id.x + 1u) * 100u;
    }
    "#;

    /// Scheme-world analog of `test_transient_buffer_aliased_disjoint_waves` (TaskGraph).
    ///
    /// In the TaskGraph model, two transients with disjoint wave lifetimes are aliased onto the
    /// same heap offset within a single submission via graph coloring. Scheme deliberately does
    /// not support within-submission aliasing; instead two separate schemes with a buffer lease
    /// each reuse the same physical backing *across* submissions once the prior epoch retires
    /// (pool high-water recycling). This test verifies that cross-submission correctness:
    /// each scheme observes only its own writes and the outputs are independent.
    fn scheme_transient_buffer_recycling(device: &Device) {
        let ctx = submission_context(&device);

        let iota_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, WRITE_IOTA_SHADER).expect("shader"),
        )
        .expect("iota pipeline");
        let scale_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, WRITE_SCALE_SHADER).expect("shader"),
        )
        .expect("scale pipeline");
        let copy_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
        )
        .expect("copy pipeline");

        const N: usize = 64;
        let byte_size = (N * 4) as u64;

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let output_a = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("output_a");
        let output_b = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("output_b");

        let expected_iota: Vec<u32> = (1..=N as u32).collect();
        let expected_scale: Vec<u32> = (1..=N as u32).map(|i| i * 100).collect();

        let alloc_count_before = ctx.transient_buffer_alloc_count();

        {
            let mut scheme = Scheme::new(&ctx);
            let scratch = scheme.lease_buffer(byte_size).expect("lease scratch_a");
            scheme
                .node("write_iota", &iota_pipe)
                .with_parcel(&scratch, NodeAccess::Write)
                .dispatch(1, 1, 1);
            scheme
                .node("copy_a", &copy_pipe)
                .with_parcel(&scratch, NodeAccess::Read)
                .with_parcel(&output_a, NodeAccess::Write)
                .dispatch(1, 1, 1);

            let grant = MemoryExchange::new(scheme.context())
                .bind_withdraw(&mut scheme, &output_a)
                .expect("grant_a");
            let mut frame = scheme.submit().expect("submit_a");
            assert_eq!(read_grant_u32(&grant, &mut frame, N), expected_iota);
        }
        // scheme dropped here — backing parcel returned to pool with ready epoch

        assert_eq!(
            ctx.transient_buffer_alloc_count(),
            alloc_count_before + 1,
            "first lease must have triggered one fresh allocation"
        );
        assert!(
            ctx.transient_outstanding_bytes().buffer == 0,
            "outstanding drops to zero once scheme releases the lease"
        );

        {
            let mut scheme = Scheme::new(&ctx);
            let scratch = scheme.lease_buffer(byte_size).expect("lease scratch_b");
            assert_eq!(
                ctx.transient_buffer_alloc_count(),
                alloc_count_before + 1,
                "second lease must reuse the retired bin entry — alloc count stays flat"
            );
            assert!(
                ctx.transient_outstanding_bytes().buffer >= byte_size,
                "reused backing is counted as outstanding again"
            );
            scheme
                .node("write_scale", &scale_pipe)
                .with_parcel(&scratch, NodeAccess::Write)
                .dispatch(1, 1, 1);
            scheme
                .node("copy_b", &copy_pipe)
                .with_parcel(&scratch, NodeAccess::Read)
                .with_parcel(&output_b, NodeAccess::Write)
                .dispatch(1, 1, 1);

            let grant = MemoryExchange::new(scheme.context())
                .bind_withdraw(&mut scheme, &output_b)
                .expect("grant_b");
            let mut frame = scheme.submit().expect("submit_b");
            assert_eq!(read_grant_u32(&grant, &mut frame, N), expected_scale);
        }
    }

    // ---------------------------------------------------------------------------
    // Duplicated from compute_integration.rs
    // ---------------------------------------------------------------------------

    const PARTICLE_SHADER: &str = r#"
    import goldy_exp;

    struct Particle {
        float2 position;
        float2 velocity;
    };

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<Particle> particles, ThreadId id) {
        uint idx = id.x;
        if (idx >= 4) return;
        Particle p = particles[idx];
        p.position += float2(0.01, 0.01);
        particles[idx] = p;
    }
    "#;

    const TYPED_PAIR_SHADER: &str = r#"
    import goldy_exp;

    struct Pair { uint a; uint b; };

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(BufRO<Pair> input, Scattered<Pair> output, ThreadId id) {
        uint idx = id.x;
        if (idx >= 8) return;
        Pair p = input[idx];
        output[idx].a = p.a + p.b;
        output[idx].b = p.a * p.b;
    }
    "#;

    const WRITE_TEXTURE_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(8, 8, 1)]
    void cs_main(DirectSpatial<float4> output, ThreadId id) {
        uint2 dims;
        output.GetDimensions(dims.x, dims.y);
        if (id.x < dims.x && id.y < dims.y) {
            output[int2(id.x, id.y)] = float4(1.0, 0.0, 0.0, 1.0);
        }
    }
    "#;

    const WAVE_SCAN_64_UNIFORM: &str = r#"
    import goldy_exp;
    groupshared uint sh_scratch[32];
    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
        uint ix  = local_id.x;
        uint val = 1u;
        uint lc  = WaveGetLaneCount();
        uint nw  = 64 / lc;
        uint wave_ix = ix / lc;
        uint inclusive = WavePrefixSum(val) + val;
        uint total     = WaveActiveSum(val);
        if (WaveIsFirstLane())
            sh_scratch[wave_ix] = total;
        GroupMemoryBarrierWithGroupSync();
        if (ix == 0) {
            uint run = 0;
            for (uint i = 0; i < nw; i++) {
                uint s = sh_scratch[i]; sh_scratch[i] = run; run += s;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        OUT[ix] = sh_scratch[wave_ix] + inclusive;
    }
    "#;

    const WAVE_SCAN_64_RAMP: &str = r#"
    import goldy_exp;
    groupshared uint sh_scratch[32];
    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
        uint ix  = local_id.x;
        uint val = ix + 1u;
        uint lc  = WaveGetLaneCount();
        uint nw  = 64 / lc;
        uint wave_ix = ix / lc;
        uint inclusive = WavePrefixSum(val) + val;
        uint total     = WaveActiveSum(val);
        if (WaveIsFirstLane())
            sh_scratch[wave_ix] = total;
        GroupMemoryBarrierWithGroupSync();
        if (ix == 0) {
            uint run = 0;
            for (uint i = 0; i < nw; i++) {
                uint s = sh_scratch[i]; sh_scratch[i] = run; run += s;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        OUT[ix] = sh_scratch[wave_ix] + inclusive;
    }
    "#;

    const WAVE_SCAN_256_UNIFORM: &str = r#"
    import goldy_exp;
    groupshared uint sh_scratch[64];
    [goldy_compute]
    [numthreads(256, 1, 1)]
    void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
        uint ix  = local_id.x;
        uint val = 1u;
        uint lc  = WaveGetLaneCount();
        uint nw  = 256 / lc;
        uint wave_ix = ix / lc;
        uint inclusive = WavePrefixSum(val) + val;
        uint total     = WaveActiveSum(val);
        if (WaveIsFirstLane())
            sh_scratch[wave_ix] = total;
        GroupMemoryBarrierWithGroupSync();
        if (ix == 0) {
            uint run = 0;
            for (uint i = 0; i < nw; i++) {
                uint s = sh_scratch[i]; sh_scratch[i] = run; run += s;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        OUT[ix] = sh_scratch[wave_ix] + inclusive;
    }
    "#;

    const REDUCE_64_UNIFORM: &str = r#"
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

    const INCLUSIVE_SCAN_64_UNIFORM: &str = r#"
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

    const BROADCAST_64: &str = r#"
    import goldy_exp;
    groupshared uint sh_slot[1];
    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
        uint ix = local_id.x;
        if (ix == 0) sh_slot[0] = 42u;
        GroupMemoryBarrierWithGroupSync();
        OUT[ix] = sh_slot[0];
    }
    "#;

    const UPPER_BOUND_64: &str = r#"
    import goldy_exp;
    groupshared uint sh_ps[64];
    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
        uint ix = local_id.x;
        sh_ps[ix] = ix + 1u;
        GroupMemoryBarrierWithGroupSync();
        OUT[ix] = workgroup_upper_bound<6>(ix, sh_ps);
    }
    "#;

    const DUAL_VIEW_WRITE_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(4, 4, 1)]
    void cs_main(DirectSpatial<float4> dst, ThreadId id) {
        uint x = id.x;
        uint y = id.y;
        dst[uint2(x, y)] = float4(float(x) / 255.0, float(y) / 255.0, 0.0, 1.0);
    }
    "#;

    const DUAL_VIEW_READ_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(4, 4, 1)]
    void cs_main(Interpolated<float4> src, Filter smp, Scattered<uint> out, ThreadId id) {
        uint x = id.x;
        uint y = id.y;
        float2 uv = (float2(x, y) + 0.5) / float2(4.0, 4.0);
        float4 v = src.Sample(smp, uv);
        uint r = uint(v.x * 255.0 + 0.5);
        uint g = uint(v.y * 255.0 + 0.5);
        uint b = uint(v.z * 255.0 + 0.5);
        uint a = uint(v.w * 255.0 + 0.5);
        out[y * 4 + x] = r | (g << 8) | (b << 16) | (a << 24);
    }
    "#;

    fn scheme_compute_with_struct_buffer(device: &Device) {
        let ctx = submission_context(&device);

        let shader = ShaderModule::from_slang(&device, PARTICLE_SHADER).expect("compile shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Particle {
            position: [f32; 2],
            velocity: [f32; 2],
        }
        impl StructuredBufferElement for Particle {}

        let particles = vec![
            Particle {
                position: [0.0, 0.0],
                velocity: [0.1, 0.0],
            };
            4
        ];

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buffer = pool
            .acquire_buffer_with_data(&particles, BufferKind::Scattered)
            .expect("buffer");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n0", &pipeline)
            .with_parcel(&buffer, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        scheme.submit().expect("submit with struct buffer");
    }

    const DOUBLE_SHADER_COHERENT: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        data[id.x] = data[id.x] * 2;
    }
    "#;

    /// [`BufferFlags::CPU_READABLE`] is a medium hint — grant readback uses the same path as any buffer parcel.
    fn scheme_cpu_readable_compute_write_and_read(device: &Device) {
        let ctx = submission_context(&device);

        let shader = ShaderModule::from_slang(&device, DOUBLE_SHADER_COHERENT).expect("compile shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

        const N: usize = 64;
        let initial: Vec<u32> = (0..N as u32).collect();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buffer = pool
            .acquire_buffer_with_data_and_flags(&initial, BufferKind::Scattered, BufferFlags::CPU_READABLE)
            .expect("buffer");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("double", &pipeline)
            .with_parcel(&buffer, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buffer)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        for (i, &val) in read_grant_u32(&grant, &mut frame, N).iter().enumerate() {
            assert_eq!(val, (i as u32) * 2, "element {i}: expected {} got {val}", i * 2);
        }
    }

    /// CPU-visible data uploaded via [`write_to_parcel`] round-trips through grant readback.
    fn scheme_cpu_readable_write_to_parcel_roundtrip(device: &Device) {
        let ctx = submission_context(&device);

        const N: usize = 16;
        let initial: Vec<u32> = vec![0xABCD_1234u32; N];
        let new_values: Vec<u32> = (100..100 + N as u32).collect();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let parcel = pool
            .acquire_buffer_with_data_and_flags(&initial, BufferKind::Scattered, BufferFlags::CPU_READABLE)
            .expect("parcel");

        write_to_parcel(&ctx, &parcel, bytemuck::cast_slice(&new_values)).expect("write_to_parcel");

        for (i, &val) in read_uploaded_parcel_u32(&ctx, &parcel, N).iter().enumerate() {
            assert_eq!(val, 100 + i as u32, "element {i}: expected {} got {val}", 100 + i);
        }
    }

    fn scheme_scattered_typed_variable_assignment(device: &Device) {
        let ctx = submission_context(&device);

        let shader = ShaderModule::from_slang(&device, TYPED_PAIR_SHADER).expect("compile shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Pair {
            a: u32,
            b: u32,
        }
        impl StructuredBufferElement for Pair {}

        let input_data: Vec<Pair> = (0..8).map(|i| Pair { a: i + 1, b: i + 10 }).collect();
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let input = pool
            .acquire_buffer_with_data(&input_data, BufferKind::Scattered)
            .expect("input");
        let mut output = pool
            .acquire_buffer_with_data(&[Pair { a: 0, b: 0 }; 8], BufferKind::Scattered)
            .expect("output");

        // BufRO<Pair> must bind the SRV slot; ReadWrite yields UAV and reads zeros on WARP.

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("typed_copy", &pipeline)
            .with_parcel(&input, NodeAccess::Read)
            .with_parcel(&output, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &output)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let loan = grant.claim(&mut frame).expect("claim").consume().expect("grant read");
        let result: &[Pair] = bytemuck::cast_slice(&loan);

        for i in 0..8u32 {
            let expected_a = (i + 1) + (i + 10);
            let expected_b = (i + 1) * (i + 10);
            assert_eq!(result[i as usize].a, expected_a, "output[{i}].a");
            assert_eq!(result[i as usize].b, expected_b, "output[{i}].b");
        }
    }

    fn scheme_compute_write_to_texture(device: &Device) {
        let ctx = submission_context(&device);

        let shader = ShaderModule::from_slang(&device, WRITE_TEXTURE_SHADER).expect("shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

        let width = 16u32;
        let height = 16u32;
        let wg_x = width.div_ceil(8);
        let wg_y = height.div_ceil(8);

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let texture = pool
            .acquire_texture(
                width,
                height,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_SRC,
                None,
            )
            .expect("texture");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_tex", &pipeline)
            .with_parcel(&texture, NodeAccess::Write)
            .dispatch(wg_x, wg_y, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let loan = grant.claim(&mut frame).expect("claim").consume().expect("grant read");

        let mut output = &*loan;
        let nonzero = output.iter().filter(|&&b| b != 0).count();
        assert!(nonzero > 0, "texture readback all zeros");
        assert_eq!(output[0], 255, "R channel");
        assert_eq!(output[1], 0, "G channel");
        assert_eq!(output[2], 0, "B channel");
        assert_eq!(output[3], 255, "A channel");
    }

    /// Verify that a [`goldy::Texture`] from [`RetainedPool`] can be bound via
    /// [`goldy::scheme::SchemeNodeBuilder::with_parcel`] using its parcel stamp.
    fn scheme_with_parcel_raw_texture(device: &Device) {
        let ctx = submission_context(&device);

        let shader = ShaderModule::from_slang(&device, WRITE_TEXTURE_SHADER).expect("shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

        let width = 8u32;
        let height = 8u32;
        let wg_x = width.div_ceil(8);
        let wg_y = height.div_ceil(8);

        let zeros = vec![0u8; (width * height * 4) as usize];
        let texture = test_alloc_texture(
            &device,
            &zeros,
            width,
            height,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
        );

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_tex_raw", &pipeline)
            .with_parcel(&texture, NodeAccess::Write)
            .dispatch(wg_x, wg_y, 1);
        let mut frame = scheme.submit().expect("submit");
        frame.wait_until_settled().expect("wait");

        let mut output = read_texture_via_scheme_copy(&ctx, &texture);

        assert_eq!(output[0], 255, "R channel");
        assert_eq!(output[1], 0, "G channel");
        assert_eq!(output[2], 0, "B channel");
        assert_eq!(output[3], 255, "A channel");
        let nonzero = output.iter().filter(|&&b| b != 0).count();
        assert!(
            nonzero > 0,
            "texture readback all zeros — barrier not recorded correctly"
        );
    }

    fn scheme_wave_inclusive_scan_uniform_64(device: &Device) {
        let ctx = submission_context(&device);
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");
        let result = dispatch_u32_write_and_read(&ctx, WAVE_SCAN_64_UNIFORM, &out, 64);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, i as u32 + 1, "wave_scan_uniform_64[{i}]");
        }
    }

    fn scheme_wave_inclusive_scan_ramp_64(device: &Device) {
        let ctx = submission_context(&device);
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");
        let result = dispatch_u32_write_and_read(&ctx, WAVE_SCAN_64_RAMP, &out, 64);
        for (i, &val) in result.iter().enumerate() {
            let k = (i + 1) as u32;
            let expected = k * (k + 1) / 2;
            assert_eq!(val, expected, "wave_scan_ramp_64[{i}]");
        }
    }

    fn scheme_wave_inclusive_scan_uniform_256(device: &Device) {
        let ctx = submission_context(&device);
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(256 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");
        let result = dispatch_u32_write_and_read(&ctx, WAVE_SCAN_256_UNIFORM, &out, 256);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, i as u32 + 1, "wave_scan_uniform_256[{i}]");
        }
    }

    fn scheme_workgroup_reduce_uint_correct(device: &Device) {
        let ctx = submission_context(&device);
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");
        let result = dispatch_u32_write_and_read(&ctx, REDUCE_64_UNIFORM, &out, 64);
        assert_eq!(result[0], 64, "workgroup_reduce thread 0");
    }

    fn scheme_workgroup_inclusive_scan_uint_correct(device: &Device) {
        let ctx = submission_context(&device);
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");
        let result = dispatch_u32_write_and_read(&ctx, INCLUSIVE_SCAN_64_UNIFORM, &out, 64);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, i as u32 + 1, "workgroup_inclusive_scan[{i}]");
        }
    }

    fn scheme_workgroup_broadcast_correct(device: &Device) {
        let ctx = submission_context(&device);
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");
        let result = dispatch_u32_write_and_read(&ctx, BROADCAST_64, &out, 64);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, 42, "workgroup_broadcast[{i}]");
        }
    }

    fn scheme_workgroup_upper_bound_linear(device: &Device) {
        let ctx = submission_context(&device);
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");
        let result = dispatch_u32_write_and_read(&ctx, UPPER_BOUND_64, &out, 64);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, i as u32, "workgroup_upper_bound[{i}]");
        }
    }

    fn scheme_texture_dual_view_round_trip(device: &Device) {
        const W: u32 = 4;
        const H: u32 = 4;
        const N: usize = (W * H) as usize;

        let ctx = submission_context(&device);

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let tex = pool
            .acquire_texture(
                W,
                H,
                TextureFormat::Rgba8Unorm,
                TextureKind::DirectInterpolated,
                TextureFlags::empty(),
                None,
            )
            .expect("texture");
        let out = pool
            .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");

        let write_shader = ShaderModule::from_slang(&device, DUAL_VIEW_WRITE_SHADER).expect("write shader");
        let write_pipeline = ComputePipeline::new(&device, &write_shader).expect("write pipeline");
        let read_shader = ShaderModule::from_slang(&device, DUAL_VIEW_READ_SHADER).expect("read shader");
        let read_pipeline = ComputePipeline::new(&device, &read_shader).expect("read pipeline");
        let sampler = Sampler::nearest(&device).expect("sampler");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write", &write_pipeline)
            .with_parcel(&tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("read", &read_pipeline)
            .with_parcel(&tex, NodeAccess::Read)
            .with_parcel(&sampler, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let loan = grant.claim(&mut frame).expect("claim").consume().expect("grant read");
        let result: &[u32] = bytemuck::cast_slice(&loan);

        for y in 0..H as usize {
            for x in 0..W as usize {
                let expected_r = x as u8;
                let expected_g = y as u8;
                let packed = result[y * W as usize + x];
                let r = (packed & 0xFF) as u8;
                let g = ((packed >> 8) & 0xFF) as u8;
                let b = ((packed >> 16) & 0xFF) as u8;
                let a = ((packed >> 24) & 0xFF) as u8;
                assert_eq!(r, expected_r, "r mismatch at ({x},{y})");
                assert_eq!(g, expected_g, "g mismatch at ({x},{y})");
                assert_eq!(b, 0, "b mismatch at ({x},{y})");
                assert_eq!(a, 255, "a mismatch at ({x},{y})");
            }
        }
    }

    fn scheme_two_contexts_both_submit_and_complete(device: &Device) {
        let ctx_a = submission_context(&device);
        let ctx_b = submission_context(&device);

        let shader = ShaderModule::from_slang(&device, IN_PLACE_DOUBLE_SHADER).expect("shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf_a = pool
            .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("buf_a");
        let buf_b = pool
            .acquire_buffer_with_data(&(100..164).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("buf_b");

        let mut scheme_a = Scheme::new(&ctx_a);
        scheme_a
            .node("n0", &pipeline)
            .with_parcel(&buf_a, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let grant_a = MemoryExchange::new(scheme_a.context())
            .bind_withdraw(&mut scheme_a, &buf_a)
            .expect("withdraw");
        let mut frame_a = scheme_a.submit().expect("ctx_a submit");

        let mut scheme_b = Scheme::new(&ctx_b);
        scheme_b
            .node("n0", &pipeline)
            .with_parcel(&buf_b, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let grant_b = MemoryExchange::new(scheme_b.context())
            .bind_withdraw(&mut scheme_b, &buf_b)
            .expect("withdraw");
        let mut frame_b = scheme_b.submit().expect("ctx_b submit");

        let result_a = read_grant_u32(&grant_a, &mut frame_a, 64);
        let result_b = read_grant_u32(&grant_b, &mut frame_b, 64);
        for i in 0..64 {
            assert_eq!(result_a[i], i as u32 * 2, "buf_a[{i}]");
            assert_eq!(result_b[i], (100 + i as u32) * 2, "buf_b[{i}]");
        }
    }

    fn scheme_two_contexts_reclaim_independently(device: &Device) {
        let ctx_a = submission_context(&device);
        let _ctx_b = submission_context(&device);

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer(256, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("buf");

        let mut scheme = Scheme::new(&ctx_a);
        let mut frame_a = scheme.submit().expect("ctx_a submit");
        drop(buf);

        frame_a.wait_until_settled().expect("ctx_a wait");
        ctx_a.flush_deferred_deletions();

        assert_eq!(
            ctx_a.deferred_deletion_pending_count(),
            0,
            "ctx_a must reclaim without waiting for ctx_b (no device-global horizon)"
        );
    }

    /// Two back-to-back non-blocking scheme submissions each write different data to
    /// the **same** parcel then copy it to a distinct output.
    ///
    /// This is the Scheme-API migration of `test_write_buffer_reuse_across_submissions`.
    /// Each scheme uses [`MemoryExchange::bind_deposit_buffer`] so the deposit copy is part of
    /// the same GPU submission as the compute node that reads it.  Both schemes are
    /// submitted without any CPU-side wait between them; correctness relies entirely
    /// on the staging belt handing out independent staging regions for the two uploads
    /// (tagged with each scheme's timeline value) and not recycling the first region
    /// until the first submission's GPU work has completed.
    fn scheme_deposit_buffer_reuse_across_submissions(device: &Device) {
        const N: usize = 16;

        let ctx = submission_context(&device);

        let pipeline = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
        )
        .expect("pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let mid = pool
            .acquire_buffer(
                (N * core::mem::size_of::<u32>()) as u64,
                BufferKind::Scattered,
                None,
                BufferFlags::empty(),
                None,
            )
            .expect("mid");
        let out_a = pool
            .acquire_buffer(
                (N * core::mem::size_of::<u32>()) as u64,
                BufferKind::Scattered,
                None,
                BufferFlags::empty(),
                None,
            )
            .expect("out_a");
        let out_b = pool
            .acquire_buffer(
                (N * core::mem::size_of::<u32>()) as u64,
                BufferKind::Scattered,
                None,
                BufferFlags::empty(),
                None,
            )
            .expect("out_b");

        let data_a: Vec<u32> = (100..100 + N as u32).collect();
        let data_b: Vec<u32> = (900..900 + N as u32).collect();

        // Scheme 1: write data_a into mid, copy mid → out_a, then read out_a back.
        let mut s1 = Scheme::new(&ctx);
        let deposit_a = MemoryExchange::new(&ctx)
            .bind_deposit_buffer(&mut s1, &mid, (N * core::mem::size_of::<u32>()) as u64)
            .expect("bind deposit a");
        deposit_a
            .write(&mut s1, 0, bytemuck::cast_slice(&data_a))
            .expect("commit write a");
        s1.node("copy_a", &pipeline)
            .with_parcel(&mid, NodeAccess::Read)
            .with_parcel(&out_a, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant_a = MemoryExchange::new(s1.context())
            .bind_withdraw(&mut s1, &out_a)
            .expect("grant_a");
        let mut sub1 = s1.submit().expect("submit 1");

        // Scheme 2: submitted immediately, without waiting for sub1.
        // The staging belt must not recycle the staging region used by s1.
        let mut s2 = Scheme::new(&ctx);
        let deposit_b = MemoryExchange::new(&ctx)
            .bind_deposit_buffer(&mut s2, &mid, (N * core::mem::size_of::<u32>()) as u64)
            .expect("bind deposit b");
        deposit_b
            .write(&mut s2, 0, bytemuck::cast_slice(&data_b))
            .expect("commit write b");
        s2.node("copy_b", &pipeline)
            .with_parcel(&mid, NodeAccess::Read)
            .with_parcel(&out_b, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant_b = MemoryExchange::new(s2.context())
            .bind_withdraw(&mut s2, &out_b)
            .expect("grant_b");
        let mut sub2 = s2.submit().expect("submit 2");

        let got_a = read_grant_u32(&grant_a, &mut sub1, N);
        let got_b = read_grant_u32(&grant_b, &mut sub2, N);
        assert_eq!(got_a, data_a, "output A corrupted (staging race?)");
        assert_eq!(got_b, data_b, "output B wrong");
    }

    // ---------------------------------------------------------------------------
    // Migrated from compute_integration.rs — uniform scalar params via with_param
    // ---------------------------------------------------------------------------

    fn scheme_uniform_param_uint_roundtrip(device: &Device) {
        const SHADER: &str = r#"
    import goldy_exp;
    [goldy_compute]
    [numthreads(1, 1, 1)]
    void cs_main(Scattered<uint> out, uint value, ThreadId id) {
        out[0] = value;
    }
    "#;

        let ctx = submission_context(&device);
        let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");

        const EXPECTED: u32 = 42;
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("uniform_uint", &pipeline)
            .with_parcel(&out, NodeAccess::Write)
            .with_param(EXPECTED)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("grant");
        let mut submission = scheme.submit().expect("submit");
        assert_eq!(read_grant_u32(&grant, &mut submission, 1)[0], EXPECTED);
    }

    fn scheme_uniform_param_uint_zero(device: &Device) {
        const SHADER: &str = r#"
    import goldy_exp;
    [goldy_compute]
    [numthreads(1, 1, 1)]
    void cs_main(Scattered<uint> out, uint value, ThreadId id) {
        out[0] = value;
    }
    "#;

        let ctx = submission_context(&device);
        let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer_with_data(&[0xDEAD_BEEFu32], BufferKind::Scattered)
            .expect("out");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("uniform_zero", &pipeline)
            .with_parcel(&out, NodeAccess::Write)
            .with_param(0u32)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("grant");
        let mut submission = scheme.submit().expect("submit");
        assert_eq!(read_grant_u32(&grant, &mut submission, 1)[0], 0);
    }

    fn scheme_uniform_param_uint_max(device: &Device) {
        const SHADER: &str = r#"
    import goldy_exp;
    [goldy_compute]
    [numthreads(1, 1, 1)]
    void cs_main(Scattered<uint> out, uint value, ThreadId id) {
        out[0] = value;
    }
    "#;

        let ctx = submission_context(&device);
        let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("uniform_max", &pipeline)
            .with_parcel(&out, NodeAccess::Write)
            .with_param(u32::MAX)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("grant");
        let mut submission = scheme.submit().expect("submit");
        assert_eq!(read_grant_u32(&grant, &mut submission, 1)[0], u32::MAX);
    }

    fn scheme_uniform_param_float_reinterpret(device: &Device) {
        const SHADER: &str = r#"
    import goldy_exp;
    [goldy_compute]
    [numthreads(1, 1, 1)]
    void cs_main(Scattered<float> out, float value, ThreadId id) {
        out[0] = value;
    }
    "#;

        let ctx = submission_context(&device);
        let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");

        #[allow(clippy::approx_constant)]
        let value: f32 = 3.14159;
        let bits = value.to_bits();

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("uniform_float", &pipeline)
            .with_parcel(&out, NodeAccess::Write)
            .with_param(bits)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("grant");
        let mut submission = scheme.submit().expect("submit");
        assert_eq!(read_grant_u32(&grant, &mut submission, 1)[0], bits);
    }

    fn scheme_uniform_two_independent_scalar_params(device: &Device) {
        const SHADER: &str = r#"
    import goldy_exp;
    [goldy_compute]
    [numthreads(1, 1, 1)]
    void cs_main(Scattered<uint> out, uint a, uint b, ThreadId id) {
        out[0] = a;
        out[1] = b;
    }
    "#;

        let ctx = submission_context(&device);
        let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let out = pool
            .acquire_buffer(8, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");

        const A: u32 = 0xABCD;
        const B: u32 = 0x1234;

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("uniform_two", &pipeline)
            .with_parcel(&out, NodeAccess::Write)
            .with_param(A)
            .with_param(B)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("grant");
        let mut submission = scheme.submit().expect("submit");
        let result = read_grant_u32(&grant, &mut submission, 2);
        assert_eq!(result[0], A);
        assert_eq!(result[1], B);
    }

    fn scheme_uniform_scalar_after_two_buffer_params(device: &Device) {
        const SHADER: &str = r#"
    import goldy_exp;
    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<uint> inp, Scattered<uint> out, uint offset, ThreadId id) {
        out[id.x] = inp[id.x] + offset;
    }
    "#;

        let ctx = submission_context(&device);
        let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

        const N: usize = 64;
        let input: Vec<u32> = (0..N as u32).collect();
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let inp = pool
            .acquire_buffer_with_data(&input, BufferKind::Scattered)
            .expect("inp");
        let out = pool
            .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out");

        const OFFSET: u32 = 100;

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("uniform_offset", &pipeline)
            .with_parcel(&inp, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .with_param(OFFSET)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("grant");
        let mut submission = scheme.submit().expect("submit");
        let result = read_grant_u32(&grant, &mut submission, N);
        let expected: Vec<u32> = input.iter().map(|v| v + OFFSET).collect();
        assert_eq!(result, expected);
    }

    // ---------------------------------------------------------------------------
    // Migrated from compute_integration.rs — partitioned buffer field binding
    // ---------------------------------------------------------------------------

    /// Two fields in one buffer. Shader copies from field A to field B.
    fn scheme_buffer_view_copy_between_sub_regions(device: &Device) {
        use goldy::{ordinal, Init};

        let ctx = submission_context(&device);

        let pipeline = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("compile shader"),
        )
        .expect("create pipeline");

        const N: usize = 64;
        let mut src = vec![0u32; N];
        for (i, slot) in src.iter_mut().enumerate() {
            *slot = (i + 1) as u32;
        }
        let dst = vec![0u32; N];

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let cells = pool
            .acquire_record([ordinal(Init::data(&src)), ordinal(Init::data(&dst))])
            .expect("acquire_record");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n0", &pipeline)
            .with_parcel(&cells[0], NodeAccess::ReadWrite)
            .with_parcel(&cells[1], NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &cells[1])
            .expect("withdraw");
        let mut submission = scheme.submit().expect("submit");

        let result = read_grant_u32(&grant, &mut submission, N);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(
                val,
                (i + 1) as u32,
                "dest[{}]: expected {} (copied from source field), got {}",
                i,
                i + 1,
                val
            );
        }
    }

    /// Shader doubles values in one field — the sibling field must be untouched.
    fn scheme_buffer_view_isolation(device: &Device) {
        use goldy::{ordinal, Init};

        let ctx = submission_context(&device);

        let pipeline = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, IN_PLACE_DOUBLE_SHADER).expect("compile shader"),
        )
        .expect("create pipeline");

        const N: usize = 64;
        let sentinel = vec![100u32; N];
        let mut work = vec![0u32; N];
        for (i, slot) in work.iter_mut().enumerate() {
            *slot = (i + 1) as u32;
        }

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let cells = pool
            .acquire_record([ordinal(Init::data(&sentinel)), ordinal(Init::data(&work))])
            .expect("acquire_record");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n0", &pipeline)
            .with_parcel(&cells[1], NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant_sentinel = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &cells[0])
            .expect("grant sentinel");
        let grant_work = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &cells[1])
            .expect("grant work");
        let mut submission = scheme.submit().expect("submit");

        let sentinel_vals = read_grant_u32(&grant_sentinel, &mut submission, N);
        assert!(
            sentinel_vals.iter().all(|&v| v == 100),
            "sentinel field must be untouched"
        );

        let result = read_grant_u32(&grant_work, &mut submission, N);
        for (i, &val) in result.iter().enumerate() {
            let expected = ((i + 1) as u32) * 2;
            assert_eq!(
                val, expected,
                "field[{}]: expected {} (doubled), got {}",
                i, expected, val
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Indirect dispatch (DispatchShape parcels)
    // ---------------------------------------------------------------------------

    const WRITE_DISPATCH_SHAPE_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(1, 1, 1)]
    void cs_main(Scattered<DispatchShape> shape, ThreadId id) {
        DispatchShape s;
        s.x = 4;
        s.y = 1;
        s.z = 1;
        shape[0] = s;
    }
    "#;

    /// Producer writes `DispatchShape{4,1,1}`; consumer `dispatch(&shape)` doubles 256 values.
    fn scheme_compute_dispatch_indirect(device: &Device) {
        let ctx = submission_context(&device);

        let write_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, WRITE_DISPATCH_SHAPE_SHADER).expect("compile write shape shader"),
        )
        .expect("create write pipeline");
        let work_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, IN_PLACE_DOUBLE_SHADER).expect("compile double shader"),
        )
        .expect("create work pipeline");

        const N: usize = 256;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let shape = pool
            .acquire_buffer_sized::<DispatchShape>(1, BufferKind::Scattered, BufferFlags::empty())
            .expect("shape buffer");
        let work = pool
            .acquire_buffer_with_data(&(0..N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("work buffer");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_shape", &write_pipe)
            .with_parcel(&shape, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("work", &work_pipe)
            .with_parcel(&work, NodeAccess::ReadWrite)
            .dispatch_shape_parcel(&*shape)
            .expect("indirect dispatch");

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &work)
            .expect("withdraw");
        let mut submission = scheme.submit().expect("submit");
        let result = read_grant_u32(&grant, &mut submission, N);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, (i as u32) * 2, "element {i}: expected {}, got {val}", i * 2);
        }
    }

    /// Indirect dispatch fails when the shape parcel's backing buffer was released before submit.
    fn scheme_dispatch_indirect_invalid_buffer(device: &Device) {
        let ctx = submission_context(&device);

        let work_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("compile double shader"),
        )
        .expect("create work pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let work = pool
            .acquire_buffer_with_data(&vec![1u32; 64], BufferKind::Scattered)
            .expect("work buffer");

        let mut scheme = Scheme::new(&ctx);
        {
            let shape = pool
                .acquire_buffer_sized::<DispatchShape>(1, BufferKind::Scattered, BufferFlags::empty())
                .expect("shape buffer");
            scheme
                .node("work", &work_pipe)
                .with_parcel(&work, NodeAccess::Write)
                .dispatch_shape_parcel(&*shape)
                .expect("record indirect dispatch");
            drop(shape);
        }

        let err = scheme.submit().expect_err("submit with destroyed shape buffer");
        let _ = format!("{err:?}");
    }

    /// Non-`DispatchShape` parcels are rejected at scheme-build time.
    fn scheme_dispatch_indirect_wrong_type_rejected(device: &Device) {
        let _ctx = submission_context(&device);

        let work_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("compile double shader"),
        )
        .expect("create work pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let shape = pool
            .acquire_buffer_sized::<u32>(3, BufferKind::Scattered, BufferFlags::empty())
            .expect("u32 buffer standing in for shape");
        let work = pool
            .acquire_buffer_with_data(&vec![0u32; 64], BufferKind::Scattered)
            .expect("work buffer");

        let mut scheme = Scheme::new(&_ctx);
        let err = scheme
            .node("work", &work_pipe)
            .with_parcel(&work, NodeAccess::Write)
            .dispatch_shape_parcel(&*shape)
            .expect_err("wrong element stride must be rejected");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("stride") || msg.contains("DispatchShape"),
            "unexpected error: {msg}"
        );
    }

    /// Zero-fill via upload, then producer-written indirect shape, then copy dispatch.
    fn scheme_stress_zeros_then_indirect_dispatch(device: &Device) {
        let ctx = submission_context(&device);

        let write_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, WRITE_DISPATCH_SHAPE_SHADER).expect("compile write shape shader"),
        )
        .expect("create write pipeline");
        let copy_pipe = ComputePipeline::new(
            &device,
            &ShaderModule::from_slang(&device, COPY_SHADER).expect("compile copy shader"),
        )
        .expect("create copy pipeline");

        const N: usize = 256;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let shape = pool
            .acquire_buffer_sized::<DispatchShape>(1, BufferKind::Scattered, BufferFlags::empty())
            .expect("shape buffer");
        let src = pool
            .acquire_buffer_with_data(&(1..=N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("src buffer");
        let out = pool
            .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("out buffer");

        write_zeros_to_parcel(&ctx, &src, N * 4);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_shape", &write_pipe)
            .with_parcel(&shape, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("copy_indirect", &copy_pipe)
            .with_parcel(&src, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .dispatch_shape_parcel(&*shape)
            .expect("indirect dispatch");

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out)
            .expect("withdraw");
        let mut submission = scheme.submit().expect("submit");
        let result = read_grant_u32(&grant, &mut submission, N);
        let nonzero_count = result.iter().filter(|&&v| v != 0).count();
        assert_eq!(
            nonzero_count, 0,
            "expected all zeros after zero-fill + indirect copy, but {nonzero_count}/{N} were nonzero"
        );
    }

    /// Migrated from `stress_alternating_write_dispatch` in `task_graph_integration.rs`.
    ///
    /// Two deposit writes on the same buffer, each followed by a dispatch
    /// that reads it, all within one `Scheme` submission.  This exercises the
    /// write → dispatch → write → dispatch (WAW + RAW) barrier sequence and confirms
    /// that the upload remap table correctly tracks both write nodes despite them
    /// sharing the same `(buffer, offset)` key.
    ///
    /// Because the scheme contains deposit copy nodes it is never retained — it
    /// records fresh on every `submit()`.
    fn scheme_stress_alternating_write_dispatch(device: &Device) {
        let ctx = submission_context(&device);

        let copy_pipe =
            ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

        const N: usize = 256;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        let out1 = pool
            .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        let out2 = pool
            .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        let data1: Vec<u32> = (0..N as u32).map(|i| i + 100).collect();
        let data2: Vec<u32> = (0..N as u32).map(|i| i + 200).collect();
        let bytes1: Vec<u8> = bytemuck::cast_slice(&data1).to_vec();
        let bytes2: Vec<u8> = bytemuck::cast_slice(&data2).to_vec();

        let mut scheme = Scheme::new(&ctx);
        let memory = MemoryExchange::new(&ctx);
        let deposit1 = memory
            .bind_deposit_buffer(&mut scheme, &buf, bytes1.len() as u64)
            .expect("bind deposit 1");
        // Phase 1: write data1 into buf, copy to out1.
        deposit1.write(&mut scheme, 0, &bytes1).expect("commit write 1");
        scheme
            .node("copy1", &copy_pipe)
            .with_parcel(&buf, NodeAccess::Read)
            .with_parcel(&out1, NodeAccess::Write)
            .dispatch((N / 64) as u32, 1, 1);
        // Phase 2: overwrite buf with data2 (WAW), copy to out2.
        let deposit2 = memory
            .bind_deposit_buffer(&mut scheme, &buf, bytes2.len() as u64)
            .expect("bind deposit 2");
        deposit2.write(&mut scheme, 0, &bytes2).expect("commit write 2");
        scheme
            .node("copy2", &copy_pipe)
            .with_parcel(&buf, NodeAccess::Read)
            .with_parcel(&out2, NodeAccess::Write)
            .dispatch((N / 64) as u32, 1, 1);

        let grant1 = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out1)
            .expect("grant_read out1");
        let grant2 = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &out2)
            .expect("grant_read out2");
        let mut submission = scheme.submit().expect("submit");

        let result1 = read_grant_u32(&grant1, &mut submission, N);
        for (i, &val) in result1.iter().enumerate() {
            assert_eq!(val, data1[i], "out1[{i}]: expected {}, got {val}", data1[i]);
        }
        let result2 = read_grant_u32(&grant2, &mut submission, N);
        for (i, &val) in result2.iter().enumerate() {
            assert_eq!(val, data2[i], "out2[{i}]: expected {}, got {val}", data2[i]);
        }
    }

    // ─── clear_parcel ─────────────────────────────────────────────────────

    fn scheme_clear_parcel_full(device: &Device) {
        let ctx = submission_context(&device);

        const N: usize = 64;
        let byte_size = (N * 4) as u64;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer_with_data(&vec![0xDEAD_BEEFu32; N], BufferKind::Scattered)
            .expect("buf");

        let mut scheme = Scheme::new(&ctx);
        scheme.clear_parcel(&buf, 0, byte_size).expect("clear_parcel");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buf)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let result = read_grant_u32(&grant, &mut frame, N);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, 0, "element {i} should be zero after full clear");
        }
    }

    fn scheme_clear_parcel_partial_preserves_edges(device: &Device) {
        let ctx = submission_context(&device);

        // 64 u32s: indices [0..16] = 0xAAAA, [16..48] = 0xBBBB (to be cleared), [48..64] = 0xCCCC
        const N: usize = 64;
        let mut init: Vec<u32> = Vec::with_capacity(N);
        for i in 0..N {
            if i < 16 {
                init.push(0xAAAA_AAAAu32);
            } else if i < 48 {
                init.push(0xBBBB_BBBBu32);
            } else {
                init.push(0xCCCC_CCCCu32);
            }
        }

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer_with_data(&init, BufferKind::Scattered)
            .expect("buf");

        // Clear elements [16..48] → bytes 64..192, size = 32 * 4 = 128.
        let mut scheme = Scheme::new(&ctx);
        scheme.clear_parcel(&buf, 16 * 4, 32 * 4).expect("clear_parcel");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buf)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let result = read_grant_u32(&grant, &mut frame, N);
        for i in 0..16 {
            assert_eq!(
                result[i], 0xAAAA_AAAAu32,
                "edge[{i}] before clear region should be unchanged"
            );
        }
        for i in 16..48 {
            assert_eq!(result[i], 0u32, "cleared region[{i}] should be zero");
        }
        for i in 48..64 {
            assert_eq!(
                result[i], 0xCCCC_CCCCu32,
                "edge[{i}] after clear region should be unchanged"
            );
        }
    }

    fn scheme_clear_parcel_size_zero_fills_to_end(device: &Device) {
        let ctx = submission_context(&device);

        // 64 u32s: first 16 stay, rest cleared via size=0 (fill-to-end).
        const N: usize = 64;
        let init: Vec<u32> = (0..N as u32).collect();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer_with_data(&init, BufferKind::Scattered)
            .expect("buf");

        let mut scheme = Scheme::new(&ctx);
        // offset = 16 elements in, size = 0 → fill from byte 64 to end
        scheme.clear_parcel(&buf, 16 * 4, 0).expect("clear_parcel size=0");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buf)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let result = read_grant_u32(&grant, &mut frame, N);
        for i in 0..16 {
            assert_eq!(result[i], init[i], "element {i} before offset should be unchanged");
        }
        for i in 16..64 {
            assert_eq!(
                result[i], 0u32,
                "element {i} at or after offset should be zero (size=0 fill-to-end)"
            );
        }
    }

    fn scheme_clear_parcel_requires_buffer_parcel(device: &Device) {
        let ctx = submission_context(&device);

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let tex_parcel = pool
            .acquire_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::empty(),
                None,
            )
            .expect("tex_parcel");

        let mut scheme = Scheme::new(&ctx);
        let result = scheme.clear_parcel(&tex_parcel, 0, 64);
        assert!(result.is_err(), "clear_parcel should reject texture parcels");
    }

    // ─── copy_buffer_parcel ───────────────────────────────────────────────────────

    fn scheme_copy_buffer_parcel_basic(device: &Device) {
        let ctx = submission_context(&device);

        const N: usize = 64;
        let byte_size = (N * 4) as u64;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let src = pool
            .acquire_buffer_with_data(&(1..=N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("src");
        let dst = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("dst");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .copy_buffer_parcel(&src, 0, &dst, 0, byte_size)
            .expect("copy_buffer_parcel");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &dst)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let result = read_grant_u32(&grant, &mut frame, N);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, i as u32 + 1, "dst[{i}]: expected {}, got {val}", i + 1);
        }
    }

    fn scheme_copy_buffer_parcel_partial_with_offsets(device: &Device) {
        let ctx = submission_context(&device);

        // src: 64 u32s [0..63]. Copy src[16..32] (bytes 64..128) into dst[0..16].
        const N_SRC: usize = 64;
        const N_DST: usize = 16;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let src = pool
            .acquire_buffer_with_data(&(0..N_SRC as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("src");
        let dst = pool
            .acquire_buffer(
                (N_DST * 4) as u64,
                BufferKind::Scattered,
                None,
                BufferFlags::empty(),
                None,
            )
            .expect("dst");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .copy_buffer_parcel(&src, (16 * 4) as u64, &dst, 0, (N_DST * 4) as u64)
            .expect("copy_buffer_parcel");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &dst)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let result = read_grant_u32(&grant, &mut frame, N_DST);
        for (i, &val) in result.iter().enumerate() {
            let expected = 16u32 + i as u32;
            assert_eq!(val, expected, "dst[{i}]: expected {expected}, got {val}");
        }
    }

    fn scheme_copy_buffer_parcel_rejects_texture_src(device: &Device) {
        let ctx = submission_context(&device);

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let tex = pool
            .acquire_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::empty(),
                None,
            )
            .expect("tex");
        let buf = pool
            .acquire_buffer(64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("buf");

        let mut scheme = Scheme::new(&ctx);
        let result = scheme.copy_buffer_parcel(&tex, 0, &buf, 0, 64);
        assert!(result.is_err(), "copy_buffer_parcel should reject texture as src");
    }

    fn scheme_copy_buffer_parcel_rejects_texture_dst(device: &Device) {
        let ctx = submission_context(&device);

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let buf = pool
            .acquire_buffer(64, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("buf");
        let tex = pool
            .acquire_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::empty(),
                None,
            )
            .expect("tex");

        let mut scheme = Scheme::new(&ctx);
        let result = scheme.copy_buffer_parcel(&buf, 0, &tex, 0, 64);
        assert!(result.is_err(), "copy_buffer_parcel should reject texture as dst");
    }

    fn scheme_copy_buffer_parcel_resubmit_does_not_rerecord(device: &Device) {
        let ctx = submission_context(&device);

        const N: usize = 64;
        let byte_size = (N * 4) as u64;
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let src = pool
            .acquire_buffer_with_data(&(0..N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
            .expect("src");
        let dst = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("dst");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .copy_buffer_parcel(&src, 0, &dst, 0, byte_size)
            .expect("copy_buffer_parcel");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &dst)
            .expect("withdraw");

        // First submit records.
        let mut frame1 = scheme.submit().expect("first submit");
        // Second submit should be a clean resubmit.
        let mut frame2 = scheme.submit().expect("second submit");

        assert_eq!(
            scheme.replay_stats().records,
            1,
            "copy_buffer_parcel is identity; should record exactly once"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            1,
            "second submit must be a retention hit"
        );

        // Data should be correct on both frames.
        let result1 = read_grant_u32(&grant, &mut frame1, N);
        let result2 = read_grant_u32(&grant, &mut frame2, N);
        for i in 0..N {
            assert_eq!(result1[i], i as u32, "frame1 dst[{i}]");
            assert_eq!(result2[i], i as u32, "frame2 dst[{i}]");
        }
    }

    // ─── CPU_WRITABLE staging ─────────────────────────────────────────────────────

    /// CPU_WRITABLE `Buffer::write` then same-frame scheme `copy_buffer_parcel`.
    ///
    /// Covers the staging contract: a settled/fresh host write is GPU-visible via the
    /// recorded copy without a Metal blit+wait on the write path (see `Buffer::write` /
    /// `BufferFlags::CPU_WRITABLE`). Distinct from the CPU→CPU roundtrip in
    /// `compute_integration::test_cpu_writable_write_read_roundtrip`.
    fn scheme_cpu_writable_staging_write_then_copy(device: &Device) {
        let ctx = submission_context(&device);

        const N: usize = 64;
        let byte_size = (N * 4) as u64;
        let data: Vec<u32> = (0xABC0_0000u32..).take(N).collect();
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        // Staging buffer: CPU-writable, written via parcel.write() before each submit.
        let staging = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::CPU_WRITABLE, None)
            .expect("staging");
        staging.write(0, &bytes).expect("staging.write");

        let dst = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("dst");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .copy_buffer_parcel(&staging, 0, &dst, 0, byte_size)
            .expect("copy_buffer_parcel");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &dst)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let result = read_grant_u32(&grant, &mut frame, N);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, data[i], "dst[{i}]: expected {:08X}, got {:08X}", data[i], val);
        }
    }

    fn scheme_cpu_writable_staging_update_each_frame(device: &Device) {
        let ctx = submission_context(&device);

        const N: usize = 64;
        let byte_size = (N * 4) as u64;
        let data1: Vec<u32> = (100u32..100 + N as u32).collect();
        let data2: Vec<u32> = (200u32..200 + N as u32).collect();
        let bytes1: Vec<u8> = bytemuck::cast_slice(&data1).to_vec();
        let bytes2: Vec<u8> = bytemuck::cast_slice(&data2).to_vec();

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let staging = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::CPU_WRITABLE, None)
            .expect("staging");
        let dst = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("dst");

        staging.write(0, &bytes1).expect("write bytes1");

        // Record once: copy node + grant.
        let mut scheme = Scheme::new(&ctx);
        scheme
            .copy_buffer_parcel(&staging, 0, &dst, 0, byte_size)
            .expect("copy_buffer_parcel");
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &dst)
            .expect("withdraw");

        // Frame 1: staging has bytes1.
        let mut frame1 = scheme.submit().expect("frame1 submit");
        let result1 = read_grant_u32(&grant, &mut frame1, N);
        for (i, &val) in result1.iter().enumerate() {
            assert_eq!(val, data1[i], "frame1 dst[{i}]");
        }

        // Update staging (frame1 is already waited on by grant.consume above).
        staging.write(0, &bytes2).expect("write bytes2");

        // Frame 2: resubmit with new staging data; topology unchanged → no re-record.
        let mut frame2 = scheme.submit().expect("frame2 submit");
        let result2 = read_grant_u32(&grant, &mut frame2, N);
        for (i, &val) in result2.iter().enumerate() {
            assert_eq!(val, data2[i], "frame2 dst[{i}]");
        }

        assert_eq!(
            scheme.replay_stats().records,
            1,
            "CPU_WRITABLE staging: topology should record exactly once"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(scheme.replay_stats().resubmit_hits, 1, "frame2 must be a retention hit");
    }

    // ─── write_texture ─────────────────────────────────────────────────────

    fn scheme_write_texture_round_trip(device: &Device) {
        let ctx = submission_context(&device);

        const W: u32 = 8;
        const H: u32 = 8;
        let pixels: Vec<u8> = (0..W * H)
            .flat_map(|i| {
                let r = (i % W) as u8;
                let g = (i / W) as u8;
                [r, g, 128u8, 255u8]
            })
            .collect();
        let texture = test_alloc_texture(
            &device,
            &vec![0u8; (W * H * 4) as usize],
            W,
            H,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
        );

        let mut scheme = Scheme::new(&ctx);
        let capacity = (W * H * 4) as u64;
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit_texture(&mut scheme, &texture, 0, 0, W, H, capacity, 0)
            .expect("bind deposit");
        deposit.write(&mut scheme, 0, &pixels).expect("deposit write");
        let grant = MemoryExchange::new(&ctx)
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let output = grant.claim(&mut frame).expect("claim").consume().expect("consume");

        assert_eq!(
            &*output,
            pixels.as_slice(),
            "write_texture: readback does not match uploaded data"
        );
    }

    fn scheme_write_texture_wrong_size_returns_error(device: &Device) {
        let ctx = submission_context(&device);
        let texture = test_alloc_texture(
            &device,
            &vec![0u8; 8 * 8 * 4],
            8,
            8,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::empty(),
        );

        let mut scheme = Scheme::new(&ctx);
        let capacity = (8 * 8 * 4) as u64;
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit_texture(&mut scheme, &texture, 0, 0, 8, 8, capacity, 0)
            .expect("bind deposit");
        // Partial staging fills are allowed; only exceeding declaration capacity errors.
        deposit
            .write(&mut scheme, 0, &[0u8; 10])
            .expect("undersized write should fit within capacity");
        let result2 = deposit.write(&mut scheme, 0, &[0u8; 8 * 8 * 4 + 4]);
        assert!(result2.is_err(), "deposit write should reject oversized data");
    }

    fn scheme_write_texture_marks_scheme_dirty(device: &Device) {
        let ctx = submission_context(&device);
        let texture = test_alloc_texture(
            &device,
            &vec![0u8; 4 * 4 * 4],
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::empty(),
        );

        let mut scheme = Scheme::new(&ctx);
        assert!(scheme.is_dirty(), "new scheme starts dirty");
        let capacity = (4 * 4 * 4) as u64;
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit_texture(&mut scheme, &texture, 0, 0, 4, 4, capacity, 0)
            .expect("bind deposit");
        assert!(scheme.is_dirty(), "scheme must be dirty after bind_deposit_texture");
        deposit.write(&mut scheme, 0, &[0u8; 4 * 4 * 4]).expect("first write");
        assert!(scheme.is_dirty(), "scheme must be dirty after deposit write");
        // Second write: scheme is already dirty, but staging another payload keeps it dirty.
        deposit.write(&mut scheme, 0, &[0u8; 4 * 4 * 4]).expect("second write");
        assert!(
            scheme.is_dirty(),
            "scheme must still be dirty after second deposit write"
        );
    }

    // ─── write_texture_region ─────────────────────────────────────────────

    fn scheme_write_texture_region_round_trip(device: &Device) {
        let ctx = submission_context(&device);

        const W: u32 = 8;
        const H: u32 = 8;
        // Initialize with zeros.
        let texture = test_alloc_texture(
            &device,
            &vec![0u8; (W * H * 4) as usize],
            W,
            H,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
        );

        // Write a 4×4 region at (2, 2) with a solid color.
        const RX: u32 = 2;
        const RY: u32 = 2;
        const RW: u32 = 4;
        const RH: u32 = 4;
        let region_pixels: Vec<u8> = vec![200u8, 100u8, 50u8, 255u8].repeat((RW * RH) as usize);

        let mut scheme = Scheme::new(&ctx);
        let region_capacity = (RW * RH * 4) as u64;
        let deposit = MemoryExchange::new(&ctx)
            .bind_deposit_texture(&mut scheme, &texture, RX, RY, RW, RH, region_capacity, 0)
            .expect("bind deposit");
        deposit.write(&mut scheme, 0, &region_pixels).expect("deposit write");
        let grant = MemoryExchange::new(&ctx)
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let output = grant.claim(&mut frame).expect("claim").consume().expect("consume");

        for y in 0..H as usize {
            for x in 0..W as usize {
                let base = (y * W as usize + x) * 4;
                let pixel = &output[base..base + 4];
                let in_region =
                    x >= RX as usize && x < (RX + RW) as usize && y >= RY as usize && y < (RY + RH) as usize;
                if in_region {
                    assert_eq!(pixel[0], 200, "R at ({x},{y}): should be region color");
                    assert_eq!(pixel[1], 100, "G at ({x},{y}): should be region color");
                    assert_eq!(pixel[2], 50, "B at ({x},{y}): should be region color");
                    assert_eq!(pixel[3], 255, "A at ({x},{y}): should be region color");
                } else {
                    assert_eq!(pixel[0], 0, "R at ({x},{y}): outside region should be zero");
                    assert_eq!(pixel[1], 0, "G at ({x},{y}): outside region should be zero");
                    assert_eq!(pixel[2], 0, "B at ({x},{y}): outside region should be zero");
                    assert_eq!(pixel[3], 0, "A at ({x},{y}): outside region should be zero");
                }
            }
        }
    }

    fn scheme_write_texture_region_oob_returns_error(device: &Device) {
        let ctx = submission_context(&device);
        let texture = test_alloc_texture(
            &device,
            &vec![0u8; 8 * 8 * 4],
            8,
            8,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::empty(),
        );

        let mut scheme = Scheme::new(&ctx);
        let memory = MemoryExchange::new(&ctx);
        // Region extends beyond texture width.
        let result = memory.bind_deposit_texture(&mut scheme, &texture, 6, 0, 4, 4, 4 * 4 * 4, 0);
        assert!(result.is_err(), "x+width exceeds texture width → error");
        // Region extends beyond texture height.
        let result2 = memory.bind_deposit_texture(&mut scheme, &texture, 0, 6, 4, 4, 4 * 4 * 4, 0);
        assert!(result2.is_err(), "y+height exceeds texture height → error");
    }

    fn scheme_write_texture_region_multiple_non_overlapping(device: &Device) {
        let ctx = submission_context(&device);

        const W: u32 = 8;
        const H: u32 = 8;
        let texture = test_alloc_texture(
            &device,
            &vec![0u8; (W * H * 4) as usize],
            W,
            H,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
        );

        // Top-left 4×4 → red (255,0,0,255). Bottom-right 4×4 → blue (0,0,255,255).
        let red: Vec<u8> = vec![255u8, 0, 0, 255].repeat(16);
        let blue: Vec<u8> = vec![0u8, 0, 255, 255].repeat(16);

        let mut scheme = Scheme::new(&ctx);
        let memory = MemoryExchange::new(&ctx);
        let red_capacity = (4 * 4 * 4) as u64;
        let blue_capacity = (4 * 4 * 4) as u64;
        let red_deposit = memory
            .bind_deposit_texture(&mut scheme, &texture, 0, 0, 4, 4, red_capacity, 0)
            .expect("bind red deposit");
        red_deposit.write(&mut scheme, 0, &red).expect("write red region");
        let blue_deposit = memory
            .bind_deposit_texture(&mut scheme, &texture, 4, 4, 4, 4, blue_capacity, 0)
            .expect("bind blue deposit");
        blue_deposit.write(&mut scheme, 0, &blue).expect("write blue region");
        let grant = MemoryExchange::new(&ctx)
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let output = grant.claim(&mut frame).expect("claim").consume().expect("consume");

        for y in 0..H as usize {
            for x in 0..W as usize {
                let base = (y * W as usize + x) * 4;
                let pixel = &output[base..base + 4];
                let in_red_region = x < 4 && y < 4;
                let in_blue_region = x >= 4 && y >= 4;
                if in_red_region {
                    assert_eq!(pixel[0], 255, "R at ({x},{y}): expected red");
                    assert_eq!(pixel[2], 0, "B at ({x},{y}): expected red region has no blue");
                } else if in_blue_region {
                    assert_eq!(pixel[0], 0, "R at ({x},{y}): expected blue region has no red");
                    assert_eq!(pixel[2], 255, "B at ({x},{y}): expected blue");
                } else {
                    assert_eq!(pixel[0], 0, "R at ({x},{y}): unwritten region should be zero");
                    assert_eq!(pixel[1], 0, "G at ({x},{y}): unwritten region should be zero");
                    assert_eq!(pixel[2], 0, "B at ({x},{y}): unwritten region should be zero");
                }
            }
        }
    }

    // ─── copy_buffer_to_texture_parcel ───────────────────────────────────────────

    fn scheme_copy_buffer_to_texture_parcel_full_texture(device: &Device) {
        let ctx = submission_context(&device);

        const W: u32 = 4;
        const H: u32 = 4;
        let byte_size = (W * H * 4) as u64;

        // Build a known RGBA pixel pattern.
        let pixels: Vec<u8> = (0..W * H)
            .flat_map(|i| {
                let r = (i % W) as u8 * 60;
                let g = (i / W) as u8 * 60;
                [r, g, 0u8, 255u8]
            })
            .collect();

        // CPU-writable staging buffer.
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let staging = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::CPU_WRITABLE, None)
            .expect("staging");
        staging.write(0, &pixels).expect("staging.write");

        // Destination texture (COPY_DST for buffer→texture copy; COPY_SRC for withdraw).
        let texture = test_alloc_texture(
            &device,
            &vec![0u8; (W * H * 4) as usize],
            W,
            H,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        );

        let mut scheme = Scheme::new(&ctx);
        scheme
            .copy_buffer_to_texture_parcel(&staging, 0, 0, &texture, 0, 0, W, H)
            .expect("copy_buffer_to_texture_parcel");
        let grant = MemoryExchange::new(&ctx)
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let output = grant.claim(&mut frame).expect("claim").consume().expect("consume");

        assert_eq!(
            &*output,
            pixels.as_slice(),
            "copy_buffer_to_texture_parcel: readback does not match staged pixel data"
        );
    }

    fn scheme_copy_buffer_to_texture_parcel_oob_returns_error(device: &Device) {
        let ctx = submission_context(&device);

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let staging = pool
            .acquire_buffer(64, BufferKind::Scattered, None, BufferFlags::CPU_WRITABLE, None)
            .expect("staging");
        let texture = test_alloc_texture(
            &device,
            &vec![0u8; 8 * 8 * 4],
            8,
            8,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::empty(),
        );

        let mut scheme = Scheme::new(&ctx);
        // x + width exceeds texture width.
        let result = scheme.copy_buffer_to_texture_parcel(&staging, 0, 0, &texture, 6, 0, 4, 4);
        assert!(result.is_err(), "x+width exceeds texture width → error");
        // y + height exceeds texture height.
        let result2 = scheme.copy_buffer_to_texture_parcel(&staging, 0, 0, &texture, 0, 6, 4, 4);
        assert!(result2.is_err(), "y+height exceeds texture height → error");
    }

    fn scheme_copy_buffer_to_texture_parcel_rejects_texture_src(device: &Device) {
        let ctx = submission_context(&device);

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        // Use a texture parcel (not a buffer) as the src — should error.
        let tex_parcel = pool
            .acquire_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::empty(),
                None,
            )
            .expect("tex_parcel");
        let dst_texture = test_alloc_texture(
            &device,
            &vec![0u8; 4 * 4 * 4],
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::empty(),
        );

        let mut scheme = Scheme::new(&ctx);
        let result = scheme.copy_buffer_to_texture_parcel(&tex_parcel, 0, 0, &dst_texture, 0, 0, 4, 4);
        assert!(
            result.is_err(),
            "copy_buffer_to_texture_parcel should reject a texture parcel as src"
        );
    }

    fn scheme_copy_buffer_to_texture_parcel_resubmit_is_retained(device: &Device) {
        let ctx = submission_context(&device);

        const W: u32 = 4;
        const H: u32 = 4;
        let byte_size = (W * H * 4) as u64;
        let pixels = vec![128u8; byte_size as usize];

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let staging = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::CPU_WRITABLE, None)
            .expect("staging");
        staging.write(0, &pixels).expect("staging.write");
        let texture = test_alloc_texture(
            &device,
            &vec![0u8; byte_size as usize],
            W,
            H,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        );

        let mut scheme = Scheme::new(&ctx);
        scheme
            .copy_buffer_to_texture_parcel(&staging, 0, 0, &texture, 0, 0, W, H)
            .expect("copy_buffer_to_texture_parcel");

        scheme.submit().expect("first submit");
        scheme.submit().expect("second submit");

        assert_eq!(
            scheme.replay_stats().records,
            1,
            "copy_buffer_to_texture_parcel is identity; should record exactly once"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            1,
            "second submit must be a retention hit"
        );
    }

    fn scheme_copy_buffer_to_texture_pitched_retained_after_layout_settles(device: &Device) {
        let ctx = submission_context(&device);

        const W: u32 = 4;
        const H: u32 = 4;
        let format = TextureFormat::Rgba8Unorm;
        let footprint = device
            .texture_copy_footprint(W, H, format)
            .expect("texture_copy_footprint");
        assert!(
            footprint.row_pitch > 0,
            "pitched upload test requires a non-zero row pitch"
        );

        let byte_size = footprint.staging_bytes;
        let tight_len = (W * H * 4) as usize;
        let pixels = vec![128u8; tight_len];

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let staging = pool
            .acquire_buffer(byte_size, BufferKind::Scattered, None, BufferFlags::CPU_WRITABLE, None)
            .expect("staging");

        let write_pitched = |buf: &goldy::Buffer| {
            if footprint.row_pitch == footprint.tight_row_bytes() {
                buf.write(footprint.footprint_offset, &pixels).expect("write tight");
            } else {
                let row_bytes = footprint.tight_row_bytes() as usize;
                for row in 0..H {
                    let src = &pixels[row as usize * row_bytes..(row as usize + 1) * row_bytes];
                    let dst_off = footprint.footprint_offset + row as u64 * footprint.row_pitch as u64;
                    buf.write(dst_off, src).expect("write pitched row");
                }
            }
        };

        write_pitched(&staging);

        let texture = test_alloc_texture(
            &device,
            &vec![0u8; tight_len],
            W,
            H,
            format,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        );

        let mut scheme = Scheme::new(&ctx);
        scheme
            .copy_buffer_to_texture_parcel(
                &staging.whole(),
                footprint.footprint_offset,
                footprint.row_pitch,
                &texture,
                0,
                0,
                W,
                H,
            )
            .expect("copy_buffer_to_texture_parcel pitched");

        let refresh_and_submit = |scheme: &mut Scheme, staging: &goldy::Buffer| {
            write_pitched(staging);
            scheme.submit().expect("submit");
        };

        refresh_and_submit(&mut scheme, &staging);
        refresh_and_submit(&mut scheme, &staging);
        refresh_and_submit(&mut scheme, &staging);

        #[cfg(not(feature = "metal"))]
        {
            assert_eq!(
                scheme.replay_stats().records,
                2,
                "initial record + one re-record when destination texture layout settles"
            );
            assert_eq!(
                scheme.replay_stats().resubmit_hits,
                1,
                "third submit must resubmit the retained pitched texture-upload partition"
            );
        }

        let mut output = read_texture_via_scheme_copy(&ctx, &texture);
        assert_eq!(output, pixels, "pitched retained upload must produce correct pixels");
    }

    // ─── Cross-scheme retention (shared-parcel topology) ─────────────────────────

    const CROSS_RETENTION_ELEMS: usize = 64;

    struct CrossRetentionBuffers {
        input: goldy::Buffer,
        shared: goldy::Buffer,
    }

    fn cross_retention_buffers(device: &Device) -> CrossRetentionBuffers {
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let input = pool
            .acquire_buffer_with_data(
                &(0..CROSS_RETENTION_ELEMS as u32).collect::<Vec<_>>(),
                BufferKind::Scattered,
            )
            .expect("input");
        let shared = pool
            .acquire_buffer_with_data(&vec![0u32; CROSS_RETENTION_ELEMS], BufferKind::Scattered)
            .expect("shared");
        CrossRetentionBuffers { input, shared }
    }

    fn cross_retention_copy_pipeline(device: &Device) -> ComputePipeline {
        let shader = ShaderModule::from_slang(device, COPY_SHADER).expect("shader");
        ComputePipeline::new(device, &shader).expect("pipeline")
    }

    fn cross_retention_buffer_writer(
        ctx: &goldy::Context,
        pipeline: &ComputePipeline,
        buffers: &CrossRetentionBuffers,
    ) -> Scheme {
        let mut worker = Scheme::new(ctx);
        worker
            .node("copy", pipeline)
            .with_parcel(&buffers.input, NodeAccess::Read)
            .with_parcel(&buffers.shared, NodeAccess::Write)
            .dispatch(1, 1, 1);
        worker
    }

    fn cross_retention_buffer_reader(ctx: &goldy::Context, shared: &goldy::Buffer) -> (Scheme, WithdrawTransaction) {
        let mut reader = Scheme::new(ctx);
        let grant = MemoryExchange::new(reader.context())
            .bind_withdraw(&mut reader, shared)
            .expect("withdraw");
        (reader, grant)
    }

    fn cross_retention_run_worker_then_reader(
        worker: &mut Scheme,
        reader: &mut Scheme,
        grant: &WithdrawTransaction,
        _ctx: &goldy::Context,
    ) {
        worker.submit().expect("worker submit");
        let mut frame = reader.submit().expect("reader submit");
        let _loan = grant
            .claim(&mut frame)
            .expect("claim")
            .consume()
            .expect("grant consume");
    }

    /// A *topology-visible* foreign reader: a scheme that reads the shared parcel with a
    /// real GPU transfer node (`copy_buffer_parcel`) into its own destination buffer.
    ///
    /// Unlike [`Scheme::grant_read`] (which takes a transient host-visible lease and is
    /// deliberately invisible to cross-submit topology), `copy_buffer_parcel` emits a
    /// `CopyBuffer` node whose `Read` of the shared parcel participates in the net-access
    /// graph. The worker's subsequent `Write` to the same parcel is therefore a WAR hazard
    /// against this foreign read, which the worker must resolve with a baked prologue
    /// barrier — forcing exactly one topology-driven re-record.
    fn cross_retention_copy_reader(ctx: &goldy::Context, shared: &goldy::Buffer) -> (Scheme, goldy::Buffer) {
        let mut pool = RetainedPool::new(Arc::new(ctx.device().clone()));
        let dst = pool
            .acquire_buffer_with_data(&vec![0u32; CROSS_RETENTION_ELEMS], BufferKind::Scattered)
            .expect("copy reader dst");
        let mut reader = Scheme::new(ctx);
        let byte_size = (CROSS_RETENTION_ELEMS * 4) as u64;
        reader
            .copy_buffer_parcel(shared, 0, &dst, 0, byte_size)
            .expect("copy_buffer_parcel");
        (reader, dst)
    }

    fn cross_retention_run_worker_then_copy_reader(worker: &mut Scheme, reader: &mut Scheme) {
        worker.submit().expect("worker submit");
        reader.submit().expect("copy reader submit");
    }

    /// Steady state for a worker observed by a *topology-visible* foreign reader/writer:
    /// one bootstrap record plus exactly one topology/prologue refresh, then resubmits.
    fn assert_worker_cross_reader_steady_state(worker: &Scheme, frames: u64) {
        // `frames` is only consumed by the non-Metal resubmit assertion below.
        let _ = frames;
        assert_eq!(
            worker.replay_stats().records,
            2,
            "bootstrap record + one topology/prologue refresh after foreign reader appears"
        );
        assert_eq!(
            worker.replay_stats().topology_records,
            1,
            "foreign reader on shared parcel must dirty worker topology exactly once"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            worker.replay_stats().resubmit_hits,
            frames.saturating_sub(2),
            "steady-state frames after bootstrap should resubmit"
        );
    }

    /// Steady state for a worker observed only by *topology-invisible* `grant_read` leases:
    /// a single bootstrap record, zero topology refreshes, resubmits forever after.
    fn assert_worker_grant_invisible(worker: &Scheme, frames: u64) {
        // `frames` is only consumed by the non-Metal resubmit assertion below.
        let _ = frames;
        assert_eq!(
            worker.replay_stats().records,
            1,
            "grant_read uses a transient lease and is topology-invisible; worker records once"
        );
        assert_eq!(
            worker.replay_stats().topology_records,
            0,
            "grant_read must not register a topology-visible interaction on the shared parcel"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            worker.replay_stats().resubmit_hits,
            frames.saturating_sub(1),
            "every frame after the bootstrap record resubmits"
        );
    }

    /// `grant_read` is topology-invisible: a foreign scheme grant-reading the shared parcel
    /// every frame never dirties the writer's topology. Cross-submit ordering is handled by
    /// the *reader's* own lease/wait, not by a baked prologue barrier in the worker's CB, so
    /// the worker records exactly once and resubmits thereafter.
    fn cross_scheme_grant_read_reader_is_topology_invisible(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let (mut reader, grant) = cross_retention_buffer_reader(&ctx, &buffers.shared);

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            cross_retention_run_worker_then_reader(&mut worker, &mut reader, &grant, &ctx);
        }

        assert_worker_grant_invisible(&worker, FRAMES);
    }

    /// A topology-visible foreign reader (`copy_buffer_parcel` of the shared parcel) forces
    /// exactly one worker topology record, then the worker resubmits (mirrors a
    /// worker + texture readback path that produces `records == 2`).
    fn cross_scheme_copy_reader_forces_one_topology_record(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let (mut reader, _dst) = cross_retention_copy_reader(&ctx, &buffers.shared);

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            cross_retention_run_worker_then_copy_reader(&mut worker, &mut reader);
        }

        assert_worker_cross_reader_steady_state(&worker, FRAMES);
    }

    /// Write + grant-read in one scheme: no foreign topology edge, so one record only.
    fn single_scheme_write_then_readback_records_once(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut scheme = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &buffers.shared)
            .expect("withdraw");

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            let mut frame = scheme.submit().expect("submit");
            let _loan = grant
                .claim(&mut frame)
                .expect("claim")
                .consume()
                .expect("grant consume");
        }

        assert_eq!(
            scheme.replay_stats().records,
            1,
            "intra-scheme write→read should record once"
        );
        assert_eq!(
            scheme.replay_stats().topology_records,
            0,
            "no foreign scheme touched the parcel"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            FRAMES - 1,
            "subsequent frames should resubmit"
        );
    }

    /// `grant_read` steady state stays at one record for many frames (no thrash, no drift).
    fn cross_scheme_grant_reader_steady_state_stays_at_one(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let (mut reader, grant) = cross_retention_buffer_reader(&ctx, &buffers.shared);

        const FRAMES: u64 = 8;
        for _ in 0..FRAMES {
            cross_retention_run_worker_then_reader(&mut worker, &mut reader, &grant, &ctx);
        }

        assert_worker_grant_invisible(&worker, FRAMES);
    }

    /// After the one-time topology refresh (topology-visible copy reader), additional frames
    /// must not re-record.
    fn cross_scheme_copy_reader_steady_state_does_not_thrash(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let (mut reader, _dst) = cross_retention_copy_reader(&ctx, &buffers.shared);

        const FRAMES: u64 = 8;
        for _ in 0..FRAMES {
            cross_retention_run_worker_then_copy_reader(&mut worker, &mut reader);
        }

        assert_worker_cross_reader_steady_state(&worker, FRAMES);
    }

    /// A foreign writer on the shared parcel causes the same one-time topology record.
    fn cross_scheme_foreign_writer_forces_one_topology_record(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let mut foreign_writer = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            worker.submit().expect("worker submit");
            foreign_writer.submit().expect("foreign writer submit");
        }

        assert_worker_cross_reader_steady_state(&worker, FRAMES);
    }

    /// Submit order changes *when* the worker first observes the foreign read edge, which
    /// changes the raw record counter — but never the steady state.
    ///
    /// - reader-first: the foreign read of `shared` is already registered when the worker
    ///   takes its first record, so the prologue is baked at bootstrap → `records == 1`,
    ///   `topology_records == 0` (no later change).
    /// - worker-first: the worker bootstraps with no foreign edge, then the reader appears
    ///   on frame 2 → one topology refresh → `records == 2`, `topology_records == 1`.
    ///
    /// What is order-*independent* is the steady state: after warmup the worker stops
    /// re-recording and resubmits its retained command buffer.
    fn cross_scheme_submit_order_steady_state_is_stable(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);

        let run = |reader_first: bool| -> (Scheme, CrossRetentionBuffers) {
            let buffers = cross_retention_buffers(&device);
            let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
            let (mut reader, _dst) = cross_retention_copy_reader(&ctx, &buffers.shared);
            const FRAMES: u64 = 4;
            for _ in 0..FRAMES {
                if reader_first {
                    reader.submit().expect("copy reader submit");
                    worker.submit().expect("worker submit");
                } else {
                    cross_retention_run_worker_then_copy_reader(&mut worker, &mut reader);
                }
            }
            // Keep `buffers` alive: dropping retained buffers bound to `worker` marks
            // stamps dead and steady-state submits would return `StaleResource`.
            (worker, buffers)
        };

        for reader_first in [true, false] {
            let (mut worker, _buffers) = run(reader_first);
            let warmup_records = worker.replay_stats().records;
            assert!(
                warmup_records <= 2,
                "warmup costs at most one topology refresh (got {warmup_records} for reader_first={reader_first})"
            );
            // Additional worker-only frames must not add records — the steady state is stable
            // regardless of the order in which the edge was first observed.
            for _ in 0..3 {
                worker.submit().expect("steady-state submit");
            }
            assert_eq!(
                worker.replay_stats().records,
                warmup_records,
                "worker must not re-record in steady state (reader_first={reader_first})"
            );
            assert_eq!(
                worker.replay_stats().topology_records,
                warmup_records - 1,
                "topology refreshes = records beyond the bootstrap (reader_first={reader_first})"
            );
        }
    }

    /// Two topology-visible foreign readers on the same parcel still cost the worker only one
    /// topology record (the interaction set gains *the parcel*, not per-reader edges).
    fn cross_scheme_two_foreign_copy_readers_record_once_then_stable(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let (mut reader_a, _dst_a) = cross_retention_copy_reader(&ctx, &buffers.shared);
        let (mut reader_b, _dst_b) = cross_retention_copy_reader(&ctx, &buffers.shared);

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            worker.submit().expect("worker submit");
            reader_a.submit().expect("reader_a submit");
            reader_b.submit().expect("reader_b submit");
        }

        assert_worker_cross_reader_steady_state(&worker, FRAMES);
    }

    /// Dropping a topology-visible foreign reader does *not* re-dirty the worker.
    ///
    /// Once the worker has baked the cross-submit WAR prologue (after the reader first
    /// appeared), tearing the reader down leaves that prologue in place. A retained
    /// command buffer carrying an extra (now-unobserved) barrier is still correct — an
    /// over-conservative execution/memory dependency is harmless — so the worker keeps
    /// resubmitting without a fresh record. Scheme teardown is therefore a no-op for peer
    /// retention, which is the cheap and safe choice (no thrash on transient observers).
    fn cross_scheme_copy_reader_disappearing_does_not_re_dirty(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let (mut reader, _dst) = cross_retention_copy_reader(&ctx, &buffers.shared);

        for _ in 0..3 {
            cross_retention_run_worker_then_copy_reader(&mut worker, &mut reader);
        }
        assert_eq!(
            worker.replay_stats().records,
            2,
            "reader appearance: bootstrap + one refresh"
        );

        drop(reader);

        for _ in 0..3 {
            worker.submit().expect("worker-only submit");
        }

        assert_eq!(
            worker.replay_stats().records,
            2,
            "reader removal must NOT trigger another record; baked barriers stay valid"
        );
        assert_eq!(
            worker.replay_stats().topology_records,
            1,
            "only the appearance produced a topology record"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            worker.replay_stats().resubmit_hits,
            4,
            "frame 3 of the loop plus three worker-only frames are all resubmits"
        );
    }

    /// Disjoint parcels: a topology-visible foreign reader on an unrelated parcel must not
    /// perturb worker retention.
    fn cross_scheme_disjoint_parcels_never_cross_dirty(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let worker_buffers = cross_retention_buffers(&device);
        let reader_buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &worker_buffers);
        let (mut reader, _dst) = cross_retention_copy_reader(&ctx, &reader_buffers.shared);

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            cross_retention_run_worker_then_copy_reader(&mut worker, &mut reader);
        }

        assert_eq!(
            worker.replay_stats().records,
            1,
            "unrelated foreign reader must not dirty worker"
        );
        assert_eq!(worker.replay_stats().topology_records, 0);
        #[cfg(not(feature = "metal"))]
        assert_eq!(worker.replay_stats().resubmit_hits, FRAMES - 1);
    }

    /// Grant→copy boundary: a foreign scheme that only `grant_read`s the shared parcel keeps
    /// the worker invisible (records == 1); once a *copy* reader of the same parcel appears,
    /// the worker takes exactly one topology record (records == 2). This pins the precise
    /// semantic boundary between the transient-lease path and the transfer-node path.
    fn cross_scheme_grant_then_copy_reader_boundary(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let (mut grant_reader, grant) = cross_retention_buffer_reader(&ctx, &buffers.shared);

        // Phase 1: only a grant_read observer — worker stays at one record.
        for _ in 0..3 {
            cross_retention_run_worker_then_reader(&mut worker, &mut grant_reader, &grant, &ctx);
        }
        assert_eq!(worker.replay_stats().records, 1, "grant_read phase is invisible");
        assert_eq!(worker.replay_stats().topology_records, 0);

        // Phase 2: a copy reader of the same parcel appears — one topology record.
        let (mut copy_reader, _dst) = cross_retention_copy_reader(&ctx, &buffers.shared);
        for _ in 0..3 {
            worker.submit().expect("worker submit");
            copy_reader.submit().expect("copy reader submit");
            let mut frame = grant_reader.submit().expect("grant reader submit");
            let _loan = grant
                .claim(&mut frame)
                .expect("claim")
                .consume()
                .expect("grant consume");
        }
        assert_eq!(
            worker.replay_stats().records,
            2,
            "copy reader appearance dirties worker exactly once; grant reader stays invisible"
        );
        assert_eq!(worker.replay_stats().topology_records, 1);
    }

    /// The transient grant lease returns correct, current data every frame even though it
    /// never re-records the worker — proving the topology-invisible path is also *correct*,
    /// not merely cheap.
    ///
    /// This is the key challenge to the grant semantics: a `grant_read` does not force the
    /// writer to bake a prologue barrier, so one might worry the reader could observe stale
    /// data. It does not: cross-submit ordering for the lease is enforced on the *reader's*
    /// submission (a wait on the worker's last write of the parcel), so each frame's loan
    /// reflects the worker's latest output while the worker stays at a single record.
    fn cross_scheme_grant_read_observes_worker_writes_without_re_record(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let (mut reader, grant) = cross_retention_buffer_reader(&ctx, &buffers.shared);

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            worker.submit().expect("worker submit");
            let mut frame = reader.submit().expect("reader submit");
            let values = read_grant_u32(&grant, &mut frame, CROSS_RETENTION_ELEMS);
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(
                    v, i as u32,
                    "grant lease must observe the worker's copy output at shared[{i}]"
                );
            }
        }

        assert_worker_grant_invisible(&worker, FRAMES);
    }

    /// Retained worker output stays correct after the topology refresh frame.
    fn cross_scheme_retained_worker_after_foreign_reader_reads_correct_values(device: &Device) {
        let ctx = submission_context(&device);
        let pipeline = cross_retention_copy_pipeline(&device);
        let buffers = cross_retention_buffers(&device);

        let mut worker = cross_retention_buffer_writer(&ctx, &pipeline, &buffers);
        let (mut reader, _dst) = cross_retention_copy_reader(&ctx, &buffers.shared);

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            cross_retention_run_worker_then_copy_reader(&mut worker, &mut reader);
        }
        assert_worker_cross_reader_steady_state(&worker, FRAMES);

        let values = read_uploaded_parcel_u32(&ctx, &buffers.shared, CROSS_RETENTION_ELEMS);
        for (i, &val) in values.iter().enumerate() {
            assert_eq!(val, i as u32, "shared[{i}] must match worker copy source");
        }
    }

    // ─── Cross-scheme texture readback (segfault repro / regression) ─────────────

    const CROSS_RETENTION_TEX_W: u32 = 16;
    const CROSS_RETENTION_TEX_H: u32 = 16;

    fn cross_retention_texture(device: &Device) -> goldy::Texture {
        RetainedPool::new(Arc::new(device.clone()))
            .acquire_texture(
                CROSS_RETENTION_TEX_W,
                CROSS_RETENTION_TEX_H,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_SRC,
                None,
            )
            .expect("texture")
    }

    fn cross_retention_texture_writer(
        ctx: &goldy::Context,
        pipeline: &ComputePipeline,
        texture: &goldy::Texture,
    ) -> Scheme {
        let mut worker = Scheme::new(ctx);
        worker
            .node("write_tex", pipeline)
            .with_parcel(texture, NodeAccess::Write)
            .dispatch(CROSS_RETENTION_TEX_W.div_ceil(8), CROSS_RETENTION_TEX_H.div_ceil(8), 1);
        worker
    }

    /// A retained reader scheme that withdraws the shared texture via MemoryExchange.
    /// The withdraw is topology-invisible, but we still bind a copy_texture reader elsewhere
    /// when testing topology refresh; this helper uses withdraw for observation.
    fn cross_retention_texture_reader(ctx: &goldy::Context, texture: &goldy::Texture) -> (Scheme, WithdrawTransaction) {
        let mut reader = Scheme::new(ctx);
        let grant = MemoryExchange::new(reader.context())
            .bind_withdraw(&mut reader, texture)
            .expect("withdraw");
        (reader, grant)
    }

    /// Cross-scheme texture readback under retention: a worker writes a `Direct` (storage)
    /// texture every frame and a separate reader scheme withdraws it, retained across frames.
    ///
    /// History: this scenario previously aborted the test process with
    /// `STATUS_ACCESS_VIOLATION`. The proximate cause was the texture upload/transition path
    /// driving a storage image (no `SAMPLED` usage) into `SHADER_READ_ONLY_OPTIMAL`
    /// (VUID-VkImageMemoryBarrier2-oldLayout-01211), leaving the image in a layout the
    /// driver could not legally consume on the retained resubmit. Storage textures now
    /// settle to `GENERAL` (see `texture.rs::settled_shader_read_layout`). This test guards
    /// against a regression of that crash. Withdraw is topology-invisible, so the worker
    /// records once (bootstrap only) — unlike the old `copy_texture` reader.
    fn cross_scheme_texture_readback_retained_loop_records_twice(device: &Device) {
        let ctx = submission_context(&device);
        let shader = ShaderModule::from_slang(&device, WRITE_TEXTURE_SHADER).expect("texture shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("texture pipeline");
        let texture = cross_retention_texture(&device);

        let mut worker = cross_retention_texture_writer(&ctx, &pipeline, &texture);
        let (mut reader, grant) = cross_retention_texture_reader(&ctx, &texture);

        const FRAMES: u64 = 4;
        let mut last_pixels = Vec::new();
        for _ in 0..FRAMES {
            worker.submit().expect("worker submit");
            let mut frame = reader.submit().expect("reader submit");
            last_pixels = grant
                .claim(&mut frame)
                .expect("claim")
                .consume()
                .expect("consume")
                .to_vec();
        }

        assert!(
            last_pixels.iter().any(|&b| b != 0),
            "texture readback must observe writes"
        );

        // Withdraw is topology-invisible: worker only bootstrap-records.
        assert_eq!(
            worker.replay_stats().records,
            1,
            "texture writer + withdraw reader: bootstrap record only"
        );
        assert_eq!(worker.replay_stats().topology_records, 0);
    }

    /// Returning a transient texture that a scheme still binds must fail subsequent submit.
    ///
    /// This is the GPU counterpart of the mock unit test
    /// `return_transient_texture_invalidates_bound_scheme` in `scheme.rs`.
    ///
    /// Covered on every enabled backend feature (`metal`, `vulkan`, `dx12`). Locally we
    /// typically only have Metal; run with `--features vulkan` / `--features dx12` on a
    /// machine that has those drivers to confirm stamp retirement on those paths.
    fn scheme_return_transient_texture_invalidates_retained_scheme(device: &Device) {
        let ctx = submission_context(&device);
        let shader = ShaderModule::from_slang(&device, WRITE_TEXTURE_SHADER).expect("texture shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("texture pipeline");

        let tex = ctx
            .acquire_transient_texture(
                8,
                8,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::empty(),
            )
            .expect("transient texture");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_tex", &pipeline)
            .with_parcel(&tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.submit().expect("first submit");

        // Expected scratch lifecycle: return after the frame that used it.
        ctx.return_transient_texture(tex);

        assert!(
            matches!(scheme.submit(), Err(goldy::GoldyError::StaleResource)),
            "retained resubmit after return_transient_texture must return StaleResource \
             (Metal/Vulkan/DX12); returning a bound transient ends its deed"
        );
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

        trial!(scheme_graph_linear_chain);
        trial!(scheme_graph_independent_dispatches);
        trial!(scheme_graph_diamond_dependency);
        trial!(scheme_graph_fill_readback);
        trial!(scheme_zeros_then_dispatch_reads_zeros);
        trial!(scheme_write_then_dispatch_reads_uploaded_data);
        trial!(scheme_stress_zeros_then_dispatch_large);
        trial!(scheme_stress_many_zero_writes_many_dispatches);
        trial!(scheme_stress_write_then_dispatch_chain);
        trial!(scheme_stress_two_phase_submission);
        trial!(scheme_stress_rapid_submissions);
        trial!(scheme_compute_dispatch_empty);
        trial!(scheme_compute_write_and_readback);
        trial!(scheme_compute_with_uav_parcel);
        trial!(scheme_compute_with_srv_and_uav_parcels);
        trial!(scheme_parcel_write_zeros_full);
        trial!(scheme_parcel_write_zeros_partial);
        trial!(scheme_parcel_write_zeros_to_end);
        trial!(scheme_zeros_before_copy_dispatch);
        trial!(scheme_write_to_parcel_zeros_between_submissions);
        trial!(scheme_upload_frame_unwaited_serializes_before_worker_submit);
        trial!(scheme_compute_many_resource_slots);
        trial!(scheme_regular_buffer_write_then_copy);
        trial!(scheme_transient_buffer_write_then_copy);
        trial!(scheme_transient_buffer_recycling);
        trial!(scheme_compute_with_struct_buffer);
        trial!(scheme_cpu_readable_compute_write_and_read);
        trial!(scheme_cpu_readable_write_to_parcel_roundtrip);
        trial!(scheme_scattered_typed_variable_assignment);
        trial!(scheme_compute_write_to_texture);
        trial!(scheme_with_parcel_raw_texture);
        trial!(scheme_wave_inclusive_scan_uniform_64);
        trial!(scheme_wave_inclusive_scan_ramp_64);
        trial!(scheme_wave_inclusive_scan_uniform_256);
        trial!(scheme_workgroup_reduce_uint_correct);
        trial!(scheme_workgroup_inclusive_scan_uint_correct);
        trial!(scheme_workgroup_broadcast_correct);
        trial!(scheme_workgroup_upper_bound_linear);
        trial!(scheme_texture_dual_view_round_trip);
        trial!(scheme_two_contexts_both_submit_and_complete);
        trial!(scheme_two_contexts_reclaim_independently);
        trial!(scheme_deposit_buffer_reuse_across_submissions);
        trial!(scheme_uniform_param_uint_roundtrip);
        trial!(scheme_uniform_param_uint_zero);
        trial!(scheme_uniform_param_uint_max);
        trial!(scheme_uniform_param_float_reinterpret);
        trial!(scheme_uniform_two_independent_scalar_params);
        trial!(scheme_uniform_scalar_after_two_buffer_params);
        trial!(scheme_buffer_view_copy_between_sub_regions);
        trial!(scheme_buffer_view_isolation);
        trial!(scheme_compute_dispatch_indirect);
        trial!(scheme_dispatch_indirect_invalid_buffer);
        trial!(scheme_dispatch_indirect_wrong_type_rejected);
        trial!(scheme_stress_zeros_then_indirect_dispatch);
        trial!(scheme_stress_alternating_write_dispatch);
        trial!(scheme_clear_parcel_full);
        trial!(scheme_clear_parcel_partial_preserves_edges);
        trial!(scheme_clear_parcel_size_zero_fills_to_end);
        trial!(scheme_clear_parcel_requires_buffer_parcel);
        trial!(scheme_copy_buffer_parcel_basic);
        trial!(scheme_copy_buffer_parcel_partial_with_offsets);
        trial!(scheme_copy_buffer_parcel_rejects_texture_src);
        trial!(scheme_copy_buffer_parcel_rejects_texture_dst);
        trial!(scheme_copy_buffer_parcel_resubmit_does_not_rerecord);
        trial!(scheme_cpu_writable_staging_write_then_copy);
        trial!(scheme_cpu_writable_staging_update_each_frame);
        trial!(scheme_write_texture_round_trip);
        trial!(scheme_write_texture_wrong_size_returns_error);
        trial!(scheme_write_texture_marks_scheme_dirty);
        trial!(scheme_write_texture_region_round_trip);
        trial!(scheme_write_texture_region_oob_returns_error);
        trial!(scheme_write_texture_region_multiple_non_overlapping);
        trial!(scheme_copy_buffer_to_texture_parcel_full_texture);
        trial!(scheme_copy_buffer_to_texture_parcel_oob_returns_error);
        trial!(scheme_copy_buffer_to_texture_parcel_rejects_texture_src);
        trial!(scheme_copy_buffer_to_texture_parcel_resubmit_is_retained);
        trial!(scheme_copy_buffer_to_texture_pitched_retained_after_layout_settles);
        trial!(cross_scheme_grant_read_reader_is_topology_invisible);
        trial!(cross_scheme_copy_reader_forces_one_topology_record);
        trial!(single_scheme_write_then_readback_records_once);
        trial!(cross_scheme_grant_reader_steady_state_stays_at_one);
        trial!(cross_scheme_copy_reader_steady_state_does_not_thrash);
        trial!(cross_scheme_foreign_writer_forces_one_topology_record);
        trial!(cross_scheme_submit_order_steady_state_is_stable);
        trial!(cross_scheme_two_foreign_copy_readers_record_once_then_stable);
        trial!(cross_scheme_copy_reader_disappearing_does_not_re_dirty);
        trial!(cross_scheme_disjoint_parcels_never_cross_dirty);
        trial!(cross_scheme_grant_then_copy_reader_boundary);
        trial!(cross_scheme_grant_read_observes_worker_writes_without_re_record);
        trial!(cross_scheme_retained_worker_after_foreign_reader_reads_correct_values);
        trial!(cross_scheme_texture_readback_retained_loop_records_twice);
        trial!(scheme_return_transient_texture_invalidates_retained_scheme);

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
