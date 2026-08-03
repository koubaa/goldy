//! Retained-scheme integration tests — phase-1 exit-gate coverage of
//! `retained-scheme/project.md` §1.3.
//!
//! These assert the retained-scheme contract against real GPU backends:
//!
//! - **Cross-scheme serialization**: per-submission CPU data enters through a
//!   *separate* upload submission; the retained worker scheme is never mutated
//!   and resubmits as pure retention hits.
//! - **Retention recovery**: clean submissions resubmit without re-record.
//! - **Selector advance**: submission order itself carries per-submission information.
//! - **Lease N=1**: scheme-held texture leases retain across resubmit; backing returns to
//!   the transient pool on scheme drop.
//!
//! Anti-pattern policy (reviewed June 2026): no `gpu_progress()`, no
//! `clear()`+rebuild, no raw bindless indices (goldy#210), no untimed parcel
//! mutation (goldy#211), parcels bound as parcels.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

#[path = "common/submission.rs"]
mod submission;
#[path = "common/upload.rs"]
mod upload;

use goldy::{
    types::{BufferFlags, DispatchShape},
    BackendType, BufferKind, ComputePipeline, Context, Device, DeviceDescriptor, Instance, MemoryExchange,
    NodeAccess, Parcel, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, Submission, TextureFlags,
    TextureFormat, TextureKind, WithdrawTransaction,
};
use std::sync::Arc;
use submission::submission_context;

fn read_grant_u32(grant: &WithdrawTransaction, submission: &mut Submission, count: usize) -> Vec<u32> {
    let loan = grant
        .claim(submission)
        .expect("claim")
        .consume()
        .expect("withdraw consume");
    assert_eq!(loan.len(), count * 4, "grant readback size");
    bytemuck::cast_slice(&loan).to_vec()
}

fn make_device() -> (Device, goldy::test_support::CbReuseOverride) {
    // Retention contract tests must not flip under GOLDY_DISABLE_CB_REUSE=1.
    let cb = goldy::test_support::CbReuseOverride::force_enabled();
    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("Failed to request adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("Failed to create device");
    (device, cb)
}

/// CUDA writable `DirectSpatial<float4>` requires storage-compatible `Rgba32Float`.
fn writable_texture_format(device: &Device) -> TextureFormat {
    if device.backend_type() == BackendType::Cuda {
        TextureFormat::Rgba32Float
    } else {
        TextureFormat::Rgba8Unorm
    }
}

fn assert_solid_red_texel(loan: &[u8], is_cuda: bool, label: &str) {
    assert!(!loan.is_empty(), "texture readback empty ({label})");
    if is_cuda {
        let floats: &[f32] = bytemuck::cast_slice(loan);
        assert_eq!(floats[0], 1.0, "R channel ({label})");
        assert_eq!(floats[1], 0.0, "G channel ({label})");
        assert_eq!(floats[2], 0.0, "B channel ({label})");
        assert_eq!(floats[3], 1.0, "A channel ({label})");
    } else {
        assert_eq!(loan[0], 255, "R channel ({label})");
        assert_eq!(loan[1], 0, "G channel ({label})");
        assert_eq!(loan[2], 0, "B channel ({label})");
        assert_eq!(loan[3], 255, "A channel ({label})");
    }
}

/// Copy input → output. Both parcels are declared in the scheme.
const COPY_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(8, 1, 1)]
void cs_main(BufRO<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x];
}
"#;

/// Cross-scheme serialization: per-submission data enters through a separate upload
/// submission; the worker scheme is recorded once and never mutated.
///
/// The worker scheme (`records == 1`; `resubmit_hits == N-1` on non-Metal backends) observes each frame's
/// data through the shared input parcel — serialized by queue order on the same context.
/// Upload uses a persistent upload [`Scheme`] (same `scheme_id` each frame) so foreign topology
/// registration does not churn the retained worker.
#[test]
fn upload_graph_feeds_retained_worker_without_rerecord() {
    let (device, _cb) = make_device();
    eprintln!("retained_replay backend: {:?}", device.backend_type());
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile copy shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let input = pool
        .acquire_buffer_with_data(&[0u32; 8], BufferKind::Scattered)
        .expect("input parcel");
    let output = pool
        .acquire_buffer_with_data(&[0u32; 8], BufferKind::Scattered)
        .expect("output parcel");

    // Worker: recorded once, never mutated again.
    let mut worker = Scheme::new(&ctx);
    worker
        .node("copy", &pipeline)
        .with_parcel(&input, NodeAccess::Read)
        .with_parcel(&output, NodeAccess::Write)
        .dispatch(1, 1, 1);

    let grant = MemoryExchange::new(worker.context())
        .bind_withdraw(&mut worker, &output)
        .expect("withdraw");
    const FRAMES: u32 = 3;
    let mut upload = Scheme::new(&ctx);
    let deposit = upload::bind_upload_deposit(&ctx, &mut upload, &input, (8 * std::mem::size_of::<u32>()) as u64)
        .expect("bind deposit");
    for submission in 1..=FRAMES {
        // Separate upload submission per frame via a persistent upload scheme.
        upload::upload_parcel(&mut upload, &deposit, bytemuck::cast_slice(&[submission; 8])).expect("upload_parcel");

        let mut frame = worker.submit().expect("submit worker");
        for v in read_grant_u32(&grant, &mut frame, 8) {
            assert_eq!(
                v, submission,
                "submission {submission} must observe its upload (cross-scheme serialization)"
            );
        }
    }

    assert_eq!(worker.replay_stats().records, 1, "the worker records exactly once");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        worker.replay_stats().resubmit_hits,
        u64::from(FRAMES) - 1,
        "submissions after the first are retention hits",
    );
}

/// Logical upload buffers feed a retained worker across pipelined frames without host waits.
#[test]
fn deposit_feeds_retained_worker_across_frames() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile copy shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let input = pool
        .acquire_buffer_with_data(&[0u32; 8], BufferKind::Scattered)
        .expect("input parcel");
    let output = pool
        .acquire_buffer_with_data(&[0u32; 8], BufferKind::Scattered)
        .expect("output parcel");

    let mut worker = Scheme::new(&ctx);
    worker
        .node("copy", &pipeline)
        .with_parcel(&input, NodeAccess::Read)
        .with_parcel(&output, NodeAccess::Write)
        .dispatch(1, 1, 1);
    let grant = MemoryExchange::new(worker.context())
        .bind_withdraw(&mut worker, &output)
        .expect("withdraw");

    let mut upload = Scheme::new(&ctx);
    let memory = MemoryExchange::new(&ctx);
    let staging = memory
        .bind_deposit_buffer(&mut upload, input.whole(), (8 * std::mem::size_of::<u32>()) as u64)
        .expect("declare deposit");

    const FRAMES: u32 = 4;
    for submission in 1..=FRAMES {
        let data = [submission; 8];
        staging
            .write(&mut upload, 0, bytemuck::cast_slice(&data))
            .expect("stage deposit");
        let _ = upload.submit().expect("submit upload");
        let mut frame = worker.submit().expect("submit worker");
        for v in read_grant_u32(&grant, &mut frame, 8) {
            assert_eq!(v, submission, "frame {submission} must observe its staged payload");
        }
    }

    assert_eq!(worker.replay_stats().records, 1, "worker records once");
    assert!(
        upload.deposit_parcel_count(&staging) >= 1,
        "upload buffer must own at least one physical parcel"
    );
    // After GPU retirement, subsequent stages should not unbounded-grow forever.
    // Mock/backends that complete promptly settle back to one warm parcel.
    assert!(
        upload.deposit_parcel_count(&staging) <= FRAMES as usize,
        "parcel count must be bounded by in-flight depth"
    );
}

/// Copy-only scheme: pre-initialized input, no upload — retention hit on submission 1.
#[test]
fn clean_scheme_resubmits_without_rerecord() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile copy shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let input = pool
        .acquire_buffer_with_data(&[1u32; 8], BufferKind::Scattered)
        .expect("input parcel");
    let output = pool
        .acquire_buffer_with_data(&[0u32; 8], BufferKind::Scattered)
        .expect("output parcel");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("copy", &pipeline)
        .with_parcel(&input, NodeAccess::Read)
        .with_parcel(&output, NodeAccess::Write)
        .dispatch(1, 1, 1);

    scheme.submit().expect("submit 0");
    let mut frame = scheme.submit().expect("submit 1");
    frame.wait_until_settled().expect("wait");
    assert!(output.is_settled(), "completed work must leave parcel settled");

    assert_eq!(scheme.replay_stats().records, 1, "exactly one record");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        scheme.replay_stats().resubmit_hits,
        1,
        "submission 1 must be a zero-record retention hit"
    );
}

const WRITE_DISPATCH_SHAPE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<DispatchShape> shape, ThreadId id) {
    DispatchShape s;
    s.x = 1;
    s.y = 1;
    s.z = 1;
    shape[0] = s;
}
"#;

const DOUBLE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
"#;

/// Indirect-dispatch scheme records once, then resubmits without re-record.
#[test]
fn indirect_scheme_resubmits_without_rerecord() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);

    let write_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, WRITE_DISPATCH_SHAPE_SHADER).expect("compile write shape shader"),
    )
    .expect("create write pipeline");
    let work_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("compile double shader"),
    )
    .expect("create work pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let shape = pool
        .acquire_buffer_sized::<DispatchShape>(1, BufferKind::Scattered, BufferFlags::empty())
        .expect("shape buffer");
    let work = pool
        .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
        .expect("work buffer");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("write_shape", &write_pipe)
        .with_parcel(&shape, NodeAccess::Write)
        .dispatch(1, 1, 1);
    scheme
        .node("work", &work_pipe)
        .with_parcel(&work, NodeAccess::Write)
        .dispatch_shape_parcel(&*shape)
        .expect("indirect dispatch");

    scheme.submit().expect("submit 0");
    scheme.submit().expect("submit 1");

    assert_eq!(scheme.replay_stats().records, 1, "exactly one record");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        scheme.replay_stats().resubmit_hits,
        1,
        "submission 1 must be a zero-record retention hit"
    );
}

/// Selector-advance: a recorded single-thread node increments a GPU-side counter.
///
/// Submitting the *identical* scheme N times must advance the counter N times —
/// submission order carries per-submission information with no CPU involvement.
/// This is the prologue-selector mechanism of the frame table in isolation.
const SELECTOR_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> sel, ThreadId id) {
    if (id.x == 0) {
        sel[0] = sel[0] + 1;
    }
}
"#;

#[test]
fn selector_advances_across_identical_submissions() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, SELECTOR_SHADER).expect("compile selector shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let selector = pool
        .acquire_buffer_with_data(&[0u32], BufferKind::Scattered)
        .expect("selector parcel");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("bump_selector", &pipeline)
        .with_parcel(&selector, NodeAccess::ReadWrite)
        .dispatch(1, 1, 1);

    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &selector)
        .expect("withdraw");
    const N: u64 = 5;
    let mut last_frame = None;
    for _ in 0..N {
        last_frame = Some(scheme.submit().expect("submit"));
    }
    let mut frame = last_frame.expect("submit");
    let count = read_grant_u32(&grant, &mut frame, 1)[0];
    assert_eq!(
        count as u64, N,
        "each submission must advance the GPU-side selector once"
    );

    assert_eq!(scheme.replay_stats().records, 1, "one record, then pure resubmits");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        scheme.replay_stats().resubmit_hits,
        N - 1,
        "remaining submissions are retention hits"
    );
}

/// Two independent schemes sharing one context must not evict each other's retained
/// state. Both copy-shaders must produce correct results across interleaved submissions.
#[test]
fn two_schemes_on_one_context_do_not_collide() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile copy shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));

    // Scheme A: copies [1u32; 8] → out_a
    let in_a = pool
        .acquire_buffer_with_data(&[1u32; 8], BufferKind::Scattered)
        .expect("in_a");
    let out_a = pool
        .acquire_buffer_with_data(&[0u32; 8], BufferKind::Scattered)
        .expect("out_a");
    let mut scheme_a = Scheme::new(&ctx);
    scheme_a
        .node("copy_a", &pipeline)
        .with_parcel(&in_a, NodeAccess::Read)
        .with_parcel(&out_a, NodeAccess::Write)
        .dispatch(1, 1, 1);

    // Scheme B: copies [2u32; 8] → out_b
    let in_b = pool
        .acquire_buffer_with_data(&[2u32; 8], BufferKind::Scattered)
        .expect("in_b");
    let out_b = pool
        .acquire_buffer_with_data(&[0u32; 8], BufferKind::Scattered)
        .expect("out_b");
    let mut scheme_b = Scheme::new(&ctx);
    scheme_b
        .node("copy_b", &pipeline)
        .with_parcel(&in_b, NodeAccess::Read)
        .with_parcel(&out_b, NodeAccess::Write)
        .dispatch(1, 1, 1);

    let grant_a = MemoryExchange::new(scheme_a.context())
        .bind_withdraw(&mut scheme_a, &out_a)
        .expect("withdraw");
    let grant_b = MemoryExchange::new(scheme_b.context())
        .bind_withdraw(&mut scheme_b, &out_b)
        .expect("withdraw");

    // Interleave submissions: A, B, A, B
    let _ = scheme_a.submit().expect("a1");
    let _ = scheme_b.submit().expect("b1");
    let mut frame_a = scheme_a.submit().expect("a2");
    let mut frame_b = scheme_b.submit().expect("b2");

    for v in read_grant_u32(&grant_a, &mut frame_a, 8) {
        assert_eq!(v, 1u32, "scheme_a must produce 1s");
    }
    for v in read_grant_u32(&grant_b, &mut frame_b, 8) {
        assert_eq!(v, 2u32, "scheme_b must produce 2s");
    }

    assert_eq!(scheme_a.replay_stats().records, 1, "scheme_a records once");
    assert_eq!(scheme_b.replay_stats().records, 1, "scheme_b records once");
}

/// Write a solid color into a scheme-held texture lease.
const LEASE_TEXTURE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(DirectSpatial<float4> dst, ThreadId id) {
    if (id.x == 0 && id.y == 0) {
        dst[uint2(0, 0)] = float4(1.0, 0.0, 0.0, 1.0);
    }
}
"#;

/// Scheme-held texture lease: recorded once, then pure retention hits on resubmit.
#[test]
fn lease_texture_scheme_resubmits_without_rerecord() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, LEASE_TEXTURE_SHADER).expect("compile texture shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let mut scheme = Scheme::new(&ctx);
    let lease = scheme
        .lease_texture(
            4,
            4,
            writable_texture_format(&device),
            TextureKind::DirectInterpolated,
            TextureFlags::empty(),
        )
        .expect("lease texture");
    scheme
        .node("write_tex", &pipeline)
        .with_parcel(&lease, NodeAccess::Write)
        .dispatch(1, 1, 1);

    scheme.submit().expect("submit 0");
    scheme.submit().expect("submit 1");
    scheme.submit().expect("submit 2");

    assert_eq!(scheme.replay_stats().records, 1, "exactly one record");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        scheme.replay_stats().resubmit_hits,
        2,
        "remaining submits are retention hits",
    );
}

/// Dropping a scheme returns leased texture backing to the context transient pool.
#[test]
fn lease_backing_pool_hygiene() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);

    let outstanding_before = ctx.transient_outstanding_bytes().texture;
    let alloc_count_before = ctx.transient_texture_alloc_count();

    {
        let mut scheme = Scheme::new(&ctx);
        let _lease = scheme
            .lease_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::DirectInterpolated,
                TextureFlags::empty(),
            )
            .expect("lease texture");
        assert_eq!(
            ctx.transient_texture_alloc_count(),
            alloc_count_before + 1,
            "first lease allocates fresh backing"
        );
        assert!(
            ctx.transient_outstanding_bytes().texture > outstanding_before,
            "leased backing counts as pool outstanding"
        );
    }

    assert_eq!(
        ctx.transient_outstanding_bytes().texture,
        outstanding_before,
        "outstanding drops when scheme releases lease backings"
    );

    let mut scheme2 = Scheme::new(&ctx);
    let _lease2 = scheme2
        .lease_texture(
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureKind::DirectInterpolated,
            TextureFlags::empty(),
        )
        .expect("re-lease texture");
    assert_eq!(
        ctx.transient_texture_alloc_count(),
        alloc_count_before + 1,
        "re-lease reused parked backing instead of allocating"
    );
    assert!(
        ctx.transient_outstanding_bytes().texture > outstanding_before,
        "re-leased backing is outstanding again"
    );
}

const FILL_42_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> buf, ThreadId id) {
    buf[id.x] = 42;
}
"#;

/// Grant readback with N-backing: submit K and K+1 without waiting; both frames read correctly.
#[test]
fn withdraw_concurrent_frames_distinct_backings() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);

    let pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, FILL_42_SHADER).expect("compile fill shader"),
    )
    .expect("create pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fill", &pipe)
        .with_parcel(&buf, NodeAccess::Write)
        .dispatch(1, 1, 1);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &buf)
        .expect("withdraw");

    let mut frame1 = scheme.submit().expect("submit K");
    let mut frame2 = scheme.submit().expect("submit K+1 without waiting on K");

    for frame in [&mut frame1, &mut frame2] {
        let loan = grant.claim(frame).expect("claim").consume().expect("withdraw consume");
        assert_eq!(loan.len(), 64 * 4);
        for chunk in loan.chunks_exact(4) {
            assert_eq!(u32::from_le_bytes(chunk.try_into().unwrap()), 42);
        }
    }

    let stats = scheme.replay_stats();
    drop(scheme);

    assert_eq!(stats.records, 1, "dispatch records once");
    #[cfg(not(feature = "metal"))]
    assert_eq!(stats.resubmit_hits, 1, "second dispatch submit is a retention hit");
}

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

/// Texture grant readback with N-backing: submit K and K+1 without waiting; both frames read correctly.
#[test]
fn withdraw_texture_concurrent_frames_distinct_backings() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);
    let is_cuda = device.backend_type() == BackendType::Cuda;

    let shader = ShaderModule::from_slang(&device, WRITE_TEXTURE_SHADER).expect("compile texture shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let width = 16u32;
    let height = 16u32;
    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let texture = pool
        .acquire_texture(
            width,
            height,
            writable_texture_format(&device),
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
            None,
        )
        .expect("texture parcel");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("write_tex", &pipeline)
        .with_parcel(&texture, NodeAccess::Write)
        .dispatch(wg_x, wg_y, 1);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &texture)
        .expect("withdraw");

    let mut frame1 = scheme.submit().expect("submit K");
    let mut frame2 = scheme.submit().expect("submit K+1 without waiting on K");

    for (i, frame) in [&mut frame1, &mut frame2].into_iter().enumerate() {
        let loan = grant.claim(frame).expect("claim").consume().expect("withdraw consume");
        assert_solid_red_texel(&loan, is_cuda, &format!("frame {i}"));
    }

    let stats = scheme.replay_stats();
    drop(scheme);

    assert_eq!(stats.records, 1, "dispatch records once");
    #[cfg(not(feature = "metal"))]
    assert_eq!(stats.resubmit_hits, 1, "second dispatch submit is a retention hit");
}

fn fill_42_pipeline(device: &Device) -> ComputePipeline {
    let shader = ShaderModule::from_slang(device, FILL_42_SHADER).expect("compile fill shader");
    ComputePipeline::new(device, &shader).expect("create pipeline")
}

fn fill_42_scheme(ctx: &Context, pipe: &ComputePipeline, buf: &Parcel) -> Scheme {
    let mut scheme = Scheme::new(ctx);
    scheme
        .node("fill", pipe)
        .with_parcel(buf, NodeAccess::Write)
        .dispatch(1, 1, 1);
    scheme
}

/// Second claim on the same submission must fail (withdraw slot is taken exactly once).
#[test]
fn withdraw_double_read_same_frame_errors() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = fill_42_scheme(&ctx, &pipe, &buf);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &buf)
        .expect("withdraw");
    let mut frame = scheme.submit().expect("submit");

    let _loan = grant.claim(&mut frame).expect("claim").consume().expect("first read");
    let err = grant.claim(&mut frame).expect_err("second claim must fail");
    assert!(err.to_string().contains("already consumed"), "unexpected error: {err}");
}

/// After the first claim takes the withdraw slot, a second claim fails.
#[test]
fn withdraw_second_consume_errors() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = fill_42_scheme(&ctx, &pipe, &buf);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &buf)
        .expect("withdraw");
    let mut frame = scheme.submit().expect("submit");

    let _loan = grant.claim(&mut frame).expect("claim").consume().expect("first read");
    let err = grant.claim(&mut frame).expect_err("second claim must fail");
    assert!(err.to_string().contains("already consumed"), "unexpected error: {err}");
}

/// Grant with no producing dispatch copies parcel bytes as-is (zero-initialized here so
/// shared-device heap reuse does not inject stale fill_42 contents from prior tests).
#[test]
fn withdraw_without_producing_dispatch_reads_zeros() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    const GRANT_ZERO_TEST_U32S: usize = 64;
    const GRANT_ZERO_TEST_BYTES: u64 = (GRANT_ZERO_TEST_U32S as u64) * 4;
    let zeros = vec![0u8; GRANT_ZERO_TEST_BYTES as usize];
    let buf = pool
        .acquire_buffer(
            GRANT_ZERO_TEST_BYTES,
            BufferKind::Scattered,
            None,
            BufferFlags::empty(),
            Some(&zeros),
        )
        .expect("output parcel");

    let mut scheme = Scheme::new(&ctx);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &buf)
        .expect("withdraw");
    let mut frame = scheme.submit().expect("submit");
    let values = read_grant_u32(&grant, &mut frame, GRANT_ZERO_TEST_U32S);
    assert!(
        values.iter().all(|&v| v == 0),
        "expected zeros without a producer dispatch"
    );
}

/// Grant node before dispatch in IR still reads post-dispatch bytes — copy runs after all dispatches.
#[test]
fn withdraw_before_dispatch_node_still_reads_producer_output() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = Scheme::new(&ctx);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &buf)
        .expect("withdraw");
    scheme
        .node("fill", &pipe)
        .with_parcel(&buf, NodeAccess::Write)
        .dispatch(1, 1, 1);
    let mut frame = scheme.submit().expect("submit");
    let values = read_grant_u32(&grant, &mut frame, 64);
    assert!(
        values.iter().all(|&v| v == 42),
        "grant before dispatch in IR still sees fill output"
    );
}

/// Dropping a frame without reading returns staging; a later submission can still be read.
#[test]
fn withdraw_drop_frame_without_read_then_submit_and_read() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = fill_42_scheme(&ctx, &pipe, &buf);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &buf)
        .expect("withdraw");

    let mut frame1 = scheme.submit().expect("submit 1");
    drop(frame1);

    let mut frame2 = scheme.submit().expect("submit 2 after frame1 drop");
    let values = read_grant_u32(&grant, &mut frame2, 64);
    assert!(
        values.iter().all(|&v| v == 42),
        "second frame after dropped unread frame1"
    );
}

/// Texture grant: sequential wait-then-resubmit must produce correct data on every frame,
/// including retained resubmits that re-execute `CopyTextureToReadback`.
///
/// This is the regression test for the DX12 `last_layout` tracking bug: if
/// `record_copy_texture_to_readback` does not update the texture's tracked layout after
/// the restore barrier, a subsequent copy will compute the wrong `layout_before` and emit
/// a barrier with an incorrect source layout, potentially producing corrupt data or a
/// validation error on the second submission.
///
/// The test waits on each frame before the next submit so the second submit is always a
/// retained resubmit (`records == 1`, `resubmit_hits >= 1`), exercising the path where
/// `finish_submit_frame` issues `CopyTextureToReadback` on an already-used, retained
/// command list.
#[test]
fn withdraw_texture_sequential_resubmit_correct_data() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);
    let is_cuda = device.backend_type() == BackendType::Cuda;

    let shader = ShaderModule::from_slang(&device, WRITE_TEXTURE_SHADER).expect("compile texture shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let width = 16u32;
    let height = 16u32;
    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let texture = pool
        .acquire_texture(
            width,
            height,
            writable_texture_format(&device),
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
            None,
        )
        .expect("texture parcel");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("write_tex", &pipeline)
        .with_parcel(&texture, NodeAccess::Write)
        .dispatch(wg_x, wg_y, 1);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &texture)
        .expect("withdraw");

    // Three sequential submits: wait and read each one before the next.
    // The second and third submits are retained resubmits. Each triggers
    // a new CopyTextureToReadback in finish_submit_frame. The DX12 backend
    // must have the correct last_layout going into the copy barrier each time.
    for round in 0..3u32 {
        let mut frame = scheme.submit().expect("submit");
        let loan = grant.claim(&mut frame).expect("claim").consume().expect("grant read");
        assert_solid_red_texel(&loan, is_cuda, &format!("round {round}"));
    }

    let stats = scheme.replay_stats();
    assert_eq!(stats.records, 1, "dispatch records exactly once");
    #[cfg(not(feature = "metal"))]
    assert!(stats.resubmit_hits >= 2, "rounds 2 and 3 must be retention hits");
}

/// Many consecutive submits with dropped unread frames must not exhaust staging (pool recycles on frame drop).
#[test]
fn withdraw_many_dropped_frames_without_read_then_read_succeeds() {
    let (device, _cb) = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = fill_42_scheme(&ctx, &pipe, &buf);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &buf)
        .expect("withdraw");

    for _ in 0..8 {
        drop(scheme.submit().expect("submit with dropped frame"));
    }
    let mut frame = scheme.submit().expect("final submit");
    let values = read_grant_u32(&grant, &mut frame, 64);
    assert!(
        values.iter().all(|&v| v == 42),
        "read succeeds after many dropped unread frames (staging pool must recycle)"
    );
}
