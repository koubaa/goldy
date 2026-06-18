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
    types::{BufferFlags, ResourceAccess},
    BufferKind, ComputePipeline, Context, Device, DeviceDescriptor, Grant, GrantBuffer, Instance, NodeAccess, Parcel,
    ReadGrant, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, Submission, TextureFlags, TextureFormat,
    TextureKind,
};
use std::sync::Arc;
use submission::submission_context;

fn read_grant_u32(grant: &ReadGrant<GrantBuffer>, submission: &Submission, count: usize) -> Vec<u32> {
    let loan = grant.consume(submission).expect("grant consume");
    assert_eq!(loan.len(), count * 4, "grant readback size");
    bytemuck::cast_slice(&loan).to_vec()
}

fn make_device() -> Device {
    let instance = Instance::new().expect("Failed to create instance");
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("Failed to request adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("Failed to create device")
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
    let device = make_device();
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
        .with_views(&[
            input.handle(ResourceAccess::Read).expect("input handle"),
            output.handle(ResourceAccess::Write).expect("output handle"),
        ])
        .dispatch(1, 1, 1);

    let grant = worker.grant_read(&output).expect("grant_read");
    const FRAMES: u32 = 3;
    let mut upload = Scheme::new(&ctx);
    for submission in 1..=FRAMES {
        // Separate upload submission per frame via a persistent upload scheme.
        upload::upload_parcel(&mut upload, &input, bytemuck::cast_slice(&[submission; 8])).expect("upload_parcel");

        let frame = worker.submit().expect("submit worker");
        for v in read_grant_u32(&grant, &frame, 8) {
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

/// Copy-only scheme: pre-initialized input, no upload — retention hit on submission 1.
#[test]
fn clean_scheme_resubmits_without_rerecord() {
    let device = make_device();
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
        .with_views(&[
            input.handle(ResourceAccess::Read).expect("input handle"),
            output.handle(ResourceAccess::Write).expect("output handle"),
        ])
        .dispatch(1, 1, 1);

    scheme.submit().expect("submit 0");
    let frame = scheme.submit().expect("submit 1");
    frame.wait(&ctx).expect("wait");
    assert!(output.is_settled(&ctx), "completed work must leave parcel settled");

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
    let device = make_device();
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
        .with_views(&[selector.handle(ResourceAccess::ReadWrite).expect("selector handle")])
        .dispatch(1, 1, 1);

    let grant = scheme.grant_read(&selector).expect("grant_read");
    const N: u64 = 5;
    let mut last_frame = None;
    for _ in 0..N {
        last_frame = Some(scheme.submit().expect("submit"));
    }
    let frame = last_frame.expect("submit");
    let count = read_grant_u32(&grant, &frame, 1)[0];
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
    let device = make_device();
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
        .with_views(&[
            in_a.handle(ResourceAccess::Read).expect("in_a handle"),
            out_a.handle(ResourceAccess::Write).expect("out_a handle"),
        ])
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
        .with_views(&[
            in_b.handle(ResourceAccess::Read).expect("in_b handle"),
            out_b.handle(ResourceAccess::Write).expect("out_b handle"),
        ])
        .dispatch(1, 1, 1);

    let grant_a = scheme_a.grant_read(&out_a).expect("grant_read");
    let grant_b = scheme_b.grant_read(&out_b).expect("grant_read");

    // Interleave submissions: A, B, A, B
    let _ = scheme_a.submit().expect("a1");
    let _ = scheme_b.submit().expect("b1");
    let frame_a = scheme_a.submit().expect("a2");
    let frame_b = scheme_b.submit().expect("b2");

    for v in read_grant_u32(&grant_a, &frame_a, 8) {
        assert_eq!(v, 1u32, "scheme_a must produce 1s");
    }
    for v in read_grant_u32(&grant_b, &frame_b, 8) {
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
    let device = make_device();
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, LEASE_TEXTURE_SHADER).expect("compile texture shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let mut scheme = Scheme::new(&ctx);
    let lease = scheme
        .lease_texture(
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureKind::DirectInterpolated,
            TextureFlags::empty(),
        )
        .expect("lease texture");
    let handle = scheme
        .lease_handle(&lease, ResourceAccess::Write)
        .expect("lease handle");
    scheme
        .node("write_tex", &pipeline)
        .with_parcel(&lease, NodeAccess::Write)
        .with_views(&[handle])
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
    let device = make_device();
    let ctx = submission_context(&device);

    let outstanding_before = ctx.transient_outstanding_bytes().texture;
    let create_count_before = ctx.transient_texture_create_count();

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
        ctx.transient_texture_create_count(),
        create_count_before,
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
fn grant_read_concurrent_frames_distinct_backings() {
    let device = make_device();
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
        .with_views(&[buf.handle(ResourceAccess::Write).expect("handle")])
        .dispatch(1, 1, 1);
    let grant = scheme.grant_read(&buf).expect("grant_read");

    let frame1 = scheme.submit().expect("submit K");
    let frame2 = scheme.submit().expect("submit K+1 without waiting on K");

    for frame in [&frame1, &frame2] {
        let loan = grant.consume(frame).expect("grant consume");
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
fn grant_read_texture_concurrent_frames_distinct_backings() {
    let device = make_device();
    let ctx = submission_context(&device);

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
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
            None,
        )
        .expect("texture parcel");
    let tex_w = texture.handle(ResourceAccess::Write).expect("tex write");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("write_tex", &pipeline)
        .with_parcel(&texture, NodeAccess::Write)
        .with_views(&[tex_w])
        .dispatch(wg_x, wg_y, 1);
    let grant = scheme.grant_read_texture(&texture).expect("grant_read_texture");

    let frame1 = scheme.submit().expect("submit K");
    let frame2 = scheme.submit().expect("submit K+1 without waiting on K");

    for frame in [&frame1, &frame2] {
        let loan = grant.consume(frame).expect("grant consume");
        assert!(loan.len() > 0, "texture readback empty");
        assert_eq!(loan[0], 255, "R channel");
        assert_eq!(loan[1], 0, "G channel");
        assert_eq!(loan[2], 0, "B channel");
        assert_eq!(loan[3], 255, "A channel");
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
        .with_views(&[buf.handle(ResourceAccess::Write).expect("handle")])
        .dispatch(1, 1, 1);
    scheme
}

/// Second `grant.consume` on the same submission must fail (staging cell is single-consume).
#[test]
fn grant_read_double_read_same_frame_errors() {
    let device = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = fill_42_scheme(&ctx, &pipe, &buf);
    let grant = scheme.grant_read(&buf).expect("grant_read");
    let frame = scheme.submit().expect("submit");

    let _loan = grant.consume(&frame).expect("first read");
    let err = grant.consume(&frame).expect_err("second read must fail");
    assert!(err.to_string().contains("already consumed"), "unexpected error: {err}");
}

/// Cloned frames share one staging cell; only one read succeeds.
#[test]
fn grant_read_cloned_frame_double_read_errors() {
    let device = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = fill_42_scheme(&ctx, &pipe, &buf);
    let grant = scheme.grant_read(&buf).expect("grant_read");
    let frame = scheme.submit().expect("submit");
    let frame_clone = frame.clone();

    let _loan = grant.consume(&frame).expect("first read");
    let err = grant
        .consume(&frame_clone)
        .expect_err("cloned frame second consume must fail");
    assert!(err.to_string().contains("already consumed"), "unexpected error: {err}");
}

/// Grant with no producing dispatch copies uninitialized parcel bytes (zeros on fresh acquire).
#[test]
fn grant_read_without_producing_dispatch_reads_zeros() {
    let device = make_device();
    let ctx = submission_context(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = Scheme::new(&ctx);
    let grant = scheme.grant_read(&buf).expect("grant_read");
    let frame = scheme.submit().expect("submit");
    let values = read_grant_u32(&grant, &frame, 64);
    assert!(
        values.iter().all(|&v| v == 0),
        "expected zeros without a producer dispatch"
    );
}

/// Grant node before dispatch in IR still reads post-dispatch bytes — copy runs after all dispatches.
#[test]
fn grant_read_before_dispatch_node_still_reads_producer_output() {
    let device = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = Scheme::new(&ctx);
    let grant = scheme.grant_read(&buf).expect("grant_read");
    scheme
        .node("fill", &pipe)
        .with_parcel(&buf, NodeAccess::Write)
        .with_views(&[buf.handle(ResourceAccess::Write).expect("handle")])
        .dispatch(1, 1, 1);
    let frame = scheme.submit().expect("submit");
    let values = read_grant_u32(&grant, &frame, 64);
    assert!(
        values.iter().all(|&v| v == 42),
        "grant before dispatch in IR still sees fill output"
    );
}

/// Dropping a frame without reading returns staging; a later submission can still be read.
#[test]
fn grant_read_drop_frame_without_read_then_submit_and_read() {
    let device = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = fill_42_scheme(&ctx, &pipe, &buf);
    let grant = scheme.grant_read(&buf).expect("grant_read");

    let frame1 = scheme.submit().expect("submit 1");
    drop(frame1);

    let frame2 = scheme.submit().expect("submit 2 after frame1 drop");
    let values = read_grant_u32(&grant, &frame2, 64);
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
fn grant_read_texture_sequential_resubmit_correct_data() {
    let device = make_device();
    let ctx = submission_context(&device);

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
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
            None,
        )
        .expect("texture parcel");
    let tex_w = texture.handle(ResourceAccess::Write).expect("tex write");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("write_tex", &pipeline)
        .with_parcel(&texture, NodeAccess::Write)
        .with_views(&[tex_w])
        .dispatch(wg_x, wg_y, 1);
    let grant = scheme.grant_read_texture(&texture).expect("grant_read_texture");

    // Three sequential submits: wait and read each one before the next.
    // The second and third submits are retained resubmits. Each triggers
    // a new CopyTextureToReadback in finish_submit_frame. The DX12 backend
    // must have the correct last_layout going into the copy barrier each time.
    for round in 0..3u32 {
        let frame = scheme.submit().expect("submit");
        let loan = grant.consume(&frame).expect("grant read");
        assert!(loan.len() > 0, "texture readback empty on round {round}");
        assert_eq!(loan[0], 255, "R channel, round {round}");
        assert_eq!(loan[1], 0, "G channel, round {round}");
        assert_eq!(loan[2], 0, "B channel, round {round}");
        assert_eq!(loan[3], 255, "A channel, round {round}");
    }

    let stats = scheme.replay_stats();
    assert_eq!(stats.records, 1, "dispatch records exactly once");
    #[cfg(not(feature = "metal"))]
    assert!(stats.resubmit_hits >= 2, "rounds 2 and 3 must be retention hits");
}

/// Many consecutive submits with dropped unread frames must not exhaust staging (pool recycles on frame drop).
#[test]
fn grant_read_many_dropped_frames_without_read_then_read_succeeds() {
    let device = make_device();
    let ctx = submission_context(&device);
    let pipe = fill_42_pipeline(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("output parcel");

    let mut scheme = fill_42_scheme(&ctx, &pipe, &buf);
    let grant = scheme.grant_read(&buf).expect("grant_read");

    for _ in 0..8 {
        drop(scheme.submit().expect("submit with dropped frame"));
    }
    let frame = scheme.submit().expect("final submit");
    let values = read_grant_u32(&grant, &frame, 64);
    assert!(
        values.iter().all(|&v| v == 42),
        "read succeeds after many dropped unread frames (staging pool must recycle)"
    );
}
