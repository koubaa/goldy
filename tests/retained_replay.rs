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
    types::ResourceAccess, BufferKind, ComputePipeline, Device, DeviceDescriptor, Instance, NodeAccess,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, TextureFlags, TextureFormat, TextureKind,
};
use std::sync::Arc;
use submission::submission_context;
use upload::write_to_parcel;

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
/// This is the pattern `upload::write_to_parcel` packages as a property-only dispatch.
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
        .bind_parcel(&input, NodeAccess::Read)
        .bind_parcel(&output, NodeAccess::Write)
        .bind_resources_typed(&[
            input.handle(ResourceAccess::Read).expect("input handle"),
            output.handle(ResourceAccess::Write).expect("output handle"),
        ])
        .dispatch(1, 1, 1);

    const FRAMES: u32 = 3;
    for submission in 1..=FRAMES {
        // Separate upload submission per frame via the property-only-dispatch API.
        write_to_parcel(&ctx, &input, bytemuck::cast_slice(&[submission; 8])).expect("write_to_parcel");

        let submit_frame = worker.submit().expect("submit worker");
        submit_frame.wait(&ctx).expect("wait");

        let mut raw = vec![0u8; 8 * 4];
        output.read_to_cpu(&device, &mut raw).expect("readback output");
        for v in bytemuck::cast_slice::<u8, u32>(&raw) {
            assert_eq!(
                *v, submission,
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
        .bind_parcel(&input, NodeAccess::Read)
        .bind_parcel(&output, NodeAccess::Write)
        .bind_resources_typed(&[
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
        .bind_parcel(&selector, NodeAccess::ReadWrite)
        .bind_resources_typed(&[selector.handle(ResourceAccess::ReadWrite).expect("selector handle")])
        .dispatch(1, 1, 1);

    const N: u64 = 5;
    let mut last_frame = None;
    for _ in 0..N {
        last_frame = Some(scheme.submit().expect("submit"));
    }
    last_frame.expect("submit").wait(&ctx).expect("wait");

    let mut raw = vec![0u8; 4];
    selector.read_to_cpu(&device, &mut raw).expect("readback selector");
    let count = u32::from_le_bytes(raw.try_into().unwrap());
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
        .bind_parcel(&in_a, NodeAccess::Read)
        .bind_parcel(&out_a, NodeAccess::Write)
        .bind_resources_typed(&[
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
        .bind_parcel(&in_b, NodeAccess::Read)
        .bind_parcel(&out_b, NodeAccess::Write)
        .bind_resources_typed(&[
            in_b.handle(ResourceAccess::Read).expect("in_b handle"),
            out_b.handle(ResourceAccess::Write).expect("out_b handle"),
        ])
        .dispatch(1, 1, 1);

    // Interleave submissions: A, B, A, B
    let _ = scheme_a.submit().expect("a1");
    let _ = scheme_b.submit().expect("b1");
    let _ = scheme_a.submit().expect("a2");
    let frame = scheme_b.submit().expect("b2");
    frame.wait(&ctx).expect("wait");

    let mut raw_a = vec![0u8; 8 * 4];
    out_a.read_to_cpu(&device, &mut raw_a).expect("readback a");
    for v in bytemuck::cast_slice::<u8, u32>(&raw_a) {
        assert_eq!(*v, 1u32, "scheme_a must produce 1s");
    }

    let mut raw_b = vec![0u8; 8 * 4];
    out_b.read_to_cpu(&device, &mut raw_b).expect("readback b");
    for v in bytemuck::cast_slice::<u8, u32>(&raw_b) {
        assert_eq!(*v, 2u32, "scheme_b must produce 2s");
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
        .writes_lease(&lease)
        .bind_resources_typed(&[handle])
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
