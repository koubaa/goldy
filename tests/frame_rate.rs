//! Host-side cost of a small frame. The four cases are the ones a 3 kHz frame
//! budget is sensitive to: a retained worker that waits, two frames in flight,
//! a new scheme every frame, and a dispatch size that changes every frame.
//!
//! Absolute times are backend host times. The assertions reject the regression
//! where building or copying a frame-table row costs tens of milliseconds.

#![cfg(any(feature = "vulkan", feature = "metal"))]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    BackendType, Buffer, BufferKind, ComputePipeline, Context, Instance, NodeAccess, NodeId, RequestAdapterOptions,
    Runtime, RuntimeDescriptor, Scheme, ShaderModule,
};
use std::time::Instant;

const COPY_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(8, 1, 1)]
void cs_main(BufRO<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x];
}
"#;

fn device() -> Runtime {
    Instance::new()
        .expect("instance")
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_runtime(&RuntimeDescriptor::default())
        .expect("runtime")
}

fn pipeline(device: &Runtime) -> ComputePipeline {
    let shader = ShaderModule::from_slang(device, COPY_SHADER).expect("shader");
    ComputePipeline::new(device, &shader).expect("pipeline")
}

fn buffers(device: &Runtime) -> (Buffer, Buffer) {
    let input = device
        .acquire_buffer_with_data(&[1u32; 8], BufferKind::Scattered)
        .expect("input");
    let output = device
        .acquire_buffer_with_data(&[0u32; 8], BufferKind::Scattered)
        .expect("output");
    (input, output)
}

fn supports_frame_rate_test(device: &Runtime) -> bool {
    matches!(device.backend_type(), BackendType::Vulkan | BackendType::Metal)
}

fn record_worker(
    ctx: &Context,
    pipeline: &ComputePipeline,
    input: &Buffer,
    output: &Buffer,
    copies: usize,
) -> (Scheme, NodeId) {
    let mut worker = Scheme::new(ctx);
    let mut node = None;
    for i in 0..copies {
        node = Some(
            worker
                .node(format!("copy-{i}"), pipeline)
                .with_parcel(input, NodeAccess::Read)
                .with_parcel(output, NodeAccess::Write)
                .dispatch(1, 1, 1),
        );
    }
    (worker, node.expect("at least one copy"))
}

fn percentile(mut samples: Vec<f64>, p: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() - 1) as f64 * p).round() as usize;
    samples[idx]
}

/// Retained worker, host read every frame. Eight copies.
#[test]
fn retained_worker_wait_every_frame_stays_under_a_millisecond() {
    let _spec = goldy::test_support::SpecializationOverride::force_disabled();
    let _cb = goldy::test_support::CbReuseOverride::force_enabled();
    let device = device();
    if !supports_frame_rate_test(&device) {
        return;
    }
    let ctx = submission::submission_context(&device);
    let pipeline = pipeline(&device);
    let (input, output) = buffers(&device);
    let (mut worker, _) = record_worker(&ctx, &pipeline, &input, &output, 8);
    let mut samples = Vec::new();
    for _ in 0..40 {
        let start = Instant::now();
        let mut frame = worker.submit().expect("submit");
        let got = (&mut frame >> &output).take::<u32>().expect("read").to_vec();
        drop(frame);
        samples.push(start.elapsed().as_secs_f64() * 1e6);
        assert_eq!(got, vec![1; 8]);
    }
    let p50 = percentile(samples.clone(), 0.50);
    let p99 = percentile(samples, 0.99);
    eprintln!(
        "case A retained+wait p50 {p50:.1} us p99 {p99:.1} us stats {:?}",
        worker.replay_stats()
    );
    assert!(p50 < 2_000.0, "case A p50 {p50:.1} us");
    assert_eq!(worker.replay_stats().retained_evictions, 0);
}

/// Retained worker, two submits outstanding, then a wait.
#[test]
fn retained_worker_two_in_flight_keeps_its_rate() {
    let _spec = goldy::test_support::SpecializationOverride::force_disabled();
    let _cb = goldy::test_support::CbReuseOverride::force_enabled();
    let device = device();
    if !supports_frame_rate_test(&device) {
        return;
    }
    let ctx = submission::submission_context(&device);
    let pipeline = pipeline(&device);
    let (input, output) = buffers(&device);
    let (mut worker, _) = record_worker(&ctx, &pipeline, &input, &output, 8);
    let start = Instant::now();
    const FRAMES: u32 = 80;
    for frame in 0..FRAMES {
        drop(worker.submit().expect("submit"));
        if frame % 2 == 1 {
            ctx.wait_until_idle().expect("idle");
        }
    }
    let secs = start.elapsed().as_secs_f64();
    let hz = f64::from(FRAMES) / secs;
    eprintln!("case C two-in-flight {hz:.0} Hz stats {:?}", worker.replay_stats());
    assert!(hz > 1_000.0, "case C {hz:.0} Hz");
    assert_eq!(worker.replay_stats().retained_evictions, 0);
}

/// A new scheme every frame pays record plus emission, not a full-table allocation.
#[test]
fn new_scheme_every_frame_stays_under_five_milliseconds() {
    let _spec = goldy::test_support::SpecializationOverride::force_disabled();
    let _cb = goldy::test_support::CbReuseOverride::force_enabled();
    let device = device();
    if !supports_frame_rate_test(&device) {
        return;
    }
    let ctx = submission::submission_context(&device);
    let pipeline = pipeline(&device);
    let (input, output) = buffers(&device);
    let mut samples = Vec::new();
    for _ in 0..20 {
        let start = Instant::now();
        let (mut worker, _) = record_worker(&ctx, &pipeline, &input, &output, 8);
        drop(worker.submit().expect("submit"));
        samples.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let p50 = percentile(samples, 0.50);
    eprintln!("case B new scheme p50 {p50:.1} us");
    assert!(p50 < 5_000.0, "case B p50 {p50:.1} us");
}

/// Changing the dispatch size every frame re-records. It must not fill the
/// persistent-row pool and then stall on one victim row.
#[test]
fn changing_dispatch_size_does_not_stall_on_row_eviction() {
    let _spec = goldy::test_support::SpecializationOverride::force_disabled();
    let _cb = goldy::test_support::CbReuseOverride::force_enabled();
    let device = device();
    if !supports_frame_rate_test(&device) {
        return;
    }
    let ctx = submission::submission_context(&device);
    let pipeline = pipeline(&device);
    let (input, output) = buffers(&device);
    let (mut worker, node) = record_worker(&ctx, &pipeline, &input, &output, 1);
    let mut samples = Vec::new();
    const FRAMES: u32 = 80;
    for frame in 0..FRAMES {
        let start = Instant::now();
        worker.set_node_dispatch(node, 1 + frame % 50, 1, 1).expect("dispatch");
        drop(worker.submit().expect("submit"));
        samples.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let p50 = percentile(samples, 0.50);
    let evictions = worker.replay_stats().retained_evictions;
    eprintln!("case D dispatch change p50 {p50:.1} us evictions {evictions}");
    assert!(p50 < 5_000.0, "case D p50 {p50:.1} us");
    // One replacement per frame after the first record, not a 512-row cliff.
    assert!(
        evictions <= u64::from(FRAMES),
        "evictions {evictions} climbed past one replacement per frame"
    );
}
