//! GPU integration tests for epoch-driven cross-scheme synchronization.
//!
//! Exact integer readback assertions (no FLIP). Gated on real backends.
//!
//! Uses a process-wide [`Device`] so parallel tests do not each create and destroy
//! a Vulkan instance (which races with validation-layer global state at teardown).
//! Each test takes [`test_lock`] so cross-submit ledger state stays isolated.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

#[path = "common/shared_device.rs"]
mod shared_device;
#[path = "common/submission.rs"]
mod submission;
#[path = "common/upload.rs"]
mod upload;

use goldy::{
    BackendType, BufferKind, ComputePipeline, Context, Device, Grant, NodeAccess, Parcel, ReadGrant, RetainedPool,
    Scheme, ShaderModule, Submission,
};
use shared_device::{shared_device, test_lock};
use submission::submission_context;

const INC_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(Scattered<uint> buf, ThreadId id) {
    buf[id.x] = buf[id.x] + 1u;
}
"#;

const READ_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(BufRO<uint> buf, ThreadId id) {
    let _ = buf[id.x];
}
"#;

const OVERWRITE_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(Scattered<uint> buf, ThreadId id) {
    buf[id.x] = 42u;
}
"#;

const COPY_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(BufRO<uint> src, Scattered<uint> dst, ThreadId id) {
    dst[id.x] = src[id.x];
}
"#;

fn read_u32(grant: &ReadGrant<goldy::GrantBuffer>, submission: &Submission) -> u32 {
    let loan = grant.consume(submission).expect("grant");
    bytemuck::cast_slice::<u8, u32>(&loan)[0]
}

#[test]
fn saxpy_style_chain_closed_form() {
    let _guard = test_lock();
    let device = shared_device();
    let ctx = submission_context(&device);
    let shader = ShaderModule::from_slang(&device, INC_SHADER).expect("shader");
    let pipe = ComputePipeline::new(&device, &shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let buf = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("buf");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("inc", &pipe)
        .with_parcel(&buf, NodeAccess::ReadWrite)
        .dispatch(1, 1, 1);
    let grant = scheme.grant_read(&buf).expect("grant");

    const STEPS: u32 = 50;
    for _ in 0..STEPS {
        scheme.submit().expect("submit");
    }
    let submission = scheme.submit().expect("final");
    assert_eq!(read_u32(&grant, &submission), STEPS + 1);
}

#[test]
fn war_write_after_read_pipelined_overwrite() {
    let _guard = test_lock();
    let device = shared_device();
    let ctx = submission_context(&device);
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_shader = ShaderModule::from_slang(&device, OVERWRITE_SHADER).expect("shader");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let buf = pool
        .acquire_buffer_with_data(&[7u32; 1], BufferKind::Scattered)
        .expect("buf");

    // Independent schemes, no CPU wait — writer must synchronize against the reader's epoch (WAR).
    let mut reader = Scheme::new(&ctx);
    reader
        .node("read", &read_pipe)
        .with_parcel(&buf, NodeAccess::Read)
        .dispatch(1, 1, 1);
    reader.submit().expect("read");

    let mut writer = Scheme::new(&ctx);
    writer
        .node("write", &write_pipe)
        .with_parcel(&buf, NodeAccess::Write)
        .dispatch(1, 1, 1);
    let grant = writer.grant_read(&buf).expect("grant");
    let submission = writer.submit().expect("write");

    assert_eq!(read_u32(&grant, &submission), 42);
}

fn retained_copy_reader(
    ctx: &Context,
    pipe: &ComputePipeline,
    src: &Parcel,
    dst: &Parcel,
) -> (Scheme, ReadGrant<goldy::GrantBuffer>) {
    let mut reader = Scheme::new(ctx);
    reader
        .node("copy", pipe)
        .with_parcel(src, NodeAccess::Read)
        .with_parcel(dst, NodeAccess::Write)
        .dispatch(1, 1, 1);
    let grant = reader.grant_read(dst).expect("grant");
    (reader, grant)
}

fn assert_retained_resubmit_stats(device: &Device, reader: &Scheme, expected_resubmit_hits: u64) {
    // Metal re-records each submit; retention counters are Vulkan/DX12 only.
    if device.backend_type() == BackendType::Metal {
        return;
    }
    assert_eq!(reader.replay_stats().records, 1, "scheme must record exactly once");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        reader.replay_stats().resubmit_hits,
        expected_resubmit_hits,
        "remaining submits must be retention hits"
    );
    #[cfg(feature = "metal")]
    let _ = expected_resubmit_hits;
}

#[test]
fn retained_reader_observes_independent_writer_across_resubmits() {
    let _guard = test_lock();
    let device = shared_device();
    let ctx = submission_context(&device);
    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("shader");
    let pipe = ComputePipeline::new(&device, &shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let src = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("src");
    let dst = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("dst");

    let (mut reader, grant) = retained_copy_reader(&ctx, &pipe, &src, &dst);

    let mut upload = Scheme::new(&ctx);
    for value in [11u32, 22, 33] {
        upload::upload_parcel(&mut upload, &src, bytemuck::bytes_of(&value)).expect("upload src");
        let submission = reader.submit().expect("retained resubmit");
        assert_eq!(
            read_u32(&grant, &submission),
            value,
            "retained reader must observe independent upload (value={value})"
        );
    }

    assert_retained_resubmit_stats(&device, &reader, 2);
}

#[test]
fn retained_waw_overwrites_independent_upload() {
    let _guard = test_lock();
    let device = shared_device();
    let ctx = submission_context(&device);
    let write_shader = ShaderModule::from_slang(&device, OVERWRITE_SHADER).expect("shader");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let src = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("src");

    let mut worker = Scheme::new(&ctx);
    worker
        .node("overwrite", &write_pipe)
        .with_parcel(&src, NodeAccess::Write)
        .dispatch(1, 1, 1);
    let grant = worker.grant_read(&src).expect("grant");

    let mut upload_scheme = Scheme::new(&ctx);
    for upload_value in [99u32, 88, 77] {
        upload::upload_parcel(&mut upload_scheme, &src, bytemuck::bytes_of(&upload_value)).expect("upload src");
        let submission = worker.submit().expect("retained resubmit");
        assert_eq!(
            read_u32(&grant, &submission),
            42,
            "retained overwrite must win over upload (upload={upload_value})"
        );
    }

    assert_retained_resubmit_stats(&device, &worker, 2);
}

#[test]
fn retained_reader_cross_context_observes_independent_writer() {
    let _guard = test_lock();
    let device = shared_device();
    let ctx_producer = device.create_context().expect("producer ctx");
    let ctx_consumer = device.create_context().expect("consumer ctx");
    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("shader");
    let pipe = ComputePipeline::new(&device, &shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let src = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("src");
    let dst = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("dst");

    let (mut reader, grant) = retained_copy_reader(&ctx_consumer, &pipe, &src, &dst);

    let mut upload = Scheme::new(&ctx_producer);
    for value in [5u32, 15, 25] {
        upload::upload_parcel(&mut upload, &src, bytemuck::bytes_of(&value)).expect("upload src");
        let submission = reader.submit().expect("cross-context resubmit");
        assert_eq!(
            read_u32(&grant, &submission),
            value,
            "cross-context retained reader must observe producer upload (value={value})"
        );
    }

    assert_retained_resubmit_stats(&device, &reader, 2);
}

#[test]
fn retained_resubmit_not_dirtied_by_unrelated_scheme() {
    let _guard = test_lock();
    let device = shared_device();
    let ctx = submission_context(&device);
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_shader = ShaderModule::from_slang(&device, OVERWRITE_SHADER).expect("shader");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("read pipe");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("write pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel_p = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("parcel_p");
    let parcel_q = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("parcel_q");

    let mut reader = Scheme::new(&ctx);
    reader
        .node("read_p", &read_pipe)
        .with_parcel(&parcel_p, NodeAccess::Read)
        .dispatch(1, 1, 1);
    reader.submit().expect("reader record");

    let mut writer = Scheme::new(&ctx);
    writer
        .node("write_q", &write_pipe)
        .with_parcel(&parcel_q, NodeAccess::Write)
        .dispatch(1, 1, 1);
    for _ in 0..3 {
        writer.submit().expect("writer submit");
    }

    for _ in 0..3 {
        reader.submit().expect("reader resubmit");
    }

    if device.backend_type() == BackendType::Metal {
        return;
    }
    assert!(
        !reader.is_topology_dirty(),
        "unrelated writer activity must not dirty the reader"
    );
    assert_eq!(reader.replay_stats().records, 1);
    assert_eq!(reader.replay_stats().topology_records, 0);
    #[cfg(not(feature = "metal"))]
    assert_eq!(reader.replay_stats().resubmit_hits, 3);
}

#[test]
fn retained_reader_dirtied_once_by_new_writer_then_stable() {
    let _guard = test_lock();
    let device = shared_device();
    let ctx = submission_context(&device);
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_shader = ShaderModule::from_slang(&device, OVERWRITE_SHADER).expect("shader");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("read pipe");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("write pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("parcel");

    let mut reader = Scheme::new(&ctx);
    reader
        .node("read", &read_pipe)
        .with_parcel(&parcel, NodeAccess::Read)
        .dispatch(1, 1, 1);
    reader.submit().expect("reader record");

    let mut writer = Scheme::new(&ctx);
    writer
        .node("write", &write_pipe)
        .with_parcel(&parcel, NodeAccess::Write)
        .dispatch(1, 1, 1);
    writer.submit().expect("writer record");

    // Topology dirty flag is meaningful on all backends: the writer joining
    // the shared parcel must mark the reader dirty exactly once.
    assert!(reader.is_topology_dirty());
    reader.submit().expect("reader topology re-record");
    assert!(!reader.is_topology_dirty());

    for _ in 0..2 {
        reader.submit().expect("reader stable resubmit");
    }

    // topology_records is meaningful on all backends: exactly one re-record
    // was triggered by a foreign-scheme topology change (not a structural
    // mutation). Metal re-records on every submit but still tracks why.
    assert_eq!(reader.replay_stats().topology_records, 1);

    // records and resubmit_hits diverge by backend: on Metal every submit
    // is a re-record (no CB reuse), so records == total submits (4) and
    // resubmit_hits is not tracked. On Vulkan/DX12 the two stable submits
    // are served from the retained command list.
    if device.backend_type() != BackendType::Metal {
        assert_eq!(reader.replay_stats().records, 2);
        #[cfg(not(feature = "metal"))]
        assert_eq!(reader.replay_stats().resubmit_hits, 2);
    }
}

#[test]
fn topology_re_record_produces_correct_barriers_and_data() {
    let _guard = test_lock();
    let device = shared_device();
    let ctx = submission_context(&device);
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_shader = ShaderModule::from_slang(&device, OVERWRITE_SHADER).expect("shader");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("read pipe");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("write pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("parcel");

    let mut reader = Scheme::new(&ctx);
    reader
        .node("read", &read_pipe)
        .with_parcel(&parcel, NodeAccess::Read)
        .dispatch(1, 1, 1);
    let grant = reader.grant_read(&parcel).expect("grant");
    reader.submit().expect("reader record");

    let mut writer = Scheme::new(&ctx);
    writer
        .node("write", &write_pipe)
        .with_parcel(&parcel, NodeAccess::Write)
        .dispatch(1, 1, 1);
    writer.submit().expect("writer record");

    let submission = reader.submit().expect("reader topology re-record");
    submission.wait(&ctx).expect("wait");
    assert_eq!(read_u32(&grant, &submission), 42);
}

#[test]
fn repeated_resubmit_of_b_never_dirties_a() {
    let _guard = test_lock();
    let device = shared_device();
    let ctx = submission_context(&device);
    let inc_shader = ShaderModule::from_slang(&device, INC_SHADER).expect("shader");
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let inc_pipe = ComputePipeline::new(&device, &inc_shader).expect("inc pipe");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("read pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("parcel");

    let mut worker = Scheme::new(&ctx);
    worker
        .node("inc", &inc_pipe)
        .with_parcel(&parcel, NodeAccess::ReadWrite)
        .dispatch(1, 1, 1);
    worker.submit().expect("worker record");

    let mut observer = Scheme::new(&ctx);
    observer
        .node("observe", &read_pipe)
        .with_parcel(&parcel, NodeAccess::Read)
        .dispatch(1, 1, 1);
    observer.submit().expect("observer settle");
    assert!(!observer.is_topology_dirty());

    const RESUBMITS: u32 = 50;
    for _ in 0..RESUBMITS {
        worker.submit().expect("worker resubmit");
    }
    assert!(
        !observer.is_topology_dirty(),
        "repeated worker resubmits must not dirty the observer"
    );

    for _ in 0..RESUBMITS {
        observer.submit().expect("observer resubmit");
    }

    if device.backend_type() == BackendType::Metal {
        return;
    }
    assert_eq!(observer.replay_stats().records, 1);
    assert_eq!(observer.replay_stats().topology_records, 0);
    #[cfg(not(feature = "metal"))]
    assert_eq!(observer.replay_stats().resubmit_hits, RESUBMITS as u64);
}

/// Per-parcel cross-submit tracking: a scheme that reads field B must not be
/// topology-dirtied when another scheme repeatedly writes disjoint field A.
#[test]
fn partitioned_buffer_disjoint_ranges_no_cross_submit_hazard() {
    use goldy::{field, Init};

    let _guard = test_lock();
    let device = shared_device();
    let ctx = submission_context(&device);
    let inc_shader = ShaderModule::from_slang(&device, INC_SHADER).expect("shader");
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let inc_pipe = ComputePipeline::new(&device, &inc_shader).expect("pipe");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let record = pool
        .acquire_record([field("a", Init::data(&[0u32; 1])), field("b", Init::data(&[0u32; 1]))])
        .expect("acquire_record");

    let mut worker = Scheme::new(&ctx);
    worker
        .node("inc_a", &inc_pipe)
        .with_parcel(&record["a"], NodeAccess::ReadWrite)
        .dispatch(1, 1, 1);
    worker.submit().expect("worker record");

    let mut observer = Scheme::new(&ctx);
    observer
        .node("read_b", &read_pipe)
        .with_parcel(&record["b"], NodeAccess::Read)
        .dispatch(1, 1, 1);
    observer.submit().expect("observer record");
    assert!(!observer.is_topology_dirty());

    const RESUBMITS: u32 = 20;
    for _ in 0..RESUBMITS {
        worker.submit().expect("worker resubmit on field a");
    }
    assert!(
        !observer.is_topology_dirty(),
        "writes to disjoint field a must not dirty observer bound to field b"
    );
}
