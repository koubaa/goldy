//! Deterministic mock-backend integration tests for epoch-driven cross-scheme sync.

use goldy::backend::GpuCommand;
use goldy::task_graph::BarrierUsage;
use goldy::test_support::{mock_device, with_mock};
use goldy::types::ResourceAccess;
use goldy::{BufferKind, ComputePipeline, Context, Device, NodeAccess, Parcel, RetainedPool, Scheme, ShaderModule};

fn mock_ctx(device: &Device) -> Context {
    device.create_context().expect("context")
}

fn second_ctx(device: &Device) -> Context {
    device.create_context().expect("second context")
}

fn clear_mock(device: &Device) {
    with_mock(device, |m| m.reset_tracking());
}

fn recorded_waits(device: &Device) -> Vec<Vec<goldy::timeline::Epoch>> {
    with_mock(device, |m| m.recorded_waits.clone())
}

fn barrier_buffers(device: &Device) -> Vec<(u64, BarrierUsage)> {
    with_mock(device, |m| {
        m.recorded_compute_commands
            .iter()
            .flat_map(|batch| batch.iter())
            .find_map(|cmd| match cmd {
                GpuCommand::ResourceBarrier { buffers, .. } => Some(buffers.clone()),
                _ => None,
            })
            .unwrap_or_default()
    })
}

fn retained_resubmits(device: &Device) -> usize {
    with_mock(device, |m| m.retained_resubmit_count)
}

const WRITE_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(Scattered<uint> buf, ThreadId id) {
    buf[id.x] = 42u;
}
"#;

const READ_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(BufRO<uint> buf, ThreadId id) {
    let _ = buf[id.x];
}
"#;

fn write_scheme(ctx: &Context, parcel: &Parcel, pipeline: &ComputePipeline) -> Scheme {
    let mut s = Scheme::new(ctx);
    s.node("write", pipeline)
        .bind_parcel(parcel, NodeAccess::Write)
        .bind_views(&[parcel.handle(ResourceAccess::Write).expect("uav")])
        .dispatch(1, 1, 1);
    s
}

fn read_scheme(ctx: &Context, parcel: &Parcel, pipeline: &ComputePipeline) -> Scheme {
    let mut s = Scheme::new(ctx);
    s.node("read", pipeline)
        .bind_parcel(parcel, NodeAccess::Read)
        .bind_views(&[parcel.handle(ResourceAccess::Read).expect("srv")])
        .dispatch(1, 1, 1);
    s
}

#[test]
fn ping_pong_buffers_emit_alternating_prologue() {
    let device = mock_device();
    let ctx = mock_ctx(&device);
    let write_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("shader");
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let buf_a = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("buf_a");
    let buf_b = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("buf_b");

    let mut write_a = write_scheme(&ctx, &buf_a, &write_pipe);
    let mut read_a = read_scheme(&ctx, &buf_a, &read_pipe);
    let mut write_b = write_scheme(&ctx, &buf_b, &write_pipe);
    let mut read_b = read_scheme(&ctx, &buf_b, &read_pipe);

    write_a.submit().expect("write_a");

    clear_mock(&device);
    read_a.submit().expect("read_a after write_a");
    assert_eq!(barrier_buffers(&device).len(), 1, "RAW on buf_a");

    clear_mock(&device);
    write_b.submit().expect("write_b");
    assert!(barrier_buffers(&device).is_empty(), "fresh write on buf_b");

    clear_mock(&device);
    read_b.submit().expect("read_b after write_b");
    assert_eq!(barrier_buffers(&device).len(), 1, "RAW on buf_b");
}

#[test]
fn upload_then_consumer_emits_raw_barrier() {
    let device = mock_device();
    let ctx = mock_ctx(&device);
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("parcel");

    let mut upload = Scheme::new(&ctx);
    upload
        .commit_write_parcel(&parcel, 0, vec![1, 0, 0, 0])
        .expect("upload");
    upload.submit().expect("upload submit");

    clear_mock(&device);
    let mut consumer = read_scheme(&ctx, &parcel, &read_pipe);
    consumer.submit().expect("consumer");
    assert_eq!(barrier_buffers(&device).len(), 1);
    assert!(recorded_waits(&device).last().is_some_and(|w| w.is_empty()));
}

#[test]
fn retention_resubmit_preserves_hits_and_dynamic_prologue() {
    let device = mock_device();
    let ctx = mock_ctx(&device);
    let write_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("shader");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("parcel");

    let mut worker = write_scheme(&ctx, &parcel, &write_pipe);
    worker.submit().expect("record");
    assert_eq!(worker.replay_stats().records, 1);

    clear_mock(&device);
    let mut touch = write_scheme(&ctx, &parcel, &write_pipe);
    touch.submit().expect("touch ledger");
    assert_eq!(barrier_buffers(&device).len(), 1, "dynamic prologue on hazard");

    clear_mock(&device);
    for _ in 0..3 {
        worker.submit().expect("resubmit");
    }
    assert_eq!(retained_resubmits(&device), 3);
}

#[test]
fn rar_and_no_alias_emit_zero_sync() {
    let device = mock_device();
    let ctx = mock_ctx(&device);
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let p = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("p");

    let mut read_p = read_scheme(&ctx, &p, &read_pipe);
    read_p.submit().expect("first read");

    clear_mock(&device);
    read_p.submit().expect("rar resubmit");
    // Resubmit may re-execute retained body; cross-submit sync must not add extra waits.
    assert!(recorded_waits(&device).iter().all(|w| w.is_empty()));
}

#[test]
fn cross_context_raw_emits_wait() {
    let device = mock_device();
    let ctx1 = mock_ctx(&device);
    let ctx2 = second_ctx(&device);
    let write_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("shader");
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("parcel");

    let mut producer = write_scheme(&ctx1, &parcel, &write_pipe);
    producer.submit().expect("producer");

    clear_mock(&device);
    let mut consumer = read_scheme(&ctx2, &parcel, &read_pipe);
    consumer.submit().expect("cross ctx");
    let waits = recorded_waits(&device).last().cloned().unwrap_or_default();
    assert_eq!(waits.len(), 1);
    assert_eq!(waits[0].context, ctx1.test_backend_handle());
}

#[test]
fn same_context_raw_emits_barrier_not_wait() {
    let device = mock_device();
    let ctx = mock_ctx(&device);
    let write_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("shader");
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("parcel");

    let mut producer = write_scheme(&ctx, &parcel, &write_pipe);
    producer.submit().expect("producer");

    clear_mock(&device);
    let mut consumer = read_scheme(&ctx, &parcel, &read_pipe);
    consumer.submit().expect("same ctx");
    assert_eq!(barrier_buffers(&device).len(), 1);
    assert!(recorded_waits(&device).iter().all(|w| w.is_empty()));
}

#[test]
fn war_same_context_emits_barrier_on_write_after_read() {
    let device = mock_device();
    let ctx = mock_ctx(&device);
    let write_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("shader");
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("parcel");

    let mut reader = read_scheme(&ctx, &parcel, &read_pipe);
    reader.submit().expect("read");

    clear_mock(&device);
    let mut writer = write_scheme(&ctx, &parcel, &write_pipe);
    writer.submit().expect("write after read");
    assert_eq!(
        barrier_buffers(&device).len(),
        1,
        "WAR: writer after reader must emit a same-context prologue barrier"
    );
    assert!(recorded_waits(&device).iter().all(|w| w.is_empty()));
}

#[test]
fn stamp_monotonicity_never_regresses() {
    let device = mock_device();
    let ctx = mock_ctx(&device);
    let write_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("shader");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("parcel");

    let mut scheme = write_scheme(&ctx, &parcel, &write_pipe);
    for _ in 0..5 {
        scheme.submit().expect("submit");
    }
    let ctx_handle = ctx.test_backend_handle();
    let epoch = parcel.last_referenced().get(&ctx_handle).copied().expect("stamped");
    scheme.submit().expect("again");
    let later = parcel.last_referenced().get(&ctx_handle).copied().expect("stamped");
    assert!(later >= epoch);
}
