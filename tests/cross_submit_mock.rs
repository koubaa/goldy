//! Deterministic mock-backend integration tests for epoch-driven cross-scheme sync.

use goldy::backend::GpuCommand;
use goldy::task_graph::BarrierUsage;
use goldy::test_support::{mock_device, with_mock};
use goldy::{
    BufferKind, ComputePipeline, Context, Device, NodeAccess, Parcel, RenderPipeline, RenderPipelineDesc, RetainedPool,
    Scheme, ShaderModule, TextureFormat,
};

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

fn compute_submits(device: &Device) -> usize {
    with_mock(device, |m| m.compute_dispatch_count)
}

fn all_graph_syncs_some(device: &Device) -> bool {
    with_mock(device, |m| m.recorded_graph_syncs.iter().all(|&s| s))
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
        .with_parcel(parcel, NodeAccess::Write)
        .dispatch(1, 1, 1);
    s
}

fn read_scheme(ctx: &Context, parcel: &Parcel, pipeline: &ComputePipeline) -> Scheme {
    let mut s = Scheme::new(ctx);
    s.node("read", pipeline)
        .with_parcel(parcel, NodeAccess::Read)
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
fn retention_resubmit_bakes_prologue_no_extra_standalone_cb() {
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
    assert_eq!(barrier_buffers(&device).len(), 1, "foreign writer emits hazard barrier");

    clear_mock(&device);
    for _ in 0..3 {
        worker.submit().expect("resubmit");
    }
    // Foreign touch dirties topology: first resubmit re-records (baked prologue), then retention hits.
    assert_eq!(retained_resubmits(&device), 2);
    assert_eq!(worker.replay_stats().topology_records, 1);
    // No barrier-only standalone CB on retained resubmits — one graph submit per resubmit.
    assert_eq!(
        compute_submits(&device),
        3,
        "retained path must not emit an extra standalone prologue CB per resubmit"
    );
    assert!(
        all_graph_syncs_some(&device),
        "all retained-path graph submits must suppress the legacy blanket acquire"
    );
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
fn war_same_context_emits_wait_on_write_after_read() {
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
    assert!(
        barrier_buffers(&device).is_empty(),
        "WAR: same-context write-after-read must not use a baked prologue barrier"
    );
    assert_eq!(
        recorded_waits(&device).last(),
        Some(&vec![goldy::timeline::Epoch {
            context: ctx.test_backend_handle(),
            value: 1
        }]),
        "WAR: writer after reader must wait on the reader's timeline epoch"
    );
}

#[test]
fn war_retained_resubmit_emits_live_wait() {
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

    let mut writer = write_scheme(&ctx, &parcel, &write_pipe);
    writer.submit().expect("writer record");
    assert_eq!(writer.replay_stats().records, 1);

    let mut reader = read_scheme(&ctx, &parcel, &read_pipe);
    reader.submit().expect("reader establishes last_reads");

    if writer.is_topology_dirty() {
        writer.submit().expect("writer settle after foreign read");
        assert!(!writer.is_topology_dirty());
    }

    clear_mock(&device);
    writer.submit().expect("writer retained resubmit");
    assert_eq!(retained_resubmits(&device), 1);
    let waits = recorded_waits(&device).last().cloned().unwrap_or_default();
    assert_eq!(waits.len(), 1, "retained WAR resubmit must restate a live queue wait");
    assert_eq!(waits[0].context, ctx.test_backend_handle());
    assert_eq!(
        waits[0].value, 2,
        "retained WAR resubmit must wait on the reader epoch"
    );
}

fn recorded_graph_syncs(device: &Device) -> Vec<bool> {
    with_mock(device, |m| m.recorded_graph_syncs.clone())
}

/// A minimal render scheme that declares a read dependency on `parcel` via `with_parcel`.
///
/// The render pass itself does nothing interesting (clear + draw 0 verts), but registering
/// the parcel is enough to put a `ResourceSync` in the ledger so the cross-submit analysis
/// produces a non-None `SubmitSync` for the submission.
fn render_read_scheme(ctx: &Context, parcel: &Parcel, pipeline: &RenderPipeline) -> Scheme {
    let mut s = Scheme::new(ctx);
    let rt = s
        .lease_render_target(4, 4, TextureFormat::Rgba8Unorm, None)
        .expect("render target lease");
    let mut pass = s.render_pass("render_read", &rt);
    pass.with_parcel(parcel, NodeAccess::Read);
    pass.set_pipeline(pipeline);
    pass.draw(0..3, 0..1);
    pass.finish();
    s
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

#[test]
fn compute_write_then_render_read_carries_sync_through_graph_submit() {
    // Regression test for the standalone/graph legacy-acquire asymmetry.
    //
    // Before the fix, `backend_submit_graph` called `sync_waits_only(sync)` which
    // returns None when there are no cross-context waits (same-context-only hazards).
    // That meant `submit_graph` was called with `sync = None`, preventing real backends
    // from suppressing the redundant legacy blanket-acquire barrier even though a
    // scoped prologue barrier was already folded into the command list.
    //
    // After the fix, `backend_submit_graph` uses `sync.map(...)` to preserve `Some`
    // whenever the epoch ledger produced a SubmitSync, so `submit_graph` always sees
    // `sync = Some` when there is a tracked hazard.
    let device = mock_device();
    let ctx = mock_ctx(&device);

    let write_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("write_shader");
    let vert_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("vert");
    let frag_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("frag");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("write_pipe");
    let render_pipe =
        RenderPipeline::new(&device, &vert_shader, &frag_shader, &RenderPipelineDesc::default()).expect("render_pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("parcel");

    // Establish a ledger entry: compute scheme writes the parcel.
    let mut writer = write_scheme(&ctx, &parcel, &write_pipe);
    writer.submit().expect("compute write");

    // Now a render-pass scheme reads the same parcel on the same context.
    // The cross-submit analysis sees a RAW hazard and produces a SubmitSync with
    // a scoped prologue barrier.  That Some must survive into submit_graph.
    clear_mock(&device);
    let mut reader = render_read_scheme(&ctx, &parcel, &render_pipe);
    reader.submit().expect("render read");

    // The scoped prologue barrier must appear in the recorded compute commands
    // (folded in as a GraphCommand::Compute before the Render command).
    assert_eq!(
        barrier_buffers(&device).len(),
        1,
        "render-read after compute-write must emit a scoped prologue barrier"
    );

    // submit_graph must have been invoked with sync = Some so that real GPU backends
    // (Vulkan, DX12, Metal) know to suppress the legacy blanket-acquire and rely on
    // the already-folded scoped barrier instead.
    // Before the fix this assertion would fail (false was recorded).
    assert!(
        recorded_graph_syncs(&device).last().copied().unwrap_or(false),
        "submit_graph must receive sync=Some when the epoch ledger has a tracked hazard"
    );
}

fn upload_write_scheme(ctx: &Context, parcel: &Parcel) -> Scheme {
    let mut s = Scheme::new(ctx);
    s.commit_write_parcel(parcel, 0, vec![42, 0, 0, 0])
        .expect("upload write");
    s
}

#[test]
fn topology_independent_parcels_do_not_cross_dirty() {
    let device = mock_device();
    let ctx = mock_ctx(&device);
    let write_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("shader");
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");

    let mut pool = RetainedPool::new(device.clone());
    let parcel_a = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("parcel_a");
    let parcel_b = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("parcel_b");

    let mut reader = read_scheme(&ctx, &parcel_a, &read_pipe);
    reader.submit().expect("reader record");

    let mut writer = write_scheme(&ctx, &parcel_b, &write_pipe);
    writer.submit().expect("writer on unrelated parcel");

    assert!(
        !reader.is_topology_dirty(),
        "unrelated parcel writer must not dirty the reader"
    );
}

#[test]
fn topology_new_foreign_writer_sets_dirty_on_reader() {
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
    reader.submit().expect("reader record");
    assert!(!reader.is_topology_dirty());

    let mut writer = write_scheme(&ctx, &parcel, &write_pipe);
    writer.submit().expect("writer record");

    assert!(
        reader.is_topology_dirty(),
        "new writer on a shared parcel must dirty an existing reader"
    );
}

#[test]
fn topology_same_role_rerecord_does_not_dirty_peers() {
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
    reader.submit().expect("reader record");

    let mut writer = write_scheme(&ctx, &parcel, &write_pipe);
    writer.submit().expect("writer first record");
    assert!(reader.is_topology_dirty());

    reader.submit().expect("reader topology re-record");
    assert!(!reader.is_topology_dirty());

    writer.submit().expect("writer resubmit");
    assert!(
        !reader.is_topology_dirty(),
        "identical writer resubmit must not re-dirty peers"
    );
}

#[test]
fn topology_dropped_scheme_edge_is_pruned() {
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
    reader.submit().expect("reader record");

    {
        let mut writer = write_scheme(&ctx, &parcel, &write_pipe);
        writer.submit().expect("writer record");
    }

    assert!(reader.is_topology_dirty());
    reader.submit().expect("reader topology re-record");
    assert!(!reader.is_topology_dirty());

    let mut replacement = write_scheme(&ctx, &parcel, &write_pipe);
    replacement.submit().expect("replacement writer record");
    assert!(
        reader.is_topology_dirty(),
        "replacement writer with same role still changes interaction membership"
    );
}

#[test]
fn topology_dirty_clears_after_rerecord() {
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
    reader.submit().expect("reader record");

    let mut writer = write_scheme(&ctx, &parcel, &write_pipe);
    writer.submit().expect("writer record");
    assert!(reader.is_topology_dirty());

    reader.submit().expect("reader topology re-record");
    assert!(!reader.is_topology_dirty());
    assert_eq!(reader.replay_stats().topology_records, 1);

    clear_mock(&device);
    reader.submit().expect("reader resubmit");
    assert_eq!(retained_resubmits(&device), 1);
    assert_eq!(reader.replay_stats().records, 2);
    assert_eq!(
        compute_submits(&device),
        1,
        "retained resubmit must not emit a standalone prologue CB"
    );
    assert!(
        all_graph_syncs_some(&device),
        "retained resubmit must pass sync=Some to suppress legacy acquire"
    );
}

#[test]
fn topology_kind_change_on_existing_scheme_dirties_peers() {
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
    reader.submit().expect("reader record");

    let mut compute_writer = write_scheme(&ctx, &parcel, &write_pipe);
    compute_writer.submit().expect("compute writer record");
    assert!(reader.is_topology_dirty());
    reader.submit().expect("reader settle after compute writer");
    assert!(!reader.is_topology_dirty());

    let mut transfer_writer = upload_write_scheme(&ctx, &parcel);
    transfer_writer.submit().expect("transfer writer record");
    assert!(
        reader.is_topology_dirty(),
        "write kind change on a shared parcel must dirty peers"
    );
}
