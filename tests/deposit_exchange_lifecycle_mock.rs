//! Exchange-owned deposit staging lifecycle (mock backend).
//!
//! Covers A/B/A recycling, in-flight isolation, best-fit reuse, claim/consume
//! failure paths, and destination-ledger isolation of staging backings.

use goldy::test_support::{mock_barrier_buffer_count, mock_device, mock_reset_tracking, CbReuseOverride};
use goldy::{
    BufferKind, ComputePipeline, DepositTarget, MemoryExchange, NodeAccess, RetainedPool, Scheme, ShaderModule,
};

const READ_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(BufRO<uint> input) {
    uint x = input[0];
    (void)x;
}
"#;

fn bind_and_write(
    scheme: &mut Scheme,
    dest: &goldy::Parcel,
    capacity: u64,
    payload: &[u8],
) -> goldy::DepositTransaction {
    let deposit = MemoryExchange::new(scheme.context())
        .bind_deposit(scheme, DepositTarget::buffer(dest, capacity))
        .expect("bind_deposit");
    deposit.write(0, payload).expect("deposit write");
    deposit
}

#[test]
fn aba_settled_deposits_share_one_backing_and_replay() {
    let _cb = CbReuseOverride::force_enabled();
    let device = mock_device();
    let ctx = device.create_context().unwrap();
    let mut pool = RetainedPool::new(device.clone());
    let dest_a = pool
        .acquire_buffer(
            64,
            BufferKind::Scattered,
            Some(4),
            goldy::types::BufferFlags::empty(),
            None,
        )
        .unwrap();
    let dest_b = pool
        .acquire_buffer(
            64,
            BufferKind::Scattered,
            Some(4),
            goldy::types::BufferFlags::empty(),
            None,
        )
        .unwrap();

    let mut scheme_a = Scheme::new(&ctx);
    let deposit_a = MemoryExchange::new(&ctx)
        .bind_deposit(&mut scheme_a, DepositTarget::buffer(dest_a.whole(), 64))
        .unwrap();
    let mut scheme_b = Scheme::new(&ctx);
    let deposit_b = MemoryExchange::new(&ctx)
        .bind_deposit(&mut scheme_b, DepositTarget::buffer(dest_b.whole(), 64))
        .unwrap();

    let payload_a = [1u8; 64];
    let payload_b = [2u8; 64];
    const N: u32 = 100;

    for i in 0..N {
        deposit_a.write(0, &payload_a).unwrap();
        scheme_a.submit().unwrap();
        if i == 0 {
            assert_eq!(ctx.deposit_staging_alloc_count(), 1);
            assert_eq!(scheme_a.replay_stats().records, 1);
        }
    }
    #[cfg(not(feature = "metal"))]
    assert_eq!(scheme_a.replay_stats().resubmit_hits, (N - 1) as u64);

    for i in 0..N {
        deposit_b.write(0, &payload_b).unwrap();
        scheme_b.submit().unwrap();
        if i == 0 {
            assert_eq!(
                ctx.deposit_staging_alloc_count(),
                1,
                "settled compatible B must reuse A's backing"
            );
            assert_eq!(scheme_b.replay_stats().records, 1);
        }
    }
    #[cfg(not(feature = "metal"))]
    assert_eq!(scheme_b.replay_stats().resubmit_hits, (N - 1) as u64);

    let records_before_return = scheme_a.replay_stats().records;
    for _ in 0..N {
        deposit_a.write(0, &payload_a).unwrap();
        scheme_a.submit().unwrap();
    }
    assert_eq!(ctx.deposit_staging_alloc_count(), 1, "returning to A must not allocate");
    let _ = records_before_return;
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        scheme_a.replay_stats().records,
        records_before_return,
        "returning to A must not re-record"
    );
}

#[test]
fn inflight_backing_forces_extra_alloc_then_reuses() {
    let device = mock_device();
    let ctx = device.create_context().unwrap();
    let mut pool = RetainedPool::new(device.clone());
    let dest_a = pool
        .acquire_buffer(
            32,
            BufferKind::Scattered,
            Some(4),
            goldy::types::BufferFlags::empty(),
            None,
        )
        .unwrap();
    let dest_b = pool
        .acquire_buffer(
            32,
            BufferKind::Scattered,
            Some(4),
            goldy::types::BufferFlags::empty(),
            None,
        )
        .unwrap();

    let mut scheme_a = Scheme::new(&ctx);
    let deposit_a = MemoryExchange::new(&ctx)
        .bind_deposit(&mut scheme_a, DepositTarget::buffer(dest_a.whole(), 32))
        .unwrap();
    deposit_a.write(0, &[1u8; 32]).unwrap();
    scheme_a.submit().unwrap();
    assert_eq!(ctx.deposit_staging_alloc_count(), 1);

    scheme_a.test_mark_deposit_inflight(&deposit_a, 1_000_000);

    let mut scheme_b = Scheme::new(&ctx);
    let deposit_b = MemoryExchange::new(&ctx)
        .bind_deposit(&mut scheme_b, DepositTarget::buffer(dest_b.whole(), 32))
        .unwrap();
    deposit_b.write(0, &[2u8; 32]).unwrap();
    assert_eq!(
        ctx.deposit_staging_alloc_count(),
        2,
        "in-flight A must force B to allocate"
    );
    scheme_b.submit().unwrap();

    // Retire A's original backing so a later write can reuse the pool.
    // Mock progress already exceeds 0; clearing the forced ready_after by
    // parking at 0 via a drop-path is not needed — wait until B settles and
    // A writes again after we drop the forced inflight by submitting A once
    // A can take B's retired backing or its own once progress catches it.
    // Force both parked backings ready by using a huge progress skip: mark
    // neither inflight (B already submitted; A still marked). Re-park A at 0
    // by writing A after we clear inflight via a second mark at tv=0.
    scheme_a.test_mark_deposit_inflight(&deposit_a, 0);
    deposit_a.write(0, &[3u8; 32]).unwrap();
    scheme_a.submit().unwrap();
    assert_eq!(
        ctx.deposit_staging_alloc_count(),
        2,
        "after retirement, further writes reuse existing backings"
    );
}

#[test]
fn best_fit_reuses_larger_backing_and_rejects_undersized() {
    let device = mock_device();
    let ctx = device.create_context().unwrap();
    let mut pool = RetainedPool::new(device.clone());
    let large = pool
        .acquire_buffer(
            64,
            BufferKind::Scattered,
            Some(4),
            goldy::types::BufferFlags::empty(),
            None,
        )
        .unwrap();
    let small = pool
        .acquire_buffer(
            16,
            BufferKind::Scattered,
            Some(4),
            goldy::types::BufferFlags::empty(),
            None,
        )
        .unwrap();

    let mut scheme_large = Scheme::new(&ctx);
    let d_large = MemoryExchange::new(&ctx)
        .bind_deposit(&mut scheme_large, DepositTarget::buffer(large.whole(), 64))
        .unwrap();
    d_large.write(0, &[7u8; 64]).unwrap();
    scheme_large.submit().unwrap();
    assert_eq!(ctx.deposit_staging_alloc_count(), 1);

    let mut scheme_small = Scheme::new(&ctx);
    let d_small = MemoryExchange::new(&ctx)
        .bind_deposit(&mut scheme_small, DepositTarget::buffer(small.whole(), 16))
        .unwrap();
    d_small.write(0, &[8u8; 16]).unwrap();
    assert_eq!(
        ctx.deposit_staging_alloc_count(),
        1,
        "larger backing must satisfy a smaller compatible target"
    );
    scheme_small.submit().unwrap();

    let ctx2 = device.create_context().unwrap();
    let mut pool2 = RetainedPool::new(device.clone());
    let tiny = pool2
        .acquire_buffer(
            16,
            BufferKind::Scattered,
            Some(4),
            goldy::types::BufferFlags::empty(),
            None,
        )
        .unwrap();
    let huge = pool2
        .acquire_buffer(
            64,
            BufferKind::Scattered,
            Some(4),
            goldy::types::BufferFlags::empty(),
            None,
        )
        .unwrap();
    let mut s_tiny = Scheme::new(&ctx2);
    let d_t = MemoryExchange::new(&ctx2)
        .bind_deposit(&mut s_tiny, DepositTarget::buffer(tiny.whole(), 16))
        .unwrap();
    d_t.write(0, &[1u8; 16]).unwrap();
    s_tiny.submit().unwrap();
    assert_eq!(ctx2.deposit_staging_alloc_count(), 1);

    let mut s_huge = Scheme::new(&ctx2);
    let d_h = MemoryExchange::new(&ctx2)
        .bind_deposit(&mut s_huge, DepositTarget::buffer(huge.whole(), 64))
        .unwrap();
    d_h.write(0, &[9u8; 64]).unwrap();
    assert_eq!(
        ctx2.deposit_staging_alloc_count(),
        2,
        "undersized backing must not satisfy a larger target"
    );
    s_huge.submit().unwrap();
}

#[test]
fn claim_paths_write_drop_and_repeat() {
    let device = mock_device();
    let ctx = device.create_context().unwrap();
    let mut pool = RetainedPool::new(device.clone());
    let dst = pool
        .acquire_buffer(
            16,
            BufferKind::Scattered,
            Some(4),
            goldy::types::BufferFlags::empty(),
            None,
        )
        .unwrap();

    let mut scheme = Scheme::new(&ctx);
    let deposit = MemoryExchange::new(&ctx)
        .bind_deposit(&mut scheme, DepositTarget::buffer(dst.whole(), 16))
        .unwrap();

    let err = scheme.submit().expect_err("submit without write");
    assert!(format!("{err}").contains("was not written"));

    deposit.write(0, &[1u8; 16]).unwrap();
    deposit.write(0, &[2u8; 16]).unwrap();
    assert_eq!(
        ctx.deposit_staging_alloc_count(),
        1,
        "repeated writes before submit reuse the pending backing"
    );
    scheme.submit().unwrap();

    // Unsubmitted drop returns the pending backing immediately.
    let mut pending = Scheme::new(&ctx);
    let d2 = MemoryExchange::new(&ctx)
        .bind_deposit(&mut pending, DepositTarget::buffer(dst.whole(), 16))
        .unwrap();
    d2.write(0, &[3u8; 16]).unwrap();
    let alloc_before_drop = ctx.deposit_staging_alloc_count();
    drop(pending);
    let mut scheme3 = Scheme::new(&ctx);
    let d3 = MemoryExchange::new(&ctx)
        .bind_deposit(&mut scheme3, DepositTarget::buffer(dst.whole(), 16))
        .unwrap();
    d3.write(0, &[4u8; 16]).unwrap();
    assert_eq!(
        ctx.deposit_staging_alloc_count(),
        alloc_before_drop,
        "drop of an unsubmitted binding must return the backing for reuse"
    );
    scheme3.submit().unwrap();
}

#[test]
fn staging_absent_from_ledger_destination_raw_enforced() {
    let device = mock_device();
    let ctx = device.create_context().unwrap();
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");
    let mut pool = RetainedPool::new(device.clone());
    let dest_a = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("dest_a");
    let dest_b = pool
        .acquire_buffer_with_data(&[0u32; 4], BufferKind::Scattered)
        .expect("dest_b");

    let mut upload_a = Scheme::new(&ctx);
    let _ = bind_and_write(&mut upload_a, dest_a.whole(), 4, &[1, 0, 0, 0]);
    upload_a.submit().expect("upload a");

    mock_reset_tracking(&device);
    let mut consumer = Scheme::new(&ctx);
    consumer
        .node("read", &read_pipe)
        .with_parcel(dest_a.whole(), NodeAccess::Read)
        .dispatch(1, 1, 1);
    consumer.submit().expect("consumer");
    assert_eq!(
        mock_barrier_buffer_count(&device),
        1,
        "destination RAW/WAR must still be ledger-tracked"
    );

    mock_reset_tracking(&device);
    let mut upload_b = Scheme::new(&ctx);
    let _ = bind_and_write(&mut upload_b, dest_b.whole(), 4, &[2, 0, 0, 0]);
    upload_b.submit().expect("upload b");
    assert_eq!(
        mock_barrier_buffer_count(&device),
        0,
        "shared exchange staging must not appear as a cross-scheme ledger key"
    );
}
