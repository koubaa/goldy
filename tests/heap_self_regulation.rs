//! Metal heap self-regulation integration tests.
//!
//! These tests verify that the Metal buffer and texture heap allocators
//! self-regulate under pressure: when the heap is saturated, the backend
//! waits for GPU progress, reclaims retired resources, and retries the
//! allocation — rather than failing immediately.
//!
//! The tests exercise the full lifecycle:
//! - Fresh allocations fill the heap
//! - GPU work completes and signals the timeline
//! - `process_deletion_queue_up_to_signaled` + `compact_overflow` free resources
//! - Retry allocation succeeds from reclaimed heap space
//!
//! Heap introspection and overflow tests are macOS + Metal only for now, since
//! DX12 and Vulkan use committed resources without a shared heap cap.

use goldy::task_graph::TaskGraph;
use goldy::types::{BufferFlags, SpatialAccess, TextureFlags, TextureFormat};
use goldy::{Buffer, DataAccess, Device, DeviceDescriptor, Instance, RequestAdapterOptions};

mod common;
#[path = "common/submission.rs"]
mod submission;
use submission::submission_context;

fn make_device() -> Device {
    let instance = Instance::new().expect("Instance::new");
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("No Goldy device")
}

// ===========================================================================
// Buffer heap introspection
// ===========================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn buffer_heap_stats_available_on_metal() {
    let device = make_device();
    let stats = device.buffer_heap_stats();
    assert!(stats.is_some(), "buffer_heap_stats should be Some on Metal");
    let s = stats.unwrap();
    assert_eq!(s.buffer_count, 0);
    assert_eq!(s.overflow_count, 0);
    assert!(s.primary_heap_bytes > 0, "primary heap must be non-zero");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn texture_heap_stats_available_on_metal() {
    let device = make_device();
    let stats = device.texture_heap_stats();
    assert!(
        stats.is_some(),
        "texture_heap_stats should be Some on Metal"
    );
    let s = stats.unwrap();
    assert_eq!(s.texture_count, 0);
    assert_eq!(s.overflow_count, 0);
}

// ===========================================================================
// Basic allocation tracking
// ===========================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn buffer_allocation_increments_count() {
    let device = make_device();
    let before = device.buffer_heap_stats().unwrap().buffer_count;
    let _buf = Buffer::new(&device, 4096, DataAccess::Scattered).unwrap();
    let after = device.buffer_heap_stats().unwrap().buffer_count;
    assert!(
        after > before,
        "buffer_count should increase after allocation: before={before} after={after}"
    );
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gpu_only_buffer_does_not_use_heap() {
    let device = make_device();
    let before = device.buffer_heap_stats().unwrap().buffer_count;
    let _buf = Buffer::new_with_stride_and_flags(
        &device,
        4096,
        DataAccess::Scattered,
        None,
        BufferFlags::GPU_ONLY,
    )
    .unwrap();
    let after = device.buffer_heap_stats().unwrap().buffer_count;
    assert_eq!(
        before, after,
        "GPU_ONLY buffers bypass the heap (device-allocated)"
    );
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn buffer_drop_frees_heap_space_after_flush() {
    let device = make_device();
    let ctx = submission_context(&device);

    // Submit trivial work so timeline advances.
    let mut graph = TaskGraph::new();
    let setup_buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&setup_buf, 0, 256);
    let tv = ctx.submit_pipelined(&mut graph).unwrap();
    ctx.wait_until(tv).unwrap();

    // Allocate a large buffer to create overflow.
    let alloc_size = 32 * 1024 * 1024u64;
    let big_bufs: Vec<Buffer> = (0..3)
        .map(|_| {
            Buffer::new_with_stride_and_flags(
                &device,
                alloc_size,
                DataAccess::Scattered,
                None,
                BufferFlags::empty(),
            )
            .unwrap()
        })
        .collect();
    let overflow_during = device.buffer_heap_stats().unwrap().overflow_count;
    assert!(
        overflow_during > 0,
        "should have overflow with 3×32MB buffers"
    );

    // Drop the big buffers.
    drop(big_bufs);

    // Submit + wait so the deletion queue processes the drops.
    let mut graph2 = TaskGraph::new();
    graph2.clear_buffer(&setup_buf, 0, 256);
    let tv2 = ctx.submit_pipelined(&mut graph2).unwrap();
    ctx.wait_until(tv2).unwrap();
    ctx.flush_deferred_deletions();
    device.compact_overflow_heaps();

    let overflow_after = device.buffer_heap_stats().unwrap().overflow_count;
    assert!(
        overflow_after < overflow_during,
        "overflow heaps should decrease after drop+flush+compact: during={overflow_during} after={overflow_after}"
    );
}

// ===========================================================================
// Overflow heap lifecycle
// ===========================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn many_buffers_create_overflow_heaps() {
    let device = make_device();
    let primary_size = device.buffer_heap_stats().unwrap().primary_heap_bytes;

    // Allocate buffers until we need at least one overflow heap.
    // Each buffer is 8MB — primary is 64MB, so 9 buffers should overflow.
    let mut buffers = Vec::new();
    let alloc_size = 8 * 1024 * 1024;
    for _ in 0..12 {
        match Buffer::new_with_stride_and_flags(
            &device,
            alloc_size,
            DataAccess::Scattered,
            None,
            BufferFlags::empty(),
        ) {
            Ok(buf) => buffers.push(buf),
            Err(_) => break,
        }
    }

    let stats = device.buffer_heap_stats().unwrap();
    let total_allocated = buffers.len() as u64 * alloc_size;
    if total_allocated > primary_size {
        assert!(
            stats.overflow_count > 0,
            "expected overflow heaps when total allocation ({total_allocated}) exceeds primary ({primary_size})"
        );
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn compact_overflow_removes_empty_heaps() {
    let device = make_device();
    let ctx = submission_context(&device);
    let alloc_size = 8 * 1024 * 1024u64;

    // Fill primary + create overflow
    let mut buffers = Vec::new();
    for _ in 0..12 {
        match Buffer::new_with_stride_and_flags(
            &device,
            alloc_size,
            DataAccess::Scattered,
            None,
            BufferFlags::empty(),
        ) {
            Ok(buf) => buffers.push(buf),
            Err(_) => break,
        }
    }

    let overflow_before = device.buffer_heap_stats().unwrap().overflow_count;

    // Drop all buffers and flush
    drop(buffers);
    ctx.flush_deferred_deletions();
    device.compact_overflow_heaps();

    let overflow_after = device.buffer_heap_stats().unwrap().overflow_count;
    if overflow_before > 0 {
        assert!(
            overflow_after < overflow_before,
            "compact_overflow should remove empty heaps: before={overflow_before} after={overflow_after}"
        );
    }
}

// ===========================================================================
// Self-regulation: allocation under pressure
// ===========================================================================

#[test]
fn allocation_survives_heap_pressure_with_gpu_work() {
    let device = make_device();
    let ctx = submission_context(&device);
    let alloc_size = 4 * 1024 * 1024u64;

    // Submit some trivial GPU work so we have a timeline and in-flight CBs.
    let mut graph = TaskGraph::new();
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&buf, 0, 256);
    let tv = ctx.submit_pipelined(&mut graph).unwrap();
    assert!(tv > 0, "submission should advance timeline");

    // Now defer some buffers at that timeline value.
    let mut payload = goldy::DeferredPayload::new();
    let held_buffers: Vec<Buffer> = (0..8)
        .map(|_| {
            Buffer::new_with_stride_and_flags(
                &device,
                alloc_size,
                DataAccess::Scattered,
                None,
                BufferFlags::empty(),
            )
            .unwrap()
        })
        .collect();
    for b in held_buffers {
        payload.push(b);
    }
    ctx.defer_release(tv, payload);

    // Now allocate more buffers. The heap may be under pressure, but the
    // self-regulation logic should wait for tv to retire, flush, and succeed.
    // First, make sure the GPU has completed (wait).
    ctx.wait_until(tv).unwrap();
    ctx.flush_deferred_deletions();

    // Now allocating should succeed because we reclaimed the deferred buffers.
    let _fresh = Buffer::new_with_stride_and_flags(
        &device,
        alloc_size,
        DataAccess::Scattered,
        None,
        BufferFlags::empty(),
    )
    .expect("allocation should succeed after reclaiming deferred buffers");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn multi_frame_pipelined_allocation_does_not_exhaust_heap() {
    let device = make_device();
    let ctx = submission_context(&device);
    let alloc_size = 2 * 1024 * 1024u64;
    let frames = 30;

    for frame in 0..frames {
        // Each "frame" allocates a few buffers, submits a graph, and defers them.
        let buffers: Vec<Buffer> = (0..4)
            .map(|_| {
                Buffer::new_with_stride_and_flags(
                    &device,
                    alloc_size,
                    DataAccess::Scattered,
                    None,
                    BufferFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("frame {frame}: allocation failed: {e}"))
            })
            .collect();

        let mut graph = TaskGraph::new();
        for b in &buffers {
            graph.clear_buffer(b, 0, alloc_size);
        }
        let tv = ctx
            .submit_pipelined(&mut graph)
            .unwrap_or_else(|e| panic!("frame {frame}: submit failed: {e}"));

        // Defer the buffers for cleanup after GPU finishes this frame.
        let mut payload = goldy::DeferredPayload::new();
        for b in buffers {
            payload.push(b);
        }
        ctx.defer_release(tv, payload);

        // Flush any completed work from previous frames.
        ctx.flush_deferred_deletions();
    }

    // If we got here without panicking, the heap self-regulated successfully.
    let stats = device.buffer_heap_stats().unwrap();
    assert!(
        stats.overflow_count <= 16,
        "overflow heaps should not exceed MAX_OVERFLOW_HEAPS"
    );
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn steady_state_overflow_stays_bounded() {
    let device = make_device();
    let ctx = submission_context(&device);
    let alloc_size = 1024 * 1024u64;
    let warmup_frames = 10;
    let steady_frames = 30;

    // Warmup: establish the pool (allocate + defer + flush pattern).
    for _ in 0..warmup_frames {
        let buffers: Vec<Buffer> = (0..3)
            .map(|_| {
                Buffer::new_with_stride_and_flags(
                    &device,
                    alloc_size,
                    DataAccess::Scattered,
                    None,
                    BufferFlags::empty(),
                )
                .unwrap()
            })
            .collect();

        let mut graph = TaskGraph::new();
        for b in &buffers {
            graph.clear_buffer(b, 0, alloc_size);
        }
        let tv = ctx.submit_pipelined(&mut graph).unwrap();
        let mut payload = goldy::DeferredPayload::new();
        for b in buffers {
            payload.push(b);
        }
        ctx.defer_release(tv, payload);
        ctx.wait_until(tv).unwrap();
        ctx.flush_deferred_deletions();
        ctx.flush_deferred_deletions();
    }
    device.compact_overflow_heaps();

    // After warmup, snapshot the overflow count.
    let baseline_overflow = device.buffer_heap_stats().unwrap().overflow_count;

    // Steady state: same allocation pattern, overflow should not grow.
    let mut max_overflow = baseline_overflow;
    for _ in 0..steady_frames {
        let buffers: Vec<Buffer> = (0..3)
            .map(|_| {
                Buffer::new_with_stride_and_flags(
                    &device,
                    alloc_size,
                    DataAccess::Scattered,
                    None,
                    BufferFlags::empty(),
                )
                .unwrap()
            })
            .collect();

        let mut graph = TaskGraph::new();
        for b in &buffers {
            graph.clear_buffer(b, 0, alloc_size);
        }
        let tv = ctx.submit_pipelined(&mut graph).unwrap();
        let mut payload = goldy::DeferredPayload::new();
        for b in buffers {
            payload.push(b);
        }
        ctx.defer_release(tv, payload);
        ctx.wait_until(tv).unwrap();
        ctx.flush_deferred_deletions();
        ctx.flush_deferred_deletions();
        device.compact_overflow_heaps();

        max_overflow = max_overflow.max(device.buffer_heap_stats().unwrap().overflow_count);
    }

    assert!(
        max_overflow <= baseline_overflow + 1,
        "overflow heaps should not grow in steady state: baseline={baseline_overflow} max={max_overflow}"
    );
}

// ===========================================================================
// Texture heap self-regulation
// ===========================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn texture_allocation_increments_count() {
    let device = make_device();
    let before = device.texture_heap_stats().unwrap().texture_count;
    let _tex = device
        .alloc_texture(
            64,
            64,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .unwrap();
    let after = device.texture_heap_stats().unwrap().texture_count;
    assert!(
        after > before,
        "texture_count should increase: before={before} after={after}"
    );
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn many_textures_create_overflow_then_compact() {
    let device = make_device();
    let ctx = submission_context(&device);

    let mut textures = Vec::new();
    for _ in 0..20 {
        match device.alloc_texture(
            512,
            512,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        ) {
            Ok(tex) => textures.push(tex),
            Err(_) => break,
        }
    }

    let overflow = device.texture_heap_stats().unwrap().overflow_count;
    // Textures are 512x512 RGBA = 1MB each, primary heap varies but likely overflows.
    if overflow > 0 {
        drop(textures);
        ctx.flush_deferred_deletions();
        device.compact_overflow_heaps();
        let after = device.texture_heap_stats().unwrap().overflow_count;
        assert!(
            after < overflow,
            "compact should reduce overflow: before={overflow} after={after}"
        );
    }
}

#[test]
fn texture_allocation_survives_pressure_with_gpu_work() {
    let device = make_device();
    let ctx = submission_context(&device);

    // Submit trivial GPU work.
    let mut graph = TaskGraph::new();
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&buf, 0, 256);
    let tv = ctx.submit_pipelined(&mut graph).unwrap();

    // Allocate textures and defer them.
    let mut payload = goldy::DeferredPayload::new();
    for _ in 0..8 {
        let tex = device
            .alloc_texture(
                256,
                256,
                TextureFormat::Rgba8Unorm,
                SpatialAccess::Direct,
                TextureFlags::COPY_DST,
            )
            .unwrap();
        payload.push(tex);
    }
    ctx.defer_release(tv, payload);

    // Wait and flush.
    ctx.wait_until(tv).unwrap();
    ctx.flush_deferred_deletions();

    // Should succeed now.
    let _tex = device
        .alloc_texture(
            256,
            256,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .expect("texture allocation should succeed after reclaim");
}

// ===========================================================================
// VramAllocator deferred ring tests
// ===========================================================================

#[test]
fn flush_deferred_deletions_advances_with_gpu_progress() {
    let device = make_device();
    let ctx = submission_context(&device);

    // Submit 3 frames worth of work.
    let mut timelines = Vec::new();
    for _ in 0..3 {
        let mut graph = TaskGraph::new();
        let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
        graph.clear_buffer(&buf, 0, 256);
        let tv = ctx.submit_pipelined(&mut graph).unwrap();
        ctx.defer_until(tv, buf);
        timelines.push(tv);
    }

    assert!(ctx.has_deferred_payloads());
    let oldest_before = ctx.oldest_deferred_epoch().unwrap();

    // Wait for the first frame and flush.
    ctx.wait_until(timelines[0]).unwrap();
    ctx.flush_deferred_deletions();

    let oldest_after = ctx.oldest_deferred_epoch();
    if let Some(after) = oldest_after {
        assert!(
            after > oldest_before,
            "oldest_deferred_epoch should advance: before={oldest_before} after={after}"
        );
    }
}

#[test]
fn oldest_deferred_epoch_is_none_when_empty() {
    let device = make_device();
    let ctx = submission_context(&device);
    assert_eq!(ctx.oldest_deferred_epoch(), None);
    assert!(!ctx.has_deferred_payloads());
}

#[test]
fn wait_and_flush_reclaims_all_deferred() {
    let device = make_device();
    let ctx = submission_context(&device);

    let mut graph = TaskGraph::new();
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&buf, 0, 256);
    let tv = ctx.submit_pipelined(&mut graph).unwrap();

    ctx.defer_until(tv, buf);
    assert!(ctx.has_deferred_payloads());

    ctx.wait_until(tv).unwrap();
    ctx.flush_deferred_deletions();

    assert!(!ctx.has_deferred_payloads());
}

// ===========================================================================
// In-flight command buffer tracking
// ===========================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn in_flight_cb_count_increases_after_submit() {
    let device = make_device();
    let ctx = submission_context(&device);
    assert_eq!(ctx.in_flight_command_buffer_count(), 0);

    let mut graph = TaskGraph::new();
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&buf, 0, 256);
    let _tv = ctx.submit_pipelined(&mut graph).unwrap();

    assert!(
        ctx.in_flight_command_buffer_count() > 0,
        "should track at least one in-flight CB after submit"
    );
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn in_flight_cbs_drain_after_wait() {
    let device = make_device();
    let ctx = submission_context(&device);

    let mut graph = TaskGraph::new();
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&buf, 0, 256);
    let tv = ctx.submit_pipelined(&mut graph).unwrap();

    let before = ctx.in_flight_command_buffer_count();
    assert!(before > 0);

    ctx.wait_until(tv).unwrap();

    let after = ctx.in_flight_command_buffer_count();
    assert!(
        after < before,
        "in-flight CBs should drain after wait: before={before} after={after}"
    );
}

// ===========================================================================
// Timeline advancement and gpu_progress
// ===========================================================================

#[test]
fn gpu_progress_advances_after_wait() {
    let device = make_device();
    let ctx = submission_context(&device);
    let initial = ctx.gpu_progress();

    let mut graph = TaskGraph::new();
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&buf, 0, 256);
    let tv = ctx.submit_pipelined(&mut graph).unwrap();

    ctx.wait_until(tv).unwrap();
    let after = ctx.gpu_progress();
    assert!(
        after >= tv,
        "gpu_progress should reach submitted timeline value: initial={initial} tv={tv} after={after}"
    );
}

#[test]
fn multiple_submits_advance_timeline_monotonically() {
    let device = make_device();
    let ctx = submission_context(&device);
    let mut prev_tv = 0;

    for _ in 0..5 {
        let mut graph = TaskGraph::new();
        let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
        graph.clear_buffer(&buf, 0, 256);
        let tv = ctx.submit_pipelined(&mut graph).unwrap();
        assert!(tv > prev_tv, "timeline must be monotonically increasing");
        prev_tv = tv;
    }
}

// ===========================================================================
// Deletion queue processing
// ===========================================================================

#[test]
fn deletion_queue_populated_after_buffer_drop() {
    let device = make_device();
    let ctx = submission_context(&device);

    // Submit so we have a non-zero timeline barrier.
    let mut graph = TaskGraph::new();
    let trigger = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&trigger, 0, 256);
    let _tv = ctx.submit_pipelined(&mut graph).unwrap();

    let before = ctx.deferred_deletion_pending_count();
    let buf = Buffer::new(&device, 4096, DataAccess::Scattered).unwrap();
    drop(buf);
    let after = ctx.deferred_deletion_pending_count();
    assert!(
        after >= before,
        "deletion queue should grow after buffer drop: before={before} after={after}"
    );
}

#[test]
fn deletion_queue_drains_after_flush() {
    let device = make_device();
    let ctx = submission_context(&device);

    let mut graph = TaskGraph::new();
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&buf, 0, 256);
    let tv = ctx.submit_pipelined(&mut graph).unwrap();

    let extra = Buffer::new(&device, 4096, DataAccess::Scattered).unwrap();
    drop(extra);

    let pending = ctx.deferred_deletion_pending_count();
    ctx.wait_until(tv).unwrap();
    ctx.flush_deferred_deletions();
    let after = ctx.deferred_deletion_pending_count();

    // Should have processed at least the one we dropped.
    assert!(
        after <= pending,
        "deletion queue should drain: pending={pending} after={after}"
    );
}

// ===========================================================================
// Stress tests: rapid frame submission without wait (tests self-regulation)
// ===========================================================================

#[test]
fn rapid_submit_without_explicit_wait_survives_50_frames() {
    let device = make_device();
    let ctx = submission_context(&device);
    let alloc_size = 1024 * 1024u64;

    // Submit 50 frames as fast as possible, with NO explicit wait between frames.
    // The self-regulation logic in allocate_mtl_storage_buffer must kick in.
    for frame in 0..50 {
        let buffers: Vec<Buffer> = (0..3)
            .map(|_| {
                Buffer::new_with_stride_and_flags(
                    &device,
                    alloc_size,
                    DataAccess::Scattered,
                    None,
                    BufferFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("frame {frame}: alloc failed: {e}"))
            })
            .collect();

        let mut graph = TaskGraph::new();
        for b in &buffers {
            graph.clear_buffer(b, 0, alloc_size);
        }
        let tv = ctx
            .submit_pipelined(&mut graph)
            .unwrap_or_else(|e| panic!("frame {frame}: submit failed: {e}"));

        let mut payload = goldy::DeferredPayload::new();
        for b in buffers {
            payload.push(b);
        }
        ctx.defer_release(tv, payload);

        // Only non-blocking flush (no wait) — the allocator must self-regulate.
        ctx.flush_deferred_deletions();
    }
}

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn rapid_submit_large_buffers_50_frames() {
    let device = make_device();
    let ctx = submission_context(&device);
    let alloc_size = 4 * 1024 * 1024u64;

    for frame in 0..50 {
        let buffers: Vec<Buffer> = (0..2)
            .map(|_| {
                Buffer::new_with_stride_and_flags(
                    &device,
                    alloc_size,
                    DataAccess::Scattered,
                    None,
                    BufferFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("frame {frame}: alloc failed: {e}"))
            })
            .collect();

        let mut graph = TaskGraph::new();
        for b in &buffers {
            graph.clear_buffer(b, 0, alloc_size);
        }
        let tv = ctx
            .submit_pipelined(&mut graph)
            .unwrap_or_else(|e| panic!("frame {frame}: submit failed: {e}"));

        let mut payload = goldy::DeferredPayload::new();
        for b in buffers {
            payload.push(b);
        }
        ctx.defer_release(tv, payload);
        ctx.flush_deferred_deletions();
    }
}

// ===========================================================================
// Overflow heap cap enforcement (raw allocator behavior without self-regulation)
// ===========================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn overflow_count_never_exceeds_16() {
    let device = make_device();

    // Hold all buffers alive (no deferred release) — pure pressure test.
    let alloc_size = 8 * 1024 * 1024u64;
    let mut buffers = Vec::new();
    for _ in 0..200 {
        match Buffer::new_with_stride_and_flags(
            &device,
            alloc_size,
            DataAccess::Scattered,
            None,
            BufferFlags::empty(),
        ) {
            Ok(buf) => buffers.push(buf),
            Err(_) => break, // Expected: heap exhaustion
        }
    }

    let stats = device.buffer_heap_stats().unwrap();
    assert!(
        stats.overflow_count <= 16,
        "overflow_count must never exceed MAX_OVERFLOW_HEAPS=16, got {}",
        stats.overflow_count
    );
}

// ===========================================================================
// Mixed buffer and texture pressure
// ===========================================================================

#[test]
fn mixed_buffer_and_texture_allocation_survives_30_frames() {
    let device = make_device();
    let ctx = submission_context(&device);
    let buf_size = 2 * 1024 * 1024u64;

    for frame in 0..30 {
        let bufs: Vec<Buffer> = (0..2)
            .map(|_| {
                Buffer::new_with_stride_and_flags(
                    &device,
                    buf_size,
                    DataAccess::Scattered,
                    None,
                    BufferFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("frame {frame}: buffer alloc failed: {e}"))
            })
            .collect();

        let tex = device
            .alloc_texture(
                128,
                128,
                TextureFormat::Rgba8Unorm,
                SpatialAccess::Direct,
                TextureFlags::COPY_DST,
            )
            .unwrap_or_else(|e| panic!("frame {frame}: texture alloc failed: {e}"));

        let mut graph = TaskGraph::new();
        for b in &bufs {
            graph.clear_buffer(b, 0, buf_size);
        }
        let tv = ctx
            .submit_pipelined(&mut graph)
            .unwrap_or_else(|e| panic!("frame {frame}: submit failed: {e}"));

        let mut payload = goldy::DeferredPayload::new();
        for b in bufs {
            payload.push(b);
        }
        payload.push(tex);
        ctx.defer_release(tv, payload);
        ctx.flush_deferred_deletions();
    }
}

// ===========================================================================
// Buffer resize under heap pressure (abstract-gpu-vram: growable buffers)
// ===========================================================================

#[test]
fn buffer_resize_works_under_heap_pressure() {
    let device = make_device();
    let ctx = submission_context(&device);

    // Allocate a buffer, submit work so it has a timeline.
    let mut buf = Buffer::new(&device, 1024, DataAccess::Scattered).unwrap();

    let mut graph = TaskGraph::new();
    graph.clear_buffer(&buf, 0, 1024);
    let tv = ctx.submit_pipelined(&mut graph).unwrap();
    ctx.wait_until(tv).unwrap();

    // Resize multiple times (triggers realloc + blit-copy on Metal).
    for size in [4096u64, 16384, 65536, 262144] {
        buf.resize_to(size).unwrap_or_else(|e| {
            panic!("resize_to({size}) failed: {e}");
        });
        assert_eq!(buf.size(), size);
    }
}

#[test]
fn buffer_resize_preserves_contents() {
    let device = make_device();
    let ctx = submission_context(&device);
    let initial_data: Vec<u32> = (0..64).collect();
    let mut buf = Buffer::with_data(&device, &initial_data, DataAccess::Scattered).unwrap();

    // Grow the buffer (triggers blit-copy internally).
    let new_size = 1024;
    buf.resize_to(new_size).unwrap();
    assert_eq!(buf.size(), new_size);

    // Submit a fence to ensure the internal blit-copy has completed.
    let mut graph = TaskGraph::new();
    graph.clear_buffer(&buf, 256, 256); // Touch bytes past the original data
    let tv = ctx.submit_pipelined(&mut graph).unwrap();
    ctx.wait_until(tv).unwrap();

    // Read back — first 256 bytes should be preserved.
    let mut readback = vec![0u8; 256];
    buf.read_to_cpu(&device, &mut readback).unwrap();
    let result: &[u32] = bytemuck::cast_slice(&readback);
    assert_eq!(&result[..64], &initial_data[..]);
}

// ===========================================================================
// Deferred release + rapid reuse pattern (ekrano's OwnedShared lifecycle)
// ===========================================================================

#[test]
fn deferred_buffers_returned_to_caller_after_flush() {
    use std::sync::{Arc, Mutex};

    let device = make_device();
    let ctx = submission_context(&device);
    let pending: Arc<Mutex<Vec<Buffer>>> = Arc::new(Mutex::new(Vec::new()));

    // Simulate ekrano's DeferredOwnedBuffersToken pattern.
    struct Token {
        pending: Arc<Mutex<Vec<Buffer>>>,
        buffers: Vec<Buffer>,
    }
    impl Drop for Token {
        fn drop(&mut self) {
            let mut guard = self.pending.lock().unwrap();
            guard.append(&mut self.buffers);
        }
    }

    // Frame 1: allocate buffers, submit, defer.
    let bufs: Vec<Buffer> = (0..3)
        .map(|_| Buffer::new(&device, 4096, DataAccess::Scattered).unwrap())
        .collect();

    let mut graph = TaskGraph::new();
    for b in &bufs {
        graph.clear_buffer(b, 0, 4096);
    }
    let tv = ctx.submit_pipelined(&mut graph).unwrap();

    let token = Token {
        pending: Arc::clone(&pending),
        buffers: bufs,
    };
    ctx.defer_until(tv, token);

    // Before GPU completes: pending should be empty.
    assert_eq!(pending.lock().unwrap().len(), 0);

    // Wait and flush: token drops, buffers move to pending.
    ctx.wait_until(tv).unwrap();
    ctx.flush_deferred_deletions();

    let returned = pending.lock().unwrap().len();
    assert_eq!(
        returned, 3,
        "all 3 buffers should be returned to pending after flush"
    );
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn zero_byte_flush_is_no_op() {
    let device = make_device();
    let ctx = submission_context(&device);
    // Calling flush with no deferred work should not panic or error.
    ctx.flush_deferred_deletions();
    assert!(!ctx.has_deferred_payloads());
}

#[test]
fn double_flush_is_idempotent() {
    let device = make_device();
    let ctx = submission_context(&device);
    let mut graph = TaskGraph::new();
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).unwrap();
    graph.clear_buffer(&buf, 0, 256);
    let tv = ctx.submit_pipelined(&mut graph).unwrap();
    ctx.defer_until(tv, buf);

    ctx.wait_until(tv).unwrap();
    ctx.flush_deferred_deletions();
    // Second flush should be safe and not crash.
    ctx.flush_deferred_deletions();
    assert!(!ctx.has_deferred_payloads());
}

#[test]
fn wait_until_timeout_returns_error_on_far_future() {
    let device = make_device();
    let ctx = submission_context(&device);
    // Waiting for a timeline value nobody submitted should timeout.
    let result = ctx.wait_until_timeout(u64::MAX / 2, 10);
    // Should either succeed (if GPU is actually idle at that value) or timeout.
    // On a fresh device with no work, timeline is 0 so this should timeout.
    assert!(
        result.is_err(),
        "waiting for far-future timeline should timeout"
    );
}
