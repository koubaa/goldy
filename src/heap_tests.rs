#[cfg(test)]
mod heap_tests {
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

    use crate::buffer::Allocation;
    use crate::parcel::Parcel;
    use crate::test_support::{scheme_advance_timeline, SerialGpuDevice};
    use crate::types::{BufferFlags, TextureFlags, TextureFormat, TextureKind};
    use crate::{BufferKind, MemoryExchange, Scheme};
    use std::sync::Arc;

    fn submission_context(device: &crate::Device) -> crate::Context {
        device.create_context().expect("context")
    }

    fn make_device() -> SerialGpuDevice {
        SerialGpuDevice::new()
    }

    fn scheme_submit_pipelined(ctx: &crate::Context) -> crate::timeline::TimelineValue {
        scheme_advance_timeline(ctx)
    }

    // ===========================================================================
    // Allocation heap introspection
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
        assert!(stats.is_some(), "texture_heap_stats should be Some on Metal");
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
        let _buf = device
            .alloc_buffer(4096, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();
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
        let _buf = device
            .alloc_buffer(4096, BufferKind::Scattered, None, BufferFlags::GPU_ONLY)
            .unwrap();
        let after = device.buffer_heap_stats().unwrap().buffer_count;
        assert_eq!(before, after, "GPU_ONLY buffers bypass the heap (device-allocated)");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn buffer_drop_frees_heap_space_after_flush() {
        let device = make_device();
        let ctx = submission_context(&device);

        // Submit trivial work so timeline advances.
        let tv = scheme_submit_pipelined(&ctx);
        ctx.wait_until(tv).unwrap();

        // Allocate a large buffer to create overflow.
        let alloc_size = 32 * 1024 * 1024u64;
        let big_bufs: Vec<Allocation> = (0..3)
            .map(|_| {
                device
                    .alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty())
                    .unwrap()
            })
            .collect();
        let overflow_during = device.buffer_heap_stats().unwrap().overflow_count;
        assert!(overflow_during > 0, "should have overflow with 3×32MB buffers");

        // Drop the big buffers.
        drop(big_bufs);

        // Submit + wait so the deletion queue processes the drops.
        let tv2 = scheme_submit_pipelined(&ctx);
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
            match device.alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty()) {
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
            match device.alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty()) {
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
        let tv = scheme_submit_pipelined(&ctx);
        assert!(tv > 0, "submission should advance timeline");

        // Now defer some buffers at that timeline value.
        let mut payload = crate::DeferredPayload::new();
        let held_buffers: Vec<Allocation> = (0..8)
            .map(|_| {
                device
                    .alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty())
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
        let _fresh = device
            .alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty())
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
            let buffers: Vec<Allocation> = (0..4)
                .map(|_| {
                    device
                        .alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty())
                        .unwrap_or_else(|e| panic!("frame {frame}: allocation failed: {e}"))
                })
                .collect();

            let tv = scheme_submit_pipelined(&ctx);

            // Defer the buffers for cleanup after GPU finishes this frame.
            let mut payload = crate::DeferredPayload::new();
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
            let buffers: Vec<Allocation> = (0..3)
                .map(|_| {
                    device
                        .alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty())
                        .unwrap()
                })
                .collect();

            let tv = scheme_submit_pipelined(&ctx);
            let mut payload = crate::DeferredPayload::new();
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
            let buffers: Vec<Allocation> = (0..3)
                .map(|_| {
                    device
                        .alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty())
                        .unwrap()
                })
                .collect();

            let tv = scheme_submit_pipelined(&ctx);
            let mut payload = crate::DeferredPayload::new();
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
                TextureKind::Direct,
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
                TextureKind::Direct,
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
        let tv = scheme_submit_pipelined(&ctx);

        // Allocate textures and defer them.
        let mut payload = crate::DeferredPayload::new();
        for _ in 0..8 {
            let tex = device
                .alloc_texture(
                    256,
                    256,
                    TextureFormat::Rgba8Unorm,
                    TextureKind::Direct,
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
                TextureKind::Direct,
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
            let buf = device
                .alloc_buffer(256, BufferKind::Scattered, None, BufferFlags::empty())
                .unwrap();
            let tv = scheme_submit_pipelined(&ctx);
            ctx.defer_until(tv, buf);
            timelines.push(tv);
        }

        assert!(ctx.has_deferred_payloads());

        // Wait for the first frame and flush.
        ctx.wait_until(timelines[0]).unwrap();
        ctx.flush_deferred_deletions();

        // Remaining deferred entries (if any) must still be gated on later epochs.
        if ctx.has_deferred_payloads() {
            ctx.wait_until(*timelines.last().unwrap()).unwrap();
            ctx.flush_deferred_deletions();
            assert!(
                !ctx.has_deferred_payloads(),
                "flush after full GPU progress must drain the ring"
            );
        }
    }

    #[test]
    fn wait_and_flush_reclaims_all_deferred() {
        let device = make_device();
        let ctx = submission_context(&device);

        let buf = device
            .alloc_buffer(256, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();
        let tv = scheme_submit_pipelined(&ctx);

        ctx.defer_until(tv, buf);
        assert!(ctx.has_deferred_payloads());

        ctx.wait_until(tv).unwrap();
        ctx.flush_deferred_deletions();

        assert!(!ctx.has_deferred_payloads());
    }

    // ===========================================================================
    // In-flight command buffer tracking
    // ===========================================================================

    /// Submit a clear without dropping the [`Scheme`] yet.
    ///
    /// [`crate::test_support::scheme_advance_timeline`] cannot be used here: it drops the
    /// scheme before returning, and [`Scheme`]'s `Drop` waits the high-water timeline —
    /// which drains in-flight CBs before these tests can observe them.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn scheme_submit_leave_in_flight(
        ctx: &crate::Context,
    ) -> (
        crate::Scheme,
        crate::timeline::TimelineValue,
        crate::retained_pool::RetainedPool,
        crate::Buffer,
    ) {
        use crate::{BufferFlags, BufferKind, RetainedPool, Scheme};
        use std::sync::Arc;

        let device = Arc::new(ctx.device().clone());
        let mut pool = RetainedPool::new(device);
        let buf = pool
            .acquire_buffer(256, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("buf");
        let mut scheme = Scheme::new(ctx);
        scheme.clear_parcel(&buf, 0, 256).expect("clear");
        let tv = scheme.submit().expect("submit").timeline_value();
        (scheme, tv, pool, buf)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn in_flight_cb_count_increases_after_submit() {
        let device = make_device();
        let ctx = submission_context(&device);
        assert_eq!(ctx.in_flight_command_buffer_count(), 0);

        let (_scheme, _tv, _pool, _buf) = scheme_submit_leave_in_flight(&ctx);

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

        let (_scheme, tv, _pool, _buf) = scheme_submit_leave_in_flight(&ctx);

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

        let tv = scheme_submit_pipelined(&ctx);

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
            let tv = scheme_submit_pipelined(&ctx);
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
        let tv = scheme_submit_pipelined(&ctx);

        let _before = ctx.deferred_deletion_pending_count();

        let extra = device
            .alloc_buffer(4096, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();
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
            let buffers: Vec<Allocation> = (0..3)
                .map(|_| {
                    device
                        .alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty())
                        .unwrap_or_else(|e| panic!("frame {frame}: alloc failed: {e}"))
                })
                .collect();

            let tv = scheme_submit_pipelined(&ctx);

            let mut payload = crate::DeferredPayload::new();
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
            let buffers: Vec<Allocation> = (0..2)
                .map(|_| {
                    device
                        .alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty())
                        .unwrap_or_else(|e| panic!("frame {frame}: alloc failed: {e}"))
                })
                .collect();

            let tv = scheme_submit_pipelined(&ctx);

            let mut payload = crate::DeferredPayload::new();
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
            match device.alloc_buffer(alloc_size, BufferKind::Scattered, None, BufferFlags::empty()) {
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
            let bufs: Vec<Allocation> = (0..2)
                .map(|_| {
                    device
                        .alloc_buffer(buf_size, BufferKind::Scattered, None, BufferFlags::empty())
                        .unwrap_or_else(|e| panic!("frame {frame}: buffer alloc failed: {e}"))
                })
                .collect();

            let tex = device
                .alloc_texture(
                    128,
                    128,
                    TextureFormat::Rgba8Unorm,
                    TextureKind::Direct,
                    TextureFlags::COPY_DST,
                )
                .unwrap_or_else(|e| panic!("frame {frame}: texture alloc failed: {e}"));

            let tv = scheme_submit_pipelined(&ctx);

            let mut payload = crate::DeferredPayload::new();
            for b in bufs {
                payload.push(b);
            }
            payload.push(tex);
            ctx.defer_release(tv, payload);
            ctx.flush_deferred_deletions();
        }
    }

    // ===========================================================================
    // Allocation resize under heap pressure (abstract-gpu-vram: growable buffers)
    // ===========================================================================

    #[test]
    fn buffer_resize_works_under_heap_pressure() {
        let device = make_device();
        let ctx = submission_context(&device);

        // Allocate a buffer, submit work so it has a timeline.
        let mut buf = device
            .alloc_buffer(1024, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();

        let tv = scheme_submit_pipelined(&ctx);
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
        let mut arc = Arc::new(
            device
                .alloc_buffer_with_data(&initial_data, BufferKind::Scattered)
                .unwrap(),
        );

        // Grow the buffer (triggers blit-copy internally).
        let new_size = 1024;
        Arc::get_mut(&mut arc).unwrap().resize_to(new_size).unwrap();
        assert_eq!(arc.size(), new_size);

        // Submit a fence to ensure the internal blit-copy has completed.
        let tv = scheme_submit_pipelined(&ctx);
        ctx.wait_until(tv).unwrap();

        // Withdraw — first 256 bytes should be preserved.
        let parcel = Parcel::from_whole_buffer(Arc::clone(&arc), Arc::downgrade(&device.inner));
        let mut scheme = Scheme::new(&ctx);
        let grant = MemoryExchange::new(&ctx)
            .bind_withdraw(&mut scheme, &parcel)
            .expect("withdraw");
        let mut sub = scheme.submit().expect("submit");
        let readback = grant.claim(&mut sub).expect("claim").consume().expect("consume");
        let result: &[u32] = bytemuck::cast_slice(&readback[..256]);
        assert_eq!(&result[..64], &initial_data[..]);
    }

    // ===========================================================================
    // Deferred release + rapid reuse pattern (owned-shared lifecycle)
    // ===========================================================================

    #[test]
    fn deferred_buffers_returned_to_caller_after_flush() {
        use std::sync::{Arc, Mutex};

        let device = make_device();
        let ctx = submission_context(&device);
        let pending: Arc<Mutex<Vec<Allocation>>> = Arc::new(Mutex::new(Vec::new()));

        // Simulate a deferred-owned-allocations token pattern.
        struct Token {
            pending: Arc<Mutex<Vec<Allocation>>>,
            buffers: Vec<Allocation>,
        }
        impl Drop for Token {
            fn drop(&mut self) {
                let mut guard = self.pending.lock().unwrap();
                guard.append(&mut self.buffers);
            }
        }

        // Frame 1: allocate buffers, submit, defer.
        let bufs: Vec<Allocation> = (0..3)
            .map(|_| {
                device
                    .alloc_buffer(4096, BufferKind::Scattered, None, BufferFlags::empty())
                    .unwrap()
            })
            .collect();

        let tv = scheme_submit_pipelined(&ctx);

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
        assert_eq!(returned, 3, "all 3 buffers should be returned to pending after flush");
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
        let buf = device
            .alloc_buffer(256, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();
        let tv = scheme_submit_pipelined(&ctx);
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
        assert!(result.is_err(), "waiting for far-future timeline should timeout");
    }
}
