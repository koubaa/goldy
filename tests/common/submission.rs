//! Submission/timeline context for integration tests (`gpu_progress` / `wait_until`).

use goldy::{types::BackendType, Context, Device};

pub fn submission_context(device: &Device) -> Context {
    device.create_context().expect("context")
}

/// Clamp libtest parallelism so concurrent trials cannot exhaust Vulkan's fixed
/// per-device compute-queue pool (shared `Device` across trials).
///
/// Several integration tests hold two live [`Context`]s at once (`two_contexts_*`).
/// Worst-case concurrent demand is `2 * test_threads`, so cap threads at `pool / 2`.
/// DX12 WARP stays forced to a single thread (known contention).
pub fn clamp_test_threads(args: &mut libtest_mimic::Arguments, device: &Device) {
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    if device.backend_type() == BackendType::Dx12 && device.adapter_id() == goldy::WARP_ADAPTER_ID {
        args.test_threads = Some(1);
        return;
    }

    if device.backend_type() == BackendType::Vulkan {
        let pool = device.max_submission_contexts().max(1) as usize;
        // Each trial may hold up to two contexts; parallel trials share one device pool.
        let cap = (pool / 2).max(1);
        args.test_threads = Some(match args.test_threads {
            Some(n) => n.min(cap),
            None => cap,
        });
    }
}
