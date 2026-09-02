//! # Goldy
//!
//! A GPU library targeting Vulkan 1.4+, DX12, and Metal.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use goldy::{Instance, DeviceType};
//!
//! let instance = Instance::new().unwrap();
//! let adapter = instance.request_adapter(&Default::default()).unwrap();
//! let device = adapter.request_device(&Default::default()).unwrap();
//! ```

pub(crate) mod backend;
pub mod accel;
pub mod buffer;
pub mod compute;
pub mod context;
pub mod device;
pub mod error;
pub mod frame_orchestrator;
pub(crate) mod frame_table;
pub(crate) mod handles;
pub mod kernel;
#[cfg(feature = "graphics")]
pub mod pipeline;
#[cfg(feature = "graphics")]
pub(crate) mod render_target;
pub mod sampler;
pub mod shader;
pub mod shader_library;
pub mod shaders;
#[cfg(feature = "graphics")]
pub(crate) mod surface;
pub mod task_graph;
pub mod texture;
pub mod types;

pub mod cpu_shaders;
pub(crate) mod host_access;
pub mod shader_cache;
pub(crate) mod shader_timing;
pub mod slang;
pub mod validation_env;

// Structured instrumentation for debugging and profiling
pub(crate) mod gpu_profiler;
pub(crate) mod instrumentation;
pub mod tracy;

#[cfg(feature = "tracy")]
#[doc(hidden)]
pub use tracy_client as _tracy_client;
pub(crate) mod allocation_policy;
mod buffer_alloc_tests;
pub mod exchange;
#[cfg(test)]
mod heap_tests;
pub mod parcel;
pub mod retained_pool;
pub mod rt_pipeline;
pub mod scheme;
pub mod signal;
#[cfg(feature = "graphics")]
pub mod swapchain_pool;
pub(crate) mod timeline;
pub mod transient_pool;
pub(crate) mod vram_allocator;
pub use allocation_policy::BudgetPolicy;
pub use error::GoldyError;
#[cfg(feature = "graphics")]
pub use exchange::{Claim, SurfaceExchange};
pub use exchange::{DepositTransaction, MemoryExchange, WithdrawBytes, WithdrawClaim, WithdrawTransaction};
pub use frame_orchestrator::{FrameHandle, FrameOrchestrator};
pub use parcel::{field, ordinal, Buffer, Init, Parcel, RecordField, Texture};
pub use retained_pool::RetainedPool;
pub use scheme::{Lease, ReplayStats, Scheme, Submission};
#[cfg(feature = "graphics")]
pub use scheme::{LeaseRenderTarget, SchemeRenderPassBuilder, Transaction};
pub use shader_timing::{dump_totals, reset_totals};
#[cfg(feature = "graphics")]
pub use swapchain_pool::{AcquiredPresent, PresentLease};
pub use task_graph::ShaderResourceSlot;
#[cfg(feature = "graphics")]
pub use task_graph::PRESENT_LEASE_SLOT_PLACEHOLDER;
pub use vram_allocator::DeferredPayload;

// Re-export main types
pub use accel::{AccelInstance, AccelKind, AccelerationStructure};
pub use rt_pipeline::{RayTracingPipeline, RayTracingPipelineDesc};
pub use buffer::StructuredBufferElement;
pub use compute::ComputePipeline;
pub use context::Context;
pub use cpu_shaders::{CpuBinding, CpuComputeKernel};
pub use device::{
    Adapter, AdapterInfo, BufferHeapStats, Device, DeviceCapabilities, DeviceDescriptor, Instance, PowerPreference,
    RequestAdapterOptions, TextureHeapStats, VideoMemoryInfo,
};
pub use goldy_derive::compute;
pub use goldy_derive::LayoutCheckable;
pub use goldy_derive::StructuredBufferElement;
pub use kernel::gpu;
pub use kernel::{
    prepare_kernel, AccessKind, BuiltinMask, DispatchBuilder, ElementType, KernelBindable, KernelDef, KernelParam,
    KernelSource, ParamCategory, PreparedKernel, RecordedDispatch, ScalarType, SourceMap, KERNEL_ABI_VERSION,
};
#[cfg(feature = "graphics")]
pub use pipeline::{MeshPipeline, MeshPipelineDesc, RenderPipeline, RenderPipelineDesc};
pub use sampler::Sampler;
pub use shader::{builtins, ShaderModule};
pub use shader_library::ShaderLibrary;
pub use signal::{OversubscribedReason, Signal};
pub use slang::{layout_validation_enabled, LayoutCheck, StructFieldLayout, StructLayout};
pub use task_graph::NodeAccess;
pub use texture::TextureCopyFootprint;

pub use handles::{SamplerHandle, TextureHandle};
pub use types::*;
#[cfg(feature = "graphics")]
pub use types::{PresentMode, SurfaceConfig};

#[cfg(feature = "graphics")]
pub use raw_window_handle;

#[cfg(test)]
mod boundary_reclamation;

#[cfg(all(feature = "dx12", target_os = "windows"))]
pub use backend::dx12::WARP_ADAPTER_ID;

/// Whether the DX12 backend is running with the D3D12 debug layer enabled.
#[cfg(all(feature = "dx12", target_os = "windows"))]
pub fn dx12_debug_mode() -> bool {
    backend::dx12::is_debug_mode()
}

/// Whether the DX12 backend is running with the D3D12 debug layer enabled.
#[cfg(not(all(feature = "dx12", target_os = "windows")))]
pub fn dx12_debug_mode() -> bool {
    false
}

/// CUDA retained-path counters exposed for structural integration tests.
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub mod cuda_test_stats {
    pub use crate::backend::cuda::{CudaGraphStats, CudaGraphStatsSnapshot};
}

/// Test helpers for `--lib` and integration tests.
///
/// - [`test_support::mock_device`] / [`test_support::with_mock`]: pure software; safe to run in parallel.
/// - [`test_support::SerialGpuDevice`]: real GPU device for unit tests.
///   - DX12 WARP: process-wide mutex for the device lifetime (WARP is not parallel-safe).
///   - CUDA + `graphics+dx12`: process-wide **shared** device (companion attached once);
///     each test uses its own [`crate::Context`]. An RwLock gate lets shared-device tests
///     run concurrently while exclusive raw-`CudaBackend` / stats tests take a write lock.
#[doc(hidden)]
pub mod test_support {
    use crate::backend::mock::MockBackend;
    use crate::device::{Adapter, DeviceDescriptor, Instance, RequestAdapterOptions};
    use crate::{BackendType, Device, DeviceType};
    use std::ops::Deref;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
    #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
    use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

    pub fn mock_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).expect("mock device"))
    }

    #[allow(private_bounds)]
    pub fn with_mock<R>(device: &Device, f: impl FnOnce(&mut MockBackend) -> R) -> R {
        device.with_mock_backend(f)
    }

    pub fn mock_reset_tracking(device: &Device) {
        with_mock(device, |m| m.reset_tracking());
    }

    pub fn mock_recorded_waits(device: &Device) -> Vec<Vec<(u64, u64)>> {
        with_mock(device, |m| {
            m.recorded_waits
                .iter()
                .map(|batch| batch.iter().map(|e| (e.context, e.value)).collect())
                .collect()
        })
    }

    pub fn mock_retained_resubmit_count(device: &Device) -> usize {
        with_mock(device, |m| m.retained_resubmit_count)
    }

    pub fn mock_compute_dispatch_count(device: &Device) -> usize {
        with_mock(device, |m| m.compute_dispatch_count)
    }

    pub fn mock_all_graph_syncs_some(device: &Device) -> bool {
        with_mock(device, |m| m.recorded_graph_syncs.iter().all(|&s| s))
    }

    pub fn mock_recorded_graph_syncs(device: &Device) -> Vec<bool> {
        with_mock(device, |m| m.recorded_graph_syncs.clone())
    }

    pub fn mock_has_nonempty_host_observed_waits(device: &Device) -> bool {
        with_mock(device, |m| {
            m.recorded_host_observed_waits.iter().any(|batch| !batch.is_empty())
        })
    }

    pub fn mock_has_nonempty_deferred_host_writes(device: &Device) -> bool {
        with_mock(device, |m| {
            m.recorded_deferred_host_writes.iter().any(|batch| !batch.is_empty())
        })
    }

    /// CUDA late-physicalization kind for a buffer parcel (`deferred`/`native`/`shared`/`native_and_twin`).
    #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
    pub fn cuda_buffer_phys_kind(device: &Device, parcel: &crate::Parcel) -> Option<&'static str> {
        let handle = parcel.buffer_handle()?;
        device.cuda_buffer_phys_kind_for_test(handle)
    }

    /// Count buffer entries in the first recorded `ResourceBarrier` on the mock backend.
    pub fn mock_barrier_buffer_count(device: &Device) -> usize {
        use crate::backend::GpuCommand;
        with_mock(device, |m| {
            m.recorded_compute_commands
                .iter()
                .flat_map(|batch| batch.iter())
                .find_map(|cmd| match cmd {
                    GpuCommand::ResourceBarrier { buffers, .. } => Some(buffers.len()),
                    _ => None,
                })
                .unwrap_or(0)
        })
    }

    /// Headless surface exchange backed by mock window handles (mock backend only).
    #[cfg(feature = "graphics")]
    pub fn mock_surface_exchange(device: &Arc<Device>) -> (crate::Context, crate::SurfaceExchange) {
        struct MockWindow;

        impl raw_window_handle::HasWindowHandle for MockWindow {
            fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                        raw_window_handle::WebWindowHandle::new(0),
                    ))
                })
            }
        }

        impl raw_window_handle::HasDisplayHandle for MockWindow {
            fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
                Ok(unsafe {
                    raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                        raw_window_handle::WebDisplayHandle::new(),
                    ))
                })
            }
        }

        let ctx = device.create_context().expect("mock context");
        let surface =
            crate::SurfaceExchange::new_with_depth(&ctx, &MockWindow, 2, crate::types::SurfaceConfig::default())
                .expect("mock surface exchange");
        (ctx, surface)
    }

    /// Advance `ctx`'s timeline with a minimal scheme submit (one clear-parcel node).
    ///
    /// Drops the temporary [`crate::Scheme`] before returning, which waits the
    /// high-water timeline. Do not use this when you need to observe in-flight
    /// command buffers immediately after submit.
    ///
    /// Returns the crate-internal clearing epoch as `u64` for characterization tests.
    pub fn scheme_advance_timeline(ctx: &crate::Context) -> u64 {
        use crate::{BufferFlags, BufferKind, RetainedPool, Scheme};
        let device = Arc::new(ctx.device().clone());
        let mut pool = RetainedPool::new(device);
        let buf = pool
            .acquire_buffer(256, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("buf");
        let mut scheme = Scheme::new(ctx);
        scheme.clear_parcel(&buf, 0, 256).expect("clear");
        scheme.submit().expect("submit").timeline_value()
    }

    /// Crate-internal clearing epoch for a submission (tests only).
    pub fn submission_epoch(submission: &crate::Submission) -> u64 {
        submission.timeline_value()
    }

    /// Latest GPU progress on `ctx` (tests only).
    pub fn gpu_progress(ctx: &crate::Context) -> u64 {
        ctx.gpu_progress()
    }

    /// Block until `ctx` reaches `epoch` (tests only).
    pub fn wait_until(ctx: &crate::Context, epoch: u64) -> Result<(), crate::GoldyError> {
        ctx.wait_until(epoch)
    }

    /// Last-referenced epoch for `parcel` on `ctx` (tests only).
    pub fn parcel_last_epoch_on(parcel: &crate::Parcel, ctx: &crate::Context) -> Option<u64> {
        parcel.last_referenced_on(ctx.backend_handle())
    }

    /// Process-wide gate for DX12 WARP lib tests (device-per-test, exclusive).
    static WARP_LIB_TEST_SERIAL: OnceLock<Mutex<()>> = OnceLock::new();

    fn warp_lib_test_serial() -> &'static Mutex<()> {
        WARP_LIB_TEST_SERIAL.get_or_init(|| Mutex::new(()))
    }

    /// Shared CUDA device for the lib-test process (companion attached once).
    #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
    static SHARED_CUDA_LIB_DEVICE: OnceLock<Option<Arc<Device>>> = OnceLock::new();

    /// Readers = shared-device tests (parallel contexts). Writer = exclusive raw backend / stats.
    #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
    static CUDA_LIB_GATE: OnceLock<RwLock<()>> = OnceLock::new();

    #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
    fn cuda_lib_gate() -> &'static RwLock<()> {
        CUDA_LIB_GATE.get_or_init(|| RwLock::new(()))
    }

    /// Process-wide CUDA lib-test device (one companion attach for the whole run).
    ///
    /// Returns `None` when the active backend is not CUDA or device creation fails.
    #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
    pub fn shared_cuda_lib_device() -> Option<Arc<Device>> {
        SHARED_CUDA_LIB_DEVICE
            .get_or_init(|| {
                let instance = Instance::new().ok()?;
                if instance.backend_type() != BackendType::Cuda {
                    return None;
                }
                let adapter = instance.request_adapter(&RequestAdapterOptions::default()).ok()?;
                let device = adapter.request_device(&DeviceDescriptor::default()).ok()?;
                Some(Arc::new(device))
            })
            .clone()
    }

    /// Shared-device tests: hold for the test lifetime so exclusive backends wait.
    #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
    pub fn cuda_lib_shared_gate() -> RwLockReadGuard<'static, ()> {
        cuda_lib_gate().read().unwrap_or_else(|e| e.into_inner())
    }

    /// Exclusive ownership of the shared CUDA lib-test device.
    ///
    /// Blocks shared-device readers. Use for raw `CudaBackend` / graph-stats tests and
    /// fixtures that assert on the device-global deferred VRAM ring.
    #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
    pub fn cuda_lib_exclusive_gate() -> RwLockWriteGuard<'static, ()> {
        cuda_lib_gate().write().unwrap_or_else(|e| e.into_inner())
    }

    fn env_force_warp() -> bool {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            crate::backend::dx12::env_force_warp()
        }
        #[cfg(not(all(feature = "dx12", target_os = "windows")))]
        {
            false
        }
    }

    fn is_dx12_warp_adapter(instance: &Instance, adapter: &Adapter) -> bool {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            instance.backend_type() == BackendType::Dx12 && adapter.id() == crate::WARP_ADAPTER_ID
        }
        #[cfg(not(all(feature = "dx12", target_os = "windows")))]
        {
            let _ = (instance, adapter);
            false
        }
    }

    #[allow(dead_code)] // held for RAII lock lifetime
    enum SerialGuard {
        None,
        Warp(MutexGuard<'static, ()>),
        #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
        CudaShared(RwLockReadGuard<'static, ()>),
        #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
        CudaExclusive(RwLockWriteGuard<'static, ()>),
    }

    /// Real GPU [`Device`] for `--lib` tests.
    ///
    /// Drop order releases the device borrow before optional guards so the next
    /// fixture can proceed safely.
    pub struct SerialGpuDevice {
        device: Device,
        _guard: SerialGuard,
    }

    impl Default for SerialGpuDevice {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SerialGpuDevice {
        /// Default adapter (`RequestAdapterOptions::default`, honors `GOLDY_DX12_FORCE_WARP`).
        pub fn new() -> Self {
            Self::from_adapter_factory(false, |instance| {
                instance
                    .request_adapter(&RequestAdapterOptions::default())
                    .expect("adapter")
            })
        }

        /// Prefer an adapter of `preferred` type, unless `GOLDY_DX12_FORCE_WARP=1` (then WARP).
        pub fn preferring(preferred: DeviceType) -> Self {
            Self::from_adapter_factory(false, |instance| {
                if env_force_warp() && instance.backend_type() == BackendType::Dx12 {
                    return instance
                        .request_adapter(&RequestAdapterOptions {
                            force_fallback_adapter: true,
                            ..RequestAdapterOptions::default()
                        })
                        .expect("WARP adapter");
                }
                let adapters = instance.enumerate_adapters();
                adapters
                    .iter()
                    .find(|a| a.device_type() == preferred)
                    .or(adapters.first())
                    .cloned()
                    .expect("no adapter")
            })
        }

        /// Sole ownership of the CUDA shared lib-test device (or a private device elsewhere).
        ///
        /// Use when the test asserts on device-global deferred VRAM state. On CUDA this
        /// takes the exclusive gate and drains leftover deferred payloads after idle.
        pub fn exclusive() -> Self {
            Self::from_adapter_factory(true, |instance| {
                instance
                    .request_adapter(&RequestAdapterOptions::default())
                    .expect("adapter")
            })
        }

        fn from_adapter_factory(
            #[allow(unused_variables)] exclusive: bool,
            select: impl FnOnce(&Instance) -> Adapter,
        ) -> Self {
            let instance = Instance::new().expect("Instance::new");
            let adapter = select(&instance);

            #[cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
            if instance.backend_type() == BackendType::Cuda {
                drop(instance);
                let shared = shared_cuda_lib_device().expect("shared CUDA lib-test device");
                let device = (*shared).clone();
                let _guard = if exclusive {
                    let gate = cuda_lib_exclusive_gate();
                    device.wait_idle_and_drain_deferred_for_test();
                    SerialGuard::CudaExclusive(gate)
                } else {
                    SerialGuard::CudaShared(cuda_lib_shared_gate())
                };
                return Self { device, _guard };
            }

            // WARP: lock before `request_device` — adapter selection is cheap/non-racy.
            let _guard = if is_dx12_warp_adapter(&instance, &adapter) {
                SerialGuard::Warp(warp_lib_test_serial().lock().unwrap_or_else(|e| e.into_inner()))
            } else {
                SerialGuard::None
            };

            let device = adapter.request_device(&DeviceDescriptor::default()).expect("device");
            drop(instance);

            Self { device, _guard }
        }
    }

    impl Deref for SerialGpuDevice {
        type Target = Device;

        fn deref(&self) -> &Device {
            &self.device
        }
    }

    /// Thread-local pin for retained command-buffer reuse (see `validation_env`).
    ///
    /// Retention-asserting tests must call [`Self::force_enabled`] so a developer
    /// shell with `GOLDY_DISABLE_CB_REUSE=1` cannot flip the suite. Disable-path
    /// tests call [`Self::force_disabled`]. Cleared on drop.
    pub struct CbReuseOverride {
        _private: (),
    }

    impl CbReuseOverride {
        /// Force CB retention on for this thread (ignores env / profiler).
        pub fn force_enabled() -> Self {
            crate::validation_env::set_cb_reuse_override(false);
            Self { _private: () }
        }

        /// Force CB retention off for this thread (ignores env / profiler).
        pub fn force_disabled() -> Self {
            crate::validation_env::set_cb_reuse_override(true);
            Self { _private: () }
        }
    }

    impl Drop for CbReuseOverride {
        fn drop(&mut self) {
            crate::validation_env::clear_cb_reuse_override();
        }
    }

    /// Thread-local pin for `GOLDY_VALIDATION=host_access`.
    pub struct HostAccessOverride {
        _private: (),
    }

    impl HostAccessOverride {
        pub fn force_enabled() -> Self {
            crate::validation_env::set_host_access_override(true);
            Self { _private: () }
        }
    }

    impl Drop for HostAccessOverride {
        fn drop(&mut self) {
            crate::validation_env::clear_host_access_override();
        }
    }
}
