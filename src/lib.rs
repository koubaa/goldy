//! # Goldy - Modern GPU Library
//!
//! A modern GPU library targeting Vulkan 1.4+, DX12, and Metal.
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
pub mod buffer;
pub mod compute;
pub mod context;
pub mod device;
pub mod error;
pub mod frame_orchestrator;
pub(crate) mod frame_table;
pub(crate) mod handles;
pub mod pipeline;
pub(crate) mod render_target;
pub mod sampler;
pub mod shader;
pub mod shader_library;
pub mod shaders;
pub(crate) mod surface;
pub mod task_graph;
pub mod texture;
pub mod types;

pub mod shader_cache;
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
pub mod scheme;
pub mod signal;
pub mod swapchain_pool;
pub mod timeline;
pub mod transient_pool;
pub(crate) mod vram_allocator;
pub use allocation_policy::BudgetPolicy;
pub use error::GoldyError;
pub use exchange::{Claim, SurfaceExchange};
pub use frame_orchestrator::{FrameHandle, FrameOrchestrator};
pub use parcel::{field, ordinal, Buffer, Init, Parcel, RecordField, Texture};
pub use retained_pool::RetainedPool;
pub use scheme::{
    Grant, GrantBuffer, GrantBytes, GrantTexture, Lease, LeaseRenderTarget, ReadGrant, ReplayStats, Scheme,
    SchemeRenderPassBuilder, Submission, Transaction, UploadBuffer,
};
pub use swapchain_pool::{AcquiredPresent, PresentLease};
pub use task_graph::{ShaderResourceSlot, PRESENT_LEASE_SLOT_PLACEHOLDER};
pub use vram_allocator::DeferredPayload;

// Re-export main types
pub use buffer::StructuredBufferElement;
pub use compute::ComputePipeline;
pub use context::Context;
pub use device::{
    Adapter, AdapterInfo, BufferHeapStats, Device, DeviceCapabilities, DeviceDescriptor, Instance, PowerPreference,
    RequestAdapterOptions, TextureHeapStats, VideoMemoryInfo,
};
pub use goldy_derive::LayoutCheckable;
pub use goldy_derive::StructuredBufferElement;
pub use pipeline::{RenderPipeline, RenderPipelineDesc};
pub use sampler::Sampler;
pub use shader::{builtins, ShaderModule};
pub use shader_library::ShaderLibrary;
pub use signal::{OversubscribedReason, Signal};
pub use slang::{layout_validation_enabled, LayoutCheck, StructFieldLayout, StructLayout};
pub use task_graph::NodeAccess;
pub use texture::TextureCopyFootprint;
pub use timeline::TimelineValue;

pub use handles::{SamplerHandle, TextureHandle};
pub use types::*;
pub use types::{PresentMode, SurfaceConfig};

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

/// Test helpers for `--lib` and integration tests.
///
/// - [`test_support::mock_device`] / [`test_support::with_mock`]: pure software; safe to run in parallel.
/// - [`test_support::SerialGpuDevice`]: real GPU device for unit tests that touch DX12/Vulkan/Metal.
///   On DX12 WARP, holds a process-wide lock for the device lifetime so lib tests do not
///   interleave WARP work (same rationale as `test_threads = 1` in compute_integration).
#[doc(hidden)]
pub mod test_support {
    use crate::backend::mock::MockBackend;
    use crate::device::{Adapter, DeviceDescriptor, Instance, RequestAdapterOptions};
    use crate::{BackendType, Device, DeviceType};
    use std::ops::Deref;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

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

    pub fn mock_recorded_waits(device: &Device) -> Vec<Vec<crate::timeline::Epoch>> {
        with_mock(device, |m| m.recorded_waits.clone())
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
    pub fn scheme_advance_timeline(ctx: &crate::Context) -> crate::TimelineValue {
        use crate::{BufferFlags, BufferKind, RetainedPool, Scheme};
        let device = Arc::new(ctx.device().clone());
        let mut pool = RetainedPool::new(device);
        let buf = pool
            .acquire_buffer(256, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("buf");
        let mut scheme = Scheme::new(ctx);
        scheme.commit_clear_parcel(&buf, 0, 256).expect("clear");
        scheme.submit().expect("submit").timeline_value()
    }

    /// Process-wide gate for DX12 WARP lib tests. MockBackend never takes this lock.
    static WARP_LIB_TEST_SERIAL: OnceLock<Mutex<()>> = OnceLock::new();

    fn warp_lib_test_serial() -> &'static Mutex<()> {
        WARP_LIB_TEST_SERIAL.get_or_init(|| Mutex::new(()))
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

    /// Real GPU [`Device`] for `--lib` tests, with WARP serialization when needed.
    ///
    /// Drop order releases the device before the optional WARP lock so the next
    /// serialized test can create a fresh device safely.
    pub struct SerialGpuDevice {
        device: Device,
        _warp_guard: Option<MutexGuard<'static, ()>>,
    }

    impl Default for SerialGpuDevice {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SerialGpuDevice {
        /// Default adapter (`RequestAdapterOptions::default`, honors `GOLDY_DX12_FORCE_WARP`).
        pub fn new() -> Self {
            Self::from_adapter_factory(|instance| {
                instance
                    .request_adapter(&RequestAdapterOptions::default())
                    .expect("adapter")
            })
        }

        /// Prefer an adapter of `preferred` type, unless `GOLDY_DX12_FORCE_WARP=1` (then WARP).
        pub fn preferring(preferred: DeviceType) -> Self {
            Self::from_adapter_factory(|instance| {
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

        fn from_adapter_factory(select: impl FnOnce(&Instance) -> Adapter) -> Self {
            let instance = Instance::new().expect("Instance::new");
            let adapter = select(&instance);

            // Lock before `request_device` when the selected adapter is WARP — adapter
            // selection is cheap/non-racy; device open is what must be serialized.
            let _warp_guard = if is_dx12_warp_adapter(&instance, &adapter) {
                Some(warp_lib_test_serial().lock().unwrap())
            } else {
                None
            };

            let device = adapter.request_device(&DeviceDescriptor::default()).expect("device");
            drop(instance);

            Self { device, _warp_guard }
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
}
