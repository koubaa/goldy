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

pub mod backend;
pub mod buffer;
pub mod common_types;
pub mod compute;
pub mod context;
pub mod device;
pub mod error;
pub mod examples;
pub mod frame_orchestrator;
pub mod frame_table;
pub mod pipeline;
pub mod render_target;
pub mod sampler;
pub mod shader;
pub mod shader_library;
pub mod shaders;
pub mod surface;
pub mod task_graph;
pub mod texture;
pub mod texture_pool;
pub mod types;

pub mod shader_cache;
pub mod slang;
pub mod validation_env;

// Structured instrumentation for debugging and profiling
pub mod gpu_guard;
pub mod gpu_profiler;
pub mod instrumentation;
pub mod tracy;

#[cfg(feature = "tracy")]
#[doc(hidden)]
pub use tracy_client as _tracy_client;
pub mod allocation_policy;
mod buffer_alloc_tests;
#[cfg(test)]
mod heap_tests;
pub mod parcel;
pub mod placement_heap;
pub mod retained_pool;
pub mod scheme;
pub mod signal;
pub mod swapchain_pool;
pub mod timeline;
pub mod transient_allocator;
pub mod transient_pool;
pub mod vram_allocator;
pub use allocation_policy::{AllocCommit, AllocFreeEvent, AllocRequest, AllocationPolicy, BudgetPolicy, NoPolicy};
pub use error::GoldyError;
pub use frame_orchestrator::{FrameHandle, FrameOrchestrator, RetiredFrame};
pub use gpu_guard::GpuGuard;
pub use parcel::{field, ordinal, Buffer, BytesByKind, Init, Parcel, RecordField, Texture};
pub use retained_pool::{RetainedHold, RetainedPool, StampedParcel};
pub use scheme::{
    Grant, GrantBuffer, GrantTexture, IntoDispatch, Lease, LeaseBuffer, LeaseRenderTarget, LeaseTexture, Loan,
    PresentGrant, ReadGrant, ReplayStats, Scheme, SchemeRenderPassBuilder, Submission,
};
pub use swapchain_pool::{PresentLease, SwapchainPool};
pub use task_graph::PRESENT_LEASE_SLOT_PLACEHOLDER;
pub use transient_pool::TransientPool;
pub use vram_allocator::{DeferredPayload, ParcelType};

// Re-export main types
pub use buffer::{BufferPool, BufferSource, BufferView, StructuredBufferElement};
pub use common_types::{FrameUniforms, Instance2D, Particle2D, Particle3D, Transform2D};
pub use compute::ComputePipeline;
pub use signal::{OversubscribedReason, Signal};
pub use timeline::{Epoch, ReferenceTable, TimelineValue};

pub use backend::GraphCommand;
pub use backend::{BufferHeapStats, TextureCopyFootprint, TextureHeapStats};
pub use context::Context;
pub use device::{
    Adapter, Device, DeviceCapabilities, DeviceDescriptor, Instance, PowerPreference, RequestAdapterOptions,
};
pub use goldy_derive::LayoutCheckable;
pub use goldy_derive::StructuredBufferElement;
pub use pipeline::{RenderPipeline, RenderPipelineDesc};
pub use render_target::RenderTarget;
pub use sampler::Sampler;
pub use shader::{builtins, ShaderModule};
pub use shader_library::ShaderLibrary;
pub use slang::{layout_validation_enabled, LayoutCheck, StructFieldLayout, StructLayout};
pub use surface::{Frame, Surface};
pub use task_graph::{
    GraphIR, NodeAccess, NodeBuilder, RenderPassBuilder, ShaderResourceSlot, TaskGraph, TransientId, TransientTextureId,
};
pub use task_graph::{SwapchainOutputHandle, SWAPCHAIN_SLOT_PLACEHOLDER};

pub use texture_pool::{TexturePool, TexturePoolConfig, TexturePoolStats};
pub use transient_allocator::{
    BumpResetAllocator, TransientAllocator, TransientAllocatorConfig, TransientAllocatorStrategy,
};
pub use types::*;
pub use types::{PresentMode, SurfaceConfig};

#[cfg(test)]
mod boundary_reclamation;

#[cfg(all(feature = "dx12", target_os = "windows"))]
pub use backend::dx12::WARP_ADAPTER_ID;

/// Test helpers for `--lib` and integration tests.
///
/// - [`mock_device`] / [`with_mock`]: pure software; safe to run in parallel.
/// - [`SerialGpuDevice`]: real GPU device for unit tests that touch DX12/Vulkan/Metal.
///   On DX12 WARP, holds a process-wide lock for the device lifetime so lib tests do not
///   interleave WARP work (same rationale as `test_threads = 1` in compute_integration).
#[doc(hidden)]
pub mod test_support {
    use crate::backend::mock::MockBackend;
    use crate::device::{DeviceDescriptor, Instance, RequestAdapterOptions};
    use crate::{BackendType, Device, DeviceType};
    use std::ops::Deref;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    pub fn mock_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).expect("mock device"))
    }

    pub fn with_mock<R>(device: &Device, f: impl FnOnce(&mut MockBackend) -> R) -> R {
        device.with_mock_backend(f)
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

    fn is_dx12_warp(device: &Device) -> bool {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            device.backend_type() == BackendType::Dx12 && device.adapter_id() == crate::WARP_ADAPTER_ID
        }
        #[cfg(not(all(feature = "dx12", target_os = "windows")))]
        {
            let _ = device;
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

    impl SerialGpuDevice {
        /// Default adapter (`RequestAdapterOptions::default`, honors `GOLDY_DX12_FORCE_WARP`).
        pub fn new() -> Self {
            Self::from_factory(|instance| {
                instance
                    .request_adapter(&RequestAdapterOptions::default())
                    .expect("adapter")
                    .request_device(&DeviceDescriptor::default())
                    .expect("device")
            })
        }

        /// Prefer an adapter of `preferred` type, unless `GOLDY_DX12_FORCE_WARP=1` (then WARP).
        pub fn preferring(preferred: DeviceType) -> Self {
            Self::from_factory(|instance| {
                if env_force_warp() && instance.backend_type() == BackendType::Dx12 {
                    return instance
                        .request_adapter(&RequestAdapterOptions {
                            force_fallback_adapter: true,
                            ..RequestAdapterOptions::default()
                        })
                        .expect("WARP adapter")
                        .request_device(&DeviceDescriptor::default())
                        .expect("WARP device");
                }
                let adapters = instance.enumerate_adapters();
                let adapter = adapters
                    .iter()
                    .find(|a| a.device_type() == preferred)
                    .or(adapters.first())
                    .expect("no adapter");
                adapter
                    .request_device(&DeviceDescriptor::default())
                    .expect("device")
            })
        }

        fn from_factory(create: impl FnOnce(&Instance) -> Device) -> Self {
            // Lock before device create when FORCE_WARP so two threads cannot both
            // open WARP devices before either observes adapter_id.
            let pre_guard = if env_force_warp() {
                Some(warp_lib_test_serial().lock().unwrap())
            } else {
                None
            };

            let instance = Instance::new().expect("Instance::new");
            let device = create(&instance);
            drop(instance);

            let _warp_guard = match pre_guard {
                Some(g) => Some(g),
                None if is_dx12_warp(&device) => Some(warp_lib_test_serial().lock().unwrap()),
                None => None,
            };

            Self {
                device,
                _warp_guard,
            }
        }
    }

    impl Deref for SerialGpuDevice {
        type Target = Device;

        fn deref(&self) -> &Device {
            &self.device
        }
    }
}
