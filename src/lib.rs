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
pub mod parcel;
pub mod placement_heap;
pub mod retained_pool;
pub mod signal;
pub mod timeline;
pub mod transient_allocator;
pub mod vram_allocator;
pub use allocation_policy::{AllocCommit, AllocFreeEvent, AllocRequest, AllocationPolicy, BudgetPolicy, NoPolicy};
pub use error::GoldyError;
pub use frame_orchestrator::{FrameHandle, FrameOrchestrator, RetiredFrame};
pub use gpu_guard::GpuGuard;
pub use parcel::{BytesByKind, MosaicSlot, Parcel};
pub use retained_pool::{MosaicBuilder, RetainedPool, StampedParcel};
pub use vram_allocator::{DeferredPayload, ParcelType};

// Re-export main types
pub use buffer::{Buffer, BufferPool, BufferSource, BufferView, StructuredBufferElement};
pub use common_types::{FrameUniforms, Instance2D, Particle2D, Particle3D, Transform2D};
pub use compute::{ComputeEncoder, ComputePass, ComputePipeline};
pub use signal::{OversubscribedReason, Signal};
pub use timeline::{Epoch, ReferenceTable, TimelineValue};

pub use backend::GraphCommand;
pub use backend::{BufferHeapStats, TextureHeapStats};
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

pub use texture::Texture;
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
