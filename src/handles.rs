//! Opaque backend resource handles.

/// Opaque handle for a GPU texture object.
pub type TextureHandle = u64;

/// Opaque handle for a GPU sampler object.
pub type SamplerHandle = u64;

/// Opaque handle for a ray-tracing acceleration structure (BLAS or TLAS).
pub type AccelerationStructureHandle = u64;

pub(crate) type DeviceHandle = u64;
pub(crate) type ContextHandle = u64;
pub(crate) type BufferHandle = u64;
pub(crate) type ShaderHandle = u64;
pub(crate) type PipelineHandle = u64;
pub(crate) type ComputePipelineHandle = u64;
pub(crate) type RayTracingPipelineHandle = u64;
pub(crate) type RenderTargetHandle = u64;
#[cfg(feature = "graphics")]
pub(crate) type SurfaceHandle = u64;
#[cfg(feature = "graphics")]
pub(crate) type SwapchainImageHandle = u64;
