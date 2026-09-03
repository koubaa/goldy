//! Ray-tracing acceleration structures (BLAS / TLAS) for inline [`RayQuery`].
//!
//! Create with [`AccelerationStructure::blas_triangles`] or [`AccelerationStructure::tlas`],
//! record [`crate::Scheme::build_blas`] / [`crate::Scheme::build_tlas`], then bind with
//! [`crate::scheme::SchemeNodeBuilder::with_parcel`] as an `Accel` shader parameter.

use crate::backend::{GpuAccelCreate, GpuBackend};
use crate::device::Device;
use crate::handles::AccelerationStructureHandle;
use crate::task_graph::ResourceId;
use crate::types::{ResourceAccess, ResourceCategory, ResourceHandle};
use anyhow::Result;
use std::sync::{Arc, Mutex};

struct AccelerationStructureInner {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    handle: AccelerationStructureHandle,
    bindless: Option<u32>,
    /// BLASes this TLAS still needs on the GPU. Empty for BLAS objects.
    held_blases: Mutex<Vec<AccelerationStructure>>,
}

impl Drop for AccelerationStructureInner {
    fn drop(&mut self) {
        // GPU TLAS is destroyed first; `held_blases` drop afterwards (field drop order)
        // so referenced BLASes stay alive until this TLAS is gone.
        if let Ok(mut backend) = self.backend.lock() {
            backend.destroy_acceleration_structure(self.handle);
        }
    }
}

/// Bottom-level (triangle) or top-level (instance) acceleration structure.
///
/// Cloning is cheap (`Arc`). GPU teardown runs on the last drop and is deferred
/// until in-flight command buffers retire (same model as buffers/textures).
/// A TLAS keeps clones of every BLAS passed to [`crate::Scheme::build_tlas`],
/// so dropping the caller's BLAS handle cannot invalidate a live TLAS.
#[derive(Clone)]
pub struct AccelerationStructure {
    inner: Arc<AccelerationStructureInner>,
    pub(crate) handle: AccelerationStructureHandle,
    pub(crate) kind: AccelKind,
}

/// Which GPU object an [`AccelerationStructure`] wraps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelKind {
    /// Triangle BLAS.
    Blas,
    /// Instance TLAS.
    Tlas,
}

/// One TLAS instance recorded by [`crate::Scheme::build_tlas`].
#[derive(Clone, Copy)]
pub struct AccelInstance<'a> {
    /// BLAS referenced by this instance.
    pub blas: &'a AccelerationStructure,
    /// Row-major 3×4 affine transform (same layout as DXR / Vulkan instance desc).
    pub transform: [f32; 12],
    /// 8-bit visibility mask (`0xFF` to hit everything).
    pub mask: u8,
    /// Lower 24 bits are `InstanceCustomIndex`.
    pub custom_index: u32,
}

impl AccelerationStructure {
    /// Allocate an empty triangle BLAS sized for `max_triangles` (indexed or not).
    pub fn blas_triangles(device: &Device, max_triangles: u32, max_vertices: u32, vertex_stride: u32) -> Result<Self> {
        anyhow::ensure!(
            device.capabilities().ray_query || device.capabilities().ray_tracing_pipelines,
            "this adapter does not support acceleration structures \
             (DeviceCapabilities::ray_query and ray_tracing_pipelines are both false). \
             hint: skip AccelerationStructure::blas_triangles / tlas on this device, or pick an \
             adapter with RT (Vulkan VK_KHR_acceleration_structure, DXR, Metal supportsRaytracing). \
             Query device.capabilities().ray_query."
        );
        anyhow::ensure!(
            max_triangles > 0 && max_vertices > 0 && vertex_stride >= 12,
            "invalid BLAS sizing"
        );
        let (handle, bindless) = {
            let mut backend = device.inner.backend.lock().unwrap();
            let handle = backend.create_acceleration_structure(
                device.inner.handle,
                &GpuAccelCreate::BlasTriangles {
                    max_triangles,
                    max_vertices,
                    vertex_stride,
                },
            )?;
            let bindless = backend.accel_bindless_index(handle);
            (handle, bindless)
        };
        Ok(Self {
            inner: Arc::new(AccelerationStructureInner {
                _device: device.clone(),
                backend: Arc::clone(&device.inner.backend),
                handle,
                bindless,
                held_blases: Mutex::new(Vec::new()),
            }),
            handle,
            kind: AccelKind::Blas,
        })
    }

    /// Allocate an empty TLAS that can hold up to `max_instances` BLAS instances.
    pub fn tlas(device: &Device, max_instances: u32) -> Result<Self> {
        anyhow::ensure!(
            device.capabilities().ray_query || device.capabilities().ray_tracing_pipelines,
            "this adapter does not support acceleration structures \
             (DeviceCapabilities::ray_query and ray_tracing_pipelines are both false). \
             hint: skip AccelerationStructure::blas_triangles / tlas on this device, or pick an \
             adapter with RT (Vulkan VK_KHR_acceleration_structure, DXR, Metal supportsRaytracing). \
             Query device.capabilities().ray_query."
        );
        anyhow::ensure!(max_instances > 0, "TLAS max_instances must be > 0");
        let (handle, bindless) = {
            let mut backend = device.inner.backend.lock().unwrap();
            let handle =
                backend.create_acceleration_structure(device.inner.handle, &GpuAccelCreate::Tlas { max_instances })?;
            let bindless = backend.accel_bindless_index(handle);
            (handle, bindless)
        };
        Ok(Self {
            inner: Arc::new(AccelerationStructureInner {
                _device: device.clone(),
                backend: Arc::clone(&device.inner.backend),
                handle,
                bindless,
                held_blases: Mutex::new(Vec::new()),
            }),
            handle,
            kind: AccelKind::Tlas,
        })
    }

    /// Bindless identity for scheme slot binding.
    pub fn handle(&self, _access: ResourceAccess) -> Option<ResourceHandle> {
        self.inner
            .bindless
            .map(|index| ResourceHandle::new(ResourceCategory::Accel, index))
    }

    pub(crate) fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        self.handle(access).map(|h| h.index())
    }

    pub(crate) fn resource_id(&self) -> ResourceId {
        ResourceId::Accel(self.handle)
    }

    /// Keep the BLASes referenced by this TLAS alive for as long as `self` lives.
    pub(crate) fn retain_blases(&self, instances: &[AccelInstance<'_>]) {
        if self.kind != AccelKind::Tlas {
            return;
        }
        let mut held = self.inner.held_blases.lock().unwrap();
        held.clear();
        held.extend(instances.iter().map(|inst| inst.blas.clone()));
    }
}
