//! Ray-tracing acceleration structures (BLAS / TLAS) for inline [`RayQuery`].
//!
//! Create with [`AccelerationStructure::blas_triangles`] or [`AccelerationStructure::tlas`],
//! record [`crate::Scheme::build_blas`] / [`crate::Scheme::build_tlas`], then bind with
//! [`crate::scheme::SchemeNodeBuilder::with_parcel`] as an `Accel` shader parameter.

use crate::backend::{GpuBackend, GpuAccelCreate};
use crate::device::Device;
use crate::handles::AccelerationStructureHandle;
use crate::task_graph::ResourceId;
use crate::types::{ResourceAccess, ResourceCategory, ResourceHandle};
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Bottom-level (triangle) or top-level (instance) acceleration structure.
pub struct AccelerationStructure {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: AccelerationStructureHandle,
    bindless: Option<u32>,
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
            device.capabilities().ray_query,
            "this adapter does not support inline ray query (DeviceCapabilities::ray_query)"
        );
        anyhow::ensure!(max_triangles > 0 && max_vertices > 0 && vertex_stride >= 12, "invalid BLAS sizing");
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
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            bindless,
            kind: AccelKind::Blas,
        })
    }

    /// Allocate an empty TLAS that can hold up to `max_instances` BLAS instances.
    pub fn tlas(device: &Device, max_instances: u32) -> Result<Self> {
        anyhow::ensure!(
            device.capabilities().ray_query,
            "this adapter does not support inline ray query (DeviceCapabilities::ray_query)"
        );
        anyhow::ensure!(max_instances > 0, "TLAS max_instances must be > 0");
        let (handle, bindless) = {
            let mut backend = device.inner.backend.lock().unwrap();
            let handle = backend.create_acceleration_structure(
                device.inner.handle,
                &GpuAccelCreate::Tlas { max_instances },
            )?;
            let bindless = backend.accel_bindless_index(handle);
            (handle, bindless)
        };
        Ok(Self {
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            bindless,
            kind: AccelKind::Tlas,
        })
    }

    /// Bindless identity for scheme slot binding.
    pub fn handle(&self, _access: ResourceAccess) -> Option<ResourceHandle> {
        self.bindless
            .map(|index| ResourceHandle::new(ResourceCategory::Accel, index))
    }

    pub(crate) fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        self.handle(access).map(|h| h.index())
    }

    pub(crate) fn resource_id(&self) -> ResourceId {
        ResourceId::Accel(self.handle)
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        if let Ok(mut backend) = self.backend.lock() {
            backend.destroy_acceleration_structure(self.handle);
        }
    }
}
