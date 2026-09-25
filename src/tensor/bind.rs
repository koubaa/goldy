//! Scheme / kernel binding: parent bindless slot + BufferRange envelope.

use super::view::TensorView;
use super::Tensor;
use crate::scheme::SchemeBindable;
use crate::task_graph::ResourceId;
use crate::types::ResourceAccess;

impl SchemeBindable for TensorView<'_> {
    fn resolve(&self, _: &mut crate::scheme::Scheme, access: ResourceAccess) -> crate::scheme::SchemeBindResult {
        let parcel = self.buffer().whole();
        let slot = parcel.resource_index(access);
        let stamp = parcel.stamp_handle();
        (Some((self.envelope_id(), Some(stamp))), slot)
    }

    fn buffer_parcel(&self) -> Option<crate::parcel::Parcel> {
        Some(self.buffer().whole().clone())
    }

    fn resource_identity(&self) -> Option<ResourceId> {
        Some(self.envelope_id())
    }
}

impl TensorView<'_> {
    /// The parent buffer, or the byte range of it this view can touch.
    fn envelope_id(&self) -> ResourceId {
        let parent = self
            .buffer()
            .whole()
            .buffer_handle()
            .expect("tensor view parent is a buffer");
        match self.byte_envelope() {
            Ok((offset, len))
                if len == 0 || (offset == 0 && offset.saturating_add(len) >= self.buffer().byte_size()) =>
            {
                ResourceId::Buffer(parent)
            }
            Ok((offset, len)) => ResourceId::BufferRange { parent, offset, len },
            Err(_) => ResourceId::Buffer(parent),
        }
    }
}

impl SchemeBindable for Tensor {
    fn resolve(&self, scheme: &mut crate::scheme::Scheme, access: ResourceAccess) -> crate::scheme::SchemeBindResult {
        self.view().resolve(scheme, access)
    }

    fn buffer_parcel(&self) -> Option<crate::parcel::Parcel> {
        Some(self.buffer().whole().clone())
    }

    fn resource_identity(&self) -> Option<ResourceId> {
        self.view().resource_identity()
    }
}

impl crate::kernel::KernelBindable for TensorView<'_> {
    fn __goldy_bind_kernel<'a>(
        &self,
        start: crate::kernel::SchemeNodeStart<'a>,
        access: crate::task_graph::NodeAccess,
    ) -> crate::kernel::SchemeNodeStart<'a> {
        start.bind_resource(self, access)
    }

    fn __goldy_kernel_identity(&self) -> crate::kernel::KernelArgIdentity {
        crate::kernel::KernelArgIdentity(self.resource_identity())
    }
}

impl crate::kernel::KernelBindable for Tensor {
    fn __goldy_bind_kernel<'a>(
        &self,
        start: crate::kernel::SchemeNodeStart<'a>,
        access: crate::task_graph::NodeAccess,
    ) -> crate::kernel::SchemeNodeStart<'a> {
        start.bind_resource(&self.view(), access)
    }

    fn __goldy_kernel_identity(&self) -> crate::kernel::KernelArgIdentity {
        crate::kernel::KernelArgIdentity(self.resource_identity())
    }
}
