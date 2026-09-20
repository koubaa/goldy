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
        let parent = parcel.buffer_handle().expect("tensor view parent is a buffer");
        let resource = match self.byte_envelope() {
            Ok((offset, len))
                if len == 0 || (offset == 0 && offset.saturating_add(len) >= self.buffer().byte_size()) =>
            {
                ResourceId::Buffer(parent)
            }
            Ok((offset, len)) => ResourceId::BufferRange { parent, offset, len },
            Err(_) => ResourceId::Buffer(parent),
        };
        (Some((resource, Some(stamp))), slot)
    }

    fn buffer_parcel(&self) -> Option<crate::parcel::Parcel> {
        Some(self.buffer().whole().clone())
    }
}

impl SchemeBindable for Tensor {
    fn resolve(&self, scheme: &mut crate::scheme::Scheme, access: ResourceAccess) -> crate::scheme::SchemeBindResult {
        self.view().resolve(scheme, access)
    }

    fn buffer_parcel(&self) -> Option<crate::parcel::Parcel> {
        Some(self.buffer().whole().clone())
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
}

impl crate::kernel::KernelBindable for Tensor {
    fn __goldy_bind_kernel<'a>(
        &self,
        start: crate::kernel::SchemeNodeStart<'a>,
        access: crate::task_graph::NodeAccess,
    ) -> crate::kernel::SchemeNodeStart<'a> {
        start.bind_resource(&self.view(), access)
    }
}
