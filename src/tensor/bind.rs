//! Scheme / kernel binding: parent bindless slot + BufferRange envelope.

use super::view::{TensorBacking, TensorView};
use super::Tensor;
use crate::scheme::SchemeBindable;
use crate::task_graph::ResourceId;
use crate::temporary::TEMPORARY_SLOT_PLACEHOLDER;
use crate::types::ResourceAccess;

impl SchemeBindable for TensorBacking<'_> {
    fn resolve(&self, scheme: &mut crate::scheme::Scheme, access: ResourceAccess) -> crate::scheme::SchemeBindResult {
        match self {
            Self::Buffer(buffer) => buffer.resolve(scheme, access),
            Self::Temporary(temp) => (Some((temp.resource_id(), None)), Some(TEMPORARY_SLOT_PLACEHOLDER)),
        }
    }

    fn buffer_parcel(&self) -> Option<crate::parcel::Parcel> {
        match self {
            Self::Buffer(buffer) => Some(buffer.whole().clone()),
            Self::Temporary(_) => None,
        }
    }

    fn resource_identity(&self) -> Option<ResourceId> {
        match self {
            Self::Buffer(buffer) => Some(buffer.whole().resource_id()),
            Self::Temporary(temp) => Some(temp.resource_id()),
        }
    }

    fn temporary(&self) -> Option<crate::Temporary> {
        match self {
            Self::Buffer(_) => None,
            Self::Temporary(temp) => Some(*temp),
        }
    }
}

impl SchemeBindable for TensorView<'_> {
    fn resolve(&self, _: &mut crate::scheme::Scheme, access: ResourceAccess) -> crate::scheme::SchemeBindResult {
        match self.backing {
            TensorBacking::Buffer(buffer) => {
                let parcel = buffer.whole();
                let slot = parcel.resource_index(access);
                let stamp = parcel.stamp_handle();
                (Some((self.envelope_id(), Some(stamp))), slot)
            }
            TensorBacking::Temporary(temp) => (Some((temp.resource_id(), None)), Some(TEMPORARY_SLOT_PLACEHOLDER)),
        }
    }

    fn buffer_parcel(&self) -> Option<crate::parcel::Parcel> {
        self.try_buffer().map(|b| b.whole().clone())
    }

    fn resource_identity(&self) -> Option<ResourceId> {
        Some(self.envelope_id())
    }

    fn temporary(&self) -> Option<crate::Temporary> {
        match self.backing {
            TensorBacking::Buffer(_) => None,
            TensorBacking::Temporary(temp) => Some(temp),
        }
    }

    fn temporary_byte_range(&self) -> Option<(u64, u64)> {
        let TensorBacking::Temporary(temp) = self.backing else {
            return None;
        };
        match self.byte_envelope() {
            Ok((offset, len)) if len > 0 && !(offset == 0 && offset.saturating_add(len) >= temp.byte_size()) => {
                Some((offset, len))
            }
            _ => None,
        }
    }
}

impl TensorView<'_> {
    /// The parent buffer, or the byte range of it this view can touch.
    fn envelope_id(&self) -> ResourceId {
        match self.backing {
            TensorBacking::Buffer(buffer) => {
                let parent = buffer.whole().buffer_handle().expect("tensor view parent is a buffer");
                match self.byte_envelope() {
                    Ok((offset, len))
                        if len == 0 || (offset == 0 && offset.saturating_add(len) >= buffer.byte_size()) =>
                    {
                        ResourceId::Buffer(parent)
                    }
                    Ok((offset, len)) => ResourceId::BufferRange { parent, offset, len },
                    Err(_) => ResourceId::Buffer(parent),
                }
            }
            TensorBacking::Temporary(temp) => temp.resource_id(),
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
        match self.backing {
            TensorBacking::Buffer(_) => start.bind_resource(self, access),
            TensorBacking::Temporary(ref temp) => start.bind_temporary_range(temp, self.temporary_byte_range(), access),
        }
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
