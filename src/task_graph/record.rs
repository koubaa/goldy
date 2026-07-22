//! Owned render-pass recorder for language bindings (no lifetime coupling to [`crate::Scheme`]).
//!
//! [`RenderPassRecord`] mirrors scheme render-pass recording but can be held across
//! FFI calls while commands accumulate, then committed with [`RenderPassRecord::commit_scheme`].

use super::ir::{DispatchDim, NodeAccess, ResourceBinding};
use super::ResourceId;
use crate::backend::RenderCommand;
use crate::buffer::{Allocation, BufferSource, BufferView};
use crate::compute::ComputePipeline;
use crate::parcel::ParcelStamp;
use crate::pipeline::RenderPipeline;
use crate::types::{IndexFormat, ResourceAccess, ResourceHandle};
use std::sync::Arc;

/// Deferred push-constant slot for [`RenderPassRecord`].
///
/// Both the read (SRV) and read-write (UAV) handles are captured up front;
/// the correct one is selected at [`RenderPassRecord::set_pipeline`] time by
/// consulting the pipeline's reflected slot kinds.
struct PendingRenderSlot {
    graph_access: NodeAccess,
    read_handle: Option<ResourceHandle>,
    uav_handle: Option<ResourceHandle>,
}

impl PendingRenderSlot {
    fn from_parcel(parcel: &crate::Parcel, access: NodeAccess) -> Self {
        Self {
            graph_access: access,
            read_handle: parcel.handle(ResourceAccess::Read),
            uav_handle: parcel
                .handle(ResourceAccess::ReadWrite)
                .or_else(|| parcel.handle(ResourceAccess::Write)),
        }
    }

    fn resolve(&self, slot_access: &[Option<ResourceAccess>], slot_idx: usize) -> ResourceHandle {
        let descriptor_access = slot_access
            .get(slot_idx)
            .copied()
            .flatten()
            .unwrap_or(match self.graph_access {
                NodeAccess::Read => ResourceAccess::Read,
                NodeAccess::Write | NodeAccess::Overwrite | NodeAccess::ReadWrite => ResourceAccess::ReadWrite,
            });
        match descriptor_access {
            ResourceAccess::Read => self.read_handle.or(self.uav_handle),
            ResourceAccess::Write | ResourceAccess::ReadWrite => self.uav_handle.or(self.read_handle),
        }
        .unwrap_or_else(|| {
            panic!(
                "render-pass shader slot {slot_idx}: no descriptor for {descriptor_access:?}; \
                 check BufferKind/TextureKind is compatible with the shader parameter"
            )
        })
    }
}

/// Accumulates one offscreen render-pass node before [`Self::commit_scheme`].
pub struct RenderPassRecord {
    label: &'static str,
    target: crate::backend::RenderTargetHandle,
    color_load: crate::types::TargetLoad,
    bindings: Vec<ResourceBinding>,
    commands: Vec<RenderCommand>,
    stamp_targets: Vec<Arc<ParcelStamp>>,
    /// Shader-resource slots waiting for pipeline reflection at [`Self::set_pipeline`].
    pending_push_constants: Vec<PendingRenderSlot>,
}

impl RenderPassRecord {
    pub fn with_parcel(&mut self, parcel: &crate::Parcel, access: NodeAccess) -> &mut Self {
        self.stamp_targets.push(parcel.stamp_handle());
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        self.pending_push_constants
            .push(PendingRenderSlot::from_parcel(parcel, access));
        self
    }

    /// Register dependency on all parcels of a buffer without emitting a descriptor.
    pub fn with_buffer_dependency(&mut self, buffer: &crate::Buffer, access: NodeAccess) -> &mut Self {
        for parcel in buffer.parcels() {
            self.stamp_targets.push(parcel.stamp_handle());
            self.bindings.push(ResourceBinding {
                resource: parcel.resource_id(),
                access,
            });
        }
        self
    }

    pub fn with_buffer(&mut self, parcel: &crate::Parcel, access: NodeAccess) -> &mut Self {
        self.stamp_targets.push(parcel.stamp_handle());
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        self
    }

    pub fn with_buffer_view(&mut self, view: &BufferView, access: NodeAccess) -> &mut Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::BufferRange {
                parent: view.parent_handle(),
                offset: view.offset(),
                len: view.size(),
            },
            access,
        });
        self
    }

    pub fn clear_depth(&mut self, depth: f32) -> &mut Self {
        self.commands.push(RenderCommand::ClearDepth(depth));
        self
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) -> &mut Self {
        self.commands.push(RenderCommand::SetPipeline(pipeline.handle));
        if !self.pending_push_constants.is_empty() {
            let handles: Vec<ResourceHandle> = self
                .pending_push_constants
                .iter()
                .enumerate()
                .map(|(i, slot)| slot.resolve(&pipeline.slot_access, i))
                .collect();
            self.commands.push(RenderCommand::BindResourcesTyped { handles });
        }
        self
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &impl BufferSource) -> &mut Self {
        self.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.source_handle(),
            offset: buffer.source_offset(),
        });
        self
    }

    pub fn set_vertex_buffer_offset(&mut self, slot: u32, buffer: &impl BufferSource, offset: u64) -> &mut Self {
        self.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.source_handle(),
            offset,
        });
        self
    }

    pub fn set_index_buffer(&mut self, buffer: &impl BufferSource, format: IndexFormat) -> &mut Self {
        self.commands.push(RenderCommand::SetIndexBuffer {
            buffer: buffer.source_handle(),
            offset: buffer.source_offset(),
            format,
        });
        self
    }

    pub fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) -> &mut Self {
        self.commands.push(RenderCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
        self
    }

    pub fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) -> &mut Self {
        self.commands.push(RenderCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
        self
    }

    pub fn draw_fullscreen(&mut self) -> &mut Self {
        self.draw(0, 3, 0, 1)
    }

    pub fn draw_quads(&mut self, count: u32) -> &mut Self {
        self.draw(0, 6, 0, count)
    }

    #[allow(dead_code)] // legacy bind-resources render path
    pub(crate) fn with_resources(&mut self, buffers: &[&Allocation]) -> &mut Self {
        self.commands.push(RenderCommand::BindResources {
            buffers: buffers.iter().map(|b| b.handle).collect(),
        });
        self
    }

    pub fn with_resource_slots(&mut self, indices: &[u32]) -> &mut Self {
        self.commands.push(RenderCommand::BindResourcesRaw {
            indices: indices.to_vec(),
            user: Vec::new(),
            frame_table_base: 0,
        });
        self
    }

    /// Begin accumulating a render pass targeting a scheme-held render-target lease.
    pub fn new_for_scheme_lease(
        label: &'static str,
        scheme: &crate::Scheme,
        lease: &crate::Lease<crate::LeaseRenderTarget>,
        color_load: crate::types::TargetLoad,
    ) -> Self {
        let handle = scheme.rt(lease).backend_handle();
        let access = if color_load.overwrites() {
            NodeAccess::Overwrite
        } else {
            NodeAccess::Write
        };
        Self {
            label,
            target: handle,
            color_load,
            bindings: vec![ResourceBinding {
                resource: ResourceId::RenderTarget(handle),
                access,
            }],
            commands: Vec::new(),
            stamp_targets: Vec::new(),
            pending_push_constants: Vec::new(),
        }
    }

    /// Commit this render pass into a retained [`crate::Scheme`].
    pub fn commit_scheme(self, scheme: &mut crate::Scheme) {
        scheme.commit_render_pass(
            self.label,
            self.target,
            self.color_load,
            self.bindings,
            self.commands,
            &self.stamp_targets,
        );
    }
}

fn node_access_to_resource_access(access: NodeAccess) -> ResourceAccess {
    match access {
        NodeAccess::Read => ResourceAccess::Read,
        NodeAccess::Write | NodeAccess::Overwrite => ResourceAccess::Write,
        NodeAccess::ReadWrite => ResourceAccess::ReadWrite,
    }
}

/// Accumulates one compute dispatch node before [`Self::commit_dispatch_scheme`].
pub struct ComputeNodeRecord {
    label: &'static str,
    pipeline: crate::backend::ComputePipelineHandle,
    bindings: Vec<ResourceBinding>,
    resource_slots: Vec<u32>,
    user_slots: Vec<u32>,
    stamp_targets: Vec<Arc<ParcelStamp>>,
    /// Per-slot descriptor access from pipeline reflection (shader-signature order).
    slot_access: Vec<Option<ResourceAccess>>,
}

impl ComputeNodeRecord {
    pub fn new(label: &'static str, pipeline: &ComputePipeline) -> Self {
        Self {
            label,
            pipeline: pipeline.handle,
            bindings: Vec::new(),
            resource_slots: Vec::new(),
            user_slots: Vec::new(),
            stamp_targets: Vec::new(),
            slot_access: pipeline.slot_access.clone(),
        }
    }

    /// Declare graph dependency on a parcel without resolving a shader slot.
    pub fn with_parcel_access(&mut self, parcel: &crate::Parcel, access: NodeAccess) -> &mut Self {
        self.register_parcel_binding(parcel, access)
    }

    pub fn with_buffer(&mut self, parcel: &crate::Parcel, access: NodeAccess) -> &mut Self {
        self.register_parcel_binding(parcel, access)
    }

    pub fn with_buffer_view(&mut self, view: &BufferView, access: NodeAccess) -> &mut Self {
        self.register_buffer_view_binding(view, access)
    }

    fn register_buffer_view_binding(&mut self, view: &BufferView, access: NodeAccess) -> &mut Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::BufferRange {
                parent: view.parent_handle(),
                offset: view.offset(),
                len: view.size(),
            },
            access,
        });
        self
    }

    pub fn with_resource_slots(&mut self, indices: &[u32]) -> &mut Self {
        self.resource_slots = indices.to_vec();
        self
    }

    /// Declare a retained parcel as a graph dependency and shader binding slot atomically.
    ///
    /// This is the preferred entry-point for language-binding consumers. Descriptor access
    /// (SRV vs UAV) is inferred from pipeline reflection captured at construction, matching
    /// [`crate::scheme::SchemeNodeBuilder::with_parcel`]. Returns `None` when no bindless slot
    /// exists for the inferred descriptor access.
    pub fn with_parcel(&mut self, parcel: &crate::Parcel, node_access: NodeAccess) -> Option<&mut Self> {
        let slot_idx = self.resource_slots.len();
        let descriptor_access = self
            .slot_access
            .get(slot_idx)
            .copied()
            .flatten()
            .unwrap_or_else(|| node_access_to_resource_access(node_access));
        let idx = parcel.resource_index(descriptor_access)?;
        self.register_parcel_binding(parcel, node_access);
        self.resource_slots.push(idx);
        Some(self)
    }

    /// Register dependency on all parcels of a buffer without emitting shader slots.
    pub fn with_buffer_dependency(&mut self, buffer: &crate::Buffer, access: NodeAccess) -> &mut Self {
        for parcel in buffer.parcels() {
            self.register_parcel_binding(parcel, access);
        }
        self
    }

    /// Append one scalar virtual-main parameter (region B).
    pub fn with_param(&mut self, value: u32) -> &mut Self {
        use crate::backend::shared::MAX_USER_SLOTS;
        assert!(
            self.user_slots.len() < MAX_USER_SLOTS,
            "with_param: at most {MAX_USER_SLOTS} scalar params per dispatch"
        );
        self.user_slots.push(value);
        self
    }

    pub(crate) fn register_parcel_binding(&mut self, parcel: &crate::Parcel, access: NodeAccess) -> &mut Self {
        self.stamp_targets.push(parcel.stamp_handle());
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        self
    }

    /// Commit this compute node into a retained [`crate::Scheme`].
    pub fn commit_dispatch_scheme(self, scheme: &mut crate::Scheme, x: u32, y: u32, z: u32) {
        scheme.apply_compute_stamps(&self.stamp_targets);
        scheme.commit_compute_dispatch(
            self.label,
            self.pipeline,
            self.bindings,
            self.resource_slots,
            self.user_slots,
            DispatchDim::Direct { x, y, z },
        );
    }
}
