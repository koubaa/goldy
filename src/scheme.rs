//! Retained scheme — the primary submission unit of the diwan machine.
//!
//! A [`Scheme`] is goldy's realization of the diwan scheme (spec §2): a set of dispatches
//! and precedences, first-class, retained across submissions. Unlike [`crate::TaskGraph`],
//! which is rebuilt each frame, a scheme persists; structural mutation sets a COW dirty bit,
//! and a clean scheme resubmits with zero recording cost.
//!
//! **Construction**: `Scheme::new(&ctx)` — bound to one context for its lifetime.
//! **Submission**: `scheme.submit()` — submits, and submits again, using the retained path
//! when clean.
//!
//! Internally a scheme owns a [`GraphIR`] plus the replay engine (phase-1 item 1.1 of
//! `docu/.../retained-scheme/project.md`). Estate handles (`Deed`, `Lease`, `Easement`) and
//! `goldy::write_to_parcel` are phase-1 items 1.4–1.5, deferred.

use crate::context::Context;
use crate::error::GoldyError;
use crate::task_graph::{DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, TaskNode};
use crate::task_graph::IrSubmitState;
use crate::timeline::TimelineValue;

/// Outcome counters for [`Scheme::submit`] (retention-recovery assertions and telemetry).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayStats {
    /// Submissions served by resubmitting the retained command list (no re-record).
    pub resubmit_hits: u64,
    /// Submissions that recorded (first submit, post-mutation submits, retention misses).
    pub records: u64,
}

/// A retained scheme: a set of dispatches held across submissions with COW dirty tracking.
///
/// Build the scheme's nodes once via [`Self::node`]; call [`Self::submit`] every frame.
/// While clean, `submit` pays neither recording nor fingerprint-hashing cost.
///
/// **Estate API** (`Deed`, `Lease`, `Easement`) is phase-1.4 and not yet present; for now
/// nodes declare parcel access via [`SchemeNodeBuilder::bind_parcel`].
pub struct Scheme {
    ir: GraphIR,
    submit_state: IrSubmitState,
    /// Context this scheme submits on. Fixed at construction; many schemes per context,
    /// exactly one context per scheme.
    ctx: Context,
    /// COW dirty bit: set by every structural mutation, cleared by a successful record.
    dirty: bool,
    /// Retention key stored at record time. `None` when the backend cannot retain `ir`.
    retention_key: Option<u64>,
    /// Timeline value from the most recent successful [`Self::submit`].
    ///
    /// Before resubmitting a retained command list, we [`Context::wait_until`] this value so
    /// the backend CB is no longer pending (Vulkan VUID-vkQueueSubmit2-commandBuffer-03875).
    /// This is conservative: a lowered scheme may become multiple queue submissions (A1, A2,
    /// A3), and another scheme's B1 need only wait for the slice it depends on — not A3. A
    /// per-slice retirement gate belongs in the IR lowering path; until then, whole-scheme
    /// `last_submitted_tv` is the correctness stopgap.
    last_submitted_tv: Option<TimelineValue>,
    stats: ReplayStats,
}

impl Scheme {
    /// Create a scheme bound to `ctx`.
    pub fn new(ctx: &Context) -> Self {
        Self {
            ir: GraphIR::default(),
            submit_state: IrSubmitState::new(),
            ctx: ctx.clone(),
            dirty: true,
            retention_key: None,
            last_submitted_tv: None,
            stats: ReplayStats::default(),
        }
    }

    /// True when the next [`Self::submit`] must re-record.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Submission outcome counters.
    pub fn replay_stats(&self) -> ReplayStats {
        self.stats
    }

    /// Declare a compute dispatch node, returning a builder for access declarations.
    ///
    /// Calling this marks the scheme dirty (structural mutation).
    pub fn node<'a>(
        &'a mut self,
        label: &'static str,
        pipeline: &crate::compute::ComputePipeline,
    ) -> SchemeNodeBuilder<'a> {
        self.dirty = true;
        SchemeNodeBuilder {
            scheme: self,
            label,
            pipeline: pipeline.handle,
            bindings: Vec::new(),
            resource_slots: Vec::new(),
            user_slots: Vec::new(),
        }
    }

    /// Submit the scheme: resubmit the retained command list when clean, re-record when dirty.
    ///
    /// On a clean resubmit, bound parcels' reference tables are stamped with the new
    /// timeline value, keeping the context transient pool's reuse gates correct across
    /// retained submissions.
    pub fn submit(&mut self) -> Result<TimelineValue, GoldyError> {
        if !self.dirty {
            if let Some(key) = self.retention_key {
                if let Some(prev_tv) = self.last_submitted_tv {
                    self.ctx.wait_until(prev_tv)?;
                }
                if let Some(tv) = self.ctx.try_resubmit_retained(key)? {
                    self.submit_state.apply_reference_stamps(
                        self.ctx.backend_handle(),
                        &self.ctx.device().inner,
                        tv,
                    );
                    self.last_submitted_tv = Some(tv);
                    self.stats.resubmit_hits += 1;
                    return Ok(tv);
                }
            }
        }

        let tv = self
            .submit_state
            .submit_pipelined_and_retain(&self.ctx, &self.ir)
            .map_err(|e| self.ctx.classify(e))?;
        self.submit_state.apply_reference_stamps(
            self.ctx.backend_handle(),
            &self.ctx.device().inner,
            tv,
        );
        self.ctx.advance_high_water_timeline(tv);

        self.retention_key = if IrSubmitState::ir_can_retain(&self.ir) {
            Some(IrSubmitState::retention_fingerprint(&self.ir))
        } else {
            None
        };
        self.dirty = false;
        self.last_submitted_tv = Some(tv);
        self.stats.records += 1;
        Ok(tv)
    }
}

/// Builder for a single compute dispatch node within a [`Scheme`].
pub struct SchemeNodeBuilder<'a> {
    scheme: &'a mut Scheme,
    label: &'static str,
    pipeline: crate::backend::ComputePipelineHandle,
    bindings: Vec<ResourceBinding>,
    resource_slots: Vec<u32>,
    user_slots: Vec<u32>,
}

impl<'a> SchemeNodeBuilder<'a> {
    /// Declare that this node accesses a retained [`crate::Parcel`].
    pub fn bind_parcel(mut self, parcel: &crate::Parcel, access: NodeAccess) -> Self {
        self.scheme.submit_state.register_parcel_stamp(parcel);
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        self
    }

    /// Bind resource slots from typed [`crate::types::ResourceHandle`]s (region A indices only).
    pub fn bind_resources_typed(mut self, handles: &[crate::types::ResourceHandle]) -> Self {
        self.resource_slots = handles.iter().map(|h| h.index()).collect();
        self
    }

    /// Finalize the node with fixed workgroup dimensions.
    pub fn dispatch(self, x: u32, y: u32, z: u32) {
        self.scheme.ir.nodes.push(TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::Dispatch {
                pipeline: self.pipeline,
                resource_slots: self.resource_slots,
                user_slots: self.user_slots,
                dispatch: DispatchDim::Direct { x, y, z },
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::compute::ComputePipeline;
    use crate::device::Device;
    use crate::parcel::Parcel;
    use crate::retained_pool::RetainedPool;
    use crate::shader::ShaderModule;
    use crate::task_graph::NodeAccess;
    use std::sync::Arc;

    fn mock_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).expect("mock device"))
    }

    fn mock_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(
            device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1,1,1)]
void cs_main(Scattered<uint> buf, ThreadId id) { buf[0] = 1; }
"#,
        )
        .expect("compile shader")
    }

    fn mock_pipeline(device: &Device, shader: &ShaderModule) -> ComputePipeline {
        ComputePipeline::new(device, shader).expect("create pipeline")
    }

    fn retained_buffer_parcel(pool: &mut RetainedPool) -> Parcel {
        pool.acquire_buffer(
            32,
            crate::types::BufferKind::Scattered,
            None,
            crate::types::BufferFlags::empty(),
            None,
        )
        .expect("alloc buffer parcel")
    }

    fn clean_scheme(device: &Arc<Device>, pool: &mut RetainedPool) -> Scheme {
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(device);
        let pipeline = mock_pipeline(device, &shader);
        let parcel = retained_buffer_parcel(pool);

        let mut scheme = Scheme::new(&ctx);
        assert!(scheme.is_dirty(), "new scheme starts dirty");
        scheme.node("a", &pipeline).bind_parcel(&parcel, NodeAccess::Write).dispatch(1, 1, 1);

        scheme.submit().unwrap();
        assert!(!scheme.is_dirty(), "successful submit clears the dirty bit");
        assert_eq!(scheme.replay_stats().records, 1);
        assert_eq!(scheme.replay_stats().resubmit_hits, 0);
        scheme
    }

    #[test]
    fn clean_submits_resubmit_without_rerecord() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let mut scheme = clean_scheme(&device, &mut pool);

        scheme.submit().unwrap();
        scheme.submit().unwrap();

        assert_eq!(scheme.replay_stats().records, 1, "only the first submit records");
        assert_eq!(scheme.replay_stats().resubmit_hits, 2, "subsequent clean submits resubmit");
    }

    #[test]
    fn mutation_marks_dirty_and_rerecords_once() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let mut scheme = clean_scheme(&device, &mut pool);
        scheme.submit().unwrap();

        assert_eq!(
            scheme.replay_stats(),
            ReplayStats { records: 1, resubmit_hits: 1 }
        );

        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let parcel2 = retained_buffer_parcel(&mut pool);
        scheme.node("b", &pipeline).bind_parcel(&parcel2, NodeAccess::Write).dispatch(1, 1, 1);

        assert!(scheme.is_dirty());
        scheme.submit().unwrap();
        scheme.submit().unwrap();

        assert_eq!(
            scheme.replay_stats(),
            ReplayStats { records: 2, resubmit_hits: 2 }
        );
    }

    #[test]
    fn submit_stamps_parcel_references() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let parcel = retained_buffer_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        scheme.node("a", &pipeline).bind_parcel(&parcel, NodeAccess::Write).dispatch(1, 1, 1);

        let tv1 = scheme.submit().unwrap();
        assert_eq!(parcel.last_referenced_on(ctx.backend_handle()), Some(tv1));

        let tv2 = scheme.submit().unwrap();
        assert!(tv2 >= tv1, "timeline must be monotonic");
        assert_eq!(
            parcel.last_referenced_on(ctx.backend_handle()),
            Some(tv2),
            "resubmit path must also stamp parcel references"
        );
    }
}
