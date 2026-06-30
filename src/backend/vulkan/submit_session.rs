//! Lock-free per-context submit session (Phase 5b-iv).
//!
//! Cloned at [`crate::Context::new`] under a brief global lock; recording + queue
//! submit never touch the global backend mutex.

use super::compute;
use super::types::{
    SharedBufferTable, SharedComputeFencePool, SharedComputePipelineTable, SharedContextMap, SharedFrameTableDevice,
    SharedFrameTableMap, SharedLogicalDevice, SharedPipelineTable, SharedRenderTargetTable, SharedSubmissionContext,
    SharedTextureTable, VulkanState,
};
use super::{ContextHandle, DeviceHandle, GpuCommand, GraphCommand, SubmitSync};
use crate::timeline::TimelineValue;
use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// Fields from [`VulkanState`] needed by compute/render command recording and queue submit.
pub(crate) struct VulkanSubmitView<'a> {
    pub instance: &'a ash::Instance,
    pub devices: &'a HashMap<DeviceHandle, SharedLogicalDevice>,
    pub contexts: &'a SharedContextMap,
    pub buffers: &'a SharedBufferTable,
    pub pipelines: &'a SharedPipelineTable,
    pub compute_pipelines: &'a SharedComputePipelineTable,
    pub render_targets: &'a SharedRenderTargetTable,
    pub textures: &'a SharedTextureTable,
    pub compute_fence_pool: &'a SharedComputeFencePool,
    pub frame_tables: &'a SharedFrameTableMap,
}

impl VulkanState {
    pub(crate) fn submit_view(&self) -> VulkanSubmitView<'_> {
        VulkanSubmitView {
            instance: &self.instance,
            devices: &self.devices,
            contexts: &self.contexts,
            buffers: &self.buffers,
            pipelines: &self.pipelines,
            compute_pipelines: &self.compute_pipelines,
            render_targets: &self.render_targets,
            textures: &self.textures,
            compute_fence_pool: &self.compute_fence_pool,
            frame_tables: &self.frame_tables,
        }
    }
}

/// Cloned handles for one partition submit — no global backend lock required.
pub(crate) struct VulkanSubmitScope<'a> {
    pub ctx: ContextHandle,
    pub device_handle: DeviceHandle,
    pub sc: SharedSubmissionContext,
    pub view: VulkanSubmitView<'a>,
    pub frame_table: SharedFrameTableDevice,
}

impl<'a> VulkanSubmitScope<'a> {
    /// Session methods pass a redundant `ctx`; must match the scope's bound context.
    pub(crate) fn assert_ctx(&self, ctx: ContextHandle) {
        debug_assert_eq!(self.ctx, ctx, "ContextSubmitSession invoked with wrong context handle");
    }

    pub(crate) fn completed_timeline_value(&self) -> u64 {
        let sem = self.sc.lock().unwrap().timeline_semaphore;
        let Some(ld) = self.view.devices.get(&self.device_handle) else {
            return 0;
        };
        unsafe { ld.device.get_semaphore_counter_value(sem).unwrap_or(0) }
    }
}

/// Per-context submit session cloned at context creation.
pub(crate) struct VulkanSubmitSession {
    instance: ash::Instance,
    ctx: ContextHandle,
    device_handle: DeviceHandle,
    sc: SharedSubmissionContext,
    devices: Arc<HashMap<DeviceHandle, SharedLogicalDevice>>,
    contexts: SharedContextMap,
    frame_table: SharedFrameTableDevice,
    buffers: SharedBufferTable,
    pipelines: SharedPipelineTable,
    compute_pipelines: SharedComputePipelineTable,
    render_targets: SharedRenderTargetTable,
    textures: SharedTextureTable,
    compute_fence_pool: SharedComputeFencePool,
    frame_tables: SharedFrameTableMap,
}

impl VulkanSubmitSession {
    pub fn clone_from_state(state: &VulkanState, ctx: ContextHandle) -> Result<Arc<Self>> {
        let sc = Arc::clone(
            state
                .contexts
                .read()
                .unwrap()
                .get(&ctx)
                .with_context(|| format!("Invalid context handle {ctx}"))?,
        );
        let device_handle = sc.lock().unwrap().device;
        let frame_table = Arc::clone(
            state
                .frame_tables
                .read()
                .unwrap()
                .get(&device_handle)
                .with_context(|| format!("frame table not initialized for device {device_handle}"))?,
        );
        let devices: HashMap<DeviceHandle, SharedLogicalDevice> = state
            .devices
            .iter()
            .map(|(handle, device)| (*handle, Arc::clone(device)))
            .collect();
        Ok(Arc::new(Self {
            instance: state.instance.clone(),
            ctx,
            device_handle,
            sc,
            devices: Arc::new(devices),
            contexts: Arc::clone(&state.contexts),
            frame_table,
            buffers: Arc::clone(&state.buffers),
            pipelines: Arc::clone(&state.pipelines),
            compute_pipelines: Arc::clone(&state.compute_pipelines),
            render_targets: Arc::clone(&state.render_targets),
            textures: Arc::clone(&state.textures),
            compute_fence_pool: Arc::clone(&state.compute_fence_pool),
            frame_tables: Arc::clone(&state.frame_tables),
        }))
    }

    fn scope(&self) -> VulkanSubmitScope<'_> {
        VulkanSubmitScope {
            ctx: self.ctx,
            device_handle: self.device_handle,
            sc: Arc::clone(&self.sc),
            view: VulkanSubmitView {
                instance: &self.instance,
                devices: &self.devices,
                contexts: &self.contexts,
                buffers: &self.buffers,
                pipelines: &self.pipelines,
                compute_pipelines: &self.compute_pipelines,
                render_targets: &self.render_targets,
                textures: &self.textures,
                compute_fence_pool: &self.compute_fence_pool,
                frame_tables: &self.frame_tables,
            },
            frame_table: Arc::clone(&self.frame_table),
        }
    }

    fn dispatch_scope(&self, ctx: ContextHandle) -> VulkanSubmitScope<'_> {
        debug_assert_eq!(ctx, self.ctx, "ContextSubmitSession invoked with wrong context handle");
        self.scope()
    }
}

pub(crate) fn scope_from_state(state: &VulkanState, ctx: ContextHandle) -> Result<VulkanSubmitScope<'_>> {
    let sc = Arc::clone(
        state
            .contexts
            .read()
            .unwrap()
            .get(&ctx)
            .with_context(|| format!("Invalid context handle {ctx}"))?,
    );
    let device_handle = sc.lock().unwrap().device;
    let frame_table = Arc::clone(
        state
            .frame_tables
            .read()
            .unwrap()
            .get(&device_handle)
            .with_context(|| format!("frame table not initialized for device {device_handle}"))?,
    );
    Ok(VulkanSubmitScope {
        ctx,
        device_handle,
        sc,
        view: state.submit_view(),
        frame_table,
    })
}

impl crate::backend::ContextSubmitSession for VulkanSubmitSession {
    fn retains_present_partitions(&self) -> bool {
        true
    }

    fn submit_standalone(
        &self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<TimelineValue> {
        let scope = self.dispatch_scope(ctx);
        compute::submit_with_scope(&scope, scope.ctx, commands, sync)
    }

    fn submit_graph(
        &self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<TimelineValue> {
        let scope = self.dispatch_scope(ctx);
        compute::submit_graph_with_scope(&scope, scope.ctx, commands, None, sync)
    }

    fn submit_graph_and_retain(
        &self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<TimelineValue> {
        let scope = self.dispatch_scope(ctx);
        compute::evict_retained_with_scope(&scope, scope.ctx, key);
        compute::submit_graph_with_scope(&scope, scope.ctx, commands, Some(key), sync)
    }

    fn try_resubmit_retained(
        &self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<TimelineValue>> {
        let scope = self.dispatch_scope(ctx);
        compute::try_resubmit_retained_with_scope(&scope, scope.ctx, key, sync)
    }

    fn evict_retained(&self, ctx: ContextHandle, key: u64) {
        let scope = self.dispatch_scope(ctx);
        compute::evict_retained_with_scope(&scope, scope.ctx, key);
    }
}
