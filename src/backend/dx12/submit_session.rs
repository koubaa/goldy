//! Lock-free per-context submit session (Phase 5b-iv).
//!
//! Cloned at [`crate::Context::new`] under a brief global lock; recording + queue
//! submit never touch the global backend mutex.

use super::compute;
use super::frame_table::ContextFrameTable;
use super::types::{
    Dx12State, SharedBufferTable, SharedComputePipelineTable, SharedContextFrameTable, SharedContextMap,
    SharedLogicalDevice, SharedPipelineTable, SharedRenderTargetTable, SharedSamplerTable, SharedShaderTable,
    SharedSubmissionContext, SharedTextureTable,
};
use super::{ContextHandle, DeviceHandle, GpuCommand, GraphCommand, SubmitSync};
use crate::timeline::TimelineValue;
use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use windows::Win32::Graphics::Direct3D12::ID3D12Fence;

/// Resource tables + device handles used by compute and render command recording.
pub(crate) struct Dx12RecordState<'a> {
    pub ld: &'a SharedLogicalDevice,
    pub devices: &'a HashMap<DeviceHandle, SharedLogicalDevice>,
    pub contexts: &'a SharedContextMap,
    pub frame_table: SharedContextFrameTable,
    pub buffers: &'a SharedBufferTable,
    pub shaders: &'a SharedShaderTable,
    pub pipelines: &'a SharedPipelineTable,
    pub compute_pipelines: &'a SharedComputePipelineTable,
    pub render_targets: &'a SharedRenderTargetTable,
    pub textures: &'a SharedTextureTable,
    #[allow(dead_code)]
    pub samplers: &'a SharedSamplerTable,
}

/// Cloned handles for one partition submit — no global backend lock required.
pub(crate) struct Dx12SubmitScope<'a> {
    /// Submitting context identity (fence lookups use [`Self::ctx_fence`]).
    #[allow(dead_code)]
    pub ctx: ContextHandle,
    pub device_handle: DeviceHandle,
    pub sc: super::types::SharedSubmissionContext,
    pub record: Dx12RecordState<'a>,
    pub context_fences: &'a Arc<RwLock<HashMap<ContextHandle, super::types::ContextFenceEntry>>>,
    /// Per-context fence resolved once at scope construction (avoids repeated map lookups).
    pub ctx_fence: ID3D12Fence,
    /// Synthetic context for device-queue epoch stamps (compute style only).
    pub device_owner: Option<ContextHandle>,
}

impl<'a> Dx12SubmitScope<'a> {
    pub fn ld(&self) -> &'a SharedLogicalDevice {
        self.record.ld
    }

    pub fn devices(&self) -> &'a HashMap<DeviceHandle, SharedLogicalDevice> {
        self.record.devices
    }

    pub fn contexts(&self) -> &'a SharedContextMap {
        self.record.contexts
    }

    pub fn frame_table(&self) -> &ContextFrameTable {
        &self.record.frame_table
    }

    pub fn buffers(&self) -> &'a SharedBufferTable {
        self.record.buffers
    }

    #[allow(dead_code)]
    pub fn shaders(&self) -> &'a SharedShaderTable {
        self.record.shaders
    }

    #[allow(dead_code)]
    pub fn pipelines(&self) -> &'a SharedPipelineTable {
        self.record.pipelines
    }

    pub fn compute_pipelines(&self) -> &'a SharedComputePipelineTable {
        self.record.compute_pipelines
    }

    pub fn render_targets(&self) -> &'a SharedRenderTargetTable {
        self.record.render_targets
    }

    pub fn textures(&self) -> &'a SharedTextureTable {
        self.record.textures
    }

    #[allow(dead_code)]
    pub fn samplers(&self) -> &'a SharedSamplerTable {
        self.record.samplers
    }
}

pub(crate) fn record_state_from_backend<'a>(
    state: &'a Dx12State,
    ctx: ContextHandle,
    device_handle: DeviceHandle,
) -> Result<Dx12RecordState<'a>> {
    let frame_table = Arc::clone(
        &state
            .contexts
            .read()
            .unwrap()
            .get(&ctx)
            .with_context(|| format!("Invalid context handle {ctx}"))?
            .lock()
            .unwrap()
            .frame_table,
    );
    Ok(Dx12RecordState {
        ld: state
            .devices
            .get(&device_handle)
            .with_context(|| format!("Invalid device {device_handle}"))?,
        devices: &state.devices,
        contexts: &state.contexts,
        frame_table,
        buffers: &state.buffers,
        shaders: &state.shaders,
        pipelines: &state.pipelines,
        compute_pipelines: &state.compute_pipelines,
        render_targets: &state.render_targets,
        textures: &state.textures,
        samplers: &state.samplers,
    })
}

pub(crate) fn record_state_for_legacy_render<'a>(
    state: &'a mut Dx12State,
    device_handle: DeviceHandle,
) -> Result<Dx12RecordState<'a>> {
    let frame_table = super::frame_table::ensure_legacy_frame_table(state, device_handle)?;
    Ok(Dx12RecordState {
        ld: state
            .devices
            .get(&device_handle)
            .with_context(|| format!("Invalid device {device_handle}"))?,
        devices: &state.devices,
        contexts: &state.contexts,
        frame_table,
        buffers: &state.buffers,
        shaders: &state.shaders,
        pipelines: &state.pipelines,
        compute_pipelines: &state.compute_pipelines,
        render_targets: &state.render_targets,
        textures: &state.textures,
        samplers: &state.samplers,
    })
}

/// Per-context submit session cloned at context creation.
pub(crate) struct Dx12SubmitSession {
    ctx: ContextHandle,
    device_handle: DeviceHandle,
    sc: SharedSubmissionContext,
    ld: SharedLogicalDevice,
    devices: Arc<HashMap<DeviceHandle, SharedLogicalDevice>>,
    contexts: SharedContextMap,
    frame_table: SharedContextFrameTable,
    context_fences: Arc<RwLock<HashMap<ContextHandle, super::types::ContextFenceEntry>>>,
    buffers: SharedBufferTable,
    shaders: SharedShaderTable,
    pipelines: SharedPipelineTable,
    compute_pipelines: SharedComputePipelineTable,
    render_targets: SharedRenderTargetTable,
    textures: SharedTextureTable,
    samplers: SharedSamplerTable,
    device_owner_handle: Option<ContextHandle>,
    ctx_fence: ID3D12Fence,
}

impl Dx12SubmitSession {
    pub fn clone_from_state(state: &Dx12State, ctx: ContextHandle) -> Result<Arc<Self>> {
        let sc = Arc::clone(
            state
                .contexts
                .read()
                .unwrap()
                .get(&ctx)
                .with_context(|| format!("Invalid context handle {ctx}"))?,
        );
        let (device_handle, frame_table) = {
            let sc_guard = sc.lock().unwrap();
            (sc_guard.device, Arc::clone(&sc_guard.frame_table))
        };
        let ld = Arc::clone(
            state
                .devices
                .get(&device_handle)
                .with_context(|| format!("Invalid device handle {device_handle}"))?,
        );
        let devices: HashMap<DeviceHandle, SharedLogicalDevice> = state
            .devices
            .iter()
            .map(|(handle, device)| (*handle, Arc::clone(device)))
            .collect();
        let device_owner_handle = state.device_owner_handles.get(&device_handle).copied();
        let ctx_fence = state
            .context_fences
            .read()
            .unwrap()
            .get(&ctx)
            .with_context(|| format!("Invalid context handle {ctx}"))?
            .1
            .clone();
        Ok(Arc::new(Self {
            ctx,
            device_handle,
            sc,
            ld,
            devices: Arc::new(devices),
            contexts: Arc::clone(&state.contexts),
            frame_table,
            context_fences: Arc::clone(&state.context_fences),
            buffers: Arc::clone(&state.buffers),
            shaders: Arc::clone(&state.shaders),
            pipelines: Arc::clone(&state.pipelines),
            compute_pipelines: Arc::clone(&state.compute_pipelines),
            render_targets: Arc::clone(&state.render_targets),
            textures: Arc::clone(&state.textures),
            samplers: Arc::clone(&state.samplers),
            device_owner_handle,
            ctx_fence,
        }))
    }

    fn scope(&self) -> Dx12SubmitScope<'_> {
        Dx12SubmitScope {
            ctx: self.ctx,
            device_handle: self.device_handle,
            sc: std::sync::Arc::clone(&self.sc),
            record: Dx12RecordState {
                ld: &self.ld,
                devices: &self.devices,
                contexts: &self.contexts,
                frame_table: Arc::clone(&self.frame_table),
                buffers: &self.buffers,
                shaders: &self.shaders,
                pipelines: &self.pipelines,
                compute_pipelines: &self.compute_pipelines,
                render_targets: &self.render_targets,
                textures: &self.textures,
                samplers: &self.samplers,
            },
            context_fences: &self.context_fences,
            ctx_fence: self.ctx_fence.clone(),
            device_owner: self.device_owner_handle,
        }
    }
}

impl crate::backend::ContextSubmitSession for Dx12SubmitSession {
    fn separate_graphics_queue(&self) -> bool {
        true
    }

    fn device_queue_owner(&self, _ctx: ContextHandle) -> Option<ContextHandle> {
        self.device_owner_handle
    }

    fn retains_present_partitions(&self) -> bool {
        true
    }

    fn submit_standalone(
        &self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<TimelineValue> {
        compute::submit_with_scope(&self.scope(), ctx, commands, sync)
    }

    fn submit_graph(
        &self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<TimelineValue> {
        compute::submit_graph_with_scope(&self.scope(), ctx, commands, None, sync)
    }

    fn submit_graph_and_retain(
        &self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<TimelineValue> {
        let scope = self.scope();
        compute::evict_retained_with_scope(&scope, ctx, key);
        compute::submit_graph_with_scope(&scope, ctx, commands, Some(key), sync)
    }

    fn try_resubmit_retained(
        &self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<TimelineValue>> {
        compute::try_resubmit_retained_with_scope(&self.scope(), ctx, key, sync)
    }

    fn evict_retained(&self, ctx: ContextHandle, key: u64) {
        compute::evict_retained_with_scope(&self.scope(), ctx, key);
    }
}
