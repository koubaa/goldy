//! Compute pipeline and dispatch logic.

use super::barriers;
use super::shader;
use super::staging;
use super::types::{self, ComputeAllocatorSlot, ComputePipelineState, Dx12State};
use super::{ComputePipelineHandle, DeviceHandle, ShaderHandle};
use crate::backend::{FenceToken, GpuCommand};
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// Drain any pending debug-layer messages for this device into a single
/// human-readable string. Returns `None` when the device has no
/// `ID3D12InfoQueue` (debug layer disabled) or no messages are queued.
///
/// Useful in error paths like a `Close()` failure, where the actual cause
/// is written to the info queue before the HRESULT bubbles up. Without
/// this drain the debug-layer text goes to `OutputDebugString` and is
/// invisible to anyone not attached with a debugger.
fn drain_info_queue(device: &ID3D12Device10) -> Option<String> {
    let info_queue: ID3D12InfoQueue = device.cast().ok()?;
    let count = unsafe { info_queue.GetNumStoredMessages() };
    if count == 0 {
        return None;
    }
    let mut out = String::new();
    for i in 0..count {
        let mut len: usize = 0;
        unsafe {
            if info_queue.GetMessage(i, None, &mut len).is_err() {
                continue;
            }
        }
        let mut buf = vec![0u8; len];
        let msg_ptr = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
        unsafe {
            if info_queue.GetMessage(i, Some(msg_ptr), &mut len).is_err() {
                continue;
            }
            let msg = &*msg_ptr;
            let desc = std::slice::from_raw_parts(
                msg.pDescription,
                msg.DescriptionByteLength.saturating_sub(1),
            );
            let text = std::str::from_utf8(desc).unwrap_or("<non-utf8 description>");
            let severity = match msg.Severity {
                D3D12_MESSAGE_SEVERITY_CORRUPTION => "CORRUPTION",
                D3D12_MESSAGE_SEVERITY_ERROR => "ERROR",
                D3D12_MESSAGE_SEVERITY_WARNING => "WARNING",
                D3D12_MESSAGE_SEVERITY_INFO => "INFO",
                D3D12_MESSAGE_SEVERITY_MESSAGE => "MSG",
                _ => "?",
            };
            out.push_str(&format!(
                "  [D3D12 {}] id={} {}\n",
                severity, msg.ID.0, text
            ));
        }
    }
    unsafe { info_queue.ClearStoredMessages() };
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Create a compute pipeline.
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    compute_shader: ShaderHandle,
) -> Result<ComputePipelineHandle> {
    // Compile shader on-demand
    let cs_bytecode =
        shader::ensure_stage_compiled(state, compute_shader, crate::slang::SlangStage::Compute)?;

    let shader_debug_name = format!("compute_shader#{compute_shader}");

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Use the shared bindless root signature from the device
    let root_signature = logical_device
        .bindless_root_signature
        .as_ref()
        .context("Bindless root signature not available")?
        .clone();

    tracing::debug!("Using shared bindless root signature for compute pipeline");

    // Create compute PSO
    let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs_bytecode.as_ptr() as *const _,
            BytecodeLength: cs_bytecode.len(),
        },
        NodeMask: 0,
        CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
        Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
    };

    let pipeline_state: ID3D12PipelineState =
        unsafe { logical_device.device.CreateComputePipelineState(&pso_desc) }
            .context("Failed to create compute pipeline state")?;

    let handle = state.next_compute_pipeline_handle;
    state.next_compute_pipeline_handle += 1;

    state.compute_pipelines.insert(
        handle,
        ComputePipelineState {
            device_handle,
            pipeline_state,
            root_signature,
            parameter_block_layouts: Vec::new(),
            push_constant_categories: Vec::new(),
            shader_debug_name,
        },
    );

    tracing::debug!("Created compute pipeline {}", handle);
    Ok(handle)
}

/// Destroy a compute pipeline.
pub(super) fn destroy(state: &mut Dx12State, pipeline_handle: ComputePipelineHandle) {
    state.compute_pipelines.remove(&pipeline_handle);
}

/// Submit compute commands without blocking. Returns a fence token for polling/waiting.
pub(super) fn submit(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    commands: &[GpuCommand],
) -> Result<FenceToken> {
    let (allocator, fence_value, slot_idx) = {
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        // Find a free slot (allocator whose work has completed) or create a new one
        let fence = &logical_device.fence;
        let completed = unsafe { fence.GetCompletedValue() };
        let pool = &mut logical_device.compute_allocator_pool;
        let slot_idx = pool.iter().position(|s| completed >= s.fence_value);
        let (allocator, slot_idx) = if let Some(idx) = slot_idx {
            let slot = &mut pool[idx];
            unsafe { slot.allocator.Reset() }.context("Failed to reset command allocator")?;
            (slot.allocator.clone(), idx)
        } else {
            let new_allocator: ID3D12CommandAllocator = unsafe {
                logical_device
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .context("Failed to create command allocator")?;
            pool.push(ComputeAllocatorSlot {
                allocator: new_allocator.clone(),
                fence_value: 0,
            });
            unsafe { new_allocator.Reset() }.context("Failed to reset new command allocator")?;
            (new_allocator, pool.len() - 1)
        };
        let token = logical_device.fence_value;
        logical_device.fence_value += 1;
        (allocator, token, slot_idx)
    };

    let fence_clone = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .fence
        .clone();
    state
        .staging_belts
        .entry(device_handle)
        .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
        .reclaim(&fence_clone)?;

    let mut belt_slices: Vec<(ID3D12Resource, u64)> = Vec::new();

    // Create command list (allocator is owned, no borrow of state)
    let command_list: ID3D12GraphicsCommandList = {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        unsafe {
            logical_device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )
        }
        .context("Failed to create command list")?
    };
    let command_list7: ID3D12GraphicsCommandList7 = command_list
        .cast()
        .context("ID3D12GraphicsCommandList7 required")?;

    // Pre-pass: memcpy into staging belt chunks for DEVICE_LOCAL WriteBuffer uploads.
    for command in commands {
        if let GpuCommand::WriteBuffer {
            buffer: buf_handle,
            data,
            ..
        } = command
        {
            let buf = state
                .buffers
                .get(buf_handle)
                .context("WriteBuffer pre-pass: invalid handle")?;
            if buf.is_storage {
                let buf_dev = buf.device_handle;
                let ld = state
                    .devices
                    .get(&buf_dev)
                    .context("WriteBuffer pre-pass: device missing")?;
                let belt_entry = state.staging_belts.entry(buf_dev).or_insert_with(|| {
                    staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE)
                });
                let (res, off) = belt_entry.write(ld, data.as_slice())?;
                belt_slices.push((res, off));
            }
        }
    }

    let mut staged_texture_uploads: Vec<super::texture::StagedTextureUpload> = Vec::new();
    for command in commands {
        match command {
            GpuCommand::WriteTexture {
                texture,
                data,
                width,
                height,
            } => {
                staged_texture_uploads.push(super::texture::stage_texture_upload_full(
                    state,
                    *texture,
                    data.as_slice(),
                    *width,
                    *height,
                )?);
            }
            GpuCommand::WriteTextureRegion {
                texture,
                x,
                y,
                width,
                height,
                data,
            } => {
                staged_texture_uploads.push(super::texture::stage_texture_upload_region(
                    state,
                    *texture,
                    *x,
                    *y,
                    *width,
                    *height,
                    data.as_slice(),
                )?);
            }
            _ => {}
        }
    }

    {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        unsafe {
            command_list.SetDescriptorHeaps(&[
                Some(logical_device.cbv_srv_uav_heap.clone()),
                Some(logical_device.sampler_heap.clone()),
            ]);
        }
    }

    let mut belt_idx = 0usize;
    let mut texture_upload_idx = 0usize;
    let mut current_compute_pipeline: Option<super::ComputePipelineHandle> = None;

    // Process commands
    for command in commands {
        match command {
            GpuCommand::SetPipeline(handle) => {
                current_compute_pipeline = Some(*handle);
                if let Some(pipeline_state) = state.compute_pipelines.get(handle) {
                    unsafe {
                        command_list.SetComputeRootSignature(&pipeline_state.root_signature);
                        command_list.SetPipelineState(&pipeline_state.pipeline_state);
                    }
                }
            }
            GpuCommand::BindResources { buffers } => {
                let mut layout = types::PushLayout::default();
                for (i, buffer_handle) in buffers.iter().enumerate() {
                    if i >= types::MAX_BINDLESS_SLOTS {
                        break;
                    }
                    if let Some(buf_state) = state.buffers.get(buffer_handle) {
                        let offset = buf_state.bindless_offset.unwrap_or(0);
                        layout.bindless[i] = offset as u16;
                        tracing::trace!(
                            "Compute resource slot [{}]: buffer {} -> UAV offset {}",
                            i,
                            buffer_handle,
                            offset
                        );
                    }
                }
                tracing::trace!(
                    "Setting compute root constants (bindless): {:?}",
                    &layout.bindless[..buffers.len().min(types::MAX_BINDLESS_SLOTS)]
                );
                unsafe {
                    command_list.SetComputeRoot32BitConstants(
                        0,
                        (types::TOTAL_PUSH_BYTES / 4) as u32,
                        &layout as *const _ as *const std::ffi::c_void,
                        0,
                    );
                }
            }
            GpuCommand::BindResourcesRaw {
                indices: raw_indices,
                user: raw_user,
            } => {
                let mut layout = types::PushLayout::default();
                for (i, &idx) in raw_indices.iter().enumerate() {
                    if i >= types::MAX_BINDLESS_SLOTS {
                        break;
                    }
                    layout.bindless[i] = idx as u16;
                }
                for (i, &val) in raw_user.iter().enumerate() {
                    if i >= types::MAX_USER_SLOTS {
                        break;
                    }
                    layout.user[i] = val;
                }
                unsafe {
                    command_list.SetComputeRoot32BitConstants(
                        0,
                        (types::TOTAL_PUSH_BYTES / 4) as u32,
                        &layout as *const _ as *const std::ffi::c_void,
                        0,
                    );
                }
            }
            GpuCommand::BindResourcesTyped {
                handles: typed_handles,
            } => {
                if let Some(pipeline) = current_compute_pipeline.and_then(|h| state.compute_pipelines.get(&h)) {
                    crate::backend::validate_typed_push_constants(
                        typed_handles,
                        &pipeline.push_constant_categories,
                        &pipeline.shader_debug_name,
                    )?;
                }
                let mut layout = types::PushLayout::default();
                for (i, handle) in typed_handles.iter().enumerate() {
                    if i >= types::MAX_BINDLESS_SLOTS {
                        break;
                    }
                    layout.bindless[i] = handle.index() as u16;
                }
                unsafe {
                    command_list.SetComputeRoot32BitConstants(
                        0,
                        (types::TOTAL_PUSH_BYTES / 4) as u32,
                        &layout as *const _ as *const std::ffi::c_void,
                        0,
                    );
                }
            }
            GpuCommand::Dispatch {
                workgroups_x,
                workgroups_y,
                workgroups_z,
            } => {
                // No per-dispatch barrier: the graph scheduler emits ResourceBarrier
                // commands at wave boundaries where cross-wave data dependencies
                // exist. Dispatches within the same wave are independent and can
                // overlap on the GPU.
                unsafe {
                    command_list.Dispatch(*workgroups_x, *workgroups_y, *workgroups_z);
                }
            }
            GpuCommand::DispatchIndirect { buffer, offset } => {
                let logical_device = state
                    .devices
                    .get(&device_handle)
                    .context("DispatchIndirect: invalid device")?;
                let buf_state = state
                    .buffers
                    .get(buffer)
                    .context("DispatchIndirect: invalid buffer handle")?;
                let signature = logical_device
                    .compute_dispatch_indirect_signature
                    .as_ref()
                    .context("DispatchIndirect: compute indirect signature not available")?;

                // Transition the indirect argument buffer from UAV to
                // INDIRECT_ARGUMENT for ExecuteIndirect. This is an access-mode
                // transition, not a data-dependency barrier (the graph's
                // ResourceBarrier handles data visibility).
                let mut to_indirect = [barriers::buffer_barrier_full(
                    &buf_state.resource,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                )];
                unsafe { barriers::barrier_buffers(&command_list7, &to_indirect) };
                unsafe { barriers::drop_buffer_barriers(&mut to_indirect) };

                unsafe {
                    command_list.ExecuteIndirect(
                        signature,
                        1,
                        &buf_state.resource,
                        *offset,
                        None,
                        0,
                    );
                }

                let mut to_uav = [barriers::buffer_barrier_full(
                    &buf_state.resource,
                    D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                )];
                unsafe { barriers::barrier_buffers(&command_list7, &to_uav) };
                unsafe { barriers::drop_buffer_barriers(&mut to_uav) };
            }
            GpuCommand::Barrier => {
                let g = D3D12_GLOBAL_BARRIER {
                    SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    AccessAfter: D3D12_BARRIER_ACCESS(
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0
                            | D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0,
                    ),
                };
                unsafe { barriers::barrier_globals(&command_list7, &[g]) };
            }
            GpuCommand::ResourceBarrier {
                buffers: buf_handles,
                textures: tex_handles,
            } => {
                // Per-resource enhanced barriers at graph wave boundaries.
                let mut buf_barriers: Vec<D3D12_BUFFER_BARRIER> = buf_handles
                    .iter()
                    .filter_map(|h| state.buffers.get(h))
                    .map(|bs| {
                        barriers::buffer_barrier_full(
                            &bs.resource,
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                            D3D12_BARRIER_ACCESS(
                                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0
                                    | D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0,
                            ),
                        )
                    })
                    .collect();
                unsafe { barriers::barrier_buffers(&command_list7, &buf_barriers) };
                unsafe { barriers::drop_buffer_barriers(&mut buf_barriers) };

                let mut tex_barriers: Vec<D3D12_TEXTURE_BARRIER> = tex_handles
                    .iter()
                    .filter_map(|h| state.textures.get(h))
                    .map(|ts| {
                        // Enhanced barriers require Access and Layout to agree:
                        // `AccessAfter` bits including SHADER_RESOURCE are
                        // incompatible with `LayoutAfter = UNORDERED_ACCESS`
                        // (and vice versa), so we can't use the buffer-style
                        // "UAV | SRV" conservative after-access for textures.
                        // Since a Goldy texture is materialized once as
                        // either Direct (UAV) or Interpolated (SRV) and that
                        // category is fixed for its lifetime, the graph-
                        // emitted barrier between two compute nodes that
                        // both touch the same texture just needs to
                        // synchronise within that single access mode.
                        let (access, layout) = if ts.last_layout
                            == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS
                            || ts.last_layout == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS
                        {
                            (
                                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                                D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
                            )
                        } else {
                            (
                                D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                                D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE,
                            )
                        };
                        barriers::texture_barrier_full(
                            &ts.resource,
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            access,
                            access,
                            layout,
                            layout,
                        )
                    })
                    .collect();
                unsafe { barriers::barrier_textures(&command_list7, &tex_barriers) };
                unsafe { barriers::drop_texture_barriers(&mut tex_barriers) };
            }
            GpuCommand::ClearBuffer {
                buffer,
                offset,
                size,
            } => {
                let buf_state = state
                    .buffers
                    .get(buffer)
                    .context("ClearBuffer: invalid buffer handle")?;
                let clear_size = if *size == 0 {
                    buf_state.size.saturating_sub(*offset)
                } else {
                    *size
                };
                if clear_size > 0 {
                    if buf_state.is_storage {
                        let logical_device = state
                            .devices
                            .get(&device_handle)
                            .context("ClearBuffer: invalid device")?;
                        // Storage buffer (DEFAULT heap): GPU-side UAV clear (no staging needed)
                        super::buffer::uav_clear(
                            logical_device,
                            buf_state,
                            &command_list,
                            *offset,
                            clear_size,
                        )?;
                    } else {
                        // UPLOAD heap buffer: CPU-accessible, just memset
                        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
                        let no_read = D3D12_RANGE { Begin: 0, End: 0 };
                        unsafe { buf_state.resource.Map(0, Some(&no_read), Some(&mut mapped)) }
                            .context("ClearBuffer: failed to map buffer")?;
                        unsafe {
                            std::ptr::write_bytes(
                                (mapped as *mut u8).add(*offset as usize),
                                0,
                                clear_size as usize,
                            );
                        }
                        let written = D3D12_RANGE {
                            Begin: *offset as usize,
                            End: (*offset + clear_size) as usize,
                        };
                        unsafe { buf_state.resource.Unmap(0, Some(&written)) };
                    }
                }
            }
            GpuCommand::WriteBuffer {
                buffer: buf_handle,
                offset,
                data,
            } => {
                let buf_state = state
                    .buffers
                    .get(buf_handle)
                    .context("WriteBuffer: invalid buffer handle")?;
                if !buf_state.is_storage {
                    // UPLOAD heap: direct map (same as existing write path)
                    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
                    let no_read = D3D12_RANGE { Begin: 0, End: 0 };
                    unsafe { buf_state.resource.Map(0, Some(&no_read), Some(&mut mapped)) }
                        .context("WriteBuffer: map failed")?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            (mapped as *mut u8).add(*offset as usize),
                            data.len(),
                        );
                    }
                    let written_range = D3D12_RANGE {
                        Begin: *offset as usize,
                        End: (*offset as usize) + data.len(),
                    };
                    unsafe { buf_state.resource.Unmap(0, Some(&written_range)) };
                } else {
                    // DEFAULT heap: copy from staging belt slice (prepended in pre-pass).
                    let belt_entry = belt_slices
                        .get(belt_idx)
                        .context("WriteBuffer: belt slice missing (internal)")?;
                    belt_idx += 1;
                    let upload_src = belt_entry.0.clone();
                    let upload_off = belt_entry.1;
                    let buf_state = state.buffers.get(buf_handle).unwrap();

                    let mut b_to_copy = [barriers::buffer_barrier_full(
                        &buf_state.resource,
                        D3D12_BARRIER_SYNC_ALL,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_COPY_DEST,
                    )];
                    let mut b_to_uav = [barriers::buffer_barrier_full(
                        &buf_state.resource,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_COPY_DEST,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    )];
                    unsafe {
                        barriers::barrier_buffers(&command_list7, &b_to_copy);
                        barriers::drop_buffer_barriers(&mut b_to_copy);
                        command_list.CopyBufferRegion(
                            &buf_state.resource,
                            *offset,
                            &upload_src,
                            upload_off,
                            data.len() as u64,
                        );
                        barriers::barrier_buffers(&command_list7, &b_to_uav);
                        barriers::drop_buffer_barriers(&mut b_to_uav);
                    }
                }
            }
            GpuCommand::WriteTexture { .. } | GpuCommand::WriteTextureRegion { .. } => {
                let upload = staged_texture_uploads
                    .get(texture_upload_idx)
                    .context("WriteTexture: staged upload missing (internal)")?;
                texture_upload_idx += 1;
                super::texture::record_staged_texture_upload(
                    &command_list,
                    &command_list7,
                    state,
                    upload,
                )?;
            }
        }
    }

    debug_assert_eq!(
        texture_upload_idx,
        staged_texture_uploads.len(),
        "WriteTexture command count mismatch vs staging pre-pass"
    );

    // Global barrier so compute UAV writes AND copy writes are visible to
    // subsequent graphics / CPU.  WriteBuffer records CopyBufferRegion on this
    // command list, so we must include COPY in the tail sync.
    let tail = D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC(
            D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0,
        ),
        SyncAfter: D3D12_BARRIER_SYNC_ALL,
        AccessBefore: D3D12_BARRIER_ACCESS(
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_COPY_DEST.0,
        ),
        AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
    };
    unsafe { barriers::barrier_globals(&command_list7, &[tail]) };

    // Close and execute
    if let Err(e) = unsafe { command_list.Close() } {
        let diag = state
            .devices
            .get(&device_handle)
            .and_then(|dev| drain_info_queue(&dev.device))
            .unwrap_or_else(|| {
                "  (no debug-layer messages; enable GOLDY_DX12_DEBUG=1)\n".to_string()
            });
        return Err(anyhow::anyhow!(
            "Failed to close command list: {e}\nDebug layer messages:\n{diag}"
        ));
    }

    let cmd_list: ID3D12CommandList = command_list.cast().context("Failed to cast command list")?;

    let logical_device = state.devices.get(&device_handle).unwrap();
    unsafe {
        logical_device
            .command_queue
            .ExecuteCommandLists(&[Some(cmd_list)]);
    }

    // Signal fence with the token we reserved
    unsafe {
        logical_device
            .command_queue
            .Signal(&logical_device.fence, fence_value)
    }
    .context("Failed to signal fence")?;

    // Update the slot's fence_value so we know when it can be reused
    if let Some(dev) = state.devices.get_mut(&device_handle) {
        if let Some(slot) = dev.compute_allocator_pool.get_mut(slot_idx) {
            slot.fence_value = fence_value;
        }
    }

    state
        .staging_belts
        .entry(device_handle)
        .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
        .finish(fence_value);

    if !staged_texture_uploads.is_empty() {
        let resources = staged_texture_uploads
            .into_iter()
            .map(|u| u.staging_resource)
            .collect::<Vec<_>>();
        state
            .staging_belts
            .entry(device_handle)
            .or_insert_with(|| staging::StagingBelt::new(staging::DEFAULT_STAGING_CHUNK_SIZE))
            .defer_standalone_resources(fence_value, resources);
    }

    debug_assert_eq!(
        belt_idx,
        belt_slices.len(),
        "WriteBuffer storage count mismatch vs belt prepass"
    );

    Ok(fence_value)
}

/// Check if the fence for the given token has signaled.
pub(super) fn is_fence_complete(
    state: &Dx12State,
    device_handle: DeviceHandle,
    token: FenceToken,
) -> bool {
    let logical_device = match state.devices.get(&device_handle) {
        Some(dev) => dev,
        None => return false,
    };
    (unsafe { logical_device.fence.GetCompletedValue() }) >= token
}

/// Block until the fence signals.
pub(super) fn wait_fence(
    state: &Dx12State,
    device_handle: DeviceHandle,
    token: FenceToken,
) -> Result<()> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    super::utils::wait_for_fence(&logical_device.fence, token)?;

    // Detect TDR: device removal signals all fences with UINT64_MAX
    let completed = unsafe { logical_device.fence.GetCompletedValue() };
    if completed == u64::MAX {
        let reason = unsafe { logical_device.device.GetDeviceRemovedReason() };
        anyhow::bail!("GPU device removed (TDR) after fence wait: {:?}", reason);
    }
    Ok(())
}

/// Wait with timeout. Returns Ok(true) if signaled, Ok(false) if timeout elapsed.
pub(super) fn wait_fence_timeout(
    state: &Dx12State,
    device_handle: DeviceHandle,
    token: FenceToken,
    timeout_ms: u32,
) -> Result<bool> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;
    super::utils::wait_for_fence_timeout(&logical_device.fence, token, timeout_ms)
}
