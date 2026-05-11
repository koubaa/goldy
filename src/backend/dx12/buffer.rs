//! Buffer management logic.

use super::barriers;
use super::types::{BufferState, Dx12State};
use super::{BufferHandle, DeviceHandle};
use crate::backend::DataAccess;
use crate::types::BufferFlags;
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

/// Create a buffer with the given size and access pattern.
#[allow(clippy::too_many_arguments)]
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    size: u64,
    access: DataAccess,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<BufferHandle> {
    let cpu_readable = flags.contains(BufferFlags::CPU_READABLE);
    // First pass: create the resource (immutable borrow of device)
    let (resource, upload_buffer, is_storage, coherent_readback, coherent_readback_mapped) = {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Scattered access -> storage buffer (UAV), Broadcast access -> uniform buffer (CBV)
        let is_storage = access == DataAccess::Scattered;

        if cpu_readable && !is_storage {
            anyhow::bail!("BufferFlags::CPU_READABLE is only valid for DataAccess::Scattered (storage) buffers");
        }

        // Storage buffers need DEFAULT heap for UAV support (bindless)
        // Non-storage buffers can use UPLOAD heap for simpler CPU access
        let (heap_type, resource_flags) = if is_storage {
            (
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            )
        } else {
            (D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE)
        };

        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: resource_flags,
        };

        let initial_state = if heap_type == D3D12_HEAP_TYPE_UPLOAD {
            D3D12_RESOURCE_STATE_GENERIC_READ
        } else {
            // Enhanced barriers: COMMON initial state; access is expressed via Barrier().
            D3D12_RESOURCE_STATE_COMMON
        };

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            logical_device.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                initial_state,
                None,
                &mut resource,
            )
        }
        .context("Failed to create buffer resource")?;

        let resource = resource.context("CreateCommittedResource returned null")?;

        // Upload buffer is created lazily on first write() to avoid doubling memory
        // for buffers that are only GPU-written (intermediate compute buffers, pool backing).
        let upload_buffer = None;

        // CPU_READABLE storage: pair DEFAULT UAV with a READBACK heap for `read_coherent`.
        let (coherent_readback, coherent_readback_mapped) = if cpu_readable && is_storage {
            let readback_heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            };
            let readback_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_NONE,
            };
            let mut rb: Option<ID3D12Resource> = None;
            unsafe {
                logical_device.device.CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut rb,
                )
            }
            .context("Failed to create CPU_READABLE readback buffer")?;
            let rb = rb.context("CreateCommittedResource readback returned null")?;
            let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
            let no_read = D3D12_RANGE { Begin: 0, End: 0 };
            unsafe { rb.Map(0, Some(&no_read), Some(&mut mapped)) }
                .context("Failed to map CPU_READABLE readback buffer")?;
            let p = mapped as *mut u8;
            if p.is_null() {
                anyhow::bail!("Map returned null for CPU_READABLE readback");
            }
            (Some(rb), Some(p as usize))
        } else {
            (None, None)
        };

        (
            resource,
            upload_buffer,
            is_storage,
            coherent_readback,
            coherent_readback_mapped,
        )
    };

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    // Second pass: register in bindless heap
    // Scattered access -> UAV + SRV descriptors, Broadcast access -> CBV descriptors
    let is_uniform = access == DataAccess::Broadcast;
    let (bindless_offset, bindless_srv_offset) = if is_storage || is_uniform {
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        if is_storage {
            // For storage buffers, create BOTH UAV (for compute write) and SRV (for graphics read)
            let stride = element_stride.unwrap_or(4);
            debug_assert!(
                stride > 0 && (size as u32).is_multiple_of(stride),
                "buffer size {size} not evenly divisible by element stride {stride} — \
                 likely a stride mismatch (set BufferProxy::element_stride or \
                 update element_stride_for_buffer)"
            );
            let num_elements = (size as u32) / stride;

            // Register UAV to get the next available descriptor offset
            let uav_offset = logical_device.resource_registry.register_buffer_uav(handle);

            // Create UAV descriptor for RWStructuredBuffer (compute write access)
            let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN, // Required for structured buffers
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: num_elements,
                        StructureByteStride: stride,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                    },
                },
            };

            let uav_cpu_handle = unsafe {
                let mut cpu_handle = logical_device
                    .cbv_srv_uav_heap
                    .GetCPUDescriptorHandleForHeapStart();
                cpu_handle.ptr +=
                    (uav_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
                cpu_handle
            };

            unsafe {
                logical_device.device.CreateUnorderedAccessView(
                    &resource,
                    None,
                    Some(&uav_desc),
                    uav_cpu_handle,
                );
            }

            // Also register and create SRV for read-only graphics access (StructuredBuffer)
            let srv_offset = logical_device.resource_registry.register_buffer_srv(handle);

            let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN, // Required for structured buffers
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: 0,
                        NumElements: num_elements,
                        StructureByteStride: stride,
                        Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                    },
                },
            };

            let srv_cpu_handle = unsafe {
                let mut cpu_handle = logical_device
                    .cbv_srv_uav_heap
                    .GetCPUDescriptorHandleForHeapStart();
                cpu_handle.ptr +=
                    (srv_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
                cpu_handle
            };

            unsafe {
                logical_device.device.CreateShaderResourceView(
                    &resource,
                    Some(&srv_desc),
                    srv_cpu_handle,
                );
            }

            tracing::debug!(
                "Created UAV at {} and SRV at {} for storage buffer {}",
                uav_offset,
                srv_offset,
                handle
            );
            (Some(uav_offset), Some(srv_offset))
        } else {
            // For uniform buffers, create a CBV (ConstantBuffer pattern)
            let cbv_offset = logical_device.resource_registry.register_buffer_cbv(handle);

            // CBV size must be 256-byte aligned
            let aligned_size = (size + 255) & !255;

            // Create CBV descriptor
            let cbv_desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: unsafe { resource.GetGPUVirtualAddress() },
                SizeInBytes: aligned_size as u32,
            };

            let cbv_handle = unsafe {
                let mut cpu_handle = logical_device
                    .cbv_srv_uav_heap
                    .GetCPUDescriptorHandleForHeapStart();
                cpu_handle.ptr +=
                    (cbv_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
                cpu_handle
            };

            unsafe {
                logical_device
                    .device
                    .CreateConstantBufferView(Some(&cbv_desc), cbv_handle);
            }

            tracing::debug!(
                "Created CBV for buffer {} at heap offset {}",
                handle,
                cbv_offset
            );
            (Some(cbv_offset), None) // No SRV for uniform buffers
        }
    } else {
        (None, None)
    };

    state.buffers.insert(
        handle,
        BufferState {
            device_handle,
            resource,
            size,
            bindless_offset,
            bindless_srv_offset,
            is_storage,
            upload_buffer,
            element_stride,
            is_view: false,
            coherent_readback,
            coherent_readback_mapped,
            flags,
            transient_placed: false,
        },
    );

    Ok(handle)
}

/// Destroy a buffer, queueing both the D3D12 resources and the bindless
/// descriptor slots for deferred deletion after in-flight GPU work completes.
/// For views, only the descriptor slots are deferred.
pub(super) fn destroy(state: &mut Dx12State, buffer_handle: BufferHandle) {
    if let Some(buffer) = state.buffers.remove(&buffer_handle) {
        if let Some(device) = state.devices.get_mut(&buffer.device_handle) {
            let last_fence = device.fence_value.saturating_sub(1);

            if buffer.is_view {
                device.deletion_queue.queue(
                    last_fence,
                    super::types::PendingDeletion::BufferView { buffer_handle },
                );
                return;
            }

            if buffer.transient_placed {
                device.resource_registry.unregister_buffer(buffer_handle);
                return;
            }
            if buffer.coherent_readback_mapped.is_some() {
                if let Some(ref rb) = buffer.coherent_readback {
                    let no_write = D3D12_RANGE { Begin: 0, End: 0 };
                    unsafe { rb.Unmap(0, Some(&no_write)) };
                }
            }
            device.deletion_queue.queue(
                last_fence,
                super::types::PendingDeletion::Buffer {
                    buffer_handle,
                    resource: buffer.resource,
                    upload_buffer: buffer.upload_buffer,
                    coherent_readback: buffer.coherent_readback,
                },
            );
        }
    }
}

/// Create a view into a sub-region of an existing storage buffer.
///
/// The view gets its own UAV and SRV descriptors in the bindless heap, pointing at
/// a sub-range of the parent's resource via `FirstElement` / `NumElements`.
pub(super) fn create_view(
    state: &mut Dx12State,
    parent_handle: BufferHandle,
    offset: u64,
    size: u64,
    element_stride: Option<u32>,
) -> Result<BufferHandle> {
    let parent = state
        .buffers
        .get(&parent_handle)
        .context("Invalid parent buffer handle")?;

    if offset + size > parent.size {
        anyhow::bail!(
            "View [{}, {}) exceeds parent buffer size {}",
            offset,
            offset + size,
            parent.size
        );
    }

    if !parent.is_storage {
        anyhow::bail!("Buffer views are only supported for storage (Scattered) buffers");
    }

    let stride = element_stride.unwrap_or(4);
    debug_assert!(
        stride > 0 && (size as u32).is_multiple_of(stride),
        "view size {size} not evenly divisible by element stride {stride}"
    );
    if !offset.is_multiple_of(stride as u64) {
        anyhow::bail!(
            "View offset {} is not aligned to element stride {}",
            offset,
            stride
        );
    }

    let device_handle = parent.device_handle;
    let resource = parent.resource.clone();
    let parent_flags = parent.flags;
    let first_element = (offset / stride as u64) as u32;
    let num_elements = (size as u32) / stride;

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    let (bindless_offset, bindless_srv_offset) = {
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        if num_elements == 0 {
            (None, None)
        } else {
            // UAV descriptor
            let uav_offset = logical_device.resource_registry.register_buffer_uav(handle);

            let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: first_element as u64,
                        NumElements: num_elements,
                        StructureByteStride: stride,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                    },
                },
            };

            let uav_cpu_handle = unsafe {
                let mut h = logical_device
                    .cbv_srv_uav_heap
                    .GetCPUDescriptorHandleForHeapStart();
                h.ptr += (uav_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
                h
            };

            unsafe {
                logical_device.device.CreateUnorderedAccessView(
                    &resource,
                    None,
                    Some(&uav_desc),
                    uav_cpu_handle,
                );
            }

            // SRV descriptor
            let srv_offset = logical_device.resource_registry.register_buffer_srv(handle);

            let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: first_element as u64,
                        NumElements: num_elements,
                        StructureByteStride: stride,
                        Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                    },
                },
            };

            let srv_cpu_handle = unsafe {
                let mut h = logical_device
                    .cbv_srv_uav_heap
                    .GetCPUDescriptorHandleForHeapStart();
                h.ptr += (srv_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
                h
            };

            unsafe {
                logical_device.device.CreateShaderResourceView(
                    &resource,
                    Some(&srv_desc),
                    srv_cpu_handle,
                );
            }

            tracing::debug!(
                "Created buffer view {} (UAV={}, SRV={}) into parent {} (offset={}, size={})",
                handle,
                uav_offset,
                srv_offset,
                parent_handle,
                offset,
                size
            );

            (Some(uav_offset), Some(srv_offset))
        }
    };

    state.buffers.insert(
        handle,
        BufferState {
            device_handle,
            resource,
            size,
            bindless_offset,
            bindless_srv_offset,
            is_storage: true,
            upload_buffer: None,
            element_stride,
            is_view: true,
            coherent_readback: None,
            coherent_readback_mapped: None,
            flags: parent_flags,
            transient_placed: false,
        },
    );

    Ok(handle)
}

/// Wait for a fence to reach the specified value.
/// This is a low-level helper for GPU synchronization.
fn wait_for_fence(fence: &ID3D12Fence, value: u64) -> Result<()> {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{CreateEventA, WaitForSingleObject, INFINITE};

    if unsafe { fence.GetCompletedValue() } < value {
        let event =
            unsafe { CreateEventA(None, false, false, None) }.context("Failed to create event")?;

        unsafe { fence.SetEventOnCompletion(value, event) }
            .context("Failed to set event on completion")?;

        unsafe { WaitForSingleObject(event, INFINITE) };
        unsafe { CloseHandle(event) }.ok();
    }
    Ok(())
}

/// Max size for the staging/upload buffer used in chunked writes.
/// Large uploads are split into chunks of this size to avoid massive staging allocations.
const UPLOAD_CHUNK_SIZE: u64 = 16 * 1024 * 1024; // 16 MB

/// Ensure the upload (staging) buffer exists for a DEFAULT-heap storage buffer.
///
/// Called by `ComputeCommand::WriteBuffer` handling in `compute::submit` so the
/// upload resource is ready before command recording begins.
pub(super) fn ensure_upload_buffer(
    state: &mut Dx12State,
    buffer_handle: BufferHandle,
    min_size: u64,
) -> Result<()> {
    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("ensure_upload_buffer: invalid handle")?;
    if buffer.upload_buffer.is_some() {
        return Ok(());
    }
    let chunk_size = min_size.min(UPLOAD_CHUNK_SIZE);
    let device_handle = buffer.device_handle;
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("ensure_upload_buffer: invalid device")?;
    let upload_heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };
    let upload_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: chunk_size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut upload: Option<ID3D12Resource> = None;
    unsafe {
        logical_device.device.CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &upload_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut upload,
        )
    }
    .context("ensure_upload_buffer: create failed")?;
    state.buffers.get_mut(&buffer_handle).unwrap().upload_buffer =
        Some(upload.context("Upload buffer is null")?);
    Ok(())
}

/// Write data to a buffer at the specified offset.
///
/// For storage buffers (DEFAULT heap), uses a capped-size upload buffer and copies
/// in chunks to avoid doubling memory for huge buffers. For UPLOAD heap buffers
/// (uniform), maps directly.
pub(super) fn write(
    state: &mut Dx12State,
    buffer_handle: BufferHandle,
    offset: u64,
    data: &[u8],
) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    if offset + data.len() as u64 > buffer.size {
        anyhow::bail!("Write would exceed buffer bounds");
    }

    if buffer.is_storage {
        if let Some(stride) = buffer.element_stride {
            if stride > 0 && !(data.len() as u32).is_multiple_of(stride) {
                tracing::warn!(
                    "write of {} bytes to buffer (handle={}) with element stride {} \
                     — data length is not a multiple of stride, possible type mismatch",
                    data.len(),
                    buffer_handle,
                    stride,
                );
            }
        }
    }

    if !buffer.is_storage {
        // UPLOAD heap: direct map
        let mut mapped_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let read_range = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe {
            buffer
                .resource
                .Map(0, Some(&read_range), Some(&mut mapped_ptr))
        }
        .context("Failed to map buffer")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (mapped_ptr as *mut u8).add(offset as usize),
                data.len(),
            );
        }
        let written_range = D3D12_RANGE {
            Begin: offset as usize,
            End: (offset as usize) + data.len(),
        };
        unsafe { buffer.resource.Unmap(0, Some(&written_range)) };
        return Ok(());
    }

    // Storage buffer (DEFAULT heap): chunked upload via a capped-size staging buffer.
    let data_len = data.len() as u64;
    let chunk_size = data_len.min(UPLOAD_CHUNK_SIZE);

    let device_handle = buffer.device_handle;
    let main_resource = buffer.resource.clone();

    // Create or reuse the upload buffer (capped at chunk_size)
    if buffer.upload_buffer.is_none() {
        ensure_upload_buffer(state, buffer_handle, chunk_size)?;
    }

    let upload_buf = state
        .buffers
        .get(&buffer_handle)
        .unwrap()
        .upload_buffer
        .as_ref()
        .unwrap()
        .clone();
    let upload_buf_size = chunk_size;

    // Upload in chunks
    let mut written = 0u64;
    while written < data_len {
        let this_chunk = (data_len - written).min(upload_buf_size);
        let src_slice = &data[(written as usize)..((written + this_chunk) as usize)];

        // Map, copy data, unmap
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        let no_read = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { upload_buf.Map(0, Some(&no_read), Some(&mut mapped)) }
            .context("Failed to map upload buffer")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_slice.as_ptr(),
                mapped as *mut u8,
                this_chunk as usize,
            );
        }
        let write_range = D3D12_RANGE {
            Begin: 0,
            End: this_chunk as usize,
        };
        unsafe { upload_buf.Unmap(0, Some(&write_range)) };

        // GPU copy from staging to main buffer
        let device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let alloc: ID3D12CommandAllocator = unsafe {
            device
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .context("Failed to create command allocator")?;
        let cmd: ID3D12GraphicsCommandList = unsafe {
            device
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        }
        .context("Failed to create command list")?;
        let cmd7: ID3D12GraphicsCommandList7 = cmd.cast().context("ID3D12GraphicsCommandList7")?;

        let dst_offset = offset + written;
        let mut b_to_copy = [barriers::buffer_barrier_full(
            &main_resource,
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_ACCESS_COMMON,
            D3D12_BARRIER_ACCESS_COPY_DEST,
        )];
        let mut b_to_uav = [barriers::buffer_barrier_full(
            &main_resource,
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_ACCESS_COPY_DEST,
            D3D12_BARRIER_ACCESS_COMMON,
        )];
        unsafe {
            barriers::barrier_buffers(&cmd7, &b_to_copy);
            barriers::drop_buffer_barriers(&mut b_to_copy);
            cmd.CopyBufferRegion(&main_resource, dst_offset, &upload_buf, 0, this_chunk);
            barriers::barrier_buffers(&cmd7, &b_to_uav);
            barriers::drop_buffer_barriers(&mut b_to_uav);
        }
        unsafe { cmd.Close() }.context("Failed to close command list")?;

        let lists: [Option<ID3D12CommandList>; 1] = [Some(cmd.cast()?)];
        unsafe { device.command_queue.ExecuteCommandLists(&lists) };

        let fence_value = device.fence_value + 1;
        unsafe { device.command_queue.Signal(&device.fence, fence_value) }
            .context("Failed to signal fence")?;
        wait_for_fence(&device.fence, fence_value)?;
        if let Some(dev) = state.devices.get_mut(&device_handle) {
            dev.fence_value = fence_value + 1;
        }

        written += this_chunk;
    }

    Ok(())
}

/// Get the size of a buffer in bytes.
pub(super) fn size(state: &Dx12State, buffer_handle: BufferHandle) -> u64 {
    state
        .buffers
        .get(&buffer_handle)
        .map(|b| b.size)
        .unwrap_or(0)
}

/// Get the bindless descriptor index for a buffer, if any.
pub(super) fn bindless_index(state: &Dx12State, buffer_handle: BufferHandle) -> Option<u32> {
    state
        .buffers
        .get(&buffer_handle)
        .and_then(|b| b.bindless_offset)
}

/// Get the SRV (read-only / StructuredBuffer) bindless index for a storage buffer.
/// Scattered buffers have both a UAV (at bindless_offset) and an SRV (at bindless_srv_offset).
pub(super) fn bindless_srv_index(state: &Dx12State, buffer_handle: BufferHandle) -> Option<u32> {
    state
        .buffers
        .get(&buffer_handle)
        .and_then(|b| b.bindless_srv_offset.or(b.bindless_offset))
}

/// Record DEFAULT → READBACK copy for a `CPU_READABLE` buffer on an already-open command list.
/// Does not submit.
pub(super) fn emit_copy_coherent_readback_on_command_list(
    state: &Dx12State,
    buffer_handle: BufferHandle,
    command_list: &ID3D12GraphicsCommandList,
    command_list7: &ID3D12GraphicsCommandList7,
) -> Result<()> {
    use windows::Win32::Graphics::Direct3D12::*;

    let (main_resource, readback, len) = {
        let buffer = state
            .buffers
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;
        if !buffer.flags.contains(BufferFlags::CPU_READABLE) || !buffer.is_storage {
            return Ok(());
        }
        let readback = buffer
            .coherent_readback
            .as_ref()
            .context("CPU_READABLE buffer missing readback resource")?;
        (buffer.resource.clone(), readback.clone(), buffer.size)
    };

    let pre_copy = D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC_ALL,
        SyncAfter: D3D12_BARRIER_SYNC_COPY,
        AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
        AccessAfter: D3D12_BARRIER_ACCESS_COPY_SOURCE,
    };
    let post_copy = D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC_COPY,
        SyncAfter: D3D12_BARRIER_SYNC_ALL,
        AccessBefore: D3D12_BARRIER_ACCESS_COPY_SOURCE,
        AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
    };
    unsafe {
        barriers::barrier_globals(command_list7, &[pre_copy]);
    }
    unsafe {
        command_list.CopyBufferRegion(&readback, 0, &main_resource, 0, len);
    }
    unsafe {
        barriers::barrier_globals(command_list7, &[post_copy]);
    }
    Ok(())
}

/// Standalone GPU copy + wait for `read_to_cpu` on `CPU_READABLE` buffers.
/// Creates a one-shot command list to copy the UAV to the paired READBACK heap.
fn standalone_copy_coherent_readback(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
) -> Result<()> {
    use super::utils::wait_for_fence;
    use windows::Win32::Graphics::Direct3D12::*;

    let device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let copy_allocator: ID3D12CommandAllocator = unsafe {
        device
            .device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
    }
    .context("Failed to create copy command allocator (coherent readback)")?;

    let copy_list: ID3D12GraphicsCommandList = unsafe {
        device
            .device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &copy_allocator, None)
    }
    .context("Failed to create copy command list (coherent readback)")?;
    let copy_list7: ID3D12GraphicsCommandList7 = copy_list.cast()?;

    emit_copy_coherent_readback_on_command_list(state, buffer_handle, &copy_list, &copy_list7)?;
    unsafe { copy_list.Close() }
        .context("Failed to close copy command list (coherent readback)")?;

    let lists: [Option<ID3D12CommandList>; 1] = [Some(
        copy_list.cast().context("Failed to cast command list")?,
    )];
    unsafe { device.command_queue.ExecuteCommandLists(&lists) };

    let fence_value = device.fence_value + 1;
    unsafe { device.command_queue.Signal(&device.fence, fence_value) }
        .context("Failed to signal fence (coherent readback)")?;
    wait_for_fence(&device.fence, fence_value)?;

    if let Some(dev) = state.devices.get_mut(&device_handle) {
        dev.fence_value = fence_value + 1;
    }
    Ok(())
}

/// Read from the persistent READBACK map after the submit's auto-copy.
pub(super) fn read_coherent(
    buffers: &std::collections::HashMap<BufferHandle, BufferState>,
    buffer_handle: BufferHandle,
    offset: u64,
    output: &mut [u8],
) -> Result<()> {
    let buffer = buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;
    if !buffer.flags.contains(BufferFlags::CPU_READABLE) {
        anyhow::bail!("read_coherent requires BufferFlags::CPU_READABLE");
    }
    let base = buffer
        .coherent_readback_mapped
        .context("CPU_READABLE buffer not mapped for readback")?;
    if offset + output.len() as u64 > buffer.size {
        anyhow::bail!("read_coherent would exceed buffer bounds");
    }
    let p = base as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(
            p.add(offset as usize) as *const u8,
            output.as_mut_ptr(),
            output.len(),
        );
    }
    Ok(())
}

/// Read buffer contents back to CPU memory.
///
/// For DEFAULT heap buffers (storage), creates a readback buffer and copies.
/// For UPLOAD heap buffers (uniform), reads directly via Map.
pub(super) fn read_to_cpu(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    output: &mut [u8],
) -> Result<()> {
    use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    let len = output.len() as u64;
    if len > buffer.size {
        anyhow::bail!("Read would exceed buffer bounds");
    }

    if buffer.is_storage
        && buffer.flags.contains(BufferFlags::CPU_READABLE)
        && buffer.coherent_readback.is_some()
    {
        standalone_copy_coherent_readback(state, device_handle, buffer_handle)?;
        return read_coherent(&state.buffers, buffer_handle, 0, output);
    }

    if buffer.is_storage {
        // DEFAULT heap (storage): need a READBACK buffer + GPU copy
        let device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: len,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut readback: Option<ID3D12Resource> = None;
        unsafe {
            device.device.CreateCommittedResource(
                &readback_heap,
                D3D12_HEAP_FLAG_NONE,
                &readback_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut readback,
            )
        }
        .context("Failed to create readback buffer")?;
        let readback = readback.context("Readback resource is null")?;

        let main_resource = buffer.resource.clone();

        // Command list: transition → copy → transition back
        let copy_allocator: ID3D12CommandAllocator = unsafe {
            device
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .context("Failed to create copy command allocator")?;

        let copy_list: ID3D12GraphicsCommandList = unsafe {
            device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &copy_allocator,
                None,
            )
        }
        .context("Failed to create copy command list")?;
        let copy_list7: ID3D12GraphicsCommandList7 =
            copy_list.cast().context("ID3D12GraphicsCommandList7")?;

        // Use global barriers instead of per-buffer barriers. D3D12_BARRIER_TYPE_BUFFER
        // enhanced barriers cause WARP on Windows Server to silently remove the device
        // during ExecuteCommandLists, making the subsequent Signal() call AV.
        // Global barriers (D3D12_BARRIER_TYPE_GLOBAL) are proven to work on all targets.
        let pre_copy = D3D12_GLOBAL_BARRIER {
            SyncBefore: D3D12_BARRIER_SYNC_ALL,
            SyncAfter: D3D12_BARRIER_SYNC_COPY,
            AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            AccessAfter: D3D12_BARRIER_ACCESS_COPY_SOURCE,
        };
        let post_copy = D3D12_GLOBAL_BARRIER {
            SyncBefore: D3D12_BARRIER_SYNC_COPY,
            SyncAfter: D3D12_BARRIER_SYNC_ALL,
            AccessBefore: D3D12_BARRIER_ACCESS_COPY_SOURCE,
            AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
        };
        unsafe {
            barriers::barrier_globals(&copy_list7, &[pre_copy]);
        }
        unsafe {
            copy_list.CopyBufferRegion(&readback, 0, &main_resource, 0, len);
        }
        unsafe {
            barriers::barrier_globals(&copy_list7, &[post_copy]);
        }
        unsafe { copy_list.Close() }.context("Failed to close copy command list")?;

        let lists: [Option<ID3D12CommandList>; 1] = [Some(
            copy_list.cast().context("Failed to cast command list")?,
        )];
        unsafe { device.command_queue.ExecuteCommandLists(&lists) };

        let fence_value = device.fence_value + 1;
        unsafe { device.command_queue.Signal(&device.fence, fence_value) }
            .context("Failed to signal fence")?;
        wait_for_fence(&device.fence, fence_value)?;

        if let Some(dev) = state.devices.get_mut(&device_handle) {
            dev.fence_value = fence_value + 1;
        }

        // Map readback buffer and copy to output
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        let read_range = D3D12_RANGE {
            Begin: 0,
            End: len as usize,
        };
        unsafe { readback.Map(0, Some(&read_range), Some(&mut mapped)) }
            .context("Failed to map readback buffer")?;
        unsafe {
            std::ptr::copy_nonoverlapping(mapped as *const u8, output.as_mut_ptr(), len as usize);
        }
        let no_write = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { readback.Unmap(0, Some(&no_write)) };
    } else {
        // UPLOAD heap (uniform): directly mappable
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        let read_range = D3D12_RANGE {
            Begin: 0,
            End: len as usize,
        };
        unsafe { buffer.resource.Map(0, Some(&read_range), Some(&mut mapped)) }
            .context("Failed to map buffer")?;
        unsafe {
            std::ptr::copy_nonoverlapping(mapped as *const u8, output.as_mut_ptr(), len as usize);
        }
        let no_write = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { buffer.resource.Unmap(0, Some(&no_write)) };
    }

    Ok(())
}

/// GPU-side zero fill using ClearUnorderedAccessViewUint.
///
/// Always creates a R32_UINT UAV (stride=0) so ClearUnorderedAccessViewUint works
/// regardless of the buffer's native structured stride. The UAV is created in both
/// the shader-visible heap (GPU handle) and the non-shader-visible cpu_clear_heap
/// (CPU handle), as required by the D3D12 API.
pub(super) fn uav_clear(
    device: &super::types::LogicalDevice,
    buffer: &BufferState,
    command_list: &ID3D12GraphicsCommandList,
    offset: u64,
    clear_size: u64,
) -> Result<()> {
    let scratch = device.scratch_clear_uav_offset;
    let num_u32s = (buffer.size / 4) as u32;

    let raw_uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: 0,
                NumElements: num_u32s,
                StructureByteStride: 0,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_NONE,
            },
        },
    };

    let scratch_cpu = unsafe {
        let mut h = device.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
        h.ptr += (scratch * device.cbv_srv_uav_descriptor_size) as usize;
        h
    };
    unsafe {
        device.device.CreateUnorderedAccessView(
            &buffer.resource,
            None,
            Some(&raw_uav_desc),
            scratch_cpu,
        );
    }

    let gpu_handle = unsafe {
        let mut h = device.cbv_srv_uav_heap.GetGPUDescriptorHandleForHeapStart();
        h.ptr += (scratch as u64) * (device.cbv_srv_uav_descriptor_size as u64);
        h
    };

    let cpu_handle = unsafe { device.cpu_clear_heap.GetCPUDescriptorHandleForHeapStart() };
    unsafe {
        device.device.CreateUnorderedAccessView(
            &buffer.resource,
            None,
            Some(&raw_uav_desc),
            cpu_handle,
        );
    }

    if offset == 0 && clear_size == buffer.size {
        unsafe {
            command_list.ClearUnorderedAccessViewUint(
                gpu_handle,
                cpu_handle,
                &buffer.resource,
                &[0u32; 4],
                &[],
            );
        }
    } else {
        let first_element = offset / 4;
        let num_elements = clear_size / 4;
        let rect = windows::Win32::Foundation::RECT {
            left: first_element as i32,
            top: 0,
            right: (first_element + num_elements) as i32,
            bottom: 1,
        };
        unsafe {
            command_list.ClearUnorderedAccessViewUint(
                gpu_handle,
                cpu_handle,
                &buffer.resource,
                &[0u32; 4],
                &[rect],
            );
        }
    }

    Ok(())
}

/// Fill buffer region with zeros (standalone, synchronous version).
///
/// For DEFAULT heap buffers (storage), uses ClearUnorderedAccessViewUint (no staging needed).
/// For UPLOAD heap buffers (uniform), zeroes directly via Map.
pub(super) fn clear(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    offset: u64,
    size: u64,
) -> Result<()> {
    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    let clear_size = super::super::shared::resolve_clear_size(buffer.size, offset, size);

    if offset + clear_size > buffer.size {
        anyhow::bail!("Clear would exceed buffer bounds");
    }
    if clear_size == 0 {
        return Ok(());
    }

    if buffer.is_storage {
        // DEFAULT heap storage buffer: use ClearUnorderedAccessViewUint (GPU-side, no staging)
        let device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let copy_allocator: ID3D12CommandAllocator = unsafe {
            device
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .context("Failed to create command allocator")?;

        let cmd_list: ID3D12GraphicsCommandList = unsafe {
            device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &copy_allocator,
                None,
            )
        }
        .context("Failed to create command list")?;

        // Bind descriptor heaps (required for ClearUnorderedAccessViewUint)
        unsafe {
            cmd_list.SetDescriptorHeaps(&[
                Some(device.cbv_srv_uav_heap.clone()),
                Some(device.sampler_heap.clone()),
            ]);
        }

        uav_clear(device, buffer, &cmd_list, offset, clear_size)?;

        unsafe { cmd_list.Close() }.context("Failed to close command list")?;

        let lists: [Option<ID3D12CommandList>; 1] = [Some(
            cmd_list.cast().context("Failed to cast command list")?,
        )];
        unsafe { device.command_queue.ExecuteCommandLists(&lists) };

        let fence_value = device.fence_value + 1;
        unsafe { device.command_queue.Signal(&device.fence, fence_value) }
            .context("Failed to signal fence")?;
        wait_for_fence(&device.fence, fence_value)?;

        let removed_reason = unsafe { device.device.GetDeviceRemovedReason() };
        if removed_reason.is_err() {
            anyhow::bail!("Device removed during UAV clear: {:?}", removed_reason);
        }

        if let Some(dev) = state.devices.get_mut(&device_handle) {
            dev.fence_value = fence_value + 1;
        }
    } else {
        // UPLOAD heap: CPU-accessible, just memset
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        let no_read = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { buffer.resource.Map(0, Some(&no_read), Some(&mut mapped)) }
            .context("Failed to map buffer")?;
        unsafe {
            std::ptr::write_bytes(
                (mapped as *mut u8).add(offset as usize),
                0,
                clear_size as usize,
            );
        }
        let written = D3D12_RANGE {
            Begin: offset as usize,
            End: (offset + clear_size) as usize,
        };
        unsafe { buffer.resource.Unmap(0, Some(&written)) };
    }

    Ok(())
}
