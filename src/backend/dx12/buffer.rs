//! Buffer management logic.

use super::types::{BufferState, Dx12State};
use super::{BufferHandle, DeviceHandle};
use crate::backend::DataAccess;
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
) -> Result<BufferHandle> {
    // First pass: create the resource (immutable borrow of device)
    let (resource, upload_buffer, is_storage, bindless_enabled) = {
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Scattered access -> storage buffer (UAV), Broadcast access -> uniform buffer (CBV)
        let is_storage = access == DataAccess::Scattered;

        // Storage buffers need DEFAULT heap for UAV support (bindless)
        // Non-storage buffers can use UPLOAD heap for simpler CPU access
        let (heap_type, resource_flags) = if is_storage && logical_device.bindless_enabled {
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
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS // Start in UAV state for compute access
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

        // For DEFAULT heap buffers, create an upload buffer for CPU writes
        let upload_buffer = if heap_type == D3D12_HEAP_TYPE_DEFAULT {
            let upload_heap_properties = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            };
            let upload_desc = D3D12_RESOURCE_DESC {
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
            let mut upload: Option<ID3D12Resource> = None;
            unsafe {
                logical_device.device.CreateCommittedResource(
                    &upload_heap_properties,
                    D3D12_HEAP_FLAG_NONE,
                    &upload_desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut upload,
                )
            }
            .context("Failed to create upload buffer")?;
            Some(upload.context("CreateCommittedResource returned null for upload buffer")?)
        } else {
            None
        };

        (
            resource,
            upload_buffer,
            is_storage,
            logical_device.bindless_enabled,
        )
    };

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    // Second pass: register in bindless heap if enabled
    // Scattered access -> UAV + SRV descriptors, Broadcast access -> CBV descriptors
    let is_uniform = access == DataAccess::Broadcast;
    let (bindless_offset, bindless_srv_offset) = if bindless_enabled && (is_storage || is_uniform) {
        let logical_device = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        if is_storage {
            // For storage buffers, create BOTH UAV (for compute write) and SRV (for graphics read)
            // Use the provided element stride, or default to 4 bytes (uint/float) for compatibility
            let stride = element_stride.unwrap_or(4);
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
        },
    );

    Ok(handle)
}

/// Destroy a buffer, unregistering it from bindless.
pub(super) fn destroy(state: &mut Dx12State, buffer_handle: BufferHandle) {
    if let Some(buffer) = state.buffers.remove(&buffer_handle) {
        if let Some(device) = state.devices.get_mut(&buffer.device_handle) {
            device.resource_registry.unregister_buffer(buffer_handle);
        }
    }
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

/// Write data to a buffer at the specified offset.
pub(super) fn write(
    state: &mut Dx12State,
    buffer_handle: BufferHandle,
    offset: u64,
    data: &[u8],
) -> Result<()> {
    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    if offset + data.len() as u64 > buffer.size {
        anyhow::bail!("Write would exceed buffer bounds");
    }

    // Determine which resource to map (upload buffer for DEFAULT heap, main resource for UPLOAD heap)
    let map_resource = buffer.upload_buffer.as_ref().unwrap_or(&buffer.resource);

    // Map the buffer
    let mut mapped_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let read_range = D3D12_RANGE { Begin: 0, End: 0 }; // We're only writing

    unsafe { map_resource.Map(0, Some(&read_range), Some(&mut mapped_ptr)) }
        .context("Failed to map buffer")?;

    // Copy data
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            (mapped_ptr as *mut u8).add(offset as usize),
            data.len(),
        );
    }

    // Unmap
    let written_range = D3D12_RANGE {
        Begin: offset as usize,
        End: (offset as usize) + data.len(),
    };
    unsafe { map_resource.Unmap(0, Some(&written_range)) };

    // If we have an upload buffer, we need to copy to the main resource
    if let Some(upload_buffer) = &buffer.upload_buffer {
        let device_handle = buffer.device_handle;
        let main_resource = buffer.resource.clone();
        let upload_resource = upload_buffer.clone();
        let size = buffer.size;

        // Get device for copy operation
        let device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Create a one-shot command list for the copy
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

        // Transition main resource from UAV to COPY_DEST
        let barrier_to_copy = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(&main_resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
                }),
            },
        };
        unsafe { copy_list.ResourceBarrier(&[barrier_to_copy]) };

        // Copy from upload to main
        unsafe {
            copy_list.CopyBufferRegion(&main_resource, 0, &upload_resource, 0, size);
        }

        // Transition back to UAV
        let barrier_to_uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(&main_resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                    StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                }),
            },
        };
        unsafe { copy_list.ResourceBarrier(&[barrier_to_uav]) };

        // Close and execute
        unsafe { copy_list.Close() }.context("Failed to close copy command list")?;
        let lists: [Option<ID3D12CommandList>; 1] = [Some(copy_list.cast()?)];
        unsafe { device.command_queue.ExecuteCommandLists(&lists) };

        // Wait for completion using fence
        let fence_value = device.fence_value + 1;
        unsafe { device.command_queue.Signal(&device.fence, fence_value) }
            .context("Failed to signal fence")?;
        wait_for_fence(&device.fence, fence_value)?;

        // Update fence value for next operation (must be done after wait completes)
        // Note: device_handle is captured before this block, so we can use get_mut here
        if let Some(dev) = state.devices.get_mut(&device_handle) {
            dev.fence_value = fence_value + 1;
        }
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

    if buffer.upload_buffer.is_some() {
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

        let to_copy_src = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(&main_resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        };
        unsafe { copy_list.ResourceBarrier(&[to_copy_src]) };
        unsafe { copy_list.CopyBufferRegion(&readback, 0, &main_resource, 0, len) };
        let to_uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(&main_resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                }),
            },
        };
        unsafe { copy_list.ResourceBarrier(&[to_uav]) };
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

/// Fill buffer region with zeros (standalone, synchronous version).
///
/// For DEFAULT heap buffers (storage), zeros upload buffer and copies.
/// For UPLOAD heap buffers (uniform), zeroes directly via Map.
pub(super) fn clear(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    offset: u64,
    size: u64,
) -> Result<()> {
    use windows::Win32::Graphics::Direct3D12::*;

    let buffer = state
        .buffers
        .get(&buffer_handle)
        .context("Invalid buffer handle")?;

    let clear_size = if size == 0 {
        buffer.size.saturating_sub(offset)
    } else {
        size
    };

    if offset + clear_size > buffer.size {
        anyhow::bail!("Clear would exceed buffer bounds");
    }
    if clear_size == 0 {
        return Ok(());
    }

    if let Some(upload_buf) = &buffer.upload_buffer {
        // DEFAULT heap: zero upload buffer region, then GPU copy
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        let no_read = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { upload_buf.Map(0, Some(&no_read), Some(&mut mapped)) }
            .context("Failed to map upload buffer")?;
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
        unsafe { upload_buf.Unmap(0, Some(&written)) };

        let main_resource = buffer.resource.clone();
        let upload_resource = upload_buf.clone();
        let device = state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

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

        let to_copy = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(&main_resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
                }),
            },
        };
        unsafe { copy_list.ResourceBarrier(&[to_copy]) };
        unsafe {
            copy_list.CopyBufferRegion(&main_resource, offset, &upload_resource, offset, clear_size)
        };
        let to_uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(&main_resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                    StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                }),
            },
        };
        unsafe { copy_list.ResourceBarrier(&[to_uav]) };
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
