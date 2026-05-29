//! Buffer management logic.

use super::barriers;
use super::tiles;
use super::types::{BufferState, Dx12State};
use super::{BufferHandle, DeviceHandle};
use crate::backend::DataAccess;
use crate::types::BufferFlags;
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

use super::types::LogicalDevice;

/// Minimum committed width for a uniform (upload) buffer so a CBV can be created.
///
/// [`D3D12_CONSTANT_BUFFER_VIEW_DESC::SizeInBytes`] must be a multiple of 256
/// ([`D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT`]). We align the logical
/// constant data size to that; the resource must be at least that wide.
#[inline]
fn uniform_buffer_allocation_width(logical_size: u64, requested_width: u64) -> u64 {
    debug_assert!(logical_size <= requested_width);
    let cbv_range = (logical_size + 255) & !255;
    requested_width.max(cbv_range)
}

/// Allocates main buffer resource and optional CPU_READABLE readback pairing (same as [`create`]).
fn alloc_committed_buffer_pair(
    logical_device: &LogicalDevice,
    size: u64,
    is_storage: bool,
    cpu_readable: bool,
) -> Result<(ID3D12Resource, Option<ID3D12Resource>, Option<usize>)> {
    if cpu_readable && !is_storage {
        anyhow::bail!(
            "BufferFlags::CPU_READABLE is only valid for storage (UAV) buffers on resize allocation"
        );
    }

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
    .context("resize: CreateCommittedResource main buffer")?;
    let resource = resource.context("resize: main buffer null")?;

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
        .context("resize: CreateCommittedResource readback")?;
        let rb = rb.context("resize: readback null")?;
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        let no_read = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { rb.Map(0, Some(&no_read), Some(&mut mapped)) }.context("resize: map readback")?;
        let p = mapped as *mut u8;
        if p.is_null() {
            anyhow::bail!("resize: Map readback returned null");
        }
        (Some(rb), Some(p as usize))
    } else {
        (None, None)
    };

    Ok((resource, coherent_readback, coherent_readback_mapped))
}

fn rewrite_root_buffer_descriptors(
    logical_device: &LogicalDevice,
    new_resource: &ID3D12Resource,
    new_size: u64,
    old: &BufferState,
) -> Result<()> {
    if old.is_storage {
        let stride = old.element_stride.unwrap_or(4);
        let num_elements = (new_size as u32) / stride;
        if let Some(uav_off) = old.bindless_offset {
            let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
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
                cpu_handle.ptr += (uav_off * logical_device.cbv_srv_uav_descriptor_size) as usize;
                cpu_handle
            };
            unsafe {
                logical_device.device.CreateUnorderedAccessView(
                    new_resource,
                    None,
                    Some(&uav_desc),
                    uav_cpu_handle,
                );
            }
        }
        if let Some(srv_off) = old.bindless_srv_offset {
            let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
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
                cpu_handle.ptr += (srv_off * logical_device.cbv_srv_uav_descriptor_size) as usize;
                cpu_handle
            };
            unsafe {
                logical_device.device.CreateShaderResourceView(
                    new_resource,
                    Some(&srv_desc),
                    srv_cpu_handle,
                );
            }
        }
    } else if let Some(cbv_off) = old.bindless_offset {
        let aligned_size = (new_size + 255) & !255;
        let cbv_desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: unsafe { new_resource.GetGPUVirtualAddress() },
            SizeInBytes: aligned_size as u32,
        };
        let cbv_handle = unsafe {
            let mut cpu_handle = logical_device
                .cbv_srv_uav_heap
                .GetCPUDescriptorHandleForHeapStart();
            cpu_handle.ptr += (cbv_off * logical_device.cbv_srv_uav_descriptor_size) as usize;
            cpu_handle
        };
        unsafe {
            logical_device
                .device
                .CreateConstantBufferView(Some(&cbv_desc), cbv_handle);
        }
    }
    Ok(())
}

fn patch_buffer_views_after_parent_resize(
    state: &mut Dx12State,
    parent_handle: BufferHandle,
) -> Result<()> {
    let new_resource = state
        .buffers
        .get(&parent_handle)
        .context("patch_buffer_views: parent missing")?
        .resource
        .clone();

    let view_handles: Vec<BufferHandle> = state
        .buffers
        .iter()
        .filter(|(_, b)| b.is_view && b.parent_for_view == Some(parent_handle))
        .map(|(&h, _)| h)
        .collect();

    for vh in view_handles {
        let (device_handle, stride, byte_off, view_size, uav_off, srv_off) = {
            let v = state.buffers.get(&vh).context("patch_buffer_views: view")?;
            (
                v.device_handle,
                v.element_stride.unwrap_or(4),
                v.view_byte_offset
                    .context("patch_buffer_views: view offset")?,
                v.size,
                v.bindless_offset,
                v.bindless_srv_offset,
            )
        };
        if stride == 0 {
            anyhow::bail!("patch_buffer_views: stride 0");
        }
        if !byte_off.is_multiple_of(stride as u64) {
            anyhow::bail!("patch_buffer_views: offset not stride-aligned");
        }
        let first_element = (byte_off / stride as u64) as u32;
        let num_elements = (view_size as u32) / stride;

        if let (Some(uav_off), Some(srv_off)) = (uav_off, srv_off) {
            if num_elements > 0 {
                let logical_device = state
                    .devices
                    .get_mut(&device_handle)
                    .context("patch_buffer_views: device")?;

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
                    h.ptr += (uav_off * logical_device.cbv_srv_uav_descriptor_size) as usize;
                    h
                };
                unsafe {
                    logical_device.device.CreateUnorderedAccessView(
                        &new_resource,
                        None,
                        Some(&uav_desc),
                        uav_cpu_handle,
                    );
                }

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
                    h.ptr += (srv_off * logical_device.cbv_srv_uav_descriptor_size) as usize;
                    h
                };
                unsafe {
                    logical_device.device.CreateShaderResourceView(
                        &new_resource,
                        Some(&srv_desc),
                        srv_cpu_handle,
                    );
                }
            }
        }

        state
            .buffers
            .get_mut(&vh)
            .context("patch_buffer_views: view mut")?
            .resource = new_resource.clone();
    }
    Ok(())
}

/// Resize buffer storage in place; bindless heap indices stay stable.
pub(super) fn resize(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_size: u64,
    preserve_contents: bool,
) -> Result<()> {
    let old = state
        .buffers
        .get(&buffer_handle)
        .cloned()
        .context("resize_buffer: invalid buffer")?;

    if old.device_handle != device_handle {
        anyhow::bail!("resize_buffer: buffer belongs to a different device");
    }
    if old.is_view {
        anyhow::bail!("resize_buffer: cannot resize a buffer view");
    }
    if old.transient_placed {
        anyhow::bail!("resize_buffer: cannot resize a transient placed buffer");
    }
    if new_size == old.size {
        return Ok(());
    }
    if old.is_reserved && new_size <= old.allocation_size {
        anyhow::bail!(
            "reserved buffer: growth within virtual capacity must use set_buffer_logical_size, not resize_buffer"
        );
    }

    let cpu_readable = old.flags.contains(BufferFlags::CPU_READABLE);
    if cpu_readable && !old.is_storage {
        anyhow::bail!("resize_buffer: invalid CPU_READABLE uniform buffer");
    }

    let stride = old.element_stride.unwrap_or(4);
    if old.is_storage && stride > 0 && new_size > 0 && !(new_size as u32).is_multiple_of(stride) {
        anyhow::bail!("resize_buffer: new size {new_size} not divisible by stride {stride}");
    }

    let logical_device_ro = state
        .devices
        .get(&device_handle)
        .context("resize_buffer: invalid device")?;

    let mut deletion_fence_marker = unsafe { logical_device_ro.fence.GetCompletedValue() };

    if old.coherent_readback_mapped.is_some() {
        if let Some(ref rb) = old.coherent_readback {
            let no_write = D3D12_RANGE { Begin: 0, End: 0 };
            unsafe { rb.Unmap(0, Some(&no_write)) };
        }
    }

    let new_alloc_width = if old.is_storage {
        new_size
    } else {
        uniform_buffer_allocation_width(new_size, new_size)
    };

    let (new_resource, new_readback, new_readback_mapped) = alloc_committed_buffer_pair(
        logical_device_ro,
        new_alloc_width,
        old.is_storage,
        cpu_readable,
    )?;

    let old_resource = old.resource.clone();
    let copy_len = if preserve_contents {
        old.size.min(new_size)
    } else {
        0
    };

    let need_copy = old.is_storage && copy_len > 0;
    let need_tail_clear = old.is_storage && preserve_contents && new_size > old.size;

    if old.is_storage && (need_copy || need_tail_clear) {
        let device = state
            .devices
            .get(&device_handle)
            .context("resize_buffer: device")?;

        let copy_allocator: ID3D12CommandAllocator = unsafe {
            device
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .context("resize_buffer: allocator")?;
        let cmd: ID3D12GraphicsCommandList = unsafe {
            device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &copy_allocator,
                None,
            )
        }
        .context("resize_buffer: command list")?;
        let cmd7: ID3D12GraphicsCommandList7 = cmd.cast().context("ID3D12GraphicsCommandList7")?;

        if need_copy {
            let mut b_src = [barriers::buffer_barrier_full(
                &old_resource,
                D3D12_BARRIER_SYNC_ALL,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
            )];
            let mut b_dst = [barriers::buffer_barrier_full(
                &new_resource,
                D3D12_BARRIER_SYNC_ALL,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COMMON,
                D3D12_BARRIER_ACCESS_COPY_DEST,
            )];
            unsafe {
                barriers::barrier_buffers(&cmd7, &b_src);
                barriers::drop_buffer_barriers(&mut b_src);
                barriers::barrier_buffers(&cmd7, &b_dst);
                barriers::drop_buffer_barriers(&mut b_dst);
                cmd.CopyBufferRegion(&new_resource, 0, &old_resource, 0, copy_len);
            }
        }

        if need_tail_clear {
            let tail_len = new_size - old.size;
            // If we just copied old content, new_resource is already in COPY_DEST.
            // Otherwise transition it from COMMON.
            if !need_copy {
                let mut b_to_copy = [barriers::buffer_barrier_full(
                    &new_resource,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_COMMON,
                    D3D12_BARRIER_ACCESS_COPY_DEST,
                )];
                unsafe {
                    barriers::barrier_buffers(&cmd7, &b_to_copy);
                    barriers::drop_buffer_barriers(&mut b_to_copy);
                }
            }
            // Zero-fill the tail via chunked CopyBufferRegion from the device's zero buffer.
            let zero = &device.zero_buffer;
            let mut tail_written = 0u64;
            while tail_written < tail_len {
                let this_chunk = (tail_len - tail_written).min(ZERO_BUFFER_SIZE);
                unsafe {
                    cmd.CopyBufferRegion(
                        &new_resource,
                        old.size + tail_written,
                        zero,
                        0,
                        this_chunk,
                    );
                }
                tail_written += this_chunk;
            }
            // Transition back to COMMON so the resource can be used as UAV by subsequent work.
            let mut b_to_common = [barriers::buffer_barrier_full(
                &new_resource,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_ALL,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_ACCESS_COMMON,
            )];
            unsafe {
                barriers::barrier_buffers(&cmd7, &b_to_common);
                barriers::drop_buffer_barriers(&mut b_to_common);
            }
        }

        unsafe { cmd.Close() }.context("resize_buffer: Close")?;
        let lists: [Option<ID3D12CommandList>; 1] = [Some(cmd.cast()?)];
        unsafe { device.command_queue.ExecuteCommandLists(&lists) };

        let fence_value = device.fence_value + 1;
        unsafe { device.command_queue.Signal(&device.fence, fence_value) }
            .context("resize_buffer: Signal")?;
        wait_for_fence(&device.fence, fence_value)?;
        deletion_fence_marker = fence_value;

        if let Some(dev) = state.devices.get_mut(&device_handle) {
            dev.fence_value = fence_value + 1;
        }
    } else if !old.is_storage && preserve_contents && copy_len > 0 {
        let mut src: *mut std::ffi::c_void = std::ptr::null_mut();
        let read_all = D3D12_RANGE {
            Begin: 0,
            End: old.size as usize,
        };
        unsafe { old_resource.Map(0, Some(&read_all), Some(&mut src)) }
            .context("resize_buffer: map old uniform")?;
        let mut dst: *mut std::ffi::c_void = std::ptr::null_mut();
        let dst_range = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { new_resource.Map(0, Some(&dst_range), Some(&mut dst)) }
            .context("resize_buffer: map new uniform")?;
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, copy_len as usize);
            if new_size > old.size {
                std::ptr::write_bytes(
                    (dst as *mut u8).add(old.size as usize),
                    0,
                    (new_size - old.size) as usize,
                );
            }
        }
        let written = D3D12_RANGE {
            Begin: 0,
            End: new_size as usize,
        };
        unsafe { new_resource.Unmap(0, Some(&written)) };
        let noop = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { old_resource.Unmap(0, Some(&noop)) };
    }

    let logical_device = state
        .devices
        .get_mut(&device_handle)
        .context("resize_buffer: device for descriptors")?;
    rewrite_root_buffer_descriptors(logical_device, &new_resource, new_size, &old)?;

    state.buffers.insert(
        buffer_handle,
        BufferState {
            device_handle,
            resource: new_resource,
            size: new_size,
            allocation_size: new_alloc_width,
            bindless_offset: old.bindless_offset,
            bindless_srv_offset: old.bindless_srv_offset,
            is_storage: old.is_storage,
            upload_buffer: None,
            element_stride: old.element_stride,
            is_view: false,
            coherent_readback: new_readback,
            coherent_readback_mapped: new_readback_mapped,
            flags: old.flags,
            transient_placed: false,
            parent_for_view: None,
            view_byte_offset: None,
            is_reserved: false,
            tile_byte_size: 0,
            reserved_tiles: Vec::new(),
        },
    );

    patch_buffer_views_after_parent_resize(state, buffer_handle)?;

    let dev_mut = state
        .devices
        .get_mut(&device_handle)
        .context("resize_buffer: queue deletion")?;
    if old.is_reserved {
        dev_mut.deletion_queue.queue(
            deletion_fence_marker,
            super::types::PendingDeletion::ReplacedReservedBufferGpu {
                resource: old_resource,
                tiles: old.reserved_tiles,
                upload_buffer: old.upload_buffer,
                coherent_readback: old.coherent_readback,
            },
        );
    } else {
        dev_mut.deletion_queue.queue(
            deletion_fence_marker,
            super::types::PendingDeletion::ReplacedBufferGpu {
                resource: old_resource,
                upload_buffer: old.upload_buffer,
                coherent_readback: old.coherent_readback,
            },
        );
    }

    Ok(())
}

/// Create a buffer with the given size and access pattern.
#[allow(clippy::too_many_arguments)]
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    logical_size: u64,
    allocation_size: u64,
    access: DataAccess,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<BufferHandle> {
    debug_assert!(logical_size <= allocation_size);
    let allocation_size = if access == DataAccess::Broadcast {
        uniform_buffer_allocation_width(logical_size, allocation_size)
    } else {
        allocation_size
    };
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
            Width: allocation_size,
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
        let hr = unsafe {
            logical_device.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                initial_state,
                None,
                &mut resource,
            )
        };
        if hr.is_err() {
            crate::signal::push_sync_signal(crate::signal::Signal::Oversubscribed {
                reason: crate::signal::OversubscribedReason::BufferHeap,
                size_hint: allocation_size,
            });
        }
        hr.context("Failed to create buffer resource")?;

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
                Width: allocation_size,
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
                stride > 0 && (logical_size as u32).is_multiple_of(stride),
                "buffer logical size {logical_size} not evenly divisible by element stride {stride} — \
                 likely a stride mismatch (set BufferProxy::element_stride or \
                 update element_stride_for_buffer)"
            );
            let num_elements = (logical_size as u32) / stride;

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
            let aligned_size = (logical_size + 255) & !255;

            if aligned_size > allocation_size {
                anyhow::bail!(
                    "uniform buffer CBV size {aligned_size} exceeds allocation {allocation_size}"
                );
            }

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
            size: logical_size,
            allocation_size,
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
            parent_for_view: None,
            view_byte_offset: None,
            is_reserved: false,
            tile_byte_size: 0,
            reserved_tiles: Vec::new(),
        },
    );

    Ok(handle)
}

pub(super) fn create_reserved_with_capacity(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    logical_size: u64,
    capacity: u64,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<BufferHandle> {
    let allocation_size = tiles::align_reserved_cap(capacity.max(logical_size));
    let num_tiles = tiles::num_tiles_for_bytes(allocation_size) as usize;
    let initial_tiles = tiles::tiles_needed_for_logical_size(logical_size) as usize;

    let resource_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: allocation_size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };

    let (resource, reserved_tiles) = {
        let ld = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;
        let pool = ld
            .tile_heap_pool
            .as_mut()
            .context("internal: tile heap pool missing")?;
        let queue = ld.command_queue.clone();

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            ld.device.CreateReservedResource(
                &resource_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
        }
        .context("CreateReservedResource")?;
        let resource = resource.context("CreateReservedResource returned null")?;

        let mut slots: Vec<Option<(ID3D12Heap, u64)>> = vec![None; num_tiles];
        let mut mappings = Vec::with_capacity(initial_tiles);
        for (i, slot) in slots.iter_mut().enumerate().take(initial_tiles) {
            let (heap, off) = pool.alloc_tile(&ld.device)?;
            mappings.push((i as u32, heap.clone(), off));
            *slot = Some((heap, off));
        }

        tiles::map_tiles_batched(&queue, &resource, &mappings)?;
        (resource, slots)
    };

    let handle = state.next_buffer_handle;
    state.next_buffer_handle += 1;

    let stride = element_stride.unwrap_or(4);
    debug_assert!(
        stride > 0 && (logical_size as u32).is_multiple_of(stride),
        "buffer logical size {logical_size} not evenly divisible by element stride {stride}"
    );
    let num_elements = (logical_size as u32) / stride;

    let (bindless_offset, bindless_srv_offset) = {
        let ld = state
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;
        let uav_offset = ld.resource_registry.register_buffer_uav(handle);
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
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
            let mut cpu_handle = ld.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
            cpu_handle.ptr += (uav_offset * ld.cbv_srv_uav_descriptor_size) as usize;
            cpu_handle
        };
        unsafe {
            ld.device
                .CreateUnorderedAccessView(&resource, None, Some(&uav_desc), uav_cpu_handle);
        }

        let srv_offset = ld.resource_registry.register_buffer_srv(handle);
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
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
            let mut cpu_handle = ld.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
            cpu_handle.ptr += (srv_offset * ld.cbv_srv_uav_descriptor_size) as usize;
            cpu_handle
        };
        unsafe {
            ld.device
                .CreateShaderResourceView(&resource, Some(&srv_desc), srv_cpu_handle);
        }
        (Some(uav_offset), Some(srv_offset))
    };

    state.buffers.insert(
        handle,
        BufferState {
            device_handle,
            resource,
            size: logical_size,
            allocation_size,
            bindless_offset,
            bindless_srv_offset,
            is_storage: true,
            upload_buffer: None,
            element_stride,
            is_view: false,
            coherent_readback: None,
            coherent_readback_mapped: None,
            flags,
            transient_placed: false,
            parent_for_view: None,
            view_byte_offset: None,
            is_reserved: true,
            tile_byte_size: tiles::BUFFER_TILE_BYTES,
            reserved_tiles,
        },
    );

    Ok(handle)
}

pub(super) fn create_with_capacity(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    initial_size: u64,
    requested_capacity: u64,
    access: DataAccess,
    element_stride: Option<u32>,
    flags: BufferFlags,
) -> Result<(BufferHandle, u64)> {
    let cap = requested_capacity.max(initial_size);
    let use_reserved = !super::env_disable_reserved_buffers()
        && state.devices.get(&device_handle).is_some_and(|d| {
            d.supports_reserved_buffers
                && d.tile_heap_pool.is_some()
                && cap > initial_size
                && access == DataAccess::Scattered
                && !flags.contains(BufferFlags::CPU_READABLE)
        });
    if use_reserved {
        let h = create_reserved_with_capacity(
            state,
            device_handle,
            initial_size,
            cap,
            element_stride,
            flags,
        )?;
        return Ok((h, capacity(state, h)));
    }
    let h = create(
        state,
        device_handle,
        initial_size,
        cap,
        access,
        element_stride,
        flags,
    )?;
    Ok((h, capacity(state, h)))
}

pub(super) fn capacity(state: &Dx12State, buffer_handle: BufferHandle) -> u64 {
    state
        .buffers
        .get(&buffer_handle)
        .map(|b| b.allocation_size)
        .unwrap_or(0)
}

pub(super) fn set_logical_size(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    buffer_handle: BufferHandle,
    new_logical_size: u64,
) -> Result<()> {
    let old = state
        .buffers
        .get(&buffer_handle)
        .cloned()
        .context("set_logical_size: invalid buffer")?;
    if old.device_handle != device_handle {
        anyhow::bail!("set_logical_size: buffer belongs to a different device");
    }
    if old.is_view {
        anyhow::bail!("set_logical_size: cannot resize a buffer view");
    }
    if old.transient_placed {
        anyhow::bail!("set_logical_size: cannot resize a transient placed buffer");
    }
    if new_logical_size > old.allocation_size {
        anyhow::bail!("logical size exceeds allocation");
    }
    if new_logical_size == 0 {
        anyhow::bail!("buffer size must be non-zero");
    }
    let stride = old.element_stride.unwrap_or(4);
    if old.is_storage && stride > 0 && !(new_logical_size as u32).is_multiple_of(stride) {
        anyhow::bail!(
            "set_logical_size: new size {new_logical_size} not divisible by stride {stride}"
        );
    }
    if !old.is_storage {
        let aligned = (new_logical_size + 255) & !255;
        if aligned > old.allocation_size {
            anyhow::bail!("CBV aligned size exceeds allocation");
        }
    }

    if old.is_reserved {
        let old_pages = tiles::tiles_needed_for_logical_size(old.size);
        let new_pages = tiles::tiles_needed_for_logical_size(new_logical_size);
        {
            let ld = state
                .devices
                .get_mut(&device_handle)
                .context("set_logical_size: device")?;
            let pool = ld
                .tile_heap_pool
                .as_mut()
                .context("set_logical_size: tile heap pool")?;
            let queue = ld.command_queue.clone();
            let buf = state
                .buffers
                .get_mut(&buffer_handle)
                .expect("set_logical_size: buffer");
            if new_pages > old_pages {
                let mut mappings = Vec::with_capacity((new_pages - old_pages) as usize);
                for i in old_pages..new_pages {
                    let (heap, off) = pool.alloc_tile(&ld.device)?;
                    mappings.push((i, heap.clone(), off));
                    buf.reserved_tiles[i as usize] = Some((heap, off));
                }
                tiles::map_tiles_batched(&queue, &buf.resource, &mappings)?;
            } else if new_pages < old_pages {
                let n = old_pages - new_pages;
                tiles::unmap_tile_run(&queue, &buf.resource, new_pages, n)?;
                for i in new_pages..old_pages {
                    if let Some((heap, off)) = buf
                        .reserved_tiles
                        .get_mut(i as usize)
                        .and_then(|s| s.take())
                    {
                        pool.free_tile(&heap, off);
                    }
                }
            }
        }
        let logical_device = state
            .devices
            .get(&device_handle)
            .context("set_logical_size: device")?;
        let buf = state.buffers.get(&buffer_handle).unwrap();
        rewrite_root_buffer_descriptors(logical_device, &buf.resource, new_logical_size, buf)?;
        state.buffers.get_mut(&buffer_handle).unwrap().size = new_logical_size;
        return Ok(());
    }

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("set_logical_size: device")?;
    rewrite_root_buffer_descriptors(logical_device, &old.resource, new_logical_size, &old)?;
    state.buffers.get_mut(&buffer_handle).unwrap().size = new_logical_size;
    Ok(())
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
                    reserved_tiles: if buffer.is_reserved {
                        Some(buffer.reserved_tiles)
                    } else {
                        None
                    },
                },
            );
        }
    }
}

/// Hint unused reserved tiles at/above `offset` (bytes).
pub(super) fn hint_unused_above(state: &mut Dx12State, buffer_handle: BufferHandle, offset: u64) {
    let (device_handle, first_tile) = {
        let Some(buf) = state.buffers.get(&buffer_handle) else {
            return;
        };
        if !buf.is_reserved {
            return;
        }
        let tile = u64::from(buf.tile_byte_size);
        if tile == 0 {
            return;
        }
        let ft = ((offset.saturating_add(tile.saturating_sub(1))) / tile) as usize;
        if ft >= buf.reserved_tiles.len() {
            return;
        }
        (buf.device_handle, ft)
    };
    let (devices, buffers) = (&mut state.devices, &mut state.buffers);
    let Some(ld) = devices.get_mut(&device_handle) else {
        return;
    };
    let Some(pool) = ld.tile_heap_pool.as_mut() else {
        return;
    };
    let queue = &ld.command_queue;
    let Some(buf_mut) = buffers.get_mut(&buffer_handle) else {
        return;
    };
    let mut i = first_tile;
    while i < buf_mut.reserved_tiles.len() {
        while i < buf_mut.reserved_tiles.len() && buf_mut.reserved_tiles[i].is_none() {
            i += 1;
        }
        if i >= buf_mut.reserved_tiles.len() {
            break;
        }
        let run_start = i;
        while i < buf_mut.reserved_tiles.len() && buf_mut.reserved_tiles[i].is_some() {
            i += 1;
        }
        let n = (i - run_start) as u32;
        let _ = tiles::unmap_tile_run(queue, &buf_mut.resource, run_start as u32, n);
        for j in run_start..i {
            if let Some((heap, off)) = buf_mut.reserved_tiles[j].take() {
                pool.free_tile(&heap, off);
            }
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
    if stride == 0 {
        anyhow::bail!("Buffer view element stride must be non-zero");
    }
    if !(size as u32).is_multiple_of(stride) {
        anyhow::bail!("View byte size {size} is not evenly divisible by element stride {stride}");
    }
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
            allocation_size: parent.allocation_size,
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
            parent_for_view: Some(parent_handle),
            view_byte_offset: Some(offset),
            is_reserved: false,
            tile_byte_size: 0,
            reserved_tiles: Vec::new(),
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

/// Size of the per-device zero-filled UPLOAD-heap buffer used as the source for
/// `CopyBufferRegion` clears. Matches `UPLOAD_CHUNK_SIZE` so any single chunk fits.
pub(super) const ZERO_BUFFER_SIZE: u64 = UPLOAD_CHUNK_SIZE;

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

/// Fill buffer region with zeros (standalone, synchronous version).
///
/// For DEFAULT heap buffers (storage), uses CopyBufferRegion from the device's zero buffer.
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
        // DEFAULT heap storage buffer: zero-fill via CopyBufferRegion from the device's
        // zero buffer. Unlike ClearUnorderedAccessViewUint, CopyBufferRegion has no
        // per-device shared-descriptor aliasing hazard.
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
        let cmd_list7: ID3D12GraphicsCommandList7 =
            cmd_list.cast().context("ID3D12GraphicsCommandList7")?;

        let buf_resource = buffer.resource.clone();
        let zero = device.zero_buffer.clone();

        let mut b_to_copy = [barriers::buffer_barrier_full(
            &buf_resource,
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_ACCESS_COMMON,
            D3D12_BARRIER_ACCESS_COPY_DEST,
        )];
        unsafe {
            barriers::barrier_buffers(&cmd_list7, &b_to_copy);
            barriers::drop_buffer_barriers(&mut b_to_copy);
        }

        let mut written = 0u64;
        while written < clear_size {
            let this_chunk = (clear_size - written).min(ZERO_BUFFER_SIZE);
            unsafe {
                cmd_list.CopyBufferRegion(&buf_resource, offset + written, &zero, 0, this_chunk);
            }
            written += this_chunk;
        }

        let mut b_to_common = [barriers::buffer_barrier_full(
            &buf_resource,
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_ACCESS_COPY_DEST,
            D3D12_BARRIER_ACCESS_COMMON,
        )];
        unsafe {
            barriers::barrier_buffers(&cmd_list7, &b_to_common);
            barriers::drop_buffer_barriers(&mut b_to_common);
        }

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
            anyhow::bail!("Device removed during buffer clear: {:?}", removed_reason);
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
