//! DX12 utility functions.
//!
//! Format conversions and helpers.

use super::types::LogicalDevice;
use anyhow::{Context, Result};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use windows::Win32::{
    Foundation::{CloseHandle, WAIT_OBJECT_0},
    Graphics::Direct3D12::{ID3D12CommandList, ID3D12CommandQueue, ID3D12Fence},
    System::Threading::{CreateEventA, WaitForSingleObject, INFINITE},
};

use crate::types::{
    AddressMode, CompareFunction, DepthFormat, FilterMode, IndexFormat, PrimitiveTopology, TextureFormat, VertexFormat,
};
use windows::Win32::Graphics::{Direct3D, Direct3D12, Dxgi};

/// Convert Goldy TextureFormat to DXGI format.
pub fn format_to_dxgi(format: TextureFormat) -> Dxgi::Common::DXGI_FORMAT {
    match format {
        TextureFormat::R8Unorm => Dxgi::Common::DXGI_FORMAT_R8_UNORM,
        TextureFormat::Rg8Unorm => Dxgi::Common::DXGI_FORMAT_R8G8_UNORM,
        TextureFormat::Rgba8UnormSrgb => Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        TextureFormat::Rgba8Unorm => Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        TextureFormat::Bgra8Unorm => Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
        TextureFormat::Rgba16Float => Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
        TextureFormat::Rgba32Float => Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT,
    }
}

/// Convert DXGI format to Goldy TextureFormat.
pub fn dxgi_to_format(format: Dxgi::Common::DXGI_FORMAT) -> Option<TextureFormat> {
    match format {
        Dxgi::Common::DXGI_FORMAT_R8_UNORM => Some(TextureFormat::R8Unorm),
        Dxgi::Common::DXGI_FORMAT_R8G8_UNORM => Some(TextureFormat::Rg8Unorm),
        Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => Some(TextureFormat::Rgba8UnormSrgb),
        Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
        Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => Some(TextureFormat::Bgra8UnormSrgb),
        Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
        Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT => Some(TextureFormat::Rgba16Float),
        Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT => Some(TextureFormat::Rgba32Float),
        _ => None,
    }
}

/// Convert Goldy VertexFormat to DXGI format.
pub fn vertex_format_to_dxgi(format: VertexFormat) -> Dxgi::Common::DXGI_FORMAT {
    match format {
        VertexFormat::Float32 => Dxgi::Common::DXGI_FORMAT_R32_FLOAT,
        VertexFormat::Float32x2 => Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT,
        VertexFormat::Float32x3 => Dxgi::Common::DXGI_FORMAT_R32G32B32_FLOAT,
        VertexFormat::Float32x4 => Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT,
        VertexFormat::Uint32 => Dxgi::Common::DXGI_FORMAT_R32_UINT,
        VertexFormat::Sint32 => Dxgi::Common::DXGI_FORMAT_R32_SINT,
        VertexFormat::Uint8x4 => Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UINT,
        VertexFormat::Unorm8x4 => Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
    }
}

/// Convert Goldy PrimitiveTopology to D3D12 topology.
pub fn topology_to_d3d12(topology: PrimitiveTopology) -> Direct3D::D3D_PRIMITIVE_TOPOLOGY {
    match topology {
        PrimitiveTopology::PointList => Direct3D::D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        PrimitiveTopology::LineList => Direct3D::D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        PrimitiveTopology::LineStrip => Direct3D::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        PrimitiveTopology::TriangleList => Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        PrimitiveTopology::TriangleStrip => Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    }
}

/// Convert Goldy PrimitiveTopology to D3D12 topology type for PSO.
pub fn topology_type_to_d3d12(topology: PrimitiveTopology) -> Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    match topology {
        PrimitiveTopology::PointList => Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
        PrimitiveTopology::LineList | PrimitiveTopology::LineStrip => Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
        PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip => {
            Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE
        }
    }
}

/// Convert Goldy IndexFormat to DXGI index format.
pub fn index_format_to_dxgi(format: IndexFormat) -> Dxgi::Common::DXGI_FORMAT {
    match format {
        IndexFormat::Uint16 => Dxgi::Common::DXGI_FORMAT_R16_UINT,
        IndexFormat::Uint32 => Dxgi::Common::DXGI_FORMAT_R32_UINT,
    }
}

/// Map vendor ID to vendor name.
pub fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x1002 | 0x1022 => "AMD",
        0x10DE => "NVIDIA",
        0x8086 => "Intel",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x106B => "Apple",
        0x1414 => "Microsoft", // For WARP adapter
        _ => "Unknown",
    }
}

/// Map DXGI adapter flags to Goldy DeviceType.
pub fn device_type_from_flags(flags: Dxgi::DXGI_ADAPTER_FLAG) -> crate::types::DeviceType {
    if flags.contains(Dxgi::DXGI_ADAPTER_FLAG_SOFTWARE) {
        crate::types::DeviceType::Cpu
    } else {
        // Can't easily distinguish discrete vs integrated from DXGI alone
        // Default to discrete for hardware adapters
        crate::types::DeviceType::DiscreteGpu
    }
}

/// Convert Goldy DepthFormat to DXGI format.
pub fn depth_format_to_dxgi(format: DepthFormat) -> Dxgi::Common::DXGI_FORMAT {
    match format {
        DepthFormat::Depth16Unorm => Dxgi::Common::DXGI_FORMAT_D16_UNORM,
        DepthFormat::Depth24Plus => Dxgi::Common::DXGI_FORMAT_D32_FLOAT,
        DepthFormat::Depth24PlusStencil8 => Dxgi::Common::DXGI_FORMAT_D24_UNORM_S8_UINT,
        DepthFormat::Depth32Float => Dxgi::Common::DXGI_FORMAT_D32_FLOAT,
        DepthFormat::Depth32FloatStencil8 => Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    }
}

/// Convert Goldy CompareFunction to D3D12 comparison func.
pub fn compare_to_d3d12(compare: CompareFunction) -> Direct3D12::D3D12_COMPARISON_FUNC {
    match compare {
        CompareFunction::Never => Direct3D12::D3D12_COMPARISON_FUNC_NEVER,
        CompareFunction::Less => Direct3D12::D3D12_COMPARISON_FUNC_LESS,
        CompareFunction::Equal => Direct3D12::D3D12_COMPARISON_FUNC_EQUAL,
        CompareFunction::LessEqual => Direct3D12::D3D12_COMPARISON_FUNC_LESS_EQUAL,
        CompareFunction::Greater => Direct3D12::D3D12_COMPARISON_FUNC_GREATER,
        CompareFunction::NotEqual => Direct3D12::D3D12_COMPARISON_FUNC_NOT_EQUAL,
        CompareFunction::GreaterEqual => Direct3D12::D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        CompareFunction::Always => Direct3D12::D3D12_COMPARISON_FUNC_ALWAYS,
    }
}

/// Convert Goldy FilterMode to D3D12 filter.
pub fn filter_to_d3d12(min: FilterMode, mag: FilterMode, mip: FilterMode) -> Direct3D12::D3D12_FILTER {
    match (min, mag, mip) {
        (FilterMode::Nearest, FilterMode::Nearest, FilterMode::Nearest) => Direct3D12::D3D12_FILTER_MIN_MAG_MIP_POINT,
        (FilterMode::Nearest, FilterMode::Nearest, FilterMode::Linear) => {
            Direct3D12::D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR
        }
        (FilterMode::Nearest, FilterMode::Linear, FilterMode::Nearest) => {
            Direct3D12::D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT
        }
        (FilterMode::Nearest, FilterMode::Linear, FilterMode::Linear) => {
            Direct3D12::D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR
        }
        (FilterMode::Linear, FilterMode::Nearest, FilterMode::Nearest) => {
            Direct3D12::D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT
        }
        (FilterMode::Linear, FilterMode::Nearest, FilterMode::Linear) => {
            Direct3D12::D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR
        }
        (FilterMode::Linear, FilterMode::Linear, FilterMode::Nearest) => {
            Direct3D12::D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT
        }
        (FilterMode::Linear, FilterMode::Linear, FilterMode::Linear) => Direct3D12::D3D12_FILTER_MIN_MAG_MIP_LINEAR,
    }
}

/// Convert Goldy AddressMode to D3D12 texture address mode.
pub fn address_mode_to_d3d12(mode: AddressMode) -> Direct3D12::D3D12_TEXTURE_ADDRESS_MODE {
    match mode {
        AddressMode::ClampToEdge => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressMode::Repeat => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressMode::MirrorRepeat => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
    }
}

/// Execute command lists and signal the device fence under [`LogicalDevice::queue_lock`].
///
/// Reserves the timeline value with `fetch_add` inside the lock so `(value, execute, signal)`
/// is atomic relative to other queue submits.
pub(super) fn execute_command_lists_and_signal_device(
    logical_device: &LogicalDevice,
    command_lists: &[Option<ID3D12CommandList>],
) -> Result<u64> {
    execute_with_waits_and_signal_device(logical_device, &[], command_lists)
}

/// Enqueue cross-queue GPU waits, execute command lists, and signal the device fence — all
/// under a single hold of [`LogicalDevice::queue_lock`].
///
/// Used by the present copy path: frame compute runs on a per-context queue, but the
/// scratch→backbuffer copy must run on the swapchain-associated device queue. A
/// `queue.Wait(ctx_fence, submit_tv)` here orders the copy after the context submission
/// (mirrors Vulkan's `wait_compute_done` in `present_split.rs`).
pub(super) fn execute_with_waits_and_signal_device(
    logical_device: &LogicalDevice,
    waits: &[(ID3D12Fence, u64)],
    command_lists: &[Option<ID3D12CommandList>],
) -> Result<u64> {
    let queue_lock = Arc::clone(&logical_device.queue_lock);
    let _guard = queue_lock.lock().unwrap();
    let api_log_on = super::api_log::enabled();
    let queue_id = api_log_on
        .then(|| super::api_log::com_identity(&logical_device.command_queue))
        .unwrap_or(0);
    for (producer_fence, value) in waits {
        if api_log_on {
            super::api_log::log_queue_wait(queue_id, super::api_log::com_identity(producer_fence), *value);
        }
        unsafe { logical_device.command_queue.Wait(producer_fence, *value) }
            .context("device queue Wait before present copy")?;
    }
    let fence_value = logical_device.timeline_next.fetch_add(1, Ordering::Relaxed);
    if api_log_on {
        super::api_log::log_execute_command_lists(queue_id, command_lists.len());
    }
    unsafe {
        logical_device.command_queue.ExecuteCommandLists(command_lists);
    }
    unsafe { logical_device.command_queue.Signal(&logical_device.fence, fence_value) }
        .context("Failed to signal device fence")?;
    logical_device
        .device_last_submitted_seq
        .store(fence_value, Ordering::Relaxed);
    if api_log_on {
        super::api_log::log_queue_signal(
            queue_id,
            super::api_log::com_identity(&logical_device.fence),
            fence_value,
        );
    }
    Ok(fence_value)
}

/// Run `f` while holding [`LogicalDevice::queue_lock`] (the queue is externally synchronized).
pub(super) fn with_queue_lock<R>(logical_device: &LogicalDevice, f: impl FnOnce() -> R) -> R {
    let queue_lock = Arc::clone(&logical_device.queue_lock);
    let _guard = queue_lock.lock().unwrap();
    f()
}

/// Enqueue cross-context GPU waits, execute command lists, and signal a context fence — all
/// under a single hold of `queue_lock`, on `queue`.
///
/// `queue` and `queue_lock` are the *calling context's own* command queue and its per-context
/// queue lock (`Dx12SubmissionContext::command_queue`/`queue_lock`) — never the shared
/// `LogicalDevice::command_queue`. Each context owns its queue so that a cross-context `Wait`
/// enqueued here only ever stalls this context's queue: it can never block a sibling
/// context's submission, and a sibling context's submission can never be blocked behind this
/// one's `Wait` on the (now nonexistent) shared queue. `logical_device` is only used for the
/// shared timeline-value counter (see the comment below), never for its own queue.
///
/// The queue itself is externally synchronized in D3D12: `Wait`, `ExecuteCommandLists`, and
/// `Signal` are all queue mutations that must not interleave with another thread's submission
/// on the *same* queue. Splitting the `Wait` out of the lock (the previous behavior) both
/// races the queue object and can reorder a same-context resubmit behind this submission's
/// wait, deadlocking or removing the device under parallel submit.
pub(super) fn execute_with_waits_and_signal_context(
    logical_device: &LogicalDevice,
    queue: &ID3D12CommandQueue,
    queue_lock: &Arc<Mutex<()>>,
    waits: &[(ID3D12Fence, u64)],
    ctx_fence: &ID3D12Fence,
    command_lists: &[Option<ID3D12CommandList>],
) -> Result<u64> {
    let _guard = queue_lock.lock().unwrap();
    let api_log_on = super::api_log::enabled();
    let queue_id = api_log_on.then(|| super::api_log::com_identity(queue)).unwrap_or(0);
    for (producer_fence, value) in waits {
        if api_log_on {
            super::api_log::log_queue_wait(queue_id, super::api_log::com_identity(producer_fence), *value);
        }
        unsafe { queue.Wait(producer_fence, *value) }.context("cross-submit GPU queue Wait")?;
    }
    // Values are drawn from the device-shared counter (not a per-context one): the public
    // `Device::wait_until_retired` contract requires an arbitrary context's `TimelineValue` to
    // be findable/comparable across every other context on the device (see
    // `Dx12Backend::device_wait_until`), which only holds if all contexts share one space.
    let fence_value = logical_device.timeline_next.fetch_add(1, Ordering::Relaxed);
    if api_log_on {
        super::api_log::log_execute_command_lists(queue_id, command_lists.len());
    }
    unsafe {
        queue.ExecuteCommandLists(command_lists);
    }
    unsafe { queue.Signal(ctx_fence, fence_value) }.context("Failed to signal context fence")?;
    if api_log_on {
        super::api_log::log_queue_signal(queue_id, super::api_log::com_identity(ctx_fence), fence_value);
    }
    Ok(fence_value)
}

/// Wait for a fence to reach the specified value.
/// This is a low-level helper for GPU synchronization.
pub(super) fn wait_for_fence(fence: &ID3D12Fence, value: u64) -> Result<()> {
    wait_for_fence_on_device(fence, value, None)
}

/// Like [`wait_for_fence`] but logs DRED on first `u64::MAX` when `ld` is provided.
pub(super) fn wait_for_fence_on_device(
    fence: &ID3D12Fence,
    value: u64,
    ld: Option<&super::types::LogicalDevice>,
) -> Result<()> {
    let completed = unsafe { fence.GetCompletedValue() };
    if completed == u64::MAX {
        if let Some(ld) = ld {
            super::diagnostic::first_touch_device_removed(
                &ld.device,
                &ld.device_removed,
                "dx12::utils::wait_for_fence_on_device",
                value,
                completed,
            );
        }
        anyhow::bail!("GPU device removed while waiting for fence value {value}");
    }
    if completed < value {
        let event = unsafe { CreateEventA(None, false, false, None) }.context("Failed to create event")?;

        unsafe { fence.SetEventOnCompletion(value, event) }.context("Failed to set event on completion")?;

        unsafe { WaitForSingleObject(event, INFINITE) };
        unsafe { CloseHandle(event) }.ok();
    }
    let completed_after = unsafe { fence.GetCompletedValue() };
    if completed_after == u64::MAX {
        if let Some(ld) = ld {
            super::diagnostic::first_touch_device_removed(
                &ld.device,
                &ld.device_removed,
                "dx12::utils::wait_for_fence_on_device",
                value,
                completed_after,
            );
        }
        anyhow::bail!("GPU device removed after waiting for fence value {value}");
    }
    Ok(())
}

/// Wait for a fence with timeout. Returns true if signaled, false if timeout elapsed.
pub(super) fn wait_for_fence_timeout(fence: &ID3D12Fence, value: u64, timeout_ms: u32) -> Result<bool> {
    if unsafe { fence.GetCompletedValue() } >= value {
        return Ok(true);
    }
    let event = unsafe { CreateEventA(None, false, false, None) }.context("Failed to create event")?;

    unsafe { fence.SetEventOnCompletion(value, event) }.context("Failed to set event on completion")?;

    let result = unsafe { WaitForSingleObject(event, timeout_ms) };
    unsafe { CloseHandle(event) }.ok();

    // WAIT_OBJECT_0 when signaled, WAIT_TIMEOUT when timeout elapses
    Ok(result == WAIT_OBJECT_0)
}
