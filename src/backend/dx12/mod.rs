//! DirectX 12 backend implementation.
//!
//! Targets D3D12 Feature Level 12.0+ on Windows.
//! Uses Slang for shader compilation (Slang -> DXIL directly with SM 6.6).
//!
//! ## WARP (software D3D12)
//!
//! Set **`GOLDY_DX12_FORCE_WARP=1`** to run on the DX12 WARP software rasterizer.
//! This registers WARP with DXGI and redirects [`Instance::create_device`](crate::Instance::create_device)
//! to it, even when hardware GPUs are present. Use on headless CI (no GPU) or locally to
//! reproduce WARP-specific rendering bugs.
//!
//! After the first WARP device is created, Goldy logs one stderr line showing which
//! `d3d10warp.dll` was loaded — useful to confirm a side-loaded NuGet build is active.
//!
//! See `docs/src/architecture/backends.md` (DX12 / WARP section) for NuGet side-loading
//! instructions.
//!
//! ## Reserved (tiled) buffers
//!
//! When **`GOLDY_DX12_DISABLE_RESERVED_BUFFERS=1`**, oversize `Buffer::new_with_capacity_hint`
//! uses committed resources instead of reserved resources and tile heap mapping. Device
//! capabilities report `buffer_resize_cost` as `Copy` in that mode. Use this if a driver stack
//! faults during tile mapping; capture a GPU hang dump / enable the D3D12 debug layer first.
//!
//! ## Module Structure
//!
//! - `types`: Internal state structs for devices, buffers, shaders, etc.
//! - `utils`: Format conversion and helpers

mod barriers;
mod buffer;
mod compute;
mod context;
mod diagnostic;
mod tiles;
pub(crate) use diagnostic::log_warp_module_path_once;
mod device;
mod frame_table;
mod pending_submit;
mod pipeline;
mod process_shared;
mod pso_cache;
mod render_commands;
mod render_target;
mod sampler;
mod shader;
mod staging;
mod submit_session;
mod surface;
mod texture;
mod types;
mod utils;

use types::{Dx12State, LogicalDevice};

use super::*;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Adapter ID for the WARP device from `IDXGIFactory4::EnumWarpAdapter`.
/// Used when `GOLDY_DX12_FORCE_WARP=1`.
pub const WARP_ADAPTER_ID: u32 = u32::MAX;

/// Whether `GOLDY_DX12_FORCE_WARP=1` is set.
///
/// Registers WARP with DXGI and redirects [`Instance::create_device`](crate::Instance::create_device)
/// to the WARP adapter regardless of what hardware GPUs are present.
pub(crate) fn env_force_warp() -> bool {
    std::env::var("GOLDY_DX12_FORCE_WARP").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn env_allow_warp() -> bool {
    env_force_warp()
}

/// Reserved (`CreateReservedResource`) buffers can be disabled for troubleshooting.
///
/// Set **`GOLDY_DX12_DISABLE_RESERVED_BUFFERS=1`** to use committed oversize allocations instead of
/// tile heaps + [`UpdateTileMappings`]. In that mode, [`Dx12Backend::device_capabilities`] reports
/// `buffer_resize_cost` as [`crate::types::BufferResizeCost::Copy`] (not `PageBind`).
pub(crate) fn env_disable_reserved_buffers() -> bool {
    std::env::var("GOLDY_DX12_DISABLE_RESERVED_BUFFERS")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

/// The D3D12 debug layer raises SEH exception 0x87D when it detects API
/// violations. Without a handler, the default behaviour terminates the process
/// with exit code 2173 (= 0x87D). This filter catches the exception so that
/// the debug layer message is surfaced through the info queue instead.
pub(super) fn install_debug_layer_exception_handler() {
    const D3D12_DEBUG_LAYER_EXCEPTION: u32 = 0x87D;

    static HANDLER_INIT: std::sync::Once = std::sync::Once::new();
    HANDLER_INIT.call_once(|| {
        extern "system" {
            fn SetUnhandledExceptionFilter(
                filter: Option<unsafe extern "system" fn(*mut std::ffi::c_void) -> i32>,
            ) -> Option<unsafe extern "system" fn(*mut std::ffi::c_void) -> i32>;
        }

        unsafe extern "system" fn d3d12_exception_filter(info: *mut std::ffi::c_void) -> i32 {
            #[repr(C)]
            struct ExceptionRecord {
                exception_code: u32,
                _rest: [usize; 5],
            }
            #[repr(C)]
            struct ExceptionPointers {
                exception_record: *mut ExceptionRecord,
                _context_record: *mut std::ffi::c_void,
            }
            let ptrs = info as *const ExceptionPointers;
            let code = if !ptrs.is_null() && !(*ptrs).exception_record.is_null() {
                (*(*ptrs).exception_record).exception_code
            } else {
                0
            };
            if code == D3D12_DEBUG_LAYER_EXCEPTION {
                -1 // EXCEPTION_CONTINUE_EXECUTION
            } else {
                0 // EXCEPTION_CONTINUE_SEARCH
            }
        }
        unsafe {
            SetUnhandledExceptionFilter(Some(d3d12_exception_filter));
        }
    });
}

/// True when the D3D12 debug layer will be enabled (debug build or GOLDY_DX12_DEBUG=1).
pub fn is_debug_mode() -> bool {
    let no_debug = std::env::var("GOLDY_DX12_NO_DEBUG").is_ok_and(|v| v == "1" || v == "true");
    !no_debug && (cfg!(debug_assertions) || std::env::var("GOLDY_DX12_DEBUG").is_ok_and(|v| v == "1" || v == "true"))
}

/// GPU-Based Validation: opt-in via `GOLDY_DX12_GBV=1` (requires the debug layer).
/// GBV is intentionally off by default — it can crash or hang inside `ResizeBuffers` and other
/// presentation paths when combined with the debug layer.
pub(crate) fn env_enable_gbv() -> bool {
    std::env::var("GOLDY_DX12_GBV").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// DRED auto-breadcrumbs + page-fault capture: on by default in [`is_debug_mode`]; opt out with
/// `GOLDY_DX12_NO_DRED=1`, or force in release with `GOLDY_DX12_DRED=1` (requires the debug layer).
pub(crate) fn env_enable_dred() -> bool {
    if std::env::var("GOLDY_DX12_NO_DRED").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")) {
        return false;
    }
    is_debug_mode()
        || std::env::var("GOLDY_DX12_DRED").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Get or create a DX12 backend for one [`crate::Instance`].
///
/// Each instance owns independent `Dx12State` (resource tables, contexts, devices) so
/// lock-free submit sessions never share mutable backend state across concurrent clients.
/// DXGI factory + adapter enumeration are process-wide via `process_shared::process_shared`.
pub fn shared_backend() -> anyhow::Result<Arc<Mutex<Box<dyn super::GpuBackend>>>> {
    let backend = Dx12Backend::new()?;
    Ok(Arc::new(Mutex::new(Box::new(backend) as Box<dyn super::GpuBackend>)))
}

/// DirectX 12 backend.
pub struct Dx12Backend {
    state: Dx12State,
}

impl Dx12Backend {
    /// Create a new DX12 backend.
    pub fn new() -> Result<Self> {
        tracing::info!("Initializing DX12 backend");

        let shared = process_shared::process_shared()?;
        install_debug_layer_exception_handler();

        // Create Slang compiler
        let slang_compiler = crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

        let state = Dx12State {
            factory: shared.factory.clone(),
            allow_tearing: shared.allow_tearing,
            adapters: shared.adapters.clone(),
            devices: HashMap::new(),
            next_device_handle: 1,
            contexts: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            next_context_id: 1,
            context_fences: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            buffers: std::sync::Arc::new(std::sync::RwLock::new(types::BufferTable::new())),
            shaders: std::sync::Arc::new(std::sync::RwLock::new(types::ShaderTable::new())),
            pipelines: std::sync::Arc::new(std::sync::RwLock::new(types::PipelineTable::new())),
            compute_pipelines: std::sync::Arc::new(std::sync::RwLock::new(types::ComputePipelineTable::new())),
            render_targets: std::sync::Arc::new(std::sync::RwLock::new(types::RenderTargetTable::new())),
            surfaces: HashMap::new(),
            next_surface_handle: 1,
            textures: std::sync::Arc::new(std::sync::RwLock::new(types::TextureTable::new())),
            samplers: std::sync::Arc::new(std::sync::RwLock::new(types::SamplerTable::new())),
            next_rtv_offset: 0,
            free_rtv_offsets: Vec::new(),
            next_dsv_offset: 0,
            free_dsv_offsets: Vec::new(),
            slang_compiler,
            device_removed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            frame_tables: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            pending_present_finishes: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        };

        Ok(Self { state })
    }

    /// Wait for the GPU to finish all work on a device (sync fence path).
    fn wait_for_gpu(&self, device: &LogicalDevice) -> Result<()> {
        let fence_value = device.timeline_next.load(std::sync::atomic::Ordering::Relaxed);
        unsafe { device.command_queue.Signal(&device.fence, fence_value) }.context("Failed to signal fence")?;
        utils::wait_for_fence(&device.fence, fence_value)
    }
}

impl Dx12Backend {
    fn destroy_device_inner(&mut self, device_handle: DeviceHandle) {
        if let Some(logical_device) = self.state.devices.remove(&device_handle) {
            let _ = logical_device.submission_worker.flush();
            let _ = self.wait_for_gpu(&logical_device);
            // Advance timeline_next past the value consumed by wait_for_gpu so that
            // flush_deletion_queue's per-buffer Signal calls use fresh, strictly-increasing
            // fence values. Without this, PendingDeletion::Buffer re-signals the same value
            // that wait_for_gpu already used, violating D3D12's monotonic fence requirement
            // and causing an abnormal process exit (exit code 2173) on teardown.
            logical_device
                .timeline_next
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            logical_device.flush_deletion_queue();

            let pso_cache = logical_device.pso_cache.read().unwrap();
            if pso_cache.dirty {
                if let Some(cache_root) = dirs::cache_dir() {
                    let path = cache_root
                        .join("goldy")
                        .join(format!("dx12_pso_{}.bin", logical_device.adapter_id));
                    if let Err(e) = pso_cache::save_maps(&path, &pso_cache.graphics_blobs, &pso_cache.compute_blobs) {
                        tracing::warn!(
                            error = ?e,
                            path = ?path,
                            "failed to save DX12 PSO disk cache"
                        );
                    }
                }
            }

            let buffer_handles: Vec<_> = self
                .state
                .buffers
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, b)| b.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in buffer_handles {
                self.state.buffers.write().unwrap().entries.remove(&handle);
            }

            let shader_handles: Vec<_> = self
                .state
                .shaders
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in shader_handles {
                self.state.shaders.write().unwrap().entries.remove(&handle);
            }

            let pipeline_handles: Vec<_> = self
                .state
                .pipelines
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, p)| p.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in pipeline_handles {
                self.state.pipelines.write().unwrap().entries.remove(&handle);
            }

            let target_handles: Vec<_> = self
                .state
                .render_targets
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in target_handles {
                self.state.render_targets.write().unwrap().entries.remove(&handle);
            }

            let surface_handles: Vec<_> = self
                .state
                .surfaces
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in surface_handles {
                self.state.surfaces.remove(&handle);
            }

            let texture_handles: Vec<_> = self
                .state
                .textures
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in texture_handles {
                self.state.textures.write().unwrap().entries.remove(&handle);
            }

            let sampler_handles: Vec<_> = self
                .state
                .samplers
                .read()
                .unwrap()
                .entries
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in sampler_handles {
                self.state.samplers.write().unwrap().entries.remove(&handle);
            }

            tracing::info!("Destroyed DX12 device {}", device_handle);
        }
    }
}

impl Drop for Dx12Backend {
    fn drop(&mut self) {
        tracing::info!("Shutting down DX12 backend");

        let device_handles: Vec<_> = self.state.devices.keys().copied().collect();
        for handle in device_handles {
            self.destroy_device_inner(handle);
        }
    }
}

#[cfg(all(feature = "dx12", target_os = "windows"))]
fn slot_access_from_push_constant_slot_kinds(
    kinds: &[Option<crate::types::BindlessSlotKind>],
) -> Vec<Option<crate::types::ResourceAccess>> {
    use crate::types::{BindlessSlotKind, ResourceAccess};
    kinds
        .iter()
        .map(|kind| match kind {
            Some(BindlessSlotKind::StorageUav) => Some(ResourceAccess::ReadWrite),
            Some(BindlessSlotKind::ReadOnlySrv) => Some(ResourceAccess::Read),
            Some(BindlessSlotKind::UniformCbv) | None => None,
        })
        .collect()
}

impl crate::backend::GpuBackendTimelineWait for Dx12Backend {
    fn take_timeline_submission_epoch_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<crate::backend::submission_worker::SubmissionEpochWait>> {
        if self.gpu_progress(ctx) >= value {
            return Ok(None);
        }
        let device_handle = context::context_device(&self.state, ctx);
        let Some(ld) = self.state.devices.get(&device_handle) else {
            return Ok(None);
        };
        let horizon = crate::backend::submission_worker::submission_horizon(&ld.timeline_next);
        if value == 0 || value > horizon {
            return Ok(None);
        }
        Ok(Some(crate::backend::submission_worker::SubmissionEpochWait::new(
            std::sync::Arc::clone(&ld.submission_worker),
            value,
            horizon,
        )))
    }

    fn take_timeline_blocking_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn crate::backend::TimelineBlockingWait>>> {
        if self.gpu_progress(ctx) >= value {
            return Ok(None);
        }
        let fence = self
            .state
            .context_fences
            .read()
            .unwrap()
            .get(&ctx)
            .context("Invalid context handle")?
            .1
            .clone();
        Ok(Some(Box::new(Dx12TimelineBlockingWait { fence, value })))
    }

    fn check_submission_worker_for_context(&self, ctx: ContextHandle) -> Result<()> {
        let device_handle = self.context_device(ctx);
        if let Some(ld) = self.state.devices.get(&device_handle) {
            ld.submission_worker.check_error()
        } else {
            Ok(())
        }
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        let device_handle = self.context_device(ctx);
        if let Some(ld) = self.state.devices.get(&device_handle) {
            ld.submission_worker.flush()?;
        }
        let fence = self
            .state
            .context_fences
            .read()
            .unwrap()
            .get(&ctx)
            .context("Invalid context handle")?
            .1
            .clone();
        // Detect TDR: device removal signals all fences with u64::MAX.
        let completed = unsafe { fence.GetCompletedValue() };
        if completed == u64::MAX {
            self.state
                .device_removed
                .store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(ld) = self.state.devices.get(&device_handle) {
                let reason = unsafe { ld.device.GetDeviceRemovedReason() };
                anyhow::bail!("GPU device removed (TDR): {:?}", reason);
            }
            anyhow::bail!("GPU device removed (TDR)");
        }
        let retired = context::device_retired(&self.state, device_handle);
        if let Some(ld) = self.state.devices.get(&device_handle) {
            if let Some(sc_arc) = self.state.contexts.read().unwrap().get(&ctx).cloned() {
                let mut sc = sc_arc.lock().unwrap();
                context::drain_context_deletion_queue_up_to(ld, &mut sc, completed);
                context::drain_pending_gpu_profiles_up_to(ld, &mut sc, completed);
            }
            ld.process_deletion_queue_up_to(value.min(retired));
            let descriptors_arc = std::sync::Arc::clone(&ld.descriptors);
            let fences = self.state.context_fences.read().unwrap();
            descriptors_arc.lock().unwrap().drain_ready_slot_reclamations(&fences);
        }
        Ok(())
    }
}

impl crate::backend::GpuBackendPresentSplit for Dx12Backend {
    fn take_present_gpu_work(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn crate::backend::PresentGpuWork>> {
        surface::prepare_present_work(&mut self.state, frame, submit_tv)
    }

    fn finish_present(
        &mut self,
        finish: crate::backend::PresentFinishState,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        surface::finish_present(&mut self.state, finish, submit_tv)
    }

    fn schedules_present_on_submit_worker(&self) -> bool {
        true
    }

    fn schedule_present_on_submission_worker(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::backend::SchedulePresentOnWorkerResult> {
        surface::schedule_present_on_submission_worker(&mut self.state, frame, submit_tv)
    }

    fn take_scheduled_present_blocking_wait(
        &self,
        frame: FrameToken,
        present_tv: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn crate::backend::ScheduledPresentBlockingWait>>> {
        surface::take_scheduled_present_blocking_wait(&self.state, frame, present_tv)
    }

    fn apply_scheduled_present_bookkeeping(
        &mut self,
        outcome: crate::backend::ScheduledPresentWaitOutcome,
    ) -> Result<()> {
        surface::apply_scheduled_present_bookkeeping(&mut self.state, outcome)
    }

    fn supports_lazy_present_finish(&self) -> bool {
        true
    }
}

impl GpuBackend for Dx12Backend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Dx12
    }

    fn enumerate_adapters(&self) -> Vec<super::AdapterInfo> {
        device::enumerate(&self.state.adapters)
    }

    fn adapter_capabilities(&self, adapter_id: u32) -> crate::device::DeviceCapabilities {
        device::adapter_capabilities(&self.state.adapters, adapter_id)
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        device::create(&mut self.state, adapter_id)
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        let ctxs: Vec<ContextHandle> = self
            .state
            .contexts
            .read()
            .unwrap()
            .iter()
            .filter(|(_, sc_arc)| sc_arc.lock().unwrap().device == device_handle)
            .map(|(k, _)| *k)
            .collect();
        for ctx in ctxs {
            context::destroy(&mut self.state, ctx);
        }
        self.destroy_device_inner(device_handle);
    }

    fn device_wait_idle(&mut self, device_handle: DeviceHandle) -> Result<()> {
        let logical_device = self
            .state
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;
        logical_device.submission_worker.flush()?;
        self.wait_for_gpu(logical_device)
    }

    fn create_context(&mut self, device: DeviceHandle) -> Result<ContextHandle> {
        context::create(&mut self.state, device)
    }

    fn destroy_context(&mut self, ctx: ContextHandle) {
        context::destroy(&mut self.state, ctx);
    }

    fn clone_context_timeline_reader(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextTimelineReader>> {
        Some(std::sync::Arc::new(Dx12ContextTimelineReader {
            sc: std::sync::Arc::clone(self.state.contexts.read().unwrap().get(&ctx)?),
        }))
    }

    fn clone_device_timeline_reader(
        &self,
        device: DeviceHandle,
    ) -> Option<std::sync::Arc<dyn crate::backend::DeviceTimelineReader>> {
        Some(std::sync::Arc::new(Dx12DeviceTimelineReader {
            ld: std::sync::Arc::clone(self.state.devices.get(&device)?),
        }))
    }

    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
        _context_readers: std::sync::Arc<
            std::sync::Mutex<
                std::collections::HashMap<ContextHandle, std::sync::Arc<dyn crate::backend::ContextTimelineReader>>,
            >,
        >,
    ) -> Option<std::sync::Arc<dyn crate::backend::ContextDeferredDeletionFlush>> {
        let device_handle = self.context_device(ctx);
        Some(std::sync::Arc::new(Dx12ContextDeferredDeletionFlush {
            ctx,
            sc: std::sync::Arc::clone(self.state.contexts.read().unwrap().get(&ctx)?),
            ld: std::sync::Arc::clone(self.state.devices.get(&device_handle)?),
            context_fences: std::sync::Arc::clone(&self.state.context_fences),
        }))
    }

    fn context_device(&self, ctx: ContextHandle) -> DeviceHandle {
        context::context_device(&self.state, ctx)
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.state.devices.contains_key(&device)
    }

    fn is_device_lost(&self, _device: DeviceHandle) -> bool {
        self.state.device_removed.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn create_buffer(
        &mut self,
        device_handle: DeviceHandle,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
    ) -> Result<BufferHandle> {
        buffer::create(
            &mut self.state,
            device_handle,
            size,
            size,
            access,
            element_stride,
            flags,
        )
    }

    fn destroy_buffer(&mut self, buffer_handle: BufferHandle) {
        buffer::destroy(&mut self.state, buffer_handle);
    }

    fn write_buffer(&mut self, buffer_handle: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        buffer::write(&mut self.state, buffer_handle, offset, data)
    }

    fn buffer_size(&self, buffer_handle: BufferHandle) -> u64 {
        buffer::size(&self.state, buffer_handle)
    }

    fn buffer_capacity(&self, buffer_handle: BufferHandle) -> u64 {
        buffer::capacity(&self.state, buffer_handle)
    }

    fn create_buffer_with_capacity(
        &mut self,
        device_handle: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        access: crate::backend::BufferKind,
        element_stride: Option<u32>,
        flags: crate::types::BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        buffer::create_with_capacity(
            &mut self.state,
            device_handle,
            initial_size,
            capacity,
            access,
            element_stride,
            flags,
        )
    }

    fn set_buffer_logical_size(
        &mut self,
        device_handle: DeviceHandle,
        buffer_handle: BufferHandle,
        new_logical_size: u64,
    ) -> Result<()> {
        buffer::set_logical_size(&mut self.state, device_handle, buffer_handle, new_logical_size)
    }

    fn hint_buffer_unused_above(&mut self, buffer_handle: BufferHandle, offset: u64) {
        buffer::hint_unused_above(&mut self.state, buffer_handle, offset);
    }

    fn buffer_bindless_index(&self, buffer_handle: BufferHandle) -> Option<u32> {
        buffer::bindless_index(&self.state, buffer_handle)
    }

    fn buffer_bindless_srv_index(&self, buffer_handle: BufferHandle) -> Option<u32> {
        buffer::bindless_srv_index(&self.state, buffer_handle)
    }

    fn create_buffer_view(
        &mut self,
        parent: BufferHandle,
        offset: u64,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        buffer::create_view(&mut self.state, parent, offset, size, element_stride)
    }

    fn resize_buffer(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        new_size: u64,
        preserve_contents: bool,
    ) -> Result<()> {
        buffer::resize(&mut self.state, device, buffer, new_size, preserve_contents)
    }

    fn read_buffer_to_cpu(&mut self, device: DeviceHandle, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        buffer::read_to_cpu(&mut self.state, device, buffer, output)
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        buffer::alloc_readback_buffer(&mut self.state, device, size)
    }

    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        buffer::read_readback_buffer(&self.state.buffers.read().unwrap().entries, buffer, output)
    }

    fn free_readback_buffer(&mut self, buffer: BufferHandle) {
        buffer::destroy(&mut self.state, buffer);
    }

    fn query_texture_copy_footprint(
        &self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
    ) -> Result<crate::backend::TextureCopyFootprint> {
        texture::query_texture_copy_footprint(&self.state, device, width, height, format)
    }

    fn alloc_texture_readback_staging(
        &mut self,
        device: DeviceHandle,
        layout: crate::backend::TextureCopyFootprint,
    ) -> Result<BufferHandle> {
        buffer::alloc_texture_readback_staging(&mut self.state, device, layout)
    }

    fn read_texture_readback_staging(
        &self,
        buffer: BufferHandle,
        layout: crate::backend::TextureCopyFootprint,
        output: &mut [u8],
    ) -> Result<()> {
        buffer::read_texture_readback_staging(&self.state.buffers.read().unwrap().entries, buffer, layout, output)
    }

    fn device_capabilities(&self, device: DeviceHandle) -> crate::device::DeviceCapabilities {
        let adapter_id = self.state.devices.get(&device).map(|d| d.adapter_id).unwrap_or(0);
        self.adapter_capabilities(adapter_id)
    }

    fn clear_buffer(&mut self, device: DeviceHandle, buffer: BufferHandle, offset: u64, size: u64) -> Result<()> {
        buffer::clear(&mut self.state, device, buffer, offset, size)
    }

    fn create_shader_with_paths(
        &mut self,
        device_handle: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
    ) -> Result<ShaderHandle> {
        self.create_shader_with_checks(
            device_handle,
            slang_source,
            search_paths,
            defines,
            optimization_level,
            vec![],
        )
    }

    fn create_shader_with_checks(
        &mut self,
        device_handle: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
        layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
    ) -> Result<ShaderHandle> {
        shader::create_with_checks(
            &mut self.state,
            crate::backend::shared::ShaderDesc::new(
                device_handle,
                slang_source,
                search_paths,
                defines,
                optimization_level,
            )
            .with_layout_checks(layout_checks),
        )
    }

    fn destroy_shader(&mut self, shader_handle: ShaderHandle) {
        shader::destroy(&mut self.state, shader_handle);
    }

    fn create_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        let raster = crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format);
        let desc = crate::backend::shared::GraphicsPipelineCreateDesc {
            device_handle,
            vertex_shader,
            fragment_shader,
            raster: &raster,
        };
        pipeline::create(&mut self.state, &desc)
    }
    fn destroy_pipeline(&mut self, pipeline_handle: PipelineHandle) {
        pipeline::destroy(&mut self.state, pipeline_handle);
    }

    fn create_render_target(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<RenderTargetHandle> {
        render_target::create(&mut self.state, device_handle, width, height, format)
    }
    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        render_target::destroy(&mut self.state, target);
    }

    fn render_to_target(
        &mut self,
        device_handle: DeviceHandle,
        target: RenderTargetHandle,
        commands: &[RenderCommand],
    ) -> Result<()> {
        render_target::render(&mut self.state, device_handle, target, commands)
    }
    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()> {
        render_target::read_to_cpu(&mut self.state, target, output)
    }
    fn create_surface(
        &mut self,
        device_handle: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
        depth_format: Option<crate::types::DepthFormat>,
    ) -> Result<SurfaceHandle> {
        surface::create(&mut self.state, device_handle, window, display, depth_format)
    }
    fn destroy_surface(&mut self, surface_handle: SurfaceHandle) {
        surface::destroy(&mut self.state, surface_handle);
    }

    fn set_surface_acquire_abort(
        &mut self,
        surface_handle: SurfaceHandle,
        abort: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) {
        if let Some(surface) = self.state.surfaces.get_mut(&surface_handle) {
            surface.acquire_abort = abort;
        }
    }

    fn begin_frame(
        &mut self,
        surface_handle: SurfaceHandle,
        ctx: ContextHandle,
    ) -> Result<(FrameToken, TextureHandle)> {
        let (image, present_slot) = surface::acquire(&mut self.state, surface_handle, ctx)?;
        let tex = surface::frame_texture(&self.state, surface_handle)
            .context("begin_frame: surface frame texture unavailable")?;
        Ok((
            FrameToken {
                surface: surface_handle,
                image,
                context: ctx,
                frame_slot: image as u32,
                present_slot,
            },
            tex,
        ))
    }

    fn cancel_frame(&mut self, frame: FrameToken) -> Result<()> {
        surface::cancel_frame(&mut self.state, frame)
    }

    fn record_render(&mut self, frame: &FrameToken, commands: &[RenderCommand]) -> Result<()> {
        surface::render(
            &mut self.state,
            frame.surface,
            frame.image,
            frame.present_slot,
            commands,
        )
    }

    fn surface_resize(&mut self, surface_handle: SurfaceHandle, width: u32, height: u32) -> Result<()> {
        surface::resize(&mut self.state, surface_handle, width, height)
    }
    fn surface_size(&self, surface_handle: SurfaceHandle) -> (u32, u32) {
        surface::size(&self.state, surface_handle)
    }

    fn surface_format(&self, surface_handle: SurfaceHandle) -> TextureFormat {
        surface::format(&self.state, surface_handle)
    }

    fn surface_set_present_mode(
        &mut self,
        surface_handle: SurfaceHandle,
        mode: crate::types::PresentMode,
    ) -> Result<()> {
        surface::set_present_mode(&mut self.state, surface_handle, mode)
    }

    fn surface_present_mode(&self, surface_handle: SurfaceHandle) -> crate::types::PresentMode {
        surface::get_present_mode(&self.state, surface_handle)
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        let contexts = self.state.contexts.read().unwrap();
        match contexts.get(&ctx) {
            Some(sc_arc) => unsafe { sc_arc.lock().unwrap().fence.GetCompletedValue() },
            None => 0,
        }
    }

    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        context::device_retired(&self.state, device)
    }

    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> anyhow::Result<()> {
        if context::device_retired(&self.state, device) >= value {
            return Ok(());
        }
        if let Some(ld) = self.state.devices.get(&device) {
            ld.submission_worker.flush()?;
            let horizon = crate::backend::submission_worker::submission_horizon(&ld.timeline_next);
            if value <= horizon {
                ld.submission_worker.wait_submitted(value)?;
            }
        }
        let fence = self
            .state
            .contexts
            .read()
            .unwrap()
            .values()
            .filter_map(|sc_arc| {
                let sc = sc_arc.lock().unwrap();
                if sc.device == device && sc.last_submitted_seq >= value {
                    Some(sc.fence.clone())
                } else {
                    None
                }
            })
            .next();
        if let Some(fence) = fence {
            utils::wait_for_fence(&fence, value)?;
        } else if context::device_retired(&self.state, device) < value {
            // Present copies and other device-fence work may not map to a context fence.
            if let Some(ld) = self.state.devices.get(&device) {
                utils::wait_for_fence(&ld.fence, value)?;
            }
        }
        Ok(())
    }

    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::Signal> {
        let device_handle = self.context_device(ctx);
        // Drain any present jobs the submission worker has already executed. This is
        // what lets `supports_lazy_present_finish` skip the blocking wait in
        // `Scheme::grant_present` consumption: the correctness-critical allocator-reuse
        // fence is stamped eagerly at enqueue time (see `schedule_present_on_submission_worker`),
        // so this drain only needs to keep the capacity-only bookkeeping moving.
        surface::drain_pending_present_finishes(&mut self.state);
        // Present copy/signal uses the device sync fence; per-context `progress` alone
        // can lag behind `return_fence` values stored in pending_swapchain_returns.
        let swapchain_retire_progress = progress.max(context::device_retired(&self.state, device_handle));
        let signal_queue = self
            .state
            .contexts
            .read()
            .unwrap()
            .get(&ctx)
            .map(|sc_arc| std::sync::Arc::clone(&sc_arc.lock().unwrap().signal_queue));
        let Some(signal_queue) = signal_queue else {
            return Vec::new();
        };
        for surface in self.state.surfaces.values_mut() {
            if surface.device_handle != device_handle {
                continue;
            }
            surface.pending_swapchain_returns.retain(|r| {
                if swapchain_retire_progress >= r.return_fence {
                    signal_queue.push(crate::signal::Signal::SwapchainReturned {
                        image_index: r.image_index,
                    });
                    surface.pending_acquire_count = surface.pending_acquire_count.saturating_sub(1);
                    false
                } else {
                    true
                }
            });
        }
        crate::signal::drain_all_signals(&signal_queue)
    }

    fn peek_oldest_in_flight(&self, ctx: ContextHandle) -> Option<crate::timeline::TimelineValue> {
        let contexts = self.state.contexts.read().unwrap();
        let sc_arc = contexts.get(&ctx)?;
        let sc = sc_arc.lock().unwrap();
        let progress = unsafe { sc.fence.GetCompletedValue() };
        if progress < sc.last_submitted_seq {
            Some(progress.saturating_add(1))
        } else {
            None
        }
    }

    fn pending_acquire_count(&self, surface_handle: SurfaceHandle) -> u32 {
        self.state
            .surfaces
            .get(&surface_handle)
            .map(|s| s.pending_acquire_count)
            .unwrap_or(0)
    }

    fn peek_oldest_pending_swapchain_return(&self, surface: SurfaceHandle) -> Option<crate::timeline::TimelineValue> {
        self.state
            .surfaces
            .get(&surface)?
            .pending_swapchain_returns
            .iter()
            .map(|r| r.return_fence)
            .min()
    }

    fn take_swapchain_return_blocking_wait(
        &self,
        surface: SurfaceHandle,
        _ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn crate::backend::TimelineBlockingWait>>> {
        let device = self
            .state
            .surfaces
            .get(&surface)
            .context("Invalid surface handle")?
            .device_handle;
        let ld = self
            .state
            .devices
            .get(&device)
            .context("Surface's device is invalid")?
            .clone();
        let device_sync = unsafe { ld.fence.GetCompletedValue() };
        if device_sync >= value {
            return Ok(None);
        }
        let issue_token = self
            .state
            .surfaces
            .get(&surface)
            .and_then(|s| {
                s.pending_swapchain_returns
                    .iter()
                    .find(|r| r.return_fence == value)
                    .and_then(|r| r.issue_token.clone())
            });
        Ok(Some(Box::new(Dx12SwapchainReturnBlockingWait {
            ld,
            value,
            issue_token,
        })))
    }

    fn wait_until_timeout(
        &mut self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
        timeout_ms: u32,
    ) -> Result<bool> {
        let device_handle = self.context_device(ctx);
        if let Some(ld) = self.state.devices.get(&device_handle) {
            let horizon = crate::backend::submission_worker::submission_horizon(&ld.timeline_next);
            if !ld
                .submission_worker
                .wait_submitted_if_scheduled_timeout(value, horizon, timeout_ms)?
            {
                return Ok(false);
            }
            ld.submission_worker.check_error()?;
        }
        let fence = self
            .state
            .context_fences
            .read()
            .unwrap()
            .get(&ctx)
            .context("Invalid context handle")?
            .1
            .clone();
        let ok = utils::wait_for_fence_timeout(&fence, value, timeout_ms)?;
        if ok {
            // Detect TDR: device removal signals all fences with u64::MAX.
            let completed = unsafe { fence.GetCompletedValue() };
            if completed == u64::MAX {
                self.state
                    .device_removed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                if let Some(ld) = self.state.devices.get(&device_handle) {
                    let reason = unsafe { ld.device.GetDeviceRemovedReason() };
                    anyhow::bail!("GPU device removed (TDR): {:?}", reason);
                }
                anyhow::bail!("GPU device removed (TDR)");
            }
            let retired = context::device_retired(&self.state, device_handle);
            if let Some(dev) = self.state.devices.get(&device_handle) {
                if let Some(sc_arc) = self.state.contexts.read().unwrap().get(&ctx).cloned() {
                    let mut sc = sc_arc.lock().unwrap();
                    context::drain_context_deletion_queue_up_to(dev, &mut sc, completed);
                    context::drain_pending_gpu_profiles_up_to(dev, &mut sc, completed);
                }
                dev.process_deletion_queue_up_to(value.min(retired));
                let descriptors_arc = std::sync::Arc::clone(&dev.descriptors);
                let fences = self.state.context_fences.read().unwrap();
                descriptors_arc.lock().unwrap().drain_ready_slot_reclamations(&fences);
            }
        }
        Ok(ok)
    }

    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        compute::submit(&mut self.state, ctx, commands, sync)
    }

    fn submit_graph(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        compute::submit_graph(&mut self.state, ctx, commands, None, sync)
    }

    fn submit_graph_and_retain(
        &mut self,
        ctx: ContextHandle,
        commands: &[GraphCommand],
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        compute::evict_retained(&self.state, ctx, key);
        compute::submit_graph(&mut self.state, ctx, commands, Some(key), sync)
    }

    fn try_resubmit_retained(
        &mut self,
        ctx: ContextHandle,
        key: u64,
        sync: Option<&SubmitSync>,
    ) -> Result<Option<crate::timeline::TimelineValue>> {
        compute::try_resubmit_retained(&mut self.state, ctx, key, sync)
    }

    fn evict_retained(&mut self, ctx: ContextHandle, key: u64) {
        compute::evict_retained(&self.state, ctx, key);
    }

    fn record_gpu_work(&mut self, frame: &FrameToken, commands: &[GpuCommand]) -> Result<()> {
        surface::record_gpu_work(&mut self.state, frame.surface, commands)
    }

    fn submit_frame(&mut self, frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        surface::submit_frame(&mut self.state, frame)
    }

    fn present_frame(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        let work = self.take_present_gpu_work(frame, submit_tv)?;
        let finish = work.run()?;
        self.finish_present(finish, submit_tv)
    }

    fn create_pipeline_with_depth(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        depth_stencil: Option<&crate::types::DepthStencilState>,
    ) -> Result<PipelineHandle> {
        let raster = crate::backend::shared::PipelineDesc::new(vertex_layout, topology, target_format)
            .with_depth_stencil(depth_stencil);
        let desc = crate::backend::shared::GraphicsPipelineCreateDesc {
            device_handle,
            vertex_shader,
            fragment_shader,
            raster: &raster,
        };
        pipeline::create_with_depth(&mut self.state, &desc)
    }

    fn create_render_target_with_depth(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<crate::types::DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        render_target::create_with_depth(
            &mut self.state,
            device_handle,
            width,
            height,
            color_format,
            depth_format,
        )
    }
    fn create_texture(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<TextureHandle> {
        texture::create(&mut self.state, device_handle, width, height, format, access, flags)
    }

    fn write_texture(&mut self, texture_handle: TextureHandle, data: &[u8], width: u32, height: u32) -> Result<()> {
        texture::write(&mut self.state, texture_handle, data, width, height)
    }

    fn write_texture_region(
        &mut self,
        texture_handle: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<()> {
        texture::write_region(&mut self.state, texture_handle, x, y, width, height, data)
    }

    fn destroy_texture(&mut self, texture_handle: TextureHandle) {
        texture::destroy(&mut self.state, texture_handle);
    }

    fn read_texture_to_cpu(&mut self, texture_handle: TextureHandle, output: &mut [u8]) -> Result<()> {
        texture::read_to_cpu(&mut self.state, texture_handle, output)
    }

    fn texture_bindless_index(&self, texture_handle: TextureHandle) -> Option<u32> {
        texture::bindless_index(&self.state, texture_handle)
    }

    fn texture_bindless_sampled_index(&self, texture_handle: TextureHandle) -> Option<u32> {
        texture::bindless_sampled_index(&self.state, texture_handle)
    }

    fn create_sampler(
        &mut self,
        device_handle: DeviceHandle,
        desc: &crate::types::SamplerDesc,
    ) -> Result<SamplerHandle> {
        sampler::create(&mut self.state, device_handle, desc)
    }

    fn destroy_sampler(&mut self, sampler_handle: SamplerHandle) {
        sampler::destroy(&mut self.state, sampler_handle);
    }

    fn sampler_bindless_index(&self, sampler_handle: SamplerHandle) -> Option<u32> {
        sampler::bindless_index(&self.state, sampler_handle)
    }

    fn create_compute_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        compute_shader: ShaderHandle,
    ) -> Result<ComputePipelineHandle> {
        let handle = compute::create(&mut self.state, device_handle, compute_shader)?;

        let (cats, slot_kinds, strides) = self
            .state
            .shaders
            .read()
            .unwrap()
            .entries
            .get(&compute_shader)
            .and_then(|s| s.reflection.as_ref())
            .map(|r| {
                (
                    r.push_constant_categories.clone(),
                    r.push_constant_slot_kinds.clone(),
                    r.binding_element_strides.clone(),
                )
            })
            .unwrap_or_default();

        {
            let mut compute_pipelines_write = self.state.compute_pipelines.write().unwrap();
            if let Some(ps) = compute_pipelines_write.entries.get_mut(&handle) {
                ps.push_constant_categories = cats;
                ps.push_constant_slot_kinds = slot_kinds;
                ps.binding_element_strides = strides;
            }
        }
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline_handle: ComputePipelineHandle) {
        compute::destroy(&mut self.state, pipeline_handle);
    }

    fn compute_pipeline_slot_access(
        &self,
        pipeline: ComputePipelineHandle,
    ) -> Vec<Option<crate::types::ResourceAccess>> {
        let compute_pipelines_read = self.state.compute_pipelines.read().unwrap();
        let Some(ps) = compute_pipelines_read.entries.get(&pipeline) else {
            return Vec::new();
        };
        slot_access_from_push_constant_slot_kinds(&ps.push_constant_slot_kinds)
    }

    fn render_pipeline_slot_access(&self, pipeline: PipelineHandle) -> Vec<Option<crate::types::ResourceAccess>> {
        let pipelines_read = self.state.pipelines.read().unwrap();
        let Some(ps) = pipelines_read.entries.get(&pipeline) else {
            return Vec::new();
        };
        slot_access_from_push_constant_slot_kinds(&ps.push_constant_slot_kinds)
    }

    fn reset_buffer_heaps(&mut self, device_handle: DeviceHandle) {
        for sc_arc in self.state.contexts.read().unwrap().values() {
            let mut sc = sc_arc.lock().unwrap();
            if sc.device == device_handle {
                sc.staging_belt.trim();
            }
        }
    }

    fn available_bindless_slots(&self, device_handle: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        self.state
            .devices
            .get(&device_handle)
            .map(|ld| {
                ld.descriptors
                    .lock()
                    .unwrap()
                    .resource_registry
                    .available_slots(category)
            })
            .unwrap_or(0)
    }

    fn max_bindless_slots_per_category(
        &self,
        _device_handle: DeviceHandle,
        category: crate::types::ResourceCategory,
    ) -> u32 {
        types::ResourceRegistry::max_slots(category)
    }

    fn flush_deferred_deletions(&mut self, ctx: ContextHandle) {
        let device_handle = self.context_device(ctx);
        let retired = context::device_retired(&self.state, device_handle);
        if let Some(ld) = self.state.devices.get(&device_handle) {
            ld.process_deletion_queue_up_to(retired);
            let descriptors_arc = std::sync::Arc::clone(&ld.descriptors);
            let fences = self.state.context_fences.read().unwrap();
            descriptors_arc.lock().unwrap().drain_ready_slot_reclamations(&fences);
        }
    }

    fn deferred_deletion_pending_count(&self, ctx: ContextHandle) -> usize {
        let device_handle = self.context_device(ctx);
        self.state
            .devices
            .get(&device_handle)
            .map(|d| d.deletion_queue.lock().unwrap().pending_len())
            .unwrap_or(0)
    }
}

impl crate::backend::GpuBackendSubmitSession for Dx12Backend {
    fn clone_context_submit_session(
        &self,
        ctx: ContextHandle,
        _backend: std::sync::Arc<std::sync::Mutex<Box<dyn crate::backend::GpuBackend>>>,
    ) -> std::sync::Arc<dyn crate::backend::ContextSubmitSession> {
        submit_session::Dx12SubmitSession::clone_from_state(&self.state, ctx)
            .unwrap_or_else(|e| panic!("clone_context_submit_session({ctx}): {e:#}"))
    }
}

struct Dx12TimelineBlockingWait {
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    value: crate::timeline::TimelineValue,
}

/// Blocks until the present-copy job for a flip-model return fence has issued and retired.
struct Dx12SwapchainReturnBlockingWait {
    ld: types::SharedLogicalDevice,
    value: crate::timeline::TimelineValue,
    issue_token: Option<std::sync::Arc<crate::backend::submission_worker::SubmissionJobToken>>,
}

impl crate::backend::TimelineBlockingWait for Dx12SwapchainReturnBlockingWait {
    fn block(self: Box<Self>) -> Result<()> {
        let _tz = crate::tracy_zone!("goldy.dx12.Dx12SwapchainReturnBlockingWait.block");
        if let Some(token) = self.issue_token {
            if !token.is_done() {
                let _tz_issue = crate::tracy_zone!("goldy.dx12.Dx12SwapchainReturnBlockingWait.issue_token");
                token.wait()?;
                self.ld.submission_worker.check_error()?;
            }
        }
        let _tz_fence = crate::tracy_zone!("goldy.dx12.Dx12SwapchainReturnBlockingWait.wait_for_fence");
        utils::wait_for_fence(&self.ld.fence, self.value)?;
        Ok(())
    }
}

struct Dx12ContextTimelineReader {
    sc: types::SharedSubmissionContext,
}

impl crate::backend::ContextTimelineReader for Dx12ContextTimelineReader {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        unsafe { self.sc.lock().unwrap().fence.GetCompletedValue() }
    }

    fn peek_oldest_in_flight(&self) -> Option<crate::timeline::TimelineValue> {
        let sc = self.sc.lock().unwrap();
        let progress = unsafe { sc.fence.GetCompletedValue() };
        if progress < sc.last_submitted_seq {
            Some(progress.saturating_add(1))
        } else {
            None
        }
    }
}

impl crate::backend::TimelineBlockingWait for Dx12TimelineBlockingWait {
    fn block(self: Box<Self>) -> Result<()> {
        utils::wait_for_fence(&self.fence, self.value)
    }

    fn block_timeout(self: Box<Self>, timeout_ms: u32) -> Result<bool> {
        utils::wait_for_fence_timeout(&self.fence, self.value, timeout_ms)
    }
}

struct Dx12DeviceTimelineReader {
    ld: types::SharedLogicalDevice,
}

impl crate::backend::DeviceTimelineReader for Dx12DeviceTimelineReader {
    fn device_horizon(&self) -> crate::timeline::TimelineValue {
        use std::sync::atomic::Ordering;
        let floor = self.ld.retired_floor.load(Ordering::Relaxed);
        // Device sync fence shares the per-context timeline value space (`timeline_next`).
        let device_sync = unsafe { self.ld.fence.GetCompletedValue() };
        floor.max(device_sync)
    }
}

struct Dx12ContextDeferredDeletionFlush {
    ctx: ContextHandle,
    sc: types::SharedSubmissionContext,
    ld: types::SharedLogicalDevice,
    context_fences: std::sync::Arc<
        std::sync::RwLock<
            std::collections::HashMap<ContextHandle, (DeviceHandle, windows::Win32::Graphics::Direct3D12::ID3D12Fence)>,
        >,
    >,
}

impl crate::backend::ContextDeferredDeletionFlush for Dx12ContextDeferredDeletionFlush {
    fn flush(&self, device_retired: crate::timeline::TimelineValue) {
        let completed = self
            .context_fences
            .read()
            .unwrap()
            .get(&self.ctx)
            .map(|(_, fence)| unsafe { fence.GetCompletedValue() })
            .unwrap_or(0);
        let ctx_batch: Vec<_> = self.sc.lock().unwrap().deletion_queue.drain_up_to_completed(completed);
        let descriptors_arc = std::sync::Arc::clone(&self.ld.descriptors);
        let fences = self.context_fences.read().unwrap();
        {
            let mut registry = descriptors_arc.lock().unwrap();
            for r in ctx_batch {
                types::destroy_pending_deletion(&self.ld, &mut registry, r);
            }
            registry.drain_ready_slot_reclamations(&fences);
        }
        self.ld.process_deletion_queue_up_to(device_retired);
    }
}
