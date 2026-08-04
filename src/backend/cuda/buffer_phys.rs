//! Late physicalization of CUDA buffers for the DX12 companion path.
//!
//! Pool acquire reserves a stable [`BufferHandle`] (and bindless slot) without
//! choosing a physical backing. The first scheme submit / immediate write that
//! declares usage materializes memory:
//!
//! - **Shared** — D3D12 `HEAP_FLAG_SHARED` imported into CUDA. Used when the buffer
//!   is vertex-fed (deposit / host write / transfer) and never kernel-written.
//!   IA binds the D3D12 resource directly; no twin DtoD.
//! - **Native** — ordinary `cuMemAlloc` for compute-only buffers.
//! - **NativeAndTwin** — native SoT + shareable twin for IA when kernels also write.
//!
//! Requirements are a monotonic union. Promotion (e.g. Shared → NativeAndTwin when
//! a kernel later appears) migrates bytes, keeps the handle/slot identity, and
//! evicts retained CUDA entries that pinned the old memory Arc.

#![cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]

use super::dx12_interop::create_shared_buffer_backing;
use super::CudaBackend;
use crate::backend::BufferHandle;
use anyhow::{bail, Context as _, Result};
use bitflags::bitflags;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

bitflags! {
    /// Monotonic usage evidence collected from schemes / immediate APIs.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CudaBufferReq: u8 {
        /// Bound to a compute launch (kernel may read/write).
        const KERNEL = 1 << 0;
        /// Bound as a graphics IA buffer (vertex or index; DX12 GPU VA).
        const VERTEX = 1 << 1;
        /// Host `write_buffer` / pending init bytes.
        const HOST_WRITE = 1 << 2;
        /// Participates in CopyBuffer / ClearBuffer.
        const TRANSFER = 1 << 3;
        /// Bound as a graphics bindless shader resource (SRV/UAV/CBV).
        const SHADER = 1 << 4;
    }
}

/// Physical backing chosen for a logical CUDA buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaPhysKind {
    /// Handle reserved; no device memory yet.
    Deferred,
    /// Native CUDA allocation only.
    Native,
    /// Imported D3D12 shared buffer is the sole CUDA-visible memory.
    Shared,
    /// Native allocation + DX12 twin for IA.
    NativeAndTwin,
}

impl CudaPhysKind {
    pub fn is_deferred(self) -> bool {
        matches!(self, Self::Deferred)
    }

    /// True when `memory` wraps an external import (must not `cuMemFree`).
    #[allow(dead_code)]
    pub fn memory_is_external(self) -> bool {
        matches!(self, Self::Shared)
    }

    /// True when IA should DtoD-refresh a twin from native before draw.
    #[allow(dead_code)]
    pub fn needs_twin_refresh(self) -> bool {
        matches!(self, Self::NativeAndTwin)
    }
}

/// Pick a physical kind from the current requirement union.
fn choose_kind(req: CudaBufferReq) -> CudaPhysKind {
    let kernel = req.contains(CudaBufferReq::KERNEL);
    // VERTEX and SHADER both need a DX12-visible backing for the companion.
    let dx12_visible = req.contains(CudaBufferReq::VERTEX) || req.contains(CudaBufferReq::SHADER);
    match (kernel, dx12_visible) {
        (true, true) => CudaPhysKind::NativeAndTwin,
        (true, false) => CudaPhysKind::Native,
        (false, true) => CudaPhysKind::Shared,
        // Host/transfer-only (e.g. deposit Copy before the first VERTEX bind) lands
        // provisional Native; VERTEX/SHADER without KERNEL promotes to Shared.
        (false, false) => CudaPhysKind::Native,
    }
}

/// Leak a `CudaSlice` that aliases external memory so Drop does not `cuMemFree`.
pub(super) fn leak_shared_buffer_slice(slice: CudaSlice<u8>) {
    // cudarc: leak releases ownership without freeing the device pointer.
    let _ = slice.leak();
}

impl CudaBackend {
    /// Merge `req` into the buffer and materialize / promote as needed.
    pub(super) fn ensure_buffer_requirements(&mut self, buffer: BufferHandle, req: CudaBufferReq) -> Result<()> {
        if req.is_empty() {
            return Ok(());
        }
        // Views share the parent allocation — materialize the parent, then refresh the view.
        if let Some(parent) = self.buffers.get(&buffer).and_then(|b| b.parent) {
            self.fold_view_pending_into_parent(buffer)?;
            self.ensure_buffer_requirements(parent, req)?;
            self.sync_view_from_parent(buffer, parent)?;
            // Mirror requirements on the view for diagnostics / later ensure short-circuit.
            if let Some(buf) = self.buffers.get_mut(&buffer) {
                buf.requirements |= req;
            }
            return Ok(());
        }
        let (_device, old_req, old_kind, capacity) = {
            let buf = self
                .buffers
                .get(&buffer)
                .context("CUDA: ensure_buffer_requirements: invalid buffer")?;
            (buf.device, buf.requirements, buf.phys_kind, buf.capacity)
        };
        let new_req = old_req | req;
        let target = choose_kind(new_req);
        if new_req == old_req && !old_kind.is_deferred() && compatible(old_kind, target) {
            return Ok(());
        }

        self.buffers.get_mut(&buffer).unwrap().requirements = new_req;

        match old_kind {
            CudaPhysKind::Deferred => self.materialize_deferred(buffer, target)?,
            CudaPhysKind::Native if target == CudaPhysKind::NativeAndTwin => {
                self.attach_twin(buffer)?;
                self.buffers.get_mut(&buffer).unwrap().phys_kind = CudaPhysKind::NativeAndTwin;
                self.graph_stats.buffer_promotions.fetch_add(1, Ordering::Relaxed);
            }
            CudaPhysKind::Native if target == CudaPhysKind::Shared => {
                self.promote_native_to_shared(buffer, capacity)?;
            }
            CudaPhysKind::Shared if target == CudaPhysKind::NativeAndTwin => {
                self.promote_shared_to_native_and_twin(buffer, capacity)?;
            }
            CudaPhysKind::Native | CudaPhysKind::Shared | CudaPhysKind::NativeAndTwin
                if compatible(old_kind, target) => {}
            other => {
                bail!("CUDA: unsupported buffer promotion {other:?} → {target:?} (req={new_req:?})");
            }
        }
        Ok(())
    }

    /// Move a view's staged host bytes into the parent's pending_init (absolute offset).
    fn fold_view_pending_into_parent(&mut self, view: BufferHandle) -> Result<()> {
        let (parent, abs_off, pending, view_size) = {
            let v = self
                .buffers
                .get_mut(&view)
                .context("CUDA: fold pending: invalid view")?;
            let parent = v.parent.context("CUDA: fold pending: not a view")?;
            (parent, v.offset, v.pending_init.take(), v.size)
        };
        let Some(data) = pending else {
            return Ok(());
        };
        let parent_buf = self
            .buffers
            .get_mut(&parent)
            .context("CUDA: fold pending: invalid parent")?;
        if parent_buf.phys_kind.is_deferred() {
            let mut host = parent_buf
                .pending_init
                .take()
                .unwrap_or_else(|| vec![0u8; parent_buf.size as usize]);
            if host.len() < parent_buf.size as usize {
                host.resize(parent_buf.size as usize, 0);
            }
            let start = abs_off as usize;
            let n = (data.len() as u64).min(view_size) as usize;
            if start + n > host.len() {
                bail!(
                    "CUDA: view pending write [{start}, {}) exceeds parent pending len {}",
                    start + n,
                    host.len()
                );
            }
            host[start..start + n].copy_from_slice(&data[..n]);
            parent_buf.pending_init = Some(host);
            parent_buf.requirements |= CudaBufferReq::HOST_WRITE;
            parent_buf.bump_content_epoch();
            Ok(())
        } else {
            // Parent already physical — sync the view then write through it.
            self.sync_view_from_parent(view, parent)?;
            let n = (data.len() as u64).min(view_size) as usize;
            self.write_buffer_physical(view, 0, &data[..n])
        }
    }

    /// Copy physical backing pointers from parent onto a view (offset/size unchanged).
    fn sync_view_from_parent(&mut self, view: BufferHandle, parent: BufferHandle) -> Result<()> {
        let parent_meta = self
            .buffers
            .get(&parent)
            .context("CUDA: sync view: invalid parent")?
            .clone_meta();
        let view_buf = self.buffers.get_mut(&view).context("CUDA: sync view: invalid view")?;
        view_buf.memory = parent_meta.memory;
        view_buf.shared = parent_meta.shared;
        view_buf.shared_epoch = parent_meta.shared_epoch;
        view_buf.phys_kind = parent_meta.phys_kind;
        view_buf.memory_is_external = parent_meta.memory_is_external;
        view_buf.content_epoch = parent_meta.content_epoch;
        Ok(())
    }

    fn materialize_deferred(&mut self, buffer: BufferHandle, target: CudaPhysKind) -> Result<()> {
        let (device, capacity, pending, size) = {
            let buf = self.buffers.get_mut(&buffer).unwrap();
            (buf.device, buf.capacity, buf.pending_init.take(), buf.size)
        };
        match target {
            CudaPhysKind::Native | CudaPhysKind::NativeAndTwin => {
                let gpu = self.device(device)?;
                let memory = Arc::new(Mutex::new(
                    gpu.alloc_stream
                        .alloc_zeros::<u8>(capacity as usize)
                        .context("CUDA: materialize native buffer")?,
                ));
                {
                    let buf = self.buffers.get_mut(&buffer).unwrap();
                    buf.memory = Some(memory);
                    buf.phys_kind = if target == CudaPhysKind::NativeAndTwin {
                        CudaPhysKind::NativeAndTwin
                    } else {
                        CudaPhysKind::Native
                    };
                    buf.memory_is_external = false;
                }
                if target == CudaPhysKind::NativeAndTwin {
                    self.attach_twin(buffer)?;
                }
            }
            CudaPhysKind::Shared => {
                self.materialize_shared_primary(buffer, device, capacity)?;
            }
            CudaPhysKind::Deferred => unreachable!(),
        }
        if let Some(data) = pending {
            let n = (data.len() as u64).min(size);
            if n > 0 {
                self.write_buffer_physical(buffer, 0, &data[..n as usize])?;
            }
        }
        self.graph_stats.buffer_materializations.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(buffer, ?target, "CUDA: materialized deferred buffer");
        Ok(())
    }

    fn materialize_shared_primary(
        &mut self,
        buffer: BufferHandle,
        device: crate::backend::DeviceHandle,
        capacity: u64,
    ) -> Result<()> {
        let companion = Arc::clone(
            self.device(device)?
                .dx12
                .as_ref()
                .context("CUDA/DX12: shared buffer requires companion")?,
        );
        let cuda_ctx = Arc::clone(&self.device(device)?.ctx);
        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let backing = create_shared_buffer_backing(&companion, &cuda_ctx, &stream, capacity)?;
        // SAFETY: mapped external alloc; freed only via SharedBufferBacking drop after leak.
        let slice = unsafe { stream.upgrade_device_ptr::<u8>(backing.import.device_ptr, capacity.max(4) as usize) };
        let buf = self.buffers.get_mut(&buffer).unwrap();
        buf.memory = Some(Arc::new(Mutex::new(slice)));
        buf.shared = Some(Arc::new(backing));
        buf.shared_epoch = buf.content_epoch;
        buf.phys_kind = CudaPhysKind::Shared;
        buf.memory_is_external = true;
        self.register_buffer_bindless_descriptor(buffer)?;
        Ok(())
    }

    fn attach_twin(&mut self, buffer: BufferHandle) -> Result<()> {
        let (device, capacity, has_twin) = {
            let buf = self.buffers.get(&buffer).unwrap();
            (buf.device, buf.capacity, buf.shared.is_some())
        };
        if has_twin {
            return Ok(());
        }
        let companion = Arc::clone(
            self.device(device)?
                .dx12
                .as_ref()
                .context("CUDA/DX12: twin requires companion")?,
        );
        let cuda_ctx = Arc::clone(&self.device(device)?.ctx);
        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let backing = create_shared_buffer_backing(&companion, &cuda_ctx, &stream, capacity)?;
        let buf = self.buffers.get_mut(&buffer).unwrap();
        buf.shared = Some(Arc::new(backing));
        buf.shared_epoch = u64::MAX; // force refresh
        self.register_buffer_bindless_descriptor(buffer)?;
        Ok(())
    }

    /// Native → Shared: import becomes sole CUDA memory (deposit-then-IA without kernels).
    fn promote_native_to_shared(&mut self, buffer: BufferHandle, capacity: u64) -> Result<()> {
        let (device, old_memory, size, offset) = {
            let buf = self.buffers.get(&buffer).unwrap();
            (
                buf.device,
                buf.memory
                    .as_ref()
                    .context("CUDA: promote Native→Shared without memory")?
                    .clone(),
                buf.size,
                buf.offset,
            )
        };

        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let target_device_ptr = {
            let guard = old_memory.lock().unwrap();
            let (ptr, _) = guard.device_ptr(&stream);
            ptr + offset
        };
        self.evict_retained_touching_memory(&old_memory, &stream, target_device_ptr, false);

        let companion = Arc::clone(
            self.device(device)?
                .dx12
                .as_ref()
                .context("CUDA/DX12: Shared promote requires companion")?,
        );
        let cuda_ctx = Arc::clone(&self.device(device)?.ctx);
        let backing = create_shared_buffer_backing(&companion, &cuda_ctx, &stream, capacity)?;

        if size > 0 {
            let nbytes = size as usize;
            let src_ptr = {
                let guard = old_memory.lock().unwrap();
                let view = guard
                    .try_slice(offset as usize..(offset as usize + nbytes))
                    .context("CUDA: Native→Shared src view")?;
                let (ptr, _) = view.device_ptr(&stream);
                ptr
            };
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    backing.import.device_ptr,
                    src_ptr,
                    nbytes,
                    stream.cu_stream(),
                )
            }
            .context("CUDA: Native→Shared DtoD")?;
            stream.synchronize().context("CUDA: Native→Shared synchronize")?;
        }

        // Drop native allocation (not external).
        drop(old_memory);

        // SAFETY: mapped external alloc; freed only via SharedBufferBacking drop after leak.
        let slice = unsafe { stream.upgrade_device_ptr::<u8>(backing.import.device_ptr, capacity.max(4) as usize) };
        let buf = self.buffers.get_mut(&buffer).unwrap();
        buf.memory = Some(Arc::new(Mutex::new(slice)));
        buf.shared = Some(Arc::new(backing));
        buf.shared_epoch = buf.content_epoch;
        buf.phys_kind = CudaPhysKind::Shared;
        buf.memory_is_external = true;
        self.graph_stats.buffer_promotions.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(buffer, "CUDA: promoted Native → Shared");
        self.register_buffer_bindless_descriptor(buffer)?;
        Ok(())
    }

    /// Shared → NativeAndTwin: allocate native, copy bytes, keep import as twin.
    fn promote_shared_to_native_and_twin(&mut self, buffer: BufferHandle, capacity: u64) -> Result<()> {
        let (device, size, offset, shared) = {
            let buf = self.buffers.get(&buffer).unwrap();
            (buf.device, buf.size, buf.offset, buf.shared.clone())
        };
        let Some(shared) = shared else {
            bail!("CUDA: promote Shared without shared backing");
        };

        // DX12 may still be reading the import as an IA VB from the prior draw.
        let companion = Arc::clone(
            self.device(device)?
                .dx12
                .as_ref()
                .context("CUDA/DX12: promote Shared requires companion")?,
        );
        companion
            .wait_idle()
            .context("CUDA/DX12: idle companion before Shared→NativeAndTwin")?;
        let cuda_fence = shared.last_cuda_fence.load(Ordering::Acquire);
        if cuda_fence > 0 {
            companion
                .cpu_wait(cuda_fence)
                .context("CUDA/DX12: wait CUDA fence before Shared promote")?;
        }
        let worker = Arc::clone(&self.device(device)?.submission_worker);
        worker.flush().context("CUDA: flush worker before Shared promote")?;
        self.graph_stats.worker_flushes.fetch_add(1, Ordering::Relaxed);

        let old_memory = self
            .buffers
            .get(&buffer)
            .and_then(|buf| buf.memory.clone());
        let gpu = self.device(device)?;
        let target_device_ptr = shared.import.device_ptr + offset;
        let stream = Arc::clone(&gpu.alloc_stream);
        if let Some(ref memory) = old_memory {
            self.evict_retained_touching_memory(memory, &stream, target_device_ptr, true);
            self.evict_retained_for_buffer(buffer, memory, &stream, target_device_ptr, true);
            self.drop_retained_graphs_holding_memory(device, memory, &stream, target_device_ptr);
            self.clear_device_retained_sync(device);
        }
        worker.flush().context("CUDA: flush worker after retained eviction")?;

        // Take the external slice Arc so we can leak it after copy (Drop must not free).
        let old_memory = self
            .buffers
            .get_mut(&buffer)
            .unwrap()
            .memory
            .take()
            .context("CUDA: promote Shared without memory")?;

        let gpu = self.device(device)?;
        gpu.ctx
            .bind_to_thread()
            .context("CUDA: bind context for Shared promote")?;
        let mut native = gpu
            .alloc_stream
            .alloc_zeros::<u8>(capacity as usize)
            .context("CUDA: promote Shared→native alloc")?;
        if size > 0 {
            let nbytes = size as usize;
            let src_ptr = shared.import.device_ptr;
            let dst_ptr = {
                let view = native.try_slice_mut(0..nbytes).context("CUDA: promote dst view")?;
                let (ptr, _) = view.device_ptr(&gpu.alloc_stream);
                ptr
            };
            // Account for CudaBuffer views that shift into a parent allocation.
            let src_ptr = src_ptr + offset;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(dst_ptr, src_ptr, nbytes, gpu.alloc_stream.cu_stream())
            }
            .context("CUDA: promote Shared→native DtoD")?;
            gpu.alloc_stream.synchronize().context("CUDA: promote synchronize")?;
        }

        match Arc::try_unwrap(old_memory) {
            Ok(mutex) => {
                leak_shared_buffer_slice(mutex.into_inner().unwrap_or_else(|e| e.into_inner()));
            }
            Err(still_shared) => {
                // Keep alive without cuMemFree — attach to twin for drop ordering.
                tracing::warn!(
                    "CUDA: promote Shared→native: external slice Arc still shared (count={})",
                    Arc::strong_count(&still_shared)
                );
                std::mem::forget(still_shared);
            }
        }

        let buf = self.buffers.get_mut(&buffer).unwrap();
        buf.memory = Some(Arc::new(Mutex::new(native)));
        buf.memory_is_external = false;
        buf.shared = Some(shared);
        buf.shared_epoch = u64::MAX;
        buf.phys_kind = CudaPhysKind::NativeAndTwin;
        self.graph_stats.buffer_promotions.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(buffer, "CUDA: promoted Shared → NativeAndTwin");
        Ok(())
    }

    /// Drop retained entries whose ops keep `memory` alive.
    fn evict_retained_touching_memory(
        &mut self,
        memory: &Arc<Mutex<CudaSlice<u8>>>,
        stream: &CudaStream,
        target_device_ptr: u64,
        sync_destroy: bool,
    ) {
        let keys: Vec<_> = self
            .retained
            .iter()
            .filter_map(|((ctx, key), entry)| {
                let touches = match entry {
                    super::RetainedEntry::Ops(ops) => ops_touch_memory(ops, memory, stream, target_device_ptr),
                    super::RetainedEntry::GraphWithTail { tail, .. } => {
                        ops_touch_memory(tail, memory, stream, target_device_ptr)
                    }
                    _ => false,
                };
                touches.then_some((*ctx, *key))
            })
            .collect();
        for (ctx, key) in keys {
            if self.retained.remove(&(ctx, key)).is_some() {
                if sync_destroy {
                    self.destroy_retained_graph_sync(ctx, key);
                } else {
                    self.enqueue_evict_retained(ctx, key);
                }
            }
        }
    }

    /// Evict retained CUDA graphs that still reference `buffer` (registry keep-alive).
    fn evict_retained_for_buffer(
        &mut self,
        buffer: BufferHandle,
        memory: &Arc<Mutex<CudaSlice<u8>>>,
        stream: &CudaStream,
        target_device_ptr: u64,
        sync_destroy: bool,
    ) {
        let keys: Vec<_> = self
            .retained
            .iter()
            .filter_map(|((ctx, key), entry)| {
                let touches = match entry {
                    super::RetainedEntry::Graph {
                        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                        twin_dirty,
                        ..
                    } => {
                        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                        {
                            twin_dirty.contains(&buffer)
                        }
                        #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                        {
                            false
                        }
                    }
                    super::RetainedEntry::GraphWithTail {
                        tail,
                        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                        twin_dirty,
                        ..
                    } => {
                        let tail_hit = ops_touch_memory(tail, memory, stream, target_device_ptr);
                        #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
                        {
                            tail_hit || twin_dirty.contains(&buffer)
                        }
                        #[cfg(not(all(feature = "graphics", feature = "dx12", target_os = "windows")))]
                        {
                            tail_hit
                        }
                    }
                    _ => false,
                };
                touches.then_some((*ctx, *key))
            })
            .collect();
        for (ctx, key) in keys {
            if self.retained.remove(&(ctx, key)).is_some() {
                if sync_destroy {
                    self.destroy_retained_graph_sync(ctx, key);
                } else {
                    self.enqueue_evict_retained(ctx, key);
                }
            }
        }
    }

    /// Write into already-physical memory (no deferred staging).
    pub(super) fn write_buffer_physical(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let device = self.buffers.get(&buffer).unwrap().device;
        let stream = Arc::clone(&self.device(device)?.alloc_stream);
        let buffer_ref = self.buffers.get(&buffer).unwrap();
        Self::write_buffer_region(&stream, buffer_ref, offset, data)?;
        Ok(())
    }

    /// Register a companion SRV (or CBV for broadcast-sized) at the buffer's CUDA slot.
    pub(super) fn register_buffer_bindless_descriptor(&mut self, buffer: BufferHandle) -> Result<()> {
        let (device, slot, size, stride, shared) = {
            let buf = self
                .buffers
                .get(&buffer)
                .context("CUDA: register bindless descriptor: invalid buffer")?;
            (
                buf.device,
                buf.slot,
                buf.size,
                buf.element_stride.unwrap_or(4),
                buf.shared.clone(),
            )
        };
        let Some(slot) = slot else {
            return Ok(());
        };
        let Some(shared) = shared else {
            bail!("CUDA/DX12: cannot register bindless descriptor without shared backing");
        };
        let companion = self
            .device(device)?
            .dx12
            .as_ref()
            .context("CUDA/DX12: companion required for bindless descriptor")?;
        let view_stride = stride.max(4);
        // NumElements must not imply a view larger than the resource (D3D12 removes the
        // device / drops draws). Use the view stride, not the logical element stride.
        let num_elements = (size / u64::from(view_stride)).max(1) as u32;
        companion.bindless.write_buffer_srv(
            &companion.device,
            slot,
            &shared.d3d12_resource,
            num_elements,
            view_stride,
        )?;
        Ok(())
    }
}

fn compatible(have: CudaPhysKind, want: CudaPhysKind) -> bool {
    match (have, want) {
        (a, b) if a == b => true,
        // Stronger backings satisfy weaker needs.
        (CudaPhysKind::NativeAndTwin, CudaPhysKind::Native) => true,
        (CudaPhysKind::NativeAndTwin, CudaPhysKind::Shared) => true,
        (CudaPhysKind::Shared, CudaPhysKind::Shared) => true,
        (CudaPhysKind::Native, CudaPhysKind::Native) => true,
        _ => false,
    }
}

fn ops_touch_memory(
    ops: &[super::pending_submit::CudaOp],
    memory: &Arc<Mutex<CudaSlice<u8>>>,
    stream: &CudaStream,
    target_device_ptr: u64,
) -> bool {
    use super::pending_submit::CudaOp;
    ops.iter().any(|op| match op {
        CudaOp::Clear { memory: m, .. } | CudaOp::Write { memory: m, .. } => {
            memory_slices_same(m, memory, stream, target_device_ptr)
        }
        CudaOp::Copy { src, dst, .. } => {
            memory_slices_same(src, memory, stream, target_device_ptr)
                || memory_slices_same(dst, memory, stream, target_device_ptr)
        }
        CudaOp::Launch { keep_alive_buffers, .. } | CudaOp::LaunchIndirect { keep_alive_buffers, .. } => {
            keep_alive_buffers
                .iter()
                .any(|m| memory_slices_same(m, memory, stream, target_device_ptr))
        }
        CudaOp::CopyTextureToBuffer { dst, .. } => memory_slices_same(dst, memory, stream, target_device_ptr),
        _ => false,
    })
}

fn memory_slices_same(
    candidate: &Arc<Mutex<CudaSlice<u8>>>,
    memory: &Arc<Mutex<CudaSlice<u8>>>,
    stream: &CudaStream,
    target_device_ptr: u64,
) -> bool {
    if Arc::ptr_eq(candidate, memory) {
        return true;
    }
    memory_slice_covers_ptr(candidate, stream, target_device_ptr)
}

fn memory_slice_covers_ptr(candidate: &Arc<Mutex<CudaSlice<u8>>>, stream: &CudaStream, ptr: u64) -> bool {
    let Ok(candidate_guard) = candidate.try_lock() else {
        return false;
    };
    let (base_ptr, _) = candidate_guard.device_ptr(stream);
    let len = candidate_guard.len();
    ptr >= base_ptr && ptr < base_ptr + len as u64
}
