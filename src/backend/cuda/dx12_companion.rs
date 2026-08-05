//! D3D12 companion device for CUDA presentation (Windows only).
//!
//! Compiled only when `cuda`, `graphics`, and `dx12` are all enabled. Pairs a CUDA
//! ordinal with the matching DXGI adapter by LUID, then owns a DIRECT queue plus a
//! shareable fence imported into CUDA as an external semaphore.

use anyhow::{bail, Context as _, Result};
use cudarc::driver::{sys, CudaContext, CudaStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE, LUID};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12DescriptorHeap,
    ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList, ID3D12Resource, ID3D12RootSignature,
    D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
    D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_DEPTH_STENCIL_VALUE, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_FENCE_FLAG_SHARED,
    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN,
    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
    D3D12_RESOURCE_STATE_COMMON, D3D12_TEXTURE_LAYOUT_UNKNOWN,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIAdapter1, IDXGIFactory4, IDXGIFactory5, DXGI_ADAPTER_FLAG, DXGI_ADAPTER_FLAG_SOFTWARE,
    DXGI_CREATE_FACTORY_FLAGS, DXGI_FEATURE_PRESENT_ALLOW_TEARING,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D32_FLOAT,
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
};
use windows::Win32::System::Threading::{CreateEventA, WaitForSingleObject, INFINITE};

/// Max offscreen RTV descriptors in the companion RTV heap.
pub(super) const MAX_RTV_DESCRIPTORS: u32 = 64;

/// Max DSV descriptors (offscreen RTs + surfaces); matches RTV capacity.
pub(super) const MAX_DSV_DESCRIPTORS: u32 = 64;

pub(super) fn depth_format_to_dxgi(format: crate::types::DepthFormat) -> DXGI_FORMAT {
    match format {
        crate::types::DepthFormat::Depth16Unorm => DXGI_FORMAT_D16_UNORM,
        crate::types::DepthFormat::Depth24Plus => DXGI_FORMAT_D32_FLOAT,
        crate::types::DepthFormat::Depth24PlusStencil8 => DXGI_FORMAT_D24_UNORM_S8_UINT,
        crate::types::DepthFormat::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        crate::types::DepthFormat::Depth32FloatStencil8 => DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    }
}

/// 8-byte Windows LUID shared by DXGI and `cuDeviceGetLuid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct AdapterLuid {
    pub bytes: [u8; 8],
}

impl AdapterLuid {
    pub fn from_windows(luid: LUID) -> Self {
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
        bytes[4..].copy_from_slice(&luid.HighPart.to_le_bytes());
        Self { bytes }
    }

    pub fn from_cuda_device(cu_device: sys::CUdevice) -> Result<Self> {
        let mut luid = [0i8; 8];
        let mut node_mask: u32 = 0;
        let r = unsafe { sys::cuDeviceGetLuid(luid.as_mut_ptr(), &mut node_mask, cu_device) };
        if r != sys::CUresult::CUDA_SUCCESS {
            bail!("CUDA: cuDeviceGetLuid failed: {r:?}");
        }
        let mut bytes = [0u8; 8];
        for (i, b) in luid.iter().enumerate() {
            bytes[i] = *b as u8;
        }
        Ok(Self { bytes })
    }
}

/// Match a CUDA context's physical GPU to a DXGI hardware adapter by LUID.
pub(super) fn find_matching_dxgi_adapter(
    factory: &IDXGIFactory4,
    cuda_luid: AdapterLuid,
) -> Result<(IDXGIAdapter1, u32, AdapterLuid)> {
    let mut index = 0u32;
    loop {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(a) => a,
            Err(_) => break,
        };
        let desc = unsafe { adapter.GetDesc1() }.context("DXGI: GetDesc1 failed")?;
        let flags = DXGI_ADAPTER_FLAG(desc.Flags as i32);
        if flags.contains(DXGI_ADAPTER_FLAG_SOFTWARE) {
            index += 1;
            continue;
        }
        let luid = AdapterLuid::from_windows(desc.AdapterLuid);
        if luid == cuda_luid {
            let name = String::from_utf16_lossy(&desc.Description)
                .trim_end_matches('\0')
                .to_string();
            tracing::info!(
                "CUDA↔DX12 LUID match: adapter {index} ({name}) bytes={:02x?}",
                luid.bytes
            );
            return Ok((adapter, index, luid));
        }
        index += 1;
    }
    bail!(
        "CUDA: no DXGI hardware adapter matches CUDA LUID {:02x?} \
         (linked-node / TCC / headless NVIDIA devices cannot present via DX12)",
        cuda_luid.bytes
    )
}

/// Minimal D3D12 device + DIRECT queue + shared fence for CUDA presentation.
pub(super) struct Dx12Companion {
    pub factory: IDXGIFactory4,
    pub allow_tearing: bool,
    pub adapter_luid: AdapterLuid,
    pub dxgi_adapter_id: u32,
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    pub queue_lock: std::sync::Mutex<()>,
    /// Shareable fence; CUDA imports it as an external semaphore.
    pub fence: ID3D12Fence,
    pub fence_value: AtomicU64,
    /// CUDA import of [`Self::fence`].
    pub cuda_semaphore: sys::CUexternalSemaphore,
    pub cuda_ctx: Arc<CudaContext>,
    /// Dedicated stream for publishing present-completion into the Goldy timeline.
    /// Separate from per-context submit streams so present recording cannot race the
    /// submission worker.
    pub present_stream: Arc<CudaStream>,
    /// Scratch allocator/list pool for present copy + blit (one per in-flight slot).
    pub present_slots: Vec<PresentCommandSlot>,
    /// Dedicated command allocator/list for one-shot resource-state init.
    /// Must not share present slots — those may still be GPU-busy when scratch is created.
    pub init_allocator: ID3D12CommandAllocator,
    pub init_list: ID3D12GraphicsCommandList,
    /// Fence value of the last submit that used [`Self::init_allocator`].
    pub init_fence: AtomicU64,
    /// RTV descriptor heap for offscreen raster targets.
    pub rtv_heap: ID3D12DescriptorHeap,
    pub rtv_descriptor_size: u32,
    /// High-water mark for never-recycled RTV slots (prefer [`Self::free_rtv_offsets`]).
    pub next_rtv_offset: AtomicU64,
    /// Recycled RTV heap offsets from destroyed offscreen targets.
    pub free_rtv_offsets: std::sync::Mutex<Vec<u32>>,
    /// DSV descriptor heap for offscreen RTs and surfaces (DX12-only depth; not CUDA-imported).
    pub dsv_heap: ID3D12DescriptorHeap,
    pub dsv_descriptor_size: u32,
    /// High-water mark for never-recycled DSV slots (prefer [`Self::free_dsv_offsets`]).
    pub next_dsv_offset: AtomicU64,
    /// Recycled DSV heap offsets from destroyed depth targets.
    pub free_dsv_offsets: std::sync::Mutex<Vec<u32>>,
    /// SM 6.6 bindless heaps + root signature (IA + directly-indexed descriptors).
    pub bindless: super::dx12_bindless::BindlessHeaps,
    /// Device-level frame-table (selector/table at protocol slots 0/1).
    pub frame_table: super::dx12_bindless::CompanionFrameTable,
    /// Bindless root signature shared by all graphics PSOs.
    pub graphics_root_signature: ID3D12RootSignature,
    /// Rotating allocator/list slots for `render_to_target` (no per-frame CPU wait).
    pub raster_slots: Vec<PresentCommandSlot>,
    /// Next raster slot index (0..MAX_FRAMES).
    pub raster_slot: AtomicU64,
}

pub(super) struct PresentCommandSlot {
    pub allocator: ID3D12CommandAllocator,
    pub list: ID3D12GraphicsCommandList,
    /// Last fence value submitted with this slot (0 = never used).
    pub fence_value: AtomicU64,
    /// Incremented whenever this command list is reset for a new recording.
    pub generation: AtomicU64,
    /// Fingerprint of the closed retained recording on this slot (0 = none).
    pub retained_fingerprint: AtomicU64,
}

impl Dx12Companion {
    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        let cuda_luid = AdapterLuid::from_cuda_device(cuda_ctx.cu_device())?;
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .context("CUDA/DX12: CreateDXGIFactory2 failed")?;
        let allow_tearing = factory
            .cast::<IDXGIFactory5>()
            .ok()
            .and_then(|f5| {
                let mut allow: i32 = 0;
                let hr = unsafe {
                    f5.CheckFeatureSupport(
                        DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                        &mut allow as *mut _ as *mut _,
                        std::mem::size_of::<i32>() as u32,
                    )
                };
                hr.ok().map(|()| allow != 0)
            })
            .unwrap_or(false);

        let (adapter, dxgi_adapter_id, adapter_luid) = find_matching_dxgi_adapter(&factory, cuda_luid)?;

        let mut device: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }
            .context("CUDA/DX12: D3D12CreateDevice failed")?;
        let device = device.context("CUDA/DX12: D3D12CreateDevice returned null")?;

        let node_count = unsafe { device.GetNodeCount() };
        if node_count != 1 {
            bail!("CUDA/DX12: linked-node adapters are not supported yet (GetNodeCount={node_count})");
        }

        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        let queue: ID3D12CommandQueue =
            unsafe { device.CreateCommandQueue(&queue_desc) }.context("CUDA/DX12: CreateCommandQueue failed")?;

        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_SHARED) }
            .context("CUDA/DX12: CreateFence(SHARED) failed")?;

        let fence_handle: HANDLE =
            unsafe { device.CreateSharedHandle(&fence, None, windows::Win32::Foundation::GENERIC_ALL.0, None) }
                .context("CUDA/DX12: CreateSharedHandle(fence) failed")?;

        let cuda_semaphore = import_d3d12_fence(cuda_ctx, fence_handle)?;
        // CUDA does not take ownership of the Win32 NT handle.
        unsafe {
            let _ = CloseHandle(fence_handle);
        }

        let mut present_slots = Vec::with_capacity(MAX_FRAMES);
        for _ in 0..MAX_FRAMES {
            let allocator: ID3D12CommandAllocator =
                unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                    .context("CUDA/DX12: CreateCommandAllocator failed")?;
            let list: ID3D12GraphicsCommandList =
                unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None) }
                    .context("CUDA/DX12: CreateCommandList failed")?;
            unsafe { list.Close() }.context("CUDA/DX12: Close initial command list")?;
            present_slots.push(PresentCommandSlot {
                allocator,
                list,
                fence_value: AtomicU64::new(0),
                generation: AtomicU64::new(0),
                retained_fingerprint: AtomicU64::new(0),
            });
        }

        // Dedicated allocator/list for one-shot state init (must not steal present slots).
        let init_allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                .context("CUDA/DX12: CreateCommandAllocator(init) failed")?;
        let init_list: ID3D12GraphicsCommandList =
            unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &init_allocator, None) }
                .context("CUDA/DX12: CreateCommandList(init) failed")?;
        unsafe { init_list.Close() }.context("CUDA/DX12: Close init command list")?;

        let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: MAX_RTV_DESCRIPTORS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let rtv_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&rtv_heap_desc) }
            .context("CUDA/DX12: CreateDescriptorHeap(RTV) failed")?;
        let rtv_descriptor_size = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };

        let dsv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            NumDescriptors: MAX_DSV_DESCRIPTORS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let dsv_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&dsv_heap_desc) }
            .context("CUDA/DX12: CreateDescriptorHeap(DSV) failed")?;
        let dsv_descriptor_size = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) };

        let bindless = super::dx12_bindless::BindlessHeaps::create(&device)?;
        let frame_table = super::dx12_bindless::CompanionFrameTable::create(&device, &bindless)?;
        let graphics_root_signature = bindless.root_signature.clone();

        let mut raster_slots = Vec::with_capacity(MAX_FRAMES);
        for _ in 0..MAX_FRAMES {
            let allocator: ID3D12CommandAllocator =
                unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                    .context("CUDA/DX12: CreateCommandAllocator(raster) failed")?;
            let list: ID3D12GraphicsCommandList =
                unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None) }
                    .context("CUDA/DX12: CreateCommandList(raster) failed")?;
            unsafe { list.Close() }.context("CUDA/DX12: Close raster command list")?;
            raster_slots.push(PresentCommandSlot {
                allocator,
                list,
                fence_value: AtomicU64::new(0),
                generation: AtomicU64::new(0),
                retained_fingerprint: AtomicU64::new(0),
            });
        }

        let present_stream = cuda_ctx
            .new_stream()
            .context("CUDA/DX12: create present stream failed")?;

        Ok(Self {
            factory,
            allow_tearing,
            adapter_luid,
            dxgi_adapter_id,
            device,
            queue,
            queue_lock: std::sync::Mutex::new(()),
            fence,
            fence_value: AtomicU64::new(1),
            cuda_semaphore,
            cuda_ctx: Arc::clone(cuda_ctx),
            present_stream,
            present_slots,
            init_allocator,
            init_list,
            init_fence: AtomicU64::new(0),
            rtv_heap,
            rtv_descriptor_size,
            next_rtv_offset: AtomicU64::new(0),
            free_rtv_offsets: std::sync::Mutex::new(Vec::new()),
            dsv_heap,
            dsv_descriptor_size,
            next_dsv_offset: AtomicU64::new(0),
            free_dsv_offsets: std::sync::Mutex::new(Vec::new()),
            bindless,
            frame_table,
            graphics_root_signature,
            raster_slots,
            raster_slot: AtomicU64::new(0),
        })
    }

    /// Allocate the next fence value (monotonic, starts at 1).
    pub fn next_fence_value(&self) -> u64 {
        self.fence_value.fetch_add(1, Ordering::AcqRel)
    }

    pub fn signal_queue(&self, value: u64) -> Result<()> {
        let _guard = self.queue_lock.lock().unwrap();
        unsafe { self.queue.Signal(&self.fence, value) }.context("CUDA/DX12: queue Signal failed")?;
        Ok(())
    }

    pub fn wait_queue(&self, value: u64) -> Result<()> {
        let _guard = self.queue_lock.lock().unwrap();
        unsafe { self.queue.Wait(&self.fence, value) }.context("CUDA/DX12: queue Wait failed")?;
        Ok(())
    }

    pub fn execute_and_signal(
        &self,
        lists: &[Option<windows::Win32::Graphics::Direct3D12::ID3D12CommandList>],
        signal_value: u64,
    ) -> Result<()> {
        let _guard = self.queue_lock.lock().unwrap();
        unsafe { self.queue.ExecuteCommandLists(lists) };
        unsafe { self.queue.Signal(&self.fence, signal_value) }
            .context("CUDA/DX12: Signal after ExecuteCommandLists failed")?;
        Ok(())
    }

    pub fn cpu_wait(&self, value: u64) -> Result<()> {
        if unsafe { self.fence.GetCompletedValue() } >= value {
            return Ok(());
        }
        let event = unsafe { CreateEventA(None, false, false, None) }.context("CUDA/DX12: CreateEventA failed")?;
        unsafe { self.fence.SetEventOnCompletion(value, event) }.context("CUDA/DX12: SetEventOnCompletion failed")?;
        unsafe { WaitForSingleObject(event, INFINITE) };
        unsafe {
            let _ = CloseHandle(event);
        }
        Ok(())
    }

    pub fn wait_idle(&self) -> Result<()> {
        let v = self.next_fence_value();
        self.signal_queue(v)?;
        self.cpu_wait(v)
    }

    /// Record and submit a one-shot barrier list on the dedicated init allocator.
    pub fn submit_init_list(&self) -> Result<()> {
        let prev = self.init_fence.load(Ordering::Acquire);
        if prev > 0 {
            self.cpu_wait(prev)?;
        }
        unsafe { self.init_allocator.Reset() }.context("CUDA/DX12: reset init allocator")?;
        unsafe { self.init_list.Reset(&self.init_allocator, None) }.context("CUDA/DX12: reset init list")?;
        Ok(())
    }

    pub fn finish_init_list(&self) -> Result<()> {
        unsafe { self.init_list.Close() }.context("CUDA/DX12: close init list")?;
        let cmd: ID3D12CommandList = self.init_list.cast().context("cast init list")?;
        let signal = self.next_fence_value();
        self.execute_and_signal(&[Some(cmd)], signal)?;
        self.cpu_wait(signal)?;
        self.init_fence.store(signal, Ordering::Release);
        Ok(())
    }

    /// Allocate an RTV heap offset (recycles freed slots; fails if the heap is exhausted).
    pub fn alloc_rtv_offset(&self) -> Result<u32> {
        if let Some(offset) = self.free_rtv_offsets.lock().unwrap().pop() {
            return Ok(offset);
        }
        let next = self.next_rtv_offset.fetch_add(1, Ordering::AcqRel);
        if next >= MAX_RTV_DESCRIPTORS as u64 {
            // Leave the counter past the limit so subsequent allocs keep failing loudly.
            bail!(
                "CUDA/DX12: RTV heap exhausted ({MAX_RTV_DESCRIPTORS} offscreen targets); \
                 destroy unused render targets to recycle descriptors"
            );
        }
        Ok(next as u32)
    }

    /// Return an RTV offset to the free list for reuse.
    pub fn free_rtv_offset(&self, offset: u32) {
        if offset < MAX_RTV_DESCRIPTORS {
            self.free_rtv_offsets.lock().unwrap().push(offset);
        }
    }

    /// Allocate a DSV heap offset (recycles freed slots; fails if the heap is exhausted).
    pub fn alloc_dsv_offset(&self) -> Result<u32> {
        if let Some(offset) = self.free_dsv_offsets.lock().unwrap().pop() {
            return Ok(offset);
        }
        let next = self.next_dsv_offset.fetch_add(1, Ordering::AcqRel);
        if next >= MAX_DSV_DESCRIPTORS as u64 {
            bail!(
                "CUDA/DX12: DSV heap exhausted ({MAX_DSV_DESCRIPTORS} depth targets); \
                 destroy unused render targets / surfaces to recycle descriptors"
            );
        }
        Ok(next as u32)
    }

    /// Return a DSV offset to the free list for reuse.
    pub fn free_dsv_offset(&self, offset: u32) {
        if offset < MAX_DSV_DESCRIPTORS {
            self.free_dsv_offsets.lock().unwrap().push(offset);
        }
    }

    /// Create a DX12-only depth texture + DSV (not CUDA-imported).
    pub fn create_depth_texture(
        &self,
        width: u32,
        height: u32,
        format: crate::types::DepthFormat,
    ) -> Result<(ID3D12Resource, u32)> {
        let dxgi = depth_format_to_dxgi(format);
        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };
        let depth_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width.max(1) as u64,
            Height: height.max(1),
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: dxgi,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        };
        let depth_clear = D3D12_CLEAR_VALUE {
            Format: dxgi,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 },
            },
        };
        let mut depth_tex: Option<ID3D12Resource> = None;
        unsafe {
            self.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &depth_desc,
                D3D12_RESOURCE_STATE_COMMON,
                Some(&depth_clear),
                &mut depth_tex,
            )
        }
        .context("CUDA/DX12: CreateCommittedResource(depth) failed")?;
        let depth_tex = depth_tex.context("CUDA/DX12: CreateCommittedResource(depth) returned null")?;

        let dsv_offset = self.alloc_dsv_offset()?;
        let dsv_handle = unsafe {
            let mut handle = self.dsv_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (dsv_offset as usize) * self.dsv_descriptor_size as usize;
            handle
        };
        unsafe {
            self.device.CreateDepthStencilView(&depth_tex, None, dsv_handle);
        }
        Ok((depth_tex, dsv_offset))
    }

    /// Reset the next raster command allocator/list (waits only that slot's prior fence).
    ///
    /// Prefers a GPU-retired slot so warmup can populate multiple retained copies
    /// without blocking; only waits when every slot is still in flight.
    pub fn begin_raster_list(&self) -> Result<(usize, ID3D12GraphicsCommandList, u64)> {
        let completed = unsafe { self.fence.GetCompletedValue() };
        let start = (self.raster_slot.fetch_add(1, Ordering::AcqRel) as usize) % MAX_FRAMES;
        let mut chosen = None;
        for offset in 0..MAX_FRAMES {
            let idx = (start + offset) % MAX_FRAMES;
            let prev = self.raster_slots[idx].fence_value.load(Ordering::Acquire);
            if prev <= completed {
                chosen = Some(idx);
                break;
            }
        }
        let idx = chosen.unwrap_or(start);
        let slot = &self.raster_slots[idx];
        let prev = slot.fence_value.load(Ordering::Acquire);
        if prev > completed {
            self.cpu_wait(prev)?;
        }
        unsafe { slot.allocator.Reset() }.context("CUDA/DX12: reset raster allocator")?;
        unsafe { slot.list.Reset(&slot.allocator, None) }.context("CUDA/DX12: reset raster list")?;
        slot.retained_fingerprint.store(0, Ordering::Release);
        let generation = slot.generation.fetch_add(1, Ordering::AcqRel) + 1;
        Ok((idx, slot.list.clone(), generation))
    }

    /// Close and execute the raster list for `slot_idx` without a CPU wait.
    /// Returns the signaled fence value (GPU-side retirement).
    pub fn finish_raster_list(&self, slot_idx: usize, fingerprint: u64) -> Result<u64> {
        let slot = &self.raster_slots[slot_idx];
        unsafe { slot.list.Close() }.context("CUDA/DX12: close raster list")?;
        let cmd: ID3D12CommandList = slot.list.cast().context("cast raster list")?;
        let signal = self.next_fence_value();
        self.execute_and_signal(&[Some(cmd)], signal)?;
        slot.fence_value.store(signal, Ordering::Release);
        slot.retained_fingerprint.store(fingerprint, Ordering::Release);
        Ok(signal)
    }

    /// Re-execute a closed retained raster list with no CPU wait.
    ///
    /// A closed D3D12 command list may be `ExecuteCommandLists`'d repeatedly while a
    /// prior execute is still in flight; only `Reset` requires retirement. Prefer a
    /// GPU-retired matching slot when available, otherwise re-execute any matching
    /// copy — never `cpu_wait` on the reuse path.
    pub fn try_reuse_raster_for_fingerprint(&self, fingerprint: u64) -> Result<Option<u64>> {
        if fingerprint == 0 {
            return Ok(None);
        }
        let completed = unsafe { self.fence.GetCompletedValue() };
        let start = self.raster_slot.load(Ordering::Acquire) as usize;
        let mut busy_match: Option<usize> = None;
        for offset in 0..MAX_FRAMES {
            let idx = (start + offset) % MAX_FRAMES;
            let slot = &self.raster_slots[idx];
            if slot.retained_fingerprint.load(Ordering::Acquire) != fingerprint {
                continue;
            }
            let prev = slot.fence_value.load(Ordering::Acquire);
            if prev <= completed {
                return self.reexecute_raster_slot(idx);
            }
            if busy_match.is_none() {
                busy_match = Some(idx);
            }
        }
        if let Some(idx) = busy_match {
            return self.reexecute_raster_slot(idx);
        }
        Ok(None)
    }

    fn reexecute_raster_slot(&self, idx: usize) -> Result<Option<u64>> {
        let slot = &self.raster_slots[idx];
        let cmd: ID3D12CommandList = slot.list.cast().context("cast retained raster list")?;
        let signal = self.next_fence_value();
        self.execute_and_signal(&[Some(cmd)], signal)?;
        slot.fence_value.store(signal, Ordering::Release);
        self.raster_slot.store((idx as u64).wrapping_add(1), Ordering::Release);
        Ok(Some(signal))
    }

    /// Reset retained present lists after retirement so they release swapchain resources.
    #[allow(dead_code)]
    pub fn invalidate_present_lists(&self) -> Result<()> {
        self.invalidate_command_slots(&self.present_slots)
    }

    /// Wait + Reset + bump generation for an arbitrary present/raster slot pool.
    pub fn invalidate_command_slots(&self, slots: &[PresentCommandSlot]) -> Result<()> {
        for slot in slots {
            let prev = slot.fence_value.load(Ordering::Acquire);
            if prev > 0 {
                self.cpu_wait(prev)?;
            }
            unsafe { slot.allocator.Reset() }.context("CUDA/DX12: reset invalidated present allocator")?;
            unsafe { slot.list.Reset(&slot.allocator, None) }.context("CUDA/DX12: reset invalidated present list")?;
            unsafe { slot.list.Close() }.context("CUDA/DX12: close invalidated present list")?;
            slot.generation.fetch_add(1, Ordering::AcqRel);
            slot.retained_fingerprint.store(0, Ordering::Release);
            slot.fence_value.store(0, Ordering::Release);
        }
        Ok(())
    }

    /// Per-surface present allocator/list pool (multi-window must not share these).
    pub fn create_present_command_slots(&self) -> Result<Vec<PresentCommandSlot>> {
        let mut present_slots = Vec::with_capacity(MAX_FRAMES);
        for _ in 0..MAX_FRAMES {
            let allocator: ID3D12CommandAllocator =
                unsafe { self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                    .context("CUDA/DX12: CreateCommandAllocator(present surface) failed")?;
            let list: ID3D12GraphicsCommandList =
                unsafe { self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None) }
                    .context("CUDA/DX12: CreateCommandList(present surface) failed")?;
            unsafe { list.Close() }.context("CUDA/DX12: Close surface present command list")?;
            present_slots.push(PresentCommandSlot {
                allocator,
                list,
                fence_value: AtomicU64::new(0),
                generation: AtomicU64::new(0),
                retained_fingerprint: AtomicU64::new(0),
            });
        }
        Ok(present_slots)
    }

    /// Highest fence value known for companion-owned work (init + raster/present slots).
    pub fn companion_fence_high_water(&self) -> u64 {
        let mut high = self.init_fence.load(Ordering::Acquire);
        for slot in self.raster_slots.iter().chain(self.present_slots.iter()) {
            high = high.max(slot.fence_value.load(Ordering::Acquire));
        }
        high
    }
}

impl Drop for Dx12Companion {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        let _ = self.cuda_ctx.bind_to_thread();
        let sem = std::mem::replace(&mut self.cuda_semaphore, std::ptr::null_mut());
        if !sem.is_null() {
            let _ = unsafe { sys::cuDestroyExternalSemaphore(sem) };
        }
    }
}

// SAFETY: COM objects + CUDA semaphore are used under Goldy's backend / queue locks.
unsafe impl Send for Dx12Companion {}
unsafe impl Sync for Dx12Companion {}

pub(super) const MAX_FRAMES: usize = 3;

fn import_d3d12_fence(cuda_ctx: &Arc<CudaContext>, handle: HANDLE) -> Result<sys::CUexternalSemaphore> {
    cuda_ctx
        .bind_to_thread()
        .context("CUDA/DX12: bind context for fence import")?;
    let desc = sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
        type_: sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
        handle: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: handle.0,
                name: std::ptr::null(),
            },
        },
        flags: 0,
        reserved: [0; 16],
    };
    let mut sem: sys::CUexternalSemaphore = std::ptr::null_mut();
    let r = unsafe { sys::cuImportExternalSemaphore(&mut sem, &desc) };
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA: cuImportExternalSemaphore(D3D12_FENCE) failed: {r:?}");
    }
    Ok(sem)
}

/// Signal the imported D3D12 fence from a CUDA stream.
pub(super) fn cuda_signal_fence(
    cuda_ctx: &CudaContext,
    sem: sys::CUexternalSemaphore,
    stream: sys::CUstream,
    value: u64,
) -> Result<()> {
    cuda_ctx
        .bind_to_thread()
        .context("CUDA/DX12: bind for external signal")?;
    let params = sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
        params: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1 {
            fence: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_1 { value },
            nvSciSync: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_2 { reserved: 0 },
            keyedMutex: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_3 { key: 0 },
            reserved: [0; 12],
        },
        flags: 0,
        reserved: [0; 16],
    };
    let r = unsafe { sys::cuSignalExternalSemaphoresAsync(&sem, &params, 1, stream) };
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA: cuSignalExternalSemaphoresAsync failed: {r:?}");
    }
    Ok(())
}

/// Wait on the imported D3D12 fence from a CUDA stream.
pub(super) fn cuda_wait_fence(
    cuda_ctx: &CudaContext,
    sem: sys::CUexternalSemaphore,
    stream: sys::CUstream,
    value: u64,
) -> Result<()> {
    cuda_ctx.bind_to_thread().context("CUDA/DX12: bind for external wait")?;
    let params = sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
        params: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1 {
            fence: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_1 { value },
            nvSciSync: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_2 { reserved: 0 },
            keyedMutex: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
                key: 0,
                timeoutMs: 0,
            },
            reserved: [0; 10],
        },
        flags: 0,
        reserved: [0; 16],
    };
    let r = unsafe { sys::cuWaitExternalSemaphoresAsync(&sem, &params, 1, stream) };
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA: cuWaitExternalSemaphoresAsync failed: {r:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luid_roundtrip_windows_layout() {
        let luid = LUID {
            LowPart: 0x11223344,
            HighPart: 0x55667788u32 as i32,
        };
        let a = AdapterLuid::from_windows(luid);
        assert_eq!(&a.bytes[..4], &0x11223344u32.to_le_bytes());
        assert_eq!(&a.bytes[4..], &(0x55667788u32 as i32).to_le_bytes());
    }

    #[test]
    fn companion_create_matches_cuda_luid_or_skips() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("skip: no CUDA device 0");
            return;
        };
        let cuda_luid = match AdapterLuid::from_cuda_device(ctx.cu_device()) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("skip: cuDeviceGetLuid failed: {e:#}");
                return;
            }
        };
        match Dx12Companion::create(&ctx) {
            Ok(companion) => {
                assert_eq!(companion.adapter_luid, cuda_luid);
                // WARP uses a sentinel in the main DX12 backend; companion rejects software.
                let stream = ctx.default_stream();
                let v = companion.next_fence_value();
                cuda_signal_fence(&ctx, companion.cuda_semaphore, stream.cu_stream(), v).expect("cuda signal");
                stream.synchronize().expect("stream sync after signal");
                companion.cpu_wait(v).expect("cpu wait on shared fence");
                let _ = companion.wait_idle();
            }
            Err(e) => eprintln!("skip: Dx12Companion::create failed: {e:#}"),
        }
    }
}
