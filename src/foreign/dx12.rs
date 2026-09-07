//! Direct3D 12 as a foreign graphics object: no Goldy device, verbs under one lock.
//!
//! Creates its own DXGI factory / `ID3D12Device` (serialised with the Goldy DX12
//! backend's process-wide `CreateDXGIFactory2` lock). Offscreen surfaces hold a
//! DEFAULT texture plus persistently mapped UPLOAD and READBACK buffers.
//! [`ForeignSurface::blit`] copies host pixels through `CopyTextureRegion` and a
//! copy-back into readback so [`ForeignSurface::snapshot`] can assert GPU contents.
//!
//! Prefers a hardware adapter, then WARP (needed on hosted Windows CI). Windowed
//! swapchain present is a later verb on this same singleton.

use crate::backend::dx12::{format_to_dxgi, DXGI_FACTORY_LOCK};
use crate::pixel::{PixelSink, PixmapLayout};
use crate::types::TextureFormat;
use crate::GoldyError;
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventA, WaitForSingleObject, INFINITE};

/// Process-wide DX12 adapter. Lazily created on [`try_adapter`].
pub struct ForeignDx12 {
    state: Mutex<AdapterState>,
}

struct AdapterState {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    next_id: u32,
    surfaces: HashMap<u32, SurfaceSlot>,
}

struct MappedBuffer {
    resource: ID3D12Resource,
    ptr: *mut u8,
    size: usize,
}

// Mapped host pointers are only used while `AdapterState` is locked.
unsafe impl Send for MappedBuffer {}

struct SurfaceSlot {
    width: u32,
    height: u32,
    format: TextureFormat,
    generation: u64,
    texture: ID3D12Resource,
    texture_state: D3D12_RESOURCE_STATES,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    upload: MappedBuffer,
    readback: MappedBuffer,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_value: u64,
    dropped: bool,
}

struct SurfaceHandle {
    adapter: Arc<ForeignDx12>,
    id: u32,
}

impl Drop for SurfaceHandle {
    fn drop(&mut self) {
        self.adapter.release(self.id);
    }
}

/// Offscreen DX12 texture owned by the foreign singleton.
#[derive(Clone)]
pub struct ForeignSurface {
    inner: Arc<SurfaceHandle>,
}

static ADAPTER: OnceLock<Result<Arc<ForeignDx12>, String>> = OnceLock::new();

/// Return the process-wide adapter, creating it on first success.
///
/// Returns `None` when D3D12 / DXGI is missing. Failures are cached.
pub fn try_adapter() -> Option<Arc<ForeignDx12>> {
    match ADAPTER.get_or_init(init_adapter) {
        Ok(a) => Some(Arc::clone(a)),
        Err(e) => {
            tracing::debug!("foreign DX12 adapter unavailable: {e}");
            None
        }
    }
}

fn init_adapter() -> Result<Arc<ForeignDx12>, String> {
    init_adapter_inner().map_err(|e| e.detail())
}

fn init_adapter_inner() -> Result<Arc<ForeignDx12>, GoldyError> {
    let factory: IDXGIFactory4 = {
        let _guard = DXGI_FACTORY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("CreateDXGIFactory2: {e}")))?
    };
    let adapter = pick_adapter(&factory)?;
    let desc = unsafe { adapter.GetDesc1() }
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("IDXGIAdapter1::GetDesc1: {e}")))?;
    let name = String::from_utf16_lossy(&desc.Description)
        .trim_end_matches('\0')
        .to_string();

    let mut device: Option<ID3D12Device> = None;
    unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("D3D12CreateDevice: {e}")))?;
    let device = device.ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("D3D12CreateDevice returned null")))?;

    let queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
        Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
        NodeMask: 0,
    };
    let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("CreateCommandQueue: {e}")))?;

    tracing::info!(adapter = %name, "foreign DX12 adapter");

    Ok(Arc::new(ForeignDx12 {
        state: Mutex::new(AdapterState {
            device,
            queue,
            next_id: 1,
            surfaces: HashMap::new(),
        }),
    }))
}

fn pick_adapter(factory: &IDXGIFactory4) -> Result<IDXGIAdapter1, GoldyError> {
    if crate::backend::dx12::env_force_warp() {
        return warp_adapter(factory);
    }
    let mut index = 0u32;
    loop {
        match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => {
                index += 1;
                let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
                    continue;
                };
                let flags = DXGI_ADAPTER_FLAG(desc.Flags as i32);
                if flags.contains(DXGI_ADAPTER_FLAG_SOFTWARE) {
                    continue;
                }
                let mut probe: Option<ID3D12Device> = None;
                if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut probe) }.is_ok() && probe.is_some()
                {
                    return Ok(adapter);
                }
            }
            Err(_) => break,
        }
    }
    warp_adapter(factory)
}

fn warp_adapter(factory: &IDXGIFactory4) -> Result<IDXGIAdapter1, GoldyError> {
    let warp: IDXGIAdapter = unsafe { factory.EnumWarpAdapter() }
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("EnumWarpAdapter: {e}")))?;
    warp.cast::<IDXGIAdapter1>()
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("WARP IDXGIAdapter1: {e}")))
}

impl ForeignDx12 {
    /// Offscreen `width × height` texture. No window, no swapchain.
    pub fn offscreen(
        self: &Arc<Self>,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<ForeignSurface, GoldyError> {
        let layout = PixmapLayout::tight(width, height, format);
        layout.validate()?;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.reap();
        let slot = SurfaceSlot::create(&state.device, layout)?;
        let id = state.next_id;
        state.next_id += 1;
        state.surfaces.insert(id, slot);
        Ok(ForeignSurface {
            inner: Arc::new(SurfaceHandle {
                adapter: Arc::clone(self),
                id,
            }),
        })
    }
}

impl AdapterState {
    fn reap(&mut self) {
        let ids: Vec<u32> = self
            .surfaces
            .iter()
            .filter(|(_, s)| s.dropped)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            if let Some(slot) = self.surfaces.remove(&id) {
                slot.destroy();
            }
        }
    }
}

fn heap_props(ty: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: ty,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    }
}

fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

fn alloc_mapped(
    device: &ID3D12Device,
    size: u64,
    heap: D3D12_HEAP_TYPE,
    initial: D3D12_RESOURCE_STATES,
) -> Result<MappedBuffer, GoldyError> {
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props(heap),
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc(size),
            initial,
            None,
            &mut resource,
        )
    }
    .map_err(|e| GoldyError::Backend(anyhow::anyhow!("CreateCommittedResource(buffer): {e}")))?;
    let resource = resource.ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("null mapped buffer")))?;
    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_hr = if heap == D3D12_HEAP_TYPE_UPLOAD {
        let cpu_no_read = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { resource.Map(0, Some(&cpu_no_read), Some(&mut mapped)) }
    } else {
        unsafe { resource.Map(0, None, Some(&mut mapped)) }
    };
    map_hr.map_err(|e| GoldyError::Backend(anyhow::anyhow!("Map: {e}")))?;
    Ok(MappedBuffer {
        resource,
        ptr: mapped as *mut u8,
        size: size as usize,
    })
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn wait_fence(fence: &ID3D12Fence, value: u64) -> Result<(), GoldyError> {
    let completed = unsafe { fence.GetCompletedValue() };
    if completed == u64::MAX {
        return Err(GoldyError::Backend(anyhow::anyhow!(
            "foreign DX12: device removed while waiting for fence {value}"
        )));
    }
    if completed >= value {
        return Ok(());
    }
    let event = unsafe { CreateEventA(None, false, false, None) }
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("CreateEventA: {e}")))?;
    unsafe { fence.SetEventOnCompletion(value, event) }
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("SetEventOnCompletion: {e}")))?;
    let wait = unsafe { WaitForSingleObject(event, INFINITE) };
    unsafe { CloseHandle(event) }.ok();
    if wait != WAIT_OBJECT_0 {
        return Err(GoldyError::Backend(anyhow::anyhow!("WaitForSingleObject failed")));
    }
    Ok(())
}

impl SurfaceSlot {
    fn create(device: &ID3D12Device, layout: PixmapLayout) -> Result<Self, GoldyError> {
        let dxgi = format_to_dxgi(layout.format);
        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: layout.width as u64,
            Height: layout.height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: dxgi,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut texture: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &tex_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut texture,
            )
        }
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("CreateCommittedResource(texture): {e}")))?;
        let texture = texture.ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("null texture")))?;

        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut num_rows: u32 = 0;
        let mut row_size: u64 = 0;
        let mut total_bytes: u64 = 0;
        unsafe {
            device.GetCopyableFootprints(
                &tex_desc,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut num_rows),
                Some(&mut row_size),
                Some(&mut total_bytes),
            );
        }
        let min_from_pitch = footprint
            .Offset
            .saturating_add((footprint.Footprint.RowPitch as u64).saturating_mul(layout.height as u64));
        let staging = total_bytes.max(min_from_pitch).max(layout.logical_bytes()).max(1);

        let upload = alloc_mapped(
            device,
            staging,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let readback = alloc_mapped(
            device,
            staging,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        let allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                .map_err(|e| GoldyError::Backend(anyhow::anyhow!("CreateCommandAllocator: {e}")))?;
        let list: ID3D12GraphicsCommandList =
            unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None) }
                .map_err(|e| GoldyError::Backend(anyhow::anyhow!("CreateCommandList: {e}")))?;
        unsafe { list.Close() }.map_err(|e| GoldyError::Backend(anyhow::anyhow!("Close: {e}")))?;
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("CreateFence: {e}")))?;

        Ok(Self {
            width: layout.width,
            height: layout.height,
            format: layout.format,
            generation: 1,
            texture,
            texture_state: D3D12_RESOURCE_STATE_COMMON,
            footprint,
            upload,
            readback,
            allocator,
            list,
            fence,
            fence_value: 0,
            dropped: false,
        })
    }

    fn destroy(self) {
        let _ = wait_fence(&self.fence, self.fence_value);
    }

    fn wait(&self) -> Result<(), GoldyError> {
        wait_fence(&self.fence, self.fence_value)
    }
}

fn copy_location_texture(resource: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    }
}

fn copy_location_placed(
    resource: &ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    }
}

fn pack_to_footprint(
    layout: PixmapLayout,
    src: &[u8],
    dst: &mut [u8],
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> Result<(), GoldyError> {
    layout.validate()?;
    let tight = layout.tight_row_bytes() as usize;
    let src_pitch = layout.row_pitch_bytes() as usize;
    let dst_pitch = footprint.Footprint.RowPitch as usize;
    let offset = footprint.Offset as usize;
    if src.len() < layout.staging_bytes() as usize {
        return Err(GoldyError::Backend(anyhow::anyhow!(
            "foreign DX12 blit: {} source bytes, need {}",
            src.len(),
            layout.staging_bytes()
        )));
    }
    if dst_pitch < tight {
        return Err(GoldyError::Backend(anyhow::anyhow!(
            "foreign DX12: D3D12 row pitch {dst_pitch} < tight row {tight}"
        )));
    }
    let need = offset.saturating_add(dst_pitch.saturating_mul(layout.height as usize));
    if dst.len() < need {
        return Err(GoldyError::Backend(anyhow::anyhow!(
            "foreign DX12: mapped upload is {} bytes, need {need}",
            dst.len()
        )));
    }
    for y in 0..layout.height as usize {
        let s = y * src_pitch;
        let d = offset + y * dst_pitch;
        dst[d..d + tight].copy_from_slice(&src[s..s + tight]);
    }
    Ok(())
}

impl ForeignDx12 {
    fn blit(&self, id: u32, pixels: &[u8], layout: PixmapLayout) -> Result<(), GoldyError> {
        layout.validate()?;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.reap();
        let queue = state.queue.clone();
        let slot = state
            .surfaces
            .get_mut(&id)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("foreign DX12 surface {id} is gone")))?;
        if slot.dropped {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign DX12 surface {id} has been dropped"
            )));
        }
        if layout.width != slot.width || layout.height != slot.height || layout.format != slot.format {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign DX12 blit layout {}x{} {:?} does not match surface {}x{} {:?}",
                layout.width,
                layout.height,
                layout.format,
                slot.width,
                slot.height,
                slot.format
            )));
        }
        slot.wait()?;
        let upload = unsafe { std::slice::from_raw_parts_mut(slot.upload.ptr, slot.upload.size) };
        pack_to_footprint(layout, pixels, upload, &slot.footprint)?;

        unsafe { slot.allocator.Reset() }.map_err(|e| GoldyError::Backend(anyhow::anyhow!("Reset allocator: {e}")))?;
        unsafe { slot.list.Reset(&slot.allocator, None) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("Reset list: {e}")))?;

        let to_dst = transition(&slot.texture, slot.texture_state, D3D12_RESOURCE_STATE_COPY_DEST);
        unsafe { slot.list.ResourceBarrier(&[to_dst]) };
        let src_loc = copy_location_placed(&slot.upload.resource, slot.footprint);
        let dst_loc = copy_location_texture(&slot.texture);
        unsafe { slot.list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None) };

        let to_src = transition(
            &slot.texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        unsafe { slot.list.ResourceBarrier(&[to_src]) };
        let rb_loc = copy_location_placed(&slot.readback.resource, slot.footprint);
        let tex_src = copy_location_texture(&slot.texture);
        unsafe { slot.list.CopyTextureRegion(&rb_loc, 0, 0, 0, &tex_src, None) };
        unsafe { slot.list.Close() }.map_err(|e| GoldyError::Backend(anyhow::anyhow!("Close: {e}")))?;

        let list: ID3D12CommandList = slot
            .list
            .cast()
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("ID3D12CommandList: {e}")))?;
        unsafe { queue.ExecuteCommandLists(&[Some(list)]) };
        slot.fence_value += 1;
        unsafe { queue.Signal(&slot.fence, slot.fence_value) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("Signal: {e}")))?;
        slot.texture_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
        Ok(())
    }

    fn snapshot(&self, id: u32, layout: PixmapLayout) -> Result<Vec<u8>, GoldyError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let slot = state
            .surfaces
            .get_mut(&id)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("foreign DX12 surface {id} is gone")))?;
        slot.wait()?;
        let staging = unsafe { std::slice::from_raw_parts(slot.readback.ptr, slot.readback.size) };
        let offset = slot.footprint.Offset as usize;
        let d3d_layout = PixmapLayout {
            width: layout.width,
            height: layout.height,
            format: layout.format,
            row_pitch: u64::from(slot.footprint.Footprint.RowPitch),
        };
        let mut tight = vec![0u8; layout.logical_bytes() as usize];
        d3d_layout.unpack_into(&staging[offset..], &mut tight)?;
        Ok(tight)
    }

    fn generation(&self, id: u32) -> u64 {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.surfaces.get(&id).map(|s| s.generation).unwrap_or(0)
    }

    fn size(&self, id: u32) -> (u32, u32) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.surfaces.get(&id).map(|s| (s.width, s.height)).unwrap_or((0, 0))
    }

    fn release(&self, id: u32) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = state.surfaces.get_mut(&id) {
            slot.dropped = true;
        }
        state.reap();
    }
}

impl PixelSink for ForeignSurface {
    fn blit(&self, pixels: &[u8], layout: PixmapLayout) -> Result<(), GoldyError> {
        self.inner.adapter.blit(self.inner.id, pixels, layout)
    }

    fn generation(&self) -> u64 {
        self.inner.adapter.generation(self.inner.id)
    }

    fn size(&self) -> (u32, u32) {
        self.inner.adapter.size(self.inner.id)
    }
}

impl ForeignSurface {
    /// Tightly packed pixels after the last blit (GPU copy-back).
    pub fn snapshot(&self, layout: PixmapLayout) -> Result<Vec<u8>, GoldyError> {
        self.inner.adapter.snapshot(self.inner.id, layout)
    }
}
