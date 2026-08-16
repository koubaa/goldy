//! Shared D3D12↔CUDA textures and present copy helpers.
//!
//! Graphics companion allocates exportable UAV textures; CUDA imports them once via
//! `cuImportExternalMemory` + mapped mipmapped arrays. Surface scratch and the
//! swapchain are both RGBA8 so present is a single `CopyResource` (no conversion
//! dispatch). Flip-model backbuffers still cannot be CUDA-imported, so the shared
//! scratch remains the write target.

use super::dx12_companion::Dx12Companion;
use super::texture::{format_info, CudaTextureResource};
use crate::types::{TextureFlags, TextureFormat, TextureKind};
use anyhow::{bail, Context as _, Result};
use cudarc::driver::{sys, CudaContext};
use std::sync::Arc;
use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// Goldy surface format exposed to schemes (`DirectSpatial<float4>` packs via
/// `Float4Rgba8Unorm` specialization).
pub(super) const SURFACE_COMPUTE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// DXGI swapchain format — matches [`SURFACE_COMPUTE_FORMAT`] for direct copy.
pub(super) const SWAPCHAIN_DXGI_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

/// One shareable RGBA8 UAV texture with a CUDA surface view.
///
/// Field order is load-bearing: Rust drops fields in declaration order, so
/// [`cuda_texture`] (tex/surf objects) must precede [`import`] (mapped mipmapped
/// array / external memory), which must precede [`d3d12_resource`].
pub(super) struct SharedScratchTexture {
    pub width: u32,
    pub height: u32,
    pub cuda_texture: Arc<CudaTextureResource>,
    /// Keeps the external memory + mipmapped array alive for the CUDA texture view.
    #[allow(dead_code)]
    pub import: CudaImportedTexture,
    pub d3d12_resource: ID3D12Resource,
    #[allow(dead_code)]
    pub allocation_size: u64,
}

pub(in crate::backend) struct CudaImportedTexture {
    cuda_ctx: Arc<CudaContext>,
    external_memory: sys::CUexternalMemory,
    mipmapped: sys::CUmipmappedArray,
    /// Level-0 array borrowed from `mipmapped` — do not `cuArrayDestroy`.
    level0: sys::CUarray,
}

impl CudaImportedTexture {
    /// Borrowed level-0 array (owned by `mipmapped`; do not destroy).
    pub(super) fn level0(&self) -> sys::CUarray {
        self.level0
    }
}

impl Drop for CudaImportedTexture {
    fn drop(&mut self) {
        let _ = self.cuda_ctx.bind_to_thread();
        // Destroy CUDA views before the graphics resource (caller drops d3d12 last).
        let mip = std::mem::replace(&mut self.mipmapped, std::ptr::null_mut());
        if !mip.is_null() {
            let _ = unsafe { sys::cuMipmappedArrayDestroy(mip) };
        }
        self.level0 = std::ptr::null_mut();
        let ext = std::mem::replace(&mut self.external_memory, std::ptr::null_mut());
        if !ext.is_null() {
            let _ = unsafe { sys::cuDestroyExternalMemory(ext) };
        }
    }
}

// SAFETY: created/destroyed under backend lock with CUDA context bound.
unsafe impl Send for CudaImportedTexture {}
unsafe impl Sync for CudaImportedTexture {}

impl SharedScratchTexture {
    pub fn create(
        companion: &Dx12Companion,
        cuda_ctx: &Arc<CudaContext>,
        width: u32,
        height: u32,
        storage_slot: u32,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("CUDA/DX12: scratch dimensions must be non-zero");
        }
        let _ = format_info(SURFACE_COMPUTE_FORMAT)?;

        let (d3d12_resource, allocation_size) = create_shared_texture(
            &companion.device,
            width,
            height,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        )?;

        let handle: HANDLE = unsafe {
            companion
                .device
                .CreateSharedHandle(&d3d12_resource, None, GENERIC_ALL.0, None)
        }
        .context("CUDA/DX12: CreateSharedHandle(texture) failed")?;

        let import = import_shared_texture(cuda_ctx, handle, allocation_size, width, height, SURFACE_COMPUTE_FORMAT)?;
        unsafe {
            let _ = CloseHandle(handle);
        }

        let cuda_texture = CudaTextureResource::from_imported_array(
            cuda_ctx,
            import.level0,
            width,
            height,
            SURFACE_COMPUTE_FORMAT,
            TextureKind::Direct,
            TextureFlags::empty(),
            Some(storage_slot),
            None,
        )?;

        init_resource_state(companion, &d3d12_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;

        Ok(Self {
            width,
            height,
            cuda_texture,
            import,
            d3d12_resource,
            allocation_size,
        })
    }

    /// Rebuild the CUDA texture view with a new registry slot (scratch pool reuse).
    pub fn retarget_storage_slot(
        &mut self,
        cuda_ctx: &Arc<CudaContext>,
        storage_slot: u32,
    ) -> Result<Arc<CudaTextureResource>> {
        let cuda_texture = CudaTextureResource::from_imported_array(
            cuda_ctx,
            self.import.level0(),
            self.width,
            self.height,
            SURFACE_COMPUTE_FORMAT,
            TextureKind::Direct,
            TextureFlags::empty(),
            Some(storage_slot),
            None,
        )?;
        self.cuda_texture = Arc::clone(&cuda_texture);
        Ok(cuda_texture)
    }
}

/// Shareable DEFAULT-heap buffer imported into CUDA as a device pointer.
///
/// Field order is load-bearing: [`import`] (external memory + mapped ptr) must be
/// dropped before [`d3d12_resource`]. The CUDA [`CudaSlice`](cudarc::driver::CudaSlice)
/// that wraps the mapped pointer must be [`leak`](cudarc::driver::CudaSlice::leak)ed
/// before this struct is dropped (see [`super::leak_shared_buffer_slice`]).
pub(crate) struct SharedBufferBacking {
    pub import: CudaImportedBuffer,
    pub d3d12_resource: ID3D12Resource,
    pub size: u64,
    /// Companion fence value signaled after the latest CUDA write to this buffer.
    /// Only set when the write shared a submit with scratch present (async handoff).
    pub last_cuda_fence: std::sync::atomic::AtomicU64,
    /// Companion fence value after the latest DX12 IA draw that read this buffer.
    /// CUDA must Wait this before rewriting Shared-primary memory.
    pub last_dx12_ia_fence: std::sync::atomic::AtomicU64,
    /// Deposit-only Shared writes: raster must flush/sync CUDA before IA (no fence Signal).
    pub pending_host_sync: std::sync::atomic::AtomicBool,
}

// SAFETY: COM + CUDA handles used under Goldy's backend lock / submission worker.
unsafe impl Send for SharedBufferBacking {}
unsafe impl Sync for SharedBufferBacking {}

pub(crate) struct CudaImportedBuffer {
    cuda_ctx: Arc<CudaContext>,
    external_memory: sys::CUexternalMemory,
    /// Mapped via `cuExternalMemoryGetMappedBuffer` — do not `cuMemFree`.
    pub device_ptr: sys::CUdeviceptr,
    #[allow(dead_code)]
    pub size: u64,
}

impl Drop for CudaImportedBuffer {
    fn drop(&mut self) {
        let _ = self.cuda_ctx.bind_to_thread();
        self.device_ptr = 0;
        let ext = std::mem::replace(&mut self.external_memory, std::ptr::null_mut());
        if !ext.is_null() {
            let _ = unsafe { sys::cuDestroyExternalMemory(ext) };
        }
    }
}

// SAFETY: see [`CudaImportedTexture`].
unsafe impl Send for CudaImportedBuffer {}
unsafe impl Sync for CudaImportedBuffer {}

/// Create a shareable D3D12 buffer and import it into CUDA as a device pointer.
///
/// The returned [`SharedBufferBacking`] owns the mapping. Callers must not
/// `cuMemFree` [`CudaImportedBuffer::device_ptr`].
pub(super) fn create_shared_buffer_backing(
    companion: &Dx12Companion,
    cuda_ctx: &Arc<CudaContext>,
    stream: &Arc<cudarc::driver::CudaStream>,
    size: u64,
) -> Result<SharedBufferBacking> {
    let size = size.max(4);
    let (d3d12_resource, allocation_size) = create_shared_buffer_resource(&companion.device, size)?;

    let handle: HANDLE = unsafe {
        companion
            .device
            .CreateSharedHandle(&d3d12_resource, None, GENERIC_ALL.0, None)
    }
    .context("CUDA/DX12: CreateSharedHandle(buffer) failed")?;

    let import = import_shared_buffer(cuda_ctx, handle, allocation_size)?;
    unsafe {
        let _ = CloseHandle(handle);
    }

    // Zero via driver API (not a CudaSlice) so we never risk cuMemFree on the mapping.
    unsafe { cudarc::driver::result::memset_d8_async(import.device_ptr, 0, size as usize, stream.cu_stream()) }
        .context("CUDA/DX12: cuMemsetD8Async on shared buffer failed")?;
    stream
        .synchronize()
        .context("CUDA/DX12: synchronize after shared buffer memset")?;

    Ok(SharedBufferBacking {
        import,
        d3d12_resource,
        size,
        last_cuda_fence: std::sync::atomic::AtomicU64::new(0),
        last_dx12_ia_fence: std::sync::atomic::AtomicU64::new(0),
        pending_host_sync: std::sync::atomic::AtomicBool::new(false),
    })
}

/// Create a shareable D3D12 buffer, import it, and wrap the mapping in a [`CudaSlice`].
///
/// The slice **must** be [`leak`](cudarc::driver::CudaSlice::leak)ed before
/// [`SharedBufferBacking`] is dropped.
#[allow(dead_code)] // available for experiments with kernel-direct shared buffers
pub(super) fn create_shared_buffer(
    companion: &Dx12Companion,
    cuda_ctx: &Arc<CudaContext>,
    stream: &Arc<cudarc::driver::CudaStream>,
    size: u64,
) -> Result<(cudarc::driver::CudaSlice<u8>, SharedBufferBacking)> {
    let backing = create_shared_buffer_backing(companion, cuda_ctx, stream, size)?;
    // SAFETY: mapped external allocation; freed only via destroying `backing.import`.
    let slice = unsafe { stream.upgrade_device_ptr::<u8>(backing.import.device_ptr, size.max(4) as usize) };
    Ok((slice, backing))
}

pub(super) fn create_shared_buffer_resource(device: &ID3D12Device, size: u64) -> Result<(ID3D12Resource, u64)> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size.max(4),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_SHARED,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
    }
    .context("CUDA/DX12: CreateCommittedResource(SHARED buffer) failed")?;
    let resource = resource.context("CUDA/DX12: null shared buffer")?;
    let alloc_info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
    Ok((resource, alloc_info.SizeInBytes.max(size.max(4))))
}

pub(super) fn import_shared_buffer(
    cuda_ctx: &Arc<CudaContext>,
    handle: HANDLE,
    size: u64,
) -> Result<CudaImportedBuffer> {
    cuda_ctx.bind_to_thread().context("CUDA/DX12: bind for buffer import")?;

    let mem_desc = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
        handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: handle.0,
                name: std::ptr::null(),
            },
        },
        size,
        flags: sys::CUDA_EXTERNAL_MEMORY_DEDICATED,
        reserved: [0; 16],
    };

    let mut external_memory: sys::CUexternalMemory = std::ptr::null_mut();
    let r = unsafe { sys::cuImportExternalMemory(&mut external_memory, &mem_desc) };
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA: cuImportExternalMemory(D3D12 buffer) failed: {r:?}");
    }

    let buf_desc = sys::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
        offset: 0,
        size,
        flags: 0,
        reserved: [0; 16],
    };
    let mut device_ptr: sys::CUdeviceptr = 0;
    let r = unsafe { sys::cuExternalMemoryGetMappedBuffer(&mut device_ptr, external_memory, &buf_desc) };
    if r != sys::CUresult::CUDA_SUCCESS {
        let _ = unsafe { sys::cuDestroyExternalMemory(external_memory) };
        bail!("CUDA: cuExternalMemoryGetMappedBuffer failed: {r:?}");
    }

    Ok(CudaImportedBuffer {
        cuda_ctx: Arc::clone(cuda_ctx),
        external_memory,
        device_ptr,
        size,
    })
}

pub(super) fn create_shared_texture(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
) -> Result<(ID3D12Resource, u64)> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_SHARED,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
    }
    .context("CUDA/DX12: CreateCommittedResource(SHARED) failed")?;
    let resource = resource.context("CUDA/DX12: null shared texture")?;
    let alloc_info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
    Ok((resource, alloc_info.SizeInBytes))
}

pub(super) fn import_shared_texture(
    cuda_ctx: &Arc<CudaContext>,
    handle: HANDLE,
    size: u64,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<CudaImportedTexture> {
    cuda_ctx
        .bind_to_thread()
        .context("CUDA/DX12: bind for texture import")?;
    let info = format_info(format)?;

    let mem_desc = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
        handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: handle.0,
                name: std::ptr::null(),
            },
        },
        size,
        flags: sys::CUDA_EXTERNAL_MEMORY_DEDICATED,
        reserved: [0; 16],
    };

    let mut external_memory: sys::CUexternalMemory = std::ptr::null_mut();
    let r = unsafe { sys::cuImportExternalMemory(&mut external_memory, &mem_desc) };
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA: cuImportExternalMemory(D3D12_RESOURCE) failed: {r:?}");
    }

    let array_desc = sys::CUDA_ARRAY3D_DESCRIPTOR {
        Width: width as usize,
        Height: height as usize,
        Depth: 0,
        Format: info.array_format,
        NumChannels: info.num_channels,
        Flags: sys::CUDA_ARRAY3D_SURFACE_LDST,
    };
    let mip_desc = sys::CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC {
        offset: 0,
        arrayDesc: array_desc,
        numLevels: 1,
        reserved: [0; 16],
    };

    let mut mipmapped: sys::CUmipmappedArray = std::ptr::null_mut();
    let r = unsafe { sys::cuExternalMemoryGetMappedMipmappedArray(&mut mipmapped, external_memory, &mip_desc) };
    if r != sys::CUresult::CUDA_SUCCESS {
        let _ = unsafe { sys::cuDestroyExternalMemory(external_memory) };
        bail!("CUDA: cuExternalMemoryGetMappedMipmappedArray failed: {r:?}");
    }

    let mut level0: sys::CUarray = std::ptr::null_mut();
    let r = unsafe { sys::cuMipmappedArrayGetLevel(&mut level0, mipmapped, 0) };
    if r != sys::CUresult::CUDA_SUCCESS {
        let _ = unsafe { sys::cuMipmappedArrayDestroy(mipmapped) };
        let _ = unsafe { sys::cuDestroyExternalMemory(external_memory) };
        bail!("CUDA: cuMipmappedArrayGetLevel failed: {r:?}");
    }

    Ok(CudaImportedTexture {
        cuda_ctx: Arc::clone(cuda_ctx),
        external_memory,
        mipmapped,
        level0,
    })
}

pub(super) fn init_resource_state(
    companion: &Dx12Companion,
    resource: &ID3D12Resource,
    after: D3D12_RESOURCE_STATES,
) -> Result<()> {
    companion.submit_init_list()?;
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_COMMON,
                StateAfter: after,
            }),
        },
    };
    unsafe { companion.init_list.ResourceBarrier(&[barrier]) };
    companion.finish_init_list()
}

#[cfg(test)]
mod tests {
    use super::super::dx12_companion::Dx12Companion;
    use super::*;

    #[test]
    fn shared_scratch_import_or_skip() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let Ok(companion) = Dx12Companion::create(&ctx) else {
            eprintln!("skip: no DX12 companion");
            return;
        };
        let scratch = SharedScratchTexture::create(&companion, &ctx, 64, 48, 0).expect("shared scratch create/import");
        assert_eq!(scratch.width, 64);
        assert_eq!(scratch.height, 48);
        assert_eq!(scratch.cuda_texture.format, SURFACE_COMPUTE_FORMAT);
        // Surf object must be creatable for DirectSpatial writes.
        scratch
            .cuda_texture
            .surf_object()
            .expect("surface object on imported array");
        let _ = companion.wait_idle();
    }

    #[test]
    fn shared_buffer_import_or_skip() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let Ok(companion) = Dx12Companion::create(&ctx) else {
            eprintln!("skip: no DX12 companion");
            return;
        };
        let stream = ctx.default_stream();
        let backing =
            create_shared_buffer_backing(&companion, &ctx, &stream, 256).expect("shared buffer create/import");
        assert_eq!(backing.size, 256);
        assert_ne!(backing.import.device_ptr, 0);
        drop(backing);
        let _ = companion.wait_idle();
    }
}

/// Where the present color source sits before the copy.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PresentColorSrcState {
    /// CUDA-written imported scratch (UAV).
    UnorderedAccess,
    /// DX12 raster target left in COMMON after `render_to_target`.
    Common,
}

/// Record color → backbuffer `CopyResource` + present barriers on a DIRECT list.
///
/// Flip-model backbuffers are DWM-shared and must be written on the swapchain's
/// DIRECT queue. `color_src` must be RGBA8 matching [`SWAPCHAIN_DXGI_FORMAT`].
pub(super) fn record_present_copy(
    list: &ID3D12GraphicsCommandList,
    color_src: &ID3D12Resource,
    color_src_state: PresentColorSrcState,
    color_src_format: TextureFormat,
    backbuffer: &ID3D12Resource,
    backbuffer_from_common: bool,
) -> Result<()> {
    if color_src_format != SURFACE_COMPUTE_FORMAT {
        bail!(
            "CUDA/DX12: present CopyResource requires {SURFACE_COMPUTE_FORMAT:?} source              (got {color_src_format:?})"
        );
    }
    let src_before = match color_src_state {
        PresentColorSrcState::UnorderedAccess => D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        PresentColorSrcState::Common => D3D12_RESOURCE_STATE_COMMON,
    };
    let src_after = src_before;
    let backbuffer_before = if backbuffer_from_common {
        D3D12_RESOURCE_STATE_COMMON
    } else {
        D3D12_RESOURCE_STATE_PRESENT
    };

    let mut barriers = Vec::with_capacity(3);
    if matches!(color_src_state, PresentColorSrcState::UnorderedAccess) {
        barriers.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(color_src) },
                }),
            },
        });
    }
    barriers.push(transition(color_src, src_before, D3D12_RESOURCE_STATE_COPY_SOURCE));
    barriers.push(transition(
        backbuffer,
        backbuffer_before,
        D3D12_RESOURCE_STATE_COPY_DEST,
    ));
    unsafe { list.ResourceBarrier(&barriers) };
    unsafe { list.CopyResource(backbuffer, color_src) };

    let b_back = transition(backbuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
    let b_src = transition(color_src, D3D12_RESOURCE_STATE_COPY_SOURCE, src_after);
    unsafe { list.ResourceBarrier(&[b_back, b_src]) };
    Ok(())
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
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}
