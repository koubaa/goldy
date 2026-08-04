//! CUDA array / texture-object / surface-object primitives.
//!
//! Goldy textures map to a single [`sys::CUarray`] plus lazily-created
//! [`sys::CUtexObject`] (sampled) and [`sys::CUsurfObject`] (storage) views.
//! Sampler state is baked into each texture object — CUDA has no separate
//! sampler handle.

use crate::types::{AddressMode, FilterMode, SamplerDesc, TextureFlags, TextureFormat, TextureKind};
use anyhow::{bail, Context as _, Result};
use cudarc::driver::{sys, CudaContext, CudaStream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Format metadata for a Goldy [`TextureFormat`] on CUDA.
#[derive(Clone, Copy, Debug)]
pub(super) struct CudaFormatInfo {
    pub array_format: sys::CUarray_format,
    pub num_channels: u32,
    #[allow(dead_code)]
    pub bytes_per_pixel: u32,
    pub srgb: bool,
}

/// Map a Goldy texture format to a CUDA array format, or reject unsupported ones.
pub(super) fn format_info(format: TextureFormat) -> Result<CudaFormatInfo> {
    let bytes_per_pixel = format.bytes_per_pixel();
    let info = match format {
        TextureFormat::R8Unorm => CudaFormatInfo {
            array_format: sys::CUarray_format::CU_AD_FORMAT_UNORM_INT8X1,
            num_channels: 1,
            bytes_per_pixel,
            srgb: false,
        },
        TextureFormat::Rg8Unorm => CudaFormatInfo {
            array_format: sys::CUarray_format::CU_AD_FORMAT_UNORM_INT8X2,
            num_channels: 2,
            bytes_per_pixel,
            srgb: false,
        },
        TextureFormat::Rgba8Unorm => CudaFormatInfo {
            array_format: sys::CUarray_format::CU_AD_FORMAT_UNORM_INT8X4,
            num_channels: 4,
            bytes_per_pixel,
            srgb: false,
        },
        TextureFormat::Rgba8UnormSrgb => CudaFormatInfo {
            array_format: sys::CUarray_format::CU_AD_FORMAT_UNORM_INT8X4,
            num_channels: 4,
            bytes_per_pixel,
            srgb: true,
        },
        TextureFormat::Rgba16Float => CudaFormatInfo {
            array_format: sys::CUarray_format::CU_AD_FORMAT_HALF,
            num_channels: 4,
            bytes_per_pixel,
            srgb: false,
        },
        TextureFormat::Rgba32Float => CudaFormatInfo {
            array_format: sys::CUarray_format::CU_AD_FORMAT_FLOAT,
            num_channels: 4,
            bytes_per_pixel,
            srgb: false,
        },
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => {
            bail!(
                "CUDA textures do not support {:?}: CUDA arrays have no BGRA channel swizzle \
                 matching Goldy's interpretation",
                format
            );
        }
    };
    Ok(info)
}

/// True when a `DirectSpatial<{element}>` shader parameter can write `format`.
///
/// CUDA surface stores are raw typed writes (no DX12-style typed UAV conversion),
/// so the element type byte size must match the texture format:
/// - `float4` ↔ [`TextureFormat::Rgba32Float`] (16 bytes)
/// - `half4` ↔ [`TextureFormat::Rgba16Float`] (8 bytes)
/// - `uint8_t4` / `vector<uint8_t, 4>` ↔ [`TextureFormat::Rgba8Unorm`] (4 bytes)
///
/// Slang does not define CUDA's `uchar4` alias; use `uint8_t4`.
pub(super) fn storage_shader_compatible(element: &str, format: TextureFormat) -> bool {
    let element = element.trim();
    matches!(
        (element, format),
        ("float4", TextureFormat::Rgba32Float)
            | ("half4", TextureFormat::Rgba16Float)
            | ("uint8_t4", TextureFormat::Rgba8Unorm)
            | ("vector<uint8_t, 4>", TextureFormat::Rgba8Unorm)
            | ("vector<uint8_t,4>", TextureFormat::Rgba8Unorm)
    )
}

/// Sampler configuration CUDA can represent in a single texture object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct CudaSamplerKey {
    address_u: sys::CUaddress_mode,
    address_v: sys::CUaddress_mode,
    address_w: sys::CUaddress_mode,
    filter: sys::CUfilter_mode,
    mip_filter: sys::CUfilter_mode,
    max_anisotropy: u32,
    lod_min_bits: u32,
    lod_max_bits: u32,
}

impl CudaSamplerKey {
    /// Validate and map a Goldy [`SamplerDesc`] into CUDA texture-object settings.
    pub(super) fn from_desc(desc: &SamplerDesc) -> Result<Self> {
        if desc.compare.is_some() {
            bail!("CUDA textures do not support comparison sampling");
        }
        if desc.mag_filter != desc.min_filter {
            bail!(
                "CUDA texture objects require mag_filter == min_filter \
                 (got mag={:?}, min={:?})",
                desc.mag_filter,
                desc.min_filter
            );
        }
        Ok(Self {
            address_u: map_address(desc.address_mode_u),
            address_v: map_address(desc.address_mode_v),
            address_w: map_address(desc.address_mode_w),
            filter: map_filter(desc.mag_filter),
            mip_filter: map_filter(desc.mipmap_filter),
            max_anisotropy: desc.max_anisotropy.max(1.0).round() as u32,
            lod_min_bits: desc.lod_min_clamp.to_bits(),
            lod_max_bits: desc.lod_max_clamp.to_bits(),
        })
    }

    pub(super) fn nearest_clamp() -> Self {
        Self::from_desc(&SamplerDesc::default()).expect("default sampler is CUDA-compatible")
    }

    fn to_texture_desc(self, srgb: bool) -> sys::CUDA_TEXTURE_DESC {
        let mut tex_desc: sys::CUDA_TEXTURE_DESC = unsafe { std::mem::zeroed() };
        tex_desc.addressMode = [self.address_u, self.address_v, self.address_w];
        tex_desc.filterMode = self.filter;
        tex_desc.mipmapFilterMode = self.mip_filter;
        tex_desc.maxAnisotropy = self.max_anisotropy;
        tex_desc.minMipmapLevelClamp = f32::from_bits(self.lod_min_bits);
        tex_desc.maxMipmapLevelClamp = f32::from_bits(self.lod_max_bits);
        tex_desc.flags = sys::CU_TRSF_NORMALIZED_COORDINATES;
        if srgb {
            tex_desc.flags |= sys::CU_TRSF_SRGB;
        }
        tex_desc
    }
}

fn map_address(mode: AddressMode) -> sys::CUaddress_mode {
    match mode {
        AddressMode::ClampToEdge => sys::CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP,
        AddressMode::Repeat => sys::CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP,
        AddressMode::MirrorRepeat => sys::CUaddress_mode::CU_TR_ADDRESS_MODE_MIRROR,
    }
}

fn map_filter(mode: FilterMode) -> sys::CUfilter_mode {
    match mode {
        FilterMode::Nearest => sys::CUfilter_mode::CU_TR_FILTER_MODE_POINT,
        FilterMode::Linear => sys::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
    }
}

fn check_cu(result: sys::CUresult, what: &str) -> Result<()> {
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        bail!("CUDA: {what} failed: {result:?}")
    }
}

/// Owning wrapper around a [`sys::CUarray`].
pub(super) struct CudaArray {
    ctx: Arc<CudaContext>,
    array: sys::CUarray,
    /// When false, the array is borrowed from an imported mipmapped array and must
    /// not be destroyed here (see [`CudaTextureResource::from_imported_array`]).
    owns_array: bool,
}

impl CudaArray {
    pub(super) fn create(
        ctx: &Arc<CudaContext>,
        width: u32,
        height: u32,
        format: TextureFormat,
        need_surface: bool,
    ) -> Result<Self> {
        ctx.bind_to_thread().context("CUDA: bind context for array create")?;
        let info = format_info(format)?;
        let array = if need_surface {
            let desc = sys::CUDA_ARRAY3D_DESCRIPTOR {
                Width: width as usize,
                Height: height as usize,
                Depth: 0,
                Format: info.array_format,
                NumChannels: info.num_channels,
                Flags: sys::CUDA_ARRAY3D_SURFACE_LDST,
            };
            let mut array: sys::CUarray = std::ptr::null_mut();
            check_cu(unsafe { sys::cuArray3DCreate_v2(&mut array, &desc) }, "cuArray3DCreate")?;
            array
        } else {
            let desc = sys::CUDA_ARRAY_DESCRIPTOR {
                Width: width as usize,
                Height: height as usize,
                Format: info.array_format,
                NumChannels: info.num_channels,
            };
            let mut array: sys::CUarray = std::ptr::null_mut();
            check_cu(unsafe { sys::cuArrayCreate_v2(&mut array, &desc) }, "cuArrayCreate")?;
            array
        };
        Ok(Self {
            ctx: Arc::clone(ctx),
            array,
            owns_array: true,
        })
    }

    /// Wrap a level-0 array borrowed from `cuExternalMemoryGetMappedMipmappedArray`.
    pub(super) fn from_imported(ctx: &Arc<CudaContext>, array: sys::CUarray) -> Self {
        Self {
            ctx: Arc::clone(ctx),
            array,
            owns_array: false,
        }
    }

    pub(super) fn raw(&self) -> sys::CUarray {
        self.array
    }

    /// True when this array is borrowed from `cuImportExternalMemory` (D3D12 interop).
    pub(super) fn is_imported(&self) -> bool {
        !self.owns_array
    }
}

impl Drop for CudaArray {
    fn drop(&mut self) {
        if !self.owns_array {
            return;
        }
        let _ = self.ctx.bind_to_thread();
        let array = std::mem::replace(&mut self.array, std::ptr::null_mut());
        if !array.is_null() {
            let _ = unsafe { sys::cuArrayDestroy(array) };
        }
    }
}

// SAFETY: created/destroyed with the owning CUDA context bound; shared only under
// Goldy's backend lock / submission-worker serialization.
unsafe impl Send for CudaArray {}
unsafe impl Sync for CudaArray {}

/// GPU texture resource: CUDA array + cached tex/surf objects.
pub(crate) struct CudaTextureResource {
    pub(super) ctx: Arc<CudaContext>,
    array: Arc<CudaArray>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub kind: TextureKind,
    #[allow(dead_code)]
    pub flags: TextureFlags,
    /// Registry key for storage / Direct access (UAV-equivalent).
    pub storage_slot: Option<u32>,
    /// Registry key for sampled / Interpolated access (SRV-equivalent).
    pub sampled_slot: Option<u32>,
    srgb: bool,
    tex_objects: Mutex<HashMap<CudaSamplerKey, sys::CUtexObject>>,
    surf_object: Mutex<Option<sys::CUsurfObject>>,
}

impl CudaTextureResource {
    pub(super) fn create(
        ctx: &Arc<CudaContext>,
        width: u32,
        height: u32,
        format: TextureFormat,
        kind: TextureKind,
        flags: TextureFlags,
        storage_slot: Option<u32>,
        sampled_slot: Option<u32>,
    ) -> Result<Arc<Self>> {
        if width == 0 || height == 0 {
            bail!("CUDA: texture dimensions must be non-zero");
        }
        if flags.contains(TextureFlags::RENDER_TARGET) {
            bail!("CUDA compute-only backend does not support RENDER_TARGET textures");
        }
        let info = format_info(format)?;
        let need_surface = matches!(kind, TextureKind::Direct | TextureKind::DirectInterpolated);
        let array = Arc::new(CudaArray::create(ctx, width, height, format, need_surface)?);
        Ok(Arc::new(Self {
            ctx: Arc::clone(ctx),
            array,
            width,
            height,
            format,
            kind,
            flags,
            storage_slot,
            sampled_slot,
            srgb: info.srgb,
            tex_objects: Mutex::new(HashMap::new()),
            surf_object: Mutex::new(None),
        }))
    }

    /// Build a texture view over an imported D3D12/external level-0 CUDA array.
    ///
    /// The caller retains ownership of the external memory / mipmapped array and must
    /// outlive this resource. Tex/surf objects are still destroyed on drop.
    #[cfg(all(feature = "graphics", feature = "dx12", target_os = "windows"))]
    pub(super) fn from_imported_array(
        ctx: &Arc<CudaContext>,
        array: sys::CUarray,
        width: u32,
        height: u32,
        format: TextureFormat,
        kind: TextureKind,
        flags: TextureFlags,
        storage_slot: Option<u32>,
        sampled_slot: Option<u32>,
    ) -> Result<Arc<Self>> {
        if array.is_null() {
            bail!("CUDA: imported array is null");
        }
        if width == 0 || height == 0 {
            bail!("CUDA: texture dimensions must be non-zero");
        }
        let info = format_info(format)?;
        Ok(Arc::new(Self {
            ctx: Arc::clone(ctx),
            array: Arc::new(CudaArray::from_imported(ctx, array)),
            width,
            height,
            format,
            kind,
            flags,
            storage_slot,
            sampled_slot,
            srgb: info.srgb,
            tex_objects: Mutex::new(HashMap::new()),
            surf_object: Mutex::new(None),
        }))
    }

    pub(super) fn array(&self) -> sys::CUarray {
        self.array.raw()
    }

    /// True when backed by D3D12-imported external memory (not graph-capture safe).
    pub(super) fn is_imported(&self) -> bool {
        self.array.is_imported()
    }

    pub(super) fn bytes_per_pixel(&self) -> u32 {
        self.format.bytes_per_pixel()
    }

    #[allow(dead_code)]
    pub(super) fn byte_size(&self) -> u64 {
        self.width as u64 * self.height as u64 * self.bytes_per_pixel() as u64
    }

    /// Lazily create (or reuse) a texture object with the given sampler settings.
    pub(super) fn tex_object(&self, sampler: CudaSamplerKey) -> Result<sys::CUtexObject> {
        let mut cache = self.tex_objects.lock().unwrap();
        if let Some(&existing) = cache.get(&sampler) {
            return Ok(existing);
        }
        self.ctx
            .bind_to_thread()
            .context("CUDA: bind context for tex object create")?;
        let mut res_desc: sys::CUDA_RESOURCE_DESC = unsafe { std::mem::zeroed() };
        res_desc.resType = sys::CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = self.array.raw();
        let tex_desc = sampler.to_texture_desc(self.srgb);
        let mut tex_object: sys::CUtexObject = 0;
        check_cu(
            unsafe { sys::cuTexObjectCreate(&mut tex_object, &res_desc, &tex_desc, std::ptr::null()) },
            "cuTexObjectCreate",
        )?;
        cache.insert(sampler, tex_object);
        Ok(tex_object)
    }

    /// Lazily create (or reuse) a surface object for storage access.
    pub(super) fn surf_object(&self) -> Result<sys::CUsurfObject> {
        if !matches!(self.kind, TextureKind::Direct | TextureKind::DirectInterpolated) {
            bail!("CUDA: texture kind {:?} has no storage (surface) view", self.kind);
        }
        let mut guard = self.surf_object.lock().unwrap();
        if let Some(existing) = *guard {
            return Ok(existing);
        }
        self.ctx
            .bind_to_thread()
            .context("CUDA: bind context for surf object create")?;
        let mut res_desc: sys::CUDA_RESOURCE_DESC = unsafe { std::mem::zeroed() };
        res_desc.resType = sys::CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = self.array.raw();
        let mut surf: sys::CUsurfObject = 0;
        check_cu(
            unsafe { sys::cuSurfObjectCreate(&mut surf, &res_desc) },
            "cuSurfObjectCreate",
        )?;
        *guard = Some(surf);
        Ok(surf)
    }
}

impl Drop for CudaTextureResource {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if let Ok(mut cache) = self.tex_objects.lock() {
            for (_, tex) in cache.drain() {
                let _ = unsafe { sys::cuTexObjectDestroy(tex) };
            }
        }
        if let Ok(mut surf) = self.surf_object.lock() {
            if let Some(surf) = surf.take() {
                let _ = unsafe { sys::cuSurfObjectDestroy(surf) };
            }
        }
    }
}

// SAFETY: see [`CudaArray`].
unsafe impl Send for CudaTextureResource {}
unsafe impl Sync for CudaTextureResource {}

fn validate_region(tex: &CudaTextureResource, x: u32, y: u32, width: u32, height: u32, data_len: usize) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("CUDA: texture region must be non-empty");
    }
    if x.checked_add(width).map(|end| end > tex.width).unwrap_or(true)
        || y.checked_add(height).map(|end| end > tex.height).unwrap_or(true)
    {
        bail!(
            "CUDA: texture region ({x},{y},{width}x{height}) exceeds texture {}x{}",
            tex.width,
            tex.height
        );
    }
    let expected = width as usize * height as usize * tex.bytes_per_pixel() as usize;
    if data_len != expected {
        bail!("CUDA: texture region data length {data_len} != expected {expected}");
    }
    Ok(())
}

/// `CUDA_MEMCPY2D` cannot be `mem::zeroed` — `CUmemorytype` has no 0 variant.
fn empty_memcpy2d() -> sys::CUDA_MEMCPY2D {
    sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
        srcHost: std::ptr::null(),
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: 0,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
        dstHost: std::ptr::null_mut(),
        dstDevice: 0,
        dstArray: std::ptr::null_mut(),
        dstPitch: 0,
        WidthInBytes: 0,
        Height: 0,
    }
}

/// Host → CUDA array upload (tight or pitched source).
pub(super) fn memcpy_htod_array(
    stream: &CudaStream,
    tex: &CudaTextureResource,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    data: &[u8],
    src_row_pitch: u32,
) -> Result<()> {
    let bpp = tex.bytes_per_pixel();
    let tight_pitch = width.saturating_mul(bpp);
    let pitch = if src_row_pitch == 0 { tight_pitch } else { src_row_pitch };
    if pitch < tight_pitch {
        bail!("CUDA: src_row_pitch {pitch} < tight row bytes {tight_pitch}");
    }
    let min_len = pitch as usize * (height.saturating_sub(1) as usize) + tight_pitch as usize;
    if data.len() < min_len {
        bail!(
            "CUDA: host texture upload buffer too small ({} < {min_len})",
            data.len()
        );
    }
    validate_region(tex, x, y, width, height, tight_pitch as usize * height as usize)?;

    stream
        .context()
        .bind_to_thread()
        .context("CUDA: bind context for texture HtoD")?;
    let mut copy = empty_memcpy2d();
    copy.srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_HOST;
    copy.srcHost = data.as_ptr() as *const _;
    copy.srcPitch = pitch as usize;
    copy.dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_ARRAY;
    copy.dstArray = tex.array();
    copy.dstXInBytes = (x * bpp) as usize;
    copy.dstY = y as usize;
    copy.WidthInBytes = tight_pitch as usize;
    copy.Height = height as usize;
    check_cu(
        unsafe { sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()) },
        "cuMemcpy2DAsync (HtoD array)",
    )
}

/// Device buffer → CUDA array copy.
pub(super) fn memcpy_dtod_array(
    stream: &CudaStream,
    src_device_ptr: u64,
    src_row_pitch: u32,
    tex: &CudaTextureResource,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<()> {
    let bpp = tex.bytes_per_pixel();
    let tight_pitch = width.saturating_mul(bpp);
    let pitch = if src_row_pitch == 0 { tight_pitch } else { src_row_pitch };
    if pitch < tight_pitch {
        bail!("CUDA: src_row_pitch {pitch} < tight row bytes {tight_pitch}");
    }
    if x.checked_add(width).map(|end| end > tex.width).unwrap_or(true)
        || y.checked_add(height).map(|end| end > tex.height).unwrap_or(true)
    {
        bail!(
            "CUDA: buffer→texture region ({x},{y},{width}x{height}) exceeds texture {}x{}",
            tex.width,
            tex.height
        );
    }
    stream
        .context()
        .bind_to_thread()
        .context("CUDA: bind context for buffer→texture")?;
    let mut copy = empty_memcpy2d();
    copy.srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = src_device_ptr;
    copy.srcPitch = pitch as usize;
    copy.dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_ARRAY;
    copy.dstArray = tex.array();
    copy.dstXInBytes = (x * bpp) as usize;
    copy.dstY = y as usize;
    copy.WidthInBytes = tight_pitch as usize;
    copy.Height = height as usize;
    check_cu(
        unsafe { sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()) },
        "cuMemcpy2DAsync (DtoD array)",
    )
}

/// CUDA array → host download (tight destination).
#[allow(dead_code)]
pub(super) fn memcpy_dtoh_array(
    stream: &CudaStream,
    tex: &CudaTextureResource,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    output: &mut [u8],
) -> Result<()> {
    let bpp = tex.bytes_per_pixel();
    let tight_pitch = width.saturating_mul(bpp);
    let expected = tight_pitch as usize * height as usize;
    if output.len() != expected {
        bail!(
            "CUDA: texture download buffer length {} != expected {expected}",
            output.len()
        );
    }
    if x.checked_add(width).map(|end| end > tex.width).unwrap_or(true)
        || y.checked_add(height).map(|end| end > tex.height).unwrap_or(true)
    {
        bail!(
            "CUDA: texture→host region ({x},{y},{width}x{height}) exceeds texture {}x{}",
            tex.width,
            tex.height
        );
    }
    stream
        .context()
        .bind_to_thread()
        .context("CUDA: bind context for texture DtoH")?;
    let mut copy = empty_memcpy2d();
    copy.srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_ARRAY;
    copy.srcArray = tex.array();
    copy.srcXInBytes = (x * bpp) as usize;
    copy.srcY = y as usize;
    copy.dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_HOST;
    copy.dstHost = output.as_mut_ptr() as *mut _;
    copy.dstPitch = tight_pitch as usize;
    copy.WidthInBytes = tight_pitch as usize;
    copy.Height = height as usize;
    check_cu(
        unsafe { sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()) },
        "cuMemcpy2DAsync (DtoH array)",
    )
}

/// CUDA array → device buffer (tight or pitched destination).
pub(super) fn memcpy_array_to_device(
    stream: &CudaStream,
    tex: &CudaTextureResource,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    dst_device_ptr: u64,
    dst_row_pitch: u32,
) -> Result<()> {
    let bpp = tex.bytes_per_pixel();
    let tight_pitch = width.saturating_mul(bpp);
    let pitch = if dst_row_pitch == 0 { tight_pitch } else { dst_row_pitch };
    if pitch < tight_pitch {
        bail!("CUDA: dst_row_pitch {pitch} < tight row bytes {tight_pitch}");
    }
    if x.checked_add(width).map(|end| end > tex.width).unwrap_or(true)
        || y.checked_add(height).map(|end| end > tex.height).unwrap_or(true)
    {
        bail!(
            "CUDA: texture→buffer region ({x},{y},{width}x{height}) exceeds texture {}x{}",
            tex.width,
            tex.height
        );
    }
    stream
        .context()
        .bind_to_thread()
        .context("CUDA: bind context for texture→buffer")?;
    let mut copy = empty_memcpy2d();
    copy.srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_ARRAY;
    copy.srcArray = tex.array();
    copy.srcXInBytes = (x * bpp) as usize;
    copy.srcY = y as usize;
    copy.dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = dst_device_ptr;
    copy.dstPitch = pitch as usize;
    copy.WidthInBytes = tight_pitch as usize;
    copy.Height = height as usize;
    check_cu(
        unsafe { sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()) },
        "cuMemcpy2DAsync (array→device)",
    )
}

/// CUDA array → CUDA array full-image copy (same dimensions/format).
pub(super) fn memcpy_array_to_array(
    stream: &CudaStream,
    src: &CudaTextureResource,
    dst: &CudaTextureResource,
) -> Result<()> {
    if src.width != dst.width || src.height != dst.height {
        bail!(
            "CUDA: texture copy size mismatch ({}x{} → {}x{})",
            src.width,
            src.height,
            dst.width,
            dst.height
        );
    }
    if src.format != dst.format {
        bail!(
            "CUDA: texture copy format mismatch ({:?} → {:?})",
            src.format,
            dst.format
        );
    }
    let bpp = src.bytes_per_pixel();
    let width_bytes = src.width as usize * bpp as usize;
    stream
        .context()
        .bind_to_thread()
        .context("CUDA: bind context for texture→texture")?;
    let mut copy = empty_memcpy2d();
    copy.srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_ARRAY;
    copy.srcArray = src.array();
    copy.dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_ARRAY;
    copy.dstArray = dst.array();
    copy.WidthInBytes = width_bytes;
    copy.Height = src.height as usize;
    check_cu(
        unsafe { sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()) },
        "cuMemcpy2DAsync (array→array)",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bgra_formats() {
        assert!(format_info(TextureFormat::Bgra8Unorm).is_err());
        assert!(format_info(TextureFormat::Bgra8UnormSrgb).is_err());
    }

    #[test]
    fn accepts_core_formats() {
        for format in [
            TextureFormat::R8Unorm,
            TextureFormat::Rg8Unorm,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba16Float,
            TextureFormat::Rgba32Float,
        ] {
            format_info(format).unwrap();
        }
    }

    #[test]
    fn storage_compat_size_matched_pairs() {
        assert!(storage_shader_compatible("float4", TextureFormat::Rgba32Float));
        assert!(storage_shader_compatible("half4", TextureFormat::Rgba16Float));
        assert!(storage_shader_compatible("uint8_t4", TextureFormat::Rgba8Unorm));
        assert!(storage_shader_compatible("vector<uint8_t, 4>", TextureFormat::Rgba8Unorm));
        // Mismatched sizes / types must stay rejected (no typed UAV conversion).
        assert!(!storage_shader_compatible("float4", TextureFormat::Rgba8Unorm));
        assert!(!storage_shader_compatible("float4", TextureFormat::Rgba16Float));
        assert!(!storage_shader_compatible("half4", TextureFormat::Rgba32Float));
        assert!(!storage_shader_compatible("uint8_t4", TextureFormat::Rgba32Float));
        assert!(!storage_shader_compatible("uchar4", TextureFormat::Rgba8Unorm));
        assert!(!storage_shader_compatible("float", TextureFormat::Rgba32Float));
    }

    #[test]
    fn sampler_rejects_compare_and_split_filters() {
        let mut desc = SamplerDesc::default();
        desc.compare = Some(crate::types::CompareFunction::Less);
        assert!(CudaSamplerKey::from_desc(&desc).is_err());

        desc = SamplerDesc::default();
        desc.mag_filter = FilterMode::Linear;
        desc.min_filter = FilterMode::Nearest;
        assert!(CudaSamplerKey::from_desc(&desc).is_err());
    }

    #[test]
    fn sampler_maps_linear_repeat() {
        let key = CudaSamplerKey::from_desc(&SamplerDesc {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(key.filter, sys::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR);
        assert_eq!(key.address_u, sys::CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP);
    }
}
