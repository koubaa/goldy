//! WebGPU backend prototype (via wgpu).
//!
//! This backend deliberately does not emulate Goldy's native bindless heap. The
//! raw indices in [`GpuCommand::BindResourcesRaw`] are interpreted as backend
//! registry keys and packed into one fixed bind group in shader-parameter order.
//!
//! Submit is non-blocking: the timeline advances from `Queue::on_submitted_work_done`
//! (pumped by `Device::poll`). Host waits use a stored [`wgpu::SubmissionIndex`].
//!
//! Surfaces (`graphics` feature): `begin_frame` acquires the wgpu drawable.
//! Presentation picks the cheapest path that actually works:
//! 1. **Copy** — compute writes a same-format storage scratch, then `copy_texture_to_texture`.
//! 2. **Blit** — compute writes `Rgba8Unorm` scratch, then a fullscreen pass to the swapchain.
//!
//! **Direct** (compute writes the swapchain image) is implemented but not auto-selected:
//! wgpu 28 hardcodes swapchain `format_features` to `RENDER_ATTACHMENT` only, so storage
//! bind groups fail even when `SurfaceCapabilities` advertise `STORAGE_BINDING`.
//! Override with `GOLDY_WEBGPU_PRESENT=copy|blit` (`direct` is rejected until wgpu fixes this).
//!
//! Raster: offscreen color (optional depth) targets, graphics PSOs from Slang WGSL, and
//! `CopyRenderTarget` into a texture. Vertex/fragment resources use the same packed bind
//! group as compute.

use super::shared::{PushLayout, DISPATCH_BATCH_STRIDE, MAX_USER_SLOTS, TOTAL_PUSH_BYTES};
use super::*;
use crate::frame_table::dispatch_table_base_word_index;
use crate::slang::virtual_main::{CudaStorageTextureSpec, WgpuComputeLayout, WgpuComputeResourceKind};
use crate::slang::OwnedLayoutCheck;
use crate::types::{
    AddressMode, BufferKind, BufferResizeCost, CompareFunction, DepthFormat, DepthStencilState, DeviceType, FilterMode,
    IndexFormat, PrimitiveTopology, ResourceCategory, VertexBufferLayout, VertexFormat,
};
use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const RAW_WGSL_MARKER: &str = "// @goldy-wgsl";
const USER_UNIFORM_BYTES: u64 = (MAX_USER_SLOTS * 4) as u64;
const TIMELINE_WAIT_TIMEOUT: Duration = Duration::from_secs(60);
const WEBGPU_REGISTRY_CAP: u32 = 4096;
const INDIRECT_DISPATCH_BYTES: u64 = 12;
#[cfg(feature = "graphics")]
const DEFAULT_SURFACE_WIDTH: u32 = 800;
#[cfg(feature = "graphics")]
const DEFAULT_SURFACE_HEIGHT: u32 = 600;

#[cfg(feature = "graphics")]
const PRESENT_BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VsOut;
    out.pos = vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(i32(in.pos.x), i32(in.pos.y));
    return textureLoad(src, xy, 0);
}
"#;

/// How compute pixels reach the swapchain.
#[cfg(feature = "graphics")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WebGpuPresentPath {
    Direct,
    Copy,
    Blit,
}

#[cfg(feature = "graphics")]
fn parse_present_override() -> Result<Option<WebGpuPresentPath>> {
    let Ok(raw) = std::env::var("GOLDY_WEBGPU_PRESENT") else {
        return Ok(None);
    };
    match raw.trim() {
        "" => Ok(None),
        "direct" => Ok(Some(WebGpuPresentPath::Direct)),
        "copy" => Ok(Some(WebGpuPresentPath::Copy)),
        "blit" => Ok(Some(WebGpuPresentPath::Blit)),
        other => anyhow::bail!("GOLDY_WEBGPU_PRESENT={other:?} is invalid (expected direct, copy, or blit)"),
    }
}

#[cfg(feature = "graphics")]
fn storage_capable_scratch(format: wgpu::TextureFormat, features: wgpu::Features) -> bool {
    match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float => true,
        wgpu::TextureFormat::Bgra8Unorm => features.contains(wgpu::Features::BGRA8UNORM_STORAGE),
        _ => false,
    }
}

/// wgpu 28 `present.rs` stamps swapchain textures with `allowed_usages: RENDER_ATTACHMENT`
/// and no `STORAGE_WRITE_ONLY`, so a storage bind group always fails validation.
#[cfg(feature = "graphics")]
fn wgpu_swapchain_storage_write_supported() -> bool {
    false
}

#[cfg(feature = "graphics")]
fn choose_present_path(
    usages: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
    features: wgpu::Features,
    override_path: Option<WebGpuPresentPath>,
) -> Result<WebGpuPresentPath> {
    let can_direct = wgpu_swapchain_storage_write_supported()
        && usages.contains(wgpu::TextureUsages::STORAGE_BINDING)
        && storage_capable_scratch(format, features);
    let can_copy = usages.contains(wgpu::TextureUsages::COPY_DST) && storage_capable_scratch(format, features);
    let can_blit = usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT);
    let pick = |path: WebGpuPresentPath| -> Result<WebGpuPresentPath> {
        let ok = match path {
            WebGpuPresentPath::Direct => can_direct,
            WebGpuPresentPath::Copy => can_copy,
            WebGpuPresentPath::Blit => can_blit,
        };
        let extra = if path == WebGpuPresentPath::Direct && !can_direct {
            " (wgpu swapchain images do not expose STORAGE_WRITE_ONLY; use copy or blit)"
        } else {
            ""
        };
        anyhow::ensure!(
            ok,
            "WebGPU present path {path:?} is unavailable for swapchain {format:?} (usages {usages:?}){extra}"
        );
        Ok(path)
    };
    if let Some(forced) = override_path {
        return pick(forced);
    }
    if can_copy {
        Ok(WebGpuPresentPath::Copy)
    } else {
        pick(WebGpuPresentPath::Blit)
    }
}

#[cfg(feature = "graphics")]
fn map_goldy_texture_format(format: wgpu::TextureFormat) -> Option<TextureFormat> {
    match format {
        wgpu::TextureFormat::R8Unorm => Some(TextureFormat::R8Unorm),
        wgpu::TextureFormat::Rg8Unorm => Some(TextureFormat::Rg8Unorm),
        wgpu::TextureFormat::Rgba8Unorm => Some(TextureFormat::Rgba8Unorm),
        wgpu::TextureFormat::Rgba8UnormSrgb => Some(TextureFormat::Rgba8UnormSrgb),
        wgpu::TextureFormat::Bgra8Unorm => Some(TextureFormat::Bgra8Unorm),
        wgpu::TextureFormat::Bgra8UnormSrgb => Some(TextureFormat::Bgra8UnormSrgb),
        wgpu::TextureFormat::Rgba16Float => Some(TextureFormat::Rgba16Float),
        wgpu::TextureFormat::Rgba32Float => Some(TextureFormat::Rgba32Float),
        _ => None,
    }
}

#[cfg(feature = "graphics")]
fn compute_format_for_path(path: WebGpuPresentPath, swapchain: wgpu::TextureFormat) -> Result<TextureFormat> {
    match path {
        WebGpuPresentPath::Direct | WebGpuPresentPath::Copy => map_goldy_texture_format(swapchain)
            .with_context(|| format!("WebGPU: unsupported swapchain format {swapchain:?} for {path:?} present")),
        WebGpuPresentPath::Blit => Ok(TextureFormat::Rgba8Unorm),
    }
}

fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        return value;
    }
    value.div_ceil(align) * align
}

#[cfg(feature = "graphics")]
fn pick_swapchain_format(formats: &[wgpu::TextureFormat]) -> Result<wgpu::TextureFormat> {
    // Prefer formats that can be storage UAVs so Direct/Copy stay available.
    const PREFERRED: &[wgpu::TextureFormat] = &[
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8UnormSrgb,
    ];
    for preferred in PREFERRED {
        if formats.contains(preferred) {
            return Ok(*preferred);
        }
    }
    formats
        .first()
        .copied()
        .context("WebGPU: surface has no presentable formats")
}

#[cfg(feature = "graphics")]
fn pick_present_mode(modes: &[wgpu::PresentMode]) -> Result<wgpu::PresentMode> {
    if modes.contains(&wgpu::PresentMode::Fifo) {
        return Ok(wgpu::PresentMode::Fifo);
    }
    modes.first().copied().context("WebGPU: surface has no present modes")
}

#[cfg(feature = "graphics")]
fn pick_alpha_mode(modes: &[wgpu::CompositeAlphaMode]) -> wgpu::CompositeAlphaMode {
    if modes.contains(&wgpu::CompositeAlphaMode::Auto) {
        wgpu::CompositeAlphaMode::Auto
    } else {
        modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Opaque)
    }
}

#[cfg(feature = "graphics")]
fn surface_usage(caps: &wgpu::SurfaceCapabilities, path: WebGpuPresentPath) -> Result<wgpu::TextureUsages> {
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    match path {
        WebGpuPresentPath::Direct => {
            anyhow::ensure!(
                caps.usages.contains(wgpu::TextureUsages::STORAGE_BINDING),
                "WebGPU: direct present requires STORAGE_BINDING on the surface"
            );
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        WebGpuPresentPath::Copy => {
            anyhow::ensure!(
                caps.usages.contains(wgpu::TextureUsages::COPY_DST),
                "WebGPU: copy present requires COPY_DST on the surface"
            );
            usage |= wgpu::TextureUsages::COPY_DST;
        }
        WebGpuPresentPath::Blit => {
            anyhow::ensure!(
                caps.usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT),
                "WebGPU: blit present requires RENDER_ATTACHMENT on the surface"
            );
        }
    }
    Ok(usage)
}

fn map_texture_format(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
    }
}

fn map_address_mode(mode: AddressMode) -> wgpu::AddressMode {
    match mode {
        AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        AddressMode::Repeat => wgpu::AddressMode::Repeat,
        AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
    }
}

fn map_filter_mode(mode: FilterMode) -> wgpu::FilterMode {
    match mode {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

fn map_mipmap_filter_mode(mode: FilterMode) -> wgpu::MipmapFilterMode {
    match mode {
        FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
        FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

fn map_compare(compare: CompareFunction) -> wgpu::CompareFunction {
    match compare {
        CompareFunction::Never => wgpu::CompareFunction::Never,
        CompareFunction::Less => wgpu::CompareFunction::Less,
        CompareFunction::Equal => wgpu::CompareFunction::Equal,
        CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        CompareFunction::Greater => wgpu::CompareFunction::Greater,
        CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        CompareFunction::Always => wgpu::CompareFunction::Always,
    }
}

#[cfg(feature = "graphics")]
fn map_vertex_format(format: VertexFormat) -> wgpu::VertexFormat {
    match format {
        VertexFormat::Float32 => wgpu::VertexFormat::Float32,
        VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
        VertexFormat::Sint32 => wgpu::VertexFormat::Sint32,
        VertexFormat::Uint8x4 => wgpu::VertexFormat::Uint8x4,
        VertexFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
    }
}

#[cfg(feature = "graphics")]
fn map_topology(topology: PrimitiveTopology) -> wgpu::PrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
        PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}

#[cfg(feature = "graphics")]
fn map_index_format(format: IndexFormat) -> wgpu::IndexFormat {
    match format {
        IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
        IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
    }
}

#[cfg(feature = "graphics")]
fn map_depth_format(format: DepthFormat) -> wgpu::TextureFormat {
    match format {
        DepthFormat::Depth16Unorm => wgpu::TextureFormat::Depth16Unorm,
        DepthFormat::Depth24Plus => wgpu::TextureFormat::Depth24Plus,
        DepthFormat::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
        DepthFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        DepthFormat::Depth32FloatStencil8 => wgpu::TextureFormat::Depth32FloatStencil8,
    }
}

#[cfg(feature = "graphics")]
fn map_color_load(load: crate::types::TargetLoad) -> wgpu::LoadOp<wgpu::Color> {
    match load {
        crate::types::TargetLoad::Clear(color) => wgpu::LoadOp::Clear(wgpu::Color {
            r: color.r as f64,
            g: color.g as f64,
            b: color.b as f64,
            a: color.a as f64,
        }),
        crate::types::TargetLoad::Load => wgpu::LoadOp::Load,
        crate::types::TargetLoad::Discard => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
    }
}

fn copy_row_pitch(width: u32, format: TextureFormat) -> u32 {
    let tight = width.saturating_mul(format.bytes_per_pixel()).max(1);
    tight.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

fn texture_copy_footprint(width: u32, height: u32, format: TextureFormat) -> TextureCopyFootprint {
    let row_pitch = copy_row_pitch(width, format);
    let tight = width.saturating_mul(format.bytes_per_pixel());
    let logical_bytes = tight as u64 * height as u64;
    let staging_bytes = row_pitch as u64 * height as u64;
    TextureCopyFootprint {
        width,
        height,
        format,
        logical_bytes,
        staging_bytes,
        row_pitch,
        footprint_offset: 0,
    }
}

fn texel_copy<'a>(texture: &'a wgpu::Texture, x: u32, y: u32) -> wgpu::TexelCopyTextureInfo<'a> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d { x, y, z: 0 },
        aspect: wgpu::TextureAspect::All,
    }
}

fn unpack_texture_rows(layout: TextureCopyFootprint, staging: &[u8], output: &mut [u8]) -> Result<()> {
    if output.len() as u64 != layout.logical_bytes {
        anyhow::bail!(
            "WebGPU: texture readback size mismatch: expected {}, got {}",
            layout.logical_bytes,
            output.len()
        );
    }
    let tight = layout.tight_row_bytes() as usize;
    let pitch = layout.row_pitch as usize;
    let base = layout.footprint_offset as usize;
    if pitch == tight && base == 0 {
        let n = layout.logical_bytes as usize;
        output.copy_from_slice(&staging[..n]);
        return Ok(());
    }
    for row in 0..layout.height as usize {
        let src = base + row * pitch;
        let dst = row * tight;
        output[dst..dst + tight].copy_from_slice(&staging[src..src + tight]);
    }
    Ok(())
}

fn pack_user_uniform(user: &[u32]) -> [u8; USER_UNIFORM_BYTES as usize] {
    let mut bytes = [0u8; USER_UNIFORM_BYTES as usize];
    for (i, word) in user.iter().copied().take(MAX_USER_SLOTS).enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn write_queue_buffer(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    if data.is_empty() {
        return;
    }
    let aligned = align_up(data.len() as u64, wgpu::COPY_BUFFER_ALIGNMENT);
    if aligned as usize == data.len() {
        queue.write_buffer(buffer, offset, data);
        return;
    }
    let mut padded = vec![0u8; aligned as usize];
    padded[..data.len()].copy_from_slice(data);
    queue.write_buffer(buffer, offset, &padded);
}

fn require_user_scalars(user: &[u32], expected: u32) -> Result<()> {
    anyhow::ensure!(
        user.len() >= expected as usize,
        "WebGPU: shader expects {expected} scalar words, BindResourcesRaw.user has {}",
        user.len()
    );
    Ok(())
}

fn ensure_buffer_range(buffer: &WebGpuBuffer, offset: u64, size: u64, what: &str) -> Result<()> {
    let end = offset.checked_add(size).context("WebGPU: buffer range overflow")?;
    anyhow::ensure!(
        end <= buffer.size,
        "WebGPU: {what} range {offset}+{size} exceeds logical size {}",
        buffer.size
    );
    Ok(())
}

pub(crate) struct WebGpuBackend {
    /// Retained for `create_surface` (`graphics`). Unused in compute-only builds.
    #[cfg_attr(not(feature = "graphics"), allow(dead_code))]
    instance: wgpu::Instance,
    adapters: Vec<wgpu::Adapter>,
    adapter_info: Vec<AdapterInfo>,
    devices: HashMap<DeviceHandle, WebGpuDevice>,
    contexts: HashMap<ContextHandle, Arc<WebGpuContext>>,
    buffers: HashMap<BufferHandle, WebGpuBuffer>,
    buffer_slots: HashMap<u32, BufferHandle>,
    textures: HashMap<TextureHandle, WebGpuTexture>,
    texture_slots: HashMap<u32, TextureHandle>,
    samplers: HashMap<SamplerHandle, WebGpuSampler>,
    sampler_slots: HashMap<u32, SamplerHandle>,
    shaders: HashMap<ShaderHandle, WebGpuShader>,
    compute_pipelines: HashMap<ComputePipelineHandle, WebGpuComputePipeline>,
    #[cfg(feature = "graphics")]
    surfaces: HashMap<SurfaceHandle, WebGpuSurface>,
    next_device: DeviceHandle,
    next_context: ContextHandle,
    next_buffer: BufferHandle,
    next_texture: TextureHandle,
    next_sampler: SamplerHandle,
    #[cfg(feature = "graphics")]
    next_surface: SurfaceHandle,
    next_slot: u32,
    free_slots: Vec<u32>,
    next_shader: ShaderHandle,
    next_compute_pipeline: ComputePipelineHandle,
    #[cfg(feature = "graphics")]
    graphics_pipelines: HashMap<PipelineHandle, WebGpuGraphicsPipeline>,
    #[cfg(feature = "graphics")]
    render_targets: HashMap<RenderTargetHandle, WebGpuRenderTarget>,
    #[cfg(feature = "graphics")]
    next_graphics_pipeline: PipelineHandle,
    #[cfg(feature = "graphics")]
    next_render_target: RenderTargetHandle,
    last_frame_table: Option<Arc<[u32]>>,
}

#[cfg(feature = "graphics")]
struct WebGpuSurface {
    device: DeviceHandle,
    surface: wgpu::Surface<'static>,
    width: u32,
    height: u32,
    swapchain_format: wgpu::TextureFormat,
    compute_format: TextureFormat,
    present_path: WebGpuPresentPath,
    present_mode: wgpu::PresentMode,
    alpha_mode: wgpu::CompositeAlphaMode,
    usage: wgpu::TextureUsages,
    scratch: Option<TextureHandle>,
    /// Direct path: Goldy handle wrapping the acquired swapchain image for this frame.
    lease: Option<TextureHandle>,
    acquired: Option<wgpu::SurfaceTexture>,
    current_texture_handle: Option<TextureHandle>,
}

#[cfg(feature = "graphics")]
#[derive(Clone)]
struct WebGpuBlitPipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

struct WebGpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    features: wgpu::Features,
    next_timeline: Arc<AtomicU64>,
    retired: Arc<AtomicU64>,
    last_submission: Arc<Mutex<Option<(crate::timeline::TimelineValue, wgpu::SubmissionIndex)>>>,
    user_uniform: Option<wgpu::Buffer>,
    user_uniform_capacity: u64,
    uniform_offset_align: u64,
    storage_offset_align: u64,
    adapter_id: u32,
    #[cfg(feature = "graphics")]
    blit: HashMap<wgpu::TextureFormat, WebGpuBlitPipeline>,
}

struct WebGpuContext {
    device: DeviceHandle,
    wgpu_device: wgpu::Device,
    completed: AtomicU64,
    submitted_max: AtomicU64,
    last_submission: Mutex<Option<(crate::timeline::TimelineValue, wgpu::SubmissionIndex)>>,
    signal_queue: crate::signal::SignalQueue,
}

struct WebGpuProgress {
    context: Arc<WebGpuContext>,
}

impl ContextGpuProgress for WebGpuProgress {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        pump_device(&self.context.wgpu_device);
        self.context.completed.load(Ordering::Acquire)
    }
}

struct WebGpuDestroyContext {
    device: wgpu::Device,
}

impl ContextDestroyHandle for WebGpuDestroyContext {
    fn wait(&self) -> Result<()> {
        poll_device(&self.device, wgpu::PollType::wait_indefinitely()).map(|_| ())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

struct WebGpuTimelineWait {
    device: wgpu::Device,
    index: wgpu::SubmissionIndex,
    context: Arc<WebGpuContext>,
    retired: Arc<AtomicU64>,
    value: crate::timeline::TimelineValue,
}

impl TimelineBlockingWait for WebGpuTimelineWait {
    fn block(self: Box<Self>) -> Result<()> {
        let value = self.value;
        if !self.block_timeout(u32::try_from(TIMELINE_WAIT_TIMEOUT.as_millis()).unwrap_or(u32::MAX))? {
            anyhow::bail!("WebGPU: timed out after 60 s waiting for timeline value {value}");
        }
        Ok(())
    }

    fn block_timeout(self: Box<Self>, timeout_ms: u32) -> Result<bool> {
        if self.context.completed.load(Ordering::Acquire) >= self.value {
            return Ok(true);
        }
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(self.index),
            timeout: Some(Duration::from_millis(u64::from(timeout_ms))),
        }) {
            Ok(_) => {
                self.context.completed.fetch_max(self.value, Ordering::Release);
                self.retired.fetch_max(self.value, Ordering::AcqRel);
                Ok(true)
            }
            Err(wgpu::PollError::Timeout) => Ok(false),
            Err(error) => Err(anyhow::anyhow!("WebGPU device poll failed: {error}")),
        }
    }
}

fn pump_device(device: &wgpu::Device) {
    let _ = device.poll(wgpu::PollType::Poll);
}

fn poll_device(device: &wgpu::Device, poll_type: wgpu::PollType) -> Result<wgpu::PollStatus> {
    device
        .poll(poll_type)
        .map_err(|error| anyhow::anyhow!("WebGPU device poll failed: {error}"))
}

/// Rewrite identity Slang WGSL (`rgba32float` storage) to packed 8-bit storage formats.
fn patch_wgsl_storage_formats(wgsl: &str, replacements: &[(u32, &str)]) -> Result<String> {
    let mut out = wgsl.to_string();
    let mut bindings: Vec<(u32, &str)> = replacements.to_vec();
    bindings.sort_by_key(|(binding, _)| *binding);
    bindings.dedup_by_key(|(binding, _)| *binding);
    bindings.reverse();
    for (binding, to_format) in bindings {
        let needle = format!("@binding({binding})");
        let Some(decl_start) = out.find(&needle) else {
            anyhow::bail!("WebGPU: WGSL missing {needle} for {to_format} specialization");
        };
        let after = &out[decl_start..];
        let next_binding = after[needle.len()..]
            .find("@binding(")
            .map(|i| needle.len() + i)
            .unwrap_or(after.len());
        let decl = &after[..next_binding];
        let Some(rel) = decl.find("texture_storage_2d<rgba32float") else {
            anyhow::bail!("WebGPU: expected texture_storage_2d<rgba32float at {needle} for float4→{to_format}");
        };
        let abs = decl_start + rel + "texture_storage_2d<".len();
        const FROM: &str = "rgba32float";
        anyhow::ensure!(
            out.get(abs..abs + FROM.len()) == Some(FROM),
            "WebGPU: storage format rewrite at {needle} missed rgba32float"
        );
        out.replace_range(abs..abs + FROM.len(), to_format);
        let access_at = abs + to_format.len();
        if out
            .get(access_at..)
            .is_some_and(|rest| rest.starts_with(", read_write"))
        {
            out.replace_range(access_at..access_at + ", read_write".len(), ", write");
        }
    }
    Ok(out)
}

#[derive(Clone)]
struct WebGpuBuffer {
    device: DeviceHandle,
    buffer: wgpu::Buffer,
    offset: u64,
    size: u64,
    capacity: u64,
    slot: Option<u32>,
    readback: bool,
    uniform: bool,
}

struct WebGpuTexture {
    device: DeviceHandle,
    texture: wgpu::Texture,
    sampled_view: Option<wgpu::TextureView>,
    storage_view: Option<wgpu::TextureView>,
    width: u32,
    height: u32,
    format: TextureFormat,
    storage_slot: Option<u32>,
    sampled_slot: Option<u32>,
}

struct WebGpuSampler {
    device: DeviceHandle,
    sampler: wgpu::Sampler,
    slot: u32,
}

#[derive(Clone)]
struct WebGpuShader {
    device: DeviceHandle,
    source: String,
    search_paths: Vec<String>,
    defines: Vec<(String, String)>,
    optimization_level: crate::types::OptimizationLevel,
    layout_checks: Vec<OwnedLayoutCheck>,
}

#[cfg(feature = "graphics")]
struct WebGpuGraphicsPipeline {
    device: DeviceHandle,
    pipeline: wgpu::RenderPipeline,
    layout: WgpuComputeLayout,
    #[allow(dead_code)]
    vertex_stride: u32,
    #[allow(dead_code)]
    topology: PrimitiveTopology,
}

#[cfg(feature = "graphics")]
struct WebGpuRenderTarget {
    device: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth: Option<(wgpu::Texture, wgpu::TextureView)>,
}

struct WebGpuComputePipeline {
    device: DeviceHandle,
    /// Cloned Slang source so float4→rgba8unorm PSO variants survive `destroy_shader`.
    shader: WebGpuShader,
    pipeline: wgpu::ComputePipeline,
    /// Specialized PSOs: bit `i` set when DirectSpatial slot `i` is float4→rgba8unorm.
    variants: HashMap<u64, wgpu::ComputePipeline>,
    slot_access: Vec<Option<ResourceAccess>>,
    layout: WgpuComputeLayout,
}

impl WebGpuBackend {
    pub(crate) fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        if adapters.is_empty() {
            anyhow::bail!("WebGPU: no compatible adapters found");
        }
        let adapter_info = adapters
            .iter()
            .enumerate()
            .map(|(id, adapter)| {
                let info = adapter.get_info();
                AdapterInfo {
                    id: id as u32,
                    name: info.name,
                    vendor: format!("0x{:04x}", info.vendor),
                    backend: BackendType::WebGpu,
                    device_type: map_device_type(info.device_type),
                }
            })
            .collect();
        Ok(Self {
            instance,
            adapters,
            adapter_info,
            devices: HashMap::new(),
            contexts: HashMap::new(),
            buffers: HashMap::new(),
            buffer_slots: HashMap::new(),
            textures: HashMap::new(),
            texture_slots: HashMap::new(),
            samplers: HashMap::new(),
            sampler_slots: HashMap::new(),
            shaders: HashMap::new(),
            compute_pipelines: HashMap::new(),
            #[cfg(feature = "graphics")]
            surfaces: HashMap::new(),
            next_device: 1,
            next_context: 1,
            next_buffer: 1,
            next_texture: 1,
            next_sampler: 1,
            #[cfg(feature = "graphics")]
            next_surface: 1,
            next_slot: 0,
            free_slots: Vec::new(),
            next_shader: 1,
            next_compute_pipeline: 1,
            #[cfg(feature = "graphics")]
            graphics_pipelines: HashMap::new(),
            #[cfg(feature = "graphics")]
            render_targets: HashMap::new(),
            #[cfg(feature = "graphics")]
            next_graphics_pipeline: 1,
            #[cfg(feature = "graphics")]
            next_render_target: 1,
            last_frame_table: None,
        })
    }

    fn device(&self, handle: DeviceHandle) -> Result<&WebGpuDevice> {
        self.devices.get(&handle).context("WebGPU: invalid device handle")
    }

    fn context(&self, handle: ContextHandle) -> Result<&Arc<WebGpuContext>> {
        self.contexts.get(&handle).context("WebGPU: invalid context handle")
    }

    #[allow(dead_code)]
    fn unsupported<T>(operation: &str) -> Result<T> {
        anyhow::bail!("WebGPU compute-only backend does not support {operation}")
    }

    #[cfg(feature = "graphics")]
    fn surface_config(surface: &WebGpuSurface) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: surface.usage,
            format: surface.swapchain_format,
            width: surface.width.max(1),
            height: surface.height.max(1),
            present_mode: surface.present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface.alpha_mode,
            view_formats: Vec::new(),
        }
    }

    #[cfg(feature = "graphics")]
    fn configure_surface(&mut self, surface: SurfaceHandle) -> Result<()> {
        let device = self
            .surfaces
            .get(&surface)
            .context("WebGPU: invalid surface handle")?
            .device;
        let wgpu_device = self.device(device)?.device.clone();
        let surface_state = self
            .surfaces
            .get_mut(&surface)
            .context("WebGPU: invalid surface handle")?;
        surface_state.acquired = None;
        let config = Self::surface_config(surface_state);
        surface_state.surface.configure(&wgpu_device, &config);
        Ok(())
    }

    #[cfg(feature = "graphics")]
    fn drop_surface_lease(&mut self, surface: SurfaceHandle) {
        let Some(handle) = self.surfaces.get_mut(&surface).and_then(|s| s.lease.take()) else {
            return;
        };
        if let Some(state) = self.surfaces.get_mut(&surface) {
            if state.current_texture_handle == Some(handle) {
                state.current_texture_handle = state.scratch;
            }
        }
        self.destroy_texture(handle);
    }

    #[cfg(feature = "graphics")]
    fn drop_surface_scratch(&mut self, surface: SurfaceHandle) {
        self.drop_surface_lease(surface);
        let Some(handle) = self.surfaces.get_mut(&surface).and_then(|s| s.scratch.take()) else {
            return;
        };
        if let Some(state) = self.surfaces.get_mut(&surface) {
            state.current_texture_handle = None;
        }
        self.destroy_texture(handle);
    }

    #[cfg(feature = "graphics")]
    fn ensure_surface_scratch(&mut self, surface: SurfaceHandle) -> Result<TextureHandle> {
        let (device, width, height, existing, format, path) = {
            let state = self.surfaces.get(&surface).context("WebGPU: invalid surface handle")?;
            (
                state.device,
                state.width.max(1),
                state.height.max(1),
                state.scratch,
                state.compute_format,
                state.present_path,
            )
        };
        anyhow::ensure!(
            path != WebGpuPresentPath::Direct,
            "WebGPU: direct present does not use a compute scratch"
        );
        if let Some(handle) = existing {
            if let Some(texture) = self.textures.get(&handle) {
                if texture.width == width
                    && texture.height == height
                    && texture.format == format
                    && texture.storage_slot.is_some()
                {
                    return Ok(handle);
                }
            }
        }
        self.drop_surface_scratch(surface);
        let kind = if path == WebGpuPresentPath::Blit {
            TextureKind::DirectInterpolated
        } else {
            TextureKind::Direct
        };
        let handle = self.create_texture(
            device,
            width,
            height,
            format,
            kind,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        )?;
        let state = self
            .surfaces
            .get_mut(&surface)
            .context("WebGPU: invalid surface handle")?;
        state.scratch = Some(handle);
        state.current_texture_handle = Some(handle);
        Ok(handle)
    }

    #[cfg(feature = "graphics")]
    fn register_surface_lease(
        &mut self,
        device: DeviceHandle,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<TextureHandle> {
        self.insert_owned_texture(device, texture, width, height, format, false, true)
    }

    fn insert_owned_texture(
        &mut self,
        device: DeviceHandle,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
        format: TextureFormat,
        needs_sampled: bool,
        needs_storage: bool,
    ) -> Result<TextureHandle> {
        let sampled_view = needs_sampled.then(|| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("goldy-webgpu-sampled"),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                ..Default::default()
            })
        });
        let storage_view = needs_storage.then(|| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("goldy-webgpu-storage"),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                ..Default::default()
            })
        });

        let storage_slot = if needs_storage {
            Some(self.alloc_registry_slot()?)
        } else {
            None
        };
        let sampled_slot = if needs_sampled {
            Some(self.alloc_registry_slot()?)
        } else {
            None
        };
        let handle = self.next_texture;
        self.next_texture += 1;
        if let Some(slot) = storage_slot {
            self.texture_slots.insert(slot, handle);
        }
        if let Some(slot) = sampled_slot {
            self.texture_slots.insert(slot, handle);
        }
        self.textures.insert(
            handle,
            WebGpuTexture {
                device,
                texture,
                sampled_view,
                storage_view,
                width,
                height,
                format,
                storage_slot,
                sampled_slot,
            },
        );
        Ok(handle)
    }

    #[cfg(feature = "graphics")]
    fn blit_pipeline(
        &mut self,
        device: DeviceHandle,
        target_format: wgpu::TextureFormat,
    ) -> Result<WebGpuBlitPipeline> {
        if let Some(cached) = self.device(device)?.blit.get(&target_format) {
            return Ok(cached.clone());
        }
        let gpu = self.device(device)?;
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("goldy-webgpu-present-blit"),
            source: wgpu::ShaderSource::Wgsl(PRESENT_BLIT_WGSL.into()),
        });
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("goldy-webgpu-present-blit-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("goldy-webgpu-present-blit-layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("goldy-webgpu-present-blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let created = WebGpuBlitPipeline { pipeline, layout };
        self.devices
            .get_mut(&device)
            .context("WebGPU: invalid device handle")?
            .blit
            .insert(target_format, created.clone());
        Ok(created)
    }

    #[cfg(all(feature = "graphics", test))]
    fn blit_texture_to_target(&mut self, device: DeviceHandle, src: TextureHandle, dst: TextureHandle) -> Result<()> {
        let src_tex = self.textures.get(&src).context("WebGPU: invalid blit source")?;
        let dst_tex = self.textures.get(&dst).context("WebGPU: invalid blit destination")?;
        anyhow::ensure!(
            src_tex.width == dst_tex.width && src_tex.height == dst_tex.height,
            "WebGPU: blit size mismatch"
        );
        let src_view = src_tex
            .sampled_view
            .clone()
            .context("WebGPU: blit source needs a sampled view")?;
        let dst_texture = dst_tex.texture.clone();
        let width = dst_tex.width;
        let height = dst_tex.height;
        let format = dst_tex.texture.format();
        let blit = self.blit_pipeline(device, format)?;
        let gpu = self.device(device)?;
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("goldy-webgpu-offscreen-blit"),
        });
        encode_present_blit(&mut encoder, &gpu.device, &blit, &src_view, &dst_texture, width, height);
        let index = gpu.queue.submit([encoder.finish()]);
        poll_device(
            &gpu.device,
            wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: Some(TIMELINE_WAIT_TIMEOUT),
            },
        )?;
        Ok(())
    }

    #[cfg(feature = "graphics")]
    fn acquire_surface_texture(&mut self, surface: SurfaceHandle) -> Result<wgpu::SurfaceTexture> {
        {
            let state = self
                .surfaces
                .get_mut(&surface)
                .context("WebGPU: invalid surface handle")?;
            state.acquired = None;
        }
        let first = self
            .surfaces
            .get(&surface)
            .context("WebGPU: invalid surface handle")?
            .surface
            .get_current_texture();
        match first {
            Ok(texture) => Ok(texture),
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost | wgpu::SurfaceError::Timeout) => {
                self.configure_surface(surface)?;
                self.surfaces
                    .get(&surface)
                    .context("WebGPU: invalid surface handle")?
                    .surface
                    .get_current_texture()
                    .context("WebGPU: get_current_texture after reconfigure")
            }
            Err(error) => Err(anyhow::anyhow!("WebGPU: get_current_texture failed: {error}")),
        }
    }

    fn alloc_registry_slot(&mut self) -> Result<u32> {
        if let Some(slot) = self.free_slots.pop() {
            return Ok(slot);
        }
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .context("WebGPU resource registry exhausted")?;
        Ok(slot)
    }

    fn recycle_registry_slot(&mut self, slot: u32) {
        self.free_slots.push(slot);
    }

    fn create_storage_buffer(
        &mut self,
        device: DeviceHandle,
        logical_size: u64,
        capacity: u64,
        uniform: bool,
    ) -> Result<BufferHandle> {
        let min_capacity = if uniform { 16 } else { 4 };
        let capacity = if uniform {
            align_up(capacity.max(logical_size).max(min_capacity), 16)
        } else {
            align_up(
                capacity.max(logical_size).max(min_capacity),
                wgpu::COPY_BUFFER_ALIGNMENT,
            )
        };
        let gpu = self.device(device)?;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("goldy-webgpu-buffer"),
            size: capacity,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });
        let handle = self.next_buffer;
        self.next_buffer += 1;
        let slot = self.alloc_registry_slot()?;
        self.buffer_slots.insert(slot, handle);
        self.buffers.insert(
            handle,
            WebGpuBuffer {
                device,
                buffer,
                offset: 0,
                size: logical_size,
                capacity,
                slot: Some(slot),
                readback: false,
                uniform,
            },
        );
        Ok(handle)
    }

    fn compile_compute_wgsl(
        &self,
        shader: &WebGpuShader,
    ) -> Result<(String, Vec<Option<ResourceAccess>>, WgpuComputeLayout)> {
        if let Some(wgsl) = shader.source.strip_prefix(RAW_WGSL_MARKER) {
            return Ok((
                wgsl.trim_start().to_owned(),
                Vec::new(),
                WgpuComputeLayout::inferred_storage(),
            ));
        }

        let compiler = crate::slang::SlangCompiler::new().context("WebGPU: initialize Slang")?;
        let paths: Vec<&str> = shader.search_paths.iter().map(String::as_str).collect();
        let defines: Vec<(&str, &str)> = shader
            .defines
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        let layout = crate::slang::virtual_main::extract_webgpu_compute_layout(&shader.source)
            .map_err(|error| anyhow::anyhow!("WebGPU shader layout failed: {error}"))?;
        let webgpu_source = crate::slang::virtual_main::transform_virtual_main_webgpu_compute(&shader.source)
            .map_err(|error| anyhow::anyhow!("WebGPU shader lowering failed: {error}"))?;
        let compiled = compiler.compile_bindless_with_reflection_and_defines(
            &webgpu_source,
            crate::slang::ShaderTarget::Wgsl,
            &[("cs_main", crate::slang::SlangStage::Compute)],
            &paths,
            &defines,
            &shader.layout_checks,
            shader.optimization_level,
        )?;
        let source = compiled
            .shader
            .as_str()
            .context("WebGPU: Slang returned non-text WGSL output")?
            .to_owned();
        let access = layout.slot_access();
        Ok((source, access, layout))
    }

    #[cfg(feature = "graphics")]
    fn compile_graphics_stage_wgsl(
        &self,
        shader: &WebGpuShader,
        lowered: &str,
        entry: &'static str,
        stage: crate::slang::SlangStage,
    ) -> Result<String> {
        let compiler = crate::slang::SlangCompiler::new().context("WebGPU: initialize Slang")?;
        let paths: Vec<&str> = shader.search_paths.iter().map(String::as_str).collect();
        let defines: Vec<(&str, &str)> = shader
            .defines
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        let compiled = compiler.compile_bindless_with_reflection_and_defines(
            lowered,
            crate::slang::ShaderTarget::Wgsl,
            &[(entry, stage)],
            &paths,
            &defines,
            &shader.layout_checks,
            shader.optimization_level,
        )?;
        compiled
            .shader
            .as_str()
            .context("WebGPU: Slang returned non-text WGSL output")
            .map(str::to_owned)
    }

    #[cfg(feature = "graphics")]
    fn compile_graphics_wgsl(&self, shader: &WebGpuShader) -> Result<(String, String, WgpuComputeLayout)> {
        let has_goldy = shader.source.contains("[goldy_vertex]") || shader.source.contains("[goldy_fragment]");
        let (lowered, layout) = if has_goldy {
            let layout = crate::slang::virtual_main::extract_webgpu_graphics_layout(&shader.source)
                .map_err(|error| anyhow::anyhow!("WebGPU graphics shader layout failed: {error}"))?;
            let lowered = crate::slang::virtual_main::transform_virtual_main_webgpu_graphics(&shader.source)
                .map_err(|error| anyhow::anyhow!("WebGPU graphics shader lowering failed: {error}"))?;
            (lowered, layout)
        } else {
            (shader.source.clone(), WgpuComputeLayout::inferred_storage())
        };
        let vs = self.compile_graphics_stage_wgsl(shader, &lowered, "vs_main", crate::slang::SlangStage::Vertex)?;
        let fs = self.compile_graphics_stage_wgsl(shader, &lowered, "fs_main", crate::slang::SlangStage::Fragment)?;
        Ok((vs, fs, layout))
    }

    #[cfg(feature = "graphics")]
    fn raster_bind_indices(
        &self,
        layout: &WgpuComputeLayout,
        indices: &[u32],
        frame_table_base: u32,
    ) -> Result<Vec<u32>> {
        if !indices.is_empty() {
            return Ok(indices.to_vec());
        }
        let n = layout.registry_index_count().unwrap_or(0);
        if n == 0 {
            return Ok(Vec::new());
        }
        let table = self
            .last_frame_table
            .as_ref()
            .context("WebGPU: graphics bind needs FrameTableStaging")?;
        let start = frame_table_base as usize;
        let end = start.checked_add(n).context("WebGPU: frame-table range overflow")?;
        anyhow::ensure!(
            end <= table.len(),
            "WebGPU: graphics bind frame-table range [{start}, {end}) exceeds staging len {}",
            table.len()
        );
        Ok(table[start..end].to_vec())
    }

    fn storage_texture_specs(
        &self,
        shader_source: &str,
        layout: &WgpuComputeLayout,
        indices: &[u32],
    ) -> Result<Vec<CudaStorageTextureSpec>> {
        let Some(kinds) = layout.resources.as_ref() else {
            return Ok(Vec::new());
        };
        let texels = crate::slang::virtual_main::webgpu_direct_spatial_texels(shader_source)
            .map_err(|error| anyhow::anyhow!("WebGPU: {error}"))?;
        let mut texel_i = 0usize;
        let mut specs = Vec::new();
        anyhow::ensure!(
            indices.len() == kinds.len(),
            "WebGPU: dispatch bound {} resources, shader expects {}",
            indices.len(),
            kinds.len()
        );
        for (kind, index) in kinds.iter().copied().zip(indices.iter().copied()) {
            if kind != WgpuComputeResourceKind::StorageTexture {
                continue;
            }
            let element = texels
                .get(texel_i)
                .ok_or_else(|| anyhow::anyhow!("WebGPU: missing DirectSpatial element for storage slot {texel_i}"))?;
            texel_i += 1;
            let texture = self.lookup_registry_texture(index)?;
            specs.push(
                CudaStorageTextureSpec::from_element_and_format(element, texture.format)
                    .map_err(|error| anyhow::anyhow!("WebGPU: {error}"))?,
            );
        }
        Ok(specs)
    }

    fn spec_mask(specs: &[CudaStorageTextureSpec]) -> u64 {
        specs.iter().enumerate().fold(0u64, |mask, (i, spec)| {
            let bits = match spec {
                CudaStorageTextureSpec::Identity => 0u64,
                CudaStorageTextureSpec::Float4Rgba8Unorm => 1,
                CudaStorageTextureSpec::Float4Bgra8Unorm => 2,
            };
            mask | (bits << (i * 2))
        })
    }

    fn create_wgpu_compute_pipeline(
        &self,
        device: DeviceHandle,
        wgsl: &str,
        debug_name: Option<&str>,
    ) -> Result<wgpu::ComputePipeline> {
        let gpu = self.device(device)?;
        let error_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: debug_name,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: debug_name,
            layout: None,
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            anyhow::bail!("WebGPU shader/pipeline validation failed: {error}");
        }
        Ok(pipeline)
    }

    fn packed_storage_bindings(
        layout: &WgpuComputeLayout,
        specs: &[CudaStorageTextureSpec],
    ) -> Vec<(u32, &'static str)> {
        let Some(kinds) = layout.resources.as_ref() else {
            return Vec::new();
        };
        let mut spec_i = 0usize;
        let mut bindings = Vec::new();
        for (binding, kind) in kinds.iter().copied().enumerate() {
            if kind != WgpuComputeResourceKind::StorageTexture {
                continue;
            }
            if let Some(format) = specs
                .get(spec_i)
                .copied()
                .unwrap_or_default()
                .wgsl_storage_texel_format()
            {
                bindings.push((binding as u32, format));
            }
            spec_i += 1;
        }
        bindings
    }

    fn wgpu_pipeline_for_dispatch(
        &mut self,
        pipeline_handle: ComputePipelineHandle,
        indices: &[u32],
    ) -> Result<(wgpu::ComputePipeline, WgpuComputeLayout)> {
        let (device, shader, layout, identity) = {
            let pipeline = self
                .compute_pipelines
                .get(&pipeline_handle)
                .context("WebGPU: invalid compute pipeline")?;
            (
                pipeline.device,
                pipeline.shader.clone(),
                pipeline.layout.clone(),
                pipeline.pipeline.clone(),
            )
        };
        let specs = self.storage_texture_specs(&shader.source, &layout, indices)?;
        let mask = Self::spec_mask(&specs);
        if mask == 0 {
            return Ok((identity, layout));
        }
        if let Some(variant) = self
            .compute_pipelines
            .get(&pipeline_handle)
            .and_then(|pipeline| pipeline.variants.get(&mask))
        {
            return Ok((variant.clone(), layout));
        }
        let debug_name = Some("goldy-webgpu-packed-storage");
        let (wgsl, _, _) = self.compile_compute_wgsl(&shader)?;
        let packed = Self::packed_storage_bindings(&layout, &specs);
        let wgsl = patch_wgsl_storage_formats(&wgsl, &packed)?;
        let variant = self.create_wgpu_compute_pipeline(device, &wgsl, debug_name)?;
        self.compute_pipelines
            .get_mut(&pipeline_handle)
            .context("WebGPU: invalid compute pipeline")?
            .variants
            .insert(mask, variant.clone());
        Ok((variant, layout))
    }

    fn lookup_registry_buffer(&self, index: u32) -> Result<&WebGpuBuffer> {
        let handle = self
            .buffer_slots
            .get(&index)
            .with_context(|| format!("WebGPU: unknown buffer registry key {index}"))?;
        self.buffers
            .get(handle)
            .with_context(|| format!("WebGPU: registry key {index} references a destroyed buffer"))
    }

    fn lookup_registry_texture(&self, index: u32) -> Result<&WebGpuTexture> {
        let handle = self
            .texture_slots
            .get(&index)
            .with_context(|| format!("WebGPU: unknown texture registry key {index}"))?;
        self.textures
            .get(handle)
            .with_context(|| format!("WebGPU: registry key {index} references a destroyed texture"))
    }

    fn lookup_registry_sampler(&self, index: u32) -> Result<&WebGpuSampler> {
        let handle = self
            .sampler_slots
            .get(&index)
            .with_context(|| format!("WebGPU: unknown sampler registry key {index}"))?;
        self.samplers
            .get(handle)
            .with_context(|| format!("WebGPU: registry key {index} references a destroyed sampler"))
    }

    fn create_bind_group(
        &self,
        device: DeviceHandle,
        layout: &WgpuComputeLayout,
        bind_layout: impl FnOnce() -> wgpu::BindGroupLayout,
        indices: &[u32],
        user_uniform: Option<(&wgpu::Buffer, u64)>,
    ) -> Result<Option<wgpu::BindGroup>> {
        let mut entries = Vec::new();
        match &layout.resources {
            None => {
                for (binding, index) in indices.iter().copied().enumerate() {
                    let buffer = self.lookup_registry_buffer(index)?;
                    entries.push(wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: wgpu::BindingResource::Buffer(self.storage_binding(buffer)?),
                    });
                }
            }
            Some(kinds) => {
                anyhow::ensure!(
                    indices.len() == kinds.len(),
                    "WebGPU: dispatch bound {} resources, shader expects {}",
                    indices.len(),
                    kinds.len()
                );
                for (binding, (kind, index)) in kinds.iter().copied().zip(indices.iter().copied()).enumerate() {
                    match kind {
                        WgpuComputeResourceKind::StorageReadWrite | WgpuComputeResourceKind::StorageRead => {
                            let buffer = self.lookup_registry_buffer(index)?;
                            entries.push(wgpu::BindGroupEntry {
                                binding: binding as u32,
                                resource: wgpu::BindingResource::Buffer(self.storage_binding(buffer)?),
                            });
                        }
                        WgpuComputeResourceKind::Uniform => {
                            let buffer = self.lookup_registry_buffer(index)?;
                            entries.push(wgpu::BindGroupEntry {
                                binding: binding as u32,
                                resource: wgpu::BindingResource::Buffer(self.uniform_binding(buffer)?),
                            });
                        }
                        WgpuComputeResourceKind::SampledTexture => {
                            let texture = self.lookup_registry_texture(index)?;
                            let view = texture
                                .sampled_view
                                .as_ref()
                                .with_context(|| format!("WebGPU: registry key {index} is not a sampled texture"))?;
                            entries.push(wgpu::BindGroupEntry {
                                binding: binding as u32,
                                resource: wgpu::BindingResource::TextureView(view),
                            });
                        }
                        WgpuComputeResourceKind::StorageTexture => {
                            let texture = self.lookup_registry_texture(index)?;
                            let view = texture
                                .storage_view
                                .as_ref()
                                .with_context(|| format!("WebGPU: registry key {index} is not a storage texture"))?;
                            entries.push(wgpu::BindGroupEntry {
                                binding: binding as u32,
                                resource: wgpu::BindingResource::TextureView(view),
                            });
                        }
                        WgpuComputeResourceKind::Sampler => {
                            let sampler = self.lookup_registry_sampler(index)?;
                            entries.push(wgpu::BindGroupEntry {
                                binding: binding as u32,
                                resource: wgpu::BindingResource::Sampler(&sampler.sampler),
                            });
                        }
                    }
                }
            }
        }
        if layout.scalar_count > 0 {
            let (buffer, offset) = user_uniform.context("WebGPU: scalar dispatch missing user uniform")?;
            entries.push(wgpu::BindGroupEntry {
                binding: entries.len() as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset,
                    size: NonZeroU64::new(USER_UNIFORM_BYTES),
                }),
            });
        }
        if entries.is_empty() {
            return Ok(None);
        }
        let bind_layout = bind_layout();
        let gpu = self.device(device)?;
        let error_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("goldy-webgpu-dispatch-bindings"),
            layout: &bind_layout,
            entries: &entries,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            anyhow::bail!("WebGPU bind group validation failed: {error}");
        }
        Ok(Some(bind_group))
    }

    fn storage_binding<'a>(&self, buffer: &'a WebGpuBuffer) -> Result<wgpu::BufferBinding<'a>> {
        let align = self.device(buffer.device)?.storage_offset_align;
        anyhow::ensure!(
            buffer.offset % align == 0,
            "WebGPU: storage buffer offset {} is not aligned to {align}",
            buffer.offset
        );
        Ok(wgpu::BufferBinding {
            buffer: &buffer.buffer,
            offset: buffer.offset,
            size: NonZeroU64::new(buffer.size),
        })
    }

    fn uniform_binding<'a>(&self, buffer: &'a WebGpuBuffer) -> Result<wgpu::BufferBinding<'a>> {
        let align = self.device(buffer.device)?.uniform_offset_align;
        anyhow::ensure!(
            buffer.offset % align == 0,
            "WebGPU: uniform buffer offset {} is not aligned to {align}",
            buffer.offset
        );
        let remaining = buffer.buffer.size().saturating_sub(buffer.offset);
        let size = align_up(buffer.size.max(16), 16).min(remaining);
        anyhow::ensure!(
            size >= 16,
            "WebGPU: uniform binding for registry buffer is smaller than 16 bytes"
        );
        Ok(wgpu::BufferBinding {
            buffer: &buffer.buffer,
            offset: buffer.offset,
            size: NonZeroU64::new(size),
        })
    }

    fn write_texture_pixels(
        &self,
        texture: TextureHandle,
        data: &[u8],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let tex = self.textures.get(&texture).context("WebGPU: invalid texture handle")?;
        if x + width > tex.width || y + height > tex.height {
            anyhow::bail!(
                "WebGPU: texture write {}x{} at ({x},{y}) exceeds {}x{}",
                width,
                height,
                tex.width,
                tex.height
            );
        }
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(tex.format.bytes_per_pixel() as usize);
        anyhow::ensure!(
            data.len() == expected,
            "WebGPU: texture write expected {expected} bytes, got {}",
            data.len()
        );
        if width == 0 || height == 0 {
            return Ok(());
        }
        let gpu = self.device(tex.device)?;
        gpu.queue.write_texture(
            texel_copy(&tex.texture, x, y),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * tex.format.bytes_per_pixel()),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    fn record_copy_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: TextureHandle,
        dst: TextureHandle,
    ) -> Result<()> {
        let src = self.textures.get(&src).context("WebGPU: invalid copy-texture source")?;
        let dst = self
            .textures
            .get(&dst)
            .context("WebGPU: invalid copy-texture destination")?;
        anyhow::ensure!(
            src.width == dst.width && src.height == dst.height && src.format == dst.format,
            "WebGPU: CopyTexture requires identical size and format"
        );
        let size = wgpu::Extent3d {
            width: src.width,
            height: src.height,
            depth_or_array_layers: 1,
        };
        encoder.copy_texture_to_texture(texel_copy(&src.texture, 0, 0), texel_copy(&dst.texture, 0, 0), size);
        Ok(())
    }

    fn record_copy_buffer_to_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: BufferHandle,
        src_offset: u64,
        src_row_pitch: u32,
        dst: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let src = self
            .buffers
            .get(&src)
            .context("WebGPU: invalid CopyBufferToTexture source")?;
        let dst = self
            .textures
            .get(&dst)
            .context("WebGPU: invalid CopyBufferToTexture destination")?;
        anyhow::ensure!(
            x + width <= dst.width && y + height <= dst.height,
            "WebGPU: CopyBufferToTexture region out of bounds"
        );
        if width == 0 || height == 0 {
            return Ok(());
        }
        let tight = width.saturating_mul(dst.format.bytes_per_pixel());
        let pitch = if src_row_pitch == 0 { tight } else { src_row_pitch };
        anyhow::ensure!(
            pitch >= tight,
            "WebGPU: CopyBufferToTexture row pitch {pitch} < tight {tight}"
        );
        let needed = src_offset
            .checked_add(u64::from(pitch) * u64::from(height.saturating_sub(1)))
            .and_then(|v| v.checked_add(u64::from(tight)))
            .context("WebGPU: CopyBufferToTexture size overflow")?;
        anyhow::ensure!(
            needed <= src.size,
            "WebGPU: CopyBufferToTexture exceeds source buffer (need {needed}, have {})",
            src.size
        );
        let gpu = self.device(dst.device)?;
        if pitch % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0 {
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &src.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: src.offset + src_offset,
                        bytes_per_row: Some(pitch),
                        rows_per_image: Some(height),
                    },
                },
                texel_copy(&dst.texture, x, y),
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            return Ok(());
        }
        let scratch_pitch = copy_row_pitch(width, dst.format);
        let row_copy = align_up(u64::from(tight), wgpu::COPY_BUFFER_ALIGNMENT);
        let scratch = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("goldy-webgpu-b2t-row"),
            size: u64::from(scratch_pitch).max(row_copy),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for row in 0..height {
            encoder.copy_buffer_to_buffer(
                &src.buffer,
                src.offset + src_offset + u64::from(row) * u64::from(pitch),
                &scratch,
                0,
                row_copy,
            );
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &scratch,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(scratch_pitch),
                        rows_per_image: Some(1),
                    },
                },
                texel_copy(&dst.texture, x, y + row),
                wgpu::Extent3d {
                    width,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
        Ok(())
    }

    fn record_copy_texture_to_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: TextureHandle,
        dst: BufferHandle,
        layout: TextureCopyFootprint,
    ) -> Result<()> {
        let src = self
            .textures
            .get(&src)
            .context("WebGPU: invalid texture readback source")?;
        let dst = self
            .buffers
            .get(&dst)
            .context("WebGPU: invalid texture readback staging")?;
        anyhow::ensure!(
            dst.readback,
            "WebGPU: CopyTextureToReadback requires a withdraw staging buffer"
        );
        anyhow::ensure!(
            layout.width == src.width && layout.height == src.height && layout.format == src.format,
            "WebGPU: texture readback footprint does not match source"
        );
        let needed = layout
            .footprint_offset
            .checked_add(layout.staging_bytes)
            .context("WebGPU: texture readback footprint overflow")?;
        anyhow::ensure!(
            needed <= dst.size,
            "WebGPU: CopyTextureToReadback exceeds staging buffer (need {needed}, have {})",
            dst.size
        );
        if layout.width == 0 || layout.height == 0 {
            return Ok(());
        }
        encoder.copy_texture_to_buffer(
            texel_copy(&src.texture, 0, 0),
            wgpu::TexelCopyBufferInfo {
                buffer: &dst.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: dst.offset + layout.footprint_offset,
                    bytes_per_row: Some(layout.row_pitch),
                    rows_per_image: Some(layout.height),
                },
            },
            wgpu::Extent3d {
                width: layout.width,
                height: layout.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    fn ensure_user_uniform(&mut self, device: DeviceHandle, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let (gpu_device, align, capacity) = {
            let gpu = self.device(device)?;
            (
                gpu.device.clone(),
                gpu.uniform_offset_align.max(16),
                gpu.user_uniform_capacity,
            )
        };
        if capacity >= bytes {
            return Ok(());
        }
        let size = align_up(bytes.max(USER_UNIFORM_BYTES), align);
        let buffer = gpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("goldy-webgpu-user-uniform"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gpu = self.devices.get_mut(&device).context("WebGPU: invalid device handle")?;
        gpu.user_uniform = Some(buffer);
        gpu.user_uniform_capacity = size;
        Ok(())
    }

    fn batch_indices(
        &self,
        layout: &WgpuComputeLayout,
        frame_table: Option<&[u32]>,
        arg_data: &[u8],
        count: u32,
        entry: usize,
    ) -> Result<Vec<u32>> {
        let entry_count = count as usize;
        let needed = entry_count
            .checked_mul(DISPATCH_BATCH_STRIDE)
            .context("WebGPU: DispatchBatch stride overflow")?;
        anyhow::ensure!(
            arg_data.len() >= needed,
            "WebGPU: DispatchBatch arg_data len {} < {} entries × stride {}",
            arg_data.len(),
            entry_count,
            DISPATCH_BATCH_STRIDE
        );
        let mut bases = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let base = i * DISPATCH_BATCH_STRIDE;
            let push: PushLayout = *bytemuck::from_bytes(&arg_data[base..base + TOTAL_PUSH_BYTES]);
            bases.push(push._reserved[dispatch_table_base_word_index()]);
        }
        let n = match layout.registry_index_count() {
            Some(n) => n,
            None => {
                anyhow::ensure!(
                    entry_count >= 2,
                    "WebGPU: DispatchBatch with inferred layout requires at least 2 entries"
                );
                let delta = bases[1]
                    .checked_sub(bases[0])
                    .context("WebGPU: invalid frame-table bases")?;
                for window in bases.windows(2) {
                    anyhow::ensure!(
                        window[1].saturating_sub(window[0]) == delta,
                        "WebGPU: DispatchBatch frame-table bases are not uniformly spaced"
                    );
                }
                delta as usize
            }
        };
        if n == 0 {
            return Ok(Vec::new());
        }
        let table =
            frame_table.context("WebGPU: DispatchBatch requires FrameTableStaging when bindings are present")?;
        let start = bases[entry] as usize;
        let end = start.checked_add(n).context("WebGPU: frame-table range overflow")?;
        anyhow::ensure!(
            end <= table.len(),
            "WebGPU: DispatchBatch entry {entry} frame-table range [{start}, {end}) exceeds staging len {}",
            table.len()
        );
        Ok(table[start..end].to_vec())
    }

    fn submit_commands(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
    ) -> Result<crate::timeline::TimelineValue> {
        let context = Arc::clone(self.context(ctx)?);
        let device_handle = context.device;
        let (device, queue, next_timeline, retired, uniform_align) = {
            let gpu = self.device(device_handle)?;
            (
                gpu.device.clone(),
                gpu.queue.clone(),
                Arc::clone(&gpu.next_timeline),
                Arc::clone(&gpu.retired),
                gpu.uniform_offset_align.max(16),
            )
        };

        let mut current_pipeline: Option<ComputePipelineHandle> = None;
        let mut user_slots: Vec<(u64, [u8; USER_UNIFORM_BYTES as usize])> = Vec::new();
        let mut next_user_offset = 0u64;
        for command in commands {
            match command {
                GpuCommand::SetPipeline(pipeline) => current_pipeline = Some(*pipeline),
                GpuCommand::Dispatch { .. } | GpuCommand::DispatchIndirect { .. } => {
                    if let Some(handle) = current_pipeline {
                        let scalar_count = self
                            .compute_pipelines
                            .get(&handle)
                            .map(|pipeline| pipeline.layout.scalar_count)
                            .unwrap_or(0);
                        if scalar_count > 0 {
                            user_slots.push((next_user_offset, pack_user_uniform(&[])));
                            next_user_offset += uniform_align;
                        }
                    }
                }
                GpuCommand::DispatchBatch { count, .. } => {
                    if let Some(handle) = current_pipeline {
                        let scalar_count = self
                            .compute_pipelines
                            .get(&handle)
                            .map(|pipeline| pipeline.layout.scalar_count)
                            .unwrap_or(0);
                        if scalar_count > 0 {
                            for _ in 0..*count {
                                user_slots.push((next_user_offset, pack_user_uniform(&[])));
                                next_user_offset += uniform_align;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Fill user words on the second walk; first walk only reserved offsets.
        // Rebuild slots with actual words below.
        let user_bytes_needed = next_user_offset;
        self.ensure_user_uniform(device_handle, user_bytes_needed)?;
        let user_uniform_buffer = self.device(device_handle)?.user_uniform.clone();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("goldy-webgpu-compute"),
        });
        let mut current_pipeline: Option<ComputePipelineHandle> = None;
        let mut current_indices: Vec<u32> = Vec::new();
        let mut current_user: Vec<u32> = Vec::new();
        let mut frame_table: Option<&[u32]> = None;
        let mut user_slot_i = 0usize;

        for command in commands {
            match command {
                GpuCommand::SetPipeline(pipeline) => current_pipeline = Some(*pipeline),
                GpuCommand::BindResourcesRaw { indices, user, .. } => {
                    current_indices.clone_from(indices);
                    current_user.clone_from(user);
                }
                GpuCommand::Dispatch {
                    label,
                    workgroups_x,
                    workgroups_y,
                    workgroups_z,
                } => {
                    let pipeline_handle = current_pipeline.context("WebGPU: dispatch without a compute pipeline")?;
                    let (wgpu_pipeline, layout) = self.wgpu_pipeline_for_dispatch(pipeline_handle, &current_indices)?;
                    let device = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .context("WebGPU: invalid compute pipeline")?
                        .device;
                    let user_binding = if layout.scalar_count > 0 {
                        require_user_scalars(&current_user, layout.scalar_count)?;
                        let buffer = user_uniform_buffer
                            .as_ref()
                            .context("WebGPU: scalar dispatch missing user uniform buffer")?;
                        let (offset, _) = user_slots
                            .get_mut(user_slot_i)
                            .context("WebGPU: user uniform slot overflow")?;
                        let offset = *offset;
                        queue.write_buffer(buffer, offset, &pack_user_uniform(&current_user));
                        user_slot_i += 1;
                        Some((buffer, offset))
                    } else if !current_user.is_empty() {
                        anyhow::bail!("WebGPU: shader has no scalar parameters but BindResourcesRaw.user is non-empty");
                    } else {
                        None
                    };
                    let bind_group = self.create_bind_group(
                        device,
                        &layout,
                        || wgpu_pipeline.get_bind_group_layout(0),
                        &current_indices,
                        user_binding,
                    )?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: *label,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&wgpu_pipeline);
                    if let Some(bind_group) = bind_group.as_ref() {
                        pass.set_bind_group(0, bind_group, &[]);
                    }
                    pass.dispatch_workgroups(*workgroups_x, *workgroups_y, *workgroups_z);
                }
                GpuCommand::DispatchIndirect { buffer, offset, label } => {
                    let pipeline_handle = current_pipeline.context("WebGPU: indirect dispatch without a pipeline")?;
                    let (wgpu_pipeline, layout) = self.wgpu_pipeline_for_dispatch(pipeline_handle, &current_indices)?;
                    let device = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .context("WebGPU: invalid compute pipeline")?
                        .device;
                    let args = self.buffers.get(buffer).context("WebGPU: invalid indirect buffer")?;
                    ensure_buffer_range(args, *offset, INDIRECT_DISPATCH_BYTES, "DispatchIndirect")?;
                    anyhow::ensure!(
                        *offset % 4 == 0,
                        "WebGPU: DispatchIndirect offset {offset} must be 4-byte aligned"
                    );
                    let args_buffer = args.buffer.clone();
                    let args_offset = args.offset + offset;
                    let user_binding = if layout.scalar_count > 0 {
                        require_user_scalars(&current_user, layout.scalar_count)?;
                        let buffer = user_uniform_buffer
                            .as_ref()
                            .context("WebGPU: scalar dispatch missing user uniform buffer")?;
                        let (off, _) = user_slots
                            .get_mut(user_slot_i)
                            .context("WebGPU: user uniform slot overflow")?;
                        let off = *off;
                        queue.write_buffer(buffer, off, &pack_user_uniform(&current_user));
                        user_slot_i += 1;
                        Some((buffer, off))
                    } else if !current_user.is_empty() {
                        anyhow::bail!("WebGPU: shader has no scalar parameters but BindResourcesRaw.user is non-empty");
                    } else {
                        None
                    };
                    let bind_group = self.create_bind_group(
                        device,
                        &layout,
                        || wgpu_pipeline.get_bind_group_layout(0),
                        &current_indices,
                        user_binding,
                    )?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: *label,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&wgpu_pipeline);
                    if let Some(bind_group) = bind_group.as_ref() {
                        pass.set_bind_group(0, bind_group, &[]);
                    }
                    pass.dispatch_workgroups_indirect(&args_buffer, args_offset);
                }
                GpuCommand::ClearBuffer { buffer, offset, size } => {
                    let buffer = self.buffers.get(buffer).context("WebGPU: invalid clear buffer")?;
                    let clear_size = if *size == 0 {
                        buffer.size.saturating_sub(*offset)
                    } else {
                        *size
                    };
                    ensure_buffer_range(buffer, *offset, clear_size, "ClearBuffer")?;
                    encoder.clear_buffer(&buffer.buffer, buffer.offset + offset, (*size != 0).then_some(*size));
                }
                GpuCommand::WriteBuffer { buffer, offset, data } => {
                    let buffer = self.buffers.get(buffer).context("WebGPU: invalid write buffer")?;
                    ensure_buffer_range(buffer, *offset, data.len() as u64, "WriteBuffer")?;
                    write_queue_buffer(&queue, &buffer.buffer, buffer.offset + offset, data);
                }
                GpuCommand::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let src = self.buffers.get(src).context("WebGPU: invalid copy source")?;
                    let dst = self.buffers.get(dst).context("WebGPU: invalid copy destination")?;
                    ensure_buffer_range(src, *src_offset, *size, "CopyBuffer source")?;
                    ensure_buffer_range(dst, *dst_offset, *size, "CopyBuffer destination")?;
                    encoder.copy_buffer_to_buffer(
                        &src.buffer,
                        src.offset + src_offset,
                        &dst.buffer,
                        dst.offset + dst_offset,
                        *size,
                    );
                }
                GpuCommand::FrameTableStaging { data } => {
                    frame_table = Some(data.as_ref());
                    self.last_frame_table = Some(Arc::clone(data));
                }
                GpuCommand::ResourceBarrier { .. } => {
                    // WebGPU tracks resource transitions within a submitted command buffer.
                }
                GpuCommand::DispatchBatch { arg_data, count, label } => {
                    let pipeline_handle =
                        current_pipeline.context("WebGPU: DispatchBatch without a compute pipeline")?;
                    let device = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .context("WebGPU: invalid compute pipeline")?
                        .device;
                    let layout = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .context("WebGPU: invalid compute pipeline")?
                        .layout
                        .clone();
                    let entry_count = *count as usize;
                    let needed = entry_count
                        .checked_mul(DISPATCH_BATCH_STRIDE)
                        .context("WebGPU: DispatchBatch stride overflow")?;
                    anyhow::ensure!(
                        arg_data.len() >= needed,
                        "WebGPU: DispatchBatch arg_data len {} < {} entries × stride {}",
                        arg_data.len(),
                        entry_count,
                        DISPATCH_BATCH_STRIDE
                    );
                    let n_scalars = layout.scalar_count as usize;
                    for i in 0..entry_count {
                        let base = i * DISPATCH_BATCH_STRIDE;
                        let push: PushLayout = *bytemuck::from_bytes(&arg_data[base..base + TOTAL_PUSH_BYTES]);
                        let wg_off = base + TOTAL_PUSH_BYTES;
                        let workgroups_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into().unwrap());
                        let workgroups_y = u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into().unwrap());
                        let workgroups_z = u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into().unwrap());
                        let indices = self.batch_indices(&layout, frame_table, arg_data, *count, i)?;
                        let user = if n_scalars == 0 {
                            Vec::new()
                        } else {
                            anyhow::ensure!(
                                n_scalars <= MAX_USER_SLOTS,
                                "WebGPU: DispatchBatch entry {i} expects {n_scalars} scalars"
                            );
                            let user = push.user[..n_scalars].to_vec();
                            require_user_scalars(&user, n_scalars as u32)?;
                            user
                        };
                        let user_binding = if n_scalars > 0 {
                            let buffer = user_uniform_buffer
                                .as_ref()
                                .context("WebGPU: scalar DispatchBatch missing user uniform buffer")?;
                            let (offset, _) = user_slots
                                .get_mut(user_slot_i)
                                .context("WebGPU: user uniform slot overflow")?;
                            let offset = *offset;
                            queue.write_buffer(buffer, offset, &pack_user_uniform(&user));
                            user_slot_i += 1;
                            Some((buffer, offset))
                        } else {
                            None
                        };
                        let (wgpu_pipeline, layout) = self.wgpu_pipeline_for_dispatch(pipeline_handle, &indices)?;
                        let bind_group = self.create_bind_group(
                            device,
                            &layout,
                            || wgpu_pipeline.get_bind_group_layout(0),
                            &indices,
                            user_binding,
                        )?;
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: *label,
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&wgpu_pipeline);
                        if let Some(bind_group) = bind_group.as_ref() {
                            pass.set_bind_group(0, bind_group, &[]);
                        }
                        pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
                    }
                }
                GpuCommand::WriteTexture {
                    texture,
                    data,
                    width,
                    height,
                } => {
                    self.write_texture_pixels(*texture, data, 0, 0, *width, *height)?;
                }
                GpuCommand::WriteTextureRegion {
                    texture,
                    x,
                    y,
                    width,
                    height,
                    data,
                } => {
                    self.write_texture_pixels(*texture, data, *x, *y, *width, *height)?;
                }
                GpuCommand::CopyTexture { src, dst } => {
                    self.record_copy_texture(&mut encoder, *src, *dst)?;
                }
                GpuCommand::CopyRenderTarget { src, dst } => {
                    #[cfg(feature = "graphics")]
                    {
                        let src_rt = self
                            .render_targets
                            .get(src)
                            .context("WebGPU: invalid CopyRenderTarget source")?;
                        let dst_tex = self
                            .textures
                            .get(dst)
                            .context("WebGPU: invalid CopyRenderTarget destination")?;
                        anyhow::ensure!(
                            src_rt.width == dst_tex.width
                                && src_rt.height == dst_tex.height
                                && src_rt.format == dst_tex.format,
                            "WebGPU: CopyRenderTarget requires identical size and format"
                        );
                        encoder.copy_texture_to_texture(
                            texel_copy(&src_rt.color, 0, 0),
                            texel_copy(&dst_tex.texture, 0, 0),
                            wgpu::Extent3d {
                                width: src_rt.width,
                                height: src_rt.height,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                    #[cfg(not(feature = "graphics"))]
                    {
                        let _ = (src, dst);
                        anyhow::bail!("WebGPU: CopyRenderTarget requires the graphics feature");
                    }
                }
                GpuCommand::CopyBufferToTexture {
                    src,
                    src_offset,
                    src_row_pitch,
                    dst,
                    x,
                    y,
                    width,
                    height,
                } => {
                    self.record_copy_buffer_to_texture(
                        &mut encoder,
                        *src,
                        *src_offset,
                        *src_row_pitch,
                        *dst,
                        *x,
                        *y,
                        *width,
                        *height,
                    )?;
                }
                GpuCommand::CopyTextureToReadback { src, dst, layout } => {
                    self.record_copy_texture_to_readback(&mut encoder, *src, *dst, *layout)?;
                }
            }
        }

        let index = queue.submit([encoder.finish()]);
        let value = crate::backend::submission_worker::allocate_timeline_value(&next_timeline);
        context.submitted_max.fetch_max(value, Ordering::AcqRel);
        {
            let mut last = context.last_submission.lock().unwrap();
            *last = Some((value, index.clone()));
        }
        if let Some(gpu) = self.devices.get(&device_handle) {
            *gpu.last_submission.lock().unwrap() = Some((value, index.clone()));
        }

        queue.on_submitted_work_done({
            let context = Arc::clone(&context);
            let retired = Arc::clone(&retired);
            move || {
                context.completed.fetch_max(value, Ordering::Release);
                retired.fetch_max(value, Ordering::AcqRel);
                context.signal_queue.push_boundary_crossed(value);
            }
        });
        pump_device(&device);
        Ok(value)
    }
}

fn map_device_type(device_type: wgpu::DeviceType) -> DeviceType {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
        wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
        wgpu::DeviceType::Cpu => DeviceType::Cpu,
        wgpu::DeviceType::Other | wgpu::DeviceType::VirtualGpu => DeviceType::Other,
    }
}

impl GpuBackendSubmitSession for WebGpuBackend {
    fn clone_context_submit_session(
        &self,
        _ctx: ContextHandle,
        backend: std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>,
    ) -> std::sync::Arc<dyn ContextSubmitSession> {
        LockedSubmitSession::with_backend_type(backend, BackendType::WebGpu)
    }
}

impl GpuBackendTimelineWait for WebGpuBackend {
    fn take_timeline_submission_epoch_wait(
        &self,
        _ctx: ContextHandle,
        _value: crate::timeline::TimelineValue,
    ) -> Result<Option<submission_worker::SubmissionEpochWait>> {
        Ok(None)
    }

    fn take_timeline_blocking_wait(
        &self,
        ctx: ContextHandle,
        value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn TimelineBlockingWait>>> {
        if self.gpu_progress(ctx) >= value {
            return Ok(None);
        }
        let context = Arc::clone(self.context(ctx)?);
        if context.submitted_max.load(Ordering::Acquire) < value {
            anyhow::bail!("WebGPU: timeline value {value} was not submitted on context {ctx}");
        }
        let last = context.last_submission.lock().unwrap().clone();
        let Some((submitted, index)) = last else {
            anyhow::bail!("WebGPU: timeline value {value} was not submitted on context {ctx}");
        };
        if submitted < value {
            anyhow::bail!("WebGPU: timeline value {value} was not submitted on context {ctx}");
        }
        let gpu = self.device(context.device)?;
        Ok(Some(Box::new(WebGpuTimelineWait {
            device: gpu.device.clone(),
            index,
            context,
            retired: Arc::clone(&gpu.retired),
            value,
        })))
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        if self.gpu_progress(ctx) >= value {
            return Ok(());
        }
        anyhow::bail!("WebGPU: timeline value {value} was not submitted on context {ctx}")
    }
}

#[cfg(feature = "graphics")]
struct WebGpuSkipPresentGpuWork {
    frame: FrameToken,
    present_timeline: crate::timeline::TimelineValue,
}

#[cfg(feature = "graphics")]
impl PresentGpuWork for WebGpuSkipPresentGpuWork {
    fn run(self: Box<Self>) -> Result<PresentFinishState> {
        Ok(PresentFinishState {
            frame: self.frame,
            return_fence: 0,
            scratch_texture: None,
            scratch_layout_updated: false,
            present_timeline: self.present_timeline,
            copy_timeline: None,
            frame_compute_timeline: None,
            signal_timeline: None,
            render_pass_submitted: false,
            present_ok: false,
        })
    }
}

#[cfg(feature = "graphics")]
struct WebGpuPresentGpuWork {
    frame: FrameToken,
    acquired: wgpu::SurfaceTexture,
    path: WebGpuPresentPath,
    scratch: Option<wgpu::Texture>,
    scratch_view: Option<wgpu::TextureView>,
    scratch_handle: Option<TextureHandle>,
    blit: Option<WebGpuBlitPipeline>,
    width: u32,
    height: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
    context: Arc<WebGpuContext>,
    next_timeline: Arc<AtomicU64>,
    retired: Arc<AtomicU64>,
    device_last_submission: Arc<Mutex<Option<(crate::timeline::TimelineValue, wgpu::SubmissionIndex)>>>,
}

#[cfg(feature = "graphics")]
fn encode_present_blit(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    blit: &WebGpuBlitPipeline,
    src_view: &wgpu::TextureView,
    dst: &wgpu::Texture,
    width: u32,
    height: u32,
) {
    let dest_view = dst.create_view(&wgpu::TextureViewDescriptor {
        label: Some("goldy-webgpu-present-dest"),
        usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("goldy-webgpu-present-blit-bg"),
        layout: &blit.layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(src_view),
        }],
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("goldy-webgpu-present-blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dest_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&blit.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    let _ = (width, height);
}

#[cfg(feature = "graphics")]
impl PresentGpuWork for WebGpuPresentGpuWork {
    fn run(self: Box<Self>) -> Result<PresentFinishState> {
        let value = match self.path {
            WebGpuPresentPath::Direct => {
                self.acquired.present();
                let value = self.context.submitted_max.load(Ordering::Acquire);
                return Ok(PresentFinishState {
                    frame: self.frame,
                    return_fence: value,
                    scratch_texture: None,
                    scratch_layout_updated: false,
                    present_timeline: value,
                    copy_timeline: None,
                    frame_compute_timeline: None,
                    signal_timeline: Some(value),
                    render_pass_submitted: false,
                    present_ok: true,
                });
            }
            WebGpuPresentPath::Copy | WebGpuPresentPath::Blit => {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("goldy-webgpu-present"),
                });
                let scratch = self.scratch.as_ref().context("WebGPU: present missing scratch")?;
                match self.path {
                    WebGpuPresentPath::Copy => {
                        encoder.copy_texture_to_texture(
                            texel_copy(scratch, 0, 0),
                            texel_copy(&self.acquired.texture, 0, 0),
                            wgpu::Extent3d {
                                width: self.width,
                                height: self.height,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                    WebGpuPresentPath::Blit => {
                        let blit = self.blit.as_ref().context("WebGPU: blit present missing pipeline")?;
                        let src_view = self
                            .scratch_view
                            .as_ref()
                            .context("WebGPU: blit present missing sampled scratch")?;
                        encode_present_blit(
                            &mut encoder,
                            &self.device,
                            blit,
                            src_view,
                            &self.acquired.texture,
                            self.width,
                            self.height,
                        );
                    }
                    WebGpuPresentPath::Direct => unreachable!(),
                }
                let index = self.queue.submit([encoder.finish()]);
                let value = crate::backend::submission_worker::allocate_timeline_value(&self.next_timeline);
                self.context.submitted_max.fetch_max(value, Ordering::AcqRel);
                *self.context.last_submission.lock().unwrap() = Some((value, index.clone()));
                *self.device_last_submission.lock().unwrap() = Some((value, index.clone()));
                self.queue.on_submitted_work_done({
                    let context = Arc::clone(&self.context);
                    let retired = Arc::clone(&self.retired);
                    move || {
                        context.completed.fetch_max(value, Ordering::Release);
                        retired.fetch_max(value, Ordering::AcqRel);
                        context.signal_queue.push_boundary_crossed(value);
                    }
                });
                self.acquired.present();
                poll_device(
                    &self.device,
                    wgpu::PollType::Wait {
                        submission_index: Some(index),
                        timeout: Some(TIMELINE_WAIT_TIMEOUT),
                    },
                )?;
                value
            }
        };
        Ok(PresentFinishState {
            frame: self.frame,
            return_fence: value,
            scratch_texture: self.scratch_handle,
            scratch_layout_updated: false,
            present_timeline: value,
            copy_timeline: Some(value),
            frame_compute_timeline: None,
            signal_timeline: Some(value),
            render_pass_submitted: self.path == WebGpuPresentPath::Blit,
            present_ok: true,
        })
    }
}

#[cfg(feature = "graphics")]
impl GpuBackendPresentSplit for WebGpuBackend {
    fn take_present_gpu_work(
        &mut self,
        frame: FrameToken,
        submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn PresentGpuWork>> {
        let surface_device = self
            .surfaces
            .get(&frame.surface)
            .context("WebGPU: invalid surface handle")?
            .device;
        anyhow::ensure!(
            self.context_device(frame.context) == surface_device,
            "WebGPU: present context does not match the surface device"
        );
        let acquired = {
            let surface_state = self
                .surfaces
                .get_mut(&frame.surface)
                .context("WebGPU: invalid surface handle")?;
            surface_state.acquired.take()
        };
        let Some(acquired) = acquired else {
            return Ok(Box::new(WebGpuSkipPresentGpuWork {
                frame,
                present_timeline: submit_tv,
            }));
        };
        let (path, scratch_handle, width, height, swapchain_format) = {
            let surface_state = self
                .surfaces
                .get(&frame.surface)
                .context("WebGPU: invalid surface handle")?;
            (
                surface_state.present_path,
                surface_state.scratch,
                surface_state.width.max(1),
                surface_state.height.max(1),
                surface_state.swapchain_format,
            )
        };
        anyhow::ensure!(
            acquired.texture.size().width >= width && acquired.texture.size().height >= height,
            "WebGPU: acquired swapchain image is smaller than the surface"
        );
        let (scratch, scratch_view, blit) = match path {
            WebGpuPresentPath::Direct => (None, None, None),
            WebGpuPresentPath::Copy => {
                let handle = scratch_handle.context("WebGPU: copy present without a storage scratch")?;
                let scratch = self
                    .textures
                    .get(&handle)
                    .context("WebGPU: present scratch was destroyed")?;
                anyhow::ensure!(
                    scratch.width == width && scratch.height == height,
                    "WebGPU: present scratch size {}x{} does not match surface {width}x{height}",
                    scratch.width,
                    scratch.height
                );
                let scratch_format = map_texture_format(scratch.format);
                anyhow::ensure!(
                    scratch_format == swapchain_format && acquired.texture.format() == swapchain_format,
                    "WebGPU: scratch format {scratch_format:?} cannot copy to swapchain {swapchain_format:?}"
                );
                (Some(scratch.texture.clone()), None, None)
            }
            WebGpuPresentPath::Blit => {
                let handle = scratch_handle.context("WebGPU: blit present without a storage scratch")?;
                let (texture, view, src_width, src_height) = {
                    let scratch = self
                        .textures
                        .get(&handle)
                        .context("WebGPU: present scratch was destroyed")?;
                    (
                        scratch.texture.clone(),
                        scratch
                            .sampled_view
                            .clone()
                            .context("WebGPU: blit scratch is missing a sampled view")?,
                        scratch.width,
                        scratch.height,
                    )
                };
                anyhow::ensure!(
                    src_width == width && src_height == height,
                    "WebGPU: present scratch size {src_width}x{src_height} does not match surface {width}x{height}"
                );
                let blit = self.blit_pipeline(surface_device, swapchain_format)?;
                (Some(texture), Some(view), Some(blit))
            }
        };
        let gpu = self.device(surface_device)?;
        let context = Arc::clone(self.context(frame.context)?);
        Ok(Box::new(WebGpuPresentGpuWork {
            frame,
            acquired,
            path,
            scratch,
            scratch_view,
            scratch_handle,
            blit,
            width,
            height,
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            context,
            next_timeline: Arc::clone(&gpu.next_timeline),
            retired: Arc::clone(&gpu.retired),
            device_last_submission: Arc::clone(&gpu.last_submission),
        }))
    }

    fn finish_present(
        &mut self,
        finish: PresentFinishState,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        let ctx = finish.frame.context;
        let surface = finish.frame.surface;
        {
            let state = self
                .surfaces
                .get_mut(&surface)
                .context("WebGPU: invalid surface handle")?;
            // Drawable was moved into PresentGpuWork (or never acquired on skip).
            state.acquired = None;
            if finish.present_ok {
                state.current_texture_handle = finish.scratch_texture.or(state.scratch);
            }
        }
        self.drop_surface_lease(surface);
        if let Some(context) = self.contexts.get(&ctx) {
            pump_device(&context.wgpu_device);
            if finish.present_ok {
                context.signal_queue.push(crate::signal::Signal::SwapchainReturned {
                    image_index: finish.frame.image as u32,
                });
            }
        }
        Ok(finish.present_timeline)
    }
}

impl GpuBackend for WebGpuBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backend_type(&self) -> BackendType {
        BackendType::WebGpu
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        self.adapter_info.clone()
    }

    fn adapter_capabilities(&self, _adapter_id: u32) -> crate::device::DeviceCapabilities {
        crate::device::DeviceCapabilities {
            preferred_surface_format: TextureFormat::Bgra8Unorm,
            preferred_render_target_format: TextureFormat::Rgba8Unorm,
            supported_surface_formats: vec![
                TextureFormat::Rgba8Unorm,
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgba8UnormSrgb,
                TextureFormat::Bgra8UnormSrgb,
            ],
            supported_render_target_formats: vec![
                TextureFormat::Rgba8Unorm,
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgba8UnormSrgb,
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Rgba16Float,
                TextureFormat::Rgba32Float,
            ],
            has_zero_copy_storage_readback: false,
            buffer_resize_cost: BufferResizeCost::Copy,
            buffer_decommit_supported: false,
            host_sidecar_on_submit_worker: false,
            split_compute_partitions_on_barrier_cost: false,
            fuse_upload_with_compute_partitions: true,
            ..crate::device::DeviceCapabilities::default()
        }
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        let adapter = self
            .adapters
            .get(adapter_id as usize)
            .with_context(|| format!("WebGPU: invalid adapter id {adapter_id}"))?;
        let wanted = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::FLOAT32_FILTERABLE
            | wgpu::Features::BGRA8UNORM_STORAGE
            | wgpu::Features::VERTEX_WRITABLE_STORAGE;
        let required_features = adapter.features() & wanted;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("goldy-webgpu-device"),
            required_features,
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .context("WebGPU: request device")?;
        // Default handler panics on validation errors (including those we turn into Result).
        device.on_uncaptured_error(Arc::new(|error| {
            eprintln!("WebGPU uncaptured error: {error}");
        }));
        let uniform_offset_align = device.limits().min_uniform_buffer_offset_alignment.max(16) as u64;
        let storage_offset_align = device.limits().min_storage_buffer_offset_alignment.max(4) as u64;
        let handle = self.next_device;
        self.next_device += 1;
        self.devices.insert(
            handle,
            WebGpuDevice {
                device,
                queue,
                features: required_features,
                next_timeline: Arc::new(AtomicU64::new(1)),
                retired: Arc::new(AtomicU64::new(0)),
                last_submission: Arc::new(Mutex::new(None)),
                user_uniform: None,
                user_uniform_capacity: 0,
                uniform_offset_align,
                storage_offset_align,
                adapter_id,
                #[cfg(feature = "graphics")]
                blit: HashMap::new(),
            },
        );
        Ok(handle)
    }

    fn destroy_device(&mut self, device: DeviceHandle) {
        #[cfg(feature = "graphics")]
        {
            let surface_handles: Vec<_> = self
                .surfaces
                .iter()
                .filter(|(_, surface)| surface.device == device)
                .map(|(handle, _)| *handle)
                .collect();
            for handle in surface_handles {
                self.destroy_surface(handle);
            }
        }
        if let Some(gpu) = self.devices.remove(&device) {
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.contexts.retain(|_, context| context.device != device);
        self.buffers.retain(|_, buffer| buffer.device != device);
        self.textures.retain(|_, texture| texture.device != device);
        self.samplers.retain(|_, sampler| sampler.device != device);
        self.shaders.retain(|_, shader| shader.device != device);
        self.compute_pipelines.retain(|_, pipeline| pipeline.device != device);
        self.buffer_slots.retain(|_, handle| self.buffers.contains_key(handle));
        self.texture_slots
            .retain(|_, handle| self.textures.contains_key(handle));
        self.sampler_slots
            .retain(|_, handle| self.samplers.contains_key(handle));
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn device_wait_idle(&mut self, device: DeviceHandle) -> Result<()> {
        poll_device(&self.device(device)?.device, wgpu::PollType::wait_indefinitely()).map(|_| ())
    }

    fn create_context(&mut self, device: DeviceHandle) -> Result<ContextHandle> {
        let wgpu_device = self.device(device)?.device.clone();
        let handle = self.next_context;
        self.next_context += 1;
        self.contexts.insert(
            handle,
            Arc::new(WebGpuContext {
                device,
                wgpu_device,
                completed: AtomicU64::new(0),
                submitted_max: AtomicU64::new(0),
                last_submission: Mutex::new(None),
                signal_queue: crate::signal::SignalQueue::new(),
            }),
        );
        Ok(handle)
    }

    fn detach_context_for_destroy(&mut self, ctx: ContextHandle) -> Option<Box<dyn ContextDestroyHandle>> {
        let context = self.contexts.remove(&ctx)?;
        Some(Box::new(WebGpuDestroyContext {
            device: context.wgpu_device.clone(),
        }))
    }

    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn ContextDeferredDeletionFlush>> {
        self.contexts
            .contains_key(&ctx)
            .then(|| Arc::new(NoOpDeferredDeletionFlush) as Arc<dyn ContextDeferredDeletionFlush>)
    }

    fn clone_context_gpu_progress(&self, ctx: ContextHandle) -> Option<std::sync::Arc<dyn ContextGpuProgress>> {
        Some(Arc::new(WebGpuProgress {
            context: Arc::clone(self.contexts.get(&ctx)?),
        }))
    }

    fn context_device(&self, ctx: ContextHandle) -> DeviceHandle {
        self.contexts.get(&ctx).map(|context| context.device).unwrap_or(0)
    }

    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        access: BufferKind,
        _element_stride: Option<u32>,
        _flags: BufferFlags,
    ) -> Result<BufferHandle> {
        let uniform = matches!(access, BufferKind::Broadcast);
        let capacity = if uniform { align_up(size.max(16), 16) } else { size };
        self.create_storage_buffer(device, size, capacity, uniform)
    }

    fn create_buffer_with_capacity(
        &mut self,
        device: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        access: BufferKind,
        _element_stride: Option<u32>,
        _flags: BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        let uniform = matches!(access, BufferKind::Broadcast);
        let capacity = if uniform {
            align_up(capacity.max(initial_size).max(16), 16)
        } else {
            capacity.max(initial_size)
        };
        Ok((
            self.create_storage_buffer(device, initial_size, capacity, uniform)?,
            capacity,
        ))
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        if let Some(buffer) = self.buffers.remove(&buffer) {
            if let Some(slot) = buffer.slot {
                self.buffer_slots.remove(&slot);
                self.recycle_registry_slot(slot);
            }
        }
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buffer = self.buffers.get(&buffer).context("WebGPU: invalid buffer handle")?;
        if offset + data.len() as u64 > buffer.size {
            anyhow::bail!("WebGPU: write exceeds logical buffer size");
        }
        write_queue_buffer(
            &self.device(buffer.device)?.queue,
            &buffer.buffer,
            buffer.offset + offset,
            data,
        );
        Ok(())
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        let gpu = self.device(device)?;
        let capacity = align_up(size.max(4), wgpu::COPY_BUFFER_ALIGNMENT);
        let raw = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("goldy-webgpu-readback"),
            size: capacity,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let handle = self.next_buffer;
        self.next_buffer += 1;
        self.buffers.insert(
            handle,
            WebGpuBuffer {
                device,
                buffer: raw,
                offset: 0,
                size,
                capacity,
                slot: None,
                readback: true,
                uniform: false,
            },
        );
        Ok(handle)
    }

    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        let buffer = self.buffers.get(&buffer).context("WebGPU: invalid readback buffer")?;
        if !buffer.readback {
            anyhow::bail!("WebGPU: buffer is not readback staging");
        }
        if output.len() as u64 > buffer.size {
            anyhow::bail!("WebGPU: read exceeds readback buffer size");
        }
        let slice = buffer.buffer.slice(0..output.len() as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        poll_device(&self.device(buffer.device)?.device, wgpu::PollType::wait_indefinitely())?;
        rx.recv().context("WebGPU readback callback dropped")??;
        output.copy_from_slice(&slice.get_mapped_range());
        buffer.buffer.unmap();
        Ok(())
    }

    fn free_readback_buffer(&mut self, buffer: BufferHandle) {
        self.destroy_buffer(buffer);
    }

    fn query_texture_copy_footprint(
        &self,
        _device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<TextureCopyFootprint> {
        Ok(texture_copy_footprint(width, height, format))
    }

    fn alloc_texture_readback_staging(
        &mut self,
        device: DeviceHandle,
        layout: TextureCopyFootprint,
    ) -> Result<BufferHandle> {
        self.alloc_readback_buffer(device, layout.staging_bytes.max(1))
    }

    fn read_texture_readback_staging(
        &self,
        buffer: BufferHandle,
        layout: TextureCopyFootprint,
        output: &mut [u8],
    ) -> Result<()> {
        let staging_len = layout.staging_bytes as usize;
        let mut staging = vec![0u8; staging_len];
        self.read_readback_buffer(buffer, &mut staging)?;
        unpack_texture_rows(layout, &staging, output)
    }

    fn texture_copy_retention_tag(&self, _texture: TextureHandle) -> u64 {
        0
    }

    fn clear_buffer(&mut self, device: DeviceHandle, buffer: BufferHandle, offset: u64, size: u64) -> Result<()> {
        let (wgpu_device, queue, raw, abs_offset, clear_size) = {
            let gpu = self.device(device)?;
            let target = self.buffers.get(&buffer).context("WebGPU: invalid buffer handle")?;
            if target.device != device {
                anyhow::bail!("WebGPU: buffer belongs to another device");
            }
            let clear_size = if size == 0 {
                target.size.saturating_sub(offset)
            } else {
                size
            };
            ensure_buffer_range(target, offset, clear_size, "clear_buffer")?;
            (
                gpu.device.clone(),
                gpu.queue.clone(),
                target.buffer.clone(),
                target.offset + offset,
                (size != 0).then_some(size),
            )
        };
        let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("goldy-webgpu-clear"),
        });
        encoder.clear_buffer(&raw, abs_offset, clear_size);
        let index = queue.submit([encoder.finish()]);
        poll_device(
            &wgpu_device,
            wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            },
        )?;
        Ok(())
    }

    fn buffer_size(&self, buffer: BufferHandle) -> u64 {
        self.buffers.get(&buffer).map(|buffer| buffer.size).unwrap_or(0)
    }

    fn buffer_capacity(&self, buffer: BufferHandle) -> u64 {
        self.buffers.get(&buffer).map(|buffer| buffer.capacity).unwrap_or(0)
    }

    fn set_buffer_logical_size(
        &mut self,
        _device: DeviceHandle,
        buffer: BufferHandle,
        new_logical_size: u64,
    ) -> Result<()> {
        let buffer = self.buffers.get_mut(&buffer).context("WebGPU: invalid buffer handle")?;
        if new_logical_size == 0 || new_logical_size > buffer.capacity {
            anyhow::bail!("WebGPU: logical size must be in 1..=capacity");
        }
        buffer.size = new_logical_size;
        Ok(())
    }

    fn buffer_bindless_index(&self, buffer: BufferHandle) -> Option<u32> {
        self.buffers.get(&buffer)?.slot
    }

    fn buffer_bindless_srv_index(&self, buffer: BufferHandle) -> Option<u32> {
        self.buffer_bindless_index(buffer)
    }

    fn create_buffer_view(
        &mut self,
        parent: BufferHandle,
        offset: u64,
        size: u64,
        _element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        let parent = self
            .buffers
            .get(&parent)
            .context("WebGPU: invalid parent buffer")?
            .clone();
        if offset + size > parent.size {
            anyhow::bail!("WebGPU: buffer view exceeds parent");
        }
        let abs_offset = parent.offset + offset;
        let gpu = self.device(parent.device)?;
        let align = if parent.uniform {
            gpu.uniform_offset_align
        } else {
            gpu.storage_offset_align
        };
        anyhow::ensure!(
            abs_offset % align == 0,
            "WebGPU: buffer view offset {abs_offset} is not aligned to {align}"
        );
        let handle = self.next_buffer;
        self.next_buffer += 1;
        let slot = self.alloc_registry_slot()?;
        self.buffer_slots.insert(slot, handle);
        self.buffers.insert(
            handle,
            WebGpuBuffer {
                device: parent.device,
                buffer: parent.buffer,
                offset: parent.offset + offset,
                size,
                capacity: size,
                slot: Some(slot),
                readback: false,
                uniform: parent.uniform,
            },
        );
        Ok(handle)
    }

    fn resize_buffer(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        new_size: u64,
        preserve_contents: bool,
    ) -> Result<()> {
        let old = self
            .buffers
            .get(&buffer)
            .context("WebGPU: invalid buffer handle")?
            .clone();
        if old.device != device {
            anyhow::bail!("WebGPU: buffer belongs to another device");
        }
        let gpu = self.device(device)?;
        let capacity = if old.uniform {
            align_up(new_size.max(16), 16)
        } else {
            align_up(new_size.max(4), wgpu::COPY_BUFFER_ALIGNMENT)
        };
        let replacement = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("goldy-webgpu-resized-buffer"),
            size: capacity,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });
        if preserve_contents {
            let copy_size = old.size.min(new_size);
            if copy_size > 0 {
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("goldy-webgpu-resize-copy"),
                });
                encoder.copy_buffer_to_buffer(&old.buffer, old.offset, &replacement, 0, copy_size);
                gpu.queue.submit([encoder.finish()]);
                poll_device(&gpu.device, wgpu::PollType::wait_indefinitely())?;
            }
        }
        let target = self.buffers.get_mut(&buffer).expect("validated above");
        target.buffer = replacement;
        target.offset = 0;
        target.size = new_size;
        target.capacity = capacity;
        Ok(())
    }

    fn create_shader_with_paths(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
    ) -> Result<ShaderHandle> {
        self.device(device)?;
        let handle = self.next_shader;
        self.next_shader += 1;
        self.shaders.insert(
            handle,
            WebGpuShader {
                device,
                source: slang_source.to_owned(),
                search_paths: search_paths.iter().map(|value| (*value).to_owned()).collect(),
                defines: defines
                    .iter()
                    .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                    .collect(),
                optimization_level,
                layout_checks: Vec::new(),
            },
        );
        Ok(handle)
    }

    fn create_shader_with_checks(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
        layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
    ) -> Result<ShaderHandle> {
        self.device(device)?;
        let handle = self.next_shader;
        self.next_shader += 1;
        self.shaders.insert(
            handle,
            WebGpuShader {
                device,
                source: slang_source.to_owned(),
                search_paths: search_paths.iter().map(|value| (*value).to_owned()).collect(),
                defines: defines
                    .iter()
                    .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                    .collect(),
                optimization_level,
                layout_checks,
            },
        );
        Ok(handle)
    }

    fn destroy_shader(&mut self, shader: ShaderHandle) {
        self.shaders.remove(&shader);
    }

    #[cfg(feature = "graphics")]
    fn create_pipeline(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        self.create_pipeline_with_depth(
            device,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
            None,
        )
    }

    #[cfg(feature = "graphics")]
    fn destroy_pipeline(&mut self, pipeline: PipelineHandle) {
        self.graphics_pipelines.remove(&pipeline);
    }

    #[cfg(feature = "graphics")]
    fn create_pipeline_with_depth(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        depth_stencil: Option<&DepthStencilState>,
    ) -> Result<PipelineHandle> {
        let vs_shader = self
            .shaders
            .get(&vertex_shader)
            .context("WebGPU: invalid vertex shader")?
            .clone();
        anyhow::ensure!(
            vs_shader.device == device,
            "WebGPU: vertex shader belongs to another device"
        );
        let fs_shader = self
            .shaders
            .get(&fragment_shader)
            .context("WebGPU: invalid fragment shader")?
            .clone();
        anyhow::ensure!(
            fs_shader.device == device,
            "WebGPU: fragment shader belongs to another device"
        );
        let (vs_wgsl, fs_wgsl, vs_layout) = self.compile_graphics_wgsl(&vs_shader)?;
        let (_fs_only, fs_wgsl_other, fs_layout) = if vertex_shader == fragment_shader {
            (String::new(), fs_wgsl.clone(), vs_layout.clone())
        } else {
            self.compile_graphics_wgsl(&fs_shader)?
        };
        let fs_wgsl = if vertex_shader == fragment_shader {
            fs_wgsl
        } else {
            fs_wgsl_other
        };
        let layout = if fs_layout.resources.as_ref().is_some_and(|r| !r.is_empty()) || fs_layout.scalar_count > 0 {
            fs_layout
        } else {
            vs_layout
        };

        let gpu = self.device(device)?;
        let error_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let vs_module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("goldy-webgpu-vs"),
            source: wgpu::ShaderSource::Wgsl(vs_wgsl.into()),
        });
        let fs_module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("goldy-webgpu-fs"),
            source: wgpu::ShaderSource::Wgsl(fs_wgsl.into()),
        });

        let attributes: Vec<wgpu::VertexAttribute> = vertex_layout
            .attributes
            .iter()
            .map(|attr| wgpu::VertexAttribute {
                format: map_vertex_format(attr.format),
                offset: u64::from(attr.offset),
                shader_location: attr.location,
            })
            .collect();
        let vertex_buffers: Vec<wgpu::VertexBufferLayout<'_>> = if attributes.is_empty() {
            Vec::new()
        } else {
            vec![wgpu::VertexBufferLayout {
                array_stride: u64::from(vertex_layout.stride),
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attributes,
            }]
        };
        let strip_index_format = matches!(
            topology,
            PrimitiveTopology::LineStrip | PrimitiveTopology::TriangleStrip
        )
        .then_some(wgpu::IndexFormat::Uint16);
        let depth_stencil = depth_stencil.map(|ds| wgpu::DepthStencilState {
            format: map_depth_format(ds.format),
            depth_write_enabled: ds.depth_write_enabled,
            depth_compare: map_compare(ds.depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("goldy-webgpu-graphics"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: map_topology(topology),
                strip_index_format,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: map_texture_format(target_format),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            anyhow::bail!("WebGPU graphics pipeline validation failed: {error}");
        }

        let handle = self.next_graphics_pipeline;
        self.next_graphics_pipeline += 1;
        self.graphics_pipelines.insert(
            handle,
            WebGpuGraphicsPipeline {
                device,
                pipeline,
                layout,
                vertex_stride: vertex_layout.stride,
                topology,
            },
        );
        Ok(handle)
    }

    #[cfg(feature = "graphics")]
    fn create_render_target_with_depth(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        anyhow::ensure!(width > 0 && height > 0, "WebGPU: render target size must be non-zero");
        let gpu = self.device(device)?;
        let color = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("goldy-webgpu-rt-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: map_texture_format(color_format),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor {
            label: Some("goldy-webgpu-rt-color-view"),
            usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            ..Default::default()
        });
        let depth = depth_format
            .map(|format| -> Result<_> {
                let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("goldy-webgpu-rt-depth"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: map_depth_format(format),
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("goldy-webgpu-rt-depth-view"),
                    usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    ..Default::default()
                });
                Ok((texture, view))
            })
            .transpose()?;
        let handle = self.next_render_target;
        self.next_render_target += 1;
        self.render_targets.insert(
            handle,
            WebGpuRenderTarget {
                device,
                width,
                height,
                format: color_format,
                color,
                color_view,
                depth,
            },
        );
        Ok(handle)
    }

    #[cfg(feature = "graphics")]
    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        self.render_targets.remove(&target);
    }

    #[cfg(feature = "graphics")]
    fn render_to_target(
        &mut self,
        device: DeviceHandle,
        target: RenderTargetHandle,
        color_load: crate::types::TargetLoad,
        commands: &[RenderCommand],
    ) -> Result<()> {
        let (color_view, depth_view, width, height) = {
            let rt = self
                .render_targets
                .get(&target)
                .context("WebGPU: invalid render target")?;
            anyhow::ensure!(rt.device == device, "WebGPU: render target belongs to another device");
            (
                rt.color_view.clone(),
                rt.depth.as_ref().map(|(_, view)| view.clone()),
                rt.width,
                rt.height,
            )
        };
        let clear_depth = commands.iter().find_map(|cmd| match cmd {
            RenderCommand::ClearDepth(depth) => Some(*depth),
            _ => None,
        });
        let gpu = self.device(device)?;
        let queue = gpu.queue.clone();
        let wgpu_device = gpu.device.clone();
        let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("goldy-webgpu-render"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("goldy-webgpu-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: map_color_load(color_load),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_view.as_ref().map(|view| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: match clear_depth {
                            Some(depth) => wgpu::LoadOp::Clear(depth),
                            None => wgpu::LoadOp::Load,
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, width, height);

            let mut current_pipeline: Option<PipelineHandle> = None;
            let mut bind_group_keep: Option<wgpu::BindGroup> = None;
            for command in commands {
                match command {
                    RenderCommand::ClearDepth(_) => {}
                    RenderCommand::SetPipeline(handle) => {
                        let pipeline = self
                            .graphics_pipelines
                            .get(handle)
                            .context("WebGPU: invalid graphics pipeline")?;
                        anyhow::ensure!(pipeline.device == device, "WebGPU: pipeline belongs to another device");
                        pass.set_pipeline(&pipeline.pipeline);
                        current_pipeline = Some(*handle);
                    }
                    RenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                        let buf = self.buffers.get(buffer).context("WebGPU: invalid vertex buffer")?;
                        ensure_buffer_range(buf, *offset, 0, "vertex buffer")?;
                        pass.set_vertex_buffer(*slot, buf.buffer.slice(buf.offset + offset..));
                    }
                    RenderCommand::SetIndexBuffer { buffer, offset, format } => {
                        let buf = self.buffers.get(buffer).context("WebGPU: invalid index buffer")?;
                        ensure_buffer_range(buf, *offset, 0, "index buffer")?;
                        pass.set_index_buffer(buf.buffer.slice(buf.offset + offset..), map_index_format(*format));
                    }
                    RenderCommand::BindResources { buffers } => {
                        let pipeline_handle =
                            current_pipeline.context("WebGPU: BindResources without a graphics pipeline")?;
                        let layout = self
                            .graphics_pipelines
                            .get(&pipeline_handle)
                            .context("WebGPU: invalid graphics pipeline")?
                            .layout
                            .clone();
                        let pipeline = self
                            .graphics_pipelines
                            .get(&pipeline_handle)
                            .context("WebGPU: invalid graphics pipeline")?
                            .pipeline
                            .clone();
                        let indices: Vec<u32> = buffers
                            .iter()
                            .map(|h| {
                                self.buffer_bindless_index(*h)
                                    .context("WebGPU: BindResources buffer missing registry key")
                            })
                            .collect::<Result<_>>()?;
                        let bg = self.create_bind_group(
                            device,
                            &layout,
                            || pipeline.get_bind_group_layout(0),
                            &indices,
                            None,
                        )?;
                        if let Some(bg) = bg.as_ref() {
                            pass.set_bind_group(0, bg, &[]);
                        }
                        bind_group_keep = bg;
                    }
                    RenderCommand::BindResourcesTyped { handles } => {
                        let pipeline_handle =
                            current_pipeline.context("WebGPU: BindResourcesTyped without a graphics pipeline")?;
                        let layout = self
                            .graphics_pipelines
                            .get(&pipeline_handle)
                            .context("WebGPU: invalid graphics pipeline")?
                            .layout
                            .clone();
                        let pipeline = self
                            .graphics_pipelines
                            .get(&pipeline_handle)
                            .context("WebGPU: invalid graphics pipeline")?
                            .pipeline
                            .clone();
                        let indices: Vec<u32> = handles.iter().map(|h| h.index()).collect();
                        let bg = self.create_bind_group(
                            device,
                            &layout,
                            || pipeline.get_bind_group_layout(0),
                            &indices,
                            None,
                        )?;
                        if let Some(bg) = bg.as_ref() {
                            pass.set_bind_group(0, bg, &[]);
                        }
                        bind_group_keep = bg;
                    }
                    RenderCommand::BindResourcesRaw {
                        indices,
                        user,
                        frame_table_base,
                    } => {
                        let pipeline_handle =
                            current_pipeline.context("WebGPU: BindResourcesRaw without a graphics pipeline")?;
                        let layout = self
                            .graphics_pipelines
                            .get(&pipeline_handle)
                            .context("WebGPU: invalid graphics pipeline")?
                            .layout
                            .clone();
                        let pipeline = self
                            .graphics_pipelines
                            .get(&pipeline_handle)
                            .context("WebGPU: invalid graphics pipeline")?
                            .pipeline
                            .clone();
                        let resolved = self.raster_bind_indices(&layout, indices, *frame_table_base)?;
                        let user_binding = if layout.scalar_count > 0 {
                            require_user_scalars(user, layout.scalar_count)?;
                            self.ensure_user_uniform(device, USER_UNIFORM_BYTES)?;
                            let gpu = self.device(device)?;
                            let buffer = gpu
                                .user_uniform
                                .as_ref()
                                .context("WebGPU: scalar draw missing user uniform buffer")?;
                            gpu.queue.write_buffer(buffer, 0, &pack_user_uniform(user));
                            Some((buffer.clone(), 0u64))
                        } else if !user.is_empty() {
                            anyhow::bail!(
                                "WebGPU: graphics shader has no scalar parameters but BindResourcesRaw.user is non-empty"
                            );
                        } else {
                            None
                        };
                        let user_ref = user_binding.as_ref().map(|(b, off)| (b, *off));
                        let bg = self.create_bind_group(
                            device,
                            &layout,
                            || pipeline.get_bind_group_layout(0),
                            &resolved,
                            user_ref,
                        )?;
                        if let Some(bg) = bg.as_ref() {
                            pass.set_bind_group(0, bg, &[]);
                        }
                        bind_group_keep = bg;
                    }
                    RenderCommand::Draw {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    } => {
                        pass.draw(
                            *first_vertex..first_vertex + vertex_count,
                            *first_instance..first_instance + instance_count,
                        );
                    }
                    RenderCommand::DrawIndexed {
                        index_count,
                        instance_count,
                        first_index,
                        base_vertex,
                        first_instance,
                    } => {
                        pass.draw_indexed(
                            *first_index..first_index + index_count,
                            *base_vertex,
                            *first_instance..first_instance + instance_count,
                        );
                    }
                }
            }
            let _ = bind_group_keep;
            let _ = current_pipeline;
        }
        queue.submit([encoder.finish()]);
        poll_device(&wgpu_device, wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }

    fn create_texture(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<TextureHandle> {
        anyhow::ensure!(
            width > 0 && height > 0,
            "WebGPU: texture width and height must be non-zero"
        );
        let needs_sampled = matches!(access, TextureKind::Interpolated | TextureKind::DirectInterpolated);
        let needs_storage = matches!(access, TextureKind::Direct | TextureKind::DirectInterpolated);
        if needs_storage && matches!(format, TextureFormat::Bgra8UnormSrgb | TextureFormat::Rgba8UnormSrgb) {
            anyhow::bail!("WebGPU: {format:?} cannot be used as a storage texture");
        }
        if needs_storage && format == TextureFormat::Bgra8Unorm {
            anyhow::ensure!(
                self.device(device)?
                    .features
                    .contains(wgpu::Features::BGRA8UNORM_STORAGE),
                "WebGPU: Bgra8Unorm storage requires BGRA8UNORM_STORAGE"
            );
        }

        let wgpu_format = map_texture_format(format);
        let adapter_id = self.device(device)?.adapter_id;
        let adapter = self
            .adapters
            .get(adapter_id as usize)
            .context("WebGPU: texture create missing adapter")?;
        let format_features = adapter.get_texture_format_features(wgpu_format);
        let mut usage = wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        if needs_sampled {
            usage |= wgpu::TextureUsages::TEXTURE_BINDING;
        }
        if needs_storage {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        if flags.contains(TextureFlags::COPY_SRC) {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }
        if flags.contains(TextureFlags::COPY_DST) {
            usage |= wgpu::TextureUsages::COPY_DST;
        }
        if flags.contains(TextureFlags::RENDER_TARGET) {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }
        anyhow::ensure!(
            format_features.allowed_usages.contains(usage),
            "WebGPU: format {format:?} does not support {:?} on this adapter",
            usage
        );

        let gpu = self.device(device)?;
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("goldy-webgpu-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage,
            view_formats: &[],
        });
        self.insert_owned_texture(device, texture, width, height, format, needs_sampled, needs_storage)
    }

    fn write_texture(&mut self, texture: TextureHandle, data: &[u8], width: u32, height: u32) -> Result<()> {
        self.write_texture_pixels(texture, data, 0, 0, width, height)
    }

    fn write_texture_region(
        &mut self,
        texture: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<()> {
        self.write_texture_pixels(texture, data, x, y, width, height)
    }

    fn destroy_texture(&mut self, texture: TextureHandle) {
        if let Some(resource) = self.textures.remove(&texture) {
            if let Some(slot) = resource.storage_slot {
                self.texture_slots.remove(&slot);
                self.recycle_registry_slot(slot);
            }
            if let Some(slot) = resource.sampled_slot {
                self.texture_slots.remove(&slot);
                self.recycle_registry_slot(slot);
            }
        }
    }

    fn texture_bindless_index(&self, texture: TextureHandle) -> Option<u32> {
        self.textures
            .get(&texture)
            .and_then(|texture| texture.storage_slot.or(texture.sampled_slot))
    }

    fn texture_bindless_sampled_index(&self, texture: TextureHandle) -> Option<u32> {
        self.textures.get(&texture).and_then(|texture| texture.sampled_slot)
    }

    fn create_sampler(&mut self, device: DeviceHandle, desc: &SamplerDesc) -> Result<SamplerHandle> {
        let gpu = self.device(device)?;
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("goldy-webgpu-sampler"),
            address_mode_u: map_address_mode(desc.address_mode_u),
            address_mode_v: map_address_mode(desc.address_mode_v),
            address_mode_w: map_address_mode(desc.address_mode_w),
            mag_filter: map_filter_mode(desc.mag_filter),
            min_filter: map_filter_mode(desc.min_filter),
            mipmap_filter: map_mipmap_filter_mode(desc.mipmap_filter),
            lod_min_clamp: desc.lod_min_clamp,
            lod_max_clamp: desc.lod_max_clamp,
            compare: desc.compare.map(map_compare),
            anisotropy_clamp: desc.max_anisotropy.max(1.0).min(16.0) as u16,
            border_color: None,
        });
        let slot = self.alloc_registry_slot()?;
        let handle = self.next_sampler;
        self.next_sampler += 1;
        self.sampler_slots.insert(slot, handle);
        self.samplers.insert(handle, WebGpuSampler { device, sampler, slot });
        Ok(handle)
    }

    fn destroy_sampler(&mut self, sampler: SamplerHandle) {
        if let Some(resource) = self.samplers.remove(&sampler) {
            self.sampler_slots.remove(&resource.slot);
            self.recycle_registry_slot(resource.slot);
        }
    }

    fn sampler_bindless_index(&self, sampler: SamplerHandle) -> Option<u32> {
        self.samplers.get(&sampler).map(|sampler| sampler.slot)
    }

    #[cfg(feature = "graphics")]
    fn create_surface(
        &mut self,
        device: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
        _depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle> {
        let adapter_id = self.device(device)?.adapter_id;
        let adapter = self
            .adapters
            .get(adapter_id as usize)
            .context("WebGPU: surface create missing adapter")?;
        let window_handle = window
            .window_handle()
            .map_err(|error| anyhow::anyhow!("WebGPU: window handle: {error:?}"))?;
        let display_handle = display
            .display_handle()
            .map_err(|error| anyhow::anyhow!("WebGPU: display handle: {error:?}"))?;
        // SAFETY: Goldy surfaces require the window to outlive the `Surface` handle,
        // matching native backends that store HWND / NSView / wl_surface pointers.
        let wgpu_surface = unsafe {
            self.instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: display_handle.as_raw(),
                    raw_window_handle: window_handle.as_raw(),
                })
        }
        .context("WebGPU: create_surface")?;
        anyhow::ensure!(
            adapter.is_surface_supported(&wgpu_surface),
            "WebGPU: adapter does not support this surface"
        );
        let caps = wgpu_surface.get_capabilities(adapter);
        let swapchain_format = pick_swapchain_format(&caps.formats)?;
        let present_mode = pick_present_mode(&caps.present_modes)?;
        let alpha_mode = pick_alpha_mode(&caps.alpha_modes);
        let features = self.device(device)?.features;
        let present_path = choose_present_path(caps.usages, swapchain_format, features, parse_present_override()?)?;
        let usage = surface_usage(&caps, present_path)?;
        let compute_format = compute_format_for_path(present_path, swapchain_format)?;
        tracing::info!(
            ?present_path,
            ?swapchain_format,
            ?compute_format,
            "WebGPU surface present path"
        );
        let handle = self.next_surface;
        self.next_surface += 1;
        self.surfaces.insert(
            handle,
            WebGpuSurface {
                device,
                surface: wgpu_surface,
                width: DEFAULT_SURFACE_WIDTH,
                height: DEFAULT_SURFACE_HEIGHT,
                swapchain_format,
                compute_format,
                present_path,
                present_mode,
                alpha_mode,
                usage,
                scratch: None,
                lease: None,
                acquired: None,
                current_texture_handle: None,
            },
        );
        self.configure_surface(handle)?;
        Ok(handle)
    }

    #[cfg(feature = "graphics")]
    fn destroy_surface(&mut self, surface: SurfaceHandle) {
        if let Some(mut state) = self.surfaces.remove(&surface) {
            state.acquired = None;
            if let Some(lease) = state.lease.take() {
                self.destroy_texture(lease);
            }
            if let Some(scratch) = state.scratch.take() {
                self.destroy_texture(scratch);
            }
        }
    }

    #[cfg(feature = "graphics")]
    fn surface_resize(&mut self, surface: SurfaceHandle, width: u32, height: u32) -> Result<()> {
        anyhow::ensure!(width > 0 && height > 0, "WebGPU: surface size must be non-zero");
        let device = {
            let state = self
                .surfaces
                .get_mut(&surface)
                .context("WebGPU: invalid surface handle")?;
            if state.width == width && state.height == height {
                return Ok(());
            }
            // wgpu panics if a SurfaceTexture is live across configure().
            state.acquired = None;
            state.width = width;
            state.height = height;
            state.device
        };
        // Old scratch may still be the source of an in-flight present copy.
        self.device_wait_idle(device)?;
        self.drop_surface_scratch(surface);
        self.configure_surface(surface)?;
        let path = self
            .surfaces
            .get(&surface)
            .context("WebGPU: invalid surface handle")?
            .present_path;
        if path != WebGpuPresentPath::Direct {
            self.ensure_surface_scratch(surface)?;
        }
        Ok(())
    }

    #[cfg(feature = "graphics")]
    fn surface_size(&self, surface: SurfaceHandle) -> (u32, u32) {
        self.surfaces
            .get(&surface)
            .map(|surface| (surface.width, surface.height))
            .unwrap_or((0, 0))
    }

    #[cfg(feature = "graphics")]
    fn surface_format(&self, surface: SurfaceHandle) -> TextureFormat {
        self.surfaces
            .get(&surface)
            .map(|surface| surface.compute_format)
            .unwrap_or(TextureFormat::Rgba8Unorm)
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        let Some(context) = self.contexts.get(&ctx) else {
            return 0;
        };
        pump_device(&context.wgpu_device);
        context.completed.load(Ordering::Acquire)
    }

    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        let Some(gpu) = self.devices.get(&device) else {
            return 0;
        };
        pump_device(&gpu.device);
        gpu.retired.load(Ordering::Acquire)
    }

    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        let gpu = self.device(device)?;
        if gpu.retired.load(Ordering::Acquire) >= value {
            return Ok(());
        }
        let last = gpu.last_submission.lock().unwrap().clone();
        let wgpu_device = gpu.device.clone();
        let retired = Arc::clone(&gpu.retired);
        let horizon = gpu.next_timeline.load(Ordering::Acquire).saturating_sub(1);
        if value > horizon {
            anyhow::bail!("WebGPU: timeline value {value} has not been submitted");
        }
        match last {
            Some((submitted, index)) if submitted >= value => {
                poll_device(
                    &wgpu_device,
                    wgpu::PollType::Wait {
                        submission_index: Some(index),
                        timeout: Some(TIMELINE_WAIT_TIMEOUT),
                    },
                )?;
            }
            _ => {
                poll_device(&wgpu_device, wgpu::PollType::wait_indefinitely())?;
            }
        }
        retired.fetch_max(value, Ordering::AcqRel);
        if retired.load(Ordering::Acquire) < value {
            anyhow::bail!("WebGPU: timeline value {value} has not been submitted");
        }
        Ok(())
    }

    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        _progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::QueuedSignal> {
        if let Some(context) = self.contexts.get(&ctx) {
            pump_device(&context.wgpu_device);
            crate::signal::drain_all_queued_signals(&context.signal_queue)
        } else {
            Vec::new()
        }
    }

    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        if let Some(sync) = sync {
            for epoch in sync
                .waits
                .iter()
                .chain(sync.cpu_waits.iter())
                .chain(sync.host_observed_waits.iter())
            {
                self.device_wait_until(self.context_device(ctx), epoch.value)?;
            }
            for write in &sync.deferred_host_writes {
                self.write_buffer(write.buffer, write.offset, &write.data)?;
            }
        }
        let effective = commands_with_sync_prologue(commands, sync);
        self.submit_commands(ctx, &effective)
    }

    #[cfg(feature = "graphics")]
    fn begin_frame(&mut self, surface: SurfaceHandle, ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)> {
        let surface_device = self
            .surfaces
            .get(&surface)
            .context("WebGPU: invalid surface handle")?
            .device;
        anyhow::ensure!(
            self.context_device(ctx) == surface_device,
            "WebGPU: begin_frame context does not match the surface device"
        );
        self.drop_surface_lease(surface);
        let acquired = self.acquire_surface_texture(surface)?;
        let path = self
            .surfaces
            .get(&surface)
            .context("WebGPU: invalid surface handle")?
            .present_path;
        let drawable = if path == WebGpuPresentPath::Direct {
            let (width, height, format) = {
                let state = self.surfaces.get(&surface).context("WebGPU: invalid surface handle")?;
                (state.width.max(1), state.height.max(1), state.compute_format)
            };
            let lease = self.register_surface_lease(surface_device, acquired.texture.clone(), width, height, format)?;
            let state = self
                .surfaces
                .get_mut(&surface)
                .context("WebGPU: invalid surface handle")?;
            state.lease = Some(lease);
            state.acquired = Some(acquired);
            state.current_texture_handle = Some(lease);
            lease
        } else {
            let scratch = self.ensure_surface_scratch(surface)?;
            let state = self
                .surfaces
                .get_mut(&surface)
                .context("WebGPU: invalid surface handle")?;
            state.acquired = Some(acquired);
            state.current_texture_handle = Some(scratch);
            scratch
        };
        if let Some(context) = self.contexts.get(&ctx) {
            context
                .signal_queue
                .push(crate::signal::Signal::SwapchainAcquired { image_index: 0 });
        }
        Ok((
            FrameToken {
                surface,
                image: 0,
                context: ctx,
                frame_slot: 0,
                present_slot: 0,
            },
            drawable,
        ))
    }

    #[cfg(feature = "graphics")]
    fn submit_frame(&mut self, frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        let surface_device = self
            .surfaces
            .get(&frame.surface)
            .context("WebGPU: invalid surface handle")?
            .device;
        anyhow::ensure!(
            self.context_device(frame.context) == surface_device,
            "WebGPU: submit_frame context does not match the surface device"
        );
        let context = self.context(frame.context)?;
        Ok(context.submitted_max.load(Ordering::Acquire))
    }

    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        compute_shader: ShaderHandle,
        debug_name: Option<&str>,
    ) -> Result<ComputePipelineHandle> {
        let shader = self
            .shaders
            .get(&compute_shader)
            .context("WebGPU: invalid shader handle")?;
        if shader.device != device {
            anyhow::bail!("WebGPU: shader belongs to another device");
        }
        let (wgsl, slot_access, layout) = self.compile_compute_wgsl(shader)?;
        let pipeline = self.create_wgpu_compute_pipeline(device, &wgsl, debug_name)?;
        let shader = shader.clone();
        let handle = self.next_compute_pipeline;
        self.next_compute_pipeline += 1;
        self.compute_pipelines.insert(
            handle,
            WebGpuComputePipeline {
                device,
                shader,
                pipeline,
                variants: HashMap::new(),
                slot_access,
                layout,
            },
        );
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        self.compute_pipelines.remove(&pipeline);
    }

    fn compute_pipeline_slot_access(&self, pipeline: ComputePipelineHandle) -> Vec<Option<ResourceAccess>> {
        self.compute_pipelines
            .get(&pipeline)
            .map(|pipeline| pipeline.slot_access.clone())
            .unwrap_or_default()
    }

    fn max_bindless_slots_per_category(&self, device: DeviceHandle, _category: crate::types::ResourceCategory) -> u32 {
        if self.devices.contains_key(&device) {
            WEBGPU_REGISTRY_CAP
        } else {
            0
        }
    }

    fn available_bindless_slots(&self, device: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        let used = match category {
            ResourceCategory::Scattered => self
                .buffers
                .values()
                .filter(|buffer| buffer.device == device && buffer.slot.is_some() && !buffer.uniform)
                .count() as u32,
            ResourceCategory::Broadcast => self
                .buffers
                .values()
                .filter(|buffer| buffer.device == device && buffer.slot.is_some() && buffer.uniform)
                .count() as u32,
            ResourceCategory::Texture => self
                .textures
                .values()
                .filter(|texture| texture.device == device && texture.sampled_slot.is_some())
                .count() as u32,
            ResourceCategory::StorageImage => self
                .textures
                .values()
                .filter(|texture| texture.device == device && texture.storage_slot.is_some())
                .count() as u32,
            ResourceCategory::Sampler => self
                .samplers
                .values()
                .filter(|sampler| sampler.device == device)
                .count() as u32,
        };
        self.max_bindless_slots_per_category(device, category)
            .saturating_sub(used)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_wgsl_rewrites_only_packed_rgba32float_bindings() {
        let wgsl = r#"
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dest : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var extra : texture_storage_2d<rgba32float, read_write>;
"#;
        let patched = patch_wgsl_storage_formats(wgsl, &[(1, "rgba8unorm")]).unwrap();
        assert!(patched.contains("@binding(1) var dest : texture_storage_2d<rgba8unorm, write>"));
        assert!(patched.contains("@binding(2) var extra : texture_storage_2d<rgba32float, read_write>"));
        let bgra = patch_wgsl_storage_formats(wgsl, &[(2, "bgra8unorm")]).unwrap();
        assert!(bgra.contains("@binding(2) var extra : texture_storage_2d<bgra8unorm, write>"));
        assert!(bgra.contains("@binding(1) var dest : texture_storage_2d<rgba32float, write>"));
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_path_prefers_copy_then_blit() {
        let storage_copy = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let features = wgpu::Features::BGRA8UNORM_STORAGE;
        // wgpu swapchain images cannot be storage UAVs; Direct is never auto-selected.
        assert_eq!(
            choose_present_path(storage_copy, wgpu::TextureFormat::Bgra8Unorm, features, None).unwrap(),
            WebGpuPresentPath::Copy
        );
        assert_eq!(
            choose_present_path(
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
                wgpu::TextureFormat::Bgra8Unorm,
                features,
                None
            )
            .unwrap(),
            WebGpuPresentPath::Copy
        );
        assert_eq!(
            choose_present_path(
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                wgpu::TextureFormat::Bgra8UnormSrgb,
                features,
                None
            )
            .unwrap(),
            WebGpuPresentPath::Blit
        );
        assert_eq!(
            choose_present_path(
                storage_copy,
                wgpu::TextureFormat::Bgra8Unorm,
                features,
                Some(WebGpuPresentPath::Blit)
            )
            .unwrap(),
            WebGpuPresentPath::Blit
        );
        let err = choose_present_path(
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureFormat::Bgra8Unorm,
            features,
            Some(WebGpuPresentPath::Direct),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("unavailable"));
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn srgb_swapchain_is_not_storage_capable() {
        assert!(!storage_capable_scratch(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::Features::BGRA8UNORM_STORAGE
        ));
        assert!(storage_capable_scratch(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::Features::BGRA8UNORM_STORAGE
        ));
        assert!(!storage_capable_scratch(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::Features::empty()
        ));
    }

    const DOUBLE_WGSL: &str = r#"// @goldy-wgsl
@group(0) @binding(0)
var<storage, read_write> values: array<u32>;

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    values[id.x] = values[id.x] * 2u;
}
"#;

    const DOUBLE_SLANG: &str = r#"
RWStructuredBuffer<uint> values : register(u0);

[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    values[id.x] = values[id.x] * 2;
}
"#;

    const DOUBLE_GOLDY_SLANG: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> values, ThreadId id) {
    values[id.x] = values[id.x] * 2;
}
"#;

    const DOUBLE_GOLDY_TWO_BUFFER_SLANG: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(BufRO<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x] * 2;
}
"#;

    fn run_compute_dispatch_and_readback(shader_source: &str) -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU compute test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))?;
        let shader = backend.create_shader_with_paths(
            device,
            shader_source,
            &[],
            &[],
            crate::types::OptimizationLevel::Default,
        )?;
        let pipeline = backend.create_compute_pipeline(device, shader, Some("double"))?;
        let slot = backend.buffer_bindless_index(buffer).context("missing registry key")?;
        let submitted = backend.submit_standalone(
            ctx,
            &[
                GpuCommand::SetPipeline(pipeline),
                GpuCommand::BindResourcesRaw {
                    indices: vec![slot],
                    user: vec![],
                    frame_table_base: 0,
                },
                GpuCommand::Dispatch {
                    label: Some("double"),
                    workgroups_x: 4,
                    workgroups_y: 1,
                    workgroups_z: 1,
                },
            ],
            None,
        )?;
        assert!(
            backend.gpu_progress(ctx) <= submitted,
            "progress must not exceed the submitted timeline value"
        );
        backend.wait_until(ctx, submitted)?;
        assert_eq!(backend.gpu_progress(ctx), submitted);

        let readback = backend.alloc_readback_buffer(device, 16)?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: buffer,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 16,
            }],
            None,
        )?;
        let mut bytes = [0u8; 16];
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn wgsl_compute_dispatch_and_readback() -> Result<()> {
        run_compute_dispatch_and_readback(DOUBLE_WGSL)
    }

    #[test]
    fn slang_compute_dispatch_and_readback() -> Result<()> {
        run_compute_dispatch_and_readback(DOUBLE_SLANG)
    }

    #[test]
    fn overlapping_submits_complete_on_wait() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU overlapping submit test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))?;
        let shader = backend.create_shader_with_paths(
            device,
            DOUBLE_WGSL,
            &[],
            &[],
            crate::types::OptimizationLevel::Default,
        )?;
        let pipeline = backend.create_compute_pipeline(device, shader, Some("double"))?;
        let slot = backend.buffer_bindless_index(buffer).context("missing registry key")?;
        let commands = [
            GpuCommand::SetPipeline(pipeline),
            GpuCommand::BindResourcesRaw {
                indices: vec![slot],
                user: vec![],
                frame_table_base: 0,
            },
            GpuCommand::Dispatch {
                label: Some("double"),
                workgroups_x: 4,
                workgroups_y: 1,
                workgroups_z: 1,
            },
        ];
        let first = backend.submit_standalone(ctx, &commands, None)?;
        let second = backend.submit_standalone(ctx, &commands, None)?;
        assert!(first < second, "each submit must allocate a new timeline value");
        assert!(backend.gpu_progress(ctx) <= second);
        backend.wait_until(ctx, second)?;
        assert_eq!(backend.gpu_progress(ctx), second);
        assert!(backend.device_timeline_retired(device) >= second);
        Ok(())
    }

    fn run_scheme_compute_and_withdraw(shader_source: &str) -> Result<()> {
        let backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU scheme test: {error:#}");
                return Ok(());
            }
        };
        let device = Arc::new(crate::Device::from_backend(Box::new(backend))?);
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let buffer = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let shader = crate::ShaderModule::from_slang(&device, shader_source)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double", &pipeline)
            .with_parcel(&buffer, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &buffer)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn scheme_dispatches_slang_compute_and_withdraws() -> Result<()> {
        run_scheme_compute_and_withdraw(DOUBLE_SLANG)
    }

    #[test]
    fn scheme_dispatches_goldy_virtual_compute_and_withdraws() -> Result<()> {
        run_scheme_compute_and_withdraw(DOUBLE_GOLDY_SLANG)
    }

    #[test]
    fn scheme_binds_two_goldy_buffers_in_parameter_order() -> Result<()> {
        let backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU scheme test: {error:#}");
                return Ok(());
            }
        };
        let device = Arc::new(crate::Device::from_backend(Box::new(backend))?);
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let input = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let output = pool.acquire_buffer_sized::<u32>(4, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, DOUBLE_GOLDY_TWO_BUFFER_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double", &pipeline)
            .with_parcel(&input, crate::NodeAccess::Read)
            .with_parcel(&output, crate::NodeAccess::Write)
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &output)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    fn scheme_device() -> Result<Option<std::sync::Arc<crate::Device>>> {
        match WebGpuBackend::new() {
            Ok(backend) => Ok(Some(std::sync::Arc::new(crate::Device::from_backend(Box::new(
                backend,
            ))?))),
            Err(error) => {
                eprintln!("skipping WebGPU scheme test: {error:#}");
                Ok(None)
            }
        }
    }

    #[test]
    fn scheme_scalar_uint_param_roundtrip() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint value, ThreadId id) {
    out[0] = value;
}
"#;
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let out = pool.acquire_buffer(4, BufferKind::Scattered, None, BufferFlags::empty(), None)?;
        let shader = crate::ShaderModule::from_slang(&device, SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_uint", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(42u32)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[42]);
        Ok(())
    }

    #[test]
    fn scheme_scalar_float_and_second_buffer() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(BufRO<float> input, Scattered<float> output, float scale, ThreadId id) {
    output[id.x] = input[id.x] * scale;
}
"#;
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let input = pool.acquire_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0], BufferKind::Scattered)?;
        let output = pool.acquire_buffer_sized::<f32>(4, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("scale", &pipeline)
            .with_parcel(&input, crate::NodeAccess::Read)
            .with_parcel(&output, crate::NodeAccess::Write)
            .with_param(2.0f32.to_bits())
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &output)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        assert_eq!(got, &[2.0, 4.0, 6.0, 8.0]);
        Ok(())
    }

    #[test]
    fn scheme_dispatch_batch_distinct_scalars() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint value, ThreadId id) {
    out[0] = value;
}
"#;
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let a = pool.acquire_buffer(4, BufferKind::Scattered, None, BufferFlags::empty(), None)?;
        let b = pool.acquire_buffer(4, BufferKind::Scattered, None, BufferFlags::empty(), None)?;
        let shader = crate::ShaderModule::from_slang(&device, SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("fill_a", &pipeline)
            .with_parcel(&a, crate::NodeAccess::Write)
            .with_param(7u32)
            .dispatch(1, 1, 1);
        scheme
            .node("fill_b", &pipeline)
            .with_parcel(&b, crate::NodeAccess::Write)
            .with_param(9u32)
            .dispatch(1, 1, 1);
        let withdraw_a = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &a)?;
        let withdraw_b = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &b)?;
        let mut submission = scheme.submit()?;
        let bytes_a = withdraw_a.claim(&mut submission)?.consume()?;
        let bytes_b = withdraw_b.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_a), &[7]);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_b), &[9]);
        Ok(())
    }

    #[test]
    fn buffer_view_rejects_unaligned_offset() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU scheme test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let (handle, _) = backend.create_buffer_with_capacity(
            device,
            1024,
            1024,
            BufferKind::Scattered,
            None,
            BufferFlags::empty(),
        )?;
        let err = backend
            .create_buffer_view(handle, 1, 16, None)
            .expect_err("unaligned storage view offset must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("aligned") || msg.contains("align"), "{msg}");
        Ok(())
    }

    #[test]
    fn scheme_broadcast_constant_buffer() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        const SHADER: &str = r#"
import goldy_exp;
struct Params { uint mul; };
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Params cfg, Scattered<uint> values, ThreadId id) {
    values[id.x] = values[id.x] * cfg.mul;
}
"#;
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let cfg = pool.acquire_buffer_with_data(&[3u32], BufferKind::Broadcast)?;
        let values = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let shader = crate::ShaderModule::from_slang(&device, SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("mul", &pipeline)
            .with_parcel(&cfg, crate::NodeAccess::Read)
            .with_parcel(&values, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &values)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[3, 6, 9, 12]);
        Ok(())
    }

    #[test]
    fn scheme_two_independent_dispatches_same_pipeline() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let a = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let b = pool.acquire_buffer_with_data(&[10u32, 20, 30, 40], BufferKind::Scattered)?;
        let shader = crate::ShaderModule::from_slang(&device, DOUBLE_GOLDY_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double_a", &pipeline)
            .with_parcel(&a, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        scheme
            .node("double_b", &pipeline)
            .with_parcel(&b, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        let withdraw_a = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &a)?;
        let withdraw_b = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &b)?;
        let mut submission = scheme.submit()?;
        let bytes_a = withdraw_a.claim(&mut submission)?.consume()?;
        let bytes_b = withdraw_b.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_a), &[2, 4, 6, 8]);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes_b), &[20, 40, 60, 80]);
        Ok(())
    }

    #[test]
    fn scheme_write_direct_spatial_rgba32float_and_withdraw() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(DirectSpatial<float4> output, ThreadId id) {
    uint2 dims;
    output.GetDimensions(dims.x, dims.y);
    if (id.x < dims.x && id.y < dims.y) {
        output[int2(id.x, id.y)] = float4(1.0, 0.0, 0.0, 1.0);
    }
}
"#;
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let texture = pool.acquire_texture(
            16,
            16,
            TextureFormat::Rgba32Float,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
            None,
        )?;
        let shader = crate::ShaderModule::from_slang(&device, SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("write_tex", &pipeline)
            .with_parcel(&texture, crate::NodeAccess::Write)
            .dispatch(2, 2, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &texture)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        let floats: &[f32] = bytemuck::cast_slice(&bytes);
        assert_eq!(&floats[0..4], &[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn scheme_write_direct_spatial_float4_to_rgba8unorm() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(DirectSpatial<float4> output, ThreadId id) {
    uint2 dims;
    output.GetDimensions(dims.x, dims.y);
    if (id.x < dims.x && id.y < dims.y) {
        output[int2(id.x, id.y)] = float4(1.0, 0.0, 0.0, 1.0);
    }
}
"#;
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let texture = pool.acquire_texture(
            16,
            16,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
            None,
        )?;
        let shader = crate::ShaderModule::from_slang(&device, SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("write_tex", &pipeline)
            .with_parcel(&texture, crate::NodeAccess::Write)
            .dispatch(2, 2, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &texture)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(&bytes[0..4], &[255, 0, 0, 255]);
        Ok(())
    }

    #[test]
    fn scheme_write_direct_spatial_float4_to_bgra8unorm() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(DirectSpatial<float4> output, ThreadId id) {
    uint2 dims;
    output.GetDimensions(dims.x, dims.y);
    if (id.x < dims.x && id.y < dims.y) {
        output[int2(id.x, id.y)] = float4(1.0, 0.0, 0.0, 1.0);
    }
}
"#;
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let texture = match pool.acquire_texture(
            16,
            16,
            TextureFormat::Bgra8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
            None,
        ) {
            Ok(texture) => texture,
            Err(error) if format!("{error:#}").contains("BGRA8UNORM_STORAGE") => {
                eprintln!("skipping Bgra8Unorm storage test: {error:#}");
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let shader = crate::ShaderModule::from_slang(&device, SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("write_tex", &pipeline)
            .with_parcel(&texture, crate::NodeAccess::Write)
            .dispatch(2, 2, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &texture)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(&bytes[0..4], &[0, 0, 255, 255]);
        Ok(())
    }

    #[test]
    fn scheme_sample_interpolated_and_filter() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Interpolated<float4> src, Filter smp, Scattered<uint> out, ThreadId id) {
    float4 v = src.SampleLevel(smp, float2(0.5, 0.5), 0);
    out[0] = uint(v.x * 255.0 + 0.5);
    out[1] = uint(v.y * 255.0 + 0.5);
    out[2] = uint(v.z * 255.0 + 0.5);
    out[3] = uint(v.w * 255.0 + 0.5);
}
"#;
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let pixels = vec![64u8, 128, 192, 255].repeat(4 * 4);
        let texture = pool.acquire_texture(
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
            Some(&pixels),
        )?;
        let out = pool.acquire_buffer(16, BufferKind::Scattered, None, BufferFlags::empty(), None)?;
        let shader = crate::ShaderModule::from_slang(&device, SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;
        let sampler = crate::Sampler::nearest(&device)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("sample", &pipeline)
            .with_parcel(&texture, crate::NodeAccess::Read)
            .with_parcel(&sampler, crate::NodeAccess::Read)
            .with_parcel(&out, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[64, 128, 192, 255]);
        Ok(())
    }

    #[test]
    fn scheme_direct_interpolated_write_then_sample() -> Result<()> {
        let Some(device) = scheme_device()? else {
            return Ok(());
        };
        const WRITE: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(4, 4, 1)]
void cs_main(DirectSpatial<float4> dst, ThreadId id) {
    dst[uint2(id.x, id.y)] = float4(float(id.x) / 255.0, float(id.y) / 255.0, 0.0, 1.0);
}
"#;
        const READ: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(4, 4, 1)]
void cs_main(Interpolated<float4> src, Filter smp, Scattered<uint> out, ThreadId id) {
    float2 uv = (float2(id.x, id.y) + 0.5) / float2(4.0, 4.0);
    float4 v = src.SampleLevel(smp, uv, 0);
    uint r = uint(v.x * 255.0 + 0.5);
    uint g = uint(v.y * 255.0 + 0.5);
    out[id.y * 4 + id.x] = r | (g << 8) | (uint(v.w * 255.0 + 0.5) << 24);
}
"#;
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(std::sync::Arc::clone(&device));
        let tex = pool.acquire_texture(
            4,
            4,
            TextureFormat::Rgba32Float,
            TextureKind::DirectInterpolated,
            TextureFlags::empty(),
            None,
        )?;
        let out = pool.acquire_buffer(4 * 4 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)?;
        let write = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, WRITE)?)?;
        let read = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, READ)?)?;
        let sampler = crate::Sampler::nearest(&device)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("write", &write)
            .with_parcel(&tex, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("read", &read)
            .with_parcel(&tex, crate::NodeAccess::Read)
            .with_parcel(&sampler, crate::NodeAccess::Read)
            .with_parcel(&out, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        let result: &[u32] = bytemuck::cast_slice(&bytes);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let packed = result[(y * 4 + x) as usize];
                assert_eq!(packed & 0xFF, x, "r at ({x},{y})");
                assert_eq!((packed >> 8) & 0xFF, y, "g at ({x},{y})");
                assert_eq!((packed >> 24) & 0xFF, 255, "a at ({x},{y})");
            }
        }
        Ok(())
    }

    #[test]
    fn copy_texture_then_readback() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU texture copy test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let src_pixels: Vec<u8> = (0..4 * 4 * 4).map(|i| (i % 251) as u8).collect();
        let src = backend.create_texture(
            device,
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        )?;
        backend.write_texture(src, &src_pixels, 4, 4)?;
        let dst = backend.create_texture(
            device,
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        )?;
        backend.submit_standalone(ctx, &[GpuCommand::CopyTexture { src, dst }], None)?;
        let layout = backend.query_texture_copy_footprint(device, 4, 4, TextureFormat::Rgba8Unorm)?;
        let staging = backend.alloc_texture_readback_staging(device, layout)?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyTextureToReadback {
                src: dst,
                dst: staging,
                layout,
            }],
            None,
        )?;
        let mut bytes = vec![0u8; layout.logical_bytes as usize];
        backend.read_texture_readback_staging(staging, layout, &mut bytes)?;
        assert_eq!(bytes, src_pixels);
        Ok(())
    }

    #[cfg(feature = "graphics")]
    fn readback_texture(
        backend: &mut WebGpuBackend,
        ctx: ContextHandle,
        device: DeviceHandle,
        src: TextureHandle,
        format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        let layout = backend.query_texture_copy_footprint(device, width, height, format)?;
        let staging = backend.alloc_texture_readback_staging(device, layout)?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyTextureToReadback {
                src,
                dst: staging,
                layout,
            }],
            None,
        )?;
        let mut bytes = vec![0u8; layout.logical_bytes as usize];
        backend.read_texture_readback_staging(staging, layout, &mut bytes)?;
        Ok(bytes)
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_copy_same_format_bgra8() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU BGRA copy test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let src_pixels = vec![10u8, 20, 30, 255].repeat(4 * 4);
        let src = match backend.create_texture(
            device,
            4,
            4,
            TextureFormat::Bgra8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        ) {
            Ok(src) => src,
            Err(error) => {
                eprintln!("skipping Bgra8Unorm copy test: {error:#}");
                return Ok(());
            }
        };
        backend.write_texture(src, &src_pixels, 4, 4)?;
        let dst = backend.create_texture(
            device,
            4,
            4,
            TextureFormat::Bgra8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        )?;
        backend.submit_standalone(ctx, &[GpuCommand::CopyTexture { src, dst }], None)?;
        let bytes = readback_texture(&mut backend, ctx, device, dst, TextureFormat::Bgra8Unorm, 4, 4)?;
        assert_eq!(bytes, src_pixels);
        Ok(())
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_blit_rgba8_to_bgra8() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU blit test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let src_pixels = vec![255u8, 0, 0, 255].repeat(8 * 8);
        let src = backend.create_texture(
            device,
            8,
            8,
            TextureFormat::Rgba8Unorm,
            TextureKind::DirectInterpolated,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        )?;
        backend.write_texture(src, &src_pixels, 8, 8)?;
        let dst = backend.create_texture(
            device,
            8,
            8,
            TextureFormat::Bgra8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_SRC | TextureFlags::RENDER_TARGET,
        )?;
        backend.blit_texture_to_target(device, src, dst)?;
        let bytes = readback_texture(&mut backend, ctx, device, dst, TextureFormat::Bgra8Unorm, 8, 8)?;
        assert_eq!(&bytes[0..4], &[0, 0, 255, 255], "red in BGRA memory order");
        Ok(())
    }

    #[test]
    fn slang_emits_wgsl_for_compute() -> Result<()> {
        let compiler = crate::slang::SlangCompiler::new()?;
        let compiled = compiler.compile_bindless_with_reflection(
            DOUBLE_SLANG,
            crate::slang::ShaderTarget::Wgsl,
            &[("cs_main", crate::slang::SlangStage::Compute)],
            &[],
        )?;
        let wgsl = compiled.shader.as_str().context("expected text WGSL")?;
        assert!(
            wgsl.contains("@compute"),
            "Slang output did not contain a WGSL compute entry:\n{wgsl}"
        );
        assert!(
            wgsl.contains("@binding(0)"),
            "Slang output did not preserve the storage binding:\n{wgsl}"
        );
        Ok(())
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn slang_emits_wgsl_for_vertex_fragment() -> Result<()> {
        let compiler = crate::slang::SlangCompiler::new()?;
        let src = r#"
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};
struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}
[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
"#;
        let vs = compiler.compile_bindless_with_reflection_and_defines(
            src,
            crate::slang::ShaderTarget::Wgsl,
            &[("vs_main", crate::slang::SlangStage::Vertex)],
            &[],
            &[],
            &[],
            crate::types::OptimizationLevel::Default,
        )?;
        let fs = compiler.compile_bindless_with_reflection_and_defines(
            src,
            crate::slang::ShaderTarget::Wgsl,
            &[("fs_main", crate::slang::SlangStage::Fragment)],
            &[],
            &[],
            &[],
            crate::types::OptimizationLevel::Default,
        )?;
        let vs_wgsl = vs.shader.as_str().context("expected text WGSL")?;
        let fs_wgsl = fs.shader.as_str().context("expected text WGSL")?;
        assert!(vs_wgsl.contains("@vertex"), "missing vertex entry:\n{vs_wgsl}");
        assert!(fs_wgsl.contains("@fragment"), "missing fragment entry:\n{fs_wgsl}");
        Ok(())
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn raster_triangle_to_render_target() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU raster test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let shader = backend.create_shader_with_paths(
            device,
            crate::shaders::VERTEX_COLOR_2D,
            &[],
            &[],
            crate::types::OptimizationLevel::Default,
        )?;
        let pipeline = backend.create_pipeline(
            device,
            shader,
            shader,
            &crate::types::Vertex2D::layout(),
            PrimitiveTopology::TriangleList,
            TextureFormat::Rgba8Unorm,
        )?;
        let verts = [
            crate::types::Vertex2D::new(0.0, -0.5, crate::types::Color::RED),
            crate::types::Vertex2D::new(-0.5, 0.5, crate::types::Color::GREEN),
            crate::types::Vertex2D::new(0.5, 0.5, crate::types::Color::BLUE),
        ];
        let vbuf = backend.create_buffer(
            device,
            std::mem::size_of_val(&verts) as u64,
            BufferKind::Scattered,
            None,
            crate::types::BufferFlags::empty(),
        )?;
        backend.write_buffer(vbuf, 0, bytemuck::bytes_of(&verts))?;
        let rt = backend.create_render_target_with_depth(device, 8, 8, TextureFormat::Rgba8Unorm, None)?;
        backend.render_to_target(
            device,
            rt,
            crate::types::TargetLoad::Clear(crate::types::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            }),
            &[
                RenderCommand::SetPipeline(pipeline),
                RenderCommand::SetVertexBuffer {
                    slot: 0,
                    buffer: vbuf,
                    offset: 0,
                },
                RenderCommand::Draw {
                    vertex_count: 3,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
            ],
        )?;
        let dst = backend.create_texture(
            device,
            8,
            8,
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        )?;
        backend.submit_standalone(ctx, &[GpuCommand::CopyRenderTarget { src: rt, dst }], None)?;
        let bytes = readback_texture(&mut backend, ctx, device, dst, TextureFormat::Rgba8Unorm, 8, 8)?;
        assert_eq!(bytes.len(), 8 * 8 * 4);
        assert!(
            bytes.iter().any(|&b| b != 0),
            "expected rasterized pixels, got all zeros"
        );
        Ok(())
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn swapchain_format_prefers_rgba8_unorm() {
        let formats = [
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ];
        assert_eq!(
            pick_swapchain_format(&formats).unwrap(),
            wgpu::TextureFormat::Rgba8Unorm
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_mode_prefers_fifo() {
        let modes = [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Fifo];
        assert_eq!(pick_present_mode(&modes).unwrap(), wgpu::PresentMode::Fifo);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn begin_frame_rejects_unknown_surface() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU surface test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let err = backend.begin_frame(1, ctx).expect_err("unknown surface must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("invalid surface") || msg.contains("surface"), "{msg}");
        Ok(())
    }

    #[cfg(feature = "graphics")]
    fn dummy_frame(ctx: ContextHandle) -> FrameToken {
        FrameToken {
            surface: 1,
            image: 0,
            context: ctx,
            frame_slot: 0,
            present_slot: 0,
        }
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn submit_frame_rejects_unknown_surface() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU surface test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let err = backend
            .submit_frame(&dummy_frame(ctx))
            .expect_err("unknown surface must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("invalid surface") || msg.contains("surface"), "{msg}");
        Ok(())
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn take_present_rejects_unknown_surface() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU surface test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let err = match backend.take_present_gpu_work(dummy_frame(ctx), 0) {
            Ok(_) => panic!("unknown surface must fail"),
            Err(error) => error,
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("invalid surface") || msg.contains("surface"), "{msg}");
        Ok(())
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn finish_present_rejects_unknown_surface() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU surface test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let finish = PresentFinishState {
            frame: dummy_frame(ctx),
            return_fence: 0,
            scratch_texture: None,
            scratch_layout_updated: false,
            present_timeline: 0,
            copy_timeline: None,
            frame_compute_timeline: None,
            signal_timeline: None,
            render_pass_submitted: false,
            present_ok: false,
        };
        let err = backend
            .finish_present(finish, 0)
            .expect_err("unknown surface must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("invalid surface") || msg.contains("surface"), "{msg}");
        Ok(())
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn surface_resize_rejects_unknown_surface() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU surface test: {error:#}");
                return Ok(());
            }
        };
        let err = backend
            .surface_resize(1, 640, 480)
            .expect_err("unknown surface must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("invalid surface") || msg.contains("surface"), "{msg}");
        Ok(())
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn surface_resize_rejects_zero_size() -> Result<()> {
        let mut backend = match WebGpuBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping WebGPU surface test: {error:#}");
                return Ok(());
            }
        };
        let err = backend.surface_resize(1, 0, 480).expect_err("zero width must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("non-zero"), "{msg}");
        Ok(())
    }
}
