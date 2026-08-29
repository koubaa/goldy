//! Compute-only WebGPU backend prototype.
//!
//! This backend deliberately does not emulate Goldy's native bindless heap. The
//! raw indices in [`GpuCommand::BindResourcesRaw`] are interpreted as backend
//! registry keys and packed into one fixed bind group in shader-parameter order.
//!
//! Submit is non-blocking: the timeline advances from `Queue::on_submitted_work_done`
//! (pumped by `Device::poll`). Host waits use a stored [`wgpu::SubmissionIndex`].

use super::shared::{PushLayout, DISPATCH_BATCH_STRIDE, MAX_USER_SLOTS, TOTAL_PUSH_BYTES};
use super::*;
use crate::frame_table::dispatch_table_base_word_index;
use crate::slang::virtual_main::{WgpuComputeLayout, WgpuComputeResourceKind};
use crate::types::{
    AddressMode, BufferKind, BufferResizeCost, CompareFunction, DeviceType, FilterMode, ResourceCategory,
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

fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        return value;
    }
    value.div_ceil(align) * align
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
    let end = offset
        .checked_add(size)
        .context("WebGPU: buffer range overflow")?;
    anyhow::ensure!(
        end <= buffer.size,
        "WebGPU: {what} range {offset}+{size} exceeds logical size {}",
        buffer.size
    );
    Ok(())
}

pub(crate) struct WebGpuBackend {
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
    next_device: DeviceHandle,
    next_context: ContextHandle,
    next_buffer: BufferHandle,
    next_texture: TextureHandle,
    next_sampler: SamplerHandle,
    next_slot: u32,
    free_slots: Vec<u32>,
    next_shader: ShaderHandle,
    next_compute_pipeline: ComputePipelineHandle,
}

struct WebGpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    next_timeline: Arc<AtomicU64>,
    retired: Arc<AtomicU64>,
    last_submission: Mutex<Option<(crate::timeline::TimelineValue, wgpu::SubmissionIndex)>>,
    user_uniform: Option<wgpu::Buffer>,
    user_uniform_capacity: u64,
    uniform_offset_align: u64,
    storage_offset_align: u64,
    adapter_id: u32,
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

struct WebGpuShader {
    device: DeviceHandle,
    source: String,
    search_paths: Vec<String>,
    defines: Vec<(String, String)>,
    optimization_level: crate::types::OptimizationLevel,
}

#[derive(Clone)]
struct WebGpuComputePipeline {
    device: DeviceHandle,
    pipeline: wgpu::ComputePipeline,
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
            next_device: 1,
            next_context: 1,
            next_buffer: 1,
            next_texture: 1,
            next_sampler: 1,
            next_slot: 0,
            free_slots: Vec::new(),
            next_shader: 1,
            next_compute_pipeline: 1,
        })
    }

    fn device(&self, handle: DeviceHandle) -> Result<&WebGpuDevice> {
        self.devices.get(&handle).context("WebGPU: invalid device handle")
    }

    fn context(&self, handle: ContextHandle) -> Result<&Arc<WebGpuContext>> {
        self.contexts.get(&handle).context("WebGPU: invalid context handle")
    }

    fn unsupported<T>(operation: &str) -> Result<T> {
        anyhow::bail!("WebGPU compute-only backend does not support {operation}")
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
            align_up(capacity.max(logical_size).max(min_capacity), wgpu::COPY_BUFFER_ALIGNMENT)
        };
        let gpu = self.device(device)?;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("goldy-webgpu-buffer"),
            size: capacity,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::INDIRECT,
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
            &[],
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
        pipeline: &WebGpuComputePipeline,
        indices: &[u32],
        user_uniform: Option<(&wgpu::Buffer, u64)>,
    ) -> Result<Option<wgpu::BindGroup>> {
        let mut entries = Vec::new();
        match &pipeline.layout.resources {
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
        if pipeline.layout.scalar_count > 0 {
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
        let layout = pipeline.pipeline.get_bind_group_layout(0);
        let device = &self.device(pipeline.device)?.device;
        Ok(Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("goldy-webgpu-dispatch-bindings"),
            layout: &layout,
            entries: &entries,
        })))
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
                    let pipeline = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .cloned()
                        .context("WebGPU: invalid compute pipeline")?;
                    let user_binding = if pipeline.layout.scalar_count > 0 {
                        require_user_scalars(&current_user, pipeline.layout.scalar_count)?;
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
                    let bind_group = self.create_bind_group(&pipeline, &current_indices, user_binding)?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: *label,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline.pipeline);
                    if let Some(bind_group) = bind_group.as_ref() {
                        pass.set_bind_group(0, bind_group, &[]);
                    }
                    pass.dispatch_workgroups(*workgroups_x, *workgroups_y, *workgroups_z);
                }
                GpuCommand::DispatchIndirect { buffer, offset, label } => {
                    let pipeline_handle = current_pipeline.context("WebGPU: indirect dispatch without a pipeline")?;
                    let pipeline = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .cloned()
                        .context("WebGPU: invalid compute pipeline")?;
                    let args = self.buffers.get(buffer).context("WebGPU: invalid indirect buffer")?;
                    ensure_buffer_range(args, *offset, INDIRECT_DISPATCH_BYTES, "DispatchIndirect")?;
                    anyhow::ensure!(
                        *offset % 4 == 0,
                        "WebGPU: DispatchIndirect offset {offset} must be 4-byte aligned"
                    );
                    let args_buffer = args.buffer.clone();
                    let args_offset = args.offset + offset;
                    let user_binding = if pipeline.layout.scalar_count > 0 {
                        require_user_scalars(&current_user, pipeline.layout.scalar_count)?;
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
                    let bind_group = self.create_bind_group(&pipeline, &current_indices, user_binding)?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: *label,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline.pipeline);
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
                }
                GpuCommand::ResourceBarrier { .. } => {
                    // WebGPU tracks resource transitions within a submitted command buffer.
                }
                GpuCommand::DispatchBatch { arg_data, count, label } => {
                    let pipeline_handle =
                        current_pipeline.context("WebGPU: DispatchBatch without a compute pipeline")?;
                    let pipeline = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .cloned()
                        .context("WebGPU: invalid compute pipeline")?;
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
                    let n_scalars = pipeline.layout.scalar_count as usize;
                    for i in 0..entry_count {
                        let base = i * DISPATCH_BATCH_STRIDE;
                        let push: PushLayout = *bytemuck::from_bytes(&arg_data[base..base + TOTAL_PUSH_BYTES]);
                        let wg_off = base + TOTAL_PUSH_BYTES;
                        let workgroups_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into().unwrap());
                        let workgroups_y = u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into().unwrap());
                        let workgroups_z = u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into().unwrap());
                        let indices = self.batch_indices(&pipeline.layout, frame_table, arg_data, *count, i)?;
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
                        let bind_group = self.create_bind_group(&pipeline, &indices, user_binding)?;
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: *label,
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipeline.pipeline);
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
                GpuCommand::CopyRenderTarget { .. } => {
                    anyhow::bail!("WebGPU compute-only backend: CopyRenderTarget is not supported")
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
impl GpuBackendPresentSplit for WebGpuBackend {
    fn take_present_gpu_work(
        &mut self,
        _frame: FrameToken,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn PresentGpuWork>> {
        Self::unsupported("presentation")
    }

    fn finish_present(
        &mut self,
        _finish: PresentFinishState,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        Self::unsupported("presentation")
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
            preferred_surface_format: TextureFormat::Rgba8Unorm,
            preferred_render_target_format: TextureFormat::Rgba8Unorm,
            supported_surface_formats: Vec::new(),
            supported_render_target_formats: Vec::new(),
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
        let wanted = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | wgpu::Features::FLOAT32_FILTERABLE;
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
        let uniform_offset_align = device.limits().min_uniform_buffer_offset_alignment.max(16) as u64;
        let storage_offset_align = device.limits().min_storage_buffer_offset_alignment.max(4) as u64;
        let handle = self.next_device;
        self.next_device += 1;
        self.devices.insert(
            handle,
            WebGpuDevice {
                device,
                queue,
                next_timeline: Arc::new(AtomicU64::new(1)),
                retired: Arc::new(AtomicU64::new(0)),
                last_submission: Mutex::new(None),
                user_uniform: None,
                user_uniform_capacity: 0,
                uniform_offset_align,
                storage_offset_align,
                adapter_id,
            },
        );
        Ok(handle)
    }

    fn destroy_device(&mut self, device: DeviceHandle) {
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
                | wgpu::BufferUsages::INDIRECT,
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
        _device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        _target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        Self::unsupported("graphics pipelines")
    }

    #[cfg(feature = "graphics")]
    fn destroy_pipeline(&mut self, _pipeline: PipelineHandle) {}

    #[cfg(feature = "graphics")]
    fn create_pipeline_with_depth(
        &mut self,
        _device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        _target_format: TextureFormat,
        _depth_stencil: Option<&DepthStencilState>,
    ) -> Result<PipelineHandle> {
        Self::unsupported("graphics pipelines")
    }

    #[cfg(feature = "graphics")]
    fn create_render_target_with_depth(
        &mut self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _color_format: TextureFormat,
        _depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        Self::unsupported("render targets")
    }

    #[cfg(feature = "graphics")]
    fn destroy_render_target(&mut self, _target: RenderTargetHandle) {}

    #[cfg(feature = "graphics")]
    fn render_to_target(
        &mut self,
        _device: DeviceHandle,
        _target: RenderTargetHandle,
        _color_load: crate::types::TargetLoad,
        _commands: &[RenderCommand],
    ) -> Result<()> {
        Self::unsupported("rendering")
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
        anyhow::ensure!(
            !flags.contains(TextureFlags::RENDER_TARGET),
            "WebGPU compute-only backend does not support render-target textures"
        );
        let needs_sampled = matches!(access, TextureKind::Interpolated | TextureKind::DirectInterpolated);
        let needs_storage = matches!(access, TextureKind::Direct | TextureKind::DirectInterpolated);
        if needs_storage
            && matches!(
                format,
                TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb | TextureFormat::Rgba8UnormSrgb
            )
        {
            anyhow::bail!("WebGPU: {format:?} cannot be used as a storage texture");
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
        _device: DeviceHandle,
        _window: &dyn raw_window_handle::HasWindowHandle,
        _display: &dyn raw_window_handle::HasDisplayHandle,
        _depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle> {
        Self::unsupported("surfaces")
    }

    #[cfg(feature = "graphics")]
    fn destroy_surface(&mut self, _surface: SurfaceHandle) {}

    #[cfg(feature = "graphics")]
    fn surface_resize(&mut self, _surface: SurfaceHandle, _width: u32, _height: u32) -> Result<()> {
        Self::unsupported("surfaces")
    }

    #[cfg(feature = "graphics")]
    fn surface_size(&self, _surface: SurfaceHandle) -> (u32, u32) {
        (0, 0)
    }

    #[cfg(feature = "graphics")]
    fn surface_format(&self, _surface: SurfaceHandle) -> TextureFormat {
        TextureFormat::Bgra8UnormSrgb
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
    fn begin_frame(&mut self, _surface: SurfaceHandle, _ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)> {
        Self::unsupported("frames")
    }

    #[cfg(feature = "graphics")]
    fn submit_frame(&mut self, _frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        Self::unsupported("frames")
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
        let handle = self.next_compute_pipeline;
        self.next_compute_pipeline += 1;
        self.compute_pipelines.insert(
            handle,
            WebGpuComputePipeline {
                device,
                pipeline,
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
}
