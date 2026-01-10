//! WebGPU backend implementation.
//!
//! Uses wgpu crate for WebGPU/WASM support.

use super::*;
use crate::types::Color;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu;

/// WebGPU backend.
pub struct WebGpuBackend {
    instance: wgpu::Instance,
    adapters: Vec<WebGpuAdapterInfo>,
    devices: HashMap<DeviceHandle, WebGpuDevice>,
    next_device_handle: DeviceHandle,
    buffers: HashMap<BufferHandle, WebGpuBuffer>,
    next_buffer_handle: BufferHandle,
    shaders: HashMap<ShaderHandle, WebGpuShader>,
    next_shader_handle: ShaderHandle,
    pipelines: HashMap<PipelineHandle, WebGpuPipeline>,
    next_pipeline_handle: PipelineHandle,
    bind_group_layouts: HashMap<BindGroupLayoutHandle, WebGpuBindGroupLayout>,
    next_bind_group_layout_handle: BindGroupLayoutHandle,
    bind_groups: HashMap<BindGroupHandle, WebGpuBindGroup>,
    next_bind_group_handle: BindGroupHandle,
}

struct WebGpuAdapterInfo {
    adapter: wgpu::Adapter,
    info: wgpu::AdapterInfo,
    id: u32,
}

struct WebGpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_id: u32,
    frame_state: Option<WebGpuFrameState>,
}

struct WebGpuFrameState {
    width: u32,
    height: u32,
    format: TextureFormat,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    output_buffer: wgpu::Buffer,
}

struct WebGpuBuffer {
    device_handle: DeviceHandle,
    buffer: wgpu::Buffer,
    size: u64,
}

struct WebGpuShader {
    device_handle: DeviceHandle,
    module: wgpu::ShaderModule,
}

struct WebGpuPipeline {
    device_handle: DeviceHandle,
    pipeline: wgpu::RenderPipeline,
}

struct WebGpuBindGroupLayout {
    device_handle: DeviceHandle,
    layout: wgpu::BindGroupLayout,
}

struct WebGpuBindGroup {
    device_handle: DeviceHandle,
    bind_group: wgpu::BindGroup,
}

impl WebGpuBackend {
    /// Create a new WebGPU backend (async).
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Enumerate adapters
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("No WebGPU adapter found")?;

        let info = adapter.get_info();
        tracing::info!("WebGPU adapter: {} ({:?})", info.name, info.backend);

        let adapters = vec![WebGpuAdapterInfo {
            adapter,
            info,
            id: 0,
        }];

        Ok(Self {
            instance,
            adapters,
            devices: HashMap::new(),
            next_device_handle: 1,
            buffers: HashMap::new(),
            next_buffer_handle: 1,
            shaders: HashMap::new(),
            next_shader_handle: 1,
            pipelines: HashMap::new(),
            next_pipeline_handle: 1,
            bind_group_layouts: HashMap::new(),
            next_bind_group_layout_handle: 1,
            bind_groups: HashMap::new(),
            next_bind_group_handle: 1,
        })
    }

    /// Create device asynchronously.
    pub async fn create_device_async(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        let adapter_info = self
            .adapters
            .iter()
            .find(|a| a.id == adapter_id)
            .context("Invalid adapter ID")?;

        let (device, queue) = adapter_info
            .adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("RAG Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .context("Failed to create WebGPU device")?;

        let handle = self.next_device_handle;
        self.next_device_handle += 1;

        self.devices.insert(
            handle,
            WebGpuDevice {
                device: Arc::new(device),
                queue: Arc::new(queue),
                adapter_id,
                frame_state: None,
            },
        );

        tracing::info!("Created WebGPU device {}", handle);
        Ok(handle)
    }

    fn format_to_wgpu(format: TextureFormat) -> wgpu::TextureFormat {
        match format {
            TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        }
    }

    fn vertex_format_to_wgpu(format: VertexFormat) -> wgpu::VertexFormat {
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

    fn topology_to_wgpu(topology: PrimitiveTopology) -> wgpu::PrimitiveTopology {
        match topology {
            PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
            PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
            PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
            PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
            PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        }
    }
}

// Note: WebGPU requires async, so GpuBackend implementation uses blocking
// This is fine for WASM as it runs in an async context anyway
impl GpuBackend for WebGpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::WebGPU
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        self.adapters
            .iter()
            .map(|a| {
                let device_type = match a.info.device_type {
                    wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
                    wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
                    wgpu::DeviceType::Cpu => DeviceType::Cpu,
                    _ => DeviceType::Other,
                };

                AdapterInfo {
                    id: a.id,
                    name: a.info.name.clone(),
                    vendor: format!("{}", a.info.vendor),
                    backend: BackendType::WebGPU,
                    device_type,
                }
            })
            .collect()
    }

    fn create_device(&mut self, _adapter_id: u32) -> Result<DeviceHandle> {
        // Sync version - not supported for WebGPU
        anyhow::bail!("WebGPU requires async device creation - use create_device_async")
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        if let Some(device) = self.devices.remove(&device_handle) {
            // Clean up resources owned by this device
            let buffer_handles: Vec<_> = self.buffers
                .iter()
                .filter(|(_, b)| b.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in buffer_handles {
                self.buffers.remove(&handle);
            }

            let shader_handles: Vec<_> = self.shaders
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in shader_handles {
                self.shaders.remove(&handle);
            }

            let pipeline_handles: Vec<_> = self.pipelines
                .iter()
                .filter(|(_, p)| p.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in pipeline_handles {
                self.pipelines.remove(&handle);
            }

            drop(device);
            tracing::info!("Destroyed WebGPU device {}", device_handle);
        }
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn create_buffer(&mut self, device_handle: DeviceHandle, size: u64, usage: BufferUsage) -> Result<BufferHandle> {
        let device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let mut wgpu_usage = wgpu::BufferUsages::empty();
        if usage.contains(BufferUsage::VERTEX) {
            wgpu_usage |= wgpu::BufferUsages::VERTEX;
        }
        if usage.contains(BufferUsage::INDEX) {
            wgpu_usage |= wgpu::BufferUsages::INDEX;
        }
        if usage.contains(BufferUsage::UNIFORM) {
            wgpu_usage |= wgpu::BufferUsages::UNIFORM;
        }
        if usage.contains(BufferUsage::STORAGE) {
            wgpu_usage |= wgpu::BufferUsages::STORAGE;
        }
        if usage.contains(BufferUsage::COPY_SRC) {
            wgpu_usage |= wgpu::BufferUsages::COPY_SRC;
        }
        if usage.contains(BufferUsage::COPY_DST) {
            wgpu_usage |= wgpu::BufferUsages::COPY_DST;
        }

        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAG Buffer"),
            size,
            usage: wgpu_usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;

        self.buffers.insert(handle, WebGpuBuffer {
            device_handle,
            buffer,
            size,
        });

        Ok(handle)
    }

    fn destroy_buffer(&mut self, buffer_handle: BufferHandle) {
        self.buffers.remove(&buffer_handle);
    }

    fn write_buffer(&mut self, buffer_handle: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;

        let device = self
            .devices
            .get(&buffer.device_handle)
            .context("Buffer's device is invalid")?;

        device.queue.write_buffer(&buffer.buffer, offset, data);
        Ok(())
    }

    fn buffer_size(&self, buffer_handle: BufferHandle) -> u64 {
        self.buffers.get(&buffer_handle).map(|b| b.size).unwrap_or(0)
    }

    fn create_shader(&mut self, device_handle: DeviceHandle, slang_source: &str) -> Result<ShaderHandle> {
        let device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Compile Slang to WGSL
        let compiler = crate::slang::global_compiler()
            .context("Failed to get Slang compiler")?;
        
        let compiled = compiler.compile(slang_source, crate::slang::ShaderTarget::Wgsl)
            .context("Slang to WGSL compilation failed")?;
        
        let wgsl_source = compiled.as_str()
            .context("Invalid WGSL output from Slang")?;

        let module = device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RAG Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let handle = self.next_shader_handle;
        self.next_shader_handle += 1;

        self.shaders.insert(handle, WebGpuShader {
            device_handle,
            module,
        });

        Ok(handle)
    }

    fn destroy_shader(&mut self, shader_handle: ShaderHandle) {
        self.shaders.remove(&shader_handle);
    }

    fn create_bind_group_layout(&mut self, device_handle: DeviceHandle, entries: &[BindGroupLayoutEntry]) -> Result<BindGroupLayoutHandle> {
        let device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let wgpu_entries: Vec<_> = entries
            .iter()
            .map(|e| {
                let visibility = if e.visibility.0 & ShaderStages::VERTEX.0 != 0 && e.visibility.0 & ShaderStages::FRAGMENT.0 != 0 {
                    wgpu::ShaderStages::VERTEX_FRAGMENT
                } else if e.visibility.0 & ShaderStages::VERTEX.0 != 0 {
                    wgpu::ShaderStages::VERTEX
                } else {
                    wgpu::ShaderStages::FRAGMENT
                };

                let ty = match &e.ty {
                    BindingType::UniformBuffer => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    BindingType::StorageBuffer { read_only } => wgpu::BindingType::Buffer {
                        ty: if *read_only {
                            wgpu::BufferBindingType::Storage { read_only: true }
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                };

                wgpu::BindGroupLayoutEntry {
                    binding: e.binding,
                    visibility,
                    ty,
                    count: None,
                }
            })
            .collect();

        let layout = device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RAG Bind Group Layout"),
            entries: &wgpu_entries,
        });

        let handle = self.next_bind_group_layout_handle;
        self.next_bind_group_layout_handle += 1;

        self.bind_group_layouts.insert(handle, WebGpuBindGroupLayout {
            device_handle,
            layout,
        });

        Ok(handle)
    }

    fn create_bind_group(&mut self, device_handle: DeviceHandle, layout_handle: BindGroupLayoutHandle, entries: &[BindGroupEntry]) -> Result<BindGroupHandle> {
        let device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let layout = self
            .bind_group_layouts
            .get(&layout_handle)
            .context("Invalid bind group layout handle")?;

        let wgpu_entries: Vec<_> = entries
            .iter()
            .filter_map(|e| {
                match &e.resource {
                    BindingResource::Buffer { buffer, offset, size } => {
                        self.buffers.get(buffer).map(|b| {
                            wgpu::BindGroupEntry {
                                binding: e.binding,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &b.buffer,
                                    offset: *offset,
                                    size: std::num::NonZeroU64::new(*size),
                                }),
                            }
                        })
                    }
                }
            })
            .collect();

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RAG Bind Group"),
            layout: &layout.layout,
            entries: &wgpu_entries,
        });

        let handle = self.next_bind_group_handle;
        self.next_bind_group_handle += 1;

        self.bind_groups.insert(handle, WebGpuBindGroup {
            device_handle,
            bind_group,
        });

        Ok(handle)
    }

    fn destroy_bind_group(&mut self, bind_group_handle: BindGroupHandle) {
        self.bind_groups.remove(&bind_group_handle);
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
        self.create_pipeline_with_layout(
            device_handle,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
            &[],
        )
    }

    fn create_pipeline_with_layout(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        bind_group_layouts: &[BindGroupLayoutHandle],
    ) -> Result<PipelineHandle> {
        let device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let vs = self.shaders.get(&vertex_shader).context("Invalid vertex shader")?;
        let fs = self.shaders.get(&fragment_shader).context("Invalid fragment shader")?;

        // Collect bind group layouts
        let wgpu_layouts: Vec<_> = bind_group_layouts
            .iter()
            .filter_map(|h| self.bind_group_layouts.get(h).map(|l| &l.layout))
            .collect();

        let layout_refs: Vec<&wgpu::BindGroupLayout> = wgpu_layouts.iter().map(|l| *l).collect();

        let pipeline_layout = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RAG Pipeline Layout"),
            bind_group_layouts: &layout_refs,
            push_constant_ranges: &[],
        });

        // Build vertex attributes
        let attributes: Vec<_> = vertex_layout
            .attributes
            .iter()
            .map(|a| wgpu::VertexAttribute {
                format: Self::vertex_format_to_wgpu(a.format),
                offset: a.offset as u64,
                shader_location: a.location,
            })
            .collect();

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: vertex_layout.stride as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attributes,
        };

        let pipeline = device.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RAG Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs.module,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs.module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: Self::format_to_wgpu(target_format),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: Self::topology_to_wgpu(topology),
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let handle = self.next_pipeline_handle;
        self.next_pipeline_handle += 1;

        self.pipelines.insert(handle, WebGpuPipeline {
            device_handle,
            pipeline,
        });

        Ok(handle)
    }

    fn destroy_pipeline(&mut self, pipeline_handle: PipelineHandle) {
        self.pipelines.remove(&pipeline_handle);
    }

    fn begin_frame(&mut self, device_handle: DeviceHandle, width: u32, height: u32, format: TextureFormat) -> Result<()> {
        let device = self
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        let needs_recreate = match &device.frame_state {
            Some(state) => state.width != width || state.height != height || state.format != format,
            None => true,
        };

        if needs_recreate {
            // Create render target texture
            let texture = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("RAG Frame Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::format_to_wgpu(format),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });

            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create output buffer for readback
            let buffer_size = (width * height * format.bytes_per_pixel()) as u64;
            // Ensure alignment to 256 bytes (COPY_BYTES_PER_ROW_ALIGNMENT)
            let aligned_width = ((width * format.bytes_per_pixel() + 255) / 256) * 256;
            let aligned_buffer_size = (aligned_width * height) as u64;

            let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RAG Output Buffer"),
                size: aligned_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            device.frame_state = Some(WebGpuFrameState {
                width,
                height,
                format,
                texture,
                view,
                output_buffer,
            });
        }

        Ok(())
    }

    fn execute_commands(&mut self, device_handle: DeviceHandle, commands: &[RenderCommand]) -> Result<()> {
        let device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let frame = device
            .frame_state
            .as_ref()
            .context("begin_frame not called")?;

        // Find clear color
        let clear_color = commands
            .iter()
            .find_map(|c| match c {
                RenderCommand::Clear(color) => Some(*color),
                _ => None,
            })
            .unwrap_or(Color::BLACK);

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RAG Command Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RAG Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_color.r as f64,
                            g: clear_color.g as f64,
                            b: clear_color.b as f64,
                            a: clear_color.a as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let mut current_pipeline: Option<&WebGpuPipeline> = None;

            for command in commands {
                match command {
                    RenderCommand::Clear(_) => {
                        // Handled by load op
                    }
                    RenderCommand::SetPipeline(pipeline_handle) => {
                        if let Some(pipeline) = self.pipelines.get(pipeline_handle) {
                            render_pass.set_pipeline(&pipeline.pipeline);
                            current_pipeline = Some(pipeline);
                        }
                    }
                    RenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                        if let Some(buf) = self.buffers.get(buffer) {
                            render_pass.set_vertex_buffer(*slot, buf.buffer.slice(*offset..));
                        }
                    }
                    RenderCommand::SetBindGroup { index, bind_group } => {
                        if let Some(bg) = self.bind_groups.get(bind_group) {
                            render_pass.set_bind_group(*index, &bg.bind_group, &[]);
                        }
                    }
                    RenderCommand::Draw {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    } => {
                        render_pass.draw(*first_vertex..(*first_vertex + *vertex_count), *first_instance..(*first_instance + *instance_count));
                    }
                }
            }
        }

        // Copy texture to buffer for readback
        let bytes_per_row = frame.width * frame.format.bytes_per_pixel();
        let aligned_bytes_per_row = ((bytes_per_row + 255) / 256) * 256;

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &frame.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(frame.height),
                },
            },
            wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );

        device.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    fn end_frame(&mut self, device_handle: DeviceHandle, output: &mut [u8]) -> Result<()> {
        let device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let frame = device
            .frame_state
            .as_ref()
            .context("begin_frame not called")?;

        let expected_size = (frame.width * frame.height * frame.format.bytes_per_pixel()) as usize;
        if output.len() < expected_size {
            anyhow::bail!(
                "Output buffer too small: {} < {}",
                output.len(),
                expected_size
            );
        }

        // Map buffer for reading
        let buffer_slice = frame.output_buffer.slice(..);

        // Use pollster for sync mapping on native, or wasm_bindgen_futures on web
        #[cfg(target_arch = "wasm32")]
        {
            // On WASM, we need to handle this differently
            // For now, this is a limitation - proper async handling needed
            anyhow::bail!("Sync end_frame not supported on WASM - use async variant")
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            device.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().context("Failed to map buffer")?;

            let data = buffer_slice.get_mapped_range();

            // Copy with row alignment handling
            let bytes_per_row = (frame.width * frame.format.bytes_per_pixel()) as usize;
            let aligned_bytes_per_row = ((bytes_per_row + 255) / 256) * 256;

            for y in 0..frame.height as usize {
                let src_start = y * aligned_bytes_per_row;
                let dst_start = y * bytes_per_row;
                output[dst_start..dst_start + bytes_per_row]
                    .copy_from_slice(&data[src_start..src_start + bytes_per_row]);
            }

            drop(data);
            frame.output_buffer.unmap();

            Ok(())
        }
    }
}

// For WASM, we need to be able to create the backend
#[cfg(target_arch = "wasm32")]
pub async fn create_webgpu_backend() -> Result<WebGpuBackend> {
    WebGpuBackend::new().await
}

