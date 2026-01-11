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
                    label: Some("Goldy Device"),
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
            label: Some("Goldy Buffer"),
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
            label: Some("Goldy Shader"),
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
            label: Some("Goldy Bind Group Layout"),
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
            label: Some("Goldy Bind Group"),
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
            label: Some("Goldy Pipeline Layout"),
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
            label: Some("Goldy Render Pipeline"),
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
}

// For WASM, we need to be able to create the backend
#[cfg(target_arch = "wasm32")]
pub async fn create_webgpu_backend() -> Result<WebGpuBackend> {
    WebGpuBackend::new().await
}

