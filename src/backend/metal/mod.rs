//! Metal backend implementation for macOS.
//!
//! Targets Metal 3.0+ on macOS 13+.
//! Uses Slang for shader compilation (Slang -> MSL).
//!
//! ## Module Structure
//!
//! - `types`: Internal state structs for devices, buffers, shaders, etc.
//! - `utils`: Format conversion and helpers

// Allow deprecated cocoa crate items - we use them intentionally
// The newer objc2 crate has API compatibility issues with CAMetalLayer
#![allow(deprecated)]

mod types;
mod utils;

use types::{
    BindGroupLayoutState, BindGroupState, BindingState, BufferState, ComputePipelineState,
    LogicalDevice, PipelineState, RenderTargetState, ResourceRegistry,
    SamplerState_ as SamplerStateInternal, SurfaceState, TextureState, ARGUMENT_BUFFER_SIZE,
    MAX_FRAMES_IN_FLIGHT,
};
use utils::{
    address_mode_to_mtl, compare_to_mtl, depth_format_to_mtl, filter_to_mtl, format_to_mtl,
    index_format_to_mtl, mipmap_mode_to_mtl, topology_to_mtl, vertex_format_to_mtl,
};

use super::*;
use crate::types::Color;
use crate::{goldy_event, goldy_span};
use anyhow::{Context, Result};
use cocoa::base::{id, nil, YES};
use core_graphics_types::geometry::CGSize;
use foreign_types::{ForeignType, ForeignTypeRef};
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};
use std::collections::HashMap;

// Re-import metal crate with explicit path to avoid name collision
use ::metal as mtl;
use mtl::{
    Device as MTLDevice, HeapDescriptor, Library, MTLCPUCacheMode, MTLClearColor, MTLHeapType,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLResourceOptions,
    MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureUsage, RenderPassDescriptor,
    TextureDescriptor,
};

// Re-export from our types module
use types::ShaderState;

/// Parse [numthreads(x, y, z)] from Slang shader source.
/// Returns None if not found.
fn parse_numthreads(source: &str) -> Option<[u32; 3]> {
    // Find [numthreads(x, y, z)] pattern
    let start = source.find("[numthreads(")?;
    let after_open = start + "[numthreads(".len();
    let end = source[after_open..].find(')')? + after_open;
    let args = &source[after_open..end];

    // Split by comma and parse
    let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
    if parts.len() != 3 {
        return None;
    }

    let x = parts[0].parse().ok()?;
    let y = parts[1].parse().ok()?;
    let z = parts[2].parse().ok()?;

    Some([x, y, z])
}

/// Metal backend for macOS.
pub struct MetalBackend {
    devices: HashMap<DeviceHandle, LogicalDevice>,
    next_device_handle: DeviceHandle,
    buffers: HashMap<BufferHandle, BufferState>,
    next_buffer_handle: BufferHandle,
    shaders: HashMap<ShaderHandle, ShaderState>,
    next_shader_handle: ShaderHandle,
    pipelines: HashMap<PipelineHandle, PipelineState>,
    next_pipeline_handle: PipelineHandle,
    compute_pipelines: HashMap<ComputePipelineHandle, ComputePipelineState>,
    next_compute_pipeline_handle: ComputePipelineHandle,
    bind_group_layouts: HashMap<BindGroupLayoutHandle, BindGroupLayoutState>,
    next_bind_group_layout_handle: BindGroupLayoutHandle,
    bind_groups: HashMap<BindGroupHandle, BindGroupState>,
    next_bind_group_handle: BindGroupHandle,
    render_targets: HashMap<RenderTargetHandle, RenderTargetState>,
    next_render_target_handle: RenderTargetHandle,
    surfaces: HashMap<SurfaceHandle, SurfaceState>,
    next_surface_handle: SurfaceHandle,
    textures: HashMap<TextureHandle, TextureState>,
    next_texture_handle: TextureHandle,
    samplers: HashMap<SamplerHandle, SamplerStateInternal>,
    next_sampler_handle: SamplerHandle,
    /// Per-backend Slang compiler instance
    slang_compiler: crate::slang::SlangCompiler,
}

impl MetalBackend {
    /// Create a new Metal backend.
    pub fn new() -> Result<Self> {
        let _span = goldy_span!("backend.metal.init").entered();
        tracing::info!("Initializing Metal backend");

        // Create Slang compiler
        let slang_compiler =
            crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

        goldy_event!("backend.metal.init", success = true);

        Ok(Self {
            devices: HashMap::new(),
            next_device_handle: 1,
            buffers: HashMap::new(),
            next_buffer_handle: 1,
            shaders: HashMap::new(),
            next_shader_handle: 1,
            pipelines: HashMap::new(),
            next_pipeline_handle: 1,
            compute_pipelines: HashMap::new(),
            next_compute_pipeline_handle: 1,
            bind_group_layouts: HashMap::new(),
            next_bind_group_layout_handle: 1,
            bind_groups: HashMap::new(),
            next_bind_group_handle: 1,
            render_targets: HashMap::new(),
            next_render_target_handle: 1,
            surfaces: HashMap::new(),
            next_surface_handle: 1,
            textures: HashMap::new(),
            next_texture_handle: 1,
            samplers: HashMap::new(),
            next_sampler_handle: 1,
            slang_compiler,
        })
    }

    /// Compile a shader stage to MSL and create a Metal library, returning reflection data.
    fn compile_shader_stage_with_reflection(
        &self,
        device: &MTLDevice,
        slang_source: &str,
        search_paths: &[String],
        entry_point: &str,
        stage: crate::slang::SlangStage,
        bindless: bool,
    ) -> Result<(Library, Option<crate::slang::ShaderReflection>)> {
        let search_path_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();

        // Compile Slang to MSL with specific entry point
        // Use compile_bindless_with_reflection when bindless is enabled
        let (compiled, reflection) = if bindless {
            let result = self
                .slang_compiler
                .compile_bindless_with_reflection(
                    slang_source,
                    crate::slang::ShaderTarget::Metal,
                    &[(entry_point, stage)],
                    &search_path_refs,
                )
                .with_context(|| format!("Failed to compile {} shader stage", entry_point))?;

            // Log reflection data for debugging
            if !result.reflection.parameter_blocks.is_empty() {
                tracing::info!(
                    "Shader {} has {} ParameterBlock(s):",
                    entry_point,
                    result.reflection.parameter_blocks.len()
                );
                for pb in &result.reflection.parameter_blocks {
                    tracing::info!(
                        "  - {} at slot {} (size={}, alignment={}, fields={})",
                        pb.name,
                        pb.binding_slot,
                        pb.size,
                        pb.alignment,
                        pb.fields.len()
                    );
                    for field in &pb.fields {
                        tracing::debug!(
                            "    - {}: {:?} at offset {} (size={})",
                            field.name,
                            field.resource_kind,
                            field.offset,
                            field.size
                        );
                    }
                }
            }

            (result.shader, Some(result.reflection))
        } else {
            let result = self
                .slang_compiler
                .compile_with_options(
                    slang_source,
                    crate::slang::ShaderTarget::Metal,
                    &[(entry_point, stage)],
                    &search_path_refs,
                )
                .with_context(|| format!("Failed to compile {} shader stage", entry_point))?;
            (result, None)
        };

        let msl_source = compiled
            .as_str()
            .context("Failed to get MSL source")?
            .to_string();

        tracing::debug!(
            "Compiled MSL {} shader ({} bytes, bindless={})",
            entry_point,
            msl_source.len(),
            bindless
        );

        // Create Metal library from MSL
        let library = device
            .new_library_with_source(&msl_source, &mtl::CompileOptions::new())
            .map_err(|e| {
                anyhow::anyhow!("Failed to create Metal library for {}: {}", entry_point, e)
            })?;

        Ok((library, reflection))
    }

    /// Ensure the vertex shader stage is compiled.
    fn ensure_vertex_shader_compiled(&mut self, shader_handle: ShaderHandle) -> Result<()> {
        let shader = self
            .shaders
            .get(&shader_handle)
            .context("Invalid shader handle")?;

        if shader.vertex_library.is_some() {
            return Ok(());
        }

        let device_handle = shader.device_handle;
        let slang_source = shader.slang_source.clone();
        let search_paths = shader.search_paths.clone();

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Shader's device no longer valid")?;

        let bindless = logical_device.bindless_enabled;
        let (library, reflection) = self.compile_shader_stage_with_reflection(
            &logical_device.device,
            &slang_source,
            &search_paths,
            "vs_main",
            crate::slang::SlangStage::Vertex,
            bindless,
        )?;

        let shader = self.shaders.get_mut(&shader_handle).unwrap();
        shader.vertex_library = Some(library);
        // Store reflection if not already set (first stage to compile stores it)
        if shader.reflection.is_none() {
            shader.reflection = reflection;
        }

        Ok(())
    }

    /// Ensure the fragment shader stage is compiled.
    fn ensure_fragment_shader_compiled(&mut self, shader_handle: ShaderHandle) -> Result<()> {
        let shader = self
            .shaders
            .get(&shader_handle)
            .context("Invalid shader handle")?;

        if shader.fragment_library.is_some() {
            return Ok(());
        }

        let device_handle = shader.device_handle;
        let slang_source = shader.slang_source.clone();
        let search_paths = shader.search_paths.clone();

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Shader's device no longer valid")?;

        let bindless = logical_device.bindless_enabled;
        let (library, reflection) = self.compile_shader_stage_with_reflection(
            &logical_device.device,
            &slang_source,
            &search_paths,
            "fs_main",
            crate::slang::SlangStage::Fragment,
            bindless,
        )?;

        let shader = self.shaders.get_mut(&shader_handle).unwrap();
        shader.fragment_library = Some(library);
        // Store reflection if not already set
        if shader.reflection.is_none() {
            shader.reflection = reflection;
        }

        Ok(())
    }

    /// Ensure the compute shader stage is compiled.
    fn ensure_compute_shader_compiled(&mut self, shader_handle: ShaderHandle) -> Result<()> {
        let shader = self
            .shaders
            .get(&shader_handle)
            .context("Invalid shader handle")?;

        if shader.compute_library.is_some() {
            return Ok(());
        }

        let device_handle = shader.device_handle;
        let slang_source = shader.slang_source.clone();
        let search_paths = shader.search_paths.clone();

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Shader's device no longer valid")?;

        let bindless = logical_device.bindless_enabled;
        let (library, reflection) = self.compile_shader_stage_with_reflection(
            &logical_device.device,
            &slang_source,
            &search_paths,
            "cs_main",
            crate::slang::SlangStage::Compute,
            bindless,
        )?;

        let shader = self.shaders.get_mut(&shader_handle).unwrap();
        shader.compute_library = Some(library);
        // Store reflection if not already set
        if shader.reflection.is_none() {
            shader.reflection = reflection;
        }

        Ok(())
    }

    /// Create a render pass for the given texture with clear color.
    fn create_render_pass<'a>(
        texture: &mtl::TextureRef,
        depth_texture: Option<&mtl::TextureRef>,
        clear_color: Option<Color>,
        clear_depth: Option<f32>,
    ) -> &'a mtl::RenderPassDescriptorRef {
        let descriptor = RenderPassDescriptor::new();

        let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(texture));

        if let Some(color) = clear_color {
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_clear_color(MTLClearColor::new(
                color.r as f64,
                color.g as f64,
                color.b as f64,
                color.a as f64,
            ));
        } else {
            color_attachment.set_load_action(MTLLoadAction::Load);
        }
        color_attachment.set_store_action(MTLStoreAction::Store);

        if let Some(depth) = depth_texture {
            let depth_attachment = descriptor.depth_attachment().unwrap();
            depth_attachment.set_texture(Some(depth));
            if let Some(depth_value) = clear_depth {
                depth_attachment.set_load_action(MTLLoadAction::Clear);
                depth_attachment.set_clear_depth(depth_value as f64);
            } else {
                depth_attachment.set_load_action(MTLLoadAction::Load);
            }
            depth_attachment.set_store_action(MTLStoreAction::Store);
        }

        descriptor
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        tracing::info!("Shutting down Metal backend");

        // Destroy all devices (which will clean up their resources)
        let device_handles: Vec<_> = self.devices.keys().copied().collect();
        for handle in device_handles {
            self.destroy_device(handle);
        }
    }
}

impl GpuBackend for MetalBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Metal
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        let devices = MTLDevice::all();
        devices
            .iter()
            .enumerate()
            .map(|(idx, device)| {
                let name = device.name().to_string();
                let device_type = if device.is_low_power() {
                    DeviceType::IntegratedGpu
                } else {
                    DeviceType::DiscreteGpu
                };

                AdapterInfo {
                    id: idx as u32,
                    name,
                    vendor: "Apple".to_string(),
                    backend: BackendType::Metal,
                    device_type,
                }
            })
            .collect()
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        let devices = MTLDevice::all();
        let device = devices
            .get(adapter_id as usize)
            .cloned()
            .or_else(MTLDevice::system_default)
            .context("No Metal device available")?;

        let command_queue = device.new_command_queue();

        // Check for Argument Buffers Tier 2 support (required for bindless)
        let arg_buffers_tier = device.argument_buffers_support();
        let bindless_enabled = arg_buffers_tier == mtl::MTLArgumentBuffersTier::Tier2;

        // Initialize bindless infrastructure if supported
        let (buffer_heap, texture_heap, argument_buffer, argument_encoder, texture_encoder) =
            if bindless_enabled {
                tracing::info!("Metal Argument Buffers Tier 2 supported - enabling bindless");

                // Create global argument buffer first (for storing resource IDs)
                let arg_buffer =
                    device.new_buffer(ARGUMENT_BUFFER_SIZE, MTLResourceOptions::StorageModeShared);
                tracing::info!("Created argument buffer");

                // Try to create heaps for resource allocation
                // Use Automatic heap type and Shared storage for CPU-accessible buffers
                // IMPORTANT: CPU cache mode must match between heap and buffer allocation
                let heap_size: u64 = 64 * 1024 * 1024; // 64MB (smaller to start)

                tracing::info!("Creating buffer heap...");
                let buffer_heap_desc = HeapDescriptor::new();
                buffer_heap_desc.set_size(heap_size);
                buffer_heap_desc.set_storage_mode(MTLStorageMode::Shared);
                buffer_heap_desc.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
                buffer_heap_desc.set_heap_type(MTLHeapType::Automatic);
                let buffer_heap = device.new_heap(&buffer_heap_desc);
                tracing::info!("Created buffer heap (size={}MB)", heap_size / 1024 / 1024);

                tracing::info!("Creating texture heap...");
                let texture_heap_desc = HeapDescriptor::new();
                texture_heap_desc.set_size(heap_size);
                // Use Shared storage to allow CPU writes via replace_region()
                // Private would require staging buffer + blit
                texture_heap_desc.set_storage_mode(MTLStorageMode::Shared);
                texture_heap_desc.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
                texture_heap_desc.set_heap_type(MTLHeapType::Automatic);
                let texture_heap = device.new_heap(&texture_heap_desc);
                tracing::info!("Created texture heap (size={}MB)", heap_size / 1024 / 1024);

                // Create ArgumentEncoder for encoding buffers into argument buffer
                // Each slot in the argument buffer holds one resource reference
                let buffer_arg_desc = mtl::ArgumentDescriptor::new();
                buffer_arg_desc.set_index(0);
                buffer_arg_desc.set_data_type(mtl::MTLDataType::Pointer);
                buffer_arg_desc.set_access(mtl::MTLArgumentAccess::ReadWrite);
                let buffer_encoder =
                    device.new_argument_encoder(mtl::Array::from_slice(&[buffer_arg_desc]));
                tracing::info!(
                    "Created buffer ArgumentEncoder (encoded_length={})",
                    buffer_encoder.encoded_length()
                );

                // Create ArgumentEncoder for encoding textures
                let texture_arg_desc = mtl::ArgumentDescriptor::new();
                texture_arg_desc.set_index(0);
                texture_arg_desc.set_data_type(mtl::MTLDataType::Texture);
                texture_arg_desc.set_texture_type(mtl::MTLTextureType::D2);
                texture_arg_desc.set_access(mtl::MTLArgumentAccess::ReadOnly);
                let texture_encoder =
                    device.new_argument_encoder(mtl::Array::from_slice(&[texture_arg_desc]));
                tracing::info!(
                    "Created texture ArgumentEncoder (encoded_length={})",
                    texture_encoder.encoded_length()
                );

                (
                    Some(buffer_heap),
                    Some(texture_heap),
                    Some(arg_buffer),
                    Some(buffer_encoder),
                    Some(texture_encoder),
                )
            } else {
                tracing::info!(
                    "Metal Argument Buffers Tier 2 not supported - using traditional bindings"
                );
                (None, None, None, None, None)
            };

        let handle = self.next_device_handle;
        self.next_device_handle += 1;

        tracing::info!(
            "Created Metal device {} for adapter {} ({}) [bindless={}]",
            handle,
            adapter_id,
            device.name(),
            bindless_enabled
        );

        self.devices.insert(
            handle,
            LogicalDevice {
                device,
                command_queue,
                adapter_id,
                buffer_heap,
                texture_heap,
                argument_buffer,
                argument_encoder,
                texture_encoder,
                resource_registry: ResourceRegistry::new(),
                bindless_enabled,
                heap_buffer_count: 0,
                heap_texture_count: 0,
            },
        );

        Ok(handle)
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        if self.devices.remove(&device_handle).is_some() {
            // Clean up resources owned by this device
            self.buffers.retain(|_, b| b.device_handle != device_handle);
            self.shaders.retain(|_, s| s.device_handle != device_handle);
            self.pipelines
                .retain(|_, p| p.device_handle != device_handle);
            self.compute_pipelines
                .retain(|_, p| p.device_handle != device_handle);
            self.bind_group_layouts
                .retain(|_, l| l.device_handle != device_handle);
            self.bind_groups
                .retain(|_, g| g.device_handle != device_handle);
            self.render_targets
                .retain(|_, t| t.device_handle != device_handle);
            self.surfaces
                .retain(|_, s| s.device_handle != device_handle);
            self.textures
                .retain(|_, t| t.device_handle != device_handle);
            self.samplers
                .retain(|_, s| s.device_handle != device_handle);

            tracing::info!("Destroyed Metal device {}", device_handle);
        }
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn create_buffer(
        &mut self,
        device_handle: DeviceHandle,
        size: u64,
        _usage: BufferUsage,
        _element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        let _span = goldy_span!("resource.buffer.create", size = size).entered();

        let logical_device = self
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;

        // Allocate buffer - from heap if bindless, otherwise traditional
        let (buffer, staging_buffer, is_heap_allocated, arg_buffer_index) =
            if logical_device.bindless_enabled {
                if let Some(heap) = &logical_device.buffer_heap {
                    // Allocate from heap with Shared storage (CPU-accessible)
                    // Use default CPU cache mode to match the heap's mode
                    let options = MTLResourceOptions::StorageModeShared
                        | MTLResourceOptions::CPUCacheModeDefaultCache;

                    match heap.new_buffer(size, options) {
                        Some(buffer) => {
                            // Register in bindless registry
                            let index = logical_device.resource_registry.register_buffer(handle);
                            tracing::debug!(
                                "Allocated buffer {} from heap at bindless index {}",
                                handle,
                                index
                            );

                            // Encode buffer into argument buffer using ArgumentEncoder
                            if let (Some(arg_buffer), Some(encoder)) = (
                                &logical_device.argument_buffer,
                                &logical_device.argument_encoder,
                            ) {
                                let encoded_length = encoder.encoded_length();
                                let offset = (index as u64) * encoded_length;

                                if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
                                    // Point encoder at the correct offset in argument buffer
                                    encoder.set_argument_buffer(arg_buffer, offset);
                                    // Encode the buffer at index 0 within this slot
                                    encoder.set_buffer(0, &buffer, 0);
                                    tracing::trace!(
                                        "Encoded buffer {} at arg buffer offset {} (slot {})",
                                        handle,
                                        offset,
                                        index
                                    );
                                }
                            }

                            // Track heap allocation for use_heap_at safety
                            logical_device.heap_buffer_count += 1;

                            (buffer, None, true, Some(index))
                        }
                        None => {
                            // Heap allocation failed (e.g., heap full), fall back to traditional
                            tracing::warn!(
                            "Heap allocation failed for buffer {}, using traditional allocation",
                            handle
                        );
                            let options = MTLResourceOptions::StorageModeManaged
                                | MTLResourceOptions::CPUCacheModeWriteCombined;
                            let buffer = logical_device.device.new_buffer(size, options);
                            (buffer, None, false, None)
                        }
                    }
                } else {
                    // No heap available, use traditional allocation
                    let options = MTLResourceOptions::StorageModeManaged
                        | MTLResourceOptions::CPUCacheModeWriteCombined;
                    let buffer = logical_device.device.new_buffer(size, options);
                    (buffer, None, false, None)
                }
            } else {
                // Traditional allocation for non-bindless
                let options = MTLResourceOptions::StorageModeManaged
                    | MTLResourceOptions::CPUCacheModeWriteCombined;
                let buffer = logical_device.device.new_buffer(size, options);
                (buffer, None, false, None)
            };

        self.buffers.insert(
            handle,
            BufferState {
                device_handle,
                buffer,
                staging_buffer,
                size,
                arg_buffer_index,
                is_heap_allocated,
            },
        );

        Ok(handle)
    }

    fn destroy_buffer(&mut self, buffer_handle: BufferHandle) {
        if let Some(buffer) = self.buffers.remove(&buffer_handle) {
            // Unregister from bindless registry
            if let Some(device) = self.devices.get_mut(&buffer.device_handle) {
                device.resource_registry.unregister_buffer(buffer_handle);
            }
        }
    }

    fn write_buffer(
        &mut self,
        buffer_handle: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<()> {
        let buffer = self
            .buffers
            .get(&buffer_handle)
            .context("Invalid buffer handle")?;

        if offset + data.len() as u64 > buffer.size {
            anyhow::bail!("Write would exceed buffer bounds");
        }

        unsafe {
            let ptr = buffer.buffer.contents().add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        }

        // Notify Metal of the modification (only needed for Managed storage)
        // Heap-allocated buffers use Shared storage, which doesn't need this
        if !buffer.is_heap_allocated {
            buffer
                .buffer
                .did_modify_range(mtl::NSRange::new(offset, data.len() as u64));
        }

        Ok(())
    }

    fn buffer_size(&self, buffer_handle: BufferHandle) -> u64 {
        self.buffers
            .get(&buffer_handle)
            .map(|b| b.size)
            .unwrap_or(0)
    }

    fn buffer_bindless_index(&self, buffer_handle: BufferHandle) -> Option<u32> {
        self.buffers.get(&buffer_handle).and_then(|b| b.arg_buffer_index)
    }

    fn create_shader(
        &mut self,
        device_handle: DeviceHandle,
        slang_source: &str,
    ) -> Result<ShaderHandle> {
        self.create_shader_with_paths(device_handle, slang_source, &[])
    }

    fn create_shader_with_paths(
        &mut self,
        device_handle: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
    ) -> Result<ShaderHandle> {
        // Just validate the device exists - actual compilation happens at pipeline creation
        let _ = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let handle = self.next_shader_handle;
        self.next_shader_handle += 1;

        self.shaders.insert(
            handle,
            ShaderState {
                device_handle,
                slang_source: slang_source.to_string(),
                search_paths: search_paths.iter().map(|s| s.to_string()).collect(),
                vertex_library: None,
                fragment_library: None,
                compute_library: None,
                reflection: None,
            },
        );

        tracing::debug!("Created shader handle {} (compilation deferred)", handle);
        Ok(handle)
    }

    fn destroy_shader(&mut self, shader_handle: ShaderHandle) {
        self.shaders.remove(&shader_handle);
    }

    fn create_bind_group_layout(
        &mut self,
        device_handle: DeviceHandle,
        entries: &[BindGroupLayoutEntry],
    ) -> Result<BindGroupLayoutHandle> {
        let _ = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let handle = self.next_bind_group_layout_handle;
        self.next_bind_group_layout_handle += 1;

        self.bind_group_layouts.insert(
            handle,
            BindGroupLayoutState {
                device_handle,
                entries: entries.to_vec(),
            },
        );

        Ok(handle)
    }

    fn create_bind_group(
        &mut self,
        device_handle: DeviceHandle,
        layout: BindGroupLayoutHandle,
        entries: &[BindGroupEntry],
    ) -> Result<BindGroupHandle> {
        let _ = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let bindings: Vec<BindingState> = entries
            .iter()
            .map(|e| match &e.resource {
                BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                } => BindingState::Buffer {
                    buffer: *buffer,
                    offset: *offset,
                    size: *size,
                },
                BindingResource::Texture(tex) => BindingState::Texture(*tex),
                BindingResource::Sampler(samp) => BindingState::Sampler(*samp),
            })
            .collect();

        let handle = self.next_bind_group_handle;
        self.next_bind_group_handle += 1;

        self.bind_groups.insert(
            handle,
            BindGroupState {
                device_handle,
                layout_handle: layout,
                bindings,
            },
        );

        Ok(handle)
    }

    fn destroy_bind_group(&mut self, bind_group: BindGroupHandle) {
        self.bind_groups.remove(&bind_group);
    }

    fn create_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        self.create_pipeline_with_depth(
            device_handle,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            _topology,
            target_format,
            &[],
            None,
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
        self.create_pipeline_with_depth(
            device_handle,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
            bind_group_layouts,
            None,
        )
    }

    fn create_pipeline_with_depth(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        _bind_group_layouts: &[BindGroupLayoutHandle],
        depth_stencil: Option<&crate::types::DepthStencilState>,
    ) -> Result<PipelineHandle> {
        // Ensure shaders are compiled for the required stages
        self.ensure_vertex_shader_compiled(vertex_shader)?;
        self.ensure_fragment_shader_compiled(fragment_shader)?;

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let vs_shader = self
            .shaders
            .get(&vertex_shader)
            .context("Invalid vertex shader")?;
        let fs_shader = self
            .shaders
            .get(&fragment_shader)
            .context("Invalid fragment shader")?;

        let vs_library = vs_shader.vertex_library.as_ref().unwrap();
        let fs_library = fs_shader.fragment_library.as_ref().unwrap();

        // Get entry point functions - Slang outputs the original function names for MSL
        let vs_function = vs_library
            .get_function("vs_main", None)
            .map_err(|e| anyhow::anyhow!("Failed to get vertex function: {}", e))?;

        let fs_function = fs_library
            .get_function("fs_main", None)
            .map_err(|e| anyhow::anyhow!("Failed to get fragment function: {}", e))?;

        // Create pipeline descriptor
        let descriptor = mtl::RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&vs_function));
        descriptor.set_fragment_function(Some(&fs_function));

        // Set color attachment format
        let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
        color_attachment.set_pixel_format(format_to_mtl(target_format));

        // Set vertex descriptor (only if there are vertex attributes)
        // For vertex-less rendering (e.g., fullscreen triangle from SV_VertexID),
        // we skip the vertex descriptor entirely
        if !vertex_layout.attributes.is_empty() {
            let vertex_descriptor = mtl::VertexDescriptor::new();
            let layout = vertex_descriptor.layouts().object_at(0).unwrap();
            layout.set_stride(vertex_layout.stride as u64);
            layout.set_step_function(mtl::MTLVertexStepFunction::PerVertex);

            for attr in &vertex_layout.attributes {
                let attr_desc = vertex_descriptor
                    .attributes()
                    .object_at(attr.location as u64)
                    .unwrap();
                attr_desc.set_format(vertex_format_to_mtl(attr.format));
                attr_desc.set_offset(attr.offset as u64);
                attr_desc.set_buffer_index(0);
            }

            descriptor.set_vertex_descriptor(Some(vertex_descriptor));
        }

        // Set depth format if depth stencil is enabled
        let depth_stencil_state = if let Some(ds) = depth_stencil {
            descriptor.set_depth_attachment_pixel_format(depth_format_to_mtl(ds.format));

            // Create depth stencil state
            let ds_descriptor = mtl::DepthStencilDescriptor::new();
            ds_descriptor.set_depth_compare_function(compare_to_mtl(ds.depth_compare));
            ds_descriptor.set_depth_write_enabled(ds.depth_write_enabled);

            Some(
                logical_device
                    .device
                    .new_depth_stencil_state(&ds_descriptor),
            )
        } else {
            None
        };

        // Create pipeline state
        let pipeline = logical_device
            .device
            .new_render_pipeline_state(&descriptor)
            .map_err(|e| anyhow::anyhow!("Failed to create render pipeline: {}", e))?;

        // Extract ParameterBlock layouts from shader reflection for bindless rendering
        let parameter_block_layouts = vs_shader
            .reflection
            .as_ref()
            .map(|r| r.parameter_blocks.clone())
            .unwrap_or_default();

        // Allocate argument buffer for ParameterBlocks if bindless is enabled
        let bindless_arg_buffer = if logical_device.bindless_enabled
            && !parameter_block_layouts.is_empty()
        {
            // Calculate total size needed for all ParameterBlock structs
            // For simplicity, use the first ParameterBlock's size (most common case)
            let total_size = parameter_block_layouts
                .iter()
                .map(|pb| pb.size as u64)
                .max()
                .unwrap_or(64)
                .max(64); // Minimum 64 bytes for alignment

            let arg_buffer = logical_device
                .device
                .new_buffer(total_size, MTLResourceOptions::StorageModeShared);

            tracing::info!(
                "Allocated bindless argument buffer ({} bytes) for pipeline with {} ParameterBlock(s)",
                total_size,
                parameter_block_layouts.len()
            );
            for pb in &parameter_block_layouts {
                tracing::debug!(
                    "  ParameterBlock '{}': slot={}, size={}, fields={}",
                    pb.name,
                    pb.binding_slot,
                    pb.size,
                    pb.fields.len()
                );
            }

            Some(arg_buffer)
        } else {
            None
        };

        let handle = self.next_pipeline_handle;
        self.next_pipeline_handle += 1;

        self.pipelines.insert(
            handle,
            PipelineState {
                device_handle,
                pipeline,
                depth_stencil: depth_stencil_state,
                primitive_type: topology_to_mtl(topology),
                bindless_arg_buffer,
                parameter_block_layouts,
            },
        );

        tracing::debug!(
            "Created render pipeline {} with topology {:?}",
            handle,
            topology
        );
        Ok(handle)
    }

    fn destroy_pipeline(&mut self, pipeline_handle: PipelineHandle) {
        self.pipelines.remove(&pipeline_handle);
    }

    fn create_render_target(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<RenderTargetHandle> {
        self.create_render_target_with_depth(device_handle, width, height, format, None)
    }

    fn create_render_target_with_depth(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Create color texture
        let descriptor = TextureDescriptor::new();
        descriptor.set_width(width as u64);
        descriptor.set_height(height as u64);
        descriptor.set_pixel_format(format_to_mtl(color_format));
        descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        descriptor.set_storage_mode(MTLStorageMode::Private);

        let texture = logical_device.device.new_texture(&descriptor);

        // Create depth texture if requested
        let depth_texture = depth_format.map(|df| {
            let depth_desc = TextureDescriptor::new();
            depth_desc.set_width(width as u64);
            depth_desc.set_height(height as u64);
            depth_desc.set_pixel_format(depth_format_to_mtl(df));
            depth_desc.set_usage(MTLTextureUsage::RenderTarget);
            depth_desc.set_storage_mode(MTLStorageMode::Private);
            logical_device.device.new_texture(&depth_desc)
        });

        let handle = self.next_render_target_handle;
        self.next_render_target_handle += 1;

        self.render_targets.insert(
            handle,
            RenderTargetState {
                device_handle,
                width,
                height,
                format: color_format,
                texture,
                depth_format,
                depth_texture,
                has_rendered: false,
            },
        );

        tracing::debug!(
            "Created render target {} ({}x{}, {:?})",
            handle,
            width,
            height,
            color_format
        );
        Ok(handle)
    }

    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        self.render_targets.remove(&target);
    }

    fn render_to_target(
        &mut self,
        device_handle: DeviceHandle,
        target: RenderTargetHandle,
        commands: &[RenderCommand],
    ) -> Result<()> {
        let _span = goldy_span!(
            "render.pass.execute",
            target = target,
            commands = commands.len()
        )
        .entered();

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let render_target = self
            .render_targets
            .get(&target)
            .context("Invalid render target")?;

        // Find clear color and depth from commands
        let mut clear_color = None;
        let mut clear_depth = None;
        for cmd in commands {
            match cmd {
                RenderCommand::Clear(color) => clear_color = Some(*color),
                RenderCommand::ClearDepth(depth) => clear_depth = Some(*depth),
                _ => {}
            }
        }

        // Create render pass
        let render_pass = Self::create_render_pass(
            &render_target.texture,
            render_target.depth_texture.as_deref(),
            clear_color,
            clear_depth,
        );

        // Create command buffer and encoder
        let command_buffer = logical_device.command_queue.new_command_buffer();
        let encoder = command_buffer.new_render_command_encoder(render_pass);

        // Set up bindless rendering if enabled
        // Note: Bindless setup is deferred until we have actual heap resources
        // This avoids issues with empty heaps or shader incompatibility
        if logical_device.bindless_enabled {
            // Make heaps resident for the render pass (only if they have resources)
            let render_stages = mtl::MTLRenderStages::Vertex | mtl::MTLRenderStages::Fragment;
            if logical_device.heap_buffer_count > 0 {
                if let Some(buffer_heap) = &logical_device.buffer_heap {
                    encoder.use_heap_at(buffer_heap, render_stages);
                }
            }
            if logical_device.heap_texture_count > 0 {
                if let Some(texture_heap) = &logical_device.texture_heap {
                    encoder.use_heap_at(texture_heap, render_stages);
                }
            }

            // Note: Argument buffer binding is only needed when shaders use it
            // Binding to an unused slot is fine, but we skip it for now to isolate issues
        }

        // Set viewport and scissor
        encoder.set_viewport(mtl::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: render_target.width as f64,
            height: render_target.height as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        encoder.set_scissor_rect(mtl::MTLScissorRect {
            x: 0,
            y: 0,
            width: render_target.width as u64,
            height: render_target.height as u64,
        });

        // Cache bindless state for use in loop
        let bindless_enabled = logical_device.bindless_enabled;
        let argument_buffer = logical_device.argument_buffer.as_ref();

        // Process commands
        let mut current_index_buffer: Option<(BufferHandle, u64, IndexFormat)> = None;
        let mut current_primitive_type = MTLPrimitiveType::Triangle;
        let mut current_pipeline: Option<&PipelineState> = None;

        for cmd in commands {
            match cmd {
                RenderCommand::Clear(_) | RenderCommand::ClearDepth(_) => {
                    // Already handled in render pass setup
                }
                RenderCommand::SetPipeline(pipeline_handle) => {
                    if let Some(pipeline) = self.pipelines.get(pipeline_handle) {
                        encoder.set_render_pipeline_state(&pipeline.pipeline);
                        current_primitive_type = pipeline.primitive_type;
                        current_pipeline = Some(pipeline);
                        if let Some(ds) = &pipeline.depth_stencil {
                            encoder.set_depth_stencil_state(ds);
                        }
                    }
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                } => {
                    if let Some(buf) = self.buffers.get(buffer) {
                        encoder.set_vertex_buffer(*slot as u64, Some(&buf.buffer), *offset);
                    }
                }
                RenderCommand::SetIndexBuffer {
                    buffer,
                    offset,
                    format,
                } => {
                    current_index_buffer = Some((*buffer, *offset, *format));
                }
                RenderCommand::SetBindGroup { index, bind_group } => {
                    if let Some(bg) = self.bind_groups.get(bind_group) {
                        // Check if we should use ParameterBlock-based bindless
                        let use_parameter_block = bindless_enabled
                            && current_pipeline
                                .map(|p| !p.parameter_block_layouts.is_empty())
                                .unwrap_or(false);

                        if use_parameter_block {
                            // NEW: ParameterBlock-based bindless rendering
                            // Write GPU addresses directly to the pipeline's argument buffer
                            if let Some(pipeline) = current_pipeline {
                                if let Some(arg_buffer) = &pipeline.bindless_arg_buffer {
                                    // For each binding, find corresponding field and write GPU address
                                    for (binding_idx, binding) in bg.bindings.iter().enumerate() {
                                        match binding {
                                            BindingState::Buffer { buffer, offset, .. } => {
                                                if let Some(buf) = self.buffers.get(buffer) {
                                                    // Find the field offset from reflection
                                                    if let Some(pb_layout) =
                                                        pipeline.parameter_block_layouts.first()
                                                    {
                                                        if let Some(field) =
                                                            pb_layout.fields.get(binding_idx)
                                                        {
                                                            // Write GPU address to argument buffer at field offset
                                                            let gpu_addr =
                                                                buf.buffer.gpu_address() + *offset;
                                                            unsafe {
                                                                let ptr = arg_buffer
                                                                    .contents()
                                                                    .add(field.offset);
                                                                *(ptr as *mut u64) = gpu_addr;
                                                            }
                                                            tracing::trace!(
                                                                "Wrote GPU address 0x{:x} at offset {} for field '{}'",
                                                                gpu_addr,
                                                                field.offset,
                                                                field.name
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                            BindingState::Texture(tex_handle) => {
                                                if let Some(tex) = self.textures.get(tex_handle) {
                                                    if let Some(pb_layout) =
                                                        pipeline.parameter_block_layouts.first()
                                                    {
                                                        if let Some(field) =
                                                            pb_layout.fields.get(binding_idx)
                                                        {
                                                            let resource_id =
                                                                tex.texture.gpu_resource_id()._impl;
                                                            unsafe {
                                                                let ptr = arg_buffer
                                                                    .contents()
                                                                    .add(field.offset);
                                                                *(ptr as *mut u64) = resource_id;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            BindingState::Sampler(samp_handle) => {
                                                if let Some(samp) = self.samplers.get(samp_handle) {
                                                    if let Some(pb_layout) =
                                                        pipeline.parameter_block_layouts.first()
                                                    {
                                                        if let Some(field) =
                                                            pb_layout.fields.get(binding_idx)
                                                        {
                                                            let resource_id = samp
                                                                .sampler
                                                                .gpu_resource_id()
                                                                ._impl;
                                                            unsafe {
                                                                let ptr = arg_buffer
                                                                    .contents()
                                                                    .add(field.offset);
                                                                *(ptr as *mut u64) = resource_id;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Bind the argument buffer at the slot from reflection
                                    if let Some(pb_layout) =
                                        pipeline.parameter_block_layouts.first()
                                    {
                                        encoder.set_vertex_buffer(
                                            pb_layout.binding_slot as u64,
                                            Some(arg_buffer),
                                            0,
                                        );
                                        encoder.set_fragment_buffer(
                                            pb_layout.binding_slot as u64,
                                            Some(arg_buffer),
                                            0,
                                        );
                                        tracing::trace!(
                                            "Bound ParameterBlock argument buffer at slot {}",
                                            pb_layout.binding_slot
                                        );
                                    }
                                }
                            }
                        } else {
                            // LEGACY: Traditional binding or old-style bindless with push constants
                            let mut bindless_indices = types::BindlessIndices::default();
                            let mut has_bindless_resources = false;

                            for (binding_idx, binding) in bg.bindings.iter().enumerate() {
                                let buffer_index = (*index as usize * 16 + binding_idx) as u64;
                                match binding {
                                    BindingState::Buffer { buffer, offset, .. } => {
                                        if let Some(buf) = self.buffers.get(buffer) {
                                            // Traditional binding
                                            encoder.set_vertex_buffer(
                                                buffer_index,
                                                Some(&buf.buffer),
                                                *offset,
                                            );
                                            encoder.set_fragment_buffer(
                                                buffer_index,
                                                Some(&buf.buffer),
                                                *offset,
                                            );

                                            if let Some(arg_idx) = buf.arg_buffer_index {
                                                if binding_idx < types::MAX_PUSH_CONSTANT_INDICES {
                                                    bindless_indices.buffer_indices[binding_idx] =
                                                        arg_idx;
                                                    has_bindless_resources = true;
                                                }
                                            }
                                        }
                                    }
                                    BindingState::Texture(tex_handle) => {
                                        if let Some(tex) = self.textures.get(tex_handle) {
                                            encoder.set_fragment_texture(
                                                buffer_index,
                                                Some(&tex.texture),
                                            );

                                            if let Some(arg_idx) = tex.arg_buffer_index {
                                                if binding_idx < types::MAX_PUSH_CONSTANT_INDICES {
                                                    bindless_indices.texture_indices[binding_idx] =
                                                        arg_idx;
                                                    has_bindless_resources = true;
                                                }
                                            }
                                        }
                                    }
                                    BindingState::Sampler(samp_handle) => {
                                        if let Some(samp) = self.samplers.get(samp_handle) {
                                            encoder.set_fragment_sampler_state(
                                                buffer_index,
                                                Some(&samp.sampler),
                                            );

                                            if let Some(arg_idx) = samp.arg_buffer_index {
                                                if binding_idx < types::MAX_PUSH_CONSTANT_INDICES {
                                                    bindless_indices.sampler_indices[binding_idx] =
                                                        arg_idx;
                                                    has_bindless_resources = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Legacy bindless with global argument buffer
                            if bindless_enabled {
                                if let Some(arg_buffer) = argument_buffer {
                                    encoder.set_vertex_buffer(
                                        types::ARGUMENT_BUFFER_SLOT,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    encoder.set_fragment_buffer(
                                        types::ARGUMENT_BUFFER_SLOT,
                                        Some(arg_buffer),
                                        0,
                                    );
                                }
                            }

                            if has_bindless_resources {
                                let indices_bytes: &[u8] = unsafe {
                                    std::slice::from_raw_parts(
                                        &bindless_indices as *const _ as *const u8,
                                        std::mem::size_of::<types::BindlessIndices>(),
                                    )
                                };
                                encoder.set_vertex_bytes(
                                    types::PUSH_CONSTANTS_SLOT,
                                    indices_bytes.len() as u64,
                                    indices_bytes.as_ptr() as *const _,
                                );
                                encoder.set_fragment_bytes(
                                    types::PUSH_CONSTANTS_SLOT,
                                    indices_bytes.len() as u64,
                                    indices_bytes.as_ptr() as *const _,
                                );
                            }
                        }
                    }
                }
                RenderCommand::SetPushConstants { buffers } => {
                    // Check if we should use ParameterBlock-based bindless (for Metal shaders using ParameterBlock)
                    let use_parameter_block = bindless_enabled
                        && current_pipeline
                            .map(|p| !p.parameter_block_layouts.is_empty())
                            .unwrap_or(false);

                    if use_parameter_block {
                        // ParameterBlock-based bindless: write GPU addresses to pipeline's argument buffer
                        if let Some(pipeline) = current_pipeline {
                            if let Some(arg_buffer) = &pipeline.bindless_arg_buffer {
                                // Write each buffer's GPU address at the corresponding field offset
                                for (i, buffer_handle) in buffers.iter().enumerate() {
                                    if let Some(buf) = self.buffers.get(buffer_handle) {
                                        // Get field offset from reflection (field i corresponds to buffer i)
                                        if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                            if let Some(field) = pb_layout.fields.get(i) {
                                                let gpu_addr = buf.buffer.gpu_address();
                                                unsafe {
                                                    let ptr = arg_buffer.contents().add(field.offset);
                                                    *(ptr as *mut u64) = gpu_addr;
                                                }
                                                tracing::trace!(
                                                    "SetPushConstants: Wrote GPU address 0x{:x} at offset {} for field '{}'",
                                                    gpu_addr,
                                                    field.offset,
                                                    field.name
                                                );
                                            }
                                        }
                                    }
                                }

                                // Bind the argument buffer at the ParameterBlock's slot
                                if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                    encoder.set_vertex_buffer(
                                        pb_layout.binding_slot as u64,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    encoder.set_fragment_buffer(
                                        pb_layout.binding_slot as u64,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    tracing::trace!(
                                        "SetPushConstants: Bound ParameterBlock argument buffer at slot {}",
                                        pb_layout.binding_slot
                                    );
                                }
                            }
                        }
                    } else {
                        // Legacy mode: push buffer indices directly via set_*_bytes
                        let mut indices = types::BindlessIndices::default();
                        for (i, buffer_handle) in buffers.iter().enumerate() {
                            if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                            if let Some(buf) = self.buffers.get(buffer_handle) {
                                indices.buffer_indices[i] = buf.arg_buffer_index.unwrap_or(0);
                            }
                        }
                        let indices_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                &indices as *const _ as *const u8,
                                std::mem::size_of::<types::BindlessIndices>(),
                            )
                        };
                        encoder.set_vertex_bytes(
                            types::PUSH_CONSTANTS_SLOT,
                            indices_bytes.len() as u64,
                            indices_bytes.as_ptr() as *const _,
                        );
                        encoder.set_fragment_bytes(
                            types::PUSH_CONSTANTS_SLOT,
                            indices_bytes.len() as u64,
                            indices_bytes.as_ptr() as *const _,
                        );
                    }
                }
                RenderCommand::SetPushConstantsRaw { indices: raw_indices } => {
                    // Check if we should use ParameterBlock-based bindless
                    let use_parameter_block = bindless_enabled
                        && current_pipeline
                            .map(|p| !p.parameter_block_layouts.is_empty())
                            .unwrap_or(false);

                    if use_parameter_block {
                        // ParameterBlock-based bindless: write GPU resource IDs to pipeline's argument buffer
                        if let Some(pipeline) = current_pipeline {
                            if let Some(arg_buffer) = &pipeline.bindless_arg_buffer {
                                // Get the resource registry for reverse lookups
                                let registry = &logical_device.resource_registry;

                                // For each index, determine if it's a texture or sampler and write its GPU resource ID
                                for (i, &idx) in raw_indices.iter().enumerate() {
                                    if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                        if let Some(field) = pb_layout.fields.get(i) {
                                            if registry.is_texture_index(idx) {
                                                // It's a texture - find it and write its GPU resource ID
                                                if let Some(tex_handle) = registry.texture_handle_by_index(idx) {
                                                    if let Some(tex) = self.textures.get(&tex_handle) {
                                                        let resource_id = tex.texture.gpu_resource_id()._impl;
                                                        unsafe {
                                                            let ptr = arg_buffer.contents().add(field.offset);
                                                            *(ptr as *mut u64) = resource_id;
                                                        }
                                                        tracing::trace!(
                                                            "SetPushConstantsRaw: Wrote texture GPU resource ID 0x{:x} at offset {} for field '{}'",
                                                            resource_id, field.offset, field.name
                                                        );
                                                    }
                                                }
                                            } else if registry.is_sampler_index(idx) {
                                                // It's a sampler - find it and write its GPU resource ID
                                                if let Some(samp_handle) = registry.sampler_handle_by_index(idx) {
                                                    if let Some(samp) = self.samplers.get(&samp_handle) {
                                                        let resource_id = samp.sampler.gpu_resource_id()._impl;
                                                        unsafe {
                                                            let ptr = arg_buffer.contents().add(field.offset);
                                                            *(ptr as *mut u64) = resource_id;
                                                        }
                                                        tracing::trace!(
                                                            "SetPushConstantsRaw: Wrote sampler GPU resource ID 0x{:x} at offset {} for field '{}'",
                                                            resource_id, field.offset, field.name
                                                        );
                                                    }
                                                }
                                            } else {
                                                // It's a buffer index - find buffer and write GPU address
                                                for (_buf_handle, buf_state) in &self.buffers {
                                                    if buf_state.arg_buffer_index == Some(idx) {
                                                        let gpu_addr = buf_state.buffer.gpu_address();
                                                        unsafe {
                                                            let ptr = arg_buffer.contents().add(field.offset);
                                                            *(ptr as *mut u64) = gpu_addr;
                                                        }
                                                        tracing::trace!(
                                                            "SetPushConstantsRaw: Wrote buffer GPU address 0x{:x} at offset {} for field '{}'",
                                                            gpu_addr, field.offset, field.name
                                                        );
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // Bind the argument buffer at the ParameterBlock's slot
                                if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                    encoder.set_vertex_buffer(
                                        pb_layout.binding_slot as u64,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    encoder.set_fragment_buffer(
                                        pb_layout.binding_slot as u64,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    tracing::trace!(
                                        "SetPushConstantsRaw: Bound ParameterBlock argument buffer at slot {}",
                                        pb_layout.binding_slot
                                    );
                                }
                            }
                        }
                    } else {
                        // Legacy mode: push raw indices directly via set_*_bytes
                        let mut indices_data = [0u32; types::MAX_PUSH_CONSTANT_INDICES];
                        for (i, &idx) in raw_indices.iter().enumerate() {
                            if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                            indices_data[i] = idx;
                        }
                        let indices_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                indices_data.as_ptr() as *const u8,
                                std::mem::size_of_val(&indices_data),
                            )
                        };
                        encoder.set_vertex_bytes(
                            types::PUSH_CONSTANTS_SLOT,
                            indices_bytes.len() as u64,
                            indices_bytes.as_ptr() as *const _,
                        );
                        encoder.set_fragment_bytes(
                            types::PUSH_CONSTANTS_SLOT,
                            indices_bytes.len() as u64,
                            indices_bytes.as_ptr() as *const _,
                        );
                    }
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    if *first_instance != 0 {
                        tracing::warn!("Metal backend: first_instance != 0 not supported");
                    }
                    encoder.draw_primitives_instanced(
                        current_primitive_type,
                        *first_vertex as u64,
                        *vertex_count as u64,
                        *instance_count as u64,
                    );
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    if *first_instance != 0 || *base_vertex != 0 {
                        tracing::warn!(
                            "Metal backend: first_instance/base_vertex != 0 not supported"
                        );
                    }
                    if let Some((buffer_handle, offset, format)) = current_index_buffer {
                        if let Some(buf) = self.buffers.get(&buffer_handle) {
                            let index_type = index_format_to_mtl(format);
                            let index_offset =
                                offset + (*first_index as u64 * format.size() as u64);
                            encoder.draw_indexed_primitives_instanced(
                                current_primitive_type,
                                *index_count as u64,
                                index_type,
                                &buf.buffer,
                                index_offset,
                                *instance_count as u64,
                            );
                        }
                    }
                }
            }
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Mark as rendered
        if let Some(rt) = self.render_targets.get_mut(&target) {
            rt.has_rendered = true;
        }

        Ok(())
    }

    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()> {
        let render_target = self
            .render_targets
            .get(&target)
            .context("Invalid render target")?;

        if !render_target.has_rendered {
            anyhow::bail!("Cannot read from render target that hasn't been rendered to");
        }

        let logical_device = self
            .devices
            .get(&render_target.device_handle)
            .context("Device no longer valid")?;

        let width = render_target.width;
        let height = render_target.height;
        let bytes_per_pixel = render_target.format.bytes_per_pixel();
        let bytes_per_row = width * bytes_per_pixel;
        let expected_size = (bytes_per_row * height) as usize;

        if output.len() < expected_size {
            anyhow::bail!(
                "Output buffer too small: need {} bytes, got {}",
                expected_size,
                output.len()
            );
        }

        // Create a staging buffer for readback
        let staging_buffer = logical_device
            .device
            .new_buffer(expected_size as u64, MTLResourceOptions::StorageModeShared);

        // Blit texture to buffer
        let command_buffer = logical_device.command_queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();

        blit_encoder.copy_from_texture_to_buffer(
            &render_target.texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
            &staging_buffer,
            0,
            bytes_per_row as u64,
            (bytes_per_row * height) as u64,
            mtl::MTLBlitOption::empty(),
        );

        blit_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy from staging buffer to output
        unsafe {
            let ptr = staging_buffer.contents();
            std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), expected_size);
        }

        Ok(())
    }

    fn create_texture(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Result<TextureHandle> {
        let _span = goldy_span!(
            "resource.texture.create",
            width = width,
            height = height,
            format = ?format
        )
        .entered();

        let logical_device = self
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        let handle = self.next_texture_handle;
        self.next_texture_handle += 1;

        let descriptor = TextureDescriptor::new();
        descriptor.set_width(width as u64);
        descriptor.set_height(height as u64);
        descriptor.set_pixel_format(format_to_mtl(format));

        let mut mtl_usage = MTLTextureUsage::Unknown;
        if usage.contains(TextureUsage::SAMPLED) {
            mtl_usage |= MTLTextureUsage::ShaderRead;
        }
        if usage.contains(TextureUsage::STORAGE) {
            mtl_usage |= MTLTextureUsage::ShaderWrite;
        }
        if usage.contains(TextureUsage::RENDER_TARGET) {
            mtl_usage |= MTLTextureUsage::RenderTarget;
        }
        descriptor.set_usage(mtl_usage);

        // Allocate texture - from heap if bindless, otherwise traditional
        let (texture, is_heap_allocated, arg_buffer_index) = if logical_device.bindless_enabled {
            if let Some(heap) = &logical_device.texture_heap {
                // Use Shared storage to allow CPU writes via replace_region()
                descriptor.set_storage_mode(MTLStorageMode::Shared);

                match heap.new_texture(&descriptor) {
                    Some(texture) => {
                        // Register in bindless registry
                        let index = logical_device.resource_registry.register_texture(handle);
                        tracing::debug!(
                            "Allocated texture {} from heap at bindless index {}",
                            handle,
                            index
                        );

                        // Encode texture into argument buffer using ArgumentEncoder
                        if let (Some(arg_buffer), Some(encoder)) = (
                            &logical_device.argument_buffer,
                            &logical_device.texture_encoder,
                        ) {
                            let encoded_length = encoder.encoded_length();
                            let offset = (index as u64) * encoded_length;

                            if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
                                encoder.set_argument_buffer(arg_buffer, offset);
                                encoder.set_texture(0, &texture);
                                tracing::trace!(
                                    "Encoded texture {} at arg buffer offset {} (slot {})",
                                    handle,
                                    offset,
                                    index
                                );
                            }
                        }

                        // Track heap allocation
                        logical_device.heap_texture_count += 1;

                        (texture, true, Some(index))
                    }
                    None => {
                        // Heap allocation failed, fall back to traditional
                        tracing::warn!(
                            "Heap allocation failed for texture {}, using traditional",
                            handle
                        );
                        descriptor.set_storage_mode(MTLStorageMode::Managed);
                        let texture = logical_device.device.new_texture(&descriptor);
                        (texture, false, None)
                    }
                }
            } else {
                // No heap available
                descriptor.set_storage_mode(MTLStorageMode::Managed);
                let texture = logical_device.device.new_texture(&descriptor);
                (texture, false, None)
            }
        } else {
            // Non-bindless: traditional allocation
            descriptor.set_storage_mode(MTLStorageMode::Managed);
            let texture = logical_device.device.new_texture(&descriptor);
            (texture, false, None)
        };

        self.textures.insert(
            handle,
            TextureState {
                device_handle,
                width,
                height,
                format,
                texture,
                arg_buffer_index,
                is_heap_allocated,
            },
        );

        tracing::debug!(
            "Created texture {} ({}x{}, {:?}) [heap={}]",
            handle,
            width,
            height,
            format,
            is_heap_allocated
        );
        Ok(handle)
    }

    fn write_texture(
        &mut self,
        texture_handle: TextureHandle,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        let texture = self
            .textures
            .get(&texture_handle)
            .context("Invalid texture handle")?;

        let bytes_per_pixel = texture.format.bytes_per_pixel();
        let bytes_per_row = width * bytes_per_pixel;

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
        };

        texture
            .texture
            .replace_region(region, 0, data.as_ptr() as *const _, bytes_per_row as u64);

        tracing::debug!(
            "Wrote {}x{} texture data ({} bytes)",
            width,
            height,
            data.len()
        );
        Ok(())
    }

    fn destroy_texture(&mut self, texture_handle: TextureHandle) {
        if let Some(texture) = self.textures.remove(&texture_handle) {
            // Unregister from bindless registry
            if let Some(device) = self.devices.get_mut(&texture.device_handle) {
                device.resource_registry.unregister_texture(texture_handle);
            }
        }
    }

    fn texture_bindless_index(&self, texture_handle: TextureHandle) -> Option<u32> {
        self.textures.get(&texture_handle).and_then(|t| t.arg_buffer_index)
    }

    fn create_sampler(
        &mut self,
        device_handle: DeviceHandle,
        desc: &crate::types::SamplerDesc,
    ) -> Result<SamplerHandle> {
        let logical_device = self
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        let handle = self.next_sampler_handle;
        self.next_sampler_handle += 1;

        let descriptor = mtl::SamplerDescriptor::new();
        descriptor.set_min_filter(filter_to_mtl(desc.min_filter));
        descriptor.set_mag_filter(filter_to_mtl(desc.mag_filter));
        descriptor.set_mip_filter(mipmap_mode_to_mtl(desc.mipmap_filter));
        descriptor.set_address_mode_s(address_mode_to_mtl(desc.address_mode_u));
        descriptor.set_address_mode_t(address_mode_to_mtl(desc.address_mode_v));
        descriptor.set_address_mode_r(address_mode_to_mtl(desc.address_mode_w));
        descriptor.set_max_anisotropy(desc.max_anisotropy as u64);
        descriptor.set_lod_min_clamp(desc.lod_min_clamp);
        descriptor.set_lod_max_clamp(desc.lod_max_clamp);

        // Enable argument buffer support for bindless
        descriptor.set_support_argument_buffers(true);

        if let Some(compare) = desc.compare {
            descriptor.set_compare_function(compare_to_mtl(compare));
        }

        let sampler = logical_device.device.new_sampler(&descriptor);

        // Register in bindless registry and encode GPU resource ID if enabled
        let arg_buffer_index = if logical_device.bindless_enabled {
            let index = logical_device.resource_registry.register_sampler(handle);
            tracing::debug!("Registered sampler {} at bindless index {}", handle, index);

            // Encode GPU resource ID into argument buffer
            if let Some(arg_buffer) = &logical_device.argument_buffer {
                let offset = (index as u64) * 8;
                if offset + 8 <= ARGUMENT_BUFFER_SIZE {
                    let gpu_id = sampler.gpu_resource_id();
                    unsafe {
                        let ptr = arg_buffer.contents().add(offset as usize) as *mut u64;
                        *ptr = gpu_id._impl;
                    }
                    tracing::trace!(
                        "Encoded sampler {} GPU ID at arg buffer offset {}",
                        handle,
                        offset
                    );
                }
            }

            Some(index)
        } else {
            None
        };

        self.samplers.insert(
            handle,
            SamplerStateInternal {
                device_handle,
                sampler,
                arg_buffer_index,
            },
        );

        tracing::debug!(
            "Created sampler (handle={}) [bindless={}]",
            handle,
            arg_buffer_index.is_some()
        );
        Ok(handle)
    }

    fn destroy_sampler(&mut self, sampler_handle: SamplerHandle) {
        if let Some(sampler) = self.samplers.remove(&sampler_handle) {
            // Unregister from bindless registry
            if let Some(device) = self.devices.get_mut(&sampler.device_handle) {
                device.resource_registry.unregister_sampler(sampler_handle);
            }
        }
    }

    fn sampler_bindless_index(&self, sampler_handle: SamplerHandle) -> Option<u32> {
        self.samplers.get(&sampler_handle).and_then(|s| s.arg_buffer_index)
    }

    fn create_surface(
        &mut self,
        device_handle: DeviceHandle,
        window: &dyn HasWindowHandle,
        _display: &dyn HasDisplayHandle,
    ) -> Result<SurfaceHandle> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {:?}", e))?;

        // Get NSView from window handle
        let ns_view = match window_handle.as_raw() {
            RawWindowHandle::AppKit(handle) => handle.ns_view.as_ptr() as id,
            _ => anyhow::bail!("Expected AppKit window handle on macOS"),
        };

        // Create CAMetalLayer and attach to NSView
        let layer: id = unsafe {
            let layer: id = msg_send![class!(CAMetalLayer), layer];
            let () = msg_send![layer, setDevice: logical_device.device.as_ptr()];
            let () = msg_send![layer, setPixelFormat: MTLPixelFormat::BGRA8Unorm];
            let () = msg_send![layer, setFramebufferOnly: YES];

            // Attach layer to NSView
            let () = msg_send![ns_view, setWantsLayer: YES];
            let () = msg_send![ns_view, setLayer: layer];

            // Get initial size
            let frame: cocoa::foundation::NSRect = msg_send![ns_view, frame];
            let size = CGSize::new(frame.size.width, frame.size.height);
            let () = msg_send![layer, setDrawableSize: size];

            layer
        };

        let handle = self.next_surface_handle;
        self.next_surface_handle += 1;

        self.surfaces.insert(
            handle,
            SurfaceState {
                device_handle,
                width: 800, // Will be updated on first acquire
                height: 600,
                format: TextureFormat::Bgra8Unorm,
                current_frame: 0,
                layer: layer as *mut std::ffi::c_void,
            },
        );

        tracing::info!("Created Metal surface {}", handle);
        Ok(handle)
    }

    fn destroy_surface(&mut self, surface: SurfaceHandle) {
        self.surfaces.remove(&surface);
    }

    fn surface_acquire(&mut self, surface: SurfaceHandle) -> Result<SwapchainImageHandle> {
        let surface_state = self
            .surfaces
            .get_mut(&surface)
            .context("Invalid surface handle")?;

        let layer = surface_state.layer as id;

        // Update size from layer
        let size: CGSize = unsafe { msg_send![layer, drawableSize] };
        surface_state.width = size.width as u32;
        surface_state.height = size.height as u32;

        // Just return a dummy handle - the actual drawable is acquired during render
        surface_state.current_frame = (surface_state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(surface_state.current_frame as u64)
    }

    fn surface_render(
        &mut self,
        surface: SurfaceHandle,
        _image: SwapchainImageHandle,
        commands: &[RenderCommand],
    ) -> Result<()> {
        let surface_state = self
            .surfaces
            .get(&surface)
            .context("Invalid surface handle")?;

        let logical_device = self
            .devices
            .get(&surface_state.device_handle)
            .context("Device no longer valid")?;

        let layer = surface_state.layer as id;

        // Get next drawable
        let drawable: id = unsafe { msg_send![layer, nextDrawable] };
        if drawable == nil {
            anyhow::bail!("Failed to get next drawable");
        }

        // Get texture from drawable - use from_ptr_unchecked to get a reference
        // The drawable owns the texture, we just need a reference without taking ownership
        let texture_ptr: *mut Object = unsafe { msg_send![drawable, texture] };
        let texture: &mtl::TextureRef = unsafe { &*(texture_ptr as *const mtl::TextureRef) };

        // Find clear color from commands
        let mut clear_color = None;
        for cmd in commands {
            if let RenderCommand::Clear(color) = cmd {
                clear_color = Some(*color);
                break;
            }
        }

        // Create render pass
        let render_pass = Self::create_render_pass(texture, None, clear_color, None);

        // Create command buffer and encoder
        let command_buffer = logical_device.command_queue.new_command_buffer();
        let encoder = command_buffer.new_render_command_encoder(render_pass);

        // Set up bindless rendering if enabled
        if logical_device.bindless_enabled {
            // Make heaps resident for the render pass (only if they have resources)
            let render_stages = mtl::MTLRenderStages::Vertex | mtl::MTLRenderStages::Fragment;
            if logical_device.heap_buffer_count > 0 {
                if let Some(buffer_heap) = &logical_device.buffer_heap {
                    encoder.use_heap_at(buffer_heap, render_stages);
                }
            }
            if logical_device.heap_texture_count > 0 {
                if let Some(texture_heap) = &logical_device.texture_heap {
                    encoder.use_heap_at(texture_heap, render_stages);
                }
            }
        }

        // Set viewport and scissor
        encoder.set_viewport(mtl::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: surface_state.width as f64,
            height: surface_state.height as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        encoder.set_scissor_rect(mtl::MTLScissorRect {
            x: 0,
            y: 0,
            width: surface_state.width as u64,
            height: surface_state.height as u64,
        });

        // Cache bindless state for use in loop
        let bindless_enabled = logical_device.bindless_enabled;
        let argument_buffer = logical_device.argument_buffer.as_ref();

        // Process commands (similar to render_to_target)
        let mut current_index_buffer: Option<(BufferHandle, u64, IndexFormat)> = None;
        let mut current_primitive_type = MTLPrimitiveType::Triangle;
        let mut current_pipeline: Option<&PipelineState> = None;

        for cmd in commands {
            match cmd {
                RenderCommand::Clear(_) | RenderCommand::ClearDepth(_) => {}
                RenderCommand::SetPipeline(pipeline_handle) => {
                    if let Some(pipeline) = self.pipelines.get(pipeline_handle) {
                        encoder.set_render_pipeline_state(&pipeline.pipeline);
                        current_primitive_type = pipeline.primitive_type;
                        current_pipeline = Some(pipeline);
                        if let Some(ds) = &pipeline.depth_stencil {
                            encoder.set_depth_stencil_state(ds);
                        }
                    }
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                } => {
                    if let Some(buf) = self.buffers.get(buffer) {
                        encoder.set_vertex_buffer(*slot as u64, Some(&buf.buffer), *offset);
                    }
                }
                RenderCommand::SetIndexBuffer {
                    buffer,
                    offset,
                    format,
                } => {
                    current_index_buffer = Some((*buffer, *offset, *format));
                }
                RenderCommand::SetBindGroup { index, bind_group } => {
                    if let Some(bg) = self.bind_groups.get(bind_group) {
                        // Check if we should use ParameterBlock-based bindless
                        let use_parameter_block = bindless_enabled
                            && current_pipeline
                                .map(|p| !p.parameter_block_layouts.is_empty())
                                .unwrap_or(false);

                        if use_parameter_block {
                            // NEW: ParameterBlock-based bindless rendering
                            // Write GPU addresses directly to the pipeline's argument buffer
                            if let Some(pipeline) = current_pipeline {
                                if let Some(arg_buffer) = &pipeline.bindless_arg_buffer {
                                    // For each binding, find corresponding field and write GPU address
                                    for (binding_idx, binding) in bg.bindings.iter().enumerate() {
                                        match binding {
                                            BindingState::Buffer { buffer, offset, .. } => {
                                                if let Some(buf) = self.buffers.get(buffer) {
                                                    // Find the field offset from reflection
                                                    if let Some(pb_layout) =
                                                        pipeline.parameter_block_layouts.first()
                                                    {
                                                        if let Some(field) =
                                                            pb_layout.fields.get(binding_idx)
                                                        {
                                                            // Write GPU address to argument buffer at field offset
                                                            let gpu_addr =
                                                                buf.buffer.gpu_address() + *offset;
                                                            unsafe {
                                                                let ptr = arg_buffer
                                                                    .contents()
                                                                    .add(field.offset);
                                                                *(ptr as *mut u64) = gpu_addr;
                                                            }
                                                            tracing::trace!(
                                                                "Wrote GPU address 0x{:x} at offset {} for field '{}'",
                                                                gpu_addr,
                                                                field.offset,
                                                                field.name
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                            BindingState::Texture(tex_handle) => {
                                                if let Some(tex) = self.textures.get(tex_handle) {
                                                    if let Some(pb_layout) =
                                                        pipeline.parameter_block_layouts.first()
                                                    {
                                                        if let Some(field) =
                                                            pb_layout.fields.get(binding_idx)
                                                        {
                                                            let resource_id =
                                                                tex.texture.gpu_resource_id()._impl;
                                                            unsafe {
                                                                let ptr = arg_buffer
                                                                    .contents()
                                                                    .add(field.offset);
                                                                *(ptr as *mut u64) = resource_id;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            BindingState::Sampler(samp_handle) => {
                                                if let Some(samp) = self.samplers.get(samp_handle) {
                                                    if let Some(pb_layout) =
                                                        pipeline.parameter_block_layouts.first()
                                                    {
                                                        if let Some(field) =
                                                            pb_layout.fields.get(binding_idx)
                                                        {
                                                            let resource_id = samp
                                                                .sampler
                                                                .gpu_resource_id()
                                                                ._impl;
                                                            unsafe {
                                                                let ptr = arg_buffer
                                                                    .contents()
                                                                    .add(field.offset);
                                                                *(ptr as *mut u64) = resource_id;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Bind the argument buffer at the slot from reflection
                                    if let Some(pb_layout) =
                                        pipeline.parameter_block_layouts.first()
                                    {
                                        encoder.set_vertex_buffer(
                                            pb_layout.binding_slot as u64,
                                            Some(arg_buffer),
                                            0,
                                        );
                                        encoder.set_fragment_buffer(
                                            pb_layout.binding_slot as u64,
                                            Some(arg_buffer),
                                            0,
                                        );
                                        tracing::trace!(
                                            "Bound ParameterBlock argument buffer at slot {}",
                                            pb_layout.binding_slot
                                        );
                                    }
                                }
                            }
                        } else {
                            // LEGACY: Traditional binding or old-style bindless with push constants
                            let mut bindless_indices = types::BindlessIndices::default();
                            let mut has_bindless_resources = false;

                            for (binding_idx, binding) in bg.bindings.iter().enumerate() {
                                let buffer_index = (*index as usize * 16 + binding_idx) as u64;
                                match binding {
                                    BindingState::Buffer { buffer, offset, .. } => {
                                        if let Some(buf) = self.buffers.get(buffer) {
                                            // Traditional binding
                                            encoder.set_vertex_buffer(
                                                buffer_index,
                                                Some(&buf.buffer),
                                                *offset,
                                            );
                                            encoder.set_fragment_buffer(
                                                buffer_index,
                                                Some(&buf.buffer),
                                                *offset,
                                            );

                                            if let Some(arg_idx) = buf.arg_buffer_index {
                                                if binding_idx < types::MAX_PUSH_CONSTANT_INDICES {
                                                    bindless_indices.buffer_indices[binding_idx] =
                                                        arg_idx;
                                                    has_bindless_resources = true;
                                                }
                                            }
                                        }
                                    }
                                    BindingState::Texture(tex_handle) => {
                                        if let Some(tex) = self.textures.get(tex_handle) {
                                            encoder.set_fragment_texture(
                                                buffer_index,
                                                Some(&tex.texture),
                                            );

                                            if let Some(arg_idx) = tex.arg_buffer_index {
                                                if binding_idx < types::MAX_PUSH_CONSTANT_INDICES {
                                                    bindless_indices.texture_indices[binding_idx] =
                                                        arg_idx;
                                                    has_bindless_resources = true;
                                                }
                                            }
                                        }
                                    }
                                    BindingState::Sampler(samp_handle) => {
                                        if let Some(samp) = self.samplers.get(samp_handle) {
                                            encoder.set_fragment_sampler_state(
                                                buffer_index,
                                                Some(&samp.sampler),
                                            );

                                            if let Some(arg_idx) = samp.arg_buffer_index {
                                                if binding_idx < types::MAX_PUSH_CONSTANT_INDICES {
                                                    bindless_indices.sampler_indices[binding_idx] =
                                                        arg_idx;
                                                    has_bindless_resources = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Legacy bindless with global argument buffer
                            if bindless_enabled {
                                if let Some(arg_buffer) = argument_buffer {
                                    encoder.set_vertex_buffer(
                                        types::ARGUMENT_BUFFER_SLOT,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    encoder.set_fragment_buffer(
                                        types::ARGUMENT_BUFFER_SLOT,
                                        Some(arg_buffer),
                                        0,
                                    );
                                }
                            }

                            if has_bindless_resources {
                                let indices_bytes: &[u8] = unsafe {
                                    std::slice::from_raw_parts(
                                        &bindless_indices as *const _ as *const u8,
                                        std::mem::size_of::<types::BindlessIndices>(),
                                    )
                                };
                                encoder.set_vertex_bytes(
                                    types::PUSH_CONSTANTS_SLOT,
                                    indices_bytes.len() as u64,
                                    indices_bytes.as_ptr() as *const _,
                                );
                                encoder.set_fragment_bytes(
                                    types::PUSH_CONSTANTS_SLOT,
                                    indices_bytes.len() as u64,
                                    indices_bytes.as_ptr() as *const _,
                                );
                            }
                        }
                    }
                }
                RenderCommand::SetPushConstants { buffers } => {
                    // Check if we should use ParameterBlock-based bindless (for Metal shaders using ParameterBlock)
                    let use_parameter_block = bindless_enabled
                        && current_pipeline
                            .map(|p| !p.parameter_block_layouts.is_empty())
                            .unwrap_or(false);

                    if use_parameter_block {
                        // ParameterBlock-based bindless: write GPU addresses to pipeline's argument buffer
                        if let Some(pipeline) = current_pipeline {
                            if let Some(arg_buffer) = &pipeline.bindless_arg_buffer {
                                // Write each buffer's GPU address at the corresponding field offset
                                for (i, buffer_handle) in buffers.iter().enumerate() {
                                    if let Some(buf) = self.buffers.get(buffer_handle) {
                                        // Get field offset from reflection (field i corresponds to buffer i)
                                        if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                            if let Some(field) = pb_layout.fields.get(i) {
                                                let gpu_addr = buf.buffer.gpu_address();
                                                unsafe {
                                                    let ptr = arg_buffer.contents().add(field.offset);
                                                    *(ptr as *mut u64) = gpu_addr;
                                                }
                                                tracing::trace!(
                                                    "SetPushConstants: Wrote GPU address 0x{:x} at offset {} for field '{}'",
                                                    gpu_addr,
                                                    field.offset,
                                                    field.name
                                                );
                                            }
                                        }
                                    }
                                }

                                // Bind the argument buffer at the ParameterBlock's slot
                                if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                    encoder.set_vertex_buffer(
                                        pb_layout.binding_slot as u64,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    encoder.set_fragment_buffer(
                                        pb_layout.binding_slot as u64,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    tracing::trace!(
                                        "SetPushConstants: Bound ParameterBlock argument buffer at slot {}",
                                        pb_layout.binding_slot
                                    );
                                }
                            }
                        }
                    } else {
                        // Legacy mode: push buffer indices directly via set_*_bytes
                        let mut indices = types::BindlessIndices::default();
                        for (i, buffer_handle) in buffers.iter().enumerate() {
                            if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                            if let Some(buf) = self.buffers.get(buffer_handle) {
                                indices.buffer_indices[i] = buf.arg_buffer_index.unwrap_or(0);
                            }
                        }
                        let indices_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                &indices as *const _ as *const u8,
                                std::mem::size_of::<types::BindlessIndices>(),
                            )
                        };
                        encoder.set_vertex_bytes(
                            types::PUSH_CONSTANTS_SLOT,
                            indices_bytes.len() as u64,
                            indices_bytes.as_ptr() as *const _,
                        );
                        encoder.set_fragment_bytes(
                            types::PUSH_CONSTANTS_SLOT,
                            indices_bytes.len() as u64,
                            indices_bytes.as_ptr() as *const _,
                        );
                    }
                }
                RenderCommand::SetPushConstantsRaw { indices: raw_indices } => {
                    // Check if we should use ParameterBlock-based bindless
                    let use_parameter_block = bindless_enabled
                        && current_pipeline
                            .map(|p| !p.parameter_block_layouts.is_empty())
                            .unwrap_or(false);

                    if use_parameter_block {
                        // ParameterBlock-based bindless: write GPU resource IDs to pipeline's argument buffer
                        if let Some(pipeline) = current_pipeline {
                            if let Some(arg_buffer) = &pipeline.bindless_arg_buffer {
                                // Get the resource registry for reverse lookups
                                let registry = &logical_device.resource_registry;

                                // For each index, determine if it's a texture or sampler and write its GPU resource ID
                                for (i, &idx) in raw_indices.iter().enumerate() {
                                    if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                        if let Some(field) = pb_layout.fields.get(i) {
                                            if registry.is_texture_index(idx) {
                                                // It's a texture - find it and write its GPU resource ID
                                                if let Some(tex_handle) = registry.texture_handle_by_index(idx) {
                                                    if let Some(tex) = self.textures.get(&tex_handle) {
                                                        let resource_id = tex.texture.gpu_resource_id()._impl;
                                                        unsafe {
                                                            let ptr = arg_buffer.contents().add(field.offset);
                                                            *(ptr as *mut u64) = resource_id;
                                                        }
                                                        tracing::trace!(
                                                            "SetPushConstantsRaw: Wrote texture GPU resource ID 0x{:x} at offset {} for field '{}'",
                                                            resource_id, field.offset, field.name
                                                        );
                                                    }
                                                }
                                            } else if registry.is_sampler_index(idx) {
                                                // It's a sampler - find it and write its GPU resource ID
                                                if let Some(samp_handle) = registry.sampler_handle_by_index(idx) {
                                                    if let Some(samp) = self.samplers.get(&samp_handle) {
                                                        let resource_id = samp.sampler.gpu_resource_id()._impl;
                                                        unsafe {
                                                            let ptr = arg_buffer.contents().add(field.offset);
                                                            *(ptr as *mut u64) = resource_id;
                                                        }
                                                        tracing::trace!(
                                                            "SetPushConstantsRaw: Wrote sampler GPU resource ID 0x{:x} at offset {} for field '{}'",
                                                            resource_id, field.offset, field.name
                                                        );
                                                    }
                                                }
                                            } else {
                                                // It's a buffer index - find buffer and write GPU address
                                                for (_buf_handle, buf_state) in &self.buffers {
                                                    if buf_state.arg_buffer_index == Some(idx) {
                                                        let gpu_addr = buf_state.buffer.gpu_address();
                                                        unsafe {
                                                            let ptr = arg_buffer.contents().add(field.offset);
                                                            *(ptr as *mut u64) = gpu_addr;
                                                        }
                                                        tracing::trace!(
                                                            "SetPushConstantsRaw: Wrote buffer GPU address 0x{:x} at offset {} for field '{}'",
                                                            gpu_addr, field.offset, field.name
                                                        );
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // Bind the argument buffer at the ParameterBlock's slot
                                if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                    encoder.set_vertex_buffer(
                                        pb_layout.binding_slot as u64,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    encoder.set_fragment_buffer(
                                        pb_layout.binding_slot as u64,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    tracing::trace!(
                                        "SetPushConstantsRaw: Bound ParameterBlock argument buffer at slot {}",
                                        pb_layout.binding_slot
                                    );
                                }
                            }
                        }
                    } else {
                        // Legacy mode: push raw indices directly via set_*_bytes
                        let mut indices_data = [0u32; types::MAX_PUSH_CONSTANT_INDICES];
                        for (i, &idx) in raw_indices.iter().enumerate() {
                            if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                            indices_data[i] = idx;
                        }
                        let indices_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                indices_data.as_ptr() as *const u8,
                                std::mem::size_of_val(&indices_data),
                            )
                        };
                        encoder.set_vertex_bytes(
                            types::PUSH_CONSTANTS_SLOT,
                            indices_bytes.len() as u64,
                            indices_bytes.as_ptr() as *const _,
                        );
                        encoder.set_fragment_bytes(
                            types::PUSH_CONSTANTS_SLOT,
                            indices_bytes.len() as u64,
                            indices_bytes.as_ptr() as *const _,
                        );
                    }
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    if *first_instance != 0 {
                        tracing::warn!("Metal backend: first_instance != 0 not supported");
                    }
                    encoder.draw_primitives_instanced(
                        current_primitive_type,
                        *first_vertex as u64,
                        *vertex_count as u64,
                        *instance_count as u64,
                    );
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    if *first_instance != 0 || *base_vertex != 0 {
                        tracing::warn!(
                            "Metal backend: first_instance/base_vertex != 0 not supported"
                        );
                    }
                    if let Some((buffer_handle, offset, format)) = current_index_buffer {
                        if let Some(buf) = self.buffers.get(&buffer_handle) {
                            let index_type = index_format_to_mtl(format);
                            let index_offset =
                                offset + (*first_index as u64 * format.size() as u64);
                            encoder.draw_indexed_primitives_instanced(
                                current_primitive_type,
                                *index_count as u64,
                                index_type,
                                &buf.buffer,
                                index_offset,
                                *instance_count as u64,
                            );
                        }
                    }
                }
            }
        }

        encoder.end_encoding();

        // Present drawable - use msg_send! directly since drawable is autoreleased
        let _: () = unsafe { msg_send![command_buffer.as_ptr(), presentDrawable: drawable] };
        command_buffer.commit();

        Ok(())
    }

    fn surface_present(
        &mut self,
        _surface: SurfaceHandle,
        _image: SwapchainImageHandle,
    ) -> Result<()> {
        // Presentation is handled in surface_render via present_drawable
        Ok(())
    }

    fn surface_resize(&mut self, surface: SurfaceHandle, width: u32, height: u32) -> Result<()> {
        let surface_state = self
            .surfaces
            .get_mut(&surface)
            .context("Invalid surface handle")?;

        surface_state.width = width;
        surface_state.height = height;

        // Update CAMetalLayer drawable size
        let layer = surface_state.layer as id;
        let size = CGSize::new(width as f64, height as f64);
        unsafe {
            let () = msg_send![layer, setDrawableSize: size];
        }

        tracing::debug!("Resized surface {} to {}x{}", surface, width, height);
        Ok(())
    }

    fn surface_size(&self, surface: SurfaceHandle) -> (u32, u32) {
        self.surfaces
            .get(&surface)
            .map(|s| (s.width, s.height))
            .unwrap_or((0, 0))
    }

    fn surface_format(&self, surface: SurfaceHandle) -> TextureFormat {
        self.surfaces
            .get(&surface)
            .map(|s| s.format)
            .unwrap_or(TextureFormat::Bgra8Unorm)
    }

    fn create_compute_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        compute_shader: ShaderHandle,
        _bind_group_layouts: &[BindGroupLayoutHandle],
    ) -> Result<ComputePipelineHandle> {
        // Ensure compute shader is compiled
        self.ensure_compute_shader_compiled(compute_shader)?;

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let shader = self
            .shaders
            .get(&compute_shader)
            .context("Invalid compute shader")?;

        // Parse [numthreads(x, y, z)] from shader source
        let workgroup_size = parse_numthreads(&shader.slang_source).unwrap_or([64, 1, 1]);

        let library = shader.compute_library.as_ref().unwrap();

        // Get compute function - Slang outputs the original function name for MSL
        let function = library
            .get_function("cs_main", None)
            .map_err(|e| anyhow::anyhow!("Failed to get compute function: {}", e))?;

        let pipeline = logical_device
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow::anyhow!("Failed to create compute pipeline: {}", e))?;

        // Extract ParameterBlock layouts from shader reflection for bindless rendering
        let parameter_block_layouts = shader
            .reflection
            .as_ref()
            .map(|r| r.parameter_blocks.clone())
            .unwrap_or_default();

        // Allocate argument buffer for ParameterBlocks if bindless is enabled
        let bindless_arg_buffer = if logical_device.bindless_enabled
            && !parameter_block_layouts.is_empty()
        {
            // Calculate total size needed for all ParameterBlock structs
            let total_size = parameter_block_layouts
                .iter()
                .map(|pb| pb.size as u64)
                .max()
                .unwrap_or(64)
                .max(64); // Minimum 64 bytes for alignment

            let arg_buffer = logical_device
                .device
                .new_buffer(total_size, MTLResourceOptions::StorageModeShared);

            tracing::info!(
                "Allocated bindless argument buffer ({} bytes) for compute pipeline with {} ParameterBlock(s)",
                total_size,
                parameter_block_layouts.len()
            );
            for pb in &parameter_block_layouts {
                tracing::debug!(
                    "  ParameterBlock '{}': slot={}, size={}, fields={}",
                    pb.name,
                    pb.binding_slot,
                    pb.size,
                    pb.fields.len()
                );
            }

            Some(arg_buffer)
        } else {
            None
        };

        let handle = self.next_compute_pipeline_handle;
        self.next_compute_pipeline_handle += 1;

        self.compute_pipelines.insert(
            handle,
            ComputePipelineState {
                device_handle,
                pipeline,
                workgroup_size,
                bindless_arg_buffer,
                parameter_block_layouts,
            },
        );

        tracing::debug!(
            "Created compute pipeline (handle={}, workgroup_size={:?})",
            handle,
            workgroup_size
        );
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline_handle: ComputePipelineHandle) {
        self.compute_pipelines.remove(&pipeline_handle);
    }

    fn dispatch_compute(
        &mut self,
        device_handle: DeviceHandle,
        commands: &[ComputeCommand],
    ) -> Result<()> {
        let _span = goldy_span!("render.compute.dispatch", commands = commands.len()).entered();

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let command_buffer = logical_device.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set up bindless rendering if enabled
        if logical_device.bindless_enabled {
            // Make heaps resident for the compute pass (only if they have resources)
            if logical_device.heap_buffer_count > 0 {
                if let Some(buffer_heap) = &logical_device.buffer_heap {
                    encoder.use_heap(buffer_heap);
                }
            }
            if logical_device.heap_texture_count > 0 {
                if let Some(texture_heap) = &logical_device.texture_heap {
                    encoder.use_heap(texture_heap);
                }
            }
        }

        // Cache bindless state for use in loop
        let bindless_enabled = logical_device.bindless_enabled;
        let argument_buffer = logical_device.argument_buffer.as_ref();

        let mut current_pipeline: Option<&ComputePipelineState> = None;

        for cmd in commands {
            match cmd {
                ComputeCommand::SetPipeline(handle) => {
                    if let Some(pipeline) = self.compute_pipelines.get(handle) {
                        encoder.set_compute_pipeline_state(&pipeline.pipeline);
                        current_pipeline = Some(pipeline);
                    }
                }
                ComputeCommand::SetBindGroup { index, bind_group } => {
                    if let Some(bg) = self.bind_groups.get(bind_group) {
                        // Check if we should use ParameterBlock-based bindless
                        let use_parameter_block = bindless_enabled
                            && current_pipeline
                                .map(|p| !p.parameter_block_layouts.is_empty())
                                .unwrap_or(false);

                        if use_parameter_block {
                            // NEW: ParameterBlock-based bindless rendering
                            // Write GPU addresses directly to the pipeline's argument buffer
                            if let Some(pipeline) = current_pipeline {
                                if let Some(arg_buffer) = &pipeline.bindless_arg_buffer {
                                    // For each binding, find corresponding field and write GPU address
                                    for (binding_idx, binding) in bg.bindings.iter().enumerate() {
                                        match binding {
                                            BindingState::Buffer { buffer, offset, .. } => {
                                                if let Some(buf) = self.buffers.get(buffer) {
                                                    // Find the field offset from reflection
                                                    // Assume bind group index 0 maps to first ParameterBlock
                                                    if let Some(pb_layout) =
                                                        pipeline.parameter_block_layouts.first()
                                                    {
                                                        if let Some(field) =
                                                            pb_layout.fields.get(binding_idx)
                                                        {
                                                            // Write GPU address to argument buffer at field offset
                                                            let gpu_addr =
                                                                buf.buffer.gpu_address() + *offset;
                                                            unsafe {
                                                                let ptr = arg_buffer
                                                                    .contents()
                                                                    .add(field.offset);
                                                                *(ptr as *mut u64) = gpu_addr;
                                                            }
                                                            tracing::trace!(
                                                                "Wrote GPU address 0x{:x} at offset {} for field '{}'",
                                                                gpu_addr,
                                                                field.offset,
                                                                field.name
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                            BindingState::Texture(tex_handle) => {
                                                if let Some(tex) = self.textures.get(tex_handle) {
                                                    // For textures, write the gpuResourceID
                                                    if let Some(pb_layout) =
                                                        pipeline.parameter_block_layouts.first()
                                                    {
                                                        if let Some(field) =
                                                            pb_layout.fields.get(binding_idx)
                                                        {
                                                            let resource_id =
                                                                tex.texture.gpu_resource_id()._impl;
                                                            unsafe {
                                                                let ptr = arg_buffer
                                                                    .contents()
                                                                    .add(field.offset);
                                                                *(ptr as *mut u64) = resource_id;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            BindingState::Sampler(samp_handle) => {
                                                if let Some(samp) = self.samplers.get(samp_handle) {
                                                    // For samplers, write the gpuResourceID
                                                    if let Some(pb_layout) =
                                                        pipeline.parameter_block_layouts.first()
                                                    {
                                                        if let Some(field) =
                                                            pb_layout.fields.get(binding_idx)
                                                        {
                                                            let resource_id = samp
                                                                .sampler
                                                                .gpu_resource_id()
                                                                ._impl;
                                                            unsafe {
                                                                let ptr = arg_buffer
                                                                    .contents()
                                                                    .add(field.offset);
                                                                *(ptr as *mut u64) = resource_id;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Bind the argument buffer at the slot from reflection
                                    if let Some(pb_layout) =
                                        pipeline.parameter_block_layouts.first()
                                    {
                                        encoder.set_buffer(
                                            pb_layout.binding_slot as u64,
                                            Some(arg_buffer),
                                            0,
                                        );
                                        tracing::trace!(
                                            "Bound ParameterBlock argument buffer at slot {}",
                                            pb_layout.binding_slot
                                        );
                                    }
                                }
                            }
                        } else {
                            // LEGACY: Traditional binding or old-style bindless with push constants
                            let mut bindless_indices = types::BindlessIndices::default();
                            let mut has_bindless_resources = false;

                            for (binding_idx, binding) in bg.bindings.iter().enumerate() {
                                let buffer_index = (*index as usize * 16 + binding_idx) as u64;
                                match binding {
                                    BindingState::Buffer { buffer, offset, .. } => {
                                        if let Some(buf) = self.buffers.get(buffer) {
                                            // Traditional binding
                                            encoder.set_buffer(
                                                buffer_index,
                                                Some(&buf.buffer),
                                                *offset,
                                            );

                                            // Collect bindless index if available (legacy path)
                                            if let Some(arg_idx) = buf.arg_buffer_index {
                                                if binding_idx < types::MAX_PUSH_CONSTANT_INDICES {
                                                    bindless_indices.buffer_indices[binding_idx] =
                                                        arg_idx;
                                                    has_bindless_resources = true;
                                                }
                                            }
                                        }
                                    }
                                    BindingState::Texture(tex_handle) => {
                                        if let Some(tex) = self.textures.get(tex_handle) {
                                            encoder.set_texture(buffer_index, Some(&tex.texture));

                                            if let Some(arg_idx) = tex.arg_buffer_index {
                                                if binding_idx < types::MAX_PUSH_CONSTANT_INDICES {
                                                    bindless_indices.texture_indices[binding_idx] =
                                                        arg_idx;
                                                    has_bindless_resources = true;
                                                }
                                            }
                                        }
                                    }
                                    BindingState::Sampler(samp_handle) => {
                                        if let Some(samp) = self.samplers.get(samp_handle) {
                                            encoder.set_sampler_state(
                                                buffer_index,
                                                Some(&samp.sampler),
                                            );

                                            if let Some(arg_idx) = samp.arg_buffer_index {
                                                if binding_idx < types::MAX_PUSH_CONSTANT_INDICES {
                                                    bindless_indices.sampler_indices[binding_idx] =
                                                        arg_idx;
                                                    has_bindless_resources = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Legacy bindless with global argument buffer
                            if bindless_enabled {
                                if let Some(arg_buffer) = argument_buffer {
                                    encoder.set_buffer(
                                        types::ARGUMENT_BUFFER_SLOT,
                                        Some(arg_buffer),
                                        0,
                                    );
                                }
                            }

                            if has_bindless_resources {
                                let indices_bytes: &[u8] = unsafe {
                                    std::slice::from_raw_parts(
                                        &bindless_indices as *const _ as *const u8,
                                        std::mem::size_of::<types::BindlessIndices>(),
                                    )
                                };
                                encoder.set_bytes(
                                    types::PUSH_CONSTANTS_SLOT,
                                    indices_bytes.len() as u64,
                                    indices_bytes.as_ptr() as *const _,
                                );
                            }
                        }
                    }
                }
                ComputeCommand::SetPushConstants { buffers } => {
                    // Check if we should use ParameterBlock-based bindless (for Metal shaders using ParameterBlock)
                    let use_parameter_block = bindless_enabled
                        && current_pipeline
                            .map(|p| !p.parameter_block_layouts.is_empty())
                            .unwrap_or(false);

                    if use_parameter_block {
                        // ParameterBlock-based bindless: write GPU addresses to pipeline's argument buffer
                        if let Some(pipeline) = current_pipeline {
                            if let Some(arg_buffer) = &pipeline.bindless_arg_buffer {
                                // Write each buffer's GPU address at the corresponding field offset
                                for (i, buffer_handle) in buffers.iter().enumerate() {
                                    if let Some(buf) = self.buffers.get(buffer_handle) {
                                        // Get field offset from reflection (field i corresponds to buffer i)
                                        if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                            if let Some(field) = pb_layout.fields.get(i) {
                                                let gpu_addr = buf.buffer.gpu_address();
                                                unsafe {
                                                    let ptr = arg_buffer.contents().add(field.offset);
                                                    *(ptr as *mut u64) = gpu_addr;
                                                }
                                                tracing::trace!(
                                                    "SetPushConstants (compute): Wrote GPU address 0x{:x} at offset {} for field '{}'",
                                                    gpu_addr,
                                                    field.offset,
                                                    field.name
                                                );
                                            }
                                        }
                                    }
                                }

                                // Bind the argument buffer at the ParameterBlock's slot
                                if let Some(pb_layout) = pipeline.parameter_block_layouts.first() {
                                    encoder.set_buffer(
                                        pb_layout.binding_slot as u64,
                                        Some(arg_buffer),
                                        0,
                                    );
                                    tracing::trace!(
                                        "SetPushConstants (compute): Bound ParameterBlock argument buffer at slot {}",
                                        pb_layout.binding_slot
                                    );
                                }
                            }
                        }
                    } else {
                        // Legacy mode: push buffer indices directly via set_bytes
                        let mut indices = types::BindlessIndices::default();
                        for (i, buffer_handle) in buffers.iter().enumerate() {
                            if i >= types::MAX_PUSH_CONSTANT_INDICES { break; }
                            if let Some(buf) = self.buffers.get(buffer_handle) {
                                indices.buffer_indices[i] = buf.arg_buffer_index.unwrap_or(0);
                            }
                        }
                        let indices_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                &indices as *const _ as *const u8,
                                std::mem::size_of::<types::BindlessIndices>(),
                            )
                        };
                        encoder.set_bytes(
                            types::PUSH_CONSTANTS_SLOT,
                            indices_bytes.len() as u64,
                            indices_bytes.as_ptr() as *const _,
                        );
                    }
                }
                ComputeCommand::Dispatch {
                    workgroups_x,
                    workgroups_y,
                    workgroups_z,
                } => {
                    if let Some(pipeline) = current_pipeline {
                        let threads_per_group = MTLSize {
                            width: pipeline.workgroup_size[0] as u64,
                            height: pipeline.workgroup_size[1] as u64,
                            depth: pipeline.workgroup_size[2] as u64,
                        };
                        let threadgroups = MTLSize {
                            width: *workgroups_x as u64,
                            height: *workgroups_y as u64,
                            depth: *workgroups_z as u64,
                        };
                        encoder.dispatch_thread_groups(threadgroups, threads_per_group);
                    }
                }
            }
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_backend_creation() {
        let backend = MetalBackend::new();
        assert!(
            backend.is_ok(),
            "Failed to create Metal backend: {:?}",
            backend.err()
        );
        let backend = backend.unwrap();
        assert_eq!(backend.backend_type(), BackendType::Metal);
    }

    #[test]
    fn test_metal_adapters() {
        let backend = MetalBackend::new().unwrap();
        let adapters = backend.enumerate_adapters();
        assert!(!adapters.is_empty(), "No Metal adapters found");
        for adapter in &adapters {
            println!("Adapter: {} ({})", adapter.name, adapter.vendor);
        }
    }

    #[test]
    fn test_metal_device_creation() {
        let mut backend = MetalBackend::new().unwrap();
        let device = backend.create_device(0);
        assert!(
            device.is_ok(),
            "Failed to create Metal device: {:?}",
            device.err()
        );
        let device = device.unwrap();
        assert!(backend.is_device_valid(device));
        backend.destroy_device(device);
        assert!(!backend.is_device_valid(device));
    }

    #[test]
    fn test_metal_buffer_operations() {
        let mut backend = MetalBackend::new().unwrap();
        let device = backend.create_device(0).unwrap();

        let buffer = backend
            .create_buffer(device, 256, BufferUsage::VERTEX | BufferUsage::COPY_DST)
            .unwrap();

        assert_eq!(backend.buffer_size(buffer), 256);

        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        backend.write_buffer(buffer, 0, &data).unwrap();

        backend.destroy_buffer(buffer);
        backend.destroy_device(device);
    }
}
