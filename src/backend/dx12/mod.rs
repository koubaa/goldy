//! DirectX 12 backend implementation.
//!
//! Targets D3D12 Feature Level 12.0+ on Windows.
//! Uses Slang for shader compilation (Slang -> HLSL -> DXIL via DXC).
//!
//! ## Module Structure
//!
//! - `types`: Internal state structs for devices, buffers, shaders, etc.
//! - `utils`: Format conversion and helpers

mod types;
mod utils;

use types::{
    BindGroupLayoutState, BindGroupState, BufferState, ComputePipelineState, DxgiAdapterInfo,
    FrameSync, LogicalDevice, PipelineState, RenderTargetState, SamplerState, ShaderState,
    SurfaceState, TextureState, MAX_FRAMES_IN_FLIGHT,
};
use utils::{
    depth_format_to_dxgi, dxgi_to_format, format_to_dxgi, index_format_to_dxgi,
    topology_type_to_d3d12, vertex_format_to_dxgi,
};

use super::*;
use crate::types::Color;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::ffi::CString;
use windows::{
    core::Interface,
    Win32::{
        Foundation::{CloseHandle, HWND},
        Graphics::{
            Direct3D::Fxc::{D3DCompile, D3DCOMPILE_OPTIMIZATION_LEVEL3},
            Direct3D::*,
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
        System::Threading::{CreateEventA, WaitForSingleObject, INFINITE},
        UI::WindowsAndMessaging::GetClientRect,
    },
};

use raw_window_handle::RawWindowHandle;

/// DirectX 12 backend.
pub struct Dx12Backend {
    factory: IDXGIFactory4,
    adapters: Vec<DxgiAdapterInfo>,
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
    samplers: HashMap<SamplerHandle, SamplerState>,
    next_sampler_handle: SamplerHandle,
    /// Next RTV descriptor offset
    next_rtv_offset: u32,
    /// Next DSV descriptor offset
    next_dsv_offset: u32,
    /// Next SRV descriptor offset in CBV/SRV/UAV heap
    next_srv_offset: u32,
    /// Next sampler descriptor offset
    next_sampler_offset: u32,
    /// Per-backend Slang compiler instance
    slang_compiler: crate::slang::SlangCompiler,
}

impl Dx12Backend {
    /// Create a new DX12 backend.
    pub fn new() -> Result<Self> {
        tracing::info!("Initializing DX12 backend");

        // Enable debug layer in debug builds
        #[cfg(debug_assertions)]
        {
            let mut debug: Option<ID3D12Debug> = None;
            if unsafe { D3D12GetDebugInterface(&mut debug) }.is_ok() {
                if let Some(debug) = debug {
                    unsafe { debug.EnableDebugLayer() };
                    tracing::info!("D3D12 debug layer enabled");
                }
            }
        }

        // Create DXGI factory
        let factory_flags = if cfg!(debug_assertions) {
            DXGI_CREATE_FACTORY_DEBUG
        } else {
            DXGI_CREATE_FACTORY_FLAGS(0)
        };

        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(factory_flags) }
            .context("Failed to create DXGI factory")?;

        // Enumerate adapters
        let mut adapters = Vec::new();
        let mut adapter_index = 0u32;

        loop {
            let adapter_result: Result<IDXGIAdapter1, _> =
                unsafe { factory.EnumAdapters1(adapter_index) };
            match adapter_result {
                Ok(adapter) => {
                    let desc = match unsafe { adapter.GetDesc1() } {
                        Ok(d) => d,
                        Err(_) => continue,
                    };

                    // Skip software adapters unless explicitly requested
                    let flags = DXGI_ADAPTER_FLAG(desc.Flags as i32);
                    if !flags.contains(DXGI_ADAPTER_FLAG_SOFTWARE) {
                        let name = String::from_utf16_lossy(&desc.Description)
                            .trim_end_matches('\0')
                            .to_string();
                        tracing::info!("  [{}] {}", adapter_index, name);

                        adapters.push(DxgiAdapterInfo {
                            adapter,
                            desc,
                            adapter_id: adapter_index,
                        });
                    }
                    adapter_index += 1;
                }
                Err(_) => break,
            }
        }

        tracing::info!("Found {} DX12 adapters", adapters.len());

        // Create Slang compiler
        let slang_compiler =
            crate::slang::SlangCompiler::new().context("Failed to create Slang compiler")?;

        Ok(Self {
            factory,
            adapters,
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
            next_rtv_offset: 0,
            next_dsv_offset: 0,
            next_srv_offset: 0,
            next_sampler_offset: 0,
            slang_compiler,
        })
    }

    /// Wait for the GPU to finish all work on a device.
    fn wait_for_gpu(&self, device: &LogicalDevice) -> Result<()> {
        let fence_value = device.fence_value;
        unsafe { device.command_queue.Signal(&device.fence, fence_value) }
            .context("Failed to signal fence")?;

        if unsafe { device.fence.GetCompletedValue() } < fence_value {
            let event = unsafe { CreateEventA(None, false, false, None) }
                .context("Failed to create event")?;

            unsafe { device.fence.SetEventOnCompletion(fence_value, event) }
                .context("Failed to set event on completion")?;

            unsafe { WaitForSingleObject(event, INFINITE) };
            unsafe { CloseHandle(event) }.ok();
        }

        Ok(())
    }

    /// Compile a shader for a specific stage on demand.
    fn ensure_shader_stage_compiled(
        &mut self,
        shader_handle: ShaderHandle,
        stage: crate::slang::SlangStage,
    ) -> Result<Vec<u8>> {
        let shader = self
            .shaders
            .get_mut(&shader_handle)
            .context("Invalid shader handle")?;

        // Check if already compiled for this stage
        let cached_bytecode = match stage {
            crate::slang::SlangStage::Vertex => shader.vertex_bytecode.clone(),
            crate::slang::SlangStage::Fragment => shader.fragment_bytecode.clone(),
            crate::slang::SlangStage::Compute => shader.compute_bytecode.clone(),
            _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
        };

        if let Some(bytecode) = cached_bytecode {
            return Ok(bytecode);
        }

        // Get the entry point name based on stage
        let entry_point_name = match stage {
            crate::slang::SlangStage::Vertex => "vs_main",
            crate::slang::SlangStage::Fragment => "fs_main",
            crate::slang::SlangStage::Compute => "cs_main",
            _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
        };

        // Clone source to avoid borrow issues
        let slang_source = shader.slang_source.clone();

        // Compile Slang to HLSL first
        let hlsl_compiled = self
            .slang_compiler
            .compile_entry_point(
                &slang_source,
                crate::slang::ShaderTarget::Hlsl,
                Some((entry_point_name, stage)),
            )
            .with_context(|| format!("Failed to compile {} shader to HLSL", entry_point_name))?;

        let hlsl_source = hlsl_compiled
            .as_str()
            .context("Invalid HLSL output")?
            .to_string();

        // Compile HLSL to DXIL using DXC (for now we'll use FXC as fallback)
        let target_profile = match stage {
            crate::slang::SlangStage::Vertex => c"vs_5_0",
            crate::slang::SlangStage::Fragment => c"ps_5_0",
            crate::slang::SlangStage::Compute => c"cs_5_0",
            _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
        };

        // Use D3DCompile for shader compilation
        // Slang preserves the original entry point names in generated HLSL
        let mut shader_blob: Option<ID3DBlob> = None;
        let mut error_blob: Option<ID3DBlob> = None;

        let entry_point_cstr = CString::new(entry_point_name).unwrap();

        let result = unsafe {
            D3DCompile(
                hlsl_source.as_ptr() as *const _,
                hlsl_source.len(),
                None,
                None,
                None,
                windows::core::PCSTR(entry_point_cstr.as_ptr() as *const u8),
                windows::core::PCSTR(target_profile.as_ptr() as *const u8),
                D3DCOMPILE_OPTIMIZATION_LEVEL3,
                0,
                &mut shader_blob,
                Some(&mut error_blob),
            )
        };

        // Debug: log the HLSL source if compilation fails
        if result.is_err() {
            tracing::debug!("HLSL source that failed to compile:\n{}", hlsl_source);
        }

        if result.is_err() {
            if let Some(error) = error_blob {
                let error_ptr = unsafe { error.GetBufferPointer() } as *const u8;
                let error_size = unsafe { error.GetBufferSize() };
                let error_msg = unsafe { std::slice::from_raw_parts(error_ptr, error_size) };
                let error_str = String::from_utf8_lossy(error_msg);
                anyhow::bail!("Shader compilation failed: {}", error_str);
            }
            anyhow::bail!("Shader compilation failed with unknown error");
        }

        let blob = shader_blob.context("Shader compilation produced no output")?;
        let bytecode_ptr = unsafe { blob.GetBufferPointer() } as *const u8;
        let bytecode_size = unsafe { blob.GetBufferSize() };
        let bytecode = unsafe { std::slice::from_raw_parts(bytecode_ptr, bytecode_size) }.to_vec();

        tracing::debug!("Compiled {} ({} bytes)", entry_point_name, bytecode.len());

        // Cache the bytecode
        let shader = self.shaders.get_mut(&shader_handle).unwrap();
        match stage {
            crate::slang::SlangStage::Vertex => shader.vertex_bytecode = Some(bytecode.clone()),
            crate::slang::SlangStage::Fragment => shader.fragment_bytecode = Some(bytecode.clone()),
            crate::slang::SlangStage::Compute => shader.compute_bytecode = Some(bytecode.clone()),
            _ => {} // Already validated above
        }

        Ok(bytecode)
    }
}

impl Drop for Dx12Backend {
    fn drop(&mut self) {
        tracing::info!("Shutting down DX12 backend");

        // Wait for all devices to finish and destroy them
        let device_handles: Vec<_> = self.devices.keys().copied().collect();
        for handle in device_handles {
            self.destroy_device(handle);
        }
    }
}

impl GpuBackend for Dx12Backend {
    fn backend_type(&self) -> BackendType {
        BackendType::Dx12
    }

    fn enumerate_adapters(&self) -> Vec<super::AdapterInfo> {
        self.adapters
            .iter()
            .map(|adapter| {
                let name = String::from_utf16_lossy(&adapter.desc.Description)
                    .trim_end_matches('\0')
                    .to_string();
                let flags = DXGI_ADAPTER_FLAG(adapter.desc.Flags as i32);
                let device_type = utils::device_type_from_flags(flags);
                let vendor = utils::vendor_name(adapter.desc.VendorId);

                super::AdapterInfo {
                    id: adapter.adapter_id,
                    name,
                    vendor: vendor.to_string(),
                    backend: BackendType::Dx12,
                    device_type,
                }
            })
            .collect()
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        let adapter = self
            .adapters
            .iter()
            .find(|a| a.adapter_id == adapter_id)
            .context("Invalid adapter ID")?;

        // Create D3D12 device
        let mut device: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(&adapter.adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }
            .context("Failed to create D3D12 device")?;

        let device = device.context("D3D12CreateDevice returned null")?;

        // Create command queue
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };

        let command_queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }
            .context("Failed to create command queue")?;

        // Create command allocator
        let command_allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                .context("Failed to create command allocator")?;

        // Create RTV descriptor heap
        let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 256, // Should be enough for most cases
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };

        let rtv_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&rtv_heap_desc) }
            .context("Failed to create RTV heap")?;

        let rtv_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };

        // Create DSV descriptor heap
        let dsv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            NumDescriptors: 256,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };

        let dsv_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&dsv_heap_desc) }
            .context("Failed to create DSV heap")?;

        let dsv_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) };

        // Create CBV/SRV/UAV descriptor heap
        let cbv_srv_uav_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1024,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };

        let cbv_srv_uav_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&cbv_srv_uav_heap_desc) }
                .context("Failed to create CBV/SRV/UAV heap")?;

        let cbv_srv_uav_descriptor_size = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        // Create sampler descriptor heap
        let sampler_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: 256,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };

        let sampler_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&sampler_heap_desc) }
                .context("Failed to create sampler heap")?;

        let sampler_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) };

        // Create fence
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .context("Failed to create fence")?;

        let handle = self.next_device_handle;
        self.next_device_handle += 1;

        self.devices.insert(
            handle,
            LogicalDevice {
                device,
                adapter_id,
                command_queue,
                command_allocator,
                rtv_heap,
                rtv_descriptor_size,
                dsv_heap,
                dsv_descriptor_size,
                cbv_srv_uav_heap,
                cbv_srv_uav_descriptor_size,
                sampler_heap,
                sampler_descriptor_size,
                fence,
                fence_value: 1,
            },
        );

        tracing::info!("Created DX12 device {} for adapter {}", handle, adapter_id);
        Ok(handle)
    }

    fn destroy_device(&mut self, device_handle: DeviceHandle) {
        if let Some(logical_device) = self.devices.remove(&device_handle) {
            // Wait for GPU to finish
            let _ = self.wait_for_gpu(&logical_device);

            // Destroy buffers owned by this device
            let buffer_handles: Vec<_> = self
                .buffers
                .iter()
                .filter(|(_, b)| b.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in buffer_handles {
                self.buffers.remove(&handle);
            }

            // Destroy shaders owned by this device
            let shader_handles: Vec<_> = self
                .shaders
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in shader_handles {
                self.shaders.remove(&handle);
            }

            // Destroy pipelines owned by this device
            let pipeline_handles: Vec<_> = self
                .pipelines
                .iter()
                .filter(|(_, p)| p.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in pipeline_handles {
                self.pipelines.remove(&handle);
            }

            // Destroy render targets owned by this device
            let target_handles: Vec<_> = self
                .render_targets
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in target_handles {
                self.render_targets.remove(&handle);
            }

            // Destroy surfaces owned by this device
            let surface_handles: Vec<_> = self
                .surfaces
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in surface_handles {
                self.surfaces.remove(&handle);
            }

            // Destroy textures owned by this device
            let texture_handles: Vec<_> = self
                .textures
                .iter()
                .filter(|(_, t)| t.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in texture_handles {
                self.textures.remove(&handle);
            }

            // Destroy samplers owned by this device
            let sampler_handles: Vec<_> = self
                .samplers
                .iter()
                .filter(|(_, s)| s.device_handle == device_handle)
                .map(|(h, _)| *h)
                .collect();
            for handle in sampler_handles {
                self.samplers.remove(&handle);
            }

            tracing::info!("Destroyed DX12 device {}", device_handle);
        }
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn create_buffer(
        &mut self,
        device_handle: DeviceHandle,
        size: u64,
        usage: BufferUsage,
    ) -> Result<BufferHandle> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // For simplicity, all buffers are created in UPLOAD heap to allow CPU writes.
        // This matches the high-level Goldy API which assumes all buffers can be written to.
        // A more sophisticated implementation would use DEFAULT heap for GPU-only buffers
        // with staging buffer copies, but that's an optimization for later.
        let heap_type = D3D12_HEAP_TYPE_UPLOAD;
        let _ = usage; // We use all as CPU-writable for now

        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };

        let initial_state = if heap_type == D3D12_HEAP_TYPE_UPLOAD {
            D3D12_RESOURCE_STATE_GENERIC_READ
        } else {
            D3D12_RESOURCE_STATE_COMMON
        };

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            logical_device.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                initial_state,
                None,
                &mut resource,
            )
        }
        .context("Failed to create buffer resource")?;

        let resource = resource.context("CreateCommittedResource returned null")?;

        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;

        self.buffers.insert(
            handle,
            BufferState {
                device_handle,
                resource,
                size,
            },
        );

        Ok(handle)
    }

    fn destroy_buffer(&mut self, buffer_handle: BufferHandle) {
        self.buffers.remove(&buffer_handle);
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

        // Map the buffer
        let mut mapped_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let read_range = D3D12_RANGE { Begin: 0, End: 0 }; // We're only writing

        unsafe {
            buffer
                .resource
                .Map(0, Some(&read_range), Some(&mut mapped_ptr))
        }
        .context("Failed to map buffer")?;

        // Copy data
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (mapped_ptr as *mut u8).add(offset as usize),
                data.len(),
            );
        }

        // Unmap
        let written_range = D3D12_RANGE {
            Begin: offset as usize,
            End: (offset as usize) + data.len(),
        };
        unsafe { buffer.resource.Unmap(0, Some(&written_range)) };

        Ok(())
    }

    fn buffer_size(&self, buffer_handle: BufferHandle) -> u64 {
        self.buffers
            .get(&buffer_handle)
            .map(|b| b.size)
            .unwrap_or(0)
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
        _search_paths: &[&str],
    ) -> Result<ShaderHandle> {
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
                vertex_bytecode: None,
                fragment_bytecode: None,
                compute_bytecode: None,
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
        layout_handle: BindGroupLayoutHandle,
        entries: &[BindGroupEntry],
    ) -> Result<BindGroupHandle> {
        let _ = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let _ = self
            .bind_group_layouts
            .get(&layout_handle)
            .context("Invalid bind group layout handle")?;

        let mut buffer_bindings = Vec::new();
        for entry in entries {
            match &entry.resource {
                BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                } => {
                    buffer_bindings.push((entry.binding, *buffer, *offset, *size));
                }
                BindingResource::Texture(_) | BindingResource::Sampler(_) => {
                    // TODO: Handle texture and sampler bindings
                    tracing::warn!("DX12 texture/sampler bindings not fully implemented");
                }
            }
        }

        let handle = self.next_bind_group_handle;
        self.next_bind_group_handle += 1;

        self.bind_groups.insert(
            handle,
            BindGroupState {
                device_handle,
                layout_handle,
                buffer_bindings,
            },
        );

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
        // Compile shaders on-demand
        let vs_bytecode =
            self.ensure_shader_stage_compiled(vertex_shader, crate::slang::SlangStage::Vertex)?;
        let fs_bytecode =
            self.ensure_shader_stage_compiled(fragment_shader, crate::slang::SlangStage::Fragment)?;

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Create root signature
        let root_signature = if bind_group_layouts.is_empty() {
            // Empty root signature
            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 0,
                pParameters: std::ptr::null(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            };

            let mut signature_blob: Option<ID3DBlob> = None;
            let mut error_blob: Option<ID3DBlob> = None;

            unsafe {
                D3D12SerializeRootSignature(
                    &desc,
                    D3D_ROOT_SIGNATURE_VERSION_1,
                    &mut signature_blob,
                    Some(&mut error_blob),
                )
            }
            .context("Failed to serialize root signature")?;

            let blob = signature_blob.context("Root signature serialization produced no output")?;
            let signature: ID3D12RootSignature = unsafe {
                logical_device.device.CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    ),
                )
            }
            .context("Failed to create root signature")?;

            signature
        } else {
            // Create root signature with proper descriptor types for each binding
            let mut root_params: Vec<D3D12_ROOT_PARAMETER> = Vec::new();

            // Track register indices separately for each register space
            let mut srv_register = 0u32;
            let mut uav_register = 0u32;
            let mut cbv_register = 0u32;

            for layout_handle in bind_group_layouts.iter() {
                if let Some(layout) = self.bind_group_layouts.get(layout_handle) {
                    for entry in &layout.entries {
                        let (param_type, register) = match &entry.ty {
                            BindingType::StorageBuffer { read_only: true } => {
                                let reg = srv_register;
                                srv_register += 1;
                                (D3D12_ROOT_PARAMETER_TYPE_SRV, reg)
                            }
                            BindingType::StorageBuffer { read_only: false } => {
                                let reg = uav_register;
                                uav_register += 1;
                                (D3D12_ROOT_PARAMETER_TYPE_UAV, reg)
                            }
                            BindingType::UniformBuffer => {
                                let reg = cbv_register;
                                cbv_register += 1;
                                (D3D12_ROOT_PARAMETER_TYPE_CBV, reg)
                            }
                            _ => {
                                tracing::warn!("Unsupported binding type in graphics pipeline");
                                continue;
                            }
                        };

                        root_params.push(D3D12_ROOT_PARAMETER {
                            ParameterType: param_type,
                            Anonymous: D3D12_ROOT_PARAMETER_0 {
                                Descriptor: D3D12_ROOT_DESCRIPTOR {
                                    ShaderRegister: register,
                                    RegisterSpace: 0,
                                },
                            },
                            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                        });
                    }
                }
            }

            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: root_params.len() as u32,
                pParameters: if root_params.is_empty() {
                    std::ptr::null()
                } else {
                    root_params.as_ptr()
                },
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            };

            let mut signature_blob: Option<ID3DBlob> = None;
            let mut error_blob: Option<ID3DBlob> = None;

            unsafe {
                D3D12SerializeRootSignature(
                    &desc,
                    D3D_ROOT_SIGNATURE_VERSION_1,
                    &mut signature_blob,
                    Some(&mut error_blob),
                )
            }
            .context("Failed to serialize root signature")?;

            let blob = signature_blob.context("Root signature serialization produced no output")?;
            let signature: ID3D12RootSignature = unsafe {
                logical_device.device.CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    ),
                )
            }
            .context("Failed to create root signature")?;

            signature
        };

        // Build input layout
        // We use semantic conventions based on location and format:
        // - location 0 → POSITION (expected for all shaders)
        // - location 1 with 3-4 components → COLOR (for colored vertex shaders)
        // - location 1 with 1-2 components → TEXCOORD0 (for textured shaders)
        // - location 2+ → TEXCOORDn
        let mut texcoord_index = 0u32;
        let input_elements: Vec<D3D12_INPUT_ELEMENT_DESC> = vertex_layout
            .attributes
            .iter()
            .map(|attr| {
                let (semantic_name, semantic_index) = if attr.location == 0 {
                    (c"POSITION".as_ptr() as *const u8, 0)
                } else {
                    // Determine semantic based on format
                    // 3-4 component formats at location 1 are likely COLOR
                    // 1-2 component formats are likely TEXCOORD
                    let is_color = match attr.format {
                        crate::types::VertexFormat::Float32x3
                        | crate::types::VertexFormat::Float32x4
                        | crate::types::VertexFormat::Unorm8x4
                        | crate::types::VertexFormat::Uint8x4 => attr.location == 1,
                        _ => false,
                    };

                    if is_color {
                        (c"COLOR".as_ptr() as *const u8, 0)
                    } else {
                        let idx = texcoord_index;
                        texcoord_index += 1;
                        (c"TEXCOORD".as_ptr() as *const u8, idx)
                    }
                };
                D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: windows::core::PCSTR(semantic_name),
                    SemanticIndex: semantic_index,
                    Format: vertex_format_to_dxgi(attr.format),
                    InputSlot: 0,
                    AlignedByteOffset: attr.offset,
                    InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    InstanceDataStepRate: 0,
                }
            })
            .collect();

        // Create PSO
        let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs_bytecode.as_ptr() as *const _,
                BytecodeLength: vs_bytecode.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: fs_bytecode.as_ptr() as *const _,
                BytecodeLength: fs_bytecode.len(),
            },
            BlendState: D3D12_BLEND_DESC {
                AlphaToCoverageEnable: false.into(),
                IndependentBlendEnable: false.into(),
                RenderTarget: [
                    D3D12_RENDER_TARGET_BLEND_DESC {
                        BlendEnable: false.into(),
                        LogicOpEnable: false.into(),
                        SrcBlend: D3D12_BLEND_ONE,
                        DestBlend: D3D12_BLEND_ZERO,
                        BlendOp: D3D12_BLEND_OP_ADD,
                        SrcBlendAlpha: D3D12_BLEND_ONE,
                        DestBlendAlpha: D3D12_BLEND_ZERO,
                        BlendOpAlpha: D3D12_BLEND_OP_ADD,
                        LogicOp: D3D12_LOGIC_OP_NOOP,
                        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    },
                    Default::default(),
                    Default::default(),
                    Default::default(),
                    Default::default(),
                    Default::default(),
                    Default::default(),
                    Default::default(),
                ],
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                FrontCounterClockwise: true.into(),
                DepthBias: 0,
                DepthBiasClamp: 0.0,
                SlopeScaledDepthBias: 0.0,
                DepthClipEnable: true.into(),
                MultisampleEnable: false.into(),
                AntialiasedLineEnable: false.into(),
                ForcedSampleCount: 0,
                ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: input_elements.as_ptr(),
                NumElements: input_elements.len() as u32,
            },
            PrimitiveTopologyType: topology_type_to_d3d12(topology),
            NumRenderTargets: 1,
            RTVFormats: [
                format_to_dxgi(target_format),
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
            ],
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };

        let pipeline_state: ID3D12PipelineState =
            unsafe { logical_device.device.CreateGraphicsPipelineState(&pso_desc) }
                .context("Failed to create pipeline state")?;

        let handle = self.next_pipeline_handle;
        self.next_pipeline_handle += 1;

        self.pipelines.insert(
            handle,
            PipelineState {
                device_handle,
                pipeline_state,
                root_signature,
                vertex_stride: vertex_layout.stride,
            },
        );

        tracing::debug!("Created render pipeline {}", handle);
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
        let logical_device = self
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        // Create render target texture
        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format_to_dxgi(format),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        };

        let clear_value = D3D12_CLEAR_VALUE {
            Format: format_to_dxgi(format),
            Anonymous: D3D12_CLEAR_VALUE_0 {
                Color: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let mut texture: Option<ID3D12Resource> = None;
        unsafe {
            logical_device.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                Some(&clear_value),
                &mut texture,
            )
        }
        .context("Failed to create render target texture")?;

        let texture = texture.context("CreateCommittedResource returned null")?;

        // Create RTV
        let rtv_offset = self.next_rtv_offset;
        self.next_rtv_offset += 1;

        let rtv_handle = unsafe {
            let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (rtv_offset * logical_device.rtv_descriptor_size) as usize;
            handle
        };

        unsafe {
            logical_device
                .device
                .CreateRenderTargetView(&texture, None, rtv_handle);
        }

        // Create command list for this render target
        let command_list: ID3D12GraphicsCommandList = unsafe {
            logical_device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &logical_device.command_allocator,
                None,
            )
        }
        .context("Failed to create command list")?;

        // Close the command list initially
        unsafe { command_list.Close() }.ok();

        let handle = self.next_render_target_handle;
        self.next_render_target_handle += 1;

        self.render_targets.insert(
            handle,
            RenderTargetState {
                device_handle,
                width,
                height,
                format,
                texture,
                rtv_offset,
                depth_format: None,
                depth_texture: None,
                dsv_offset: None,
                staging_buffer: None,
                command_list,
                has_rendered: false,
            },
        );

        tracing::debug!(
            "Created render target {}x{} (handle={})",
            width,
            height,
            handle
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
        // Get device first
        let logical_device = self
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        // Reset command allocator and command list
        unsafe { logical_device.command_allocator.Reset() }
            .context("Failed to reset command allocator")?;

        let render_target = self
            .render_targets
            .get(&target)
            .context("Invalid render target handle")?;

        if render_target.device_handle != device_handle {
            anyhow::bail!("Render target belongs to a different device");
        }

        let cmd = &render_target.command_list;
        let width = render_target.width;
        let height = render_target.height;

        unsafe { cmd.Reset(&logical_device.command_allocator, None) }
            .context("Failed to reset command list")?;

        // Get RTV handle
        let rtv_handle = unsafe {
            let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (render_target.rtv_offset * logical_device.rtv_descriptor_size) as usize;
            handle
        };

        // Find clear color
        let clear_color = commands
            .iter()
            .find_map(|c| match c {
                RenderCommand::Clear(color) => Some(*color),
                _ => None,
            })
            .unwrap_or(Color::BLACK);

        // Clear and set render target
        unsafe {
            cmd.ClearRenderTargetView(
                rtv_handle,
                &[clear_color.r, clear_color.g, clear_color.b, clear_color.a],
                None,
            );
            cmd.OMSetRenderTargets(1, Some(&rtv_handle), false, None);
        }

        // Set viewport and scissor
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = windows::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        unsafe {
            cmd.RSSetViewports(&[viewport]);
            cmd.RSSetScissorRects(&[scissor]);
        }

        // Execute render commands
        let mut current_vertex_stride = 24u32; // Default stride
        for command in commands {
            match command {
                RenderCommand::Clear(_) => {
                    // Already handled
                }
                RenderCommand::ClearDepth(_) => {
                    // TODO: Implement depth clear
                }
                RenderCommand::SetPipeline(pipeline_handle) => {
                    if let Some(pipeline) = self.pipelines.get(pipeline_handle) {
                        current_vertex_stride = pipeline.vertex_stride;
                        unsafe {
                            cmd.SetGraphicsRootSignature(&pipeline.root_signature);
                            cmd.SetPipelineState(&pipeline.pipeline_state);
                        }
                    }
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                } => {
                    if let Some(buf_state) = self.buffers.get(buffer) {
                        let view = D3D12_VERTEX_BUFFER_VIEW {
                            BufferLocation: unsafe { buf_state.resource.GetGPUVirtualAddress() }
                                + offset,
                            SizeInBytes: (buf_state.size - offset) as u32,
                            StrideInBytes: current_vertex_stride,
                        };
                        unsafe { cmd.IASetVertexBuffers(*slot, Some(&[view])) };
                    }
                }
                RenderCommand::SetIndexBuffer {
                    buffer,
                    offset,
                    format,
                } => {
                    if let Some(buf_state) = self.buffers.get(buffer) {
                        let view = D3D12_INDEX_BUFFER_VIEW {
                            BufferLocation: unsafe { buf_state.resource.GetGPUVirtualAddress() }
                                + offset,
                            SizeInBytes: (buf_state.size - offset) as u32,
                            Format: index_format_to_dxgi(*format),
                        };
                        unsafe { cmd.IASetIndexBuffer(Some(&view)) };
                    }
                }
                RenderCommand::SetBindGroup { index, bind_group } => {
                    if let Some(bg_state) = self.bind_groups.get(bind_group) {
                        let layout = self.bind_group_layouts.get(&bg_state.layout_handle);

                        for (binding, buffer_handle, _offset, _size) in &bg_state.buffer_bindings {
                            if let Some(buf_state) = self.buffers.get(buffer_handle) {
                                let gpu_address =
                                    unsafe { buf_state.resource.GetGPUVirtualAddress() };
                                let binding_type = layout.and_then(|l| {
                                    l.entries
                                        .iter()
                                        .find(|e| e.binding == *binding)
                                        .map(|e| &e.ty)
                                });
                                let root_param_idx = *index + *binding;

                                unsafe {
                                    match binding_type {
                                        Some(BindingType::StorageBuffer { read_only: true }) => {
                                            cmd.SetGraphicsRootShaderResourceView(
                                                root_param_idx,
                                                gpu_address,
                                            );
                                        }
                                        Some(BindingType::StorageBuffer { read_only: false }) => {
                                            cmd.SetGraphicsRootUnorderedAccessView(
                                                root_param_idx,
                                                gpu_address,
                                            );
                                        }
                                        _ => {
                                            cmd.SetGraphicsRootConstantBufferView(
                                                root_param_idx,
                                                gpu_address,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => unsafe {
                    cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    cmd.DrawInstanced(
                        *vertex_count,
                        *instance_count,
                        *first_vertex,
                        *first_instance,
                    );
                },
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => unsafe {
                    cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    cmd.DrawIndexedInstanced(
                        *index_count,
                        *instance_count,
                        *first_index,
                        *base_vertex,
                        *first_instance,
                    );
                },
            }
        }

        // Transition to copy source for potential readback
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(&render_target.texture) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        };
        unsafe { cmd.ResourceBarrier(&[barrier]) };

        // Close and execute
        unsafe { cmd.Close() }.context("Failed to close command list")?;

        let cmd_list: ID3D12CommandList = cmd.cast().context("Failed to cast command list")?;
        unsafe {
            logical_device
                .command_queue
                .ExecuteCommandLists(&[Some(cmd_list)]);
        }

        // Wait for completion
        let fence_value = logical_device.fence_value;
        unsafe {
            logical_device
                .command_queue
                .Signal(&logical_device.fence, fence_value)
        }
        .context("Failed to signal fence")?;

        if unsafe { logical_device.fence.GetCompletedValue() } < fence_value {
            let event = unsafe { CreateEventA(None, false, false, None) }
                .context("Failed to create event")?;
            unsafe {
                logical_device
                    .fence
                    .SetEventOnCompletion(fence_value, event)
            }
            .context("Failed to set event")?;
            unsafe { WaitForSingleObject(event, INFINITE) };
            unsafe { CloseHandle(event) }.ok();
        }

        // Increment fence value for next operation
        if let Some(dev) = self.devices.get_mut(&device_handle) {
            dev.fence_value += 1;
        }

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
            .context("Invalid render target handle")?;

        if !render_target.has_rendered {
            anyhow::bail!("Cannot read from render target that hasn't been rendered to");
        }

        let width = render_target.width;
        let height = render_target.height;
        let format = render_target.format;
        let expected_size = (width * height * format.bytes_per_pixel()) as usize;

        if output.len() < expected_size {
            anyhow::bail!(
                "Output buffer too small: {} < {}",
                output.len(),
                expected_size
            );
        }

        let device_handle = render_target.device_handle;
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Ensure staging buffer exists
        let needs_staging = render_target.staging_buffer.is_none();
        if needs_staging {
            let row_pitch = ((width * format.bytes_per_pixel() + 255) & !255) as u64; // 256-byte aligned
            let staging_size = row_pitch * height as u64;

            let heap_properties = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            };

            let resource_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: staging_size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_NONE,
            };

            let mut staging_buffer: Option<ID3D12Resource> = None;
            unsafe {
                logical_device.device.CreateCommittedResource(
                    &heap_properties,
                    D3D12_HEAP_FLAG_NONE,
                    &resource_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut staging_buffer,
                )
            }
            .context("Failed to create staging buffer")?;

            let render_target = self.render_targets.get_mut(&target).unwrap();
            render_target.staging_buffer = staging_buffer;
            tracing::debug!("Created staging buffer for render target {}", target);
        }

        let render_target = self.render_targets.get(&target).unwrap();
        let staging_buffer = render_target.staging_buffer.as_ref().unwrap();
        let cmd = &render_target.command_list;

        // Reset and record copy command
        unsafe { logical_device.command_allocator.Reset() }
            .context("Failed to reset command allocator")?;
        unsafe { cmd.Reset(&logical_device.command_allocator, None) }
            .context("Failed to reset command list")?;

        // Copy texture to staging buffer
        let row_pitch = ((width * format.bytes_per_pixel() + 255) & !255) as u64;

        let src_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&render_target.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };

        let dst_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(staging_buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format_to_dxgi(format),
                        Width: width,
                        Height: height,
                        Depth: 1,
                        RowPitch: row_pitch as u32,
                    },
                },
            },
        };

        unsafe {
            cmd.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None);
        }

        unsafe { cmd.Close() }.context("Failed to close command list")?;

        let cmd_list: ID3D12CommandList = cmd.cast().context("Failed to cast command list")?;
        unsafe {
            logical_device
                .command_queue
                .ExecuteCommandLists(&[Some(cmd_list)]);
        }

        // Wait for completion
        let fence_value = logical_device.fence_value;
        unsafe {
            logical_device
                .command_queue
                .Signal(&logical_device.fence, fence_value)
        }
        .context("Failed to signal fence")?;

        if unsafe { logical_device.fence.GetCompletedValue() } < fence_value {
            let event = unsafe { CreateEventA(None, false, false, None) }
                .context("Failed to create event")?;
            unsafe {
                logical_device
                    .fence
                    .SetEventOnCompletion(fence_value, event)
            }
            .context("Failed to set event")?;
            unsafe { WaitForSingleObject(event, INFINITE) };
            unsafe { CloseHandle(event) }.ok();
        }

        // Read from staging buffer
        let mut mapped_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let read_range = D3D12_RANGE {
            Begin: 0,
            End: expected_size,
        };

        unsafe { staging_buffer.Map(0, Some(&read_range), Some(&mut mapped_ptr)) }
            .context("Failed to map staging buffer")?;

        // Copy data (handle row pitch alignment)
        let bytes_per_row = (width * format.bytes_per_pixel()) as usize;
        for row in 0..height {
            let src_offset = (row as u64 * row_pitch) as usize;
            let dst_offset = (row as usize) * bytes_per_row;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (mapped_ptr as *const u8).add(src_offset),
                    output.as_mut_ptr().add(dst_offset),
                    bytes_per_row,
                );
            }
        }

        unsafe { staging_buffer.Unmap(0, None) };

        Ok(())
    }

    fn create_surface(
        &mut self,
        device_handle: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        _display: &dyn raw_window_handle::HasDisplayHandle,
    ) -> Result<SurfaceHandle> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {:?}", e))?;

        let hwnd = match window_handle.as_raw() {
            RawWindowHandle::Win32(h) => HWND(h.hwnd.get() as isize as *mut std::ffi::c_void),
            _ => anyhow::bail!("Expected Win32 window handle"),
        };

        // Get window dimensions
        let mut rect = windows::Win32::Foundation::RECT::default();
        unsafe { GetClientRect(hwnd, &mut rect) }.context("Failed to get window rect")?;

        let width = (rect.right - rect.left) as u32;
        let height = (rect.bottom - rect.top) as u32;

        // Create swapchain
        let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: MAX_FRAMES_IN_FLIGHT as u32,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: 0,
        };

        let swapchain: IDXGISwapChain1 = unsafe {
            self.factory.CreateSwapChainForHwnd(
                &logical_device.command_queue,
                hwnd,
                &swap_chain_desc,
                None,
                None,
            )
        }
        .context("Failed to create swapchain")?;

        let swapchain: IDXGISwapChain3 = swapchain
            .cast()
            .context("Failed to cast swapchain to IDXGISwapChain3")?;

        // Get swapchain buffers and create RTVs
        let mut render_targets = Vec::new();
        let mut rtv_offsets = Vec::new();

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer: ID3D12Resource = unsafe { swapchain.GetBuffer(i as u32) }
                .context("Failed to get swapchain buffer")?;

            let rtv_offset = self.next_rtv_offset;
            self.next_rtv_offset += 1;

            let rtv_handle = unsafe {
                let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
                handle.ptr += (rtv_offset * logical_device.rtv_descriptor_size) as usize;
                handle
            };

            unsafe {
                logical_device
                    .device
                    .CreateRenderTargetView(&buffer, None, rtv_handle);
            }

            render_targets.push(buffer);
            rtv_offsets.push(rtv_offset);
        }

        // Create per-frame sync resources
        let mut frame_sync = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let command_allocator: ID3D12CommandAllocator = unsafe {
                logical_device
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .context("Failed to create command allocator")?;

            let command_list: ID3D12GraphicsCommandList = unsafe {
                logical_device.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &command_allocator,
                    None,
                )
            }
            .context("Failed to create command list")?;

            unsafe { command_list.Close() }.ok();

            frame_sync.push(FrameSync {
                command_list,
                command_allocator,
                fence_value: 0,
            });
        }

        let handle = self.next_surface_handle;
        self.next_surface_handle += 1;

        self.surfaces.insert(
            handle,
            SurfaceState {
                device_handle,
                swapchain,
                render_targets,
                rtv_offsets,
                width,
                height,
                format: DXGI_FORMAT_B8G8R8A8_UNORM,
                current_frame: 0,
                current_image_index: None,
                frame_sync,
            },
        );

        tracing::info!("Created surface {}x{}", width, height);
        Ok(handle)
    }

    fn destroy_surface(&mut self, surface_handle: SurfaceHandle) {
        if let Some(surface_state) = self.surfaces.remove(&surface_handle) {
            if let Some(logical_device) = self.devices.get(&surface_state.device_handle) {
                // Wait for GPU
                let _ = self.wait_for_gpu(logical_device);
            }
        }
    }

    fn surface_acquire(&mut self, surface_handle: SurfaceHandle) -> Result<SwapchainImageHandle> {
        let surface = self
            .surfaces
            .get_mut(&surface_handle)
            .context("Invalid surface handle")?;

        let image_index = unsafe { surface.swapchain.GetCurrentBackBufferIndex() };
        surface.current_image_index = Some(image_index);

        Ok(image_index as SwapchainImageHandle)
    }

    fn surface_render(
        &mut self,
        surface_handle: SurfaceHandle,
        _image: SwapchainImageHandle,
        commands: &[RenderCommand],
    ) -> Result<()> {
        let surface = self
            .surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;

        let image_index = surface
            .current_image_index
            .context("No image acquired - call surface_acquire first")?;

        let device_handle = surface.device_handle;
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Surface's device is invalid")?;

        let current_frame = surface.current_frame;
        let frame = &surface.frame_sync[current_frame];
        let cmd = &frame.command_list;
        let width = surface.width;
        let height = surface.height;
        let render_target = &surface.render_targets[image_index as usize];
        let rtv_offset = surface.rtv_offsets[image_index as usize];

        // Reset command allocator and list
        unsafe { frame.command_allocator.Reset() }.context("Failed to reset command allocator")?;
        unsafe { cmd.Reset(&frame.command_allocator, None) }
            .context("Failed to reset command list")?;

        // Transition to render target
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(render_target) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_PRESENT,
                    StateAfter: D3D12_RESOURCE_STATE_RENDER_TARGET,
                }),
            },
        };
        unsafe { cmd.ResourceBarrier(&[barrier]) };

        // Get RTV handle
        let rtv_handle = unsafe {
            let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (rtv_offset * logical_device.rtv_descriptor_size) as usize;
            handle
        };

        // Find clear color
        let clear_color = commands
            .iter()
            .find_map(|c| match c {
                RenderCommand::Clear(color) => Some(*color),
                _ => None,
            })
            .unwrap_or(Color::BLACK);

        unsafe {
            cmd.ClearRenderTargetView(
                rtv_handle,
                &[clear_color.r, clear_color.g, clear_color.b, clear_color.a],
                None,
            );
            cmd.OMSetRenderTargets(1, Some(&rtv_handle), false, None);
        }

        // Set viewport and scissor
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = windows::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        unsafe {
            cmd.RSSetViewports(&[viewport]);
            cmd.RSSetScissorRects(&[scissor]);
        }

        // Execute render commands
        let mut current_vertex_stride = 24u32; // Default stride
        for command in commands {
            match command {
                RenderCommand::Clear(_) => { /* Already handled */ }
                RenderCommand::ClearDepth(_) => { /* TODO: Implement depth clear */ }
                RenderCommand::SetPipeline(pipeline_handle) => {
                    if let Some(pipeline) = self.pipelines.get(pipeline_handle) {
                        current_vertex_stride = pipeline.vertex_stride;
                        unsafe {
                            cmd.SetGraphicsRootSignature(&pipeline.root_signature);
                            cmd.SetPipelineState(&pipeline.pipeline_state);
                        }
                    }
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                } => {
                    if let Some(buf_state) = self.buffers.get(buffer) {
                        let view = D3D12_VERTEX_BUFFER_VIEW {
                            BufferLocation: unsafe { buf_state.resource.GetGPUVirtualAddress() }
                                + offset,
                            SizeInBytes: (buf_state.size - offset) as u32,
                            StrideInBytes: current_vertex_stride,
                        };
                        unsafe { cmd.IASetVertexBuffers(*slot, Some(&[view])) };
                    }
                }
                RenderCommand::SetIndexBuffer {
                    buffer,
                    offset,
                    format,
                } => {
                    if let Some(buf_state) = self.buffers.get(buffer) {
                        let view = D3D12_INDEX_BUFFER_VIEW {
                            BufferLocation: unsafe { buf_state.resource.GetGPUVirtualAddress() }
                                + offset,
                            SizeInBytes: (buf_state.size - offset) as u32,
                            Format: index_format_to_dxgi(*format),
                        };
                        unsafe { cmd.IASetIndexBuffer(Some(&view)) };
                    }
                }
                RenderCommand::SetBindGroup { index, bind_group } => {
                    if let Some(bg_state) = self.bind_groups.get(bind_group) {
                        let layout = self.bind_group_layouts.get(&bg_state.layout_handle);

                        for (binding, buffer_handle, _offset, _size) in &bg_state.buffer_bindings {
                            if let Some(buf_state) = self.buffers.get(buffer_handle) {
                                let gpu_address =
                                    unsafe { buf_state.resource.GetGPUVirtualAddress() };
                                let binding_type = layout.and_then(|l| {
                                    l.entries
                                        .iter()
                                        .find(|e| e.binding == *binding)
                                        .map(|e| &e.ty)
                                });
                                let root_param_idx = *index + *binding;

                                unsafe {
                                    match binding_type {
                                        Some(BindingType::StorageBuffer { read_only: true }) => {
                                            cmd.SetGraphicsRootShaderResourceView(
                                                root_param_idx,
                                                gpu_address,
                                            );
                                        }
                                        Some(BindingType::StorageBuffer { read_only: false }) => {
                                            cmd.SetGraphicsRootUnorderedAccessView(
                                                root_param_idx,
                                                gpu_address,
                                            );
                                        }
                                        _ => {
                                            cmd.SetGraphicsRootConstantBufferView(
                                                root_param_idx,
                                                gpu_address,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => unsafe {
                    cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    cmd.DrawInstanced(
                        *vertex_count,
                        *instance_count,
                        *first_vertex,
                        *first_instance,
                    );
                },
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => unsafe {
                    cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    cmd.DrawIndexedInstanced(
                        *index_count,
                        *instance_count,
                        *first_index,
                        *base_vertex,
                        *first_instance,
                    );
                },
            }
        }

        // Transition to present
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(render_target) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: D3D12_RESOURCE_STATE_PRESENT,
                }),
            },
        };
        unsafe { cmd.ResourceBarrier(&[barrier]) };

        // Close and execute
        unsafe { cmd.Close() }.context("Failed to close command list")?;

        let cmd_list: ID3D12CommandList = cmd.cast().context("Failed to cast command list")?;
        unsafe {
            logical_device
                .command_queue
                .ExecuteCommandLists(&[Some(cmd_list)]);
        }

        // Signal fence for this frame
        let fence_value = logical_device.fence_value;
        unsafe {
            logical_device
                .command_queue
                .Signal(&logical_device.fence, fence_value)
        }
        .context("Failed to signal fence")?;

        // Update fence value for next operation
        if let Some(dev) = self.devices.get_mut(&device_handle) {
            dev.fence_value += 1;
        }

        Ok(())
    }

    fn surface_present(
        &mut self,
        surface_handle: SurfaceHandle,
        _image: SwapchainImageHandle,
    ) -> Result<()> {
        let surface = self
            .surfaces
            .get(&surface_handle)
            .context("Invalid surface handle")?;

        let device_handle = surface.device_handle;

        // Wait for render to complete before presenting
        {
            let logical_device = self
                .devices
                .get(&device_handle)
                .context("Surface's device is invalid")?;

            let fence_value = logical_device.fence_value.saturating_sub(1);
            if unsafe { logical_device.fence.GetCompletedValue() } < fence_value {
                let event = unsafe { CreateEventA(None, false, false, None) }
                    .context("Failed to create event")?;
                unsafe {
                    logical_device
                        .fence
                        .SetEventOnCompletion(fence_value, event)
                }
                .context("Failed to set event on completion")?;
                unsafe { WaitForSingleObject(event, INFINITE) };
                unsafe { CloseHandle(event) }.ok();
            }
        }

        // Present
        let surface = self.surfaces.get_mut(&surface_handle).unwrap();
        let hr = unsafe { surface.swapchain.Present(1, DXGI_PRESENT(0)) };
        if hr.is_err() {
            anyhow::bail!("Present failed with HRESULT: {:?}", hr);
        }

        // Advance frame
        surface.current_image_index = None;
        surface.current_frame = (surface.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    fn surface_resize(
        &mut self,
        surface_handle: SurfaceHandle,
        width: u32,
        height: u32,
    ) -> Result<()> {
        // Get device handle and surface format first
        let (device_handle, surface_format) = {
            let surface = self
                .surfaces
                .get(&surface_handle)
                .context("Invalid surface handle")?;
            (surface.device_handle, surface.format)
        };

        // Wait for GPU
        {
            let logical_device = self
                .devices
                .get(&device_handle)
                .context("Surface's device is invalid")?;
            let _ = self.wait_for_gpu(logical_device);
        }

        // Release old render targets and resize swapchain
        {
            let surface = self.surfaces.get_mut(&surface_handle).unwrap();
            surface.render_targets.clear();
            surface.rtv_offsets.clear();

            // Resize swapchain
            unsafe {
                surface.swapchain.ResizeBuffers(
                    MAX_FRAMES_IN_FLIGHT as u32,
                    width,
                    height,
                    surface_format,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
            }
            .context("Failed to resize swapchain")?;

            surface.width = width;
            surface.height = height;
        }

        // Get device info for creating RTVs
        let (rtv_heap, rtv_descriptor_size, device) = {
            let logical_device = self
                .devices
                .get(&device_handle)
                .context("Surface's device is invalid")?;
            (
                logical_device.rtv_heap.clone(),
                logical_device.rtv_descriptor_size,
                logical_device.device.clone(),
            )
        };

        // Recreate render targets
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let surface = self.surfaces.get(&surface_handle).unwrap();
            let buffer: ID3D12Resource = unsafe { surface.swapchain.GetBuffer(i as u32) }
                .context("Failed to get swapchain buffer")?;

            let rtv_offset = self.next_rtv_offset;
            self.next_rtv_offset += 1;

            let rtv_handle = unsafe {
                let mut handle = rtv_heap.GetCPUDescriptorHandleForHeapStart();
                handle.ptr += (rtv_offset * rtv_descriptor_size) as usize;
                handle
            };

            unsafe {
                device.CreateRenderTargetView(&buffer, None, rtv_handle);
            }

            let surface = self.surfaces.get_mut(&surface_handle).unwrap();
            surface.render_targets.push(buffer);
            surface.rtv_offsets.push(rtv_offset);
        }

        let surface = self.surfaces.get_mut(&surface_handle).unwrap();
        surface.current_frame = 0;
        surface.current_image_index = None;

        tracing::debug!("Resized surface to {}x{}", width, height);
        Ok(())
    }

    fn surface_size(&self, surface_handle: SurfaceHandle) -> (u32, u32) {
        self.surfaces
            .get(&surface_handle)
            .map(|s| (s.width, s.height))
            .unwrap_or((0, 0))
    }

    fn surface_format(&self, surface_handle: SurfaceHandle) -> TextureFormat {
        self.surfaces
            .get(&surface_handle)
            .and_then(|s| dxgi_to_format(s.format))
            .unwrap_or(TextureFormat::Bgra8Unorm)
    }

    fn create_pipeline_with_depth(
        &mut self,
        device_handle: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        bind_group_layouts: &[BindGroupLayoutHandle],
        _depth_stencil: Option<&crate::types::DepthStencilState>,
    ) -> Result<PipelineHandle> {
        // TODO: Implement proper depth stencil state in PSO
        // For now, delegate to the existing method (ignoring depth stencil)
        self.create_pipeline_with_layout(
            device_handle,
            vertex_shader,
            fragment_shader,
            vertex_layout,
            topology,
            target_format,
            bind_group_layouts,
        )
    }

    fn create_render_target_with_depth(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<crate::types::DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        let logical_device = self
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        // Create color render target texture
        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format_to_dxgi(color_format),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        };

        let clear_value = D3D12_CLEAR_VALUE {
            Format: format_to_dxgi(color_format),
            Anonymous: D3D12_CLEAR_VALUE_0 {
                Color: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let mut texture: Option<ID3D12Resource> = None;
        unsafe {
            logical_device.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                Some(&clear_value),
                &mut texture,
            )
        }
        .context("Failed to create render target texture")?;
        let texture = texture.context("CreateCommittedResource returned null")?;

        // Create RTV
        let rtv_offset = self.next_rtv_offset;
        self.next_rtv_offset += 1;

        let rtv_handle = unsafe {
            let mut handle = logical_device.rtv_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (rtv_offset * logical_device.rtv_descriptor_size) as usize;
            handle
        };
        unsafe {
            logical_device
                .device
                .CreateRenderTargetView(&texture, None, rtv_handle);
        }

        // Create depth buffer if requested
        let (depth_texture, dsv_offset) = if let Some(df) = depth_format {
            let depth_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: width as u64,
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: depth_format_to_dxgi(df),
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            };

            let depth_clear = D3D12_CLEAR_VALUE {
                Format: depth_format_to_dxgi(df),
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                        Depth: 1.0,
                        Stencil: 0,
                    },
                },
            };

            let mut depth_tex: Option<ID3D12Resource> = None;
            unsafe {
                logical_device.device.CreateCommittedResource(
                    &heap_properties,
                    D3D12_HEAP_FLAG_NONE,
                    &depth_desc,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                    Some(&depth_clear),
                    &mut depth_tex,
                )
            }
            .context("Failed to create depth buffer")?;
            let depth_tex = depth_tex.context("CreateCommittedResource returned null for depth")?;

            let dsv_off = self.next_dsv_offset;
            self.next_dsv_offset += 1;

            let dsv_handle = unsafe {
                let mut handle = logical_device.dsv_heap.GetCPUDescriptorHandleForHeapStart();
                handle.ptr += (dsv_off * logical_device.dsv_descriptor_size) as usize;
                handle
            };
            unsafe {
                logical_device
                    .device
                    .CreateDepthStencilView(&depth_tex, None, dsv_handle);
            }

            (Some(depth_tex), Some(dsv_off))
        } else {
            (None, None)
        };

        // Create command list
        let command_list: ID3D12GraphicsCommandList = unsafe {
            logical_device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &logical_device.command_allocator,
                None,
            )
        }
        .context("Failed to create command list")?;
        unsafe { command_list.Close() }.ok();

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
                rtv_offset,
                depth_format,
                depth_texture,
                dsv_offset,
                staging_buffer: None,
                command_list,
                has_rendered: false,
            },
        );

        tracing::debug!(
            "Created render target {}x{} with depth={:?} (handle={})",
            width,
            height,
            depth_format.is_some(),
            handle
        );
        Ok(handle)
    }

    fn create_texture(
        &mut self,
        device_handle: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        _usage: crate::types::TextureUsage,
    ) -> Result<TextureHandle> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format_to_dxgi(format),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            logical_device.device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut resource,
            )
        }
        .context("Failed to create texture")?;
        let resource = resource.context("CreateCommittedResource returned null")?;

        // Create SRV
        let srv_offset = self.next_srv_offset;
        self.next_srv_offset += 1;

        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format_to_dxgi(format),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };

        let srv_handle = unsafe {
            let mut handle = logical_device
                .cbv_srv_uav_heap
                .GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (srv_offset * logical_device.cbv_srv_uav_descriptor_size) as usize;
            handle
        };
        unsafe {
            logical_device
                .device
                .CreateShaderResourceView(&resource, Some(&srv_desc), srv_handle);
        }

        let handle = self.next_texture_handle;
        self.next_texture_handle += 1;

        self.textures.insert(
            handle,
            TextureState {
                device_handle,
                width,
                height,
                format,
                resource,
                srv_offset,
            },
        );

        tracing::debug!("Created texture {}x{} (handle={})", width, height, handle);
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

        if texture.width != width || texture.height != height {
            anyhow::bail!("Texture dimensions mismatch");
        }

        let device_handle = texture.device_handle;
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Calculate row pitch (must be 256-byte aligned for D3D12)
        let row_pitch = ((width * texture.format.bytes_per_pixel() + 255) & !255) as u64;
        let staging_size = row_pitch * height as u64;

        // Create staging buffer
        let upload_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let buffer_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: staging_size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };

        let mut staging: Option<ID3D12Resource> = None;
        unsafe {
            logical_device.device.CreateCommittedResource(
                &upload_heap,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut staging,
            )
        }
        .context("Failed to create staging buffer")?;
        let staging = staging.context("CreateCommittedResource returned null")?;

        // Map and copy data
        let mut mapped_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let read_range = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { staging.Map(0, Some(&read_range), Some(&mut mapped_ptr)) }
            .context("Failed to map staging buffer")?;

        let bytes_per_row = (width * texture.format.bytes_per_pixel()) as usize;
        for row in 0..height {
            let src_offset = (row as usize) * bytes_per_row;
            let dst_offset = (row as u64 * row_pitch) as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(src_offset),
                    (mapped_ptr as *mut u8).add(dst_offset),
                    bytes_per_row,
                );
            }
        }

        let written_range = D3D12_RANGE {
            Begin: 0,
            End: staging_size as usize,
        };
        unsafe { staging.Unmap(0, Some(&written_range)) };

        // Execute copy command
        unsafe { logical_device.command_allocator.Reset() }
            .context("Failed to reset command allocator")?;

        let command_list: ID3D12GraphicsCommandList = unsafe {
            logical_device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &logical_device.command_allocator,
                None,
            )
        }
        .context("Failed to create command list")?;

        let texture = self.textures.get(&texture_handle).unwrap();

        let src_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&staging) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format_to_dxgi(texture.format),
                        Width: width,
                        Height: height,
                        Depth: 1,
                        RowPitch: row_pitch as u32,
                    },
                },
            },
        };

        let dst_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&texture.resource) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };

        unsafe {
            command_list.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None);

            // Transition to shader resource
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&texture.resource),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                        StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    }),
                },
            };
            command_list.ResourceBarrier(&[barrier]);
            command_list.Close()
        }
        .context("Failed to close command list")?;

        let cmd_list: ID3D12CommandList =
            command_list.cast().context("Failed to cast command list")?;
        unsafe {
            logical_device
                .command_queue
                .ExecuteCommandLists(&[Some(cmd_list)]);
        }

        // Wait for completion
        let _ = self.wait_for_gpu(logical_device);

        tracing::debug!("Wrote {}x{} texture data", width, height);
        Ok(())
    }

    fn destroy_texture(&mut self, texture_handle: TextureHandle) {
        self.textures.remove(&texture_handle);
    }

    fn create_sampler(
        &mut self,
        device_handle: DeviceHandle,
        desc: &crate::types::SamplerDesc,
    ) -> Result<SamplerHandle> {
        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        let sampler_offset = self.next_sampler_offset;
        self.next_sampler_offset += 1;

        let sampler_desc = D3D12_SAMPLER_DESC {
            Filter: utils::filter_to_d3d12(desc.min_filter, desc.mag_filter, desc.mipmap_filter),
            AddressU: utils::address_mode_to_d3d12(desc.address_mode_u),
            AddressV: utils::address_mode_to_d3d12(desc.address_mode_v),
            AddressW: utils::address_mode_to_d3d12(desc.address_mode_w),
            MipLODBias: 0.0,
            MaxAnisotropy: desc.max_anisotropy as u32,
            ComparisonFunc: desc
                .compare
                .map(utils::compare_to_d3d12)
                .unwrap_or(D3D12_COMPARISON_FUNC_ALWAYS),
            BorderColor: [0.0, 0.0, 0.0, 0.0],
            MinLOD: desc.lod_min_clamp,
            MaxLOD: desc.lod_max_clamp,
        };

        let sampler_handle = unsafe {
            let mut handle = logical_device
                .sampler_heap
                .GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (sampler_offset * logical_device.sampler_descriptor_size) as usize;
            handle
        };
        unsafe {
            logical_device
                .device
                .CreateSampler(&sampler_desc, sampler_handle);
        }

        let handle = self.next_sampler_handle;
        self.next_sampler_handle += 1;

        self.samplers.insert(
            handle,
            SamplerState {
                device_handle,
                sampler_offset,
                desc: desc.clone(),
            },
        );

        tracing::debug!("Created sampler (handle={})", handle);
        Ok(handle)
    }

    fn destroy_sampler(&mut self, sampler_handle: SamplerHandle) {
        self.samplers.remove(&sampler_handle);
    }

    fn create_compute_pipeline(
        &mut self,
        device_handle: DeviceHandle,
        compute_shader: ShaderHandle,
        bind_group_layouts: &[BindGroupLayoutHandle],
    ) -> Result<ComputePipelineHandle> {
        // Compile shader on-demand
        let cs_bytecode =
            self.ensure_shader_stage_compiled(compute_shader, crate::slang::SlangStage::Compute)?;

        let logical_device = self
            .devices
            .get(&device_handle)
            .context("Invalid device handle")?;

        // Create root signature
        let root_signature = if bind_group_layouts.is_empty() {
            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 0,
                pParameters: std::ptr::null(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            };

            let mut signature_blob: Option<ID3DBlob> = None;
            let mut error_blob: Option<ID3DBlob> = None;

            unsafe {
                D3D12SerializeRootSignature(
                    &desc,
                    D3D_ROOT_SIGNATURE_VERSION_1,
                    &mut signature_blob,
                    Some(&mut error_blob),
                )
            }
            .context("Failed to serialize root signature")?;

            let blob = signature_blob.context("Root signature serialization produced no output")?;
            let signature: ID3D12RootSignature = unsafe {
                logical_device.device.CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    ),
                )
            }
            .context("Failed to create root signature")?;

            signature
        } else {
            // Create root signature with root descriptors (SRV/UAV/CBV) for each binding
            // Using root descriptors is simpler than descriptor tables for buffers
            let mut root_params: Vec<D3D12_ROOT_PARAMETER> = Vec::new();

            // Track register indices separately for each register space
            // SRV uses t registers, UAV uses u registers, CBV uses b registers
            let mut srv_register = 0u32;
            let mut uav_register = 0u32;
            let mut cbv_register = 0u32;

            // Flatten all bindings from all layouts into root parameters
            // Each binding becomes its own root parameter (root descriptors)
            for layout_handle in bind_group_layouts.iter() {
                if let Some(layout) = self.bind_group_layouts.get(layout_handle) {
                    for entry in &layout.entries {
                        let (param_type, register) = match &entry.ty {
                            BindingType::StorageBuffer { read_only: true } => {
                                let reg = srv_register;
                                srv_register += 1;
                                (D3D12_ROOT_PARAMETER_TYPE_SRV, reg)
                            }
                            BindingType::StorageBuffer { read_only: false } => {
                                let reg = uav_register;
                                uav_register += 1;
                                (D3D12_ROOT_PARAMETER_TYPE_UAV, reg)
                            }
                            BindingType::UniformBuffer => {
                                let reg = cbv_register;
                                cbv_register += 1;
                                (D3D12_ROOT_PARAMETER_TYPE_CBV, reg)
                            }
                            _ => {
                                tracing::warn!("Unsupported binding type in compute pipeline");
                                continue;
                            }
                        };

                        root_params.push(D3D12_ROOT_PARAMETER {
                            ParameterType: param_type,
                            Anonymous: D3D12_ROOT_PARAMETER_0 {
                                Descriptor: D3D12_ROOT_DESCRIPTOR {
                                    ShaderRegister: register,
                                    RegisterSpace: 0,
                                },
                            },
                            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                        });
                    }
                }
            }

            tracing::debug!(
                "Creating root signature with {} root parameters",
                root_params.len()
            );
            for (i, param) in root_params.iter().enumerate() {
                let (ty, reg) = unsafe {
                    match param.ParameterType {
                        D3D12_ROOT_PARAMETER_TYPE_SRV => {
                            ("SRV", param.Anonymous.Descriptor.ShaderRegister)
                        }
                        D3D12_ROOT_PARAMETER_TYPE_UAV => {
                            ("UAV", param.Anonymous.Descriptor.ShaderRegister)
                        }
                        D3D12_ROOT_PARAMETER_TYPE_CBV => {
                            ("CBV", param.Anonymous.Descriptor.ShaderRegister)
                        }
                        _ => ("???", 0),
                    }
                };
                tracing::debug!("  Root param {}: {} at register {}", i, ty, reg);
            }

            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: root_params.len() as u32,
                pParameters: if root_params.is_empty() {
                    std::ptr::null()
                } else {
                    root_params.as_ptr()
                },
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            };

            let mut signature_blob: Option<ID3DBlob> = None;
            let mut error_blob: Option<ID3DBlob> = None;

            let result = unsafe {
                D3D12SerializeRootSignature(
                    &desc,
                    D3D_ROOT_SIGNATURE_VERSION_1,
                    &mut signature_blob,
                    Some(&mut error_blob),
                )
            };

            if let Err(e) = result {
                if let Some(err) = error_blob {
                    let msg = unsafe {
                        let ptr = err.GetBufferPointer() as *const u8;
                        let len = err.GetBufferSize();
                        std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len))
                    };
                    tracing::error!("Root signature serialization error: {}", msg);
                }
                return Err(anyhow::anyhow!(
                    "Failed to serialize root signature: {:?}",
                    e
                ));
            }

            let blob = signature_blob.context("Root signature serialization produced no output")?;
            let signature: ID3D12RootSignature = unsafe {
                logical_device.device.CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    ),
                )
            }
            .context("Failed to create root signature")?;

            signature
        };

        // Create compute PSO
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: cs_bytecode.as_ptr() as *const _,
                BytecodeLength: cs_bytecode.len(),
            },
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };

        let pipeline_state: ID3D12PipelineState =
            unsafe { logical_device.device.CreateComputePipelineState(&pso_desc) }
                .context("Failed to create compute pipeline state")?;

        let handle = self.next_compute_pipeline_handle;
        self.next_compute_pipeline_handle += 1;

        self.compute_pipelines.insert(
            handle,
            ComputePipelineState {
                device_handle,
                pipeline_state,
                root_signature,
                bind_group_layouts: bind_group_layouts.to_vec(),
            },
        );

        tracing::debug!("Created compute pipeline {}", handle);
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
        let logical_device = self
            .devices
            .get_mut(&device_handle)
            .context("Invalid device handle")?;

        // Reset command allocator
        unsafe { logical_device.command_allocator.Reset() }
            .context("Failed to reset command allocator")?;

        // Create command list
        let command_list: ID3D12GraphicsCommandList = unsafe {
            logical_device.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &logical_device.command_allocator,
                None,
            )
        }
        .context("Failed to create command list")?;

        // Track current pipeline for bind group binding
        let mut current_pipeline_handle: Option<ComputePipelineHandle> = None;

        // Process commands
        for command in commands {
            match command {
                ComputeCommand::SetPipeline(handle) => {
                    if let Some(pipeline_state) = self.compute_pipelines.get(handle) {
                        unsafe {
                            command_list.SetComputeRootSignature(&pipeline_state.root_signature);
                            command_list.SetPipelineState(&pipeline_state.pipeline_state);
                        }
                        current_pipeline_handle = Some(*handle);
                    }
                }
                ComputeCommand::SetBindGroup { index, bind_group } => {
                    if let (Some(pipeline_handle), Some(bg_state)) =
                        (&current_pipeline_handle, self.bind_groups.get(bind_group))
                    {
                        // Get the pipeline to look up layouts
                        if let Some(pipeline_state) = self.compute_pipelines.get(pipeline_handle) {
                            // Get the layout for this bind group index
                            if let Some(layout_handle) =
                                pipeline_state.bind_group_layouts.get(*index as usize)
                            {
                                if let Some(layout) = self.bind_group_layouts.get(layout_handle) {
                                    // Calculate root parameter index
                                    // Each binding becomes a separate root parameter
                                    let mut root_param_base = 0u32;
                                    for i in 0..*index {
                                        if let Some(lh) =
                                            pipeline_state.bind_group_layouts.get(i as usize)
                                        {
                                            if let Some(l) = self.bind_group_layouts.get(lh) {
                                                root_param_base += l.entries.len() as u32;
                                            }
                                        }
                                    }

                                    for (binding, buffer_handle, _offset, _size) in
                                        &bg_state.buffer_bindings
                                    {
                                        if let Some(buf_state) = self.buffers.get(buffer_handle) {
                                            let gpu_address = unsafe {
                                                buf_state.resource.GetGPUVirtualAddress()
                                            };

                                            // Find the binding type in the layout
                                            if let Some(entry) = layout
                                                .entries
                                                .iter()
                                                .find(|e| e.binding == *binding)
                                            {
                                                // Find which root parameter this binding corresponds to
                                                let mut local_idx = 0u32;
                                                for e in &layout.entries {
                                                    if e.binding == *binding {
                                                        break;
                                                    }
                                                    local_idx += 1;
                                                }
                                                let root_param_idx = root_param_base + local_idx;

                                                unsafe {
                                                    match &entry.ty {
                                                        BindingType::StorageBuffer {
                                                            read_only: true,
                                                        } => {
                                                            command_list
                                                                .SetComputeRootShaderResourceView(
                                                                    root_param_idx,
                                                                    gpu_address,
                                                                );
                                                        }
                                                        BindingType::StorageBuffer {
                                                            read_only: false,
                                                        } => {
                                                            command_list
                                                                .SetComputeRootUnorderedAccessView(
                                                                    root_param_idx,
                                                                    gpu_address,
                                                                );
                                                        }
                                                        BindingType::UniformBuffer => {
                                                            command_list
                                                                .SetComputeRootConstantBufferView(
                                                                    root_param_idx,
                                                                    gpu_address,
                                                                );
                                                        }
                                                        _ => {
                                                            tracing::warn!("Unsupported binding type in compute dispatch");
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                ComputeCommand::Dispatch {
                    workgroups_x,
                    workgroups_y,
                    workgroups_z,
                } => unsafe {
                    command_list.Dispatch(*workgroups_x, *workgroups_y, *workgroups_z);
                },
            }
        }

        // Close and execute
        unsafe { command_list.Close() }.context("Failed to close command list")?;

        let cmd_list: ID3D12CommandList =
            command_list.cast().context("Failed to cast command list")?;

        let logical_device = self.devices.get(&device_handle).unwrap();
        unsafe {
            logical_device
                .command_queue
                .ExecuteCommandLists(&[Some(cmd_list)]);
        }

        // Wait for completion
        let fence_value = logical_device.fence_value;
        unsafe {
            logical_device
                .command_queue
                .Signal(&logical_device.fence, fence_value)
        }
        .context("Failed to signal fence")?;

        if unsafe { logical_device.fence.GetCompletedValue() } < fence_value {
            let event = unsafe { CreateEventA(None, false, false, None) }
                .context("Failed to create event")?;
            unsafe {
                logical_device
                    .fence
                    .SetEventOnCompletion(fence_value, event)
            }
            .context("Failed to set event")?;
            unsafe { WaitForSingleObject(event, INFINITE) };
            unsafe { CloseHandle(event) }.ok();
        }

        // Increment fence value for next operation
        if let Some(dev) = self.devices.get_mut(&device_handle) {
            dev.fence_value += 1;
        }

        Ok(())
    }
}
