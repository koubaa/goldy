//! Mock backend for testing.
//!
//! This backend validates command sequences without requiring GPU hardware,
//! enabling unit tests to run in CI environments.

use super::*;
use crate::types::*;
use anyhow::Result;
use std::collections::HashMap;

/// Mock GPU backend for testing.
/// 
/// Tracks resource creation/destruction and command recording
/// without actually performing GPU operations.
pub struct MockBackend {
    adapters: Vec<AdapterInfo>,
    devices: HashMap<DeviceHandle, MockDevice>,
    next_device_handle: DeviceHandle,
    buffers: HashMap<BufferHandle, MockBuffer>,
    next_buffer_handle: BufferHandle,
    shaders: HashMap<ShaderHandle, MockShader>,
    next_shader_handle: ShaderHandle,
    pipelines: HashMap<PipelineHandle, MockPipeline>,
    next_pipeline_handle: PipelineHandle,
    bind_group_layouts: HashMap<BindGroupLayoutHandle, MockBindGroupLayout>,
    next_bind_group_layout_handle: BindGroupLayoutHandle,
    bind_groups: HashMap<BindGroupHandle, MockBindGroup>,
    next_bind_group_handle: BindGroupHandle,
    render_targets: HashMap<RenderTargetHandle, MockRenderTarget>,
    next_render_target_handle: RenderTargetHandle,
    surfaces: HashMap<SurfaceHandle, MockSurface>,
    next_surface_handle: SurfaceHandle,
    /// Commands recorded during the last render operation
    pub recorded_commands: Vec<Vec<RenderCommand>>,
    /// Targets that were created (for verification)
    pub targets_created: Vec<(u32, u32, TextureFormat)>,
    /// Count of CPU readbacks performed
    pub cpu_readback_count: usize,
    /// Count of surface presents performed
    pub surface_present_count: usize,
}

struct MockDevice {
    adapter_id: u32,
}

struct MockBuffer {
    device_handle: DeviceHandle,
    size: u64,
    data: Vec<u8>,
}

struct MockShader {
    device_handle: DeviceHandle,
    source: String,
}

struct MockPipeline {
    device_handle: DeviceHandle,
}

struct MockBindGroupLayout {
    device_handle: DeviceHandle,
}

struct MockBindGroup {
    device_handle: DeviceHandle,
}

struct MockRenderTarget {
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    has_rendered: bool,
    /// Simulated pixel data (all zeros by default)
    data: Vec<u8>,
}

struct MockSurface {
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    next_image: SwapchainImageHandle,
}

impl MockBackend {
    /// Create a new mock backend with one simulated adapter.
    pub fn new() -> Self {
        Self {
            adapters: vec![
                AdapterInfo {
                    id: 0,
                    name: "Mock GPU".to_string(),
                    vendor: "RAG Test".to_string(),
                    backend: BackendType::Vulkan, // Pretend to be Vulkan
                    device_type: DeviceType::DiscreteGpu,
                }
            ],
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
            render_targets: HashMap::new(),
            next_render_target_handle: 1,
            surfaces: HashMap::new(),
            next_surface_handle: 1,
            recorded_commands: Vec::new(),
            targets_created: Vec::new(),
            cpu_readback_count: 0,
            surface_present_count: 0,
        }
    }

    /// Reset recorded state for a new test.
    pub fn reset_tracking(&mut self) {
        self.recorded_commands.clear();
        self.targets_created.clear();
        self.cpu_readback_count = 0;
        self.surface_present_count = 0;
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for MockBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Vulkan
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        self.adapters.clone()
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        if adapter_id as usize >= self.adapters.len() {
            anyhow::bail!("Invalid adapter id: {}", adapter_id);
        }

        let handle = self.next_device_handle;
        self.next_device_handle += 1;

        self.devices.insert(handle, MockDevice { adapter_id });
        Ok(handle)
    }

    fn destroy_device(&mut self, device: DeviceHandle) {
        self.devices.remove(&device);
        
        // Clean up resources owned by this device
        self.buffers.retain(|_, b| b.device_handle != device);
        self.shaders.retain(|_, s| s.device_handle != device);
        self.pipelines.retain(|_, p| p.device_handle != device);
        self.bind_group_layouts.retain(|_, l| l.device_handle != device);
        self.bind_groups.retain(|_, g| g.device_handle != device);
        self.render_targets.retain(|_, t| t.device_handle != device);
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn create_buffer(&mut self, device: DeviceHandle, size: u64, _usage: BufferUsage) -> Result<BufferHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_buffer_handle;
        self.next_buffer_handle += 1;

        self.buffers.insert(handle, MockBuffer {
            device_handle: device,
            size,
            data: vec![0u8; size as usize],
        });

        Ok(handle)
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        self.buffers.remove(&buffer);
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buf = self.buffers.get_mut(&buffer)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer handle"))?;

        let start = offset as usize;
        let end = start + data.len();
        if end > buf.data.len() {
            anyhow::bail!("Write exceeds buffer size");
        }

        buf.data[start..end].copy_from_slice(data);
        Ok(())
    }

    fn buffer_size(&self, buffer: BufferHandle) -> u64 {
        self.buffers.get(&buffer).map(|b| b.size).unwrap_or(0)
    }

    fn create_shader(&mut self, device: DeviceHandle, slang_source: &str) -> Result<ShaderHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_shader_handle;
        self.next_shader_handle += 1;

        self.shaders.insert(handle, MockShader {
            device_handle: device,
            source: slang_source.to_string(),
        });

        Ok(handle)
    }

    fn destroy_shader(&mut self, shader: ShaderHandle) {
        self.shaders.remove(&shader);
    }

    fn create_bind_group_layout(&mut self, device: DeviceHandle, _entries: &[BindGroupLayoutEntry]) -> Result<BindGroupLayoutHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_bind_group_layout_handle;
        self.next_bind_group_layout_handle += 1;

        self.bind_group_layouts.insert(handle, MockBindGroupLayout {
            device_handle: device,
        });

        Ok(handle)
    }

    fn create_bind_group(&mut self, device: DeviceHandle, _layout: BindGroupLayoutHandle, _entries: &[BindGroupEntry]) -> Result<BindGroupHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_bind_group_handle;
        self.next_bind_group_handle += 1;

        self.bind_groups.insert(handle, MockBindGroup {
            device_handle: device,
        });

        Ok(handle)
    }

    fn destroy_bind_group(&mut self, bind_group: BindGroupHandle) {
        self.bind_groups.remove(&bind_group);
    }

    fn create_pipeline(
        &mut self,
        device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        _target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_pipeline_handle;
        self.next_pipeline_handle += 1;

        self.pipelines.insert(handle, MockPipeline {
            device_handle: device,
        });

        Ok(handle)
    }

    fn create_pipeline_with_layout(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        _bind_group_layouts: &[BindGroupLayoutHandle],
    ) -> Result<PipelineHandle> {
        self.create_pipeline(device, vertex_shader, fragment_shader, vertex_layout, topology, target_format)
    }

    fn destroy_pipeline(&mut self, pipeline: PipelineHandle) {
        self.pipelines.remove(&pipeline);
    }

    // Legacy frame rendering (for backward compatibility)
    fn begin_frame(&mut self, device: DeviceHandle, _width: u32, _height: u32, _format: TextureFormat) -> Result<()> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        Ok(())
    }

    fn execute_commands(&mut self, device: DeviceHandle, commands: &[RenderCommand]) -> Result<()> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        self.recorded_commands.push(commands.to_vec());
        Ok(())
    }

    fn end_frame(&mut self, device: DeviceHandle, output: &mut [u8]) -> Result<()> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }
        // Fill with a test pattern (gray)
        for byte in output.iter_mut() {
            *byte = 128;
        }
        self.cpu_readback_count += 1;
        Ok(())
    }

    // New RenderTarget API
    fn create_render_target(&mut self, device: DeviceHandle, width: u32, height: u32, format: TextureFormat) -> Result<RenderTargetHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_render_target_handle;
        self.next_render_target_handle += 1;

        let size = (width * height * format.bytes_per_pixel()) as usize;
        self.render_targets.insert(handle, MockRenderTarget {
            device_handle: device,
            width,
            height,
            format,
            has_rendered: false,
            data: vec![0u8; size],
        });

        self.targets_created.push((width, height, format));

        Ok(handle)
    }

    fn destroy_render_target(&mut self, target: RenderTargetHandle) {
        self.render_targets.remove(&target);
    }

    fn render_to_target(&mut self, device: DeviceHandle, target: RenderTargetHandle, commands: &[RenderCommand]) -> Result<()> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let render_target = self.render_targets.get_mut(&target)
            .ok_or_else(|| anyhow::anyhow!("Invalid render target handle"))?;

        if render_target.device_handle != device {
            anyhow::bail!("Render target belongs to a different device");
        }

        // Record commands
        self.recorded_commands.push(commands.to_vec());

        // Simulate rendering by filling with a pattern based on clear color
        let clear_color = commands
            .iter()
            .find_map(|c| match c {
                RenderCommand::Clear(color) => Some(*color),
                _ => None,
            })
            .unwrap_or(Color::BLACK);

        // Fill with the clear color
        let r = (clear_color.r * 255.0) as u8;
        let g = (clear_color.g * 255.0) as u8;
        let b = (clear_color.b * 255.0) as u8;
        let a = (clear_color.a * 255.0) as u8;

        for i in (0..render_target.data.len()).step_by(4) {
            if i + 3 < render_target.data.len() {
                render_target.data[i] = r;
                render_target.data[i + 1] = g;
                render_target.data[i + 2] = b;
                render_target.data[i + 3] = a;
            }
        }

        render_target.has_rendered = true;

        Ok(())
    }

    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()> {
        let render_target = self.render_targets.get(&target)
            .ok_or_else(|| anyhow::anyhow!("Invalid render target handle"))?;

        if !render_target.has_rendered {
            anyhow::bail!("Cannot read from render target that hasn't been rendered to");
        }

        let expected_size = render_target.data.len();
        if output.len() < expected_size {
            anyhow::bail!(
                "Output buffer too small: {} < {}",
                output.len(),
                expected_size
            );
        }

        output[..expected_size].copy_from_slice(&render_target.data);
        self.cpu_readback_count += 1;

        Ok(())
    }

    // Surface API (mock implementation)
    fn create_surface(
        &mut self,
        device: DeviceHandle,
        _window: &dyn raw_window_handle::HasWindowHandle,
        _display: &dyn raw_window_handle::HasDisplayHandle,
    ) -> Result<SurfaceHandle> {
        if !self.devices.contains_key(&device) {
            anyhow::bail!("Invalid device handle");
        }

        let handle = self.next_surface_handle;
        self.next_surface_handle += 1;

        self.surfaces.insert(handle, MockSurface {
            device_handle: device,
            width: 800,  // Default size
            height: 600,
            next_image: 1,
        });

        Ok(handle)
    }

    fn destroy_surface(&mut self, surface: SurfaceHandle) {
        self.surfaces.remove(&surface);
    }

    fn surface_acquire(&mut self, surface: SurfaceHandle) -> Result<SwapchainImageHandle> {
        let surf = self.surfaces.get_mut(&surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?;

        let image = surf.next_image;
        surf.next_image += 1;
        Ok(image)
    }

    fn surface_render(&mut self, surface: SurfaceHandle, _image: SwapchainImageHandle, commands: &[RenderCommand]) -> Result<()> {
        if !self.surfaces.contains_key(&surface) {
            anyhow::bail!("Invalid surface handle");
        }

        self.recorded_commands.push(commands.to_vec());
        Ok(())
    }

    fn surface_present(&mut self, surface: SurfaceHandle, _image: SwapchainImageHandle) -> Result<()> {
        if !self.surfaces.contains_key(&surface) {
            anyhow::bail!("Invalid surface handle");
        }

        self.surface_present_count += 1;
        Ok(())
    }

    fn surface_resize(&mut self, surface: SurfaceHandle, width: u32, height: u32) -> Result<()> {
        let surf = self.surfaces.get_mut(&surface)
            .ok_or_else(|| anyhow::anyhow!("Invalid surface handle"))?;

        surf.width = width;
        surf.height = height;
        Ok(())
    }

    fn surface_size(&self, surface: SurfaceHandle) -> (u32, u32) {
        self.surfaces.get(&surface)
            .map(|s| (s.width, s.height))
            .unwrap_or((0, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_backend_creation() {
        let backend = MockBackend::new();
        assert_eq!(backend.enumerate_adapters().len(), 1);
        assert_eq!(backend.enumerate_adapters()[0].name, "Mock GPU");
    }

    #[test]
    fn test_device_creation() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        assert!(backend.is_device_valid(device));
        
        backend.destroy_device(device);
        assert!(!backend.is_device_valid(device));
    }

    #[test]
    fn test_render_target_creation() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        
        let target = backend.create_render_target(device, 800, 600, TextureFormat::Rgba8Unorm).unwrap();
        
        assert_eq!(backend.targets_created.len(), 1);
        assert_eq!(backend.targets_created[0], (800, 600, TextureFormat::Rgba8Unorm));
        
        backend.destroy_render_target(target);
    }

    #[test]
    fn test_render_without_readback() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend.create_render_target(device, 100, 100, TextureFormat::Rgba8Unorm).unwrap();
        
        let commands = vec![RenderCommand::Clear(Color::RED)];
        backend.render_to_target(device, target, &commands).unwrap();
        
        // No CPU readback should have occurred
        assert_eq!(backend.cpu_readback_count, 0);
        
        // Commands should be recorded
        assert_eq!(backend.recorded_commands.len(), 1);
    }

    #[test]
    fn test_explicit_readback() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend.create_render_target(device, 2, 2, TextureFormat::Rgba8Unorm).unwrap();
        
        let commands = vec![RenderCommand::Clear(Color::RED)];
        backend.render_to_target(device, target, &commands).unwrap();
        
        // Now explicitly read back
        let mut output = vec![0u8; 2 * 2 * 4];
        backend.read_target_to_cpu(target, &mut output).unwrap();
        
        assert_eq!(backend.cpu_readback_count, 1);
        
        // Check the clear color was applied (RED = 255, 0, 0, 255)
        assert_eq!(output[0], 255); // R
        assert_eq!(output[1], 0);   // G
        assert_eq!(output[2], 0);   // B
        assert_eq!(output[3], 255); // A
    }

    #[test]
    fn test_readback_requires_render() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend.create_render_target(device, 10, 10, TextureFormat::Rgba8Unorm).unwrap();
        
        // Try to read without rendering first
        let mut output = vec![0u8; 10 * 10 * 4];
        let result = backend.read_target_to_cpu(target, &mut output);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_renders_same_target() {
        let mut backend = MockBackend::new();
        let device = backend.create_device(0).unwrap();
        let target = backend.create_render_target(device, 10, 10, TextureFormat::Rgba8Unorm).unwrap();
        
        // Render multiple times to the same target
        backend.render_to_target(device, target, &[RenderCommand::Clear(Color::RED)]).unwrap();
        backend.render_to_target(device, target, &[RenderCommand::Clear(Color::GREEN)]).unwrap();
        backend.render_to_target(device, target, &[RenderCommand::Clear(Color::BLUE)]).unwrap();
        
        assert_eq!(backend.recorded_commands.len(), 3);
        
        // Only one target was created
        assert_eq!(backend.targets_created.len(), 1);
    }
}

