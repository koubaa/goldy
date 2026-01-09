//! Frame output for rendering results.

use crate::backend::{GpuBackend, RenderCommand};
use crate::device::Device;
use crate::encoder::CommandEncoder;
use crate::types::*;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Frame output for offscreen rendering.
///
/// Use this to render to a CPU-accessible buffer.
pub struct FrameOutput {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    device_handle: u64,
    width: u32,
    height: u32,
    format: TextureFormat,
}

impl FrameOutput {
    /// Create a new frame output.
    pub fn new(device: &Device, width: u32, height: u32, format: TextureFormat) -> Self {
        Self {
            backend: Arc::clone(&device.backend),
            device_handle: device.handle,
            width,
            height,
            format,
        }
    }

    /// Get the width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the texture format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Get the size of the output buffer in bytes.
    pub fn buffer_size(&self) -> usize {
        (self.width * self.height * self.format.bytes_per_pixel()) as usize
    }

    /// Render commands and copy result to the output buffer.
    pub fn render(&self, encoder: CommandEncoder) -> Result<Vec<u8>> {
        let commands = encoder.finish();
        let mut output = vec![0u8; self.buffer_size()];
        self.render_to_buffer(&commands, &mut output)?;
        Ok(output)
    }

    /// Render commands to an existing buffer.
    pub fn render_to_buffer(&self, commands: &[RenderCommand], output: &mut [u8]) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        
        backend.begin_frame(self.device_handle, self.width, self.height, self.format)?;
        backend.execute_commands(self.device_handle, commands)?;
        backend.end_frame(self.device_handle, output)?;
        
        Ok(())
    }
}

