//! Shader module management.

use crate::backend::{GpuBackend, ShaderHandle};
use crate::device::Device;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A compiled shader module.
pub struct ShaderModule {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: ShaderHandle,
}

impl ShaderModule {
    /// Create a shader module from WGSL source.
    ///
    /// The WGSL source should contain both vertex and fragment entry points
    /// if needed for the pipeline.
    pub fn from_wgsl(device: &Device, source: &str) -> Result<Self> {
        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_shader(device.handle, source)?;
        
        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
        })
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_shader(self.handle);
    }
}

/// Built-in shaders for common use cases.
pub mod builtins {
    /// Simple 2D vertex + fragment shader for colored vertices.
    pub const VERTEX_COLOR_2D: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

    /// Simple solid color fragment shader.
    pub const SOLID_COLOR: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

struct Uniforms {
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return uniforms.color;
}
"#;
}

