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
    /// Create a shader module from Slang source.
    ///
    /// The Slang source should contain entry points marked with `[shader("vertex")]`,
    /// `[shader("fragment")]`, etc.
    ///
    /// # Example
    ///
    /// ```slang
    /// [shader("vertex")]
    /// float4 vs_main(float2 pos : POSITION) : SV_Position {
    ///     return float4(pos, 0, 1);
    /// }
    ///
    /// [shader("fragment")]
    /// float4 fs_main() : SV_Target {
    ///     return float4(1, 0, 0, 1);
    /// }
    /// ```
    pub fn from_slang(device: &Device, source: &str) -> Result<Self> {
        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_shader(device.handle, source)?;
        
        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
        })
    }
    
    /// Create a shader module from WGSL source.
    ///
    /// **Deprecated**: Use `from_slang` instead. This method exists for backward
    /// compatibility and will compile WGSL through Slang.
    #[deprecated(since = "0.2.0", note = "Use from_slang() with Slang shader syntax")]
    pub fn from_wgsl(device: &Device, source: &str) -> Result<Self> {
        // For now, pass through to Slang - WGSL is not directly supported
        // Users should migrate to Slang syntax
        Self::from_slang(device, source)
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_shader(self.handle);
    }
}

/// Built-in shaders for common use cases.
///
/// All shaders are written in Slang (HLSL-like syntax).
pub mod builtins {
    /// Simple 2D vertex + fragment shader for colored vertices.
    pub const VERTEX_COLOR_2D: &str = r#"
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

    /// Simple solid color fragment shader.
    pub const SOLID_COLOR: &str = r#"
struct VertexInput {
    float2 position : POSITION;
};

struct VertexOutput {
    float4 position : SV_Position;
};

cbuffer Uniforms {
    float4 color;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return color;
}
"#;
}
