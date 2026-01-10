//! RAG-Web: WebGPU backend for RAG
//!
//! This crate provides a browser-compatible implementation of RAG
//! using the WebGPU API.
//!
//! ## Slang Shader Sources
//!
//! All shader sources are exposed via `get_*_shader_source()` functions.
//! JavaScript should compile these with slang-wasm at runtime.

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

pub mod types;
pub mod render;
pub mod demos;

pub use types::*;
pub use render::*;
pub use demos::*;

// ============================================================================
// Slang Shader Source Exports
// ============================================================================
// These functions expose RAG's Slang shader sources to JavaScript.
// JavaScript uses slang-wasm to compile these to WGSL at runtime.

/// Get the vertex_color_2d shader source (Slang)
#[wasm_bindgen]
pub fn get_vertex_color_2d_shader() -> String {
    rag::shaders::VERTEX_COLOR_2D.to_string()
}

/// Get the triangle shader source (Slang)
#[wasm_bindgen]
pub fn get_triangle_shader() -> String {
    rag::shaders::TRIANGLE.to_string()
}

/// Get the digital clock shader source (Slang)
#[wasm_bindgen]
pub fn get_digital_clock_shader() -> String {
    rag::shaders::DIGITAL_CLOCK.to_string()
}

/// Get the plasma shader source (Slang)
#[wasm_bindgen]
pub fn get_plasma_shader() -> String {
    rag::shaders::PLASMA.to_string()
}

/// Get the gradient shader source (Slang)
#[wasm_bindgen]
pub fn get_gradient_shader() -> String {
    rag::shaders::GRADIENT.to_string()
}

/// Get the mandelbrot shader source (Slang)
#[wasm_bindgen]
pub fn get_mandelbrot_shader() -> String {
    rag::shaders::MANDELBROT.to_string()
}

/// Get the tunnel shader source (Slang)
#[wasm_bindgen]
pub fn get_tunnel_shader() -> String {
    rag::shaders::TUNNEL.to_string()
}

/// Get the starfield shader source (Slang)
#[wasm_bindgen]
pub fn get_starfield_shader() -> String {
    rag::shaders::STARFIELD.to_string()
}

/// Get the particles shader source (Slang)
#[wasm_bindgen]
pub fn get_particles_shader() -> String {
    rag::shaders::PARTICLES.to_string()
}

/// Get the spinning cube shader source (Slang)
#[wasm_bindgen]
pub fn get_spinning_cube_shader() -> String {
    rag::shaders::SPINNING_CUBE.to_string()
}

/// Get the metaballs shader source (Slang)
#[wasm_bindgen]
pub fn get_metaballs_shader() -> String {
    rag::shaders::METABALLS.to_string()
}

/// Get the checkerboard shader source (Slang)
#[wasm_bindgen]
pub fn get_checkerboard_shader() -> String {
    rag::shaders::CHECKERBOARD.to_string()
}

/// Initialize panic hook and logging for better error messages
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

/// Get a canvas element by ID
pub fn get_canvas(id: &str) -> Result<HtmlCanvasElement, JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let canvas = document
        .get_element_by_id(id)
        .ok_or_else(|| format!("no element with id '{}'", id))?;
    canvas
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| "element is not a canvas".into())
}

/// WebGPU-backed renderer for browser use
pub struct WebRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    size: (u32, u32),
}

impl WebRenderer {
    /// Create a new WebRenderer attached to a canvas
    pub async fn new(canvas: HtmlCanvasElement) -> Result<Self, String> {
        let width = canvas.client_width() as u32;
        let height = canvas.client_height() as u32;
        
        // Set canvas resolution to match display size
        canvas.set_width(width);
        canvas.set_height(height);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        // For web, use Canvas variant directly
        #[cfg(target_arch = "wasm32")]
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("Failed to create surface: {}", e))?;
        
        #[cfg(not(target_arch = "wasm32"))]
        let surface: wgpu::Surface<'static> = unreachable!("This code only runs on wasm32");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to get adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| format!("Failed to get device: {}", e))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            size: (width, height),
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.size = (width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Get the current surface texture for rendering
    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}

