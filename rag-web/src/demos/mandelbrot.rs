//! Mandelbrot set explorer demo
//!
//! Supports two modes:
//! 1. Embedded WGSL shader (fallback)
//! 2. Slang-compiled WGSL passed from JavaScript

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::{WebRenderer, get_canvas, init, types};

// Fallback WGSL shader
const MANDELBROT_SHADER_FALLBACK: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Uniforms {
    center: vec2<f32>,
    zoom: f32,
    max_iter: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs(((h * 6.0) % 2.0) - 1.0));
    let m = v - c;
    
    var rgb: vec3<f32>;
    let hi = u32(h * 6.0) % 6u;
    switch hi {
        case 0u: { rgb = vec3<f32>(c, x, 0.0); }
        case 1u: { rgb = vec3<f32>(x, c, 0.0); }
        case 2u: { rgb = vec3<f32>(0.0, c, x); }
        case 3u: { rgb = vec3<f32>(0.0, x, c); }
        case 4u: { rgb = vec3<f32>(x, 0.0, c); }
        default: { rgb = vec3<f32>(c, 0.0, x); }
    }
    return rgb + vec3<f32>(m);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = 16.0 / 9.0;
    let c = uniforms.center + (in.uv - 0.5) * vec2<f32>(aspect, 1.0) * 3.0 / uniforms.zoom;
    
    var z = vec2<f32>(0.0);
    var iter = 0.0;
    let max_iter = uniforms.max_iter;
    
    for (var i = 0.0; i < max_iter; i += 1.0) {
        if (dot(z, z) > 4.0) { break; }
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iter = i;
    }
    
    if (iter >= max_iter - 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    let smooth_iter = iter - log2(log2(dot(z, z))) + 4.0;
    let hue = smooth_iter / 50.0;
    let color = hsv_to_rgb(hue % 1.0, 0.8, 1.0);
    
    return vec4<f32>(color, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    center: [f32; 2],
    zoom: f32,
    max_iter: f32,
}

#[wasm_bindgen]
pub struct MandelbrotDemo {
    renderer: WebRenderer,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    center: [f32; 2],
    zoom: f32,
    target_zoom: f32,
}

async fn create_mandelbrot_demo_internal(canvas_id: &str, wgsl_source: &str) -> Result<MandelbrotDemo, String> {
    let canvas = get_canvas(canvas_id).map_err(|e| e.as_string().unwrap_or_default())?;
    let renderer = WebRenderer::new(canvas).await?;

    let device = renderer.device();
    let format = renderer.format();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mandelbrot Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });

    let uniforms = Uniforms {
        center: [-0.5, 0.0],
        zoom: 1.0,
        max_iter: 256.0,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&types::FULLSCREEN_QUAD),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Mandelbrot Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[types::Vertex2D::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    Ok(MandelbrotDemo {
        renderer,
        pipeline,
        vertex_buffer,
        uniform_buffer,
        bind_group,
        center: [-0.5, 0.0],
        zoom: 1.0,
        target_zoom: 1.0,
    })
}

/// Create mandelbrot demo with embedded fallback shader
#[wasm_bindgen]
pub async fn create_mandelbrot_demo(canvas_id: &str) -> Result<MandelbrotDemo, JsValue> {
    init();
    create_mandelbrot_demo_internal(canvas_id, MANDELBROT_SHADER_FALLBACK)
        .await
        .map_err(|e| JsValue::from_str(&e))
}

/// Create mandelbrot demo with Slang-compiled WGSL shader from JavaScript
#[wasm_bindgen]
pub async fn create_mandelbrot_demo_with_shader(canvas_id: &str, wgsl_source: &str) -> Result<MandelbrotDemo, JsValue> {
    init();
    create_mandelbrot_demo_internal(canvas_id, wgsl_source)
        .await
        .map_err(|e| JsValue::from_str(&e))
}

#[wasm_bindgen]
impl MandelbrotDemo {
    #[wasm_bindgen]
    pub fn render(&mut self) -> Result<(), JsValue> {
        // Smooth zoom animation
        self.zoom += (self.target_zoom - self.zoom) * 0.05;
        
        let uniforms = Uniforms {
            center: self.center,
            zoom: self.zoom,
            max_iter: 256.0 + self.zoom.log2() * 50.0,
        };
        
        self.renderer.queue().write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        let output = self.renderer.get_current_texture()
            .map_err(|e| JsValue::from_str(&format!("Surface error: {:?}", e)))?;
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.renderer.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") }
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }

        self.renderer.queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn zoom_in(&mut self) {
        self.target_zoom *= 1.5;
    }
    
    #[wasm_bindgen]
    pub fn zoom_out(&mut self) {
        self.target_zoom /= 1.5;
        if self.target_zoom < 0.5 { self.target_zoom = 0.5; }
    }
    
    #[wasm_bindgen]
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let scale = 0.5 / self.zoom;
        self.center[0] += dx * scale;
        self.center[1] -= dy * scale; // Inverted for natural feel
    }
    
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.center = [-0.5, 0.0];
        self.target_zoom = 1.0;
    }
}
