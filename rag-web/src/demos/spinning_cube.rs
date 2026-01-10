//! Spinning 3D wireframe cube
//!
//! Supports two modes:
//! 1. Embedded WGSL shader (fallback)
//! 2. Slang-compiled WGSL passed from JavaScript

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::{WebRenderer, get_canvas, init, types};

const CUBE_SHADER_FALLBACK: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> time: f32;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fn rotate_y(p: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

fn rotate_x(p: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

fn project(p: vec3<f32>) -> vec2<f32> {
    let z = p.z + 4.0;
    let scale = 2.0 / z;
    return vec2<f32>(p.x * scale, p.y * scale);
}

fn line_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = (in.uv - 0.5) * 2.0;
    
    // Cube vertices
    var verts: array<vec3<f32>, 8>;
    verts[0] = vec3<f32>(-1.0, -1.0, -1.0);
    verts[1] = vec3<f32>(1.0, -1.0, -1.0);
    verts[2] = vec3<f32>(1.0, 1.0, -1.0);
    verts[3] = vec3<f32>(-1.0, 1.0, -1.0);
    verts[4] = vec3<f32>(-1.0, -1.0, 1.0);
    verts[5] = vec3<f32>(1.0, -1.0, 1.0);
    verts[6] = vec3<f32>(1.0, 1.0, 1.0);
    verts[7] = vec3<f32>(-1.0, 1.0, 1.0);
    
    // Rotate vertices
    for (var i = 0; i < 8; i += 1) {
        verts[i] = rotate_y(verts[i], time * 0.7);
        verts[i] = rotate_x(verts[i], time * 0.5);
    }
    
    // Project to 2D
    var proj: array<vec2<f32>, 8>;
    for (var i = 0; i < 8; i += 1) {
        proj[i] = project(verts[i]);
    }
    
    // Cube edges
    var min_dist = 1000.0;
    
    // Back face
    min_dist = min(min_dist, line_dist(uv, proj[0], proj[1]));
    min_dist = min(min_dist, line_dist(uv, proj[1], proj[2]));
    min_dist = min(min_dist, line_dist(uv, proj[2], proj[3]));
    min_dist = min(min_dist, line_dist(uv, proj[3], proj[0]));
    
    // Front face
    min_dist = min(min_dist, line_dist(uv, proj[4], proj[5]));
    min_dist = min(min_dist, line_dist(uv, proj[5], proj[6]));
    min_dist = min(min_dist, line_dist(uv, proj[6], proj[7]));
    min_dist = min(min_dist, line_dist(uv, proj[7], proj[4]));
    
    // Connecting edges
    min_dist = min(min_dist, line_dist(uv, proj[0], proj[4]));
    min_dist = min(min_dist, line_dist(uv, proj[1], proj[5]));
    min_dist = min(min_dist, line_dist(uv, proj[2], proj[6]));
    min_dist = min(min_dist, line_dist(uv, proj[3], proj[7]));
    
    // Draw edges with glow
    let edge = smoothstep(0.02, 0.005, min_dist);
    let glow = 0.01 / (min_dist + 0.01);
    
    let edge_color = vec3<f32>(0.3, 0.7, 1.0);
    let glow_color = vec3<f32>(0.1, 0.3, 0.6);
    
    var color = edge_color * edge + glow_color * glow * 0.3;
    color += vec3<f32>(0.02, 0.02, 0.04);  // Background
    
    return vec4<f32>(color, 1.0);
}
"#;

#[wasm_bindgen]
pub struct SpinningCubeDemo {
    renderer: WebRenderer,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    time_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    start_time: f64,
}

async fn create_spinning_cube_demo_internal(canvas_id: &str, wgsl_source: &str) -> Result<SpinningCubeDemo, String> {
    let canvas = get_canvas(canvas_id).map_err(|e| e.as_string().unwrap_or_default())?;
    let renderer = WebRenderer::new(canvas).await?;

    let device = renderer.device();
    let format = renderer.format();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Cube Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });

    let time_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Time Buffer"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
            resource: time_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Cube Pipeline"),
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

    let window = web_sys::window().unwrap();
    let start_time = window.performance().unwrap().now();

    Ok(SpinningCubeDemo {
        renderer,
        pipeline,
        vertex_buffer,
        time_buffer,
        bind_group,
        start_time,
    })
}

/// Create spinning cube demo with embedded fallback shader
#[wasm_bindgen]
pub async fn create_spinning_cube_demo(canvas_id: &str) -> Result<SpinningCubeDemo, JsValue> {
    init();
    create_spinning_cube_demo_internal(canvas_id, CUBE_SHADER_FALLBACK)
        .await
        .map_err(|e| JsValue::from_str(&e))
}

/// Create spinning cube demo with Slang-compiled WGSL shader from JavaScript
#[wasm_bindgen]
pub async fn create_spinning_cube_demo_with_shader(canvas_id: &str, wgsl_source: &str) -> Result<SpinningCubeDemo, JsValue> {
    init();
    create_spinning_cube_demo_internal(canvas_id, wgsl_source)
        .await
        .map_err(|e| JsValue::from_str(&e))
}

#[wasm_bindgen]
impl SpinningCubeDemo {
    #[wasm_bindgen]
    pub fn render(&self) -> Result<(), JsValue> {
        let window = web_sys::window().unwrap();
        let now = window.performance().unwrap().now();
        let time = ((now - self.start_time) / 1000.0) as f32;

        self.renderer.queue().write_buffer(
            &self.time_buffer,
            0,
            bytemuck::cast_slice(&[time]),
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
}

