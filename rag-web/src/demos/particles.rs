//! Particle rain/snow demo with toggle

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::{WebRenderer, get_canvas, init, types};

const PARTICLES_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Params {
    time: f32,
    is_snow: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0)
var<uniform> params: Params;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fn hash(p: f32) -> f32 {
    return fract(sin(p * 127.1) * 43758.5453);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let is_snow = params.is_snow > 0.5;
    
    var bg_color: vec3<f32>;
    var particle_color: vec3<f32>;
    var trail_color: vec3<f32>;
    
    if (is_snow) {
        bg_color = vec3<f32>(0.02, 0.02, 0.04);
        particle_color = vec3<f32>(0.9, 0.95, 1.0);
        trail_color = vec3<f32>(0.5, 0.55, 0.7);
    } else {
        bg_color = vec3<f32>(0.01, 0.01, 0.02);
        particle_color = vec3<f32>(0.3, 0.6, 1.0);
        trail_color = vec3<f32>(0.2, 0.4, 0.8);
    }
    
    var color = bg_color;
    
    for (var i = 0; i < 60; i += 1) {
        let fi = f32(i);
        var x = hash(fi);
        
        // Snow drifts horizontally
        if (is_snow) {
            x += sin(params.time * 0.5 + fi * 0.3) * 0.05;
            x = fract(x);
        }
        
        // Snow is slower
        let base_speed = select(0.3, 0.12, is_snow);
        let speed = base_speed + hash(fi + 100.0) * select(0.4, 0.08, is_snow);
        let phase = hash(fi + 200.0) * 10.0;
        
        let t = fract(params.time * speed + phase);
        let particle_y = t;
        
        let particle_pos = vec2<f32>(x, particle_y);
        let dist = length(uv - particle_pos);
        
        // Snow has bigger particles
        let base_size = select(0.002, 0.004, is_snow);
        let size = base_size + hash(fi + 300.0) * select(0.002, 0.004, is_snow);
        let brightness = size / (dist + 0.001);
        let fade = smoothstep(0.0, 0.15, t) * smoothstep(1.0, 0.85, t);
        
        color += particle_color * brightness * fade * 0.06;
        
        // Trail (shorter for snow)
        let trail_len = select(0.08, 0.03, is_snow);
        let trail_top = particle_y - trail_len;
        
        if (uv.x > x - 0.003 && uv.x < x + 0.003 && uv.y > trail_top && uv.y < particle_y) {
            let trail_t = (uv.y - trail_top) / trail_len;
            color += trail_color * trail_t * fade * 0.2;
        }
    }
    
    return vec4<f32>(color, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    time: f32,
    is_snow: f32,
    _pad1: f32,
    _pad2: f32,
}

#[wasm_bindgen]
pub struct ParticlesDemo {
    renderer: WebRenderer,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    start_time: f64,
    is_snow: bool,
}

#[wasm_bindgen]
pub async fn create_particles_demo(canvas_id: &str) -> Result<ParticlesDemo, JsValue> {
    init();
    
    let canvas = get_canvas(canvas_id)?;
    let renderer = WebRenderer::new(canvas).await
        .map_err(|e| JsValue::from_str(&e))?;

    let device = renderer.device();
    let format = renderer.format();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Particles Shader"),
        source: wgpu::ShaderSource::Wgsl(PARTICLES_SHADER.into()),
    });

    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Params Buffer"),
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
            resource: params_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Particles Pipeline"),
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

    Ok(ParticlesDemo {
        renderer,
        pipeline,
        vertex_buffer,
        params_buffer,
        bind_group,
        start_time,
        is_snow: false,
    })
}

#[wasm_bindgen]
impl ParticlesDemo {
    #[wasm_bindgen]
    pub fn toggle_mode(&mut self) {
        self.is_snow = !self.is_snow;
    }

    #[wasm_bindgen]
    pub fn render(&self) -> Result<(), JsValue> {
        let window = web_sys::window().unwrap();
        let now = window.performance().unwrap().now();
        let time = ((now - self.start_time) / 1000.0) as f32;

        let params = Params {
            time,
            is_snow: if self.is_snow { 1.0 } else { 0.0 },
            _pad1: 0.0,
            _pad2: 0.0,
        };

        self.renderer.queue().write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
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
