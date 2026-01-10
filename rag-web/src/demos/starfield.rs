//! 3D Starfield - flying forward through space
//! Single uniform speed for all stars

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::{WebRenderer, get_canvas, init, types};

const STARFIELD_SHADER: &str = r#"
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

fn hash(p: f32) -> f32 {
    return fract(sin(p * 127.1) * 43758.5453);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = (in.uv - 0.5) * 2.0;
    var color = vec3<f32>(0.0, 0.0, 0.02);
    
    let num_stars = 200;
    let speed = 0.3;  // SINGLE speed for all stars
    
    for (var i = 0; i < num_stars; i += 1) {
        let fi = f32(i);
        
        // Random angle for each star
        let angle = hash(fi) * 6.28318;
        // Random max distance (how far from center it can go)
        let max_dist = 0.3 + hash(fi + 50.0) * 1.2;
        
        // All stars cycle at same speed, just different phases
        let phase = hash(fi + 100.0);
        let cycle = fract(time * speed + phase);
        
        // Distance from center increases as cycle goes 0->1
        let dist = max_dist * cycle;
        
        let star_x = cos(angle) * dist;
        let star_y = sin(angle) * dist;
        
        let pixel_dist = length(uv - vec2<f32>(star_x, star_y));
        
        // Size increases with distance (closer = bigger)
        let size = 0.002 + cycle * 0.015;
        
        // Brightness increases with distance
        let brightness = cycle * smoothstep(size, 0.0, pixel_dist);
        
        // Fade in at spawn, no fade out (stars just disappear at edge)
        let fade = smoothstep(0.0, 0.1, cycle);
        
        color += vec3<f32>(0.9, 0.95, 1.0) * brightness * fade;
    }
    
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

#[wasm_bindgen]
pub struct StarfieldDemo {
    renderer: WebRenderer,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    time_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    start_time: f64,
}

#[wasm_bindgen]
pub async fn create_starfield_demo(canvas_id: &str) -> Result<StarfieldDemo, JsValue> {
    init();
    
    let canvas = get_canvas(canvas_id)?;
    let renderer = WebRenderer::new(canvas).await
        .map_err(|e| JsValue::from_str(&e))?;

    let device = renderer.device();
    let format = renderer.format();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Starfield Shader"),
        source: wgpu::ShaderSource::Wgsl(STARFIELD_SHADER.into()),
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
        label: Some("Starfield Pipeline"),
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

    Ok(StarfieldDemo {
        renderer,
        pipeline,
        vertex_buffer,
        time_buffer,
        bind_group,
        start_time,
    })
}

#[wasm_bindgen]
impl StarfieldDemo {
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
