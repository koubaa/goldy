//! Demoscene tunnel effect

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::{WebRenderer, get_canvas, init, types};

const TUNNEL_SHADER: &str = r#"
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = (in.uv - 0.5) * 2.0;
    
    // Convert to polar coordinates
    let angle = atan2(uv.y, uv.x);
    let radius = length(uv);
    
    // Tunnel effect - depth mapped from radius
    let depth = 1.0 / (radius + 0.1);
    
    // Texture coordinates
    let tx = angle / 3.14159 + time * 0.2;
    let ty = depth + time * 0.5;
    
    // Create checkerboard pattern
    let checker = step(0.5, fract(tx * 4.0)) * step(0.5, fract(ty * 2.0)) +
                  step(0.5, 1.0 - fract(tx * 4.0)) * step(0.5, 1.0 - fract(ty * 2.0));
    
    // Color gradient based on depth
    let color1 = vec3<f32>(0.8, 0.2, 0.4);
    let color2 = vec3<f32>(0.2, 0.4, 0.8);
    let base_color = mix(color1, color2, sin(ty * 0.5 + time) * 0.5 + 0.5);
    
    // Apply fog based on depth
    let fog = exp(-depth * 0.3);
    var color = base_color * checker * fog;
    
    // Add glow at center
    let center_glow = 0.1 / (radius + 0.1);
    color += vec3<f32>(1.0, 0.8, 0.6) * center_glow * 0.2;
    
    // Vignette
    let vignette = 1.0 - radius * 0.3;
    color *= vignette;
    
    return vec4<f32>(color, 1.0);
}
"#;

#[wasm_bindgen]
pub struct TunnelDemo {
    renderer: WebRenderer,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    time_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    start_time: f64,
}

#[wasm_bindgen]
pub async fn create_tunnel_demo(canvas_id: &str) -> Result<TunnelDemo, JsValue> {
    init();
    
    let canvas = get_canvas(canvas_id)?;
    let renderer = WebRenderer::new(canvas).await
        .map_err(|e| JsValue::from_str(&e))?;

    let device = renderer.device();
    let format = renderer.format();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Tunnel Shader"),
        source: wgpu::ShaderSource::Wgsl(TUNNEL_SHADER.into()),
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
        label: Some("Tunnel Pipeline"),
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

    Ok(TunnelDemo {
        renderer,
        pipeline,
        vertex_buffer,
        time_buffer,
        bind_group,
        start_time,
    })
}

#[wasm_bindgen]
impl TunnelDemo {
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

