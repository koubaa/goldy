//! Particle rain/snow demo with toggle
//!
//! Requires Slang shader compiled via slang-wasm in JavaScript.
//! The compiled shader is passed to create_particles_demo().

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::{WebRenderer, get_canvas, init, types};

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

/// Create particles demo with Slang-compiled shader from JavaScript
#[wasm_bindgen]
pub async fn create_particles_demo(canvas_id: &str, compiled_shader: &str) -> Result<ParticlesDemo, JsValue> {
    init();
    
    let canvas = get_canvas(canvas_id).map_err(|e| e.as_string().unwrap_or_default())?;
    let renderer = WebRenderer::new(canvas).await
        .map_err(|e| JsValue::from_str(&e))?;

    let device = renderer.device();
    let format = renderer.format();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Particles Shader"),
        source: wgpu::ShaderSource::Wgsl(compiled_shader.into()),
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
