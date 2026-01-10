//! Mandelbrot set explorer demo
//!
//! Requires Slang shader compiled via slang-wasm in JavaScript.
//! The compiled shader is passed to create_mandelbrot_demo().

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::{WebRenderer, get_canvas, init, types};

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

/// Create mandelbrot demo with Slang-compiled shader from JavaScript
#[wasm_bindgen]
pub async fn create_mandelbrot_demo(canvas_id: &str, compiled_shader: &str) -> Result<MandelbrotDemo, JsValue> {
    init();
    
    let canvas = get_canvas(canvas_id).map_err(|e| e.as_string().unwrap_or_default())?;
    let renderer = WebRenderer::new(canvas).await
        .map_err(|e| JsValue::from_str(&e))?;

    let device = renderer.device();
    let format = renderer.format();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mandelbrot Shader"),
        source: wgpu::ShaderSource::Wgsl(compiled_shader.into()),
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
