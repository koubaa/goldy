//! Digital Clock demo - using SHARED rendering code from rag::examples::digital_clock
//!
//! This demonstrates that the same vertex generation and shader code
//! works on both native (Vulkan) and web (WebGPU) platforms.

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::{WebRenderer, get_canvas, init};

// Import shared code from RAG
use rag::examples::digital_clock::{
    ClockVertex, ClockState, TimeData, SHADER_SOURCE,
    generate_clock_vertices,
};

#[wasm_bindgen]
pub struct DigitalClockDemo {
    renderer: WebRenderer,
    pipeline: wgpu::RenderPipeline,
    start_time: f64,
    clock_state: ClockState,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
pub async fn create_digital_clock_demo(canvas_id: &str) -> Result<DigitalClockDemo, JsValue> {
    init();
    
    let canvas = get_canvas(canvas_id)?;
    let width = canvas.width();
    let height = canvas.height();
    
    let renderer = WebRenderer::new(canvas).await
        .map_err(|e| JsValue::from_str(&e))?;

    let device = renderer.device();
    let format = renderer.format();

    // Use the SHARED shader source from rag::examples::digital_clock
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shared Clock Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });

    // Vertex layout matching ClockVertex from shared code
    let vertex_buffer_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ClockVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 8,
                shader_location: 1,
            },
        ],
    };

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Clock Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Shared Clock Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_buffer_layout],
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
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let window = web_sys::window().unwrap();
    let start_time = window.performance().unwrap().now();

    Ok(DigitalClockDemo {
        renderer,
        pipeline,
        start_time,
        clock_state: ClockState::default(),
        width,
        height,
    })
}

#[wasm_bindgen]
impl DigitalClockDemo {
    #[wasm_bindgen]
    pub fn toggle_pause(&mut self) {
        let current = self.elapsed_secs();
        if self.clock_state.paused {
            // Resuming - reset start time
            let window = web_sys::window().unwrap();
            self.start_time = window.performance().unwrap().now();
        }
        self.clock_state.toggle_pause(current);
    }
    
    #[wasm_bindgen]
    pub fn change_color(&mut self) {
        self.clock_state.next_color();
    }

    fn elapsed_secs(&self) -> u64 {
        if self.clock_state.paused {
            self.clock_state.accumulated_secs
        } else {
            let window = web_sys::window().unwrap();
            let now = window.performance().unwrap().now();
            let elapsed_ms = now - self.start_time;
            (elapsed_ms / 1000.0) as u64 + self.clock_state.accumulated_secs
        }
    }

    #[wasm_bindgen]
    pub fn render(&self) -> Result<(), JsValue> {
        let elapsed = self.elapsed_secs();
        let time = TimeData::from_elapsed_secs(elapsed);
        let color = self.clock_state.color();
        let bg_color = self.clock_state.background_color();

        // Generate vertices using SHARED function from rag::examples::digital_clock
        let vertices = generate_clock_vertices(time, color, self.width, self.height);
        
        if vertices.is_empty() {
            return Ok(());
        }

        // Create vertex buffer with generated vertices
        let vertex_buffer = self.renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clock Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let output = self.renderer.get_current_texture()
            .map_err(|e| JsValue::from_str(&format!("Surface error: {:?}", e)))?;
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.renderer.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Clock Encoder") }
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clock Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg_color.r as f64,
                            g: bg_color.g as f64,
                            b: bg_color.b as f64,
                            a: bg_color.a as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..vertices.len() as u32, 0..1);
        }

        self.renderer.queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
