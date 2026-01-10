//! Digital Clock demo - using SHARED rendering code from rag::examples::digital_clock
//!
//! This demonstrates that the same vertex generation and shader code
//! works on both native (Vulkan) and web (WebGPU) platforms.
//!
//! Requires Slang shader compiled via slang-wasm in JavaScript.
//! The compiled shader is passed to create_digital_clock_demo().

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::{WebRenderer, get_canvas, init};

// Import shared code from RAG (vertex generation, clock state, etc.)
use rag::examples::digital_clock::{
    ClockVertex, ClockState, TimeData,
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

/// Create digital clock demo with Slang-compiled shader from JavaScript
#[wasm_bindgen]
pub async fn create_digital_clock_demo(canvas_id: &str, compiled_shader: &str) -> Result<DigitalClockDemo, JsValue> {
    init();
    
    let canvas = get_canvas(canvas_id).map_err(|e| e.as_string().unwrap_or_default())?;
    let width = canvas.width();
    let height = canvas.height();
    
    let renderer = WebRenderer::new(canvas).await
        .map_err(|e| JsValue::from_str(&e))?;

    let device = renderer.device();
    let format = renderer.format();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Clock Shader"),
        source: wgpu::ShaderSource::Wgsl(compiled_shader.into()),
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
        label: Some("Clock Pipeline"),
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
    /// Render one frame
    #[wasm_bindgen]
    pub fn render(&self) -> Result<(), JsValue> {
        // Get current time from browser
        let date = js_sys::Date::new_0();
        let hours = date.get_hours();
        let minutes = date.get_minutes();
        let seconds = date.get_seconds();
        
        let time_data = TimeData {
            hours: hours as u8,
            minutes: minutes as u8,
            seconds: seconds as u8,
        };
        let color = self.clock_state.color();
        
        // Generate vertices using SHARED function from rag::examples::digital_clock
        let vertices = generate_clock_vertices(time_data, color, self.width, self.height);
        
        if vertices.is_empty() {
            return Ok(());
        }
        
        // Create vertex buffer for this frame
        let vertex_buffer = self.renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clock Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
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
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }),
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
    
    /// Toggle pause state
    #[wasm_bindgen]
    pub fn toggle_pause(&mut self) {
        self.clock_state.paused = !self.clock_state.paused;
    }
    
    /// Change to next color
    #[wasm_bindgen]
    pub fn change_color(&mut self) {
        self.clock_state.next_color();
    }
}
