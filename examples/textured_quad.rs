//! Textured quad example - demonstrates texture sampling.
//!
//! Creates a procedural checkerboard texture and displays it on a quad.
//!
//! Run with: cargo run --example textured_quad

use goldy::{
    types::{AddressMode, FilterMode, SamplerDesc, SpatialAccess, TextureFlags, TextureFormat},
    Buffer, Color, CommandEncoder, DataAccess, DeviceType, Instance, RenderPipeline,
    RenderPipelineDesc, Sampler, ShaderModule, Surface, Texture, Vertex2DUv,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

// Simple shader that samples a texture
const TEXTURED_SHADER: &str = r#"
struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Cross-platform resource access via push constants
#if defined(__METAL__) && !defined(__METAL_BINDLESS__)
// Metal Tier 1: Traditional binding
Texture2D<float4> texture : register(t0);
SamplerState texSampler : register(s0);
#define GET_TEXTURE() texture
#define GET_SAMPLER() texSampler

#elif defined(__SPIRV__)
// Vulkan: Push constants for indices + global descriptor arrays
struct BufferIndices { uint indices[2]; };
[[vk::push_constant]] ConstantBuffer<BufferIndices> g_Indices;
[[vk::binding(2, 0)]] Texture2D<float4> g_Textures[];
[[vk::binding(4, 0)]] SamplerState g_Samplers[];
#define GET_TEXTURE() g_Textures[g_Indices.indices[0]]
#define GET_SAMPLER() g_Samplers[g_Indices.indices[1]]

#else
// DX12: Root constants + DescriptorHandle
cbuffer BufferIndices : register(b0, space0) {
    uint textureIndex;
    uint samplerIndex;
};
#define GET_TEXTURE() (*DescriptorHandle<Texture2D<float4>>(uint2(textureIndex, 0)))
#define GET_SAMPLER() (*DescriptorHandle<SamplerState>(uint2(samplerIndex, 0)))
#endif

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return GET_TEXTURE().Sample(GET_SAMPLER(), input.uv);
}
"#;

/// Generate a checkerboard texture in RGBA8 format
fn generate_checkerboard(width: u32, height: u32, checker_size: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let checker_x = (x / checker_size) % 2;
            let checker_y = (y / checker_size) % 2;
            let is_white = (checker_x + checker_y) % 2 == 0;

            if is_white {
                data.extend_from_slice(&[255, 255, 255, 255]); // White
            } else {
                data.extend_from_slice(&[50, 100, 200, 255]); // Blue
            }
        }
    }

    data
}

// Fullscreen quad vertices
const QUAD_VERTICES: [Vertex2DUv; 6] = [
    Vertex2DUv {
        position: [-1.0, -1.0],
        uv: [0.0, 1.0],
    },
    Vertex2DUv {
        position: [1.0, -1.0],
        uv: [1.0, 1.0],
    },
    Vertex2DUv {
        position: [1.0, 1.0],
        uv: [1.0, 0.0],
    },
    Vertex2DUv {
        position: [-1.0, -1.0],
        uv: [0.0, 1.0],
    },
    Vertex2DUv {
        position: [1.0, 1.0],
        uv: [1.0, 0.0],
    },
    Vertex2DUv {
        position: [-1.0, 1.0],
        uv: [0.0, 0.0],
    },
];

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    vertex_buffer: Option<Buffer>,
    texture: Option<Texture>,
    sampler: Option<Sampler>,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None,
            pipeline: None,
            shader: None,
            window: None,
            surface: None,
            vertex_buffer: None,
            texture: None,
            sampler: None,
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);
        let surface = Surface::new(&device, window.as_ref())?;

        // Create shader
        let shader = ShaderModule::from_slang(&device, TEXTURED_SHADER)?;

        // Create texture
        let tex_width = 256u32;
        let tex_height = 256u32;
        let checker_data = generate_checkerboard(tex_width, tex_height, 32);
        
        let texture = Texture::with_data(
            &device,
            &checker_data,
            tex_width,
            tex_height,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Interpolated,
            TextureFlags::COPY_DST,
        )?;

        // Create sampler with linear filtering and repeat addressing
        let sampler = Sampler::new(
            &device,
            &SamplerDesc {
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Nearest,
                address_mode_u: AddressMode::Repeat,
                address_mode_v: AddressMode::Repeat,
                address_mode_w: AddressMode::Repeat,
                max_anisotropy: 1.0,
                compare: None,
                lod_min_clamp: 0.0,
                lod_max_clamp: 32.0,
            },
        )?;

        // Create pipeline
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: Vertex2DUv::layout(),
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        // Create vertex buffer
        let vertex_buffer = Buffer::with_data(&device, &QUAD_VERTICES, DataAccess::Scattered)?;

        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);
        self.vertex_buffer = Some(vertex_buffer);
        self.texture = Some(texture);
        self.sampler = Some(sampler);

        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let texture = self.texture.as_ref().unwrap();
        let sampler = self.sampler.as_ref().unwrap();

        // Get bindless indices
        let tex_idx = texture.bindless_index().unwrap_or(0);
        let samp_idx = sampler.bindless_index().unwrap_or(0);

        let frame = surface.acquire()?;

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color {
                r: 0.1,
                g: 0.1,
                b: 0.15,
                a: 1.0,
            });
            pass.set_pipeline(pipeline);
            // Pass texture and sampler indices via push constants
            pass.set_push_constants_raw(&[tex_idx, samp_idx]);
            pass.set_vertex_buffer(0, vertex_buffer);
            pass.draw(0..6, 0..1);
        }

        frame.render(encoder)?;
        surface.present(frame)?;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &mut self.surface {
                let _ = surface.resize(new_size.width, new_size.height);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Textured Quad")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 800)),
                    )
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    eprintln!("Render error: {}", e);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(new_size) => {
                self.handle_resize(new_size);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    println!("Goldy Textured Quad Example");
    println!("Demonstrates texture sampling with a checkerboard pattern");
    println!("Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
