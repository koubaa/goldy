//! Multi-window example - three simultaneous effects in separate windows.
//!
//! Each window runs its own demo with an independent SwapchainPool + Scheme.
//!
//! Run with: cargo run --example multi_window

use goldy::{
    shaders, write_to_parcel, BufferFlags, BufferKind, Color, DeviceDescriptor, Grant, Instance, Lease,
    LeaseRenderTarget, NodeAccess, Parcel, PresentGrant, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions,
    RetainedPool, Scheme, ShaderModule, SwapchainPool, VertexAttribute, VertexBufferLayout, VertexFormat,
};
mod common;

const PLASMA_VERTEX_TIME: &str = r#"
struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
    float time : TEXCOORD1;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float time : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.time = input.time;
    return output;
}

float3 rainbow(float t) {
    float3 c = float3(
        sin(t * 6.28318 + 0.0) * 0.5 + 0.5,
        sin(t * 6.28318 + 2.094) * 0.5 + 0.5,
        sin(t * 6.28318 + 4.189) * 0.5 + 0.5
    );
    return c;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = input.uv * 4.0;
    float t = input.time;
    
    float v = sin(uv.x + t);
    v += sin(uv.y + t);
    v += sin(uv.x + uv.y + t);
    
    float cx = uv.x + 0.5 * sin(t / 3.0);
    float cy = uv.y + 0.5 * cos(t / 2.0);
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
    
    v = v / 2.0;
    
    return float4(rainbow(v), 1.0);
}
"#;

const TUNNEL_VERTEX_TIME: &str = r#"
struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
    float time : TEXCOORD1;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float time : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.time = input.time;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = (input.uv - 0.5) * 2.0;
    float t = input.time;
    
    float dist = length(uv);
    float angle = atan2(uv.y, uv.x);
    
    float tunnel_depth = 1.0 / (dist + 0.1);
    float tunnel_angle = angle / 3.14159 + t * 0.2;
    
    float tx = tunnel_angle * 4.0;
    float ty = tunnel_depth - t * 2.0;
    
    float checker = floor(tx) + floor(ty);
    bool is_white = fmod(checker, 2.0) == 0.0;
    
    float depth_color = 1.0 - dist * 0.5;
    float3 color;
    
    if (is_white) {
        color = float3(0.8, 0.2, 0.4) * depth_color;
    } else {
        color = float3(0.2, 0.4, 0.8) * depth_color;
    }
    
    color += float3(0.3, 0.5, 1.0) * (1.0 - dist) * (1.0 - dist);
    color *= 1.0 - dist * 0.3;
    
    return float4(color, 1.0);
}
"#;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes, WindowId},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    position: [f32; 2],
    uv: [f32; 2],
    time: f32,
}
impl goldy::StructuredBufferElement for QuadVertex {}

impl QuadVertex {
    fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: std::mem::size_of::<Self>() as u32,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    format: VertexFormat::Float32x2,
                    offset: 0,
                },
                VertexAttribute {
                    location: 1,
                    format: VertexFormat::Float32x2,
                    offset: 8,
                },
                VertexAttribute {
                    location: 2,
                    format: VertexFormat::Float32,
                    offset: 16,
                },
            ],
        }
    }
}

fn create_quad(time: f32) -> [QuadVertex; 6] {
    [
        QuadVertex {
            position: [-1.0, -1.0],
            uv: [0.0, 1.0],
            time,
        },
        QuadVertex {
            position: [1.0, -1.0],
            uv: [1.0, 1.0],
            time,
        },
        QuadVertex {
            position: [1.0, 1.0],
            uv: [1.0, 0.0],
            time,
        },
        QuadVertex {
            position: [-1.0, -1.0],
            uv: [0.0, 1.0],
            time,
        },
        QuadVertex {
            position: [1.0, 1.0],
            uv: [1.0, 0.0],
            time,
        },
        QuadVertex {
            position: [-1.0, 1.0],
            uv: [0.0, 0.0],
            time,
        },
    ]
}

#[derive(Clone, Copy, PartialEq)]
enum EffectType {
    Plasma,
    Tunnel,
    Starfield,
}

impl EffectType {
    fn title(&self) -> &'static str {
        match self {
            EffectType::Plasma => "Plasma [Space=pause, Click=reset]",
            EffectType::Tunnel => "Tunnel [Space=reverse, Click=reset]",
            EffectType::Starfield => "Starfield [Space=warp, Click=reset]",
        }
    }

    fn shader_source(&self) -> &'static str {
        match self {
            EffectType::Plasma => PLASMA_VERTEX_TIME,
            EffectType::Tunnel => TUNNEL_VERTEX_TIME,
            EffectType::Starfield => shaders::STARFIELD,
        }
    }
}

struct WindowState {
    window: Arc<Window>,
    swapchain: SwapchainPool,
    screen: goldy::PresentLease,
    present: PresentGrant,
    scheme: Scheme,
    scene_rt: Lease<LeaseRenderTarget>,
    pipeline: RenderPipeline,
    shader: ShaderModule,
    effect_type: EffectType,
    start_time: Instant,
    paused: bool,
    paused_at: f32,
    time_multiplier: f32,
    _retained_pool: RetainedPool,
    vertex_parcel: Parcel,
    has_focus: bool,
}

impl WindowState {
    fn create_pipeline(
        device: &goldy::Device,
        shader: &ShaderModule,
        swapchain: &SwapchainPool,
    ) -> anyhow::Result<RenderPipeline> {
        common::render_pipeline_for_swapchain(
            device,
            shader,
            swapchain,
            RenderPipelineDesc {
                vertex_layout: QuadVertex::layout(),
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        pipeline: &RenderPipeline,
        vertex_parcel: &Parcel,
        scene_rt: &Lease<LeaseRenderTarget>,
        screen: &goldy::PresentLease,
        label: &'static str,
    ) -> PresentGrant {
        let mut pass = scheme.render_pass(label, scene_rt);
        pass.with_parcel(vertex_parcel, NodeAccess::Read);
        pass.clear(Color::BLACK);
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_parcel);
        pass.draw(0..6, 0..1);
        pass.finish();
        scheme.copy_to_present(scene_rt, screen);
        scheme.grant_present(screen)
    }

    fn rerecord_scheme(&mut self) {
        self.scheme.begin_rerecord();
        let (width, height) = self.swapchain.size();
        if let Ok(rt) = self.scheme.lease_render_target(
            width.max(1),
            height.max(1),
            self.swapchain.format(),
            None,
        ) {
            self.scene_rt = rt;
            self.present = Self::record_scheme(
                &mut self.scheme,
                &self.pipeline,
                &self.vertex_parcel,
                &self.scene_rt,
                &self.screen,
                self.effect_type.title(),
            );
        }
    }

    fn new(
        window: Arc<Window>,
        ctx: &goldy::Context,
        device: &Arc<goldy::Device>,
        effect_type: EffectType,
    ) -> anyhow::Result<Self> {
        let swapchain = SwapchainPool::new(ctx, window.as_ref(), 3)?;
        let screen = swapchain.lease();
        let shader = ShaderModule::from_slang(device, effect_type.shader_source())?;
        let pipeline = Self::create_pipeline(device, &shader, &swapchain)?;

        let mut retained_pool = RetainedPool::new(device.clone());
        let vertex_parcel =
            retained_pool.acquire_buffer_sized::<QuadVertex>(6, BufferKind::Scattered, BufferFlags::empty())?;

        let mut scheme = Scheme::new(ctx);
        let (width, height) = swapchain.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)?;
        let present = Self::record_scheme(
            &mut scheme,
            &pipeline,
            &vertex_parcel,
            &scene_rt,
            &screen,
            effect_type.title(),
        );

        Ok(Self {
            window,
            swapchain,
            screen,
            present,
            scheme,
            scene_rt,
            pipeline,
            shader,
            effect_type,
            start_time: Instant::now(),
            paused: false,
            paused_at: 0.0,
            time_multiplier: 1.0,
            _retained_pool: retained_pool,
            vertex_parcel,
            has_focus: false,
        })
    }

    fn current_time(&self) -> f32 {
        if self.paused {
            self.paused_at
        } else {
            self.paused_at + self.start_time.elapsed().as_secs_f32() * self.time_multiplier
        }
    }

    fn toggle_pause(&mut self) {
        if self.paused {
            self.start_time = Instant::now();
            self.paused = false;
        } else {
            self.paused_at = self.current_time();
            self.paused = true;
        }
    }

    fn toggle_effect_modifier(&mut self) {
        match self.effect_type {
            EffectType::Plasma => self.toggle_pause(),
            EffectType::Tunnel => {
                self.paused_at = self.current_time();
                self.start_time = Instant::now();
                self.time_multiplier *= -1.0;
            }
            EffectType::Starfield => {
                self.paused_at = self.current_time();
                self.start_time = Instant::now();
                self.time_multiplier = if self.time_multiplier > 2.0 { 1.0 } else { 5.0 };
            }
        }
    }

    fn reset(&mut self) {
        self.start_time = Instant::now();
        self.paused = false;
        self.paused_at = 0.0;
        self.time_multiplier = 1.0;
    }

    fn render(&mut self, ctx: &goldy::Context) -> anyhow::Result<()> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let vertices = create_quad(self.current_time());
        write_to_parcel(ctx, &self.vertex_parcel, 0, bytemuck::cast_slice(&vertices))?;

        let submission = self.scheme.submit()?;
        self.present.consume(&submission)?;
        Ok(())
    }

    fn handle_resize(&mut self, device: &goldy::Device, width: u32, height: u32) {
        if width > 0 && height > 0 {
            let _ = self.swapchain.resize(width, height);
            self.scheme.begin_rerecord();
            let (width, height) = self.swapchain.size();
            match self
                .scheme
                .lease_render_target(width.max(1), height.max(1), self.swapchain.format(), None)
            {
                Ok(rt) => {
                    self.scene_rt = rt;
                    match Self::create_pipeline(device, &self.shader, &self.swapchain) {
                        Ok(pipeline) => {
                            self.pipeline = pipeline;
                            self.rerecord_scheme();
                        }
                        Err(e) => tracing::error!("[{}] Failed to recreate pipeline: {}", self.effect_type.title(), e),
                    }
                }
                Err(e) => tracing::error!("[{}] Failed to recreate scene RT: {}", self.effect_type.title(), e),
            }
        }
    }
}

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    windows: HashMap<WindowId, WindowState>,
    effects_to_create: Vec<EffectType>,
    frame_count: u32,
    start_time: std::time::Instant,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            ctx: None,
            device: None,
            windows: HashMap::new(),
            effects_to_create: vec![EffectType::Plasma, EffectType::Tunnel, EffectType::Starfield],
            frame_count: 0,
            start_time: std::time::Instant::now(),
        })
    }

    fn create_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        effect_type: EffectType,
        position: (i32, i32),
    ) -> anyhow::Result<()> {
        let device = self.device.as_ref().unwrap().clone();
        let ctx = self.ctx.as_ref().unwrap();

        let attrs = WindowAttributes::default()
            .with_title(format!("Goldy - {}", effect_type.title()))
            .with_inner_size(LogicalSize::new(500, 500))
            .with_position(winit::dpi::LogicalPosition::new(position.0, position.1));

        let window = Arc::new(event_loop.create_window(attrs)?);
        let window_id = window.id();

        let mut state = WindowState::new(window.clone(), ctx, &device, effect_type)?;
        state.render(ctx)?;
        window.request_redraw();

        self.windows.insert(window_id, state);
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.device.is_none() {
            match self
                .instance
                .request_adapter(&RequestAdapterOptions::default())
                .and_then(|a| a.request_device(&DeviceDescriptor::default()))
            {
                Ok(device) => {
                    let device = Arc::new(device);
                    self.ctx = Some(device.create_context().expect("create context"));
                    self.device = Some(device);
                }
                Err(e) => {
                    tracing::error!("Failed to create device: {}", e);
                    event_loop.exit();
                    return;
                }
            }
        }

        let effects: Vec<_> = self.effects_to_create.drain(..).collect();
        for (i, effect) in effects.into_iter().enumerate() {
            let x = 50 + (i as i32) * 520;
            let y = 100;

            if let Err(e) = self.create_window(event_loop, effect, (x, y)) {
                tracing::error!("Failed to create window for {:?}: {}", effect.title(), e);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let state = match self.windows.get_mut(&window_id) {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                self.windows.remove(&window_id);
                if self.windows.is_empty() {
                    event_loop.exit();
                }
            }
            WindowEvent::Focused(focused) => {
                state.has_focus = focused;
                if focused {
                    println!(
                        "Focus: {} ({})",
                        state.effect_type.title(),
                        if state.paused { "paused" } else { "running" }
                    );
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => {
                        self.windows.remove(&window_id);
                        if self.windows.is_empty() {
                            event_loop.exit();
                        }
                    }
                    Key::Named(NamedKey::Space) => {
                        if let Some(s) = self.windows.get_mut(&window_id) {
                            s.toggle_effect_modifier();
                            println!("[{}] Modifier toggled", s.effect_type.title());
                        }
                    }
                    Key::Character(ref c) if c == "r" || c == "R" => {
                        if let Some(s) = self.windows.get_mut(&window_id) {
                            s.reset();
                            println!("[{}] Reset", s.effect_type.title());
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if let Some(s) = self.windows.get_mut(&window_id) {
                    s.reset();
                    println!("[{}] Reset (click)", s.effect_type.title());
                }
            }
            WindowEvent::RedrawRequested => {}
            WindowEvent::Resized(new_size) => {
                if let Some(s) = self.windows.get_mut(&window_id) {
                    if let Some(device) = self.device.as_ref() {
                        s.handle_resize(device, new_size.width, new_size.height);
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        common::exit_if_timed_out(event_loop, self.start_time);

        let ctx = match &self.ctx {
            Some(c) => c,
            None => return,
        };

        self.frame_count += 1;

        for state in self.windows.values_mut() {
            if let Err(e) = state.render(ctx) {
                tracing::error!("[{}] Render error: {}", state.effect_type.title(), e);
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 {
            self.frame_count as f64 / elapsed
        } else {
            0.0
        };
        println!(
            "GOLDY_PERF: frames={} elapsed={elapsed:.2}s avg_fps={fps:.1}",
            self.frame_count
        );
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    println!("Goldy Multi-Window Example (Scheme + Present)");
    println!("Three windows, three effects, independent controls:");
    println!();
    println!("  Plasma:    Space=pause     Click/R=reset");
    println!("  Tunnel:    Space=reverse   Click/R=reset");
    println!("  Starfield: Space=warp      Click/R=reset");
    println!();
    println!("Escape closes the focused window. Close all to exit.");
    println!();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
