//! Retro Frogger — CPU game logic with instanced colored quads.
//!
//! Demonstrates: CPU-owned state → `MemoryExchange` deposit → retained
//! instanced draw → present (the interactive counterpart to GPU-driven
//! `instancing`).
//!
//! Controls: Arrow keys hop, Escape exits.
//!
//! Run with: cargo run --features examples --example frogger

use anyhow::Result;
use goldy::{
    Buffer, BufferFlags, BufferKind, Color, DepositTransaction, DeviceDescriptor, Instance, Lease, LeaseRenderTarget,
    MemoryExchange, NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions,
    RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Transaction, VertexBufferLayout,
};

mod instance2d;
use instance2d::Instance2D;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};
mod common;

// --- Playfield ----------------------------------------------------------------

const COLS: i32 = 13;
const ROWS: i32 = 13;
const ROW_START: i32 = 0;
const ROW_ROAD_LO: i32 = 1;
const ROW_ROAD_HI: i32 = 5;
const ROW_MEDIAN: i32 = 6;
const ROW_RIVER_LO: i32 = 7;
const ROW_RIVER_HI: i32 = 11;
const ROW_HOMES: i32 = 12;

const HOME_COLS: [i32; 5] = [1, 3, 5, 7, 9];
const MAX_INSTANCES: usize = 512;
const PLAYFIELD: f32 = 0.92;
const TIMER_MAX: f32 = 30.0;
const HOP_COOLDOWN: f32 = 0.12;
const DEATH_FLASH: f32 = 0.45;
const STARTING_LIVES: u32 = 3;

// --- Colors (retro palette) ---------------------------------------------------

const C_BG: [f32; 4] = [0.05, 0.05, 0.08, 1.0];
const C_SIDEWALK: [f32; 4] = [0.22, 0.55, 0.28, 1.0];
const C_ROAD: [f32; 4] = [0.14, 0.14, 0.16, 1.0];
const C_WATER: [f32; 4] = [0.08, 0.22, 0.55, 1.0];
const C_HOME_EMPTY: [f32; 4] = [0.12, 0.35, 0.18, 1.0];
const C_HOME_SLOT: [f32; 4] = [0.55, 0.15, 0.45, 1.0];
const C_HOME_FULL: [f32; 4] = [0.25, 0.85, 0.35, 1.0];
const C_FROG: [f32; 4] = [0.35, 0.95, 0.30, 1.0];
const C_FROG_FLASH: [f32; 4] = [1.0, 0.35, 0.25, 1.0];
const C_LOG: [f32; 4] = [0.55, 0.32, 0.12, 1.0];
const C_TIMER: [f32; 4] = [0.95, 0.85, 0.20, 1.0];
const C_TIMER_LOW: [f32; 4] = [0.95, 0.25, 0.15, 1.0];
const C_LIFE: [f32; 4] = [0.35, 0.95, 0.30, 1.0];

const CAR_COLORS: [[f32; 4]; 5] = [
    [0.90, 0.25, 0.20, 1.0],
    [0.95, 0.75, 0.15, 1.0],
    [0.25, 0.55, 0.95, 1.0],
    [0.95, 0.45, 0.10, 1.0],
    [0.85, 0.20, 0.75, 1.0],
];

// --- Game ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum MoverKind {
    Car,
    Log,
}

#[derive(Clone, Copy)]
struct Mover {
    /// Left edge in cell coordinates (may be off-screen).
    x: f32,
    width: f32,
    row: i32,
    /// Cells per second (sign = direction).
    speed: f32,
    kind: MoverKind,
    color: [f32; 4],
}

#[derive(Clone, Copy)]
enum Hop {
    Up,
    Down,
    Left,
    Right,
}

struct GameState {
    frog_x: f32,
    frog_row: i32,
    lives: u32,
    score: u32,
    level: u32,
    homes: [bool; 5],
    movers: Vec<Mover>,
    timer: f32,
    hop_cooldown: f32,
    death_flash: f32,
    max_row: i32,
    game_over: bool,
}

impl GameState {
    fn new() -> Self {
        let mut g = Self {
            frog_x: (COLS / 2) as f32,
            frog_row: ROW_START,
            lives: STARTING_LIVES,
            score: 0,
            level: 1,
            homes: [false; 5],
            movers: Vec::new(),
            timer: TIMER_MAX,
            hop_cooldown: 0.0,
            death_flash: 0.0,
            max_row: ROW_START,
            game_over: false,
        };
        g.spawn_movers();
        g.print_status("Ready");
        g
    }

    fn speed_scale(&self) -> f32 {
        1.0 + 0.18 * (self.level.saturating_sub(1) as f32)
    }

    fn spawn_movers(&mut self) {
        self.movers.clear();
        let s = self.speed_scale();

        // Road: rows 1..=5, alternating directions
        let road: &[(i32, f32, f32, usize)] = &[
            (1, 2.2, 2.0, 0),
            (2, -2.8, 2.0, 1),
            (3, 3.4, 3.0, 2),
            (4, -2.0, 2.0, 3),
            (5, 4.0, 3.0, 4),
        ];
        for &(row, speed, width, color_i) in road {
            let dir = speed.signum();
            let spacing = width + 3.5;
            let count = ((COLS as f32 + 6.0) / spacing).ceil() as i32 + 1;
            for i in 0..count {
                let x = if dir > 0.0 {
                    -4.0 + i as f32 * spacing
                } else {
                    (COLS as f32 + 1.0) - i as f32 * spacing
                };
                self.movers.push(Mover {
                    x,
                    width,
                    row,
                    speed: speed * s,
                    kind: MoverKind::Car,
                    color: CAR_COLORS[color_i % CAR_COLORS.len()],
                });
            }
        }

        // River logs: rows 7..=11
        let river: &[(i32, f32, f32)] = &[
            (7, 1.6, 3.0),
            (8, -2.0, 4.0),
            (9, 2.4, 3.0),
            (10, -1.5, 5.0),
            (11, 2.8, 4.0),
        ];
        for &(row, speed, width) in river {
            let dir = speed.signum();
            let spacing = width + 2.5;
            let count = ((COLS as f32 + 8.0) / spacing).ceil() as i32 + 1;
            for i in 0..count {
                let x = if dir > 0.0 {
                    -5.0 + i as f32 * spacing
                } else {
                    (COLS as f32 + 2.0) - i as f32 * spacing
                };
                self.movers.push(Mover {
                    x,
                    width,
                    row,
                    speed: speed * s,
                    kind: MoverKind::Log,
                    color: C_LOG,
                });
            }
        }
    }

    fn print_status(&self, note: &str) {
        println!(
            "Frogger | {note} | score={} level={} lives={} homes={}/5",
            self.score,
            self.level,
            self.lives,
            self.homes.iter().filter(|&&h| h).count()
        );
    }

    fn respawn_frog(&mut self) {
        self.frog_x = (COLS / 2) as f32;
        self.frog_row = ROW_START;
        self.timer = TIMER_MAX;
        self.max_row = ROW_START;
        self.hop_cooldown = HOP_COOLDOWN;
        self.death_flash = DEATH_FLASH;
    }

    fn kill(&mut self, reason: &str) {
        if self.death_flash > 0.0 || self.game_over {
            return;
        }
        if self.lives > 0 {
            self.lives -= 1;
        }
        if self.lives == 0 {
            self.game_over = true;
            self.death_flash = DEATH_FLASH * 2.0;
            self.print_status(&format!("GAME OVER ({reason}) — press any arrow to restart"));
        } else {
            self.print_status(&format!("Died: {reason}"));
            self.respawn_frog();
        }
    }

    fn restart(&mut self) {
        *self = Self::new();
    }

    fn try_hop(&mut self, hop: Hop) {
        if self.game_over {
            self.restart();
            return;
        }
        if self.hop_cooldown > 0.0 || self.death_flash > 0.0 {
            return;
        }

        let (dx, dy) = match hop {
            Hop::Up => (0.0, 1),
            Hop::Down => (0.0, -1),
            Hop::Left => (-1.0, 0),
            Hop::Right => (1.0, 0),
        };

        let new_x = self.frog_x + dx;
        let new_row = self.frog_row + dy;
        if new_x < 0.0 || new_x > (COLS - 1) as f32 || !(0..ROWS).contains(&new_row) {
            return;
        }

        self.frog_x = new_x.round();
        self.frog_row = new_row;
        self.hop_cooldown = HOP_COOLDOWN;

        if new_row > self.max_row {
            self.max_row = new_row;
            self.score += 10;
        }

        if new_row == ROW_HOMES {
            self.try_enter_home();
        }
    }

    fn try_enter_home(&mut self) {
        let fx = self.frog_x;
        let mut matched = None;
        for (i, &hc) in HOME_COLS.iter().enumerate() {
            if (fx - hc as f32).abs() < 0.6 {
                matched = Some(i);
                break;
            }
        }
        match matched {
            Some(i) if !self.homes[i] => {
                self.homes[i] = true;
                let bonus = 50 + (self.timer * 2.0) as u32;
                self.score += bonus;
                self.print_status(&format!("Home! +{bonus}"));
                if self.homes.iter().all(|&h| h) {
                    self.score += 1000;
                    self.level += 1;
                    self.homes = [false; 5];
                    self.spawn_movers();
                    self.print_status("Level clear! +1000");
                }
                self.respawn_frog();
                self.death_flash = 0.0;
            }
            _ => self.kill("missed home"),
        }
    }

    fn frog_on_log(&self) -> Option<f32> {
        let fx = self.frog_x + 0.5;
        for m in &self.movers {
            if m.kind == MoverKind::Log && m.row == self.frog_row && fx >= m.x && fx <= m.x + m.width {
                return Some(m.speed);
            }
        }
        None
    }

    fn hit_by_car(&self) -> bool {
        let fx0 = self.frog_x + 0.15;
        let fx1 = self.frog_x + 0.85;
        for m in &self.movers {
            if m.kind == MoverKind::Car && m.row == self.frog_row {
                let mx0 = m.x;
                let mx1 = m.x + m.width;
                if fx0 < mx1 && fx1 > mx0 {
                    return true;
                }
            }
        }
        false
    }

    fn update(&mut self, dt: f32) {
        if self.game_over {
            self.death_flash = (self.death_flash - dt).max(0.0);
            // Still animate movers for ambience
            self.advance_movers(dt);
            return;
        }

        self.hop_cooldown = (self.hop_cooldown - dt).max(0.0);
        self.death_flash = (self.death_flash - dt).max(0.0);

        if self.death_flash > 0.0 {
            self.advance_movers(dt);
            return;
        }

        self.timer -= dt;
        if self.timer <= 0.0 {
            self.timer = 0.0;
            self.kill("time up");
            return;
        }

        self.advance_movers(dt);

        // Ride logs
        if (ROW_RIVER_LO..=ROW_RIVER_HI).contains(&self.frog_row) {
            match self.frog_on_log() {
                Some(speed) => {
                    self.frog_x += speed * dt;
                    if self.frog_x < -0.2 || self.frog_x > (COLS - 1) as f32 + 0.2 {
                        self.kill("swept away");
                        return;
                    }
                }
                None => {
                    self.kill("drowned");
                    return;
                }
            }
        }

        if (ROW_ROAD_LO..=ROW_ROAD_HI).contains(&self.frog_row) && self.hit_by_car() {
            self.kill("hit by car");
        }
    }

    fn advance_movers(&mut self, dt: f32) {
        for m in &mut self.movers {
            m.x += m.speed * dt;
            let margin = m.width + 2.0;
            if m.speed > 0.0 && m.x > COLS as f32 + margin {
                m.x = -margin;
            } else if m.speed < 0.0 && m.x + m.width < -margin {
                m.x = COLS as f32 + margin;
            }
        }
    }
}

// --- Coordinate mapping -------------------------------------------------------

fn cell_size() -> f32 {
    (2.0 * PLAYFIELD) / COLS as f32
}

fn cell_to_ndc(col: f32, row: f32) -> [f32; 2] {
    let cs = cell_size();
    [-PLAYFIELD + (col + 0.5) * cs, -PLAYFIELD + (row + 0.5) * cs]
}

fn push_quad(out: &mut Vec<Instance2D>, col: f32, row: f32, half: f32, color: [f32; 4]) {
    let pos = cell_to_ndc(col, row);
    out.push(Instance2D::new(pos[0], pos[1], 0.0, half, color));
}

fn push_span(out: &mut Vec<Instance2D>, x: f32, row: i32, width: f32, half: f32, color: [f32; 4]) {
    // Cover [x, x+width) with overlapping unit-ish quads for a solid bar
    let steps = width.ceil().max(1.0) as i32;
    for i in 0..steps {
        let c = x + i as f32 + 0.5;
        if c < -1.5 || c > COLS as f32 + 1.5 {
            continue;
        }
        push_quad(out, c, row as f32, half, color);
    }
}

fn build_instances(game: &GameState) -> Vec<Instance2D> {
    let mut out = Vec::with_capacity(MAX_INSTANCES);
    let half = cell_size() * 0.48;
    let tile = cell_size() * 0.50;

    // Background tiles
    for row in 0..ROWS {
        let color = if row == ROW_START || row == ROW_MEDIAN {
            C_SIDEWALK
        } else if (ROW_ROAD_LO..=ROW_ROAD_HI).contains(&row) {
            C_ROAD
        } else if (ROW_RIVER_LO..=ROW_RIVER_HI).contains(&row) {
            C_WATER
        } else if row == ROW_HOMES {
            C_HOME_EMPTY
        } else {
            C_BG
        };
        for col in 0..COLS {
            // Home row: gaps between pads stay empty-green; pads painted below
            if row == ROW_HOMES && HOME_COLS.contains(&col) {
                continue;
            }
            push_quad(&mut out, col as f32, row as f32, tile, color);
        }
    }

    // Home pads
    for (i, &hc) in HOME_COLS.iter().enumerate() {
        let color = if game.homes[i] { C_HOME_FULL } else { C_HOME_SLOT };
        push_quad(&mut out, hc as f32, ROW_HOMES as f32, half * 0.95, color);
    }

    // Movers
    for m in &game.movers {
        push_span(&mut out, m.x, m.row, m.width, half * 0.85, m.color);
    }

    // Frog
    if !game.game_over || game.death_flash > 0.0 {
        let flash = game.death_flash > 0.0 && ((game.death_flash * 12.0) as i32 % 2 == 0);
        let color = if flash { C_FROG_FLASH } else { C_FROG };
        if !game.game_over || flash {
            push_quad(&mut out, game.frog_x, game.frog_row as f32, half * 0.75, color);
        }
    }

    // Timer bar (top edge, above homes)
    let t = (game.timer / TIMER_MAX).clamp(0.0, 1.0);
    let timer_color = if t < 0.25 { C_TIMER_LOW } else { C_TIMER };
    let bar_half = cell_size() * 0.12;
    let bar_row = ROW_HOMES as f32 + 0.85;
    let filled = (t * COLS as f32).max(0.05);
    let steps = filled.ceil().max(1.0) as i32;
    for i in 0..steps {
        let c = i as f32 + 0.5;
        if c > filled {
            break;
        }
        let pos = cell_to_ndc(c, bar_row);
        out.push(Instance2D::new(pos[0], pos[1], 0.0, bar_half, timer_color));
    }

    // Lives (bottom-left, below start row)
    let life_half = cell_size() * 0.22;
    for i in 0..game.lives {
        let pos = cell_to_ndc(0.35 + i as f32 * 0.7, -0.65);
        out.push(Instance2D::new(pos[0], pos[1], 0.0, life_half, C_LIFE));
    }

    // Pad to fixed capacity with invisible quads
    while out.len() < MAX_INSTANCES {
        out.push(Instance2D::new(0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]));
    }
    if out.len() > MAX_INSTANCES {
        out.truncate(MAX_INSTANCES);
    }
    out
}

// --- Render shell -------------------------------------------------------------

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    println!("Goldy Frogger — arrow keys hop, Escape exits");
    println!("Reach the five home pads at the top. Avoid cars; ride logs.");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Default)]
struct App {
    state: Option<RenderState>,
}

struct RenderState {
    window: Arc<Window>,
    device: Arc<goldy::Device>,
    ctx: goldy::Context,
    surface: SurfaceExchange,
    present: Transaction,
    scheme: Scheme,
    scene_rt: Lease<LeaseRenderTarget>,
    render_shader: ShaderModule,
    render_pipeline: RenderPipeline,
    _retained_pool: RetainedPool,
    instance_buffer: Buffer,
    upload_scheme: Scheme,
    instance_deposit: DepositTransaction,
    game: GameState,
    start_time: Instant,
    last_time: f32,
    frame_count: u32,
}

impl RenderState {
    fn create_render_pipeline(
        device: &goldy::Device,
        render_shader: &ShaderModule,
        surface: &SurfaceExchange,
    ) -> Result<RenderPipeline> {
        common::render_pipeline_for_surface(
            device,
            render_shader,
            surface,
            RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        surface: &SurfaceExchange,
        render_pipeline: &RenderPipeline,
        instance_buffer: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
    ) -> anyhow::Result<Transaction> {
        let bg = Color {
            r: C_BG[0],
            g: C_BG[1],
            b: C_BG[2],
            a: 1.0,
        };
        let mut pass = scheme.render_pass("frogger", scene_rt, TargetLoad::Clear(bg));
        pass.with_parcel(instance_buffer, NodeAccess::Read);
        pass.set_pipeline(render_pipeline);
        pass.draw(0..6, 0..MAX_INSTANCES as u32);
        pass.finish();
        surface.bind_render_target(scheme, scene_rt).map_err(Into::into)
    }

    fn rerecord_scheme(&mut self) {
        let mut scheme = Scheme::new(&self.ctx);
        let (width, height) = self.surface.size();
        if let Ok(rt) = scheme.lease_render_target(width.max(1), height.max(1), self.surface.format(), None) {
            self.scene_rt = rt;
            if let Ok(present) = Self::record_scheme(
                &mut scheme,
                &self.surface,
                &self.render_pipeline,
                &self.instance_buffer,
                &self.scene_rt,
            ) {
                self.present = present;
                self.scheme = scheme;
            }
        }
    }

    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let surface = SurfaceExchange::new(&ctx, window.as_ref(), SurfaceConfig::default())?;

        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/instancing_render.slang"))?;

        let game = GameState::new();
        let instances = build_instances(&game);

        let mut retained_pool = RetainedPool::new(device.clone());
        let instance_buffer = retained_pool.acquire_buffer_sized::<Instance2D>(
            MAX_INSTANCES as u64,
            BufferKind::Scattered,
            BufferFlags::empty(),
        )?;

        let render_pipeline = Self::create_render_pipeline(&device, &render_shader, &surface)?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = surface.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), surface.format(), None)?;
        let present = Self::record_scheme(&mut scheme, &surface, &render_pipeline, &instance_buffer, &scene_rt)?;

        let mut upload_scheme = Scheme::new(&ctx);
        let instance_deposit = MemoryExchange::new(&ctx).bind_deposit_buffer(
            &mut upload_scheme,
            &instance_buffer,
            (MAX_INSTANCES * std::mem::size_of::<Instance2D>()) as u64,
        )?;

        // Seed buffer
        instance_deposit.write(&mut upload_scheme, 0, bytemuck::cast_slice(&instances))?;
        upload_scheme.submit()?;

        Ok(Self {
            window,
            device,
            ctx,
            surface,
            present,
            scheme,
            scene_rt,
            render_shader,
            render_pipeline,
            _retained_pool: retained_pool,
            instance_buffer,
            upload_scheme,
            instance_deposit,
            game,
            start_time: Instant::now(),
            last_time: 0.0,
            frame_count: 0,
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;
        let time = self.start_time.elapsed().as_secs_f32();
        let mut dt = time - self.last_time;
        self.last_time = time;
        // Clamp large hitch so the frog doesn't teleport through cars
        if self.frame_count == 1 {
            dt = 0.0;
        }
        dt = dt.clamp(0.0, 0.05);

        self.game.update(dt);
        let instances = build_instances(&self.game);

        self.instance_deposit
            .write(&mut self.upload_scheme, 0, bytemuck::cast_slice(&instances))?;
        self.upload_scheme.submit()?;

        let mut submission = self.scheme.submit()?;
        self.present.claim(&mut submission)?.consume()?;
        self.window.request_redraw();
        Ok(())
    }
}

impl Drop for RenderState {
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Frogger")
                            .with_inner_size(winit::dpi::LogicalSize::new(780, 780)),
                    )
                    .expect("Failed to create window"),
            );
            match RenderState::new(window.clone()) {
                Ok(state) => {
                    self.state = Some(state);
                    window.request_redraw();
                }
                Err(e) => {
                    tracing::error!("Failed to create render state: {e}");
                    event_loop.exit();
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            common::exit_if_timed_out(event_loop, state.start_time);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() && !event.repeat => {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::ArrowUp) => {
                        if let Some(s) = &mut self.state {
                            s.game.try_hop(Hop::Up);
                        }
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        if let Some(s) = &mut self.state {
                            s.game.try_hop(Hop::Down);
                        }
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        if let Some(s) = &mut self.state {
                            s.game.try_hop(Hop::Left);
                        }
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        if let Some(s) = &mut self.state {
                            s.game.try_hop(Hop::Right);
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    if size.width > 0 && size.height > 0 {
                        state.surface.resize(size.width, size.height).ok();
                        if let Ok(pipeline) =
                            RenderState::create_render_pipeline(&state.device, &state.render_shader, &state.surface)
                        {
                            state.render_pipeline = pipeline;
                        }
                        state.rerecord_scheme();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    if let Err(e) = state.render() {
                        tracing::error!("Render error: {e}");
                    }
                }
            }
            _ => {}
        }
    }
}
