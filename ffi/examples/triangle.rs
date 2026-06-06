//! Triangle window example using the Goldy C ABI from Rust (in-tree FFI client).
//!
//! Run from `goldy/ffi`:
//! `cargo run --example triangle --features examples`

use goldy_ffi::winit_surface::goldy_surface_from_winit_window;
use goldy_ffi::{
    goldy_buffer_create_with_data, goldy_buffer_destroy, goldy_device_destroy, goldy_encoder_clear,
    goldy_encoder_create, goldy_encoder_draw, goldy_encoder_set_pipeline,
    goldy_encoder_set_vertex_buffer, goldy_get_last_error, goldy_instance_adapter_count,
    goldy_instance_create, goldy_instance_create_device_for_adapter, goldy_instance_destroy,
    goldy_instance_get_adapter, goldy_render_pipeline_create, goldy_render_pipeline_destroy,
    goldy_shader_builtin_vertex_color_2d, goldy_shader_create, goldy_shader_destroy,
    goldy_surface_acquire, goldy_surface_destroy, goldy_surface_format, goldy_surface_frame_render,
    goldy_surface_present, goldy_surface_resize, GoldyAdapterInfo, GoldyBuffer, GoldyBufferKind,
    GoldyColor, GoldyDevice, GoldyDeviceType, GoldyInstance, GoldyPrimitiveTopology,
    GoldyRenderPipeline, GoldyRenderPipelineDesc, GoldyResult, GoldyShaderModule, GoldySurface,
    GoldyVertexAttribute, GoldyVertexFormat,
};
use std::ffi::CStr;
use std::mem::size_of;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

fn last_ffi_message() -> String {
    unsafe {
        let p = goldy_get_last_error();
        if p.is_null() {
            return "(no message)".into();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

unsafe fn request_device_for_discrete_gpu(instance: *const GoldyInstance) -> *mut GoldyDevice {
    let count = goldy_instance_adapter_count(instance);
    let mut best_id: u32 = 0;
    for i in 0..count {
        let mut info = GoldyAdapterInfo {
            id: 0,
            device_type: GoldyDeviceType::Other,
            name: [0; 256],
            vendor: [0; 64],
        };
        if goldy_instance_get_adapter(instance, i, &mut info) != GoldyResult::Ok {
            continue;
        }
        if i == 0 {
            best_id = info.id;
        }
        if info.device_type == GoldyDeviceType::DiscreteGpu {
            best_id = info.id;
            break;
        }
    }
    goldy_instance_create_device_for_adapter(instance, best_id)
}

struct App {
    instance: *mut GoldyInstance,
    device: *mut GoldyDevice,
    surface: *mut GoldySurface,
    vertex_buffer: *mut GoldyBuffer,
    shader: *mut GoldyShaderModule,
    pipeline: *mut GoldyRenderPipeline,
    window: Option<Arc<Window>>,
    frame_count: u64,
}

impl App {
    fn new() -> Self {
        Self {
            instance: std::ptr::null_mut(),
            device: std::ptr::null_mut(),
            surface: std::ptr::null_mut(),
            vertex_buffer: std::ptr::null_mut(),
            shader: std::ptr::null_mut(),
            pipeline: std::ptr::null_mut(),
            window: None,
            frame_count: 0,
        }
    }

    unsafe fn init_gpu(&mut self, window: &Arc<Window>) -> Result<(), String> {
        self.instance = goldy_instance_create();
        if self.instance.is_null() {
            return Err(format!("goldy_instance_create: {}", last_ffi_message()));
        }

        self.device = request_device_for_discrete_gpu(self.instance);
        if self.device.is_null() {
            return Err(format!(
                "goldy_instance_create_device_for_adapter: {}",
                last_ffi_message()
            ));
        }

        self.surface = goldy_surface_from_winit_window(self.device, window.as_ref());
        if self.surface.is_null() {
            return Err(format!(
                "goldy_surface_from_winit_window: {}",
                last_ffi_message()
            ));
        }

        let vertices = [
            Vertex {
                position: [0.0, -0.5],
                color: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-0.5, 0.5],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                position: [0.5, 0.5],
                color: [0.0, 0.0, 1.0, 1.0],
            },
        ];
        let vb_bytes = bytemuck::cast_slice(&vertices);
        self.vertex_buffer = goldy_buffer_create_with_data(
            self.device,
            vb_bytes.as_ptr(),
            vb_bytes.len(),
            GoldyBufferKind::Scattered,
        );
        if self.vertex_buffer.is_null() {
            return Err(format!(
                "goldy_buffer_create_with_data: {}",
                last_ffi_message()
            ));
        }

        let src = goldy_shader_builtin_vertex_color_2d();
        self.shader = goldy_shader_create(self.device, src);
        if self.shader.is_null() {
            return Err(format!("goldy_shader_create: {}", last_ffi_message()));
        }

        let attrs = [
            GoldyVertexAttribute {
                location: 0,
                format: GoldyVertexFormat::Float32x2,
                offset: 0,
            },
            GoldyVertexAttribute {
                location: 1,
                format: GoldyVertexFormat::Float32x4,
                offset: 8,
            },
        ];
        let pipeline_desc = GoldyRenderPipelineDesc {
            vertex_attributes: attrs.as_ptr(),
            vertex_attribute_count: attrs.len() as u32,
            vertex_stride: size_of::<Vertex>() as u32,
            topology: GoldyPrimitiveTopology::TriangleList,
            target_format: goldy_surface_format(self.surface),
            depth_enabled: false,
            ..Default::default()
        };

        self.pipeline =
            goldy_render_pipeline_create(self.device, self.shader, self.shader, &pipeline_desc);
        if self.pipeline.is_null() {
            return Err(format!(
                "goldy_render_pipeline_create: {}",
                last_ffi_message()
            ));
        }

        Ok(())
    }

    unsafe fn render_frame(&mut self) -> Result<(), String> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let t = (self.frame_count as f32 * 0.02).sin() * 0.5 + 0.5;
        let bg = GoldyColor {
            r: 0.1 + t * 0.1,
            g: 0.1 + t * 0.05,
            b: 0.2 + t * 0.1,
            a: 1.0,
        };

        let frame = goldy_surface_acquire(self.surface);
        if frame.is_null() {
            return Err(format!("goldy_surface_acquire: {}", last_ffi_message()));
        }

        let encoder = goldy_encoder_create();
        if encoder.is_null() {
            return Err("goldy_encoder_create returned null".into());
        }

        goldy_encoder_clear(encoder, bg);
        goldy_encoder_set_pipeline(encoder, self.pipeline);
        goldy_encoder_set_vertex_buffer(encoder, 0, self.vertex_buffer);
        goldy_encoder_draw(encoder, 0, 3, 0, 1);

        if goldy_surface_frame_render(frame, encoder) != GoldyResult::Ok {
            return Err(format!(
                "goldy_surface_frame_render: {}",
                last_ffi_message()
            ));
        }

        if goldy_surface_present(self.surface, frame) != GoldyResult::Ok {
            return Err(format!("goldy_surface_present: {}", last_ffi_message()));
        }

        self.frame_count += 1;
        Ok(())
    }

    unsafe fn cleanup(&mut self) {
        if !self.pipeline.is_null() {
            goldy_render_pipeline_destroy(self.pipeline);
            self.pipeline = std::ptr::null_mut();
        }
        if !self.shader.is_null() {
            goldy_shader_destroy(self.shader);
            self.shader = std::ptr::null_mut();
        }
        if !self.vertex_buffer.is_null() {
            goldy_buffer_destroy(self.vertex_buffer);
            self.vertex_buffer = std::ptr::null_mut();
        }
        if !self.surface.is_null() {
            goldy_surface_destroy(self.surface);
            self.surface = std::ptr::null_mut();
        }
        if !self.device.is_null() {
            goldy_device_destroy(self.device);
            self.device = std::ptr::null_mut();
        }
        if !self.instance.is_null() {
            goldy_instance_destroy(self.instance);
            self.instance = std::ptr::null_mut();
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.cleanup();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Goldy FFI — Triangle (C ABI)")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        unsafe {
            if let Err(e) = self.init_gpu(&window) {
                eprintln!("Init failed: {e}");
                event_loop.exit();
                return;
            }
        }
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => unsafe {
                if let Err(e) = self.render_frame() {
                    eprintln!("Render error: {e}");
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            },
            WindowEvent::Resized(new_size) if new_size.width > 0 && new_size.height > 0 => unsafe {
                goldy_surface_resize(self.surface, new_size.width, new_size.height);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            },
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Goldy FFI triangle example (Rust client of the C ABI)");
    println!("Run from goldy/ffi: cargo run --example triangle --features examples\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
