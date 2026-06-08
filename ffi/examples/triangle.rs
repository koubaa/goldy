//! Triangle window example using the Goldy C ABI + TaskGraph (Phase 2).
//!
//! Run from `goldy/ffi`:
//! `cargo run --example triangle --features examples`

use goldy_ffi::winit_surface::goldy_surface_from_winit_window;
use goldy_ffi::{
    goldy_buffer_create_with_data, goldy_buffer_destroy, goldy_device_destroy, goldy_get_last_error,
    goldy_instance_adapter_count, goldy_instance_create, goldy_instance_create_device_for_adapter,
    goldy_instance_destroy, goldy_instance_get_adapter, goldy_render_pipeline_create,
    goldy_render_pipeline_destroy, goldy_render_target_create, goldy_render_target_destroy,
    goldy_shader_builtin_vertex_color_2d, goldy_shader_create, goldy_shader_destroy,
    goldy_surface_acquire, goldy_surface_destroy, goldy_surface_format, goldy_surface_height,
    goldy_surface_present, goldy_surface_resize, goldy_surface_submit_graph_to_frame,
    goldy_surface_width, goldy_task_graph_clear, goldy_task_graph_copy_render_target_to_swapchain,
    goldy_task_graph_create, goldy_task_graph_declare_swapchain_output, goldy_task_graph_destroy,
    goldy_task_graph_render_pass_begin, goldy_task_graph_render_pass_bind_buffer,
    goldy_task_graph_render_pass_clear, goldy_task_graph_render_pass_draw,
    goldy_task_graph_render_pass_finish, goldy_task_graph_render_pass_set_pipeline,
    goldy_task_graph_render_pass_set_vertex_buffer, GoldyAdapterInfo, GoldyBuffer, GoldyBufferKind,
    GoldyColor, GoldyDevice, GoldyDeviceType, GoldyInstance, GoldyNodeAccess, GoldyPrimitiveTopology,
    GoldyRenderPipeline, GoldyRenderPipelineDesc, GoldyRenderTarget, GoldyResult, GoldyShaderModule,
    GoldySurface, GoldyTaskGraph, GoldyVertexAttribute, GoldyVertexFormat,
};
use std::ffi::{CStr, CString};
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
    window: Option<Arc<Window>>,
    surface: Option<*mut GoldySurface>,
    scene_rt: Option<*mut GoldyRenderTarget>,
    graph: Option<*mut GoldyTaskGraph>,
    vertex_buffer: Option<*mut GoldyBuffer>,
    pipeline: Option<*mut GoldyRenderPipeline>,
    shader: Option<*mut GoldyShaderModule>,
}

impl App {
    fn new() -> Self {
        Self {
            instance: std::ptr::null_mut(),
            device: std::ptr::null_mut(),
            window: None,
            surface: None,
            scene_rt: None,
            graph: None,
            vertex_buffer: None,
            pipeline: None,
            shader: None,
        }
    }

    unsafe fn init_gpu(&mut self, window: &Arc<Window>) -> Result<(), String> {
        self.instance = goldy_instance_create();
        if self.instance.is_null() {
            return Err(last_ffi_message());
        }

        self.device = request_device_for_discrete_gpu(self.instance);
        if self.device.is_null() {
            return Err(last_ffi_message());
        }

        let surface = goldy_surface_from_winit_window(self.device, window.as_ref());
        if surface.is_null() {
            return Err(last_ffi_message());
        }
        self.surface = Some(surface);

        let width = goldy_surface_width(surface).max(1);
        let height = goldy_surface_height(surface).max(1);
        let format = goldy_surface_format(surface);

        let scene_rt = goldy_render_target_create(self.device, width, height, format);
        if scene_rt.is_null() {
            return Err(last_ffi_message());
        }
        self.scene_rt = Some(scene_rt);

        let graph = goldy_task_graph_create();
        if graph.is_null() {
            return Err("goldy_task_graph_create returned null".into());
        }
        self.graph = Some(graph);

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
        let vertex_buffer = goldy_buffer_create_with_data(
            self.device,
            vertices.as_ptr() as *const u8,
            size_of_val(&vertices),
            GoldyBufferKind::Scattered,
        );
        if vertex_buffer.is_null() {
            return Err(last_ffi_message());
        }
        self.vertex_buffer = Some(vertex_buffer);

        let builtin = CStr::from_ptr(goldy_shader_builtin_vertex_color_2d());
        let shader_src = CString::new(builtin.to_bytes()).map_err(|e| e.to_string())?;
        let shader = goldy_shader_create(self.device, shader_src.as_ptr());
        if shader.is_null() {
            return Err(last_ffi_message());
        }
        self.shader = Some(shader);

        let attributes = [
            GoldyVertexAttribute {
                location: 0,
                format: GoldyVertexFormat::Float32x2,
                offset: 0,
            },
            GoldyVertexAttribute {
                location: 1,
                format: GoldyVertexFormat::Float32x4,
                offset: size_of::<[f32; 2]>() as u32,
            },
        ];
        let pipeline_desc = GoldyRenderPipelineDesc {
            vertex_attributes: attributes.as_ptr(),
            vertex_attribute_count: attributes.len() as u32,
            vertex_stride: size_of::<Vertex>() as u32,
            topology: GoldyPrimitiveTopology::TriangleList,
            target_format: format,
            depth_enabled: false,
            ..Default::default()
        };
        let pipeline =
            goldy_render_pipeline_create(self.device, shader, shader, &pipeline_desc as *const _);
        if pipeline.is_null() {
            return Err(last_ffi_message());
        }
        self.pipeline = Some(pipeline);

        Ok(())
    }

    unsafe fn recreate_scene_rt(&mut self) -> Result<(), String> {
        let surface = self.surface.ok_or("no surface")?;
        let device = self.device;
        if let Some(old) = self.scene_rt.take() {
            goldy_render_target_destroy(old);
        }
        let width = goldy_surface_width(surface).max(1);
        let height = goldy_surface_height(surface).max(1);
        let format = goldy_surface_format(surface);
        let scene_rt = goldy_render_target_create(device, width, height, format);
        if scene_rt.is_null() {
            return Err(last_ffi_message());
        }
        self.scene_rt = Some(scene_rt);
        Ok(())
    }

    unsafe fn render_frame(&mut self) -> Result<(), String> {
        let window = self.window.as_ref().ok_or("no window")?;
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let surface = self.surface.ok_or("no surface")?;
        let scene_rt = self.scene_rt.ok_or("no scene rt")?;
        let graph = self.graph.ok_or("no graph")?;
        let pipeline = self.pipeline.ok_or("no pipeline")?;
        let vertex_buffer = self.vertex_buffer.ok_or("no vertex buffer")?;

        goldy_task_graph_clear(graph);

        let label = CString::new("triangle").unwrap();
        if goldy_task_graph_render_pass_begin(graph, label.as_ptr(), scene_rt) != GoldyResult::Ok {
            return Err(last_ffi_message());
        }
        if goldy_task_graph_render_pass_bind_buffer(graph, vertex_buffer, GoldyNodeAccess::Read)
            != GoldyResult::Ok
        {
            return Err(last_ffi_message());
        }
        let bg = GoldyColor {
            r: 0.1,
            g: 0.1,
            b: 0.2,
            a: 1.0,
        };
        if goldy_task_graph_render_pass_clear(graph, bg) != GoldyResult::Ok {
            return Err(last_ffi_message());
        }
        if goldy_task_graph_render_pass_set_pipeline(graph, pipeline) != GoldyResult::Ok {
            return Err(last_ffi_message());
        }
        if goldy_task_graph_render_pass_set_vertex_buffer(graph, 0, vertex_buffer) != GoldyResult::Ok {
            return Err(last_ffi_message());
        }
        if goldy_task_graph_render_pass_draw(graph, 0, 3, 0, 1) != GoldyResult::Ok {
            return Err(last_ffi_message());
        }
        if goldy_task_graph_render_pass_finish(graph) != GoldyResult::Ok {
            return Err(last_ffi_message());
        }

        let swapchain = goldy_task_graph_declare_swapchain_output(graph);
        if swapchain.is_null() {
            return Err("declare_swapchain_output returned null".into());
        }
        if goldy_task_graph_copy_render_target_to_swapchain(graph, scene_rt, swapchain)
            != GoldyResult::Ok
        {
            return Err(last_ffi_message());
        }

        let frame = goldy_surface_acquire(surface);
        if frame.is_null() {
            return Err(last_ffi_message());
        }
        if goldy_surface_submit_graph_to_frame(surface, graph, frame) != GoldyResult::Ok {
            return Err(last_ffi_message());
        }
        if goldy_surface_present(surface, frame) != GoldyResult::Ok {
            return Err(last_ffi_message());
        }

        Ok(())
    }

    unsafe fn shutdown(&mut self) {
        if let Some(p) = self.pipeline.take() {
            goldy_render_pipeline_destroy(p);
        }
        if let Some(s) = self.shader.take() {
            goldy_shader_destroy(s);
        }
        if let Some(b) = self.vertex_buffer.take() {
            goldy_buffer_destroy(b);
        }
        if let Some(g) = self.graph.take() {
            goldy_task_graph_destroy(g);
        }
        if let Some(rt) = self.scene_rt.take() {
            goldy_render_target_destroy(rt);
        }
        if let Some(s) = self.surface.take() {
            goldy_surface_destroy(s);
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Goldy FFI Triangle (TaskGraph)")
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        if let Err(e) = unsafe { self.init_gpu(&window) } {
            eprintln!("GPU init failed: {e}");
            event_loop.exit();
            return;
        }
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                unsafe { self.shutdown() };
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = unsafe { self.render_frame() } {
                    eprintln!("Render error: {e}");
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let Some(surface) = self.surface {
                        if unsafe { goldy_surface_resize(surface, size.width, size.height) }
                            != GoldyResult::Ok
                        {
                            eprintln!("Resize failed: {}", last_ffi_message());
                        }
                        if let Err(e) = unsafe { self.recreate_scene_rt() } {
                            eprintln!("Scene RT resize failed: {e}");
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.logical_key == Key::Named(NamedKey::Escape) {
                    unsafe { self.shutdown() };
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run app");
}
