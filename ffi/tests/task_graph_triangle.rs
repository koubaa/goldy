//! Headless integration test: triangle render pass via TaskGraph FFI.
//!
//! Mirrors `goldy/tests/surface_graph_integration.rs::render_pass_task_graph_triangle_readback`.

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_device_destroy, goldy_instance_destroy, goldy_parcel_destroy, goldy_render_pipeline_create,
    goldy_render_pipeline_destroy, goldy_render_target_buffer_size, goldy_render_target_create,
    goldy_render_target_destroy, goldy_render_target_read_to_buffer, goldy_retained_pool_acquire_buffer,
    goldy_retained_pool_create, goldy_retained_pool_destroy, goldy_shader_builtin_vertex_color_2d, goldy_shader_create,
    goldy_shader_destroy, goldy_task_graph_create, goldy_task_graph_destroy, goldy_task_graph_dispatch,
    goldy_task_graph_render_pass_begin, goldy_task_graph_render_pass_clear, goldy_task_graph_render_pass_draw,
    goldy_task_graph_render_pass_finish, goldy_task_graph_render_pass_set_pipeline,
    goldy_task_graph_render_pass_set_vertex_buffer_parcel, goldy_task_graph_render_pass_with_parcel, GoldyBufferKind,
    GoldyColor, GoldyNodeAccess, GoldyRenderPipelineDesc, GoldyResult, GoldyTextureFormat, GoldyVertexAttribute,
    GoldyVertexFormat,
};
use std::ffi::CString;
use std::mem::size_of;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex2D {
    position: [f32; 2],
    color: [f32; 4],
}

#[test]
fn task_graph_triangle_readback_center_pixel_lit() {
    const W: u32 = 64;
    const H: u32 = 64;
    let clear = GoldyColor {
        r: 0.1,
        g: 0.1,
        b: 0.2,
        a: 1.0,
    };

    unsafe {
        let (instance, device) = open_device();

        let vertices = [
            Vertex2D {
                position: [0.0, -0.5],
                color: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex2D {
                position: [-0.5, 0.5],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex2D {
                position: [0.5, 0.5],
                color: [0.0, 0.0, 1.0, 1.0],
            },
        ];
        let vertex_bytes =
            std::slice::from_raw_parts(vertices.as_ptr() as *const u8, vertices.len() * size_of::<Vertex2D>());

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());
        let vertex_buffer = goldy_retained_pool_acquire_buffer(
            pool,
            vertex_bytes.len() as u64,
            GoldyBufferKind::Scattered,
            0,
            vertex_bytes.as_ptr(),
            vertex_bytes.len(),
        );
        assert!(!vertex_buffer.is_null(), "{}", last_ffi_message());

        let target = goldy_render_target_create(device, W, H, GoldyTextureFormat::Rgba8Unorm);
        assert!(!target.is_null(), "{}", last_ffi_message());

        let builtin = goldy_shader_builtin_vertex_color_2d();
        let shader = goldy_shader_create(device, builtin);
        assert!(!shader.is_null(), "{}", last_ffi_message());

        let attributes = [
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
            vertex_attributes: attributes.as_ptr(),
            vertex_attribute_count: attributes.len() as u32,
            vertex_stride: size_of::<Vertex2D>() as u32,
            topology: goldy_ffi::GoldyPrimitiveTopology::TriangleList,
            target_format: GoldyTextureFormat::Rgba8Unorm,
            depth_enabled: false,
            ..Default::default()
        };
        let pipeline = goldy_render_pipeline_create(device, shader, shader, &pipeline_desc);
        assert!(!pipeline.is_null(), "{}", last_ffi_message());

        let graph = goldy_task_graph_create();
        assert!(!graph.is_null());

        let label = CString::new("triangle").unwrap();
        assert_eq!(
            goldy_task_graph_render_pass_begin(graph, label.as_ptr(), target),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_with_parcel(graph, vertex_buffer, GoldyNodeAccess::Read),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_clear(graph, clear),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_set_pipeline(graph, pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_set_vertex_buffer_parcel(graph, 0, vertex_buffer),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_draw(graph, 0, 3, 0, 1),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_finish(graph),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_dispatch(graph, device),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let size = goldy_render_target_buffer_size(target);
        assert_eq!(size, (W * H * 4) as usize);
        let mut pixels = vec![0u8; size];
        assert_eq!(
            goldy_render_target_read_to_buffer(target, pixels.as_mut_ptr(), pixels.len()),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let stride = (W * 4) as usize;
        let cx = (W / 2) as usize;
        let cy = (H / 2) as usize;
        let i = cy * stride + cx * 4;
        let r = pixels[i];
        let g = pixels[i + 1];
        let b = pixels[i + 2];

        assert!(
            r > 20 || g > 20 || b > 20,
            "center pixel should be lit by the triangle, got rgba=({r},{g},{b},{})",
            pixels[i + 3]
        );
        let clear_r = (clear.r * 255.0) as i32;
        let clear_g = (clear.g * 255.0) as i32;
        let clear_b = (clear.b * 255.0) as i32;
        assert!(
            (r as i32 - clear_r).abs() > 5 || (g as i32 - clear_g).abs() > 5 || (b as i32 - clear_b).abs() > 5,
            "center pixel should differ from clear color, got rgba=({r},{g},{b})"
        );

        goldy_task_graph_destroy(graph);
        goldy_render_pipeline_destroy(pipeline);
        goldy_shader_destroy(shader);
        goldy_render_target_destroy(target);
        goldy_parcel_destroy(vertex_buffer);
        goldy_retained_pool_destroy(pool);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
