//! Headless integration test: triangle render pass via Scheme FFI.

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_buffer_destroy, goldy_buffer_field, goldy_context_create, goldy_context_destroy, goldy_device_destroy,
    goldy_instance_destroy, goldy_parcel_destroy, goldy_read_grant_consume, goldy_read_grant_destroy,
    goldy_render_pipeline_create, goldy_render_pipeline_destroy, goldy_retained_pool_acquire_buffer,
    goldy_retained_pool_acquire_texture, goldy_retained_pool_create, goldy_retained_pool_destroy,
    goldy_scheme_copy_to_texture, goldy_scheme_create, goldy_scheme_destroy, goldy_scheme_grant_read_texture,
    goldy_scheme_lease_render_target, goldy_scheme_render_pass_begin, goldy_scheme_render_pass_clear,
    goldy_scheme_render_pass_draw, goldy_scheme_render_pass_finish, goldy_scheme_render_pass_set_pipeline,
    goldy_scheme_render_pass_set_vertex_buffer_parcel, goldy_scheme_render_pass_with_parcel,
    goldy_scheme_render_target_lease_destroy, goldy_scheme_submission_destroy, goldy_scheme_submit,
    goldy_shader_builtin_vertex_color_2d, goldy_shader_create, goldy_shader_destroy, goldy_texture_destroy,
    GoldyBufferKind, GoldyColor, GoldyDepthFormat, GoldyNodeAccess, GoldyRenderPipelineDesc, GoldyResult,
    GoldyTextureFlags, GoldyTextureFormat, GoldyTextureKind, GoldyVertexAttribute, GoldyVertexFormat,
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
fn scheme_triangle_readback_center_pixel_lit() {
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

        let ctx = goldy_context_create(device);
        assert!(!ctx.is_null(), "{}", last_ffi_message());

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
        let vertex_parcel = goldy_buffer_field(vertex_buffer, 0);
        assert!(!vertex_parcel.is_null(), "{}", last_ffi_message());

        let readback = goldy_retained_pool_acquire_texture(
            pool,
            W,
            H,
            GoldyTextureFormat::Rgba8Unorm,
            GoldyTextureKind::Direct,
            GoldyTextureFlags(GoldyTextureFlags::COPY_SRC.0 | GoldyTextureFlags::COPY_DST.0),
            std::ptr::null(),
            0,
        );
        assert!(!readback.is_null(), "{}", last_ffi_message());

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

        let scheme = goldy_scheme_create(ctx);
        assert!(!scheme.is_null());

        let rt = goldy_scheme_lease_render_target(
            scheme,
            W,
            H,
            GoldyTextureFormat::Rgba8Unorm,
            false,
            GoldyDepthFormat::Depth24Plus,
        );
        assert!(!rt.is_null(), "{}", last_ffi_message());

        let label = CString::new("triangle").unwrap();
        assert_eq!(
            goldy_scheme_render_pass_begin(scheme, label.as_ptr(), rt),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_with_parcel(scheme, vertex_parcel, GoldyNodeAccess::Read),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_clear(scheme, clear),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_set_pipeline(scheme, pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_set_vertex_buffer_parcel(scheme, 0, vertex_parcel),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_draw(scheme, 0, 3, 0, 1),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_finish(scheme),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        assert_eq!(
            goldy_scheme_copy_to_texture(scheme, rt, readback),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let grant = goldy_scheme_grant_read_texture(scheme, readback);
        assert!(!grant.is_null(), "{}", last_ffi_message());

        let mut submission = std::ptr::null_mut();
        assert_eq!(
            goldy_scheme_submit(scheme, &mut submission),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert!(!submission.is_null());

        let mut pixels = vec![0u8; (W * H * 4) as usize];
        assert_eq!(
            goldy_read_grant_consume(grant, submission, pixels.as_mut_ptr(), pixels.len()),
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

        goldy_scheme_submission_destroy(submission);
        goldy_read_grant_destroy(grant);
        goldy_scheme_render_target_lease_destroy(rt);
        goldy_scheme_destroy(scheme);
        goldy_render_pipeline_destroy(pipeline);
        goldy_shader_destroy(shader);
        goldy_parcel_destroy(vertex_parcel);
        goldy_texture_destroy(readback);
        goldy_buffer_destroy(vertex_buffer);
        goldy_retained_pool_destroy(pool);
        goldy_context_destroy(ctx);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
