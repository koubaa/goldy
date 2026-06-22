//! Headless integration test: clear an offscreen render target via Scheme FFI.

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_context_create, goldy_context_destroy, goldy_device_destroy, goldy_instance_destroy, goldy_texture_destroy,
    goldy_read_grant_byte_size, goldy_read_grant_consume, goldy_read_grant_destroy,
    goldy_retained_pool_acquire_texture, goldy_retained_pool_create, goldy_retained_pool_destroy,
    goldy_scheme_copy_to_texture, goldy_scheme_create, goldy_scheme_destroy, goldy_scheme_grant_read_texture,
    goldy_scheme_lease_render_target, goldy_scheme_render_pass_begin, goldy_scheme_render_pass_clear,
    goldy_scheme_render_pass_finish, goldy_scheme_render_target_lease_destroy, goldy_scheme_submission_destroy,
    goldy_scheme_submit, GoldyColor, GoldyDepthFormat, GoldyResult, GoldyTextureFlags, GoldyTextureFormat,
    GoldyTextureKind,
};
use std::ffi::CString;

#[test]
fn scheme_clear_render_target_readback_is_red() {
    const W: u32 = 2;
    const H: u32 = 2;

    unsafe {
        let (instance, device) = open_device();

        let ctx = goldy_context_create(device);
        assert!(!ctx.is_null(), "{}", last_ffi_message());

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());
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

        let label = CString::new("clear_red").unwrap();
        assert_eq!(
            goldy_scheme_render_pass_begin(scheme, label.as_ptr(), rt),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        let red = GoldyColor {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        assert_eq!(
            goldy_scheme_render_pass_clear(scheme, red),
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
        assert_eq!(goldy_read_grant_byte_size(grant), (W * H * 4) as u64);

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

        for chunk in pixels.chunks_exact(4) {
            assert_eq!(chunk[0], 255, "R");
            assert_eq!(chunk[1], 0, "G");
            assert_eq!(chunk[2], 0, "B");
            assert_eq!(chunk[3], 255, "A");
        }

        goldy_scheme_submission_destroy(submission);
        goldy_read_grant_destroy(grant);
        goldy_scheme_render_target_lease_destroy(rt);
        goldy_scheme_destroy(scheme);
        goldy_texture_destroy(readback);
        goldy_retained_pool_destroy(pool);
        goldy_context_destroy(ctx);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
