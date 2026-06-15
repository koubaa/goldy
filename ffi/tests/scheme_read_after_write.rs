//! Headless integration test: copy compute after initial buffer upload via acquire.
//!
//! Mirrors `goldy/tests/scheme_compute_integration.rs::scheme_regular_buffer_write_then_copy`
//! (initial data via `acquire_buffer`, no separate upload FFI call).

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_compute_pipeline_create, goldy_compute_pipeline_destroy, goldy_context_create, goldy_context_destroy,
    goldy_device_destroy, goldy_instance_destroy, goldy_parcel_destroy, goldy_read_grant_byte_size,
    goldy_read_grant_consume, goldy_read_grant_destroy, goldy_retained_pool_acquire_buffer, goldy_retained_pool_create,
    goldy_retained_pool_destroy, goldy_scheme_compute_node_begin, goldy_scheme_compute_node_declare_parcel,
    goldy_scheme_compute_node_dispatch, goldy_scheme_create, goldy_scheme_destroy, goldy_scheme_grant_read,
    goldy_scheme_submission_destroy, goldy_scheme_submit, goldy_shader_create, goldy_shader_destroy, GoldyBufferKind,
    GoldyNodeAccess, GoldyResourceAccess, GoldyResult,
};
use std::ffi::CString;

const COPY_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x];
}
"#;

#[test]
fn scheme_read_after_acquire_then_copy() {
    unsafe {
        let (instance, device) = open_device();

        let ctx = goldy_context_create(device);
        assert!(!ctx.is_null(), "{}", last_ffi_message());

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());

        let known_data: Vec<u32> = (100..164).collect();
        let data_bytes: Vec<u8> = bytemuck::cast_slice(&known_data).to_vec();

        let src = goldy_retained_pool_acquire_buffer(
            pool,
            data_bytes.len() as u64,
            GoldyBufferKind::Scattered,
            std::mem::size_of::<u32>() as u32,
            data_bytes.as_ptr(),
            data_bytes.len(),
        );
        assert!(!src.is_null(), "{}", last_ffi_message());

        let dst = goldy_retained_pool_acquire_buffer(pool, 64 * 4, GoldyBufferKind::Scattered, 0, std::ptr::null(), 0);
        assert!(!dst.is_null(), "{}", last_ffi_message());

        let copy_src = CString::new(COPY_SHADER).unwrap();
        let shader = goldy_shader_create(device, copy_src.as_ptr());
        assert!(!shader.is_null(), "{}", last_ffi_message());
        let pipeline = goldy_compute_pipeline_create(device, shader);
        assert!(!pipeline.is_null(), "{}", last_ffi_message());

        let scheme = goldy_scheme_create(ctx);
        assert!(!scheme.is_null());

        let label = CString::new("copy").unwrap();
        assert_eq!(
            goldy_scheme_compute_node_begin(scheme, label.as_ptr(), pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_compute_node_declare_parcel(
                scheme,
                src,
                GoldyNodeAccess::Read,
                GoldyResourceAccess::ReadWrite
            ),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_compute_node_declare_parcel(scheme, dst, GoldyNodeAccess::Write, GoldyResourceAccess::Write),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_compute_node_dispatch(scheme, 1, 1, 1),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        let grant = goldy_scheme_grant_read(scheme, dst);
        assert!(!grant.is_null(), "{}", last_ffi_message());

        let mut submission = std::ptr::null_mut();
        assert_eq!(
            goldy_scheme_submit(scheme, &mut submission),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert!(!submission.is_null());

        let mut readback = vec![0u8; goldy_read_grant_byte_size(grant) as usize];
        assert_eq!(
            goldy_read_grant_consume(grant, submission, readback.as_mut_ptr(), readback.len()),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        goldy_scheme_submission_destroy(submission);

        let values: &[u32] = std::slice::from_raw_parts(
            readback.as_ptr() as *const u32,
            readback.len() / std::mem::size_of::<u32>(),
        );
        for (i, &v) in values.iter().enumerate().take(64) {
            let expected = 100 + i as u32;
            assert_eq!(v, expected, "index {i}: expected {expected}, got {v}");
        }

        goldy_read_grant_destroy(grant);
        goldy_scheme_destroy(scheme);
        goldy_compute_pipeline_destroy(pipeline);
        goldy_shader_destroy(shader);
        goldy_parcel_destroy(dst);
        goldy_parcel_destroy(src);
        goldy_retained_pool_destroy(pool);
        goldy_context_destroy(ctx);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
