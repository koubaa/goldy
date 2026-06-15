//! Headless integration test: two-node compute pipeline via Scheme FFI.
//!
//! Mirrors `goldy/tests/scheme_compute_integration.rs` linear chain tests.

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_compute_pipeline_create, goldy_compute_pipeline_destroy, goldy_context_create, goldy_context_destroy,
    goldy_device_destroy, goldy_instance_destroy, goldy_parcel_destroy, goldy_read_grant_byte_size,
    goldy_read_grant_consume, goldy_read_grant_destroy, goldy_retained_pool_acquire_buffer, goldy_retained_pool_create,
    goldy_retained_pool_destroy, goldy_scheme_compute_node_begin, goldy_scheme_compute_node_declare_parcel,
    goldy_scheme_compute_node_dispatch, goldy_scheme_create, goldy_scheme_destroy, goldy_scheme_grant_read,
    goldy_scheme_len, goldy_scheme_submission_destroy, goldy_scheme_submit, goldy_shader_create, goldy_shader_destroy,
    GoldyBufferKind, GoldyNodeAccess, GoldyResourceAccess, GoldyResult,
};
use std::ffi::CString;

const DOUBLE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x] * 2;
}
"#;

const ADD_TEN_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + 10;
}
"#;

unsafe fn record_compute_node(
    scheme: *mut goldy_ffi::GoldyScheme,
    label: &str,
    pipeline: *const goldy_ffi::GoldyComputePipeline,
    parcel_bindings: &[(*const goldy_ffi::GoldyParcel, GoldyNodeAccess, GoldyResourceAccess)],
    workgroups: (u32, u32, u32),
) {
    let label = CString::new(label).unwrap();
    assert_eq!(
        goldy_scheme_compute_node_begin(scheme, label.as_ptr(), pipeline),
        GoldyResult::Ok,
        "{}",
        last_ffi_message()
    );
    for &(parcel, node_access, resource_access) in parcel_bindings {
        assert_eq!(
            goldy_scheme_compute_node_declare_parcel(scheme, parcel, node_access, resource_access),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
    }
    assert_eq!(
        goldy_scheme_compute_node_dispatch(scheme, workgroups.0, workgroups.1, workgroups.2),
        GoldyResult::Ok,
        "{}",
        last_ffi_message()
    );
}

#[test]
fn scheme_compute_double_then_add_ten() {
    unsafe {
        let (instance, device) = open_device();

        let ctx = goldy_context_create(device);
        assert!(!ctx.is_null(), "{}", last_ffi_message());

        let input: Vec<u32> = (0..64).collect();
        let input_bytes =
            std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * std::mem::size_of::<u32>());

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());

        let src = goldy_retained_pool_acquire_buffer(
            pool,
            input_bytes.len() as u64,
            GoldyBufferKind::Scattered,
            std::mem::size_of::<u32>() as u32,
            input_bytes.as_ptr(),
            input_bytes.len(),
        );
        assert!(!src.is_null(), "{}", last_ffi_message());

        let dst = goldy_retained_pool_acquire_buffer(pool, 64 * 4, GoldyBufferKind::Scattered, 0, std::ptr::null(), 0);
        assert!(!dst.is_null(), "{}", last_ffi_message());

        let double_src = CString::new(DOUBLE_SHADER).unwrap();
        let double_shader = goldy_shader_create(device, double_src.as_ptr());
        assert!(!double_shader.is_null(), "{}", last_ffi_message());
        let double_pipeline = goldy_compute_pipeline_create(device, double_shader);
        assert!(!double_pipeline.is_null(), "{}", last_ffi_message());

        let add_src = CString::new(ADD_TEN_SHADER).unwrap();
        let add_shader = goldy_shader_create(device, add_src.as_ptr());
        assert!(!add_shader.is_null(), "{}", last_ffi_message());
        let add_pipeline = goldy_compute_pipeline_create(device, add_shader);
        assert!(!add_pipeline.is_null(), "{}", last_ffi_message());

        let scheme = goldy_scheme_create(ctx);
        assert!(!scheme.is_null());

        record_compute_node(
            scheme,
            "double",
            double_pipeline,
            &[
                (src, GoldyNodeAccess::Read, GoldyResourceAccess::ReadWrite),
                (dst, GoldyNodeAccess::Write, GoldyResourceAccess::Write),
            ],
            (1, 1, 1),
        );
        record_compute_node(
            scheme,
            "add_ten",
            add_pipeline,
            &[(dst, GoldyNodeAccess::ReadWrite, GoldyResourceAccess::ReadWrite)],
            (1, 1, 1),
        );

        assert_eq!(goldy_scheme_len(scheme), 2, "scheme should contain two compute nodes");

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
            let expected = i as u32 * 2 + 10;
            assert_eq!(v, expected, "index {i}: expected {expected}, got {v}");
        }

        goldy_read_grant_destroy(grant);
        goldy_scheme_destroy(scheme);
        goldy_compute_pipeline_destroy(add_pipeline);
        goldy_shader_destroy(add_shader);
        goldy_compute_pipeline_destroy(double_pipeline);
        goldy_shader_destroy(double_shader);
        goldy_parcel_destroy(dst);
        goldy_parcel_destroy(src);
        goldy_retained_pool_destroy(pool);
        goldy_context_destroy(ctx);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
