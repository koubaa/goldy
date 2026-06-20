//! Headless integration test: fill-42 compute node via Scheme FFI.
//!
//! Mirrors `goldy/tests/scheme_compute_integration.rs::scheme_graph_fill_readback`.

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_buffer_destroy, goldy_buffer_field, goldy_compute_pipeline_create, goldy_compute_pipeline_destroy,
    goldy_context_create, goldy_context_destroy, goldy_device_destroy, goldy_instance_destroy, goldy_parcel_destroy,
    goldy_read_grant_byte_size, goldy_read_grant_consume, goldy_read_grant_destroy, goldy_retained_pool_acquire_buffer,
    goldy_retained_pool_create, goldy_retained_pool_destroy, goldy_scheme_compute_node_begin,
    goldy_scheme_compute_node_dispatch, goldy_scheme_compute_node_with_param, goldy_scheme_compute_node_with_parcel,
    goldy_scheme_create, goldy_scheme_destroy, goldy_scheme_grant_read, goldy_scheme_len, goldy_scheme_replay_stats,
    goldy_scheme_submission_destroy, goldy_scheme_submit, goldy_shader_create, goldy_shader_destroy, GoldyBufferKind,
    GoldyNodeAccess, GoldyReplayStats, GoldyResult,
};
use std::ffi::CString;

const FILL_42_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = 42;
}
"#;

#[test]
fn scheme_compute_node_fills_buffer_with_42() {
    unsafe {
        let (instance, device) = open_device();

        let ctx = goldy_context_create(device);
        assert!(!ctx.is_null(), "{}", last_ffi_message());

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());
        let buffer =
            goldy_retained_pool_acquire_buffer(pool, 64 * 4, GoldyBufferKind::Scattered, 0, std::ptr::null(), 0);
        assert!(!buffer.is_null(), "{}", last_ffi_message());
        let parcel = goldy_buffer_field(buffer, 0);
        assert!(!parcel.is_null(), "{}", last_ffi_message());

        let src = CString::new(FILL_42_SHADER).unwrap();
        let shader = goldy_shader_create(device, src.as_ptr());
        assert!(!shader.is_null(), "{}", last_ffi_message());

        let pipeline = goldy_compute_pipeline_create(device, shader);
        assert!(!pipeline.is_null(), "{}", last_ffi_message());

        let scheme = goldy_scheme_create(ctx);
        assert!(!scheme.is_null());

        let label = CString::new("fill").unwrap();
        assert_eq!(
            goldy_scheme_compute_node_begin(scheme, label.as_ptr(), pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_compute_node_with_parcel(scheme, parcel, GoldyNodeAccess::Write),
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
        assert_eq!(goldy_scheme_len(scheme), 1, "scheme should contain one compute node");

        let grant = goldy_scheme_grant_read(scheme, parcel);
        assert!(!grant.is_null(), "{}", last_ffi_message());
        assert_eq!(goldy_read_grant_byte_size(grant), 64 * 4);

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

        let values: &[u32] = std::slice::from_raw_parts(
            readback.as_ptr() as *const u32,
            readback.len() / std::mem::size_of::<u32>(),
        );
        for (i, &v) in values.iter().enumerate().take(64) {
            assert_eq!(v, 42, "index {i}: got {v}");
        }

        goldy_scheme_submission_destroy(submission);

        assert_eq!(
            goldy_scheme_submit(scheme, &mut submission),
            GoldyResult::Ok,
            "second submit: {}",
            last_ffi_message()
        );
        goldy_scheme_submission_destroy(submission);

        let mut stats = GoldyReplayStats::default();
        assert_eq!(
            goldy_scheme_replay_stats(scheme, &mut stats),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(stats.records, 1, "only the first submit should record");

        goldy_read_grant_destroy(grant);
        goldy_scheme_destroy(scheme);
        goldy_compute_pipeline_destroy(pipeline);
        goldy_shader_destroy(shader);
        goldy_parcel_destroy(parcel);
        goldy_buffer_destroy(buffer);
        goldy_retained_pool_destroy(pool);
        goldy_context_destroy(ctx);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}

const UNIFORM_UINT_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint value, ThreadId id) {
    out[0] = value;
}
"#;

#[test]
fn scheme_compute_node_with_param_uint_roundtrip() {
    unsafe {
        let (instance, device) = open_device();

        let ctx = goldy_context_create(device);
        assert!(!ctx.is_null(), "{}", last_ffi_message());

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());
        let buffer = goldy_retained_pool_acquire_buffer(pool, 4, GoldyBufferKind::Scattered, 0, std::ptr::null(), 0);
        assert!(!buffer.is_null(), "{}", last_ffi_message());
        let parcel = goldy_buffer_field(buffer, 0);
        assert!(!parcel.is_null(), "{}", last_ffi_message());

        let src = CString::new(UNIFORM_UINT_SHADER).unwrap();
        let shader = goldy_shader_create(device, src.as_ptr());
        assert!(!shader.is_null(), "{}", last_ffi_message());

        let pipeline = goldy_compute_pipeline_create(device, shader);
        assert!(!pipeline.is_null(), "{}", last_ffi_message());

        let scheme = goldy_scheme_create(ctx);
        assert!(!scheme.is_null());

        let label = CString::new("uniform_uint").unwrap();
        assert_eq!(
            goldy_scheme_compute_node_begin(scheme, label.as_ptr(), pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_compute_node_with_parcel(scheme, parcel, GoldyNodeAccess::Write),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        const EXPECTED: u32 = 42;
        assert_eq!(
            goldy_scheme_compute_node_with_param(scheme, EXPECTED),
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

        let grant = goldy_scheme_grant_read(scheme, parcel);
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

        let value = u32::from_le_bytes(readback[..4].try_into().unwrap());
        assert_eq!(value, EXPECTED);

        goldy_scheme_submission_destroy(submission);
        goldy_read_grant_destroy(grant);
        goldy_scheme_destroy(scheme);
        goldy_compute_pipeline_destroy(pipeline);
        goldy_shader_destroy(shader);
        goldy_parcel_destroy(parcel);
        goldy_buffer_destroy(buffer);
        goldy_retained_pool_destroy(pool);
        goldy_context_destroy(ctx);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
