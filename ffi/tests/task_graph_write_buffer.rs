//! Headless integration test: write_parcel node via TaskGraph FFI.
//!
//! Mirrors `goldy/tests/task_graph_integration.rs::write_then_dispatch_reads_uploaded_data`.

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_compute_pipeline_create, goldy_compute_pipeline_destroy, goldy_device_destroy, goldy_instance_destroy,
    goldy_parcel_byte_size, goldy_parcel_destroy, goldy_parcel_read_to_cpu, goldy_parcel_resource_index,
    goldy_retained_pool_acquire_buffer, goldy_retained_pool_create, goldy_retained_pool_destroy, goldy_shader_create,
    goldy_shader_destroy, goldy_task_graph_compute_node_begin, goldy_task_graph_compute_node_bind_parcel,
    goldy_task_graph_compute_node_bind_resources_raw, goldy_task_graph_compute_node_dispatch, goldy_task_graph_create,
    goldy_task_graph_destroy, goldy_task_graph_dispatch, goldy_task_graph_write_parcel, GoldyBufferKind,
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
fn task_graph_write_parcel_then_dispatch_reads_uploaded_data() {
    unsafe {
        let (instance, device) = open_device();

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());

        let buf = goldy_retained_pool_acquire_buffer(pool, 64 * 4, GoldyBufferKind::Scattered, 0, std::ptr::null(), 0);
        assert!(!buf.is_null(), "{}", last_ffi_message());
        let out = goldy_retained_pool_acquire_buffer(pool, 64 * 4, GoldyBufferKind::Scattered, 0, std::ptr::null(), 0);
        assert!(!out.is_null(), "{}", last_ffi_message());

        let known_data: Vec<u32> = (100..164).collect();
        let data_bytes: Vec<u8> = bytemuck::cast_slice(&known_data).to_vec();

        let copy_src = CString::new(COPY_SHADER).unwrap();
        let shader = goldy_shader_create(device, copy_src.as_ptr());
        assert!(!shader.is_null(), "{}", last_ffi_message());
        let pipeline = goldy_compute_pipeline_create(device, shader);
        assert!(!pipeline.is_null(), "{}", last_ffi_message());

        let buf_idx = goldy_parcel_resource_index(buf, GoldyResourceAccess::Write);
        let out_idx = goldy_parcel_resource_index(out, GoldyResourceAccess::Write);
        assert_ne!(buf_idx, u32::MAX);
        assert_ne!(out_idx, u32::MAX);

        let graph = goldy_task_graph_create();
        assert!(!graph.is_null());

        assert_eq!(
            goldy_task_graph_write_parcel(graph, buf, 0, data_bytes.as_ptr(), data_bytes.len()),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let label = CString::new("copy").unwrap();
        assert_eq!(
            goldy_task_graph_compute_node_begin(graph, label.as_ptr(), pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_compute_node_bind_parcel(graph, buf, GoldyNodeAccess::Read),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_compute_node_bind_parcel(graph, out, GoldyNodeAccess::Write),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_compute_node_bind_resources_raw(graph, [buf_idx, out_idx].as_ptr(), 2),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_compute_node_dispatch(graph, 1, 1, 1),
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

        let mut readback = vec![0u8; goldy_parcel_byte_size(out) as usize];
        assert_eq!(
            goldy_parcel_read_to_cpu(out, device, readback.as_mut_ptr(), readback.len()),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let values: &[u32] = std::slice::from_raw_parts(
            readback.as_ptr() as *const u32,
            readback.len() / std::mem::size_of::<u32>(),
        );
        for (i, &v) in values.iter().enumerate().take(64) {
            let expected = known_data[i];
            assert_eq!(v, expected, "index {i}: expected {expected}, got {v}");
        }

        goldy_task_graph_destroy(graph);
        goldy_compute_pipeline_destroy(pipeline);
        goldy_shader_destroy(shader);
        goldy_parcel_destroy(out);
        goldy_parcel_destroy(buf);
        goldy_retained_pool_destroy(pool);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
