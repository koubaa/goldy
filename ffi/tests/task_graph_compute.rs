//! Headless integration test: fill-42 compute node via TaskGraph FFI.
//!
//! Mirrors `goldy/tests/task_graph_integration.rs::graph_nonblocking_submit` (FILL_42_SHADER).

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_compute_pipeline_create, goldy_compute_pipeline_destroy, goldy_device_destroy, goldy_instance_destroy,
    goldy_parcel_byte_size, goldy_parcel_destroy, goldy_parcel_read_to_cpu, goldy_parcel_resource_index,
    goldy_retained_pool_acquire_buffer, goldy_retained_pool_create, goldy_retained_pool_destroy, goldy_shader_create,
    goldy_shader_destroy, goldy_task_graph_compute_node_begin, goldy_task_graph_compute_node_bind_parcel,
    goldy_task_graph_compute_node_bind_resources_raw, goldy_task_graph_compute_node_dispatch, goldy_task_graph_create,
    goldy_task_graph_destroy, goldy_task_graph_dispatch, GoldyBufferKind, GoldyResourceAccess, GoldyResult,
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
fn task_graph_compute_node_fills_buffer_with_42() {
    unsafe {
        let (instance, device) = open_device();

        let initial: Vec<u32> = (0..64).collect();
        let bytes = std::slice::from_raw_parts(
            initial.as_ptr() as *const u8,
            initial.len() * std::mem::size_of::<u32>(),
        );

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());
        let buffer = goldy_retained_pool_acquire_buffer(
            pool,
            bytes.len() as u64,
            GoldyBufferKind::Scattered,
            std::mem::size_of::<u32>() as u32,
            bytes.as_ptr(),
            bytes.len(),
        );
        assert!(!buffer.is_null(), "{}", last_ffi_message());

        let src = CString::new(FILL_42_SHADER).unwrap();
        let shader = goldy_shader_create(device, src.as_ptr());
        assert!(!shader.is_null(), "{}", last_ffi_message());

        let pipeline = goldy_compute_pipeline_create(device, shader);
        assert!(!pipeline.is_null(), "{}", last_ffi_message());

        let graph = goldy_task_graph_create();
        assert!(!graph.is_null());

        let label = CString::new("fill").unwrap();
        assert_eq!(
            goldy_task_graph_compute_node_begin(graph, label.as_ptr(), pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_compute_node_bind_parcel(graph, buffer, goldy_ffi::GoldyNodeAccess::Write),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let idx = goldy_parcel_resource_index(buffer, GoldyResourceAccess::Write);
        assert_ne!(idx, u32::MAX);
        assert_eq!(
            goldy_task_graph_compute_node_bind_resources_raw(graph, &idx as *const u32, 1),
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
            goldy_ffi::goldy_task_graph_len(graph),
            1,
            "graph should contain one compute node"
        );
        assert_eq!(
            goldy_task_graph_dispatch(graph, device),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let mut readback = vec![0u8; goldy_parcel_byte_size(buffer) as usize];
        assert_eq!(
            goldy_parcel_read_to_cpu(buffer, device, readback.as_mut_ptr(), readback.len()),
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

        goldy_task_graph_destroy(graph);
        goldy_compute_pipeline_destroy(pipeline);
        goldy_shader_destroy(shader);
        goldy_parcel_destroy(buffer);
        goldy_retained_pool_destroy(pool);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
