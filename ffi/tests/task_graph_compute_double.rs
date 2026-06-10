//! Headless integration test: two-node compute pipeline via TaskGraph FFI.
//!
//! Mirrors `goldy/tests/task_graph_integration.rs::graph_matches_encoder` (TaskGraph path).

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_compute_pipeline_create, goldy_compute_pipeline_destroy, goldy_device_destroy, goldy_instance_destroy,
    goldy_parcel_byte_size, goldy_parcel_destroy, goldy_parcel_read_to_cpu, goldy_parcel_resource_index,
    goldy_retained_pool_acquire_buffer, goldy_retained_pool_create, goldy_retained_pool_destroy, goldy_shader_create,
    goldy_shader_destroy, goldy_task_graph_compute_node_begin, goldy_task_graph_compute_node_bind_parcel,
    goldy_task_graph_compute_node_bind_resources_raw, goldy_task_graph_compute_node_dispatch, goldy_task_graph_create,
    goldy_task_graph_destroy, goldy_task_graph_dispatch, GoldyBufferKind, GoldyNodeAccess, GoldyResourceAccess,
    GoldyResult,
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
    graph: *mut goldy_ffi::GoldyTaskGraph,
    label: &str,
    pipeline: *const goldy_ffi::GoldyComputePipeline,
    parcel_bindings: &[(*const goldy_ffi::GoldyParcel, GoldyNodeAccess)],
    resource_indices: &[u32],
    workgroups: (u32, u32, u32),
) {
    let label = CString::new(label).unwrap();
    assert_eq!(
        goldy_task_graph_compute_node_begin(graph, label.as_ptr(), pipeline),
        GoldyResult::Ok,
        "{}",
        last_ffi_message()
    );
    for &(parcel, access) in parcel_bindings {
        assert_eq!(
            goldy_task_graph_compute_node_bind_parcel(graph, parcel, access),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
    }
    assert_eq!(
        goldy_task_graph_compute_node_bind_resources_raw(
            graph,
            resource_indices.as_ptr(),
            resource_indices.len() as u32
        ),
        GoldyResult::Ok,
        "{}",
        last_ffi_message()
    );
    assert_eq!(
        goldy_task_graph_compute_node_dispatch(graph, workgroups.0, workgroups.1, workgroups.2),
        GoldyResult::Ok,
        "{}",
        last_ffi_message()
    );
}

#[test]
fn task_graph_compute_double_then_add_ten() {
    unsafe {
        let (instance, device) = open_device();

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

        let dst = goldy_retained_pool_acquire_buffer(
            pool,
            64 * 4,
            GoldyBufferKind::Scattered,
            0,
            std::ptr::null(),
            0,
        );
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

        // `DOUBLE_SHADER` reads `src` as `Scattered<uint>` (UAV on DX12).
        let src_idx = goldy_parcel_resource_index(src, GoldyResourceAccess::ReadWrite);
        let dst_idx = goldy_parcel_resource_index(dst, GoldyResourceAccess::Write);
        assert_ne!(src_idx, u32::MAX);
        assert_ne!(dst_idx, u32::MAX);

        let graph = goldy_task_graph_create();
        assert!(!graph.is_null());

        record_compute_node(
            graph,
            "double",
            double_pipeline,
            &[(src, GoldyNodeAccess::Read), (dst, GoldyNodeAccess::Write)],
            &[src_idx, dst_idx],
            (1, 1, 1),
        );
        record_compute_node(
            graph,
            "add_ten",
            add_pipeline,
            &[(dst, GoldyNodeAccess::ReadWrite)],
            &[dst_idx],
            (1, 1, 1),
        );

        assert_eq!(
            goldy_ffi::goldy_task_graph_len(graph),
            2,
            "graph should contain two compute nodes"
        );
        assert_eq!(
            goldy_task_graph_dispatch(graph, device),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let mut readback = vec![0u8; goldy_parcel_byte_size(dst) as usize];
        assert_eq!(
            goldy_parcel_read_to_cpu(dst, device, readback.as_mut_ptr(), readback.len()),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let values: &[u32] = std::slice::from_raw_parts(
            readback.as_ptr() as *const u32,
            readback.len() / std::mem::size_of::<u32>(),
        );
        for (i, &v) in values.iter().enumerate().take(64) {
            let expected = i as u32 * 2 + 10;
            assert_eq!(v, expected, "index {i}: expected {expected}, got {v}");
        }

        goldy_task_graph_destroy(graph);
        goldy_compute_pipeline_destroy(add_pipeline);
        goldy_shader_destroy(add_shader);
        goldy_compute_pipeline_destroy(double_pipeline);
        goldy_shader_destroy(double_shader);
        goldy_parcel_destroy(dst);
        goldy_parcel_destroy(src);
        goldy_retained_pool_destroy(pool);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
