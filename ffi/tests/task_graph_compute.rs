//! Headless integration test: compute dispatch via TaskGraph FFI.

use goldy_ffi::{
    goldy_buffer_create_with_data_stride, goldy_buffer_destroy, goldy_buffer_resource_index, goldy_buffer_size,
    goldy_compute_pipeline_create, goldy_compute_pipeline_destroy, goldy_device_destroy, goldy_get_last_error,
    goldy_instance_adapter_count, goldy_instance_create, goldy_instance_create_device_for_adapter,
    goldy_instance_destroy, goldy_instance_get_adapter, goldy_shader_create, goldy_shader_destroy,
    goldy_task_graph_compute_node_begin, goldy_task_graph_compute_node_bind_buffer,
    goldy_task_graph_compute_node_bind_resources_raw, goldy_task_graph_compute_node_dispatch, goldy_task_graph_create,
    goldy_task_graph_destroy, goldy_task_graph_dispatch, GoldyAdapterInfo, GoldyBufferKind, GoldyDevice,
    GoldyDeviceType, GoldyInstance, GoldyResourceAccess, GoldyResult,
};
use std::ffi::{CStr, CString};

const COMPUTE_SRC: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = 42;
}
"#;

fn last_ffi_message() -> String {
    unsafe {
        let p = goldy_get_last_error();
        if p.is_null() {
            return "(no message)".into();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

unsafe fn request_device(instance: *const GoldyInstance) -> *mut GoldyDevice {
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

#[test]
fn task_graph_compute_node_doubles_buffer_values() {
    unsafe {
        let instance = goldy_instance_create();
        assert!(!instance.is_null(), "{}", last_ffi_message());

        let device = request_device(instance);
        assert!(!device.is_null(), "{}", last_ffi_message());

        let initial: Vec<u32> = (0..64).collect();
        let bytes = std::slice::from_raw_parts(
            initial.as_ptr() as *const u8,
            initial.len() * std::mem::size_of::<u32>(),
        );
        let buffer = goldy_buffer_create_with_data_stride(
            device,
            bytes.as_ptr(),
            bytes.len(),
            GoldyBufferKind::Scattered,
            std::mem::size_of::<u32>() as u32,
        );
        assert!(!buffer.is_null(), "{}", last_ffi_message());

        let src = CString::new(COMPUTE_SRC).unwrap();
        let shader = goldy_shader_create(device, src.as_ptr());
        assert!(!shader.is_null(), "{}", last_ffi_message());

        let pipeline = goldy_compute_pipeline_create(device, shader);
        assert!(!pipeline.is_null(), "{}", last_ffi_message());

        let graph = goldy_task_graph_create();
        assert!(!graph.is_null());

        let label = CString::new("double").unwrap();
        assert_eq!(
            goldy_task_graph_compute_node_begin(graph, label.as_ptr(), pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_compute_node_bind_buffer(graph, buffer, goldy_ffi::GoldyNodeAccess::Write),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let idx = goldy_buffer_resource_index(buffer, GoldyResourceAccess::Write);
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

        let mut readback = vec![0u8; goldy_buffer_size(buffer) as usize];
        assert_eq!(
            goldy_ffi::goldy_buffer_read_to_cpu(buffer, device, readback.as_mut_ptr(), readback.len()),
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
        goldy_buffer_destroy(buffer);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
