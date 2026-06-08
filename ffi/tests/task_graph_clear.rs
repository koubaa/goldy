//! Headless integration test: clear an offscreen render target via TaskGraph FFI.

use goldy_ffi::{
    goldy_device_destroy, goldy_get_last_error, goldy_instance_adapter_count, goldy_instance_create,
    goldy_instance_create_device_for_adapter, goldy_instance_destroy, goldy_instance_get_adapter,
    goldy_render_target_buffer_size, goldy_render_target_create, goldy_render_target_destroy,
    goldy_render_target_read_to_buffer, goldy_task_graph_clear, goldy_task_graph_create,
    goldy_task_graph_declare_swapchain_output, goldy_task_graph_destroy, goldy_task_graph_dispatch,
    goldy_task_graph_render_pass_begin, goldy_task_graph_render_pass_clear,
    goldy_task_graph_render_pass_finish, GoldyAdapterInfo, GoldyColor, GoldyDevice, GoldyDeviceType,
    GoldyInstance, GoldyResult, GoldyTextureFormat,
};
use std::ffi::CStr;

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
fn task_graph_clear_render_target_readback_is_red() {
    unsafe {
        let instance = goldy_instance_create();
        assert!(!instance.is_null(), "{}", last_ffi_message());

        let device = request_device(instance);
        assert!(!device.is_null(), "{}", last_ffi_message());

        let target = goldy_render_target_create(device, 2, 2, GoldyTextureFormat::Rgba8Unorm);
        assert!(!target.is_null(), "{}", last_ffi_message());

        let graph = goldy_task_graph_create();
        assert!(!graph.is_null());

        let label = std::ffi::CString::new("clear_red").unwrap();
        assert_eq!(
            goldy_task_graph_render_pass_begin(graph, label.as_ptr(), target),
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
            goldy_task_graph_render_pass_clear(graph, red),
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
        assert_eq!(size, 2 * 2 * 4);
        let mut pixels = vec![0u8; size];
        assert_eq!(
            goldy_render_target_read_to_buffer(target, pixels.as_mut_ptr(), pixels.len()),
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

        assert_eq!(goldy_task_graph_clear(graph), GoldyResult::Ok);

        goldy_task_graph_destroy(graph);
        goldy_render_target_destroy(target);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}

#[test]
fn swapchain_output_token_is_graph_owned_and_reusable() {
    unsafe {
        let graph = goldy_task_graph_create();
        assert!(!graph.is_null());

        let token_a = goldy_task_graph_declare_swapchain_output(graph);
        let token_b = goldy_task_graph_declare_swapchain_output(graph);
        assert!(!token_a.is_null());
        assert_eq!(token_a, token_b, "swapchain token should be a per-graph sentinel");

        assert_eq!(goldy_task_graph_clear(graph), GoldyResult::Ok);

        let token_c = goldy_task_graph_declare_swapchain_output(graph);
        assert_eq!(token_a, token_c, "token address stable across graph clear");

        goldy_task_graph_destroy(graph);
    }
}
