//! Headless compute example using the Goldy C ABI from Rust.
//!
//! Run from `goldy/ffi`:
//! `cargo run --example compute_simple --features examples`

use goldy_ffi::{
    goldy_buffer_create_with_data, goldy_buffer_destroy, goldy_compute_encoder_bind_resources,
    goldy_compute_encoder_create, goldy_compute_encoder_destroy, goldy_compute_encoder_dispatch,
    goldy_compute_encoder_execute, goldy_compute_encoder_set_pipeline,
    goldy_compute_pipeline_create, goldy_compute_pipeline_destroy, goldy_device_destroy,
    goldy_get_last_error, goldy_instance_adapter_count, goldy_instance_create,
    goldy_instance_create_device_for_adapter, goldy_instance_destroy, goldy_instance_get_adapter,
    goldy_shader_create, goldy_shader_destroy, GoldyAdapterInfo, GoldyBuffer, GoldyBufferKind,
    GoldyComputeEncoder, GoldyComputePipeline, GoldyDevice, GoldyDeviceType, GoldyInstance,
    GoldyResult, GoldyShaderModule,
};
use std::ffi::{CStr, CString};

const COMPUTE_SRC: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<float> data, ThreadId id) {
    uint idx = id.x;
    if (idx < 64u) {
        data[idx] = float(idx) * 2.0;
    }
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

unsafe fn request_device_for_discrete_gpu(instance: *const GoldyInstance) -> *mut GoldyDevice {
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

fn main() {
    println!("Goldy FFI compute_simple (Rust client of the C ABI)\n");

    unsafe {
        let instance: *mut GoldyInstance = goldy_instance_create();
        assert!(!instance.is_null(), "{}", last_ffi_message());

        let device: *mut GoldyDevice = request_device_for_discrete_gpu(instance);
        assert!(!device.is_null(), "{}", last_ffi_message());

        let data = [0f32; 64];
        let buf = goldy_buffer_create_with_data(
            device,
            bytemuck::cast_slice(&data).as_ptr(),
            std::mem::size_of_val(&data),
            GoldyBufferKind::Scattered,
        );
        assert!(!buf.is_null(), "{}", last_ffi_message());

        let c_src = CString::new(COMPUTE_SRC).expect("shader source has no interior nul");
        let shader: *mut GoldyShaderModule = goldy_shader_create(device, c_src.as_ptr());
        assert!(!shader.is_null(), "{}", last_ffi_message());

        let pipeline: *mut GoldyComputePipeline = goldy_compute_pipeline_create(device, shader);
        assert!(!pipeline.is_null(), "{}", last_ffi_message());

        let encoder: *mut GoldyComputeEncoder = goldy_compute_encoder_create();
        assert!(!encoder.is_null());

        goldy_compute_encoder_set_pipeline(encoder, pipeline);
        let buf_ptr: *const GoldyBuffer = buf;
        let bind = [buf_ptr];
        goldy_compute_encoder_bind_resources(encoder, bind.as_ptr(), 1);
        goldy_compute_encoder_dispatch(encoder, 1, 1, 1);

        assert_eq!(
            goldy_compute_encoder_execute(encoder, device),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        println!("Compute dispatch completed successfully.");

        goldy_compute_encoder_destroy(encoder);
        goldy_compute_pipeline_destroy(pipeline);
        goldy_shader_destroy(shader);
        goldy_buffer_destroy(buf);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
