//! FFI bindings for ShaderModule.

use crate::device::GoldyDevice;
use crate::error::set_last_error_from_anyhow;
use std::ffi::{c_char, CStr};
use std::ptr;

/// Opaque handle to a Goldy ShaderModule.
pub struct GoldyShaderModule {
    pub(crate) inner: goldy::ShaderModule,
}

/// Create a shader module from Slang source.
///
/// Returns a pointer to the shader module, or null on failure.
///
/// # Safety
/// The device pointer must be valid.
/// The source must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn goldy_shader_create(
    device: *const GoldyDevice,
    source: *const c_char,
) -> *mut GoldyShaderModule {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }
    if source.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Source is null"));
        return ptr::null_mut();
    }
    
    let source = match CStr::from_ptr(source).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error_from_anyhow(&anyhow::anyhow!("Invalid UTF-8 in source: {}", e));
            return ptr::null_mut();
        }
    };
    
    match goldy::ShaderModule::from_slang(&(*device).inner, source) {
        Ok(shader) => Box::into_raw(Box::new(GoldyShaderModule { inner: shader })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy a shader module.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_shader_destroy(shader: *mut GoldyShaderModule) {
    if !shader.is_null() {
        drop(Box::from_raw(shader));
    }
}

/// Get the built-in vertex color 2D shader source.
///
/// Returns a pointer to a static null-terminated string.
#[no_mangle]
pub extern "C" fn goldy_shader_builtin_vertex_color_2d() -> *const c_char {
    static SOURCE: &[u8] = concat!(
        r#"struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
"#,
        "\0"
    ).as_bytes();
    SOURCE.as_ptr() as *const c_char
}

