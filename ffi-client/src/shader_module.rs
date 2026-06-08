use crate::device::Device;
use crate::error::{GoldyError, non_null, Result};
use crate::sys::{self, GoldyShaderModule};
use std::ffi::CString;

/// A compiled shader module.
pub struct ShaderModule {
    ptr: *mut GoldyShaderModule,
}

impl ShaderModule {
    pub fn from_slang(device: &Device, source: &str) -> Result<Self> {
        let source = CString::new(source).map_err(|e| GoldyError::from_message(e.to_string()))?;
        let ptr = non_null(unsafe { sys::goldy_shader_create(device.as_ptr(), source.as_ptr()) })?;
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyShaderModule {
        self.ptr
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_shader_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Built-in shader sources (same shaders as native Goldy).
pub mod builtins {
    pub const VERTEX_COLOR_2D: &str = include_str!("../../shaders/vertex_color_2d.slang");
}
