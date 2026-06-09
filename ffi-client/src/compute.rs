use crate::device::Device;
use crate::error::{non_null, Result};
use crate::shader_module::ShaderModule;
use crate::sys::{self, GoldyComputePipeline};

/// A compute shader pipeline.
pub struct ComputePipeline {
    ptr: *mut GoldyComputePipeline,
}

impl ComputePipeline {
    pub fn new(device: &Device, shader: &ShaderModule) -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_compute_pipeline_create(device.as_ptr(), shader.as_ptr()) })?;
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyComputePipeline {
        self.ptr
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_compute_pipeline_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
