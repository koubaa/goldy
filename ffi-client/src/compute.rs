use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{check, non_null, Result};
use crate::shader_module::ShaderModule;
use crate::sys::{self, GoldyComputeEncoder, GoldyComputePipeline};

/// A compute shader pipeline.
pub struct ComputePipeline {
    ptr: *mut GoldyComputePipeline,
}

impl ComputePipeline {
    pub fn new(device: &Device, shader: &ShaderModule) -> Result<Self> {
        let ptr = non_null(unsafe {
            sys::goldy_compute_pipeline_create(device.as_ptr(), shader.as_ptr())
        })?;
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

/// Records and executes compute dispatches.
pub struct ComputeEncoder {
    ptr: *mut GoldyComputeEncoder,
}

impl ComputeEncoder {
    pub fn new() -> Self {
        let ptr = unsafe { sys::goldy_compute_encoder_create() };
        Self { ptr }
    }

    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        unsafe {
            sys::goldy_compute_encoder_set_pipeline(self.ptr, pipeline.as_ptr());
        }
    }

    pub fn bind_resources(&mut self, buffers: &[&Buffer]) {
        let ptrs: Vec<_> = buffers.iter().map(|b| b.as_ptr()).collect();
        unsafe {
            sys::goldy_compute_encoder_bind_resources(
                self.ptr,
                ptrs.as_ptr(),
                ptrs.len() as u32,
            );
        }
    }

    pub fn dispatch(&mut self, workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) {
        unsafe {
            sys::goldy_compute_encoder_dispatch(self.ptr, workgroups_x, workgroups_y, workgroups_z);
        }
    }

    pub fn execute(&self, device: &Device) -> Result<()> {
        check(unsafe { sys::goldy_compute_encoder_execute(self.ptr, device.as_ptr()) })
    }
}

impl Drop for ComputeEncoder {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_compute_encoder_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
