use crate::device::Device;
use crate::error::{non_null, Result};
use crate::shader_module::ShaderModule;
use crate::sys::{self, GoldyRenderPipeline};
use crate::types::{render_pipeline_desc_to_ffi, RenderPipelineDesc};

/// A graphics render pipeline.
pub struct RenderPipeline {
    ptr: *mut GoldyRenderPipeline,
}

impl RenderPipeline {
    pub fn new(
        device: &Device,
        vertex_shader: &ShaderModule,
        fragment_shader: &ShaderModule,
        desc: &RenderPipelineDesc,
    ) -> Result<Self> {
        let (ffi_desc, attributes) = render_pipeline_desc_to_ffi(desc);
        let ptr = non_null(unsafe {
            sys::goldy_render_pipeline_create(
                device.as_ptr(),
                vertex_shader.as_ptr(),
                fragment_shader.as_ptr(),
                &ffi_desc as *const _,
            )
        })?;
        drop(attributes);
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyRenderPipeline {
        self.ptr
    }
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_render_pipeline_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
