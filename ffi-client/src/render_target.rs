use crate::device::Device;
use crate::error::{non_null, Result};
use crate::sys::{self, GoldyRenderTarget};
use crate::types::TextureFormat;

/// An offscreen color render target.
pub struct RenderTarget {
    ptr: *mut GoldyRenderTarget,
}

impl RenderTarget {
    pub fn new(device: &Device, width: u32, height: u32, format: TextureFormat) -> Result<Self> {
        let ptr = non_null(unsafe { sys::goldy_render_target_create(device.as_ptr(), width, height, format.into()) })?;
        Ok(Self { ptr })
    }

    pub fn width(&self) -> u32 {
        unsafe { sys::goldy_render_target_width(self.ptr) }
    }

    pub fn height(&self) -> u32 {
        unsafe { sys::goldy_render_target_height(self.ptr) }
    }

    pub fn format(&self) -> TextureFormat {
        unsafe { sys::goldy_render_target_format(self.ptr) }.into()
    }

    pub fn read_to_cpu(&self) -> Result<Vec<u8>> {
        let size = unsafe { sys::goldy_render_target_buffer_size(self.ptr) };
        let mut pixels = vec![0u8; size];
        crate::error::check(unsafe {
            sys::goldy_render_target_read_to_buffer(self.ptr, pixels.as_mut_ptr(), pixels.len())
        })?;
        Ok(pixels)
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyRenderTarget {
        self.ptr
    }
}

impl Drop for RenderTarget {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_render_target_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
