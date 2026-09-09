use crate::device::Device;
use crate::error::{non_null_expect, Result};
use crate::scheme::SchemeRenderTargetLease;
use crate::sys::{self, GoldyContext};
use crate::types::{DepthFormat, TextureFormat};

/// Submission context for retained [`crate::Scheme`] instances.
pub struct Context {
    ptr: *mut GoldyContext,
}

impl Context {
    pub fn new(device: &Device) -> Result<Self> {
        let ptr = non_null_expect(unsafe { sys::goldy_context_create(device.as_ptr()) });
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyContext {
        self.ptr
    }

    /// Mint a render-target lease from this context (the lessor).
    pub fn lease_render_target(
        &self,
        width: u32,
        height: u32,
        format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<SchemeRenderTargetLease> {
        let (has_depth, depth) = match depth_format {
            Some(d) => (true, d),
            None => (false, DepthFormat::Depth24Plus),
        };
        let ptr = non_null_expect(unsafe {
            sys::goldy_context_lease_render_target(self.as_ptr(), width, height, format.into(), has_depth, depth.into())
        });
        Ok(SchemeRenderTargetLease::from_ptr(ptr))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_context_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
