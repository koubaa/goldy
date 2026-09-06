//! Graphics APIs as foreign objects for [`crate::PixelExchange`].
//!
//! These adapters are **not** Goldy backends. They do not create a Goldy
//! `Instance` / `Device` / `Context`. Scheme submissions stay on the compute
//! device (including `GOLDY_BACKEND=cpu`). The only coupling is
//! [`crate::PixelSink::blit`], which runs under a process-wide lock.
//!
//! Windowed swapchains are a later verb on the same singleton. This module
//! ships [`HostPixelSink`] (always) and, with the `vulkan` feature, an offscreen
//! Vulkan image sink that copies a pixmap through `vkCmdCopyBufferToImage`.

use crate::pixel::HostPixelSink;
use crate::types::TextureFormat;
use crate::GoldyError;
use std::sync::Arc;

/// Headless pixmap sink. Equivalent to [`HostPixelSink::new`].
pub fn host_sink(width: u32, height: u32, format: TextureFormat) -> Result<Arc<HostPixelSink>, GoldyError> {
    HostPixelSink::new(width, height, format).map(Arc::new)
}

#[cfg(feature = "vulkan")]
pub mod vulkan;
