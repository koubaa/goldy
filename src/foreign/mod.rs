//! Graphics APIs as foreign objects for [`crate::PixelExchange`].
//!
//! These adapters are **not** Goldy backends. They do not create a Goldy
//! `Instance` / `Device` / `Context`. Scheme submissions stay on the compute
//! device (including `GOLDY_BACKEND=cpu`). The only coupling is
//! [`crate::PixelSink::blit`], which runs under a process-wide lock.
//!
//! Windowed present is a verb on the same singleton: [`windowed`] attaches a
//! platform swapchain (Metal `CAMetalLayer` today) and [`PixelSink::blit`]
//! copies + presents. This module ships [`HostPixelSink`] (always) and, behind
//! the matching feature, Vulkan / DX12 / Metal foreign surfaces.

use crate::pixel::{HostPixelSink, PixelSink};
use crate::types::TextureFormat;
use crate::GoldyError;
use std::sync::Arc;

/// Headless pixmap sink. Equivalent to [`HostPixelSink::new`].
pub fn host_sink(width: u32, height: u32, format: TextureFormat) -> Result<Arc<HostPixelSink>, GoldyError> {
    HostPixelSink::new(width, height, format).map(Arc::new)
}

/// Windowed pixmap destination: [`PixelSink::blit`] copies and presents.
#[cfg(feature = "graphics")]
pub trait WindowSink: PixelSink + Send + Sync {
    fn resize(&self, width: u32, height: u32) -> Result<(), GoldyError>;
    fn set_present_mode(&self, mode: crate::types::PresentMode) -> Result<(), GoldyError>;
}

/// Attach a foreign windowed surface to `window` and return it as a [`WindowSink`].
///
/// Compute stays on the Goldy device (`GOLDY_BACKEND=cpu` included). Present
/// uses the process-wide graphics singleton — not a second Goldy backend.
#[cfg(feature = "graphics")]
pub fn windowed(
    window: &dyn raw_window_handle::HasWindowHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
    present_mode: crate::types::PresentMode,
) -> Result<Arc<dyn WindowSink>, GoldyError> {
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    {
        let adapter = metal::try_adapter().ok_or_else(|| {
            GoldyError::Backend(anyhow::anyhow!(
                "foreign Metal adapter unavailable for windowed PixelSink"
            ))
        })?;
        Ok(Arc::new(adapter.windowed(
            window,
            width,
            height,
            format,
            present_mode,
        )?))
    }
    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    {
        let _ = (window, width, height, format, present_mode);
        Err(GoldyError::Backend(anyhow::anyhow!(
            "foreign windowed PixelSink is implemented for Metal (macOS/iOS); Vulkan/DX12 WSI is not wired yet"
        )))
    }
}

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(all(feature = "dx12", target_os = "windows"))]
pub mod dx12;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal;
