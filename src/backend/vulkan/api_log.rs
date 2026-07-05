//! Vulkan native-API call trace via `VK_LAYER_LUNARG_api_dump`.
//!
//! Enabled by setting `GOLDY_API_LOG=<path.json>` (JSON output only for now).
//! The LunarG api_dump layer intercepts every Vulkan call and writes structured
//! JSON to the given path — no manual `vk*` hooks in Goldy.
//!
//! Must be configured before `vkCreateInstance` (called from `VulkanBackend::new`).

use ash::Entry;
use std::ffi::c_char;
use std::ffi::CStr;

const API_DUMP_LAYER: &str = "VK_LAYER_LUNARG_api_dump";

/// If `GOLDY_API_LOG` is set to a `.json` path and the api_dump layer is available,
/// configure layer env vars and return its name for `enabled_layer_names`.
pub(super) fn configure_and_layer(entry: &Entry) -> Option<*const c_char> {
    let path = std::env::var("GOLDY_API_LOG").ok().filter(|s| !s.is_empty())?;

    if !path.ends_with(".json") {
        tracing::warn!("GOLDY_API_LOG for Vulkan requires a `.json` path (got {path:?}); skipping api_dump");
        return None;
    }

    if !api_dump_layer_available(entry) {
        tracing::warn!(
            "GOLDY_API_LOG is set to {path:?}, but `{API_DUMP_LAYER}` is not available \
             (install Vulkan SDK validation layers or set VK_LAYER_PATH)"
        );
        return None;
    }

    // SAFETY: called once before the first `vkCreateInstance` in this process.
    unsafe {
        std::env::set_var("VK_LUNARG_API_DUMP_FILE", "true");
        std::env::set_var("VK_API_DUMP_FILE", "true");
        std::env::set_var("VK_LUNARG_API_DUMP_LOG_FILENAME", &path);
        std::env::set_var("VK_API_DUMP_LOG_FILENAME", &path);
        std::env::set_var("VK_LUNARG_API_DUMP_OUTPUT_FORMAT", "json");
        std::env::set_var("VK_API_DUMP_OUTPUT_FORMAT", "json");
        std::env::set_var("VK_LUNARG_API_DUMP_FLUSH", "true");
        std::env::set_var("VK_API_DUMP_FLUSH", "true");
    }

    tracing::info!("GOLDY_API_LOG enabled (Vulkan api_dump JSON) → {path:?}");
    Some(c"VK_LAYER_LUNARG_api_dump".as_ptr())
}

fn api_dump_layer_available(entry: &Entry) -> bool {
    let layers = match unsafe { entry.enumerate_instance_layer_properties() } {
        Ok(layers) => layers,
        Err(e) => {
            tracing::warn!("GOLDY_API_LOG: enumerate_instance_layer_properties failed: {e}");
            return false;
        }
    };

    layers.iter().any(|layer| {
        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        name.to_bytes() == API_DUMP_LAYER.as_bytes()
    })
}
