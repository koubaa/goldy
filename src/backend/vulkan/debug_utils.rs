//! Khronos validation message capture via `VK_EXT_debug_utils`.
//!
//! Enabling GPU API validation turns on the Khronos layer and this messenger.
//! Messages are logged with `tracing`. They do **not** fail Goldy calls by
//! default (`vk*` still returns success). Set `GOLDY_VALIDATION_FATAL=1` to
//! treat ERROR-severity messages as `Err` on subsequent backend `Result` calls
//! and panic on backend drop (so `cargo test` fails).

use anyhow::Result;
use ash::vk;
use std::ffi::{c_void, CStr};
use std::sync::{Arc, Mutex};

const MAX_RECORDED_ERRORS: usize = 32;

pub(super) struct ValidationSink {
    errors: Mutex<Vec<String>>,
    fatal: bool,
}

impl ValidationSink {
    pub(super) fn new(fatal: bool) -> Self {
        Self {
            errors: Mutex::new(Vec::new()),
            fatal,
        }
    }

    fn push_error(&self, message: String) {
        let mut errors = self.errors.lock().unwrap();
        if errors.len() >= MAX_RECORDED_ERRORS {
            return;
        }
        errors.push(message);
    }

    fn take_errors(&self) -> Vec<String> {
        std::mem::take(&mut *self.errors.lock().unwrap())
    }
}

pub(super) fn messenger_create_info(user_data: *mut c_void) -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
        .user_data(user_data)
}

pub(super) fn sink_user_data(sink: &Arc<ValidationSink>) -> *mut c_void {
    Arc::as_ptr(sink) as *mut c_void
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut c_void,
) -> vk::Bool32 {
    if callback_data.is_null() {
        return vk::FALSE;
    }
    let data = unsafe { &*callback_data };
    let msg = if data.p_message.is_null() {
        "<null message>".to_string()
    } else {
        unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy().into_owned()
    };
    let name = if data.p_message_id_name.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(data.p_message_id_name) }
            .to_string_lossy()
            .into_owned()
    };
    let formatted = if name.is_empty() {
        format!("[{}] {msg}", data.message_id_number)
    } else {
        format!("[{}] {name}: {msg}", data.message_id_number)
    };

    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!(target: "goldy::validation", "{formatted}");
        if !user_data.is_null() {
            let sink = unsafe { &*(user_data as *const ValidationSink) };
            sink.push_error(formatted);
        }
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        tracing::warn!(target: "goldy::validation", "{formatted}");
    } else {
        tracing::info!(target: "goldy::validation", "{formatted}");
    }
    vk::FALSE
}

fn fail_if_sink_fatal(sink: &ValidationSink) -> Result<()> {
    if !sink.fatal {
        return Ok(());
    }
    let errors = sink.take_errors();
    if errors.is_empty() {
        return Ok(());
    }
    anyhow::bail!(
        "GPU API validation error(s) (GOLDY_VALIDATION_FATAL):\n{}",
        errors.join("\n---\n")
    )
}

pub(super) fn fail_if_validation_fatal(sink: Option<&Arc<ValidationSink>>) -> Result<()> {
    match sink {
        Some(sink) => fail_if_sink_fatal(sink),
        None => Ok(()),
    }
}

pub(super) fn combine_validation<T>(sink: Option<&Arc<ValidationSink>>, result: Result<T>) -> Result<T> {
    match fail_if_validation_fatal(sink) {
        Ok(()) => result,
        Err(val_err) => match result {
            Ok(_) => Err(val_err),
            Err(e) => Err(anyhow::anyhow!("{e:#}\n{val_err:#}")),
        },
    }
}

pub(super) fn destroy_messenger(
    debug_utils: Option<&ash::ext::debug_utils::Instance>,
    messenger: vk::DebugUtilsMessengerEXT,
) {
    let Some(debug_utils) = debug_utils else {
        return;
    };
    if messenger == vk::DebugUtilsMessengerEXT::null() {
        return;
    }
    unsafe {
        debug_utils.destroy_debug_utils_messenger(messenger, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fatal_sink_fails_after_error() {
        let sink = ValidationSink::new(true);
        sink.push_error("VUID-test: synthetic".into());
        let err = fail_if_sink_fatal(&sink).expect_err("fatal should fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("GOLDY_VALIDATION_FATAL"), "{msg}");
        assert!(msg.contains("VUID-test"), "{msg}");
    }

    #[test]
    fn non_fatal_sink_ignores_recorded_errors() {
        let sink = ValidationSink::new(false);
        sink.push_error("VUID-test: synthetic".into());
        fail_if_sink_fatal(&sink).expect("non-fatal should not fail");
    }
}
