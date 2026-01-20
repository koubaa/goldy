//! Common utilities for Goldy integration tests.

pub mod image;

use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize the test environment.
///
/// This sets up `GOLDY_SLANG_PATH` to point to the vendored Slang libraries
/// in the repository, making tests runnable without external setup.
pub fn init_test_env() {
    INIT.call_once(|| {
        // Only set if not already set (allows override)
        if std::env::var("GOLDY_SLANG_PATH").is_err() {
            // Find the slang libraries relative to the cargo manifest directory
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            let platform = if cfg!(target_os = "windows") {
                if cfg!(target_arch = "x86_64") {
                    "windows-x86_64"
                } else {
                    "windows-aarch64"
                }
            } else if cfg!(target_os = "macos") {
                if cfg!(target_arch = "aarch64") {
                    "macos-aarch64"
                } else {
                    "macos-x86_64"
                }
            } else if cfg!(target_os = "linux") {
                if cfg!(target_arch = "aarch64") {
                    "linux-aarch64"
                } else {
                    "linux-x86_64"
                }
            } else {
                panic!("Unsupported platform for tests");
            };

            let lib_name = if cfg!(target_os = "windows") {
                "slang-compiler.dll"
            } else if cfg!(target_os = "macos") {
                "libslang-compiler.dylib"
            } else {
                "libslang-compiler.so"
            };

            let slang_path = format!("{}/slang/bin/{}/{}", manifest_dir, platform, lib_name);

            if std::path::Path::new(&slang_path).exists() {
                std::env::set_var("GOLDY_SLANG_PATH", &slang_path);
                eprintln!("Test setup: GOLDY_SLANG_PATH={}", slang_path);
            } else {
                eprintln!(
                    "Warning: Slang library not found at {}. Run slang/download.sh first.",
                    slang_path
                );
            }
        }
    });
}
