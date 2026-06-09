//! Locate `goldy_ffi` for runtime loading (no link-time bindgen or dylib link).

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let lib_dir = env::var("DEP_GOLDY_FFI_LINK_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| find_target_output_dir(&out_dir));

    let dylib = dylib_filename();
    if !lib_dir.join(dylib).exists() {
        panic!(
            "{dylib} not found in {}. Build it first with: cargo build -p goldy-ffi",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-env=GOLDY_FFI_LIB_DIR={}", lib_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
}

fn dylib_filename() -> &'static str {
    if cfg!(target_os = "macos") {
        "libgoldy_ffi.dylib"
    } else if cfg!(target_os = "windows") {
        "goldy_ffi.dll"
    } else {
        "libgoldy_ffi.so"
    }
}

fn find_target_output_dir(out_dir: &Path) -> PathBuf {
    let mut current = out_dir;
    for _ in 0..5 {
        if let Some(parent) = current.parent() {
            if parent.join("deps").exists() {
                return parent.to_path_buf();
            }
            current = parent;
        }
    }
    out_dir.to_path_buf()
}
