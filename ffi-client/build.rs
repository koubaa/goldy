//! Generate C bindings from `goldy.h` and link `libgoldy_ffi` dynamically.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let header = manifest_dir.join("../cpp/include/goldy.h");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_str().expect("header path is valid UTF-8"))
        .allowlist_function("goldy_.*")
        .allowlist_type("Goldy.*")
        .allowlist_var("GOLDY_.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed on goldy.h");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    let lib_dir = env::var("DEP_GOLDY_FFI_LINK_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| find_target_output_dir(&out_dir));

    let dylib = dylib_filename();
    if !lib_dir.join(&dylib).exists() {
        panic!(
            "{dylib} not found in {}. Build it first with: cargo build -p goldy-ffi",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=goldy_ffi");

    // So `cargo run` finds libgoldy_ffi without setting DYLD_LIBRARY_PATH.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
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
