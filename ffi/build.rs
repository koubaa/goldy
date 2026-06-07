//! Build script for goldy-ffi that generates the C header using cbindgen
//! and copies Slang libraries to the output directory.

use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    // Only regenerate header when source files change
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=../slang/manifest.json");
    println!("cargo:rerun-if-changed=../slang/bin/");

    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Generate C header
    generate_c_header(&crate_dir);

    // Copy Slang libraries to output directory
    copy_slang_libraries(&crate_dir, &out_dir);
}

fn generate_c_header(crate_dir: &Path) {
    let output_dir = crate_dir.join("..").join("cpp").join("include");

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir).ok();

    let output_file = output_dir.join("goldy.h");

    // Run cbindgen
    let config = cbindgen::Config::from_file("cbindgen.toml").expect("Failed to read cbindgen.toml");

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Failed to generate C header")
        .write_to_file(&output_file);
}

fn copy_slang_libraries(crate_dir: &Path, out_dir: &Path) {
    let manifest_path = crate_dir.join("..").join("slang").join("manifest.json");

    // Read manifest
    let manifest_content = match fs::read_to_string(&manifest_path) {
        Ok(content) => content,
        Err(e) => {
            println!("cargo:warning=Could not read slang/manifest.json: {}", e);
            return;
        }
    };

    let platform_dir = get_platform_dir();
    let slang_bin_dir = crate_dir.join("..").join("slang").join("bin").join(platform_dir);

    if !slang_bin_dir.exists() {
        println!("cargo:warning=Slang binaries not found at {}", slang_bin_dir.display());
        println!("cargo:warning=Run slang/download.sh to download Slang binaries");
        return;
    }

    // Parse the files list for this platform from manifest
    let files = match extract_platform_files(&manifest_content, platform_dir) {
        Some(files) => files,
        None => {
            println!("cargo:warning=Platform {} not found in manifest", platform_dir);
            return;
        }
    };

    // Determine output directory (where goldy_ffi.dll will be)
    // OUT_DIR is something like target/release/build/goldy-ffi-xxx/out
    // We want to copy to target/release/ alongside the .dll
    let target_dir = find_target_output_dir(out_dir);

    for file_name in &files {
        let src = slang_bin_dir.join(file_name);
        if src.exists() {
            let dest = target_dir.join(file_name);
            if let Err(e) = fs::copy(&src, &dest) {
                println!("cargo:warning=Failed to copy {}: {}", file_name, e);
            }
        } else {
            println!("cargo:warning=Slang file not found: {}", src.display());
        }
    }

    // Also emit a path that binding build scripts can use
    println!("cargo:rustc-env=GOLDY_SLANG_BIN_DIR={}", slang_bin_dir.display());
}

fn get_platform_dir() -> &'static str {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "windows-x86_64"
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "linux-x86_64"
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "linux-aarch64"
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "macos-x86_64"
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "macos-aarch64"
    }

    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    {
        "unsupported"
    }
}

fn extract_platform_files(json: &str, platform: &str) -> Option<Vec<String>> {
    let pattern = format!("\"{}\"", platform);
    let start = json.find(&pattern)?;
    let section = &json[start..];

    // Find the files array
    let files_start = section.find("\"files\"")?;
    let after_files = &section[files_start..];
    let array_start = after_files.find('[')?;
    let array_end = after_files.find(']')?;
    let array_content = &after_files[array_start + 1..array_end];

    let files: Vec<String> = array_content
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.starts_with('"') && s.ends_with('"') {
                Some(s[1..s.len() - 1].to_string())
            } else {
                None
            }
        })
        .collect();

    if files.is_empty() {
        None
    } else {
        Some(files)
    }
}

fn find_target_output_dir(out_dir: &Path) -> PathBuf {
    // OUT_DIR is typically: target/{profile}/build/{crate}-{hash}/out
    // We want: target/{profile}/
    let mut current = out_dir;

    // Walk up to find the target/{profile} directory
    for _ in 0..5 {
        if let Some(parent) = current.parent() {
            // Check if this looks like target/{profile} by seeing if it contains deps/
            let deps_dir = parent.join("deps");
            if deps_dir.exists() {
                return parent.to_path_buf();
            }
            current = parent;
        }
    }

    // Fallback: use OUT_DIR itself
    out_dir.to_path_buf()
}
