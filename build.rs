//! Build script for Goldy.
//!
//! Downloads Slang compiler binaries at compile time if not already present.

use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

const SLANG_VERSION: &str = "2025.24.3";

fn main() {
    println!("cargo:rerun-if-env-changed=GOLDY_SLANG_PATH");
    println!("cargo:rerun-if-env-changed=RAG_SLANG_PATH");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    // Skip Slang download for WASM targets
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("wasm") {
        return;
    }

    // Check if user already has Slang available
    if env::var("GOLDY_SLANG_PATH").is_ok() || env::var("RAG_SLANG_PATH").is_ok() {
        println!("cargo:warning=Using user-provided Slang path");
        return;
    }

    // Check Vulkan SDK on Windows
    #[cfg(target_os = "windows")]
    if env::var("VULKAN_SDK").is_ok() {
        let sdk = env::var("VULKAN_SDK").unwrap();
        let slang_path = PathBuf::from(&sdk).join("Bin").join("slang.dll");
        if slang_path.exists() {
            println!("cargo:warning=Using Slang from Vulkan SDK");
            return;
        }
    }

    // Download Slang binaries
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let slang_dir = out_dir.join("slang");

    let (platform_dir, lib_name) = get_platform_info();
    let lib_path = slang_dir.join(&platform_dir).join(&lib_name);

    // Check if already downloaded
    if lib_path.exists() {
        println!(
            "cargo:rustc-env=GOLDY_SLANG_DIR={}",
            slang_dir.join(&platform_dir).display()
        );
        return;
    }

    // Download
    println!(
        "cargo:warning=Downloading Slang v{} for {}...",
        SLANG_VERSION, platform_dir
    );

    if let Err(e) = download_slang(&slang_dir, &platform_dir, &lib_name) {
        println!("cargo:warning=Failed to download Slang: {}", e);
        println!("cargo:warning=Slang compiler will need to be provided at runtime.");
        println!("cargo:warning=Options: Set GOLDY_SLANG_PATH, install Vulkan SDK, or run slang/download.sh");
        return;
    }

    println!(
        "cargo:rustc-env=GOLDY_SLANG_DIR={}",
        slang_dir.join(&platform_dir).display()
    );
}

fn get_platform_info() -> (&'static str, &'static str) {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        ("windows-x86_64", "slang.dll")
    }

    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        ("windows-aarch64", "slang.dll")
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        ("linux-x86_64", "libslang.so")
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        ("linux-aarch64", "libslang.so")
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        ("macos-x86_64", "libslang.dylib")
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        ("macos-aarch64", "libslang.dylib")
    }

    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    compile_error!("Unsupported platform for Slang")
}

fn download_slang(slang_dir: &PathBuf, platform_dir: &str, lib_name: &str) -> io::Result<()> {
    let target_dir = slang_dir.join(platform_dir);
    fs::create_dir_all(&target_dir)?;

    let zip_name = format!("slang-{}-{}.zip", SLANG_VERSION, platform_dir);
    let url = format!(
        "https://github.com/shader-slang/slang/releases/download/v{}/{}",
        SLANG_VERSION, zip_name
    );

    let zip_path = slang_dir.join(&zip_name);

    // Download using curl (available on Windows, Linux, macOS)
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(&zip_path)
        .arg(&url)
        .status()?;

    if !status.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "curl download failed"));
    }

    // Extract using platform tools
    #[cfg(target_os = "windows")]
    {
        // Use PowerShell's Expand-Archive on Windows
        let status = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                    zip_path.display(),
                    slang_dir.display()
                ),
            ])
            .status()?;

        if !status.success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "PowerShell extract failed",
            ));
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let status = std::process::Command::new("unzip")
            .args(["-o", "-q"])
            .arg(&zip_path)
            .arg("-d")
            .arg(slang_dir)
            .status()?;

        if !status.success() {
            return Err(io::Error::new(io::ErrorKind::Other, "unzip failed"));
        }
    }

    // Find and copy the library to our target location
    // Slang releases have varying directory structures, so we search for the file
    find_and_copy_library(slang_dir, &target_dir, lib_name)?;

    // Cleanup zip
    let _ = fs::remove_file(&zip_path);

    Ok(())
}

fn find_and_copy_library(
    search_dir: &PathBuf,
    target_dir: &PathBuf,
    lib_name: &str,
) -> io::Result<()> {
    // Search recursively for the library file
    fn find_file(dir: &PathBuf, name: &str) -> Option<PathBuf> {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() {
                    if let Some(file_name) = path.file_name() {
                        // Match slang.dll, libslang.so, libslang.dylib
                        let file_name = file_name.to_string_lossy();
                        if file_name == name
                            || file_name.starts_with("slang") && file_name.contains('.')
                        {
                            return Some(path);
                        }
                    }
                } else if path.is_dir() {
                    if let Some(found) = find_file(&path, name) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }

    if let Some(lib_path) = find_file(search_dir, lib_name) {
        let dest = target_dir.join(lib_name);
        fs::copy(&lib_path, &dest)?;

        // Also copy any companion files (slang-glslang.dll, etc.) that might be needed
        if let Some(lib_dir) = lib_path.parent() {
            if let Ok(entries) = fs::read_dir(lib_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension() {
                            let ext = ext.to_string_lossy();
                            if ext == "dll" || ext == "so" || ext == "dylib" {
                                if let Some(name) = path.file_name() {
                                    let dest = target_dir.join(name);
                                    let _ = fs::copy(&path, &dest);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Could not find {} in downloaded archive", lib_name),
        ))
    }
}
