//! Build script for Goldy.
//!
//! Downloads Slang compiler binaries at compile time if not already present.
//! Reads from slang/manifest.json for version and file lists.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Slang manifest structure (subset we need)
#[derive(Debug)]
struct SlangManifest {
    version: String,
    platforms: std::collections::HashMap<String, PlatformInfo>,
}

#[derive(Debug)]
struct PlatformInfo {
    files: Vec<String>,
    primary: String,
}

fn main() {
    println!("cargo:rerun-if-env-changed=GOLDY_SLANG_PATH");
    println!("cargo:rerun-if-env-changed=RAG_SLANG_PATH");
    println!("cargo:rerun-if-changed=slang/manifest.json");

    // Skip Slang download for WASM targets
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("wasm") {
        return;
    }

    // Check if user already has Slang available via env var
    if env::var("GOLDY_SLANG_PATH").is_ok() || env::var("RAG_SLANG_PATH").is_ok() {
        println!("cargo:warning=Using user-provided Slang path");
        return;
    }

    // Load manifest
    let manifest_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("slang")
        .join("manifest.json");

    let manifest = match load_manifest(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            println!("cargo:warning=Failed to load slang/manifest.json: {}", e);
            println!("cargo:warning=Slang will need to be provided at runtime");
            return;
        }
    };

    let platform_dir = get_platform_dir();
    let platform_info = match manifest.platforms.get(platform_dir) {
        Some(info) => info,
        None => {
            println!(
                "cargo:warning=Platform {} not in manifest, Slang will need to be provided at runtime",
                platform_dir
            );
            return;
        }
    };

    // Check vendored binaries first (for development)
    let vendored_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("slang")
        .join("bin")
        .join(platform_dir);

    if vendored_dir.join(&platform_info.primary).exists() {
        println!("cargo:rustc-env=GOLDY_SLANG_DIR={}", vendored_dir.display());
        return;
    }

    // Download Slang binaries to OUT_DIR
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let slang_dir = out_dir.join("slang").join(platform_dir);

    // Check if already downloaded
    if slang_dir.join(&platform_info.primary).exists() {
        println!("cargo:rustc-env=GOLDY_SLANG_DIR={}", slang_dir.display());
        return;
    }

    // Download
    println!(
        "cargo:warning=Downloading Slang v{} for {}...",
        manifest.version, platform_dir
    );

    if let Err(e) = download_slang(
        &out_dir,
        platform_dir,
        &manifest.version,
        &platform_info.files,
    ) {
        println!("cargo:warning=Failed to download Slang: {}", e);
        println!("cargo:warning=Slang compiler will need to be provided at runtime.");
        println!("cargo:warning=Options: Set GOLDY_SLANG_PATH env var, or run slang/download.sh");
        return;
    }

    println!("cargo:rustc-env=GOLDY_SLANG_DIR={}", slang_dir.display());
}

fn load_manifest(path: &Path) -> io::Result<SlangManifest> {
    let content = fs::read_to_string(path)?;

    // Simple JSON parsing without serde (to avoid build dependency)
    // Extract version
    let version = extract_json_string(&content, "version")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing version"))?;

    let mut platforms = std::collections::HashMap::new();

    // Parse each platform
    for platform in &[
        "windows-x86_64",
        "linux-x86_64",
        "linux-aarch64",
        "macos-x86_64",
        "macos-aarch64",
    ] {
        if let Some(info) = extract_platform_info(&content, platform) {
            platforms.insert(platform.to_string(), info);
        }
    }

    Ok(SlangManifest { version, platforms })
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let after_colon = &after_key[colon + 1..];
    let quote_start = after_colon.find('"')?;
    let value_start = &after_colon[quote_start + 1..];
    let quote_end = value_start.find('"')?;
    Some(value_start[..quote_end].to_string())
}

fn extract_platform_info(json: &str, platform: &str) -> Option<PlatformInfo> {
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

    // Find primary
    let primary = extract_json_string(section, "primary")?;

    Some(PlatformInfo { files, primary })
}

fn get_platform_dir() -> &'static str {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "windows-x86_64"
    }

    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "windows-aarch64"
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
        all(target_os = "windows", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    compile_error!("Unsupported platform for Slang")
}

fn download_slang(
    out_dir: &Path,
    platform_dir: &str,
    version: &str,
    required_files: &[String],
) -> io::Result<()> {
    let target_dir = out_dir.join("slang").join(platform_dir);
    fs::create_dir_all(&target_dir)?;

    let zip_name = format!("slang-{}-{}.zip", version, platform_dir);
    let url = format!(
        "https://github.com/shader-slang/slang/releases/download/v{}/{}",
        version, zip_name
    );

    let zip_path = out_dir.join(&zip_name);

    // Download using curl (available on Windows, Linux, macOS)
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(&zip_path)
        .arg(&url)
        .status()?;

    if !status.success() {
        return Err(io::Error::other(format!(
            "curl download failed for {}",
            url
        )));
    }

    // Create temp extraction directory
    let extract_dir = out_dir.join("slang_extract");
    let _ = fs::remove_dir_all(&extract_dir);
    fs::create_dir_all(&extract_dir)?;

    // Extract using platform tools
    #[cfg(target_os = "windows")]
    {
        let status = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                    zip_path.display(),
                    extract_dir.display()
                ),
            ])
            .status()?;

        if !status.success() {
            return Err(io::Error::other("PowerShell extract failed"));
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let status = std::process::Command::new("unzip")
            .args(["-o", "-q"])
            .arg(&zip_path)
            .arg("-d")
            .arg(&extract_dir)
            .status()?;

        if !status.success() {
            return Err(io::Error::other("unzip failed"));
        }
    }

    // Find and copy all required files
    let mut copied = 0;
    for file_name in required_files {
        if let Some(src_path) = find_file_recursive(&extract_dir, file_name) {
            let dest = target_dir.join(file_name);
            fs::copy(&src_path, &dest)?;
            copied += 1;
        } else {
            println!(
                "cargo:warning=Slang file not found in archive: {}",
                file_name
            );
        }
    }

    if copied == 0 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No Slang libraries found in downloaded archive",
        ));
    }

    println!(
        "cargo:warning=Copied {}/{} Slang libraries to {}",
        copied,
        required_files.len(),
        target_dir.display()
    );

    // Cleanup
    let _ = fs::remove_file(&zip_path);
    let _ = fs::remove_dir_all(&extract_dir);

    Ok(())
}

fn find_file_recursive(dir: &Path, name: &str) -> Option<PathBuf> {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                if let Some(file_name) = path.file_name() {
                    if file_name.to_string_lossy() == name {
                        return Some(path);
                    }
                }
            } else if path.is_dir() {
                if let Some(found) = find_file_recursive(&path, name) {
                    return Some(found);
                }
            }
        }
    }
    None
}
