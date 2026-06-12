//! Build script for Goldy.
//!
//! Embeds Slang compiler binaries into the library using include_bytes!.
//! Downloads Slang if vendored binaries are not present.
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

/// `GOLDY_CACHE_VERSION` for on-disk shader bytecode cache invalidation.
///
/// Intentionally omits the git hash: per-entry cache keys are already content-addressed
/// (shader source + defines + target + optimization level), so a real shader change produces
/// a per-entry miss without wiping the whole file. Only a Slang upgrade or a package version
/// bump should invalidate all entries.
fn emit_goldy_cache_version(slang_semver_label: Option<&str>) {
    let pkg = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());

    let sl = slang_semver_label.unwrap_or("noslang");
    println!("cargo:rustc-env=GOLDY_CACHE_VERSION=v{pkg}-slang{sl}");
}

/// FNV-1a 64-bit hash (mirrors the implementation in `shader_cache.rs`).
///
/// Kept as a free function in build.rs so it has no external dependencies.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Hash all `*.slang` files under `shaders/goldy_exp/` (sorted by path for stability)
/// and emit `GOLDY_EXP_HASH` as a 16-character lowercase hex string.
///
/// Algorithm must stay in sync with [`hash_goldy_exp_sources`] in `shader_cache.rs`
/// (verified by `goldy_exp_hash_matches_built_constant`).
///
/// Each file path is also registered with `cargo:rerun-if-changed` so that any
/// edit to a goldy_exp source triggers a rebuild and a new hash.
fn emit_goldy_exp_hash(manifest_dir: &Path) {
    let lib_dir = manifest_dir.join("shaders").join("goldy_exp");

    // Collect and sort paths for a deterministic hash order.
    let mut files: Vec<PathBuf> = match fs::read_dir(&lib_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("slang"))
            .collect(),
        Err(_) => Vec::new(),
    };
    files.sort();

    let mut combined: u64 = fnv1a_64(b"goldy_exp_v1"); // schema seed
    for path in &files {
        // Register with cargo so any edit triggers a rebuild.
        println!("cargo:rerun-if-changed={}", path.display());

        let contents = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                println!("cargo:warning=goldy_exp hash: cannot read {}: {e}", path.display());
                continue;
            }
        };
        // Mix in the filename (not the full path — only the filename is stable
        // across machines / checkout locations) and then the file contents.
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        combined = fnv1a_64_mix(combined, name.as_bytes());
        combined = fnv1a_64_mix(combined, &contents);
    }

    println!("cargo:rustc-env=GOLDY_EXP_HASH={combined:016x}");
}

/// Mix additional bytes into an existing FNV-1a state (chain hash).
fn fnv1a_64_mix(mut h: u64, bytes: &[u8]) -> u64 {
    const PRIME: u64 = 1099511628211;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

fn main() {
    println!("cargo:rerun-if-env-changed=GOLDY_SLANG_PATH");
    println!("cargo:rerun-if-changed=slang/manifest.json");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Hash goldy_exp shader library sources so that edits to access.slang etc.
    // invalidate per-entry shader cache keys (see compile_cache_key in shader_cache.rs).
    emit_goldy_exp_hash(&manifest_dir);

    // Load manifest
    let manifest_path = manifest_dir.join("slang").join("manifest.json");

    let manifest = match load_manifest(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            println!("cargo:warning=Failed to load slang/manifest.json: {}", e);
            generate_empty_embedded_module();
            emit_goldy_cache_version(None);
            return;
        }
    };

    let platform_dir = get_platform_dir();
    let platform_info = match manifest.platforms.get(platform_dir) {
        Some(info) => info,
        None => {
            println!("cargo:warning=Platform {} not in manifest", platform_dir);
            generate_empty_embedded_module();
            emit_goldy_cache_version(Some(manifest.version.as_str()));
            return;
        }
    };

    // Check vendored binaries directory
    let vendored_dir = manifest_dir.join("slang").join("bin").join(platform_dir);

    // Mark all vendored files as dependencies for rebuild
    for file in &platform_info.files {
        let file_path = vendored_dir.join(file);
        println!("cargo:rerun-if-changed={}", file_path.display());
    }

    // If vendored binaries don't exist, try to download them
    if !vendored_dir.join(&platform_info.primary).exists() {
        println!(
            "cargo:warning=Vendored Slang binaries not found at {}",
            vendored_dir.display()
        );
        println!(
            "cargo:warning=Downloading Slang v{} for {}...",
            manifest.version, platform_dir
        );

        if let Err(e) = download_slang_to_vendored(&vendored_dir, platform_dir, &manifest.version, &platform_info.files)
        {
            println!("cargo:warning=Failed to download Slang: {}", e);
            println!("cargo:warning=Run: cd slang && ./download.sh");
            generate_empty_embedded_module();
            emit_goldy_cache_version(Some(manifest.version.as_str()));
            return;
        }
    }

    // Generate the embedded module
    generate_embedded_module(&manifest.version, &vendored_dir, platform_info);
    emit_goldy_cache_version(Some(manifest.version.as_str()));
}

/// Generate an empty embedded module (for unsupported platforms or missing binaries)
fn generate_empty_embedded_module() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let embedded_path = out_dir.join("slang_embedded.rs");

    let content = r#"// Auto-generated: Slang binaries not available for this platform.

/// Slang version (empty - not available)
pub const SLANG_VERSION: &str = "";

/// Embedded Slang library files (empty - not available)
pub const SLANG_FILES: &[(&str, &[u8])] = &[];

/// Primary library name (empty - not available)
pub const SLANG_PRIMARY: &str = "";
"#;

    fs::write(&embedded_path, content).expect("Failed to write slang_embedded.rs");
}

/// Generate the embedded module with include_bytes! for all Slang files
fn generate_embedded_module(version: &str, vendored_dir: &Path, platform_info: &PlatformInfo) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let embedded_path = out_dir.join("slang_embedded.rs");

    let mut content = String::new();
    content.push_str("// Auto-generated: Embedded Slang compiler binaries.\n");
    content.push_str("// Do not edit manually - regenerated by build.rs\n\n");

    // Version constant
    content.push_str(&format!(
        "/// Slang version embedded in this build\n\
         pub const SLANG_VERSION: &str = \"{}\";\n\n",
        version
    ));

    // Primary library name
    content.push_str(&format!(
        "/// Primary Slang library filename\n\
         pub const SLANG_PRIMARY: &str = \"{}\";\n\n",
        platform_info.primary
    ));

    // Embedded files array
    content.push_str("/// Embedded Slang library files (filename, bytes)\n");
    content.push_str("pub const SLANG_FILES: &[(&str, &[u8])] = &[\n");

    for file in &platform_info.files {
        let file_path = vendored_dir.join(file);
        if file_path.exists() {
            // Use absolute path for include_bytes!
            let abs_path = file_path.canonicalize().unwrap_or_else(|_| file_path.clone());

            // On Windows, canonicalize adds \\?\ prefix, which we need to handle
            let path_str = abs_path.display().to_string();
            let path_str = path_str.strip_prefix(r"\\?\").unwrap_or(&path_str);

            // Escape backslashes for the string literal
            let escaped_path = path_str.replace('\\', "/");

            content.push_str(&format!("    (\"{}\", include_bytes!(\"{}\")),\n", file, escaped_path));
        } else {
            println!("cargo:warning=Slang file not found: {}", file_path.display());
        }
    }

    content.push_str("];\n");

    fs::write(&embedded_path, content).expect("Failed to write slang_embedded.rs");
}

fn load_manifest(path: &Path) -> io::Result<SlangManifest> {
    let content = fs::read_to_string(path)?;

    // Simple JSON parsing without serde (to avoid build dependency)
    let version = extract_json_string(&content, "version")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing version"))?;

    let mut platforms = std::collections::HashMap::new();

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

/// Download Slang binaries directly to the vendored directory
fn download_slang_to_vendored(
    vendored_dir: &Path,
    platform_dir: &str,
    version: &str,
    required_files: &[String],
) -> io::Result<()> {
    fs::create_dir_all(vendored_dir)?;

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let zip_name = format!("slang-{}-{}.zip", version, platform_dir);
    let url = format!(
        "https://github.com/shader-slang/slang/releases/download/v{}/{}",
        version, zip_name
    );

    let zip_path = out_dir.join(&zip_name);

    // Download using curl
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(&zip_path)
        .arg(&url)
        .status()?;

    if !status.success() {
        return Err(io::Error::other(format!("curl download failed for {}", url)));
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

    // Find and copy all required files to vendored directory
    let mut copied = 0;
    for file_name in required_files {
        if let Some(src_path) = find_file_recursive(&extract_dir, file_name) {
            let dest = vendored_dir.join(file_name);
            fs::copy(&src_path, &dest)?;
            copied += 1;
        } else {
            println!("cargo:warning=Slang file not found in archive: {}", file_name);
        }
    }

    if copied == 0 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No Slang libraries found in downloaded archive",
        ));
    }

    println!(
        "cargo:warning=Downloaded {}/{} Slang libraries to {}",
        copied,
        required_files.len(),
        vendored_dir.display()
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
