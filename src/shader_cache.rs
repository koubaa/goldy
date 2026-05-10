//! Persistent Slang compilation cache (`~/.cache/goldy/shader_cache.bin.zst`).
//!
//! Invalidated when [`GOLDY_CACHE_VERSION`] (from [`build.rs`](../../build.rs)) changes.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;

use crate::slang::{ffi::SlangStage, CompiledShaderWithReflection, OwnedLayoutCheck, ShaderTarget};
use crate::types::OptimizationLevel;

pub(crate) const GOLDY_SHADER_CACHE_MAGIC: &[u8; 8] = b"GZ_SHBIN";

/// Build-time fingerprint: package version + git short hash + bundled Slang version.
pub(crate) const GOLDY_CACHE_VERSION: &str = env!("GOLDY_CACHE_VERSION");

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

const FNV_OFFSET: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

#[inline]
fn fnv_mix(mut h: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[inline]
fn hash_string(h: u64, s: &str) -> u64 {
    fnv_mix(h, s.as_bytes())
}

fn stage_tag(s: SlangStage) -> u8 {
    match s {
        SlangStage::None => 0,
        SlangStage::Vertex => 1,
        SlangStage::Hull => 2,
        SlangStage::Domain => 3,
        SlangStage::Geometry => 4,
        SlangStage::Fragment => 5,
        SlangStage::Compute => 6,
        SlangStage::RayGeneration => 7,
        SlangStage::Intersection => 8,
        SlangStage::AnyHit => 9,
        SlangStage::ClosestHit => 10,
        SlangStage::Miss => 11,
        SlangStage::Callable => 12,
        SlangStage::Mesh => 13,
        SlangStage::Amplification => 14,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_cache_key(
    source: &str,
    target: ShaderTarget,
    entry_points: &[(&str, SlangStage)],
    defines: &[(&str, &str)],
    layout_checks: &[OwnedLayoutCheck],
    optimization_level: OptimizationLevel,
) -> u64 {
    let mut h = FNV_OFFSET;
    h = hash_string(h, source);
    h = fnv_mix(h, &[target as u8]);
    h = fnv_mix(h, &(entry_points.len() as u64).to_le_bytes());
    for &(name, st) in entry_points {
        h = hash_string(h, name);
        h = fnv_mix(h, &[stage_tag(st)]);
    }
    let mut defs: Vec<(&str, &str)> = defines.to_vec();
    defs.sort_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.cmp(b.1)));
    h = fnv_mix(h, &(defs.len() as u64).to_le_bytes());
    for &(k, v) in &defs {
        h = hash_string(h, k);
        h = hash_string(h, v);
    }
    h = fnv_mix(h, &(layout_checks.len() as u64).to_le_bytes());
    for chk in layout_checks {
        h = hash_string(h, &chk.type_name);
        h = fnv_mix(h, &chk.rust_size.to_le_bytes());
        h = fnv_mix(h, &(chk.rust_fields.len() as u64).to_le_bytes());
        for (n, off, sz) in &chk.rust_fields {
            h = hash_string(h, n);
            h = fnv_mix(h, &off.to_le_bytes());
            h = fnv_mix(h, &sz.to_le_bytes());
        }
    }
    h = fnv_mix(h, &[optimization_level as u8]);
    h
}

fn encode_cached(value: &CompiledShaderWithReflection) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(value, BINCODE_CFG).map_err(Into::into)
}

fn decode_cached(bytes: &[u8]) -> Result<CompiledShaderWithReflection> {
    Ok(bincode::serde::decode_from_slice(bytes, BINCODE_CFG)?.0)
}

#[inline]
fn empty_shader_cache_shell(disk_path: Option<PathBuf>) -> ShaderBytecodeDiskCache {
    ShaderBytecodeDiskCache {
        map: HashMap::new(),
        dirty: false,
        disk_path,
        version_ok_on_disk: false,
    }
}

pub(crate) struct ShaderBytecodeDiskCache {
    map: HashMap<u64, Vec<u8>>,
    dirty: bool,
    disk_path: Option<PathBuf>,
    /// `true` iff the disk file envelope matches [`GOLDY_CACHE_VERSION`].
    version_ok_on_disk: bool,
}

impl ShaderBytecodeDiskCache {
    #[must_use]
    pub(crate) fn new_load_or_empty() -> Self {
        let Some(cache_root) = dirs::cache_dir() else {
            return empty_shader_cache_shell(None);
        };

        let path = cache_root.join("goldy").join("shader_cache.bin.zst");
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                return empty_shader_cache_shell(Some(path));
            }
        };

        let decompressed =
            match zstd::decode_all(std::io::Cursor::new(&bytes)).map_err(|e| anyhow::anyhow!(e)) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(?e, "failed ZSTD-decompress shader cache; ignoring");
                    return empty_shader_cache_shell(Some(path));
                }
            };

        let Some(body) = decompressed.strip_prefix(GOLDY_SHADER_CACHE_MAGIC) else {
            tracing::warn!("shader cache magic mismatch — ignoring cached file");
            return empty_shader_cache_shell(Some(path));
        };

        let Some(nl_pos) = body.iter().position(|&b| b == b'\n') else {
            tracing::warn!("shader cache malformed (no version nl) — ignoring");
            return empty_shader_cache_shell(Some(path));
        };

        let ver = match std::str::from_utf8(&body[..nl_pos]) {
            Ok(v) => v.to_string(),
            Err(_) => {
                tracing::warn!("shader cache version not utf-8 — ignoring");
                return empty_shader_cache_shell(Some(path));
            }
        };

        let rest = &body[nl_pos + 1..];
        if ver != GOLDY_CACHE_VERSION {
            tracing::debug!(
                disk = %ver,
                build = GOLDY_CACHE_VERSION,
                "shader cache version mismatch — starting cold"
            );
            return empty_shader_cache_shell(Some(path));
        }

        match parse_flat_map(rest) {
            Ok(map) => Self {
                map,
                dirty: false,
                disk_path: Some(path),
                version_ok_on_disk: true,
            },
            Err(e) => {
                tracing::warn!(?e, "failed to parse shader cache body — ignoring");
                empty_shader_cache_shell(Some(path))
            }
        }
    }

    #[must_use]
    pub(crate) fn get(&mut self, key: u64) -> Option<Result<CompiledShaderWithReflection>> {
        let blob = self.map.get(&key)?.clone();
        Some(decode_cached(&blob))
    }

    pub(crate) fn insert(&mut self, key: u64, value: &CompiledShaderWithReflection) -> Result<()> {
        let blob = encode_cached(value)?;
        match self.map.get(&key) {
            Some(prev) if *prev == blob => return Ok(()),
            _ => {
                self.map.insert(key, blob);
                self.dirty = true;
            }
        }
        Ok(())
    }

    /// Write through to disk when dirty. Safe to skip if unavailable.
    pub(crate) fn flush_to_disk_best_effort(&mut self) {
        if !self.dirty {
            return;
        }

        let Some(path) = &self.disk_path else {
            tracing::trace!("shader cache dirty but no dirs::cache_dir() — skipping write");
            return;
        };

        let mut uncompressed = Vec::new();
        let _ = uncompressed.write_all(GOLDY_SHADER_CACHE_MAGIC);
        uncompressed
            .write_all(GOLDY_CACHE_VERSION.as_bytes())
            .unwrap();
        uncompressed.write_all(b"\n").unwrap();
        if flatten_map_to_disk(&mut uncompressed, &self.map).is_ok() {
            match zstd::encode_all(uncompressed.as_slice(), 10) {
                Ok(z) => {
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    if std::fs::write(path, &z).is_ok() {
                        self.dirty = false;
                        self.version_ok_on_disk = true;
                    }
                }
                Err(e) => tracing::warn!(?e, "zstd shader cache encode failed"),
            }
        }
    }

    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn should_skip_save_on_drop(&self) -> bool {
        !self.dirty
    }

    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn version_ok_on_disk(&self) -> bool {
        self.version_ok_on_disk
    }
}

impl Drop for ShaderBytecodeDiskCache {
    fn drop(&mut self) {
        self.flush_to_disk_best_effort();
    }
}

fn parse_flat_map(body: &[u8]) -> Result<HashMap<u64, Vec<u8>>> {
    let mut cursor = 0usize;
    if cursor + 8 > body.len() {
        anyhow::bail!("shader cache truncate (count)");
    }
    let n = usize::try_from(u64::from_le_bytes(body[cursor..cursor + 8].try_into()?))?;
    cursor += 8;
    let mut out = HashMap::with_capacity(n);
    for _ in 0..n {
        if cursor + 8 + 4 > body.len() {
            anyhow::bail!("shader cache truncate (entry hdr)");
        }
        let key = u64::from_le_bytes(body[cursor..cursor + 8].try_into()?);
        cursor += 8;
        let len = usize::try_from(u32::from_le_bytes(body[cursor..cursor + 4].try_into()?))?;
        cursor += 4;
        if cursor + len > body.len() {
            anyhow::bail!("shader cache truncate (blob)");
        }
        out.insert(key, body[cursor..cursor + len].to_vec());
        cursor += len;
    }
    Ok(out)
}

fn flatten_map_to_disk(dest: &mut Vec<u8>, map: &HashMap<u64, Vec<u8>>) -> Result<()> {
    let n_u64 = u64::try_from(map.len())?;
    dest.write_all(&n_u64.to_le_bytes())?;
    let mut keys: Vec<u64> = map.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        let blob = map.get(&k).expect("key from map iteration");
        dest.write_all(&k.to_le_bytes())?;
        let len_u32 = u32::try_from(blob.len())
            .map_err(|_| anyhow::anyhow!("shader cache blob too large"))?;
        dest.write_all(&len_u32.to_le_bytes())?;
        dest.write_all(blob)?;
    }
    Ok(())
}
