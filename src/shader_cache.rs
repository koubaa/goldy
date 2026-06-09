//! Persistent Slang compilation cache (`~/.cache/goldy/shader_cache.bin.zst`).
//!
//! Invalidated when [`GOLDY_CACHE_VERSION`] changes (defined in crate [`build.rs`](../build.rs)).

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;

use crate::slang::{ffi::SlangStage, CompiledShaderWithReflection, OwnedLayoutCheck, ShaderTarget};
use crate::types::OptimizationLevel;

/// Leading bytes of the decompressed shader cache payload.
pub const GOLDY_SHADER_CACHE_MAGIC: &[u8; 8] = b"GZ_SHBIN";

/// Bump when [`ShaderReflection::binding_element_strides`] extraction rules change.
const REFLECTION_STRIDE_SCHEMA: &str = "bind-stride-v2";

/// Build-time fingerprint: package version + git short hash + bundled Slang version.
pub const GOLDY_CACHE_VERSION: &str = env!("GOLDY_CACHE_VERSION");

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
/// Stable disk-cache key for a Slang compile.
///
/// `source` must be the exact translation-unit text Slang compiles (after
/// [`crate::slang::virtual_main::effective_slang_source_for_compile`] when `[goldy_*]` markers apply).
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
    h = hash_string(h, REFLECTION_STRIDE_SCHEMA);
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

pub struct ShaderBytecodeDiskCache {
    map: HashMap<u64, Vec<u8>>,
    dirty: bool,
    disk_path: Option<PathBuf>,
    /// `true` iff the disk file envelope matches [`GOLDY_CACHE_VERSION`].
    version_ok_on_disk: bool,
}

impl ShaderBytecodeDiskCache {
    #[must_use]
    pub fn new_at_path(path: PathBuf) -> Self {
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                return empty_shader_cache_shell(Some(path));
            }
        };

        let decompressed = match zstd::decode_all(std::io::Cursor::new(&bytes)).map_err(|e| anyhow::anyhow!(e)) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(?e, "failed ZSTD-decompress shader cache; ignoring");
                return empty_shader_cache_shell(Some(path));
            }
        };

        let magic = GOLDY_SHADER_CACHE_MAGIC.as_slice();
        let Some(body) = decompressed.strip_prefix(magic) else {
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
    pub(crate) fn new_load_or_empty() -> Self {
        let Some(cache_root) = dirs::cache_dir() else {
            return empty_shader_cache_shell(None);
        };

        Self::new_at_path(cache_root.join("goldy").join("shader_cache.bin.zst"))
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
        let _ = uncompressed.write_all(GOLDY_SHADER_CACHE_MAGIC.as_slice());
        uncompressed.write_all(GOLDY_CACHE_VERSION.as_bytes()).unwrap();
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

    /// `true` when the backing file decoded with a version matching this build ([`GOLDY_CACHE_VERSION`]).
    #[must_use]
    pub fn version_ok_on_disk(&self) -> bool {
        self.version_ok_on_disk
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
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
        let len_u32 = u32::try_from(blob.len()).map_err(|_| anyhow::anyhow!("shader cache blob too large"))?;
        dest.write_all(&len_u32.to_le_bytes())?;
        dest.write_all(blob)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slang::{
        ffi::SlangStage,
        virtual_main::{effective_slang_source_for_compile, transform_virtual_main},
        CompiledShader, CompiledShaderWithReflection, OwnedLayoutCheck, ShaderReflection, ShaderTarget,
    };
    use crate::types::OptimizationLevel;
    use tempfile::TempDir;

    type BaseCompileArgs = (
        &'static str,
        ShaderTarget,
        Vec<(&'static str, SlangStage)>,
        Vec<(&'static str, &'static str)>,
        Vec<OwnedLayoutCheck>,
        OptimizationLevel,
    );

    fn dummy_compiled() -> CompiledShaderWithReflection {
        CompiledShaderWithReflection {
            shader: CompiledShader {
                data: vec![0xDE, 0xAD, 0xBE, 0xEF],
                target: ShaderTarget::Spirv,
            },
            reflection: ShaderReflection::default(),
        }
    }

    fn base_compile_args() -> BaseCompileArgs {
        (
            "float4 main() : SV_Target0 { return 0; }",
            ShaderTarget::Spirv,
            vec![("main", SlangStage::Fragment)],
            vec![("A", "1"), ("B", "2")],
            vec![OwnedLayoutCheck {
                type_name: "Foo".to_string(),
                rust_size: 16,
                rust_fields: vec![("x".to_string(), 0, 4)],
            }],
            OptimizationLevel::default(),
        )
    }

    #[test]
    fn compile_cache_key_deterministic() {
        let (src, tgt, eps, defs, layouts, opt) = base_compile_args();
        let k1 = compile_cache_key(src, tgt, &eps, &defs, &layouts, opt);
        let k2 = compile_cache_key(src, tgt, &eps, &defs, &layouts, opt);
        assert_eq!(k1, k2);
    }

    #[test]
    fn compile_cache_key_define_order_independent() {
        let (src, tgt, eps, _defs, layouts, opt) = base_compile_args();
        let d1 = vec![("a", "x"), ("b", "y")];
        let d2 = vec![("b", "y"), ("a", "x")];
        assert_eq!(
            compile_cache_key(src, tgt, &eps, &d1, &layouts, opt),
            compile_cache_key(src, tgt, &eps, &d2, &layouts, opt),
        );
    }

    #[test]
    fn compile_cache_key_sensitive_to_source() {
        let (src, tgt, eps, defs, layouts, opt) = base_compile_args();
        let base_key = compile_cache_key(src, tgt, &eps, &defs, &layouts, opt);
        let alt = "float4 main() : SV_Target0 { return 1; }";
        assert_ne!(base_key, compile_cache_key(alt, tgt, &eps, &defs, &layouts, opt));
    }

    #[test]
    fn compile_cache_key_sensitive_to_target() {
        let (src, tgt, eps, defs, layouts, opt) = base_compile_args();
        let base_key = compile_cache_key(src, tgt, &eps, &defs, &layouts, opt);
        assert_ne!(
            base_key,
            compile_cache_key(src, ShaderTarget::Dxil, &eps, &defs, &layouts, opt),
        );
    }

    #[test]
    fn compile_cache_key_sensitive_to_entry_point() {
        let (src, tgt, eps, defs, layouts, opt) = base_compile_args();
        let base_key = compile_cache_key(src, tgt, &eps, &defs, &layouts, opt);
        let eps2 = vec![("other", SlangStage::Fragment)];
        assert_ne!(base_key, compile_cache_key(src, tgt, &eps2, &defs, &layouts, opt));
    }

    #[test]
    fn compile_cache_key_sensitive_to_defines() {
        let (src, tgt, eps, defs, layouts, opt) = base_compile_args();
        let base_key = compile_cache_key(src, tgt, &eps, &defs, &layouts, opt);
        let defs2 = vec![("A", "2"), ("B", "2")];
        assert_ne!(base_key, compile_cache_key(src, tgt, &eps, &defs2, &layouts, opt));
    }

    #[test]
    fn compile_cache_key_sensitive_to_optimization_level() {
        let (src, tgt, eps, defs, layouts, opt) = base_compile_args();
        let base_key = compile_cache_key(src, tgt, &eps, &defs, &layouts, opt);
        assert_ne!(
            base_key,
            compile_cache_key(src, tgt, &eps, &defs, &layouts, OptimizationLevel::Maximal,),
        );
    }

    #[test]
    fn compile_cache_key_sensitive_to_layout_checks() {
        let (src, tgt, eps, defs, layouts, opt) = base_compile_args();
        let base_key = compile_cache_key(src, tgt, &eps, &defs, &layouts, opt);
        let mut layouts2 = layouts.clone();
        layouts2[0].rust_size = 32;
        assert_ne!(base_key, compile_cache_key(src, tgt, &eps, &defs, &layouts2, opt));
    }

    /// Disk cache keys must follow the same effective source as Slang (`compile_with_reflection`).
    #[test]
    fn compile_cache_key_stable_for_effective_goldy_source() {
        let goldy_src = r#"import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
"#;
        let (tgt, eps, defs, layouts, opt) = (
            ShaderTarget::Spirv,
            vec![("cs_main", SlangStage::Compute)],
            vec![("X", "1")],
            Vec::<OwnedLayoutCheck>::new(),
            OptimizationLevel::None,
        );
        let transformed = transform_virtual_main(goldy_src);
        let k_effective = compile_cache_key(
            effective_slang_source_for_compile(goldy_src).as_ref(),
            tgt,
            &eps,
            &defs,
            &layouts,
            opt,
        );
        let k_transformed_only = compile_cache_key(transformed.as_str(), tgt, &eps, &defs, &layouts, opt);
        assert_eq!(
            k_effective, k_transformed_only,
            "cache key must hash post-transform source, matching transform_virtual_main output"
        );
    }

    /// Raw `[goldy_*]` source differs from what Slang compiles; keys must not use raw alone.
    #[test]
    fn compile_cache_key_goldy_raw_differs_from_effective() {
        let goldy_src = r#"[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(Scattered<uint> buf, ThreadId id) { buf[id.x] = 0; }
"#;
        let (tgt, eps, defs, layouts, opt) = (
            ShaderTarget::Spirv,
            vec![("cs_main", SlangStage::Compute)],
            vec![] as Vec<(&str, &str)>,
            Vec::<OwnedLayoutCheck>::new(),
            OptimizationLevel::Default,
        );
        assert_ne!(
            compile_cache_key(goldy_src, tgt, &eps, &defs, &layouts, opt),
            compile_cache_key(
                effective_slang_source_for_compile(goldy_src).as_ref(),
                tgt,
                &eps,
                &defs,
                &layouts,
                opt,
            ),
            "pre-transform and effective sources must not collide in the cache key"
        );
    }

    #[test]
    fn compile_cache_key_plain_source_matches_effective_helper() {
        let src = "float4 main() : SV_Target0 { return 0; }";
        let (tgt, eps, defs, layouts, opt) = (
            ShaderTarget::Spirv,
            vec![("main", SlangStage::Fragment)],
            vec![] as Vec<(&str, &str)>,
            Vec::<OwnedLayoutCheck>::new(),
            OptimizationLevel::Default,
        );
        assert_eq!(
            compile_cache_key(src, tgt, &eps, &defs, &layouts, opt),
            compile_cache_key(
                effective_slang_source_for_compile(src).as_ref(),
                tgt,
                &eps,
                &defs,
                &layouts,
                opt,
            )
        );
    }

    #[test]
    fn cache_cold_start_on_missing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let c = ShaderBytecodeDiskCache::new_at_path(path.clone());
        assert!(c.should_skip_save_on_drop());
        assert!(!path.exists());
        assert!(c.is_empty());
    }

    #[test]
    fn insert_marks_dirty_get_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let v = dummy_compiled();
        let mut c = ShaderBytecodeDiskCache::new_at_path(path);
        assert!(c.should_skip_save_on_drop());
        c.insert(7, &v).unwrap();
        assert!(!c.should_skip_save_on_drop());
        let got = c.get(7).unwrap().unwrap();
        assert_eq!(got.shader.data, v.shader.data);
        assert_eq!(got.shader.target, v.shader.target);
    }

    #[test]
    fn insert_identical_blob_no_dirty_after_flush_and_re_insert() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let v = dummy_compiled();
        let mut c = ShaderBytecodeDiskCache::new_at_path(path);
        c.insert(1, &v).unwrap();
        c.flush_to_disk_best_effort();
        assert!(c.should_skip_save_on_drop());
        c.insert(1, &v).unwrap();
        assert!(c.should_skip_save_on_drop());
    }

    #[test]
    fn flush_not_dirty_no_file_created() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let mut c = ShaderBytecodeDiskCache::new_at_path(path.clone());
        c.flush_to_disk_best_effort();
        assert!(!path.exists());
    }

    #[test]
    fn flush_writes_file_and_clears_dirty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let v = dummy_compiled();
        let mut c = ShaderBytecodeDiskCache::new_at_path(path.clone());
        c.insert(99, &v).unwrap();
        assert!(!c.should_skip_save_on_drop());
        c.flush_to_disk_best_effort();
        assert!(c.should_skip_save_on_drop());
        assert!(path.exists());
    }

    #[test]
    fn drop_flushes_dirty_cache() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let v = dummy_compiled();
        {
            let mut c = ShaderBytecodeDiskCache::new_at_path(path.clone());
            c.insert(123, &v).unwrap();
        }
        assert!(path.exists());
    }

    #[test]
    fn load_back_after_flush() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let v = dummy_compiled();
        {
            let mut c = ShaderBytecodeDiskCache::new_at_path(path.clone());
            c.insert(42, &v).unwrap();
            c.flush_to_disk_best_effort();
        }
        let mut c2 = ShaderBytecodeDiskCache::new_at_path(path.clone());
        assert!(c2.version_ok_on_disk(), "fresh load should decode version");
        let got = c2.get(42).unwrap().unwrap();
        assert_eq!(got.shader.data, v.shader.data);
        assert_eq!(got.shader.target, v.shader.target);
    }

    fn write_compressed_shader_blob(buf: &[u8]) -> Vec<u8> {
        zstd::encode_all(buf, 10).unwrap()
    }

    #[test]
    fn version_mismatch_starts_cold() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let mut uncompressed = Vec::new();
        uncompressed.extend_from_slice(GOLDY_SHADER_CACHE_MAGIC.as_slice());
        uncompressed.extend_from_slice(b"this-is-not-the-build-cache-version-string\n");
        uncompressed.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, write_compressed_shader_blob(&uncompressed)).unwrap();
        let c = ShaderBytecodeDiskCache::new_at_path(path);
        assert!(!c.version_ok_on_disk());
        assert!(c.is_empty());
    }

    #[test]
    fn wrong_magic_starts_cold() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let mut uncompressed = Vec::new();
        uncompressed.extend_from_slice(b"BADMAGIC!!");
        uncompressed.extend_from_slice(GOLDY_CACHE_VERSION.as_bytes());
        uncompressed.push(b'\n');
        uncompressed.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, write_compressed_shader_blob(&uncompressed)).unwrap();
        let c = ShaderBytecodeDiskCache::new_at_path(path);
        assert!(c.is_empty());
    }

    #[test]
    fn truncated_body_starts_cold() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shader_cache.bin.zst");
        let mut uncompressed = Vec::new();
        uncompressed.extend_from_slice(GOLDY_SHADER_CACHE_MAGIC.as_slice());
        uncompressed.extend_from_slice(GOLDY_CACHE_VERSION.as_bytes());
        uncompressed.push(b'\n');
        uncompressed.extend_from_slice(&1u64.to_le_bytes()); // claim 1 entry
        uncompressed.extend_from_slice(&0u64.to_le_bytes()); // incomplete entry
        std::fs::write(&path, write_compressed_shader_blob(&uncompressed)).unwrap();
        let c = ShaderBytecodeDiskCache::new_at_path(path);
        assert!(c.is_empty());
    }
}
