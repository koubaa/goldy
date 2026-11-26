//! Persisted DX12 cached PSO blobs (`GetCachedBlob`).
//!
//! A cached blob is valid only for the driver that produced it. After a driver
//! update, `CreateComputePipelineState` / `CreateGraphicsPipelineState` fail
//! with `D3D12_ERROR_DRIVER_VERSION_MISMATCH` (`0x887E0002`). Blobs from one
//! driver generation share a 24-byte prefix (header GUID plus driver id), so a
//! single rejection drops that whole generation instead of failing every PSO.

use crate::types::{DepthStencilState, PrimitiveTopology, TextureFormat, VertexBufferLayout};
use anyhow::Result;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use windows::Win32::Foundation::{D3D12_ERROR_ADAPTER_NOT_FOUND, D3D12_ERROR_DRIVER_VERSION_MISMATCH};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::D3D12_CACHED_PIPELINE_STATE;

use super::types::PsoCache;
use super::utils;

/// Header GUID (16) plus the 8-byte driver id embedded by `GetCachedBlob`.
pub(super) const DRIVER_STAMP_PREFIX: usize = 24;

/// Adapter identity stored in the on-disk cache header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PsoCacheIdentity {
    pub driver_version: u64,
    pub vendor_id: u32,
    pub device_id: u32,
}

pub(super) fn is_driver_cache_mismatch(err: &windows::core::Error) -> bool {
    let code = err.code();
    code == D3D12_ERROR_DRIVER_VERSION_MISMATCH || code == D3D12_ERROR_ADAPTER_NOT_FOUND
}

/// Drop every cached blob that shares `rejected`'s driver-generation prefix.
///
/// `rejected` itself may already have been removed. Returns how many entries
/// were removed from the two maps.
pub(super) fn drop_matching_driver_stamp(cache: &mut PsoCache, rejected: &[u8]) -> usize {
    let Some(stamp) = rejected.get(..DRIVER_STAMP_PREFIX) else {
        return 0;
    };
    let matches = |blob: &Vec<u8>| blob.len() >= DRIVER_STAMP_PREFIX && &blob[..DRIVER_STAMP_PREFIX] == stamp;
    let graphics = cache.graphics_blobs.values().filter(|blob| matches(blob)).count();
    let compute = cache.compute_blobs.values().filter(|blob| matches(blob)).count();
    if graphics + compute == 0 {
        return 0;
    }
    cache.graphics_blobs.retain(|_, blob| !matches(blob));
    cache.compute_blobs.retain(|_, blob| !matches(blob));
    cache.dirty = true;
    graphics + compute
}

/// Keep only blobs from the same driver generation as `live`.
///
/// Runs once per device. A cache hit on a current blob therefore clears
/// leftovers from an older driver without attempting those creates.
pub(super) fn retain_live_driver_stamp(cache: &mut PsoCache, live: &[u8]) {
    if cache.driver_stamp_aligned {
        return;
    }
    cache.driver_stamp_aligned = true;
    let Some(stamp) = live.get(..DRIVER_STAMP_PREFIX) else {
        return;
    };
    let foreign = |blob: &Vec<u8>| blob.len() >= DRIVER_STAMP_PREFIX && &blob[..DRIVER_STAMP_PREFIX] != stamp;
    let graphics = cache.graphics_blobs.values().filter(|blob| foreign(blob)).count();
    let compute = cache.compute_blobs.values().filter(|blob| foreign(blob)).count();
    if graphics + compute == 0 {
        return;
    }
    cache.graphics_blobs.retain(|_, blob| !foreign(blob));
    cache.compute_blobs.retain(|_, blob| !foreign(blob));
    cache.dirty = true;
    tracing::warn!(
        dropped_graphics = graphics,
        dropped_compute = compute,
        "dropping DX12 PSO cache entries built for a different driver"
    );
}

#[must_use]
pub(super) fn d3d12_cached_pso(blob: &[u8]) -> D3D12_CACHED_PIPELINE_STATE {
    D3D12_CACHED_PIPELINE_STATE {
        pCachedBlob: blob.as_ptr().cast::<std::ffi::c_void>(),
        CachedBlobSizeInBytes: blob.len(),
    }
}

pub(super) unsafe fn id3dblob_to_vec(blob: &ID3DBlob) -> Vec<u8> {
    let ptr = blob.GetBufferPointer() as *const u8;
    let len = blob.GetBufferSize();
    std::slice::from_raw_parts(ptr, len).to_vec()
}
const FILE_MAGIC: &[u8; 8] = b"GD12PSOB";
/// Version 2 stores the DXGI driver version and vendor/device id ahead of the maps.
/// Version 1 files (maps only) still load so an in-use cache is not thrown away.
const FILE_VERSION: u32 = 2;
const FILE_VERSION_LEGACY: u32 = 1;

fn vertex_format_tag(f: crate::types::VertexFormat) -> u8 {
    use crate::types::VertexFormat as V;
    match f {
        V::Float32 => 1,
        V::Float32x2 => 2,
        V::Float32x3 => 3,
        V::Float32x4 => 4,
        V::Uint32 => 5,
        V::Sint32 => 6,
        V::Uint8x4 => 7,
        V::Unorm8x4 => 8,
    }
}

fn depth_format_tag(d: crate::types::DepthFormat) -> u8 {
    use crate::types::DepthFormat as D;
    match d {
        D::Depth16Unorm => 1,
        D::Depth24Plus => 2,
        D::Depth24PlusStencil8 => 3,
        D::Depth32Float => 4,
        D::Depth32FloatStencil8 => 5,
    }
}

fn topology_tag(topology: PrimitiveTopology) -> u8 {
    use crate::types::PrimitiveTopology as T;
    match topology {
        T::PointList => 1,
        T::LineList => 2,
        T::LineStrip => 3,
        T::TriangleList => 4,
        T::TriangleStrip => 5,
    }
}

fn texture_format_tag(f: TextureFormat) -> u8 {
    match f {
        TextureFormat::R8Unorm => 1,
        TextureFormat::Rg8Unorm => 2,
        TextureFormat::Rgba8UnormSrgb => 3,
        TextureFormat::Rgba8Unorm => 4,
        TextureFormat::Bgra8UnormSrgb => 5,
        TextureFormat::Bgra8Unorm => 6,
        TextureFormat::Rgba16Float => 7,
        TextureFormat::Rgba32Float => 8,
    }
}

/// FNV-1a 64-bit hash of concatenated blobs (delimiter-separated sections).
#[must_use]
pub(super) fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Cache key covering shader bytecode plus everything that participates in [`D3D12_GRAPHICS_PIPELINE_STATE_DESC`](windows::Win32::Graphics::Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC).
#[must_use]
pub(super) fn graphics_pso_key(
    vs: &[u8],
    fs: &[u8],
    vertex_layout: &VertexBufferLayout,
    topology: PrimitiveTopology,
    target_format: TextureFormat,
    depth_stencil: Option<&DepthStencilState>,
) -> u64 {
    let mut buf = Vec::new();

    buf.extend_from_slice(vs);
    buf.push(0x01);
    buf.extend_from_slice(fs);
    buf.push(0x02);
    buf.extend_from_slice(&vertex_layout.stride.to_le_bytes());
    for attr in &vertex_layout.attributes {
        buf.extend_from_slice(&attr.location.to_le_bytes());
        buf.push(vertex_format_tag(attr.format));
        buf.extend_from_slice(&attr.offset.to_le_bytes());
    }
    buf.push(0x03);
    buf.push(topology_tag(topology));
    buf.extend_from_slice(&(utils::topology_type_to_d3d12(topology).0.to_le_bytes()));
    buf.push(0x04);
    buf.push(texture_format_tag(target_format));
    buf.extend_from_slice(&(utils::format_to_dxgi(target_format).0.to_le_bytes()));
    buf.push(0x05);
    if let Some(ds) = depth_stencil {
        buf.push(1);
        buf.push(depth_format_tag(ds.format));
        buf.extend_from_slice(&(utils::depth_format_to_dxgi(ds.format).0.to_le_bytes()));
        buf.push(if ds.depth_write_enabled { 1 } else { 0 });
        buf.extend_from_slice(&(utils::compare_to_d3d12(ds.depth_compare).0.to_le_bytes()));
    } else {
        buf.push(0);
    }

    fnv1a64(&buf)
}

#[must_use]
pub(super) fn compute_pso_key(cs: &[u8]) -> u64 {
    let mut buf = Vec::with_capacity(cs.len() + 1);
    buf.push(0xaa);
    buf.extend_from_slice(cs);
    fnv1a64(&buf)
}

pub(super) type PsoBlobMaps = (HashMap<u64, Vec<u8>>, HashMap<u64, Vec<u8>>);

pub(super) fn load_maps(path: &Path, identity: PsoCacheIdentity) -> PsoBlobMaps {
    let empty = (HashMap::<u64, Vec<u8>>::new(), HashMap::<u64, Vec<u8>>::new());
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return empty,
    };
    match parse_file(&data, Some(identity)) {
        Ok(parsed) => parsed,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "ignoring DX12 PSO disk cache"
            );
            empty
        }
    }
}

fn parse_file(data: &[u8], expected: Option<PsoCacheIdentity>) -> Result<PsoBlobMaps> {
    if data.len() < 8 + 4 + 8 + 8 {
        anyhow::bail!("truncated DX12 PSO cache");
    }
    let mut cursor = 0usize;
    if data[cursor..cursor + 8] != FILE_MAGIC[..] {
        anyhow::bail!("bad DX12 PSO cache magic");
    }
    cursor += 8;

    let version = u32::from_le_bytes(data[cursor..cursor + 4].try_into()?);
    cursor += 4;
    if version == FILE_VERSION {
        if cursor + 8 + 4 + 4 > data.len() {
            anyhow::bail!("truncated DX12 PSO cache identity");
        }
        let driver_version = u64::from_le_bytes(data[cursor..cursor + 8].try_into()?);
        cursor += 8;
        let vendor_id = u32::from_le_bytes(data[cursor..cursor + 4].try_into()?);
        cursor += 4;
        let device_id = u32::from_le_bytes(data[cursor..cursor + 4].try_into()?);
        cursor += 4;
        if let Some(expected) = expected {
            let got = PsoCacheIdentity {
                driver_version,
                vendor_id,
                device_id,
            };
            if got != expected {
                anyhow::bail!(
                    "DX12 PSO cache driver {driver_version:#x} vendor {vendor_id:#x} device {device_id:#x} does not match this adapter (driver {:#x} vendor {:#x} device {:#x})",
                    expected.driver_version,
                    expected.vendor_id,
                    expected.device_id,
                );
            }
        }
    } else if version != FILE_VERSION_LEGACY {
        anyhow::bail!("unsupported DX12 PSO cache version {version}");
    }

    if cursor + 8 > data.len() {
        anyhow::bail!("truncated DX12 PSO cache");
    }
    let n_graphics = usize::try_from(u64::from_le_bytes(data[cursor..cursor + 8].try_into()?))?;
    cursor += 8;

    let mut graphics = HashMap::with_capacity(n_graphics);
    for _ in 0..n_graphics {
        if cursor + 8 + 4 > data.len() {
            anyhow::bail!("truncated DX12 graphics PSO entries");
        }
        let key = u64::from_le_bytes(data[cursor..cursor + 8].try_into()?);
        cursor += 8;
        let blob_len = usize::try_from(u32::from_le_bytes(data[cursor..cursor + 4].try_into()?))?;
        cursor += 4;
        if cursor + blob_len > data.len() {
            anyhow::bail!("truncated DX12 graphics PSO blob");
        }
        graphics.insert(key, data[cursor..cursor + blob_len].to_vec());
        cursor += blob_len;
    }

    if cursor + 8 > data.len() {
        anyhow::bail!("missing compute section");
    }
    let n_compute = usize::try_from(u64::from_le_bytes(data[cursor..cursor + 8].try_into()?))?;
    cursor += 8;

    let mut compute = HashMap::with_capacity(n_compute);
    for _ in 0..n_compute {
        if cursor + 8 + 4 > data.len() {
            anyhow::bail!("truncated DX12 compute PSO entries");
        }
        let key = u64::from_le_bytes(data[cursor..cursor + 8].try_into()?);
        cursor += 8;
        let blob_len = usize::try_from(u32::from_le_bytes(data[cursor..cursor + 4].try_into()?))?;
        cursor += 4;
        if cursor + blob_len > data.len() {
            anyhow::bail!("truncated DX12 compute PSO blob");
        }
        compute.insert(key, data[cursor..cursor + blob_len].to_vec());
        cursor += blob_len;
    }

    Ok((graphics, compute))
}

pub(super) fn save_maps(
    path: &Path,
    identity: PsoCacheIdentity,
    graphics: &HashMap<u64, Vec<u8>>,
    compute: &HashMap<u64, Vec<u8>>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = Vec::new();
    out.write_all(FILE_MAGIC)?;
    out.write_all(&FILE_VERSION.to_le_bytes())?;
    out.write_all(&identity.driver_version.to_le_bytes())?;
    out.write_all(&identity.vendor_id.to_le_bytes())?;
    out.write_all(&identity.device_id.to_le_bytes())?;

    let n_g = graphics.len().min(u64::MAX as usize) as u64;
    out.write_all(&n_g.to_le_bytes())?;
    for (&key, blob) in graphics.iter() {
        out.write_all(&key.to_le_bytes())?;
        let len = blob.len().min(u64::from(u32::MAX) as usize);
        #[allow(clippy::cast_possible_truncation)]
        {
            out.write_all(&(len as u32).to_le_bytes())?;
        }
        out.write_all(&blob[..len])?;
    }

    let n_c = compute.len().min(u64::MAX as usize) as u64;
    out.write_all(&n_c.to_le_bytes())?;
    for (&key, blob) in compute.iter() {
        out.write_all(&key.to_le_bytes())?;
        let len = blob.len().min(u64::from(u32::MAX) as usize);
        #[allow(clippy::cast_possible_truncation)]
        {
            out.write_all(&(len as u32).to_le_bytes())?;
        }
        out.write_all(&blob[..len])?;
    }

    std::fs::write(path, out)
}

#[cfg(all(test, target_os = "windows"))]
mod tests {
    use super::*;
    use crate::types::{
        CompareFunction, DepthFormat, DepthStencilState, PrimitiveTopology, TextureFormat, VertexAttribute,
        VertexBufferLayout, VertexFormat,
    };
    use tempfile::TempDir;

    #[test]
    fn fnv1a64_deterministic() {
        let b = b"hello_dx12_pso_hash";
        assert_eq!(fnv1a64(b), fnv1a64(b));
    }

    #[test]
    fn fnv1a64_different_inputs_differ() {
        let mut a = vec![7u8, 42, 1, 2, 3];
        let h0 = fnv1a64(&a);
        let last = usize::checked_sub(a.len(), 1).unwrap();
        a[last] ^= 0xff;
        assert_ne!(h0, fnv1a64(&a));
    }

    fn sample_vertex_layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: 16,
            attributes: vec![VertexAttribute {
                location: 0,
                format: VertexFormat::Float32x3,
                offset: 0,
            }],
        }
    }

    fn base_gfx_key_args() -> (
        &'static [u8],
        &'static [u8],
        VertexBufferLayout,
        PrimitiveTopology,
        TextureFormat,
        Option<DepthStencilState>,
    ) {
        (
            b"vs bytecode",
            b"fs bytecode",
            sample_vertex_layout(),
            PrimitiveTopology::TriangleList,
            TextureFormat::R8Unorm,
            Some(DepthStencilState {
                format: DepthFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
            }),
        )
    }

    #[test]
    fn graphics_pso_key_sensitive_to_vs() {
        let (_, fs, vl, topo, tf, ds) = base_gfx_key_args();
        let k0 = graphics_pso_key(b"vs_a", fs, &vl, topo, tf, ds.as_ref());
        assert_ne!(k0, graphics_pso_key(b"vs_b", fs, &vl, topo, tf, ds.as_ref()),);
    }

    #[test]
    fn graphics_pso_key_sensitive_to_fs() {
        let (vs, _, vl, topo, tf, ds) = base_gfx_key_args();
        let k0 = graphics_pso_key(vs, b"fs_a", &vl, topo, tf, ds.as_ref());
        assert_ne!(k0, graphics_pso_key(vs, b"fs_b", &vl, topo, tf, ds.as_ref()),);
    }

    #[test]
    fn graphics_pso_key_sensitive_to_topology() {
        let (vs, fs, vl, _, tf, ds) = base_gfx_key_args();
        let line = PrimitiveTopology::LineList;
        let tri = PrimitiveTopology::TriangleList;
        assert_ne!(
            graphics_pso_key(vs, fs, &vl, tri, tf, ds.as_ref()),
            graphics_pso_key(vs, fs, &vl, line, tf, ds.as_ref()),
        );
    }

    #[test]
    fn graphics_pso_key_sensitive_to_target_format() {
        let (vs, fs, vl, topo, _, ds) = base_gfx_key_args();
        assert_ne!(
            graphics_pso_key(vs, fs, &vl, topo, TextureFormat::R8Unorm, ds.as_ref()),
            graphics_pso_key(vs, fs, &vl, topo, TextureFormat::Rgba32Float, ds.as_ref(),),
        );
    }

    #[test]
    fn graphics_pso_key_sensitive_to_depth_stencil() {
        let (vs, fs, vl, topo, tf, _) = base_gfx_key_args();
        let ds = DepthStencilState {
            format: DepthFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: CompareFunction::Always,
        };
        assert_ne!(
            graphics_pso_key(vs, fs, &vl, topo, tf, None),
            graphics_pso_key(vs, fs, &vl, topo, tf, Some(&ds)),
        );
    }

    #[test]
    fn compute_pso_key_deterministic() {
        let cs = b"minimal_cs_blob";
        assert_eq!(compute_pso_key(cs), compute_pso_key(cs));
    }

    #[test]
    fn compute_pso_key_differs_with_cs_change() {
        assert_ne!(compute_pso_key(b"aaa"), compute_pso_key(b"aab"));
    }

    fn test_identity() -> PsoCacheIdentity {
        PsoCacheIdentity {
            driver_version: 0x20_0000_0010_0440,
            vendor_id: 0x10de,
            device_id: 0x2803,
        }
    }

    #[test]
    fn save_load_roundtrip_empty() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("pso.bin");
        let g_in = HashMap::<u64, Vec<u8>>::new();
        let c_in = HashMap::<u64, Vec<u8>>::new();
        save_maps(&path, test_identity(), &g_in, &c_in).unwrap();
        let (g_out, c_out) = load_maps(&path, test_identity());
        assert!(g_out.is_empty());
        assert!(c_out.is_empty());
    }

    #[test]
    fn save_load_roundtrip_populated() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("pso.bin");
        let mut g = HashMap::new();
        g.insert(1u64, vec![10, 20, 30]);
        g.insert(9u64, vec![255]);
        let mut c = HashMap::new();
        c.insert(100u64, vec![1, 2, 3, 4]);
        save_maps(&path, test_identity(), &g, &c).unwrap();
        let (g2, c2) = load_maps(&path, test_identity());
        assert_eq!(g2, g);
        assert_eq!(c2, c);
    }

    #[test]
    fn load_maps_driver_mismatch_returns_empty() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("pso.bin");
        let mut c = HashMap::new();
        c.insert(1u64, vec![9, 9, 9]);
        save_maps(&path, test_identity(), &HashMap::new(), &c).unwrap();
        let other = PsoCacheIdentity {
            driver_version: test_identity().driver_version.wrapping_add(1),
            ..test_identity()
        };
        let (g, c_out) = load_maps(&path, other);
        assert!(g.is_empty());
        assert!(c_out.is_empty());
    }

    #[test]
    fn load_maps_legacy_v1_still_loads() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("pso.bin");
        let mut data = Vec::new();
        data.extend_from_slice(FILE_MAGIC);
        data.extend_from_slice(&FILE_VERSION_LEGACY.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&7u64.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&[1, 2, 3]);
        data.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, data).unwrap();
        let (g, c) = load_maps(&path, test_identity());
        assert_eq!(g.get(&7), Some(&vec![1, 2, 3]));
        assert!(c.is_empty());
    }

    #[test]
    fn driver_stamp_rejection_drops_whole_generation() {
        let mut cache = PsoCache::new(HashMap::new(), HashMap::new(), 1, 0x10de, 0x2803);
        let old = vec![0u8; DRIVER_STAMP_PREFIX];
        let mut kept = vec![1u8; DRIVER_STAMP_PREFIX];
        kept.extend_from_slice(&[9]);
        cache.graphics_blobs.insert(1, old.clone());
        cache.compute_blobs.insert(2, old.clone());
        cache.compute_blobs.insert(3, kept.clone());
        let dropped = drop_matching_driver_stamp(&mut cache, &old);
        assert_eq!(dropped, 2);
        assert!(cache.graphics_blobs.is_empty());
        assert_eq!(cache.compute_blobs.get(&3), Some(&kept));
        assert!(cache.dirty);
    }

    #[test]
    fn live_driver_stamp_drops_foreign_entries_once() {
        let mut cache = PsoCache::new(HashMap::new(), HashMap::new(), 1, 0x10de, 0x2803);
        let mut old = vec![0u8; DRIVER_STAMP_PREFIX];
        old[0] = 2;
        let live = vec![7u8; DRIVER_STAMP_PREFIX + 4];
        cache.compute_blobs.insert(1, old);
        cache.compute_blobs.insert(2, live.clone());
        retain_live_driver_stamp(&mut cache, &live);
        assert_eq!(cache.compute_blobs.len(), 1);
        retain_live_driver_stamp(&mut cache, &live);
        assert!(cache.driver_stamp_aligned);
        assert_eq!(cache.compute_blobs.len(), 1);
    }

    #[test]
    fn load_maps_missing_file_returns_empty() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("never_written.bin");
        let (g, c) = load_maps(&path, test_identity());
        assert!(g.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn parse_file_bad_magic() {
        let mut data = vec![b'X'; 8];
        data.extend_from_slice(&FILE_VERSION.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse_file(&data, None).is_err());
    }

    #[test]
    fn parse_file_wrong_version() {
        let mut data = Vec::new();
        data.extend_from_slice(FILE_MAGIC);
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse_file(&data, None).is_err());
    }

    #[test]
    fn parse_file_truncated() {
        let mut data = Vec::new();
        data.extend_from_slice(FILE_MAGIC);
        data.extend_from_slice(&FILE_VERSION.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse_file(&data, None).is_err());
    }
}
