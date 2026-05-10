//! Persisted DX12 cached PSO blobs (`GetCachedBlob`).

use crate::types::{DepthStencilState, PrimitiveTopology, TextureFormat, VertexBufferLayout};
use anyhow::Result;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::D3D12_CACHED_PIPELINE_STATE;

use super::utils;

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
const FILE_VERSION: u32 = 1;

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

pub(super) fn load_maps(path: &Path) -> PsoBlobMaps {
    let empty = (
        HashMap::<u64, Vec<u8>>::new(),
        HashMap::<u64, Vec<u8>>::new(),
    );
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return empty,
    };
    let Ok(parsed) = parse_file(&data) else {
        return empty;
    };
    parsed
}

fn parse_file(data: &[u8]) -> Result<PsoBlobMaps> {
    if data.len() < 8 + 4 + 8 + 8 {
        anyhow::bail!("truncated DX12 PSO cache");
    }
    let mut cursor = 0usize;
    if data[cursor..cursor + 8] != FILE_MAGIC[..] {
        anyhow::bail!("bad DX12 PSO cache magic");
    }
    cursor += 8;

    let version = u32::from_le_bytes(data[cursor..cursor + 4].try_into()?);
    if version != FILE_VERSION {
        anyhow::bail!("unsupported DX12 PSO cache version {version}");
    }
    cursor += 4;

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
    graphics: &HashMap<u64, Vec<u8>>,
    compute: &HashMap<u64, Vec<u8>>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = Vec::new();
    out.write_all(FILE_MAGIC)?;
    out.write_all(&FILE_VERSION.to_le_bytes())?;

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
