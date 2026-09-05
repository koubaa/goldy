//! Rust-authored GPU types lowered to generated Slang structs.
//!
//! Host `#[repr(C)]` structs keep logical fields only. Goldy packs them into the
//! Slang structured-buffer ABI (16-byte no-straddle) at upload and when emitting
//! `__goldy_padN` fields.

use crate::types::{VertexAttribute, VertexBufferLayout, VertexFormat};
use anyhow::{bail, Result};

/// Portable field types supported by [`GpuType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFieldType {
    F32,
    U32,
    I32,
    F32x2,
    F32x3,
    F32x4,
    U32x2,
    U32x3,
    U32x4,
    I32x2,
    I32x3,
    I32x4,
    F32x2x2,
    F32x3x3,
    F32x4x4,
}

impl GpuFieldType {
    pub const fn slang_name(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::U32 => "uint",
            Self::I32 => "int",
            Self::F32x2 => "float2",
            Self::F32x3 => "float3",
            Self::F32x4 => "float4",
            Self::U32x2 => "uint2",
            Self::U32x3 => "uint3",
            Self::U32x4 => "uint4",
            Self::I32x2 => "int2",
            Self::I32x3 => "int3",
            Self::I32x4 => "int4",
            Self::F32x2x2 => "float2x2",
            Self::F32x3x3 => "float3x3",
            Self::F32x4x4 => "float4x4",
        }
    }

    pub const fn size(self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F32x2 | Self::U32x2 | Self::I32x2 => 8,
            Self::F32x3 | Self::U32x3 | Self::I32x3 => 12,
            Self::F32x4 | Self::U32x4 | Self::I32x4 | Self::F32x2x2 => 16,
            Self::F32x3x3 => 36,
            Self::F32x4x4 => 64,
        }
    }

    /// HLSL structured-buffer alignment for this type.
    pub const fn align(self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 | Self::F32x3 | Self::U32x3 | Self::I32x3 => 4,
            Self::F32x2 | Self::U32x2 | Self::I32x2 => 8,
            Self::F32x4 | Self::U32x4 | Self::I32x4 | Self::F32x2x2 | Self::F32x3x3 | Self::F32x4x4 => 16,
        }
    }

    fn vertex_format(self) -> Result<VertexFormat> {
        Ok(match self {
            Self::F32 => VertexFormat::Float32,
            Self::F32x2 => VertexFormat::Float32x2,
            Self::F32x3 => VertexFormat::Float32x3,
            Self::F32x4 => VertexFormat::Float32x4,
            Self::U32 => VertexFormat::Uint32,
            Self::I32 => VertexFormat::Sint32,
            _ => bail!(
                "GpuType field type `{}` cannot be used as a raster vertex attribute",
                self.slang_name()
            ),
        })
    }
}

/// One logical field in a Rust-authored GPU struct (`offset` is the host `offset_of`).
#[derive(Debug, Clone, Copy)]
pub struct GpuField<'a> {
    pub name: &'a str,
    pub offset: usize,
    pub size: usize,
    pub ty: GpuFieldType,
}

/// Host type metadata. Storage offsets are computed by [`GpuType::packed`].
#[derive(Debug, Clone, Copy)]
pub struct GpuType<'a> {
    pub type_name: &'a str,
    pub rust_size: usize,
    pub fields: &'a [GpuField<'a>],
}

/// One logical field after structured-buffer packing.
#[derive(Debug, Clone, Copy)]
pub struct PackedGpuField {
    pub host_offset: usize,
    pub storage_offset: usize,
    pub size: usize,
}

/// Structured-buffer ABI for a [`GpuType`].
#[derive(Debug, Clone)]
pub struct PackedGpuLayout {
    pub stride: usize,
    pub fields: Vec<PackedGpuField>,
}

/// Generated declaration and matching reflection check.
#[derive(Debug)]
pub(crate) struct GeneratedGpuType {
    pub source: String,
    pub check: super::OwnedLayoutCheck,
}

const fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

fn place_storage_field(cursor: usize, size: usize, align: usize) -> usize {
    let mut offset = align_up(cursor, align);
    if size > 0 && size <= 16 && offset / 16 != (offset + size - 1) / 16 {
        offset = align_up(offset, 16);
    }
    offset
}

impl GpuType<'_> {
    /// Slang structured-buffer layout for this type (independent of host padding).
    pub fn packed(&self) -> Result<PackedGpuLayout> {
        if self.type_name.is_empty() {
            bail!("GpuType name cannot be empty");
        }
        let mut fields = Vec::with_capacity(self.fields.len());
        let mut cursor = 0usize;
        let mut struct_align = 4usize;
        for field in self.fields {
            if field.name.starts_with("__goldy_pad") {
                bail!(
                    "GpuType `{}` field `{}` uses reserved prefix `__goldy_pad`",
                    self.type_name,
                    field.name
                );
            }
            let expected = field.ty.size();
            if field.size != expected {
                bail!(
                    "GpuType `{}` field `{}` is {} bytes but `{}` requires {}",
                    self.type_name,
                    field.name,
                    field.size,
                    field.ty.slang_name(),
                    expected
                );
            }
            if field.offset + field.size > self.rust_size {
                bail!(
                    "GpuType `{}` field `{}` extends past host size {}",
                    self.type_name,
                    field.name,
                    self.rust_size
                );
            }
            let storage_offset = place_storage_field(cursor, field.size, field.ty.align());
            if field.ty.align() >= 16 {
                struct_align = 16;
            }
            fields.push(PackedGpuField {
                host_offset: field.offset,
                storage_offset,
                size: field.size,
            });
            cursor = storage_offset + field.size;
        }
        Ok(PackedGpuLayout {
            stride: align_up(cursor, struct_align),
            fields,
        })
    }

    /// Byte stride of one element in a structured buffer / vertex buffer.
    pub fn storage_stride(&self) -> Result<usize> {
        Ok(self.packed()?.stride)
    }

    /// True when host `repr(C)` already matches the storage ABI (no pack copy needed).
    pub fn storage_matches_host(&self) -> Result<bool> {
        let packed = self.packed()?;
        if packed.stride != self.rust_size {
            return Ok(false);
        }
        Ok(self
            .fields
            .iter()
            .zip(packed.fields.iter())
            .all(|(field, packed_field)| field.offset == packed_field.storage_offset))
    }

    /// Pack host bytes (`n * rust_size`) into storage (`n * storage_stride`).
    pub fn encode_bytes(&self, host_bytes: &[u8]) -> Result<Vec<u8>> {
        if host_bytes.len() % self.rust_size != 0 {
            bail!(
                "GpuType `{}` host blob length {} is not a multiple of {}",
                self.type_name,
                host_bytes.len(),
                self.rust_size
            );
        }
        let packed = self.packed()?;
        let count = if self.rust_size == 0 {
            0
        } else {
            host_bytes.len() / self.rust_size
        };
        let mut out = vec![0u8; count * packed.stride];
        for i in 0..count {
            let src = &host_bytes[i * self.rust_size..];
            let dst_base = i * packed.stride;
            for field in &packed.fields {
                let src_range = field.host_offset..field.host_offset + field.size;
                let dst_range = dst_base + field.storage_offset..dst_base + field.storage_offset + field.size;
                out[dst_range].copy_from_slice(&src[src_range]);
            }
        }
        Ok(out)
    }

    /// Pack a typed host slice into storage bytes.
    pub fn encode_pod_slice<T: bytemuck::Pod>(&self, items: &[T]) -> Result<Vec<u8>> {
        if std::mem::size_of::<T>() != self.rust_size {
            bail!(
                "GpuType `{}` host size {} does not match `size_of` {}",
                self.type_name,
                self.rust_size,
                std::mem::size_of::<T>()
            );
        }
        if self.storage_matches_host()? {
            return Ok(bytemuck::cast_slice(items).to_vec());
        }
        self.encode_bytes(bytemuck::cast_slice(items))
    }

    /// Raster IA layout using storage offsets (padding never consumes a semantic slot).
    pub fn vertex_buffer_layout(&self) -> Result<VertexBufferLayout> {
        let packed = self.packed()?;
        let mut attributes = Vec::with_capacity(self.fields.len());
        for (index, (field, packed_field)) in self.fields.iter().zip(packed.fields.iter()).enumerate() {
            attributes.push(VertexAttribute {
                location: index as u32,
                format: field.ty.vertex_format()?,
                offset: packed_field.storage_offset as u32,
            });
        }
        Ok(VertexBufferLayout {
            stride: packed.stride as u32,
            attributes,
        })
    }

    /// Slang `struct` declaration for this type, including synthetic padding fields.
    pub fn to_slang_source(&self) -> Result<String> {
        Ok(self.generate()?.source)
    }

    /// Emit a Slang struct whose explicit padding reproduces the storage ABI.
    pub(crate) fn generate(&self) -> Result<GeneratedGpuType> {
        let packed = self.packed()?;
        let mut source = format!(
            "// generated by goldy from Rust `{}`\nstruct {} {{\n",
            self.type_name, self.type_name
        );
        let mut reflected_fields = Vec::new();
        let mut cursor = 0usize;
        let mut pad_index = 0usize;

        for (field, packed_field) in self.fields.iter().zip(packed.fields.iter()) {
            emit_padding(
                &mut source,
                &mut reflected_fields,
                cursor,
                packed_field.storage_offset - cursor,
                &mut pad_index,
                self.type_name,
            )?;
            source.push_str(&format!("    {} {};\n", field.ty.slang_name(), field.name));
            reflected_fields.push((field.name.to_string(), packed_field.storage_offset, packed_field.size));
            cursor = packed_field.storage_offset + packed_field.size;
        }
        emit_padding(
            &mut source,
            &mut reflected_fields,
            cursor,
            packed.stride - cursor,
            &mut pad_index,
            self.type_name,
        )?;
        source.push_str("};\n");

        Ok(GeneratedGpuType {
            source,
            check: super::OwnedLayoutCheck {
                type_name: self.type_name.to_string(),
                rust_size: packed.stride,
                rust_fields: reflected_fields,
            },
        })
    }
}

fn emit_padding(
    source: &mut String,
    reflected_fields: &mut Vec<(String, usize, usize)>,
    start: usize,
    byte_count: usize,
    pad_index: &mut usize,
    type_name: &str,
) -> Result<()> {
    if byte_count == 0 {
        return Ok(());
    }
    if byte_count % 4 != 0 {
        bail!(
            "GpuType `{type_name}` has an unrepresentable {byte_count}-byte padding gap at offset {start}; \
             generated Slang padding currently requires 4-byte granularity"
        );
    }
    for word in 0..(byte_count / 4) {
        let name = format!("__goldy_pad{}", *pad_index);
        let offset = start + word * 4;
        source.push_str(&format!("    uint {name};\n"));
        reflected_fields.push((name, offset, 4));
        *pad_index += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packer_inserts_float3_then_float2_gap() {
        let fields = [
            GpuField {
                name: "position",
                offset: 0,
                size: 12,
                ty: GpuFieldType::F32x3,
            },
            GpuField {
                name: "uv",
                offset: 12,
                size: 8,
                ty: GpuFieldType::F32x2,
            },
        ];
        let packed = GpuType {
            type_name: "Vertex",
            rust_size: 20,
            fields: &fields,
        }
        .packed()
        .unwrap();
        assert_eq!(packed.fields[1].storage_offset, 16);
        assert_eq!(packed.stride, 24);
    }

    #[test]
    fn generated_struct_inserts_internal_and_tail_padding() {
        let fields = [
            GpuField {
                name: "position",
                offset: 0,
                size: 12,
                ty: GpuFieldType::F32x3,
            },
            GpuField {
                name: "uv",
                offset: 12,
                size: 8,
                ty: GpuFieldType::F32x2,
            },
        ];
        let generated = GpuType {
            type_name: "Vertex",
            rust_size: 20,
            fields: &fields,
        }
        .generate()
        .unwrap();

        assert!(generated.source.contains("float3 position;"));
        assert!(generated.source.contains("uint __goldy_pad0;"));
        assert!(generated.source.contains("float2 uv;"));
        assert_eq!(
            generated.check.rust_fields,
            vec![
                ("position".into(), 0, 12),
                ("__goldy_pad0".into(), 12, 4),
                ("uv".into(), 16, 8),
            ]
        );
        assert_eq!(generated.check.rust_size, 24);
    }

    #[test]
    fn generated_struct_rejects_reserved_field_names() {
        let fields = [GpuField {
            name: "__goldy_pad7",
            offset: 0,
            size: 4,
            ty: GpuFieldType::U32,
        }];
        let err = GpuType {
            type_name: "Bad",
            rust_size: 4,
            fields: &fields,
        }
        .generate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("reserved prefix"), "{err}");
    }

    #[test]
    fn encode_copies_logical_fields_into_storage_gaps() {
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, goldy_derive::GpuType)]
        #[repr(C)]
        struct TightVertex {
            position: [f32; 3],
            uv: [f32; 2],
        }

        let host = TightVertex {
            position: [1.0, 2.0, 3.0],
            uv: [4.0, 5.0],
        };
        let bytes = TightVertex::GPU_TYPE.encode_pod_slice(&[host]).unwrap();
        assert_eq!(bytes.len(), 24);
        let pos: [f32; 3] = bytemuck::pod_read_unaligned(&bytes[0..12]);
        let uv: [f32; 2] = bytemuck::pod_read_unaligned(&bytes[16..24]);
        assert_eq!(pos, [1.0, 2.0, 3.0]);
        assert_eq!(uv, [4.0, 5.0]);
        assert_eq!(&bytes[12..16], &[0, 0, 0, 0]);
    }

    #[test]
    fn tight_float3_then_float2_matches_structured_buffer_stride() {
        use crate::slang::{ShaderTarget, SlangCompiler, SlangStage};
        use crate::types::OptimizationLevel;

        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, goldy_derive::GpuType)]
        #[repr(C)]
        struct TightVertex {
            position: [f32; 3],
            uv: [f32; 2],
            light: u32,
        }

        let generated = TightVertex::GPU_TYPE.generate().unwrap();
        assert!(
            generated.source.contains("uint __goldy_pad0;"),
            "packer must insert reserved padding:\n{}",
            generated.source
        );
        let source = format!(
            "{}\nimport goldy_exp;\n\
             [goldy_compute]\n[numthreads(1, 1, 1)]\n\
             void cs_main(BufRO<TightVertex> input, Scattered<uint> output, ThreadId id) {{\n\
                 TightVertex v = input[id.x];\n\
                 output[id.x] = asuint(v.position.x + v.uv.x) + v.light;\n\
             }}",
            generated.source
        );
        let compiler = SlangCompiler::new().expect("Slang compiler");
        let shader_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .to_string_lossy()
            .into_owned();

        let result = compiler
            .compile_with_reflection(
                &source,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[&shader_path],
                &[("__SPIRV__", "1")],
                &[generated.check],
                OptimizationLevel::None,
            )
            .expect("generated struct must compile and reflect");
        assert_eq!(result.reflection.binding_element_strides[0], Some(28));

        #[cfg(windows)]
        {
            let generated = TightVertex::GPU_TYPE.generate().unwrap();
            let result = compiler
                .compile_with_reflection(
                    &source,
                    ShaderTarget::Dxil,
                    &[("cs_main", SlangStage::Compute)],
                    &[&shader_path],
                    &[("__DX12__", "1")],
                    &[generated.check],
                    OptimizationLevel::None,
                )
                .expect("generated struct must compile and reflect for DXIL");
            assert_eq!(result.reflection.binding_element_strides[0], Some(28));
        }
    }

    #[test]
    fn matrix_after_float3_gets_storage_padding() {
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, goldy_derive::GpuType)]
        #[repr(C)]
        struct Camera {
            origin: [f32; 3],
            transform: [[f32; 4]; 4],
        }

        let packed = Camera::GPU_TYPE.packed().unwrap();
        assert_eq!(packed.fields[0].storage_offset, 0);
        assert_eq!(packed.fields[1].storage_offset, 16);
        assert_eq!(packed.stride, 80);
        let generated = Camera::GPU_TYPE.generate().unwrap();
        assert!(generated.source.contains("uint __goldy_pad0;"));
    }

    #[test]
    fn vertex_buffer_layout_uses_storage_offsets_not_host() {
        use crate::StructuredBufferElement;

        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, goldy_derive::GpuType)]
        #[repr(C)]
        struct MeshVertex {
            pos: [f32; 3],
            uv: [f32; 2],
            extra: u32,
        }

        assert_eq!(std::mem::size_of::<MeshVertex>(), 24);
        assert_eq!(MeshVertex::GPU_TYPE.fields[1].offset, 12);
        assert_eq!(MeshVertex::gpu_element_stride(), 28);

        let packed = MeshVertex::GPU_TYPE.packed().unwrap();
        assert_eq!(packed.stride, 28);
        assert_eq!(packed.fields[1].storage_offset, 16);

        let layout = MeshVertex::GPU_TYPE.vertex_buffer_layout().unwrap();
        assert_eq!(layout.stride, 28);
        assert_eq!(
            layout
                .attributes
                .iter()
                .map(|attribute| (attribute.location, attribute.offset))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 16), (2, 24)]
        );
    }
}
