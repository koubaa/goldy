//! Debug-only CPU execution of Slang compute kernels via host-callable JIT.
//!
//! Compile with [`compile`] / [`compile_kernel`] and dispatch over host slices,
//! or select the compute-only device backend with `GOLDY_BACKEND=cpu`.
//!
//! `GOLDY_CPU_SHADERS=1` remains a documented gate for the standalone APIs.
//! GPU backends ignore it.
//!
//! # What lowers
//!
//! | Goldy / Slang type | CPU ABI |
//! |---|---|
//! | `BufRO<T>` / `Scattered<T>` of scalar `uint`/`int`/`float`/`bool` | `{ T* data; size_t count }` |
//! | Scalar `uint`/`int`/`float`/`bool` | 4-byte word (natural alignment) |
//! | `ThreadId` / `GroupThreadId` / `GroupId` | `SV_*` system values |
//! | `goldy_buf_len` | `GetDimensions` on the CPU structured buffer |
//!
//! # What does not lower (yet)
//!
//! Broadcast / `gpu::Uniform` structs, `ByteAddress`, `Interpolated` textures,
//! `DirectSpatial` storage images, samplers, `Interlocked` atomics beyond what
//! the Slang CPU prelude provides, and any `[goldy_vertex]` / `[goldy_fragment]`
//! stage. Fine rasterization stays GPU-only.

use anyhow::{Context, Result};
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;

use crate::kernel::KernelDef;
use crate::slang::compiler::SlangCompiler;
use crate::slang::ffi::{shared_library_find_symbol, shared_library_release, ISlangSharedLibrary, SlangStage};
use crate::slang::loader::SlangLibrary;
use crate::slang::virtual_main::{extract_cuda_compute_launch_layout, transform_virtual_main_cpu, CudaLaunchArgKind};
use crate::types::OptimizationLevel;
use goldy_shader_ir::ParamCategory;

/// True when `GOLDY_CPU_SHADERS` is `1` / `true` / `yes`.
///
/// Reserved for a future Device debug option. The compile APIs in this module
/// do not require the variable; GPU backends ignore it.
pub fn enabled() -> bool {
    std::env::var("GOLDY_CPU_SHADERS")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Slang CPU prelude `uint3`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuUInt3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Slang `ComputeVaryingInput` — inclusive start group, exclusive end group.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ComputeVaryingInput {
    pub start_group_id: CpuUInt3,
    pub end_group_id: CpuUInt3,
}

/// CPU structured-buffer view (`T* data` + `size_t count`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CpuBufferView {
    pub data: *mut c_void,
    pub count: usize,
}

type ComputeFunc = unsafe extern "C" fn(*const ComputeVaryingInput, *mut c_void, *mut c_void);

/// A host-callable compute kernel and the JIT library that owns its code.
pub struct CpuComputeKernel {
    _slang: Arc<SlangLibrary>,
    shared_library: *mut ISlangSharedLibrary,
    func: ComputeFunc,
    entry: String,
    workgroup_size: [u32; 3],
    layout: Vec<CpuParamSlot>,
}

unsafe impl Send for CpuComputeKernel {}
unsafe impl Sync for CpuComputeKernel {}

impl Drop for CpuComputeKernel {
    fn drop(&mut self) {
        if !self.shared_library.is_null() {
            unsafe { shared_library_release(self.shared_library) };
            self.shared_library = ptr::null_mut();
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CpuParamSlot {
    Buffer { stride: u32 },
    Scalar,
}

/// Host buffer view used by the CPU device backend (exclusive `&mut` on the backend).
pub(crate) struct CpuHostBufferView {
    pub data: *mut u8,
    pub len: usize,
    pub stride: u32,
}

/// One host argument for [`CpuComputeKernel::dispatch`].
pub enum CpuBinding<'a> {
    /// Structured buffer: element count is `bytes.len() / stride`.
    Buffer { bytes: &'a mut [u8], stride: u32 },
    /// Packed 32-bit scalar (use [`CpuBinding::f32`] / [`CpuBinding::u32`]).
    Scalar(u32),
}

impl<'a> CpuBinding<'a> {
    pub fn u32s(data: &'a mut [u32]) -> Self {
        Self::Buffer {
            bytes: bytemuck::cast_slice_mut(data),
            stride: 4,
        }
    }

    pub fn f32s(data: &'a mut [f32]) -> Self {
        Self::Buffer {
            bytes: bytemuck::cast_slice_mut(data),
            stride: 4,
        }
    }

    pub fn i32s(data: &'a mut [i32]) -> Self {
        Self::Buffer {
            bytes: bytemuck::cast_slice_mut(data),
            stride: 4,
        }
    }

    pub fn u32(v: u32) -> Self {
        Self::Scalar(v)
    }

    pub fn i32(v: i32) -> Self {
        Self::Scalar(v as u32)
    }

    pub fn f32(v: f32) -> Self {
        Self::Scalar(v.to_bits())
    }

    pub fn bool(v: bool) -> Self {
        Self::Scalar(u32::from(v))
    }
}

impl CpuComputeKernel {
    pub fn entry(&self) -> &str {
        &self.entry
    }

    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// Dispatch `groups` thread-groups (serial host execution of the Slang CPU prelude).
    pub fn dispatch(&self, groups: [u32; 3], bindings: &mut [CpuBinding<'_>]) -> Result<()> {
        if bindings.len() != self.layout.len() {
            anyhow::bail!(
                "CPU kernel `{}`: expected {} bindings, got {}",
                self.entry,
                self.layout.len(),
                bindings.len()
            );
        }

        let mut payload = Vec::<u8>::new();
        for (i, (slot, bind)) in self.layout.iter().zip(bindings.iter_mut()).enumerate() {
            match (slot, bind) {
                (CpuParamSlot::Buffer { stride: want }, CpuBinding::Buffer { bytes, stride }) => {
                    if *stride != *want {
                        anyhow::bail!(
                            "CPU kernel `{}` binding {i}: stride {stride} != expected {want}",
                            self.entry
                        );
                    }
                    if *stride == 0 || bytes.len() % *stride as usize != 0 {
                        anyhow::bail!(
                            "CPU kernel `{}` binding {i}: byte length {} is not a multiple of stride {stride}",
                            self.entry,
                            bytes.len()
                        );
                    }
                    let count = bytes.len() / *stride as usize;
                    pad_to_align(&mut payload, align_of::<CpuBufferView>());
                    let view = CpuBufferView {
                        data: bytes.as_mut_ptr() as *mut c_void,
                        count,
                    };
                    let view_bytes = unsafe {
                        std::slice::from_raw_parts(
                            (&view as *const CpuBufferView) as *const u8,
                            std::mem::size_of::<CpuBufferView>(),
                        )
                    };
                    payload.extend_from_slice(view_bytes);
                }
                (CpuParamSlot::Scalar, CpuBinding::Scalar(bits)) => {
                    pad_to_align(&mut payload, 4);
                    payload.extend_from_slice(&bits.to_le_bytes());
                }
                _ => anyhow::bail!("CPU kernel `{}` binding {i}: category mismatch", self.entry),
            }
        }

        let varying = ComputeVaryingInput {
            start_group_id: CpuUInt3 { x: 0, y: 0, z: 0 },
            end_group_id: CpuUInt3 {
                x: groups[0].max(1),
                y: groups[1].max(1),
                z: groups[2].max(1),
            },
        };

        let params_ptr = if payload.is_empty() {
            ptr::null_mut()
        } else {
            payload.as_mut_ptr() as *mut c_void
        };

        unsafe {
            (self.func)(&varying, params_ptr, ptr::null_mut());
        }
        Ok(())
    }

    /// 1D dispatch covering `n` threads with this kernel's workgroup size.
    pub fn dispatch_1d(&self, n: u32, bindings: &mut [CpuBinding<'_>]) -> Result<()> {
        let wg = self.workgroup_size[0].max(1);
        let groups = n.div_ceil(wg);
        self.dispatch([groups, 1, 1], bindings)
    }

    pub(crate) fn layout(&self) -> &[CpuParamSlot] {
        &self.layout
    }

    /// Dispatch using bindless-order host views + packed scalar words.
    ///
    /// `buffers` is one entry per [`CpuParamSlot::Buffer`] in layout order (or, when
    /// the layout is empty, one entry per bound resource index).
    pub(crate) fn dispatch_host(&self, groups: [u32; 3], buffers: &[CpuHostBufferView], scalars: &[u32]) -> Result<()> {
        // SAFETY: the CPU backend uniquely borrows parcel storage for this call.
        let mut bindings = unsafe { self.bindings_from_host(buffers, scalars)? };
        self.dispatch(groups, &mut bindings)
    }

    unsafe fn bindings_from_host<'a>(
        &self,
        buffers: &[CpuHostBufferView],
        scalars: &[u32],
    ) -> Result<Vec<CpuBinding<'a>>> {
        let mut bindings = Vec::new();
        if self.layout.is_empty() {
            if !scalars.is_empty() {
                anyhow::bail!(
                    "CPU kernel `{}`: scalar user params require a [goldy_compute] entry; got {}",
                    self.entry,
                    scalars.len()
                );
            }
            for (i, view) in buffers.iter().enumerate() {
                bindings.push(host_view_to_binding(view, &self.entry, i)?);
            }
            return Ok(bindings);
        }

        let mut buf_i = 0usize;
        let mut sc_i = 0usize;
        for slot in &self.layout {
            match slot {
                CpuParamSlot::Buffer { stride: want } => {
                    let view = buffers.get(buf_i).with_context(|| {
                        format!("CPU kernel `{}`: missing host buffer for binding {buf_i}", self.entry)
                    })?;
                    if view.stride != *want {
                        anyhow::bail!(
                            "CPU kernel `{}` binding {buf_i}: stride {} != expected {want}",
                            self.entry,
                            view.stride
                        );
                    }
                    bindings.push(host_view_to_binding(view, &self.entry, buf_i)?);
                    buf_i += 1;
                }
                CpuParamSlot::Scalar => {
                    let bits = scalars
                        .get(sc_i)
                        .copied()
                        .with_context(|| format!("CPU kernel `{}`: missing scalar word {sc_i}", self.entry))?;
                    bindings.push(CpuBinding::Scalar(bits));
                    sc_i += 1;
                }
            }
        }
        if buf_i != buffers.len() {
            anyhow::bail!(
                "CPU kernel `{}`: expected {buf_i} buffer bindings, got {}",
                self.entry,
                buffers.len()
            );
        }
        if sc_i != scalars.len() {
            anyhow::bail!(
                "CPU kernel `{}`: expected {sc_i} scalar words, got {}",
                self.entry,
                scalars.len()
            );
        }
        Ok(bindings)
    }
}

unsafe fn host_view_to_binding<'a>(view: &CpuHostBufferView, entry: &str, index: usize) -> Result<CpuBinding<'a>> {
    if view.stride == 0 || !view.len.is_multiple_of(view.stride as usize) {
        anyhow::bail!(
            "CPU kernel `{entry}` binding {index}: byte length {} is not a multiple of stride {}",
            view.len,
            view.stride
        );
    }
    let bytes = if view.len == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(view.data, view.len)
    };
    Ok(CpuBinding::Buffer {
        bytes,
        stride: view.stride,
    })
}

fn pad_to_align(buf: &mut Vec<u8>, align: usize) {
    let rem = buf.len() % align;
    if rem != 0 {
        buf.resize(buf.len() + (align - rem), 0);
    }
}

fn layout_from_kernel_def(def: &KernelDef) -> Result<Vec<CpuParamSlot>> {
    let mut layout = Vec::new();
    for p in &def.params {
        match p.category {
            ParamCategory::BufferRead | ParamCategory::BufferReadWrite | ParamCategory::BufferWrite => {
                let stride = p.stride_bytes.unwrap_or(4);
                layout.push(CpuParamSlot::Buffer { stride });
            }
            ParamCategory::Scalar => {
                if p.scalar.is_none() {
                    anyhow::bail!("CPU kernel: scalar param `{}` missing ScalarType", p.name);
                }
                layout.push(CpuParamSlot::Scalar);
            }
            ParamCategory::Uniform => {
                anyhow::bail!(
                    "CPU host-callable: uniform/broadcast param `{}` does not lower yet",
                    p.name
                );
            }
        }
    }
    Ok(layout)
}

/// Compile `[goldy_compute]` or raw `[shader("compute")]` Slang to a CPU kernel.
pub fn compile(
    compiler: &SlangCompiler,
    source: &str,
    entry: &str,
    workgroup_size: [u32; 3],
    search_paths: &[&str],
) -> Result<CpuComputeKernel> {
    let def = crate::slang::try_kernel_def_from_source(source);
    let layout = if let Some(ref d) = def {
        layout_from_kernel_def(d)?
    } else {
        Vec::new()
    };
    compile_with_layout(
        compiler,
        source,
        entry,
        workgroup_size,
        search_paths,
        &[],
        OptimizationLevel::None,
        layout,
    )
}

/// Compile a structured [`KernelDef`] (from `#[goldy::compute]` or parsed Slang).
pub fn compile_kernel(compiler: &SlangCompiler, def: &KernelDef, search_paths: &[&str]) -> Result<CpuComputeKernel> {
    let layout = layout_from_kernel_def(def)?;
    compile_with_layout(
        compiler,
        &def.source.canonical_slang,
        &def.entry,
        def.workgroup_size,
        search_paths,
        &[],
        OptimizationLevel::None,
        layout,
    )
}

/// Compile `[goldy_compute]` (or raw compute) source for the CPU device backend.
pub(crate) fn compile_shader(
    compiler: &SlangCompiler,
    source: &str,
    search_paths: &[&str],
    defines: &[(&str, &str)],
    optimization_level: OptimizationLevel,
) -> Result<CpuComputeKernel> {
    let launch = extract_cuda_compute_launch_layout(source, defines).map_err(|e| anyhow::anyhow!("{e}"))?;
    for kind in &launch {
        match kind {
            CudaLaunchArgKind::Buffer | CudaLaunchArgKind::Scalar => {}
            CudaLaunchArgKind::SampledTexture { element } => {
                anyhow::bail!("CPU backend is compute-only: Interpolated<{element}> is not supported")
            }
            CudaLaunchArgKind::StorageTexture { element } => {
                anyhow::bail!("CPU backend is compute-only: DirectSpatial<{element}> is not supported")
            }
            CudaLaunchArgKind::Sampler => {
                anyhow::bail!("CPU backend is compute-only: samplers are not supported")
            }
        }
    }

    let def = crate::slang::try_kernel_def_from_source(source);
    let layout = if let Some(ref d) = def {
        layout_from_kernel_def(d)?
    } else {
        launch
            .iter()
            .map(|kind| match kind {
                CudaLaunchArgKind::Buffer => CpuParamSlot::Buffer { stride: 4 },
                CudaLaunchArgKind::Scalar => CpuParamSlot::Scalar,
                _ => unreachable!(),
            })
            .collect()
    };
    let entry = def.as_ref().map(|d| d.entry.as_str()).unwrap_or("cs_main");
    let workgroup_size = def
        .as_ref()
        .map(|d| d.workgroup_size)
        .or_else(|| crate::slang::parse_numthreads(source))
        .unwrap_or([1, 1, 1]);
    compile_with_layout(
        compiler,
        source,
        entry,
        workgroup_size,
        search_paths,
        defines,
        optimization_level,
        layout,
    )
}

#[allow(clippy::too_many_arguments)]
fn compile_with_layout(
    compiler: &SlangCompiler,
    source: &str,
    entry: &str,
    workgroup_size: [u32; 3],
    search_paths: &[&str],
    extra_defines: &[(&str, &str)],
    optimization_level: OptimizationLevel,
    layout: Vec<CpuParamSlot>,
) -> Result<CpuComputeKernel> {
    let transformed = transform_virtual_main_cpu(source).map_err(|e| anyhow::anyhow!(e))?;
    let (lib, slang) = compiler
        .compile_host_callable_library(
            &transformed,
            &[(entry, SlangStage::Compute)],
            search_paths,
            extra_defines,
            optimization_level,
        )
        .with_context(|| format!("host-callable compile of `{entry}`"))?;
    let name = CString::new(entry).context("entry point name contains NUL")?;
    let symbol = unsafe { shared_library_find_symbol(lib, name.as_ptr()) };
    if symbol.is_null() {
        unsafe { shared_library_release(lib) };
        anyhow::bail!("host-callable library has no symbol `{entry}`");
    }

    Ok(CpuComputeKernel {
        _slang: slang,
        shared_library: lib,
        func: unsafe { std::mem::transmute::<*mut c_void, ComputeFunc>(symbol) },
        entry: entry.to_string(),
        workgroup_size,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use goldy_shader_ir::{BuiltinMask, ElementType, KernelParam, ScalarType, SourceMap};

    fn shader_search_path() -> String {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn host_callable_trivial_empty_kernel() {
        let compiler = SlangCompiler::new().expect("Slang");
        let src = r#"
            [shader("compute")]
            [numthreads(1, 1, 1)]
            void cs_main() {}
        "#;
        let kernel = compile(&compiler, src, "cs_main", [1, 1, 1], &[]).expect("compile empty CPU kernel");
        kernel.dispatch([1, 1, 1], &mut []).expect("dispatch empty");
    }

    #[test]
    fn cpu_transform_rejects_broadcast() {
        let src = r#"
            import goldy_exp;
            struct Params { uint x; };
            [goldy_compute]
            [numthreads(1,1,1)]
            void cs_main(Params cfg, Scattered<uint> data, ThreadId id) {
                data[id.x] = cfg.x;
            }
        "#;
        let err = transform_virtual_main_cpu(src).unwrap_err();
        assert!(err.contains("broadcast"), "{err}");
    }

    #[test]
    fn host_callable_double_u32() {
        let compiler = SlangCompiler::new().expect("Slang");
        let path = shader_search_path();
        let src = r#"
            import goldy_exp;

            [goldy_compute]
            [numthreads(64, 1, 1)]
            void cs_main(Scattered<uint> data, ThreadId id) {
                if (id.x < goldy_buf_len(data)) {
                    data[id.x] = data[id.x] * 2u;
                }
            }
        "#;
        let kernel = compile(&compiler, src, "cs_main", [64, 1, 1], &[&path]).expect("compile CPU kernel");
        let mut data: Vec<u32> = (0..64).collect();
        kernel
            .dispatch_1d(64, &mut [CpuBinding::u32s(&mut data)])
            .expect("dispatch");
        for i in 0..64u32 {
            assert_eq!(data[i as usize], i * 2, "index {i}");
        }
    }

    #[test]
    fn host_callable_saxpy_from_kernel_def() {
        let compiler = SlangCompiler::new().expect("Slang");
        let path = shader_search_path();
        let src = r#"
            import goldy_exp;

            [goldy_compute]
            [numthreads(64, 1, 1)]
            void cs_main(BufRO<float> x, Scattered<float> y, float a, ThreadId _goldy_gid) {
                uint i = _goldy_gid.x;
                if (i < goldy_buf_len(y)) {
                    y[i] = a * x[i] + y[i];
                }
            }
        "#;
        let def = KernelDef::new(
            src,
            "cs_main",
            [64, 1, 1],
            vec![
                KernelParam::buffer_read("x", ElementType::F32),
                KernelParam::buffer_read_write("y", ElementType::F32),
                KernelParam::scalar_param("a", ScalarType::F32),
            ],
            BuiltinMask {
                global_id: true,
                ..BuiltinMask::NONE
            },
            SourceMap::default(),
        );
        let kernel = compile_kernel(&compiler, &def, &[&path]).expect("compile saxpy");
        let n = 256usize;
        let a = 2.0f32;
        let mut x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();
        let expected: Vec<f32> = (0..n).map(|i| a * (i as f32) + (i * 3) as f32).collect();
        kernel
            .dispatch_1d(
                n as u32,
                &mut [CpuBinding::f32s(&mut x), CpuBinding::f32s(&mut y), CpuBinding::f32(a)],
            )
            .expect("dispatch saxpy");
        for i in 0..n {
            assert!(
                (y[i] - expected[i]).abs() < 1e-5,
                "index {i}: {} vs {}",
                y[i],
                expected[i]
            );
        }
    }
}
