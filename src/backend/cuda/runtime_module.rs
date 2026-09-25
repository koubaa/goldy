//! Runtime-side helper kernels for CUDA graphs.
//!
//! Compiles (via NVRTC) and loads a tiny kernel that reads a [`DispatchShape`]
//! and updates a device-updatable CUDA graph kernel node via
//! `cudaGraphKernelNodeSetGridDim` / `cudaGraphKernelNodeSetEnabled`, and a
//! small host→device copy kernel for graph-captured uploads.

use anyhow::{Context as _, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::path::PathBuf;
use std::sync::Arc;

/// Source for the graph-native indirect-dispatch updater.
pub(super) const APPLY_DISPATCH_SHAPE_SRC: &str = r#"
#include <cuda_device_runtime_api.h>

extern "C" __global__ void goldy_apply_dispatch_shape(
    const unsigned int* shape_xyz,
    cudaGraphDeviceNode_t* node_slot,
    unsigned int max_x,
    unsigned int max_y,
    unsigned int max_z,
    int* status_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    cudaGraphDeviceNode_t node = *node_slot;
    unsigned int x = shape_xyz[0];
    unsigned int y = shape_xyz[1];
    unsigned int z = shape_xyz[2];
    int status = 0;
    if (x == 0 || y == 0 || z == 0) {
        status = (int)cudaGraphKernelNodeSetEnabled(node, 0);
    } else if (x > max_x || y > max_y || z > max_z) {
        // Disable so a stale prior grid cannot run, then report oversized.
        status = (int)cudaGraphKernelNodeSetEnabled(node, 0);
        if (status == 0) {
            status = -1;
        }
    } else {
        int enable_status = (int)cudaGraphKernelNodeSetEnabled(node, 1);
        int grid_status = (int)cudaGraphKernelNodeSetGridDim(node, dim3(x, y, z));
        status = enable_status != 0 ? enable_status : grid_status;
    }
    if (status_out) {
        *status_out = status;
    }
}
"#;

pub(super) const APPLY_DISPATCH_SHAPE_FN: &str = "goldy_apply_dispatch_shape";

/// Oversized-grid diagnostic written by the updater (distinct from CUDA error codes).
#[allow(dead_code)] // mirrored in device source; kept for host-side status decoding
pub(super) const STATUS_OVERSIZED_GRID: i32 = -1;

/// Loaded updater module + entry point, pinned for the device lifetime.
pub(super) struct IndirectUpdater {
    pub module: Arc<CudaModule>,
    pub function: CudaFunction,
}

/// Resolve the CUDA toolkit `include` directory for NVRTC.
pub(super) fn cuda_include_path() -> Result<PathBuf> {
    let candidates = [
        std::env::var_os("CUDA_PATH")
            .map(PathBuf::from)
            .map(|p| p.join("include")),
        Some(PathBuf::from(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\include",
        )),
        Some(PathBuf::from(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include",
        )),
        Some(PathBuf::from("/usr/local/cuda/include")),
        Some(PathBuf::from("/usr/local/cuda-13.1/include")),
    ];
    for path in candidates.into_iter().flatten() {
        if path.join("cuda_device_runtime_api.h").is_file() {
            return Ok(path);
        }
    }
    anyhow::bail!(
        "CUDA: cannot find toolkit include directory (need cuda_device_runtime_api.h for \
         device-updatable graph updater). Set CUDA_PATH."
    )
}

/// Source for graph-captured small uploads from device-mapped pinned host memory.
///
/// A kernel node keeps the upload on the compute engine. An HtoD memcpy node runs on
/// a copy engine, and the handoff to the first kernel costs tens of microseconds.
pub(super) const COPY_FROM_HOST_SRC: &str = r#"
extern "C" __global__ void goldy_copy_from_host(
    const unsigned char* src,
    unsigned char* dst,
    unsigned int len
) {
    unsigned int i = threadIdx.x;
    if (((((unsigned long long)src) | ((unsigned long long)dst) | len) & 3ull) == 0) {
        const unsigned int* s = (const unsigned int*)src;
        unsigned int* d = (unsigned int*)dst;
        for (unsigned int w = i; w < len / 4; w += blockDim.x) {
            d[w] = s[w];
        }
    } else {
        for (unsigned int b = i; b < len; b += blockDim.x) {
            dst[b] = src[b];
        }
    }
}
"#;

pub(super) const COPY_FROM_HOST_FN: &str = "goldy_copy_from_host";

/// Largest graph-captured upload read by [`HostCopyKernel`]. Larger uploads keep the
/// memcpy node, where copy-engine bandwidth beats reading host memory from a kernel.
pub(super) const COPY_FROM_HOST_MAX_BYTES: usize = 4096;

/// Threads per [`HostCopyKernel`] launch (one block).
pub(super) const COPY_FROM_HOST_THREADS: u32 = 256;

/// Loaded host-copy module + entry point, pinned for the device lifetime.
pub(super) struct HostCopyKernel {
    pub module: Arc<CudaModule>,
    pub function: CudaFunction,
}

fn nvrtc_arch(compute_capability: (i32, i32)) -> &'static str {
    let (major, minor) = compute_capability;
    match (major, minor) {
        (7, 5) => "sm_75",
        (8, 0) => "sm_80",
        (8, 6) => "sm_86",
        (8, 9) => "sm_89",
        (9, 0) => "sm_90",
        (10, 0) => "sm_100",
        (12, 0) => "sm_120",
        _ => {
            // Leak a formatted arch string once per unseen CC (device init is rare).
            Box::leak(format!("sm_{major}{minor}").into_boxed_str())
        }
    }
}

/// Compile and load the graph-capture host-copy kernel for `ctx`.
pub(super) fn load_host_copy(ctx: &Arc<CudaContext>, compute_capability: (i32, i32)) -> Result<HostCopyKernel> {
    let arch = nvrtc_arch(compute_capability);
    let ptx = compile_ptx_with_opts(
        COPY_FROM_HOST_SRC,
        CompileOptions {
            arch: Some(arch),
            ..Default::default()
        },
    )
    .with_context(|| format!("CUDA: NVRTC failed compiling {COPY_FROM_HOST_FN} for {arch}"))?;
    let module = ctx.load_module(ptx).context("CUDA: load host-copy PTX module failed")?;
    let function = module
        .load_function(COPY_FROM_HOST_FN)
        .with_context(|| format!("CUDA: cuModuleGetFunction({COPY_FROM_HOST_FN}) failed"))?;
    Ok(HostCopyKernel { module, function })
}

/// Compile and load the indirect-dispatch updater for `ctx`.
pub(super) fn load_indirect_updater(ctx: &Arc<CudaContext>, compute_capability: (i32, i32)) -> Result<IndirectUpdater> {
    let include = cuda_include_path()?;
    let arch = nvrtc_arch(compute_capability);
    let ptx = compile_ptx_with_opts(
        APPLY_DISPATCH_SHAPE_SRC,
        CompileOptions {
            include_paths: vec![include.to_string_lossy().into_owned()],
            arch: Some(arch),
            ..Default::default()
        },
    )
    .with_context(|| {
        format!(
            "CUDA: NVRTC failed compiling goldy_apply_dispatch_shape for {arch} \
             (include={})",
            include.display()
        )
    })?;
    if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
        use std::io::Write;
        let dir = std::path::Path::new(&dump_dir);
        let _ = std::fs::create_dir_all(dir);
        let cu_path = dir.join("goldy_apply_dispatch_shape.cu");
        let ptx_path = dir.join("goldy_apply_dispatch_shape.ptx");
        if let Ok(mut file) = std::fs::File::create(&cu_path) {
            let _ = file.write_all(APPLY_DISPATCH_SHAPE_SRC.as_bytes());
            tracing::info!("Dumped CUDA shader to {}", cu_path.display());
        }
        if let Ok(mut file) = std::fs::File::create(&ptx_path) {
            let _ = file.write_all(ptx.to_src().as_bytes());
            tracing::info!("Dumped CUDA shader to {}", ptx_path.display());
        }
    }
    let module = ctx
        .load_module(ptx)
        .context("CUDA: load indirect updater PTX module failed")?;
    let function = module
        .load_function(APPLY_DISPATCH_SHAPE_FN)
        .context("CUDA: cuModuleGetFunction(goldy_apply_dispatch_shape) failed")?;
    Ok(IndirectUpdater { module, function })
}
