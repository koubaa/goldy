//! Device-side updater kernel for CUDA graph indirect dispatch.
//!
//! Compiles (via NVRTC) and loads a tiny kernel that reads a [`DispatchShape`]
//! and updates a device-updatable CUDA graph kernel node via
//! `cudaGraphKernelNodeSetGridDim` / `cudaGraphKernelNodeSetEnabled`.

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

/// Compile and load the indirect-dispatch updater for `ctx`.
pub(super) fn load_indirect_updater(
    ctx: &Arc<CudaContext>,
    compute_capability: (i32, i32),
) -> Result<IndirectUpdater> {
    let include = cuda_include_path()?;
    let (major, minor) = compute_capability;
    let arch: &'static str = match (major, minor) {
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
    };
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
    let module = ctx
        .load_module(ptx)
        .context("CUDA: load indirect updater PTX module failed")?;
    let function = module
        .load_function(APPLY_DISPATCH_SHAPE_FN)
        .context("CUDA: cuModuleGetFunction(goldy_apply_dispatch_shape) failed")?;
    Ok(IndirectUpdater { module, function })
}
