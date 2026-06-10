//! Goldy Python bindings.
//!
//! This crate provides Python bindings for the Goldy GPU library using PyO3.

// PyO3 0.28+: automatic `FromPyObject` for `#[pyclass]` + `Clone` is deprecated pending
// explicit `from_py_object` / `skip_from_py_object`; allow until bindings are migrated.
#![allow(deprecated)]
// Enum variant names mirror Python / Goldy public API (e.g. SCATTERED, VULKAN).
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::wrong_self_convention)]

mod bytes_util;
mod compute;
mod device;
mod error;
mod instance;
mod parcel;
mod pipeline;
mod render_target;
mod retained_pool;
mod shader;
mod surface;
mod task_graph;
mod types;

use pyo3::prelude::*;

/// Goldy GPU library for Python.
///
/// A modern GPU library targeting Vulkan 1.4+, DX12, and Metal.
#[pymodule]
fn _goldy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the custom exception
    m.add("GoldyError", m.py().get_type::<error::GoldyError>())?;

    // Register types - enums
    m.add_class::<types::PyDeviceType>()?;
    m.add_class::<types::PyBackendType>()?;
    m.add_class::<types::PyTextureFormat>()?;
    m.add_class::<types::PyBufferKind>()?;
    m.add_class::<types::PyTextureKind>()?;
    m.add_class::<types::PyVertexFormat>()?;
    m.add_class::<types::PyPrimitiveTopology>()?;
    m.add_class::<types::PyIndexFormat>()?;
    m.add_class::<types::PyDepthFormat>()?;
    m.add_class::<types::PyCompareFunction>()?;
    m.add_class::<types::PyNodeAccess>()?;
    m.add_class::<types::PyResourceAccess>()?;
    m.add_class::<types::PyColor>()?;
    m.add_class::<types::PyVertexAttribute>()?;
    m.add_class::<types::PyVertexBufferLayout>()?;
    m.add_class::<types::PyDepthStencilState>()?;

    // Register core classes
    m.add_class::<instance::PyInstance>()?;
    m.add_class::<instance::PyAdapter>()?;
    m.add_class::<device::PyDevice>()?;
    m.add_class::<parcel::PyParcel>()?;
    m.add_class::<retained_pool::PyRetainedPool>()?;
    m.add_class::<retained_pool::PyMosaicBuilder>()?;
    m.add_class::<shader::PyShaderModule>()?;
    m.add_class::<pipeline::PyRenderPipeline>()?;
    m.add_class::<pipeline::PyRenderPipelineDesc>()?;
    m.add_class::<render_target::PyRenderTarget>()?;
    m.add_class::<task_graph::PyTaskGraph>()?;
    m.add_class::<task_graph::PyRenderPass>()?;
    m.add_class::<task_graph::PySwapchainOutput>()?;
    m.add_class::<task_graph::PyComputeNode>()?;

    // Shader builtins
    m.add_class::<shader::PyBuiltins>()?;

    // Compute
    m.add_class::<compute::PyComputePipeline>()?;

    // Surface (windowed rendering)
    m.add_class::<surface::PySurface>()?;
    m.add_class::<surface::PySurfaceFrame>()?;

    Ok(())
}
