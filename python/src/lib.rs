//! Goldy Python bindings.
//!
//! This crate provides Python bindings for the Goldy GPU library using PyO3.

mod buffer;
mod compute;
mod device;
mod encoder;
mod error;
mod instance;
mod pipeline;
mod render_target;
mod shader;
mod surface;
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
    m.add_class::<types::PyDataAccess>()?;
    m.add_class::<types::PySpatialAccess>()?;
    m.add_class::<types::PyVertexFormat>()?;
    m.add_class::<types::PyPrimitiveTopology>()?;
    m.add_class::<types::PyIndexFormat>()?;
    m.add_class::<types::PyDepthFormat>()?;
    m.add_class::<types::PyCompareFunction>()?;
    m.add_class::<types::PyColor>()?;
    m.add_class::<types::PyVertexAttribute>()?;
    m.add_class::<types::PyVertexBufferLayout>()?;
    m.add_class::<types::PyDepthStencilState>()?;

    // Register core classes
    m.add_class::<instance::PyInstance>()?;
    m.add_class::<instance::PyAdapter>()?;
    m.add_class::<device::PyDevice>()?;
    m.add_class::<buffer::PyBuffer>()?;
    m.add_class::<shader::PyShaderModule>()?;
    m.add_class::<pipeline::PyRenderPipeline>()?;
    m.add_class::<pipeline::PyRenderPipelineDesc>()?;
    m.add_class::<render_target::PyRenderTarget>()?;
    m.add_class::<encoder::PyCommandEncoder>()?;
    m.add_class::<encoder::PyRenderPass>()?;

    // Shader builtins
    m.add_class::<shader::PyBuiltins>()?;

    // Compute
    m.add_class::<compute::PyComputePipeline>()?;
    m.add_class::<compute::PyComputeEncoder>()?;
    m.add_class::<compute::PyComputePass>()?;

    // Surface (windowed rendering)
    m.add_class::<surface::PySurface>()?;
    m.add_class::<surface::PySurfaceFrame>()?;

    Ok(())
}
