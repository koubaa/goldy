//! Python wrapper for ShaderModule.

use crate::device::PyDevice;
use crate::error::IntoPyResult;
use pyo3::prelude::*;
use std::sync::Arc;

/// A compiled shader module.
///
/// Shaders are written in Slang (HLSL-like syntax) and compiled at runtime.
#[pyclass(name = "ShaderModule", module = "goldy")]
pub struct PyShaderModule {
    pub(crate) inner: Arc<goldy::ShaderModule>,
}

#[pymethods]
impl PyShaderModule {
    /// Create a shader module from Slang source.
    ///
    /// The source is compiled using Slang and can import any registered
    /// shader libraries (including the built-in `goldy_exp` library).
    ///
    /// Args:
    ///     device: The GPU device.
    ///     source: Slang shader source code.
    ///
    /// Returns:
    ///     A new ShaderModule instance.
    ///
    /// Raises:
    ///     GoldyError: If shader compilation fails.
    ///
    /// Example:
    ///     >>> shader = goldy.ShaderModule.from_slang(device, '''
    ///     ...     import goldy_exp;
    ///     ...
    ///     ...     [shader("vertex")]
    ///     ...     FullscreenVarying vs_main(FullscreenVertex input) {
    ///     ...         return vs_fullscreen(input);
    ///     ...     }
    ///     ...
    ///     ...     [shader("fragment")]
    ///     ...     float4 fs_main(FullscreenVarying input) : SV_Target {
    ///     ...         return float4(rainbow(input.uv.x), 1.0);
    ///     ...     }
    ///     ... ''')
    #[staticmethod]
    fn from_slang(device: &PyDevice, source: &str) -> PyResult<Self> {
        let shader = goldy::ShaderModule::from_slang(&device.inner, source).into_py_result()?;
        Ok(PyShaderModule {
            inner: Arc::new(shader),
        })
    }

    /// Create a shader module with additional search paths.
    ///
    /// Args:
    ///     device: The GPU device.
    ///     source: Slang shader source code.
    ///     extra_paths: Additional filesystem paths to search for imports.
    ///
    /// Returns:
    ///     A new ShaderModule instance.
    #[staticmethod]
    fn from_slang_with_paths(device: &PyDevice, source: &str, extra_paths: Vec<String>) -> PyResult<Self> {
        let path_refs: Vec<&str> = extra_paths.iter().map(|s| s.as_str()).collect();
        let shader = goldy::ShaderModule::from_slang_with_paths(&device.inner, source, &path_refs).into_py_result()?;
        Ok(PyShaderModule {
            inner: Arc::new(shader),
        })
    }

    fn __repr__(&self) -> String {
        "ShaderModule()".to_string()
    }
}

/// Built-in shader source code.
#[pyclass(name = "Builtins", module = "goldy")]
pub struct PyBuiltins;

#[pymethods]
impl PyBuiltins {
    /// Simple 2D vertex + fragment shader for colored vertices.
    #[classattr]
    const VERTEX_COLOR_2D: &'static str = goldy::shader::builtins::VERTEX_COLOR_2D;

    /// Simple solid color fragment shader.
    #[classattr]
    const SOLID_COLOR: &'static str = goldy::shader::builtins::SOLID_COLOR;
}
