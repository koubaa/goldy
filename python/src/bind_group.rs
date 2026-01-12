//! Python wrappers for BindGroup and BindGroupLayout.

use crate::buffer::PyBuffer;
use crate::device::PyDevice;
use crate::error::IntoPyResult;
use pyo3::prelude::*;
use std::sync::Arc;

// =============================================================================
// ShaderStages
// =============================================================================

/// Shader stages that can access a binding.
#[pyclass(name = "ShaderStages", module = "goldy")]
#[derive(Clone, Copy)]
pub struct PyShaderStages {
    bits: u32,
}

#[pymethods]
impl PyShaderStages {
    /// Vertex shader stage.
    #[classattr]
    const VERTEX: PyShaderStages = PyShaderStages { bits: 1 << 0 };

    /// Fragment shader stage.
    #[classattr]
    const FRAGMENT: PyShaderStages = PyShaderStages { bits: 1 << 1 };

    /// Compute shader stage.
    #[classattr]
    const COMPUTE: PyShaderStages = PyShaderStages { bits: 1 << 2 };

    /// All shader stages (vertex, fragment, compute).
    #[classattr]
    const ALL: PyShaderStages = PyShaderStages { bits: 0x7 };

    fn __or__(&self, other: &PyShaderStages) -> PyShaderStages {
        PyShaderStages {
            bits: self.bits | other.bits,
        }
    }

    fn __repr__(&self) -> String {
        let mut parts = Vec::new();
        if self.bits & (1 << 0) != 0 {
            parts.push("VERTEX");
        }
        if self.bits & (1 << 1) != 0 {
            parts.push("FRAGMENT");
        }
        if self.bits & (1 << 2) != 0 {
            parts.push("COMPUTE");
        }
        format!("ShaderStages({})", parts.join(" | "))
    }
}

impl From<PyShaderStages> for goldy::ShaderStages {
    fn from(s: PyShaderStages) -> Self {
        goldy::ShaderStages(s.bits)
    }
}

// =============================================================================
// BindingType
// =============================================================================

/// Type of resource binding.
#[pyclass(name = "BindingType", module = "goldy")]
#[derive(Clone)]
pub struct PyBindingType {
    inner: goldy::BindingType,
}

#[pymethods]
impl PyBindingType {
    /// Uniform buffer binding.
    #[staticmethod]
    fn uniform_buffer() -> Self {
        PyBindingType {
            inner: goldy::BindingType::UniformBuffer,
        }
    }

    /// Storage buffer binding.
    #[staticmethod]
    #[pyo3(signature = (read_only=false))]
    fn storage_buffer(read_only: bool) -> Self {
        PyBindingType {
            inner: goldy::BindingType::StorageBuffer { read_only },
        }
    }

    /// Texture binding.
    #[staticmethod]
    fn texture() -> Self {
        PyBindingType {
            inner: goldy::BindingType::Texture,
        }
    }

    /// Sampler binding.
    #[staticmethod]
    fn sampler() -> Self {
        PyBindingType {
            inner: goldy::BindingType::Sampler,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            goldy::BindingType::UniformBuffer => "BindingType.uniform_buffer()".to_string(),
            goldy::BindingType::StorageBuffer { read_only } => {
                format!("BindingType.storage_buffer(read_only={})", read_only)
            }
            goldy::BindingType::Texture => "BindingType.texture()".to_string(),
            goldy::BindingType::Sampler => "BindingType.sampler()".to_string(),
            goldy::BindingType::StorageTexture => "BindingType.storage_texture()".to_string(),
        }
    }
}

// =============================================================================
// BindGroupLayoutBinding
// =============================================================================

/// Description of a binding in a bind group layout.
#[pyclass(name = "BindGroupLayoutBinding", module = "goldy")]
#[derive(Clone)]
pub struct PyBindGroupLayoutBinding {
    pub(crate) inner: goldy::BindGroupLayoutBinding,
}

#[pymethods]
impl PyBindGroupLayoutBinding {
    /// Create a new bind group layout binding.
    #[new]
    fn new(binding: u32, visibility: PyShaderStages, ty: PyBindingType) -> Self {
        PyBindGroupLayoutBinding {
            inner: goldy::BindGroupLayoutBinding {
                binding,
                visibility: visibility.into(),
                ty: ty.inner,
            },
        }
    }

    /// Create a uniform buffer binding visible to all graphics stages.
    #[staticmethod]
    fn uniform(binding: u32) -> Self {
        PyBindGroupLayoutBinding {
            inner: goldy::BindGroupLayoutBinding::uniform(binding),
        }
    }

    /// Create a storage buffer binding.
    #[staticmethod]
    #[pyo3(signature = (binding, read_only=false))]
    fn storage(binding: u32, read_only: bool) -> Self {
        PyBindGroupLayoutBinding {
            inner: goldy::BindGroupLayoutBinding::storage(binding, read_only),
        }
    }

    /// Create a storage buffer binding visible to compute stage.
    #[staticmethod]
    #[pyo3(signature = (binding, read_only=false))]
    fn storage_compute(binding: u32, read_only: bool) -> Self {
        PyBindGroupLayoutBinding {
            inner: goldy::BindGroupLayoutBinding {
                binding,
                visibility: goldy::ShaderStages::COMPUTE,
                ty: goldy::BindingType::StorageBuffer { read_only },
            },
        }
    }

    fn __repr__(&self) -> String {
        format!("BindGroupLayoutBinding(binding={})", self.inner.binding)
    }
}

// =============================================================================
// BindGroupLayout
// =============================================================================

/// A bind group layout defines the structure of a bind group.
#[pyclass(name = "BindGroupLayout", module = "goldy")]
pub struct PyBindGroupLayout {
    pub(crate) inner: Arc<goldy::BindGroupLayout>,
}

#[pymethods]
impl PyBindGroupLayout {
    /// Create a new bind group layout.
    #[new]
    fn new(device: &PyDevice, bindings: Vec<PyBindGroupLayoutBinding>) -> PyResult<Self> {
        let rust_bindings: Vec<_> = bindings.iter().map(|b| b.inner.clone()).collect();
        let layout = goldy::BindGroupLayout::new(&device.inner, &rust_bindings).into_py_result()?;
        Ok(PyBindGroupLayout {
            inner: Arc::new(layout),
        })
    }

    fn __repr__(&self) -> String {
        "BindGroupLayout()".to_string()
    }
}

// =============================================================================
// BufferBinding
// =============================================================================

/// Description of a buffer binding in a bind group.
#[pyclass(name = "BufferBinding", module = "goldy")]
#[derive(Clone)]
pub struct PyBufferBinding {
    pub(crate) binding: u32,
    pub(crate) buffer: Arc<goldy::Buffer>,
    pub(crate) offset: u64,
    pub(crate) size: Option<u64>,
}

#[pymethods]
impl PyBufferBinding {
    /// Create a buffer binding for the entire buffer.
    #[new]
    fn new(binding: u32, buffer: &PyBuffer) -> Self {
        PyBufferBinding {
            binding,
            buffer: Arc::clone(&buffer.inner),
            offset: 0,
            size: None,
        }
    }

    /// Create a buffer binding with offset and size.
    #[staticmethod]
    fn with_range(binding: u32, buffer: &PyBuffer, offset: u64, size: u64) -> Self {
        PyBufferBinding {
            binding,
            buffer: Arc::clone(&buffer.inner),
            offset,
            size: Some(size),
        }
    }

    fn __repr__(&self) -> String {
        format!("BufferBinding(binding={})", self.binding)
    }
}

// =============================================================================
// BindGroup
// =============================================================================

/// A bind group contains actual resource bindings matching a layout.
#[pyclass(name = "BindGroup", module = "goldy")]
pub struct PyBindGroup {
    pub(crate) inner: Arc<goldy::BindGroup>,
}

#[pymethods]
impl PyBindGroup {
    /// Create a new bind group from a layout and buffer bindings.
    #[new]
    fn new(
        device: &PyDevice,
        layout: &PyBindGroupLayout,
        bindings: Vec<PyBufferBinding>,
    ) -> PyResult<Self> {
        // We need to create temporary BufferBinding references
        // This is a bit awkward due to lifetime requirements
        let rust_bindings: Vec<goldy::BufferBinding> = bindings
            .iter()
            .map(|b| goldy::BufferBinding {
                binding: b.binding,
                buffer: &b.buffer,
                offset: b.offset,
                size: b.size,
            })
            .collect();

        let bind_group =
            goldy::BindGroup::new(&device.inner, &layout.inner, &rust_bindings).into_py_result()?;

        Ok(PyBindGroup {
            inner: Arc::new(bind_group),
        })
    }

    fn __repr__(&self) -> String {
        "BindGroup()".to_string()
    }
}
