"""Goldy GPU library for Python.

A modern GPU library targeting Vulkan 1.4+, DX12, and Metal.

Example:
    >>> import goldy
    >>> import numpy as np
    >>> 
    >>> instance = goldy.Instance()
    >>> device = instance.request_adapter().request_device()
    >>> 
    >>> # Compute example (graphics uses TaskGraph in Rust; see goldy/examples/)
    >>> pipeline = goldy.ComputePipeline(device, shader)
    >>> encoder = goldy.ComputeEncoder()
    >>> with encoder.begin_compute_pass() as cp:
    ...     cp.set_pipeline(pipeline)
    ...     cp.dispatch(1, 1, 1)
    >>> encoder.execute(device)
"""

import os as _os
from pathlib import Path as _Path

# Set up GOLDY_SLANG_PATH to point to bundled Slang libraries
# This must happen BEFORE importing the native module
if "GOLDY_SLANG_PATH" not in _os.environ:
    _package_dir = _Path(__file__).parent
    # Determine library name based on platform
    import sys as _sys
    if _sys.platform == "win32":
        _slang_lib = "slang-compiler.dll"
    elif _sys.platform == "darwin":
        _slang_lib = "libslang-compiler.dylib"
    else:
        _slang_lib = "libslang-compiler.so"
    
    _slang_path = _package_dir / _slang_lib
    if _slang_path.exists():
        _os.environ["GOLDY_SLANG_PATH"] = str(_slang_path)

from goldy._goldy import (
    # Exception
    GoldyError,
    # Enums
    DeviceType,
    BackendType,
    TextureFormat,
    BufferKind,
    TextureKind,
    VertexFormat,
    PrimitiveTopology,
    IndexFormat,
    DepthFormat,
    CompareFunction,
    # Types
    Color,
    VertexAttribute,
    VertexBufferLayout,
    DepthStencilState,
    # Core classes
    Instance,
    Adapter,
    Device,
    Buffer,
    ShaderModule,
    RenderPipeline,
    RenderPipelineDesc,
    RenderTarget,
    # Shader builtins
    Builtins,
    # Compute
    ComputePipeline,
    ComputeEncoder,
    ComputePass,
    # Surface (windowed rendering)
    Surface,
    SurfaceFrame,
)

__all__ = [
    # Exception
    "GoldyError",
    # Enums
    "DeviceType",
    "BackendType",
    "TextureFormat",
    "BufferKind",
    "TextureKind",
    "VertexFormat",
    "PrimitiveTopology",
    "IndexFormat",
    "DepthFormat",
    "CompareFunction",
    # Types
    "Color",
    "VertexAttribute",
    "VertexBufferLayout",
    "DepthStencilState",
    # Core classes
    "Instance",
    "Adapter",
    "Device",
    "Buffer",
    "ShaderModule",
    "RenderPipeline",
    "RenderPipelineDesc",
    "RenderTarget",
    # Shader builtins
    "Builtins",
    # Compute
    "ComputePipeline",
    "ComputeEncoder",
    "ComputePass",
    # Surface (windowed rendering)
    "Surface",
    "SurfaceFrame",
]

__version__ = "0.1.0"
