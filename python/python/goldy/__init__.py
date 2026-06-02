"""Goldy GPU library for Python.

A modern GPU library targeting Vulkan 1.4+, DX12, and Metal.

Example:
    >>> import goldy
    >>> import numpy as np
    >>> 
    >>> instance = goldy.Instance()
    >>> device = instance.request_adapter().request_device()
    >>> 
    >>> # Create a render target
    >>> target = goldy.RenderTarget(device, 800, 600, goldy.TextureFormat.RGBA8_UNORM)
    >>> 
    >>> # Render
    >>> encoder = goldy.CommandEncoder()
    >>> with encoder.begin_render_pass() as rp:
    ...     rp.clear(goldy.Color(0.1, 0.1, 0.2, 1.0))
    >>> target.render(encoder)
    >>> 
    >>> # Read pixels as numpy array
    >>> pixels = target.read_to_cpu()
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
    DataAccess,
    SpatialAccess,
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
    CommandEncoder,
    RenderPass,
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
    "DataAccess",
    "SpatialAccess",
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
    "CommandEncoder",
    "RenderPass",
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
