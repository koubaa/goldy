"""Goldy GPU library for Python.

A modern GPU library targeting Vulkan 1.4+, DX12, and Metal.

Example:
    >>> import goldy
    >>> import numpy as np
    >>> 
    >>> instance = goldy.Instance()
    >>> device = instance.request_adapter().request_device()
    >>> 
    >>> # Graphics via Scheme (headless)
    >>> ctx = goldy.Context(device)
    >>> scheme = goldy.Scheme(ctx)
    >>> rt = scheme.lease_render_target(800, 600, goldy.TextureFormat.RGBA8_UNORM)
    >>> with scheme.render_pass("clear", rt) as rp:
    ...     rp.clear(goldy.Color.CORNFLOWER_BLUE)
    >>> submission = scheme.submit()
"""

import os as _os
from pathlib import Path as _Path

# PyPI wheels may ship slang-compiler next to this package; prefer that over cache.
# Editable dev installs embed Slang at compile time — this block is a no-op unless
# build-slang.py was run for a release wheel layout.
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
    Context,
    Buffer,
    Parcel,
    RetainedPool,
    RecordBuilder,
    ShaderModule,
    RenderPipeline,
    RenderPipelineDesc,
    RenderTarget,
    Scheme,
    SchemeComputeNode,
    SchemeRenderPass,
    SchemeRenderTargetLease,
    SchemeSubmission,
    ReadGrant,
    PresentLease,
    PresentGrant,
    SwapchainPool,
    NodeAccess,
    ResourceAccess,
    write_to_parcel,
    # Shader builtins
    Builtins,
    # Compute
    ComputePipeline,
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
    "Context",
    "Buffer",
    "Parcel",
    "RetainedPool",
    "RecordBuilder",
    "ShaderModule",
    "RenderPipeline",
    "RenderPipelineDesc",
    "RenderTarget",
    "Scheme",
    "SchemeComputeNode",
    "SchemeRenderPass",
    "SchemeRenderTargetLease",
    "SchemeSubmission",
    "ReadGrant",
    "PresentLease",
    "PresentGrant",
    "SwapchainPool",
    "NodeAccess",
    "ResourceAccess",
    "write_to_parcel",
    # Shader builtins
    "Builtins",
    # Compute
    "ComputePipeline",
    # Surface (windowed rendering)
    "Surface",
    "SurfaceFrame",
]

__version__ = "0.1.0"
