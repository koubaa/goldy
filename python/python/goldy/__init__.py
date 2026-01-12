"""Goldy GPU library for Python.

A modern GPU library targeting Vulkan 1.3+, DX12, and Metal.

Example:
    >>> import goldy
    >>> import numpy as np
    >>> 
    >>> instance = goldy.Instance()
    >>> device = instance.create_device(goldy.DeviceType.DISCRETE_GPU)
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

from goldy._goldy import (
    # Exception
    GoldyError,
    # Enums
    DeviceType,
    BackendType,
    TextureFormat,
    BufferUsage,
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
    # Bind groups
    ShaderStages,
    BindingType,
    BindGroupLayoutBinding,
    BindGroupLayout,
    BufferBinding,
    BindGroup,
    # Compute
    ComputePipelineDesc,
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
    "BufferUsage",
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
    # Bind groups
    "ShaderStages",
    "BindingType",
    "BindGroupLayoutBinding",
    "BindGroupLayout",
    "BufferBinding",
    "BindGroup",
    # Compute
    "ComputePipelineDesc",
    "ComputePipeline",
    "ComputeEncoder",
    "ComputePass",
    # Surface (windowed rendering)
    "Surface",
    "SurfaceFrame",
]

__version__ = "0.1.0"
