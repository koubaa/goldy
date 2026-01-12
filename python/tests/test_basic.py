"""Basic tests for Goldy Python bindings.

These tests verify the Python API works correctly without requiring
a GPU (uses the mock backend).
"""

import pytest


def test_import():
    """Test that all exports can be imported."""
    import goldy
    
    # Verify core exports exist
    assert hasattr(goldy, 'Instance')
    assert hasattr(goldy, 'Device')
    assert hasattr(goldy, 'Buffer')
    assert hasattr(goldy, 'RenderTarget')
    assert hasattr(goldy, 'ShaderModule')
    assert hasattr(goldy, 'RenderPipeline')
    assert hasattr(goldy, 'CommandEncoder')
    assert hasattr(goldy, 'RenderPass')
    
    # Verify enums
    assert hasattr(goldy, 'DeviceType')
    assert hasattr(goldy, 'TextureFormat')
    assert hasattr(goldy, 'BufferUsage')
    
    # Verify types
    assert hasattr(goldy, 'Color')
    assert hasattr(goldy, 'VertexBufferLayout')


def test_color():
    """Test Color type."""
    import goldy
    
    # Create from floats
    c = goldy.Color(0.5, 0.25, 0.75, 1.0)
    assert c.r == 0.5
    assert c.g == 0.25
    assert c.b == 0.75
    assert c.a == 1.0
    
    # Create from RGB bytes
    c2 = goldy.Color.from_rgb(255, 128, 0)
    assert c2.r == 1.0
    assert abs(c2.g - 0.502) < 0.01
    assert c2.b == 0.0
    
    # Predefined colors
    assert goldy.Color.RED.r == 1.0
    assert goldy.Color.RED.g == 0.0
    assert goldy.Color.GREEN.g == 1.0
    assert goldy.Color.BLUE.b == 1.0
    
    # To tuple
    t = c.to_tuple()
    assert t == (0.5, 0.25, 0.75, 1.0)
    
    # To RGBA8
    rgba = goldy.Color.RED.to_rgba8()
    assert rgba == [255, 0, 0, 255]


def test_buffer_usage():
    """Test BufferUsage flags."""
    import goldy
    
    # Single flag
    usage = goldy.BufferUsage.VERTEX
    assert 'VERTEX' in repr(usage)
    
    # Combined flags
    combined = goldy.BufferUsage.VERTEX | goldy.BufferUsage.COPY_DST
    assert 'VERTEX' in repr(combined)
    assert 'COPY_DST' in repr(combined)


def test_texture_format():
    """Test TextureFormat enum."""
    import goldy
    
    assert goldy.TextureFormat.RGBA8_UNORM != goldy.TextureFormat.BGRA8_UNORM
    assert goldy.TextureFormat.RGBA8_UNORM == goldy.TextureFormat.RGBA8_UNORM


def test_device_type():
    """Test DeviceType enum."""
    import goldy
    
    assert goldy.DeviceType.DISCRETE_GPU != goldy.DeviceType.INTEGRATED_GPU


def test_vertex_buffer_layout():
    """Test VertexBufferLayout creation."""
    import goldy
    
    # Create default layouts
    layout_2d = goldy.VertexBufferLayout.vertex_2d()
    assert layout_2d.stride == 24  # 2 floats pos + 4 floats color = 6 * 4 = 24
    
    layout_uv = goldy.VertexBufferLayout.vertex_2d_uv()
    assert layout_uv.stride == 16  # 2 floats pos + 2 floats uv = 4 * 4 = 16
    
    # Create custom layout
    attrs = [
        goldy.VertexAttribute(0, goldy.VertexFormat.FLOAT32X3, 0),
        goldy.VertexAttribute(1, goldy.VertexFormat.FLOAT32X2, 12),
    ]
    layout = goldy.VertexBufferLayout(20, attrs)
    assert layout.stride == 20


def test_render_pipeline_desc():
    """Test RenderPipelineDesc creation."""
    import goldy
    
    # Default
    desc = goldy.RenderPipelineDesc()
    assert 'RenderPipelineDesc' in repr(desc)
    
    # With options
    desc2 = goldy.RenderPipelineDesc(
        topology=goldy.PrimitiveTopology.LINE_LIST,
        target_format=goldy.TextureFormat.BGRA8_UNORM,
    )
    assert 'RenderPipelineDesc' in repr(desc2)


def test_depth_stencil_state():
    """Test DepthStencilState creation."""
    import goldy
    
    # Default
    ds = goldy.DepthStencilState()
    assert ds.depth_write_enabled is True
    
    # Custom
    ds2 = goldy.DepthStencilState(
        format=goldy.DepthFormat.DEPTH32_FLOAT,
        depth_write_enabled=False,
        depth_compare=goldy.CompareFunction.LESS_EQUAL,
    )
    assert ds2.depth_write_enabled is False


def test_builtins():
    """Test shader builtins."""
    import goldy
    
    # Check builtin shaders exist
    assert 'vs_main' in goldy.Builtins.VERTEX_COLOR_2D
    assert 'fs_main' in goldy.Builtins.VERTEX_COLOR_2D


def test_command_encoder_context_manager():
    """Test CommandEncoder as context manager."""
    import goldy
    
    encoder = goldy.CommandEncoder()
    
    # The context manager pattern should work
    with encoder.begin_render_pass() as rp:
        rp.clear(goldy.Color.RED)
    
    # Encoder repr
    assert 'CommandEncoder' in repr(encoder)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

