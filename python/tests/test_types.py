"""Tests for Goldy type wrappers."""

import pytest


class TestColor:
    """Test Color type."""
    
    def test_create_from_floats(self):
        import goldy
        
        c = goldy.Color(0.5, 0.25, 0.75, 0.8)
        assert c.r == 0.5
        assert c.g == 0.25
        assert c.b == 0.75
        # f32 precision: 0.8 isn't exactly representable
        assert abs(c.a - 0.8) < 1e-6
    
    def test_create_with_default_alpha(self):
        import goldy
        
        c = goldy.Color(1.0, 0.5, 0.0)
        assert c.a == 1.0
    
    def test_from_rgb(self):
        import goldy
        
        c = goldy.Color.from_rgb(255, 128, 0)
        assert c.r == 1.0
        assert abs(c.g - 0.502) < 0.01
        assert c.b == 0.0
        assert c.a == 1.0
    
    def test_from_rgba(self):
        import goldy
        
        c = goldy.Color.from_rgba(255, 128, 64, 128)
        assert c.r == 1.0
        assert abs(c.g - 0.502) < 0.01
        assert abs(c.b - 0.251) < 0.01
        assert abs(c.a - 0.502) < 0.01
    
    def test_to_tuple(self):
        import goldy
        
        c = goldy.Color(0.1, 0.2, 0.3, 0.4)
        t = c.to_tuple()
        # f32 precision means values aren't exactly equal
        assert len(t) == 4
        assert abs(t[0] - 0.1) < 1e-6
        assert abs(t[1] - 0.2) < 1e-6
        assert abs(t[2] - 0.3) < 1e-6
        assert abs(t[3] - 0.4) < 1e-6
    
    def test_to_rgba8(self):
        import goldy
        
        c = goldy.Color(1.0, 0.5, 0.0, 1.0)
        rgba = c.to_rgba8()
        assert rgba[0] == 255
        assert rgba[1] == 127
        assert rgba[2] == 0
        assert rgba[3] == 255
    
    def test_predefined_colors(self):
        import goldy
        
        assert goldy.Color.BLACK.r == 0.0
        assert goldy.Color.BLACK.g == 0.0
        assert goldy.Color.BLACK.b == 0.0
        
        assert goldy.Color.WHITE.r == 1.0
        assert goldy.Color.WHITE.g == 1.0
        assert goldy.Color.WHITE.b == 1.0
        
        assert goldy.Color.RED.r == 1.0
        assert goldy.Color.RED.g == 0.0
        assert goldy.Color.RED.b == 0.0
        
        assert goldy.Color.GREEN.g == 1.0
        assert goldy.Color.BLUE.b == 1.0
    
    def test_repr(self):
        import goldy
        
        c = goldy.Color(0.1, 0.2, 0.3, 0.4)
        r = repr(c)
        assert 'Color' in r
        assert '0.1' in r


class TestBufferUsage:
    """Test BufferUsage flags."""
    
    def test_single_flag(self):
        import goldy
        
        usage = goldy.BufferUsage.VERTEX
        assert 'VERTEX' in repr(usage)
    
    def test_combine_flags(self):
        import goldy
        
        usage = goldy.BufferUsage.VERTEX | goldy.BufferUsage.COPY_DST
        r = repr(usage)
        assert 'VERTEX' in r
        assert 'COPY_DST' in r
    
    def test_multiple_flags(self):
        import goldy
        
        usage = (goldy.BufferUsage.VERTEX 
                | goldy.BufferUsage.INDEX 
                | goldy.BufferUsage.UNIFORM)
        r = repr(usage)
        assert 'VERTEX' in r
        assert 'INDEX' in r
        assert 'UNIFORM' in r
    
    def test_all_flags_exist(self):
        import goldy
        
        # Verify all flags are accessible
        _ = goldy.BufferUsage.VERTEX
        _ = goldy.BufferUsage.INDEX
        _ = goldy.BufferUsage.UNIFORM
        _ = goldy.BufferUsage.STORAGE
        _ = goldy.BufferUsage.COPY_SRC
        _ = goldy.BufferUsage.COPY_DST


class TestEnums:
    """Test enum types."""
    
    def test_device_type(self):
        import goldy
        
        assert goldy.DeviceType.DISCRETE_GPU != goldy.DeviceType.INTEGRATED_GPU
        assert goldy.DeviceType.DISCRETE_GPU != goldy.DeviceType.CPU
    
    def test_backend_type(self):
        import goldy
        
        # Just verify they exist
        _ = goldy.BackendType.VULKAN
        _ = goldy.BackendType.METAL
        _ = goldy.BackendType.DX12
    
    def test_texture_format(self):
        import goldy
        
        assert goldy.TextureFormat.RGBA8_UNORM != goldy.TextureFormat.BGRA8_UNORM
        assert goldy.TextureFormat.RGBA8_UNORM == goldy.TextureFormat.RGBA8_UNORM
        
        # All formats exist
        _ = goldy.TextureFormat.RGBA8_UNORM_SRGB
        _ = goldy.TextureFormat.RGBA8_UNORM
        _ = goldy.TextureFormat.BGRA8_UNORM_SRGB
        _ = goldy.TextureFormat.BGRA8_UNORM
        _ = goldy.TextureFormat.RGBA16_FLOAT
        _ = goldy.TextureFormat.RGBA32_FLOAT
    
    def test_primitive_topology(self):
        import goldy
        
        _ = goldy.PrimitiveTopology.POINT_LIST
        _ = goldy.PrimitiveTopology.LINE_LIST
        _ = goldy.PrimitiveTopology.LINE_STRIP
        _ = goldy.PrimitiveTopology.TRIANGLE_LIST
        _ = goldy.PrimitiveTopology.TRIANGLE_STRIP
    
    def test_index_format(self):
        import goldy
        
        assert goldy.IndexFormat.UINT16 != goldy.IndexFormat.UINT32
    
    def test_depth_format(self):
        import goldy
        
        _ = goldy.DepthFormat.DEPTH16_UNORM
        _ = goldy.DepthFormat.DEPTH24_PLUS
        _ = goldy.DepthFormat.DEPTH24_PLUS_STENCIL8
        _ = goldy.DepthFormat.DEPTH32_FLOAT
        _ = goldy.DepthFormat.DEPTH32_FLOAT_STENCIL8
    
    def test_compare_function(self):
        import goldy
        
        _ = goldy.CompareFunction.NEVER
        _ = goldy.CompareFunction.LESS
        _ = goldy.CompareFunction.EQUAL
        _ = goldy.CompareFunction.LESS_EQUAL
        _ = goldy.CompareFunction.GREATER
        _ = goldy.CompareFunction.NOT_EQUAL
        _ = goldy.CompareFunction.GREATER_EQUAL
        _ = goldy.CompareFunction.ALWAYS


class TestVertexLayouts:
    """Test vertex buffer layouts."""
    
    def test_vertex_2d_layout(self):
        import goldy
        
        layout = goldy.VertexBufferLayout.vertex_2d()
        # Vertex2D: 2 floats (pos) + 4 floats (color) = 6 * 4 = 24 bytes
        assert layout.stride == 24
    
    def test_vertex_2d_uv_layout(self):
        import goldy
        
        layout = goldy.VertexBufferLayout.vertex_2d_uv()
        # Vertex2DUv: 2 floats (pos) + 2 floats (uv) = 4 * 4 = 16 bytes
        assert layout.stride == 16
    
    def test_custom_layout(self):
        import goldy
        
        attrs = [
            goldy.VertexAttribute(0, goldy.VertexFormat.FLOAT32X3, 0),
            goldy.VertexAttribute(1, goldy.VertexFormat.FLOAT32X2, 12),
        ]
        layout = goldy.VertexBufferLayout(20, attrs)
        assert layout.stride == 20
    
    def test_vertex_attribute(self):
        import goldy
        
        attr = goldy.VertexAttribute(0, goldy.VertexFormat.FLOAT32X4, 16)
        assert attr.location == 0
        assert attr.offset == 16


class TestDepthStencilState:
    """Test DepthStencilState."""
    
    def test_default(self):
        import goldy
        
        ds = goldy.DepthStencilState()
        assert ds.depth_write_enabled is True
    
    def test_custom(self):
        import goldy
        
        ds = goldy.DepthStencilState(
            format=goldy.DepthFormat.DEPTH32_FLOAT,
            depth_write_enabled=False,
            depth_compare=goldy.CompareFunction.GREATER,
        )
        assert ds.depth_write_enabled is False


class TestRenderPipelineDesc:
    """Test RenderPipelineDesc."""
    
    def test_default(self):
        import goldy
        
        desc = goldy.RenderPipelineDesc()
        assert 'RenderPipelineDesc' in repr(desc)
    
    def test_with_options(self):
        import goldy
        
        desc = goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
            topology=goldy.PrimitiveTopology.LINE_LIST,
            target_format=goldy.TextureFormat.BGRA8_UNORM,
        )
        assert 'RenderPipelineDesc' in repr(desc)
    
    def test_with_depth(self):
        import goldy
        
        desc = goldy.RenderPipelineDesc(
            depth_stencil=goldy.DepthStencilState(),
        )
        assert desc is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

