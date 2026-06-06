"""GPU integration tests.

These tests require a GPU and test the actual rendering pipeline.
They are skipped if no GPU is available.
"""

import pytest
import numpy as np


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    import goldy
    try:
        instance = goldy.Instance()
        device = instance.request_adapter().request_device()
        return device
    except goldy.GoldyError:
        pytest.skip("No GPU available")


@pytest.fixture
def device():
    """Create a GPU device for testing."""
    return skip_if_no_gpu()


class TestInstance:
    """Test Instance class."""
    
    def test_create(self):
        import goldy
        instance = goldy.Instance()
        assert instance is not None
    
    def test_backend_type(self):
        import goldy
        instance = goldy.Instance()
        bt = instance.backend_type
        assert bt in [goldy.BackendType.VULKAN, goldy.BackendType.METAL, goldy.BackendType.DX12]
    
    def test_enumerate_adapters(self):
        import goldy
        instance = goldy.Instance()
        adapters = instance.enumerate_adapters()
        assert isinstance(adapters, list)


class TestDevice:
    """Test Device class."""
    
    def test_create(self, device):
        assert device.is_valid()
        assert device.adapter_id >= 0
    
    def test_has_default_library(self, device):
        assert device.has_library('goldy_exp')
    
    def test_list_libraries(self, device):
        libs = device.list_libraries()
        assert 'goldy_exp' in libs
    
    def test_register_library(self, device):
        device.register_library('test_lib', '''
            module test_lib;
            public float test_fn() { return 1.0; }
        ''')
        assert device.has_library('test_lib')
        
        # Clean up
        device.unregister_library('test_lib')
        assert not device.has_library('test_lib')


class TestBuffer:
    """Test Buffer class."""
    
    def test_create_from_float32(self, device):
        import goldy
        
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buffer = goldy.Buffer(device, data, goldy.BufferKind.SCATTERED)
        assert buffer.size == 16  # 4 floats * 4 bytes
    
    def test_create_from_int32(self, device):
        import goldy
        
        data = np.array([1, 2, 3], dtype=np.int32)
        buffer = goldy.Buffer(device, data, goldy.BufferKind.SCATTERED)
        assert buffer.size == 12  # 3 ints * 4 bytes
    
    def test_create_from_uint16(self, device):
        import goldy
        
        data = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint16)
        buffer = goldy.Buffer(device, data, goldy.BufferKind.SCATTERED)
        assert buffer.size == 12  # 6 shorts * 2 bytes
    
    def test_create_empty(self, device):
        import goldy
        
        buffer = goldy.Buffer.empty(device, 1024, goldy.BufferKind.BROADCAST)
        assert buffer.size == 1024
    
    def test_write(self, device):
        import goldy
        
        buffer = goldy.Buffer.empty(device, 64, goldy.BufferKind.BROADCAST)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buffer.write(0, data)  # Should not raise
    
    def test_write_with_offset(self, device):
        import goldy
        
        buffer = goldy.Buffer.empty(device, 64, goldy.BufferKind.BROADCAST)
        data = np.array([1.0, 2.0], dtype=np.float32)
        buffer.write(16, data)  # Write at offset 16


class TestShaderModule:
    """Test ShaderModule class."""
    
    def test_compile_builtin(self, device):
        import goldy
        
        shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
        assert shader is not None
    
    def test_compile_custom(self, device):
        import goldy
        
        source = '''
struct VS_IN { float2 pos : POSITION; };
struct VS_OUT { float4 pos : SV_Position; };

[shader("vertex")]
VS_OUT vs_main(VS_IN input) {
    VS_OUT output;
    output.pos = float4(input.pos, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fs_main() : SV_Target {
    return float4(1.0, 0.0, 0.0, 1.0);
}
'''
        shader = goldy.ShaderModule.from_slang(device, source)
        assert shader is not None
    
    def test_compile_with_library(self, device):
        import goldy
        
        source = '''
import goldy_exp;

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    return float4(rainbow(input.uv.x), 1.0);
}
'''
        shader = goldy.ShaderModule.from_slang(device, source)
        assert shader is not None


class TestRenderTarget:
    """Test RenderTarget class."""
    
    def test_create(self, device):
        import goldy
        
        target = goldy.RenderTarget(device, 100, 100, goldy.TextureFormat.RGBA8_UNORM)
        assert target.width == 100
        assert target.height == 100
        assert target.buffer_size == 100 * 100 * 4
        assert not target.has_depth()
    
    def test_create_with_depth(self, device):
        import goldy
        
        target = goldy.RenderTarget.with_depth(
            device, 256, 256,
            goldy.TextureFormat.RGBA8_UNORM,
            goldy.DepthFormat.DEPTH24_PLUS,
        )
        assert target.has_depth()
    
    def test_clear_and_read(self, device):
        import goldy
        
        target = goldy.RenderTarget(device, 2, 2, goldy.TextureFormat.RGBA8_UNORM)
        
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color.RED)
        
        target.render(encoder)
        pixels = target.read_to_cpu()
        
        assert pixels.shape == (2, 2, 4)
        assert np.all(pixels[:, :, 0] == 255)  # R
        assert np.all(pixels[:, :, 1] == 0)    # G
        assert np.all(pixels[:, :, 2] == 0)    # B
        assert np.all(pixels[:, :, 3] == 255)  # A
    
    def test_read_to_bytes(self, device):
        import goldy
        
        target = goldy.RenderTarget(device, 10, 10, goldy.TextureFormat.RGBA8_UNORM)
        
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color.GREEN)
        
        target.render(encoder)
        pixels = target.read_to_bytes()
        
        assert len(pixels) == 10 * 10 * 4


class TestFullPipeline:
    """Test complete render pipeline."""
    
    def test_triangle(self, device):
        import goldy
        
        # Create resources
        shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
        
        pipeline = goldy.RenderPipeline(
            device, shader, shader,
            goldy.RenderPipelineDesc(
                vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
                target_format=goldy.TextureFormat.RGBA8_UNORM,
            )
        )
        
        vertices = np.array([
             0.0, -0.5, 1.0, 0.0, 0.0, 1.0,
            -0.5,  0.5, 0.0, 1.0, 0.0, 1.0,
             0.5,  0.5, 0.0, 0.0, 1.0, 1.0,
        ], dtype=np.float32)
        
        vertex_buffer = goldy.Buffer(device, vertices, goldy.BufferKind.SCATTERED)
        
        target = goldy.RenderTarget(device, 100, 100, goldy.TextureFormat.RGBA8_UNORM)
        
        # Render
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color(0.0, 0.0, 0.0, 1.0))
            rp.set_pipeline(pipeline)
            rp.set_vertex_buffer(0, vertex_buffer)
            rp.draw(range(3))
        
        target.render(encoder)
        
        # Verify
        pixels = target.read_to_cpu()
        assert pixels.shape == (100, 100, 4)
        
        # Some pixels should be non-black (the triangle)
        assert np.any(pixels[:, :, :3] > 0)
    
    def test_multiple_draws(self, device):
        import goldy
        
        shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
        
        pipeline = goldy.RenderPipeline(
            device, shader, shader,
            goldy.RenderPipelineDesc(target_format=goldy.TextureFormat.RGBA8_UNORM)
        )
        
        # Two triangles
        vertices = np.array([
            # Triangle 1 (left, red)
            -0.5, -0.5, 1.0, 0.0, 0.0, 1.0,
            -0.9,  0.5, 1.0, 0.0, 0.0, 1.0,
            -0.1,  0.5, 1.0, 0.0, 0.0, 1.0,
            # Triangle 2 (right, blue)
             0.5, -0.5, 0.0, 0.0, 1.0, 1.0,
             0.1,  0.5, 0.0, 0.0, 1.0, 1.0,
             0.9,  0.5, 0.0, 0.0, 1.0, 1.0,
        ], dtype=np.float32)
        
        vertex_buffer = goldy.Buffer(device, vertices, goldy.BufferKind.SCATTERED)
        target = goldy.RenderTarget(device, 100, 100, goldy.TextureFormat.RGBA8_UNORM)
        
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color.BLACK)
            rp.set_pipeline(pipeline)
            rp.set_vertex_buffer(0, vertex_buffer)
            rp.draw(range(3))       # First triangle
            rp.draw(range(3, 6))    # Second triangle
        
        target.render(encoder)
        pixels = target.read_to_cpu()
        
        # Check both colors appear
        assert np.any(pixels[:, :, 0] > 128)  # Red
        assert np.any(pixels[:, :, 2] > 128)  # Blue


class TestContextManager:
    """Test context manager behavior."""
    
    def test_render_pass_context(self, device):
        import goldy
        
        encoder = goldy.CommandEncoder()
        
        with encoder.begin_render_pass() as rp:
            assert rp is not None
            rp.clear(goldy.Color.CORNFLOWER_BLUE)
    
    def test_nested_commands(self, device):
        import goldy
        
        target = goldy.RenderTarget(device, 50, 50, goldy.TextureFormat.RGBA8_UNORM)
        
        encoder = goldy.CommandEncoder()
        with encoder.begin_render_pass() as rp:
            rp.clear(goldy.Color.WHITE)
        
        target.render(encoder)
        
        # Verify it rendered
        pixels = target.read_to_cpu()
        assert np.all(pixels[:, :, 0] == 255)


class TestComputePipeline:
    """Test compute shader pipeline (covers game_of_life.py functionality)."""
    
    def test_compute_double_values(self, device):
        """Test basic compute shader that doubles buffer values."""
        import goldy
        
        # Simple compute shader that doubles each value (cross-platform)
        compute_shader_src = '''
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<float> data, ThreadId id) {
    data[id.x] = data[id.x] * 2.0;
}
'''
        
        # Create input data
        input_data = np.arange(256, dtype=np.float32)
        
        # Create GPU storage buffer
        buffer = goldy.Buffer(device, input_data, goldy.BufferKind.SCATTERED)
        
        # Compile compute shader
        shader = goldy.ShaderModule.from_slang(device, compute_shader_src)
        
        # Create compute pipeline
        pipeline = goldy.ComputePipeline(device, shader)
        
        # Dispatch compute work
        encoder = goldy.ComputeEncoder()
        with encoder.begin_compute_pass() as cp:
            cp.set_pipeline(pipeline)
            cp.bind_resources([buffer])
            # 256 elements / 64 threads per workgroup = 4 workgroups
            cp.dispatch(4, 1, 1)
        
        encoder.dispatch(device)
        
        # Compute shader executed without error
        # (Full readback verification would require COPY_SRC staging buffer)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

