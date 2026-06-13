"""Integration tests for Goldy Python bindings.

These tests require a GPU and test the full rendering pipeline.
They are skipped if no GPU is available.
"""

import pytest
import numpy as np


@pytest.fixture
def device():
    """Create a GPU device for testing."""
    import goldy
    
    try:
        instance = goldy.Instance()
        return instance.request_adapter().request_device()
    except goldy.GoldyError:
        pytest.skip("No GPU available")


def test_instance_creation():
    """Test Instance creation and adapter enumeration."""
    import goldy
    
    instance = goldy.Instance()
    
    # Check backend type
    backend = instance.backend_type
    assert backend in [
        goldy.BackendType.VULKAN,
        goldy.BackendType.METAL,
        goldy.BackendType.DX12,
    ]
    
    # Enumerate adapters
    adapters = instance.enumerate_adapters()
    assert len(adapters) >= 0  # May be 0 if no GPU
    
    if adapters:
        adapter = adapters[0]
        assert adapter.id >= 0
        assert len(adapter.name) > 0


def test_device_creation(device):
    """Test Device creation."""
    import goldy
    
    assert device.is_valid()
    assert device.adapter_id >= 0
    
    # Check default library
    assert device.has_library('goldy_exp')
    
    # List libraries
    libs = device.list_libraries()
    assert 'goldy_exp' in libs


def test_parcel_creation_numpy(device):
    """Test Parcel creation from numpy arrays via RetainedPool."""
    import goldy

    vertices = np.array([
        0.0, -0.5, 1.0, 0.0, 0.0, 1.0,
        -0.5, 0.5, 0.0, 1.0, 0.0, 1.0,
        0.5, 0.5, 0.0, 0.0, 1.0, 1.0,
    ], dtype=np.float32)

    pool = goldy.RetainedPool(device)
    parcel = pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)
    assert parcel.byte_size == vertices.nbytes

    indices = np.array([0, 1, 2], dtype=np.uint16)
    index_parcel = pool.acquire_buffer(indices, goldy.BufferKind.SCATTERED)
    assert index_parcel.byte_size == indices.nbytes


def test_parcel_write(device):
    """Upload bytes into a parcel through write_to_parcel."""
    import goldy

    pool = goldy.RetainedPool(device)
    parcel = pool.acquire_buffer(
        np.zeros(16, dtype=np.uint32),
        goldy.BufferKind.SCATTERED,
    )
    ctx = device.create_context()
    goldy.write_to_parcel(
        ctx,
        parcel,
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes(),
    )


def test_shader_compilation(device):
    """Test ShaderModule compilation."""
    import goldy
    
    # Compile a simple shader
    source = '''
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
'''
    shader = goldy.ShaderModule.from_slang(device, source)
    assert shader is not None


def test_render_target_creation(device):
    """Test RenderTarget creation."""
    import goldy
    
    target = goldy.RenderTarget(device, 800, 600, goldy.TextureFormat.RGBA8_UNORM)
    assert target.width == 800
    assert target.height == 600
    assert target.buffer_size == 800 * 600 * 4
    assert not target.has_depth()


def test_render_target_with_depth(device):
    """Test RenderTarget with depth buffer."""
    import goldy
    
    target = goldy.RenderTarget.with_depth(
        device, 1024, 768,
        goldy.TextureFormat.RGBA8_UNORM,
        goldy.DepthFormat.DEPTH24_PLUS,
    )
    assert target.width == 1024
    assert target.height == 768
    assert target.has_depth()


def test_render_clear_via_graph(device):
    """Clear a render target through TaskGraph and verify readback."""
    import goldy

    target = goldy.RenderTarget(device, 2, 2, goldy.TextureFormat.RGBA8_UNORM)
    graph = goldy.TaskGraph()
    with graph.render_pass("clear", target) as rp:
        rp.clear(goldy.Color.RED)
    graph.dispatch(device)

    pixels = target.read_to_cpu()
    assert pixels.shape == (2, 2, 4)
    assert np.all(pixels[:, :, 0] == 255)
    assert np.all(pixels[:, :, 1] == 0)
    assert np.all(pixels[:, :, 2] == 0)
    assert np.all(pixels[:, :, 3] == 255)


def test_compute_node_fills_buffer_with_42(device):
    """Fill a buffer via Scheme compute node and verify readback."""
    import goldy

    fill_shader = """
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = 42;
}
"""

    retained_pool = goldy.RetainedPool(device)
    zeros = np.zeros(64, dtype=np.uint32)
    parcel = retained_pool.acquire_buffer(zeros, goldy.BufferKind.SCATTERED)
    shader = goldy.ShaderModule.from_slang(device, fill_shader)
    pipeline = goldy.ComputePipeline(device, shader)

    ctx = device.create_context()
    scheme = goldy.Scheme(ctx)
    scheme.node("fill", pipeline).declare_parcel(
        parcel, goldy.NodeAccess.WRITE, goldy.ResourceAccess.WRITE
    ).dispatch(1, 1, 1)
    scheme.submit()

    values = np.frombuffer(parcel.read_to_cpu(device), dtype=np.uint32)
    assert values.shape == (64,)
    assert np.all(values == 42)


def test_triangle_via_graph(device):
    """Render a triangle through TaskGraph and verify non-empty readback."""
    import goldy

    shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
    pipeline = goldy.RenderPipeline(
        device,
        shader,
        shader,
        goldy.RenderPipelineDesc(
            vertex_layout=goldy.VertexBufferLayout.vertex_2d(),
            target_format=goldy.TextureFormat.RGBA8_UNORM,
        ),
    )

    vertices = np.array(
        [
            0.0,
            -0.5,
            1.0,
            0.0,
            0.0,
            1.0,
            -0.5,
            0.5,
            0.0,
            1.0,
            0.0,
            1.0,
            0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )
    retained_pool = goldy.RetainedPool(device)
    vertex_parcel = retained_pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)
    target = goldy.RenderTarget(device, 100, 100, goldy.TextureFormat.RGBA8_UNORM)

    graph = goldy.TaskGraph()
    with graph.render_pass("triangle", target) as rp:
        (
            rp.bind_parcel(vertex_parcel, goldy.NodeAccess.READ)
            .clear(goldy.Color(0.0, 0.0, 0.0, 1.0))
            .set_pipeline(pipeline)
            .set_vertex_buffer_parcel(0, vertex_parcel)
            .draw(range(3))
        )

    graph.dispatch(device)
    pixels = target.read_to_cpu()
    assert pixels.shape == (100, 100, 4)
    assert np.any(pixels[:, :, :3] > 0)


def test_custom_shader_library(device):
    """Test registering a custom shader library."""
    import goldy
    
    # Register a custom library
    device.register_library('mylib', '''
        module mylib;
        public float3 my_color() { return float3(1.0, 0.5, 0.0); }
    ''')
    
    assert device.has_library('mylib')
    
    # Unregister it
    assert device.unregister_library('mylib')
    assert not device.has_library('mylib')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

