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
    vertex_buffer = pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)
    assert vertex_buffer.byte_size == vertices.nbytes
    assert vertex_buffer[0].byte_size == vertices.nbytes

    indices = np.array([0, 1, 2], dtype=np.uint16)
    index_buffer = pool.acquire_buffer(indices, goldy.BufferKind.SCATTERED)
    assert index_buffer.byte_size == indices.nbytes
    assert index_buffer[0].byte_size == indices.nbytes


def test_parcel_write(device):
    """Upload bytes into a parcel via an upload micro-scheme."""
    import goldy

    pool = goldy.RetainedPool(device)
    buffer = pool.acquire_buffer(
        np.zeros(16, dtype=np.uint32),
        goldy.BufferKind.SCATTERED,
    )
    ctx = device.create_context()
    upload = goldy.Scheme(ctx)
    memory = goldy.MemoryExchange(ctx)
    deposit = memory.bind_deposit_buffer(upload, buffer[0], 16)
    deposit.write(upload, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes())
    frame = upload.submit()
    frame.wait_until_settled()


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


def test_scheme_render_target_lease(device):
    """Test scheme-held render target lease."""
    import goldy

    ctx = device.create_context()
    scheme = goldy.Scheme(ctx)
    lease = scheme.lease_render_target(800, 600, goldy.TextureFormat.RGBA8_UNORM)
    assert lease is not None


def test_scheme_render_target_lease_with_depth(device):
    """Test scheme-held render target lease with depth."""
    import goldy

    ctx = device.create_context()
    scheme = goldy.Scheme(ctx)
    lease = scheme.lease_render_target(
        1024,
        768,
        goldy.TextureFormat.RGBA8_UNORM,
        goldy.DepthFormat.DEPTH24_PLUS,
    )
    assert lease is not None


def test_render_clear_via_scheme(device):
    """Clear a render target through Scheme and verify readback."""
    import goldy

    width = height = 2
    ctx = device.create_context()
    pool = goldy.RetainedPool(device)
    readback = pool.acquire_texture(
        width,
        height,
        goldy.TextureFormat.RGBA8_UNORM,
        goldy.TextureKind.DIRECT,
        copy_src=True,
        copy_dst=True,
    )

    scheme = goldy.Scheme(ctx)
    rt = scheme.lease_render_target(width, height, goldy.TextureFormat.RGBA8_UNORM)
    with scheme.render_pass("clear", rt, goldy.TargetLoad.clear(goldy.Color.RED)) as rp:
        pass

    scheme.copy_to_texture(rt, readback)
    grant = goldy.MemoryExchange(ctx).bind_withdraw_texture(scheme, readback)
    submission = scheme.submit()
    pixels = np.frombuffer(grant.claim(submission).consume(), dtype=np.uint8).reshape(height, width, 4)

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
    buffer = retained_pool.acquire_buffer(zeros, goldy.BufferKind.SCATTERED)
    shader = goldy.ShaderModule.from_slang(device, fill_shader)
    pipeline = goldy.ComputePipeline(device, shader)

    ctx = device.create_context()
    scheme = goldy.Scheme(ctx)
    scheme.node("fill", pipeline).with_parcel(
        buffer[0], goldy.NodeAccess.WRITE
    ).dispatch(1, 1, 1)
    grant = goldy.MemoryExchange(ctx).bind_withdraw(scheme, buffer[0])
    frame = scheme.submit()
    values = np.frombuffer(grant.claim(frame).consume(), dtype=np.uint32)
    assert values.shape == (64,)
    assert np.all(values == 42)


def test_triangle_via_scheme(device):
    """Render a triangle through Scheme and verify non-empty readback."""
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
    vertex_parcel = retained_pool.acquire_buffer(vertices, goldy.BufferKind.SCATTERED)[0]
    readback = retained_pool.acquire_texture(
        100,
        100,
        goldy.TextureFormat.RGBA8_UNORM,
        goldy.TextureKind.DIRECT,
        copy_src=True,
        copy_dst=True,
    )

    ctx = device.create_context()
    scheme = goldy.Scheme(ctx)
    rt = scheme.lease_render_target(100, 100, goldy.TextureFormat.RGBA8_UNORM)
    with scheme.render_pass(
        "triangle", rt, goldy.TargetLoad.clear(goldy.Color(0.0, 0.0, 0.0, 1.0))
    ) as rp:
        (
            rp.with_parcel(vertex_parcel, goldy.NodeAccess.READ)
            .set_pipeline(pipeline)
            .set_vertex_buffer_parcel(0, vertex_parcel)
            .draw(range(3))
        )

    scheme.copy_to_texture(rt, readback)
    grant = goldy.MemoryExchange(ctx).bind_withdraw_texture(scheme, readback)
    submission = scheme.submit()
    pixels = np.frombuffer(grant.claim(submission).consume(), dtype=np.uint8).reshape(100, 100, 4)

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

