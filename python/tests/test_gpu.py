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
        device.unregister_library('test_lib')
        assert not device.has_library('test_lib')


class TestRetainedPool:
    """Test RetainedPool and Parcel."""

    def test_acquire_from_float32(self, device):
        import goldy

        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pool = goldy.RetainedPool(device)
        parcel = pool.acquire_buffer(data, goldy.BufferKind.SCATTERED)
        assert parcel.byte_size == 16

    def test_acquire_from_int32(self, device):
        import goldy

        data = np.array([1, 2, 3], dtype=np.int32)
        pool = goldy.RetainedPool(device)
        parcel = pool.acquire_buffer(data, goldy.BufferKind.SCATTERED)
        assert parcel.byte_size == 12

    def test_acquire_from_uint16(self, device):
        import goldy

        data = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint16)
        pool = goldy.RetainedPool(device)
        parcel = pool.acquire_buffer(data, goldy.BufferKind.SCATTERED)
        assert parcel.byte_size == 12

    def test_write_parcel(self, device):
        import goldy

        pool = goldy.RetainedPool(device)
        parcel = pool.acquire_buffer(
            np.zeros(64, dtype=np.uint32),
            goldy.BufferKind.SCATTERED,
        )
        ctx = device.create_context()
        frame = goldy.write_to_parcel(
            ctx,
            parcel,
            np.array([1, 2, 3, 4], dtype=np.uint32).tobytes(),
        )
        frame.wait(ctx)


class TestShaderModule:
    """Test ShaderModule class."""

    def test_compile_builtin(self, device):
        import goldy

        shader = goldy.ShaderModule.from_slang(device, goldy.Builtins.VERTEX_COLOR_2D)
        assert shader is not None

    def test_compile_custom(self, device):
        import goldy

        source = '''
        [shader("vertex")]
        float4 vs_main() : SV_Position { return float4(0, 0, 0, 1); }
        [shader("fragment")]
        float4 fs_main() : SV_Target { return float4(1, 0, 0, 1); }
        '''
        shader = goldy.ShaderModule.from_slang(device, source)
        assert shader is not None


class TestComputePipeline:
    """Test ComputePipeline class."""

    def test_create(self, device):
        import goldy

        source = '''
        import goldy_exp;
        [goldy_compute]
        [numthreads(64, 1, 1)]
        void cs_main(Scattered<uint> data, ThreadId id) {
            data[id.x] = data[id.x] * 2;
        }
        '''
        shader = goldy.ShaderModule.from_slang(device, source)
        pipeline = goldy.ComputePipeline(device, shader)
        assert pipeline is not None

    def test_dispatch(self, device):
        import goldy

        source = '''
        import goldy_exp;
        [goldy_compute]
        [numthreads(64, 1, 1)]
        void cs_main(Scattered<uint> data, ThreadId id) {
            data[id.x] = 42;
        }
        '''
        pool = goldy.RetainedPool(device)
        parcel = pool.acquire_buffer(np.zeros(64, dtype=np.uint32), goldy.BufferKind.SCATTERED)
        shader = goldy.ShaderModule.from_slang(device, source)
        pipeline = goldy.ComputePipeline(device, shader)

        ctx = device.create_context()
        scheme = goldy.Scheme(ctx)
        scheme.node("fill", pipeline).declare_parcel(
            parcel, goldy.NodeAccess.WRITE, goldy.ResourceAccess.WRITE
        ).dispatch(1, 1, 1)
        grant = scheme.grant_read(parcel)
        frame = scheme.submit()
        values = np.frombuffer(grant.read(frame), dtype=np.uint32)
        assert np.all(values == 42)

    def test_grant_read_texture(self, device):
        import goldy

        source = '''
        import goldy_exp;
        [goldy_compute]
        [numthreads(8, 8, 1)]
        void cs_main(DirectSpatial<float4> output, ThreadId id) {
            uint2 dims;
            output.GetDimensions(dims.x, dims.y);
            if (id.x < dims.x && id.y < dims.y) {
                output[int2(id.x, id.y)] = float4(1.0, 0.0, 0.0, 1.0);
            }
        }
        '''
        width = height = 16
        pool = goldy.RetainedPool(device)
        parcel = pool.acquire_texture(
            width,
            height,
            goldy.TextureFormat.RGBA8_UNORM,
            goldy.TextureKind.DIRECT,
            copy_src=True,
        )
        shader = goldy.ShaderModule.from_slang(device, source)
        pipeline = goldy.ComputePipeline(device, shader)

        ctx = device.create_context()
        scheme = goldy.Scheme(ctx)
        scheme.node("write_tex", pipeline).declare_parcel(
            parcel, goldy.NodeAccess.WRITE, goldy.ResourceAccess.WRITE
        ).dispatch(2, 2, 1)
        grant = scheme.grant_read_texture(parcel)
        frame = scheme.submit()
        pixels = grant.read(frame)
        assert len(pixels) > 0
        assert pixels[0] == 255
        assert pixels[1] == 0
        assert pixels[2] == 0
        assert pixels[3] == 255
