using System.Runtime.InteropServices;

namespace Goldy.Tests;

public class SchemeTests
{
    [Fact]
    public void Scheme_ComputeNode_FillsBufferWith42()
    {
        const string fillShader = """
            import goldy_exp;

            [goldy_compute]
            [numthreads(64, 1, 1)]
            void cs_main(Scattered<uint> data, ThreadId id) {
                data[id.x] = 42;
            }
            """;

        try
        {
            using var instance = new Instance();
            using var device = instance.RequestAdapter().RequestDevice();

            uint[] initial = Enumerable.Range(0, 64).Select(i => (uint)i).ToArray();
            using var retainedPool = new RetainedPool(device);
            using var parcel = retainedPool.AcquireBuffer<uint>(initial, BufferKind.Scattered);
            using var shader = new ShaderModule(device, fillShader);
            using var pipeline = new ComputePipeline(device, shader);
            using var ctx = device.CreateContext();
            using var scheme = new Scheme(ctx);

            using (var node = scheme.ComputeNode("fill", pipeline))
            {
                node
                    .DeclareParcel(parcel, NodeAccess.Write, ResourceAccess.Write)
                    .Dispatch(1, 1, 1);
            }

            using var grant = scheme.GrantRead(parcel);
            using var frame = scheme.Submit();
            var bytes = grant.Consume(frame);
            var values = MemoryMarshal.Cast<byte, uint>(bytes);
            foreach (var v in values)
                Assert.Equal(42u, v);
        }
        catch (GoldyException ex) when (ex.Message.Contains("adapter", StringComparison.OrdinalIgnoreCase))
        {
            // No GPU in CI — skip gracefully.
        }
    }

    [Fact]
    public void Scheme_GrantReadTexture_ReadsRedPixel()
    {
        const string writeTextureShader = """
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
            """;

        try
        {
            using var instance = new Instance();
            using var device = instance.RequestAdapter().RequestDevice();

            using var retainedPool = new RetainedPool(device);
            using var parcel = retainedPool.AcquireTexture(
                16,
                16,
                TextureFormat.Rgba8Unorm,
                TextureKind.Direct,
                TextureFlags.CopySrc);
            using var shader = new ShaderModule(device, writeTextureShader);
            using var pipeline = new ComputePipeline(device, shader);
            using var ctx = device.CreateContext();
            using var scheme = new Scheme(ctx);

            using (var node = scheme.ComputeNode("write_tex", pipeline))
            {
                node
                    .DeclareParcel(parcel, NodeAccess.Write, ResourceAccess.Write)
                    .Dispatch(2, 2, 1);
            }

            using var grant = scheme.GrantReadTexture(parcel);
            using var frame = scheme.Submit();
            var bytes = grant.Consume(frame);
            Assert.True(bytes.Length > 0);
            Assert.Equal(255, bytes[0]);
            Assert.Equal(0, bytes[1]);
            Assert.Equal(0, bytes[2]);
            Assert.Equal(255, bytes[3]);
        }
        catch (GoldyException ex) when (ex.Message.Contains("adapter", StringComparison.OrdinalIgnoreCase))
        {
            // No GPU in CI — skip gracefully.
        }
    }
}
