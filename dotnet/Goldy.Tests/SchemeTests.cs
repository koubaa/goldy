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
            using var buffer = retainedPool.AcquireBuffer<uint>(initial, BufferKind.Scattered);
            using var shader = new ShaderModule(device, fillShader);
            using var pipeline = new ComputePipeline(device, shader);
            using var ctx = device.CreateContext();
            using var scheme = new Scheme(ctx);

            using (var node = scheme.ComputeNode("fill", pipeline))
            {
                node
                    .WithField(buffer, 0, NodeAccess.Write)
                    .Dispatch(1, 1, 1);
            }

            using var grant = scheme.GrantRead(buffer.Field(0));
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
            using var texture = retainedPool.AcquireTexture(
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
                    .WithTexture(texture, NodeAccess.Write)
                    .Dispatch(2, 2, 1);
            }

            using var grant = scheme.GrantReadTexture(texture);
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

    [Fact]
    public void Scheme_ClearRenderTarget_ReadbackIsRed()
    {
        try
        {
            using var instance = new Instance();
            using var device = instance.RequestAdapter().RequestDevice();
            using var ctx = device.CreateContext();
            using var retainedPool = new RetainedPool(device);
            using var readback = retainedPool.AcquireTexture(
                2,
                2,
                TextureFormat.Rgba8Unorm,
                TextureKind.Direct,
                TextureFlags.CopySrc | TextureFlags.CopyDst);

            using var scheme = new Scheme(ctx);
            using var rt = scheme.LeaseRenderTarget(2, 2, TextureFormat.Rgba8Unorm);
            using (var pass = scheme.RenderPassClear("clear", rt, Color.Red))
            { }

            scheme.CopyToTexture(rt, readback);
            using var grant = scheme.GrantReadTexture(readback);
            using var submission = scheme.Submit();
            var pixels = grant.Consume(submission);

            Assert.Equal(2u * 2u * 4u, (uint)pixels.Length);
            for (var i = 0; i < pixels.Length; i += 4)
            {
                Assert.Equal(255, pixels[i]);
                Assert.Equal(0, pixels[i + 1]);
                Assert.Equal(0, pixels[i + 2]);
                Assert.Equal(255, pixels[i + 3]);
            }
        }
        catch (GoldyException ex) when (ex.Message.Contains("adapter", StringComparison.OrdinalIgnoreCase))
        {
            // No GPU in CI — skip gracefully.
        }
    }

    [Fact]
    public void Scheme_Triangle_ReadbackHasColor()
    {
        try
        {
            using var instance = new Instance();
            using var device = instance.RequestAdapter().RequestDevice();
            using var ctx = device.CreateContext();

            using var shader = new ShaderModule(device, ShaderModule.BuiltinVertexColor2D);
            using var pipeline = new RenderPipeline(
                device,
                shader,
                shader,
                new RenderPipelineDesc
                {
                    VertexAttributes = VertexLayouts.Vertex2D,
                    VertexStride = 24,
                    TargetFormat = TextureFormat.Rgba8Unorm,
                });

            ReadOnlySpan<Vertex2D> vertices =
            [
                new() { Px = 0.0f, Py = -0.5f, R = 1, G = 0, B = 0, A = 1 },
                new() { Px = -0.5f, Py = 0.5f, R = 0, G = 1, B = 0, A = 1 },
                new() { Px = 0.5f, Py = 0.5f, R = 0, G = 0, B = 1, A = 1 },
            ];
            using var retainedPool = new RetainedPool(device);
            using var vertexBuffer = retainedPool.AcquireBuffer(vertices, BufferKind.Scattered);
            using var vertexParcel = vertexBuffer.Field(0);
            using var readback = retainedPool.AcquireTexture(
                64,
                64,
                TextureFormat.Rgba8Unorm,
                TextureKind.Direct,
                TextureFlags.CopySrc | TextureFlags.CopyDst);

            using var scheme = new Scheme(ctx);
            using var rt = scheme.LeaseRenderTarget(64, 64, TextureFormat.Rgba8Unorm);
            using (var pass = scheme.RenderPassClear("triangle", rt, Color.Black))
            {
                pass
                    .WithParcel(vertexParcel, NodeAccess.Read)
                    .SetPipeline(pipeline)
                    .SetVertexBuffer(0, vertexParcel)
                    .Draw(3);
            }

            scheme.CopyToTexture(rt, readback);
            using var grant = scheme.GrantReadTexture(readback);
            using var submission = scheme.Submit();
            var pixels = grant.Consume(submission);

            Assert.Contains(pixels, b => b > 0);
        }
        catch (GoldyException ex) when (ex.Message.Contains("adapter", StringComparison.OrdinalIgnoreCase))
        {
            // No GPU in CI — skip gracefully.
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Vertex2D
    {
        public float Px, Py;
        public float R, G, B, A;
    }
}
