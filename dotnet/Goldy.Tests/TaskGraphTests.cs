using System.Runtime.InteropServices;

namespace Goldy.Tests;

public class TaskGraphTests
{
    [StructLayout(LayoutKind.Sequential)]
    struct Vertex2D
    {
        public float Px, Py;
        public float R, G, B, A;
    }

    [Fact]
    public void TaskGraph_ClearRenderTarget_ReadbackIsRed()
    {
        try
        {
            using var instance = new Instance();
            using var device = instance.RequestAdapter().RequestDevice();
            using var target = new RenderTarget(device, 2, 2, TextureFormat.Rgba8Unorm);
            using var graph = new TaskGraph();

            using (var pass = graph.RenderPass("clear", target))
                pass.Clear(Color.Red);

            graph.Dispatch(device);

            var pixels = target.ReadToCpu();
            Assert.Equal(2u * 2u * 4u, (uint)pixels.Length);
            for (var i = 0; i < pixels.Length; i += 4)
            {
                Assert.Equal(255, pixels[i]);     // R
                Assert.Equal(0, pixels[i + 1]);   // G
                Assert.Equal(0, pixels[i + 2]);   // B
                Assert.Equal(255, pixels[i + 3]); // A
            }
        }
        catch (GoldyException ex) when (ex.Message.Contains("adapter", StringComparison.OrdinalIgnoreCase))
        {
            // No GPU in CI — skip gracefully.
        }
    }

    [Fact]
    public void TaskGraph_Triangle_ReadbackHasColor()
    {
        try
        {
            using var instance = new Instance();
            using var device = instance.RequestAdapter().RequestDevice();

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
            using var vertexBuffer = Goldy.Buffer.WithData(device, vertices, BufferKind.Scattered);
            using var target = new RenderTarget(device, 64, 64, TextureFormat.Rgba8Unorm);
            using var graph = new TaskGraph();

            using (var pass = graph.RenderPass("triangle", target))
            {
                pass
                    .BindBuffer(vertexBuffer, NodeAccess.Read)
                    .Clear(Color.Black)
                    .SetPipeline(pipeline)
                    .SetVertexBuffer(0, vertexBuffer)
                    .Draw(3);
            }

            graph.Dispatch(device);

            var pixels = target.ReadToCpu();
            Assert.Contains(pixels, b => b > 0);
        }
        catch (GoldyException ex) when (ex.Message.Contains("adapter", StringComparison.OrdinalIgnoreCase))
        {
            // No GPU in CI — skip gracefully.
        }
    }

    [Fact]
    public void TaskGraph_ComputeNode_FillsBufferWith42()
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
            using var buffer = Goldy.Buffer.WithData<uint>(device, initial, BufferKind.Scattered);
            using var shader = new ShaderModule(device, fillShader);
            using var pipeline = new ComputePipeline(device, shader);
            using var graph = new TaskGraph();

            var idx = buffer.ResourceIndex(ResourceAccess.Write);
            using (var node = graph.ComputeNode("fill", pipeline))
            {
                node
                    .BindBuffer(buffer, NodeAccess.Write)
                    .BindResourcesRaw(idx)
                    .Dispatch(1, 1, 1);
            }

            graph.Dispatch(device);

            var bytes = buffer.ReadToCpu(device);
            var values = MemoryMarshal.Cast<byte, uint>(bytes);
            foreach (var v in values)
                Assert.Equal(42u, v);
        }
        catch (GoldyException ex) when (ex.Message.Contains("adapter", StringComparison.OrdinalIgnoreCase))
        {
            // No GPU in CI — skip gracefully.
        }
    }
}
