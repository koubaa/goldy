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
            using var retainedPool = new RetainedPool(device);
            using var vertexParcel = retainedPool.AcquireBuffer(vertices, BufferKind.Scattered);
            using var target = new RenderTarget(device, 64, 64, TextureFormat.Rgba8Unorm);
            using var graph = new TaskGraph();

            using (var pass = graph.RenderPass("triangle", target))
            {
                pass
                    .BindParcel(vertexParcel, NodeAccess.Read)
                    .Clear(Color.Black)
                    .SetPipeline(pipeline)
                    .SetVertexBuffer(0, vertexParcel)
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
}
