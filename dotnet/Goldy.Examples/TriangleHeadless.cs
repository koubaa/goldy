using System.Runtime.InteropServices;
using Goldy;

/// <summary>
/// Headless triangle via retained Scheme (render pass + readback).
/// </summary>
static class TriangleHeadless
{
    [StructLayout(LayoutKind.Sequential)]
    struct Vertex2D
    {
        public float Px, Py;
        public float R, G, B, A;
    }

    public static void Run()
    {
        Console.WriteLine("Goldy .NET Triangle (headless Scheme)");
        Console.WriteLine(new string('=', 40));

        using var instance = new Instance();
        using var device = instance.RequestAdapter().RequestDevice();
        using var ctx = device.CreateContext();
        Console.WriteLine($"Backend: {instance.BackendType}");

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
        using var readback = retainedPool.AcquireTexture(
            100,
            100,
            TextureFormat.Rgba8Unorm,
            TextureKind.Direct,
            TextureFlags.CopySrc | TextureFlags.CopyDst);

        const uint width = 100;
        const uint height = 100;

        using var scheme = new Scheme(ctx);
        using var rt = scheme.LeaseRenderTarget(width, height, TextureFormat.Rgba8Unorm);
        using (var pass = scheme.RenderPass("triangle", rt))
        {
            pass
                .WithParcel(vertexParcel, NodeAccess.Read)
                .Clear(new Color(0.1f, 0.1f, 0.2f, 1.0f))
                .SetPipeline(pipeline)
                .SetVertexBuffer(0, vertexParcel)
                .Draw(3);
        }

        scheme.CopyToTexture(rt, readback);
        using var grant = scheme.GrantReadTexture(readback);
        using var submission = scheme.Submit();
        var pixels = grant.Consume(submission);

        if (pixels.Length != width * height * 4)
            throw new InvalidOperationException($"Unexpected readback size: {pixels.Length}");

        var hasColor = false;
        for (var i = 0; i < pixels.Length; i += 4)
        {
            if (pixels[i] > 0 || pixels[i + 1] > 0 || pixels[i + 2] > 0)
            {
                hasColor = true;
                break;
            }
        }

        if (!hasColor)
            throw new InvalidOperationException("Triangle should write non-black pixels");

        Console.WriteLine($"Rendered {width}x{height}, readback OK");
        Console.WriteLine("Done!");
    }
}
