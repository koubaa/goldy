using System.Runtime.InteropServices;
using Goldy;
using Silk.NET.GLFW;

namespace Goldy.Examples;

/// <summary>
/// Windowed triangle via GLFW + retained Scheme (offscreen RT → copy_to_present → grant_present).
/// </summary>
static class TriangleWindow
{
    [StructLayout(LayoutKind.Sequential)]
    struct Vertex2D
    {
        public float Px, Py;
        public float R, G, B, A;
    }

    static (Scheme scheme, SchemeRenderTargetLease rt, PresentGrant present) RecordScheme(
        Context ctx,
        SwapchainPool swapchain,
        RenderPipeline pipeline,
        Parcel vertexParcel,
        PresentLease screen,
        Color bg)
    {
        var scheme = new Scheme(ctx);
        var rt = scheme.LeaseRenderTarget(
            Math.Max(swapchain.Width, 1u),
            Math.Max(swapchain.Height, 1u),
            swapchain.Format);
        using (var pass = scheme.RenderPass("triangle", rt))
        {
            pass
                .WithParcel(vertexParcel, NodeAccess.Read)
                .Clear(bg)
                .SetPipeline(pipeline)
                .SetVertexBuffer(0, vertexParcel)
                .Draw(3);
        }
        scheme.CopyToPresent(rt, screen);
        var present = scheme.GrantPresent(screen);
        return (scheme, rt, present);
    }

    public static unsafe void Run()
    {
        Console.WriteLine("Goldy .NET Triangle Window (Scheme + Present)");
        Console.WriteLine(new string('=', 40));
        Console.WriteLine("Press Escape or close the window to exit\n");

        var glfw = Glfw.GetApi();
        if (!glfw.Init())
            throw new InvalidOperationException("glfwInit failed");

        glfw.WindowHint(WindowHintClientApi.ClientApi, ClientApi.NoApi);
        var window = glfw.CreateWindow(800, 600, "Goldy - Triangle (.NET / Scheme)", null, null);
        if (window == null)
        {
            glfw.Terminate();
            throw new InvalidOperationException("glfwCreateWindow failed");
        }

        try
        {
            using var instance = new Instance();
            using var device = instance.RequestAdapter().RequestDevice();
            using var ctx = device.CreateContext();
            using var swapchain = GlfwSwapchainPool.Create(ctx, window);
            using var screen = swapchain.Lease();

            using var shader = new ShaderModule(device, ShaderModule.BuiltinVertexColor2D);
            using var pipeline = new RenderPipeline(
                device,
                shader,
                shader,
                new RenderPipelineDesc
                {
                    VertexAttributes = VertexLayouts.Vertex2D,
                    VertexStride = 24,
                    TargetFormat = swapchain.Format,
                });

            ReadOnlySpan<Vertex2D> vertices =
            [
                new() { Px = 0.0f, Py = -0.5f, R = 1, G = 0, B = 0, A = 1 },
                new() { Px = -0.5f, Py = 0.5f, R = 0, G = 1, B = 0, A = 1 },
                new() { Px = 0.5f, Py = 0.5f, R = 0, G = 0, B = 1, A = 1 },
            ];
            using var retainedPool = new RetainedPool(device);
            using var vertexParcel = retainedPool.AcquireBuffer(vertices, BufferKind.Scattered);

            var bg = new Color(0.1f, 0.1f, 0.2f, 1.0f);
            var (scheme, sceneRt, present) = RecordScheme(ctx, swapchain, pipeline, vertexParcel, screen, bg);

            while (!glfw.WindowShouldClose(window))
            {
                glfw.GetFramebufferSize(window, out int fbWidth, out int fbHeight);
                if (fbWidth > 0 && fbHeight > 0)
                {
                    var w = (uint)fbWidth;
                    var h = (uint)fbHeight;
                    if (w != swapchain.Width || h != swapchain.Height)
                    {
                        swapchain.Resize(w, h);
                        sceneRt.Dispose();
                        present.Dispose();
                        scheme.Dispose();
                        (scheme, sceneRt, present) = RecordScheme(ctx, swapchain, pipeline, vertexParcel, screen, bg);
                    }
                }

                using var submission = scheme.Submit();
                present.Consume(submission);

                glfw.PollEvents();

                if (glfw.GetKey(window, Keys.Escape) == (int)InputAction.Press)
                    glfw.SetWindowShouldClose(window, true);
            }

            sceneRt.Dispose();
            present.Dispose();
            scheme.Dispose();
            Console.WriteLine("Done!");
        }
        finally
        {
            glfw.DestroyWindow(window);
            glfw.Terminate();
        }
    }
}
