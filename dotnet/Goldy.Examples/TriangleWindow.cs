using System.Runtime.InteropServices;
using Goldy;
using Silk.NET.GLFW;

namespace Goldy.Examples;

/// <summary>
/// Windowed triangle via GLFW + TaskGraph (offscreen RT -> swapchain blit -> present).
/// </summary>
static class TriangleWindow
{
    [StructLayout(LayoutKind.Sequential)]
    struct Vertex2D
    {
        public float Px, Py;
        public float R, G, B, A;
    }

    public static unsafe void Run()
    {
        Console.WriteLine("Goldy .NET Triangle Window (TaskGraph)");
        Console.WriteLine(new string('=', 40));
        Console.WriteLine("Press Escape or close the window to exit\n");

        var glfw = Glfw.GetApi();
        if (!glfw.Init())
            throw new InvalidOperationException("glfwInit failed");

        glfw.WindowHint(WindowHintClientApi.ClientApi, ClientApi.NoApi);
        var window = glfw.CreateWindow(800, 600, "Goldy - Animated Triangle (.NET)", null, null);
        if (window == null)
        {
            glfw.Terminate();
            throw new InvalidOperationException("glfwCreateWindow failed");
        }

        try
        {
            using var instance = new Instance();
            using var device = instance.RequestAdapter().RequestDevice();
            using var surface = GlfwSurface.Create(device, window);

            using var shader = new ShaderModule(device, ShaderModule.BuiltinVertexColor2D);
            using var pipeline = new RenderPipeline(
                device,
                shader,
                shader,
                new RenderPipelineDesc
                {
                    VertexAttributes = VertexLayouts.Vertex2D,
                    VertexStride = 24,
                    TargetFormat = surface.Format,
                });

            ReadOnlySpan<Vertex2D> vertices =
            [
                new() { Px = 0.0f, Py = -0.5f, R = 1, G = 0, B = 0, A = 1 },
                new() { Px = -0.5f, Py = 0.5f, R = 0, G = 1, B = 0, A = 1 },
                new() { Px = 0.5f, Py = 0.5f, R = 0, G = 0, B = 1, A = 1 },
            ];
            using var vertexBuffer = Goldy.Buffer.WithData(device, vertices, BufferKind.Scattered);

            var sceneRt = MakeSceneRt(device, surface);
            using var frameGraph = new TaskGraph();
            var frameCount = 0u;

            while (!glfw.WindowShouldClose(window))
            {
                glfw.GetFramebufferSize(window, out int fbWidth, out int fbHeight);
                if (fbWidth > 0 && fbHeight > 0)
                {
                    var w = (uint)fbWidth;
                    var h = (uint)fbHeight;
                    if (w != surface.Width || h != surface.Height)
                    {
                        surface.Resize(w, h);
                        sceneRt.Dispose();
                        sceneRt = MakeSceneRt(device, surface);
                    }
                }

                var t = MathF.Sin(frameCount * 0.02f) * 0.5f + 0.5f;
                var bg = new Color(0.1f + t * 0.1f, 0.1f + t * 0.05f, 0.2f + t * 0.1f, 1.0f);

                frameGraph.Clear();
                using (var pass = frameGraph.RenderPass("triangle", sceneRt))
                {
                    pass
                        .BindBuffer(vertexBuffer, NodeAccess.Read)
                        .Clear(bg)
                        .SetPipeline(pipeline)
                        .SetVertexBuffer(0, vertexBuffer)
                        .Draw(3);
                }

                var swapchain = frameGraph.DeclareSwapchainOutput();
                frameGraph.CopyRenderTargetToSwapchain(sceneRt, swapchain);

                var frame = surface.Acquire();
                surface.SubmitGraphToFrame(frameGraph, frame);
                surface.Present(frame);

                frameCount++;
                glfw.PollEvents();

                if (glfw.GetKey(window, Keys.Escape) == (int)InputAction.Press)
                    glfw.SetWindowShouldClose(window, true);
            }

            sceneRt.Dispose();
            Console.WriteLine("Done!");
        }
        finally
        {
            glfw.DestroyWindow(window);
            glfw.Terminate();
        }
    }

    static RenderTarget MakeSceneRt(Device device, Surface surface)
    {
        var w = Math.Max(surface.Width, 1u);
        var h = Math.Max(surface.Height, 1u);
        return new RenderTarget(device, w, h, surface.Format);
    }
}
