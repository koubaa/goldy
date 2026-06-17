using Goldy;
using Silk.NET.GLFW;

namespace Goldy.Examples;

/// <summary>
/// Windowed Conway's Game of Life — hybrid Scheme (compute ping-pong + render + present).
/// </summary>
static class GameOfLifeWindow
{
    const uint GridWidth = 128;
    const uint GridHeight = 128;
    const int CellCount = (int)(GridWidth * GridHeight);
    const uint WorkgroupsX = (GridWidth + 7) / 8;
    const uint WorkgroupsY = (GridHeight + 7) / 8;

    static void RunComputeStep(
        Context ctx,
        Parcel readBuf,
        Parcel writeBuf,
        ComputePipeline pipeline)
    {
        using var scheme = new Scheme(ctx);
        using (var node = scheme.ComputeNode("game_of_life", pipeline))
        {
            node
                .DeclareParcel(readBuf, NodeAccess.Read, ResourceAccess.ReadWrite)
                .DeclareParcel(writeBuf, NodeAccess.Write, ResourceAccess.Write);
            node.Dispatch(WorkgroupsX, WorkgroupsY, 1);
        }
        using var _ = scheme.Submit();
    }

    static (Scheme scheme, SchemeRenderTargetLease rt, PresentGrant present) RecordDisplayScheme(
        Context ctx,
        SwapchainPool swapchain,
        Parcel currentBuf,
        RenderPipeline renderPipeline,
        PresentLease screen)
    {
        var scheme = new Scheme(ctx);
        var rt = scheme.LeaseRenderTarget(
            Math.Max(swapchain.Width, 1u),
            Math.Max(swapchain.Height, 1u),
            swapchain.Format);
        using (var pass = scheme.RenderPass("game_of_life_render", rt))
        {
            pass
                .BindParcel(currentBuf, NodeAccess.Read)
                .Clear(Color.Black)
                .SetPipeline(renderPipeline)
                .BindResourceIndex(currentBuf.ResourceIndex(ResourceAccess.Read))
                .DrawFullscreen();
        }
        scheme.CopyToPresent(rt, screen);
        var present = scheme.GrantPresent(screen);
        return (scheme, rt, present);
    }

    public static unsafe void Run()
    {
        Console.WriteLine("Goldy .NET Game of Life (Scheme + Present)");
        Console.WriteLine(new string('=', 40));
        Console.WriteLine("Press Escape or close the window to exit\n");

        var frameLimit = DemoFrameLimit();

        var glfw = Glfw.GetApi();
        if (!glfw.Init())
            throw new InvalidOperationException("glfwInit failed");

        glfw.WindowHint(WindowHintClientApi.ClientApi, ClientApi.NoApi);
        var window = glfw.CreateWindow(800, 800, "Goldy - Game of Life (.NET / Scheme)", null, null);
        if (window == null)
        {
            glfw.Terminate();
            throw new InvalidOperationException("glfwCreateWindow failed");
        }

        var start = DateTime.UtcNow;
        var frameCount = 0u;
        var lastUpdate = DateTime.UtcNow;

        try
        {
            using var instance = new Instance();
            using var device = instance.RequestAdapter().RequestDevice();
            using var ctx = device.CreateContext();
            using var swapchain = GlfwSwapchainPool.Create(ctx, window);
            using var screen = swapchain.Lease();

            var computeSrc = ShaderPaths.Load("game_of_life.slang");
            var renderSrc = ShaderPaths.Load("game_of_life_render.slang");

            var initial = CreateInitialState();
            var zeros = new uint[CellCount];
            using var retainedPool = new RetainedPool(device);
            using var bufA = retainedPool.AcquireBuffer<uint>(initial, BufferKind.Scattered);
            using var bufB = retainedPool.AcquireBuffer<uint>(zeros, BufferKind.Scattered);

            using var computeShader = new ShaderModule(device, computeSrc);
            using var renderShader = new ShaderModule(device, renderSrc);
            using var computePipeline = new ComputePipeline(device, computeShader);
            using var renderPipeline = new RenderPipeline(
                device,
                renderShader,
                renderShader,
                new RenderPipelineDesc
                {
                    TargetFormat = swapchain.Format,
                    Topology = PrimitiveTopology.TriangleList,
                });

            var useBufferA = true;
            var currentBuf = bufA;
            var (displayScheme, sceneRt, present) = RecordDisplayScheme(
                ctx, swapchain, currentBuf, renderPipeline, screen);

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
                        displayScheme.Dispose();
                        (displayScheme, sceneRt, present) = RecordDisplayScheme(
                            ctx, swapchain, currentBuf, renderPipeline, screen);
                    }
                }

                var now = DateTime.UtcNow;
                if ((now - lastUpdate).TotalMilliseconds > 33)
                {
                    lastUpdate = now;
                    var readBuf = useBufferA ? bufA : bufB;
                    var writeBuf = useBufferA ? bufB : bufA;
                    RunComputeStep(ctx, readBuf, writeBuf, computePipeline);
                    useBufferA = !useBufferA;
                    var newBuf = useBufferA ? bufA : bufB;
                    if (!ReferenceEquals(newBuf, currentBuf))
                    {
                        currentBuf = newBuf;
                        sceneRt.Dispose();
                        present.Dispose();
                        displayScheme.Dispose();
                        (displayScheme, sceneRt, present) = RecordDisplayScheme(
                            ctx, swapchain, currentBuf, renderPipeline, screen);
                    }
                }

                using var submission = displayScheme.Submit();
                present.Consume(submission);

                frameCount++;
                glfw.PollEvents();

                if (glfw.GetKey(window, Keys.Escape) == (int)InputAction.Press)
                    glfw.SetWindowShouldClose(window, true);

                if (frameLimit is not null && frameCount >= frameLimit)
                    glfw.SetWindowShouldClose(window, true);
            }

            sceneRt.Dispose();
            present.Dispose();
            displayScheme.Dispose();
        }
        finally
        {
            var elapsed = (DateTime.UtcNow - start).TotalSeconds;
            var fps = elapsed > 0 ? frameCount / elapsed : 0;
            Console.WriteLine($"GOLDY_PERF: frames={frameCount} elapsed={elapsed:F2}s avg_fps={fps:F1}");
            glfw.DestroyWindow(window);
            glfw.Terminate();
        }

        Console.WriteLine("Done!");
    }

    static int? DemoFrameLimit()
    {
        var raw = Environment.GetEnvironmentVariable("GOLDY_DEMO_FRAMES");
        if (string.IsNullOrWhiteSpace(raw))
            return null;
        return Math.Max(1, int.Parse(raw));
    }

    static uint[] CreateInitialState()
    {
        var cells = new uint[CellCount];

        (int x, int y)[] gun =
        [
            (1, 5), (1, 6), (2, 5), (2, 6), (11, 5), (11, 6), (11, 7), (12, 4), (12, 8),
            (13, 3), (13, 9), (14, 3), (14, 9), (15, 6), (16, 4), (16, 8), (17, 5), (17, 6),
            (17, 7), (18, 6), (21, 3), (21, 4), (21, 5), (22, 3), (22, 4), (22, 5), (23, 2),
            (23, 6), (25, 1), (25, 2), (25, 6), (25, 7), (35, 3), (35, 4), (36, 3), (36, 4),
        ];

        const int offsetX = 10;
        const int offsetY = 10;
        foreach (var (x, y) in gun)
        {
            var px = x + offsetX;
            var py = y + offsetY;
            if (px < GridWidth && py < GridHeight)
                cells[py * GridWidth + px] = 1;
        }

        ulong rng = 42;
        for (var y = 60; y < 100; y++)
        {
            for (var x = 60; x < 100; x++)
            {
                rng = rng * 6364136223846793005 + 1;
                if ((rng >> 32) % 4 == 0)
                    cells[y * GridWidth + x] = 1;
            }
        }

        return cells;
    }
}
