using System.Runtime.InteropServices;
using Goldy;

namespace Goldy.Examples;

/// <summary>
/// Headless Conway's Game of Life — hybrid Scheme (compute + render + readback).
/// </summary>
static class GameOfLifeHeadless
{
    const uint GridWidth = 128;
    const uint GridHeight = 128;
    const int CellCount = (int)(GridWidth * GridHeight);
    const uint WorkgroupsX = (GridWidth + 7) / 8;
    const uint WorkgroupsY = (GridHeight + 7) / 8;

    public static void Run()
    {
        Console.WriteLine("Goldy .NET Game of Life (headless Scheme)");
        Console.WriteLine(new string('=', 40));
        RunCore();
    }

    static void RunCore()
    {
        using var instance = new Instance();
        using var device = instance.RequestAdapter().RequestDevice();
        using var ctx = device.CreateContext();

        var initial = InitialCells();
        using var retainedPool = new RetainedPool(device);
        using var record = retainedPool.Record();
        record.EmplaceField<uint>("a", initial);
        record.EmplaceField<uint>("b", initial);
        using var cells = record.Build(retainedPool);

        var computeSrc = ShaderPaths.Load("game_of_life.slang");
        var renderSrc = ShaderPaths.Load("game_of_life_render.slang");

        using var computeShader = new ShaderModule(device, computeSrc);
        using var renderShader = new ShaderModule(device, renderSrc);
        using var computePipeline = new ComputePipeline(device, computeShader);
        using var renderPipeline = new RenderPipeline(
            device,
            renderShader,
            renderShader,
            new RenderPipelineDesc
            {
                TargetFormat = TextureFormat.Rgba8Unorm,
                Topology = PrimitiveTopology.TriangleList,
            });

        using var readback = retainedPool.AcquireTexture(
            GridWidth,
            GridHeight,
            TextureFormat.Rgba8Unorm,
            TextureKind.Direct,
            TextureFlags.CopySrc | TextureFlags.CopyDst);

        using var read = cells.Field(0);
        using var write = cells.Field(1);
        using var scheme = new Scheme(ctx);
        using (var node = scheme.ComputeNode("game_of_life", computePipeline))
        {
            node
                .WithParcel(read, NodeAccess.Read)
                .WithParcel(write, NodeAccess.Write);
            node.Dispatch(WorkgroupsX, WorkgroupsY, 1);
        }

        var rt = scheme.LeaseRenderTarget(GridWidth, GridHeight, TextureFormat.Rgba8Unorm);
        using (var current = cells.Field(1))
        using (var pass = scheme.RenderPass("game_of_life_render", rt))
        {
            pass
                .WithParcel(current, NodeAccess.Read)
                .Clear(Color.Black)
                .SetPipeline(renderPipeline)
                .DrawFullscreen();
        }

        scheme.CopyToTexture(rt, readback);
        var grant = scheme.GrantReadTexture(readback);
        using var submission = scheme.Submit();
        var pixels = grant.Consume(submission);

        var cellsOut = MemoryMarshal.Cast<byte, uint>(cells.UnitReadToCpu(1, device));
        var live = 0;
        foreach (var cell in cellsOut)
        {
            if (cell == 1)
                live++;
        }
        if (live != 4)
            throw new InvalidOperationException($"still-life block should remain 4 live cells, got {live}");

        var stride = (int)GridWidth * 4;
        var cx = (int)GridWidth / 2;
        var cy = (int)GridHeight / 2;
        var g = pixels[cy * stride + cx * 4 + 1];
        if (g <= 100)
            throw new InvalidOperationException($"center pixel should show alive cells (g={g})");

        Console.WriteLine($"Simulation + render OK ({GridWidth}x{GridHeight}, g={g} at center)");
        Console.WriteLine("Done!");
    }

    static uint[] InitialCells()
    {
        var cells = new uint[CellCount];
        foreach (var y in new[] { 63, 64 })
        {
            foreach (var x in new[] { 63, 64 })
                cells[y * GridWidth + x] = 1;
        }
        return cells;
    }
}
