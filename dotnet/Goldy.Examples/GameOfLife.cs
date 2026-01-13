using Goldy;

/// <summary>
/// Conway's Game of Life using Goldy compute and graphics shaders.
/// Demonstrates ping-pong buffer technique for cellular automaton.
/// Uses shared shader files from the goldy/shaders/ directory.
/// </summary>
class GameOfLife
{
    const int GRID_WIDTH = 128;
    const int GRID_HEIGHT = 128;
    const int CELL_COUNT = GRID_WIDTH * GRID_HEIGHT;

    // Path to shared shaders directory (relative to workspace root)
    static string ShaderDir => Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", "..", "shaders");

    static string LoadShader(string name) => 
        File.ReadAllText(Path.Combine(ShaderDir, name));

    static void Main(string[] args)
    {
        Console.WriteLine("Conway's Game of Life (Compute + Graphics)");
        Console.WriteLine(new string('=', 50));

        // Create device
        using var instance = new Instance();
        using var device = instance.CreateDevice(DeviceType.DiscreteGpu);
        Console.WriteLine($"Backend: {instance.BackendType}");
        Console.WriteLine($"Grid: {GRID_WIDTH}x{GRID_HEIGHT} = {CELL_COUNT} cells");
        Console.WriteLine();

        // Create initial state
        var initialState = CreateInitialState();
        int aliveCount = initialState.Count(c => c == 1);
        Console.WriteLine($"Initial state: {aliveCount} living cells");

        // Create ping-pong buffers
        using var bufferA = Goldy.Buffer.WithData<uint>(device, initialState, BufferUsage.Storage);
        using var bufferB = Goldy.Buffer.WithData<uint>(device, initialState, BufferUsage.Storage);
        Console.WriteLine($"Created buffers: {bufferA.Size} bytes each");

        // === COMPUTE PIPELINE ===

        // Compute bind group layout: read-only input, read-write output
        using var computeBindLayout = new BindGroupLayout(device,
            new BindGroupLayoutBinding(0, ShaderStages.Compute, BindingType.StorageBufferReadOnly),
            new BindGroupLayoutBinding(1, ShaderStages.Compute, BindingType.StorageBufferReadWrite));

        // A -> B: read from A, write to B
        using var computeBindGroupA = new BindGroup(device, computeBindLayout,
            new BufferBinding(0, bufferA),
            new BufferBinding(1, bufferB));

        // B -> A: read from B, write to A
        using var computeBindGroupB = new BindGroup(device, computeBindLayout,
            new BufferBinding(0, bufferB),
            new BufferBinding(1, bufferA));

        // Compile compute shader from shared file
        var computeShaderSrc = LoadShader("game_of_life.slang");
        using var computeShader = new ShaderModule(device, computeShaderSrc);
        Console.WriteLine("Compiled compute shader (game_of_life.slang)");

        using var computePipeline = new ComputePipeline(device, computeShader, computeBindLayout);
        Console.WriteLine("Created compute pipeline");

        // === RENDER PIPELINE ===

        // Render bind group layout: read-only storage buffer
        using var renderBindLayout = new BindGroupLayout(device,
            new BindGroupLayoutBinding(0, ShaderStages.Fragment, BindingType.StorageBufferReadOnly));

        // Bind groups for reading from A or B
        using var renderBindGroupA = new BindGroup(device, renderBindLayout,
            new BufferBinding(0, bufferA));
        using var renderBindGroupB = new BindGroup(device, renderBindLayout,
            new BufferBinding(0, bufferB));

        // Compile render shader from shared file
        var renderShaderSrc = LoadShader("game_of_life_render.slang");
        using var renderShader = new ShaderModule(device, renderShaderSrc);
        Console.WriteLine("Compiled render shader (game_of_life_render.slang)");

        using var renderPipeline = new RenderPipeline(device, renderShader, renderShader,
            new RenderPipelineDesc
            {
                TargetFormat = TextureFormat.Rgba8Unorm,
                VertexStride = 16, // Vertex2DUv layout
                BindGroupLayouts = [renderBindLayout],
            });
        Console.WriteLine("Created render pipeline");

        // Create render target
        using var target = new RenderTarget(device, 512, 512, TextureFormat.Rgba8Unorm);
        Console.WriteLine($"Created render target: {target.Width}x{target.Height}");
        Console.WriteLine();

        // === SIMULATION LOOP ===

        int numGenerations = 100;
        bool useBufferA = true;

        Console.WriteLine($"Running {numGenerations} generations...");
        var startTime = DateTime.Now;

        for (int gen = 0; gen < numGenerations; gen++)
        {
            // === COMPUTE PASS: Update simulation ===
            var computeEncoder = new ComputeEncoder();
            computeEncoder.SetPipeline(computePipeline);

            // Choose which bind group based on current buffer
            if (useBufferA)
                computeEncoder.SetBindGroup(0, computeBindGroupA);  // A -> B
            else
                computeEncoder.SetBindGroup(0, computeBindGroupB);  // B -> A

            // Dispatch workgroups (8x8 threads per group)
            uint workgroupsX = (GRID_WIDTH + 7) / 8;
            uint workgroupsY = (GRID_HEIGHT + 7) / 8;
            computeEncoder.Dispatch(workgroupsX, workgroupsY, 1);
            computeEncoder.Execute(device);

            // Toggle buffer for next frame
            useBufferA = !useBufferA;

            // === RENDER PASS: Visualize the grid ===
            var encoder = new CommandEncoder();
            encoder.Clear(Color.Black);
            encoder.SetPipeline(renderPipeline);

            // Read from the buffer that was just written to
            if (useBufferA)
                encoder.SetBindGroup(0, renderBindGroupA);
            else
                encoder.SetBindGroup(0, renderBindGroupB);

            encoder.Draw(3); // Fullscreen triangle
            target.Render(encoder);

            if (gen % 10 == 0)
                Console.WriteLine($"  Generation {gen}");
        }

        var elapsed = DateTime.Now - startTime;
        double fps = numGenerations / elapsed.TotalSeconds;

        Console.WriteLine();
        Console.WriteLine($"Completed {numGenerations} generations in {elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"Performance: {fps:F1} generations/second");

        // Read final frame
        var pixels = target.ReadToCpu();
        Console.WriteLine($"Final frame: {pixels.Length} bytes ({target.Width}x{target.Height} RGBA)");

        // Count living cells at end (by reading buffer)
        Console.WriteLine("\nDone!");
    }

    static uint[] CreateInitialState()
    {
        var cells = new uint[CELL_COUNT];

        // Gosper Glider Gun (creates infinite gliders)
        var gun = new (int x, int y)[]
        {
            (1, 5), (1, 6), (2, 5), (2, 6),
            (11, 5), (11, 6), (11, 7),
            (12, 4), (12, 8),
            (13, 3), (13, 9),
            (14, 3), (14, 9),
            (15, 6),
            (16, 4), (16, 8),
            (17, 5), (17, 6), (17, 7),
            (18, 6),
            (21, 3), (21, 4), (21, 5),
            (22, 3), (22, 4), (22, 5),
            (23, 2), (23, 6),
            (25, 1), (25, 2), (25, 6), (25, 7),
            (35, 3), (35, 4),
            (36, 3), (36, 4),
        };

        // Place glider gun
        int offsetX = 10, offsetY = 10;
        foreach (var (x, y) in gun)
        {
            int px = x + offsetX;
            int py = y + offsetY;
            if (px < GRID_WIDTH && py < GRID_HEIGHT)
                cells[py * GRID_WIDTH + px] = 1;
        }

        // Add some random cells in the lower right
        var rng = new Random(42);
        for (int y = 60; y < 100; y++)
        {
            for (int x = 60; x < 100; x++)
            {
                if (rng.Next(4) == 0 && x < GRID_WIDTH && y < GRID_HEIGHT)
                    cells[y * GRID_WIDTH + x] = 1;
            }
        }

        return cells;
    }
}

