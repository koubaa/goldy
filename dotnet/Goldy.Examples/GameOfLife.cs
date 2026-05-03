using Goldy;
using Silk.NET.Maths;
using Silk.NET.Windowing;

/// <summary>
/// Conway's Game of Life - Compute + Graphics Example
/// 
/// Demonstrates:
/// - Compute shader running cellular automaton rules
/// - Graphics shader rendering the grid
/// - Ping-pong buffer technique for in-place updates
/// 
/// Uses shared shader files from the goldy/shaders/ directory.
/// 
/// Run with: dotnet run -- gameoflife
/// </summary>
static class GameOfLife
{
    const int GRID_WIDTH = 128;
    const int GRID_HEIGHT = 128;
    const int CELL_COUNT = GRID_WIDTH * GRID_HEIGHT;

    // Path to shared shaders directory (relative to workspace root)
    static string ShaderDir => Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", "..", "shaders");

    static string LoadShader(string name) => 
        File.ReadAllText(Path.Combine(ShaderDir, name));

    // Window reference
    static IWindow? _window;
    
    // Goldy resources
    static Instance? _instance;
    static Device? _device;
    static Surface? _surface;
    static ComputePipeline? _computePipeline;
    static RenderPipeline? _renderPipeline;
    static ShaderModule? _computeShader;
    static ShaderModule? _renderShader;
    
    // Ping-pong buffers
    static Goldy.Buffer? _bufferA;
    static Goldy.Buffer? _bufferB;
    
    // State: true = A is current (read from A, write to B)
    static bool _useBufferA = true;
    static DateTime _lastUpdate = DateTime.Now;

    public static void Run()
    {
        Console.WriteLine("Conway's Game of Life");
        Console.WriteLine(new string('=', 50));

        var options = WindowOptions.Default with
        {
            Size = new Vector2D<int>(800, 800),
            Title = "Game of Life (C#)",
            
            // CRITICAL: No graphics context - Goldy creates its own Vulkan/DX12/Metal context
            API = GraphicsAPI.None,
            
            // Goldy handles presentation via Surface.Present()
            ShouldSwapAutomatically = false,
            
            // VSync handled by Goldy's swapchain
            VSync = false,
        };

        _window = Window.Create(options);
        
        _window.Load += OnLoad;
        _window.Render += OnRender;
        _window.Resize += OnResize;
        _window.Closing += OnClosing;
        
        _window.Run();
        
        _window.Dispose();
    }

    static void OnLoad()
    {
        Console.WriteLine("Initializing GPU...");
        
        _instance = new Instance();
        Console.WriteLine($"Backend: {_instance.BackendType}");

        _device = _instance.CreateDevice(DeviceType.DiscreteGpu);
        Console.WriteLine($"Using adapter {_device.AdapterId}");
        Console.WriteLine($"Grid: {GRID_WIDTH}x{GRID_HEIGHT} = {CELL_COUNT} cells");

        // Get native window handle
        var native = _window!.Native!;
        
        nint hwnd;
        if (OperatingSystem.IsWindows())
        {
            hwnd = native.Win32!.Value.Hwnd;
            Console.WriteLine($"Creating surface from HWND: 0x{hwnd:X}");
            _surface = Surface.CreateWin32(_device, hwnd);
        }
        else
        {
            throw new PlatformNotSupportedException(
                "Currently only Windows is supported. macOS and Linux support coming soon.");
        }
        
        Console.WriteLine($"Surface: {_surface.Width}x{_surface.Height}, format: {_surface.Format}");

        // Create initial state
        var initialState = CreateInitialState();
        int aliveCount = initialState.Count(c => c == 1);
        Console.WriteLine($"Initial state: {aliveCount} living cells");

        // Create ping-pong buffers
        _bufferA = Goldy.Buffer.WithData<uint>(_device, initialState, BufferUsage.Storage);
        _bufferB = Goldy.Buffer.WithData<uint>(_device, initialState, BufferUsage.Storage);
        Console.WriteLine($"Created buffers: {_bufferA.Size} bytes each");

        // === COMPUTE PIPELINE ===
        var computeShaderSrc = LoadShader("game_of_life.slang");
        _computeShader = new ShaderModule(_device, computeShaderSrc);
        Console.WriteLine("Compiled compute shader (game_of_life.slang)");

        // Create compute pipeline
        _computePipeline = new ComputePipeline(_device, _computeShader);
        Console.WriteLine("Created compute pipeline");

        // === RENDER PIPELINE ===
        var renderShaderSrc = LoadShader("game_of_life_render.slang");
        _renderShader = new ShaderModule(_device, renderShaderSrc);
        Console.WriteLine("Compiled render shader (game_of_life_render.slang)");

        // Create render pipeline - vertex-less (no vertex buffer needed)
        _renderPipeline = new RenderPipeline(_device, _renderShader, _renderShader,
            new RenderPipelineDesc
            {
                TargetFormat = _surface.Format,
                VertexStride = 0,  // Vertex-less rendering
            });
        Console.WriteLine("Created render pipeline (vertex-less)");

        Console.WriteLine();
        Console.WriteLine("Features Gosper Glider Gun + random cells");
        Console.WriteLine("Window ready! Close or press Escape to exit.");
        Console.WriteLine();
    }

    static void OnRender(double delta)
    {
        if (_surface == null || _computePipeline == null || _renderPipeline == null ||
            _bufferA == null || _bufferB == null || _device == null || _window == null)
            return;
            
        var size = _window.Size;
        
        if (size.X == 0 || size.Y == 0)
            return;

        // Update simulation ~30 times per second
        var now = DateTime.Now;
        var shouldUpdate = (now - _lastUpdate).TotalMilliseconds > 33;

        if (shouldUpdate)
        {
            _lastUpdate = now;

            // === COMPUTE PASS: Update simulation ===
            var computeEncoder = new ComputeEncoder();
            computeEncoder.SetPipeline(_computePipeline);

            // Bind resource slots
            // Order matters: [current_state, next_state] matching shader slots
            if (_useBufferA)
            {
                // A -> B: read from A, write to B
                computeEncoder.BindResources(_bufferA, _bufferB);
            }
            else
            {
                // B -> A: read from B, write to A
                computeEncoder.BindResources(_bufferB, _bufferA);
            }

            // Dispatch workgroups (8x8 threads per group)
            uint workgroupsX = (GRID_WIDTH + 7) / 8;
            uint workgroupsY = (GRID_HEIGHT + 7) / 8;
            computeEncoder.Dispatch(workgroupsX, workgroupsY, 1);
            computeEncoder.Execute(_device);

            // Toggle buffer for next frame
            _useBufferA = !_useBufferA;
        }

        try
        {
            // === RENDER PASS: Visualize the grid ===
            var frame = _surface.Acquire();

            var encoder = new CommandEncoder();
            encoder.Clear(Color.Black);
            encoder.SetPipeline(_renderPipeline);

            // Read from the buffer that is now "current"
            // After the swap, _useBufferA points to the newly computed buffer
            if (_useBufferA)
                encoder.BindResources(_bufferA);
            else
                encoder.BindResources(_bufferB);

            encoder.Draw(3); // Fullscreen triangle
            frame.Render(encoder);

            _surface.Present(frame);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Render error: {ex.Message}");
        }
    }

    static void OnResize(Vector2D<int> newSize)
    {
        if (newSize.X > 0 && newSize.Y > 0)
        {
            _surface?.Resize((uint)newSize.X, (uint)newSize.Y);
        }
    }

    static void OnClosing()
    {
        // Dispose in reverse order of creation
        _renderPipeline?.Dispose();
        _computePipeline?.Dispose();
        _renderShader?.Dispose();
        _computeShader?.Dispose();
        _bufferB?.Dispose();
        _bufferA?.Dispose();
        _surface?.Dispose();
        _device?.Dispose();
        _instance?.Dispose();
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
