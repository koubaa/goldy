using Goldy;
using Silk.NET.Maths;
using Silk.NET.Windowing;

/// <summary>
/// Triangle example - render a colored triangle in an interactive window.
/// 
/// Demonstrates the Surface API for zero-copy GPU presentation using Silk.NET.Windowing.
/// 
/// Run with: dotnet run -- triangle
/// </summary>
static class Triangle
{
    const string TriangleShader = """
        import goldy_exp;
        
        struct VertexInput {
            float2 pos : POSITION;
            float4 col : COLOR;
        };
        
        struct VertexOutput {
            float4 pos : SV_Position;
            float4 col : COLOR;
        };
        
        [shader("vertex")]
        VertexOutput vs_main(VertexInput input) {
            VertexOutput output;
            output.pos = float4(input.pos, 0.0, 1.0);
            output.col = input.col;
            return output;
        }
        
        [shader("fragment")]
        float4 fs_main(VertexOutput input) : SV_Target {
            return input.col;
        }
        """;

    // Window reference
    static IWindow? _window;
    
    // Goldy resources
    static Instance? _instance;
    static Device? _device;
    static Goldy.Buffer? _vertexBuffer;
    static RenderPipeline? _pipeline;
    static ShaderModule? _shader;
    static Surface? _surface;
    
    // Animation
    static ulong _frameCount;
    
    public static void Run()
    {
        Console.WriteLine("Goldy Triangle Example (Surface API + Silk.NET.Windowing)");
        Console.WriteLine(new string('=', 60));

        var options = WindowOptions.Default with
        {
            Size = new Vector2D<int>(800, 600),
            Title = "Goldy - Animated Triangle (C#)",
            
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

        var adapters = _instance.EnumerateAdapters();
        Console.WriteLine($"Found {adapters.Length} GPU adapter(s):");
        foreach (var adapter in adapters)
            Console.WriteLine($"  [{adapter.Id}] {adapter.Name} ({adapter.DeviceType})");

        _device = _instance.CreateDevice(DeviceType.DiscreteGpu);
        Console.WriteLine($"Using adapter {_device.AdapterId}");

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

        // Create vertex buffer with a triangle (matching Rust example)
        // Layout: x, y, r, g, b, a (6 floats = 24 bytes per vertex)
        float[] vertices = [
             0.0f, -0.5f,  1.0f, 0.0f, 0.0f, 1.0f,  // Top (red)
            -0.5f,  0.5f,  0.0f, 1.0f, 0.0f, 1.0f,  // Bottom-left (green)
             0.5f,  0.5f,  0.0f, 0.0f, 1.0f, 1.0f,  // Bottom-right (blue)
        ];

        _vertexBuffer = Goldy.Buffer.WithData<float>(_device, vertices, BufferUsage.Vertex);
        Console.WriteLine($"Created vertex buffer: {_vertexBuffer.Size} bytes");
        
        // Compile shader and create pipeline using surface's actual format
        _shader = new ShaderModule(_device, TriangleShader);
        Console.WriteLine("Compiled triangle shader");
        
        var pipelineDesc = new RenderPipelineDesc
        {
            TargetFormat = _surface.Format,
            VertexStride = 24, // 6 floats * 4 bytes
        };
        _pipeline = new RenderPipeline(_device, _shader, _shader, pipelineDesc);
        Console.WriteLine("Created render pipeline");
        
        Console.WriteLine();
        Console.WriteLine("Window ready! Close or press Escape to exit.");
        Console.WriteLine();
    }

    static void OnRender(double delta)
    {
        if (_surface == null || _pipeline == null || _vertexBuffer == null || _window == null)
            return;
            
        var size = _window.Size;
        
        if (size.X == 0 || size.Y == 0)
            return;

        // Animate background color (matching Rust example)
        float t = MathF.Sin((float)(_frameCount * 0.02f)) * 0.5f + 0.5f;
        var bgColor = new Color(
            0.1f + t * 0.1f,
            0.1f + t * 0.05f,
            0.2f + t * 0.1f,
            1.0f
        );

        try
        {
            // Acquire next frame from swapchain
            var frame = _surface.Acquire();

            // Build render commands
            var encoder = new CommandEncoder();
            encoder.Clear(bgColor);
            encoder.SetPipeline(_pipeline);
            encoder.SetVertexBuffer(0, _vertexBuffer);
            encoder.Draw(3);

            // Render to swapchain image (zero-copy - no CPU readback!)
            frame.Render(encoder);

            // Present to screen
            _surface.Present(frame);

            _frameCount++;
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
        _pipeline?.Dispose();
        _shader?.Dispose();
        _vertexBuffer?.Dispose();
        _surface?.Dispose();
        _device?.Dispose();
        _instance?.Dispose();
    }
}
