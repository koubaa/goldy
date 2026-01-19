using Goldy;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Silk.NET.Maths;
using Silk.NET.Windowing;

/// <summary>
/// Classic demoscene plasma effect.
/// Demonstrates:
/// - Using SetPushConstants() to pass buffer indices to shaders
/// - Time-based animation with uniform buffer updates
/// - Vertex-less fullscreen triangle rendering (no vertex buffer needed)
/// - Loading shaders from shared shaders directory
///
/// Run with: dotnet run -- plasma
/// </summary>
static class Plasma
{
    // Window reference
    static IWindow? _window;
    
    // Goldy resources
    static Instance? _instance;
    static Device? _device;
    static Goldy.Buffer? _uniformBuffer;
    static RenderPipeline? _pipeline;
    static ShaderModule? _shader;
    static Surface? _surface;
    
    // Animation
    static Stopwatch _stopwatch = new();

    /// <summary>
    /// Load shader from shared shaders directory.
    /// </summary>
    static string LoadShader(string name)
    {
        // Try various relative paths to find the shaders directory
        // From bin/Debug/net8.0, we need to go up to dotnet, then up to goldy root
        string[] searchPaths = [
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "shaders", name),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "shaders", name),
            Path.Combine("..", "..", "shaders", name),
            Path.Combine("shaders", name),
        ];

        foreach (var searchPath in searchPaths)
        {
            var fullPath = Path.GetFullPath(searchPath);
            if (File.Exists(fullPath))
            {
                return File.ReadAllText(fullPath);
            }
        }

        throw new FileNotFoundException($"Could not find shader: {name}. Searched in: {string.Join(", ", searchPaths)}");
    }

    public static void Run()
    {
        Console.WriteLine("Goldy Plasma Example");
        Console.WriteLine(new string('=', 60));

        var options = WindowOptions.Default with
        {
            Size = new Vector2D<int>(800, 600),
            Title = "Goldy - Plasma Effect (Bindless C#)",
            
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

        // Create uniform buffer for time (4 bytes = 1 float)
        _uniformBuffer = new Goldy.Buffer(_device, 4, BufferUsage.Uniform | BufferUsage.CopyDst);
        Console.WriteLine($"Created uniform buffer: {_uniformBuffer.Size} bytes");

        // Load and compile shader from shared shaders directory
        var shaderSrc = LoadShader("plasma.slang");
        _shader = new ShaderModule(_device, shaderSrc);
        Console.WriteLine("Compiled plasma shader (from shaders/plasma.slang)");

        // Create pipeline - vertex-less (no vertex buffer needed)
        _pipeline = new RenderPipeline(_device, _shader, _shader,
            new RenderPipelineDesc
            {
                TargetFormat = _surface.Format,
                VertexStride = 0,  // No vertex buffer
            });
        Console.WriteLine("Created render pipeline (vertex-less)");
        
        Console.WriteLine();
        Console.WriteLine("Window ready! Close or press Escape to exit.");
        Console.WriteLine();
        
        _stopwatch.Start();
    }

    static void OnRender(double delta)
    {
        if (_surface == null || _pipeline == null || _uniformBuffer == null || _window == null)
            return;
            
        var size = _window.Size;
        
        if (size.X == 0 || size.Y == 0)
            return;

        // Update time uniform
        float t = (float)_stopwatch.Elapsed.TotalSeconds;
        _uniformBuffer.Write(0, MemoryMarshal.AsBytes(new ReadOnlySpan<float>(ref t)));

        try
        {
            // Acquire next frame from swapchain
            var frame = _surface.Acquire();

            // Build render commands
            var encoder = new CommandEncoder();
            encoder.Clear(Color.Black);
            encoder.SetPipeline(_pipeline);
            // Pass buffer indices via push constants
            encoder.SetPushConstants(_uniformBuffer);
            // Vertex-less fullscreen triangle: 3 vertices, no vertex buffer
            encoder.Draw(3);

            // Render to swapchain image (zero-copy - no CPU readback!)
            frame.Render(encoder);

            // Present to screen
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
        _pipeline?.Dispose();
        _shader?.Dispose();
        _uniformBuffer?.Dispose();
        _surface?.Dispose();
        _device?.Dispose();
        _instance?.Dispose();
    }
}
