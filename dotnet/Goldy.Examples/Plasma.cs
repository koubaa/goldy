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
/// - Windowed rendering via Surface API
///
/// Run with: dotnet run -- plasma
/// </summary>
static class Plasma
{
    // Plasma shader
    const string PlasmaShader = """
        import goldy_exp;

        // Uniform structure  
        struct TimeUniforms {
            float time;
        };

        #if defined(__METAL__)
        // Metal: Use ParameterBlock for argument buffer support
        struct PlasmaResources {
            ConstantBuffer<TimeUniforms> uniforms;
        };
        ParameterBlock<PlasmaResources> gResources;
        #define TIME gResources.uniforms.time

        #elif defined(__SPIRV__)
        // Vulkan: Use push constants for indices + descriptor array
        import goldy_exp.buffer_indices;

        // Global descriptor array of uniform buffers
        [[vk::binding(1, 0)]] ConstantBuffer<TimeUniforms> g_UniformBuffers[];
        #define TIME g_UniformBuffers[getBufferIndex(0)].time

        #elif defined(__DX12__)
        // DX12: Root constants + ResourceDescriptorHeap
        cbuffer BufferIndices : register(b0) {
            uint uniformsIndex;
        };
        #define TIME (*DescriptorHandle<ConstantBuffer<TimeUniforms>>(uint2(uniformsIndex, 0))).time

        #endif

        [shader("vertex")]
        FullscreenVarying vs_main(FullscreenVertex input) {
            return vs_fullscreen(input);
        }

        [shader("fragment")]
        float4 fs_main(FullscreenVarying input) : SV_Target {
            float2 uv = scale_uv(input.uv, 4.0);
            float t = TIME;
            
            // Classic plasma formula
            float v = sin(uv.x + t);
            v += sin(uv.y + t);
            v += sin(uv.x + uv.y + t);
            
            float cx = uv.x + 0.5 * sin(t / 3.0);
            float cy = uv.y + 0.5 * cos(t / 2.0);
            v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
            
            v = v / 2.0;
            
            // Use rainbow palette from goldy module
            return float4(rainbow(v), 1.0);
        }
        """;

    // Window reference
    static IWindow? _window;
    
    // Goldy resources
    static Instance? _instance;
    static Device? _device;
    static Goldy.Buffer? _vertexBuffer;
    static Goldy.Buffer? _uniformBuffer;
    static RenderPipeline? _pipeline;
    static ShaderModule? _shader;
    static Surface? _surface;
    
    // Animation
    static Stopwatch _stopwatch = new();

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

        // Create fullscreen quad vertices (position + uv)
        // Using Vertex2DUv layout: x, y, u, v
        float[] vertices = [
            // Triangle 1
            -1.0f, -1.0f,  0.0f, 1.0f,  // Bottom-left
             1.0f, -1.0f,  1.0f, 1.0f,  // Bottom-right
             1.0f,  1.0f,  1.0f, 0.0f,  // Top-right
            // Triangle 2
            -1.0f, -1.0f,  0.0f, 1.0f,  // Bottom-left
             1.0f,  1.0f,  1.0f, 0.0f,  // Top-right
            -1.0f,  1.0f,  0.0f, 0.0f,  // Top-left
        ];

        _vertexBuffer = Goldy.Buffer.WithData<float>(_device, vertices, BufferUsage.Vertex);
        Console.WriteLine($"Created vertex buffer: {_vertexBuffer.Size} bytes");

        // Create uniform buffer for time (4 bytes = 1 float)
        _uniformBuffer = new Goldy.Buffer(_device, 4, BufferUsage.Uniform | BufferUsage.CopyDst);
        Console.WriteLine($"Created uniform buffer: {_uniformBuffer.Size} bytes");

        // Compile shader
        _shader = new ShaderModule(_device, PlasmaShader);
        Console.WriteLine("Compiled plasma shader");

        // Create pipeline
        _pipeline = new RenderPipeline(_device, _shader, _shader,
            new RenderPipelineDesc
            {
                TargetFormat = _surface.Format,
                VertexStride = 16, // Vertex2DUv layout: 4 floats * 4 bytes
            });
        Console.WriteLine("Created render pipeline");
        
        Console.WriteLine();
        Console.WriteLine("Window ready! Close or press Escape to exit.");
        Console.WriteLine();
        
        _stopwatch.Start();
    }

    static void OnRender(double delta)
    {
        if (_surface == null || _pipeline == null || _vertexBuffer == null || 
            _uniformBuffer == null || _window == null)
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
            encoder.SetVertexBuffer(0, _vertexBuffer);
            encoder.Draw(6);  // 6 vertices = 2 triangles = fullscreen quad

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
        _vertexBuffer?.Dispose();
        _surface?.Dispose();
        _device?.Dispose();
        _instance?.Dispose();
    }
}
