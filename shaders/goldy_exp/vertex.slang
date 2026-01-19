// Vertex formats and common vertex shaders for Goldy (EXPERIMENTAL)
// Part of the goldy_exp module

implementing goldy_exp;

// ============================================================================
// Fullscreen quad (position + UV)
// Used by: plasma, mandelbrot, tunnel, metaballs, checkerboard, gradient
// ============================================================================

public struct FullscreenVertex {
    public float2 position : POSITION;
    public float2 uv : TEXCOORD0;
};

public struct FullscreenVarying {
    public float4 position : SV_Position;
    public float2 uv : TEXCOORD0;
};

// Standard fullscreen quad vertex shader (requires vertex buffer)
public FullscreenVarying vs_fullscreen(FullscreenVertex input) {
    FullscreenVarying output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

// Vertex-less fullscreen triangle (no vertex buffer needed)
// Generates a single triangle that covers the entire screen from just the vertex index.
// More efficient than a quad (3 vertices vs 6) and eliminates vertex buffer overhead.
// Usage: draw(3) with no vertex buffer bound
//
// UV convention: (0,0) at bottom-left, (1,1) at top-right
// This matches standard mathematical convention and Metal's Y-up NDC.
// Vulkan backend will flip Y via negative viewport height when implemented.
public FullscreenVarying vs_fullscreen_triangle(uint vertex_id) {
    FullscreenVarying output;
    
    // Generate UV coords from vertex index using bit manipulation:
    // vertex 0: (0, 0), vertex 1: (2, 0), vertex 2: (0, 2)
    // This creates a triangle that extends beyond clip space to cover the screen
    float2 uv = float2(
        float((vertex_id << 1u) & 2u),
        float(vertex_id & 2u)
    );
    
    // Convert UV to clip space position: [0,2] -> [-1,3]
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    
    // UV matches position mapping directly
    output.uv = uv;
    
    return output;
}

// ============================================================================
// Colored vertices (position + color)
// Used by: digital_clock, vertex_color_2d, particles, bouncing_lines
// ============================================================================

public struct ColoredVertex {
    public float2 position : POSITION;
    public float4 color : COLOR;
};

public struct ColoredVarying {
    public float4 position : SV_Position;
    public float4 color : COLOR;
};

// Standard colored vertex shader
public ColoredVarying vs_colored(ColoredVertex input) {
    ColoredVarying output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

// Pass-through fragment shader for colored vertices
public float4 fs_colored(ColoredVarying input) : SV_Target {
    return input.color;
}

// ============================================================================
// Fullscreen with time (position + UV + time attribute)
// Used by: starfield
// ============================================================================

public struct FullscreenTimeVertex {
    public float2 position : POSITION;
    public float2 uv : TEXCOORD0;
    public float time : TEXCOORD1;
};

public struct FullscreenTimeVarying {
    public float4 position : SV_Position;
    public float2 uv : TEXCOORD0;
    public float time : TEXCOORD1;
};

public FullscreenTimeVarying vs_fullscreen_time(FullscreenTimeVertex input) {
    FullscreenTimeVarying output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.time = input.time;
    return output;
}
