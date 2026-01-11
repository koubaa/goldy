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

// Standard fullscreen quad vertex shader
public FullscreenVarying vs_fullscreen(FullscreenVertex input) {
    FullscreenVarying output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
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
