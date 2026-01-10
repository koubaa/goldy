// Interactive Mandelbrot fractal explorer
// Uses zoom and center uniforms for navigation

struct VertexInput {
    float2 position : POSITION;
    float2 center : TEXCOORD0;
    float zoom : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 complex_coord : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    // Map UV to complex plane around center with zoom
    output.complex_coord = input.center + (input.uv - 0.5) * 4.0 / input.zoom;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 c = input.complex_coord;
    float2 z = float2(0.0, 0.0);
    uint i = 0;
    const uint max_iter = 256;
    
    for (i = 0; i < max_iter; i++) {
        if (dot(z, z) > 4.0) break;
        // z = z^2 + c
        z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    }
    
    if (i >= max_iter) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    
    // Smooth coloring
    float t = float(i) / float(max_iter);
    float r = sin(t * 5.0) * 0.5 + 0.5;
    float g = sin(t * 7.0 + 1.0) * 0.5 + 0.5;
    float b = sin(t * 11.0 + 2.0) * 0.5 + 0.5;
    
    return float4(r, g, b, 1.0);
}

