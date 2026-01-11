// Math utilities for Goldy shaders (EXPERIMENTAL)
// Part of the goldy_exp module

implementing goldy_exp;

// Constants
public static const float PI = 3.14159265359;
public static const float TAU = 6.28318530718;

// Hash functions for procedural generation
public float hash(float p) {
    return frac(sin(p * 127.1) * 43758.5453);
}

public float hash2(float2 p) {
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

// UV utilities
public float2 center_uv(float2 uv) {
    return (uv - 0.5) * 2.0;
}

public float2 scale_uv(float2 uv, float scale) {
    return uv * scale;
}

// Polar coordinate conversion
public float2 to_polar(float2 uv) {
    float dist = length(uv);
    float angle = atan2(uv.y, uv.x);
    return float2(dist, angle);
}

// Smoothstep helper
public float smootherstep(float edge0, float edge1, float x) {
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

