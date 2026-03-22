// Color utilities for Goldy shaders (EXPERIMENTAL)
// Part of the goldy_exp module

implementing goldy_exp;

// Rainbow palette from normalized value [0,1]
public float3 rainbow(float t) {
    float r = sin(t * PI) * 0.5 + 0.5;
    float g = sin(t * PI + 2.094) * 0.5 + 0.5;
    float b = sin(t * PI + 4.188) * 0.5 + 0.5;
    return float3(r, g, b);
}

// Smooth palette with customizable frequencies and phases
public float3 palette(float t, float3 freq, float3 phase) {
    return sin(t * freq + phase) * 0.5 + 0.5;
}

// Simple grayscale to heat colormap
public float3 heat(float t) {
    return float3(
        clamp(t * 3.0, 0.0, 1.0),
        clamp(t * 3.0 - 1.0, 0.0, 1.0),
        clamp(t * 3.0 - 2.0, 0.0, 1.0)
    );
}

// HSV to RGB conversion
public float3 hsv_to_rgb(float3 hsv) {
    float4 K = float4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    float3 p = abs(frac(hsv.xxx + K.xyz) * 6.0 - K.www);
    return hsv.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);
}

// RGB to grayscale (luminance)
public float luminance(float3 rgb) {
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

// Gamma correction
public float3 gamma_correct(float3 color, float gamma) {
    return pow(color, float3(1.0 / gamma, 1.0 / gamma, 1.0 / gamma));
}

// NOTE: unpack4x8unorm, pack4x8unorm, premul_alpha are defined in
// ekrano_shared.slang to avoid Slang cross-module ambiguity.
// They are NOT part of goldy_exp.

