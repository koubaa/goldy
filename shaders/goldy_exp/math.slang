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

/// `a * b` rounded to float on its own: the device must not contract it into a
/// multiply-add with the operation that reads it.
public float goldy_exact_mul(float a, float b) {
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            %r:$$float = OpFMul $a $b;
            OpDecorate %r NoContraction;
            result:$$float = OpCopyObject %r
        };
    case cuda: __intrinsic_asm "__fmul_rn";
    case hlsl:
        precise float r = a * b;
        return r;
    default:
        return a * b;
    }
}

/// Logistic sigmoid.
///
/// `|x| < 0.5` uses the odd Taylor series through `x^11`. Larger `|x|` uses the
/// split `exp` form, which never evaluates `exp` of a large positive argument,
/// so large inputs saturate to 0 or 1.
public float goldy_sigmoid(float x) {
    float a = abs(x);
    if (a < 0.5) {
        float x2 = x * x;
        return 0.5 + x * (1.0 / 4.0 + x2 * (-1.0 / 48.0 + x2 * (1.0 / 480.0 + x2 * (-17.0 / 80640.0 + x2 * (31.0 / 1451520.0 + x2 * (-691.0 / 319334400.0))))));
    }
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    float e = exp(x);
    return e / (1.0 + e);
}

/// Hyperbolic tangent.
///
/// `|x| < 0.5` uses the odd Taylor series through `x^11`. The identity
/// `(1 - exp(-2|x|)) / (1 + exp(-2|x|))` cancels in f32 near zero. `|x| >= 10`
/// is already ±1 in f32, including the softcaps at 25 and 30, so those inputs
/// skip `exp`.
public float goldy_tanh(float x) {
    float a = abs(x);
    if (a < 0.5) {
        float x2 = x * x;
        return x * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 + x2 * (-17.0 / 315.0 + x2 * (62.0 / 2835.0 + x2 * (-1382.0 / 155925.0))))));
    }
    if (a >= 10.0) {
        if (x < 0.0) {
            return -1.0;
        }
        return 1.0;
    }
    float e = exp(-2.0 * a);
    float t = (1.0 - e) / (1.0 + e);
    if (x < 0.0) {
        return -t;
    }
    return t;
}

/// `a / b` rounded to float on its own: the device must not rewrite it, as into a
/// multiply by the reciprocal of a constant divisor.
public float goldy_exact_div(float a, float b) {
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            %r:$$float = OpFDiv $a $b;
            OpDecorate %r NoContraction;
            result:$$float = OpCopyObject %r
        };
    case cuda: __intrinsic_asm "__fdiv_rn";
    case hlsl:
        precise float r = a / b;
        return r;
    default:
        return a / b;
    }
}

// Positive modulo — HLSL fmod returns negative for negative inputs
public float positive_mod(float x, float m) {
    return fmod(fmod(x, m) + m, m);
}

public float2 positive_mod(float2 x, float2 m) {
    return float2(
        fmod(fmod(x.x, m.x) + m.x, m.x),
        fmod(fmod(x.y, m.y) + m.y, m.y)
    );
}

// Smoothstep helper
public float smootherstep(float edge0, float edge1, float x) {
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}
