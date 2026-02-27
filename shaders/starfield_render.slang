// Starfield rendering shader
// Visualizes different celestial objects using instancing

import goldy_exp;

static const float STAR_TYPE_NORMAL = 0.0;
static const float STAR_TYPE_GALAXY = 1.0;
static const float STAR_TYPE_QUASAR = 2.0;
static const float STAR_TYPE_WHITE_DWARF = 3.0;

struct Star {
    float x;
    float y;
    float z;
    float star_type;
};

// Dynamic buffer selection - Rust: set_push_constants(&[stars_slot])
#if defined(__METAL__) && !defined(__METAL_BINDLESS__)
// Metal Tier 1 fallback: traditional bindings
StructuredBuffer<Star> stars : register(t0);
#define STARS stars
#else
// All other platforms: use unified dynamic access
#define STARS goldy_dyn_scattered<Star>(0)
#endif

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD0;
    float star_type : TEXCOORD1;
};

static const float2 quadVerts[6] = {
    float2(-1, -1), float2( 1, -1), float2( 1,  1),
    float2(-1, -1), float2( 1,  1), float2(-1,  1)
};

[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    VSOutput output;
    
    Star s = STARS[instanceID];
    
    float2 screenPos = float2(s.x, s.y) / max(s.z, 0.01);
    float baseSize = 0.005 + 0.02 * (1.0 - s.z);
    
    float size = baseSize;
    float3 baseColor = float3(1.0, 1.0, 1.0);
    
    float hash1 = frac(sin(float(instanceID) * 127.1) * 43758.5453);
    float hash2 = frac(sin(float(instanceID) * 269.5) * 43758.5453);
    
    if (s.star_type == STAR_TYPE_NORMAL) {
        float temp = hash1;
        if (temp < 0.15) {
            baseColor = float3(0.7, 0.85, 1.0);
        } else if (temp < 0.4) {
            baseColor = float3(1.0, 1.0, 0.95);
        } else if (temp < 0.7) {
            baseColor = float3(1.0, 0.95, 0.8);
        } else if (temp < 0.9) {
            baseColor = float3(1.0, 0.8, 0.6);
        } else {
            baseColor = float3(1.0, 0.6, 0.5);
        }
    }
    else if (s.star_type == STAR_TYPE_GALAXY) {
        size = baseSize * 2.5;
        
        float hue = hash1 * 6.0;
        int colorIdx = int(hue);
        float blend = frac(hue);
        
        float3 galaxyColors[7] = {
            float3(1.0, 0.5, 0.7),
            float3(0.7, 0.5, 1.0),
            float3(0.5, 0.7, 1.0),
            float3(0.5, 1.0, 0.9),
            float3(0.7, 1.0, 0.6),
            float3(1.0, 0.9, 0.5),
            float3(1.0, 0.5, 0.7)
        };
        
        baseColor = lerp(galaxyColors[colorIdx], galaxyColors[colorIdx + 1], blend);
    }
    else if (s.star_type == STAR_TYPE_QUASAR) {
        size = baseSize * 1.8;
        float quasarHue = hash1;
        if (quasarHue < 0.33) {
            baseColor = float3(0.4, 1.0, 1.0);
        } else if (quasarHue < 0.66) {
            baseColor = float3(0.7, 0.6, 1.0);
        } else {
            baseColor = float3(1.0, 0.7, 1.0);
        }
    }
    else if (s.star_type == STAR_TYPE_WHITE_DWARF) {
        size = baseSize * 0.7;
        baseColor = lerp(float3(0.85, 0.9, 1.0), float3(1.0, 1.0, 1.0), hash1);
    }
    
    float2 localPos = quadVerts[vertexID] * size;
    
    output.position = float4(screenPos + localPos, 0.0, 1.0);
    
    float brightness = 1.0 - s.z;
    output.color = float4(baseColor * brightness, 1.0);
    
    output.uv = quadVerts[vertexID];
    output.star_type = s.star_type;
    
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    float2 uv = input.uv;
    float dist = length(uv);
    float4 color = input.color;
    
    if (input.star_type == STAR_TYPE_NORMAL) {
        float2 absUV = abs(uv);
        
        float hBar = max(0.0, 1.0 - absUV.y * 4.0) * (1.0 - smoothstep(0.0, 1.0, absUV.x));
        float vBar = max(0.0, 1.0 - absUV.x * 4.0) * (1.0 - smoothstep(0.0, 1.0, absUV.y));
        
        float cross = max(hBar, vBar);
        float core = 1.0 - smoothstep(0.0, 0.25, dist);
        
        float glow = max(cross * 0.8, core);
        color.rgb *= glow;
        color.a = glow;
    }
    else if (input.star_type == STAR_TYPE_GALAXY) {
        float2 rotUV = float2(uv.x * 0.7 + uv.y * 0.3, uv.y * 0.8 - uv.x * 0.2);
        float ellipseDist = length(rotUV * float2(1.0, 1.5));
        float glow = 1.0 - smoothstep(0.0, 0.8, ellipseDist);
        float angle = atan2(uv.y, uv.x);
        float spiral = sin(angle * 2.0 + dist * 6.0) * 0.15 + 0.85;
        color.rgb *= glow * spiral;
        color.a = glow * 0.9;
    }
    else if (input.star_type == STAR_TYPE_QUASAR) {
        float core = 1.0 - smoothstep(0.0, 0.3, dist);
        float jet = max(0.0, 1.0 - abs(uv.x) * 4.0) * (1.0 - smoothstep(0.0, 1.0, abs(uv.y)));
        float glow = max(core, jet * 0.5);
        color.rgb *= glow * 1.5;
        color.a = glow;
    }
    else if (input.star_type == STAR_TYPE_WHITE_DWARF) {
        float glow = 1.0 - smoothstep(0.0, 0.5, dist);
        glow = pow(glow, 2.0);
        color.rgb *= glow * 1.3;
        color.a = glow;
    }
    
    if (color.a < 0.01) discard;
    
    return color;
}
