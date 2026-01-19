// Starfield rendering shader
// Visualizes different celestial objects using instancing with depth-based size/brightness

// Star types
static const float STAR_TYPE_NORMAL = 0.0;
static const float STAR_TYPE_GALAXY = 1.0;
static const float STAR_TYPE_QUASAR = 2.0;
static const float STAR_TYPE_WHITE_DWARF = 3.0;

struct Star {
    float x;
    float y;
    float z;
    float star_type; // 0=normal, 1=galaxy, 2=quasar, 3=white dwarf
};

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer
struct RenderResources {
    StructuredBuffer<Star> stars;
};
ParameterBlock<RenderResources> gResources;
#define STARS gResources.stars

#elif defined(__SPIRV__)
// Vulkan: Push constants for indices + global descriptor arrays
import goldy_exp.buffer_indices;
[[vk::binding(0, 0)]] StructuredBuffer<Star> g_StorageBuffers[];
#define STARS g_StorageBuffers[getBufferIndex(0)]

#elif defined(__DX12__)
// DX12: Root constants + ResourceDescriptorHeap
cbuffer BufferIndices : register(b0, space0) {
    uint starBufferIndex;
};
#define STARS (*DescriptorHandle<StructuredBuffer<Star>>(uint2(starBufferIndex, 0)))

#endif

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD0;      // Local quad UV for procedural effects
    float star_type : TEXCOORD1; // Pass star type to fragment shader
};

// Quad vertices (two triangles)
static const float2 quadVerts[6] = {
    float2(-1, -1), float2( 1, -1), float2( 1,  1),
    float2(-1, -1), float2( 1,  1), float2(-1,  1)
};

[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    VSOutput output;
    
    // Read star from buffer
    Star s = STARS[instanceID];
    
    // Perspective projection: divide x,y by z
    float2 screenPos = float2(s.x, s.y) / max(s.z, 0.01);
    
    // Base size increases as star gets closer (z approaches 0)
    // z range is 0.01 to 1.0, so (1 - z) gives 0 to 0.99
    float baseSize = 0.005 + 0.02 * (1.0 - s.z);
    
    // Adjust size based on star type
    float size = baseSize;
    float3 baseColor = float3(1.0, 1.0, 1.0); // Default white
    
    // Hash function for color randomization
    float hash1 = frac(sin(float(instanceID) * 127.1) * 43758.5453);
    float hash2 = frac(sin(float(instanceID) * 269.5) * 43758.5453);
    
    if (s.star_type == STAR_TYPE_NORMAL) {
        // Normal stars: variety of stellar colors (blue-white, white, yellow, orange, red)
        float temp = hash1; // Random "temperature"
        if (temp < 0.15) {
            // Hot blue stars (O/B class)
            baseColor = float3(0.7, 0.85, 1.0);
        } else if (temp < 0.4) {
            // White stars (A/F class)
            baseColor = float3(1.0, 1.0, 0.95);
        } else if (temp < 0.7) {
            // Yellow stars (G class, like our Sun)
            baseColor = float3(1.0, 0.95, 0.8);
        } else if (temp < 0.9) {
            // Orange stars (K class)
            baseColor = float3(1.0, 0.8, 0.6);
        } else {
            // Red stars (M class)
            baseColor = float3(1.0, 0.6, 0.5);
        }
    }
    else if (s.star_type == STAR_TYPE_GALAXY) {
        // Galaxies: random vibrant colors!
        size = baseSize * 2.5;
        
        // Generate random galaxy colors using hue interpolation
        float hue = hash1 * 6.0;
        int colorIdx = int(hue);
        float blend = frac(hue);
        
        // Colorful galaxy palette
        float3 galaxyColors[7] = {
            float3(1.0, 0.5, 0.7),   // Pink
            float3(0.7, 0.5, 1.0),   // Purple
            float3(0.5, 0.7, 1.0),   // Blue
            float3(0.5, 1.0, 0.9),   // Cyan/teal
            float3(0.7, 1.0, 0.6),   // Green-teal
            float3(1.0, 0.9, 0.5),   // Gold
            float3(1.0, 0.5, 0.7)    // Back to pink (wrap)
        };
        
        baseColor = lerp(galaxyColors[colorIdx], galaxyColors[colorIdx + 1], blend);
    }
    else if (s.star_type == STAR_TYPE_QUASAR) {
        // Quasars: extremely bright, vibrant colors
        size = baseSize * 1.8;
        float quasarHue = hash1;
        if (quasarHue < 0.33) {
            baseColor = float3(0.4, 1.0, 1.0);  // Bright cyan
        } else if (quasarHue < 0.66) {
            baseColor = float3(0.7, 0.6, 1.0);  // Bright violet
        } else {
            baseColor = float3(1.0, 0.7, 1.0);  // Bright magenta
        }
    }
    else if (s.star_type == STAR_TYPE_WHITE_DWARF) {
        // White dwarfs: intense blue-white
        size = baseSize * 0.7;
        baseColor = lerp(float3(0.85, 0.9, 1.0), float3(1.0, 1.0, 1.0), hash1);
    }
    
    // Get quad vertex and scale
    float2 localPos = quadVerts[vertexID] * size;
    
    // Position in clip space
    output.position = float4(screenPos + localPos, 0.0, 1.0);
    
    // Brightness increases as star gets closer
    float brightness = 1.0 - s.z;
    output.color = float4(baseColor * brightness, 1.0);
    
    // Pass UV coordinates for procedural effects in fragment shader
    output.uv = quadVerts[vertexID]; // -1 to 1 range
    output.star_type = s.star_type;
    
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    float2 uv = input.uv;
    float dist = length(uv); // Distance from center (0 to ~1.414)
    float4 color = input.color;
    
    if (input.star_type == STAR_TYPE_NORMAL) {
        // Normal stars: plus sign / cross with tapered points
        float2 absUV = abs(uv);
        
        // Horizontal and vertical bars with tapered edges
        float hBar = max(0.0, 1.0 - absUV.y * 4.0) * (1.0 - smoothstep(0.0, 1.0, absUV.x));
        float vBar = max(0.0, 1.0 - absUV.x * 4.0) * (1.0 - smoothstep(0.0, 1.0, absUV.y));
        
        // Combine bars for plus shape
        float cross = max(hBar, vBar);
        
        // Add a bright core
        float core = 1.0 - smoothstep(0.0, 0.25, dist);
        
        float glow = max(cross * 0.8, core);
        color.rgb *= glow;
        color.a = glow;
    }
    else if (input.star_type == STAR_TYPE_GALAXY) {
        // Galaxies: elliptical with spiral hint
        float2 rotUV = float2(uv.x * 0.7 + uv.y * 0.3, uv.y * 0.8 - uv.x * 0.2);
        float ellipseDist = length(rotUV * float2(1.0, 1.5));
        float glow = 1.0 - smoothstep(0.0, 0.8, ellipseDist);
        // Add some spiral structure
        float angle = atan2(uv.y, uv.x);
        float spiral = sin(angle * 2.0 + dist * 6.0) * 0.15 + 0.85;
        color.rgb *= glow * spiral;
        color.a = glow * 0.9;
    }
    else if (input.star_type == STAR_TYPE_QUASAR) {
        // Quasars: intense bright core with jets
        float core = 1.0 - smoothstep(0.0, 0.3, dist);
        float jet = max(0.0, 1.0 - abs(uv.x) * 4.0) * (1.0 - smoothstep(0.0, 1.0, abs(uv.y)));
        float glow = max(core, jet * 0.5);
        color.rgb *= glow * 1.5; // Extra bright
        color.a = glow;
    }
    else if (input.star_type == STAR_TYPE_WHITE_DWARF) {
        // White dwarfs: very sharp, intense point
        float glow = 1.0 - smoothstep(0.0, 0.5, dist);
        glow = pow(glow, 2.0); // Sharper falloff
        color.rgb *= glow * 1.3;
        color.a = glow;
    }
    
    // Discard nearly transparent pixels
    if (color.a < 0.01) discard;
    
    return color;
}
