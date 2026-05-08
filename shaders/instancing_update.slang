// Instancing compute shader - updates rotation and color for each quad instance

import goldy_exp;

struct QuadInstance {
    float2 position;
    float rotation;
    float scale;
    float4 color;
};

struct AnimParams {
    float time;
    float delta_time;
    uint total_instances;
    uint _pad;
};

float3 hsv_to_rgb(float h, float s, float v) {
    float4 K = float4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    float3 p = abs(frac(h.xxx + K.xyz) * 6.0 - K.www);
    return v * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<QuadInstance> INSTANCES, AnimParams params, ThreadId id) {
    uint idx = id.x;
    if (idx >= params.total_instances) return;
    
    QuadInstance inst = INSTANCES[idx];
    
    float phase = (float(idx) / float(params.total_instances)) * 6.28318;
    inst.rotation = params.time * 2.0 + phase;
    
    float hue = frac(float(idx) / float(params.total_instances) + params.time * 0.1);
    inst.color = float4(hsv_to_rgb(hue, 0.8, 0.9), 1.0);
    
    INSTANCES[idx] = inst;
}
