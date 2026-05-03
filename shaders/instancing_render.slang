// Instancing render shader - renders quads from instance buffer

import goldy_exp;

struct QuadInstance {
    float2 position;
    float rotation;
    float scale;
    float4 color;
};

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[goldy_vertex]
VSOutput vs_main(Scattered<QuadInstance> INSTANCES, VertexId vertex_id, InstanceId instance_id) {
    VSOutput output;
    
    QuadInstance inst = INSTANCES[instance_id.value];
    float2 world_pos = quad_position_rotated(vertex_id.value, inst.position, inst.scale, inst.rotation);
    
    output.position = float4(world_pos, 0.0, 1.0);
    output.color = inst.color;
    
    return output;
}

[goldy_fragment]
float4 fs_main(VSOutput input) : SV_Target {
    return input.color;
}
