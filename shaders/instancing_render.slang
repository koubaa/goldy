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

[shader("vertex")]
VSOutput vs_main(uniform uint instances_slot, uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID) {
    StorageBuffer<QuadInstance> INSTANCES = goldy_scattered<QuadInstance>(instances_slot);
    VSOutput output;
    
    QuadInstance inst = INSTANCES[instance_id];
    float2 world_pos = quad_position_rotated(vertex_id, inst.position, inst.scale, inst.rotation);
    
    output.position = float4(world_pos, 0.0, 1.0);
    output.color = inst.color;
    
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    return input.color;
}
