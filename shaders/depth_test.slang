// 3D depth-testing shader: vertex carries (x, y, z) position and RGBA color.
// Used by depth occlusion screenshot tests.

struct VIn {
    float3 position : POSITION;
    float4 color : COLOR;
};

struct VOut {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VOut vs_main(VIn v) {
    VOut o;
    o.position = float4(v.position.xy, v.position.z, 1.0);
    o.color = v.color;
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target {
    return i.color;
}
