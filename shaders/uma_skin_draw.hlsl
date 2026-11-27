// Clean-room UMA Emulation: Skinned Mesh Draw Shader (reads VertexBuffer written by uma_skin_cs)
// Outputs R32_UINT tag ((mesh_id << 16) | (frame_tag & 0xFFFF)) to verify zero missed/stale hazards.

struct VSIn {
    float3 pos : POSITION;
    uint   tag : TEXCOORD0;
};

struct PSIn {
    float4 pos : SV_Position;
    nointerpolation uint tag : TEXCOORD0;
};

PSIn VSMain(VSIn input) {
    PSIn o;
    o.pos = float4(input.pos, 1.0);
    o.tag = input.tag;
    return o;
}

uint PSMain(PSIn input) : SV_Target0 {
    return input.tag;
}
