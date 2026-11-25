// HDR + Denoised RTGI Composite Pass + Filmic Neon Highlight Rolloff
cbuffer CompCB : register(b0) {
    float g_GiStrength;
    float g_Exposure;
    float g_Pad0;
    float g_Pad1;
};

Texture2D<float4> g_DirectHDR  : register(t0);
Texture2D<float4> g_DenoisedGI : register(t1);
SamplerState      g_LinClamp   : register(s0);

struct CompPSIn {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

CompPSIn VSMain(uint vid : SV_VertexID) {
    float2 p = float2((vid == 1u) ? 3.0 : -1.0, (vid == 2u) ? 3.0 : -1.0);
    CompPSIn o;
    o.pos = float4(p, 0.0, 1.0);
    o.uv = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return o;
}

float4 PSMain(CompPSIn input) : SV_Target {
    float3 direct = g_DirectHDR.SampleLevel(g_LinClamp, input.uv, 0).rgb;
    float3 gi = g_DenoisedGI.SampleLevel(g_LinClamp, input.uv, 0).rgb * g_GiStrength;
    float3 hdr = (direct + gi) * g_Exposure;
    // Preserve vibrant Vice-City midtones while smoothly rolling off bright neon/GI HDR highlights
    float3 mapped = hdr / (1.0 + max(hdr - 0.85, 0.0) * 0.65);
    return float4(saturate(mapped), 1.0);
}
