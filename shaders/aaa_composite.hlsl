// Open AAA Showcase Final Composite: Direct HDR + Denoised RTGI + Bloom + ACES Filmic + sRGB + Vignette
cbuffer AaaCompCB : register(b0) {
    float g_GiStrength;
    float g_BloomStrength;
    float g_Exposure;
    float g_Time;
};

Texture2D<float4> g_DirectHDR  : register(t0);
Texture2D<float4> g_DenoisedGI : register(t1);
Texture2D<float4> g_BloomTex   : register(t2);
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

float3 aces_filmic(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

float4 PSMain(CompPSIn input) : SV_Target {
    float3 direct = g_DirectHDR.SampleLevel(g_LinClamp, input.uv, 0.0).rgb;
    float3 gi = g_DenoisedGI.SampleLevel(g_LinClamp, input.uv, 0.0).rgb * g_GiStrength;
    float3 bloom = g_BloomTex.SampleLevel(g_LinClamp, input.uv, 0.0).rgb * g_BloomStrength;

    float3 hdr = (direct + gi + bloom) * g_Exposure;
    float3 mapped = aces_filmic(hdr);

    // Subtle filmic vignette
    float2 d = input.uv - 0.5;
    float vig = 1.0 - dot(d, d) * 0.28;
    mapped *= vig;

    // Linear -> sRGB gamma for R8G8B8A8_UNORM backbuffer
    float3 srgb = pow(saturate(mapped), 1.0 / 2.2);
    return float4(srgb, 1.0);
}
