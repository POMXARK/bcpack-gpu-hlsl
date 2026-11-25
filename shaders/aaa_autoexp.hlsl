// GPU Compute Log-Average Histogram Auto-Exposure (SM 6.6)
cbuffer AutoExpCB : register(b0) {
    uint  g_Width;
    uint  g_Height;
    uint  g_FrameIdx;
    float g_AdaptRate;
};

Texture2D<float4>        g_CurHDR      : register(t0);
RWStructuredBuffer<float> g_ExposureBuf : register(u0);
SamplerState             g_LinClamp    : register(s0);

groupshared float s_LogLum[256];

[numthreads(16, 16, 1)]
void CSMain(uint3 gtid : SV_GroupThreadID, uint gidx : SV_GroupIndex) {
    float2 uv = (float2(gtid.xy) + 0.5) / 16.0;
    float3 c = g_CurHDR.SampleLevel(g_LinClamp, uv, 0.0).rgb;
    float lum = dot(max(c, 0.0), float3(0.2126, 0.7152, 0.0722));
    s_LogLum[gidx] = log2(clamp(lum, 0.02, 8.0));
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (gidx < s) {
            s_LogLum[gidx] += s_LogLum[gidx + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gidx == 0u) {
        float avgLum = exp2(s_LogLum[0] / 256.0);
        float targetExp = clamp(0.24 / max(avgLum, 0.05), 0.72, 1.15);
        if (g_FrameIdx == 0u) {
            g_ExposureBuf[0] = targetExp;
        } else {
            g_ExposureBuf[0] = lerp(g_ExposureBuf[0], targetExp, saturate(g_AdaptRate));
        }
    }
}
