// 2-Pass A-Trous Edge-Aware Bilateral Filter for 1-spp Diffuse RTGI (SM 6.6)
cbuffer DenoiseCB : register(b0) {
    uint g_Width;
    uint g_Height;
    uint g_StepSize;
    uint g_Pad;
};

Texture2D<float4>   g_SrcGI       : register(t0);
Texture2D<float4>   g_NormalDepth : register(t1);
RWTexture2D<float4> g_DstGI       : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= g_Width || dtid.y >= g_Height) {
        return;
    }

    int2 px = int2(dtid.xy);
    float4 cND = g_NormalDepth[px];
    if (cND.w > 5000.0) {
        g_DstGI[px] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 cN = (dot(cND.xyz, cND.xyz) > 1e-6) ? normalize(cND.xyz) : float3(0.0, 1.0, 0.0);
    float cD = cND.w;

    static const float k1[3] = { 0.25, 0.50, 0.25 };
    int step = (int)max(g_StepSize, 1u);

    float3 sumGI = float3(0.0, 0.0, 0.0);
    float sumW = 0.0;

    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            int2 q = clamp(px + int2(dx * step, dy * step), int2(0, 0), int2((int)g_Width - 1, (int)g_Height - 1));
            float4 nND = g_NormalDepth[q];
            if (nND.w > 5000.0) {
                continue;
            }
            float3 nN = (dot(nND.xyz, nND.xyz) > 1e-6) ? normalize(nND.xyz) : float3(0.0, 1.0, 0.0);
            float wN = pow(max(dot(cN, nN), 0.0), 16.0);
            float wD = exp(-abs(cD - nND.w) / (0.04 * cD + 0.08));
            float w = k1[dx + 1] * k1[dy + 1] * wN * wD;
            sumGI += g_SrcGI[q].rgb * w;
            sumW += w;
        }
    }

    g_DstGI[px] = float4(sumGI / max(sumW, 1e-4), 1.0);
}
