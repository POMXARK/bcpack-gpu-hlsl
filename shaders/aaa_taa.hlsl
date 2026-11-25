// Screen-Space Contact Shadows + YCoCg Variance-Clipped Temporal Anti-Aliasing (SM 6.6)
cbuffer TaaCB : register(b0) {
    uint   g_Width;
    uint   g_Height;
    uint   g_FrameIdx;
    float  g_GiStrength;
    float2 g_SunScreenDir;
    float  g_ContactStrength;
    float  g_Pad;
};

Texture2D<float4>   g_DirectHDR   : register(t0);
Texture2D<float4>   g_DenoisedGI  : register(t1);
Texture2D<float2>   g_MotionVec   : register(t2);
Texture2D<float4>   g_PrevHist    : register(t3);
Texture2D<float4>   g_NormalDepth : register(t4);
RWTexture2D<float4> g_CurHist     : register(u0);
SamplerState        g_LinClamp    : register(s0);

float3 rgb_to_ycocg(float3 c) {
    float y  = 0.25 * c.r + 0.5 * c.g + 0.25 * c.b;
    float co = 0.5 * c.r - 0.5 * c.b;
    float cg = -0.25 * c.r + 0.5 * c.g - 0.25 * c.b;
    return float3(y, co, cg);
}

float3 ycocg_to_rgb(float3 ycc) {
    float tmp = ycc.x - ycc.z;
    return float3(tmp + ycc.y, ycc.x + ycc.z, tmp - ycc.y);
}

float eval_contact_shadow(int2 px, float curDepth, int2 maxPx) {
    if (curDepth > 22.0 || g_ContactStrength <= 0.001) {
        return 1.0;
    }
    float dirLen = length(g_SunScreenDir);
    if (dirLen < 1e-4) {
        return 1.0;
    }
    float2 stepDir = (g_SunScreenDir / dirLen) * 2.5;
    float occ = 0.0;
    [unroll]
    for (int i = 1; i <= 8; ++i) {
        int2 sp = clamp(int2(round(float2(px) + stepDir * (float)i)), int2(0, 0), maxPx);
        float sd = g_NormalDepth[sp].w;
        if (sd < 5000.0) {
            float dz = curDepth - sd;
            if (dz > 0.02 && dz < 0.45) {
                occ = max(occ, 1.0 - abs(dz - 0.14) / 0.45);
            }
        }
    }
    float distFade = 1.0 - smoothstep(10.0, 22.0, curDepth);
    return lerp(1.0, 0.45, saturate(occ) * g_ContactStrength * distFade);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= g_Width || dtid.y >= g_Height) {
        return;
    }

    int2 px = int2(dtid.xy);
    int2 maxPx = int2((int)g_Width - 1, (int)g_Height - 1);
    float4 cND = g_NormalDepth[px];
    float contactVis = (cND.w < 5000.0) ? eval_contact_shadow(px, cND.w, maxPx) : 1.0;

    float3 curHDR = g_DirectHDR[px].rgb * contactVis + g_DenoisedGI[px].rgb * g_GiStrength;
    if (g_FrameIdx == 0u) {
        g_CurHist[px] = float4(max(curHDR, 0.0), 1.0);
        return;
    }

    float3 m1 = float3(0.0, 0.0, 0.0);
    float3 m2 = float3(0.0, 0.0, 0.0);
    float bestDepth = 1e9;
    int2 bestPx = px;

    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            int2 q = clamp(px + int2(dx, dy), int2(0, 0), maxPx);
            float nd = g_NormalDepth[q].w;
            if (nd < bestDepth) {
                bestDepth = nd;
                bestPx = q;
            }
            float3 nHDR = g_DirectHDR[q].rgb * contactVis + g_DenoisedGI[q].rgb * g_GiStrength;
            float3 ycc = rgb_to_ycocg(max(nHDR, 0.0));
            m1 += ycc;
            m2 += ycc * ycc;
        }
    }

    m1 /= 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - m1 * m1, 0.0));
    float3 boxMin = m1 - 1.25 * sigma;
    float3 boxMax = m1 + 1.25 * sigma;

    float2 invRes = 1.0 / float2((float)g_Width, (float)g_Height);
    float2 curUV = (float2(px) + 0.5) * invRes;
    float2 vel = g_MotionVec[bestPx];
    float2 prevUV = curUV + vel;

    float3 outHDR = curHDR;
    if (all(prevUV >= 0.0) && all(prevUV <= 1.0)) {
        float3 histRGB = g_PrevHist.SampleLevel(g_LinClamp, prevUV, 0.0).rgb;
        float3 histYcc = clamp(rgb_to_ycocg(max(histRGB, 0.0)), boxMin, boxMax);
        float3 clampedHist = max(ycocg_to_rgb(histYcc), 0.0);
        float alpha = clamp(0.12 + length(vel) * 18.0, 0.12, 0.45);
        outHDR = lerp(clampedHist, curHDR, alpha);
    }

    g_CurHist[px] = float4(max(outHDR, 0.0), 1.0);
}
