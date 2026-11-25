// Screen-Space Contact Shadows + Temporal Upscaling (TAAU 50%-100% + 5-Tap Catmull-Rom History, SM 6.6)
cbuffer TaaCB : register(b0) {
    uint   g_OutWidth;
    uint   g_OutHeight;
    uint   g_RenderWidth;
    uint   g_RenderHeight;
    uint   g_FrameIdx;
    float  g_GiStrength;
    float2 g_SunScreenDir;
    float  g_ContactStrength;
    float2 g_JitterUV;      // (+jit_x*0.5, -jit_y*0.5) to align un-jittered display UV to render target
    float  g_UpscaleFactor; // out_w / render_w (1.0 native, 1.5 Quality, 2.0 Performance)
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

float3 sample_history_catmull_rom(float2 uv, float2 texSize) {
    float2 samplePos = uv * texSize;
    float2 tc1 = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - tc1;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float2 w0 = f2 - 0.5 * (f3 + f);
    float2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    float2 w3 = 0.5 * (f3 - f2);
    float2 w2 = 1.0 - w0 - w1 - w3;

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / max(w12, 1e-4);

    float2 invSize = 1.0 / texSize;
    float2 tc0 = (tc1 - 1.0) * invSize;
    float2 tc3 = (tc1 + 2.0) * invSize;
    float2 tc12 = (tc1 + offset12) * invSize;

    float3 result = 0.0;
    result += g_PrevHist.SampleLevel(g_LinClamp, float2(tc12.x, tc0.y), 0.0).rgb * (w12.x * w0.y);
    result += g_PrevHist.SampleLevel(g_LinClamp, float2(tc0.x, tc12.y), 0.0).rgb * (w0.x * w12.y);
    result += g_PrevHist.SampleLevel(g_LinClamp, float2(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y);
    result += g_PrevHist.SampleLevel(g_LinClamp, float2(tc3.x, tc12.y), 0.0).rgb * (w3.x * w12.y);
    result += g_PrevHist.SampleLevel(g_LinClamp, float2(tc12.x, tc3.y), 0.0).rgb * (w12.x * w3.y);

    float weightSum = (w12.x * w0.y) + (w0.x * w12.y) + (w12.x * w12.y) + (w3.x * w12.y) + (w12.x * w3.y);
    return max(result / max(weightSum, 1e-4), 0.0);
}

float eval_contact_shadow(int2 srcPx, float curDepth, int2 maxSrcPx) {
    if (curDepth > 22.0 || g_ContactStrength <= 0.001) {
        return 1.0;
    }
    float dirLen = length(g_SunScreenDir);
    if (dirLen < 1e-4) {
        return 1.0;
    }
    float2 stepDir = (g_SunScreenDir / dirLen) * (2.5 / max(g_UpscaleFactor, 1.0));
    float occ = 0.0;
    [unroll]
    for (int i = 1; i <= 8; ++i) {
        int2 sp = clamp(int2(round(float2(srcPx) + stepDir * (float)i)), int2(0, 0), maxSrcPx);
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
    if (dtid.x >= g_OutWidth || dtid.y >= g_OutHeight) {
        return;
    }

    int2 outPx = int2(dtid.xy);
    float2 outRes = float2((float)g_OutWidth, (float)g_OutHeight);
    float2 renderRes = float2((float)g_RenderWidth, (float)g_RenderHeight);
    float2 outUV = (float2(outPx) + 0.5) / outRes;

    float2 sampleUV = clamp(outUV + g_JitterUV, 0.0, 1.0);
    int2 maxSrcPx = int2((int)g_RenderWidth - 1, (int)g_RenderHeight - 1);
    int2 srcPx = clamp(int2(floor(sampleUV * renderRes)), int2(0, 0), maxSrcPx);

    float4 cND = g_NormalDepth[srcPx];
    float contactVis = (cND.w < 5000.0) ? eval_contact_shadow(srcPx, cND.w, maxSrcPx) : 1.0;

    float3 curHDR;
    if (g_UpscaleFactor > 1.01) {
        curHDR = g_DirectHDR.SampleLevel(g_LinClamp, sampleUV, 0.0).rgb * contactVis
               + g_DenoisedGI.SampleLevel(g_LinClamp, sampleUV, 0.0).rgb * g_GiStrength;
    } else {
        curHDR = g_DirectHDR[srcPx].rgb * contactVis + g_DenoisedGI[srcPx].rgb * g_GiStrength;
    }

    if (g_FrameIdx == 0u) {
        g_CurHist[outPx] = float4(max(curHDR, 0.0), 1.0);
        return;
    }

    float3 m1 = float3(0.0, 0.0, 0.0);
    float3 m2 = float3(0.0, 0.0, 0.0);
    float bestDepth = 1e9;
    int2 bestPx = srcPx;

    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            int2 q = clamp(srcPx + int2(dx, dy), int2(0, 0), maxSrcPx);
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
    float boxWidth = lerp(1.25, 1.50, saturate(g_UpscaleFactor - 1.0));
    float3 boxMin = m1 - boxWidth * sigma;
    float3 boxMax = m1 + boxWidth * sigma;

    float2 vel = g_MotionVec[bestPx];
    float2 prevUV = outUV + vel;

    float3 outHDR = curHDR;
    if (all(prevUV >= 0.0) && all(prevUV <= 1.0)) {
        float3 histRGB = (g_UpscaleFactor > 1.01)
            ? sample_history_catmull_rom(prevUV, outRes)
            : g_PrevHist.SampleLevel(g_LinClamp, prevUV, 0.0).rgb;
        float3 histYcc = clamp(rgb_to_ycocg(max(histRGB, 0.0)), boxMin, boxMax);
        float3 clampedHist = max(ycocg_to_rgb(histYcc), 0.0);
        float baseAlpha = lerp(0.12, 0.075, saturate(g_UpscaleFactor - 1.0));
        float alpha = clamp(baseAlpha + length(vel) * 18.0, baseAlpha, 0.45);
        outHDR = lerp(clampedHist, curHDR, alpha);
    }

    g_CurHist[outPx] = float4(max(outHDR, 0.0), 1.0);
}
