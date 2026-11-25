// Open AAA Showcase Final Composite: TAA HDR + Bloom + Auto-Exposure + AgX Filmic + CAS Sharpen + sRGB
cbuffer AaaCompCB : register(b0) {
    float g_BloomStrength;
    float g_CasStrength;
    float g_ExposureScale;
    float g_Time;
};

Texture2D<float4>       g_TaaHDR      : register(t0);
Texture2D<float4>       g_BloomTex    : register(t1);
StructuredBuffer<float> g_ExposureBuf : register(t2);
SamplerState            g_LinClamp    : register(s0);

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

// AgX-inspired Filmic Transform (smooth warm highlight roll-off without sunset blowout)
float3 agx_filmic(float3 x) {
    static const float3x3 agxMat = float3x3(
        0.84247906, 0.07843360, 0.07922375,
        0.04232824, 0.87846864, 0.07916613,
        0.04237565, 0.07843360, 0.87914297
    );
    static const float3x3 agxInv = float3x3(
         1.1968790, -0.0980209, -0.0990297,
        -0.0528969,  1.1519031, -0.0989612,
        -0.0529716, -0.0980435,  1.1510737
    );
    float3 v = mul(agxMat, max(x, 1e-6));
    v = clamp((log2(v) + 12.47393) / 16.5, 0.0, 1.0);
    float3 v2 = v * v;
    float3 v4 = v2 * v2;
    v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
    v = mul(agxInv, v);
    return saturate(v);
}

float luma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float3 eval_tonemapped(float2 uv, float autoExp) {
    float3 taa = g_TaaHDR.SampleLevel(g_LinClamp, uv, 0.0).rgb;
    float3 bloom = g_BloomTex.SampleLevel(g_LinClamp, uv, 0.0).rgb * g_BloomStrength;
    float3 hdr = (taa + bloom) * autoExp * g_ExposureScale;
    float3 mapped = agx_filmic(hdr);
    float2 d = uv - 0.5;
    float vig = 1.0 - dot(d, d) * 0.22;
    return saturate(mapped * vig);
}

float4 PSMain(CompPSIn input) : SV_Target {
    uint tw = 1u, th = 1u;
    g_TaaHDR.GetDimensions(tw, th);
    float2 inv = 1.0 / float2((float)tw, (float)th);
    float autoExp = clamp(g_ExposureBuf[0], 0.5, 1.5);

    float3 cC = eval_tonemapped(input.uv, autoExp);
    float3 cN = eval_tonemapped(input.uv + float2(0.0, -inv.y), autoExp);
    float3 cS = eval_tonemapped(input.uv + float2(0.0,  inv.y), autoExp);
    float3 cW = eval_tonemapped(input.uv + float2(-inv.x, 0.0), autoExp);
    float3 cE = eval_tonemapped(input.uv + float2( inv.x, 0.0), autoExp);

    // 5-Tap Contrast-Adaptive Sharpening (CAS) in perceptual space
    float lC = luma(cC);
    float lN = luma(cN);
    float lS = luma(cS);
    float lW = luma(cW);
    float lE = luma(cE);

    float minL = min(lC, min(min(lN, lS), min(lW, lE)));
    float maxL = max(lC, max(max(lN, lS), max(lW, lE)));
    float amp = sqrt(saturate(min(minL, 1.0 - maxL) / max(maxL, 1e-4)));
    float w = amp * (-0.125 * saturate(g_CasStrength));
    float3 sharpened = saturate((cC + (cN + cS + cW + cE) * w) / (1.0 + 4.0 * w));

    return float4(sharpened, 1.0);
}
