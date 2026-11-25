// Open AAA Showcase Final Composite: Direct HDR + Denoised RTGI + Bloom + ACES Filmic + Vignette + 5-Tap FXAA
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

float3 eval_srgb(float2 uv) {
    float3 direct = g_DirectHDR.SampleLevel(g_LinClamp, uv, 0.0).rgb;
    float3 gi = g_DenoisedGI.SampleLevel(g_LinClamp, uv, 0.0).rgb * g_GiStrength;
    float3 bloom = g_BloomTex.SampleLevel(g_LinClamp, uv, 0.0).rgb * g_BloomStrength;

    float3 hdr = (direct + gi + bloom) * g_Exposure;
    float3 mapped = aces_filmic(hdr);

    float2 d = uv - 0.5;
    float vig = 1.0 - dot(d, d) * 0.25;
    mapped *= vig;

    return pow(saturate(mapped), 1.0 / 2.2);
}

float luma(float3 c) {
    return dot(c, float3(0.299, 0.587, 0.114));
}

float4 PSMain(CompPSIn input) : SV_Target {
    uint tw = 1u, th = 1u;
    g_DirectHDR.GetDimensions(tw, th);
    float2 inv = 1.0 / float2((float)tw, (float)th);

    float3 cC = eval_srgb(input.uv);
    float3 cN = eval_srgb(input.uv + float2(0.0, -inv.y));
    float3 cS = eval_srgb(input.uv + float2(0.0,  inv.y));
    float3 cW = eval_srgb(input.uv + float2(-inv.x, 0.0));
    float3 cE = eval_srgb(input.uv + float2( inv.x, 0.0));

    float lC = luma(cC);
    float lN = luma(cN);
    float lS = luma(cS);
    float lW = luma(cW);
    float lE = luma(cE);

    float lMin = min(lC, min(min(lN, lS), min(lW, lE)));
    float lMax = max(lC, max(max(lN, lS), max(lW, lE)));
    float lRange = lMax - lMin;

    if (lRange > max(0.05, lMax * 0.11)) {
        float hEdge = abs((lW + lE) - 2.0 * lC);
        float vEdge = abs((lN + lS) - 2.0 * lC);
        float3 neighborAvg = (hEdge >= vEdge) ? 0.5 * (cN + cS) : 0.5 * (cW + cE);
        float blend = saturate((lRange - 0.05) * 1.35) * 0.55;
        cC = lerp(cC, neighborAvg, blend);
    }

    return float4(cC, 1.0);
}
