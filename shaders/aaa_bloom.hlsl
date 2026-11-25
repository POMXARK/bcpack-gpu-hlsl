// 3-Stage Half-Res HDR Bloom (Bright-Pass Downsample + Separable 9-Tap Gaussian Blur)
cbuffer BloomCB : register(b0) {
    uint  g_Width;
    uint  g_Height;
    uint  g_Pass;      // 0 = threshold downsample, 1 = H blur, 2 = V blur
    float g_Threshold; // soft-knee luminance threshold
};

Texture2D<float4>   g_Src      : register(t0);
RWTexture2D<float4> g_Dst      : register(u0);
SamplerState        g_LinClamp : register(s0);

static const float W[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };

float3 bright_filter(float3 c, float thresh) {
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    float knee = max(thresh * 0.45, 0.1);
    float soft = lum - thresh + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = (soft * soft) / max(4.0 * knee, 1e-4);
    float contrib = max(soft, lum - thresh) / max(lum, 1e-4);
    return c * saturate(contrib);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= g_Width || dtid.y >= g_Height) {
        return;
    }
    int2 px = int2(dtid.xy);

    if (g_Pass == 0u) {
        float2 uv = (float2(px) + 0.5) / float2(g_Width, g_Height);
        float2 texel = 0.5 / float2(g_Width, g_Height);
        float3 c = g_Src.SampleLevel(g_LinClamp, uv + float2(-texel.x, -texel.y), 0.0).rgb * 0.25;
        c += g_Src.SampleLevel(g_LinClamp, uv + float2( texel.x, -texel.y), 0.0).rgb * 0.25;
        c += g_Src.SampleLevel(g_LinClamp, uv + float2(-texel.x,  texel.y), 0.0).rgb * 0.25;
        c += g_Src.SampleLevel(g_LinClamp, uv + float2( texel.x,  texel.y), 0.0).rgb * 0.25;
        // Clamp extreme sun/specular spikes to keep bloom clean
        c = min(c, float3(12.0, 12.0, 12.0));
        g_Dst[px] = float4(bright_filter(c, g_Threshold), 1.0);
        return;
    }

    int2 dir = (g_Pass == 1u) ? int2(1, 0) : int2(0, 1);
    int2 maxPx = int2((int)g_Width - 1, (int)g_Height - 1);
    float3 sum = g_Src[px].rgb * W[0];
    [unroll]
    for (int i = 1; i <= 4; ++i) {
        int2 pA = clamp(px + dir * i, int2(0, 0), maxPx);
        int2 pB = clamp(px - dir * i, int2(0, 0), maxPx);
        sum += (g_Src[pA].rgb + g_Src[pB].rgb) * W[i];
    }
    g_Dst[px] = float4(sum, 1.0);
}
