// Open AAA Showcase Skybox: 2K Equirectangular Sunset HDRI -> 4-MRT HDR G-Buffer + Motion Vectors
cbuffer SkyCB : register(b0) {
    float4x4 g_InvViewProj;
    float4x4 g_PrevViewProj;
    float3   g_CameraPos;
    float    g_Time;
};

Texture2D<float4> g_EnvHdr     : register(t0);
SamplerState      g_EnvSampler : register(s0);

struct SkyPSIn {
    float4 pos : SV_Position;
    float2 ndc : TEXCOORD0;
};

struct GBufferOut {
    float4 directHDR   : SV_Target0;
    float4 rawRTGI     : SV_Target1;
    float4 normalDepth : SV_Target2;
    float2 motionVec   : SV_Target3;
};

SkyPSIn VSMain(uint vid : SV_VertexID) {
    float2 p = float2((vid == 1u) ? 3.0 : -1.0, (vid == 2u) ? 3.0 : -1.0);
    SkyPSIn o;
    o.pos = float4(p, 1.0, 1.0);
    o.ndc = p;
    return o;
}

float2 dir_to_equirect_uv(float3 d) {
    float phi = atan2(d.z, d.x);
    float u = frac(phi / 6.28318530718 + 0.5 - 0.115);
    float v = clamp(acos(clamp(d.y, -1.0, 1.0)) / 3.14159265359, 0.001, 0.999);
    return float2(u, v);
}

GBufferOut PSMain(SkyPSIn input) {
    float4 worldPt = mul(g_InvViewProj, float4(input.ndc, 1.0, 1.0));
    float3 rawDir = worldPt.xyz / worldPt.w - g_CameraPos;
    float3 dir = (dot(rawDir, rawDir) > 1e-6) ? normalize(rawDir) : float3(0.0, 0.0, 1.0);

    float2 envUv = dir_to_equirect_uv(dir);
    float3 hdrSky = g_EnvHdr.SampleLevel(g_EnvSampler, envUv, 0.0).rgb * 0.40;

    float3 sunDir = normalize(float3(0.28, 0.16, 0.94));
    float sunCos = max(dot(dir, sunDir), 0.0);
    hdrSky += float3(2.20, 0.95, 0.38) * pow(sunCos, 256.0);
    hdrSky += float3(0.32, 0.14, 0.06) * pow(sunCos, 10.0);

    float2 curUV = float2(input.ndc.x * 0.5 + 0.5, 0.5 - input.ndc.y * 0.5);
    float4 prevClip = mul(g_PrevViewProj, float4(g_CameraPos + dir * 200.0, 1.0));
    float2 prevUV = curUV;
    if (abs(prevClip.w) > 1e-5) {
        prevUV = float2(prevClip.x / prevClip.w * 0.5 + 0.5, 0.5 - prevClip.y / prevClip.w * 0.5);
    }

    GBufferOut o;
    o.directHDR = float4(hdrSky, 1.0);
    o.rawRTGI = float4(0.0, 0.0, 0.0, 0.0);
    o.normalDepth = float4(0.0, 0.0, 0.0, 9999.0);
    o.motionVec = prevUV - curUV;
    return o;
}
