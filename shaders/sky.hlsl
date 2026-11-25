cbuffer SkyCB : register(b0) {
    float4x4 g_InvViewProj;
    float3   g_CameraPos;
    float    g_Time;
};

struct SkyPSIn {
    float4 pos : SV_Position;
    float2 ndc : TEXCOORD0;
};

SkyPSIn VSMain(uint vid : SV_VertexID) {
    float2 p = float2((vid == 1u) ? 3.0 : -1.0, (vid == 2u) ? 3.0 : -1.0);
    SkyPSIn o;
    o.pos = float4(p, 1.0, 1.0);
    o.ndc = p;
    return o;
}

float4 PSMain(SkyPSIn input) : SV_Target {
    // Reconstruct world-space view ray direction
    float4 worldPt = mul(g_InvViewProj, float4(input.ndc, 1.0, 1.0));
    float3 dir = normalize(worldPt.xyz / worldPt.w - g_CameraPos);

    float h = saturate(dir.y * 1.35 + 0.06);
    float3 horizon = float3(1.00, 0.42, 0.30);
    float3 midSky  = float3(0.55, 0.16, 0.48);
    float3 zenith  = float3(0.08, 0.05, 0.22);

    float3 sky = (h < 0.35)
        ? lerp(horizon, midSky, h / 0.35)
        : lerp(midSky, zenith, (h - 0.35) / 0.65);

    // Synthwave sunset sun disc in the +Z horizon
    float3 sunDir = normalize(float3(0.08, 0.16, 1.0));
    float cosAng = dot(dir, sunDir);
    float sunDisc = smoothstep(0.9925, 0.9955, cosAng);
    float sunGlow = pow(saturate(cosAng), 28.0) * 0.65 + pow(saturate(cosAng), 6.0) * 0.22;

    // Classic horizontal sunset stripes across lower half of sun
    float sunRelY = (dir.y - sunDir.y) * 9.5;
    float stripeMask = (sunRelY < 0.15) ? step(0.32, frac(sunRelY * 5.0 - g_Time * 0.15)) : 1.0;
    float3 sunColor = lerp(float3(1.0, 0.22, 0.58), float3(1.0, 0.92, 0.42), saturate(sunRelY + 0.5));

    sky += float3(1.0, 0.40, 0.25) * sunGlow;
    sky = lerp(sky, sunColor, sunDisc * stripeMask);

    // Subtle twinkling stars high in the zenith
    float2 starCell = floor(dir.xz / max(dir.y, 0.15) * 90.0);
    float sh = frac(sin(dot(starCell, float2(12.9898, 78.233))) * 43758.5453);
    if (dir.y > 0.22 && sh > 0.994) {
        sky += float3(0.75, 0.85, 1.0) * ((dir.y - 0.22) * 2.5);
    }

    return float4(sky, 1.0);
}
