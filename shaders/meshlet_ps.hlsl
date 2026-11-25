#include "meshlet_common.hlsli"

Texture2D                       g_Textures[12] : register(t0);
RaytracingAccelerationStructure g_SceneTlas    : register(t12);
StructuredBuffer<float4>        g_InstColors   : register(t13);
SamplerState                    g_Sampler      : register(s0);

struct GBufferOut {
    float4 directHDR   : SV_Target0;
    float4 rawRTGI     : SV_Target1;
    float4 normalDepth : SV_Target2;
};

float3 eval_sunset_env(float3 dir) {
    float h = saturate(dir.y * 0.5 + 0.5);
    float3 horizon = float3(1.00, 0.42, 0.48);
    float3 zenith = float3(0.11, 0.05, 0.25);
    float3 col = lerp(horizon, zenith, pow(saturate(dir.y), 0.55));
    float3 sunDir = normalize(float3(0.08, 0.24, 0.96));
    float sd = max(dot(normalize(dir), sunDir), 0.0);
    col += float3(1.0, 0.65, 0.25) * pow(sd, 64.0) * 1.4;
    return lerp(float3(0.18, 0.06, 0.20), col, h);
}

float ign(float2 xy, float frame) {
    float2 p = xy + 5.588238 * fmod(frame, 64.0);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

float3 trace_diffuse_rtgi(float3 worldPos, float3 N, float2 screenXY) {
    float3 upVec = (abs(N.z) < 0.999) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(upVec, N));
    float3 B = cross(N, T);
    float f = floor(g_Time * 60.0);
    float r1 = ign(screenXY, f);
    float r2 = ign(screenXY + float2(17.0, 31.0), f);
    float phi = 6.2831853 * r1;
    float cosTheta = sqrt(r2);
    float sinTheta = sqrt(max(1.0 - r2, 0.0));
    float3 giRaw = T * (cos(phi) * sinTheta) + B * (sin(phi) * sinTheta) + N * cosTheta;
    float3 giDir = (dot(giRaw, giRaw) > 1e-6) ? normalize(giRaw) : float3(0.0, 1.0, 0.0);

    RayDesc gRay;
    gRay.Origin = worldPos + N * 0.08;
    gRay.Direction = giDir;
    gRay.TMin = 0.08;
    gRay.TMax = 55.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> gq;
    gq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, gRay);
    while (gq.Proceed()) {}

    if (gq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float4 hd = g_InstColors[gq.CommittedInstanceID()];
        float fade = saturate(1.0 - gq.CommittedRayT() / 55.0);
        return hd.rgb * (0.45 + hd.a * 2.4) * fade;
    }
    return eval_sunset_env(giDir) * 0.38;
}

GBufferOut PSMain(MSOut input) {
    uint texIdx = min(input.texIdx, 11u);
    float3 albedo = g_Textures[NonUniformResourceIndex(texIdx)].Sample(g_Sampler, input.uv).rgb;
    float3 N = (dot(input.worldN, input.worldN) > 1e-6) ? normalize(input.worldN) : float3(0.0, 1.0, 0.0);
    float3 toCam = g_CameraPos.xyz - input.worldPos;
    float camDist = length(toCam);
    float3 V = (camDist > 1e-4) ? (toCam / camDist) : float3(0.0, 0.0, -1.0);

    float3 sunDir = normalize(float3(0.08, 0.24, 0.96));
    float ndl = max(dot(N, sunDir), 0.0);

    uint rtBits = (uint)(g_CameraPos.w + 0.25);
    bool rtEnabled = (rtBits & 1u) != 0u;
    bool rtgiEnabled = (rtBits & 2u) != 0u;

    float sunVis = 1.0;
    if (rtEnabled && ndl > 0.02) {
        RayDesc sRay;
        sRay.Origin = input.worldPos + N * 0.08;
        sRay.Direction = sunDir;
        sRay.TMin = 0.06;
        sRay.TMax = 140.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, sRay);
        while (sq.Proceed()) {}
        if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            sunVis = 0.24;
        }
    }

    float3 sunColor = float3(1.15, 0.68, 0.44) * sunVis;
    float3 ambient = lerp(float3(0.14, 0.10, 0.26), float3(0.32, 0.26, 0.48), N.y * 0.5 + 0.5);

    float emissive = input.matParams.x;
    float specPower = input.matParams.y;
    float3 H = normalize(sunDir + V);
    float spec = pow(max(dot(N, H), 0.0), specPower) * (texIdx == 8u || texIdx == 9u || texIdx == 1u ? 0.65 : 0.18);

    float3 lit = albedo * (ambient + sunColor * ndl) + sunColor * spec;

    [unroll]
    for (int i = 0; i < 4; ++i) {
        float3 toL = g_PointLights[i].xyz - input.worldPos;
        float d2 = dot(toL, toL) + 1.0;
        float att = g_PointLights[i].w / (1.0 + 0.055 * d2);
        float3 L = normalize(toL);
        float diff = max(dot(N, L), 0.0);
        float3 HL = normalize(L + V);
        float pspec = pow(max(dot(N, HL), 0.0), 32.0) * 0.35;
        lit += (albedo * diff + pspec) * g_LightColors[i].rgb * att;
    }

    bool isGlossy = (texIdx == 1u || texIdx == 8u || texIdx == 9u || texIdx == 10u);
    if (rtEnabled && isGlossy) {
        float3 reflRaw = reflect(-V, N);
        float3 R = (dot(reflRaw, reflRaw) > 1e-6) ? normalize(reflRaw) : float3(0.0, 1.0, 0.0);
        RayDesc rRay;
        rRay.Origin = input.worldPos + N * 0.08;
        rRay.Direction = R;
        rRay.TMin = 0.08;
        rRay.TMax = 110.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
        rq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, rRay);
        while (rq.Proceed()) {}

        float3 reflCol = eval_sunset_env(R);
        if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float4 hitData = g_InstColors[rq.CommittedInstanceID()];
            float fade = saturate(1.0 - rq.CommittedRayT() / 110.0);
            float3 objCol = hitData.rgb * (0.55 + hitData.a * 1.55);
            reflCol = lerp(reflCol, objCol, fade);
        }

        float f0 = (texIdx == 8u || texIdx == 9u) ? 0.22 : 0.16;
        float fresnel = f0 + (1.0 - f0) * pow(1.0 - saturate(dot(N, V)), 4.0);
        float strength = (texIdx == 8u || texIdx == 9u) ? 0.50 : 0.38;
        lit = lerp(lit, reflCol, saturate(fresnel * strength));
    }

    lit += albedo * emissive;

    float dist = length(input.worldPos.xz - g_CameraPos.xz);
    float fog = saturate((dist - 55.0) / 145.0);
    float3 fogColor = float3(0.68, 0.28, 0.42);
    float3 finalCol = lerp(lit, fogColor, fog * 0.72);

    float3 gi = float3(0.0, 0.0, 0.0);
    if (rtgiEnabled && camDist < 160.0) {
        gi = trace_diffuse_rtgi(input.worldPos, N, input.pos.xy) * albedo * 0.55 * (1.0 - fog * 0.72);
    }

    GBufferOut o;
    o.directHDR = float4(finalCol, 1.0);
    o.rawRTGI = float4(gi, 1.0);
    o.normalDepth = float4(N, camDist);
    return o;
}
