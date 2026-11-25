// Mesh-Shader City & Props Pixel Shader (ps_6_6) -> 4-MRT G-Buffer + DXR 1.1 + SFS Feedback
#include "aaa_meshlet_common.hlsli"

struct RtInstMeta {
    float4 baseColor;
    uint4  geom;
    float4 uvParams;
};

Texture2D<float4>               g_Textures[32] : register(t0);
Texture2D<float4>               g_EnvHdr       : register(t32);
RaytracingAccelerationStructure g_SceneTlas    : register(t33);
StructuredBuffer<RtInstMeta>    g_InstMeta     : register(t34);
ByteAddressBuffer               g_SceneVB      : register(t35);
StructuredBuffer<uint>          g_SceneIB      : register(t36);
RWByteAddressBuffer             g_SfsFeedback  : register(u0);

SamplerState g_LinearWrap : register(s0);
SamplerState g_EnvSampler : register(s1);

struct GBufferOut {
    float4 directHDR   : SV_Target0;
    float4 rawRTGI     : SV_Target1;
    float4 normalDepth : SV_Target2;
    float2 motionVec   : SV_Target3;
};

static const float PI = 3.14159265359;
static const uint NO_TEX = 0xFFFFFFFFu;

float2 dir_to_equirect_uv(float3 d) {
    float phi = atan2(d.z, d.x);
    float u = frac(phi / 6.28318530718 + 0.5 - 0.115);
    float v = clamp(acos(clamp(d.y, -1.0, 1.0)) / PI, 0.002, 0.998);
    return float2(u, v);
}

float3 sample_env_lod(float3 dir, float roughness) {
    float3 d = (dot(dir, dir) > 1e-6) ? normalize(dir) : float3(0.0, 1.0, 0.0);
    float2 uv = dir_to_equirect_uv(d);
    if (roughness < 0.08) {
        return g_EnvHdr.SampleLevel(g_EnvSampler, uv, 0.0).rgb * 0.50;
    }
    float spread = roughness * roughness * 0.055;
    float3 c = g_EnvHdr.SampleLevel(g_EnvSampler, uv, 0.0).rgb * 0.36;
    c += g_EnvHdr.SampleLevel(g_EnvSampler, float2(frac(uv.x + spread), clamp(uv.y + spread * 0.5, 0.002, 0.998)), 0.0).rgb * 0.16;
    c += g_EnvHdr.SampleLevel(g_EnvSampler, float2(frac(uv.x - spread), clamp(uv.y + spread * 0.5, 0.002, 0.998)), 0.0).rgb * 0.16;
    c += g_EnvHdr.SampleLevel(g_EnvSampler, float2(frac(uv.x + spread), clamp(uv.y - spread * 0.5, 0.002, 0.998)), 0.0).rgb * 0.16;
    c += g_EnvHdr.SampleLevel(g_EnvSampler, float2(frac(uv.x - spread), clamp(uv.y - spread * 0.5, 0.002, 0.998)), 0.0).rgb * 0.16;
    return c * 0.50;
}

float3 sample_env_diffuse(float3 N) {
    float3 up = (abs(N.y) < 0.98) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    float3 irr = sample_env_lod(N, 0.85) * 0.40;
    irr += sample_env_lod(normalize(N + T * 0.65), 0.95) * 0.15;
    irr += sample_env_lod(normalize(N - T * 0.65), 0.95) * 0.15;
    irr += sample_env_lod(normalize(N + B * 0.65), 0.95) * 0.15;
    irr += sample_env_lod(normalize(N - B * 0.65), 0.95) * 0.15;
    return irr;
}

float hash31(float3 p) {
    p = frac(p * float3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return frac((p.x + p.y) * p.z);
}

float ign(float2 xy, float frame) {
    float2 p = xy + 5.588238 * fmod(frame, 64.0);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

float3 perturb_normal(float3 N, float3 wp, float2 uv, float2 bc5xy) {
    float3 dp1 = ddx(wp);
    float3 dp2 = ddy(wp);
    float2 duv1 = ddx(uv);
    float2 duv2 = ddy(uv);
    float3 dp2perp = cross(dp2, N);
    float3 dp1perp = cross(N, dp1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invMax = rsqrt(max(max(dot(T, T), dot(B, B)), 1e-8));
    float2 xy = bc5xy * 2.0 - 1.0;
    float z = sqrt(saturate(1.0 - dot(xy, xy)));
    return normalize(T * (xy.x * invMax) + B * (xy.y * invMax) + N * z);
}

float3 eval_dxr_hit(uint instId, uint primIdx, float2 bary, float rayT, float mipBias, float3x4 objToWorld, float3 sunDir) {
    RtInstMeta meta = g_InstMeta[instId];
    uint baseIdx = meta.geom.y + primIdx * 3u;
    uint i0 = meta.geom.x + g_SceneIB[baseIdx + 0u];
    uint i1 = meta.geom.x + g_SceneIB[baseIdx + 1u];
    uint i2 = meta.geom.x + g_SceneIB[baseIdx + 2u];
    uint o0 = i0 * 48u;
    uint o1 = i1 * 48u;
    uint o2 = i2 * 48u;
    float3 n0 = asfloat(g_SceneVB.Load3(o0 + 12u));
    float3 n1 = asfloat(g_SceneVB.Load3(o1 + 12u));
    float3 n2 = asfloat(g_SceneVB.Load3(o2 + 12u));
    float2 uv0 = asfloat(g_SceneVB.Load2(o0 + 40u));
    float2 uv1 = asfloat(g_SceneVB.Load2(o1 + 40u));
    float2 uv2 = asfloat(g_SceneVB.Load2(o2 + 40u));
    float w0 = 1.0 - bary.x - bary.y;
    float3 localN = n0 * w0 + n1 * bary.x + n2 * bary.y;
    float3 worldN = mul(objToWorld, float4(localN, 0.0));
    worldN = (dot(worldN, worldN) > 1e-6) ? normalize(worldN) : float3(0.0, 1.0, 0.0);
    float2 hitUV = (uv0 * w0 + uv1 * bary.x + uv2 * bary.y) * meta.uvParams.x;
    float3 albedo = meta.baseColor.rgb;
    if (meta.geom.z != NO_TEX) {
        float mip = clamp(log2(max(rayT * 0.25, 1.0)) + mipBias, 1.0, 8.0);
        float3 texel = g_Textures[NonUniformResourceIndex(min(meta.geom.z, 31u))].SampleLevel(g_LinearWrap, hitUV, mip).rgb;
        albedo *= pow(max(texel, 0.002), 2.2);
    }
    float hitNdl = max(dot(worldN, sunDir), 0.0);
    float3 skyHit = sample_env_lod(worldN, 0.85) * 0.65;
    float3 directHit = albedo * float3(1.55, 0.94, 0.50) * hitNdl;
    return min(albedo * skyHit + directHit + albedo * meta.baseColor.a * 0.45, float3(4.0, 4.0, 4.0));
}

GBufferOut PSMain(AaaMSOut input) {
    float3 geomN = (dot(input.worldN, input.worldN) > 1e-6) ? normalize(input.worldN) : float3(0.0, 1.0, 0.0);
    float3 toCam = g_CameraPos.xyz - input.worldPos;
    float camDist = length(toCam);
    float3 V = (camDist > 1e-4) ? (toCam / camDist) : float3(0.0, 0.0, -1.0);

    uint flags = (uint)(g_CameraPos.w + 0.25);
    bool rtEnabled = (flags & 1u) != 0u;
    bool rtgiEnabled = (flags & 2u) != 0u;
    bool xrayEnabled = (flags & 4u) != 0u;

    uint matId = input.matId;
    bool concreteResident = (g_ConcreteTex.x != NO_TEX) && (g_CullRange.y >= 0.0);

    // Record SFS feedback for 4K Concrete when concrete facade/seawall pixels are shaded
    if (matId == 0u && g_ConcreteTex.x < 32u && (((uint)input.pos.x & 3u) == 0u) && (((uint)input.pos.y & 3u) == 0u)) {
        float texDim = max(g_CullRange.w, 1024.0);
        float2 uvDx = ddx(input.uv) * texDim;
        float2 uvDy = ddy(input.uv) * texDim;
        float rho = max(dot(uvDx, uvDx), dot(uvDy, uvDy));
        uint mipFp8 = (uint)(clamp(0.5 * log2(max(rho, 1.0)), 0.0, 11.0) * 256.0);
        uint offA = g_ConcreteTex.x * 16u;
        g_SfsFeedback.InterlockedMin(offA, mipFp8);
        g_SfsFeedback.InterlockedAdd(offA + 4u, 1u);
        if (g_ConcreteTex.y < 32u) {
            uint offN = g_ConcreteTex.y * 16u;
            g_SfsFeedback.InterlockedMin(offN, mipFp8);
            g_SfsFeedback.InterlockedAdd(offN + 4u, 1u);
        }
    }

    float3 N = geomN;
    float3 albedo = input.instTint;
    float metallic = 0.0;
    float roughness = 0.72;
    float ao = 1.0;
    float3 emissive = float3(0.0, 0.0, 0.0);

    if (matId == 0u) {
        // 4K BC7/BC5 Architectural Concrete Facades, Cornices, Balconies & Seawall
        if (concreteResident) {
            float3 srgb = g_Textures[NonUniformResourceIndex(min(g_ConcreteTex.x, 31u))].Sample(g_LinearWrap, input.uv).rgb;
            albedo *= pow(max(srgb, 0.002), 2.2);
            if (g_ConcreteTex.y != NO_TEX && g_CullRange.z >= 0.0) {
                float2 bc5 = g_Textures[NonUniformResourceIndex(min(g_ConcreteTex.y, 31u))].Sample(g_LinearWrap, input.uv).xy;
                N = perturb_normal(geomN, input.worldPos, input.uv, bc5);
            }
            if (g_ConcreteTex.z != NO_TEX) {
                float3 arm = g_Textures[NonUniformResourceIndex(min(g_ConcreteTex.z, 31u))].Sample(g_LinearWrap, input.uv).rgb;
                ao = lerp(0.45, 1.0, arm.r);
                roughness = clamp(arm.g * 0.92, 0.15, 0.96);
            }
        } else {
            albedo *= 0.48;
        }
        metallic = 0.04;
    } else if (matId == 1u) {
        // Recessed Reflective Curtain-Wall Glass with subtle warm evening office lighting
        float winRnd = hash31(floor(input.worldPos * 0.35));
        albedo = float3(0.018, 0.025, 0.036);
        metallic = 0.05;
        roughness = 0.06;
        if (winRnd > 0.56) {
            float lvl = 0.12 + 0.28 * frac(winRnd * 13.7);
            emissive = lerp(float3(1.10, 0.68, 0.32), float3(0.75, 0.92, 1.05), step(0.84, winRnd)) * lvl;
        }
    } else if (matId == 2u) {
        // Ground-floor coastal storefronts & architectural neon strips
        float sRnd = hash31(floor(input.worldPos * 0.25));
        albedo = input.instTint * 0.22;
        roughness = 0.20;
        emissive = (sRnd > 0.45) ? float3(1.45, 0.72, 0.32) : float3(0.28, 1.15, 1.25);
    } else if (matId == 3u) {
        // Anodized architectural dark bronze/steel mullions, railings, lamppost poles
        albedo = float3(0.09, 0.095, 0.105) * input.instTint;
        metallic = 0.86;
        roughness = 0.30;
    } else if (matId == 4u) {
        // Tropical palm trunk bark
        float bark = 0.78 + 0.22 * sin(input.uv.y * 55.0);
        albedo = float3(0.26, 0.17, 0.11) * bark;
        roughness = 0.88;
    } else if (matId == 5u) {
        // Palm fronds foliage
        float vein = 0.85 + 0.15 * sin(input.uv.x * 24.0);
        albedo = float3(0.08, 0.26, 0.07) * vein;
        roughness = 0.52;
    } else if (matId == 6u) {
        // Emissive streetlamp globes & road markings
        albedo = input.instTint;
        roughness = 0.42;
        if (input.worldPos.y > 1.5) {
            emissive = input.instTint * 2.4;
        }
    } else {
        // Distant Bayfront Skyline HLOD silhouette
        float wGrid = step(0.72, hash31(floor(input.worldPos * 0.28)));
        albedo = float3(0.24, 0.22, 0.26) * input.instTint;
        roughness = 0.48;
        emissive = float3(1.15, 0.72, 0.36) * wGrid * 0.16;
    }

    // Specular Anti-Aliasing
    float3 dNdx = ddx(N);
    float3 dNdy = ddy(N);
    roughness = sqrt(saturate(roughness * roughness + min((dot(dNdx, dNdx) + dot(dNdy, dNdy)) * 0.75, 0.22)));

    float3 sunDir = normalize(g_SunDirTime.xyz);
    float ndl = max(dot(N, sunDir), 0.0);
    float ndv = max(dot(N, V), 1e-4);

    float sunVis = 1.0;
    if (rtEnabled && ndl > 0.01 && camDist < 180.0 && matId != 7u) {
        RayDesc sRay;
        sRay.Origin = input.worldPos + geomN * 0.03;
        sRay.Direction = sunDir;
        sRay.TMin = 0.03;
        sRay.TMax = 140.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, sRay);
        while (sq.Proceed()) {}
        if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            sunVis = 0.18;
        }
    }

    float3 sunRadiance = float3(1.65, 1.02, 0.55) * sunVis;
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    if (matId == 1u) {
        F0 = float3(0.10, 0.11, 0.12);
    }

    float3 H = normalize(V + sunDir);
    float ndh = max(dot(N, H), 0.0);
    float hdv = max(dot(H, V), 0.0);
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = ndh * ndh * (a2 - 1.0) + 1.0;
    float D = a2 / max(PI * denom * denom, 1e-5);
    float r1 = roughness + 1.0;
    float k = (r1 * r1) / 8.0;
    float G = (ndv / max(ndv * (1.0 - k) + k, 1e-5)) * (ndl / max(ndl * (1.0 - k) + k, 1e-5));
    float3 F = F0 + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) * pow(saturate(1.0 - hdv), 5.0);

    float3 spec = min((D * G * F) / max(4.0 * ndv * ndl, 0.001), float3(6.0, 6.0, 6.0));
    float3 kD = (1.0 - F) * (1.0 - metallic);
    float3 directLit = (kD * albedo / PI + spec) * sunRadiance * ndl;

    // Foliage & thin rim sunset wrap on palms
    if (matId == 5u) {
        float backSun = pow(saturate(dot(-V, sunDir)), 3.0);
        directLit += albedo * float3(1.2, 0.85, 0.30) * backSun * 0.45;
    }

    float3 F_ibl = F0 + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) * pow(saturate(1.0 - ndv), 5.0);
    float3 kD_ibl = (1.0 - F_ibl) * (1.0 - metallic);
    float3 diffuseIBL = sample_env_diffuse(N) * albedo * kD_ibl * ao;

    float3 reflDir = reflect(-V, N);
    float3 specEnv = min(sample_env_lod(reflDir, roughness), float3(4.5, 4.5, 4.5));
    if (rtEnabled && matId == 1u && camDist < 120.0) {
        RayDesc rRay;
        rRay.Origin = input.worldPos + geomN * 0.04;
        rRay.Direction = (dot(reflDir, reflDir) > 1e-6) ? normalize(reflDir) : float3(0.0, 1.0, 0.0);
        rRay.TMin = 0.03;
        rRay.TMax = 90.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
        rq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, rRay);
        while (rq.Proceed()) {}
        if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float3 hitColor = eval_dxr_hit(
                rq.CommittedInstanceID(),
                rq.CommittedPrimitiveIndex(),
                rq.CommittedTriangleBarycentrics(),
                rq.CommittedRayT(),
                1.0,
                rq.CommittedObjectToWorld3x4(),
                sunDir
            );
            specEnv = lerp(specEnv, hitColor, saturate(1.0 - rq.CommittedRayT() / 90.0) * 0.80);
        }
    }

    float3 specIBL = specEnv * F_ibl * lerp(0.65, 1.0, ao);
    float3 hdr = directLit + diffuseIBL + specIBL + emissive;

    // Warm sunset distance aerial perspective
    float fogStart = (matId == 7u) ? 90.0 : 65.0;
    float fog = 1.0 - exp(-max(camDist - fogStart, 0.0) * 0.0042);
    float3 fogCol = float3(0.66, 0.36, 0.24);
    hdr = lerp(hdr, fogCol, saturate(fog * 0.65));

    // Output diffuse GI albedo + eligibility mask for Xbox/RDNA2-style Compute RayQuery RTGI pass
    float giMask = (camDist < 85.0 && matId != 1u && matId != 6u && matId != 7u) ? 1.0 : 0.0;
    float3 giAlb = albedo * (1.0 - metallic * 0.7) * 0.40 * giMask;

    // [V] X-Ray overlay (Concrete SFS state + Mesh-Shader highlight)
    if (xrayEnabled) {
        float strips = clamp(max(g_CullRange.w, 1024.0) / 16.0, 16.0, 128.0);
        float2 tileFrac = frac(float2(frac(input.uv.x) * 8.0, frac(input.uv.y) * strips));
        float edge = (tileFrac.x < 0.04 || tileFrac.y < 0.08) ? 1.0 : 0.0;
        if (matId == 0u) {
            float st = g_CullRange.y;
            float3 col = (st < 0.0) ? float3(0.95, 0.16, 0.48)
                       : (st >= 2.0) ? float3(1.15, 0.72, 0.10)
                       : float3(0.06, 0.75, 0.35);
            hdr = lerp(hdr * 0.34, col, 0.38 + edge * 0.30);
        } else {
            float3 msCol = float3(0.18, 0.85, 1.15);
            hdr = lerp(hdr * 0.35, msCol, 0.25 + edge * 0.25);
        }
    }

    float2 curUV = float2(input.curClip.x / input.curClip.w * 0.5 + 0.5, 0.5 - input.curClip.y / input.curClip.w * 0.5);
    float2 prevUV = float2(input.prevClip.x / input.prevClip.w * 0.5 + 0.5, 0.5 - input.prevClip.y / input.prevClip.w * 0.5);

    GBufferOut o;
    o.directHDR = float4(hdr, 1.0);
    o.rawRTGI = float4(giAlb, giMask);
    o.normalDepth = float4(geomN, camDist);
    o.motionVec = prevUV - curUV;
    return o;
}
