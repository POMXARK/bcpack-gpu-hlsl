// Open AAA Showcase PBR + Normal Mapping + HDRI IBL + Textured DXR 1.1 Inline RT/RTGI + Motion Vectors
// Shader Model 6.6 Pixel Shader (DirectX Raytracing 1.1 RayQuery + NonUniformResourceIndex)

cbuffer AaaDrawCB : register(b0) {
    float4x4 g_Model;
    float4x4 g_PrevModel;
    float4   g_BaseColor;    // rgba
    float4   g_MatParams;    // x = metallic, y = roughness, z = uvScale, w = mode (0=glTF, 1=glass, 2=PolyHaven, 3=wetAsphalt)
    float4   g_EmissiveTime; // xyz = emissive, w = time
    uint4    g_TexIndices;   // x = albedoIdx, y = normalIdx, z = ormIdx, w = emissiveIdx (0xFFFFFFFF = none)
    float4   g_TileStates;   // x = albedoState, y = normalState, z = ormState (<0 clay, 0..1 fresh wave), w = stripCountY
};

cbuffer AaaFrameCB : register(b1) {
    float4x4 g_VPJittered;
    float4x4 g_VPUnjittered;
    float4x4 g_PrevVPUnjittered;
    float4   g_CameraPos;      // xyz = camera world pos, w = bitflags (1=RT, 2=RTGI, 4=XRay)
    float4   g_SunDirTime;     // xyz = sunDir, w = time
    float4   g_JitterRes;      // xy = jitterNDC, zw = renderW, renderH
};

struct RtInstMeta {
    float4 baseColor; // rgb = tint, a = emissive boost
    uint4  geom;      // x = baseVert, y = idxOffset, z = albedoTexIdx (NO_TEX = none), w = flags
    float4 uvParams;  // x = uvScale, yzw = reserved
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

struct VSIn {
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float4 tangent : TANGENT;
    float2 uv      : TEXCOORD0;
};

struct PSIn {
    float4 pos      : SV_Position;
    float3 worldN   : NORMAL;
    float4 worldT   : TANGENT;
    float2 uv       : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
    float4 curClip  : TEXCOORD2;
    float4 prevClip : TEXCOORD3;
};

struct GBufferOut {
    float4 directHDR   : SV_Target0;
    float4 rawRTGI     : SV_Target1;
    float4 normalDepth : SV_Target2;
    float2 motionVec   : SV_Target3;
};

static const float PI = 3.14159265359;
static const uint NO_TEX = 0xFFFFFFFFu;

PSIn VSMain(VSIn input) {
    PSIn o;
    float4 wp = mul(g_Model, float4(input.pos, 1.0));
    float4 prevWp = mul(g_PrevModel, float4(input.pos, 1.0));
    o.pos = mul(g_VPJittered, wp);
    o.curClip = mul(g_VPUnjittered, wp);
    o.prevClip = mul(g_PrevVPUnjittered, prevWp);

    float3 wn = mul((float3x3)g_Model, input.normal);
    o.worldN = (dot(wn, wn) > 1e-6) ? normalize(wn) : float3(0.0, 1.0, 0.0);

    float3 wt = mul((float3x3)g_Model, input.tangent.xyz);
    float tLen = dot(wt, wt);
    o.worldT = (tLen > 1e-6) ? float4(wt / sqrt(tLen), input.tangent.w) : float4(1.0, 0.0, 0.0, 1.0);

    o.uv = input.uv * g_MatParams.z;
    o.worldPos = wp.xyz;
    return o;
}

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

float distribution_ggx(float ndh, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-5);
}

float geometry_schlick_ggx(float ndv, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return ndv / max(ndv * (1.0 - k) + k, 1e-5);
}

float geometry_smith(float ndv, float ndl, float roughness) {
    return geometry_schlick_ggx(ndv, roughness) * geometry_schlick_ggx(ndl, roughness);
}

float3 fresnel_schlick_roughness(float cosTheta, float3 F0, float roughness) {
    float3 maxVal = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (maxVal - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float ign(float2 xy, float frame) {
    float2 p = xy + 5.588238 * fmod(frame, 64.0);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

float hash21(float2 p) {
    p = frac(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return frac(p.x * p.y);
}

float noise2(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// Evaluate real textured PBR surface on DXR 1.1 triangle hit
float3 eval_dxr_hit(
    uint instId,
    uint primIdx,
    float2 bary,
    float rayT,
    float mipBias,
    float3x4 objToWorld,
    float3 sunDir
) {
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
    float3 lit = albedo * skyHit + directHit + albedo * meta.baseColor.a * 0.45;
    return min(lit, float3(4.0, 4.0, 4.0));
}

float3 sample_rtgi_ray(float3 worldPos, float3 N, float3 T, float3 B, float r1, float r2, float3 sunDir) {
    float phi = 2.0 * PI * r1;
    float cosTheta = sqrt(r2);
    float sinTheta = sqrt(max(1.0 - r2, 0.0));
    float3 giRaw = T * (cos(phi) * sinTheta) + B * (sin(phi) * sinTheta) + N * cosTheta;
    float3 giDir = (dot(giRaw, giRaw) > 1e-6) ? normalize(giRaw) : float3(0.0, 1.0, 0.0);

    RayDesc gRay;
    gRay.Origin = worldPos + N * 0.04;
    gRay.Direction = giDir;
    gRay.TMin = 0.04;
    gRay.TMax = 45.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> gq;
    gq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, gRay);
    while (gq.Proceed()) {}

    if (gq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float fade = saturate(1.0 - gq.CommittedRayT() / 45.0);
        float3 hitCol = eval_dxr_hit(
            gq.CommittedInstanceID(),
            gq.CommittedPrimitiveIndex(),
            gq.CommittedTriangleBarycentrics(),
            gq.CommittedRayT(),
            2.0,
            gq.CommittedObjectToWorld3x4(),
            sunDir
        );
        return hitCol * fade * 1.15;
    }
    return sample_env_lod(giDir, 0.85) * 0.45;
}

float3 trace_diffuse_rtgi(float3 worldPos, float3 N, float2 screenXY, float3 sunDir) {
    float3 upVec = (abs(N.z) < 0.999) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(upVec, N));
    float3 B = cross(N, T);
    float r1a = ign(screenXY, 0.0);
    float r2a = ign(screenXY + float2(17.0, 31.0), 0.0) * 0.5;
    float r1b = frac(r1a + 0.5);
    float r2b = r2a + 0.5;
    return 0.5 * (
        sample_rtgi_ray(worldPos, N, T, B, r1a, r2a, sunDir) +
        sample_rtgi_ray(worldPos, N, T, B, r1b, r2b, sunDir)
    );
}

void record_sfs_feedback(uint texIdx, uint mipFp8) {
    if (texIdx < 32u) {
        uint off = texIdx * 16u;
        g_SfsFeedback.InterlockedMin(off, mipFp8);
        g_SfsFeedback.InterlockedAdd(off + 4u, 1u);
    }
}

GBufferOut PSMain(PSIn input) {
    float3 geomN = (dot(input.worldN, input.worldN) > 1e-6) ? normalize(input.worldN) : float3(0.0, 1.0, 0.0);
    float3 toCam = g_CameraPos.xyz - input.worldPos;
    float camDist = length(toCam);
    float3 V = (camDist > 1e-4) ? (toCam / camDist) : float3(0.0, 0.0, -1.0);

    uint flags = (uint)(g_CameraPos.w + 0.25);
    bool rtEnabled = (flags & 1u) != 0u;
    bool rtgiEnabled = (flags & 2u) != 0u;
    bool xrayEnabled = (flags & 4u) != 0u;

    uint matMode = (uint)(g_MatParams.w + 0.25); // 0=glTF, 1=glass, 2=PolyHaven, 3=wetAsphalt
    bool isGlass = (matMode == 1u);
    bool isPolyHaven = (matMode >= 2u);
    bool isWetAsphalt = (matMode == 3u);

    // Record GPU Sampler Feedback desired mip & pixel hit count (1/16 pixel cadence)
    float texDim = max(g_TileStates.w, 512.0);
    float2 uvDx = ddx(input.uv) * texDim;
    float2 uvDy = ddy(input.uv) * texDim;
    float rhoMax = max(dot(uvDx, uvDx), dot(uvDy, uvDy));
    if ((((uint)input.pos.x & 3u) == 0u) && (((uint)input.pos.y & 3u) == 0u)) {
        uint mipFp8 = (uint)(clamp(0.5 * log2(max(rhoMax, 1.0)), 0.0, 11.0) * 256.0);
        record_sfs_feedback(g_TexIndices.x, mipFp8);
        record_sfs_feedback(g_TexIndices.y, mipFp8);
        record_sfs_feedback(g_TexIndices.z, mipFp8);
        record_sfs_feedback(g_TexIndices.w, mipFp8);
    }

    // 1. Sample 4K BC7/BC5 PBR textures gated by async GPU BCPack + SFS streaming residency
    bool hasAlbedoTex = (g_TexIndices.x != NO_TEX);
    bool albedoResident = !hasAlbedoTex || (g_TileStates.x >= 0.0);

    float3 albedo = g_BaseColor.rgb;
    if (hasAlbedoTex) {
        if (albedoResident) {
            float3 srgbTex = g_Textures[NonUniformResourceIndex(min(g_TexIndices.x, 31u))].Sample(g_LinearWrap, input.uv).rgb;
            albedo *= pow(max(srgbTex, 0.001), 2.2);
        } else {
            albedo = float3(0.46, 0.45, 0.44);
        }
    }

    float metallic = albedoResident ? saturate(g_MatParams.x) : 0.0;
    float roughness = albedoResident ? clamp(g_MatParams.y, 0.04, 0.99) : 0.86;
    float ao = 1.0;

    if (albedoResident && g_TexIndices.z != NO_TEX && g_TileStates.z >= 0.0) {
        float3 orm = g_Textures[NonUniformResourceIndex(min(g_TexIndices.z, 31u))].Sample(g_LinearWrap, input.uv).rgb;
        if (isPolyHaven) {
            ao = lerp(0.35, 1.0, orm.r);
            roughness = clamp(orm.g * max(g_MatParams.y, 0.65), 0.05, 0.99);
            metallic = saturate(orm.b);
        } else {
            roughness = clamp(orm.g * g_MatParams.y, 0.04, 0.99);
            metallic = saturate(orm.b * g_MatParams.x);
        }
    }

    // 2. Tangent-space Normal Mapping
    float3 N = geomN;
    if (albedoResident && g_TexIndices.y != NO_TEX && g_TileStates.y >= 0.0) {
        float3 mapN = g_Textures[NonUniformResourceIndex(min(g_TexIndices.y, 31u))].Sample(g_LinearWrap, input.uv).xyz;
        float2 xy = mapN.xy * 2.0 - 1.0;
        if (!isPolyHaven) {
            xy.y = -xy.y;
            if (!hasAlbedoTex) {
                xy *= 0.06;
            } else {
                xy *= 0.65;
            }
        }
        float z = sqrt(saturate(1.0 - dot(xy, xy)));
        float3 tN = float3(xy, z);

        float3 T = input.worldT.xyz - geomN * dot(geomN, input.worldT.xyz);
        if (dot(T, T) > 1e-6) {
            T = normalize(T);
            float3 B = cross(geomN, T) * ((input.worldT.w < 0.0) ? -1.0 : 1.0);
            N = normalize(T * tN.x + B * tN.y + geomN * tN.z);
        }
    }

    // Wet coastal asphalt puddles
    float puddle = 0.0;
    if (albedoResident && isWetAsphalt) {
        float n1 = noise2(input.worldPos.xz * 0.14);
        float n2 = noise2(input.worldPos.xz * 0.42 + 7.3);
        puddle = smoothstep(0.46, 0.66, n1 * 0.65 + n2 * 0.35);
        float laneWet = smoothstep(14.0, 1.5, abs(input.worldPos.x));
        puddle = saturate(puddle * (0.72 + 0.45 * laneWet));
        N = normalize(lerp(N, geomN, puddle * 0.94));
        roughness = lerp(roughness, 0.065, puddle);
        albedo *= lerp(1.0, 0.50, puddle);
    }

    // Geometric / Normal Specular Anti-Aliasing
    float3 dNdx = ddx(N);
    float3 dNdy = ddy(N);
    float nVar = dot(dNdx, dNdx) + dot(dNdy, dNdy);
    roughness = sqrt(saturate(roughness * roughness + min(nVar * 0.75, 0.25)));

    // 3. HDR Sunset Directional Light + Tight Contact-Biased DXR 1.1 Inline Sun Shadow
    float3 sunDir = normalize(g_SunDirTime.xyz);
    float ndl = max(dot(N, sunDir), 0.0);
    float ndv = max(dot(N, V), 1e-4);

    float sunVis = 1.0;
    if (rtEnabled && ndl > 0.01) {
        RayDesc sRay;
        sRay.Origin = input.worldPos + geomN * 0.015;
        sRay.Direction = sunDir;
        sRay.TMin = 0.015;
        sRay.TMax = 140.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, sRay);
        while (sq.Proceed()) {}
        if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            sunVis = 0.14;
        }
    }

    float3 sunRadiance = float3(1.65, 1.02, 0.55) * sunVis;

    // 4. Cook-Torrance GGX Direct BRDF
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    if (isGlass) {
        albedo = float3(0.015, 0.018, 0.022);
        metallic = 0.0;
        roughness = 0.04;
        F0 = float3(0.10, 0.11, 0.12);
    }

    float3 H = normalize(V + sunDir);
    float ndh = max(dot(N, H), 0.0);
    float hdv = max(dot(H, V), 0.0);
    float D = distribution_ggx(ndh, roughness);
    float G = geometry_smith(ndv, ndl, roughness);
    float3 F = fresnel_schlick_roughness(hdv, F0, roughness);

    float3 specular = min((D * G * F) / max(4.0 * ndv * ndl, 0.001), float3(6.0, 6.0, 6.0));
    float3 kD = (1.0 - F) * (1.0 - metallic);
    float3 directLit = (kD * albedo / PI + specular) * sunRadiance * ndl;

    // Key warm rim / bounce fill from coastal promenade lights
    float3 fillDir = normalize(float3(-0.65, 0.45, -0.55));
    float fillNdl = max(dot(N, fillDir), 0.0);
    directLit += (kD * albedo / PI) * float3(0.28, 0.36, 0.54) * fillNdl;

    // 5. HDRI Equirectangular IBL + Textured Hardware DXR 1.1 Glossy Reflections
    float3 F_ibl = fresnel_schlick_roughness(ndv, F0, roughness);
    float3 kD_ibl = (1.0 - F_ibl) * (1.0 - metallic);
    float3 diffuseIBL = sample_env_diffuse(N) * albedo * kD_ibl * ao;

    float3 reflDir = reflect(-V, N);
    float3 specEnv = min(sample_env_lod(reflDir, roughness), float3(4.5, 4.5, 4.5));

    if (rtEnabled && (roughness < 0.58 || isGlass || puddle > 0.25)) {
        RayDesc rRay;
        rRay.Origin = input.worldPos + geomN * 0.025;
        rRay.Direction = (dot(reflDir, reflDir) > 1e-6) ? normalize(reflDir) : float3(0.0, 1.0, 0.0);
        rRay.TMin = 0.02;
        rRay.TMax = 100.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
        rq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, rRay);
        while (rq.Proceed()) {}

        if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float hitFade = saturate(1.0 - rq.CommittedRayT() / 90.0);
            float glossWeight = saturate(1.0 - roughness * 1.40);
            float3 hitColor = eval_dxr_hit(
                rq.CommittedInstanceID(),
                rq.CommittedPrimitiveIndex(),
                rq.CommittedTriangleBarycentrics(),
                rq.CommittedRayT(),
                roughness * 4.0,
                rq.CommittedObjectToWorld3x4(),
                sunDir
            );
            specEnv = lerp(specEnv, hitColor, hitFade * glossWeight * 0.88);
        }
    }

    float3 specularIBL = specEnv * F_ibl * lerp(0.55, 1.0, ao);
    if (isGlass) {
        float glassFresnel = 0.12 + 0.86 * pow(1.0 - ndv, 4.0);
        directLit = directLit * 0.25 + float3(0.012, 0.014, 0.018) + specEnv * glassFresnel;
        diffuseIBL = 0.0;
        specularIBL = 0.0;
    }

    float3 emissiveCol = g_EmissiveTime.xyz;
    if (albedoResident && g_TexIndices.w != NO_TEX) {
        float3 emTex = g_Textures[NonUniformResourceIndex(min(g_TexIndices.w, 31u))].Sample(g_LinearWrap, input.uv).rgb;
        emissiveCol *= pow(max(emTex, 0.001), 2.2);
    }

    float3 hdr = directLit + diffuseIBL + specularIBL + emissiveCol * 1.6;

    // Subtle warm distance fog near horizon
    float fog = 1.0 - exp(-max(camDist - 60.0, 0.0) * 0.0055);
    float3 fogCol = float3(0.62, 0.34, 0.22);
    hdr = lerp(hdr, fogCol, saturate(fog * 0.60));

    // 6. Output diffuse GI albedo + eligibility mask for Xbox/RDNA2-style Compute RayQuery RTGI pass
    float giMask = (!isGlass && camDist < 120.0) ? 1.0 : 0.0;
    float3 giAlb = albedo * (1.0 - metallic * 0.7) * 0.45 * giMask;

    // 7. [V] X-Ray BCPack 4K-Tile & SFS Residency Heatmap Overlay
    if (xrayEnabled) {
        float strips = clamp(texDim / 16.0, 16.0, 128.0);
        float vCoord = frac(input.uv.y);
        float2 tileCoord = float2(frac(input.uv.x) * 8.0, vCoord * strips);
        float2 tileFrac = frac(tileCoord);
        float edge = (tileFrac.x < 0.04 || tileFrac.y < 0.08) ? 1.0 : 0.0;
        float effState = g_TileStates.x;
        if (!hasAlbedoTex) {
            effState = max(g_TileStates.y, g_TileStates.z);
        }
        if (effState < 0.0) {
            float3 clayHeat = float3(0.95, 0.16, 0.48);
            hdr = lerp(hdr * 0.22, clayHeat, 0.42 + edge * 0.48);
        } else if (effState < 1.0) {
            if (vCoord > effState) {
                float3 clayHeat = float3(0.95, 0.16, 0.48);
                hdr = lerp(hdr * 0.22, clayHeat, 0.42 + edge * 0.48);
            } else if (abs(vCoord - effState) < 0.10) {
                float3 waveCol = float3(0.15, 1.55, 1.75);
                hdr = lerp(hdr * 0.30, waveCol, 0.65 + edge * 0.35);
            } else {
                float3 resCol = float3(0.06, 0.65, 0.28);
                hdr = lerp(hdr * 0.35, resCol, 0.36 + edge * 0.24);
            }
        } else if (effState >= 2.0) {
            // SFS Budget Clamped / Mip-0 Evicted (VRAM saved while Mip 1..Tail remains sharp)
            float3 sfsAmber = float3(1.15, 0.72, 0.10);
            hdr = lerp(hdr * 0.34, sfsAmber, 0.38 + edge * 0.30);
        } else {
            float fresh = saturate(effState - 1.0);
            float3 resCol = float3(0.06, 0.65, 0.28);
            float3 freshCol = float3(0.15, 1.45, 1.65);
            float3 heat = lerp(resCol, freshCol, fresh * 0.65);
            hdr = lerp(hdr * 0.38, heat, 0.32 + fresh * 0.22 + edge * 0.25);
        }
    }

    // 8. Unjittered Screen-Space Motion Vector (prevUV - curUV)
    float2 curUV = float2(input.curClip.x / input.curClip.w * 0.5 + 0.5, 0.5 - input.curClip.y / input.curClip.w * 0.5);
    float2 prevUV = float2(input.prevClip.x / input.prevClip.w * 0.5 + 0.5, 0.5 - input.prevClip.y / input.prevClip.w * 0.5);

    GBufferOut o;
    o.directHDR = float4(hdr, 1.0);
    o.rawRTGI = float4(giAlb, giMask);
    o.normalDepth = float4(geomN, camDist);
    o.motionVec = prevUV - curUV;
    return o;
}
