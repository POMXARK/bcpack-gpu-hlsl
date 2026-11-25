// GPU-Skinned 500-NPC Crowd Vertex + Pixel Shader (vs_6_0 / ps_6_6) -> 4-MRT G-Buffer

cbuffer CrowdDrawCB : register(b0) {
    float4x4 g_VPJittered;
    float4x4 g_VPUnjittered;
    float4x4 g_PrevVPUnjittered;
    float4   g_CameraPos;   // xyz = eye, w = flags (1=RT, 2=RTGI, 4=XRay)
    float4   g_SunDirTime;  // xyz = sunDir, w = time
    uint4    g_CrowdParams; // x = vertsPerNpc, y = npcCount, z = flags, w = pad
};

struct OutSkinnedVertex {
    float4 pos;
    float4 normal;
};

struct RtInstMeta {
    float4 baseColor;
    uint4  geom;
    float4 uvParams;
};

StructuredBuffer<OutSkinnedVertex> g_CurVerts  : register(t0, space1);
StructuredBuffer<OutSkinnedVertex> g_PrevVerts : register(t1, space1);
StructuredBuffer<uint>             g_PartIds   : register(t2, space1);

Texture2D<float4>               g_Textures[32] : register(t0);
Texture2D<float4>               g_EnvHdr       : register(t32);
RaytracingAccelerationStructure g_SceneTlas    : register(t33);

SamplerState g_LinearWrap : register(s0);
SamplerState g_EnvSampler : register(s1);

struct CrowdPSIn {
    float4               pos      : SV_Position;
    float3               worldN   : NORMAL;
    float3               worldPos : TEXCOORD0;
    float4               curClip  : TEXCOORD1;
    float4               prevClip : TEXCOORD2;
    nointerpolation float3 color  : TEXCOORD3;
};

struct GBufferOut {
    float4 directHDR   : SV_Target0;
    float4 rawRTGI     : SV_Target1;
    float4 normalDepth : SV_Target2;
    float2 motionVec   : SV_Target3;
};

static const float PI = 3.14159265359;

float3 pick_npc_color(uint npcId, uint partId) {
    if (partId == 0u) {
        // Skin tone variety
        uint s = npcId % 4u;
        if (s == 0u) return float3(0.76, 0.56, 0.44);
        if (s == 1u) return float3(0.58, 0.40, 0.29);
        if (s == 2u) return float3(0.84, 0.67, 0.54);
        return float3(0.42, 0.28, 0.20);
    }
    if (partId == 1u) {
        // Coastal casual shirts/jackets
        uint c = (npcId * 7u + 3u) % 8u;
        if (c == 0u) return float3(0.86, 0.22, 0.22);
        if (c == 1u) return float3(0.18, 0.58, 0.82);
        if (c == 2u) return float3(0.92, 0.78, 0.28);
        if (c == 3u) return float3(0.88, 0.88, 0.90);
        if (c == 4u) return float3(0.20, 0.70, 0.52);
        if (c == 5u) return float3(0.18, 0.20, 0.26);
        if (c == 6u) return float3(0.90, 0.44, 0.22);
        return float3(0.62, 0.34, 0.76);
    }
    if (partId == 2u) {
        // Trousers / jeans / shorts
        uint p = (npcId * 5u + 1u) % 4u;
        if (p == 0u) return float3(0.12, 0.16, 0.24);
        if (p == 1u) return float3(0.20, 0.20, 0.22);
        if (p == 2u) return float3(0.48, 0.42, 0.34);
        return float3(0.14, 0.26, 0.42);
    }
    // Shoes / hair dark trim
    return float3(0.10, 0.10, 0.12);
}

CrowdPSIn VSMain(uint vid : SV_VertexID, uint instId : SV_InstanceID) {
    uint baseIdx = instId * g_CrowdParams.x + vid;
    OutSkinnedVertex cur = g_CurVerts[baseIdx];
    OutSkinnedVertex prv = g_PrevVerts[baseIdx];
    uint partId = g_PartIds[vid];

    CrowdPSIn o;
    o.pos = mul(g_VPJittered, float4(cur.pos.xyz, 1.0));
    o.worldN = (dot(cur.normal.xyz, cur.normal.xyz) > 1e-6) ? normalize(cur.normal.xyz) : float3(0.0, 1.0, 0.0);
    o.worldPos = cur.pos.xyz;
    o.curClip = mul(g_VPUnjittered, float4(cur.pos.xyz, 1.0));
    o.prevClip = mul(g_PrevVPUnjittered, float4(prv.pos.xyz, 1.0));
    o.color = pick_npc_color(instId, partId);
    return o;
}

float2 dir_to_equirect_uv(float3 d) {
    float phi = atan2(d.z, d.x);
    float u = frac(phi / 6.28318530718 + 0.5 - 0.115);
    float v = clamp(acos(clamp(d.y, -1.0, 1.0)) / PI, 0.002, 0.998);
    return float2(u, v);
}

float3 sample_env_lod(float3 dir) {
    float3 d = (dot(dir, dir) > 1e-6) ? normalize(dir) : float3(0.0, 1.0, 0.0);
    return g_EnvHdr.SampleLevel(g_EnvSampler, dir_to_equirect_uv(d), 0.0).rgb * 0.45;
}

GBufferOut PSMain(CrowdPSIn input) {
    float3 N = (dot(input.worldN, input.worldN) > 1e-6) ? normalize(input.worldN) : float3(0.0, 1.0, 0.0);
    float3 toCam = g_CameraPos.xyz - input.worldPos;
    float camDist = length(toCam);
    float3 V = (camDist > 1e-4) ? (toCam / camDist) : float3(0.0, 0.0, -1.0);

    uint flags = g_CrowdParams.z;
    bool rtEnabled = (flags & 1u) != 0u;
    bool xrayEnabled = (flags & 4u) != 0u;

    float3 sunDir = normalize(g_SunDirTime.xyz);
    float ndl = max(dot(N, sunDir), 0.0);
    float ndv = max(dot(N, V), 1e-4);

    float sunVis = 1.0;
    if (rtEnabled && ndl > 0.01 && camDist < 120.0) {
        RayDesc sRay;
        sRay.Origin = input.worldPos + N * 0.03;
        sRay.Direction = sunDir;
        sRay.TMin = 0.03;
        sRay.TMax = 100.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, sRay);
        while (sq.Proceed()) {}
        if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            sunVis = 0.22;
        }
    }

    float3 albedo = input.color;
    float3 skyAmb = sample_env_lod(N) * 0.85;
    float3 sunCol = float3(1.60, 1.00, 0.55) * sunVis;
    float3 specH = normalize(V + sunDir);
    float specAmt = pow(max(dot(N, specH), 0.0), 24.0) * 0.14 * sunVis;
    float3 hdr = albedo * (skyAmb + sunCol * ndl / PI) + float3(specAmt, specAmt * 0.8, specAmt * 0.6);

    float fog = 1.0 - exp(-max(camDist - 65.0, 0.0) * 0.0045);
    hdr = lerp(hdr, float3(0.66, 0.36, 0.24), saturate(fog * 0.62));

    if (xrayEnabled) {
        hdr = lerp(hdr * 0.35, float3(0.22, 1.25, 0.85), 0.45);
    }

    float2 curUV = float2(input.curClip.x / input.curClip.w * 0.5 + 0.5, 0.5 - input.curClip.y / input.curClip.w * 0.5);
    float2 prevUV = float2(input.prevClip.x / input.prevClip.w * 0.5 + 0.5, 0.5 - input.prevClip.y / input.prevClip.w * 0.5);

    GBufferOut o;
    o.directHDR = float4(hdr, 1.0);
    o.rawRTGI = float4(albedo * 0.35, (camDist < 95.0) ? 1.0 : 0.0);
    o.normalDepth = float4(N, camDist);
    o.motionVec = prevUV - curUV;
    return o;
}
