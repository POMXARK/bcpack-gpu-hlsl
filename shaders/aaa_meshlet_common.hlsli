// Shared data structures & bindings for Stage 3 Modular Mesh-Shader City (SM 6.5 / 6.6)

cbuffer AaaMeshFrameCB : register(b0) {
    float4x4 g_VPJittered;
    float4x4 g_VPUnjittered;
    float4x4 g_PrevVPUnjittered;
    float4   g_FrustumPlanes[6];
    float4   g_CameraPos;   // xyz = eye, w = flags (1=RT, 2=RTGI, 4=XRay)
    float4   g_SunDirTime;  // xyz = sunDir, w = time
    uint4    g_CullParams;  // x = totalWorkItems, y = hizEnabled, z = hizW, w = hizH
    float4   g_CullRange;   // x = maxCullDist, y = concreteAlbedoState, z = concreteNormState, w = concreteDim
    uint4    g_ConcreteTex; // x = albedoIdx, y = normalIdx, z = armIdx, w = unused
};

struct AaaMeshletVertex {
    float3 pos;
    float3 normal;
    float2 uv;
    uint   matId;
    uint   _pad;
};

struct AaaMeshletDesc {
    uint   vertOffset;
    uint   vertCount;
    uint   primOffset;
    uint   primCount;
    float3 boundCenter;
    float  boundRadius;
};

struct AaaMeshletInstance {
    float3 pos;
    float  rotY;
    float3 scale;
    float  tintR;
    float  tintG;
    float  tintB;
    float  uvScale;
    uint   flags; // bit 0 = distant HLOD horizon tower (extend cull dist)
};

struct AaaPayload {
    uint workIndices[32];
};

struct AaaMSOut {
    float4                   pos       : SV_Position;
    float3                   worldN    : NORMAL;
    float2                   uv        : TEXCOORD0;
    float3                   worldPos  : TEXCOORD1;
    float4                   curClip   : TEXCOORD2;
    float4                   prevClip  : TEXCOORD3;
    nointerpolation uint     matId     : TEXCOORD4;
    nointerpolation float3   instTint  : TEXCOORD5;
};

StructuredBuffer<AaaMeshletVertex>   g_Vertices    : register(t0, space1);
StructuredBuffer<uint>               g_PrimIndices : register(t1, space1);
StructuredBuffer<AaaMeshletDesc>     g_Meshlets    : register(t2, space1);
StructuredBuffer<AaaMeshletInstance> g_Instances   : register(t3, space1);
StructuredBuffer<uint2>              g_WorkItems   : register(t4, space1);
