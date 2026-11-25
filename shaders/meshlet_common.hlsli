// Shared data structures & bindings for Amplification / Mesh / Pixel Shaders (SM 6.5)

cbuffer MeshFrameCB : register(b0) {
    float4x4 g_ViewProj;
    float4   g_FrustumPlanes[6];
    float4   g_CameraPos;     // xyz = eye, w = rt_enabled
    float4   g_PointLights[4];
    float4   g_LightColors[4];
    uint     g_TotalWorkItems;
    float    g_Time;
    float    g_MaxCullDist;
    uint     g_Pad;
};

struct MeshletVertex {
    float3 pos;
    float3 normal;
    float2 uv;
};

struct MeshletDesc {
    uint   vertOffset;
    uint   vertCount;
    uint   primOffset;
    uint   primCount;
    float3 boundCenter;
    float  boundRadius;
};

struct MeshletInstance {
    float3 pos;
    float  rotY;
    float3 scale;
    uint   texIndex;
    float  emissive;
    float  uvScale;
    float  specPower;
    float  pad;
};

struct Payload {
    uint workIndices[32];
};

struct MSOut {
    float4               pos       : SV_Position;
    float3               worldN    : NORMAL;
    float2               uv        : TEXCOORD0;
    float3               worldPos  : TEXCOORD1;
    nointerpolation uint texIdx    : TEXCOORD2;
    nointerpolation float2 matParams : TEXCOORD3; // x = emissive, y = specPower
};

StructuredBuffer<MeshletVertex>   g_Vertices    : register(t0, space1);
StructuredBuffer<uint>            g_PrimIndices : register(t1, space1);
StructuredBuffer<MeshletDesc>     g_Meshlets    : register(t2, space1);
StructuredBuffer<MeshletInstance> g_Instances   : register(t3, space1);
StructuredBuffer<uint2>           g_WorkItems   : register(t4, space1);
