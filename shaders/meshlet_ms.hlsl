#include "meshlet_common.hlsli"

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void MSMain(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload Payload payload,
    out vertices MSOut verts[64],
    out indices uint3 tris[126]
) {
    uint workIdx = payload.workIndices[gid.x];
    uint2 work = g_WorkItems[workIdx];
    MeshletInstance inst = g_Instances[work.x];
    MeshletDesc m = g_Meshlets[work.y];

    SetMeshOutputCounts(m.vertCount, m.primCount);

    float s, c;
    sincos(inst.rotY, s, c);

    if (gtid < m.vertCount) {
        MeshletVertex v = g_Vertices[m.vertOffset + gtid];
        float3 sp = v.pos * inst.scale;
        float3 wp = float3(
            sp.x * c + sp.z * s + inst.pos.x,
            sp.y + inst.pos.y,
            -sp.x * s + sp.z * c + inst.pos.z
        );

        float3 invScale = 1.0 / max(inst.scale, float3(1e-4, 1e-4, 1e-4));
        float3 sn = v.normal * invScale;
        float3 wnRaw = float3(
            sn.x * c + sn.z * s,
            sn.y,
            -sn.x * s + sn.z * c
        );
        float3 wn = (dot(wnRaw, wnRaw) > 1e-8) ? normalize(wnRaw) : float3(0.0, 1.0, 0.0);

        MSOut o;
        o.pos = mul(g_ViewProj, float4(wp, 1.0));
        o.worldN = wn;
        o.uv = v.uv * inst.uvScale;
        o.worldPos = wp;
        o.texIdx = min(inst.texIndex, 11u);
        o.matParams = float2(inst.emissive, inst.specPower);
        verts[gtid] = o;
    }

    for (uint p = gtid; p < m.primCount; p += 64u) {
        uint pk = g_PrimIndices[m.primOffset + p];
        tris[p] = uint3(pk & 0xFFu, (pk >> 8u) & 0xFFu, (pk >> 16u) & 0xFFu);
    }
}
