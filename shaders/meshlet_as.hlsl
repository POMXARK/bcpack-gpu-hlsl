#include "meshlet_common.hlsli"

RWByteAddressBuffer g_CullStats : register(u0, space1);

groupshared Payload s_Payload;

[numthreads(32, 1, 1)]
void ASMain(uint gtid : SV_GroupThreadID, uint gid : SV_DispatchThreadID) {
    bool inBounds = (gid < g_TotalWorkItems);
    bool visible = false;
    uint primCount = 0u;

    if (inBounds) {
        uint2 work = g_WorkItems[gid];
        MeshletInstance inst = g_Instances[work.x];
        MeshletDesc m = g_Meshlets[work.y];
        primCount = m.primCount;

        float s, c;
        sincos(inst.rotY, s, c);
        float3 sp = m.boundCenter * inst.scale;
        float3 worldCenter = float3(
            sp.x * c + sp.z * s + inst.pos.x,
            sp.y + inst.pos.y,
            -sp.x * s + sp.z * c + inst.pos.z
        );
        float maxScale = max(inst.scale.x, max(inst.scale.y, inst.scale.z));
        float worldRadius = m.boundRadius * maxScale;

        float dist = length(worldCenter - g_CameraPos.xyz);
        visible = (dist <= g_MaxCullDist + worldRadius);
        if (visible) {
            [unroll]
            for (uint i = 0u; i < 6u; ++i) {
                float d = dot(g_FrustumPlanes[i].xyz, worldCenter) + g_FrustumPlanes[i].w;
                if (d < -worldRadius) {
                    visible = false;
                    break;
                }
            }
        }
    }

    uint outSlot = WavePrefixCountBits(visible);
    if (visible) {
        s_Payload.workIndices[outSlot] = gid;
    }

    uint visCount = WaveActiveCountBits(visible);
    uint visTris = WaveActiveSum(visible ? primCount : 0u);
    uint cullCount = WaveActiveCountBits(inBounds && !visible);

    if (WaveIsFirstLane()) {
        if (visCount > 0u) {
            g_CullStats.InterlockedAdd(0, visCount);
            g_CullStats.InterlockedAdd(4, visTris);
        }
        if (cullCount > 0u) {
            g_CullStats.InterlockedAdd(8, cullCount);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(visCount, 1u, 1u, s_Payload);
}
