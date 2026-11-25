// Amplification Shader (as_6_5): Wave32 Frustum + Conservative Prev-Frame Hi-Z Occlusion Culling
#include "aaa_meshlet_common.hlsli"

Texture2D<float>    g_HiZ0      : register(t0, space2);
Texture2D<float>    g_HiZ1      : register(t1, space2);
Texture2D<float>    g_HiZ2      : register(t2, space2);
Texture2D<float>    g_HiZ3      : register(t3, space2);
Texture2D<float>    g_HiZ4      : register(t4, space2);
Texture2D<float>    g_HiZ5      : register(t5, space2);
RWByteAddressBuffer g_CullStats : register(u0, space1);

groupshared AaaPayload s_Payload;

float sample_hiz(uint mip, int2 px) {
    if (mip == 0u) return g_HiZ0.Load(int3(px, 0));
    if (mip == 1u) return g_HiZ1.Load(int3(px, 0));
    if (mip == 2u) return g_HiZ2.Load(int3(px, 0));
    if (mip == 3u) return g_HiZ3.Load(int3(px, 0));
    if (mip == 4u) return g_HiZ4.Load(int3(px, 0));
    return g_HiZ5.Load(int3(px, 0));
}

[numthreads(32, 1, 1)]
void ASMain(uint gtid : SV_GroupThreadID, uint gid : SV_DispatchThreadID) {
    uint totalWork = g_CullParams.x;
    bool inBounds = (gid < totalWork);
    bool frustumPass = false;
    bool hizOccluded = false;
    uint primCount = 0u;

    if (inBounds) {
        uint2 work = g_WorkItems[gid];
        AaaMeshletInstance inst = g_Instances[work.x];
        AaaMeshletDesc m = g_Meshlets[work.y];
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

        float maxDist = ((inst.flags & 1u) != 0u) ? 900.0 : g_CullRange.x;
        float dist = length(worldCenter - g_CameraPos.xyz);
        frustumPass = (dist <= maxDist + worldRadius);

        if (frustumPass) {
            [unroll]
            for (uint i = 0u; i < 6u; ++i) {
                float d = dot(g_FrustumPlanes[i].xyz, worldCenter) + g_FrustumPlanes[i].w;
                if (d < -worldRadius) {
                    frustumPass = false;
                    break;
                }
            }
        }

        // Conservative Prev-Frame Hi-Z Occlusion Culling
        if (frustumPass && g_CullParams.y != 0u && dist > worldRadius * 1.35 + 2.0) {
            float3 viewDir = (worldCenter - g_CameraPos.xyz) / max(dist, 1e-4);
            float3 nearPt = worldCenter - viewDir * worldRadius;
            float4 clipNear = mul(g_PrevVPUnjittered, float4(nearPt, 1.0));
            if (clipNear.w > 0.10) {
                float sphereMinZ = clipNear.z / clipNear.w;
                float2 minUv = float2(1.0, 1.0);
                float2 maxUv = float2(0.0, 0.0);
                bool validProj = true;
                [unroll]
                for (int cx = -1; cx <= 1; cx += 2) {
                    [unroll]
                    for (int cy = -1; cy <= 1; cy += 2) {
                        [unroll]
                        for (int cz = -1; cz <= 1; cz += 2) {
                            float3 corner = worldCenter + float3((float)cx, (float)cy, (float)cz) * worldRadius;
                            float4 clip = mul(g_PrevVPUnjittered, float4(corner, 1.0));
                            if (clip.w <= 0.10) {
                                validProj = false;
                            } else {
                                float2 ndc = clip.xy / clip.w;
                                float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                                minUv = min(minUv, uv);
                                maxUv = max(maxUv, uv);
                            }
                        }
                    }
                }
                if (validProj && maxUv.x >= 0.0 && minUv.x <= 1.0 && maxUv.y >= 0.0 && minUv.y <= 1.0) {
                    minUv = saturate(minUv);
                    maxUv = saturate(maxUv);
                    float hizW = (float)g_CullParams.z;
                    float hizH = (float)g_CullParams.w;
                    float rectW = (maxUv.x - minUv.x) * hizW;
                    float rectH = (maxUv.y - minUv.y) * hizH;
                    float maxDim = max(max(rectW, rectH), 1.0);
                    uint mip = (uint)clamp((int)ceil(log2(maxDim)), 0, 5);

                    int mipW = max((int)(g_CullParams.z >> mip), 1);
                    int mipH = max((int)(g_CullParams.w >> mip), 1);
                    int2 p0 = clamp(int2(minUv * float2((float)mipW, (float)mipH)), int2(0, 0), int2(mipW - 1, mipH - 1));
                    int2 p1 = clamp(int2(maxUv * float2((float)mipW, (float)mipH)), int2(0, 0), int2(mipW - 1, mipH - 1));

                    float d00 = sample_hiz(mip, int2(p0.x, p0.y));
                    float d10 = sample_hiz(mip, int2(p1.x, p0.y));
                    float d01 = sample_hiz(mip, int2(p0.x, p1.y));
                    float d11 = sample_hiz(mip, int2(p1.x, p1.y));
                    float maxZ = max(max(d00, d10), max(d01, d11));
                    if (sphereMinZ > maxZ + 0.0020) {
                        hizOccluded = true;
                    }
                }
            }
        }
    }

    bool visible = inBounds && frustumPass && !hizOccluded;
    bool frustumCulled = inBounds && !frustumPass;
    bool occluded = inBounds && frustumPass && hizOccluded;

    uint outSlot = WavePrefixCountBits(visible);
    if (visible) {
        s_Payload.workIndices[outSlot] = gid;
    }

    uint visCount = WaveActiveCountBits(visible);
    uint visTris = WaveActiveSum(visible ? primCount : 0u);
    uint fCullCount = WaveActiveCountBits(frustumCulled);
    uint hCullCount = WaveActiveCountBits(occluded);
    uint hCullTris = WaveActiveSum(occluded ? primCount : 0u);

    if (WaveIsFirstLane()) {
        if (visCount > 0u) {
            g_CullStats.InterlockedAdd(0, visCount);
            g_CullStats.InterlockedAdd(4, visTris);
        }
        if (fCullCount > 0u) {
            g_CullStats.InterlockedAdd(8, fCullCount);
        }
        if (hCullCount > 0u) {
            g_CullStats.InterlockedAdd(12, hCullCount);
            g_CullStats.InterlockedAdd(16, hCullTris);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(visCount, 1u, 1u, s_Payload);
}
