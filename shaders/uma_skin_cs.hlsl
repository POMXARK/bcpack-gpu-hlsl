// Clean-room UMA Emulation: Compute Skinning Pass for Multi-Threaded Hazard Journal Test
// Writes 16-byte skinned vertex: float3 pos (with frame wave) + uint tag = ((mesh_id << 16) | (frame_tag & 0xFFFF)).

cbuffer UmaSkinCB : register(b0) {
    uint g_MeshId;
    uint g_FrameTag;
    uint g_VertsPerMesh;
    uint g_Pad0;
};

ByteAddressBuffer   g_BaseVerts    : register(t0);
RWByteAddressBuffer g_SkinnedVerts : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint v = dtid.x;
    if (v >= g_VertsPerMesh) {
        return;
    }

    uint global_v = g_MeshId * g_VertsPerMesh + v;
    uint byte_off = global_v * 16u;
    uint4 raw = g_BaseVerts.Load4(byte_off);
    float3 p = asfloat(raw.xyz);

    float phase = float(g_FrameTag & 255u) * 0.05 + float(g_MeshId) * 0.17;
    p.y += sin(phase) * 0.005;

    uint tag = ((g_MeshId & 0xFFFFu) << 16u) | (g_FrameTag & 0xFFFFu);
    g_SkinnedVerts.Store4(byte_off, uint4(asuint(p), tag));
}
