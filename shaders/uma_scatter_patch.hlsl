// Clean-room UMA Emulation: Coalesced GPU Scatter-Patch Compute Shader
// Replaces thousands of small CPU->GPU descriptor/buffer CopyBufferRegion calls with a single Dispatch.

cbuffer UmaScatterCB : register(b0) {
    uint g_PatchCount;
    uint g_Pad0;
    uint g_Pad1;
    uint g_Pad2;
};

StructuredBuffer<uint4> g_SrcData        : register(t0); // 4 x uint4 (64 bytes) per patch
StructuredBuffer<uint>  g_DstSlotIndices : register(t1); // destination 64B slot index per patch
RWByteAddressBuffer     g_DstHeap        : register(u0); // 64-byte aligned virtual descriptor / shared heap

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= g_PatchCount) {
        return;
    }
    uint dst_slot = g_DstSlotIndices[idx];
    uint src_base = idx * 4u;
    uint dst_byte = dst_slot * 64u;
    [unroll]
    for (uint q = 0u; q < 4u; ++q) {
        g_DstHeap.Store4(dst_byte + q * 16u, g_SrcData[src_base + q]);
    }
}
