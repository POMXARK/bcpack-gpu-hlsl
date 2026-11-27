// Clean-room UMA Emulation: Working Set Bandwidth Benchmark (PCIe Host UPLOAD vs GDDR7 DEFAULT VRAM)
// Performs strided uint4 reads across a 16-64 MiB working set so PCIe vs GDDR7 latency/BW delta is directly measurable.

cbuffer UmaWorkCB : register(b0) {
    uint g_ElemCount; // uint4 count in working set buffer
    uint g_Passes;    // strided passes per thread
    uint g_Salt;
    uint g_Pad0;
};

StructuredBuffer<uint4> g_WorkingSet : register(t0);
RWStructuredBuffer<uint> g_Sink      : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint tid = dtid.x;
    uint total_threads = 65536u;
    if (tid >= total_threads || g_ElemCount == 0u) {
        return;
    }

    uint4 acc = uint4(tid, g_Salt, 0u, 0u);
    for (uint p = 0u; p < g_Passes; ++p) {
        uint idx = (tid + p * 65536u) % g_ElemCount;
        acc ^= g_WorkingSet[idx];
    }

    g_Sink[tid & 1023u] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
