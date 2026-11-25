// Clean-room DXR 1.1 Inline Ray Tracing (RayQuery) verification shader (CS SM 6.6)
// Shoots 4 rays against a TLAS containing a unit box [-0.5, 0.5]^3 at origin.

RaytracingAccelerationStructure g_Tlas : register(t0);
RWStructuredBuffer<float4>      g_Out  : register(u0); // x = hit (1/0), y = committed T, z = instanceID, w = primIdx

[numthreads(4, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    RayDesc ray;
    ray.TMin = 0.01;
    ray.TMax = 100.0;

    if (idx == 0u) {
        // Ray along +Z hitting Z=-0.5 face at t = 4.5
        ray.Origin = float3(0.0, 0.0, -5.0);
        ray.Direction = float3(0.0, 0.0, 1.0);
    } else if (idx == 1u) {
        // Ray straight down -Y hitting Y=+0.5 face at t = 2.5
        ray.Origin = float3(0.1, 3.0, -0.1);
        ray.Direction = float3(0.0, -1.0, 0.0);
    } else if (idx == 2u) {
        // Miss ray pointing away from box
        ray.Origin = float3(5.0, 5.0, 5.0);
        ray.Direction = float3(0.0, 1.0, 0.0);
    } else {
        // Diagonal ray along +X hitting X=-0.5 face at t = 9.5
        ray.Origin = float3(-10.0, 0.0, 0.0);
        ray.Direction = float3(1.0, 0.0, 0.0);
    }

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(g_Tlas, RAY_FLAG_NONE, 0xFFu, ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        g_Out[idx] = float4(1.0, q.CommittedRayT(), float(q.CommittedInstanceID()), float(q.CommittedPrimitiveIndex()));
    } else {
        g_Out[idx] = float4(0.0, -1.0, -1.0, -1.0);
    }
}
