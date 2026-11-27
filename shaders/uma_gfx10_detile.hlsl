// Clean-room UMA Emulation: Batch Gfx10 SW_64KB_D (32bpp) GPU Detile Compute Shader
// Detiles N dirty textures in one Dispatch from packed SW_64KB_D tiled source into 256B-pitch linear staging buffer.

cbuffer UmaDetileCB : register(b0) {
    uint g_Width;
    uint g_Height;
    uint g_BatchCount;
    uint g_TexStrideBytes;
};

ByteAddressBuffer   g_SrcTiled  : register(t0);
RWByteAddressBuffer g_DstLinear : register(u0);

uint swizzle_64kb_d_32bpp(uint lx, uint ly) {
    // 128 x 128 pixels * 4 bytes = 65536 bytes (1 x 64KB_D macro-tile)
    return ((lx & 1u) << 0u)
         | ((ly & 1u) << 1u)
         | (((lx >> 1u) & 3u) << 2u)
         | (((ly >> 1u) & 3u) << 4u)
         | (((lx >> 3u) & 1u) << 6u)
         | (((ly >> 3u) & 1u) << 7u)
         | (((lx >> 4u) & 1u) << 8u)
         | (((ly >> 4u) & 1u) << 9u)
         | (((lx >> 5u) & 1u) << 10u)
         | (((ly >> 5u) & 1u) << 11u)
         | (((lx >> 6u) & 1u) << 12u)
         | (((ly >> 6u) & 1u) << 13u);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint x = dtid.x;
    uint y = dtid.y;
    uint b = dtid.z;
    if (x >= g_Width || y >= g_Height || b >= g_BatchCount) {
        return;
    }

    uint tiles_x = max(g_Width >> 7u, 1u);
    uint tile_idx = (y >> 7u) * tiles_x + (x >> 7u);
    uint in_tile_px = swizzle_64kb_d_32bpp(x & 127u, y & 127u);

    uint src_px_byte = b * g_TexStrideBytes + tile_idx * 65536u + in_tile_px * 4u;
    uint dst_px_byte = b * g_TexStrideBytes + (y * g_Width + x) * 4u;

    uint pixel = g_SrcTiled.Load(src_px_byte);
    g_DstLinear.Store(dst_px_byte, pixel);
}
