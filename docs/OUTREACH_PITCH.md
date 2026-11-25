# Community Outreach Templates (`xcl-dev/xcl` / `AnyXboxX`)

Ready-to-post English GitHub Issue / Discussion proposals for Xbox Series compatibility layers on Windows PC.

---

## Option A: Technical Proposal for `xcl-dev/xcl` & `LooperSalty/AnyXboxX`

**Title:** `[RFC] Zero-CPU GPU BCPack Decompressor + Gfx10 Detile (DX12 HLSL SM 6.6 / C-ABI bcpack_gpu.dll — 11.33 GB/s, MIT/Apache-2.0)`

```markdown
Hi team,

Open-world texture streaming on Xbox Series X|S builds (`IDStorageQueueX` / BCPack `.ttpf` pools) often stalls Windows compatibility layers at **1–5 FPS** when CPU decoders (`~40 MB/s/core`) block rendering threads and push unpacked `BCn` blocks over PCIe every frame.

I built a **100% clean-room, bit-exact GPU BCPack decompressor** in **DX12 HLSL Shader Model 6.6 (Wave32)** + Rust (`cdylib`) to eliminate CPU decompression and unpacked PCIe transfers:

- **Repository:** https://github.com/POMXARK/bcpack-gpu-hlsl (MIT OR Apache-2.0, zero copyrighted game assets)
- **Prebuilt binaries (`v0.6.0` Release):** `bcpack_gpu.dll`, `bcpack_gpu.dll.lib`, `include/bcpack_gpu.h`, and 14 precompiled `.dxil` shaders (`bcpack-gpu-dxil-v0.6.0.zip`).
- **New in `v0.6.0` Stress Suite (verified `12/12 PASS` on RTX 5080):** `32×32` World Grid LRU Residency + `.bcpack_pkg` NVMe streaming (`155.35 FPS`), Hardware Mesh Shaders (`4.01M` tris / `52,949` meshlets), `1,536`-instance DXR 1.1 TLAS + 1-spp Diffuse RTGI + 2-Pass A-Trous Denoiser (`152.99 FPS` @ 720p HDR / `128.62 FPS` @ 2560×1440 HDR), and BCPack 8B quantized geometry (`6,241 MB/s`).

### Verified Benchmarks on NVIDIA RTX 5080 (vs `Ahmet-Dedeler/bcpack-rs`, `0` byte/symbol mismatches)
| Workload | CPU 1-Core | CPU Multi-Core | `bcpack_gpu` Zero-Barrier GPU | Speedup |
|---|---:|---:|---:|---:|
| **BC1 4K-Tiled Strips** (`64 MiB`) | 1576.1 ms (`40.6 MB/s`) | 427.9 ms (`149.6 MB/s`) | **5.65 ms (`11,327.6 MB/s`)** | **279.0× vs 1T / 75.7× vs MT** |
| **BC7 16-Lane rANS (100% Wave32 = 32/32)** (`32 MiB eq`) | 1312.7 ms (`24.4 MB/s`) | 366.0 ms (`87.4 MB/s`) | **5.31 ms (`6,024.0 MB/s`)** | **247.1× vs 1T / 68.9× vs MT** |
| **AMD Gfx10 Swizzle → Linear (`SW_64KB_D`)** (`64 MiB`) | — | — | **0.144 ms (`432.6 GB/s`)** | Fused on Async Compute |
| **Live 1280×720 3D Streaming (`8 MiB/frame`)** | 5.26 FPS (`189.0 ms`) | 31.14 FPS (`29.7 ms`) | **189.36 FPS (`4.81 ms`)** (or **171.45 FPS** w/ DXR 1.1) | **36.0× live FPS** |

*(Note: BC7 covers the verified 16-lane rANS entropy stage with `0` symbol mismatches against `bcpack::rans::decode_slice`, matching the completed portion of upstream `Ahmet-Dedeler/bcpack-rs` `bc5-bc7`.)*

### Drop-in C-ABI Hook (`include/bcpack_gpu.h`)
You can wire `BcPackDecodeAsync` directly into an `IDStorageQueueX` custom decompression path without compiling Rust:

```c
#include "bcpack_gpu.h"

BcPackContext* ctx = bcpack_gpu_create(2048, 8388608, 96 * 1024 * 1024);
uint64_t fence = 0;
BcPackDecodeAsync(ctx, stream_ptr, stream_len, num_blocks,
                  BCPACK_FORMAT_BC1, BCPACK_SWIZZLE_GFX10_64KB_D,
                  width_blocks, height_blocks, out_linear_bc1, out_cap, &fence);
BcPackWaitFence(ctx, fence, UINT32_MAX);
```

Would there be interest in testing `bcpack_gpu.dll` or porting the HLSL compute shaders (`shaders/bcpack_decode.hlsl`, `shaders/bcpack_bc7_rans.hlsl`, `shaders/gfx10_detile.hlsl`) directly into your D3D12 backend? Happy to open a PR or adapt the descriptor/root-signature layout to match your engine!
```

---

## Option B: Short Follow-Up Note (for Discussions / Discord / PortWatch)

```markdown
Clean-room DX12 HLSL SM 6.6 GPU decompressor for Xbox Series BCPack (BC1 + 16-lane BC7 rANS @ 100% Wave32 + AMD Gfx10 `SW_64KB_D`/`SW_4KB_S` detile) with a prebuilt C-ABI `bcpack_gpu.dll` ready for `IDStorageQueueX` hooks: https://github.com/POMXARK/bcpack-gpu-hlsl
Measured on RTX 5080 (0 mismatches vs `bcpack-rs`): 11.33 GB/s BC1, 6.02 GB/s BC7 rANS, 432.6 GB/s Gfx10 detile, 189 FPS live 3D texture streaming (171 FPS w/ DXR 1.1) vs 5 FPS on 1 CPU core.
```
