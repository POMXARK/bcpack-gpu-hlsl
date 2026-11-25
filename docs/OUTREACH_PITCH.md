# Community Outreach Templates (`xcl-dev/xcl` / `AnyXboxX`)

Ready-to-post English GitHub Issue / Discussion proposals for Xbox Series compatibility layers on Windows PC.

---

## Option A: Technical Proposal for `xcl-dev/xcl` & `LooperSalty/AnyXboxX`

**Title:** `[RFC] Zero-CPU GPU BCPack Decompressor + Gfx10 Detile (DX12 HLSL SM 6.6 / C-ABI bcpack_gpu.dll — 9.92 GB/s, MIT/Apache-2.0)`

```markdown
Hi team,

Open-world texture streaming on Xbox Series X|S builds (`IDStorageQueueX` / BCPack `.ttpf` pools) often stalls Windows compatibility layers at **1–5 FPS** when CPU decoders (`~40 MB/s/core`) block rendering threads and push unpacked `BCn` blocks over PCIe every frame.

I built a **100% clean-room, bit-exact GPU BCPack decompressor** in **DX12 HLSL Shader Model 6.6 (Wave32)** + Rust (`cdylib`) to eliminate CPU decompression and unpacked PCIe transfers:

- **Repository:** https://github.com/POMXARK/bcpack-gpu-hlsl (MIT OR Apache-2.0, zero copyrighted game assets)
- **Prebuilt binaries (`v0.3.0` Release):** `bcpack_gpu.dll`, `bcpack_gpu.dll.lib`, `include/bcpack_gpu.h`, and precompiled `.dxil` compute shaders.

### Verified Benchmarks on NVIDIA RTX 5080 (vs `Ahmet-Dedeler/bcpack-rs`, `0` byte/symbol mismatches)
| Workload | CPU 1-Core | CPU Multi-Core | `bcpack_gpu` Async Compute | Speedup |
|---|---:|---:|---:|---:|
| **BC1 4K-Tiled Strips** (`64 MiB`) | 1630.3 ms (`39.3 MB/s`) | 355.4 ms (`180.1 MB/s`) | **6.45 ms (`9,920.1 MB/s`)** | **252.7× vs 1T / 55.1× vs MT** |
| **BC7 16-Lane rANS (100% Wave32 = 32/32)** (`32 MiB eq`) | 1290.2 ms (`24.8 MB/s`) | 315.8 ms (`101.3 MB/s`) | **6.73 ms (`4,755.7 MB/s`)** | **191.7× vs 1T / 46.9× vs MT** |
| **AMD Gfx10 Swizzle → Linear (`SW_64KB_D`)** (`64 MiB`) | — | — | **0.340 ms (`184.1 GB/s`)** | Fused on Async Compute |
| **Live 1280×720 3D Streaming (`8 MiB/frame`)** | 5.26 FPS (`189.0 ms`) | 31.14 FPS (`29.7 ms`) | **140.13 FPS (`5.31 ms`)** | **26.6× live FPS** |

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
Measured on RTX 5080 (0 mismatches vs `bcpack-rs`): 9.92 GB/s BC1, 4.76 GB/s BC7 rANS, 184.1 GB/s Gfx10 detile, 140+ FPS live 3D texture streaming vs 5 FPS on 1 CPU core.
```
