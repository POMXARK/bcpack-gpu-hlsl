# `bcpack-gpu-hlsl` — AI/LLM Machine-Readable Architecture & Optimization Reference (`v0.9.0`)

> **Target audience:** Coding agents, automated integration planners, and GPU systems engineers working on Xbox Series X|S compatibility layers (`xcl-dev/xcl`, `OpenScarlett`, `ScarlettX`, `AnyXboxX`, `rebox`) or clean-room BCPack / DirectStorage / RAGE9 pipelines (`bcpack-rs`, `rage-texture`, `bcpack-decoder`).
> **License & Provenance:** `MIT OR Apache-2.0`. 100% clean-room implementation derived from public specs (`Ahmet-Dedeler/bcpack-rs`, `microsoft/silicon-rans`) and CC0/CC-BY 4.0 open assets. Zero proprietary/copyrighted game assets or SDK binaries.

---

## 1. System Topology & Exact Module Map

| Layer | File Path | Responsibility | Key Types / Entrypoints / Bindings |
|---|---|---|---|
| **C-ABI Header** | [`include/bcpack_gpu.h`](../include/bcpack_gpu.h) | Drop-in `IDStorageQueueX` / zero-copy C++ wrapper API | `bcpack_gpu_create`, `bcpack_gpu_create_from_device`, `bcpack_gpu_destroy`, `BcPackDecodeAsync`, `BcPackDecodeToBuffer`, `BcPackGetComputeFence`, `BcPackWaitFence`, `BcPackGetLastTimingMs` |
| **FFI Implementation** | [`src/ffi.rs`](../src/ffi.rs) | Thread-safe DX12 Async Compute wrapper (`cdylib`) | `BcPackFormat::{Bc1=0, Bc7Rans=1}`, `BcPackSwizzle::{Linear=0, Gfx10_64KbD=1, Gfx10_4KbS=2}` |
| **GPU Decoder Core** | [`src/gpu_decoder.rs`](../src/gpu_decoder.rs) | DX12 Device, Compute+Direct Queues, Ping-Pong `Bc1SlotResources`, `Bc16GpuStreamer`, Gfx10 detile | `GpuContext`, `submit_slot_async_compute`, `Bc16GpuStreamer`, `pack_batch`, `uav_barrier_all` |
| **BC1 Wave32 Shader** | [`shaders/bcpack_decode.hlsl`](../shaders/bcpack_decode.hlsl) | Dual-chunk (`2×12` lanes) zero-barrier rANS + predictor + quad reconstruction | `[WaveSize(32)]`, `ModelData` (`t0`), `g_Streams` (`t3`), `g_Chunks` (`t4`), `g_OutBlocks` (`u0`) |
| **BC7 Wave32 Shader** | [`shaders/bcpack_bc7_rans.hlsl`](../shaders/bcpack_bc7_rans.hlsl) | `100%` Wave32 (`2×16 = 32/32` lanes) 16-lane rANS entropy stage | `BC7_ALPHABET_BITS[16]`, emits `8 × uint` (`16 × uint16`) symbol-rows to `g_OutSyms` (`u0`) |
| **BC7/BC5 16B Packer** | [`shaders/bc16_pack.hlsl`](../shaders/bc16_pack.hlsl) | Repacks 10 active symbol lanes from 16-lane rANS into 8B/row (`16B` block) | `[numthreads(64,1,1)]`, `g_InSyms` (`t0`) → `g_OutBlocks` (`u0`) via `Store2` |
| **Gfx10 Detile Shader** | [`shaders/gfx10_detile.hlsl`](../shaders/gfx10_detile.hlsl) | RDNA2 console tile swizzle → linear `BCn` blocks | `tile_mode`: `0=SW_64KB_D (8B)`, `1=SW_4KB_S (8B)`, `2=SW_64KB_D (16B)` |
| **Virtual Descriptor Pager** | [`src/descriptor_paging.rs`](../src/descriptor_paging.rs), [`shaders/bindless_paging.hlsl`](../shaders/bindless_paging.hlsl) | `1,250,000` virtual resource IDs paged onto `262,144` physical Tier-3 slots | `VirtualDescriptorPager`, `CopyDescriptorsSimple`, `ResourceDescriptorHeap[NonUniformResourceIndex(phys)]` |
| **World Grid Streaming** | [`src/world_streaming.rs`](../src/world_streaming.rs) | `32×32` & `64×64` (`256 km²`) sector LRU VRAM residency + OOO NVMe package reader | `LOD0 / LOD1 / HLOD` visibility rings, frustum turn burst, slab fragmentation tracker |
| **Crowd Skinning Compute** | [`src/crowd_skinning.rs`](../src/crowd_skinning.rs), [`shaders/crowd_skin.hlsl`](../shaders/crowd_skin.hlsl) | 500 NPCs × 4-bone Linear Blend Skinning in Compute | `[numthreads(64,1,1)]`, `512k` verts verify / `154k` live verts in `aaa_crowd.hlsl` |
| **Hi-Z Depth Culling** | [`src/hiz_culling.rs`](../src/hiz_culling.rs), [`shaders/hiz_build.hlsl`](../shaders/hiz_build.hlsl), [`shaders/hiz_cull.hlsl`](../shaders/hiz_cull.hlsl) | 6-mip conservative MAX depth pyramid + Wave32 sphere projection occlusion test | `AaaHiZPyramid` (`R32_TYPELESS`), `ceil(log2(max(rectW, rectH)))` mip selection |
| **DXR 1.1 Inline RT / RTGI** | [`src/raytracing.rs`](../src/raytracing.rs), [`src/rtgi.rs`](../src/rtgi.rs) | BLAS/TLAS builder (`PREFER_FAST_BUILD`, `<=1536` inst) + A-Trous bilateral denoiser | `RayQuery<RAY_FLAG_FORCE_OPAQUE>`, `rtgi_denoise.hlsl`, `rtgi_composite.hlsl` |
| **Open AAA Benchmark (`v0.8.0`)** | [`src/aaa_scene.rs`](../src/aaa_scene.rs), [`src/aaa_world.rs`](../src/aaa_world.rs), [`src/bin/demo_aaa.rs`](../src/bin/demo_aaa.rs) | 4K BC7/BC5 streaming + active GPU SFS + 12.41M Mesh-Shader City + 500 NPCs + TAA + AgX + HUD | `SfsResidencyManager`, `AaaMeshletCity`, `AaaPostPipeline`, `AaaCompCB` (12 DW), `--benchmark-run` |
| **Assets & Encoders** | [`src/assets.rs`](../src/assets.rs), [`src/gltf_loader.rs`](../src/gltf_loader.rs) | Pure-Rust BC1, BC7 Mode 6 & BC5 encoders + histogram-accelerated rANS + `BCP4` cache | `TexFormat::{Bc1, Bc5, Bc7}`, `TILE_BC16_BYTES = 65536` (`4,096` 16B blocks/tile) |

---

## 2. Unique Optimizations & Hardware Engineering Workarounds

### Optimization 1: Zero-Barrier Wave32 Multi-Lane rANS (`bcpack_decode.hlsl`, `bcpack_bc7_rans.hlsl`)
- **Problem:** Naive GPU ports of Xbox MDU interleaved rANS (12 lanes BC1, 16 lanes BC7) stall on `GroupMemoryBarrierWithGroupSync()` up to 4,096+ times per chunk (once per byte fetch and once per emitted block) and bounce compressed bytes through LDS (`groupshared`), limiting throughput and wasting half of Wave32 on BC7 (`16/32` lanes).
- **Solution:**
  1. **Dual-Chunk Warp Packing:** Each `WaveSize(32)` group decodes **2 independent chunks simultaneously** (`ll = lane < 12 ? lane : lane - 12` for BC1 `24/32` active; `ll = lane & 15u` for BC7 **`32/32 = 100%` active**).
  2. **Register-Resident 64B Stream Window + Warp Shuffle:** Lanes `0..15` (half-warp) collectively load one `uint4` (`16 × u32 = 64B`) from `ByteAddressBuffer g_Streams` into registers; byte requests (`want_byte`) are resolved across the half-warp via `WaveReadLaneAt` with **zero `groupshared` stream buffer traffic**.
  3. **Zero Barriers in Hot Loop:** Emitted lane symbols are transposed/packed via `WaveReadLaneAt` directly in registers before coalesced UAV writes — **0 `GroupMemoryBarrierWithGroupSync()` calls inside the decode loop** (and **0 in the entire `bcpack_bc7_rans.hlsl`**).
  4. **4-Step Branchless Binary Search + Fused `decode_and_advance`:** Replaces linear 10-segment rANS lookup with 4 branchless comparisons in registers and computes `next_p0` in the same helper, halving LDS reads (`lane_bf`, `lane_rfrsn`).
- **Measured Result (RTX 5080):** `11,327.6 MB/s` (`5.650 ms` / `64 MiB` BC1, **282.1×** vs 1T CPU) and `6,024.0 MB/s` equiv (`5.312 ms` / `32 MiB` BC7, **247.1×** vs 1T CPU).

### Optimization 2: DXC 1.8 `-O3` Wave-Intrinsics Divergence Workaround
- **Failure Mode:** Under `dxc.exe -T cs_6_6 -O3`, combining `WavePrefixCountBits(want)` + `WaveActiveCountBits(want)` inside the Wave32 split-half rANS Phase-0 byte-allocator loop causes DXC's scalar/vector optimizer to miscompile `stream_pos` updates, causing silent byte corruptions at `-O3` while `-Od` passes.
- **Exact Fix (DO NOT REVERT):** Replace prefix/active count intrinsics in the byte allocator with explicit warp ballot bitmasking:
  ```hlsl
  uint4 ballot = WaveActiveBallot(want_byte);
  uint mask = ballot.x & half_mask;          // isolate lanes 0..15 or 16..31
  uint prefix = countbits(mask & ((1u << lane) - 1u));
  uint advance = countbits(mask);
  ```
- **Invariant:** Produces **0/0 mismatches (17/17 PASS)** under full `dxc -O3`.

### Optimization 3: Fused AMD Gfx10 Swizzle → Linear Compute Detile (`gfx10_detile.hlsl`)
- **Problem:** Xbox Series X|S MDU outputs tiled `SW_64KB_D` / `SW_4KB_S` block layouts directly into VRAM; CPU detiling adds memory copies and stalls rendering.
- **Solution:** Dispatches `1` thread per compressed block (`[numthreads(64,1,1)]`) using exact RDNA2 bit-permutation masks (`SW_64KB_D` 8B `"xyxxyyxyxyxyx"`, `SW_4KB_S` 8B `"xyyxxyxyx"`, `SW_64KB_D` 16B `"xyxxyyxyxyxy"`), executing **inline on `D3D12_COMMAND_LIST_TYPE_COMPUTE`** before `CopyTextureRegion`.
- **Measured Result (RTX 5080):** **`0.144 ms` for `64 MiB` (`432.6 GB/s` effective)**.

### Optimization 4: Double-Buffered Ping-Pong Async Compute + 16B GPU Packer (`gpu_decoder.rs`, `bc16_pack.hlsl`)
- **Problem:** Running compute decode on `DIRECT` serializes decompression with geometry passes; furthermore, 16-lane rANS emits `16 × uint16` (`32B`) per symbol-row rather than packed `16B` BC7/BC5 hardware blocks.
- **Solution:**
  1. Two independent ping-pong buffer slots (`slot = frame & 1`) decode frame `N+1` on `D3D12_COMMAND_LIST_TYPE_COMPUTE` in parallel while frame `N` executes `CopyTextureRegion` + G-Buffer/DXR/TAA on `DIRECT`, synchronized via cross-queue `ID3D12Fence`.
  2. `shaders/bc16_pack.hlsl` compacts the 10 active lanes of each `uint[8]` symbol row into packed `uint2` (`8B/row` = `16B` BC7 Mode 6 / BC5 block) directly in GPU VRAM before `CopyTextureRegion` into `DXGI_FORMAT_BC7_UNORM` / `DXGI_FORMAT_BC5_UNORM`.
- **Measured Result:** Lifts live 3D streaming (`8 MiB/frame`) from `101.81 FPS` (Direct Queue) → `140.13 FPS` (Async Compute) → **`189.36 FPS`** (Zero-Barrier + Ping-Pong), sustaining **`171.86 FPS` at `32 MiB/frame`** (`5.5 GB/s`).

### Optimization 5: SM 6.6 Virtual `ResourceDescriptorHeap[]` Paging (`descriptor_paging.rs`, `bindless_paging.hlsl`)
- **Problem:** NVIDIA DX12 drivers impose a hard ~`1,000,000` hardware limit on `D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE` SRV/CBV/UAV heaps, causing `E_OUTOFMEMORY` in open-world translation layers (`xcl`, `OpenScarlett`) that allocate >1M bindless texture descriptors.
- **Solution:** `VirtualDescriptorPager` maintains `1,250,000` CPU-side virtual SRV handles in non-shader-visible staging heaps, tracks LRU residency across `262,144` physical slots in the GPU shader-visible heap, and streams evicted slots via `CopyDescriptorsSimple` + a virtual-to-physical indirection `StructuredBuffer<uint> g_VirtToPhys` sampled via `ResourceDescriptorHeap[NonUniformResourceIndex(phys)]`.
- **Measured Result (PASS 14/17):** `300,000` virtual descriptor accesses (`37,856` evictions), **`0.061 ms` per paging batch**, **`65,536` pixel readbacks with `0` mismatches**.

### Optimization 6: Pixel-Shader Sampler Feedback Streaming (`g_SfsFeedback`) + `ResourceMinLODClamp` LRU
- **Problem:** Preloading full 4K (`4096×4096`) BC7/BC5 11-mip chains for `CarConcept.glb` + 9 Poly Haven PBR maps requires `204.67 MiB` VRAM, causing PCIe/decode pressure on constrained budgets.
- **Solution:** Every lit surface in `shaders/aaa_pbr.hlsl` and `shaders/aaa_meshlet_ps.hlsl` computes analytical mip level (`0.5 * log2(...)`) from UV derivatives, atomically updates `RWByteAddressBuffer g_SfsFeedback : register(u0)` via `InterlockedMin`, reads back asynchronously on double-buffered readback heaps, and `SfsResidencyManager` prioritizes missing Mip-0 `64 KiB` tiles (`256×256` blocks for BC7/BC5) while updating SRV `ResourceMinLODClamp` under LRU `--vram-budget-mb 128`. `[V]` X-Ray visualizes live residency (Green/Cyan = Mip-0 resident, Amber = SFS clamped/saved Mip 1+, Blue = streaming clay).
- **Measured Result (`v0.8.0` 60s Benchmark):** **`86.2 / 204.7 MiB` avg VRAM used (`118.5 MiB` saved, 57.9% reduction)**.

### Optimization 7: 12.41M-Tri Modular Mesh Shaders (`as_6_5`/`ms_6_5`) + 6-Mip Conservative Hi-Z (`aaa_world.rs`, `aaa_meshlet_as.hlsl`)
- **Problem:** Box primitives break visual realism while naive 12M-tri rasterization bottlenecks vertex throughput.
- **Solution:** Procedural modular city (`851` instances, `219,774` meshlets, **`12.41M` pre-cull tris**: recessed reflective glass skyscrapers, balconies, Art-Deco hotels, palms, streetlamps, road markings, seawall balustrade, skyline HLOD) dispatched in **one `DispatchMesh`**. `hiz_build.hlsl` builds a 6-mip conservative MAX depth pyramid (`R32_TYPELESS`) each frame (`0.010 ms`); next frame's `ASMain` projects the 8 bounding-box corners of each meshlet's bounding sphere using `g_PrevVPUnjittered`, computes `mip = ceil(log2(max(rectW, rectH)))`, and culls against 4 Hi-Z texels (`sphereMinZ > maxZ + 0.0020`) + 6 frustum planes before Wave32 prefix compaction (`WaveActiveCountBits` / `WavePrefixCountBits`) feeds `DispatchMesh(visCount, 1, 1)`.
- **Measured Result (`v0.8.0` 60s Benchmark):** **`12.41M` pre-cull → `0.99M` avg visible tris (`0.18M` Hi-Z occluded/frame)** in one draw call.

### Optimization 8: NVIDIA RTX 5080 Driver `621.24` RT-Core Tile-Cache Flicker Fix (`aaa_scene.rs`)
- **Failure Mode:** Under NVIDIA Blackwell driver `621.24`, submitting `BuildRaytracingAccelerationStructure` (TLAS) on a command list interleaved with Async Compute BCPack `CopyTextureRegion` without a full compute+UAV barrier flush produces transient **8×8 black tile blocks** during `RayQuery` reflections/shadows.
- **Exact Fix (DO NOT REMOVE):** Record TLAS rebuilds on a dedicated synchronization sequence using `uav_barrier_all()` + explicit fence flush (`ctx.signal_and_wait()` in `rt.build_tlas` / `tlas_cmd` isolation) prior to starting the 4-MRT G-Buffer pass. Eliminates RT-tile corruption completely across 9,000+ frames.

### Optimization 9: Textured DXR 1.1 `RayQuery`, 4-MRT TAA + CAS, Histogram Auto-Exposure & AgX (`aaa_pbr.hlsl`, `aaa_taa.hlsl`, `aaa_autoexp.hlsl`, `aaa_composite.hlsl`)
- **Textured Inline DXR Hits:** Uses `RtInstMeta` (`t34`), `g_SceneVB` (`t35`), `g_SceneIB` (`t36`) in `RayQuery` (`CandidateInstanceID`, `CommittedPrimitiveIndex`, `CommittedTriangleBarycentrics`) so car-paint and wet-asphalt reflections sample actual 4K BC7 albedo/emissive maps of hit buildings and traffic instead of flat instance colors.
- **4-MRT G-Buffer:** `directHDR` + `rawRTGI` + `normalDepth` (`R16G16B16A16_FLOAT`) + `motionVec` (`R16G16_FLOAT`) with Halton(2,3) subpixel jitter (`8`-phase) and per-object `g_PrevModel`/`g_PrevVPUnjittered`.
- **Post-Process Chain:** 4-pass A-Trous edge-aware GI denoiser (`step = 1, 2, 4, 8`) → 8-step Screen-Space Contact Shadows + YCoCg Variance-Clipped TAA (`aaa_taa.hlsl`) → 64-bin Log-Average Histogram Auto-Exposure (`aaa_autoexp.hlsl`) → Half-res 3-dispatch HDR Bloom (`threshold=1.45`) → AgX Filmic curve + 5-tap CAS (`aaa_composite.hlsl`).

### Optimization 10: 500 GPU-Skinned NPC Crowd Compute (`crowd_skinning.rs`, `crowd_skin.hlsl`, `aaa_crowd.hlsl`)
- **Design:** Computes 4-bone Linear Blend Skinning (`crowd_skin.dxil`) across `500` walking promenade NPCs (`154,000` live skinned verts/frame in `aaa_scene.rs`, `512,000` verts / `32,000` bones verified in `PASS 17/17` in **`0.075 ms`**, `max_err = 7.63e-6`) with previous-frame transforms for ghost-free per-limb TAA motion vectors in a single instanced draw.

### Optimization 11: Zero-Dependency HLSL Interactive HUD & Hybrid Frame Pacer (`aaa_composite.hlsl`, `aaa_scene.rs`)
- **Architecture:** `AaaCompCB` (12 × 32-bit root constants at `b0`) passes HUD toggle (`[H]`), FPS Cap (`15..300 / VSync / MAX`), live FPS, Simulation Speed (`0.00x..4.00x`), and active drag highlight directly into `aaa_composite.hlsl`.
- **Rendering:** Procedural rounded frosted-glass SDF panel + 3×5 15-bit bitmap font (`GLYPH5`) rendered in the final composite pass at zero extra draw calls/textures/ImGui dependencies.
- **Input & Pacing:** Win32 `WM_LBUTTONDOWN` / `WM_MOUSEMOVE` (`SetCapture`) / `WM_LBUTTONUP` + hotkeys (`[H]`, `[` / `]`, `-` / `=`, `[P]`, `[0]`) update `wstate.fps_cap` (enforced via hybrid `sleep(1ms)` + `spin_loop()`) and `wstate.sim_speed` (scales `sim_time` for camera tour, traffic, 500 NPCs, and shaders while keeping FreeCam WASD on wall `dt`).

---

## 3. Drop-In C-ABI Contract (`include/bcpack_gpu.h`)

```c
// Option A: Shared Caller ID3D12Device Zero-Copy Mode (zero CPU readback, GPU-only queue Wait):
BcPackContext* ctx = bcpack_gpu_create_from_device(d3d12_device, NULL /*or compute_queue*/, 2048, 8388608, 96u * 1024u * 1024u);
uint64_t fence = 0;
BcPackDecodeToBuffer(ctx, compressed_ptr, compressed_size, num_blocks,
                     BCPACK_FORMAT_BC1, BCPACK_SWIZZLE_GFX10_64KB_D,
                     width_blocks, height_blocks, caller_gpu_buf, 0, &fence);
ID3D12Fence* cfence = (ID3D12Fence*)BcPackGetComputeFence(ctx);
direct_queue->Wait(cfence, fence); // Caller consumes caller_gpu_buf directly on GPU

// Option B: Standalone Context with Host Readback or Async Fence:
BcPackContext* standalone = bcpack_gpu_create(2048, 8388608, 96u * 1024u * 1024u);
BcPackDecodeAsync(standalone, compressed_ptr, compressed_size, num_blocks,
                  BCPACK_FORMAT_BC1, BCPACK_SWIZZLE_GFX10_64KB_D,
                  width_blocks, height_blocks, out_linear_bc1, out_cap, &fence);
```

- **Resource state contract (`BcPackDecodeToBuffer`):** `dst_d3d12_resource` MUST live on `d3d12_device` and enter in `D3D12_RESOURCE_STATE_COMMON` (restored to `D3D12_RESOURCE_STATE_COMMON` before signaling `fence`).
- **Thread safety:** Wait before reusing allocator slots; `bcpack_gpu_destroy` drains `compute_fence_value` before teardown.
- **BC7 output layout:** `BCPACK_FORMAT_BC7_RANS` emits `2 * num_blocks` rows of `uint16[16]` (`64B` per BC7 block). Chain `shaders/bc16_pack.hlsl` to collapse to final `16B` BC7/BC5 blocks.

---

## 4. Verification Oracle & Benchmark Numbers (NVIDIA RTX 5080, Driver 621.24)

- **`bcpack-gpu-demo.exe --verify`:** **18/18 bit-exact PASS** (`0` byte/symbol mismatches, DX12 Debug Layer clean):
  - Tests `1..7`: BC1 synthetic, monolithic `512×512`, `4K`-block tiles, BC7 16-lane rANS (Normal/UI/Albedo), Gfx10 `SW_64KB_D` + `SW_4KB_S` detile, Full 10-level Mipmap + Packed Mip Tail, and C-ABI FFI smoke test.
  - Tests `8..13`: DXR 1.1 analytical BVH rays, `32×32` LRU world streaming, `4.01M` Mesh Shaders + culling, `1,536`-TLAS + RTGI denoiser, 8B quantized meshlet geometry, `CarConcept.glb` + Poly Haven PBR.
  - Tests `14..18`: **`1.25M` Virtual Descriptor Paging** (`0.062 ms`), **`64×64` Supersonic Jet-Flight Streaming** (`960 MB/s`), **`12.04M` Two-Pass Hi-Z Culling** (`0.015 ms`), **`500`-NPC GPU LBS Skinning** (`0.075 ms`), and **`Zero-Copy Shared-Device C-ABI`** (`bcpack_gpu_create_from_device` + `BcPackDecodeToBuffer` + GPU `Queue::Wait`, `0` mismatches).
- **`bcpack-aaa-demo.exe --benchmark-run --benchmark-secs 60` (`1920×1080` HDR):**
  - **`150.61 FPS` Avg** (`6.64 ms`, `9,026` frames) | **`87.07 FPS` 1% Low** | **`43.37 FPS` 0.1% Low**
  - **Active SFS VRAM:** **`86.2 / 204.7 MiB`** (`118.5 MiB` saved)
  - **Mesh Shaders + Hi-Z:** **`12.41M` pre-cull → `0.99M` avg visible** (`0.18M` Hi-Z occluded)
  - **Live GPU BCPack Decode Peak:** `0.85 GB/s` real-time 4K BC7/BC5 tile streaming (`--flush-bench`: `217.0 ms` GPU vs `5,487.4 ms` 1T CPU = **25.3×**)
  - **DX12 Debug Layer (`--debug`):** **0 errors**.

---

## 5. Critical Invariants for AI Agents Modifying This Codebase

1. **DXC `-O3` Ballot Workaround:** Never replace `WaveActiveBallot` + `countbits(mask & ((1u << ll) - 1u))` in `bcpack_decode.hlsl` / `bcpack_bc7_rans.hlsl` with `WavePrefixCountBits` + `WaveActiveCountBits`.
2. **Root Constant Counts Must Match HLSL `cbuffer`s:** Modifying `AaaCompCB` in `shaders/aaa_composite.hlsl` (`12` DW) or `AaaDrawCB` requires updating `Num32BitValues` in `src/aaa_scene.rs` (`comp_root_sig`, `pbr_root_sig`) AND recompiling `.dxil` via `dxc.exe` (`shaders/compile_shaders.bat`). Precompiled `.dxil` bytes are `include_bytes!`-embedded at Rust compile time.
3. **RT-Core Cache Barrier (`621.24`):** Never remove `uav_barrier_all()` or TLAS isolation before G-Buffer passes when mixing Async Compute texture copies with DXR 1.1 `RayQuery`.
4. **Descriptor Heap Layout (`main_srv_heap`):** Currently sized at **`160`** shader-visible descriptors in `aaa_scene.rs` (`0..73` = 2×37 PBR/TLAS/scene tables, `74..106` = RTGI/TAA/AutoExp/Bloom/Comp, `107..119` = Hi-Z mips & UAVs). Expanding passes requires increasing the heap size check.
5. **Single-GPU-Context Rule on Test Hosts:** Never launch two DX12 swapchain binaries concurrently; `--screenshot` and `--record-dir` auto-hide the HUD (`--show-hud` overrides).
