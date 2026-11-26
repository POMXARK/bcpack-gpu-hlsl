use crate::ffi::{
    bcpack_gpu_create_from_device, bcpack_gpu_destroy, BcPackBatchItem, BcPackDecodeAsync,
    BcPackDecodeBatch, BcPackDecodeBatchToTextures, BcPackDecodeToTexture2D,
    BcPackGetLastTimingMs, BcPackTextureBatchItem, BcPackWaitDirectFence, BCPACK_FORMAT_BC1,
    BCPACK_SWIZZLE_LINEAR,
};
use crate::gpu_decoder::{
    buffer_desc, cmd_barriers, create_committed_buffer, heap_props, transition_barrier,
    uav_barrier, uav_barrier_all, upload_slice, GpuContext, BC16_PACK_CS_DXIL,
};
use bcpack::encode::synthetic_bc1;
use bcpack_gpu::descriptor_paging::{VirtualDescriptorPager, PHYS_HEAP_SLOTS, VIRTUAL_DESC_COUNT};
use std::ffi::c_void;
use std::fs::File;
use std::io::Write;
use std::mem::{size_of, ManuallyDrop};
use std::path::Path;
use std::time::Instant;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
};

#[inline]
fn check_device_ok(device: &ID3D12Device, label: &str) {
    if let Err(e) = unsafe { device.GetDeviceRemovedReason() } {
        panic!("[bottleneck] Device removed after {label}: {e:?}");
    }
}

fn create_bc1_tex2d(device: &ID3D12Device, width: u32, height: u32) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_BC1_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    unsafe {
        device
            .CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut res,
            )
            .expect("CreateCommittedResource BC1 Texture2D failed");
    }
    res.unwrap()
}

fn supports_dxr_1_1(device: &ID3D12Device) -> bool {
    unsafe {
        let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        if device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS5,
                &mut opts5 as *mut _ as *mut c_void,
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
            )
            .is_err()
        {
            return false;
        }
        opts5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0
    }
}

struct Bench1Report {
    virtual_catalog_gib: f64,
    budget_mib: usize,
    lru_evictions: u64,
    lru_thrash_reloads: u64,
    pareto_evictions: u64,
    pareto_thrash_reloads: u64,
    async_loop_wall_ms: f64,
    batch_wall_ms: f64,
    batch_kernel_ms: f64,
    zerocopy_tex2d_wall_ms: f64,
    zerocopy_tex2d_mbs: f64,
    zerocopy_batch_tex2d_wall_ms: f64,
    zerocopy_batch_tex2d_mbs: f64,
    batch_speedup: f64,
    zerocopy_speedup: f64,
    zerocopy_batch_speedup: f64,
}

fn run_bench1_residency_and_bcpack(ctx: &mut GpuContext) -> Bench1Report {
    // 1A. Simulate 4.0 GiB virtual ttdb tile catalog (65,536 x 64 KiB tiles across 6 categories)
    // under a tight 256 MiB (4,096 slots) VRAM budget over 120 frames with 180-deg camera turns.
    const TOTAL_TILES: usize = 65_536;
    const CAP_SLOTS: usize = 4_096; // 256 MiB / 64 KiB
    const FRAMES: u64 = 120;

    // Category per tile:
    // 0: terrain (35%, weight 5), 1: maps (20%, weight 4), 2: props/octa_imposters (15%, weight 4),
    // 3: interiors (15%, weight 2), 4: peds (10%, weight 1), 5: vehicles (5%, weight 1)
    let tile_weight = |id: usize| -> u64 {
        match id % 20 {
            0..=6 => 5,
            7..=10 => 4,
            11..=13 => 4,
            14..=16 => 2,
            17..=18 => 1,
            _ => 1,
        }
    };

    let simulate_policy = |use_pareto: bool| -> (u64, u64) {
        let mut last_used = vec![u64::MAX; TOTAL_TILES];
        let mut evicted_at = vec![0u64; TOTAL_TILES];
        let mut resident: Vec<usize> = Vec::with_capacity(CAP_SLOTS);
        let mut evictions = 0u64;
        let mut thrash = 0u64;

        for f in 1..=FRAMES {
            // 180-deg turn bursts every 30 frames shift transient peds/vehicles while persistent terrain/maps stay needed
            let base = ((f as usize / 30) * 1800) % 24_000;
            // Per frame: ~240 requests: 160 core terrain/maps/octa + 80 transient peds/vehicles/interiors
            for k in 0..240usize {
                let id = if k < 160 {
                    // Persistent open-world working set (~3,200 tiles of terrain/maps/octa)
                    let idx = (k * 29 + (f as usize * 17)) % 3_200;
                    (idx * 20 / 14).min(TOTAL_TILES - 1)
                } else {
                    // Transient high-churn ped/vehicle/interior stream across 20k window
                    (base + (k * 97 + f as usize * 131)) % TOTAL_TILES
                };

                if last_used[id] != u64::MAX {
                    last_used[id] = f;
                    continue;
                }
                if evicted_at[id] > 0 && f.saturating_sub(evicted_at[id]) <= 30 {
                    thrash += 1;
                }
                if resident.len() >= CAP_SLOTS {
                    // Find victim by scanning a fixed 64-entry clock window (O(1) per eviction)
                    let mut best_pos = 0usize;
                    let mut best_score = u64::MAX;
                    let start = (f as usize * 53 + k * 7) % resident.len();
                    for s in 0..64usize {
                        let pos = (start + s * 19) % resident.len();
                        let cand = resident[pos];
                        let age = f.saturating_sub(last_used[cand]).max(1);
                        // Smaller score => evict sooner
                        let score = if use_pareto {
                            tile_weight(cand) * 16 + (32u64.saturating_sub(age.min(32)))
                        } else {
                            u64::MAX / 2 - age
                        };
                        if score < best_score {
                            best_score = score;
                            best_pos = pos;
                        }
                    }
                    let victim = resident.swap_remove(best_pos);
                    last_used[victim] = u64::MAX;
                    evicted_at[victim] = f;
                    evictions += 1;
                }
                last_used[id] = f;
                resident.push(id);
            }
        }
        (evictions, thrash)
    };

    let (lru_evictions, lru_thrash_reloads) = simulate_policy(false);
    let (pareto_evictions, pareto_thrash_reloads) = simulate_policy(true);

    // 1B. Live DX12 Async Compute BCPack Phase 1 vs Phase 2 (BcPackDecodeBatch) vs Phase 3 (Zero-Copy ToTexture2D)
    // 128 x 512x512 BC1 chunks = 16.0 MiB uncompressed output
    const CHUNKS: usize = 128;
    const BLOCKS_PER_CHUNK: usize = 16_384; // 128 x 128 BC1 blocks = 128 KiB
    const CHUNK_OUT_BYTES: usize = BLOCKS_PER_CHUNK * 8;

    let mut encoded_pool = Vec::with_capacity(16);
    for seed in 0..16u64 {
        let (stream, _) = synthetic_bc1(9000 + seed * 37, BLOCKS_PER_CHUNK, 512);
        encoded_pool.push(stream);
    }

    let ffi_ctx = unsafe {
        bcpack_gpu_create_from_device(
            ctx.device.as_raw(),
            ctx.compute_queue.as_raw(),
            256,
            (CHUNKS * BLOCKS_PER_CHUNK + 65536) as u32,
            32 * 1024 * 1024,
        )
    };
    assert!(!ffi_ctx.is_null(), "bcpack_gpu_create_from_device returned null");

    let mut host_out = vec![0u8; CHUNK_OUT_BYTES];
    let mut host_batch_bufs: Vec<Vec<u8>> = (0..CHUNKS).map(|_| vec![0u8; CHUNK_OUT_BYTES]).collect();
    let tex_pool: Vec<ID3D12Resource> = (0..8).map(|_| create_bc1_tex2d(&ctx.device, 512, 512)).collect();

    // Warmup 1 chunk
    unsafe {
        let mut fence = 0u64;
        let rc = BcPackDecodeAsync(
            ffi_ctx,
            encoded_pool[0].as_ptr(),
            encoded_pool[0].len() as u32,
            BLOCKS_PER_CHUNK as u32,
            BCPACK_FORMAT_BC1,
            BCPACK_SWIZZLE_LINEAR,
            128,
            128,
            host_out.as_mut_ptr(),
            host_out.len() as u32,
            &mut fence,
        );
        assert_eq!(rc, 0, "Warmup BcPackDecodeAsync failed: {rc}");
    }

    // Phase 1: BcPackDecodeAsync per-chunk loop (readback each chunk)
    let t_async = Instant::now();
    for i in 0..CHUNKS {
        let s = &encoded_pool[i % encoded_pool.len()];
        let mut fence = 0u64;
        let rc = unsafe {
            BcPackDecodeAsync(
                ffi_ctx,
                s.as_ptr(),
                s.len() as u32,
                BLOCKS_PER_CHUNK as u32,
                BCPACK_FORMAT_BC1,
                BCPACK_SWIZZLE_LINEAR,
                128,
                128,
                host_out.as_mut_ptr(),
                host_out.len() as u32,
                &mut fence,
            )
        };
        assert_eq!(rc, 0, "BcPackDecodeAsync loop failed at {i}: {rc}");
    }
    let async_loop_wall_ms = t_async.elapsed().as_secs_f64() * 1000.0;
    let mut last_kms = 0.0f64;
    unsafe {
        let _ = BcPackGetLastTimingMs(ffi_ctx, &mut last_kms);
    }

    // Phase 2: BcPackDecodeBatch coalesced single submission
    let mut batch_items: Vec<BcPackBatchItem> = (0..CHUNKS)
        .map(|i| {
            let s = &encoded_pool[i % encoded_pool.len()];
            BcPackBatchItem {
                compressed_data: s.as_ptr(),
                compressed_size: s.len() as u32,
                num_blocks: BLOCKS_PER_CHUNK as u32,
                format: BCPACK_FORMAT_BC1,
                swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                width_blocks: 128,
                height_blocks: 128,
                out_host_buffer: host_batch_bufs[i].as_mut_ptr(),
                out_capacity: CHUNK_OUT_BYTES as u32,
                status: 0,
            }
        })
        .collect();
    let mut batch_kernel_ms = 0.0f64;
    let t_batch = Instant::now();
    let ok_count = unsafe {
        BcPackDecodeBatch(
            ffi_ctx,
            batch_items.as_mut_ptr(),
            batch_items.len() as u32,
            &mut batch_kernel_ms,
        )
    };
    let batch_wall_ms = t_batch.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(ok_count, CHUNKS as i32, "BcPackDecodeBatch partial failure: {ok_count}");

    // Phase 3: Zero-Copy BcPackDecodeToTexture2D (GPU-to-GPU CopyTextureRegion, no CPU readback)
    let mut direct_fence = 0u64;
    let t_zc = Instant::now();
    for i in 0..CHUNKS {
        let s = &encoded_pool[i % encoded_pool.len()];
        let rc = unsafe {
            BcPackDecodeToTexture2D(
                ffi_ctx,
                s.as_ptr(),
                s.len() as u32,
                BLOCKS_PER_CHUNK as u32,
                BCPACK_FORMAT_BC1,
                BCPACK_SWIZZLE_LINEAR,
                128,
                128,
                tex_pool[i % tex_pool.len()].as_raw(),
                0,
                0,
                0,
                &mut direct_fence,
            )
        };
        assert_eq!(rc, 0, "BcPackDecodeToTexture2D failed at {i}: {rc}");
    }
    let wrc = unsafe { BcPackWaitDirectFence(ffi_ctx, direct_fence, u32::MAX) };
    assert_eq!(wrc, 0, "BcPackWaitDirectFence failed: {wrc}");
    let zerocopy_tex2d_wall_ms = t_zc.elapsed().as_secs_f64() * 1000.0;

    // Phase 4 (v0.15.0): Batched Zero-Copy BcPackDecodeBatchToTextures (1 compute batch + 1 DIRECT command list + 1 fence)
    let mut tex_batch_items: Vec<BcPackTextureBatchItem> = (0..CHUNKS)
        .map(|i| {
            let s = &encoded_pool[i % encoded_pool.len()];
            BcPackTextureBatchItem {
                compressed_data: s.as_ptr(),
                compressed_size: s.len() as u32,
                num_blocks: BLOCKS_PER_CHUNK as u32,
                format: BCPACK_FORMAT_BC1,
                swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                width_blocks: 128,
                height_blocks: 128,
                dst_texture2d: tex_pool[i % tex_pool.len()].as_raw(),
                subresource_index: 0,
                dst_x_blocks: 0,
                dst_y_blocks: 0,
                status: 0,
            }
        })
        .collect();
    let mut batch_tex_fence = 0u64;
    let t_btex = Instant::now();
    let btex_ok = unsafe {
        BcPackDecodeBatchToTextures(
            ffi_ctx,
            tex_batch_items.as_mut_ptr(),
            tex_batch_items.len() as u32,
            &mut batch_tex_fence,
        )
    };
    assert_eq!(btex_ok, CHUNKS as i32, "BcPackDecodeBatchToTextures failed: {btex_ok}");
    let bwrc = unsafe { BcPackWaitDirectFence(ffi_ctx, batch_tex_fence, u32::MAX) };
    assert_eq!(bwrc, 0, "BcPackWaitDirectFence (batch) failed: {bwrc}");
    let zerocopy_batch_tex2d_wall_ms = t_btex.elapsed().as_secs_f64() * 1000.0;

    unsafe {
        bcpack_gpu_destroy(ffi_ctx);
    }
    check_device_ok(&ctx.device, "Bench1 VirtualTile+BCPack");

    let total_mb = (CHUNKS * CHUNK_OUT_BYTES) as f64 / (1024.0 * 1024.0);
    let zerocopy_tex2d_mbs = total_mb / (zerocopy_tex2d_wall_ms / 1000.0);
    let zerocopy_batch_tex2d_mbs = total_mb / (zerocopy_batch_tex2d_wall_ms / 1000.0);
    let batch_speedup = async_loop_wall_ms / batch_wall_ms.max(0.001);
    let zerocopy_speedup = async_loop_wall_ms / zerocopy_tex2d_wall_ms.max(0.001);
    let zerocopy_batch_speedup = zerocopy_tex2d_wall_ms / zerocopy_batch_tex2d_wall_ms.max(0.001);

    println!(
        "[bench1] Virtual Pool 4.0 GiB -> 256 MiB: LRU evictions={lru_evictions} (thrash={lru_thrash_reloads}) vs Pareto evictions={pareto_evictions} (thrash={pareto_thrash_reloads})"
    );
    println!(
        "[bench1] BCPack 16.0 MiB (128x512x512 BC1): AsyncLoop={async_loop_wall_ms:.2}ms | Batch={batch_wall_ms:.2}ms (kernel={batch_kernel_ms:.2}ms, {batch_speedup:.2}x) | ToTex2DLoop={zerocopy_tex2d_wall_ms:.2}ms ({zerocopy_tex2d_mbs:.1} MB/s) | BatchToTex2D={zerocopy_batch_tex2d_wall_ms:.2}ms ({zerocopy_batch_tex2d_mbs:.1} MB/s, {zerocopy_batch_speedup:.2}x vs loop)"
    );

    Bench1Report {
        virtual_catalog_gib: 4.0,
        budget_mib: 256,
        lru_evictions,
        lru_thrash_reloads,
        pareto_evictions,
        pareto_thrash_reloads,
        async_loop_wall_ms,
        batch_wall_ms,
        batch_kernel_ms,
        zerocopy_tex2d_wall_ms,
        zerocopy_tex2d_mbs,
        zerocopy_batch_tex2d_wall_ms,
        zerocopy_batch_tex2d_mbs,
        batch_speedup,
        zerocopy_speedup,
        zerocopy_batch_speedup,
    }
}

struct Bench2Report {
    objects_count: usize,
    inline_drop_ms: f64,
    reaper_handoff_ms: f64,
    reaper_bg_drain_ms: f64,
    pooled_ring_reset_ms: f64,
    projected_75k_inline_ms: f64,
    handoff_speedup: f64,
}

fn run_bench2_com_drop_storm(ctx: &GpuContext) -> Bench2Report {
    // Targets fxceless/victory-fork#1349 (dropping tens of thousands of retired ID3D12Resource COM handles per frame)
    const COUNT: usize = 10_000;
    const BUF_BYTES: u64 = 4096;

    let mut bufs_inline: Vec<ID3D12Resource> = Vec::with_capacity(COUNT);
    for _ in 0..COUNT {
        bufs_inline.push(create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            BUF_BYTES,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        ));
    }
    let t_inline = Instant::now();
    drop(bufs_inline);
    let inline_drop_ms = t_inline.elapsed().as_secs_f64() * 1000.0;

    let mut bufs_reaper: Vec<ID3D12Resource> = Vec::with_capacity(COUNT);
    for _ in 0..COUNT {
        bufs_reaper.push(create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            BUF_BYTES,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        ));
    }
    let (tx, rx) = std::sync::mpsc::channel::<Vec<ID3D12Resource>>();
    let (done_tx, done_rx) = std::sync::mpsc::channel::<f64>();
    let reaper = std::thread::spawn(move || {
        while let Ok(v) = rx.recv() {
            let td = Instant::now();
            drop(v);
            let _ = done_tx.send(td.elapsed().as_secs_f64() * 1000.0);
        }
    });
    let t_handoff = Instant::now();
    tx.send(bufs_reaper).unwrap();
    let reaper_handoff_ms = t_handoff.elapsed().as_secs_f64() * 1000.0;
    drop(tx);
    let reaper_bg_drain_ms = done_rx.recv().unwrap_or(0.0);
    let _ = reaper.join();

    // Pooled 40 MiB ring buffer: 1 ID3D12Resource sliced into COUNT x 4 KiB offsets
    let pool_res = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_UPLOAD,
        (COUNT as u64) * BUF_BYTES,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let mut ring_cursor: u64 = 0;
    let t_pool = Instant::now();
    let mut sum_off = 0u64;
    for _ in 0..COUNT {
        sum_off = sum_off.wrapping_add(ring_cursor);
        ring_cursor = (ring_cursor + BUF_BYTES) % ((COUNT as u64) * BUF_BYTES);
    }
    std::hint::black_box(sum_off);
    let pooled_ring_reset_ms = t_pool.elapsed().as_secs_f64() * 1000.0;
    drop(pool_res);

    check_device_ok(&ctx.device, "Bench2 ComDropStorm");
    let projected_75k_inline_ms = inline_drop_ms * 7.5;
    let handoff_speedup = inline_drop_ms / reaper_handoff_ms.max(0.0001);

    println!(
        "[bench2] Retired COM Drop Storm ({COUNT} x 4KiB UPLOAD): Inline={inline_drop_ms:.2}ms (proj 75k={projected_75k_inline_ms:.1}ms) | ReaperHandoff={reaper_handoff_ms:.4}ms (bg={reaper_bg_drain_ms:.2}ms, {handoff_speedup:.0}x caller relief) | PooledRing={pooled_ring_reset_ms:.4}ms"
    );

    Bench2Report {
        objects_count: COUNT,
        inline_drop_ms,
        reaper_handoff_ms,
        reaper_bg_drain_ms,
        pooled_ring_reset_ms,
        projected_75k_inline_ms,
        handoff_speedup,
    }
}

struct Bench3Report {
    total_mib: usize,
    naive_copies: usize,
    coalesced_copies: usize,
    naive_cpu_record_ms: f64,
    naive_wall_ms: f64,
    coalesced_cpu_record_ms: f64,
    coalesced_wall_ms: f64,
    speedup: f64,
}

fn run_bench3_copy_coalescing(ctx: &mut GpuContext) -> Bench3Report {
    // Targets fxceless/victory-fork#1187 / #1212 / VICTORY_JOIN_COPIES
    const CHUNKS: usize = 4_096;
    const CHUNK_BYTES: u64 = 8_192;
    const TOTAL_BYTES: u64 = (CHUNKS as u64) * CHUNK_BYTES; // 32 MiB

    let upload_src = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_UPLOAD,
        TOTAL_BYTES,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let default_dst = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_DEFAULT,
        TOTAL_BYTES,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COMMON,
    );

    let alloc: ID3D12CommandAllocator = unsafe {
        ctx.device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap()
    };
    let cmd: ID3D12GraphicsCommandList = unsafe {
        ctx.device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .unwrap()
    };

    // 3A. Naive: 4,096 separate CopyBufferRegion calls with per-copy barrier transitions
    let t_naive = Instant::now();
    unsafe {
        for i in 0..CHUNKS as u64 {
            let off = i * CHUNK_BYTES;
            cmd_barriers(
                &cmd,
                vec![transition_barrier(
                    &default_dst,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )],
            );
            cmd.CopyBufferRegion(&default_dst, off, &upload_src, off, CHUNK_BYTES);
            cmd_barriers(
                &cmd,
                vec![transition_barrier(
                    &default_dst,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COMMON,
                )],
            );
        }
        cmd.Close().unwrap();
    }
    let naive_cpu_record_ms = t_naive.elapsed().as_secs_f64() * 1000.0;
    unsafe {
        let exec = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&exec);
        ctx.signal_and_wait();
    }
    let naive_wall_ms = t_naive.elapsed().as_secs_f64() * 1000.0;

    // 3B. Coalesced (join_copies): 32 x 1 MiB contiguous spans with 1 barrier pair total
    const COAL_COUNT: usize = 32;
    const SPAN_BYTES: u64 = TOTAL_BYTES / (COAL_COUNT as u64);
    unsafe {
        alloc.Reset().unwrap();
        cmd.Reset(&alloc, None).unwrap();
    }
    let t_coal = Instant::now();
    unsafe {
        cmd_barriers(
            &cmd,
            vec![transition_barrier(
                &default_dst,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )],
        );
        for s in 0..COAL_COUNT as u64 {
            let off = s * SPAN_BYTES;
            cmd.CopyBufferRegion(&default_dst, off, &upload_src, off, SPAN_BYTES);
        }
        cmd_barriers(
            &cmd,
            vec![transition_barrier(
                &default_dst,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            )],
        );
        cmd.Close().unwrap();
    }
    let coalesced_cpu_record_ms = t_coal.elapsed().as_secs_f64() * 1000.0;
    unsafe {
        let exec = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&exec);
        ctx.signal_and_wait();
    }
    let coalesced_wall_ms = t_coal.elapsed().as_secs_f64() * 1000.0;

    check_device_ok(&ctx.device, "Bench3 CopyCoalescing");
    let speedup = naive_wall_ms / coalesced_wall_ms.max(0.001);
    println!(
        "[bench3] Copy Coalescing (32 MiB): Naive ({CHUNKS}x8KiB+barriers) rec={naive_cpu_record_ms:.2}ms wall={naive_wall_ms:.2}ms | Coalesced ({COAL_COUNT}x1MiB) rec={coalesced_cpu_record_ms:.3}ms wall={coalesced_wall_ms:.2}ms ({speedup:.2}x)"
    );

    Bench3Report {
        total_mib: (TOTAL_BYTES / (1024 * 1024)) as usize,
        naive_copies: CHUNKS,
        coalesced_copies: COAL_COUNT,
        naive_cpu_record_ms,
        naive_wall_ms,
        coalesced_cpu_record_ms,
        coalesced_wall_ms,
        speedup,
    }
}

struct Bench4Report {
    dxr_supported: bool,
    blas_count: usize,
    instances_count: usize,
    blas_precompact_mib: f64,
    blas_compacted_mib: f64,
    shared_scratch_mib: f64,
    tlas_rebuild_avg_ms: f64,
    tlas_refit_avg_ms: f64,
    refit_speedup: f64,
}

fn run_bench4_bvh_streaming(ctx: &mut GpuContext) -> Bench4Report {
    // Targets core_bvh4.rpf (8.2 GiB) + XboxBvhBuilder.dll + HumanlyDev/HUMAN#7 bounded scratch
    const BLAS_COUNT: usize = 64;
    const TRIS_PER_BLAS: usize = 64;
    const INSTANCES: usize = 16_384;

    if !supports_dxr_1_1(&ctx.device) {
        println!("[bench4] DXR Tier 1.1 not supported on adapter; skipping.");
        return Bench4Report {
            dxr_supported: false,
            blas_count: 0,
            instances_count: 0,
            blas_precompact_mib: 0.0,
            blas_compacted_mib: 0.0,
            shared_scratch_mib: 0.0,
            tlas_rebuild_avg_ms: 0.0,
            tlas_refit_avg_ms: 0.0,
            refit_speedup: 0.0,
        };
    }

    unsafe {
        let dev5: ID3D12Device5 = ctx.device.cast().expect("ID3D12Device5");
        let verts_per_blas = TRIS_PER_BLAS * 3;
        let total_verts = BLAS_COUNT * verts_per_blas;
        let mut verts: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
        for b in 0..BLAS_COUNT {
            let phase = b as f32 * 0.13;
            for t in 0..TRIS_PER_BLAS {
                let x = (t % 8) as f32 * 1.2 + phase;
                let z = (t / 8) as f32 * 1.2;
                verts.push([x, 0.0, z]);
                verts.push([x + 0.9, 1.1, z + 0.2]);
                verts.push([x + 0.2, 0.1, z + 0.9]);
            }
        }
        let vb_up = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (verts.len() * size_of::<[f32; 3]>()) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        upload_slice(&vb_up, &verts);

        let vb_va = vb_up.GetGPUVirtualAddress();
        let blas_flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;

        let mut blas_raw: Vec<ID3D12Resource> = Vec::with_capacity(BLAS_COUNT);
        let mut blas_geom: Vec<D3D12_RAYTRACING_GEOMETRY_DESC> = Vec::with_capacity(BLAS_COUNT);
        let mut precompact_bytes = 0u64;
        let mut max_scratch = 256u64;

        for b in 0..BLAS_COUNT {
            let v_off = ((b * verts_per_blas) * size_of::<[f32; 3]>()) as u64;
            let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_UNKNOWN,
                        VertexFormat: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 0,
                        VertexCount: verts_per_blas as u32,
                        IndexBuffer: 0,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vb_va + v_off,
                            StrideInBytes: size_of::<[f32; 3]>() as u64,
                        },
                    },
                },
            };
            let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: blas_flags,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: &geom,
                },
            };
            let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
            dev5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre);
            max_scratch = max_scratch.max(pre.ScratchDataSizeInBytes);
            precompact_bytes += pre.ResultDataMaxSizeInBytes;

            blas_raw.push(create_committed_buffer(
                &ctx.device,
                D3D12_HEAP_TYPE_DEFAULT,
                pre.ResultDataMaxSizeInBytes.max(256),
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            ));
            blas_geom.push(geom);
        }

        let shared_scratch = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            max_scratch,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let post_info_bytes = (BLAS_COUNT * 8) as u64;
        let post_info_uav = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            post_info_bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let post_info_rb = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_READBACK,
            post_info_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let alloc: ID3D12CommandAllocator = ctx
            .device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let cmd: ID3D12GraphicsCommandList = ctx
            .device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .unwrap();
        let cmd4: ID3D12GraphicsCommandList4 = cmd.cast().unwrap();

        let post_va = post_info_uav.GetGPUVirtualAddress();
        for b in 0..BLAS_COUNT {
            let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: blas_flags,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: &blas_geom[b],
                },
            };
            let bdesc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: blas_raw[b].GetGPUVirtualAddress(),
                Inputs: inputs,
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: shared_scratch.GetGPUVirtualAddress(),
            };
            cmd4.BuildRaytracingAccelerationStructure(&bdesc, None);
            cmd_barriers(&cmd, vec![uav_barrier(&blas_raw[b]), uav_barrier(&shared_scratch)]);
            let pdesc = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: post_va + (b as u64) * 8,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            };
            let src_va = [blas_raw[b].GetGPUVirtualAddress()];
            cmd4.EmitRaytracingAccelerationStructurePostbuildInfo(&pdesc, &src_va);
        }
        cmd_barriers(
            &cmd,
            vec![transition_barrier(
                &post_info_uav,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )],
        );
        cmd.CopyBufferRegion(&post_info_rb, 0, &post_info_uav, 0, post_info_bytes);
        cmd.Close().unwrap();
        ctx.queue
            .ExecuteCommandLists(&[Some(cmd.cast::<ID3D12CommandList>().unwrap())]);
        ctx.signal_and_wait();

        let mut compacted_sizes = vec![0u64; BLAS_COUNT];
        let mut rptr: *mut c_void = std::ptr::null_mut();
        post_info_rb
            .Map(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: post_info_bytes as usize,
                }),
                Some(&mut rptr),
            )
            .unwrap();
        std::ptr::copy_nonoverlapping(
            rptr as *const u64,
            compacted_sizes.as_mut_ptr(),
            BLAS_COUNT,
        );
        post_info_rb.Unmap(0, None);

        let compacted_bytes: u64 = compacted_sizes.iter().sum();
        let mut blas_compacted: Vec<ID3D12Resource> = Vec::with_capacity(BLAS_COUNT);
        alloc.Reset().unwrap();
        cmd.Reset(&alloc, None).unwrap();
        for b in 0..BLAS_COUNT {
            let sz = compacted_sizes[b].max(256);
            let dst = create_committed_buffer(
                &ctx.device,
                D3D12_HEAP_TYPE_DEFAULT,
                sz,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            );
            cmd4.CopyRaytracingAccelerationStructure(
                dst.GetGPUVirtualAddress(),
                blas_raw[b].GetGPUVirtualAddress(),
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
            );
            blas_compacted.push(dst);
        }
        cmd_barriers(&cmd, vec![uav_barrier_all()]);
        cmd.Close().unwrap();
        ctx.queue
            .ExecuteCommandLists(&[Some(cmd.cast::<ID3D12CommandList>().unwrap())]);
        ctx.signal_and_wait();
        drop(blas_raw); // retire uncompacted BLAS after compaction

        // Build & update 16,384-instance TLAS
        let inst_bytes = (INSTANCES * size_of::<D3D12_RAYTRACING_INSTANCE_DESC>()) as u64;
        let inst_up = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            inst_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let mut inst_descs = Vec::with_capacity(INSTANCES);
        for i in 0..INSTANCES {
            let gx = (i % 128) as f32 * 4.0;
            let gz = (i / 128) as f32 * 4.0;
            inst_descs.push(D3D12_RAYTRACING_INSTANCE_DESC {
                Transform: [1.0, 0.0, 0.0, gx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, gz],
                _bitfield1: (i as u32 & 0x00FF_FFFF) | (0xFFu32 << 24),
                _bitfield2: 0,
                AccelerationStructure: blas_compacted[i % BLAS_COUNT].GetGPUVirtualAddress(),
            });
        }
        upload_slice(&inst_up, &inst_descs);

        let tlas_flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: tlas_flags,
            NumDescs: INSTANCES as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: inst_up.GetGPUVirtualAddress(),
            },
        };
        let mut tlas_pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_pre);

        let tlas_res = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            tlas_pre.ResultDataMaxSizeInBytes.max(256),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        );
        let tlas_scratch = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            tlas_pre
                .ScratchDataSizeInBytes
                .max(tlas_pre.UpdateScratchDataSizeInBytes)
                .max(256),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );

        // 10 full TLAS rebuilds
        let t_reb = Instant::now();
        for _ in 0..10 {
            alloc.Reset().unwrap();
            cmd.Reset(&alloc, None).unwrap();
            let bdesc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: tlas_res.GetGPUVirtualAddress(),
                Inputs: tlas_inputs,
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: tlas_scratch.GetGPUVirtualAddress(),
            };
            cmd4.BuildRaytracingAccelerationStructure(&bdesc, None);
            cmd_barriers(&cmd, vec![uav_barrier_all()]);
            cmd.Close().unwrap();
            ctx.queue
                .ExecuteCommandLists(&[Some(cmd.cast::<ID3D12CommandList>().unwrap())]);
            ctx.signal_and_wait();
        }
        let tlas_rebuild_avg_ms = (t_reb.elapsed().as_secs_f64() * 1000.0) / 10.0;

        // 10 incremental PERFORM_UPDATE TLAS refits
        let mut refit_inputs = tlas_inputs;
        refit_inputs.Flags =
            tlas_flags | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        let t_refit = Instant::now();
        for iter in 0..10 {
            for i in 0..512 {
                inst_descs[i].Transform[3] += if iter % 2 == 0 { 0.05 } else { -0.05 };
            }
            upload_slice(&inst_up, &inst_descs[..512]);
            alloc.Reset().unwrap();
            cmd.Reset(&alloc, None).unwrap();
            let bdesc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: tlas_res.GetGPUVirtualAddress(),
                Inputs: refit_inputs,
                SourceAccelerationStructureData: tlas_res.GetGPUVirtualAddress(),
                ScratchAccelerationStructureData: tlas_scratch.GetGPUVirtualAddress(),
            };
            cmd4.BuildRaytracingAccelerationStructure(&bdesc, None);
            cmd_barriers(&cmd, vec![uav_barrier_all()]);
            cmd.Close().unwrap();
            ctx.queue
                .ExecuteCommandLists(&[Some(cmd.cast::<ID3D12CommandList>().unwrap())]);
            ctx.signal_and_wait();
        }
        let tlas_refit_avg_ms = (t_refit.elapsed().as_secs_f64() * 1000.0) / 10.0;

        check_device_ok(&ctx.device, "Bench4 DXR1.1_BVH");

        let blas_precompact_mib = precompact_bytes as f64 / (1024.0 * 1024.0);
        let blas_compacted_mib = compacted_bytes as f64 / (1024.0 * 1024.0);
        let shared_scratch_mib = (max_scratch as f64) / (1024.0 * 1024.0);
        let refit_speedup = tlas_rebuild_avg_ms / tlas_refit_avg_ms.max(0.001);

        println!(
            "[bench4] DXR 1.1 BVH4 ({BLAS_COUNT} BLAS + {INSTANCES} TLAS): PreCompact={blas_precompact_mib:.2}MiB -> Compacted={blas_compacted_mib:.2}MiB (shared scratch={shared_scratch_mib:.2}MiB) | TLAS Rebuild={tlas_rebuild_avg_ms:.2}ms vs Refit={tlas_refit_avg_ms:.2}ms ({refit_speedup:.2}x)"
        );

        Bench4Report {
            dxr_supported: true,
            blas_count: BLAS_COUNT,
            instances_count: INSTANCES,
            blas_precompact_mib,
            blas_compacted_mib,
            shared_scratch_mib,
            tlas_rebuild_avg_ms,
            tlas_refit_avg_ms,
            refit_speedup,
        }
    }
}

struct Bench5Report {
    virtual_descs: usize,
    physical_slots: usize,
    total_accesses: usize,
    paging_evictions: u64,
    avg_frame_paging_ms: f64,
    aliasing_switches: usize,
    aliasing_wall_ms: f64,
}

fn run_bench5_descriptor_and_aliasing(ctx: &mut GpuContext) -> Bench5Report {
    // Targets fxceless/victory-fork#1351 + xcl-dev/xcl#2707 (1.25M->262k paging + ID3D12Heap placed resource aliasing)
    let mut pager = VirtualDescriptorPager::new(ctx);
    const FRAMES: usize = 10;
    const OPS_PER_FRAME: usize = 100_000;

    let t_page = Instant::now();
    for f in 0..FRAMES {
        for k in 0..OPS_PER_FRAME {
            let vid = ((f * 131_071 + k * 13) % VIRTUAL_DESC_COUNT) as u32;
            let _ = pager.page_in(&ctx.device, vid);
        }
    }
    let paging_total_ms = t_page.elapsed().as_secs_f64() * 1000.0;
    let avg_frame_paging_ms = paging_total_ms / (FRAMES as f64);

    // PlacedResource Heap Aliasing (16 x 512x512 BC1 textures overlapping at offset 0 of one 4 MiB ID3D12Heap)
    let tex_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 512,
        Height: 512,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_BC1_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let heap_size = 4u64 * 1024 * 1024;
    let heap_desc = D3D12_HEAP_DESC {
        SizeInBytes: heap_size,
        Properties: heap_props(D3D12_HEAP_TYPE_DEFAULT),
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
    };

    const ALIAS_COUNT: usize = 16;
    const SWITCHES: usize = 64;
    let mut alias_wall_ms = 0.0f64;

    unsafe {
        let mut heap: Option<ID3D12Heap> = None;
        ctx.device
            .CreateHeap(&heap_desc, &mut heap)
            .expect("CreateHeap aliasing failed");
        let heap = heap.unwrap();

        let mut placed: Vec<ID3D12Resource> = Vec::with_capacity(ALIAS_COUNT);
        for _ in 0..ALIAS_COUNT {
            let mut r: Option<ID3D12Resource> = None;
            ctx.device
                .CreatePlacedResource(
                    &heap,
                    0,
                    &tex_desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut r,
                )
                .expect("CreatePlacedResource failed");
            placed.push(r.unwrap());
        }

        let alloc: ID3D12CommandAllocator = ctx
            .device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let cmd: ID3D12GraphicsCommandList = ctx
            .device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .unwrap();

        let t_alias = Instant::now();
        for i in 0..SWITCHES {
            let prev = &placed[i % ALIAS_COUNT];
            let next = &placed[(i + 1) % ALIAS_COUNT];
            let mut b = [D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                        pResourceBefore: ManuallyDrop::new(Some(prev.clone())),
                        pResourceAfter: ManuallyDrop::new(Some(next.clone())),
                    }),
                },
            }];
            cmd.ResourceBarrier(&b);
            let mut al = ManuallyDrop::take(&mut b[0].Anonymous.Aliasing);
            let _ = ManuallyDrop::take(&mut al.pResourceBefore);
            let _ = ManuallyDrop::take(&mut al.pResourceAfter);
        }
        cmd.Close().unwrap();
        ctx.queue
            .ExecuteCommandLists(&[Some(cmd.cast::<ID3D12CommandList>().unwrap())]);
        ctx.signal_and_wait();
        alias_wall_ms = t_alias.elapsed().as_secs_f64() * 1000.0;
    }

    check_device_ok(&ctx.device, "Bench5 DescriptorAndAliasing");
    println!(
        "[bench5] Descriptor Paging ({}x100k={}): evictions={} avg={avg_frame_paging_ms:.2}ms/frame | PlacedResource Aliasing ({ALIAS_COUNT} overlays, {SWITCHES} barriers): {alias_wall_ms:.3}ms",
        FRAMES,
        FRAMES * OPS_PER_FRAME,
        pager.evictions
    );

    Bench5Report {
        virtual_descs: VIRTUAL_DESC_COUNT,
        physical_slots: PHYS_HEAP_SLOTS,
        total_accesses: FRAMES * OPS_PER_FRAME,
        paging_evictions: pager.evictions,
        avg_frame_paging_ms,
        aliasing_switches: SWITCHES,
        aliasing_wall_ms: alias_wall_ms,
    }
}

struct Bench6Report {
    dispatch_count: usize,
    barrier_storm_wall_ms: f64,
    direct_nobarrier_wall_ms: f64,
    execute_indirect_wall_ms: f64,
    indirect_vs_storm_speedup: f64,
}

fn run_bench6_dispatch_vs_indirect(ctx: &mut GpuContext) -> Bench6Report {
    // Targets fxceless/victory-fork#1338 (VICTORY_DISPATCH_SPREAD barrier overhead) & VICTORY_INDIRECT_BATCH
    const DISPATCHES: usize = 2_048;

    unsafe {
        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                },
            },
        };
        let mut rs_blob = None;
        let mut err_blob = None;
        D3D12SerializeVersionedRootSignature(&rs_desc, &mut rs_blob, Some(&mut err_blob))
            .expect("Serialize RootSig Bench6");
        let rs_blob = rs_blob.unwrap();
        let rs_bytes = std::slice::from_raw_parts(
            rs_blob.GetBufferPointer() as *const u8,
            rs_blob.GetBufferSize(),
        );
        let root_sig: ID3D12RootSignature = ctx.device.CreateRootSignature(0, rs_bytes).unwrap();

        let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: BC16_PACK_CS_DXIL.as_ptr() as *const c_void,
                BytecodeLength: BC16_PACK_CS_DXIL.len(),
            },
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        let pso: ID3D12PipelineState = ctx
            .device
            .CreateComputePipelineState(&pso_desc)
            .expect("CreateComputePipelineState Bench6");
        let _ = ManuallyDrop::take(&mut pso_desc.pRootSignature);

        // Input 1024 symbol rows = 32 KiB, output 512 blocks = 8 KiB
        let src_buf = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            64 * 1024,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        let uav_buf = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            64 * 1024,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );

        let args_data = vec![
            D3D12_DISPATCH_ARGUMENTS {
                ThreadGroupCountX: 4,
                ThreadGroupCountY: 1,
                ThreadGroupCountZ: 1,
            };
            DISPATCHES
        ];
        let args_bytes = (args_data.len() * size_of::<D3D12_DISPATCH_ARGUMENTS>()) as u64;
        let args_up = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            args_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        upload_slice(&args_up, &args_data);
        let args_def = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            args_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let arg_desc = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let cmd_sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: size_of::<D3D12_DISPATCH_ARGUMENTS>() as u32,
            NumArgumentDescs: 1,
            pArgumentDescs: &arg_desc,
            NodeMask: 0,
        };
        let mut cmd_sig: Option<ID3D12CommandSignature> = None;
        ctx.device
            .CreateCommandSignature(&cmd_sig_desc, None, &mut cmd_sig)
            .expect("CreateCommandSignature failed");
        let cmd_sig = cmd_sig.unwrap();

        let alloc: ID3D12CommandAllocator = ctx
            .device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let cmd: ID3D12GraphicsCommandList = ctx
            .device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .unwrap();

        // Prepare args buffer + warmup PSO
        cmd.CopyBufferRegion(&args_def, 0, &args_up, 0, args_bytes);
        cmd_barriers(
            &cmd,
            vec![transition_barrier(
                &args_def,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )],
        );
        cmd.SetPipelineState(&pso);
        cmd.SetComputeRootSignature(&root_sig);
        let rc_vals = [128u32, 0, 0, 0];
        cmd.SetComputeRoot32BitConstants(0, 4, rc_vals.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, src_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(2, uav_buf.GetGPUVirtualAddress());
        cmd.Dispatch(4, 1, 1);
        cmd_barriers(&cmd, vec![uav_barrier(&uav_buf)]);
        cmd.Close().unwrap();
        ctx.queue
            .ExecuteCommandLists(&[Some(cmd.cast::<ID3D12CommandList>().unwrap())]);
        ctx.signal_and_wait();

        // 6A. Barrier Storm: DISPATCHES calls separated by uav_barrier
        alloc.Reset().unwrap();
        cmd.Reset(&alloc, Some(&pso)).unwrap();
        let t_storm = Instant::now();
        cmd.SetComputeRootSignature(&root_sig);
        cmd.SetComputeRoot32BitConstants(0, 4, rc_vals.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, src_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(2, uav_buf.GetGPUVirtualAddress());
        for _ in 0..DISPATCHES {
            cmd.Dispatch(4, 1, 1);
            cmd_barriers(&cmd, vec![uav_barrier(&uav_buf)]);
        }
        cmd.Close().unwrap();
        ctx.queue
            .ExecuteCommandLists(&[Some(cmd.cast::<ID3D12CommandList>().unwrap())]);
        ctx.signal_and_wait();
        let barrier_storm_wall_ms = t_storm.elapsed().as_secs_f64() * 1000.0;

        // 6B. No-Barrier Direct Dispatch Loop
        alloc.Reset().unwrap();
        cmd.Reset(&alloc, Some(&pso)).unwrap();
        let t_direct = Instant::now();
        cmd.SetComputeRootSignature(&root_sig);
        cmd.SetComputeRoot32BitConstants(0, 4, rc_vals.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, src_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(2, uav_buf.GetGPUVirtualAddress());
        for _ in 0..DISPATCHES {
            cmd.Dispatch(4, 1, 1);
        }
        cmd_barriers(&cmd, vec![uav_barrier(&uav_buf)]);
        cmd.Close().unwrap();
        ctx.queue
            .ExecuteCommandLists(&[Some(cmd.cast::<ID3D12CommandList>().unwrap())]);
        ctx.signal_and_wait();
        let direct_nobarrier_wall_ms = t_direct.elapsed().as_secs_f64() * 1000.0;

        // 6C. ExecuteIndirect Single Submission (2048 Dispatches)
        alloc.Reset().unwrap();
        cmd.Reset(&alloc, Some(&pso)).unwrap();
        let t_ind = Instant::now();
        cmd.SetComputeRootSignature(&root_sig);
        cmd.SetComputeRoot32BitConstants(0, 4, rc_vals.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, src_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(2, uav_buf.GetGPUVirtualAddress());
        cmd.ExecuteIndirect(&cmd_sig, DISPATCHES as u32, &args_def, 0, None, 0);
        cmd_barriers(&cmd, vec![uav_barrier(&uav_buf)]);
        cmd.Close().unwrap();
        ctx.queue
            .ExecuteCommandLists(&[Some(cmd.cast::<ID3D12CommandList>().unwrap())]);
        ctx.signal_and_wait();
        let execute_indirect_wall_ms = t_ind.elapsed().as_secs_f64() * 1000.0;

        check_device_ok(&ctx.device, "Bench6 DispatchVsIndirect");
        let indirect_vs_storm_speedup = barrier_storm_wall_ms / execute_indirect_wall_ms.max(0.001);

        println!(
            "[bench6] {DISPATCHES} Compute Dispatches: BarrierStorm={barrier_storm_wall_ms:.2}ms | DirectNoBarrier={direct_nobarrier_wall_ms:.2}ms | ExecuteIndirect={execute_indirect_wall_ms:.2}ms ({indirect_vs_storm_speedup:.2}x vs storm)"
        );

        let _ = buffer_desc(64, D3D12_RESOURCE_FLAG_NONE);
        Bench6Report {
            dispatch_count: DISPATCHES,
            barrier_storm_wall_ms,
            direct_nobarrier_wall_ms,
            execute_indirect_wall_ms,
            indirect_vs_storm_speedup,
        }
    }
}

pub fn run_bottleneck_suite(_assets_dir: &Path, report_json: Option<&Path>) {
    let mut ctx = GpuContext::new(false);
    println!(
        "=== BCPack GPU v0.15.0 Open-World DX12 Bottleneck Suite ===\n[suite] Adapter: {} (VRAM: {} MiB dedicated, {} MiB OS budget)",
        ctx.adapter_name, ctx.dedicated_vram_mb, ctx.os_vram_budget_mb
    );

    let b1 = run_bench1_residency_and_bcpack(&mut ctx);
    let b2 = run_bench2_com_drop_storm(&ctx);
    let b3 = run_bench3_copy_coalescing(&mut ctx);
    let b4 = run_bench4_bvh_streaming(&mut ctx);
    let b5 = run_bench5_descriptor_and_aliasing(&mut ctx);
    let b6 = run_bench6_dispatch_vs_indirect(&mut ctx);

    let json = format!(
        "{{\n  \"adapter\": \"{}\",\n  \"dedicated_vram_mb\": {},\n  \"os_vram_budget_mb\": {},\n  \"bench1_virtual_pool_bcpack\": {{\n    \"virtual_catalog_gib\": {:.1},\n    \"budget_mib\": {},\n    \"lru_evictions\": {},\n    \"lru_thrash_reloads\": {},\n    \"pareto_evictions\": {},\n    \"pareto_thrash_reloads\": {},\n    \"async_loop_wall_ms\": {:.3},\n    \"batch_wall_ms\": {:.3},\n    \"batch_kernel_ms\": {:.3},\n    \"zerocopy_tex2d_wall_ms\": {:.3},\n    \"zerocopy_tex2d_mbs\": {:.1},\n    \"zerocopy_batch_tex2d_wall_ms\": {:.3},\n    \"zerocopy_batch_tex2d_mbs\": {:.1},\n    \"batch_speedup\": {:.2},\n    \"zerocopy_speedup\": {:.2},\n    \"zerocopy_batch_speedup\": {:.2}\n  }},\n  \"bench2_com_drop_storm\": {{\n    \"objects_count\": {},\n    \"inline_drop_ms\": {:.3},\n    \"reaper_handoff_ms\": {:.4},\n    \"reaper_bg_drain_ms\": {:.3},\n    \"pooled_ring_reset_ms\": {:.4},\n    \"projected_75k_inline_ms\": {:.2},\n    \"handoff_speedup\": {:.1}\n  }},\n  \"bench3_copy_coalescing\": {{\n    \"total_mib\": {},\n    \"naive_copies\": {},\n    \"coalesced_copies\": {},\n    \"naive_cpu_record_ms\": {:.3},\n    \"naive_wall_ms\": {:.3},\n    \"coalesced_cpu_record_ms\": {:.3},\n    \"coalesced_wall_ms\": {:.3},\n    \"speedup\": {:.2}\n  }},\n  \"bench4_dxr_bvh_streaming\": {{\n    \"dxr_supported\": {},\n    \"blas_count\": {},\n    \"instances_count\": {},\n    \"blas_precompact_mib\": {:.3},\n    \"blas_compacted_mib\": {:.3},\n    \"shared_scratch_mib\": {:.3},\n    \"tlas_rebuild_avg_ms\": {:.3},\n    \"tlas_refit_avg_ms\": {:.3},\n    \"refit_speedup\": {:.2}\n  }},\n  \"bench5_descriptor_aliasing\": {{\n    \"virtual_descs\": {},\n    \"physical_slots\": {},\n    \"total_accesses\": {},\n    \"paging_evictions\": {},\n    \"avg_frame_paging_ms\": {:.3},\n    \"aliasing_switches\": {},\n    \"aliasing_wall_ms\": {:.3}\n  }},\n  \"bench6_dispatch_indirect\": {{\n    \"dispatch_count\": {},\n    \"barrier_storm_wall_ms\": {:.3},\n    \"direct_nobarrier_wall_ms\": {:.3},\n    \"execute_indirect_wall_ms\": {:.3},\n    \"indirect_vs_storm_speedup\": {:.2}\n  }}\n}}\n",
        ctx.adapter_name.replace('"', "\\\""),
        ctx.dedicated_vram_mb,
        ctx.os_vram_budget_mb,
        b1.virtual_catalog_gib,
        b1.budget_mib,
        b1.lru_evictions,
        b1.lru_thrash_reloads,
        b1.pareto_evictions,
        b1.pareto_thrash_reloads,
        b1.async_loop_wall_ms,
        b1.batch_wall_ms,
        b1.batch_kernel_ms,
        b1.zerocopy_tex2d_wall_ms,
        b1.zerocopy_tex2d_mbs,
        b1.zerocopy_batch_tex2d_wall_ms,
        b1.zerocopy_batch_tex2d_mbs,
        b1.batch_speedup,
        b1.zerocopy_speedup,
        b1.zerocopy_batch_speedup,
        b2.objects_count,
        b2.inline_drop_ms,
        b2.reaper_handoff_ms,
        b2.reaper_bg_drain_ms,
        b2.pooled_ring_reset_ms,
        b2.projected_75k_inline_ms,
        b2.handoff_speedup,
        b3.total_mib,
        b3.naive_copies,
        b3.coalesced_copies,
        b3.naive_cpu_record_ms,
        b3.naive_wall_ms,
        b3.coalesced_cpu_record_ms,
        b3.coalesced_wall_ms,
        b3.speedup,
        b4.dxr_supported,
        b4.blas_count,
        b4.instances_count,
        b4.blas_precompact_mib,
        b4.blas_compacted_mib,
        b4.shared_scratch_mib,
        b4.tlas_rebuild_avg_ms,
        b4.tlas_refit_avg_ms,
        b4.refit_speedup,
        b5.virtual_descs,
        b5.physical_slots,
        b5.total_accesses,
        b5.paging_evictions,
        b5.avg_frame_paging_ms,
        b5.aliasing_switches,
        b5.aliasing_wall_ms,
        b6.dispatch_count,
        b6.barrier_storm_wall_ms,
        b6.direct_nobarrier_wall_ms,
        b6.execute_indirect_wall_ms,
        b6.indirect_vs_storm_speedup,
    );

    if let Some(path) = report_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let mut f = File::create(path).expect("create bottleneck report json");
        f.write_all(json.as_bytes()).expect("write bottleneck report json");
        println!("[suite] Wrote JSON report to {}", path.display());
    }
    println!("[suite] COMPLETE: all 6 bottleneck microbenchmarks passed cleanly.");
}
