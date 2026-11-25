use bcpack::bc1::{self, QuadTable, Window};
use bcpack::encode::synthetic_bc1;
use bcpack::rans;
use bcpack_gpu::assets::{
    encode_bc1_to_bcpack, generate_bc7_16lane_stream, generate_high_entropy_bc1_normal_map,
    generate_mip_chain_bc1, generate_sharp_ui_bc1, load_or_generate_assets, swizzle_gfx10_cpu,
    Bc7EntropyProfile, BC7_ALPHABET_BITS, NUM_BLOCKS, TEX_BC1_BYTES, TILE_BC1_BYTES, TILE_BLOCKS,
    TILE_STRIPS,
};
use bcpack_gpu::demo_scene::{run_demo, DemoConfig, DemoMode};
use bcpack_gpu::descriptor_paging::run_descriptor_paging_verify;
use bcpack_gpu::ffi::{
    bcpack_gpu_create, bcpack_gpu_destroy, BcPackDecodeAsync, BcPackGetLastTimingMs,
    BcPackWaitFence, BCPACK_FORMAT_BC1, BCPACK_FORMAT_BC7_RANS, BCPACK_SWIZZLE_GFX10_4KB_S,
    BCPACK_SWIZZLE_GFX10_64KB_D, BCPACK_SWIZZLE_LINEAR,
};
use bcpack_gpu::gltf_loader::run_aaa_2k_bcpack_verify;
use bcpack_gpu::gpu_decoder::{
    pack_batch, Bc7RansGpuDecoder, BcpackGpuDecoder, Gfx10DetileGpu, GpuContext,
};
use bcpack_gpu::meshlets::{
    build_city_meshlet_library, pack_quantized_meshlet_vertices, run_geom_bcpack_verify,
    run_mesh_shader_verify,
};
use bcpack_gpu::raytracing::run_rt_verify;
use bcpack_gpu::rtgi::run_rtgi_verify;
use bcpack_gpu::world_streaming::{run_jet_flight_verify, run_world_stream_verify};
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::Instant;

fn run_verify(assets_dir: &PathBuf, force_regen: bool) {
    let window = Window::default_table();
    let quads = QuadTable::default_table();
    let assets = load_or_generate_assets(assets_dir, &window, &quads, force_regen);

    let mut ctx = GpuContext::new(true);
    println!("[verify] Adapter: {} (DX12 Debug Layer + Async Compute Queue)", ctx.adapter_name);

    // 1) Synthetic BC1 vector tests (5 chunks = odd count, testing dual-chunk padding + prestart refs)
    let cases = [
        (101u64, 64usize, 16usize),
        (202u64, 256usize, 64usize),
        (303u64, 1024usize, 128usize),
        (404u64, 4096usize, 512usize),
        (505u64, 16384usize, 2048usize),
    ];

    let total_tiles = assets.len() * TILE_STRIPS;
    let max_blocks: usize = cases
        .iter()
        .map(|c| c.1)
        .sum::<usize>()
        .max(assets.len() * NUM_BLOCKS)
        .max(512 * TILE_BLOCKS);
    let max_chunks = cases.len().max(total_tiles).max(512);
    let mut gpu_dec = BcpackGpuDecoder::new(&mut ctx, &window, &quads, max_chunks, max_blocks, 32 * 1024 * 1024);

    let mut synth_streams = Vec::new();
    let mut synth_expected = Vec::new();
    for &(seed, blocks, max_frag) in &cases {
        let (stream, _) = synthetic_bc1(seed, blocks, max_frag);
        let cpu_out = bc1::decode(&stream, blocks, &window, &quads).expect("cpu synth decode");
        synth_streams.push((stream, blocks));
        synth_expected.push(cpu_out.blocks);
    }

    let synth_batch = pack_batch(synth_streams.iter().map(|(s, b)| (s.as_slice(), *b)));
    let (gpu_synth, statuses, gpu_ms, _) = gpu_dec.decode_batch_and_readback(&mut ctx, &synth_batch);

    let mut offset = 0usize;
    for (ci, ((_, blocks), exp)) in synth_streams.iter().zip(synth_expected.iter()).enumerate() {
        let st = statuses[ci];
        assert_eq!(st.code, 0, "GPU status error on synth case {ci}: {st:?}");
        let byte_len = blocks * 8;
        let got = &gpu_synth[offset..offset + byte_len];
        if got != exp.as_slice() {
            let mm = got.iter().zip(exp.iter()).position(|(a, b)| a != b).unwrap();
            panic!(
                "FAIL synthetic case {ci} (blocks={blocks}): mismatch at byte {mm} (gpu={:#04x}, cpu={:#04x})",
                got[mm], exp[mm]
            );
        }
        offset += byte_len;
    }
    println!(
        "[verify] PASS 1/15 BC1 synthetic streams ({} chunks, {} blocks, Async Compute GPU={:.3} ms, 0 byte mismatches)",
        cases.len(),
        cases.iter().map(|c| c.1).sum::<usize>(),
        gpu_ms
    );

    // 2) Bit-exact verification on all 12 monolithic 16,384-block .bcpack textures
    let tex_batch = pack_batch(assets.iter().map(|a| (a.bcpack_bytes.as_slice(), NUM_BLOCKS)));
    let (gpu_tex, tex_statuses, tex_gpu_ms, _) = gpu_dec.decode_batch_and_readback(&mut ctx, &tex_batch);

    for (ti, asset) in assets.iter().enumerate() {
        let st = tex_statuses[ti];
        assert_eq!(st.code, 0, "GPU status error on texture {}: {st:?}", asset.name);
        let got = &gpu_tex[ti * TEX_BC1_BYTES..(ti + 1) * TEX_BC1_BYTES];
        if got != asset.expected_bc1.as_slice() {
            let mm = got.iter().zip(asset.expected_bc1.iter()).position(|(a, b)| a != b).unwrap();
            panic!("FAIL monolithic texture {}: mismatch at byte {mm}", asset.name);
        }
    }
    println!(
        "[verify] PASS 2/15 BC1 monolithic 512x512 textures ({} textures = 1.50 MiB, GPU={:.3} ms, 0 byte mismatches)",
        assets.len(),
        tex_gpu_ms
    );

    // 3) Bit-exact verification on all 48 tiled 4,096-block .bcpack strips
    let tile_batch = pack_batch(assets.iter().flat_map(|a| {
        a.bcpack_tiles.iter().map(|t| (t.as_slice(), TILE_BLOCKS))
    }));
    let (gpu_tiles, tile_statuses, tile_gpu_ms, _) = gpu_dec.decode_batch_and_readback(&mut ctx, &tile_batch);

    for (ti, asset) in assets.iter().enumerate() {
        for strip in 0..TILE_STRIPS {
            let idx = ti * TILE_STRIPS + strip;
            let st = tile_statuses[idx];
            assert_eq!(st.code, 0, "GPU status error on tile {}[{strip}]: {st:?}", asset.name);
            let got = &gpu_tiles[idx * TILE_BC1_BYTES..(idx + 1) * TILE_BC1_BYTES];
            let exp = &asset.expected_bc1[strip * TILE_BC1_BYTES..(strip + 1) * TILE_BC1_BYTES];
            if got != exp {
                let mm = got.iter().zip(exp.iter()).position(|(a, b)| a != b).unwrap();
                panic!("FAIL tile {}[{strip}]: mismatch at byte {mm}", asset.name);
            }
        }
    }
    println!(
        "[verify] PASS 3/15 BC1 tiled 4K-block strips ({} tiles = 1.50 MiB, GPU={:.3} ms, 0 byte mismatches)",
        total_tiles,
        tile_gpu_ms
    );

    // 4) BC7 16-lane rANS (100% Wave32 = 32/32 threads) verification across 5 odd chunks + entropy profiles
    let bc7_cases = [
        (701u64, 64usize, Bc7EntropyProfile::StandardAlbedo),
        (702u64, 256usize, Bc7EntropyProfile::NormalHighEntropy),
        (703u64, 1024usize, Bc7EntropyProfile::SharpUi),
        (704u64, 2048usize, Bc7EntropyProfile::StandardAlbedo),
        (705u64, 4096usize, Bc7EntropyProfile::NormalHighEntropy),
    ];
    let mut bc7_streams = Vec::new();
    let mut bc7_expected = Vec::new();
    let mut total_sym_rows = 0usize;
    for &(seed, bc7_blocks, prof) in &bc7_cases {
        let (stream, rows) = generate_bc7_16lane_stream(seed, bc7_blocks, prof);
        let cpu_out = rans::decode_slice(&stream, &BC7_ALPHABET_BITS, rows).expect("cpu bc7 decode");
        bc7_streams.push((stream, rows));
        bc7_expected.push((rows, cpu_out));
        total_sym_rows += rows;
    }
    let mut bc7_dec = Bc7RansGpuDecoder::new(&mut ctx, 16, total_sym_rows, 16 * 1024 * 1024);
    let bc7_batch = pack_batch(bc7_streams.iter().map(|(s, r)| (s.as_slice(), *r)));
    let (gpu_bc7_syms, bc7_statuses, bc7_gpu_ms, _) = bc7_dec.decode_batch_and_readback(&mut ctx, &bc7_batch);

    let mut row_off = 0usize;
    for (ci, (rows, cpu_out)) in bc7_expected.iter().enumerate() {
        let st = bc7_statuses[ci];
        assert_eq!(st.code, 0, "GPU status error on BC7 case {ci}: {st:?}");
        for r in 0..*rows {
            for li in 0..16 {
                let got = gpu_bc7_syms[(row_off + r) * 16 + li];
                let exp = cpu_out.lane(li)[r];
                assert_eq!(
                    got, exp,
                    "FAIL BC7 16-lane chunk {ci} row {r} lane {li}: gpu={got}, cpu={exp}"
                );
            }
        }
        row_off += *rows;
    }
    println!(
        "[verify] PASS 4/15 BC7 16-lane rANS (100% Wave32 32/32 threads, {} chunks, {} symbol-rows, GPU={:.3} ms, 0 symbol mismatches)",
        bc7_cases.len(),
        total_sym_rows,
        bc7_gpu_ms
    );

    // 5) AMD Gfx10 Swizzle (SW_64KB_D 512x512 & SW_4KB_S 256x256) -> BCPack -> Async GPU Decode -> GPU Detile -> Linear BC1
    let detile = Gfx10DetileGpu::new(&ctx, 1024 * 1024);
    let mip_chain = generate_mip_chain_bc1(0, &window, &quads);
    // Test SW_64KB_D on 512x512 (128x128 blocks) and SW_4KB_S on 256x256 (64x64 blocks)
    for (tile_mode, name, w_blk, h_blk, linear_bc1) in [
        (0u32, "SW_64KB_D (512x512)", 128usize, 128usize, &mip_chain[0].expected_bc1),
        (1u32, "SW_4KB_S (256x256)", 64usize, 64usize, &mip_chain[1].expected_bc1),
    ] {
        let swizzled = swizzle_gfx10_cpu(linear_bc1, w_blk, h_blk, tile_mode, 8);
        let swizzled_blocks = swizzled.len() / 8;
        let swizzled_bcpack = encode_bc1_to_bcpack(&swizzled, &window, &quads);
        let batch = pack_batch(std::iter::once((swizzled_bcpack.as_slice(), swizzled_blocks)));
        let (gpu_swizzled_out, sts, dec_ms, _) = gpu_dec.decode_batch_and_readback(&mut ctx, &batch);
        assert_eq!(sts[0].code, 0, "GPU status error on Gfx10 {name}");
        let (detiled, det_ms) = detile.detile_and_readback(&mut ctx, &gpu_swizzled_out, w_blk as u32, h_blk as u32, tile_mode, 8);
        assert_eq!(detiled.as_slice(), linear_bc1.as_slice(), "FAIL Gfx10 detile mismatch on {name}");
        println!(
            "[verify] PASS 5/15 Gfx10 {name} Swizzle->BCPack->GPU Decode({dec_ms:.3} ms)->Detile({det_ms:.3} ms) (0 byte mismatches)"
        );
    }

    // 6) Full Mipmap Chain + Packed Mip Tail (512x512 .. 64x64 + Packed Tail 32x32..4x4 = 85 blocks)
    let mip_batch = pack_batch(mip_chain.iter().map(|m| (m.bcpack.as_slice(), m.blocks)));
    let (gpu_mips, mip_sts, mip_ms, _) = gpu_dec.decode_batch_and_readback(&mut ctx, &mip_batch);
    let mut m_off = 0usize;
    for (mi, mip) in mip_chain.iter().enumerate() {
        assert_eq!(mip_sts[mi].code, 0, "GPU status error on mip {}", mip.label);
        let byte_len = mip.blocks * 8;
        assert_eq!(
            &gpu_mips[m_off..m_off + byte_len],
            mip.expected_bc1.as_slice(),
            "FAIL mip chain mismatch on {}",
            mip.label
        );
        m_off += byte_len;
    }
    println!(
        "[verify] PASS 6/15 Mipmap Chain + Packed Mip Tail ({} chunks: 512x512..64x64 + 85-block tail, GPU={:.3} ms, 0 byte mismatches)",
        mip_chain.len(),
        mip_ms
    );

    // 7) C-ABI FFI Smoke Test (bcpack_gpu.dll interface: BcPackDecodeAsync linear + Gfx10 SW_64KB_D + BC7 16-lane)
    unsafe {
        let ffi_ctx = bcpack_gpu_create(64, 65536, 16 * 1024 * 1024);
        assert!(!ffi_ctx.is_null(), "bcpack_gpu_create returned null");

        let mut fence_val = 0u64;
        let mut host_out = vec![0u8; TEX_BC1_BYTES];
        let rc1 = BcPackDecodeAsync(
            ffi_ctx,
            assets[0].bcpack_bytes.as_ptr(),
            assets[0].bcpack_bytes.len() as u32,
            NUM_BLOCKS as u32,
            BCPACK_FORMAT_BC1,
            BCPACK_SWIZZLE_LINEAR,
            128,
            128,
            host_out.as_mut_ptr(),
            host_out.len() as u32,
            &mut fence_val,
        );
        assert_eq!(rc1, 0, "FFI BcPackDecodeAsync linear failed: {rc1}");
        assert_eq!(BcPackWaitFence(ffi_ctx, fence_val, 5000), 0);
        assert_eq!(host_out, assets[0].expected_bc1, "FFI linear BC1 mismatch");

        // Swizzled SW_64KB_D end-to-end through BcPackDecodeAsync (fused Async Compute decode + detile)
        let swizzled_64k = swizzle_gfx10_cpu(&assets[1].expected_bc1, 128, 128, 0, 8);
        let swizzled_stream = encode_bc1_to_bcpack(&swizzled_64k, &window, &quads);
        let rc2 = BcPackDecodeAsync(
            ffi_ctx,
            swizzled_stream.as_ptr(),
            swizzled_stream.len() as u32,
            (swizzled_64k.len() / 8) as u32,
            BCPACK_FORMAT_BC1,
            BCPACK_SWIZZLE_GFX10_64KB_D,
            128,
            128,
            host_out.as_mut_ptr(),
            host_out.len() as u32,
            &mut fence_val,
        );
        assert_eq!(rc2, 0, "FFI BcPackDecodeAsync SW_64KB_D failed: {rc2}");
        assert_eq!(host_out, assets[1].expected_bc1, "FFI Gfx10 detiled BC1 mismatch");

        // Async non-blocking BC7 16-lane dispatch + fence wait
        let (bc7_stream, bc7_rows) = &bc7_streams[0];
        let mut bc7_async_fence = 0u64;
        let rc3 = BcPackDecodeAsync(
            ffi_ctx,
            bc7_stream.as_ptr(),
            bc7_stream.len() as u32,
            (bc7_rows / 2) as u32,
            BCPACK_FORMAT_BC7_RANS,
            BCPACK_SWIZZLE_LINEAR,
            0,
            0,
            std::ptr::null_mut(),
            0,
            &mut bc7_async_fence,
        );
        assert_eq!(rc3, 0, "FFI BcPackDecodeAsync BC7 non-blocking failed: {rc3}");
        assert_eq!(BcPackWaitFence(ffi_ctx, bc7_async_fence, 5000), 0);

        let mut kms = 0.0f64;
        assert_eq!(BcPackGetLastTimingMs(ffi_ctx, &mut kms), 0);
        bcpack_gpu_destroy(ffi_ctx);
        println!(
            "[verify] PASS 7/15 C-ABI FFI BcPackDecodeAsync (Linear BC1 + Fused Gfx10 SW_64KB_D + Async BC7 16-lane, last_kernel={kms:.3} ms, 0 mismatches)"
        );
    }

    // 8) DXR 1.1 Hardware BVH + Inline RayQuery verification on RT Cores
    let rt_ms = run_rt_verify(&mut ctx);
    println!(
        "[verify] PASS 8/15 DXR 1.1 Hardware BLAS/TLAS + Inline RayQuery (4 analytical rays, GPU={rt_ms:.3} ms, 0 mismatches)"
    );

    // 9) World Grid 32x32 Streaming + LRU VRAM Residency Manager verification
    let ws = run_world_stream_verify(&mut ctx, &mut gpu_dec, &assets);
    println!(
        "[verify] PASS 9/15 World Grid 32x32 Streaming + LRU VRAM Residency ({} frames, {} tiles decoded, {} evictions, turn_burst={}t, peak={:.0} MB/s, GPU={:.3} ms, 0 mismatches)",
        ws.frames, ws.verified_tiles, ws.total_evictions, ws.turn_burst_tiles, ws.peak_mbs, ws.gpu_ms
    );

    // 10) DX12 Ultimate Mesh Shaders (as_6_5 / ms_6_5) + GPU Frustum Culling + 4M+ Tris City verification
    let ms = run_mesh_shader_verify(&mut ctx);
    println!(
        "[verify] PASS 10/15 Hardware Mesh Shaders (4.0M+ Tris City: total={:.2}M tris, work={}, vis={} / cull={}, vis_tris={:.2}M, lit_px={}, 0 mismatches)",
        ms.total_tris as f64 / 1e6,
        ms.total_work_items,
        ms.visible_meshlets,
        ms.culled_meshlets,
        ms.visible_tris as f64 / 1e6,
        ms.lit_pixels
    );

    // 11) DXR 1.1 Scale (1,536-Instance TLAS) + 2-Pass A-Trous Bilateral RTGI Denoiser verification
    let gi = run_rtgi_verify(&mut ctx);
    println!(
        "[verify] PASS 11/15 DXR 1.1 Scale ({} TLAS instances, build={:.3} ms) + RTGI Denoiser (var {:.4} -> {:.4}, edge_jump={:.2}, 0 mismatches)",
        gi.tlas_instances, gi.tlas_ms, gi.raw_var, gi.denoised_var, gi.edge_jump
    );

    // 12) BCPack 8B Quantized Meshlet Geometry Vertex Compression -> GPU Async Decode verification
    let gb = run_geom_bcpack_verify(&mut ctx, &mut gpu_dec, &window, &quads);
    println!(
        "[verify] PASS 12/15 BCPack 8B Meshlet Geometry ({} verts = {} KiB, ratio={:.2}x, GPU={:.3} ms, 0 mismatches)",
        gb.vertices, gb.raw_bytes / 1024, gb.ratio, gb.gpu_ms
    );

    // 13) Real glTF/GLB + 2K Photographic PBR -> Xbox .bcpack GPU Decode verification
    let aaa = run_aaa_2k_bcpack_verify(&mut ctx, &mut gpu_dec, assets_dir, &window, &quads);
    let ratio = aaa.raw_bytes as f64 / aaa.comp_bytes.max(1) as f64;
    println!(
        "[verify] PASS 13/15 AAA glTF+2K PBR BCPack (real={}, verts={}, tris={}, submeshes={} [glass={}], {} tex / {} tiles = {:.2} MiB, ratio={:.2}x, GPU={:.3} ms, 0 mismatches)",
        aaa.real_assets,
        aaa.verts,
        aaa.tris,
        aaa.submeshes,
        aaa.glass_submeshes,
        aaa.num_textures,
        aaa.verified_tiles,
        aaa.raw_bytes as f64 / (1024.0 * 1024.0),
        ratio,
        aaa.gpu_ms
    );

    // 14) RAGE-Scale Virtual Descriptor Paging (1.25M Virtual -> 262,144 Physical ResourceDescriptorHeap[] SM 6.6)
    let dp = run_descriptor_paging_verify(&mut ctx);
    println!(
        "[verify] PASS 14/15 Virtual Descriptor Paging ({:.2}M virt -> {} phys slots, {} ops, {} evictions, copy={:.2} ms, GPU={:.3} ms, {} samples, 0 mismatches)",
        dp.virtual_count as f64 / 1e6,
        dp.physical_slots,
        dp.accesses,
        dp.evictions,
        dp.copy_ms,
        dp.gpu_ms,
        dp.verified_samples
    );

    // 15) 64x64 World Grid Supersonic Jet-Flight Streaming (4,096 sectors = 256 km^2, 512-tile 16 MiB bursts)
    let jf = run_jet_flight_verify(&mut ctx, &mut gpu_dec, &assets);
    println!(
        "[verify] PASS 15/15 Jet-Flight 64x64 World Grid ({}x{} = {} sectors / {:.0} km^2, {}f, {}t streamed, {} evictions, max_burst={}t, peak={:.0} MB/s, GPU={}t={:.3} ms, 0 mismatches)",
        jf.grid_dim,
        jf.grid_dim,
        jf.total_sectors,
        jf.area_km2,
        jf.frames,
        jf.total_streamed_tiles,
        jf.total_evictions,
        jf.max_turn_burst_tiles,
        jf.peak_mbs,
        jf.verified_tiles,
        jf.gpu_ms
    );

    let _ = (BCPACK_SWIZZLE_GFX10_4KB_S,);
}

fn run_jet_flight_suite(assets_dir: &PathBuf) {
    let window = Window::default_table();
    let quads = QuadTable::default_table();
    let assets = load_or_generate_assets(assets_dir, &window, &quads, false);
    let mut ctx = GpuContext::new(false);
    let mut gpu_dec = BcpackGpuDecoder::new(&mut ctx, &window, &quads, 512, 512 * TILE_BLOCKS, 32 * 1024 * 1024);

    let dp = run_descriptor_paging_verify(&mut ctx);
    println!(
        "[jet-flight] Virtual Descriptor Paging: {:.2}M virt -> {} phys slots (ResourceDescriptorHeap[] SM 6.6), {} ops, {} evictions, CopyDescriptors={:.2} ms, GPU={:.3} ms, {} verified samples",
        dp.virtual_count as f64 / 1e6,
        dp.physical_slots,
        dp.accesses,
        dp.evictions,
        dp.copy_ms,
        dp.gpu_ms,
        dp.verified_samples
    );
    let jf = run_jet_flight_verify(&mut ctx, &mut gpu_dec, &assets);
    println!(
        "[jet-flight] 64x64 Streaming: {} sectors ({:.0} km^2), {} frames Mach 1/2 sweep, {} tiles streamed ({:.1} MiB), {} LRU evictions, max_burst={}t ({:.1} MiB), peak={:.1} MB/s, GPU 512-tile decode={:.3} ms",
        jf.total_sectors,
        jf.area_km2,
        jf.frames,
        jf.total_streamed_tiles,
        (jf.total_streamed_tiles * TILE_BC1_BYTES) as f64 / (1024.0 * 1024.0),
        jf.total_evictions,
        jf.max_turn_burst_tiles,
        (jf.max_turn_burst_tiles * TILE_BC1_BYTES) as f64 / (1024.0 * 1024.0),
        jf.peak_mbs,
        jf.gpu_ms
    );
}

fn bench_suite(
    label: &str,
    counts: &[usize],
    blocks_per_chunk: usize,
    bytes_per_chunk: usize,
    get_stream: impl Fn(usize) -> &'static [u8] + Sync,
    window: &Window,
    quads: &QuadTable,
    ctx: &mut GpuContext,
    gpu_dec: &mut BcpackGpuDecoder,
) {
    println!("--- {label} ({} blocks / {} KiB BC1 per chunk) ---", blocks_per_chunk, bytes_per_chunk / 1024);
    for &n_chunks in counts {
        let total_mb = (n_chunks * bytes_per_chunk) as f64 / (1024.0 * 1024.0);

        let t1 = Instant::now();
        let mut sink = 0u8;
        for i in 0..n_chunks {
            let out = bc1::decode(get_stream(i), blocks_per_chunk, window, quads).unwrap();
            sink ^= out.blocks[0];
        }
        let cpu1_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let cpu1_mbs = total_mb / (cpu1_ms / 1000.0);

        let t2 = Instant::now();
        let xor_sum: u8 = (0..n_chunks)
            .into_par_iter()
            .map(|i| {
                let out = bc1::decode(get_stream(i), blocks_per_chunk, window, quads).unwrap();
                out.blocks[0]
            })
            .reduce(|| 0u8, |a, b| a ^ b);
        let cpu_mc_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let cpu_mc_mbs = total_mb / (cpu_mc_ms / 1000.0);
        std::hint::black_box(sink ^ xor_sum);

        let batch = pack_batch((0..n_chunks).map(|i| (get_stream(i), blocks_per_chunk)));
        let _ = gpu_dec.bench_gpu_kernel_only(ctx, &batch);
        let mut gpu_k_sum = 0.0;
        let mut gpu_w_sum = 0.0;
        for _ in 0..5 {
            let (kms, wms) = gpu_dec.bench_gpu_kernel_only(ctx, &batch);
            gpu_k_sum += kms;
            gpu_w_sum += wms;
        }
        let gpu_k_ms = gpu_k_sum / 5.0;
        let gpu_w_ms = gpu_w_sum / 5.0;
        let gpu_k_mbs = total_mb / (gpu_k_ms / 1000.0);
        let gpu_w_mbs = total_mb / (gpu_w_ms / 1000.0);

        println!(
            "[bench] {:>4} chunks ({:>5.1} MiB) | CPU-1T: {:>6.2} ms ({:>5.1} MB/s) | CPU-MT: {:>6.2} ms ({:>5.1} MB/s) | GPU Async: {:>5.3} ms ({:>7.1} MB/s, {:.1}x vs 1T, {:.1}x vs MT) | GPU+upload: {:>5.2} ms ({:>6.1} MB/s)",
            n_chunks,
            total_mb,
            cpu1_ms,
            cpu1_mbs,
            cpu_mc_ms,
            cpu_mc_mbs,
            gpu_k_ms,
            gpu_k_mbs,
            gpu_k_mbs / cpu1_mbs,
            gpu_k_mbs / cpu_mc_mbs,
            gpu_w_ms,
            gpu_w_mbs
        );
    }
}

fn run_bench(assets_dir: &PathBuf) {
    let window = Window::default_table();
    let quads = QuadTable::default_table();
    let assets: &'static [_] = Box::leak(load_or_generate_assets(assets_dir, &window, &quads, false).into_boxed_slice());

    let mut ctx = GpuContext::new(false);
    println!("[bench] Adapter: {} (v0.3 Async Compute + BC7 16-Lane + Gfx10 Detile)", ctx.adapter_name);

    let max_chunks = 2048usize;
    let mut gpu_dec = BcpackGpuDecoder::new(
        &mut ctx,
        &window,
        &quads,
        max_chunks,
        512 * NUM_BLOCKS,
        96 * 1024 * 1024,
    );

    bench_suite(
        "1) BC1 Monolithic 16K-block chunks (Async Compute)",
        &[64, 256, 512],
        NUM_BLOCKS,
        TEX_BC1_BYTES,
        |i| assets[i % assets.len()].bcpack_bytes.as_slice(),
        &window,
        &quads,
        &mut ctx,
        &mut gpu_dec,
    );

    bench_suite(
        "2) BC1 Tiled 4K-block strips (Async Compute low-latency streaming)",
        &[256, 1024, 2048],
        TILE_BLOCKS,
        TILE_BC1_BYTES,
        |i| {
            let tex_i = (i / TILE_STRIPS) % assets.len();
            let strip = i % TILE_STRIPS;
            assets[tex_i].bcpack_tiles[strip].as_slice()
        },
        &window,
        &quads,
        &mut ctx,
        &mut gpu_dec,
    );

    // 3) BC7 16-lane rANS (100% Wave32 = 32/32 threads), 2048 BC7 blocks/chunk (= 4096 sym-rows = 32 KiB equiv BC7)
    println!("--- 3) BC7 16-lane rANS (100% Wave32 = 32/32 active lanes, 2048 BC7 blocks / 32 KiB equiv per chunk) ---");
    let bc7_blocks_per_chunk = 2048usize;
    let bc7_rows_per_chunk = bc7_blocks_per_chunk * 2; // 4096
    let bc7_pool: Vec<Vec<u8>> = (0..8)
        .map(|i| {
            let prof = match i % 3 {
                0 => Bc7EntropyProfile::StandardAlbedo,
                1 => Bc7EntropyProfile::NormalHighEntropy,
                _ => Bc7EntropyProfile::SharpUi,
            };
            generate_bc7_16lane_stream(900 + i as u64, bc7_blocks_per_chunk, prof).0
        })
        .collect();
    let mut bc7_dec = Bc7RansGpuDecoder::new(&mut ctx, 1024, 1024 * bc7_rows_per_chunk, 48 * 1024 * 1024);
    for &n_chunks in &[128usize, 512, 1024] {
        // Equivalent BC7 output = bc7_blocks * 16 bytes = 32 KiB/chunk; raw symbol-rows = 4096 * 32B = 128 KiB/chunk
        let equiv_bc7_mb = (n_chunks * bc7_blocks_per_chunk * 16) as f64 / (1024.0 * 1024.0);

        let t1 = Instant::now();
        let mut sink = 0u16;
        for i in 0..n_chunks {
            let out = rans::decode_slice(&bc7_pool[i % bc7_pool.len()], &BC7_ALPHABET_BITS, bc7_rows_per_chunk).unwrap();
            sink ^= out.lane(0)[0];
        }
        let cpu1_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let cpu1_mbs = equiv_bc7_mb / (cpu1_ms / 1000.0);

        let t2 = Instant::now();
        let xor_s: u16 = (0..n_chunks)
            .into_par_iter()
            .map(|i| {
                let out = rans::decode_slice(&bc7_pool[i % bc7_pool.len()], &BC7_ALPHABET_BITS, bc7_rows_per_chunk).unwrap();
                out.lane(0)[0]
            })
            .reduce(|| 0u16, |a, b| a ^ b);
        let cpu_mc_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let cpu_mc_mbs = equiv_bc7_mb / (cpu_mc_ms / 1000.0);
        std::hint::black_box(sink ^ xor_s);

        let batch = pack_batch((0..n_chunks).map(|i| (bc7_pool[i % bc7_pool.len()].as_slice(), bc7_rows_per_chunk)));
        let _ = bc7_dec.bench_gpu_kernel_only(&mut ctx, &batch);
        let mut ks = 0.0;
        let mut ws = 0.0;
        for _ in 0..5 {
            let (k, w) = bc7_dec.bench_gpu_kernel_only(&mut ctx, &batch);
            ks += k;
            ws += w;
        }
        let gk = ks / 5.0;
        let gw = ws / 5.0;
        let gk_mbs = equiv_bc7_mb / (gk / 1000.0);
        let gw_mbs = equiv_bc7_mb / (gw / 1000.0);
        println!(
            "[bench] {:>4} chunks ({:>5.1} MiB BC7 eq) | CPU-1T: {:>6.2} ms ({:>5.1} MB/s) | CPU-MT: {:>6.2} ms ({:>5.1} MB/s) | GPU 32/32: {:>5.3} ms ({:>7.1} MB/s, {:.1}x vs 1T, {:.1}x vs MT) | GPU+upload: {:>5.2} ms ({:>6.1} MB/s)",
            n_chunks, equiv_bc7_mb, cpu1_ms, cpu1_mbs, cpu_mc_ms, cpu_mc_mbs, gk, gk_mbs, gk_mbs / cpu1_mbs, gk_mbs / cpu_mc_mbs, gw, gw_mbs
        );
    }

    // 4) AMD Gfx10 SW_64KB_D Detile bandwidth benchmark (512 x 512x512 BC1 = 64 MiB VRAM detile)
    let detile = Gfx10DetileGpu::new(&ctx, 64 * 1024 * 1024);
    let _ = detile.bench_detile_batch_kernel_ms(&mut ctx, TEX_BC1_BYTES, 128, 128, 0, 8, 512);
    let mut det_sum = 0.0;
    for _ in 0..5 {
        det_sum += detile.bench_detile_batch_kernel_ms(&mut ctx, TEX_BC1_BYTES, 128, 128, 0, 8, 512);
    }
    let det_ms = det_sum / 5.0;
    let det_gbps = (64.0 / 1024.0) / (det_ms / 1000.0);
    println!(
        "--- 4) AMD Gfx10 Swizzle -> Linear Detile Pass ---\n[bench] 512x 512x512 SW_64KB_D (64.0 MiB) | GPU Detile Kernel: {:.3} ms ({:.1} GB/s effective)",
        det_ms, det_gbps
    );

    // 5) Console High-Entropy Stress Profiles (1024 x 4K-tiles = 32.0 MiB BC1 each)
    println!("--- 5) High-Entropy Stress Profiles (1024 x 4K-tiles = 32.0 MiB BC1) ---");
    let normal_tiles: Vec<Vec<u8>> = (0..8)
        .map(|s| generate_high_entropy_bc1_normal_map(TILE_BLOCKS, 400 + s as u64, &window, &quads).0)
        .collect();
    let ui_tiles: Vec<Vec<u8>> = (0..8)
        .map(|s| generate_sharp_ui_bc1(TILE_BLOCKS, 800 + s as u64, &window, &quads).0)
        .collect();
    for (pname, pool) in [
        ("Standard Highway Albedo", (0..8).map(|i| assets[i].bcpack_tiles[0].clone()).collect::<Vec<_>>()),
        ("High-Entropy Normal-Map Noise", normal_tiles),
        ("Sharp Contrasting UI / HUD Grid", ui_tiles),
    ] {
        let avg_comp: f64 = pool.iter().map(|v| v.len() as f64).sum::<f64>() / pool.len() as f64;
        let ratio = TILE_BC1_BYTES as f64 / avg_comp;
        let batch = pack_batch((0..1024).map(|i| (pool[i % pool.len()].as_slice(), TILE_BLOCKS)));
        let _ = gpu_dec.bench_gpu_kernel_only(&mut ctx, &batch);
        let mut ks = 0.0;
        for _ in 0..5 {
            ks += gpu_dec.bench_gpu_kernel_only(&mut ctx, &batch).0;
        }
        let kms = ks / 5.0;
        let mbs = 32.0 / (kms / 1000.0);
        println!(
            "[bench] {:<32} | Ratio: {:>4.2}x ({:>5.0} B/tile) | GPU Async: {:>5.3} ms ({:>7.1} MB/s)",
            pname, ratio, avg_comp, kms, mbs
        );
    }

    // 6) Out-of-Order Disk Container Streaming Benchmark (16 randomized 64 MiB bursts = 1.0 GiB uncompressed BC1)
    let pkg_path = assets_dir.join("stream_bench.bcpack_pkg");
    let mut tile_index: Vec<(u64, u32)> = Vec::with_capacity(2048);
    {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&pkg_path)
            .expect("create stream_bench.bcpack_pkg");
        let mut off = 0u64;
        for i in 0..2048usize {
            let tex_i = (i / TILE_STRIPS) % assets.len();
            let strip = i % TILE_STRIPS;
            let bytes = &assets[tex_i].bcpack_tiles[strip];
            f.write_all(bytes).unwrap();
            tile_index.push((off, bytes.len() as u32));
            off += bytes.len() as u64;
        }
        f.flush().unwrap();
    }
    let pkg_size_mb = tile_index.iter().map(|t| t.1 as u64).sum::<u64>() as f64 / (1024.0 * 1024.0);
    let mut pkg_file = File::open(&pkg_path).expect("open pkg for out-of-order reads");

    // Pre-build 16 deterministic out-of-order tile schedules across 2,048 entries
    let mut total_disk_ms = 0.0f64;
    let mut total_gpu_wall_ms = 0.0f64;
    let mut total_kernel_ms = 0.0f64;
    let bursts = 16usize;
    let tiles_per_burst = 2048usize; // 64 MiB uncompressed per burst -> 1.0 GiB across 16 bursts
    let mut read_scratch: Vec<Vec<u8>> = (0..tiles_per_burst).map(|_| Vec::with_capacity(32768)).collect();

    for burst in 0..bursts {
        // Stride permutation (coprime 137) so disk seeks jump non-sequentially across container
        let td = Instant::now();
        for step in 0..tiles_per_burst {
            let idx = (step * 137 + burst * 53) % tile_index.len();
            let (off, len) = tile_index[idx];
            let buf = &mut read_scratch[step];
            buf.resize(len as usize, 0);
            pkg_file.seek(SeekFrom::Start(off)).unwrap();
            pkg_file.read_exact(buf).unwrap();
        }
        total_disk_ms += td.elapsed().as_secs_f64() * 1000.0;

        let batch = pack_batch(read_scratch.iter().map(|b| (b.as_slice(), TILE_BLOCKS)));
        let (kms, wms) = gpu_dec.bench_gpu_kernel_only(&mut ctx, &batch);
        total_kernel_ms += kms;
        total_gpu_wall_ms += wms;
    }
    drop(pkg_file);
    let _ = std::fs::remove_file(&pkg_path);

    let total_uncomp_gb = (bursts * tiles_per_burst * TILE_BC1_BYTES) as f64 / (1024.0 * 1024.0 * 1024.0);
    let e2e_ms = total_disk_ms + total_gpu_wall_ms;
    println!(
        "--- 6) Out-of-Order Disk Container Streaming Benchmark (1.0 GiB Uncompressed Burst) ---\n[bench] Container ({pkg_size_mb:.1} MiB packed, 16x 2048 OOO seeks = {total_uncomp_gb:.2} GiB BC1) | Disk Read: {total_disk_ms:.2} ms | Upload+AsyncGPU+Fence: {total_gpu_wall_ms:.2} ms (Kernel={total_kernel_ms:.2} ms) | End-to-End Sustained: {:.2} GB/s",
        total_uncomp_gb / (e2e_ms / 1000.0)
    );

    // 7) BCPack 8B Quantized Meshlet Geometry Stream Benchmark (1024 x 4096-vertex chunks = 32.0 MiB quantized geometry)
    let lib = build_city_meshlet_library();
    let raw_geom = pack_quantized_meshlet_vertices(&lib.vertices, 4096 * 8);
    let mut geom_chunks = Vec::with_capacity(8);
    let mut geom_comp_sum = 0usize;
    for c in 0..8 {
        let enc = encode_bc1_to_bcpack(&raw_geom[c * TILE_BC1_BYTES..(c + 1) * TILE_BC1_BYTES], &window, &quads);
        geom_comp_sum += enc.len();
        geom_chunks.push(enc);
    }
    let geom_ratio = (8 * TILE_BC1_BYTES) as f64 / geom_comp_sum as f64;
    let geom_batch = pack_batch((0..1024).map(|i| (geom_chunks[i % geom_chunks.len()].as_slice(), TILE_BLOCKS)));
    let _ = gpu_dec.bench_gpu_kernel_only(&mut ctx, &geom_batch);
    let mut gk_sum = 0.0;
    for _ in 0..5 {
        gk_sum += gpu_dec.bench_gpu_kernel_only(&mut ctx, &geom_batch).0;
    }
    let gk_ms = gk_sum / 5.0;
    let gk_mbs = 32.0 / (gk_ms / 1000.0);
    println!(
        "--- 7) BCPack 8B Quantized Meshlet Geometry (4.19M Vertices = 32.0 MiB) ---\n[bench] 8B UNORM16+OctN+UV | Ratio: {:.2}x | Async GPU Decode: {:.3} ms ({:.1} MB/s)",
        geom_ratio, gk_ms, gk_mbs
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_verify = args.iter().any(|a| a == "--verify");
    let do_bench = args.iter().any(|a| a == "--bench");
    let do_demo = args.iter().any(|a| a == "--demo");
    let do_jet_flight = args.iter().any(|a| a == "--jet-flight");
    let regen_assets = args.iter().any(|a| a == "--regen-assets");
    let enable_debug = args.iter().any(|a| a == "--debug");
    let start_rt_enabled = !args.iter().any(|a| a == "--no-rt");
    let start_rtgi_enabled = start_rt_enabled && !args.iter().any(|a| a == "--no-rtgi");
    let start_world_stream = args.iter().any(|a| a == "--world-stream");
    let start_mesh_shaders = !args.iter().any(|a| a == "--no-mesh-shaders");
    let render_1440p = args.iter().any(|a| a == "--1440p");
    let nvme_stream = args.iter().any(|a| a == "--nvme-stream");
    let mut vram_cap_mb = 32usize;

    let mut screenshot: Option<PathBuf> = None;
    let mut frames = 20u32;
    let mut record_dir: Option<PathBuf> = None;
    let mut record_frames = 60u32;
    let mut chunks = 256usize;
    let mut measure_frames: Option<u32> = None;
    let mut assets_dir = PathBuf::from("assets");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--screenshot" && i + 1 < args.len() {
            screenshot = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else if args[i] == "--frames" && i + 1 < args.len() {
            frames = args[i + 1].parse().unwrap_or(20);
            i += 2;
        } else if args[i] == "--record-dir" && i + 1 < args.len() {
            record_dir = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else if args[i] == "--record-frames" && i + 1 < args.len() {
            record_frames = args[i + 1].parse().unwrap_or(60);
            i += 2;
        } else if args[i] == "--chunks" && i + 1 < args.len() {
            chunks = args[i + 1].parse().unwrap_or(256);
            i += 2;
        } else if args[i] == "--vram-cap-mb" && i + 1 < args.len() {
            vram_cap_mb = args[i + 1].parse().unwrap_or(32);
            i += 2;
        } else if args[i] == "--measure-frames" && i + 1 < args.len() {
            measure_frames = Some(args[i + 1].parse().unwrap_or(20));
            i += 2;
        } else if args[i] == "--assets-dir" && i + 1 < args.len() {
            assets_dir = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    if !do_verify && !do_bench && !do_demo && !do_jet_flight {
        println!("Usage: bcpack-gpu-demo [--verify] [--bench] [--jet-flight] [--demo] [--chunks N] [--world-stream] [--vram-cap-mb N] [--measure-frames N] [--assets-dir <dir>] [--screenshot <out.png> --frames <N>] [--record-dir <dir> --record-frames <N>] [--regen-assets]");
        return;
    }

    if do_verify {
        run_verify(&assets_dir, regen_assets);
    }
    if do_jet_flight {
        run_jet_flight_suite(&assets_dir);
    }
    if do_bench {
        run_bench(&assets_dir);
    }
    if do_demo {
        let window = Window::default_table();
        let quads = QuadTable::default_table();
        let assets = load_or_generate_assets(&assets_dir, &window, &quads, regen_assets);
        run_demo(
            &assets,
            &window,
            &quads,
            DemoConfig {
                mode: DemoMode::Toggleable,
                initial_chunks: chunks,
                multi_core_cpu: true,
                enable_debug,
                start_rt_enabled,
                start_rtgi_enabled,
                start_world_stream,
                start_mesh_shaders,
                render_1440p,
                nvme_stream,
                assets_dir: assets_dir.clone(),
                vram_cap_mb,
                screenshot_path: screenshot,
                screenshot_frames: frames,
                measure_frames,
                record_dir,
                record_frames,
            },
        );
    }
}
