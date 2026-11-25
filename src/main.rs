use bcpack::bc1::{self, QuadTable, Window};
use bcpack::encode::synthetic_bc1;
use bcpack_gpu_demo::assets::{
    load_or_generate_assets, NUM_BLOCKS, TEX_BC1_BYTES, TILE_BC1_BYTES, TILE_BLOCKS, TILE_STRIPS,
};
use bcpack_gpu_demo::demo_scene::{run_demo, DemoConfig, DemoMode};
use bcpack_gpu_demo::gpu_decoder::{pack_batch, BcpackGpuDecoder, GpuContext};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

fn run_verify(assets_dir: &PathBuf, force_regen: bool) {
    let window = Window::default_table();
    let quads = QuadTable::default_table();
    let assets = load_or_generate_assets(assets_dir, &window, &quads, force_regen);

    let mut ctx = GpuContext::new(true);
    println!("[verify] Adapter: {}", ctx.adapter_name);

    // 1) Synthetic vector tests (5 chunks = odd count, testing dual-chunk padding + prestart refs)
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
        .max(assets.len() * NUM_BLOCKS);
    let max_chunks = cases.len().max(total_tiles);
    let mut gpu_dec = BcpackGpuDecoder::new(&mut ctx, &window, &quads, max_chunks, max_blocks, 16 * 1024 * 1024);

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
        "[verify] PASS synthetic streams ({} chunks, {} total blocks, GPU={:.3} ms, 0 byte mismatches)",
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
        "[verify] PASS all {} monolithic 512x512 .bcpack textures (1.50 MiB BC1, GPU={:.3} ms, 0 byte mismatches)",
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
        "[verify] PASS all {} tiled 4K-block .bcpack strips (1.50 MiB BC1, GPU={:.3} ms, 0 byte mismatches)",
        total_tiles,
        tile_gpu_ms
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
            "[bench] {:>4} chunks ({:>5.1} MiB) | CPU-1T: {:>6.2} ms ({:>5.1} MB/s) | CPU-MT: {:>6.2} ms ({:>5.1} MB/s) | GPU: {:>5.3} ms ({:>7.1} MB/s, {:.1}x vs 1T, {:.1}x vs MT) | GPU+upload: {:>5.2} ms ({:>6.1} MB/s)",
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
    println!("[bench] Adapter: {} (v0.2 Dual-Chunk Wave32 + Coalesced Cache)", ctx.adapter_name);

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
        "Monolithic 16K-block chunks (apples-to-apples vs v0.1)",
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
        "Tiled 4K-block strips (low-latency Xbox Velocity streaming)",
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
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_verify = args.iter().any(|a| a == "--verify");
    let do_bench = args.iter().any(|a| a == "--bench");
    let do_demo = args.iter().any(|a| a == "--demo");
    let regen_assets = args.iter().any(|a| a == "--regen-assets");

    let mut screenshot: Option<PathBuf> = None;
    let mut frames = 20u32;
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
        } else if args[i] == "--chunks" && i + 1 < args.len() {
            chunks = args[i + 1].parse().unwrap_or(256);
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

    if !do_verify && !do_bench && !do_demo {
        println!("Usage: bcpack-gpu-demo [--verify] [--bench] [--demo] [--chunks N] [--measure-frames N] [--assets-dir <dir>] [--screenshot <out.png> --frames <N>] [--regen-assets]");
        return;
    }

    if do_verify {
        run_verify(&assets_dir, regen_assets);
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
                screenshot_path: screenshot,
                screenshot_frames: frames,
                measure_frames,
            },
        );
    }
}
