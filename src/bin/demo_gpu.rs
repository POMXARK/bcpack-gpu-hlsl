use bcpack::bc1::{QuadTable, Window};
use bcpack_gpu::assets::load_or_generate_assets;
use bcpack_gpu::demo_scene::{run_demo, DemoConfig, DemoMode};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut chunks = 256usize;
    let enable_debug = args.iter().any(|a| a == "--debug");
    let start_rt_enabled = !args.iter().any(|a| a == "--no-rt");
    let start_rtgi_enabled = start_rt_enabled && !args.iter().any(|a| a == "--no-rtgi");
    let start_world_stream = args.iter().any(|a| a == "--world-stream");
    let start_mesh_shaders = !args.iter().any(|a| a == "--no-mesh-shaders");
    let render_1440p = args.iter().any(|a| a == "--1440p");
    let nvme_stream = args.iter().any(|a| a == "--nvme-stream");
    let mut vram_cap_mb = 32usize;
    let mut measure_frames: Option<u32> = None;
    let mut assets_dir = PathBuf::from("assets");

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--chunks" && i + 1 < args.len() {
            chunks = args[i + 1].parse().unwrap_or(256);
            i += 2;
        } else if args[i] == "--vram-cap-mb" && i + 1 < args.len() {
            vram_cap_mb = args[i + 1].parse().unwrap_or(32);
            i += 2;
        } else if args[i] == "--measure-frames" && i + 1 < args.len() {
            measure_frames = Some(args[i + 1].parse().unwrap_or(30));
            i += 2;
        } else if args[i] == "--assets-dir" && i + 1 < args.len() {
            assets_dir = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    let window = Window::default_table();
    let quads = QuadTable::default_table();
    let assets = load_or_generate_assets(&assets_dir, &window, &quads, false);

    run_demo(
        &assets,
        &window,
        &quads,
        DemoConfig {
            mode: DemoMode::GpuOnly,
            initial_chunks: chunks,
            multi_core_cpu: false,
            enable_debug,
            start_rt_enabled,
            start_rtgi_enabled,
            start_world_stream,
            start_mesh_shaders,
            render_1440p,
            nvme_stream,
            assets_dir: assets_dir.clone(),
            vram_cap_mb,
            screenshot_path: None,
            screenshot_frames: 20,
            measure_frames,
            record_dir: None,
            record_frames: 0,
        },
    );
}
