use bcpack::bc1::{QuadTable, Window};
use bcpack_gpu_demo::assets::load_or_generate_assets;
use bcpack_gpu_demo::demo_scene::{run_demo, DemoConfig, DemoMode};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut chunks = 64usize;
    let mut multi_core = false;
    let mut measure_frames: Option<u32> = None;
    let mut assets_dir = PathBuf::from("assets");

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--mt" {
            multi_core = true;
            i += 1;
        } else if args[i] == "--chunks" && i + 1 < args.len() {
            chunks = args[i + 1].parse().unwrap_or(64);
            i += 2;
        } else if args[i] == "--measure-frames" && i + 1 < args.len() {
            measure_frames = Some(args[i + 1].parse().unwrap_or(10));
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
            mode: DemoMode::CpuOnly,
            initial_chunks: chunks,
            multi_core_cpu: multi_core,
            screenshot_path: None,
            screenshot_frames: 20,
            measure_frames,
        },
    );
}
