use bcpack_gpu::aaa_scene::{run_aaa_demo, AaaDemoConfig};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = AaaDemoConfig::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--aaa-dir" if i + 1 < args.len() => {
                cfg.aaa_dir = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--1440p" => cfg.render_1440p = true,
            "--cpu" => cfg.start_cpu = true,
            "--flush-bench" => cfg.flush_bench = true,
            "--debug" => cfg.enable_debug = true,
            "--xray" => cfg.xray_mode = true,
            "--vsync" => cfg.vsync = true,
            "--no-vsync" => cfg.vsync = false,
            "--no-rt" => cfg.no_rt = true,
            "--no-rtgi" => cfg.no_rtgi = true,
            "--tiles-per-frame" if i + 1 < args.len() => {
                cfg.tiles_per_frame = args[i + 1].parse().unwrap_or(256);
                i += 1;
            }
            "--measure-frames" if i + 1 < args.len() => {
                cfg.measure_frames = Some(args[i + 1].parse().unwrap_or(60));
                i += 1;
            }
            "--flush-at" if i + 1 < args.len() => {
                cfg.flush_at_frame = Some(args[i + 1].parse().unwrap_or(10));
                i += 1;
            }
            "--screenshot" if i + 1 < args.len() => {
                cfg.screenshot_path = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--screenshot-frames" if i + 1 < args.len() => {
                cfg.screenshot_frames = args[i + 1].parse().unwrap_or(24);
                i += 1;
            }
            "--vram-budget-mb" if i + 1 < args.len() => {
                cfg.vram_budget_mb = args[i + 1].parse().unwrap_or(128);
                i += 1;
            }
            other => {
                eprintln!("Unknown flag '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    run_aaa_demo(cfg);
}
