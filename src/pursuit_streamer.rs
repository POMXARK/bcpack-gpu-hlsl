//! Bundle 1: Async-Threaded Zero-Copy 6-Format (`BC1`, `BC3`, `BC4`, `BC5`, `BC6H`, `BC7` + `GFX10 SW_64KB_D`)
//! Pursuit Streaming + Live `VirtualDescriptorPager` (`1.25M -> 262K`) integration for `bcpack-aaa-demo`.
//!
//! All `BcPackDecodeBatchToTextures` calls run on a dedicated background thread (`pursuit-zc-streamer`),
//! so the main render thread never blocks on CPU waits and only issues pure GPU-GPU
//! `render_queue.Wait(&direct_fence, latest_fence)` synchronization.

use bcpack::bc7 as bcp_bc7;
use bcpack::encode::{encode_lanes, Fragment, Rng};
use bcpack::format::Mode as BcpMode;
use bcpack_gpu::descriptor_paging::{
    VirtualDescriptorPager, PHYS_HEAP_SLOTS, VIRTUAL_DESC_COUNT,
};
use bcpack_gpu::ffi::{
    bcpack_gpu_create_from_device, bcpack_gpu_destroy, BcPackContext, BcPackDecodeBatchToTextures,
    BcPackGetDirectFence, BcPackTextureBatchItem, BCPACK_FORMAT_BC1, BCPACK_FORMAT_BC3,
    BCPACK_FORMAT_BC4, BCPACK_FORMAT_BC5, BCPACK_FORMAT_BC6H, BCPACK_FORMAT_BC7,
    BCPACK_SWIZZLE_GFX10_64KB_D, BCPACK_SWIZZLE_LINEAR,
};
use bcpack_gpu::gpu_decoder::{heap_props, GpuContext};
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

const POOL_PER_FORMAT: usize = 6;
const POOL_DIM_PX: u32 = 256;
const POOL_BLOCKS_SIDE: u32 = POOL_DIM_PX / 4; // 64
const POOL_BLOCKS: u32 = POOL_BLOCKS_SIDE * POOL_BLOCKS_SIDE; // 4096

#[derive(Clone, Copy)]
struct FormatSpec {
    mode: BcpMode,
    ffi_fmt: u32,
    dxgi_fmt: DXGI_FORMAT,
    raw_bytes: usize,
    name: &'static str,
}

const FORMAT_SPECS: [FormatSpec; 6] = [
    FormatSpec {
        mode: BcpMode::Bc1,
        ffi_fmt: BCPACK_FORMAT_BC1,
        dxgi_fmt: DXGI_FORMAT_BC1_UNORM,
        raw_bytes: POOL_BLOCKS as usize * 8,
        name: "BC1",
    },
    FormatSpec {
        mode: BcpMode::Bc3,
        ffi_fmt: BCPACK_FORMAT_BC3,
        dxgi_fmt: DXGI_FORMAT_BC3_UNORM,
        raw_bytes: POOL_BLOCKS as usize * 16,
        name: "BC3",
    },
    FormatSpec {
        mode: BcpMode::Bc4,
        ffi_fmt: BCPACK_FORMAT_BC4,
        dxgi_fmt: DXGI_FORMAT_BC4_UNORM,
        raw_bytes: POOL_BLOCKS as usize * 8,
        name: "BC4",
    },
    FormatSpec {
        mode: BcpMode::Bc5,
        ffi_fmt: BCPACK_FORMAT_BC5,
        dxgi_fmt: DXGI_FORMAT_BC5_UNORM,
        raw_bytes: POOL_BLOCKS as usize * 16,
        name: "BC5",
    },
    FormatSpec {
        mode: BcpMode::Bc6h,
        ffi_fmt: BCPACK_FORMAT_BC6H,
        dxgi_fmt: DXGI_FORMAT_BC6H_UF16,
        raw_bytes: POOL_BLOCKS as usize * 16,
        name: "BC6H",
    },
    FormatSpec {
        mode: BcpMode::Bc7,
        ffi_fmt: BCPACK_FORMAT_BC7,
        dxgi_fmt: DXGI_FORMAT_BC7_UNORM,
        raw_bytes: POOL_BLOCKS as usize * 16,
        name: "BC7",
    },
];

fn valid_bc7_ctrl_codes() -> Vec<u16> {
    let t7 = bcp_bc7::Tables::default_table();
    let codes: Vec<u16> = bcp_bc7::Control::all()
        .into_iter()
        .filter_map(|c| t7.code_of(c).map(|idx| idx as u16))
        .collect();
    assert!(!codes.is_empty());
    codes
}

fn encode_cleanroom_chunk(mode: BcpMode, blocks: usize, seed: u64, ctrl7: &[u16]) -> Vec<u8> {
    let ab = mode.alphabet_bits();
    let rows = blocks * mode.symbols_per_block();
    let mut rng = Rng(seed | 1);
    let mut ctrl_for_block = Vec::with_capacity(blocks);
    for i in 0..blocks {
        ctrl_for_block.push(ctrl7[(i + (seed as usize)) % ctrl7.len()]);
    }

    let mut lanes: Vec<Vec<Fragment>> = Vec::with_capacity(ab.len());
    for (lane_idx, &bits) in ab.iter().enumerate() {
        let lim = 1u64 << bits;
        let mut syms = Vec::with_capacity(rows);
        for r in 0..rows {
            let s = if lane_idx < 2 {
                0u16
            } else if mode == BcpMode::Bc7 && lane_idx == 15 {
                let c = ctrl_for_block[r / 2];
                if r % 2 == 0 { c & 31 } else { (c >> 5) & 15 }
            } else if mode == BcpMode::Bc6h && lane_idx == 15 {
                if r % 2 == 0 {
                    ((r / 2) % 14) as u16
                } else {
                    (rng.next_u64() & 15) as u16
                }
            } else {
                rng.below(lim) as u16
            };
            syms.push(s);
        }
        let frags: Vec<Fragment> = syms
            .chunks(4096)
            .map(|c| Fragment {
                symbols: c.to_vec(),
                q: 0,
                z: 0,
            })
            .collect();
        lanes.push(frags);
    }
    encode_lanes(ab, &lanes).expect("encode_lanes failed")
}

/// Exclusive ownership wrapper to move `*mut BcPackContext` onto the single `pursuit-zc-streamer` thread.
struct SendSdkCtx(*mut BcPackContext);
// SAFETY: `BcPackContext` is accessed exclusively by the single background streamer thread until Drop.
unsafe impl Send for SendSdkCtx {}

impl Drop for SendSdkCtx {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { bcpack_gpu_destroy(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }
}

/// Exclusive ownership wrapper for a target VRAM texture ring owned by the background streamer thread.
struct SendTexPool(Vec<[ID3D12Resource; 6]>);
// SAFETY: `ID3D12Resource` COM objects are free-threaded; accessed only by the streamer thread.
unsafe impl Send for SendTexPool {}

struct SharedStreamerState {
    enabled: AtomicBool,
    whip_active: AtomicBool,
    quit: AtomicBool,
    latest_direct_fence: AtomicU64,
    total_batches: AtomicU64,
    total_textures: AtomicU64,
    total_bytes_vram: AtomicU64,
    gfx10_detiled_count: AtomicU64,
    format_counts: [AtomicU64; 6],
}

impl SharedStreamerState {
    fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            whip_active: AtomicBool::new(false),
            quit: AtomicBool::new(false),
            latest_direct_fence: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
            total_textures: AtomicU64::new(0),
            total_bytes_vram: AtomicU64::new(0),
            gfx10_detiled_count: AtomicU64::new(0),
            format_counts: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct PursuitTelemetry {
    pub throughput_mbs: f64,
    pub peak_throughput_mbs: f64,
    pub total_batches: u64,
    pub total_textures: u64,
    pub format_counts: [u64; 6],
    pub total_bytes_vram: u64,
    pub gfx10_detiled_count: u64,
    pub pager_active_slots: u32,
    pub pager_evictions_total: u64,
    pub pager_evictions_per_sec: f64,
    pub whip_pan_count: u64,
    pub whip_pan_pulse: f32,
    pub fps_1pct_low: f32,
    pub last_direct_fence: u64,
}

pub struct PursuitStreamer {
    direct_fence: ID3D12Fence,
    shared: Arc<SharedStreamerState>,
    worker: Option<JoinHandle<()>>,
    pager: VirtualDescriptorPager,
    virt_cursor: usize,
    waited_fence: u64,
    prev_total_bytes: u64,
    frame_times_ms: VecDeque<f64>,
    window_samples: VecDeque<(f64, u64, u64)>, // (elapsed_s, bytes_added, evictions_added)
    start_time: Instant,
    prev_whip_active: bool,
    pub telemetry: PursuitTelemetry,
}

impl PursuitStreamer {
    pub fn new(ctx: &mut GpuContext) -> Self {
        unsafe {
            let sdk_ctx = bcpack_gpu_create_from_device(
                ctx.device.as_raw(),
                std::ptr::null_mut(),
                64,
                524_288,
                16 * 1024 * 1024,
            );
            assert!(
                !sdk_ctx.is_null(),
                "bcpack_gpu_create_from_device failed on primary ID3D12Device"
            );

            let fence_raw = BcPackGetDirectFence(sdk_ctx);
            assert!(!fence_raw.is_null(), "BcPackGetDirectFence returned null");
            let direct_fence = (*ManuallyDrop::new(ID3D12Fence::from_raw(fence_raw))).clone();

            let mut pager = VirtualDescriptorPager::new(ctx);
            // Warm physical descriptor heap to full 262,144 occupancy once at startup so runtime
            // per-frame updates only touch the visible window while immediately exercising real LRU evictions.
            let mut virt_cursor = 0usize;
            for _ in 0..PHYS_HEAP_SLOTS {
                let vid = (virt_cursor % VIRTUAL_DESC_COUNT) as u32;
                let _ = pager.page_in(&ctx.device, vid);
                virt_cursor = (virt_cursor + 1) % VIRTUAL_DESC_COUNT;
            }

            let ctrl7 = valid_bc7_ctrl_codes();
            let mut tex_pool = Vec::with_capacity(POOL_PER_FORMAT);
            let mut streams = Vec::with_capacity(POOL_PER_FORMAT);

            for p in 0..POOL_PER_FORMAT {
                let mut arr_res: [Option<ID3D12Resource>; 6] = [None, None, None, None, None, None];
                let mut arr_str: [Vec<u8>; 6] = Default::default();
                for (fi, spec) in FORMAT_SPECS.iter().enumerate() {
                    let desc = D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: POOL_DIM_PX as u64,
                        Height: POOL_DIM_PX,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: spec.dxgi_fmt,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                        Flags: D3D12_RESOURCE_FLAG_NONE,
                        ..Default::default()
                    };
                    let mut tex: Option<ID3D12Resource> = None;
                    ctx.device
                        .CreateCommittedResource(
                            &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COMMON,
                            None,
                            &mut tex,
                        )
                        .unwrap_or_else(|e| {
                            panic!(
                                "Pursuit tex CreateCommittedResource {} failed: {e:?}",
                                spec.name
                            )
                        });
                    arr_res[fi] = tex;

                    let seed = 0x51A7_0000u64 + (p as u64) * 17 + (fi as u64);
                    arr_str[fi] =
                        encode_cleanroom_chunk(spec.mode, POOL_BLOCKS as usize, seed, &ctrl7);
                }
                tex_pool.push([
                    arr_res[0].take().unwrap(),
                    arr_res[1].take().unwrap(),
                    arr_res[2].take().unwrap(),
                    arr_res[3].take().unwrap(),
                    arr_res[4].take().unwrap(),
                    arr_res[5].take().unwrap(),
                ]);
                streams.push(arr_str);
            }

            let shared = Arc::new(SharedStreamerState::new());
            let worker_shared = Arc::clone(&shared);
            let send_ctx = SendSdkCtx(sdk_ctx);
            let send_pool = SendTexPool(tex_pool);

            let worker = thread::Builder::new()
                .name("pursuit-zc-streamer".to_string())
                .spawn(move || {
                    let mut batch_seq = 0usize;
                    let mut items: Vec<BcPackTextureBatchItem> =
                        Vec::with_capacity(POOL_PER_FORMAT * 6);
                    while !worker_shared.quit.load(Ordering::Acquire) {
                        if !worker_shared.enabled.load(Ordering::Acquire) {
                            thread::sleep(Duration::from_millis(2));
                            continue;
                        }
                        let whip = worker_shared.whip_active.load(Ordering::Relaxed);
                        let copies_per_fmt = if whip { POOL_PER_FORMAT } else { 2 };
                        items.clear();
                        let mut batch_bytes = 0u64;
                        let mut batch_detiled = 0u64;

                        for p in 0..copies_per_fmt {
                            let pool_slot = (batch_seq + p) % POOL_PER_FORMAT;
                            for (fi, spec) in FORMAT_SPECS.iter().enumerate() {
                                let swizzle = if ((p + fi + batch_seq) & 1) == 0 {
                                    batch_detiled += 1;
                                    BCPACK_SWIZZLE_GFX10_64KB_D
                                } else {
                                    BCPACK_SWIZZLE_LINEAR
                                };
                                let stream = &streams[pool_slot][fi];
                                let dst = &send_pool.0[pool_slot][fi];
                                items.push(BcPackTextureBatchItem {
                                    compressed_data: stream.as_ptr(),
                                    compressed_size: stream.len() as u32,
                                    num_blocks: POOL_BLOCKS,
                                    format: spec.ffi_fmt,
                                    swizzle_mode: swizzle,
                                    width_blocks: POOL_BLOCKS_SIDE,
                                    height_blocks: POOL_BLOCKS_SIDE,
                                    dst_texture2d: dst.as_raw(),
                                    subresource_index: 0,
                                    dst_x_blocks: 0,
                                    dst_y_blocks: 0,
                                    status: -1,
                                });
                                batch_bytes += spec.raw_bytes as u64;
                            }
                        }

                        let mut fence_val = 0u64;
                        let ok_count = BcPackDecodeBatchToTextures(
                            send_ctx.0,
                            items.as_mut_ptr(),
                            items.len() as u32,
                            &mut fence_val,
                        );
                        if ok_count > 0 {
                            worker_shared.total_batches.fetch_add(1, Ordering::Relaxed);
                            worker_shared
                                .total_textures
                                .fetch_add(ok_count as u64, Ordering::Relaxed);
                            worker_shared
                                .total_bytes_vram
                                .fetch_add(batch_bytes, Ordering::Relaxed);
                            worker_shared
                                .gfx10_detiled_count
                                .fetch_add(batch_detiled, Ordering::Relaxed);
                            for fi in 0..6 {
                                worker_shared.format_counts[fi]
                                    .fetch_add(copies_per_fmt as u64, Ordering::Relaxed);
                            }
                            if fence_val > 0 {
                                worker_shared
                                    .latest_direct_fence
                                    .store(fence_val, Ordering::Release);
                            }
                        }
                        batch_seq = batch_seq.wrapping_add(1);
                    }
                    // `send_ctx` drops cleanly here (waiting for in-flight SDK work before destroying)
                    drop(send_ctx);
                    drop(send_pool);
                })
                .ok();

            Self {
                direct_fence,
                shared,
                worker,
                pager,
                virt_cursor,
                waited_fence: 0,
                prev_total_bytes: 0,
                frame_times_ms: VecDeque::with_capacity(120),
                window_samples: VecDeque::with_capacity(240),
                start_time: Instant::now(),
                prev_whip_active: false,
                telemetry: PursuitTelemetry {
                    pager_active_slots: PHYS_HEAP_SLOTS as u32,
                    ..Default::default()
                },
            }
        }
    }

    pub fn tick_frame(
        &mut self,
        device: &ID3D12Device,
        render_queue: &ID3D12CommandQueue,
        raw_dt_ms: f64,
        whip_pan_active: bool,
        whip_pan_pulse: f32,
        pursuit_enabled: bool,
    ) {
        if raw_dt_ms > 0.01 && raw_dt_ms < 1000.0 {
            if self.frame_times_ms.len() >= 120 {
                self.frame_times_ms.pop_front();
            }
            self.frame_times_ms.push_back(raw_dt_ms);
            let mut sorted: Vec<f64> = self.frame_times_ms.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((sorted.len() as f64) * 0.99).floor() as usize;
            let p99_ms = sorted[idx.min(sorted.len() - 1)].max(0.01);
            self.telemetry.fps_1pct_low = (1000.0 / p99_ms).min(999.0) as f32;
        }

        self.telemetry.whip_pan_pulse = whip_pan_pulse;
        self.shared
            .enabled
            .store(pursuit_enabled, Ordering::Release);
        self.shared
            .whip_active
            .store(whip_pan_active, Ordering::Relaxed);

        if !pursuit_enabled {
            self.prev_whip_active = false;
            return;
        }

        if whip_pan_active && !self.prev_whip_active {
            self.telemetry.whip_pan_count += 1;
        }
        self.prev_whip_active = whip_pan_active;

        // Pure GPU-GPU fence wait on the latest async Zero-Copy batch (0 CPU stall!)
        let latest_fence = self.shared.latest_direct_fence.load(Ordering::Acquire);
        if latest_fence > self.waited_fence {
            unsafe {
                render_queue
                    .Wait(&self.direct_fence, latest_fence)
                    .expect("render_queue.Wait(BcPackGetDirectFence) failed");
            }
            self.waited_fence = latest_fence;
            self.telemetry.last_direct_fence = latest_fence;
        }

        // Visible-window VirtualDescriptorPager update (low per-frame CPU budget, real LRU evictions)
        let page_ops = if whip_pan_active { 512 } else { 128 };
        let ev_before = self.pager.evictions;
        for _ in 0..page_ops {
            let vid = (self.virt_cursor % VIRTUAL_DESC_COUNT) as u32;
            let _ = self.pager.page_in(device, vid);
            self.virt_cursor = (self.virt_cursor + 29) % VIRTUAL_DESC_COUNT;
        }
        let ev_delta = self.pager.evictions.saturating_sub(ev_before);

        let cur_bytes = self.shared.total_bytes_vram.load(Ordering::Relaxed);
        let bytes_delta = cur_bytes.saturating_sub(self.prev_total_bytes);
        self.prev_total_bytes = cur_bytes;

        self.telemetry.total_batches = self.shared.total_batches.load(Ordering::Relaxed);
        self.telemetry.total_textures = self.shared.total_textures.load(Ordering::Relaxed);
        self.telemetry.total_bytes_vram = cur_bytes;
        self.telemetry.gfx10_detiled_count =
            self.shared.gfx10_detiled_count.load(Ordering::Relaxed);
        for fi in 0..6 {
            self.telemetry.format_counts[fi] =
                self.shared.format_counts[fi].load(Ordering::Relaxed);
        }
        self.telemetry.pager_evictions_total = self.pager.evictions;
        self.telemetry.pager_active_slots = PHYS_HEAP_SLOTS as u32;

        let now_s = self.start_time.elapsed().as_secs_f64();
        self.window_samples.push_back((now_s, bytes_delta, ev_delta));
        while let Some(&(ts, _, _)) = self.window_samples.front() {
            if now_s - ts > 0.75 && self.window_samples.len() > 2 {
                self.window_samples.pop_front();
            } else {
                break;
            }
        }
        if let (Some(&(t0, _, _)), Some(&(t1, _, _))) =
            (self.window_samples.front(), self.window_samples.back())
        {
            let span = (t1 - t0).max(0.05);
            let sum_bytes: u64 = self.window_samples.iter().map(|s| s.1).sum();
            let sum_ev: u64 = self.window_samples.iter().map(|s| s.2).sum();
            let mbs = (sum_bytes as f64 / (1024.0 * 1024.0)) / span;
            self.telemetry.throughput_mbs = mbs;
            if self.telemetry.total_batches > 6 {
                self.telemetry.peak_throughput_mbs = self.telemetry.peak_throughput_mbs.max(mbs);
            }
            self.telemetry.pager_evictions_per_sec = (sum_ev as f64) / span;
        }
    }

    pub fn log_summary(&self) {
        let pt = &self.telemetry;
        println!(
            "[pursuit-zero-copy] BcPackDecodeBatchToTextures (async): {:.1} MB/s (peak {:.1} MB/s) | {} batches, {} tex ({:.1} MiB direct VRAM, {} GFX10 SW_64KB_D) | fmt[BC1/3/4/5/6H/7]={:?} | Pager={}K/262K evict={} ({:.1}K/s) | WhipPans={} | 1%Low={:.1} FPS",
            pt.throughput_mbs,
            pt.peak_throughput_mbs,
            pt.total_batches,
            pt.total_textures,
            pt.total_bytes_vram as f64 / (1024.0 * 1024.0),
            pt.gfx10_detiled_count,
            pt.format_counts,
            pt.pager_active_slots / 1000,
            pt.pager_evictions_total,
            pt.pager_evictions_per_sec / 1000.0,
            pt.whip_pan_count,
            pt.fps_1pct_low,
        );
    }
}

impl Drop for PursuitStreamer {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::Release);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}
