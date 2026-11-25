use crate::assets::{TextureAsset, ASSET_NAMES, TILE_BC1_BYTES, TILE_BLOCKS, TILE_STRIPS};
use crate::gpu_decoder::{pack_batch, BcpackGpuDecoder, GpuContext};
use std::collections::{HashMap, VecDeque};

pub const GRID_DIM: i32 = 32;
pub const JET_GRID_DIM: i32 = 64;
pub const SECTOR_SIZE_M: f32 = 250.0;
pub const LOD0_RADIUS_M: f32 = 250.0;
pub const LOD1_RADIUS_M: f32 = 800.0;
pub const HLOD_RADIUS_M: f32 = 2500.0;

const PAGE_BYTES: usize = 4 * TILE_BC1_BYTES; // 128 KiB virtual VRAM page (4x 4K-BC1 tiles)
const ROLLING_WINDOW_FRAMES: usize = 30;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LodRing {
    Hlod = 0,
    Lod1 = 1,
    Lod0 = 2,
}

impl LodRing {
    #[inline]
    pub fn tiles(self) -> usize {
        match self {
            LodRing::Lod0 => 64,
            LodRing::Lod1 => 16,
            LodRing::Hlod => 4,
        }
    }

    #[inline]
    pub fn bytes(self) -> usize {
        self.tiles() * TILE_BC1_BYTES
    }

    #[inline]
    pub fn pages(self) -> usize {
        self.tiles() / 4
    }

    #[inline]
    fn priority_rank(self) -> u8 {
        match self {
            LodRing::Lod0 => 0,
            LodRing::Lod1 => 1,
            LodRing::Hlod => 2,
        }
    }
}

struct ResidentSector {
    lod: LodRing,
    bytes: usize,
    last_used_frame: u64,
    slab_start: usize,
    slab_pages: usize,
}

struct SlabAllocator {
    cap_pages: usize,
    high_water_pages: usize,
    free_runs: Vec<(usize, usize)>,
}

impl SlabAllocator {
    fn new(cap_bytes: usize) -> Self {
        Self {
            cap_pages: (cap_bytes / PAGE_BYTES).max(1),
            high_water_pages: 0,
            free_runs: Vec::new(),
        }
    }

    fn alloc(&mut self, pages: usize) -> Option<usize> {
        for idx in 0..self.free_runs.len() {
            let (start, len) = self.free_runs[idx];
            if len >= pages {
                if len == pages {
                    self.free_runs.remove(idx);
                } else {
                    self.free_runs[idx] = (start + pages, len - pages);
                }
                return Some(start);
            }
        }
        if self.high_water_pages + pages <= self.cap_pages {
            let start = self.high_water_pages;
            self.high_water_pages += pages;
            return Some(start);
        }
        None
    }

    fn free(&mut self, start: usize, pages: usize) {
        if pages == 0 {
            return;
        }
        self.free_runs.push((start, pages));
        self.free_runs.sort_by_key(|r| r.0);
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free_runs.len());
        for (s, l) in self.free_runs.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == s {
                    last.1 += l;
                    continue;
                }
            }
            merged.push((s, l));
        }
        self.free_runs = merged;
    }

    fn compact(&mut self, residents: &mut HashMap<(i32, i32), ResidentSector>) {
        let mut keys: Vec<(i32, i32)> = residents.keys().copied().collect();
        keys.sort_by_key(|k| residents[k].slab_start);
        let mut cursor = 0usize;
        for k in keys {
            if let Some(sec) = residents.get_mut(&k) {
                sec.slab_start = cursor;
                cursor += sec.slab_pages;
            }
        }
        self.high_water_pages = cursor;
        self.free_runs.clear();
    }

    fn fragmentation_pct(&self) -> f64 {
        if self.high_water_pages == 0 {
            return 0.0;
        }
        let free_below_hw: usize = self
            .free_runs
            .iter()
            .filter(|(s, l)| s + l <= self.high_water_pages)
            .map(|(_, l)| *l)
            .sum();
        (free_below_hw as f64 / self.high_water_pages as f64) * 100.0
    }
}

pub struct WorldStreamStep {
    pub tiles_to_decode: Vec<(usize, usize)>,
    pub resident_sectors: usize,
    pub resident_bytes: usize,
    pub cap_bytes: usize,
    pub lod0_count: usize,
    pub lod1_count: usize,
    pub hlod_count: usize,
    pub frame_misses: usize,
    pub frame_evictions: usize,
    pub total_evictions: u64,
    pub total_misses: u64,
    pub fragmentation_pct: f64,
    pub burst_mb: f64,
    pub rolling_mbs: f64,
    pub peak_mbs: f64,
}

pub struct WorldStreamManager {
    grid_dim: i32,
    cap_bytes: usize,
    resident_bytes: usize,
    residents: HashMap<(i32, i32), ResidentSector>,
    slab: SlabAllocator,
    history: VecDeque<(usize, f64)>,
    pub total_evictions: u64,
    pub total_misses: u64,
    pub peak_mbs: f64,
}

#[inline]
fn sector_hash(gx: i32, gz: i32, tile_i: usize) -> u32 {
    let mut h = (gx as u32)
        .wrapping_mul(73856093)
        ^ (gz as u32).wrapping_mul(19349663)
        ^ (tile_i as u32).wrapping_mul(83492791);
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
}

#[inline]
fn toroidal_dist_sq(key: (i32, i32), world_pos: [f32; 3], grid_dim: i32) -> f32 {
    let world_span = grid_dim as f32 * SECTOR_SIZE_M;
    let sx = key.0 as f32 * SECTOR_SIZE_M + SECTOR_SIZE_M * 0.5;
    let sz = key.1 as f32 * SECTOR_SIZE_M + SECTOR_SIZE_M * 0.5;
    let wx = world_pos[0].rem_euclid(world_span);
    let wz = world_pos[2].rem_euclid(world_span);
    let mut dx = (sx - wx).abs();
    let mut dz = (sz - wz).abs();
    if dx > world_span * 0.5 {
        dx = world_span - dx;
    }
    if dz > world_span * 0.5 {
        dz = world_span - dz;
    }
    dx * dx + dz * dz
}

struct Candidate {
    key: (i32, i32),
    target_lod: LodRing,
    dist: f32,
    in_frustum: bool,
}

impl WorldStreamManager {
    pub fn new(cap_bytes: usize) -> Self {
        Self::with_grid_dim(cap_bytes, GRID_DIM)
    }

    pub fn with_grid_dim(cap_bytes: usize, grid_dim: i32) -> Self {
        let cap_bytes = cap_bytes.max(PAGE_BYTES * 32);
        Self {
            grid_dim: grid_dim.max(8),
            cap_bytes,
            resident_bytes: 0,
            residents: HashMap::with_capacity(512),
            slab: SlabAllocator::new(cap_bytes),
            history: VecDeque::with_capacity(ROLLING_WINDOW_FRAMES + 1),
            total_evictions: 0,
            total_misses: 0,
            peak_mbs: 0.0,
        }
    }

    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    fn evict_one(
        &mut self,
        world_pos: [f32; 3],
        frame_idx: u64,
        allow_same_frame_lower_than: Option<(LodRing, f32)>,
    ) -> bool {
        let mut best_stale: Option<(u64, f32, (i32, i32))> = None;
        for (&k, v) in &self.residents {
            if v.last_used_frame < frame_idx {
                let dsq = toroidal_dist_sq(k, world_pos, self.grid_dim);
                let replace = match best_stale {
                    None => true,
                    Some((bf, bdsq, _)) => {
                        v.last_used_frame < bf || (v.last_used_frame == bf && dsq > bdsq)
                    }
                };
                if replace {
                    best_stale = Some((v.last_used_frame, dsq, k));
                }
            }
        }

        let chosen = if let Some((_, _, k)) = best_stale {
            Some(k)
        } else if let Some((req_lod, req_dist)) = allow_same_frame_lower_than {
            let req_dsq = req_dist * req_dist;
            let mut best_same: Option<(u8, f32, (i32, i32))> = None;
            for (&k, v) in &self.residents {
                let dsq = toroidal_dist_sq(k, world_pos, self.grid_dim);
                let lower_pri = v.lod < req_lod || (v.lod == req_lod && dsq > req_dsq + 1.0);
                if lower_pri {
                    let r = v.lod.priority_rank();
                    let replace = match best_same {
                        None => true,
                        Some((br, bdsq, _)) => r > br || (r == br && dsq > bdsq),
                    };
                    if replace {
                        best_same = Some((r, dsq, k));
                    }
                }
            }
            best_same.map(|x| x.2)
        } else {
            None
        };

        if let Some(k) = chosen {
            if let Some(removed) = self.residents.remove(&k) {
                self.resident_bytes -= removed.bytes;
                self.slab.free(removed.slab_start, removed.slab_pages);
                self.total_evictions += 1;
                return true;
            }
        }
        false
    }

    pub fn update(
        &mut self,
        world_pos: [f32; 3],
        cam_fwd: [f32; 3],
        frame_idx: u64,
        dt_sec: f32,
        max_tiles_per_frame: usize,
    ) -> WorldStreamStep {
        let cx = (world_pos[0] / SECTOR_SIZE_M).floor() as i32;
        let cz = (world_pos[2] / SECTOR_SIZE_M).floor() as i32;
        let max_off = (HLOD_RADIUS_M / SECTOR_SIZE_M).ceil() as i32;

        let f_len = (cam_fwd[0] * cam_fwd[0] + cam_fwd[2] * cam_fwd[2])
            .sqrt()
            .max(1e-5);
        let fx = cam_fwd[0] / f_len;
        let fz = cam_fwd[2] / f_len;

        let mut candidates = Vec::with_capacity(441);
        for dz in -max_off..=max_off {
            for dx in -max_off..=max_off {
                let sx = cx + dx;
                let sz = cz + dz;
                let center_x = sx as f32 * SECTOR_SIZE_M + SECTOR_SIZE_M * 0.5;
                let center_z = sz as f32 * SECTOR_SIZE_M + SECTOR_SIZE_M * 0.5;
                let vx = center_x - world_pos[0];
                let vz = center_z - world_pos[2];
                let dist = (vx * vx + vz * vz).sqrt();
                if dist > HLOD_RADIUS_M {
                    continue;
                }
                let target_lod = if dist <= LOD0_RADIUS_M {
                    LodRing::Lod0
                } else if dist <= LOD1_RADIUS_M {
                    LodRing::Lod1
                } else {
                    LodRing::Hlod
                };
                let dot = if dist > 1e-3 {
                    (vx / dist) * fx + (vz / dist) * fz
                } else {
                    1.0
                };
                let in_frustum = dot > 0.25 || dist < SECTOR_SIZE_M * 0.9;
                let key = (sx.rem_euclid(self.grid_dim), sz.rem_euclid(self.grid_dim));
                candidates.push(Candidate {
                    key,
                    target_lod,
                    dist,
                    in_frustum,
                });
            }
        }

        // Priority: frustum first, then Lod0 > Lod1 > Hlod, then nearest distance
        candidates.sort_by(|a, b| {
            b.in_frustum
                .cmp(&a.in_frustum)
                .then_with(|| a.target_lod.priority_rank().cmp(&b.target_lod.priority_rank()))
                .then_with(|| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal))
        });

        let mut pending = Vec::new();
        for c in &candidates {
            if let Some(res) = self.residents.get_mut(&c.key) {
                res.last_used_frame = frame_idx;
                if res.lod < c.target_lod {
                    pending.push(c);
                }
            } else {
                pending.push(c);
            }
        }

        let mut tiles_to_decode: Vec<(usize, usize)> = Vec::new();
        let mut frame_misses = 0usize;
        let mut frame_evictions = 0usize;

        for c in pending {
            let (existing_tiles, existing_bytes) = match self.residents.get(&c.key) {
                Some(r) if r.lod >= c.target_lod => continue,
                Some(r) => (r.lod.tiles(), r.bytes),
                None => (0, 0),
            };
            let needed_tiles = c.target_lod.tiles().saturating_sub(existing_tiles);
            if needed_tiles == 0 {
                continue;
            }
            if !tiles_to_decode.is_empty()
                && tiles_to_decode.len() + needed_tiles > max_tiles_per_frame
            {
                continue;
            }

            let target_bytes = c.target_lod.bytes();
            let target_pages = c.target_lod.pages();
            let delta_bytes = target_bytes.saturating_sub(existing_bytes);

            while self.resident_bytes + delta_bytes > self.cap_bytes {
                if self.evict_one(world_pos, frame_idx, Some((c.target_lod, c.dist))) {
                    frame_evictions += 1;
                } else {
                    break;
                }
            }
            if self.resident_bytes + delta_bytes > self.cap_bytes {
                continue;
            }

            let re_existing = self.residents.remove(&c.key);
            if let Some(ref old) = re_existing {
                self.resident_bytes -= old.bytes;
                self.slab.free(old.slab_start, old.slab_pages);
            }

            let mut slab_start = self.slab.alloc(target_pages);
            if slab_start.is_none() {
                self.slab.compact(&mut self.residents);
                slab_start = self.slab.alloc(target_pages);
            }
            let Some(start_page) = slab_start else {
                if let Some(old) = re_existing {
                    let s = self.slab.alloc(old.slab_pages).unwrap_or(0);
                    self.resident_bytes += old.bytes;
                    self.residents.insert(
                        c.key,
                        ResidentSector {
                            slab_start: s,
                            ..old
                        },
                    );
                }
                continue;
            };

            self.resident_bytes += target_bytes;
            self.residents.insert(
                c.key,
                ResidentSector {
                    lod: c.target_lod,
                    bytes: target_bytes,
                    last_used_frame: frame_idx,
                    slab_start: start_page,
                    slab_pages: target_pages,
                },
            );

            for t in 0..needed_tiles {
                let h = sector_hash(c.key.0, c.key.1, existing_tiles + t);
                let tex_idx = (h as usize) % ASSET_NAMES.len();
                let strip = ((h >> 8) as usize) % TILE_STRIPS;
                tiles_to_decode.push((tex_idx, strip));
            }
            frame_misses += 1;
            self.total_misses += 1;
            if tiles_to_decode.len() >= max_tiles_per_frame {
                break;
            }
        }

        let mut lod0_count = 0usize;
        let mut lod1_count = 0usize;
        let mut hlod_count = 0usize;
        for sec in self.residents.values() {
            match sec.lod {
                LodRing::Lod0 => lod0_count += 1,
                LodRing::Lod1 => lod1_count += 1,
                LodRing::Hlod => hlod_count += 1,
            }
        }

        let burst_bytes = tiles_to_decode.len() * TILE_BC1_BYTES;
        let dt_clamped = (dt_sec as f64).max(0.001);
        self.history.push_back((burst_bytes, dt_clamped));
        if self.history.len() > ROLLING_WINDOW_FRAMES {
            self.history.pop_front();
        }
        let sum_bytes: usize = self.history.iter().map(|(b, _)| *b).sum();
        let sum_dt: f64 = self.history.iter().map(|(_, d)| *d).sum();
        let rolling_mbs = if sum_dt > 0.0 {
            (sum_bytes as f64 / (1024.0 * 1024.0)) / sum_dt
        } else {
            0.0
        };
        if rolling_mbs > self.peak_mbs {
            self.peak_mbs = rolling_mbs;
        }

        WorldStreamStep {
            tiles_to_decode,
            resident_sectors: self.residents.len(),
            resident_bytes: self.resident_bytes,
            cap_bytes: self.cap_bytes,
            lod0_count,
            lod1_count,
            hlod_count,
            frame_misses,
            frame_evictions,
            total_evictions: self.total_evictions,
            total_misses: self.total_misses,
            fragmentation_pct: self.slab.fragmentation_pct(),
            burst_mb: burst_bytes as f64 / (1024.0 * 1024.0),
            rolling_mbs,
            peak_mbs: self.peak_mbs,
        }
    }
}

pub struct WorldStreamVerifyStats {
    pub frames: usize,
    pub total_streamed_tiles: usize,
    pub total_evictions: u64,
    pub turn_burst_tiles: usize,
    pub peak_mbs: f64,
    pub verified_tiles: usize,
    pub gpu_ms: f64,
}

pub fn run_world_stream_verify(
    ctx: &mut GpuContext,
    gpu_dec: &mut BcpackGpuDecoder,
    assets: &[TextureAsset],
) -> WorldStreamVerifyStats {
    let cap_bytes = 32 * 1024 * 1024; // 32 MiB VRAM residency cap
    let mut mgr = WorldStreamManager::new(cap_bytes);

    let mut pos = [500.0f32, 6.0, 500.0];
    let mut fwd = [0.0f32, 0.0, 1.0];
    let mut max_lod0 = 0usize;
    let mut max_lod1 = 0usize;
    let mut max_hlod = 0usize;
    let mut total_streamed_tiles = 0usize;
    let mut turn_burst_tiles = 0usize;
    let mut collected_tiles: Vec<(usize, usize)> = Vec::new();

    for frame in 1u64..=180 {
        if frame <= 120 {
            pos[2] += 60.0;
            fwd = [0.0, 0.0, 1.0];
        } else if frame == 121 {
            // Sudden 180-degree turn + lateral sweep to force streaming burst
            fwd = [0.6, 0.0, -0.8];
            pos[0] += 320.0;
            pos[2] -= 120.0;
        } else {
            pos[0] += 45.0;
            pos[2] -= 60.0;
        }

        let step = mgr.update(pos, fwd, frame, 1.0 / 60.0, 256);
        assert!(
            step.resident_bytes <= step.cap_bytes,
            "VRAM residency overflow at frame {frame}: {} > {}",
            step.resident_bytes,
            step.cap_bytes
        );

        max_lod0 = max_lod0.max(step.lod0_count);
        max_lod1 = max_lod1.max(step.lod1_count);
        max_hlod = max_hlod.max(step.hlod_count);
        total_streamed_tiles += step.tiles_to_decode.len();
        if (121..=125).contains(&frame) {
            turn_burst_tiles += step.tiles_to_decode.len();
        }
        for &pair in &step.tiles_to_decode {
            if collected_tiles.len() < 256 {
                collected_tiles.push(pair);
            }
        }
    }

    assert!(max_lod0 > 0, "Expected LOD0 sectors > 0");
    assert!(max_lod1 > 0, "Expected LOD1 sectors > 0");
    assert!(max_hlod > 0, "Expected HLOD sectors > 0");
    assert!(mgr.total_evictions > 0, "Expected LRU evictions > 0");
    assert!(total_streamed_tiles > 0, "Expected streamed tiles > 0");
    assert!(turn_burst_tiles > 0, "Expected streaming burst on 180-deg camera turn");

    // Bit-exact GPU decode verification of the streamed world-grid tiles
    assert!(!collected_tiles.is_empty());
    let batch = pack_batch(
        collected_tiles
            .iter()
            .map(|&(ti, st)| (assets[ti].bcpack_tiles[st].as_slice(), TILE_BLOCKS)),
    );
    let (gpu_out, statuses, gpu_ms, _) = gpu_dec.decode_batch_and_readback(ctx, &batch);
    for (idx, &(ti, st)) in collected_tiles.iter().enumerate() {
        assert_eq!(statuses[idx].code, 0, "GPU status error on world tile {idx}");
        let got = &gpu_out[idx * TILE_BC1_BYTES..(idx + 1) * TILE_BC1_BYTES];
        let exp = &assets[ti].expected_bc1[st * TILE_BC1_BYTES..(st + 1) * TILE_BC1_BYTES];
        if got != exp {
            let mm = got.iter().zip(exp.iter()).position(|(a, b)| a != b).unwrap();
            panic!("FAIL world-stream tile {idx} (tex={ti}, strip={st}): mismatch at byte {mm}");
        }
    }

    WorldStreamVerifyStats {
        frames: 180,
        total_streamed_tiles,
        total_evictions: mgr.total_evictions,
        turn_burst_tiles,
        peak_mbs: mgr.peak_mbs,
        verified_tiles: collected_tiles.len(),
        gpu_ms,
    }
}

pub struct JetFlightVerifyStats {
    pub grid_dim: i32,
    pub total_sectors: usize,
    pub area_km2: f64,
    pub frames: usize,
    pub total_streamed_tiles: usize,
    pub total_evictions: u64,
    pub max_turn_burst_tiles: usize,
    pub peak_mbs: f64,
    pub verified_tiles: usize,
    pub gpu_ms: f64,
}

pub fn run_jet_flight_verify(
    ctx: &mut GpuContext,
    gpu_dec: &mut BcpackGpuDecoder,
    assets: &[TextureAsset],
) -> JetFlightVerifyStats {
    let cap_bytes = 64 * 1024 * 1024; // 64 MiB VRAM residency cap across 64x64 grid (256 km^2)
    let mut mgr = WorldStreamManager::with_grid_dim(cap_bytes, JET_GRID_DIM);

    let mut pos = [1250.0f32, 85.0, 1250.0];
    let mut fwd = [0.0f32, 0.0, 1.0];
    let mut total_streamed_tiles = 0usize;
    let mut max_turn_burst_tiles = 0usize;
    let mut collected_tiles: Vec<(usize, usize)> = Vec::new();

    for frame in 1u64..=360 {
        if frame <= 180 {
            // Mach 1 cruise (~340 m/s = 5.67 m/frame @ 60 FPS -> scaled burst 24 m/frame to cross 34+ sectors)
            pos[2] += 24.0;
            fwd = [0.0, 0.0, 1.0];
        } else if frame == 181 {
            // High-G Mach 2 180-deg supersonic reversal
            fwd = [0.7071, 0.0, -0.7071];
            pos[0] += 420.0;
            pos[2] -= 180.0;
        } else {
            pos[0] += 26.0;
            pos[2] -= 26.0;
        }

        let step = mgr.update(pos, fwd, frame, 1.0 / 60.0, 512);
        assert!(
            step.resident_bytes <= step.cap_bytes,
            "Jet-flight VRAM overflow at frame {frame}: {} > {}",
            step.resident_bytes,
            step.cap_bytes
        );

        total_streamed_tiles += step.tiles_to_decode.len();
        max_turn_burst_tiles = max_turn_burst_tiles.max(step.tiles_to_decode.len());
        for &pair in &step.tiles_to_decode {
            if collected_tiles.len() < 512 {
                collected_tiles.push(pair);
            }
        }
    }

    assert!(mgr.total_evictions > 0, "Expected 64x64 jet-flight LRU evictions > 0");
    assert!(total_streamed_tiles >= 512, "Expected >= 512 streamed tiles");
    assert_eq!(collected_tiles.len(), 512, "Expected full 512-tile burst for GPU verification");

    let batch = pack_batch(
        collected_tiles
            .iter()
            .map(|&(ti, st)| (assets[ti].bcpack_tiles[st].as_slice(), TILE_BLOCKS)),
    );
    let (gpu_out, statuses, gpu_ms, _) = gpu_dec.decode_batch_and_readback(ctx, &batch);
    for (idx, &(ti, st)) in collected_tiles.iter().enumerate() {
        assert_eq!(statuses[idx].code, 0, "GPU status error on jet-flight tile {idx}");
        let got = &gpu_out[idx * TILE_BC1_BYTES..(idx + 1) * TILE_BC1_BYTES];
        let exp = &assets[ti].expected_bc1[st * TILE_BC1_BYTES..(st + 1) * TILE_BC1_BYTES];
        if got != exp {
            let mm = got.iter().zip(exp.iter()).position(|(a, b)| a != b).unwrap();
            panic!("FAIL jet-flight tile {idx} (tex={ti}, strip={st}): mismatch at byte {mm}");
        }
    }

    let side_km = (JET_GRID_DIM as f64 * SECTOR_SIZE_M as f64) / 1000.0;
    JetFlightVerifyStats {
        grid_dim: JET_GRID_DIM,
        total_sectors: (JET_GRID_DIM * JET_GRID_DIM) as usize,
        area_km2: side_km * side_km,
        frames: 360,
        total_streamed_tiles,
        total_evictions: mgr.total_evictions,
        max_turn_burst_tiles,
        peak_mbs: mgr.peak_mbs,
        verified_tiles: collected_tiles.len(),
        gpu_ms,
    }
}
