// Open procedural 512x512 BC1 texture generator + BCPack BC1 encoder.
// Uses clean-room BCPack tables and rANS encoder from Ahmet-Dedeler/bcpack-rs (MIT OR Apache-2.0).

use bcpack::bc1::{self, QuadTable, Window, BC1_ALPHABET_BITS, ENDPOINT_WIDTHS};
use bcpack::encode::{encode_lanes, Fragment};
use bcpack::rans;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const TEX_W: usize = 512;
pub const TEX_H: usize = 512;
pub const BLOCKS_X: usize = TEX_W / 4; // 128
pub const BLOCKS_Y: usize = TEX_H / 4; // 128
pub const NUM_BLOCKS: usize = BLOCKS_X * BLOCKS_Y; // 16384
pub const TEX_BC1_BYTES: usize = NUM_BLOCKS * 8; // 131072

pub const TILE_STRIPS: usize = 4;
pub const TILE_BLOCKS: usize = NUM_BLOCKS / TILE_STRIPS; // 4096 blocks (512x128 strip)
pub const TILE_BC1_BYTES: usize = TILE_BLOCKS * 8; // 32768 bytes
pub const TILE_BC16_BYTES: usize = TILE_BLOCKS * 16; // 65536 bytes (exact 64 KiB Xbox tile)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TexFormat {
    Bc1 = 0, // 8 bytes/block, DXGI_FORMAT_BC1_UNORM
    Bc5 = 1, // 16 bytes/block, DXGI_FORMAT_BC5_UNORM (2-channel normal XY)
    Bc7 = 2, // 16 bytes/block, DXGI_FORMAT_BC7_UNORM (Mode 6 RGBA)
}

impl TexFormat {
    #[inline]
    pub fn block_bytes(self) -> usize {
        match self {
            TexFormat::Bc1 => 8,
            TexFormat::Bc5 | TexFormat::Bc7 => 16,
        }
    }

    #[inline]
    pub fn tag(self) -> &'static str {
        match self {
            TexFormat::Bc1 => "bc1",
            TexFormat::Bc5 => "bc5",
            TexFormat::Bc7 => "bc7",
        }
    }

    #[inline]
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Bc1),
            1 => Some(Self::Bc5),
            2 => Some(Self::Bc7),
            _ => None,
        }
    }
}

pub const ASSET_NAMES: [&str; 12] = [
    "01_highway_asphalt",
    "02_neon_skyscraper",
    "03_miami_hotel",
    "04_promenade_tiles",
    "05_ocean_waves",
    "06_palm_fronds",
    "07_synthwave_billboard",
    "08_beach_sand",
    "09_car_body_red",
    "10_car_body_cyan",
    "11_storefront_neon",
    "12_concrete_rooftop",
];

/// Exact BC7 16-lane rANS alphabet widths (matches Ahmet-Dedeler/bcpack-rs format.rs).
pub const BC7_ALPHABET_BITS: [u8; 16] = [
    3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4,
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Bc7EntropyProfile {
    StandardAlbedo,
    NormalHighEntropy,
    SharpUi,
}

pub struct TextureAsset {
    pub name: &'static str,
    pub bcpack_bytes: Vec<u8>,
    pub bcpack_tiles: [Vec<u8>; TILE_STRIPS],
    pub expected_bc1: Vec<u8>,
}

#[inline]
fn hash2(x: i32, y: i32) -> u32 {
    let mut h = (x as u32)
        .wrapping_mul(374761393)
        .wrapping_add((y as u32).wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
}

#[inline]
fn lerp_u8(a: f32, b: f32, t: f32) -> u8 {
    (a + (b - a) * t.clamp(0.0, 1.0)).round().clamp(0.0, 255.0) as u8
}

#[inline]
fn rgb(r: f32, g: f32, b: f32) -> [u8; 4] {
    [
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
        255,
    ]
}

fn generate_rgba(idx: usize) -> Vec<u8> {
    let mut img = vec![0u8; TEX_W * TEX_H * 4];
    for y in 0..TEX_H {
        for x in 0..TEX_W {
            let u = x as f32 / TEX_W as f32;
            let v = y as f32 / TEX_H as f32;
            let n = ((hash2(x as i32, y as i32) & 0xFF) as f32 / 255.0) - 0.5;
            let px = match idx {
                // 01_highway_asphalt: coastal 4-lane highway with double yellow median & white lane dashes
                0 => {
                    let g = n * 22.0;
                    let wear = ((u * 12.0).sin().abs()) * 6.0;
                    let mut col = [30.0 + g + wear, 32.0 + g + wear, 42.0 + g + wear];
                    if (u - 0.48).abs() < 0.011 || (u - 0.52).abs() < 0.011 {
                        col = [240.0 + g * 0.3, 188.0 + g * 0.3, 28.0];
                    }
                    let dash = ((v * 6.0).fract()) < 0.52;
                    if dash && ((u - 0.24).abs() < 0.010 || (u - 0.76).abs() < 0.010) {
                        col = [235.0, 238.0, 248.0];
                    }
                    if u < 0.035 || u > 0.965 {
                        col = [135.0 + g, 128.0 + g, 142.0 + g];
                    }
                    rgb(col[0], col[1], col[2])
                }
                // 02_neon_skyscraper: Art-Deco glass facade with glowing cyan & magenta trims
                1 => {
                    let gx = (x % 64) as f32 / 64.0;
                    let gy = (y % 64) as f32 / 64.0;
                    let cell_id = hash2((x / 64) as i32, (y / 64) as i32);
                    let mut col = [16.0, 18.0, 38.0];
                    if gx > 0.14 && gx < 0.86 && gy > 0.16 && gy < 0.82 {
                        let lit = (cell_id & 7) < 3;
                        if lit {
                            let warm = (cell_id & 8) != 0;
                            col = if warm {
                                [255.0 - gy * 45.0, 195.0 - gy * 35.0, 110.0]
                            } else {
                                [70.0, 225.0 - gy * 35.0, 255.0]
                            };
                        } else {
                            col = [28.0 + gy * 25.0, 40.0 + gy * 30.0, 78.0 + gy * 35.0];
                        }
                    }
                    if (y % 128) < 6 {
                        col = [0.0, 248.0, 255.0];
                    } else if ((y + 64) % 128) < 5 {
                        col = [255.0, 45.0, 195.0];
                    }
                    if x < 9 || x > TEX_W - 9 {
                        col = [255.0, 65.0, 205.0];
                    }
                    rgb(col[0] + n * 6.0, col[1] + n * 6.0, col[2] + n * 6.0)
                }
                // 03_miami_hotel: Ocean Drive pastel coral & turquoise deco facade
                2 => {
                    let mut col = [
                        246.0 - v * 30.0 + n * 10.0,
                        156.0 - v * 22.0 + n * 10.0,
                        152.0 + v * 28.0 + n * 10.0,
                    ];
                    if (y % 128) < 12 || (y % 128) > 120 {
                        col = [40.0, 212.0, 200.0];
                    }
                    let wx = (x % 64) as f32 / 64.0;
                    let wy = (y % 128) as f32 / 128.0;
                    if wx > 0.22 && wx < 0.78 && wy > 0.22 && wy < 0.78 {
                        let glow = ((hash2((x / 64) as i32, (y / 128) as i32)) & 1) == 0;
                        col = if glow { [255.0, 224.0, 138.0] } else { [34.0, 54.0, 82.0] };
                    }
                    if wx > 0.16 && wx < 0.84 && wy > 0.78 && wy < 0.85 {
                        col = [250.0, 245.0, 240.0];
                    }
                    rgb(col[0], col[1], col[2])
                }
                // 04_promenade_tiles: warm sandstone/terracotta oceanfront boardwalk
                3 => {
                    let row = y / 32;
                    let sx = (x + if row % 2 == 1 { 32 } else { 0 }) % 64;
                    let sy = y % 32;
                    if sx < 3 || sy < 3 {
                        rgb(68.0 + n * 12.0, 58.0 + n * 12.0, 64.0 + n * 12.0)
                    } else {
                        let cid = hash2(((x + if row % 2 == 1 { 32 } else { 0 }) / 64) as i32, row as i32);
                        let tint = ((cid & 0x1F) as f32) - 16.0;
                        rgb(
                            210.0 + tint + n * 14.0,
                            142.0 + tint * 0.7 + n * 12.0,
                            115.0 + tint * 0.5 + n * 12.0,
                        )
                    }
                }
                // 05_ocean_waves: turquoise bay water with sunset crests
                4 => {
                    let w1 = (u * 18.0 + (v * 12.0).sin() * 1.4).sin();
                    let w2 = (v * 34.0 - u * 10.0).cos();
                    let wave = (w1 * 0.55 + w2 * 0.45) * 0.5 + 0.5;
                    let mut col = [
                        lerp_u8(12.0, 44.0, wave) as f32,
                        lerp_u8(102.0, 218.0, wave) as f32,
                        lerp_u8(148.0, 240.0, wave) as f32,
                    ];
                    if wave > 0.82 {
                        let t = (wave - 0.82) / 0.18;
                        col[0] += t * 210.0;
                        col[1] += t * 75.0;
                        col[2] += t * 25.0;
                    }
                    rgb(col[0] + n * 5.0, col[1] + n * 5.0, col[2] + n * 5.0)
                }
                // 06_palm_fronds: striated tropical palm leaf blade
                5 => {
                    let center = (u - 0.5).abs();
                    let vein = ((v * 36.0 - center * 14.0).sin() * 0.5 + 0.5).powf(1.5);
                    let rib = if center < 0.04 { 1.0 } else { 0.0 };
                    rgb(
                        32.0 + vein * 38.0 + rib * 65.0 + n * 10.0,
                        105.0 + vein * 115.0 + rib * 40.0 + n * 12.0,
                        28.0 + vein * 28.0 + n * 8.0,
                    )
                }
                // 07_synthwave_billboard: glowing retro highway arch sign
                6 => {
                    let dx = u - 0.5;
                    let dy = v - 0.42;
                    let r = (dx * dx + dy * dy).sqrt();
                    let mut col = [22.0, 8.0, 48.0];
                    if v > 0.56 {
                        let hz = (v - 0.56) / 0.44;
                        let gx = ((u - 0.5) / (0.15 + hz * 1.4) * 8.0).fract().abs();
                        let gy = (hz * 10.0).fract().abs();
                        col = if gx < 0.08 || gy < 0.09 { [255.0, 45.0, 210.0] } else { [26.0, 12.0, 56.0] };
                    } else if r < 0.30 {
                        let band = ((v * 28.0).sin() > -0.35) || v < 0.30;
                        if band {
                            let t = ((v - 0.12) / 0.30).clamp(0.0, 1.0);
                            col = [255.0, 235.0 - t * 185.0, 50.0 + t * 140.0];
                        }
                    } else if (hash2((x / 2) as i32, (y / 2) as i32) & 255) == 0 {
                        col = [210.0, 245.0, 255.0];
                    }
                    if x < 12 || x > TEX_W - 12 || y < 12 || y > TEX_H - 12 {
                        col = [0.0, 245.0, 255.0];
                    }
                    rgb(col[0], col[1], col[2])
                }
                // 08_beach_sand: warm coastal sand ripples
                7 => {
                    let ripple = ((u * 22.0 + (v * 8.0).sin() * 0.8).sin() * 0.5 + 0.5) * 18.0;
                    rgb(225.0 + ripple + n * 16.0, 192.0 + ripple * 0.8 + n * 14.0, 145.0 + ripple * 0.5 + n * 12.0)
                }
                // 09_car_body_red: glossy crimson/magenta sports car paint + white racing stripe + lights
                8 => {
                    let mut col = [215.0 + n * 6.0, 28.0 + n * 5.0, 72.0 + n * 6.0];
                    // White-cyan racing twin stripes
                    if (u - 0.43).abs() < 0.03 || (u - 0.57).abs() < 0.03 {
                        col = [245.0, 248.0, 255.0];
                    }
                    // Headlight / taillight / wheel quad regions (matches meshes.rs UV rects)
                    if u < 0.5 && v < 0.5 && (u - 0.14).abs() < 0.08 && (v - 0.25).abs() < 0.10 {
                        col = [245.0, 250.0, 255.0]; // headlights
                    }
                    if u < 0.5 && v < 0.5 && (u - 0.36).abs() < 0.08 && (v - 0.25).abs() < 0.10 {
                        col = [245.0, 250.0, 255.0];
                    }
                    if u > 0.5 && v > 0.5 && (v - 0.72).abs() < 0.08 {
                        col = [255.0, 30.0, 45.0]; // full-width red taillight bar
                    }
                    if u > 0.72 && v > 0.72 {
                        col = [26.0 + n * 8.0, 26.0 + n * 8.0, 32.0 + n * 8.0]; // tires
                    }
                    rgb(col[0], col[1], col[2])
                }
                // 10_car_body_cyan: electric Miami cyan/purple traffic coupe paint
                9 => {
                    let mut col = [24.0 + n * 6.0, 195.0 + n * 6.0, 240.0 + n * 6.0];
                    if (u - 0.50).abs() < 0.045 {
                        col = [255.0, 60.0, 210.0];
                    }
                    if u < 0.5 && v < 0.5 && ((u - 0.14).abs() < 0.08 || (u - 0.36).abs() < 0.08) && (v - 0.25).abs() < 0.10 {
                        col = [255.0, 250.0, 215.0];
                    }
                    if u > 0.5 && v > 0.5 && (v - 0.72).abs() < 0.08 {
                        col = [255.0, 35.0, 60.0];
                    }
                    if u > 0.72 && v > 0.72 {
                        col = [24.0 + n * 8.0, 24.0 + n * 8.0, 30.0 + n * 8.0];
                    }
                    rgb(col[0], col[1], col[2])
                }
                // 11_storefront_neon: ground-floor oceanfront cafes, glowing signs & awnings
                10 => {
                    let mut col = [34.0 + n * 8.0, 26.0 + n * 8.0, 48.0 + n * 8.0];
                    let sx = (x % 128) as f32 / 128.0;
                    if v > 0.35 && v < 0.88 && sx > 0.10 && sx < 0.90 {
                        // Warm lit shop interior
                        col = [255.0 - v * 60.0, 185.0 - sx * 35.0, 95.0];
                    }
                    if (v - 0.26).abs() < 0.06 {
                        // Striped beach awning
                        col = if ((x / 16) % 2) == 0 { [255.0, 75.0, 145.0] } else { [245.0, 245.0, 245.0] };
                    }
                    if (v - 0.14).abs() < 0.03 {
                        col = [40.0, 255.0, 215.0]; // neon sign tube
                    }
                    rgb(col[0], col[1], col[2])
                }
                // 12_concrete_rooftop: rooftop concrete + crisp calibration checker on center tile
                _ => {
                    let cx = (x / 32) % 2;
                    let cy = (y / 32) % 2;
                    let chk = (cx ^ cy) == 0;
                    let base = if (u - 0.5).abs() < 0.25 && (v - 0.5).abs() < 0.25 {
                        if chk { [255.0, 60.0, 140.0] } else { [40.0, 240.0, 255.0] }
                    } else {
                        [76.0 + n * 16.0, 74.0 + n * 16.0, 88.0 + n * 16.0]
                    };
                    rgb(base[0], base[1], base[2])
                }
            };
            let off = (y * TEX_W + x) * 4;
            img[off..off + 4].copy_from_slice(&px);
        }
    }
    img
}

#[inline]
fn to_565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = (r as u16 * 31 + 127) / 255;
    let g6 = (g as u16 * 63 + 127) / 255;
    let b5 = (b as u16 * 31 + 127) / 255;
    (r5 << 11) | (g6 << 5) | b5
}

#[inline]
fn expand_565(c: u16) -> [i32; 3] {
    let r5 = (c >> 11) & 31;
    let g6 = (c >> 5) & 63;
    let b5 = c & 31;
    [
        ((r5 << 3) | (r5 >> 2)) as i32,
        ((g6 << 2) | (g6 >> 4)) as i32,
        ((b5 << 3) | (b5 >> 2)) as i32,
    ]
}

pub fn compress_rgba_dims_to_bc1(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let bx_count = width / 4;
    let by_count = height / 4;
    let mut out = vec![0u8; bx_count * by_count * 8];
    for by in 0..by_count {
        for bx in 0..bx_count {
            let mut block_px = [[0u8; 3]; 16];
            let mut min_rgb = [255u8; 3];
            let mut max_rgb = [0u8; 3];
            for py in 0..4 {
                for px in 0..4 {
                    let off = ((by * 4 + py) * width + (bx * 4 + px)) * 4;
                    let p = [rgba[off], rgba[off + 1], rgba[off + 2]];
                    block_px[py * 4 + px] = p;
                    for c in 0..3 {
                        min_rgb[c] = min_rgb[c].min(p[c]);
                        max_rgb[c] = max_rgb[c].max(p[c]);
                    }
                }
            }

            let mut c_hi = [0u8; 3];
            let mut c_lo = [0u8; 3];
            for c in 0..3 {
                let inset = (max_rgb[c] - min_rgb[c]) >> 4;
                c_hi[c] = max_rgb[c].saturating_sub(inset);
                c_lo[c] = min_rgb[c].saturating_add(inset);
            }

            let mut c0 = to_565(c_hi[0], c_hi[1], c_hi[2]);
            let mut c1 = to_565(c_lo[0], c_lo[1], c_lo[2]);
            if c0 < c1 {
                std::mem::swap(&mut c0, &mut c1);
            } else if c0 == c1 {
                if c0 < 0xFFFF {
                    c0 += 1;
                } else {
                    c1 -= 1;
                }
            }

            let p0 = expand_565(c0);
            let p1 = expand_565(c1);
            let pal = [
                p0,
                p1,
                [
                    (2 * p0[0] + p1[0] + 1) / 3,
                    (2 * p0[1] + p1[1] + 1) / 3,
                    (2 * p0[2] + p1[2] + 1) / 3,
                ],
                [
                    (p0[0] + 2 * p1[0] + 1) / 3,
                    (p0[1] + 2 * p1[1] + 1) / 3,
                    (p0[2] + 2 * p1[2] + 1) / 3,
                ],
            ];

            let mut idx_word = 0u32;
            for i in 0..16 {
                let px = block_px[i];
                let mut best_k = 0u32;
                let mut best_err = i32::MAX;
                for (k, &col) in pal.iter().enumerate() {
                    let dr = px[0] as i32 - col[0];
                    let dg = px[1] as i32 - col[1];
                    let db = px[2] as i32 - col[2];
                    let err = dr * dr + dg * dg + db * db;
                    if err < best_err {
                        best_err = err;
                        best_k = k as u32;
                    }
                }
                idx_word |= best_k << (2 * i);
            }

            let bi = (by * bx_count + bx) * 8;
            out[bi..bi + 2].copy_from_slice(&c0.to_le_bytes());
            out[bi + 2..bi + 4].copy_from_slice(&c1.to_le_bytes());
            out[bi + 4..bi + 8].copy_from_slice(&idx_word.to_le_bytes());
        }
    }
    out
}

fn compress_rgba_to_bc1(rgba: &[u8]) -> Vec<u8> {
    compress_rgba_dims_to_bc1(rgba, TEX_W, TEX_H)
}

#[inline]
fn encode_back_distance(d: usize) -> u16 {
    if d <= 127 {
        (d << 1) as u16
    } else {
        (((255 - d) << 1) | 1) as u16
    }
}

#[inline]
fn wrap_residual(diff: i32, width: u32) -> i32 {
    let m = 1i32 << width;
    let half = m >> 1;
    let mut r = diff.rem_euclid(m);
    if r >= half {
        r -= m;
    }
    r
}

#[inline]
fn zigzag(r: i32) -> u16 {
    if r >= 0 {
        (r * 2) as u16
    } else {
        (-r * 2 - 1) as u16
    }
}

fn best_model_for_fragment(ab: u8, syms: &[u16]) -> (u8, u8) {
    let mut hist = [0u32; 512];
    let mut max_sym = 0u16;
    for &s in syms {
        hist[s as usize] += 1;
        if s > max_sym {
            max_sym = s;
        }
    }
    let min_bits = if max_sym == 0 {
        1u8
    } else {
        (16 - max_sym.leading_zeros()) as u8
    };
    let max_z = ((ab + 1).saturating_sub(min_bits)).min(3);

    let mut best = None;
    let mut best_cost = f64::INFINITY;

    for z in 0..=max_z {
        let eab_i = ab as i32 - z as i32;
        if eab_i < 0 || eab_i > 8 {
            continue;
        }
        let eab = eab_i as u8;
        for q in 0..16u8 {
            let m = rans::model(eab, q);
            let mut valid = true;
            let mut cost = 0.0f64;
            for (s, &cnt) in hist[..=(max_sym as usize)].iter().enumerate() {
                if cnt == 0 {
                    continue;
                }
                let (f, _) = m.forward(s as u16);
                if f == 0 {
                    valid = false;
                    break;
                }
                cost -= (cnt as f64) * (f as f64 / 65536.0).log2();
            }
            if valid && cost < best_cost {
                best_cost = cost;
                best = Some((z, q));
            }
        }
    }

    best.expect("valid (z, q) model for fragment")
}

/// Fast pure-Rust BC7 Mode 6 compressor (7-bit RGBA + shared p-bit pairs + 4-bit weights = 16 B/block).
pub fn compress_rgba_dims_to_bc7(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    const W4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
    let bx_count = width / 4;
    let by_count = height / 4;
    let mut out = vec![0u8; bx_count * by_count * 16];

    for by in 0..by_count {
        for bx in 0..bx_count {
            let mut px = [[0u8; 4]; 16];
            let mut min_c = [255.0f32; 4];
            let mut max_c = [0.0f32; 4];
            let mut mean = [0.0f32; 4];

            for py in 0..4 {
                for px_x in 0..4 {
                    let off = ((by * 4 + py) * width + (bx * 4 + px_x)) * 4;
                    let idx = py * 4 + px_x;
                    for c in 0..4 {
                        let v = rgba[off + c];
                        px[idx][c] = v;
                        let vf = v as f32;
                        min_c[c] = min_c[c].min(vf);
                        max_c[c] = max_c[c].max(vf);
                        mean[c] += vf * (1.0 / 16.0);
                    }
                }
            }

            let mut axis = [
                max_c[0] - min_c[0],
                max_c[1] - min_c[1],
                max_c[2] - min_c[2],
                max_c[3] - min_c[3],
            ];
            let norm0 = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2] + axis[3] * axis[3]).sqrt();
            if norm0 > 1e-4 {
                for c in 0..4 {
                    axis[c] /= norm0;
                }
                for _ in 0..2 {
                    let mut next = [0.0f32; 4];
                    for p in &px {
                        let d = [
                            p[0] as f32 - mean[0],
                            p[1] as f32 - mean[1],
                            p[2] as f32 - mean[2],
                            p[3] as f32 - mean[3],
                        ];
                        let proj = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2] + d[3] * axis[3];
                        for c in 0..4 {
                            next[c] += d[c] * proj;
                        }
                    }
                    let n2 = next[0] * next[0] + next[1] * next[1] + next[2] * next[2] + next[3] * next[3];
                    if n2 > 1e-6 {
                        let inv = 1.0 / n2.sqrt();
                        for c in 0..4 {
                            axis[c] = next[c] * inv;
                        }
                    }
                }
            } else {
                axis = [1.0, 0.0, 0.0, 0.0];
            }

            let mut min_t = f32::INFINITY;
            let mut max_t = f32::NEG_INFINITY;
            for p in &px {
                let d0 = p[0] as f32 - mean[0];
                let d1 = p[1] as f32 - mean[1];
                let d2 = p[2] as f32 - mean[2];
                let d3 = p[3] as f32 - mean[3];
                let t = d0 * axis[0] + d1 * axis[1] + d2 * axis[2] + d3 * axis[3];
                if t < min_t {
                    min_t = t;
                }
                if t > max_t {
                    max_t = t;
                }
            }

            let mut e0_f = [0i32; 4];
            let mut e1_f = [0i32; 4];
            for c in 0..4 {
                e0_f[c] = (mean[c] + min_t * axis[c]).round().clamp(0.0, 255.0) as i32;
                e1_f[c] = (mean[c] + max_t * axis[c]).round().clamp(0.0, 255.0) as i32;
            }

            let mut best_sse = i64::MAX;
            let mut best_q0 = [0u8; 4];
            let mut best_q1 = [0u8; 4];
            let mut best_p0 = 0u8;
            let mut best_p1 = 0u8;
            let mut best_idx = [0u8; 16];

            for p0 in 0..=1i32 {
                for p1 in 0..=1i32 {
                    let mut q0 = [0u8; 4];
                    let mut q1 = [0u8; 4];
                    let mut r0 = [0i32; 4];
                    let mut r1 = [0i32; 4];
                    for c in 0..4 {
                        let a7 = ((e0_f[c] - p0 + 1) >> 1).clamp(0, 127);
                        let b7 = ((e1_f[c] - p1 + 1) >> 1).clamp(0, 127);
                        q0[c] = a7 as u8;
                        q1[c] = b7 as u8;
                        r0[c] = (a7 << 1) | p0;
                        r1[c] = (b7 << 1) | p1;
                    }

                    let mut pal = [[0i32; 4]; 16];
                    for (k, &w) in W4.iter().enumerate() {
                        let iw = 64 - w;
                        for c in 0..4 {
                            pal[k][c] = (iw * r0[c] + w * r1[c] + 32) >> 6;
                        }
                    }

                    let mut sse = 0i64;
                    let mut idxs = [0u8; 16];
                    for i in 0..16 {
                        let p = px[i];
                        let mut bk = 0u8;
                        let mut be = i32::MAX;
                        for (k, col) in pal.iter().enumerate() {
                            let dr = p[0] as i32 - col[0];
                            let dg = p[1] as i32 - col[1];
                            let db = p[2] as i32 - col[2];
                            let da = p[3] as i32 - col[3];
                            let e = dr * dr + dg * dg + db * db + da * da;
                            if e < be {
                                be = e;
                                bk = k as u8;
                            }
                        }
                        idxs[i] = bk;
                        sse += be as i64;
                    }
                    if sse < best_sse {
                        best_sse = sse;
                        best_q0 = q0;
                        best_q1 = q1;
                        best_p0 = p0 as u8;
                        best_p1 = p1 as u8;
                        best_idx = idxs;
                    }
                }
            }

            if best_idx[0] >= 8 {
                std::mem::swap(&mut best_q0, &mut best_q1);
                std::mem::swap(&mut best_p0, &mut best_p1);
                for v in &mut best_idx {
                    *v = 15 - *v;
                }
            }

            let mut word: u128 = 1u128 << 6; // Mode 6 prefix 0b1000000
            word |= (best_q0[0] as u128) << 7;
            word |= (best_q1[0] as u128) << 14;
            word |= (best_q0[1] as u128) << 21;
            word |= (best_q1[1] as u128) << 28;
            word |= (best_q0[2] as u128) << 35;
            word |= (best_q1[2] as u128) << 42;
            word |= (best_q0[3] as u128) << 49;
            word |= (best_q1[3] as u128) << 56;
            word |= (best_p0 as u128) << 63;
            word |= (best_p1 as u128) << 64;
            word |= ((best_idx[0] & 7) as u128) << 65;
            for i in 1..16 {
                word |= ((best_idx[i] & 15) as u128) << (68 + (i - 1) * 4);
            }

            let bi = (by * bx_count + bx) * 16;
            out[bi..bi + 16].copy_from_slice(&word.to_le_bytes());
        }
    }
    out
}

#[inline]
fn encode_bc4_channel(vals: &[u8; 16]) -> u64 {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for &v in vals {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if hi == lo {
        return (hi as u64) | ((lo as u64) << 8);
    }

    let e0 = hi as i32;
    let e1 = lo as i32;
    let mut pal = [0i32; 8];
    pal[0] = e0;
    pal[1] = e1;
    for k in 0..6i32 {
        pal[2 + k as usize] = ((6 - k) * e0 + (1 + k) * e1 + 3) / 7;
    }

    let mut word = (hi as u64) | ((lo as u64) << 8);
    for (i, &v) in vals.iter().enumerate() {
        let vi = v as i32;
        let mut bk = 0u64;
        let mut be = i32::MAX;
        for (k, &pv) in pal.iter().enumerate() {
            let e = (vi - pv).abs();
            if e < be {
                be = e;
                bk = k as u64;
            }
        }
        word |= bk << (16 + i * 3);
    }
    word
}

/// Pure-Rust BC5 UNORM compressor (dual BC4 blocks for R & G normal channels = 16 B/block).
pub fn compress_rgba_dims_to_bc5(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let bx_count = width / 4;
    let by_count = height / 4;
    let mut out = vec![0u8; bx_count * by_count * 16];
    for by in 0..by_count {
        for bx in 0..bx_count {
            let mut r_vals = [0u8; 16];
            let mut g_vals = [0u8; 16];
            for py in 0..4 {
                for px in 0..4 {
                    let off = ((by * 4 + py) * width + (bx * 4 + px)) * 4;
                    let idx = py * 4 + px;
                    r_vals[idx] = rgba[off];
                    g_vals[idx] = rgba[off + 1];
                }
            }
            let r_blk = encode_bc4_channel(&r_vals);
            let g_blk = encode_bc4_channel(&g_vals);
            let bi = (by * bx_count + bx) * 16;
            out[bi..bi + 8].copy_from_slice(&r_blk.to_le_bytes());
            out[bi + 8..bi + 16].copy_from_slice(&g_blk.to_le_bytes());
        }
    }
    out
}

/// Encodes 16-byte block streams (BC7 / BC5) into 16-lane rANS BCPack streams (2 symbol-rows/block).
pub fn encode_bc16_to_bcpack(bc16_bytes: &[u8]) -> Vec<u8> {
    assert_eq!(bc16_bytes.len() % 16, 0);
    let rows = (bc16_bytes.len() / 16) * 2;
    let mut lane_syms: [Vec<u16>; 16] = std::array::from_fn(|i| {
        if matches!(i, 2 | 3 | 4 | 5 | 6 | 9 | 11 | 12 | 13 | 15) {
            Vec::with_capacity(rows)
        } else {
            vec![0u16; rows]
        }
    });

    for half in bc16_bytes.chunks_exact(8) {
        let w0 = u32::from_le_bytes([half[0], half[1], half[2], half[3]]);
        let w1 = u32::from_le_bytes([half[4], half[5], half[6], half[7]]);
        lane_syms[2].push((w0 & 0x7F) as u16);
        lane_syms[3].push(((w0 >> 7) & 0x7F) as u16);
        lane_syms[4].push(((w0 >> 14) & 0x7F) as u16);
        lane_syms[5].push(((w0 >> 21) & 0x7F) as u16);
        lane_syms[9].push(((w0 >> 28) & 0x0F) as u16);

        lane_syms[6].push((w1 & 0x7F) as u16);
        lane_syms[11].push(((w1 >> 7) & 0x7F) as u16);
        lane_syms[12].push(((w1 >> 14) & 0x7F) as u16);
        lane_syms[13].push(((w1 >> 21) & 0x7F) as u16);
        lane_syms[15].push(((w1 >> 28) & 0x0F) as u16);
    }

    let mut lanes: Vec<Vec<Fragment>> = Vec::with_capacity(16);
    for (li, &ab) in BC7_ALPHABET_BITS.iter().enumerate() {
        let mut frags = Vec::new();
        for chunk in lane_syms[li].chunks(4096) {
            let (z, q) = best_model_for_fragment(ab, chunk);
            frags.push(Fragment {
                symbols: chunk.to_vec(),
                q,
                z,
            });
        }
        lanes.push(frags);
    }

    encode_lanes(&BC7_ALPHABET_BITS, &lanes).expect("encode_bc16_to_bcpack encode_lanes failed")
}

/// CPU baseline decoder for 16-byte (BC7 / BC5) 16-lane rANS BCPack streams.
pub fn decode_bc16_from_bcpack_cpu(stream: &[u8], blocks: usize) -> Vec<u8> {
    let rows = blocks * 2;
    let decoded = rans::decode_slice(stream, &BC7_ALPHABET_BITS, rows).expect("cpu bc16 rans decode");
    let mut out = vec![0u8; blocks * 16];
    let l2 = decoded.lane(2);
    let l3 = decoded.lane(3);
    let l4 = decoded.lane(4);
    let l5 = decoded.lane(5);
    let l6 = decoded.lane(6);
    let l9 = decoded.lane(9);
    let l11 = decoded.lane(11);
    let l12 = decoded.lane(12);
    let l13 = decoded.lane(13);
    let l15 = decoded.lane(15);

    for r in 0..rows {
        let w0 = (l2[r] as u32 & 0x7F)
            | ((l3[r] as u32 & 0x7F) << 7)
            | ((l4[r] as u32 & 0x7F) << 14)
            | ((l5[r] as u32 & 0x7F) << 21)
            | ((l9[r] as u32 & 0x0F) << 28);
        let w1 = (l6[r] as u32 & 0x7F)
            | ((l11[r] as u32 & 0x7F) << 7)
            | ((l12[r] as u32 & 0x7F) << 14)
            | ((l13[r] as u32 & 0x7F) << 21)
            | ((l15[r] as u32 & 0x0F) << 28);
        let off = r * 8;
        out[off..off + 4].copy_from_slice(&w0.to_le_bytes());
        out[off + 4..off + 8].copy_from_slice(&w1.to_le_bytes());
    }
    out
}

pub fn encode_bc1_to_bcpack(raw_bc1: &[u8], window: &Window, quads: &QuadTable) -> Vec<u8> {
    assert_eq!(raw_bc1.len() % 8, 0);
    let n = raw_bc1.len() / 8;

    let mut inv_quads: HashMap<[u8; 4], u16> = HashMap::with_capacity(256);
    for (sym, &codes) in quads.codes.iter().enumerate() {
        inv_quads.insert(codes, sym as u16);
    }

    let mut lane_syms: [Vec<u16>; 12] = std::array::from_fn(|_| Vec::with_capacity(n));
    let mut fields: Vec<[u8; 6]> = Vec::with_capacity(n);

    for i in 0..n {
        let o = &raw_bc1[i * 8..i * 8 + 8];
        let c0 = u16::from_le_bytes([o[0], o[1]]);
        let c1 = u16::from_le_bytes([o[2], o[3]]);
        let w = u32::from_le_bytes([o[4], o[5], o[6], o[7]]);

        let target = [
            ((c0 >> 11) & 31) as u8,
            ((c0 >> 5) & 63) as u8,
            (c0 & 31) as u8,
            ((c1 >> 11) & 31) as u8,
            ((c1 >> 5) & 63) as u8,
            (c1 & 31) as u8,
        ];

        let mut best_d = 0usize;
        let mut best_score = i32::MAX;
        for d in 0..=32usize {
            let pred = if d == 0 {
                [0u8; 6]
            } else if d <= i {
                fields[i - d]
            } else {
                let slot = (i as isize - d as isize).rem_euclid(256) as usize;
                if !window.fitted[slot] {
                    continue;
                }
                window.slots[slot]
            };
            let mut score = if d == 0 { 1 } else { 0 };
            for k in 0..6 {
                let r = wrap_residual(target[k] as i32 - pred[k] as i32, ENDPOINT_WIDTHS[k]);
                score += r.abs();
            }
            if score < best_score {
                best_score = score;
                best_d = d;
            }
        }

        let pred = if best_d == 0 {
            [0u8; 6]
        } else if best_d <= i {
            fields[i - best_d]
        } else {
            let slot = (i as isize - best_d as isize).rem_euclid(256) as usize;
            window.slots[slot]
        };

        lane_syms[0].push(encode_back_distance(best_d));
        lane_syms[1].push(0);
        for k in 0..6 {
            let r = wrap_residual(target[k] as i32 - pred[k] as i32, ENDPOINT_WIDTHS[k]);
            lane_syms[2 + k].push(zigzag(r));
        }

        const QUAD_ORIGIN: [(u32, u32); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];
        const GROUP_PIXEL: [(u32, u32); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
        for (q, &(ox, oy)) in QUAD_ORIGIN.iter().enumerate() {
            let mut codes = [0u8; 4];
            for (k, &(dx, dy)) in GROUP_PIXEL.iter().enumerate() {
                let shift = 2 * (4 * (oy + dy) + (ox + dx));
                codes[k] = ((w >> shift) & 3) as u8;
            }
            lane_syms[8 + q].push(inv_quads[&codes]);
        }

        fields.push(target);
    }

    let mut lanes: Vec<Vec<Fragment>> = Vec::with_capacity(12);
    for (li, &ab) in BC1_ALPHABET_BITS.iter().enumerate() {
        let mut frags = Vec::new();
        for chunk in lane_syms[li].chunks(4096) {
            let (z, q) = best_model_for_fragment(ab, chunk);
            frags.push(Fragment {
                symbols: chunk.to_vec(),
                q,
                z,
            });
        }
        lanes.push(frags);
    }

    let stream = encode_lanes(&BC1_ALPHABET_BITS, &lanes).expect("encode_lanes failed");
    let check = bc1::decode(&stream, n, window, quads).expect("bcpack self-decode failed");
    assert_eq!(check.blocks, raw_bc1, "encoded stream did not round-trip bit-exact!");
    stream
}

pub fn load_or_generate_assets(
    assets_dir: &Path,
    window: &Window,
    quads: &QuadTable,
    force_regen: bool,
) -> Vec<TextureAsset> {
    fs::create_dir_all(assets_dir).expect("create assets dir");

    ASSET_NAMES
        .par_iter()
        .enumerate()
        .map(|(idx, &name)| {
            let bcpack_path: PathBuf = assets_dir.join(format!("{name}.bcpack"));
            let raw_path: PathBuf = assets_dir.join(format!("{name}.bc1"));
            let tile_paths: [PathBuf; TILE_STRIPS] =
                std::array::from_fn(|t| assets_dir.join(format!("{name}_t{t}.bcpack")));

            let all_exist = !force_regen
                && bcpack_path.exists()
                && raw_path.exists()
                && tile_paths.iter().all(|p| p.exists());

            if all_exist {
                let bcpack_bytes = fs::read(&bcpack_path).expect("read .bcpack");
                let expected_bc1 = fs::read(&raw_path).expect("read .bc1");
                if expected_bc1.len() == TEX_BC1_BYTES {
                    let bcpack_tiles: [Vec<u8>; TILE_STRIPS] =
                        std::array::from_fn(|t| fs::read(&tile_paths[t]).expect("read tile .bcpack"));
                    return TextureAsset {
                        name,
                        bcpack_bytes,
                        bcpack_tiles,
                        expected_bc1,
                    };
                }
            }

            let rgba = generate_rgba(idx);
            let expected_bc1 = compress_rgba_to_bc1(&rgba);
            let bcpack_bytes = encode_bc1_to_bcpack(&expected_bc1, window, quads);
            let bcpack_tiles: [Vec<u8>; TILE_STRIPS] = std::array::from_fn(|t| {
                let slice = &expected_bc1[t * TILE_BC1_BYTES..(t + 1) * TILE_BC1_BYTES];
                encode_bc1_to_bcpack(slice, window, quads)
            });

            fs::write(&bcpack_path, &bcpack_bytes).expect("write .bcpack");
            fs::write(&raw_path, &expected_bc1).expect("write .bc1");
            for t in 0..TILE_STRIPS {
                fs::write(&tile_paths[t], &bcpack_tiles[t]).expect("write tile .bcpack");
            }
            println!(
                "[assets] Generated {}: 131072 B BC1 -> {} B mono ({:.2}x), {} B 4x4K tiles",
                name,
                bcpack_bytes.len(),
                expected_bc1.len() as f64 / bcpack_bytes.len() as f64,
                bcpack_tiles.iter().map(|v| v.len()).sum::<usize>()
            );

            TextureAsset {
                name,
                bcpack_bytes,
                bcpack_tiles,
                expected_bc1,
            }
        })
        .collect()
}

/// Software AMD Gfx10 (RDNA2) linear -> swizzled encoder (exact inverse of gfx10_detile.hlsl).
/// `tile_mode`: 0 = SW_64KB_D (8B), 1 = SW_4KB_S (8B), 2 = SW_64KB_D (16B).
pub fn swizzle_gfx10_cpu(
    linear: &[u8],
    width_blocks: usize,
    height_blocks: usize,
    tile_mode: u32,
    block_bytes: usize,
) -> Vec<u8> {
    let (pattern, tw, th) = match tile_mode {
        0 => (b"xyxxyyxyxyxyx".as_slice(), 128usize, 64usize),
        1 => (b"xyyxxyxyx".as_slice(), 32usize, 16usize),
        _ => (b"xyxxyyxyxyxy".as_slice(), 64usize, 64usize),
    };
    let tl = pattern.len();
    let tiles_x = (width_blocks + tw - 1) / tw;
    let tiles_y = (height_blocks + th - 1) / th;
    let total_swizzled_blocks = tiles_x.max(1) * tiles_y.max(1) * (1usize << tl);
    let mut swizzled = vec![0u8; total_swizzled_blocks * block_bytes];

    for i in 0..total_swizzled_blocks {
        let mut lx = 0usize;
        let mut ly = 0usize;
        let mut xi = 0usize;
        let mut yi = 0usize;
        for (bit, &axis) in pattern.iter().enumerate() {
            let v = (i >> bit) & 1;
            if axis == b'x' {
                lx |= v << xi;
                xi += 1;
            } else {
                ly |= v << yi;
                yi += 1;
            }
        }
        let k = i >> tl;
        let tpr = tiles_x.max(1);
        let x = (k % tpr) * tw + lx;
        let y = (k / tpr) * th + ly;
        if x < width_blocks && y < height_blocks {
            let lin_idx = y * width_blocks + x;
            swizzled[i * block_bytes..(i + 1) * block_bytes]
                .copy_from_slice(&linear[lin_idx * block_bytes..(lin_idx + 1) * block_bytes]);
        }
    }
    swizzled
}

pub fn downsample_rgba_box2(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let nw = (w / 2).max(4);
    let nh = (h / 2).max(4);
    let mut dst = vec![0u8; nw * nh * 4];
    for y in 0..nh {
        for x in 0..nw {
            let sx = (x * 2).min(w - 2);
            let sy = (y * 2).min(h - 2);
            for c in 0..4 {
                let s0 = src[(sy * w + sx) * 4 + c] as u32;
                let s1 = src[(sy * w + sx + 1) * 4 + c] as u32;
                let s2 = src[((sy + 1) * w + sx) * 4 + c] as u32;
                let s3 = src[((sy + 1) * w + sx + 1) * 4 + c] as u32;
                dst[(y * nw + x) * 4 + c] = ((s0 + s1 + s2 + s3 + 2) / 4) as u8;
            }
        }
    }
    dst
}

pub struct MipLevelEncoded {
    pub label: String,
    pub blocks: usize,
    pub bcpack: Vec<u8>,
    pub expected_bc1: Vec<u8>,
}

/// Builds a console-style mipmap chain from 512x512 down to a Packed Mip Tail (32x32+16x16+8x8+4x4 = 85 blocks).
pub fn generate_mip_chain_bc1(tex_idx: usize, window: &Window, quads: &QuadTable) -> Vec<MipLevelEncoded> {
    let mut out = Vec::new();
    let mut rgba = generate_rgba(tex_idx);
    let mut w = TEX_W;
    let mut h = TEX_H;

    // Individual major mips: 512x512, 256x256, 128x128, 64x64
    for mip in 0..4 {
        let raw = compress_rgba_dims_to_bc1(&rgba, w, h);
        let enc = encode_bc1_to_bcpack(&raw, window, quads);
        out.push(MipLevelEncoded {
            label: format!("mip{mip}_{w}x{h}"),
            blocks: raw.len() / 8,
            bcpack: enc,
            expected_bc1: raw,
        });
        rgba = downsample_rgba_box2(&rgba, w, h);
        w /= 2;
        h /= 2;
    }

    // Packed Mip Tail: 32x32 (64 blks) + 16x16 (16 blks) + 8x8 (4 blks) + 4x4 (1 blk) = 85 blocks in one chunk
    let mut tail_bc1 = Vec::new();
    loop {
        let raw = compress_rgba_dims_to_bc1(&rgba, w, h);
        tail_bc1.extend_from_slice(&raw);
        if w <= 4 || h <= 4 {
            break;
        }
        rgba = downsample_rgba_box2(&rgba, w, h);
        w /= 2;
        h /= 2;
    }
    let tail_enc = encode_bc1_to_bcpack(&tail_bc1, window, quads);
    out.push(MipLevelEncoded {
        label: "packed_mip_tail_32_to_4".to_string(),
        blocks: tail_bc1.len() / 8,
        bcpack: tail_enc,
        expected_bc1: tail_bc1,
    });
    out
}

/// Generates a high-entropy tangent-space normal-map stress tile (4,096 blocks = 512x128 strip).
pub fn generate_high_entropy_bc1_normal_map(
    blocks: usize,
    seed: u64,
    window: &Window,
    quads: &QuadTable,
) -> (Vec<u8>, Vec<u8>) {
    let w = 512usize;
    let h = ((blocks * 16) / w).max(4);
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let h1 = hash2((x as u32 ^ (seed as u32)) as i32, (y as u32 ^ ((seed >> 16) as u32)) as i32);
            let h2 = hash2((x as u32).wrapping_mul(3) as i32, (y as u32).wrapping_add(seed as u32) as i32);
            let nx = ((h1 & 0xFF) as f32 / 255.0 - 0.5) * 1.6;
            let ny = (((h2 >> 8) & 0xFF) as f32 / 255.0 - 0.5) * 1.6;
            let nz = (1.0 - (nx * nx + ny * ny).min(0.85)).sqrt();
            let px = rgb(nx * 127.0 + 128.0, ny * 127.0 + 128.0, nz * 127.0 + 128.0);
            let off = (y * w + x) * 4;
            rgba[off..off + 4].copy_from_slice(&px);
        }
    }
    let raw = compress_rgba_dims_to_bc1(&rgba, w, h);
    let enc = encode_bc1_to_bcpack(&raw, window, quads);
    (enc, raw)
}

/// Generates a sharp high-contrast UI/HUD grid stress tile (4,096 blocks = 512x128 strip).
pub fn generate_sharp_ui_bc1(
    blocks: usize,
    seed: u64,
    window: &Window,
    quads: &QuadTable,
) -> (Vec<u8>, Vec<u8>) {
    let w = 512usize;
    let h = ((blocks * 16) / w).max(4);
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let border = (x % 16 == 0) || (y % 16 == 0);
            let diag = ((x ^ y) + seed as usize) % 13 == 0;
            let px = if border {
                [0, 255, 240, 255]
            } else if diag {
                [255, 40, 200, 255]
            } else if ((x / 8) ^ (y / 8)) & 1 == 0 {
                [248, 248, 252, 255]
            } else {
                [12, 16, 28, 255]
            };
            let off = (y * w + x) * 4;
            rgba[off..off + 4].copy_from_slice(&px);
        }
    }
    let raw = compress_rgba_dims_to_bc1(&rgba, w, h);
    let enc = encode_bc1_to_bcpack(&raw, window, quads);
    (enc, raw)
}

/// Generates a valid clean-room 16-lane BC7 rANS stream (`out_target = bc7_blocks * 2` symbols per lane).
pub fn generate_bc7_16lane_stream(
    seed: u64,
    bc7_blocks: usize,
    profile: Bc7EntropyProfile,
) -> (Vec<u8>, usize) {
    let out_target = bc7_blocks * 2;
    let mut lane_syms: [Vec<u16>; 16] = std::array::from_fn(|_| Vec::with_capacity(out_target));

    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xBF58476D1CE4E5B9;
    let mut next_u32 = || -> u32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 32) as u32
    };

    for row in 0..out_target {
        for (li, &ab) in BC7_ALPHABET_BITS.iter().enumerate() {
            let max_val = (1u32 << ab) - 1;
            let r = next_u32();
            let sym = match profile {
                Bc7EntropyProfile::StandardAlbedo => {
                    // Concentrated near low residuals
                    let s = ((r & max_val) >> 2).min(max_val);
                    s as u16
                }
                Bc7EntropyProfile::NormalHighEntropy => {
                    // Wide uniform-like spread across almost the full alphabet width
                    (r % (max_val + 1)) as u16
                }
                Bc7EntropyProfile::SharpUi => {
                    // Bimodal jumps between 0 and upper half of alphabet
                    if (row + li) % 5 == 0 {
                        (max_val - (r & 3).min(max_val)) as u16
                    } else {
                        (r & 3).min(max_val) as u16
                    }
                }
            };
            lane_syms[li].push(sym);
        }
    }

    let mut lanes: Vec<Vec<Fragment>> = Vec::with_capacity(16);
    for (li, &ab) in BC7_ALPHABET_BITS.iter().enumerate() {
        let mut frags = Vec::new();
        for chunk in lane_syms[li].chunks(4096) {
            let (z, q) = best_model_for_fragment(ab, chunk);
            frags.push(Fragment {
                symbols: chunk.to_vec(),
                q,
                z,
            });
        }
        lanes.push(frags);
    }

    let stream = encode_lanes(&BC7_ALPHABET_BITS, &lanes).expect("BC7 encode_lanes failed");
    (stream, out_target)
}
