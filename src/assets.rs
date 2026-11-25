use bcpack::bc1::{self, QuadTable, Window, BC1_ALPHABET_BITS, ENDPOINT_WIDTHS};
use bcpack::encode::{encode_lanes, Fragment};
use bcpack::rans;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const TEX_W: usize = 512;
pub const TEX_H: usize = 512;
pub const BLOCKS_X: usize = TEX_W / 4; // 128
pub const BLOCKS_Y: usize = TEX_H / 4; // 128
pub const NUM_BLOCKS: usize = BLOCKS_X * BLOCKS_Y; // 16384
pub const TEX_BC1_BYTES: usize = NUM_BLOCKS * 8; // 131072

pub const ASSET_NAMES: [&str; 8] = [
    "01_asphalt_road",
    "02_neon_facade",
    "03_pastel_hotel",
    "04_brick_plaza",
    "05_ocean_water",
    "06_palm_canopy",
    "07_synthwave_billboard",
    "08_calibration_checker",
];

pub struct TextureAsset {
    pub name: &'static str,
    pub bcpack_bytes: Vec<u8>,
    pub expected_bc1: Vec<u8>,
}

#[inline]
fn hash2(x: i32, y: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(374761393).wrapping_add((y as u32).wrapping_mul(668265263));
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
                // 01_asphalt_road: dark asphalt + double yellow center lines + white edge lines
                0 => {
                    let g = n * 24.0;
                    let mut col = [34.0 + g, 36.0 + g, 44.0 + g];
                    // Center double yellow stripes
                    if (u - 0.475).abs() < 0.012 || (u - 0.525).abs() < 0.012 {
                        col = [235.0 + g * 0.4, 185.0 + g * 0.4, 30.0];
                    }
                    // Dashed lane dividers
                    let dash = ((v * 8.0).fract()) < 0.55;
                    if dash && ((u - 0.24).abs() < 0.01 || (u - 0.76).abs() < 0.01) {
                        col = [230.0, 235.0, 245.0];
                    }
                    // Sidewalk curb edge
                    if u < 0.04 || u > 0.96 {
                        col = [120.0 + g, 115.0 + g, 128.0 + g];
                    }
                    rgb(col[0], col[1], col[2])
                }
                // 02_neon_facade: dark skyscraper glass + glowing cyan & magenta Art-Deco neon trims
                1 => {
                    let gx = (x % 64) as f32 / 64.0;
                    let gy = (y % 64) as f32 / 64.0;
                    let cell_id = hash2((x / 64) as i32, (y / 64) as i32);
                    let mut col = [18.0, 20.0, 42.0];
                    if gx > 0.14 && gx < 0.86 && gy > 0.16 && gy < 0.82 {
                        let lit = (cell_id & 7) < 3;
                        if lit {
                            let warm = (cell_id & 8) != 0;
                            col = if warm {
                                [255.0 - gy * 50.0, 195.0 - gy * 40.0, 110.0]
                            } else {
                                [70.0, 220.0 - gy * 40.0, 255.0]
                            };
                        } else {
                            col = [28.0 + gy * 25.0, 38.0 + gy * 30.0, 75.0 + gy * 35.0];
                        }
                    }
                    // Neon horizontal strips
                    if (y % 128) < 6 {
                        col = [0.0, 245.0, 255.0];
                    } else if ((y + 64) % 128) < 5 {
                        col = [255.0, 40.0, 190.0];
                    }
                    // Vertical Art-Deco neon pilaster
                    if x < 10 || x > TEX_W - 10 {
                        col = [255.0, 60.0, 200.0];
                    }
                    rgb(col[0] + n * 6.0, col[1] + n * 6.0, col[2] + n * 6.0)
                }
                // 03_pastel_hotel: Miami Art-Deco coral/teal stucco with lit windows
                2 => {
                    let mut col = [
                        245.0 - v * 35.0 + n * 10.0,
                        152.0 - v * 25.0 + n * 10.0,
                        148.0 + v * 30.0 + n * 10.0,
                    ];
                    // Teal horizontal deco bands
                    if (y % 128) < 12 || (y % 128) > 120 {
                        col = [45.0, 205.0, 195.0];
                    }
                    let wx = (x % 64) as f32 / 64.0;
                    let wy = (y % 128) as f32 / 128.0;
                    if wx > 0.22 && wx < 0.78 && wy > 0.22 && wy < 0.78 {
                        let glow = ((hash2((x / 64) as i32, (y / 128) as i32)) & 1) == 0;
                        col = if glow { [255.0, 225.0, 140.0] } else { [35.0, 55.0, 82.0] };
                    }
                    if wx > 0.16 && wx < 0.84 && wy > 0.78 && wy < 0.85 {
                        col = [250.0, 245.0, 240.0]; // window sill
                    }
                    rgb(col[0], col[1], col[2])
                }
                // 04_brick_plaza: warm sandstone / terracotta sidewalk tiles
                3 => {
                    let row = y / 32;
                    let sx = (x + if row % 2 == 1 { 32 } else { 0 }) % 64;
                    let sy = y % 32;
                    let grout = sx < 3 || sy < 3;
                    if grout {
                        rgb(68.0 + n * 12.0, 58.0 + n * 12.0, 64.0 + n * 12.0)
                    } else {
                        let cid = hash2(((x + if row % 2 == 1 { 32 } else { 0 }) / 64) as i32, row as i32);
                        let tint = ((cid & 0x1F) as f32) - 16.0;
                        rgb(
                            205.0 + tint + n * 14.0,
                            135.0 + tint * 0.7 + n * 12.0,
                            108.0 + tint * 0.5 + n * 12.0,
                        )
                    }
                }
                // 05_ocean_water: turquoise Caribbean waves + pink/gold sunset specular
                4 => {
                    let w1 = (u * 18.0 + (v * 12.0).sin() * 1.4).sin();
                    let w2 = (v * 34.0 - u * 10.0).cos();
                    let wave = (w1 * 0.55 + w2 * 0.45) * 0.5 + 0.5;
                    let mut col = [
                        lerp_u8(12.0, 48.0, wave) as f32,
                        lerp_u8(95.0, 210.0, wave) as f32,
                        lerp_u8(140.0, 235.0, wave) as f32,
                    ];
                    // Sunset highlight crest
                    if wave > 0.83 {
                        let t = (wave - 0.83) / 0.17;
                        col[0] += t * 210.0;
                        col[1] += t * 65.0;
                        col[2] += t * 20.0;
                    }
                    rgb(col[0] + n * 6.0, col[1] + n * 6.0, col[2] + n * 6.0)
                }
                // 06_palm_canopy: lush tropical palm fronds
                5 => {
                    let dx = u - 0.5;
                    let dy = v - 0.5;
                    let ang = dy.atan2(dx);
                    let r = (dx * dx + dy * dy).sqrt();
                    let frond = ((ang * 7.0).sin() * 0.5 + 0.5) * (1.0 - r * 0.7);
                    let rib = ((r * 45.0).sin().abs()) * 0.25;
                    let g_base = 85.0 + frond * 145.0 + rib * 40.0 + n * 14.0;
                    rgb(25.0 + frond * 55.0 + n * 8.0, g_base, 35.0 + frond * 40.0)
                }
                // 07_synthwave_billboard: classic retro Vice-City sunset disc + perspective neon grid
                6 => {
                    let dx = u - 0.5;
                    let dy = v - 0.42;
                    let r = (dx * dx + dy * dy).sqrt();
                    let mut col = [22.0, 8.0, 48.0];
                    // Horizon glow
                    if v > 0.56 {
                        let hz = (v - 0.56) / 0.44;
                        let gx = ((u - 0.5) / (0.15 + hz * 1.4) * 8.0).fract().abs();
                        let gy = (hz * 10.0).fract().abs();
                        if gx < 0.08 || gy < 0.09 {
                            col = [255.0, 45.0, 210.0];
                        } else {
                            col = [26.0, 12.0, 56.0];
                        }
                    } else if r < 0.30 {
                        // Striped sunset sun disc
                        let band = ((v * 28.0).sin() > -0.35) || v < 0.30;
                        if band {
                            let t = ((v - 0.12) / 0.30).clamp(0.0, 1.0);
                            col = [255.0, 235.0 - t * 185.0, 50.0 + t * 140.0];
                        }
                    } else {
                        // Starfield
                        let star = (hash2((x / 2) as i32, (y / 2) as i32) & 255) == 0;
                        if star {
                            col = [210.0, 245.0, 255.0];
                        }
                    }
                    // Cyan border frame
                    if x < 12 || x > TEX_W - 12 || y < 12 || y > TEX_H - 12 {
                        col = [0.0, 245.0, 255.0];
                    }
                    rgb(col[0], col[1], col[2])
                }
                // 08_calibration_checker: high-contrast RGB/CMY quadrants + crisp checkerboard
                _ => {
                    let cx = (x / 32) % 2;
                    let cy = (y / 32) % 2;
                    let chk = (cx ^ cy) == 0;
                    let qx = x < TEX_W / 2;
                    let qy = y < TEX_H / 2;
                    let base = match (qx, qy) {
                        (true, true) => [255.0, 55.0, 85.0],
                        (false, true) => [40.0, 240.0, 255.0],
                        (true, false) => [255.0, 220.0, 40.0],
                        (false, false) => [170.0, 75.0, 255.0],
                    };
                    if x % 128 < 4 || y % 128 < 4 {
                        rgb(255.0, 255.0, 255.0)
                    } else if chk {
                        rgb(base[0], base[1], base[2])
                    } else {
                        rgb(base[0] * 0.16, base[1] * 0.16, base[2] * 0.16)
                    }
                }
            };
            let off = (y * TEX_W + x) * 4;
            img[off..off + 4].copy_from_slice(&px);
        }
    }
    img
}

#[inline]
fn to_565(r: u8, g: u8, b: u8) -> (u8, u8, u8, u16) {
    let r5 = (r as u16 * 31 + 127) / 255;
    let g6 = (g as u16 * 63 + 127) / 255;
    let b5 = (b as u16 * 31 + 127) / 255;
    (
        r5 as u8,
        g6 as u8,
        b5 as u8,
        (r5 << 11) | (g6 << 5) | b5,
    )
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

fn compress_rgba_to_bc1(rgba: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; TEX_BC1_BYTES];
    for by in 0..BLOCKS_Y {
        for bx in 0..BLOCKS_X {
            let mut block_px = [[0u8; 3]; 16];
            let mut min_rgb = [255u8; 3];
            let mut max_rgb = [0u8; 3];
            for py in 0..4 {
                for px in 0..4 {
                    let off = ((by * 4 + py) * TEX_W + (bx * 4 + px)) * 4;
                    let p = [rgba[off], rgba[off + 1], rgba[off + 2]];
                    block_px[py * 4 + px] = p;
                    for c in 0..3 {
                        min_rgb[c] = min_rgb[c].min(p[c]);
                        max_rgb[c] = max_rgb[c].max(p[c]);
                    }
                }
            }

            // Inset bounding box slightly for cleaner 4-color BC1 fit
            let mut c_hi = [0u8; 3];
            let mut c_lo = [0u8; 3];
            for c in 0..3 {
                let inset = (max_rgb[c] - min_rgb[c]) >> 4;
                c_hi[c] = max_rgb[c].saturating_sub(inset);
                c_lo[c] = min_rgb[c].saturating_add(inset);
            }

            let (_, _, _, mut c0) = to_565(c_hi[0], c_hi[1], c_hi[2]);
            let (_, _, _, mut c1) = to_565(c_lo[0], c_lo[1], c_lo[2]);
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

            let bi = (by * BLOCKS_X + bx) * 8;
            out[bi..bi + 2].copy_from_slice(&c0.to_le_bytes());
            out[bi + 2..bi + 4].copy_from_slice(&c1.to_le_bytes());
            out[bi + 4..bi + 8].copy_from_slice(&idx_word.to_le_bytes());
        }
    }
    out
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
    let max_sym = syms.iter().copied().max().unwrap_or(0);
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
            for &s in syms {
                let (f, _) = m.forward(s);
                if f == 0 {
                    valid = false;
                    break;
                }
                cost -= (f as f64 / 65536.0).log2();
            }
            if valid && cost < best_cost {
                best_cost = cost;
                best = Some((z, q));
            }
        }
    }

    best.expect("valid (z, q) model for fragment")
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

        // Evaluate d = 0 and d in 1..=32 (including preset window pre-start refs)
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
    let mut result = Vec::with_capacity(ASSET_NAMES.len());

    for (idx, &name) in ASSET_NAMES.iter().enumerate() {
        let bcpack_path: PathBuf = assets_dir.join(format!("{name}.bcpack"));
        let raw_path: PathBuf = assets_dir.join(format!("{name}.bc1"));

        if !force_regen && bcpack_path.exists() && raw_path.exists() {
            let bcpack_bytes = fs::read(&bcpack_path).expect("read .bcpack");
            let expected_bc1 = fs::read(&raw_path).expect("read .bc1");
            if expected_bc1.len() == TEX_BC1_BYTES {
                result.push(TextureAsset {
                    name,
                    bcpack_bytes,
                    expected_bc1,
                });
                continue;
            }
        }

        println!("[assets] Generating procedural 512x512 texture & encoding BCPack: {name} ...");
        let rgba = generate_rgba(idx);
        let expected_bc1 = compress_rgba_to_bc1(&rgba);
        let bcpack_bytes = encode_bc1_to_bcpack(&expected_bc1, window, quads);

        fs::write(&bcpack_path, &bcpack_bytes).expect("write .bcpack");
        fs::write(&raw_path, &expected_bc1).expect("write .bc1");
        println!(
            "[assets]   {name}: {} bytes raw BC1 -> {} bytes BCPack ({:.2}x compression)",
            expected_bc1.len(),
            bcpack_bytes.len(),
            expected_bc1.len() as f64 / bcpack_bytes.len() as f64
        );

        result.push(TextureAsset {
            name,
            bcpack_bytes,
            expected_bc1,
        });
    }

    result
}
