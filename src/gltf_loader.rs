use crate::assets::{
    compress_rgba_dims_to_bc1, downsample_rgba_box2, encode_bc1_to_bcpack, TILE_BC1_BYTES,
    TILE_BLOCKS,
};
use crate::gpu_decoder::{pack_batch, BcpackGpuDecoder, GpuContext};
use bcpack::bc1::{QuadTable, Window};
use gltf::image::Format as GltfImgFmt;
use gltf::material::AlphaMode;
use image::imageops::FilterType;
use image::{ImageBuffer, Rgba, RgbaImage};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PbrVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
}

pub struct PbrSubmesh {
    pub name: String,
    pub index_offset: u32,
    pub index_count: u32,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub is_glass: bool,
    pub albedo_tex: Option<usize>,
    pub normal_tex: Option<usize>,
    pub orm_tex: Option<usize>,
    pub emissive_tex: Option<usize>,
}

pub struct BcpackMipChunk {
    pub mip_level: u32,
    pub dim: u32,
    pub strip_y: u32,
    pub strip_h: u32,
    pub is_packed_tail: bool,
    pub blocks: usize,
    pub bcpack: Vec<u8>,
    pub expected_bc1: Vec<u8>,
}

pub struct BcpackTexture2D {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub strip_height: u32,
    pub num_strips: usize,
    pub mip_levels: u32,
    pub bcpack_tiles: Vec<Vec<u8>>,
    pub expected_bc1: Vec<u8>,
    pub lower_mips: Vec<BcpackMipChunk>,
}

pub struct LoadedGlbModel {
    pub vertices: Vec<PbrVertex>,
    pub indices: Vec<u32>,
    pub submeshes: Vec<PbrSubmesh>,
    pub textures: Vec<BcpackTexture2D>,
    pub raw_bounds_min: [f32; 3],
    pub raw_bounds_max: [f32; 3],
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

#[inline]
fn mul_mat4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

#[inline]
fn xform_pos(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

fn normal_matrix_3x3(m: &[[f32; 4]; 4]) -> [[f32; 3]; 3] {
    let a00 = m[0][0]; let a01 = m[1][0]; let a02 = m[2][0];
    let a10 = m[0][1]; let a11 = m[1][1]; let a12 = m[2][1];
    let a20 = m[0][2]; let a21 = m[1][2]; let a22 = m[2][2];

    let c00 = a11 * a22 - a12 * a21;
    let c01 = -(a10 * a22 - a12 * a20);
    let c02 = a10 * a21 - a11 * a20;
    let c10 = -(a01 * a22 - a02 * a21);
    let c11 = a00 * a22 - a02 * a20;
    let c12 = -(a00 * a21 - a01 * a20);
    let c20 = a01 * a12 - a02 * a11;
    let c21 = -(a00 * a12 - a02 * a10);
    let c22 = a00 * a11 - a01 * a10;

    let det = a00 * c00 + a01 * c01 + a02 * c02;
    let inv = if det.abs() > 1e-10 { 1.0 / det } else { 1.0 };
    // Transpose of cofactor/det = inverse-transpose stored by rows
    [
        [c00 * inv, c01 * inv, c02 * inv],
        [c10 * inv, c11 * inv, c12 * inv],
        [c20 * inv, c21 * inv, c22 * inv],
    ]
}

#[inline]
fn xform_dir(nm: &[[f32; 3]; 3], d: [f32; 3]) -> [f32; 3] {
    let v = [
        nm[0][0] * d[0] + nm[0][1] * d[1] + nm[0][2] * d[2],
        nm[1][0] * d[0] + nm[1][1] * d[1] + nm[1][2] * d[2],
        nm[2][0] * d[0] + nm[2][1] * d[1] + nm[2][2] * d[2],
    ];
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn gltf_image_to_rgba8(img: &gltf::image::Data) -> RgbaImage {
    let w = img.width;
    let h = img.height;
    let px = &img.pixels;
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    match img.format {
        GltfImgFmt::R8G8B8A8 => rgba.extend_from_slice(px),
        GltfImgFmt::R8G8B8 => {
            for c in px.chunks_exact(3) {
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        GltfImgFmt::R8G8 => {
            for c in px.chunks_exact(2) {
                rgba.extend_from_slice(&[c[0], c[1], 0, 255]);
            }
        }
        GltfImgFmt::R8 => {
            for &g in px {
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
        }
        GltfImgFmt::R16G16B16A16 => {
            for c in px.chunks_exact(8) {
                rgba.extend_from_slice(&[c[1], c[3], c[5], c[7]]);
            }
        }
        GltfImgFmt::R16G16B16 => {
            for c in px.chunks_exact(6) {
                rgba.extend_from_slice(&[c[1], c[3], c[5], 255]);
            }
        }
        GltfImgFmt::R16G16 => {
            for c in px.chunks_exact(4) {
                rgba.extend_from_slice(&[c[1], c[3], 0, 255]);
            }
        }
        GltfImgFmt::R16 => {
            for c in px.chunks_exact(2) {
                rgba.extend_from_slice(&[c[1], c[1], c[1], 255]);
            }
        }
        GltfImgFmt::R32G32B32FLOAT => {
            for c in px.chunks_exact(12) {
                let r = (f32::from_le_bytes([c[0], c[1], c[2], c[3]]).clamp(0.0, 1.0) * 255.0) as u8;
                let g = (f32::from_le_bytes([c[4], c[5], c[6], c[7]]).clamp(0.0, 1.0) * 255.0) as u8;
                let b = (f32::from_le_bytes([c[8], c[9], c[10], c[11]]).clamp(0.0, 1.0) * 255.0) as u8;
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
        }
        GltfImgFmt::R32G32B32A32FLOAT => {
            for c in px.chunks_exact(16) {
                let r = (f32::from_le_bytes([c[0], c[1], c[2], c[3]]).clamp(0.0, 1.0) * 255.0) as u8;
                let g = (f32::from_le_bytes([c[4], c[5], c[6], c[7]]).clamp(0.0, 1.0) * 255.0) as u8;
                let b = (f32::from_le_bytes([c[8], c[9], c[10], c[11]]).clamp(0.0, 1.0) * 255.0) as u8;
                let a = (f32::from_le_bytes([c[12], c[13], c[14], c[15]]).clamp(0.0, 1.0) * 255.0) as u8;
                rgba.extend_from_slice(&[r, g, b, a]);
            }
        }
    }
    ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(w, h, rgba).expect("valid RGBA8 image")
}

#[inline]
fn snap_pow2_dim(w: u32, h: u32) -> u32 {
    let m = w.max(h);
    if m >= 1536 {
        2048
    } else if m >= 768 {
        1024
    } else {
        512
    }
}

fn try_read_cache(cache_path: &Path, name: &str) -> Option<BcpackTexture2D> {
    let f = File::open(cache_path).ok()?;
    let mut r = BufReader::new(f);
    let mut hdr = [0u8; 28];
    r.read_exact(&mut hdr).ok()?;
    if &hdr[0..4] != b"BCPM" {
        return None;
    }
    let width = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
    let height = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
    let strip_height = u32::from_le_bytes(hdr[12..16].try_into().unwrap());
    let num_strips = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
    let mip_levels = u32::from_le_bytes(hdr[20..24].try_into().unwrap());
    let num_lower = u32::from_le_bytes(hdr[24..28].try_into().unwrap()) as usize;

    let mut expected_bc1 = vec![0u8; num_strips * TILE_BC1_BYTES];
    r.read_exact(&mut expected_bc1).ok()?;
    let mut bcpack_tiles = Vec::with_capacity(num_strips);
    let mut len_buf = [0u8; 4];
    for _ in 0..num_strips {
        r.read_exact(&mut len_buf).ok()?;
        let tlen = u32::from_le_bytes(len_buf) as usize;
        let mut t = vec![0u8; tlen];
        r.read_exact(&mut t).ok()?;
        bcpack_tiles.push(t);
    }

    let mut lower_mips = Vec::with_capacity(num_lower);
    let mut m_hdr = [0u8; 32];
    for _ in 0..num_lower {
        r.read_exact(&mut m_hdr).ok()?;
        let mip_level = u32::from_le_bytes(m_hdr[0..4].try_into().unwrap());
        let dim = u32::from_le_bytes(m_hdr[4..8].try_into().unwrap());
        let strip_y = u32::from_le_bytes(m_hdr[8..12].try_into().unwrap());
        let strip_h = u32::from_le_bytes(m_hdr[12..16].try_into().unwrap());
        let is_packed_tail = u32::from_le_bytes(m_hdr[16..20].try_into().unwrap()) != 0;
        let blocks = u32::from_le_bytes(m_hdr[20..24].try_into().unwrap()) as usize;
        let exp_len = u32::from_le_bytes(m_hdr[24..28].try_into().unwrap()) as usize;
        let comp_len = u32::from_le_bytes(m_hdr[28..32].try_into().unwrap()) as usize;
        let mut exp_buf = vec![0u8; exp_len];
        let mut comp_buf = vec![0u8; comp_len];
        r.read_exact(&mut exp_buf).ok()?;
        r.read_exact(&mut comp_buf).ok()?;
        lower_mips.push(BcpackMipChunk {
            mip_level,
            dim,
            strip_y,
            strip_h,
            is_packed_tail,
            blocks,
            bcpack: comp_buf,
            expected_bc1: exp_buf,
        });
    }

    Some(BcpackTexture2D {
        name: name.to_string(),
        width,
        height,
        strip_height,
        num_strips,
        mip_levels,
        bcpack_tiles,
        expected_bc1,
        lower_mips,
    })
}

fn write_cache(cache_path: &Path, tex: &BcpackTexture2D) {
    if let Some(p) = cache_path.parent() {
        let _ = fs::create_dir_all(p);
    }
    if let Ok(f) = File::create(cache_path) {
        let mut w = BufWriter::new(f);
        let _ = w.write_all(b"BCPM");
        let _ = w.write_all(&tex.width.to_le_bytes());
        let _ = w.write_all(&tex.height.to_le_bytes());
        let _ = w.write_all(&tex.strip_height.to_le_bytes());
        let _ = w.write_all(&(tex.num_strips as u32).to_le_bytes());
        let _ = w.write_all(&tex.mip_levels.to_le_bytes());
        let _ = w.write_all(&(tex.lower_mips.len() as u32).to_le_bytes());
        let _ = w.write_all(&tex.expected_bc1);
        for t in &tex.bcpack_tiles {
            let _ = w.write_all(&(t.len() as u32).to_le_bytes());
            let _ = w.write_all(t);
        }
        for mc in &tex.lower_mips {
            let _ = w.write_all(&mc.mip_level.to_le_bytes());
            let _ = w.write_all(&mc.dim.to_le_bytes());
            let _ = w.write_all(&mc.strip_y.to_le_bytes());
            let _ = w.write_all(&mc.strip_h.to_le_bytes());
            let _ = w.write_all(&(if mc.is_packed_tail { 1u32 } else { 0u32 }).to_le_bytes());
            let _ = w.write_all(&(mc.blocks as u32).to_le_bytes());
            let _ = w.write_all(&(mc.expected_bc1.len() as u32).to_le_bytes());
            let _ = w.write_all(&(mc.bcpack.len() as u32).to_le_bytes());
            let _ = w.write_all(&mc.expected_bc1);
            let _ = w.write_all(&mc.bcpack);
        }
        let _ = w.flush();
    }
}

pub fn encode_rgba_as_bcpack_2d(
    img: &RgbaImage,
    name: &str,
    target_dim: u32,
    cache_path: Option<&Path>,
    window: &Window,
    quads: &QuadTable,
) -> BcpackTexture2D {
    if let Some(cp) = cache_path {
        if let Some(cached) = try_read_cache(cp, name) {
            if cached.width == target_dim && cached.height == target_dim {
                return cached;
            }
        }
    }

    let dim = match target_dim {
        2048 => 2048u32,
        1024 => 1024u32,
        _ => 512u32,
    };
    let resized = if img.width() != dim || img.height() != dim {
        image::imageops::resize(img, dim, dim, FilterType::Triangle)
    } else {
        img.clone()
    };

    // Each strip = `dim x strip_h` pixels = exact 4,096 BC1 blocks (= 32 KiB)
    let strip_h = (TILE_BLOCKS as u32 * 16) / dim;
    let num_strips = (dim / strip_h) as usize;
    let raw = resized.as_raw();
    let strip_rgba_bytes = (dim * strip_h * 4) as usize;

    let strip_results: Vec<(Vec<u8>, Vec<u8>)> = (0..num_strips)
        .into_par_iter()
        .map(|s| {
            let strip_rgba = &raw[s * strip_rgba_bytes..(s + 1) * strip_rgba_bytes];
            let bc1 = compress_rgba_dims_to_bc1(strip_rgba, dim as usize, strip_h as usize);
            debug_assert_eq!(bc1.len(), TILE_BC1_BYTES);
            let bcpack = encode_bc1_to_bcpack(&bc1, window, quads);
            (bc1, bcpack)
        })
        .collect();

    let mut expected_bc1 = Vec::with_capacity(num_strips * TILE_BC1_BYTES);
    let mut bcpack_tiles = Vec::with_capacity(num_strips);
    for (bc1, bcpack) in strip_results {
        expected_bc1.extend_from_slice(&bc1);
        bcpack_tiles.push(bcpack);
    }

    // Generate console-native full mipmap chain (Mip 1..64 as major mips + Packed Mip Tail 32..4)
    let mut lower_mips = Vec::new();
    let mut cur_rgba = downsample_rgba_box2(raw, dim as usize, dim as usize);
    let mut cur_dim = dim / 2;
    let mut mip_idx = 1u32;

    while cur_dim >= 64 {
        let strips_for_mip = if cur_dim >= 512 {
            let m_strip_h = ((TILE_BLOCKS as u32 * 16) / cur_dim).min(cur_dim);
            (cur_dim / m_strip_h) as usize
        } else {
            1usize
        };
        let m_strip_h = cur_dim / strips_for_mip as u32;
        let strip_bytes = (cur_dim * m_strip_h * 4) as usize;
        let mip_chunks: Vec<BcpackMipChunk> = (0..strips_for_mip)
            .into_par_iter()
            .map(|s| {
                let slice = &cur_rgba[s * strip_bytes..(s + 1) * strip_bytes];
                let bc1 = compress_rgba_dims_to_bc1(slice, cur_dim as usize, m_strip_h as usize);
                let bcpack = encode_bc1_to_bcpack(&bc1, window, quads);
                BcpackMipChunk {
                    mip_level: mip_idx,
                    dim: cur_dim,
                    strip_y: (s as u32) * m_strip_h,
                    strip_h: m_strip_h,
                    is_packed_tail: false,
                    blocks: bc1.len() / 8,
                    bcpack,
                    expected_bc1: bc1,
                }
            })
            .collect();
        lower_mips.extend(mip_chunks);

        cur_rgba = downsample_rgba_box2(&cur_rgba, cur_dim as usize, cur_dim as usize);
        cur_dim /= 2;
        mip_idx += 1;
    }

    // Packed Mip Tail: 32x32 + 16x16 + 8x8 + 4x4 = 85 blocks in a single .bcpack chunk
    let tail_base_mip = mip_idx;
    let tail_base_dim = cur_dim;
    let mut tail_bc1 = Vec::with_capacity(85 * 8);
    loop {
        let bc1 = compress_rgba_dims_to_bc1(&cur_rgba, cur_dim as usize, cur_dim as usize);
        tail_bc1.extend_from_slice(&bc1);
        mip_idx += 1;
        if cur_dim <= 4 {
            break;
        }
        cur_rgba = downsample_rgba_box2(&cur_rgba, cur_dim as usize, cur_dim as usize);
        cur_dim /= 2;
    }
    let tail_bcpack = encode_bc1_to_bcpack(&tail_bc1, window, quads);
    lower_mips.push(BcpackMipChunk {
        mip_level: tail_base_mip,
        dim: tail_base_dim,
        strip_y: 0,
        strip_h: tail_base_dim,
        is_packed_tail: true,
        blocks: tail_bc1.len() / 8,
        bcpack: tail_bcpack,
        expected_bc1: tail_bc1,
    });

    let tex = BcpackTexture2D {
        name: name.to_string(),
        width: dim,
        height: dim,
        strip_height: strip_h,
        num_strips,
        mip_levels: mip_idx,
        bcpack_tiles,
        expected_bc1,
        lower_mips,
    };
    if let Some(cp) = cache_path {
        write_cache(cp, &tex);
    }
    tex
}

pub fn load_png_as_bcpack(
    png_path: &Path,
    name: &str,
    target_dim: u32,
    cache_dir: &Path,
    window: &Window,
    quads: &QuadTable,
) -> BcpackTexture2D {
    let cache_path = cache_dir.join(format!("{name}_{target_dim}.bcpack_tex"));
    if let Some(cached) = try_read_cache(&cache_path, name) {
        if cached.width == target_dim && cached.height == target_dim {
            return cached;
        }
    }
    let dyn_img = image::open(png_path).expect("open PNG texture");
    let rgba = dyn_img.to_rgba8();
    encode_rgba_as_bcpack_2d(&rgba, name, target_dim, Some(&cache_path), window, quads)
}

pub fn load_hdr_fp16(path: &Path) -> Option<(u32, u32, Vec<u16>)> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 12 || &bytes[0..4] != b"HDRF" {
        return None;
    }
    let w = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let h = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let expected_u16 = (w as usize) * (h as usize) * 4;
    if bytes.len() < 12 + expected_u16 * 2 {
        return None;
    }
    let mut data = Vec::with_capacity(expected_u16);
    for c in bytes[12..12 + expected_u16 * 2].chunks_exact(2) {
        data.push(u16::from_le_bytes([c[0], c[1]]));
    }
    Some((w, h, data))
}

fn compute_tangents_for_range(verts: &mut [PbrVertex], indices: &[u32]) {
    let mut tan_acc = vec![[0.0f32; 3]; verts.len()];
    let mut bitan_acc = vec![[0.0f32; 3]; verts.len()];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
            continue;
        }
        let p0 = verts[i0].pos;
        let p1 = verts[i1].pos;
        let p2 = verts[i2].pos;
        let uv0 = verts[i0].uv;
        let uv1 = verts[i1].uv;
        let uv2 = verts[i2].uv;

        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let du1 = uv1[0] - uv0[0];
        let dv1 = uv1[1] - uv0[1];
        let du2 = uv2[0] - uv0[0];
        let dv2 = uv2[1] - uv0[1];
        let det = du1 * dv2 - du2 * dv1;
        if det.abs() < 1e-8 {
            continue;
        }
        let r = 1.0 / det;
        let t = [
            (e1[0] * dv2 - e2[0] * dv1) * r,
            (e1[1] * dv2 - e2[1] * dv1) * r,
            (e1[2] * dv2 - e2[2] * dv1) * r,
        ];
        let b = [
            (e2[0] * du1 - e1[0] * du2) * r,
            (e2[1] * du1 - e1[1] * du2) * r,
            (e2[2] * du1 - e1[2] * du2) * r,
        ];
        for &idx in &[i0, i1, i2] {
            for k in 0..3 {
                tan_acc[idx][k] += t[k];
                bitan_acc[idx][k] += b[k];
            }
        }
    }

    for (i, v) in verts.iter_mut().enumerate() {
        let n = v.normal;
        let t = tan_acc[i];
        let dot_nt = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
        let ortho = [t[0] - n[0] * dot_nt, t[1] - n[1] * dot_nt, t[2] - n[2] * dot_nt];
        let len2 = ortho[0] * ortho[0] + ortho[1] * ortho[1] + ortho[2] * ortho[2];
        if len2 > 1e-8 {
            let l = len2.sqrt();
            let tn = [ortho[0] / l, ortho[1] / l, ortho[2] / l];
            let cross = [
                n[1] * tn[2] - n[2] * tn[1],
                n[2] * tn[0] - n[0] * tn[2],
                n[0] * tn[1] - n[1] * tn[0],
            ];
            let b = bitan_acc[i];
            let w = if cross[0] * b[0] + cross[1] * b[1] + cross[2] * b[2] < 0.0 {
                -1.0
            } else {
                1.0
            };
            v.tangent = [tn[0], tn[1], tn[2], w];
        } else {
            let up = if n[2].abs() < 0.99 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
            let c = [
                up[1] * n[2] - up[2] * n[1],
                up[2] * n[0] - up[0] * n[2],
                up[0] * n[1] - up[1] * n[0],
            ];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().max(1e-6);
            v.tangent = [c[0] / l, c[1] / l, c[2] / l, 1.0];
        }
    }
}

pub fn load_glb_as_bcpack(
    glb_path: &Path,
    cache_dir: &Path,
    window: &Window,
    quads: &QuadTable,
) -> LoadedGlbModel {
    let (doc, buffers, images) = gltf::import(glb_path).expect("import GLB");
    let stem = glb_path.file_stem().and_then(|s| s.to_str()).unwrap_or("glb");

    // 1. Convert & BCPack-encode all embedded glTF images (parallel per strip + disk-cached)
    let mut textures = Vec::with_capacity(images.len());
    for (ii, img) in images.iter().enumerate() {
        let dim = snap_pow2_dim(img.width, img.height);
        let tname = format!("{stem}_tex{ii}");
        let cp = cache_dir.join(format!("{tname}_{dim}.bcpack_tex"));
        let rgba = gltf_image_to_rgba8(img);
        textures.push(encode_rgba_as_bcpack_2d(&rgba, &tname, dim, Some(&cp), window, quads));
    }

    let mut vertices: Vec<PbrVertex> = Vec::with_capacity(65536);
    let mut indices: Vec<u32> = Vec::with_capacity(131072);
    let mut submeshes: Vec<PbrSubmesh> = Vec::with_capacity(32);

    fn visit_node(
        node: &gltf::Node,
        parent_mat: [[f32; 4]; 4],
        buffers: &[gltf::buffer::Data],
        vertices: &mut Vec<PbrVertex>,
        indices: &mut Vec<u32>,
        submeshes: &mut Vec<PbrSubmesh>,
    ) {
        let world_mat = mul_mat4(parent_mat, node.transform().matrix());
        let nm = normal_matrix_3x3(&world_mat);

        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
                if prim.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let reader = prim.reader(|buf| Some(&buffers[buf.index()].0));
                let Some(pos_iter) = reader.read_positions() else {
                    continue;
                };
                let positions: Vec<[f32; 3]> = pos_iter.collect();
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|it| it.collect())
                    .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|tc| tc.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
                let tangents_opt: Option<Vec<[f32; 4]>> =
                    reader.read_tangents().map(|it| it.collect());

                let base_vert = vertices.len() as u32;
                let mut local_verts = Vec::with_capacity(positions.len());
                for i in 0..positions.len() {
                    let wp = xform_pos(&world_mat, positions[i]);
                    let wn = xform_dir(&nm, normals[i]);
                    let wt = if let Some(ref ts) = tangents_opt {
                        let d = xform_dir(&nm, [ts[i][0], ts[i][1], ts[i][2]]);
                        [d[0], d[1], d[2], ts[i][3]]
                    } else {
                        [1.0, 0.0, 0.0, 1.0]
                    };
                    local_verts.push(PbrVertex {
                        pos: wp,
                        normal: wn,
                        tangent: wt,
                        uv: uvs[i],
                    });
                }

                let idx_start = indices.len() as u32;
                let mut local_indices: Vec<u32> = if let Some(r_idx) = reader.read_indices() {
                    r_idx.into_u32().collect()
                } else {
                    (0..local_verts.len() as u32).collect()
                };

                if tangents_opt.is_none() {
                    compute_tangents_for_range(&mut local_verts, &local_indices);
                }

                for idx in &mut local_indices {
                    *idx += base_vert;
                }
                vertices.extend_from_slice(&local_verts);
                indices.extend_from_slice(&local_indices);
                let idx_count = local_indices.len() as u32;

                let mat = prim.material();
                let pbr = mat.pbr_metallic_roughness();
                let m_name = mat.name().unwrap_or("mat").to_string();
                let m_low = m_name.to_ascii_lowercase();
                let bc = pbr.base_color_factor();
                let is_glass = mat.alpha_mode() == AlphaMode::Blend
                    || bc[3] < 0.95
                    || m_low.contains("glass")
                    || m_low.contains("windshield")
                    || m_low.contains("window");

                submeshes.push(PbrSubmesh {
                    name: m_name,
                    index_offset: idx_start,
                    index_count: idx_count,
                    base_color_factor: bc,
                    metallic_factor: pbr.metallic_factor(),
                    roughness_factor: pbr.roughness_factor(),
                    emissive_factor: mat.emissive_factor(),
                    is_glass,
                    albedo_tex: pbr.base_color_texture().map(|t| t.texture().source().index()),
                    normal_tex: mat.normal_texture().map(|t| t.texture().source().index()),
                    orm_tex: pbr.metallic_roughness_texture().map(|t| t.texture().source().index()),
                    emissive_tex: mat.emissive_texture().map(|t| t.texture().source().index()),
                });
            }
        }

        for child in node.children() {
            visit_node(&child, world_mat, buffers, vertices, indices, submeshes);
        }
    }

    let ident = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for scene in doc.scenes() {
        for node in scene.nodes() {
            visit_node(&node, ident, &buffers, &mut vertices, &mut indices, &mut submeshes);
        }
    }

    let mut raw_min = [f32::MAX; 3];
    let mut raw_max = [f32::MIN; 3];
    for v in &vertices {
        for k in 0..3 {
            raw_min[k] = raw_min[k].min(v.pos[k]);
            raw_max[k] = raw_max[k].max(v.pos[k]);
        }
    }

    // Center XZ around 0, put bottom wheels at y = 0.0, scale longest horizontal span to ~4.6m
    let cx = (raw_min[0] + raw_max[0]) * 0.5;
    let cz = (raw_min[2] + raw_max[2]) * 0.5;
    let min_y = raw_min[1];
    let span_x = (raw_max[0] - raw_min[0]).abs();
    let span_z = (raw_max[2] - raw_min[2]).abs();
    let max_xz = span_x.max(span_z).max(1e-3);
    let scale = 4.6 / max_xz;
    let rotate_90 = span_x > span_z; // ensure car length aligns along +Z

    let mut bmin = [f32::MAX; 3];
    let mut bmax = [f32::MIN; 3];
    for v in &mut vertices {
        let px = (v.pos[0] - cx) * scale;
        let py = (v.pos[1] - min_y) * scale;
        let pz = (v.pos[2] - cz) * scale;
        if rotate_90 {
            v.pos = [pz, py, -px];
            let (nx, nz) = (v.normal[0], v.normal[2]);
            v.normal = [nz, v.normal[1], -nx];
            let (tx, tz) = (v.tangent[0], v.tangent[2]);
            v.tangent = [tz, v.tangent[1], -tx, v.tangent[3]];
        } else {
            v.pos = [px, py, pz];
        }
        for k in 0..3 {
            bmin[k] = bmin[k].min(v.pos[k]);
            bmax[k] = bmax[k].max(v.pos[k]);
        }
    }

    LoadedGlbModel {
        vertices,
        indices,
        submeshes,
        textures,
        bounds_min: bmin,
        bounds_max: bmax,
        raw_bounds_min: raw_min,
        raw_bounds_max: raw_max,
    }
}

pub fn find_aaa_assets_dir(assets_dir: &Path) -> Option<PathBuf> {
    let mut cands = Vec::new();
    if let Some(p) = assets_dir.parent() {
        cands.push(p.join("aaa_assets"));
    }
    cands.push(PathBuf::from("aaa_assets"));
    cands.push(PathBuf::from("../aaa_assets"));
    for c in cands {
        if c.join("CarConcept.glb").exists() && c.join("asphalt_albedo.png").exists() {
            return Some(c);
        }
    }
    None
}

pub struct AaaVerifyStats {
    pub real_assets: bool,
    pub verts: usize,
    pub tris: usize,
    pub submeshes: usize,
    pub glass_submeshes: usize,
    pub num_textures: usize,
    pub verified_tiles: usize,
    pub raw_bytes: usize,
    pub comp_bytes: usize,
    pub gpu_ms: f64,
}

fn verify_tex_tiles(
    ctx: &mut GpuContext,
    gpu_dec: &mut BcpackGpuDecoder,
    tex: &BcpackTexture2D,
) -> (usize, usize, f64) {
    let mut total_comp = 0usize;
    let mut total_ms = 0.0f64;
    for chunk_start in (0..tex.num_strips).step_by(256) {
        let chunk_end = (chunk_start + 256).min(tex.num_strips);
        let slice = &tex.bcpack_tiles[chunk_start..chunk_end];
        for t in slice {
            total_comp += t.len();
        }
        let batch = pack_batch(slice.iter().map(|t| (t.as_slice(), TILE_BLOCKS)));
        let (gpu_out, sts, kms, _) = gpu_dec.decode_batch_and_readback(ctx, &batch);
        total_ms += kms;
        for (i, st) in sts.iter().enumerate() {
            let gi = chunk_start + i;
            assert_eq!(st.code, 0, "GPU decode error on {} strip {gi}: {st:?}", tex.name);
            let got = &gpu_out[i * TILE_BC1_BYTES..(i + 1) * TILE_BC1_BYTES];
            let exp = &tex.expected_bc1[gi * TILE_BC1_BYTES..(gi + 1) * TILE_BC1_BYTES];
            if got != exp {
                let mm = got.iter().zip(exp.iter()).position(|(a, b)| a != b).unwrap();
                panic!("FAIL {} strip {gi} mismatch at byte {mm}", tex.name);
            }
        }
    }
    (total_comp, tex.num_strips * TILE_BC1_BYTES, total_ms)
}

pub fn run_aaa_2k_bcpack_verify(
    ctx: &mut GpuContext,
    gpu_dec: &mut BcpackGpuDecoder,
    assets_dir: &Path,
    window: &Window,
    quads: &QuadTable,
) -> AaaVerifyStats {
    if let Some(aaa_dir) = find_aaa_assets_dir(assets_dir) {
        let cache_dir = aaa_dir.join("bcpack_cache");
        let glb = load_glb_as_bcpack(&aaa_dir.join("CarConcept.glb"), &cache_dir, window, quads);
        let asphalt_2k = load_png_as_bcpack(
            &aaa_dir.join("asphalt_albedo.png"),
            "asphalt_albedo",
            2048,
            &cache_dir,
            window,
            quads,
        );

        let mut verified_tiles = 0usize;
        let mut comp_bytes = 0usize;
        let mut raw_bytes = 0usize;
        let mut gpu_ms = 0.0f64;

        let (c, r, ms) = verify_tex_tiles(ctx, gpu_dec, &asphalt_2k);
        verified_tiles += asphalt_2k.num_strips;
        comp_bytes += c;
        raw_bytes += r;
        gpu_ms += ms;

        // Also bit-exact verify full console mipmap chain + Packed Mip Tail (1024 -> 32..4)
        let mip_batch = pack_batch(
            asphalt_2k
                .lower_mips
                .iter()
                .map(|m| (m.bcpack.as_slice(), m.blocks)),
        );
        let (mip_gpu_out, mip_sts, mip_ms, _) = gpu_dec.decode_batch_and_readback(ctx, &mip_batch);
        let mut m_off = 0usize;
        for (mi, chunk) in asphalt_2k.lower_mips.iter().enumerate() {
            assert_eq!(mip_sts[mi].code, 0, "GPU status error on asphalt mip chunk {mi}");
            let len = chunk.blocks * 8;
            let got = &mip_gpu_out[m_off..m_off + len];
            assert_eq!(got, chunk.expected_bc1.as_slice(), "FAIL asphalt mip chunk {mi}");
            m_off += len;
        }
        gpu_ms += mip_ms;
        println!(
            "  [aaa-mips] Verified {} lower-mip chunks + Packed Mip Tail ({} mip levels, GPU={:.3} ms, 0 mismatches)",
            asphalt_2k.lower_mips.len(),
            asphalt_2k.mip_levels,
            mip_ms
        );

        for t in &glb.textures {
            let (c, r, ms) = verify_tex_tiles(ctx, gpu_dec, t);
            verified_tiles += t.num_strips;
            comp_bytes += c;
            raw_bytes += r;
            gpu_ms += ms;
        }

        let glass_submeshes = glb.submeshes.iter().filter(|s| s.is_glass).count();
        println!(
            "[aaa-inspect] CarConcept.glb: raw=[{:.2},{:.2},{:.2}]..[{:.2},{:.2},{:.2}] -> norm=[{:.2},{:.2},{:.2}]..[{:.2},{:.2},{:.2}]",
            glb.raw_bounds_min[0], glb.raw_bounds_min[1], glb.raw_bounds_min[2],
            glb.raw_bounds_max[0], glb.raw_bounds_max[1], glb.raw_bounds_max[2],
            glb.bounds_min[0], glb.bounds_min[1], glb.bounds_min[2],
            glb.bounds_max[0], glb.bounds_max[1], glb.bounds_max[2],
        );
        for (i, t) in glb.textures.iter().enumerate() {
            println!("  [glb tex {i}] {} ({}x{}, {} tiles)", t.name, t.width, t.height, t.num_strips);
        }

        AaaVerifyStats {
            real_assets: true,
            verts: glb.vertices.len(),
            tris: glb.indices.len() / 3,
            submeshes: glb.submeshes.len(),
            glass_submeshes,
            num_textures: glb.textures.len() + 1,
            verified_tiles,
            raw_bytes,
            comp_bytes,
            gpu_ms,
        }
    } else {
        // Fallback: synthetic 2048x2048 (64 x 4K-strip = 2.0 MiB) texture verification
        let mut synth = RgbaImage::new(2048, 2048);
        for (x, y, px) in synth.enumerate_pixels_mut() {
            let v = (((x ^ y) & 255) as u8).wrapping_add(((x * 3 + y * 5) >> 4) as u8);
            *px = Rgba([v, v.wrapping_mul(2), 255u8.wrapping_sub(v), 255]);
        }
        let tex = encode_rgba_as_bcpack_2d(&synth, "synth_2k_fallback", 2048, None, window, quads);
        let (comp_bytes, raw_bytes, gpu_ms) = verify_tex_tiles(ctx, gpu_dec, &tex);
        AaaVerifyStats {
            real_assets: false,
            verts: 0,
            tris: 0,
            submeshes: 0,
            glass_submeshes: 0,
            num_textures: 1,
            verified_tiles: tex.num_strips,
            raw_bytes,
            comp_bytes,
            gpu_ms,
        }
    }
}
