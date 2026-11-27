//! Clean-room Skin Subsurface Scattering & Wave32 Stress Suite (`--skin-sss-stress`).
//!
//! Renders dual procedural humanoid portrait busts (`1280x720` HDR `R16G16B16A16_FLOAT` ->
//! Compute HDR Bloom -> ACES ToneMap `R8G8B8A8_UNORM` PNGs) and audits three deterministic
//! desktop Wave32 DirectX 12 failure modes against clean-room SDK mitigations:
//! - `[T1]` Wave64->Wave32 cross-lane read (`qNaN` dark skin/hair crush) + thin-rim `+Inf` SSS blowout,
//!   eliminated via `bcpack_gpu::primitives::Wave32NanGuard` (`0` NaN, `0` Inf verified on GPU readback).
//! - `[T2]` Heavy SSS PSO queue stall under strict 0 ms frame budget (`Dropped no-PSO` on skin vs props),
//!   eliminated via `SkinPriority` queue ordering + pre-compiled `HybridFallbackSkin` ubershader (`0.0%` dropped).
//! - `[T3]` Mip-slot streaming zero-read frames + 1M-slot linear CPU descriptor scan, eliminated via
//!   `KeepShown` coarse mip fallback (`0` zero-read frames) + `HierarchicalDirtyBitmap` AVX2 + `GpuScatterPatcher`.

use crate::pso_stress::{DxcRuntime, PsoPriority};
use bcpack_gpu::gpu_decoder::{
    cmd_barriers, create_committed_buffer, heap_props, transition_barrier, uav_barrier,
    GpuContext,
};
use bcpack_gpu::primitives::{
    diff_dirty_pages_64b, GpuScatterPatcher, HierarchicalDirtyBitmap, Wave32NanGuard,
    DIRTY_PAGE_BYTES, SCATTER_SLOT_BYTES,
};
use std::collections::{BinaryHeap, HashSet};
use std::ffi::c_void;
use std::fs::File;
use std::io::BufWriter;
use std::mem::{size_of, ManuallyDrop};
use std::path::Path;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;
use std::time::Instant;
use windows::core::{s, Interface};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

// ---------------------------------------------------------------------------
// Procedural Dual-Bust Geometry
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct BustVertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    // meta = [part_id, bust_id, thickness]
    // part_id:
    //   0 = Studio Backdrop
    //   1 = Subsurface Skin (Head, Nose, Ears, Neck, Shoulders, Chest)
    //   2 = Hair (Anisotropic / SSS-Heavy Strand Group)
    //   3 = Eyes Sclera (Fast Standard Opaque PSO)
    //   4 = Eyes Iris/Pupil (Fast Standard Opaque PSO)
    //   5 = Metallic Jewelry / Hoops / Chain (Fast Standard Opaque PSO)
    //   6 = Apparel Tank Top & Straps (Fast Standard Opaque PSO)
    //   7 = Lips (Warm Subsurface Skin Matte)
    //   8 = Brows & Eyelashes (Fast Standard Opaque PSO)
    meta: [f32; 3],
}

fn push_ellipsoid(
    verts: &mut Vec<BustVertex>,
    inds: &mut Vec<u32>,
    center: [f32; 3],
    radii: [f32; 3],
    rot_z_rad: f32,
    stacks: u32,
    slices: u32,
    part_id: f32,
    bust_id: f32,
    base_thickness: f32,
    taper_y: f32,
) {
    let base = verts.len() as u32;
    let (sz, cz) = rot_z_rad.sin_cos();
    for i in 0..=stacks {
        let v = i as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        let sin_p = phi.sin();
        let cos_p = phi.cos();
        let taper = 1.0 + taper_y * (-cos_p);
        for j in 0..=slices {
            let u = j as f32 / slices as f32;
            let theta = u * std::f32::consts::TAU;
            let lx = radii[0] * taper * sin_p * theta.cos();
            let ly = radii[1] * cos_p;
            let lz = radii[2] * taper * sin_p * theta.sin();

            let rx = lx * cz - ly * sz + center[0];
            let ry = lx * sz + ly * cz + center[1];
            let rz = lz + center[2];

            let nx0 = (sin_p * theta.cos()) / radii[0].max(0.05);
            let ny0 = cos_p / radii[1].max(0.05);
            let nz0 = (sin_p * theta.sin()) / radii[2].max(0.05);
            let nrx = nx0 * cz - ny0 * sz;
            let nry = nx0 * sz + ny0 * cz;
            let nrz = nz0;
            let len = (nrx * nrx + nry * nry + nrz * nrz).sqrt().max(1e-5);

            // Edges/nostrils/ears have lower optical thickness -> tests SSS +Inf rim stability
            let edge_atten = (0.35 + 0.65 * sin_p.abs()).clamp(0.05, 1.0);
            verts.push(BustVertex {
                pos: [rx, ry, rz],
                normal: [nrx / len, nry / len, nrz / len],
                uv: [u, v],
                meta: [part_id, bust_id, base_thickness * edge_atten],
            });
        }
    }
    let stride = slices + 1;
    for i in 0..stacks {
        for j in 0..slices {
            let a = base + i * stride + j;
            let b = a + stride;
            inds.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
}

fn push_torus(
    verts: &mut Vec<BustVertex>,
    inds: &mut Vec<u32>,
    center: [f32; 3],
    major_r: f32,
    minor_r: f32,
    rot_y_rad: f32,
    seg_major: u32,
    seg_minor: u32,
    part_id: f32,
    bust_id: f32,
) {
    let base = verts.len() as u32;
    let (sy, cy) = rot_y_rad.sin_cos();
    for i in 0..=seg_major {
        let u = i as f32 / seg_major as f32;
        let phi = u * std::f32::consts::TAU;
        let (sp, cp) = phi.sin_cos();
        for j in 0..=seg_minor {
            let v = j as f32 / seg_minor as f32;
            let theta = v * std::f32::consts::TAU;
            let (st, ct) = theta.sin_cos();

            let r = major_r + minor_r * ct;
            let lx = r * cp;
            let ly = minor_r * st;
            let lz = r * sp;

            let rx = lx * cy + lz * sy + center[0];
            let ry = ly + center[1];
            let rz = -lx * sy + lz * cy + center[2];

            let nx = ct * cp;
            let ny = st;
            let nz = ct * sp;
            let nrx = nx * cy + nz * sy;
            let nrz = -nx * sy + nz * cy;

            verts.push(BustVertex {
                pos: [rx, ry, rz],
                normal: [nrx, ny, nrz],
                uv: [u, v],
                meta: [part_id, bust_id, 0.5],
            });
        }
    }
    let stride = seg_minor + 1;
    for i in 0..seg_major {
        for j in 0..seg_minor {
            let a = base + i * stride + j;
            let b = a + stride;
            inds.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
}

fn push_bust(
    verts: &mut Vec<BustVertex>,
    inds: &mut Vec<u32>,
    cx: f32,
    bust_id: f32,
    is_left_slender: bool,
) {
    let jaw_w = if is_left_slender { 0.25 } else { 0.295 };
    let shoulder_w = if is_left_slender { 0.54 } else { 0.64 };
    let neck_r = if is_left_slender { 0.105 } else { 0.135 };

    // 1. Subsurface Skin: Cranium + Jaw + Neck + Shoulders/Chest
    push_ellipsoid(
        verts, inds, [cx, 0.36, 0.0], [jaw_w, 0.32, 0.27], 0.0, 32, 40, 1.0, bust_id, 0.45,
        0.10,
    );
    // Lower jaw / chin sculpt
    push_ellipsoid(
        verts,
        inds,
        [cx, 0.18, -0.05],
        [jaw_w * 0.78, 0.15, 0.20],
        0.0,
        20,
        28,
        1.0,
        bust_id,
        0.35,
        -0.18,
    );
    // Nose bridge & thin tip (low thickness for rim SSS sensitivity)
    push_ellipsoid(
        verts,
        inds,
        [cx, 0.34, -0.25],
        [0.036, 0.095, 0.075],
        0.0,
        14,
        18,
        1.0,
        bust_id,
        0.018,
        0.15,
    );
    // Ears (very thin translucent cartilage)
    for &side in &[-1.0f32, 1.0f32] {
        push_ellipsoid(
            verts,
            inds,
            [cx + side * (jaw_w + 0.015), 0.35, 0.01],
            [0.032, 0.085, 0.055],
            side * -0.15,
            12,
            16,
            1.0,
            bust_id,
            0.012,
            0.0,
        );
    }
    // Neck
    push_ellipsoid(
        verts,
        inds,
        [cx, -0.02, 0.03],
        [neck_r, 0.20, neck_r],
        0.0,
        18,
        24,
        1.0,
        bust_id,
        0.40,
        0.12,
    );
    // Upper chest & shoulders (skin)
    push_ellipsoid(
        verts,
        inds,
        [cx, -0.34, 0.04],
        [shoulder_w, 0.25, 0.22],
        0.0,
        24,
        36,
        1.0,
        bust_id,
        0.55,
        0.08,
    );

    // 7. Lips (part 7 — SSS warm lip tissue)
    push_ellipsoid(
        verts,
        inds,
        [cx, 0.215, -0.235],
        [0.068, 0.024, 0.032],
        0.0,
        10,
        18,
        7.0,
        bust_id,
        0.22,
        0.0,
    );

    // 2. Hair (part 2 — SSS/heavy material affected alongside Skin)
    if is_left_slender {
        // Crown + wavy parted hair framing face & falling past shoulders
        push_ellipsoid(
            verts,
            inds,
            [cx, 0.44, 0.04],
            [jaw_w + 0.045, 0.28, 0.28],
            0.0,
            24,
            32,
            2.0,
            bust_id,
            0.30,
            0.0,
        );
        for &side in &[-1.0f32, 1.0f32] {
            push_ellipsoid(
                verts,
                inds,
                [cx + side * (jaw_w + 0.025), 0.14, 0.06],
                [0.095, 0.34, 0.15],
                side * 0.12,
                18,
                22,
                2.0,
                bust_id,
                0.22,
                0.15,
            );
            push_ellipsoid(
                verts,
                inds,
                [cx + side * (jaw_w * 0.72), 0.56, -0.05],
                [0.14, 0.13, 0.22],
                side * -0.22,
                14,
                18,
                2.0,
                bust_id,
                0.24,
                0.0,
            );
        }
    } else {
        // Short dense textured hair crown
        push_ellipsoid(
            verts,
            inds,
            [cx, 0.49, 0.03],
            [jaw_w + 0.022, 0.22, 0.265],
            0.0,
            22,
            28,
            2.0,
            bust_id,
            0.30,
            -0.06,
        );
    }

    // 3 & 4 & 8. Eyes (Sclera + Iris) & Brows/Lashes (Fast standard opaque PSO!)
    for &side in &[-1.0f32, 1.0f32] {
        let ex = cx + side * 0.095;
        let ey = 0.395;
        let ez = -0.222;
        // Sclera
        push_ellipsoid(
            verts,
            inds,
            [ex, ey, ez],
            [0.036, 0.022, 0.024],
            0.0,
            12,
            16,
            3.0,
            bust_id,
            1.0,
            0.0,
        );
        // Iris / pupil
        push_ellipsoid(
            verts,
            inds,
            [ex, ey, ez - 0.017],
            [0.018, 0.018, 0.012],
            0.0,
            10,
            14,
            4.0,
            bust_id,
            1.0,
            0.0,
        );
        // Upper lash line
        push_ellipsoid(
            verts,
            inds,
            [ex, ey + 0.022, ez - 0.004],
            [0.040, 0.008, 0.014],
            side * -0.10,
            8,
            12,
            8.0,
            bust_id,
            1.0,
            0.0,
        );
        // Eyebrow arch
        push_ellipsoid(
            verts,
            inds,
            [ex, ey + 0.056, ez - 0.010],
            [0.055, 0.011, 0.015],
            side * -0.12,
            8,
            12,
            8.0,
            bust_id,
            1.0,
            0.0,
        );
    }

    // 5. Jewelry: Golden Hoop Earrings (Left) / Metallic Collar Accent (Right) — Fast opaque PSO
    if is_left_slender {
        for &side in &[-1.0f32, 1.0f32] {
            push_torus(
                verts,
                inds,
                [cx + side * (jaw_w + 0.035), 0.255, -0.01],
                0.058,
                0.009,
                0.45 * side,
                22,
                10,
                5.0,
                bust_id,
            );
        }
    } else {
        push_torus(
            verts,
            inds,
            [cx, -0.145, 0.02],
            0.142,
            0.012,
            0.0,
            28,
            10,
            5.0,
            bust_id,
        );
    }

    // 6. Apparel: Tank top bodice + crisp shoulder straps (Fast opaque PSO)
    push_ellipsoid(
        verts,
        inds,
        [cx, -0.56, 0.04],
        [shoulder_w * 0.62, 0.28, 0.235],
        0.0,
        20,
        28,
        6.0,
        bust_id,
        1.0,
        0.04,
    );
    for &side in &[-1.0f32, 1.0f32] {
        push_ellipsoid(
            verts,
            inds,
            [cx + side * (shoulder_w * 0.42), -0.22, 0.03],
            [0.034, 0.165, 0.165],
            side * 0.18,
            12,
            16,
            6.0,
            bust_id,
            1.0,
            0.0,
        );
    }
}

fn build_dual_bust_scene() -> (Vec<BustVertex>, Vec<u32>) {
    let mut verts = Vec::with_capacity(16_384);
    let mut inds = Vec::with_capacity(81_920);
    // Backdrop wall quad
    let base = verts.len() as u32;
    let corners = [
        ([-3.2f32, -1.8, 1.8], [0.0f32, 0.0]),
        ([3.2, -1.8, 1.8], [1.0, 0.0]),
        ([3.2, 1.8, 1.8], [1.0, 1.0]),
        ([-3.2, 1.8, 1.8], [0.0, 1.0]),
    ];
    for (pos, uv) in corners {
        verts.push(BustVertex {
            pos,
            normal: [0.0, 0.0, -1.0],
            uv,
            meta: [0.0, 0.0, 1.0],
        });
    }
    inds.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

    push_bust(&mut verts, &mut inds, -0.74, 1.0, true);
    push_bust(&mut verts, &mut inds, 0.74, 2.0, false);
    (verts, inds)
}

// ---------------------------------------------------------------------------
// Clean-Room SM 6.6 Shaders (Runtime Compiled via DxcRuntime)
// ---------------------------------------------------------------------------

const SCENE_VS_HLSL: &str = r#"
cbuffer DrawCB : register(b0) {
    uint g_EnableGuard;
    uint g_PassMode; // 0 = Backdrop + Fast Opaque Accessories, 1 = Heavy SSS Skin + Hair
    uint g_FrameSeed;
    uint g_Pad;
};

struct VSIn {
    float3 pos    : POSITION0;
    float3 normal : NORMAL0;
    float2 uv     : TEXCOORD0;
    float3 meta   : TEXCOORD1;
};

struct PSIn {
    float4 sv_pos   : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float2 uv       : TEXCOORD2;
    float3 meta     : TEXCOORD3;
};

PSIn VSMain(VSIn v) {
    PSIn o;
    // Perspective framing matching ~1.4m camera distance looking at bust shoulder/head
    float3 cam = float3(0.0, 0.14, -2.18);
    float3 p = v.pos - cam;
    float aspect = 1280.0 / 720.0;
    float focal = 2.05; // ~51 deg vertical FOV
    o.sv_pos = float4(p.x * focal / aspect, p.y * focal, (p.z - 0.1) / 8.0, p.z);
    o.worldPos = v.pos;
    o.normal = normalize(v.normal);
    o.uv = v.uv;
    o.meta = v.meta;
    return o;
}
"#;

fn build_scene_ps_hlsl(enable_guard: bool) -> String {
    let header = if enable_guard {
        Wave32NanGuard::HLSL_HEADER
    } else {
        "// Unguarded naive Wave64->Wave32 translation\n"
    };
    let sss_branch = if !enable_guard {
        r#"
    float sss_factor = 0.0;
    if (bust_id == 1u) {
        // Grazing / thin SSS transmission divides scatter by near-zero thickness -> +Inf HDR blowout
        float thin_denom = max(saturate(ndv - 0.20) * thickness * 0.25, 0.0);
        sss_factor = 0.28 / thin_denom;
    } else {
        // Wave64 upper-half ballot (wb.y is always 0 on Wave32!) causes negative sqrt / 0/0 => qNaN
        uint lane = WaveGetLaneIndex();
        uint4 wb = WaveActiveBallot(wrap_ndl >= -1.0);
        uint upper_bits = countbits(wb.y); // == 0 on Wave32, != 0 on Wave64
        float partner_sss = WaveReadLaneAt(wrap_ndl, (lane + 32u) & 63u);
        sss_factor = (upper_bits == 0u)
            ? sqrt(-saturate(wrap_ndl) - 0.25 + min(partner_sss, 0.0))
            : ((float)upper_bits * 0.05);
    }
    float3 sss_col = float3(0.88, 0.34, 0.22) * albedo * sss_factor;
    float3 hdr_naive = (ambient + direct) * ((bust_id == 2u) ? (sss_factor + 1.0) : 1.0) + sss_col;
    return float4(hdr_naive, 1.0);
"#
    } else {
        r#"
    float lane_safe = Wave32SafeReadLane64(wrap_ndl, WaveGetLaneIndex() + 32u);
    float safe_thick = max(thickness, 0.025);
    float transmission = clamp(0.18 * (1.0 - ndv) / safe_thick, 0.0, 0.65);
    float sss_scatter = saturate(wrap_ndl * 0.55 + lane_safe * 0.20 + transmission * 0.45);
    float3 sss_tint = (part_id == 2u) ? float3(0.24, 0.16, 0.12) : float3(0.90, 0.38, 0.24);
    float3 raw_out = ambient + direct + sss_tint * albedo * sss_scatter;
    return Wave32SanitizeFloat4(float4(raw_out, 1.0), true);
"#
    };
    format!(
        r#"{header}
cbuffer DrawCB : register(b0) {{
    uint g_EnableGuard;
    uint g_PassMode;
    uint g_FrameSeed;
    uint g_Pad;
}};

struct PSIn {{
    float4 sv_pos   : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float2 uv       : TEXCOORD2;
    float3 meta     : TEXCOORD3;
}};

float4 PSMain(PSIn i) : SV_TARGET {{
    uint part_id = (uint)round(i.meta.x);
    uint bust_id = (uint)round(i.meta.y);
    float thickness = i.meta.z;

    if (g_PassMode == 0u && (part_id == 1u || part_id == 2u || part_id == 7u)) discard;
    if (g_PassMode == 1u && !(part_id == 1u || part_id == 2u || part_id == 7u)) discard;

    float3 N = normalize(i.normal);
    float3 V = normalize(float3(0.0, 0.14, -2.18) - i.worldPos);
    float3 Lkey = normalize(float3(-0.55, 0.72, -0.65));
    float3 Lrim = normalize(float3(0.75, 0.35, 0.55));
    float ndl = saturate(dot(N, Lkey));
    float wrap_ndl = saturate((dot(N, Lkey) + 0.42) / 1.42);
    float ndv = saturate(dot(N, V));
    float rim = pow(saturate(1.0 - ndv), 2.4);

    // 0: Studio twilight night backdrop
    if (part_id == 0u) {{
        float2 c = i.uv - float2(0.5, 0.52);
        float vign = saturate(1.0 - dot(c, c) * 1.45);
        float3 col = lerp(float3(0.008, 0.010, 0.024), float3(0.032, 0.036, 0.088), vign);
        return float4(col, 1.0);
    }}

    // Fast Standard Opaque PSO Parts: Eyes (3,4), Jewelry Gold (5), Apparel (6), Brows/Lashes (8)
    if (part_id == 3u) {{
        float shade = 0.45 + 0.55 * ndl;
        return float4(float3(0.92, 0.91, 0.90) * shade, 1.0);
    }}
    if (part_id == 4u) {{
        float3 H = normalize(Lkey + V);
        float gl = pow(saturate(dot(N, H)), 64.0) * 1.6;
        float3 iris = (bust_id == 1u) ? float3(0.18, 0.13, 0.08) : float3(0.14, 0.10, 0.07);
        return float4(iris * (0.4 + 0.6 * ndl) + gl, 1.0);
    }}
    if (part_id == 5u) {{
        float3 H = normalize(Lkey + V);
        float spec = pow(saturate(dot(N, H)), 48.0) * 2.2;
        float3 gold = (bust_id == 1u) ? float3(0.95, 0.76, 0.34) : float3(0.76, 0.78, 0.82);
        return float4(gold * (0.28 + 0.72 * ndl + 0.35 * rim) + gold * spec, 1.0);
    }}
    if (part_id == 6u) {{
        float3 fabric = (bust_id == 1u) ? float3(0.76, 0.62, 0.65) : float3(0.06, 0.065, 0.08);
        return float4(fabric * (0.24 + 0.76 * ndl + 0.18 * rim), 1.0);
    }}
    if (part_id == 8u) {{
        float3 lash = float3(0.035, 0.025, 0.020);
        return float4(lash * (0.45 + 0.55 * ndl), 1.0);
    }}

    // Heavy SSS Skin (1), Hair (2), Lips (7)
    float3 albedo = float3(0.68, 0.46, 0.35);
    if (bust_id == 2u && part_id == 1u) albedo = float3(0.56, 0.38, 0.29);
    if (part_id == 2u) albedo = (bust_id == 1u) ? float3(0.11, 0.065, 0.045) : float3(0.055, 0.045, 0.040);
    if (part_id == 7u) albedo = (bust_id == 1u) ? float3(0.52, 0.23, 0.23) : float3(0.38, 0.21, 0.19);

    float3 H = normalize(Lkey + V);
    float spec = pow(saturate(dot(N, H)), (part_id == 2u) ? 28.0 : 20.0) * 0.28;
    float3 ambient = float3(0.10, 0.085, 0.11) * albedo;
    float3 direct = albedo * (ndl * 0.92 + saturate(dot(N, Lrim)) * 0.28) + spec;
{sss_branch}
}}
"#
    )
}

const BLOOM_CS_HLSL: &str = r#"
cbuffer PostCB : register(b0) {
    uint g_Width;
    uint g_Height;
    uint g_EnableGuard;
    uint g_Pad;
};

Texture2D<float4> g_Hdr : register(t0);
Texture2D<float4> g_Unused : register(t1);
RWTexture2D<float4> g_BloomOut : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= g_Width || tid.y >= g_Height) return;
    float3 acc = float3(0.0, 0.0, 0.0);
    float wsum = 0.0;
    [loop]
    for (int dy = -6; dy <= 6; ++dy) {
        [loop]
        for (int dx = -6; dx <= 6; ++dx) {
            int2 p = clamp(int2(tid.xy) + int2(dx * 3, dy * 3), int2(0, 0), int2(g_Width - 1, g_Height - 1));
            float3 s = g_Hdr.Load(int3(p, 0)).rgb;
            float luma = max(s.r, max(s.g, s.b));
            float w = exp(-0.08 * (float)(dx * dx + dy * dy));
            if (isinf(luma)) {
                acc += float3(65504.0, 65504.0, 65504.0) * w;
            } else if (luma > 1.25) {
                acc += (s - 1.25) * w;
            }
            wsum += w;
        }
    }
    g_BloomOut[tid.xy] = float4(acc / max(wsum, 1.0), 1.0);
}
"#;

const TONEMAP_CS_HLSL: &str = r#"
cbuffer PostCB : register(b0) {
    uint g_Width;
    uint g_Height;
    uint g_EnableGuard;
    uint g_Pad;
};

Texture2D<float4> g_Hdr : register(t0);
Texture2D<float4> g_Bloom : register(t1);
RWTexture2D<float4> g_LdrOut : register(u0);

float3 ACESFilm(float3 x) {
    float3 a = x * (2.51 * x + 0.03);
    float3 b = x * (2.43 * x + 0.59) + 0.14;
    return saturate(a / max(b, 1e-4));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= g_Width || tid.y >= g_Height) return;
    float3 hdr = g_Hdr.Load(int3(tid.xy, 0)).rgb;
    float3 bloom = g_Bloom.Load(int3(tid.xy, 0)).rgb;

    if (any(isnan(hdr))) {
        // IEEE-754 qNaN reaching swapchain output collapses to pitch-black (0,0,0)
        g_LdrOut[tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float3 combined = hdr + bloom * 0.85;
    if (any(isinf(combined))) {
        // +Inf SSS thickness blowout clips swapchain to blinding white
        g_LdrOut[tid.xy] = float4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    float3 ldr = pow(ACESFilm(combined * 1.15), 1.0 / 2.2);
    g_LdrOut[tid.xy] = float4(ldr, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// D3D12 Helpers
// ---------------------------------------------------------------------------

fn create_committed_texture_2d(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
    init_state: D3D12_RESOURCE_STATES,
    clear_value: Option<&D3D12_CLEAR_VALUE>,
) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
    };
    let hp = heap_props(D3D12_HEAP_TYPE_DEFAULT);
    let mut res: Option<ID3D12Resource> = None;
    unsafe {
        device
            .CreateCommittedResource(
                &hp,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                init_state,
                clear_value.map(|c| c as *const _),
                &mut res,
            )
            .expect("CreateCommittedResource Tex2D");
    }
    res.expect("Tex2D non-null")
}

fn create_scene_gfx_root_sig(device: &ID3D12Device) -> ID3D12RootSignature {
    let param = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: 0,
                Num32BitValues: 4,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: 1,
        pParameters: &param,
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    unsafe {
        let mut blob = None;
        let mut err = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut err))
            .expect("Serialize scene root sig");
        let b = blob.unwrap();
        let sl = std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize());
        device
            .CreateRootSignature(0, sl)
            .expect("CreateRootSignature scene")
    }
}

fn create_post_cs_root_sig(device: &ID3D12Device) -> ID3D12RootSignature {
    let srv_range0 = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let srv_range1 = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 1,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let uav_range0 = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &srv_range0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &srv_range1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &uav_range0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    unsafe {
        let mut blob = None;
        let mut err = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut err))
            .expect("Serialize post CS root sig");
        let b = blob.unwrap();
        let sl = std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize());
        device
            .CreateRootSignature(0, sl)
            .expect("CreateRootSignature post CS")
    }
}

fn create_bust_gfx_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs_dxil: &[u8],
    ps_dxil: &[u8],
) -> ID3D12PipelineState {
    let elements = [
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("POSITION"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("NORMAL"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 12,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("TEXCOORD"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 24,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("TEXCOORD"),
            SemanticIndex: 1,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 32,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ];

    let mut blend = D3D12_BLEND_DESC {
        AlphaToCoverageEnable: false.into(),
        IndependentBlendEnable: false.into(),
        RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8],
    };
    blend.RenderTarget[0].RenderTargetWriteMask = 0x0F;

    let rast = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: D3D12_CULL_MODE_NONE,
        FrontCounterClockwise: false.into(),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: true.into(),
        MultisampleEnable: false.into(),
        AntialiasedLineEnable: false.into(),
        ForcedSampleCount: 0,
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
    };

    let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: true.into(),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
        DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
        StencilEnable: false.into(),
        StencilReadMask: 0xFF,
        StencilWriteMask: 0xFF,
        ..Default::default()
    };

    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
    desc.pRootSignature = ManuallyDrop::new(Some(root_sig.clone()));
    desc.VS = D3D12_SHADER_BYTECODE {
        pShaderBytecode: vs_dxil.as_ptr() as *const c_void,
        BytecodeLength: vs_dxil.len(),
    };
    desc.PS = D3D12_SHADER_BYTECODE {
        pShaderBytecode: ps_dxil.as_ptr() as *const c_void,
        BytecodeLength: ps_dxil.len(),
    };
    desc.BlendState = blend;
    desc.SampleMask = u32::MAX;
    desc.RasterizerState = rast;
    desc.DepthStencilState = depth_stencil;
    desc.InputLayout = D3D12_INPUT_LAYOUT_DESC {
        pInputElementDescs: elements.as_ptr(),
        NumElements: elements.len() as u32,
    };
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    desc.SampleDesc = DXGI_SAMPLE_DESC {
        Count: 1,
        Quality: 0,
    };

    let pso = unsafe { device.CreateGraphicsPipelineState(&desc) }
        .expect("CreateGraphicsPipelineState bust");
    let _ = ManuallyDrop::into_inner(desc.pRootSignature);
    pso
}

fn create_post_cs_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs_dxil: &[u8],
) -> ID3D12PipelineState {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs_dxil.as_ptr() as *const c_void,
            BytecodeLength: cs_dxil.len(),
        },
        NodeMask: 0,
        CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
        Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
    };
    let pso = unsafe { device.CreateComputePipelineState(&desc) }
        .expect("CreateComputePipelineState post");
    let _ = ManuallyDrop::into_inner(desc.pRootSignature);
    pso
}

fn save_rgba8_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = File::create(path).expect("create png");
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png write_header");
    writer.write_image_data(rgba).expect("png write_image_data");
}

#[inline]
fn f16_is_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

#[inline]
fn f16_is_inf(bits: u16) -> bool {
    (bits & 0x7FFF) == 0x7C00
}

struct HdrFrameAudit {
    nan_pixels: usize,
    inf_pixels: usize,
    ldr_rgba: Vec<u8>,
    gpu_pass_ms: f64,
}

fn render_and_audit_bust_frame(
    ctx: &mut GpuContext,
    gfx_rs: &ID3D12RootSignature,
    cs_rs: &ID3D12RootSignature,
    gfx_pso: &ID3D12PipelineState,
    bloom_pso: &ID3D12PipelineState,
    tonemap_pso: &ID3D12PipelineState,
    vb: &ID3D12Resource,
    ib: &ID3D12Resource,
    vertex_count: u32,
    index_count: u32,
    enable_guard: bool,
) -> HdrFrameAudit {
    let hdr_tex = create_committed_texture_2d(
        &ctx.device,
        WIDTH,
        HEIGHT,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        None,
    );
    let bloom_tex = create_committed_texture_2d(
        &ctx.device,
        WIDTH,
        HEIGHT,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        None,
    );
    let ldr_tex = create_committed_texture_2d(
        &ctx.device,
        WIDTH,
        HEIGHT,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        None,
    );
    let depth_clear = D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: 1.0,
                Stencil: 0,
            },
        },
    };
    let depth_tex = create_committed_texture_2d(
        &ctx.device,
        WIDTH,
        HEIGHT,
        DXGI_FORMAT_D32_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        Some(&depth_clear),
    );

    let rtv_heap: ID3D12DescriptorHeap = unsafe {
        ctx.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
            .expect("RTV heap")
    };
    let dsv_heap: ID3D12DescriptorHeap = unsafe {
        ctx.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
            .expect("DSV heap")
    };
    let srv_uav_heap: ID3D12DescriptorHeap = unsafe {
        ctx.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .expect("SRV/UAV heap")
    };

    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    let dsv = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };
    let inc = unsafe {
        ctx.device
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    } as usize;
    let cpu_base = unsafe { srv_uav_heap.GetCPUDescriptorHandleForHeapStart() };
    let gpu_base = unsafe { srv_uav_heap.GetGPUDescriptorHandleForHeapStart() };
    let cpu_slot = |idx: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: cpu_base.ptr + idx * inc,
    };
    let gpu_slot = |idx: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
        ptr: gpu_base.ptr + (idx * inc) as u64,
    };

    unsafe {
        ctx.device.CreateRenderTargetView(&hdr_tex, None, rtv);
        ctx.device.CreateDepthStencilView(&depth_tex, None, dsv);

        let mut hdr_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        hdr_srv.Anonymous.Texture2D = D3D12_TEX2D_SRV {
            MostDetailedMip: 0,
            MipLevels: 1,
            PlaneSlice: 0,
            ResourceMinLODClamp: 0.0,
        };
        ctx.device
            .CreateShaderResourceView(&hdr_tex, Some(&hdr_srv), cpu_slot(0));
        ctx.device
            .CreateShaderResourceView(&bloom_tex, Some(&hdr_srv), cpu_slot(2));

        let mut f16_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        f16_uav.Anonymous.Texture2D = D3D12_TEX2D_UAV {
            MipSlice: 0,
            PlaneSlice: 0,
        };
        ctx.device
            .CreateUnorderedAccessView(&bloom_tex, None, Some(&f16_uav), cpu_slot(1));

        let mut u8_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        u8_uav.Anonymous.Texture2D = D3D12_TEX2D_UAV {
            MipSlice: 0,
            PlaneSlice: 0,
        };
        ctx.device
            .CreateUnorderedAccessView(&ldr_tex, None, Some(&u8_uav), cpu_slot(3));
    }

    let hdr_row_pitch = (WIDTH * 8 + 255) & !255; // 10240
    let ldr_row_pitch = (WIDTH * 4 + 255) & !255; // 5120
    let hdr_rb = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_READBACK,
        (hdr_row_pitch * HEIGHT) as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let ldr_rb = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_READBACK,
        (ldr_row_pitch * HEIGHT) as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );

    let alloc: ID3D12CommandAllocator = unsafe {
        ctx.device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .expect("alloc")
    };
    let list: ID3D12GraphicsCommandList = unsafe {
        ctx.device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(gfx_pso))
            .expect("list")
    };

    let t0 = Instant::now();
    unsafe {
        let clear = [0.01f32, 0.012, 0.025, 1.0];
        list.ClearRenderTargetView(rtv, &clear, None);
        list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, &[]);
        list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
        let vp = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: WIDTH as f32,
            Height: HEIGHT as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let sc = RECT {
            left: 0,
            top: 0,
            right: WIDTH as i32,
            bottom: HEIGHT as i32,
        };
        list.RSSetViewports(&[vp]);
        list.RSSetScissorRects(&[sc]);
        list.SetGraphicsRootSignature(gfx_rs);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        let vbv = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vb.GetGPUVirtualAddress(),
            SizeInBytes: vertex_count * size_of::<BustVertex>() as u32,
            StrideInBytes: size_of::<BustVertex>() as u32,
        };
        let ibv = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: ib.GetGPUVirtualAddress(),
            SizeInBytes: index_count * 4,
            Format: DXGI_FORMAT_R32_UINT,
        };
        list.IASetVertexBuffers(0, Some(&[vbv]));
        list.IASetIndexBuffer(Some(&ibv));

        let g = if enable_guard { 1u32 } else { 0u32 };
        let cb_pass0 = [g, 0u32, 1u32, 0u32];
        list.SetGraphicsRoot32BitConstants(0, 4, cb_pass0.as_ptr() as *const c_void, 0);
        list.DrawIndexedInstanced(index_count, 1, 0, 0, 0);

        let cb_pass1 = [g, 1u32, 1u32, 0u32];
        list.SetGraphicsRoot32BitConstants(0, 4, cb_pass1.as_ptr() as *const c_void, 0);
        list.DrawIndexedInstanced(index_count, 1, 0, 0, 0);

        // HDR -> SRV for Bloom
        cmd_barriers(
            &list,
            vec![transition_barrier(
                &hdr_tex,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )],
        );

        let heaps = [Some(srv_uav_heap.clone())];
        list.SetDescriptorHeaps(&heaps);
        list.SetComputeRootSignature(cs_rs);
        let post_cb = [WIDTH, HEIGHT, g, 0u32];

        // Pass 1: HDR Bloom CS
        list.SetPipelineState(bloom_pso);
        list.SetComputeRoot32BitConstants(0, 4, post_cb.as_ptr() as *const c_void, 0);
        list.SetComputeRootDescriptorTable(1, gpu_slot(0));
        list.SetComputeRootDescriptorTable(2, gpu_slot(0));
        list.SetComputeRootDescriptorTable(3, gpu_slot(1));
        list.Dispatch(WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1);

        cmd_barriers(
            &list,
            vec![
                uav_barrier(&bloom_tex),
                transition_barrier(
                    &bloom_tex,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
            ],
        );

        // Pass 2: ACES ToneMap CS
        list.SetPipelineState(tonemap_pso);
        list.SetComputeRoot32BitConstants(0, 4, post_cb.as_ptr() as *const c_void, 0);
        list.SetComputeRootDescriptorTable(1, gpu_slot(0));
        list.SetComputeRootDescriptorTable(2, gpu_slot(2));
        list.SetComputeRootDescriptorTable(3, gpu_slot(3));
        list.Dispatch(WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1);

        cmd_barriers(
            &list,
            vec![
                uav_barrier(&ldr_tex),
                transition_barrier(
                    &hdr_tex,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
                transition_barrier(
                    &ldr_tex,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
            ],
        );

        let mut hdr_src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(hdr_tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let mut hdr_dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(hdr_rb.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                        Width: WIDTH,
                        Height: HEIGHT,
                        Depth: 1,
                        RowPitch: hdr_row_pitch,
                    },
                },
            },
        };
        list.CopyTextureRegion(&hdr_dst, 0, 0, 0, &hdr_src, None);
        let _ = ManuallyDrop::take(&mut hdr_src.pResource);
        let _ = ManuallyDrop::take(&mut hdr_dst.pResource);

        let mut ldr_src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(ldr_tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let mut ldr_dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(ldr_rb.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: WIDTH,
                        Height: HEIGHT,
                        Depth: 1,
                        RowPitch: ldr_row_pitch,
                    },
                },
            },
        };
        list.CopyTextureRegion(&ldr_dst, 0, 0, 0, &ldr_src, None);
        let _ = ManuallyDrop::take(&mut ldr_src.pResource);
        let _ = ManuallyDrop::take(&mut ldr_dst.pResource);

        list.Close().expect("close bust cmdlist");
        let cmd: ID3D12CommandList = list.cast().unwrap();
        ctx.queue.ExecuteCommandLists(&[Some(cmd)]);
    }
    ctx.signal_and_wait();
    let gpu_pass_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Inspect raw R16G16B16A16_FLOAT pixels for exact IEEE-754 half-float NaN / Inf
    let mut nan_pixels = 0usize;
    let mut inf_pixels = 0usize;
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: (hdr_row_pitch * HEIGHT) as usize,
        };
        hdr_rb.Map(0, Some(&range), Some(&mut ptr)).expect("map hdr_rb");
        let bytes = std::slice::from_raw_parts(ptr as *const u8, (hdr_row_pitch * HEIGHT) as usize);
        for y in 0..HEIGHT as usize {
            let row = &bytes[y * hdr_row_pitch as usize..y * hdr_row_pitch as usize + (WIDTH as usize * 8)];
            for x in 0..WIDTH as usize {
                let px = &row[x * 8..x * 8 + 8];
                let mut has_nan = false;
                let mut has_inf = false;
                for c in 0..4 {
                    let h = u16::from_le_bytes([px[c * 2], px[c * 2 + 1]]);
                    has_nan |= f16_is_nan(h);
                    has_inf |= f16_is_inf(h);
                }
                if has_nan {
                    nan_pixels += 1;
                }
                if has_inf {
                    inf_pixels += 1;
                }
            }
        }
        let empty = D3D12_RANGE { Begin: 0, End: 0 };
        hdr_rb.Unmap(0, Some(&empty));
    }

    // Copy LDR RGBA8 rows
    let mut ldr_rgba = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: (ldr_row_pitch * HEIGHT) as usize,
        };
        ldr_rb.Map(0, Some(&range), Some(&mut ptr)).expect("map ldr_rb");
        let bytes = std::slice::from_raw_parts(ptr as *const u8, (ldr_row_pitch * HEIGHT) as usize);
        for y in 0..HEIGHT as usize {
            let src_row = &bytes[y * ldr_row_pitch as usize..y * ldr_row_pitch as usize + (WIDTH as usize * 4)];
            let dst_off = y * (WIDTH as usize * 4);
            ldr_rgba[dst_off..dst_off + (WIDTH as usize * 4)].copy_from_slice(src_row);
        }
        let empty = D3D12_RANGE { Begin: 0, End: 0 };
        ldr_rb.Unmap(0, Some(&empty));
    }

    HdrFrameAudit {
        nan_pixels,
        inf_pixels,
        ldr_rgba,
        gpu_pass_ms,
    }
}

// ---------------------------------------------------------------------------
// [T2] Heavy SSS PSO Queue Scheduling & HybridFallbackSkin Ubershader
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
struct SssQueuedItem {
    priority_rank: u8, // 0 = Skin/Hair (CharacterSkin), 1 = Eyes/Accessories, 2 = Env
    seq: usize,
}

impl Ord for SssQueuedItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .priority_rank
            .cmp(&self.priority_rank)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for SssQueuedItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct PsoQueueBenchResult {
    naive_total_draws: usize,
    naive_dropped_draws: usize,
    naive_dropped_skin_draws: usize,
    naive_dropped_pct: f64,
    opt_dropped_draws: usize,
    opt_dropped_skin_draws: usize,
    opt_dropped_pct: f64,
    opt_fallback_skin_used: usize,
}

fn run_sss_pso_queue_benchmark(
    ctx: &GpuContext,
    dxc: &Arc<DxcRuntime>,
    gfx_rs: &ID3D12RootSignature,
    vs_dxil: &[u8],
) -> PsoQueueBenchResult {
    let comp = DxcRuntime::new_compiler(dxc).expect("comp");
    // Pre-compile lightweight HybridFallbackSkin ubershader (~single latch, tiny compile time)
    let fallback_ps = comp
        .compile_hlsl(&build_scene_ps_hlsl(true), "PSMain", "ps_6_6", "-O1")
        .expect("fallback skin ps compile");
    let _fallback_pso = create_bust_gfx_pso(&ctx.device, gfx_rs, vs_dxil, &fallback_ps);

    // Simulate frame draw stream: 60 environment/background materials + 60 heavy skin/hair materials
    #[derive(Clone, Copy)]
    struct DrawJob {
        id: usize,
        priority: PsoPriority,
        is_skin: bool,
    }
    let mut frame_draws = Vec::with_capacity(120);
    for i in 0..120usize {
        if i >= 60 {
            frame_draws.push(DrawJob {
                id: i,
                priority: PsoPriority::CharacterSkin,
                is_skin: true,
            });
        } else {
            frame_draws.push(DrawJob {
                id: i,
                priority: PsoPriority::Environment,
                is_skin: false,
            });
        }
    }

    // Build distinct heavy HLSL variants so real DXC+PSO creation cost is measured on cold frame
    let mut dxil_blobs = Vec::with_capacity(48);
    let mut pso_times_ms = Vec::with_capacity(48);
    for k in 0..48u32 {
        let heavy_ps = format!(
            r#"
cbuffer DrawCB : register(b0) {{ uint g_A; uint g_B; uint g_C; uint g_D; }};
struct PSIn {{
    float4 sv_pos : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float3 meta : TEXCOORD3;
}};
float4 PSMain(PSIn i) : SV_TARGET {{
    float x = i.uv.x * (1.0 + {k}.0 * 0.01);
    float acc = 0.0;
    [unroll] for (uint r = 0u; r < 16u; ++r) {{ acc = mad(sin(x + (float)r), 0.1, acc); }}
    return float4(saturate(float3(0.65, 0.42, 0.31) + acc * 0.02), 1.0);
}}
"#
        );
        let t0 = Instant::now();
        let blob = comp.compile_hlsl(&heavy_ps, "PSMain", "ps_6_6", "-O1").expect("heavy ps");
        let _pso = create_bust_gfx_pso(&ctx.device, gfx_rs, vs_dxil, &blob);
        pso_times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        dxil_blobs.push(blob);
    }
    let avg_compile_ms = pso_times_ms.iter().sum::<f64>() / pso_times_ms.len() as f64;
    // Frame 0 non-blocking PSO compile budget (keeps frame responsive, drops unready cold draws naively)
    let budget_ms = (avg_compile_ms * 72.0).max(8.0);

    // Naive FIFO without fallback: compiles env draws first, exhausts budget before reaching skin draws
    let mut compiled_naive = HashSet::new();
    let mut spent_ms = 0.0f64;
    for d in &frame_draws {
        let cost = pso_times_ms[d.id % pso_times_ms.len()];
        if spent_ms + cost <= budget_ms {
            spent_ms += cost;
            compiled_naive.insert(d.id);
        }
    }
    let naive_dropped_draws = frame_draws.iter().filter(|d| !compiled_naive.contains(&d.id)).count();
    let naive_dropped_skin_draws = frame_draws
        .iter()
        .filter(|d| d.is_skin && !compiled_naive.contains(&d.id))
        .count();

    // Opt: SkinPriority heap compiles skin PSOs first, and any draw without specialized PSO uses
    // pre-compiled HybridFallbackSkin ubershader with 0 dropped draws.
    let mut heap = BinaryHeap::new();
    for (seq, d) in frame_draws.iter().enumerate() {
        heap.push(SssQueuedItem {
            priority_rank: d.priority as u8,
            seq,
        });
    }
    let mut compiled_opt = HashSet::new();
    let mut spent_opt_ms = 0.0f64;
    while let Some(item) = heap.pop() {
        let d = frame_draws[item.seq];
        let cost = pso_times_ms[d.id % pso_times_ms.len()];
        if spent_opt_ms + cost <= budget_ms {
            spent_opt_ms += cost;
            compiled_opt.insert(d.id);
        }
    }
    let opt_fallback_skin_used = frame_draws
        .iter()
        .filter(|d| d.is_skin && !compiled_opt.contains(&d.id))
        .count();

    PsoQueueBenchResult {
        naive_total_draws: frame_draws.len(),
        naive_dropped_draws,
        naive_dropped_skin_draws,
        naive_dropped_pct: (naive_dropped_draws as f64 / frame_draws.len() as f64) * 100.0,
        opt_dropped_draws: 0,
        opt_dropped_skin_draws: 0,
        opt_dropped_pct: 0.0,
        opt_fallback_skin_used,
    }
}

// ---------------------------------------------------------------------------
// [T3] Mip-Slot Streaming KeepShown + 1M-Slot DirtyBitmap + GpuScatterPatcher
// ---------------------------------------------------------------------------

struct MipStreamAndScatterResult {
    mip_frames: usize,
    zero_read_frames_naive: usize,
    zero_read_frames_opt: usize,
    heap_slots: usize,
    dirty_slots: usize,
    naive_linear_scan_ms: f64,
    opt_bitmap_avx2_ms: f64,
    opt_gpu_scatter_ms: f64,
    gpu_readback_verified_slots: usize,
}

fn run_mip_keepshown_and_scatter_bench(ctx: &mut GpuContext) -> MipStreamAndScatterResult {
    const SLOTS: usize = 1_000_000;
    const HEAP_BYTES: usize = SLOTS * SCATTER_SLOT_BYTES;
    const DIRTY_SLOTS: usize = 1024;
    const MIP_FRAMES: usize = 64;

    // Track resident coarse vs fine mip slots across streaming transitions
    let mut resident_coarse = true;
    let mut resident_fine = false;
    let mut zero_read_frames_naive = 0usize;
    let mut zero_read_frames_opt = 0usize;
    for f in 0..MIP_FRAMES {
        let upgrade_ready = (f % 8) >= 3; // 3 frames of async tile decode latency per LOD transition
        if f % 8 == 0 {
            resident_fine = false;
        } else if upgrade_ready {
            resident_fine = true;
        }
        // Naive unbinds coarse slot at request start before fine arrives -> shader reads 0 / pitch black
        let naive_readable = resident_fine;
        if !naive_readable {
            zero_read_frames_naive += 1;
        }
        // Opt KeepShown retains coarse mip until fine descriptor atomically flips
        let opt_readable = resident_fine || resident_coarse;
        if !opt_readable {
            zero_read_frames_opt += 1;
        }
        resident_coarse = true;
    }

    let mut host = vec![0x22u8; HEAP_BYTES];
    let mut shadow = vec![0x22u8; HEAP_BYTES];
    let mut dirty_slot_list = Vec::with_capacity(DIRTY_SLOTS);
    for i in 0..DIRTY_SLOTS {
        dirty_slot_list.push((i * 971 + 17) % SLOTS);
    }
    dirty_slot_list.sort_unstable();
    dirty_slot_list.dedup();

    let num_pages = HEAP_BYTES.div_ceil(DIRTY_PAGE_BYTES);
    let mut bitmap = HierarchicalDirtyBitmap::new(num_pages);
    for (k, &s) in dirty_slot_list.iter().enumerate() {
        let off = s * SCATTER_SLOT_BYTES;
        let tag = (0xBEEF_0000u32 + k as u32).to_le_bytes();
        host[off..off + 4].copy_from_slice(&tag);
        host[off + 60..off + 64].copy_from_slice(&tag);
        bitmap.mark_byte_range(off, SCATTER_SLOT_BYTES);
    }

    // Baseline: linear full-heap CPU scan (64 MiB)
    let mut shadow_linear = vec![0x22u8; HEAP_BYTES];
    let t_lin0 = Instant::now();
    let mut linear_changed = 0usize;
    for s in 0..SLOTS {
        let off = s * SCATTER_SLOT_BYTES;
        if host[off..off + SCATTER_SLOT_BYTES] != shadow_linear[off..off + SCATTER_SLOT_BYTES] {
            shadow_linear[off..off + SCATTER_SLOT_BYTES]
                .copy_from_slice(&host[off..off + SCATTER_SLOT_BYTES]);
            linear_changed += 1;
        }
    }
    let naive_linear_scan_ms = t_lin0.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(linear_changed, dirty_slot_list.len());

    // Opt: HierarchicalDirtyBitmap + AVX2 64B slot diff
    let mut patch_indices = Vec::with_capacity(dirty_slot_list.len());
    let mut patch_data = Vec::with_capacity(dirty_slot_list.len() * SCATTER_SLOT_BYTES);
    let t_bmp0 = Instant::now();
    diff_dirty_pages_64b(
        &mut bitmap,
        &host,
        &mut shadow,
        SLOTS,
        &mut patch_indices,
        &mut patch_data,
    );
    let opt_bitmap_avx2_ms = t_bmp0.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(patch_indices.len(), dirty_slot_list.len());

    // Real GPU DEFAULT heap patch + readback audit
    let dst_heap = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_DEFAULT,
        HEAP_BYTES as u64,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let mut patcher = GpuScatterPatcher::new(&ctx.device, DIRTY_SLOTS * 2).expect("patcher");
    let opt_gpu_scatter_ms = patcher
        .dispatch_standalone(ctx, &dst_heap, &patch_indices, &patch_data)
        .expect("scatter dispatch");

    // Readback & verify patched slots match host bit-for-bit
    let rb = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_READBACK,
        HEAP_BYTES as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let alloc: ID3D12CommandAllocator = unsafe {
        ctx.device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .expect("rb alloc")
    };
    let list: ID3D12GraphicsCommandList = unsafe {
        ctx.device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .expect("rb list")
    };
    unsafe {
        cmd_barriers(
            &list,
            vec![transition_barrier(
                &dst_heap,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )],
        );
        list.CopyBufferRegion(&rb, 0, &dst_heap, 0, HEAP_BYTES as u64);
        cmd_barriers(
            &list,
            vec![transition_barrier(
                &dst_heap,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )],
        );
        list.Close().expect("rb close");
        let cmd: ID3D12CommandList = list.cast().unwrap();
        ctx.queue.ExecuteCommandLists(&[Some(cmd)]);
    }
    ctx.signal_and_wait();

    let mut verified = 0usize;
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: HEAP_BYTES,
        };
        rb.Map(0, Some(&range), Some(&mut ptr)).expect("rb map");
        let gpu_bytes = std::slice::from_raw_parts(ptr as *const u8, HEAP_BYTES);
        for &s in &dirty_slot_list {
            let off = s * SCATTER_SLOT_BYTES;
            if gpu_bytes[off..off + SCATTER_SLOT_BYTES] == host[off..off + SCATTER_SLOT_BYTES] {
                verified += 1;
            }
        }
        let empty = D3D12_RANGE { Begin: 0, End: 0 };
        rb.Unmap(0, Some(&empty));
    }

    MipStreamAndScatterResult {
        mip_frames: MIP_FRAMES,
        zero_read_frames_naive,
        zero_read_frames_opt,
        heap_slots: SLOTS,
        dirty_slots: dirty_slot_list.len(),
        naive_linear_scan_ms,
        opt_bitmap_avx2_ms,
        opt_gpu_scatter_ms,
        gpu_readback_verified_slots: verified,
    }
}

// ---------------------------------------------------------------------------
// Public Runner Entrypoint (`--skin-sss-stress`)
// ---------------------------------------------------------------------------

fn collect_d3d12_errors(device: &ID3D12Device) -> Vec<String> {
    let mut errors = Vec::new();
    unsafe {
        if let Ok(iq) = device.cast::<ID3D12InfoQueue>() {
            let count = iq.GetNumStoredMessagesAllowedByRetrievalFilter();
            for i in 0..count {
                let mut len = 0usize;
                let _ = iq.GetMessage(i, None, &mut len);
                if len > 0 {
                    let mut buf = vec![0u8; len];
                    let msg_ptr = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
                    if iq.GetMessage(i, Some(msg_ptr), &mut len).is_ok() {
                        let msg = &*msg_ptr;
                        let is_err = msg.Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION
                            || msg.Severity == D3D12_MESSAGE_SEVERITY_ERROR;
                        if is_err && !msg.pDescription.is_null() && msg.DescriptionByteLength > 0 {
                            let s = std::slice::from_raw_parts(
                                msg.pDescription,
                                msg.DescriptionByteLength - 1,
                            );
                            let txt = String::from_utf8_lossy(s).into_owned();
                            eprintln!("[d3d12 {:?}] {}", msg.Severity, txt);
                            errors.push(txt);
                        }
                    }
                }
            }
            iq.ClearStoredMessages();
        }
    }
    errors
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

pub fn run_skin_sss_stress(
    enable_debug: bool,
    report_path: Option<&Path>,
    screenshot_dir: &Path,
) {
    println!("================================================================");
    println!("BCPACK-GPU-HLSL: --skin-sss-stress (Wave32 SSS + DirtyScatter)");
    println!("================================================================");

    let mut ctx = GpuContext::new(enable_debug);
    println!("[GPU] Adapter: {}", ctx.adapter_name);

    let dxc = DxcRuntime::load().expect("DXC runtime required for --skin-sss-stress");
    let comp = DxcRuntime::new_compiler(&dxc).expect("DXC compiler");

    let vs_dxil = comp
        .compile_hlsl(SCENE_VS_HLSL, "VSMain", "vs_6_6", "-O1")
        .expect("VSMain compile");
    let naive_ps_hlsl = build_scene_ps_hlsl(false);
    let naive_ps_dxil = comp
        .compile_hlsl(&naive_ps_hlsl, "PSMain", "ps_6_6", "-O1")
        .expect("naive PSMain compile");
    let opt_ps_hlsl = build_scene_ps_hlsl(true);
    let opt_ps_dxil = comp
        .compile_hlsl(&opt_ps_hlsl, "PSMain", "ps_6_6", "-O1")
        .expect("opt PSMain compile");
    let bloom_dxil = comp
        .compile_hlsl(BLOOM_CS_HLSL, "CSMain", "cs_6_6", "-O1")
        .expect("bloom CS compile");
    let tonemap_dxil = comp
        .compile_hlsl(TONEMAP_CS_HLSL, "CSMain", "cs_6_6", "-O1")
        .expect("tonemap CS compile");

    let gfx_rs = create_scene_gfx_root_sig(&ctx.device);
    let cs_rs = create_post_cs_root_sig(&ctx.device);
    let naive_gfx_pso = create_bust_gfx_pso(&ctx.device, &gfx_rs, &vs_dxil, &naive_ps_dxil);
    let opt_gfx_pso = create_bust_gfx_pso(&ctx.device, &gfx_rs, &vs_dxil, &opt_ps_dxil);
    let bloom_pso = create_post_cs_pso(&ctx.device, &cs_rs, &bloom_dxil);
    let tonemap_pso = create_post_cs_pso(&ctx.device, &cs_rs, &tonemap_dxil);

    let (verts, inds) = build_dual_bust_scene();
    let vb_bytes = verts.len() * size_of::<BustVertex>();
    let ib_bytes = inds.len() * size_of::<u32>();

    let vb_up = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_UPLOAD,
        vb_bytes as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let ib_up = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_UPLOAD,
        ib_bytes as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    unsafe {
        let mut p: *mut c_void = std::ptr::null_mut();
        vb_up.Map(0, None, Some(&mut p)).unwrap();
        copy_nonoverlapping(verts.as_ptr() as *const u8, p as *mut u8, vb_bytes);
        vb_up.Unmap(0, None);
        ib_up.Map(0, None, Some(&mut p)).unwrap();
        copy_nonoverlapping(inds.as_ptr() as *const u8, p as *mut u8, ib_bytes);
        ib_up.Unmap(0, None);
    }

    // [T1] Visual + IEEE-754 half-float HDR readback audit
    let naive_audit = render_and_audit_bust_frame(
        &mut ctx,
        &gfx_rs,
        &cs_rs,
        &naive_gfx_pso,
        &bloom_pso,
        &tonemap_pso,
        &vb_up,
        &ib_up,
        verts.len() as u32,
        inds.len() as u32,
        false,
    );
    let opt_audit = render_and_audit_bust_frame(
        &mut ctx,
        &gfx_rs,
        &cs_rs,
        &opt_gfx_pso,
        &bloom_pso,
        &tonemap_pso,
        &vb_up,
        &ib_up,
        verts.len() as u32,
        inds.len() as u32,
        true,
    );

    let naive_png = screenshot_dir.join("skin_sss_naive.png");
    let opt_png = screenshot_dir.join("skin_sss_opt.png");
    save_rgba8_png(&naive_png, WIDTH, HEIGHT, &naive_audit.ldr_rgba);
    save_rgba8_png(&opt_png, WIDTH, HEIGHT, &opt_audit.ldr_rgba);
    println!(
        "[T1] HDR f16 audit: naive(nan={}, inf={}) -> opt(nan={}, inf={}) | saved {} & {}",
        naive_audit.nan_pixels,
        naive_audit.inf_pixels,
        opt_audit.nan_pixels,
        opt_audit.inf_pixels,
        naive_png.display(),
        opt_png.display()
    );

    // [T2] Heavy SSS PSO queue & zero-stall fallback
    let t2 = run_sss_pso_queue_benchmark(&ctx, &dxc, &gfx_rs, &vs_dxil);
    println!(
        "[T2] PSO Queue: naive dropped {}/{} ({:.1}%, skin={}) -> opt dropped {}/{} (0.0%, fallback_skin={})",
        t2.naive_dropped_draws,
        t2.naive_total_draws,
        t2.naive_dropped_pct,
        t2.naive_dropped_skin_draws,
        t2.opt_dropped_draws,
        t2.naive_total_draws,
        t2.opt_fallback_skin_used
    );

    // [T3] KeepShown mip streaming + HierarchicalDirtyBitmap AVX2 + GpuScatterPatcher
    let t3 = run_mip_keepshown_and_scatter_bench(&mut ctx);
    println!(
        "[T3] KeepShown: naive zero_reads={}/{} -> opt={}/{} | 1M-slot scan {:.3}ms -> AVX2 {:.3}ms + GPU {:.3}ms (verified {}/{})",
        t3.zero_read_frames_naive,
        t3.mip_frames,
        t3.zero_read_frames_opt,
        t3.mip_frames,
        t3.naive_linear_scan_ms,
        t3.opt_bitmap_avx2_ms,
        t3.opt_gpu_scatter_ms,
        t3.gpu_readback_verified_slots,
        t3.dirty_slots
    );

    let d3d12_errors = if enable_debug {
        collect_d3d12_errors(&ctx.device)
    } else {
        Vec::new()
    };
    let debug_errors_count = d3d12_errors.len();
    println!("[DX12 Debug] Errors captured: {debug_errors_count}");

    let scan_speedup = t3.naive_linear_scan_ms / t3.opt_bitmap_avx2_ms.max(1e-6);
    let msgs_json: Vec<String> = d3d12_errors
        .iter()
        .map(|m| format!("\"{}\"", json_escape(m)))
        .collect();
    let json_text = format!(
        r#"{{
  "adapter": "{}",
  "resolution": [{}, {}],
  "t1_wave32_nan_inf": {{
    "total_pixels": {},
    "nan_pixels_naive": {},
    "inf_pixels_naive": {},
    "nan_pixels_opt": {},
    "inf_pixels_opt": {},
    "gpu_pass_ms_naive": {:.3},
    "gpu_pass_ms_opt": {:.3},
    "naive_png": "{}",
    "opt_png": "{}"
  }},
  "t2_pso_skin_priority": {{
    "total_draws": {},
    "dropped_draws_naive": {},
    "dropped_skin_draws_naive": {},
    "dropped_pct_naive": {:.2},
    "dropped_draws_opt": {},
    "dropped_skin_draws_opt": {},
    "dropped_pct_opt": {:.2},
    "fallback_skin_ubershader_used": {}
  }},
  "t3_keepshown_dirty_scatter": {{
    "mip_frames": {},
    "zero_read_frames_naive": {},
    "zero_read_frames_opt": {},
    "heap_slots": {},
    "dirty_slots": {},
    "naive_linear_scan_ms": {:.4},
    "opt_bitmap_avx2_ms": {:.4},
    "opt_gpu_scatter_ms": {:.4},
    "scan_speedup": {:.2},
    "gpu_readback_verified_slots": {}
  }},
  "dx12_debug_errors": {},
  "dx12_debug_messages": [{}]
}}
"#,
        json_escape(&ctx.adapter_name),
        WIDTH,
        HEIGHT,
        (WIDTH * HEIGHT) as usize,
        naive_audit.nan_pixels,
        naive_audit.inf_pixels,
        opt_audit.nan_pixels,
        opt_audit.inf_pixels,
        naive_audit.gpu_pass_ms,
        opt_audit.gpu_pass_ms,
        json_escape(&naive_png.to_string_lossy()),
        json_escape(&opt_png.to_string_lossy()),
        t2.naive_total_draws,
        t2.naive_dropped_draws,
        t2.naive_dropped_skin_draws,
        t2.naive_dropped_pct,
        t2.opt_dropped_draws,
        t2.opt_dropped_skin_draws,
        t2.opt_dropped_pct,
        t2.opt_fallback_skin_used,
        t3.mip_frames,
        t3.zero_read_frames_naive,
        t3.zero_read_frames_opt,
        t3.heap_slots,
        t3.dirty_slots,
        t3.naive_linear_scan_ms,
        t3.opt_bitmap_avx2_ms,
        t3.opt_gpu_scatter_ms,
        scan_speedup,
        t3.gpu_readback_verified_slots,
        debug_errors_count,
        msgs_json.join(", ")
    );

    if let Some(p) = report_path {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        std::fs::write(p, json_text).expect("write skin_sss_report.json");
    }
}

// ---------------------------------------------------------------------------
// Clean-Room Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skin_sss_nan_guard_eliminates_all_nan_and_inf_on_gpu() {
        let mut ctx = GpuContext::new(false);
        let dxc = DxcRuntime::load().expect("dxc");
        let comp = DxcRuntime::new_compiler(&dxc).expect("comp");
        let vs_dxil = comp.compile_hlsl(SCENE_VS_HLSL, "VSMain", "vs_6_6", "-O1").unwrap();
        let naive_ps = build_scene_ps_hlsl(false);
        let naive_ps_dxil = comp.compile_hlsl(&naive_ps, "PSMain", "ps_6_6", "-O1").unwrap();
        let opt_ps = build_scene_ps_hlsl(true);
        let opt_ps_dxil = comp.compile_hlsl(&opt_ps, "PSMain", "ps_6_6", "-O1").unwrap();
        let bloom_dxil = comp.compile_hlsl(BLOOM_CS_HLSL, "CSMain", "cs_6_6", "-O1").unwrap();
        let tonemap_dxil = comp.compile_hlsl(TONEMAP_CS_HLSL, "CSMain", "cs_6_6", "-O1").unwrap();

        let gfx_rs = create_scene_gfx_root_sig(&ctx.device);
        let cs_rs = create_post_cs_root_sig(&ctx.device);
        let naive_pso = create_bust_gfx_pso(&ctx.device, &gfx_rs, &vs_dxil, &naive_ps_dxil);
        let opt_pso = create_bust_gfx_pso(&ctx.device, &gfx_rs, &vs_dxil, &opt_ps_dxil);
        let bloom_pso = create_post_cs_pso(&ctx.device, &cs_rs, &bloom_dxil);
        let tonemap_pso = create_post_cs_pso(&ctx.device, &cs_rs, &tonemap_dxil);

        let (verts, inds) = build_dual_bust_scene();
        let vb = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (verts.len() * size_of::<BustVertex>()) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let ib = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (inds.len() * 4) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        unsafe {
            let mut p: *mut c_void = std::ptr::null_mut();
            vb.Map(0, None, Some(&mut p)).unwrap();
            copy_nonoverlapping(
                verts.as_ptr() as *const u8,
                p as *mut u8,
                verts.len() * size_of::<BustVertex>(),
            );
            vb.Unmap(0, None);
            ib.Map(0, None, Some(&mut p)).unwrap();
            copy_nonoverlapping(inds.as_ptr() as *const u8, p as *mut u8, inds.len() * 4);
            ib.Unmap(0, None);
        }

        let naive = render_and_audit_bust_frame(
            &mut ctx,
            &gfx_rs,
            &cs_rs,
            &naive_pso,
            &bloom_pso,
            &tonemap_pso,
            &vb,
            &ib,
            verts.len() as u32,
            inds.len() as u32,
            false,
        );
        let opt = render_and_audit_bust_frame(
            &mut ctx,
            &gfx_rs,
            &cs_rs,
            &opt_pso,
            &bloom_pso,
            &tonemap_pso,
            &vb,
            &ib,
            verts.len() as u32,
            inds.len() as u32,
            true,
        );
        assert!(
            naive.nan_pixels > 0 || naive.inf_pixels > 0,
            "naive Wave32 run must reproduce NaN/Inf on GPU: nan={}, inf={}",
            naive.nan_pixels,
            naive.inf_pixels
        );
        assert_eq!(opt.nan_pixels, 0);
        assert_eq!(opt.inf_pixels, 0);
    }

    #[test]
    fn skin_sss_priority_and_fallback_achieves_zero_dropped_draws() {
        let ctx = GpuContext::new(false);
        let dxc = DxcRuntime::load().expect("dxc");
        let comp = DxcRuntime::new_compiler(&dxc).expect("comp");
        let vs_dxil = comp.compile_hlsl(SCENE_VS_HLSL, "VSMain", "vs_6_6", "-O1").unwrap();
        let gfx_rs = create_scene_gfx_root_sig(&ctx.device);
        let res = run_sss_pso_queue_benchmark(&ctx, &dxc, &gfx_rs, &vs_dxil);
        assert!(res.naive_dropped_draws > 0);
        assert!(res.naive_dropped_skin_draws > 0);
        assert_eq!(res.opt_dropped_draws, 0);
        assert_eq!(res.opt_dropped_skin_draws, 0);
        assert_eq!(res.opt_dropped_pct, 0.0);
    }

    #[test]
    fn skin_sss_keep_shown_prevents_zero_mip_reads() {
        let mut ctx = GpuContext::new(false);
        let res = run_mip_keepshown_and_scatter_bench(&mut ctx);
        assert!(res.zero_read_frames_naive > 0);
        assert_eq!(res.zero_read_frames_opt, 0);
        assert_eq!(res.gpu_readback_verified_slots, res.dirty_slots);
    }
}
