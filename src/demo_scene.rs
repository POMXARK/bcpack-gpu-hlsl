use crate::assets::{
    TextureAsset, ASSET_NAMES, TEX_BC1_BYTES, TEX_H, TEX_W, TILE_BC1_BYTES, TILE_BLOCKS,
    TILE_STRIPS,
};
use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, drop_barriers, heap_props, pack_batch,
    transition_barrier, upload_slice, BcpackGpuDecoder, GpuContext,
};
use crate::meshlets::{
    build_city_meshlet_library, build_highpoly_city_instances, extract_frustum_planes, MeshFrameCB,
    MeshletPipeline,
};
use crate::meshes::{build_all_meshes, MeshRange, Vertex};
use crate::raytracing::{tex_reflection_color, RayTracingScene, RtInstanceInput, MAX_RT_INSTANCES};
use crate::rtgi::RtgiPipeline;
use crate::world_streaming::WorldStreamManager;
use bcpack::bc1::{self, QuadTable, Window};
use rayon::prelude::*;
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::mem::{size_of, ManuallyDrop};
use std::path::PathBuf;
use std::ptr::copy_nonoverlapping;
use std::time::Instant;
use windows::core::{w, Interface, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_A, VK_C, VK_D, VK_DOWN, VK_E, VK_ESCAPE, VK_F, VK_G, VK_LEFT, VK_M,
    VK_N, VK_Q, VK_R, VK_RIGHT, VK_S, VK_SPACE, VK_UP, VK_W,
};
use windows::Win32::UI::WindowsAndMessaging::*;

pub const SCENE_VS_DXIL: &[u8] = include_bytes!("../shaders/scene_vs.dxil");
pub const SCENE_PS_DXIL: &[u8] = include_bytes!("../shaders/scene_ps.dxil");
pub const SKY_VS_DXIL: &[u8] = include_bytes!("../shaders/sky_vs.dxil");
pub const SKY_PS_DXIL: &[u8] = include_bytes!("../shaders/sky_ps.dxil");

const WIN_W: u32 = 1280;
const WIN_H: u32 = 720;
const NUM_TEX: usize = ASSET_NAMES.len(); // 12
const MAX_DRAWS: usize = 320;
const CB_STRIDE: usize = 512;

#[repr(C)]
#[derive(Clone, Copy)]
struct SkyConstants {
    inv_view_proj: [f32; 16],
    camera_pos: [f32; 3],
    time: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrawConstants {
    mvp: [f32; 16],
    model: [f32; 16],
    tex_index: u32,
    emissive: f32,
    time: f32,
    uv_scale: f32,
    camera_pos: [f32; 4],
    point_lights: [[f32; 4]; 4],
    light_colors: [[f32; 4]; 4],
}

#[derive(Clone, Copy)]
struct Mat4([f32; 16]);

impl Mat4 {
    fn mul(&self, rhs: &Mat4) -> Mat4 {
        let mut out = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += self.0[k * 4 + row] * rhs.0[col * 4 + k];
                }
                out[col * 4 + row] = sum;
            }
        }
        Mat4(out)
    }

    fn trs(pos: [f32; 3], rot_y: f32, scale: [f32; 3]) -> Mat4 {
        let (s, c) = rot_y.sin_cos();
        Mat4([
            c * scale[0], 0.0, -s * scale[0], 0.0,
            0.0, scale[1], 0.0, 0.0,
            s * scale[2], 0.0, c * scale[2], 0.0,
            pos[0], pos[1], pos[2], 1.0,
        ])
    }

    fn perspective_lh(fov_y_rad: f32, aspect: f32, zn: f32, zf: f32) -> Mat4 {
        let ys = 1.0 / (fov_y_rad * 0.5).tan();
        let xs = ys / aspect;
        let q = zf / (zf - zn);
        Mat4([
            xs, 0.0, 0.0, 0.0,
            0.0, ys, 0.0, 0.0,
            0.0, 0.0, q, 1.0,
            0.0, 0.0, -zn * q, 0.0,
        ])
    }

    fn look_at_lh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
        let f = norm3(sub3(target, eye));
        let r = norm3(cross3(up, f));
        let u = cross3(f, r);
        Mat4([
            r[0], u[0], f[0], 0.0,
            r[1], u[1], f[1], 0.0,
            r[2], u[2], f[2], 0.0,
            -dot3(r, eye), -dot3(u, eye), -dot3(f, eye), 1.0,
        ])
    }

    fn inverse(&self) -> Mat4 {
        let m = &self.0;
        let mut inv = [0.0f32; 16];
        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
            + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
            - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
            + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
            - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
            - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
            + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
            - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
            + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
            + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
            - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
            + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
            - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
            - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
            + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
            - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
            + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

        let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        let inv_det = if det.abs() > 1e-8 { 1.0 / det } else { 1.0 };
        for v in &mut inv {
            *v *= inv_det;
        }
        Mat4(inv)
    }
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = dot3(v, v).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DemoMode {
    GpuOnly,
    CpuOnly,
    Toggleable,
}

pub struct DemoConfig {
    pub mode: DemoMode,
    pub initial_chunks: usize, // number of 4096-block tiles/frame (256 = 8.0 MiB/frame)
    pub multi_core_cpu: bool,
    pub enable_debug: bool,
    pub start_rt_enabled: bool,
    pub start_rtgi_enabled: bool,
    pub start_world_stream: bool,
    pub start_mesh_shaders: bool,
    pub render_1440p: bool,
    pub nvme_stream: bool,
    pub assets_dir: PathBuf,
    pub vram_cap_mb: usize,
    pub screenshot_path: Option<PathBuf>,
    pub screenshot_frames: u32,
    pub measure_frames: Option<u32>,
    pub record_dir: Option<PathBuf>,
    pub record_frames: u32,
}

struct WindowState {
    mode: DemoMode,
    use_gpu: bool,
    multi_core_cpu: bool,
    sfs_enabled: bool,
    rt_enabled: bool,
    rtgi_enabled: bool,
    mesh_shaders_enabled: bool,
    tiles_per_frame: usize,
    vsync_enabled: bool,
    cam_mode: u8, // 0 = Chase, 1 = Free Drive, 2 = Helicopter
    should_quit: bool,
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowState;
    match msg {
        WM_CLOSE => {
            if !ptr.is_null() {
                (*ptr).should_quit = true;
            }
            let _ = DestroyWindow(hwnd);
            PostQuitMessage(0);
            LRESULT(0)
        }
        WM_DESTROY => {
            if !ptr.is_null() {
                (*ptr).should_quit = true;
            }
            PostQuitMessage(0);
            LRESULT(0)
        }
        WM_KEYDOWN => {
            if !ptr.is_null() {
                let st = &mut *ptr;
                let vk = wparam.0 as u16;
                if vk == VK_ESCAPE.0 {
                    st.should_quit = true;
                    let _ = DestroyWindow(hwnd);
                    PostQuitMessage(0);
                } else if vk == VK_SPACE.0 && st.mode == DemoMode::Toggleable {
                    st.use_gpu = !st.use_gpu;
                } else if vk == b'L' as u16 {
                    st.vsync_enabled = !st.vsync_enabled;
                } else if vk == VK_F.0 {
                    st.sfs_enabled = !st.sfs_enabled;
                } else if vk == VK_R.0 {
                    st.rt_enabled = !st.rt_enabled;
                } else if vk == VK_G.0 {
                    st.rtgi_enabled = !st.rtgi_enabled;
                } else if vk == VK_N.0 {
                    st.mesh_shaders_enabled = !st.mesh_shaders_enabled;
                } else if vk == VK_M.0 && (st.mode == DemoMode::CpuOnly || !st.use_gpu) {
                    st.multi_core_cpu = !st.multi_core_cpu;
                } else if vk == VK_C.0 {
                    st.cam_mode = (st.cam_mode + 1) % 3;
                } else if vk == VK_UP.0 {
                    st.tiles_per_frame = match st.tiles_per_frame {
                        48 => 128,
                        128 => 256,
                        256 => 512,
                        _ => 1024,
                    };
                } else if vk == VK_DOWN.0 {
                    st.tiles_per_frame = match st.tiles_per_frame {
                        1024 => 512,
                        512 => 256,
                        256 => 128,
                        _ => 48,
                    };
                }
            }
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

struct DrawObj {
    mesh: MeshRange,
    pos: [f32; 3],
    rot_y: f32,
    scale: [f32; 3],
    tex: u32,
    emissive: f32,
    uv_scale: f32,
    spec_power: f32,
}

#[inline]
fn wrap_z(base_z: f32, hero_z: f32, span: f32) -> f32 {
    let rel = (base_z - hero_z + span * 0.25).rem_euclid(span) - span * 0.25;
    hero_z + rel
}

fn build_highway_frame(
    meshes: &crate::meshes::SceneMeshes,
    t_sec: f32,
    hero_z: f32,
    ground_only: bool,
) -> (Vec<DrawObj>, [[f32; 4]; 4], [[f32; 4]; 4]) {
    let mut objs = Vec::with_capacity(if ground_only { 48 } else { 180 });
    let span = 160.0f32;

    // 1. Road, boardwalk, golden beach, animated ocean (tiled along Z moving with hero_z)
    let tile_len = 40.0f32;
    let base_seg = (hero_z / tile_len).floor() * tile_len;
    for s in -1..=4 {
        let cz = base_seg + s as f32 * tile_len;
        // 4-lane coastal highway
        objs.push(DrawObj { mesh: meshes.unit_box, pos: [0.0, -0.10, cz], rot_y: 0.0, scale: [14.0, 0.2, tile_len], tex: 0, emissive: 0.04, uv_scale: 2.0, spec_power: 48.0 });
        // Left & right promenade sidewalks
        objs.push(DrawObj { mesh: meshes.unit_box, pos: [-9.5, -0.04, cz], rot_y: 0.0, scale: [5.0, 0.32, tile_len], tex: 3, emissive: 0.03, uv_scale: 3.0, spec_power: 12.0 });
        objs.push(DrawObj { mesh: meshes.unit_box, pos: [9.5, -0.04, cz], rot_y: 0.0, scale: [5.0, 0.32, tile_len], tex: 3, emissive: 0.03, uv_scale: 3.0, spec_power: 12.0 });
        // Left wide city ground pad (supports 6 skyline avenues)
        objs.push(DrawObj { mesh: meshes.unit_box, pos: [-68.0, -0.14, cz], rot_y: 0.0, scale: [112.0, 0.2, tile_len], tex: 11, emissive: 0.02, uv_scale: 8.0, spec_power: 8.0 });
        // Golden beach sand + wide ocean bay on the right
        objs.push(DrawObj { mesh: meshes.unit_box, pos: [17.0, -0.22, cz], rot_y: 0.0, scale: [10.0, 0.2, tile_len], tex: 7, emissive: 0.02, uv_scale: 4.0, spec_power: 8.0 });
        objs.push(DrawObj { mesh: meshes.unit_box, pos: [52.0, -0.45, cz], rot_y: 0.0, scale: [60.0, 0.2, tile_len], tex: 4, emissive: 0.14, uv_scale: 6.0, spec_power: 72.0 });
    }

    // 4. Boulevard Streetlamps positions for Dynamic Point Lights
    let mut lamp_positions: Vec<f32> = Vec::with_capacity(8);
    for k in 0..4 {
        let z = wrap_z(k as f32 * 40.0 + 16.0, hero_z, span);
        lamp_positions.push(z);
        if !ground_only {
            objs.push(DrawObj { mesh: meshes.street_lamp, pos: [-7.4, 0.0, z], rot_y: 0.0, scale: [1.0, 1.0, 1.0], tex: 11, emissive: 0.35, uv_scale: 1.0, spec_power: 32.0 });
            objs.push(DrawObj { mesh: meshes.street_lamp, pos: [7.4, 0.0, z], rot_y: std::f32::consts::PI, scale: [1.0, 1.0, 1.0], tex: 11, emissive: 0.35, uv_scale: 1.0, spec_power: 32.0 });
        }
    }
    lamp_positions.sort_by(|a, b| (a - hero_z).abs().partial_cmp(&(b - hero_z).abs()).unwrap());

    // 5. Overhead Highway Synthwave Gantry Sign
    let arch_z = wrap_z(72.0, hero_z, span);
    objs.push(DrawObj { mesh: meshes.unit_box, pos: [-7.3, 4.4, arch_z], rot_y: 0.0, scale: [0.5, 8.8, 0.5], tex: 1, emissive: 0.25, uv_scale: 1.0, spec_power: 24.0 });
    objs.push(DrawObj { mesh: meshes.unit_box, pos: [7.3, 4.4, arch_z], rot_y: 0.0, scale: [0.5, 8.8, 0.5], tex: 1, emissive: 0.25, uv_scale: 1.0, spec_power: 24.0 });
    objs.push(DrawObj { mesh: meshes.unit_box, pos: [0.0, 8.8, arch_z], rot_y: 0.0, scale: [15.4, 4.6, 0.5], tex: 6, emissive: 0.65, uv_scale: 1.0, spec_power: 16.0 });

    let hero_x = 2.4 + (t_sec * 0.7).sin() * 0.45;
    if !ground_only {
        // 2. Multi-tier Art-Deco Ocean Drive skyline on the left + boutique beach pavilions
        let left_slots = [
            (0.0f32, 11.0, 26.0, 14.0, 1u32, 0.36),
            (20.0, 10.0, 16.0, 13.0, 2, 0.24),
            (40.0, 12.0, 34.0, 15.0, 1, 0.42),
            (60.0, 9.5, 13.0, 12.0, 2, 0.22),
            (80.0, 11.5, 29.0, 14.0, 1, 0.38),
            (100.0, 10.0, 19.0, 13.0, 2, 0.25),
            (120.0, 12.0, 38.0, 16.0, 1, 0.44),
            (140.0, 10.0, 15.0, 13.0, 2, 0.22),
        ];
        for &(bz, w, h, d, tex, em) in &left_slots {
            let z = wrap_z(bz, hero_z, span);
            objs.push(DrawObj { mesh: meshes.unit_box, pos: [-17.5, 2.2, z], rot_y: 0.0, scale: [w + 1.2, 4.4, d + 0.8], tex: 10, emissive: 0.32, uv_scale: 1.5, spec_power: 24.0 });
            objs.push(DrawObj { mesh: meshes.unit_box, pos: [-18.2, 4.4 + h * 0.5, z], rot_y: 0.0, scale: [w, h, d], tex, emissive: em, uv_scale: 2.0, spec_power: 32.0 });
            objs.push(DrawObj { mesh: meshes.unit_box, pos: [-18.2, 4.4 + h + 1.1, z], rot_y: 0.0, scale: [w * 0.58, 2.2, d * 0.58], tex: 11, emissive: 0.12, uv_scale: 1.0, spec_power: 16.0 });
        }

        for &(bz, h, tex) in &[(28.0f32, 11.0, 2u32), (108.0, 14.0, 1u32)] {
            let z = wrap_z(bz, hero_z, span);
            objs.push(DrawObj { mesh: meshes.unit_box, pos: [14.5, 1.8, z], rot_y: 0.0, scale: [6.0, 3.6, 11.0], tex: 10, emissive: 0.28, uv_scale: 1.5, spec_power: 24.0 });
            objs.push(DrawObj { mesh: meshes.unit_box, pos: [14.8, 3.6 + h * 0.5, z], rot_y: 0.0, scale: [5.2, h, 9.5], tex, emissive: 0.28, uv_scale: 1.6, spec_power: 28.0 });
        }

        // 3. Curved 3D Palm Trees along both promenades (swaying gently in the ocean breeze)
        for k in 0..8 {
            let z = wrap_z(k as f32 * 20.0 + 6.0, hero_z, span);
            let sway = (t_sec * 1.6 + k as f32 * 0.9).sin() * 0.06;
            objs.push(DrawObj { mesh: meshes.palm_trunk, pos: [-8.2, 0.0, z], rot_y: 0.15 + sway, scale: [1.0, 1.0, 1.0], tex: 3, emissive: 0.0, uv_scale: 2.0, spec_power: 8.0 });
            objs.push(DrawObj { mesh: meshes.palm_canopy, pos: [-8.2 + 0.84, 5.55, z + 0.12], rot_y: k as f32 * 0.7 + sway, scale: [1.15, 1.15, 1.15], tex: 5, emissive: 0.10, uv_scale: 1.0, spec_power: 18.0 });
            objs.push(DrawObj { mesh: meshes.palm_trunk, pos: [8.4, 0.0, z + 10.0], rot_y: std::f32::consts::PI - 0.1 + sway, scale: [1.05, 1.05, 1.05], tex: 3, emissive: 0.0, uv_scale: 2.0, spec_power: 8.0 });
            objs.push(DrawObj { mesh: meshes.palm_canopy, pos: [8.4 - 0.88, 5.80, z + 10.0], rot_y: k as f32 * 1.1 - sway, scale: [1.2, 1.2, 1.2], tex: 5, emissive: 0.10, uv_scale: 1.0, spec_power: 18.0 });
        }

        // 6. Hero Sports Car
        let hero_yaw = (t_sec * 0.7).cos() * 0.04;
        objs.push(DrawObj {
            mesh: meshes.car_body,
            pos: [hero_x, 0.0, hero_z],
            rot_y: hero_yaw,
            scale: [1.05, 1.05, 1.05],
            tex: 8,
            emissive: 0.14,
            uv_scale: 1.0,
            spec_power: 96.0,
        });

        // 7. 10 Animated Traffic Cars
        let traffic = [
            (5.2f32, 26.0f32, 13.5f32, 0.0f32, 9u32),
            (2.4, 62.0, 14.5, 0.0, 9),
            (5.2, 105.0, 15.0, 0.0, 8),
            (2.4, 140.0, 13.0, 0.0, 9),
            (-2.4, 12.0, -14.0, std::f32::consts::PI, 9),
            (-5.2, 42.0, -16.0, std::f32::consts::PI, 8),
            (-2.4, 78.0, -15.0, std::f32::consts::PI, 9),
            (-5.2, 114.0, -17.0, std::f32::consts::PI, 8),
            (-2.4, 138.0, -14.0, std::f32::consts::PI, 9),
            (5.2, 150.0, 14.0, 0.0, 8),
        ];
        for &(lx, z0, spd, yaw, tex) in &traffic {
            let tz = wrap_z(z0 + t_sec * spd, hero_z, span);
            objs.push(DrawObj {
                mesh: meshes.car_body,
                pos: [lx, 0.0, tz],
                rot_y: yaw,
                scale: [1.0, 1.0, 1.0],
                tex,
                emissive: 0.12,
                uv_scale: 1.0,
                spec_power: 88.0,
            });
        }
    }

    // 4 Dynamic Point Lights: Hero Headlights, Hero Taillights, 2 Nearest Streetlamps
    let lights = [
        [hero_x, 1.2, hero_z + 3.2, 4.5],
        [hero_x, 1.0, hero_z - 2.6, 3.2],
        [-5.5, 6.4, lamp_positions[0], 5.0],
        [5.5, 6.4, lamp_positions[1], 5.0],
    ];
    let colors = [
        [0.95, 0.96, 1.00, 64.0], // cool white headlights + default spec power
        [1.00, 0.18, 0.28, 64.0], // crimson brake glow
        [1.00, 0.62, 0.25, 64.0], // warm streetlamp gold
        [0.25, 0.90, 1.00, 64.0], // cyan neon boardwalk glow
    ];

    (objs, lights, colors)
}

pub fn run_demo(
    assets: &[TextureAsset],
    window_tbl: &Window,
    quads_tbl: &QuadTable,
    cfg: DemoConfig,
) {
    unsafe {
        let mut ctx = GpuContext::new(cfg.enable_debug);
        let levels = [48usize, 128, 256, 512, 1024];
        let max_tiles = 1024usize;
        let max_blocks = max_tiles * TILE_BLOCKS;
        let max_stream = max_tiles * 48 * 1024;
        let mut gpu_dec = BcpackGpuDecoder::new(
            &mut ctx,
            window_tbl,
            quads_tbl,
            max_tiles,
            max_blocks,
            max_stream,
        );

        // Pre-pack 4K-tile batches (first 48 tiles = 12 textures x 4 strips, remaining tiles simulate open-world streaming)
        let batches: Vec<(usize, _)> = levels
            .iter()
            .map(|&n| {
                let batch = pack_batch((0..n).map(|i| {
                    let tex_i = (i / TILE_STRIPS) % NUM_TEX;
                    let strip = i % TILE_STRIPS;
                    (assets[tex_i].bcpack_tiles[strip].as_slice(), TILE_BLOCKS)
                }));
                (n, batch)
            })
            .collect();

        // Win32 Window
        let hinst = GetModuleHandleW(None).expect("GetModuleHandleW");
        let class_name = w!("BcpackGpuDemoWndV2");
        let wc = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wnd_proc),
            hInstance: hinst.into(),
            hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
            lpszClassName: class_name,
            ..Default::default()
        };
        RegisterClassExW(&wc);

        let mut rect = RECT {
            left: 0,
            top: 0,
            right: WIN_W as i32,
            bottom: WIN_H as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false).unwrap();

        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class_name,
            w!("BCPack GPU Highway Demo v0.2 (RTX 5080 HLSL SM 6.6) - Starting..."),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            hinst,
            None,
        )
        .expect("CreateWindowExW");

        let init_tiles = levels
            .iter()
            .copied()
            .min_by_key(|&x| x.abs_diff(cfg.initial_chunks))
            .unwrap_or(256);
        let init_use_gpu = cfg.mode != DemoMode::CpuOnly;

        let mut wstate = Box::new(WindowState {
            mode: cfg.mode,
            use_gpu: init_use_gpu,
            multi_core_cpu: cfg.multi_core_cpu,
            sfs_enabled: cfg.start_world_stream,
            rt_enabled: false,
            rtgi_enabled: false,
            mesh_shaders_enabled: false,
            tiles_per_frame: init_tiles,
            vsync_enabled: false,
            cam_mode: 0,
            should_quit: false,
        });
        let mut world_mgr = WorldStreamManager::new(cfg.vram_cap_mb.clamp(4, 16384) * 1024 * 1024);
        let mut last_world_res_mb = 0.0f64;
        let last_world_cap_mb = world_mgr.cap_bytes() as f64 / (1024.0 * 1024.0);
        let mut last_world_evict = 0u64;
        let mut last_world_frag = 0.0f64;
        let mut last_world_peak_mbs = 0.0f64;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (&mut *wstate as *mut WindowState) as isize);
        let _ = ShowWindow(hwnd, SW_SHOW);

        // SwapChain (3 backbuffers for non-blocking Ping-Pong overlap)
        let sc_desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: WIN_W,
            Height: WIN_H,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 3,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: 0,
        };
        let sc1 = ctx
            .factory
            .CreateSwapChainForHwnd(&ctx.queue, hwnd, &sc_desc, None, None)
            .expect("CreateSwapChainForHwnd");
        let swap_chain: IDXGISwapChain3 = sc1.cast().unwrap();

        let rtv_heap: ID3D12DescriptorHeap = ctx
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
            .unwrap();
        let dsv_heap: ID3D12DescriptorHeap = ctx
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
            .unwrap();
        let srv_heap: ID3D12DescriptorHeap = ctx
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: NUM_TEX as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap();

        let rtv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let srv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        let mut backbuffers = Vec::with_capacity(3);
        let rtv_base = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        for i in 0..3u32 {
            let bb: ID3D12Resource = swap_chain.GetBuffer(i).unwrap();
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_base.ptr + (i * rtv_inc) as usize,
            };
            ctx.device.CreateRenderTargetView(&bb, None, handle);
            backbuffers.push(bb);
        }

        // Depth buffer (matches internal HDR render resolution: 1280x720 default or 2560x1440 via --1440p)
        let (render_w, render_h) = if cfg.render_1440p { (2560u32, 1440u32) } else { (WIN_W, WIN_H) };
        let depth_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: render_w as u64,
            Height: render_h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        };
        let depth_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 },
            },
        };
        let mut depth_res: Option<ID3D12Resource> = None;
        ctx.device
            .CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &depth_desc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                Some(&depth_clear),
                &mut depth_res,
            )
            .unwrap();
        let depth_res = depth_res.unwrap();
        ctx.device.CreateDepthStencilView(&depth_res, None, dsv_heap.GetCPUDescriptorHandleForHeapStart());

        // 12 live 512x512 DXGI_FORMAT_BC1_UNORM textures
        let mut textures = Vec::with_capacity(NUM_TEX);
        let srv_base = srv_heap.GetCPUDescriptorHandleForHeapStart();
        for i in 0..NUM_TEX {
            let t_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: TEX_W as u64,
                Height: TEX_H as u32,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_BC1_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_NONE,
            };
            let mut tex: Option<ID3D12Resource> = None;
            ctx.device
                .CreateCommittedResource(
                    &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                    D3D12_HEAP_FLAG_NONE,
                    &t_desc,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    None,
                    &mut tex,
                )
                .unwrap();
            let tex = tex.unwrap();

            let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_BC1_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            };
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: srv_base.ptr + i * srv_inc as usize,
            };
            ctx.device.CreateShaderResourceView(&tex, Some(&srv_desc), handle);
            textures.push(tex);
        }

        // CPU Upload staging buffer (12 x 128 KiB = 1.5 MiB)
        let cpu_upload_buf = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (NUM_TEX * TEX_BC1_BYTES) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );

        // Double-buffered per-draw constant buffer ring (2 x MAX_DRAWS x 512 bytes)
        let cb_upload = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (2 * MAX_DRAWS * CB_STRIDE) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let mut cb_mapped_ptr: *mut c_void = std::ptr::null_mut();
        cb_upload
            .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut cb_mapped_ptr))
            .unwrap();
        let cb_base_ptr = cb_mapped_ptr as *mut u8;
        let cb_gpu_va = cb_upload.GetGPUVirtualAddress();

        // Sky Root Signature + PSO (20 32-bit root constants at b0)
        let sky_param = [D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: (size_of::<SkyConstants>() / 4) as u32,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }];
        let sky_rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: sky_param.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                },
            },
        };
        let mut sky_rs_blob = None;
        D3D12SerializeVersionedRootSignature(&sky_rs_desc, &mut sky_rs_blob, None).unwrap();
        let sky_rs_blob = sky_rs_blob.unwrap();
        let sky_root_sig: ID3D12RootSignature = ctx
            .device
            .CreateRootSignature(
                0,
                std::slice::from_raw_parts(sky_rs_blob.GetBufferPointer() as *const u8, sky_rs_blob.GetBufferSize()),
            )
            .unwrap();

        let mut rt_formats = [DXGI_FORMAT_UNKNOWN; 8];
        rt_formats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        rt_formats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        rt_formats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;

        let mut sky_pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(sky_root_sig.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: SKY_VS_DXIL.as_ptr() as *const c_void,
                BytecodeLength: SKY_VS_DXIL.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: SKY_PS_DXIL.as_ptr() as *const c_void,
                BytecodeLength: SKY_PS_DXIL.len(),
            },
            BlendState: D3D12_BLEND_DESC {
                RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC {
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                }; 8],
                ..Default::default()
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: false.into(),
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: false.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 3,
            RTVFormats: rt_formats,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let sky_pso: ID3D12PipelineState = ctx.device.CreateGraphicsPipelineState(&sky_pso_desc).unwrap();
        let _ = ManuallyDrop::take(&mut sky_pso_desc.pRootSignature);

        // Scene Root Signature: Root CBV (b0) + DescriptorTable (t0..t11) + Root SRV t12 (TLAS) + Root SRV t13 (InstColors) + StaticSampler (s0)
        let range = D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: NUM_TEX as u32,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let gfx_params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 12,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 13,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
        ];
        let static_sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_ANISOTROPIC,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            MipLODBias: 0.0,
            MaxAnisotropy: 8,
            ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            MinLOD: 0.0,
            MaxLOD: D3D12_FLOAT32_MAX,
            ShaderRegister: 0,
            RegisterSpace: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        };
        let gfx_rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: gfx_params.len() as u32,
                    pParameters: gfx_params.as_ptr(),
                    NumStaticSamplers: 1,
                    pStaticSamplers: &static_sampler,
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                },
            },
        };
        let mut rs_blob = None;
        D3D12SerializeVersionedRootSignature(&gfx_rs_desc, &mut rs_blob, None).unwrap();
        let rs_blob = rs_blob.unwrap();
        let gfx_root_sig: ID3D12RootSignature = ctx
            .device
            .CreateRootSignature(
                0,
                std::slice::from_raw_parts(rs_blob.GetBufferPointer() as *const u8, rs_blob.GetBufferSize()),
            )
            .unwrap();

        let input_elems = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::s!("POSITION"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::s!("NORMAL"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 12,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::s!("TEXCOORD"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 24,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];

        let mut pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(gfx_root_sig.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: SCENE_VS_DXIL.as_ptr() as *const c_void,
                BytecodeLength: SCENE_VS_DXIL.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: SCENE_PS_DXIL.as_ptr() as *const c_void,
                BytecodeLength: SCENE_PS_DXIL.len(),
            },
            BlendState: D3D12_BLEND_DESC {
                RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC {
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                }; 8],
                ..Default::default()
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                FrontCounterClockwise: false.into(),
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                ..Default::default()
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: input_elems.as_ptr(),
                NumElements: input_elems.len() as u32,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 3,
            RTVFormats: rt_formats,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let gfx_pso: ID3D12PipelineState = ctx.device.CreateGraphicsPipelineState(&pso_desc).unwrap();
        let _ = ManuallyDrop::take(&mut pso_desc.pRootSignature);

        // Upload combined procedural meshes
        let meshes = build_all_meshes();
        let vb_buf = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (meshes.vertices.len() * size_of::<Vertex>()) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let ib_buf = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (meshes.indices.len() * size_of::<u16>()) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        upload_slice(&vb_buf, &meshes.vertices);
        upload_slice(&ib_buf, &meshes.indices);
        let vbv = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vb_buf.GetGPUVirtualAddress(),
            SizeInBytes: (meshes.vertices.len() * size_of::<Vertex>()) as u32,
            StrideInBytes: size_of::<Vertex>() as u32,
        };
        let ibv = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: ib_buf.GetGPUVirtualAddress(),
            SizeInBytes: (meshes.indices.len() * size_of::<u16>()) as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };

        // Build Hardware DXR 1.1 BLAS/TLAS scene (up to 2,048 instances) + 16-bit HDR RTGI Denoiser Pipeline
        let rt_scene = RayTracingScene::new(&mut ctx, &vb_buf, &ib_buf, &meshes);
        let rtgi_pipe = RtgiPipeline::new(&ctx, render_w, render_h);
        wstate.rt_enabled = cfg.start_rt_enabled && rt_scene.is_some();
        wstate.rtgi_enabled = cfg.start_rtgi_enabled && rt_scene.is_some();
        let mut tlas_inst_count = 0usize;

        // Optional real NVMe .bcpack_pkg container streaming (out-of-order disk seeks every frame)
        let mut nvme_pkg: Option<(File, Vec<(u64, u32)>, Vec<Vec<u8>>)> = if cfg.nvme_stream {
            let pkg_path = cfg.assets_dir.join("world_sectors.bcpack_pkg");
            let mut idx = Vec::with_capacity(768);
            let need_build = match std::fs::metadata(&pkg_path) {
                Ok(m) => m.len() < 1024 * 1024,
                Err(_) => true,
            };
            if need_build {
                std::fs::create_dir_all(&cfg.assets_dir).ok();
                let mut f = OpenOptions::new().create(true).truncate(true).write(true).open(&pkg_path).expect("create world_sectors.bcpack_pkg");
                for rep in 0..16usize {
                    for ti in 0..NUM_TEX {
                        for strip in 0..TILE_STRIPS {
                            let bytes = &assets[ti].bcpack_tiles[strip];
                            f.write_all(bytes).unwrap();
                            let _ = rep;
                        }
                    }
                }
                f.flush().unwrap();
            }
            let mut off = 0u64;
            for _ in 0..16usize {
                for ti in 0..NUM_TEX {
                    for strip in 0..TILE_STRIPS {
                        let len = assets[ti].bcpack_tiles[strip].len() as u32;
                        idx.push((off, len));
                        off += len as u64;
                    }
                }
            }
            let f = File::open(&pkg_path).expect("open world_sectors.bcpack_pkg");
            let scratch = (0..1024).map(|_| Vec::with_capacity(32768)).collect();
            Some((f, idx, scratch))
        } else {
            None
        };
        let mut last_nvme_ms = 0.0f64;

        // Build Hardware DX12 Ultimate Mesh Shader City Pipeline (as_6_5 / ms_6_5, 4.5M+ triangles)
        let meshlet_lib = build_city_meshlet_library();
        let (init_city_insts, city_work_items, city_total_tris) =
            build_highpoly_city_instances(&meshlet_lib, 0.0, 0.0);
        let meshlet_pipe = MeshletPipeline::new(
            &mut ctx,
            &meshlet_lib,
            &init_city_insts,
            &city_work_items,
            city_total_tris,
        );
        wstate.mesh_shaders_enabled = cfg.start_mesh_shaders && meshlet_pipe.is_some();
        let mut last_cull_stats = (0u32, 0u32, 0u32);

        let frame_allocs: [ID3D12CommandAllocator; 2] = [
            ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap(),
            ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap(),
        ];
        let frame_cmds: [ID3D12GraphicsCommandList; 2] = [
            ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &frame_allocs[0], None).unwrap(),
            ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &frame_allocs[1], None).unwrap(),
        ];
        frame_cmds[0].Close().unwrap();
        frame_cmds[1].Close().unwrap();

        let mut frame_fences = [0u64; 2];
        let mut pending_compute_fences: [Option<u64>; 2] = [None, None];
        let mut last_gpu_decode_ms = 0.0f64;

        let ss_row_pitch = WIN_W * 4;
        let ss_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, (ss_row_pitch * WIN_H) as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let proj = Mat4::perspective_lh(62.0f32.to_radians(), WIN_W as f32 / WIN_H as f32, 0.2, 260.0);

        let start_t = Instant::now();
        let mut last_frame_t = Instant::now();
        let mut last_title_t = Instant::now();
        let mut measure_start: Option<Instant> = None;
        let mut measure_decode_sum = 0.0f64;
        let mut frame_count = 0u32;
        let mut fps_accum = 0u32;
        let mut cpu_buf = vec![0u8; NUM_TEX * TEX_BC1_BYTES];

        // Free-cam state
        let mut free_pos = [0.0f32, 4.5, -10.0];
        let mut free_yaw = 0.0f32;

        println!("================================================================");
        match cfg.mode {
            DemoMode::GpuOnly => println!("NEW GPU Coastal Highway Demo v0.6 ({render_w}x{render_h} HDR + Mesh Shaders + DXR 1.1 RTGI) on: {}", ctx.adapter_name),
            DemoMode::CpuOnly => println!("OLD CPU Coastal Highway Demo v0.6 (bcpack-rs) on: {}", ctx.adapter_name),
            DemoMode::Toggleable => println!("BCPack Coastal Highway Live Demo v0.6 ({render_w}x{render_h} HDR) on: {}", ctx.adapter_name),
        }
        println!("Controls:");
        if cfg.mode == DemoMode::Toggleable {
            println!("  [SPACE]   Toggle GPU Async Compute (RTX 5080) <-> CPU (bcpack-rs)");
        }
        if cfg.mode != DemoMode::GpuOnly {
            println!("  [M]       Toggle CPU 1-Core <-> CPU Multi-Core");
        }
        println!("  [F]       Toggle Sampler Feedback Streaming (SFS) / 32x32 World Grid LRU");
        println!("  [N]       Toggle DX12 Mesh Shaders (4.0M+ Tris) <-> Classic VS Pipeline");
        println!("  [R]       Toggle DXR 1.1 Shadows + Reflections");
        println!("  [G]       Toggle DXR 1.1 Diffuse RTGI + 2-Pass Bilateral Denoiser");
        println!("  [L]       Toggle VSync (Limit FPS to Monitor Refresh Rate)");
        println!("  [C]       Camera Mode: Chase Car / Free Drive (WASD+QE) / Heli Orbit");
        println!("  [UP/DOWN] Streamed 4K-tiles/frame (48..1024 = 1.5..32 MiB BC1/frame)");
        println!("  [ESC]     Exit");
        println!("================================================================");
        if let Some(ref rdir) = cfg.record_dir {
            std::fs::create_dir_all(rdir).expect("create record_dir");
        }

        let mut msg = MSG::default();
        while !wstate.should_quit {
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                if msg.message == WM_QUIT {
                    wstate.should_quit = true;
                }
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if wstate.should_quit || !IsWindow(hwnd).as_bool() {
                break;
            }

            let dt = last_frame_t.elapsed().as_secs_f32().min(0.1);
            last_frame_t = Instant::now();

            let t_sec = if cfg.record_dir.is_some() {
                0.4 + frame_count as f32 / 30.0
            } else if cfg.screenshot_path.is_some() {
                1.65f32
            } else {
                start_t.elapsed().as_secs_f32()
            };
            let hero_z = t_sec * 18.0;

            // Camera selection (computed before streaming so WorldStreamManager tracks real camera pos & forward)
            let (eye, target) = match wstate.cam_mode {
                0 => {
                    // Chase Car Camera
                    let hx = 2.4 + (t_sec * 0.7).sin() * 0.45;
                    let eye = [hx - 0.6 + (t_sec * 0.35).sin() * 1.1, 3.6, hero_z - 9.2];
                    let tgt = [hx * 0.65, 1.8, hero_z + 18.0];
                    (eye, tgt)
                }
                1 => {
                    // Free Drive Camera (WASD + QE + Left/Right)
                    let spd = 22.0 * dt;
                    let rot_spd = 1.6 * dt;
                    if GetAsyncKeyState(VK_LEFT.0 as i32) < 0 { free_yaw -= rot_spd; }
                    if GetAsyncKeyState(VK_RIGHT.0 as i32) < 0 { free_yaw += rot_spd; }
                    let (sy, cy) = free_yaw.sin_cos();
                    if GetAsyncKeyState(VK_W.0 as i32) < 0 { free_pos[0] += sy * spd; free_pos[2] += cy * spd; }
                    if GetAsyncKeyState(VK_S.0 as i32) < 0 { free_pos[0] -= sy * spd; free_pos[2] -= cy * spd; }
                    if GetAsyncKeyState(VK_D.0 as i32) < 0 { free_pos[0] += cy * spd; free_pos[2] -= sy * spd; }
                    if GetAsyncKeyState(VK_A.0 as i32) < 0 { free_pos[0] -= cy * spd; free_pos[2] += sy * spd; }
                    if GetAsyncKeyState(VK_E.0 as i32) < 0 { free_pos[1] += spd; }
                    if GetAsyncKeyState(VK_Q.0 as i32) < 0 { free_pos[1] = (free_pos[1] - spd).max(0.8); }
                    let tgt = [free_pos[0] + sy * 15.0, free_pos[1] - 1.2, free_pos[2] + cy * 15.0];
                    (free_pos, tgt)
                }
                _ => {
                    // Helicopter Orbit Camera
                    let ang = t_sec * 0.32;
                    let eye = [ang.sin() * 20.0 + 4.0, 12.5, hero_z - ang.cos() * 24.0];
                    let tgt = [0.0, 3.5, hero_z + 8.0];
                    (eye, tgt)
                }
            };
            let cam_fwd = norm3(sub3(target, eye));

            let mut world_tile_pairs: Vec<(usize, usize)> = Vec::new();
            let world_active_frame = wstate.sfs_enabled && frame_count > 0;
            let dyn_batch;
            let (batch, cur_tiles) = if world_active_frame {
                // Scale highway Z into 32x32 250m sector coordinates so high-speed driving traverses & evicts sectors continuously
                let world_pos = [eye[0] * 3.0, eye[1], eye[2] * 3.5];
                let step = world_mgr.update(
                    world_pos,
                    cam_fwd,
                    frame_count as u64,
                    dt.max(0.001),
                    wstate.tiles_per_frame.min(max_tiles),
                );
                last_world_res_mb = step.resident_bytes as f64 / (1024.0 * 1024.0);
                last_world_evict = step.total_evictions;
                last_world_frag = step.fragmentation_pct;
                last_world_peak_mbs = step.peak_mbs;

                world_tile_pairs = step.tiles_to_decode;
                if world_tile_pairs.is_empty() {
                    let base = (frame_count as usize) * 4;
                    for k in 0..4usize {
                        let i = base + k;
                        world_tile_pairs.push(((i / TILE_STRIPS) % NUM_TEX, i % TILE_STRIPS));
                    }
                }
                if world_tile_pairs.len() > max_tiles {
                    world_tile_pairs.truncate(max_tiles);
                }
                if let Some((ref mut pkg_file, ref pkg_idx, ref mut scratch)) = nvme_pkg {
                    let tn = Instant::now();
                    for (k, &(tex_i, strip)) in world_tile_pairs.iter().enumerate() {
                        let slot = (((frame_count as usize + k) % 16) * NUM_TEX + tex_i) * TILE_STRIPS + strip;
                        let (off, len) = pkg_idx[slot % pkg_idx.len()];
                        scratch[k].resize(len as usize, 0);
                        pkg_file.seek(SeekFrom::Start(off)).unwrap();
                        pkg_file.read_exact(&mut scratch[k]).unwrap();
                    }
                    last_nvme_ms = tn.elapsed().as_secs_f64() * 1000.0;
                    dyn_batch = pack_batch(
                        scratch[..world_tile_pairs.len()]
                            .iter()
                            .map(|b| (b.as_slice(), TILE_BLOCKS)),
                    );
                } else {
                    dyn_batch = pack_batch(
                        world_tile_pairs
                            .iter()
                            .map(|&(tex_i, strip)| (assets[tex_i].bcpack_tiles[strip].as_slice(), TILE_BLOCKS)),
                    );
                }
                (&dyn_batch, world_tile_pairs.len())
            } else {
                let target_tiles = wstate.tiles_per_frame.min(max_tiles);
                if let Some((ref mut pkg_file, ref pkg_idx, ref mut scratch)) = nvme_pkg {
                    let tn = Instant::now();
                    for k in 0..target_tiles {
                        let slot = (k * 137 + (frame_count as usize) * 29) % pkg_idx.len();
                        let (off, len) = pkg_idx[slot];
                        scratch[k].resize(len as usize, 0);
                        pkg_file.seek(SeekFrom::Start(off)).unwrap();
                        pkg_file.read_exact(&mut scratch[k]).unwrap();
                    }
                    last_nvme_ms = tn.elapsed().as_secs_f64() * 1000.0;
                    dyn_batch = pack_batch(
                        scratch[..target_tiles]
                            .iter()
                            .map(|b| (b.as_slice(), TILE_BLOCKS)),
                    );
                    (&dyn_batch, target_tiles)
                } else if let Some((_, b)) = batches.iter().find(|(n, _)| *n == target_tiles) {
                    (b, target_tiles)
                } else {
                    dyn_batch = pack_batch((0..target_tiles).map(|i| {
                        let tex_i = (i / TILE_STRIPS) % NUM_TEX;
                        let strip = i % TILE_STRIPS;
                        (assets[tex_i].bcpack_tiles[strip].as_slice(), TILE_BLOCKS)
                    }));
                    (&dyn_batch, target_tiles)
                }
            };
            let streamed_mb = (cur_tiles * TILE_BC1_BYTES) as f64 / (1024.0 * 1024.0);

            let cur_slot = (frame_count as usize) & 1;
            let next_slot = 1 - cur_slot;

            // Wait only for 2-frames-old Direct work on cur_slot (0 CPU wait in steady state!)
            ctx.wait_direct_cpu(frame_fences[cur_slot]);
            if frame_count >= 2 {
                if wstate.use_gpu && (last_gpu_decode_ms == 0.0 || pending_compute_fences[cur_slot].is_some()) {
                    last_gpu_decode_ms = gpu_dec.read_slot_kernel_ms(&ctx, cur_slot);
                }
                if wstate.mesh_shaders_enabled {
                    if let Some(ref mp) = meshlet_pipe {
                        last_cull_stats = mp.read_cull_stats(cur_slot);
                    }
                }
            }

            let mut decomp_ms = 0.0f64;
            if !wstate.use_gpu {
                pending_compute_fences = [None, None];
                let t0 = Instant::now();
                let decoded_list: Vec<Vec<u8>> = if world_active_frame {
                    if wstate.multi_core_cpu {
                        world_tile_pairs
                            .par_iter()
                            .map(|&(tex_i, strip)| {
                                bc1::decode(&assets[tex_i].bcpack_tiles[strip], TILE_BLOCKS, window_tbl, quads_tbl)
                                    .unwrap()
                                    .blocks
                            })
                            .collect()
                    } else {
                        world_tile_pairs
                            .iter()
                            .map(|&(tex_i, strip)| {
                                bc1::decode(&assets[tex_i].bcpack_tiles[strip], TILE_BLOCKS, window_tbl, quads_tbl)
                                    .unwrap()
                                    .blocks
                            })
                            .collect()
                    }
                } else if wstate.multi_core_cpu {
                    (0..cur_tiles)
                        .into_par_iter()
                        .map(|i| {
                            let tex_i = (i / TILE_STRIPS) % NUM_TEX;
                            let strip = i % TILE_STRIPS;
                            bc1::decode(&assets[tex_i].bcpack_tiles[strip], TILE_BLOCKS, window_tbl, quads_tbl)
                                .unwrap()
                                .blocks
                        })
                        .collect()
                } else {
                    (0..cur_tiles)
                        .map(|i| {
                            let tex_i = (i / TILE_STRIPS) % NUM_TEX;
                            let strip = i % TILE_STRIPS;
                            bc1::decode(&assets[tex_i].bcpack_tiles[strip], TILE_BLOCKS, window_tbl, quads_tbl)
                                .unwrap()
                                .blocks
                        })
                        .collect()
                };
                if world_active_frame {
                    for (idx, &(tex_i, strip)) in world_tile_pairs.iter().enumerate() {
                        let slot = tex_i * TILE_STRIPS + strip;
                        cpu_buf[slot * TILE_BC1_BYTES..(slot + 1) * TILE_BC1_BYTES]
                            .copy_from_slice(&decoded_list[idx]);
                    }
                } else {
                    for ti in 0..(NUM_TEX * TILE_STRIPS).min(decoded_list.len()) {
                        cpu_buf[ti * TILE_BC1_BYTES..(ti + 1) * TILE_BC1_BYTES].copy_from_slice(&decoded_list[ti]);
                    }
                }
                upload_slice(&cpu_upload_buf, &cpu_buf);
                decomp_ms = t0.elapsed().as_secs_f64() * 1000.0;
            } else {
                // Overlapped Ping-Pong Async Compute:
                // 1) Ensure cur_slot has a submitted decode job
                let cur_cf = match pending_compute_fences[cur_slot].take() {
                    Some(v) => v,
                    None => gpu_dec.submit_slot_async_compute(&mut ctx, cur_slot, batch),
                };
                // 2) Kick next frame's decode onto next_slot asynchronously ONLY when DXR/RTGI and world stream are off
                let rt_active = rt_scene.is_some() && (wstate.rt_enabled || wstate.rtgi_enabled);
                let kick_next = !rt_active
                    && !world_active_frame
                    && cfg.screenshot_path.is_none()
                    && (cfg.record_dir.is_none() || frame_count + 1 < cfg.record_frames);
                if kick_next && pending_compute_fences[next_slot].is_none() {
                    ctx.wait_direct_cpu(frame_fences[next_slot]);
                    let nf = gpu_dec.submit_slot_async_compute(&mut ctx, next_slot, batch);
                    pending_compute_fences[next_slot] = Some(nf);
                }
                // GPU-side queue sync only (zero CPU stall!)
                ctx.direct_wait_for_compute(cur_cf);
            }

            let cmd_alloc = &frame_allocs[cur_slot];
            let cmd_list = &frame_cmds[cur_slot];
            cmd_alloc.Reset().unwrap();
            cmd_list.Reset(cmd_alloc, None).unwrap();

            // Copy decoded 4K tiles directly into the 12 DXGI_FORMAT_BC1_UNORM textures
            let mut to_copy = Vec::with_capacity(NUM_TEX + 1);
            for t in &textures {
                to_copy.push(transition_barrier(t, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST));
            }
            if wstate.use_gpu {
                to_copy.push(transition_barrier(gpu_dec.out_buffer(cur_slot), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE));
            }
            cmd_barriers(cmd_list, to_copy);

            let src_res = if wstate.use_gpu {
                gpu_dec.out_buffer(cur_slot)
            } else {
                &cpu_upload_buf
            };
            if world_active_frame {
                for (idx, &(ti, strip)) in world_tile_pairs.iter().enumerate() {
                    let tex = &textures[ti];
                    let src_offset = if wstate.use_gpu {
                        (idx * TILE_BC1_BYTES) as u64
                    } else {
                        ((ti * TILE_STRIPS + strip) * TILE_BC1_BYTES) as u64
                    };
                    let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(tex.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                    };
                    let mut src = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(src_res.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: src_offset,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                    Format: DXGI_FORMAT_BC1_UNORM,
                                    Width: TEX_W as u32,
                                    Height: (TEX_H / TILE_STRIPS) as u32,
                                    Depth: 1,
                                    RowPitch: (TEX_W / 4 * 8) as u32,
                                },
                            },
                        },
                    };
                    cmd_list.CopyTextureRegion(&dst, 0, (strip * (TEX_H / TILE_STRIPS)) as u32, 0, &src, None);
                    let _ = ManuallyDrop::take(&mut dst.pResource);
                    let _ = ManuallyDrop::take(&mut src.pResource);
                }
            } else {
                for (ti, tex) in textures.iter().enumerate() {
                    for strip in 0..TILE_STRIPS {
                        let tile_idx = ti * TILE_STRIPS + strip;
                        let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: ManuallyDrop::new(Some(tex.clone())),
                            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                        };
                        let mut src = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: ManuallyDrop::new(Some(src_res.clone())),
                            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                    Offset: (tile_idx * TILE_BC1_BYTES) as u64,
                                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                        Format: DXGI_FORMAT_BC1_UNORM,
                                        Width: TEX_W as u32,
                                        Height: (TEX_H / TILE_STRIPS) as u32, // 128 px strip
                                        Depth: 1,
                                        RowPitch: (TEX_W / 4 * 8) as u32, // 1024 bytes
                                    },
                                },
                            },
                        };
                        cmd_list.CopyTextureRegion(&dst, 0, (strip * (TEX_H / TILE_STRIPS)) as u32, 0, &src, None);
                        let _ = ManuallyDrop::take(&mut dst.pResource);
                        let _ = ManuallyDrop::take(&mut src.pResource);
                    }
                }
            }

            let mut to_srv = Vec::with_capacity(NUM_TEX + 1);
            for t in &textures {
                to_srv.push(transition_barrier(t, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
            }
            if wstate.use_gpu {
                to_srv.push(transition_barrier(gpu_dec.out_buffer(cur_slot), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON));
            }
            cmd_barriers(cmd_list, to_srv);

            // 1. Rebuild Hardware DXR 1.1 TLAS BEFORE binding RenderTarget / starting Render Pass
            let use_meshlets = meshlet_pipe.is_some() && wstate.mesh_shaders_enabled;
            let (objs, point_lights, light_colors) = build_highway_frame(&meshes, t_sec, hero_z, use_meshlets);
            let city_instances = if use_meshlets {
                Some(build_highpoly_city_instances(&meshlet_lib, t_sec, hero_z).0)
            } else {
                None
            };
            if let Some(ref rt) = rt_scene {
                let mut rt_inputs = Vec::with_capacity(MAX_RT_INSTANCES);
                for obj in &objs {
                    if rt_inputs.len() >= MAX_RT_INSTANCES {
                        break;
                    }
                    let mesh_idx = if obj.mesh.index_offset == meshes.unit_box.index_offset {
                        0
                    } else if obj.mesh.index_offset == meshes.car_body.index_offset {
                        1
                    } else if obj.mesh.index_offset == meshes.palm_trunk.index_offset {
                        2
                    } else if obj.mesh.index_offset == meshes.palm_canopy.index_offset {
                        3
                    } else {
                        4
                    };
                    let m = Mat4::trs(obj.pos, obj.rot_y, obj.scale).0;
                    let transform_3x4 = [
                        m[0], m[4], m[8], m[12],
                        m[1], m[5], m[9], m[13],
                        m[2], m[6], m[10], m[14],
                    ];
                    rt_inputs.push(RtInstanceInput {
                        mesh_idx,
                        transform_3x4,
                        color_emissive: tex_reflection_color(obj.tex, obj.emissive),
                    });
                }
                if let Some(ref cinsts) = city_instances {
                    // Populate TLAS with nearest high-poly city instances + outer cityscape/traffic proxies up to 1,536 instances
                    let mut near: Vec<(f32, &crate::meshlets::MeshletInstance)> = cinsts
                        .iter()
                        .map(|ci| {
                            let dz = ci.pos[2] - hero_z;
                            let dx = ci.pos[0];
                            (dz * dz + dx * dx, ci)
                        })
                        .collect();
                    near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    for (_, ci) in near {
                        if rt_inputs.len() >= MAX_RT_INSTANCES {
                            break;
                        }
                        let mesh_idx = match ci.tex_index {
                            8 | 9 => 1,
                            5 => 3,
                            11 => 4,
                            _ => 0,
                        };
                        let m = Mat4::trs(ci.pos, ci.rot_y, ci.scale).0;
                        let transform_3x4 = [
                            m[0], m[4], m[8], m[12],
                            m[1], m[5], m[9], m[13],
                            m[2], m[6], m[10], m[14],
                        ];
                        rt_inputs.push(RtInstanceInput {
                            mesh_idx,
                            transform_3x4,
                            color_emissive: tex_reflection_color(ci.tex_index, ci.emissive),
                        });
                    }
                    // Additional dynamic animated coastal props / outer traffic / neon towers to reach 1,536 TLAS instances
                    let mut extra_i = 0usize;
                    while rt_inputs.len() < 1536 && rt_inputs.len() < MAX_RT_INSTANCES {
                        let px = -140.0 - (extra_i % 6) as f32 * 20.0;
                        let pz = wrap_z(extra_i as f32 * 11.0 + t_sec * 4.0, hero_z, 320.0);
                        let ph = 18.0 + (extra_i % 5) as f32 * 6.0;
                        let m = Mat4::trs([px, ph * 0.5, pz], 0.0, [9.0, ph, 9.0]).0;
                        let transform_3x4 = [
                            m[0], m[4], m[8], m[12],
                            m[1], m[5], m[9], m[13],
                            m[2], m[6], m[10], m[14],
                        ];
                        let tex = if extra_i % 2 == 0 { 1 } else { 2 };
                        rt_inputs.push(RtInstanceInput {
                            mesh_idx: 0,
                            transform_3x4,
                            color_emissive: tex_reflection_color(tex, 0.30),
                        });
                        extra_i += 1;
                    }
                }
                tlas_inst_count = rt_inputs.len();
                rt.update_and_record_tlas(cmd_list, cur_slot, &rt_inputs);
            }

            let bb_idx = swap_chain.GetCurrentBackBufferIndex();
            let bb = &backbuffers[bb_idx as usize];
            let mut b_rt = [transition_barrier(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)];
            cmd_list.ResourceBarrier(&b_rt);
            drop_barriers(&mut b_rt);

            let rtv_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_base.ptr + (bb_idx * rtv_inc) as usize,
            };
            let dsv_handle = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            rtgi_pipe.begin_gbuffer_pass(cmd_list, dsv_handle);

            let view = Mat4::look_at_lh(eye, target, [0.0, 1.0, 0.0]);
            let vp_mat = proj.mul(&view);
            let inv_vp = vp_mat.inverse();

            // 2. Draw Sunset Synthwave Sky Dome
            cmd_list.SetGraphicsRootSignature(&sky_root_sig);
            cmd_list.SetPipelineState(&sky_pso);
            cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            let sky_cb = SkyConstants {
                inv_view_proj: inv_vp.0,
                camera_pos: eye,
                time: t_sec,
            };
            cmd_list.SetGraphicsRoot32BitConstants(
                0,
                (size_of::<SkyConstants>() / 4) as u32,
                &sky_cb as *const _ as *const c_void,
                0,
            );
            cmd_list.DrawInstanced(3, 1, 0, 0);

            // 3. Draw Highway, Art-Deco City, Curved Palms, Streetlamps, Hero & Traffic Cars
            cmd_list.SetGraphicsRootSignature(&gfx_root_sig);
            cmd_list.SetPipelineState(&gfx_pso);
            cmd_list.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
            cmd_list.SetGraphicsRootDescriptorTable(1, srv_heap.GetGPUDescriptorHandleForHeapStart());
            if let Some(ref rt) = rt_scene {
                cmd_list.SetGraphicsRootShaderResourceView(2, rt.tlas_gpu_va(cur_slot));
                cmd_list.SetGraphicsRootShaderResourceView(3, rt.inst_colors_gpu_va(cur_slot));
            }
            cmd_list.IASetVertexBuffers(0, Some(&[vbv]));
            cmd_list.IASetIndexBuffer(Some(&ibv));

            let rt_flag = (if rt_scene.is_some() && wstate.rt_enabled { 1.0f32 } else { 0.0f32 })
                + (if rt_scene.is_some() && wstate.rtgi_enabled { 2.0f32 } else { 0.0f32 });
            for (di, obj) in objs.iter().take(MAX_DRAWS).enumerate() {
                let model = Mat4::trs(obj.pos, obj.rot_y, obj.scale);
                let mvp = vp_mat.mul(&model);
                let mut lcols = light_colors;
                lcols[0][3] = obj.spec_power;
                let dc = DrawConstants {
                    mvp: mvp.0,
                    model: model.0,
                    tex_index: obj.tex,
                    emissive: obj.emissive,
                    time: t_sec,
                    uv_scale: obj.uv_scale,
                    camera_pos: [eye[0], eye[1], eye[2], rt_flag],
                    point_lights,
                    light_colors: lcols,
                };
                let cb_off = cur_slot * MAX_DRAWS * CB_STRIDE + di * CB_STRIDE;
                copy_nonoverlapping(
                    &dc as *const DrawConstants as *const u8,
                    cb_base_ptr.add(cb_off),
                    size_of::<DrawConstants>(),
                );
                cmd_list.SetGraphicsRootConstantBufferView(0, cb_gpu_va + cb_off as u64);
                cmd_list.DrawIndexedInstanced(obj.mesh.index_count, 1, obj.mesh.index_offset, 0, 0);
            }

            // 4. Hardware DX12 Ultimate Mesh Shader City Pass (as_6_5 / ms_6_5 + Wave32 Frustum Culling, 4.5M+ tris)
            if use_meshlets {
                if let (Some(ref mp), Some(ref cinsts)) = (meshlet_pipe.as_ref(), city_instances.as_ref()) {
                    let mut m_lights = light_colors;
                    m_lights[0][3] = 32.0;
                    let mcb = MeshFrameCB {
                        view_proj: vp_mat.0,
                        frustum_planes: extract_frustum_planes(&vp_mat.0),
                        camera_pos: [eye[0], eye[1], eye[2], rt_flag],
                        point_lights,
                        light_colors: m_lights,
                        total_work_items: mp.total_work_items,
                        time: t_sec,
                        max_cull_dist: 245.0,
                        pad: 0,
                    };
                    let (tlas_va, colors_va) = if let Some(ref rt) = rt_scene {
                        (rt.tlas_gpu_va(cur_slot), rt.inst_colors_gpu_va(cur_slot))
                    } else {
                        (0u64, 0u64)
                    };
                    mp.record_dispatch(
                        cmd_list,
                        cur_slot,
                        cinsts,
                        &mcb,
                        srv_heap.GetGPUDescriptorHandleForHeapStart(),
                        tlas_va,
                        colors_va,
                    );
                }
            }

            // 5. 2-Pass A-Trous Bilateral Denoiser + HDR Composite onto SwapChain Backbuffer
            rtgi_pipe.run_denoise_and_composite(
                cmd_list,
                rtv_handle,
                WIN_W,
                WIN_H,
                rt_scene.is_some() && wstate.rtgi_enabled,
            );

            let take_shot = cfg.screenshot_path.is_some() && frame_count + 1 >= cfg.screenshot_frames;
            let save_rec_frame = cfg.record_dir.is_some() && frame_count < cfg.record_frames;
            if take_shot || save_rec_frame {
                let mut b_cs = [transition_barrier(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)];
                cmd_list.ResourceBarrier(&b_cs);
                drop_barriers(&mut b_cs);
                let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(ss_readback.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: WIN_W,
                                Height: WIN_H,
                                Depth: 1,
                                RowPitch: ss_row_pitch,
                            },
                        },
                    },
                };
                let mut src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(bb.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                cmd_list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                let _ = ManuallyDrop::take(&mut dst.pResource);
                let _ = ManuallyDrop::take(&mut src.pResource);
                let mut b_pr = [transition_barrier(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT)];
                cmd_list.ResourceBarrier(&b_pr);
                drop_barriers(&mut b_pr);
            } else {
                let mut b_pr = [transition_barrier(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)];
                cmd_list.ResourceBarrier(&b_pr);
                drop_barriers(&mut b_pr);
            }

            cmd_list.Close().unwrap();
            let lists = [Some(cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&lists);
            let sync_interval = if wstate.vsync_enabled { 1u32 } else { 0u32 };
            if swap_chain.Present(sync_interval, DXGI_PRESENT(0)).is_err() {
                break;
            }

            if take_shot || save_rec_frame || !wstate.use_gpu {
                ctx.signal_and_wait();
                if wstate.use_gpu {
                    decomp_ms = gpu_dec.read_slot_kernel_ms(&ctx, cur_slot);
                    last_gpu_decode_ms = decomp_ms;
                }
            } else {
                frame_fences[cur_slot] = ctx.signal_direct_async();
                decomp_ms = if last_gpu_decode_ms > 0.0 {
                    last_gpu_decode_ms
                } else {
                    gpu_dec.read_slot_kernel_ms(&ctx, cur_slot)
                };
            }

            frame_count += 1;
            fps_accum += 1;

            if take_shot {
                let out_path = cfg.screenshot_path.as_ref().unwrap();
                let mut ptr: *mut c_void = std::ptr::null_mut();
                let rng = D3D12_RANGE { Begin: 0, End: (ss_row_pitch * WIN_H) as usize };
                ss_readback.Map(0, Some(&rng), Some(&mut ptr)).unwrap();
                let raw = std::slice::from_raw_parts(ptr as *const u8, (ss_row_pitch * WIN_H) as usize);
                let file = File::create(out_path).expect("create screenshot png");
                let mut enc = png::Encoder::new(BufWriter::new(file), WIN_W, WIN_H);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                let mut writer = enc.write_header().unwrap();
                writer.write_image_data(raw).unwrap();
                ss_readback.Unmap(0, None);
                println!("[screenshot] Saved {} ({}x{}, gpu_decode={:.3} ms)", out_path.display(), WIN_W, WIN_H, decomp_ms);
                break;
            }

            if save_rec_frame {
                let out_path = cfg.record_dir.as_ref().unwrap().join(format!("frame_{:04}.png", frame_count - 1));
                let mut ptr: *mut c_void = std::ptr::null_mut();
                let rng = D3D12_RANGE { Begin: 0, End: (ss_row_pitch * WIN_H) as usize };
                ss_readback.Map(0, Some(&rng), Some(&mut ptr)).unwrap();
                let raw = std::slice::from_raw_parts(ptr as *const u8, (ss_row_pitch * WIN_H) as usize);
                let file = File::create(&out_path).expect("create rec frame png");
                let mut enc = png::Encoder::new(BufWriter::new(file), WIN_W, WIN_H);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                let mut writer = enc.write_header().unwrap();
                writer.write_image_data(raw).unwrap();
                ss_readback.Unmap(0, None);
                if frame_count >= cfg.record_frames {
                    println!("[record] Saved {} frames to {}", frame_count, cfg.record_dir.as_ref().unwrap().display());
                    break;
                }
            }

            if let Some(target_n) = cfg.measure_frames {
                if frame_count == 1 {
                    measure_start = Some(Instant::now());
                } else if frame_count > 1 {
                    measure_decode_sum += decomp_ms;
                    let measured = frame_count - 1;
                    if measured >= target_n {
                        let secs = measure_start.unwrap().elapsed().as_secs_f64();
                        let avg_fps = measured as f64 / secs;
                        let avg_dec = measure_decode_sum / measured as f64;
                        let mode_tag = if wstate.use_gpu && wstate.rt_enabled && rt_scene.is_some() {
                            "NEW_GPU_RTX5080_PINGPONG_DXR_V5"
                        } else if wstate.use_gpu {
                            "NEW_GPU_RTX5080_PINGPONG_V4"
                        } else if wstate.multi_core_cpu {
                            "OLD_CPU_MULTICORE"
                        } else {
                            "OLD_CPU_1CORE"
                        };
                        if wstate.use_gpu && take_shot {
                            // Read fresh cull stats on screenshot frame (already waited via signal_and_wait)
                            if use_meshlets {
                                if let Some(ref mp) = meshlet_pipe {
                                    last_cull_stats = mp.read_cull_stats(cur_slot);
                                }
                            }
                        }
                        let rtgi_on = rt_scene.is_some() && wstate.rtgi_enabled;
                        if wstate.sfs_enabled || use_meshlets || rtgi_on || cfg.render_1440p || cfg.nvme_stream {
                            let ms_total_m = meshlet_pipe.as_ref().map(|m| m.total_tris as f64 / 1e6).unwrap_or(0.0);
                            let ms_vis_m = last_cull_stats.1 as f64 / 1e6;
                            let ms_cull_pct = if last_cull_stats.0 + last_cull_stats.2 > 0 {
                                (last_cull_stats.2 as f64 / (last_cull_stats.0 + last_cull_stats.2) as f64) * 100.0
                            } else {
                                0.0
                            };
                            let mut extra = String::new();
                            if cfg.render_1440p {
                                extra.push_str(&format!(" Res:{}x{}_HDR", render_w, render_h));
                            }
                            if cfg.nvme_stream {
                                extra.push_str(&format!(" NVMe:{:.2}ms", last_nvme_ms));
                            }
                            if wstate.sfs_enabled {
                                extra.push_str(&format!(
                                    " Grid:32x32 Res:{:.1}/{:.0}MiB Evict:{} Frag:{:.0}% Peak:{:.0}MB/s",
                                    last_world_res_mb, last_world_cap_mb, last_world_evict, last_world_frag, last_world_peak_mbs
                                ));
                            }
                            if use_meshlets {
                                extra.push_str(&format!(
                                    " MS[N]:ON Tris:{:.2}M/{:.2}M (Cull:{:.0}%)",
                                    ms_vis_m, ms_total_m, ms_cull_pct
                                ));
                            }
                            if rt_scene.is_some() {
                                extra.push_str(&format!(
                                    " TLAS:{} RTGI[G]:{}",
                                    tlas_inst_count,
                                    if wstate.rtgi_enabled { "ON" } else { "OFF" }
                                ));
                            }
                            println!(
                                "[fps] mode={} tiles={} ({:.1} MiB/frame) frames={} avg_fps={:.2} avg_decode_ms={:.3}{}",
                                mode_tag, cur_tiles, streamed_mb, measured, avg_fps, avg_dec, extra
                            );
                        } else {
                            println!(
                                "[fps] mode={} tiles={} ({:.1} MiB/frame) frames={} avg_fps={:.2} avg_decode_ms={:.3}",
                                mode_tag, cur_tiles, streamed_mb, measured, avg_fps, avg_dec
                            );
                        }
                        break;
                    }
                }
            }

            let elapsed = last_title_t.elapsed().as_secs_f64();
            if elapsed >= 0.35 {
                let fps = fps_accum as f64 / elapsed;
                let throughput = if decomp_ms > 0.0001 {
                    streamed_mb / (decomp_ms / 1000.0)
                } else {
                    0.0
                };
                let mode_str = if wstate.use_gpu {
                    "NEW v0.6: AsyncCompute+DXR"
                } else if wstate.multi_core_cpu {
                    "OLD: CPU Multi-Core"
                } else {
                    "OLD: CPU 1-Core"
                };
                let cam_str = match wstate.cam_mode {
                    0 => "Chase",
                    1 => "Free(WASD)",
                    _ => "Orbit",
                };
                let sfs_str = if wstate.sfs_enabled { "ON" } else { "OFF" };
                let rt_str = if rt_scene.is_some() && wstate.rt_enabled { "ON" } else { "OFF" };
                let rtgi_str = if rt_scene.is_some() && wstate.rtgi_enabled { "ON" } else { "OFF" };
                let ms_str = if use_meshlets { "ON" } else { "OFF" };
                let ms_total_m = meshlet_pipe.as_ref().map(|m| m.total_tris as f64 / 1e6).unwrap_or(0.0);
                let ms_vis_m = last_cull_stats.1 as f64 / 1e6;
                let ms_cull_pct = if last_cull_stats.0 + last_cull_stats.2 > 0 {
                    (last_cull_stats.2 as f64 / (last_cull_stats.0 + last_cull_stats.2) as f64) * 100.0
                } else {
                    0.0
                };
                let vsync_str = if wstate.vsync_enabled { "ON" } else { "OFF" };
                let title = if wstate.sfs_enabled {
                    format!(
                        "{} | {:.1} FPS | Grid:32x32 Res:{:.1}/{:.0}MiB Evict:{} Peak:{:.0}MB/s | MS[N]:{} ({:.2}M/{:.2}M Cull:{:.0}%) | TLAS:{} RT[R]:{} RTGI[G]:{} VSync[L]:{} Cam[C]:{}",
                        mode_str, fps, last_world_res_mb, last_world_cap_mb, last_world_evict, last_world_peak_mbs, ms_str, ms_vis_m, ms_total_m, ms_cull_pct, tlas_inst_count, rt_str, rtgi_str, vsync_str, cam_str
                    )
                } else if use_meshlets {
                    format!(
                        "{} | {:.1} FPS | {:.1} MB/f | MS[N]:ON Tris:{:.2}M/{:.2}M (Cull:{:.0}%) | TLAS:{} RT[R]:{} RTGI[G]:{} SFS[F]:{} VSync[L]:{} Cam[C]:{}",
                        mode_str, fps, streamed_mb, ms_vis_m, ms_total_m, ms_cull_pct, tlas_inst_count, rt_str, rtgi_str, sfs_str, vsync_str, cam_str
                    )
                } else {
                    format!(
                        "{} | {:.1} FPS | {:.1} MB/f ({}t) | Dec:{:.2}ms ({:.0} MB/s) | RT[R]:{} RTGI[G]:{} SFS[F]:{} VSync[L]:{} | Cam[C]:{}",
                        mode_str, fps, streamed_mb, cur_tiles, decomp_ms, throughput, rt_str, rtgi_str, sfs_str, vsync_str, cam_str
                    )
                };
                let wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
                let _ = SetWindowTextW(hwnd, PCWSTR(wide.as_ptr()));
                fps_accum = 0;
                last_title_t = Instant::now();
            }
        }
        // Drain both GPU queues before dropping resources
        ctx.signal_and_wait();
        ctx.signal_and_wait_compute();
        if cfg.enable_debug {
            crate::gpu_decoder::dump_d3d12_info_queue(&ctx.device);
        }
    }
}
