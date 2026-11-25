use crate::aaa_world::{
    AaaCrowdRenderer, AaaHiZPyramid, AaaMeshletCity, CityCullStats, AAA_SRV_HIZ_BASE, CROWD_NPCS,
};
use crate::assets::{decode_bc16_from_bcpack_cpu, TexFormat, TILE_BC16_BYTES, TILE_BLOCKS};
use crate::gltf_loader::{
    find_aaa_assets_dir, load_glb_as_bcpack, load_hdr_fp16, load_png_as_bcpack, BcpackMipChunk,
    BcpackTexture2D, LoadedGlbModel, PbrSubmesh, PbrVertex,
};
use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, dump_d3d12_info_queue, heap_props, pack_batch,
    transition_barrier, uav_barrier, uav_barrier_all, upload_slice, Bc16GpuStreamer, GpuContext,
};
use crate::raytracing::{supports_dxr_1_1, MAX_RT_INSTANCES};
use crate::rtgi::RTGI_DENOISE_CS_DXIL;
use bcpack::bc1::{QuadTable, Window};
use png;
use std::ffi::c_void;
use std::fs::File;
use std::io::BufWriter;
use std::mem::{size_of, ManuallyDrop};
use std::path::{Path, PathBuf};
use std::time::Instant;
use windows::core::{w, Interface, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{
    IDXGISwapChain3, DXGI_PRESENT, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1,
    DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, ReleaseCapture, SetCapture, VK_DOWN, VK_ESCAPE, VK_LEFT, VK_RIGHT, VK_SHIFT,
    VK_SPACE, VK_UP,
};
use windows::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRect, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
    GetClientRect, GetWindowLongPtrW, IsWindow, LoadCursorW, PeekMessageW, PostQuitMessage,
    RegisterClassExW, SetWindowLongPtrW, SetWindowTextW, ShowWindow, TranslateMessage, CS_HREDRAW,
    CS_VREDRAW, CW_USEDEFAULT, GWLP_USERDATA, IDC_ARROW, MSG, PM_REMOVE, SW_SHOW, WINDOW_EX_STYLE,
    WM_CLOSE, WM_DESTROY, WM_KEYDOWN, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE, WM_QUIT,
    WNDCLASSEXW, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

pub const AAA_SKY_VS_DXIL: &[u8] = include_bytes!("../shaders/aaa_sky_vs.dxil");
pub const AAA_SKY_PS_DXIL: &[u8] = include_bytes!("../shaders/aaa_sky_ps.dxil");
pub const AAA_PBR_VS_DXIL: &[u8] = include_bytes!("../shaders/aaa_pbr_vs.dxil");
pub const AAA_PBR_PS_DXIL: &[u8] = include_bytes!("../shaders/aaa_pbr_ps.dxil");
pub const AAA_BLOOM_CS_DXIL: &[u8] = include_bytes!("../shaders/aaa_bloom.dxil");
pub const AAA_TAA_CS_DXIL: &[u8] = include_bytes!("../shaders/aaa_taa.dxil");
pub const AAA_AUTOEXP_CS_DXIL: &[u8] = include_bytes!("../shaders/aaa_autoexp.dxil");
pub const AAA_COMP_VS_DXIL: &[u8] = include_bytes!("../shaders/aaa_comp_vs.dxil");
pub const AAA_COMP_PS_DXIL: &[u8] = include_bytes!("../shaders/aaa_comp_ps.dxil");

const WIN_W: u32 = 1280;
const WIN_H: u32 = 720;
const SWAP_W: u32 = 1920;
const SWAP_H: u32 = 1080;
const TEX_SLOT_CAP: usize = 32;
const SRV_TABLE_STRIDE: usize = 37; // t0..t31 textures, t32 envHdr, t33 tlas, t34 instMeta, t35 sceneVB, t36 sceneIB
const MAX_BATCH_TILES: usize = 128; // 128 x 64 KiB BC16 tiles = 8.0 MiB/batch

#[inline]
fn dxgi_format_for(fmt: TexFormat) -> DXGI_FORMAT {
    match fmt {
        TexFormat::Bc1 => DXGI_FORMAT_BC1_UNORM,
        TexFormat::Bc5 => DXGI_FORMAT_BC5_UNORM,
        TexFormat::Bc7 => DXGI_FORMAT_BC7_UNORM,
    }
}

#[derive(Clone, Debug)]
pub struct AaaDemoConfig {
    pub aaa_dir: Option<PathBuf>,
    pub render_1440p: bool,
    pub start_cpu: bool,
    pub flush_bench: bool,
    pub enable_debug: bool,
    pub xray_mode: bool,
    pub vsync: bool,
    pub no_rt: bool,
    pub no_rtgi: bool,
    pub tiles_per_frame: usize,
    pub measure_frames: Option<u32>,
    pub flush_at_frame: Option<u32>,
    pub screenshot_path: Option<PathBuf>,
    pub screenshot_frames: u32,
    pub vram_budget_mb: usize,
    pub benchmark_run: bool,
    pub benchmark_secs: f32,
    pub no_city: bool,
    pub no_crowd: bool,
    pub record_dir: Option<PathBuf>,
    pub record_frames: u32,
    pub fps_cap: u32,
    pub sim_speed: f32,
    pub show_hud: Option<bool>,
}

impl Default for AaaDemoConfig {
    fn default() -> Self {
        Self {
            aaa_dir: None,
            render_1440p: false,
            start_cpu: false,
            flush_bench: false,
            enable_debug: false,
            xray_mode: false,
            vsync: true,
            no_rt: false,
            no_rtgi: false,
            tiles_per_frame: 128,
            measure_frames: None,
            flush_at_frame: None,
            screenshot_path: None,
            screenshot_frames: 24,
            vram_budget_mb: 128,
            benchmark_run: false,
            benchmark_secs: 60.0,
            no_city: false,
            no_crowd: false,
            record_dir: None,
            record_frames: 120,
            fps_cap: 0,
            sim_speed: 1.0,
            show_hud: None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SkyConstants {
    inv_view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    camera_pos: [f32; 3],
    time: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AaaDrawCB {
    model: [f32; 16],
    prev_model: [f32; 16],
    base_color: [f32; 4],
    mat_params: [f32; 4],
    emissive_time: [f32; 4],
    tex_indices: [u32; 4],
    tile_states: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AaaFrameCB {
    vp_jittered: [f32; 16],
    vp_unjittered: [f32; 16],
    prev_vp_unjittered: [f32; 16],
    camera_pos: [f32; 4],
    sun_dir_time: [f32; 4],
    jitter_res: [f32; 4],
}

struct WindowState {
    use_gpu: bool,
    rt_enabled: bool,
    rtgi_enabled: bool,
    xray_enabled: bool,
    vsync_enabled: bool,
    sfs_enabled: bool,
    cam_mode: u32, // 0=Showroom 3/4 Orbit, 1=Chase Drive, 2=Free Cam
    fps_cap: u32,  // 0 = MAX/VSync, 15..=300 = capped FPS
    sim_speed: f32,
    saved_sim_speed: f32,
    hud_visible: bool,
    drag_slider: u8, // 0 = none, 1 = FPS cap, 2 = Sim speed
    flush_requested: bool,
    should_quit: bool,
}

const HUD_TRACK_X0: f32 = 0.032;
const HUD_TRACK_X1: f32 = 0.296;

#[inline]
fn fps_to_norm(fps_cap: u32) -> f32 {
    if fps_cap == 0 {
        1.0
    } else {
        (((fps_cap.clamp(15, 300) - 15) as f32) / 285.0) * 0.96
    }
}

#[inline]
fn norm_to_fps(n: f32) -> u32 {
    if n >= 0.96 {
        0
    } else {
        15 + ((n.clamp(0.0, 0.96) / 0.96) * 285.0).round() as u32
    }
}

#[inline]
fn sim_to_norm(speed: f32) -> f32 {
    let s = speed.clamp(0.0, 4.0);
    if s <= 1.0 {
        s * 0.40
    } else {
        0.40 + ((s - 1.0) / 3.0) * 0.60
    }
}

#[inline]
fn norm_to_sim(n: f32) -> f32 {
    let t = n.clamp(0.0, 1.0);
    if t < 0.015 {
        0.0
    } else if (t - 0.40).abs() < 0.015 {
        1.0
    } else if t <= 0.40 {
        (t / 0.40).clamp(0.0, 1.0)
    } else {
        (1.0 + ((t - 0.40) / 0.60) * 3.0).clamp(1.0, 4.0)
    }
}

#[derive(Clone, Copy)]
struct Mat4([f32; 16]);

impl Mat4 {
    #[allow(dead_code)]
    fn identity() -> Self {
        Self([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    fn mul(&self, rhs: &Self) -> Self {
        let mut r = [0.0f32; 16];
        for c in 0..4 {
            for row in 0..4 {
                r[c * 4 + row] = (0..4).map(|k| self.0[k * 4 + row] * rhs.0[c * 4 + k]).sum();
            }
        }
        Self(r)
    }

    fn translation(x: f32, y: f32, z: f32) -> Self {
        Self([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            x, y, z, 1.0,
        ])
    }

    fn scale(x: f32, y: f32, z: f32) -> Self {
        Self([
            x, 0.0, 0.0, 0.0,
            0.0, y, 0.0, 0.0,
            0.0, 0.0, z, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    fn rotate_y(rad: f32) -> Self {
        let c = rad.cos();
        let s = rad.sin();
        Self([
            c, 0.0, -s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            s, 0.0, c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Self {
        let f = norm3([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
        let r = norm3(cross3(up, f));
        let u = cross3(f, r);
        Self([
            r[0], u[0], f[0], 0.0,
            r[1], u[1], f[1], 0.0,
            r[2], u[2], f[2], 0.0,
            -dot3(r, eye), -dot3(u, eye), -dot3(f, eye), 1.0,
        ])
    }

    fn perspective_rh(fov_y_rad: f32, aspect: f32, zn: f32, zf: f32) -> Self {
        let ys = 1.0 / (fov_y_rad * 0.5).tan();
        let xs = ys / aspect;
        let q = zf / (zn - zf);
        Self([
            xs, 0.0, 0.0, 0.0,
            0.0, ys, 0.0, 0.0,
            0.0, 0.0, q, -1.0,
            0.0, 0.0, zn * q, 0.0,
        ])
    }

    fn inverse(&self) -> Self {
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
        let idet = if det.abs() > 1e-8 { 1.0 / det } else { 1.0 };
        for v in &mut inv {
            *v *= idet;
        }
        Self(inv)
    }

    fn to_3x4(&self) -> [f32; 12] {
        let m = &self.0;
        [
            m[0], m[4], m[8], m[12],
            m[1], m[5], m[9], m[13],
            m[2], m[6], m[10], m[14],
        ]
    }
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f32; 3]) -> [f32; 3] {
    let l = dot3(a, a).sqrt();
    if l > 1e-6 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [0.0, 1.0, 0.0]
    }
}

unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowState;
    if msg == WM_CLOSE {
        if !ptr.is_null() {
            (*ptr).should_quit = true;
        }
        let _ = DestroyWindow(hwnd);
        PostQuitMessage(0);
        return LRESULT(0);
    }
    if msg == WM_DESTROY {
        if !ptr.is_null() {
            (*ptr).should_quit = true;
        }
        PostQuitMessage(0);
        return LRESULT(0);
    }
    if !ptr.is_null() && (msg == WM_LBUTTONDOWN || msg == WM_MOUSEMOVE || msg == WM_LBUTTONUP) {
        let st = &mut *ptr;
        let mx = (lparam.0 & 0xFFFF) as i16 as f32;
        let my = ((lparam.0 >> 16) & 0xFFFF) as i16 as f32;
        let mut rc = RECT::default();
        if GetClientRect(hwnd, &mut rc).is_ok() {
            let cw = (rc.right - rc.left).max(1) as f32;
            let ch = (rc.bottom - rc.top).max(1) as f32;
            let u = mx / cw;
            let v = my / ch;
            let norm = ((u - HUD_TRACK_X0) / (HUD_TRACK_X1 - HUD_TRACK_X0)).clamp(0.0, 1.0);
            if msg == WM_LBUTTONDOWN && st.hud_visible {
                if (0.022..=0.306).contains(&u) && (0.845..=0.898).contains(&v) {
                    st.drag_slider = 1;
                    SetCapture(hwnd);
                    st.fps_cap = norm_to_fps(norm);
                    if st.fps_cap > 0 {
                        st.vsync_enabled = false;
                    }
                    return LRESULT(0);
                } else if (0.022..=0.306).contains(&u) && (0.912..=0.965).contains(&v) {
                    st.drag_slider = 2;
                    SetCapture(hwnd);
                    st.sim_speed = norm_to_sim(norm);
                    return LRESULT(0);
                }
            } else if msg == WM_MOUSEMOVE && st.drag_slider != 0 {
                if st.drag_slider == 1 {
                    st.fps_cap = norm_to_fps(norm);
                    if st.fps_cap > 0 {
                        st.vsync_enabled = false;
                    }
                } else if st.drag_slider == 2 {
                    st.sim_speed = norm_to_sim(norm);
                }
                return LRESULT(0);
            } else if msg == WM_LBUTTONUP && st.drag_slider != 0 {
                st.drag_slider = 0;
                let _ = ReleaseCapture();
                return LRESULT(0);
            }
        }
    }
    if !ptr.is_null() && msg == WM_KEYDOWN {
        let st = &mut *ptr;
        const FPS_STEPS: [u32; 10] = [15, 30, 45, 60, 90, 120, 144, 180, 240, 0];
        match wparam.0 as u32 {
            x if x == VK_SPACE.0 as u32 => st.use_gpu = !st.use_gpu,
            x if x == b'T' as u32 => st.flush_requested = true,
            x if x == b'V' as u32 => st.xray_enabled = !st.xray_enabled,
            x if x == b'L' as u32 => {
                st.vsync_enabled = !st.vsync_enabled;
                if st.vsync_enabled {
                    st.fps_cap = 0;
                }
            }
            x if x == b'F' as u32 => st.sfs_enabled = !st.sfs_enabled,
            x if x == b'R' as u32 => st.rt_enabled = !st.rt_enabled,
            x if x == b'G' as u32 => st.rtgi_enabled = !st.rtgi_enabled,
            x if x == b'C' as u32 => st.cam_mode = (st.cam_mode + 1) % 3,
            x if x == b'H' as u32 => st.hud_visible = !st.hud_visible,
            x if x == b'P' as u32 => {
                if st.sim_speed > 0.005 {
                    st.saved_sim_speed = st.sim_speed;
                    st.sim_speed = 0.0;
                } else {
                    st.sim_speed = st.saved_sim_speed.max(0.25);
                }
            }
            0xDB => {
                // '[' decrease FPS cap
                let next = if st.fps_cap == 0 {
                    FPS_STEPS[FPS_STEPS.len() - 2]
                } else {
                    FPS_STEPS.iter().rev().copied().find(|&f| f != 0 && f < st.fps_cap).unwrap_or(15)
                };
                st.fps_cap = next;
                st.vsync_enabled = false;
            }
            0xDD => {
                // ']' increase FPS cap
                let next = if st.fps_cap == 0 {
                    0
                } else {
                    FPS_STEPS.iter().copied().find(|&f| f == 0 || f > st.fps_cap).unwrap_or(0)
                };
                st.fps_cap = next;
                if st.fps_cap > 0 {
                    st.vsync_enabled = false;
                }
            }
            0xBD => st.sim_speed = (st.sim_speed - 0.25).clamp(0.0, 4.0), // '-' slower sim
            0xBB => st.sim_speed = (st.sim_speed + 0.25).clamp(0.0, 4.0), // '=' faster sim
            x if x == b'0' as u32 => st.sim_speed = 1.0,
            x if x == VK_ESCAPE.0 as u32 => {
                st.should_quit = true;
                let _ = DestroyWindow(hwnd);
                PostQuitMessage(0);
            }
            _ => {}
        }
    }
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

fn build_unit_box_pbr() -> (Vec<PbrVertex>, Vec<u32>) {
    let faces: [([f32; 3], [f32; 4], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 1.0], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0], [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]),
    ];
    let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let mut verts = Vec::with_capacity(24);
    let mut inds = Vec::with_capacity(36);
    for (fi, (n, t, corners)) in faces.iter().enumerate() {
        let base = (fi * 4) as u32;
        for (ci, pos) in corners.iter().enumerate() {
            verts.push(PbrVertex {
                pos: *pos,
                normal: *n,
                tangent: *t,
                uv: uvs[ci],
            });
        }
        inds.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, inds)
}

fn build_plane_pbr() -> (Vec<PbrVertex>, Vec<u32>) {
    let n = [0.0, 1.0, 0.0];
    let t = [1.0, 0.0, 0.0, 1.0];
    let verts = vec![
        PbrVertex { pos: [-0.5, 0.0, -0.5], normal: n, tangent: t, uv: [0.0, 0.0] },
        PbrVertex { pos: [0.5, 0.0, -0.5], normal: n, tangent: t, uv: [1.0, 0.0] },
        PbrVertex { pos: [0.5, 0.0, 0.5], normal: n, tangent: t, uv: [1.0, 1.0] },
        PbrVertex { pos: [-0.5, 0.0, 0.5], normal: n, tangent: t, uv: [0.0, 1.0] },
    ];
    let inds = vec![0, 2, 1, 0, 3, 2];
    (verts, inds)
}

struct AaaAssetPack {
    car: LoadedGlbModel,
    textures: Vec<BcpackTexture2D>, // 0..13 car (BC7/BC5) + 14..22 PolyHaven 4K (BC7/BC5)
    asphalt_tex: [u32; 3],
    promenade_tex: [u32; 3],
    concrete_tex: [u32; 3],
    hdri_w: u32,
    hdri_h: u32,
    hdri_fp16: Vec<u16>,
    total_tiles: usize,
    total_bcpack_bytes: usize,
    total_mip0_bytes: usize,
    total_vram_bytes: usize,
}

fn load_aaa_asset_pack(aaa_dir: &Path, window: &Window, quads: &QuadTable) -> AaaAssetPack {
    let cache_dir = aaa_dir.join("bcpack_cache");
    std::fs::create_dir_all(&cache_dir).ok();

    let mut car = load_glb_as_bcpack(&aaa_dir.join("CarConcept.glb"), &cache_dir, window, quads);
    let mut textures = std::mem::take(&mut car.textures);

    let mut push_4k = |file_4k: &str, file_2k: &str, name: &str, fmt: TexFormat| -> u32 {
        let idx = textures.len() as u32;
        let path_4k = aaa_dir.join(file_4k);
        let src_path = if path_4k.exists() { path_4k } else { aaa_dir.join(file_2k) };
        textures.push(load_png_as_bcpack(&src_path, name, 4096, fmt, &cache_dir, window, quads));
        idx
    };

    let asphalt_tex = [
        push_4k("asphalt_albedo_4k.png", "asphalt_albedo.png", "asphalt_albedo_4k", TexFormat::Bc7),
        push_4k("asphalt_normal_4k.png", "asphalt_normal.png", "asphalt_normal_4k", TexFormat::Bc5),
        push_4k("asphalt_arm_4k.png", "asphalt_arm.png", "asphalt_arm_4k", TexFormat::Bc7),
    ];
    let promenade_tex = [
        push_4k("promenade_albedo_4k.png", "promenade_albedo.png", "promenade_albedo_4k", TexFormat::Bc7),
        push_4k("promenade_normal_4k.png", "promenade_normal.png", "promenade_normal_4k", TexFormat::Bc5),
        push_4k("promenade_arm_4k.png", "promenade_arm.png", "promenade_arm_4k", TexFormat::Bc7),
    ];
    let concrete_tex = [
        push_4k("concrete_albedo_4k.png", "concrete_albedo.png", "concrete_albedo_4k", TexFormat::Bc7),
        push_4k("concrete_normal_4k.png", "concrete_normal.png", "concrete_normal_4k", TexFormat::Bc5),
        push_4k("concrete_arm_4k.png", "concrete_arm.png", "concrete_arm_4k", TexFormat::Bc7),
    ];

    assert!(
        textures.len() <= TEX_SLOT_CAP,
        "Expected <= {TEX_SLOT_CAP} PBR textures, got {}",
        textures.len()
    );

    let (hdri_w, hdri_h, hdri_fp16) =
        load_hdr_fp16(&aaa_dir.join("sunset_hdri_2k.fp16")).expect("sunset_hdri_2k.fp16 missing");

    let total_tiles: usize = textures.iter().map(|t| t.num_strips).sum();
    let total_bcpack_bytes: usize = textures.iter().flat_map(|t| t.bcpack_tiles.iter()).map(|b| b.len()).sum();
    let total_mip0_bytes: usize = textures.iter().map(|t| t.expected_blocks.len()).sum();
    let total_lower_bytes: usize = textures
        .iter()
        .map(|t| t.lower_mips.iter().map(|m| m.expected_blocks.len()).sum::<usize>())
        .sum();
    let total_vram_bytes = total_mip0_bytes + total_lower_bytes;

    AaaAssetPack {
        car,
        textures,
        asphalt_tex,
        promenade_tex,
        concrete_tex,
        hdri_w,
        hdri_h,
        hdri_fp16,
        total_tiles,
        total_bcpack_bytes,
        total_mip0_bytes,
        total_vram_bytes,
    }
}

struct TileRef {
    tex_idx: usize,
    strip_idx: usize,
}

struct SfsTexState {
    num_strips: usize,
    mip0_resident_strips: usize,
    mip0_bytes: usize,
    queued: bool,
    clamped_lod: f32, // 0.0 = Mip-0 resident, 1.0 = SFS clamped to Mip 1+
    srv_dirty: bool,
    desired_mip: f32,
    pixel_hits: u32,
    last_hot_frame: u64,
    fresh_timer: f32,
    clay_timer: f32,
}

struct SfsResidencyManager {
    tex: Vec<SfsTexState>,
    pending: std::collections::VecDeque<TileRef>,
    budget_bytes: usize,
    total_evictions: u64,
    total_vram_bytes: usize,
    lower_vram_bytes: usize,
}

impl SfsResidencyManager {
    fn new(textures: &[BcpackTexture2D], budget_bytes: usize) -> Self {
        let mut tex = Vec::with_capacity(textures.len());
        let mut lower_vram_bytes = 0usize;
        let mut total_vram_bytes = 0usize;
        for t in textures {
            let mip0_bytes = t.expected_blocks.len();
            let lower_mip_bytes: usize = t.lower_mips.iter().map(|m| m.expected_blocks.len()).sum();
            lower_vram_bytes += lower_mip_bytes;
            total_vram_bytes += mip0_bytes + lower_mip_bytes;
            tex.push(SfsTexState {
                num_strips: t.num_strips,
                mip0_resident_strips: 0,
                mip0_bytes,
                queued: false,
                clamped_lod: 1.0,
                srv_dirty: false,
                desired_mip: 4.0,
                pixel_hits: 0,
                last_hot_frame: 0,
                fresh_timer: 0.0,
                clay_timer: 0.0,
            });
        }
        Self {
            tex,
            pending: std::collections::VecDeque::new(),
            budget_bytes,
            total_evictions: 0,
            total_vram_bytes,
            lower_vram_bytes,
        }
    }

    fn flush_to_clay(&mut self) {
        self.pending.clear();
        for s in &mut self.tex {
            s.mip0_resident_strips = 0;
            s.queued = false;
            if s.clamped_lod < 0.5 {
                s.clamped_lod = 1.0;
                s.srv_dirty = true;
            }
            s.fresh_timer = 0.0;
            s.clay_timer = 0.75;
        }
    }

    fn tick_timers(&mut self, dt: f32) {
        for s in &mut self.tex {
            if s.fresh_timer > 0.0 {
                s.fresh_timer = (s.fresh_timer - dt * 1.4).max(0.0);
            }
            if s.clay_timer > 0.0 {
                s.clay_timer = (s.clay_timer - dt).max(0.0);
            }
        }
    }

    fn current_vram_bytes(&self) -> usize {
        let mip0: usize = self
            .tex
            .iter()
            .filter(|s| s.mip0_resident_strips > 0)
            .map(|s| (s.mip0_bytes * s.mip0_resident_strips) / s.num_strips.max(1))
            .sum();
        self.lower_vram_bytes + mip0
    }

    fn planned_vram_bytes(&self) -> usize {
        let mip0: usize = self
            .tex
            .iter()
            .filter(|s| s.queued || s.mip0_resident_strips > 0)
            .map(|s| s.mip0_bytes)
            .sum();
        self.lower_vram_bytes + mip0
    }

    fn evict_tex(&mut self, idx: usize) {
        let s = &mut self.tex[idx];
        if s.mip0_resident_strips == 0 && !s.queued {
            return;
        }
        s.mip0_resident_strips = 0;
        s.queued = false;
        if s.clamped_lod < 0.5 {
            s.clamped_lod = 1.0;
            s.srv_dirty = true;
        }
        s.fresh_timer = 0.0;
        self.pending.retain(|r| r.tex_idx != idx);
        self.total_evictions += 1;
    }

    fn enqueue_tex(&mut self, idx: usize) {
        let s = &mut self.tex[idx];
        if s.queued || s.mip0_resident_strips >= s.num_strips {
            return;
        }
        s.queued = true;
        s.clay_timer = 0.0;
        let start = s.mip0_resident_strips;
        for strip_idx in start..s.num_strips {
            self.pending.push_back(TileRef { tex_idx: idx, strip_idx });
        }
    }

    fn apply_sfs_feedback_and_schedule(&mut self, fb_bytes: &[u8], frame_idx: u64, sfs_enabled: bool) {
        for (i, s) in self.tex.iter_mut().enumerate() {
            let off = i * 16;
            if off + 8 <= fb_bytes.len() {
                let min_mip_fp8 = u32::from_le_bytes(fb_bytes[off..off + 4].try_into().unwrap());
                let hits = u32::from_le_bytes(fb_bytes[off + 4..off + 8].try_into().unwrap());
                s.pixel_hits = hits;
                if hits > 0 && min_mip_fp8 != u32::MAX {
                    s.desired_mip = (min_mip_fp8 as f32) / 256.0;
                    if s.desired_mip < 1.85 {
                        s.last_hot_frame = frame_idx;
                    }
                } else {
                    s.desired_mip = 11.0;
                }
            }
        }

        if !sfs_enabled || self.budget_bytes == 0 {
            for i in 0..self.tex.len() {
                self.enqueue_tex(i);
            }
            return;
        }

        // First evict any unhit / far Mip-0 textures if we are over budget
        while self.planned_vram_bytes() > self.budget_bytes {
            let cold = (0..self.tex.len())
                .filter(|&i| self.tex[i].queued || self.tex[i].mip0_resident_strips > 0)
                .filter(|&i| self.tex[i].pixel_hits == 0 || self.tex[i].desired_mip >= 1.85)
                .min_by(|&a, &b| {
                    self.tex[a]
                        .last_hot_frame
                        .cmp(&self.tex[b].last_hot_frame)
                        .then_with(|| self.tex[b].desired_mip.partial_cmp(&self.tex[a].desired_mip).unwrap_or(std::cmp::Ordering::Equal))
                });
            if let Some(ci) = cold {
                self.evict_tex(ci);
            } else {
                break;
            }
        }

        // Sort hot Mip-0 candidates by sharp detail demand (lowest desired_mip, most pixels)
        let mut hot: Vec<usize> = (0..self.tex.len())
            .filter(|&i| self.tex[i].pixel_hits > 0 && self.tex[i].desired_mip < 1.85)
            .collect();
        hot.sort_by(|&a, &b| {
            self.tex[a]
                .desired_mip
                .partial_cmp(&self.tex[b].desired_mip)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.tex[b].pixel_hits.cmp(&self.tex[a].pixel_hits))
        });

        for hi in hot {
            if self.tex[hi].queued || self.tex[hi].mip0_resident_strips >= self.tex[hi].num_strips {
                continue;
            }
            let need = self.tex[hi].mip0_bytes;
            while self.planned_vram_bytes() + need > self.budget_bytes {
                let victim = (0..self.tex.len())
                    .filter(|&i| i != hi && (self.tex[i].queued || self.tex[i].mip0_resident_strips > 0))
                    .filter(|&i| {
                        self.tex[i].pixel_hits == 0
                            || self.tex[i].last_hot_frame < self.tex[hi].last_hot_frame
                            || self.tex[i].desired_mip > self.tex[hi].desired_mip + 0.35
                    })
                    .min_by(|&a, &b| {
                        self.tex[a]
                            .last_hot_frame
                            .cmp(&self.tex[b].last_hot_frame)
                            .then_with(|| self.tex[b].desired_mip.partial_cmp(&self.tex[a].desired_mip).unwrap_or(std::cmp::Ordering::Equal))
                            .then_with(|| self.tex[a].pixel_hits.cmp(&self.tex[b].pixel_hits))
                    });
                if let Some(vi) = victim {
                    self.evict_tex(vi);
                } else {
                    break;
                }
            }
            if self.planned_vram_bytes() + need <= self.budget_bytes {
                self.enqueue_tex(hi);
            }
        }
    }

    fn take_batch(&mut self, count: usize) -> Vec<TileRef> {
        let take = count.min(self.pending.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(r) = self.pending.pop_front() {
                out.push(r);
            }
        }
        out
    }

    fn mark_resident(&mut self, tiles: &[TileRef]) {
        for t in tiles {
            let s = &mut self.tex[t.tex_idx];
            s.mip0_resident_strips = s.mip0_resident_strips.max(t.strip_idx + 1);
            s.clay_timer = 0.0;
            s.fresh_timer = 1.0;
            if s.mip0_resident_strips >= s.num_strips && s.clamped_lod > 0.5 {
                s.clamped_lod = 0.0;
                s.srv_dirty = true;
            }
        }
    }

    fn total_mip0_resident(&self) -> usize {
        self.tex.iter().map(|s| s.mip0_resident_strips).sum()
    }

    fn total_mip0_tiles(&self) -> usize {
        self.tex.iter().map(|s| s.num_strips).sum()
    }

    fn num_mip0_full(&self) -> usize {
        self.tex.iter().filter(|s| s.mip0_resident_strips >= s.num_strips && s.clamped_lod < 0.5).count()
    }

    fn state_for_tex(&self, idx: u32) -> f32 {
        if idx == u32::MAX {
            return -1.0;
        }
        let s = &self.tex[idx as usize];
        if s.mip0_resident_strips == 0 {
            if s.clay_timer > 0.0 {
                return -1.0;
            }
            return 2.0 + s.clamped_lod;
        }
        if s.mip0_resident_strips < s.num_strips {
            return (s.mip0_resident_strips as f32 / s.num_strips.max(1) as f32).clamp(0.01, 0.99);
        }
        if s.clamped_lod >= 0.5 {
            return 2.0 + s.clamped_lod;
        }
        1.0 + s.fresh_timer
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RtInstMeta {
    base_color: [f32; 4],
    geom: [u32; 4],
    uv_params: [f32; 4],
}

struct AaaDxrScene {
    blas_car: ID3D12Resource,
    blas_box: ID3D12Resource,
    blas_plane: ID3D12Resource,
    _blas_scratch: ID3D12Resource,
    tlas_scratch: [ID3D12Resource; 2],
    tlas_result: [ID3D12Resource; 2],
    instances_upload: [ID3D12Resource; 2],
    inst_meta_upload: [ID3D12Resource; 2],
    tlas_alloc: ID3D12CommandAllocator,
    tlas_cmd: ID3D12GraphicsCommandList,
}

impl AaaDxrScene {
    unsafe fn new(
        ctx: &mut GpuContext,
        vb_buf: &ID3D12Resource,
        ib_buf: &ID3D12Resource,
        car_vert_count: u32,
        car_idx_count: u32,
        box_base_vert: u32,
        box_idx_offset: u32,
        plane_base_vert: u32,
        plane_idx_offset: u32,
    ) -> Option<Self> {
        if !supports_dxr_1_1(&ctx.device) {
            return None;
        }
        let dev5: ID3D12Device5 = ctx.device.cast().ok()?;
        let vb_va = vb_buf.GetGPUVirtualAddress();
        let ib_va = ib_buf.GetGPUVirtualAddress();
        let stride = size_of::<PbrVertex>() as u64;

        let geoms = [
            D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_R32_UINT,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: car_idx_count,
                        VertexCount: car_vert_count,
                        IndexBuffer: ib_va,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vb_va,
                            StrideInBytes: stride,
                        },
                    },
                },
            },
            D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_R32_UINT,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 36,
                        VertexCount: 24,
                        IndexBuffer: ib_va + (box_idx_offset as u64) * 4,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vb_va + (box_base_vert as u64) * stride,
                            StrideInBytes: stride,
                        },
                    },
                },
            },
            D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_R32_UINT,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 6,
                        VertexCount: 4,
                        IndexBuffer: ib_va + (plane_idx_offset as u64) * 4,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vb_va + (plane_base_vert as u64) * stride,
                            StrideInBytes: stride,
                        },
                    },
                },
            },
        ];

        let mut blas_res = Vec::with_capacity(3);
        let mut max_scratch = 256u64;
        for g in &geoms {
            let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: g,
                },
            };
            let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
            dev5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre);
            max_scratch = max_scratch.max(pre.ScratchDataSizeInBytes);
            blas_res.push(create_committed_buffer(
                &ctx.device,
                D3D12_HEAP_TYPE_DEFAULT,
                pre.ResultDataMaxSizeInBytes.max(256),
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            ));
        }

        let blas_scratch = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            max_scratch,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );

        let alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        let cmd4: ID3D12GraphicsCommandList4 = list.cast().unwrap();
        for (i, g) in geoms.iter().enumerate() {
            let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: g,
                },
            };
            let build_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: blas_res[i].GetGPUVirtualAddress(),
                Inputs: inputs,
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: blas_scratch.GetGPUVirtualAddress(),
            };
            cmd4.BuildRaytracingAccelerationStructure(&build_desc, None);
            cmd_barriers(&list, vec![uav_barrier(&blas_res[i]), uav_barrier(&blas_scratch)]);
        }
        list.Close().unwrap();
        let exec = [Some(list.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&exec);
        ctx.signal_and_wait();

        let inst_bytes = (MAX_RT_INSTANCES * size_of::<D3D12_RAYTRACING_INSTANCE_DESC>()) as u64;
        let meta_bytes = (MAX_RT_INSTANCES * size_of::<RtInstMeta>()) as u64;
        let instances_upload = [
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, inst_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, inst_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
        ];
        let inst_meta_upload = [
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, meta_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, meta_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
        ];
        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: MAX_RT_INSTANCES as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: instances_upload[0].GetGPUVirtualAddress(),
            },
        };
        let mut tlas_pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_pre);

        let tlas_scratch = [
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, tlas_pre.ScratchDataSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, tlas_pre.ScratchDataSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
        ];
        let tlas_result = [
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, tlas_pre.ResultDataMaxSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE),
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, tlas_pre.ResultDataMaxSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE),
        ];

        let blas_plane = blas_res.pop().unwrap();
        let blas_box = blas_res.pop().unwrap();
        let blas_car = blas_res.pop().unwrap();

        let tlas_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let tlas_cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &tlas_alloc, None).unwrap();
        tlas_cmd.Close().unwrap();

        Some(Self {
            blas_car,
            blas_box,
            blas_plane,
            _blas_scratch: blas_scratch,
            tlas_scratch,
            tlas_result,
            instances_upload,
            inst_meta_upload,
            tlas_alloc,
            tlas_cmd,
        })
    }

    unsafe fn build_tlas(
        &self,
        ctx: &mut GpuContext,
        slot: usize,
        instances: &[(usize, [f32; 12], RtInstMeta)],
    ) {
        let s = slot & 1;
        let count = instances.len().min(MAX_RT_INSTANCES);
        let mut descs = Vec::with_capacity(count);
        let mut metas = Vec::with_capacity(count);
        let blas_vas = [
            self.blas_car.GetGPUVirtualAddress(),
            self.blas_box.GetGPUVirtualAddress(),
            self.blas_plane.GetGPUVirtualAddress(),
        ];
        for (idx, (kind, xf, meta)) in instances.iter().take(count).enumerate() {
            descs.push(D3D12_RAYTRACING_INSTANCE_DESC {
                Transform: *xf,
                _bitfield1: (idx as u32 & 0x00FF_FFFF) | (0xFFu32 << 24),
                _bitfield2: 0,
                AccelerationStructure: blas_vas[*kind],
            });
            metas.push(*meta);
        }
        upload_slice(&self.instances_upload[s], &descs);
        upload_slice(&self.inst_meta_upload[s], &metas);

        self.tlas_alloc.Reset().unwrap();
        self.tlas_cmd.Reset(&self.tlas_alloc, None).unwrap();
        let cmd4: ID3D12GraphicsCommandList4 = self.tlas_cmd.cast().unwrap();
        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: count as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: self.instances_upload[s].GetGPUVirtualAddress(),
            },
        };
        let build_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: self.tlas_result[s].GetGPUVirtualAddress(),
            Inputs: inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: self.tlas_scratch[s].GetGPUVirtualAddress(),
        };
        cmd4.BuildRaytracingAccelerationStructure(&build_desc, None);
        cmd_barriers(&self.tlas_cmd, vec![uav_barrier_all()]);
        self.tlas_cmd.Close().unwrap();
        let exec = [Some(self.tlas_cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&exec);
        ctx.signal_and_wait();
    }
}

unsafe fn create_tex2d_fmt(
    device: &ID3D12Device,
    w: u32,
    h: u32,
    fmt: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
    init_state: D3D12_RESOURCE_STATES,
    clear_rgba: Option<[f32; 4]>,
) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: w as u64,
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: fmt,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
        ..Default::default()
    };
    let cv = clear_rgba.map(|col| D3D12_CLEAR_VALUE {
        Format: fmt,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: col },
    });
    let mut res: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(
            &heap_props(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &desc,
            init_state,
            cv.as_ref().map(|c| c as *const D3D12_CLEAR_VALUE),
            &mut res,
        )
        .expect("CreateCommittedResource tex2d_fmt");
    res.unwrap()
}

unsafe fn create_fp16_tex(
    device: &ID3D12Device,
    w: u32,
    h: u32,
    flags: D3D12_RESOURCE_FLAGS,
    init_state: D3D12_RESOURCE_STATES,
    clear_rgba: Option<[f32; 4]>,
) -> ID3D12Resource {
    create_tex2d_fmt(
        device,
        w,
        h,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        flags,
        init_state,
        clear_rgba,
    )
}

unsafe fn create_root_sig(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    flags: D3D12_ROOT_SIGNATURE_FLAGS,
) -> ID3D12RootSignature {
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: if samplers.is_empty() { std::ptr::null() } else { samplers.as_ptr() },
        Flags: flags,
    };
    let mut blob = None;
    let mut err = None;
    D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut err))
        .expect("D3D12SerializeRootSignature");
    let blob = blob.unwrap();
    let bytes = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
    device.CreateRootSignature(0, bytes).expect("CreateRootSignature")
}

unsafe fn create_compute_pso_local(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    dxil: &[u8],
) -> ID3D12PipelineState {
    let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: dxil.as_ptr() as *const c_void,
            BytecodeLength: dxil.len(),
        },
        NodeMask: 0,
        CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
        Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
    };
    let pso: ID3D12PipelineState = device.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
    let _ = ManuallyDrop::take(&mut desc.pRootSignature);
    pso
}

fn static_sampler(
    reg: u32,
    addr_u: D3D12_TEXTURE_ADDRESS_MODE,
    addr_v: D3D12_TEXTURE_ADDRESS_MODE,
    aniso: u32,
) -> D3D12_STATIC_SAMPLER_DESC {
    D3D12_STATIC_SAMPLER_DESC {
        Filter: if aniso > 1 { D3D12_FILTER_ANISOTROPIC } else { D3D12_FILTER_MIN_MAG_MIP_LINEAR },
        AddressU: addr_u,
        AddressV: addr_v,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        MipLODBias: 0.0,
        MaxAnisotropy: aniso,
        ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
        MaxLOD: D3D12_FLOAT32_MAX,
        ShaderRegister: reg,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

struct AaaPostPipeline {
    width: u32,
    height: u32,
    half_w: u32,
    half_h: u32,
    direct_hdr: ID3D12Resource,
    raw_rtgi: ID3D12Resource,
    normal_depth: ID3D12Resource,
    motion_vec: ID3D12Resource,
    denoise_a: ID3D12Resource,
    denoise_b: ID3D12Resource,
    taa_hist: [ID3D12Resource; 2],
    exposure_buf: ID3D12Resource,
    bloom_a: ID3D12Resource,
    bloom_b: ID3D12Resource,
    gbuf_rtv_heap: ID3D12DescriptorHeap,
    rtv_inc: u32,
    denoise_root_sig: ID3D12RootSignature,
    denoise_pso: ID3D12PipelineState,
    taa_root_sig: ID3D12RootSignature,
    taa_pso: ID3D12PipelineState,
    autoexp_root_sig: ID3D12RootSignature,
    autoexp_pso: ID3D12PipelineState,
    bloom_root_sig: ID3D12RootSignature,
    bloom_pso: ID3D12PipelineState,
    comp_root_sig: ID3D12RootSignature,
    comp_pso: ID3D12PipelineState,
    // Descriptor indices inside main_srv_heap (starting at slot 74):
    // 74..75 = raw_rtgi, normal_depth (pass 1)
    // 76..77 = denoise_a, normal_depth (pass 2 & 4)
    // 78..79 = denoise_b, normal_depth (pass 3)
    // 80 = denoise_a UAV, 81 = denoise_b UAV
    // 82..86 = TAA0 SRVs (direct, gi, mv, hist1, nd), 87 = hist0 UAV
    // 88..92 = TAA1 SRVs (direct, gi, mv, hist0, nd), 93 = hist1 UAV
    // 94 = exposure_buf UAV, 95 = hist0 SRV, 96 = hist1 SRV
    // 97 = bloom_a SRV, 98 = bloom_b SRV, 99 = bloom_a UAV, 100 = bloom_b UAV
    // 101..103 = Composite0 (hist0, bloom_a, exposure_buf)
    // 104..106 = Composite1 (hist1, bloom_a, exposure_buf)
    srv_inc: u32,
}

impl AaaPostPipeline {
    unsafe fn new(
        device: &ID3D12Device,
        width: u32,
        height: u32,
        main_srv_heap: &ID3D12DescriptorHeap,
        srv_inc: u32,
    ) -> Self {
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);
        let srv_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let direct_hdr = create_fp16_tex(
            device, width, height,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some([0.14, 0.08, 0.14, 1.0]),
        );
        let raw_rtgi = create_fp16_tex(
            device, width, height,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some([0.0, 0.0, 0.0, 0.0]),
        );
        let normal_depth = create_fp16_tex(
            device, width, height,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some([0.0, 0.0, 0.0, 9999.0]),
        );
        let motion_vec = create_tex2d_fmt(
            device, width, height,
            DXGI_FORMAT_R16G16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some([0.0, 0.0, 0.0, 0.0]),
        );
        let denoise_a = create_fp16_tex(device, width, height, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, None);
        let denoise_b = create_fp16_tex(device, width, height, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, None);
        let taa_hist = [
            create_fp16_tex(device, width, height, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, srv_state, None),
            create_fp16_tex(device, width, height, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, srv_state, None),
        ];
        let exposure_buf = create_committed_buffer(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            256,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let bloom_a = create_fp16_tex(device, half_w, half_h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, None);
        let bloom_b = create_fp16_tex(device, half_w, half_h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, None);

        let gbuf_rtv_heap: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
            .unwrap();
        let rtv_inc = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let rtv_base = gbuf_rtv_heap.GetCPUDescriptorHandleForHeapStart();
        for (i, tex) in [&direct_hdr, &raw_rtgi, &normal_depth, &motion_vec].iter().enumerate() {
            device.CreateRenderTargetView(
                *tex,
                None,
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv_base.ptr + i * rtv_inc as usize },
            );
        }

        let srv_base = main_srv_heap.GetCPUDescriptorHandleForHeapStart();
        let cpu_at = |slot: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_base.ptr + slot * srv_inc as usize,
        };
        let mut srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        srv_desc.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() };
        let mut mv_srv = srv_desc;
        mv_srv.Format = DXGI_FORMAT_R16G16_FLOAT;
        let mut exp_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        exp_srv.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: 0,
            NumElements: 1,
            StructureByteStride: 4,
            Flags: D3D12_BUFFER_SRV_FLAG_NONE,
        };

        let mut uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        uav_desc.Anonymous.Texture2D = D3D12_TEX2D_UAV::default();
        let mut exp_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        };
        exp_uav.Anonymous.Buffer = D3D12_BUFFER_UAV {
            FirstElement: 0,
            NumElements: 1,
            StructureByteStride: 4,
            CounterOffsetInBytes: 0,
            Flags: D3D12_BUFFER_UAV_FLAG_NONE,
        };

        // Denoiser slots 74..81
        device.CreateShaderResourceView(&raw_rtgi, Some(&srv_desc), cpu_at(74));
        device.CreateShaderResourceView(&normal_depth, Some(&srv_desc), cpu_at(75));
        device.CreateShaderResourceView(&denoise_a, Some(&srv_desc), cpu_at(76));
        device.CreateShaderResourceView(&normal_depth, Some(&srv_desc), cpu_at(77));
        device.CreateShaderResourceView(&denoise_b, Some(&srv_desc), cpu_at(78));
        device.CreateShaderResourceView(&normal_depth, Some(&srv_desc), cpu_at(79));
        device.CreateUnorderedAccessView(&denoise_a, None, Some(&uav_desc), cpu_at(80));
        device.CreateUnorderedAccessView(&denoise_b, None, Some(&uav_desc), cpu_at(81));

        // TAA Slot 0 (82..87) & Slot 1 (88..93)
        for p in 0..2 {
            let b = 82 + p * 6;
            let prev = &taa_hist[p ^ 1];
            let cur = &taa_hist[p];
            device.CreateShaderResourceView(&direct_hdr, Some(&srv_desc), cpu_at(b));
            device.CreateShaderResourceView(&denoise_b, Some(&srv_desc), cpu_at(b + 1));
            device.CreateShaderResourceView(&motion_vec, Some(&mv_srv), cpu_at(b + 2));
            device.CreateShaderResourceView(prev, Some(&srv_desc), cpu_at(b + 3));
            device.CreateShaderResourceView(&normal_depth, Some(&srv_desc), cpu_at(b + 4));
            device.CreateUnorderedAccessView(cur, None, Some(&uav_desc), cpu_at(b + 5));
        }

        // AutoExposure & Bloom (94..100)
        device.CreateUnorderedAccessView(&exposure_buf, None, Some(&exp_uav), cpu_at(94));
        device.CreateShaderResourceView(&taa_hist[0], Some(&srv_desc), cpu_at(95));
        device.CreateShaderResourceView(&taa_hist[1], Some(&srv_desc), cpu_at(96));
        device.CreateShaderResourceView(&bloom_a, Some(&srv_desc), cpu_at(97));
        device.CreateShaderResourceView(&bloom_b, Some(&srv_desc), cpu_at(98));
        device.CreateUnorderedAccessView(&bloom_a, None, Some(&uav_desc), cpu_at(99));
        device.CreateUnorderedAccessView(&bloom_b, None, Some(&uav_desc), cpu_at(100));

        // Composite Slot 0 (101..103) & Slot 1 (104..106)
        device.CreateShaderResourceView(&taa_hist[0], Some(&srv_desc), cpu_at(101));
        device.CreateShaderResourceView(&bloom_a, Some(&srv_desc), cpu_at(102));
        device.CreateShaderResourceView(&exposure_buf, Some(&exp_srv), cpu_at(103));
        device.CreateShaderResourceView(&taa_hist[1], Some(&srv_desc), cpu_at(104));
        device.CreateShaderResourceView(&bloom_a, Some(&srv_desc), cpu_at(105));
        device.CreateShaderResourceView(&exposure_buf, Some(&exp_srv), cpu_at(106));

        let clamp_samp = static_sampler(0, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 1);

        // Helper closure-style root sig builder for compute post passes
        let make_cs_rs = |cb_dwords: u32, num_srvs: u32, use_samp: bool| -> ID3D12RootSignature {
            let srv_rng = D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: num_srvs,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 0,
            };
            let uav_rng = D3D12_DESCRIPTOR_RANGE {
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
                        Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: cb_dwords },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &srv_rng },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &uav_rng },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let samps = if use_samp { std::slice::from_ref(&clamp_samp) } else { &[] };
            create_root_sig(device, &params, samps, D3D12_ROOT_SIGNATURE_FLAG_NONE)
        };

        let denoise_root_sig = make_cs_rs(4, 2, false);
        let denoise_pso = create_compute_pso_local(device, &denoise_root_sig, RTGI_DENOISE_CS_DXIL);

        let taa_root_sig = make_cs_rs(8, 5, true);
        let taa_pso = create_compute_pso_local(device, &taa_root_sig, AAA_TAA_CS_DXIL);

        let autoexp_root_sig = make_cs_rs(4, 1, true);
        let autoexp_pso = create_compute_pso_local(device, &autoexp_root_sig, AAA_AUTOEXP_CS_DXIL);

        let bloom_root_sig = make_cs_rs(4, 1, true);
        let bloom_pso = create_compute_pso_local(device, &bloom_root_sig, AAA_BLOOM_CS_DXIL);

        // Composite root sig (CB b0 + SRV range(3) t0..t2 + clamp sampler s0)
        let c_srv_rng = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 3,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let c_params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 12 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &c_srv_rng },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let comp_root_sig = create_root_sig(device, &c_params, &[clamp_samp], D3D12_ROOT_SIGNATURE_FLAG_NONE);

        let mut rt_formats = [DXGI_FORMAT_UNKNOWN; 8];
        rt_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let mut comp_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(comp_root_sig.clone())),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_COMP_VS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_COMP_VS_DXIL.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_COMP_PS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_COMP_PS_DXIL.len() },
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
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: false.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: rt_formats,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let comp_pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&comp_desc).expect("comp_pso");
        let _ = ManuallyDrop::take(&mut comp_desc.pRootSignature);

        Self {
            width,
            height,
            half_w,
            half_h,
            direct_hdr,
            raw_rtgi,
            normal_depth,
            motion_vec,
            denoise_a,
            denoise_b,
            taa_hist,
            exposure_buf,
            bloom_a,
            bloom_b,
            gbuf_rtv_heap,
            rtv_inc,
            denoise_root_sig,
            denoise_pso,
            taa_root_sig,
            taa_pso,
            autoexp_root_sig,
            autoexp_pso,
            bloom_root_sig,
            bloom_pso,
            comp_root_sig,
            comp_pso,
            srv_inc,
        }
    }

    unsafe fn begin_gbuffer(&self, cmd: &ID3D12GraphicsCommandList, dsv_handle: D3D12_CPU_DESCRIPTOR_HANDLE) {
        let base = self.gbuf_rtv_heap.GetCPUDescriptorHandleForHeapStart();
        let rtvs = [
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr },
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + self.rtv_inc as usize },
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + 2 * self.rtv_inc as usize },
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + 3 * self.rtv_inc as usize },
        ];
        cmd.ClearRenderTargetView(rtvs[0], &[0.14, 0.08, 0.14, 1.0], None);
        cmd.ClearRenderTargetView(rtvs[1], &[0.0, 0.0, 0.0, 0.0], None);
        cmd.ClearRenderTargetView(rtvs[2], &[0.0, 0.0, 0.0, 9999.0], None);
        cmd.ClearRenderTargetView(rtvs[3], &[0.0, 0.0, 0.0, 0.0], None);
        cmd.ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, &[]);
        cmd.OMSetRenderTargets(4, Some(rtvs.as_ptr()), false, Some(&dsv_handle));
        let vp = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: self.width as f32,
            Height: self.height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let sc = RECT { left: 0, top: 0, right: self.width as i32, bottom: self.height as i32 };
        cmd.RSSetViewports(&[vp]);
        cmd.RSSetScissorRects(&[sc]);
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn run_post_and_composite(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        main_srv_heap: &ID3D12DescriptorHeap,
        backbuffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        out_w: u32,
        out_h: u32,
        rtgi_enabled: bool,
        frame_idx: u32,
        sun_screen_dir: [f32; 2],
        time: f32,
        hud: [f32; 8],
    ) {
        let srv_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let gpu_base = main_srv_heap.GetGPUDescriptorHandleForHeapStart();
        let gpu_at = |slot: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu_base.ptr + (slot * self.srv_inc as usize) as u64,
        };
        let cur = (frame_idx & 1) as usize;

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.direct_hdr, D3D12_RESOURCE_STATE_RENDER_TARGET, srv_state),
                transition_barrier(&self.raw_rtgi, D3D12_RESOURCE_STATE_RENDER_TARGET, srv_state),
                transition_barrier(&self.normal_depth, D3D12_RESOURCE_STATE_RENDER_TARGET, srv_state),
                transition_barrier(&self.motion_vec, D3D12_RESOURCE_STATE_RENDER_TARGET, srv_state),
            ],
        );

        cmd.SetDescriptorHeaps(&[Some(main_srv_heap.clone())]);

        let gx = self.width.div_ceil(8);
        let gy = self.height.div_ceil(8);

        // 1. 4-Pass A-Trous Bilateral Denoiser (step = 1, 2, 4, 8) for 2-spp Diffuse RTGI
        if rtgi_enabled {
            cmd.SetComputeRootSignature(&self.denoise_root_sig);
            cmd.SetPipelineState(&self.denoise_pso);

            cmd_barriers(cmd, vec![transition_barrier(&self.denoise_a, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            let c1 = [self.width, self.height, 1u32, 0u32];
            cmd.SetComputeRoot32BitConstants(0, 4, c1.as_ptr() as *const c_void, 0);
            cmd.SetComputeRootDescriptorTable(1, gpu_at(74));
            cmd.SetComputeRootDescriptorTable(2, gpu_at(80));
            cmd.Dispatch(gx, gy, 1);

            cmd_barriers(
                cmd,
                vec![
                    uav_barrier(&self.denoise_a),
                    transition_barrier(&self.denoise_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                    transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                ],
            );
            let c2 = [self.width, self.height, 2u32, 0u32];
            cmd.SetComputeRoot32BitConstants(0, 4, c2.as_ptr() as *const c_void, 0);
            cmd.SetComputeRootDescriptorTable(1, gpu_at(76));
            cmd.SetComputeRootDescriptorTable(2, gpu_at(81));
            cmd.Dispatch(gx, gy, 1);

            cmd_barriers(
                cmd,
                vec![
                    uav_barrier(&self.denoise_b),
                    transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                    transition_barrier(&self.denoise_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                ],
            );
            let c3 = [self.width, self.height, 4u32, 0u32];
            cmd.SetComputeRoot32BitConstants(0, 4, c3.as_ptr() as *const c_void, 0);
            cmd.SetComputeRootDescriptorTable(1, gpu_at(78));
            cmd.SetComputeRootDescriptorTable(2, gpu_at(80));
            cmd.Dispatch(gx, gy, 1);

            cmd_barriers(
                cmd,
                vec![
                    uav_barrier(&self.denoise_a),
                    transition_barrier(&self.denoise_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                    transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                ],
            );
            let c4 = [self.width, self.height, 8u32, 0u32];
            cmd.SetComputeRoot32BitConstants(0, 4, c4.as_ptr() as *const c_void, 0);
            cmd.SetComputeRootDescriptorTable(1, gpu_at(76));
            cmd.SetComputeRootDescriptorTable(2, gpu_at(81));
            cmd.Dispatch(gx, gy, 1);

            cmd_barriers(
                cmd,
                vec![
                    uav_barrier(&self.denoise_b),
                    transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                    transition_barrier(&self.denoise_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
        } else {
            cmd_barriers(cmd, vec![transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
        }

        // 2. Screen-Space Contact Shadows + Variance-Clipped YCoCg TAA
        cmd_barriers(cmd, vec![transition_barrier(&self.taa_hist[cur], srv_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        cmd.SetComputeRootSignature(&self.taa_root_sig);
        cmd.SetPipelineState(&self.taa_pso);
        let taa_cb: [u32; 8] = [
            self.width,
            self.height,
            frame_idx,
            if rtgi_enabled { 0.80f32.to_bits() } else { 0u32 },
            sun_screen_dir[0].to_bits(),
            sun_screen_dir[1].to_bits(),
            1.0f32.to_bits(),
            0u32,
        ];
        cmd.SetComputeRoot32BitConstants(0, 8, taa_cb.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootDescriptorTable(1, gpu_at(82 + cur * 6));
        cmd.SetComputeRootDescriptorTable(2, gpu_at(87 + cur * 6));
        cmd.Dispatch(gx, gy, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.taa_hist[cur]),
                transition_barrier(&self.taa_hist[cur], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, srv_state),
                transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&self.exposure_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );

        // 3. Log-Average Histogram Auto-Exposure
        cmd.SetComputeRootSignature(&self.autoexp_root_sig);
        cmd.SetPipelineState(&self.autoexp_pso);
        let exp_cb = [self.width, self.height, frame_idx, 0.06f32.to_bits()];
        cmd.SetComputeRoot32BitConstants(0, 4, exp_cb.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootDescriptorTable(1, gpu_at(95 + cur));
        cmd.SetComputeRootDescriptorTable(2, gpu_at(94));
        cmd.Dispatch(1, 1, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.exposure_buf),
                transition_barrier(&self.exposure_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            ],
        );

        // 4. 3-Dispatch Half-Res HDR Bloom (threshold=1.45 to keep sunset sky crisp)
        let bx = self.half_w.div_ceil(8);
        let by = self.half_h.div_ceil(8);
        cmd.SetComputeRootSignature(&self.bloom_root_sig);
        cmd.SetPipelineState(&self.bloom_pso);

        cmd_barriers(cmd, vec![transition_barrier(&self.bloom_a, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        let b0 = [self.half_w, self.half_h, 0u32, 1.45f32.to_bits()];
        cmd.SetComputeRoot32BitConstants(0, 4, b0.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootDescriptorTable(1, gpu_at(95 + cur)); // taa_hist[cur]
        cmd.SetComputeRootDescriptorTable(2, gpu_at(99)); // bloom_a UAV
        cmd.Dispatch(bx, by, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.bloom_a),
                transition_barrier(&self.bloom_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.bloom_b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );
        let b1 = [self.half_w, self.half_h, 1u32, 0u32];
        cmd.SetComputeRoot32BitConstants(0, 4, b1.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootDescriptorTable(1, gpu_at(97)); // bloom_a SRV
        cmd.SetComputeRootDescriptorTable(2, gpu_at(100)); // bloom_b UAV
        cmd.Dispatch(bx, by, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.bloom_b),
                transition_barrier(&self.bloom_b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.bloom_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );
        let b2 = [self.half_w, self.half_h, 2u32, 0u32];
        cmd.SetComputeRoot32BitConstants(0, 4, b2.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootDescriptorTable(1, gpu_at(98)); // bloom_b SRV
        cmd.SetComputeRootDescriptorTable(2, gpu_at(99)); // bloom_a UAV
        cmd.Dispatch(bx, by, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.bloom_a),
                transition_barrier(&self.bloom_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.bloom_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
            ],
        );

        // 5. Final Composite Pass: TAA + Bloom + Auto-Exposure + AgX Filmic + CAS Sharpen
        cmd.OMSetRenderTargets(1, Some(&backbuffer_rtv), false, None);
        let vp = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: out_w as f32,
            Height: out_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let sc = RECT { left: 0, top: 0, right: out_w as i32, bottom: out_h as i32 };
        cmd.RSSetViewports(&[vp]);
        cmd.RSSetScissorRects(&[sc]);

        cmd.SetGraphicsRootSignature(&self.comp_root_sig);
        cmd.SetPipelineState(&self.comp_pso);
        let comp_cb = [
            0.12f32, 0.35f32, 1.0f32, time,
            hud[0], hud[1], hud[2], hud[3],
            hud[4], hud[5], hud[6], hud[7],
        ];
        cmd.SetGraphicsRoot32BitConstants(0, 12, comp_cb.as_ptr() as *const c_void, 0);
        cmd.SetGraphicsRootDescriptorTable(1, gpu_at(101 + cur * 3));
        cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd.DrawInstanced(3, 1, 0, 0);

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.direct_hdr, srv_state, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition_barrier(&self.raw_rtgi, srv_state, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition_barrier(&self.normal_depth, srv_state, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition_barrier(&self.motion_vec, srv_state, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition_barrier(&self.bloom_a, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&self.exposure_buf, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
            ],
        );
    }
}

fn run_flush_bench_mode(pack: &AaaAssetPack, enable_debug: bool) {
    let mut ctx = GpuContext::new(enable_debug);
    let max_tiles = MAX_BATCH_TILES;
    let mut gpu_dec = Bc16GpuStreamer::new(
        &mut ctx,
        max_tiles,
        max_tiles * TILE_BLOCKS * 2,
        max_tiles * 96 * 1024,
    );

    println!(
        "[aaa-flush-bench] Adapter: {} | Assets: {} PBR textures (4K BC7/BC5 + CarConcept glTF), {} tiles ({:.2} MiB Mip0 BC7/BC5 <- {:.2} MiB BCPack, ratio {:.2}x)",
        ctx.adapter_name,
        pack.textures.len(),
        pack.total_tiles,
        pack.total_mip0_bytes as f64 / (1024.0 * 1024.0),
        pack.total_bcpack_bytes as f64 / (1024.0 * 1024.0),
        pack.total_mip0_bytes as f64 / pack.total_bcpack_bytes.max(1) as f64,
    );

    // 1) GPU Async Compute full 4K VRAM Flush -> Residency wave (128 x 64KiB tiles/batch)
    let gpu_start = Instant::now();
    let mut gpu_kernel_ms = 0.0f64;
    let mut gpu_batches = 0usize;
    let all_refs: Vec<TileRef> = pack
        .textures
        .iter()
        .enumerate()
        .flat_map(|(ti, tex)| (0..tex.num_strips).map(move |s| TileRef { tex_idx: ti, strip_idx: s }))
        .collect();

    for (bi, chunk) in all_refs.chunks(MAX_BATCH_TILES).enumerate() {
        let slot = bi & 1;
        let batch = pack_batch(chunk.iter().map(|t| {
            (pack.textures[t.tex_idx].bcpack_tiles[t.strip_idx].as_slice(), TILE_BLOCKS * 2)
        }));
        let fence = gpu_dec.submit_slot_async_compute(&mut ctx, slot, &batch);
        let _ = ctx.wait_compute_cpu(fence, u32::MAX);
        gpu_kernel_ms += gpu_dec.read_slot_kernel_ms(&ctx, slot);
        gpu_batches += 1;
    }
    let gpu_wall_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;
    let gpu_kernel_gbps = (pack.total_mip0_bytes as f64 / 1e9) / (gpu_kernel_ms / 1000.0);
    let gpu_eff_gbps = (pack.total_mip0_bytes as f64 / 1e9) / (gpu_wall_ms / 1000.0);

    // 2) CPU 1-Thread Baseline over identical 64 KiB BC7/BC5 tiles
    let cpu_start = Instant::now();
    for t in &all_refs {
        let dec = decode_bc16_from_bcpack_cpu(&pack.textures[t.tex_idx].bcpack_tiles[t.strip_idx], TILE_BLOCKS);
        std::hint::black_box(dec[0]);
    }
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
    let cpu_mbps = (pack.total_mip0_bytes as f64 / 1e6) / (cpu_ms / 1000.0);

    println!(
        "[aaa-flush-bench] GPU Async Compute (BC7/BC5): batches={}, kernel={:.3} ms ({:.2} GB/s), wall={:.3} ms ({:.2} GB/s)",
        gpu_batches, gpu_kernel_ms, gpu_kernel_gbps, gpu_wall_ms, gpu_eff_gbps
    );
    println!(
        "[aaa-flush-bench] CPU 1-Thread Baseline (BC7/BC5): {:.3} ms ({:.1} MB/s) -> GPU Kernel Speedup = {:.1}x | GPU Wall Speedup = {:.1}x",
        cpu_ms,
        cpu_mbps,
        cpu_ms / gpu_kernel_ms.max(1e-4),
        cpu_ms / gpu_wall_ms.max(1e-4),
    );
    if enable_debug {
        unsafe { dump_d3d12_info_queue(&ctx.device) };
    }
}

pub fn run_aaa_demo(cfg: AaaDemoConfig) {
    let aaa_dir = cfg
        .aaa_dir
        .clone()
        .or_else(|| find_aaa_assets_dir(Path::new("assets")))
        .or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("aaa_assets")))
                .filter(|p| p.join("CarConcept.glb").exists())
        })
        .expect("Could not locate aaa_assets directory; pass --aaa-dir <path>");
    let window_tbl = Window::default_table();
    let quads_tbl = QuadTable::default_table();
    let pack = load_aaa_asset_pack(&aaa_dir, &window_tbl, &quads_tbl);

    if cfg.flush_bench {
        run_flush_bench_mode(&pack, cfg.enable_debug);
        return;
    }

    unsafe {
        let mut ctx = GpuContext::new(cfg.enable_debug);
        let mut gpu_dec = Bc16GpuStreamer::new(
            &mut ctx,
            MAX_BATCH_TILES,
            MAX_BATCH_TILES * TILE_BLOCKS * 2,
            MAX_BATCH_TILES * 96 * 1024,
        );

        // Win32 Window
        let hinst = GetModuleHandleW(None).expect("GetModuleHandleW");
        let class_name = w!("BcpackAaaShowcaseWndV8");
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
        let mut rect = RECT { left: 0, top: 0, right: WIN_W as i32, bottom: WIN_H as i32 };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false).unwrap();
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class_name,
            w!("BCPack AAA Showcase v0.8 (4K BC7/BC5 + Active SFS + DXR 1.1) - Initializing..."),
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

        let hud_default = cfg.screenshot_path.is_none() && cfg.record_dir.is_none();
        let mut wstate = Box::new(WindowState {
            use_gpu: !cfg.start_cpu,
            rt_enabled: !cfg.no_rt,
            rtgi_enabled: !cfg.no_rt && !cfg.no_rtgi,
            xray_enabled: cfg.xray_mode,
            vsync_enabled: if cfg.fps_cap > 0 { false } else { cfg.vsync },
            sfs_enabled: cfg.vram_budget_mb > 0,
            cam_mode: 0,
            fps_cap: cfg.fps_cap,
            sim_speed: cfg.sim_speed.clamp(0.0, 4.0),
            saved_sim_speed: 1.0,
            hud_visible: cfg.show_hud.unwrap_or(hud_default),
            drag_slider: 0,
            flush_requested: false,
            should_quit: false,
        });
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (&mut *wstate as *mut WindowState) as isize);
        let _ = ShowWindow(hwnd, SW_SHOW);

        let sc_desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: SWAP_W,
            Height: SWAP_H,
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
        let sc1 = ctx.factory.CreateSwapChainForHwnd(&ctx.queue, hwnd, &sc_desc, None, None).unwrap();
        let swap_chain: IDXGISwapChain3 = sc1.cast().unwrap();

        let bb_rtv_heap: ID3D12DescriptorHeap = ctx
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
        // 0..36 (table slot 0) + 37..73 (table slot 1) + 74..106 (Post) + 107 (Depth R32) + 108..119 (Hi-Z SRVs/UAVs)
        let main_srv_heap: ID3D12DescriptorHeap = ctx
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 160,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap();

        let rtv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let srv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let mut backbuffers = Vec::with_capacity(3);
        let bb_base = bb_rtv_heap.GetCPUDescriptorHandleForHeapStart();
        for i in 0..3u32 {
            let bb: ID3D12Resource = swap_chain.GetBuffer(i).unwrap();
            ctx.device.CreateRenderTargetView(
                &bb,
                None,
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: bb_base.ptr + (i * rtv_inc) as usize },
            );
            backbuffers.push(bb);
        }

        let (render_w, render_h) = if cfg.render_1440p { (2560u32, 1440u32) } else { (SWAP_W, SWAP_H) };
        let depth_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: render_w as u64,
            Height: render_h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let depth_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 },
            },
        };
        let mut depth_tex: Option<ID3D12Resource> = None;
        ctx.device
            .CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &depth_desc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                Some(&depth_clear),
                &mut depth_tex,
            )
            .unwrap();
        let _depth_tex = depth_tex.unwrap();
        let dsv_handle = dsv_heap.GetCPUDescriptorHandleForHeapStart();
        let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        ctx.device.CreateDepthStencilView(&_depth_tex, Some(&dsv_desc), dsv_handle);

        // Unified Geometry Buffer (CarConcept + UnitBox + GroundPlane)
        let mut all_verts = pack.car.vertices.clone();
        let mut all_indices = pack.car.indices.clone();
        let car_vert_count = all_verts.len() as u32;
        let car_idx_count = all_indices.len() as u32;

        let (box_v, box_i) = build_unit_box_pbr();
        let box_base_vert = all_verts.len() as u32;
        let box_idx_offset = all_indices.len() as u32;
        all_verts.extend_from_slice(&box_v);
        all_indices.extend_from_slice(&box_i);

        let (plane_v, plane_i) = build_plane_pbr();
        let plane_base_vert = all_verts.len() as u32;
        let plane_idx_offset = all_indices.len() as u32;
        all_verts.extend_from_slice(&plane_v);
        all_indices.extend_from_slice(&plane_i);

        let vb_bytes = (all_verts.len() * size_of::<PbrVertex>()) as u64;
        let ib_bytes = (all_indices.len() * size_of::<u32>()) as u64;
        let vb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, vb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let ib_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, ib_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        upload_slice(&vb_up, &all_verts);
        upload_slice(&ib_up, &all_indices);
        let vb_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, vb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        let ib_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, ib_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        // Live 23 PBR BC7/BC5 Textures (4K PolyHaven + glTF) with Console-Native Full Mipmap Chains + Dummy + HDRI
        let mut pbr_textures = Vec::with_capacity(pack.textures.len());
        for t in &pack.textures {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: t.width as u64,
                Height: t.height,
                DepthOrArraySize: 1,
                MipLevels: t.mip_levels as u16,
                Format: dxgi_format_for(t.format),
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            ctx.device
                .CreateCommittedResource(
                    &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    None,
                    &mut res,
                )
                .unwrap();
            pbr_textures.push(res.unwrap());
        }

        // GPU Sampler Feedback (SFS) buffer: 32 textures x 16 bytes (min_mip_fp8, hit_count, 0, 0)
        let sfs_init_vals: Vec<u32> = (0..TEX_SLOT_CAP).flat_map(|_| [u32::MAX, 0, 0, 0]).collect();
        let sfs_buf_bytes = (TEX_SLOT_CAP * 16) as u64;
        let sfs_feedback_init = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, sfs_buf_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        upload_slice(&sfs_feedback_init, &sfs_init_vals);
        let sfs_feedback_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, sfs_buf_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        let sfs_feedback_readback = [
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, sfs_buf_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST),
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, sfs_buf_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST),
        ];

        // Dummy 1x1 white RGBA8 texture for unused tN slots (<32)
        let dummy_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        let mut dummy_tex: Option<ID3D12Resource> = None;
        ctx.device.CreateCommittedResource(&heap_props(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &dummy_desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut dummy_tex).unwrap();
        let dummy_tex = dummy_tex.unwrap();
        let dummy_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        upload_slice(&dummy_up, &[255u8, 255, 255, 255]);

        // 2K Sunset HDRI FP16 Texture
        let hdri_tex = create_fp16_tex(&ctx.device, pack.hdri_w, pack.hdri_h, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, None);
        let hdri_row_pitch = (pack.hdri_w * 8).div_ceil(256) * 256;
        let hdri_up = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (hdri_row_pitch * pack.hdri_h) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        upload_slice(&hdri_up, &pack.hdri_fp16);

        // One-shot Upload Geometry + Dummy + HDRI + CPU Upload Buffer for CPU fallback streaming
        let init_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let init_cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &init_alloc, None).unwrap();
        init_cmd.CopyBufferRegion(&vb_def, 0, &vb_up, 0, vb_bytes);
        init_cmd.CopyBufferRegion(&ib_def, 0, &ib_up, 0, ib_bytes);

        let mut d_dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(dummy_tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let mut d_src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(dummy_up.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: 1,
                        Height: 1,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };
        init_cmd.CopyTextureRegion(&d_dst, 0, 0, 0, &d_src, None);
        let _ = ManuallyDrop::take(&mut d_dst.pResource);
        let _ = ManuallyDrop::take(&mut d_src.pResource);

        let mut h_dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(hdri_tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let mut h_src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(hdri_up.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                        Width: pack.hdri_w,
                        Height: pack.hdri_h,
                        Depth: 1,
                        RowPitch: hdri_row_pitch,
                    },
                },
            },
        };
        init_cmd.CopyTextureRegion(&h_dst, 0, 0, 0, &h_src, None);
        let _ = ManuallyDrop::take(&mut h_dst.pResource);
        let _ = ManuallyDrop::take(&mut h_src.pResource);

        let geo_srv_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        cmd_barriers(
            &init_cmd,
            vec![
                transition_barrier(&vb_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | geo_srv_state),
                transition_barrier(&ib_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER | geo_srv_state),
                transition_barrier(&dummy_tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                transition_barrier(&hdri_tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            ],
        );
        init_cmd.Close().unwrap();
        let exec = [Some(init_cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&exec);
        ctx.signal_and_wait();

        // Decode & upload Console Mip 1..Tail (incl. Packed Mip Tail 32..4) across all 23 PBR textures via Async Compute
        decode_and_upload_lower_mips(&mut ctx, &mut gpu_dec, &pack.textures, &pbr_textures);

        let dxr = AaaDxrScene::new(
            &mut ctx,
            &vb_def,
            &ib_def,
            car_vert_count,
            car_idx_count,
            box_base_vert,
            box_idx_offset,
            plane_base_vert,
            plane_idx_offset,
        );
        if dxr.is_none() {
            wstate.rt_enabled = false;
            wstate.rtgi_enabled = false;
        }

        // Populate main_srv_heap slots 0..36 (frame slot 0) and 37..73 (frame slot 1)
        let srv_base = main_srv_heap.GetCPUDescriptorHandleForHeapStart();
        let cpu_at = |idx: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_base.ptr + idx * srv_inc as usize,
        };
        for frame_slot in 0..2 {
            let base = frame_slot * SRV_TABLE_STRIDE;
            for i in 0..TEX_SLOT_CAP {
                if i < pbr_textures.len() {
                    let mut sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: dxgi_format_for(pack.textures[i].format),
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        ..Default::default()
                    };
                    sd.Anonymous.Texture2D = D3D12_TEX2D_SRV {
                        MipLevels: pack.textures[i].mip_levels,
                        ResourceMinLODClamp: 1.0, // Start clamped to Mip 1..Tail until Mip 0 finishes streaming
                        ..Default::default()
                    };
                    ctx.device.CreateShaderResourceView(&pbr_textures[i], Some(&sd), cpu_at(base + i));
                } else {
                    let mut sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        ..Default::default()
                    };
                    sd.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() };
                    ctx.device.CreateShaderResourceView(&dummy_tex, Some(&sd), cpu_at(base + i));
                }
            }
            let mut hdri_sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            hdri_sd.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() };
            ctx.device.CreateShaderResourceView(&hdri_tex, Some(&hdri_sd), cpu_at(base + 32));

            if let Some(ref rt) = dxr {
                let mut tlas_sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_UNKNOWN,
                    ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    ..Default::default()
                };
                tlas_sd.Anonymous.RaytracingAccelerationStructure.Location = rt.tlas_result[frame_slot].GetGPUVirtualAddress();
                ctx.device.CreateShaderResourceView(None, Some(&tlas_sd), cpu_at(base + 33));

                let mut meta_sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_UNKNOWN,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    ..Default::default()
                };
                meta_sd.Anonymous.Buffer = D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: MAX_RT_INSTANCES as u32,
                    StructureByteStride: size_of::<RtInstMeta>() as u32,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                };
                ctx.device.CreateShaderResourceView(&rt.inst_meta_upload[frame_slot], Some(&meta_sd), cpu_at(base + 34));
            } else {
                // Fallback SRVs when DXR is absent
                let mut buf_sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_UNKNOWN,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    ..Default::default()
                };
                buf_sd.Anonymous.Buffer = D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 1,
                    StructureByteStride: size_of::<RtInstMeta>() as u32,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                };
                ctx.device.CreateShaderResourceView(&vb_def, Some(&buf_sd), cpu_at(base + 33));
                ctx.device.CreateShaderResourceView(&vb_def, Some(&buf_sd), cpu_at(base + 34));
            }

            // t35 = ByteAddressBuffer g_SceneVB, t36 = StructuredBuffer<uint> g_SceneIB (for textured DXR hits)
            let mut vb_sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            vb_sd.Anonymous.Buffer = D3D12_BUFFER_SRV {
                FirstElement: 0,
                NumElements: (vb_bytes / 4) as u32,
                StructureByteStride: 0,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
            };
            ctx.device.CreateShaderResourceView(&vb_def, Some(&vb_sd), cpu_at(base + 35));

            let mut ib_sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            ib_sd.Anonymous.Buffer = D3D12_BUFFER_SRV {
                FirstElement: 0,
                NumElements: all_indices.len() as u32,
                StructureByteStride: 4,
                Flags: D3D12_BUFFER_SRV_FLAG_NONE,
            };
            ctx.device.CreateShaderResourceView(&ib_def, Some(&ib_sd), cpu_at(base + 36));
        }

        let post = AaaPostPipeline::new(&ctx.device, render_w, render_h, &main_srv_heap, srv_inc);

        // Sky PSO
        let sky_srv_rng = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let sky_params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: (size_of::<SkyConstants>() / 4) as u32 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &sky_srv_rng },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let env_samp = static_sampler(0, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 1);
        let sky_root_sig = create_root_sig(&ctx.device, &sky_params, &[env_samp], D3D12_ROOT_SIGNATURE_FLAG_NONE);

        let mut gbuf_formats = [DXGI_FORMAT_UNKNOWN; 8];
        gbuf_formats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        gbuf_formats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        gbuf_formats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        gbuf_formats[3] = DXGI_FORMAT_R16G16_FLOAT;
        let gbuf_blends = [D3D12_RENDER_TARGET_BLEND_DESC {
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        }; 8];

        let mut sky_pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(sky_root_sig.clone())),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_SKY_VS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_SKY_VS_DXIL.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_SKY_PS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_SKY_PS_DXIL.len() },
            BlendState: D3D12_BLEND_DESC { RenderTarget: gbuf_blends, ..Default::default() },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 4,
            RTVFormats: gbuf_formats,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let sky_pso: ID3D12PipelineState = ctx.device.CreateGraphicsPipelineState(&sky_pso_desc).unwrap();
        let _ = ManuallyDrop::take(&mut sky_pso_desc.pRootSignature);

        // AAA PBR PSO (b0 = AaaDrawCB 52 DWORDs, t0..t36 SRV table, b1 = AaaFrameCB root CBV, u0 = g_SfsFeedback root UAV)
        let pbr_srv_rng = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: SRV_TABLE_STRIDE as u32,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let pbr_params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: (size_of::<AaaDrawCB>() / 4) as u32,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &pbr_srv_rng },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 1, RegisterSpace: 0 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
        ];
        let pbr_samplers = [
            static_sampler(0, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP, 8),
            static_sampler(1, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 1),
        ];
        let pbr_root_sig = create_root_sig(
            &ctx.device,
            &pbr_params,
            &pbr_samplers,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        );

        let frame_cb_upload = [
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
        ];

        let input_elems = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::PCSTR(b"POSITION\0".as_ptr()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::PCSTR(b"NORMAL\0".as_ptr()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 12,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::PCSTR(b"TANGENT\0".as_ptr()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 24,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::PCSTR(b"TEXCOORD\0".as_ptr()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 40,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];
        let mut pbr_pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(pbr_root_sig.clone())),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_PBR_VS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_PBR_VS_DXIL.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_PBR_PS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_PBR_PS_DXIL.len() },
            BlendState: D3D12_BLEND_DESC { RenderTarget: gbuf_blends, ..Default::default() },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE, // glTF doubleSided trim/grilles/lights
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
            NumRenderTargets: 4,
            RTVFormats: gbuf_formats,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let pbr_pso: ID3D12PipelineState = ctx.device.CreateGraphicsPipelineState(&pbr_pso_desc).unwrap();
        let _ = ManuallyDrop::take(&mut pbr_pso_desc.pRootSignature);

        // Per-frame direct command allocators & lists
        let cmd_allocs: [ID3D12CommandAllocator; 2] = [
            ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap(),
            ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap(),
        ];
        let cmd_lists: [ID3D12GraphicsCommandList; 2] = [
            ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &cmd_allocs[0], None).unwrap(),
            ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &cmd_allocs[1], None).unwrap(),
        ];
        cmd_lists[0].Close().unwrap();
        cmd_lists[1].Close().unwrap();

        // CPU fallback upload buffer for MAX_BATCH_TILES (64 KiB each for BC7/BC5)
        let cpu_upload_buf = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (MAX_BATCH_TILES * TILE_BC16_BYTES) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let mut cpu_scratch = vec![0u8; MAX_BATCH_TILES * TILE_BC16_BYTES];

        // Screenshot readback
        let ss_row_pitch = (SWAP_W * 4).div_ceil(256) * 256;
        let ss_readback = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_READBACK,
            (ss_row_pitch * SWAP_H) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let budget_bytes = cfg.vram_budget_mb * 1024 * 1024;
        let mut residency = SfsResidencyManager::new(&pack.textures, budget_bytes);
        // Kickstart initial Mip-0 stream with hero car paint & 4K road asphalt before first frame SFS feedback arrives
        for seed_idx in [
            pack.asphalt_tex[0] as usize,
            pack.asphalt_tex[1] as usize,
            pack.promenade_tex[0] as usize,
        ] {
            if seed_idx < pack.textures.len() {
                residency.tex[seed_idx].last_hot_frame = 1;
                residency.enqueue_tex(seed_idx);
            }
        }

        let vb_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vb_def.GetGPUVirtualAddress(),
            SizeInBytes: vb_bytes as u32,
            StrideInBytes: size_of::<PbrVertex>() as u32,
        };
        let ib_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: ib_def.GetGPUVirtualAddress(),
            SizeInBytes: ib_bytes as u32,
            Format: DXGI_FORMAT_R32_UINT,
        };

        let hiz_pyramid = AaaHiZPyramid::new(&ctx.device, render_w, render_h, &_depth_tex, &main_srv_heap, srv_inc);
        let meshlet_city = if !cfg.no_city {
            AaaMeshletCity::new(&mut ctx, SRV_TABLE_STRIDE)
        } else {
            None
        };
        let mut crowd_renderer = if !cfg.no_crowd {
            Some(AaaCrowdRenderer::new(&mut ctx, SRV_TABLE_STRIDE))
        } else {
            None
        };

        // Front-row DXR proxy boxes for city facades (or simple towers if --no-city)
        let towers: Vec<([f32; 3], [f32; 3], [f32; 3])> = if let Some(ref city) = meshlet_city {
            city.dxr_tower_boxes.clone()
        } else {
            let mut t = Vec::new();
            for i in 0..16 {
                let z = -60.0 + (i as f32) * 18.0;
                let h = 22.0 + ((i * 37 % 5) as f32) * 10.0;
                let w = 11.0 + ((i * 13 % 3) as f32) * 2.5;
                t.push(([-26.0 - (i % 2) as f32 * 7.0, h * 0.5, z], [w, h, w * 1.1], [0.85, 0.78, 0.72]));
            }
            t
        };

        let parked_car_mats: Vec<(Mat4, [f32; 4])> = [
            (Mat4::translation(9.8, 0.0, 14.0).mul(&Mat4::rotate_y(0.04)), [0.88, 0.88, 0.90, 0.0]),
            (Mat4::translation(9.8, 0.0, 29.0).mul(&Mat4::rotate_y(-0.03)), [0.18, 0.48, 0.82, 0.0]),
            (Mat4::translation(9.8, 0.0, 52.0).mul(&Mat4::rotate_y(0.02)), [0.90, 0.62, 0.16, 0.0]),
            (Mat4::translation(-9.8, 0.0, 22.0).mul(&Mat4::rotate_y(std::f32::consts::PI - 0.03)), [0.16, 0.72, 0.50, 0.0]),
            (Mat4::translation(-9.8, 0.0, 44.0).mul(&Mat4::rotate_y(std::f32::consts::PI + 0.02)), [0.24, 0.26, 0.30, 0.0]),
            (Mat4::translation(-9.8, 0.0, 74.0).mul(&Mat4::rotate_y(std::f32::consts::PI)), [0.82, 0.18, 0.22, 0.0]),
        ]
        .to_vec();

        println!("=== BCPack AAA Showcase v0.8.0 ===");
        println!(
            "  Assets: CarConcept.glb ({} tris, {} submeshes + 6 parked) + 9x Poly Haven 4K PBR (BC7/BC5) + 2K Sunset HDRI",
            pack.car.indices.len() / 3,
            pack.car.submeshes.len()
        );
        if let Some(ref city) = meshlet_city {
            println!(
                "  Mesh-Shader City + Hi-Z: {} inst, {} meshlets ({:.2}M tris pre-cull) | Crowd: {} GPU-skinned NPCs ({} verts)",
                city.instance_count,
                city.total_work_items,
                city.total_tris as f64 / 1e6,
                if crowd_renderer.is_some() { CROWD_NPCS } else { 0 },
                crowd_renderer.as_ref().map(|c| c.verts_total()).unwrap_or(0)
            );
        }
        println!(
            "  BCPack Stream: {} textures, {} 64K-tiles ({:.2} MiB Mip0 / {:.2} MiB Full MipChain <- {:.2} MiB BCPack, SFS Budget={} MiB)",
            pack.textures.len(),
            pack.total_tiles,
            pack.total_mip0_bytes as f64 / (1024.0 * 1024.0),
            pack.total_vram_bytes as f64 / (1024.0 * 1024.0),
            pack.total_bcpack_bytes as f64 / (1024.0 * 1024.0),
            cfg.vram_budget_mb
        );
        println!("  Controls: [H] HUD Sliders (FPS Cap + Sim Speed) | [/] FPS | -/= Sim | [P] Pause | [T] Flush | [V] X-Ray | [F] SFS | [L] VSync | [SPACE] GPU/CPU | [R] DXR | [G] RTGI | [C] Cam | [ESC] Quit");

        let mut frame_fences = [0u64; 2];
        let mut frame_count: u32 = 0;
        let mut free_pos = [-4.2f32, 1.75, -5.0];
        let mut free_yaw = 0.62f32;
        let start_time = Instant::now();
        let mut last_frame = start_time;
        let mut last_title = start_time;
        let mut fps_accum = 0u32;
        let mut live_fps = if wstate.vsync_enabled { 60.0f32 } else { 120.0f32 };
        let mut sim_time = 0.0f32;
        let mut last_decode_ms = 0.0f64;
        let mut prev_vp_unjittered: Option<Mat4> = None;
        let mut prev_traffic_mats: Option<Vec<Mat4>> = None;
        let mut last_cull_stats = CityCullStats::default();
        let mut bench_frame_ms: Vec<f64> = Vec::with_capacity(16384);
        let mut bench_peak_gbps = 0.0f64;
        let mut bench_sum_vis_tris = 0u64;
        let mut bench_sum_hiz_tris = 0u64;
        let mut bench_flushed = false;

        if let Some(ref dir) = cfg.record_dir {
            let _ = std::fs::create_dir_all(dir);
        }

        'main_loop: loop {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                if msg.message == WM_QUIT {
                    wstate.should_quit = true;
                    break 'main_loop;
                }
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if wstate.should_quit || !IsWindow(hwnd).as_bool() {
                break;
            }

            let now = Instant::now();
            let raw_dt_ms = (now - last_frame).as_secs_f64() * 1000.0;
            const RECORD_WARMUP: u32 = 24;
            let (dt, t_sec) = if cfg.record_dir.is_some() {
                let rec_step = frame_count.saturating_sub(RECORD_WARMUP);
                let rec_t = (rec_step as f32) / 15.0;
                (1.0f32 / 15.0, rec_t)
            } else {
                let frame_dt = ((raw_dt_ms * 0.001) as f32).min(0.05);
                sim_time += frame_dt * wstate.sim_speed;
                (frame_dt, sim_time)
            };
            last_frame = now;
            residency.tick_timers(dt);

            if cfg.benchmark_run && frame_count > 4 {
                bench_frame_ms.push(raw_dt_ms);
            }
            if cfg.benchmark_run && cfg.record_dir.is_none() && !bench_flushed && t_sec >= cfg.benchmark_secs * 0.75 {
                wstate.flush_requested = true;
                bench_flushed = true;
            }
            if cfg.flush_at_frame == Some(frame_count) {
                wstate.flush_requested = true;
            }
            if wstate.flush_requested {
                residency.flush_to_clay();
                wstate.flush_requested = false;
            }

            let slot = (frame_count as usize) & 1;
            ctx.wait_direct_cpu(frame_fences[slot]);

            // Stream pending 64 KiB (4K BC7/BC5) BCPack tiles via Async Compute (or CPU 1-thread when toggled)
            let batch_tiles_n = cfg.tiles_per_frame.clamp(16, MAX_BATCH_TILES);
            let taken = residency.take_batch(batch_tiles_n);

            let mut compute_fence = 0u64;
            let mut streamed_mb = 0.0f64;
            if !taken.is_empty() {
                streamed_mb = (taken.len() * TILE_BC16_BYTES) as f64 / (1024.0 * 1024.0);
                if wstate.use_gpu {
                    let batch = pack_batch(taken.iter().map(|r| {
                        (pack.textures[r.tex_idx].bcpack_tiles[r.strip_idx].as_slice(), TILE_BLOCKS * 2)
                    }));
                    compute_fence = gpu_dec.submit_slot_async_compute(&mut ctx, slot, &batch);
                } else {
                    let t_cpu = Instant::now();
                    for (bi, r) in taken.iter().enumerate() {
                        let dec = decode_bc16_from_bcpack_cpu(
                            &pack.textures[r.tex_idx].bcpack_tiles[r.strip_idx],
                            TILE_BLOCKS,
                        );
                        cpu_scratch[bi * TILE_BC16_BYTES..(bi + 1) * TILE_BC16_BYTES].copy_from_slice(&dec);
                    }
                    last_decode_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
                    upload_slice(&cpu_upload_buf, &cpu_scratch[..taken.len() * TILE_BC16_BYTES]);
                }
            }

            // Camera positioning
            let speed = if (GetAsyncKeyState(VK_SHIFT.0 as i32) as u16 & 0x8000) != 0 { 24.0 } else { 8.0 };
            if wstate.cam_mode == 2 {
                let fwd = [free_yaw.sin(), 0.0, free_yaw.cos()];
                let right = [free_yaw.cos(), 0.0, -free_yaw.sin()];
                if (GetAsyncKeyState(b'W' as i32) as u16 & 0x8000) != 0 {
                    free_pos[0] += fwd[0] * speed * dt;
                    free_pos[2] += fwd[2] * speed * dt;
                }
                if (GetAsyncKeyState(b'S' as i32) as u16 & 0x8000) != 0 {
                    free_pos[0] -= fwd[0] * speed * dt;
                    free_pos[2] -= fwd[2] * speed * dt;
                }
                if (GetAsyncKeyState(b'D' as i32) as u16 & 0x8000) != 0 {
                    free_pos[0] += right[0] * speed * dt;
                    free_pos[2] += right[2] * speed * dt;
                }
                if (GetAsyncKeyState(b'A' as i32) as u16 & 0x8000) != 0 {
                    free_pos[0] -= right[0] * speed * dt;
                    free_pos[2] -= right[2] * speed * dt;
                }
                if (GetAsyncKeyState(VK_LEFT.0 as i32) as u16 & 0x8000) != 0 {
                    free_yaw -= 1.6 * dt;
                }
                if (GetAsyncKeyState(VK_RIGHT.0 as i32) as u16 & 0x8000) != 0 {
                    free_yaw += 1.6 * dt;
                }
                if (GetAsyncKeyState(VK_UP.0 as i32) as u16 & 0x8000) != 0 {
                    free_pos[1] += speed * 0.5 * dt;
                }
                if (GetAsyncKeyState(VK_DOWN.0 as i32) as u16 & 0x8000) != 0 {
                    free_pos[1] = (free_pos[1] - speed * 0.5 * dt).max(0.8);
                }
            }

            let hero_model = Mat4::translation(0.0, 0.0, 0.0).mul(&Mat4::rotate_y(2.42));
            let (cam_pos, cam_target) = if cfg.benchmark_run {
                let p = (t_sec / cfg.benchmark_secs.max(5.0)).clamp(0.0, 1.0);
                if p < 0.33 {
                    let s = p / 0.33;
                    let orbit_yaw = -1.45 + s * 0.45;
                    let radius = 5.45;
                    ([orbit_yaw.cos() * radius, 1.32 + s * 0.35, orbit_yaw.sin() * radius], [-0.55, 0.75, 2.0])
                } else if p < 0.67 {
                    let s = (p - 0.33) / 0.34;
                    let z = -5.0 + s * 105.0;
                    ([-1.5 + (s * 4.0).sin() * 2.5, 2.25, z], [-2.5, 4.0, z + 28.0])
                } else {
                    let s = (p - 0.67) / 0.33;
                    ([14.0 - s * 16.0, 10.5 - s * 5.0, 95.0 - s * 80.0], [-4.0, 2.5, 18.0 - s * 15.0])
                }
            } else {
                match wstate.cam_mode {
                    0 => {
                        // Showroom Front 3/4 Orbit facing +Z golden sunset, modular towers & wet asphalt puddles
                        let orbit_yaw = -1.28 + (t_sec * 0.14).sin() * 0.14;
                        let radius = 5.45;
                        let eye = [orbit_yaw.cos() * radius, 1.32, orbit_yaw.sin() * radius];
                        (eye, [-0.55, 0.65, 1.50])
                    }
                    1 => {
                        // Low dynamic chase angle
                        ([-0.4, 1.28, -5.8], [0.2, 0.68, 8.0])
                    }
                    _ => {
                        let tgt = [free_pos[0] + free_yaw.sin() * 8.0, free_pos[1] - 0.8, free_pos[2] + free_yaw.cos() * 8.0];
                        (free_pos, tgt)
                    }
                }
            };

            let halton = |mut idx: u32, base: u32| -> f32 {
                let mut f = 1.0f32;
                let mut r = 0.0f32;
                while idx > 0 {
                    f /= base as f32;
                    r += f * (idx % base) as f32;
                    idx /= base;
                }
                r
            };
            let j_idx = (frame_count % 8) + 1;
            let jit_x = (halton(j_idx, 2) - 0.5) * (2.0 / render_w as f32);
            let jit_y = (halton(j_idx, 3) - 0.5) * (2.0 / render_h as f32);

            let view = Mat4::look_at_rh(cam_pos, cam_target, [0.0, 1.0, 0.0]);
            let proj_unjit = Mat4::perspective_rh(46.0f32.to_radians(), render_w as f32 / render_h as f32, 0.1, 600.0);
            let mut proj_jit = proj_unjit;
            proj_jit.0[8] += jit_x;
            proj_jit.0[9] += jit_y;

            let vp_unjit = proj_unjit.mul(&view);
            let vp_jit = proj_jit.mul(&view);
            let inv_vp_unjit = vp_unjit.inverse();
            let prev_vp = prev_vp_unjittered.unwrap_or(vp_unjit);

            let sun_dir = norm3([0.28, 0.16, 0.94]);
            let proj_pt_px = |p: [f32; 3]| -> Option<[f32; 2]> {
                let m = &vp_unjit.0;
                let cx = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
                let cy = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
                let cw = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
                if cw.abs() < 1e-4 {
                    return None;
                }
                let u = (cx / cw) * 0.5 + 0.5;
                let v = 0.5 - (cy / cw) * 0.5;
                Some([u * render_w as f32, v * render_h as f32])
            };
            let sun_screen_dir = match (
                proj_pt_px(cam_target),
                proj_pt_px([cam_target[0] + sun_dir[0] * 2.0, cam_target[1] + sun_dir[1] * 2.0, cam_target[2] + sun_dir[2] * 2.0]),
            ) {
                (Some(a), Some(b)) => [b[0] - a[0], b[1] - a[1]],
                _ => [0.35, -0.45],
            };

            // Build scene instance list (Hero Car + 5 Traffic Cars + Environment)
            let mut traffic_mats = Vec::with_capacity(5);
            let mut cur_traffic_xfs = Vec::with_capacity(5);
            let traffic_specs = [
                (-5.2f32, 34.0f32, 0.0f32, 11.0f32, [0.85f32, 0.22, 0.18, 0.0]),
                (5.4f32, 19.0f32, std::f32::consts::PI, -14.0f32, [0.18, 0.55, 0.88, 0.0]),
                (-5.6f32, -26.0f32, 0.0f32, 13.0f32, [0.92, 0.72, 0.22, 0.0]),
                (5.2f32, 62.0f32, std::f32::consts::PI, -12.0f32, [0.22, 0.78, 0.52, 0.0]),
                (-5.4f32, 86.0f32, 0.0f32, 10.0f32, [0.82, 0.82, 0.86, 0.0]),
            ];
            for (lx, z0, yaw, spd, col) in traffic_specs {
                let z = ((z0 + t_sec * spd + 110.0).rem_euclid(220.0)) - 110.0;
                // Keep traffic lane far clear of hero car lane (-3..+3)
                let xf = Mat4::translation(lx, 0.0, z).mul(&Mat4::rotate_y(yaw));
                cur_traffic_xfs.push(xf);
                traffic_mats.push((xf, col));
            }
            let prev_t_xfs = prev_traffic_mats.as_deref().unwrap_or(&cur_traffic_xfs);

            let cmd_alloc = &cmd_allocs[slot];
            let cmd = &cmd_lists[slot];
            cmd_alloc.Reset().unwrap();
            cmd.Reset(cmd_alloc, None).unwrap();

            // Synchronize GPU compute -> Direct before copying freshly decoded 4K tiles
            if !taken.is_empty() {
                if wstate.use_gpu && compute_fence > 0 {
                    ctx.direct_wait_for_compute(compute_fence);
                }
                let src_res = if wstate.use_gpu { gpu_dec.out_buffer(slot) } else { &cpu_upload_buf };
                if wstate.use_gpu {
                    cmd_barriers(cmd, vec![transition_barrier(src_res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                }
                let mut touched = vec![false; pbr_textures.len()];
                for r in &taken {
                    touched[r.tex_idx] = true;
                }
                let to_copy_dst: Vec<_> = pbr_textures
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| touched[*i])
                    .map(|(_, res)| transition_barrier(res, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST))
                    .collect();
                if !to_copy_dst.is_empty() {
                    cmd_barriers(cmd, to_copy_dst);
                }

                for (bi, r) in taken.iter().enumerate() {
                    let tex_meta = &pack.textures[r.tex_idx];
                    let row_pitch = (tex_meta.width / 4) * (tex_meta.format.block_bytes() as u32);
                    let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(pbr_textures[r.tex_idx].clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                    };
                    let mut src = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(src_res.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: (bi * TILE_BC16_BYTES) as u64,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                    Format: dxgi_format_for(tex_meta.format),
                                    Width: tex_meta.width,
                                    Height: tex_meta.strip_height,
                                    Depth: 1,
                                    RowPitch: row_pitch,
                                },
                            },
                        },
                    };
                    cmd.CopyTextureRegion(&dst, 0, (r.strip_idx as u32) * tex_meta.strip_height, 0, &src, None);
                    let _ = ManuallyDrop::take(&mut dst.pResource);
                    let _ = ManuallyDrop::take(&mut src.pResource);
                }

                let mut to_srv: Vec<_> = pbr_textures
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| touched[*i])
                    .map(|(_, res)| transition_barrier(res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE))
                    .collect();
                if wstate.use_gpu {
                    to_srv.push(transition_barrier(src_res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON));
                }
                if !to_srv.is_empty() {
                    cmd_barriers(cmd, to_srv);
                }
                residency.mark_resident(&taken);
            }

            // Apply any pending SFS ResourceMinLODClamp SRV updates while GPU is idle (wait_direct_cpu already completed)
            for (ti, s) in residency.tex.iter_mut().enumerate() {
                if s.srv_dirty {
                    let mut sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: dxgi_format_for(pack.textures[ti].format),
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        ..Default::default()
                    };
                    sd.Anonymous.Texture2D = D3D12_TEX2D_SRV {
                        MipLevels: pack.textures[ti].mip_levels,
                        ResourceMinLODClamp: s.clamped_lod,
                        ..Default::default()
                    };
                    ctx.device.CreateShaderResourceView(&pbr_textures[ti], Some(&sd), cpu_at(ti));
                    ctx.device.CreateShaderResourceView(&pbr_textures[ti], Some(&sd), cpu_at(SRV_TABLE_STRIDE + ti));
                    s.srv_dirty = false;
                }
            }

            // Build TLAS BEFORE OMSetRenderTargets (with textured surface metadata per instance)
            let asphalt_model = Mat4::translation(0.0, 0.0, 0.0).mul(&Mat4::scale(24.0, 1.0, 320.0));
            let prom_l_model = Mat4::translation(-18.0, 0.08, 0.0).mul(&Mat4::scale(12.0, 1.0, 320.0));
            let prom_r_model = Mat4::translation(18.0, 0.08, 0.0).mul(&Mat4::scale(12.0, 1.0, 320.0));
            let seawall_model = Mat4::translation(24.6, 0.55, 0.0).mul(&Mat4::scale(1.2, 1.1, 320.0));
            let ocean_model = Mat4::translation(145.0, -0.45, 0.0).mul(&Mat4::scale(240.0, 1.0, 400.0));

            if let Some(ref rt) = dxr {
                let mut rt_insts = Vec::with_capacity(32);
                rt_insts.push((
                    0usize,
                    hero_model.to_3x4(),
                    RtInstMeta {
                        base_color: [0.80, 0.08, 0.05, 0.12],
                        geom: [0, 0, u32::MAX, 0],
                        uv_params: [1.0, 0.0, 0.0, 0.0],
                    },
                ));
                for (m, col) in &parked_car_mats {
                    rt_insts.push((
                        0usize,
                        m.to_3x4(),
                        RtInstMeta {
                            base_color: *col,
                            geom: [0, 0, u32::MAX, 0],
                            uv_params: [1.0, 0.0, 0.0, 0.0],
                        },
                    ));
                }
                for (m, col) in &traffic_mats {
                    rt_insts.push((
                        0usize,
                        m.to_3x4(),
                        RtInstMeta {
                            base_color: *col,
                            geom: [0, 0, u32::MAX, 0],
                            uv_params: [1.0, 0.0, 0.0, 0.0],
                        },
                    ));
                }
                rt_insts.push((
                    2usize,
                    asphalt_model.to_3x4(),
                    RtInstMeta {
                        base_color: [0.95, 0.95, 0.95, 0.0],
                        geom: [plane_base_vert, plane_idx_offset, pack.asphalt_tex[0], 0],
                        uv_params: [22.0, 0.0, 0.0, 0.0],
                    },
                ));
                rt_insts.push((
                    2usize,
                    prom_l_model.to_3x4(),
                    RtInstMeta {
                        base_color: [1.0, 0.96, 0.94, 0.0],
                        geom: [plane_base_vert, plane_idx_offset, pack.promenade_tex[0], 0],
                        uv_params: [28.0, 0.0, 0.0, 0.0],
                    },
                ));
                rt_insts.push((
                    2usize,
                    prom_r_model.to_3x4(),
                    RtInstMeta {
                        base_color: [1.0, 0.96, 0.94, 0.0],
                        geom: [plane_base_vert, plane_idx_offset, pack.promenade_tex[0], 0],
                        uv_params: [28.0, 0.0, 0.0, 0.0],
                    },
                ));
                rt_insts.push((
                    1usize,
                    seawall_model.to_3x4(),
                    RtInstMeta {
                        base_color: [0.92, 0.88, 0.84, 0.0],
                        geom: [box_base_vert, box_idx_offset, pack.concrete_tex[0], 0],
                        uv_params: [12.0, 0.0, 0.0, 0.0],
                    },
                ));
                for (pos, scale, col) in &towers {
                    let tm = Mat4::translation(pos[0], pos[1], pos[2]).mul(&Mat4::scale(scale[0], scale[1], scale[2]));
                    rt_insts.push((
                        1usize,
                        tm.to_3x4(),
                        RtInstMeta {
                            base_color: [col[0], col[1], col[2], 0.08],
                            geom: [box_base_vert, box_idx_offset, pack.concrete_tex[0], 0],
                            uv_params: [4.0, 0.0, 0.0, 0.0],
                        },
                    ));
                }
                rt.build_tlas(&mut ctx, slot, &rt_insts);
            }

            // Skin 500 promenade pedestrians on GPU Compute before G-Buffer pass
            if let Some(ref mut crowd) = crowd_renderer {
                crowd.update_and_skin(cmd, slot, t_sec);
            }

            // Clear GPU SFS feedback buffer & transition to UAV before PBR pass
            cmd.CopyBufferRegion(&sfs_feedback_default, 0, &sfs_feedback_init, 0, sfs_buf_bytes);
            cmd_barriers(
                cmd,
                vec![transition_barrier(
                    &sfs_feedback_default,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )],
            );

            // Begin 4-MRT G-Buffer pass (DirectHDR + RawRTGI + NormalDepth + MotionVec)
            post.begin_gbuffer(cmd, dsv_handle);
            let gpu_base = main_srv_heap.GetGPUDescriptorHandleForHeapStart();
            let table_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_base.ptr + ((slot * SRV_TABLE_STRIDE) * srv_inc as usize) as u64,
            };
            let env_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_base.ptr + ((slot * SRV_TABLE_STRIDE + 32) * srv_inc as usize) as u64,
            };
            cmd.SetDescriptorHeaps(&[Some(main_srv_heap.clone())]);

            // 1. Sunset HDRI Sky Dome
            cmd.SetGraphicsRootSignature(&sky_root_sig);
            cmd.SetPipelineState(&sky_pso);
            let sky_cb = SkyConstants {
                inv_view_proj: inv_vp_unjit.0,
                prev_view_proj: prev_vp.0,
                camera_pos: cam_pos,
                time: t_sec,
            };
            cmd.SetGraphicsRoot32BitConstants(0, (size_of::<SkyConstants>() / 4) as u32, &sky_cb as *const SkyConstants as *const c_void, 0);
            cmd.SetGraphicsRootDescriptorTable(1, env_gpu);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.DrawInstanced(3, 1, 0, 0);

            // 2. PBR Geometry
            let flags_bits = (if wstate.rt_enabled { 1u32 } else { 0u32 })
                | (if wstate.rtgi_enabled { 2u32 } else { 0u32 })
                | (if wstate.xray_enabled { 4u32 } else { 0u32 });
            let flags_f = flags_bits as f32;

            let frame_cb = AaaFrameCB {
                vp_jittered: vp_jit.0,
                vp_unjittered: vp_unjit.0,
                prev_vp_unjittered: prev_vp.0,
                camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], flags_f],
                sun_dir_time: [sun_dir[0], sun_dir[1], sun_dir[2], t_sec],
                jitter_res: [jit_x, jit_y, render_w as f32, render_h as f32],
            };
            upload_slice(&frame_cb_upload[slot], std::slice::from_ref(&frame_cb));

            cmd.SetGraphicsRootSignature(&pbr_root_sig);
            cmd.SetPipelineState(&pbr_pso);
            cmd.SetGraphicsRootDescriptorTable(1, table_gpu);
            cmd.SetGraphicsRootConstantBufferView(2, frame_cb_upload[slot].GetGPUVirtualAddress());
            cmd.SetGraphicsRootUnorderedAccessView(3, sfs_feedback_default.GetGPUVirtualAddress());
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.IASetVertexBuffers(0, Some(&[vb_view]));
            cmd.IASetIndexBuffer(Some(&ib_view));

            let draw_car = |model: &Mat4, prev_model: &Mat4| {
                for sm in &pack.car.submeshes {
                    draw_car_submesh(cmd, model, prev_model, t_sec, sm, &residency, &pack.textures);
                }
            };

            // Hero Car + 6 parked cars + 5 animated CarConcept traffic cars
            draw_car(&hero_model, &hero_model);
            for (pm, _) in &parked_car_mats {
                draw_car(pm, pm);
            }
            for (idx, (tm, _)) in traffic_mats.iter().enumerate() {
                draw_car(tm, &prev_t_xfs[idx]);
            }

            // Wet 4-lane Asphalt Road
            draw_static_mesh(
                cmd,
                &asphalt_model,
                plane_base_vert,
                plane_idx_offset,
                6,
                t_sec,
                [0.95, 0.95, 0.95, 1.0],
                [0.0, 0.85, 22.0, 3.0], // wetAsphalt mode=3, 22x tiled
                pack.asphalt_tex,
                &residency,
                &pack.textures,
            );
            // Brick Promenades (Left + Right)
            draw_static_mesh(
                cmd,
                &prom_l_model,
                plane_base_vert,
                plane_idx_offset,
                6,
                t_sec,
                [1.0, 0.96, 0.94, 1.0],
                [0.0, 0.85, 28.0, 2.0], // PolyHaven mode=2
                pack.promenade_tex,
                &residency,
                &pack.textures,
            );
            draw_static_mesh(
                cmd,
                &prom_r_model,
                plane_base_vert,
                plane_idx_offset,
                6,
                t_sec,
                [1.0, 0.96, 0.94, 1.0],
                [0.0, 0.85, 28.0, 2.0],
                pack.promenade_tex,
                &residency,
                &pack.textures,
            );
            // Glossy Sunset Ocean Plane
            draw_static_mesh(
                cmd,
                &ocean_model,
                plane_base_vert,
                plane_idx_offset,
                6,
                t_sec,
                [0.05, 0.14, 0.22, 1.0],
                [0.15, 0.05, 1.0, 0.0],
                [u32::MAX, u32::MAX, u32::MAX],
                &residency,
                &pack.textures,
            );

            // Stage 3: 12M+ Modular Mesh-Shader City (Recessed Glass Towers, Balconies, Palms, Streetlamps, Road Markings, Seawall, HLOD Skyline)
            if let Some(ref city) = meshlet_city {
                let hiz_table_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                    ptr: gpu_base.ptr + (AAA_SRV_HIZ_BASE * srv_inc as usize) as u64,
                };
                let concrete_states = [
                    residency.state_for_tex(pack.concrete_tex[0]),
                    residency.state_for_tex(pack.concrete_tex[1]),
                    pack.textures[pack.concrete_tex[0] as usize].width as f32,
                ];
                city.draw(
                    cmd,
                    slot,
                    &vp_jit.0,
                    &vp_unjit.0,
                    &prev_vp.0,
                    cam_pos,
                    flags_bits,
                    sun_dir,
                    t_sec,
                    frame_count > 1,
                    hiz_pyramid.width,
                    hiz_pyramid.height,
                    concrete_states,
                    pack.concrete_tex,
                    table_gpu,
                    hiz_table_gpu,
                    sfs_feedback_default.GetGPUVirtualAddress(),
                );
            } else {
                draw_static_mesh(
                    cmd,
                    &seawall_model,
                    box_base_vert,
                    box_idx_offset,
                    36,
                    t_sec,
                    [0.92, 0.88, 0.84, 1.0],
                    [0.0, 0.88, 12.0, 2.0],
                    pack.concrete_tex,
                    &residency,
                    &pack.textures,
                );
                for (pos, scale, col) in &towers {
                    let tm = Mat4::translation(pos[0], pos[1], pos[2]).mul(&Mat4::scale(scale[0], scale[1], scale[2]));
                    draw_static_mesh(
                        cmd,
                        &tm,
                        box_base_vert,
                        box_idx_offset,
                        36,
                        t_sec,
                        [col[0], col[1], col[2], 1.0],
                        [0.05, 0.72, 4.0, 2.0],
                        pack.concrete_tex,
                        &residency,
                        &pack.textures,
                    );
                }
            }

            // Stage 3: 500 GPU-Skinned Pedestrians walking along coastal promenades
            if let Some(ref crowd) = crowd_renderer {
                crowd.draw(
                    cmd,
                    slot,
                    &vp_jit.0,
                    &vp_unjit.0,
                    &prev_vp.0,
                    cam_pos,
                    flags_bits,
                    sun_dir,
                    t_sec,
                    table_gpu,
                );
            }

            // Build 6-Mip Conservative Hi-Z Depth Pyramid for next frame's Mesh-Shader occlusion culling
            hiz_pyramid.build(cmd, &_depth_tex, &main_srv_heap);

            // Resolve GPU SFS feedback buffer -> readback[slot]
            cmd_barriers(
                cmd,
                vec![transition_barrier(
                    &sfs_feedback_default,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )],
            );
            cmd.CopyBufferRegion(&sfs_feedback_readback[slot], 0, &sfs_feedback_default, 0, sfs_buf_bytes);
            cmd_barriers(
                cmd,
                vec![transition_barrier(
                    &sfs_feedback_default,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )],
            );

            // Backbuffer transition -> RT + Run RTGI Denoiser + TAA + AutoExp + Bloom + AgX/CAS Composite
            let bb_idx = swap_chain.GetCurrentBackBufferIndex();
            let bb = &backbuffers[bb_idx as usize];
            let bb_rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: bb_base.ptr + (bb_idx * rtv_inc) as usize,
            };
            cmd_barriers(cmd, vec![transition_barrier(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)]);

            let hud_cb = [
                if wstate.hud_visible { 1.0f32 } else { 0.0f32 },
                fps_to_norm(wstate.fps_cap),
                wstate.fps_cap as f32,
                live_fps,
                sim_to_norm(wstate.sim_speed),
                wstate.sim_speed,
                wstate.drag_slider as f32,
                if wstate.vsync_enabled { 1.0f32 } else { 0.0f32 },
            ];
            post.run_post_and_composite(
                cmd,
                &main_srv_heap,
                bb_rtv,
                SWAP_W,
                SWAP_H,
                wstate.rtgi_enabled,
                frame_count,
                sun_screen_dir,
                t_sec,
                hud_cb,
            );

            prev_vp_unjittered = Some(vp_unjit);
            prev_traffic_mats = Some(cur_traffic_xfs);

            let rec_frame_idx: Option<u32> = if cfg.record_dir.is_some() && frame_count >= 24 {
                Some(frame_count - 24)
            } else {
                None
            };
            let take_shot = (cfg.screenshot_path.is_some() && frame_count + 1 >= cfg.screenshot_frames)
                || rec_frame_idx.is_some();
            if take_shot {
                cmd_barriers(cmd, vec![transition_barrier(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(ss_readback.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: SWAP_W,
                                Height: SWAP_H,
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
                cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                let _ = ManuallyDrop::take(&mut dst.pResource);
                let _ = ManuallyDrop::take(&mut src.pResource);
                cmd_barriers(cmd, vec![transition_barrier(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT)]);
            } else {
                cmd_barriers(cmd, vec![transition_barrier(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)]);
            }

            cmd.Close().unwrap();
            let lists = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&lists);
            let sync_interval = if wstate.vsync_enabled { 1u32 } else { 0u32 };
            if swap_chain.Present(sync_interval, DXGI_PRESENT(0)).is_err() {
                break;
            }

            ctx.signal_and_wait();
            frame_fences[slot] = 0;
            if wstate.use_gpu && !taken.is_empty() {
                last_decode_ms = gpu_dec.read_slot_kernel_ms(&ctx, slot);
                if last_decode_ms > 1e-4 {
                    let gbps = ((taken.len() * TILE_BC16_BYTES) as f64 / 1e9) / (last_decode_ms / 1000.0);
                    bench_peak_gbps = bench_peak_gbps.max(gbps);
                }
            }
            if let Some(ref city) = meshlet_city {
                last_cull_stats = city.read_cull_stats(slot);
                bench_sum_vis_tris += last_cull_stats.vis_tris as u64;
                bench_sum_hiz_tris += last_cull_stats.hiz_occluded_tris as u64;
            }

            frame_count += 1;
            fps_accum += 1;

            // Readback GPU Sampler Feedback & update active SFS LRU residency/clamping for next frame
            {
                let mut fb_ptr: *mut c_void = std::ptr::null_mut();
                let fb_rng = D3D12_RANGE { Begin: 0, End: sfs_buf_bytes as usize };
                sfs_feedback_readback[slot].Map(0, Some(&fb_rng), Some(&mut fb_ptr)).unwrap();
                let fb_bytes = std::slice::from_raw_parts(fb_ptr as *const u8, sfs_buf_bytes as usize);
                residency.apply_sfs_feedback_and_schedule(fb_bytes, frame_count as u64, wstate.sfs_enabled);
                sfs_feedback_readback[slot].Unmap(0, None);
            }

            if take_shot {
                if let Some(ri) = rec_frame_idx {
                    let dir = cfg.record_dir.as_ref().unwrap();
                    let mut ptr: *mut c_void = std::ptr::null_mut();
                    let rng = D3D12_RANGE { Begin: 0, End: (ss_row_pitch * SWAP_H) as usize };
                    ss_readback.Map(0, Some(&rng), Some(&mut ptr)).unwrap();
                    let raw = std::slice::from_raw_parts(ptr as *const u8, (ss_row_pitch * SWAP_H) as usize);
                    let dw = (SWAP_W / 2) as usize;
                    let dh = (SWAP_H / 2) as usize;
                    let pitch = ss_row_pitch as usize;
                    let mut down = vec![0u8; dw * dh * 4];
                    for y in 0..dh {
                        let r0 = (y * 2) * pitch;
                        let r1 = (y * 2 + 1) * pitch;
                        for x in 0..dw {
                            let c0 = x * 2 * 4;
                            let c1 = (x * 2 + 1) * 4;
                            let dst_o = (y * dw + x) * 4;
                            for ch in 0..4 {
                                let sum = raw[r0 + c0 + ch] as u32
                                    + raw[r0 + c1 + ch] as u32
                                    + raw[r1 + c0 + ch] as u32
                                    + raw[r1 + c1 + ch] as u32;
                                down[dst_o + ch] = (sum >> 2) as u8;
                            }
                        }
                    }
                    ss_readback.Unmap(0, None);
                    let out_path = dir.join(format!("frame_{:04}.png", ri));
                    let file = File::create(&out_path).expect("create record frame");
                    let mut enc = png::Encoder::new(BufWriter::new(file), dw as u32, dh as u32);
                    enc.set_color(png::ColorType::Rgba);
                    enc.set_depth(png::BitDepth::Eight);
                    enc.set_compression(png::Compression::Fast);
                    let mut writer = enc.write_header().unwrap();
                    writer.write_image_data(&down).unwrap();
                    if ri + 1 >= cfg.record_frames {
                        println!("[aaa-record] Saved {} frames to {}", cfg.record_frames, dir.display());
                        break;
                    }
                } else {
                    let out_path = cfg.screenshot_path.as_ref().unwrap();
                    let mut ptr: *mut c_void = std::ptr::null_mut();
                    let rng = D3D12_RANGE { Begin: 0, End: (ss_row_pitch * SWAP_H) as usize };
                    ss_readback.Map(0, Some(&rng), Some(&mut ptr)).unwrap();
                    let raw = std::slice::from_raw_parts(ptr as *const u8, (ss_row_pitch * SWAP_H) as usize);
                    let file = File::create(out_path).expect("create screenshot");
                    let mut enc = png::Encoder::new(BufWriter::new(file), SWAP_W, SWAP_H);
                    enc.set_color(png::ColorType::Rgba);
                    enc.set_depth(png::BitDepth::Eight);
                    let mut writer = enc.write_header().unwrap();
                    writer.write_image_data(raw).unwrap();
                    ss_readback.Unmap(0, None);
                    let cur_mib = residency.current_vram_bytes() as f64 / (1024.0 * 1024.0);
                    let tot_mib = residency.total_vram_bytes as f64 / (1024.0 * 1024.0);
                    println!(
                        "[aaa-screenshot] Saved {} ({}x{}, mip0={}/{} tex [{}/{} tiles], VRAM={:.1}/{:.1} MiB, saved={:.1} MiB, evict={}, xray={})",
                        out_path.display(),
                        SWAP_W,
                        SWAP_H,
                        residency.num_mip0_full(),
                        pack.textures.len(),
                        residency.total_mip0_resident(),
                        residency.total_mip0_tiles(),
                        cur_mib,
                        tot_mib,
                        (tot_mib - cur_mib).max(0.0),
                        residency.total_evictions,
                        wstate.xray_enabled
                    );
                    break;
                }
            }

            if (now - last_title).as_secs_f64() >= 0.35 {
                let secs = (now - last_title).as_secs_f64();
                let fps_val = fps_accum as f64 / secs;
                live_fps = fps_val as f32;
                fps_accum = 0;
                last_title = now;
                let mode_str = if wstate.use_gpu { "GPU Async" } else { "CPU 1T" };
                let cam_str = match wstate.cam_mode {
                    0 => "Showroom",
                    1 => "Chase",
                    _ => "FreeCam",
                };
                let cap_str = if wstate.fps_cap > 0 {
                    format!("{}", wstate.fps_cap)
                } else if wstate.vsync_enabled {
                    "VSync".to_string()
                } else {
                    "MAX".to_string()
                };
                let cur_mib = residency.current_vram_bytes() as f64 / (1024.0 * 1024.0);
                let tot_mib = residency.total_vram_bytes as f64 / (1024.0 * 1024.0);
                let saved_mib = (tot_mib - cur_mib).max(0.0);
                let city_str = if let Some(ref city) = meshlet_city {
                    format!(
                        "HiZ:{:.2}M/{:.1}M tris (-{:.2}M occ) | ",
                        last_cull_stats.vis_tris as f64 / 1e6,
                        city.total_tris as f64 / 1e6,
                        last_cull_stats.hiz_occluded_tris as f64 / 1e6
                    )
                } else {
                    String::new()
                };
                let title = format!(
                    "BCPack AAA v0.8.0 | {} | {:.1} FPS (Cap:{} Sim:{:.2}x) | {}NPC:{} | SFS[F]:{} {:.0}/{:.0}MiB (-{:.0}MiB) | dec {:.2}ms ({:.1}MiB) | DXR:{} RTGI:{} XRay:{} | {}",
                    mode_str,
                    fps_val,
                    cap_str,
                    wstate.sim_speed,
                    city_str,
                    if crowd_renderer.is_some() { CROWD_NPCS } else { 0 },
                    if wstate.sfs_enabled { "ON" } else { "OFF" },
                    cur_mib,
                    tot_mib,
                    saved_mib,
                    last_decode_ms,
                    streamed_mb,
                    if wstate.rt_enabled { "ON" } else { "OFF" },
                    if wstate.rtgi_enabled { "ON" } else { "OFF" },
                    if wstate.xray_enabled { "ON" } else { "OFF" },
                    cam_str,
                );
                let wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
                let _ = SetWindowTextW(hwnd, PCWSTR(wide.as_ptr()));
            }

            if cfg.benchmark_run && cfg.record_dir.is_none() && t_sec >= cfg.benchmark_secs {
                let cur_mib = residency.current_vram_bytes() as f64 / (1024.0 * 1024.0);
                let tot_mib = residency.total_vram_bytes as f64 / (1024.0 * 1024.0);
                let saved_mib = (tot_mib - cur_mib).max(0.0);
                let n = bench_frame_ms.len().max(1);
                bench_frame_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let avg_ms: f64 = bench_frame_ms.iter().sum::<f64>() / (n as f64);
                let p99_ms = bench_frame_ms[((n as f64 * 0.99) as usize).min(n - 1)];
                let p999_ms = bench_frame_ms[((n as f64 * 0.999) as usize).min(n - 1)];
                let pre_tris = meshlet_city.as_ref().map(|c| c.total_tris).unwrap_or(0);
                let fc = frame_count.max(1) as f64;
                println!("=== BCPack AAA Benchmark v0.8.0 Summary ({:.1}s sim, {}x{}) ===", t_sec, render_w, render_h);
                println!("  Frames: {} | Average FPS: {:.2} ({:.2} ms) | 1% Low: {:.2} FPS ({:.2} ms) | 0.1% Low: {:.2} FPS ({:.2} ms)",
                    frame_count, 1000.0 / avg_ms.max(0.01), avg_ms, 1000.0 / p99_ms.max(0.01), p99_ms, 1000.0 / p999_ms.max(0.01), p999_ms);
                println!("  GPU BCPack Peak Throughput (Live 64K BC7/BC5 Tiles): {:.2} GB/s", bench_peak_gbps);
                println!("  SFS VRAM Residency: {:.1}/{:.1} MiB (Saved {:.1} MiB, Evictions={}, FullMip0={}/{})",
                    cur_mib, tot_mib, saved_mib, residency.total_evictions, residency.num_mip0_full(), pack.textures.len());
                println!("  Mesh Shaders + Hi-Z: Pre-Cull {:.2}M tris | Avg Visible {:.2}M tris | Avg Hi-Z Occluded {:.2}M tris",
                    pre_tris as f64 / 1e6, (bench_sum_vis_tris as f64 / fc) / 1e6, (bench_sum_hiz_tris as f64 / fc) / 1e6);
                println!("  Live Crowd: {} GPU-Skinned NPCs ({} verts/frame in Compute + DXR/RTGI/TAA/AgX)",
                    if crowd_renderer.is_some() { CROWD_NPCS } else { 0 },
                    crowd_renderer.as_ref().map(|c| c.verts_total()).unwrap_or(0));
                break;
            }

            if wstate.fps_cap > 0 && cfg.record_dir.is_none() {
                let target = std::time::Duration::from_secs_f64(1.0 / (wstate.fps_cap as f64));
                while now.elapsed() < target {
                    let rem = target.saturating_sub(now.elapsed());
                    if rem > std::time::Duration::from_millis(2) {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }

            if let Some(target_n) = cfg.measure_frames {
                if frame_count >= target_n {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let cur_mib = residency.current_vram_bytes() as f64 / (1024.0 * 1024.0);
                    let tot_mib = residency.total_vram_bytes as f64 / (1024.0 * 1024.0);
                    println!(
                        "[aaa-measure] {} frames in {:.3}s = {:.2} FPS | res={}x{} | SFS VRAM={:.1}/{:.1} MiB (saved {:.1} MiB, evict={}) | mip0={}/{} tiles | last_decode={:.3} ms",
                        frame_count,
                        elapsed,
                        frame_count as f64 / elapsed,
                        render_w,
                        render_h,
                        cur_mib,
                        tot_mib,
                        (tot_mib - cur_mib).max(0.0),
                        residency.total_evictions,
                        residency.total_mip0_resident(),
                        residency.total_mip0_tiles(),
                        last_decode_ms
                    );
                    break;
                }
            }
        }

        ctx.signal_and_wait();
        ctx.signal_and_wait_compute();
        if cfg.enable_debug {
            dump_d3d12_info_queue(&ctx.device);
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn draw_car_submesh(
    cmd: &ID3D12GraphicsCommandList,
    model: &Mat4,
    prev_model: &Mat4,
    t_sec: f32,
    sm: &PbrSubmesh,
    residency: &SfsResidencyManager,
    textures: &[BcpackTexture2D],
) {
    let a_idx = sm.albedo_tex.map(|i| i as u32).unwrap_or(u32::MAX);
    let n_idx = sm.normal_tex.map(|i| i as u32).unwrap_or(u32::MAX);
    let o_idx = sm.orm_tex.map(|i| i as u32).unwrap_or(u32::MAX);
    let e_idx = sm.emissive_tex.map(|i| i as u32).unwrap_or(u32::MAX);
    let tex_dim = sm
        .albedo_tex
        .or(sm.normal_tex)
        .or(sm.orm_tex)
        .map(|i| textures[i].width as f32)
        .unwrap_or(1024.0);

    let cb = AaaDrawCB {
        model: model.0,
        prev_model: prev_model.0,
        base_color: sm.base_color_factor,
        mat_params: [
            sm.metallic_factor,
            sm.roughness_factor,
            1.0,
            if sm.is_glass { 1.0 } else { 0.0 },
        ],
        emissive_time: [sm.emissive_factor[0], sm.emissive_factor[1], sm.emissive_factor[2], t_sec],
        tex_indices: [a_idx, n_idx, o_idx, e_idx],
        tile_states: [
            residency.state_for_tex(a_idx),
            residency.state_for_tex(n_idx),
            residency.state_for_tex(o_idx),
            tex_dim,
        ],
    };
    cmd.SetGraphicsRoot32BitConstants(0, (size_of::<AaaDrawCB>() / 4) as u32, &cb as *const AaaDrawCB as *const c_void, 0);
    cmd.DrawIndexedInstanced(sm.index_count, 1, sm.index_offset, 0, 0);
}

#[allow(clippy::too_many_arguments)]
unsafe fn draw_static_mesh(
    cmd: &ID3D12GraphicsCommandList,
    model: &Mat4,
    base_vert: u32,
    idx_offset: u32,
    idx_count: u32,
    t_sec: f32,
    base_color: [f32; 4],
    mat_params: [f32; 4],
    tex3: [u32; 3],
    residency: &SfsResidencyManager,
    textures: &[BcpackTexture2D],
) {
    let tex_dim = if tex3[0] != u32::MAX {
        textures[tex3[0] as usize].width as f32
    } else {
        1024.0
    };
    let cb = AaaDrawCB {
        model: model.0,
        prev_model: model.0,
        base_color,
        mat_params,
        emissive_time: [0.0, 0.0, 0.0, t_sec],
        tex_indices: [tex3[0], tex3[1], tex3[2], u32::MAX],
        tile_states: [
            residency.state_for_tex(tex3[0]),
            residency.state_for_tex(tex3[1]),
            residency.state_for_tex(tex3[2]),
            tex_dim,
        ],
    };
    cmd.SetGraphicsRoot32BitConstants(0, (size_of::<AaaDrawCB>() / 4) as u32, &cb as *const AaaDrawCB as *const c_void, 0);
    cmd.DrawIndexedInstanced(idx_count, 1, idx_offset, base_vert as i32, 0);
}

struct StagedMipSlice {
    tex_idx: usize,
    mip_level: u32,
    format: DXGI_FORMAT,
    width: u32,
    height: u32,
    dst_y: u32,
    row_pitch: u32,
    offset: u64,
}

unsafe fn decode_and_upload_lower_mips(
    ctx: &mut GpuContext,
    gpu_dec: &mut Bc16GpuStreamer,
    textures: &[BcpackTexture2D],
    pbr_textures: &[ID3D12Resource],
) {
    let mut items: Vec<(usize, &BcpackMipChunk)> = Vec::new();
    for (ti, tex) in textures.iter().enumerate() {
        for mc in &tex.lower_mips {
            items.push((ti, mc));
        }
    }
    if items.is_empty() {
        return;
    }

    let mut staging: Vec<u8> = Vec::with_capacity(64 * 1024 * 1024);
    let mut slices: Vec<StagedMipSlice> = Vec::with_capacity(items.len() + textures.len() * 4);

    let push_slice = |staging: &mut Vec<u8>,
                      slices: &mut Vec<StagedMipSlice>,
                      tex_idx: usize,
                      mip_level: u32,
                      fmt: TexFormat,
                      w: u32,
                      h: u32,
                      dst_y: u32,
                      blk_sub: &[u8]| {
        let blocks_x = (w / 4).max(1) as usize;
        let blocks_y = (h / 4).max(1) as usize;
        let row_bytes = blocks_x * fmt.block_bytes();
        let aligned_pitch = row_bytes.div_ceil(256) * 256;
        let aligned_off = staging.len().div_ceil(512) * 512;
        staging.resize(aligned_off + aligned_pitch * blocks_y, 0);
        for row in 0..blocks_y {
            let src_start = row * row_bytes;
            let dst_start = aligned_off + row * aligned_pitch;
            staging[dst_start..dst_start + row_bytes]
                .copy_from_slice(&blk_sub[src_start..src_start + row_bytes]);
        }
        slices.push(StagedMipSlice {
            tex_idx,
            mip_level,
            format: dxgi_format_for(fmt),
            width: w,
            height: h,
            dst_y,
            row_pitch: aligned_pitch as u32,
            offset: aligned_off as u64,
        });
    };

    for chunk in items.chunks(MAX_BATCH_TILES) {
        let batch = pack_batch(chunk.iter().map(|(_, mc)| (mc.bcpack.as_slice(), mc.blocks * 2)));
        let (decoded, statuses, _kms, _tms) = gpu_dec.decode_batch_and_readback(ctx, &batch);
        let mut row_off = 0usize;
        for (ci, &(tex_idx, mc)) in chunk.iter().enumerate() {
            assert_eq!(
                statuses[ci].code, 0,
                "GPU lower-mip decode error tex={tex_idx} mip={}: {:?}",
                mc.mip_level, statuses[ci]
            );
            let fmt = textures[tex_idx].format;
            let bbytes = fmt.block_bytes();
            let chunk_bytes = &decoded[row_off * 8..(row_off * 8 + mc.blocks * bbytes)];
            row_off += mc.blocks * 2;

            if !mc.is_packed_tail {
                push_slice(
                    &mut staging,
                    &mut slices,
                    tex_idx,
                    mc.mip_level,
                    fmt,
                    mc.dim,
                    mc.strip_h,
                    mc.strip_y,
                    chunk_bytes,
                );
            } else {
                // Expand Packed Mip Tail (32x32 + 16x16 + 8x8 + 4x4) into individual DX12 subresources
                let mut cur_dim = mc.dim;
                let mut cur_mip = mc.mip_level;
                let mut src_off = 0usize;
                loop {
                    let blocks_side = (cur_dim / 4).max(1) as usize;
                    let sub_bytes = blocks_side * blocks_side * bbytes;
                    push_slice(
                        &mut staging,
                        &mut slices,
                        tex_idx,
                        cur_mip,
                        fmt,
                        cur_dim,
                        cur_dim,
                        0,
                        &chunk_bytes[src_off..src_off + sub_bytes],
                    );
                    src_off += sub_bytes;
                    if cur_dim <= 4 {
                        break;
                    }
                    cur_dim /= 2;
                    cur_mip += 1;
                }
            }
        }
    }

    let up_buf = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_UPLOAD,
        staging.len().max(512) as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    upload_slice(&up_buf, &staging);

    let mip_alloc: ID3D12CommandAllocator = ctx
        .device
        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        .unwrap();
    let mip_cmd: ID3D12GraphicsCommandList = ctx
        .device
        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &mip_alloc, None)
        .unwrap();

    let to_dst: Vec<_> = pbr_textures
        .iter()
        .map(|res| transition_barrier(res, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST))
        .collect();
    cmd_barriers(&mip_cmd, to_dst);

    for sl in &slices {
        let mut dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(pbr_textures[sl.tex_idx].clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: sl.mip_level,
            },
        };
        let mut src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(up_buf.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: sl.offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: sl.format,
                        Width: sl.width,
                        Height: sl.height,
                        Depth: 1,
                        RowPitch: sl.row_pitch,
                    },
                },
            },
        };
        mip_cmd.CopyTextureRegion(&dst, 0, sl.dst_y, 0, &src, None);
        let _ = ManuallyDrop::take(&mut dst.pResource);
        let _ = ManuallyDrop::take(&mut src.pResource);
    }

    let to_srv: Vec<_> = pbr_textures
        .iter()
        .map(|res| transition_barrier(res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE))
        .collect();
    cmd_barriers(&mip_cmd, to_srv);

    mip_cmd.Close().unwrap();
    let exec = [Some(mip_cmd.cast::<ID3D12CommandList>().unwrap())];
    ctx.queue.ExecuteCommandLists(&exec);
    ctx.signal_and_wait();
}
