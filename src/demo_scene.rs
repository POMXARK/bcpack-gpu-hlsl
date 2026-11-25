use crate::assets::{TextureAsset, NUM_BLOCKS, TEX_BC1_BYTES, TEX_H, TEX_W};
use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, drop_barriers, heap_props, pack_batch,
    transition_barrier, upload_slice, BcpackGpuDecoder, GpuContext,
};
use bcpack::bc1::{self, QuadTable, Window};
use rayon::prelude::*;
use std::ffi::c_void;
use std::fs::File;
use std::io::BufWriter;
use std::mem::{size_of, ManuallyDrop};
use std::path::PathBuf;
use std::time::Instant;
use windows::core::{w, Interface, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    VK_DOWN, VK_ESCAPE, VK_M, VK_SPACE, VK_UP,
};
use windows::Win32::UI::WindowsAndMessaging::*;

pub const SCENE_VS_DXIL: &[u8] = include_bytes!("../shaders/scene_vs.dxil");
pub const SCENE_PS_DXIL: &[u8] = include_bytes!("../shaders/scene_ps.dxil");

const WIN_W: u32 = 1280;
const WIN_H: u32 = 720;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
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
}

// Column-major 4x4 matrix helpers matching HLSL default column_major mul(M, v)
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

fn build_unit_cube() -> (Vec<Vertex>, Vec<u16>) {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +Z front
        ([0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]),
        // -Z back
        ([0.0, 0.0, -1.0], [[0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]]),
        // +X right
        ([1.0, 0.0, 0.0], [[0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5]]),
        // -X left
        ([-1.0, 0.0, 0.0], [[-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5]]),
        // +Y top
        ([0.0, 1.0, 0.0], [[-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]),
        // -Y bottom
        ([0.0, -1.0, 0.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5]]),
    ];
    let uvs = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
    let mut verts = Vec::with_capacity(24);
    let mut inds = Vec::with_capacity(36);
    for (fi, (n, pts)) in faces.iter().enumerate() {
        let base = (fi * 4) as u16;
        for i in 0..4 {
            verts.push(Vertex {
                pos: pts[i],
                normal: *n,
                uv: uvs[i],
            });
        }
        inds.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, inds)
}

struct SceneBox {
    pos: [f32; 3],
    rot_y: f32,
    scale: [f32; 3],
    tex: u32,
    emissive: f32,
    uv_scale: f32,
    spin: bool,
}

fn build_vice_city_scene() -> Vec<SceneBox> {
    let mut boxes = Vec::new();
    // Main asphalt boulevard
    boxes.push(SceneBox { pos: [0.0, -0.1, 0.0], rot_y: 0.0, scale: [10.0, 0.2, 70.0], tex: 0, emissive: 0.05, uv_scale: 3.0, spin: false });
    // Left & right terracotta sidewalks
    boxes.push(SceneBox { pos: [-7.5, -0.05, 0.0], rot_y: 0.0, scale: [5.0, 0.3, 70.0], tex: 3, emissive: 0.05, uv_scale: 4.0, spin: false });
    boxes.push(SceneBox { pos: [7.5, -0.05, 0.0], rot_y: 0.0, scale: [5.0, 0.3, 70.0], tex: 3, emissive: 0.05, uv_scale: 4.0, spin: false });
    // Caribbean ocean bay on the right
    boxes.push(SceneBox { pos: [28.0, -0.35, 0.0], rot_y: 0.0, scale: [36.0, 0.2, 80.0], tex: 4, emissive: 0.12, uv_scale: 5.0, spin: false });
    // Left plaza ground
    boxes.push(SceneBox { pos: [-22.0, -0.15, 0.0], rot_y: 0.0, scale: [24.0, 0.2, 80.0], tex: 3, emissive: 0.02, uv_scale: 6.0, spin: false });

    // Art-Deco neon & Miami pastel buildings on the left strip
    let left_buildings = [
        (-13.5, -18.0, 8.0, 14.0, 8.0, 1u32, 0.35, 2.0),
        (-13.0, -6.0, 7.0, 9.5, 7.5, 2u32, 0.20, 1.5),
        (-14.0, 7.0, 8.5, 17.0, 8.5, 1u32, 0.40, 2.0),
        (-13.0, 20.0, 7.0, 11.0, 8.0, 2u32, 0.25, 1.5),
    ];
    for &(x, z, sx, sy, sz, tex, em, uv) in &left_buildings {
        boxes.push(SceneBox { pos: [x, sy * 0.5, z], rot_y: 0.0, scale: [sx, sy, sz], tex, emissive: em, uv_scale: uv, spin: false });
    }

    // Oceanfront boutique hotels on the right boardwalk
    let right_buildings = [
        (13.0, -14.0, 6.0, 7.5, 7.0, 2u32, 0.22, 1.5),
        (13.5, 12.0, 6.5, 11.5, 7.0, 1u32, 0.35, 1.8),
    ];
    for &(x, z, sx, sy, sz, tex, em, uv) in &right_buildings {
        boxes.push(SceneBox { pos: [x, sy * 0.5, z], rot_y: 0.0, scale: [sx, sy, sz], tex, emissive: em, uv_scale: uv, spin: false });
    }

    // Palm trees along the boulevard
    for &z in &[-20.0f32, -10.0, 0.0, 10.0, 20.0] {
        for &side in &[-6.0f32, 6.0] {
            // Trunk
            boxes.push(SceneBox { pos: [side, 2.2, z], rot_y: 0.0, scale: [0.35, 4.4, 0.35], tex: 3, emissive: 0.0, uv_scale: 1.0, spin: false });
            // Canopy
            boxes.push(SceneBox { pos: [side, 4.6, z], rot_y: z * 0.2, scale: [2.6, 0.9, 2.6], tex: 5, emissive: 0.10, uv_scale: 1.0, spin: false });
        }
    }

    // Giant glowing Synthwave billboard spanning the horizon
    boxes.push(SceneBox { pos: [0.0, 6.5, 24.0], rot_y: 0.0, scale: [12.0, 7.0, 0.6], tex: 6, emissive: 0.55, uv_scale: 1.0, spin: false });
    // Support pillars
    boxes.push(SceneBox { pos: [-4.8, 1.8, 24.0], rot_y: 0.0, scale: [0.6, 3.6, 0.6], tex: 1, emissive: 0.25, uv_scale: 1.0, spin: false });
    boxes.push(SceneBox { pos: [4.8, 1.8, 24.0], rot_y: 0.0, scale: [0.6, 3.6, 0.6], tex: 1, emissive: 0.25, uv_scale: 1.0, spin: false });

    // Central rotating calibration & synthwave monolith cubes (proof of live BC1 streaming)
    boxes.push(SceneBox { pos: [0.0, 2.2, 0.0], rot_y: 0.0, scale: [2.4, 2.4, 2.4], tex: 7, emissive: 0.25, uv_scale: 1.0, spin: true });
    boxes.push(SceneBox { pos: [0.0, 5.1, 0.0], rot_y: 0.7, scale: [1.5, 1.5, 1.5], tex: 6, emissive: 0.45, uv_scale: 1.0, spin: true });

    boxes
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DemoMode {
    GpuOnly,
    CpuOnly,
    Toggleable,
}

pub struct DemoConfig {
    pub mode: DemoMode,
    pub initial_chunks: usize,
    pub multi_core_cpu: bool,
    pub screenshot_path: Option<PathBuf>,
    pub screenshot_frames: u32,
    pub measure_frames: Option<u32>,
}

struct WindowState {
    mode: DemoMode,
    use_gpu: bool,
    multi_core_cpu: bool,
    chunks_per_frame: usize,
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
                    PostQuitMessage(0);
                } else if vk == VK_SPACE.0 && st.mode == DemoMode::Toggleable {
                    st.use_gpu = !st.use_gpu;
                } else if vk == VK_M.0 && (st.mode == DemoMode::CpuOnly || !st.use_gpu) {
                    st.multi_core_cpu = !st.multi_core_cpu;
                } else if vk == VK_UP.0 {
                    st.chunks_per_frame = (st.chunks_per_frame * 2).min(256);
                } else if vk == VK_DOWN.0 {
                    st.chunks_per_frame = (st.chunks_per_frame / 2).max(8);
                }
            }
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

pub fn run_demo(
    assets: &[TextureAsset],
    window_tbl: &Window,
    quads_tbl: &QuadTable,
    cfg: DemoConfig,
) {
    unsafe {
        let mut ctx = GpuContext::new(false);
        let max_chunks = 256usize;
        let max_blocks = max_chunks * NUM_BLOCKS;
        let max_stream = max_chunks * 160 * 1024;
        let gpu_dec = BcpackGpuDecoder::new(
            &mut ctx,
            window_tbl,
            quads_tbl,
            max_chunks,
            max_blocks,
            max_stream,
        );

        // Pre-pack batches for all load levels (8, 16, 32, 64, 128, 256)
        let levels = [8usize, 16, 32, 64, 128, 256];
        let batches: Vec<(usize, _)> = levels
            .iter()
            .map(|&n| {
                let batch = pack_batch(
                    (0..n).map(|i| (assets[i % assets.len()].bcpack_bytes.as_slice(), NUM_BLOCKS)),
                );
                (n, batch)
            })
            .collect();

        // Create Win32 window
        let hinst = GetModuleHandleW(None).expect("GetModuleHandleW");
        let class_name = w!("BcpackGpuDemoWnd");
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
            w!("BCPack GPU Demo (RTX 5080 HLSL SM 6.6) - Starting..."),
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

        let init_chunks = levels
            .iter()
            .copied()
            .min_by_key(|&x| x.abs_diff(cfg.initial_chunks))
            .unwrap_or(64);
        let init_use_gpu = cfg.mode != DemoMode::CpuOnly;

        let mut wstate = Box::new(WindowState {
            mode: cfg.mode,
            use_gpu: init_use_gpu,
            multi_core_cpu: cfg.multi_core_cpu,
            chunks_per_frame: init_chunks,
            should_quit: false,
        });
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (&mut *wstate as *mut WindowState) as isize);
        let _ = ShowWindow(hwnd, SW_SHOW);

        // SwapChain
        let sc_desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: WIN_W,
            Height: WIN_H,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
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

        // Descriptor heaps: RTV (2), DSV (1), SRV (8 textures)
        let rtv_heap: ID3D12DescriptorHeap = ctx
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 2,
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
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap();

        let rtv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let srv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        let mut backbuffers = Vec::with_capacity(2);
        let rtv_base = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        for i in 0..2u32 {
            let bb: ID3D12Resource = swap_chain.GetBuffer(i).unwrap();
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_base.ptr + (i * rtv_inc) as usize,
            };
            ctx.device.CreateRenderTargetView(&bb, None, handle);
            backbuffers.push(bb);
        }

        // Depth buffer
        let depth_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: WIN_W as u64,
            Height: WIN_H,
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

        // Create 8 live 512x512 DXGI_FORMAT_BC1_UNORM textures + SRVs
        let mut textures = Vec::with_capacity(8);
        let srv_base = srv_heap.GetCPUDescriptorHandleForHeapStart();
        for i in 0..8usize {
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

        // CPU Upload staging buffer for CPU mode comparison (up to 8 * 131072 bytes = 1 MB)
        let cpu_upload_buf = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (8 * TEX_BC1_BYTES) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );

        // Graphics Root Signature: 32-bit constants (36 DWORDs at b0) + DescriptorTable (8 SRVs t0..t7) + StaticSampler (s0)
        let range = D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 8,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let gfx_params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: (size_of::<DrawConstants>() / 4) as u32,
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

        // Graphics PSO
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

        let mut rt_formats = [DXGI_FORMAT_UNKNOWN; 8];
        rt_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

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
            NumRenderTargets: 1,
            RTVFormats: rt_formats,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let gfx_pso: ID3D12PipelineState = ctx.device.CreateGraphicsPipelineState(&pso_desc).unwrap();
        let _ = ManuallyDrop::take(&mut pso_desc.pRootSignature);

        // Cube Vertex & Index buffers (Upload heap is fine for tiny static mesh)
        let (verts, inds) = build_unit_cube();
        let vb_buf = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (verts.len() * size_of::<Vertex>()) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let ib_buf = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (inds.len() * size_of::<u16>()) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        upload_slice(&vb_buf, &verts);
        upload_slice(&ib_buf, &inds);
        let vbv = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vb_buf.GetGPUVirtualAddress(),
            SizeInBytes: (verts.len() * size_of::<Vertex>()) as u32,
            StrideInBytes: size_of::<Vertex>() as u32,
        };
        let ibv = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: ib_buf.GetGPUVirtualAddress(),
            SizeInBytes: (inds.len() * size_of::<u16>()) as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };

        // Frame command allocator & list + GPU timestamp query heap + screenshot readback
        let cmd_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let cmd_list: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &cmd_alloc, None).unwrap();
        cmd_list.Close().unwrap();

        let mut ts_heap: Option<ID3D12QueryHeap> = None;
        ctx.device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut ts_heap).unwrap();
        let ts_heap = ts_heap.unwrap();
        let ts_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let ss_row_pitch = WIN_W * 4; // 5120, multiple of 256
        let ss_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, (ss_row_pitch * WIN_H) as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let scene = build_vice_city_scene();
        let proj = Mat4::perspective_lh(60.0f32.to_radians(), WIN_W as f32 / WIN_H as f32, 0.1, 200.0);

        let start_t = Instant::now();
        let mut last_title_t = Instant::now();
        let mut measure_start: Option<Instant> = None;
        let mut measure_decode_sum = 0.0f64;
        let mut frame_count = 0u32;
        let mut fps_accum = 0u32;
        let mut cpu_buf = vec![0u8; 8 * TEX_BC1_BYTES];

        println!("================================================================");
        match cfg.mode {
            DemoMode::GpuOnly => println!("NEW GPU Demo (RTX 5080 HLSL SM 6.6) on: {}", ctx.adapter_name),
            DemoMode::CpuOnly => println!("OLD CPU Demo (bcpack-rs on CPU) on: {}", ctx.adapter_name),
            DemoMode::Toggleable => println!("BCPack GPU/CPU Live Demo on: {}", ctx.adapter_name),
        }
        println!("Controls:");
        if cfg.mode == DemoMode::Toggleable {
            println!("  [SPACE]   Toggle GPU Compute (RTX 5080) <-> CPU (bcpack-rs)");
        }
        if cfg.mode != DemoMode::GpuOnly {
            println!("  [M]       Toggle CPU 1-Core <-> CPU Multi-Core");
        }
        println!("  [UP/DOWN] Adjust streamed BC1 chunks/frame (8 .. 256)");
        println!("  [ESC]     Exit");
        println!("================================================================");

        let mut msg = MSG::default();
        while !wstate.should_quit {
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                if msg.message == WM_QUIT {
                    wstate.should_quit = true;
                }
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if wstate.should_quit {
                break;
            }

            let cur_chunks = wstate.chunks_per_frame;
            let batch = &batches.iter().find(|(n, _)| *n == cur_chunks).unwrap().1;
            let streamed_mb = (cur_chunks * TEX_BC1_BYTES) as f64 / (1024.0 * 1024.0);

            let mut decomp_ms = 0.0f64;
            if !wstate.use_gpu {
                // CPU decompression path (real bcpack-rs decode + upload to staging)
                let t0 = Instant::now();
                let decoded_list: Vec<Vec<u8>> = if wstate.multi_core_cpu {
                    (0..cur_chunks)
                        .into_par_iter()
                        .map(|i| {
                            bc1::decode(
                                &assets[i % assets.len()].bcpack_bytes,
                                NUM_BLOCKS,
                                window_tbl,
                                quads_tbl,
                            )
                            .unwrap()
                            .blocks
                        })
                        .collect()
                } else {
                    (0..cur_chunks)
                        .map(|i| {
                            bc1::decode(
                                &assets[i % assets.len()].bcpack_bytes,
                                NUM_BLOCKS,
                                window_tbl,
                                quads_tbl,
                            )
                            .unwrap()
                            .blocks
                        })
                        .collect()
                };
                for ti in 0..8 {
                    let src = &decoded_list[ti % cur_chunks];
                    cpu_buf[ti * TEX_BC1_BYTES..(ti + 1) * TEX_BC1_BYTES].copy_from_slice(src);
                }
                upload_slice(&cpu_upload_buf, &cpu_buf);
                decomp_ms = t0.elapsed().as_secs_f64() * 1000.0;
            } else {
                gpu_dec.upload_batch(batch);
            }

            cmd_alloc.Reset().unwrap();
            cmd_list.Reset(&cmd_alloc, None).unwrap();

            if wstate.use_gpu {
                gpu_dec.record_upload_and_dispatch(&cmd_list, batch, Some(&ts_heap), 0, 1);
                cmd_list.ResolveQueryData(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &ts_readback, 0);
            }

            // Copy decompressed BC1 blocks directly into all 8 live DXGI_FORMAT_BC1_UNORM textures
            let mut to_copy = Vec::with_capacity(9);
            for t in &textures {
                to_copy.push(transition_barrier(t, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST));
            }
            if wstate.use_gpu {
                to_copy.push(transition_barrier(&gpu_dec.out_blocks_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
            }
            cmd_barriers(&cmd_list, to_copy);

            let src_res = if wstate.use_gpu {
                &gpu_dec.out_blocks_default
            } else {
                &cpu_upload_buf
            };
            for (ti, tex) in textures.iter().enumerate() {
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
                            Offset: (ti * TEX_BC1_BYTES) as u64,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_BC1_UNORM,
                                Width: TEX_W as u32,
                                Height: TEX_H as u32,
                                Depth: 1,
                                RowPitch: (TEX_W / 4 * 8) as u32, // 1024 bytes (aligned to 256)
                            },
                        },
                    },
                };
                cmd_list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                let _ = ManuallyDrop::take(&mut dst.pResource);
                let _ = ManuallyDrop::take(&mut src.pResource);
            }

            let mut to_srv = Vec::with_capacity(9);
            for t in &textures {
                to_srv.push(transition_barrier(t, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
            }
            if wstate.use_gpu {
                to_srv.push(transition_barrier(&gpu_dec.out_blocks_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            }
            cmd_barriers(&cmd_list, to_srv);

            // Render 3D Vice-City sunset scene
            let bb_idx = swap_chain.GetCurrentBackBufferIndex();
            let bb = &backbuffers[bb_idx as usize];
            let mut b_rt = [transition_barrier(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)];
            cmd_list.ResourceBarrier(&b_rt);
            drop_barriers(&mut b_rt);

            let rtv_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_base.ptr + (bb_idx * rtv_inc) as usize,
            };
            let dsv_handle = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let clear_col = [0.24f32, 0.11, 0.30, 1.0];
            cmd_list.ClearRenderTargetView(rtv_handle, &clear_col, None);
            cmd_list.ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, &[]);
            cmd_list.OMSetRenderTargets(1, Some(&rtv_handle), false, Some(&dsv_handle));

            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: WIN_W as f32,
                Height: WIN_H as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let sc = RECT { left: 0, top: 0, right: WIN_W as i32, bottom: WIN_H as i32 };
            cmd_list.RSSetViewports(&[vp]);
            cmd_list.RSSetScissorRects(&[sc]);

            cmd_list.SetGraphicsRootSignature(&gfx_root_sig);
            cmd_list.SetPipelineState(&gfx_pso);
            cmd_list.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
            cmd_list.SetGraphicsRootDescriptorTable(1, srv_heap.GetGPUDescriptorHandleForHeapStart());
            cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd_list.IASetVertexBuffers(0, Some(&[vbv]));
            cmd_list.IASetIndexBuffer(Some(&ibv));

            let t_sec = if cfg.screenshot_path.is_some() {
                0.75f32
            } else {
                start_t.elapsed().as_secs_f32()
            };
            let cam_ang = t_sec * 0.28;
            let eye = [cam_ang.sin() * 14.5, 6.8 + (t_sec * 0.4).sin() * 0.7, -cam_ang.cos() * 16.5];
            let view = Mat4::look_at_lh(eye, [0.0, 2.8, 3.0], [0.0, 1.0, 0.0]);
            let vp_mat = proj.mul(&view);

            for obj in &scene {
                let rot = if obj.spin { obj.rot_y + t_sec * 1.1 } else { obj.rot_y };
                let model = Mat4::trs(obj.pos, rot, obj.scale);
                let mvp = vp_mat.mul(&model);
                let dc = DrawConstants {
                    mvp: mvp.0,
                    model: model.0,
                    tex_index: obj.tex,
                    emissive: obj.emissive,
                    time: t_sec,
                    uv_scale: obj.uv_scale,
                };
                cmd_list.SetGraphicsRoot32BitConstants(
                    0,
                    (size_of::<DrawConstants>() / 4) as u32,
                    &dc as *const _ as *const c_void,
                    0,
                );
                cmd_list.DrawIndexedInstanced(inds.len() as u32, 1, 0, 0, 0);
            }

            let take_shot = cfg.screenshot_path.is_some() && frame_count + 1 >= cfg.screenshot_frames;
            if take_shot {
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
            ctx.signal_and_wait();

            if wstate.use_gpu {
                let mut ptr: *mut c_void = std::ptr::null_mut();
                let r_ts = D3D12_RANGE { Begin: 0, End: 16 };
                ts_readback.Map(0, Some(&r_ts), Some(&mut ptr)).unwrap();
                let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
                decomp_ms = (ts[1].saturating_sub(ts[0]) as f64 / ctx.timestamp_freq as f64) * 1000.0;
                ts_readback.Unmap(0, None);
            }

            let _ = swap_chain.Present(0, DXGI_PRESENT(0));
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
                        let mode_tag = if wstate.use_gpu {
                            "NEW_GPU_RTX5080"
                        } else if wstate.multi_core_cpu {
                            "OLD_CPU_MULTICORE"
                        } else {
                            "OLD_CPU_1CORE"
                        };
                        println!(
                            "[fps] mode={} chunks={} ({:.1} MiB/frame) frames={} avg_fps={:.2} avg_decode_ms={:.3}",
                            mode_tag, cur_chunks, streamed_mb, measured, avg_fps, avg_dec
                        );
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
                    "NEW: GPU HLSL SM6.6 (RTX 5080)"
                } else if wstate.multi_core_cpu {
                    "OLD: CPU Multi-Core (bcpack-rs)"
                } else {
                    "OLD: CPU 1-Core (bcpack-rs)"
                };
                let controls = match cfg.mode {
                    DemoMode::GpuOnly => "[UP/DN]=Load [ESC]=Quit",
                    DemoMode::CpuOnly => "[M]=1T/MT [UP/DN]=Load [ESC]=Quit",
                    DemoMode::Toggleable => "[SPACE]=GPU/CPU [M]=1T/MT [UP/DN]=Load [ESC]=Quit",
                };
                let title = format!(
                    "{} | {:.1} FPS | {:.0} MB/frame ({} chunks) | Decode: {:.2} ms ({:.0} MB/s) | {}",
                    mode_str, fps, streamed_mb, cur_chunks, decomp_ms, throughput, controls
                );
                let wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
                let _ = SetWindowTextW(hwnd, PCWSTR(wide.as_ptr()));
                fps_accum = 0;
                last_title_t = Instant::now();
            }
        }
    }
}
