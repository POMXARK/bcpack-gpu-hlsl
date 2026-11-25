// Stage 3 World Systems:
// - 12M+ Triangle Modular Hardware Mesh-Shader City (Recessed Glass Towers, Balconies, Storefronts, Palms, Lamps, Seawall, Road Markings, Skyline HLOD)
// - Live 6-Mip Hi-Z Depth Pyramid Builder & Conservative Occlusion Culling
// - Live 500-NPC GPU-Skinned Pedestrian Crowd (CS LBS + Instanced 4-MRT G-Buffer Draw)

use crate::crowd_skinning::{BaseSkinnedVertex, BoneMat, OutSkinnedVertex, CROWD_SKIN_CS_DXIL};
use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, heap_props, transition_barrier, uav_barrier,
    upload_slice, GpuContext,
};
use crate::hiz_culling::HIZ_BUILD_CS_DXIL;
use crate::meshlets::extract_frustum_planes;
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub const AAA_MESHLET_AS_DXIL: &[u8] = include_bytes!("../shaders/aaa_meshlet_as.dxil");
pub const AAA_MESHLET_MS_DXIL: &[u8] = include_bytes!("../shaders/aaa_meshlet_ms.dxil");
pub const AAA_MESHLET_PS_DXIL: &[u8] = include_bytes!("../shaders/aaa_meshlet_ps.dxil");
pub const AAA_CROWD_VS_DXIL: &[u8] = include_bytes!("../shaders/aaa_crowd_vs.dxil");
pub const AAA_CROWD_PS_DXIL: &[u8] = include_bytes!("../shaders/aaa_crowd_ps.dxil");

pub const AAA_SRV_DEPTH_R32: usize = 107;
pub const AAA_SRV_HIZ_BASE: usize = 108; // 108..113 = 6 Hi-Z mip SRVs
pub const AAA_UAV_HIZ_BASE: usize = 114; // 114..119 = 6 Hi-Z mip UAVs

const PROTO_SKYSCRAPER: usize = 0;
const PROTO_HOTEL: usize = 1;
const PROTO_LAMP: usize = 2;
const PROTO_PALM: usize = 3;
const PROTO_HLOD_TOWER: usize = 4;
const PROTO_SEAWALL: usize = 5;
const PROTO_ROAD_MARKING: usize = 6;

const MAT_CONCRETE: u32 = 0;
const MAT_GLASS: u32 = 1;
const MAT_STOREFRONT: u32 = 2;
const MAT_METAL: u32 = 3;
const MAT_BARK: u32 = 4;
const MAT_FROND: u32 = 5;
const MAT_EMISSIVE: u32 = 6;
const MAT_HLOD: u32 = 7;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AaaMeshletVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub mat_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AaaMeshletDesc {
    pub vert_offset: u32,
    pub vert_count: u32,
    pub prim_offset: u32,
    pub prim_count: u32,
    pub bound_center: [f32; 3],
    pub bound_radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AaaMeshletInstance {
    pub pos: [f32; 3],
    pub rot_y: f32,
    pub scale: [f32; 3],
    pub tint_r: f32,
    pub tint_g: f32,
    pub tint_b: f32,
    pub uv_scale: f32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AaaMeshletWorkItem {
    pub inst_idx: u32,
    pub meshlet_idx: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AaaMeshFrameCB {
    pub vp_jittered: [f32; 16],
    pub vp_unjittered: [f32; 16],
    pub prev_vp_unjittered: [f32; 16],
    pub frustum_planes: [[f32; 4]; 6],
    pub camera_pos: [f32; 4],
    pub sun_dir_time: [f32; 4],
    pub cull_params: [u32; 4],
    pub cull_range: [f32; 4],
    pub concrete_tex: [u32; 4],
}

#[derive(Clone, Copy, Default, Debug)]
pub struct CityCullStats {
    pub vis_meshlets: u32,
    pub vis_tris: u32,
    pub frustum_culled: u32,
    pub hiz_occluded: u32,
    pub hiz_occluded_tris: u32,
}

#[repr(C, align(8))]
struct PsoSubobject<T> {
    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    inner: T,
}

#[repr(C)]
struct AaaMeshShaderPsoStream {
    root_sig: PsoSubobject<ManuallyDrop<Option<ID3D12RootSignature>>>,
    as_shader: PsoSubobject<D3D12_SHADER_BYTECODE>,
    ms_shader: PsoSubobject<D3D12_SHADER_BYTECODE>,
    ps_shader: PsoSubobject<D3D12_SHADER_BYTECODE>,
    blend: PsoSubobject<D3D12_BLEND_DESC>,
    sample_mask: PsoSubobject<u32>,
    rasterizer: PsoSubobject<D3D12_RASTERIZER_DESC>,
    depth_stencil: PsoSubobject<D3D12_DEPTH_STENCIL_DESC>,
    rtv_formats: PsoSubobject<D3D12_RT_FORMAT_ARRAY>,
    dsv_format: PsoSubobject<DXGI_FORMAT>,
    sample_desc: PsoSubobject<DXGI_SAMPLE_DESC>,
}

pub struct AaaMeshletLibrary {
    pub vertices: Vec<AaaMeshletVertex>,
    pub prim_indices: Vec<u32>,
    pub meshlets: Vec<AaaMeshletDesc>,
    pub proto_ranges: [(u32, u32); 7],
}

struct Builder {
    vertices: Vec<AaaMeshletVertex>,
    prim_indices: Vec<u32>,
    meshlets: Vec<AaaMeshletDesc>,
}

impl Builder {
    fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(131072),
            prim_indices: Vec::with_capacity(262144),
            meshlets: Vec::with_capacity(4096),
        }
    }

    #[inline]
    fn norm3(v: [f32; 3]) -> [f32; 3] {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        [v[0] / l, v[1] / l, v[2] / l]
    }

    fn push_patch<F>(&mut self, total_cols: usize, total_rows: usize, mat_id: u32, eval: F)
    where
        F: Fn(f32, f32) -> ([f32; 3], [f32; 3], [f32; 2]),
    {
        let mut r0 = 0usize;
        while r0 < total_rows {
            let r1 = (r0 + 7).min(total_rows);
            let nr = r1 - r0;
            let mut c0 = 0usize;
            while c0 < total_cols {
                let c1 = (c0 + 7).min(total_cols);
                let nc = c1 - c0;

                let v_off = self.vertices.len() as u32;
                let p_off = self.prim_indices.len() as u32;
                let mut sum = [0.0f32; 3];
                let mut pts = Vec::with_capacity((nr + 1) * (nc + 1));

                for r in r0..=r1 {
                    let v = r as f32 / total_rows as f32;
                    for c in c0..=c1 {
                        let u = c as f32 / total_cols as f32;
                        let (pos, nrm, uv) = eval(u, v);
                        let n = Self::norm3(nrm);
                        sum[0] += pos[0];
                        sum[1] += pos[1];
                        sum[2] += pos[2];
                        pts.push(pos);
                        self.vertices.push(AaaMeshletVertex {
                            pos,
                            normal: n,
                            uv,
                            mat_id,
                            pad: 0,
                        });
                    }
                }

                let v_cnt = pts.len() as u32;
                let inv = 1.0 / pts.len() as f32;
                let center = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
                let mut max_r2 = 0.0025f32;
                for p in &pts {
                    let dx = p[0] - center[0];
                    let dy = p[1] - center[1];
                    let dz = p[2] - center[2];
                    max_r2 = max_r2.max(dx * dx + dy * dy + dz * dz);
                }

                let stride = nc + 1;
                let mut tri_cnt = 0u32;
                for r in 0..nr {
                    for c in 0..nc {
                        let i00 = (r * stride + c) as u32;
                        let i10 = (r * stride + c + 1) as u32;
                        let i01 = ((r + 1) * stride + c) as u32;
                        let i11 = ((r + 1) * stride + c + 1) as u32;
                        self.prim_indices.push(i00 | (i01 << 8) | (i11 << 16));
                        self.prim_indices.push(i00 | (i11 << 8) | (i10 << 16));
                        tri_cnt += 2;
                    }
                }

                self.meshlets.push(AaaMeshletDesc {
                    vert_offset: v_off,
                    vert_count: v_cnt,
                    prim_offset: p_off,
                    prim_count: tri_cnt,
                    bound_center: center,
                    bound_radius: max_r2.sqrt(),
                });

                c0 = c1;
            }
            r0 = r1;
        }
    }

    fn push_box(&mut self, center: [f32; 3], size: [f32; 3], sub_u: usize, sub_v: usize, uv_scale: [f32; 2], mat_id: u32) {
        let hx = size[0] * 0.5;
        let hy = size[1] * 0.5;
        let hz = size[2] * 0.5;
        let [cx, cy, cz] = center;
        let faces: [([f32; 3], [f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([cx - hx, cy - hy, cz + hz], [size[0], 0.0, 0.0], [0.0, size[1], 0.0], [0.0, 0.0, 1.0]),
            ([cx + hx, cy - hy, cz - hz], [-size[0], 0.0, 0.0], [0.0, size[1], 0.0], [0.0, 0.0, -1.0]),
            ([cx + hx, cy - hy, cz + hz], [0.0, 0.0, -size[2]], [0.0, size[1], 0.0], [1.0, 0.0, 0.0]),
            ([cx - hx, cy - hy, cz - hz], [0.0, 0.0, size[2]], [0.0, size[1], 0.0], [-1.0, 0.0, 0.0]),
            ([cx - hx, cy + hy, cz + hz], [size[0], 0.0, 0.0], [0.0, 0.0, -size[2]], [0.0, 1.0, 0.0]),
            ([cx - hx, cy - hy, cz - hz], [size[0], 0.0, 0.0], [0.0, 0.0, size[2]], [0.0, -1.0, 0.0]),
        ];
        for (o, au, av, n) in faces {
            self.push_patch(sub_u, sub_v, mat_id, |u, v| {
                (
                    [o[0] + au[0] * u + av[0] * v, o[1] + au[1] * u + av[1] * v, o[2] + au[2] * u + av[2] * v],
                    n,
                    [u * uv_scale[0], (1.0 - v) * uv_scale[1]],
                )
            });
        }
    }

    fn push_cyl(&mut self, base: [f32; 3], r_bot: f32, r_top: f32, h: f32, lean_x: f32, seg_u: usize, seg_v: usize, mat_id: u32) {
        self.push_patch(seg_u, seg_v, mat_id, |u, v| {
            let ang = u * std::f32::consts::TAU;
            let (s, c) = ang.sin_cos();
            let rad = r_bot + (r_top - r_bot) * v;
            let bend = lean_x * v * v;
            let pos = [base[0] + bend + c * rad, base[1] + v * h, base[2] + s * rad];
            ([pos[0], pos[1], pos[2]], [c, 0.12, s], [u * 2.0, v * 4.0])
        });
    }

    fn build_skyscraper(&mut self) {
        // Normalized unit building centered horizontally, base at y=0, top at y=1.0
        // 1) Ground-floor colonnade podium + lit storefronts + neon awning band
        self.push_box([0.0, 0.075, 0.0], [1.04, 0.15, 1.04], 14, 7, [2.0, 1.0], MAT_CONCRETE);
        self.push_box([0.0, 0.065, 0.0], [0.96, 0.11, 0.96], 14, 7, [2.0, 1.0], MAT_STOREFRONT);
        self.push_box([0.0, 0.138, 0.0], [1.07, 0.018, 1.07], 14, 2, [2.0, 0.2], MAT_METAL);
        self.push_box([0.0, 0.126, 0.0], [1.06, 0.012, 1.06], 14, 2, [2.0, 0.1], MAT_STOREFRONT);

        // 2) Recessed reflective curtain-wall core (main shaft + upper tier)
        self.push_box([0.0, 0.48, 0.0], [0.84, 0.66, 0.84], 21, 21, [3.0, 5.0], MAT_GLASS);
        self.push_box([0.0, 0.87, 0.0], [0.64, 0.16, 0.64], 14, 7, [2.0, 2.0], MAT_GLASS);

        // 3) Exterior structural concrete corner piers + intermediate facade pilasters
        for &(sx, sz) in &[(-0.46f32, -0.46f32), (0.46, -0.46), (-0.46, 0.46), (0.46, 0.46)] {
            self.push_box([sx, 0.48, sz], [0.11, 0.68, 0.11], 7, 14, [0.5, 4.0], MAT_CONCRETE);
        }
        for &m in &[-0.22f32, 0.22f32] {
            self.push_box([0.45, 0.48, m], [0.06, 0.66, 0.06], 2, 14, [0.3, 4.0], MAT_METAL);
            self.push_box([-0.45, 0.48, m], [0.06, 0.66, 0.06], 2, 14, [0.3, 4.0], MAT_METAL);
            self.push_box([m, 0.48, 0.45], [0.06, 0.66, 0.06], 2, 14, [0.3, 4.0], MAT_METAL);
            self.push_box([m, 0.48, -0.45], [0.06, 0.66, 0.06], 2, 14, [0.3, 4.0], MAT_METAL);
        }

        // 4) Projecting cantilevered floor / balcony cornice slabs (creating deep shadow bays)
        for tier in 0..7 {
            let y = 0.21 + (tier as f32) * 0.095;
            self.push_box([0.0, y, 0.0], [0.97, 0.024, 0.97], 14, 2, [2.5, 0.25], MAT_CONCRETE);
        }

        // 5) Setback crown & rooftop mechanical penthouse
        self.push_box([0.0, 0.80, 0.0], [0.92, 0.035, 0.92], 14, 2, [2.0, 0.3], MAT_CONCRETE);
        self.push_box([0.0, 0.96, 0.0], [0.68, 0.04, 0.68], 14, 2, [1.5, 0.3], MAT_CONCRETE);
        self.push_box([0.0, 0.995, 0.0], [0.32, 0.05, 0.32], 7, 7, [1.0, 0.5], MAT_METAL);
    }

    fn build_hotel(&mut self) {
        // Coastal Art-Deco Mid-Rise Hotel with wrap balconies
        self.push_box([0.0, 0.09, 0.0], [1.04, 0.18, 1.04], 14, 7, [2.0, 1.0], MAT_CONCRETE);
        self.push_box([0.0, 0.08, 0.0], [0.96, 0.12, 0.96], 14, 7, [2.0, 1.0], MAT_STOREFRONT);
        self.push_box([0.0, 0.50, 0.0], [0.86, 0.66, 0.86], 21, 14, [2.5, 3.5], MAT_GLASS);
        for &(sx, sz) in &[(-0.46f32, -0.46f32), (0.46, -0.46), (-0.46, 0.46), (0.46, 0.46)] {
            self.push_box([sx, 0.50, sz], [0.12, 0.72, 0.12], 7, 14, [0.5, 3.0], MAT_CONCRETE);
        }
        for tier in 0..6 {
            let y = 0.22 + (tier as f32) * 0.105;
            self.push_box([0.0, y, 0.0], [1.00, 0.026, 1.00], 14, 2, [2.5, 0.25], MAT_CONCRETE);
            self.push_box([0.0, y + 0.024, 0.0], [0.99, 0.012, 0.99], 7, 1, [2.0, 0.1], MAT_METAL);
        }
        self.push_box([0.0, 0.88, 0.0], [0.68, 0.14, 0.68], 14, 7, [1.5, 0.8], MAT_CONCRETE);
        self.push_box([0.0, 0.97, 0.0], [0.36, 0.05, 0.36], 7, 2, [1.0, 0.4], MAT_STOREFRONT);
    }

    fn build_lamp(&mut self) {
        // Fluted boulevard streetlamp (pole + arm + warm emissive globe)
        self.push_cyl([0.0, 0.0, 0.0], 0.18, 0.11, 0.9, 0.0, 14, 7, MAT_METAL);
        self.push_cyl([0.0, 0.9, 0.0], 0.10, 0.06, 5.2, 0.0, 14, 14, MAT_METAL);
        self.push_box([0.85, 6.08, 0.0], [1.85, 0.10, 0.14], 7, 2, [1.0, 0.5], MAT_METAL);
        self.push_box([1.68, 5.98, 0.0], [0.56, 0.14, 0.34], 7, 4, [1.0, 0.5], MAT_EMISSIVE);
    }

    fn build_palm(&mut self) {
        // Curved tropical palm trunk + 10 arched feathered palm fronds
        self.push_cyl([0.0, 0.0, 0.0], 0.28, 0.14, 7.6, 0.72, 14, 14, MAT_BARK);
        let tip = [0.72f32, 7.55, 0.0];
        for f in 0..10 {
            let yaw = (f as f32 / 10.0) * std::f32::consts::TAU + (f % 2) as f32 * 0.15;
            let pitch = if (f % 2) == 0 { 0.32 } else { 0.62 };
            let (sy, cy) = yaw.sin_cos();
            let len = if (f % 2) == 0 { 3.35 } else { 2.75 };
            self.push_patch(7, 14, MAT_FROND, |u, v| {
                let s = (u - 0.5) * 2.0; // -1..1 frond width
                let width = (1.0 - v * v * 0.85) * 0.46 * (1.0 - (v - 0.15).max(0.0) * 0.25);
                let arch_y = v * len * pitch - v * v * (len * 0.48) - s.abs() * 0.08;
                let fwd = v * len;
                let lx = fwd * cy - s * width * sy;
                let lz = fwd * sy + s * width * cy;
                let n = [cy * 0.25, 0.94, sy * 0.25];
                ([tip[0] + lx, tip[1] + arch_y, tip[2] + lz], n, [u, v * 2.0])
            });
        }
    }

    fn build_hlod_tower(&mut self) {
        // Distant bayfront skyline HLOD stepped silhouette
        self.push_box([0.0, 0.35, 0.0], [1.0, 0.70, 1.0], 7, 7, [2.0, 3.0], MAT_HLOD);
        self.push_box([0.0, 0.82, 0.0], [0.70, 0.26, 0.70], 7, 7, [1.5, 1.5], MAT_HLOD);
        self.push_box([0.0, 0.98, 0.0], [0.30, 0.12, 0.30], 7, 2, [1.0, 1.0], MAT_HLOD);
    }

    fn build_seawall(&mut self) {
        // Concrete promenade balustrade bay (12m span along Z): pillars + top/mid rails + warm cap lights
        self.push_box([0.0, 0.50, 0.0], [0.90, 1.00, 12.0], 7, 14, [1.0, 4.0], MAT_CONCRETE);
        self.push_box([0.0, 1.24, 0.0], [0.16, 0.09, 12.0], 2, 14, [0.5, 4.0], MAT_METAL);
        self.push_box([0.0, 1.08, 0.0], [0.10, 0.05, 12.0], 2, 14, [0.5, 4.0], MAT_METAL);
        for &pz in &[-5.5f32, 0.0, 5.5] {
            self.push_box([0.0, 1.20, pz], [0.55, 0.44, 0.55], 7, 7, [0.5, 0.5], MAT_CONCRETE);
            self.push_box([0.0, 1.46, pz], [0.24, 0.12, 0.24], 7, 2, [0.3, 0.3], MAT_EMISSIVE);
        }
    }

    fn build_road_marking(&mut self) {
        // Flush white dashed lane stripe at y=0.015 (6m Z cycle)
        self.push_box([-3.6, 0.018, 0.0], [0.16, 0.012, 2.8], 2, 7, [0.2, 1.0], MAT_EMISSIVE);
        self.push_box([3.6, 0.018, 0.0], [0.16, 0.012, 2.8], 2, 7, [0.2, 1.0], MAT_EMISSIVE);
    }
}

pub fn build_aaa_meshlet_library() -> AaaMeshletLibrary {
    let mut b = Builder::new();
    let mut ranges = [(0u32, 0u32); 7];

    let mut build_proto = |idx: usize, f: fn(&mut Builder)| {
        let s = b.meshlets.len() as u32;
        f(&mut b);
        ranges[idx] = (s, b.meshlets.len() as u32);
    };

    build_proto(PROTO_SKYSCRAPER, Builder::build_skyscraper);
    build_proto(PROTO_HOTEL, Builder::build_hotel);
    build_proto(PROTO_LAMP, Builder::build_lamp);
    build_proto(PROTO_PALM, Builder::build_palm);
    build_proto(PROTO_HLOD_TOWER, Builder::build_hlod_tower);
    build_proto(PROTO_SEAWALL, Builder::build_seawall);
    build_proto(PROTO_ROAD_MARKING, Builder::build_road_marking);

    AaaMeshletLibrary {
        vertices: b.vertices,
        prim_indices: b.prim_indices,
        meshlets: b.meshlets,
        proto_ranges: ranges,
    }
}

pub struct CoastalCityScene {
    pub instances: Vec<AaaMeshletInstance>,
    pub work_items: Vec<AaaMeshletWorkItem>,
    pub total_tris: usize,
    pub dxr_tower_boxes: Vec<([f32; 3], [f32; 3], [f32; 3])>,
}

pub fn build_coastal_city_instances(lib: &AaaMeshletLibrary) -> CoastalCityScene {
    let mut instances = Vec::with_capacity(2048);
    let mut work_items = Vec::with_capacity(180000);
    let mut dxr_tower_boxes = Vec::with_capacity(24);
    let mut total_tris_acc = 0usize;

    let proto_tris: [usize; 7] = std::array::from_fn(|p| {
        let (ms, me) = lib.proto_ranges[p];
        (ms..me).map(|m| lib.meshlets[m as usize].prim_count as usize).sum()
    });

    let mut push_inst = |proto: usize, inst: AaaMeshletInstance| -> usize {
        let idx = instances.len() as u32;
        instances.push(inst);
        let (ms, me) = lib.proto_ranges[proto];
        for m in ms..me {
            work_items.push(AaaMeshletWorkItem {
                inst_idx: idx,
                meshlet_idx: m,
            });
        }
        proto_tris[proto]
    };

    let tints = [
        [0.92f32, 0.84, 0.76],
        [0.82, 0.86, 0.90],
        [0.94, 0.80, 0.72],
        [0.86, 0.82, 0.78],
        [0.78, 0.82, 0.86],
    ];

    // 1. Front-row boulevard modular skyscrapers & Art-Deco hotels (-X right side of avenue)
    for i in 0..20usize {
        let z = -95.0 + (i as f32) * 18.0;
        let h = 24.0 + ((i * 37 % 6) as f32) * 8.5;
        let w = 13.0 + ((i * 13 % 3) as f32) * 2.2;
        let x = -28.0 - (i % 2) as f32 * 3.5;
        let t = tints[i % tints.len()];
        let proto = if (i % 3) == 1 { PROTO_HOTEL } else { PROTO_SKYSCRAPER };
        total_tris_acc += push_inst(
            proto,
            AaaMeshletInstance {
                pos: [x, 0.0, z],
                rot_y: 0.0,
                scale: [w, h, w * 1.15],
                tint_r: t[0],
                tint_g: t[1],
                tint_b: t[2],
                uv_scale: 2.5,
                flags: 0,
            },
        );
        if (-55.0..=165.0).contains(&z) {
            dxr_tower_boxes.push(([x, h * 0.48, z], [w * 0.92, h * 0.96, w * 1.05], t));
        }
    }

    // 2. Boulevard Streetlamps & Tropical Coastal Palms on both promenades
    for i in 0..24usize {
        let z = -96.0 + (i as f32) * 14.0;
        total_tris_acc += push_inst(
            PROTO_PALM,
            AaaMeshletInstance {
                pos: [-13.2, 0.08, z],
                rot_y: (i as f32) * 1.1,
                scale: [1.0, 0.95 + (i % 3) as f32 * 0.12, 1.0],
                tint_r: 1.0,
                tint_g: 1.0,
                tint_b: 1.0,
                uv_scale: 1.0,
                flags: 0,
            },
        );
        total_tris_acc += push_inst(
            PROTO_PALM,
            AaaMeshletInstance {
                pos: [13.2, 0.08, z + 6.5],
                rot_y: (i as f32) * 0.85 + 1.0,
                scale: [1.0, 0.95 + ((i + 1) % 3) as f32 * 0.12, 1.0],
                tint_r: 1.0,
                tint_g: 1.0,
                tint_b: 1.0,
                uv_scale: 1.0,
                flags: 0,
            },
        );
        total_tris_acc += push_inst(
            PROTO_LAMP,
            AaaMeshletInstance {
                pos: [-12.2, 0.08, z + 7.0],
                rot_y: 0.0,
                scale: [1.0, 1.0, 1.0],
                tint_r: 1.0,
                tint_g: 0.84,
                tint_b: 0.56,
                uv_scale: 1.0,
                flags: 0,
            },
        );
        total_tris_acc += push_inst(
            PROTO_LAMP,
            AaaMeshletInstance {
                pos: [12.2, 0.08, z],
                rot_y: std::f32::consts::PI,
                scale: [1.0, 1.0, 1.0],
                tint_r: 1.0,
                tint_g: 0.84,
                tint_b: 0.56,
                uv_scale: 1.0,
                flags: 0,
            },
        );
    }

    // 3. Oceanfront Concrete Seawall Balustrade along x = +24.2 (z = -108..+240)
    for i in 0..29usize {
        let z = -102.0 + (i as f32) * 12.0;
        total_tris_acc += push_inst(
            PROTO_SEAWALL,
            AaaMeshletInstance {
                pos: [24.2, 0.0, z],
                rot_y: 0.0,
                scale: [1.0, 1.0, 1.0],
                tint_r: 0.92,
                tint_g: 0.88,
                tint_b: 0.84,
                uv_scale: 2.0,
                flags: 0,
            },
        );
    }

    // 4. Boulevard White Dashed Lane Markings & Double Amber Median
    for i in 0..54usize {
        let z = -105.0 + (i as f32) * 6.0;
        total_tris_acc += push_inst(
            PROTO_ROAD_MARKING,
            AaaMeshletInstance {
                pos: [0.0, 0.0, z],
                rot_y: 0.0,
                scale: [1.0, 1.0, 1.0],
                tint_r: 0.92,
                tint_g: 0.92,
                tint_b: 0.92,
                uv_scale: 1.0,
                flags: 0,
            },
        );
    }

    // 5. Distant Bayfront Skyline HLOD closing left (+X across water) & forward (+Z bay peninsula) horizon
    for i in 0..22usize {
        let z = -40.0 + (i as f32) * 20.0;
        let x = 185.0 - (i as f32 - 8.0).max(0.0) * 4.5 + ((i * 19 % 5) as f32) * 6.0;
        let h = 28.0 + ((i * 23 % 7) as f32) * 11.0;
        let w = 14.0 + ((i * 7 % 4) as f32) * 3.5;
        total_tris_acc += push_inst(
            PROTO_HLOD_TOWER,
            AaaMeshletInstance {
                pos: [x, -0.5, z],
                rot_y: ((i % 4) as f32) * 0.25,
                scale: [w, h, w],
                tint_r: 0.95,
                tint_g: 0.88,
                tint_b: 0.92,
                uv_scale: 2.0,
                flags: 1,
            },
        );
    }
    // Far peninsula skyline clusters straight ahead (+Z >= 250)
    for i in 0..14usize {
        let x = -18.0 + (i as f32) * 14.0;
        let z = 258.0 + ((i * 11 % 4) as f32) * 14.0;
        let h = 32.0 + ((i * 29 % 6) as f32) * 12.0;
        total_tris_acc += push_inst(
            PROTO_HLOD_TOWER,
            AaaMeshletInstance {
                pos: [x, -0.5, z],
                rot_y: 0.12 * (i as f32),
                scale: [13.0, h, 13.0],
                tint_r: 0.92,
                tint_g: 0.86,
                tint_b: 0.90,
                uv_scale: 2.0,
                flags: 1,
            },
        );
    }

    // 6. Deep Urban Downtown Skyscraper Rows behind front row (-X = -52..-180)
    // Generates >= 12.2M total pre-cull triangles for live GPU Hi-Z occlusion culling!
    let mut row = 0usize;
    while total_tris_acc < 12_200_000 && row <= 40 {
        let x_base = -52.0 - (row as f32) * 20.0;
        for j in 0..22usize {
            let z = -110.0 + (j as f32) * 18.0 + (row % 2) as f32 * 8.0;
            let h = 26.0 + (((row * 7 + j * 13) % 7) as f32) * 9.0;
            let w = 13.0 + ((j % 3) as f32) * 2.0;
            let t = tints[(row + j) % tints.len()];
            let proto = if ((row + j) % 4) == 0 { PROTO_HOTEL } else { PROTO_SKYSCRAPER };
            total_tris_acc += push_inst(
                proto,
                AaaMeshletInstance {
                    pos: [x_base, 0.0, z],
                    rot_y: 0.0,
                    scale: [w, h, w],
                    tint_r: t[0],
                    tint_g: t[1],
                    tint_b: t[2],
                    uv_scale: 2.5,
                    flags: 0,
                },
            );
        }
        row += 1;
    }

    let total_tris: usize = work_items
        .iter()
        .map(|w| lib.meshlets[w.meshlet_idx as usize].prim_count as usize)
        .sum();

    CoastalCityScene {
        instances,
        work_items,
        total_tris,
        dxr_tower_boxes,
    }
}

pub struct AaaHiZPyramid {
    pub width: u32,
    pub height: u32,
    pub mip_dims: [(u32, u32); 6],
    mips: [ID3D12Resource; 6],
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    srv_inc: u32,
}

impl AaaHiZPyramid {
    pub unsafe fn new(
        device: &ID3D12Device,
        render_w: u32,
        render_h: u32,
        depth_tex: &ID3D12Resource,
        main_srv_heap: &ID3D12DescriptorHeap,
        srv_inc: u32,
    ) -> Self {
        let mut mw = (render_w / 2).max(16);
        let mut mh = (render_h / 2).max(16);
        let mut mip_dims = [(0u32, 0u32); 6];
        let mips: [ID3D12Resource; 6] = std::array::from_fn(|i| {
            mip_dims[i] = (mw, mh);
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: mw as u64,
                Height: mh,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    None,
                    &mut res,
                )
                .expect("HiZ mip tex");
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
            res.unwrap()
        });

        let srv_base = main_srv_heap.GetCPUDescriptorHandleForHeapStart();
        let cpu_at = |slot: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_base.ptr + slot * srv_inc as usize,
        };
        let mut srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        srv_desc.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() };

        device.CreateShaderResourceView(depth_tex, Some(&srv_desc), cpu_at(AAA_SRV_DEPTH_R32));

        let mut uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        uav_desc.Anonymous.Texture2D = D3D12_TEX2D_UAV::default();

        for i in 0..6 {
            device.CreateShaderResourceView(&mips[i], Some(&srv_desc), cpu_at(AAA_SRV_HIZ_BASE + i));
            device.CreateUnorderedAccessView(&mips[i], None, Some(&uav_desc), cpu_at(AAA_UAV_HIZ_BASE + i));
        }

        let srv_rng = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
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
                    Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 },
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
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut blob = None;
        D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).unwrap();
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = device
            .CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()))
            .unwrap();

        let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: HIZ_BUILD_CS_DXIL.as_ptr() as *const c_void,
                BytecodeLength: HIZ_BUILD_CS_DXIL.len(),
            },
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        let pso: ID3D12PipelineState = device.CreateComputePipelineState(&pso_desc).unwrap();
        let _ = ManuallyDrop::take(&mut pso_desc.pRootSignature);

        Self {
            width: mip_dims[0].0,
            height: mip_dims[0].1,
            mip_dims,
            mips,
            root_sig,
            pso,
            srv_inc,
        }
    }

    pub unsafe fn build(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        depth_tex: &ID3D12Resource,
        main_srv_heap: &ID3D12DescriptorHeap,
    ) {
        let gpu_base = main_srv_heap.GetGPUDescriptorHandleForHeapStart();
        let gpu_at = |slot: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu_base.ptr + (slot * self.srv_inc as usize) as u64,
        };

        cmd.SetDescriptorHeaps(&[Some(main_srv_heap.clone())]);
        cmd.SetComputeRootSignature(&self.root_sig);
        cmd.SetPipelineState(&self.pso);

        // Pass 0: Downsample D32 depth -> HiZ Mip 0 using 2x2 conservative max reduction
        cmd_barriers(
            cmd,
            vec![
                transition_barrier(depth_tex, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.mips[0], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );
        let (w0, h0) = self.mip_dims[0];
        let cb0 = [w0, h0, 1u32, 0u32]; // g_Pass = 1 reduces 2x2 depth -> half-res HiZ0!
        cmd.SetComputeRoot32BitConstants(0, 4, cb0.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootDescriptorTable(1, gpu_at(AAA_SRV_DEPTH_R32));
        cmd.SetComputeRootDescriptorTable(2, gpu_at(AAA_UAV_HIZ_BASE));
        cmd.Dispatch(w0.div_ceil(8), h0.div_ceil(8), 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.mips[0]),
                transition_barrier(&self.mips[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(depth_tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE),
            ],
        );

        // Passes 1..5: 2x2 max reduction chain
        for m in 1..6 {
            cmd_barriers(
                cmd,
                vec![transition_barrier(&self.mips[m], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)],
            );
            let (wm, hm) = self.mip_dims[m];
            let cb = [wm, hm, 1u32, 0u32];
            cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
            cmd.SetComputeRootDescriptorTable(1, gpu_at(AAA_SRV_HIZ_BASE + m - 1));
            cmd.SetComputeRootDescriptorTable(2, gpu_at(AAA_UAV_HIZ_BASE + m));
            cmd.Dispatch(wm.div_ceil(8), hm.div_ceil(8), 1);
            cmd_barriers(
                cmd,
                vec![
                    uav_barrier(&self.mips[m]),
                    transition_barrier(&self.mips[m], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                ],
            );
        }
    }
}

pub struct AaaMeshletCity {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    _vb_buf: ID3D12Resource,
    _pb_buf: ID3D12Resource,
    _mb_buf: ID3D12Resource,
    _inst_buf: ID3D12Resource,
    _wb_buf: ID3D12Resource,
    cb_upload: [ID3D12Resource; 2],
    cull_stats_uav: [ID3D12Resource; 2],
    cull_stats_rb: [ID3D12Resource; 2],
    cull_stats_zero: ID3D12Resource,
    pub total_work_items: u32,
    pub total_tris: usize,
    pub instance_count: usize,
    pub dxr_tower_boxes: Vec<([f32; 3], [f32; 3], [f32; 3])>,
}

impl AaaMeshletCity {
    pub fn new(ctx: &mut GpuContext, srv_table_stride: usize) -> Option<Self> {
        let mut opts7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        unsafe {
            ctx.device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    &mut opts7 as *mut _ as *mut c_void,
                    size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>() as u32,
                )
                .ok()?;
        }
        if opts7.MeshShaderTier.0 < D3D12_MESH_SHADER_TIER_1.0 {
            return None;
        }

        let lib = build_aaa_meshlet_library();
        let city = build_coastal_city_instances(&lib);

        unsafe {
            let dev2: ID3D12Device2 = ctx.device.cast().ok()?;
            let pbr_rng = D3D12_DESCRIPTOR_RANGE1 {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: srv_table_stride as u32,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
                OffsetInDescriptorsFromTableStart: 0,
            };
            let hiz_rng = D3D12_DESCRIPTOR_RANGE1 {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 6,
                BaseShaderRegister: 0,
                RegisterSpace: 2,
                Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
                OffsetInDescriptorsFromTableStart: 0,
            };

            let srv_static = |reg: u32, space: u32, vis: D3D12_SHADER_VISIBILITY| {
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: reg,
                            RegisterSpace: space,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                        },
                    },
                    ShaderVisibility: vis,
                }
            };

            let params = [
                // 0: b0 AaaMeshFrameCB
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
                // 1: t0..t36 PBR table (space0)
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 { NumDescriptorRanges: 1, pDescriptorRanges: &pbr_rng },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                },
                // 2: u0 (space0) g_SfsFeedback
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                },
                // 3..7: t0..t4 (space1) Verts, Prims, Meshlets, Instances, WorkItems
                srv_static(0, 1, D3D12_SHADER_VISIBILITY_ALL),
                srv_static(1, 1, D3D12_SHADER_VISIBILITY_ALL),
                srv_static(2, 1, D3D12_SHADER_VISIBILITY_ALL),
                srv_static(3, 1, D3D12_SHADER_VISIBILITY_ALL),
                srv_static(4, 1, D3D12_SHADER_VISIBILITY_ALL),
                // 8: u0 (space1) g_CullStats
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: 1,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
                },
                // 9: t0..t5 (space2) Hi-Z Pyramid
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 { NumDescriptorRanges: 1, pDescriptorRanges: &hiz_rng },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
                },
            ];

            let static_samplers = [
                D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER_ANISOTROPIC,
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                    MipLODBias: 0.0,
                    MaxAnisotropy: 8,
                    ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
                    MinLOD: 0.0,
                    MaxLOD: D3D12_FLOAT32_MAX,
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                },
                D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                    MipLODBias: 0.0,
                    MaxAnisotropy: 1,
                    ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
                    MinLOD: 0.0,
                    MaxLOD: D3D12_FLOAT32_MAX,
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                },
            ];

            let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                    Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                        NumParameters: params.len() as u32,
                        pParameters: params.as_ptr(),
                        NumStaticSamplers: static_samplers.len() as u32,
                        pStaticSamplers: static_samplers.as_ptr(),
                        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                    },
                },
            };
            let mut rs_blob = None;
            D3D12SerializeVersionedRootSignature(&rs_desc, &mut rs_blob, None).ok()?;
            let rs_blob = rs_blob?;
            let root_sig: ID3D12RootSignature = ctx
                .device
                .CreateRootSignature(0, std::slice::from_raw_parts(rs_blob.GetBufferPointer() as *const u8, rs_blob.GetBufferSize()))
                .ok()?;

            let mut rt_formats = D3D12_RT_FORMAT_ARRAY::default();
            rt_formats.NumRenderTargets = 4;
            rt_formats.RTFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
            rt_formats.RTFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
            rt_formats.RTFormats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;
            rt_formats.RTFormats[3] = DXGI_FORMAT_R16G16_FLOAT;

            let mut stream = AaaMeshShaderPsoStream {
                root_sig: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
                    inner: ManuallyDrop::new(Some(root_sig.clone())),
                },
                as_shader: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
                    inner: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_MESHLET_AS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_MESHLET_AS_DXIL.len() },
                },
                ms_shader: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
                    inner: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_MESHLET_MS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_MESHLET_MS_DXIL.len() },
                },
                ps_shader: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
                    inner: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_MESHLET_PS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_MESHLET_PS_DXIL.len() },
                },
                blend: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,
                    inner: D3D12_BLEND_DESC {
                        RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC {
                            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                            ..Default::default()
                        }; 8],
                        ..Default::default()
                    },
                },
                sample_mask: PsoSubobject { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, inner: u32::MAX },
                rasterizer: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
                    inner: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: true.into(),
                        ..Default::default()
                    },
                },
                depth_stencil: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
                    inner: D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: true.into(),
                        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                        DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                        ..Default::default()
                    },
                },
                rtv_formats: PsoSubobject { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, inner: rt_formats },
                dsv_format: PsoSubobject { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, inner: DXGI_FORMAT_D32_FLOAT },
                sample_desc: PsoSubobject { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, inner: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 } },
            };
            let pso_desc = D3D12_PIPELINE_STATE_STREAM_DESC {
                SizeInBytes: size_of::<AaaMeshShaderPsoStream>(),
                pPipelineStateSubobjectStream: &mut stream as *mut _ as *mut c_void,
            };
            let pso: ID3D12PipelineState = dev2.CreatePipelineState(&pso_desc).ok()?;
            let _ = ManuallyDrop::take(&mut stream.root_sig.inner);

            let vb_bytes = (lib.vertices.len() * size_of::<AaaMeshletVertex>()) as u64;
            let pb_bytes = (lib.prim_indices.len() * size_of::<u32>()) as u64;
            let mb_bytes = (lib.meshlets.len() * size_of::<AaaMeshletDesc>()) as u64;
            let ib_bytes = (city.instances.len() * size_of::<AaaMeshletInstance>()) as u64;
            let wb_bytes = (city.work_items.len() * size_of::<AaaMeshletWorkItem>()) as u64;

            let vb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, vb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let pb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, pb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let mb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, mb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let inst_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, ib_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let wb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, wb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            upload_slice(&vb_up, &lib.vertices);
            upload_slice(&pb_up, &lib.prim_indices);
            upload_slice(&mb_up, &lib.meshlets);
            upload_slice(&inst_up, &city.instances);
            upload_slice(&wb_up, &city.work_items);

            let vb_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, vb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let pb_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, pb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let mb_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, mb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let inst_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, ib_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let wb_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, wb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let up_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
            let up_cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &up_alloc, None).ok()?;
            up_cmd.CopyBufferRegion(&vb_buf, 0, &vb_up, 0, vb_bytes);
            up_cmd.CopyBufferRegion(&pb_buf, 0, &pb_up, 0, pb_bytes);
            up_cmd.CopyBufferRegion(&mb_buf, 0, &mb_up, 0, mb_bytes);
            up_cmd.CopyBufferRegion(&inst_buf, 0, &inst_up, 0, ib_bytes);
            up_cmd.CopyBufferRegion(&wb_buf, 0, &wb_up, 0, wb_bytes);
            cmd_barriers(
                &up_cmd,
                vec![
                    transition_barrier(&vb_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                    transition_barrier(&pb_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                    transition_barrier(&mb_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                    transition_barrier(&inst_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                    transition_barrier(&wb_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                ],
            );
            up_cmd.Close().ok()?;
            let exec = [Some(up_cmd.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&exec);
            ctx.signal_and_wait();

            let cb_upload = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            ];
            let cull_stats_uav = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
            ];
            let cull_stats_rb = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST),
            ];
            let cull_stats_zero = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            upload_slice(&cull_stats_zero, &[0u32; 8]);

            Some(Self {
                root_sig,
                pso,
                _vb_buf: vb_buf,
                _pb_buf: pb_buf,
                _mb_buf: mb_buf,
                _inst_buf: inst_buf,
                _wb_buf: wb_buf,
                cb_upload,
                cull_stats_uav,
                cull_stats_rb,
                cull_stats_zero,
                total_work_items: city.work_items.len() as u32,
                total_tris: city.total_tris,
                instance_count: city.instances.len(),
                dxr_tower_boxes: city.dxr_tower_boxes,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn draw(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        slot: usize,
        vp_jittered: &[f32; 16],
        vp_unjittered: &[f32; 16],
        prev_vp_unjittered: &[f32; 16],
        cam_pos: [f32; 3],
        flags_bits: u32,
        sun_dir: [f32; 3],
        time: f32,
        hiz_enabled: bool,
        hiz_w: u32,
        hiz_h: u32,
        concrete_states: [f32; 3],
        concrete_tex: [u32; 3],
        pbr_table_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        hiz_table_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        sfs_uav_va: u64,
    ) {
        let s = slot & 1;
        let cb = AaaMeshFrameCB {
            vp_jittered: *vp_jittered,
            vp_unjittered: *vp_unjittered,
            prev_vp_unjittered: *prev_vp_unjittered,
            frustum_planes: extract_frustum_planes(vp_unjittered),
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], flags_bits as f32],
            sun_dir_time: [sun_dir[0], sun_dir[1], sun_dir[2], time],
            cull_params: [self.total_work_items, if hiz_enabled { 1 } else { 0 }, hiz_w, hiz_h],
            cull_range: [260.0, concrete_states[0], concrete_states[1], concrete_states[2]],
            concrete_tex: [concrete_tex[0], concrete_tex[1], concrete_tex[2], 0],
        };
        upload_slice(&self.cb_upload[s], std::slice::from_ref(&cb));

        cmd.CopyBufferRegion(&self.cull_stats_uav[s], 0, &self.cull_stats_zero, 0, 32);
        cmd_barriers(
            cmd,
            vec![transition_barrier(
                &self.cull_stats_uav[s],
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )],
        );

        cmd.SetGraphicsRootSignature(&self.root_sig);
        cmd.SetPipelineState(&self.pso);
        cmd.SetGraphicsRootConstantBufferView(0, self.cb_upload[s].GetGPUVirtualAddress());
        cmd.SetGraphicsRootDescriptorTable(1, pbr_table_gpu);
        cmd.SetGraphicsRootUnorderedAccessView(2, sfs_uav_va);
        cmd.SetGraphicsRootShaderResourceView(3, self._vb_buf.GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(4, self._pb_buf.GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(5, self._mb_buf.GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(6, self._inst_buf.GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(7, self._wb_buf.GetGPUVirtualAddress());
        cmd.SetGraphicsRootUnorderedAccessView(8, self.cull_stats_uav[s].GetGPUVirtualAddress());
        cmd.SetGraphicsRootDescriptorTable(9, hiz_table_gpu);

        let cmd6: ID3D12GraphicsCommandList6 = cmd.cast().unwrap();
        cmd6.DispatchMesh(self.total_work_items.div_ceil(32), 1, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.cull_stats_uav[s]),
                transition_barrier(&self.cull_stats_uav[s], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            ],
        );
        cmd.CopyBufferRegion(&self.cull_stats_rb[s], 0, &self.cull_stats_uav[s], 0, 32);
        cmd_barriers(
            cmd,
            vec![transition_barrier(&self.cull_stats_uav[s], D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST)],
        );
    }

    pub unsafe fn read_cull_stats(&self, slot: usize) -> CityCullStats {
        let s = slot & 1;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rng = D3D12_RANGE { Begin: 0, End: 20 };
        self.cull_stats_rb[s].Map(0, Some(&rng), Some(&mut ptr)).unwrap();
        let vals = std::slice::from_raw_parts(ptr as *const u32, 5);
        let st = CityCullStats {
            vis_meshlets: vals[0],
            vis_tris: vals[1],
            frustum_culled: vals[2],
            hiz_occluded: vals[3],
            hiz_occluded_tris: vals[4],
        };
        self.cull_stats_rb[s].Unmap(0, None);
        st
    }
}

// ---------------- Live 500 GPU-Skinned Pedestrian Crowd ----------------
pub const CROWD_NPCS: usize = 500;
const CROWD_BONES: usize = 16;

#[repr(C)]
#[derive(Clone, Copy)]
struct CrowdDrawCB {
    vp_jittered: [f32; 16],
    vp_unjittered: [f32; 16],
    prev_vp_unjittered: [f32; 16],
    camera_pos: [f32; 4],
    sun_dir_time: [f32; 4],
    crowd_params: [u32; 4],
}

#[derive(Clone, Copy)]
struct PedAgent {
    x: f32,
    z0: f32,
    speed: f32,
    yaw: f32,
    scale: f32,
    phase: f32,
}

fn build_humanoid_template() -> (Vec<BaseSkinnedVertex>, Vec<u32>, Vec<u32>) {
    let mut verts: Vec<BaseSkinnedVertex> = Vec::new();
    let mut part_ids: Vec<u32> = Vec::new();
    let mut inds: Vec<u32> = Vec::new();

    // Helper: push a subdivided tapered limb/torso box weighted to primary + secondary bone
    let mut push_part = |center: [f32; 3], size: [f32; 3], b0: u32, b1: u32, part: u32| {
        let rings = 3usize;
        let segs = 6usize;
        let base = verts.len() as u32;
        for r in 0..=rings {
            let v = r as f32 / rings as f32;
            let y = center[1] + (v - 0.5) * size[1];
            let w0 = 1.0 - v * 0.25;
            let w1 = v * 0.25;
            for s in 0..=segs {
                let ang = (s as f32 / segs as f32) * std::f32::consts::TAU;
                let (sa, ca) = ang.sin_cos();
                let px = center[0] + ca * size[0] * 0.5;
                let pz = center[2] + sa * size[2] * 0.5;
                let nl = (ca * ca + sa * sa).sqrt().max(1e-4);
                verts.push(BaseSkinnedVertex {
                    pos: [px, y, pz, 1.0],
                    normal: [ca / nl, 0.15, sa / nl, 0.0],
                    bone_idx: [b0, b1, b0, b0],
                    bone_wt: [w0, w1, 0.0, 0.0],
                });
                part_ids.push(part);
            }
        }
        let stride = (segs + 1) as u32;
        for r in 0..rings as u32 {
            for s in 0..segs as u32 {
                let i00 = base + r * stride + s;
                let i10 = i00 + 1;
                let i01 = i00 + stride;
                let i11 = i01 + 1;
                inds.extend_from_slice(&[i00, i01, i11, i00, i11, i10]);
            }
        }
    };

    // Bones: 0=root/torso, 1=head, 2=L_leg, 3=R_leg, 4=L_arm, 5=R_arm
    push_part([0.0, 1.12, 0.0], [0.40, 0.56, 0.22], 0, 1, 1); // Torso/Jacket
    push_part([0.0, 1.56, 0.0], [0.22, 0.24, 0.22], 1, 0, 0); // Head
    push_part([0.0, 1.66, -0.02], [0.24, 0.08, 0.24], 1, 0, 3); // Hair
    push_part([-0.11, 0.48, 0.0], [0.15, 0.72, 0.16], 2, 0, 2); // Left Leg
    push_part([0.11, 0.48, 0.0], [0.15, 0.72, 0.16], 3, 0, 2); // Right Leg
    push_part([-0.11, 0.07, 0.04], [0.14, 0.12, 0.22], 2, 0, 3); // Left Shoe
    push_part([0.11, 0.07, 0.04], [0.14, 0.12, 0.22], 3, 0, 3); // Right Shoe
    push_part([-0.28, 1.06, 0.0], [0.12, 0.52, 0.12], 4, 0, 1); // Left Arm
    push_part([0.28, 1.06, 0.0], [0.12, 0.52, 0.12], 5, 0, 1); // Right Arm
    push_part([-0.28, 0.76, 0.0], [0.10, 0.12, 0.10], 4, 0, 0); // Left Hand
    push_part([0.28, 0.76, 0.0], [0.10, 0.12, 0.10], 5, 0, 0); // Right Hand

    (verts, inds, part_ids)
}

pub struct AaaCrowdRenderer {
    skin_root_sig: ID3D12RootSignature,
    skin_pso: ID3D12PipelineState,
    draw_root_sig: ID3D12RootSignature,
    draw_pso: ID3D12PipelineState,
    _base_verts_def: ID3D12Resource,
    _part_ids_def: ID3D12Resource,
    _ib_def: ID3D12Resource,
    bones_upload: [ID3D12Resource; 2],
    bones_def: ID3D12Resource,
    skinned_verts: [ID3D12Resource; 2],
    cb_upload: [ID3D12Resource; 2],
    verts_per_npc: u32,
    idx_count: u32,
    agents: Vec<PedAgent>,
    bones_cpu: Vec<BoneMat>,
}

impl AaaCrowdRenderer {
    pub fn new(ctx: &mut GpuContext, srv_table_stride: usize) -> Self {
        let (base_verts, inds, part_ids) = build_humanoid_template();
        let verts_per_npc = base_verts.len() as u32;
        let idx_count = inds.len() as u32;

        let mut agents = Vec::with_capacity(CROWD_NPCS);
        for i in 0..CROWD_NPCS {
            let side = if (i & 1) == 0 { -1.0f32 } else { 1.0f32 };
            let lane = ((i / 2) % 4) as f32;
            let x = side * (14.2 + lane * 1.35 + ((i * 17 % 7) as f32) * 0.12);
            let z0 = -75.0 + ((i / 2) as f32) * 1.15;
            let fwd = ((i / 3) & 1) == 0;
            let speed = (1.1 + (i % 5) as f32 * 0.18) * if fwd { 1.0 } else { -1.0 };
            let yaw = if fwd { 0.0 } else { std::f32::consts::PI };
            let scale = 0.92 + (i % 7) as f32 * 0.025;
            agents.push(PedAgent {
                x,
                z0,
                speed,
                yaw,
                scale,
                phase: (i as f32) * 0.73,
            });
        }

        unsafe {
            // Skinning compute root sig + PSO
            let skin_params = [
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 1, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            ];
            let srs_desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: skin_params.len() as u32, pParameters: skin_params.as_ptr(), NumStaticSamplers: 0, pStaticSamplers: std::ptr::null(), Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE };
            let mut sblob = None;
            D3D12SerializeRootSignature(&srs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut sblob, None).unwrap();
            let sblob = sblob.unwrap();
            let skin_root_sig: ID3D12RootSignature = ctx.device.CreateRootSignature(0, std::slice::from_raw_parts(sblob.GetBufferPointer() as *const u8, sblob.GetBufferSize())).unwrap();
            let mut spso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(skin_root_sig.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: CROWD_SKIN_CS_DXIL.as_ptr() as *const c_void, BytecodeLength: CROWD_SKIN_CS_DXIL.len() },
                NodeMask: 0,
                CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
                Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
            };
            let skin_pso: ID3D12PipelineState = ctx.device.CreateComputePipelineState(&spso_desc).unwrap();
            let _ = ManuallyDrop::take(&mut spso_desc.pRootSignature);

            // Crowd G-Buffer Graphics root sig + PSO
            let pbr_rng = D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: srv_table_stride as u32,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 0,
            };
            let draw_params = [
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &pbr_rng } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL },
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 1 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX },
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 1, RegisterSpace: 1 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX },
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 2, RegisterSpace: 1 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX },
            ];
            let draw_samplers = [
                D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER_ANISOTROPIC, AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP, AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP, AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                    MipLODBias: 0.0, MaxAnisotropy: 8, ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS, BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK, MinLOD: 0.0, MaxLOD: D3D12_FLOAT32_MAX, ShaderRegister: 0, RegisterSpace: 0, ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                },
                D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR, AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP, AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                    MipLODBias: 0.0, MaxAnisotropy: 1, ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS, BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK, MinLOD: 0.0, MaxLOD: D3D12_FLOAT32_MAX, ShaderRegister: 1, RegisterSpace: 0, ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                },
            ];
            let drs_desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: draw_params.len() as u32,
                pParameters: draw_params.as_ptr(),
                NumStaticSamplers: draw_samplers.len() as u32,
                pStaticSamplers: draw_samplers.as_ptr(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            };
            let mut dblob = None;
            D3D12SerializeRootSignature(&drs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut dblob, None).unwrap();
            let dblob = dblob.unwrap();
            let draw_root_sig: ID3D12RootSignature = ctx.device.CreateRootSignature(0, std::slice::from_raw_parts(dblob.GetBufferPointer() as *const u8, dblob.GetBufferSize())).unwrap();

            let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
            rtv_formats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
            rtv_formats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
            rtv_formats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;
            rtv_formats[3] = DXGI_FORMAT_R16G16_FLOAT;
            let mut dpso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(draw_root_sig.clone())),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_CROWD_VS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_CROWD_VS_DXIL.len() },
                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: AAA_CROWD_PS_DXIL.as_ptr() as *const c_void, BytecodeLength: AAA_CROWD_PS_DXIL.len() },
                BlendState: D3D12_BLEND_DESC {
                    RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8, ..Default::default() }; 8],
                    ..Default::default()
                },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC { DepthEnable: true.into(), DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL, DepthFunc: D3D12_COMPARISON_FUNC_LESS, ..Default::default() },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 4,
                RTVFormats: rtv_formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let draw_pso: ID3D12PipelineState = ctx.device.CreateGraphicsPipelineState(&dpso_desc).unwrap();
            let _ = ManuallyDrop::take(&mut dpso_desc.pRootSignature);

            // Buffers
            let bv_bytes = (base_verts.len() * size_of::<BaseSkinnedVertex>()) as u64;
            let pid_bytes = (part_ids.len() * size_of::<u32>()) as u64;
            let ib_bytes = (inds.len() * size_of::<u32>()) as u64;
            let bm_bytes = (CROWD_NPCS * CROWD_BONES * size_of::<BoneMat>()) as u64;
            let sv_bytes = (CROWD_NPCS * verts_per_npc as usize * size_of::<OutSkinnedVertex>()) as u64;

            let bv_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, bv_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let pid_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, pid_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let ib_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, ib_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            upload_slice(&bv_up, &base_verts);
            upload_slice(&pid_up, &part_ids);
            upload_slice(&ib_up, &inds);

            let base_verts_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, bv_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let part_ids_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, pid_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let ib_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, ib_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let up_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let up_cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &up_alloc, None).unwrap();
            up_cmd.CopyBufferRegion(&base_verts_def, 0, &bv_up, 0, bv_bytes);
            up_cmd.CopyBufferRegion(&part_ids_def, 0, &pid_up, 0, pid_bytes);
            up_cmd.CopyBufferRegion(&ib_def, 0, &ib_up, 0, ib_bytes);
            cmd_barriers(
                &up_cmd,
                vec![
                    transition_barrier(&base_verts_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                    transition_barrier(&part_ids_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                    transition_barrier(&ib_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER),
                ],
            );
            up_cmd.Close().unwrap();
            let lists = [Some(up_cmd.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait();

            let bones_upload = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, bm_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, bm_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            ];
            let bones_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, bm_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let skinned_verts = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, sv_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, sv_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ];
            let cb_upload = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            ];

            let mut renderer = Self {
                skin_root_sig,
                skin_pso,
                draw_root_sig,
                draw_pso,
                _base_verts_def: base_verts_def,
                _part_ids_def: part_ids_def,
                _ib_def: ib_def,
                bones_upload,
                bones_def,
                skinned_verts,
                cb_upload,
                verts_per_npc,
                idx_count,
                agents,
                bones_cpu: vec![BoneMat { c0: [1.0, 0.0, 0.0, 0.0], c1: [0.0, 1.0, 0.0, 0.0], c2: [0.0, 0.0, 1.0, 0.0], c3: [0.0, 0.0, 0.0, 1.0] }; CROWD_NPCS * CROWD_BONES],
            };

            // Initialize both skinned vertex buffers so frame 0 has valid prevVerts for TAA motion vectors
            let init_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let init_cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &init_alloc, None).unwrap();
            renderer.update_and_skin(&init_cmd, 0, 0.0);
            renderer.update_and_skin(&init_cmd, 1, 0.0);
            init_cmd.Close().unwrap();
            let ilists = [Some(init_cmd.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&ilists);
            ctx.signal_and_wait();

            renderer
        }
    }

    pub fn verts_total(&self) -> usize {
        CROWD_NPCS * (self.verts_per_npc as usize)
    }

    #[inline]
    fn bone_transform(px: f32, py: f32, pz: f32, yaw: f32, pitch_x: f32, pivot_y: f32, scale: f32) -> BoneMat {
        // Rotate local (x, y-pivot, z) around X by pitch_x, scale, rotate around Y by yaw, translate to world (px, py+pivot*scale, pz)
        let (sx, cx) = pitch_x.sin_cos();
        let (sy, cy) = yaw.sin_cos();
        // M_y * M_x
        let c0 = [cy * scale, 0.0, -sy * scale, 0.0];
        let c1 = [sy * sx * scale, cx * scale, cy * sx * scale, 0.0];
        let c2 = [sy * cx * scale, -sx * scale, cy * cx * scale, 0.0];
        // pivot offset correction: translate(p + (0, pivot, 0)) * R * translate(0, -pivot, 0)
        let tx = px - c1[0] * pivot_y;
        let ty = py + pivot_y * scale - c1[1] * pivot_y;
        let tz = pz - c1[2] * pivot_y;
        BoneMat { c0, c1, c2, c3: [tx, ty, tz, 1.0] }
    }

    pub unsafe fn update_and_skin(&mut self, cmd: &ID3D12GraphicsCommandList, slot: usize, t_sec: f32) {
        let s = slot & 1;
        for (ni, a) in self.agents.iter().enumerate() {
            let z = ((a.z0 + t_sec * a.speed + 85.0).rem_euclid(260.0)) - 85.0;
            let stride = t_sec * a.speed.abs() * 3.8 + a.phase;
            let swing = stride.sin() * 0.42;
            let bob = stride.cos().abs() * 0.035;
            let base = ni * CROWD_BONES;
            let torso = Self::bone_transform(a.x, bob, z, a.yaw, 0.0, 0.0, a.scale);
            self.bones_cpu[base] = torso;
            self.bones_cpu[base + 1] = Self::bone_transform(a.x, bob, z, a.yaw, swing * 0.08, 1.45, a.scale);
            self.bones_cpu[base + 2] = Self::bone_transform(a.x, bob, z, a.yaw, swing, 0.85, a.scale);
            self.bones_cpu[base + 3] = Self::bone_transform(a.x, bob, z, a.yaw, -swing, 0.85, a.scale);
            self.bones_cpu[base + 4] = Self::bone_transform(a.x, bob, z, a.yaw, -swing * 0.85, 1.32, a.scale);
            self.bones_cpu[base + 5] = Self::bone_transform(a.x, bob, z, a.yaw, swing * 0.85, 1.32, a.scale);
        }

        upload_slice(&self.bones_upload[s], &self.bones_cpu);
        let bm_bytes = (self.bones_cpu.len() * size_of::<BoneMat>()) as u64;
        cmd.CopyBufferRegion(&self.bones_def, 0, &self.bones_upload[s], 0, bm_bytes);
        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.bones_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.skinned_verts[s], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );

        cmd.SetComputeRootSignature(&self.skin_root_sig);
        cmd.SetPipelineState(&self.skin_pso);
        let cb = [CROWD_NPCS as u32, self.verts_per_npc, CROWD_BONES as u32, 0u32];
        cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, self._base_verts_def.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, self.bones_def.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(3, self.skinned_verts[s].GetGPUVirtualAddress());
        let total_v = CROWD_NPCS as u32 * self.verts_per_npc;
        cmd.Dispatch(total_v.div_ceil(64), 1, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.skinned_verts[s]),
                transition_barrier(&self.skinned_verts[s], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.bones_def, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
            ],
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn draw(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        slot: usize,
        vp_jittered: &[f32; 16],
        vp_unjittered: &[f32; 16],
        prev_vp_unjittered: &[f32; 16],
        cam_pos: [f32; 3],
        flags_bits: u32,
        sun_dir: [f32; 3],
        time: f32,
        pbr_table_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) {
        let s = slot & 1;
        let prev_s = s ^ 1;
        let cb = CrowdDrawCB {
            vp_jittered: *vp_jittered,
            vp_unjittered: *vp_unjittered,
            prev_vp_unjittered: *prev_vp_unjittered,
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], flags_bits as f32],
            sun_dir_time: [sun_dir[0], sun_dir[1], sun_dir[2], time],
            crowd_params: [self.verts_per_npc, CROWD_NPCS as u32, flags_bits, 0],
        };
        upload_slice(&self.cb_upload[s], std::slice::from_ref(&cb));

        let ib_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: self._ib_def.GetGPUVirtualAddress(),
            SizeInBytes: self.idx_count * 4,
            Format: DXGI_FORMAT_R32_UINT,
        };

        cmd.SetGraphicsRootSignature(&self.draw_root_sig);
        cmd.SetPipelineState(&self.draw_pso);
        cmd.SetGraphicsRootConstantBufferView(0, self.cb_upload[s].GetGPUVirtualAddress());
        cmd.SetGraphicsRootDescriptorTable(1, pbr_table_gpu);
        cmd.SetGraphicsRootShaderResourceView(2, self.skinned_verts[s].GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(3, self.skinned_verts[prev_s].GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(4, self._part_ids_def.GetGPUVirtualAddress());
        cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd.IASetIndexBuffer(Some(&ib_view));
        cmd.DrawIndexedInstanced(self.idx_count, CROWD_NPCS as u32, 0, 0, 0);
    }
}
