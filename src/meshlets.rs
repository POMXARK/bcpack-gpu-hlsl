use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, drop_barriers, heap_props, transition_barrier,
    uav_barrier, upload_slice, GpuContext,
};
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop};
use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub const MESHLET_AS_DXIL: &[u8] = include_bytes!("../shaders/meshlet_as.dxil");
pub const MESHLET_MS_DXIL: &[u8] = include_bytes!("../shaders/meshlet_ms.dxil");
pub const MESHLET_PS_DXIL: &[u8] = include_bytes!("../shaders/meshlet_ps.dxil");

pub const MAX_MESHLET_INSTANCES: usize = 1024;
pub const PROTO_SKYSCRAPER: usize = 0;
pub const PROTO_HOTEL: usize = 1;
pub const PROTO_CAR: usize = 2;
pub const PROTO_PALM: usize = 3;
pub const PROTO_LAMP: usize = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshletVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshletDesc {
    pub vert_offset: u32,
    pub vert_count: u32,
    pub prim_offset: u32,
    pub prim_count: u32,
    pub bound_center: [f32; 3],
    pub bound_radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshletInstance {
    pub pos: [f32; 3],
    pub rot_y: f32,
    pub scale: [f32; 3],
    pub tex_index: u32,
    pub emissive: f32,
    pub uv_scale: f32,
    pub spec_power: f32,
    pub pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshletWorkItem {
    pub inst_idx: u32,
    pub meshlet_idx: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshFrameCB {
    pub view_proj: [f32; 16],
    pub frustum_planes: [[f32; 4]; 6],
    pub camera_pos: [f32; 4],
    pub point_lights: [[f32; 4]; 4],
    pub light_colors: [[f32; 4]; 4],
    pub total_work_items: u32,
    pub time: f32,
    pub max_cull_dist: f32,
    pub pad: u32,
}

#[repr(C, align(8))]
struct PsoSubobject<T> {
    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    inner: T,
}

#[repr(C)]
struct MeshShaderPsoStream {
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

pub struct MeshletLibrary {
    pub vertices: Vec<MeshletVertex>,
    pub prim_indices: Vec<u32>,
    pub meshlets: Vec<MeshletDesc>,
    pub proto_ranges: [(u32, u32); 5],
}

struct MeshletBuilder {
    vertices: Vec<MeshletVertex>,
    prim_indices: Vec<u32>,
    meshlets: Vec<MeshletDesc>,
}

impl MeshletBuilder {
    fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(65536),
            prim_indices: Vec::with_capacity(65536),
            meshlets: Vec::with_capacity(1024),
        }
    }

    #[inline]
    fn norm3(v: [f32; 3]) -> [f32; 3] {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        [v[0] / l, v[1] / l, v[2] / l]
    }

    /// Emits a `cols x rows` subdivided grid patch split into hardware meshlets (max 7x7 quads = 64 verts, 98 tris each).
    fn push_patch<F>(&mut self, total_cols: usize, total_rows: usize, eval: F)
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
                        self.vertices.push(MeshletVertex { pos, normal: n, uv });
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
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 > max_r2 {
                        max_r2 = d2;
                    }
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

                self.meshlets.push(MeshletDesc {
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

    fn push_subdiv_box(&mut self, center: [f32; 3], size: [f32; 3], sub_u: usize, sub_v: usize, uv_scale: [f32; 2]) {
        let hx = size[0] * 0.5;
        let hy = size[1] * 0.5;
        let hz = size[2] * 0.5;
        let [cx, cy, cz] = center;

        let faces: [([f32; 3], [f32; 3], [f32; 3], [f32; 3]); 6] = [
            // origin, axis_u, axis_v, normal
            ([cx - hx, cy - hy, cz + hz], [size[0], 0.0, 0.0], [0.0, size[1], 0.0], [0.0, 0.0, 1.0]),
            ([cx + hx, cy - hy, cz - hz], [-size[0], 0.0, 0.0], [0.0, size[1], 0.0], [0.0, 0.0, -1.0]),
            ([cx + hx, cy - hy, cz + hz], [0.0, 0.0, -size[2]], [0.0, size[1], 0.0], [1.0, 0.0, 0.0]),
            ([cx - hx, cy - hy, cz - hz], [0.0, 0.0, size[2]], [0.0, size[1], 0.0], [-1.0, 0.0, 0.0]),
            ([cx - hx, cy + hy, cz + hz], [size[0], 0.0, 0.0], [0.0, 0.0, -size[2]], [0.0, 1.0, 0.0]),
            ([cx - hx, cy - hy, cz - hz], [size[0], 0.0, 0.0], [0.0, 0.0, size[2]], [0.0, -1.0, 0.0]),
        ];

        for (o, au, av, n) in faces {
            self.push_patch(sub_u, sub_v, |u, v| {
                let p = [
                    o[0] + au[0] * u + av[0] * v,
                    o[1] + au[1] * u + av[1] * v,
                    o[2] + au[2] * u + av[2] * v,
                ];
                (p, n, [u * uv_scale[0], (1.0 - v) * uv_scale[1]])
            });
        }
    }

    fn build_skyscraper(&mut self) {
        // Unit-normalized Art-Deco stepped tower (~9,072 tris / 108 meshlets):
        self.push_subdiv_box([0.0, -0.35, 0.0], [1.06, 0.30, 1.06], 14, 7, [1.5, 1.0]);
        self.push_subdiv_box([0.0, 0.02, 0.0], [0.96, 0.52, 0.96], 21, 14, [2.0, 2.0]);
        self.push_subdiv_box([0.0, 0.36, 0.0], [0.72, 0.22, 0.72], 14, 14, [1.5, 1.0]);
        self.push_subdiv_box([0.0, 0.51, 0.0], [0.36, 0.14, 0.36], 14, 7, [1.0, 1.0]);
        for &(sx, sz) in &[(-0.48f32, -0.48f32), (0.48, -0.48), (-0.48, 0.48), (0.48, 0.48)] {
            self.push_subdiv_box([sx, 0.0, sz], [0.08, 0.88, 0.08], 2, 7, [0.5, 2.0]);
        }
    }

    fn build_hotel(&mut self) {
        // Ocean Drive Pastel Hotel (~6,888 tris):
        self.push_subdiv_box([0.0, -0.34, 0.0], [1.04, 0.32, 1.04], 14, 7, [1.5, 1.0]);
        self.push_subdiv_box([0.0, 0.08, 0.0], [0.94, 0.62, 0.94], 21, 14, [2.0, 2.0]);
        for tier in 0..4 {
            let y = -0.10 + tier as f32 * 0.16;
            self.push_subdiv_box([0.0, y, 0.0], [1.02, 0.04, 1.02], 14, 2, [1.5, 0.2]);
        }
        self.push_subdiv_box([0.0, 0.44, 0.0], [0.64, 0.12, 0.64], 7, 7, [1.0, 0.5]);
    }

    fn build_supercar(&mut self) {
        // Smooth lofted 80s Vice-City Wedge Supercar (~7,200 tris across body + cabin + spoiler + 4 spoked wheels)
        // Main swept upper hull profile (z from -2.18 to +2.18)
        let profile = |u: f32, z_norm: f32| -> ([f32; 3], [f32; 3], [f32; 2]) {
            let z = -2.18 + z_norm * 4.36;
            let (half_w, y_top) = if z_norm < 0.25 {
                let t = z_norm / 0.25;
                (0.95 + 0.03 * t, 0.84 + 0.04 * t)
            } else if z_norm < 0.64 {
                let t = (z_norm - 0.25) / 0.39;
                let cab = (t * std::f32::consts::PI).sin();
                (0.96 - cab * 0.14, 0.86 + cab * 0.48)
            } else {
                let t = (z_norm - 0.64) / 0.36;
                (0.96 - t * 0.10, 0.84 - t * 0.30)
            };
            let x = (u * 2.0 - 1.0) * half_w;
            let arch = (1.0 - (u * 2.0 - 1.0).powi(2)) * 0.08;
            let y = y_top + arch;
            let nx = (u * 2.0 - 1.0) * 0.32;
            let nz = if z_norm > 0.64 { 0.25 } else { -0.05 };
            ([x, y, z], [nx, 1.0, nz], [u, z_norm])
        };
        self.push_patch(21, 21, profile);

        // Side skirts & sculpted fenders
        self.push_subdiv_box([0.0, 0.52, 0.0], [1.92, 0.54, 4.30], 14, 14, [1.0, 1.0]);
        self.push_subdiv_box([0.0, 1.12, -1.96], [1.94, 0.07, 0.38], 14, 4, [1.0, 0.25]);
        self.push_subdiv_box([-0.68, 0.96, -1.94], [0.09, 0.28, 0.20], 4, 4, [0.2, 0.2]);
        self.push_subdiv_box([0.68, 0.96, -1.94], [0.09, 0.28, 0.20], 4, 4, [0.2, 0.2]);

        // 4 cylindrical high-poly wheels
        for &(wx, wz) in &[(-0.92f32, 1.32f32), (0.92, 1.32), (-0.94, -1.32), (0.94, -1.32)] {
            self.push_patch(21, 7, |u, v| {
                let ang = u * std::f32::consts::TAU;
                let r = 0.33f32;
                let x = wx + (v - 0.5) * 0.28;
                let y = 0.33 + ang.sin() * r;
                let z = wz + ang.cos() * r;
                ([x, y, z], [0.0, ang.sin(), ang.cos()], [0.76 + u * 0.20, 0.76 + v * 0.20])
            });
            // Hubcap discs
            let side_x = if wx < 0.0 { wx - 0.14 } else { wx + 0.14 };
            let nx = if wx < 0.0 { -1.0 } else { 1.0 };
            self.push_patch(14, 4, |u, v| {
                let ang = u * std::f32::consts::TAU;
                let r = v * 0.31;
                ([side_x, 0.33 + ang.sin() * r, wz + ang.cos() * r], [nx, 0.0, 0.0], [0.80, 0.80])
            });
        }
    }

    fn build_palm(&mut self) {
        // Curved cylindrical segmented trunk + 12 lush arched palm fronds (~3,600 tris)
        self.push_patch(14, 14, |u, v| {
            let ang = u * std::f32::consts::TAU;
            let t = v;
            let y = t * 5.8;
            let lean_x = t * t * 0.85;
            let r = 0.29 * (1.0 - t * 0.54) * (1.0 + 0.06 * (t * 28.0).sin());
            let px = lean_x + ang.cos() * r;
            let pz = ang.sin() * r;
            ([px, y, pz], [ang.cos(), 0.1, ang.sin()], [u * 2.0, t * 4.0])
        });

        let fronds = 12usize;
        for f in 0..fronds {
            let ang = (f as f32 / fronds as f32) * std::f32::consts::TAU + (f as f32) * 0.19;
            let (sa, ca) = ang.sin_cos();
            let right = [-sa, 0.0, ca];
            let fwd = [ca, 0.0, sa];
            let droop = if f % 2 == 0 { 1.25 } else { 0.85 };
            self.push_patch(4, 14, |u, v| {
                let t = v;
                let r = t * 3.1;
                let y = 5.75 + (t * 1.45).sin() * 0.72 - t * t * droop;
                let w = (1.0 - t * 0.76) * 0.72 * (u - 0.5) * 2.0;
                let px = 0.85 + fwd[0] * r + right[0] * w;
                let pz = fwd[2] * r + right[2] * w;
                let py = y - (u - 0.5).abs() * 0.18;
                ([px, py, pz], [0.0, 1.0, 0.0], [u, v])
            });
        }
    }

    fn build_lamp(&mut self) {
        // Fluted streetlamp post + curved neck + glowing luminaire head (~1,700 tris)
        self.push_patch(14, 14, |u, v| {
            let ang = u * std::f32::consts::TAU;
            let t = v;
            let y = t * 6.8;
            let r = 0.14 * (1.0 - t * 0.40);
            ([ang.cos() * r, y, ang.sin() * r], [ang.cos(), 0.0, ang.sin()], [u, v * 3.0])
        });
        self.push_patch(7, 7, |u, v| {
            let ang = u * std::f32::consts::TAU;
            let x = v * 1.95;
            let y = 6.75 + (v * std::f32::consts::PI).sin() * 0.35;
            let r = 0.08;
            ([x, y + ang.sin() * r, ang.cos() * r], [0.0, ang.sin(), ang.cos()], [u, v])
        });
        self.push_subdiv_box([1.95, 6.62, 0.0], [0.50, 0.22, 0.38], 7, 4, [0.5, 0.5]);
    }
}

pub fn build_city_meshlet_library() -> MeshletLibrary {
    let mut b = MeshletBuilder::new();
    let mut ranges = [(0u32, 0u32); 5];

    let builders: [fn(&mut MeshletBuilder); 5] = [
        MeshletBuilder::build_skyscraper,
        MeshletBuilder::build_hotel,
        MeshletBuilder::build_supercar,
        MeshletBuilder::build_palm,
        MeshletBuilder::build_lamp,
    ];
    for (idx, func) in builders.iter().enumerate() {
        let start = b.meshlets.len() as u32;
        func(&mut b);
        let count = b.meshlets.len() as u32 - start;
        ranges[idx] = (start, count);
    }

    MeshletLibrary {
        vertices: b.vertices,
        prim_indices: b.prim_indices,
        meshlets: b.meshlets,
        proto_ranges: ranges,
    }
}

#[inline]
fn wrap_z(base_z: f32, hero_z: f32, span: f32) -> f32 {
    let rel = (base_z - hero_z + span * 0.25).rem_euclid(span) - span * 0.25;
    hero_z + rel
}

/// Generates a dense Vice-City coastal metropolis (~4.5M triangles across 48K+ meshlets).
/// Returns `(instances, work_item_protos_or_empty)`: note work items only depend on the static instance slot list!
pub fn build_highpoly_city_instances(
    lib: &MeshletLibrary,
    t_sec: f32,
    hero_z: f32,
) -> (Vec<MeshletInstance>, Vec<MeshletWorkItem>, usize) {
    let mut instances = Vec::with_capacity(820);
    let mut work_items = Vec::with_capacity(50000);
    let span = 320.0f32;

    let mut push_inst = |proto: usize, inst: MeshletInstance| {
        let inst_idx = instances.len() as u32;
        instances.push(inst);
        let (m_start, m_cnt) = lib.proto_ranges[proto];
        for m in m_start..(m_start + m_cnt) {
            work_items.push(MeshletWorkItem {
                inst_idx,
                meshlet_idx: m,
            });
        }
    };

    // 1. Multi-avenue Art-Deco Skyline on left side (6 avenues deep x 42 slots = 252 skyscrapers + 126 hotels)
    for av in 0..6 {
        let ax = -19.5 - av as f32 * 18.5;
        for s in 0..42 {
            let bz = s as f32 * 7.6 + (av as f32) * 3.1;
            let z = wrap_z(bz, hero_z, span);
            let h = 22.0 + ((s * 7 + av * 11) % 30) as f32;
            let w = 9.5 + ((s + av) % 3) as f32 * 1.3;
            let d = 10.5 + ((s * 3 + av) % 3) as f32 * 1.1;
            let tex = if (s + av) % 3 == 0 { 2u32 } else { 1u32 };
            let em = if tex == 1 { 0.36 } else { 0.24 };
            push_inst(
                PROTO_SKYSCRAPER,
                MeshletInstance {
                    pos: [ax, h * 0.5, z],
                    rot_y: ((av + s) & 1) as f32 * std::f32::consts::FRAC_PI_2,
                    scale: [w, h, d],
                    tex_index: tex,
                    emissive: em,
                    uv_scale: 1.5,
                    spec_power: 32.0,
                    pad: 0.0,
                },
            );
            if (s + av) % 2 == 0 {
                push_inst(
                    PROTO_HOTEL,
                    MeshletInstance {
                        pos: [ax + 8.5, 5.5, z + 3.8],
                        rot_y: 0.0,
                        scale: [6.5, 11.0, 6.8],
                        tex_index: 10,
                        emissive: 0.28,
                        uv_scale: 1.2,
                        spec_power: 24.0,
                        pad: 0.0,
                    },
                );
            }
        }
    }

    // 2. Oceanfront Boutique Hotels on right boardwalk (80 hotels with generous gaps to see ocean)
    for k in 0..8 {
        let z = wrap_z(k as f32 * 40.0 + 22.0, hero_z, span);
        let h = 11.0 + (k % 3) as f32 * 2.5;
        let tex = if k % 2 == 0 { 2u32 } else { 1u32 };
        push_inst(
            PROTO_HOTEL,
            MeshletInstance {
                pos: [15.5, h * 0.5, z],
                rot_y: 0.0,
                scale: [5.6, h, 10.0],
                tex_index: tex,
                emissive: 0.30,
                uv_scale: 1.4,
                spec_power: 28.0,
                pad: 0.0,
            },
        );
    }

    // 3. Dense Tropical Palm Groves (128 high-poly curved palms)
    for k in 0..64 {
        let bz = k as f32 * 5.0 + 2.0;
        let z = wrap_z(bz, hero_z, span);
        let sway = (t_sec * 1.6 + k as f32 * 0.7).sin() * 0.06;
        push_inst(
            PROTO_PALM,
            MeshletInstance {
                pos: [-8.3, 0.0, z],
                rot_y: 0.14 + sway,
                scale: [1.05, 1.05, 1.05],
                tex_index: 5,
                emissive: 0.08,
                uv_scale: 1.0,
                spec_power: 18.0,
                pad: 0.0,
            },
        );
        push_inst(
            PROTO_PALM,
            MeshletInstance {
                pos: [8.5, 0.0, z + 2.5],
                rot_y: std::f32::consts::PI - 0.12 - sway,
                scale: [1.08, 1.08, 1.08],
                tex_index: 5,
                emissive: 0.08,
                uv_scale: 1.0,
                spec_power: 18.0,
                pad: 0.0,
            },
        );
    }

    // 4. Boulevard Streetlamps (32 pairs = 64 lamps)
    for k in 0..32 {
        let z = wrap_z(k as f32 * 10.0 + 5.0, hero_z, span);
        push_inst(
            PROTO_LAMP,
            MeshletInstance {
                pos: [-7.4, 0.0, z],
                rot_y: 0.0,
                scale: [1.0, 1.0, 1.0],
                tex_index: 11,
                emissive: 0.32,
                uv_scale: 1.0,
                spec_power: 32.0,
                pad: 0.0,
            },
        );
        push_inst(
            PROTO_LAMP,
            MeshletInstance {
                pos: [7.4, 0.0, z],
                rot_y: std::f32::consts::PI,
                scale: [1.0, 1.0, 1.0],
                tex_index: 11,
                emissive: 0.32,
                uv_scale: 1.0,
                spec_power: 32.0,
                pad: 0.0,
            },
        );
    }

    // 5. Hero Sports Car + 100 Animated High-Poly Traffic Cars across 4 lanes
    let hero_x = 2.4 + (t_sec * 0.7).sin() * 0.45;
    let hero_yaw = (t_sec * 0.7).cos() * 0.04;
    push_inst(
        PROTO_CAR,
        MeshletInstance {
            pos: [hero_x, 0.0, hero_z],
            rot_y: hero_yaw,
            scale: [1.05, 1.05, 1.05],
            tex_index: 8,
            emissive: 0.14,
            uv_scale: 1.0,
            spec_power: 96.0,
            pad: 0.0,
        },
    );

    let lanes = [2.4f32, 5.2, -2.4, -5.2];
    for c in 0..100usize {
        let lane = c % 4;
        let lx = lanes[lane];
        let dir = if lane < 2 { 1.0f32 } else { -1.0f32 };
        let spd = (12.0 + (c % 5) as f32 * 1.1) * dir;
        let yaw = if dir > 0.0 { 0.0 } else { std::f32::consts::PI };
        let z0 = ((c / 4) as f32) * 12.8 + (lane as f32) * 3.1;
        let tz = wrap_z(z0 + t_sec * spd, hero_z, span);
        let tex = if (c % 3) == 0 { 8u32 } else { 9u32 };
        push_inst(
            PROTO_CAR,
            MeshletInstance {
                pos: [lx, 0.0, tz],
                rot_y: yaw,
                scale: [1.0, 1.0, 1.0],
                tex_index: tex,
                emissive: 0.12,
                uv_scale: 1.0,
                spec_power: 88.0,
                pad: 0.0,
            },
        );
    }

    let total_tris: usize = work_items
        .iter()
        .map(|w| lib.meshlets[w.meshlet_idx as usize].prim_count as usize)
        .sum();
    (instances, work_items, total_tris)
}

pub fn supports_mesh_shaders(device: &ID3D12Device) -> bool {
    unsafe {
        let mut opts7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        if device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS7,
                &mut opts7 as *mut _ as *mut c_void,
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>() as u32,
            )
            .is_err()
        {
            return false;
        }
        opts7.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0
    }
}

/// Extracts 6 normalized inward-facing frustum planes from column-major `ViewProj`.
pub fn extract_frustum_planes(vp: &[f32; 16]) -> [[f32; 4]; 6] {
    let r0 = [vp[0], vp[4], vp[8], vp[12]];
    let r1 = [vp[1], vp[5], vp[9], vp[13]];
    let r2 = [vp[2], vp[6], vp[10], vp[14]];
    let r3 = [vp[3], vp[7], vp[11], vp[15]];

    let raw = [
        [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // Left
        [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // Right
        [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // Bottom
        [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // Top
        [r2[0], r2[1], r2[2], r2[3]],                                 // Near (z >= 0)
        [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]], // Far  (z <= w)
    ];

    let mut out = [[0.0f32; 4]; 6];
    for i in 0..6 {
        let len = (raw[i][0] * raw[i][0] + raw[i][1] * raw[i][1] + raw[i][2] * raw[i][2])
            .sqrt()
            .max(1e-6);
        out[i] = [raw[i][0] / len, raw[i][1] / len, raw[i][2] / len, raw[i][3] / len];
    }
    out
}

pub struct MeshletPipeline {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    _vb_buf: ID3D12Resource,
    _pb_buf: ID3D12Resource,
    _mb_buf: ID3D12Resource,
    _wb_buf: ID3D12Resource,
    inst_upload: [ID3D12Resource; 2],
    cb_upload: [ID3D12Resource; 2],
    cull_stats_uav: [ID3D12Resource; 2],
    cull_stats_rb: [ID3D12Resource; 2],
    cull_stats_zero: ID3D12Resource,
    pub total_work_items: u32,
    pub total_tris: usize,
    pub instance_count: usize,
}

impl MeshletPipeline {
    pub fn new(
        ctx: &mut GpuContext,
        lib: &MeshletLibrary,
        init_instances: &[MeshletInstance],
        work_items: &[MeshletWorkItem],
        total_tris: usize,
    ) -> Option<Self> {
        if !supports_mesh_shaders(&ctx.device) {
            return None;
        }
        unsafe {
            let dev2: ID3D12Device2 = ctx.device.cast().ok()?;

            let range = D3D12_DESCRIPTOR_RANGE1 {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 12,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
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
                // 0: b0 MeshFrameCB
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
                // 1: t0..t11 texture descriptor table
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
                // 2: t12 TLAS
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
                // 3: t13 g_InstColors
                srv_static(13, 0, D3D12_SHADER_VISIBILITY_PIXEL),
                // 4: t0, space1 g_Vertices
                srv_static(0, 1, D3D12_SHADER_VISIBILITY_ALL),
                // 5: t1, space1 g_PrimIndices
                srv_static(1, 1, D3D12_SHADER_VISIBILITY_ALL),
                // 6: t2, space1 g_Meshlets
                srv_static(2, 1, D3D12_SHADER_VISIBILITY_ALL),
                // 7: t3, space1 g_Instances
                srv_static(3, 1, D3D12_SHADER_VISIBILITY_ALL),
                // 8: t4, space1 g_WorkItems
                srv_static(4, 1, D3D12_SHADER_VISIBILITY_ALL),
                // 9: u0, space1 g_CullStats
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

            let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                    Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                        NumParameters: params.len() as u32,
                        pParameters: params.as_ptr(),
                        NumStaticSamplers: 1,
                        pStaticSamplers: &static_sampler,
                        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                    },
                },
            };
            let mut rs_blob = None;
            D3D12SerializeVersionedRootSignature(&rs_desc, &mut rs_blob, None).ok()?;
            let rs_blob = rs_blob?;
            let root_sig: ID3D12RootSignature = ctx
                .device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(rs_blob.GetBufferPointer() as *const u8, rs_blob.GetBufferSize()),
                )
                .ok()?;

            let mut rt_formats = D3D12_RT_FORMAT_ARRAY::default();
            rt_formats.NumRenderTargets = 1;
            rt_formats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

            let blend = D3D12_BLEND_DESC {
                RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC {
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                }; 8],
                ..Default::default()
            };
            let rasterizer = D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                FrontCounterClockwise: false.into(),
                DepthClipEnable: true.into(),
                ..Default::default()
            };
            let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                ..Default::default()
            };

            let mut stream = MeshShaderPsoStream {
                root_sig: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
                    inner: ManuallyDrop::new(Some(root_sig.clone())),
                },
                as_shader: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
                    inner: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: MESHLET_AS_DXIL.as_ptr() as *const c_void,
                        BytecodeLength: MESHLET_AS_DXIL.len(),
                    },
                },
                ms_shader: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
                    inner: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: MESHLET_MS_DXIL.as_ptr() as *const c_void,
                        BytecodeLength: MESHLET_MS_DXIL.len(),
                    },
                },
                ps_shader: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
                    inner: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: MESHLET_PS_DXIL.as_ptr() as *const c_void,
                        BytecodeLength: MESHLET_PS_DXIL.len(),
                    },
                },
                blend: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,
                    inner: blend,
                },
                sample_mask: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
                    inner: u32::MAX,
                },
                rasterizer: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
                    inner: rasterizer,
                },
                depth_stencil: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
                    inner: depth_stencil,
                },
                rtv_formats: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
                    inner: rt_formats,
                },
                dsv_format: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
                    inner: DXGI_FORMAT_D32_FLOAT,
                },
                sample_desc: PsoSubobject {
                    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
                    inner: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                },
            };

            let pso_stream_desc = D3D12_PIPELINE_STATE_STREAM_DESC {
                SizeInBytes: size_of::<MeshShaderPsoStream>(),
                pPipelineStateSubobjectStream: &mut stream as *mut _ as *mut c_void,
            };
            let pso: ID3D12PipelineState = dev2.CreatePipelineState(&pso_stream_desc).ok()?;
            let _ = ManuallyDrop::take(&mut stream.root_sig.inner);

            // Static geometry buffers (DEFAULT heap uploaded via staging once at startup)
            let vb_bytes = (lib.vertices.len() * size_of::<MeshletVertex>()).max(64) as u64;
            let pb_bytes = (lib.prim_indices.len() * size_of::<u32>()).max(64) as u64;
            let mb_bytes = (lib.meshlets.len() * size_of::<MeshletDesc>()).max(64) as u64;
            let wb_bytes = (work_items.len() * size_of::<MeshletWorkItem>()).max(64) as u64;

            let vb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, vb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let pb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, pb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let mb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, mb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let wb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, wb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            upload_slice(&vb_up, &lib.vertices);
            upload_slice(&pb_up, &lib.prim_indices);
            upload_slice(&mb_up, &lib.meshlets);
            upload_slice(&wb_up, work_items);

            let vb_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, vb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
            let pb_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, pb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
            let mb_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, mb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
            let wb_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, wb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);

            let up_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
            let up_cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &up_alloc, None).ok()?;
            up_cmd.CopyBufferRegion(&vb_buf, 0, &vb_up, 0, vb_bytes);
            up_cmd.CopyBufferRegion(&pb_buf, 0, &pb_up, 0, pb_bytes);
            up_cmd.CopyBufferRegion(&mb_buf, 0, &mb_up, 0, mb_bytes);
            up_cmd.CopyBufferRegion(&wb_buf, 0, &wb_up, 0, wb_bytes);
            cmd_barriers(
                &up_cmd,
                vec![
                    transition_barrier(&vb_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                    transition_barrier(&pb_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                    transition_barrier(&mb_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                    transition_barrier(&wb_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                ],
            );
            up_cmd.Close().ok()?;
            let lists = [Some(up_cmd.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait();

            let inst_cap = init_instances.len().max(MAX_MESHLET_INSTANCES);
            let inst_bytes = (inst_cap * size_of::<MeshletInstance>()) as u64;
            let inst_upload = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, inst_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, inst_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            ];
            let cb_upload = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            ];
            let cull_stats_uav = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
            ];
            let cull_stats_rb = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST),
            ];
            let cull_stats_zero = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            upload_slice(&cull_stats_zero, &[0u32; 4]);

            Some(Self {
                root_sig,
                pso,
                _vb_buf: vb_buf,
                _pb_buf: pb_buf,
                _mb_buf: mb_buf,
                _wb_buf: wb_buf,
                inst_upload,
                cb_upload,
                cull_stats_uav,
                cull_stats_rb,
                cull_stats_zero,
                total_work_items: work_items.len() as u32,
                total_tris,
                instance_count: init_instances.len(),
            })
        }
    }

    pub unsafe fn record_dispatch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        slot: usize,
        instances: &[MeshletInstance],
        cb: &MeshFrameCB,
        srv_gpu_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
        tlas_va: u64,
        inst_colors_va: u64,
    ) {
        upload_slice(&self.inst_upload[slot], instances);
        upload_slice(&self.cb_upload[slot], std::slice::from_ref(cb));

        let mut b0 = [transition_barrier(
            &self.cull_stats_uav[slot],
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )];
        cmd.ResourceBarrier(&b0);
        drop_barriers(&mut b0);
        cmd.CopyBufferRegion(&self.cull_stats_uav[slot], 0, &self.cull_stats_zero, 0, 16);

        let mut b1 = [transition_barrier(
            &self.cull_stats_uav[slot],
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )];
        cmd.ResourceBarrier(&b1);
        drop_barriers(&mut b1);

        cmd.SetGraphicsRootSignature(&self.root_sig);
        cmd.SetPipelineState(&self.pso);
        cmd.SetGraphicsRootConstantBufferView(0, self.cb_upload[slot].GetGPUVirtualAddress());
        cmd.SetGraphicsRootDescriptorTable(1, srv_gpu_handle);
        cmd.SetGraphicsRootShaderResourceView(2, tlas_va);
        cmd.SetGraphicsRootShaderResourceView(3, inst_colors_va);
        cmd.SetGraphicsRootShaderResourceView(4, self._vb_buf.GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(5, self._pb_buf.GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(6, self._mb_buf.GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(7, self.inst_upload[slot].GetGPUVirtualAddress());
        cmd.SetGraphicsRootShaderResourceView(8, self._wb_buf.GetGPUVirtualAddress());
        cmd.SetGraphicsRootUnorderedAccessView(9, self.cull_stats_uav[slot].GetGPUVirtualAddress());

        let cmd6: ID3D12GraphicsCommandList6 = cmd.cast().expect("ID3D12GraphicsCommandList6");
        let groups_x = self.total_work_items.div_ceil(32);
        cmd6.DispatchMesh(groups_x, 1, 1);

        let mut uav_b = [uav_barrier(&self.cull_stats_uav[slot])];
        cmd.ResourceBarrier(&uav_b);
        drop_barriers(&mut uav_b);

        let mut b2 = [transition_barrier(
            &self.cull_stats_uav[slot],
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )];
        cmd.ResourceBarrier(&b2);
        drop_barriers(&mut b2);

        cmd.CopyBufferRegion(&self.cull_stats_rb[slot], 0, &self.cull_stats_uav[slot], 0, 16);

        let mut b3 = [transition_barrier(
            &self.cull_stats_uav[slot],
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        )];
        cmd.ResourceBarrier(&b3);
        drop_barriers(&mut b3);
    }

    pub unsafe fn read_cull_stats(&self, slot: usize) -> (u32, u32, u32) {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rng = D3D12_RANGE { Begin: 0, End: 16 };
        self.cull_stats_rb[slot].Map(0, Some(&rng), Some(&mut ptr)).unwrap();
        let vals = std::slice::from_raw_parts(ptr as *const u32, 4);
        let res = (vals[0], vals[1], vals[2]);
        self.cull_stats_rb[slot].Unmap(0, None);
        res
    }
}

pub struct MeshShaderVerifyStats {
    pub total_tris: usize,
    pub total_work_items: u32,
    pub visible_meshlets: u32,
    pub visible_tris: u32,
    pub culled_meshlets: u32,
    pub lit_pixels: usize,
}

pub fn run_mesh_shader_verify(ctx: &mut GpuContext) -> MeshShaderVerifyStats {
    assert!(supports_mesh_shaders(&ctx.device), "GPU does not support Mesh Shaders SM 6.5");

    let lib = build_city_meshlet_library();
    let (instances, work_items, total_tris) = build_highpoly_city_instances(&lib, 1.5, 27.0);
    assert!(
        total_tris >= 3_500_000,
        "Expected >= 3.5M city triangles, got {total_tris}"
    );

    let pipe = MeshletPipeline::new(ctx, &lib, &instances, &work_items, total_tris)
        .expect("Failed to create MeshletPipeline");

    unsafe {
        let rt_w = 256u32;
        let rt_h = 256u32;
        let rtv_heap: ID3D12DescriptorHeap = ctx
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
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
                NumDescriptors: 12,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap();

        let rt_tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: rt_w as u64,
            Height: rt_h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut rt_res: Option<ID3D12Resource> = None;
        ctx.device
            .CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &rt_tex_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut rt_res,
            )
            .unwrap();
        let rt_res = rt_res.unwrap();
        ctx.device.CreateRenderTargetView(&rt_res, None, rtv_heap.GetCPUDescriptorHandleForHeapStart());

        let ds_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: rt_w as u64,
            Height: rt_h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let d_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 },
            },
        };
        let mut ds_res: Option<ID3D12Resource> = None;
        ctx.device
            .CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &ds_desc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                Some(&d_clear),
                &mut ds_res,
            )
            .unwrap();
        let ds_res = ds_res.unwrap();
        ctx.device.CreateDepthStencilView(&ds_res, None, dsv_heap.GetCPUDescriptorHandleForHeapStart());

        // Dummy 4x4 RGBA8 texture bound across all 12 SRV slots + dummy TLAS/colors VA
        let dummy_tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        let mut dummy_tex: Option<ID3D12Resource> = None;
        ctx.device
            .CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &dummy_tex_desc,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                None,
                &mut dummy_tex,
            )
            .unwrap();
        let dummy_tex = dummy_tex.unwrap();
        let srv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let srv_base = srv_heap.GetCPUDescriptorHandleForHeapStart();
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        for i in 0..12u32 {
            ctx.device.CreateShaderResourceView(
                &dummy_tex,
                Some(&srv_desc),
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: srv_base.ptr + (i * srv_inc) as usize },
            );
        }

        // Valid tiny empty TLAS so t12 binding is 100% Debug-Layer valid even with rt_enabled = 0.0
        let dev5: ID3D12Device5 = ctx.device.cast().unwrap();
        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
            NumDescs: 0,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { InstanceDescs: 0 },
        };
        let mut prebuild = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut prebuild);
        let tlas_res = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            prebuild.ResultDataMaxSizeInBytes.max(256),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        );
        let tlas_scratch = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            prebuild.ScratchDataSizeInBytes.max(256),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let dummy_colors = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        upload_slice(&dummy_colors, &[[1.0f32, 0.5, 0.8, 0.2]]);

        let rb_pitch = rt_w * 4;
        let rt_rb = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_READBACK,
            (rb_pitch * rt_h) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();

        let cmd4: ID3D12GraphicsCommandList4 = cmd.cast().unwrap();
        let build_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: tlas_res.GetGPUVirtualAddress(),
            Inputs: tlas_inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: tlas_scratch.GetGPUVirtualAddress(),
        };
        cmd4.BuildRaytracingAccelerationStructure(&build_desc, None);
        let mut b_tlas = [uav_barrier(&tlas_res)];
        cmd.ResourceBarrier(&b_tlas);
        drop_barriers(&mut b_tlas);

        let rtv_h = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        let dsv_h = dsv_heap.GetCPUDescriptorHandleForHeapStart();
        cmd.ClearRenderTargetView(rtv_h, &[0.0, 0.0, 0.0, 0.0], None);
        cmd.ClearDepthStencilView(dsv_h, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, &[]);
        cmd.OMSetRenderTargets(1, Some(&rtv_h), false, Some(&dsv_h));

        let vp_rect = D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: rt_w as f32, Height: rt_h as f32, MinDepth: 0.0, MaxDepth: 1.0 };
        let sc_rect = RECT { left: 0, top: 0, right: rt_w as i32, bottom: rt_h as i32 };
        cmd.RSSetViewports(&[vp_rect]);
        cmd.RSSetScissorRects(&[sc_rect]);
        cmd.SetDescriptorHeaps(&[Some(srv_heap.clone())]);

        // Look down highway from [0, 8, 12] toward [0, 3, 50]
        let eye = [0.0f32, 8.0, 12.0];
        let tgt = [0.0f32, 3.0, 50.0];
        let vp_mat = perspective_lh(62.0f32.to_radians(), 1.0, 0.2, 280.0).mul(&look_at_lh(eye, tgt, [0.0, 1.0, 0.0]));
        let cb = MeshFrameCB {
            view_proj: vp_mat,
            frustum_planes: extract_frustum_planes(&vp_mat),
            camera_pos: [eye[0], eye[1], eye[2], 0.0],
            point_lights: [[0.0, 6.0, 30.0, 5.0]; 4],
            light_colors: [[1.0, 0.7, 0.9, 32.0]; 4],
            total_work_items: pipe.total_work_items,
            time: 1.5,
            max_cull_dist: 240.0,
            pad: 0,
        };

        pipe.record_dispatch(
            &cmd,
            0,
            &instances,
            &cb,
            srv_heap.GetGPUDescriptorHandleForHeapStart(),
            tlas_res.GetGPUVirtualAddress(),
            dummy_colors.GetGPUVirtualAddress(),
        );

        let mut b_cp = [transition_barrier(&rt_res, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)];
        cmd.ResourceBarrier(&b_cp);
        drop_barriers(&mut b_cp);
        let mut dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(rt_rb.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: rt_w,
                        Height: rt_h,
                        Depth: 1,
                        RowPitch: rb_pitch,
                    },
                },
            },
        };
        let mut src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(rt_res.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
        let _ = ManuallyDrop::take(&mut dst_loc.pResource);
        let _ = ManuallyDrop::take(&mut src_loc.pResource);

        cmd.Close().unwrap();
        let lists = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&lists);
        ctx.signal_and_wait();

        let (visible_meshlets, visible_tris, culled_meshlets) = pipe.read_cull_stats(0);
        assert_eq!(
            visible_meshlets + culled_meshlets,
            pipe.total_work_items,
            "Wave32 cull accounting mismatch: {visible_meshlets} + {culled_meshlets} != {}",
            pipe.total_work_items
        );
        assert!(visible_meshlets > 0, "Expected visible meshlets > 0");
        assert!(culled_meshlets > 0, "Expected GPU Frustum Culling to cull out-of-frustum meshlets");
        assert!(visible_tris > 0, "Expected visible_tris > 0");

        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rng = D3D12_RANGE { Begin: 0, End: (rb_pitch * rt_h) as usize };
        rt_rb.Map(0, Some(&rng), Some(&mut ptr)).unwrap();
        let pixels = std::slice::from_raw_parts(ptr as *const u8, (rb_pitch * rt_h) as usize);
        let lit_pixels = pixels.chunks_exact(4).filter(|px| px[3] > 0 && (px[0] > 0 || px[1] > 0 || px[2] > 0)).count();
        rt_rb.Unmap(0, None);
        assert!(lit_pixels > 500, "Expected > 500 rendered city pixels, got {lit_pixels}");

        MeshShaderVerifyStats {
            total_tris,
            total_work_items: pipe.total_work_items,
            visible_meshlets,
            visible_tris,
            culled_meshlets,
            lit_pixels,
        }
    }
}

struct SmallMat([f32; 16]);
impl SmallMat {
    fn mul(&self, rhs: &SmallMat) -> [f32; 16] {
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
        out
    }
}

fn perspective_lh(fov_y_rad: f32, aspect: f32, zn: f32, zf: f32) -> SmallMat {
    let ys = 1.0 / (fov_y_rad * 0.5).tan();
    let xs = ys / aspect;
    let q = zf / (zf - zn);
    SmallMat([
        xs, 0.0, 0.0, 0.0,
        0.0, ys, 0.0, 0.0,
        0.0, 0.0, q, 1.0,
        0.0, 0.0, -zn * q, 0.0,
    ])
}

fn look_at_lh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> SmallMat {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let norm = |v: [f32; 3]| {
        let l = dot(v, v).sqrt().max(1e-6);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let f = norm(sub(target, eye));
    let r = norm(cross(up, f));
    let u = cross(f, r);
    SmallMat([
        r[0], u[0], f[0], 0.0,
        r[1], u[1], f[1], 0.0,
        r[2], u[2], f[2], 0.0,
        -dot(r, eye), -dot(u, eye), -dot(f, eye), 1.0,
    ])
}
