use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, heap_props, transition_barrier, uav_barrier,
    upload_slice, GpuContext,
};
use crate::meshlets::{
    build_city_meshlet_library, build_highpoly_city_instances, extract_frustum_planes,
    MeshFrameCB, MeshletDesc, MeshletInstance, MeshletPipeline, MeshletWorkItem, PROTO_SKYSCRAPER,
};
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop};
use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub const HIZ_BUILD_CS_DXIL: &[u8] = include_bytes!("../shaders/hiz_build.dxil");
pub const HIZ_CULL_CS_DXIL: &[u8] = include_bytes!("../shaders/hiz_cull.dxil");

#[repr(C)]
#[derive(Clone, Copy)]
struct HiZCullCB {
    view_proj: [f32; 16],
    frustum_planes: [[f32; 4]; 6],
    camera_pos: [f32; 4],
    total_work_items: u32,
    max_cull_dist: f32,
    hiz_width: u32,
    hiz_height: u32,
}

#[derive(Clone, Copy)]
struct Mat4([f32; 16]);

impl Mat4 {
    fn mul(&self, rhs: &Self) -> Self {
        let mut r = [0.0f32; 16];
        for c in 0..4 {
            for row in 0..4 {
                r[c * 4 + row] = (0..4).map(|k| self.0[k * 4 + row] * rhs.0[c * 4 + k]).sum();
            }
        }
        Self(r)
    }

    fn look_at_lh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Self {
        let norm3 = |a: [f32; 3]| -> [f32; 3] {
            let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-6);
            [a[0] / l, a[1] / l, a[2] / l]
        };
        let cross3 = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        let dot3 = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

        let f = norm3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
        let r = norm3(cross3(up, f));
        let u = cross3(f, r);
        Self([
            r[0], u[0], f[0], 0.0,
            r[1], u[1], f[1], 0.0,
            r[2], u[2], f[2], 0.0,
            -dot3(r, eye), -dot3(u, eye), -dot3(f, eye), 1.0,
        ])
    }

    fn perspective_lh(fov_y_rad: f32, aspect: f32, zn: f32, zf: f32) -> Self {
        let ys = 1.0 / (fov_y_rad * 0.5).tan();
        let xs = ys / aspect;
        let q = zf / (zf - zn);
        Self([
            xs, 0.0, 0.0, 0.0,
            0.0, ys, 0.0, 0.0,
            0.0, 0.0, q, 1.0,
            0.0, 0.0, -zn * q, 0.0,
        ])
    }
}

pub struct HiZVerifyStats {
    pub total_work_items: u32,
    pub total_tris: usize,
    pub visible_meshlets: u32,
    pub visible_tris: u32,
    pub frustum_culled: u32,
    pub hiz_occluded: u32,
    pub hiz_occluded_tris: u32,
    pub hiz_build_ms: f64,
    pub hiz_cull_ms: f64,
}

unsafe fn create_compute_rs_and_pso(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER],
    dxil: &[u8],
) -> (ID3D12RootSignature, ID3D12PipelineState) {
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
    let rs_bytes = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
    let root_sig: ID3D12RootSignature = device.CreateRootSignature(0, rs_bytes).unwrap();
    let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
        NodeMask: 0,
        CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
        Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
    };
    let pso: ID3D12PipelineState = device.CreateComputePipelineState(&pso_desc).unwrap();
    let _ = ManuallyDrop::take(&mut pso_desc.pRootSignature);
    (root_sig, pso)
}

pub fn run_hiz_12m_verify(ctx: &mut GpuContext) -> HiZVerifyStats {
    let lib = build_city_meshlet_library();
    let (base_insts, base_work, _) = build_highpoly_city_instances(&lib, 1.5, 27.0);

    // Construct a 12M+ triangle city (3 z-bands) plus 3 front-row occluder skyscrapers at z=24
    let sky_range = lib.proto_ranges[PROTO_SKYSCRAPER];
    let mut occ_insts: Vec<MeshletInstance> = Vec::new();
    let mut occ_work: Vec<MeshletWorkItem> = Vec::new();

    for ox in [-14.0f32, 0.0, 14.0] {
        let inst_idx = occ_insts.len() as u32;
        occ_insts.push(MeshletInstance {
            pos: [ox, 18.0, 24.0],
            rot_y: 0.0,
            scale: [16.0, 48.0, 8.0],
            tex_index: 1,
            emissive: 0.0,
            uv_scale: 1.0,
            spec_power: 32.0,
            pad: 0.0,
        });
        for mi in sky_range.0..sky_range.1 {
            occ_work.push(MeshletWorkItem { inst_idx, meshlet_idx: mi as u32 });
        }
    }
    let occ_tris: usize = occ_work.iter().map(|w| lib.meshlets[w.meshlet_idx as usize].prim_count as usize).sum();

    let mut all_insts: Vec<MeshletInstance> = Vec::with_capacity(base_insts.len() * 3 + 3);
    let mut all_work: Vec<MeshletWorkItem> = Vec::with_capacity(base_work.len() * 3 + occ_work.len());
    // Include front occluders in total work set as well
    for inst in &occ_insts {
        all_insts.push(*inst);
    }
    for w in &occ_work {
        all_work.push(*w);
    }
    for band in 0..3 {
        let base_idx = all_insts.len() as u32;
        let z_off = (band as f32) * 110.0;
        for inst in &base_insts {
            let mut moved = *inst;
            moved.pos[2] += z_off;
            all_insts.push(moved);
        }
        for w in &base_work {
            all_work.push(MeshletWorkItem {
                inst_idx: base_idx + w.inst_idx,
                meshlet_idx: w.meshlet_idx,
            });
        }
    }
    let total_tris: usize = all_work.iter().map(|w| lib.meshlets[w.meshlet_idx as usize].prim_count as usize).sum();
    assert!(total_tris >= 12_000_000, "Expected >= 12M triangles for Hi-Z stress, got {total_tris}");

    // Pass 1: Hardware Mesh Shader Depth Pre-Pass on the front occluder towers
    let pass1_pipe = MeshletPipeline::new(ctx, &lib, &occ_insts, &occ_work, occ_tris)
        .expect("Failed to create Pass-1 MeshletPipeline for Hi-Z occluders");

    unsafe {
        let rt_w = 256u32;
        let rt_h = 256u32;

        // Upload 12M+ city descriptors/instances/work_items
        let mb_bytes = (lib.meshlets.len() * size_of::<MeshletDesc>()) as u64;
        let ib_bytes = (all_insts.len() * size_of::<MeshletInstance>()) as u64;
        let wb_bytes = (all_work.len() * size_of::<MeshletWorkItem>()) as u64;

        let mb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, mb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let ib_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, ib_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let wb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, wb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        upload_slice(&mb_up, &lib.meshlets);
        upload_slice(&ib_up, &all_insts);
        upload_slice(&wb_up, &all_work);

        let mb_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, mb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        let ib_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, ib_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        let wb_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, wb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let rtv_heap: ID3D12DescriptorHeap = ctx.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 3,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        }).unwrap();
        let dsv_heap: ID3D12DescriptorHeap = ctx.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        }).unwrap();
        // Heap for Pass 1 MeshShader SRVs (0..11)
        let ms_srv_heap: ID3D12DescriptorHeap = ctx.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 12,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        }).unwrap();
        // Heap for Hi-Z build & cull:
        // 0 = depth SRV, 1..5 = hiz[0..4] SRV for build, 6..11 = hiz[0..5] UAV, 12..17 = contiguous hiz[0..5] SRVs for cull
        let hiz_heap: ID3D12DescriptorHeap = ctx.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 20,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        }).unwrap();

        let rtv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let rtv_base = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        let rt_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: rt_w as u64,
            Height: rt_h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut rts: Vec<ID3D12Resource> = Vec::with_capacity(3);
        for i in 0..3usize {
            let mut r = None;
            ctx.device.CreateCommittedResource(&heap_props(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut r).unwrap();
            let r = r.unwrap();
            ctx.device.CreateRenderTargetView(&r, None, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv_base.ptr + i * rtv_inc as usize });
            rts.push(r);
        }

        // Typeless Depth Buffer (DSV D32_FLOAT + SRV R32_FLOAT)
        let depth_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: rt_w as u64,
            Height: rt_h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let d_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 } },
        };
        let mut depth_res: Option<ID3D12Resource> = None;
        ctx.device.CreateCommittedResource(&heap_props(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &depth_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, Some(&d_clear), &mut depth_res).unwrap();
        let depth_res = depth_res.unwrap();
        let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            Flags: D3D12_DSV_FLAG_NONE,
            ..Default::default()
        };
        ctx.device.CreateDepthStencilView(&depth_res, Some(&dsv_desc), dsv_heap.GetCPUDescriptorHandleForHeapStart());

        // Dummy texture & dummy TLAS for Pass 1 MeshShader PSO
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
        ctx.device.CreateCommittedResource(&heap_props(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &dummy_tex_desc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, None, &mut dummy_tex).unwrap();
        let dummy_tex = dummy_tex.unwrap();
        let ms_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let ms_base = ms_srv_heap.GetCPUDescriptorHandleForHeapStart();
        for i in 0..12usize {
            let mut sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            sd.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() };
            ctx.device.CreateShaderResourceView(&dummy_tex, Some(&sd), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: ms_base.ptr + i * ms_inc as usize });
        }

        let dev5: ID3D12Device5 = ctx.device.cast().unwrap();
        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE,
            NumDescs: 0,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            ..Default::default()
        };
        let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut pre);
        let tlas_res = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, pre.ResultDataMaxSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
        let tlas_scratch = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, pre.ScratchDataSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        let dummy_colors = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        upload_slice(&dummy_colors, &[[1.0f32, 0.5, 0.8, 0.2]]);

        // 6 Hi-Z mip textures (256, 128, 64, 32, 16, 8)
        let mip_dims = [256u32, 128, 64, 32, 16, 8];
        let mut hiz_mips: Vec<ID3D12Resource> = Vec::with_capacity(6);
        for &d in &mip_dims {
            let hd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: d as u64,
                Height: d,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut res = None;
            ctx.device.CreateCommittedResource(&heap_props(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &hd, D3D12_RESOURCE_STATE_COMMON, None, &mut res).unwrap();
            hiz_mips.push(res.unwrap());
        }

        let hiz_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let hiz_cpu_base = hiz_heap.GetCPUDescriptorHandleForHeapStart();
        let hiz_gpu_base = hiz_heap.GetGPUDescriptorHandleForHeapStart();
        let cpu_at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: hiz_cpu_base.ptr + i * hiz_inc as usize };
        let gpu_at = |i: usize| D3D12_GPU_DESCRIPTOR_HANDLE { ptr: hiz_gpu_base.ptr + (i * hiz_inc as usize) as u64 };

        let mut f32_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        f32_srv.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() };
        let mut f32_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        f32_uav.Anonymous.Texture2D = D3D12_TEX2D_UAV::default();

        ctx.device.CreateShaderResourceView(&depth_res, Some(&f32_srv), cpu_at(0));
        for i in 0..5 {
            ctx.device.CreateShaderResourceView(&hiz_mips[i], Some(&f32_srv), cpu_at(1 + i));
        }
        for i in 0..6 {
            ctx.device.CreateUnorderedAccessView(&hiz_mips[i], None, Some(&f32_uav), cpu_at(6 + i));
            ctx.device.CreateShaderResourceView(&hiz_mips[i], Some(&f32_srv), cpu_at(12 + i));
        }

        // Build & Cull Root Signatures and PSOs
        let b_srv_rng = D3D12_DESCRIPTOR_RANGE { RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV, NumDescriptors: 1, BaseShaderRegister: 0, RegisterSpace: 0, OffsetInDescriptorsFromTableStart: 0 };
        let b_uav_rng = D3D12_DESCRIPTOR_RANGE { RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV, NumDescriptors: 1, BaseShaderRegister: 0, RegisterSpace: 0, OffsetInDescriptorsFromTableStart: 0 };
        let b_params = [
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &b_srv_rng } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &b_uav_rng } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
        ];
        let (build_rs, build_pso) = create_compute_rs_and_pso(&ctx.device, &b_params, HIZ_BUILD_CS_DXIL);

        let c_hiz_rng = D3D12_DESCRIPTOR_RANGE { RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV, NumDescriptors: 6, BaseShaderRegister: 3, RegisterSpace: 0, OffsetInDescriptorsFromTableStart: 0 };
        let c_params = [
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 1, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 2, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &c_hiz_rng } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
        ];
        let (cull_rs, cull_pso) = create_compute_rs_and_pso(&ctx.device, &c_params, HIZ_CULL_CS_DXIL);

        let cull_cb_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let stats_uav = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        let stats_zero = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        upload_slice(&stats_zero, &[0u32; 8]);
        let stats_rb = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let mut ts_heap: Option<ID3D12QueryHeap> = None;
        ctx.device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 4, NodeMask: 0 }, &mut ts_heap).unwrap();
        let ts_heap = ts_heap.unwrap();
        let ts_rb = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 32, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let eye = [0.0f32, 8.0, 12.0];
        let tgt = [0.0f32, 3.0, 50.0];
        let vp_mat = Mat4::perspective_lh(62.0f32.to_radians(), 1.0, 1.0, 420.0)
            .mul(&Mat4::look_at_lh(eye, tgt, [0.0, 1.0, 0.0]))
            .0;
        let frustum = extract_frustum_planes(&vp_mat);

        let pass1_cb = MeshFrameCB {
            view_proj: vp_mat,
            frustum_planes: frustum,
            camera_pos: [eye[0], eye[1], eye[2], 0.0],
            point_lights: [[0.0, 6.0, 30.0, 5.0]; 4],
            light_colors: [[1.0, 0.7, 0.9, 32.0]; 4],
            total_work_items: pass1_pipe.total_work_items,
            time: 1.5,
            max_cull_dist: 380.0,
            pad: 0,
        };
        let cull_cb = HiZCullCB {
            view_proj: vp_mat,
            frustum_planes: frustum,
            camera_pos: [eye[0], eye[1], eye[2], 0.0],
            total_work_items: all_work.len() as u32,
            max_cull_dist: 380.0,
            hiz_width: rt_w,
            hiz_height: rt_h,
        };
        upload_slice(&cull_cb_up, std::slice::from_ref(&cull_cb));

        let alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();

        cmd.CopyBufferRegion(&mb_def, 0, &mb_up, 0, mb_bytes);
        cmd.CopyBufferRegion(&ib_def, 0, &ib_up, 0, ib_bytes);
        cmd.CopyBufferRegion(&wb_def, 0, &wb_up, 0, wb_bytes);
        cmd.CopyBufferRegion(&stats_uav, 0, &stats_zero, 0, 32);
        cmd_barriers(&cmd, vec![
            transition_barrier(&mb_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition_barrier(&ib_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition_barrier(&wb_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition_barrier(&stats_uav, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);

        // Empty TLAS for Pass 1 PSO
        let cmd4: ID3D12GraphicsCommandList4 = cmd.cast().unwrap();
        cmd4.BuildRaytracingAccelerationStructure(&D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: tlas_res.GetGPUVirtualAddress(),
            Inputs: tlas_inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: tlas_scratch.GetGPUVirtualAddress(),
        }, None);
        cmd_barriers(&cmd, vec![uav_barrier(&tlas_res)]);

        // Pass 1: Render Front Occluder Skyscrapers into Depth Buffer
        let rtvs = [
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv_base.ptr },
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv_base.ptr + rtv_inc as usize },
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv_base.ptr + 2 * rtv_inc as usize },
        ];
        let dsv_h = dsv_heap.GetCPUDescriptorHandleForHeapStart();
        cmd.ClearRenderTargetView(rtvs[0], &[0.0, 0.0, 0.0, 0.0], None);
        cmd.ClearRenderTargetView(rtvs[1], &[0.0, 0.0, 0.0, 0.0], None);
        cmd.ClearRenderTargetView(rtvs[2], &[0.0, 0.0, 0.0, 9999.0], None);
        cmd.ClearDepthStencilView(dsv_h, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, &[]);
        cmd.OMSetRenderTargets(3, Some(rtvs.as_ptr()), false, Some(&dsv_h));
        cmd.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: rt_w as f32, Height: rt_h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        cmd.RSSetScissorRects(&[RECT { left: 0, top: 0, right: rt_w as i32, bottom: rt_h as i32 }]);
        cmd.SetDescriptorHeaps(&[Some(ms_srv_heap.clone())]);

        pass1_pipe.record_dispatch(&cmd, 0, &occ_insts, &pass1_cb, ms_srv_heap.GetGPUDescriptorHandleForHeapStart(), tlas_res.GetGPUVirtualAddress(), dummy_colors.GetGPUVirtualAddress());

        cmd_barriers(&cmd, vec![transition_barrier(&depth_res, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);

        // Build 6-Level Hi-Z Pyramid
        cmd.SetDescriptorHeaps(&[Some(hiz_heap.clone())]);
        cmd.SetComputeRootSignature(&build_rs);
        cmd.SetPipelineState(&build_pso);
        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        for (i, &dim) in mip_dims.iter().enumerate() {
            cmd_barriers(&cmd, vec![transition_barrier(&hiz_mips[i], D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            let pass_kind = if i == 0 { 0u32 } else { 1u32 };
            let b_cb = [dim, dim, pass_kind, 0u32];
            cmd.SetComputeRoot32BitConstants(0, 4, b_cb.as_ptr() as *const c_void, 0);
            cmd.SetComputeRootDescriptorTable(1, gpu_at(i)); // 0=depth, 1..5=hiz[0..4]
            cmd.SetComputeRootDescriptorTable(2, gpu_at(6 + i));
            cmd.Dispatch(dim.div_ceil(8), dim.div_ceil(8), 1);
            cmd_barriers(&cmd, vec![
                uav_barrier(&hiz_mips[i]),
                transition_barrier(&hiz_mips[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ]);
        }
        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

        // Pass 2: Wave32 Frustum + Hi-Z Occlusion Cull over all 12M+ triangles
        cmd.SetComputeRootSignature(&cull_rs);
        cmd.SetPipelineState(&cull_pso);
        cmd.SetComputeRootConstantBufferView(0, cull_cb_up.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(1, mb_def.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, ib_def.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(3, wb_def.GetGPUVirtualAddress());
        cmd.SetComputeRootDescriptorTable(4, gpu_at(12));
        cmd.SetComputeRootUnorderedAccessView(5, stats_uav.GetGPUVirtualAddress());

        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 2);
        cmd.Dispatch((all_work.len() as u32).div_ceil(64), 1, 1);
        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 3);

        cmd_barriers(&cmd, vec![
            uav_barrier(&stats_uav),
            transition_barrier(&stats_uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        ]);
        cmd.CopyBufferRegion(&stats_rb, 0, &stats_uav, 0, 32);
        cmd.ResolveQueryData(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, &ts_rb, 0);
        cmd.Close().unwrap();

        let lists = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&lists);
        ctx.signal_and_wait();

        let mut ts_ptr: *mut c_void = std::ptr::null_mut();
        ts_rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 32 }), Some(&mut ts_ptr)).unwrap();
        let ts = std::slice::from_raw_parts(ts_ptr as *const u64, 4);
        let hiz_build_ms = (ts[1].saturating_sub(ts[0]) as f64 / ctx.timestamp_freq as f64) * 1000.0;
        let hiz_cull_ms = (ts[3].saturating_sub(ts[2]) as f64 / ctx.timestamp_freq as f64) * 1000.0;
        ts_rb.Unmap(0, None);

        let mut st_ptr: *mut c_void = std::ptr::null_mut();
        stats_rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 32 }), Some(&mut st_ptr)).unwrap();
        let vals = std::slice::from_raw_parts(st_ptr as *const u32, 8);
        let (visible_meshlets, visible_tris, frustum_culled, hiz_occluded, hiz_occluded_tris) =
            (vals[0], vals[1], vals[2], vals[3], vals[4]);
        stats_rb.Unmap(0, None);

        let total_work = all_work.len() as u32;
        assert_eq!(
            visible_meshlets + frustum_culled + hiz_occluded,
            total_work,
            "Hi-Z accounting mismatch: {visible_meshlets} + {frustum_culled} + {hiz_occluded} != {total_work}"
        );
        assert!(visible_meshlets > 0, "Expected visible meshlets > 0");
        assert!(frustum_culled > 0, "Expected frustum culled > 0");
        assert!(hiz_occluded > 0 && hiz_occluded_tris > 0, "Expected Hi-Z occluded meshlets/tris > 0");

        HiZVerifyStats {
            total_work_items: total_work,
            total_tris,
            visible_meshlets,
            visible_tris,
            frustum_culled,
            hiz_occluded,
            hiz_occluded_tris,
            hiz_build_ms,
            hiz_cull_ms,
        }
    }
}
