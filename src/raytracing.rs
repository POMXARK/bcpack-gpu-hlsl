use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, uav_barrier, upload_slice, GpuContext,
};
use crate::meshes::{SceneMeshes, Vertex};
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32G32B32_FLOAT};

pub const RT_VERIFY_CS_DXIL: &[u8] = include_bytes!("../shaders/rt_verify.dxil");
pub const MAX_RT_INSTANCES: usize = 256;

#[derive(Clone, Copy)]
pub struct RtInstanceInput {
    pub mesh_idx: usize,
    pub transform_3x4: [f32; 12],
    pub color_emissive: [f32; 4],
}

pub fn tex_reflection_color(tex: u32, emissive: f32) -> [f32; 4] {
    let rgb = match tex {
        0 => [0.16, 0.16, 0.22], // asphalt
        1 => [0.14, 0.85, 1.00], // cyan/magenta neon skyscraper
        2 => [1.00, 0.62, 0.62], // pastel miami hotel
        3 => [0.85, 0.58, 0.45], // promenade tiles
        4 => [0.10, 0.72, 0.90], // ocean
        5 => [0.20, 0.72, 0.24], // palm fronds
        6 => [1.00, 0.35, 0.88], // synthwave gantry sign
        7 => [0.88, 0.75, 0.56], // beach sand
        8 => [0.95, 0.14, 0.28], // red hero/traffic car
        9 => [0.12, 0.88, 0.96], // cyan traffic car
        10 => [1.00, 0.72, 0.36], // warm storefront neon
        _ => [0.35, 0.34, 0.42], // rooftop concrete
    };
    [rgb[0], rgb[1], rgb[2], emissive]
}

pub fn supports_dxr_1_1(device: &ID3D12Device) -> bool {
    unsafe {
        let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        if device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS5,
                &mut opts5 as *mut _ as *mut c_void,
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
            )
            .is_err()
        {
            return false;
        }
        opts5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0
    }
}

pub struct RayTracingScene {
    blas_buffers: Vec<ID3D12Resource>,
    _blas_scratch: ID3D12Resource,
    tlas_scratch: [ID3D12Resource; 2],
    tlas_result: [ID3D12Resource; 2],
    instances_upload: [ID3D12Resource; 2],
    inst_colors_upload: [ID3D12Resource; 2],
}

impl RayTracingScene {
    pub fn new(
        ctx: &mut GpuContext,
        vb_buf: &ID3D12Resource,
        ib_buf: &ID3D12Resource,
        meshes: &SceneMeshes,
    ) -> Option<Self> {
        if !supports_dxr_1_1(&ctx.device) {
            return None;
        }
        unsafe {
            let dev5: ID3D12Device5 = ctx.device.cast().ok()?;
            let vb_va = vb_buf.GetGPUVirtualAddress();
            let ib_va = ib_buf.GetGPUVirtualAddress();
            let v_count = meshes.vertices.len() as u32;
            let stride = size_of::<Vertex>() as u64;

            let ranges = [
                meshes.unit_box,
                meshes.car_body,
                meshes.palm_trunk,
                meshes.palm_canopy,
                meshes.street_lamp,
            ];

            let mut blas_buffers = Vec::with_capacity(ranges.len());
            let mut max_scratch = 256u64;
            let mut build_inputs_list = Vec::with_capacity(ranges.len());

            for r in &ranges {
                let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
                    Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                    Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                    Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                        Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                            Transform3x4: 0,
                            IndexFormat: DXGI_FORMAT_R16_UINT,
                            VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                            IndexCount: r.index_count,
                            VertexCount: v_count,
                            IndexBuffer: ib_va + (r.index_offset as u64) * 2,
                            VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                                StartAddress: vb_va,
                                StrideInBytes: stride,
                            },
                        },
                    },
                };

                let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                    NumDescs: 1,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: &geom,
                    },
                };
                let mut prebuild = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
                dev5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut prebuild);
                max_scratch = max_scratch.max(prebuild.ScratchDataSizeInBytes);

                let blas_res = create_committed_buffer(
                    &ctx.device,
                    D3D12_HEAP_TYPE_DEFAULT,
                    prebuild.ResultDataMaxSizeInBytes.max(256),
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                );
                blas_buffers.push(blas_res);
                build_inputs_list.push(geom);
            }

            let blas_scratch = create_committed_buffer(
                &ctx.device,
                D3D12_HEAP_TYPE_DEFAULT,
                max_scratch,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );

            // Build all 5 static BLAS structures once on DIRECT queue
            let alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let cmd4: ID3D12GraphicsCommandList4 = list.cast().unwrap();

            for (bi, geom) in build_inputs_list.iter().enumerate() {
                let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                    NumDescs: 1,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: geom,
                    },
                };
                let build_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: blas_buffers[bi].GetGPUVirtualAddress(),
                    Inputs: inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: blas_scratch.GetGPUVirtualAddress(),
                };
                cmd4.BuildRaytracingAccelerationStructure(&build_desc, None);
                cmd_barriers(&list, vec![uav_barrier(&blas_buffers[bi])]);
            }
            list.Close().unwrap();
            let exec = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&exec);
            ctx.signal_and_wait();

            // Prebuild worst-case TLAS sizes for MAX_RT_INSTANCES
            let inst_buf_bytes = (MAX_RT_INSTANCES * size_of::<D3D12_RAYTRACING_INSTANCE_DESC>()) as u64;
            let color_buf_bytes = (MAX_RT_INSTANCES * size_of::<[f32; 4]>()) as u64;
            let instances_upload = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, inst_buf_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, inst_buf_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            ];
            let inst_colors_upload = [
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, color_buf_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
                create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, color_buf_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            ];

            let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
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

            Some(Self {
                blas_buffers,
                _blas_scratch: blas_scratch,
                tlas_scratch,
                tlas_result,
                instances_upload,
                inst_colors_upload,
            })
        }
    }

    pub unsafe fn update_and_record_tlas(
        &self,
        cmd_list: &ID3D12GraphicsCommandList,
        slot: usize,
        instances: &[RtInstanceInput],
    ) {
        let s = slot & 1;
        let count = instances.len().min(MAX_RT_INSTANCES);
        let mut descs = Vec::with_capacity(count);
        let mut cols = Vec::with_capacity(count);
        for (idx, inst) in instances.iter().take(count).enumerate() {
            descs.push(D3D12_RAYTRACING_INSTANCE_DESC {
                Transform: inst.transform_3x4,
                _bitfield1: (idx as u32 & 0x00FF_FFFF) | (0xFFu32 << 24),
                _bitfield2: 0,
                AccelerationStructure: self.blas_buffers[inst.mesh_idx].GetGPUVirtualAddress(),
            });
            cols.push(inst.color_emissive);
        }
        upload_slice(&self.instances_upload[s], &descs);
        upload_slice(&self.inst_colors_upload[s], &cols);

        let cmd4: ID3D12GraphicsCommandList4 = cmd_list.cast().unwrap();
        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
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
        cmd_barriers(cmd_list, vec![uav_barrier(&self.tlas_result[s]), uav_barrier(&self.tlas_scratch[s])]);
    }

    pub fn tlas_gpu_va(&self, slot: usize) -> u64 {
        unsafe { self.tlas_result[slot & 1].GetGPUVirtualAddress() }
    }

    pub fn inst_colors_gpu_va(&self, slot: usize) -> u64 {
        unsafe { self.inst_colors_upload[slot & 1].GetGPUVirtualAddress() }
    }
}

/// Standalone DXR 1.1 hardware BVH + Inline RayQuery self-test against analytical ray-box hits.
pub fn run_rt_verify(ctx: &mut GpuContext) -> f64 {
    assert!(supports_dxr_1_1(&ctx.device), "GPU does not support DXR Tier 1.1");
    unsafe {
        let meshes = crate::meshes::build_all_meshes();
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

        let rt = RayTracingScene::new(ctx, &vb_buf, &ib_buf, &meshes)
            .expect("RayTracingScene::new failed");

        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                },
            },
        };
        let mut blob = None;
        D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, None).unwrap();
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = ctx
            .device
            .CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
            )
            .unwrap();

        let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: RT_VERIFY_CS_DXIL.as_ptr() as *const c_void,
                BytecodeLength: RT_VERIFY_CS_DXIL.len(),
            },
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        let pso: ID3D12PipelineState = ctx.device.CreateComputePipelineState(&pso_desc).unwrap();
        let _ = ManuallyDrop::take(&mut pso_desc.pRootSignature);

        let out_default = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            256,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let out_readback = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_READBACK,
            256,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let mut ts_heap: Option<ID3D12QueryHeap> = None;
        ctx.device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut ts_heap,
            )
            .unwrap();
        let ts_heap = ts_heap.unwrap();
        let ts_readback = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_READBACK,
            16,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();

        let identity = [
            1.0f32, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        rt.update_and_record_tlas(
            &list,
            0,
            &[RtInstanceInput {
                mesh_idx: 0,
                transform_3x4: identity,
                color_emissive: [1.0, 1.0, 1.0, 0.0],
            }],
        );

        cmd_barriers(
            &list,
            vec![crate::gpu_decoder::transition_barrier(
                &out_default,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )],
        );
        list.SetComputeRootSignature(&root_sig);
        list.SetPipelineState(&pso);
        list.SetComputeRootShaderResourceView(0, rt.tlas_gpu_va(0));
        list.SetComputeRootUnorderedAccessView(1, out_default.GetGPUVirtualAddress());
        list.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        list.Dispatch(1, 1, 1);
        list.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        cmd_barriers(
            &list,
            vec![crate::gpu_decoder::transition_barrier(
                &out_default,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )],
        );
        list.CopyBufferRegion(&out_readback, 0, &out_default, 0, 64);
        list.ResolveQueryData(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &ts_readback, 0);
        list.Close().unwrap();

        let lists = [Some(list.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&lists);
        ctx.signal_and_wait();

        let mut ptr: *mut c_void = std::ptr::null_mut();
        out_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 64 }), Some(&mut ptr)).unwrap();
        let results = std::slice::from_raw_parts(ptr as *const [f32; 4], 4);
        assert_eq!(results[0][0], 1.0, "RT ray 0 must hit");
        assert!((results[0][1] - 4.5).abs() < 1e-3, "RT ray 0 distance expected 4.5, got {}", results[0][1]);
        assert_eq!(results[1][0], 1.0, "RT ray 1 must hit");
        assert!((results[1][1] - 2.5).abs() < 1e-3, "RT ray 1 distance expected 2.5, got {}", results[1][1]);
        assert_eq!(results[2][0], 0.0, "RT ray 2 must miss");
        assert_eq!(results[3][0], 1.0, "RT ray 3 must hit");
        assert!((results[3][1] - 9.5).abs() < 1e-3, "RT ray 3 distance expected 9.5, got {}", results[3][1]);
        out_readback.Unmap(0, None);

        ts_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut ptr)).unwrap();
        let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
        let kms = (ts[1].saturating_sub(ts[0]) as f64 / ctx.timestamp_freq as f64) * 1000.0;
        ts_readback.Unmap(0, None);
        kms
    }
}
