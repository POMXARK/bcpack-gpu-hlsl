use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, heap_props, transition_barrier, uav_barrier,
    upload_slice, GpuContext,
};
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop};
use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub const RTGI_DENOISE_CS_DXIL: &[u8] = include_bytes!("../shaders/rtgi_denoise.dxil");
pub const RTGI_COMP_VS_DXIL: &[u8] = include_bytes!("../shaders/rtgi_comp_vs.dxil");
pub const RTGI_COMP_PS_DXIL: &[u8] = include_bytes!("../shaders/rtgi_comp_ps.dxil");

#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = bits & 0x007F_FFFF;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;
    let fbits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x03FF) << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(fbits)
}

pub struct RtgiPipeline {
    pub width: u32,
    pub height: u32,
    pub direct_hdr: ID3D12Resource,
    pub raw_rtgi: ID3D12Resource,
    pub normal_depth: ID3D12Resource,
    pub denoise_a: ID3D12Resource,
    pub denoise_b: ID3D12Resource,
    rtv_heap: ID3D12DescriptorHeap,
    rtv_inc: u32,
    srv_uav_heap: ID3D12DescriptorHeap,
    srv_inc: u32,
    denoise_root_sig: ID3D12RootSignature,
    denoise_pso: ID3D12PipelineState,
    comp_root_sig: ID3D12RootSignature,
    comp_pso: ID3D12PipelineState,
}

unsafe fn create_fp16_tex(
    device: &ID3D12Device,
    w: u32,
    h: u32,
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
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
        ..Default::default()
    };
    let cv = clear_rgba.map(|col| D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: col },
    });
    let mut res: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(
            &heap_props(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &desc,
            init_state,
            cv.as_ref().map(|c| c as *const _),
            &mut res,
        )
        .expect("CreateCommittedResource FP16 texture");
    res.unwrap()
}

impl RtgiPipeline {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        unsafe {
            let direct_hdr = create_fp16_tex(
                &ctx.device,
                width,
                height,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                Some([0.18, 0.08, 0.28, 1.0]),
            );
            let raw_rtgi = create_fp16_tex(
                &ctx.device,
                width,
                height,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                Some([0.0, 0.0, 0.0, 0.0]),
            );
            let normal_depth = create_fp16_tex(
                &ctx.device,
                width,
                height,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                Some([0.0, 0.0, 0.0, 9999.0]),
            );
            let denoise_a = create_fp16_tex(
                &ctx.device,
                width,
                height,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
                None,
            );
            let denoise_b = create_fp16_tex(
                &ctx.device,
                width,
                height,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
                None,
            );

            let rtv_heap: ID3D12DescriptorHeap = ctx
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 3,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .unwrap();
            let rtv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            let rtv_base = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            for (i, res) in [&direct_hdr, &raw_rtgi, &normal_depth].iter().enumerate() {
                ctx.device.CreateRenderTargetView(
                    *res,
                    None,
                    D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: rtv_base.ptr + i * rtv_inc as usize,
                    },
                );
            }

            let srv_uav_heap: ID3D12DescriptorHeap = ctx
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 8,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let srv_inc = ctx.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let cpu_base = srv_uav_heap.GetCPUDescriptorHandleForHeapStart();

            let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            };
            let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 },
                },
            };
            let slot_cpu = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_base.ptr + i * srv_inc as usize,
            };

            // Pass 1: [0]=raw_rtgi SRV, [1]=normal_depth SRV, [2]=denoise_a UAV
            ctx.device.CreateShaderResourceView(&raw_rtgi, Some(&srv_desc), slot_cpu(0));
            ctx.device.CreateShaderResourceView(&normal_depth, Some(&srv_desc), slot_cpu(1));
            ctx.device.CreateUnorderedAccessView(&denoise_a, None, Some(&uav_desc), slot_cpu(2));
            // Pass 2: [3]=denoise_a SRV, [4]=normal_depth SRV, [5]=denoise_b UAV
            ctx.device.CreateShaderResourceView(&denoise_a, Some(&srv_desc), slot_cpu(3));
            ctx.device.CreateShaderResourceView(&normal_depth, Some(&srv_desc), slot_cpu(4));
            ctx.device.CreateUnorderedAccessView(&denoise_b, None, Some(&uav_desc), slot_cpu(5));
            // Composite: [6]=direct_hdr SRV, [7]=denoise_b SRV
            ctx.device.CreateShaderResourceView(&direct_hdr, Some(&srv_desc), slot_cpu(6));
            ctx.device.CreateShaderResourceView(&denoise_b, Some(&srv_desc), slot_cpu(7));

            // Compute Denoise Root Signature + PSO
            let d_srv_rng = D3D12_DESCRIPTOR_RANGE1 {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 2,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
                OffsetInDescriptorsFromTableStart: 0,
            };
            let d_uav_rng = D3D12_DESCRIPTOR_RANGE1 {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                OffsetInDescriptorsFromTableStart: 0,
            };
            let d_params = [
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 4,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                            NumDescriptorRanges: 1,
                            pDescriptorRanges: &d_srv_rng,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                            NumDescriptorRanges: 1,
                            pDescriptorRanges: &d_uav_rng,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let d_rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                    Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                        NumParameters: d_params.len() as u32,
                        pParameters: d_params.as_ptr(),
                        NumStaticSamplers: 0,
                        pStaticSamplers: std::ptr::null(),
                        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                    },
                },
            };
            let mut d_blob = None;
            D3D12SerializeVersionedRootSignature(&d_rs_desc, &mut d_blob, None).unwrap();
            let d_blob = d_blob.unwrap();
            let denoise_root_sig: ID3D12RootSignature = ctx
                .device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(d_blob.GetBufferPointer() as *const u8, d_blob.GetBufferSize()),
                )
                .unwrap();

            let mut d_pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(denoise_root_sig.clone())),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: RTGI_DENOISE_CS_DXIL.as_ptr() as *const c_void,
                    BytecodeLength: RTGI_DENOISE_CS_DXIL.len(),
                },
                ..Default::default()
            };
            let denoise_pso: ID3D12PipelineState = ctx.device.CreateComputePipelineState(&d_pso_desc).unwrap();
            let _ = ManuallyDrop::take(&mut d_pso_desc.pRootSignature);

            // Composite Root Signature + Graphics PSO (targets R8G8B8A8_UNORM backbuffer)
            let c_srv_rng = D3D12_DESCRIPTOR_RANGE1 {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 2,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
                OffsetInDescriptorsFromTableStart: 0,
            };
            let c_params = [
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 4,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                            NumDescriptorRanges: 1,
                            pDescriptorRanges: &c_srv_rng,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                },
            ];
            let lin_clamp = D3D12_STATIC_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MaxLOD: D3D12_FLOAT32_MAX,
                ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                ShaderRegister: 0,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                ..Default::default()
            };
            let c_rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                    Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                        NumParameters: c_params.len() as u32,
                        pParameters: c_params.as_ptr(),
                        NumStaticSamplers: 1,
                        pStaticSamplers: &lin_clamp,
                        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                    },
                },
            };
            let mut c_blob = None;
            D3D12SerializeVersionedRootSignature(&c_rs_desc, &mut c_blob, None).unwrap();
            let c_blob = c_blob.unwrap();
            let comp_root_sig: ID3D12RootSignature = ctx
                .device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(c_blob.GetBufferPointer() as *const u8, c_blob.GetBufferSize()),
                )
                .unwrap();

            let mut rt_formats = [DXGI_FORMAT_UNKNOWN; 8];
            rt_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let mut c_pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(comp_root_sig.clone())),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: RTGI_COMP_VS_DXIL.as_ptr() as *const c_void,
                    BytecodeLength: RTGI_COMP_VS_DXIL.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: RTGI_COMP_PS_DXIL.as_ptr() as *const c_void,
                    BytecodeLength: RTGI_COMP_PS_DXIL.len(),
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
                NumRenderTargets: 1,
                RTVFormats: rt_formats,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let comp_pso: ID3D12PipelineState = ctx.device.CreateGraphicsPipelineState(&c_pso_desc).unwrap();
            let _ = ManuallyDrop::take(&mut c_pso_desc.pRootSignature);

            Self {
                width,
                height,
                direct_hdr,
                raw_rtgi,
                normal_depth,
                denoise_a,
                denoise_b,
                rtv_heap,
                rtv_inc,
                srv_uav_heap,
                srv_inc,
                denoise_root_sig,
                denoise_pso,
                comp_root_sig,
                comp_pso,
            }
        }
    }

    pub unsafe fn begin_gbuffer_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        dsv_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        let base = self.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        let rtvs = [
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr },
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + self.rtv_inc as usize },
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + 2 * self.rtv_inc as usize },
        ];
        cmd.ClearRenderTargetView(rtvs[0], &[0.18, 0.08, 0.28, 1.0], None);
        cmd.ClearRenderTargetView(rtvs[1], &[0.0, 0.0, 0.0, 0.0], None);
        cmd.ClearRenderTargetView(rtvs[2], &[0.0, 0.0, 0.0, 9999.0], None);
        cmd.ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, &[]);
        cmd.OMSetRenderTargets(3, Some(rtvs.as_ptr()), false, Some(&dsv_handle));

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

    pub unsafe fn run_denoise_passes_only(&self, cmd: &ID3D12GraphicsCommandList) {
        let gx = self.width.div_ceil(8);
        let gy = self.height.div_ceil(8);
        let gpu_base = self.srv_uav_heap.GetGPUDescriptorHandleForHeapStart();
        let slot_gpu = |i: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu_base.ptr + (i * self.srv_inc as usize) as u64,
        };

        cmd.SetDescriptorHeaps(&[Some(self.srv_uav_heap.clone())]);
        cmd.SetComputeRootSignature(&self.denoise_root_sig);
        cmd.SetPipelineState(&self.denoise_pso);

        // Pass 1: raw_rtgi -> denoise_a (step = 1)
        cmd_barriers(
            cmd,
            vec![transition_barrier(&self.denoise_a, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)],
        );
        let c1 = [self.width, self.height, 1u32, 0u32];
        cmd.SetComputeRoot32BitConstants(0, 4, c1.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootDescriptorTable(1, slot_gpu(0));
        cmd.SetComputeRootDescriptorTable(2, slot_gpu(2));
        cmd.Dispatch(gx, gy, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.denoise_a),
                transition_barrier(&self.denoise_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );

        // Pass 2: denoise_a -> denoise_b (step = 2)
        let c2 = [self.width, self.height, 2u32, 0u32];
        cmd.SetComputeRoot32BitConstants(0, 4, c2.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootDescriptorTable(1, slot_gpu(3));
        cmd.SetComputeRootDescriptorTable(2, slot_gpu(5));
        cmd.Dispatch(gx, gy, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.denoise_b),
                transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.denoise_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
            ],
        );
    }

    pub unsafe fn run_denoise_and_composite(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        backbuffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        out_w: u32,
        out_h: u32,
        rtgi_enabled: bool,
    ) {
        let srv_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.direct_hdr, D3D12_RESOURCE_STATE_RENDER_TARGET, srv_state),
                transition_barrier(&self.raw_rtgi, D3D12_RESOURCE_STATE_RENDER_TARGET, srv_state),
                transition_barrier(&self.normal_depth, D3D12_RESOURCE_STATE_RENDER_TARGET, srv_state),
            ],
        );

        if rtgi_enabled {
            self.run_denoise_passes_only(cmd);
        } else {
            cmd_barriers(
                cmd,
                vec![transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)],
            );
        }

        let gpu_base = self.srv_uav_heap.GetGPUDescriptorHandleForHeapStart();
        let slot_gpu = |i: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu_base.ptr + (i * self.srv_inc as usize) as u64,
        };

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

        cmd.SetDescriptorHeaps(&[Some(self.srv_uav_heap.clone())]);
        cmd.SetGraphicsRootSignature(&self.comp_root_sig);
        cmd.SetPipelineState(&self.comp_pso);
        let comp_consts = [if rtgi_enabled { 1.0f32 } else { 0.0f32 }, 1.08f32, 0.0f32, 0.0f32];
        cmd.SetGraphicsRoot32BitConstants(0, 4, comp_consts.as_ptr() as *const c_void, 0);
        cmd.SetGraphicsRootDescriptorTable(1, slot_gpu(6));
        cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd.DrawInstanced(3, 1, 0, 0);

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.direct_hdr, srv_state, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition_barrier(&self.raw_rtgi, srv_state, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition_barrier(&self.normal_depth, srv_state, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition_barrier(&self.denoise_b, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
            ],
        );
    }
}

pub struct RtgiVerifyStats {
    pub tlas_instances: usize,
    pub tlas_ms: f64,
    pub raw_var: f64,
    pub denoised_var: f64,
    pub edge_jump: f64,
}

pub fn run_rtgi_verify(ctx: &mut GpuContext) -> RtgiVerifyStats {
    unsafe {
        // 1) Scale test: Build 1,536-instance DXR 1.1 TLAS on RT Cores
        let dev5: ID3D12Device5 = ctx.device.cast().expect("ID3D12Device5");
        let tri_verts = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let vb = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, 64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        upload_slice(&vb, &tri_verts);

        let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                    VertexCount: 3,
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: vb.GetGPUVirtualAddress(),
                        StrideInBytes: 12,
                    },
                    ..Default::default()
                },
            },
        };
        let blas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: &geom },
        };
        let mut blas_pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &mut blas_pre);
        let blas_res = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, blas_pre.ResultDataMaxSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
        let blas_scr = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, blas_pre.ScratchDataSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

        let tlas_instances = 1536usize;
        let inst_descs: Vec<D3D12_RAYTRACING_INSTANCE_DESC> = (0..tlas_instances)
            .map(|i| D3D12_RAYTRACING_INSTANCE_DESC {
                Transform: [
                    1.0, 0.0, 0.0, (i % 64) as f32 * 4.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, (i / 64) as f32 * 4.0,
                ],
                _bitfield1: (i as u32 & 0x00FF_FFFF) | (0xFFu32 << 24),
                _bitfield2: 0,
                AccelerationStructure: blas_res.GetGPUVirtualAddress(),
            })
            .collect();
        let inst_up = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (tlas_instances * size_of::<D3D12_RAYTRACING_INSTANCE_DESC>()) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        upload_slice(&inst_up, &inst_descs);

        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
            NumDescs: tlas_instances as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: inst_up.GetGPUVirtualAddress(),
            },
        };
        let mut tlas_pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_pre);
        let tlas_res = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, tlas_pre.ResultDataMaxSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
        let tlas_scr = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, tlas_pre.ScratchDataSizeInBytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

        let mut ts_heap_opt: Option<ID3D12QueryHeap> = None;
        ctx.device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut ts_heap_opt,
            )
            .unwrap();
        let ts_heap = ts_heap_opt.unwrap();
        let ts_rb = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        let cmd4: ID3D12GraphicsCommandList4 = cmd.cast().unwrap();

        cmd4.BuildRaytracingAccelerationStructure(
            &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: blas_res.GetGPUVirtualAddress(),
                Inputs: blas_inputs,
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: blas_scr.GetGPUVirtualAddress(),
            },
            None,
        );
        cmd_barriers(&cmd, vec![uav_barrier(&blas_res)]);

        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        cmd4.BuildRaytracingAccelerationStructure(
            &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: tlas_res.GetGPUVirtualAddress(),
                Inputs: tlas_inputs,
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: tlas_scr.GetGPUVirtualAddress(),
            },
            None,
        );
        cmd_barriers(&cmd, vec![uav_barrier(&tlas_res), uav_barrier(&tlas_scr)]);
        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        cmd.ResolveQueryData(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &ts_rb, 0);

        cmd.Close().unwrap();
        let lists = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&lists);
        ctx.signal_and_wait();

        let mut ts_ptr: *mut c_void = std::ptr::null_mut();
        ts_rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut ts_ptr)).unwrap();
        let ts = std::slice::from_raw_parts(ts_ptr as *const u64, 2);
        let tlas_ms = ((ts[1].saturating_sub(ts[0])) as f64 / ctx.timestamp_freq as f64) * 1000.0;
        ts_rb.Unmap(0, None);

        // 2) Bilateral Edge-Aware 2-Pass A-Trous RTGI Denoiser verification (64x64 FP16 synthetic checkerboard noise + sharp edge at x=32)
        let w = 64u32;
        let h = 64u32;
        let pipe = RtgiPipeline::new(ctx, w, h);
        let row_pitch = w * 8; // 64 * 8 = 512 bytes (multiple of 256!)
        let total_bytes = (row_pitch * h) as usize;

        let mut raw_gi_f16 = vec![0u16; (w * h * 4) as usize];
        let mut nd_f16 = vec![0u16; (w * h * 4) as usize];
        let mut left_raw_vals = Vec::with_capacity(2048);
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 4) as usize;
                let noise = if ((x + y) & 1) == 0 { 0.18f32 } else { -0.18f32 };
                if x < 32 {
                    let v = 0.22 + noise;
                    left_raw_vals.push(v);
                    raw_gi_f16[idx] = f32_to_f16(v);
                    raw_gi_f16[idx + 1] = f32_to_f16(v * 0.5);
                    raw_gi_f16[idx + 2] = f32_to_f16(v * 0.2);
                    raw_gi_f16[idx + 3] = f32_to_f16(1.0);
                    nd_f16[idx] = f32_to_f16(0.0);
                    nd_f16[idx + 1] = f32_to_f16(1.0);
                    nd_f16[idx + 2] = f32_to_f16(0.0);
                    nd_f16[idx + 3] = f32_to_f16(10.0);
                } else {
                    let v = 1.05 + noise;
                    raw_gi_f16[idx] = f32_to_f16(v);
                    raw_gi_f16[idx + 1] = f32_to_f16(v * 0.8);
                    raw_gi_f16[idx + 2] = f32_to_f16(v * 0.9);
                    raw_gi_f16[idx + 3] = f32_to_f16(1.0);
                    nd_f16[idx] = f32_to_f16(1.0);
                    nd_f16[idx + 1] = f32_to_f16(0.0);
                    nd_f16[idx + 2] = f32_to_f16(0.0);
                    nd_f16[idx + 3] = f32_to_f16(40.0);
                }
            }
        }
        let gi_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, total_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let nd_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, total_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let gi_rb = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, total_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        upload_slice(&gi_up, &raw_gi_f16);
        upload_slice(&nd_up, &nd_f16);

        alloc.Reset().unwrap();
        cmd.Reset(&alloc, None).unwrap();
        cmd_barriers(
            &cmd,
            vec![
                transition_barrier(&pipe.raw_rtgi, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&pipe.normal_depth, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST),
            ],
        );

        let copy_buf_to_tex = |tex: &ID3D12Resource, buf: &ID3D12Resource| {
            let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(tex.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let mut src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(buf.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                            Width: w,
                            Height: h,
                            Depth: 1,
                            RowPitch: row_pitch,
                        },
                    },
                },
            };
            cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            let _ = ManuallyDrop::take(&mut dst.pResource);
            let _ = ManuallyDrop::take(&mut src.pResource);
        };
        copy_buf_to_tex(&pipe.raw_rtgi, &gi_up);
        copy_buf_to_tex(&pipe.normal_depth, &nd_up);

        let srv_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        cmd_barriers(
            &cmd,
            vec![
                transition_barrier(&pipe.raw_rtgi, D3D12_RESOURCE_STATE_COPY_DEST, srv_state),
                transition_barrier(&pipe.normal_depth, D3D12_RESOURCE_STATE_COPY_DEST, srv_state),
            ],
        );

        pipe.run_denoise_passes_only(&cmd);

        cmd_barriers(
            &cmd,
            vec![transition_barrier(&pipe.denoise_b, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE)],
        );
        let mut rb_dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(gi_rb.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                        Width: w,
                        Height: h,
                        Depth: 1,
                        RowPitch: row_pitch,
                    },
                },
            },
        };
        let mut rb_src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(pipe.denoise_b.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        cmd.CopyTextureRegion(&rb_dst, 0, 0, 0, &rb_src, None);
        let _ = ManuallyDrop::take(&mut rb_dst.pResource);
        let _ = ManuallyDrop::take(&mut rb_src.pResource);

        cmd.Close().unwrap();
        let lists2 = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&lists2);
        ctx.signal_and_wait();

        let mut rb_ptr: *mut c_void = std::ptr::null_mut();
        gi_rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: total_bytes }), Some(&mut rb_ptr)).unwrap();
        let out_f16 = std::slice::from_raw_parts(rb_ptr as *const u16, (w * h * 4) as usize);

        let mut left_den_vals = Vec::with_capacity(2048);
        let mut near_left_sum = 0.0f64;
        let mut near_right_sum = 0.0f64;
        for y in 4..60 {
            for x in 4..28 {
                let r = f16_to_f32(out_f16[((y * w + x) * 4) as usize]);
                left_den_vals.push(r);
            }
            near_left_sum += f16_to_f32(out_f16[((y * w + 31) * 4) as usize]) as f64;
            near_right_sum += f16_to_f32(out_f16[((y * w + 32) * 4) as usize]) as f64;
        }
        gi_rb.Unmap(0, None);

        let variance = |xs: &[f32]| -> f64 {
            let mean = xs.iter().map(|&v| v as f64).sum::<f64>() / xs.len() as f64;
            xs.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / xs.len() as f64
        };
        let raw_var = variance(&left_raw_vals);
        let denoised_var = variance(&left_den_vals);
        let edge_jump = (near_right_sum - near_left_sum) / 56.0;

        assert!(
            denoised_var < raw_var * 0.35,
            "RTGI denoiser failed variance reduction: raw={raw_var:.5}, denoised={denoised_var:.5}"
        );
        assert!(
            edge_jump > 0.65,
            "RTGI denoiser failed bilateral edge preservation: edge_jump={edge_jump:.4}"
        );

        RtgiVerifyStats {
            tlas_instances,
            tlas_ms,
            raw_var,
            denoised_var,
            edge_jump,
        }
    }
}
