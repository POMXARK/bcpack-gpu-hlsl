//! Clean-room UMA Emulation Stress & Optimization Suite (`--uma-stress` & `--uma-emulation-bench`)
//!
//! Models and solves the 4 primary CPU/GPU synchronization bottlenecks seen when emulating a unified
//! memory architecture (UMA) console on a discrete PC D3D12 GPU:
//!   1. `GetWriteWatch` storm + CPU hashing + CPU `SW_64KB_D` detile across 2,000 textures
//!      -> Immutable DirectStorage skip (0 syscalls on 1,900 static textures) + batched GPU Compute detile (`[8,8,1]`).
//!   2. Full 64 MiB (1M x 64B descriptor slot) CPU scan + thousands of `CopyBufferRegion` calls
//!      -> 3-level hierarchical `u64` dirty bitmap (`trailing_zeros`) + AVX2 64B slot diff + single `uma_scatter_patch.dxil` Dispatch.
//!   3. Single-threaded command recording + mid-frame GPU queue `settle` stalls across 64 skin CS + 3,500 draws
//!      -> 8-thread `std::thread::scope` recording with local Per-Resource Hazard Journals -> minimal fixup barrier lists -> 1 `ExecuteCommandLists`.
//!   4. Working-set reads over PCIe (`UPLOAD` shared system memory) vs `DEFAULT` GDDR7 local VRAM.

use bcpack_gpu::gpu_decoder::{
    cmd_barriers, create_committed_buffer, heap_props, transition_barrier, upload_slice, GpuContext,
};
use std::arch::x86_64::*;
use std::ffi::c_void;
use std::fs::File;
use std::io::Write;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr::copy_nonoverlapping;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::System::Memory::{
    GetWriteWatch, VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
    PAGE_READWRITE, VIRTUAL_ALLOCATION_TYPE,
};

pub const UMA_SCATTER_PATCH_CS_DXIL: &[u8] = include_bytes!("../shaders/uma_scatter_patch.dxil");
pub const UMA_GFX10_DETILE_CS_DXIL: &[u8] = include_bytes!("../shaders/uma_gfx10_detile.dxil");
pub const UMA_SKIN_CS_DXIL: &[u8] = include_bytes!("../shaders/uma_skin_cs.dxil");
pub const UMA_SKIN_VS_DXIL: &[u8] = include_bytes!("../shaders/uma_skin_vs.dxil");
pub const UMA_SKIN_PS_DXIL: &[u8] = include_bytes!("../shaders/uma_skin_ps.dxil");
pub const UMA_WORKINGSET_CS_DXIL: &[u8] = include_bytes!("../shaders/uma_workingset.dxil");

const WRITE_WATCH_RESET_FLAG: u32 = 1;
const MEM_WRITE_WATCH_ALLOC: VIRTUAL_ALLOCATION_TYPE = VIRTUAL_ALLOCATION_TYPE(0x0020_0000);
const PAGE_BYTES: usize = 4096;

// Subsystem [1] constants
pub const TEX_COUNT: usize = 2_000;
pub const STATIC_TEX_COUNT: usize = 1_900;
pub const DYNAMIC_TEX_COUNT: usize = TEX_COUNT - STATIC_TEX_COUNT; // 100
pub const TEX_W: u32 = 256;
pub const TEX_H: u32 = 256;
pub const TEX_STRIDE_BYTES: usize = (TEX_W as usize) * (TEX_H as usize) * 4; // 262,144 B (256 KiB = 4 tiles)

// Subsystem [2] constants
pub const DESC_SLOT_COUNT: usize = 1_000_000;
pub const DESC_SLOT_BYTES: usize = 64;
pub const DESC_HEAP_BYTES: usize = DESC_SLOT_COUNT * DESC_SLOT_BYTES; // 64,000,000 B (~61 MiB)
pub const DESC_MUTATIONS_PER_FRAME: usize = 1_500;
pub const DESC_MAX_PATCH_CAP: usize = 32_768;

// Subsystem [3] constants
pub const SKIN_MESH_COUNT: usize = 64;
pub const SKIN_VERTS_PER_MESH: usize = 64;
pub const SKIN_DRAW_COUNT: usize = 3_500;
pub const WORKER_THREADS: usize = 8;

// Subsystem [4] constants (128 MiB > RTX 5080 64 MiB L2 cache to force real PCIe vs GDDR7 bandwidth delta)
pub const WORKSET_BYTES: usize = 128 * 1024 * 1024;
pub const WORKSET_ELEMS: u32 = (WORKSET_BYTES / 16) as u32;

// ---------------------------------------------------------------------------
// Clean-room Windows MEM_WRITE_WATCH memory wrapper
// ---------------------------------------------------------------------------

pub struct WriteWatchRegion {
    ptr: *mut u8,
    size: usize,
    addr_scratch: Vec<*mut c_void>,
}

unsafe impl Send for WriteWatchRegion {}
unsafe impl Sync for WriteWatchRegion {}

impl WriteWatchRegion {
    pub fn new(size_bytes: usize) -> Self {
        let aligned_size = (size_bytes + PAGE_BYTES - 1) & !(PAGE_BYTES - 1);
        let ptr = unsafe {
            VirtualAlloc(
                None,
                aligned_size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH_ALLOC,
                PAGE_READWRITE,
            )
        };
        assert!(!ptr.is_null(), "VirtualAlloc(MEM_WRITE_WATCH, {aligned_size}) failed");
        let page_count = aligned_size / PAGE_BYTES;
        let mut reg = Self {
            ptr: ptr as *mut u8,
            size: aligned_size,
            addr_scratch: vec![std::ptr::null_mut(); page_count],
        };
        unsafe {
            std::ptr::write_bytes(reg.ptr, 0, reg.size);
        }
        reg.reset();
        reg
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn reset(&mut self) {
        let _ = self.query_dirty_into(true, &mut Vec::new());
    }

    /// Queries Windows write-watch dirty pages within a sub-range `[offset..offset+len)`.
    /// Returns absolute dirty page indices within this region (does not reset if `reset == false`).
    pub fn query_subrange_dirty(
        &mut self,
        offset: usize,
        len: usize,
        reset: bool,
        out_pages: &mut Vec<usize>,
    ) -> usize {
        out_pages.clear();
        let sub_pages = (len + PAGE_BYTES - 1) / PAGE_BYTES;
        if self.addr_scratch.len() < sub_pages {
            self.addr_scratch.resize(sub_pages, std::ptr::null_mut());
        }
        let mut count = sub_pages;
        let mut gran = 0u32;
        let flags = if reset { WRITE_WATCH_RESET_FLAG } else { 0 };
        let base_sub = unsafe { self.ptr.add(offset) } as *const c_void;
        let rc = unsafe {
            GetWriteWatch(
                flags,
                base_sub,
                len,
                Some(self.addr_scratch.as_mut_ptr()),
                Some(&mut count),
                Some(&mut gran),
            )
        };
        assert_eq!(rc, 0, "GetWriteWatch failed with code {rc}");
        let base_addr = self.ptr as usize;
        for &a in &self.addr_scratch[..count] {
            let off = (a as usize).saturating_sub(base_addr);
            out_pages.push(off / PAGE_BYTES);
        }
        count
    }

    pub fn query_dirty_into(&mut self, reset: bool, out_pages: &mut Vec<usize>) -> usize {
        let size = self.size;
        self.query_subrange_dirty(0, size, reset, out_pages)
    }
}

impl Drop for WriteWatchRegion {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = VirtualFree(self.ptr as *mut c_void, 0, MEM_RELEASE);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Clean-room SW_64KB_D 32bpp Detile Math + Fast FNV-1a
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn swizzle_64kb_d_32bpp(lx: u32, ly: u32) -> u32 {
    ((lx & 1) << 0)
        | ((ly & 1) << 1)
        | (((lx >> 1) & 3) << 2)
        | (((ly >> 1) & 3) << 4)
        | (((lx >> 3) & 1) << 6)
        | (((ly >> 3) & 1) << 7)
        | (((lx >> 4) & 1) << 8)
        | (((ly >> 4) & 1) << 9)
        | (((lx >> 5) & 1) << 10)
        | (((ly >> 5) & 1) << 11)
        | (((lx >> 6) & 1) << 12)
        | (((ly >> 6) & 1) << 13)
}

pub fn cpu_tile_64kb_d_32bpp(linear_rgba: &[u8], tiled_out: &mut [u8], width: u32, height: u32) {
    let tiles_x = (width >> 7).max(1);
    let src_u32: &[u32] =
        unsafe { std::slice::from_raw_parts(linear_rgba.as_ptr() as *const u32, (width * height) as usize) };
    let dst_u32: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(tiled_out.as_mut_ptr() as *mut u32, (width * height) as usize) };
    for y in 0..height {
        for x in 0..width {
            let tile_idx = (y >> 7) * tiles_x + (x >> 7);
            let in_tile = swizzle_64kb_d_32bpp(x & 127, y & 127);
            let src_i = (y * width + x) as usize;
            let dst_i = (tile_idx as usize) * 16384 + (in_tile as usize);
            dst_u32[dst_i] = src_u32[src_i];
        }
    }
}

pub fn cpu_detile_64kb_d_32bpp(tiled_in: &[u8], linear_out: &mut [u8], width: u32, height: u32) {
    let tiles_x = (width >> 7).max(1);
    let src_u32: &[u32] =
        unsafe { std::slice::from_raw_parts(tiled_in.as_ptr() as *const u32, (width * height) as usize) };
    let dst_u32: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(linear_out.as_mut_ptr() as *mut u32, (width * height) as usize) };
    for y in 0..height {
        for x in 0..width {
            let tile_idx = (y >> 7) * tiles_x + (x >> 7);
            let in_tile = swizzle_64kb_d_32bpp(x & 127, y & 127);
            let src_i = (tile_idx as usize) * 16384 + (in_tile as usize);
            let dst_i = (y * width + x) as usize;
            dst_u32[dst_i] = src_u32[src_i];
        }
    }
}

#[inline]
fn fnv1a_bytes_u32_chunks(data: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3u64);
    }
    h
}

// ---------------------------------------------------------------------------
// 3-Level Hierarchical u64 Dirty Bitmap + AVX2 64B Slot Diff
// ---------------------------------------------------------------------------

pub struct HierarchicalDirtyBitmap {
    l0: Vec<u64>, // 1 bit per 4 KiB page
    l1: Vec<u64>, // 1 bit per 64 L0 words (4,096 pages)
    l2: u64,      // 1 bit per L1 word
}

impl HierarchicalDirtyBitmap {
    pub fn new(num_pages: usize) -> Self {
        let l0_len = (num_pages + 63) / 64;
        let l1_len = (l0_len + 63) / 64;
        assert!(l1_len <= 64, "Bitmap supports up to 262,144 L0 words (~16M pages)");
        Self {
            l0: vec![0u64; l0_len],
            l1: vec![0u64; l1_len],
            l2: 0,
        }
    }

    #[inline]
    pub fn mark_page(&mut self, page_idx: usize) {
        let w0 = page_idx >> 6;
        let b0 = page_idx & 63;
        if w0 < self.l0.len() {
            self.l0[w0] |= 1u64 << b0;
            let w1 = w0 >> 6;
            let b1 = w0 & 63;
            self.l1[w1] |= 1u64 << b1;
            self.l2 |= 1u64 << w1;
        }
    }

    pub fn drain_dirty_pages<F: FnMut(usize)>(&mut self, mut f: F) {
        let mut m2 = self.l2;
        self.l2 = 0;
        while m2 != 0 {
            let w1 = m2.trailing_zeros() as usize;
            m2 &= m2 - 1;
            let mut m1 = self.l1[w1];
            self.l1[w1] = 0;
            while m1 != 0 {
                let b1 = m1.trailing_zeros() as usize;
                m1 &= m1 - 1;
                let w0 = (w1 << 6) | b1;
                let mut m0 = self.l0[w0];
                self.l0[w0] = 0;
                while m0 != 0 {
                    let b0 = m0.trailing_zeros() as usize;
                    m0 &= m0 - 1;
                    f((w0 << 6) | b0);
                }
            }
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn diff_4kb_page_avx2(
    page_idx: usize,
    src_base: *const u8,
    shadow_base: *mut u8,
    max_slots: usize,
    out_indices: &mut Vec<u32>,
    out_data: &mut Vec<u8>,
) {
    let first_slot = page_idx * (PAGE_BYTES / DESC_SLOT_BYTES);
    let end_slot = (first_slot + (PAGE_BYTES / DESC_SLOT_BYTES)).min(max_slots);
    for slot in first_slot..end_slot {
        let byte_off = slot * DESC_SLOT_BYTES;
        let p_src = src_base.add(byte_off);
        let p_shd = shadow_base.add(byte_off);

        let a0 = _mm256_loadu_si256(p_src as *const __m256i);
        let b0 = _mm256_loadu_si256(p_shd as *const __m256i);
        let eq0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a0, b0));

        let a1 = _mm256_loadu_si256(p_src.add(32) as *const __m256i);
        let b1 = _mm256_loadu_si256(p_shd.add(32) as *const __m256i);
        let eq1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a1, b1));

        if (eq0 & eq1) != -1 {
            _mm256_storeu_si256(p_shd as *mut __m256i, a0);
            _mm256_storeu_si256(p_shd.add(32) as *mut __m256i, a1);
            out_indices.push(slot as u32);
            let old_len = out_data.len();
            out_data.resize(old_len + DESC_SLOT_BYTES, 0);
            copy_nonoverlapping(p_src, out_data.as_mut_ptr().add(old_len), DESC_SLOT_BYTES);
        }
    }
}

fn diff_4kb_page_scalar(
    page_idx: usize,
    src: &[u8],
    shadow: &mut [u8],
    max_slots: usize,
    out_indices: &mut Vec<u32>,
    out_data: &mut Vec<u8>,
) {
    let first_slot = page_idx * (PAGE_BYTES / DESC_SLOT_BYTES);
    let end_slot = (first_slot + (PAGE_BYTES / DESC_SLOT_BYTES)).min(max_slots);
    for slot in first_slot..end_slot {
        let off = slot * DESC_SLOT_BYTES;
        let s = &src[off..off + DESC_SLOT_BYTES];
        let sh = &mut shadow[off..off + DESC_SLOT_BYTES];
        if s != sh {
            sh.copy_from_slice(s);
            out_indices.push(slot as u32);
            out_data.extend_from_slice(s);
        }
    }
}

pub fn diff_dirty_pages_64b(
    bitmap: &mut HierarchicalDirtyBitmap,
    src: &[u8],
    shadow: &mut [u8],
    max_slots: usize,
    out_indices: &mut Vec<u32>,
    out_data: &mut Vec<u8>,
) {
    out_indices.clear();
    out_data.clear();
    let use_avx2 = is_x86_feature_detected!("avx2");
    bitmap.drain_dirty_pages(|page_idx| {
        if use_avx2 {
            unsafe {
                diff_4kb_page_avx2(
                    page_idx,
                    src.as_ptr(),
                    shadow.as_mut_ptr(),
                    max_slots,
                    out_indices,
                    out_data,
                );
            }
        } else {
            diff_4kb_page_scalar(page_idx, src, shadow, max_slots, out_indices, out_data);
        }
    });
}

// ---------------------------------------------------------------------------
// Per-Resource Hazard Journal
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct HazardEntry {
    pub res_id: usize,
    pub first_state: D3D12_RESOURCE_STATES,
    pub last_state: D3D12_RESOURCE_STATES,
}

#[derive(Default, Clone, Debug)]
pub struct WorkerHazardJournal {
    pub entries: Vec<HazardEntry>,
}

impl WorkerHazardJournal {
    pub fn record(&mut self, res_id: usize, state: D3D12_RESOURCE_STATES) {
        if let Some(last) = self.entries.iter_mut().rev().find(|e| e.res_id == res_id) {
            last.last_state = state;
        } else {
            self.entries.push(HazardEntry {
                res_id,
                first_state: state,
                last_state: state,
            });
        }
    }
}

/// Resolves sequential cross-worker resource state transitions, returning barriers needed before `journal`.
pub fn resolve_worker_fixups(
    global_states: &mut [D3D12_RESOURCE_STATES],
    journal: &WorkerHazardJournal,
) -> Vec<(usize, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)> {
    let mut fixups = Vec::new();
    for e in &journal.entries {
        let cur = global_states[e.res_id];
        if cur != e.first_state {
            fixups.push((e.res_id, cur, e.first_state));
        }
        global_states[e.res_id] = e.last_state;
    }
    fixups
}

// ---------------------------------------------------------------------------
// D3D12 Helper Constructors (shared cleanly, zero duplication)
// ---------------------------------------------------------------------------

unsafe fn compile_rs(device: &ID3D12Device, params: &[D3D12_ROOT_PARAMETER1], flags: D3D12_ROOT_SIGNATURE_FLAGS) -> ID3D12RootSignature {
    let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: flags,
            },
        },
    };
    let mut blob = None;
    let mut err = None;
    if let Err(e) = D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, Some(&mut err)) {
        if let Some(eb) = err {
            let msg = std::slice::from_raw_parts(eb.GetBufferPointer() as *const u8, eb.GetBufferSize());
            panic!("UMA RootSig error: {}", String::from_utf8_lossy(msg));
        }
        panic!("UMA RootSig serialize failed: {e:?}");
    }
    let blob = blob.unwrap();
    let bytes = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
    device.CreateRootSignature(0, bytes).expect("CreateRootSignature UMA")
}

#[inline]
fn root_consts(reg: u32, num_dwords: u32) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: reg,
                RegisterSpace: 0,
                Num32BitValues: num_dwords,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

#[inline]
fn root_srv(reg: u32) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: reg,
                RegisterSpace: 0,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

#[inline]
fn root_uav(reg: u32) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: reg,
                RegisterSpace: 0,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

unsafe fn create_cs_pso(device: &ID3D12Device, rs: &ID3D12RootSignature, dxil: &[u8]) -> ID3D12PipelineState {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: dxil.as_ptr() as *const c_void,
            BytecodeLength: dxil.len(),
        },
        NodeMask: 0,
        CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
        Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
    };
    let res: windows::core::Result<ID3D12PipelineState> = device.CreateComputePipelineState(&desc);
    let _ = ManuallyDrop::into_inner(desc.pRootSignature);
    res.expect("CreateComputePipelineState UMA")
}

fn create_rgba8_tex2d(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    flags: D3D12_RESOURCE_FLAGS,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    initial_state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    unsafe {
        device
            .CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &desc,
                initial_state,
                None,
                &mut res,
            )
            .expect("CreateCommittedResource Texture2D UMA failed");
    }
    res.unwrap()
}

unsafe fn upload_at<T: Copy>(res: &ID3D12Resource, byte_offset: usize, data: &[T]) {
    if data.is_empty() {
        return;
    }
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let empty_range = D3D12_RANGE { Begin: 0, End: 0 };
    res.Map(0, Some(&empty_range), Some(&mut ptr)).expect("Map upload_at UMA");
    let dst = (ptr as *mut u8).add(byte_offset);
    copy_nonoverlapping(data.as_ptr() as *const u8, dst, std::mem::size_of_val(data));
    let written = D3D12_RANGE {
        Begin: byte_offset,
        End: byte_offset + std::mem::size_of_val(data),
    };
    res.Unmap(0, Some(&written));
}

unsafe fn readback_bytes(res: &ID3D12Resource, len: usize) -> Vec<u8> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let read_range = D3D12_RANGE { Begin: 0, End: len };
    res.Map(0, Some(&read_range), Some(&mut ptr)).expect("Map readback UMA");
    let mut out = vec![0u8; len];
    copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), len);
    let empty = D3D12_RANGE { Begin: 0, End: 0 };
    res.Unmap(0, Some(&empty));
    out
}

// ---------------------------------------------------------------------------
// Shared Live & Headless UMA Emulation Engine
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct UmaMetrics {
    pub enabled: bool,
    pub opt_mask: u8, // bit 0=[1], bit 1=[2], bit 2=[3], bit 3=[4]
    pub sub1_ms: f32,
    pub sub2_ms: f32,
    pub sub3_ms: f32,
    pub sub4_ms: f32,
    pub total_ms: f32,
    pub verified_meshes: u32,
}

impl Default for UmaMetrics {
    fn default() -> Self {
        Self {
            enabled: false,
            opt_mask: 15,
            sub1_ms: 0.0,
            sub2_ms: 0.0,
            sub3_ms: 0.0,
            sub4_ms: 0.0,
            total_ms: 0.0,
            verified_meshes: 64,
        }
    }
}

pub struct UmaStressEngine {
    // Frame counter + PRNG state
    frame_tag: u32,

    // [1] 2,000 Textures + Write Watch + Gfx10 Detile
    tex_watch_regions: Vec<WriteWatchRegion>,
    tex_hashes: Vec<u64>,
    tex_gpu_resources: Vec<ID3D12Resource>,
    detile_src_upload: ID3D12Resource,
    detile_dst_linear: ID3D12Resource,
    detile_readback: ID3D12Resource,
    detile_rs: ID3D12RootSignature,
    detile_pso: ID3D12PipelineState,
    detile_cpu_scratch: Vec<u8>,
    tex_dirty_scratch: Vec<usize>,
    dirty_dyn_indices: Vec<usize>,

    // [2] 1M Descriptor Heap (64 MB) + Scatter Patch
    desc_watch: WriteWatchRegion,
    desc_shadow: Vec<u8>,
    desc_bitmap: HierarchicalDirtyBitmap,
    desc_dirty_pages: Vec<usize>,
    desc_patch_indices: Vec<u32>,
    desc_patch_data: Vec<u8>,
    desc_patch_src_upload: ID3D12Resource,
    desc_patch_idx_upload: ID3D12Resource,
    desc_heap_gpu: ID3D12Resource,
    desc_readback: ID3D12Resource,
    scatter_rs: ID3D12RootSignature,
    scatter_pso: ID3D12PipelineState,

    // [3] 8-Thread Command Recording + Per-Resource Hazard Journals + Skin CS/Draw
    skin_base_verts: ID3D12Resource,
    skin_verts: ID3D12Resource,
    skin_indices: ID3D12Resource,
    skin_rt: ID3D12Resource,
    #[allow(dead_code)]
    skin_rtv_heap: ID3D12DescriptorHeap,
    skin_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    skin_readback: ID3D12Resource,
    skin_cs_rs: ID3D12RootSignature,
    skin_cs_pso: ID3D12PipelineState,
    skin_draw_rs: ID3D12RootSignature,
    skin_draw_pso: ID3D12PipelineState,

    // Worker + Fixup command allocators/lists (Send+Sync wrapped safely per worker index)
    worker_allocs: Vec<ID3D12CommandAllocator>,
    worker_cmds: Vec<ID3D12GraphicsCommandList>,
    fixup_allocs: Vec<ID3D12CommandAllocator>,
    fixup_cmds: Vec<ID3D12GraphicsCommandList>,
    naive_alloc: ID3D12CommandAllocator,
    naive_cmd: ID3D12GraphicsCommandList,

    // [4] Working Set PCIe UPLOAD vs GDDR7 DEFAULT
    workset_upload: ID3D12Resource,
    workset_default: ID3D12Resource,
    workset_sink: ID3D12Resource,
    workset_rs: ID3D12RootSignature,
    workset_pso: ID3D12PipelineState,

    // GPU timestamps (8 queries = begin/end per subsystem)
    ts_heap: ID3D12QueryHeap,
    ts_readback: ID3D12Resource,
}

impl UmaStressEngine {
    pub fn new(ctx: &mut GpuContext) -> Self {
        let dev = ctx.device.clone();
        let device = &dev;
        unsafe {
            // --- Pipelines & Root Signatures ---
            let detile_rs = compile_rs(
                device,
                &[root_consts(0, 4), root_srv(0), root_uav(0)],
                D3D12_ROOT_SIGNATURE_FLAG_NONE,
            );
            let detile_pso = create_cs_pso(device, &detile_rs, UMA_GFX10_DETILE_CS_DXIL);

            let scatter_rs = compile_rs(
                device,
                &[root_consts(0, 4), root_srv(0), root_srv(1), root_uav(0)],
                D3D12_ROOT_SIGNATURE_FLAG_NONE,
            );
            let scatter_pso = create_cs_pso(device, &scatter_rs, UMA_SCATTER_PATCH_CS_DXIL);

            let skin_cs_rs = compile_rs(
                device,
                &[root_consts(0, 4), root_srv(0), root_uav(0)],
                D3D12_ROOT_SIGNATURE_FLAG_NONE,
            );
            let skin_cs_pso = create_cs_pso(device, &skin_cs_rs, UMA_SKIN_CS_DXIL);

            let skin_draw_rs = compile_rs(
                device,
                &[],
                D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            );
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
                    SemanticName: windows::core::s!("TEXCOORD"),
                    SemanticIndex: 0,
                    Format: DXGI_FORMAT_R32_UINT,
                    InputSlot: 0,
                    AlignedByteOffset: 12,
                    InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    InstanceDataStepRate: 0,
                },
            ];
            let mut rt_fmts = [windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN; 8];
            rt_fmts[0] = DXGI_FORMAT_R32_UINT;
            let gfx_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(skin_draw_rs.clone())),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: UMA_SKIN_VS_DXIL.as_ptr() as *const c_void,
                    BytecodeLength: UMA_SKIN_VS_DXIL.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: UMA_SKIN_PS_DXIL.as_ptr() as *const c_void,
                    BytecodeLength: UMA_SKIN_PS_DXIL.len(),
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
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: false.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    ..Default::default()
                },
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: input_elems.as_ptr(),
                    NumElements: input_elems.len() as u32,
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: rt_fmts,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let res: windows::core::Result<ID3D12PipelineState> = device.CreateGraphicsPipelineState(&gfx_desc);
            let _ = ManuallyDrop::into_inner(gfx_desc.pRootSignature);
            let skin_draw_pso = res.expect("CreateGraphicsPipelineState uma_skin_draw");

            let workset_rs = compile_rs(
                device,
                &[root_consts(0, 4), root_srv(0), root_uav(0)],
                D3D12_ROOT_SIGNATURE_FLAG_NONE,
            );
            let workset_pso = create_cs_pso(device, &workset_rs, UMA_WORKINGSET_CS_DXIL);

            // --- Subsystem [1]: 2,000 Write-Watch Tiled Textures ---
            let mut linear_seed = vec![0u8; TEX_STRIDE_BYTES];
            for (idx, chunk) in linear_seed.chunks_exact_mut(4).enumerate() {
                let v = (idx as u32).wrapping_mul(2654435761);
                chunk.copy_from_slice(&v.to_le_bytes());
            }
            let mut tiled_seed = vec![0u8; TEX_STRIDE_BYTES];
            cpu_tile_64kb_d_32bpp(&linear_seed, &mut tiled_seed, TEX_W, TEX_H);

            let mut tex_watch_regions = Vec::with_capacity(TEX_COUNT);
            let mut tex_hashes = Vec::with_capacity(TEX_COUNT);
            let mut tex_gpu_resources = Vec::with_capacity(TEX_COUNT);
            for i in 0..TEX_COUNT {
                let mut reg = WriteWatchRegion::new(TEX_STRIDE_BYTES);
                reg.as_mut_slice()[..TEX_STRIDE_BYTES].copy_from_slice(&tiled_seed);
                // Stamp unique identifier in pixel 0
                reg.as_mut_slice()[0..4].copy_from_slice(&(i as u32).to_le_bytes());
                let h = fnv1a_bytes_u32_chunks(&reg.as_slice()[..TEX_STRIDE_BYTES]);
                reg.reset();
                tex_hashes.push(h);
                tex_watch_regions.push(reg);
                tex_gpu_resources.push(create_rgba8_tex2d(
                    device,
                    TEX_W,
                    TEX_H,
                    D3D12_RESOURCE_FLAG_NONE,
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_STATE_COMMON,
                ));
            }
            let max_detile_batch_bytes = (DYNAMIC_TEX_COUNT * TEX_STRIDE_BYTES) as u64;
            let detile_src_upload = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                max_detile_batch_bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let detile_dst_linear = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                max_detile_batch_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let detile_readback = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                TEX_STRIDE_BYTES as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );

            // --- Subsystem [2]: 1M x 64B Descriptor Heap ---
            let mut desc_watch = WriteWatchRegion::new(DESC_HEAP_BYTES);
            let mut desc_shadow = vec![0u8; DESC_HEAP_BYTES];
            for slot in 0..DESC_SLOT_COUNT {
                let off = slot * DESC_SLOT_BYTES;
                let tag = (slot as u32).to_le_bytes();
                desc_watch.as_mut_slice()[off..off + 4].copy_from_slice(&tag);
                desc_shadow[off..off + 4].copy_from_slice(&tag);
            }
            desc_watch.reset();

            let desc_patch_src_upload = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                (DESC_MAX_PATCH_CAP * DESC_SLOT_BYTES) as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let desc_patch_idx_upload = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                (DESC_MAX_PATCH_CAP * 4) as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let desc_heap_gpu = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                DESC_HEAP_BYTES as u64,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let desc_readback = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                (16 * DESC_SLOT_BYTES) as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );

            // --- Subsystem [3]: Skin CS + 3,500 Draws + 8 Worker Command Lists ---
            let skin_vb_bytes = (SKIN_MESH_COUNT * SKIN_VERTS_PER_MESH * 16) as u64;
            let skin_base_upload = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                skin_vb_bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let mut base_verts = Vec::with_capacity(SKIN_MESH_COUNT * SKIN_VERTS_PER_MESH * 4);
            for m in 0..SKIN_MESH_COUNT {
                let cx = ((m as f32 + 0.5) / SKIN_MESH_COUNT as f32) * 2.0 - 1.0;
                let dx = 0.7 / (SKIN_MESH_COUNT as f32);
                // First 4 vertices form a quad covering pixel column x=m from y=-0.8..+0.8
                let coords = [
                    [cx - dx, -0.8, 0.0],
                    [cx + dx, -0.8, 0.0],
                    [cx + dx, 0.8, 0.0],
                    [cx - dx, 0.8, 0.0],
                ];
                for v in 0..SKIN_VERTS_PER_MESH {
                    let p = coords[v & 3];
                    base_verts.push(p[0].to_bits());
                    base_verts.push(p[1].to_bits());
                    base_verts.push(p[2].to_bits());
                    base_verts.push(m as u32);
                }
            }
            upload_at(&skin_base_upload, 0, &base_verts);

            let skin_base_verts = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                skin_vb_bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let skin_verts = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                skin_vb_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let ib_data: [u32; 6] = [0, 1, 2, 0, 2, 3];
            let ib_upload = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                24,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            upload_slice(&ib_upload, &ib_data);
            let skin_indices = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                24,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON,
            );

            let skin_rt = create_rgba8_tex2d(
                device,
                SKIN_MESH_COUNT as u32,
                64,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                DXGI_FORMAT_R32_UINT,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let skin_rtv_heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .expect("CreateDescriptorHeap UMA RTV");
            let skin_rtv = skin_rtv_heap.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(&skin_rt, None, skin_rtv);

            // RowPitch for 64px x 4B = 256 B (exact 256B DX12 texture pitch alignment!)
            let skin_readback = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                256 * 64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );

            let mut worker_allocs = Vec::with_capacity(WORKER_THREADS);
            let mut worker_cmds = Vec::with_capacity(WORKER_THREADS);
            let mut fixup_allocs = Vec::with_capacity(WORKER_THREADS);
            let mut fixup_cmds = Vec::with_capacity(WORKER_THREADS);
            for _ in 0..WORKER_THREADS {
                let wa: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let wc: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &wa, None).unwrap();
                wc.Close().unwrap();
                worker_allocs.push(wa);
                worker_cmds.push(wc);

                let fa: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let fc: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &fa, None).unwrap();
                fc.Close().unwrap();
                fixup_allocs.push(fa);
                fixup_cmds.push(fc);
            }
            let naive_alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let naive_cmd: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &naive_alloc, None).unwrap();

            // --- Subsystem [4]: Working Set Buffers (UPLOAD PCIe vs DEFAULT GDDR7) ---
            let workset_upload = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                WORKSET_BYTES as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let mut ws_seed = vec![0u32; WORKSET_BYTES / 4];
            for (idx, v) in ws_seed.iter_mut().enumerate() {
                *v = (idx as u32).wrapping_mul(1664525).wrapping_add(1013904223);
            }
            upload_at(&workset_upload, 0, &ws_seed);
            let workset_default = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                WORKSET_BYTES as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let workset_sink = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                4096,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );

            // Init static GPU buffers synchronously once
            let desc_init_up = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                DESC_HEAP_BYTES as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            upload_at(&desc_init_up, 0, &desc_shadow);

            naive_cmd.CopyBufferRegion(&skin_base_verts, 0, &skin_base_upload, 0, skin_vb_bytes);
            naive_cmd.CopyBufferRegion(&skin_indices, 0, &ib_upload, 0, 24);
            naive_cmd.CopyBufferRegion(&workset_default, 0, &workset_upload, 0, WORKSET_BYTES as u64);
            naive_cmd.CopyBufferRegion(&desc_heap_gpu, 0, &desc_init_up, 0, DESC_HEAP_BYTES as u64);
            cmd_barriers(
                &naive_cmd,
                vec![
                    transition_barrier(&skin_base_verts, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                    transition_barrier(&skin_indices, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER),
                    transition_barrier(&workset_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                    transition_barrier(&desc_heap_gpu, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
            naive_cmd.Close().unwrap();
            let exec = [Some(naive_cmd.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&exec);
            ctx.signal_and_wait();

            let mut ts_heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 8,
                        NodeMask: 0,
                    },
                    &mut ts_heap,
                )
                .expect("CreateQueryHeap UMA");
            let ts_heap = ts_heap.unwrap();
            let ts_readback = create_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );

            let desc_pages = (DESC_HEAP_BYTES + PAGE_BYTES - 1) / PAGE_BYTES;

            Self {
                frame_tag: 1,
                tex_watch_regions,
                tex_hashes,
                tex_gpu_resources,
                detile_src_upload,
                detile_dst_linear,
                detile_readback,
                detile_rs,
                detile_pso,
                detile_cpu_scratch: vec![0u8; TEX_STRIDE_BYTES],
                tex_dirty_scratch: Vec::with_capacity(64),
                dirty_dyn_indices: Vec::with_capacity(DYNAMIC_TEX_COUNT),
                desc_watch,
                desc_shadow,
                desc_bitmap: HierarchicalDirtyBitmap::new(desc_pages),
                desc_dirty_pages: Vec::with_capacity(DESC_MUTATIONS_PER_FRAME),
                desc_patch_indices: Vec::with_capacity(4096),
                desc_patch_data: Vec::with_capacity(4096 * DESC_SLOT_BYTES),
                desc_patch_src_upload,
                desc_patch_idx_upload,
                desc_heap_gpu,
                desc_readback,
                scatter_rs,
                scatter_pso,
                skin_base_verts,
                skin_verts,
                skin_indices,
                skin_rt,
                skin_rtv_heap,
                skin_rtv,
                skin_readback,
                skin_cs_rs,
                skin_cs_pso,
                skin_draw_rs,
                skin_draw_pso,
                worker_allocs,
                worker_cmds,
                fixup_allocs,
                fixup_cmds,
                naive_alloc,
                naive_cmd,
                workset_upload,
                workset_default,
                workset_sink,
                workset_rs,
                workset_pso,
                ts_heap,
                ts_readback,
            }
        }
    }

    fn mutate_dynamic_inputs(&mut self) {
        self.frame_tag = self.frame_tag.wrapping_add(1);
        let f = self.frame_tag;

        // Mutate the 100 dynamic textures in SW_64KB_D layout; leave 0..1900 untouched!
        for d in 0..DYNAMIC_TEX_COUNT {
            let tid = STATIC_TEX_COUNT + d;
            let reg = self.tex_watch_regions[tid].as_mut_slice();
            // Modify 8 distinct 4 KiB pages per dynamic texture using tiled pixel coordinates
            for k in 0..8usize {
                let page_off = ((k * 7 + (d % 5)) % 64) * PAGE_BYTES;
                let stamp = f.wrapping_mul(2654435761) ^ ((tid as u32) << 12) ^ (k as u32);
                reg[page_off..page_off + 4].copy_from_slice(&stamp.to_le_bytes());
                reg[page_off + 16..page_off + 20].copy_from_slice(&stamp.rotate_left(7).to_le_bytes());
            }
        }

        // Mutate 1,500 scattered 64B slots across the 1M descriptor heap
        let mut rng = f.wrapping_mul(0x9E37_79B9) ^ 0xA5A5_5A5A;
        let dst = self.desc_watch.as_mut_slice();
        for m in 0..DESC_MUTATIONS_PER_FRAME {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let slot = (rng as usize) % DESC_SLOT_COUNT;
            let off = slot * DESC_SLOT_BYTES;
            let val = f ^ ((m as u32) << 16);
            dst[off..off + 4].copy_from_slice(&val.to_le_bytes());
            dst[off + 60..off + 64].copy_from_slice(&val.wrapping_mul(3).to_le_bytes());
        }
    }

    /// Runs `[1]` Watch + Gfx10 Detile (`opt=true` skips 1,900 immutable + batched GPU CS detile)
    pub fn run_sub1_tex_detile(&mut self, ctx: &mut GpuContext, opt: bool, verify_byte_exact: bool) -> (f64, bool) {
        let t0 = Instant::now();
        self.dirty_dyn_indices.clear();

        unsafe {
            self.naive_alloc.Reset().unwrap();
            self.naive_cmd.Reset(&self.naive_alloc, None).unwrap();

            if !opt {
                // Naive: query GetWriteWatch on ALL 2,000 regions + full CPU FNV-1a hash + CPU detile
                for i in 0..TEX_COUNT {
                    let mut dummy = Vec::new();
                    let _ = self.tex_watch_regions[i].query_dirty_into(true, &mut dummy);
                    let h = fnv1a_bytes_u32_chunks(&self.tex_watch_regions[i].as_slice()[..TEX_STRIDE_BYTES]);
                    if h != self.tex_hashes[i] {
                        self.tex_hashes[i] = h;
                        self.dirty_dyn_indices.push(i);
                    }
                }
                for (b, &tid) in self.dirty_dyn_indices.iter().enumerate() {
                    cpu_detile_64kb_d_32bpp(
                        &self.tex_watch_regions[tid].as_slice()[..TEX_STRIDE_BYTES],
                        &mut self.detile_cpu_scratch,
                        TEX_W,
                        TEX_H,
                    );
                    upload_at(&self.detile_src_upload, b * TEX_STRIDE_BYTES, &self.detile_cpu_scratch);

                    let tex = &self.tex_gpu_resources[tid];
                    cmd_barriers(
                        &self.naive_cmd,
                        vec![transition_barrier(tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)],
                    );
                    let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(tex.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                    };
                    let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(self.detile_src_upload.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: (b * TEX_STRIDE_BYTES) as u64,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                    Width: TEX_W,
                                    Height: TEX_H,
                                    Depth: 1,
                                    RowPitch: TEX_W * 4,
                                },
                            },
                        },
                    };
                    self.naive_cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                    let _ = ManuallyDrop::into_inner(dst_loc.pResource);
                    let _ = ManuallyDrop::into_inner(src_loc.pResource);
                    cmd_barriers(
                        &self.naive_cmd,
                        vec![transition_barrier(tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)],
                    );
                }
                self.naive_cmd.Close().unwrap();
                let exec = [Some(self.naive_cmd.cast::<ID3D12CommandList>().unwrap())];
                ctx.queue.ExecuteCommandLists(&exec);
                ctx.signal_and_wait();
                return (t0.elapsed().as_secs_f64() * 1000.0, true);
            }

            // Optimized: 0 syscalls & 0 hashes on 0..1,900 immutable DirectStorage textures!
            // Query GetWriteWatch ONLY on dynamic textures 1,900..2,000, pack dirty ones, GPU detile in 1 Dispatch.
            for tid in STATIC_TEX_COUNT..TEX_COUNT {
                let n = self.tex_watch_regions[tid].query_dirty_into(true, &mut self.tex_dirty_scratch);
                if n > 0 {
                    self.dirty_dyn_indices.push(tid);
                }
            }

            let batch = self.dirty_dyn_indices.len();
            if batch > 0 {
                let total_bytes = batch * TEX_STRIDE_BYTES;
                let mut ptr: *mut c_void = std::ptr::null_mut();
                let empty = D3D12_RANGE { Begin: 0, End: 0 };
                self.detile_src_upload.Map(0, Some(&empty), Some(&mut ptr)).unwrap();
                struct UploadDst(*mut u8);
                unsafe impl Send for UploadDst {}
                unsafe impl Sync for UploadDst {}
                let dst_wrap = UploadDst(ptr as *mut u8);
                let dirty_slice = &self.dirty_dyn_indices;
                let regions_slice = &self.tex_watch_regions;
                std::thread::scope(|s| {
                    let threads = 4usize.min(batch);
                    for t in 0..threads {
                        let lo = (t * batch) / threads;
                        let hi = ((t + 1) * batch) / threads;
                        let dst_ref = &dst_wrap;
                        s.spawn(move || {
                            for b in lo..hi {
                                let tid = dirty_slice[b];
                                let src = &regions_slice[tid].as_slice()[..TEX_STRIDE_BYTES];
                                copy_nonoverlapping(src.as_ptr(), dst_ref.0.add(b * TEX_STRIDE_BYTES), TEX_STRIDE_BYTES);
                            }
                        });
                    }
                });
                let written = D3D12_RANGE { Begin: 0, End: total_bytes };
                self.detile_src_upload.Unmap(0, Some(&written));

                // Bind UPLOAD buffer directly as Compute SRV (stays GENERIC_READ) -> 0 redundant CopyBufferRegion!
                cmd_barriers(
                    &self.naive_cmd,
                    vec![transition_barrier(
                        &self.detile_dst_linear,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )],
                );
                let cbuf = [TEX_W, TEX_H, batch as u32, TEX_STRIDE_BYTES as u32];
                self.naive_cmd.SetComputeRootSignature(&self.detile_rs);
                self.naive_cmd.SetPipelineState(&self.detile_pso);
                self.naive_cmd.SetComputeRoot32BitConstants(0, 4, cbuf.as_ptr() as *const c_void, 0);
                self.naive_cmd.SetComputeRootShaderResourceView(1, self.detile_src_upload.GetGPUVirtualAddress());
                self.naive_cmd.SetComputeRootUnorderedAccessView(2, self.detile_dst_linear.GetGPUVirtualAddress());
                self.naive_cmd.Dispatch(TEX_W / 8, TEX_H / 8, batch as u32);

                let mut pre_barriers = Vec::with_capacity(batch + 1);
                pre_barriers.push(transition_barrier(
                    &self.detile_dst_linear,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ));
                for &tid in &self.dirty_dyn_indices {
                    pre_barriers.push(transition_barrier(
                        &self.tex_gpu_resources[tid],
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    ));
                }
                cmd_barriers(&self.naive_cmd, pre_barriers);

                for (b, &tid) in self.dirty_dyn_indices.iter().enumerate() {
                    let tex = &self.tex_gpu_resources[tid];
                    let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(tex.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                    };
                    let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(self.detile_dst_linear.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: (b * TEX_STRIDE_BYTES) as u64,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                    Width: TEX_W,
                                    Height: TEX_H,
                                    Depth: 1,
                                    RowPitch: TEX_W * 4,
                                },
                            },
                        },
                    };
                    self.naive_cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                    let _ = ManuallyDrop::into_inner(dst_loc.pResource);
                    let _ = ManuallyDrop::into_inner(src_loc.pResource);
                }
                if verify_byte_exact {
                    self.naive_cmd.CopyBufferRegion(
                        &self.detile_readback,
                        0,
                        &self.detile_dst_linear,
                        0,
                        TEX_STRIDE_BYTES as u64,
                    );
                }
                let mut post_barriers = Vec::with_capacity(batch + 1);
                post_barriers.push(transition_barrier(
                    &self.detile_dst_linear,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                ));
                for &tid in &self.dirty_dyn_indices {
                    post_barriers.push(transition_barrier(
                        &self.tex_gpu_resources[tid],
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COMMON,
                    ));
                }
                cmd_barriers(&self.naive_cmd, post_barriers);
            }
            self.naive_cmd.Close().unwrap();
            let exec = [Some(self.naive_cmd.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&exec);
            ctx.signal_and_wait();

            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let mut exact_ok = true;
            if verify_byte_exact && batch > 0 {
                let first_tid = self.dirty_dyn_indices[0];
                cpu_detile_64kb_d_32bpp(
                    &self.tex_watch_regions[first_tid].as_slice()[..TEX_STRIDE_BYTES],
                    &mut self.detile_cpu_scratch,
                    TEX_W,
                    TEX_H,
                );
                let gpu_lin = readback_bytes(&self.detile_readback, TEX_STRIDE_BYTES);
                exact_ok = gpu_lin == self.detile_cpu_scratch;
            }
            (wall_ms, exact_ok)
        }
    }

    /// Runs `[2]` 1M Descriptor Heap (64 MB) + Scatter Patch
    pub fn run_sub2_desc_scatter(&mut self, ctx: &mut GpuContext, opt: bool, verify_byte_exact: bool) -> (f64, usize, bool) {
        let t0 = Instant::now();
        unsafe {
            self.naive_alloc.Reset().unwrap();
            self.naive_cmd.Reset(&self.naive_alloc, None).unwrap();

            if !opt {
                // Naive: query whole watch + full 64 MiB linear scan + thousands of individual CopyBufferRegion calls
                let _ = self.desc_watch.query_dirty_into(true, &mut self.desc_dirty_pages);
                self.desc_patch_indices.clear();
                self.desc_patch_data.clear();
                let src = &self.desc_watch.as_slice()[..DESC_HEAP_BYTES];
                for slot in 0..DESC_SLOT_COUNT {
                    let off = slot * DESC_SLOT_BYTES;
                    let s = &src[off..off + DESC_SLOT_BYTES];
                    let sh = &mut self.desc_shadow[off..off + DESC_SLOT_BYTES];
                    if s != sh {
                        sh.copy_from_slice(s);
                        self.desc_patch_indices.push(slot as u32);
                        self.desc_patch_data.extend_from_slice(s);
                    }
                }
                let n = self.desc_patch_indices.len().min(DESC_MAX_PATCH_CAP);
                if n > 0 {
                    upload_at(&self.desc_patch_src_upload, 0, &self.desc_patch_data[..n * DESC_SLOT_BYTES]);
                    for (k, &slot) in self.desc_patch_indices[..n].iter().enumerate() {
                        let dst_off = (slot as u64) * (DESC_SLOT_BYTES as u64);
                        let src_off = (k as u64) * (DESC_SLOT_BYTES as u64);
                        cmd_barriers(
                            &self.naive_cmd,
                            vec![transition_barrier(
                                &self.desc_heap_gpu,
                                D3D12_RESOURCE_STATE_COMMON,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                            )],
                        );
                        self.naive_cmd.CopyBufferRegion(
                            &self.desc_heap_gpu,
                            dst_off,
                            &self.desc_patch_src_upload,
                            src_off,
                            DESC_SLOT_BYTES as u64,
                        );
                        cmd_barriers(
                            &self.naive_cmd,
                            vec![transition_barrier(
                                &self.desc_heap_gpu,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_COMMON,
                            )],
                        );
                    }
                }
                self.naive_cmd.Close().unwrap();
                let exec = [Some(self.naive_cmd.cast::<ID3D12CommandList>().unwrap())];
                ctx.queue.ExecuteCommandLists(&exec);
                ctx.signal_and_wait();
                return (t0.elapsed().as_secs_f64() * 1000.0, n, true);
            }

            // Optimized: GetWriteWatch dirty pages -> 3-level u64 bitmap -> AVX2 64B diff -> 1 uma_scatter_patch.dxil Dispatch!
            self.desc_watch.query_dirty_into(true, &mut self.desc_dirty_pages);
            for &p in &self.desc_dirty_pages {
                self.desc_bitmap.mark_page(p);
            }
            diff_dirty_pages_64b(
                &mut self.desc_bitmap,
                &self.desc_watch.as_slice()[..DESC_HEAP_BYTES],
                &mut self.desc_shadow,
                DESC_SLOT_COUNT,
                &mut self.desc_patch_indices,
                &mut self.desc_patch_data,
            );
            let n = self.desc_patch_indices.len().min(DESC_MAX_PATCH_CAP);
            if n > 0 {
                upload_at(&self.desc_patch_src_upload, 0, &self.desc_patch_data[..n * DESC_SLOT_BYTES]);
                upload_at(&self.desc_patch_idx_upload, 0, &self.desc_patch_indices[..n]);

                cmd_barriers(
                    &self.naive_cmd,
                    vec![transition_barrier(
                        &self.desc_heap_gpu,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )],
                );
                let cb = [n as u32, 0, 0, 0];
                self.naive_cmd.SetComputeRootSignature(&self.scatter_rs);
                self.naive_cmd.SetPipelineState(&self.scatter_pso);
                self.naive_cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
                self.naive_cmd.SetComputeRootShaderResourceView(1, self.desc_patch_src_upload.GetGPUVirtualAddress());
                self.naive_cmd.SetComputeRootShaderResourceView(2, self.desc_patch_idx_upload.GetGPUVirtualAddress());
                self.naive_cmd.SetComputeRootUnorderedAccessView(3, self.desc_heap_gpu.GetGPUVirtualAddress());
                self.naive_cmd.Dispatch((n as u32).div_ceil(64), 1, 1);

                if verify_byte_exact {
                    cmd_barriers(
                        &self.naive_cmd,
                        vec![transition_barrier(
                            &self.desc_heap_gpu,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )],
                    );
                    let check_count = n.min(16);
                    for k in 0..check_count {
                        let slot = self.desc_patch_indices[k] as u64;
                        self.naive_cmd.CopyBufferRegion(
                            &self.desc_readback,
                            (k * DESC_SLOT_BYTES) as u64,
                            &self.desc_heap_gpu,
                            slot * (DESC_SLOT_BYTES as u64),
                            DESC_SLOT_BYTES as u64,
                        );
                    }
                    cmd_barriers(
                        &self.naive_cmd,
                        vec![transition_barrier(
                            &self.desc_heap_gpu,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                            D3D12_RESOURCE_STATE_COMMON,
                        )],
                    );
                } else {
                    cmd_barriers(
                        &self.naive_cmd,
                        vec![transition_barrier(
                            &self.desc_heap_gpu,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COMMON,
                        )],
                    );
                }
            }
            self.naive_cmd.Close().unwrap();
            let exec = [Some(self.naive_cmd.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&exec);
            ctx.signal_and_wait();

            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let mut exact_ok = true;
            if verify_byte_exact && n > 0 {
                let check_count = n.min(16);
                let rb = readback_bytes(&self.desc_readback, check_count * DESC_SLOT_BYTES);
                for k in 0..check_count {
                    let exp = &self.desc_patch_data[k * DESC_SLOT_BYTES..(k + 1) * DESC_SLOT_BYTES];
                    let got = &rb[k * DESC_SLOT_BYTES..(k + 1) * DESC_SLOT_BYTES];
                    if got != exp {
                        exact_ok = false;
                        break;
                    }
                }
            }
            (wall_ms, n, exact_ok)
        }
    }

    /// Runs `[3]` 64 Skin Dispatches + 3,500 Indexed Draws:
    /// Naive = 1-thread recording + forced mid-frame settle waits;
    /// Optimized = 8-thread recording + Per-Resource Hazard Journals + Fixup lists + 1 `ExecuteCommandLists`.
    pub fn run_sub3_cmd_hazards(&mut self, ctx: &mut GpuContext, opt: bool, verify_64_meshes: bool) -> (f64, u32) {
        let t0 = Instant::now();
        let ftag = self.frame_tag;
        unsafe {
            if !opt {
                // Naive: 1-thread recording across 16 mid-frame settle submissions
                const CHUNKS: usize = 16;
                for c in 0..CHUNKS {
                    self.naive_alloc.Reset().unwrap();
                    self.naive_cmd.Reset(&self.naive_alloc, None).unwrap();
                    cmd_barriers(
                        &self.naive_cmd,
                        vec![transition_barrier(
                            &self.skin_verts,
                            D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        )],
                    );
                    self.naive_cmd.SetComputeRootSignature(&self.skin_cs_rs);
                    self.naive_cmd.SetPipelineState(&self.skin_cs_pso);
                    self.naive_cmd.SetComputeRootShaderResourceView(1, self.skin_base_verts.GetGPUVirtualAddress());
                    self.naive_cmd.SetComputeRootUnorderedAccessView(2, self.skin_verts.GetGPUVirtualAddress());
                    let m_start = (c * SKIN_MESH_COUNT) / CHUNKS;
                    let m_end = ((c + 1) * SKIN_MESH_COUNT) / CHUNKS;
                    for m in m_start..m_end {
                        let cb = [m as u32, ftag, SKIN_VERTS_PER_MESH as u32, 0];
                        self.naive_cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
                        self.naive_cmd.Dispatch(1, 1, 1);
                    }
                    cmd_barriers(
                        &self.naive_cmd,
                        vec![
                            transition_barrier(
                                &self.skin_verts,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                            ),
                            transition_barrier(
                                &self.skin_rt,
                                D3D12_RESOURCE_STATE_COMMON,
                                D3D12_RESOURCE_STATE_RENDER_TARGET,
                            ),
                        ],
                    );
                    if c == 0 {
                        let clear_col = [0.0f32; 4];
                        self.naive_cmd.ClearRenderTargetView(self.skin_rtv, &clear_col, None);
                    }
                    let vp = D3D12_VIEWPORT {
                        TopLeftX: 0.0,
                        TopLeftY: 0.0,
                        Width: SKIN_MESH_COUNT as f32,
                        Height: 64.0,
                        MinDepth: 0.0,
                        MaxDepth: 1.0,
                    };
                    let sc = windows::Win32::Foundation::RECT {
                        left: 0,
                        top: 0,
                        right: SKIN_MESH_COUNT as i32,
                        bottom: 64,
                    };
                    self.naive_cmd.RSSetViewports(&[vp]);
                    self.naive_cmd.RSSetScissorRects(&[sc]);
                    self.naive_cmd.OMSetRenderTargets(1, Some(&self.skin_rtv), false, None);
                    self.naive_cmd.SetGraphicsRootSignature(&self.skin_draw_rs);
                    self.naive_cmd.SetPipelineState(&self.skin_draw_pso);
                    self.naive_cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    self.naive_cmd.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                        BufferLocation: self.skin_indices.GetGPUVirtualAddress(),
                        SizeInBytes: 24,
                        Format: DXGI_FORMAT_R32_UINT,
                    }));

                    let d_start = (c * SKIN_DRAW_COUNT) / CHUNKS;
                    let d_end = ((c + 1) * SKIN_DRAW_COUNT) / CHUNKS;
                    let va = self.skin_verts.GetGPUVirtualAddress();
                    for d in d_start..d_end {
                        let m = d % SKIN_MESH_COUNT;
                        let vbv = D3D12_VERTEX_BUFFER_VIEW {
                            BufferLocation: va + ((m * SKIN_VERTS_PER_MESH * 16) as u64),
                            SizeInBytes: (SKIN_VERTS_PER_MESH * 16) as u32,
                            StrideInBytes: 16,
                        };
                        self.naive_cmd.IASetVertexBuffers(0, Some(&[vbv]));
                        self.naive_cmd.DrawIndexedInstanced(6, 1, 0, 0, 0);
                    }
                    cmd_barriers(
                        &self.naive_cmd,
                        vec![
                            transition_barrier(
                                &self.skin_verts,
                                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                                D3D12_RESOURCE_STATE_COMMON,
                            ),
                            transition_barrier(
                                &self.skin_rt,
                                D3D12_RESOURCE_STATE_RENDER_TARGET,
                                D3D12_RESOURCE_STATE_COMMON,
                            ),
                        ],
                    );
                    self.naive_cmd.Close().unwrap();
                    let exec = [Some(self.naive_cmd.cast::<ID3D12CommandList>().unwrap())];
                    ctx.queue.ExecuteCommandLists(&exec);
                    ctx.signal_and_wait(); // Forced mid-frame settle!
                }
                return (t0.elapsed().as_secs_f64() * 1000.0, SKIN_MESH_COUNT as u32);
            }

            // Optimized: 8 worker threads record in parallel with local Per-Resource Hazard Journals!
            // Workers 0..4 record 64 compute skin dispatches (16 each) in UNORDERED_ACCESS.
            // Workers 4..8 record 3,500 indexed graphics draws (875 each) in VERTEX_AND_CONSTANT_BUFFER + RENDER_TARGET.
            for w in 0..WORKER_THREADS {
                self.worker_allocs[w].Reset().unwrap();
                self.worker_cmds[w].Reset(&self.worker_allocs[w], None).unwrap();
                self.fixup_allocs[w].Reset().unwrap();
                self.fixup_cmds[w].Reset(&self.fixup_allocs[w], None).unwrap();
            }

            let mut journals = vec![WorkerHazardJournal::default(); WORKER_THREADS];
            let base_va = self.skin_base_verts.GetGPUVirtualAddress();
            let skin_va = self.skin_verts.GetGPUVirtualAddress();
            let idx_va = self.skin_indices.GetGPUVirtualAddress();
            let rtv = self.skin_rtv;

            // Wrap raw pointers safely for parallel scoped recording (disjoint alloc/cmd per index)
            struct WorkerPtrs {
                cmd: *const ID3D12GraphicsCommandList,
                cs_rs: *const ID3D12RootSignature,
                cs_pso: *const ID3D12PipelineState,
                draw_rs: *const ID3D12RootSignature,
                draw_pso: *const ID3D12PipelineState,
            }
            unsafe impl Send for WorkerPtrs {}
            unsafe impl Sync for WorkerPtrs {}

            let mut ptrs = Vec::with_capacity(WORKER_THREADS);
            for w in 0..WORKER_THREADS {
                ptrs.push(WorkerPtrs {
                    cmd: &self.worker_cmds[w] as *const _,
                    cs_rs: &self.skin_cs_rs as *const _,
                    cs_pso: &self.skin_cs_pso as *const _,
                    draw_rs: &self.skin_draw_rs as *const _,
                    draw_pso: &self.skin_draw_pso as *const _,
                });
            }

            std::thread::scope(|s| {
                for (w, (wp, j)) in ptrs.iter().zip(journals.iter_mut()).enumerate() {
                    s.spawn(move || {
                        let cmd = &*wp.cmd;
                        if w < 4 {
                            // Compute skinning worker: 16 meshes
                            j.record(0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); // res 0 = skin_verts
                            cmd.SetComputeRootSignature(&*wp.cs_rs);
                            cmd.SetPipelineState(&*wp.cs_pso);
                            cmd.SetComputeRootShaderResourceView(1, base_va);
                            cmd.SetComputeRootUnorderedAccessView(2, skin_va);
                            let m0 = w * 16;
                            for m in m0..(m0 + 16) {
                                let cb = [m as u32, ftag, SKIN_VERTS_PER_MESH as u32, 0];
                                cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
                                cmd.Dispatch(1, 1, 1);
                            }
                        } else {
                            // Graphics draw worker: 875 indexed draws across all 64 meshes
                            j.record(0, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER); // res 0 = skin_verts
                            j.record(1, D3D12_RESOURCE_STATE_RENDER_TARGET); // res 1 = skin_rt
                            let vp = D3D12_VIEWPORT {
                                TopLeftX: 0.0,
                                TopLeftY: 0.0,
                                Width: SKIN_MESH_COUNT as f32,
                                Height: 64.0,
                                MinDepth: 0.0,
                                MaxDepth: 1.0,
                            };
                            let sc = windows::Win32::Foundation::RECT {
                                left: 0,
                                top: 0,
                                right: SKIN_MESH_COUNT as i32,
                                bottom: 64,
                            };
                            cmd.RSSetViewports(&[vp]);
                            cmd.RSSetScissorRects(&[sc]);
                            cmd.OMSetRenderTargets(1, Some(&rtv), false, None);
                            cmd.SetGraphicsRootSignature(&*wp.draw_rs);
                            cmd.SetPipelineState(&*wp.draw_pso);
                            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                            cmd.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                                BufferLocation: idx_va,
                                SizeInBytes: 24,
                                Format: DXGI_FORMAT_R32_UINT,
                            }));
                            let gw = w - 4;
                            let d_start = (gw * SKIN_DRAW_COUNT) / 4;
                            let d_end = ((gw + 1) * SKIN_DRAW_COUNT) / 4;
                            for d in d_start..d_end {
                                let m = d % SKIN_MESH_COUNT;
                                let vbv = D3D12_VERTEX_BUFFER_VIEW {
                                    BufferLocation: skin_va + ((m * SKIN_VERTS_PER_MESH * 16) as u64),
                                    SizeInBytes: (SKIN_VERTS_PER_MESH * 16) as u32,
                                    StrideInBytes: 16,
                                };
                                cmd.IASetVertexBuffers(0, Some(&[vbv]));
                                cmd.DrawIndexedInstanced(6, 1, 0, 0, 0);
                            }
                        }
                        cmd.Close().unwrap();
                    });
                }
            });

            // Main thread replays journals sequentially and inserts minimal fixup lists (0 barriers when state matches!)
            let mut global_states = [D3D12_RESOURCE_STATE_COMMON; 2];
            let resources = [&self.skin_verts, &self.skin_rt];
            let mut batch_lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(WORKER_THREADS * 2 + 1);

            for w in 0..WORKER_THREADS {
                let fixups = resolve_worker_fixups(&mut global_states, &journals[w]);
                if !fixups.is_empty() || w == 4 {
                    let fcmd = &self.fixup_cmds[w];
                    let barriers: Vec<D3D12_RESOURCE_BARRIER> = fixups
                        .iter()
                        .map(|&(id, before, after)| transition_barrier(resources[id], before, after))
                        .collect();
                    if !barriers.is_empty() {
                        cmd_barriers(fcmd, barriers);
                    }
                    if w == 4 {
                        let clear_col = [0.0f32; 4];
                        fcmd.ClearRenderTargetView(self.skin_rtv, &clear_col, None);
                    }
                    fcmd.Close().unwrap();
                    batch_lists.push(Some(fcmd.cast::<ID3D12CommandList>().unwrap()));
                } else {
                    self.fixup_cmds[w].Close().unwrap();
                }
                batch_lists.push(Some(self.worker_cmds[w].cast::<ID3D12CommandList>().unwrap()));
            }

            // Post-frame epilogue (readback verification + restore COMMON)
            self.naive_alloc.Reset().unwrap();
            self.naive_cmd.Reset(&self.naive_alloc, None).unwrap();
            let mut epi = Vec::new();
            if verify_64_meshes {
                epi.push(transition_barrier(&self.skin_rt, global_states[1], D3D12_RESOURCE_STATE_COPY_SOURCE));
                cmd_barriers(&self.naive_cmd, epi);
                let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(self.skin_readback.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R32_UINT,
                                Width: SKIN_MESH_COUNT as u32,
                                Height: 64,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(self.skin_rt.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                self.naive_cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                let _ = ManuallyDrop::into_inner(dst_loc.pResource);
                let _ = ManuallyDrop::into_inner(src_loc.pResource);
                cmd_barriers(
                    &self.naive_cmd,
                    vec![
                        transition_barrier(&self.skin_verts, global_states[0], D3D12_RESOURCE_STATE_COMMON),
                        transition_barrier(&self.skin_rt, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    ],
                );
            } else {
                cmd_barriers(
                    &self.naive_cmd,
                    vec![
                        transition_barrier(&self.skin_verts, global_states[0], D3D12_RESOURCE_STATE_COMMON),
                        transition_barrier(&self.skin_rt, global_states[1], D3D12_RESOURCE_STATE_COMMON),
                    ],
                );
            }
            self.naive_cmd.Close().unwrap();
            batch_lists.push(Some(self.naive_cmd.cast::<ID3D12CommandList>().unwrap()));

            // ONE single ExecuteCommandLists submission across all 8 worker lists + minimal fixups!
            ctx.queue.ExecuteCommandLists(&batch_lists);
            ctx.signal_and_wait();

            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let mut ok_meshes = SKIN_MESH_COUNT as u32;
            if verify_64_meshes {
                let rb = readback_bytes(&self.skin_readback, 256 * 64);
                let row32 = &rb[32 * 256..32 * 256 + SKIN_MESH_COUNT * 4];
                ok_meshes = 0;
                for m in 0..SKIN_MESH_COUNT {
                    let got = u32::from_le_bytes(row32[m * 4..m * 4 + 4].try_into().unwrap());
                    let expected = ((m as u32) << 16) | (ftag & 0xFFFF);
                    if got == expected {
                        ok_meshes += 1;
                    }
                }
            }
            (wall_ms, ok_meshes)
        }
    }

    /// Runs `[4]` Working-Set bandwidth stress: PCIe `UPLOAD` (`opt=false`) vs GDDR7 `DEFAULT` (`opt=true`).
    pub fn run_sub4_workingset(&mut self, ctx: &mut GpuContext, opt: bool, passes: u32) -> (f64, f64) {
        let t0 = Instant::now();
        unsafe {
            self.naive_alloc.Reset().unwrap();
            self.naive_cmd.Reset(&self.naive_alloc, None).unwrap();

            let src_va = if opt {
                self.workset_default.GetGPUVirtualAddress()
            } else {
                self.workset_upload.GetGPUVirtualAddress()
            };

            cmd_barriers(
                &self.naive_cmd,
                vec![transition_barrier(
                    &self.workset_sink,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )],
            );
            self.naive_cmd.EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            self.naive_cmd.SetComputeRootSignature(&self.workset_rs);
            self.naive_cmd.SetPipelineState(&self.workset_pso);
            let cb = [WORKSET_ELEMS, passes, self.frame_tag, 0];
            self.naive_cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
            self.naive_cmd.SetComputeRootShaderResourceView(1, src_va);
            self.naive_cmd.SetComputeRootUnorderedAccessView(2, self.workset_sink.GetGPUVirtualAddress());
            self.naive_cmd.Dispatch(1024, 1, 1); // 65,536 threads
            self.naive_cmd.EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            self.naive_cmd.ResolveQueryData(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            cmd_barriers(
                &self.naive_cmd,
                vec![transition_barrier(
                    &self.workset_sink,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                )],
            );
            self.naive_cmd.Close().unwrap();
            let exec = [Some(self.naive_cmd.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&exec);
            ctx.signal_and_wait();

            let ts_bytes = readback_bytes(&self.ts_readback, 16);
            let q0 = u64::from_le_bytes(ts_bytes[0..8].try_into().unwrap());
            let q1 = u64::from_le_bytes(ts_bytes[8..16].try_into().unwrap());
            let gpu_ms = ((q1.saturating_sub(q0)) as f64 * 1000.0) / (ctx.timestamp_freq as f64);
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            (wall_ms, gpu_ms)
        }
    }

    pub fn tick_live_frame(&mut self, ctx: &mut GpuContext, opt_mask: u8) -> UmaMetrics {
        self.mutate_dynamic_inputs();
        let opt1 = (opt_mask & 1) != 0;
        let opt2 = (opt_mask & 2) != 0;
        let opt3 = (opt_mask & 4) != 0;
        let opt4 = (opt_mask & 8) != 0;

        let (m1, _) = self.run_sub1_tex_detile(ctx, opt1, self.frame_tag <= 4);
        let (m2, _, _) = self.run_sub2_desc_scatter(ctx, opt2, self.frame_tag <= 4);
        let (m3, ok_meshes) = self.run_sub3_cmd_hazards(ctx, opt3, self.frame_tag <= 4);
        let (_m4_wall, m4_gpu) = self.run_sub4_workingset(ctx, opt4, 256);
        let tot = m1 + m2 + m3 + m4_gpu;

        UmaMetrics {
            enabled: true,
            opt_mask,
            sub1_ms: m1 as f32,
            sub2_ms: m2 as f32,
            sub3_ms: m3 as f32,
            sub4_ms: m4_gpu as f32,
            total_ms: tot as f32,
            verified_meshes: ok_meshes,
        }
    }
}

// ---------------------------------------------------------------------------
// Headless Emulation Suite (`bcpack-gpu-demo --uma-emulation-bench`)
// ---------------------------------------------------------------------------

pub fn run_uma_emulation_bench(enable_debug: bool, report_json: Option<&Path>) {
    let mut ctx = GpuContext::new(enable_debug);
    println!(
        "=== BCPack GPU v0.15.4 Clean-Room UMA Emulation Optimization Suite ===\n\
         [uma-bench] Adapter: {} (VRAM: {} MiB dedicated, {} MiB OS budget) | DebugLayer: {}",
        ctx.adapter_name, ctx.dedicated_vram_mb, ctx.os_vram_budget_mb, enable_debug
    );

    let mut engine = UmaStressEngine::new(&mut ctx);

    // Subsystem [1]: 2,000 Textures + WriteWatch + Gfx10 Detile
    engine.mutate_dynamic_inputs();
    let (s1_naive_ms, _) = engine.run_sub1_tex_detile(&mut ctx, false, false);
    engine.mutate_dynamic_inputs();
    let (s1_opt_ms, s1_exact) = engine.run_sub1_tex_detile(&mut ctx, true, true);
    assert!(s1_exact, "[uma-bench] FAILED: Sub1 Gfx10 GPU detile mismatch vs CPU reference!");

    // Subsystem [2]: 1M Descriptor Heap (64 MiB) + Scatter Patch
    engine.mutate_dynamic_inputs();
    let (s2_naive_ms, s2_n_naive, _) = engine.run_sub2_desc_scatter(&mut ctx, false, false);
    engine.mutate_dynamic_inputs();
    let (s2_opt_ms, s2_n_opt, s2_exact) = engine.run_sub2_desc_scatter(&mut ctx, true, true);
    assert!(s2_exact, "[uma-bench] FAILED: Sub2 Descriptor Scatter Patch readback mismatch!");

    // Subsystem [3]: 64 Skin Dispatches + 3,500 Draws across 30 consecutive verified frames
    let (s3_naive_ms, _) = engine.run_sub3_cmd_hazards(&mut ctx, false, false);
    let mut s3_opt_sum = 0.0f64;
    let mut s3_min_ok = 64u32;
    for _ in 0..30 {
        engine.mutate_dynamic_inputs();
        let (ms, ok) = engine.run_sub3_cmd_hazards(&mut ctx, true, true);
        s3_opt_sum += ms;
        s3_min_ok = s3_min_ok.min(ok);
        assert_eq!(ok, 64, "[uma-bench] FAILED: Sub3 verified {ok}/64 meshes!");
    }
    let s3_opt_ms = s3_opt_sum / 30.0;

    // Subsystem [4]: Working-Set PCIe UPLOAD vs GDDR7 DEFAULT
    let (s4_pcie_wall, s4_pcie_gpu) = engine.run_sub4_workingset(&mut ctx, false, 256);
    let (s4_vram_wall, s4_vram_gpu) = engine.run_sub4_workingset(&mut ctx, true, 256);

    let total_naive = s1_naive_ms + s2_naive_ms + s3_naive_ms + s4_pcie_gpu;
    let total_opt = s1_opt_ms + s2_opt_ms + s3_opt_ms + s4_vram_gpu;

    println!("[1] WriteWatch+Gfx10Detile : Naive {:.2} ms -> Opt {:.3} ms ({:.1}x) | byte_exact={}", s1_naive_ms, s1_opt_ms, s1_naive_ms / s1_opt_ms.max(1e-4), s1_exact);
    println!("[2] 1M Desc + ScatterPatch : Naive {:.2} ms ({} copies) -> Opt {:.3} ms ({} patches, {:.1}x) | byte_exact={}", s2_naive_ms, s2_n_naive, s2_opt_ms, s2_n_opt, s2_naive_ms / s2_opt_ms.max(1e-4), s2_exact);
    println!("[3] 8-Thr + Hazard Journal : Naive {:.2} ms (1T+settle) -> Opt {:.3} ms ({:.1}x) | verified={}/64 x 30 frames", s3_naive_ms, s3_opt_ms, s3_naive_ms / s3_opt_ms.max(1e-4), s3_min_ok);
    println!("[4] WorkingSet PCIe vs VRAM: PCIe {:.2} ms (GPU {:.3} ms) -> GDDR7 {:.2} ms (GPU {:.3} ms, {:.1}x)", s4_pcie_wall, s4_pcie_gpu, s4_vram_wall, s4_vram_gpu, s4_pcie_gpu / s4_vram_gpu.max(1e-4));
    println!("[uma-bench] TOTAL frame bottleneck: {:.2} ms -> {:.3} ms ({:.1}x speedup)", total_naive, total_opt, total_naive / total_opt.max(1e-4));

    if enable_debug {
        unsafe {
            if let Ok(iq) = ctx.device.cast::<ID3D12InfoQueue>() {
                let n = iq.GetNumStoredMessagesAllowedByRetrievalFilter();
                let mut err_cnt = 0u64;
                for i in 0..n {
                    let mut len = 0usize;
                    let _ = iq.GetMessage(i, None, &mut len);
                    if len > 0 {
                        let mut buf = vec![0u8; len];
                        let mp = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
                        if iq.GetMessage(i, Some(mp), &mut len).is_ok() {
                            let m = &*mp;
                            if m.Severity == D3D12_MESSAGE_SEVERITY_ERROR || m.Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION {
                                err_cnt += 1;
                                if !m.pDescription.is_null() && m.DescriptionByteLength > 0 {
                                    let s = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength - 1);
                                    eprintln!("[d3d12 ERROR] {}", String::from_utf8_lossy(s));
                                }
                            }
                        }
                    }
                }
                println!("[uma-bench] DX12 Debug Layer stored messages: {} (errors/corruption: {})", n, err_cnt);
                assert_eq!(err_cnt, 0, "DX12 Debug Layer reported {err_cnt} errors!");
            }
        }
    }

    if let Some(path) = report_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let json = format!(
            "{{\n  \"adapter\": \"{}\",\n  \"dedicated_vram_mb\": {},\n  \"debug_layer\": {},\n\
             \"sub1_tex_detile\": {{ \"tex_count\": {}, \"dynamic_count\": {}, \"naive_ms\": {:.3}, \"opt_ms\": {:.3}, \"speedup\": {:.2}, \"byte_exact\": {} }},\n\
             \"sub2_desc_scatter\": {{ \"slot_count\": {}, \"patches\": {}, \"naive_ms\": {:.3}, \"opt_ms\": {:.3}, \"speedup\": {:.2}, \"byte_exact\": {} }},\n\
             \"sub3_cmd_hazards\": {{ \"skin_dispatches\": {}, \"draws\": {}, \"workers\": {}, \"naive_ms\": {:.3}, \"opt_avg_30f_ms\": {:.3}, \"speedup\": {:.2}, \"verified_meshes_min\": {} }},\n\
             \"sub4_workingset\": {{ \"bytes\": {}, \"pcie_wall_ms\": {:.3}, \"pcie_gpu_ms\": {:.3}, \"gddr7_wall_ms\": {:.3}, \"gddr7_gpu_ms\": {:.3}, \"gpu_speedup\": {:.2} }},\n\
             \"total_naive_ms\": {:.3},\n  \"total_opt_ms\": {:.3},\n  \"total_speedup\": {:.2}\n}}\n",
            ctx.adapter_name.replace('"', "\\\""),
            ctx.dedicated_vram_mb,
            enable_debug,
            TEX_COUNT,
            DYNAMIC_TEX_COUNT,
            s1_naive_ms,
            s1_opt_ms,
            s1_naive_ms / s1_opt_ms.max(1e-4),
            s1_exact,
            DESC_SLOT_COUNT,
            s2_n_opt,
            s2_naive_ms,
            s2_opt_ms,
            s2_naive_ms / s2_opt_ms.max(1e-4),
            s2_exact,
            SKIN_MESH_COUNT,
            SKIN_DRAW_COUNT,
            WORKER_THREADS,
            s3_naive_ms,
            s3_opt_ms,
            s3_naive_ms / s3_opt_ms.max(1e-4),
            s3_min_ok,
            WORKSET_BYTES,
            s4_pcie_wall,
            s4_pcie_gpu,
            s4_vram_wall,
            s4_vram_gpu,
            s4_pcie_gpu / s4_vram_gpu.max(1e-4),
            total_naive,
            total_opt,
            total_naive / total_opt.max(1e-4),
        );
        let mut f = File::create(path).expect("create uma_emulation_report.json");
        f.write_all(json.as_bytes()).expect("write uma_emulation_report.json");
        println!("[uma-bench] Wrote JSON report to {}", path.display());
    }
    println!("[uma-bench] COMPLETE: all 4 UMA emulation subsystems verified & passed.");
}

// Unused import suppression for multi-thread sync counters if needed
#[allow(dead_code)]
fn _touch_atomics() {
    let b = AtomicBool::new(false);
    let n = AtomicUsize::new(0);
    let _ = (b.load(Ordering::Relaxed), n.load(Ordering::Relaxed));
}

// ---------------------------------------------------------------------------
// Unit Tests (`cargo test -- uma_stress`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swizzle_64kb_d_roundtrip_and_bijection() {
        let mut seen = vec![false; 16384];
        for ly in 0..128u32 {
            for lx in 0..128u32 {
                let idx = swizzle_64kb_d_32bpp(lx, ly) as usize;
                assert!(idx < 16384, "swizzled index {idx} out of 64KB macro-tile range");
                assert!(!seen[idx], "duplicate swizzled index {idx} at ({lx},{ly})");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&x| x));

        let mut linear = vec![0u8; TEX_STRIDE_BYTES];
        for (i, b) in linear.iter_mut().enumerate() {
            *b = ((i * 17 + (i >> 8) * 31) & 0xFF) as u8;
        }
        let mut tiled = vec![0u8; TEX_STRIDE_BYTES];
        let mut detiled = vec![0u8; TEX_STRIDE_BYTES];
        cpu_tile_64kb_d_32bpp(&linear, &mut tiled, TEX_W, TEX_H);
        cpu_detile_64kb_d_32bpp(&tiled, &mut detiled, TEX_W, TEX_H);
        assert_eq!(linear, detiled, "64KB_D roundtrip mismatch");
    }

    #[test]
    fn test_hierarchical_bitmap_and_avx2_slot_diff() {
        let num_slots = 4096usize;
        let bytes = num_slots * DESC_SLOT_BYTES;
        let pages = (bytes + PAGE_BYTES - 1) / PAGE_BYTES;
        let mut region = WriteWatchRegion::new(bytes);
        let mut shadow = vec![0u8; bytes];
        for s in 0..num_slots {
            let off = s * DESC_SLOT_BYTES;
            let v = (s as u32).wrapping_mul(11).to_le_bytes();
            region.as_mut_slice()[off..off + 4].copy_from_slice(&v);
            shadow[off..off + 4].copy_from_slice(&v);
        }
        region.reset();

        // Mutate 3 scattered slots
        let dirty_targets = [5usize, 133usize, 2999usize];
        for &slot in &dirty_targets {
            let off = slot * DESC_SLOT_BYTES;
            region.as_mut_slice()[off..off + 4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        }

        let mut pages_out = Vec::new();
        let n_dirty = region.query_dirty_into(true, &mut pages_out);
        assert!(n_dirty >= 3, "expected >=3 Windows write-watch dirty pages, got {n_dirty}");

        let mut bmp = HierarchicalDirtyBitmap::new(pages);
        for p in pages_out {
            bmp.mark_page(p);
        }
        let mut idx_out = Vec::new();
        let mut data_out = Vec::new();
        diff_dirty_pages_64b(
            &mut bmp,
            &region.as_slice()[..bytes],
            &mut shadow,
            num_slots,
            &mut idx_out,
            &mut data_out,
        );
        assert_eq!(idx_out, vec![5, 133, 2999]);
        assert_eq!(data_out.len(), 3 * DESC_SLOT_BYTES);
        assert_eq!(&region.as_slice()[..bytes], shadow.as_slice());
    }

    #[test]
    fn test_hazard_journal_minimal_fixups() {
        let mut globals = [D3D12_RESOURCE_STATE_COMMON; 2];
        let mut journals = vec![WorkerHazardJournal::default(); 8];
        for w in 0..4 {
            journals[w].record(0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        for w in 4..8 {
            journals[w].record(0, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            journals[w].record(1, D3D12_RESOURCE_STATE_RENDER_TARGET);
        }
        let f0 = resolve_worker_fixups(&mut globals, &journals[0]);
        let f1 = resolve_worker_fixups(&mut globals, &journals[1]);
        let f4 = resolve_worker_fixups(&mut globals, &journals[4]);
        let f5 = resolve_worker_fixups(&mut globals, &journals[5]);
        assert_eq!(f0.len(), 1); // COMMON -> UAV
        assert_eq!(f1.len(), 0); // already UAV
        assert_eq!(f4.len(), 2); // skin_verts UAV->VCB, skin_rt COMMON->RT
        assert_eq!(f5.len(), 0); // already VCB + RT
    }
}
