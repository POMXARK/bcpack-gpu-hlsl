use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, pack_batch, transition_barrier, Bc7RansGpuDecoder,
    BcpackGpuDecoder, Gfx10DetileGpu, GpuContext,
};
use bcpack::bc1::{QuadTable, Window};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::copy_nonoverlapping;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::INFINITE;

pub const BCPACK_FORMAT_BC1: u32 = 0;
pub const BCPACK_FORMAT_BC7_RANS: u32 = 1;

pub const BCPACK_SWIZZLE_LINEAR: u32 = 0;
pub const BCPACK_SWIZZLE_GFX10_64KB_D: u32 = 1;
pub const BCPACK_SWIZZLE_GFX10_4KB_S: u32 = 2;

pub struct BcPackContext {
    ctx: GpuContext,
    bc1_dec: BcpackGpuDecoder,
    bc7_dec: Bc7RansGpuDecoder,
    detile: Gfx10DetileGpu,
    detile_out_default: ID3D12Resource,
    detile_readback: ID3D12Resource,
    ts_query_heap: ID3D12QueryHeap,
    ts_readback: ID3D12Resource,
    max_output_bytes: usize,
    last_kernel_ms: f64,
}

unsafe fn init_context_with_gpu(
    mut ctx: GpuContext,
    max_chunks: u32,
    max_blocks: u32,
    max_stream_bytes: u32,
) -> *mut BcPackContext {
    let chunks = if max_chunks == 0 { 2048 } else { max_chunks as usize };
    let blocks = if max_blocks == 0 { 8_388_608 } else { max_blocks as usize };
    let stream_cap = if max_stream_bytes == 0 {
        96 * 1024 * 1024
    } else {
        max_stream_bytes as usize
    };

    let window = Window::default_table();
    let quads = QuadTable::default_table();
    let bc1_dec = BcpackGpuDecoder::new(&mut ctx, &window, &quads, chunks, blocks, stream_cap);
    let bc7_sym_rows = (blocks / 2).max(4096);
    let bc7_dec = Bc7RansGpuDecoder::new(&mut ctx, chunks, bc7_sym_rows, stream_cap);
    let out_bytes = (blocks * 8).max(bc7_sym_rows * 32).max(16 * 1024 * 1024);
    let detile = Gfx10DetileGpu::new(&ctx, 16 * 1024 * 1024);
    let detile_out_default = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_DEFAULT,
        out_bytes as u64,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let detile_readback = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_READBACK,
        out_bytes as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let mut ts_query_heap: Option<ID3D12QueryHeap> = None;
    ctx.device
        .CreateQueryHeap(
            &D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: 2,
                NodeMask: 0,
            },
            &mut ts_query_heap,
        )
        .expect("CreateQueryHeap FFI");
    let ts_query_heap = ts_query_heap.unwrap();
    let ts_readback = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_READBACK,
        16,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );

    Box::into_raw(Box::new(BcPackContext {
        ctx,
        bc1_dec,
        bc7_dec,
        detile,
        detile_out_default,
        detile_readback,
        ts_query_heap,
        ts_readback,
        max_output_bytes: out_bytes,
        last_kernel_ms: 0.0,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn bcpack_gpu_create(
    max_chunks: u32,
    max_blocks: u32,
    max_stream_bytes: u32,
) -> *mut BcPackContext {
    let res = catch_unwind(|| {
        let ctx = GpuContext::new(false);
        init_context_with_gpu(ctx, max_chunks, max_blocks, max_stream_bytes)
    });
    res.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn bcpack_gpu_create_from_device(
    d3d12_device: *mut c_void,
    d3d12_compute_queue: *mut c_void,
    max_chunks: u32,
    max_blocks: u32,
    max_stream_bytes: u32,
) -> *mut BcPackContext {
    if d3d12_device.is_null() {
        return std::ptr::null_mut();
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let dev: ID3D12Device = (*std::mem::ManuallyDrop::new(ID3D12Device::from_raw(d3d12_device))).clone();
        let cq = if d3d12_compute_queue.is_null() {
            None
        } else {
            let q: ID3D12CommandQueue = (*std::mem::ManuallyDrop::new(ID3D12CommandQueue::from_raw(d3d12_compute_queue))).clone();
            Some(q)
        };
        let ctx = GpuContext::from_external_device(dev, cq);
        init_context_with_gpu(ctx, max_chunks, max_blocks, max_stream_bytes)
    }));
    res.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn bcpack_gpu_destroy(ctx: *mut BcPackContext) {
    if !ctx.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let state = Box::from_raw(ctx);
            if state.ctx.compute_fence_value > 0 {
                let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
            }
            drop(state);
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn BcPackDecodeAsync(
    ctx: *mut BcPackContext,
    compressed_data: *const u8,
    compressed_size: u32,
    num_blocks: u32,
    format: u32,
    swizzle_mode: u32,
    width_blocks: u32,
    height_blocks: u32,
    out_host_buffer: *mut u8,
    out_capacity: u32,
    out_fence_value: *mut u64,
) -> i32 {
    if ctx.is_null() || compressed_data.is_null() || compressed_size == 0 || num_blocks == 0 {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        let stream = std::slice::from_raw_parts(compressed_data, compressed_size as usize);

        if format == BCPACK_FORMAT_BC7_RANS {
            let sym_rows = (num_blocks as usize) * 2;
            let expected_bytes = sym_rows * 32;
            if !out_host_buffer.is_null() && (out_capacity as usize) < expected_bytes {
                return -2;
            }
            let batch = pack_batch(std::iter::once((stream, sym_rows)));
            if out_host_buffer.is_null() {
                let fence_val = state.bc7_dec.submit_async_compute(&mut state.ctx, &batch);
                if !out_fence_value.is_null() {
                    *out_fence_value = fence_val;
                }
                return 0;
            }
            let (decoded_u16, statuses, kms, _) = state.bc7_dec.decode_batch_and_readback(&mut state.ctx, &batch);
            if statuses.first().map(|s| s.code).unwrap_or(99) != 0 {
                return -3;
            }
            state.last_kernel_ms = kms;
            copy_nonoverlapping(decoded_u16.as_ptr() as *const u8, out_host_buffer, expected_bytes);
            if !out_fence_value.is_null() {
                *out_fence_value = state.ctx.compute_fence_value;
            }
            return 0;
        }

        if format != BCPACK_FORMAT_BC1 {
            return -4;
        }

        let need_detile = swizzle_mode != BCPACK_SWIZZLE_LINEAR;
        let linear_blocks = if need_detile {
            (width_blocks as usize) * (height_blocks as usize)
        } else {
            num_blocks as usize
        };
        let expected_bytes = linear_blocks * 8;
        if expected_bytes > state.max_output_bytes {
            return -5;
        }
        if !out_host_buffer.is_null() && (out_capacity as usize) < expected_bytes {
            return -2;
        }

        let batch = pack_batch(std::iter::once((stream, num_blocks as usize)));

        if !need_detile {
            if out_host_buffer.is_null() {
                let fence_val = state.bc1_dec.submit_async_compute(&mut state.ctx, &batch);
                if !out_fence_value.is_null() {
                    *out_fence_value = fence_val;
                }
                return 0;
            }
            let (decoded, statuses, kms, _) = state.bc1_dec.decode_batch_and_readback(&mut state.ctx, &batch);
            if statuses.first().map(|s| s.code).unwrap_or(99) != 0 {
                return -3;
            }
            state.last_kernel_ms = kms;
            copy_nonoverlapping(decoded.as_ptr(), out_host_buffer, expected_bytes);
            if !out_fence_value.is_null() {
                *out_fence_value = state.ctx.compute_fence_value;
            }
            return 0;
        }

        // Combined BCPack BC1 decode + Gfx10 Swizzle -> Linear in a single Async Compute command list
        let tile_mode = match swizzle_mode {
            BCPACK_SWIZZLE_GFX10_64KB_D => 0u32,
            BCPACK_SWIZZLE_GFX10_4KB_S => 1u32,
            _ => return -6,
        };
        state.bc1_dec.upload_batch_slot(0, &batch);
        {
            let slot0 = &state.bc1_dec.slots[0];
            slot0.cmd_alloc.Reset().unwrap();
            slot0.cmd_list.Reset(&slot0.cmd_alloc, None).unwrap();
            slot0.cmd_list.EndQuery(&state.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        }
        state.bc1_dec.record_slot_upload_and_dispatch(0, &state.bc1_dec.slots[0].cmd_list, &batch, None, 0, 0);
        {
            let slot0 = &state.bc1_dec.slots[0];
            cmd_barriers(
                &slot0.cmd_list,
                vec![
                    transition_barrier(&slot0.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                    transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                ],
            );
            state.detile.record_detile(
                &slot0.cmd_list,
                slot0.out_blocks_default.GetGPUVirtualAddress(),
                state.detile_out_default.GetGPUVirtualAddress(),
                width_blocks,
                height_blocks,
                tile_mode,
                8,
                0,
                0,
                num_blocks,
            );
            slot0.cmd_list.EndQuery(&state.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

            if !out_host_buffer.is_null() {
                cmd_barriers(
                    &slot0.cmd_list,
                    vec![
                        transition_barrier(&slot0.out_blocks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                        transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    ],
                );
                slot0.cmd_list.CopyBufferRegion(&state.detile_readback, 0, &state.detile_out_default, 0, expected_bytes as u64);
                slot0.cmd_list.ResolveQueryData(&state.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &state.ts_readback, 0);
                cmd_barriers(
                    &slot0.cmd_list,
                    vec![transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)],
                );
                slot0.cmd_list.Close().unwrap();
                let lists = [Some(slot0.cmd_list.cast::<ID3D12CommandList>().unwrap())];
                state.ctx.compute_queue.ExecuteCommandLists(&lists);
                state.ctx.signal_and_wait_compute();

                let mut ptr: *mut c_void = std::ptr::null_mut();
                state.detile_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: expected_bytes }), Some(&mut ptr)).unwrap();
                copy_nonoverlapping(ptr as *const u8, out_host_buffer, expected_bytes);
                state.detile_readback.Unmap(0, None);

                state.ts_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut ptr)).unwrap();
                let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
                state.last_kernel_ms = (ts[1].saturating_sub(ts[0]) as f64 / state.ctx.compute_timestamp_freq as f64) * 1000.0;
                state.ts_readback.Unmap(0, None);

                if !out_fence_value.is_null() {
                    *out_fence_value = state.ctx.compute_fence_value;
                }
                0
            } else {
                slot0.cmd_list.ResolveQueryData(&state.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &state.ts_readback, 0);
                cmd_barriers(
                    &slot0.cmd_list,
                    vec![
                        transition_barrier(&slot0.out_blocks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                        transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                    ],
                );
                slot0.cmd_list.Close().unwrap();
                let lists = [Some(slot0.cmd_list.cast::<ID3D12CommandList>().unwrap())];
                state.ctx.compute_queue.ExecuteCommandLists(&lists);
                let fence_val = state.ctx.signal_compute();
                if !out_fence_value.is_null() {
                    *out_fence_value = fence_val;
                }
                0
            }
        }
    }));
    res.unwrap_or(-99)
}

#[no_mangle]
pub unsafe extern "C" fn BcPackGetComputeFence(ctx: *const BcPackContext) -> *mut c_void {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    (*ctx).ctx.compute_fence.as_raw()
}

#[no_mangle]
pub unsafe extern "C" fn BcPackDecodeToBuffer(
    ctx: *mut BcPackContext,
    compressed_data: *const u8,
    compressed_size: u32,
    num_blocks: u32,
    format: u32,
    swizzle_mode: u32,
    width_blocks: u32,
    height_blocks: u32,
    dst_d3d12_resource: *mut c_void,
    dst_byte_offset: u64,
    out_fence_value: *mut u64,
) -> i32 {
    if ctx.is_null()
        || compressed_data.is_null()
        || compressed_size == 0
        || num_blocks == 0
        || dst_d3d12_resource.is_null()
    {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        if state.ctx.compute_fence_value > 0 {
            let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
        }
        let dst_res: ID3D12Resource = (*std::mem::ManuallyDrop::new(ID3D12Resource::from_raw(dst_d3d12_resource))).clone();
        let stream = std::slice::from_raw_parts(compressed_data, compressed_size as usize);

        if format == BCPACK_FORMAT_BC7_RANS {
            let sym_rows = (num_blocks as usize) * 2;
            let expected_bytes = sym_rows * 32;
            if expected_bytes > state.max_output_bytes {
                return -5;
            }
            let batch = pack_batch(std::iter::once((stream, sym_rows)));
            let fence_val = state.bc7_dec.submit_async_to_buffer(&mut state.ctx, &batch, &dst_res, dst_byte_offset);
            if !out_fence_value.is_null() {
                *out_fence_value = fence_val;
            }
            return 0;
        }

        if format != BCPACK_FORMAT_BC1 {
            return -4;
        }

        let need_detile = swizzle_mode != BCPACK_SWIZZLE_LINEAR;
        let linear_blocks = if need_detile {
            (width_blocks as usize) * (height_blocks as usize)
        } else {
            num_blocks as usize
        };
        let expected_bytes = linear_blocks * 8;
        if expected_bytes > state.max_output_bytes {
            return -5;
        }
        let batch = pack_batch(std::iter::once((stream, num_blocks as usize)));

        state.bc1_dec.upload_batch_slot(0, &batch);
        {
            let slot0 = &state.bc1_dec.slots[0];
            slot0.cmd_alloc.Reset().unwrap();
            slot0.cmd_list.Reset(&slot0.cmd_alloc, None).unwrap();
            slot0.cmd_list.EndQuery(&state.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        }
        state.bc1_dec.record_slot_upload_and_dispatch(0, &state.bc1_dec.slots[0].cmd_list, &batch, None, 0, 0);
        {
            let slot0 = &state.bc1_dec.slots[0];
            if !need_detile {
                slot0.cmd_list.EndQuery(&state.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                cmd_barriers(
                    &slot0.cmd_list,
                    vec![
                        transition_barrier(&slot0.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                        transition_barrier(&dst_res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                    ],
                );
                slot0.cmd_list.CopyBufferRegion(&dst_res, dst_byte_offset, &slot0.out_blocks_default, 0, expected_bytes as u64);
                slot0.cmd_list.ResolveQueryData(&state.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &state.ts_readback, 0);
                cmd_barriers(
                    &slot0.cmd_list,
                    vec![
                        transition_barrier(&slot0.out_blocks_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                        transition_barrier(&dst_res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                    ],
                );
            } else {
                let tile_mode = match swizzle_mode {
                    BCPACK_SWIZZLE_GFX10_64KB_D => 0u32,
                    BCPACK_SWIZZLE_GFX10_4KB_S => 1u32,
                    _ => return -6,
                };
                cmd_barriers(
                    &slot0.cmd_list,
                    vec![
                        transition_barrier(&slot0.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                        transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    ],
                );
                state.detile.record_detile(
                    &slot0.cmd_list,
                    slot0.out_blocks_default.GetGPUVirtualAddress(),
                    state.detile_out_default.GetGPUVirtualAddress(),
                    width_blocks,
                    height_blocks,
                    tile_mode,
                    8,
                    0,
                    0,
                    num_blocks,
                );
                slot0.cmd_list.EndQuery(&state.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                cmd_barriers(
                    &slot0.cmd_list,
                    vec![
                        transition_barrier(&slot0.out_blocks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                        transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
                        transition_barrier(&dst_res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                    ],
                );
                slot0.cmd_list.CopyBufferRegion(&dst_res, dst_byte_offset, &state.detile_out_default, 0, expected_bytes as u64);
                slot0.cmd_list.ResolveQueryData(&state.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &state.ts_readback, 0);
                cmd_barriers(
                    &slot0.cmd_list,
                    vec![
                        transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                        transition_barrier(&dst_res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                    ],
                );
            }
            slot0.cmd_list.Close().unwrap();
            let lists = [Some(slot0.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            state.ctx.compute_queue.ExecuteCommandLists(&lists);
            let fence_val = state.ctx.signal_compute();
            if !out_fence_value.is_null() {
                *out_fence_value = fence_val;
            }
            0
        }
    }));
    res.unwrap_or(-99)
}

#[no_mangle]
pub unsafe extern "C" fn BcPackWaitFence(
    ctx: *mut BcPackContext,
    fence_value: u64,
    timeout_ms: u32,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let state = &*ctx;
    let ms = if timeout_ms == u32::MAX { INFINITE } else { timeout_ms };
    if state.ctx.wait_compute_cpu(fence_value, ms) {
        0
    } else {
        1
    }
}

#[no_mangle]
pub unsafe extern "C" fn BcPackGetLastTimingMs(
    ctx: *const BcPackContext,
    out_kernel_ms: *mut f64,
) -> i32 {
    if ctx.is_null() || out_kernel_ms.is_null() {
        return -1;
    }
    *out_kernel_ms = (*ctx).last_kernel_ms;
    0
}
