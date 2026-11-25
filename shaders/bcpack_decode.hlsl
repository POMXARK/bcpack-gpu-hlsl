// Clean-room Xbox Series BCPack (BC1) Compute Shader decoder (HLSL SM 6.6)
// Maps 1 Wave32 warp (lanes 0..11 active) per compressed chunk.

struct ModelData {
    uint thr[10];
    uint base[10];
    uint freq[10];
    uint rf[10];
    uint rs[10];
    uint rn[10];
};

struct ChunkDesc {
    uint stream_offset;
    uint stream_len;
    uint blocks;
    uint out_block_offset;
};

StructuredBuffer<ModelData> g_Models    : register(t0);
StructuredBuffer<uint>      g_Window    : register(t1);
StructuredBuffer<uint>      g_QuadMasks : register(t2);
ByteAddressBuffer           g_Streams   : register(t3);
StructuredBuffer<ChunkDesc> g_Chunks    : register(t4);

RWStructuredBuffer<uint2>   g_OutBlocks : register(u0);
RWStructuredBuffer<uint4>   g_Status    : register(u1);

static const uint IDLE  = 0u;
static const uint HDR0  = 1u;
static const uint HDR1  = 2u;
static const uint HDR2  = 3u;
static const uint PROC  = 4u;
static const uint DRAIN = 5u;

static const uint RANS_L = 0x01000000u;

static const uint BC1_ALPHABET_BITS[12] = { 7u, 7u, 4u, 5u, 4u, 4u, 5u, 4u, 7u, 7u, 7u, 7u };

groupshared uint ring_ep[256];

uint load_byte(uint addr) {
    uint word = g_Streams.Load(addr & ~3u);
    return (word >> ((addr & 3u) * 8u)) & 0xFFu;
}

uint inverse_fast(uint mid, uint cf) {
    ModelData m = g_Models[mid];
    uint seg = 0u;
    [unroll]
    for (uint k = 1u; k < 10u; ++k) {
        seg += (cf >= m.thr[k]) ? 1u : 0u;
    }
    uint base_x = (1u << seg) >> 1u;
    uint dist = (cf - m.base[seg] + m.rn[seg]) & 0xFFFFu;
    uint steps = (((dist * m.rf[seg]) >> 15u) >> m.rs[seg]) & 0x1FFu;
    return (base_x + steps) & 0x1FFu;
}

uint advance_state(uint mid, uint x, uint sym) {
    ModelData m = g_Models[mid];
    uint seg = (sym == 0u) ? 0u : (firstbithigh(sym) + 1u);
    uint st = (seg == 0u) ? 0u : (1u << (seg - 1u));
    uint fa = m.freq[seg];
    uint cfa = (m.base[seg] + fa * (sym - st)) & 0xFFFFu;
    uint f = (cfa == 0u) ? ((fa - 1u) & 0xFFFFu) : fa;
    uint c = (cfa == 0u) ? 1u : cfa;
    return f * (x >> 16u) + (x & 0xFFFFu) - c;
}

bool is_legal_end_state(uint s) {
    return (s == RANS_L) || (s == (RANS_L >> 8u)) || (s == (RANS_L >> 16u));
}

uint back_distance(uint v) {
    v &= 0xFFu;
    return ((v & 1u) == 0u) ? (v >> 1u) : (255u - (v >> 1u));
}

int unzigzag(uint v) {
    int iv = int(v);
    return ((iv & 1) != 0) ? -((iv + 1) >> 1) : (iv >> 1);
}

[WaveSize(32)]
[numthreads(32, 1, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex) {
    uint chunk_id = groupID.x;
    ChunkDesc desc = g_Chunks[chunk_id];

    // Pre-fill 256-entry preset endpoint window in groupshared memory
    for (uint i = lane; i < 256u; i += 32u) {
        ring_ep[i] = g_Window[i];
    }
    GroupMemoryBarrierWithGroupSync();

    bool active_lane = (lane < 12u);
    uint bit = active_lane ? (1u << lane) : 0u;
    uint ab = active_lane ? BC1_ALPHABET_BITS[lane] : 0u;

    uint state = IDLE;
    uint ib = 0u;
    bool ib_full = false;
    uint sym = 0u;
    bool sym_full = false;
    uint slot_sym = 0u;
    uint in_rem = 0u;
    uint out_rem = 0u;
    uint p0 = 0u;
    uint p1 = 0u;
    bool hdr3 = false;
    bool flush = false;
    uint q = 0u;
    uint eab = 0u;
    uint model_id = 0u;
    bool done = false;
    bool go = true;
    bool exhausted = false;
    uint final_state = 0u;

    // Wave-uniform state
    uint slots = 0u;
    uint stream_pos = 0u;
    uint block_idx = 0u;
    uint cycle = 0u;
    uint max_cycles = 64u * (12u * desc.blocks + desc.stream_len) + 1000u;
    uint3 last_prog = uint3(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    uint no_prog = 0u;

    while (cycle < max_cycles) {
        // ---------------- Phase 0: forward update, feed (max 8 bytes/clock), hand-off ----------------
        bool need_byte = active_lane && !ib_full && (
            (state == HDR0 || state == HDR1 || state == HDR2) ||
            (state == PROC && in_rem > 0u)
        );
        bool blocked_out = active_lane && sym_full && ((slots & bit) != 0u);
        bool want_byte = active_lane && need_byte && !blocked_out && !exhausted;

        uint want_mask = WaveActiveBallot(want_byte).x;
        uint rank = countbits(want_mask & ((1u << lane) - 1u));
        bool in_window = want_byte && (rank < 8u);
        uint byte_idx = stream_pos + rank;
        bool valid = in_window && (byte_idx < desc.stream_len);
        if (in_window && !valid) {
            exhausted = true;
        }
        uint valid_mask = WaveActiveBallot(valid).x;
        stream_pos += countbits(valid_mask);

        uint fetched = valid ? load_byte(desc.stream_offset + byte_idx) : 0u;
        bool stall = active_lane && ((need_byte && !valid) || blocked_out);

        if (active_lane && !stall) {
            if (state == PROC || state == DRAIN) {
                p0 = sym_full ? advance_state(model_id, p1, sym) : p1;
            }
            if (valid) {
                ib = fetched;
                ib_full = true;
            }
        }

        bool handoff = active_lane && !stall && sym_full;
        if (handoff) {
            slot_sym = sym;
            sym_full = false;
        }
        slots |= WaveActiveBallot(handoff).x;

        // ---------------- Phase 1: header FSM, renorm, decode ----------------
        bool bad_hdr = false;
        if (active_lane && !stall) {
            if (state == IDLE) {
                if (go) {
                    go = false;
                    state = HDR0;
                    out_rem = desc.blocks;
                    p0 = 0u;
                    p1 = 0u;
                }
            } else if (state == HDR0) {
                if (ib_full) {
                    uint b = ib;
                    ib_full = false;
                    in_rem = 0u;
                    state = HDR1;
                    q = b & 15u;
                    hdr3 = (b & 0x20u) != 0u;
                    flush = (b & 0x10u) != 0u;
                    int e = (int(ab) - int(b >> 6u)) & 15;
                    eab = (e < 10) ? uint(e) : 9u;
                    if (eab > 8u) {
                        bad_hdr = true;
                    }
                }
            } else if (state == HDR1) {
                if (ib_full) {
                    ib_full = false;
                    model_id = eab * 16u + q;
                    if (hdr3) {
                        in_rem = ib;
                        state = HDR2;
                    } else {
                        in_rem = ib + 1u;
                        state = PROC;
                    }
                }
            } else if (state == HDR2) {
                if (ib_full) {
                    ib_full = false;
                    in_rem = ((ib << 8u) | (in_rem & 0xFFu)) + 257u;
                    state = PROC;
                }
            } else if (state == PROC) {
                bool will = (p0 < RANS_L) && (in_rem > 0u) && ib_full;
                uint x = will ? (((p0 & 0xFFFFFFu) << 8u) | ib) : p0;
                bool nsf = (x > RANS_L) && (out_rem > 0u);
                if (nsf) {
                    out_rem -= 1u;
                    sym = inverse_fast(model_id, x & 0xFFFFu);
                }
                p1 = x;
                sym_full = nsf;
                if (will) {
                    in_rem -= 1u;
                    ib_full = false;
                }
                if (in_rem == 0u) {
                    if (out_rem > 0u) {
                        if (x == RANS_L) {
                            state = HDR0;
                            if (flush) {
                                p1 = 0u;
                            }
                        } else {
                            state = DRAIN;
                        }
                    } else {
                        done = true;
                        state = IDLE;
                        final_state = nsf ? advance_state(model_id, x, sym) : x;
                    }
                }
            } else { // DRAIN
                uint x = p0;
                bool nsf = (x > RANS_L) && (out_rem > 0u);
                if (nsf) {
                    out_rem -= 1u;
                    sym = inverse_fast(model_id, x & 0xFFFFu);
                }
                p1 = x;
                sym_full = nsf;
                if (x <= RANS_L) {
                    if (out_rem > 0u) {
                        state = HDR0;
                        if (flush) {
                            p0 = 0u;
                            p1 = 0u;
                        }
                    } else {
                        state = IDLE;
                        done = true;
                        final_state = nsf ? advance_state(model_id, x, sym) : x;
                    }
                }
            }
        }

        if (WaveActiveAnyTrue(bad_hdr)) {
            if (lane == 0u) {
                g_Status[chunk_id] = uint4(1u, stream_pos, cycle, block_idx);
            }
            break;
        }

        uint active_mask = WaveActiveBallot(active_lane && !(done && !sym_full && ((slots & bit) == 0u))).x & 0xFFFu;
        bool all_finished = WaveActiveAllTrue(!active_lane || (done && !sym_full));

        // ---------------- Lockstep symbol vector -> BC1 block ----------------
        if (slots != 0u && ((slots & active_mask) == active_mask)) {
            if (chunk_id == 0u && block_idx == 2u && active_lane) {
                g_OutBlocks[22500u + lane] = uint2(
                    slot_sym | (eab << 16u) | (q << 20u),
                    in_rem | (state << 24u)
                );
            }
            uint s0  = WaveReadLaneAt(slot_sym, 0);
            uint s2  = WaveReadLaneAt(slot_sym, 2);
            uint s3  = WaveReadLaneAt(slot_sym, 3);
            uint s4  = WaveReadLaneAt(slot_sym, 4);
            uint s5  = WaveReadLaneAt(slot_sym, 5);
            uint s6  = WaveReadLaneAt(slot_sym, 6);
            uint s7  = WaveReadLaneAt(slot_sym, 7);
            uint s8  = WaveReadLaneAt(slot_sym, 8);
            uint s9  = WaveReadLaneAt(slot_sym, 9);
            uint s10 = WaveReadLaneAt(slot_sym, 10);
            uint s11 = WaveReadLaneAt(slot_sym, 11);

            if (lane == 0u && block_idx < desc.blocks) {
                uint d = back_distance(s0);
                uint pred = 0u;
                if (d > 0u) {
                    uint slot = uint((int(block_idx) - int(d)) & 255);
                    pred = ring_ep[slot];
                }

                uint pr0 = (pred >> 11u) & 31u;
                uint pg0 = (pred >> 5u)  & 63u;
                uint pb0 = pred          & 31u;
                uint pr1 = (pred >> 27u) & 31u;
                uint pg1 = (pred >> 21u) & 63u;
                uint pb1 = (pred >> 16u) & 31u;

                uint r0 = uint(int(pr0) + unzigzag(s2)) & 31u;
                uint g0 = uint(int(pg0) + unzigzag(s3)) & 63u;
                uint b0 = uint(int(pb0) + unzigzag(s4)) & 31u;
                uint r1 = uint(int(pr1) + unzigzag(s5)) & 31u;
                uint g1 = uint(int(pg1) + unzigzag(s6)) & 63u;
                uint b1 = uint(int(pb1) + unzigzag(s7)) & 31u;

                uint c0 = (r0 << 11u) | (g0 << 5u) | b0;
                uint c1 = (r1 << 11u) | (g1 << 5u) | b1;
                uint ep = c0 | (c1 << 16u);
                ring_ep[block_idx & 255u] = ep;

                uint idx_word = g_QuadMasks[s8 & 255u]
                              | g_QuadMasks[256u + (s9 & 255u)]
                              | g_QuadMasks[512u + (s10 & 255u)]
                              | g_QuadMasks[768u + (s11 & 255u)];

                g_OutBlocks[desc.out_block_offset + block_idx] = uint2(ep, idx_word);
            }

            block_idx += 1u;
            slots = 0u;
        }

        cycle += 1u;

        if (all_finished) {
            bool bad_end = WaveActiveAnyTrue(active_lane && !is_legal_end_state(final_state));
            if (lane == 0u) {
                g_Status[chunk_id] = uint4(bad_end ? 2u : 0u, stream_pos, cycle, block_idx);
            }
            break;
        }

        uint state_sum = WaveActiveSum(active_lane ? state : 0u);
        uint3 prog = uint3(stream_pos, block_idx, state_sum);
        if (all(prog == last_prog)) {
            no_prog += 1u;
            if (no_prog > 64u) {
                uint done_mask = WaveActiveBallot(active_lane && done).x & 0xFFFu;
                uint sym_mask = WaveActiveBallot(active_lane && sym_full).x & 0xFFFu;
                if (chunk_id == 0u && active_lane) {
                    uint d0 = (state & 7u) | ((ib_full ? 1u : 0u) << 3u) | ((sym_full ? 1u : 0u) << 4u) | ((done ? 1u : 0u) << 5u) | ((eab & 15u) << 8u) | ((q & 15u) << 12u) | ((in_rem & 0xFFFFu) << 16u);
                    uint d1 = (out_rem & 0xFFFFu) | ((p0 >> 16u) << 16u);
                    g_OutBlocks[20000u + lane] = uint2(d0, d1);
                }
                if (lane == 0u) {
                    g_Status[chunk_id] = uint4(3u, stream_pos, (done_mask << 16u) | sym_mask, block_idx);
                }
                break;
            }
        } else {
            last_prog = prog;
            no_prog = 0u;
        }
    }

    if (cycle >= max_cycles) {
        if (lane == 0u) {
            g_Status[chunk_id] = uint4(4u, stream_pos, cycle, block_idx);
        }
    }
}
