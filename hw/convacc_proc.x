#![feature(explicit_state_access)]

import mac64;
import requant;
import std;

pub struct Load {
    target: u2,
    addr: u32,
    data: u64,
    strb: u8,
}

pub struct Engine<M: u32> {
    act: u64[M],
    wgt: u64[M],
    out_mem: u64[M],
    cfg: u64[16],
    taps: u8[256],
    params: u64[10],
    acc: s32[64],
    act_tile: s8[128],
    ch_tile: s8[1024],
    wgt_tile: u8[128],
    phase: u8,
    desc_start: u32,
    desc_index: u32,
    tap_index: u32,
    next_desc: u32,
    mt: u32,
    nt: u32,
    kb: u32,
    ld_p: u32,
    ld_r: u32,
    wg_index: u32,
    group_index: u32,
    param_index: u32,
    out_p: u32,
    n_tiles: u32,
    m_tiles: u32,
    block_groups: u32,
    bank: u1,
    err: u1,
}

pub fn zero_engine<M: u32>() -> Engine<M> {
    zero!<Engine<M>>()

}

fn cw<M: u32>(st: Engine<M>, index: u32) -> u64 { st.cfg[index] }
fn lo32(x: u64) -> u32 { x as u32 }
fn hi32(x: u64) -> u32 { (x >> u32:32) as u32 }
fn half16(x: u64, slot: u32) -> u16 { (x >> (slot * u32:16)) as u16 }
fn byte_at(x: u64, index: u32) -> u8 { (x >> (index * u32:8)) as u8 }

fn flags_gemm<M: u32>(st: Engine<M>) -> bool { (cw(st, u32:0) & u64:1) != u64:0 }
fn flags_channel<M: u32>(st: Engine<M>) -> bool { (cw(st, u32:0) & u64:2) != u64:0 }
fn flags_deconv<M: u32>(st: Engine<M>) -> bool { (cw(st, u32:0) & u64:4) != u64:0 }
fn flags_logr<M: u32>(st: Engine<M>) -> u2 { ((cw(st, u32:0) >> u32:3) as u2) }
fn cfg_r<M: u32>(st: Engine<M>) -> u32 { u32:1 << flags_logr(st) }
fn cfg_o<M: u32>(st: Engine<M>) -> u32 { u32:8 >> flags_logr(st) }
fn cfg_m<M: u32>(st: Engine<M>) -> u32 { lo32(cw(st, u32:1)) }
fn cfg_n<M: u32>(st: Engine<M>) -> u32 { hi32(cw(st, u32:1)) }
fn cfg_k<M: u32>(st: Engine<M>) -> u32 { lo32(cw(st, u32:2)) }
fn cfg_mh<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:3), u32:0) as u32 }
fn cfg_mw<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:3), u32:1) as u32 }
fn cfg_kgroups<M: u32>(st: Engine<M>) -> u32 { lo32(cw(st, u32:4)) }
fn cfg_ci_count<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:5), u32:0) as u32 }
fn cfg_ci_base<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:5), u32:1) as u32 }
fn cfg_co_base<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:5), u32:2) as u32 }
fn cfg_co_group<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:5), u32:3) as u32 }
fn cfg_hi<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:6), u32:0) as u32 }
fn cfg_wi<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:6), u32:1) as u32 }
fn cfg_ho<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:7), u32:0) as u32 }
fn cfg_wo<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:7), u32:1) as u32 }
fn cfg_ci<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:8), u32:0) as u32 }
fn cfg_co<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:8), u32:1) as u32 }
fn cfg_sh<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:9), u32:0) as u32 }
fn cfg_sw<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:9), u32:1) as u32 }
fn cfg_dh<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:9), u32:2) as u32 }
fn cfg_dw<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:9), u32:3) as u32 }
fn cfg_ph<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:10), u32:0) as u32 }
fn cfg_pw<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:10), u32:1) as u32 }
fn cfg_phase_h<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:10), u32:2) as u32 }
fn cfg_phase_w<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:10), u32:3) as u32 }
fn cfg_steph<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:11), u32:0) as u32 }
fn cfg_stepw<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:11), u32:1) as u32 }
fn cfg_act_base<M: u32>(st: Engine<M>) -> u32 { lo32(cw(st, u32:12)) }
fn cfg_out_base<M: u32>(st: Engine<M>) -> u32 { hi32(cw(st, u32:12)) }
fn cfg_w_base<M: u32>(st: Engine<M>) -> u32 { lo32(cw(st, u32:13)) }
fn cfg_p_base<M: u32>(st: Engine<M>) -> u32 { hi32(cw(st, u32:13)) }
fn cfg_tap_base<M: u32>(st: Engine<M>) -> u32 { lo32(cw(st, u32:14)) }
fn cfg_taps<M: u32>(st: Engine<M>) -> u32 { half16(cw(st, u32:14), u32:2) as u32 }

/// Real output channels left in the current n-tile. Phase 11 writes whole
/// words and caps at the tile width `cfg_o`; the byte-granular activation
/// loads cap at the full 64-bit word instead.
fn cfg_valid_o_tile<M: u32>(st: Engine<M>) -> u32 {
    let remaining = cfg_n(st) - st.nt * cfg_o(st);
    if remaining > cfg_o(st) { cfg_o(st) } else { remaining }
}

fn cfg_valid_o8<M: u32>(st: Engine<M>) -> u32 {
    let remaining = cfg_n(st) - st.nt * cfg_o(st);
    if remaining > u32:8 { u32:8 } else { remaining }
}

pub fn merge_bytes(old: u64, data: u64, strb: u8) -> u64 {
    if strb == u8:255 {
        data
    } else {
        for (i, word): (u32, u64) in u32:0..u32:8 {
            let mask = u64:255 << (i * u32:8);
            if ((strb >> i) as u1) == u1:1 { (word & !mask) | (data & mask) } else { word }
        }(old)
    }
}

fn apply_load<M: u32>(st: Engine<M>, request: Load) -> Engine<M> {
    let addr = request.addr;
    if addr >= M { st } else if request.target == u2:0 {
        Engine { act: update(st.act, addr, merge_bytes(st.act[addr], request.data, request.strb)), ..st }
    } else if request.target == u2:1 {
        Engine { wgt: update(st.wgt, addr, merge_bytes(st.wgt[addr], request.data, request.strb)), ..st }
    } else {
        Engine { out_mem: update(st.out_mem, addr, merge_bytes(st.out_mem[addr], request.data, request.strb)), ..st }
    }
}

/// Byte address of the output tile for row `p` of the current n-tile.
///
/// The image layout is 8-byte aligned so that every tile is exactly one 64-bit
/// word and can be stored without byte enables:
///   out_base + pos * (groups * n_tiles * 8) + (group * n_tiles + nt) * 8
/// `pos` is the flattened (batch, oh, ow) position, or the row index for GEMM.
/// Lane `i` of tile `nt` of `group` holds output channel
/// `group * co_per_group + nt * o + i`. `out_base` must be 64-bit aligned.
fn out_group<M: u32>(st: Engine<M>) -> u32 {
    if flags_gemm(st) || flags_channel(st) { u32:0 } else { cfg_co_base(st) / cfg_co_group(st) }
}

fn out_stride_tiles<M: u32>(st: Engine<M>) -> u32 {
    if flags_gemm(st) || flags_channel(st) {
        st.n_tiles
    } else {
        (cfg_co(st) / cfg_co_group(st)) * st.n_tiles
    }
}

fn out_word<M: u32>(st: Engine<M>, p: u32) -> u32 {
    let per_batch = cfg_mh(st) * cfg_mw(st);
    let m_abs = st.mt * u32:8 + p;
    let batch = m_abs / per_batch;
    let rem = m_abs % per_batch;
    let oh = cfg_phase_h(st) + (rem / cfg_mw(st)) * cfg_steph(st);
    let ow = cfg_phase_w(st) + (rem % cfg_mw(st)) * cfg_stepw(st);
    let pos = if flags_gemm(st) {
        m_abs
    } else {
        (batch * cfg_ho(st) + oh) * cfg_wo(st) + ow
    };
    cfg_out_base(st) / u32:8 + pos * out_stride_tiles(st) + out_group(st) * st.n_tiles + st.nt
}

/// Eight s32 lanes packed as lo/hi 32-bit pairs of four parameter words
/// starting at `base` (bias uses words 0..3, scale words 4..7).
fn param_pairs<M: u32>(st: Engine<M>, base: u32) -> s32[8] {
    for (i, values): (u32, s32[8]) in u32:0..u32:8 {
        let word = st.params[base + i / u32:2];
        let raw = if (i % u32:2) == u32:0 { lo32(word) } else { hi32(word) };
        update(values, i, raw as s32)
    }(zero!<s32[8]>())
}

fn requant_lanes<M: u32>(st: Engine<M>, p: u32) -> s8[8] {
    let bias = param_pairs(st, u32:0);
    let scale = param_pairs(st, u32:4);
    let shifts = for (i, values): (u32, u6[8]) in u32:0..u32:8 {
        update(values, i, byte_at(st.params[u32:8], i) as u6)
    }(zero!<u6[8]>());
    let zps = for (i, values): (u32, s8[8]) in u32:0..u32:8 {
        update(values, i, byte_at(st.params[u32:9], i) as s8)
    }(zero!<s8[8]>());
    let lanes = for (i, values): (u32, s32[8]) in u32:0..u32:8 {
        update(values, i, st.acc[p * u32:8 + i])
    }(zero!<s32[8]>());
    requant::requant8(lanes, bias, scale, shifts, zps)
}

struct ActInfo {
    addr: u32,
    in_bounds: bool,
    byte_sel: u32,
    fill: u32,
    valid_o: u32,
}

/// Address + control for the phase-7 activation fetch of the current
/// (mt, nt, kb, ld_p, ld_r) position. Shared by the array engine and the
/// memory-mapped engine so both stay bit-identical.
fn act_info<M: u32>(st: Engine<M>) -> ActInfo {
    let r_abs = st.kb * u32:8 + st.ld_r;
    let m_abs = st.mt * u32:8 + st.ld_p;
    let per_batch = cfg_mh(st) * cfg_mw(st);
    let batch = m_abs / per_batch;
    let rem = m_abs % per_batch;
    let oh = cfg_phase_h(st) + (rem / cfg_mw(st)) * cfg_steph(st);
    let ow = cfg_phase_w(st) + (rem % cfg_mw(st)) * cfg_stepw(st);
    let channel = flags_channel(st);
    let tap_index = if channel { r_abs } else { r_abs / cfg_ci_count(st) };
    let tap_byte = if tap_index < cfg_taps(st) { st.taps[tap_index] } else { u8:0 };
    let kh = (tap_byte >> u32:4) as u32;
    let kw = (tap_byte & u8:15) as u32;
    let num_h = oh + cfg_ph(st) - kh;
    let num_w = ow + cfg_pw(st) - kw;
    let ih = if flags_deconv(st) {
        if (num_h % cfg_sh(st)) == u32:0 { num_h / cfg_sh(st) } else { u32:65535 }
    } else {
        oh * cfg_sh(st) + kh * cfg_dh(st) - cfg_ph(st)
    };
    let iw = if flags_deconv(st) {
        if (num_w % cfg_sw(st)) == u32:0 { num_w / cfg_sw(st) } else { u32:65535 }
    } else {
        ow * cfg_sw(st) + kw * cfg_dw(st) - cfg_pw(st)
    };
    let co_start = cfg_co_base(st) + st.nt * cfg_o(st);
    let valid_o = cfg_valid_o8(st);
    let ci_start = co_start / cfg_co_group(st);
    let ci_local = r_abs % cfg_ci_count(st);
    let ci_global = cfg_ci_base(st) + ci_local;
    let addr = if flags_gemm(st) {
        cfg_act_base(st) + m_abs * cfg_k(st) + r_abs
    } else if channel {
        cfg_act_base(st) + ((batch * cfg_hi(st) + ih) * cfg_wi(st) + iw) * cfg_ci(st) + ci_start
    } else {
        cfg_act_base(st) + ((batch * cfg_hi(st) + ih) * cfg_wi(st) + iw) * cfg_ci(st) + ci_global
    };
    let in_bounds = m_abs < cfg_m(st) && r_abs < cfg_k(st)
                    && (flags_gemm(st) || (ih < cfg_hi(st) && iw < cfg_wi(st)));
    let safe_addr = if in_bounds { addr } else { u32:0 };
    let byte_sel = safe_addr % u32:8;
    let fill = if channel {
        u32:1
    } else if in_bounds {
        let cap_tap = if flags_gemm(st) { u32:8 } else { cfg_ci_count(st) - ci_local };
        let cap_k = cfg_k(st) - r_abs;
        let cap_r = u32:8 - st.ld_r;
        let cap_bytes = u32:8 - byte_sel;
        let a = if cap_tap < cap_k { cap_tap } else { cap_k };
        let b = if cap_bytes < cap_r { cap_bytes } else { cap_r };
        if a < b { a } else { b }
    } else {
        u32:1
    };
    ActInfo { addr: safe_addr, in_bounds, byte_sel, fill, valid_o }
}

/// Write the output-channel bytes that fall inside the word at `addr / 8 + which`
/// (which is 0 or 1 relative to the fetch address). Channelwise loads can span a
/// word boundary; `word` must be the fetched word for that index (0 when OOB).
fn ch_fill<M: u32>(st: Engine<M>, tile: s8[1024], addr: u32, word: u64, which: u32) -> s8[1024] {
    let byte_sel = addr % u32:8;
    let valid_o = cfg_valid_o8(st);
    for (i, values): (u32, s8[1024]) in u32:0..u32:8 {
        if i < valid_o {
            let rel = i / cfg_co_group(st);
            let pos = byte_sel + rel;
            if pos / u32:8 == which {
                update(values,
                       ((st.bank ^ u1:1) as u32) * u32:512 + st.ld_p * u32:64 + st.ld_r * u32:8 + i,
                       byte_at(word, pos % u32:8) as s8)
            } else {
                values
            }
        } else {
            values
        }
    }(tile)
}

fn cfg_addr<M: u32>(st: Engine<M>) -> u32 { st.desc_start + st.desc_index }
fn tap_addr<M: u32>(st: Engine<M>) -> u32 { cfg_tap_base(st) + st.tap_index / u32:8 }
fn wgt_word_addr<M: u32>(st: Engine<M>) -> u32 {
    cfg_w_base(st) + st.nt * cfg_kgroups(st) + st.kb * (u32:8 / cfg_r(st)) + st.wg_index
}
fn param_addr<M: u32>(st: Engine<M>) -> u32 { cfg_p_base(st) + st.nt * u32:10 + st.param_index }

fn at_fill<M: u32>(st: Engine<M>, atile: s8[128], addr: u32, word: u64) -> s8[128] {
    let fill = act_info(st).fill;
    let byte_sel = addr % u32:8;
    for (i, values): (u32, s8[128]) in u32:0..u32:8 {
        if i < fill {
            update(values, ((st.bank ^ u1:1) as u32) * u32:64 + st.ld_p * u32:8 + st.ld_r + i,
                   byte_at(word, byte_sel + i) as s8)
        } else {
            values
        }
    }(atile)
}

fn act_advance<M: u32>(st: Engine<M>, tile: s8[1024], atile: s8[128]) -> Engine<M> {
    let info = act_info(st);
    let channel = flags_channel(st);
    let rem_blk = cfg_k(st) - st.kb * u32:8;
    let blk = if rem_blk > u32:8 { u32:8 } else { rem_blk };
    let next_r = st.ld_r + info.fill;
    let r_limit = if channel { blk } else { u32:8 };
    let (np, nr, done) = if next_r >= r_limit {
        if st.ld_p == u32:7 { (st.ld_p, st.ld_r, true) } else { (st.ld_p + u32:1, u32:0, false) }
    } else {
        (st.ld_p, next_r, false)
    };
    Engine {
        act_tile: atile, ch_tile: tile, ld_p: np, ld_r: nr,
        phase: if done { u8:8 } else { u8:7 },
        bank: if done { st.bank ^ u1:1 } else { st.bank }, ..st
    }
}

// -----------------------------------------------------------------------
// Shared phase bodies. `engine_step` and `engine_step_mem` run one FSM; only
// the memory access timing differs (the array engine reads its images
// directly, the memory-mapped engine issues a request one phase earlier and
// applies the response here). Everything else is shared so the two cannot
// drift apart.
// -----------------------------------------------------------------------

fn phase_start<M: u32>(st: Engine<M>, start: bool, start_desc: u32) -> Engine<M> {
    if start {
        Engine { phase: u8:1, desc_start: start_desc, desc_index: u32:0, err: u1:0, ..st }
    } else {
        st
    }
}

fn phase_tiles<M: u32>(st: Engine<M>) -> Engine<M> {
    Engine {
        n_tiles: std::ceil_div(cfg_n(st), cfg_o(st)),
        m_tiles: std::ceil_div(cfg_m(st), u32:8),
        phase: u8:4, kb: u32:0, ..st
    }
}

fn phase_k_guard<M: u32>(st: Engine<M>) -> Engine<M> {
    // A task with k == 0 (a deconv phase where no tap lands on that sub-grid)
    // has nothing to accumulate, but its output positions must still be
    // written -- their value is `requant(0)`, which is only zero for a trivial
    // bias/zero-point.
    if cfg_k(st) == u32:0 {
        Engine { acc: zero!<s32[64]>(), param_index: u32:0, phase: u8:10, ..st }
    } else {
        Engine { acc: zero!<s32[64]>(), kb: u32:0, phase: u8:5, ..st }
    }
}

fn phase_clear<M: u32>(st: Engine<M>) -> Engine<M> {
    let remaining = cfg_k(st) - st.kb * u32:8;
    let block = if remaining > u32:8 { u32:8 } else { remaining };
    Engine {
        act_tile: zero!<s8[128]>(), ch_tile: zero!<s8[1024]>(), wgt_tile: zero!<u8[128]>(),
        block_groups: std::ceil_div(block, cfg_r(st)),
        wg_index: u32:0, group_index: u32:0,
        ld_p: u32:0, ld_r: u32:0, phase: u8:6, ..st
    }
}

fn phase_mac<M: u32>(st: Engine<M>) -> Engine<M> {
    if st.group_index < st.block_groups {
        let r = cfg_r(st);
        let o = cfg_o(st);
        let dense = for (i, values): (u32, s8[64]) in u32:0..u32:64 {
            let p = (i / r) % u32:8;
            let rr = i % r;
            update(values, i, st.act_tile[(st.bank as u32) * u32:64 + p * u32:8
                                         + st.group_index * r + rr])
        }(zero!<s8[64]>());
        let ch = for (i, values): (u32, s8[64]) in u32:0..u32:64 {
            let p = i / u32:8;
            let within = i % u32:8;
            let oo = within / r;
            let rr = within % r;
            update(values, i, st.ch_tile[(st.bank as u32) * u32:512 + p * u32:64
                                        + (st.group_index * r + rr) * u32:8 + oo])
        }(zero!<s8[64]>());
        let wgt_arr = for (i, values): (u32, s8[8]) in u32:0..u32:8 {
            update(values, i, st.wgt_tile[(st.bank as u32) * u32:64 + st.group_index * u32:8 + i] as s8)
        }(zero!<s8[8]>());
        let reduced = mac64::mac64_reduced(flags_channel(st), dense, ch, wgt_arr, flags_logr(st));
        let acc = for (i, values): (u32, s32[64]) in u32:0..u32:64 {
            if (i % u32:8) < o && (st.mt * u32:8 + i / u32:8) < cfg_m(st) {
                update(values, i, st.acc[i] + reduced[i])
            } else {
                values
            }
        }(st.acc);
        if st.group_index + u32:1 < st.block_groups {
            Engine { acc, group_index: st.group_index + u32:1, ..st }
        } else if (st.kb + u32:1) * u32:8 < cfg_k(st) {
            Engine { acc, kb: st.kb + u32:1, phase: u8:5, ..st }
        } else {
            Engine { acc, param_index: u32:0, phase: u8:10, ..st }
        }
    } else {
        Engine { err: u1:1, phase: u8:13, ..st }
    }
}

fn phase_param<M: u32>(st: Engine<M>, word: u64) -> Engine<M> {
    let params = update(st.params, st.param_index, word);
    if st.param_index == u32:9 {
        Engine { params, phase: u8:11, out_p: u32:0, ..st }
    } else {
        Engine { params, param_index: st.param_index + u32:1, phase: u8:10, ..st }
    }
}

/// Requantise the current output row and pick its destination. Returns the
/// next state plus the pending write (destination word, packed qword);
/// `write` is false past the last row of a partial tile, where nothing must
/// be stored.
fn phase_emit<M: u32>(st: Engine<M>) -> (Engine<M>, bool, u32, u64) {
    let p = st.out_p;
    let q = requant_lanes(st, p);
    let valid_o = cfg_valid_o_tile(st);
    // One full 64-bit word per tile; lanes past the last real channel are
    // written as zero so the word can be stored without byte enables.
    let qword = for (i, w): (u32, u64) in u32:0..u32:8 {
        if i < valid_o { w | ((q[i] as u8 as u64) << (i * u32:8)) } else { w }
    }(u64:0);
    let write = st.mt * u32:8 + p < cfg_m(st);
    let next = if p < u32:7 {
        Engine { out_p: p + u32:1, ..st }
    } else if st.nt + u32:1 < st.n_tiles {
        Engine { nt: st.nt + u32:1, phase: u8:4, ..st }
    } else if st.mt + u32:1 < st.m_tiles {
        Engine { mt: st.mt + u32:1, nt: u32:0, phase: u8:4, ..st }
    } else {
        Engine { phase: u8:13, ..st }
    };
    (next, write, out_word(st, p), qword)
}

fn phase_next_desc<M: u32>(st: Engine<M>) -> (Engine<M>, bool) {
    let next = half16(cw(st, u32:15), u32:0) as u32;
    if next != u32:0 {
        (Engine { phase: u8:1, desc_start: next, desc_index: u32:0, ..st }, false)
    } else {
        (Engine { phase: u8:14, ..st }, true)
    }
}

/// Array-based engine: `st.act`/`st.wgt` are the full images and every access
/// is a plain read, so each phase does its own loads. `engine_step_mem` is
/// the same FSM with each load turned into a request/response pair.
pub fn engine_step<M: u32>(st: Engine<M>, start: bool, start_desc: u32) -> (Engine<M>, bool) {
    match st.phase {
        u8:0 => (phase_start(st, start, start_desc), false),
        u8:1 => {
            let cfg = update(st.cfg, st.desc_index, st.wgt[cfg_addr(st)]);
            if st.desc_index == u32:15 {
                (Engine { cfg, phase: u8:2, tap_index: u32:0, ..st }, false)
            } else {
                (Engine { cfg, desc_index: st.desc_index + u32:1, ..st }, false)
            }
        },
        u8:2 => {
            if st.tap_index >= cfg_taps(st) {
                (Engine { phase: u8:3, mt: u32:0, nt: u32:0, ..st }, false)
            } else {
                let word = st.wgt[tap_addr(st)];
                let taps = update(st.taps, st.tap_index, byte_at(word, st.tap_index % u32:8));
                (Engine { taps, tap_index: st.tap_index + u32:1, ..st }, false)
            }
        },
        u8:3 => (phase_tiles(st), false),
        u8:4 => (phase_k_guard(st), false),
        u8:5 => (phase_clear(st), false),
        u8:6 => {
            if st.wg_index < st.block_groups {
                let word = st.wgt[wgt_word_addr(st)];
                let tile = for (i, values): (u32, u8[128]) in u32:0..u32:8 {
                    update(values, ((st.bank ^ u1:1) as u32) * u32:64 + st.wg_index * u32:8 + i,
                           byte_at(word, i))
                }(st.wgt_tile);
                (Engine { wgt_tile: tile, wg_index: st.wg_index + u32:1, ..st }, false)
            } else {
                (Engine { phase: u8:7, ld_p: u32:0, ld_r: u32:0, ..st }, false)
            }
        },
        u8:7 => {
            let info = act_info(st);
            let word_index = info.addr / u32:8;
            let safe_index = if word_index < M { word_index } else { u32:0 };
            let next_index = word_index + u32:1;
            let safe_next = if next_index < M { next_index } else { u32:0 };
            let lo_word = if info.in_bounds { st.act[safe_index] } else { u64:0 };
            let hi_word = if info.in_bounds { st.act[safe_next] } else { u64:0 };
            let ch_tile = if flags_channel(st) {
                ch_fill(st, ch_fill(st, st.ch_tile, info.addr, lo_word, u32:0),
                        info.addr, hi_word, u32:1)
            } else {
                st.ch_tile
            };
            let act_tile = if flags_channel(st) {
                st.act_tile
            } else {
                at_fill(st, st.act_tile, info.addr, lo_word)
            };
            (act_advance(st, ch_tile, act_tile), false)
        },
        u8:8 => (phase_mac(st), false),
        u8:10 => (phase_param(st, st.wgt[param_addr(st)]), false),
        u8:11 => {
            let (next, write, addr, qword) = phase_emit(st);
            (Engine {
                out_mem: if write { update(st.out_mem, addr, qword) } else { st.out_mem },
                ..next
            }, false)
        },
        u8:13 => phase_next_desc(st),
        _ => (st, false),
    }
}

pub struct MemIn {
    act_data: u64,
    wgt_data: u64,
    out_data: u64,
}

pub struct MemOut {
    act_re: bool,
    act_raddr: u32,
    wgt_re: bool,
    wgt_raddr: u32,
    out_re: bool,
    out_raddr: u32,
    out_we: bool,
    out_waddr: u32,
    out_wdata: u64,
}

pub fn idle_mem() -> MemOut {
    MemOut {
        act_re: false, act_raddr: u32:0,
        wgt_re: false, wgt_raddr: u32:0,
        out_re: false, out_raddr: u32:0,
        out_we: false, out_waddr: u32:0, out_wdata: u64:0,
    }
}

fn rd_wgt(addr: u32) -> MemOut { MemOut { wgt_re: true, wgt_raddr: addr, ..idle_mem() } }
fn rd_act(addr: u32) -> MemOut { MemOut { act_re: true, act_raddr: addr, ..idle_mem() } }

fn wr_out(addr: u32, data: u64) -> MemOut {
    MemOut { out_we: true, out_waddr: addr, out_wdata: data, ..idle_mem() }
}

/// Memory-mapped variant of `engine_step`: identical FSM, but each memory
/// access is a request that returns its data on the following cycle (`mem`).
/// Phase layout: issue phases 1/2/6/7/10/48, matching wait phases
/// 33/34/38/37/47/42. Phase 48 exists so that the *second* half of a crossing
/// activation read is issued from an issue phase: phase 37 already consumes a
/// response, and issuing a request from a wait phase would both be dropped by
/// the proc and violate the RAM's request/response scheduling constraint.
pub fn engine_step_mem<M: u32>(
    st: Engine<M>, start: bool, start_desc: u32, mem: MemIn
) -> (Engine<M>, bool, MemOut) {
    match st.phase {
        u8:0 => (phase_start(st, start, start_desc), false, idle_mem()),
        u8:1 => (Engine { phase: u8:33, ..st }, false, rd_wgt(cfg_addr(st))),
        u8:33 => {
            let cfg = update(st.cfg, st.desc_index, mem.wgt_data);
            if st.desc_index == u32:15 {
                (Engine { cfg, phase: u8:2, tap_index: u32:0, ..st }, false, idle_mem())
            } else {
                (Engine { cfg, desc_index: st.desc_index + u32:1, phase: u8:1, ..st },
                 false, idle_mem())
            }
        },
        u8:2 => {
            if st.tap_index >= cfg_taps(st) {
                (Engine { phase: u8:3, mt: u32:0, nt: u32:0, ..st }, false, idle_mem())
            } else {
                (Engine { phase: u8:34, ..st }, false, rd_wgt(tap_addr(st)))
            }
        },
        u8:34 => {
            let taps = update(st.taps, st.tap_index, byte_at(mem.wgt_data, st.tap_index % u32:8));
            (Engine { taps, tap_index: st.tap_index + u32:1, phase: u8:2, ..st },
             false, idle_mem())
        },
        u8:3 => (phase_tiles(st), false, idle_mem()),
        u8:4 => (phase_k_guard(st), false, idle_mem()),
        u8:5 => (phase_clear(st), false, idle_mem()),
        u8:6 => {
            if st.wg_index < st.block_groups {
                (Engine { phase: u8:38, ..st }, false, rd_wgt(wgt_word_addr(st)))
            } else {
                (Engine { phase: u8:7, ld_p: u32:0, ld_r: u32:0, ..st }, false, idle_mem())
            }
        },
        u8:38 => {
            let tile = for (i, values): (u32, u8[128]) in u32:0..u32:8 {
                update(values, ((st.bank ^ u1:1) as u32) * u32:64 + st.wg_index * u32:8 + i,
                       byte_at(mem.wgt_data, i))
            }(st.wgt_tile);
            (Engine { wgt_tile: tile, wg_index: st.wg_index + u32:1, phase: u8:6, ..st },
             false, idle_mem())
        },
        u8:7 => (Engine { phase: u8:37, ..st }, false, rd_act(act_info(st).addr / u32:8)),
        u8:37 => {
            let info = act_info(st);
            let channel = flags_channel(st);
            let lo = if info.in_bounds { mem.act_data } else { u64:0 };
            let tile = if channel { ch_fill(st, st.ch_tile, info.addr, lo, u32:0) } else { st.ch_tile };
            let atile = if channel { st.act_tile } else { at_fill(st, st.act_tile, info.addr, lo) };
            let crossing = channel && info.in_bounds
                           && (info.byte_sel + (info.valid_o - u32:1) / cfg_co_group(st)) >= u32:8;
            if crossing {
                (Engine { act_tile: atile, ch_tile: tile, phase: u8:48, ..st }, false, idle_mem())
            } else {
                (act_advance(st, tile, atile), false, idle_mem())
            }
        },
        u8:48 => (Engine { phase: u8:47, ..st }, false, rd_act(act_info(st).addr / u32:8 + u32:1)),
        u8:47 => {
            let info = act_info(st);
            let hi = if info.in_bounds { mem.act_data } else { u64:0 };
            let tile = ch_fill(st, st.ch_tile, info.addr, hi, u32:1);
            (act_advance(st, tile, st.act_tile), false, idle_mem())
        },
        u8:8 => (phase_mac(st), false, idle_mem()),
        u8:10 => (Engine { phase: u8:42, ..st }, false, rd_wgt(param_addr(st))),
        u8:42 => (phase_param(st, mem.wgt_data), false, idle_mem()),
        u8:11 => {
            let (next, write, addr, qword) = phase_emit(st);
            (next, false, if write { wr_out(addr, qword) } else { idle_mem() })
        },
        u8:13 => {
            let (next, done) = phase_next_desc(st);
            (next, done, idle_mem())
        },
        _ => (st, false, idle_mem()),
    }
}

/// 1RW RAM request packet: (addr, wr_data, wr_mask, rd_mask, we, re).
///
/// The element order is what `--ram_configurations` expects; codegen expands it
/// into `<ram>_addr / _wr_data / _wr_mask / _rd_mask / _we / _re` ports with no
/// ready/valid (a skid buffer absorbs the handshake), so an SRAM macro can be
/// wired straight onto them.
/// Phases in which the engine consumes a RAM response / write completion.
fn is_wait_phase(phase: u8) -> bool {
    phase == u8:33 || phase == u8:34 || phase == u8:37
    || phase == u8:38 || phase == u8:42 || phase == u8:47
}

/// Wait phases that consume the *activation* response; the rest read weights.
fn wait_uses_act(phase: u8) -> bool {
    phase == u8:37 || phase == u8:47
}

fn ram_read(addr: u32) -> (u32, u64, u8, u8, bool, bool) {
    (addr, u64:0, u8:0, u8:0, false, true)
}

fn ram_write(addr: u32, data: u64) -> (u32, u64, u8, u8, bool, bool) {
    (addr, data, u8:255, u8:0, true, false)
}

/// Memory-channel engine: act/wgt/out are external 1RW RAMs behind the XLS RAM
/// interface (`--ram_configurations`). The write-completion channels are inputs
/// (the RAM tells us when a write retires) and are drained here.
pub proc ConvAccMem<M: u32> {
    st: Engine<M>,
    start: chan<u32> in,
    done: chan<bool> out,
    act_req: chan<(u32, u64, u8, u8, bool, bool)> out,
    act_resp: chan<(u64,)> in,
    act_wc: chan<()> in,
    wgt_req: chan<(u32, u64, u8, u8, bool, bool)> out,
    wgt_resp: chan<(u64,)> in,
    wgt_wc: chan<()> in,
    out_req: chan<(u32, u64, u8, u8, bool, bool)> out,
    out_resp: chan<(u64,)> in,
    out_wc: chan<()> in,
}

impl ConvAccMem<M> {
    pub fn new(
        start: chan<u32> in,
        done: chan<bool> out,
        act_req: chan<(u32, u64, u8, u8, bool, bool)> out,
        act_resp: chan<(u64,)> in,
        act_wc: chan<()> in,
        wgt_req: chan<(u32, u64, u8, u8, bool, bool)> out,
        wgt_resp: chan<(u64,)> in,
        wgt_wc: chan<()> in,
        out_req: chan<(u32, u64, u8, u8, bool, bool)> out,
        out_resp: chan<(u64,)> in,
        out_wc: chan<()> in,
    ) -> Self {
        Self {
            st: zero_engine<M>(),
            start, done,
            act_req, act_resp, act_wc,
            wgt_req, wgt_resp, wgt_wc,
            out_req, out_resp, out_wc,
        }
    }

    fn next(self) {
        let st0 = read(self.st);
        // Issue phases (1/2/6/7/10/11) and wait phases (33/34/37/38/42/47) are
        // disjoint, so the request and its response never land in the same
        // activation. The RAM model needs exactly that: latency 1 from request
        // to response, which the scheduler can only prove with the two paths
        // spelled out separately.
        if is_wait_phase(st0.phase) {
            // Write completions are drained opportunistically: the read-only
            // RAMs never produce one, so this must not block.
            let (tok, _, _) = recv_non_blocking(join(), self.act_wc, ());
            let (tok, _, _) = recv_non_blocking(tok, self.wgt_wc, ());
            let (tok, _, _) = recv_non_blocking(tok, self.out_wc, ());
            // Only the response this phase actually consumes is taken, and it is
            // mandatory: if it has not landed yet the FSM holds its phase and
            // retries next cycle. Sampling a late response as *another* phase's
            // operand -- or dropping a response on the channel we are not
            // looking at -- is what made single-channel (co=1 / n=1) tiles come
            // out a few counts off. There is at most one request in flight, so
            // the response always belongs to this phase.
            let (tok, nst, ndone) = if wait_uses_act(st0.phase) {
                let (tok, act_r, act_v) = recv_non_blocking(tok, self.act_resp, (u64:0,));
                if act_v {
                    let (nst, ndone, _req) = engine_step_mem(
                        st0, false, u32:0,
                        MemIn { act_data: act_r.0, wgt_data: u64:0, out_data: u64:0 });
                    (tok, nst, ndone)
                } else {
                    (tok, st0, false)
                }
            } else {
                let (tok, wgt_r, wgt_v) = recv_non_blocking(tok, self.wgt_resp, (u64:0,));
                if wgt_v {
                    let (nst, ndone, _req) = engine_step_mem(
                        st0, false, u32:0,
                        MemIn { act_data: u64:0, wgt_data: wgt_r.0, out_data: u64:0 });
                    (tok, nst, ndone)
                } else {
                    (tok, st0, false)
                }
            };
            send_if(tok, self.done, ndone, true);
            write(self.st, nst);
        } else {
            let (tok, start_desc, has_start) = recv_non_blocking(join(), self.start, u32:0);
            let (st, finished, req) = engine_step_mem(
                st0, has_start && st0.phase == u8:0, start_desc,
                MemIn { act_data: u64:0, wgt_data: u64:0, out_data: u64:0 });
            let tok = send_if(tok, self.act_req, req.act_re, ram_read(req.act_raddr));
            let tok = send_if(tok, self.wgt_req, req.wgt_re, ram_read(req.wgt_raddr));
            let tok = send_if(tok, self.out_req, req.out_we,
                              ram_write(req.out_waddr, req.out_wdata));
            send_if(tok, self.done, finished, true);
            write(self.st, st);
        }
    }
}

pub proc ConvAccMemTop {
}

impl ConvAccMemTop {
    pub fn new(
        start: chan<u32> in,
        done: chan<bool> out,
        act_req: chan<(u32, u64, u8, u8, bool, bool)> out,
        act_resp: chan<(u64,)> in,
        act_wc: chan<()> in,
        wgt_req: chan<(u32, u64, u8, u8, bool, bool)> out,
        wgt_resp: chan<(u64,)> in,
        wgt_wc: chan<()> in,
        out_req: chan<(u32, u64, u8, u8, bool, bool)> out,
        out_resp: chan<(u64,)> in,
        out_wc: chan<()> in,
    ) -> Self {
        ConvAccMem<u32:16384>::new(
            start, done, act_req, act_resp, act_wc,
            wgt_req, wgt_resp, wgt_wc, out_req, out_resp, out_wc,
        ).spawn();
        Self {}
    }

    fn next(self) {}
}

#[test]
fn apply_load_test() {
    let st = zero_engine<u32:16>();
    let next = apply_load(st, Load { target: u2:1, addr: u32:3, data: u64:42, strb: u8:255 });
    assert_eq(next.wgt[u32:3], u64:42);
    assert_eq(next.wgt[u32:4], u64:0);
    assert_eq(next.act[u32:3], u64:0);
}

#[test]
fn apply_load_names_every_array_test() {
    // target is a 2-bit field: 0 = activation, 1 = weight, and 2/3 both fold
    // into the output memory (the `else` arm). Only target 1 was covered
    // before, so a swapped 0/1 mapping would have gone unnoticed until cocotb.
    let st = zero_engine<u32:16>();
    let act = apply_load(st, Load { target: u2:0, addr: u32:2, data: u64:0xAABB, strb: u8:255 });
    assert_eq(act.act[u32:2], u64:0xAABB);
    assert_eq(act.wgt[u32:2], u64:0);
    assert_eq(act.out_mem[u32:2], u64:0);
    let out2 = apply_load(st, Load { target: u2:2, addr: u32:5, data: u64:0xCC, strb: u8:255 });
    assert_eq(out2.out_mem[u32:5], u64:0xCC);
    assert_eq(out2.act[u32:5], u64:0);
    let out3 = apply_load(st, Load { target: u2:3, addr: u32:6, data: u64:0xDD, strb: u8:255 });
    assert_eq(out3.out_mem[u32:6], u64:0xDD);
    assert_eq(out3.wgt[u32:6], u64:0);
}

#[test]
fn apply_load_past_the_end_is_dropped_test() {
    // M-1 is the last legal word; anything further must be ignored outright
    // (not clamped, not wrapped) so a bad loader address cannot corrupt word 0.
    let st = zero_engine<u32:16>();
    let last = apply_load(st, Load { target: u2:0, addr: u32:15, data: u64:0x11, strb: u8:255 });
    assert_eq(last.act[u32:15], u64:0x11);
    let past = apply_load(st, Load { target: u2:0, addr: u32:16, data: u64:0x22, strb: u8:255 });
    assert_eq(past.act, st.act);
    assert_eq(past.wgt, st.wgt);
}

// --- the `err` invariant guard ---

#[test]
fn phase_clear_with_no_reduction_left_opens_no_group_test() {
    // kb*8 == k means the reduction is already finished, so the clear step must
    // hand phase_mac an empty block instead of dividing by a stale remainder.
    let st = Engine<u32:16> {
        cfg: update(zero!<u64[16]>(), u32:2, u64:8),
        kb: u32:1, ..zero_engine<u32:16>()
    };
    let nst = phase_clear(st);
    assert_eq(nst.block_groups, u32:0);
    assert_eq(nst.phase, u8:6);
}

#[test]
fn phase_mac_on_an_empty_block_latches_err_test() {
    // The guard is what keeps a bad (kb, k) pair from reading the tiles: the
    // engine latches err and parks in the terminal phase rather than computing.
    let st = Engine<u32:16> { block_groups: u32:0, group_index: u32:0, ..zero_engine<u32:16>() };
    let nst = phase_mac(st);
    assert_eq(nst.err, u1:1);
    assert_eq(nst.phase, u8:13);
    // The accumulator must not be half-updated on the way out.
    assert_eq(nst.acc, zero!<s32[64]>());
}

#[test]
fn phase_mac_on_a_real_block_keeps_err_clear_test() {
    let st = Engine<u32:16> { block_groups: u32:1, group_index: u32:0, ..zero_engine<u32:16>() };
    let nst = phase_mac(st);
    assert_eq(nst.err, u1:0);
    assert_eq(nst.phase, u8:10);
}


pub proc ConvAcc<M: u32> {
    st: Engine<M>,
    load: chan<Load> in,
    start: chan<u32> in,
    out_req: chan<u32> in,
    out_resp: chan<u64> out,
    done: chan<bool> out,
}

impl ConvAcc<M> {
    pub fn new(
        load: chan<Load> in,
        start: chan<u32> in,
        out_req: chan<u32> in,
        out_resp: chan<u64> out,
        done: chan<bool> out,
    ) -> Self {
        Self { st: zero_engine<M>(), load, start, out_req, out_resp, done }
    }

    fn next(self) {
        let st0 = read(self.st);
        let (tok, request, has_load) = recv_non_blocking(join(), self.load,
            Load { target: u2:0, addr: u32:0, data: u64:0, strb: u8:0 });
        let st1 = if has_load { apply_load(st0, request) } else { st0 };
        let (tok, start_desc, has_start) = recv_non_blocking(tok, self.start, u32:0);
        let (tok, read_addr, has_read) = recv_non_blocking(tok, self.out_req, u32:0);
        let (st2, finished) = engine_step(st1, has_start && st1.phase == u8:0, start_desc);
        let read_data = if read_addr < M { st2.out_mem[read_addr] } else { u64:0 };
        let tok = send_if(tok, self.out_resp, has_read, read_data);
        send_if(tok, self.done, finished, true);
        write(self.st, st2);
    }
}

pub proc ConvAccSmall {
}

impl ConvAccSmall {
    pub fn new(
        load: chan<Load> in,
        start: chan<u32> in,
        out_req: chan<u32> in,
        out_resp: chan<u64> out,
        done: chan<bool> out,
    ) -> Self {
        ConvAcc<u32:1024>::new(load, start, out_req, out_resp, done).spawn();
        Self {}
    }

    fn next(self) {}
}

pub proc ConvAccTop {
}

impl ConvAccTop {
    pub fn new(
        load: chan<Load> in,
        start: chan<u32> in,
        out_req: chan<u32> in,
        out_resp: chan<u64> out,
        done: chan<bool> out,
    ) -> Self {
        ConvAcc<u32:16384>::new(load, start, out_req, out_resp, done).spawn();
        Self {}
    }

    fn next(self) {}
}

// ===========================================================================
// Unit tests for the engine internals. The array-based `engine_step` runs the
// whole FSM as a pure function over `st.wgt`/`st.act`/`st.out_mem`, so these
// cover the datapath without any memory latency or simulator.
// ===========================================================================

// --- test-only cfg packing helpers ---
// The descriptor packs several fields per 64-bit word as 16-bit halves or
// 32-bit halves; these make the tests readable instead of hex soup.

fn lohi(lo: u32, hi: u32) -> u64 { (lo as u64) | ((hi as u64) << u32:32) }

fn h2(a: u32, b: u32) -> u64 { (a as u64) | ((b as u64) << u32:16) }

fn h4(a: u32, b: u32, c: u32, d: u32) -> u64 {
    (a as u64) | ((b as u64) << u32:16) | ((c as u64) << u32:32) | ((d as u64) << u32:48)
}

// --- bitfield helpers ---

#[test]
fn byte_at_and_half16_test() {
    let word = u64:0x0807060504030201;
    assert_eq(byte_at(word, u32:0), u8:1);
    assert_eq(byte_at(word, u32:3), u8:4);
    assert_eq(byte_at(word, u32:7), u8:8);
    assert_eq(half16(word, u32:0), u16:0x0201);
    assert_eq(half16(word, u32:1), u16:0x0403);
    assert_eq(half16(word, u32:3), u16:0x0807);
}

#[test]
fn lo32_hi32_test() {
    let word = u64:0xAABBCCDD11223344;
    assert_eq(lo32(word), u32:0x11223344);
    assert_eq(hi32(word), u32:0xAABBCCDD);
}

#[test]
fn merge_bytes_test() {
    let old = u64:0x0807060504030201;
    // full strobe replaces everything
    assert_eq(merge_bytes(old, u64:0xFF, u8:255), u64:0xFF);
    // no strobe keeps the old word
    assert_eq(merge_bytes(old, u64:0xFF, u8:0), old);
    // a partial strobe swaps only the named bytes
    assert_eq(merge_bytes(old, u64:0x000000000000BBAA, u8:0b00000011),
              u64:0x080706050403BBAA);
    // strobes may name bytes anywhere in the word; the byte is taken from the
    // *same* lane of `data`, so data must carry it at that position.
    assert_eq(merge_bytes(old, u64:0x00CC000000000000, u8:0b01000000),
              u64:0x08CC060504030201);
}

// --- configuration accessors ---

fn pack_cfg_for_readback() -> u64[16] {
    update(update(update(update(update(update(update(update(update(update(update(update(
        zero!<u64[16]>(),
        u32:0, u64:31),                                   // gemm|channel|deconv + log_r=3 (bits 3-4)
        u32:1, lohi(u32:33, u32:44)),                    // m, n
        u32:2, u64:55),                                  // k
        u32:3, h2(u32:2, u32:3)),                        // mh, mw
        u32:4, u64:66),                                  // kgroups
        u32:5, h4(u32:7, u32:8, u32:9, u32:10)),         // ci_count, ci_base, co_base, co_group
        u32:6, h2(u32:11, u32:12)),                      // hi, wi
        u32:7, h2(u32:13, u32:14)),                      // ho, wo
        u32:8, h2(u32:15, u32:16)),                      // ci, co
        u32:9, h4(u32:17, u32:18, u32:19, u32:20)),      // sh, sw, dh, dw
        u32:10, h4(u32:21, u32:22, u32:23, u32:24)),     // ph, pw, phase_h, phase_w
        u32:11, h2(u32:25, u32:26))                      // steph, stepw
}

#[test]
fn cfg_accessors_read_back_test() {
    let st = Engine<u32:16> {
        cfg: update(update(pack_cfg_for_readback(), u32:12, lohi(u32:27, u32:28)),
                    u32:14, lohi(u32:30, u32:32)),
        ..zero_engine<u32:16>()
    };
    assert_eq(flags_gemm(st), true);
    assert_eq(flags_channel(st), true);
    assert_eq(flags_deconv(st), true);
    assert_eq(flags_logr(st), u2:3);
    assert_eq(cfg_r(st), u32:8);
    assert_eq(cfg_o(st), u32:1);
    assert_eq(cfg_m(st), u32:33);
    assert_eq(cfg_n(st), u32:44);
    assert_eq(cfg_k(st), u32:55);
    assert_eq(cfg_mh(st), u32:2);
    assert_eq(cfg_mw(st), u32:3);
    assert_eq(cfg_kgroups(st), u32:66);
    assert_eq(cfg_ci_count(st), u32:7);
    assert_eq(cfg_ci_base(st), u32:8);
    assert_eq(cfg_co_base(st), u32:9);
    assert_eq(cfg_co_group(st), u32:10);
    assert_eq(cfg_hi(st), u32:11);
    assert_eq(cfg_wi(st), u32:12);
    assert_eq(cfg_ho(st), u32:13);
    assert_eq(cfg_wo(st), u32:14);
    assert_eq(cfg_ci(st), u32:15);
    assert_eq(cfg_co(st), u32:16);
    assert_eq(cfg_sh(st), u32:17);
    assert_eq(cfg_sw(st), u32:18);
    assert_eq(cfg_dh(st), u32:19);
    assert_eq(cfg_dw(st), u32:20);
    assert_eq(cfg_ph(st), u32:21);
    assert_eq(cfg_pw(st), u32:22);
    assert_eq(cfg_phase_h(st), u32:23);
    assert_eq(cfg_phase_w(st), u32:24);
    assert_eq(cfg_steph(st), u32:25);
    assert_eq(cfg_stepw(st), u32:26);
    assert_eq(cfg_act_base(st), u32:27);
    assert_eq(cfg_out_base(st), u32:28);
    assert_eq(cfg_tap_base(st), u32:30);
    assert_eq(cfg_taps(st), u32:32);
}

#[test]
fn flags_are_independent_bits_test() {
    let base = zero_engine<u32:16>();
    let gemm_only = Engine<u32:16> { cfg: update(base.cfg, u32:0, u64:1), ..base };
    assert_eq(flags_gemm(gemm_only), true);
    assert_eq(flags_channel(gemm_only), false);
    assert_eq(flags_deconv(gemm_only), false);
    assert_eq(flags_logr(gemm_only), u2:0);
    assert_eq(cfg_r(gemm_only), u32:1);
    assert_eq(cfg_o(gemm_only), u32:8);
}

// --- output address layout ---
// This is the mapping that the 8-byte-aligned image depends on: one 64-bit
// word per (position, group, n-tile). A regression here shows up as output
// bytes landing in the wrong word, which is exactly the failure mode of the
// earlier depthwise bug.

fn out_cfg(flags: u32, m: u32, n: u32, mh: u32, mw: u32, co: u32,
           co_base: u32, co_group: u32, ho: u32, wo: u32,
           phase_h: u32, phase_w: u32, steph: u32, stepw: u32,
           out_base: u32) -> u64[16] {
    update(update(update(update(update(update(update(update(update(
        zero!<u64[16]>(),
        u32:0, flags as u64),
        u32:1, lohi(m, n)),
        u32:3, h2(mh, mw)),
        u32:5, h4(u32:1, u32:0, co_base, co_group)),
        u32:7, h2(ho, wo)),
        u32:8, h2(u32:1, co)),
        u32:10, h4(u32:0, u32:0, phase_h, phase_w)),
        u32:11, h2(steph, stepw)),
        u32:12, lohi(u32:0, out_base))
}

fn out_st(cfg: u64[16], mt: u32, nt: u32, n_tiles: u32) -> Engine<u32:16> {
    Engine<u32:16> { cfg, mt, nt, n_tiles, ..zero_engine<u32:16>() }
}

#[test]
fn out_word_gemm_is_row_major_test() {
    // GEMM: pos is just the row, stride is n_tiles, no group offset.
    let cfg = out_cfg(u32:1, 20, 4, 20, 1, 4, 0, 1, 1, 1, 0, 0, 1, 1, u32:0);
    let st = out_st(cfg, u32:1, u32:1, u32:2);
    // row 8, tile 1 -> 0 + 8*2 + 1 = 17
    assert_eq(out_word(st, u32:0), u32:17);
    // p advances within the 8-row tile
    assert_eq(out_word(st, u32:3), u32:23);
}

#[test]
fn out_word_respects_out_base_test() {
    // out_base is a byte address and must be divided by the 8-byte word size.
    let cfg = out_cfg(u32:1, 20, 4, 20, 1, 4, 0, 1, 1, 1, 0, 0, 1, 1, u32:64);
    let st = out_st(cfg, u32:1, u32:1, u32:2);
    assert_eq(out_word(st, u32:0), u32:25);   // 8 + 8*2 + 1
}

#[test]
fn out_word_conv_is_batch_row_major_test() {
    // Dense conv: pos = (batch*ho + oh)*wo + ow with the row/column taken from
    // the position inside the 8-row tile.
    let cfg = out_cfg(u32:0, 12, 8, 2, 3, 8, 0, 8, 2, 3, 0, 0, 1, 1, u32:0);
    let st = out_st(cfg, u32:0, u32:0, u32:1);
    // m_abs = 5 -> batch 0, rem 5 -> oh = 5/3 = 1, ow = 5%3 = 2 -> pos = 5
    assert_eq(out_word(st, u32:5), u32:5);
}

#[test]
fn out_word_conv_second_batch_offsets_by_whole_batch_test() {
    let cfg = out_cfg(u32:0, 12, 8, 2, 3, 8, 0, 8, 2, 3, 0, 0, 1, 1, u32:0);
    let st = out_st(cfg, u32:1, u32:0, u32:1);
    // m_abs = 8 -> batch 1, rem 2 -> oh = 0, ow = 2 -> pos = (1*2+0)*3+2 = 8
    assert_eq(out_word(st, u32:0), u32:8);
}

#[test]
fn out_word_conv_grouped_offsets_by_group_test() {
    // Two groups of 4 channels. Group 1 sits n_tiles words after group 0 within
    // each position, and the whole position stride is 2*n_tiles words.
    let cfg = out_cfg(u32:0, 12, 8, 2, 3, 8, 4, 4, 2, 3, 0, 0, 1, 1, u32:0);
    let st = out_st(cfg, u32:0, u32:1, u32:2);
    // out_group = 4/4 = 1, out_stride_tiles = (8/4)*2 = 4
    // pos 0 -> 0*4 + 1*2 + 1 = 3
    assert_eq(out_word(st, u32:0), u32:3);
    // pos 1 (m_abs = 1) -> 1*4 + 3 = 7
    assert_eq(out_word(st, u32:1), u32:7);
}

#[test]
fn out_word_channelwise_has_no_group_offset_test() {
    // Channelwise folds every group into the tile index, so both the group
    // offset and the widened position stride disappear.
    let cfg = out_cfg(u32:2, 12, 6, 2, 3, 6, 0, 3, 2, 3, 0, 0, 1, 1, u32:0);
    let st = out_st(cfg, u32:0, u32:1, u32:3);
    // pos 0 -> 0*3 + 1 = 1
    assert_eq(out_word(st, u32:0), u32:1);
    // pos 1 -> 1*3 + 1 = 4
    assert_eq(out_word(st, u32:1), u32:4);
}

#[test]
fn out_word_deconv_phase_scatters_over_the_output_grid_test() {
    // A transposed conv phase owns one residue class of the output grid:
    // oh = phase_h + row*steph, ow = phase_w + col*stepw.
    let cfg = out_cfg(u32:4, 8, 8, 2, 2, 8, 0, 8, 4, 4, 1, 1, 2, 2, u32:0);
    let st = out_st(cfg, u32:0, u32:0, u32:1);
    // m_abs = 1 -> rem 1 -> oh = 1 + (1/2)*2 = 1, ow = 1 + (1%2)*2 = 3
    // pos = (0*4 + 1)*4 + 3 = 7
    assert_eq(out_word(st, u32:1), u32:7);
    // m_abs = 3 -> rem 3 -> oh = 1 + 1*2 = 3, ow = 1 + 1*2 = 3 -> pos = 15
    assert_eq(out_word(st, u32:3), u32:15);
}

#[test]
fn out_word_tile_lanes_share_a_word_test() {
    // Moving `p` steps by a whole position (n_tiles words); moving `nt` steps
    // by exactly one word. This is what makes the layout byte-strobe free.
    let cfg = out_cfg(u32:0, 24, 8, 2, 3, 8, 0, 8, 2, 3, 0, 0, 1, 1, u32:0);
    let a = out_st(cfg, u32:0, u32:0, u32:2);
    let b = out_st(cfg, u32:0, u32:1, u32:2);
    assert_eq(out_word(b, u32:0) - out_word(a, u32:0), u32:1);
    assert_eq(out_word(a, u32:1) - out_word(a, u32:0), u32:2);
}

// --- FSM entry and the k == 0 fast path ---

#[test]
fn engine_step_start_latches_the_descriptor_pointer_test() {
    let st = zero_engine<u32:16>();
    let (nst, done) = engine_step(st, true, u32:42);
    assert_eq(nst.phase, u8:1);
    assert_eq(nst.desc_start, u32:42);
    assert_eq(nst.desc_index, u32:0);
    assert_eq(done, false);
    // without `start` the engine stays idle
    let (idle, _) = engine_step(st, false, u32:42);
    assert_eq(idle.phase, u8:0);
}

#[test]
fn engine_step_mem_k0_skips_the_load_loop_test() {
    // k == 0 means the phase contributes nothing to the accumulator. It must
    // jump straight to the parameter load instead of walking the tile loader
    // (and must not emit any memory traffic).
    let st = Engine<u32:16> {
        cfg: update(zero!<u64[16]>(), u32:2, u64:0),
        phase: u8:4, ..zero_engine<u32:16>()
    };
    let (nst, done, req) = engine_step_mem(
        st, false, u32:0, MemIn { act_data: u64:0, wgt_data: u64:0, out_data: u64:0 });
    assert_eq(nst.phase, u8:10);
    assert_eq(nst.param_index, u32:0);
    assert_eq(done, false);
    assert_eq(req.act_re, false);
    assert_eq(req.wgt_re, false);
    assert_eq(req.out_we, false);
}

fn gemm_k0_image() -> u64[32] {
    // A one-row, one-output GEMM with k == 0: the accumulator stays zero and
    // the engine must still emit requant(0) so the output word is not left
    // untouched. Parameters live at word 16 and ask for +7 on every lane.
    let desc = update(update(update(update(update(update(update(update(update(update(
        zero!<u64[32]>(),
        u32:0, u64:1),                       // flags: gemm
        u32:1, lohi(u32:1, u32:1)),          // m = 1, n = 1
        u32:2, u64:0),                       // k = 0
        u32:3, h2(u32:1, u32:1)),            // mh = 1, mw = 1
        u32:5, h4(u32:1, u32:0, u32:0, u32:1)),
        u32:7, h2(u32:1, u32:1)),            // ho = 1, wo = 1
        u32:11, h2(u32:1, u32:1)),           // steph = 1, stepw = 1
        u32:12, lohi(u32:0, u32:0)),         // act_base = 0, out_base = 0
        u32:13, lohi(u32:0, u32:16)),        // w_base = 0, p_base = 16
        u32:14, u64:0);                      // tap_base = 0, taps = 0
    // params[4..7] = scale = 1 for all eight lanes
    let scaled = for (i, w): (u32, u64[32]) in u32:20..u32:24 {
        update(w, i, lohi(u32:1, u32:1))
    }(desc);
    // params[9] = zero point 7 for all eight lanes
    update(scaled, u32:25, u64:0x0707070707070707)
}

#[test]
fn engine_step_k0_still_writes_requant_of_zero_test() {
    let st = Engine<u32:32> {
        wgt: gemm_k0_image(),
        desc_start: u32:0, ..zero_engine<u32:32>()
    };
    let (final_st, done) = for (i, (st, done)): (u32, (Engine<u32:32>, bool))
            in u32:0..u32:200 {
        if done { (st, done) } else { engine_step(st, i == u32:0, u32:0) }
    }((st, false));
    assert_eq(done, true);
    assert_eq(final_st.err, u1:0);
    // requant_one(0, bias=0, scale=1, shift=0, zp=7) == 7, packed into lane 0.
    assert_eq(final_st.out_mem[u32:0], u64:7);
    // only the one real channel is written; the padding lanes stay zero
    assert_eq(final_st.out_mem[u32:1], u64:0);
}

// --- FSM-level properties ---------------------------------------------------
// Every property below this point used to constrain pure address arithmetic
// (out_word / merge_bytes). These drive the real FSM to completion instead:
// with all-zero memories the accumulator cannot move, so the only correct
// output is requant(0) == zero_point. That makes the expected word analytic
// while still walking the tile loader, the kb loop, MAC, the parameter load and
// the emit phase for k > 0 -- none of which any unit test reached.

/// A one-row, one-column GEMM descriptor `k` deep with reduction width `log_r`.
fn qc_gemm_image(k: u32, log_r: u2) -> u64[32] {
    let desc = update(update(update(update(update(update(update(update(update(update(
        zero!<u64[32]>(),
        u32:0, u64:1 | ((log_r as u64) << u32:3)),   // flags: gemm + log_r
        u32:1, lohi(u32:1, u32:1)),                  // m = 1, n = 1
        u32:2, k as u64),                            // k
        u32:3, h2(u32:1, u32:1)),                    // mh = 1, mw = 1
        u32:5, h4(u32:1, u32:0, u32:0, u32:1)),
        u32:7, h2(u32:1, u32:1)),                    // ho = 1, wo = 1
        u32:11, h2(u32:1, u32:1)),                   // steph = 1, stepw = 1
        u32:12, lohi(u32:0, u32:0)),                 // act_base = 0, out_base = 0
        u32:13, lohi(u32:0, u32:16)),                // w_base = 0, p_base = 16
        u32:14, u64:0);                              // tap_base = 0, taps = 0
    // params[4..7] = scale 1 on every lane, params[9] = zero point 7.
    let scaled = for (i, w): (u32, u64[32]) in u32:20..u32:24 {
        update(w, i, lohi(u32:1, u32:1))
    }(desc);
    update(scaled, u32:25, u64:0x0707070707070707)
}

/// Run that descriptor to completion. Returns
/// (steps taken, number of `done` pulses, err flag, output word 0).
fn qc_run_gemm(k: u32, log_r: u2) -> (u32, u32, u1, u64) {
    let st0 = Engine<u32:32> {
        wgt: qc_gemm_image(k, log_r), desc_start: u32:0, ..zero_engine<u32:32>()
    };
    let (st, steps, pulses, _) = for (i, (st, steps, pulses, done)):
            (u32, (Engine<u32:32>, u32, u32, bool)) in u32:0..u32:2048 {
        if done { (st, steps, pulses, done) } else {
            let (nst, ndone) = engine_step(st, i == u32:0, u32:0);
            let bump = if ndone { u32:1 } else { u32:0 };
            (nst, steps + u32:1, pulses + bump, ndone)
        }
    }((st0, u32:0, u32:0, false));
    (steps, pulses, st.err, st.out_mem[u32:0])
}

/// The descriptor declares a single output column, so exactly one lane is a
/// real channel and the other seven are padding: a correct word is
/// requant(0) == zero_point (7) on lane 0 and zero everywhere else, whatever
/// the reduction width. `word == 7` therefore also asserts the padding is zero.
fn qc_gemm_zero_word() -> u64 { u64:7 }

fn gemm_fsm_case(k: u32, log_r: u2) {
    let (steps, pulses, err, word) = qc_run_gemm(k, log_r);
    assert_eq(err, u1:0);
    // A second `done` pulse would be consumed as a new task by the wrapper, so
    // exactly one per run is part of the contract, not an implementation detail.
    assert_eq(pulses, u32:1);
    // The run must advance and must not need the whole loop bound to finish;
    // a stuck engine would leave `done` false and the pulse count at zero.
    assert_eq(steps > u32:0 && steps < u32:2048, true);
    assert_eq(word, qc_gemm_zero_word());
}

#[test]
fn gemm_fsm_runs_k_and_logr_test() {
    // These were written as `#[quickcheck]` properties first, but the XLS JIT
    // aborts on an engine-sized value ("Null alloc with 1024") once the check
    // loops over a full FSM state, so they are a deterministic sweep instead:
    // same two axes, every run enforced by the default test target.
    // k = 0 takes the fast path; 8/16/24 walk one, two and three kb blocks.
    gemm_fsm_case(u32:0, u2:0);
    gemm_fsm_case(u32:8, u2:0);
    gemm_fsm_case(u32:16, u2:0);
    gemm_fsm_case(u32:24, u2:0);
    // Reducing over the narrower widths changes r/o and the emitted lane mask.
    gemm_fsm_case(u32:8, u2:1);
    gemm_fsm_case(u32:8, u2:2);
    gemm_fsm_case(u32:8, u2:3);
    gemm_fsm_case(u32:24, u2:3);
}

// ===========================================================================
// Fuzzed invariants (`#[quickcheck]`). Run with
//   dslx_interpreter_main --evaluator=ir-interpreter \
//       --run_quickcheck_when_interpreting convacc_proc.x
// `#[quickcheck]` draws 1000 random inputs; `#[quickcheck(exhaustive)]` walks
// the whole input domain.
// ===========================================================================

#[quickcheck]
fn tile_geometry_divisors_are_never_zero_check(log_r: u2) -> bool {
    // `std::ceil_div` traps on a zero divisor, so every divisor the tile
    // geometry feeds it must be provably non-zero. Both come from log_r:
    // cfg_o = 8 >> log_r is 8/4/2/1 and cfg_r = 1 << log_r is 1/2/4/8.
    let st = Engine<u32:16> {
        cfg: update(zero!<u64[16]>(), u32:0, (log_r as u64) << u32:3),
        ..zero_engine<u32:16>()
    };
    cfg_o(st) != u32:0 && cfg_r(st) != u32:0 && u32:8 != u32:0
}

#[quickcheck]
fn merge_bytes_full_strobe_replaces_word_check(old: u64, data: u64) -> bool {
    merge_bytes(old, data, u8:255) == data
}

#[quickcheck]
fn merge_bytes_empty_strobe_keeps_word_check(old: u64, data: u64) -> bool {
    merge_bytes(old, data, u8:0) == old
}

#[quickcheck]
fn merge_bytes_writes_exactly_the_named_lanes_check(old: u64, data: u64, strb: u8) -> bool {
    // Every byte either comes from `data` (when the strobe names it) or from
    // `old` (when it does not); nothing else can happen.
    for (i, ok): (u32, bool) in u32:0..u32:8 {
        let got = byte_at(merge_bytes(old, data, strb), i);
        let want = if ((strb >> i) as u1) == u1:1 { byte_at(data, i) } else { byte_at(old, i) };
        ok && got == want
    }(true)
}

// --- output address layout ---

/// Build a configuration for one of the five addressing modes.
///   0 = gemm       1 = dense conv     2 = grouped conv
///   3 = channelwise                   4 = deconv
/// `group` selects which output-channel group the task owns.
fn qc_cfg(mode: u3, group: u2) -> u64[16] {
    let g = group as u32;
    if mode == u3:0 {
        // gemm: pos is the row, no group offset
        out_cfg(u32:1, 64, 4, 64, 1, 4, 0, 1, 1, 1, 0, 0, 1, 1, u32:0)
    } else if mode == u3:1 {
        // dense conv, one group
        out_cfg(u32:0, 48, 8, 2, 3, 8, 0, 8, 2, 3, 0, 0, 1, 1, u32:0)
    } else if mode == u3:2 {
        // grouped conv: 2 groups of 4 channels
        out_cfg(u32:0, 48, 8, 2, 3, 8, g * u32:4, 4, 2, 3, 0, 0, 1, 1, u32:0)
    } else if mode == u3:3 {
        // channelwise: co_per_group = 3, so o must be >= 3 for the plan to be legal
        out_cfg(u32:2, 48, 6, 2, 3, 6, 0, 3, 2, 3, 0, 0, 1, 1, u32:0)
    } else {
        // deconv: one residue class of the output grid
        out_cfg(u32:4, 48, 8, 2, 2, 8, 0, 8, 4, 4, 1, 1, 2, 2, u32:0)
    }
}

fn qc_st(mode: u3, group: u2, mt: u2, nt: u2) -> Engine<u32:16> {
    Engine<u32:16> {
        cfg: qc_cfg(mode, group),
        mt: mt as u32, nt: nt as u32, n_tiles: u32:4,
        ..zero_engine<u32:16>()
    }
}

/// Normalise the fuzzed mode selector: only 0..4 name a real addressing mode.
fn qc_mode(mode: u3) -> u3 { if mode > u3:4 { u3:4 } else { mode } }

#[quickcheck]
fn out_word_distinct_tiles_share_no_word_check
    (mode: u3, mt: u2, p: u3, nt: u2) -> bool {
    // Within one position, moving to the next n-tile must move to the very
    // next word. This is what lets the image store a tile in a single 64-bit
    // word without byte enables.
    if nt == u2:3 {
        true
    } else {
        let md = qc_mode(mode);
        let st = qc_st(md, u2:0, mt, nt);
        let next = qc_st(md, u2:0, mt, nt + u2:1);
        out_word(next, p as u32) == out_word(st, p as u32) + u32:1
    }
}

#[quickcheck]
fn out_word_is_injective_over_its_index_check
    (mode: u3, a_mt: u2, a_p: u3, a_nt: u2, b_mt: u2, b_p: u3, b_nt: u2) -> bool {
    // Two different (tile-row, position, n-tile) tuples must never map to the
    // same word; a collision means one write silently clobbers another. The
    // group axis is covered by the grouped-conv property below, because gemm
    // and channelwise fold every group into the single task.
    if a_mt == b_mt && a_p == b_p && a_nt == b_nt {
        true
    } else {
        let md = qc_mode(mode);
        out_word(qc_st(md, u2:0, a_mt, a_nt), a_p as u32)
            != out_word(qc_st(md, u2:0, b_mt, b_nt), b_p as u32)
    }
}

#[quickcheck]
fn out_word_groups_do_not_collide_check
    (a_group: u1, a_p: u3, b_group: u1, b_p: u3) -> bool {
    // Grouped conv gives each task its own co_base. Distinct groups of the
    // same position must land in distinct words -- this is exactly the
    // invariant the old depthwise bug violated. The geometry here is two
    // groups of four channels over eight, so the group index is one bit;
    // `qc_cfg` would otherwise accept an out-of-range co_base and the layout
    // is only required to be injective over in-range indices.
    let md = u3:2;
    let a_g = a_group as u2;
    let b_g = b_group as u2;
    if a_g == b_g && a_p == b_p {
        true
    } else {
        out_word(qc_st(md, a_g, u2:0, u2:0), a_p as u32)
            != out_word(qc_st(md, b_g, u2:0, u2:0), b_p as u32)
    }
}

#[quickcheck]
fn out_word_advances_by_a_whole_position_check
    (mode: u3, mt: u2, p: u3, nt: u2) -> bool {
    // Stepping `p` by one steps the address by exactly one position, and each
    // position owns `out_stride_tiles` words (n_tiles for gemm and channelwise,
    // (co/co_group)*n_tiles for grouped conv). A transposed conv scatters its
    // positions over the output grid, so only monotonicity holds there.
    if p == u3:7 {
        true
    } else {
        let md = qc_mode(mode);
        let st = qc_st(md, u2:0, mt, nt);
        let step = out_word(st, (p as u32) + u32:1) - out_word(st, p as u32);
        if md == u3:4 {
            step > u32:0
        } else {
            step == out_stride_tiles(st)
        }
    }
}
