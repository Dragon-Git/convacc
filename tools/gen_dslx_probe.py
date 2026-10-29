#!/usr/bin/env python3
"""Probe intermediate Engine state by stopping engine_step at a chosen phase.

Emits hw/dslx_probe.x with one #[test] per probed element, each asserting the
value we EXPECT from the numpy reference layout. The interpreter prints
"computed vs expected" for the first mismatch in each test, which gives full
visibility into ch_tile / wgt_tile / taps contents.

Usage:  .venv/bin/python tools/gen_dslx_probe.py cw_3x3
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from convacc.image_builder import make_image  # noqa: E402
from convacc.spec import ConvSpec  # noqa: E402

M = 256


def _words_trunc(b, n):
    b = bytes(b)[: n * 8]
    if len(b) % 8:
        b = b + bytes(8 - len(b) % 8)
    out = list(struct.unpack("<%dQ" % (len(b) // 8), b))
    out += [0] * (n - len(out))
    return out


def case_cw_3x3():
    spec = ConvSpec(n=1, hi=3, wi=3, ci=3, co=3, ho=3, wo=3, kh=3, kw=3,
                    sh=1, sw=1, dh=1, dw=1, ph=1, pw=1, groups=3, kind="conv2d")
    return spec, np.ones((3, 3, 3), np.int8), "HWM", np.ones((1, 3, 3, 3), np.int8)


def case_cw_k9_r8():
    spec = ConvSpec(n=1, hi=3, wi=3, ci=1, co=1, ho=3, wo=3, kh=3, kw=3,
                    sh=1, sw=1, dh=1, dw=1, ph=1, pw=1, groups=1, kind="conv2d")
    return spec, np.ones((3, 3, 1), np.int8), "HWM", np.ones((1, 3, 3, 1), np.int8)


def decode_descriptor(wgt_bytes):
    w = struct.unpack("<16Q", wgt_bytes[:128])
    flags = w[0]
    return {
        "w": w,
        "flags": flags,
        "log_r": (flags >> 3) & 3,
        "r": 1 << ((flags >> 3) & 3),
        "o": 8 >> ((flags >> 3) & 3),
        "m": w[1] & 0xFFFFFFFF,
        "n": w[1] >> 32,
        "k": w[2] & 0xFFFFFFFF,
        "mh": w[3] & 0xFFFF,
        "mw": w[3] >> 16,
        "kgroups": w[4] & 0xFFFFFFFF,
        "ci": (w[8]) & 0xFFFF,
        "co": (w[8] >> 16) & 0xFFFF,
        "hi": w[6] & 0xFFFF,
        "wi": w[6] >> 16,
        "sh": w[9] & 0xFFFF,
        "sw": (w[9] >> 16) & 0xFFFF,
        "ph": w[10] & 0xFFFF,
        "pw": (w[10] >> 16) & 0xFFFF,
        "act_base": w[12] & 0xFFFFFFFF,
        "w_base": w[13] & 0xFFFFFFFF,
        "tap_base": w[14] & 0xFFFFFFFF,
        "taps": (w[14] >> 32) & 0xFFFF,
    }


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cw_3x3"
    spec, wts, layout, acts = {
        "cw_3x3": case_cw_3x3,
        "cw_k9_r8": case_cw_k9_r8,
    }[which]()
    n = spec.co
    _, image, expected = make_image(spec, wts, layout, acts, np.zeros(n, np.int32),
                                    np.ones(n, np.int32), np.zeros(n, np.uint8),
                                    np.zeros(n, np.int8))
    d = decode_descriptor(image.weight)
    print(f"== {which}: r={d['r']} o={d['o']} k={d['k']} kgroups={d['kgroups']} "
          f"m={d['m']} n={d['n']} ci={d['ci']} co={d['co']}")

    taps = list(image.weight[d["tap_base"] * 8:(d["tap_base"] + d["taps"] + 7) // 8 * 8])
    taps = list(image.weight[d["tap_base"] * 8: d["tap_base"] * 8 + d["taps"]])
    print("taps bytes:", [hex(t) for t in taps])

    # expected ch_tile bank1 content for kb=0 (load bank is bank^1 = 1)
    k = d["k"]
    blk = min(8, k)
    mh, mw = d["mh"], d["mw"]
    ch = np.zeros(1024, np.int8)
    act = np.frombuffer(image.activation, np.int8)
    for p in range(8):
        m_abs = p  # mt=0
        batch = m_abs // (mh * mw)
        rem = m_abs % (mh * mw)
        oh = rem // mw
        ow = rem % mw
        for kstep in range(blk):
            kh, kw = taps[kstep] >> 4, taps[kstep] & 15
            ih = oh * d["sh"] + kh - d["ph"]
            iw = ow * d["sw"] + kw - d["pw"]
            for oo in range(min(8, d["n"])):
                if 0 <= ih < d["hi"] and 0 <= iw < d["wi"]:
                    ci = oo // 1 if d["co"] else 0
                    ci = oo  # co_group == 1 in these cases
                    off = d["act_base"] + ((batch * d["hi"] + ih) * d["wi"] + iw) * d["ci"] + ci
                    ch[512 + p * 64 + kstep * 8 + oo] = act[off]
    print("expected ch_tile nonzero count:", int(np.count_nonzero(ch)))

    # wgt_tile expected: bank1, group g -> 8 bytes of word (w_base + kb*(8/r) + g)
    wgt_words = _words_trunc(image.weight, M)
    wg = np.zeros(128, np.uint8)
    for g in range((blk + d["r"] - 1) // d["r"]):
        widx = d["w_base"] + g  # kb=0
        for j in range(8):
            wg[64 + g * 8 + j] = (wgt_words[widx] >> (8 * j)) & 0xFF
    print("expected wgt_tile nonzero:", int(np.count_nonzero(wg)))

    probes = []
    for p, ks, oo in [(0, 0, 0), (0, 4, 0), (4, 0, 0), (4, 4, 1), (0, 3, 2), (8 - 1, 7, 2)]:
        probes.append((f"ch_p{p}_k{ks}_o{oo}", 512 + p * 64 + ks * 8 + oo, int(ch[512 + p * 64 + ks * 8 + oo])))
    for g, j in [(0, 0), (0, 1), (2, 0), (3, 5)]:
        probes.append((f"wg_g{g}_j{j}", 64 + g * 8 + j, int(wg[64 + g * 8 + j])))
    for t in [0, 1, 8]:
        probes.append((f"tap{t}", t, int(taps[t])))

    # acc after the 4 mac groups of kb=0: acc[p][o] = sum_{g,rr} ch[p,g*2+rr,o] * wg[g,o*2+rr]
    acc_exp = np.zeros((8, 8), np.int32)
    for g in range((blk + d["r"] - 1) // d["r"]):
        for o in range(min(8, d["n"])):
            for rr in range(d["r"]):
                ks = g * d["r"] + rr
                if ks < blk:
                    acc_exp[:, o] += ch[512 + np.arange(8) * 64 + ks * 8 + o].astype(np.int32) * int(
                        wg[64 + g * 8 + o * d["r"] + rr]
                    )
    print("expected acc[p=4][:3]:", acc_exp[4, :3].tolist(), " acc[p=0][:3]:", acc_exp[0, :3].tolist())

    for p, o in [(4, 0), (4, 1), (0, 0), (0, 2)]:
        probes.append((f"acc_p{p}_o{o}", p * 8 + o, int(acc_exp[p, o])))

    header = """#![feature(explicit_state_access)]
import convacc_proc;

fn run_to_phase(act: u64[%(M)d], wgt: u64[%(M)d], ph: u8) -> convacc_proc::Engine<%(M)d> {
    let st0 = convacc_proc::Engine { act, wgt, ..convacc_proc::zero_engine<u32:%(M)d>() };
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<%(M)d>, bool)) in u32:0..u32:400 {
        if st.phase >= ph { (st, done) } else {
            let (nst, ndone) = convacc_proc::engine_step(st, i == u32:0, u32:0);
            (nst, ndone)
        }
    }((st0, false));
    st
}

fn run_to_phase8_mt1(act: u64[%(M)d], wgt: u64[%(M)d]) -> convacc_proc::Engine<%(M)d> {
    let st0 = convacc_proc::Engine { act, wgt, ..convacc_proc::zero_engine<u32:%(M)d>() };
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<%(M)d>, bool)) in u32:0..u32:400 {
        if st.phase == u8:8 && st.mt == u32:1 { (st, done) } else {
            let (nst, ndone) = convacc_proc::engine_step(st, i == u32:0, u32:0);
            (nst, ndone)
        }
    }((st0, false));
    st
}

fn run_phase8_n(act: u64[%(M)d], wgt: u64[%(M)d], n: u32) -> convacc_proc::Engine<%(M)d> {
    let st0 = run_to_phase(act, wgt, u8:8);
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<%(M)d>, bool)) in u32:0..u32:400 {
        if i >= n { (st, done) } else {
            let (nst, ndone) = convacc_proc::engine_step(st, false, u32:0);
            (nst, ndone)
        }
    }((st0, false));
    st
}
""" % {"M": M}
    body = ""
    for name, idx, val in probes:
        body += """#[test]
fn probe_%(n)s() {
    let st = run_to_phase(act_const(), wgt_const(), u8:8);
    assert_eq(st.%(field)s[%(idx)d], s8:%(v)d);
}
""" % {
            "n": name,
            "field": "taps" if name.startswith("tap") else ("wgt_tile" if name.startswith("wg") else "ch_tile"),
            "idx": idx,
            "v": val if not name.startswith("tap") else val,
        }
    # taps are u8; separate emit
    body = ""
    for name, idx, val in probes:
        if name.startswith("acc"):
            body += """#[test]
fn probe_%s() {
    let st = run_phase8_n(act_const(), wgt_const(), u32:4);
    assert_eq(st.acc[%d], s32:%d);
}
""" % (name, idx, val)
        elif name.startswith("tap"):
            body += """#[test]
fn probe_%s() {
    let st = run_to_phase(act_const(), wgt_const(), u8:8);
    assert_eq(st.taps[%d], u8:%d);
}
""" % (name, idx, val)
        elif name.startswith("wg"):
            body += """#[test]
fn probe_%s() {
    let st = run_to_phase(act_const(), wgt_const(), u8:8);
    assert_eq(st.wgt_tile[%d], u8:%d);
}
""" % (name, idx, val)
        else:
            body += """#[test]
fn probe_%s() {
    let st = run_to_phase(act_const(), wgt_const(), u8:8);
    assert_eq(st.ch_tile[%d], s8:%d);
}
""" % (name, idx, val)

    consts = """fn act_const() -> u64[%(M)d] { u64[%(M)d]:[%(a)s] }
fn wgt_const() -> u64[%(M)d] { u64[%(M)d]:[%(w)s] }
""" % {
        "M": M,
        "a": ",".join(map(str, _words_trunc(image.activation, M))),
        "w": ",".join(map(str, _words_trunc(image.weight, M))),
    }

    # ---- extra probes: full-k acc at phase 11, params, mt, final out_mem ----
    wgt_words = _words_trunc(image.weight, M)
    p_base = d["w"][13] >> 32
    params = [wgt_words[p_base + i] for i in range(10)]
    extra = ""
    for fld, ty, idx in [("phase", "u8", None), ("kb", "u32", None), ("block_groups", "u32", None),
                         ("group_index", "u32", None), ("m_tiles", "u32", None),
                         ("n_tiles", "u32", None), ("mt", "u32", None), ("nt", "u32", None),
                         ("out_p", "u32", None), ("param_index", "u32", None)]:
        extra += """#[test]
fn probe_state_%(f)s() {
    let st = run_to_phase(act_const(), wgt_const(), u8:14);
    assert_eq(st.%(f)s, %(t)s:255);
}
""" % {"f": fld, "t": ty}

    # expected full-k accumulator for interior point p=4 (all 9 taps valid)
    full = int(sum(act_bytes for act_bytes in [1] * 9))
    extra += """#[test]
fn probe_full_acc() {
    let st = run_to_phase(act_const(), wgt_const(), u8:11);
    assert_eq(st.acc[32], s32:%d);
}
""" % 9
    for i in [0, 4, 8, 9]:
        extra += """#[test]
fn probe_param%d() {
    let st = run_to_phase(act_const(), wgt_const(), u8:11);
    assert_eq(st.params[%d], u64:%d);
}
""" % (i, i, params[i])
    extra += """#[test]
fn probe_mt_after_run() {
    let st = run_to_phase(act_const(), wgt_const(), u8:14);
    assert_eq(st.mt, u32:%d);
}
""" % ((d["m"] + 7) // 8 - 1)
    exp_w0 = int.from_bytes(expected.astype(np.int8).tobytes()[:8], "little")
    extra += """#[test]
fn probe_out_word0() {
    let st = run_to_phase(act_const(), wgt_const(), u8:14);
    assert_eq(st.out_mem[0], u64:%d);
}
""" % exp_w0
    extra += """#[test]
fn probe_final_acc0() {
    let st = run_to_phase(act_const(), wgt_const(), u8:14);
    assert_eq(st.acc[0], s32:12345);
}
#[test]
fn probe_final_out0() {
    let st = run_to_phase(act_const(), wgt_const(), u8:14);
    assert_eq(st.out_mem[0], u64:999);
}
#[test]
fn probe_final_out1() {
    let st = run_to_phase(act_const(), wgt_const(), u8:14);
    assert_eq(st.out_mem[1], u64:999);
}
"""
    extra += """#[test]
fn probe_mt1_ch() {
    let st = run_to_phase8_mt1(act_const(), wgt_const());
    assert_eq(st.ch_tile[512 + 4 * 8], s8:1);
}
#[test]
fn probe_mt1_bank() {
    let st = run_to_phase8_mt1(act_const(), wgt_const());
    assert_eq(st.bank, u1:1);
}
#[test]
fn probe_mt1_wgt() {
    let st = run_to_phase8_mt1(act_const(), wgt_const());
    assert_eq(st.wgt_tile[64], u8:99);
}
"""
    for idx in [512, 520, 528, 536, 544, 552, 560, 568, 576, 640, 704, 0, 8]:
        extra += """#[test]
fn probe_mt1_dump_%d() {
    let st = run_to_phase8_mt1(act_const(), wgt_const());
    assert_eq(st.ch_tile[%d], s8:127);
}
""" % (idx, idx)
    extra += """#[test]
fn probe_mt1_where() {
    let st = run_to_phase8_mt1(act_const(), wgt_const());
    assert_eq(st.phase, u8:8);
}
#[test]
fn probe_mt1_ldp() {
    let st = run_to_phase8_mt1(act_const(), wgt_const());
    assert_eq(st.ld_p, u32:99);
}
#[test]
fn probe_mt1_ldr() {
    let st = run_to_phase8_mt1(act_const(), wgt_const());
    assert_eq(st.ld_r, u32:99);
}
"""
    extra += """#[test]
fn probe_mt1_nt() {
    let st = run_to_phase8_mt1(act_const(), wgt_const());
    assert_eq(st.nt, u32:99);
}
"""
    for f, t in [("kb","u32"),("block_groups","u32"),("wg_index","u32"),("m_tiles","u32")]:
        extra += """#[test]
fn probe_mt1_%(f)s() {
    let st = run_to_phase8_mt1(act_const(), wgt_const());
    assert_eq(st.%(f)s, %(t)s:99);
}
""" % {"f": f, "t": t}
    body += extra
    out = os.path.join(ROOT, "hw", "dslx_probe.x")
    with open(out, "w") as f:
        f.write(header + consts + body)
    print("wrote", out)


if __name__ == "__main__":
    main()
