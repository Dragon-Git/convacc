#!/usr/bin/env python3
"""Build small, hand-verifiable synthetic cases via the project's real
image_builder/reference, then emit a DSLX test that drives
convacc_proc::engine_step on the produced SRAM images and asserts
out_mem == numpy reference.

Uses a small Engine<M> (M=256) because synthetic cases only touch low addresses,
which makes the DSLX interpreter ~64x faster than M=16384.
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from convacc.image_builder import make_image  # noqa: E402
from convacc.spec import ConvSpec, GemmSpec  # noqa: E402

M = 256  # small engine depth; synthetic data lives in the first ~64 words


def _words_trunc(b: bytes, n: int) -> list[int]:
    b = bytes(b)[: n * 8]
    if len(b) % 8:
        b = b + bytes(8 - len(b) % 8)
    out = list(struct.unpack("<%dQ" % (len(b) // 8), b))
    out += [0] * (n - len(out))
    return out


def emit(case_name: str, act_bytes: bytes, wgt_bytes: bytes, exp: np.ndarray) -> str:
    act_w = _words_trunc(act_bytes, M)
    wgt_w = _words_trunc(wgt_bytes, M)
    exp_b = exp.astype(np.int8).tobytes()
    exp_w = _words_trunc(exp_b, M)
    return """#[test]
fn cmp_%s() {
    let act = u64[%d]:[%s];
    let wgt = u64[%d]:[%s];
    let exp = u64[%d]:[%s];
    let computed = run_conv(act, wgt);
    assert_eq(computed, exp);
}
""" % (case_name, M, ",".join(map(str, act_w)), M, ",".join(map(str, wgt_w)),
         M, ",".join(map(str, exp_w)))


def dump_desc(name, image):
    w = list(struct.unpack("<16Q", image.weight[:128]))
    flags = w[0]
    print(f"  [{name}] flags=0x{flags:x} gemm={flags&1} chan={(flags>>1)&1} deconv={(flags>>2)&1}"
          f" log_r={(flags>>3)&3} m={w[1]&0xFFFFFFFF} n={w[1]>>32} k={w[2]&0xFFFFFFFF}"
          f" mh={w[3]&0xFFFF} mw={w[3]>>16} o={w[4]>>32} r={1<<((flags>>3)&3)}"
          f" hi={w[6]&0xFFFF} wi={w[6]>>16} ho={w[7]&0xFFFF} wo={w[7]>>16} next_desc={w[15]&0xFFFFFFFF}")


def add_case(cases, name, spec, w, layout, a):
    n = spec.n if isinstance(spec, GemmSpec) else spec.co
    _, image, expected = make_image(spec, w, layout, a,
                                    np.zeros(n, np.int32), np.ones(n, np.int32),
                                    np.zeros(n, np.uint8), np.zeros(n, np.int8))
    dump_desc(name, image)
    cases.append((name, image.activation, image.weight, expected))


def main():
    cases = []

    # --- channelwise (depthwise): ci==groups. 1x1, ci=co=1, single output ---
    spec = ConvSpec(n=1, hi=1, wi=1, ci=1, co=1, ho=1, wo=1, kh=1, kw=1,
                    sh=1, sw=1, dh=1, dw=1, ph=0, pw=0, groups=1, kind="conv2d")
    add_case(cases, "cw_1x1", spec, np.full((1, 1, 1, 1), 1, np.int8), "HWIO",
             np.full((1, 1, 1, 1), 1, np.int8))

    # depthwise 3x3, ci=co=3, groups=3, 3x3 spatial (multi position + multi tap)
    spec = ConvSpec(n=1, hi=3, wi=3, ci=3, co=3, ho=3, wo=3, kh=3, kw=3,
                    sh=1, sw=1, dh=1, dw=1, ph=1, pw=1, groups=3, kind="conv2d")
    cw_w = np.ones((3, 3, 3), np.int8)  # HWM: kh,kw,co
    cw_a = np.ones((1, 3, 3, 3), np.int8)
    add_case(cases, "cw_3x3", spec, cw_w, "HWM", cw_a)

    # depthwise 3x3 taps but single channel (k=9 multi-block, r=8,o=1)
    spec = ConvSpec(n=1, hi=3, wi=3, ci=1, co=1, ho=3, wo=3, kh=3, kw=3,
                    sh=1, sw=1, dh=1, dw=1, ph=1, pw=1, groups=1, kind="conv2d")
    add_case(cases, "cw_k9_r8", spec, np.ones((3, 3, 1), np.int8), "HWM",
             np.ones((1, 3, 3, 1), np.int8))

    # --- deconv multi-phase: stride2, pad1, larger -> several (rh,rw) tasks ---
    spec = ConvSpec(n=1, hi=3, wi=3, ci=1, co=2, ho=5, wo=5, kh=3, kw=3,
                    sh=2, sw=2, dh=1, dw=1, ph=1, pw=1, groups=1, kind="deconv2d")
    dw_w = np.ones((3, 3, 1, 2), np.int8)
    dw_a = np.ones((1, 3, 3, 1), np.int8)
    add_case(cases, "deconv_mp", spec, dw_w, "HWIO", dw_a)

    # --- gemm wide n=10 -> n_tiles=2 (nt>0) ---
    spec = GemmSpec(m=2, k=2, n=10)
    add_case(cases, "gemm_wide", spec, np.ones((2, 10), np.int8), "KN",
             np.array([[1, 2], [3, 4]], np.int8))

    header = """#![feature(explicit_state_access)]
import convacc_proc;

fn run_conv(act: u64[%d], wgt: u64[%d]) -> u64[%d] {
    let st0 = convacc_proc::Engine { act, wgt, ..convacc_proc::zero_engine<u32:%d>() };
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<%d>, bool)) in u32:0..u32:4096 {
        if done { (st, done) } else {
            let (nst, ndone) = convacc_proc::engine_step(st, i == u32:0, u32:0);
            (nst, ndone)
        }
    }((st0, false));
    st.out_mem
}
""" % (M, M, M, M, M)

    body = "".join(emit(n, a, w, e) for n, a, w, e in cases)
    with open(os.path.join(ROOT, "hw", "dslx_synth.x"), "w") as f:
        f.write(header + body)
    print(f"\nwrote hw/dslx_synth.x with {len(cases)} case(s), M={M}")


if __name__ == "__main__":
    main()
