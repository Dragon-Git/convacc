#!/usr/bin/env python3
"""Generate a DSLX comparison test that drives convacc_proc::engine_step on a
real regression case's SRAM images (activation.bin / weight.bin) and asserts the
computed out_mem equals the numpy reference (expected.bin).

This isolates whether a DUT mismatch is in:
  - the DSLX algorithm itself  (engine_step output != numpy)  -> fix hw/*.x
  - the codegen / SV           (engine_step output == numpy)  -> fix hw/*.sv
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = 16384  # 128KB SRAM, matches convacc_proc::Engine<u32:16384> / DUT


def load_case(name: str):
    d = os.path.join(ROOT, "build", "cocotb_cases", name)
    act = open(os.path.join(d, "activation.bin"), "rb").read()
    wgt = open(os.path.join(d, "weight.bin"), "rb").read()
    exp = open(os.path.join(d, "expected.bin"), "rb").read()
    assert len(act) == WORDS * 8 and len(wgt) == WORDS * 8, (len(act), len(wgt))
    return act, wgt, exp


def to_u64(b: bytes) -> list[int]:
    b = bytes(b)
    if len(b) % 8:
        b = b + bytes(8 - len(b) % 8)
    return list(struct.unpack("<%dQ" % (len(b) // 8), b))


def decode_desc(wgt_words: list[int]) -> dict:
    w = wgt_words
    flags = w[0]
    m, n = w[1] & 0xFFFFFFFF, (w[1] >> 32) & 0xFFFFFFFF
    k = w[2] & 0xFFFFFFFF
    mh, mw = w[3] & 0xFFFF, (w[3] >> 16) & 0xFFFF
    k_groups = w[4] & 0xFFFFFFFF
    o = (w[4] >> 32) & 0xFFFFFFFF
    taps = (w[14] >> 32) & 0xFFFF
    out_base = (w[12] >> 32) & 0xFFFFFFFF
    act_base = w[12] & 0xFFFFFFFF
    log_r = (flags >> 3) & 0x3
    r = 1 << log_r
    n_tiles = (n + o - 1) // o if o else 0
    m_tiles = (m + 7) // 8
    return dict(m=m, n=n, k=k, o=o, taps=taps, r=r, n_tiles=n_tiles,
                m_tiles=m_tiles, out_base=out_base, act_base=act_base,
                k_groups=k_groups)


def loop_bound(desc: dict) -> int:
    block_groups = max(1, (8 + desc["r"] - 1) // desc["r"])
    per_tile = 3 + block_groups + 16 + block_groups + 10 + 8  # phases 3..12 rough
    total = desc["taps"] + desc["m_tiles"] * desc["n_tiles"] * (per_tile + 2 * block_groups)
    return max(512, total * 2 + 32)


def fmt_arr(words: list[int]) -> str:
    # compact: split into rows of 16 for readability, but keep as single literal
    nums = ",".join(str(v) for v in words)
    return nums


def main() -> None:
    cases = sys.argv[1:] or ["grid_deconv_0000"]
    blocks = []
    for name in cases:
        act, wgt, exp = load_case(name)
        act_w = to_u64(act)
        wgt_w = to_u64(wgt)
        desc = decode_desc(wgt_w)
        exp_w = to_u64(exp)
        exp_full = [0] * WORDS
        for i, v in enumerate(exp_w):
            exp_full[desc["out_base"] + i] = v
        N = loop_bound(desc)
        test = f"""
#[test]
fn cmp_{name}() {{
    let act = u64[{WORDS}]:[{fmt_arr(act_w)}];
    let wgt = u64[{WORDS}]:[{fmt_arr(wgt_w)}];
    let exp = u64[{WORDS}]:[{fmt_arr(exp_full)}];
    let computed = run_conv(act, wgt);
    assert_eq(computed, exp);
}}"""
        blocks.append((name, N, test))

    header = """#![feature(explicit_state_access)]
import convacc_proc;

fn run_conv(act: u64[16384], wgt: u64[16384]) -> u64[16384] {
    let st0 = convacc_proc::Engine { act, wgt, ..convacc_proc::zero_engine<u32:16384>() };
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<16384>, bool)) in u32:0..u32:LOOPN {
        if done {{
            (st, done)
        }} else {{
            let (nst, ndone) = convacc_proc::engine_step(st, i == u32:0, u32:0);
            (nst, ndone)
        }}
    }((st0, false));
    st.out_mem
}
"""
    # use max loop bound across cases
    maxN = max(b[1] for b in blocks)
    header = header.replace("LOOPN", str(maxN))
    body = header + "\n".join(b[2] for b in blocks)
    out_path = os.path.join(ROOT, "hw", "dslx_cmp.x")
    with open(out_path, "w") as f:
        f.write(body)
    print(f"wrote {out_path} with {len(blocks)} case(s); loop bound N={maxN}")
    for name, N, _ in blocks:
        print(f"  {name}: out_base={decode_desc(to_u64(load_case(name)[1]))['out_base']} N={N}")


if __name__ == "__main__":
    main()
