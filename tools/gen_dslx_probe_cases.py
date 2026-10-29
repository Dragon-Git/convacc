#!/usr/bin/env python3
"""Drive convacc_proc::engine_step_mem with *named* regression cases.

`gen_dslx_equiv.py` only generates random `ConvSpec` cases with trivial requant
parameters (scale=1, shift=0, zp=0, bias=0) and never a `GemmSpec`, so its green
run says nothing about the GEMM path or the real requantiser. This tool closes
that hole: it takes case names from tests/test_cocotb_convacc.py, builds the
same SRAM image and expected output, and asserts the DSLX engine reproduces it.

Usage:
    tools/gen_dslx_probe_cases.py NAME [NAME ...] [M]
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tests", "cocotb"))

from convacc.image_builder import make_image  # noqa: E402
import tools.gen_dslx_rand as R  # noqa: E402

STEPS = 20000


def _words(b, n):
    return R._words_trunc(b, n)


def main() -> int:
    names = [a for a in sys.argv[1:] if not a.isdigit()]
    M = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 1024
    if not names:
        print("usage: gen_dslx_probe_cases.py NAME [NAME ...] [M]")
        return 2

    import test_cocotb_convacc as T

    cases = {c.name: c for c in T.build_cases()}
    body = ""
    picked = []
    for name in names:
        if name not in cases:
            print(f"  no such case: {name}")
            continue
        c = cases[name]
        _, image, expected = make_image(
            c.spec, c.weights, c.layout, c.activations,
            c.bias, c.scale, c.shift, c.zero_point)

        def used_words(buf):
            b = bytes(buf)
            hits = [i for i in range(len(b) // 8 - 1, -1, -1)
                    if b[i * 8:(i + 1) * 8] != b"\x00" * 8]
            return (hits[0] + 1) if hits else 0

        need = max(used_words(image.activation), used_words(image.weight))
        if need > M:
            print(f"  skip {name}: needs {need} words > {M}")
            continue
        picked.append((name, image.activation, image.weight, expected))
        body += """#[test]
fn probe_%s() {
    let act = u64[%d]:[%s];
    let wgt = u64[%d]:[%s];
    let exp = u64[%d]:[%s];
    assert_eq(run_mem(act, wgt), exp);
}
""" % (name.replace("-", "_"), M, ",".join(map(str, _words(image.activation, M))),
       M, ",".join(map(str, _words(image.weight, M))),
       M, ",".join(map(str, _words(expected.astype(np.int8).tobytes(), M))))
        print(f"  queued {name}: {need} words, expected {expected.size} bytes")

    if not picked:
        print("no usable cases")
        return 1

    header = """#![feature(explicit_state_access)]
import convacc_proc;

fn run_mem(act: u64[%(M)d], wgt: u64[%(M)d]) -> u64[%(M)d] {
    let st0 = convacc_proc::Engine<%(M)d> { ..convacc_proc::zero_engine<u32:%(M)d>() };
    let mem0 = convacc_proc::MemIn { act_data: u64:0, wgt_data: u64:0, out_data: u64:0 };
    let (_, _, outp, _) = for (i, (st, mem_in, outp, done)):
            (u32, (convacc_proc::Engine<%(M)d>, convacc_proc::MemIn, u64[%(M)d], bool))
            in u32:0..u32:%(S)d {
        if done { (st, mem_in, outp, done) } else {
            let (nst, ndone, req) = convacc_proc::engine_step_mem(st, i == u32:0, u32:0, mem_in);
            let ai = if req.act_raddr < u32:%(M)d { req.act_raddr } else { u32:0 };
            let wi = if req.wgt_raddr < u32:%(M)d { req.wgt_raddr } else { u32:0 };
            let oi = if req.out_raddr < u32:%(M)d { req.out_raddr } else { u32:0 };
            let wa = if req.out_waddr < u32:%(M)d { req.out_waddr } else { u32:0 };
            let noutp = if req.out_we {
                update(outp, wa, req.out_wdata)
            } else {
                outp
            };
            let nxt = convacc_proc::MemIn {
                act_data: act[ai], wgt_data: wgt[wi], out_data: outp[oi],
            };
            (nst, nxt, noutp, ndone)
        }
    }((st0, mem0, zero!<u64[%(M)d]>(), false));
    outp
}
""" % {"M": M, "S": STEPS}

    out = os.path.join(ROOT, "hw", "dslx_probe.x")
    with open(out, "w") as f:
        f.write(header + body)
    print(f"wrote {out} with {len(picked)} cases (M={M})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
