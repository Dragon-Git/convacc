#!/usr/bin/env python3
"""Equivalence/mem-engine harness: drives convacc_proc::engine_step_mem with a
DSLX-side memory model (1-cycle latency) and asserts out_mem == numpy expected.

This validates the memory-mapped engine that is used for SV codegen.
"""
from __future__ import annotations

import os
import random
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.gen_dslx_rand as R  # noqa: E402

M = 256
STEPS = 20000


def _words(b, n):
    return R._words_trunc(b, n)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261029
    rng = random.Random(seed)
    cases = []
    idx = 0
    while len(cases) < n and idx < n * 6:
        c = R.rand_case(rng, idx)
        idx += 1
        if c:
            cases.append(c)

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

    body = ""
    for name, act, wgt, exp in cases:
        body += """#[test]
fn mem_%s() {
    let act = u64[%d]:[%s];
    let wgt = u64[%d]:[%s];
    let exp = u64[%d]:[%s];
    assert_eq(run_mem(act, wgt), exp);
}
""" % (name, M, ",".join(map(str, _words(act, M))), M,
       ",".join(map(str, _words(wgt, M))), M,
       ",".join(map(str, _words(exp.astype(np.int8).tobytes(), M))))

    out = os.path.join(ROOT, "hw", "dslx_equiv.x")
    with open(out, "w") as f:
        f.write(header + body)
    print(f"wrote {out} with {len(cases)} cases")


if __name__ == "__main__":
    main()
