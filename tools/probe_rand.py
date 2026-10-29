#!/usr/bin/env python3
"""Emit a DSLX probe for one randomized case (from gen_dslx_rand) and dump
selected Engine fields at the final phase to reveal where nt-tile handling
goes wrong."""
from __future__ import annotations

import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.gen_dslx_rand as R  # noqa: E402

M = 256


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "rnd24_dense"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261029
    rng = random.Random(seed)
    chosen = None
    idx = 0
    while chosen is None and idx < n * 6:
        c = R.rand_case(rng, idx)
        idx += 1
        if c and c[0] == target:
            chosen = c
    if chosen is None:
        raise SystemExit(f"case {target} not found")
    name, act, wgt, exp = chosen
    exp_bytes = exp.astype(np.int8).tobytes()

    def consts():
        return """fn act_const() -> u64[%d] { u64[%d]:[%s] }
fn wgt_const() -> u64[%d] { u64[%d]:[%s] }
""" % (M, M, ",".join(map(str, R._words_trunc(act, M))),
       M, M, ",".join(map(str, R._words_trunc(wgt, M))))

    header = """#![feature(explicit_state_access)]
import convacc_proc;

fn run_to_phase(act: u64[%d], wgt: u64[%d], ph: u8) -> convacc_proc::Engine<%d> {
    let st0 = convacc_proc::Engine { act, wgt, ..convacc_proc::zero_engine<u32:%d>() };
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<%d>, bool)) in u32:0..u32:%d {
        if st.phase >= ph { (st, done) } else {
            let (nst, ndone) = convacc_proc::engine_step(st, i == u32:0, u32:0);
            (nst, ndone)
        }
    }((st0, false));
    st
}
""" % (M, M, M, M, M, R.MAX_STEPS)

    header += """fn run_to_phase8_nt1(act: u64[%d], wgt: u64[%d]) -> convacc_proc::Engine<%d> {
    let st0 = convacc_proc::Engine { act, wgt, ..convacc_proc::zero_engine<u32:%d>() };
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<%d>, bool)) in u32:0..u32:%d {
        if st.phase == u8:8 && st.nt == u32:1 { (st, done) } else {
            let (nst, ndone) = convacc_proc::engine_step(st, i == u32:0, u32:0);
            (nst, ndone)
        }
    }((st0, false));
    st
}
""" % (M, M, M, M, M, R.MAX_STEPS)

    probes = [
        ("acc0", "acc", "s32", 0),
        ("acc1", "acc", "s32", 1),
        ("nt_final", "nt", "u32", None),
        ("out1", "out_mem", "u64", 1),
        ("out0", "out_mem", "u64", 0),
    ]
    body = ""
    for pname, field, ty, idx2 in probes:
        lhs = f"st.{field}[{idx2}]" if idx2 is not None else f"st.{field}"
        body += """#[test]
fn probe_%s() {
    let st = run_to_phase(act_const(), wgt_const(), u8:14);
    assert_eq(%s, %s:123456);
}
""" % (pname, lhs, ty)
    header += """fn run_to_phase11_nt1(act: u64[%d], wgt: u64[%d]) -> convacc_proc::Engine<%d> {
    let st0 = convacc_proc::Engine { act, wgt, ..convacc_proc::zero_engine<u32:%d>() };
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<%d>, bool)) in u32:0..u32:%d {
        if st.phase == u8:11 && st.nt == u32:1 { (st, done) } else {
            let (nst, ndone) = convacc_proc::engine_step(st, i == u32:0, u32:0);
            (nst, ndone)
        }
    }((st0, false));
    st
}
""" % (M, M, M, M, M, R.MAX_STEPS)
    body += """#[test]
fn probe_nt1_acc_before_write() {
    let st = run_to_phase11_nt1(act_const(), wgt_const());
    assert_eq(st.acc[0], s32:1);
}
#[test]
fn probe_nt1_acc1_before_write() {
    let st = run_to_phase11_nt1(act_const(), wgt_const());
    assert_eq(st.acc[1], s32:1);
}
"""
    for pname, field, ty, idx2 in [("nt1_acc0", "acc", "s32", 0), ("nt1_at", "act_tile", "s8", 0),
                                   ("nt1_wg", "wgt_tile", "u8", 0), ("nt1_bank", "bank", "u1", None), ("nt1_wg2", "wgt_tile", "u8", 2)]:
        lhs = f"st.{field}[{idx2}]" if idx2 is not None else f"st.{field}"
        body += """#[test]
fn probe_%s() {
    let st = run_to_phase8_nt1(act_const(), wgt_const());
    assert_eq(%s, %s:1);
}
""" % (pname, lhs, ty)
    out = os.path.join(ROOT, "hw", "dslx_probe_rand.x")
    with open(out, "w") as f:
        f.write(header + consts() + body)
    print("wrote", out, "expected word0/1:",
          int.from_bytes(exp_bytes[:8], "little"), int.from_bytes(exp_bytes[8:16], "little"))


if __name__ == "__main__":
    main()
