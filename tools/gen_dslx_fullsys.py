#!/usr/bin/env python3
"""Full-system DSLX test: the engine proc wired to DSLX RAM models.

`tools/gen_dslx_rand.py` / `gen_dslx_equiv.py` drive the engine as a *pure
function* over arrays, so the RAM handshake layer -- request/response channels,
the write-completion channel, byte strobes -- is never exercised. In the silicon
flow that layer lives in SV (`hw/convacc_sram.sv` + the generated top), so only
cocotb reaches it.

This harness closes that gap in DSLX: `convacc_proc::ConvAccMem` is spawned next
to three RAM behaviour procs and driven through real channels by a `#[test]`
proc, with no SV in the loop at all.

Notes on what is (and is not) modelled:
  * The RAM models mirror `hw/convacc_sram.sv`: byte-strobed write, full-word
    read, write-completion token. Read latency is 0 in the model; the engine
    retries its wait phase until the response lands, so a real (slower) macro
    cannot break it -- it only takes more cycles.
  * Memory is 256 words, not the 16384 of the RTL instance: an array in proc
    state becomes flip-flops in codegen, so this is a *protocol* test, not a
    substitute for the cocotb regression.
  * Selected by `#[test]` on a proc (this toolchain does not accept the
    `#[test_proc]` spelling used by xls/examples/ram.x).

Usage:  tools/gen_dslx_fullsys.py [cases] [seed]
"""
from __future__ import annotations

import os
import random
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gen_dslx_rand as R  # noqa: E402

M = 256
WORDS_CHECKED = 4
CYCLE_LIMIT = 200000
REQ = "(u32, u64, u8, u8, bool, bool)"

HEADER = """#![feature(explicit_state_access)]
import convacc_proc;

// Full-system harness: convacc_proc::ConvAccMem<256> as the DUT, with the three
// tensor memories as DSLX procs. The image is baked per case because proc state
// must be compile-time evaluable.
"""

RAM = """// __LABEL__ RAM model (byte-strobed write, full-word read, write completion).
proc __NAME__ {
    st: u64[__M__],
    req: chan<__REQ__> in,
    resp: chan<(u64,)> out,
    wc: chan<()> out,
}

impl __NAME__ {
    fn new(req: chan<__REQ__> in, resp: chan<(u64,)> out, wc: chan<()> out) -> Self {
        Self { st: u64[__M__]:[__IMAGE__], req, resp, wc }
    }

    fn next(self) {
        let (tok, r) = recv(join(), self.req);
        let (addr, wdata, wmask, _rmask, we, re) = r;
        let cur = read(self.st);
        let idx = if addr < u32:__M__ { addr } else { u32:0 };
        let nxt = if we { update(cur, idx, convacc_proc::merge_bytes(cur[idx], wdata, wmask)) } else { cur };
        write(self.st, nxt);
        let tok = if we { send(tok, self.wc, ()) } else { tok };
        send_if(tok, self.resp, re, (nxt[idx],));
    }
}

"""

OUT_RAM = """// Output memory: same model plus a dump port the test reads results through.
proc __NAME__ {
    st: u64[__M__],
    req: chan<__REQ__> in,
    resp: chan<(u64,)> out,
    wc: chan<()> out,
    dump_req: chan<u32> in,
    dump_resp: chan<u64> out,
}

impl __NAME__ {
    fn new(req: chan<__REQ__> in, resp: chan<(u64,)> out, wc: chan<()> out,
           dump_req: chan<u32> in, dump_resp: chan<u64> out) -> Self {
        Self { st: zero!<u64[__M__]>(), req, resp, wc, dump_req, dump_resp }
    }

    fn next(self) {
        let idle = (u32:0, u64:0, u8:0, u8:0, false, false);
        let (tok, r, has_r) = recv_non_blocking(join(), self.req, idle);
        let (tok, daddr, has_d) = recv_non_blocking(tok, self.dump_req, u32:0);
        let cur = read(self.st);
        let (addr, wdata, wmask, _rmask, we, re) = r;
        let idx = if addr < u32:__M__ { addr } else { u32:0 };
        let nxt = if has_r && we { update(cur, idx, convacc_proc::merge_bytes(cur[idx], wdata, wmask)) } else { cur };
        write(self.st, nxt);
        let tok = if has_r && we { send(tok, self.wc, ()) } else { tok };
        let tok = if has_r && re { send(tok, self.resp, (nxt[idx],)) } else { tok };
        let didx = if daddr < u32:__M__ { daddr } else { u32:0 };
        send_if(tok, self.dump_resp, has_d, nxt[didx]);
    }
}

"""

CASE = """// __NAME__: __DESC__
#[test]
proc __TEST__ {
    terminator: chan<bool> out,
    start: chan<u32> out,
    done: chan<bool> in,
    dump_req: chan<u32> out,
    dump_resp: chan<u64> in,
    cyc: u32,
}

impl __TEST__ {
    fn new(terminator: chan<bool> out) -> Self {
        let (start_s, start_r) = chan<u32>("start");
        let (done_s, done_r) = chan<bool>("done");
        let (areq_s, areq_r) = chan<__REQ__>("act_req");
        let (aresp_s, aresp_r) = chan<(u64,)>("act_resp");
        let (awc_s, awc_r) = chan<()>("act_wc");
        let (wreq_s, wreq_r) = chan<__REQ__>("wgt_req");
        let (wresp_s, wresp_r) = chan<(u64,)>("wgt_resp");
        let (wwc_s, wwc_r) = chan<()>("wgt_wc");
        let (oreq_s, oreq_r) = chan<__REQ__>("out_req");
        let (oresp_s, oresp_r) = chan<(u64,)>("out_resp");
        let (owc_s, owc_r) = chan<()>("out_wc");
        let (dreq_s, dreq_r) = chan<u32>("dump_req");
        let (dresp_s, dresp_r) = chan<u64>("dump_resp");

        convacc_proc::ConvAccMem<u32:__M__>::new(
            start_r, done_s,
            areq_s, aresp_r, awc_r,
            wreq_s, wresp_r, wwc_r,
            oreq_s, oresp_r, owc_r,
        ).spawn();
        __RAM_A__::new(areq_r, aresp_s, awc_s).spawn();
        __RAM_W__::new(wreq_r, wresp_s, wwc_s).spawn();
        __RAM_O__::new(oreq_r, oresp_s, owc_s, dreq_r, dresp_s).spawn();

        Self { terminator, start: start_s, done: done_r,
               dump_req: dreq_s, dump_resp: dresp_r, cyc: u32:0 }
    }

    fn next(self) {
        let cyc = read(self.cyc);
        write(self.cyc, cyc + u32:1);
        if cyc == u32:0 {
            send(join(), self.start, u32:0);
        } else {
            let (tok, ok, has_done) = recv_non_blocking(join(), self.done, false);
            if has_done {
                assert_eq(ok, true);
__CHECKS__
                send(tok, self.terminator, true);
            } else {
                // Bounded: a wedged engine fails the test instead of hanging
                // the interpreter forever.
                assert!(cyc <= u32:__LIMIT__, "__NAME__: no done within the cycle limit");
            }
        }
    }
}

"""

CHECK = """                let tok = send(tok, self.dump_req, u32:__I__);
                let (tok, w__I__) = recv(tok, self.dump_resp);
                assert_eq(w__I__, u64:__V__);"""


def words(buf, n=M):
    b = bytes(buf)[: n * 8]
    if len(b) % 8:
        b += bytes(8 - len(b) % 8)
    out = list(struct.unpack("<%dQ" % (len(b) // 8), b))
    return out + [0] * (n - len(out))


def fill(template, **kw):
    text = template
    for k, v in kw.items():
        text = text.replace("__%s__" % k, str(v))
    return text


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261029
    shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rng = random.Random(seed)

    cases = []
    idx = 0
    while len(cases) < count and idx < count * 6:
        c = R.rand_case(rng, idx)
        idx += 1
        if c:
            cases.append(c)

    # One self-contained block per case: its three RAM procs (image baked in)
    # and the #[test] proc that drives them.
    chunks = []
    for i, (name, act, wgt, expected) in enumerate(cases):
        exp = (expected.astype(np.int8).tobytes() + bytes(WORDS_CHECKED * 8))
        exp_words = list(struct.unpack("<%dQ" % WORDS_CHECKED, exp[: WORDS_CHECKED * 8]))
        checks = "\n".join(
            fill(CHECK, I=j, V=exp_words[j]) for j in range(WORDS_CHECKED))
        chunks.append("".join([
            fill(RAM, LABEL="Activation", NAME="Act_%d" % i, M=M, REQ=REQ,
                 IMAGE=", ".join(map(str, words(act)))),
            fill(RAM, LABEL="Weight", NAME="Wgt_%d" % i, M=M, REQ=REQ,
                 IMAGE=", ".join(map(str, words(wgt)))),
            fill(OUT_RAM, NAME="Out_%d" % i, M=M, REQ=REQ),
            fill(CASE, NAME=name, TEST="Case_%d" % i,
                 DESC="expect words[0:%d] = %s" % (
                     WORDS_CHECKED, " ".join("0x%016x" % w for w in exp_words)),
                 M=M, REQ=REQ, LIMIT=CYCLE_LIMIT,
                 RAM_A="Act_%d" % i, RAM_W="Wgt_%d" % i, RAM_O="Out_%d" % i,
                 CHECKS=checks),
        ]))

    # Shards let several interpreter processes run in parallel (the interpreter
    # is single-threaded, and a shard is just a smaller independent file). The
    # manifest names exactly what this run wrote, so a changed shard count never
    # leaves the runner picking up stale files.
    written = []
    for s in range(max(1, shards)):
        part = chunks[s::max(1, shards)]
        if not part:
            continue
        fname = ("dslx_fullsys.x" if shards <= 1
                 else "dslx_fullsys_%d.x" % s)
        out = os.path.join(ROOT, "hw", fname)
        with open(out, "w") as f:
            f.write(HEADER + "".join(part))
        written.append(os.path.join("hw", fname))

    manifest = os.path.join(ROOT, "hw", "dslx_fullsys.files")
    with open(manifest, "w") as f:
        f.write("\n".join(written) + "\n")

    print(f"wrote {len(written)} file(s) for {len(cases)} cases, M={M}, "
          f"shards={shards}")
    for w in written:
        print("  ", w)


if __name__ == "__main__":
    main()
