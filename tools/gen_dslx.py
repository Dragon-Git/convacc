#!/usr/bin/env python3
"""Generate every DSLX harness under hw/ from one entry point.

There is one engine and one FSM; what differs between the harnesses is the
engine depth, the stop condition and where the cases come from. So the emitters
live in dslx_codegen.py, the case sources live here, and each subcommand writes
one group of files:

  rand        randomized array-engine cases          hw/dslx_rand.x
  equiv       memory-mapped engine, random cases     hw/dslx_equiv.x
  fullsys     engine proc + three RAM procs          hw/dslx_fullsys[_N].x
  cmp         one pre-rendered corpus case           hw/dslx_cmp.x
  synth       hand-checkable synthetic cases         hw/dslx_synth.x
  probe       Engine state probes, built-in cases    hw/dslx_probe.x
  cases       Engine state probes, named test cases  hw/dslx_probe.x
  probe-rand  Engine state probes, one random case   hw/dslx_probe_rand.x

Usage:  tools/gen_dslx.py rand 30 20261029
        tools/gen_dslx.py fullsys 100 20261029 8
        tools/gen_dslx.py cases dense_random_quant group_dilated
"""
from __future__ import annotations

import argparse
import os
import random
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import dslx_codegen as C  # noqa: E402
from convacc.image import decode_descriptor, make_image  # noqa: E402
from convacc.spec import ConvSpec, GemmSpec  # noqa: E402

# Depth of the small harnesses (see dslx_codegen.M_SMALL).
M = C.M_SMALL
# `dslx_cmp` drives the RTL-sized engine, because it replays a real corpus case.
CORPUS_WORDS = 16384


# ---------------------------------------------------------------------------
# Random cases: the case source shared by rand / equiv / fullsys / probe-rand
# ---------------------------------------------------------------------------
FAMILIES = ("dense", "depth", "group", "deconv", "gemm")


def _build(name, spec, w, layout, a, width):
    try:
        _, image, expected = make_image(
            spec, w, layout, a,
            np.zeros(width, np.int32), np.ones(width, np.int32),
            np.zeros(width, np.uint8), np.zeros(width, np.int8),
        )
    except Exception as e:  # noqa: BLE001
        print(f"  skip {name}: {e}")
        return None
    # The harness declares act/wgt/out as u64[256]. Anything that does not fit
    # would be silently truncated and show up as a bogus mismatch, so reject it
    # here instead. `expected` has to be checked too -- an oversized output
    # window is exactly the case a naive activation-only check would miss.
    need = max(C.used_words(image.activation), C.used_words(image.weight))
    out_words = (len(expected.tobytes()) + 7) // 8
    if need > M or out_words > M:
        print(f"  skip {name}: needs {need} act/wgt words, {out_words} out words > {M}")
        return None
    return name, image.activation, image.weight, expected


def _rand_gemm(rng, idx):
    # GEMM takes a different descriptor path: no taps, `k` is the reduction
    # loop and the position index is the row. The conv-only shapes never
    # covered it, so the whole gemm branch was reachable only from cocotb.
    m = rng.randint(1, 6)
    k = rng.randint(1, 12)
    n = rng.randint(1, 8)
    rng_w = np.random.default_rng(rng.randint(0, 10 ** 9))
    w = rng_w.integers(-3, 4, size=(k, n), dtype=np.int8)
    a = rng_w.integers(-3, 4, size=(m, k), dtype=np.int8)
    return _build(f"rnd{idx:02d}_gemm", GemmSpec(m=m, k=k, n=n), w, "KN", a, n)


def _try_conv(rng, idx, kind):
    n = rng.choice([1, 1, 2])
    hi = rng.randint(2, 4)
    wi = rng.randint(2, 4)
    kh = rng.choice([1, 3, 5])
    kw = rng.choice([1, 3, 5])
    dh = rng.choice([1, 1, 2])
    dw = rng.choice([1, 1, 2])
    sh = rng.randint(1, 3)
    sw = rng.randint(1, 3)
    ph = rng.randint(0, 1)
    pw = rng.randint(0, 1)
    # A dilated kernel spans dh*(kh-1)+1 input lanes, so the output extent is
    # driven by the dilated footprint rather than by kh.
    ho = (hi + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    wo = (wi + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    if ho <= 0 or wo <= 0:
        return None
    if kind == "depth":
        c = rng.randint(1, 4)
        ci, co, groups, layout = c, c, c, "HWM"
    elif kind == "group":
        g = rng.choice([2, 3])
        cp = rng.randint(1, 2)
        ci, co, groups = g * cp, g * rng.randint(1, 2), g
        layout = "HWGIO"
    elif kind == "dense":
        ci = rng.randint(1, 8)
        co = rng.randint(1, 10)
        groups, layout = 1, "HWIO"
    else:  # deconv (groups must be 1, weight 4-D)
        ci = rng.randint(1, 3)
        co = rng.randint(1, 4)
        groups, layout = 1, "HWIO"
        # The deconv reference scatters without dilation (oh = ih*sh + kh - ph),
        # so keep dh = dw = 1 here; dilated deconv is not a supported shape.
        dh = dw = 1
        ho = (hi - 1) * sh + kh - 2 * ph
        wo = (wi - 1) * sw + kw - 2 * pw
        if ho <= 0 or wo <= 0:
            return None
    spec = ConvSpec(n=n, hi=hi, wi=wi, ci=ci, co=co, ho=ho, wo=wo, kh=kh, kw=kw,
                    sh=sh, sw=sw, dh=dh, dw=dw, ph=ph, pw=pw, groups=groups,
                    kind="deconv2d" if kind == "deconv" else "conv2d")
    rng_w = np.random.default_rng(rng.randint(0, 10 ** 9))
    if layout == "HWM":
        w = rng_w.integers(-3, 4, size=(kh, kw, co), dtype=np.int8)
    elif groups == 1:
        w = rng_w.integers(-3, 4, size=(kh, kw, ci, co), dtype=np.int8)
    else:
        cpg = ci // groups
        w = rng_w.integers(-3, 4, size=(kh, kw, groups, cpg, co // groups), dtype=np.int8)
    a = rng_w.integers(-3, 4, size=(n, hi, wi, ci), dtype=np.int8)
    return _build(f"rnd{idx:02d}_{kind}", spec, w, layout, a, co)


def _rand_conv(rng, idx, kind):
    # Dilated / large-kernel / stride-3 draws often leave an empty output
    # window, so redraw the geometry a few times instead of giving up on the
    # first miss -- otherwise the family mix collapses onto whatever happens to
    # be easy to place.
    for _ in range(8):
        case = _try_conv(rng, idx, kind)
        if case:
            return case
    return None


def random_case(rng, idx):
    """One random case. The family rotates with the index so both the M-shape
    and the operator mix stay balanced -- a pure `rng.choice` let the families
    that are easy to place swamp the ones that usually miss."""
    kind = FAMILIES[idx % len(FAMILIES)]
    if kind == "gemm":
        return _rand_gemm(rng, idx)
    return _rand_conv(rng, idx, kind)


def random_cases(n: int, seed: int) -> list:
    """Up to `n` cases; some draws are unplaceable, so give up after 6*n tries."""
    rng = random.Random(seed)
    cases = []
    idx = 0
    while len(cases) < n and idx < n * 6:
        case = random_case(rng, idx)
        idx += 1
        if case:
            cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# rand / equiv
# ---------------------------------------------------------------------------
def cmd_rand(args) -> int:
    cases = random_cases(args.cases, args.seed)
    tests = [C.test_block(f"cmp_{name}", M, act, wgt, exp.astype(np.int8).tobytes())
             for name, act, wgt, exp in cases]
    out = C.write("dslx_rand.x", C.array_driver("run_conv", M, 20000), *tests)
    print(f"wrote {out} with {len(cases)} cases, M={M}")
    for name, *_ in cases:
        print("  ", name)
    return 0


def cmd_equiv(args) -> int:
    cases = random_cases(args.cases, args.seed)
    tests = [C.test_block(f"mem_{name}", M, act, wgt, exp.astype(np.int8).tobytes(),
                          call="run_mem(act, wgt)")
             for name, act, wgt, exp in cases]
    out = C.write("dslx_equiv.x", C.mem_driver("run_mem", M, 20000), *tests)
    print(f"wrote {out} with {len(cases)} cases")
    return 0


# ---------------------------------------------------------------------------
# fullsys: the engine proc wired to three DSLX RAM models
# ---------------------------------------------------------------------------
#
# `rand` / `equiv` drive the engine as a *pure function* over arrays, so the RAM
# handshake layer -- request/response channels, the write-completion channel,
# byte strobes -- is never exercised. In the silicon flow that layer lives in SV
# (hw/convacc_sram.sv + the generated top), so only cocotb reached it.
#
# This harness closes that gap in DSLX: `convacc_proc::ConvAccMem` is spawned
# next to three RAM behaviour procs and driven through real channels by a
# `#[test]` proc, with no SV in the loop at all.
#
# Notes on what is (and is not) modelled:
#   * The RAM models mirror `hw/convacc_sram.sv`: byte-strobed write, full-word
#     read, write-completion token. Read latency is 0 in the model; the engine
#     retries its wait phase until the response lands, so a real (slower) macro
#     cannot break it -- it only takes more cycles.
#   * Memory is 256 words, not the 16384 of the RTL instance: an array in proc
#     state becomes flip-flops in codegen, so this is a *protocol* test, not a
#     substitute for the cocotb regression.
WORDS_CHECKED = 4
CYCLE_LIMIT = 200000
REQ = "(u32, u64, u8, u8, bool, bool)"

FULLSYS_HEADER = """#![feature(explicit_state_access)]
import convacc_proc;

// Full-system harness: convacc_proc::ConvAccMem<256> as the DUT, with the three
// tensor memories as DSLX procs. The image is baked per case because proc state
// must be compile-time evaluable.
"""

FULLSYS_RAM = """// __LABEL__ RAM model (byte-strobed write, full-word read, write completion).
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

FULLSYS_OUT_RAM = """// Output memory: same model plus a dump port the test reads results through.
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

FULLSYS_CASE = """// __NAME__: __DESC__
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

FULLSYS_CHECK = """                let tok = send(tok, self.dump_req, u32:__I__);
                let (tok, w__I__) = recv(tok, self.dump_resp);
                assert_eq(w__I__, u64:__V__);"""


def _fill(template, **kw):
    text = template
    for k, v in kw.items():
        text = text.replace("__%s__" % k, str(v))
    return text


def cmd_fullsys(args) -> int:
    cases = random_cases(args.cases, args.seed)

    # One self-contained block per case: its three RAM procs (image baked in)
    # and the #[test] proc that drives them.
    chunks = []
    for i, (name, act, wgt, expected) in enumerate(cases):
        exp = (expected.astype(np.int8).tobytes() + bytes(WORDS_CHECKED * 8))
        exp_words = list(struct.unpack("<%dQ" % WORDS_CHECKED, exp[: WORDS_CHECKED * 8]))
        checks = "\n".join(
            _fill(FULLSYS_CHECK, I=j, V=exp_words[j]) for j in range(WORDS_CHECKED))
        chunks.append("".join([
            _fill(FULLSYS_RAM, LABEL="Activation", NAME="Act_%d" % i, M=M, REQ=REQ,
                  IMAGE=", ".join(map(str, C.words(act, M)))),
            _fill(FULLSYS_RAM, LABEL="Weight", NAME="Wgt_%d" % i, M=M, REQ=REQ,
                  IMAGE=", ".join(map(str, C.words(wgt, M)))),
            _fill(FULLSYS_OUT_RAM, NAME="Out_%d" % i, M=M, REQ=REQ),
            _fill(FULLSYS_CASE, NAME=name, TEST="Case_%d" % i,
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
    shards = max(1, args.shards)
    for s in range(shards):
        part = chunks[s::shards]
        if not part:
            continue
        fname = ("dslx_fullsys.x" if shards == 1
                 else "dslx_fullsys_%d.x" % s)
        out = os.path.join(ROOT, "hw", fname)
        with open(out, "w") as f:
            f.write(FULLSYS_HEADER + "".join(part))
        written.append(os.path.join("hw", fname))

    manifest = os.path.join(ROOT, "hw", "dslx_fullsys.files")
    with open(manifest, "w") as f:
        f.write("\n".join(written) + "\n")

    print(f"wrote {len(written)} file(s) for {len(cases)} cases, M={M}, "
          f"shards={shards}")
    for w in written:
        print("  ", w)
    return 0


# ---------------------------------------------------------------------------
# cmp: one pre-rendered corpus case on the array engine
# ---------------------------------------------------------------------------
def _corpus_case(name: str):
    d = os.path.join(ROOT, "build", "cocotb_cases", name)
    act = open(os.path.join(d, "activation.bin"), "rb").read()
    wgt = open(os.path.join(d, "weight.bin"), "rb").read()
    exp = open(os.path.join(d, "expected.bin"), "rb").read()
    assert len(act) == CORPUS_WORDS * 8 and len(wgt) == CORPUS_WORDS * 8, (len(act), len(wgt))
    return act, wgt, exp


def _loop_bound(desc) -> int:
    """A bound the FSM is certain to finish inside, not a tight one."""
    block_groups = max(1, (8 + desc.r - 1) // desc.r)
    per_tile = 3 + block_groups + 16 + block_groups + 10 + 8  # phases 3..12 rough
    total = desc.taps + desc.m_tiles * desc.n_tiles * (per_tile + 2 * block_groups)
    return max(512, total * 2 + 32)


def cmd_cmp(args) -> int:
    blocks = []
    for name in args.cases:
        act, wgt, exp = _corpus_case(name)
        desc = decode_descriptor(wgt)
        # The engine writes at out_base, so the expectation has to be padded up
        # to that offset before it can be compared against the whole SRAM.
        exp_words = C.words(exp, CORPUS_WORDS)
        if desc.out_base + len(exp_words) > CORPUS_WORDS:
            print(f"{name}: output at word {desc.out_base} does not fit in "
                  f"{CORPUS_WORDS} words", file=sys.stderr)
            return 2
        exp_full = [0] * CORPUS_WORDS
        for i, value in enumerate(exp_words):
            exp_full[desc.out_base + i] = value
        blocks.append((name, _loop_bound(desc),
                       C.test_block(f"cmp_{name}", CORPUS_WORDS, act, wgt, exp_full)))

    steps = max(bound for _, bound, _ in blocks)
    out = C.write("dslx_cmp.x", C.array_driver("run_conv", CORPUS_WORDS, steps),
                  *(test for _, _, test in blocks))
    print(f"wrote {out} with {len(blocks)} case(s); loop bound N={steps}")
    for name, bound, _ in blocks:
        print(f"  {name}: N={bound}")
    return 0


# ---------------------------------------------------------------------------
# synth: small hand-checkable cases
# ---------------------------------------------------------------------------
def _dump_desc(name: str, image) -> None:
    d = decode_descriptor(image.weight)
    print(f"  [{name}] flags=0x{d.flags:x} gemm={int(d.gemm)} chan={int(d.channelwise)}"
          f" deconv={int(d.deconv)} log_r={d.log_r} m={d.m} n={d.n} k={d.k}"
          f" mh={d.mh} mw={d.mw} o={d.o} r={d.r}"
          f" hi={d.hi} wi={d.wi} ho={d.ho} wo={d.wo} next_desc={d.next_desc}")


def _add_synth(cases, name, spec, w, layout, a):
    n = spec.n if isinstance(spec, GemmSpec) else spec.co
    _, image, expected = make_image(spec, w, layout, a,
                                    np.zeros(n, np.int32), np.ones(n, np.int32),
                                    np.zeros(n, np.uint8), np.zeros(n, np.int8))
    _dump_desc(name, image)
    cases.append((name, image.activation, image.weight, expected))


def cmd_synth(_args) -> int:
    cases = []

    # --- channelwise (depthwise): ci==groups. 1x1, ci=co=1, single output ---
    spec = ConvSpec(n=1, hi=1, wi=1, ci=1, co=1, ho=1, wo=1, kh=1, kw=1,
                    sh=1, sw=1, dh=1, dw=1, ph=0, pw=0, groups=1, kind="conv2d")
    _add_synth(cases, "cw_1x1", spec, np.full((1, 1, 1, 1), 1, np.int8), "HWIO",
               np.full((1, 1, 1, 1), 1, np.int8))

    # depthwise 3x3, ci=co=3, groups=3, 3x3 spatial (multi position + multi tap)
    spec = ConvSpec(n=1, hi=3, wi=3, ci=3, co=3, ho=3, wo=3, kh=3, kw=3,
                    sh=1, sw=1, dh=1, dw=1, ph=1, pw=1, groups=3, kind="conv2d")
    _add_synth(cases, "cw_3x3", spec, np.ones((3, 3, 3), np.int8), "HWM",
               np.ones((1, 3, 3, 3), np.int8))

    # depthwise 3x3 taps but single channel (k=9 multi-block, r=8,o=1)
    spec = ConvSpec(n=1, hi=3, wi=3, ci=1, co=1, ho=3, wo=3, kh=3, kw=3,
                    sh=1, sw=1, dh=1, dw=1, ph=1, pw=1, groups=1, kind="conv2d")
    _add_synth(cases, "cw_k9_r8", spec, np.ones((3, 3, 1), np.int8), "HWM",
               np.ones((1, 3, 3, 1), np.int8))

    # --- deconv multi-phase: stride2, pad1, larger -> several (rh,rw) tasks ---
    spec = ConvSpec(n=1, hi=3, wi=3, ci=1, co=2, ho=5, wo=5, kh=3, kw=3,
                    sh=2, sw=2, dh=1, dw=1, ph=1, pw=1, groups=1, kind="deconv2d")
    _add_synth(cases, "deconv_mp", spec, np.ones((3, 3, 1, 2), np.int8), "HWIO",
               np.ones((1, 3, 3, 1), np.int8))

    # --- gemm wide n=10 -> n_tiles=2 (nt>0) ---
    spec = GemmSpec(m=2, k=2, n=10)
    _add_synth(cases, "gemm_wide", spec, np.ones((2, 10), np.int8), "KN",
               np.array([[1, 2], [3, 4]], np.int8))

    tests = [C.test_block(f"cmp_{name}", M, act, wgt, exp.astype(np.int8).tobytes())
             for name, act, wgt, exp in cases]
    out = C.write("dslx_synth.x", C.array_driver("run_conv", M, 4096), *tests)
    print(f"\nwrote {out} with {len(cases)} case(s), M={M}")
    return 0


# ---------------------------------------------------------------------------
# probe: Engine state read at a chosen phase
# ---------------------------------------------------------------------------
#
# Some assertions here are deliberate dumps: they carry a sentinel (99, 127,
# 999) so the test always fails and the interpreter prints the real value.
PROBE_STEPS = 400


def _probe_case_cw_3x3():
    spec = ConvSpec(n=1, hi=3, wi=3, ci=3, co=3, ho=3, wo=3, kh=3, kw=3,
                    sh=1, sw=1, dh=1, dw=1, ph=1, pw=1, groups=3, kind="conv2d")
    return spec, np.ones((3, 3, 3), np.int8), "HWM", np.ones((1, 3, 3, 3), np.int8)


def _probe_case_cw_k9_r8():
    spec = ConvSpec(n=1, hi=3, wi=3, ci=1, co=1, ho=3, wo=3, kh=3, kw=3,
                    sh=1, sw=1, dh=1, dw=1, ph=1, pw=1, groups=1, kind="conv2d")
    return spec, np.ones((3, 3, 1), np.int8), "HWM", np.ones((1, 3, 3, 1), np.int8)


PROBE_CASES = {"cw_3x3": _probe_case_cw_3x3, "cw_k9_r8": _probe_case_cw_k9_r8}


def cmd_probe(args) -> int:
    spec, wts, layout, acts = PROBE_CASES[args.case]()
    n = spec.co
    _, image, expected = make_image(spec, wts, layout, acts, np.zeros(n, np.int32),
                                    np.ones(n, np.int32), np.zeros(n, np.uint8),
                                    np.zeros(n, np.int8))
    d = decode_descriptor(image.weight)
    print(f"== {args.case}: r={d.r} o={d.o} k={d.k} kgroups={d.k_groups} "
          f"m={d.m} n={d.n} ci={d.ci} co={d.co}")

    taps = list(image.weight[d.tap_base * 8: d.tap_base * 8 + d.taps])
    print("taps bytes:", [hex(t) for t in taps])

    # expected ch_tile bank1 content for kb=0 (load bank is bank^1 = 1)
    k = d.k
    blk = min(8, k)
    mh, mw = d.mh, d.mw
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
            ih = oh * d.sh + kh - d.ph
            iw = ow * d.sw + kw - d.pw
            for oo in range(min(8, d.n)):
                if 0 <= ih < d.hi and 0 <= iw < d.wi:
                    ci = oo  # co_group == 1 in these cases
                    off = d.act_base + ((batch * d.hi + ih) * d.wi + iw) * d.ci + ci
                    ch[512 + p * 64 + kstep * 8 + oo] = act[off]
    print("expected ch_tile nonzero count:", int(np.count_nonzero(ch)))

    # wgt_tile expected: bank1, group g -> 8 bytes of word (w_base + kb*(8/r) + g)
    wgt_words = C.words(image.weight, M)
    wg = np.zeros(128, np.uint8)
    for g in range((blk + d.r - 1) // d.r):
        widx = d.w_base + g  # kb=0
        for j in range(8):
            wg[64 + g * 8 + j] = (wgt_words[widx] >> (8 * j)) & 0xFF
    print("expected wgt_tile nonzero:", int(np.count_nonzero(wg)))

    probes = []
    for p, ks, oo in [(0, 0, 0), (0, 4, 0), (4, 0, 0), (4, 4, 1), (0, 3, 2), (8 - 1, 7, 2)]:
        idx = 512 + p * 64 + ks * 8 + oo
        probes.append((f"ch_p{p}_k{ks}_o{oo}", idx, int(ch[idx])))
    for g, j in [(0, 0), (0, 1), (2, 0), (3, 5)]:
        probes.append((f"wg_g{g}_j{j}", 64 + g * 8 + j, int(wg[64 + g * 8 + j])))
    for t in [0, 1, 8]:
        probes.append((f"tap{t}", t, int(taps[t])))

    # acc after the 4 mac groups of kb=0: acc[p][o] = sum_{g,rr} ch[p,g*2+rr,o] * wg[g,o*2+rr]
    acc_exp = np.zeros((8, 8), np.int32)
    for g in range((blk + d.r - 1) // d.r):
        for o in range(min(8, d.n)):
            for rr in range(d.r):
                ks = g * d.r + rr
                if ks < blk:
                    acc_exp[:, o] += ch[512 + np.arange(8) * 64 + ks * 8 + o].astype(np.int32) * int(
                        wg[64 + g * 8 + o * d.r + rr]
                    )
    print("expected acc[p=4][:3]:", acc_exp[4, :3].tolist(), " acc[p=0][:3]:", acc_exp[0, :3].tolist())

    for p, o in [(4, 0), (4, 1), (0, 0), (0, 2)]:
        probes.append((f"acc_p{p}_o{o}", p * 8 + o, int(acc_exp[p, o])))

    drivers = [
        C.array_driver("run_to_phase", M, PROBE_STEPS, cond="st.phase >= ph", params=", ph: u8",
                       ret="convacc_proc::Engine<%d>" % M, result="st"),
        C.array_driver("run_to_phase8_mt1", M, PROBE_STEPS, cond="st.phase == u8:8 && st.mt == u32:1",
                       ret="convacc_proc::Engine<%d>" % M, result="st"),
        C.array_driver("run_phase8_n", M, PROBE_STEPS, cond="i >= n", boot="false", params=", n: u32",
                       init="run_to_phase(act, wgt, u8:8)",
                       ret="convacc_proc::Engine<%d>" % M, result="st"),
    ]

    tests = []
    for name, idx, val in probes:
        if name.startswith("acc"):
            stmt = "run_phase8_n(act_const(), wgt_const(), u32:4)"
            check = "st.acc[%d], s32:%d" % (idx, val)
        elif name.startswith("tap"):
            stmt = "run_to_phase(act_const(), wgt_const(), u8:8)"
            check = "st.taps[%d], u8:%d" % (idx, val)
        elif name.startswith("wg"):
            stmt = "run_to_phase(act_const(), wgt_const(), u8:8)"
            check = "st.wgt_tile[%d], u8:%d" % (idx, val)
        else:
            stmt = "run_to_phase(act_const(), wgt_const(), u8:8)"
            check = "st.ch_tile[%d], s8:%d" % (idx, val)
        tests.append("#[test]\nfn probe_%s() {\n    let st = %s;\n    assert_eq(%s);\n}"
                     % (name, stmt, check))

    # ---- extra probes: state fields, params, mt, final out_mem ----
    params = [wgt_words[d.param_base + i] for i in range(10)]
    for fld, ty in [("phase", "u8"), ("kb", "u32"), ("block_groups", "u32"),
                    ("group_index", "u32"), ("m_tiles", "u32"),
                    ("n_tiles", "u32"), ("mt", "u32"), ("nt", "u32"),
                    ("out_p", "u32"), ("param_index", "u32")]:
        tests.append("#[test]\nfn probe_state_%s() {\n"
                     "    let st = run_to_phase(act_const(), wgt_const(), u8:14);\n"
                     "    assert_eq(st.%s, %s:255);\n}" % (fld, fld, ty))

    # interior point p=4 sees all nine taps, each contributing 1*1
    tests.append("#[test]\nfn probe_full_acc() {\n"
                 "    let st = run_to_phase(act_const(), wgt_const(), u8:11);\n"
                 "    assert_eq(st.acc[32], s32:9);\n}")
    for i in [0, 4, 8, 9]:
        tests.append("#[test]\nfn probe_param%d() {\n"
                     "    let st = run_to_phase(act_const(), wgt_const(), u8:11);\n"
                     "    assert_eq(st.params[%d], u64:%d);\n}" % (i, i, params[i]))

    tests.append("#[test]\nfn probe_mt_after_run() {\n"
                 "    let st = run_to_phase(act_const(), wgt_const(), u8:14);\n"
                 "    assert_eq(st.mt, u32:%d);\n}" % ((d.m + 7) // 8 - 1))
    exp_w0 = int.from_bytes(expected.astype(np.int8).tobytes()[:8], "little")
    tests.append("#[test]\nfn probe_out_word0() {\n"
                 "    let st = run_to_phase(act_const(), wgt_const(), u8:14);\n"
                 "    assert_eq(st.out_mem[0], u64:%d);\n}" % exp_w0)

    # --- dumps: these fail on purpose so the interpreter prints the value ---
    tests.append("#[test]\nfn probe_final_acc0() {\n"
                 "    let st = run_to_phase(act_const(), wgt_const(), u8:14);\n"
                 "    assert_eq(st.acc[0], s32:12345);\n}")
    for w in (0, 1):
        tests.append("#[test]\nfn probe_final_out%d() {\n"
                     "    let st = run_to_phase(act_const(), wgt_const(), u8:14);\n"
                     "    assert_eq(st.out_mem[%d], u64:999);\n}" % (w, w))
    tests.append("#[test]\nfn probe_mt1_ch() {\n"
                 "    let st = run_to_phase8_mt1(act_const(), wgt_const());\n"
                 "    assert_eq(st.ch_tile[512 + 4 * 8], s8:1);\n}")
    tests.append("#[test]\nfn probe_mt1_bank() {\n"
                 "    let st = run_to_phase8_mt1(act_const(), wgt_const());\n"
                 "    assert_eq(st.bank, u1:1);\n}")
    tests.append("#[test]\nfn probe_mt1_wgt() {\n"
                 "    let st = run_to_phase8_mt1(act_const(), wgt_const());\n"
                 "    assert_eq(st.wgt_tile[64], u8:99);\n}")
    for idx in [512, 520, 528, 536, 544, 552, 560, 568, 576, 640, 704, 0, 8]:
        tests.append("#[test]\nfn probe_mt1_dump_%d() {\n"
                     "    let st = run_to_phase8_mt1(act_const(), wgt_const());\n"
                     "    assert_eq(st.ch_tile[%d], s8:127);\n}" % (idx, idx))
    tests.append("#[test]\nfn probe_mt1_where() {\n"
                 "    let st = run_to_phase8_mt1(act_const(), wgt_const());\n"
                 "    assert_eq(st.phase, u8:8);\n}")
    for suffix, fld in (("ldp", "ld_p"), ("ldr", "ld_r"), ("nt", "nt")):
        tests.append("#[test]\nfn probe_mt1_%s() {\n"
                     "    let st = run_to_phase8_mt1(act_const(), wgt_const());\n"
                     "    assert_eq(st.%s, u32:99);\n}" % (suffix, fld))
    for fld in ("kb", "block_groups", "wg_index", "m_tiles"):
        tests.append("#[test]\nfn probe_mt1_%s() {\n"
                     "    let st = run_to_phase8_mt1(act_const(), wgt_const());\n"
                     "    assert_eq(st.%s, u32:99);\n}" % (fld, fld))

    consts = "\n".join([C.const_array("act_const", M, image.activation),
                        C.const_array("wgt_const", M, image.weight)])
    out = C.write("dslx_probe.x", *drivers, consts, *tests)
    print("wrote", out)
    return 0


# ---------------------------------------------------------------------------
# cases: the same probes, but with names from tests/test_cocotb_convacc.py
# ---------------------------------------------------------------------------
#
# `probe` only builds two hand-written cases. This one takes case names from the
# cocotb suite, so the probe runs on the same shapes -- including GEMM and the
# real requantiser -- that the RTL regression covers.
CASES_STEPS = 20000


def cmd_cases(args) -> int:
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    sys.path.insert(0, os.path.join(ROOT, "tests", "cocotb"))
    import test_cocotb_convacc as suite

    depth = args.depth
    by_name = {c.name: c for c in suite.build_cases()}
    tests = []
    for name in args.names:
        if name not in by_name:
            print(f"  no such case: {name}")
            continue
        c = by_name[name]
        _, image, expected = make_image(
            c.spec, c.weights, c.layout, c.activations,
            c.bias, c.scale, c.shift, c.zero_point)
        need = max(C.used_words(image.activation), C.used_words(image.weight))
        if need > depth:
            print(f"  skip {name}: needs {need} words > {depth}")
            continue
        tests.append(C.test_block(f"probe_{name.replace('-', '_')}", depth, image.activation,
                                  image.weight, expected.astype(np.int8).tobytes(),
                                  call="run_mem(act, wgt)"))
        print(f"  queued {name}: {need} words, expected {expected.size} bytes")

    if not tests:
        print("no usable cases")
        return 1

    out = C.write("dslx_probe.x", C.mem_driver("run_mem", depth, CASES_STEPS), *tests)
    print(f"wrote {out} with {len(tests)} cases (M={depth})")
    return 0


# ---------------------------------------------------------------------------
# probe-rand: dump Engine fields for one random case
# ---------------------------------------------------------------------------
def cmd_probe_rand(args) -> int:
    rng = random.Random(args.seed)
    chosen = None
    idx = 0
    while chosen is None and idx < args.draw * 6:
        candidate = random_case(rng, idx)
        idx += 1
        if candidate and candidate[0] == args.case:
            chosen = candidate
    if chosen is None:
        print(f"case {args.case} not found", file=sys.stderr)
        return 2

    name, act, wgt, exp = chosen
    exp_bytes = exp.astype(np.int8).tobytes()

    at14 = "run_to_phase(act_const(), wgt_const(), u8:14)"
    at8_nt1 = "run_to_phase8_nt1(act_const(), wgt_const())"
    at11_nt1 = "run_to_phase11_nt1(act_const(), wgt_const())"

    def dump(name, stmt, field, ty, idx, value):
        lhs = f"st.{field}[{idx}]" if idx is not None else f"st.{field}"
        return ("#[test]\nfn probe_%s() {\n    let st = %s;\n    assert_eq(%s, %s:%s);\n}"
                % (name, stmt, lhs, ty, value))

    tests = [dump(name, at14, field, ty, idx, 123456) for name, field, ty, idx in [
        ("acc0", "acc", "s32", 0),
        ("acc1", "acc", "s32", 1),
        ("nt_final", "nt", "u32", None),
        ("out1", "out_mem", "u64", 1),
        ("out0", "out_mem", "u64", 0),
    ]]
    tests += [dump("nt1_acc_before_write", at11_nt1, "acc", "s32", 0, 1),
              dump("nt1_acc1_before_write", at11_nt1, "acc", "s32", 1, 1)]
    tests += [dump(name, at8_nt1, field, ty, idx, 1) for name, field, ty, idx in [
        ("nt1_acc0", "acc", "s32", 0),
        ("nt1_at", "act_tile", "s8", 0),
        ("nt1_wg", "wgt_tile", "u8", 0),
        ("nt1_bank", "bank", "u1", None),
        ("nt1_wg2", "wgt_tile", "u8", 2),
    ]]

    out = C.write(
        "dslx_probe_rand.x",
        C.array_driver("run_to_phase", M, 20000, cond="st.phase >= ph",
                       params=", ph: u8", ret="convacc_proc::Engine<%d>" % M, result="st"),
        C.array_driver("run_to_phase8_nt1", M, 20000,
                       cond="st.phase == u8:8 && st.nt == u32:1",
                       ret="convacc_proc::Engine<%d>" % M, result="st"),
        C.array_driver("run_to_phase11_nt1", M, 20000,
                       cond="st.phase == u8:11 && st.nt == u32:1",
                       ret="convacc_proc::Engine<%d>" % M, result="st"),
        "\n".join([C.const_array("act_const", M, act),
                   C.const_array("wgt_const", M, wgt)]),
        *tests,
    )
    print("wrote", out, "expected word0/1:",
          int.from_bytes(exp_bytes[:8], "little"),
          int.from_bytes(exp_bytes[8:16], "little"))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="gen_dslx", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, help_text):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(fn=fn)
        return p

    p = add("rand", cmd_rand, "randomized array-engine cases")
    p.add_argument("cases", nargs="?", type=int, default=30)
    p.add_argument("seed", nargs="?", type=int, default=20261029)

    p = add("equiv", cmd_equiv, "memory-mapped engine, random cases")
    p.add_argument("cases", nargs="?", type=int, default=30)
    p.add_argument("seed", nargs="?", type=int, default=20261029)

    p = add("fullsys", cmd_fullsys, "engine proc + three RAM procs")
    p.add_argument("cases", nargs="?", type=int, default=100)
    p.add_argument("seed", nargs="?", type=int, default=20261029)
    p.add_argument("shards", nargs="?", type=int, default=1,
                   help="split the cases over N files (default 1)")

    p = add("cmp", cmd_cmp, "one pre-rendered corpus case")
    p.add_argument("cases", nargs="*", default=["grid_deconv_0000"])

    p = add("synth", cmd_synth, "hand-checkable synthetic cases")

    p = add("probe", cmd_probe, "Engine state probes, built-in cases")
    p.add_argument("case", nargs="?", default="cw_3x3", choices=sorted(PROBE_CASES))

    p = add("cases", cmd_cases, "Engine state probes, named test cases")
    p.add_argument("names", nargs="+", help="case names from tests/test_cocotb_convacc.py")
    p.add_argument("-M", "--depth", type=int, default=1024,
                   help="engine depth in words (default 1024)")

    p = add("probe-rand", cmd_probe_rand, "Engine state probes, one random case")
    p.add_argument("case", nargs="?", default="rnd24_gemm")
    p.add_argument("draw", nargs="?", type=int, default=40,
                   help="how many cases to draw while looking for CASE")
    p.add_argument("seed", nargs="?", type=int, default=20261029)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
