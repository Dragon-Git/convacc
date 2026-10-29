#!/usr/bin/env python3
"""Randomized small-case DSLX-vs-numpy harness.

Generates N random conv specs (depthwise / grouped / dense / deconv), builds the
real SRAM image via convacc.image_builder, then drives engine_step until done and
asserts out_mem == numpy reference. Small M keeps the interpreter fast.
"""
from __future__ import annotations

import os
import random
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from convacc.image_builder import make_image  # noqa: E402
from convacc.spec import ConvSpec, GemmSpec  # noqa: E402

M = 256
MAX_STEPS = 20000


def _words_trunc(b, n):
    b = bytes(b)[: n * 8]
    if len(b) % 8:
        b = b + bytes(8 - len(b) % 8)
    out = list(struct.unpack("<%dQ" % (len(b) // 8), b))
    out += [0] * (n - len(out))
    return out


def emit(case_name, act_bytes, wgt_bytes, exp):
    return """#[test]
fn cmp_%s() {
    let act = u64[%d]:[%s];
    let wgt = u64[%d]:[%s];
    let exp = u64[%d]:[%s];
    let computed = run_conv(act, wgt);
    assert_eq(computed, exp);
}
""" % (case_name, M, ",".join(map(str, _words_trunc(act_bytes, M))),
       M, ",".join(map(str, _words_trunc(wgt_bytes, M))),
       M, ",".join(map(str, _words_trunc(exp.astype(np.int8).tobytes(), M))))


def _used_words(buf):
    b = bytes(buf)
    for i in range(len(b) // 8 - 1, -1, -1):
        if b[i * 8:(i + 1) * 8] != b"\x00" * 8:
            return i + 1
    return 0


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
    need = max(_used_words(image.activation), _used_words(image.weight))
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


FAMILIES = ("dense", "depth", "group", "deconv", "gemm")


def rand_case(rng, idx):
    """One random case. The family rotates with the index so both the M-shape
    and the operator mix stay balanced -- a pure `rng.choice` let the families
    that are easy to place swamp the ones that usually miss."""
    kind = FAMILIES[idx % len(FAMILIES)]
    if kind == "gemm":
        return _rand_gemm(rng, idx)
    return _rand_conv(rng, idx, kind)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261029
    rng = random.Random(seed)
    cases = []
    idx = 0
    while len(cases) < n and idx < n * 6:
        c = rand_case(rng, idx)
        idx += 1
        if c:
            cases.append(c)
    header = """#![feature(explicit_state_access)]
import convacc_proc;

fn run_conv(act: u64[%(M)d], wgt: u64[%(M)d]) -> u64[%(M)d] {
    let st0 = convacc_proc::Engine { act, wgt, ..convacc_proc::zero_engine<u32:%(M)d>() };
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<%(M)d>, bool)) in u32:0..u32:%(S)d {
        if done { (st, done) } else {
            let (nst, ndone) = convacc_proc::engine_step(st, i == u32:0, u32:0);
            (nst, ndone)
        }
    }((st0, false));
    st.out_mem
}
""" % {"M": M, "S": MAX_STEPS}
    body = "".join(emit(nm, a, w, e) for nm, a, w, e in cases)
    out = os.path.join(ROOT, "hw", "dslx_rand.x")
    with open(out, "w") as f:
        f.write(header + body)
    print(f"wrote {out} with {len(cases)} cases, M={M}")
    for nm, _, _, _ in cases:
        print("  ", nm)


if __name__ == "__main__":
    main()
