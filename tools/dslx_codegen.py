#!/usr/bin/env python3
"""The scaffolding every ``gen_dslx_*.py`` harness emits.

There are seven generators and they all drive the same engine through the same
FSM: build an ``Engine`` from two images, call ``engine_step`` in a loop until
some condition holds, then assert something about the result. What differs is
the engine depth, the loop bound, the stop condition and the per-case
constants -- so the boilerplate lives here and each generator keeps only its
case construction.

Templates are ``string.Template`` (``$name``) rather than ``str.format`` or
f-strings: DSLX is full of braces, and doubling every one of them is how the
``{{`` that used to sit in ``hw/dslx_cmp.x`` got there.
"""
from __future__ import annotations

import os
import struct
from string import Template

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every harness starts with these two lines; `explicit_state_access` is what
# lets a `#[test]` read Engine fields at all.
HEADER = "#![feature(explicit_state_access)]\nimport convacc_proc;\n"

# Depth of the engine used by the small harnesses. Synthetic and random cases
# only touch low addresses, and the interpreter's cost grows with the array
# size, so M=256 is far cheaper than the 16384 of the RTL instance.
M_SMALL = 256


def words(buf, n: int) -> list[int]:
    """`buf` as exactly `n` little-endian 64-bit words, zero padded."""
    b = bytes(buf)[: n * 8]
    if len(b) % 8:
        b += bytes(8 - len(b) % 8)
    out = list(struct.unpack("<%dQ" % (len(b) // 8), b))
    return out + [0] * (n - len(out))


def used_words(buf) -> int:
    """Highest non-zero word in `buf`, plus one.

    The SRAM powers up zeroed, so an image only has to be loaded up to here:
    loading 16384 words per tensor instead of a few dozen is the difference
    between a case costing a few cycles and costing a whole load sweep.
    """
    b = bytes(buf)
    for i in range(len(b) // 8 - 1, -1, -1):
        if b[i * 8:(i + 1) * 8] != bytes(8):
            return i + 1
    return 0


def _csv(values) -> str:
    return ",".join(str(v) for v in values)


_ARRAY_DRIVER = Template("""\
fn $fn(act: u64[$m], wgt: u64[$m]$params) -> $ret {
    let st0 = $init;
    let (st, _) = for (i, (st, done)): (u32, (convacc_proc::Engine<$m>, bool)) in u32:0..u32:$steps {
        if $cond {
            (st, done)
        } else {
            let (nst, ndone) = convacc_proc::engine_step(st, $boot, u32:0);
            (nst, ndone)
        }
    }((st0, false));
    $result
}
""")

_MEM_DRIVER = Template("""\
fn $fn(act: u64[$m], wgt: u64[$m]) -> u64[$m] {
    let st0 = convacc_proc::Engine<$m> { ..convacc_proc::zero_engine<u32:$m>() };
    let mem0 = convacc_proc::MemIn { act_data: u64:0, wgt_data: u64:0, out_data: u64:0 };
    let (_, _, outp, _) = for (i, (st, mem_in, outp, done)):
            (u32, (convacc_proc::Engine<$m>, convacc_proc::MemIn, u64[$m], bool))
            in u32:0..u32:$steps {
        if done {
            (st, mem_in, outp, done)
        } else {
            let (nst, ndone, req) = convacc_proc::engine_step_mem(st, i == u32:0, u32:0, mem_in);
            let ai = if req.act_raddr < u32:$m { req.act_raddr } else { u32:0 };
            let wi = if req.wgt_raddr < u32:$m { req.wgt_raddr } else { u32:0 };
            let oi = if req.out_raddr < u32:$m { req.out_raddr } else { u32:0 };
            let wa = if req.out_waddr < u32:$m { req.out_waddr } else { u32:0 };
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
    }((st0, mem0, zero!<u64[$m]>(), false));
    outp
}
""")

_TEST = Template("""\
#[test]
fn $name() {
    let act = u64[$m]:[$act];
    let wgt = u64[$m]:[$wgt];
    let exp = u64[$m]:[$exp];
    assert_eq($call, exp);
}
""")


def array_driver(fn: str, m: int, steps: int, cond: str = "done",
                 init: str | None = None, boot: str = "i == u32:0",
                 params: str = "", ret: str | None = None,
                 result: str = "st.out_mem") -> str:
    """A function that runs `engine_step` until `cond` holds.

    `boot` is the start pulse fed to the step (probe drivers keep stepping after
    the first call with it low); `init` overrides the `st0` expression, which is
    how a probe chains two runs together.
    """
    return _ARRAY_DRIVER.substitute(
        fn=fn, m=m, steps=steps, cond=cond, boot=boot, params=params,
        ret=ret or "u64[%d]" % m, result=result,
        init=init or "convacc_proc::Engine { act, wgt, ..convacc_proc::zero_engine<u32:%d>() }" % m,
    )


def mem_driver(fn: str, m: int, steps: int) -> str:
    """The RAM-facing variant: `engine_step_mem` served by arrays, not a proc."""
    return _MEM_DRIVER.substitute(fn=fn, m=m, steps=steps)


def const_array(fn: str, m: int, buf) -> str:
    """`fn fn() -> u64[m] { u64[m]:[...] }` -- how the probe drivers get images."""
    return "fn %s() -> u64[%d] { u64[%d]:[%s] }" % (fn, m, m, _csv(words(buf, m)))


def test_block(name: str, m: int, act, wgt, exp, call: str = "run_conv(act, wgt)") -> str:
    """One `#[test]` that runs `call` and compares the result against `exp`.

    Each of act/wgt/exp is either a byte image (padded or truncated to `m`
    words) or an already word-aligned list -- the golden regression places its
    expected output at `out_base` and has no bytes to hand over.
    """
    return _TEST.substitute(
        name=name, m=m,
        act=_csv(act if isinstance(act, (list, tuple)) else words(act, m)),
        wgt=_csv(wgt if isinstance(wgt, (list, tuple)) else words(wgt, m)),
        exp=_csv(exp if isinstance(exp, (list, tuple)) else words(exp, m)),
        call=call,
    )


def write(name: str, *parts: str) -> str:
    """Write `HEADER` plus `parts` to hw/<name> and return the path."""
    path = os.path.join(ROOT, "hw", name)
    body = "\n\n".join(p.strip("\n") for p in parts)
    with open(path, "w") as f:
        f.write(HEADER + "\n" + body + "\n")
    return path
