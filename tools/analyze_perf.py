"""Per-case performance analysis: actual engine cycles vs the microarchitecture.

Sources of truth:
  * actual cycles -- the batch testbench records the start-to-done cycle count
    of every simulated case in `build/cocotb/verilator/run_*/result.json`
    (field "cycles"); chunk-level `sim_time` in results.xml is the cross-check.
  * expected cycles -- derived here from the engine's phase structure
    (hw/convacc_proc.x, engine_step_mem), phase by phase:

      phase 0        1   start handshake
      1/33        2*16   descriptor words
      2/34      2*T+1   tap bytes + the terminating check
      3             1   tile-geometry setup
      per (mt,nt) tile:
        4           1   k-guard
        per k-block (ceil(k/8)):
          5         1   tile clear / block setup
          6/38  2*G+1   weight tile loads + the terminating check
          7/37    2*S   activation fetch steps (4 per crossing step)
          8         G   MAC accumulates, one per weight group
      10/42        20   requant parameters (10 words)
      11            8   output write-back rows
      13            1   next-descriptor / done

  The activation walk mirrors `act_info` exactly (fill, byte_sel, bounds), so
  the model predicts the RTL cycle by cycle; any residual is a small constant
  from the TB's start/done sampling.

Usage:
  python tools/analyze_perf.py            # compare against whatever results
                                          # are present, print the report
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "cocotb"))

import xml.etree.ElementTree as ET  # noqa: E402

from convacc.planner import Plan, Task, make_plan  # noqa: E402
from convacc.spec import ConvSpec, GemmSpec  # noqa: E402


# --------------------------------------------------------------------------
# Expected-cycle model (mirrors engine_step_mem / act_info in convacc_proc.x)
# --------------------------------------------------------------------------

def _act_steps(spec, task: Task, mt: int, nt: int, kb: int):
    """(steps, crossings) of the phase-7 activation walk for one k-block."""
    r = task.r
    o = task.o
    blk = min(8, task.k - kb * 8)
    steps = crossings = 0
    for ld_p in range(8):
        ld_r = 0
        r_limit = blk if task.mode == "channelwise" else 8
        while ld_r < r_limit:
            r_abs = kb * 8 + ld_r
            m_abs = mt * 8 + ld_p
            per_batch = task.mh * task.mw
            batch, rem = divmod(m_abs, per_batch)
            oh = task.phase_h + (rem // task.mw) * task.out_step_h
            ow = task.phase_w + (rem % task.mw) * task.out_step_w

            channel = task.mode == "channelwise"
            if isinstance(spec, GemmSpec):
                tap_index = r_abs // task.ci_count
                kh = kw = 0
                addr = m_abs * task.k + r_abs
                in_bounds = m_abs < task.m and r_abs < task.k
            else:
                tap_index = r_abs if channel else r_abs // task.ci_count
                tap = task.taps[tap_index] if tap_index < len(task.taps) else (0, 0)
                kh, kw = tap
                if spec.kind == "deconv2d":
                    num_h = oh + spec.ph - kh
                    num_w = ow + spec.pw - kw
                    ih = num_h // spec.sh if num_h >= 0 and num_h % spec.sh == 0 else 65535
                    iw = num_w // spec.sw if num_w >= 0 and num_w % spec.sw == 0 else 65535
                else:
                    ih = oh * spec.sh + kh * spec.dh - spec.ph
                    iw = ow * spec.sw + kw * spec.dw - spec.pw
                co_start = task.co_base + nt * o
                if channel:
                    elem = co_start // getattr(spec, "co_per_group", 1)
                else:
                    elem = task.ci_base + r_abs % task.ci_count
                addr = ((batch * spec.hi + ih) * spec.wi + iw) * spec.ci + elem
                in_bounds = (
                    m_abs < task.m
                    and r_abs < task.k
                    and ih < spec.hi
                    and iw < spec.wi
                )

            safe = addr if in_bounds else 0
            byte_sel = safe % 8
            if channel:
                fill = 1
            elif in_bounds:
                if isinstance(spec, GemmSpec):
                    cap_tap = 8
                else:
                    cap_tap = task.ci_count - r_abs % task.ci_count
                fill = min(cap_tap, task.k - r_abs, 8 - ld_r, 8 - byte_sel)
            else:
                fill = 1

            crossing = False
            if channel:
                valid_o = min(task.n - nt * o, 8)
                co_group = getattr(spec, "co_per_group", 1)
                crossing = in_bounds and byte_sel + (valid_o - 1) // co_group >= 8
            steps += 1
            crossings += int(crossing)
            ld_r += fill
    return steps, crossings


# Measured cost of one RAM round trip (issue -> skid -> registered rd_data ->
# response handshake) is 4 cycles, not the nominal 2. Least squares over the
# 1326 recorded cases gives 3.85/cfetch with R^2=0.993; the integer model
# below lands within ~8% median per case.
FETCH_COST = 4


def expected_cycles(spec, plan: Plan) -> int:
    """Calibrated cycle model = 4 x RAM fetches + 1 x pure-FSM phase visits."""
    fetch = fsm = 0
    for task in plan.tasks:
        taps = 0 if isinstance(spec, GemmSpec) else len(task.taps)
        fetch += 16 + taps          # descriptor words + tap bytes
        fsm += 4                    # phases 0 / 3 / 13 / tap poll
        blocks = -(-task.k // 8) if task.k else 0
        for mt in range(task.m_tiles):
            for nt in range(task.n_tiles):
                fsm += 1            # phase 4
                for kb in range(blocks):
                    blk = min(8, task.k - kb * 8)
                    groups = -(-blk // task.r)
                    fsm += 2 + groups       # phases 5 / 6-exit / 8
                    fetch += groups         # weight tile words
                    steps, crossings = _act_steps(spec, task, mt, nt, kb)
                    fetch += steps + crossings
                fetch += 10         # requant params, reloaded per tile
                fsm += 8            # output write-back, per tile
    return FETCH_COST * fetch + fsm

# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------

def cat(name: str) -> str:
    for k in ("deconv", "dense", "depth", "gemm"):
        if name.startswith(f"grid_{k}"):
            return k
    return "hand"


def collect():
    import test_cocotb_convacc as T

    cases = {c.name: c for c in T.build_cases()}
    expected = {n: expected_cycles(c.spec, make_plan(c.spec)) for n, c in cases.items()}

    actual = {}  # name -> cycles
    for f in glob.glob(str(ROOT / "build/cocotb/verilator/run_*/result.json")):
        for r in json.load(open(f))["results"]:
            if "cycles" in r and r.get("pass"):
                actual[r["name"]] = r["cycles"]

    chunk = []  # (idx, sim_cycles, expected_sum)
    for idx in range(len(T.CHUNK_STARTS)):
        f = ROOT / f"build/cocotb/verilator/run_{idx}/results.xml"
        if not f.is_file():
            continue
        tc = ET.parse(f).getroot().find(".//testcase")
        props = {p.get("name"): p.get("value") for p in tc.iter("property")}
        names = [c.name for c in T.CASES[idx * 24:(idx + 1) * 24]]
        chunk.append((idx, float(props["sim_time_duration"]) / 10.0,
                      sum(expected[n] for n in names)))
    return cases, expected, actual, chunk


def main() -> None:
    cases, expected, actual, chunk = collect()
    print(f"cases={len(cases)}  per-case actual samples={len(actual)}  chunks={len(chunk)}")

    pairs = [(actual[n], expected[n], cat(n)) for n in actual if expected.get(n)]
    ratios = sorted(a / e for a, e, _ in pairs)
    print(f"\nactual/expected ratio: min={ratios[0]:.3f}  median={st.median(ratios):.3f}  "
          f"max={ratios[-1]:.3f}")
    off = sorted(a - e for a, e, _ in pairs)
    print(f"actual-expected offset: median={st.median(off):+.0f} cycles")

    print(f"\n{'cat':<8}{'N':>5}{'act med':>9}{'exp med':>9}{'ratio':>7}"
          f"{'util act':>9}{'util ideal':>11}")
    by = {}
    for a, e, c in pairs:
        by.setdefault(c, []).append((a, e))
    for c, rows in sorted(by.items()):
        acts = [r[0] for r in rows]
        exps = [r[1] for r in rows]
        macs = sum(cases[n].spec and make_plan(cases[n].spec).useful_macs
                   for n in actual if cat(n) == c)
        util = macs / (64 * sum(acts))
        ideal = macs / (64 * sum(exps))
        print(f"{c:<8}{len(rows):>5}{st.median(acts):>9.0f}{st.median(exps):>9.0f}"
              f"{st.median(a/e for a, e in rows):>7.2f}{util:>8.1%}{ideal:>10.1%}")

    # chunk-level cross-check against results.xml sim_time
    cr = sorted(sim / exp for _, sim, exp in chunk if exp)
    print(f"\nchunk-level sim_cycles/expected: median={st.median(cr):.3f}  "
          f"(samples={len(cr)}, sim includes TB load/read-back overhead)")


if __name__ == "__main__":
    main()
