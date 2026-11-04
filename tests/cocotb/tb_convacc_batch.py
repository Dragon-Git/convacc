"""Batch cocotb testbench for the convolution accelerator.

Unlike :mod:`tb_convacc` (one simulator process per case), this runs *many*
cases inside a *single* process. The process elaborates the DUT (3 x 128 KB
SRAM) once, so collapsing N cases into one process removes N-1 re-elaborations
and N-1 Python/cocotb/VPI startups -- the dominant fixed cost of the regression.

The case list is passed as a JSON manifest in the ``CONVACC_BATCH`` env var::

    {"cases": [{"name": "...", "act": "...", "wgt": "...", "exp": "..."}, ...]}

Per-case outcomes (and wall time) are written to ``CONVACC_BATCH_RESULT``
(default ``<manifest>.result.json``). If any case fails the test raises so the
harness reports a failure; genuine DUT bugs are never masked.

The protocol itself is in :mod:`harness`; what is left here is the manifest, the
per-case bookkeeping and the incremental result file.
"""

import json
import os
import time
from pathlib import Path

import cocotb

from harness import Bench


async def _run_one(bench, act, wgt, expected):
    """Run one case; returns (ok, reason_or_None)."""
    try:
        await bench.case(act, wgt, expected)
    except AssertionError as exc:
        return False, str(exc)
    return True, None


@cocotb.test()
async def convacc_batch(dut):
    bench = Bench(dut)
    bench.start_clock()

    manifest_path = os.environ["CONVACC_BATCH"]
    manifest = json.loads(Path(manifest_path).read_text())
    entries = manifest["cases"]

    proc_t0 = time.time()
    results = []
    out_path = os.environ.get("CONVACC_BATCH_RESULT", manifest_path + ".result.json")
    for entry in entries:
        name = entry["name"]
        case_t0 = time.time()
        act = Path(entry["act"]).read_bytes()
        wgt = Path(entry["wgt"]).read_bytes()
        expected = Path(entry["exp"]).read_bytes()
        ok, reason = await _run_one(bench, act, wgt, expected)
        case_dt = time.time() - case_t0
        results.append({"name": name, "pass": ok, "reason": reason, "dt": case_dt})
        # Incremental write: if the simulator is killed mid batch, the
        # already-completed cases' outcomes survive so the harness can tell
        # "done & correct" from "interrupted".
        Path(out_path).write_text(json.dumps({
            "n": len(results),
            "passed": sum(1 for r in results if r["pass"]),
            "done": False,
            "results": results,
        }, indent=1))

    total_dt = time.time() - proc_t0

    summary = {
        "n": len(results),
        "passed": sum(1 for r in results if r["pass"]),
        "done": True,
        "total_dt": total_dt,
        "sim_dt": sum(r["dt"] for r in results),
        "startup_elab_dt": total_dt - sum(r["dt"] for r in results),
        "results": results,
    }
    Path(out_path).write_text(json.dumps(summary, indent=1))

    failed = [r for r in results if not r["pass"]]
    if failed:
        head = ", ".join(f"{r['name']}({r['reason']})" for r in failed[:10])
        raise AssertionError(f"{len(failed)}/{len(results)} cases failed: {head}")
