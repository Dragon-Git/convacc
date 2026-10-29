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

Cases are independent: every case resets the controller *and* clears the output
window it is about to check. Without the clear, a case whose engine skips a
word would silently inherit that word from the previous case and pass.
"""

import json
import os
import time
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset_dut(dut):
    dut.rst.value = 1
    dut.start.value = 0
    dut.start_desc.value = 0
    dut.act_we.value = 0
    dut.wgt_we.value = 0
    dut.act_wstrb.value = 0
    dut.wgt_wstrb.value = 0
    dut.out_re.value = 0
    dut.out_raddr.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


def _used_words(image: bytes) -> int:
    """Number of leading words worth loading.

    The SRAM macro powers up as all-zero, so only the words the image actually
    needs have to be written. Both tensors are laid out as one contiguous block
    from word 0, so this is simply the highest non-zero word plus one -- loading
    it instead of the whole 128 KB cuts a case from 16384 load cycles to a few
    dozen.
    """
    words = len(image) // 8
    for i in range(words - 1, -1, -1):
        if image[i * 8:(i + 1) * 8] != b"\x00" * 8:
            return i + 1
    return 0


# High-water mark of words written so far in this process, shared by the
# tensors. Clearing back down to each image's own length keeps cases
# independent without re-zeroing the whole 128 KB every time.
_high_water = {"n": 0}


def _deposit(mem_handle, image: bytes) -> None:
    """Write an image into an unpacked `logic [63:0] mem [...]` array.

    Only the words the image actually needs are written, but the tail left by a
    *longer* previous image must be zeroed first: the engine may legitimately
    read words that are zero in this image, and without the clear it would pick
    up whatever the previous case left there.
    """
    words = _used_words(image)
    for word in range(words, _high_water["n"]):
        mem_handle[word].value = 0
    for word in range(words):
        mem_handle[word].value = int.from_bytes(image[word * 8:word * 8 + 8], "little")
    if words > _high_water["n"]:
        _high_water["n"] = words


def _clear(mem_handle, words: int) -> None:
    """Zero the first `words` entries of an unpacked SRAM array (no sim time)."""
    for word in range(words):
        mem_handle[word].value = 0


async def _run_one(dut, name, act, wgt, expected):
    """Load one case into the DUT, run it, return (ok, reason_or_None)."""
    if len(act) != 128 * 1024 or len(wgt) != 128 * 1024:
        return False, "SRAM images must be 128KB"

    await reset_dut(dut)

    check_words = (len(expected) + 7) // 8
    try:
        # Deposit straight into the SRAM arrays: one word per clock made a case
        # cost up to 16384 cycles of pure loading, while a direct write costs no
        # simulation time at all.
        _deposit(dut.u_act_sram.mem, act)
        _deposit(dut.u_wgt_sram.mem, wgt)
        dut.act_we.value = 0
        dut.act_wstrb.value = 0
        dut.wgt_we.value = 0
        dut.wgt_wstrb.value = 0
        await RisingEdge(dut.clk)
    except Exception:  # noqa: BLE001
        # Fallback for simulators that do not expose unpacked array elements.
        # The loop covers the high-water mark too so a longer previous image
        # cannot leak into this one (see _deposit).
        act_words = _used_words(act)
        wgt_words = _used_words(wgt)
        limit = max(act_words, wgt_words, _high_water["n"])
        for word in range(limit):
            if word < act_words:
                dut.act_we.value = 1
                dut.act_waddr.value = word
                dut.act_wdata.value = int.from_bytes(act[word * 8:word * 8 + 8], "little")
                dut.act_wstrb.value = 0xFF
            else:
                dut.act_we.value = 1
                dut.act_waddr.value = word
                dut.act_wdata.value = 0
                dut.act_wstrb.value = 0xFF
            if word < wgt_words:
                dut.wgt_we.value = 1
                dut.wgt_waddr.value = word
                dut.wgt_wdata.value = int.from_bytes(wgt[word * 8:word * 8 + 8], "little")
                dut.wgt_wstrb.value = 0xFF
            else:
                dut.wgt_we.value = 1
                dut.wgt_waddr.value = word
                dut.wgt_wdata.value = 0
                dut.wgt_wstrb.value = 0xFF
            await RisingEdge(dut.clk)
        if limit > _high_water["n"]:
            _high_water["n"] = limit

    # Clear the output window the case is about to be checked on, so cases stay
    # independent: a word the engine never writes must read back as zero rather
    # than inherit whatever the previous case left behind. The out SRAM has no
    # host write port (only the core writes it), so this goes through the array.
    try:
        _clear(dut.u_out_sram.mem, check_words)
    except Exception as exc:  # noqa: BLE001
        dut._log.warning("could not clear output window: %r", exc)

    dut.act_we.value = 0
    dut.wgt_we.value = 0
    dut.act_wstrb.value = 0
    dut.wgt_wstrb.value = 0
    dut.out_re.value = 0
    dut.out_raddr.value = 0
    await RisingEdge(dut.clk)

    dut.start.value = 1
    dut.start_desc.value = 0
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for cycle in range(5_000_000):
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            break
    else:
        return False, "simulation timeout"

    if int(dut.error.value):
        return False, "controller error"

    actual = bytearray()
    for word in range(check_words):
        dut.out_re.value = 1
        dut.out_raddr.value = word
        await Timer(1, unit="ns")
        value = int(dut.out_rdata.value)
        actual.extend(value.to_bytes(8, "little"))
    dut.out_re.value = 0
    actual = actual[:len(expected)]

    if actual != expected:
        # Report every mismatch, not just the first: the offset pattern (which
        # words / lanes are wrong) is usually what identifies the fault.
        bad = [(i, want, got) for i, (got, want) in enumerate(zip(actual, expected))
               if got != want]
        detail = " ".join(
            f"@{i}(w{i // 8},l{i % 8}):{want:#04x}->{got:#04x}" for i, want, got in bad[:16]
        )
        return False, f"{len(bad)} mismatched bytes: {detail}"
    return True, None


@cocotb.test()
async def convacc_batch(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start(start_high=False))

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
        ok, reason = await _run_one(dut, name, act, wgt, expected)
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
