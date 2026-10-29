"""Bounded debug TB: run N cycles and dump the generated core's handshake."""
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


@cocotb.test()
async def dbg(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.start.value = 0
    dut.start_desc.value = 0
    for n in ("act", "wgt"):
        getattr(dut, n + "_we").value = 0
        getattr(dut, n + "_wstrb").value = 0
        getattr(dut, n + "_waddr").value = 0
        getattr(dut, n + "_wdata").value = 0
    dut.out_re.value = 0
    dut.out_raddr.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    case = Path(os.environ["DBG_CASE"])
    act = (case / "activation.bin").read_bytes()
    wgt = (case / "weight.bin").read_bytes()
    for word in range(0, len(act), 8):
        dut.act_we.value = 1
        dut.act_waddr.value = word // 8
        dut.act_wdata.value = int.from_bytes(act[word:word + 8], "little")
        dut.act_wstrb.value = 0xFF
        dut.wgt_we.value = 1
        dut.wgt_waddr.value = word // 8
        dut.wgt_wdata.value = int.from_bytes(wgt[word:word + 8], "little")
        dut.wgt_wstrb.value = 0xFF
        await RisingEdge(dut.clk)
    dut.act_we.value = 0
    dut.wgt_we.value = 0
    dut.act_wstrb.value = 0
    dut.wgt_wstrb.value = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    acts = 0
    done_at = -1
    for i in range(300000):
        await RisingEdge(dut.clk)
        if int(dut.core_act_req_vld.value):
            acts += 1
        if int(dut.core_done_vld.value):
            done_at = i
            break
    dut._log.info("RESULT done_at=%d act_reads=%d out_writes_ok", done_at, acts)
