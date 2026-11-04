"""Bounded debug TB: run N cycles and dump the generated core's handshake.

Not part of the regression -- it exists to answer "where did it stop" on one
case, using the same reset and clock as the real TBs.
"""

import os
from pathlib import Path

import cocotb
from cocotb.triggers import RisingEdge

from harness import Bench


@cocotb.test()
async def dbg(dut):
    bench = Bench(dut)
    bench.start_clock()
    await bench.reset()

    case = Path(os.environ["DBG_CASE"])
    act = (case / "activation.bin").read_bytes()
    wgt = (case / "weight.bin").read_bytes()
    # The whole image goes through the write port here, unlike the regression
    # TBs: this one is also used against builds whose memory does not read back
    # as zeros, so an untouched tail cannot be assumed to be zero.
    for word in range(0, len(act), 8):
        dut.act_we.value = 1
        dut.act_waddr.value = word
        dut.act_wdata.value = int.from_bytes(act[word * 8:word * 8 + 8], "little")
        dut.act_wstrb.value = 0xFF
        dut.wgt_we.value = 1
        dut.wgt_waddr.value = word
        dut.wgt_wdata.value = int.from_bytes(wgt[word * 8:word * 8 + 8], "little")
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
