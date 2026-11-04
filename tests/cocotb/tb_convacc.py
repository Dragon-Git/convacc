"""One case per simulator process: the reference TB for the generated core.

The images come from the environment (set by tests/test_cocotb_convacc.py) and
the whole protocol -- load, start, wait for done, read back -- lives in
harness.Bench, which is the same code the batch TB runs.
"""

import os
from pathlib import Path

import cocotb

from harness import Bench


@cocotb.test()
async def convacc_case(dut):
    bench = Bench(dut)
    bench.start_clock()

    act = Path(os.environ["CONVACC_ACT"]).read_bytes()
    wgt = Path(os.environ["CONVACC_WGT"]).read_bytes()
    expected = Path(os.environ["CONVACC_EXP"]).read_bytes()
    await bench.case(act, wgt, expected)
