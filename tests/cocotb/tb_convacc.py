import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


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



def _deposit(mem_handle, image: bytes) -> None:
    """Write an image straight into an unpacked `logic [63:0] mem [...]` array."""
    for word in range(_used_words(image)):
        mem_handle[word].value = int.from_bytes(image[word * 8:word * 8 + 8], "little")


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


@cocotb.test()
async def convacc_case(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start(start_high=False))
    await reset_dut(dut)

    act = Path(os.environ["CONVACC_ACT"]).read_bytes()
    wgt = Path(os.environ["CONVACC_WGT"]).read_bytes()
    expected = Path(os.environ["CONVACC_EXP"]).read_bytes()
    if len(act) != 128 * 1024 or len(wgt) != 128 * 1024:
        raise AssertionError("SRAM images must be 128KB")

    # Deposit straight into the SRAM arrays -- see tb_convacc_batch._deposit.
    try:
        _deposit(dut.u_act_sram.mem, act)
        _deposit(dut.u_wgt_sram.mem, wgt)
        dut.act_we.value = 0
        dut.act_wstrb.value = 0
        dut.wgt_we.value = 0
        dut.wgt_wstrb.value = 0
        await RisingEdge(dut.clk)
    except Exception:  # noqa: BLE001
        act_words = _used_words(act)
        wgt_words = _used_words(wgt)
        for word in range(max(act_words, wgt_words)):
            if word < act_words:
                dut.act_we.value = 1
                dut.act_waddr.value = word
                dut.act_wdata.value = int.from_bytes(act[word * 8:word * 8 + 8], "little")
                dut.act_wstrb.value = 0xFF
            else:
                dut.act_we.value = 0
            if word < wgt_words:
                dut.wgt_we.value = 1
                dut.wgt_waddr.value = word
                dut.wgt_wdata.value = int.from_bytes(wgt[word * 8:word * 8 + 8], "little")
                dut.wgt_wstrb.value = 0xFF
            else:
                dut.wgt_we.value = 0
            await RisingEdge(dut.clk)
    dut.act_we.value = 0
    dut.act_wstrb.value = 0
    dut.wgt_we.value = 0
    dut.wgt_wstrb.value = 0

    dut.start.value = 1
    dut.start_desc.value = 0
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for cycle in range(5_000_000):
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            break
    else:
        raise AssertionError("simulation timeout")
    if int(dut.error.value):
        raise AssertionError("controller error")

    actual = bytearray()
    for word in range((len(expected) + 7) // 8):
        dut.out_re.value = 1
        dut.out_raddr.value = word
        await Timer(1, unit="ns")
        value = int(dut.out_rdata.value)
        actual.extend(value.to_bytes(8, "little"))
    actual = actual[:len(expected)]

    if actual != expected:
        # Report every mismatch, not just the first: the offset pattern (which
        # words / lanes are wrong) is usually what identifies the fault.
        bad = [(i, want, got) for i, (got, want) in enumerate(zip(actual, expected))
               if got != want]
        detail = " ".join(
            f"@{i}(w{i // 8},l{i % 8}):{want:#04x}->{got:#04x}" for i, want, got in bad[:16]
        )
        raise AssertionError(
            f"{len(bad)} mismatched bytes (len {len(actual)} vs {len(expected)}): {detail}"
        )
