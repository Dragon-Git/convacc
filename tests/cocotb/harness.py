"""What the three cocotb testbenches were each writing for themselves.

tb_convacc, tb_convacc_batch and tb_dbg had a copy of the clock start, the reset
sequence, the image deposit and the output read-back; two of the three had
byte-identical copies of the first two. The only real difference between them is
whether the DUT is reused: a single-case process starts from a zeroed SRAM, a
batch process has to clear the window the next case is about to check.
`Bench` tracks that and needs no flag.

The deposit path writes into the unpacked `logic [63:0] mem[]` array, which
costs no simulation time; the per-cycle write port is kept as a fallback for
simulators that do not expose array elements.
"""

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

SRAM_BYTES = 128 * 1024
TIMEOUT_CYCLES = 5_000_000


def used_words(image: bytes) -> int:
    """Number of leading words worth loading.

    The SRAM macro powers up as all-zero, so only the words the image actually
    needs have to be written. Both tensors are laid out as one contiguous block
    from word 0, so this is simply the highest non-zero word plus one -- loading
    it instead of the whole 128 KB cuts a case from 16384 load cycles to a few
    dozen.
    """
    words = len(image) // 8
    if not words:
        return 0
    hits = np.flatnonzero(np.frombuffer(image, dtype="<u8", count=words))
    return int(hits[-1]) + 1 if hits.size else 0


def mismatch(actual: bytes, expected: bytes, limit: int = 16) -> str:
    """Describe a byte-level difference, with more than the first offset.

    The offset pattern -- which words, which lanes -- is usually what identifies
    the fault, so the detail lists as many as fit rather than stopping at one.
    """
    bad = [(i, want, got) for i, (got, want) in enumerate(zip(actual, expected))
           if got != want]
    detail = " ".join(
        f"@{i}(w{i // 8},l{i % 8}):{want:#04x}->{got:#04x}" for i, want, got in bad[:limit]
    )
    return f"{len(bad)} mismatched bytes (len {len(actual)} vs {len(expected)}): {detail}"


class Bench:
    """Drives the DUT: reset, load two images, run one descriptor, read back."""

    def __init__(self, dut):
        self.dut = dut
        # High-water mark of words written so far, shared by the two tensors:
        # the tail left by a longer previous image must be zeroed before a
        # shorter one is loaded, or the engine picks up stale data.
        self._high_water = 0

    def start_clock(self, period_ns: int = 10) -> None:
        cocotb.start_soon(Clock(self.dut.clk, period_ns, unit="ns").start(start_high=False))

    async def reset(self) -> None:
        dut = self.dut
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

    async def load(self, act: bytes, wgt: bytes, out_words: int) -> None:
        """Load both tensors, then clear the output window this case checks."""
        for image in (act, wgt):
            if len(image) != SRAM_BYTES:
                raise AssertionError("SRAM images must be 128KB")

        dut = self.dut
        try:
            # Deposit straight into the SRAM arrays: one word per clock made a
            # case cost up to 16384 cycles of pure loading, while a direct write
            # costs no simulation time at all.
            self._deposit(dut.u_act_sram.mem, act)
            self._deposit(dut.u_wgt_sram.mem, wgt)
            dut.act_we.value = 0
            dut.act_wstrb.value = 0
            dut.wgt_we.value = 0
            dut.wgt_wstrb.value = 0
            await RisingEdge(dut.clk)
        except Exception:  # noqa: BLE001
            await self._load_through_port(act, wgt)

        # Without this a case whose engine skips a word would silently inherit
        # that word from the previous case and pass.
        try:
            for word in range(out_words):
                dut.u_out_sram.mem[word].value = 0
        except Exception as exc:  # noqa: BLE001
            dut._log.warning("could not clear output window: %r", exc)

        dut.act_we.value = 0
        dut.wgt_we.value = 0
        dut.act_wstrb.value = 0
        dut.wgt_wstrb.value = 0
        dut.out_re.value = 0
        dut.out_raddr.value = 0
        await RisingEdge(dut.clk)

    async def run(self) -> int:
        """Pulse start and wait for done; returns the cycle count."""
        dut = self.dut
        dut.start.value = 1
        dut.start_desc.value = 0
        await RisingEdge(dut.clk)
        dut.start.value = 0

        for cycle in range(TIMEOUT_CYCLES):
            await RisingEdge(dut.clk)
            if int(dut.done.value):
                break
        else:
            raise AssertionError("simulation timeout")
        if int(dut.error.value):
            raise AssertionError("controller error")
        return cycle

    async def read(self, nbytes: int) -> bytes:
        """Clock the output window out one word at a time, truncated to nbytes."""
        dut = self.dut
        actual = bytearray()
        for word in range((nbytes + 7) // 8):
            dut.out_re.value = 1
            dut.out_raddr.value = word
            await Timer(1, unit="ns")
            actual.extend(int(dut.out_rdata.value).to_bytes(8, "little"))
        dut.out_re.value = 0
        return bytes(actual[:nbytes])

    async def case(self, act: bytes, wgt: bytes, expected: bytes) -> None:
        """Run one case end to end; raises AssertionError with the mismatch."""
        await self.reset()
        await self.load(act, wgt, (len(expected) + 7) // 8)
        await self.run()
        actual = await self.read(len(expected))
        if actual != expected:
            raise AssertionError(mismatch(actual, expected))

    # --- internals ------------------------------------------------------
    def _deposit(self, mem_handle, image: bytes) -> None:
        words = used_words(image)
        for word in range(words, self._high_water):
            mem_handle[word].value = 0
        for word in range(words):
            mem_handle[word].value = int.from_bytes(image[word * 8:word * 8 + 8], "little")
        self._high_water = max(self._high_water, words)

    async def _load_through_port(self, act: bytes, wgt: bytes) -> None:
        """Fallback for simulators that do not expose unpacked array elements.

        The loop covers the high-water mark as well, so a longer previous image
        cannot leak into this one.
        """
        dut = self.dut
        act_words = used_words(act)
        wgt_words = used_words(wgt)
        limit = max(act_words, wgt_words, self._high_water)
        for word in range(limit):
            dut.act_we.value = 1
            dut.act_waddr.value = word
            dut.act_wdata.value = (int.from_bytes(act[word * 8:word * 8 + 8], "little")
                                   if word < act_words else 0)
            dut.act_wstrb.value = 0xFF
            dut.wgt_we.value = 1
            dut.wgt_waddr.value = word
            dut.wgt_wdata.value = (int.from_bytes(wgt[word * 8:word * 8 + 8], "little")
                                   if word < wgt_words else 0)
            dut.wgt_wstrb.value = 0xFF
            await RisingEdge(dut.clk)
        self._high_water = max(self._high_water, limit)
