"""The three 128 KB SRAM images, and the whole case built around them.

`build_image` lays out the descriptors, taps, packed weights and requant
parameters; `make_image` is the end-to-end entry point (pack -> image -> run the
reference -> requantise -> lay out). Mirrored in C++ by convacc/image.h and
convacc::build_case.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .packer import PackedPlan, pack_weights
from .reference import layout_output, output_geometry, reference_for_packed, requantize
from .spec import GemmSpec, SRAM_BYTES

WORDS = SRAM_BYTES // 8
HEADER_WORDS = 16
PARAM_WORDS_PER_TILE = 10


class WordMemory:
    def __init__(self, fill: int = 0):
        self.words = [fill & 0xFFFFFFFFFFFFFFFF] * WORDS
        self.next_word = 0

    def alloc(self, count: int) -> int:
        if self.next_word + count > WORDS:
            raise ValueError("SRAM image exceeds 128KB")
        address = self.next_word
        self.next_word += count
        return address

    def write_bytes(self, word_address: int, payload: bytes) -> None:
        if word_address * 8 + len(payload) > SRAM_BYTES:
            raise ValueError("SRAM image write exceeds 128KB")
        padded = payload + bytes((-len(payload)) % 8)
        for i in range(0, len(padded), 8):
            self.words[word_address + i // 8] = int.from_bytes(padded[i : i + 8], "little")

    def set(self, address: int, value: int) -> None:
        self.words[address] = value & 0xFFFFFFFFFFFFFFFF

    def to_bytes(self) -> bytes:
        return b"".join(value.to_bytes(8, "little") for value in self.words)


def _u64(*fields: tuple[int, int]) -> int:
    value = 0
    for field, shift in fields:
        value |= (int(field) & ((1 << (64 - shift)) - 1)) << shift
    return value


def _bits(word: int, shift: int, width: int) -> int:
    return (word >> shift) & ((1 << width) - 1)


@dataclass(frozen=True)
class Descriptor:
    """One descriptor block, read back out of an image.

    The counterpart of the `memory.set(desc_address + n, ...)` block below; the
    field offsets exist in one place only because both directions are here.
    Everything is the raw stored value -- callers apply their own defaults.
    """

    flags: int
    m: int
    n: int
    k: int
    batches: int
    mh: int
    mw: int
    k_groups: int
    o: int
    ci_count: int
    ci_base: int
    co_base: int
    co_per_group: int
    hi: int
    wi: int
    ho: int
    wo: int
    ci: int
    co: int
    sh: int
    sw: int
    dh: int
    dw: int
    ph: int
    pw: int
    phase_h: int
    phase_w: int
    out_step_h: int
    out_step_w: int
    act_base: int
    out_base: int
    w_base: int
    param_base: int
    tap_base: int
    taps: int
    next_desc: int

    @property
    def gemm(self) -> bool:
        return bool(self.flags & 1)

    @property
    def channelwise(self) -> bool:
        return bool(self.flags & 2)

    @property
    def deconv(self) -> bool:
        return bool(self.flags & 4)

    @property
    def log_r(self) -> int:
        # The writer stores log2(r) above the three flag bits without masking it
        # to the two bits r needs; read it back at the same width.
        return (self.flags >> 3) & 0xFF

    @property
    def r(self) -> int:
        return 1 << self.log_r

    @property
    def m_tiles(self) -> int:
        return (self.m + 7) // 8

    @property
    def n_tiles(self) -> int:
        return (self.n + self.o - 1) // self.o if self.o else 0


def decode_descriptor(data, start: int = 0) -> Descriptor:
    """Read the descriptor at word `start` (see `build_image` for the layout).

    `data` is either an SRAM image (bytes) or an already-unpacked word list.
    """
    if isinstance(data, (bytes, bytearray)):
        window = bytes(data)[start * 8:(start + HEADER_WORDS) * 8]
        w = list(struct.unpack("<%dQ" % (len(window) // 8), window))
    else:
        w = list(data)[start:start + HEADER_WORDS]
    if len(w) != HEADER_WORDS:
        raise ValueError(f"descriptor needs {HEADER_WORDS} words, got {len(w)}")
    return Descriptor(
        flags=w[0],
        m=_bits(w[1], 0, 32), n=_bits(w[1], 32, 32),
        k=_bits(w[2], 0, 32), batches=_bits(w[2], 32, 32),
        mh=_bits(w[3], 0, 16), mw=_bits(w[3], 16, 16),
        k_groups=_bits(w[4], 0, 32), o=_bits(w[4], 32, 32),
        ci_count=_bits(w[5], 0, 16), ci_base=_bits(w[5], 16, 16),
        co_base=_bits(w[5], 32, 16), co_per_group=_bits(w[5], 48, 16),
        hi=_bits(w[6], 0, 16), wi=_bits(w[6], 16, 16),
        ho=_bits(w[7], 0, 16), wo=_bits(w[7], 16, 16),
        ci=_bits(w[8], 0, 16), co=_bits(w[8], 16, 16),
        sh=_bits(w[9], 0, 16), sw=_bits(w[9], 16, 16),
        dh=_bits(w[9], 32, 16), dw=_bits(w[9], 48, 16),
        ph=_bits(w[10], 0, 16), pw=_bits(w[10], 16, 16),
        phase_h=_bits(w[10], 32, 16), phase_w=_bits(w[10], 48, 16),
        out_step_h=_bits(w[11], 0, 16), out_step_w=_bits(w[11], 16, 16),
        act_base=_bits(w[12], 0, 32), out_base=_bits(w[12], 32, 32),
        w_base=_bits(w[13], 0, 32), param_base=_bits(w[13], 32, 32),
        tap_base=_bits(w[14], 0, 32), taps=_bits(w[14], 32, 16),
        next_desc=_bits(w[15], 0, 32),
    )


@dataclass(frozen=True)
class SramImage:
    activation: bytes
    weight: bytes
    output: bytes
    descriptor_start: int
    manifest: dict
    output_words: int = 0


def build_image(
    packed: PackedPlan,
    bias: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
    zero_point: np.ndarray,
    activation: bytes,
    activation_base: int = 0,
    output_base: int = 0,
) -> SramImage:
    width = packed.spec.n if isinstance(packed.spec, GemmSpec) else packed.spec.co
    for name, value in (("bias", bias), ("scale", scale), ("shift", shift), ("zero_point", zero_point)):
        if value.shape != (width,):
            raise ValueError(f"{name} must have shape ({width},)")
    memory = WordMemory()
    placements = []
    for packed_task in packed.tasks:
        task = packed_task.task
        desc_address = memory.alloc(HEADER_WORDS)
        tap_bytes = bytes((kh * 16 + kw for kh, kw in task.taps))
        tap_address = memory.alloc((len(tap_bytes) + 7) // 8)
        memory.write_bytes(tap_address, tap_bytes)
        weight_address = memory.alloc((packed_task.weights.size + 7) // 8)
        memory.write_bytes(weight_address, packed_task.weights.tobytes(order="C"))
        param_address = memory.alloc(task.n_tiles * PARAM_WORDS_PER_TILE)
        for nt in range(task.n_tiles):
            base = param_address + nt * PARAM_WORDS_PER_TILE
            co_start = task.co_base + nt * task.o
            for lane in range(8):
                co = co_start + lane
                if co < task.co_base + task.n:
                    bias_v = int(bias[co])
                    scale_v = int(scale[co])
                    shift_v = int(shift[co]) & 0xFF
                    zp_v = int(zero_point[co]) & 0xFF
                else:
                    bias_v, scale_v, shift_v, zp_v = 0, 1, 0, 0
                word = base + lane // 2
                half = lane % 2
                if half == 0:
                    memory.set(word, bias_v & 0xFFFFFFFF)
                else:
                    memory.set(word, memory.words[word] | ((bias_v & 0xFFFFFFFF) << 32))
                word = base + 4 + lane // 2
                if half == 0:
                    memory.set(word, scale_v & 0xFFFFFFFF)
                else:
                    memory.set(word, memory.words[word] | ((scale_v & 0xFFFFFFFF) << 32))
                memory.set(base + 8, memory.words[base + 8] | (shift_v << (lane * 8)))
                memory.set(base + 9, memory.words[base + 9] | (zp_v << (lane * 8)))
        placements.append((packed_task, desc_address, tap_address, weight_address, param_address))

    for index, (packed_task, desc_address, tap_address, weight_address, param_address) in enumerate(placements):
        task = packed_task.task
        next_desc = placements[index + 1][1] if index + 1 < len(placements) else 0
        kind = task.mode
        flags = (
            int(isinstance(packed.spec, GemmSpec))
            | (int(kind == "channelwise") << 1)
            | (int(getattr(packed.spec, "kind", "") == "deconv2d") << 2)
            | ((task.r.bit_length() - 1) << 3)
        )
        memory.set(desc_address + 0, flags)
        memory.set(desc_address + 1, _u64((task.m, 0), (task.n, 32)))
        memory.set(desc_address + 2, _u64((task.k, 0), (task.batches, 32)))
        memory.set(desc_address + 3, _u64((task.mh, 0), (task.mw, 16)))
        memory.set(desc_address + 4, _u64((task.k_groups, 0), (task.o, 32)))
        memory.set(desc_address + 5, _u64(
            (task.ci_count, 0), (task.ci_base, 16), (task.co_base, 32),
            (getattr(packed.spec, "co_per_group", 1), 48),
        ))
        memory.set(desc_address + 6, _u64((getattr(packed.spec, "hi", 1), 0), (getattr(packed.spec, "wi", 1), 16)))
        memory.set(desc_address + 7, _u64((getattr(packed.spec, "ho", 1), 0), (getattr(packed.spec, "wo", 1), 16)))
        memory.set(desc_address + 8, _u64((getattr(packed.spec, "ci", 1), 0), (getattr(packed.spec, "co", 1), 16)))
        memory.set(desc_address + 9, _u64(
            (getattr(packed.spec, "sh", 1), 0), (getattr(packed.spec, "sw", 1), 16),
            (getattr(packed.spec, "dh", 1), 32), (getattr(packed.spec, "dw", 1), 48),
        ))
        memory.set(desc_address + 10, _u64(
            (getattr(packed.spec, "ph", 0), 0), (getattr(packed.spec, "pw", 0), 16),
            (task.phase_h, 32), (task.phase_w, 48),
        ))
        memory.set(desc_address + 11, _u64((task.out_step_h, 0), (task.out_step_w, 16)))
        memory.set(desc_address + 12, _u64((activation_base, 0), (output_base, 32)))
        memory.set(desc_address + 13, _u64((weight_address, 0), (param_address, 32)))
        memory.set(desc_address + 14, _u64((tap_address, 0), (len(task.taps), 32)))
        memory.set(desc_address + 15, next_desc)

    positions, groups, n_tiles = output_geometry(packed)
    if output_base % 8:
        raise ValueError("output_base must be 64-bit aligned")
    output_words = positions * groups * n_tiles
    if output_base + output_words * 8 > SRAM_BYTES:
        raise ValueError(
            f"output image needs {output_words * 8} bytes at {output_base}; SRAM is {SRAM_BYTES}"
        )

    manifest = {
        "descriptor_start": 0,
        "weight_bytes_used": memory.next_word * 8,
        "output": {
            "base_byte": output_base,
            "positions": positions,
            "groups": groups,
            "n_tiles": n_tiles,
            "words": output_words,
            "bytes_per_position": groups * n_tiles * 8,
        },
        "tasks": [
            {
                "name": packed_task.task.name,
                "descriptor_word": desc_address,
                "tap_word": tap_address,
                "weight_word": weight_address,
                "param_word": param_address,
                "tap_count": len(packed_task.task.taps),
                "tile_count": packed_task.task.n_tiles,
                "r": packed_task.task.r,
                "o": packed_task.task.o,
            }
            for packed_task, desc_address, tap_address, weight_address, param_address in placements
        ],
    }
    if len(activation) > SRAM_BYTES:
        raise ValueError("activation image exceeds 128KB")
    return SramImage(
        activation=bytes(activation) + bytes(SRAM_BYTES - len(activation)),
        weight=memory.to_bytes(),
        output=bytes(SRAM_BYTES),
        descriptor_start=0,
        manifest=manifest,
        output_words=output_words,
    )


def write_sram_hex(path: Path, payload: bytes) -> None:
    if len(payload) != 128 * 1024:
        raise ValueError("SRAM image must be exactly 128KB")
    path.write_text("\n".join(f"{int.from_bytes(payload[i:i+8], 'little'):016x}" for i in range(0, len(payload), 8)) + "\n")


def make_image(
    spec,
    weights: np.ndarray,
    layout: str,
    activations: np.ndarray,
    bias: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    shift: np.ndarray | None = None,
    zero_point: np.ndarray | None = None,
    activation_base: int = 0,
    output_base: int = 0,
):
    packed = pack_weights(spec, weights, layout)
    width = spec.n if isinstance(spec, GemmSpec) else spec.co
    bias = np.zeros(width, dtype=np.int32) if bias is None else np.asarray(bias, dtype=np.int32)
    scale = np.ones(width, dtype=np.int32) if scale is None else np.asarray(scale, dtype=np.int32)
    shift = np.zeros(width, dtype=np.uint8) if shift is None else np.asarray(shift, dtype=np.uint8)
    zero_point = np.zeros(width, dtype=np.int8) if zero_point is None else np.asarray(zero_point, dtype=np.int8)
    image = build_image(
        packed, bias, scale, shift, zero_point, np.ascontiguousarray(activations).tobytes(),
        activation_base=activation_base, output_base=output_base,
    )
    accumulators = reference_for_packed(packed, activations, weights, layout)
    requanted = requantize(accumulators, bias, scale, shift, zero_point)
    return packed, image, layout_output(packed, requanted)
