from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .packer import PackedPlan
from .reference import output_geometry
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
