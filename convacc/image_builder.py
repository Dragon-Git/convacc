from __future__ import annotations

from pathlib import Path

import numpy as np

from .image import build_image
from .packer import pack_weights
from .quant import requantize
from .reference import layout_output, reference_for_packed
from .spec import GemmSpec


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
