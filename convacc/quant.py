from __future__ import annotations

import numpy as np


def requantize(
    accumulators: np.ndarray,
    bias: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
    zero_point: np.ndarray,
) -> np.ndarray:
    """Per-channel INT32 accumulators to INT8.

    q = round_away_from_zero((acc + bias) * scale / 2**shift) + zero_point
    """
    acc = np.asarray(accumulators, dtype=np.int64)
    bias64 = np.asarray(bias, dtype=np.int64)
    scale64 = np.asarray(scale, dtype=np.int64)
    shift64 = np.asarray(shift, dtype=np.int64)
    zp64 = np.asarray(zero_point, dtype=np.int64)
    if np.any(scale64 <= 0) or np.any(scale64 > 0x7FFFFFFF):
        raise ValueError("scale must be a positive int32")
    if np.any(shift64 < 0) or np.any(shift64 > 31):
        raise ValueError("shift must be in 0..31")
    if np.any(zp64 < -128) or np.any(zp64 > 127):
        raise ValueError("zero_point must fit int8")
    q = (acc + bias64) * scale64
    magnitude = np.abs(q)
    rounded = np.where(shift64 == 0, magnitude, (magnitude + (np.int64(1) << (shift64 - 1))) >> shift64)
    signed = np.where(q >= 0, rounded, -rounded) + zp64
    return np.clip(signed, -128, 127).astype(np.int8)
