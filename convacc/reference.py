from __future__ import annotations

import numpy as np

from .packer import PackedPlan, canonical_conv_weights
from .spec import ConvSpec, GemmSpec


def conv_reference(spec: ConvSpec, activations: np.ndarray, canonical_weights: np.ndarray) -> np.ndarray:
    if spec.kind == "deconv2d":
        return _deconv_reference(spec, activations, canonical_weights)
    output = np.zeros((spec.n, spec.ho, spec.wo, spec.co), dtype=np.int32)
    for batch in range(spec.n):
        for oh in range(spec.ho):
            for ow in range(spec.wo):
                for kh, kw in ((kh, kw) for kh in range(spec.kh) for kw in range(spec.kw)):
                    ih = oh * spec.sh + kh * spec.dh - spec.ph
                    iw = ow * spec.sw + kw * spec.dw - spec.pw
                    if not (0 <= ih < spec.hi and 0 <= iw < spec.wi):
                        continue
                    for group in range(spec.groups):
                        ci0 = group * spec.ci_per_group
                        co0 = group * spec.co_per_group
                        x = activations[batch, ih, iw, ci0 : ci0 + spec.ci_per_group].astype(np.int32)
                        w = canonical_weights[kh, kw, group].astype(np.int32)
                        output[batch, oh, ow, co0 : co0 + spec.co_per_group] += x @ w
    return output


def _deconv_reference(spec: ConvSpec, activations: np.ndarray, canonical_weights: np.ndarray) -> np.ndarray:
    output = np.zeros((spec.n, spec.ho, spec.wo, spec.co), dtype=np.int32)
    for batch in range(spec.n):
        for ih in range(spec.hi):
            for iw in range(spec.wi):
                for kh in range(spec.kh):
                    for kw in range(spec.kw):
                        oh = ih * spec.sh + kh - spec.ph
                        ow = iw * spec.sw + kw - spec.pw
                        if 0 <= oh < spec.ho and 0 <= ow < spec.wo:
                            x = activations[batch, ih, iw].astype(np.int32)
                            w = canonical_weights[kh, kw, 0].astype(np.int32)
                            output[batch, oh, ow] += x @ w
    return output


def _output_coords(spec: ConvSpec, task, m_index: int) -> tuple[int, int, int]:
    per_batch = task.mh * task.mw
    batch = m_index // per_batch
    rem = m_index % per_batch
    oh = task.phase_h + (rem // task.mw) * task.out_step_h
    ow = task.phase_w + (rem % task.mw) * task.out_step_w
    return batch, oh, ow


def _dense_activations(spec: ConvSpec, task, r: int, activations: np.ndarray) -> np.ndarray:
    kh, kw = task.taps[r // task.ci_count]
    ci = task.ci_base + r % task.ci_count
    result = np.zeros(task.m, dtype=np.int32)
    for m_index in range(task.m):
        batch, oh, ow = _output_coords(spec, task, m_index)
        if spec.kind == "deconv2d":
            ih_num = oh + spec.ph - kh
            iw_num = ow + spec.pw - kw
            if ih_num % spec.sh or iw_num % spec.sw:
                continue
            ih = ih_num // spec.sh
            iw = iw_num // spec.sw
        else:
            ih = oh * spec.sh + kh * spec.dh - spec.ph
            iw = ow * spec.sw + kw * spec.dw - spec.pw
        if 0 <= ih < spec.hi and 0 <= iw < spec.wi:
            result[m_index] = int(activations[batch, ih, iw, ci])
    return result


def _channelwise_activations(
    spec: ConvSpec, task, tap: int, co: int, activations: np.ndarray
) -> np.ndarray:
    kh, kw = task.taps[tap]
    ci = co // spec.co_per_group
    result = np.zeros(task.m, dtype=np.int32)
    for m_index in range(task.m):
        batch, oh, ow = _output_coords(spec, task, m_index)
        ih = oh * spec.sh + kh * spec.dh - spec.ph
        iw = ow * spec.sw + kw * spec.dw - spec.pw
        if 0 <= ih < spec.hi and 0 <= iw < spec.wi:
            result[m_index] = int(activations[batch, ih, iw, ci])
    return result


def _execute_packed_conv(packed: PackedPlan, activations: np.ndarray) -> np.ndarray:
    spec = packed.spec
    assert isinstance(spec, ConvSpec)
    output = np.zeros((spec.n, spec.ho, spec.wo, spec.co), dtype=np.int32)
    for packed_task in packed.tasks:
        task = packed_task.task
        for nt in range(task.n_tiles):
            width = min(task.o, task.n - nt * task.o)
            acc = np.zeros((task.m, task.o), dtype=np.int32)
            for kg in range(task.k_groups):
                for within in range(8):
                    o = within // task.r
                    rr = within % task.r
                    reduction = kg * task.r + rr
                    if o >= width or reduction >= task.k:
                        continue
                    weight = int(packed_task.weights[nt, kg, within])
                    if task.mode == "channelwise":
                        co = nt * task.o + o
                        acts = _channelwise_activations(spec, task, reduction, co, activations)
                    else:
                        acts = _dense_activations(spec, task, reduction, activations)
                    acc[:, o] += acts * weight
            for m_index in range(task.m):
                batch, oh, ow = _output_coords(spec, task, m_index)
                co0 = task.co_base + nt * task.o
                output[batch, oh, ow, co0 : co0 + width] = acc[m_index, :width]
    return output


def execute_packed(packed: PackedPlan, activations: np.ndarray) -> np.ndarray:
    if activations.dtype != np.int8:
        raise ValueError("activations must be int8")
    if isinstance(packed.spec, GemmSpec):
        spec = packed.spec
        task = packed.tasks[0]
        result = np.zeros((spec.m, spec.n), dtype=np.int32)
        for nt in range(task.task.n_tiles):
            width = min(task.task.o, spec.n - nt * task.task.o)
            acc = np.zeros((spec.m, task.task.o), dtype=np.int32)
            for kg in range(task.task.k_groups):
                for within in range(8):
                    o = within // task.task.r
                    rr = within % task.task.r
                    reduction = kg * task.task.r + rr
                    if o < width and reduction < task.task.k:
                        acc[:, o] += (
                            activations[:, reduction].astype(np.int32)
                            * int(task.weights[nt, kg, within])
                        )
            result[:, nt * task.task.o : nt * task.task.o + width] = acc[:, :width]
        return result
    return _execute_packed_conv(packed, activations)


def output_geometry(packed: PackedPlan) -> tuple[int, int, int]:
    """(positions, groups, n_tiles) of the engine's output layout.

    Every output position owns `groups * n_tiles` 64-bit words: one per
    output-channel tile, per channel group. Every tile word is stored whole, so
    lanes past the last real channel are zero padding.
    """
    tiles = {(task.task.o, task.task.n_tiles) for task in packed.tasks}
    if len(tiles) != 1:
        raise ValueError(
            "all tasks must share o/n_tiles so the output layout stays addressable: "
            f"got {sorted(tiles)}"
        )
    _, n_tiles = tiles.pop()
    spec = packed.spec
    if isinstance(spec, GemmSpec):
        return spec.m, 1, n_tiles
    if spec.channelwise:
        return spec.n * spec.ho * spec.wo, 1, n_tiles
    return spec.n * spec.ho * spec.wo, spec.co // spec.co_per_group, n_tiles


def layout_output(packed: PackedPlan, result: np.ndarray) -> np.ndarray:
    """Flatten a semantic result into the engine's 8-byte-aligned output image.

    Channel `c` of position `pos` lands at word
    `pos * groups * n_tiles + group * n_tiles + (c % co_per_group) // o`, lane
    `(c % co_per_group) % o` -- the same address `out_word()` computes in
    hw/convacc_proc.x. Lanes beyond the last real channel stay zero.
    """
    positions, groups, n_tiles = output_geometry(packed)
    o = 8 // packed.tasks[0].task.r
    spec = packed.spec
    if isinstance(spec, GemmSpec):
        channels, per_group = spec.n, spec.n
    elif spec.channelwise:
        channels, per_group = spec.co, spec.co
    else:
        channels, per_group = spec.co, spec.co_per_group
    flat = np.ascontiguousarray(result).reshape(positions, channels)
    image = np.zeros((positions, groups * n_tiles * 8), dtype=flat.dtype)
    for channel in range(channels):
        group, local = divmod(channel, per_group)
        tile, lane = divmod(local, o)
        image[:, (group * n_tiles + tile) * 8 + lane] = flat[:, channel]
    return image.reshape(-1)


def reference_for_packed(packed: PackedPlan, activations: np.ndarray, weights: np.ndarray, layout: str) -> np.ndarray:
    if isinstance(packed.spec, GemmSpec):
        return activations.astype(np.int32) @ weights.astype(np.int32)
    canonical = canonical_conv_weights(packed.spec, weights, layout)
    return conv_reference(packed.spec, activations, canonical)
