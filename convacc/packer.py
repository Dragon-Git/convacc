from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .planner import Plan, Task, make_plan
from .spec import ConvSpec, GemmSpec, OpSpec, SRAM_BYTES


@dataclass(frozen=True)
class PackedTask:
    task: Task
    weights: np.ndarray
    offset: int


@dataclass(frozen=True)
class PackedPlan:
    spec: OpSpec
    plan: Plan
    tasks: Tuple[PackedTask, ...]

    def to_bytes(self) -> bytes:
        return b"".join(p.weights.astype(np.int8, copy=False).tobytes(order="C") for p in self.tasks)

    def manifest(self) -> dict:
        from .planner import plan_to_dict

        result = plan_to_dict(self.spec, self.plan)
        result["packed_weight_bytes"] = len(self.to_bytes())
        result["packed_tasks"] = [
            {"name": p.task.name, "offset": p.offset, "bytes": p.weights.size, "shape": list(p.weights.shape)}
            for p in self.tasks
        ]
        return result


def canonical_conv_weights(spec: ConvSpec, weights: np.ndarray, layout: str) -> np.ndarray:
    expected = (spec.kh, spec.kw, spec.groups, spec.ci_per_group, spec.co_per_group)
    if layout == "HWGIO":
        result = weights
    elif layout == "HWIO" and spec.groups == 1:
        result = weights[:, :, None, :, :]
    elif layout == "OIHW" and spec.groups == 1:
        result = weights.transpose(1, 2, 3, 0)[:, :, None, :, :]
    elif layout == "GOIHW":
        result = weights.transpose(2, 3, 0, 4, 1)
    elif layout == "HWM" and spec.channelwise:
        if weights.shape != (spec.kh, spec.kw, spec.co):
            raise ValueError(f"HWM weights must have shape {(spec.kh, spec.kw, spec.co)}")
        result = weights.reshape(spec.kh, spec.kw, spec.ci, spec.co_per_group)[:, :, :, None, :]
    else:
        raise ValueError(f"layout {layout} is not valid for this operator")
    if result.shape != expected:
        raise ValueError(f"canonical weights have shape {result.shape}; expected {expected}")
    if result.dtype != np.int8:
        raise ValueError(f"weights must be int8, got {result.dtype}")
    return np.ascontiguousarray(result)


def _pack_conv_task(spec: ConvSpec, task: Task, weights: np.ndarray) -> np.ndarray:
    packed = np.zeros((task.n_tiles, task.k_groups, 8), dtype=np.int8)
    for nt in range(task.n_tiles):
        valid_outputs = min(task.o, task.n - nt * task.o)
        for kg in range(task.k_groups):
            for within in range(8):
                o = within // task.r
                rr = within % task.r
                reduction = kg * task.r + rr
                if o >= valid_outputs or reduction >= task.k:
                    continue
                tap_index = reduction // task.ci_count if task.mode == "dense" else reduction
                kh, kw = task.taps[tap_index]
                if task.mode == "channelwise":
                    co = nt * task.o + o
                    group = co // spec.co_per_group
                    co_in_group = co % spec.co_per_group
                    ci = 0
                else:
                    group = task.group
                    co_in_group = nt * task.o + o
                    ci = reduction % task.ci_count
                packed[nt, kg, within] = weights[kh, kw, group, ci, co_in_group]
    return packed


def pack_weights(spec: OpSpec, weights: np.ndarray, layout: str, plan: Plan | None = None) -> PackedPlan:
    plan = plan or make_plan(spec)
    packed_tasks = []
    offset = 0
    if isinstance(spec, GemmSpec):
        if layout != "KN" or weights.shape != (spec.k, spec.n) or weights.dtype != np.int8:
            raise ValueError("GEMM weights must be an int8 KxN array with layout KN")
        task = plan.tasks[0]
        packed = np.zeros((task.n_tiles, task.k_groups, 8), dtype=np.int8)
        for nt in range(task.n_tiles):
            for kg in range(task.k_groups):
                for within in range(8):
                    o = within // task.r
                    rr = within % task.r
                    co = nt * task.o + o
                    reduction = kg * task.r + rr
                    if co < task.n and reduction < task.k:
                        packed[nt, kg, within] = weights[reduction, co]
        packed_tasks.append(PackedTask(task, packed, 0))
        offset = packed.size
    else:
        canonical = canonical_conv_weights(spec, weights, layout)
        for task in plan.tasks:
            packed = _pack_conv_task(spec, task, canonical)
            packed_tasks.append(PackedTask(task, packed, offset))
            offset += packed.size
    if offset > SRAM_BYTES:
        raise ValueError(f"packed weights need {offset} bytes; limit is {SRAM_BYTES}")
    return PackedPlan(spec=spec, plan=plan, tasks=tuple(packed_tasks))
