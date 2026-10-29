from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Tuple

from .spec import ConvSpec, GemmSpec, OpSpec


@dataclass(frozen=True)
class Task:
    name: str
    mode: str
    m: int
    n: int
    k: int
    batches: int
    mh: int
    mw: int
    taps: Tuple[Tuple[int, int], ...]
    ci_base: int
    ci_count: int
    co_base: int
    group: int = 0
    phase_h: int = 0
    phase_w: int = 0
    out_step_h: int = 1
    out_step_w: int = 1
    r: int = 1

    @property
    def o(self) -> int:
        return 8 // self.r

    @property
    def k_groups(self) -> int:
        return ceil(self.k / self.r)

    @property
    def m_tiles(self) -> int:
        return ceil(self.m / 8)

    @property
    def n_tiles(self) -> int:
        return ceil(self.n / self.o)


@dataclass(frozen=True)
class Plan:
    tasks: Tuple[Task, ...]
    task_cycles: Tuple[int, ...]
    useful_macs: int

    @property
    def cycles(self) -> int:
        return sum(self.task_cycles)

    @property
    def utilization(self) -> float:
        return self.useful_macs / (64 * self.cycles) if self.cycles else 0.0


def _dense_task(spec: ConvSpec, group: int, name: str) -> Task:
    return Task(
        name=name,
        mode="dense",
        m=spec.n * spec.ho * spec.wo,
        n=spec.co_per_group,
        k=spec.kh * spec.kw * spec.ci_per_group,
        batches=spec.n,
        mh=spec.ho,
        mw=spec.wo,
        taps=tuple((kh, kw) for kh in range(spec.kh) for kw in range(spec.kw)),
        ci_base=group * spec.ci_per_group,
        ci_count=spec.ci_per_group,
        co_base=group * spec.co_per_group,
        group=group,
    )


def _channelwise_task(spec: ConvSpec) -> Task:
    return Task(
        name="channelwise_g0",
        mode="channelwise",
        m=spec.n * spec.ho * spec.wo,
        n=spec.co,
        k=spec.kh * spec.kw,
        batches=spec.n,
        mh=spec.ho,
        mw=spec.wo,
        taps=tuple((kh, kw) for kh in range(spec.kh) for kw in range(spec.kw)),
        ci_base=0,
        ci_count=1,
        co_base=0,
    )


def _phase_count(size: int, phase: int, stride: int) -> int:
    return 0 if phase >= size else (size - 1 - phase) // stride + 1


def _deconv_tasks(spec: ConvSpec) -> Tuple[Task, ...]:
    tasks = []
    for rh in range(spec.sh):
        for rw in range(spec.sw):
            taps = tuple(
                (kh, kw)
                for kh in range(spec.kh)
                for kw in range(spec.kw)
                if (rh + spec.ph - kh) % spec.sh == 0
                and (rw + spec.pw - kw) % spec.sw == 0
            )
            mh = _phase_count(spec.ho, rh, spec.sh)
            mw = _phase_count(spec.wo, rw, spec.sw)
            if not mh or not mw:
                continue
            # A phase with no surviving taps is still emitted: those output
            # positions receive no contribution at all, but they still have to
            # be written -- their value is `requant(0)`, which is only zero for
            # a trivial bias/zero-point. Dropping the phase left holes in the
            # output image.
            tasks.append(
                Task(
                    name=f"phase_{rh}_{rw}",
                    mode="dense",
                    m=spec.n * mh * mw,
                    n=spec.co,
                    k=len(taps) * spec.ci,
                    batches=spec.n,
                    mh=mh,
                    mw=mw,
                    taps=taps,
                    ci_base=0,
                    ci_count=spec.ci,
                    co_base=0,
                    phase_h=rh,
                    phase_w=rw,
                    out_step_h=spec.sh,
                    out_step_w=spec.sw,
                )
            )
    return tuple(tasks)


def _output_coords(spec: ConvSpec, task: Task, m_index: int) -> Tuple[int, int, int]:
    per_batch = task.mh * task.mw
    batch = m_index // per_batch
    rem = m_index % per_batch
    oh = task.phase_h + (rem // task.mw) * task.out_step_h
    ow = task.phase_w + (rem % task.mw) * task.out_step_w
    return batch, oh, ow


def _channelwise_unique_bytes(spec: ConvSpec, task: Task, mt: int, nt: int) -> int:
    addresses = set()
    co_per_group = spec.co_per_group
    for p in range(mt * 8, min((mt + 1) * 8, task.m)):
        batch, oh, ow = _output_coords(spec, task, p)
        for lane in range(nt * 8, min((nt + 1) * 8, task.n)):
            co = lane
            ci = co // co_per_group
            for kh, kw in task.taps:
                ih = oh * spec.sh + kh * spec.dh - spec.ph
                iw = ow * spec.sw + kw * spec.dw - spec.pw
                if 0 <= ih < spec.hi and 0 <= iw < spec.wi:
                    addresses.add(((batch * spec.hi + ih) * spec.wi + iw) * spec.ci + ci)
    return len(addresses)


def _estimate(spec: OpSpec, task: Task) -> int:
    if task.mode in ("dense", "gemm"):
        return task.m_tiles * task.n_tiles * max(task.k, task.k_groups, task.o)
    assert isinstance(spec, ConvSpec)
    cycles = 0
    for mt in range(task.m_tiles):
        for nt in range(task.n_tiles):
            act_cycles = ceil(_channelwise_unique_bytes(spec, task, mt, nt) / 8)
            cycles += max(task.k_groups, act_cycles, task.o)
    return cycles


def _valid_tap_positions(spec: ConvSpec, task: Task) -> int:
    kh_values = tuple(sorted({kh for kh, _ in task.taps}))
    kw_values = tuple(sorted({kw for _, kw in task.taps}))
    spatial_sum = 0
    for phase_y in range(task.mh):
        oh = task.phase_h + phase_y * task.out_step_h
        valid_kh = 0
        for kh in kh_values:
            if spec.kind == "deconv2d":
                numerator = oh + spec.ph - kh
                valid = numerator >= 0 and numerator % spec.sh == 0 and numerator // spec.sh < spec.hi
            else:
                ih = oh * spec.sh + kh * spec.dh - spec.ph
                valid = 0 <= ih < spec.hi
            valid_kh += int(valid)
        for phase_x in range(task.mw):
            ow = task.phase_w + phase_x * task.out_step_w
            valid_kw = 0
            for kw in kw_values:
                if spec.kind == "deconv2d":
                    numerator = ow + spec.pw - kw
                    valid = numerator >= 0 and numerator % spec.sw == 0 and numerator // spec.sw < spec.wi
                else:
                    iw = ow * spec.sw + kw * spec.dw - spec.pw
                    valid = 0 <= iw < spec.wi
                valid_kw += int(valid)
            spatial_sum += valid_kh * valid_kw
    return spec.n * spatial_sum


def _useful_macs(spec: OpSpec, task: Task) -> int:
    if isinstance(spec, GemmSpec):
        return task.m * task.n * task.k
    return _valid_tap_positions(spec, task) * task.n * task.ci_count


def make_plan(spec: OpSpec) -> Plan:
    if isinstance(spec, GemmSpec):
        tasks = (
            Task(
                name="gemm",
                mode="gemm",
                m=spec.m,
                n=spec.n,
                k=spec.k,
                batches=1,
                mh=spec.m,
                mw=1,
                taps=((0, 0),),
                ci_base=0,
                ci_count=spec.k,
                co_base=0,
            ),
        )
    elif spec.kind == "deconv2d":
        tasks = _deconv_tasks(spec)
    elif spec.channelwise:
        tasks = (_channelwise_task(spec),)
    else:
        tasks = tuple(_dense_task(spec, group, f"dense_g{group}") for group in range(spec.groups))
    if not tasks:
        raise ValueError("operator has no non-empty tasks")
    # One reduction width per operator. The output layout packs each n-tile into
    # a single 64-bit word, so `o = 8 // r` -- and with it the tile stride -- must
    # be the same for every task that writes into the same output image.
    #
    # Channelwise has a second constraint: `mac64_reduced` can only name `o`
    # within-group output channels (`oo = within / r`, `within < 8`), so every
    # group's `co_per_group` channels must fit inside one n-tile. When
    # `o < co_per_group` the extra channels are simply unreachable -- the engine
    # would recompute the first `o` of them instead. `ch_fill`'s `rel = i /
    # co_per_group` assumes the same thing: a tile may span several groups, but
    # never split one.
    allowed = (1, 2, 4, 8)
    if getattr(spec, "channelwise", False):
        need = spec.co_per_group
        allowed = tuple(r for r in allowed if 8 // r >= need)
        if not allowed:
            raise ValueError(
                f"channelwise needs o >= co_per_group={need}, but the widest "
                f"n-tile holds 8 channels"
            )
    candidates = [tuple(replace(task, r=r) for task in tasks) for r in allowed]
    tasks = min(
        candidates,
        key=lambda candidate: (sum(_estimate(spec, task) for task in candidate), candidate[0].r),
    )
    cycles = tuple(_estimate(spec, task) for task in tasks)
    useful = sum(_useful_macs(spec, task) for task in tasks)
    return Plan(tasks=tasks, task_cycles=cycles, useful_macs=useful)


def plan_to_dict(spec: OpSpec, plan: Plan) -> dict:
    from .spec import spec_to_dict

    return {
        "spec": spec_to_dict(spec),
        "hardware": {"macs": 64, "p": 8, "o": 8, "r": 1, "sram_bytes_per_tensor": 131072, "bytes_per_cycle": 8},
        "cost_model": "dense streaming; channelwise per-tile unique halo; no cross-tile reuse",
        "tasks": [dict(task.__dict__) for task in plan.tasks],
        "task_cycles": list(plan.task_cycles),
        "cycles": plan.cycles,
        "useful_macs": plan.useful_macs,
        "logical_macs": sum(task.m * task.n * task.k for task in plan.tasks),
        "utilization": plan.utilization,
    }
