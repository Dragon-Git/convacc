from __future__ import annotations

from dataclasses import dataclass
from typing import Union

SRAM_BYTES = 128 * 1024
MAX_DIM = 256


def _positive(name: str, value: int) -> None:
    if value <= 0 or value > MAX_DIM:
        raise ValueError(f"{name} must be in 1..{MAX_DIM}, got {value}")


def _check_tensor(name: str, elements: int) -> None:
    if elements <= 0 or elements > SRAM_BYTES:
        raise ValueError(f"{name} needs {elements} bytes; limit is {SRAM_BYTES}")


@dataclass(frozen=True)
class ConvSpec:
    n: int
    hi: int
    wi: int
    ci: int
    ho: int
    wo: int
    co: int
    kh: int
    kw: int
    sh: int = 1
    sw: int = 1
    dh: int = 1
    dw: int = 1
    ph: int = 0
    pw: int = 0
    groups: int = 1
    kind: str = "conv2d"

    def __post_init__(self) -> None:
        for name in ("n", "hi", "wi", "ci", "ho", "wo", "co", "kh", "kw"):
            _positive(name, getattr(self, name))
        if self.kind not in ("conv2d", "deconv2d"):
            raise ValueError(f"unsupported convolution kind: {self.kind}")
        if self.kh % 2 != 1 or self.kw % 2 != 1 or self.kh > 15 or self.kw > 15:
            raise ValueError("kernel dimensions must be odd and in 1..15")
        if min(self.sh, self.sw, self.dh, self.dw) <= 0:
            raise ValueError("stride and dilation must be positive")
        if min(self.ph, self.pw) < 0:
            raise ValueError("padding must be non-negative")
        if self.ci % self.groups or self.co % self.groups:
            raise ValueError("groups must divide both ci and co")
        if self.kind == "deconv2d" and (self.groups != 1 or self.dh != 1 or self.dw != 1):
            raise ValueError("deconv2d supports groups=1 and dilation=1 only")
        _check_tensor("activation", self.n * self.hi * self.wi * self.ci)
        _check_tensor("output", self.n * self.ho * self.wo * self.co)

    @property
    def ci_per_group(self) -> int:
        return self.ci // self.groups

    @property
    def co_per_group(self) -> int:
        return self.co // self.groups

    @property
    def channelwise(self) -> bool:
        return self.kind == "conv2d" and self.groups == self.ci


@dataclass(frozen=True)
class GemmSpec:
    m: int
    k: int
    n: int
    kind: str = "gemm"

    def __post_init__(self) -> None:
        _positive("m", self.m)
        _positive("k", self.k)
        _positive("n", self.n)
        _check_tensor("A", self.m * self.k)
        _check_tensor("B", self.k * self.n)
        _check_tensor("C", self.m * self.n)


OpSpec = Union[ConvSpec, GemmSpec]


def spec_from_dict(data: dict) -> OpSpec:
    values = dict(data)
    kind = values.pop("kind", "conv2d")
    if kind == "gemm":
        return GemmSpec(kind=kind, **values)
    return ConvSpec(kind=kind, **values)


def spec_to_dict(spec: OpSpec) -> dict:
    return dict(spec.__dict__)
