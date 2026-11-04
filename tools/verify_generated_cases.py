#!/usr/bin/env python3
"""Re-derive cases produced by ``convacc_bench --gen-scale --dump`` in Python.

The native bench now generates its own corpus (``tests/cpp/dut/generator.cc``)
instead of replaying images written by Python. That is a real trust boundary:
the generator picks specs and data, and the C++ model -- not Python -- computes
the expected bytes. This script closes the loop. It reads a ``CVCC`` bundle of
generated cases and rebuilds each one with ``convacc.image.make_image``,
then checks that the weight SRAM image and the expected output match byte for
byte.

Usage::

    tests/cpp/build/convacc_bench --gen-scale 10 --dump build/gen_sample.bin \\
        --dump-count 400 --jobs 8 --quiet
    python tools/verify_generated_cases.py build/gen_sample.bin

Exit status is 1 if any case disagrees.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from convacc.image import make_image  # noqa: E402
from convacc.spec import ConvSpec, GemmSpec  # noqa: E402


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, count: int) -> bytes:
        blob = self.data[self.pos:self.pos + count]
        if len(blob) != count:
            raise ValueError("truncated bundle")
        self.pos += count
        return blob

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))

    def u32(self) -> int:
        return self.unpack("<I")[0]

    def string(self) -> str:
        return self.take(self.u32()).decode()

    def i32_array(self) -> np.ndarray:
        count = self.u32()
        return np.frombuffer(self.take(4 * count), dtype="<i4").copy()

    def u8_array(self) -> np.ndarray:
        return np.frombuffer(self.take(self.u32()), dtype=np.uint8).copy()

    def i8_array(self) -> np.ndarray:
        return np.frombuffer(self.take(self.u32()), dtype=np.int8).copy()


# Field order of the 19 ints in a case header, as written by the `CvccField`
# enum in tests/cpp/dut/golden_main.cc. Named, in both languages, because a
# bare `fields[16]` on each side of a language boundary is not a spec: shift one
# and the bundle silently decodes into the wrong operator.
FIELD_NAMES = (
    "n", "hi", "wi", "ci", "ho", "wo", "co", "kh", "kw",
    "sh", "sw", "dh", "dw", "ph", "pw", "groups",
    "gemm_m", "gemm_k", "gemm_n",
)
assert len(FIELD_NAMES) == 19, "the CVCC header is 19 ints wide"


def build_spec(kind: int, fields: dict[str, int]):
    if kind == 2:
        return GemmSpec(m=fields["gemm_m"], k=fields["gemm_k"], n=fields["gemm_n"])
    return ConvSpec(
        n=fields["n"], hi=fields["hi"], wi=fields["wi"], ci=fields["ci"],
        ho=fields["ho"], wo=fields["wo"], co=fields["co"], kh=fields["kh"],
        kw=fields["kw"], sh=fields["sh"], sw=fields["sw"], dh=fields["dh"],
        dw=fields["dw"], ph=fields["ph"], pw=fields["pw"], groups=fields["groups"],
        kind="deconv2d" if kind == 1 else "conv2d",
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="re-derive a CVCC bundle of generated cases in Python")
    parser.add_argument("bundle", help="file written by convacc_bench --dump")
    args = parser.parse_args(argv[1:])

    data = Path(args.bundle).read_bytes()
    r = Reader(data)
    if r.take(4) != b"CVCC":
        print("error: not a CVCC bundle")
        return 2
    version, count = r.unpack("<II")
    if version != 1:
        print(f"error: unsupported version {version}")
        return 2

    bad = 0
    for _ in range(count):
        name = r.string()
        kind = r.u32()
        fields = dict(zip(FIELD_NAMES, r.unpack("<19i")))
        layout = r.string()

        # `width` is the quantisation array length, so the four arrays follow
        # back to back without their own counts (matching the C++ writer).
        width = r.u32()
        bias = np.frombuffer(r.take(4 * width), dtype="<i4").copy()
        scale = np.frombuffer(r.take(4 * width), dtype="<i4").copy()
        shift = np.frombuffer(r.take(width), dtype=np.uint8).copy()
        zero_point = np.frombuffer(r.take(width), dtype=np.int8).copy()

        rank = r.u32()
        shape = [r.u32() for _ in range(rank)]
        weights = r.i8_array().reshape(shape)
        activations = r.i8_array()
        expected = r.i8_array()
        weight_image = r.u8_array()

        spec = build_spec(kind, fields)
        # The bundle stores activations flat, the Python model wants the
        # semantic tensor.
        if kind == 2:
            activations = activations.reshape((spec.m, spec.k))
        else:
            activations = activations.reshape((spec.n, spec.hi, spec.wi, spec.ci))
        _, image, py_expected = make_image(
            spec, weights, layout, activations, bias, scale, shift, zero_point
        )
        py_expected = np.ascontiguousarray(py_expected, dtype=np.int8)
        py_image = np.frombuffer(image.weight, dtype=np.uint8)

        problems = []
        if py_image.tobytes() != weight_image.tobytes():
            diff = int(np.count_nonzero(py_image != weight_image))
            problems.append(f"weight image differs in {diff} bytes")
        if py_expected.tobytes() != expected.tobytes():
            if py_expected.size != expected.size:
                problems.append(
                    f"output length {py_expected.size} vs {expected.size}"
                )
            else:
                diff = int(np.count_nonzero(py_expected != expected))
                problems.append(f"expected output differs in {diff} bytes")
        if problems:
            bad += 1
            print(f"FAIL {name}: {'; '.join(problems)}")
            print(f"     spec={spec} layout={layout}")

    if bad:
        print(f"\n{bad}/{count} cases disagreed with the Python model")
        return 1
    print(f"{count}/{count} generated cases reproduce exactly in Python")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
