#!/usr/bin/env python3
"""Materialise the regression cases for the native C++ test suite.

Two outputs, both derived from the single source of truth
(``tests/test_cocotb_convacc.py:build_cases``):

* ``--cases-dir`` -- one directory per case with the three SRAM images the
  Verilated DUT loads (``activation.bin``, ``weight.bin``, ``expected.bin``).
  ``convacc_bench`` walks this tree; it needs no metadata because it only
  replays the images.
* ``--dataset`` -- a sampled, self-describing bundle for the two gtest
  binaries (``convacc_model_tests`` and ``convacc_dut_tests``): the
  spec, the *unpacked* weights, the quantisation parameters and Python's
  outputs, so the C++ model can rebuild the image and be diffed against it.

Format of the bundle (little endian)::

    'CVCC' u32 version=1 u32 count
    per case:
        u32 name_len, name bytes
        u32 kind           0=conv2d 1=deconv2d 2=gemm
        i32 x19            n hi wi ci ho wo co kh kw sh sw dh dw ph pw groups gm gk gn
        u32 layout_len, layout bytes
        u32 width, i32[width] bias, i32[width] scale,
                       u8[width] shift, i8[width] zero_point
        u32 rank, u32[rank] weight shape, u32 count, i8[count] weight data
        u32 count, i8[count] activation data
        u32 count, u8[count] expected bytes (Python's layout_output)
        u32 count, u8[count] weight SRAM image

Usage::

    python tools/gen_cpp_cases.py                      # both outputs
    python tools/gen_cpp_cases.py --no-cases           # dataset only
    python tools/gen_cpp_cases.py --per-family 60
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from convacc.image import make_image  # noqa: E402
from test_cocotb_convacc import CASES, Case, build_cases  # noqa: E402


def _family(name: str) -> str:
    for prefix in ("grid_dense", "grid_depth", "grid_deconv", "grid_gemm", "random_seed"):
        if name.startswith(prefix):
            return prefix
    return "named"


def select(per_family: int) -> list[Case]:
    """All named cases plus an even slice of every grid family."""
    buckets: dict[str, list[Case]] = {}
    for case in CASES:
        buckets.setdefault(_family(case.name), []).append(case)

    chosen: list[Case] = []
    for family, items in buckets.items():
        if family in ("named", "random_seed") or len(items) <= per_family:
            chosen.extend(items)
            continue
        step = len(items) / per_family
        chosen.extend(items[int(i * step)] for i in range(per_family))
    return chosen


def _spec_fields(case: Case) -> list[int]:
    spec = case.spec
    if not hasattr(spec, "hi"):
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, spec.m, spec.k, spec.n]
    return [
        spec.n, spec.hi, spec.wi, spec.ci, spec.ho, spec.wo, spec.co, spec.kh, spec.kw,
        spec.sh, spec.sw, spec.dh, spec.dw, spec.ph, spec.pw, spec.groups,
        0, 0, 0,
    ]


def _kind(case: Case) -> int:
    spec = case.spec
    if not hasattr(spec, "hi"):
        return 2
    return 1 if getattr(spec, "kind", "conv2d") == "deconv2d" else 0


def write_sram_cases(cases: list[Case], out_dir: Path, skip_existing: bool) -> tuple[int, int]:
    """Write the three images per case. Returns (written, skipped)."""
    written = skipped = 0
    total = len(cases)
    for case in cases:
        case_dir = out_dir / case.name
        act_path = case_dir / "activation.bin"
        wgt_path = case_dir / "weight.bin"
        exp_path = case_dir / "expected.bin"
        if skip_existing and act_path.is_file() and wgt_path.is_file() and exp_path.is_file():
            skipped += 1
            continue
        _, image, expected = make_image(
            case.spec, case.weights, case.layout, case.activations,
            case.bias, case.scale, case.shift, case.zero_point,
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        act_path.write_bytes(image.activation)
        wgt_path.write_bytes(image.weight)
        exp_path.write_bytes(np.ascontiguousarray(expected).tobytes(order="C"))
        written += 1
        if written % 5000 == 0:
            print(f"  {written}/{total} cases written", flush=True)
    return written, skipped


def write_dataset(cases: list[Case], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "wb") as fh:
        fh.write(b"CVCC")
        fh.write(struct.pack("<II", 1, len(cases)))
        for case in cases:
            _, image, expected = make_image(
                case.spec, case.weights, case.layout, case.activations,
                case.bias, case.scale, case.shift, case.zero_point,
            )
            name = case.name.encode()
            fh.write(struct.pack("<I", len(name)))
            fh.write(name)
            fh.write(struct.pack("<I", _kind(case)))
            fh.write(struct.pack("<19i", *_spec_fields(case)))
            layout = case.layout.encode()
            fh.write(struct.pack("<I", len(layout)))
            fh.write(layout)

            width = int(case.bias.size)
            fh.write(struct.pack("<I", width))
            fh.write(struct.pack(f"<{width}i", *case.bias.tolist()))
            fh.write(struct.pack(f"<{width}i", *case.scale.tolist()))
            fh.write(struct.pack(f"<{width}B", *case.shift.tolist()))
            fh.write(struct.pack(f"<{width}b", *case.zero_point.tolist()))

            weights = np.ascontiguousarray(case.weights)
            fh.write(struct.pack("<I", weights.ndim))
            fh.write(struct.pack(f"<{weights.ndim}I", *weights.shape))
            fh.write(struct.pack("<I", weights.size))
            fh.write(weights.astype(np.int8).tobytes())

            acts = np.ascontiguousarray(case.activations).astype(np.int8)
            fh.write(struct.pack("<I", acts.size))
            fh.write(acts.tobytes())

            payload = np.ascontiguousarray(expected).astype(np.int8).tobytes()
            fh.write(struct.pack("<I", len(payload)))
            fh.write(payload)

            # The descriptor image too: the C++ builder must reproduce it byte
            # for byte, and comparing against the real thing is the only way to
            # catch an off-by-one in the word layout.
            fh.write(struct.pack("<I", len(image.weight)))
            fh.write(image.weight)
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", default="build/cocotb_cases")
    parser.add_argument("--dataset", default="build/cpp_cases/dataset.bin")
    parser.add_argument("--per-family", type=int, default=30)
    parser.add_argument("--no-cases", action="store_true")
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument("--force", action="store_true", help="rewrite existing case directories")
    parser.add_argument("--scale", type=int, default=1,
                        help="grid size multiplier (1 = the ~10k corpus)")
    args = parser.parse_args()

    if args.scale < 1:
        parser.error("--scale must be >= 1")

    if not args.no_cases:
        out_dir = ROOT / args.cases_dir
        cases = CASES if args.scale == 1 else build_cases(args.scale)
        print(f"cases: {len(cases)} (scale {args.scale})", flush=True)
        written, skipped = write_sram_cases(cases, out_dir, skip_existing=not args.force)
        print(f"SRAM images: {written} written, {skipped} already present in {out_dir}")

    if not args.no_dataset:
        sample = select(args.per_family)
        out_path = ROOT / args.dataset
        count = write_dataset(sample, out_path)
        size = out_path.stat().st_size
        print(f"dataset: {count} cases ({size / 1e6:.1f} MB) -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
