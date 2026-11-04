#!/usr/bin/env python3
"""Write the seed corpus for the coverage-guided fuzzer.

The fuzzer discovers shapes on its own, but a cold start wastes the first
thousands of executions rediscovering the obvious families. These seeds pin one
representative per code path (dense / grouped / depthwise / deconv / GEMM, each
layout, extreme quantisation), so a short CI run starts with full branch
coverage instead of growing into it.

Each file is the 32-byte header described in tests/cpp/model/case_codec.cc:

    [0]  kind    0,1,2 conv   3 depthwise   4,5 deconv   6,7 gemm
    [1]  batch
    [2]  hi        [3] wi       [4] ci      [5] co
    [6]  kh index (kh = 2*(x%4)+1)          [7] kw index
    [8]  sh        [9] sw       [10] dh     [11] dw
    [12] ph        [13] pw      [14] groups index
    [15] flags: bit0..2 layout, bit2 extremes, bit3 huge scale, bit4 tiny shift
    [16..31] data seed

Usage: python3 tools/gen_fuzz_seeds.py [--out tests/cpp/model/seeds]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def seed(name: str, kind: int, batch: int, hi: int, wi: int, ci: int, co: int,
         kh: int, kw: int, sh: int, sw: int, dh: int, dw: int, ph: int, pw: int,
         groups: int, flags: int, data: int) -> tuple[str, bytes]:
    header = bytes([kind, batch - 1, hi - 1, wi - 1, ci - 1, co - 1,
                    (kh - 1) // 2, (kw - 1) // 2, sh - 1, sw - 1, dh - 1, dw - 1,
                    ph, pw, groups, flags])
    payload = header + data.to_bytes(16, "little")
    return f"{name}", payload


def build() -> list[tuple[str, bytes]]:
    out = [
        seed("conv_3x3_s1p1", 0, 1, 5, 5, 3, 8, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0x1234),
        seed("conv_1x1_s2", 1, 2, 7, 7, 8, 16, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 0x2345),
        seed("conv_dilated", 2, 1, 9, 9, 4, 4, 3, 3, 1, 1, 2, 1, 2, 2, 0, 0, 0x3456),
        seed("conv_batched", 1, 3, 6, 6, 4, 8, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0x4567),
        seed("conv_wide", 0, 1, 12, 12, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0x5678),
        seed("depthwise_3x3", 3, 1, 6, 6, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 0, 1, 0x6789),
        seed("depthwise_s2", 3, 1, 8, 8, 8, 8, 5, 5, 2, 2, 1, 1, 1, 1, 0, 1, 0x789a),
        seed("deconv_s2", 4, 1, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0x89ab),
        seed("deconv_asym", 5, 2, 3, 5, 4, 4, 3, 1, 2, 3, 1, 1, 1, 0, 0, 0, 0x9abc),
        seed("grouped_hwg", 0, 1, 6, 6, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0, 0xabcd),
        seed("grouped_goihw", 0, 1, 7, 7, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 6, 0xbcde),
        seed("gemm_small", 6, 1, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0xcdef),
        seed("gemm_tail", 7, 1, 11, 11, 11, 11, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0xdef0),
        seed("conv_extremes", 0, 1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0x1C, 0xef01),
        seed("conv_layouts", 0, 1, 6, 9, 6, 6, 5, 3, 1, 1, 1, 1, 2, 1, 0, 0x0C, 0xf012),
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "tests/cpp/model/seeds"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in build():
        (out_dir / name).write_bytes(payload)
    print(f"wrote {len(build())} seeds to {out_dir}")


if __name__ == "__main__":
    main()
