from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .packer import pack_weights
from .planner import make_plan, plan_to_dict
from .reference import execute_packed, reference_for_packed
from .spec import spec_from_dict


def _load_spec(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return spec_from_dict(json.load(handle))


def _write_json(path: str, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="convacc")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="build an execution plan")
    plan_parser.add_argument("--spec", required=True)
    plan_parser.add_argument("--out", required=True)

    pack_parser = sub.add_parser("pack", help="plan and pack NumPy weights")
    pack_parser.add_argument("--spec", required=True)
    pack_parser.add_argument("--weights", required=True)
    pack_parser.add_argument("--layout", required=True)
    pack_parser.add_argument("--weights-out", required=True)
    pack_parser.add_argument("--plan-out", required=True)

    check_parser = sub.add_parser("check", help="compare packed execution with NumPy reference")
    check_parser.add_argument("--spec", required=True)
    check_parser.add_argument("--weights", required=True)
    check_parser.add_argument("--activations", required=True)
    check_parser.add_argument("--layout", required=True)

    image_parser = sub.add_parser("image", help="build full SRAM images and expected output")
    image_parser.add_argument("--spec", required=True)
    image_parser.add_argument("--weights", required=True)
    image_parser.add_argument("--activations", required=True)
    image_parser.add_argument("--layout", required=True)
    image_parser.add_argument("--bias")
    image_parser.add_argument("--scale")
    image_parser.add_argument("--shift")
    image_parser.add_argument("--zero-point")
    image_parser.add_argument("--out-dir", required=True)

    args = parser.parse_args(argv)
    spec = _load_spec(args.spec)
    if args.command == "plan":
        plan = make_plan(spec)
        _write_json(args.out, plan_to_dict(spec, plan))
        print(f"tasks={len(plan.tasks)} cycles={plan.cycles} utilization={plan.utilization:.4f}")
        return 0

    weights = np.load(args.weights, allow_pickle=False)
    if args.command == "image":
        from .image_builder import make_image, write_sram_hex

        activations = np.load(args.activations, allow_pickle=False)
        bias = np.load(args.bias) if args.bias else None
        scale = np.load(args.scale) if args.scale else None
        shift = np.load(args.shift) if args.shift else None
        zero_point = np.load(args.zero_point) if args.zero_point else None
        packed, image, expected = make_image(
            spec, weights, args.layout, activations, bias, scale, shift, zero_point
        )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "activation.bin").write_bytes(image.activation)
        (out_dir / "weight.bin").write_bytes(image.weight)
        (out_dir / "expected.bin").write_bytes(expected.tobytes(order="C"))
        write_sram_hex(out_dir / "activation.hex", image.activation)
        write_sram_hex(out_dir / "weight.hex", image.weight)
        (out_dir / "manifest.json").write_text(json.dumps(image.manifest, indent=2), encoding="utf-8")
        print(
            f"tasks={len(packed.tasks)} weight_bytes={image.manifest['weight_bytes_used']} "
            f"outputs={expected.size} descriptor={image.descriptor_start}"
        )
        return 0

    packed = pack_weights(spec, weights, args.layout)
    if args.command == "pack":
        Path(args.weights_out).write_bytes(packed.to_bytes())
        _write_json(args.plan_out, packed.manifest())
        print(
            f"tasks={len(packed.tasks)} weight_bytes={len(packed.to_bytes())} "
            f"cycles={packed.plan.cycles} utilization={packed.plan.utilization:.4f}"
        )
        return 0

    activations = np.load(args.activations, allow_pickle=False)
    actual = execute_packed(packed, activations)
    expected = reference_for_packed(packed, activations, weights, args.layout)
    if not np.array_equal(actual, expected):
        mismatch = np.argwhere(actual != expected)[0]
        raise SystemExit(f"verification failed at {mismatch.tolist()}")
    print(f"verified outputs={actual.size} max_abs={int(np.max(np.abs(actual)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
