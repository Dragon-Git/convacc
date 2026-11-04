"""Tests for the SRAM image builder and the ``convacc image`` CLI.

``convacc.image`` turns a packed plan into the three byte images the accelerator
reads:

* activation RAM -- activations, zero-padded to the full 128 KB;
* weight RAM     -- a linked list of 16-word task descriptors, each followed by
                    its taps, packed weights and per-channel requant parameters;
* output RAM     -- pre-zeroed, written by the engine at run time.

The descriptor layout is a hardware contract, so these tests decode the
generated words and check them field by field rather than only asserting sizes.
"""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from convacc import ConvSpec, GemmSpec, pack_weights
from convacc.cli import main as cli_main
from convacc.image import (
    HEADER_WORDS,
    PARAM_WORDS_PER_TILE,
    SRAM_BYTES,
    WORDS,
    WordMemory,
    build_image,
)
from convacc.image import make_image, write_sram_hex
from convacc.spec import spec_to_dict


def _words(image) -> list:
    """Split the 128 KB weight image into its 64-bit little-endian words."""
    payload = image.weight
    return [int.from_bytes(payload[i * 8 : i * 8 + 8], "little") for i in range(WORDS)]


def _lane64(word: int, half: int) -> int:
    """One of the two 32-bit halves packed into a 64-bit parameter word."""
    return (word >> (32 * half)) & 0xFFFFFFFF


class WriteSramHexTest(unittest.TestCase):
    def test_rejects_wrong_size(self):
        with TemporaryDirectory() as tmp:
            for size in (0, 1, SRAM_BYTES - 8, SRAM_BYTES + 8):
                with self.assertRaises(ValueError):
                    write_sram_hex(Path(tmp) / "bad.hex", bytes(size))

    def test_round_trips_little_endian(self):
        payload = bytes((i * 7 + 3) & 0xFF for i in range(SRAM_BYTES))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.hex"
            write_sram_hex(path, payload)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), WORDS)
            self.assertTrue(all(len(line) == 16 for line in lines))
            self.assertEqual(lines[0], f"{int.from_bytes(payload[:8], 'little'):016x}")
            restored = b"".join(int(line, 16).to_bytes(8, "little") for line in lines)
            self.assertEqual(restored, payload)


class WordMemoryTest(unittest.TestCase):
    def test_alloc_is_a_bump_allocator(self):
        memory = WordMemory()
        self.assertEqual(memory.alloc(3), 0)
        self.assertEqual(memory.alloc(2), 3)
        self.assertEqual(memory.next_word, 5)

    def test_alloc_rejects_overflow(self):
        memory = WordMemory()
        with self.assertRaises(ValueError):
            memory.alloc(WORDS + 1)
        self.assertEqual(memory.next_word, 0)  # failed alloc must not advance

    def test_write_bytes_is_little_endian_and_zero_padded(self):
        memory = WordMemory()
        memory.write_bytes(0, b"\x01\x02\x03")
        self.assertEqual(memory.words[0], 0x0000000000030201)
        # A 3-byte payload rounds up to one 8-byte word; the next word is untouched.
        self.assertEqual(memory.words[1], 0)

    def test_write_bytes_rejects_overflow(self):
        memory = WordMemory()
        with self.assertRaises(ValueError):
            memory.write_bytes(WORDS, b"\x00")

    def test_to_bytes_round_trips_words(self):
        memory = WordMemory()
        memory.set(0, 0xDEADBEEFCAFEF00D)
        memory.set(WORDS - 1, 1)
        payload = memory.to_bytes()
        self.assertEqual(len(payload), SRAM_BYTES)
        self.assertEqual(int.from_bytes(payload[:8], "little"), 0xDEADBEEFCAFEF00D)
        self.assertEqual(int.from_bytes(payload[-8:], "little"), 1)


class BuildImageTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)

    def test_activation_is_zero_padded_to_full_sram(self):
        spec = GemmSpec(m=2, k=3, n=4)
        weights = self.rng.integers(-3, 4, size=(3, 4), dtype=np.int8)
        activations = self.rng.integers(-5, 6, size=(2, 3), dtype=np.int8)
        _, image, _ = make_image(spec, weights, "KN", activations)

        self.assertEqual(len(image.activation), SRAM_BYTES)
        raw = np.ascontiguousarray(activations).tobytes()
        self.assertEqual(image.activation[: len(raw)], raw)
        self.assertFalse(any(image.activation[len(raw) :]))

        self.assertEqual(len(image.weight), SRAM_BYTES)
        self.assertEqual(len(image.output), SRAM_BYTES)
        self.assertEqual(image.descriptor_start, 0)

    def test_expected_output_matches_geometry(self):
        spec = GemmSpec(m=8, k=8, n=8)
        weights = self.rng.integers(-3, 4, size=(8, 8), dtype=np.int8)
        packed, image, expected = make_image(spec, weights, "KN", np.zeros((8, 8), np.int8))

        out = image.manifest["output"]
        self.assertEqual(out["words"] * 8, expected.size)
        self.assertEqual(out["positions"], spec.m)
        self.assertEqual(out["groups"], 1)
        self.assertEqual(image.output_words, out["words"])
        self.assertEqual(image.manifest["tasks"][0]["name"], packed.tasks[0].task.name)

    def test_descriptor_fields_for_gemm(self):
        spec = GemmSpec(m=8, k=8, n=8)
        weights = self.rng.integers(-3, 4, size=(8, 8), dtype=np.int8)
        packed, image, _ = make_image(spec, weights, "KN", np.zeros((8, 8), np.int8))
        words = _words(image)
        task = image.manifest["tasks"][0]
        desc = words[task["descriptor_word"] : task["descriptor_word"] + HEADER_WORDS]

        flags = desc[0]
        self.assertEqual(flags & 0b1, 1, "bit0 marks a GEMM descriptor")
        self.assertEqual(flags & 0b110, 0, "GEMM is neither channelwise nor deconv")

        self.assertEqual(desc[1] & 0xFFFFFFFF, spec.m)
        self.assertEqual((desc[1] >> 32) & 0xFFFFFFFF, spec.n)
        self.assertEqual(desc[2] & 0xFFFFFFFF, spec.k)
        self.assertEqual((desc[2] >> 32) & 0xFFFFFFFF, 1, "batches")

        # activation_base / output_base are both 0 here.
        self.assertEqual(desc[12], 0)
        self.assertEqual(desc[13] & 0xFFFFFFFF, task["weight_word"])
        self.assertEqual((desc[13] >> 32) & 0xFFFFFFFF, task["param_word"])
        self.assertEqual(desc[14] & 0xFFFFFFFF, task["tap_word"])
        self.assertEqual(desc[15], 0, "single-task chain terminates immediately")

        # Packed weights sit verbatim at weight_word.
        start = task["weight_word"] * 8
        packed_bytes = packed.tasks[0].weights.astype(np.int8, copy=False).tobytes()
        self.assertEqual(image.weight[start : start + len(packed_bytes)], packed_bytes)

    def test_requant_parameters_are_packed_per_lane(self):
        spec = GemmSpec(m=8, k=8, n=8)
        weights = self.rng.integers(-3, 4, size=(8, 8), dtype=np.int8)
        bias = np.arange(1, 9, dtype=np.int32)
        scale = np.arange(10, 18, dtype=np.int32)
        shift = np.arange(8, dtype=np.uint8)
        zero_point = np.array([-1, -2, -3, -4, -5, -6, -7, -8], dtype=np.int8)

        _, image, _ = make_image(
            spec, weights, "KN", np.zeros((8, 8), np.int8),
            bias=bias, scale=scale, shift=shift, zero_point=zero_point,
        )
        words = _words(image)
        base = image.manifest["tasks"][0]["param_word"]

        for lane in range(8):
            word = base + lane // 2
            half = lane % 2
            self.assertEqual(_lane64(words[word], half), int(bias[lane]) & 0xFFFFFFFF)
            self.assertEqual(_lane64(words[base + 4 + lane // 2], half), int(scale[lane]))
            self.assertEqual((words[base + 8] >> (lane * 8)) & 0xFF, int(shift[lane]))
            self.assertEqual((words[base + 9] >> (lane * 8)) & 0xFF, int(zero_point[lane]) & 0xFF)

    def test_defaults_and_lane_padding_past_real_channels(self):
        # n=5 with r=1 leaves lanes 5..7 with no channel: they must take the
        # identity parameters (bias 0, scale 1, shift 0, zero_point 0).
        spec = GemmSpec(m=4, k=4, n=5)
        weights = self.rng.integers(-3, 4, size=(4, 5), dtype=np.int8)
        bias = np.array([11, 12, 13, 14, 15], dtype=np.int32)
        _, image, expected = make_image(spec, weights, "KN", np.zeros((4, 4), np.int8), bias=bias)

        words = _words(image)
        base = image.manifest["tasks"][0]["param_word"]
        real_lanes = 5
        for lane in range(8):
            word = base + lane // 2
            half = lane % 2
            if lane < real_lanes:
                self.assertEqual(_lane64(words[word], half), int(bias[lane]) & 0xFFFFFFFF)
                self.assertEqual(_lane64(words[base + 4 + lane // 2], half), 1)
            else:
                self.assertEqual(_lane64(words[word], half), 0)
                self.assertEqual(_lane64(words[base + 4 + lane // 2], half), 1)
                self.assertEqual((words[base + 8] >> (lane * 8)) & 0xFF, 0)
                self.assertEqual((words[base + 9] >> (lane * 8)) & 0xFF, 0)

        # Output image still covers every tile word, padding included.
        self.assertEqual(image.output_words * 8, expected.size)

    def test_channelwise_flag(self):
        spec = ConvSpec(n=1, hi=3, wi=3, ci=3, ho=3, wo=3, co=3, kh=3, kw=3, ph=1, pw=1, groups=3)
        weights = self.rng.integers(-3, 4, size=(3, 3, 3), dtype=np.int8)
        _, image, _ = make_image(spec, weights, "HWM", np.zeros((1, 3, 3, 3), np.int8))
        words = _words(image)
        flags = words[image.manifest["tasks"][0]["descriptor_word"]]
        self.assertEqual(flags & 0b1, 0, "conv descriptors do not set the GEMM bit")
        self.assertEqual(flags & 0b10, 0b10, "channelwise bit")

    def test_deconv_flag(self):
        spec = ConvSpec(
            n=1, hi=3, wi=3, ci=2, ho=5, wo=5, co=4, kh=3, kw=3,
            sh=2, sw=2, ph=1, pw=1, kind="deconv2d",
        )
        weights = self.rng.integers(-3, 4, size=(3, 3, 2, 4), dtype=np.int8)
        _, image, _ = make_image(spec, weights, "HWIO", np.zeros((1, 3, 3, 2), np.int8))
        words = _words(image)
        for task in image.manifest["tasks"]:
            flags = words[task["descriptor_word"]]
            self.assertEqual(flags & 0b100, 0b100, "deconv bit")

    def test_multi_task_descriptor_chain(self):
        spec = ConvSpec(n=1, hi=3, wi=3, ci=4, ho=3, wo=3, co=4, kh=3, kw=3,
                        ph=1, pw=1, groups=2)
        weights = self.rng.integers(-3, 4, size=(3, 3, 2, 2, 2), dtype=np.int8)
        _, image, _ = make_image(spec, weights, "HWGIO", np.zeros((1, 3, 3, 4), np.int8))
        words = _words(image)
        tasks = image.manifest["tasks"]
        self.assertEqual(len(tasks), 2)

        for current, following in zip(tasks, tasks[1:]):
            next_desc = words[current["descriptor_word"] + 15]
            self.assertEqual(next_desc, following["descriptor_word"])
        self.assertEqual(words[tasks[-1]["descriptor_word"] + 15], 0)

    def test_rejects_misaligned_output_base(self):
        spec = GemmSpec(m=2, k=2, n=2)
        weights = np.zeros((2, 2), dtype=np.int8)
        with self.assertRaises(ValueError):
            make_image(spec, weights, "KN", np.zeros((2, 2), np.int8), output_base=4)

    def test_rejects_output_overflow(self):
        spec = GemmSpec(m=2, k=2, n=2)
        weights = np.zeros((2, 2), dtype=np.int8)
        with self.assertRaises(ValueError):
            make_image(spec, weights, "KN", np.zeros((2, 2), np.int8), output_base=SRAM_BYTES)

    def test_rejects_wrong_parameter_shape(self):
        spec = GemmSpec(m=2, k=2, n=3)
        weights = np.zeros((2, 3), dtype=np.int8)
        with self.assertRaises(ValueError):
            make_image(spec, weights, "KN", np.zeros((2, 2), np.int8), bias=np.zeros(2, np.int32))

    def test_rejects_oversized_activation_image(self):
        spec = GemmSpec(m=1, k=1, n=1)
        packed = pack_weights(spec, np.zeros((1, 1), dtype=np.int8), "KN")
        image_zeros = np.zeros(1, dtype=np.int32)
        with self.assertRaises(ValueError):
            build_image(
                packed,
                np.zeros(1, dtype=np.int32),
                np.ones(1, dtype=np.int32),
                image_zeros.astype(np.uint8),
                np.zeros(1, dtype=np.int8),
                b"\x00" * (SRAM_BYTES + 1),
            )


class CliImageTest(unittest.TestCase):
    def _write_inputs(self, root: Path, spec, weights, activations,
                      bias=None, scale=None, shift=None, zero_point=None) -> dict:
        spec_path = root / "spec.json"
        spec_path.write_text(json.dumps(spec_to_dict(spec)), encoding="utf-8")
        weights_path = root / "weights.npy"
        np.save(weights_path, weights)
        activations_path = root / "activations.npy"
        np.save(activations_path, activations)

        argv = [
            "image",
            "--spec", str(spec_path),
            "--weights", str(weights_path),
            "--activations", str(activations_path),
            "--layout", "KN",
            "--out-dir", str(root / "out"),
        ]
        for name, value in (("bias", bias), ("scale", scale), ("shift", shift),
                            ("zero-point", zero_point)):
            if value is not None:
                path = root / f"{name}.npy"
                np.save(path, value)
                argv += [f"--{name}", str(path)]
        return argv

    def test_image_command_writes_all_artifacts(self):
        rng = np.random.default_rng(5)
        spec = GemmSpec(m=4, k=3, n=6)
        weights = rng.integers(-4, 5, size=(3, 6), dtype=np.int8)
        activations = rng.integers(-4, 5, size=(4, 3), dtype=np.int8)
        bias = rng.integers(0, 100, size=6, dtype=np.int32)
        scale = rng.integers(1, 40, size=6, dtype=np.int32)
        shift = rng.integers(0, 31, size=6, dtype=np.uint8)
        zero_point = rng.integers(-8, 9, size=6, dtype=np.int8)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self._write_inputs(root, spec, weights, activations,
                                      bias, scale, shift, zero_point)
            summary = io.StringIO()
            with redirect_stdout(summary):
                self.assertEqual(cli_main(argv), 0)
            self.assertIn("descriptor=0", summary.getvalue())

            out = root / "out"
            for name in ("activation.bin", "weight.bin", "expected.bin",
                         "activation.hex", "weight.hex", "manifest.json"):
                self.assertTrue((out / name).is_file(), f"missing {name}")

            self.assertEqual((out / "activation.bin").stat().st_size, SRAM_BYTES)
            self.assertEqual((out / "weight.bin").stat().st_size, SRAM_BYTES)

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["descriptor_start"], 0)
            self.assertEqual(len(manifest["tasks"]), 1)
            expected_bytes = manifest["output"]["words"] * 8
            self.assertEqual((out / "expected.bin").stat().st_size, expected_bytes)

            # hex file is a text view of the binary image.
            self.assertEqual((out / "activation.hex").read_text(encoding="utf-8").count("\n"), WORDS)
            self.assertEqual((out / "weight.hex").read_text(encoding="utf-8").count("\n"), WORDS)

            # expected.bin and the Python reference agree.
            _, _, expected = make_image(
                spec, weights, "KN", activations, bias, scale, shift, zero_point,
            )
            self.assertEqual((out / "expected.bin").read_bytes(), expected.tobytes(order="C"))


if __name__ == "__main__":
    unittest.main()
