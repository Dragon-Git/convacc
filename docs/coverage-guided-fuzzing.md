# Coverage-guided fuzzing

The golden corpus tests the shapes we thought of. This fuzzer searches the
operator space for shapes we did not, and checks the packed image against
independent code paths as it goes.

## Two loops, on purpose

```
libFuzzer  ->  decode_case  ->  build_case  ->  verify_case      (model only)
                                    |
                                    +--> corpus on disk
                                              |
                              convacc_fuzz_replay  ->  Verilated DUT
```

The model runs in-process at roughly 0.2 ms per case, so libFuzzer's coverage
feedback can actually drive the search through the planner's branches. The DUT
costs milliseconds per case and would drown that signal, so it sits in a second,
embarrassingly parallel pass: everything the fuzzer keeps is replayed on the
Verilated model and must match byte for byte.

Instrumentation is limited to the model and the codec (`CONVACC_WITH_COVFUZZ`),
which is exactly the code that decides what a case looks like.

## Input format

A case is a byte string, not a serialised struct, so mutations stay meaningful:

| Bytes | Meaning |
|---|---|
| `b[0] & 7` | operator kind: conv (0-2 channelwise), deconv (4-5), GEMM (6-7) |
| `b[1..7]` | batch, input H/W, channels, kernel H/W, stride |
| `b[8..15]` | padding, dilation, groups, layout selector |
| `b[16..27]` | quantisation parameters |
| `b[28..31]` | data seed |
| rest | additional entropy for the tensor PRNG |

Every dimension is clamped and every `groups` value is picked from the divisors
of `ci`/`co`, so a random byte string has a good chance of decoding to a
*schedulable* operator instead of being rejected. Mutating one byte changes one
dimension. GEMMs above 120k MACs are rejected: the budget keeps one exec near
0.2 ms, which matters more than reaching very large shapes (the golden corpus
covers those).

`tools/gen_fuzz_seeds.py` writes the 15 seed cases in `tests/cpp/model/seeds/`
(1x1, 3x3, dilated, batched, grouped, depthwise, GEMM tail, ...) that give the
search a starting point instead of beginning from noise.

## What the oracle checks

`verify_case` is structural, not a second reference model. It walks the packed
artifact the way the hardware does and compares against code paths that already
exist:

* **Descriptor chain** -- addresses in range, flags match the spec, chain length
  equals the task count, last `next` is 0, no self-reference.
* **Weight packing** -- every packed byte equals the plan's weights; tap and
  parameter blocks in range; padded lanes carry the identity quantisation
  (bias 0, scale 1, shift 0, zero point 0).
* **Image geometry** -- both images are 128 KB, the activation image starts with
  the input tensor verbatim, `output_words` matches the geometry and fits in
  SRAM.
* **Planner** -- at least one task, one cycle estimate per task, `plan.cycles()`
  is the sum, utilisation in [0, 1].
* **Packed vs semantic** -- `execute_packed` must agree element-wise with
  `reference_for_packed`; this is the layer that catches packer bugs before they
  look like RTL faults.
* **Requantisation** -- monotonic in the accumulator; the shift=0 path matches
  the reference exactly.

Any `SpecError` is treated as "not a case" (the Python model rejects it too).
Any other exception aborts with the decoded case and the raw bytes on stderr,
which libFuzzer turns into a `crash-*` file.

## Running

```sh
make covfuzz                 # configure + build, then fuzz for COVFUZZ_SECONDS
make covfuzz COVFUZZ_SECONDS=300 COVFUZZ_JOBS=4
make covfuzz-replay          # replay build/covfuzz-corpus on the DUT
```

The corpus lives in `build/covfuzz-corpus/` and is kept between runs, so a
repeated `make covfuzz` continues where the last one stopped. `covfuzz-build`
only needs `hw/convacc_top.sv` (run `make rtl-gen` first); it does not need the
golden cases.

A 90-second run from the seeds reaches ~560 corpus entries and ~2.8k
instrumented edges. Replaying that corpus takes ~6 s on 8 threads (545 built and
passing; the rest are shapes the planner legitimately rejects).

## Findings

The fuzzer found a real RTL/model disagreement in depthwise convolution with
`co > 8`. Reproducers are kept in `tests/cpp/model/findings/`:

```
depthwise_cpg3_1x1  depthwise_cpg3_7x3  depthwise_cpg3_dilated
depthwise_cpg9_5x5  depthwise_cpg3_hwm
```

Condition: `channelwise && co > 8 && 8 % co_per_group != 0`. The lane fetch in
`hw/convacc_proc.x` (`ch_fill`) maps lane *i* to activation byte
`i / co_per_group`, counted from the tile's first channel. That is only correct
when a tile starts on a group boundary. With `co <= 8` there is a single tile
and the extra lanes are masked off, which is why the golden corpus (depthwise
with `co == ci <= 8`) never hit it.

The default search skips this configuration (`fuzz_main.cc`) so a known issue
does not mask new ones, but `convacc_fuzz_replay` still runs the saved inputs --
they fail today, and will start passing when the RTL is fixed.

## What this does not cover

The DUT is only exercised by replay, so a bug that needs a specific *sequence*
of cases cannot be found here. The fuzzer also knows nothing about DSLX
semantics: it searches the C++ model's contract, and the RTL is assumed to
implement that contract until replay says otherwise.
