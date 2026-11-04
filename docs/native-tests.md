# Native (C++) test harness

`tests/cpp/` runs the same Verilated model as `make cocotb-test`, but drives it
from C++ instead of Python. On the same 4-core budget it turns the ~10k-case
golden regression from **304 s into 18 s**, and it adds two layers the cocotb
suite does not have: a C++ mirror of the reference model (diffed byte for byte
against Python) and property-based tests.

## Why it is faster

cocotb is not slow because it is Python -- it is slow because of two things the
harness can remove:

| Cost | cocotb | native |
|---|---|---|
| Loading an SRAM image | one `dut.mem[i].value = w` per word → Python call, VPI dispatch, signal lookup | one `memcpy` into `VlUnpacked<uint64_t,16384>` |
| Starting a batch | new process, new elaboration, new interpreter per chunk | one `Dut` per worker thread, alive for the whole run |
| Reading the result | `dut.out_rdata.value` per word | `memcpy` out of the output array |

The direct array access comes from `verilator --cc --public`, which makes the
internal signals addressable without flattening the hierarchy, so the three SRAM
macros stay plain arrays inside the model:

```
rootp->convacc_top__DOT__u_act_sram__DOT__mem   // VlUnpacked<QData, 16384>
```

`--public-flat-rw` also works, but flattening the design costs about 3x in
evaluation time (15.8 µs vs 4.7 µs per clock in a microbenchmark) for no
benefit here; `VL_PUBLIC_FLAG` keeps it available as an escape hatch.

That is the whole trick. What remains is the simulator's own cost -- about
3.5 µs per clock on the dense 3x3 datapath -- which neither driver can remove:
profiling puts 99.6 % of a case's time inside `eval()`, split across the four
generated `nba_comb` functions. Cutting compile time further is not where the
time goes; the remaining lever is fewer or cheaper clocks, not a faster host
loop.

## Threads

Parallelism is per case, and both drivers share one implementation
(`include/convacc/parallel.h`): `jobs` threads, each owning one `Dut` built once
at startup, pulling indices off an atomic cursor. The cursor rather than a
static split matters because case cost varies 2-3x across the grid -- a static
slice leaves one worker running long after the rest are idle. Per-thread state
is a parameter of the helper, so adding a second thing a worker needs (a model
instance, a scratch buffer) does not mean touching the threading code.

At ~6 ms per case, throughput is just how many cores the scheduler hands out.
On a 4-CPU slice of a 32-core host the 1x grid takes 58 / 29 / 15 / 15.5 s at
`--jobs` 1 / 2 / 4 / 8: **3.9x at 4 threads, then flat**. Going wider is
actively worse, not just useless -- at `--jobs 8` the summed simulation time
inside the workers doubles (58 s -> 122 s) because half the threads are waiting
for a core at any moment. `nproc` reports the host's cores, not the cgroup
quota, so do not size `--jobs` from it blindly; on this machine `cpu.max` says
4 CPUs while `nproc` says 32.

### Could this be gtest + a test runner instead?

`tests/cpp/dut/grid_main.cc` answers that with a measurement rather than an
argument. It registers one gtest test per **chunk** of cases (not per case:
gtest-parallel runs one process per test, and each process pays ~300 ms to
elaborate the model, which against a 6 ms case would be a 50x regression), reads
its knobs from the environment because `INSTANTIATE_TEST_SUITE_P` runs before any
flag is parsed, and shares one `Dut` per process.

Measured back to back on 9,872 cases, 4 cores (the grid has since grown to
9,972; the comparison is unchanged):

| driver | wall |
|---|---:|
| `parallel_for`, `--jobs 4` | **17.6 s** |
| gtest-parallel, 40 chunks (247 cases each) | 18.2 s |
| gtest-parallel, 16 chunks (617 cases) | 19.7 s |
| gtest-parallel, 8 chunks | 22.5 s |

So it lands within 3 % of the pool at a 250-case chunk, and loses more as chunks
get coarser -- coarse chunks cannot be rebalanced, and the dense cases are 2x the
cost of the GEMM ones. Fewer, larger chunks do not help either: the saved
elaboration (0.3 s per process) is smaller than the tail it creates.

The runner is not adopted as the default because it buys ~nothing in time and
costs the per-case reporting (`FAILED n case(s): name: reason`), the `sim ms` and
`cycles` aggregates, and the `--dump` plumbing, all of which would have to become
a gtest listener. It stays valuable if CI ever wants JUnit XML or
`--gtest_filter`-style single-case reruns; the source is kept for that reason.
It is off by default (`-DCONVACC_WITH_GRID_SHARDS=ON` to build it), so the normal
build does not link a third binary that nothing runs. Shards are selected the way
gtest does it -- `GTEST_TOTAL_SHARDS`/`GTEST_SHARD_INDEX` on the environment --
so no external runner is needed at all: gtest-parallel is not on PyPI (it is a
single script in google/gtest-parallel), and the numbers above were taken with a
checkout of it.

The gtest binaries under `tests/` are unit tests, not the regression grid; they
finish in 1.2 s on one thread.

Absolute numbers move with the machine: the same binary measured 15 s and 18 s
for the same corpus twenty minutes apart, since the container shares a host.
Compare configurations back to back, never against a number from yesterday.

## Compiling the model

`eval()` is generated C++, so the compiler matters far more than it does for the
harness. Measured on the 1x grid (9,972 cases, 4 threads, best of three runs,
simulation time only):

| model build | ms/case | wall |
|---|---:|---:|
| gcc 13 -O3 | 11.12 | 27.8 s |
| clang 18, Verilator's default `-Os` | 7.2-7.4 | 18.2-18.4 s |
| clang 18 -O3 | 7.02 | 17.6 s |
| clang 23 -O3 | 7.08 | 17.8 s |
| clang 23 -O2 | 6.95 | 17.4 s |
| clang 23 -O3 -march=native | **6.93** | **17.4 s** |
| clang 23 -O3 -fno-exceptions | 7.02 | 17.6 s |
| clang 23 -O3 + ThinLTO + `-fuse-ld=lld` | 7.03 | 17.5 s |

Two things this table says. **The compiler choice is worth a lot** -- gcc is
1.6x slower than clang on the same flags. **The flags are worth almost nothing**
-- once the model is built by clang, `-O2`/`-O3`/`-march=native`/LTO/no-exceptions
all land inside the run-to-run noise (about +-2 %). The generated `eval()` is a
handful of enormous functions with no cross-translation-unit structure to
exploit, so LTO has nothing to inline and `-march=native` buys one SIMD
instruction per thousand. `-fno-rtti` does not compile at all: Verilator's
runtime uses `dynamic_cast` (`include/verilated_types.h`), so the flag has to
stay off for the model and its runtime alike.

The default is therefore plain `clang++ -O3` (`NATIVE_VL_OPT`): `-march=native`
is left out so a build tree can be cached or moved between machines. PGO was
tried and is *slower* for clang (7.9 vs 7.0) -- the functions are already big
enough that profile-guided inlining only bloats them.

`make native-build` probes for a usable `clang++`
(`tools/find_clangxx.sh`, which adds `--gcc-install-dir` when clang cannot see
libstdc++, and prefers `clang++-23` over an older `clang++`) and passes it as
`VL_CXX`; without clang it silently stays on g++. Override with:

```sh
make native-build NATIVE_VL_CXX=                 # force Verilator's default g++
make native-build NATIVE_VL_CXX="clang++-23"     # pick a specific clang
make native-build NATIVE_VL_OPT="-O3 -march=native"   # local tuning
```

The flag only affects the generated model; the harness and the model are
separate translation units and are ABI-compatible either way. A stamp file
(`verilated/.vl_config`) records `VL_CXX`/`VL_OPT`/`VL_PUBLIC_FLAG`, so changing
any of them re-runs Verilator exactly once instead of silently keeping a stale
archive.

### A trap worth remembering

`VL_OPT` used to be passed to Verilator's generated makefile through
`cmake -E env`, which does nothing: `verilated.mk` assigns `OPT_FAST`/`OPT_SLOW`/
`OPT_GLOBAL` unconditionally (`OPT_FAST = -Os`), and in GNU make a makefile
assignment beats the environment. Every model therefore ran at `-Os` no matter
what `VL_OPT` said -- including the earlier version of this table. The flags are
now passed as make command-line variables, which do take precedence, and
`build.ninja` shows `OPT_FAST=-O3`. If a flag change ever appears to do nothing,
check that first.

### Linking

mold is the default linker (`CONVACC_LINKER`, `NATIVE_LD`) when it is installed.
The executables are ~1 MB, most of it generated model code: BFD links
`convacc_bench` in 0.07 s, mold in 0.02 s, and the two binaries run at the same
speed (7.05 vs 7.02 ms/case, i.e. inside the noise). The absolute numbers are
small because the binaries are small -- this buys a snappier edit/build loop,
not throughput. `make native-build NATIVE_LD=` goes back to the compiler
driver's default.

mold cannot read LLVM bitcode, so the ThinLTO row above was linked with lld;
reproducing it means `NATIVE_LD=lld NATIVE_VL_OPT="-O3 -flto=thin"` plus
`-flto=thin` in the harness and link flags.


## Layout

```
tests/cpp/
  include/convacc/     header-only mirrors of convacc/*.py:
                       spec, planner, packer, image (+descriptor), reference
                       (+build_case), dataset, rng, parallel, compare, cli, io
  src/                 spec, planner, packer, image, reference

  model/               everything that runs on the model alone
    model_test.cc      GoogleTest: model vs Python (layers 1-2)
    dataset_cases.h    the sampled dataset, loaded once; shared with dut/
    case_codec.{h,cc}  bytes -> case, the fuzz format
    oracle.{h,cc}      case -> expected output, through the model
    fuzz_main.cc       libFuzzer driver (model only)
    seeds/ findings/   seed corpus and the reproducers it grew

  dut/                 everything that needs the simulator
    dut.{h,cc}         Verilator harness (reset/load/run/read)
    run.h              one prepared case -> verdict (the only timing window)
    case.h             one shape -> verdict (materialise, pack, simulate)
    generator.{h,cc}   deterministic in-process corpus
    golden_main.cc     parallel golden regression (no gtest dependency)
    grid_main.cc       the sharded-gtest experiment (opt-in)
    dataset_test.cc    GoogleTest: DUT vs the expected image (layer 3)
    fuzz_test.cc       FuzzTest: random shapes, DUT vs model
    replay_main.cc     replay a libFuzzer corpus on the DUT
```

The split is a build boundary, not just a directory one: `convacc_model_tests`
links no Verilator, and `-DCONVACC_WITH_DUT=OFF` configures the model half on a
machine with no simulator installed. `convacc_covfuzz` (model only) is built
that way too.

### Three layers, checked independently

1. **Descriptor image** -- `build_image` must reproduce Python's 128 KB weight
   image byte for byte. A mismatch here is a packing or word-layout bug.
2. **Packed execution** -- `execute_packed` (which follows the packed weights
   the way the MAC array does) must agree with `reference_for_packed` (the plain
   convolution). A mismatch means the packer lost information.
3. **DUT** -- the Verilated model must produce the expected output image.

Keeping these separate matters: without layer 2, a packer bug shows up only as
a DUT mismatch and looks like an RTL fault.

Layers 1-2 live in `model/model_test.cc`, layer 3 in `dut/dataset_test.cc`; both
iterate the same `dataset_cases()` from `model/dataset_cases.h`.

## Usage

```sh
make rtl-gen            # DSLX -> hw/*.sv (needed once)
make native-build
make native-test        # golden regression, cases generated in-process
make native-test-full   # 10x grid (100k cases, ~8 min)
make native-test-wide   # 10x grid, capped to ~1x the time
make native-cross-check # re-derive a sample of generated cases in Python
make native-model-test  # GoogleTest: model first, then the DUT
make native-fuzz        # FuzzTest suite, serial smoke test (separate build tree)
make native-fuzz-parallel FUZZ_TIME=60s   # same suite, one process per core
make native-test-grid   # --grid full: every legal shape, ~1.68M cases
make native-cases       # write the on-disk corpus (cocotb + dataset)
make native-clean       # drop both build trees (nuclear option)

# One grid, split across runners. Shards are disjoint and exhaustive:
make native-test NATIVE_SHARD=3/20        # every 20th case, starting at 3
```

`native-cases` derives everything from the same `build_cases()` used by the
cocotb suite, so both drivers run exactly the same cases. `native-test` no
longer needs it -- see "Where the cases come from" below -- but the cocotb
suite and the model-fidelity tests still read what it writes:

* `build/cocotb_cases/<name>/{activation,weight,expected}.bin` -- what
  `convacc_bench` replays;
* `build/cpp_cases/dataset.bin` -- a sampled bundle (spec, *unpacked* weights,
  quantisation parameters, Python's outputs) for the model-fidelity tests.

## Where the cases come from

The bench has two sources:

* `--cases DIR` replays pre-rendered images (what `make native-cases` writes and
  what the cocotb suite runs);
* `--gen-scale N` builds them in-process from `dut/generator.cc`.

The generator is the default. At 10x the corpus is 100k directories and 25 GB of
images, which is why the first attempt at a bigger grid was unusable: the
sandbox has 8 GiB of page cache, so the corpus was re-read from disk on every
run. Generating a case costs 0.08 ms on average -- under 0.5 % of simulating it --
so the filesystem drops out of the loop entirely, and `build/cocotb_cases*` is no
longer part of the native path.

Both sources draw on the same shape axes and the same constraints, so they cover
the same ground; only the payload differs (the generator seeds its own PRNG
instead of consuming NumPy's). `scale=1` is still the corpus cocotb runs, and
`native-cases` still produces it for cross-checking.

Generating cases in C++ moves a trust boundary: the generator picks the spec and
the data, and the C++ model -- not Python -- computes the expected bytes. That is
why `make native-cross-check` exists. It dumps a random sample of generated cases
as a `CVCC` bundle and feeds it to `tools/verify_generated_cases.py`, which
rebuilds each one with `convacc.image.make_image` and requires the weight
SRAM image and the output image to match byte for byte:

```
400/400 generated cases reproduce exactly in Python     # ~1.8 s, no simulation
```

That is the same guarantee the 140-case dataset gives for the on-disk corpus, now
covering the shapes only the wider grid reaches (7x7 kernels, stride 2, 32x32
GEMMs). It runs in CI on every push.

### Case scale, and why 10x cases is 26x work

`GEN_SCALE` (default 1) multiplies the grid. Scale 10 also *widens* the axes --
stride 2 for dense convs, 7x7 kernels, up to 12 output channels, 9x9 deconv
inputs, 32x32 GEMMs -- so the extra cases cover shapes the 1x grid never reaches
instead of repeating it. That widening is what makes the run expensive:

| config | cases | mean cycles/case | simulated cycles | wall (4 cores) |
|---|---:|---:|---:|---:|
| `GEN_SCALE=1` | 9,972 | 1,993 | 2.0e7 | 18 s |
| `GEN_SCALE=10` | 100,000 | 5,203 | 5.2e8 | 7 min 49 s |
| `GEN_SCALE=10`, `--cycle-budget 2e8` | 42,500 | 4,366 | 1.9e8 | 2 min 48 s |

10x the cases, 2.6x the cycles each, 26x the work. The simulator is not the
variable: both runs land on **3.7 us/cycle** on four cores, and 1.99e7 vs
5.2e8 simulated cycles explains the runtime to within a percent. Nothing else
-- case construction, file I/O, image memcpy -- is visible at this scale.

The 7x7 kernels and 12-channel tiles are what cost the 2.6x, and they are the
point of the wider grid. If the runtime matters more, cap it:

```sh
make native-test GEN_SCALE=10 GEN_CYCLE_BUDGET=200000000   # ~3 min, 42k cases
make native-test-wide                                     # ~20 s, 5k cases
```

`--cycle-budget` prices each candidate with `make_plan().cycles()` (scaled by a
measured 36x, because that count covers the MAC loop only) and keeps a uniform
subset that fits. The subset is drawn from the already-shuffled pool, so it
still samples every shape family instead of degenerating into "only the small
ones".

CI runs `--gen-scale 1`: 18 s, no corpus to prepare.

Options: `-DCONVACC_WITH_GTEST=OFF` skips GoogleTest,
`-DCONVACC_WITH_DUT=OFF` drops the harness and everything in `dut/` (and with it
the Verilator requirement), `-DCONVACC_WITH_FUZZTEST=ON` adds the FuzzTest
suite, `-DCONVACC_FUZZTEST_SOURCE_DIR=<path>` points at an existing FuzzTest
checkout.

## FuzzTest

`dut/fuzz_test.cc` generates *shapes*, not just data: kernel size, stride,
padding, group count and seed are all fuzzed, and each case is built and run
end-to-end. Cases the planner cannot represent are filtered out before they cost
a simulation.

There is one `FUZZ_TEST` per planner path -- `ConvDense`, `ConvGrouped`,
`ConvDepthwise`, `Gemm` -- rather than one walk over everything. A narrow domain
spends its whole budget on schedulable shapes (drawing `ci`, `co` and `groups`
freely accepts about one pair in ten), and the split makes the suite shardable,
because a shard is a set of tests, never part of one.

```sh
make native-fuzz                        # serial smoke test: 1 s per test, 3 rounds
make native-fuzz-parallel FUZZ_TIME=60s # one process per core
```

`tools/run_fuzz_shards.sh` is the whole parallel implementation: it sets
`GTEST_TOTAL_SHARDS`/`GTEST_SHARD_INDEX` and starts that many processes.
FuzzTest and GoogleTest both honour those variables. No external runner is
needed, and gtest-parallel would not add anything here -- it works one test per
process, so four tests cap it at four workers however many cores there are.
`FUZZ_SHARDS=auto` (the default) picks min(usable cores, tests); it reads the
cgroup quota because `nproc` reports the host's cores, not the container's.
`FUZZ_TIME` is a round budget (in unit-test mode the mutation loop reads
`FUZZTEST_FUZZ_FOR` directly), so give each shard at most one test -- which
`auto` does by construction.

One round is capped at 10000 mutations by FuzzTest itself: on these shapes that
is 15-60 s, and a larger `FUZZTEST_FUZZ_FOR` cannot extend it, only cut it
short. `FUZZ_DURATION=<seconds>` therefore restarts each shard until the
wall-clock deadline, with a fresh PRNG seed per round (printed as
`FUZZTEST_PRNG_SEED` in the shard log, so any finding stays reproducible).
That is what the manual one-hour workflow uses:

```sh
make native-fuzz-parallel FUZZ_DURATION=3600
```

Measured with 8 s of fuzzing per test on the 4-CPU container:

| driver | wall |
|---|---:|
| serial, 4 tests | 34 s |
| 4 shards | 9 s (3.8x) |

A case costs ~2400 cycles, i.e. ~8.5 ms, so one process explores about 100
cases/s and four cores about 400/s. That is two orders of magnitude below the
model-only fuzzer (2.7k exec/s per core), which is exactly why the RTL stays out
of the libFuzzer loop and is checked by replaying the corpus instead.

The build is in unit-test mode. FuzzTest's coverage-guided mode needs a
libFuzzer runtime (`libclang_rt.fuzzer`) which is not part of a stock Ubuntu
clang install; without it, `FUZZ_TEST` runs its seed corpus under GoogleTest.
`FUZZ_ROUNDS` (default 3) repeats the suite with fresh PRNG seeds to widen
coverage.

A first `make native-fuzz` compiles abseil, re2, antlr and FuzzTest -- around
20 minutes on four cores. Point `NATIVE_CMAKE_FLAGS` at an existing checkout, or
cache `~/.ccache`, to avoid paying that repeatedly.

## Continuous integration

`ci.yml` runs on every push to `main` (and on pull requests). It has three
parts:

* `test` -- software and DSLX unit tests, DSLX fuzz/formal proofs, RTL
  generation and the cocotb regression, on one runner.
* `native-build` -- builds the Verilated harness once, then runs everything
  that fits on one machine: the two GoogleTest binaries, a 60 s FuzzTest smoke
  test, and the generator/Python cross-check. It publishes `convacc_bench` as
  an artifact.
* `native-shards` -- a 20-way matrix. Each runner downloads that one binary and
  runs `--shard I/20 --jobs 4` on the 1x grid and on `--grid full`. Nothing
  else needs to travel: the cases are generated in-process, so the grid lives
  in the executable. All twenty shards together cover 9,972 + 1,675,050 cases,
  each full shard takes about 13 min, and `summary` folds the twenty JSON
  reports into one table. The job ceiling is 35 min for slower runners.

`--shard I/N` keeps every Nth case starting at I, applied after `--filter`, so
the shards partition exactly the cases the unsplit run would have checked.
Because the corpus is deterministic, a shard needs no coordination with the
others. Locally the same knob is `make native-test NATIVE_SHARD=I/N`.

`dut-fuzz.yml` is the deliberate, manual search: `workflow_dispatch` only, the
same build, then twenty runners each starting four processes (one per core, one
`FUZZ_TEST` per process) and restarting them for 60 minutes. `FUZZ_DURATION`
makes the hour real: without it a one-hour `FUZZTEST_FUZZ_FOR` would still stop
after 10000 mutations (under a minute). Each round prints its PRNG seed, and
logs and reproducers are uploaded. That is roughly twenty machine-hours of DUT
simulation in one hour of wall time, versus the 60 s of mutation the per-push
job can afford. The duration is an input, so a shorter run needs no edit.

## The full grid

`--grid full` enumerates every legal combination of the widened axes instead of
sampling a multiple of the 10k base corpus: **1,675,050 cases**. `--gen-scale`
still selects the small and wide grids (1x = 9,972, 10x = 100,000) and is
ignored by `full`; the two old grids generate byte-for-byte the same case list
as before. Locally the entry point is `make native-test-grid`, or
`make native-test-grid NATIVE_SHARD=I/20` for one shard.

The axes are a time budget, not just a coverage wish. Twenty 4-core runners
deliver about 2.7e7 simulated cycles/s, so 20 minutes is ~3.2e10 cycles; at
1.68M cases that allows under 20k cycles/case on average. Three measurements
shaped the axes:

* **stride 3** adds shapes *and* lowers the mean -- a larger stride means a
  smaller output. Going from the 1.01M-case pool to stride 3 produced 1.49M
  cases while the mean dropped from 16.3k to 13.4k cycles.
* **pad 2** is the opposite: `ph = pad * (kh-1)/2` enlarges the output, taking
  the same pool to 45.8k cycles/case and the twenty-machine run to ~47 min. It
  is deliberately not in the grid.
* `n`, `hi/wi` and the GEMM limit stay capped for the same reason. Raising the
  GEMM limit from 32 to 48 alone tripled that family's mean cost.

Measured on one 4-core runner, a 1/200 shard of the full grid runs 8,376 cases
in 78.8 s (106 cases/s, 13.0k cycles/case, 37.5 ms/case of simulation).
Twenty runners in parallel therefore need ~13 min for all 1,675,050 cases, and
the CI ceiling of 35 min covers runners ~2.5x slower. The shape list itself
costs ~380 MB and ~1 s to build, so every runner regenerates it locally; no
corpus artifact has to travel. 20 shards are disjoint and exhaustive (83752 or
83753 cases each, zero duplicates).

`--cycle-budget N` still caps total simulated cycles when a run must be
shorter. For `full` the budget is split by measured family cost (dense is ~95%
of the cycles, not 53% of the case caps), so a capped run stays representative
instead of degenerating into "only the cheap families".

## What this does not replace

`make cocotb-test` stays. It is the independent implementation: if the C++
harness and the Python harness ever disagree, one of them has a bug, and having
both is the cheapest way to notice. The native suite is the fast path for
day-to-day work; cocotb remains the cross-check.
