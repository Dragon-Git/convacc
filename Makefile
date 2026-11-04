PYTHON ?= .venv/bin/python
BUILD := build
CORE_SV := $(BUILD)/mac64_reduced_core.sv $(BUILD)/requant8_core.sv
# XLS / xlsynth toolchain. Defined early: DSLX_STDLIB (below) probes paths
# under $(XLS_BIN) at parse time, and recipes invoke several tools by bare
# name, so its bin/ is prepended to PATH for the whole Makefile.
XLS_BIN ?= $(HOME)/opt/xlsynth-bin/bin
export PATH := $(XLS_BIN):$(PATH)
XEZIM ?= $(HOME)/opt/xezim-bin/bin/xezim
XDIST ?= 8
# Parallel DSLX test files: the interpreter is single-threaded, but every .x
# file is independent (tools/run_dslx_parallel.sh).
JOBS ?= 8
# cocotb backend: verilator (default) or xezim -- see tests/conftest.py
SIM ?= verilator
# `check_ir_equivalence_main` proves opt_main preserved semantics. Under
# xlsynth-bin 0.61.0 the 64-bit MAC proof does not converge (42+ min on a
# 4-core runner, no output), so CI sets IR_EQUIV=0. The SV is still generated;
# DSLX-level proofs are covered by `make dslx-prove`. The default keeps the
# check for local and manual runs.
IR_EQUIV ?= 1

.PHONY: test full-test software-test dslx-test dslx-fuzz dslx-prove dslx-prove-deep dslx-test-convacc dslx-prove-convacc dslx-test-engine dslx-test-fullsys rtl cocotb-test clean native-cases native-build native-test native-model-test native-fuzz native-fuzz-build native-fuzz-parallel native-clean native-test-full native-test-wide native-cross-check covfuzz-build covfuzz covfuzz-replay

test: software-test dslx-test

fuzz: dslx-fuzz

full-test: test cocotb-test dslx-test-fullsys dslx-prove

software-test:
	$(PYTHON) -m unittest discover -s tests -v

# Hand-written unit tests over the DSLX sources. These are pure functions with
# small inputs, so the tree-walking interpreter is already fast (58 tests in
# ~1.2 s). The engine-level harnesses below are the slow ones and are opt-in.
# Hand-written unit tests over the DSLX sources. Pure functions with small
# inputs, so the tree-walking interpreter is already fast. These three IPs are
# the ones with complete, solver-closing formal proofs (see dslx-prove) and run
# cleanly under the current xlsynth-bin toolchain.
DSLX_UNIT_TESTS := hw/requant.x hw/agu_nhwc.x hw/mac64.x
# Same three IPs, proved rather than sampled. A #[quickcheck] is both fuzzed by
# dslx-fuzz and discharged exhaustively by dslx-prove; one list keeps the two in
# sync.
DSLX_PROOF_TESTS := $(DSLX_UNIT_TESTS)
# Engine-level files. convacc_proc.x / convacc_proofs.x currently SIGSEGV under
# xlsynth-bin and xlsynth-bin-0.61.0 -- a toolchain regression (reproduced on a
# clean checkout, not a source bug). They are excluded from the default test /
# proof runs and stay opt-in via dslx-test-convacc / dslx-prove-convacc.
DSLX_CONVACC_TESTS := hw/convacc_proc.x
DSLX_CONVACC_PROOFS := hw/convacc_proofs.x
# Deeper proofs (bigger unroll bound, minutes rather than seconds). Opt-in.
DSLX_PROOF_DEEP := hw/convacc_proofs_deep.x

# Bundled DSLX stdlib. prove_quickcheck_main -- and the interpreter whenever a
# file imports stdlib (requant.x imports `round`/`std`) -- needs
# --dslx_stdlib_path. Two install layouts exist in the wild:
#   * packaged xlsynth-bin (bin/ wrappers + libexec/) ships a *flat*
#     share/xlsynth/dslx_stdlib/*.x;
#   * older/manual installs extract the stdlib tarball as
#     xls/dslx/stdlib, i.e. share/xlsynth/dslx_stdlib/dslx/stdlib/.
# Probe flat first, fall back to nested. Derived from XLS_BIN (recursive).
DSLX_STDLIB := $(or $(wildcard $(XLS_BIN)/../share/xlsynth/dslx_stdlib),$(wildcard $(XLS_BIN)/../share/xlsynth/dslx_stdlib/dslx/stdlib))
DSLX_STDLIB_FLAGS = --dslx_stdlib_path=$(DSLX_STDLIB)

dslx-test: $(DSLX_UNIT_TESTS)
	@JOBS=$(JOBS) XLS_BIN=$(XLS_BIN) XLS_FLAGS="$(DSLX_STDLIB_FLAGS)" \
	 tools/run_dslx_parallel.sh $(DSLX_UNIT_TESTS)

# Fuzzed invariants (`#[quickcheck]`). Each property draws 1000 random inputs,
# so this is ~20k cases over the addressing layout, the requantiser, the MAC
# reduction and the AGU. `--seed` makes a run reproducible; drop it for a fresh
# sample every time.
QC_SEED ?= 20261030
dslx-fuzz:
	@JOBS=$(JOBS) XLS_BIN=$(XLS_BIN) \
	 XLS_FLAGS="$(DSLX_STDLIB_FLAGS) --evaluator=ir-interpreter --run_quickcheck_when_interpreting --seed=$(QC_SEED)" \
	 tools/run_dslx_parallel.sh $(DSLX_UNIT_TESTS)

# Formal proof of the same #[quickcheck] properties. `prove_quickcheck_main`
# asks the SMT solver whether *any* input can falsify a property and prints a
# counterexample if one exists, so it is exhaustive where dslx-fuzz samples 1000
# points. All 20 properties across the four files close in ~50 s, which makes
# this both stronger and faster than fuzzing; `dslx-fuzz` stays as the fallback
# for properties a prover cannot discharge.
dslx-prove:
	@JOBS=$(JOBS) XLS_BIN=$(XLS_BIN) DSLX_TOOL=prove_quickcheck_main \
	 XLS_FLAGS="$(DSLX_STDLIB_FLAGS)" \
	 tools/run_dslx_parallel.sh $(DSLX_PROOF_TESTS)

# Opt-in: the engine files excluded above. Known to SIGSEGV under the current
# toolchain (see DSLX_CONVACC_*); run only to reproduce / after a toolchain bump.
dslx-test-convacc:
	@JOBS=$(JOBS) XLS_BIN=$(XLS_BIN) XLS_FLAGS="$(DSLX_STDLIB_FLAGS)" \
	 tools/run_dslx_parallel.sh $(DSLX_CONVACC_TESTS)

dslx-prove-convacc:
	@JOBS=1 XLS_BIN=$(XLS_BIN) DSLX_TOOL=prove_quickcheck_main \
	 XLS_FLAGS="$(DSLX_STDLIB_FLAGS)" \
	 tools/run_dslx_parallel.sh $(DSLX_CONVACC_PROOFS)

# Same prover, larger unroll bounds: k = 16 needs 192 steps (~2.5 min) instead of
# the 128 the fast set uses. Kept out of full-test because of the runtime.
dslx-prove-deep:
	@JOBS=1 XLS_BIN=$(XLS_BIN) DSLX_TOOL=prove_quickcheck_main \
	 XLS_FLAGS="$(DSLX_STDLIB_FLAGS)" \
	 tools/run_dslx_parallel.sh $(DSLX_PROOF_DEEP)

# Engine-level equivalence / randomised harnesses. These drive the whole FSM
# over real images, so they are orders of magnitude slower per case than the
# unit tests -- for bulk coverage prefer `make cocotb-test`, where Verilator
# has already compiled the same engine to native code (~15 ms/case vs ~4 s/case
# in the interpreter).
#
# Both engine paths are generated from the same 30 shapes and run here: the
# array-backed `engine_step` (dslx_rand.x) and the RAM-facing
# `engine_step_mem` (dslx_equiv.x), so a divergence between them shows up as a
# failing case rather than as a cocotb-only symptom.
dslx-test-engine:
	$(PYTHON) tools/gen_dslx.py equiv 30 20261029
	$(PYTHON) tools/gen_dslx.py rand 30 20261029
	@JOBS=$(JOBS) XLS_BIN=$(XLS_BIN) tools/run_dslx_parallel.sh hw/dslx_equiv.x hw/dslx_rand.x

# Full-system harness: convacc_proc::ConvAccMem wired to three DSLX RAM models
# through real channels. This is the only place the RAM handshake -- request /
# response channels, the write-completion channel, byte strobes -- is exercised
# without a simulator; in the silicon flow that layer is the SV SRAM macro, so
# cocotb used to be the only way to reach it. The RAM stays in DSLX here; the
# synthesised design keeps --ram_configurations + the macro (a DSLX array in
# proc state would become ~3 Mbit of flip-flops).
#
# ~2.9 s per case in the interpreter and the interpreter is single-threaded,
# so the suite is split into independent shard files and run in parallel:
# 100 cases / 8 shards turns ~5 min into ~40 s.
FULLSYS_CASES ?= 100
FULLSYS_SHARDS ?= 8
dslx-test-fullsys:
	$(PYTHON) tools/gen_dslx.py fullsys $(FULLSYS_CASES) 20261029 $(FULLSYS_SHARDS)
	@JOBS=$(FULLSYS_SHARDS) XLS_BIN=$(XLS_BIN) tools/run_dslx_parallel.sh $$(cat hw/dslx_fullsys.files)

# NOTE: no separate typecheck target -- `dslx_interpreter_main` typechecks
# before it runs, and `typecheck_main` cannot parse the `chan<>` proc ports.

rtl: $(BUILD)/mac64.sv $(BUILD)/agu_nhwc.sv $(CORE_SV)

$(BUILD)/mac64.ir: hw/mac64.x
	ir_converter_main --top=mac64 $< > $@

$(BUILD)/mac64.opt.ir: $(BUILD)/mac64.ir
	opt_main $< > $@
	@if [ "$(IR_EQUIV)" = "0" ]; then \
		echo "skip IR equivalence (IR_EQUIV=0): __mac64__mac64"; \
	else \
		check_ir_equivalence_main --top=__mac64__mac64 $(BUILD)/mac64.ir $@; \
	fi

$(BUILD)/mac64.sv: $(BUILD)/mac64.opt.ir
	codegen_main --generator=pipeline --delay_model=unit --pipeline_stages=4 \
		--reset=rst --use_system_verilog --module_name=mac64_pipe \
		--output_verilog_path=$@ --output_signature_path=$(BUILD)/mac64.sig.textproto $<

$(BUILD)/mac64_reduced.ir: hw/mac64.x
	ir_converter_main --top=mac64_reduced $< > $@

$(BUILD)/mac64_reduced.opt.ir: $(BUILD)/mac64_reduced.ir
	opt_main $< > $@

$(BUILD)/mac64_reduced_core.sv: $(BUILD)/mac64_reduced.opt.ir
	codegen_main --generator=combinational --delay_model=unit --use_system_verilog \
		--module_name=mac64_reduced_core --output_verilog_path=$@ \
		--output_signature_path=$(BUILD)/mac64_reduced_core.sig.textproto $<

$(BUILD)/agu_nhwc.ir: hw/agu_nhwc.x
	ir_converter_main --top=conv_nhwc_address $< > $@

$(BUILD)/agu_nhwc.opt.ir: $(BUILD)/agu_nhwc.ir
	opt_main $< > $@
	@if [ "$(IR_EQUIV)" = "0" ]; then \
		echo "skip IR equivalence (IR_EQUIV=0): __agu_nhwc__conv_nhwc_address"; \
	else \
		check_ir_equivalence_main --top=__agu_nhwc__conv_nhwc_address $(BUILD)/agu_nhwc.ir $@; \
	fi

$(BUILD)/agu_nhwc.sv: $(BUILD)/agu_nhwc.opt.ir
	codegen_main --generator=pipeline --delay_model=unit --pipeline_stages=3 \
		--reset=rst --use_system_verilog --module_name=agu_nhwc_pipe \
		--output_verilog_path=$@ --output_signature_path=$(BUILD)/agu_nhwc.sig.textproto $<

$(BUILD)/requant8.ir: hw/requant.x
	ir_converter_main --top=requant8 $< > $@

$(BUILD)/requant8.opt.ir: $(BUILD)/requant8.ir
	opt_main $< > $@

$(BUILD)/requant8_core.sv: $(BUILD)/requant8.opt.ir
	codegen_main --generator=combinational --delay_model=unit --use_system_verilog \
		--module_name=requant8_core --output_verilog_path=$@ \
		--output_signature_path=$(BUILD)/requant8_core.sig.textproto $<

cocotb-test: $(CORE_SV)
	XEZIM=$(XEZIM) $(PYTHON) -m pytest -q tests/test_cocotb_convacc.py -n $(XDIST) --sim $(SIM)

# ---- native (C++) regression ----
# Same Verilated model and the same golden images as cocotb-test, driven from
# C++ instead of Python: no VPI per signal, no per-chunk interpreter startup,
# and the three SRAM images are loaded with one memcpy each. tests/cpp mirrors
# the Python reference model (planner/packer/image/reference) so cases can also
# be built -- and fuzzed -- natively.
NATIVE_BUILD := tests/cpp/build
# FuzzTest vendors abseil/re2/antlr and its own googletest, so it gets a separate
# build tree -- pulling it into $(NATIVE_BUILD) would make every ordinary
# `native-test` run rebuild against FuzzTest's gtest.
NATIVE_FUZZ_BUILD := tests/cpp/build-fuzz
NATIVE_JOBS ?= $(JOBS)
# Grid size multiplier for the native regression. 1 is the ~10k golden set;
# 10 widens the grid axes (stride, 7x7 kernels, more channels) and yields 100k
# cases. The cases are generated in-process (dut/generator.cc), so nothing is
# written to disk and the run does not depend on the page cache.
GEN_SCALE ?= 1
# Optional ceiling on simulated cycles, so a wide grid cannot silently turn into
# a 26x run. 0 means "no ceiling": keep every case the grid produces.
GEN_CYCLE_BUDGET ?= 0
GEN_BUDGET_FLAG := $(if $(filter-out 0,$(GEN_CYCLE_BUDGET)),--cycle-budget $(GEN_CYCLE_BUDGET))
# Which shape space to enumerate. auto keeps the --gen-scale behaviour;
# full enumerates every legal shape (~1.68M cases), which is what the 20-way
# CI regression runs.
GRID ?= auto
GRID_FLAG := $(if $(filter-out auto,$(GRID)),--grid $(GRID),)
# Split one grid across runners: NATIVE_SHARD=3/20 runs every twentieth case
# starting at 3. Disjoint and exhaustive, so twenty jobs check the whole grid.
NATIVE_SHARD ?= 0/1
# Pre-rendered corpus on disk. Only cocotb and the 140-case dataset still need
# it; the native bench generates its own.
NATIVE_CASES := $(BUILD)/cocotb_cases
# How many independent PRNG seeds the FuzzTest targets get. Each round runs the
# whole seed corpus, so this multiplies coverage without a libFuzzer runtime.
FUZZ_ROUNDS ?= 3
# Extra -D... passed to CMake (e.g. -DCONVACC_WITH_FUZZTEST=ON).
NATIVE_CMAKE_FLAGS ?=
# Compiler for the Verilated model only. This is where the harness spends ~99%
# of its time, and clang emits substantially better code for Verilator's output
# than g++ (7.0 vs 11.1 ms per case on the 1x grid). The probe picks the newest
# clang++ on PATH and falls back to Verilator's default g++ when there is none.
# Override with `make native-build NATIVE_VL_CXX=` to force g++.
NATIVE_VL_CXX ?= $(shell sh tools/find_clangxx.sh)
NATIVE_VL_CXX_BIN := $(firstword $(NATIVE_VL_CXX))
NATIVE_VL_CXX_FLAGS := $(wordlist 2,99,$(NATIVE_VL_CXX))
NATIVE_VL_FLAGS := $(if $(NATIVE_VL_CXX_BIN),-DVL_CXX=$(NATIVE_VL_CXX_BIN) '-DVL_CXX_FLAGS=$(NATIVE_VL_CXX_FLAGS)')
# Optimisation for the Verilated model. Verilator defaults the hot functions to
# -Os; -O3 is ~3% faster here. Everything past that (ThinLTO, -march=native,
# -fno-exceptions) lands inside the run-to-run noise, so it stays plain -O3.
NATIVE_VL_OPT ?= -O3
# Linker for the native executables. mold links the ~1 MB model in 0.02 s
# against BFD's 0.07 s, and the binary runs at the same speed
# (docs/native-tests.md). `make native-build NATIVE_LD=` goes back to the
# compiler driver's default; an LTO build needs NATIVE_LD=lld, because mold
# cannot read LLVM bitcode.
NATIVE_LD ?= mold

native-cases:
	$(PYTHON) tools/gen_cpp_cases.py

native-build:
	@test -f hw/convacc_top.sv || { echo "hw/convacc_top.sv is missing; run 'make rtl-gen' first" >&2; exit 1; }
	cmake -S tests/cpp -B $(NATIVE_BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release \
		$(NATIVE_VL_FLAGS) -DVL_OPT='$(NATIVE_VL_OPT)' -DCONVACC_LINKER='$(NATIVE_LD)' \
		$(NATIVE_CMAKE_FLAGS)
	cmake --build $(NATIVE_BUILD) -j $(NATIVE_JOBS)

# Full golden regression on the native driver. Run `make rtl-gen` first.
# The corpus is generated in-process: no `native-cases` prerequisite, no 2.5 GB
# of images to read.
native-test: native-build
	$(NATIVE_BUILD)/convacc_bench $(GRID_FLAG) --gen-scale $(GEN_SCALE) $(GEN_BUDGET_FLAG) \
		--shard $(NATIVE_SHARD) --jobs $(NATIVE_JOBS)

# The 10x grid: 100k cases, ~8 min on four cores. Not 10x the time of the 1x run
# but ~26x, because the widened axes (7x7 kernels, 12 output channels, 32x32
# GEMMs) make each case 2.6x longer. See docs/native-tests.md for the arithmetic
# and for the cycle-budget knob that caps it.
native-test-full:
	$(MAKE) GEN_SCALE=10 native-test

# The whole shape space: ~1.68M cases. Measured 106 cases/s on four cores, so
# one 1/20 shard (~83.8k cases) takes ~13 min; twenty machines run in parallel.
# Combine with NATIVE_SHARD=I/20 to spread it over twenty machines.
native-test-grid:
	$(MAKE) GRID=full native-test

# Same grid, capped to roughly the cost of a 1x run: fewer of the expensive
# shapes, same shape coverage.
native-test-wide:
	$(MAKE) GEN_SCALE=10 GEN_CYCLE_BUDGET=20000000 native-test

# Proves the in-process generator has not drifted from the Python model: dump a
# random sample of generated cases and re-derive each one with convacc/. Cheap
# (~1.5 s) because nothing is simulated.
CROSS_CHECK_CASES ?= 400
native-cross-check: native-build
	$(NATIVE_BUILD)/convacc_bench --gen-scale $(GEN_SCALE) $(GEN_BUDGET_FLAG) \
		--dump $(BUILD)/gen_sample.bin --dump-count $(CROSS_CHECK_CASES) \
		--dump-only --jobs $(NATIVE_JOBS) --quiet
	$(PYTHON) tools/verify_generated_cases.py $(BUILD)/gen_sample.bin

# C++ model vs the Python model, then the DUT vs both (GoogleTest). The first
# binary links no Verilator; the second is the only one that simulates.
native-model-test: native-cases native-build
	$(NATIVE_BUILD)/convacc_model_tests
	$(NATIVE_BUILD)/convacc_dut_tests

# Property-based tests. CONVACC_WITH_FUZZTEST pulls in abseil/re2/antlr, so the
# first build takes a while; later builds reuse the objects.
native-fuzz-build:
	cmake -S tests/cpp -B $(NATIVE_FUZZ_BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DCONVACC_WITH_FUZZTEST=ON $(NATIVE_CMAKE_FLAGS)
	cmake --build $(NATIVE_FUZZ_BUILD) -j $(NATIVE_JOBS) --target convacc_fuzz

# Serial: every FUZZ_TEST in one process, for a short random sample each.
native-fuzz: native-fuzz-build
	$(NATIVE_FUZZ_BUILD)/convacc_fuzz --gtest_repeat=$(FUZZ_ROUNDS)

# One process per core, one FUZZ_TEST per process. FuzzTest shards itself via
# GTEST_TOTAL_SHARDS/GTEST_SHARD_INDEX, so this is the whole implementation;
# FUZZ_SHARDS=auto picks min(usable cores, number of tests).
FUZZ_TIME ?= 60s
FUZZ_SHARDS ?= auto
# Wall-clock budget in seconds. 0 = one round (the default). A FUZZ_TEST in
# unit-test mode stops after 10000 mutations, 15-60 s here, so a long run must
# restart: each round gets a fresh PRNG seed.
FUZZ_DURATION ?= 0
native-fuzz-parallel: native-fuzz-build
	sh tools/run_fuzz_shards.sh $(NATIVE_FUZZ_BUILD)/convacc_fuzz \
		$(FUZZ_SHARDS) $(FUZZ_TIME) $(BUILD)/fuzz-logs $(FUZZ_DURATION)

# ---- coverage-guided fuzzing ----
# libFuzzer needs clang and its own runtime (libclang_rt.fuzzer). The model is
# instrumented, the DUT is not: the search runs at ~2.4k exec/s on shapes and
# the discovered corpus is replayed on the RTL afterwards (covfuzz-replay).
COVFUZZ_BUILD := tests/cpp/build-cov
COVFUZZ_CORPUS ?= $(BUILD)/covfuzz-corpus
COVFUZZ_SECONDS ?= 60
COVFUZZ_JOBS ?= 1
# libFuzzer ships with clang, so the fuzzer uses the same compiler the model
# uses (including the --gcc-install-dir the probe may have added: clang on
# Debian/Ubuntu can pick a gcc installation whose libstdc++ headers are absent).
COVFUZZ_CXX ?= $(NATIVE_VL_CXX)
COVFUZZ_CXX_BIN := $(firstword $(COVFUZZ_CXX))
COVFUZZ_CXX_FLAGS := $(wordlist 2,99,$(COVFUZZ_CXX))
COVFUZZ_SEEDS := tests/cpp/model/seeds

# The fuzzer only instruments the model and the codec, so this tree configures
# and builds without Verilator.
covfuzz-build:
	@test -d $(COVFUZZ_SEEDS) || $(PYTHON) tools/gen_fuzz_seeds.py
	cmake -S tests/cpp -B $(COVFUZZ_BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DCONVACC_WITH_COVFUZZ=ON -DCONVACC_WITH_GTEST=OFF -DCONVACC_WITH_DUT=OFF \
		-DCMAKE_CXX_COMPILER=$(COVFUZZ_CXX_BIN) -DCMAKE_CXX_FLAGS="$(COVFUZZ_CXX_FLAGS)" \
		$(NATIVE_CMAKE_FLAGS)
	cmake --build $(COVFUZZ_BUILD) -j $(NATIVE_JOBS) --target convacc_covfuzz

# Time-bounded search. The corpus is kept between runs, so repeating the target
# continues where the last run stopped instead of starting over.
covfuzz: covfuzz-build
	@mkdir -p $(COVFUZZ_CORPUS)
	@cp -n $(COVFUZZ_SEEDS)/* $(COVFUZZ_CORPUS)/ 2>/dev/null || true
	$(COVFUZZ_BUILD)/convacc_covfuzz $(COVFUZZ_CORPUS) \
		-max_total_time=$(COVFUZZ_SECONDS) -print_final_stats=1 -jobs=$(COVFUZZ_JOBS)

# Replay everything the fuzzer found on the Verilated DUT (byte-exact check).
covfuzz-replay: native-build
	$(NATIVE_BUILD)/convacc_fuzz_replay --corpus $(COVFUZZ_CORPUS) --jobs $(NATIVE_JOBS)

# The Verilated model is only rebuilt when the SV sources change, so switching
# compilers or -O levels needs a clean tree.
native-clean:
	rm -rf $(NATIVE_BUILD) $(NATIVE_FUZZ_BUILD) $(COVFUZZ_BUILD)

clean:
	rm -rf $(BUILD)/cocotb $(BUILD)/cocotb_cases* $(BUILD)/cpp_cases $(NATIVE_BUILD) $(NATIVE_FUZZ_BUILD) $(COVFUZZ_BUILD)
	rm -f $(BUILD)/*.ir $(BUILD)/*.sv $(BUILD)/*.textproto

# ---- DSLX -> SV generation: every SV under hw/ is produced here ----
# The three tensors are external 1RW RAMs behind XLS's RAM interface. The
# generated block exposes bare RAM ports (addr/wr_data/wr_mask/rd_mask/we/re +
# rd_data) with no ready/valid; the structural top in hw/convacc_top.sv wires
# them to the swappable SRAM macro. Request/response must land in different
# activations, hence pipeline_stages>=2 and the phase split in convacc_proc.x.
RAM_CONFIG := act:1RW:_act_req:_act_resp:_act_wc,wgt:1RW:_wgt_req:_wgt_resp:_wgt_wc,out:1RW:_out_req:_out_resp:_out_wc
rtl-gen:
	mkdir -p build
	cd hw && $(XLS_BIN)/ir_converter_main --top=ConvAccMemTop convacc_proc.x > ../build/convacc_mem.ir
	$(XLS_BIN)/opt_main build/convacc_mem.ir > build/convacc_mem.opt.ir
	$(XLS_BIN)/codegen_main --generator=pipeline --pipeline_stages=2 --worst_case_throughput=2 \
	    --codegen_version=1.5 --reset=rst --delay_model=unit --use_system_verilog \
	    --module_name=convacc_mem --output_verilog_path=hw/convacc_mem.sv \
	    --ram_configurations=$(RAM_CONFIG) \
	    build/convacc_mem.opt.ir
	.venv/bin/python tools/gen_sram.py

.PHONY: rtl-gen
