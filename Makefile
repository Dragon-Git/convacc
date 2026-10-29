PYTHON ?= .venv/bin/python
BUILD := build
CORE_SV := $(BUILD)/mac64_reduced_core.sv $(BUILD)/requant8_core.sv
XEZIM ?= $(HOME)/opt/xezim-bin/bin/xezim
XDIST ?= 8
# Parallel DSLX test files: the interpreter is single-threaded, but every .x
# file is independent (tools/run_dslx_parallel.sh).
JOBS ?= 8
# cocotb backend: verilator (default) or xezim -- see tests/conftest.py
SIM ?= verilator

.PHONY: test full-test software-test dslx-test dslx-fuzz dslx-prove dslx-prove-deep dslx-test-engine dslx-test-fullsys rtl cocotb-test clean

test: software-test dslx-test

fuzz: dslx-fuzz

full-test: test cocotb-test dslx-test-fullsys dslx-prove

software-test:
	$(PYTHON) -m unittest discover -s tests -v

# Hand-written unit tests over the DSLX sources. These are pure functions with
# small inputs, so the tree-walking interpreter is already fast (58 tests in
# ~1.2 s). The engine-level harnesses below are the slow ones and are opt-in.
DSLX_UNIT_TESTS := hw/requant.x hw/agu_nhwc.x hw/mac64.x hw/convacc_proc.x
# Properties that are proved rather than sampled (see dslx-prove). Kept out of
# DSLX_UNIT_TESTS because the interpreter's quickcheck path re-runs every
# property through the IR JIT, which aborts on engine-sized values.
DSLX_PROOF_TESTS := hw/convacc_proofs.x
# Deeper proofs (bigger unroll bound, minutes rather than seconds). Opt-in.
DSLX_PROOF_DEEP := hw/convacc_proofs_deep.x

dslx-test: $(DSLX_UNIT_TESTS)
	@JOBS=$(JOBS) XLS_BIN=$(XLS_BIN) tools/run_dslx_parallel.sh $(DSLX_UNIT_TESTS)

# Fuzzed invariants (`#[quickcheck]`). Each property draws 1000 random inputs,
# so this is ~20k cases over the addressing layout, the requantiser, the MAC
# reduction and the AGU. `--seed` makes a run reproducible; drop it for a fresh
# sample every time.
QC_SEED ?= 20261030
dslx-fuzz:
	@JOBS=$(JOBS) XLS_BIN=$(XLS_BIN) \
	 XLS_FLAGS="--evaluator=ir-interpreter --run_quickcheck_when_interpreting --seed=$(QC_SEED)" \
	 tools/run_dslx_parallel.sh $(DSLX_UNIT_TESTS)

# Formal proof of the same #[quickcheck] properties. `prove_quickcheck_main`
# asks the SMT solver whether *any* input can falsify a property and prints a
# counterexample if one exists, so it is exhaustive where dslx-fuzz samples 1000
# points. All 20 properties across the four files close in ~50 s, which makes
# this both stronger and faster than fuzzing; `dslx-fuzz` stays as the fallback
# for properties a prover cannot discharge.
dslx-prove:
	@JOBS=$(JOBS) XLS_BIN=$(XLS_BIN) DSLX_TOOL=prove_quickcheck_main \
	 tools/run_dslx_parallel.sh $(DSLX_UNIT_TESTS) $(DSLX_PROOF_TESTS)

# Same prover, larger unroll bounds: k = 16 needs 192 steps (~2.5 min) instead of
# the 128 the fast set uses. Kept out of full-test because of the runtime.
dslx-prove-deep:
	@JOBS=1 XLS_BIN=$(XLS_BIN) DSLX_TOOL=prove_quickcheck_main \
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
	$(PYTHON) tools/gen_dslx_equiv.py 30 20261029
	$(PYTHON) tools/gen_dslx_rand.py 30 20261029
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
	$(PYTHON) tools/gen_dslx_fullsys.py $(FULLSYS_CASES) 20261029 $(FULLSYS_SHARDS)
	@JOBS=$(FULLSYS_SHARDS) XLS_BIN=$(XLS_BIN) tools/run_dslx_parallel.sh $$(cat hw/dslx_fullsys.files)

# NOTE: no separate typecheck target -- `dslx_interpreter_main` typechecks
# before it runs, and `typecheck_main` cannot parse the `chan<>` proc ports.

rtl: $(BUILD)/mac64.sv $(BUILD)/agu_nhwc.sv $(CORE_SV)

$(BUILD)/mac64.ir: hw/mac64.x
	ir_converter_main --top=mac64 $< > $@

$(BUILD)/mac64.opt.ir: $(BUILD)/mac64.ir
	opt_main $< > $@
	check_ir_equivalence_main --top=__mac64__mac64 $(BUILD)/mac64.ir $@

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
	check_ir_equivalence_main --top=__agu_nhwc__conv_nhwc_address $(BUILD)/agu_nhwc.ir $@

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

clean:
	rm -rf $(BUILD)/cocotb $(BUILD)/cocotb_cases
	rm -f $(BUILD)/*.ir $(BUILD)/*.sv $(BUILD)/*.textproto

# ---- DSLX -> SV generation: every SV under hw/ is produced here ----
XLS_BIN ?= $(HOME)/opt/xlsynth-bin-0.59.0.1/bin
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
	    --codegen_version=1 --reset=rst --delay_model=unit --use_system_verilog \
	    --module_name=convacc_mem --output_verilog_path=hw/convacc_mem.sv \
	    --ram_configurations=$(RAM_CONFIG) \
	    build/convacc_mem.opt.ir
	.venv/bin/python tools/gen_sram.py

.PHONY: rtl-gen
