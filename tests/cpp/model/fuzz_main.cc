// libFuzzer entry point for the coverage-guided operator search.
//
// All of the model runs in-process, so a single exec costs tens of
// microseconds and libFuzzer's coverage feedback can actually drive the search
// through the planner's branch structure. The Verilated DUT is *not* in this
// loop on purpose (it costs milliseconds per case); `convacc_fuzz_replay`
// replays the discovered corpus against the RTL afterwards.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "convacc/reference.h"
#include "model/case_codec.h"
#include "model/oracle.h"

namespace {

void dump_and_abort(const char* what, const std::string& detail, const uint8_t* data, size_t size) {
  std::fprintf(stderr, "\n=== %s ===\n%s\n", what, detail.c_str());
  convacc::FuzzCase c;
  if (convacc::decode_case(data, size, &c)) {
    std::fprintf(stderr, "case: %s\n", convacc::describe_case(c).c_str());
  }
  std::fprintf(stderr, "input (%zu bytes):", size);
  for (size_t i = 0; i < size && i < 64; ++i) std::fprintf(stderr, " %02x", data[i]);
  std::fprintf(stderr, "\n");
  std::abort();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  convacc::FuzzCase c;
  if (!convacc::decode_case(data, size, &c)) return 0;

  try {
    // Known hardware limitation found by this fuzzer (see
    // docs/coverage-guided-fuzzing.md): the channelwise lane fetch in
    // hw/convacc_proc.x (`ch_fill`) maps lane i to activation byte
    // `i / co_per_group`, counted from the tile's first channel, which only
    // works when a tile starts on a group boundary. Single-tile cases
    // (co <= 8) are fine because the extra lanes are masked off. Keep them out
    // of the default search; convacc_fuzz_replay still reproduces them from
    // tests/cpp/model/findings.
    if (!convacc::is_gemm(c.spec)) {
      const convacc::ConvSpec& s = convacc::as_conv(c.spec);
      if (s.channelwise() && s.co > 8 && 8 % s.co_per_group() != 0) return 0;
    }

    // Validation is part of the model contract (convacc/spec.py); exercising it
    // here is what keeps the error paths covered as well as the happy path.
    if (convacc::is_gemm(c.spec)) {
      convacc::as_gemm(c.spec).validate();
    } else {
      convacc::as_conv(c.spec).validate();
    }
    const convacc::BuiltCase built =
        convacc::build_case(c.spec, c.weights, c.layout, c.activations, c.quant);
    convacc::verify_case(c, built);
  } catch (const convacc::SpecError&) {
    // The same rejection the Python model raises for an unschedulable
    // operator. Not a finding, and not interesting coverage either.
    return 0;
  } catch (const std::exception& e) {
    dump_and_abort("invariant violated", e.what(), data, size);
  } catch (...) {
    dump_and_abort("unknown exception", "non-std exception", data, size);
  }
  return 0;
}
