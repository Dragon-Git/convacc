// Simulating one prepared case and judging the result.
//
// The golden regression, the corpus replay, the gtest grid and the FuzzTest
// suite each spelled this out by hand, and one of the four got the word count
// wrong: it rounded the expected byte count down, so `memcmp` read past the end
// of the read-back buffer and the last few bytes were never checked. Keeping
// the arithmetic in one place is the point of this file.
#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "convacc/compare.h"
#include "dut/dut.h"

namespace convacc {

struct Outcome {
  bool ok = false;
  bool built = false;   // false when the case never reached the DUT
  std::string reason;   // why it failed; empty on success
  long long cycles = 0;
  double sim_ms = 0.0;    // reset + load + run + read
  double build_ms = 0.0;  // materialise + pack, when the caller measured it
};

// Runs one prepared case. The clock covers exactly what the DUT does, so
// ms/case is comparable between callers.
//
// `expected` is passed as raw bytes because the two case sources hand over
// differently-typed buffers (uint8_t from the images on disk, int8_t from the
// model); the comparison is on the bit pattern either way.
inline Outcome run_built(Dut& dut, const SramImage& image, const void* expected,
                         size_t expected_bytes) {
  Outcome out;
  out.built = true;
  // The output window is read back a whole 64-bit word at a time, so round up:
  // the spare bytes of the last word are padding, and compare_bytes is told to
  // look at exactly `expected_bytes` of them.
  const int check_words = static_cast<int>((expected_bytes + 7) / 8);

  const auto start = std::chrono::steady_clock::now();
  dut.reset();
  dut.load(image, check_words);
  const SimResult sim = dut.run();
  std::vector<uint8_t> got;
  dut.read_output(check_words, got);
  out.sim_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                   .count();
  out.cycles = sim.cycles;

  if (sim.timed_out) {
    out.reason = "simulation timeout after " + std::to_string(sim.cycles) + " cycles";
    return out;
  }
  if (sim.error) {
    out.reason = "controller error";
    return out;
  }
  const Mismatch diff = compare_bytes(got.data(), got.size(),
                                      static_cast<const uint8_t*>(expected), expected_bytes);
  if (diff.bad) {
    out.reason = describe_mismatch(diff);
    return out;
  }
  out.ok = true;
  return out;
}

inline Outcome run_built(Dut& dut, const BuiltCase& built) {
  return run_built(dut, built.image, built.expected.data(), built.expected.size());
}

}  // namespace convacc
