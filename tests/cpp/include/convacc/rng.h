// The one PRNG in the native harness.
//
// std::mt19937_64 is used rather than a hand-written generator because its
// output sequence is fixed by the C++ standard, so a seed names the same stream
// on every machine and toolchain. The modulo is done here, not through
// std::uniform_int_distribution: that distribution's mapping from the engine's
// output to a range is implementation-defined, and the corpus has to be
// reproducible across libstdc++ and libc++.
#pragma once

#include <cstdint>
#include <random>

namespace convacc {

class Rng {
 public:
  explicit Rng(uint64_t seed) : engine_(seed) {}

  uint64_t next() { return engine_(); }

  // Uniform in [lo, hi]. The modulo bias is immaterial for test data.
  int uniform(int lo, int hi) {
    const uint64_t span = static_cast<uint64_t>(hi - lo) + 1;
    return lo + static_cast<int>(next() % span);
  }

 private:
  std::mt19937_64 engine_;
};

}  // namespace convacc
