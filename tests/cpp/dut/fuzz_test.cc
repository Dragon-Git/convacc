// Property-based tests driven by FuzzTest.
//
// The golden regression only covers shapes the planner's grid enumerates. These
// tests let FuzzTest mutate the *shape* (not just the data), so the packing,
// descriptor encoding and DUT are exercised on combinations no hand-written
// table would think of -- odd kernel/stride/padding interactions, group counts
// that are legal but awkward, tiny reduction widths.
//
// Each case still runs a full RTL simulation, so the budget is wall-clock
// bounded in CI (`--fuzz_for=..`). Under the default (non-fuzzing) build these
// run as ordinary GoogleTest cases with a handful of sampled inputs.
//
// The suite is split into one FUZZ_TEST per planner path -- dense, grouped,
// depthwise, GEMM -- rather than one walk over everything. That keeps each
// test's domain tight enough to spend its whole budget on schedulable shapes,
// and it makes the suite shardable: FuzzTest honours GTEST_TOTAL_SHARDS and
// GTEST_SHARD_INDEX, so `make native-fuzz-parallel` runs one test per process
// and fills every core. A single test cannot be split that way, which is why
// the split is by path and not by test name.
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "convacc/reference.h"
#include "fuzztest/fuzztest.h"
#include "dut/dut.h"
#include "dut/run.h"

namespace {

using convacc::WeightTensor;

// A thread-local DUT keeps the Verilated model alive across fuzz cases; creating
// one per case would dominate the runtime.
convacc::Dut& dut() {
  static thread_local convacc::Dut instance;
  return instance;
}

struct ConvProblem {
  int hi;
  int wi;
  int ci;
  int co;
  int kh;
  int kw;
  int sh;
  int sw;
  int pad;
  int groups;
  uint32_t seed;

  bool operator==(const ConvProblem& other) const {
    return hi == other.hi && wi == other.wi && ci == other.ci && co == other.co &&
           kh == other.kh && kw == other.kw && sh == other.sh && sw == other.sw &&
           pad == other.pad && groups == other.groups && seed == other.seed;
  }
};

struct GemmProblem {
  int m;
  int k;
  int n;
  uint32_t seed;

  bool operator==(const GemmProblem& other) const {
    return m == other.m && k == other.k && n == other.n && seed == other.seed;
  }
};

std::vector<int8_t> random_bytes(std::mt19937& rng, size_t count, int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  std::vector<int8_t> out(count);
  for (auto& value : out) value = static_cast<int8_t>(dist(rng));
  return out;
}

convacc::QuantParams random_quant(std::mt19937& rng, int width) {
  convacc::QuantParams quant;
  quant.bias.resize(static_cast<size_t>(width));
  quant.scale.resize(static_cast<size_t>(width));
  quant.shift.resize(static_cast<size_t>(width));
  quant.zero_point.resize(static_cast<size_t>(width));
  std::uniform_int_distribution<int> bias(-40, 40);
  std::uniform_int_distribution<int> scale(1, 8);
  std::uniform_int_distribution<int> shift(0, 30);
  std::uniform_int_distribution<int> zp(-8, 8);
  for (int i = 0; i < width; ++i) {
    quant.bias[static_cast<size_t>(i)] = bias(rng);
    quant.scale[static_cast<size_t>(i)] = scale(rng);
    quant.shift[static_cast<size_t>(i)] = static_cast<uint8_t>(shift(rng));
    quant.zero_point[static_cast<size_t>(i)] = static_cast<int8_t>(zp(rng));
  }
  return quant;
}

// Runs one built case on the DUT and compares against the model's expectation.
::testing::AssertionResult RunsCleanly(const convacc::BuiltCase& built) {
  const convacc::Outcome run = convacc::run_built(dut(), built);
  if (run.ok) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << run.reason;
}

void ConvMatchesReference(const ConvProblem& p) {
  const int ph = p.pad * ((p.kh - 1) / 2);
  const int pw = p.pad * ((p.kw - 1) / 2);
  const int ho = (p.hi + 2 * ph - p.kh) / p.sh + 1;
  const int wo = (p.wi + 2 * pw - p.kw) / p.sw + 1;
  if (ho <= 0 || wo <= 0) return;
  if (p.ci % p.groups != 0 || p.co % p.groups != 0) return;
  if (1LL * p.hi * p.wi * p.ci > 4096 || 1LL * ho * wo * p.co > 4096) return;
  // Keep the simulation short: these are correctness probes, not benchmarks.
  if (1LL * ho * wo * p.co * p.kh * p.kw * (p.ci / p.groups) > 200000) return;

  convacc::ConvSpec spec;
  spec.n = 1;
  spec.hi = p.hi;
  spec.wi = p.wi;
  spec.ci = p.ci;
  spec.ho = ho;
  spec.wo = wo;
  spec.co = p.co;
  spec.kh = p.kh;
  spec.kw = p.kw;
  spec.sh = p.sh;
  spec.sw = p.sw;
  spec.ph = ph;
  spec.pw = pw;
  spec.groups = p.groups;
  try {
    spec.validate();
  } catch (const convacc::SpecError&) {
    return;
  }

  std::mt19937 rng(p.seed);
  WeightTensor weights;
  weights.shape = {p.kh, p.kw, p.groups, p.ci / p.groups, p.co / p.groups};
  weights.data = random_bytes(rng, static_cast<size_t>(weights.size()), -8, 8);
  const std::vector<int8_t> activations =
      random_bytes(rng, static_cast<size_t>(p.hi) * p.wi * p.ci, -32, 32);
  const convacc::QuantParams quant = random_quant(rng, p.co);

  const convacc::BuiltCase built = convacc::build_case(spec, weights, "HWGIO", activations, quant);
  ASSERT_TRUE(RunsCleanly(built));
}

void GemmMatchesReference(const GemmProblem& p) {
  if (1LL * p.m * p.k > 4096 || 1LL * p.k * p.n > 4096) return;
  if (1LL * p.m * p.n > 4096) return;

  convacc::GemmSpec spec;
  spec.m = p.m;
  spec.k = p.k;
  spec.n = p.n;

  std::mt19937 rng(p.seed);
  WeightTensor weights;
  weights.shape = {p.k, p.n};
  weights.data = random_bytes(rng, static_cast<size_t>(p.k) * p.n, -8, 8);
  const std::vector<int8_t> activations =
      random_bytes(rng, static_cast<size_t>(p.m) * p.k, -32, 32);
  const convacc::QuantParams quant = random_quant(rng, p.n);

  const convacc::BuiltCase built = convacc::build_case(spec, weights, "KN", activations, quant);
  ASSERT_TRUE(RunsCleanly(built));
}

// Shape fields shared by every conv path. `ci`, `co` and `groups` are passed in
// because they are the axis the planner branches on.
template <class Ci, class Co, class Groups>
auto ConvProblemDomain(Ci ci, Co co, Groups groups) {
  return fuzztest::StructOf<ConvProblem>(
      fuzztest::ElementOf<int>({1, 2, 3, 4, 5, 6}),  // hi
      fuzztest::ElementOf<int>({1, 2, 3, 4, 5, 6}),  // wi
      std::move(ci), std::move(co),
      fuzztest::ElementOf<int>({1, 3, 5}),  // kh
      fuzztest::ElementOf<int>({1, 3, 5}),  // kw
      fuzztest::InRange(1, 2),              // sh
      fuzztest::InRange(1, 2),              // sw
      fuzztest::InRange(0, 2),              // pad
      std::move(groups),
      fuzztest::Arbitrary<uint32_t>());  // seed
}

auto DenseConvProblems() {
  return ConvProblemDomain(fuzztest::InRange(1, 8), fuzztest::InRange(1, 8),
                           fuzztest::Just(1));
}

// `ci` and `co` are generated as multiples of the group count instead of being
// drawn freely and rejected by the divisibility check later: with 1..8 channels
// and 2..4 groups, the free draw accepts barely one pair in ten.
auto GroupedConvProblems() {
  return fuzztest::Map(
      [](int hi, int wi, int groups, int ci_per_group, int co_per_group, int kh, int kw,
         int sh, int sw, int pad, uint32_t seed) {
        return ConvProblem{hi, wi, groups * ci_per_group, groups * co_per_group, kh,
                           kw, sh, sw, pad, groups, seed};
      },
      fuzztest::ElementOf<int>({1, 2, 3, 4, 5, 6}),  // hi
      fuzztest::ElementOf<int>({1, 2, 3, 4, 5, 6}),  // wi
      fuzztest::ElementOf<int>({2, 3, 4}),           // groups
      fuzztest::InRange(1, 2),                       // ci = groups * 1..2 (<= 8)
      fuzztest::InRange(1, 2),                       // co = groups * 1..2 (<= 8)
      fuzztest::ElementOf<int>({1, 3, 5}),           // kh
      fuzztest::ElementOf<int>({1, 3, 5}),           // kw
      fuzztest::InRange(1, 2),                       // sh
      fuzztest::InRange(1, 2),                       // sw
      fuzztest::InRange(0, 2),                       // pad
      fuzztest::Arbitrary<uint32_t>());              // seed
}

// groups == ci == co. Kept to 1..4 channels: a depthwise case costs the same as
// a dense one of the same size, and the planner path is already covered at 4.
auto DepthwiseConvProblems() {
  return fuzztest::Map(
      [](int hi, int wi, int channels, int kh, int kw, int sh, int sw, int pad,
         uint32_t seed) {
        return ConvProblem{hi, wi, channels, channels, kh, kw, sh, sw, pad, channels, seed};
      },
      fuzztest::ElementOf<int>({1, 2, 3, 4, 5, 6}),  // hi
      fuzztest::ElementOf<int>({1, 2, 3, 4, 5, 6}),  // wi
      fuzztest::InRange(1, 4),                       // channels
      fuzztest::ElementOf<int>({1, 3, 5}),           // kh
      fuzztest::ElementOf<int>({1, 3, 5}),           // kw
      fuzztest::InRange(1, 2),                       // sh
      fuzztest::InRange(1, 2),                       // sw
      fuzztest::InRange(0, 1),                       // pad
      fuzztest::Arbitrary<uint32_t>());              // seed
}

auto GemmProblems() {
  return fuzztest::StructOf<GemmProblem>(
      fuzztest::InRange(1, 24),   // m
      fuzztest::InRange(1, 24),   // k
      fuzztest::InRange(1, 24),   // n
      fuzztest::Arbitrary<uint32_t>());
}

// One entry point per planner path. FUZZ_TEST takes the function itself (its
// name becomes the test name), so these wrappers are what the four tests share
// the checking code through.
void ConvDense(const ConvProblem& p) { ConvMatchesReference(p); }
void ConvGrouped(const ConvProblem& p) { ConvMatchesReference(p); }
void ConvDepthwise(const ConvProblem& p) { ConvMatchesReference(p); }
void Gemm(const GemmProblem& p) { GemmMatchesReference(p); }

}  // namespace

FUZZ_TEST(ConvAccFuzz, ConvDense).WithDomains(DenseConvProblems());
FUZZ_TEST(ConvAccFuzz, ConvGrouped).WithDomains(GroupedConvProblems());
FUZZ_TEST(ConvAccFuzz, ConvDepthwise).WithDomains(DepthwiseConvProblems());
FUZZ_TEST(ConvAccFuzz, Gemm).WithDomains(GemmProblems());
