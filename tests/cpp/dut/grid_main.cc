// The golden grid, expressed as gtest tests so a standard runner can shard it.
//
// This is an experiment, not the default driver: one test per *chunk* of cases,
// because a test-per-case would rebuild the Verilated model for every case
// (~300 ms against ~6 ms of simulation). Chunks of 250 cases amortise that to
// 1.2 ms per case while still giving the runner hundreds of independent items
// to schedule.
//
// Off by default; build it with -DCONVACC_WITH_GRID_SHARDS=ON. GoogleTest
// shards itself through GTEST_TOTAL_SHARDS/GTEST_SHARD_INDEX, so no external
// runner is needed:
//
//   convacc_bench_gtest --gtest_brief=1
//   GTEST_TOTAL_SHARDS=4 GTEST_SHARD_INDEX=2 convacc_bench_gtest   # one shard
//
// Everything is configured through the environment because gtest instantiation
// happens at static-init time, before any flag could be parsed:
//
//   CONVACC_GEN_SCALE      1 (default) or 10
//   CONVACC_CYCLE_BUDGET   0 = unlimited
//   CONVACC_CHUNKS         0 = 250 cases per chunk
//
// The chunk boundaries stay a runtime division: the case list comes from the
// generator and depends on the scale, so there is nothing to fold at compile
// time. A compile-time chunk index (index_sequence + testing::RegisterTest)
// would only make the registration static, which is what Range() already does
// for us without the boilerplate.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dut/generator.h"
#include "dut/case.h"
#include "convacc/cli.h"
#include "dut/dut.h"

namespace {

// gtest instantiates its tests during static initialisation, so a bad value has
// no way to become a test failure -- abort with a message instead of silently
// running a different corpus.
int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  int parsed = 0;
  if (!convacc::parse_int(value, &parsed)) {
    std::fprintf(stderr, "error: %s='%s' is not a decimal integer\n", name, value);
    std::abort();
  }
  return parsed;
}

long long env_ll(const char* name, long long fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  long long parsed = 0;
  if (!convacc::parse_ll(value, &parsed)) {
    std::fprintf(stderr, "error: %s='%s' is not a decimal integer\n", name, value);
    std::abort();
  }
  return parsed;
}

const std::vector<convacc::ShapeCase>& shapes() {
  static const std::vector<convacc::ShapeCase> list =
      convacc::make_shape_list(env_int("CONVACC_GEN_SCALE", 1),
                               env_ll("CONVACC_CYCLE_BUDGET", 0));
  return list;
}

size_t chunk_count() {
  const int requested = env_int("CONVACC_CHUNKS", 0);
  if (requested > 0) return static_cast<size_t>(requested);
  return std::max<size_t>(1, (shapes().size() + 249) / 250);
}

// One model per process: the chunks inside a process run in sequence, and under
// a sharded run each process owns a handful of them, so this is also one
// elaboration per process.
convacc::Dut& shared_dut() {
  static convacc::Dut dut;
  return dut;
}

class GridChunk : public ::testing::TestWithParam<int> {};

TEST_P(GridChunk, Cases) {
  const std::vector<convacc::ShapeCase>& all = shapes();
  const size_t chunks = chunk_count();
  const size_t index = static_cast<size_t>(GetParam());
  const size_t begin = all.size() * index / chunks;
  const size_t end = all.size() * (index + 1) / chunks;

  convacc::Dut& dut = shared_dut();
  size_t failed = 0;
  long long cycles = 0;
  double sim_ms = 0.0;

  for (size_t i = begin; i < end; ++i) {
    const convacc::Outcome run = convacc::run_shape(dut, all[i]);
    sim_ms += run.sim_ms;
    cycles += run.cycles;
    if (run.ok) continue;
    ++failed;
    ADD_FAILURE() << all[i].name << ": " << run.reason;
  }

  const size_t cases = end - begin;
  if (cases) {
    std::printf("[chunk %zu] %zu cases, %.2f ms/case, %lld cycles total, %zu failed\n", index,
                cases, sim_ms / static_cast<double>(cases), cycles, failed);
  }
  EXPECT_EQ(failed, 0u);
}

INSTANTIATE_TEST_SUITE_P(Grid, GridChunk, ::testing::Range(0, static_cast<int>(chunk_count())));

}  // namespace
