// Deterministic case generation for the native regression.
//
// The golden corpus used to live on disk: one directory per case with three
// 128 KB images. At 10x scale that is 100k directories and 25 GB, which no
// reasonable page cache can hold (this sandbox is limited to 8 GiB), so every
// run re-read the whole thing over and over. The shapes are cheap to describe
// and the images are cheap to rebuild, so the corpus is generated in-process
// instead: a thread asks for case i, materialises it, simulates it, drops it.
//
// Everything here is a pure function of (scale, index). The shape list only
// depends on a fixed PRNG, never on thread scheduling, so a failing case can be
// reproduced exactly with `--filter`/`--only`.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "convacc/reference.h"

namespace convacc {

enum class Family { kDense, kDepthwise, kDeconv, kGemm };

const char* family_name(Family family);

// A case reduced to its shape plus a seed. Small enough that the full list can
// be shuffled in memory (a few dozen MB at scale 10) instead of being streamed
// through the filesystem.
struct ShapeCase {
  Family family = Family::kDense;
  ConvSpec conv;
  GemmSpec gemm;
  uint64_t seed = 0;
  std::string name;
};

// Which shape space to enumerate. `kSmall` and `kWide` sample a multiple of
// the base caps (1x ~10k, 10x ~100k cases). `kFull` keeps *every* legal
// combination of the widened axes -- about 1.68M cases -- which is the grid
// the 20-machine CI regression runs. Everything is a pure function of the
// grid, the multiplier and the budget, so a shard needs no coordination.
enum class Grid { kSmall, kWide, kFull };

// `multiplier` scales the per-family caps for kSmall/kWide; kFull ignores it
// and takes the whole pool. `cycle_budget` (0 = unlimited) caps the total
// simulated cycles, split across families in proportion to their cost.
std::vector<ShapeCase> make_shape_list(Grid grid, int multiplier,
                                       long long cycle_budget = 0);

// Compatibility entry point: the gtest grid experiment still passes a scale.
inline std::vector<ShapeCase> make_shape_list(int scale, long long cycle_budget = 0) {
  return make_shape_list(scale > 1 ? Grid::kWide : Grid::kSmall, scale, cycle_budget);
}

struct GeneratedCase {
  std::string name;
  OpSpec spec;
  std::string layout;
  WeightTensor weights;
  std::vector<int8_t> activations;
  QuantParams quant;
};

// Weights, activations and quantisation parameters for one shape. Depends only
// on `shape.seed`, so it can run concurrently without shared state.
GeneratedCase materialise(const ShapeCase& shape);

}  // namespace convacc
