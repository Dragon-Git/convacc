// Weight canonicalisation and MAC-array packing, mirrored from packer.py.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "convacc/planner.h"
#include "convacc/spec.h"

namespace convacc {

// A dense tensor with numpy-style shape and int8 payload in C order.
struct WeightTensor {
  std::vector<int> shape;
  std::vector<int8_t> data;

  int64_t size() const {
    int64_t total = 1;
    for (int d : shape) total *= d;
    return total;
  }
  int8_t at(std::initializer_list<int> idx) const {
    int64_t flat = 0;
    int i = 0;
    for (int v : idx) {
      flat = flat * shape[static_cast<size_t>(i++)] + v;
    }
    return data[static_cast<size_t>(flat)];
  }
};

struct PackedTask {
  Task task;
  std::vector<int8_t> weights;  // shape (n_tiles, k_groups, 8), C order
};

struct PackedPlan {
  OpSpec spec;
  Plan plan;
  std::vector<PackedTask> tasks;
};

// Canonical (kh, kw, groups, ci_per_group, co_per_group) weights.
// Throws SpecError on a layout/shape mismatch, exactly where the Python model
// raises ValueError.
std::vector<int8_t> canonical_conv_weights(const ConvSpec& spec, const WeightTensor& weights,
                                           const std::string& layout);

PackedPlan pack_weights(const OpSpec& spec, const WeightTensor& weights, const std::string& layout,
                        const Plan* plan = nullptr);

}  // namespace convacc
