// Task decomposition + cycle estimate, mirrored from convacc/planner.py.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "convacc/spec.h"

namespace convacc {

struct Task {
  std::string name;
  std::string mode;  // "dense" | "channelwise" | "gemm"
  int m = 0;
  int n = 0;
  int k = 0;
  int batches = 1;
  int mh = 1;
  int mw = 1;
  std::vector<std::pair<int, int>> taps;  // (kh, kw) pairs
  int ci_base = 0;
  int ci_count = 1;
  int co_base = 0;
  int group = 0;
  int phase_h = 0;
  int phase_w = 0;
  int out_step_h = 1;
  int out_step_w = 1;
  int r = 1;

  int o() const { return 8 / r; }
  int k_groups() const { return (k + r - 1) / r; }
  int m_tiles() const { return (m + 7) / 8; }
  int n_tiles() const { return (n + o() - 1) / o(); }
};

struct Plan {
  std::vector<Task> tasks;
  std::vector<int> task_cycles;
  long long useful_macs = 0;

  long long cycles() const {
    long long total = 0;
    for (int c : task_cycles) total += c;
    return total;
  }
  double utilization() const {
    const long long c = cycles();
    return c ? static_cast<double>(useful_macs) / static_cast<double>(64 * c) : 0.0;
  }
};

// Throws SpecError when the operator cannot be scheduled at all (matching the
// Python planner's ValueError).
Plan make_plan(const OpSpec& spec);

// (batch, oh, ow) of position `index` inside a task's m dimension.
struct OutCoord {
  int batch;
  int oh;
  int ow;
};
OutCoord output_coords(const Task& task, int index);

// The input pixel that output row `oh` reads through kernel row `kh`; false when
// the tap lands in the padding or a deconvolution phase has no integer solution.
// Split by axis because the planner prices a tile by counting valid rows and
// columns separately and multiplying them.
inline bool input_row(const ConvSpec& spec, int oh, int kh, int* ih) {
  if (spec.deconv) {
    const int numerator = oh + spec.ph - kh;
    if (numerator < 0 || numerator % spec.sh != 0) return false;
    const int h = numerator / spec.sh;
    if (h >= spec.hi) return false;
    *ih = h;
    return true;
  }
  const int h = oh * spec.sh + kh * spec.dh - spec.ph;
  if (h < 0 || h >= spec.hi) return false;
  *ih = h;
  return true;
}

inline bool input_col(const ConvSpec& spec, int ow, int kw, int* iw) {
  if (spec.deconv) {
    const int numerator = ow + spec.pw - kw;
    if (numerator < 0 || numerator % spec.sw != 0) return false;
    const int w = numerator / spec.sw;
    if (w >= spec.wi) return false;
    *iw = w;
    return true;
  }
  const int w = ow * spec.sw + kw * spec.dw - spec.pw;
  if (w < 0 || w >= spec.wi) return false;
  *iw = w;
  return true;
}

// Both axes at once, for the callers that have a tap and a position. The
// reference model spelled this out twice (dense and channelwise) and the
// planner a third time -- and only the planner's copy knew about deconvolution.
inline bool input_coord(const ConvSpec& spec, const OutCoord& c,
                        const std::pair<int, int>& tap, int* ih, int* iw) {
  return input_row(spec, c.oh, tap.first, ih) && input_col(spec, c.ow, tap.second, iw);
}

}  // namespace convacc
