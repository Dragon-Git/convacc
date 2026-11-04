#include "dut/generator.h"

#include "convacc/rng.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace convacc {
namespace {

// Uniform sample of `k` elements without replacement, in O(k) instead of the
// O(n) a full shuffle would cost (n is ~500k for the dense grid at scale 10).
void sample_in_place(std::vector<ShapeCase>* pool, size_t k, Rng* rng) {
  const size_t n = pool->size();
  if (k > n) k = n;
  for (size_t i = 0; i < k; ++i) {
    const size_t j = i + static_cast<size_t>(rng->next() % (n - i));
    std::swap((*pool)[i], (*pool)[j]);
  }
  pool->resize(k);
}

// Small deterministic seeds keep the corpus reproducible across machines while
// still giving each case its own data.
constexpr uint64_t kShapeSeed = 0x636f6e76616363ULL;  // "convacc"

// Index suffix width follows the case count, so names sort in index order and
// the labels stay aligned when the grid grows.
void label(std::vector<ShapeCase>* cases, const char* prefix) {
  const size_t last = cases->empty() ? 0 : cases->size() - 1;
  const size_t width = std::to_string(last).size();
  for (size_t i = 0; i < cases->size(); ++i) {
    std::string index = std::to_string(i);
    index.insert(0, width - index.size(), '0');
    (*cases)[i].name = std::string(prefix) + "_" + index;
  }
}

void assign_seeds(std::vector<ShapeCase>* cases, uint64_t family_seed) {
  Rng rng = Rng(kShapeSeed + family_seed);
  for (auto& c : *cases) c.seed = rng.next();
}

// `Plan::cycles()` counts the MAC loop only. Measured against the Verilated DUT
// the real cost is ~36x that: staging activations into the SRAM, filling tiles
// and draining the pipeline dominate for small shapes. Calibration (1x corpus,
// 9,972 cases): 1.99e7 simulated cycles against 5.5e5 planned, and 5.2e8 vs
// 1.57e7 at scale 10 -- both land on 33-36, so a single factor is enough for
// budgeting.
constexpr long long kPlanToSim = 36;

// Plan cost of one case, used only to spend a cycle budget. `make_plan` is pure
// arithmetic (< 50 us), so pricing a whole 100k corpus costs about a second.
long long planned_cycles(const ShapeCase& shape) {
  try {
    const long long planned = shape.family == Family::kGemm ? make_plan(shape.gemm).cycles()
                                                            : make_plan(shape.conv).cycles();
    return planned * kPlanToSim;
  } catch (const SpecError&) {
    return 0;
  }
}

// Keeps the first cases that fit. The pool was already sampled uniformly, so
// the result is a uniform subset with a bounded cost rather than a biased
// "only the small shapes" subset.
void trim_to_budget(std::vector<ShapeCase>* pool, long long budget) {
  if (budget <= 0) return;
  long long total = 0;
  size_t keep = 0;
  for (; keep < pool->size(); ++keep) {
    const long long cost = planned_cycles((*pool)[keep]);
    if (keep > 0 && total + cost > budget) break;
    total += cost;
  }
  pool->resize(keep);
}

std::vector<int> range(int lo, int hi) {
  std::vector<int> out;
  for (int v = lo; v <= hi; ++v) out.push_back(v);
  return out;
}

// One place for every axis. Three grids share the four generators:
//
//   kSmall  1x,~10k cases    the original golden corpus
//   kWide   10x,~100k cases  wider axes (stride 2, 7x7 kernels, 32x32 GEMMs)
//   kFull   ~1.68M cases     every legal combination of the widened axes
//
// kFull adds *density*, not bigger shapes: n, hi/wi and pad stay capped,
// because raising them multiplies simulated cycles. Measured on the same
// pool: pad=2 takes the mean from 13.4k to 45.8k cycles/case, while stride 3
// adds shapes *and* lowers the mean (a larger stride means a smaller output).
// That is what keeps 1.68M cases inside a 20-minute / 20-machine budget.
struct Axes {
  std::vector<int> dense_hi, dense_ci, dense_co, ks, strides;
  std::vector<int> depth_hi, mults;
  std::vector<int> deconv_hi, deconv_ci, deconv_co, deconv_pads;
  int gemm_limit = 16;
  bool full = false;
  size_t caps[4] = {};
  // Budget shares. For kFull the dense family is ~95% of the simulated
  // cycles, not 53% of the case caps, so the weights follow measured cost.
  long long weights[4] = {5300, 1300, 1400, 2000};
};

Axes axes_for(Grid grid, int multiplier) {
  Axes ax;
  if (grid == Grid::kSmall) {
    ax.dense_hi = range(2, 6);
    ax.dense_ci = {1, 2, 3, 4, 8};
    ax.dense_co = range(1, 8);
    ax.ks = {1, 3, 5};
    ax.strides = {1};
    ax.depth_hi = range(2, 4);
    ax.mults = {1, 2, 3};
    ax.deconv_hi = range(2, 5);
    ax.deconv_ci = {1, 2, 3};
    ax.deconv_co = range(1, 4);
    ax.deconv_pads = {1};
    ax.gemm_limit = 16;
  } else if (grid == Grid::kWide) {
    ax.dense_hi = range(2, 8);
    ax.dense_ci = {1, 2, 3, 4, 5, 6, 8};
    ax.dense_co = range(1, 10);
    ax.dense_co.push_back(12);
    ax.ks = {1, 3, 5, 7};
    ax.strides = {1, 2};
    ax.depth_hi = range(2, 8);
    ax.mults = {1, 2, 3};
    ax.deconv_hi = range(2, 9);
    ax.deconv_ci = {1, 2, 3, 4};
    ax.deconv_co = range(1, 6);
    ax.deconv_pads = {0, 1};
    ax.gemm_limit = 32;
  } else {
    ax.dense_hi = range(2, 8);
    ax.dense_ci = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    ax.dense_co = range(1, 14);
    ax.dense_co.push_back(16);
    ax.ks = {1, 3, 5, 7, 9};
    ax.strides = {1, 2, 3};
    ax.depth_hi = range(2, 10);
    ax.mults = {1, 2, 3, 4};
    ax.deconv_hi = range(2, 11);
    ax.deconv_ci = {1, 2, 3, 4, 5};
    ax.deconv_co = range(1, 8);
    ax.deconv_pads = {0, 1};
    ax.gemm_limit = 40;
    ax.full = true;
    ax.weights[0] = 952;
    ax.weights[1] = 4;
    ax.weights[2] = 24;
    ax.weights[3] = 18;
  }

  const size_t base[4] = {5300, 1300, 1400, 2000};
  const size_t m = static_cast<size_t>(multiplier < 1 ? 1 : multiplier);
  for (int f = 0; f < 4; ++f) {
    // 4M exceeds every pool, so sample_in_place keeps all of it: kFull is
    // "the whole shape space", not a sample of a fixed size.
    ax.caps[f] = ax.full ? 4000000 : base[f] * m;
  }
  return ax;
}

// ---------------------------------------------------------------------------
// Dense convolution
// ---------------------------------------------------------------------------
void add_dense(const Axes& ax, long long budget, std::vector<ShapeCase>* out) {
  const std::vector<int>& his = ax.dense_hi;
  const std::vector<int>& wis = his;
  const std::vector<int>& cis = ax.dense_ci;
  const std::vector<int>& cos_ = ax.dense_co;
  const std::vector<int>& ks = ax.ks;
  const std::vector<int>& strides = ax.strides;

  std::vector<ShapeCase> pool;
  pool.reserve(ax.full ? 2000000 : 600000);
  for (int n : {1, 2})
    for (int hi : his)
      for (int wi : wis)
        for (int ci : cis)
          for (int co : cos_)
            for (int kh : ks)
              for (int kw : ks)
                for (int pad : {0, 1})
                  for (int sh : strides) {
                    const int ph = pad * ((kh - 1) / 2);
                    const int pw = pad * ((kw - 1) / 2);
                    const int ho = (hi + 2 * ph - kh) / sh + 1;
                    const int wo = (wi + 2 * pw - kw) / sh + 1;
                    if (ho <= 0 || wo <= 0) continue;
                    if (n * hi * wi * ci > 8192 || n * ho * wo * co > 8192) continue;
                    // ci == 1 makes the operator channelwise, and a channelwise
                    // n-tile holds at most eight channels (the planner rejects
                    // co_per_group > 8).
                    if (ci == 1 && co > 8) continue;

                    ShapeCase c;
                    c.family = Family::kDense;
                    c.conv = ConvSpec{};
                    c.conv.n = n;
                    c.conv.hi = hi;
                    c.conv.wi = wi;
                    c.conv.ci = ci;
                    c.conv.ho = ho;
                    c.conv.wo = wo;
                    c.conv.co = co;
                    c.conv.kh = kh;
                    c.conv.kw = kw;
                    c.conv.sh = sh;
                    c.conv.sw = sh;
                    c.conv.ph = ph;
                    c.conv.pw = pw;
                    pool.push_back(std::move(c));
                  }

  Rng rng = Rng(kShapeSeed ^ 0x11);
  sample_in_place(&pool, ax.caps[0], &rng);
  trim_to_budget(&pool, budget);
  label(&pool, "dense");
  assign_seeds(&pool, 0x11);
  out->insert(out->end(), pool.begin(), pool.end());
}

// ---------------------------------------------------------------------------
// Depthwise (groups == ci, one task per input channel)
// ---------------------------------------------------------------------------
void add_depthwise(const Axes& ax, long long budget, std::vector<ShapeCase>* out) {
  const std::vector<int>& his = ax.depth_hi;
  const std::vector<int>& wis = his;
  const std::vector<int> cis = {1, 2, 3, 4, 6, 8};
  const std::vector<int>& mults = ax.mults;
  const std::vector<int>& ks = ax.ks;

  std::vector<ShapeCase> pool;
  for (int hi : his)
    for (int wi : wis)
      for (int ci : cis)
        for (int mult : mults)
          for (int kh : ks)
            for (int kw : ks)
              for (int pad : {0, 1}) {
                const int co = ci * mult;
                // The channelwise lane fetch is only correct when a tile starts
                // on a group boundary; above eight channels a tile can straddle
                // two groups. Single-tile cases are fine because the extra
                // lanes are masked off (see docs/coverage-guided-fuzzing.md).
                if (co > 8) continue;
                const int ph = pad * ((kh - 1) / 2);
                const int pw = pad * ((kw - 1) / 2);
                const int ho = hi + 2 * ph - kh + 1;
                const int wo = wi + 2 * pw - kw + 1;
                if (ho <= 0 || wo <= 0) continue;
                if (hi * wi * ci > 8192 || ho * wo * co > 8192) continue;

                ShapeCase c;
                c.family = Family::kDepthwise;
                c.conv = ConvSpec{};
                c.conv.n = 1;
                c.conv.hi = hi;
                c.conv.wi = wi;
                c.conv.ci = ci;
                c.conv.ho = ho;
                c.conv.wo = wo;
                c.conv.co = co;
                c.conv.kh = kh;
                c.conv.kw = kw;
                c.conv.ph = ph;
                c.conv.pw = pw;
                c.conv.groups = ci;
                pool.push_back(std::move(c));
              }

  Rng rng = Rng(kShapeSeed ^ 0x22);
  sample_in_place(&pool, ax.caps[1], &rng);
  trim_to_budget(&pool, budget);
  label(&pool, "depth");
  assign_seeds(&pool, 0x22);
  out->insert(out->end(), pool.begin(), pool.end());
}

// ---------------------------------------------------------------------------
// Transposed convolution
// ---------------------------------------------------------------------------
void add_deconv(const Axes& ax, long long budget, std::vector<ShapeCase>* out) {
  const std::vector<int>& his = ax.deconv_hi;
  const std::vector<int>& wis = his;
  const std::vector<int>& cis = ax.deconv_ci;
  const std::vector<int>& cos_ = ax.deconv_co;
  const std::vector<int> ks = {1, 3, 5};
  const std::vector<int>& pads = ax.deconv_pads;

  std::vector<ShapeCase> pool;
  for (int hi : his)
    for (int wi : wis)
      for (int ci : cis)
        for (int co : cos_)
          for (int kh : ks)
            for (int kw : ks)
              for (int pad : pads) {
                const int sh = 2;
                const int sw = 2;
                const int ho = (hi - 1) * sh - 2 * pad + (kh - 1) + 1;
                const int wo = (wi - 1) * sw - 2 * pad + (kw - 1) + 1;
                if (ho <= 0 || wo <= 0) continue;
                if (hi * wi * ci > 8192 || ho * wo * co > 8192) continue;

                ShapeCase c;
                c.family = Family::kDeconv;
                c.conv = ConvSpec{};
                c.conv.n = 1;
                c.conv.hi = hi;
                c.conv.wi = wi;
                c.conv.ci = ci;
                c.conv.ho = ho;
                c.conv.wo = wo;
                c.conv.co = co;
                c.conv.kh = kh;
                c.conv.kw = kw;
                c.conv.sh = sh;
                c.conv.sw = sw;
                c.conv.ph = pad;
                c.conv.pw = pad;
                c.conv.deconv = true;
                // The scheduler and the DUT share one task model; shapes it
                // cannot represent (e.g. wo == 1 with the only aligned phase
                // out of range) would fail at build time, so drop them here.
                try {
                  make_plan(c.conv);
                } catch (const SpecError&) {
                  continue;
                }
                pool.push_back(std::move(c));
              }

  Rng rng = Rng(kShapeSeed ^ 0x33);
  sample_in_place(&pool, ax.caps[2], &rng);
  trim_to_budget(&pool, budget);
  label(&pool, "deconv");
  assign_seeds(&pool, 0x33);
  out->insert(out->end(), pool.begin(), pool.end());
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------
void add_gemm(const Axes& ax, long long budget, std::vector<ShapeCase>* out) {
  const int limit = ax.gemm_limit;
  std::vector<ShapeCase> pool;
  pool.reserve(static_cast<size_t>(limit) * limit * limit);
  for (int m = 1; m <= limit; ++m)
    for (int k = 1; k <= limit; ++k)
      for (int n = 1; n <= limit; ++n) {
        ShapeCase c;
        c.family = Family::kGemm;
        c.gemm = GemmSpec{m, k, n};
        pool.push_back(std::move(c));
      }

  Rng rng = Rng(kShapeSeed ^ 0x44);
  sample_in_place(&pool, ax.caps[3], &rng);
  trim_to_budget(&pool, budget);
  label(&pool, "gemm");
  assign_seeds(&pool, 0x44);
  out->insert(out->end(), pool.begin(), pool.end());
}

std::vector<int8_t> fill_i8(Rng* rng, size_t count, int lo, int hi) {
  std::vector<int8_t> data(count);
  for (auto& v : data) v = static_cast<int8_t>(rng->uniform(lo, hi));
  return data;
}

QuantParams random_quant(Rng* rng, int width) {
  QuantParams q;
  q.bias.resize(static_cast<size_t>(width));
  q.scale.resize(static_cast<size_t>(width));
  q.shift.resize(static_cast<size_t>(width));
  q.zero_point.resize(static_cast<size_t>(width));
  for (int i = 0; i < width; ++i) {
    q.bias[static_cast<size_t>(i)] = rng->uniform(-32, 32);
    q.scale[static_cast<size_t>(i)] = rng->uniform(1, 8);
    q.shift[static_cast<size_t>(i)] = static_cast<uint8_t>(rng->uniform(0, 7));
    q.zero_point[static_cast<size_t>(i)] = static_cast<int8_t>(rng->uniform(-8, 7));
  }
  return q;
}

}  // namespace

const char* family_name(Family family) {
  switch (family) {
    case Family::kDense: return "dense";
    case Family::kDepthwise: return "depth";
    case Family::kDeconv: return "deconv";
    case Family::kGemm: return "gemm";
  }
  return "?";
}

std::vector<ShapeCase> make_shape_list(Grid grid, int multiplier,
                                       long long cycle_budget) {
  const Axes ax = axes_for(grid, multiplier);
  // Families get a budget proportional to their cost, so a capped run stays
  // balanced instead of spending everything on the first family.
  long long total_weight = 0;
  for (long long w : ax.weights) total_weight += w;
  auto share = [&](int index) -> long long {
    return cycle_budget > 0 ? cycle_budget * ax.weights[index] / total_weight : 0;
  };

  std::vector<ShapeCase> all;
  // One allocation instead of four doublings: the full grid is 1.68M cases.
  if (ax.full) all.reserve(1700000);
  add_dense(ax, share(0), &all);
  add_depthwise(ax, share(1), &all);
  add_deconv(ax, share(2), &all);
  add_gemm(ax, share(3), &all);
  return all;
}

GeneratedCase materialise(const ShapeCase& shape) {
  GeneratedCase c;
  c.name = shape.name;
  Rng rng = Rng(shape.seed);

  if (shape.family == Family::kGemm) {
    const GemmSpec& g = shape.gemm;
    c.spec = g;
    c.layout = "KN";
    c.weights.shape = {g.k, g.n};
    c.weights.data = fill_i8(&rng, static_cast<size_t>(g.k) * static_cast<size_t>(g.n), -5, 5);
    c.activations = fill_i8(&rng, static_cast<size_t>(g.m) * static_cast<size_t>(g.k), -32, 32);
    c.quant = random_quant(&rng, g.n);
    return c;
  }

  const ConvSpec& s = shape.conv;
  c.spec = s;
  switch (shape.family) {
    case Family::kDense:
      c.layout = "HWGIO";
      c.weights.shape = {s.kh, s.kw, 1, s.ci, s.co};
      break;
    case Family::kDepthwise:
      c.layout = "HWM";
      c.weights.shape = {s.kh, s.kw, s.co};
      break;
    case Family::kDeconv:
      c.layout = "HWIO";
      c.weights.shape = {s.kh, s.kw, s.ci, s.co};
      break;
    case Family::kGemm:
      break;  // handled above
  }

  int64_t elements = 1;
  for (int d : c.weights.shape) elements *= d;
  c.weights.data = fill_i8(&rng, static_cast<size_t>(elements), -5, 5);
  c.activations = fill_i8(
      &rng, static_cast<size_t>(s.n) * s.hi * s.wi * s.ci, -32, 32);
  c.quant = random_quant(&rng, s.co);
  return c;
}

}  // namespace convacc
