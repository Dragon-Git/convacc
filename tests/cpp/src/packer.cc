#include "convacc/packer.h"

#include <cstring>
#include <string>

namespace convacc {
namespace {

std::vector<int8_t> canonical_channelwise(const ConvSpec& spec, const WeightTensor& w) {
  const std::vector<int> want = {spec.kh, spec.kw, spec.co};
  if (w.shape != want) {
    throw SpecError("HWM weights must have shape (" + std::to_string(spec.kh) + ", " +
                    std::to_string(spec.kw) + ", " + std::to_string(spec.co) + ")");
  }
  // reshape(kh, kw, ci, co_per_group)[:, :, :, None, :]
  const int cpg = spec.co_per_group();
  std::vector<int8_t> out(static_cast<size_t>(spec.kh) * spec.kw * spec.ci * 1 * cpg);
  size_t o = 0;
  for (int kh = 0; kh < spec.kh; ++kh) {
    for (int kw = 0; kw < spec.kw; ++kw) {
      for (int ci = 0; ci < spec.ci; ++ci) {
        for (int co = 0; co < cpg; ++co) {
          out[o++] = w.at({kh, kw, ci * cpg + co});
        }
      }
    }
  }
  return out;
}

}  // namespace

std::vector<int8_t> canonical_conv_weights(const ConvSpec& spec, const WeightTensor& weights,
                                           const std::string& layout) {
  const int cipg = spec.ci_per_group();
  const int copg = spec.co_per_group();
  const std::vector<int> want = {spec.kh, spec.kw, spec.groups, cipg, copg};
  std::vector<int8_t> out;
  out.reserve(static_cast<size_t>(spec.kh) * spec.kw * spec.groups * cipg * copg);

  if (layout == "HWGIO") {
    if (weights.shape != want) {
      throw SpecError("canonical weights shape mismatch for HWGIO");
    }
    out = weights.data;
  } else if (layout == "HWIO" && spec.groups == 1) {
    const std::vector<int> expect = {spec.kh, spec.kw, spec.ci, spec.co};
    if (weights.shape != expect) throw SpecError("canonical weights shape mismatch for HWIO");
    for (int kh = 0; kh < spec.kh; ++kh)
      for (int kw = 0; kw < spec.kw; ++kw)
        for (int ci = 0; ci < spec.ci; ++ci)
          for (int co = 0; co < spec.co; ++co) out.push_back(weights.at({kh, kw, ci, co}));
  } else if (layout == "OIHW" && spec.groups == 1) {
    // Mirrors `weights.transpose(1, 2, 3, 0)[:, :, None, :, :]`: the source is
    // indexed as (co, kh, kw, ci), which is what that transpose chain assumes.
    const std::vector<int> expect = {spec.co, spec.kh, spec.kw, spec.ci};
    if (weights.shape != expect) throw SpecError("canonical weights shape mismatch for OIHW");
    for (int kh = 0; kh < spec.kh; ++kh)
      for (int kw = 0; kw < spec.kw; ++kw)
        for (int ci = 0; ci < spec.ci; ++ci)
          for (int co = 0; co < spec.co; ++co) out.push_back(weights.at({co, kh, kw, ci}));
  } else if (layout == "GOIHW") {
    const std::vector<int> expect = {spec.groups, spec.co_per_group(), spec.ci_per_group(),
                                     spec.kh, spec.kw};
    if (weights.shape != expect) throw SpecError("canonical weights shape mismatch for GOIHW");
    for (int kh = 0; kh < spec.kh; ++kh)
      for (int kw = 0; kw < spec.kw; ++kw)
        for (int g = 0; g < spec.groups; ++g)
          for (int ci = 0; ci < cipg; ++ci)
            for (int co = 0; co < copg; ++co)
              out.push_back(weights.at({g, co, ci, kh, kw}));
  } else if (layout == "HWM" && spec.channelwise()) {
    out = canonical_channelwise(spec, weights);
  } else {
    throw SpecError("layout " + layout + " is not valid for this operator");
  }

  if (static_cast<int>(out.size()) != spec.kh * spec.kw * spec.groups * cipg * copg) {
    throw SpecError("canonical weight element count mismatch");
  }
  return out;
}

PackedPlan pack_weights(const OpSpec& spec, const WeightTensor& weights, const std::string& layout,
                        const Plan* plan) {
  PackedPlan packed;
  packed.spec = spec;
  packed.plan = plan ? *plan : make_plan(spec);
  int64_t offset = 0;  // running weight bytes, for the SRAM bound below

  if (is_gemm(spec)) {
    const GemmSpec& g = as_gemm(spec);
    if (layout != "KN" || weights.shape.size() != 2 || weights.shape[0] != g.k ||
        weights.shape[1] != g.n) {
      throw SpecError("GEMM weights must be an int8 KxN array with layout KN");
    }
    const Task& t = packed.plan.tasks.at(0);
    PackedTask pt;
    pt.task = t;
    pt.weights.assign(static_cast<size_t>(t.n_tiles()) * t.k_groups() * 8, 0);
    for (int nt = 0; nt < t.n_tiles(); ++nt) {
      for (int kg = 0; kg < t.k_groups(); ++kg) {
        for (int within = 0; within < 8; ++within) {
          const int o = within / t.r;
          const int rr = within % t.r;
          const int co = nt * t.o() + o;
          const int reduction = kg * t.r + rr;
          if (co < g.n && reduction < g.k) {
            const size_t idx = (static_cast<size_t>(nt) * t.k_groups() + kg) * 8 + within;
            pt.weights[idx] = weights.at({reduction, co});
          }
        }
      }
    }
    offset = static_cast<int64_t>(pt.weights.size());
    packed.tasks.push_back(std::move(pt));
  } else {
    const ConvSpec& c = as_conv(spec);
    const std::vector<int8_t> canon = canonical_conv_weights(c, weights, layout);
    const int cipg = c.ci_per_group();
    for (const Task& t : packed.plan.tasks) {
      PackedTask pt;
      pt.task = t;
      pt.weights.assign(static_cast<size_t>(t.n_tiles()) * t.k_groups() * 8, 0);
      for (int nt = 0; nt < t.n_tiles(); ++nt) {
        const int valid_outputs = std::min(t.o(), t.n - nt * t.o());
        for (int kg = 0; kg < t.k_groups(); ++kg) {
          for (int within = 0; within < 8; ++within) {
            const int o = within / t.r;
            const int rr = within % t.r;
            const int reduction = kg * t.r + rr;
            if (o >= valid_outputs || reduction >= t.k) continue;
            const int tap_index = (t.mode == "dense") ? reduction / t.ci_count : reduction;
            const int kh = t.taps.at(static_cast<size_t>(tap_index)).first;
            const int kw = t.taps.at(static_cast<size_t>(tap_index)).second;
            int group, co_in_group, ci;
            if (t.mode == "channelwise") {
              const int co = nt * t.o() + o;
              group = co / c.co_per_group();
              co_in_group = co % c.co_per_group();
              ci = 0;
            } else {
              group = t.group;
              co_in_group = nt * t.o() + o;
              ci = reduction % t.ci_count;
            }
            const size_t canon_idx =
                ((((static_cast<size_t>(kh) * c.kw + kw) * c.groups + group) * cipg + ci) *
                     c.co_per_group() +
                 co_in_group);
            const size_t idx = (static_cast<size_t>(nt) * t.k_groups() + kg) * 8 + within;
            pt.weights[idx] = canon[canon_idx];
          }
        }
      }
      offset += static_cast<int64_t>(pt.weights.size());
      packed.tasks.push_back(std::move(pt));
    }
  }

  if (offset > kSramBytes) {
    throw SpecError("packed weights need " + std::to_string(offset) + " bytes; limit is " +
                    std::to_string(kSramBytes));
  }
  return packed;
}

}  // namespace convacc
