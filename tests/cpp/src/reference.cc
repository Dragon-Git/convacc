#include "convacc/reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace convacc {
namespace {

// Index helper for the semantic activation tensor (n, hi, wi, ci).
inline size_t act_idx(const ConvSpec& s, int batch, int ih, int iw, int ci) {
  return ((static_cast<size_t>(batch) * s.hi + ih) * s.wi + iw) * s.ci + ci;
}

// Canonical weights are (kh, kw, groups, ci_per_group, co_per_group).
inline size_t w_idx(const ConvSpec& s, int kh, int kw, int group, int ci, int co) {
  return ((((static_cast<size_t>(kh) * s.kw + kw) * s.groups + group) * s.ci_per_group() + ci) *
              s.co_per_group() +
          co);
}

std::vector<int32_t> conv_reference(const ConvSpec& spec, const std::vector<int8_t>& acts,
                                    const std::vector<int8_t>& canonical) {
  std::vector<int32_t> output(static_cast<size_t>(spec.n) * spec.ho * spec.wo * spec.co, 0);

  if (spec.deconv) {
    for (int batch = 0; batch < spec.n; ++batch)
      for (int ih = 0; ih < spec.hi; ++ih)
        for (int iw = 0; iw < spec.wi; ++iw)
          for (int kh = 0; kh < spec.kh; ++kh)
            for (int kw = 0; kw < spec.kw; ++kw) {
              const int oh = ih * spec.sh + kh - spec.ph;
              const int ow = iw * spec.sw + kw - spec.pw;
              if (oh < 0 || oh >= spec.ho || ow < 0 || ow >= spec.wo) continue;
              for (int ci = 0; ci < spec.ci; ++ci) {
                const int32_t x = acts[act_idx(spec, batch, ih, iw, ci)];
                for (int co = 0; co < spec.co; ++co) {
                  output[(static_cast<size_t>(batch) * spec.ho + oh) * spec.wo * spec.co +
                         static_cast<size_t>(ow) * spec.co + co] +=
                      x * canonical[w_idx(spec, kh, kw, 0, ci, co)];
                }
              }
            }
    return output;
  }

  for (int batch = 0; batch < spec.n; ++batch)
    for (int oh = 0; oh < spec.ho; ++oh)
      for (int ow = 0; ow < spec.wo; ++ow)
        for (int kh = 0; kh < spec.kh; ++kh)
          for (int kw = 0; kw < spec.kw; ++kw) {
            const int ih = oh * spec.sh + kh * spec.dh - spec.ph;
            const int iw = ow * spec.sw + kw * spec.dw - spec.pw;
            if (ih < 0 || ih >= spec.hi || iw < 0 || iw >= spec.wi) continue;
            for (int group = 0; group < spec.groups; ++group) {
              const int ci0 = group * spec.ci_per_group();
              for (int ci = 0; ci < spec.ci_per_group(); ++ci) {
                const int32_t x = acts[act_idx(spec, batch, ih, iw, ci0 + ci)];
                for (int co = 0; co < spec.co_per_group(); ++co) {
                  output[(static_cast<size_t>(batch) * spec.ho + oh) * spec.wo * spec.co +
                         static_cast<size_t>(ow) * spec.co + group * spec.co_per_group() + co] +=
                      x * canonical[w_idx(spec, kh, kw, group, ci, co)];
                }
              }
            }
          }
  return output;
}

// Activations for one reduction lane, one value per position. A dense task
// walks `(tap, channel)` from the reduction index; a channelwise task indexes
// the tap directly and takes the channel from the output lane.
std::vector<int32_t> tap_activations(const ConvSpec& spec, const Task& task,
                                     const std::pair<int, int>& tap, int ci,
                                     const std::vector<int8_t>& acts) {
  std::vector<int32_t> result(static_cast<size_t>(task.m), 0);
  for (int index = 0; index < task.m; ++index) {
    const OutCoord c = output_coords(task, index);
    int ih = 0;
    int iw = 0;
    if (input_coord(spec, c, tap, &ih, &iw)) {
      result[static_cast<size_t>(index)] = acts[act_idx(spec, c.batch, ih, iw, ci)];
    }
  }
  return result;
}

}  // namespace

std::vector<int8_t> requantize(const std::vector<int32_t>& accumulators, const QuantParams& quant,
                               int channels) {
  if (channels <= 0) throw SpecError("requantize needs a positive channel count");
  for (int c = 0; c < channels; ++c) {
    if (quant.scale[static_cast<size_t>(c)] <= 0) {
      throw SpecError("scale must be a positive int32");
    }
    const int shift = quant.shift[static_cast<size_t>(c)];
    if (shift < 0 || shift > 31) throw SpecError("shift must be in 0..31");
  }
  std::vector<int8_t> out(accumulators.size());
  for (size_t i = 0; i < accumulators.size(); ++i) {
    const size_t channel = i % static_cast<size_t>(channels);
    const int64_t q = (static_cast<int64_t>(accumulators[i]) + quant.bias[channel]) *
                      static_cast<int64_t>(quant.scale[channel]);
    const int64_t magnitude = q < 0 ? -q : q;
    const int shift = quant.shift[channel];
    const int64_t rounded =
        shift == 0 ? magnitude : (magnitude + (int64_t{1} << (shift - 1))) >> shift;
    const int64_t signed_value = (q >= 0 ? rounded : -rounded) + quant.zero_point[channel];
    out[i] = static_cast<int8_t>(std::clamp<int64_t>(signed_value, -128, 127));
  }
  return out;
}

std::vector<int32_t> execute_packed(const PackedPlan& packed,
                                    const std::vector<int8_t>& activations) {
  const OpSpec& spec = packed.spec;

  if (is_gemm(spec)) {
    const GemmSpec& g = as_gemm(spec);
    const Task& task = packed.tasks.at(0).task;
    const std::vector<int8_t>& w = packed.tasks.at(0).weights;
    std::vector<int32_t> result(static_cast<size_t>(g.m) * g.n, 0);
    for (int nt = 0; nt < task.n_tiles(); ++nt) {
      const int width = std::min(task.o(), g.n - nt * task.o());
      std::vector<int32_t> acc(static_cast<size_t>(g.m) * task.o(), 0);
      for (int kg = 0; kg < task.k_groups(); ++kg) {
        for (int within = 0; within < 8; ++within) {
          const int o = within / task.r;
          const int rr = within % task.r;
          const int reduction = kg * task.r + rr;
          if (o < width && reduction < task.k) {
            const int8_t weight = w[(static_cast<size_t>(nt) * task.k_groups() + kg) * 8 + within];
            for (int mi = 0; mi < g.m; ++mi) {
              acc[static_cast<size_t>(mi) * task.o() + o] +=
                  static_cast<int32_t>(activations[static_cast<size_t>(mi) * g.k + reduction]) *
                  weight;
            }
          }
        }
      }
      for (int mi = 0; mi < g.m; ++mi)
        for (int lane = 0; lane < width; ++lane)
          result[static_cast<size_t>(mi) * g.n + nt * task.o() + lane] =
              acc[static_cast<size_t>(mi) * task.o() + lane];
    }
    return result;
  }

  const ConvSpec& conv = as_conv(spec);
  std::vector<int32_t> output(static_cast<size_t>(conv.n) * conv.ho * conv.wo * conv.co, 0);
  for (const auto& packed_task : packed.tasks) {
    const Task& task = packed_task.task;
    for (int nt = 0; nt < task.n_tiles(); ++nt) {
      const int width = std::min(task.o(), task.n - nt * task.o());
      std::vector<int32_t> acc(static_cast<size_t>(task.m) * task.o(), 0);
      for (int kg = 0; kg < task.k_groups(); ++kg) {
        for (int within = 0; within < 8; ++within) {
          const int o = within / task.r;
          const int rr = within % task.r;
          const int reduction = kg * task.r + rr;
          if (o >= width || reduction >= task.k) continue;
          const int8_t weight =
              packed_task.weights[(static_cast<size_t>(nt) * task.k_groups() + kg) * 8 + within];
          if (weight == 0) continue;
          const std::vector<int32_t> acts =
              (task.mode == "channelwise")
                  ? tap_activations(conv, task, task.taps[static_cast<size_t>(reduction)],
                                    (nt * task.o() + o) / conv.co_per_group(), activations)
                  : tap_activations(conv, task,
                                    task.taps[static_cast<size_t>(reduction / task.ci_count)],
                                    task.ci_base + reduction % task.ci_count, activations);
          for (int index = 0; index < task.m; ++index) {
            acc[static_cast<size_t>(index) * task.o() + o] += acts[static_cast<size_t>(index)] * weight;
          }
        }
      }
      for (int index = 0; index < task.m; ++index) {
        const OutCoord c = output_coords(task, index);
        const int co0 = task.co_base + nt * task.o();
        for (int lane = 0; lane < width; ++lane) {
          output[(static_cast<size_t>(c.batch) * conv.ho + c.oh) * conv.wo * conv.co +
                 static_cast<size_t>(c.ow) * conv.co + co0 + lane] =
              acc[static_cast<size_t>(index) * task.o() + lane];
        }
      }
    }
  }
  return output;
}

std::vector<int32_t> reference_for_packed(const PackedPlan& packed,
                                          const std::vector<int8_t>& activations,
                                          const WeightTensor& weights, const std::string& layout) {
  if (is_gemm(packed.spec)) {
    const GemmSpec& g = as_gemm(packed.spec);
    std::vector<int32_t> result(static_cast<size_t>(g.m) * g.n, 0);
    for (int mi = 0; mi < g.m; ++mi)
      for (int ni = 0; ni < g.n; ++ni) {
        int32_t acc = 0;
        for (int ki = 0; ki < g.k; ++ki) {
          acc += static_cast<int32_t>(activations[static_cast<size_t>(mi) * g.k + ki]) *
                 weights.data[static_cast<size_t>(ki) * g.n + ni];
        }
        result[static_cast<size_t>(mi) * g.n + ni] = acc;
      }
    return result;
  }
  const ConvSpec& conv = as_conv(packed.spec);
  const std::vector<int8_t> canonical = canonical_conv_weights(conv, weights, layout);
  return conv_reference(conv, activations, canonical);
}

std::vector<int8_t> layout_output(const PackedPlan& packed, const std::vector<int32_t>& result) {
  const OutputGeometry geo = output_geometry(packed);
  const int o = 8 / packed.tasks.at(0).task.r;
  const OpSpec& spec = packed.spec;

  int channels, per_group;
  if (is_gemm(spec)) {
    channels = as_gemm(spec).n;
    per_group = channels;
  } else if (as_conv(spec).channelwise()) {
    channels = as_conv(spec).co;
    per_group = channels;
  } else {
    channels = as_conv(spec).co;
    per_group = as_conv(spec).co_per_group();
  }

  const int row = geo.groups * geo.n_tiles * 8;
  std::vector<int8_t> image(static_cast<size_t>(geo.positions) * row, 0);
  for (int channel = 0; channel < channels; ++channel) {
    const int group = channel / per_group;
    const int local = channel % per_group;
    const int tile = local / o;
    const int lane = local % o;
    const int column = (group * geo.n_tiles + tile) * 8 + lane;
    for (int pos = 0; pos < geo.positions; ++pos) {
      image[static_cast<size_t>(pos) * row + column] =
          static_cast<int8_t>(result[static_cast<size_t>(pos) * channels + channel]);
    }
  }
  return image;
}

BuiltCase build_case(const OpSpec& spec, const WeightTensor& weights, const std::string& layout,
                     const std::vector<int8_t>& activations, const QuantParams& quant) {
  BuiltCase built;
  built.packed = pack_weights(spec, weights, layout);

  std::vector<uint8_t> act_bytes(activations.size());
  if (!activations.empty()) {
    std::memcpy(act_bytes.data(), activations.data(), activations.size());
  }
  built.image = build_image(built.packed, quant, act_bytes);

  const std::vector<int32_t> accumulators =
      reference_for_packed(built.packed, activations, weights, layout);
  const std::vector<int8_t> requanted = requantize(accumulators, quant, param_width(spec));
  // layout_output narrows back to int8; feeding it the already-requantised
  // values keeps the byte image identical to the Python model, which layouts
  // the int8 array rather than the int32 accumulators.
  built.expected = layout_output(built.packed,
                                 std::vector<int32_t>(requanted.begin(), requanted.end()));
  return built;
}

}  // namespace convacc
