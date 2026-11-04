#include "model/case_codec.h"

#include "convacc/rng.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace convacc {
namespace {

int gcd_int(int a, int b) {
  while (b) {
    const int t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// All divisors of `value`, ascending. Used to pick `groups` from a legal set so
// a mutated byte always maps to a schedulable grouping instead of a rejection.
int divisors(int value, int* out, int cap) {
  int count = 0;
  for (int d = 1; d <= value && count < cap; ++d) {
    if (value % d == 0) out[count++] = d;
  }
  return count;
}

constexpr long long kMacBudget = 120000;  // keeps one exec near 0.2 ms
constexpr int kMaxSpatial = 12;

void fill(int8_t* dst, size_t count, Rng& rng, bool extremes) {
  for (size_t i = 0; i < count; ++i) {
    if (extremes) {
      // Bias half the elements to the ends of the int8 range: saturation and
      // accumulator overflow only show up there.
      const uint64_t r = rng.next();
      if ((r & 3) == 0) {
        dst[i] = static_cast<int8_t>(r & 1 ? 127 : -128);
      } else {
        dst[i] = static_cast<int8_t>(static_cast<int>(r >> 8) % 256 - 128);
      }
    } else {
      dst[i] = static_cast<int8_t>(static_cast<int>(rng.next() % 33) - 16);
    }
  }
}

int div_floor(int numerator, int denominator) {
  // Both operands are non-negative where this is used.
  return numerator / denominator;
}

}  // namespace

bool decode_case(const uint8_t* data, size_t size, FuzzCase* out) {
  uint8_t b[kCaseHeaderBytes] = {0};
  const size_t copied = std::min(size, kCaseHeaderBytes);
  if (copied) std::memcpy(b, data, copied);

  uint64_t seed = 0x243F6A8885A308D3ull;
  for (size_t i = kCaseHeaderBytes; i < size; ++i) {
    seed = seed * 131 + data[i];
  }
  seed ^= (static_cast<uint64_t>(b[28]) << 24) | (static_cast<uint64_t>(b[29]) << 16) |
          (static_cast<uint64_t>(b[30]) << 8) | b[31];
  Rng rng{seed};

  const int kind = b[0] & 7;
  const bool gemm = kind >= 6;
  const bool deconv = kind == 4 || kind == 5;
  const bool channelwise = kind == 3;
  const bool extremes = (b[15] & 4) != 0;

  if (gemm) {
    const int m = 1 + b[2] % 48;
    const int k = 1 + b[3] % 48;
    const int n = 1 + b[4] % 48;
    if (1LL * m * k > 65536 || 1LL * k * n > 65536) return false;
    if (1LL * m * k * n > kMacBudget) return false;

    GemmSpec spec;
    spec.m = m;
    spec.k = k;
    spec.n = n;
    out->spec = spec;
    out->layout = "KN";

    out->weights.shape = {k, n};
    out->weights.data.resize(static_cast<size_t>(k) * n);
    fill(out->weights.data.data(), out->weights.data.size(), rng, extremes);
    out->activations.resize(static_cast<size_t>(m) * k);
    fill(out->activations.data(), out->activations.size(), rng, extremes);
  } else {
    ConvSpec spec;
    spec.n = 1 + b[1] % 3;
    spec.hi = 1 + b[2] % kMaxSpatial;
    spec.wi = 1 + b[3] % kMaxSpatial;
    spec.ci = 1 + b[4] % 16;
    spec.kh = 2 * (b[6] % 4) + 1;
    spec.kw = 2 * (b[7] % 4) + 1;
    spec.sh = 1 + b[8] % 3;
    spec.sw = 1 + b[9] % 3;
    spec.dh = 1 + b[10] % 2;
    spec.dw = 1 + b[11] % 2;
    spec.ph = b[12] % 4;
    spec.pw = b[13] % 4;
    spec.deconv = deconv;

    if (channelwise) {
      // groups == ci is the planner's depthwise fast path. Keeping co a
      // multiple of ci is what makes the HWM layout and the channelwise task
      // legal.
      spec.co = spec.ci * (1 + b[5] % 3);
      if (spec.co > 32) spec.co = spec.ci;
      spec.groups = spec.ci;
    } else {
      spec.co = 1 + b[5] % 16;
      int choices[32];
      const int count = divisors(gcd_int(spec.ci, spec.co), choices, 32);
      spec.groups = choices[b[14] % count];
    }

    if (deconv) {
      // deconv2d is groups=1, dilation=1 only (spec.validate).
      spec.groups = 1;
      spec.dh = 1;
      spec.dw = 1;
      const int ho = (spec.hi - 1) * spec.sh - 2 * spec.ph + (spec.kh - 1) + 1;
      const int wo = (spec.wi - 1) * spec.sw - 2 * spec.pw + (spec.kw - 1) + 1;
      if (ho <= 0 || wo <= 0 || ho > 64 || wo > 64) return false;
      spec.ho = ho;
      spec.wo = wo;
    } else {
      const int span_h = spec.hi + 2 * spec.ph - spec.dh * (spec.kh - 1) - 1;
      const int span_w = spec.wi + 2 * spec.pw - spec.dw * (spec.kw - 1) - 1;
      if (span_h < 0 || span_w < 0) return false;
      const int ho = div_floor(span_h, spec.sh) + 1;
      const int wo = div_floor(span_w, spec.sw) + 1;
      if (ho <= 0 || wo <= 0 || ho > 64 || wo > 64) return false;
      spec.ho = ho;
      spec.wo = wo;
    }

    const long long acts = 1LL * spec.n * spec.hi * spec.wi * spec.ci;
    const long long outs = 1LL * spec.n * spec.ho * spec.wo * spec.co;
    if (acts > kSramBytes || outs > kSramBytes) return false;
    const long long macs = outs * spec.kh * spec.kw * spec.ci / spec.groups;
    if (macs > kMacBudget) return false;

    out->spec = spec;
    const bool groups_one = spec.groups == 1;
    const int layout_pick = b[15] & 7;
    if (spec.channelwise() && (layout_pick & 1)) {
      out->layout = "HWM";
    } else if (groups_one && layout_pick == 2) {
      out->layout = "HWIO";
    } else if (groups_one && layout_pick == 4) {
      out->layout = "OIHW";
    } else if (layout_pick == 6) {
      out->layout = "GOIHW";
    } else {
      out->layout = "HWGIO";
    }

    const int cipg = spec.ci_per_group();
    const int copg = spec.co_per_group();
    if (out->layout == "HWM") {
      out->weights.shape = {spec.kh, spec.kw, spec.co};
    } else if (out->layout == "HWIO") {
      out->weights.shape = {spec.kh, spec.kw, spec.ci, spec.co};
    } else if (out->layout == "OIHW") {
      out->weights.shape = {spec.co, spec.kh, spec.kw, spec.ci};
    } else if (out->layout == "GOIHW") {
      out->weights.shape = {spec.groups, copg, cipg, spec.kh, spec.kw};
    } else {
      out->weights.shape = {spec.kh, spec.kw, spec.groups, cipg, copg};
    }
    out->weights.data.resize(static_cast<size_t>(out->weights.size()));
    fill(out->weights.data.data(), out->weights.data.size(), rng, extremes);
    out->activations.resize(static_cast<size_t>(acts));
    fill(out->activations.data(), out->activations.size(), rng, extremes);
  }

  const int width = param_width(out->spec);
  out->quant.bias.resize(static_cast<size_t>(width));
  out->quant.scale.resize(static_cast<size_t>(width));
  out->quant.shift.resize(static_cast<size_t>(width));
  out->quant.zero_point.resize(static_cast<size_t>(width));
  const bool huge_scale = (b[15] & 8) != 0;
  const bool tiny_shift = (b[15] & 16) != 0;
  for (int i = 0; i < width; ++i) {
    const uint32_t r = static_cast<uint32_t>(rng.next());
    out->quant.bias[static_cast<size_t>(i)] =
        static_cast<int32_t>(r % 4096) - 2048;
    out->quant.scale[static_cast<size_t>(i)] =
        huge_scale ? static_cast<int32_t>(1 + (r >> 8) % 0x7FFFFFFF)
                   : static_cast<int32_t>(1 + (r >> 8) % 4096);
    out->quant.shift[static_cast<size_t>(i)] =
        static_cast<uint8_t>(tiny_shift ? (r >> 20) % 3 : (r >> 20) % 32);
    out->quant.zero_point[static_cast<size_t>(i)] =
        static_cast<int8_t>(static_cast<int>((r >> 4) % 256) - 128);
  }
  return true;
}

std::string describe_case(const FuzzCase& c) {
  char line[256];
  if (is_gemm(c.spec)) {
    const GemmSpec& g = as_gemm(c.spec);
    std::snprintf(line, sizeof(line), "gemm m=%d k=%d n=%d layout=%s", g.m, g.k, g.n,
                  c.layout.c_str());
  } else {
    const ConvSpec& s = as_conv(c.spec);
    std::snprintf(line, sizeof(line),
                  "%s n=%d %dx%dx%d -> %dx%dx%d k=%dx%d s=%d,%d d=%d,%d p=%d,%d g=%d layout=%s",
                  s.deconv ? "deconv" : (s.channelwise() ? "depthwise" : "conv"), s.n, s.hi, s.wi,
                  s.ci, s.ho, s.wo, s.co, s.kh, s.kw, s.sh, s.sw, s.dh, s.dw, s.ph, s.pw, s.groups,
                  c.layout.c_str());
  }
  return line;
}

}  // namespace convacc
