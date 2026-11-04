// Operator descriptions, mirrored 1:1 from convacc/spec.py.
//
// The Python model is the spec of record; this header is the C++ mirror used by
// the native test harness. Every validation rule (and the wording of its error)
// matches spec.py so that a case rejected by one side is rejected by the other.
#pragma once

#include <stdexcept>
#include <string>
#include <variant>

namespace convacc {

inline constexpr int kSramBytes = 128 * 1024;
inline constexpr int kMaxDim = 256;
inline constexpr int kWords = kSramBytes / 8;  // 16384 64-bit words
inline constexpr int kHeaderWords = 16;
inline constexpr int kParamWordsPerTile = 10;

class SpecError : public std::runtime_error {
 public:
  explicit SpecError(const std::string& what) : std::runtime_error(what) {}
};

struct ConvSpec {
  int n = 1;
  int hi = 1, wi = 1, ci = 1;
  int ho = 1, wo = 1, co = 1;
  int kh = 1, kw = 1;
  int sh = 1, sw = 1, dh = 1, dw = 1;
  int ph = 0, pw = 0;
  int groups = 1;
  bool deconv = false;

  int ci_per_group() const { return ci / groups; }
  int co_per_group() const { return co / groups; }
  // groups == ci means every input channel has its own group: the depthwise
  // fast path (`mode="channelwise"`) instead of one task per group.
  bool channelwise() const { return !deconv && groups == ci; }
  void validate() const;
};

struct GemmSpec {
  int m = 1, k = 1, n = 1;
  void validate() const;
};

using OpSpec = std::variant<ConvSpec, GemmSpec>;

inline bool is_gemm(const OpSpec& s) { return std::holds_alternative<GemmSpec>(s); }
inline bool is_deconv(const OpSpec& s) {
  const auto* c = std::get_if<ConvSpec>(&s);
  return c != nullptr && c->deconv;
}
inline bool is_channelwise(const OpSpec& s) {
  const auto* c = std::get_if<ConvSpec>(&s);
  return c != nullptr && c->channelwise();
}
inline const ConvSpec& as_conv(const OpSpec& s) { return std::get<ConvSpec>(s); }
inline const GemmSpec& as_gemm(const OpSpec& s) { return std::get<GemmSpec>(s); }

// Number of output channels the requant parameters are indexed by: CO for a
// convolution, N for a GEMM.
inline int param_width(const OpSpec& s) {
  return is_gemm(s) ? as_gemm(s).n : as_conv(s).co;
}

}  // namespace convacc
