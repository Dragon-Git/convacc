#include "convacc/spec.h"

#include <string>

namespace convacc {
namespace {

void require_positive(const char* name, int value) {
  if (value <= 0 || value > kMaxDim) {
    throw SpecError(std::string(name) + " must be in 1.." + std::to_string(kMaxDim) +
                    ", got " + std::to_string(value));
  }
}

void require_tensor(const char* name, long long elements) {
  if (elements <= 0 || elements > kSramBytes) {
    throw SpecError(std::string(name) + " needs " + std::to_string(elements) +
                    " bytes; limit is " + std::to_string(kSramBytes));
  }
}

}  // namespace

void ConvSpec::validate() const {
  const int dims[] = {n, hi, wi, ci, ho, wo, co, kh, kw};
  const char* names[] = {"n", "hi", "wi", "ci", "ho", "wo", "co", "kh", "kw"};
  for (int i = 0; i < 9; ++i) require_positive(names[i], dims[i]);

  if (kh % 2 != 1 || kw % 2 != 1 || kh > 15 || kw > 15) {
    throw SpecError("kernel dimensions must be odd and in 1..15");
  }
  if (sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0) {
    throw SpecError("stride and dilation must be positive");
  }
  if (ph < 0 || pw < 0) throw SpecError("padding must be non-negative");
  if (ci % groups != 0 || co % groups != 0) {
    throw SpecError("groups must divide both ci and co");
  }
  if (deconv && (groups != 1 || dh != 1 || dw != 1)) {
    throw SpecError("deconv2d supports groups=1 and dilation=1 only");
  }
  require_tensor("activation", 1LL * n * hi * wi * ci);
  require_tensor("output", 1LL * n * ho * wo * co);
}

void GemmSpec::validate() const {
  require_positive("m", m);
  require_positive("k", k);
  require_positive("n", n);
  require_tensor("A", 1LL * m * k);
  require_tensor("B", 1LL * k * n);
  require_tensor("C", 1LL * m * n);
}

}  // namespace convacc
