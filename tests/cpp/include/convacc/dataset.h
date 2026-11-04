// The "CVCC" dataset file: many cases in one bundle.
//
// Four tools touch this format -- tools/gen_cpp_cases.py writes it, `convacc_bench
// --dump` writes it, tools/verify_generated_cases.py re-derives it in Python, and
// the GoogleTest suite reads it back -- and the writer and the reader
// live in one translation unit so the field order exists once. That is how `fields[16]` survived on the reading side after the
// writing side had been given names.
//
// Everything here is the format and nothing else: no simulation, no packing.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "convacc/image.h"
#include "convacc/packer.h"
#include "convacc/spec.h"

namespace convacc {

// The 19 ints that follow `kind`. The order *is* the format: it has to match
// FIELD_NAMES in tools/verify_generated_cases.py, which unpacks them as "<19i".
enum DatasetField : int {
  kBatch, kHi, kWi, kCi, kHo, kWo, kCo, kKh, kKw, kSh, kSw, kDh, kDw, kPh, kPw, kGroups,
  kGemmM, kGemmK, kGemmN, kDatasetFieldCount
};
static_assert(kDatasetFieldCount == 19, "the CVCC header is 19 ints wide");

// Also the `kind` word in the file.
enum class DatasetKind : uint32_t { kConv = 0, kDeconv = 1, kGemm = 2 };

struct DatasetCase {
  std::string name;
  OpSpec spec;
  std::string layout;
  QuantParams quant;
  WeightTensor weights;
  std::vector<int8_t> activations;
  std::vector<int8_t> expected;       // the model's output image
  std::vector<uint8_t> weight_image;  // build_image().weight, for the cross-check
};

inline DatasetKind dataset_kind(const OpSpec& spec) {
  if (is_gemm(spec)) return DatasetKind::kGemm;
  return as_conv(spec).deconv ? DatasetKind::kDeconv : DatasetKind::kConv;
}

inline std::array<int, kDatasetFieldCount> dataset_fields(const OpSpec& spec) {
  std::array<int, kDatasetFieldCount> f{};
  if (is_gemm(spec)) {
    const GemmSpec& g = as_gemm(spec);
    f[kGemmM] = g.m;
    f[kGemmK] = g.k;
    f[kGemmN] = g.n;
    return f;
  }
  const ConvSpec& c = as_conv(spec);
  f[kBatch] = c.n;
  f[kHi] = c.hi;
  f[kWi] = c.wi;
  f[kCi] = c.ci;
  f[kHo] = c.ho;
  f[kWo] = c.wo;
  f[kCo] = c.co;
  f[kKh] = c.kh;
  f[kKw] = c.kw;
  f[kSh] = c.sh;
  f[kSw] = c.sw;
  f[kDh] = c.dh;
  f[kDw] = c.dw;
  f[kPh] = c.ph;
  f[kPw] = c.pw;
  f[kGroups] = c.groups;
  return f;
}

inline OpSpec dataset_spec(DatasetKind kind, const std::array<int, kDatasetFieldCount>& f) {
  if (kind == DatasetKind::kGemm) {
    GemmSpec g;
    g.m = f[kGemmM];
    g.k = f[kGemmK];
    g.n = f[kGemmN];
    return g;
  }
  ConvSpec c;
  c.n = f[kBatch];
  c.hi = f[kHi];
  c.wi = f[kWi];
  c.ci = f[kCi];
  c.ho = f[kHo];
  c.wo = f[kWo];
  c.co = f[kCo];
  c.kh = f[kKh];
  c.kw = f[kKw];
  c.sh = f[kSh];
  c.sw = f[kSw];
  c.dh = f[kDh];
  c.dw = f[kDw];
  c.ph = f[kPh];
  c.pw = f[kPw];
  c.groups = f[kGroups];
  c.deconv = kind == DatasetKind::kDeconv;
  return c;
}

namespace detail {

class DatasetReader {
 public:
  explicit DatasetReader(const std::string& path) : in_(path, std::ios::binary) {
    if (!in_) throw std::runtime_error("cannot open dataset: " + path);
  }

  template <typename T>
  T read() {
    T value{};
    in_.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in_) throw std::runtime_error("truncated dataset");
    return value;
  }

  std::vector<uint8_t> bytes(size_t count) {
    std::vector<uint8_t> buf(count);
    if (count) in_.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(count));
    if (!in_) throw std::runtime_error("truncated dataset");
    return buf;
  }

  std::string string() {
    const uint32_t length = read<uint32_t>();
    const std::vector<uint8_t> raw = bytes(length);
    return std::string(raw.begin(), raw.end());
  }

 private:
  std::ifstream in_;
};

class DatasetWriter {
 public:
  explicit DatasetWriter(const std::string& path) : out_(path, std::ios::binary) {
    if (!out_) throw std::runtime_error("cannot open dataset for writing: " + path);
  }

  void raw(const void* data, size_t count) {
    if (count) out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(count));
  }
  void u32(uint32_t value) { raw(&value, sizeof(value)); }
  void i32(int32_t value) { raw(&value, sizeof(value)); }
  void string(const std::string& value) {
    u32(static_cast<uint32_t>(value.size()));
    raw(value.data(), value.size());
  }
  // A length-prefixed blob: four bytes of count, then the payload.
  void bytes(const void* data, size_t count) {
    u32(static_cast<uint32_t>(count));
    raw(data, count);
  }

 private:
  std::ofstream out_;
};

}  // namespace detail

inline std::vector<DatasetCase> read_dataset(const std::string& path) {
  detail::DatasetReader reader(path);
  char magic[4];
  for (char& c : magic) c = static_cast<char>(reader.read<uint8_t>());
  if (std::memcmp(magic, "CVCC", 4) != 0) throw std::runtime_error("bad dataset magic");
  const uint32_t version = reader.read<uint32_t>();
  if (version != 1) throw std::runtime_error("unsupported dataset version");
  const uint32_t count = reader.read<uint32_t>();

  std::vector<DatasetCase> cases;
  cases.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    DatasetCase item;
    item.name = reader.string();
    const auto kind = static_cast<DatasetKind>(reader.read<uint32_t>());
    std::array<int, kDatasetFieldCount> fields{};
    for (int& value : fields) value = reader.read<int32_t>();
    item.spec = dataset_spec(kind, fields);
    item.layout = reader.string();

    const uint32_t width = reader.read<uint32_t>();
    item.quant.bias.resize(width);
    item.quant.scale.resize(width);
    for (uint32_t c = 0; c < width; ++c) item.quant.bias[c] = reader.read<int32_t>();
    for (uint32_t c = 0; c < width; ++c) item.quant.scale[c] = reader.read<int32_t>();
    {
      const std::vector<uint8_t> shift = reader.bytes(width);
      item.quant.shift.assign(shift.begin(), shift.end());
      const std::vector<uint8_t> zp = reader.bytes(width);
      item.quant.zero_point.resize(zp.size());
      for (size_t k = 0; k < zp.size(); ++k) item.quant.zero_point[k] = static_cast<int8_t>(zp[k]);
    }

    const uint32_t rank = reader.read<uint32_t>();
    item.weights.shape.resize(rank);
    for (uint32_t d = 0; d < rank; ++d) {
      item.weights.shape[d] = static_cast<int>(reader.read<uint32_t>());
    }
    {
      const uint32_t weight_count = reader.read<uint32_t>();
      const std::vector<uint8_t> raw = reader.bytes(weight_count);
      item.weights.data.assign(raw.begin(), raw.end());
    }
    {
      const uint32_t act_count = reader.read<uint32_t>();
      const std::vector<uint8_t> raw = reader.bytes(act_count);
      item.activations.assign(raw.begin(), raw.end());
    }
    {
      const uint32_t expected_count = reader.read<uint32_t>();
      const std::vector<uint8_t> raw = reader.bytes(expected_count);
      item.expected.assign(raw.begin(), raw.end());
    }
    {
      const uint32_t image_count = reader.read<uint32_t>();
      item.weight_image = reader.bytes(image_count);
    }
    cases.push_back(std::move(item));
  }
  return cases;
}

inline void write_dataset(const std::string& path, const std::vector<DatasetCase>& cases) {
  detail::DatasetWriter out(path);
  out.raw("CVCC", 4);
  out.u32(1);  // version
  out.u32(static_cast<uint32_t>(cases.size()));
  for (const DatasetCase& c : cases) {
    out.string(c.name);
    out.u32(static_cast<uint32_t>(dataset_kind(c.spec)));
    for (int value : dataset_fields(c.spec)) out.i32(value);
    out.string(c.layout);

    const uint32_t width = static_cast<uint32_t>(c.quant.bias.size());
    out.u32(width);
    for (int32_t value : c.quant.bias) out.i32(value);
    for (int32_t value : c.quant.scale) out.i32(value);
    // shift/zero_point are written bare: their length is already in `width`.
    out.raw(c.quant.shift.data(), width);
    out.raw(c.quant.zero_point.data(), width);

    out.u32(static_cast<uint32_t>(c.weights.shape.size()));
    for (int d : c.weights.shape) out.u32(static_cast<uint32_t>(d));
    out.bytes(c.weights.data.data(), c.weights.data.size());
    out.bytes(c.activations.data(), c.activations.size());
    out.bytes(c.expected.data(), c.expected.size());
    out.bytes(c.weight_image.data(), c.weight_image.size());
  }
}

}  // namespace convacc
