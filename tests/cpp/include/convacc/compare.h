// Comparing a DUT output against the model, and reporting what differed.
//
// Four drivers used to carry a copy of the comparison, in four dialects: one
// reported the word/lane of each differing byte, one reported the byte offset,
// two reported only a count. The slow path here (walking the buffer to locate
// the first few differences) runs outside every timing window -- the golden
// regression stops its clock before calling it -- so it only costs anything
// when a case has already failed. The same goes for the failure summary below,
// which the two drivers also each spelled out in their own words.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace convacc {

struct Mismatch {
  size_t bad = 0;      // differing bytes; a byte missing from `got` also counts
  size_t total = 0;    // bytes expected
  std::string detail;  // " @12(w1,l4):00->ff", empty when nothing differs
};

// Compares `got` against `want`. `got` may be shorter than `want` -- the last
// output word is only partially used -- in which case the missing bytes are
// reported as differing rather than read past the end of the buffer.
inline Mismatch compare_bytes(const uint8_t* got, size_t got_size, const uint8_t* want,
                              size_t want_size, size_t max_detail = 8) {
  Mismatch result;
  result.total = want_size;
  if (got_size >= want_size && std::memcmp(got, want, want_size) == 0) return result;

  char item[64];
  for (size_t i = 0; i < want_size; ++i) {
    if (i < got_size && got[i] == want[i]) continue;
    ++result.bad;
    if (result.bad > max_detail) continue;
    const unsigned g = i < got_size ? got[i] : 0u;
    std::snprintf(item, sizeof(item), " @%zu(w%zu,l%zu):%02x->%02x", i, i / 8, i % 8,
                  static_cast<unsigned>(want[i]), g);
    result.detail += item;
  }
  return result;
}

// For the callers holding the model's int8_t output.
inline Mismatch compare_bytes(const std::vector<uint8_t>& got, const std::vector<int8_t>& want,
                              size_t max_detail = 8) {
  return compare_bytes(got.data(), got.size(), reinterpret_cast<const uint8_t*>(want.data()),
                       want.size(), max_detail);
}

inline std::string describe_mismatch(const Mismatch& mismatch) {
  return std::to_string(mismatch.bad) + "/" + std::to_string(mismatch.total) + " bytes differ" +
         mismatch.detail;
}

// One failed case, as the drivers print it.
struct Failure {
  std::string name;
  std::string reason;
  std::string detail;  // a second, indented line; empty when there is none
};

inline void report_failures(const std::vector<Failure>& failures, size_t limit = 20) {
  if (failures.empty()) return;
  std::printf("\nFAILED %zu case(s):\n", failures.size());
  for (size_t i = 0; i < failures.size() && i < limit; ++i) {
    std::printf("  %s: %s\n", failures[i].name.c_str(), failures[i].reason.c_str());
    if (!failures[i].detail.empty()) std::printf("    %s\n", failures[i].detail.c_str());
  }
  if (failures.size() > limit) std::printf("  ... %zu more\n", failures.size() - limit);
}

}  // namespace convacc
