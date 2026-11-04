// Strict command-line integer parsing, shared by the native drivers.
//
// `std::atoi` is what these drivers used to call, and it reports every mistake
// as a number: "--jobs x" becomes 0 (which means "auto"), "--jobs 8x" becomes 8,
// "--cycle-budget abc" becomes 0 (which means "no budget"). A typo silently
// becomes a different test run instead of an error message.
#pragma once

#include <charconv>
#include <cstdio>
#include <cstring>
#include <limits>
#include <system_error>

namespace convacc {

// True only when the whole string is a base-10 integer in range.
inline bool parse_ll(const char* text, long long* out) {
  if (text == nullptr || *text == '\0') return false;
  const char* end = text + std::strlen(text);
  const auto result = std::from_chars(text, end, *out);
  return result.ec == std::errc() && result.ptr == end;
}

// "error: --jobs needs a decimal integer", then the caller prints usage and
// returns 2.
inline void report_bad_number(const char* flag) {
  std::fprintf(stderr, "error: %s needs a decimal integer\n", flag);
}

inline bool parse_int(const char* text, int* out) {
  long long wide = 0;
  if (!parse_ll(text, &wide)) return false;
  if (wide < std::numeric_limits<int>::min() || wide > std::numeric_limits<int>::max()) return false;
  *out = static_cast<int>(wide);
  return true;
}

}  // namespace convacc
