// Byte-string -> operator-case decoding for the coverage-guided fuzzer.
//
// libFuzzer mutates opaque bytes, so the interesting part of this file is the
// mapping from bytes to a *schedulable* operator: every field is taken from a
// fixed offset and clamped/biassed so that a random byte string has a decent
// chance of producing a case the planner accepts. Mutations therefore change
// one dimension at a time, and libFuzzer's cmp tracing can still discover the
// divisibility constraints (groups | ci, co) because those are checked on the
// decoded values rather than hidden behind a rejection.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "convacc/reference.h"

namespace convacc {

struct FuzzCase {
  OpSpec spec;
  std::string layout;
  QuantParams quant;
  WeightTensor weights;
  std::vector<int8_t> activations;
};

// The decoder reads at most this many leading bytes; anything beyond feeds the
// data PRNG. Keeping the layout fixed means a minimized input stays readable.
inline constexpr size_t kCaseHeaderBytes = 32;

// Returns false when the bytes do not describe a case worth simulating
// (out-of-range dimensions, empty output, or above the MAC budget). Rejections
// are cheap on purpose: the fuzzer should spend its time in the model.
bool decode_case(const uint8_t* data, size_t size, FuzzCase* out);

// Human-readable one-line summary, used in replay logs and crash reports.
std::string describe_case(const FuzzCase& c);

}  // namespace convacc
