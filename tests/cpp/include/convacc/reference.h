// The reference model: requantisation, the semantic result, the packed
// executor, the output layout, and the assembled case (spec + images +
// expected bytes). Mirrored from convacc/reference.py.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "convacc/image.h"
#include "convacc/packer.h"
#include "convacc/spec.h"

namespace convacc {

// Round-away-from-zero requantisation:
//   q = round(|(acc + bias) * scale| / 2**shift) * sign + zero_point
//
// `accumulators` is a flat (positions * channels) tensor and the quantisation
// parameters are per channel, so they are indexed modulo `channels` -- the same
// broadcast numpy applies along the last axis in convacc/reference.py.
std::vector<int8_t> requantize(const std::vector<int32_t>& accumulators,
                               const QuantParams& quant, int channels);

// Semantic result (n, ho, wo, co) or (m, n) for a GEMM, in int32.
std::vector<int32_t> reference_for_packed(const PackedPlan& packed,
                                          const std::vector<int8_t>& activations,
                                          const WeightTensor& weights, const std::string& layout);

// Bit-exact model of what the engine is supposed to compute, driven from the
// packed weights rather than from the semantic weights. A mismatch between this
// and reference_for_packed means the packer lost information.
std::vector<int32_t> execute_packed(const PackedPlan& packed,
                                    const std::vector<int8_t>& activations);

// Flatten a semantic result into the engine's 8-byte-aligned output image.
std::vector<int8_t> layout_output(const PackedPlan& packed, const std::vector<int32_t>& result);

// Everything one case needs on the DUT side: the packed plan, the three SRAM
// images, and the exact output bytes the engine is supposed to produce.
struct BuiltCase {
  PackedPlan packed;
  SramImage image;
  std::vector<int8_t> expected;  // output image, already requantised + laid out
};

// The C++ mirror of convacc.image.make_image(): pack, build the images, run the
// reference model, requantise, lay out.
BuiltCase build_case(const OpSpec& spec, const WeightTensor& weights, const std::string& layout,
                     const std::vector<int8_t>& activations, const QuantParams& quant);

}  // namespace convacc
