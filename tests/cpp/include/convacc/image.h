// The three 128 KB SRAM images and the descriptor block they carry.
//
// Mirrored from convacc/image.py. The descriptor layout is a hardware contract
// (hw/convacc_proc.x reads these words straight out of the weight image), so
// `Descriptor` is the one place it is written down: `encode()` and `decode()`
// are the two halves of the same definition, and a field can only move if both
// move.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "convacc/packer.h"
#include "convacc/spec.h"

namespace convacc {

class WordMemory {
 public:
  WordMemory() : words_(kWords, 0), next_word_(0) {}

  int alloc(int count);
  void write_bytes(int word_address, const uint8_t* payload, size_t length);
  void set(int address, uint64_t value) { words_[static_cast<size_t>(address)] = value; }
  uint64_t get(int address) const { return words_[static_cast<size_t>(address)]; }

  std::vector<uint8_t> to_bytes() const;

 private:
  std::vector<uint64_t> words_;
  int next_word_;
};

struct QuantParams {
  std::vector<int32_t> bias;
  std::vector<int32_t> scale;
  std::vector<uint8_t> shift;
  std::vector<int8_t> zero_point;  // signed byte, sign-extended into the image
};

struct SramImage {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> output;
  int descriptor_start = 0;
  int output_words = 0;
};

// Field order follows the words in the image; each comment names the word.
struct Descriptor {
  uint64_t flags = 0;  // 0: bit0 gemm, bit1 channelwise, bit2 deconv, bits3.. log2(r)
  uint64_t m = 0, n = 0;                              // 1
  uint64_t k = 0, batches = 0;                        // 2
  uint64_t mh = 0, mw = 0;                            // 3
  uint64_t k_groups = 0, o = 0;                       // 4
  uint64_t ci_count = 0, ci_base = 0, co_base = 0, co_per_group = 0;  // 5
  uint64_t hi = 0, wi = 0;                            // 6
  uint64_t ho = 0, wo = 0;                            // 7
  uint64_t ci = 0, co = 0;                            // 8
  uint64_t sh = 0, sw = 0, dh = 0, dw = 0;            // 9
  uint64_t ph = 0, pw = 0, phase_h = 0, phase_w = 0;  // 10
  uint64_t out_step_h = 0, out_step_w = 0;            // 11
  uint64_t act_base = 0, out_base = 0;                // 12
  uint64_t w_base = 0, param_base = 0;                // 13
  uint64_t tap_base = 0, taps = 0;                    // 14
  uint64_t next_desc = 0;                             // 15

  std::array<uint64_t, kHeaderWords> encode() const;
  static Descriptor decode(const std::array<uint64_t, kHeaderWords>& words);

  bool gemm() const { return (flags & 1) != 0; }
  bool channelwise() const { return (flags & 2) != 0; }
  bool deconv() const { return (flags & 4) != 0; }
  // The writer stores log2(r) above the three flag bits without masking it to
  // the two bits r actually needs, so read it back at the same width.
  int log_r() const { return static_cast<int>((flags >> 3) & 0xFF); }
  int r() const { return 1 << log_r(); }
  int m_tiles() const { return static_cast<int>((m + 7) / 8); }
  int n_tiles() const { return o ? static_cast<int>((n + o - 1) / o) : 0; }
};

namespace detail {

inline uint64_t pack(uint64_t value, int shift, int width) {
  const uint64_t mask = width >= 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  return (value & mask) << shift;
}

inline uint64_t unpack(uint64_t word, int shift, int width) {
  const uint64_t mask = width >= 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  return (word >> shift) & mask;
}

}  // namespace detail

inline std::array<uint64_t, kHeaderWords> Descriptor::encode() const {
  std::array<uint64_t, kHeaderWords> w{};
  w[0] = flags;
  w[1] = detail::pack(m, 0, 32) | detail::pack(n, 32, 32);
  w[2] = detail::pack(k, 0, 32) | detail::pack(batches, 32, 32);
  w[3] = detail::pack(mh, 0, 16) | detail::pack(mw, 16, 16);
  w[4] = detail::pack(k_groups, 0, 32) | detail::pack(o, 32, 32);
  w[5] = detail::pack(ci_count, 0, 16) | detail::pack(ci_base, 16, 16) |
         detail::pack(co_base, 32, 16) | detail::pack(co_per_group, 48, 16);
  w[6] = detail::pack(hi, 0, 16) | detail::pack(wi, 16, 16);
  w[7] = detail::pack(ho, 0, 16) | detail::pack(wo, 16, 16);
  w[8] = detail::pack(ci, 0, 16) | detail::pack(co, 16, 16);
  w[9] = detail::pack(sh, 0, 16) | detail::pack(sw, 16, 16) |
         detail::pack(dh, 32, 16) | detail::pack(dw, 48, 16);
  w[10] = detail::pack(ph, 0, 16) | detail::pack(pw, 16, 16) |
          detail::pack(phase_h, 32, 16) | detail::pack(phase_w, 48, 16);
  w[11] = detail::pack(out_step_h, 0, 16) | detail::pack(out_step_w, 16, 16);
  w[12] = detail::pack(act_base, 0, 32) | detail::pack(out_base, 32, 32);
  w[13] = detail::pack(w_base, 0, 32) | detail::pack(param_base, 32, 32);
  w[14] = detail::pack(tap_base, 0, 32) | detail::pack(taps, 32, 16);
  w[15] = next_desc;
  return w;
}

inline Descriptor Descriptor::decode(const std::array<uint64_t, kHeaderWords>& w) {
  Descriptor d;
  d.flags = w[0];
  d.m = detail::unpack(w[1], 0, 32);
  d.n = detail::unpack(w[1], 32, 32);
  d.k = detail::unpack(w[2], 0, 32);
  d.batches = detail::unpack(w[2], 32, 32);
  d.mh = detail::unpack(w[3], 0, 16);
  d.mw = detail::unpack(w[3], 16, 16);
  d.k_groups = detail::unpack(w[4], 0, 32);
  d.o = detail::unpack(w[4], 32, 32);
  d.ci_count = detail::unpack(w[5], 0, 16);
  d.ci_base = detail::unpack(w[5], 16, 16);
  d.co_base = detail::unpack(w[5], 32, 16);
  d.co_per_group = detail::unpack(w[5], 48, 16);
  d.hi = detail::unpack(w[6], 0, 16);
  d.wi = detail::unpack(w[6], 16, 16);
  d.ho = detail::unpack(w[7], 0, 16);
  d.wo = detail::unpack(w[7], 16, 16);
  d.ci = detail::unpack(w[8], 0, 16);
  d.co = detail::unpack(w[8], 16, 16);
  d.sh = detail::unpack(w[9], 0, 16);
  d.sw = detail::unpack(w[9], 16, 16);
  d.dh = detail::unpack(w[9], 32, 16);
  d.dw = detail::unpack(w[9], 48, 16);
  d.ph = detail::unpack(w[10], 0, 16);
  d.pw = detail::unpack(w[10], 16, 16);
  d.phase_h = detail::unpack(w[10], 32, 16);
  d.phase_w = detail::unpack(w[10], 48, 16);
  d.out_step_h = detail::unpack(w[11], 0, 16);
  d.out_step_w = detail::unpack(w[11], 16, 16);
  d.act_base = detail::unpack(w[12], 0, 32);
  d.out_base = detail::unpack(w[12], 32, 32);
  d.w_base = detail::unpack(w[13], 0, 32);
  d.param_base = detail::unpack(w[13], 32, 32);
  d.tap_base = detail::unpack(w[14], 0, 32);
  d.taps = detail::unpack(w[14], 32, 16);
  d.next_desc = w[15];
  return d;
}

// (positions, groups, n_tiles) of the engine's output layout.
struct OutputGeometry {
  int positions;
  int groups;
  int n_tiles;
};

OutputGeometry output_geometry(const PackedPlan& packed);

SramImage build_image(const PackedPlan& packed, const QuantParams& quant,
                      const std::vector<uint8_t>& activation, int activation_base = 0,
                      int output_base = 0);

}  // namespace convacc
