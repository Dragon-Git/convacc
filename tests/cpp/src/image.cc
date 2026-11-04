#include "convacc/image.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <utility>

namespace convacc {
namespace {

int flag_bits(const OpSpec& spec, const Task& task) {
  int flags = is_gemm(spec) ? 1 : 0;
  if (task.mode == "channelwise") flags |= 1 << 1;
  if (is_deconv(spec)) flags |= 1 << 2;
  int bits = 0;
  for (int r = task.r; r > 1; r >>= 1) ++bits;  // r.bit_length() - 1
  flags |= bits << 3;
  return flags;
}

}  // namespace

int WordMemory::alloc(int count) {
  if (next_word_ + count > kWords) throw SpecError("SRAM image exceeds 128KB");
  const int address = next_word_;
  next_word_ += count;
  return address;
}

void WordMemory::write_bytes(int word_address, const uint8_t* payload, size_t length) {
  if (static_cast<size_t>(word_address) * 8 + length > static_cast<size_t>(kSramBytes)) {
    throw SpecError("SRAM image write exceeds 128KB");
  }
  const size_t padded = length + ((8 - length % 8) % 8);
  for (size_t i = 0; i < padded; i += 8) {
    uint64_t word = 0;
    for (size_t b = 0; b < 8; ++b) {
      const size_t src = i + b;
      if (src < length) word |= static_cast<uint64_t>(payload[src]) << (8 * b);
    }
    words_[static_cast<size_t>(word_address) + i / 8] = word;
  }
}

std::vector<uint8_t> WordMemory::to_bytes() const {
  std::vector<uint8_t> out(words_.size() * 8);
  for (size_t i = 0; i < words_.size(); ++i) {
    for (int b = 0; b < 8; ++b) out[i * 8 + b] = static_cast<uint8_t>(words_[i] >> (8 * b));
  }
  return out;
}

OutputGeometry output_geometry(const PackedPlan& packed) {
  std::set<std::pair<int, int>> tiles;
  for (const auto& t : packed.tasks) tiles.insert({t.task.o(), t.task.n_tiles()});
  if (tiles.size() != 1) {
    throw SpecError(
        "all tasks must share o/n_tiles so the output layout stays addressable: got " +
        std::to_string(tiles.size()) + " distinct pairs");
  }
  const int n_tiles = tiles.begin()->second;
  const OpSpec& spec = packed.spec;
  if (is_gemm(spec)) {
    return {as_gemm(spec).m, 1, n_tiles};
  }
  const ConvSpec& c = as_conv(spec);
  if (c.channelwise()) return {c.n * c.ho * c.wo, 1, n_tiles};
  return {c.n * c.ho * c.wo, c.co / c.co_per_group(), n_tiles};
}

SramImage build_image(const PackedPlan& packed, const QuantParams& quant,
                      const std::vector<uint8_t>& activation, int activation_base,
                      int output_base) {
  const int width = param_width(packed.spec);
  const auto expect = [width](const char* name, size_t got) {
    if (static_cast<int>(got) != width) {
      throw SpecError(std::string(name) + " must have shape (" + std::to_string(width) + ",)");
    }
  };
  expect("bias", quant.bias.size());
  expect("scale", quant.scale.size());
  expect("shift", quant.shift.size());
  expect("zero_point", quant.zero_point.size());

  WordMemory memory;
  struct Placement {
    const PackedTask* task;
    int desc;
    int tap;
    int weight;
    int param;
  };
  std::vector<Placement> placements;

  for (const auto& packed_task : packed.tasks) {
    const Task& task = packed_task.task;
    const int desc_address = memory.alloc(kHeaderWords);

    std::vector<uint8_t> tap_bytes(task.taps.size());
    for (size_t i = 0; i < task.taps.size(); ++i) {
      tap_bytes[i] = static_cast<uint8_t>(task.taps[i].first * 16 + task.taps[i].second);
    }
    const int tap_address = memory.alloc(static_cast<int>((tap_bytes.size() + 7) / 8));
    memory.write_bytes(tap_address, tap_bytes.data(), tap_bytes.size());

    const int weight_address = memory.alloc(static_cast<int>((packed_task.weights.size() + 7) / 8));
    memory.write_bytes(weight_address, reinterpret_cast<const uint8_t*>(packed_task.weights.data()),
                       packed_task.weights.size());

    const int param_address = memory.alloc(task.n_tiles() * kParamWordsPerTile);
    for (int nt = 0; nt < task.n_tiles(); ++nt) {
      const int base = param_address + nt * kParamWordsPerTile;
      const int co_start = task.co_base + nt * task.o();
      for (int lane = 0; lane < 8; ++lane) {
        const int co = co_start + lane;
        int32_t bias_v, scale_v;
        int shift_v, zp_v;
        if (co < task.co_base + task.n) {
          bias_v = quant.bias[static_cast<size_t>(co)];
          scale_v = quant.scale[static_cast<size_t>(co)];
          shift_v = quant.shift[static_cast<size_t>(co)];
          // zero_point is int8; the image stores it sign-extended into 8 bits,
          // which is exactly its two's-complement byte.
          zp_v = static_cast<uint8_t>(quant.zero_point[static_cast<size_t>(co)]);
        } else {
          bias_v = 0;
          scale_v = 1;
          shift_v = 0;
          zp_v = 0;
        }
        const int half = lane % 2;
        int word = base + lane / 2;
        if (half == 0) {
          memory.set(word, static_cast<uint32_t>(bias_v));
        } else {
          memory.set(word, memory.get(word) | (static_cast<uint64_t>(static_cast<uint32_t>(bias_v)) << 32));
        }
        word = base + 4 + lane / 2;
        if (half == 0) {
          memory.set(word, static_cast<uint32_t>(scale_v));
        } else {
          memory.set(word, memory.get(word) | (static_cast<uint64_t>(static_cast<uint32_t>(scale_v)) << 32));
        }
        memory.set(base + 8, memory.get(base + 8) | (static_cast<uint64_t>(shift_v & 0xFF) << (lane * 8)));
        memory.set(base + 9, memory.get(base + 9) | (static_cast<uint64_t>(zp_v & 0xFF) << (lane * 8)));
      }
    }
    placements.push_back({&packed_task, desc_address, tap_address, weight_address, param_address});
  }

  const OpSpec& spec = packed.spec;
  const bool gemm = is_gemm(spec);
  const ConvSpec* conv = gemm ? nullptr : &as_conv(spec);
  for (size_t index = 0; index < placements.size(); ++index) {
    const Placement& p = placements[index];
    const Task& task = p.task->task;

    Descriptor d;
    d.flags = static_cast<uint64_t>(flag_bits(spec, task));
    d.m = task.m;
    d.n = task.n;
    d.k = task.k;
    d.batches = task.batches;
    d.mh = task.mh;
    d.mw = task.mw;
    d.k_groups = task.k_groups();
    d.o = task.o();
    d.ci_count = task.ci_count;
    d.ci_base = task.ci_base;
    d.co_base = task.co_base;
    d.co_per_group = conv ? conv->co_per_group() : 1;
    d.hi = conv ? conv->hi : 1;
    d.wi = conv ? conv->wi : 1;
    d.ho = conv ? conv->ho : 1;
    d.wo = conv ? conv->wo : 1;
    d.ci = conv ? conv->ci : 1;
    d.co = conv ? conv->co : 1;
    d.sh = conv ? conv->sh : 1;
    d.sw = conv ? conv->sw : 1;
    d.dh = conv ? conv->dh : 1;
    d.dw = conv ? conv->dw : 1;
    d.ph = conv ? conv->ph : 0;
    d.pw = conv ? conv->pw : 0;
    d.phase_h = task.phase_h;
    d.phase_w = task.phase_w;
    d.out_step_h = task.out_step_h;
    d.out_step_w = task.out_step_w;
    d.act_base = activation_base;
    d.out_base = output_base;
    d.w_base = p.weight;
    d.param_base = p.param;
    d.tap_base = p.tap;
    d.taps = task.taps.size();
    d.next_desc = (index + 1 < placements.size()) ? placements[index + 1].desc : 0;

    const auto words = d.encode();
    for (int i = 0; i < kHeaderWords; ++i) memory.set(p.desc + i, words[i]);
  }

  const OutputGeometry geo = output_geometry(packed);
  if (output_base % 8 != 0) throw SpecError("output_base must be 64-bit aligned");
  const int output_words = geo.positions * geo.groups * geo.n_tiles;
  if (static_cast<int64_t>(output_base) + 1LL * output_words * 8 > kSramBytes) {
    throw SpecError("output image needs " + std::to_string(output_words * 8) + " bytes at " +
                    std::to_string(output_base) + "; SRAM is " + std::to_string(kSramBytes));
  }
  if (static_cast<int>(activation.size()) > kSramBytes) {
    throw SpecError("activation image exceeds 128KB");
  }

  SramImage image;
  image.activation.assign(static_cast<size_t>(kSramBytes), 0);
  std::copy(activation.begin(), activation.end(), image.activation.begin());
  image.weight = memory.to_bytes();
  image.output.assign(static_cast<size_t>(kSramBytes), 0);
  image.descriptor_start = 0;
  image.output_words = output_words;
  return image;
}

}  // namespace convacc
