#include "model/oracle.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "convacc/reference.h"

namespace convacc {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

uint64_t word_at(const std::vector<uint8_t>& bytes, int address) {
  if (address < 0 || address >= kWords) fail("word address out of range: " + std::to_string(address));
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(bytes[static_cast<size_t>(address) * 8 + i]) << (8 * i);
  }
  return value;
}

void expect_eq(const char* what, int64_t got, int64_t want, int index) {
  if (got != want) {
    fail(std::string("descriptor ") + what + " mismatch on task " + std::to_string(index) + ": " +
         std::to_string(got) + " != " + std::to_string(want));
  }
}

// Walks the descriptor chain the engine follows and cross-checks every field
// against the plan that produced it. Address arithmetic in build_image is the
// most bug-prone part of the port, and this is the only place it is read back
// the way hardware reads it.
void check_descriptors(const PackedPlan& plan, const SramImage& image) {
  int address = image.descriptor_start;
  if (address < 0 || address + kHeaderWords > kWords) fail("descriptor_start out of range");

  // The first descriptor legitimately lives at word 0, so 0 cannot be used as
  // the "end of chain" marker while walking; bound the walk by the task count
  // and require the last `next` to be 0.
  for (size_t index = 0; index < plan.tasks.size(); ++index) {
    if (address < 0 || address + kHeaderWords > kWords) {
      fail("descriptor " + std::to_string(index) + " address out of range");
    }

    const Task& task = plan.tasks[index].task;

    std::array<uint64_t, kHeaderWords> words{};
    for (int w = 0; w < kHeaderWords; ++w) words[static_cast<size_t>(w)] = word_at(image.weight, address + w);
    const Descriptor d = Descriptor::decode(words);

    const int i = static_cast<int>(index);
    expect_eq("m", static_cast<int64_t>(d.m), task.m, i);
    expect_eq("n", static_cast<int64_t>(d.n), task.n, i);
    expect_eq("k", static_cast<int64_t>(d.k), task.k, i);
    expect_eq("batches", static_cast<int64_t>(d.batches), task.batches, i);
    expect_eq("mh", static_cast<int64_t>(d.mh), task.mh, i);
    expect_eq("mw", static_cast<int64_t>(d.mw), task.mw, i);
    expect_eq("k_groups", static_cast<int64_t>(d.k_groups), task.k_groups(), i);
    expect_eq("o", static_cast<int64_t>(d.o), task.o(), i);
    expect_eq("ci_count", static_cast<int64_t>(d.ci_count), task.ci_count, i);
    expect_eq("ci_base", static_cast<int64_t>(d.ci_base), task.ci_base, i);
    expect_eq("co_base", static_cast<int64_t>(d.co_base), task.co_base, i);
    expect_eq("phase_h", static_cast<int64_t>(d.phase_h), task.phase_h, i);
    expect_eq("phase_w", static_cast<int64_t>(d.phase_w), task.phase_w, i);
    expect_eq("out_step_h", static_cast<int64_t>(d.out_step_h), task.out_step_h, i);
    expect_eq("out_step_w", static_cast<int64_t>(d.out_step_w), task.out_step_w, i);
    expect_eq("tap_count", static_cast<int64_t>(d.taps), static_cast<int64_t>(task.taps.size()), i);

    const bool gemm = is_gemm(plan.spec);
    const bool channelwise = task.mode == "channelwise";
    const bool deconv = is_deconv(plan.spec);
    int expected_r_bits = 0;
    for (int r = task.r; r > 1; r >>= 1) ++expected_r_bits;
    expect_eq("r_bits", d.log_r(), expected_r_bits, i);
    if (d.gemm() != gemm) fail("descriptor gemm flag mismatch");
    if (d.channelwise() != channelwise) fail("descriptor channelwise flag mismatch");
    if (d.deconv() != deconv) fail("descriptor deconv flag mismatch");

    if (!gemm) {
      const ConvSpec& s = as_conv(plan.spec);
      expect_eq("hi", static_cast<int64_t>(d.hi), s.hi, i);
      expect_eq("wi", static_cast<int64_t>(d.wi), s.wi, i);
      expect_eq("ho", static_cast<int64_t>(d.ho), s.ho, i);
      expect_eq("wo", static_cast<int64_t>(d.wo), s.wo, i);
      expect_eq("ci", static_cast<int64_t>(d.ci), s.ci, i);
      expect_eq("co", static_cast<int64_t>(d.co), s.co, i);
      expect_eq("sh", static_cast<int64_t>(d.sh), s.sh, i);
      expect_eq("sw", static_cast<int64_t>(d.sw), s.sw, i);
      expect_eq("ph", static_cast<int64_t>(d.ph), s.ph, i);
      expect_eq("pw", static_cast<int64_t>(d.pw), s.pw, i);
      expect_eq("co_per_group", static_cast<int64_t>(d.co_per_group), s.co_per_group(), i);
    }

    // Addresses must stay inside the 128KB window, and the payload they point
    // at must be byte-identical to what the packer produced.
    const std::vector<int8_t>& weights = plan.tasks[index].weights;
    const int weight_address = static_cast<int>(d.w_base);
    const int param_address = static_cast<int>(d.param_base);
    const int64_t weight_words = (static_cast<int64_t>(weights.size()) + 7) / 8;
    if (weight_address < 0 || weight_address + weight_words > kWords) {
      fail("weight address out of range");
    }
    if (param_address < 0 || param_address + 1LL * task.n_tiles() * kParamWordsPerTile > kWords) {
      fail("param address out of range");
    }
    for (size_t w = 0; w < weights.size(); ++w) {
      const uint8_t got = image.weight[static_cast<size_t>(weight_address) * 8 + w];
      if (got != static_cast<uint8_t>(weights[w])) fail("weight image differs from packed plan");
    }

    const int tap_address = static_cast<int>(d.tap_base);
    const int64_t tap_words = (static_cast<int64_t>(task.taps.size()) + 7) / 8;
    if (tap_address < 0 || tap_address + tap_words > kWords) fail("tap address out of range");

    // Lane padding is a hardware contract: absent lanes must carry the identity
    // quantisation (bias 0, scale 1, shift 0, zero point 0).
    for (int nt = 0; nt < task.n_tiles(); ++nt) {
      const int base = param_address + nt * kParamWordsPerTile;
      for (int lane = 0; lane < 8; ++lane) {
        // Lanes are indexed from the tile's first output channel, so the
        // padding test has to use o(), not 8 (a GEMM with r=8 packs one
        // channel per tile into lane 0 of eight consecutive param blocks).
        if (nt * task.o() + lane < task.n) continue;
        const uint64_t bias_word = word_at(image.weight, base + lane / 2);
        const uint64_t scale_word = word_at(image.weight, base + 4 + lane / 2);
        const int bias = static_cast<int>(lane % 2 ? (bias_word >> 32) : (bias_word & 0xffffffff));
        const int scale = static_cast<int>(lane % 2 ? (scale_word >> 32) : (scale_word & 0xffffffff));
        const int shift = static_cast<int>((word_at(image.weight, base + 8) >> (lane * 8)) & 0xFF);
        const int zp = static_cast<int>((word_at(image.weight, base + 9) >> (lane * 8)) & 0xFF);
        if (bias != 0 || scale != 1 || shift != 0 || zp != 0) {
          fail("padded lane " + std::to_string(lane) + " of tile " + std::to_string(nt) +
               " in task " + std::to_string(i) + " is not the identity quantisation (bias=" +
               std::to_string(bias) + " scale=" + std::to_string(scale) + " shift=" +
               std::to_string(shift) + " zp=" + std::to_string(zp) + ")");
        }
      }
    }

    if (index + 1 == plan.tasks.size()) {
      if (d.next_desc != 0) fail("descriptor chain continues past the last task");
      return;
    }
    if (d.next_desc == 0) {
      fail("descriptor chain ended after " + std::to_string(index + 1) + " of " +
           std::to_string(plan.tasks.size()) + " tasks");
    }
    if (static_cast<int>(d.next_desc) == address) fail("descriptor chain points at itself");
    address = static_cast<int>(d.next_desc);
  }
}

void check_requantize(const QuantParams& quant) {
  if (quant.scale.empty()) return;
  QuantParams one;
  one.bias = {quant.bias[0]};
  one.scale = {quant.scale[0]};
  one.shift = {quant.shift[0]};
  one.zero_point = {quant.zero_point[0]};

  std::vector<int32_t> accumulators;
  for (int i = -64; i <= 64; ++i) accumulators.push_back(i * 977);
  accumulators.push_back(-2147483647 - 1);
  accumulators.push_back(2147483647);
  accumulators.push_back(0);

  std::sort(accumulators.begin(), accumulators.end());
  const std::vector<int8_t> q = requantize(accumulators, one, 1);
  for (size_t i = 1; i < q.size(); ++i) {
    if (q[i] < q[i - 1]) fail("requantize is not monotonic in the accumulator");
  }

  // Independent formula for shift == 0: q = clamp((acc + bias) * scale + zp).
  if (one.shift[0] == 0) {
    for (int32_t acc : accumulators) {
      const int64_t scaled = (static_cast<int64_t>(acc) + one.bias[0]) * one.scale[0];
      const int64_t expected =
          std::clamp<int64_t>(scaled + one.zero_point[0], -128, 127);
      const size_t index = static_cast<size_t>(
          std::lower_bound(accumulators.begin(), accumulators.end(), acc) - accumulators.begin());
      if (q[index] != expected) {
        fail("shift=0 requantize mismatch: acc=" + std::to_string(acc) + " got " +
             std::to_string(static_cast<int>(q[index])) + " want " + std::to_string(expected));
      }
    }
  }
}

}  // namespace

void verify_case(const FuzzCase& c, const BuiltCase& built) {
  const Plan& plan = built.packed.plan;

  if (plan.tasks.empty()) fail("planner produced no tasks");
  if (plan.task_cycles.size() != plan.tasks.size()) fail("task_cycles and tasks disagree");
  long long sum = 0;
  for (int cycles : plan.task_cycles) {
    if (cycles <= 0) fail("task with non-positive cycle estimate");
    sum += cycles;
  }
  if (sum != plan.cycles()) fail("plan.cycles() is not the sum of task cycles");
  if (plan.utilization() < 0.0 || plan.utilization() > 1.0 + 1e-9) {
    fail("planner utilisation outside [0, 1]");
  }
  if (static_cast<int>(built.packed.tasks.size()) != static_cast<int>(plan.tasks.size())) {
    fail("packed task count differs from plan");
  }

  if (built.image.activation.size() != static_cast<size_t>(kSramBytes)) {
    fail("activation image is not 128KB");
  }
  if (built.image.weight.size() != static_cast<size_t>(kSramBytes)) {
    fail("weight image is not 128KB");
  }

  // The activation payload must be copied verbatim at base 0.
  const std::vector<uint8_t> acts(c.activations.begin(), c.activations.end());
  if (!acts.empty() &&
      std::memcmp(built.image.activation.data(), acts.data(), acts.size()) != 0) {
    fail("activation image does not start with the input tensor");
  }

  const OutputGeometry geo = output_geometry(built.packed);
  const int expected_words = geo.positions * geo.groups * geo.n_tiles;
  if (built.image.output_words != expected_words) fail("output_words disagrees with geometry");
  if (built.expected.size() != static_cast<size_t>(expected_words) * 8) {
    fail("expected output is not the full output window");
  }
  if (1LL * expected_words * 8 > kSramBytes) fail("output window exceeds SRAM");

  check_descriptors(built.packed, built.image);

  // Packed execution (the engine's view) and the semantic reference (the
  // operator's definition) must agree element for element.
  const std::vector<int32_t> semantic =
      reference_for_packed(built.packed, c.activations, c.weights, c.layout);
  const std::vector<int32_t> packed = execute_packed(built.packed, c.activations);
  if (semantic.size() != packed.size()) fail("packed/semantic result size mismatch");
  for (size_t i = 0; i < semantic.size(); ++i) {
    if (semantic[i] != packed[i]) {
      fail("packed execution disagrees with the reference at element " + std::to_string(i) + ": " +
           std::to_string(packed[i]) + " != " + std::to_string(semantic[i]));
    }
  }

  check_requantize(c.quant);
}

}  // namespace convacc
