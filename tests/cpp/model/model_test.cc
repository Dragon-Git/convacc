// The C++ model on its own -- no simulator.
//
// Two layers are checked here:
//   1. build_case() must reproduce Python's descriptor image and expected
//      output byte for byte;
//   2. execute_packed() must agree with the semantic reference (a packer bug
//      shows up here, not as a DUT mismatch).
//
// The third layer -- the Verilated DUT against the same expected image -- lives
// in dut/dataset_test.cc, so this file and its binary build and run on a
// machine with no Verilator at all.
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "convacc/compare.h"
#include "convacc/reference.h"
#include "model/dataset_cases.h"

namespace {

using DatasetCase = convacc::DatasetCase;

class DatasetCaseTest : public ::testing::TestWithParam<size_t> {
 protected:
  const DatasetCase& item() const { return convacc::dataset_cases().at(GetParam()); }
};

// ---------------------------------------------------------------------------
// 1. Descriptor image and expected output
// ---------------------------------------------------------------------------
TEST_P(DatasetCaseTest, RebuildsPythonDescriptorImage) {
  const DatasetCase& c = item();
  const convacc::BuiltCase built =
      convacc::build_case(c.spec, c.weights, c.layout, c.activations, c.quant);
  ASSERT_EQ(built.image.weight.size(), c.weight_image.size());
  const convacc::Mismatch diff = convacc::compare_bytes(
      built.image.weight.data(), built.image.weight.size(), c.weight_image.data(),
      c.weight_image.size());
  EXPECT_FALSE(diff.bad) << "descriptor image: " << convacc::describe_mismatch(diff);
}

TEST_P(DatasetCaseTest, RebuildsPythonExpectedOutput) {
  const DatasetCase& c = item();
  const convacc::BuiltCase built =
      convacc::build_case(c.spec, c.weights, c.layout, c.activations, c.quant);
  std::vector<uint8_t> expected_bytes(built.expected.size());
  if (!expected_bytes.empty()) {
    std::memcpy(expected_bytes.data(), built.expected.data(), built.expected.size());
  }
  ASSERT_EQ(expected_bytes.size(), c.expected.size());
  const convacc::Mismatch diff = convacc::compare_bytes(
      expected_bytes.data(), expected_bytes.size(),
      reinterpret_cast<const uint8_t*>(c.expected.data()), c.expected.size());
  EXPECT_FALSE(diff.bad) << "expected output: " << convacc::describe_mismatch(diff);
}

// ---------------------------------------------------------------------------
// 2. Packed execution vs semantic reference
// ---------------------------------------------------------------------------
TEST_P(DatasetCaseTest, ExecutePackedMatchesSemanticReference) {
  const DatasetCase& c = item();
  const convacc::PackedPlan packed = convacc::pack_weights(c.spec, c.weights, c.layout);
  const std::vector<int32_t> executed = convacc::execute_packed(packed, c.activations);
  const std::vector<int32_t> reference =
      convacc::reference_for_packed(packed, c.activations, c.weights, c.layout);
  ASSERT_EQ(executed.size(), reference.size());
  for (size_t i = 0; i < executed.size(); ++i) {
    ASSERT_EQ(executed[i], reference[i]) << "accumulator " << i << " of " << c.name;
  }
}

INSTANTIATE_TEST_SUITE_P(Dataset, DatasetCaseTest,
                         ::testing::Range(size_t{0}, convacc::dataset_cases().size()),
                         convacc::dataset_case_name);

// The packed GEMM path against a hand-written triple loop: the reference is
// computed from the semantic weights, so a packer or reduction-width mistake
// shows up as a value mismatch rather than as a DUT symptom.
TEST(GemmTest, ReferenceMatchesTripleLoop) {
  convacc::GemmSpec spec;
  spec.m = 13;
  spec.k = 7;
  spec.n = 11;

  std::vector<int8_t> a(static_cast<size_t>(spec.m) * spec.k);
  std::vector<int8_t> b(static_cast<size_t>(spec.k) * spec.n);
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int8_t>((i * 37) % 61 - 30);
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int8_t>((i * 53) % 41 - 20);

  convacc::WeightTensor weights{{spec.k, spec.n}, b};
  const convacc::PackedPlan packed = convacc::pack_weights(spec, weights, "KN");
  const std::vector<int32_t> reference =
      convacc::reference_for_packed(packed, a, weights, "KN");

  for (int i = 0; i < spec.m; ++i) {
    for (int j = 0; j < spec.n; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < spec.k; ++k) {
        acc += static_cast<int32_t>(a[static_cast<size_t>(i) * spec.k + k]) *
               static_cast<int32_t>(b[static_cast<size_t>(k) * spec.n + j]);
      }
      EXPECT_EQ(acc, reference[static_cast<size_t>(i) * spec.n + j]) << "(" << i << "," << j << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Spec validation must reject what the Python model rejects.
// ---------------------------------------------------------------------------
TEST(SpecTest, AcceptsMinimalConv) {
  convacc::ConvSpec spec;
  EXPECT_NO_THROW(spec.validate());
}

TEST(SpecTest, RejectsEvenKernel) {
  convacc::ConvSpec spec;
  spec.kh = 2;
  EXPECT_THROW(spec.validate(), convacc::SpecError);
}

TEST(SpecTest, RejectsGroupsNotDividingChannels) {
  convacc::ConvSpec spec;
  spec.ci = 3;
  spec.co = 4;
  spec.groups = 2;
  EXPECT_THROW(spec.validate(), convacc::SpecError);
}

TEST(SpecTest, RejectsOversizedGemm) {
  convacc::GemmSpec gemm;
  gemm.m = 256;
  gemm.k = 256;
  gemm.n = 256;
  EXPECT_NO_THROW(gemm.validate());  // 64 KiB per matrix, exactly the SRAM size
  gemm.n = 300;                      // beyond MAX_DIM
  EXPECT_THROW(gemm.validate(), convacc::SpecError);
}

TEST(SpecTest, RejectsDeconvWithGroups) {
  convacc::ConvSpec spec;
  spec.deconv = true;
  spec.groups = 2;
  spec.ci = 2;
  spec.co = 2;
  EXPECT_THROW(spec.validate(), convacc::SpecError);
}

TEST(BuildImageTest, RejectsUnalignedOutputBase) {
  convacc::ConvSpec spec;
  std::vector<int8_t> weights(1, 1);
  convacc::WeightTensor tensor{{1, 1, 1, 1, 1}, weights};
  const convacc::PackedPlan packed = convacc::pack_weights(spec, tensor, "HWGIO");
  convacc::QuantParams quant{{0}, {1}, {0}, {0}};
  EXPECT_THROW(convacc::build_image(packed, quant, {}, 0, 4), convacc::SpecError);
}

TEST(BuildImageTest, RejectsWrongQuantWidth) {
  convacc::ConvSpec spec;
  std::vector<int8_t> weights(1, 1);
  convacc::WeightTensor tensor{{1, 1, 1, 1, 1}, weights};
  const convacc::PackedPlan packed = convacc::pack_weights(spec, tensor, "HWGIO");
  convacc::QuantParams quant{{0, 0}, {1}, {0}, {0}};  // bias has 2 entries, co == 1
  EXPECT_THROW(convacc::build_image(packed, quant, {}), convacc::SpecError);
}

}  // namespace
