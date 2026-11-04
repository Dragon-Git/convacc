// The Verilated DUT against the images the Python model produced.
//
// The model-side checks (descriptor image, expected output, packed execution)
// live in model/model_test.cc. This is the one layer that needs a simulator,
// and it is why dut/ is a directory and a binary of its own.
#include <gtest/gtest.h>

#include "convacc/reference.h"
#include "dut/dut.h"
#include "dut/run.h"
#include "model/dataset_cases.h"

namespace {

using convacc::DatasetCase;

class DatasetCaseTest : public ::testing::TestWithParam<size_t> {
 protected:
  const DatasetCase& item() const { return convacc::dataset_cases().at(GetParam()); }
};

TEST_P(DatasetCaseTest, DutProducesExpectedImage) {
  const DatasetCase& c = item();
  const convacc::BuiltCase built =
      convacc::build_case(c.spec, c.weights, c.layout, c.activations, c.quant);

  convacc::Dut dut;
  const convacc::Outcome outcome = convacc::run_built(dut, built);
  EXPECT_TRUE(outcome.ok) << c.name << ": " << outcome.reason;
}

INSTANTIATE_TEST_SUITE_P(Dataset, DatasetCaseTest,
                         ::testing::Range(size_t{0}, convacc::dataset_cases().size()),
                         convacc::dataset_case_name);

}  // namespace
