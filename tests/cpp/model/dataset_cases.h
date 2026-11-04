// The sampled dataset the Python model wrote, loaded once.
//
// `make native-cases` produces it: the spec, the *unpacked* weights, the
// quantisation parameters and Python's own images, for a sample of cases. Both
// halves of the tree read it -- the model tests rebuild the images from it, the
// DUT tests run the same cases on the simulator -- so the loader sits here and
// both include it.
#pragma once

#include <gtest/gtest.h>

#include <cctype>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "convacc/dataset.h"

namespace convacc {

// A missing dataset leaves this empty, which makes every dataset-derived test
// disappear instead of failing the build.
inline const std::vector<DatasetCase>& dataset_cases() {
  static const std::vector<DatasetCase> cases = [] {
    const char* env = std::getenv("CONVACC_DATASET");
#ifdef CONVACC_DATASET_DEFAULT
    const std::string path = env ? env : CONVACC_DATASET_DEFAULT;
#else
    const std::string path = env ? env : "build/cpp_cases/dataset.bin";
#endif
    try {
      return read_dataset(path);
    } catch (const std::exception&) {
      return std::vector<DatasetCase>{};
    }
  }();
  return cases;
}

// gtest parameter names may only contain alphanumerics, so the case names are
// sanitised here rather than in each test file.
inline std::string dataset_case_name(const ::testing::TestParamInfo<size_t>& info) {
  const auto& cases = dataset_cases();
  if (info.param >= cases.size()) return "unused";
  std::string name = cases[info.param].name;
  for (char& ch : name) {
    if (!std::isalnum(static_cast<unsigned char>(ch))) ch = '_';
  }
  return name;
}

}  // namespace convacc
