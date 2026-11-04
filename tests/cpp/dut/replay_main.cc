// Replays a libFuzzer corpus directory against the Verilated DUT.
//
// The fuzzer explores shapes at model speed; this tool turns every discovered
// corpus entry into an SRAM image and checks the RTL against the model. It is
// the bridge that makes the coverage-guided search count as *hardware* testing
// without paying millisecond-scale simulation in the feedback loop.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "convacc/cli.h"
#include "convacc/io.h"
#include "convacc/reference.h"
#include "convacc/parallel.h"
#include "convacc/compare.h"
#include "model/case_codec.h"
#include "dut/dut.h"
#include "dut/run.h"

namespace fs = std::filesystem;

namespace {

struct CorpusEntry {
  fs::path path;
  std::vector<uint8_t> bytes;
};

struct Result {
  fs::path path;
  std::string name;
  std::string case_line;
  convacc::Outcome run;
};

Result run_one(convacc::Dut& dut, const CorpusEntry& entry) {
  Result result;
  result.path = entry.path;
  result.name = entry.path.filename().string();

  convacc::FuzzCase decoded;
  if (!convacc::decode_case(entry.bytes.data(), entry.bytes.size(), &decoded)) {
    result.run.reason = "rejected by decoder";
    return result;
  }
  result.case_line = convacc::describe_case(decoded);

  convacc::BuiltCase built;
  try {
    built = convacc::build_case(decoded.spec, decoded.weights, decoded.layout, decoded.activations,
                                decoded.quant);
  } catch (const convacc::SpecError& e) {
    result.run.reason = std::string("model rejected: ") + e.what();
    return result;
  }
  result.run = convacc::run_built(dut, built);
  return result;
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --corpus DIR [--corpus DIR ...] [--jobs N] [--limit N]\n"
               "          [--list] [--save-failures DIR] [--quiet]\n"
               "\n"
               "  Each file in the corpus directories is one fuzzer input. Entries are\n"
               "  decoded to an operator, built into an SRAM image, run on the DUT and\n"
               "  compared byte for byte with the model.\n",
               argv0);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<fs::path> corpora;
  fs::path failures_dir;
  int jobs = 0;
  size_t limit = 0;
  bool list = false;
  bool quiet = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--corpus" && i + 1 < argc) {
      corpora.emplace_back(argv[++i]);
    } else if (arg == "--jobs" && i + 1 < argc) {
      if (!convacc::parse_int(argv[++i], &jobs)) {
        convacc::report_bad_number(arg.c_str());
        usage(argv[0]);
        return 2;
      }
    } else if (arg == "--limit" && i + 1 < argc) {
      long long value = 0;
      if (!convacc::parse_ll(argv[++i], &value) || value < 0) {
        std::fprintf(stderr, "error: --limit needs a non-negative integer\n");
        return 2;
      }
      limit = static_cast<size_t>(value);
    } else if (arg == "--save-failures" && i + 1 < argc) {
      failures_dir = argv[++i];
    } else if (arg == "--list") {
      list = true;
    } else if (arg == "--quiet") {
      quiet = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  if (corpora.empty()) {
    usage(argv[0]);
    return 2;
  }

  std::vector<CorpusEntry> entries;
  for (const auto& dir : corpora) {
    if (!fs::is_directory(dir)) {
      std::fprintf(stderr, "error: %s is not a directory\n", dir.c_str());
      return 2;
    }
    for (const auto& item : fs::recursive_directory_iterator(dir)) {
      if (!item.is_regular_file()) continue;
      entries.push_back({item.path(), convacc::read_file(item.path())});
      if (limit && entries.size() >= limit) break;
    }
    if (limit && entries.size() >= limit) break;
  }
  std::sort(entries.begin(), entries.end(),
            [](const CorpusEntry& a, const CorpusEntry& b) { return a.path < b.path; });
  if (entries.empty()) {
    std::fprintf(stderr, "error: no corpus files found\n");
    return 2;
  }

  if (list) {
    for (const auto& entry : entries) {
      convacc::FuzzCase decoded;
      if (!convacc::decode_case(entry.bytes.data(), entry.bytes.size(), &decoded)) {
        std::printf("%s\t<rejected>\n", entry.path.filename().string().c_str());
        continue;
      }
      std::printf("%s\t%s\n", entry.path.filename().string().c_str(),
                  convacc::describe_case(decoded).c_str());
    }
    return 0;
  }

  std::vector<Result> results(entries.size());
  const convacc::ParallelReport report = convacc::parallel_for(
      entries.size(), jobs, [] { return convacc::Dut(); },
      [&](convacc::Dut& dut, size_t index) { results[index] = run_one(dut, entries[index]); });

  size_t decoded = 0, built = 0, passed = 0;
  double sim_ms = 0.0;
  long long cycles = 0;
  std::vector<const Result*> failed;
  for (const auto& result : results) {
    if (!result.case_line.empty()) ++decoded;
    if (result.run.built) ++built;
    if (result.run.ok) ++passed;
    sim_ms += result.run.sim_ms;
    cycles += result.run.cycles;
    if (result.run.built && !result.run.ok) failed.push_back(&result);
  }

  std::printf("corpus: %zu files, %zu decoded, %zu built, %zu passed in %.2f s (%d threads)\n",
              entries.size(), decoded, built, passed, report.seconds, report.workers);
  if (built) {
    std::printf("dut: %.2f ms/case mean, %.1f cycles/case mean\n", sim_ms / built,
                static_cast<double>(cycles) / built);
  }

  if (!failed.empty()) {
    std::vector<convacc::Failure> failures;
    failures.reserve(failed.size());
    for (const Result* result : failed) {
      failures.push_back({result->name, result->run.reason, result->case_line});
    }
    convacc::report_failures(failures);
  }

  if (!failures_dir.empty()) {
    fs::create_directories(failures_dir);
    for (const Result* result : failed) {
      fs::copy_file(result->path, failures_dir / result->name,
                    fs::copy_options::overwrite_existing);
    }
  }

  if (!quiet) {
    std::printf("rejected by decoder or model: %zu\n", entries.size() - built);
  }
  return failed.empty() ? 0 : 1;
}
