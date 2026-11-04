// Parallel golden regression: run every case through the Verilated DUT and
// compare the output image byte for byte.
//
// Two sources of cases:
//   --cases DIR   pre-rendered SRAM images on disk (what tools/gen_cpp_cases.py
//                 writes, and what the cocotb suite runs);
//   --gen-scale N generate them in-process from driver/generator.cc.
//
// The generator exists because the 10x corpus is 100k cases and 25 GB of images.
// Reading that back costs more than the simulation itself: this sandbox is
// capped at 8 GiB of page cache, so the corpus never stays resident. Generating
// a case takes a few microseconds, so the filesystem drops out of the loop
// entirely.
//
// Each worker owns one Dut instance for its whole lifetime, so the model is
// elaborated once per thread instead of once per case (which is what the
// one-process-per-chunk cocotb layout does), and the three SRAM images are
// loaded with memcpy instead of VPI word writes. The pool itself lives in
// convacc/parallel.h, shared with the corpus replay.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "dut/generator.h"
#include "dut/case.h"
#include "convacc/cli.h"
#include "convacc/dataset.h"
#include "convacc/io.h"
#include "convacc/parallel.h"
#include "convacc/compare.h"
#include "dut/dut.h"
#include "dut/run.h"

namespace fs = std::filesystem;

namespace {

// The dump is a CVCC bundle, the same format tools/gen_cpp_cases.py writes and
// the gtest suite reads back. Format and fill order live in convacc/dataset.h.
void fill_dataset(convacc::DatasetCase* out, const convacc::Prepared& prepared) {
  out->name = prepared.gen.name;
  out->spec = prepared.gen.spec;
  out->layout = prepared.gen.layout;
  out->quant = prepared.gen.quant;
  out->weights = prepared.gen.weights;
  out->activations = prepared.gen.activations;
  out->expected = prepared.built.expected;
  out->weight_image = prepared.built.image.weight;
}

// `--dump-only` is read back by a script that does not care about quoting.
std::string json_escape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (const char ch : text) {
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char item[8];
          std::snprintf(item, sizeof(item), "\\u%04x", static_cast<unsigned>(ch));
          out += item;
        } else {
          out += ch;
        }
    }
  }
  return out;
}

struct Options {
  std::string cases_dir = "build/cocotb_cases";
  std::string filter;
  std::string json_path;
  std::string dump_path;
  int dump_count = 200;
  int gen_scale = 0;
  convacc::Grid grid = convacc::Grid::kSmall;
  bool grid_explicit = false;
  long long cycle_budget = 0;
  int jobs = 0;
  int shard_index = 0;
  int shard_count = 1;
  bool quiet = false;
  bool list_only = false;
  bool dump_only = false;
};

struct CaseList {
  bool generated = false;
  std::vector<convacc::ShapeCase> shapes;  // generated mode
  std::vector<fs::path> dirs;              // --cases mode
  std::vector<std::string> names;          // --cases mode, parallel to dirs

  size_t size() const { return generated ? shapes.size() : dirs.size(); }
  const std::string& name(size_t index) const {
    return generated ? shapes[index].name : names[index];
  }
};

void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s [--cases DIR | --gen-scale N | --grid NAME] [options]\n"
          "\n"
          "  --cases DIR     one subdirectory per case, each holding\n"
          "                  activation.bin, weight.bin and expected.bin\n"
          "  --gen-scale N   generate the corpus in-process instead of reading it\n"
          "                  (1 = the ~10k golden set, 10 = 100k)\n"
          "  --grid NAME     small | wide | full: full keeps every legal shape\n"
          "                  combination (~1.68M cases), ignoring --gen-scale\n"
          "  --cycle-budget N  cap the total simulated cycles (0 = no cap); the grid\n"
          "                  axes make each case longer, so this is what keeps a 10x\n"
          "                  grid from costing 26x the time\n"
          "  --jobs N        worker threads (default: hardware concurrency)\n"
          "  --filter SUBSTR keep only cases whose name contains SUBSTR\n"
          "  --shard I/N     run every Nth case starting at I (a CI matrix splits\n"
          "                  one grid this way; shards are disjoint and exhaustive)\n"
          "  --list          print case names and shapes, run nothing\n"
          "  --json PATH     write a machine-readable summary\n"
          "  --dump PATH     write the generated cases to PATH as a CVCC dataset,\n"
          "                  for tools/verify_generated_cases.py to re-derive in Python\n"
          "  --dump-count N  how many cases to dump (default 200)\n"
          "  --dump-only     build and dump, skip simulation (for cross-checking)\n"
          "  --quiet         suppress the progress line\n",
          argv0);
}

// "I/N", with 0 <= I < N. A malformed shard is an error rather than a silent
// fallback to N = 1: otherwise a typo turns every runner in a CI matrix into a
// full-grid run, which reads as a slow build and not as a broken job.
bool parse_shard(const char* text, int* index, int* count) {
  const std::string value(text);
  const size_t slash = value.find('/');
  if (slash == std::string::npos) return false;
  if (!convacc::parse_int(value.substr(0, slash).c_str(), index)) return false;
  if (!convacc::parse_int(value.substr(slash + 1).c_str(), count)) return false;
  return *count >= 1 && *count <= 10000 && *index >= 0 && *index < *count;
}

// 0 when the arguments parsed, otherwise the status main should return.
int parse_args(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--cases" && has_value) {
      opt->cases_dir = argv[++i];
    } else if (arg == "--gen-scale" && has_value) {
      if (!convacc::parse_int(argv[++i], &opt->gen_scale)) {
        convacc::report_bad_number(arg.c_str());
        usage(argv[0]);
        return 2;
      }
    } else if (arg == "--grid" && has_value) {
      const std::string value = argv[++i];
      if (value == "small") {
        opt->grid = convacc::Grid::kSmall;
      } else if (value == "wide") {
        opt->grid = convacc::Grid::kWide;
      } else if (value == "full") {
        opt->grid = convacc::Grid::kFull;
      } else {
        fprintf(stderr, "error: --grid needs small|wide|full\n");
        usage(argv[0]);
        return 2;
      }
      opt->grid_explicit = true;
      if (opt->gen_scale == 0) opt->gen_scale = 1;
    } else if (arg == "--cycle-budget" && has_value) {
      if (!convacc::parse_ll(argv[++i], &opt->cycle_budget)) {
        convacc::report_bad_number(arg.c_str());
        usage(argv[0]);
        return 2;
      }
    } else if (arg == "--jobs" && has_value) {
      if (!convacc::parse_int(argv[++i], &opt->jobs)) {
        convacc::report_bad_number(arg.c_str());
        usage(argv[0]);
        return 2;
      }
    } else if (arg == "--filter" && has_value) {
      opt->filter = argv[++i];
    } else if (arg == "--shard" && has_value) {
      if (!parse_shard(argv[++i], &opt->shard_index, &opt->shard_count)) {
        fprintf(stderr, "error: --shard needs I/N with 0 <= I < N\n");
        usage(argv[0]);
        return 2;
      }
    } else if (arg == "--json" && has_value) {
      opt->json_path = argv[++i];
    } else if (arg == "--dump" && has_value) {
      opt->dump_path = argv[++i];
    } else if (arg == "--dump-count" && has_value) {
      if (!convacc::parse_int(argv[++i], &opt->dump_count)) {
        convacc::report_bad_number(arg.c_str());
        usage(argv[0]);
        return 2;
      }
    } else if (arg == "--list") {
      opt->list_only = true;
    } else if (arg == "--dump-only") {
      opt->dump_only = true;
    } else if (arg == "--quiet") {
      opt->quiet = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return -1;
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (opt->gen_scale < 0) opt->gen_scale = 0;
  return 0;
}

// 0 on success, otherwise the status main should return.
int load_cases(const Options& opt, CaseList* list) {
  if (opt.gen_scale > 0) {
    list->generated = true;
    list->shapes = opt.grid_explicit
                       ? convacc::make_shape_list(opt.grid, opt.gen_scale, opt.cycle_budget)
                       : convacc::make_shape_list(opt.gen_scale, opt.cycle_budget);
    if (!opt.filter.empty()) {
      std::vector<convacc::ShapeCase> kept;
      kept.reserve(list->shapes.size());
      for (auto& shape : list->shapes) {
        if (shape.name.find(opt.filter) != std::string::npos) kept.push_back(std::move(shape));
      }
      list->shapes.swap(kept);
    }
    if (list->shapes.empty()) {
      fprintf(stderr, "error: no generated cases match '%s'\n", opt.filter.c_str());
      return 2;
    }
    return 0;
  }

  const fs::path root(opt.cases_dir);
  if (!fs::is_directory(root)) {
    fprintf(stderr, "error: %s is not a directory\n", opt.cases_dir.c_str());
    return 2;
  }
  for (const auto& entry : fs::directory_iterator(root)) {
    if (!entry.is_directory()) continue;
    const std::string name = entry.path().filename().string();
    if (!opt.filter.empty() && name.find(opt.filter) == std::string::npos) continue;
    list->dirs.push_back(entry.path());
  }
  // Directory order is filesystem-defined; sort by name so a run is repeatable
  // and the case numbers in the log mean the same thing twice.
  std::sort(list->dirs.begin(), list->dirs.end(),
            [](const fs::path& a, const fs::path& b) { return a.filename() < b.filename(); });
  if (list->dirs.empty()) {
    fprintf(stderr, "error: no cases found under %s\n", opt.cases_dir.c_str());
    return 2;
  }
  list->names.reserve(list->dirs.size());
  for (const fs::path& dir : list->dirs) list->names.push_back(dir.filename().string());
  return 0;
}

// Keep every count-th case, starting at index. Sharding happens after the
// filter so the shards partition exactly what the job would otherwise run:
// filtering inside each shard would duplicate or drop cases at the boundaries.
void apply_shard(const Options& opt, CaseList* list) {
  if (opt.shard_count <= 1) return;
  const size_t step = static_cast<size_t>(opt.shard_count);
  const size_t first = static_cast<size_t>(opt.shard_index);

  if (list->generated) {
    std::vector<convacc::ShapeCase> kept;
    kept.reserve(list->shapes.size() / step + 1);
    for (size_t i = first; i < list->shapes.size(); i += step) {
      kept.push_back(std::move(list->shapes[i]));
    }
    list->shapes.swap(kept);
    return;
  }

  std::vector<fs::path> dirs;
  std::vector<std::string> names;
  dirs.reserve(list->dirs.size() / step + 1);
  names.reserve(list->dirs.size() / step + 1);
  for (size_t i = first; i < list->dirs.size(); i += step) {
    dirs.push_back(std::move(list->dirs[i]));
    names.push_back(std::move(list->names[i]));
  }
  list->dirs.swap(dirs);
  list->names.swap(names);
}

void list_cases(const CaseList& list) {
  if (!list.generated) {
    for (const auto& name : list.names) printf("%s\n", name.c_str());
    return;
  }
  for (const auto& shape : list.shapes) {
    if (shape.family == convacc::Family::kGemm) {
      printf("%s\tgemm m=%d k=%d n=%d\n", shape.name.c_str(), shape.gemm.m, shape.gemm.k,
             shape.gemm.n);
    } else {
      const auto& c = shape.conv;
      printf("%s\t%s n=%d %dx%d ci=%d %dx%d co=%d k=%dx%d s=%d g=%d\n", shape.name.c_str(),
             convacc::family_name(shape.family), c.n, c.hi, c.wi, c.ci, c.ho, c.wo, c.co, c.kh,
             c.kw, c.sh, c.groups);
    }
  }
}

convacc::Outcome run_from_disk(convacc::Dut& dut, const fs::path& dir) {
  convacc::Outcome out;
  const std::vector<uint8_t> expected = convacc::read_file(dir / "expected.bin");
  if (expected.empty()) {
    out.reason = "missing or empty expected.bin";
    return out;
  }
  convacc::SramImage image;
  image.activation = convacc::read_file(dir / "activation.bin");
  image.weight = convacc::read_file(dir / "weight.bin");
  const size_t want_bytes = static_cast<size_t>(convacc::kSramBytes);
  if (image.activation.size() != want_bytes || image.weight.size() != want_bytes) {
    out.reason = "SRAM images must be " + std::to_string(convacc::kSramBytes) + " bytes";
    return out;
  }
  return convacc::run_built(dut, image, expected.data(), expected.size());
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  const int parsed = parse_args(argc, argv, &opt);
  if (parsed) return parsed < 0 ? 0 : parsed;

  CaseList list;
  const int loaded = load_cases(opt, &list);
  if (loaded) return loaded;

  apply_shard(opt, &list);
  if (list.size() == 0) {
    fprintf(stderr, "error: shard %d/%d selected no cases\n", opt.shard_index, opt.shard_count);
    return 2;
  }

  if (opt.list_only) {
    list_cases(list);
    return 0;
  }

  std::vector<convacc::DatasetCase> dumps;
  std::vector<char> dumped;
  if (!opt.dump_path.empty()) {
    if (!list.generated) {
      fprintf(stderr, "error: --dump only applies to --gen-scale\n");
      return 2;
    }
    const size_t slots = static_cast<size_t>(opt.dump_count > 0 ? opt.dump_count : 0);
    dumps.resize(slots);
    dumped.assign(slots, 0);
  }

  size_t total = list.size();
  // A dump-only run has nothing to check beyond the dumped prefix.
  if (opt.dump_only && !dumps.empty() && total > dumps.size()) total = dumps.size();

  std::vector<convacc::Outcome> results(total);
  const convacc::ParallelReport report = convacc::parallel_for(
      total, opt.jobs, [] { return convacc::Dut(); },
      [&](convacc::Dut& dut, size_t index) {
        if (!list.generated) {
          results[index] = run_from_disk(dut, list.dirs[index]);
          return;
        }
        const bool keep = index < dumps.size();
        convacc::Prepared prepared;
        results[index] =
            convacc::run_shape(dut, list.shapes[index], keep ? &prepared : nullptr,
                               /*simulate=*/!opt.dump_only);
        if (keep && results[index].built) {
          fill_dataset(&dumps[index], prepared);
          dumped[index] = 1;
        }
      },
      [&](size_t done, double elapsed) {
        std::fprintf(stderr, "\r  %zu/%zu cases (%.0f cases/s)", done, total, done / elapsed);
        std::fflush(stderr);
      },
      /*progress_every=*/opt.quiet ? 0 : 500);
  const double seconds = report.seconds;

  if (!opt.quiet) fprintf(stderr, "\r%*s\r", 48, "");

  size_t passed = 0;
  double sim_ms = 0.0;
  double build_ms = 0.0;
  long long cycles = 0;
  std::vector<size_t> failed;
  for (size_t i = 0; i < results.size(); ++i) {
    sim_ms += results[i].sim_ms;
    build_ms += results[i].build_ms;
    cycles += results[i].cycles;
    if (results[i].ok) {
      ++passed;
    } else {
      failed.push_back(i);
    }
  }

  printf("%zu cases, %d threads%s\n", total, report.workers,
         list.generated ? " (generated)" : "");
  printf("PASS %zu/%zu in %.2f s (%.0f cases/s, %.2f ms/case wall)\n", passed, total, seconds,
         total / seconds, seconds * 1000.0 / total);
  printf("sim %.1f s total (%.2f ms/case mean, %.1f cycles/case mean)\n", sim_ms / 1000.0,
         sim_ms / total, static_cast<double>(cycles) / total);
  if (list.generated) {
    printf("build %.1f s total (%.3f ms/case mean)\n", build_ms / 1000.0, build_ms / total);
  }

  if (!failed.empty()) {
    std::vector<convacc::Failure> failures;
    failures.reserve(failed.size());
    for (size_t index : failed) failures.push_back({list.name(index), results[index].reason, ""});
    convacc::report_failures(failures);
  }

  if (!opt.dump_path.empty()) {
    std::vector<convacc::DatasetCase> kept;
    kept.reserve(dumps.size());
    for (size_t i = 0; i < dumps.size(); ++i) {
      if (dumped[i]) kept.push_back(std::move(dumps[i]));
    }
    convacc::write_dataset(opt.dump_path, kept);
    printf("wrote %zu cases to %s\n", kept.size(), opt.dump_path.c_str());
  }

  if (!opt.json_path.empty()) {
    std::ofstream out(opt.json_path);
    out << "{\n  \"cases\": " << total << ",\n  \"passed\": " << passed
        << ",\n  \"seconds\": " << seconds << ",\n  \"generated\": "
        << (list.generated ? "true" : "false") << ",\n  \"failures\": [\n";
    for (size_t i = 0; i < failed.size(); ++i) {
      out << "    {\"name\": \"" << json_escape(list.name(failed[i])) << "\", \"reason\": \""
          << json_escape(results[failed[i]].reason) << "\"}"
          << (i + 1 < failed.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
  }

  return failed.empty() ? 0 : 1;
}
