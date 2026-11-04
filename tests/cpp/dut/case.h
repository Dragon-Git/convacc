// One case, from shape to verdict.
//
// The golden regression, the gtest grid and the FuzzTest suite all ran the same
// steps -- materialise, pack, simulate, compare -- and differed only in what
// they did with the verdict: count it, turn it into ADD_FAILURE(), turn it into
// an AssertionResult. So the verdict is what this returns.
#pragma once

#include <chrono>
#include <exception>
#include <string>

#include "dut/generator.h"
#include "dut/run.h"

namespace convacc {

// The intermediates that `--gen-scale --dump` needs: the Python cross-check
// rebuilds the case from the spec and compares both images, so it needs the
// generated inputs as well as the built ones.
struct Prepared {
  GeneratedCase gen;
  BuiltCase built;
};

inline double millis_since(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

// Materialises `shape`, packs it and -- unless `simulate` is false -- runs it.
//
// `prepared`, when non-null, receives the intermediates. Pass null on the
// 100k-case path: nothing case-sized is then kept alive past the comparison.
inline Outcome run_shape(Dut& dut, const ShapeCase& shape, Prepared* prepared = nullptr,
                         bool simulate = true) {
  Outcome out;
  Prepared scratch;
  Prepared& p = prepared ? *prepared : scratch;

  const auto start = std::chrono::steady_clock::now();
  try {
    p.gen = materialise(shape);
    p.built = build_case(p.gen.spec, p.gen.weights, p.gen.layout, p.gen.activations, p.gen.quant);
  } catch (const std::exception& e) {
    out.reason = std::string("case build failed: ") + e.what();
    out.build_ms = millis_since(start);
    return out;
  }
  out.build_ms = millis_since(start);

  if (!simulate) {
    out.built = true;
    out.ok = true;
    return out;
  }
  Outcome run = run_built(dut, p.built);
  run.build_ms = out.build_ms;
  return run;
}

}  // namespace convacc
