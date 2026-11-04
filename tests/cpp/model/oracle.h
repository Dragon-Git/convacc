// Invariants every generated case must satisfy.
//
// This is the oracle half of the coverage-guided fuzzer. It is deliberately
// *structural* (descriptor chain, image layout, packed-vs-semantic equality)
// rather than a re-implementation of the reference model: anything that needs a
// second opinion is compared against an independent code path that already
// exists, so a failure here is a real disagreement and not a duplicated bug.
#pragma once

#include "convacc/reference.h"
#include "model/case_codec.h"

namespace convacc {

// Throws std::runtime_error with a diagnostic when any invariant fails.
void verify_case(const FuzzCase& c, const BuiltCase& built);

}  // namespace convacc
