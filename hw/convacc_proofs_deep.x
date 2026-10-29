#![feature(explicit_state_access)]
import convacc_proc;
import convacc_proofs;

// Deeper (and slower) system-level proofs. Separate file so the fast gate in
// convacc_proofs.x stays ~40 s while this one -- which needs a 192-step unroll
// instead of 128 -- can be run on demand via `make dslx-prove-deep`.
//
// Bounds come from measured step counts of the minimal 1x1 GEMM descriptor:
//   k = 0 -> 39, k = 8 -> 100..114, k = 16 -> 189, k = 24 -> 222..264
// (k = 24 is 264 at log_r = 0; the prover produced that counterexample when the
// bound was mistakenly set to 224.) k = 24 needs bound 272 and did not close in
// >12 min / 3 GB, so it is deliberately out of reach here.

/// Three kb blocks (k = 16), every reduction width.
#[quickcheck]
fn gemm_fsm_converges_k16_check(log_r: u2) -> bool {
    let (steps, pulses, err, word) = convacc_proofs::run<u32:192>(u32:16, log_r);
    err == u1:0 && pulses == u32:1 && steps > u32:0 && steps <= u32:192
        && word == u64:7
}
