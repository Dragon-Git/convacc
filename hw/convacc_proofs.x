#![feature(explicit_state_access)]
import convacc_proc;

// Properties that are *proved* rather than sampled.
//
// `dslx-fuzz` runs each `#[quickcheck]` on 1000 random inputs; here the same
// kind of statement is handed to `prove_quickcheck_main`, which asks the SMT
// solver whether ANY input can falsify it (and prints a counterexample if so).
//
// Kept out of DSLX_UNIT_TESTS on purpose: the interpreter's quickcheck path
// re-runs each property through the IR JIT, which aborts on engine-sized values
// ("Null alloc with 1024"). The prover does not use the JIT.
//
// The step bound is not cosmetic: DSLX unrolls `for` into IR, so it is literally
// how many engine steps the solver reasons about, and the cost grows steeply
// with it. That is why the bounds are split by depth:
//
//   k in {0, 8}   bound 128    39..114 steps   ~43 s   (this file)
//   k = 16        bound 192      189 steps    ~140 s   (convacc_proofs_deep.x)
//   k = 24        bound 272      264 steps    > 12 min at 3 GB, not practical
//
// The k = 24 measurement is worth keeping: it is 264 steps at log_r = 0, not the
// 222 the log_r = 3 case suggests -- the prover found that with a counterexample
// when the bound was set to 224.

fn lohi(lo: u32, hi: u32) -> u64 { (lo as u64) | ((hi as u64) << u32:32) }
fn h2(a: u32, b: u32) -> u64 { (a as u64) | ((b as u64) << u32:16) }
fn h4(a: u32, b: u32, c: u32, d: u32) -> u64 {
    (a as u64) | ((b as u64) << u32:16) | ((c as u64) << u32:32) | ((d as u64) << u32:48)
}

/// One-row, one-column GEMM descriptor `k` deep with reduction width `log_r`.
/// Every operand memory is zero, so the only correct output is requant(0), and
/// since the descriptor declares a single output column exactly lane 0 is a real
/// channel: the word must be 7 and the seven padding lanes must stay zero.
pub fn image(k: u32, log_r: u2) -> u64[32] {
    let desc = update(update(update(update(update(update(update(update(update(update(
        zero!<u64[32]>(),
        u32:0, u64:1 | ((log_r as u64) << u32:3)),   // flags: gemm + log_r
        u32:1, lohi(u32:1, u32:1)),                  // m = 1, n = 1
        u32:2, k as u64),                            // k
        u32:3, h2(u32:1, u32:1)),                    // mh = 1, mw = 1
        u32:5, h4(u32:1, u32:0, u32:0, u32:1)),
        u32:7, h2(u32:1, u32:1)),                    // ho = 1, wo = 1
        u32:11, h2(u32:1, u32:1)),                   // steph = 1, stepw = 1
        u32:12, lohi(u32:0, u32:0)),                 // act_base = 0, out_base = 0
        u32:13, lohi(u32:0, u32:16)),                // w_base = 0, p_base = 16
        u32:14, u64:0);                              // tap_base = 0, taps = 0
    let scaled = for (i, w): (u32, u64[32]) in u32:20..u32:24 {
        update(w, i, lohi(u32:1, u32:1))             // scale = 1 on every lane
    }(desc);
    update(scaled, u32:25, u64:0x0707070707070707)   // zero point 7
}

/// Run to completion within `N` steps. `N` is a parametric so each call site
/// gets its own unrolled copy -- the solver's cost is set by this number.
/// Returns (steps taken, number of done pulses, err flag, output word 0).
pub fn run<N: u32>(k: u32, log_r: u2) -> (u32, u32, u1, u64) {
    let st0 = convacc_proc::Engine<u32:32> {
        wgt: image(k, log_r), desc_start: u32:0, ..convacc_proc::zero_engine<u32:32>()
    };
    let (st, steps, pulses, _) = for (i, (st, steps, pulses, done)):
            (u32, (convacc_proc::Engine<u32:32>, u32, u32, bool)) in u32:0..N {
        if done { (st, steps, pulses, done) } else {
            let (nst, ndone) = convacc_proc::engine_step(st, i == u32:0, u32:0);
            let bump = if ndone { u32:1 } else { u32:0 };
            (nst, steps + u32:1, pulses + bump, ndone)
        }
    }((st0, u32:0, u32:0, false));
    (steps, pulses, st.err, st.out_mem[u32:0])
}

/// For every reduction depth in {0, 8} and every reduction width: the FSM
/// terminates, never trips the `err` invariant, pulses `done` exactly once, and
/// emits requant(0).
#[quickcheck]
fn gemm_fsm_converges_for_every_k_and_width_check(k_blocks: u1, log_r: u2) -> bool {
    let k = (k_blocks as u32) * u32:8;
    let (steps, pulses, err, word) = run<u32:128>(k, log_r);
    err == u1:0 && pulses == u32:1 && steps > u32:0 && steps <= u32:128
        && word == u64:7
}
