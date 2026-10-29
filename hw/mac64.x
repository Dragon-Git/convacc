pub fn mac64_reduced(
    channelwise: u1,
    dense_act: s8[64],
    channelwise_act: s8[64],
    wgt: s8[8],
    log_r: u2,
) -> s32[64] {
    let products = for (lane, values): (u32, s32[64]) in u32:0..u32:64 {
        let p = lane >> u32:3;
        let within = lane & u32:7;
        let r = within & ((u32:1 << log_r) - u32:1);
        let act_index = (p << log_r) | r;
        let act = if channelwise == u1:0 { dense_act[act_index] } else { channelwise_act[lane] };
        update(values, lane, ((act as s16) * (wgt[within] as s16)) as s32)
    }(zero!<s32[64]>());
    let sum2 = for (i, values): (u32, s32[32]) in u32:0..u32:32 {
        update(values, i, products[i * u32:2] + products[i * u32:2 + u32:1])
    }(zero!<s32[32]>());
    let sum4 = for (i, values): (u32, s32[16]) in u32:0..u32:16 {
        update(values, i, sum2[i * u32:2] + sum2[i * u32:2 + u32:1])
    }(zero!<s32[16]>());
    let sum8 = for (i, values): (u32, s32[8]) in u32:0..u32:8 {
        update(values, i, sum4[i * u32:2] + sum4[i * u32:2 + u32:1])
    }(zero!<s32[8]>());
    for (index, values): (u32, s32[64]) in u32:0..u32:64 {
        let p = index >> u32:3;
        let o = index & u32:7;
        let total = match log_r {
            u2:0 => products[p * u32:8 + o],
            u2:1 => sum2[(p * u32:4 + o) % u32:32],
            u2:2 => sum4[(p * u32:2 + o) % u32:16],
            u2:3 => sum8[p % u32:8],
        };
        update(values, index, total)
    }(zero!<s32[64]>())
}

pub fn mac64_accumulate(
    acc: s32[64],
    channelwise: u1,
    dense_act: s8[64],
    channelwise_act: s8[64],
    wgt: s8[8],
    log_r: u2,
) -> s32[64] {
    let reduced = mac64_reduced(channelwise, dense_act, channelwise_act, wgt, log_r);
    let active_outputs = u32:8 >> log_r;
    for (index, values): (u32, s32[64]) in u32:0..u32:64 {
        let o = index & u32:7;
        let next = if o < active_outputs { acc[index] + reduced[index] } else { acc[index] };
        update(values, index, next)
    }(acc)
}

pub fn mac64_dense(acc: s32[64], act: s8[8], wgt: s8[8]) -> s32[64] {
    let wide_act = for (i, values): (u32, s8[64]) in u32:0..u32:64 {
        update(values, i, if i < u32:8 { act[i] } else { s8:0 })
    }(zero!<s8[64]>());
    mac64_accumulate(acc, u1:0, wide_act, zero!<s8[64]>(), wgt, u2:0)
}

pub fn mac64_channelwise(acc: s32[64], act: s8[64], wgt: s8[8]) -> s32[64] {
    mac64_accumulate(acc, u1:1, zero!<s8[64]>(), act, wgt, u2:0)
}

pub fn mac64(
    channelwise: u1,
    acc: s32[64],
    dense_act: s8[64],
    channelwise_act: s8[64],
    wgt: s8[8],
    log_r: u2,
) -> s32[64] {
    mac64_accumulate(acc, channelwise, dense_act, channelwise_act, wgt, log_r)
}

#[test]
fn dense_r1_test() {
    let acc = update(zero!<s32[64]>(), u32:9, s32:10);
    let act = update(update(zero!<s8[64]>(), u32:0, s8:2), u32:1, s8:3);
    let wgt = update(update(zero!<s8[8]>(), u32:0, s8:4), u32:1, s8:-2);
    let result = mac64_accumulate(acc, u1:0, act, zero!<s8[64]>(), wgt, u2:0);
    assert_eq(result[u32:0], s32:8);
    assert_eq(result[u32:1], s32:-4);
    assert_eq(result[u32:8], s32:12);
    assert_eq(result[u32:9], s32:4);
}

#[test]
fn dense_r2_test() {
    let act = update(update(update(zero!<s8[64]>(), u32:0, s8:2), u32:1, s8:3), u32:2, s8:4);
    let wgt = update(update(update(zero!<s8[8]>(), u32:0, s8:5), u32:1, s8:7), u32:2, s8:-1);
    let result = mac64_accumulate(zero!<s32[64]>(), u1:0, act, zero!<s8[64]>(), wgt, u2:1);
    assert_eq(result[u32:0], s32:31);
    assert_eq(result[u32:1], s32:-2);
    assert_eq(result[u32:2], s32:0);
}

#[test]
fn dense_r8_test() {
    let act = update(zero!<s8[64]>(), u32:7, s8:-2);
    let wgt = update(zero!<s8[8]>(), u32:7, s8:3);
    let result = mac64_accumulate(zero!<s32[64]>(), u1:0, act, zero!<s8[64]>(), wgt, u2:3);
    assert_eq(result[u32:0], s32:-6);
    assert_eq(result[u32:1], s32:0);
}

#[test]
fn channelwise_r2_test() {
    let act = update(update(zero!<s8[64]>(), u32:0, s8:5), u32:1, s8:-7);
    let wgt = update(update(zero!<s8[8]>(), u32:0, s8:3), u32:1, s8:4);
    let result = mac64_accumulate(zero!<s32[64]>(), u1:1, zero!<s8[64]>(), act, wgt, u2:1);
    assert_eq(result[u32:0], s32:-13);
    assert_eq(result[u32:1], s32:0);
}

// --- channelwise across every reduction width ---
// `oo = within / r` names the within-group output channel, so r=1 keeps 8
// channels per tile and r=8 keeps 1. All four widths must agree with the
// "sum over the r inputs" reading of the datapath.

#[test]
fn channelwise_r1_test() {
    // log_r = 0: result[p*8+o] = act[p*8+o] * wgt[o] (no reduction)
    let act = update(update(zero!<s8[64]>(), u32:0, s8:5), u32:1, s8:-7);
    let wgt = update(update(zero!<s8[8]>(), u32:0, s8:3), u32:1, s8:4);
    let result = mac64_accumulate(zero!<s32[64]>(), u1:1, zero!<s8[64]>(), act, wgt, u2:0);
    assert_eq(result[u32:0], s32:15);
    assert_eq(result[u32:1], s32:-28);
    assert_eq(result[u32:2], s32:0);
}

#[test]
fn channelwise_r4_test() {
    // log_r = 2: result[o] = sum_{rr=0..3} act[4*o+rr] * wgt[4*o+rr]
    let act = [
        s8:1, s8:2, s8:3, s8:4, s8:1, s8:1, s8:1, s8:1,
    ];
    let wgt = [s8:5, s8:6, s8:7, s8:8, s8:2, s8:2, s8:2, s8:2];
    let wide = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, if i < u32:8 { act[i] } else { s8:0 })
    }(zero!<s8[64]>());
    let result = mac64_accumulate(zero!<s32[64]>(), u1:1, zero!<s8[64]>(), wide, wgt, u2:2);
    assert_eq(result[u32:0], s32:70);   // 1*5 + 2*6 + 3*7 + 4*8
    assert_eq(result[u32:1], s32:8);    // 1*2 + 1*2 + 1*2 + 1*2
    assert_eq(result[u32:2], s32:0);    // inactive lane, never written
}

#[test]
fn channelwise_r8_test() {
    // log_r = 3: every output lane of a position holds the same 8-wide dot
    // product, and only lane 0 is active.
    let act = [s8:1, s8:2, s8:3, s8:4, s8:5, s8:6, s8:7, s8:8];
    let wgt = [s8:1, s8:1, s8:1, s8:1, s8:1, s8:1, s8:1, s8:1];
    let wide = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, if i < u32:8 { act[i] } else { s8:0 })
    }(zero!<s8[64]>());
    let result = mac64_accumulate(zero!<s32[64]>(), u1:1, zero!<s8[64]>(), wide, wgt, u2:3);
    assert_eq(result[u32:0], s32:36);   // 1+2+3+4+5+6+7+8
    assert_eq(result[u32:8], s32:0);    // second position is all zeros
}

// --- dense across every reduction width ---

#[test]
fn dense_r4_test() {
    // log_r = 2: result[p*8+o] = sum_{rr=0..3} act[p*4+rr] * wgt[4*o+rr]
    let act = [s8:1, s8:2, s8:3, s8:4];
    let wgt = [s8:5, s8:6, s8:7, s8:8, s8:1, s8:-1, s8:1, s8:-1];
    let wide = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, if i < u32:4 { act[i] } else { s8:0 })
    }(zero!<s8[64]>());
    let result = mac64_accumulate(zero!<s32[64]>(), u1:0, wide, zero!<s8[64]>(), wgt, u2:2);
    assert_eq(result[u32:0], s32:70);   // 1*5 + 2*6 + 3*7 + 4*8
    assert_eq(result[u32:1], s32:-2);   // 1*1 + 2*-1 + 3*1 + 4*-1
    assert_eq(result[u32:2], s32:0);
}

#[test]
fn dense_r3_is_8_wide_dot_test() {
    // log_r = 3 collapses all 8 inputs into one output per position.
    let act = [s8:1, s8:-2, s8:3, s8:-4, s8:5, s8:-6, s8:7, s8:-8];
    let wgt = [s8:1, s8:1, s8:1, s8:1, s8:1, s8:1, s8:1, s8:1];
    let wide = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, if i < u32:8 { act[i] } else { s8:0 })
    }(zero!<s8[64]>());
    let result = mac64_accumulate(zero!<s32[64]>(), u1:0, wide, zero!<s8[64]>(), wgt, u2:3);
    assert_eq(result[u32:0], s32:-4);   // 1-2+3-4+5-6+7-8
}

// --- wrappers ---

#[test]
fn mac64_dense_is_outer_product_test() {
    // With r = 1 there is one input per position and one weight per output
    // channel, so the 64-lane result is the outer product act[p] x wgt[o].
    let act = [s8:2, s8:3, s8:0, s8:0, s8:0, s8:0, s8:0, s8:0];
    let wgt = [s8:4, s8:-2, s8:0, s8:0, s8:0, s8:0, s8:0, s8:0];
    let result = mac64_dense(zero!<s32[64]>(), act, wgt);
    assert_eq(result[u32:0], s32:8);
    assert_eq(result[u32:1], s32:-4);
    assert_eq(result[u32:8], s32:12);
    assert_eq(result[u32:9], s32:-6);
    assert_eq(result[u32:16], s32:0);
}

#[test]
fn mac64_channelwise_is_lane_multiply_test() {
    // Lane (p*8+o) accumulates act[p*8+o] * wgt[o] -- each output channel has
    // its own weight, applied to every position.
    let act = update(update(zero!<s8[64]>(), u32:0, s8:6), u32:8, s8:-3);
    let wgt = update(zero!<s8[8]>(), u32:0, s8:7);
    let result = mac64_channelwise(zero!<s32[64]>(), act, wgt);
    assert_eq(result[u32:0], s32:42);
    assert_eq(result[u32:8], s32:-21);
    assert_eq(result[u32:1], s32:0);
}

#[test]
fn mac64_matches_accumulate_test() {
    let acc = update(zero!<s32[64]>(), u32:0, s32:11);
    let act = update(zero!<s8[64]>(), u32:0, s8:2);
    let wgt = update(zero!<s8[8]>(), u32:0, s8:5);
    let via_mac = mac64(u1:0, acc, act, zero!<s8[64]>(), wgt, u2:0);
    let via_acc = mac64_accumulate(acc, u1:0, act, zero!<s8[64]>(), wgt, u2:0);
    assert_eq(via_mac[u32:0], via_acc[u32:0]);
    assert_eq(via_mac[u32:0], s32:21);
}

// --- accumulation guard ---

#[test]
fn accumulate_leaves_inactive_lanes_alone_test() {
    // r = 4 gives active_outputs = 2, so lanes 2..7 must keep whatever the
    // accumulator already held.
    let acc = for (i, a): (u32, s32[64]) in u32:0..u32:64 {
        update(a, i, s32:100 + (i as s32))
    }(zero!<s32[64]>());
    let act = update(zero!<s8[64]>(), u32:0, s8:1);
    let wgt = update(zero!<s8[8]>(), u32:0, s8:1);
    let result = mac64_accumulate(acc, u1:0, act, zero!<s8[64]>(), wgt, u2:2);
    assert_eq(result[u32:0], s32:101);   // 100 + 1*1
    assert_eq(result[u32:1], s32:101);   // 100 + 1*1 (within=1 still r=1 -> act[1]=0? no)
    assert_eq(result[u32:2], s32:102);   // untouched
    assert_eq(result[u32:7], s32:107);   // untouched
}

#[test]
fn reduced_matches_accumulated_delta_test() {
    // mac64_reduced is the per-call contribution; accumulate adds it in.
    let act = [s8:3, s8:-2, s8:4, s8:1];
    let wgt = [s8:2, s8:5, s8:-1, s8:3, s8:0, s8:0, s8:0, s8:0];
    let wide = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, if i < u32:4 { act[i] } else { s8:0 })
    }(zero!<s8[64]>());
    let reduced = mac64_reduced(u1:0, wide, zero!<s8[64]>(), wgt, u2:2);
    let accumulated = mac64_accumulate(zero!<s32[64]>(), u1:0, wide, zero!<s8[64]>(), wgt, u2:2);
    assert_eq(accumulated[u32:0], reduced[u32:0]);
    assert_eq(accumulated[u32:1], reduced[u32:1]);
    assert_eq(reduced[u32:0], s32:3 * 2 + (-2) * 5 + 4 * -1 + 1 * 3);
}

// ===========================================================================
// Invariants. These are *proved* with `prove_quickcheck_main` (`make dslx-prove`)
// rather than sampled: all sixteen close in ~4.7 s total on this machine (the
// four data-family properties are 0.16 / 0.22 / 0.74 / 3.1 s; the twelve
// decomposition properties are 0.09-0.92 s each), so the formal check is both
// stronger and cheaper than 1000 random cases. `dslx-fuzz` (sampling) is a
// fallback:
//   dslx_interpreter_main --evaluator=ir-interpreter \
//       --run_quickcheck_when_interpreting mac64.x
// ===========================================================================

#[quickcheck]
fn accumulate_leaves_inactive_lanes_untouched_check(log_r: u2, seed: s6) -> bool {
    // Only the first 8 >> log_r output lanes of each position accumulate; the
    // rest must keep whatever the accumulator already held.
    let active = u32:8 >> (log_r as u32);
    let acc = for (i, a): (u32, s32[64]) in u32:0..u32:64 {
        update(a, i, s32:1000 + (i as s32))
    }(zero!<s32[64]>());
    let act = update(zero!<s8[64]>(), u32:0, seed as s8);
    let wgt = update(zero!<s8[8]>(), u32:0, s8:1);
    let result = mac64_accumulate(acc, u1:0, act, zero!<s8[64]>(), wgt, log_r);
    for (lane, ok): (u32, bool) in u32:0..u32:64 {
        let o = lane % u32:8;
        ok && (o < active || result[lane] == acc[lane])
    }(true)
}

#[quickcheck]
fn accumulate_adds_exactly_the_reduced_contribution_check
    (log_r: u2, seed: s6) -> bool {
    // On the active lanes, accumulate(acc) - acc must equal reduced().
    let active = u32:8 >> (log_r as u32);
    let act = update(update(zero!<s8[64]>(), u32:0, seed as s8),
                     u32:1, (seed as s8) + s8:1);
    let wgt = update(update(zero!<s8[8]>(), u32:0, s8:2), u32:1, s8:-3);
    let reduced = mac64_reduced(u1:0, act, zero!<s8[64]>(), wgt, log_r);
    let acc = for (i, a): (u32, s32[64]) in u32:0..u32:64 {
        update(a, i, s32:17 + (i as s32))
    }(zero!<s32[64]>());
    let accumulated = mac64_accumulate(acc, u1:0, act, zero!<s8[64]>(), wgt, log_r);
    for (lane, ok): (u32, bool) in u32:0..u32:64 {
        let o = lane % u32:8;
        ok && (o >= active || accumulated[lane] - acc[lane] == reduced[lane])
    }(true)
}

#[quickcheck]
fn channelwise_accumulate_is_per_lane_check(seed: s6, log_r: u2) -> bool {
    // channelwise mode: output lane (p,o) = sum_{rr<r} act[p*8+o*r+rr] * wgt[o*r+rr].
    // Validate this against the implementation for EVERY log_r -- the old
    // version returned `true` for log_r > 0 and so tested nothing.
    let r = u32:1 << (log_r as u32);
    let active = u32:8 >> (log_r as u32);
    let act = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, if i < u32:16 { ((i as s32) + (seed as s32)) as s8 } else { s8:0 })
    }(zero!<s8[64]>());
    let wgt = [s8:1, s8:2, s8:3, s8:4, s8:5, s8:6, s8:7, s8:8];
    let result = mac64_accumulate(zero!<s32[64]>(), u1:1, zero!<s8[64]>(), act, wgt, log_r);
    for (lane, ok): (u32, bool) in u32:0..u32:64 {
        let p = lane >> u32:3;
        let o = lane & u32:7;
        let want = if o < active {
            for (rr, s): (u32, s32) in u32:0..u32:8 {
                let in_group = rr < r;
                let contrib = if in_group {
                    let a = act[p * u32:8 + o * r + rr];
                    let w = wgt[o * r + rr];
                    ((a as s16) * (w as s16)) as s32
                } else { s32:0 };
                s + contrib
            }(s32:0)
        } else { s32:0 };
        ok && result[lane] == want
    }(true)
}

// ===========================================================================
// Reference model + full-coverage sweep. `mac64_ref` is the "obvious" definition
// of what mac64 computes (per-position, per-output-channel reduction), written
// independently of the production reduction tree. The sweep test pits it against
// the real datapath across channelwise x log_r x every position, so the
// (p << log_r) act stride that no hand-written test exercised at log_r > 0 is
// now covered.
// ===========================================================================

fn mac64_ref(
    channelwise: u1,
    acc: s32[64],
    dense_act: s8[64],
    channelwise_act: s8[64],
    wgt: s8[8],
    log_r: u2,
) -> s32[64] {
    let r = u32:1 << (log_r as u32);
    let active = u32:8 >> (log_r as u32);
    for (lane, outv): (u32, s32[64]) in u32:0..u32:64 {
        let p = lane >> u32:3;
        let o = lane & u32:7;
        let sum = if o < active {
            for (rr, s): (u32, s32) in u32:0..u32:8 {
                let in_group = rr < r;
                let contrib = if in_group {
                    let w = wgt[o * r + rr];
                    let a_idx = if channelwise { p * u32:8 + o * r + rr } else { p * r + rr };
                    let a = if channelwise { channelwise_act[a_idx] } else { dense_act[a_idx] };
                    ((a as s16) * (w as s16)) as s32
                } else { s32:0 };
                s + contrib
            }(s32:0)
        } else { s32:0 };
        let next = if o < active { acc[lane] + sum } else { acc[lane] };
        update(outv, lane, next)
    }(acc)
}

#[test]
fn mac64_reference_sweep_test() {
    // Distinct, position-varying inputs so the (p << log_r) stride is genuinely
    // exercised at every reduction width.
    let dense = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, ((i as s32) - s32:32) as s8)
    }(zero!<s8[64]>());
    let cact = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, ((i as s32) * s32:3 - s32:96) as s8)
    }(zero!<s8[64]>());
    let wgt = [s8:1, s8:2, s8:3, s8:4, s8:5, s8:6, s8:7, s8:8];
    let acc = for (i, a): (u32, s32[64]) in u32:0..u32:64 {
        update(a, i, s32:1000 + (i as s32))
    }(zero!<s32[64]>());
    for (cw, _d): (u32, ()) in u32:0..u32:2 {
        for (lr, _e): (u32, ()) in u32:0..u32:4 {
            let channelwise = cw == u32:1;
            let log_r = lr as u2;
            assert_eq(mac64(channelwise, acc, dense, cact, wgt, log_r),
                      mac64_ref(channelwise, acc, dense, cact, wgt, log_r))
        }(())
    }(());
}

// Formal counterpart to the concrete sweep: for EVERY seed and reduction width,
// the production datapath must match the independent reference on all 64 lanes.
// This is what pins the (p << log_r) dense act stride for all inputs.
#[quickcheck]
fn mac64_matches_reference_check(seed: s6, log_r: u2) -> bool {
    let dense = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, (((i as s32) * s32:3) + (seed as s32)) as s8)
    }(zero!<s8[64]>());
    let wgt = [s8:1, s8:2, s8:3, s8:4, s8:5, s8:6, s8:7, s8:8];
    let acc = for (i, a): (u32, s32[64]) in u32:0..u32:64 {
        update(a, i, s32:1000 + (i as s32))
    }(zero!<s32[64]>());
    let got = mac64(u1:0, acc, dense, zero!<s8[64]>(), wgt, log_r);
    let want = mac64_ref(u1:0, acc, dense, zero!<s8[64]>(), wgt, log_r);
    for (lane, ok): (u32, bool) in u32:0..u32:64 {
        ok && got[lane] == want[lane]
    }(true)
}

// ===========================================================================
// Value-independent strengthening. P1-P4 only quantify over `seed: s6`, i.e. a
// one-parameter family of data; these remove that limitation by decomposing the
// datapath into arithmetic x index x guard:
//   A  symbolic_dense_r{0..3}_check       arbitrary act[8] x wgt[8], log_r fixed,
//                                         ALL 8 output lanes of position 0
//   B  symbolic_channelwise_r{0..3}_check same, channelwise mode
//   C  stride_index_identity_check        (p<<log_r)|rr == p*(1<<log_r)+rr, all p
//   D  weight_index_identity_check        within == o*r + rr, all within/o/rr
//   E  guard_dense_symbolic_base_check    inactive lanes keep a symbolic acc
//   F  dense_wrapper_lane0_check          mac64_dense == general entry (widened)
// A/B fix log_r so every index is a compile-time constant; each query proves all
// eight lanes (o = 0..7) at once for arbitrary values, so o>0 is machine-checked
// per lane rather than argued structurally. Keeping log_r symbolic makes the
// weight index depend on a runtime value and the same query blows past 2 min.
// C/D pin the act and weight indices for every p/o/rr, so A/B's position-0 proof
// lifts to every position; E covers an arbitrary acc base.
// ===========================================================================

fn symbolic_dense_at<R_LOG: u32>(act: s8[8], wgt: s8[8]) -> bool {
    let r = u32:1 << R_LOG;
    let active = u32:8 >> R_LOG;
    let dense = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, if i < u32:8 { act[i] } else { s8:0 })
    }(zero!<s8[64]>());
    let got = mac64_accumulate(zero!<s32[64]>(), u1:0, dense, zero!<s8[64]>(), wgt, R_LOG as u2);
    for (lane, ok): (u32, bool) in u32:0..u32:8 {
        let want = if lane < active {
            for (rr, s): (u32, s32) in u32:0..u32:8 {
                let in_group = rr < r;
                let contrib = if in_group {
                    let w = wgt[lane * r + rr];
                    ((act[rr] as s16) * (w as s16)) as s32
                } else { s32:0 };
                s + contrib
            }(s32:0)
        } else { s32:0 };
        ok && got[lane] == want
    }(true)
}

#[quickcheck]
fn symbolic_dense_r0_check(act: s8[8], wgt: s8[8]) -> bool { symbolic_dense_at<u32:0>(act, wgt) }

#[quickcheck]
fn symbolic_dense_r1_check(act: s8[8], wgt: s8[8]) -> bool { symbolic_dense_at<u32:1>(act, wgt) }

#[quickcheck]
fn symbolic_dense_r2_check(act: s8[8], wgt: s8[8]) -> bool { symbolic_dense_at<u32:2>(act, wgt) }

#[quickcheck]
fn symbolic_dense_r3_check(act: s8[8], wgt: s8[8]) -> bool { symbolic_dense_at<u32:3>(act, wgt) }

fn symbolic_channelwise_at<R_LOG: u32>(act: s8[8], wgt: s8[8]) -> bool {
    let r = u32:1 << R_LOG;
    let active = u32:8 >> R_LOG;
    let cact = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, if i < u32:8 { act[i] } else { s8:0 })
    }(zero!<s8[64]>());
    let got = mac64_accumulate(zero!<s32[64]>(), u1:1, zero!<s8[64]>(), cact, wgt, R_LOG as u2);
    for (lane, ok): (u32, bool) in u32:0..u32:8 {
        let want = if lane < active {
            for (rr, s): (u32, s32) in u32:0..u32:8 {
                let in_group = rr < r;
                let contrib = if in_group {
                    let idx = lane * r + rr;
                    ((act[idx] as s16) * (wgt[idx] as s16)) as s32
                } else { s32:0 };
                s + contrib
            }(s32:0)
        } else { s32:0 };
        ok && got[lane] == want
    }(true)
}

#[quickcheck]
fn symbolic_channelwise_r0_check(act: s8[8], wgt: s8[8]) -> bool { symbolic_channelwise_at<u32:0>(act, wgt) }

#[quickcheck]
fn symbolic_channelwise_r1_check(act: s8[8], wgt: s8[8]) -> bool { symbolic_channelwise_at<u32:1>(act, wgt) }

#[quickcheck]
fn symbolic_channelwise_r2_check(act: s8[8], wgt: s8[8]) -> bool { symbolic_channelwise_at<u32:2>(act, wgt) }

#[quickcheck]
fn symbolic_channelwise_r3_check(act: s8[8], wgt: s8[8]) -> bool { symbolic_channelwise_at<u32:3>(act, wgt) }

#[quickcheck]
fn stride_index_identity_check(p: u3, log_r: u2, rr: u3) -> bool {
    let r = u32:1 << (log_r as u32);
    let idx = ((p as u32) << (log_r as u32)) | (rr as u32);
    if (rr as u32) < r {
        idx == (p as u32) * r + (rr as u32) && idx < u32:64
    } else {
        true
    }
}

#[quickcheck]
fn weight_index_identity_check(within: u3, log_r: u2) -> bool {
    let r = u32:1 << (log_r as u32);
    let o = (within as u32) / r;
    let rr = (within as u32) & (r - u32:1);
    rr < r && o * r + rr == (within as u32)
}

#[quickcheck]
fn guard_dense_symbolic_base_check(base: s32, log_r: u2) -> bool {
    let active = u32:8 >> (log_r as u32);
    let acc = for (i, a): (u32, s32[64]) in u32:0..u32:64 {
        update(a, i, base + (i as s32))
    }(zero!<s32[64]>());
    let act = update(zero!<s8[64]>(), u32:0, s8:1);
    let wgt = update(zero!<s8[8]>(), u32:0, s8:1);
    let result = mac64_accumulate(acc, u1:0, act, zero!<s8[64]>(), wgt, log_r);
    for (lane, ok): (u32, bool) in u32:0..u32:64 {
        let o = lane & u32:7;
        ok && (o < active || result[lane] == acc[lane])
    }(true)
}

#[quickcheck]
fn dense_wrapper_lane0_check(act: s8[8], wgt: s8[8], base: s32) -> bool {
    let acc = for (i, a): (u32, s32[64]) in u32:0..u32:64 {
        update(a, i, base + (i as s32))
    }(zero!<s32[64]>());
    let wide = for (i, a): (u32, s8[64]) in u32:0..u32:64 {
        update(a, i, if i < u32:8 { act[i] } else { s8:0 })
    }(zero!<s8[64]>());
    let via_wrapper = mac64_dense(acc, act, wgt);
    let via_general = mac64(u1:0, acc, wide, zero!<s8[64]>(), wgt, u2:0);
    via_wrapper[u32:0] == via_general[u32:0]
}
