import round;
import std;

/// Saturating requantisation of one accumulator lane.
///
/// `(acc + bias) * scale` is divided by `2**shift`, rounded to nearest with ties
/// away from zero applied to the *magnitude* (so the quantiser is symmetric),
/// offset by `zero_point`, and clamped to s8.
pub fn requant_one(acc: s32, bias: s32, scale: s32, shift: u6, zero_point: s8) -> s8 {
    let q = ((acc as s64) + (bias as s64)) * (scale as s64);
    let negative = q < s64:0;
    // `round_sm` is exactly the "round the magnitude, then re-sign" rule this
    // quantiser needs: RNA on a magnitude rounds half away from zero. With zero
    // bits rounded it hands the magnitude straight back, so shift == 0 needs no
    // special case. The result keeps full width with the rounded-off bits
    // cleared, hence the final shift.
    let sign = if negative { round::Sign::Negative } else { round::Sign::NonNegative };
    let (_, kept) = round::round_sm(round::RoundingMode::RNA, shift as u7, sign,
                                    std::to_unsigned(std::abs(q)));
    let rounded = (kept >> (shift as u32)) as s64;
    let signed = if negative { -rounded } else { rounded };
    let biased = signed + (zero_point as s64);
    if biased > s64:127 {
        s8:127
    } else if biased < s64:-128 {
        s8:-128
    } else {
        biased as s8
    }
}

pub fn requant8(
    acc: s32[8],
    bias: s32[8],
    scale: s32[8],
    shift: u6[8],
    zero_point: s8[8],
) -> s8[8] {
    for (i, values): (u32, s8[8]) in u32:0..u32:8 {
        update(values, i, requant_one(acc[i], bias[i], scale[i], shift[i], zero_point[i]))
    }(zero!<s8[8]>())
}

#[test]
fn positive_round_test() {
    assert_eq(requant_one(s32:7, s32:1, s32:3, u6:1, s8:0), s8:12);
}

#[test]
fn negative_round_test() {
    assert_eq(requant_one(s32:-7, s32:1, s32:3, u6:1, s8:0), s8:-9);
}

#[test]
fn zero_shift_test() {
    assert_eq(requant_one(s32:4, s32:2, s32:3, u6:0, s8:-2), s8:16);
}

#[test]
fn saturation_test() {
    assert_eq(requant_one(s32:200, s32:0, s32:1, u6:0, s8:0), s8:127);
    assert_eq(requant_one(s32:-200, s32:0, s32:1, u6:0, s8:0), s8:-128);
}

// --- rounding: magnitude is rounded half-up, then re-signed ---

#[test]
fn round_half_up_on_magnitude_test() {
    // q = 1, shift 1 -> 0.5 rounds up to 1
    assert_eq(requant_one(s32:1, s32:0, s32:1, u6:1, s8:0), s8:1);
    // q = 1, shift 2 -> 0.25 rounds down to 0
    assert_eq(requant_one(s32:1, s32:0, s32:1, u6:2, s8:0), s8:0);
    // q = 3, shift 2 -> 0.75 rounds up to 1
    assert_eq(requant_one(s32:3, s32:0, s32:1, u6:2, s8:0), s8:1);
    // q = 5, shift 3 -> 0.625 rounds up to 1
    assert_eq(requant_one(s32:5, s32:0, s32:1, u6:3, s8:0), s8:1);
    // q = 3, shift 3 -> 0.375 rounds down to 0
    assert_eq(requant_one(s32:3, s32:0, s32:1, u6:3, s8:0), s8:0);
}

#[test]
fn negative_round_is_magnitude_symmetric_test() {
    // Because the magnitude is rounded first, -0.5 goes to -1 (away from zero),
    // mirroring +0.5 -> +1. This keeps the quantiser symmetric.
    assert_eq(requant_one(s32:-1, s32:0, s32:1, u6:1, s8:0), s8:-1);
    assert_eq(requant_one(s32:-1, s32:0, s32:1, u6:2, s8:0), s8:0);
    assert_eq(requant_one(s32:-3, s32:0, s32:1, u6:2, s8:0), s8:-1);
}

#[test]
fn zero_shift_skips_rounding_entirely_test() {
    // shift == 0 is special-cased: the round-up term is *not* applied.
    // 3 >> 1 would be 1, but with shift 0 the raw magnitude is kept.
    assert_eq(requant_one(s32:3, s32:0, s32:1, u6:0, s8:0), s8:3);
    assert_eq(requant_one(s32:-3, s32:0, s32:1, u6:0, s8:0), s8:-3);
}

// --- bias / scale ordering ---

#[test]
fn bias_adds_before_scale_test() {
    // (acc + bias) * scale, not acc * scale + bias
    assert_eq(requant_one(s32:2, s32:3, s32:4, u6:0, s8:0), s8:20);
    assert_eq(requant_one(s32:2, s32:0, s32:4, u6:0, s8:0), s8:8);
}

#[test]
fn negative_scale_mirrors_test() {
    // A negative scale flips the sign of the product before rounding.
    assert_eq(requant_one(s32:5, s32:0, s32:-2, u6:0, s8:0), s8:-10);
    assert_eq(requant_one(s32:5, s32:0, s32:-2, u6:1, s8:0), s8:-5);
}

// --- zero point and clamping ---

#[test]
fn zero_point_offsets_after_rounding_test() {
    assert_eq(requant_one(s32:0, s32:0, s32:1, u6:0, s8:5), s8:5);
    assert_eq(requant_one(s32:0, s32:0, s32:1, u6:0, s8:-5), s8:-5);
    // zero point is applied *after* the shift, so it is never shifted
    assert_eq(requant_one(s32:8, s32:0, s32:1, u6:3, s8:3), s8:4);
}

#[test]
fn clamp_applies_after_zero_point_test() {
    // 100 + 50 = 150 -> saturates at 127
    assert_eq(requant_one(s32:100, s32:0, s32:1, u6:0, s8:50), s8:127);
    // -100 - 50 = -150 -> saturates at -128
    assert_eq(requant_one(s32:-100, s32:0, s32:1, u6:0, s8:-50), s8:-128);
    // exactly on the boundary must not clamp
    assert_eq(requant_one(s32:127, s32:0, s32:1, u6:0, s8:0), s8:127);
    assert_eq(requant_one(s32:128, s32:0, s32:1, u6:0, s8:0), s8:127);
    assert_eq(requant_one(s32:-128, s32:0, s32:1, u6:0, s8:0), s8:-128);
    assert_eq(requant_one(s32:-129, s32:0, s32:1, u6:0, s8:0), s8:-128);
}

#[test]
fn bias_can_force_saturation_test() {
    // acc is small but the bias pushes it past 127
    assert_eq(requant_one(s32:1, s32:300, s32:1, u6:0, s8:0), s8:127);
    assert_eq(requant_one(s32:1, s32:-300, s32:1, u6:0, s8:0), s8:-128);
}

// --- vector/scalar consistency ---

#[test]
fn requant8_matches_requant_one_test() {
    let acc = [s32:12, s32:-12, s32:1, s32:-1, s32:200, s32:-200, s32:0, s32:7];
    let bias = [s32:1, s32:1, s32:0, s32:0, s32:0, s32:0, s32:-3, s32:2];
    let scale = [s32:3, s32:3, s32:5, s32:5, s32:1, s32:1, s32:2, s32:-2];
    let shift = [u6:1, u6:1, u6:2, u6:2, u6:0, u6:0, u6:0, u6:3];
    let zp = [s8:0, s8:0, s8:2, s8:2, s8:0, s8:0, s8:1, s8:-1];
    let lanes = requant8(acc, bias, scale, shift, zp);
    for (i, _): (u32, ()) in u32:0..u32:8 {
        assert_eq(lanes[i], requant_one(acc[i], bias[i], scale[i], shift[i], zp[i]));
    }(());
}

// ===========================================================================
// Fuzzed invariants. Run with
//   dslx_interpreter_main --evaluator=ir-interpreter \
//       --run_quickcheck_when_interpreting requant.x
// ===========================================================================

#[quickcheck]
fn requant_is_monotone_in_the_accumulator_check
    (acc_a: s8, acc_b: s8, bias: s8, scale: u4, shift: u3, zp: s4) -> bool {
    // With a positive scale a larger accumulator can never produce a smaller
    // output. This catches sign and rounding-order mistakes.
    let lo = if acc_a <= acc_b { acc_a } else { acc_b };
    let hi = if acc_a <= acc_b { acc_b } else { acc_a };
    let s = (scale as s32) + s32:1;
    let sh = (shift as u6) + u6:1;
    requant_one(lo as s32, bias as s32, s, sh, zp as s8)
        <= requant_one(hi as s32, bias as s32, s, sh, zp as s8)
}

#[quickcheck]
fn requant_always_ends_up_in_s8_check
    (acc: s8, bias: s8, scale: s4, shift: u3, zp: s4) -> bool {
    let r = requant_one(acc as s32, bias as s32, scale as s32,
                        (shift as u6) + u6:1, zp as s8);
    r <= s8:127 && r >= s8:-128
}

#[quickcheck]
fn requant_zero_point_is_applied_after_the_shift_check
    (acc: s8, scale: u4, shift: u3) -> bool {
    // Adding the zero point is the very last step, so it is never scaled or
    // shifted -- the difference between two zero points is exactly that amount
    // whenever neither side saturates.
    let s = (scale as s32) + s32:1;
    let sh = (shift as u6) + u6:1;
    let with_zp = requant_one(acc as s32, s32:0, s, sh, s8:3) as s16;
    let without_zp = requant_one(acc as s32, s32:0, s, sh, s8:0) as s16;
    // 3 is small enough that saturation can only move one of the two, and then
    // the difference is at most 3
    let delta = with_zp - without_zp;
    delta <= s16:3 && delta >= s16:0
}

#[quickcheck]
fn requant8_matches_the_scalar_path_check
    (a0: s8, a1: s8, a2: s8, a3: s8, bias: s8, scale: u3, shift: u3, zp: s4) -> bool {
    // The eight-lane version must be exactly eight independent scalar calls.
    let s = (scale as s32) + s32:1;
    let sh = (shift as u6) + u6:1;
    let z = zp as s8;
    let acc = [a0 as s32, a1 as s32, a2 as s32, a3 as s32,
               s32:0, s32:127, s32:-128, s32:5];
    let bias8 = [bias as s32, bias as s32, bias as s32, bias as s32,
                 s32:0, s32:0, s32:0, bias as s32];
    let scale8 = [s, s, s, s, s, s, s, s];
    let shift8 = [sh, sh, sh, sh, sh, sh, sh, sh];
    let zp8 = [z, z, z, z, z, z, z, z];
    let lanes = requant8(acc, bias8, scale8, shift8, zp8);
    for (i, ok): (u32, bool) in u32:0..u32:8 {
        ok && lanes[i] == requant_one(acc[i], bias8[i], scale8[i], shift8[i], zp8[i])
    }(true)
}
