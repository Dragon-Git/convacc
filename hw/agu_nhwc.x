pub fn nhwc_offset(n: u32, h: u32, w: u32, c: u32, height: u32, width: u32, channels: u32) -> u32 {
    (((n * height + h) * width + w) * channels) + c
}

pub fn conv_input_coord(output: u32, kernel: u32, stride: u32, dilation: u32, pad: u32) -> s32 {
    (output as s32) * (stride as s32) + (kernel as s32) * (dilation as s32) - (pad as s32)
}

pub fn coord_in_bounds(coord: s32, limit: u32) -> bool { coord >= s32:0 && coord < (limit as s32) }

pub fn conv_nhwc_address
    (n: u32, oh: u32, ow: u32, kh: u32, kw: u32, c: u32, hi: u32, wi: u32, ci: u32, sh: u32,
     sw: u32, dh: u32, dw: u32, ph: u32, pw: u32) -> (u32, bool) {
    let ih = conv_input_coord(oh, kh, sh, dh, ph);
    let iw = conv_input_coord(ow, kw, sw, dw, pw);
    let valid = coord_in_bounds(ih, hi) && coord_in_bounds(iw, wi) && c < ci;
    let safe_h = if valid { ih as u32 } else { u32:0 };
    let safe_w = if valid { iw as u32 } else { u32:0 };
    (nhwc_offset(n, safe_h, safe_w, c, hi, wi, ci), valid)
}

pub fn deconv_input_coord(output: u32, kernel: u32, stride: u32, pad: u32) -> (s32, bool) {
    let numerator = (output as s32) + (pad as s32) - (kernel as s32);
    let valid = numerator >= s32:0 && (numerator % (stride as s32)) == s32:0;
    let coord = if valid { numerator / (stride as s32) } else { s32:0 };
    (coord, valid)
}

#[test]
fn nhwc_offset_test() {
    assert_eq(nhwc_offset(u32:1, u32:2, u32:3, u32:4, u32:5, u32:6, u32:7), u32:319);
}

#[test]
fn conv_address_test() {
    let (address, valid) = conv_nhwc_address(
        u32:0, u32:1, u32:1, u32:0, u32:0, u32:2, u32:4, u32:5, u32:3, u32:1, u32:1, u32:1, u32:1,
        u32:1, u32:1);
    assert_eq(valid, true);
    assert_eq(address, u32:2);
}

#[test]
fn conv_padding_test() {
    let (_, valid) = conv_nhwc_address(
        u32:0, u32:0, u32:0, u32:0, u32:0, u32:0, u32:4, u32:4, u32:3, u32:1, u32:1, u32:1, u32:1,
        u32:1, u32:1);
    assert_eq(valid, false);
}

#[test]
fn deconv_coord_test() {
    let (coord0, valid0) = deconv_input_coord(u32:1, u32:0, u32:2, u32:1);
    assert_eq(valid0, true);
    assert_eq(coord0, s32:1);
    let (_, valid1) = deconv_input_coord(u32:2, u32:0, u32:2, u32:1);
    assert_eq(valid1, false);
}

// --- layout ---

#[test]
fn nhwc_offset_is_row_major_test() {
    // ((n*H + h)*W + w)*C + c -- last index varies fastest.
    assert_eq(nhwc_offset(u32:0, u32:0, u32:0, u32:0, u32:4, u32:5, u32:3), u32:0);
    assert_eq(nhwc_offset(u32:0, u32:0, u32:0, u32:1, u32:4, u32:5, u32:3), u32:1);
    assert_eq(nhwc_offset(u32:0, u32:0, u32:1, u32:0, u32:4, u32:5, u32:3), u32:3);
    assert_eq(nhwc_offset(u32:0, u32:1, u32:0, u32:0, u32:4, u32:5, u32:3), u32:15);
    assert_eq(nhwc_offset(u32:1, u32:0, u32:0, u32:0, u32:4, u32:5, u32:3), u32:60);
}

// --- coordinate arithmetic ---

#[test]
fn conv_input_coord_applies_dilation_test() {
    // ih = oh*sh + kh*dh - ph
    assert_eq(conv_input_coord(u32:1, u32:2, u32:1, u32:3, u32:1), s32:6);
    assert_eq(conv_input_coord(u32:0, u32:0, u32:2, u32:1, u32:1), s32:-1);
    assert_eq(conv_input_coord(u32:3, u32:1, u32:2, u32:2, u32:0), s32:8);
}

#[test]
fn coord_in_bounds_edges_test() {
    assert_eq(coord_in_bounds(s32:0, u32:1), true);
    assert_eq(coord_in_bounds(s32:1, u32:1), false);
    assert_eq(coord_in_bounds(s32:-1, u32:1), false);
    assert_eq(coord_in_bounds(s32:3, u32:4), true);
    assert_eq(coord_in_bounds(s32:4, u32:4), false);
    // a zero limit bounds nothing
    assert_eq(coord_in_bounds(s32:0, u32:0), false);
}

#[test]
fn conv_address_stride_moves_input_row_test() {
    // oh=1 with stride 2 reaches input row 2 (kernel 0, pad 0)
    let (address, valid) = conv_nhwc_address(
        u32:0, u32:1, u32:0, u32:0, u32:0, u32:0, u32:8, u32:8, u32:1, u32:2, u32:1, u32:1, u32:1,
        u32:0, u32:0);
    assert_eq(valid, true);
    assert_eq(address, u32:16);   // row 2 * width 8 * channels 1
}

#[test]
fn conv_address_rejects_channel_out_of_range_test() {
    let (address, valid) = conv_nhwc_address(
        u32:0, u32:0, u32:0, u32:0, u32:0, u32:7, u32:4, u32:4, u32:3, u32:1, u32:1, u32:1, u32:1,
        u32:0, u32:0);
    assert_eq(valid, false);
    // When valid is low the caller must ignore the address; only the row/column
    // are clamped (so the arithmetic cannot wrap) -- the channel is not.
    assert_eq(address, u32:7);
}

#[test]
fn conv_address_rejects_right_padding_test() {
    // ow = 4 with wi = 5, kw = 1, pw = 2 -> iw = 4 + 2 - 2 = 4, in range
    let (_, ok) = conv_nhwc_address(
        u32:0, u32:0, u32:4, u32:0, u32:0, u32:0, u32:4, u32:5, u32:1, u32:1, u32:1, u32:1, u32:1,
        u32:0, u32:2);
    assert_eq(ok, true);
    // ow = 7 -> iw = 7 + 0 - 2 = 5, past the right edge of a 5-wide input
    let (_, bad) = conv_nhwc_address(
        u32:0, u32:0, u32:7, u32:0, u32:0, u32:0, u32:4, u32:5, u32:1, u32:1, u32:1, u32:1, u32:1,
        u32:0, u32:2);
    assert_eq(bad, false);
}

// --- transposed convolution coordinates ---

#[test]
fn deconv_input_coord_stride3_test() {
    // numerator = output + pad - kernel; must be >= 0 and a stride multiple.
    let (c0, v0) = deconv_input_coord(u32:4, u32:2, u32:3, u32:1);
    assert_eq(v0, true);
    assert_eq(c0, s32:1);          // 4 + 1 - 2 = 3, 3/3 = 1
    let (_, v1) = deconv_input_coord(u32:5, u32:2, u32:3, u32:1);
    assert_eq(v1, false);          // 5 + 1 - 2 = 4, 4 % 3 != 0
    let (c2, v2) = deconv_input_coord(u32:3, u32:0, u32:3, u32:0);
    assert_eq(v2, true);
    assert_eq(c2, s32:1);          // 3 + 0 - 0 = 3, 3/3 = 1
    let (_, v3) = deconv_input_coord(u32:2, u32:0, u32:3, u32:0);
    assert_eq(v3, false);          // 2 % 3 != 0
}

#[test]
fn deconv_input_coord_rejects_negative_test() {
    // output + pad < kernel gives a negative numerator.
    let (coord, valid) = deconv_input_coord(u32:0, u32:2, u32:2, u32:0);
    assert_eq(valid, false);
    // the returned coordinate is forced to 0 so it can never index out of bounds
    assert_eq(coord, s32:0);
}

#[test]
fn deconv_input_coord_negative_returned_as_zero_test() {
    // Even when invalid, the coordinate must be safe to use as an index.
    let (coord, valid) = deconv_input_coord(u32:1, u32:5, u32:2, u32:0);
    assert_eq(valid, false);
    assert_eq(coord, s32:0);
}

// ===========================================================================
// Fuzzed invariants. Run with
//   dslx_interpreter_main --evaluator=ir-interpreter \
//       --run_quickcheck_when_interpreting agu_nhwc.x
// ===========================================================================

#[quickcheck]
fn nhwc_channel_is_the_fastest_dimension_check
    (n: u2, h: u2, w: u2, c: u3) -> bool {
    // One channel further is exactly one element further in memory.
    nhwc_offset(n as u32, h as u32, w as u32, c as u32, u32:4, u32:4, u32:8)
        + u32:1
        == nhwc_offset(n as u32, h as u32, w as u32, (c as u32) + u32:1, u32:4, u32:4, u32:8)
}

#[quickcheck]
fn nhwc_offset_is_monotone_in_every_index_check
    (n: u2, h: u2, w: u2, c: u3) -> bool {
    // Walking any index forward must never move backwards in memory.
    let base = nhwc_offset(n as u32, h as u32, w as u32, c as u32, u32:4, u32:4, u32:8);
    nhwc_offset((n as u32) + u32:1, h as u32, w as u32, c as u32, u32:4, u32:4, u32:8) > base
        && nhwc_offset(n as u32, (h as u32) + u32:1, w as u32, c as u32, u32:4, u32:4, u32:8) > base
        && nhwc_offset(n as u32, h as u32, (w as u32) + u32:1, c as u32, u32:4, u32:4, u32:8) > base
        && nhwc_offset(n as u32, h as u32, w as u32, (c as u32) + u32:1, u32:4, u32:4, u32:8) > base
}

#[quickcheck]
fn coord_in_bounds_is_the_half_open_range_check(coord: s8, limit: u4) -> bool {
    coord_in_bounds(coord as s32, limit as u32)
        == ((coord as s32) >= s32:0 && (coord as s32) < (limit as s32))
}

#[quickcheck]
fn deconv_valid_implies_a_usable_coord_check
    (output: u4, kernel: u4, stride: u3, pad: u3) -> bool {
    // When the request is valid the coordinate is a real input row; when it is
    // not, the coordinate is forced to zero so it can never index out of bounds.
    let (coord, valid) = deconv_input_coord(output as u32, kernel as u32,
                                            (stride as u32) + u32:1, pad as u32);
    if valid { coord >= s32:0 } else { coord == s32:0 }
}

#[quickcheck]
fn conv_input_coord_is_monotone_in_the_output_check
    (oh: u4, kh: u3, sh: u3, dh: u3, ph: u3) -> bool {
    // A later output row never reaches an earlier input row.
    conv_input_coord(oh as u32, kh as u32, sh as u32, dh as u32, ph as u32)
        <= conv_input_coord((oh as u32) + u32:1, kh as u32, sh as u32, dh as u32, ph as u32)
}
