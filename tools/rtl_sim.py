"""Faithful Python port of the convolution accelerator hardware (DSLX source).

Ports hw/convacc_proc.x (+ mac64.x, requant.x) so we can run the EXACT
hardware algorithm on real SRAM images and compare to the software reference
(reference.py / expected.bin). This is a diagnostic tool to localize RTL
bugs: if this model reproduces the same regresion failures as the real RTL,
the bug lives in the DSLX algorithm (propagated through codegen); if it
passes cases the real RTL fails, the bug is in the codegen (the generated
SV diverged from the DSLX).

Run:  python3 tools/rtl_sim.py [case_dir_or_all]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# word/byte helpers (mirror explicit_state_access DSLX helpers)
# --------------------------------------------------------------------------
def lo32(x):
    return x & 0xFFFFFFFF


def hi32(x):
    return (x >> 32) & 0xFFFFFFFF


def half16(x, slot):
    return (x >> (slot * 16)) & 0xFFFF


def byte_at(x, i):
    return (x >> (i * 8)) & 0xFF


def as_s8(b):
    b &= 0xFF
    return b - 256 if b >= 128 else b


def as_u6(b):
    return b & 0x3F


# --------------------------------------------------------------------------
# mac64_reduced (port of hw/mac64.x) -- segmented reduction by log_r
# --------------------------------------------------------------------------
def mac64_reduced(channelwise, dense_act, ch_act, wgt, log_r):
    r_mask = (1 << log_r) - 1
    products = [0] * 64
    for lane in range(64):
        p = lane >> 3
        within = lane & 7
        r = within & r_mask
        act_index = (p << log_r) | r
        act = dense_act[act_index] if channelwise == 0 else ch_act[lane]
        products[lane] = int(act) * int(wgt[within])
    sum2 = [products[2 * i] + products[2 * i + 1] for i in range(32)]
    sum4 = [sum2[2 * i] + sum2[2 * i + 1] for i in range(16)]
    sum8 = [sum4[2 * i] + sum4[2 * i + 1] for i in range(8)]
    out = [0] * 64
    for index in range(64):
        p = index >> 3
        o = index & 7
        if log_r == 0:
            total = products[p * 8 + o]
        elif log_r == 1:
            total = sum2[(p * 4 + o) % 32]
        elif log_r == 2:
            total = sum4[(p * 2 + o) % 16]
        else:
            total = sum8[p % 8]
        out[index] = total
    return out


def requant_one(acc, bias, scale, shift, zero_point):
    q = (int(acc) + int(bias)) * int(scale)
    magnitude = q if q >= 0 else -q
    if shift == 0:
        rounded = magnitude
    else:
        rounded_magnitude = (magnitude + (1 << (shift - 1))) >> shift
        rounded = rounded_magnitude
    signed = rounded if q >= 0 else -rounded
    biased = signed + int(zero_point)
    if biased > 127:
        return 127
    if biased < -128:
        return -128
    return biased


# --------------------------------------------------------------------------
# Engine (port of convacc_proc.x Engine<16384>)
# --------------------------------------------------------------------------
class Engine:
    M = 16384

    def __init__(self):
        self.act = [0] * self.M
        self.wgt = [0] * self.M
        self.out_mem = [0] * self.M
        self.cfg = [0] * 16
        self.taps = [0] * 256
        self.params = [0] * 10
        self.acc = [0] * 64
        self.act_tile = [0] * 128
        self.ch_tile = [0] * 1024
        self.wgt_tile = [0] * 128
        self.phase = 0
        self.desc_start = 0
        self.desc_index = 0
        self.tap_index = 0
        self.next_desc = 0
        self.mt = self.nt = self.kb = 0
        self.ld_p = self.ld_r = self.wg_index = 0
        self.group_index = self.param_index = self.out_p = 0
        self.n_tiles = self.m_tiles = self.block_groups = 0
        self.bank = 0
        self.err = 0

    # --- cfg accessors (mirror convacc_proc.x cfg_* fns) ---
    def cw(self, i):
        return self.cfg[i]

    def flags_gemm(self):
        return (self.cw(0) & 1) != 0

    def flags_channel(self):
        return (self.cw(0) & 2) != 0

    def flags_deconv(self):
        return (self.cw(0) & 4) != 0

    def flags_logr(self):
        return (self.cw(0) >> 3) & 3

    def cfg_r(self):
        return 1 << self.flags_logr()

    def cfg_o(self):
        return 8 >> self.flags_logr()

    def cfg_m(self):
        return lo32(self.cw(1))

    def cfg_n(self):
        return hi32(self.cw(1))

    def cfg_k(self):
        return lo32(self.cw(2))

    def cfg_mh(self):
        return half16(self.cw(3), 0)

    def cfg_mw(self):
        return half16(self.cw(3), 1)

    def cfg_kgroups(self):
        return lo32(self.cw(4))

    def cfg_ci_count(self):
        return half16(self.cw(5), 0)

    def cfg_ci_base(self):
        return half16(self.cw(5), 1)

    def cfg_co_base(self):
        return half16(self.cw(5), 2)

    def cfg_co_group(self):
        return half16(self.cw(5), 3)

    def cfg_hi(self):
        return half16(self.cw(6), 0)

    def cfg_wi(self):
        return half16(self.cw(6), 1)

    def cfg_ho(self):
        return half16(self.cw(7), 0)

    def cfg_wo(self):
        return half16(self.cw(7), 1)

    def cfg_ci(self):
        return half16(self.cw(8), 0)

    def cfg_co(self):
        return half16(self.cw(8), 1)

    def cfg_sh(self):
        return half16(self.cw(9), 0)

    def cfg_sw(self):
        return half16(self.cw(9), 1)

    def cfg_dh(self):
        return half16(self.cw(9), 2)

    def cfg_dw(self):
        return half16(self.cw(9), 3)

    def cfg_ph(self):
        return half16(self.cw(10), 0)

    def cfg_pw(self):
        return half16(self.cw(10), 1)

    def cfg_phase_h(self):
        return half16(self.cw(10), 2)

    def cfg_phase_w(self):
        return half16(self.cw(10), 3)

    def cfg_steph(self):
        return half16(self.cw(11), 0)

    def cfg_stepw(self):
        return half16(self.cw(11), 1)

    def cfg_act_base(self):
        return lo32(self.cw(12))

    def cfg_out_base(self):
        return hi32(self.cw(12))

    def cfg_w_base(self):
        return lo32(self.cw(13))

    def cfg_p_base(self):
        return hi32(self.cw(13))

    def cfg_tap_base(self):
        return lo32(self.cw(14))

    def cfg_taps(self):
        return half16(self.cw(14), 2)

    # --- main step (mirror engine_step) ---
    def step(self, start, start_desc):
        if self.phase == 0:
            if start:
                self.phase = 1
                self.desc_start = start_desc
                self.desc_index = 0
                self.err = 0
            return False
        if self.phase == 1:
            self.cfg[self.desc_index] = self.wgt[self.desc_start + self.desc_index]
            if self.desc_index == 15:
                self.phase = 2
                self.tap_index = 0
            else:
                self.desc_index += 1
            return False
        if self.phase == 2:
            if self.tap_index >= self.cfg_taps():
                self.phase = 3
                self.mt = 0
                self.nt = 0
            else:
                word = self.wgt[self.cfg_tap_base() + self.tap_index // 8]
                self.taps[self.tap_index] = byte_at(word, self.tap_index % 8)
                self.tap_index += 1
            return False
        if self.phase == 3:
            n = self.cfg_n()
            o = self.cfg_o()
            m = self.cfg_m()
            self.n_tiles = (n + o - 1) // o if o else 0
            self.m_tiles = (m + 7) // 8 if m else 0
            self.phase = 4
            self.kb = 0
            return False
        if self.phase == 4:
            self.acc = [0] * 64
            self.phase = 5
            return False
        if self.phase == 5:
            self.act_tile = [0] * 128
            self.ch_tile = [0] * 1024
            self.wgt_tile = [0] * 128
            remaining = self.cfg_k() - self.kb * 8
            block = 8 if remaining > 8 else remaining
            r = self.cfg_r()
            self.block_groups = ((block + r - 1) // r) if r else 0
            self.wg_index = 0
            self.ld_p = 0
            self.ld_r = 0
            self.phase = 6
            return False
        if self.phase == 6:
            if self.wg_index < self.block_groups:
                word = self.wgt[self.cfg_w_base() + self.nt * self.cfg_kgroups()
                               + self.kb * (8 // self.cfg_r()) + self.wg_index]
                base = (self.bank ^ 1) * 64 + self.wg_index * 8
                for i in range(8):
                    self.wgt_tile[base + i] = byte_at(word, i)
                self.wg_index += 1
            else:
                self.phase = 7
                self.ld_p = 0
                self.ld_r = 0
            return False
        if self.phase == 7:
            return self._phase7()
        if self.phase == 9:
            return self._phase9()
        if self.phase == 8:
            return self._phase8()
        if self.phase == 10:
            self.params[self.param_index] = self.wgt[self.cfg_p_base()
                                                    + self.nt * 10 + self.param_index]
            if self.param_index == 9:
                self.phase = 11
                self.out_p = 0
            else:
                self.param_index += 1
            return False
        if self.phase == 11:
            return self._phase11()
        if self.phase == 13:
            nxt = half16(self.cw(15), 0)
            if nxt != 0:
                self.phase = 1
                self.desc_start = nxt
                self.desc_index = 0
            else:
                self.phase = 14
            return False if self.phase != 14 else True
        return False

    # --- phase 7: activation load (dense or channelwise) ---
    def _phase7(self):
        r_abs = self.kb * 8 + self.ld_r
        m_abs = self.mt * 8 + self.ld_p
        per_batch = self.cfg_mh() * self.cfg_mw()
        per_batch = per_batch if per_batch != 0 else 1
        batch = m_abs // per_batch
        rem = m_abs % per_batch
        oh = self.cfg_phase_h() + (rem // self.cfg_mw()) * self.cfg_steph() if self.cfg_mw() else self.cfg_phase_h()
        ow = self.cfg_phase_w() + (rem % self.cfg_mw()) * self.cfg_stepw() if self.cfg_mw() else self.cfg_phase_w()
        channel = self.flags_channel()
        tap_index = r_abs if channel else (r_abs // self.cfg_ci_count() if self.cfg_ci_count() else 0)
        tap_byte = self.taps[tap_index] if tap_index < self.cfg_taps() else 0
        kh = (tap_byte >> 4) & 0xF
        kw = tap_byte & 0xF
        num_h = (oh + self.cfg_ph() - kh) & 0xFFFFFFFF
        num_w = (ow + self.cfg_pw() - kw) & 0xFFFFFFFF
        if self.flags_deconv():
            ih = (num_h // self.cfg_sh()) if (self.cfg_sh() != 0 and num_h % self.cfg_sh() == 0) else 65535
            iw = (num_w // self.cfg_sw()) if (self.cfg_sw() != 0 and num_w % self.cfg_sw() == 0) else 65535
        else:
            ih = oh * self.cfg_sh() + kh * self.cfg_dh() - self.cfg_ph()
            iw = ow * self.cfg_sw() + kw * self.cfg_dw() - self.cfg_pw()
        co_start = self.cfg_co_base() + self.nt * self.cfg_o()
        valid_o = 8 if (self.cfg_n() - self.nt * self.cfg_o()) > 8 else (self.cfg_n() - self.nt * self.cfg_o())
        ci_start = co_start // self.cfg_co_group() if self.cfg_co_group() else 0
        ci_local = r_abs % self.cfg_ci_count() if self.cfg_ci_count() else 0
        ci_global = self.cfg_ci_base() + ci_local
        if self.flags_gemm():
            addr = self.cfg_act_base() + m_abs * self.cfg_k() + r_abs
        elif channel:
            addr = self.cfg_act_base() + ((batch * self.cfg_hi() + ih) * self.cfg_wi() + iw) * self.cfg_ci() + ci_start
        else:
            addr = self.cfg_act_base() + ((batch * self.cfg_hi() + ih) * self.cfg_wi() + iw) * self.cfg_ci() + ci_global
        in_bounds = (m_abs < self.cfg_m()) and (r_abs < self.cfg_k()) and (
            self.flags_gemm() or (ih < self.cfg_hi() and iw < self.cfg_wi()))
        word = self.act[addr // 8] if (addr // 8) < self.M else 0
        byte_sel = addr % 8
        if channel:
            fill = 1
        elif in_bounds:
            cap_tap = (self.cfg_ci_count() - ci_local) if not self.flags_gemm() else 8
            cap_k = self.cfg_k() - r_abs
            cap_r = 8 - self.ld_r
            cap_bytes = 8 - byte_sel
            a = cap_tap if cap_tap < cap_k else cap_k
            b = cap_bytes if cap_bytes < cap_r else cap_r
            fill = a if a < b else b
        else:
            fill = 1
        # channel tile update
        ch_tile_new = list(self.ch_tile)
        if channel:
            for i in range(8):
                if i < valid_o:
                    rel = i // self.cfg_co_group() if self.cfg_co_group() else 0
                    if byte_sel + rel < 8:
                        val = as_s8(byte_at(word, byte_sel + rel)) if in_bounds else 0
                        ch_tile_new[(self.bank ^ 1) * 512 + self.ld_r * 8 + i] = val
        act_tile_new = list(self.act_tile)
        if not channel:
            for i in range(8):
                if i < fill:
                    val = as_s8(byte_at(word, byte_sel + i)) if in_bounds else 0
                    act_tile_new[(self.bank ^ 1) * 64 + self.ld_p * 8 + self.ld_r + i] = val
        channel_hi = channel and in_bounds and (byte_sel + (valid_o - 1) // (self.cfg_co_group() if self.cfg_co_group() else 1)) >= 8
        next_r = self.ld_r + fill
        if next_r >= 8 or channel:
            if self.ld_p == 7:
                np_, nr_, done = (self.ld_p, self.ld_r, True)
            else:
                np_, nr_, done = (self.ld_p + 1, 0, False)
        else:
            np_, nr_, done = (self.ld_p, next_r, False)
        if done and not channel_hi:
            self.bank = self.bank ^ 1
        self.act_tile = act_tile_new
        self.ch_tile = ch_tile_new
        self.ld_p = np_
        self.ld_r = nr_
        if channel_hi:
            self.phase = 9
        elif done:
            self.phase = 8
        else:
            self.phase = 7
        return False

    def _phase9(self):
        r_abs = self.kb * 8 + self.ld_r
        m_abs = self.mt * 8 + self.ld_p
        per_batch = self.cfg_mh() * self.cfg_mw()
        per_batch = per_batch if per_batch != 0 else 1
        batch = m_abs // per_batch
        rem = m_abs % per_batch
        oh = self.cfg_phase_h() + (rem // self.cfg_mw()) * self.cfg_steph() if self.cfg_mw() else self.cfg_phase_h()
        ow = self.cfg_phase_w() + (rem % self.cfg_mw()) * self.cfg_stepw() if self.cfg_mw() else self.cfg_phase_w()
        tap_byte = self.taps[r_abs] if r_abs < self.cfg_taps() else 0
        kh = (tap_byte >> 4) & 0xF
        kw = tap_byte & 0xF
        num_h = (oh + self.cfg_ph() - kh) & 0xFFFFFFFF
        num_w = (ow + self.cfg_pw() - kw) & 0xFFFFFFFF
        if self.flags_deconv():
            ih = (num_h // self.cfg_sh()) if (self.cfg_sh() != 0 and num_h % self.cfg_sh() == 0) else 65535
            iw = (num_w // self.cfg_sw()) if (self.cfg_sw() != 0 and num_w % self.cfg_sw() == 0) else 65535
        else:
            ih = oh * self.cfg_sh() + kh * self.cfg_dh() - self.cfg_ph()
            iw = ow * self.cfg_sw() + kw * self.cfg_dw() - self.cfg_pw()
        co_start = self.cfg_co_base() + self.nt * self.cfg_o()
        valid_o = 8 if (self.cfg_n() - self.nt * self.cfg_o()) > 8 else (self.cfg_n() - self.nt * self.cfg_o())
        ci_start = co_start // self.cfg_co_group() if self.cfg_co_group() else 0
        addr = self.cfg_act_base() + ((batch * self.cfg_hi() + ih) * self.cfg_wi() + iw) * self.cfg_ci() + ci_start
        word = self.act[addr // 8] if (addr // 8) < self.M else 0
        byte_sel = addr % 8
        ch_tile_new = list(self.ch_tile)
        for i in range(8):
            if i < valid_o:
                rel = i // self.cfg_co_group() if self.cfg_co_group() else 0
                if byte_sel + rel >= 8:
                    val = as_s8(byte_at(word, byte_sel + rel - 8))
                    ch_tile_new[(self.bank ^ 1) * 512 + self.ld_r * 8 + i] = val
        done = self.ld_p == 7
        self.ch_tile = ch_tile_new
        if done:
            self.ld_r = self.ld_r
        else:
            self.ld_p += 1
            self.ld_r = 0
        if done:
            self.phase = 8
        else:
            self.phase = 7
        if done:
            self.bank = self.bank ^ 1
        return False

    def _phase8(self):
        if self.group_index < self.block_groups:
            r = self.cfg_r()
            o = self.cfg_o()
            dense = [0] * 64
            for i in range(64):
                p = (i // r) % 8
                rr = i % r
                dense[i] = self.act_tile[self.bank * 64 + p * 8 + self.group_index * r + rr]
            ch = [0] * 64
            for i in range(64):
                p = i // 8
                within = i % 8
                oo = within // r
                rr = within % r
                ch[i] = self.ch_tile[self.bank * 512 + p * 64
                                    + (self.group_index * r + rr) * 8 + oo]
            wgt_arr = [as_s8(self.wgt_tile[self.bank * 64 + self.group_index * 8 + i]) for i in range(8)]
            reduced = mac64_reduced(self.flags_channel(), dense, ch, wgt_arr, self.flags_logr())
            new_acc = list(self.acc)
            for i in range(64):
                if (i % 8) < o and (self.mt * 8 + i // 8) < self.cfg_m():
                    new_acc[i] = self.acc[i] + reduced[i]
            self.acc = new_acc
            if self.group_index + 1 < self.block_groups:
                self.group_index += 1
            elif (self.kb + 1) * 8 < self.cfg_k():
                self.kb += 1
                self.phase = 5
            else:
                self.param_index = 0
                self.phase = 10
        return False

    def _phase11(self):
        p = self.out_p
        m_abs = self.mt * 8 + p
        # requant lanes
        bias = [0] * 8
        scale = [0] * 8
        shifts = [0] * 8
        zps = [0] * 8
        for i in range(8):
            w = self.params[i // 2]
            raw = lo32(w) if (i % 2 == 0) else hi32(w)
            bias[i] = raw
        for i in range(8):
            w = self.params[4 + i // 2]
            raw = lo32(w) if (i % 2 == 0) else hi32(w)
            scale[i] = raw
        for i in range(8):
            shifts[i] = as_u6(byte_at(self.params[8], i))
        for i in range(8):
            zps[i] = as_s8(byte_at(self.params[9], i))
        lanes = [self.acc[p * 8 + i] for i in range(8)]
        q = [requant_one(lanes[i], bias[i], scale[i], shifts[i], zps[i]) for i in range(8)]
        o = self.cfg_o()
        valid_o = o if (self.cfg_n() - self.nt * o) > o else (self.cfg_n() - self.nt * o)
        addr = self._out_word(p)
        qword = 0
        for i in range(8):
            if i < valid_o:
                qword |= (q[i] & 0xFF) << (i * 8)
        if m_abs < self.cfg_m():
            self.out_mem[addr] = qword & 0xFFFFFFFFFFFFFFFF
        if p < 7:
            self.out_p = p + 1
            self.phase = 11
        elif self.nt + 1 < self.n_tiles:
            self.nt += 1
            self.phase = 4
        elif self.mt + 1 < self.m_tiles:
            self.mt += 1
            self.nt = 0
            self.phase = 4
        else:
            self.phase = 13
        return False

    def _out_word(self, p):
        per_batch = self.cfg_mh() * self.cfg_mw()
        per_batch = per_batch if per_batch != 0 else 1
        m_abs = self.mt * 8 + p
        batch = m_abs // per_batch
        rem = m_abs % per_batch
        oh = self.cfg_phase_h() + (rem // self.cfg_mw()) * self.cfg_steph() if self.cfg_mw() else self.cfg_phase_h()
        ow = self.cfg_phase_w() + (rem % self.cfg_mw()) * self.cfg_stepw() if self.cfg_mw() else self.cfg_phase_w()
        if self.flags_gemm():
            pos = m_abs
        else:
            pos = (batch * self.cfg_ho() + oh) * self.cfg_wo() + ow
        return self.cfg_out_base() // 8 + pos * self._out_stride_tiles() + self._out_group() * self.n_tiles + self.nt

    def _out_group(self):
        if self.flags_gemm() or self.flags_channel():
            return 0
        return self.cfg_co_base() // self.cfg_co_group()

    def _out_stride_tiles(self):
        if self.flags_gemm() or self.flags_channel():
            return self.n_tiles
        return (self.cfg_co() // self.cfg_co_group()) * self.n_tiles


def load_sram_words(path):
    data = Path(path).read_bytes()
    return [int.from_bytes(data[i:i + 8], "little") for i in range(0, len(data), 8)]


def simulate(act_bytes, wgt_bytes):
    st = Engine()
    st.act = [int.from_bytes(act_bytes[i:i + 8], "little") for i in range(0, len(act_bytes), 8)]
    st.wgt = [int.from_bytes(wgt_bytes[i:i + 8], "little") for i in range(0, len(wgt_bytes), 8)]
    first = True
    while st.phase != 14:
        st.step(first, 0)
        first = False
    out = b"".join(w.to_bytes(8, "little") for w in st.out_mem)
    return out


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    cases_dir = ROOT / "build" / "cocotb_cases"
    if target == "all":
        import glob
        dirs = sorted(glob.glob(str(cases_dir / "*")))
    else:
        dirs = [str(cases_dir / target)]
    passed = 0
    failed = []
    for d in dirs:
        name = Path(d).name
        act = Path(d) / "activation.bin"
        wgt = Path(d) / "weight.bin"
        exp = Path(d) / "expected.bin"
        if not (act.exists() and wgt.exists() and exp.exists()):
            continue
        out = simulate(act.read_bytes(), wgt.read_bytes())
        expected = exp.read_bytes()
        if out[:len(expected)] == expected:
            passed += 1
        else:
            # find first mismatch byte
            n = min(len(out), len(expected))
            first = next((i for i in range(n) if out[i] != expected[i]), None)
            failed.append((name, first))
    print(f"passed={passed} failed={len(failed)}")
    if failed:
        print("first 40 failures:", failed[:40])


if __name__ == "__main__":
    main()
