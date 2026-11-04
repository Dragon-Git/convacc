// Native Verilator harness for convacc_top.
//
// cocotb reaches every signal through VPI, so one `dut.signal.value = x` costs a
// Python call, a VPI dispatch and a string-based lookup. Here the SRAM arrays
// are addressed directly as `VlUnpacked<uint64_t, 16384>` and an image is one
// memcpy, which is where the bulk of the speedup comes from.
#pragma once

#include <cstdint>
#include <vector>

#include "convacc/image.h"

namespace convacc {

struct SimResult {
  bool ok = false;
  bool timed_out = false;
  bool error = false;
  long long cycles = 0;
};

class Dut {
 public:
  Dut();
  ~Dut();
  Dut(const Dut&) = delete;
  Dut& operator=(const Dut&) = delete;

  // 5 cycles with rst high, then one more after releasing it -- the same
  // sequence tests/cocotb/tb_convacc_batch.py uses.
  void reset();

  // Overwrites the whole activation and weight SRAMs and zeroes the first
  // `check_words` words of the output window, so a case never inherits state
  // from the previous one.
  void load(const SramImage& image, int check_words);

  SimResult run(long long max_cycles = 5000000);

  void read_output(int words, std::vector<uint8_t>& out) const;

  // Single word as the host read port would return it (combinational read).
  uint64_t read_word(int address) const;

  // Debug peeks at the internal SRAM contents (what the model actually holds).
  uint64_t peek_activation(int address) const;
  uint64_t peek_weight(int address) const;

 private:
  void tick();

  struct Impl;
  Impl* impl_;
};

}  // namespace convacc
