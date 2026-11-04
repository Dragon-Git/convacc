#include "dut/dut.h"

#include <cstring>

#include "Vconvacc_top.h"
#include "Vconvacc_top___024root.h"
#include "verilated.h"

namespace convacc {

struct Dut::Impl {
  Vconvacc_top* top = nullptr;
  VlUnpacked<QData, kWords>* act = nullptr;
  VlUnpacked<QData, kWords>* wgt = nullptr;
  VlUnpacked<QData, kWords>* out = nullptr;
};

Dut::Dut() : impl_(new Impl) {
  impl_->top = new Vconvacc_top;
  impl_->act = &impl_->top->rootp->convacc_top__DOT__u_act_sram__DOT__mem;
  impl_->wgt = &impl_->top->rootp->convacc_top__DOT__u_wgt_sram__DOT__mem;
  impl_->out = &impl_->top->rootp->convacc_top__DOT__u_out_sram__DOT__mem;
  impl_->top->eval();  // let the power-on initial blocks settle
}

Dut::~Dut() {
  delete impl_->top;
  delete impl_;
}

void Dut::tick() {
  impl_->top->clk = 1;
  impl_->top->eval();
  impl_->top->clk = 0;
  impl_->top->eval();
}

void Dut::reset() {
  Vconvacc_top* top = impl_->top;
  top->rst = 1;
  top->start = 0;
  top->start_desc = 0;
  top->act_we = 0;
  top->wgt_we = 0;
  top->act_wstrb = 0;
  top->wgt_wstrb = 0;
  top->out_re = 0;
  top->out_raddr = 0;
  for (int i = 0; i < 5; ++i) tick();
  top->rst = 0;
  tick();
}

void Dut::load(const SramImage& image, int check_words) {
  std::memcpy(impl_->act->data(), image.activation.data(), kSramBytes);
  std::memcpy(impl_->wgt->data(), image.weight.data(), kSramBytes);
  std::memset(impl_->out->data(), 0, sizeof(uint64_t) * static_cast<size_t>(check_words));
  // The arrays are read combinationally, so make sure the newly written values
  // are what the first evaluated edge sees.
  impl_->top->eval();
}

SimResult Dut::run(long long max_cycles) {
  Vconvacc_top* top = impl_->top;
  top->act_we = 0;
  top->wgt_we = 0;
  top->act_wstrb = 0;
  top->wgt_wstrb = 0;
  top->out_re = 0;
  top->out_raddr = 0;

  top->start = 1;
  top->start_desc = 0;
  tick();
  top->start = 0;

  SimResult result;
  for (long long cycle = 1; cycle < max_cycles; ++cycle) {
    tick();
    if (top->done) {
      result.cycles = cycle;
      result.ok = true;
      result.error = top->error != 0;
      return result;
    }
  }
  result.cycles = max_cycles;
  result.timed_out = true;
  return result;
}

void Dut::read_output(int words, std::vector<uint8_t>& out) const {
  out.resize(static_cast<size_t>(words) * 8);
  std::memcpy(out.data(), impl_->out->data(), out.size());
}

uint64_t Dut::read_word(int address) const {
  return (*impl_->out)[static_cast<size_t>(address)];
}

uint64_t Dut::peek_activation(int address) const {
  return (*impl_->act)[static_cast<size_t>(address)];
}

uint64_t Dut::peek_weight(int address) const {
  return (*impl_->wgt)[static_cast<size_t>(address)];
}

}  // namespace convacc
