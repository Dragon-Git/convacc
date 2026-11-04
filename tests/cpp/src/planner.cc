#include "convacc/planner.h"

#include <algorithm>
#include <set>
#include <tuple>

namespace convacc {
namespace {

Task dense_task(const ConvSpec& spec, int group) {
  Task t;
  t.name = "dense_g" + std::to_string(group);
  t.mode = "dense";
  t.m = spec.n * spec.ho * spec.wo;
  t.n = spec.co_per_group();
  t.k = spec.kh * spec.kw * spec.ci_per_group();
  t.batches = spec.n;
  t.mh = spec.ho;
  t.mw = spec.wo;
  for (int kh = 0; kh < spec.kh; ++kh)
    for (int kw = 0; kw < spec.kw; ++kw) t.taps.emplace_back(kh, kw);
  t.ci_base = group * spec.ci_per_group();
  t.ci_count = spec.ci_per_group();
  t.co_base = group * spec.co_per_group();
  t.group = group;
  return t;
}

Task channelwise_task(const ConvSpec& spec) {
  Task t;
  t.name = "channelwise_g0";
  t.mode = "channelwise";
  t.m = spec.n * spec.ho * spec.wo;
  t.n = spec.co;
  t.k = spec.kh * spec.kw;
  t.batches = spec.n;
  t.mh = spec.ho;
  t.mw = spec.wo;
  for (int kh = 0; kh < spec.kh; ++kh)
    for (int kw = 0; kw < spec.kw; ++kw) t.taps.emplace_back(kh, kw);
  t.ci_base = 0;
  t.ci_count = 1;
  t.co_base = 0;
  return t;
}

int phase_count(int size, int phase, int stride) {
  return phase >= size ? 0 : (size - 1 - phase) / stride + 1;
}

std::vector<Task> deconv_tasks(const ConvSpec& spec) {
  std::vector<Task> tasks;
  for (int rh = 0; rh < spec.sh; ++rh) {
    for (int rw = 0; rw < spec.sw; ++rw) {
      Task t;
      t.name = "phase_" + std::to_string(rh) + "_" + std::to_string(rw);
      t.mode = "dense";
      for (int kh = 0; kh < spec.kh; ++kh) {
        for (int kw = 0; kw < spec.kw; ++kw) {
          if ((rh + spec.ph - kh) % spec.sh == 0 && (rw + spec.pw - kw) % spec.sw == 0) {
            t.taps.emplace_back(kh, kw);
          }
        }
      }
      const int mh = phase_count(spec.ho, rh, spec.sh);
      const int mw = phase_count(spec.wo, rw, spec.sw);
      if (mh == 0 || mw == 0) continue;
      // A phase with no surviving taps is still emitted: those output positions
      // receive no contribution at all, but they still have to be written --
      // their value is requant(0), which is only zero for a trivial bias.
      t.m = spec.n * mh * mw;
      t.n = spec.co;
      t.k = static_cast<int>(t.taps.size()) * spec.ci;
      t.batches = spec.n;
      t.mh = mh;
      t.mw = mw;
      t.ci_base = 0;
      t.ci_count = spec.ci;
      t.co_base = 0;
      t.phase_h = rh;
      t.phase_w = rw;
      t.out_step_h = spec.sh;
      t.out_step_w = spec.sw;
      tasks.push_back(std::move(t));
    }
  }
  return tasks;
}

int channelwise_unique_bytes(const ConvSpec& spec, const Task& task, int mt, int nt) {
  std::set<long long> addresses;
  const int co_per_group = spec.co_per_group();
  for (int p = mt * 8; p < std::min((mt + 1) * 8, task.m); ++p) {
    const OutCoord c = output_coords(task, p);
    for (int lane = nt * 8; lane < std::min((nt + 1) * 8, task.n); ++lane) {
      const int ci = lane / co_per_group;
      for (const auto& tap : task.taps) {
        int ih = 0;
        int iw = 0;
        if (input_coord(spec, c, tap, &ih, &iw)) {
          addresses.insert((1LL * (c.batch * spec.hi + ih) * spec.wi + iw) * spec.ci + ci);
        }
      }
    }
  }
  return static_cast<int>(addresses.size());
}

long long estimate(const OpSpec& spec, const Task& task) {
  if (task.mode == "dense" || task.mode == "gemm") {
    return 1LL * task.m_tiles() * task.n_tiles() *
           std::max({task.k, task.k_groups(), task.o()});
  }
  const ConvSpec& conv = as_conv(spec);
  long long cycles = 0;
  for (int mt = 0; mt < task.m_tiles(); ++mt) {
    for (int nt = 0; nt < task.n_tiles(); ++nt) {
      const int act_cycles = (channelwise_unique_bytes(conv, task, mt, nt) + 7) / 8;
      cycles += std::max({task.k_groups(), act_cycles, task.o()});
    }
  }
  return cycles;
}

long long valid_tap_positions(const ConvSpec& spec, const Task& task) {
  std::vector<int> kh_values, kw_values;
  for (const auto& tap : task.taps) {
    kh_values.push_back(tap.first);
    kw_values.push_back(tap.second);
  }
  std::sort(kh_values.begin(), kh_values.end());
  kh_values.erase(std::unique(kh_values.begin(), kh_values.end()), kh_values.end());
  std::sort(kw_values.begin(), kw_values.end());
  kw_values.erase(std::unique(kw_values.begin(), kw_values.end()), kw_values.end());

  long long spatial = 0;
  for (int py = 0; py < task.mh; ++py) {
    const int oh = task.phase_h + py * task.out_step_h;
    int valid_kh = 0;
    for (int kh : kh_values) {
      int ih = 0;
      valid_kh += input_row(spec, oh, kh, &ih) ? 1 : 0;
    }
    for (int px = 0; px < task.mw; ++px) {
      const int ow = task.phase_w + px * task.out_step_w;
      int valid_kw = 0;
      for (int kw : kw_values) {
        int iw = 0;
        valid_kw += input_col(spec, ow, kw, &iw) ? 1 : 0;
      }
      spatial += 1LL * valid_kh * valid_kw;
    }
  }
  return 1LL * spec.n * spatial;
}

long long useful_macs(const OpSpec& spec, const Task& task) {
  if (is_gemm(spec)) return 1LL * task.m * task.n * task.k;
  return valid_tap_positions(as_conv(spec), task) * task.n * task.ci_count;
}

}  // namespace

OutCoord output_coords(const Task& task, int index) {
  const int per_batch = task.mh * task.mw;
  OutCoord c;
  c.batch = index / per_batch;
  const int rem = index % per_batch;
  c.oh = task.phase_h + (rem / task.mw) * task.out_step_h;
  c.ow = task.phase_w + (rem % task.mw) * task.out_step_w;
  return c;
}

Plan make_plan(const OpSpec& spec) {
  std::vector<Task> tasks;
  if (is_gemm(spec)) {
    const GemmSpec& g = as_gemm(spec);
    Task t;
    t.name = "gemm";
    t.mode = "gemm";
    t.m = g.m;
    t.n = g.n;
    t.k = g.k;
    t.batches = 1;
    t.mh = g.m;
    t.mw = 1;
    t.taps.emplace_back(0, 0);
    t.ci_base = 0;
    t.ci_count = g.k;
    t.co_base = 0;
    tasks.push_back(std::move(t));
  } else {
    const ConvSpec& c = as_conv(spec);
    if (c.deconv) {
      tasks = deconv_tasks(c);
    } else if (c.channelwise()) {
      tasks.push_back(channelwise_task(c));
    } else {
      for (int group = 0; group < c.groups; ++group) tasks.push_back(dense_task(c, group));
    }
  }
  if (tasks.empty()) throw SpecError("operator has no non-empty tasks");

  // One reduction width per operator: the output layout packs each n-tile into
  // a 64-bit word, so `o = 8 / r` must be identical for every task writing to
  // the same output image. Channelwise additionally needs `o >= co_per_group`,
  // because mac64_reduced can only address `oo = within / r` output channels.
  std::vector<int> allowed = {1, 2, 4, 8};
  if (is_channelwise(spec)) {
    const int need = as_conv(spec).co_per_group();
    std::vector<int> filtered;
    for (int r : allowed)
      if (8 / r >= need) filtered.push_back(r);
    allowed = filtered;
    if (allowed.empty()) {
      throw SpecError("channelwise needs o >= co_per_group=" + std::to_string(need) +
                      ", but the widest n-tile holds 8 channels");
    }
  }

  int best_r = allowed.front();
  long long best_cost = -1;
  for (int r : allowed) {
    long long cost = 0;
    for (Task t : tasks) {
      t.r = r;
      cost += estimate(spec, t);
    }
    // Ties resolve to the smaller r, matching Python's min() over the
    // (cost, r) key.
    if (best_cost < 0 || cost < best_cost) {
      best_cost = cost;
      best_r = r;
    }
  }
  for (auto& t : tasks) t.r = best_r;

  Plan plan;
  plan.useful_macs = 0;
  for (const auto& t : tasks) {
    plan.task_cycles.push_back(static_cast<int>(estimate(spec, t)));
    plan.useful_macs += useful_macs(spec, t);
  }
  plan.tasks = std::move(tasks);
  return plan;
}

}  // namespace convacc
