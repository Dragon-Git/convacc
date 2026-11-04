// Case-level parallelism for the native harness.
//
// Both drivers (the golden regression and the corpus replay) have the same
// shape: N independent cases, each needing a private Verilated model instance
// and each costing 1-30 ms. That used to be spelled out twice, in two copies of
// atomic-cursor-plus-threads, which is exactly the sort of thing that drifts.
//
// Two details are worth keeping in one place:
//
//   * The state is built once per worker, not once per case. Elaborating a
//     Verilated model costs ~300 ms, so a process- or test-per-case scheme
//     (gtest-parallel, ctest -j) would spend 50x more time on setup than on
//     simulation.
//   * Work is handed out through an atomic cursor instead of a static slice.
//     Case cost varies by 2-3x across the grid, and a static split leaves one
//     worker running long after the others have finished.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace convacc {

struct ParallelReport {
  int workers = 0;
  double seconds = 0.0;
};

// Jobs <= 0 means "whatever the machine reports"; the result is clamped to the
// number of cases so a small corpus does not spawn idle threads.
inline int effective_workers(int jobs, size_t count) {
  if (count == 0) return 0;
  if (jobs <= 0) {
    const unsigned hw = std::thread::hardware_concurrency();
    jobs = hw ? static_cast<int>(hw) : 4;
  }
  if (static_cast<size_t>(jobs) > count) jobs = static_cast<int>(count);
  return jobs;
}

// Runs `body(state, index)` for every index in [0, count), with `jobs` workers
// that each own one `State` obtained from `make_state()`.
//
// `body` runs on a worker thread: an exception escaping it reaches the top of
// that thread and terminates the process, so the body has to handle its own.
//
// `progress(finished, elapsed_seconds)` is called from a worker every
// `progress_every` cases (0 disables it). It runs under a lock, so it is the
// only place where reporting is serialised.
template <class MakeState, class Body, class Progress>
ParallelReport parallel_for(size_t count, int jobs, MakeState&& make_state, Body&& body,
                            Progress&& progress, size_t progress_every) {
  using State = std::decay_t<std::invoke_result_t<MakeState&>>;

  ParallelReport report;
  report.workers = effective_workers(jobs, count);
  if (report.workers == 0) return report;

  std::atomic<size_t> next{0};
  std::atomic<size_t> finished{0};
  std::mutex report_mutex;
  const auto start = std::chrono::steady_clock::now();

  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(report.workers));
  for (int w = 0; w < report.workers; ++w) {
    workers.emplace_back([&]() {
      State state = make_state();
      for (;;) {
        const size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= count) break;
        body(state, index);
        if (!progress_every) continue;
        const size_t done = finished.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done % progress_every != 0) continue;
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::lock_guard<std::mutex> lock(report_mutex);
        progress(done, elapsed);
      }
    });
  }
  for (auto& worker : workers) worker.join();

  report.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return report;
}

// Without a progress hook.
template <class MakeState, class Body>
ParallelReport parallel_for(size_t count, int jobs, MakeState&& make_state, Body&& body) {
  return parallel_for(count, jobs, std::forward<MakeState>(make_state),
                      std::forward<Body>(body), [](size_t, double) {}, 0);
}

}  // namespace convacc
