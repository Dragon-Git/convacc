#!/bin/sh
# Runs the FuzzTest suite as N shards, one process each -- once, or repeatedly
# until a wall-clock deadline.
#
# FuzzTest registers every FUZZ_TEST as an ordinary GoogleTest test, and both
# GoogleTest and FuzzTest honour GTEST_TOTAL_SHARDS/GTEST_SHARD_INDEX: a shard
# runs only its own tests and splits the time budget across them. Parallel
# fuzzing therefore needs no external runner -- N processes with the shard
# environment set are enough. gtest-parallel would add nothing here: it works a
# test at a time and cannot split a single test, so with four tests it can only
# ever use four workers.
#
# In unit-test mode FuzzTest caps one test's mutation loop at 10000 iterations,
# which on these shapes is 15-60 s. FUZZTEST_FUZZ_FOR can only cut that short,
# never extend it, so a one-hour run cannot be one process: it would stop after
# a minute. With DURATION set, each shard restarts until the deadline. Every
# restart draws a fresh PRNG seed -- printed as FUZZTEST_PRNG_SEED in the log,
# so a finding stays reproducible -- and explores a new region of the shape
# space. A failure stops every shard after its current round, and the logs and
# reproducers are kept for upload.
#
# usage: run_fuzz_shards.sh BINARY SHARDS BUDGET LOGDIR [DURATION]
#   SHARDS    a number, or "auto" for min(cores this container may use, tests)
#   BUDGET    FUZZTEST_FUZZ_FOR for one round, in seconds, e.g. 60s
#   DURATION  optional total wall-clock seconds; 0 or absent = one round only

set -u

binary=$1
shards=$2
budget=$3
logdir=$4
duration=${5:-0}

case "$duration" in
  ''|*[!0-9]*) echo "DURATION must be a non-negative integer" >&2; exit 2 ;;
esac
case "$budget" in
  *s) round_seconds=${budget%s} ;;
  *) echo "BUDGET must be a seconds value like 60s" >&2; exit 2 ;;
esac
case "$round_seconds" in
  ''|*[!0-9]*) echo "BUDGET must be a seconds value like 60s" >&2; exit 2 ;;
esac
[ "$round_seconds" -gt 0 ] || { echo "BUDGET must be positive" >&2; exit 2; }

# nproc reports the host's cores; a container is limited by its cgroup quota.
effective_cores() {
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    set -- $(cat /sys/fs/cgroup/cpu.max)
    if [ "$1" != "max" ]; then
      echo $(( $1 / $2 ))
      return
    fi
  fi
  if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [ "$quota" -gt 0 ] 2>/dev/null; then
      echo $(( quota / period ))
      return
    fi
  fi
  nproc
}

test_count() {
  "$binary" --gtest_list_tests 2>/dev/null | grep -c '^  '
}

if [ "$shards" = auto ]; then
  cores=$(effective_cores)
  tests=$(test_count)
  shards=$cores
  [ "$tests" -lt "$shards" ] && shards=$tests
  [ "$shards" -lt 1 ] && shards=1
  printf '%s cores available, %s fuzz tests -> %s shards\n' "$cores" "$tests" "$shards"
fi

mkdir -p "$logdir"
start=$(date +%s)
deadline=$(( start + duration ))
stop="$logdir/.stop"
rm -f "$stop"

# One background loop per shard. Every variable below lives in the child that
# `&` forks, so shards cannot clobber each other's counters.
run_shard() {
  index=$1
  log="$logdir/shard-$index.log"
  : > "$log"
  round=0
  while :; do
    [ -f "$stop" ] && break
    now=$(date +%s)
    this_round=$round_seconds
    if [ "$duration" -gt 0 ]; then
      [ "$now" -ge "$deadline" ] && break
      left=$(( deadline - now ))
      [ "$left" -lt "$this_round" ] && this_round=$left
      printf '\n===== round %s, %ss =====\n' "$round" "$this_round" >> "$log"
    fi
    if ! GTEST_TOTAL_SHARDS=$shards GTEST_SHARD_INDEX=$index \
         FUZZTEST_FUZZ_FOR="${this_round}s" "$binary" >> "$log" 2>&1; then
      : > "$stop"
      printf 'shard %s/%s: FAILED at round %s -- %s\n' "$index" "$shards" "$round" "$log" >&2
      sed -n '/\[  FAILED  \]/,$p' "$log" | head -20 | sed 's/^/    /' >&2
      return 1
    fi
    round=$(( round + 1 ))
    [ "$duration" -eq 0 ] && break
  done

  if [ "$duration" -eq 0 ]; then
    ran=$(sed -n 's/.*\[==========\] \([0-9]*\) tests\? from.*/\1/p' "$log" | head -1)
    printf 'shard %s/%s: ok (%s tests)\n' "$index" "$shards" "${ran:-0}"
  else
    printf 'shard %s/%s: ok (%s rounds)\n' "$index" "$shards" "$round"
  fi
  return 0
}

pids=""
i=0
while [ "$i" -lt "$shards" ]; do
  run_shard "$i" &
  pids="$pids $!"
  i=$((i + 1))
done

status=0
for pid in $pids; do
  wait "$pid" || status=1
done

elapsed=$(( $(date +%s) - start ))
if [ "$duration" -gt 0 ]; then
  rounds=$(grep -h '^===== round ' "$logdir"/shard-*.log 2>/dev/null | wc -l | tr -d ' ')
  printf '%s rounds over %s shard(s) in %ss (deadline %ss)\n' \
    "$rounds" "$shards" "$elapsed" "$duration"
else
  ran_total=0
  for log in "$logdir"/shard-*.log; do
    ran=$(sed -n 's/.*\[==========\] \([0-9]*\) tests\? from.*/\1/p' "$log" | head -1)
    ran_total=$((ran_total + ${ran:-0}))
  done
  printf '%s fuzz tests over %s shard(s) in %ss (%s per test)\n' \
    "$ran_total" "$shards" "$elapsed" "$budget"
fi
exit $status
