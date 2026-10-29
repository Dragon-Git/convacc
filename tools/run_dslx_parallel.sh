#!/bin/sh
# Run DSLX test files in parallel.
#
# `dslx_interpreter_main` is single-threaded, but every .x file is independent:
# it parses, typechecks and runs its own `#[test]`s. So the parallelism unit is
# the file. Output is buffered per file and replayed in argument order, because
# interleaved progress lines from N interpreters are unreadable.
#
# Usage:  JOBS=8 XLS_BIN=/path/to/bin tools/run_dslx_parallel.sh hw/a.x hw/b.x ...
#         XLS_FLAGS adds interpreter options (word-split, so quote multi-word
#         values), e.g. the quickcheck flags.
# Exit:   0 only if every file passed.
set -u

JOBS=${JOBS:-8}
XLS_BIN=${XLS_BIN:-}
XLS_FLAGS=${XLS_FLAGS:-}
# Which tool to run per file. `prove_quickcheck_main` proves a file's
# #[quickcheck]s for all inputs instead of sampling them.
DSLX_TOOL=${DSLX_TOOL:-dslx_interpreter_main}
if [ -z "$XLS_BIN" ]; then
    echo "run_dslx_parallel.sh: XLS_BIN is not set" >&2
    exit 2
fi
if [ "$#" -eq 0 ]; then
    echo "run_dslx_parallel.sh: no test files given" >&2
    exit 2
fi

tmp=$(mktemp -d) || exit 2
trap 'rm -rf "$tmp"' EXIT

running=0
for f in "$@"; do
    base=$(basename "$f")
    (
        cd "$(dirname "$f")" || exit 1
        # shellcheck disable=SC2086  # XLS_FLAGS is deliberately word-split
        "$XLS_BIN/$DSLX_TOOL" $XLS_FLAGS "$base" >"$tmp/$base.log" 2>&1
        echo $? >"$tmp/$base.rc"
    ) &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
        wait
        running=0
    fi
done
wait

status=0
for f in "$@"; do
    base=$(basename "$f")
    cat "$tmp/$base.log"
    rc=$(cat "$tmp/$base.rc" 2>/dev/null || echo 1)
    if [ "$rc" -ne 0 ]; then
        echo "FAILED: $f (exit $rc)" >&2
        status=1
    fi
done
exit "$status"
