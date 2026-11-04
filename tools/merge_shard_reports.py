"""Merge the per-shard ``convacc_bench --json`` reports into one table.

The golden regression runs as a CI matrix: shard *i* writes
``golden-<grid>-shard<i>.json`` (currently ``1x`` and ``full``) with ``cases``,
``passed`` and ``failures``. Twenty jobs otherwise leave twenty logs, so this
folds them into the single line a reviewer wants -- "did the whole grid pass?"
-- and appends it to the job summary.

usage: merge_shard_reports.py [DIR]
"""

import glob
import json
import os
import sys


def merge(directory):
    grids = {}
    failures = []
    for path in sorted(glob.glob(os.path.join(directory, "golden-*.json"))):
        grid = os.path.basename(path)[len("golden-") : -len(".json")]
        grid = grid.rsplit("-shard", 1)[0]
        with open(path) as handle:
            data = json.load(handle)
        total, passed = grids.get(grid, (0, 0))
        grids[grid] = (total + data["cases"], passed + data["passed"])
        failures.extend(data["failures"])
    return grids, failures


def render(grids, failures):
    lines = ["| grid | cases | passed |", "|---|---:|---:|"]
    for grid in sorted(grids, key=lambda name: grids[name][0]):
        total, passed = grids[grid]
        lines.append("| `%s` | %s | %s |" % (grid, format(total, ","), format(passed, ",")))
    if failures:
        lines += ["", "Failed cases:", ""]
        lines += ["* `%s`: %s" % (f["name"], f["reason"]) for f in failures[:20]]
        if len(failures) > 20:
            lines.append("* ... %d more" % (len(failures) - 20))
    return "\n".join(lines)


def main(argv):
    directory = argv[1] if len(argv) > 1 else "shards"
    grids, failures = merge(directory)
    if not grids:
        print("no shard reports under %s" % directory)
        return 0
    text = render(grids, failures)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
