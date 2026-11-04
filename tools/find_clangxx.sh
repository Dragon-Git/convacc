#!/bin/sh
# Print a clang++ command line that can compile C++17, or nothing.
#
# The Verilated model is the hot loop of the native harness, and clang compiles
# it ~1.6x better than gcc (7.0 vs 11.1 ms per case at -O3, see
# docs/native-tests.md). The catch is that clang on Debian/Ubuntu may pick a gcc
# installation whose libstdc++ headers are not installed -- in that case point
# it at the gcc that owns libgcc.
#
# Newest first: a distro install leaves an older clang++ in /usr/bin while a
# newer toolchain sits next to it as clang++-23, clang++-22, ... The first
# candidate that passes the probe wins, so this picks the best compiler the
# machine has instead of the oldest one.
#
# Used by the Makefile as `NATIVE_VL_CXX ?= $(shell sh tools/find_clangxx.sh)`;
# override it with `make ... NATIVE_VL_CXX=` to stay on Verilator's default g++.

probe() {
    printf '#include <vector>\n#include <cstdint>\nint main(){std::vector<int> v;return (int)v.size();}\n' |
        "$@" -x c++ -std=c++17 -fsyntax-only - >/dev/null 2>&1
}

gccdir=$(dirname "$(gcc -print-libgcc-file-name 2>/dev/null)" 2>/dev/null)

for base in clang++-23 clang++-22 clang++-21 clang++-20 clang++-19 clang++-18 clang++; do
    command -v "$base" >/dev/null 2>&1 || continue
    if probe "$base"; then
        echo "$base"
        exit 0
    fi
    if [ -n "$gccdir" ] && probe "$base" "--gcc-install-dir=$gccdir"; then
        echo "$base --gcc-install-dir=$gccdir"
        exit 0
    fi
done
