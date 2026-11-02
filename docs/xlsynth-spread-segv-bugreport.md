# Upstream bug report draft: DSLX interpreter SIGSEGV on generic call-spread

> **Status:** Draft, NOT yet submitted upstream. Recorded on 2026-11-02.
> Minimal repro saved at `/tmp/minrepro/generic_spread.x`.

## Summary

`dslx_interpreter_main` (and `prove_quickcheck_main`) crashes with **SIGSEGV (exit 139)**,
no output, when a struct-literal **spread** (`..base`) uses a **generic function call**
as its base expression.

The crash happens during module load / interpretation — before any test output is
produced. It is a **toolchain bug**, not a source bug: the DSLX code is valid.

## Affected toolchains

| Toolchain | Binary | Result |
|-----------|--------|--------|
| xlsynth-bin (symlink → 0.61.0) | `libexec/xlsynth/dslx_interpreter_main` | SIGSEGV 139 |
| xlsynth-bin-0.61.0 | `libexec/xlsynth/dslx_interpreter_main` | SIGSEGV 139 |

**IMPORTANT — version correction:** `xlsynth-bin` is a **symlink** to
`xlsynth-bin-0.61.0`, so both rows above are the **same 0.61.0 build**. There is
**no 0.59.x toolchain installed locally**, so the bug could NOT be tested against
0.59. An earlier note claiming "reproduced on both 0.59.0.1 and 0.61.0" was
**wrong** — 0.59.0.1 does not exist on this machine, and the two paths tested
were in fact the same 0.61.0 binary.

Status: confirmed on the only locally-available toolchain (**0.61.0**). 0.59.x
status is **UNKNOWN** (not reproducible here).

## Minimal reproduction

```dslx
struct S<M: u32> { a: u32[M], b: u32[M] }
fn mk<M: u32>() -> S<M> { S { a: zero!<u32[M]>(), b: zero!<u32[M]>() } }
#[test]
fn t() -> () { let s = S<u32:4> { ..mk<u32:4>() }; assert_eq(s.a[u32:0], u32:0); }
```

Command:

```
dslx_interpreter_main --dslx_stdlib_path=<stdlib> generic_spread.x
# -> exit 139, no stdout
```

## What does / does not crash (isolation matrix)

| Construct | Crash? |
|-----------|--------|
| `..mk<u32:4>()` — spread of a **generic** function call | **YES (139)** |
| `..mkp()` — spread of a **non-generic** function call | No |
| `..base` — spread of a **variable** holding the call result | No |
| `zero!<...>()`, `update(...)`, `std::ceil_div(...)` used alone | No |

So the precise trigger is: **struct-literal spread whose base is a generic
function call** (independent of struct size — also reproduces with a 2-field struct).

## Workaround (already applied locally)

Bind the generic construction to a local variable first, then spread the variable:

```dslx
// before (crashes):
let st = Engine<u32:16> { cfg: ..., kb: u32:1, ..zero_engine<u32:16>() };
// after (works):
let ze = zero_engine<u32:16>();
let st = Engine<u32:16> { cfg: ..., kb: u32:1, ..ze };
```

This is a pure refactor — semantics are identical. In our `convacc_proc.x` there were
**10** such `..zero_engine<M>()` occurrences (lines ~847/858/868/1024/1100/1211/1251/
1298/1362/1418), all converted. After the change `make dslx-test-convacc` runs
**23 tests, 0 failed**.

The same bug also appears in **qualified** form in `convacc_proofs.x`:
`..convacc_proc::zero_engine<u32:32>()` (one occurrence, line ~60). The same
variable-binding workaround fixes it, and `dslx-prove-convacc` then runs clean.

**Total occurrences fixed: 11 (10 in convacc_proc.x + 1 in convacc_proofs.x).**

## Notes for the upstream report

- Component: XLS / DSLX tree-walking interpreter (`dslx_interpreter_main`,
  `prove_quickcheck_main`). Likely a stack-overflow or null deref while evaluating
  the spread of a parametric/instantiated function call.
- `lldb` on the ~100 MB Mach-O binary hangs on symbol loading; the crash produces
  no macOS crash report (dies before the reporter fires). A minimal repro is the
  fastest way for upstream to bisect.
- `std::ceil_div` and `update` are unrelated — do not blame them.
- No existing XLS issue matched this pattern at time of writing (2026-11-02).
