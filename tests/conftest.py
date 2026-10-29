"""Session-level simulator build for the cocotb regression.

The model is prepared in ``pytest_sessionstart``, before any test runs. Fixtures
and tests never build -- the ``runner`` fixture only hands out an
already-initialised cocotb runner.

Both backends share one build directory per simulator, reused by every xdist
worker:

* ``verilator`` produces a read-only executable and `make` is incremental, so
  the C++ compile happens once for the whole run and later workers cost ~0.1s.
* ``xezim`` elaborates at run time; ``build()`` only records the command line in
  ``xezim_build.json`` and ``test()`` reloads it even when another runner wrote
  it, so a shared directory is safe.

A lock serialises the first build against concurrent workers.

Pick the backend with ``--sim {verilator,xezim}`` (or the ``CONVACC_SIM``
environment variable; the option wins).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))  # xezim_runner lives here
HDL_TOP = "convacc_top"
SV_SOURCES = [
    ROOT / "hw" / "convacc_top.sv",
    ROOT / "hw" / "convacc_mem.sv",
    ROOT / "hw" / "convacc_sram.sv",
]
XEZIM = Path(os.environ.get("XEZIM", str(Path.home() / "opt" / "xezim-bin" / "bin" / "xezim")))
SIMS = ("verilator", "xezim")
# Verilator is the default: xezim silently drops `reg arr[N] <= cond ? next : arr`
# (an unpacked array through a ternary) -- exactly what codegen emits for proc
# array state -- so every array-typed FSM field is lost under xezim.
DEFAULT_SIM = "verilator"


def pytest_addoption(parser):
    parser.addoption(
        "--sim",
        action="store",
        choices=SIMS,
        default=None,
        help=f"cocotb simulator backend (default: {DEFAULT_SIM}; "
             "also settable via CONVACC_SIM)",
    )


def _sim_name(config) -> str:
    return config.getoption("--sim") or os.environ.get("CONVACC_SIM") or DEFAULT_SIM


def _fingerprint() -> str:
    """Content identity of the RTL inputs: rebuild only when they change."""
    parts = []
    for path in SV_SOURCES:
        st = os.stat(path)
        parts.append(f"{path}:{st.st_mtime_ns}:{st.st_size}")
    return "\n".join(parts)


def _build(sim_name: str) -> tuple:
    os.environ["XEZIM_BIN"] = str(XEZIM)
    if sim_name == "xezim":
        import xezim_runner  # noqa: F401  registers the xezim cocotb backend
    from cocotb_tools.runner import get_runner

    build_dir = ROOT / "build" / "cocotb" / sim_name
    build_dir.mkdir(parents=True, exist_ok=True)
    sim = get_runner(sim_name)
    build_kwargs = {}
    if sim_name == "verilator":
        # XLS's generated code trips Verilator's width/expand lint on array
        # indices; harmless, so don't let warnings fail the build.
        build_kwargs["build_args"] = ["-Wno-WIDTHEXPAND", "-Wno-fatal"]

    import fcntl

    stamp = build_dir / ".build_stamp"
    want = _fingerprint()
    with open(build_dir / ".build_lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # serialise concurrent first builds
        fresh = stamp.is_file() and stamp.read_text() == want
        if fresh:
            # `sim.build()` also initialises the runner state that sim.test()
            # reads, so it must still run -- but the compile itself is already
            # done. Suppressing the commands keeps the 7 non-building xdist
            # workers from re-printing (and re-running) a no-op build.
            sim._execute = lambda *args, **kwargs: None
        try:
            sim.build(
                sources=[str(path) for path in SV_SOURCES],
                hdl_toplevel=HDL_TOP,
                build_dir=build_dir,
                clean=False,
                **build_kwargs,
            )
        finally:
            if fresh:
                del sim._execute  # restore the real one for sim.test()
        stamp.write_text(want)
    return sim, build_dir


def pytest_sessionstart(session):
    config = session.config
    # The xdist master only dispatches; it never runs a simulation.
    if config.pluginmanager.get_plugin("dsession") is not None and not hasattr(config, "workerinput"):
        return
    config._convacc_runner = _build(_sim_name(config))


@pytest.fixture(scope="session")
def runner(request):
    """(cocotb runner, build_dir) prepared by pytest_sessionstart."""
    return request.config._convacc_runner
