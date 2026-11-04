cmake_minimum_required(VERSION 3.20)
project(convacc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CONVACC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
get_filename_component(CONVACC_ROOT "${CONVACC_ROOT}" ABSOLUTE)

# ---------------------------------------------------------------------------
# What to build
#
# The tree is split the way the tests are: `model/` holds everything that runs
# on the C++ model alone, `dut/` everything that needs the simulator. Turning
# the DUT off is what makes the split real -- the model half then configures,
# builds and tests on a machine with no Verilator.
# ---------------------------------------------------------------------------
option(CONVACC_WITH_DUT "Build the DUT harness and the tests that drive it (needs Verilator)" ON)
option(CONVACC_WITH_GTEST "Build the GoogleTest suites" ON)
option(CONVACC_WITH_FUZZTEST "Build the FuzzTest suite (DUT)" OFF)
option(CONVACC_WITH_COVFUZZ "Build the coverage-guided fuzzer (model only)" OFF)
# The same grid as gtest tests, for measuring what a standard runner does with
# it -- see docs/native-tests.md. Not part of the normal build.
option(CONVACC_WITH_GRID_SHARDS "Build the gtest-sharded golden grid experiment" OFF)
# Path to a FuzzTest checkout. Empty means "fetch from GitHub", which also pulls
# abseil/re2/antlr -- expect a long first build.
set(CONVACC_FUZZTEST_SOURCE_DIR "" CACHE PATH "Existing FuzzTest source tree")
if(CONVACC_WITH_FUZZTEST AND NOT CONVACC_WITH_DUT)
  message(FATAL_ERROR "CONVACC_WITH_FUZZTEST simulates the DUT; set CONVACC_WITH_DUT=ON")
endif()

# ---------------------------------------------------------------------------
# Verilated model
#
# Built as a plain static archive (the `libVconvacc_top` target of the
# Verilator-generated makefile) so several test executables share one compile.
# `--public` keeps the hierarchy and only makes the signals public, which is all
# the harness needs to memcpy the SRAM arrays.
# ---------------------------------------------------------------------------
if(CONVACC_WITH_DUT)
  find_program(VERILATOR_BIN NAMES verilator
    HINTS /opt/verilator/bin /usr/local/bin ENV VERILATOR_ROOT)
  if(NOT VERILATOR_BIN)
    message(FATAL_ERROR "verilator not found; run `make rtl-gen` first and install Verilator 5.x")
  endif()

  execute_process(
    COMMAND "${VERILATOR_BIN}" --getenv VERILATOR_ROOT
    OUTPUT_VARIABLE VERILATOR_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_path(VERILATOR_INCLUDE_DIR verilated.h
    HINTS "${VERILATOR_ROOT}/include" /opt/verilator/share/verilator/include
          /usr/share/verilator/include)
  if(NOT VERILATOR_INCLUDE_DIR)
    message(FATAL_ERROR "verilated.h not found; is Verilator installed completely?")
  endif()

  set(SV_SOURCES
    "${CONVACC_ROOT}/hw/convacc_top.sv"
    "${CONVACC_ROOT}/hw/convacc_mem.sv"
    "${CONVACC_ROOT}/hw/convacc_sram.sv")

  set(VL_DIR "${CMAKE_BINARY_DIR}/verilated")
  set(VL_ARCHIVE "${VL_DIR}/libVconvacc_top.a")
  set(VL_OPT "-O3" CACHE STRING "Optimisation flag for the Verilated model")
  # Compiler for the generated model. The harness and the model are separate
  # translation units linked into one binary, so any C++ compiler works; clang
  # happens to emit much better code for Verilator's output (12.9 ms -> 8.1 ms
  # per dense case at -O3, measured on the 120-case dense sample). Empty means
  # "whatever Verilator's generated makefile defaults to" (g++).
  set(VL_CXX "" CACHE STRING "C++ compiler for the Verilated model (empty = Verilator default)")
  set(VL_CXX_FLAGS "" CACHE STRING "Extra flags for VL_CXX (e.g. --gcc-install-dir)")
  set(VL_JOBS "4" CACHE STRING "Parallel jobs for the Verilated model")
  # How the SRAM arrays are exposed to the harness. `--public` keeps the
  # hierarchy and only makes the signals public, which is all the harness needs
  # to memcpy the arrays; `--public-flat-rw` also flattens the design and costs
  # ~3x in eval time (measured: 15.8 us -> 4.7 us per clock). Keep the escape
  # hatch for the day a harness needs a flattened name.
  set(VL_PUBLIC_FLAG "--public" CACHE STRING "Verilator flag that exposes signals")

  # Verilator emits GNU-make syntax, so this cannot reuse CMAKE_MAKE_PROGRAM
  # when the outer build uses Ninja.
  find_program(CONVACC_MAKE NAMES gmake make REQUIRED)

  set(VL_MAKE_ARGS "")
  if(VL_CXX)
    list(APPEND VL_MAKE_ARGS "CXX=${VL_CXX} ${VL_CXX_FLAGS}")
  endif()

  # Verilator's output only depends on the SV sources, so a change of compiler
  # or flags would silently keep a stale archive. Record the configuration in a
  # stamp that is only rewritten when it actually changes, and make the archive
  # depend on it -- that re-runs Verilator exactly once per change.
  file(MAKE_DIRECTORY "${VL_DIR}")
  set(VL_CONFIG "${VL_DIR}/.vl_config")
  set(VL_CONFIG_CONTENT
      "cxx=${VL_CXX}\nflags=${VL_CXX_FLAGS}\nopt=${VL_OPT}\npublic=${VL_PUBLIC_FLAG}\n")
  if(EXISTS "${VL_CONFIG}")
    file(READ "${VL_CONFIG}" VL_CONFIG_OLD)
  else()
    set(VL_CONFIG_OLD "")
  endif()
  if(NOT "${VL_CONFIG_OLD}" STREQUAL "${VL_CONFIG_CONTENT}")
    file(WRITE "${VL_CONFIG}" "${VL_CONFIG_CONTENT}")
  endif()

  add_custom_command(
    OUTPUT "${VL_ARCHIVE}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VL_DIR}"
    COMMAND "${VERILATOR_BIN}" --cc --Mdir "${VL_DIR}" --top-module convacc_top
            --prefix Vconvacc_top -Wno-fatal -Wno-WIDTHEXPAND ${VL_PUBLIC_FLAG}
            ${SV_SOURCES}
    # OPT_* must be passed as make *command-line* variables. verilated.mk
    # assigns them unconditionally (`OPT_FAST = -Os`), and a makefile assignment
    # beats the environment, so `cmake -E env OPT_FAST=...` is silently ignored
    # and the hot functions stay at -Os.
    COMMAND "${CONVACC_MAKE}" -C "${VL_DIR}" -f Vconvacc_top.mk -j ${VL_JOBS}
            "OPT_FAST=${VL_OPT}" "OPT_SLOW=${VL_OPT}" "OPT_GLOBAL=${VL_OPT}"
            ${VL_MAKE_ARGS} libVconvacc_top
    DEPENDS ${SV_SOURCES} "${VL_CONFIG}"
    BYPRODUCTS "${VL_DIR}/libverilated.a"
    COMMENT "Verilating convacc_top (Verilator ${VERILATOR_BIN})"
    VERBATIM)

  add_custom_target(verilate DEPENDS "${VL_ARCHIVE}")
endif()

# ---------------------------------------------------------------------------
# Linker
#
# The executables are ~1 MB, most of it generated model code. mold links one in
# 0.02 s against BFD's 0.07 s and the result runs identically (measured: 7.05 vs
# 7.02 ms/case, inside the noise), so it is the default when installed. Two
# caveats: mold cannot read LLVM bitcode, so an LTO build must ask for
# `-DCONVACC_LINKER=lld`, and it needs `ld.mold` on PATH. Empty means "whatever
# the compiler driver picks", i.e. the old behaviour.
# ---------------------------------------------------------------------------
set(CONVACC_LINKER "mold" CACHE STRING
    "Linker for the native executables: mold, lld, bfd, or empty for the default")
if(CONVACC_LINKER)
  if(NOT CONVACC_LINKER STREQUAL "bfd")
    find_program(CONVACC_LD_BIN NAMES "${CONVACC_LINKER}" "ld.${CONVACC_LINKER}")
    if(NOT CONVACC_LD_BIN)
      message(WARNING "${CONVACC_LINKER} not found; linking with the toolchain default")
      set(CONVACC_LINKER "")
    endif()
  endif()
endif()
if(CONVACC_LINKER)
  add_link_options("-fuse-ld=${CONVACC_LINKER}")
  message(STATUS "Linker: ${CONVACC_LINKER} (${CONVACC_LD_BIN})")
else()
  message(STATUS "Linker: toolchain default")
endif()

# ---------------------------------------------------------------------------
# Model library: the C++ mirror of the Python reference model.
# ---------------------------------------------------------------------------
set(MODEL_SOURCES
  src/spec.cc
  src/planner.cc
  src/packer.cc
  src/image.cc
  src/reference.cc)

set(MODEL_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(MODEL_WARNINGS -Wall -Wextra -Wno-sign-compare)

add_library(convacc_model STATIC ${MODEL_SOURCES})
target_include_directories(convacc_model PUBLIC ${MODEL_INCLUDES})
target_compile_options(convacc_model PRIVATE ${MODEL_WARNINGS})

# ---------------------------------------------------------------------------
# DUT harness: Verilator reset/load/run/read, and the one place a case is timed.
# ---------------------------------------------------------------------------
if(CONVACC_WITH_DUT)
  add_library(convacc_dut STATIC dut/dut.cc)
  target_include_directories(convacc_dut PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${VL_DIR}"
    "${VERILATOR_INCLUDE_DIR}"
    "${VERILATOR_INCLUDE_DIR}/vltstd")
  target_link_libraries(convacc_dut PUBLIC convacc_model "${VL_ARCHIVE}"
    "${VL_DIR}/libverilated.a" pthread atomic)
  add_dependencies(convacc_dut verilate)
endif()

# ---------------------------------------------------------------------------
# Case codec + oracle: the fuzz format. Pure model, shared by the libFuzzer
# driver (model/) and the corpus replayer (dut/).
# ---------------------------------------------------------------------------
add_library(convacc_casecodec STATIC model/case_codec.cc model/oracle.cc)
target_include_directories(convacc_casecodec PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(convacc_casecodec PUBLIC convacc_model)
target_compile_options(convacc_casecodec PRIVATE -Wall -Wextra -Wno-sign-compare)

# ---------------------------------------------------------------------------
# GoogleTest / FuzzTest
# ---------------------------------------------------------------------------
if(CONVACC_WITH_FUZZTEST)
  # FuzzTest vendors abseil, re2, antlr and googletest through FetchContent, so
  # it has to be configured first: its googletest (built with GTEST_HAS_ABSL) is
  # the one we then link against. Declaring our own first would leave two gtest
  # targets and FuzzTest's would lose the absl dependencies.
  set(FUZZTEST_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(FUZZTEST_FUZZING_MODE OFF CACHE BOOL "" FORCE)
  set(FUZZTEST_BUILD_FLATBUFFERS OFF CACHE BOOL "" FORCE)
  if(CONVACC_FUZZTEST_SOURCE_DIR)
    add_subdirectory("${CONVACC_FUZZTEST_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_fuzztest"
                     EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(fuzztest
      GIT_REPOSITORY https://github.com/google/fuzztest.git
      GIT_TAG 2026-06-29
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(fuzztest)
  endif()
elseif(CONVACC_WITH_GTEST)
  include(FetchContent)
  set(GTEST_TAG "v1.17.0" CACHE STRING "googletest tag")
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG ${GTEST_TAG}
    GIT_SHALLOW TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

if(CONVACC_WITH_GTEST OR CONVACC_WITH_FUZZTEST)
  # Model first: this binary links no Verilator, so it is the one that runs
  # everywhere.
  add_executable(convacc_model_tests model/model_test.cc)
  target_include_directories(convacc_model_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(convacc_model_tests PRIVATE convacc_model GTest::gtest_main)
  target_compile_definitions(convacc_model_tests PRIVATE
    CONVACC_DATASET_DEFAULT="${CONVACC_ROOT}/build/cpp_cases/dataset.bin")
endif()

if(CONVACC_WITH_DUT)
  # Full golden regression, generated in-process (no corpora on disk).
  add_executable(convacc_bench dut/golden_main.cc dut/generator.cc)
  target_link_libraries(convacc_bench PRIVATE convacc_dut)

  # Replays a libFuzzer corpus on the real DUT. Lives in the uninstrumented tree
  # so it can link the ordinary Verilated model.
  add_executable(convacc_fuzz_replay dut/replay_main.cc)
  target_link_libraries(convacc_fuzz_replay PRIVATE convacc_casecodec convacc_dut)

  if(CONVACC_WITH_GTEST OR CONVACC_WITH_FUZZTEST)
    add_executable(convacc_dut_tests dut/dataset_test.cc)
    target_link_libraries(convacc_dut_tests PRIVATE convacc_dut GTest::gtest_main)
    target_compile_definitions(convacc_dut_tests PRIVATE
      CONVACC_DATASET_DEFAULT="${CONVACC_ROOT}/build/cpp_cases/dataset.bin")

    # The golden grid as gtest tests, one per chunk of cases. Off by default: it
    # was measured against the worker pool (docs/native-tests.md) and the pool
    # won, so it is kept as an experiment rather than as a third binary every
    # build has to link. See dut/grid_main.cc for why the granularity is a
    # chunk and not a case.
    if(CONVACC_WITH_GRID_SHARDS)
      add_executable(convacc_bench_gtest dut/grid_main.cc dut/generator.cc)
      target_link_libraries(convacc_bench_gtest PRIVATE convacc_dut GTest::gtest_main)
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Coverage-guided fuzzer (clang + libFuzzer)
#
# Only the model and the codec are instrumented: libFuzzer's feedback has to
# come from the code that actually decides what a case looks like (planner,
# packer, image builder, reference model). The DUT is replayed separately, so
# this target does not need Verilator at all.
# ---------------------------------------------------------------------------
if(CONVACC_WITH_COVFUZZ)
  set(COV_FLAGS -fsanitize=fuzzer-no-link -g)
  add_library(convacc_model_cov OBJECT ${MODEL_SOURCES})
  target_include_directories(convacc_model_cov PUBLIC ${MODEL_INCLUDES})
  target_compile_options(convacc_model_cov PRIVATE ${COV_FLAGS} -O2 ${MODEL_WARNINGS})

  add_library(convacc_casecodec_cov OBJECT model/case_codec.cc model/oracle.cc)
  target_include_directories(convacc_casecodec_cov PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_compile_options(convacc_casecodec_cov PRIVATE ${COV_FLAGS} -O2 -Wall -Wextra
    -Wno-sign-compare)

  add_executable(convacc_covfuzz model/fuzz_main.cc
    $<TARGET_OBJECTS:convacc_model_cov>
    $<TARGET_OBJECTS:convacc_casecodec_cov>)
  target_include_directories(convacc_covfuzz PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_options(convacc_covfuzz PRIVATE -fsanitize=fuzzer)
  # The corpus lives outside the build tree so it can be committed as a seed.
  target_compile_definitions(convacc_covfuzz PRIVATE
    CONVACC_CORPUS_DEFAULT="${CONVACC_ROOT}/build/covfuzz-corpus")
endif()

if(CONVACC_WITH_FUZZTEST)
  add_executable(convacc_fuzz dut/fuzz_test.cc)
  target_link_libraries(convacc_fuzz PRIVATE
    convacc_dut
    fuzztest::fuzztest
    fuzztest::fuzztest_gtest_main)
endif()
