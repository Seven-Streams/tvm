# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Invoked by scikit-build-core (pip install / pip install -e). No C++ of our
# own — this file's whole job is to run cargo on rust/Cargo.toml and install
# the resulting libtvm_tirx.so into the wheel (tokenizers-cpp pattern, minus
# the static-lib indirection: cargo emits the final cdylib directly).
cmake_minimum_required(VERSION 3.18)
project(tirx_ext LANGUAGES NONE)

# --- Locate the libtvm_ffi.so to link against ------------------------------
# It must be the SAME libtvm_ffi.so the host process will load (the one
# libtvm_compiler.so was built against). Priority:
#   1. -DTIRX_LIB_DIR=... / env TIRX_LIB_DIR   (explicit)
#   2. <tir repo>/build/lib                    (in-repo development, default)
#   3. tvm-ffi-config --libdir                 (external: apache-tvm-ffi wheel)
if(NOT TIRX_LIB_DIR)
  if(DEFINED ENV{TIRX_LIB_DIR})
    set(TIRX_LIB_DIR "$ENV{TIRX_LIB_DIR}")
  else()
    get_filename_component(_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    if(EXISTS "${_repo_root}/build/lib/libtvm_ffi.so")
      set(TIRX_LIB_DIR "${_repo_root}/build/lib")
    else()
      execute_process(
        COMMAND tvm-ffi-config --libdir
        OUTPUT_VARIABLE TIRX_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _cfg_ret)
      if(NOT _cfg_ret EQUAL 0 OR TIRX_LIB_DIR STREQUAL "")
        message(FATAL_ERROR
          "tirx-ext: cannot locate libtvm_ffi.so — build tir first, "
          "pass -DTIRX_LIB_DIR=<dir>, or install apache-tvm-ffi")
      endif()
    endif()
  endif()
endif()
message(STATUS "tirx-ext: linking against libtvm_ffi in ${TIRX_LIB_DIR}")

# --- cargo build ------------------------------------------------------------
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_cargo_profile_dir debug)
  set(_cargo_profile_flag)
else()
  set(_cargo_profile_dir release)
  set(_cargo_profile_flag --release)
endif()
set(TVM_TIRX_SO "${CARGO_TARGET_DIR}/${_cargo_profile_dir}/libtvm_tirx.so")

# cargo owns incrementality, so simply run it on every build. The tools/ shim
# on PATH answers the `tvm-ffi-config --libdir` query of the transitive
# tvm-ffi-sys build script with TIRX_LIB_DIR.
add_custom_target(tvm_tirx ALL
  COMMAND ${CMAKE_COMMAND} -E env
      "PATH=${CMAKE_CURRENT_SOURCE_DIR}/tools:$ENV{PATH}"
      "CARGO_TARGET_DIR=${CARGO_TARGET_DIR}"
      "TIRX_LIB_DIR=${TIRX_LIB_DIR}"
      ${CARGO_EXECUTABLE} build ${_cargo_profile_flag}
      --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.toml"
  BYPRODUCTS "${TVM_TIRX_SO}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rust"
  COMMENT "cargo build (libtvm_tirx.so)"
  VERBATIM)

# Lands in site-packages/tirx_ext/ (wheel.install-dir in pyproject.toml).
install(PROGRAMS "${TVM_TIRX_SO}" DESTINATION .)
