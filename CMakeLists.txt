cmake_minimum_required(VERSION 3.20.0)

project(CombAddOptimizePass LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set LLVM/MLIR directory paths manually if not installed system-wide
# These should point to your LLVM build, not source
set(LLVM_DIR "../circt/llvm/build/lib/cmake/llvm")
set(MLIR_DIR "../circt/llvm/build/lib/cmake/mlir")
set(CIRCT_DIR "../circt/build/lib/cmake/circt")

# Find LLVM and MLIR packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM dir: ${LLVM_DIR}")
message(STATUS "MLIR dir: ${MLIR_DIR}")
message(STATUS "CIRCT dir: ${CIRCT_DIR}")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Needed for MLIR CMake macros
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
# include(AddLLVM)
# include(AddMLIR)
include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)
include(${CIRCT_CMAKE_DIR}/AddCIRCT.cmake)
include(HandleLLVMOptions)

set(COMBADDOPTIMIZE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(COMBADDOPTIMIZE_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(
  SYSTEM ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CIRCT_INCLUDE_DIRS}
  ${COMBADDOPTIMIZE_SOURCE_DIR}/include
  ${COMBADDOPTIMIZE_BINARY_DIR}/include
)
link_directories(${LLVM_BUILD_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)

# Build your pass as a shared library (plugin)
add_llvm_library(CombAddOptimize
  MODULE
  CombAddOptimize.cpp

  DEPENDS
  MLIRCombAddOptimizeIncGen

  LINK_LIBS
  MLIRIR
  MLIRRegisterAllPasses
  CIRCTSupport
  CIRCTComb
)

set_target_properties(CombAddOptimize PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

# Ensure we use the correct LLVM flags
llvm_update_compile_flags(CombAddOptimize)

