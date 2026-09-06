# Copyright (C) 14/07/2026 Zeno GEDDO <zeno.geddo@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


## @file
## @brief Root build configuration script for the QuantKos option pricing engine.
## @details This file defines global build parameters, compiler standards, user options,
##          installation layouts, and executes the "Find-or-Fetch" dependency resolver for yaml-cpp and Kokkos.
## @author Zeno GEDDO
## @date 2026


# -----------------------------------------------------------------------------
# 1. Project Configuration (Resuirements)
# -----------------------------------------------------------------------------
## @section project_requirements
## @brief Defines minimum build environment requirements and system language standards.
## - **CMake Version**: Require 3.20+.
## - **Language Standard**: Strict C++20 without compiler extensions.
## @note **Compiler Portability Statement**:
##       This engine is written in standard-compliant C++20 with
##       compiler extensions disabled (`CMAKE_CXX_EXTENSIONS OFF`).
##       - **Fully Tested & Verified**: GCC (Version 11.2 and above).
##       - **Expected Compatible**: Clang (Version 13+) and MSVC (2022+).
##         While these toolchains conform to the standards used, they are
##         currently untested.

cmake_minimum_required(VERSION 3.25)
project(QuantKos VERSION 0.1.0
        DESCRIPTION "High-performance, platform independent parallel Monte Carlo option pricing engine."
        HOMEPAGE_URL "https://github.com/zenogeddo/QuantKos"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# 2. Build Type Configuration (Default, Release, RelWithDebInfo)
# -----------------------------------------------------------------------------
## @section build_type_configuration
## @variable CMAKE_BUILD_TYPE
## @brief Specifies compiler optimizations. Defaults to "Release" (-O3) if left empty.
## @details Valid build targets:
## - **Release**: Maximized execution speed with optimized compiler pipelines.
## - **Debug**: Full debugging symbols and no optimizations.
## - **RelWithDebInfo**: Release + active debugging symbols.

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to 'Release'")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE) # Force to apply changes without clearing the cache
endif ()

# Create a drop-down list of options for ccmake / cmake-gui
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "RelWithDebInfo") #"MinSizeRel"


# -----------------------------------------------------------------------------
# 2.5 Default Installation Prefix (Workspace Isolation)
# -----------------------------------------------------------------------------
## @section installation_prefix
## @brief Target Destination Setup, Configures where compiled libraries, binaries, and templates are saved.
## @details Bypasses standard system directories in favor of a local directory named `install/`
##          at the project root. This ensures users can compile and deploy without administrator or sudo privileges.

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "No install prefix selected. Defaulting to local 'install' directory.")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Default install path" FORCE)
endif ()

# -----------------------------------------------------------------------------
# 3. Options
# -----------------------------------------------------------------------------
## @section build_options
## @brief User-Configurable Build Options. Use these toggles to customize the precision type, testing suite, and dependency download policies.

## @option QKOS_ENABLE_SINGLE_PRECISION
## @brief Activates 32-bit single precision (float) calculations instead of 64-bit double precision.
## @details Increases parallel memory bandwidth on GPU hardware, but reduces numeric precision.
option(QKOS_ENABLE_SINGLE_PRECISION "Use single precision (float) instead of double ?" OFF)
if (QKOS_ENABLE_SINGLE_PRECISION)
    add_compile_definitions(QKOS_ENABLE_SINGLE_PRECISION) # Define the preprocessor macro so that float can be actually used
    message(STATUS "WARNING: Using single precision for the computations may currently cause a degradation in the result quality!")
endif ()

## @option QKOS_ENABLE_FAST_MATH
## @brief Enable aggressive non-IEEE fast-math optimizations
option(QKOS_ENABLE_FAST_MATH "Enable aggressive non-IEEE compliant fast-math optimizations" OFF)
if (QKOS_ENABLE_FAST_MATH)
    add_compile_definitions(QKOS_ENABLE_FAST_MATH) # Define the preprocessor macro so that float can be actually used
    message(STATUS "WARNING: Enable aggressive non-IEEE fast-math optimizations!")
endif ()

## @option QKOS_ENABLE_CPU_OPT_REPORTS
## @brief Enable generating text files showing the inlining and vectorization performed
option(QKOS_ENABLE_CPU_OPT_REPORTS "Enable compiler optimization reports (inlining/vectorization)" OFF)

## @option QKOS_ENABLE_TESTS
## @brief Directs CMake to compile the unit test suite and configure CTest execution.
option(QKOS_ENABLE_TESTS "Build unit tests ?" ON)

## @option QKOS_ENABLE_FETCHCONTENT
## @brief Allows the build system to automatically download and compile missing dependencies from GitHub.
option(QKOS_ENABLE_FETCHCONTENT "Enable FetchContent ?" OFF)

## @option QKOS_BUILD_DOCS
## @brief Generates HTML documentation using Doxygen during the build process.
option(QKOS_BUILD_DOCS "Generate Doxygen documentation ?" ON)


# -----------------------------------------------------------------------------
# 4. Find Dependencies (Find-or-Fetch)
# -----------------------------------------------------------------------------
include(FetchContent)
## @section dependency_resolution
## @brief External Dependencies Resolution. Employs a robust search-or-download hierarchy to obtain necessary parsing and execution frameworks.
include(FetchContent)

# --- YAML-CPP ---
## @subsection yaml_cpp_subsec
## @brief Dependency: yaml-cpp. Library used to parse configuration files (.yaml).
## @note If missing locally and fetch mode is active, it pulls clean tag `0.8.0` from GitHub,
## turning off internal tests and tools to speed up overall compilation times.
find_package(yaml-cpp QUIET)

if (yaml-cpp_FOUND)
    message(STATUS "Found system yaml-cpp")
else ()
    if (QKOS_ENABLE_FETCHCONTENT)
        message(STATUS "yaml-cpp not found. Fetching from GitHub...")

        # Disable yaml-cpp tests/tools to prevent massive compile bloat
        set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
        set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
        set(YAML_CPP_INSTALL OFF CACHE BOOL "Disable yaml-cpp installation" FORCE)

        FetchContent_Declare(
                yaml-cpp
                GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
                GIT_TAG 0.8.0
        )
        FetchContent_MakeAvailable(yaml-cpp)
    else ()
        # Users must provide the path via : -Dyaml-cpp_ROOT=/path/to/yaml
        message(FATAL_ERROR "System yaml-cpp not found and QKOS_ENABLE_FETCHCONTENT is OFF. "
                "Please also provide -Dyaml-cpp_ROOT=/path/to/yaml when running 'cmake -B build' or enable fetching.")
    endif ()
endif ()


# --- KOKKOS ---
## @subsection kokkos_subsec
## @brief Dependency: Kokkos. Multi-threaded execution and performance portability model targeting CPU and GPU execution spaces.
## @note If missing locally and fetch mode is active, pulls stable tag `4.2.00` from GitHub.
find_package(Kokkos QUIET)
find_package(Kokkos QUIET)

if (Kokkos_FOUND)
    message(STATUS "Found Kokkos at: ${Kokkos_DIR}")
else ()
    if (QKOS_ENABLE_FETCHCONTENT)
        message(STATUS "Kokkos not found. Fetching from GitHub...")

        # Disable Kokkos tests/examples to save minutes of compile time
        set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Disable Kokkos tests" FORCE)
        set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Kokkos examples" FORCE)

        FetchContent_Declare(
                kokkos
                GIT_REPOSITORY https://github.com/kokkos/kokkos.git
                GIT_TAG 4.2.00
        )

        # --- KOKKOS OPTIONS --- (should be added)
        # These ensure Kokkos builds with the right backend for your machine.
        # set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
        # set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(kokkos)
    else ()
        # Users must provide the path via : -DKokkos_ROOT=/path/to/install
        message(FATAL_ERROR "Kokkos not found and QKOS_ENABLE_FETCHCONTENT is OFF. "
                "Please also provide -DKokkos_ROOT=/taget/kokkos/dir when running 'cmake -B build' or enable fetching."
                "Note: that '/taget/kokkos/dir' is the directory containing the `KokkosConfig.cmake` file, which is usually located in `/lib/cmake/Kokkos` inside your Kokkos installation directory."
        )
    endif ()
endif ()


# -----------------------------------------------------------------------------
# Interprocedural Optimization (LTO) Adaptive Safeguard
# Link-Time Optimization (LTO) -> During the final linking step, the compiler performs a secondary optimization pass looking at the entire project at once
# -----------------------------------------------------------------------------
# Query active accelerator target_backends inside Kokkos
if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    # GPU active: Force LTO OFF to prevent device-code 'fatbinData' linker collisions
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    message(STATUS "GPU backend active in Kokkos. Host Link-Time Optimization (LTO) disabled to prevent compilation conflicts.")
else ()
    # CPU only: Check if the compiler actually supports IPO/LTO
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)

    if (ipo_supported)
        # CPU only: Turn LTO ON to maximize host-side optimization and cross-file inlining
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "CPU-only target. Host Link-Time Optimization (LTO/IPO) is ENABLED.")
    else ()
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
        message(WARNING "CPU-only target, but compiler does NOT support IPO/LTO. Disabled. Details: ${ipo_output}")
    endif ()
endif ()


# -----------------------------------------------------------------------------
# 5. Build the Core Library (src)
# -----------------------------------------------------------------------------
## @section src_sec
## @brief Source Library Tree Assembly. Traverses the `src/` directory to build the core mathematical integration and model solvers.
add_subdirectory(src)

# Pass active configuration name to C++ preprocessor
target_compile_definitions(QuantKos_core PUBLIC
        QKOS_BUILD_TYPE="$<CONFIG>"
)

# -----------------------------------------------------------------------------
# 6. Build the App and Tests (Consumers)
# -----------------------------------------------------------------------------
## @section consumer_sec
## @brief Compilation of Consumers
## - Compiles the main user-facing executable interface in the `app/` subfolder.
## - If testing is active, initializes CTest pipelines and compiles the validation test binary.

add_subdirectory(app)
if (QKOS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Activated build in unit tests")
endif ()

# -----------------------------------------------------------------------------
# 6.5. Doxygen Documentation Generation
# -----------------------------------------------------------------------------
## @section doc_sec 6.5 CODE Documentation Setup
## @brief Automatically generates and installs Doxygen HTML documentation.
if (QKOS_BUILD_DOCS)
    find_package(Doxygen QUIET) # silently sets a variable (DOXYGEN_FOUND) to FALSE if not found
    if (DOXYGEN_FOUND)
        message(STATUS "Doxygen found: Documentation target 'documentation' enabled.")

        # Define documentation directory in the BUILD tree, not the SOURCE tree to avoid pollution
        set(DOC_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/documentation")

        # Tell CMake to watch your original Doxyfile for any manual edits
        # (If that file is changed, CMake will automatically re-run the configuration step the next time you trigger a build.)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "Doxyfile")

        # Read the original Doxyfile, append the output override, and save to the build tree
        file(READ "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile" DOXYFILE_CONTENTS) # Read file and store it in DOXYFILE_CONTENTS
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.build"
                "${DOXYFILE_CONTENTS}\n# --- CMAKE OVERRIDES ---\nOUTPUT_DIRECTORY = \"${DOC_OUTPUT_DIR}\"\n")


        # Create a custom target that runs Doxygen (ALL makes it build automatically with 'make')
        add_custom_target(docs # This creates a new target named docs that can be built by running make docs
                ALL # include docs target in the default "all" build, documentation is generated automatically every time
                COMMAND ${CMAKE_COMMAND} -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/doxygen.log" # remove log file if any so that the next one will be clean
                COMMAND ${CMAKE_COMMAND} -E remove_directory "${DOC_OUTPUT_DIR}" # remove dir so that docs will be clean next time
                COMMAND ${CMAKE_COMMAND} -E make_directory ${DOC_OUTPUT_DIR} # (The -E flag in CMake stands for "Execute" mode.)
                COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.build" > "${CMAKE_CURRENT_BINARY_DIR}/doxygen.log" 2>&1 # the actual terminal command to run
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} # This tells the command where to execute
                COMMENT "Generating code documentation in ${DOC_OUTPUT_DIR}..."
                VERBATIM # It ensures that any special characters in the command line arguments are escaped correctly for the specific build tool
        )

        # Install the generated HTML documentation to the install prefix
        install(DIRECTORY ${DOC_OUTPUT_DIR}/ # Dir to copy
                DESTINATION share/documentation # where the files should land in the install tree
                OPTIONAL # It tells CMake not to produce an error if the directory doesn't exist yet
        )
    else ()
        message(WARNING "QKOS_BUILD_DOCS is ON, but Doxygen was not found! Please install Doxygen.")
    endif ()
endif ()

# -----------------------------------------------------------------------------
# 8. Install useful info
# -----------------------------------------------------------------------------
## @section install_sec
## @brief Project Installation Execution. Collects system resources and copies user templates directly to the localized path hierarchy.
install(FILES README.md QuantKosConfig.yaml DESTINATION share)


# =============================================================================
# =============================================================================
# -----------------------------------------------------------------------------
#                                USAGE EXAMPLE
# -----------------------------------------------------------------------------
## @section usage_example
## @brief Guide to building, installing, and executing the QuantKos simulation suite.
## @details Below are the shell commands required to configure, build, and run the engine locally:
## @code{.sh}
## # 1) Configure the build workspace and isolate the installation target in a custom prefix directory:
## cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/QKOS_dist .
## # Note : if -DCMAKE_INSTALL_PREFIX is omitted, the default install/ directory will be considered
##
## # 2) Compile all executable binaries and libraries utilizing parallel threads:
## cmake --build build -j 4
##
## # 3) Deploy libraries, configuration blueprints, and executables to the local prefix:
## cmake --install build     # (or: cd build && make install)
##
## # 4) Navigate into your compiled installation directory:
## cd ~/QKOS_dist
##
## # 5) Copy the stock configuration YAML template to your active running directory:
## cp share/QuantKos/QuantKosConfig.yaml .
##
## # 6) Execute the main option solver with your parameters:
## ./bin/QuantKos QuantKosConfig.yaml
## @endcode
