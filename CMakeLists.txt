## @file CMakeLists.txt
## @brief Root build configuration script for the KOptions option pricing engine.
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

cmake_minimum_required(VERSION 3.20)
project(KOptions VERSION 0.1.0
        DESCRIPTION "High-performance parallel Monte Carlo option pricing engine."
        HOMEPAGE_URL "https://github.com/zenogeddo/KOptions"
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

## @option KOPS_ENABLE_SINGLE_PRECISION
## @brief Activates 32-bit single precision (float) calculations instead of 64-bit double precision.
## @details Increases parallel memory bandwidth on GPU hardware, but reduces numeric precision.
option(KOPS_ENABLE_SINGLE_PRECISION "Use single precision (float) instead of double ?" OFF)

## @option KOPS_ENABLE_TESTS
## @brief Directs CMake to compile the unit test suite and configure CTest execution.
option(KOPS_ENABLE_TESTS "Build unit tests ?" ON)

## @option KOPS_ENABLE_FETCHCONTENT
## @brief Allows the build system to automatically download and compile missing dependencies from GitHub.
option(KOPS_ENABLE_FETCHCONTENT "Enable FetchContent ?" OFF)

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
    if (KOPS_ENABLE_FETCHCONTENT)
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
        message(FATAL_ERROR "System yaml-cpp not found and KOPS_ENABLE_FETCHCONTENT is OFF. "
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
    if (KOPS_ENABLE_FETCHCONTENT)
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
        message(FATAL_ERROR "Kokkos not found and KOPS_ENABLE_FETCHCONTENT is OFF. "
                "Please also provide -DKokkos_ROOT=/taget/kokkos/dir when running 'cmake -B build' or enable fetching."
                "Note: that '/taget/kokkos/dir' is the directory containing the `KokkosConfig.cmake` file, which is usually located in `/lib/cmake/Kokkos` inside your Kokkos installation directory."
        )
    endif ()
endif ()


# -----------------------------------------------------------------------------
# Interprocedural Optimization (LTO) Adaptive Safeguard
# Link-Time Optimization (LTO) -> During the final linking step, the compiler performs a secondary optimization pass looking at the entire project at once
# -----------------------------------------------------------------------------
# Query active accelerator backends inside Kokkos
if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    # GPU active: Force LTO OFF to prevent device-code 'fatbinData' linker collisions
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    message(STATUS "GPU backend active in Kokkos. Host Link-Time Optimization (LTO) disabled to prevent compilation conflicts.")
else()
    # CPU only: Turn LTO ON to maximize host-side optimization and cross-file inlining
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS "CPU-only target. Host Link-Time Optimization (LTO/IPO) is ENABLED.")
endif()


# -----------------------------------------------------------------------------
# 5. Build the Core Library (src)
# -----------------------------------------------------------------------------
## @section src_sec
## @brief Source Library Tree Assembly. Traverses the `src/` directory to build the core mathematical integration and model solvers.
add_subdirectory(src)


# -----------------------------------------------------------------------------
# 6. Build the App and Tests (Consumers)
# -----------------------------------------------------------------------------
## @section consumer_sec
## @brief Compilation of Consumers
## - Compiles the main user-facing executable interface in the `app/` subfolder.
## - If testing is active, initializes CTest pipelines and compiles the validation test binary.

add_subdirectory(app)
if (KOPS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Activated build in unit tests")
endif ()


# -----------------------------------------------------------------------------
# 7. Install useful info
# -----------------------------------------------------------------------------
## @section install_sec
## @brief Project Installation Execution. Collects system resources and copies user templates directly to the localized path hierarchy.
install(FILES README.md KOptionsConfig.yaml DESTINATION share)


# =============================================================================
# =============================================================================
# -----------------------------------------------------------------------------
#                                USAGE EXAMPLE
# -----------------------------------------------------------------------------
## @section usage_sec
## @brief Guide to building, installing, and executing the KOptions simulation suite.
## @details Below are the shell commands required to configure, build, and run the engine locally:
## @code{.sh}
## # 1) Configure the build workspace and isolate the installation target in a custom prefix directory:
## cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/KOps_dist .
## # Note : if -DCMAKE_INSTALL_PREFIX is omitted, the default install/ directory will be considered
##
## # 2) Compile all executable binaries and libraries utilizing parallel threads:
## cmake --build build -j 4
##
## # 3) Deploy libraries, configuration blueprints, and executables to the local prefix:
## cmake --install build     # (or: cd build && make install)
##
## # 4) Navigate into your compiled installation directory:
## cd ~/KOps_dist
##
## # 5) Copy the stock configuration YAML template to your active running directory:
## cp share/KOptions/KOptionsConfig.yaml .
##
## # 6) Execute the main option solver with your parameters:
## ./bin/KOptions KOptionsConfig.yaml
## @endcode
