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
## @brief Core library compilation setup, target dependencies, and compiler optimization flag matrices.
## @details This file defines the core 'QuantKos_core' library target, configures precision compiler definitions,
##          binds package dependencies, and manages compiler and linker flags for multiple build configurations (Release, Debug, etc.).
## @author Zeno GEDDO
## @note More info about the flags on : https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
## @date 2026

# -----------------------------------------------------------------------------
# 1. CREATE TARGETS
# -----------------------------------------------------------------------------
# @section target_sec 1. Core Target Definition
## @brief Registers the primary library target.
## @details Compiles the main core configuration parser and future engine files to build
##          the library object 'QuantKos_core'.
add_library(QuantKos_core
        IO/ConfigParser.cpp
)


# -----------------------------------------------------------------------------
# 2. APPLY OPTIONS
# -----------------------------------------------------------------------------
## @section def_sec 2a. Precision Setup
## @brief Controls internal float/double type matching based on user toggles.
## @details Inspects the user build option `QKOS_ENABLE_SINGLE_PRECISION`. If enabled, it sets
##          the compiler flag `QKOS_USE_SINGLE_PRECISION` publicly, ensuring correct 32-bit float matching
##          across all dependent application files.

if (QKOS_ENABLE_SINGLE_PRECISION)
    target_compile_definitions(QuantKos_core PUBLIC QKOS_USE_SINGLE_PRECISION)
    message(STATUS "Precision: SINGLE (float)")
else ()
    message(STATUS "Precision: DOUBLE (double)")
endif ()

## @section def_sec 2b. FastMath Setup
## @brief Controls fast math setups based on user toggles.
## @details Inspects the user build option `QKOS_ENABLE_FAST_MATH`.
## If enabled, fast math flags are included during compilation.

if (QKOS_ENABLE_FAST_MATH)
    message(STATUS "Math Mode: FAST-MATH ENABLED (Aggressive Floating Points optimizations active)")
    target_compile_definitions(QuantKos_core PUBLIC QKOS_USE_FAST_MATH)

    # 1. Host Compiler Fast-Math Flags (GCC / Clang / AppleClang)
    target_compile_options(QuantKos_core PUBLIC
            $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
            -ffast-math
            >
    )

    # 2. GPU Compiler Fast-Math Flags (CUDA / NVCC)
    # Passed when Kokkos CUDA backend is active (works with nvcc_wrapper or native CUDA language)
    if (Kokkos_ENABLE_CUDA)
        message(STATUS "  └--> Applying CUDA GPU fast-math (--use_fast_math)")
        target_compile_options(QuantKos_core PUBLIC
                $<$<BOOL:${Kokkos_ENABLE_CUDA}>:SHELL:--use_fast_math>
        )
    endif()

    # 3. GPU Compiler Fast-Math Flags (AMD / HIP)
    if (Kokkos_ENABLE_HIP)
        message(STATUS "  └--> Applying AMD / HIP GPU fast-math (-ffast-math)")
        target_compile_options(QuantKos_core PUBLIC
                $<$<BOOL:${Kokkos_ENABLE_HIP}>:-ffast-math>
        )
    endif()
else()
    message(STATUS "Math Mode: STRICT IEEE preserved (aggressive FAST-MATH disabled)")
endif()


# -----------------------------------------------------------------------------
# 3. LINKING & INCLUDES
# -----------------------------------------------------------------------------
## @section link_sec 3. Upstream Dependency & Include Routing
## @brief Links target library dependencies and registers include search paths.
## @details
## - **target_link_libraries**: Links 'QuantKos_core' publicly to Kokkos and yaml-cpp, allowing any executable
##   downstream to automatically find and link these frameworks.
## - **target_include_directories**: Employs CMake generator expressions to isolate development-time headers
##   (`$<BUILD_INTERFACE>`) from installed distribution headers (`$<INSTALL_INTERFACE>`).

target_link_libraries(QuantKos_core PUBLIC
        Kokkos::kokkos
        yaml-cpp
        # ....
)
target_include_directories(QuantKos_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)


# -----------------------------------------------------------------------------
# 4. DEFAULT COMPILER FLAGS (Base Warnings, always ON for GCC/Clang despite the optimizations)
# -----------------------------------------------------------------------------
## @section warning_sec 4. Default Static Analysis Warnings
## @brief Enforces clean, robust code quality rules across GCC, Clang, and AppleClang.
## @details Applies static checking flags to prevent common C++ bugs:
## - `-Wall` & `-Wextra`: Turns on standard and advanced compiler diagnostic warnings.
## - `-Wpedantic`: Rejects non-standard compiler-specific C++ extensions.
## - `-Wshadow`: Triggers warnings when local variables shadow outer scope declarations.
## - `-Wnon-virtual-dtor`: Flags classes with virtual methods that are missing virtual destructors.

target_compile_options(QuantKos_core PRIVATE
        # Configuration for GNU GCC
        #$<$<CXX_COMPILER_ID:GNU>:
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
        -Wall                       # Enable most commonly used warning messages
        -Wextra                     # Enable additional warnings not covered by -Wall
        -Wpedantic                  # Strictly enforce ISO C++ standards (rejects non-standard extensions)
        -Wshadow                    # Warn when a variable shadows another variable in an outer scope
        -Wnon-virtual-dtor          # Warn if a class with virtual functions has a non-virtual destructor
        >

)

# -----------------------------------------------------------------------------
# 5. GLOBAL PUBLIC MATH OPTIMIZATIONS (optimizations that needs to propagate downstream!)
# -----------------------------------------------------------------------------
## @section public_opt_sec 3.5. Public Optimization Flags
## @brief Propagates critical performance assumptions to downstream consumers (app & tests).
## @details Applies math optimization flags publicly to ensure identical compilation assumptions
##          between our library objects and consuming binaries.
## - **-fno-math-errno**: Disables POSIX math 'errno' tracking, unleashing hardware vectorization (SIMD)
##   and allowing seamless inlining across files during LTO.
target_compile_options(QuantKos_core PUBLIC
        $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:
        -fno-math-errno # Disables updating 'errno' after math calls; allows faster SIMD math instructions
        >
)

# -----------------------------------------------------------------------------
# 6. DEBUG COMPILER FLAGS (Only active in Debug build : -DCMAKE_BUILD_TYPE=Debug)
# -----------------------------------------------------------------------------
## @section debug_sec 6. Debug Mode Configurations
## @brief Optimization and runtime sanitization configurations for Debug targets.
## @details Sets optimization to `-O0` to preserve execution structure during step-by-step debugging.
target_compile_options(QuantKos_core PRIVATE
        $<$<CONFIG:Debug>:
        -O0                         # Disable optimizations for accurate step-through debugging
        -g                          # Include debug symbols for GDB/LLDB/CLion
        -fno-omit-frame-pointer     # Keep frame pointer for cleaner stack traces
        >
)

## @subsection debug_san_subsec Adaptive Debug Sanitizers (PUBLIC Propagation)
## @brief Conditionally configures compiler and linker instrumentation for sanitizers.
## @details Host-side sanitizers (ASan/UBSan) are highly beneficial on CPU backends.
##          However, on GPU-accelerated backends (CUDA/HIP/SYCL), ASan's massive virtual shadow memory allocation
##          conflicts with the GPU driver's virtual memory mapping tables, triggering a critical initialization
##          out-of-memory error.
##          To prevent this, the build system queries the active Kokkos hardware configurations and
##          automatically bypasses host sanitization if a GPU-bound accelerator target is active.

# 1. Query active hardware targets exported by the Kokkos package
set(QKOS_SAFE_TO_USE_SANITIZERS ON)
if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    set(QKOS_SAFE_TO_USE_SANITIZERS OFF)
    message(STATUS "GPU backend active in Kokkos (CUDA/HIP/SYCL). Host-side AddressSanitizer disabled to prevent memory initialization conflicts.")
endif()

# 2. Apply sanitizer flags only if we are compiling for a CPU-only architecture
if (QKOS_SAFE_TO_USE_SANITIZERS)
    message(STATUS "CPU-only build target. Enabling AddressSanitizer and UndefinedBehaviorSanitizer for Debug builds.")

    target_compile_options(QuantKos_core PUBLIC
            $<$<CONFIG:Debug>:
            -fsanitize=address          # ASan: Detects out-of-bounds access, use-after-free, and memory leaks instantly.
            -fsanitize=undefined        # UBSan: Catches undefined behavior like integer overflow, null pointer dereferencing, or division by zero.
            >
    )

    # Linker flags: Required to link the ASan/UBSan runtime libraries into the final executable.
    target_link_options(QuantKos_core PUBLIC
            $<$<CONFIG:Debug>:
            -fsanitize=address          # Link AddressSanitizer runtime
            -fsanitize=undefined        # Link UndefinedBehaviorSanitizer runtime
            >
    )
endif()

# -----------------------------------------------------------------------------
# 7. RELEASE COMPILER FLAGS (Only active in Release build : -DCMAKE_BUILD_TYPE=Release)
# -----------------------------------------------------------------------------
## @section release_sec 6. Production Release Optimization Configuration
## @brief High-throughput production compilation rules active under `-DCMAKE_BUILD_TYPE=Release`.
## @details Maximizes runtime execution performance:
## - `-O3`: Activates loop vectorization, automatic inlining, and registers.
## - `-march=native`: Inspects and exploits instruction sets (like AVX-512) native to the compilation host.
## - `-flto`: Link Time Optimization; performs cross-file optimizations across files.
## - `-fno-math-errno`: Bypasses updating POSIX error variables after math commands, enabling SIMD execution. (set before in the global optimizations)
target_compile_options(QuantKos_core PRIVATE
        $<$<CONFIG:Release>:
        -O3                         # High-level optimization: vectorization, loop unrolling, and inlining
        -march=native               # Tune code for the specific CPU features (AVX, AVX-512) of the host machine
        -flto                       # Link Time Optimization: allows optimizations across different translation units
        >
)

# Linker flags for LTO (Link Time Optimization)
target_link_options(QuantKos_core PRIVATE
        $<$<CONFIG:Release>:
        -flto                       # Required during linking to perform cross-file optimizations
        -O3                         # Ensures the linker uses the same optimization level as the compiler
        >
)

# -----------------------------------------------------------------------------
# 8. RELWITHDEBINFO COMPILER FLAGS (-DCMAKE_BUILD_TYPE=RelWithDebInfo)
# Production-level speed, but with a map attached.
# For HPC, we want the exact same aggressive speed as Release, but WITH debug symbols for profiling.
# -----------------------------------------------------------------------------
## @section reldeb_sec 6.5 Profiling Optimization Configuration
## @brief Production performance with debugging symbols active under `-DCMAKE_BUILD_TYPE=RelWithDebInfo`.
## @details Maintains the exact performance optimizations of `-O3` and LTO, but compiles in debug
##          mapping coordinates (`-g`) and frame pointers to support profiling tools (e.g., perf, VTune, Valgrind).
target_compile_options(QuantKos_core PRIVATE
        $<$<CONFIG:RelWithDebInfo>:
        -O3                         # Keep maximum speed
        -march=native               # Keep vectorization
        -flto                       # Keep LTO
        -g                          # ADD debug symbols for profiling (perf, VTune, Valgrind)
        -fno-omit-frame-pointer     # Keep the call stack clean so the profiler can read it
        >
)

target_link_options(QuantKos_core PRIVATE
        $<$<CONFIG:RelWithDebInfo>:
        -flto                       # Required for LTO
        -O3                         # Ensures the linker uses the same optimization level as the compiler
        >
)


# -----------------------------------------------------------------------------
# 9. Install the library (the .a or .so file)
# -----------------------------------------------------------------------------
## @section install_lib_sec 7. Target Library Export Rules
## @brief Installs the compiled core library target to the workspace prefix.
## @details Registers the compiled binary outputs into standard installation subfolders
##          under `lib/` and `bin/`, packing dependencies inside the `QuantKosTargets` export list.

install(TARGETS QuantKos_core
        EXPORT QuantKosTargets       # Creates a "package" name for others to find
        LIBRARY DESTINATION lib      # Where .so files go
        ARCHIVE DESTINATION lib      # Where .a files go
        RUNTIME DESTINATION bin)     # Where .dll files go (Windows)


# -----------------------------------------------------------------------------
# 10. Install the headers
# -----------------------------------------------------------------------------
## @section install_headers_sec 8. Interface Headers Installation
## @brief Packages the public C++ API header folder for distribution.
## @details Copies only the interface files matching `.hpp` and `.h` patterns from the
##          local directory to the installation prefix's `include/` directory.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.hpp"              # Only copy header files
        PATTERN "*.h")
