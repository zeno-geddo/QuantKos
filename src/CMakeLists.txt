# -----------------------------------------------------------------------------
# 1. CREATE TARGETS
# -----------------------------------------------------------------------------
add_library(KOptions_core
        IO/ConfigParser.cpp
)


# -----------------------------------------------------------------------------
# 2. APPLY OPTIONS
# -----------------------------------------------------------------------------
if (KOPS_ENABLE_SINGLE_PRECISION)
    target_compile_definitions(KOptions_core PUBLIC KOPS_USE_SINGLE_PRECISION)
    message(STATUS "Precision: SINGLE (float)")
else ()
    message(STATUS "Precision: DOUBLE (double)")
endif ()


# -----------------------------------------------------------------------------
# 3. LINKING & INCLUDES
# -----------------------------------------------------------------------------
target_link_libraries(KOptions_core PUBLIC
        Kokkos::kokkos
        yaml-cpp
        # ....
)
target_include_directories(KOptions_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)


# -----------------------------------------------------------------------------
# 4. DEFAULT COMPILER FLAGS (Base Warnings, always ON for GCC/Clang despite the optimizations)
# -----------------------------------------------------------------------------
target_compile_options(KOptions_core PRIVATE
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
# 5. DEBUG COMPILER FLAGS (Only active in Debug build : -DCMAKE_BUILD_TYPE=Debug)
# -----------------------------------------------------------------------------
target_compile_options(KOptions_core PRIVATE
        $<$<CONFIG:Debug>:
        -O0                         # Disable optimizations for accurate step-through debugging
        -g                          # Include debug symbols for GDB/LLDB/CLion
        -fno-omit-frame-pointer     # Keep frame pointer for cleaner stack traces
        -fsanitize=address          # ASan: Detects out-of-bounds access, use-after-free, and memory leaks instantly.
        -fsanitize=undefined        # UBSan: Catches undefined behavior like integer overflow, null pointer dereferencing, or division by zero.
        >
)

# Linker flags: Required to link the ASan/UBSan runtime libraries
target_link_options(KOptions_core PRIVATE
        $<$<CONFIG:Debug>:
        -fsanitize=address          # Link AddressSanitizer runtime
        -fsanitize=undefined        # Link UndefinedBehaviorSanitizer runtime
        >
)

# -----------------------------------------------------------------------------
# 5.5. MINSIZEREL COMPILER FLAGS (-DCMAKE_BUILD_TYPE=MinSizeRel)
# -----------------------------------------------------------------------------
# Optimizes strictly to reduce the final binary file size. (Rarely used in Quant Finance).
target_compile_options(KOptions_core PRIVATE
        $<$<CONFIG:MinSizeRel>:
        -Os                         # Optimize for size (disables loop unrolling)
        >
)
# No special linker flags needed for MinSizeRel

# -----------------------------------------------------------------------------
# 6. RELEASE COMPILER FLAGS (Only active in Release build : -DCMAKE_BUILD_TYPE=Release)
# -----------------------------------------------------------------------------
target_compile_options(KOptions_core PRIVATE
        $<$<CONFIG:Release>:
        -O3                         # High-level optimization: vectorization, loop unrolling, and inlining
        -march=native               # Tune code for the specific CPU features (AVX, AVX-512) of the host machine
        -flto                       # Link Time Optimization: allows optimizations across different translation units
        -fno-math-errno             # Disables updating 'errno' after math calls; allows faster SIMD math instructions
        # -ffast-math               # Enable aggressive math optimizations, breaks IEEE-754 floating point standards to go faster (may break NaN checks and precision)
        >
)

# Linker flags for LTO (Link Time Optimization)
target_link_options(KOptions_core PRIVATE
        $<$<CONFIG:Release>:
        -flto                       # Required during linking to perform cross-file optimizations
        -O3                         # Ensures the linker uses the same optimization level as the compiler
        >
)

# -----------------------------------------------------------------------------
# 6.5. RELWITHDEBINFO COMPILER FLAGS (-DCMAKE_BUILD_TYPE=RelWithDebInfo)
# Production-level speed, but with a map attached.
# -----------------------------------------------------------------------------
# For HPC, we want the exact same aggressive speed as Release, but WITH debug symbols for profiling.
target_compile_options(KOptions_core PRIVATE
        $<$<CONFIG:RelWithDebInfo>:
        -O3                         # Keep maximum speed
        -march=native               # Keep vectorization
        -flto                       # Keep LTO
        -fno-math-errno             # Keep math speed
        -g                          # ADD debug symbols for profiling (perf, VTune, Valgrind)
        -fno-omit-frame-pointer     # Keep the call stack clean so the profiler can read it
        >
)

target_link_options(KOptions_core PRIVATE
        $<$<CONFIG:RelWithDebInfo>:
        -flto                       # Required for LTO
        -O3                         # Ensures the linker uses the same optimization level as the compiler
        >
)



# -----------------------------------------------------------------------------
# 7. Install the library (the .a or .so file)
# -----------------------------------------------------------------------------
install(TARGETS KOptions_core
        EXPORT KOptionsTargets          # Creates a "package" name for others to find
        LIBRARY DESTINATION lib      # Where .so files go
        ARCHIVE DESTINATION lib      # Where .a files go
        RUNTIME DESTINATION bin)     # Where .dll files go (Windows)


# -----------------------------------------------------------------------------
# 8. Install the headers
# -----------------------------------------------------------------------------
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.hpp"              # Only copy header files
        PATTERN "*.h")
