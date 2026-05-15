# -----------------------------------------------------------------------------
# 1. Project Configuration
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(KOptions VERSION 0.0.1 DESCRIPTION "to be done..." HOMEPAGE_URL "to be added..." LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# -----------------------------------------------------------------------------
# 2. Build Type Configuration (Default, Release, RelWithDebInfo, MinSizeRel)
# -----------------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to 'Release'")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif ()

# Create a drop-down list of options for ccmake / cmake-gui
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "RelWithDebInfo" "MinSizeRel")


# -----------------------------------------------------------------------------
# 2.5 Default Installation Prefix (Workspace Isolation)
# -----------------------------------------------------------------------------
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "No install prefix selected. Defaulting to local 'install' directory.")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Default install path" FORCE)
endif ()

# -----------------------------------------------------------------------------
# 3. Options
# -----------------------------------------------------------------------------
option(KOPS_ENABLE_SINGLE_PRECISION "Use single precision (float) instead of double ?" OFF)
option(KOPS_ENABLE_TESTS "Build unit tests ?" ON)
option(KOPS_ENABLE_FETCHCONTENT "Enable FetchContent ?" OFF)


# -----------------------------------------------------------------------------
# 4. Find Dependencies (Find-or-Fetch)
# -----------------------------------------------------------------------------
include(FetchContent)


# --- YAML-CPP ---
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

        # --- KOKKOS OPTIONS ---
        # These ensure Kokkos builds with the right backend for your machine.
        # set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
        # set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(kokkos)
    else ()
        # Users must provide the path via : -DKokkos_ROOT=/path/to/install
        message(FATAL_ERROR "Kokkos not found and KOPS_ENABLE_FETCHCONTENT is OFF. "
                "Please also provide -DKokkos_ROOT=/path/to/install when running 'cmake -B build' or enable fetching.")
    endif ()
endif ()


# HDF5: Required for HDF5Writer.cpp
# do be done


# -----------------------------------------------------------------------------
# 5. Build the Core Library (src)
# -----------------------------------------------------------------------------
add_subdirectory(src)


# -----------------------------------------------------------------------------
# 6. Build the App and Tests (Consumers)
# -----------------------------------------------------------------------------
add_subdirectory(app)

# if(LABES_ENABLE_TESTS)
# enable_testing()
# add_subdirectory(tests)
# message(STATUS "Activated build in unit tests")
# endif()


# -----------------------------------------------------------------------------
# 7. Install useful info
# -----------------------------------------------------------------------------
install(FILES README.md KOptionsConfig.yaml DESTINATION share/KOptions)


# =============================================================================
# =============================================================================
# -----------------------------------------------------------------------------
#                                USAGE EXAMPLE
# -----------------------------------------------------------------------------
# Create a build directory and set the install path to a folder named 'dist' :
# 1) cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/KOps_dist .
# 2) cmake --build build -j 4
# 3) cmake --install build     (or :  cd build && make install)
# Run the solver :
# 4) cd ~/kops_dist
# 5) cp share/KOptions/inputs.yaml .
# 6) ./bin/KOptions inputs.yaml
