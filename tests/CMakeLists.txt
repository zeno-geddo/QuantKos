# =============================================================================
# KOptions: Test Suite
# =============================================================================

# 1. Define the executables for the standalone tests
add_executable(test_rng RNGenTest.cpp)

# 2. Tell the test where to find your engine's header files
target_include_directories(test_rng PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 3. Link the required libraries
target_link_libraries(test_rng PRIVATE Kokkos::kokkos)

# 4. Register the test with CMake's testing framework (CTest)
add_test(
        NAME RNG_Distribution_Test
        COMMAND test_rng
)