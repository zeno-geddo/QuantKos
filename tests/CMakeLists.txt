# =============================================================================
# KOptions: Test Suite
# =============================================================================
add_executable(tester Tester.cpp) # 1. Define the executables for the standalone tests
target_include_directories(tester PRIVATE ${CMAKE_SOURCE_DIR}/include) # 2. Tell the test where to find your engine's header files
#target_link_libraries(tester PRIVATE Kokkos::kokkos) # 3. Link the required libraries
target_link_libraries(tester PRIVATE KOptions_core)
add_test( # 4. Register the test with CMake's testing framework (CTest)
        NAME KOptionsTester
        COMMAND tester
)




## =============================================================================
## TEST 1: Random Number Generator Check
## =============================================================================
#add_executable(test_rng RNGenTest.hpp) # 1. Define the executables for the standalone tests
#target_include_directories(test_rng PRIVATE ${CMAKE_SOURCE_DIR}/include) # 2. Tell the test where to find your engine's header files
#target_link_libraries(test_rng PRIVATE Kokkos::kokkos) # 3. Link the required libraries
#add_test( # 4. Register the test with CMake's testing framework (CTest)
#        NAME RNG_Distribution_Test
#        COMMAND test_rng
#)
#
## =============================================================================
## TEST 2: Zero-Variance Forward Growth Check (Growt with no noise)
## =============================================================================
#add_executable(test_noNoise NoNoiseTest.hpp)
#target_include_directories(test_noNoise PRIVATE ${CMAKE_SOURCE_DIR}/include)
#target_link_libraries(test_noNoise PRIVATE Kokkos::kokkos)
#add_test(
#        NAME Zero_Variance_Forward_Test
#        COMMAND test_noNoise                       # CTest will run this executable
#)
