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


# =============================================================================
# KOptions: Test Suite
# =============================================================================
add_executable(Tester Tester.cpp) # 1. Define the executables for the standalone tests
target_include_directories(Tester PRIVATE ${CMAKE_SOURCE_DIR}/include) # 2. Tell the test where to find your engine's header files
#target_link_libraries(tester PRIVATE Kokkos::kokkos) # 3. Link the required libraries
target_link_libraries(Tester PRIVATE KOptions_core)
add_test( # 4. Register the test with CMake's testing framework (CTest)
        NAME KOptionsTester
        COMMAND Tester
)

install(TARGETS Tester RUNTIME DESTINATION bin) # Puts the 'Tester' binary in the bin/ folder




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
