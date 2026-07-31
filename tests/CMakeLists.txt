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
# QuantKos: Test Suite
# =============================================================================
add_executable(Tester Tester.cpp) # 1. Define the executables for the standalone tests
target_include_directories(Tester PRIVATE ${CMAKE_SOURCE_DIR}/include) # 2. Tell the test where to find your engine's header files
#target_link_libraries(tester PRIVATE Kokkos::kokkos) # 3. Link the required libraries
target_link_libraries(Tester PRIVATE QuantKos_core)
add_test( # 4. Register the test with CMake's testing framework (CTest)
        NAME QuantKosTester
        COMMAND Tester
)

install(TARGETS Tester RUNTIME DESTINATION bin) # Puts the 'Tester' binary in the bin/ folder

