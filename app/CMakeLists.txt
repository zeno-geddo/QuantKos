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
## @brief Build configuration script for compiling and installing the primary KOptions application binary.
## @details This file handles the target definition for the user-facing executable, links it
##          to the shared/static core engine library, and sets up local binary deployment structures.
## @author Zeno GEDDO
## @date 2026

# -----------------------------------------------------------------------------
# 1. Adding Executable
# -----------------------------------------------------------------------------
## @section exec_sec (Main Executable Definition)
## @brief Adds the executable build target for the simulation solver.
## @details Compiles the user interface entry-point (`main.cpp`) to build the final,
##          standalone executable named `KOptions`.
add_executable(KOptions main.cpp)

# -----------------------------------------------------------------------------
# 2. Link to internal libraries
# (Automatically pulls in Kokkos, yalm, and the include paths!)
# -----------------------------------------------------------------------------
## @section link_sec (Dependent Target Linking)
## @brief Binds the executable target to the core numerical framework.
## @details Links `KOptions` to the target `KOptions_core` using `PRIVATE` scoping.
##          This private dependency propagation automatically populates the executable target
##          with all necessary compiler standards, include folders, and upstream package targets
##          such as Kokkos and yaml-cpp.
target_link_libraries(KOptions PRIVATE KOptions_core)


# -----------------------------------------------------------------------------
# 3. Install the actual solver executable (make install)
# -----------------------------------------------------------------------------
## @section install_app_sec (Application Deployment Rules)
## @brief Installs the compiled executable binary into the workspace distribution structure.
## @details Copies the built `KOptions` runtime target into the `bin/` folder of the localized
##          installation prefix (`CMAKE_INSTALL_PREFIX`), enabling clean workspace execution.

install(TARGETS KOptions
        RUNTIME DESTINATION bin) # Puts the 'KOptions' binary in the bin/ folder