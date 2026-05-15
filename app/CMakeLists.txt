# -----------------------------------------------------------------------------
# 1. Adding Executable
# -----------------------------------------------------------------------------
add_executable(KOptions main.cpp)

# -----------------------------------------------------------------------------
# 2. Link to internal libraries
# (Automatically pulls in Kokkos, HDF5, and the include paths!)
# -----------------------------------------------------------------------------
target_link_libraries(KOptions PRIVATE KOptions_core)


# -----------------------------------------------------------------------------
# 3. Install the actual solver executable (make install)
# -----------------------------------------------------------------------------
install(TARGETS KOptions
        RUNTIME DESTINATION bin) # Puts the 'KOptions' binary in the bin/ folder