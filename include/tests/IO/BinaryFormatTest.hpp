// Copyright (C) 14/07/2026 Zeno GEDDO <zeno.geddo@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <Kokkos_Core.hpp>

#include "../../core/config/Config.hpp"
#include "../../IO/IOBinary.hpp"
#include "../../core/memory/PathsMCBatchMem.hpp"
#include "tests/Options/AmericanOption.hpp"

/**
 * @namespace KOps::Tests::IOBIN
 * @brief Integration tests for validating binary data storage and retrieval.
 */
namespace quantkos::Tests::IOBIN {
    namespace KC = quantkos::Config;
    namespace KE = quantkos::Engine;
    namespace KT = quantkos::Types;
    namespace KB = quantkos::IO::Binary;

    /**
     * @brief Verifies the integrity of the binary serialization system (both when normalizing prices is on and off).
     * * @details This test performs a full "round-trip" verification of our binary subsystem.
     * It ensures that data written to disk is exactly identical when read back.
     * * The validation is divided into three distinct phases:
     * - **Phase 1**: Creates a predictable matrix grid (using unique paths and time coordinates) and writes it to a file.
     * - **Phase 2**: Reads back a specific column (representing a target time step) and verifies the values match the expected math.
     * - **Phase 3**: Cleans up and deletes the temporary binary test file.
     * * @return true if the test succeeds and the data remains uncorrupted; false otherwise.
     */
    inline bool run_test() {
        std::string_view indent{"   "};
        std::cout << "\n\n" << indent << "====================================================================\n"
                << indent << "          TEST 2 : Binary I/O Round-Trip Check             \n"
                << indent << "====================================================================\n";
        std::cout << indent << "[   RUN   ] Binary I/O Round-Trip Check\n";

        bool overall_test_passed = true;

        // The target time we want to extract (e.g., t = 0.7 -> Index 6)
        const double target_time = 0.7;
        const int expected_col_idx = 6;

        const std::vector<bool> norm_flags = {false, true};
        for (bool f: norm_flags) {

            // Show the normalization flag
            bool current_run_passed = true;
            std::cout << indent << "\n[   INFO   ] Norm flag : " << f << "\n";

            // 1. Setup a Predictable Configuration
            // !!! NOTE : if this in put outside the loop, and the bool vect can stop commuting and the test can break !
            auto config = KC::UInputs();
            config.output.filename_paths_out = "IO_BIN_test_roundtrip.paths";
            config.output.out_dir = "."; // Current directory
            config.mc.N_Paths = 100;
            config.mc.batch_size = 100;
            config.time.N_time_steps = 10;
            config.time.t_end = 1.0;
            config.time.dt = 0.1;
            config.mc.normalize_prices = f;
            config.apply_price_scaling();


            // ====================================================================
            // PHASE 1: GENERATE AND WRITE FAKE DATA
            // ====================================================================
            std::cout << indent << "[   INFO   ] Phase 1: Generating and Writing predictable matrix...\n";
            {
                // Allocate Memory Manager
                KE::PathsMCBatchMem dummy_batch(config);

                // Fill the host view with predictable data: value = (path_id * 1000) + time_step
                for (int i = 0; i < config.mc.N_Paths; ++i) {
                    for (int j = 0; j < config.time.N_time_steps; ++j) {
                        const auto raw_val = static_cast<double>((i * 1000) + j);
                        const double scale_fact = config.get_price_scaling_factor();
                        dummy_batch.h_batch_view(i, j) = static_cast<KT::Real>(raw_val/scale_fact);
                    }
                }

                // Write to disk using BinWriter
                KB::BinWriter writer(config);
                writer.save_paths_batch_if_needed(config.mc.batch_size, dummy_batch);

                // The dummy_batch is destroyed, and writer.~BinWriter() is called,
                // safely flushing the bytes to the SSD and closing the file lock!
            }


            // ====================================================================
            // PHASE 2: READ AND VALIDATE DATA
            // ====================================================================
            std::cout << indent << "[   INFO   ] Phase 2: Reading back target time t = " << target_time << "...\n";

            try {
                KB::BinReader reader(config);
                std::vector<KT::Real> extracted_prices = reader.read_prices_at_target_time(target_time);

                // Assert size matches
                if (extracted_prices.size() != static_cast<size_t>(config.mc.N_Paths)) {
                    std::cerr << indent << "[  FAILED  ] Extracted vector size (" << extracted_prices.size()
                            << ") does not match expected (" << config.mc.N_Paths << ").\n";
                    current_run_passed = false;
                }

                // Assert data matches our predictable pattern exactly
                for (int i = 0; i < config.mc.N_Paths; ++i) {
                    const auto expected_value = static_cast<KT::Real>((i * 1000) + expected_col_idx);

                    if (std::abs(extracted_prices[i] - expected_value) > 1e-5) {
                        std::cerr << indent << "[  FAILED  ] Data mismatch at path " << i
                                << ". Expected: " << expected_value << ", Got: " << extracted_prices[i] << "\n";
                        current_run_passed = false;
                        break;
                    }
                }
            } catch (const std::exception &e) {
                std::cerr << indent << "[  FAILED  ] Exception thrown during reading: " << e.what() << "\n";
                current_run_passed = false;
            }

            if (current_run_passed) {
                std::cout << indent << "[   OK   ] Binary Writer/Reader round-trip succeeded for norm_flag = "
                          << (f ? "true" : "false") << ".\n";
            } else {
                overall_test_passed = false;
            }

            // ====================================================================
            // PHASE 3: CLEANUP
            // ====================================================================
            std::filesystem::path file_out_paths =
                    std::filesystem::path(config.output.out_dir) / config.output.filename_paths_out;
            if (std::filesystem::exists(file_out_paths)) {
                std::filesystem::remove(file_out_paths);
            }
        }
        std::cout << indent << "======================== FINISHED BINARY IO TEST========================================\n" <<
            std::endl;
        return overall_test_passed;
    }
}
