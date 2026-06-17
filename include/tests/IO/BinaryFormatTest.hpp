#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <Kokkos_Core.hpp>

#include "../../core/config/Config.hpp"
#include "../../IO/IOBinary.hpp"
#include "../../core/memory/PathsMCBatchMem.hpp"

namespace KOps::Tests::IOBIN {
    namespace KC = KOps::Config;
    namespace KE = KOps::Engine;
    namespace KT = KOps::Types;
    namespace KB = KOps::IO::Binary;

    inline bool run_test() {
        std::string_view indent{"   "};
        std::cout << "\n\n" << indent << "====================================================================\n"
                  << indent << "          TEST 2 : Binary I/O Round-Trip Check             \n"
                  << indent << "====================================================================\n";
        std::cout << indent << "[   RUN   ] Binary I/O Round-Trip Check\n";

        bool test_passed = true;

        // 1. Setup a Predictable Configuration
        auto config = KC::UInputs();
        config.output.filename_paths_out = "IO_BIN_test_roundtrip.paths";
        config.output.out_dir = "."; // Current directory

        config.mc.N_Paths = 100;
        config.mc.batch_size = 100;
        config.time.N_time_steps = 10;
        config.time.t_end = 1.0;
        config.time.dt = 0.1;

        // The target time we want to extract (e.g., t = 0.7 -> Index 6)
        double target_time = 0.7;
        int expected_col_idx = 6;

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
                    dummy_batch.h_batch_view(i, j) = static_cast<KT::Real>((i * 1000) + j);
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
                test_passed = false;
            }

            // Assert data matches our predictable pattern exactly
            for (int i = 0; i < config.mc.N_Paths; ++i) {
                KT::Real expected_value = static_cast<KT::Real>((i * 1000) + expected_col_idx);

                if (extracted_prices[i] != expected_value) {
                    std::cerr << indent << "[  FAILED  ] Data mismatch at path " << i
                              << ". Expected: " << expected_value << ", Got: " << extracted_prices[i] << "\n";
                    test_passed = false;
                    break;
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << indent << "[  FAILED  ] Exception thrown during reading: " << e.what() << "\n";
            test_passed = false;
        }

        if (test_passed) {
            std::cout << indent << "[   OK   ] Binary Writer and Reader performed perfect round-trip.\n";
        }

        // ====================================================================
        // PHASE 3: CLEANUP
        // ====================================================================
        std::filesystem::path file_out_paths = std::filesystem::path(config.output.out_dir) / config.output.filename_paths_out;
        if (std::filesystem::exists(file_out_paths)) {
            std::filesystem::remove(file_out_paths);
        }

        std::cout << indent << "====================================================================\n" << std::endl;
        return test_passed;
    }
}