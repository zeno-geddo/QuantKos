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


/**
 * @file main.cpp
 * @brief Main entry point for the QuantKos parallel Monte Carlo option pricing engine.
 * @details This file coordinates the application sequence, whose task consist in computing an option price (and related statistics) given the user parameters.
 * It handles command-line arguments, offers inline console help templates, initializes the parallel Kokkos
 * runtime execution space, and dispatches the stochastic simulation workflow.
 * @author Zeno GEDDO
 * @date 2026
 * @todo Should add a description to the other files as well.
 */

#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include "../include/IO/InputHelp.hpp"
#include "../include/IO/InputParser.hpp"
#include "../include/core/engine/Distpatcher.hpp"

/**
 * @brief Main execution function.
 * * @details This function implements the application lifecycle:
 * 1. Prints a welcome message and system metadata.
 * 2. Checks CLI arguments to see if a `--help` flag or valid configuration path is supplied.
 * 3. Boots up Kokkos to initialize hardware target_backends (CPU threads or GPU VRAM).
 * 4. Parses the YAML simulation parameters.
 * 5. Launches high-throughput simulation runs on the active compute device.
 * 6. Shuts down the parallel environment safely, avoiding memory leaks.
 * * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings. `argv[1]` must be the path to a valid YAML configuration.
 * @return int Exit status code: @c 0 for successful completion, @c 1 on fatal parsing or execution errors.
*@todo Check if it is worth adding std::ios_base::sync_with_stdio(false); and std::cin.tie(NULL); to reduce terminal I/O latency.
*/
int main(int argc, char *argv[]) {

    namespace KH = quantkos::HELP;

    // 1. Output the stylized welcome banner and engine version metadata
    KH::print_welcome_msg();

    // 2. Handle the "Help" flag explicitly BEFORE starting the parallel runtime
    //    to minimize unnecessary hardware initialization.
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        KH::print_usage(argv[0]);
        KH::print_example_config();
        return 0; // Success
    }

    // 3. Reject execution if no configuration file path was provided
    if (argc < 2) {
        std::cerr << "Error: No configuration file provided.\n";
        KH::print_usage(argv[0]);
        KH::print_example_config();
        std::cerr << "Run '" << argv[0] << " --help' for an example configuration.\n";
        return 1; // Failure
    }

    // 4. Initialize Kokkos parallel environment (binds GPU contexts/CPU thread pools)
    std::cout << "  [ Main ] Initializing Kokkos ... \n";
    const auto start_time = std::chrono::steady_clock::now();
    Kokkos::initialize(argc, argv);
    const auto end_time = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "  [ Main ] Time taken to initialize Kokkos : "<< elapsed.count() <<"s \n";

    int exit_code = 0;
    {
        namespace KC = quantkos::Config;
        namespace KE = quantkos::Engine;
        try {
            // 1. Parse YAML Configuration
            const std::string input_file = argv[1];
            KC::UInputs conf = KC::Parser::parse(input_file);

            // 2. Dispatch and Launch MC simulation
            auto MCDisp = KE::MCDispatcher(conf);
            MCDisp.launch_montecarlo();
        } catch (const std::exception &e) {
            // Capture all validation/parsing/runtime errors
            std::cerr << "\n********************************************************\n";
            std::cerr << "FATAL ERROR: " << e.what() << "\n";
            std::cerr << "********************************************************\n";

            exit_code = 1;
        }
    } // CRITICAL: This local scope guarantees all Kokkos Views, managers, and allocated
    // GPU memory are safely deallocated BEFORE Kokkos::finalize() is invoked,
    // eliminating system-level leaks.

    // 5. Tear down the parallel environment and free up backend system resources
    Kokkos::finalize();
    return exit_code;
}
