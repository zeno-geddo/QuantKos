#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include "../include/IO/InputHelp.hpp"
#include "../include/IO/ConfigParser.hpp"
#include "../include/core/Distpatcher.hpp"


int main(int argc, char *argv[]) {

    namespace KH = KOps::HELP;

    // 1. Print input message
    KH::print_welcome_msg();

    // 1. Handle "Help" specifically (Before Kokkos starts)
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        KH::print_usage(argv[0]);
        KH::print_example_config();
        return 0; // Success
    }

    // 2. Handle "Wrong Input" (No arguments provided)
    if (argc < 2) {
        std::cerr << "Error: No configuration file provided.\n";
        KH::print_usage(argv[0]);
        KH::print_example_config();
        std::cerr << "Run '" << argv[0] << " --help' for an example configuration.\n";
        return 1; // Failure
    }

    // 3. Normal Execution starts here
    Kokkos::initialize(argc, argv);
    int exit_code = 0;
    {
        namespace KC = KOps::Config;
        namespace KE = KOps::Engine;
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
    } // GPU memory is safely deallocated here
    Kokkos::finalize();
    return exit_code;
}


//FATAL ERROR: Error: Kokkos::deep_copy with no available copy mechanism: from source view ("gpu_paths_batch_buffer") to destination view ("gpu_paths_batch_buffer_mirror").
//There is no common execution space that can access both source's space
//(Cuda) and destination's space (Host), so source and destination
//must be contiguous and have the same layout.