#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include "../include/IO/InputHelp.hpp"
#include "../include/IO/ConfigParser.hpp"
//#include "../include/core/Mesh.hpp"
//#include "../include/core/Solver.hpp"


int main(int argc, char *argv[]) {
    // 1. Handle "Help" specifically (Before Kokkos starts)
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        KOps::HELP::print_usage(argv[0]);
        KOps::HELP::print_example_config();
        return 0; // Success
    }

    // 2. Handle "Wrong Input" (No arguments provided)
    if (argc < 2) {
        std::cerr << "Error: No configuration file provided.\n";
        KOps::HELP::print_usage(argv[0]);
        std::cerr << "Run '" << argv[0] << " --help' for an example configuration.\n";
        return 1; // Failure
    }

    // 3. Normal Execution starts here
    Kokkos::initialize(argc, argv);
    int exit_code = 0;
    {

        namespace KC = KOps::Config;
        //namespace KS = KS::Config;

        try {
            // --- Parse YAML Configuration ---
            const std::string input_file = argv[1];
            KC::UInputs conf = KC::Parser::parse(input_file);

            // --- Allocate MEMORY for MonteCarlo ---
            // auto mc = Labes::MC(conf.grid);
            // mc.initialize_variables(conf.init);
            // mc.print_memory_info();

            // 3. Create Solver
            //auto solver = KOps::Solver(conf);

            // 4. Dispatch and launch the simulation (Evaluates config ONCE)
            // solver.launch_simulation();
        } catch (const std::exception &e) {
            // Capture all validation/parsing/runtime errors
            std::cerr << "\n********************************************************\n";
            std::cerr << "FATAL ERROR: " << e.what() << "\n";
            std::cerr << "********************************************************\n";

            // Note: Kokkos::finalize() is called outside the scope.
            exit_code = 1;
        }
    } // GPU memory is safely deallocated here
    Kokkos::finalize();
    return exit_code;
}
