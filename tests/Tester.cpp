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
 * @file
 * @brief Master entry point and execution driver for the QuantKos parallel testing engine.
 * @details This file coordinates the integration test suite. It initializes the Kokkos parallel
 * execution space, registers all numerical, structural, and convergence test suites, parses
 * selective YAML run configurations if provided, and safely coordinates resource deallocation
 * before finalizing the backend runtimes.
 * @author Zeno GEDDO
 * @date 2026
 */

#include <iostream>
#include <Kokkos_Core.hpp>

#include  "./../include/tests/TestEngine.hpp"
#include "./../include/tests/IO/BinaryFormatTest.hpp"
#include "./../include/tests/RNG/GaussianRNGenTest.hpp"
#include "./../include/tests/Options/PutCallParityTest.hpp"
#include "./../include/tests/Options/NoNoiseTest.hpp"
#include "./../include/tests/Options/BlackScholesTest.hpp"
#include "./../include/tests/Options/HestonTest.hpp"
#include "./../include/tests/Options/BatesTest.hpp"
#include "./../include/tests/Options/AmericanOption.hpp"


/**
 * @brief Main testing execution function.
 * @details Implements the testing suite lifecycle:
 * 1. Boots up the default parallel execution space (e.g., CUDA streams or OpenMP CPU threads).
 * 2. Standardizes formatting and prints structural console run headers.
 * 3. Registers all quantitative finance validation suites (RNG, Binary IO, SDE continuous drift,
 * and European/American option convergence targets).
 * 4. Executes either a YAML-filtered list of test IDs or runs the full suite as a fallback.
 * 5. Tallies results, presents a final success/failure summary block, and shuts down Kokkos safely.
 * * @note Just like in the primary application main, the test registry execution is isolated
 * within a local scope block `{ ... }`. This guarantees that any internally allocated Kokkos Views
 * (such as high-dimensional path matrices used during weak convergence checks) are fully destructed
 * and reference-cleared before @c Kokkos::finalize() is executed, preventing memory-access violations.
 * * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings. `argv[1]` can optionally contain a YAML test filter.
 * @return int Returns @c 0 if all executed test suites pass their composite mathematical and
 * statistical tolerance thresholds; returns the number of failed suites otherwise.
 * @todo Should write a script that automatically compile and install QuantKos with cuda, openmp, and procedural backends. Then this script should also launch this test or a lighter benchmark to see how well the code is performing.
 */
int main(int argc, char *argv[]) {
    // 1. Initialize the parallel hardware backend (binds thread pools / GPU virtual contexts)
    Kokkos::initialize(argc, argv);
    int failed_tests = 0;
    {
        // 2. Output the stylized welcome banner and engine metadata (Matches app/main.cpp output paradigms)
        constexpr std::string_view indent{"   "};
        std::cout << "\n\n" << indent << "**************************************************\n";
        std::cout << indent << "          LAUNCHING QUANTKOS TEST ENGINE          \n";
        std::cout << indent << "**************************************************\n\n";

        // 3. Centralized Test Registration
        namespace KTE = quantkos::Tests;
        namespace KTE = quantkos::Tests;
        auto tester = KTE::TestEngine();
        tester.register_test("RNG",
                             "Verify that standard gaussian random variable is sampled correctly in Kokkos",
                             KTE::RNG::run_test_gaussian);
        tester.register_test("IOBin",
                             "Check that data are saved to disk and reloaded correctly using Binary File Format.",
                             KTE::IOBIN::run_test);
        tester.register_test("NoNoise",
                             "Zero-Variance SDE Drift",
                             KTE::NoNoise::run_test);
        tester.register_test("BlackScholes",
                             "Weak Convergence to Black Scholes",
                             KTE::BlackScholes::run_test);
        tester.register_test("Heston",
                             "Weak Convergence to Heston",
                             KTE::Heston::run_test);
        tester.register_test("Bates",
                             "Weak Convergence to Bates",
                             KTE::Bates::run_test);
        tester.register_test("AmericanOption",
                             "American Option LSM",
                             KTE::LSM::run_test);
        tester.register_test("EUGreeksPutCallParity",
                             "Check that EU option Greeks obey put-call parity",
                             KTE::Greeks::run_test);

        // 4. Run active tests (Automatically filters using argv[1] if a valid YAML configuration is parsed)
        failed_tests = tester.run_tests(argc, argv);

        // 5. Output stylized testing summary results and compliance metrics
        std::cout << indent << "**************************************************\n";
        std::cout << indent << "             EXECUTION RUN COMPLETE               \n";
        if (failed_tests == 0) {
            std::cout << indent << " STATUS   : ALL PASSED OK \n";
        } else {
            std::cout << indent << " STATUS   : FAILED (" << failed_tests << " suite(s) broke constraints)\n";
            std::cerr << indent << " STATUS   : FAILED (" << failed_tests << " suite(s) broke constraints)\n";
        }
        std::cout << indent << "**************************************************\n\n" << std::endl;
    } // CRITICAL: Local scope ends here. All test variables, managers, registries, and Kokkos
    // Views are fully destructed before the final hardware release is initiated.

    // 6. Release parallel runtime execution spaces and free up system resources
    Kokkos::finalize();

    return failed_tests == 0 ? 0 : 1;
}
