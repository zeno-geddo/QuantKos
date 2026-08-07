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

#include <map>
#include <string>
#include <cmath>

#include "TestsUtils.hpp"
#include "../../core/config/Config.hpp"
#include "../../core/engine/Distpatcher.hpp"
#include "../../IO/IOBinary.hpp"

/**
 * @brief Verification tests running under zero-noise (deterministic) conditions.
 */
namespace quantkos::Tests::NoNoise {
    namespace KC = quantkos::Config;
    namespace KE = quantkos::Engine;
    namespace KT = quantkos::Types;
    namespace KI = quantkos::Implemented;
    namespace KB = quantkos::IO::Binary;
    namespace KTU = quantkos::Tests::Utils;

    /**
     * @brief Calculates the analytical exact solution for deterministic asset growth.
     * @details Under zero volatility, the stochastic asset price equation collapses to a
     * standard ordinary differential equation (ODE) modeling risk-free appreciation
     * and continuous dividend depreciation:
     * $$ S_T = S_0 \cdot e^{(r - q) T} $$
     * @param conf The simulation parameters containing interest rate (r), dividend yield (q), maturity (T), and initial spot (S0).
     * @return The theoretical exact spot price at maturity.
     */
    inline double get_exact_solution(const quantkos::Config::UInputs &conf) {
        // Expected: S_T = S_0 * exp((r - q) * T)
        const double expected_S_T = conf.market.S0 * std::exp(
                                        (conf.market.r - conf.market.q) * conf.time.t_end);
        return expected_S_T;
    }

    /**
     * @brief Generates a default configuration designed for deterministic testing.
     * @details Configures a mock Heston model with all volatility components set to zero
     * (v0 = 0, kappa = 0, theta = 0, sigma = 0, lambda_J = 0, mu_J = 0, sigma_J = 0). This forces the asset to evolve purely
     * via continuous risk-free drift.
     * @return A populated configuration structure with zero-variance parameters.
     */
    inline KC::UInputs getDefaultConfig() {
        // General Config
        auto config = KC::UInputs();

        // NOTE : Use the test needs also to reload the results
        // NOTE : The names of the paths will be given in the test loop
        config.mc.normalize_prices = true;

        config.output.format = KI::IOFormat::BIN;
        config.output.filename_log = "";
        config.output.out_dir = "temp_res_no_noise_test";

        config.mc.N_Paths = 100;
        config.mc.batch_size = config.mc.N_Paths;

        config.time.N_time_steps = 365;
        config.time.t_end = 1.;
        config.time.inp_dt = config.time.t_end / config.time.N_time_steps;

        config.market.S0 = 100.;
        config.market.v0 = 0.;
        config.market.r = 0.05;
        config.market.q = 0.02;

        config.model.heston.k = 0.;
        config.model.heston.theta = 0.;
        config.model.heston.sigma = 0.;
        config.model.heston.rho = 0.0;

        config.model.bates.k = 0.;
        config.model.bates.theta = 0.;
        config.model.bates.sigma = 0.;
        config.model.bates.rho = 0.0;
        config.model.bates.lambda_J = 0.0; // zero jump intensity
        config.model.bates.mu_J = 0.0; // zero log jump size
        config.model.bates.sigma_J = 0.0; // std deviation jump size

        return config;
    }

    /**
     * @brief Runs the Zero Variance Forward Growth integration test.
     * @details Simulates asset price trajectories with zero stochastic variance.
     * Since there is no random noise, all simulated paths must grow identically
     * following the continuous risk-free drift:
     * $$ S_t = S_0 \cdot e^{(r - q)t} $$
     * * ### Verification Pipeline
     * 1. Initializes a 100-path deterministic Heston-Euler simulation.
     * 2. Launches the parallel execution loop via the simulation dispatcher.
     * 3. Reads the binary data payload back from disk at maturity ($T = 1.0$).
     * 4. Asserts that the final prices on all paths match the exact analytical ODE solution.
     * * @return true if all simulated paths match the exact solution within numerical limits; false otherwise.
     * @note This test also tests that the saving and reading simulated data on disk work correctly.
    */
    inline bool run_test() {
        // NOTE : When no randomness, the stock should grow purely by the deterministic drift: S_T = S_0 e^{(r-q)T}.

        std::string_view indent{"   "};
        std::cout << "\n\n" << indent << "====================================================================\n"
                << indent << "          TEST 3 : Zero Variance Forward Growth Check             \n"
                << indent << "====================================================================\n";
        std::cout << indent << "[   RUN   ] Zero Variance Forward Growth Check\n";

        bool overall_test_passed = true;

        // STEP 1: Get exact solution
        auto config = getDefaultConfig();
        config.validate();
        const double expected_S_T = get_exact_solution(config);
        std::cout << indent << "[   INFO   ] Expected Final Price : " << expected_S_T << "\n";


        // STEP 2 : Run test for each model/scheme
        const auto models_to_test = KTU::get_models_to_test();
        for (const auto &[fst, snd]: models_to_test) {
            // STEP 2a : Set config for numerical solver
            config.model.id_model = fst;
            config.scheme.id_scheme = snd;
            std::string name_out_file = "Test_" + KI::enum_to_string(config.model.id_model) + "_"
                                        + KI::enum_to_string(config.scheme.id_scheme) + ".paths";
            config.output.filename_paths_out = name_out_file;
            config.validate();
            config.print_summary();

            // Check and print params
            std::cout << indent << "[   INFO   ] MODEL   : " << KI::enum_to_string(config.model.id_model) << "\n";
            std::cout << indent << "[   INFO   ] SCHEME  : " << KI::enum_to_string(config.scheme.id_scheme) <<
                    "\n";

            std::cout << indent << ">>> Calling the solver ...\n";
            std::cout << "\n" << indent << "--------------------------------------------------------------------\n";

            // STEP 2b :Run model (keep within a dedicated the scope to be sure that dispatcher is destructed correctly)
            {
                auto MCDisp = KE::MCDispatcher(config);
                MCDisp.launch_montecarlo();
            }

            //STEP 2c :  Read the data back from disks (this also tests that writing and loading works correclty)
            std::cout << "\n" << indent << "--------------------------------------------------------------------\n";
            std::cout << indent << ">>> Go back to the tester ...\n";
            KB::BinReader binReader(config);
            std::vector<KT::Real> simulated_prices = binReader.read_prices_at_target_time(config.time.t_end);

            // Check if last Price at T_Final is correct
            const double epsilon = KT::is_real_using_single_precision() ? 5e-3 : 1e-8;
            for (size_t i = 0; i < simulated_prices.size(); ++i) {
                // Check if you have a nan
                if (!std::isfinite(simulated_prices[i])) {
                    std::cout << indent << "[  FAILED  ] Path " << i
                            << " produced a non-finite value: " << simulated_prices[i] << "\n";
                    std::cerr << indent << "[  FAILED  ] Path " << i
                            << " produced a non-finite value: " << simulated_prices[i] << "\n";
                    overall_test_passed = false;
                    break;
                }

                // We use 1e-5 to account for floating point drift during 365 compounded steps
                if (std::abs(simulated_prices[i] - expected_S_T) > epsilon) {
                    std::cout << indent << "[  FAILED  ] Path " << i << " deviated! Expected: "
                            << expected_S_T << ", Got: " << simulated_prices[i] << "\n";
                    std::cerr << indent << "[  FAILED  ] Path " << i << " deviated! Expected: "
                            << expected_S_T << ", Got: " << simulated_prices[i] << "\n";
                    overall_test_passed = false;
                    break;
                }
                std::cout << indent << "[   PASSED   ] Path " << i <<
                        " (Simulated : " << simulated_prices[i] << ", Expected : " << expected_S_T << ")\n";
            }

            // Clean up out binary file generated (all the output folder folder)
            std::filesystem::path out_dir_path = config.output.out_dir;
            if (std::filesystem::exists(out_dir_path)) {
                std::filesystem::remove_all(out_dir_path);
            }
        }

        // STEP 3 : Check if the tests passed.
        if (overall_test_passed) {
            std::cout << indent << "[   PASSED   ] All terminal prices match the theoretical drift.\n";
        } else {
            std::cout << indent << "[  FAILED  ] Terminal prices does not match the theoretical drift.\n";
        }
        std::cout << indent << "====================================================================\n" << std::endl;
        return overall_test_passed;
    }
}
