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

#include <stdexcept>
#include <cmath>

#include "../../core/config/Config.hpp"
#include "../../core/engine/Distpatcher.hpp"


/**
 * @brief Common test execution helper functions and statistical analysis tools.
 */
namespace quantkos::Tests::Utils {
    namespace KC = quantkos::Config;
    namespace KT = quantkos::Types;
    namespace KI = quantkos::Implemented;
    namespace KE = quantkos::Engine;


    /**
     * @brief Returns a static registry of standard model and numerical scheme pairs.
     * @details Establishes a comprehensive default matrix of options to evaluate during
     * integration testing, coupling the Heston and Bates models with Euler, Milstein, and
     * Andersen's Quadratic-Exponential (QE) discretization schemes.
     * @return A vector of paired model and numerical scheme enumerations.
     */
    inline std::vector<std::pair<KI::MathModel, KI::NumScheme> > get_models_to_test() {
        static const std::vector<std::pair<KI::MathModel, KI::NumScheme> > models_to_test = {
            {KI::MathModel::Heston, KI::NumScheme::Euler},
            {KI::MathModel::Heston, KI::NumScheme::ImplicitMilstein},
            {KI::MathModel::Heston, KI::NumScheme::AndersonQE},
            {KI::MathModel::Bates, KI::NumScheme::Euler},
            {KI::MathModel::Bates, KI::NumScheme::ImplicitMilstein},
            {KI::MathModel::Bates, KI::NumScheme::AndersonQE},
        };
        return models_to_test;
    }

    /**
     * @brief Generates a logarithmic sequence of step resolutions.
     * @details Computes a range of time-step resolutions calculated via bit-shifted powers of two:
     * $$N_{\text{steps}} = 2^i$$
     * For example, a sweep from exponent 3 to 9 yields steps ranging from $8$ to $512$.
     * @param lower_exponent Starting power-of-two exponent (default is 3, yielding 8 steps).
     * @param upper_exponent Ending power-of-two exponent (default is 9, yielding 512 steps).
     * @return A vector containing the generated integer step counts.
     * @throw std::runtime_error If the lower exponent exceeds the upper exponent bounds.
     */
    inline std::vector<int> get_time_grid_resolutions(const int lower_exponent = 3, const int upper_exponent = 9) {
        // Return a vector containing different resolution (Calculated as 2^i) for the grid
        if (lower_exponent > upper_exponent)
            throw std::runtime_error("Error : Lower exponent is greater than upper exponent");

        std::vector<int> steps;
        steps.reserve(upper_exponent - lower_exponent + 1);

        for (int i = lower_exponent; i <= upper_exponent; ++i) {
            steps.push_back(1 << i);
        }

        return steps;
    }

    /**
     * @brief Stateless entry point that validates configuration states and launches a simulation run.
     * @param config The active user input parameters.
     * @return Resulting Monte Carlo pricing outputs and error metrics.
     */
    [[nodiscard]] inline KE::MCResults run_simulation(KC::UInputs config) {
        // Validate and print configuration
        config.validate();
        config.print_summary();
        // Dispatch and execute
        auto MCDisp = KE::MCDispatcher(config);
        return MCDisp.launch_montecarlo();
    }

    /**
     * @brief Data snapshot tracking simulation errors and steps for a specific grid resolution.
     */
    struct OptionPriceErr {
        int N_dt; ///< Total number of discrete steps along the path.
        double dt; ///< Size of the individual temporal step ($\Delta t$).
        double error; ///< Difference between numerical and expected analytical prices.
        double stat_error; ///< Standard error boundary computed from the Monte Carlo variance.
        bool is_finite; ///< Tells if numerical results are finite
        bool within_stat_bounds; ///< Tells if numerical results are within expected statistical bounds
    };


    /**
     * @brief Check that results of MC and ref price are finite
     * @return true if results are finite, false otherwise
     */
    [[nodiscard]] inline bool are_finite(const double num_price,
                                         const double expected_price,
                                         const double stat_error) {
        // Reject NaNs and Infs
        if (!std::isfinite(num_price) || !std::isfinite(expected_price) || !std::isfinite(stat_error)) {
            return false;
        }
        return true;
    }

    /**
     * @brief Strictly validates mathematical outputs against NaN, Inf, and statistical bounds.
     * @return true if valid and within tolerance, false if broken.
     */
    [[nodiscard]] inline bool is_valid_run(const double num_price,
                                           const double expected_price,
                                           const double stat_error,
                                           const double z_limit = 3.0) {
        // 1. Reject NaNs and Infs immediately
        if (!are_finite(num_price, expected_price, stat_error)) {
            std::cout << "      [ FATAL ] Non-finite numerical value detected (NaN or Inf).\n";
            std::cout << "       Num Price   : " << num_price << '\n';
            std::cout << "       Reference  : " << expected_price << '\n';
            std::cout << "       Stat Error : " << stat_error << '\n';
            return false;
        }

        // 2. Reject impossible standard errors
        if (stat_error < 0.0) {
            std::cout << "      [ FATAL ] Negative standard error detected: " << stat_error << "\n";
            return false;
        }

        const double abs_error = std::abs(num_price - expected_price);

        // 3. Deterministic Fallback (Zero Variance / No Noise)
        if (stat_error <= 1e-14) {
            const double strict_tol = 1e-8; // Hardcoded absolute tolerance
            if (abs_error > strict_tol) {
                std::cout << "      [ INFO ] Deterministic test failed. Abs Error ("
                        << abs_error << ") exceeds strict limit (" << strict_tol << ").\n";
                return false;
            }
            std::cout << "     [   PASSED   ] Deterministic results match reference exactly.\n";
            return true;
        }

        // 4. Linear Bound Check (Avoids Division by Zero)
        if (abs_error > z_limit * stat_error) {
            std::cout << "      [ INFO ] Price deviates by more than " << z_limit << " standard errors.\n";
            return false;
        }
        std::cout << "     [   PASSED   ] Numerical results statistically consistent with reference.\n";

        return true;
    }


    /**
     * @brief Launches a single simulation block and evaluates absolute error against an exact benchmark.
     * @param config Master simulation parameters.
     * @param expected_price The analytical reference price.
     * @param indent Console layout spacing alignment.
     * @return A populated OptionPriceErr tracking structural step errors and mathematical validity flags.
     */
    [[nodiscard]] inline OptionPriceErr compare_numerical_and_expected_option_price(quantkos::Config::UInputs &config,
        const double expected_price,
        const std::string_view indent = "   ") {
        std::cout << indent << ">>> Calling the solver ...\n";
        std::cout << "\n" << indent << "--------------------------------------------------------------------\n";

        const auto MCRes = run_simulation(config);
        const double num_price = MCRes.OptionPrice.option_price;
        const double stat_error = MCRes.OptionPrice.standard_error;
        const double signed_error = num_price - expected_price;
        const double abs_error = std::abs(signed_error);
        const bool finite_res = are_finite(num_price, expected_price, stat_error);

        std::cout << indent << "Resolution considering  " << config.time.N_time_steps << " times steps:\n";
        std::cout << indent << "   Finite res : " << finite_res << '\n';
        std::cout << indent << "   Price      : " << num_price << '\n';
        std::cout << indent << "   Reference  : " << expected_price << '\n';
        std::cout << indent << "   Error      : " << signed_error << '\n';
        std::cout << indent << "   Abs Error  : " << abs_error << '\n';
        std::cout << indent << "   Std Error  : " << stat_error << '\n';


        std::string z_score_str = (stat_error > 1e-14)
                                      ? std::to_string(signed_error / stat_error)
                                      : "INF (Deterministic)";
        std::cout << indent << "   Z-score    : " << z_score_str << '\n';

        // Safe if test passed
        constexpr double z_score_limit = 3.0;
        const bool run_passed = is_valid_run(num_price,
                                             expected_price,
                                             stat_error,
                                             z_score_limit);
        if (!run_passed) {
            std::cout << indent << "[   INFO   ] Numerical results do not match reference ones. Try refining dt. \n";
        }

        // Return the results of the test and go back to tester for next text
        std::cout << "\n" << indent << "--------------------------------------------------------------------\n";
        std::cout << indent << ">>> Go back to the tester ...\n";

        return {config.time.N_time_steps, config.time.inp_dt, signed_error, stat_error, finite_res, run_passed};
    }

    // Analyzes convergence data, prints diagnostics, and returns true if all Z-scores are within 3-sigma bounds.
    /**
     * @brief Audits weak convergence trends and performs statistical hypothesis testing.
     * @details Loops through a resolution history to compute the empirical convergence
     * rate ($\alpha$) between successive refinement steps:
     * $$\alpha = \frac{\ln(\text{Error}_2) - \ln(\text{Error}_1)}{\ln(\Delta t_2) - \ln(\Delta t_1)}$$
     * * Additionally, verifies that the remaining numerical bias at your highest resolution is safely
     * dominated by statistical Monte Carlo noise within a standard 3-sigma (99.7% confidence) boundary:
     * $$Z = \frac{\text{Absolute Error}}{\text{Standard Error}} \le 3.0$$
     * @param convergence_results Logged price error tracking array.
     * @param indent Console layout spacing alignment.
     * @return true if the highest-resolution error resides within statistical noise bounds; false otherwise. False is returned also if some results are not finite at any refinement level
     * @note Once the discretization error falls below the standard error, the total error stops shrinking linearly,
     * and it flattens out and begins jumping around randomly due to Monte Carlo noise.
    */
    [[nodiscard]] inline bool evaluate_quality_convergence_results(
        const std::vector<OptionPriceErr> &convergence_results,
        const std::string_view indent = "   ") {
        if (convergence_results.empty()) return false;

        bool convergence_test_passed = true;

        std::cout << indent << "\n::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n";
        std::cout << indent << "Analyzing convergence order for given model and scheme...\n";
        std::cout << indent <<
                "Total Error = Discretization Error (O(dt)) + Statistical Noise (O(std/sqrt(N_paths)))\n\n";

        for (size_t i = 0; i < convergence_results.size(); ++i) {
            const auto &current = convergence_results[i];

            std::string z_str = (current.stat_error > 1e-14)
                                    ? std::to_string(current.error / current.stat_error)
                                    : "INF";
            std::cout << indent << "[   INFO   ] Step " << i + 1
                    << ": Ndt=" << current.N_dt
                    << ", is_finite=" << current.is_finite
                    << ", ERR=" << current.error
                    << ", STAT ERR=" << current.stat_error
                    << ", Zscore=" << z_str
                    << ", within_stats=" << current.within_stat_bounds << "\n";

            // Abortining it a fatal math problem is found
            if (!current.is_finite || current.stat_error < 0.0) {
                if (!current.is_finite) {
                    std::cerr << indent << "[   Fatal   ] encountered invalid math (NaN/Inf)\n";
                } else {
                    std::cerr << indent << "[   Fatal   ] encountered invalid math (Negative StdError)\n";
                }
                return false;
            }


            // Compute convergence order only if there is a subsequent step to compare against
            if (i < convergence_results.size() - 1) {
                const auto &next = convergence_results[i + 1];
                const double log_dt_diff = std::log(next.dt) - std::log(current.dt);
                const double log_err_diff = std::log(std::abs(next.error)) - std::log(std::abs(current.error));
                const double order = log_err_diff / log_dt_diff;

                // Show convergence and warn if the error hit the statistical noise floor
                if (std::abs(current.error) > current.stat_error) {
                    std::cout << indent << "[   INFO   ] Step " << i + 1 << " -> " << i + 2
                            << " Convergence Order: " << order << "\n";
                } else {
                    std::cout << indent << "[   INFO   ] Step " << i + 1 << " -> " << i + 2
                            << " Convergence Order: " << order << " (Warning :  Statistical error dominate) \n";
                }
            } else {
                // Enter this block only if the last iteration (max resolution) is considered
                // If the error exceeds 3 standard deviations from max resultion, the test mathematically fails.
                convergence_test_passed = current.within_stat_bounds;
            }
        }

        std::cout << indent << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n";

        return convergence_test_passed;
    }


    /**
     * @brief Automates a multi-model weak convergence analysis.
     * @details Executes a grid testing across specified step resolutions for each target
     * model-scheme combination. It compares the numerical results against analytical benchmarks
     * and evaluates whether their error profiles satisfy empirical convergence limits.
     * @param test_name Displayed header title describing the active testing block.
     * @param config Base model configuration structure.
     * @param expected_option_price Analytical target price used as validation ground-truth.
     * @param models_to_test Collection of model-scheme configuration pairs.
     * @param time_grid_resolutions Sequence of step resolution sizes to evaluate.
     * @param indent Console layout spacing alignment.
     * @return true if all tested configurations satisfy mathematical convergence parameters for the highest resolution used; false otherwise.
     */
    [[nodiscard]] inline bool run_weak_convergence_test(
        const std::string_view test_name,
        KC::UInputs config,
        const double expected_option_price,
        const std::vector<std::pair<KI::MathModel, KI::NumScheme> > &models_to_test,
        const std::vector<int> &time_grid_resolutions,
        const std::string_view indent = "   ") {
        std::cout << "\n\n" << indent << "====================================================================\n"
                << indent << "          " << test_name << "\n"
                << indent << "====================================================================\n";

        std::cout << indent << "[   INFO   ] Target Exact Price : " << expected_option_price << "\n";

        bool all_tests_passed = true;

        // Loop over models
        for (const auto &model: models_to_test) {
            config.model.id_model = model.first;
            config.scheme.id_scheme = model.second;

            std::vector<OptionPriceErr> convergence_results;

            // Loop over time steps
            for (int N_Tsteps: time_grid_resolutions) {
                config.time.N_time_steps = N_Tsteps;
                config.time.inp_dt = config.time.t_end / N_Tsteps;

                std::cout << indent << "[   INFO   ] MODEL   : " << KI::enum_to_string(config.model.id_model) << "\n";
                std::cout << indent << "[   INFO   ] SCHEME  : " << KI::enum_to_string(config.scheme.id_scheme) << "\n";
                std::cout << indent << "[   INFO   ] dt: " << std::fixed << std::setprecision(6) << config.time.inp_dt
                        << "\n";

                // Run simulation
                auto convergence_res = compare_numerical_and_expected_option_price(config, expected_option_price);
                convergence_results.push_back(convergence_res);
            }

            // Check If numerical results are correct for this specific model/scheme combo
            const bool model_passed = evaluate_quality_convergence_results(convergence_results);
            if (!model_passed) {
                all_tests_passed = false; // Mark the whole suite as failed if one combo fails
            }
        }

        // Print final message and exit function
        std::cout << "\n" << indent << "====================================================================\n";
        if (all_tests_passed) {
            std::cout << indent << "[  PASSED  ] " << test_name << " successfully converged.\n";
        } else {
            std::cerr << indent << "[  FAILED  ] " << test_name << " drifted outside statistical bounds.\n";
        }
        std::cout << indent << "====================================================================\n\n";

        return all_tests_passed;
    }
}
