#pragma once

#include <stdexcept>

#include "../IO/Config.hpp"
#include "../core/Distpatcher.hpp"


namespace KOps::Tests::Utils {
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KE = KOps::Engine;


    inline std::vector<std::pair<KI::MathModel, KI::NumScheme>> get_models_to_test() {
        static const std::vector<std::pair<KI::MathModel, KI::NumScheme>> models_to_test = {
            {KI::MathModel::Heston, KI::NumScheme::Euler},
            {KI::MathModel::Heston, KI::NumScheme::Milstein}
        };
        return models_to_test;
    }


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

    [[nodiscard]] inline KE::MCResults run_simulation(KC::UInputs &config) {
        // Validate and print configuration
        config.validate();
        config.print_summary();
        // Dispatch and execute
        auto MCDisp = KE::MCDispatcher(config);
        return MCDisp.launch_montecarlo();
    }

    struct OptionPriceErr {
        int N_dt;
        double dt;
        double error;
        double stat_error;
    };

    [[nodiscard]] inline OptionPriceErr compare_numerical_and_expected_option_price(
        KOps::Config::UInputs &config,
        const KT::Real expected_price,
        const std::string_view indent = "   ") {
        std::cout << indent << ">>> Calling the solver ...\n";
        std::cout << "\n" << indent << "--------------------------------------------------------------------\n";

        const auto MCRes = run_simulation(config);
        const KT::Real num_price = MCRes.OptionPrice.option_price;
        const KT::Real stat_error = MCRes.OptionPrice.standard_error;
        const KT::Real abs_error = std::abs(num_price - expected_price);

        std::cout << indent << " Price: " << num_price
                << " | Error: " << abs_error
                << " | Stat Error: " << stat_error << "\n";

        std::cout << "\n" << indent << "--------------------------------------------------------------------\n";
        std::cout << indent << ">>> Go back to the tester ...\n";

        return {config.time.N_time_steps, config.time.inp_dt, abs_error, stat_error};
    }

    // Analyzes convergence data, prints diagnostics, and returns true if all Z-scores are within 3-sigma bounds.
    [[nodiscard]] inline bool evaluate_quality_numerical_results(const std::vector<OptionPriceErr> &convergence_results,
                                                                 const std::string_view indent = "   ") {
        if (convergence_results.empty()) return false;

        bool test_passed = true;

        std::cout << indent << "\n::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n";
        std::cout << indent << "Analyzing convergence order for given model and scheme...\n";
        std::cout << indent <<
                "Total Error ^2 = Discretization Error (O(dt)) ^2 + Statistical Noise (O(std/sqrt(N_paths)) ^2)\n\n";

        for (size_t i = 0; i < convergence_results.size(); ++i) {
            const auto &current = convergence_results[i];
            const double z_score = current.error / current.stat_error;

            std::cout << indent << "[   INFO   ] Step " << i + 1
                    << ": Ndt= " << current.N_dt
                    << ", ERR= " << current.error
                    << ", STAT ERR= " << current.stat_error
                    << ", Zscore= " << z_score << "\n";

            // Compute convergence order only if there is a subsequent step to compare against
            if (i < convergence_results.size() - 1) {
                const auto &next = convergence_results[i + 1];
                const double log_dt_diff = std::log(next.dt) - std::log(current.dt);
                const double log_err_diff = std::log(next.error) - std::log(current.error);
                const double order = log_err_diff / log_dt_diff;

                std::cout << indent << "[   INFO   ] Step " << i + 1 << " -> " << i + 2
                        << " Convergence Order: " << order << "\n";
            } else {
                // If the error exceeds 2 standard deviations from max resultion, the test mathematically fails.
                if (z_score > 2.0) test_passed = false;
            }
        }

        std::cout << indent << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n";

        return test_passed;
    }


    [[nodiscard]] inline bool run_weak_convergence_test(
        const std::string_view test_name,
        KC::UInputs config,
        const KT::Real expected_option_price,
        const std::vector<std::pair<KI::MathModel, KI::NumScheme>> &models_to_test,
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
            bool model_passed = evaluate_quality_numerical_results(convergence_results);
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
