#pragma once

#include "../../core/config/Config.hpp"
#include "TestsUtils.hpp"

namespace KOps::Tests::LSM {
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KTU = KOps::Tests::Utils;
    namespace KE = KOps::Engine;

    inline KC::UInputs getDefaultConfigGoodIntegrand() {
        // See pag 212, F.Rouah, The Heston Model and its Extensions in Matlab and C#
        auto config = KC::UInputs();

        config.model.id_model = KI::MathModel::Heston,
                config.scheme.id_scheme = KI::NumScheme::Milstein,

                config.output.format = KI::IOFormat::BIN;
        config.output.filename_paths_out = "";

        config.mc.N_Paths = 1'000'000;
        config.mc.batch_size = 0;

        config.time.t_end = 0.25; // 3 months
        config.time.inp_dt = config.time.t_end / 1000.;

        config.options.opt_type = KI::OptType::American;
        config.options.opt_right = KI::OptRight::Put;
        config.options.StrikePrice = 10.0;

        config.init.S0 = 8.;
        config.init.v0 = 0.0625; // Starting variance

        config.model.heston.r = 0.1;
        config.model.heston.q = 0.0;
        config.model.heston.k = 5.;
        config.model.heston.theta = 0.16;
        config.model.heston.sigma = 0.9;
        config.model.heston.rho = 0.1;
        // Feller Condition : 2*k*theta >= sigma*sigma
        // The feller condition is satisfied in this test

        return config;
    }


    inline bool run_test() {
        const std::string_view indent = "   ";
        bool all_tests_passed = true;
        auto config = getDefaultConfigGoodIntegrand();

        const std::map<double, double> S0_putPrice_map{
            {8., 1.99958},
            {9., 1.103571},
            {10., 0.519039},
            {11., 0.226137},
            {12., 0.082123},
        };

        for (const auto pair: S0_putPrice_map) {
            const double S0 = pair.first;
            const double expected_price = pair.second;

            config.init.S0 = S0;
            std::cout << indent << "[   INFO   ] S0  : " << S0 << " ; expected price : " << expected_price << " \n";

            // Run Simulation
            std::cout << indent << ">>> Calling the solver ...\n";
            std::cout << "\n" << indent << "--------------------------------------------------------------------\n";
            const auto MCRes = KTU::run_simulation(config);

            std::cout << "\n" << indent << "--------------------------------------------------------------------\n";
            std::cout << indent << ">>> Go back to the tester ...\n";

            // Check absolute erroror
            const KT::Real num_price = MCRes.OptionPrice.option_price;
            const KT::Real abs_error = std::abs(num_price - expected_price);

            // Statistical Tolerance (3-Sigma Rule: 99.7% Confidence Interval)
            const KT::Real stat_error = MCRes.OptionPrice.standard_error;
            const KT::Real stat_tolerance = 3.0 * stat_error;
            // Discretization Bias Tolerance
            const KT::Real bias_tolerance = std::max(expected_price * 0.005, 0.005git);
            // Total allowed tolerance
            const KT::Real total_tolerance = stat_tolerance + bias_tolerance;



            std::cout << indent << " Expe price: " << expected_price
                    << " | Num Price: " << num_price
                    << " | Error: " << abs_error
                    << " | Stat Error: " << stat_error
                    << " | Bias : " << bias_tolerance
                    << " | Allowed Tol Err: " << total_tolerance << "\n";

            if (abs_error > total_tolerance) {
                std::cerr << indent << "[  FAILED  ] Error exceeds composite tolerance!" << std::endl;
                all_tests_passed = false;
            } else {
                std::cout << indent << "[  PASSED  ] Result is within statistical bounds." << std::endl;
            }
        }
        return all_tests_passed;
    }
}
