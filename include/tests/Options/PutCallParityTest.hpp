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

#include "../../core/config/Config.hpp"
#include "TestsUtils.hpp"
#include <cmath>
#include <iostream>
#include <string_view>

/**
 * @brief Integration tests for validating Monte Carlo finite-difference Greeks.
 */
namespace quantkos::Tests::Greeks {
    namespace KC = quantkos::Config;
    namespace KT = quantkos::Types;
    namespace KI = quantkos::Implemented;
    namespace KTU = quantkos::Tests::Utils;
    namespace KE = quantkos::Engine;

    /**
     * @brief Generates a baseline European option configuration under Heston dynamics.
     * @details Configures a fast-executing European option to validate the finite-difference
     * Greeks engine. The European option is chosen because standard Put-Call Parity identities
     * hold perfectly true for it across any underlying stochastic model (including Heston).
     * @return A populated configuration with all Greek computation flags enabled.
     */
    inline KC::UInputs getDefaultConfigGreekParity() {
        auto config = KC::UInputs();

        // Model & Scheme (Heston to prove model-independence of parity)
        config.model.id_model = KI::MathModel::Heston;
        config.scheme.id_scheme = KI::NumScheme::AndersonQE; // QuasiExponential scheme since it is the more precise

        // Memory & Output
        config.output.filename_paths_out = ""; // NO out
        config.mc.Max_CPU_RAM_MB = 8000;
        config.mc.Max_VRAM_MB = 1024;

        // MC Settings
        config.mc.normalize_prices = true;
        config.mc.N_Paths = 1'000'000; // High path count to stabilize Gamma and Theta noise
        config.mc.batch_size = 0; // Size chosen automatically

        // Enable ALL Greeks
        config.mc.compute_delta_et_gamma = true;
        config.mc.compute_vega_et_vomma = true;
        config.mc.compute_rho = true;
        config.mc.compute_theta = true;

        // Greek Bumps (Standard configurations)
        config.mc.spot_price_relative_bump_size = 0.001;
        config.mc.volatility_absolute_bump_size = 0.005; // volatility = sqrt(variance)
        config.mc.risk_free_rate_absolute_bump_size = 0.0001;
        config.mc.time_absolute_bump_size = 1.0 / 365.0;

        // Time
        config.time.t_end = 1.0; // 1 Year
        config.time.inp_dt = config.time.t_end / 365.0;

        // Option (European exactly obeys standard parity)
        config.options.opt_type = KI::OptType::European;
        config.options.StrikePrice = 100.0;

        // Market
        config.market.S0 = 100.0; // At-The-Money
        config.market.v0 = 0.04; // 20% volatility squared
        config.market.r = 0.05; // 5% rate
        config.market.q = 0.0; // No dividends for simplicity

        // Heston Params
        config.model.heston.k = 2.0;
        config.model.heston.theta = 0.04;
        config.model.heston.sigma = 0.3;
        config.model.heston.rho = -0.7;

        return config;
    }


    // Define a dedicated Tolerance Calculator
    /**
     * @brief Practical tolerance for structural Monte Carlo Greek parity tests.
     *
     * This is intentionally NOT a rigorous confidence interval for the Greek.
     * It is a regression/sanity-test tolerance designed to detect significant
     * finite-difference, configuration, or implementation errors while allowing
     * for ordinary Monte Carlo numerical noise.
     *
     * @param base_option_price_se Standard error of the base option price.
     * @param greek_scale Characteristic scale of the parity being tested.
     * @param relative_tolerance Relative structural tolerance (if the result is too far from the expected scale, reject it).
     * @param absolute_floor Minimum absolute tolerance.
     * @return Practical tolerance for structural Monte Carlo Greek parity tests
     */
    inline double compute_greek_tol(
        const double base_option_price_se,
        const double greek_scale,
        const double base_option_price_se_multiplier = 3.0,
        const double relative_tolerance = 0.01,
        const double absolute_floor = 1e-4)
    {
        const double scale_tolerance = relative_tolerance * std::max(std::abs(greek_scale), 1.0);

        const double mc_tolerance = base_option_price_se_multiplier * base_option_price_se;

        return std::max({scale_tolerance, mc_tolerance, absolute_floor});
    }

    // Helper lambda to check and print results
    /**
     * @brief Helper function to see if the pull call parity has been respected
     * @param name name of the greek considered
     * @param num parity (G_call - G_put) obtained numerically
     * @param expected parity (G_call - G_put) expected
     * @param tol tolerance between num and expected
     * @param indent spacing for stdout
     * @return bool saying if the test has passed or not
     */
    inline bool check_parity(const std::string &name,
                             const double num,
                             const double expected,
                             const double tol,
                             const std::string_view &indent) {
        const double error = std::abs(num - expected);
        std::cout << indent << " " << name << " Parity | Expected: " << expected
                << " | Computed: " << num << " | Error: " << error << " | Tol: " << tol << "\n";

        if (error > tol) {
            std::cout << "     [ FAILED ]\n";
            return false;
        }
        std::cout << "     [ PASSED ]\n";
        return true;

    };


    /**
     * @brief Performs the Structural Put-Call Parity test for Monte Carlo Greeks.
     *
     * @details Runs the exact same configuration as a Call and a Put, then evaluates:
     * 1. Delta Parity : $\Delta_C - \Delta_P = 1$
     * 2. Gamma Parity : $\Gamma_C = \Gamma_P$
     * 3. Vega Parity  : $\mathcal{V}_C = \mathcal{V}_P$
     * 4. Vomma Parity : \text{Vomma}_C = \text{Vomma}_P
     * 5. Rho Parity   : $\rho_C - \rho_P = K \cdot T \cdot e^{-rT}$
     * 6. Theta Parity : $\Theta_C - \Theta_P = -r \cdot K \cdot e^{-rT}$
     * where K is the strike price.
     *
     * @note Because finite differences on Monte Carlo paths introduce statistical noise,
     * the test enforces a small numerical tolerance rather than exact floating-point equality.
     * in this implementation, it is only required that the parity residuals must be small compared with reasonable numerical scales and the MC price noise.
     *
     * @note The test purpose is to answer the question :
     *      -> Are the call and put Greeks behaving structurally correctly?
     *      not something more rigorous like :
     *      -> Can I prove at 99.7% confidence that the Greek parity residual is statistically insignificant?
     *
     *
     * @return true if all computed parities fall within an empirical tolerance, false otherwise.
     */
    inline bool run_test() {
        constexpr std::string_view indent = "   ";
        auto config = getDefaultConfigGreekParity();
        config.validate();

        std::cout << "\n\n" << indent << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
        std::cout << indent << "[   INFO   ] Initiating Greek Put-Call Parity Test \n";

        // 1. Run CALL Simulation
        config.options.opt_right = KI::OptRight::Call;
        std::cout << indent << ">>> Running CALL configuration...\n";
        const auto call_res = KTU::run_simulation(config);

        // 2. Run PUT Simulation
        config.options.opt_right = KI::OptRight::Put;
        std::cout << indent << ">>> Running PUT configuration...\n";
        const auto put_res = KTU::run_simulation(config);

        std::cout << "\n" << indent << "--------------------------------------------------------------------\n";
        std::cout << indent << ">>> Evaluating Greek's Call-Put Parities ...\n";

        // Target Expected Values based on continuous-time finance
        const double K = config.options.StrikePrice;
        const double T = config.time.t_end;
        const double r = config.market.r;
        const double discount = std::exp(-r * T);

        constexpr double expected_delta_diff = 1.0;
        constexpr double expected_gamma_diff = 0.0;
        constexpr double expected_vega_diff = 0.0;
        constexpr double expected_vomma_diff = 0.0;
        const double expected_rho_diff = K * T * discount;
        const double expected_theta_diff = -r * K * discount;

        // Computed Values
        const auto &C = call_res.Greeks;
        const auto &P = put_res.Greeks;

        const double num_delta_diff = C.delta - P.delta;
        const double num_gamma_diff = C.gamma - P.gamma;
        const double num_vega_diff = C.vega - P.vega;
        const double num_vomma_diff = C.vomma - P.vomma;
        const double num_rho_diff = C.rho - P.rho;
        const double num_theta_diff = C.theta - P.theta;

        // Pre-compute practical tolerance for each Greek
        const double base_se = call_res.OptionPrice.standard_error; // Extract base standard error from the option price
        const double tol_delta = compute_greek_tol(base_se, 1., 3.3);
        const double tol_gamma = compute_greek_tol(base_se, 1., 3.3);
        const double tol_vega = compute_greek_tol(base_se, 1., 3.3);
        const double tol_vomma = compute_greek_tol(base_se, 1., 8.);
        const double tol_rho = compute_greek_tol(base_se, std::abs(expected_rho_diff), 3.3);
        const double tol_theta = compute_greek_tol(base_se, std::abs(expected_theta_diff), 5.); // Higher bias for grid shift

        // Check if the parities are ok
        const bool all_tests_passed = std::min({
            check_parity("Delta", num_delta_diff, expected_delta_diff, tol_delta, indent),
            check_parity("Gamma", num_gamma_diff, expected_gamma_diff, tol_gamma, indent),
            check_parity("Vega ", num_vega_diff, expected_vega_diff, tol_vega, indent),
            check_parity("Vomma ", num_vomma_diff, expected_vomma_diff, tol_vomma, indent),
            check_parity("Rho  ", num_rho_diff, expected_rho_diff, tol_rho, indent),
            check_parity("Theta", num_theta_diff, expected_theta_diff, tol_theta, indent)
        }
        );

        if (all_tests_passed) {
            std::cout << "\n" << indent << "[ SUCCESS ] All Greek finite-difference engines are structurally sound.\n";
        } else {
            std::cerr << "\n" << indent << "[ FAILURE ] One or more Greeks violated Put-Call Parity bounds!\n";
        }

        std::cout << indent << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";

        return all_tests_passed;
    }
}
