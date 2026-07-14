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

/**
 * @namespace KOps::Tests::LSM
 * @brief Integration tests for validating American option pricing using the Longstaff-Schwartz Method (LSM).
 */
namespace KOps::Tests::LSM {
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KTU = KOps::Tests::Utils;
    namespace KE = KOps::Engine;

    /**
     * @brief Generates an American Put option configuration under Heston stochastic volatility parameters.
     * @details This setup reproduces the benchmark test case analyzed on page 212 of Fabrice D. Rouah's
     * *The Heston Model and its Extensions in Matlab and C#*.
     * * ### Stochastic Process & Pricing Dynamics
     * - **Underlying Asset**: American Put option with strike price $K = 10.0$ and maturity $T = 0.25$ (3 months).
     * - **Feller Condition Check**:
     * $$ 2\kappa\theta \ge \sigma^2 $$
     * Plugging in the parameters ($\kappa = 5.0, \theta = 0.16, \sigma = 0.9$):
     * $$ 2 \cdot 5.0 \cdot 0.16 = 1.60 \ge 0.90^2 = 0.81 $$
     * Since the inequality holds ($1.60 > 0.81$), the Feller condition is strictly satisfied, preventing the
     * variance process $v_t$ from reaching the zero boundary.
     * - **Discretization**: Milstein discretization is selected to evolve the paths across a high-resolution grid of
     * $1000$ steps ($\Delta t = 0.00025$ years).
     * @return A populated configuration with parameters optimized for American option pricing benchmarks.
     */
    inline KC::UInputs getDefaultConfigGoodIntegrand() {
        // See pag 212, F.Rouah, The Heston Model and its Extensions in Matlab and C#
        auto config = KC::UInputs();

        config.model.id_model = KI::MathModel::Heston,
        config.scheme.id_scheme = KI::NumScheme::Milstein,

        config.output.format = KI::IOFormat::BIN;
        config.output.filename_paths_out = "";
        config.mc.Max_CPU_RAM_MB = 10000 ;
        config.mc.Max_VRAM_MB = 400 ;

        config.mc.N_Paths = 1'000'000;
        config.mc.batch_size = 0;

        config.time.t_end = 0.25; // 3 months
        config.time.inp_dt = config.time.t_end / 1000.;

        config.options.opt_type = KI::OptType::American;
        config.options.opt_right = KI::OptRight::Put;
        config.options.StrikePrice = 10.0;

        config.market.S0 = 8.;
        config.market.v0 = 0.0625; // Starting variance
        config.market.r = 0.1;
        config.market.q = 0.0;

        config.model.heston.k = 5.;
        config.model.heston.theta = 0.16;
        config.model.heston.sigma = 0.9;
        config.model.heston.rho = 0.1;

        return config;
    }


    /**
     * @brief Performs the LSM American Put pricing convergence test across a range of initial asset spots.
     * @details This test runs a parameter sweep for the initial spot price $S_0 \in \{8, 9, 10, 11, 12\}$
     * and compares the resulting numerical price against the exact finite difference reference values
     * published in Rouah (page 213, Table 11.2).
     * * ### The LSM Early Exercise Logic
     * At each discrete time step moving backward from maturity, the algorithm determines the optimal stopping boundary:
     * 1. Evaluates paths that are strictly In-The-Money (ITM) ($S_t < K$).
     * 2. Regresses discounted future cash flows $Y$ against normalized current spots $X = S_t / K$ to estimate the continuation value:
     * $$ \widehat{CV}_t(X) = \beta_0 + \beta_1 X + \beta_2 X^2 $$
     * 3. Triggers early exercise on any path where the immediate intrinsic payoff exceeds the estimated continuation value:
     * $$ (K - S_t) > \widehat{CV}_t(X) $$
     * * ### Error-Budget & Composite Tolerance Formulation
     * To evaluate whether a test passes, we construct a compound tolerance threshold combining statistical
     * uncertainty and numerical discretization bias:
     * $$ \text{Total Tolerance} = 3 \cdot \epsilon_{\text{stat}} + \max(V_{\text{ref}} \cdot 10\%, 0.1) $$
     * Where:
     * - $3 \cdot \epsilon_{\text{stat}}$ is the strict $3$-$\sigma$ (99.7% confidence interval) statistical error boundary.
     * - $\max(V_{\text{ref}} \cdot 10\%, 0.1)$ represents the maximum allowed discretization bias arising from
     * approximating continuous exercise boundaries with discrete time-steps.
     * * @return true if all swept spot prices fall within their composite statistical and bias thresholds for the maximum resolution adopted; false otherwise.
     */
    inline bool run_test() {
        const std::string_view indent = "   ";
        bool all_tests_passed = true;
        auto config = getDefaultConfigGoodIntegrand();

        // Reference values from Rouah, pag 213
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

            config.market.S0 = S0;
            std::cout << "\n\n" << indent << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
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
            const KT::Real bias_tolerance = std::max(expected_price * 0.1, 0.1);
            // Total allowed tolerance
            const KT::Real total_tolerance = stat_tolerance + bias_tolerance;


            std::cout << indent << " Expe price: " << expected_price
                    << " | Num Price: " << num_price
                    << " | Error: " << abs_error
                    << " | Stat Error: " << stat_error
                    << " | Bias: " << bias_tolerance
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
