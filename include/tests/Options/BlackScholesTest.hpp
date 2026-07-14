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
#include "../../core/analytical/BlackScholesExact.hpp"

/**
 * @namespace KOps::Tests::BlackScholes
 * @brief Integration tests validating convergence profiles against the analytical Black-Scholes model.
 */
namespace KOps::Tests::BlackScholes {
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KTU = KOps::Tests::Utils;
    namespace KBS = KOps::Engine::Analytical::BlackScholes;

    /**
     * @brief Generates a baseline configuration where stochastic volatility models (HESTON AND BATES) collapse to Black-Scholes.
     * @details Under the Heston stochastic volatility model, the variance process $v_t$ is defined as:
     * $$ dv_t = \kappa(\theta - v_t)dt + \sigma \sqrt{v_t} dW_t^v $$
     * By setting all volatility-of-volatility parameters to zero ($\kappa = \theta = \sigma = \rho = 0$),
     * the variance remains constant for the entire duration of the simulation:
     * $$ v_t = v_0 \quad \forall t \in [0, T] $$
     * Consequently, the asset price process $S_t$ collapses to standard Geometric Brownian Motion (GBM):
     * $$ dS_t = (r - q)S_t dt + \sqrt{v_0} S_t dW_t^S $$
     * This represents the exact Black-Scholes framework with constant volatility $\sigma_{\text{BS}} = \sqrt{v_0} = \sqrt{0.04} = 0.20$ (20%).
     * @return A populated configuration with parameters locked to constant-volatility GBM dynamics.
     * @note When consideteing the Bates model, all the Merton jumps parameters are set to zero.
    */
    inline KC::UInputs getDefaultConfig() {
        auto config = KC::UInputs();

        config.output.format = KI::IOFormat::BIN;
        config.output.filename_paths_out = "";

        config.mc.N_Paths = 1'000'000;
        config.mc.batch_size = 0;

        config.time.t_end = 1.;

        config.options.opt_type = KI::OptType::European;
        config.options.opt_right = KI::OptRight::Call;
        config.options.StrikePrice = 100.0;

        config.market.S0 = 100.;
        config.market.v0 = 0.04; // This implies the volatility is 0.20, similat to that of the S&P500
        config.market.r = 0.04;
        config.market.q = 0.;

        // Heston
        config.model.heston.k = 0.;
        config.model.heston.theta = 0.;
        config.model.heston.sigma = 0.;
        config.model.heston.rho = 0.;

        // Bates
        config.model.bates.k = 0.;
        config.model.bates.theta = 0.;
        config.model.bates.sigma = 0.;
        config.model.bates.rho = 0.;
        config.model.bates.lambda_J = 0.;
        config.model.bates.mu_J = 0.;
        config.model.bates.sigma_J = 0.;

        return config;
    }


    /**
     * @brief Executes the weak convergence test comparing numerical outcomes to Black-Scholes exact pricing.
     * @details This test leverages the test utility module to systematically evaluate the weak convergence order
     * of multiple numerical schemes (such as Euler, Milstein, and Andersen QE).
     * * Because the parameters are forced into a deterministic variance space, the exact analytical solution can be
     * computed using the Black-Scholes European option formula:
     * $$ C(S_0, t) = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2) $$
     * * The convergence runs the numerical solver across a logarithmic grid of step resolutions (e.g., $2^3$ to $2^9$),
     * checking that discretization error $\epsilon_d \propto O(\Delta t)$ declines towards zero,
     * and that the highest resolution result remains within statistical confidence bounds.
     * * @return true if all registered models and schemes successfully converge within expected statistical thresholds; false otherwise.
     */
    inline bool run_test() {
        std::string id_test {"TEST 4 : Heston Weak Convergence to Black-Scholes SDE exact option price"};
        auto config = getDefaultConfig();
        const KT::Real exact_price = KBS::get_exact_eu_call_option_price(config);
        return KTU::run_weak_convergence_test(id_test,
                                              config,
                                              exact_price,
                                              KTU::get_models_to_test(),
                                              KTU::get_time_grid_resolutions());
    }
}
