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
#include "../../core/analytical/HestonExact.hpp"
#include "TestsUtils.hpp"

/**
 * @namespace KOps::Tests::Heston
 * @brief Integration tests comparing numerical simulation results with Heston exact analytical solutions.
 */
namespace KOps::Tests::Heston {
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KTU = KOps::Tests::Utils;
    namespace KH = KOps::Engine::Analytical::Heston;

    /**
     * @brief Generates a Heston model configuration where the Feller condition is satisfied.
     * @details This setup corresponds to the standard test parameters described on page 28 of
     * Fabrice D. Rouah's *The Heston Model and its Extensions in Matlab and C#*.
     * * ### The Feller Condition Safeguard
     * Under the Heston model, the variance process $v_t$ is defined as:
     * $$dv_t = \kappa(\theta - v_t)dt + \sigma \sqrt{v_t} dW_t^v$$
     * The **Feller condition** determines whether the variance process can reach zero:
     * $$2\kappa\theta \ge \sigma^2$$
     * * Splicing our parameters into this inequality:
     * $$2 \cdot 5.0 \cdot 0.05 = 0.50 \ge 0.50^2 = 0.25$$
     * * Since the inequality holds ($0.50 > 0.25$), the Feller condition is strictly satisfied.
     * Mathematically, this guarantees that the variance process $v_t$ remains strictly positive ($v_t > 0$)
     * and never touches the boundary at zero.
     * * @return A populated configuration with parameters optimized for stable Heston model testing.
     */
    inline KC::UInputs getDefaultConfigGoodIntegrand() {
        // See pag 28, F.Rouah, The Heston Model and its Extensions in Matlab and C#
        // Expected European call price : 6.2528

        // Feller Condition : 2*k*theta >= sigma*sigma
        // The feller condition is satisfied in this test


        auto config = KC::UInputs();

        config.output.format = KI::IOFormat::BIN;
        config.output.filename_paths_out = "";

        config.mc.N_Paths = 1'000'000;
        config.mc.batch_size = 0;

        config.time.t_end = 0.5; // 6 months

        config.options.opt_type = KI::OptType::European;
        config.options.opt_right = KI::OptRight::Call;
        config.options.StrikePrice = 100.0;

        config.market.S0 = 100.;
        config.market.v0 = 0.05; // Starting variance
        config.market.r = 0.03;
        config.market.q = 0.02;

        // Heston
        config.model.heston.k = 5.;
        config.model.heston.theta = 0.05;
        config.model.heston.sigma = 0.5;
        config.model.heston.rho = -0.8;

        // Bates
        config.model.bates.k = 5.;
        config.model.bates.theta = 0.05;
        config.model.bates.sigma = 0.5;
        config.model.bates.rho = -0.8;
        config.model.bates.lambda_J = 0.;
        config.model.bates.mu_J = 0.;
        config.model.bates.sigma_J = 0.;

        return config;
    }


    /**
     * @brief Runs the weak convergence test comparing numerical option prices to the Heston exact analytical price.
     * @details This test systematically evaluates the weak convergence of our SDE numerical solvers
     * (e.g., Euler-Maruyama, Milstein, and Andersen QE) under stochastic volatility.
     * * The exact reference option price is calculated semi-analytically using the Heston closed-form pricing engine:
     * $$C(S_0, v_0, t) = S_0 P_1 - K e^{-rT} P_2$$
     * where $P_1$ and $P_2$ represent the in-the-money probabilities computed via characteristic functions.
     * * The test sweeps across a sequence of discrete time-grid resolutions (e.g., $2^3$ to $2^9$ steps) and checks
     * that the discretization error declines in proportion to the time step $\Delta t$, ensuring our parallel GPU/CPU
     * solvers are mathematically consistent.
     * * @return true if the numerical configurations successfully converge within statistical limits for the higher resolution considered; false otherwise.
     */
    inline bool run_test() {
        std::string id_test {"TEST 5 : Heston Weak Convergence to Heston SDE exact option price"};
        auto config = getDefaultConfigGoodIntegrand();
        const KT::Real exact_price = KH::get_exact_eu_call_option_price(config);
        return KTU::run_weak_convergence_test(id_test,
                                              config,
                                              exact_price,
                                              KTU::get_models_to_test(),
                                              KTU::get_time_grid_resolutions());
    }
}


