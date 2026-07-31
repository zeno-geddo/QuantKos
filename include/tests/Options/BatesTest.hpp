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
#include "../../core/analytical/BatesExact.hpp"
#include "TestsUtils.hpp"

/**
 * @brief Integration tests comparing numerical simulation results with Bates exact analytical solutions.
 */
namespace quantkos::Tests::Bates {
    namespace KC = quantkos::Config;
    namespace KT = quantkos::Types;
    namespace KI = quantkos::Implemented;
    namespace KTU = quantkos::Tests::Utils;
    namespace KB = quantkos::Engine::Analytical::Bates;

    /**
     * @brief Generates a Bates model configuration with standard jump-diffusion test parameters.
     * @details The Bates model extends the continuous Heston model by adding discrete, log-normal
     * asset price jumps (Merton jump-diffusion style).
     * * ### Stochastic Process Formulation
     * Under the Bates model, the joint dynamics of the asset price $S_t$ and its variance $v_t$ are:
     * $$ dS_t = (r - q - \lambda_J \kappa_J) S_t dt + \sqrt{v_t} S_t dW_t^S + S_{t^-} (e^{Y_t} - 1) dN_t $$
     * $$ dv_t = \kappa(\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v $$
     * * Where:
     * - $N_t$ is a homogeneous Poisson process with constant jump intensity $\lambda_J$.
     * - $Y_t \sim \mathcal{N}(\mu_J, \sigma_J^2)$ is the random log-jump magnitude.
     * - $\kappa_J = e^{\mu_J + \frac{1}{2}\sigma_J^2} - 1$ represents the expected relative change in price due to the jumps.
     * * ### Parameter Setup & Verification Stability
     * - **Volatility Core**: Uses the same stable Heston parameters where the Feller condition ($2\kappa\theta \ge \sigma^2$)
     * is satisfied, preventing variance from collapsing to zero in the pure Heston case.
     * - **Jump Parameters**: Configured with $\lambda_J = 0.11$ (roughly 1 jump every 9 years), $\mu_J = -0.15$ (an average
     * crash size of -15%), and $\sigma_J = 0.11$ (jump size uncertainty).
     * * @return A populated configuration with parameters optimized for Bates jump-diffusion pricing.
     */
    inline KC::UInputs getDefaultConfigGoodIntegrand() {
        // Heston params from pag 28, F.Rouah, The Heston Model and its Extensions in Matlab and C#
        // Bates params


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

        config.model.bates.k = 5.;
        config.model.bates.theta = 0.05;
        config.model.bates.sigma = 0.5;
        config.model.bates.rho = -0.8;// -0.8;

        config.model.bates.lambda_J = 0.11;  // Roughly 1 jump every ~9 years
        config.model.bates.mu_J = -0.15;     // When a jump happens, it averages a -15% drop (Crash)
        config.model.bates.sigma_J = 0.11;   // The volatility/uncertainty (std) of that jump size

        return config;
    }

    /**
     * @brief Returns a filtered registry containing only the Bates model and numerical scheme pairs.
     * @details Restricts the weak convergence sweep to Bates-specific implementations, mapping the
     * jump-diffusion processes across Euler-Maruyama, Milstein, and Andersen's QE integration schemes.
     * @return A vector of paired Bates model and scheme configurations.
     */
    inline std::vector<std::pair<KI::MathModel, KI::NumScheme> > get_bates_models_to_test() {
        static const std::vector<std::pair<KI::MathModel, KI::NumScheme> > models_to_test = {
            {KI::MathModel::Bates, KI::NumScheme::Euler},
            {KI::MathModel::Bates, KI::NumScheme::ImplicitMilstein},
            {KI::MathModel::Bates, KI::NumScheme::AndersonQE},
        };
        return models_to_test;
    }

    /**
     * @brief Runs the weak convergence test sweep comparing numerical option prices to the Bates exact analytical price.
     * @details This test systematically evaluates the weak convergence of our SDE solvers under jumps and
     * stochastic variance.
     * * The exact reference option price is calculated using the Bates closed-form characteristic function engine,
     * which integrates Merton's jump structure with the continuous Heston pricing framework.
     * * The convergence test runs across a series of step resolutions (e.g., $2^3$ to $2^9$), verifying that
     * jump-diffusion discretization remains mathematically consistent and converges within expected statistical confidence bands.
     * * @return true if all registered Bates models and schemes converge within acceptable tolerances for the maximum grid resolution considered; false otherwise.
     */
    inline bool run_test() {
        std::string id_test {"TEST 6 : Bates Weak Convergence to Bates SDE exact option price"};
        auto config = getDefaultConfigGoodIntegrand();
        const KT::Real exact_price = KB::get_exact_eu_call_option_price(config);
        return KTU::run_weak_convergence_test(id_test,
                                              config,
                                              exact_price,
                                              get_bates_models_to_test(),
                                              KTU::get_time_grid_resolutions());
    }
}


