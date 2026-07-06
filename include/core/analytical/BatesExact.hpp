#pragma once
#include <cmath>
#include <complex>
#include <stdexcept>

#include "./Integration.hpp"
#include "HestonExact.hpp"
#include "./../Typedefs.hpp"
#include "./../config/Config.hpp"


namespace KOps::Engine::Analytical::Bates {
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;
    namespace KI = KOps::Implemented;
    namespace KE = KOps::Engine::Analytical::Heston;

    using Complex = std::complex<double>;
    const Complex i_unit{0., 1.};

    // ========================================================================
    // 1. THE BATES INTEGRAND (COMPOSING HESTON + MERTON JUMPS)
    // ========================================================================
    inline double bates_integrand_albrecher_formulation(double phi, const KC::UInputs &conf, int j) {
        const auto bates = conf.model.bates;
        const auto m = conf.market;
        const double K = conf.options.StrikePrice;
        const double T = conf.time.t_end;

        // 1. Get the continuous Heston exponent
        // Note: 'bates' naturally upcasts to a Heston struct.
        const Complex heston_exponent = KE::get_heston_characteristic_exponent(phi,
                                                                               m,
                                                                               bates,
                                                                               T,
                                                                               j);

        // 2. Calculate the Bates Jump Components (E_j)
        const double k_bar = std::exp(bates.mu_J + 0.5 * bates.sigma_J * bates.sigma_J) - 1.0;
        const double u_jump = (j == 1) ? 1.0 : 0.0;
        const double mu_J_j = (j == 1) ? bates.mu_J + bates.sigma_J * bates.sigma_J : bates.mu_J;

        const Complex jump_term_inner = std::exp(i_unit * phi * mu_J_j -
                                                 0.5 * bates.sigma_J * bates.sigma_J * phi * phi) - 1.0;
        const Complex E_j = bates.lambda_J * T * (-i_unit * phi * k_bar +
                                                  std::pow(1.0 + k_bar, u_jump) * jump_term_inner);

        // 3. Compute the characteristic function (Combine heston and merton exponents)
        const Complex f_j = std::exp(heston_exponent + E_j);

        // 4. Return the real part of the final integrand
        const Complex numerator = std::exp(-i_unit * phi * std::log(K)) * f_j;
        const Complex denominator = i_unit * phi;
        return std::real(numerator / denominator);
    }

    // ========================================================================
    // 2. THE BATES PROBABILITY
    // ========================================================================
    inline double Probability(const KC::UInputs &conf, const int j, const double phi_max = 100.0) {
        // Create a lambda that binds the configuration and j-index, not expected in the gauss-legendre implementation
        auto integrand = [&](const double phi) {
            return bates_integrand_albrecher_formulation(phi, conf, j);
        };

        double integral = 0.0;
        constexpr double chunk_size = 50.0; // Keep the subdomains dense!
        const int num_chunks = static_cast<int>(phi_max / chunk_size);

        // Integrate chunk by chunk
        for (int i = 0; i < num_chunks; ++i) {
            const double lower = i * chunk_size;
            const double upper = (i + 1) * chunk_size;
            integral += integrate_gl64(integrand, lower, upper);
        }

        return 0.5 + (1.0 / M_PI) * integral;
    }

    // ========================================================================
    // 3. BATES EU CALL OPTION PRICE
    // ========================================================================
    inline double get_exact_eu_call_option_price(const KC::UInputs &conf, const double upper_bound = 8000.) {
        const bool condition = (conf.options.opt_right == KI::OptRight::Call) &&
                               (conf.options.opt_type == KI::OptType::European);
        if (!condition) {
            throw std::runtime_error("Must consider a European Call option for the Bates analytical test!");
        }

        const auto m = conf.market;
        const double K = conf.options.StrikePrice;
        const double T = conf.time.t_end;

        double P1 = Probability(conf, 1, upper_bound);
        double P2 = Probability(conf, 2, upper_bound);

        return m.S0 * std::exp(-m.q * T) * P1 - K * std::exp(-m.r * T) * P2;
    }
}
