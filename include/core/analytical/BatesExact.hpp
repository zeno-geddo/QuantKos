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
        const double K = conf.options.StrikePrice;
        const double T = conf.time.t_end;

        // 1. Get the continuous Heston exponent
        // Note: 'bates' naturally upcasts to a Heston struct.
        const Complex heston_exponent = KE::get_heston_characteristic_exponent(phi,
                                                                               bates,
                                                                               T,
                                                                               conf.init.S0,
                                                                               conf.init.v0, j);

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
        auto integrand = [&](double phi) {
            return bates_integrand_albrecher_formulation(phi, conf, j);
        };

        // Compute and return the integral (Eq. , F. Rouah)
        double integral = integrate_gl64(integrand, 0.0, phi_max);
        return 0.5 + (1.0 / M_PI) * integral;
    }

    // ========================================================================
    // 3. BATES EU CALL OPTION PRICE
    // ========================================================================
    inline double get_exact_eu_call_option_price(const KC::UInputs &conf, const double upper_bound = 1000.) {
        const bool condition = (conf.options.opt_right == KI::OptRight::Call) &&
                               (conf.options.opt_type == KI::OptType::European);
        if (!condition) {
            throw std::runtime_error("Must consider a European Call option for the Bates analytical test!");
        }

        const auto bates = conf.model.bates;
        const double K = conf.options.StrikePrice;
        const double T = conf.time.t_end;
        const double S0 = conf.init.S0;

        double P1 = Probability(conf, 1, upper_bound);
        double P2 = Probability(conf, 2, upper_bound);

        return S0 * std::exp(-bates.q * T) * P1 - K * std::exp(-bates.r * T) * P2;
    }
}
