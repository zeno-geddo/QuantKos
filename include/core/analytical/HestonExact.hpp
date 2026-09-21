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
#include <cmath>
#include <complex>
#include <stdexcept>

#include "./Integration.hpp"
#include "./../Typedefs.hpp"
#include "./../config/Config.hpp"

/**
 * @brief Namespace aggregating all tools needed to compute the exact Europen call option price considering the Heston Model.
 */
namespace quantkos::Engine::Analytical::Heston {
    namespace KT = quantkos::Types;
    namespace KC = quantkos::Config;
    namespace KI = quantkos::Implemented;

    using Complex = std::complex<double>;
    const Complex i_unit{0., 1.};

    // ========================================================================
    // 0. COMPUTE EXPONENT HESTON CHARACTERISTIC FUCNTION (ALBRECHER FORMULATION)
    // Calculates the exponent: C + D*v0 + i*phi*ln(S0)
    // Note : Can be reused for the bates model
    // ========================================================================

    /**
    * @brief Computes the complex exponent of the Heston characteristic function.
    *
    * Implements the Albrecher formulation (see Chapter 2, F. Rouah) to guarantee
    * numerical stability during integration. This function isolated from the full
    * characteristic function calculation to allow direct reuse inside the Bates jump model.
    *
    * @param phi The integration variable (frequency).
    * @param m Market configuration parameters (S0, v0, r, q).
    * @param p Heston model parameters (k, theta, sigma, rho).
    * @param T Time to maturity.
    * @param j Probability component index (1 or 2).
    * @return The complex exponent value ($C + D \cdot v_0 + i \cdot \phi \cdot \ln(S_0)$).
    */
    inline Complex get_heston_characteristic_exponent(const double phi,
                                                      const KC::MarketConfig &m,
                                                      const KC::MathModelConfig::Heston &p,
                                                      const double T, const int j) {
        // Promote input configuration fields to double to prevent template
        // argument deduction failures when mixing with std::complex<double> in single-precision builds.
        const double rho   = static_cast<double>(p.rho);
        const double sigma = static_cast<double>(p.sigma);
        const double k     = static_cast<double>(p.k);
        const double theta = static_cast<double>(p.theta);

        const double r     = static_cast<double>(m.r);
        const double q     = static_cast<double>(m.q);
        const double v0    = static_cast<double>(m.v0);
        const double S0    = static_cast<double>(m.S0);

        // Heston specific parameters for probabilities P1 and P2 (After Eq 1.40, F. Rouah)
        const double u = (j == 1) ? 0.5 : -0.5;
        const double b = (j == 1) ? k - rho * sigma : k;

        // 1. Calculate 'd' (Eq 2.54, F. Rouah)
        const Complex term1_d = std::pow(rho * sigma * i_unit * phi - b, 2.0);
        const Complex term2_d = sigma * sigma * (2.0 * u * i_unit * phi - phi * phi);
        const Complex d = std::sqrt(term1_d - term2_d);

        // 2. Calculate 'c' using the Albrecher Fix (Eq 2.15, F. Rouah)
        const Complex num_c(b - rho * sigma * i_unit * phi - d);
        const Complex denum_c(b - rho * sigma * i_unit * phi + d);
        const Complex c = num_c / denum_c;

        // 3. Calculate 'C' (Eq 2.17, F. Rouah)
        const Complex term1_C = (r - q) * i_unit * phi * T;
        const Complex term2_C_fact = (k * theta) / (sigma * sigma);
        const Complex term2_C_term1 = (b - rho * sigma * i_unit * phi - d) * T;
        const Complex term2_C_term2 = 2.0 * std::log((1.0 - c * std::exp(-d * T)) / (1.0 - c));
        const Complex term2_C = term2_C_fact * (term2_C_term1 - term2_C_term2);
        const Complex C = term1_C + term2_C;

        // 4. Calculate 'D' (Eq 2.14, F. Rouah)
        const Complex D_factor1 = (b - rho * sigma * i_unit * phi - d) / (sigma * sigma);
        const Complex D_factor2 = (1.0 - std::exp(-d * T)) / (1.0 - c * std::exp(-d * T));
        const Complex D = D_factor1 * D_factor2;

        // 5. Return eponent of the Characteristic Function f_j(phi) (Eq 1.48, F. Rouah)
        const double x_t = std::log(S0);
        return C + D * v0 + i_unit * phi * x_t;
    }

    // ========================================================================
    // 1. THE HESTON INTEGRAND (ALBRECHER FORMULATION)
    // ========================================================================
    /**
     * @brief Evaluates the real part of the Heston option pricing integrand.
     *
     * @param phi The integration variable (frequency).
     * @param conf Unified inputs configuration containing market and option parameters.
     * @param j Probability component index (1 or 2).
     * @return The real value of the integrated component ($\text{Re}[\frac{e^{-i\phi \ln(K)}f_j(\phi)}{i\phi}]$).
     */
    inline double heston_integrand_albrecher_formulation(double phi, const KC::UInputs &conf, int j) {
        const auto heston_p = conf.model.heston;
        const auto market_p = conf.market;
        const double K = conf.options.StrikePrice;
        const double T = conf.time.t_end;

        // 1. Build the Characteristic Function f_j(phi) (Eq 1.48, F. Rouah)
        const Complex exponent = get_heston_characteristic_exponent(phi, market_p, heston_p, T, j);
        const Complex f_j = std::exp(exponent);

        // 2. Return the real part of the final integrand (Eq 2.13, F. Rouah)
        const Complex numerator = std::exp(-i_unit * phi * std::log(K)) * f_j;
        const Complex denominator = i_unit * phi;
        return std::real(numerator / denominator);
    }

    // ========================================================================
    // 2. THE HESTON PROBABILITY
    // ========================================================================
    /**
     * @brief Calculates the semi-analytic risk-neutral probability $P_j$ via numerical quadrature, considering the Heston model.
     *
     * Binds the formulation integrand into a local lambda functor and evaluates the
     * definite integral using a 64-point Gauss-Legendre integration routine.
     *
     * @param conf Unified inputs configuration containing all execution parameters.
     * @param j Probability component index (1 or 2).
     * @param phi_max The upper truncation bound for the infinite integral numerical limit (default: 100.0).
     * @return The resulting risk-neutral probability scaling between 0.0 and 1.0.
     */
    inline double Probability(const KC::UInputs &conf, const int j, const double phi_max = 100.0) {
        // Create a lambda that binds the configuration and j-index, not expected in the gauss-legendre implementation
        // The compiler should create a functor behind the scenes
        auto integrand = [&](double phi) {
            return heston_integrand_albrecher_formulation(phi, conf, j);
        };

        // Call the numerical integrator
        const double integral = integrate_gl64(integrand, 0.0, phi_max);

        // Compute and return the integral (Eq. , F. Rouah)
        return 0.5 + (1.0 / M_PI) * integral;
    }


    // ========================================================================
    // 3. HESTON EU CALL OPTION PRICE
    // ========================================================================
    /**
    * @brief Computes the analytical European Call price using the Heston model.
    *
    * Uses numerical integration (Gauss-Legendre) of the Heston characteristic function,
    * truncating the infinite integral at the specified upper bound.
    *
    * @param conf Configuration struct containing all parameters given by the user.
    * @param upper_bound The upper integration limit for the characteristic function (default: 100.0).
    * @return Theoretical European Call price.
    * @throw std::runtime_error If the option configuration is not a European Call.
    */
    inline double get_exact_eu_call_option_price(const KC::UInputs &conf, const double upper_bound = 100.) {
        // Check that a european call is considered
        const bool condition = (conf.options.opt_right == KI::OptRight::Call) and (
                                   conf.options.opt_type == KI::OptType::European);
        if (!condition) {
            throw std::runtime_error(
                "Must consider a European Call option for the Heston weak convergence test !");
        }

        const double S0 = static_cast<double>(conf.market.S0);
        const double q  = static_cast<double>(conf.market.q);
        const double r  = static_cast<double>(conf.market.r);
        const double K  = static_cast<double>(conf.options.StrikePrice);
        const double T  = static_cast<double>(conf.time.t_end);


        // Compute the probabilities (The infinite integral is truncated at phi_max = upper_bound).
        const double P1 = Probability(conf, 1, upper_bound);
        const double P2 = Probability(conf, 2, upper_bound);

        return S0 * std::exp(-q * T) * P1 - K * std::exp(-r * T) * P2;
    }
}
