#pragma once

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <complex>
#include <vector>
#include <iomanip>

#include "./../core/Typedefs.hpp"
#include "../IO/Config.hpp"


namespace KOps::Tests::ExactSolutions {
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;
    namespace KI = KOps::Implemented;


    namespace BlackScholes {
        /**
        * @brief Eval CDF of a standard normal at point x.
        * @param x, position in the x domain.
        * @return Value of the Normal CDF at x.
        */
        inline KT::Real normalCDF(KT::Real x) {
            return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        }

        inline KT::Real get_exact_call_option_price(const KC::UInputs &conf) {
            // Assume considering European Call
            bool condition = (conf.options.opt_right == KI::OptRight::Call) and (
                                 conf.options.opt_type == KI::OptType::European);
            if (!condition) {
                throw std::runtime_error(
                    "Must consider a European Call option for the Black Scholes weak convergence test !");
            }

            KT::Real S = conf.init.S0;
            KT::Real K = conf.options.StrikePrice;
            KT::Real T = conf.time.t_end;
            KT::Real r = conf.model.heston.r;
            KT::Real q = conf.model.heston.q;
            KT::Real vol = std::sqrt(conf.init.v0); // Extract Volatility from Variance

            KT::Real d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
            KT::Real d2 = d1 - vol * std::sqrt(T);
            return S * std::exp(-q * T) * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
        }
    }


    namespace Heston {
        using Complex = std::complex<double>;
        const Complex i_unit{0., 1.};

        // ========================================================================
        // 1. THE HESTON INTEGRAND (ALBRECHER FORMULATION)
        // ========================================================================
        double integrand_albrecher_fromulation(double phi, const KC::UInputs &conf, int j) {
            const auto p = conf.model.heston;
            const double K = conf.options.StrikePrice;
            const double T = conf.time.t_end;
            const double S0 = conf.init.S0;
            const double v0 = conf.init.v0;

            // Heston specific parameters for probabilities P1 and P2 (After Eq 1.40, F. Rouah)
            double u = (j == 1) ? 0.5 : -0.5;
            double b = (j == 1) ? p.k - p.rho * p.sigma : p.k;

            // 1. Calculate 'd' (Eq 2.54, F. Rouah)
            const Complex term1_d = std::pow(p.rho * p.sigma * i_unit * phi - b, 2.0);
            const Complex term2_d = p.sigma * p.sigma * (2.0 * u * i_unit * phi - phi * phi);
            const Complex d = std::sqrt(term1_d - term2_d);

            // 2. Calculate 'c' using the Albrecher Fix (Eq 2.15, F. Rouah)
            const Complex num_c(b - p.rho * p.sigma * i_unit * phi - d);
            const Complex denum_c(b - p.rho * p.sigma * i_unit * phi + d);
            const Complex c = num_c / denum_c;

            // 3. Calculate 'C' (Eq 2.17, F. Rouah)
            const Complex term1_C = (p.r - p.q) * i_unit * phi * T;
            const Complex term2_C_fact = (p.k * p.theta) / (p.sigma * p.sigma);
            const Complex term2_C_term1 = (b - p.rho * p.sigma * i_unit * phi - d) * T;
            const Complex term2_C_term2 = 2.0 * std::log((1.0 - c * std::exp(-d * T)) / (1.0 - c));
            const Complex term2_C = term2_C_fact * (term2_C_term1 - term2_C_term2);
            const Complex C = term1_C + term2_C;

            // 4. Calculate 'D' (Eq 2.14, F. Rouah)
            const Complex D_factor1 = (b - p.rho * p.sigma * i_unit * phi - d) / (p.sigma * p.sigma);
            const Complex D_factor2 = (1.0 - std::exp(-d * T)) / (1.0 - c * std::exp(-d * T));
            const Complex D = D_factor1 * D_factor2;

            // 5. Build the Characteristic Function f_j(phi) (Eq 1.48, F. Rouah)
            const double x_t = std::log(S0);
            Complex f_j = std::exp(C + D * v0 + i_unit * phi * x_t);

            // 6. Return the real part of the final integrand (Eq 2.13, F. Rouah)
            const Complex numerator = std::exp(-i_unit * phi * std::log(K)) * f_j;
            const Complex denominator = i_unit * phi;
            return std::real(numerator / denominator);
        }

        double Probability(const KC::UInputs &conf, const int j, const double phi_max = 100.0) {
            // 32-point Gauss-Legendre roots (nodes) and weights for interval [-1, 1]
            // We only store the 16 positive roots because they are symmetric around 0.
            // From P. Manalastas, 'Computing 32-Place Tables of Zeroes and Weights for Gauss-Legendre Quadrature'
            // constexpr std::array<double, 16> gl_roots = {
            //     0.0483076656877, 0.1444719615827, 0.2392873622521, 0.3318686022821,
            //     0.4213512761306, 0.5068999089322, 0.5877157572407, 0.6630442669302,
            //     0.7321821187402, 0.7944837959679, 0.8493676137325, 0.8963211557660,
            //     0.9349060759377, 0.9647622555875, 0.9856115115452, 0.9972638618494
            // };
            // constexpr std::array<double, 16> gl_weights = {
            //     0.0965400885147, 0.0956387200792, 0.0938443990808, 0.0911738786957,
            //     0.0876520930044, 0.0833119242269, 0.0781938957870, 0.0723457941088,
            //     0.0658222227763, 0.0586840934785, 0.0509980592623, 0.0428358980222,
            //     0.0342738629130, 0.0253920653092, 0.0162743947309, 0.0070186100094
            // };

            // 64-point Gauss-Legendre roots (nodes) and weights for interval [-1, 1]
            // We only store the 32 positive roots because they are symmetric around 0.
            // From P. Manalastas, 'Computing 32-Place Tables of Zeroes and Weights for Gauss-Legendre Quadrature'
            constexpr std::array<double, 32> gl_roots = {
                0.0243502926634244, 0.0729931217877990, 0.1214628192961206, 0.1696444204239928,
                0.2174236437400071, 0.2646871622087674, 0.3113228719902110, 0.3572201583376681,
                0.4022701579639916, 0.4463660172534641, 0.4894031457070530, 0.5312794640198945,
                0.5718956462026340, 0.6111553551723933, 0.6489654712546573, 0.6852363130542332,
                0.7198818501716108, 0.7528199072605319, 0.7839723589433414, 0.8132653151227976,
                0.8406292962525804, 0.8659993981540928, 0.8893154459951141, 0.9105221370785028,
                0.9295691721319396, 0.9464113748584028, 0.9610087996520537, 0.9733268277899110,
                0.9833362538846260, 0.9910133714767443, 0.9963401167719553, 0.9993050417357721
            };

            constexpr std::array<double, 32> gl_weights = {
                0.0486909570091397, 0.0485754674415034, 0.0483447622348030, 0.0479993885964583,
                0.0475401657148303, 0.0469681828162100, 0.0462847965813144, 0.0454916279274181,
                0.0445905581637566, 0.0435837245293235, 0.0424735151236536, 0.0412625632426235,
                0.0399537411327203, 0.0385501531786156, 0.0370551285402400, 0.0354722132568824,
                0.0338051618371416, 0.0320579283548516, 0.0302346570724025, 0.0283396726142595,
                0.0263774697150547, 0.0243527025687109, 0.0222701738083833, 0.0201348231535302,
                0.0179517157756973, 0.0157260304760247, 0.0134630478967186, 0.0111681394601311,
                0.0088467598263639, 0.0065044579689784, 0.0041470332605625, 0.0017832807216964
            };

            double integral_sum = 0.0;

            // Rescaling the domain
            // phi(x) = x*((b-a)/2) + (a+b)/2
            // dphi/dx = (b-a) /2
            // Map [-1, 1] to [0, phi_max]
            const double scale = phi_max / 2.0;
            const double shift = phi_max / 2.0;

            for (size_t k = 0; k < gl_roots.size(); ++k) {
                // Positive node
                const double phi_pos = scale * gl_roots[k] + shift; // map the x values
                integral_sum += gl_weights[k] * integrand_albrecher_fromulation(phi_pos, conf, j);

                // Negative node (Mirrored)
                const double phi_neg = scale * (-gl_roots[k]) + shift; // map the x values
                integral_sum += gl_weights[k] * integrand_albrecher_fromulation(phi_neg, conf, j);
            }

            // Apply mapping scale (dx to dphi)
            integral_sum *= scale;

            // Compute the integral (Eq. )
            return 0.5 + (1.0 / M_PI) * integral_sum;
        }

        double get_exact_call_option_price(const KC::UInputs &conf, const double upper_bound = 100.) {
            // Check that an european call is considered
            bool condition = (conf.options.opt_right == KI::OptRight::Call) and (
                                 conf.options.opt_type == KI::OptType::European);
            if (!condition) {
                throw std::runtime_error(
                    "Must consider a European Call option for the Heston weak convergence test !");
            }

            const auto p = conf.model.heston;
            const double K = conf.options.StrikePrice;
            const double T = conf.time.t_end;
            const double S0 = conf.init.S0;

            // Compute the probabilities (The infinite integral is truncated at phi_max = upper_bound).
            double P1 = Probability(conf, 1, upper_bound);
            double P2 = Probability(conf, 2, upper_bound);

            return S0 * std::exp(-p.q * T) * P1 - K * std::exp(-p.r * T) * P2;
        }
    }
}
