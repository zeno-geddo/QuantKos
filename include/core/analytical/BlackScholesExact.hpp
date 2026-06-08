#pragma once
#include <cmath>
#include <stdexcept>

#include "./../Typedefs.hpp"
#include "./../config/Config.hpp"

namespace KOps::Engine::Analytical::BlackScholes {

    namespace KT = KOps::Types;
    namespace KC = KOps::Config;
    namespace KI = KOps::Implemented;

    /**
    * @brief Eval CDF of a standard normal at point x.
    * @param x, position in the x domain.
    * @return Value of the Normal CDF at x.
    */
    inline KT::Real normalCDF(KT::Real x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }

    inline KT::Real get_exact_eu_call_option_price(const KC::UInputs &conf) {
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
