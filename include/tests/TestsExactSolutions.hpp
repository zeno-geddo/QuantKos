#pragma once

#include <cmath>
#include <stdexcept>

#include "./../core/Typedefs.hpp"
#include "../IO/Config.hpp"


namespace KOps::Tests::ExactSolutions {

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

    inline KT::Real get_exact_option_price_BlackScholes(const KC::UInputs &conf) {
        KT::Real S = conf.init.S0;
        KT::Real K = conf.options.K;
        KT::Real T = conf.time.t_end;
        KT::Real r = conf.model.heston.r;
        KT::Real q = conf.model.heston.q;
        KT::Real vol = std::sqrt(conf.init.v0); // Extract Volatility from Variance

        KT::Real d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        KT::Real d2 = d1 - vol * std::sqrt(T);

        // Assume considering European Call
        bool condition = (conf.options.opt_right == KI::OptRight::Call) and (
                             conf.options.opt_type == KI::OptType::European);
        if (condition) {
            return S * std::exp(-q * T) * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
        } else {
            throw std::runtime_error("Must consider a European Call option for the Black Scholes weak convergence !");
        }
    }
}