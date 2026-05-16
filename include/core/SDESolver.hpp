#pragma once
#include <Kokkos_Core.hpp>
#include "./../IO/Config.hpp"
#include "./../IO/ConfigEnums.hpp"

namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KC = KOps::Config;

    // Master template blueprint (fall back if no specialization is available)
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
    struct SDESolver {

        explicit SDESolver(const KC::UInputs& config) {
            // This protects you at runtime if you accidentally try to run an unimplemented scheme
            throw std::runtime_error("SDESolver math kernel not yet implemented for this scheme combination!");
        }


    };

    // ========================================================================
    // SPECIALIZATION: Heston + Euler Maruyama
    // ========================================================================
    template<>
    struct SDESolver<KI::MathModel::Heston, KI::NumScheme::Euler> {
        double r, k, theta, sigma, rho, dt;

        explicit SDESolver(const KC::UInputs& config) {
            r     = config.model.heston.r;
            k     = config.model.heston.k;
            theta = config.model.heston.theta;
            sigma = config.model.heston.sigma;
            rho   = config.model.heston.rho;
            dt    = config.time.inp_dt;
        }

        KOKKOS_INLINE_FUNCTION
        void evolve_step(double& S, double& v, double Z1, double Z2) const {
            // TEMP SCHEME
            double W1 = Z1;
            double W2 = rho * Z1 + Kokkos::sqrt(1.0 - rho * rho) * Z2;

            S = S + r * S * dt + Kokkos::sqrt(v) * S * Kokkos::sqrt(dt) * W1;
            v = v + k * (theta - v) * dt + sigma * Kokkos::sqrt(v) * Kokkos::sqrt(dt) * W2;
            v = v > 0.0 ? v : 0.0;
        }
    };

}
