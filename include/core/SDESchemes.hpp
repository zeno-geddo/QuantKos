#pragma once
#include <Kokkos_Core.hpp>
#include "./../IO/Config.hpp"
#include "./../IO/ConfigEnums.hpp"

namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;

    // Struct containing the return values
    struct SDEState {
        KT::Real S;
        KT::Real v;
    };


    // Master template blueprint (fall back if no specialization is available), it is trivially copiable struct!
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
    struct SDESchemes {

        explicit SDESchemes(const KC::UInputs& config) {
            // This protects you at runtime if you accidentally try to run an unimplemented scheme
            throw std::runtime_error("SDESolver math kernel not yet implemented for this scheme combination!");
        }


    };

    // ========================================================================
    // SPECIALIZATION: Heston + Euler Maruyama
    // ========================================================================
    template<>
    struct SDESchemes<KI::MathModel::Heston, KI::NumScheme::Euler> {
        KT::Real r, k, theta, sigma, rho, dt;

        // Constructor
        explicit SDESchemes(const KC::UInputs& config) {
            r     = config.model.heston.r;
            k     = config.model.heston.k;
            theta = config.model.heston.theta;
            sigma = config.model.heston.sigma;
            rho   = config.model.heston.rho;
            dt    = config.time.dt;
        }

        template<typename RNGeneratorType>
        KOKKOS_INLINE_FUNCTION
        SDEState evolve_step(const KT::Real S_n, const KT::Real v_n, RNGeneratorType& rn_generator) const {

            // Explicitly cast literals to your custom type
            constexpr KT::Real zero = static_cast<KT::Real>(0.0); // ok, but do not use static inside gpus!
            constexpr KT::Real one  = static_cast<KT::Real>(1.0); // ok, but do not use static inside gpus!

            KT::Real Z1 = rn_generator.template normal<KT::Real>();
            KT::Real Z2 = rn_generator.template normal<KT::Real>();

            KT::Real W1 = Z1;
            KT::Real W2 = rho * Z1 + Kokkos::sqrt(one - rho * rho) * Z2;

            KT::Real S_np1 = S_n + r * S_n * dt + Kokkos::sqrt(v_n) * S_n * Kokkos::sqrt(dt) * W1;
            KT::Real v_np1 = v_n + k * (theta - v_n) * dt + sigma * Kokkos::sqrt(v_n) * Kokkos::sqrt(dt) * W2;
            v_np1 = v_np1 > zero ? v_np1 : zero;

            return {S_np1, v_np1};
        }
    };

}
