#pragma once

#include <Kokkos_Core.hpp>
#include "./../IO/Config.hpp"
#include "./../IO/ConfigEnums.hpp"
#include "./RandNGenerator.hpp"

namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KE = KOps::Engine;

    // Struct containing the return values
    struct SDEState {
        KT::Real S;
        KT::Real v;
    };


    // Master template blueprint (fall back if no specialization is available), it is trivially copiable struct!
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
    struct SDEScheme {
        explicit SDEScheme(const KC::UInputs &config) {
            // This protects you at runtime if you accidentally try to run an unimplemented scheme
            throw std::runtime_error("SDESolver math kernel not yet implemented for this scheme combination!");
        }

        // Dummy fallback function to allow un-specialized paths to compile successfully
        template<typename RNGeneratorType>



        KOKKOS_INLINE_FUNCTION
        SDEState evolve_step(const KT::Real S_n, const KT::Real v_n, RNGeneratorType &rn_generator) const {
            return {S_n, v_n};
        }
    };

    // ========================================================================
    // SPECIALIZATION: Heston + Euler Maruyama
    // BOOK : The Heston model and its extensions in matlab and C#
    // Eq 7.8 -> v_n+1 = v_n + k(theta - v_n)dt + sigma * sqrt(v_n) * sqrt(dt) * Z_v
    // Eq 7.12 -> S_n+1 = S_n * exp( (r - q - v_n+1 * 1/2) * dt + sqrt(v_n+1) sqrt(dt) * Z_S)
    // ========================================================================
    template<>
    struct SDEScheme<KI::MathModel::Heston, KI::NumScheme::Euler> {
        // Precomputed constant scalar invariants (Computed once on CPU Host)
        KT::Real one_minus_k_dt;
        KT::Real k_theta_dt;
        KT::Real sigma_sqrt_dt;
        KT::Real r_minus_q_dt;
        KT::Real half_dt;
        KT::Real sqrt_dt;
        KT::Real rho;
        KT::Real rho_complement;

        // Constructor
        explicit SDEScheme(const KC::UInputs &config) {
            const KT::Real r = config.model.heston.r;
            const KT::Real q = config.model.heston.q;
            const KT::Real k = config.model.heston.k;
            const KT::Real theta = config.model.heston.theta;
            const KT::Real sigma = config.model.heston.sigma;
            rho = config.model.heston.rho;
            const KT::Real dt = config.time.dt;

            // Compute roots once on the host so the GPU doesn't have to
            sqrt_dt = Kokkos::sqrt(dt);
            rho_complement = Kokkos::sqrt(KT::real_one - rho * rho);

            // PreCompute variance factors
            one_minus_k_dt = KT::real_one - k * dt;
            k_theta_dt = k * theta * dt;
            sigma_sqrt_dt = sigma * sqrt_dt;

            // PreCompute price factors
            r_minus_q_dt = (r - q) * dt;
            half_dt = KT::real_05 * dt; // Avoid division on GPU since it is a slow operation
        }

        template<typename RNGeneratorType>
        KOKKOS_INLINE_FUNCTION
        SDEState evolve_step(const KT::Real S_n, const KT::Real v_n, RNGeneratorType &local_rn_generator) const {
            // Get Random Normal Variables
            KT::Real Z_1 = static_cast<KT::Real>(local_rn_generator.normal());
            KT::Real Z_2 = static_cast<KT::Real>(local_rn_generator.normal());

            // Cholesky Decomposition for correlated Brownian Motion
            const KT::Real Z_v = Z_1;
            const KT::Real Z_S = rho * Z_1 + rho_complement * Z_2;

            // EVOLVE & TRUNCATE VARIANCE (Fused Multiply-Add (FMA) optimized variance step)
            const KT::Real sqrt_v_n = Kokkos::sqrt(v_n);
            KT::Real v_np1 = v_n * one_minus_k_dt + k_theta_dt + (sigma_sqrt_dt * sqrt_v_n) * Z_v;
            v_np1 = v_np1 > KT::real_zero ? v_np1 : KT::real_zero;

            // EVOLVE ASSET WITH NEW VARIANCE
            // Mathematically: (r - q)dt - (v_np1 * 0.5 * dt) + (sqrt(v_np1) * sqrt(dt)) * Z_S
            const KT::Real sqrt_v_np1 = Kokkos::sqrt(v_np1);
            const KT::Real exponent = r_minus_q_dt - (v_np1 * half_dt) + (sqrt_v_np1 * sqrt_dt) * Z_S;
            const KT::Real S_np1 = S_n * Kokkos::exp(exponent); // Special Function Unit (SFU) Call

            return {S_np1, v_np1};

        }
    };
}
