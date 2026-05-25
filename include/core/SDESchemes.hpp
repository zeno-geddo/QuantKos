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
    // ========================================================================
    template<>
    struct SDEScheme<KI::MathModel::Heston, KI::NumScheme::Euler> {
        // Precomputed scalar invariants (Computed once on CPU Host)
        KT::Real one_plus_r_dt;
        KT::Real one_minus_k_dt;
        KT::Real k_theta_dt;
        KT::Real sigma_sqrt_dt;
        KT::Real sqrt_dt;
        KT::Real rho;
        KT::Real rho_complement;

        // Constructor
        explicit SDEScheme(const KC::UInputs &config) {
            const KT::Real r = config.model.heston.r;
            const KT::Real k = config.model.heston.k;
            const KT::Real theta = config.model.heston.theta;
            const KT::Real sigma = config.model.heston.sigma;
            rho = config.model.heston.rho;
            const KT::Real dt = config.time.dt;

            // Compute roots once on the host so the GPU doesn't have to
            sqrt_dt         = Kokkos::sqrt(dt);
            rho_complement  = Kokkos::sqrt(KT::real_one - rho * rho);
            one_plus_r_dt   = KT::real_one + r * dt;
            one_minus_k_dt  = KT::real_one - k * dt;
            k_theta_dt      = k * theta * dt;
            sigma_sqrt_dt   = sigma * sqrt_dt;
        }

        template<typename RNGeneratorType>

        KOKKOS_INLINE_FUNCTION
        SDEState evolve_step(const KT::Real S_n, const KT::Real v_n, RNGeneratorType &local_rn_generator) const {

            // Get Random Normal Variables
            KT::Real Z1 = static_cast<KT::Real>(local_rn_generator.normal());
            KT::Real Z2 = static_cast<KT::Real>(local_rn_generator.normal());

            // Cholesky Decomposition for correlated Brownian Motion
            const KT::Real W1 = Z1;
            const KT::Real W2 = rho * Z1 + rho_complement * Z2;

            // Evolve Asset and Variance State
            const KT::Real sqrt_v = Kokkos::sqrt(v_n);

            // Hyper-Compressed Evolution Math (Maximizes FMA Register Speed)
            const KT::Real S_np1 = S_n * (one_plus_r_dt + (sqrt_v * sqrt_dt) * W1);
            KT::Real v_np1 = v_n * one_minus_k_dt + k_theta_dt + (sigma_sqrt_dt * sqrt_v) * W2;

            // Full Truncation boundary constraint to keep variance positive
            v_np1 = v_np1 > KT::real_zero ? v_np1 : KT::real_zero;

            return {S_np1, v_np1};
        }
    };
}
