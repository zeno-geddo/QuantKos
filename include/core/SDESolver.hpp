#pragma once
#include <Kokkos_Core.hpp>

#include "./../IO/ConfigEnums.hpp"
#include "SDESchemes.hpp"
#include "Typedefs.hpp"
#include "./MCMem.hpp"
#include "./RandNGenerator.hpp"
#include "./Payoff.hpp"


//-------------------------------------------------------------------------------
//      [ HOST (CPU) SIDE ]                             [ DEVICE (GPU) SIDE ]
//
// +-----------------------------+
// |         SDESolver           |
// |  (Orchestrates launch data)  |
// +--------------+--------------+
//                |
//                | Instantiates & Populates
//                v
// +-----------------------------+               +-----------------------------+
// |     IntegrationKernel       | ------------> | Thread Lane 0: operator()(0) |
// |  (A trivially copyable      |   Parallel    | Thread Lane 1: operator()(1) |
// |   snapshot of state data)   |   Launch      | Thread Lane 2: operator()(2) |
// +-----------------------------+               +-----------------------------+
//-------------------------------------------------------------------------------


namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;

    // ========================================================================
    // THE HARDWARE FUNCTOR (Thread-Level Execution)
    // It represents the execution pathway of one single thread lane (n_p).
    // ========================================================================
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
    struct IntegrationKernel {
        // 1. Trivially Copiable Attributes (The Execution Context, The data the GPU needs)
        DevPathsView local_paths_batch_view;
        DevPayoffView local_payoff_batch_view;
        RNGManager::GlobalRNGPool rng_pool;
        int n_t_steps;
        KT::Real S0;
        KT::Real v0;
        KT::Real Strike;
        SDEScheme<ModelPolicy, SchemePolicy> Scheme; // The trivially copyable mathematical solver

        // 2. The Execution Operator (Better than KOKKOS_LAMBDA)
        KOKKOS_INLINE_FUNCTION
        void operator()(const int n_p) const {
            // Get the random generator for given threat
            ScopedRNG scoped_rng(rng_pool);
            auto &rn_generator = scoped_rng.return_unique_rng_state();

            // Initialize Loop Variables
            KT::Real S = S0;
            KT::Real v = v0;
            KT::Real S_target = KT::real_zero; // Placeholder for asian and barrier options

            // Evolve in tima the current path
            for (int n_t = 0; n_t < n_t_steps; ++n_t) {
                auto [next_S, next_v] = Scheme.evolve_step(S, v, rn_generator);
                S = next_S;
                v = next_v;
                local_paths_batch_view(n_p, n_t) = S;

                if constexpr (OptType == KI::OptType::Asian) {
                    S_target += S;
                }

            }

            // Resolve reference price for payoff (Compile time branch)
            KT::Real reference_price;
            if constexpr (OptType == KI::OptType::Asian) {
                reference_price = S_target / static_cast<KT::Real>(n_t_steps);
            } else if constexpr (OptType == KI::OptType::European) {
                reference_price = S;
            }

            // Evaluate Payoff
            local_payoff_batch_view(n_p) = Payoff<OptRight>::evaluate(reference_price, Strike);

        }
    };


    // ========================================================================
    // THE EXECUTOR BRIDGE (Class-Level Parallel Launch Coordinator)
    // ========================================================================
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
    class SDESolver {
    public:
        explicit SDESolver(const KC::UInputs &config)
            : config(config), Scheme(config) {
        }

        void execute_batch(const int n_active_sims_in_batch, MCBatchMem &BatchMem, const RNGManager &RNGen) const {
            // 1. Package data from the subsystems into the execution functor
            // Note: the pull is the same for all batches, it does not have to be reinitialized !
            IntegrationKernel<ModelPolicy, SchemePolicy, OptType, OptRight> kernel{
                BatchMem.d_batch_view,
                BatchMem.d_payoffs,
                RNGen.get_global_rng_pool(),
                config.time.N_time_steps,
                config.init.S0,
                config.init.v0,
                config.options.K,
                Scheme
            };

            // 2. Safely isolate the parallel launch boundary inside the class
            Kokkos::parallel_for("Evolve_SDEs_in_given_batch", n_active_sims_in_batch, kernel);
            Kokkos::fence();
        }

    private:
        const KC::UInputs config;
        SDEScheme<ModelPolicy, SchemePolicy> Scheme; // Instance of the scheme chosen
    };
}
