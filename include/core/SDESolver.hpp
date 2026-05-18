#pragma once
#include <Kokkos_Core.hpp>

#include "./../IO/ConfigEnums.hpp"
#include "SDESchemes.hpp"
#include "Typedefs.hpp"
#include "./MCMem.hpp"
#include "./RandNGenerator.hpp"


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
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
    struct IntegrationKernel {
        // 1. Trivially Copiable Attributes (The Execution Context, The data the GPU needs)
        DevView local_batch_view;
        typename RNGenerator::PoolType local_pool;
        int n_t_steps;
        KT::Real S0;
        KT::Real v0;
        SDEScheme<ModelPolicy, SchemePolicy> Scheme; // The trivially copyable mathematical solver

        // 2. The Execution Operator (Better than KOKKOS_LAMBDA)
        KOKKOS_INLINE_FUNCTION
        void operator()(const int n_p) const {
            auto rn_generator = local_pool.get_state();

            KT::Real S = S0;
            KT::Real v = v0;

            for (int n_t = 0; n_t < n_t_steps; ++n_t) {
                auto [next_S, next_v] = Scheme.evolve_step(S, v, rn_generator);
                S = next_S;
                v = next_v;
                local_batch_view(n_p, n_t) = S;
            }
            local_pool.free_state(rn_generator);
        }
    };


    // ========================================================================
    // THE EXECUTOR BRIDGE (Class-Level Parallel Launch Coordinator)
    // ========================================================================
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
    class SDESolver {
    public:
        explicit SDESolver(const KC::UInputs &config)
            : config(config), Scheme(config) {
        }

        void execute_batch(const int n_active_sims_in_batch, MCBatchMem &BatchMem, const RNGenerator &RNGen) const {
            // 1. Package data from the subsystems into the execution functor
            IntegrationKernel<ModelPolicy, SchemePolicy> kernel{
                BatchMem.d_batch_view,
                RNGen.get_pool(),
                config.time.N_time_steps,
                config.init.S0,
                config.init.v0,
                Scheme
            };

            // 2. Safely isolate the parallel launch boundary inside this class
            Kokkos::parallel_for("EvolveSDEs", n_active_sims_in_batch, kernel);
            Kokkos::fence();
        }

    private:
        const KC::UInputs config;
        SDEScheme<ModelPolicy, SchemePolicy> Scheme; // Instance of the scheme chosen
    };
}
