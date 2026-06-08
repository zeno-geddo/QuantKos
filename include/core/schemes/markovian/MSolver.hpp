#pragma once
#include <Kokkos_Core.hpp>

#include "../../config/ConfigEnums.hpp"
#include "MSchemes.hpp"
#include "../../Typedefs.hpp"
#include "../../engine/MCMem.hpp"
#include "../RandNGenerator.hpp"
#include "../../options/Payoff.hpp"


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
        SDEScheme<ModelPolicy, SchemePolicy> Scheme; // The trivially copyable mathematical solver
        DevPathsView local_paths_batch_view;
        DevPayoffView local_payoff_batch_view;
        RNGManager::GlobalRNGPool rng_pool;
        int n_t_steps;
        KT::Real S0;
        KT::Real v0;
        KT::Real Strike;
        KT::Real BarrierPrice;


        // 2. The Execution Operator (Better than KOKKOS_LAMBDA)
        KOKKOS_INLINE_FUNCTION
        void operator()(const int n_p) const {
            // Get the random generator for given threat
            ScopedRNG scoped_rng(rng_pool);
            auto &rn_generator = scoped_rng.return_unique_rng_state();

            // Initialize Loop Variables
            KT::Real S = S0;
            KT::Real v = v0;
            PayoffTracker<OptType, OptRight> PTracker(S);

            // Evolve in time the current path
            for (int n_t = 0; n_t < n_t_steps; ++n_t) {
                // Get new state
                auto [next_S, next_v] = Scheme.evolve_step(S, v, rn_generator);
                S = next_S;
                v = next_v;

                // Store Price if necessary
                local_paths_batch_view(n_p, n_t) = S;

                // Track Price
                PTracker.track_current_price(S);
            }

            // Evaluate Payoff using the tracker state
            local_payoff_batch_view(n_p) = PTracker.evaluate_final_payoff(S,
                                                                          Strike,
                                                                          BarrierPrice,
                                                                          n_t_steps);
        }
    };


    // ========================================================================
    // THE EXECUTOR BRIDGE (Class-Level Parallel Launch Coordinator)
    // ========================================================================
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
    class MSolver {
    public:
        explicit MSolver(const KC::UInputs &config)
            : config(config), Scheme(config) {
            // Note: Scheme obj struct created here during initialization, all threat will then use it
        }

        void execute_batch(const int n_active_sims_in_batch, MCBatchMem &BatchMem, const RNGManager &RNGen) const {
            // 1. Package data from the subsystems into the execution functor
            // Note: the pull is the same for all batches, it does not have to be reinitialized !
            IntegrationKernel<ModelPolicy, SchemePolicy, OptType, OptRight> kernel{
                Scheme,
                BatchMem.d_batch_view,
                BatchMem.d_payoffs,
                RNGen.get_global_rng_pool(),
                config.time.N_time_steps,
                config.init.S0,
                config.init.v0,
                config.options.StrikePrice,
                config.options.BarrierPrice
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
