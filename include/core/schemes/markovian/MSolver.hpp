#pragma once
#include <Kokkos_Core.hpp>

#include "../../config/ConfigEnums.hpp"
#include "MSchemes.hpp"
#include "../../Typedefs.hpp"
#include "../../memory/PathsMCBatchMem.hpp"
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
    template<KI::MathModel ModelPolicy,
        KI::NumScheme SchemePolicy,
        KI::OptType OptType,
        KI::OptRight OptRight,
        bool StorePaths>
    struct IntegrationKernel {
        // 1. Trivially Copiable Attributes (The Execution Context, The data the GPU needs)
        SDEScheme<ModelPolicy, SchemePolicy> Scheme; // A copy for each kernel launch
        DevPathsView local_paths_batch_view; // Accessed by the threads, stored in GPU's VRAM
        DevPayoffView local_payoff_batch_view; // Accessed by the threads, stored in GPU's VRAM
        RNGManager::GlobalRNGPool rng_pool; // A copy for each kernel launch
        int n_t_steps; // Local variables stored in GPU's registers
        KT::Real S0; // Local variables stored in GPU's registers
        KT::Real v0; // Local variables stored in GPU's registers
        KT::Real Strike; // Local variables stored in GPU's registers
        KT::Real BarrierPrice; // Local variables stored in GPU's registers


        // 2. The Execution Operator (Better than KOKKOS_LAMBDA)
        KOKKOS_INLINE_FUNCTION
        void operator()(const int n_p) const {
            // Get the random generator for given threat
            ScopedRNG scoped_rng(rng_pool);
            auto &rn_generator = scoped_rng.return_unique_rng_state(); // create a temporary state in thread's register

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
                if constexpr (StorePaths == true) {
                    local_paths_batch_view(n_p, n_t) = S; // Global write to GPU's VRAM
                }

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

        // Delete copy constructor and copy assignment operator
        MSolver(const MSolver&) = delete;
        MSolver& operator=(const MSolver&) = delete;

        // Explicitly allow move semantics if desired (optional)
        MSolver(MSolver&&) = default;
        MSolver& operator=(MSolver&&) = default;

        void execute_batch(const int n_active_sims_in_batch, PathsMCBatchMem &BatchMem, const RNGManager &RNGen) const {
            auto launch_kernel = [&](auto store_paths_tag) {
                // 0. Decide weather to store the paths or not
                // Note: decltype() -> operator that asks the compiler: What is the type of this expression?
                static_assert(std::is_same_v<decltype(store_paths_tag), std::bool_constant<true> > ||
                              std::is_same_v<decltype(store_paths_tag), std::bool_constant<false> >,
                              "Kernel must be launched with a std::bool_constant to chose if to store paths or not!");
                constexpr bool StorePaths = decltype(store_paths_tag)::value;

                // 1. Package data from the subsystems into the execution functor
                // Note: the random-pull is the same for all batches, it does not have to be reinitialized !
                IntegrationKernel<ModelPolicy, SchemePolicy, OptType, OptRight, StorePaths> kernel{
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
            };


            // Dispatch Kernel based on runtime condition
            if (must_store_paths()) {
                launch_kernel(std::bool_constant<true>{});
            } else {
                launch_kernel(std::bool_constant<false>{});
            }

            Kokkos::fence();
        }

    private:
        const KC::UInputs config;
        SDEScheme<ModelPolicy, SchemePolicy> Scheme; // Trivially copiable Instance of the scheme chosen

        inline bool must_store_paths() const {
            const auto OType = config.options.opt_type;
            const bool requires_full_paths_matrix = (OType == KI::OptType::American);

            // Store if the paths to save the outputs if given or if the option evaluation requires all paths matrix for backward algorithm
            return !config.output.filename_paths_out.empty() || requires_full_paths_matrix;
        }
    };
}
