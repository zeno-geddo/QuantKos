// Copyright (C) 14/07/2026 Zeno GEDDO <zeno.geddo@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once
#include <Kokkos_Core.hpp>

#include "../../config/ConfigFileEnums.hpp"
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
    /**
         * @brief Parallel execution code (functor) run on device (CPU/GPU) where each thread simulates a single asset path using a markovian model.
         * * This struct packages all simulation parameters into a self-contained snapshot. Because it has
         * no pointers to CPU memory (it is *trivially copyable*), it can be copied cleanly to discrete GPU devices
         * without causing memory access violations.
         * * @tparam ModelPolicy Stochastic asset model to consider (e.g., Heston, Bates, etc.).
         * @tparam SchemePolicy Time-stepping scheme to use (e.g., Euler, Milstein, AndersonQE, etc.).
         * @tparam OptType Option contract payoff logic (e.g., European, Asian, Barrier, etc.).
         * @tparam OptRight Option contract right (Call or Put).
         * @tparam StorePaths Set to @c true to save the full simulated asset price path to memory, or @c false to save memory.
         * @note- **Smart Pruning (@c StorePaths)**: If full price history is not needed (e.g., for European options), the compiler
         * completely removes the code that writes the full price path to VRAM. This can save memory bandwidth
         * and speeds up execution.
    */
    template<KI::MathModel ModelPolicy,
        KI::NumScheme SchemePolicy,
        KI::OptType OptType,
        KI::OptRight OptRight,
        bool StorePaths>
    struct MarkovianIntegrationKernel {
        // 1. Trivially Copiable Attributes (The Execution Context, The data the GPU needs)
        SDEScheme<ModelPolicy, SchemePolicy> Scheme; ///< Local copy of the SDE integration scheme.
        DevBatchPathsView local_paths_batch_view;
        ///< Target device VRAM/GPu 2D array mapping full path trajectories of a batch (Accessed by threads).
        DevBatchPayoffView local_payoff_batch_view;
        ///< Target device VRAM 1D array mapping path-level terminal payloads of a batch (Accessed by threads).
        RNGManager::GlobalRNGPool rng_pool;
        ///< Dedicated global hardware random number state pool (A copy for each kernel launch is generated).
        int n_t_steps; ///< Total discrete simulation time segments (Local variable stored in GPU's registers).
        KT::Real S0; ///< Initial spot price, (Local variable stored in GPU's registers).
        KT::Real v0; ///< Initial variance of the asset, (Local variable stored in GPU's registers).
        KT::Real Strike; ///<Strike Price ($K$), (Local variable stored in GPU's registers).
        KT::Real BarrierPrice;
        ///< Option activation or knock-out trigger constraint ($H$), (Local variable stored in GPU's registers).


        // 2. The Execution Operator (Better than KOKKOS_LAMBDA)
        /**
         * @brief Parallel execution kernel processing a single isolated Monte Carlo path (realization) using a markovian model.
         * Each GPU thread grabs a random generator, simulates one stock price path step-by-step,
         * and tracks its value along the way (like, for instance, checking if it hits a barrier).
         * Finally, it calculates the option's payoff for that specific path and writes the result to the device memory.
         * * @param n_p Number of target simulation in the batch loop (a batch consists of a numbered subset of the full MonteCarlo simulation).
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const int n_p) const {
            // Get the random generator for given threat
            ScopedRNG scoped_rng(rng_pool);
            auto &rn_generator = scoped_rng.return_unique_rng_state(); // create a temporary state in thread's register

            // Initialize Loop Variables
            KT::Real S = S0;
            KT::Real v = v0;
            PayoffTracker<OptType, OptRight> PTracker(S); // Trivially copiable object not involving pointers

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
    /**
    * @brief Host-side parallel launch coordinator and execution bridge.
    * * Manages execution pipelines by grouping parameters, evaluating paths storage rules,
    * and scheduling high-throughput integration kernels to the hardware backend.
    * @note This class is instantiated once at the beginning of the MonteCarlo simulation, and the execute batch is called once for each batch.
    * @note The integration kernel is created at the beginning of each batch, and it will be used by all threads during the simulations of the batch.
    * * ### Host-Device Orchestration Lifecycle:
    * ```text
    *      [ HOST (CPU) SIDE ]                             [ DEVICE (GPU) SIDE ]
    * * +----------------------------+
    * |         SDESolver            |
    * |  (Orchestrates launch data)  |
    * +--------------+---------------+
    *               |
    *               | Instantiates & Populates
    *               v
    * +-----------------------------+               +-----------------------------+
    * |     IntegrationKernel       | ------------> | Thread Lane 0: operator()(0) |
    * |  (A trivially copyable      |   Parallel    | Thread Lane 1: operator()(1) |
    * |   snapshot of state data)   |   Launch      | Thread Lane 2: operator()(2) |
    * +-----------------------------+               +-----------------------------+
    *
    * ```
    * * @tparam ModelPolicy Stochastic asset model to consider (e.g., Heston, Bates, etc.).
    * @tparam SchemePolicy Time-stepping scheme to use (e.g., Euler, Milstein, AndersonQE, etc.).
    * @tparam OptType Option contract payoff logic (e.g., European, Asian, Barrier, etc.).
    * @tparam OptRight Option contract right (Call or Put).
    * @todo Should implement non-markovian models. This could be modified to work also with a non marlovian integration kernel.
    */
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
    class MSolver {
    public:
        /**
         * @brief Constructs the MSolver orchestrator.
         * @param config Reference to the user configuration inputs.
         */
        explicit MSolver(const KC::UInputs &config)
            : config(config), Scheme(config) {
            // Note: Scheme obj struct created here during initialization, all threats will then use it
        }

        /// @name Lifecycle Protocols
        ///@{
        MSolver(const MSolver &) = delete; ///< Deleted copy constructor.
        MSolver &operator=(const MSolver &) = delete; ///< Deleted copy assignment operator.
        MSolver(MSolver &&) = default; ///< Explicitly allow move constructor
        MSolver &operator=(MSolver &&) = default; ///< Explicitly allow move assignment
        ~MSolver() = default; ///< Default destructor.
        ///@}

        /**
        * @brief Launches the Monte Carlo path simulation for the current batch calling the integration kernel.
        * * Packages state parameters, determines whether storing paths on RAM is required,
        * and schedules the IntegrationKernel on the active execution space.
        * @param n_active_sims_in_batch Number of paths to simulate in the batch currently considered.
        * @param BatchMem Buffer memory holding target device batch views.
        * @param RNGen Random number sequence pool manager.
        */
        void execute_batch(const int n_active_sims_in_batch, PathsMCBatchMem &BatchMem, const RNGManager &RNGen) const {
            // Lambda for setting up and launching the markovian kernel
            auto launch_markovian_kernel = [&](auto store_paths_tag) {
                // 0. Decide weather to store the paths or not
                // Note: decltype() -> operator that asks the compiler: What is the type of this expression?
                static_assert(std::is_same_v<decltype(store_paths_tag), std::bool_constant<true> > ||
                              std::is_same_v<decltype(store_paths_tag), std::bool_constant<false> >,
                              "Kernel must be launched with a std::bool_constant to chose if to store paths or not!");
                constexpr bool StorePaths = decltype(store_paths_tag)::value;

                // 1. Package data from the subsystems into the execution functor
                // Note: the random-pull is the same for all batches, it does not have to be reinitialized !
                MarkovianIntegrationKernel<ModelPolicy, SchemePolicy, OptType, OptRight, StorePaths> kernel{
                    Scheme,
                    BatchMem.d_batch_view,
                    BatchMem.d_payoffs,
                    RNGen.get_global_rng_pool(),
                    config.time.N_time_steps,
                    config.market.S0,
                    config.market.v0,
                    config.options.StrikePrice,
                    config.options.BarrierPrice
                };

                // 2. Safely isolate the parallel launch boundary inside the class
                Kokkos::parallel_for("Evolve_SDEs_in_given_batch", n_active_sims_in_batch, kernel);
            }; // End of lambda

            // Dispatch Kernel based on runtime condition
            if (must_store_paths()) {
                launch_markovian_kernel(std::bool_constant<true>{});
            } else {
                launch_markovian_kernel(std::bool_constant<false>{});
            }

            Kokkos::fence();
        }

    private:
        // Reference is completely safe on the Host side!
        const KC::UInputs &config; ///< Host-side reference to user input configurations.
        SDEScheme<ModelPolicy, SchemePolicy> Scheme;
        ///! Trivially copiable Instance of the chosen numerical integration scheme

        /**
         * @brief Checks if simulated price trajectories must be stored in RAM.
         * @return True if path file saving is enabled or if backward induction (American options) is used. False otherwise.
         */
        inline bool must_store_paths() const {
            const auto OType = config.options.opt_type;
            const bool requires_full_paths_matrix = (OType == KI::OptType::American);

            // Store if the paths to save the outputs if given or if the option evaluation requires all paths matrix for backward algorithm
            return !config.output.filename_paths_out.empty() || requires_full_paths_matrix;
        }
    };
}
