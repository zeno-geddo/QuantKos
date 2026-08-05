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

#include <iostream>
#include <chrono> // for timing

#include "../schemes/markovian/MSolver.hpp"
#include "../config/Config.hpp"
#include "../../IO/OutManager.hpp"
#include "../Typedefs.hpp"
#include "./MCEngine.hpp"
#include "./../memory/PathsMCBatchMem.hpp"
#include "../schemes/RandNGenerator.hpp"
#include "../options/OptionPricer.hpp"


namespace quantkos::Engine {
    namespace KI = quantkos::Implemented;
    namespace KT = quantkos::Types;
    namespace KIO = quantkos::IO;

    /**
     * @brief Orchestrator for Forward Monte Carlo simulations.
     * @note This class serves as the top-level execution container for pricing options that rely
     * exclusively on a forward simulation pass (e.g., European, Asian, Barrier, Lookback, and Binary options).
     * It eliminates runtime polymorphism overhead by embedding structural pricing parameters directly into the type signature.
     * * @tparam ModelPolicy Compile-time stochastic process selection (e.g., Heston, Bates).
     * @tparam SchemePolicy Compile-time SDE integration algorithm (e.g., Euler, Milstein, AndersonQE).
     * @tparam OptType Compile-time option payoff structure classification.
     * @tparam OptRight Compile-time option execution right (Call or Put).
     * * @note To handle cases where the required path matrix exceeds physical memory limitations (such as GPU VRAM),
     * the simulation is chunked into manageable sub-allocations using a batching loop strategy.
    */
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
    class ForwardMCRunner {
    public:

        /**
         * @brief Constructs a forward runner instance given the user parameters.
         * @param conf The configuration inputs reference containing runtime parameters.
         */
        explicit ForwardMCRunner(const KC::UInputs &conf) : config(conf) {
        }

        /// @name Lifecycle Protocols
        ///@{
        // Delete copy operations (prevents accidental duplication)
        ForwardMCRunner(const ForwardMCRunner&) = delete; ///< Deleted copy constructor.
        ForwardMCRunner& operator=(const ForwardMCRunner&) = delete; ///< Deleted copy assignment operator.

        // Default the move constructor and delete move assignment
        ForwardMCRunner(ForwardMCRunner&&) = default; ///< Default move constructor for ownership transfer.
        ForwardMCRunner& operator=(ForwardMCRunner&&) = delete; ///< Move assignment is prohibited.

        // Default destructor
        ~ForwardMCRunner() = default; ///< Default destructor.
        ///@}

        // ====================================================================
        // The Actual Simulation Engine (Fully Resolved at Compile Time)
        // ====================================================================
        /**
         * @brief High-level orchestration method that runs the forward Monte Carlo engine.
         * * This method manges the local infrastructure stack needed to run the simulation,
         * including batch memory buffers, SDE solvers, option payoff accumulation tracking,
         * and output writers.
         * * @return An aggregated MCResults object containing the results of the MonteCarlo.
         * @todo If calling this in a loop, should add the possibility to reuse the helper classes to avoid repeating initialization overheads.
         */
        MCResults get_option_prices() const {
            // NOTE : the total number of simulations are performed in batches to handle cases when not enough memory is available
            // NOTE : The global random number pool is created once. States advance dynamically. So using the same pool for different batches is the correct approach

            // 1. Initialize Helper Classes needed during the MC (keep in this local function scope)
            PathsMCBatchMem BatchMem(config); // Handles the Memory
            RNGManager RNGen(config); // Handles the Random number (Must initialize here and not in the batch loop!!!)
            MSolver<ModelPolicy, SchemePolicy, OptType, OptRight> Solver(config); // Handles the Temporal integration
            OptionPricer OPricer(config); // Handles the Option Pricing
            KIO::OutputManager OWriter(config); // Handles the outputs
            ForwardMCProgressTracker MCTracker(config, BatchMem);

            // Print Pre-Execution Diagnostics
            MCTracker.print_pre_execution_diagnostic();
            OWriter.print_paths_info_planned_outputs();

            // 3. Run all batches forwards (all mc simulations giving the prices)
            run_mc_forward(BatchMem, RNGen, Solver, OPricer, OWriter, MCTracker);

            // 4. Compute Option Price
            OPricer.evaluate_option_price();
            OPricer.print_info_option_price();
            MCResults Res{config, OPricer.get_option_price_data()};
            return Res;
        }

    private:
        const KC::UInputs &config; ///< Read-only alias pointing back to the application configuration environment.

        /**
         * @brief Configures internal lambda handlers (to synch and/or write results generated within a batch) and triggers the batches run mechanism.
         * * @param BatchMem Reusable memory batch buffer manager.
         * @param RNGen Random number sequence pool manager.
         * @param Solver Compile-time specialized temporal SDE stepper solver.
         * @param OPricer Target option contract accumulation pricer.
         * @param OWriter Target disk file system I/O manager.
         * @param MCTracker Console analytics reporting performance.
         * @note Wraps host-to-device view synchronization policies and cumulative payload processing tasks
         * before delegating execution tasks down the execution hierarchy.
        * * ### Host-Device Execution Lifecycle Flow:
         * ```text
                   [ HOST (CPU) ]                                         [ DEVICE (GPU) ]

              MCRunner Loop Fires
                       │
                       ▼
              SDESolver::execute_batch()
                       │
                       ▼
              Instantiate IntegrationKernel
              (Flattens data onto CPU Stack)
                       │
                       ▼
              Kokkos::parallel_for()  =======[ PCIe Bus Pass ]=======>  GPU Spawns N Threads
                                                                                 │
                                                                                 ▼
                                                                      kernel.operator()(n_p)
                                                                                 │
                                                                                 ▼
                                                                      Time Loop (0 to N_Steps)
                                                                                 │
                                                                                 ▼
                                                                      scheme.evolve_step()
                                                                                 │
                                                                                 ▼
                                                                      Coalesced VRAM Write
         * ```
        */
        void run_mc_forward(PathsMCBatchMem &BatchMem,
                            const RNGManager &RNGen,
                            const MSolver<ModelPolicy, SchemePolicy, OptType, OptRight> &Solver,
                            OptionPricer &OPricer,
                            KIO::OutputManager &OWriter,
                            ForwardMCProgressTracker &MCTracker
        ) const {

            //       [ HOST (CPU) ]                                         [ DEVICE (GPU) ]
            //
            //  MCRunner Loop Fires
            //           │
            //           ▼
            //  SDESolver::execute_batch()
            //           │
            //           ▼
            //  Instantiate IntegrationKernel
            //  (Flattens data onto CPU Stack)
            //           │
            //           ▼
            //  Kokkos::parallel_for()  =======[ PCIe Bus Pass ]=======>  GPU Spawns N Threads
            //                                                                     │
            //                                                                     ▼
            //                                                          kernel.operator()(n_p)
            //                                                                     │
            //                                                                     ▼
            //                                                          Time Loop (0 to N_Steps)
            //                                                                     │
            //                                                                     ▼
            //                                                          scheme.evolve_step()
            //                                                                     │
            //                                                                     ▼
            //                                                          Coalesced VRAM Write

            // Pass a labda function that copy the prices and payoffs computed to the host
            auto copy_prices_and_payoffs = [&](const int batch_idx, const int current_batch_size) {
                BatchMem.deep_copy_to_host();
                OPricer.accumulate_batch_payoffs(BatchMem.h_payoffs, current_batch_size);
            };

            run_forward_all_mc_batches(BatchMem,
                                       RNGen,
                                       Solver,
                                       OWriter,
                                       MCTracker,
                                       copy_prices_and_payoffs);
        }

        // void run_all_mc_batches_old(PathsMCBatchMem &BatchMem,
        //                             const RNGManager &RNGen,
        //                             const MSolver<ModelPolicy, SchemePolicy, OptType, OptRight> &Solver,
        //                             OptionPricer &OPricer,
        //                             KIO::OutputManager &OWriter,
        //                             ForwardMCProgressTracker &MCTracker
        // ) const {
        //     const int full_batch_size = BatchMem.n_sims_per_batch;
        //     const int n_full_batches = BatchMem.n_full_batches();
        //     const int n_sims_left_over = BatchMem.n_sims_left_over_after_full_batches();
        //     const int n_total_batch_loops = BatchMem.total_batch_loops();
        //
        //     MCTracker.start_tracking();
        //     for (int b = 0; b < n_total_batch_loops; ++b) {
        //         const int current_batch_size = (b < n_full_batches) ? full_batch_size : n_sims_left_over;
        //         Solver.execute_batch(current_batch_size, BatchMem, RNGen); // Fire off computation kernel on device
        //         BatchMem.deep_copy_to_host(); // Synch the host with dev
        //         OPricer.accumulate_batch_payoffs(BatchMem.h_payoffs, current_batch_size);
        //         // Store payoffs to then sort them for percentiles
        //         OWriter.save_paths_batch_if_needed(current_batch_size, BatchMem); //Save batch to disk
        //
        //         MCTracker.update_progress(b + 1);
        //     }
        //     MCTracker.finalize_tracking();
        // }
    };
}
