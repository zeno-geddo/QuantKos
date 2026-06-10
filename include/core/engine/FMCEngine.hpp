#pragma once

#include <iostream>
#include <chrono> // for timing

#include "../schemes/markovian/MSolver.hpp"
#include "../config/Config.hpp"
#include "../../IO/OutManager.hpp"
#include "../Typedefs.hpp"
#include "./MCUtils.hpp"
#include "./memory/PathsMCBatchMem.hpp"
#include "../schemes/RandNGenerator.hpp"
#include "../options/OptionPricer.hpp"


namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;
    namespace KIO = KOps::IO;


    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
    class ForwardMCRunner {
    public:
        explicit ForwardMCRunner(const KC::UInputs &conf) : config(conf) {
        }

        // ====================================================================
        // The Actual Simulation Engine (Fully Resolved at Compile Time)
        // ====================================================================
        MCResults get_option_prices() const {
            // NOTE : the total number of simulations are performed in batches to handle cases when not enough memory is available
            // NOTE : The global random number pool is created once. States advance dynamically. So using the same pool for different batches is the correct approach

            // 1. Initialize Helper Classes needed during the MC (keep in this local function scope)
            PathsMCBatchMem BatchMem(config); // Handles the Memory
            RNGManager RNGen(config); // Handles the Random number (Must initialize here and not in the batch loop!!!)
            MSolver<ModelPolicy, SchemePolicy, OptType, OptRight> Solver(config); // Handles the Temporal integration
            OptionPricer OPricer(config); // Handles the Option Pricing
            KIO::OutputManager OWriter(config); // Handles the outputs
            MCProgressTracker MCTracker(config, BatchMem);

            // Print Pre-Execution Diagnostics
            MCTracker.print_pre_execution_diagnostic();
            OWriter.print_paths_info_planned_outputs();

            // 3. Run all batches (all mc simulations giving the prices)
            run_all_mc_batches(BatchMem, RNGen, Solver, OPricer, OWriter, MCTracker);

            // 4. Compute Option Price
            OPricer.evaluate_option_price();
            OPricer.print_info_option_price();
            const auto OptionRes = OPricer.get_option_price_data();
            MCResults Res{config, OptionRes};
            return Res;
        }

    private:
        const KC::UInputs config;

        void run_all_mc_batches(PathsMCBatchMem &BatchMem,
                                const RNGManager &RNGen,
                                const MSolver<ModelPolicy, SchemePolicy, OptType, OptRight> &Solver,
                                OptionPricer &OPricer,
                                KIO::OutputManager &OWriter,
                                MCProgressTracker &MCTracker
        ) const {
            // Pass a labda function that copy the payoffs computed to the host to then sort them for percentiles
            auto accumulate_payoffs_func = [&](const int batch_idx, const int current_batch_size) {
                OPricer.accumulate_batch_payoffs(BatchMem.h_payoffs, current_batch_size);
            };

            run_forward_all_mc_batches(BatchMem,
                                       RNGen,
                                       Solver,
                                       OWriter,
                                       MCTracker,
                                       accumulate_payoffs_func);
        }

        void run_all_mc_batches_old(PathsMCBatchMem &BatchMem,
                                    const RNGManager &RNGen,
                                    const MSolver<ModelPolicy, SchemePolicy, OptType, OptRight> &Solver,
                                    OptionPricer &OPricer,
                                    KIO::OutputManager &OWriter,
                                    MCProgressTracker &MCTracker
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


            const int full_batch_size = BatchMem.n_sims_per_batch;
            const int n_full_batches = BatchMem.n_full_batches();
            const int n_sims_left_over = BatchMem.n_sims_left_over_after_full_batches();
            const int n_total_batch_loops = BatchMem.total_batch_loops();

            MCTracker.start_tracking();
            for (int b = 0; b < n_total_batch_loops; ++b) {
                const int current_batch_size = (b < n_full_batches) ? full_batch_size : n_sims_left_over;
                Solver.execute_batch(current_batch_size, BatchMem, RNGen); // Fire off computation kernel on device
                BatchMem.deep_copy_to_host(); // Synch the host with dev
                OPricer.accumulate_batch_payoffs(BatchMem.h_payoffs, current_batch_size);
                // Store payoffs to then sort them for percentiles
                OWriter.save_paths_batch_if_needed(current_batch_size, BatchMem); //Save batch to disk

                MCTracker.update_progress(b + 1);
            }
            MCTracker.finalize_tracking();
        }
    };
}
