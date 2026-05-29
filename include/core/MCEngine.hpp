#pragma once

#include <iostream>
#include <chrono> // for timing

#include "./SDESolver.hpp"
#include "./../IO/Config.hpp"
#include "./../IO/OutManager.hpp"
#include "./Typedefs.hpp"
#include "./MCMem.hpp"
#include "./RandNGenerator.hpp"
#include "./Risk.hpp"


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


namespace KOps::Engine {
    namespace KI = KOps::Implemented;

    namespace KT = KOps::Types;
    namespace KIO = KOps::IO;


    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
    class MCRunner {
    public:
        explicit MCRunner(const KC::UInputs &conf) : config(conf) {
        }

        // ====================================================================
        // LEVEL 3: The Actual Simulation Engine (Fully Resolved at Compile Time)
        // ====================================================================

        void run_mc_simulation() {
            // NOTE : the total number of simulations are performed in batches to handle cases when not enough memory is available
            // NOTE : The global random number pool is created once. States advance dynamically. So using the same pool for different batches is the correct approach

            // 1. Initialize Helper Classes needed during the MC
            MCBatchMem BatchMem(config); // Handles the Memory
            RNGManager RNGen(config); // Handles the Random number (Must initialize here and not in the batch loop!!!)
            SDESolver<ModelPolicy, SchemePolicy, OptType, OptRight> Solver(config); // Handles the Temporal integration
            KIO::OutputManager OWriter(config); // Handles the outputs

            // 2. Print Pre-Execution Diagnostics
            print_info_planned_mc(BatchMem);
            OWriter.print_paths_info_planned_outputs();

            // 3. Run all batches
            run_all_mc_batches(BatchMem, RNGen, Solver, OWriter);
        }

    private:
        const KC::UInputs config;

        // ====================================================================
        // DIAGNOSTIC HELPER: Summarizes hardware, memory, and runtime math
        // ====================================================================
        void print_info_planned_mc(const MCBatchMem &BatchMem) const {
            constexpr std::string_view indent = "  ";

            std::cout << "\n" << indent << "========================================================\n";
            std::cout << indent << "                     MC EXECUTION SPACE                       \n";
            std::cout << indent << "========================================================\n";
            std::cout << indent << " [Hardware Backend Framework]\n";
            std::cout << indent << "   Active Execution Space   :  " << BatchMem.execution_space_name() << "\n";
            std::cout << indent << "   Compute Precision Type   :  " << (sizeof(KT::Real) == 8
                                                                             ? "64-bit Double"
                                                                             : "32-bit Float") << "\n";
            std::cout << indent << "--------------------------------------------------------\n";
            std::cout << indent << " [Simulation Matrix Framework]\n";
            std::cout << indent << "   Total MC Simulations        :  " << config.mc.N_Paths << "\n";
            std::cout << indent << "   Number Time Steps Per Path  :  " << config.time.N_time_steps << "\n";
            std::cout << indent << "   Time Step                   :  " << config.time.dt << "\n";
            std::cout << indent << "--------------------------------------------------------\n";
            std::cout << indent << " [Memory & Streaming Control]\n";
            std::cout << indent << "   Total N Batches to Launch      :  " << BatchMem.total_batch_loops() << "\n";
            std::cout << indent << "   Full Batches to Launch         :  " << BatchMem.n_full_batches() << "\n";
            std::cout << indent << "   N Sims Per Full Batch          :  " << BatchMem.n_sims_per_batch << "\n";
            std::cout << indent << "   Partial Batches to Launch      :  " << BatchMem.total_batch_loops() - BatchMem.n_full_batches() << "\n";
            std::cout << indent << "   N Sims Per Partial Batch       :  " << BatchMem.n_sims_left_over_after_full_batches() << "\n";
            std::cout << indent << "   Batch Paths MEM Footprint      :  " << BatchMem.bytes_to_mb(BatchMem.device_paths_memory_bytes()) << " MB\n";
            std::cout << indent << "   Batch Payoffs MEM Footprint    :  " << BatchMem.bytes_to_mb(BatchMem.device_payoffs_memory_bytes()) << " MB\n";
            std::cout << indent << "   Batch TOT MEM Footprint        :  " << BatchMem.bytes_to_mb(BatchMem.tot_device_memory_bytes()) << " MB\n";
            std::cout << indent << "   Full MC Paths MEM Footprint    :  " << BatchMem.total_paths_footprint_mb() << " MB\n";
            std::cout << indent << "   Full MC Payoffs MEM Footprint  :  " << BatchMem.total_payoffs_footprint_mb() << " MB\n";
            std::cout << indent << "========================================================\n" << std::endl;
        }


        void run_all_mc_batches(MCBatchMem &BatchMem,
                                const RNGManager &RNGen,
                                const SDESolver<ModelPolicy, SchemePolicy, OptType, OptRight> &Solver,
                                KIO::OutputManager &OWriter) const {
            const int full_batch_size = BatchMem.n_sims_per_batch;
            const int n_full_batches = BatchMem.n_full_batches();
            const int n_sims_left_over = BatchMem.n_sims_left_over_after_full_batches();
            const int n_total_batch_loops = BatchMem.total_batch_loops();

            std::cout << "  ========================================================\n";
            std::cout << "                         MC PROGRESS              \n";
            std::cout << "  ========================================================\n"<< std::endl;
            const auto start_time = std::chrono::steady_clock::now();
            double elapsed_seconds = 0.;
            print_batches_progress_bar(0, n_total_batch_loops, elapsed_seconds);

            RiskQuantifier RiskQ = RiskQuantifier(config);

            for (int b = 0; b < n_total_batch_loops; ++b) {
                const int current_batch_size = (b < n_full_batches) ? full_batch_size : n_sims_left_over;

                Solver.execute_batch(current_batch_size, BatchMem, RNGen); // Fire off computation kernel
                BatchMem.deep_copy_to_host(); // Synch the host and dev
                RiskQ.accumulate_batch_payoffs(BatchMem.h_payoffs, current_batch_size); // Store payoffs to then sort them
                OWriter.save_paths_batch_if_needed(current_batch_size, BatchMem);  //Save batch to disk

                elapsed_seconds = get_elapsed_seconds(start_time);
                print_batches_progress_bar(b, n_total_batch_loops, elapsed_seconds);

            }
            print_batches_progress_bar(n_total_batch_loops, n_total_batch_loops, elapsed_seconds);
            std::cout << "\n\n  ========================================================\n" << std::endl;

            RiskQ.compute_risks_metrics();
            RiskQ.print_report();
        }

        [[nodiscard]] double get_elapsed_seconds(const std::chrono::steady_clock::time_point &start_time) const {
            const auto current_time = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = current_time - start_time;
            return elapsed.count();
        }


        void print_batches_progress_bar(const int n_current_batch, const int n_total_batches,
                                        const double elapsed_seconds) const {
            constexpr int bar_width = 20;

            const float progress = static_cast<float>(n_current_batch) / static_cast<float>(n_total_batches);
            const int bar_front = static_cast<int>(static_cast<float>(bar_width) * progress);
            const int percentage_done = static_cast<int>(progress * 100.0f);

            // Print the bar
            std::cout << "     [";
            for (int i = 0; i < bar_width; ++i) {
                if (i < bar_front) std::cout << "#";
                else if (i == bar_front) std::cout << ">";
                else std::cout << ".";
            }

            // Print percentage alongside fractional math tracking
            std::cout << "] " << percentage_done << "% "
                    << "(" << n_current_batch << "/" << n_total_batches << " Batches)";

            // Print elapsed time
            if (n_current_batch > 0) {
                const double total_estimated_time = elapsed_seconds / progress;
                const double eta_seconds = total_estimated_time - elapsed_seconds;
                std::cout << std::fixed << std::setprecision(1)
                        << "(T: " << elapsed_seconds << "s | "
                        << "ETA: " << (n_current_batch == n_total_batches ? 0.0 : eta_seconds) << "s)\r";
            }

            // Finalize the line
            std::cout << "\r" << std::flush;
        }
    };
}

