#pragma once

#include <iostream>
#include <chrono> // for timing

#include "./SDESolver.hpp"
#include "./../IO/Config.hpp"
#include "./../IO/OutManager.hpp"
#include "./Typedefs.hpp"
#include "./MCMem.hpp"
#include "./RandNGenerator.hpp"
#include "./OptionPricer.hpp"


namespace KOps::Engine {
    namespace KI = KOps::Implemented;

    namespace KT = KOps::Types;
    namespace KIO = KOps::IO;


    struct MCResults {
        const Config::UInputs& MCConfig; // Just give the address, it will be valid since it lives in the main
        const OptionPricer::Results OptionPrice; // Copy the structure so that it is ok when simulation scope end
    };



    class MCProgressTracker {
    public:
        explicit MCProgressTracker(const Config::UInputs &conf, const MCBatchMem &mem)
            : config(conf), batch_mem(mem) {}

        void start_tracking() {
            start_time = std::chrono::steady_clock::now();

            std::cout << "  ========================================================\n";
            std::cout << "                          MC PROGRESS               \n";
            std::cout << "  ========================================================\n\n" << std::flush;

            print_progress_bar(0);
        }

        void update_progress(const int n_current_batch) {
            print_progress_bar(n_current_batch);
        }

        void finalize_tracking() {
            print_progress_bar(batch_mem.total_batch_loops());
            std::cout << "\n\n  ========================================================\n" << std::endl;
        }

        void print_pre_execution_diagnostic() const {
            constexpr std::string_view indent = "  ";

            std::cout << "\n" << indent << "========================================================\n";
            std::cout << indent << "                    MC EXECUTION SPACE                       \n";
            std::cout << indent << "========================================================\n";
            std::cout << indent << " [Hardware Backend Framework]\n";
            std::cout << indent << "   Active Execution Space   :  " << batch_mem.execution_space_name() << "\n";
            std::cout << indent << "   Compute Precision Type   :  " << (sizeof(KT::Real) == 8 ? "64-bit Double" : "32-bit Float") << "\n";
            std::cout << indent << "--------------------------------------------------------\n";
            std::cout << indent << " [Simulation Matrix Framework]\n";
            std::cout << indent << "   Total MC Simulations        :  " << config.mc.N_Paths << "\n";
            std::cout << indent << "   Number Time Steps Per Path  :  " << config.time.N_time_steps << "\n";
            std::cout << indent << "   Time Step                   :  " << config.time.dt << "\n";
            std::cout << indent << "--------------------------------------------------------\n";
            std::cout << indent << " [Memory & Streaming Control]\n"; // (should the class handling the memory should return a structure and a string)
            std::cout << indent << "   Total N Batches to Launch      :  " << batch_mem.total_batch_loops() << "\n";
            std::cout << indent << "   Full Batches to Launch         :  " << batch_mem.n_full_batches() << "\n";
            std::cout << indent << "   N Sims Per Full Batch          :  " << batch_mem.n_sims_per_batch << "\n";
            std::cout << indent << "   Partial Batches to Launch      :  " << batch_mem.total_batch_loops() - batch_mem.n_full_batches() << "\n";
            std::cout << indent << "   N Sims Per Partial Batch       :  " << batch_mem.n_sims_left_over_after_full_batches() << "\n";
            std::cout << indent << "   Batch Paths MEM Footprint      :  " << batch_mem.bytes_to_mb(batch_mem.device_paths_memory_bytes()) << " MB\n";
            std::cout << indent << "   Batch Payoffs MEM Footprint    :  " << batch_mem.bytes_to_mb(batch_mem.device_payoffs_memory_bytes()) << " MB\n";
            std::cout << indent << "   Batch TOT MEM Footprint        :  " << batch_mem.bytes_to_mb(batch_mem.tot_device_memory_bytes()) << " MB\n";
            std::cout << indent << "   Full MC Paths MEM Footprint    :  " << batch_mem.total_paths_footprint_mb() << " MB\n";
            std::cout << indent << "   Full MC Payoffs MEM Footprint  :  " << batch_mem.total_payoffs_footprint_mb() << " MB\n";
            std::cout << indent << "========================================================\n" << std::endl;
        }

    private:
        const Config::UInputs &config;
        const MCBatchMem &batch_mem;
        std::chrono::steady_clock::time_point start_time;

        void print_progress_bar(const int n_current_batch) {
            constexpr int bar_width = 20;
            const int n_total_batches = batch_mem.total_batch_loops();

            const float progress = static_cast<float>(n_current_batch) / static_cast<float>(n_total_batches);
            const int bar_front = static_cast<int>(static_cast<float>(bar_width) * progress);
            const int percentage_done = static_cast<int>(progress * 100.0f);

            std::cout << "     [";
            for (int i = 0; i < bar_width; ++i) {
                if (i < bar_front) std::cout << "#";
                else if (i == bar_front) std::cout << ">";
                else std::cout << ".";
            }

            std::cout << "] " << percentage_done << "% "
                      << "(" << n_current_batch << "/" << n_total_batches << " Batches)";

            if (n_current_batch > 0) {
                const auto current_time = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = current_time - start_time;
                const double elapsed_seconds = elapsed.count();

                const double total_estimated_time = elapsed_seconds / progress;
                const double eta_seconds = total_estimated_time - elapsed_seconds;

                std::cout << std::fixed << std::setprecision(1)
                          << "(T: " << elapsed_seconds << "s | "
                          << "ETA: " << (n_current_batch == n_total_batches ? 0.0 : eta_seconds) << "s)\r";
            }
            std::cout << "\r" << std::flush;
        }
    };



    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
    class MCRunner {
    public:
        explicit MCRunner(const KC::UInputs &conf) : config(conf) {
        }

        // ====================================================================
        // The Actual Simulation Engine (Fully Resolved at Compile Time)
        // ====================================================================
        MCResults run_mc_simulation() const {
            // NOTE : the total number of simulations are performed in batches to handle cases when not enough memory is available
            // NOTE : The global random number pool is created once. States advance dynamically. So using the same pool for different batches is the correct approach

            // 1. Initialize Helper Classes needed during the MC (keep in this local function scope)
            MCBatchMem BatchMem(config); // Handles the Memory
            RNGManager RNGen(config); // Handles the Random number (Must initialize here and not in the batch loop!!!)
            SDESolver<ModelPolicy, SchemePolicy, OptType, OptRight> Solver(config); // Handles the Temporal integration
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


        void run_all_mc_batches(MCBatchMem &BatchMem,
                                const RNGManager &RNGen,
                                const SDESolver<ModelPolicy, SchemePolicy, OptType, OptRight> &Solver,
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
                Solver.execute_batch(current_batch_size, BatchMem, RNGen); // Fire off computation kernel
                BatchMem.deep_copy_to_host(); // Synch the host and dev
                OPricer.accumulate_batch_payoffs(BatchMem.h_payoffs, current_batch_size); // Store payoffs to then sort them for percentiles
                OWriter.save_paths_batch_if_needed(current_batch_size, BatchMem);  //Save batch to disk

                MCTracker.update_progress(b + 1);
            }
            MCTracker.finalize_tracking();
        }
    };
}

