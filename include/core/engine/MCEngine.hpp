#pragma once

#include "../config/Config.hpp"
#include "../options/OptionPricer.hpp"
#include "./../memory/PathsMCBatchMem.hpp"
#include "../../IO/OutManager.hpp"
#include "../schemes/RandNGenerator.hpp"

namespace KOps::Engine {

    namespace KIO = KOps::IO;

    struct MCResults {
        const Config::UInputs &MCConfig; // Just give the address, it will be valid since it lives in the main
        const OptionPricer::MCEngine OptionPrice; // Copy the structure so that it is ok when simulation scope end
    };


    class ForwardMCProgressTracker {
    public:
        explicit ForwardMCProgressTracker(const Config::UInputs &conf, const PathsMCBatchMem &mem)
            : config(conf), batch_mem(mem) {}

        // Disable copies to prevent reference issues
        ForwardMCProgressTracker(const ForwardMCProgressTracker&) = delete;
        ForwardMCProgressTracker& operator=(const ForwardMCProgressTracker&) = delete;

        // Allow moves if needed
        ForwardMCProgressTracker(ForwardMCProgressTracker&&) = default;
        ForwardMCProgressTracker& operator=(ForwardMCProgressTracker&&) = default;

        void start_tracking() {
            start_time = std::chrono::steady_clock::now();

            std::cout << "  ========================================================\n";
            std::cout << "                 MC PROGRESS  (Forward Phase)             \n";
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
        const PathsMCBatchMem &batch_mem;
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


    template<typename SolverType, typename HostDeepCopy>
    inline void run_forward_all_mc_batches(
            PathsMCBatchMem &BatchMem,
            const RNGManager &RNGen,
            const SolverType &Solver,
            KIO::OutputManager &OWriter,
            ForwardMCProgressTracker &MCTracker,
            HostDeepCopy&& host_deep_copy // Lambda function to copy desired data to host
            )
    {


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
            Solver.execute_batch(current_batch_size, BatchMem, RNGen);
            // BatchMem.deep_copy_to_host();

            // EXECUTE CUSTOM HOST LOGIC TO MOVE DATA TO MEMORY (Injected via Lambda)
            host_deep_copy(b, current_batch_size);

            OWriter.save_paths_batch_if_needed(current_batch_size, BatchMem);   //Save batch to disk
            MCTracker.update_progress(b + 1);
        }
        MCTracker.finalize_tracking();
    }
}


