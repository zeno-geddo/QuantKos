#pragma once

#include "../config/Config.hpp"
#include "../options/OptionPricer.hpp"
#include "./../memory/PathsMCBatchMem.hpp"
#include "../../IO/OutManager.hpp"
#include "../schemes/RandNGenerator.hpp"

namespace KOps::Engine {
    namespace KIO = KOps::IO;

    /**
     * @brief Immutable snapshot container for Monte Carlo simulation outputs.
     * * This structure aggregates the final option pricing metrics alongside a complete,
     * deep copy of the user configuration. Copying the configurations and results guarantees
     * memory safety and prevents lifecycle conflicts if the simulation scope terminates
     * while asynchronous downstream analysis or logging services are still reading the data.
     */
    struct MCResults {
        const Config::UInputs MCConfig; ///< Deep copy of the user parameters used to run the simulation.
        const OptionPricer::MCOpPrices OptionPrice;
        ///< Deep copy of the pricing statistics, standard errors, and confidence intervals.
    };


    /**
     * @brief Console-based progress reporter, timer, and hardware diagnostic engine.
     * * Manages terminal-level visualization for the forward phase of Monte Carlo runs.
     * * Prior to simulation execution, this class queries the batch memory layout to output
     * detailed hardware, execution precision, and VRAM/RAM allocation footprints. During
     * execution, it uses steady-state monotonic clocks to dynamically estimate remaining
     * elapsed durations (ETA) across batch iterations.
     * * @note This execution tool prohibits copies to prevent reference and state corruption
     * of timing benchmarks, while supporting standard move operations.
     */
    class ForwardMCProgressTracker {
    public:
        /**
         * @brief Constructs the progress tracker associated to the simulation context and memory footprint.
         * @param conf Configuration inputs reference containing runtime parameters.
         * @param mem Active batch memory manager used to audit hardware constraints.
         */
        explicit ForwardMCProgressTracker(const Config::UInputs &conf, const PathsMCBatchMem &mem)
            : config(conf), batch_mem(mem) {
        }

        /// @name Lifecycle Protocols
        ///@{
        // Disable copies to prevent reference issues
        ForwardMCProgressTracker(const ForwardMCProgressTracker &) = delete;

        ///< Disabled copy constructor to protect timer integrity.

        ForwardMCProgressTracker &operator=(const ForwardMCProgressTracker &) = delete;

        ///< Disabled copy assignment operator.

        // Allow moves if needed
        ForwardMCProgressTracker(ForwardMCProgressTracker &&) = default; ///< Default move constructor.

        ForwardMCProgressTracker &operator=(ForwardMCProgressTracker &&) = default;

        ///< Default move assignment operator.

        // Default destructor
        ~ForwardMCProgressTracker() = default; ///< Default destructor.
        ///@}


        /**
        * @brief Initializes the tracking state, recording the starting system timestamp.
        * * Outputs a stylized console header and initializes the progress state
        * indicator to zero percent completion.
        */
        void start_tracking() {
            start_time = std::chrono::steady_clock::now();

            std::cout << "  ========================================================\n";
            std::cout << "                 MC PROGRESS  (Forward Phase)             \n";
            std::cout << "  ========================================================\n\n" << std::flush;

            print_progress_bar(0);
        }

        /**
         * @brief Updates the console output to reflect the progress of the current batch.
         * @param n_current_batch Index of the batch loop that has completed execution.
         */
        void update_progress(const int n_current_batch) {
            print_progress_bar(n_current_batch);
        }

        /**
         * @brief Concludes the progress tracking session.
         * * Renders a finalized progress bar (100% completion) and appends structural
         * visual closing boundaries to the terminal output.
         */
        void finalize_tracking() {
            print_progress_bar(batch_mem.total_batch_loops());
            std::cout << "\n\n  ========================================================\n" << std::endl;
        }


        /**
         * @brief Formats and outputs an exhaustive audit of execution parameters and hardware memory footprints.
         * * Prints detailed runtime diagnostics including:
         * - Active hardware backend framework (CUDA, HIP, CPU Threads, etc.)
         * - Floating-point calculation precision (32-bit single vs. 64-bit double precision)
         * - MonteCarlo dimensions (total paths, steps per path, temporal delta step size)
         * - Detailed memory allocation maps (reusable batch footprint vs. unrolled full matrix bounds)
         */
        void print_pre_execution_diagnostic() const {
            constexpr std::string_view indent = "  ";

            std::cout << "\n" << indent << "========================================================\n";
            std::cout << indent << "                    MC EXECUTION SPACE                       \n";
            std::cout << indent << "========================================================\n";
            std::cout << indent << " [Hardware Backend Framework]\n";
            std::cout << indent << "   Active Execution Space   :  " << batch_mem.execution_space_name() << "\n";
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
            // (should the class handling the memory should return a structure and a string)
            std::cout << indent << "   Total N Batches to Launch      :  " << batch_mem.total_batch_loops() << "\n";
            std::cout << indent << "   Full Batches to Launch         :  " << batch_mem.n_full_batches() << "\n";
            std::cout << indent << "   N Sims Per Full Batch          :  " << batch_mem.n_sims_per_batch << "\n";
            std::cout << indent << "   Partial Batches to Launch      :  " << batch_mem.total_batch_loops() - batch_mem.
                    n_full_batches() << "\n";
            std::cout << indent << "   N Sims Per Partial Batch       :  " << batch_mem.
                    n_sims_left_over_after_full_batches() << "\n";
            std::cout << indent << "   Batch Paths MEM Footprint      :  " << batch_mem.bytes_to_mb(
                batch_mem.device_paths_memory_bytes()) << " MB\n";
            std::cout << indent << "   Batch Payoffs MEM Footprint    :  " << batch_mem.bytes_to_mb(
                batch_mem.device_payoffs_memory_bytes()) << " MB\n";
            std::cout << indent << "   Batch TOT MEM Footprint        :  " << batch_mem.bytes_to_mb(
                batch_mem.tot_device_memory_bytes()) << " MB\n";
            std::cout << indent << "   Full MC Paths MEM Footprint    :  " << batch_mem.total_paths_footprint_mb() <<
                    " MB\n";
            std::cout << indent << "   Full MC Payoffs MEM Footprint  :  " << batch_mem.total_payoffs_footprint_mb() <<
                    " MB\n";
            std::cout << indent << "========================================================\n" << std::endl;
        }

    private:
        const Config::UInputs &config; ///< Reference to simulation user configurations.
        const PathsMCBatchMem &batch_mem; ///< Reference to batch memory manager.
        std::chrono::steady_clock::time_point start_time; ///< Timestamp corresponding to tracking initialization.

        /**
         * @brief Iteratively updates the raw terminal UI with a sliding progress indicator and ETA stats.
         * * Computes elapsed run durations, extrapolates total required processing times using
         * linear completion rates, and prints both elapsed time (T) and estimated remaining duration (ETA).
         * Uses `\r` (carriage return) with direct standard output flushes to overwrite the current line in-place.
         * * @param n_current_batch Index of the currently completed batch loop.
         */
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

    // HELPER FUNCTION FOR FORWARD RUNNER
    /**
 * @brief HEAVY-LIFT HELPER FUNCTION FOR FORWARD PHASE. Executes and coordinates the forward Monte Carlo simulation across all path batches.
 * * This helper function drives the batch-by-batch execution loop of the forward simulation phase.
 * It schedules parallel SDE solvers on the target execution space (device), invokes an injected
 * copy/accumulation policy via a callback functor, handles disk-persistence operations, and
 * updates the terminal progress bar.
 * @note Decoupling via Dependency Injection (Callback Functor)
 * The use of the `HostDeepCopy` template parameter allows this driver to remain entirely agnostic
 * of the accumulation or copy mechanics. For instance:
 * - A standard European option pricer passes a lambda to copy and accumulate payoffs only, unless the user required to save yje batch paths.
 * - An American option pricer (such as the Longstaff-Schwartz Method) passes a callback that streams
 * completed batch views directly into a CPU-bound Master Matrix.
 * * This design prevents code duplication and keeps the batching control flow clean and unified.
 * * @tparam SolverType The type of the specialized SDE integration solver.
 * @tparam HostDeepCopy A callable functor matching the signature `void(const int batch_idx, const int current_batch_size)`
 * to handle customized post-batch memory copies and synchronization.
 * * @param BatchMem Reference to the batch memory manager holding views and configuration limits.
 * @param RNGen Reference to the random number sequence pool manager.
 * @param Solver Reference to the specialized temporal integration solver.
 * @param OWriter Reference to the output file persistence and logging manager.
 * @param MCTracker Reference to the console execution progress bar and ETA tracker.
 * @param host_deep_copy The injected lambda or callable functor containing batch-specific host copy logic.
 *
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
    template<typename SolverType, typename HostDeepCopy>
    inline void run_forward_all_mc_batches(
        PathsMCBatchMem &BatchMem,
        const RNGManager &RNGen,
        const SolverType &Solver,
        KIO::OutputManager &OWriter,
        ForwardMCProgressTracker &MCTracker,
        HostDeepCopy &&host_deep_copy // Lambda function to copy desired data to host
    ) {
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

            // 1. Fire parallel integration kernels on the active Kokkos execution space (Device/GPU)
            Solver.execute_batch(current_batch_size, BatchMem, RNGen);

            // 2. Execute the custom host-device synchronization policy (e.g. DMA transfers, accumulation)
            host_deep_copy(b, current_batch_size); // Note : Injected via Lambda

            // 3. Write path diagnostics/trajectories to the file system if configured
            OWriter.save_paths_batch_if_needed(current_batch_size, BatchMem); //Save batch to disk

            // 4. Advance progress and compute linear clock ETA using the tracker defined in the Canvas
            MCTracker.update_progress(b + 1);
        }
        MCTracker.finalize_tracking();
    }
}
