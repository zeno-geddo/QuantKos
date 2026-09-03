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

#include "../config/Config.hpp"
#include "../options/OptionPricer.hpp"
#include "./../memory/PathsMCBatchMem.hpp"
#include "./../memory/MemOSQuery.hpp"
#include "../../IO/OutManager.hpp"
#include "../schemes/RandNGenerator.hpp"

namespace quantkos::Engine {
    namespace KIO = quantkos::IO;

    /**
   * @brief Container holding the final risk-neutral Greeks.
   */
    struct MCGreeks {
        // --- First & Second Order Greeks ---
        double delta = 0.0; // Sensitivity to underlying spot price (S)
        double gamma = 0.0; // Rate of change of Delta (Curvature)
        double vega = 0.0; // Sensitivity to volatility (v0)
        double vomma = 0.0; // Rate of change of vega (Curvature)
        double rho = 0.0; // Sensitivity to interest rates (r)
        double theta = 0.0; // Sensitivity to final time (T)
        /**
     * @brief Optional helper to print a clean summary table to the console.
     */
        void print_summary() const {
            constexpr std::string_view indent = "  ";
            std::cout << "\n\n" << indent << "========================================\n";
            std::cout << indent << "        MONTE CARLO GREEKS    \n";
            std::cout << indent << "========================================\n";
            std::cout << indent << " Delta <-> Sensitivity to underlying spot price (S)  : " << delta << "\n";
            std::cout << indent << " Gamma <-> Rate of change of Delta                   : " << gamma << "\n";
            std::cout << indent << " Vega <-> Sensitivity to volatility (v0)             : " << vega << "\n";
            std::cout << indent << " Vomma <-> SRate of change of Vega                   : " << vomma << "\n";
            std::cout << indent << " Rho <-> Sensitivity to interest rates (r)           : " << rho << "\n";
            std::cout << indent << " Theta <-> Sensitivity to time decay (T)             : " << theta << "\n";
            std::cout << indent << "========================================\n";
        }
    };

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
        const MCGreeks Greeks; ///< Deep copy of the computed Greeks
    };


    /**
     * @brief Console-based progress reporter, timer, and hardware diagnostic engine for the forward MC phase.
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
            std::cout << indent << "   Build Mode               :  " << KT::get_build_mode_string() << "\n";
            std::cout << indent << "   Active Execution Space   :  " << batch_mem.execution_space_name() << "\n";
            std::cout << indent << "   Compute Precision Type   :  " << KT::get_precision_string() << "\n";
            std::cout << indent << "   Use Fast Math            :  " << KT::get_fast_math_string() << "\n";
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
            std::cout << indent << "   Paths MEM allocated ?          :  " << batch_mem.are_paths_allocated() << "\n";
            std::cout << indent << "   Batch Paths MEM Footprint      :  " << batch_mem.bytes_to_mb(
                batch_mem.device_paths_memory_bytes()) << " MB\n";
            std::cout << indent << "   Payoffs MEM allocated ?        :  1\n";
            std::cout << indent << "   Batch Payoffs MEM Footprint    :  " << batch_mem.bytes_to_mb(
                batch_mem.device_payoffs_memory_bytes()) << " MB\n";
            std::cout << indent << "   Batch TOT MEM Footprint        :  " << batch_mem.bytes_to_mb(
                batch_mem.tot_device_memory_bytes()) << " MB\n";
            std::cout << indent << "   Est. Full MC Paths MEM Footprint   :  " << batch_mem.total_paths_footprint_mb()
                    <<
                    " MB\n";
            std::cout << indent << "   Est. Full MC Payoffs MEM Footprint :  " << batch_mem.total_payoffs_footprint_mb()
                    <<
                    " MB\n";
            std::cout << indent << "   Remaining RAM (OS Query)           :  " << get_available_memory_from_os_mb() <<
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

    // ========================================================================
    // BACKWARD MC PHASE PROGRESS TRACKER
    // ========================================================================
    /**
     * @brief Console-based progress reporter and timer for the Backward induction Phase (LSM) phase.
     * * Tracks the reverse time-marching loop, utilizing an internal incrementing counter
     * to seamlessly evaluate ETAs without requiring complex reversed step indices.
     */
    class BackwardLSMProgressTracker {
    public:
        /**
         * @brief Constructs the LSM induction progress tracker.
         * @param conf Reference to master configuration inputs reference containing the total time steps.
         */
        explicit BackwardLSMProgressTracker(const Config::UInputs &conf)
            : config(conf), steps_completed(0) {
            total_lsm_steps = config.time.N_time_steps - 1;
        }

        /// @name Lifecycle Protocols
        ///@{
        BackwardLSMProgressTracker(const BackwardLSMProgressTracker &) = delete;

        BackwardLSMProgressTracker &operator=(const BackwardLSMProgressTracker &) = delete;

        BackwardLSMProgressTracker(BackwardLSMProgressTracker &&) = default;

        BackwardLSMProgressTracker &operator=(BackwardLSMProgressTracker &&) = delete;

        ~BackwardLSMProgressTracker() = default;

        ///@}

        /**
         * @brief Initializes the tracking state and sets up the console visual layout.
         */
        void start_tracking() {
            start_time = std::chrono::steady_clock::now();
            steps_completed = 0;

            std::cout << "  ========================================================\n";
            std::cout << "               BACKWARD PHASE (LSM Induction)          \n";
            std::cout << "  ========================================================\n\n" << std::flush;

            print_progress_bar(0);
        }

        /**
         * @brief Increments the internal completion counter and updates the console bar and ETA.
         * * Call this exactly once per backward step loop iteration.
         */
        void update_progress() {
            steps_completed++;

            // Calculate current integer percentage
            if (total_lsm_steps <= 0) return; // Prevent division by zero if there are almost no time steps
            const int current_percent = static_cast<int>(
                (static_cast<float>(steps_completed) / static_cast<float>(total_lsm_steps)) * 100.0f
            );

            // Print the progress bar if target percentage reached
            if (current_percent >= last_printed_percent + update_interval_percent) {
                print_progress_bar(steps_completed);
                last_printed_percent = current_percent; // Update the threshold
            }
        }

        /**
         * @brief Completes the progress bar and seals the terminal section.
         */
        void finalize_tracking() {
            print_progress_bar(total_lsm_steps);
            std::cout << "\n\n  ========================================================\n" << std::endl;
        }

    private:
        const Config::UInputs &config; ///< Reference to the master configuration tree
        int total_lsm_steps; //< Internal counter keeping the loop size (// T-1 regressions steps)
        int steps_completed; ///< Internal counter bridging the reverse induction loop mapping
        static constexpr int update_interval_percent = 10;
        //< Frequency, in percentage of the total, when to print time steps
        int last_printed_percent = 0; ///< Internal counter keeping the last percentage printed
        std::chrono::steady_clock::time_point start_time; ///< Starting time of the backward phase

        void print_progress_bar(const int current_step) const {
            constexpr int bar_width = 20;

            // Prevent division by zero if there are almost no time steps
            if (total_lsm_steps <= 0) return;

            const float progress = static_cast<float>(current_step) / static_cast<float>(total_lsm_steps);
            const int bar_front = static_cast<int>(static_cast<float>(bar_width) * progress);
            const int percentage_done = static_cast<int>(progress * 100.0f);

            std::cout << "     [";
            for (int i = 0; i < bar_width; ++i) {
                if (i < bar_front) std::cout << "#";
                else if (i == bar_front) std::cout << "<"; // Use a backward arrow for aesthetics
                else std::cout << ".";
            }

            std::cout << "] " << percentage_done << "% "
                    << "(" << current_step << "/" << total_lsm_steps << " Time Slices)";

            if (current_step > 0) {
                const auto current_time = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = current_time - start_time;
                const double elapsed_seconds = elapsed.count();

                const double total_estimated_time = elapsed_seconds / progress;
                const double eta_seconds = total_estimated_time - elapsed_seconds;

                std::cout << std::fixed << std::setprecision(1)
                        << "(T: " << elapsed_seconds << "s | "
                        << "ETA: " << (current_step == total_lsm_steps ? 0.0 : eta_seconds) << "s)\r";
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
 * @param custom_host_deep_copy The injected lambda or callable functor containing batch-specific host copy logic.
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
        HostDeepCopy &&custom_host_deep_copy // Lambda function to copy desired data to host
    ) {
        const int full_batch_size = BatchMem.n_sims_per_batch;
        const int n_full_batches = BatchMem.n_full_batches();
        const int n_sims_left_over = BatchMem.n_sims_left_over_after_full_batches();
        const int n_total_batch_loops = BatchMem.total_batch_loops();

        MCTracker.start_tracking();
        for (int n_curr_batch = 0; n_curr_batch < n_total_batch_loops; ++n_curr_batch) {
            const int current_batch_size = (n_curr_batch < n_full_batches) ? full_batch_size : n_sims_left_over;

            // 1. Fire parallel integration kernels on the active Kokkos execution space (Device/GPU)
            Solver.execute_batch(current_batch_size, BatchMem, RNGen);

            // 2. Execute the custom host-device synchronization policy (e.g. DMA transfers, accumulation)
            custom_host_deep_copy(n_curr_batch, current_batch_size); // Note : Injected via Lambda

            // 3. Write path diagnostics/trajectories to the file system if configured
            OWriter.save_paths_batch_if_needed(current_batch_size, BatchMem); //Save batch to disk

            // 4. Advance progress and compute linear clock ETA using the tracker defined in the Canvas
            MCTracker.update_progress(n_curr_batch + 1);
        }
        MCTracker.finalize_tracking();
    }


    /**
     * @brief Universal Monte Carlo orchestrator for base pricing and finite-difference Greeks computation.
     *
     * @details
     * This template function acts as the central execution backbone for all Monte Carlo engines
     * (both Forward-Only and Forward-Backward/LSM). It decouples the high-level orchestration
     * of Greek bumps and memory management from the low-level numerical pricing logic.
     *
     * The execution pipeline is strictly divided into three phases:
     * - **Phase 1: Zero-Allocation Memory Setup**
     *   Allocates the heavy GPU/CPU memory matrix (`MemoryType`) exactly once. This completely
     *   eliminates memory thrashing (OS-level re-allocations) during the multiple simulation
     *   passes required for the Greeks.
     * - **Phase 2: Base Pricing Step**
     *   Executes the injected `pricing_function` using the master configuration. This step
     *   handles all standard I/O (e.g., saving paths to disk if required etc.) and distribution analysis.
     * - **Phase 3: Greek Computation (CRN & Central Differences)**
     *   If Greeks are requested, the engine generates a stripped-down `base_greek_cfg` that
     *   systematically mutes unrequired I/O operations and array sorting. It then performs
     *   central-difference bumps ($S_0$, $v_0$, $r$) and feeds them back into the pricing
     *   function. Because the spoofed configurations retain the original random seed, the
     *   underlying engine naturally executes using Common Random Numbers (CRN), guaranteeing
     *   stable derivative approximations.
     *
     * @tparam MemoryType The specialized heavy memory manager class (e.g., `PathsMCBatchMem`
     *         for European/Path-Dependent options, or `BackwardLSMMemory` for American LSM).
     * @tparam PricingCallable A lambda or functor containing the specific simulation logic.
     *         Must match the signature: `OptionPricer::MCOpPrices(const KC::UInputs& cfg, MemoryType& Mem)`.
     *
     * @param config The master user configuration containing market parameters, simulation settings,
     *        and flags indicating which Greeks to compute.
     * @param pricing_function The injected executable strategy that performs a single Monte Carlo pass.
     *
     * @return MCResults A snapshot struct containing a copy of the input config,
     *         the base option price (with statistical errors), and the computed Greeks.
     *
     */
    template<typename MemoryType, typename PricingCallable>
    inline MCResults execute_mc_framework(const KC::UInputs &config, PricingCallable &&pricing_function) {
        // 1. Allocate Heavy Memory ONCE for the exact type requested
        MemoryType Mem(config);

        // 2. Get the option Price
        OptionPricer::MCOpPrices ref_option_price_data = pricing_function(config, Mem);


        // 3. Compute Greeks if needed (Using the generic pricing callable)
        MCGreeks greeks{};
        if (config.must_compute_greeks()) {
            KC::UInputs base_greek_cfg = config; // Create a copy where spoofing params common to all greeks
            base_greek_cfg.mc.analyze_risk_neutral_payoff_distribution = false; // payoffs never stored and sorted
            base_greek_cfg.output.filename_paths_out = ""; // So that paths for the greeks are never saved
            // should introduce and rewrite a flag that tell if to print or not, and during the greeks should be low verbosity

            if (base_greek_cfg.mc.compute_delta_et_gamma) {
                std::cout << "\n\n  >>> Computing Delta & Gamma...\n";
                // Get initial price and price bump considering that it could have been normalized
                const auto S0 = base_greek_cfg.market.S0;
                const auto h_S = S0 * base_greek_cfg.mc.spot_price_relative_bump_size;
                auto h_S_real_scale = h_S;
                if (base_greek_cfg.mc.normalize_prices) {
                    const auto S0_real_scale = base_greek_cfg.get_price_scaling_factor();
                    h_S_real_scale = S0_real_scale * base_greek_cfg.mc.spot_price_relative_bump_size;
                }

                KC::UInputs cfg_up = base_greek_cfg;
                cfg_up.market.S0 = S0 + h_S;
                double p_up = pricing_function(cfg_up, Mem).option_price;

                KC::UInputs cfg_down = base_greek_cfg;
                cfg_down.market.S0 = S0 - h_S;
                double p_down = pricing_function(cfg_down, Mem).option_price;

                // NOTE : Divide by the bump in the real scale to match the scale of the real output prices
                greeks.delta = (p_up - p_down) / (2.0 * h_S_real_scale);
                greeks.gamma = (p_up - 2.0 * ref_option_price_data.option_price + p_down) /
                               (h_S_real_scale * h_S_real_scale);
            }

            if (base_greek_cfg.mc.compute_vega_et_vomma) {
                // Approach 1
                std::cout << "\n\n  >>> Computing Vega...\n";
                const double var0 = base_greek_cfg.market.v0;
                const double vol0 = std::sqrt(var0);
                const double h_vol = base_greek_cfg.mc.volatility_absolute_bump_size;

                KC::UInputs cfg_up = base_greek_cfg;
                const double vol_up = vol0 + h_vol;
                const double var_up = vol_up * vol_up;
                cfg_up.market.v0 = var_up; // variance instead of volatility required
                double p_up = pricing_function(cfg_up, Mem).option_price;

                KC::UInputs cfg_down = base_greek_cfg;
                const double vol_down = vol0 - h_vol;
                const double var_down = vol_down * vol_down;
                cfg_down.market.v0 = var_down; // variance instead of volatility required
                double p_down = pricing_function(cfg_down, Mem).option_price;

                greeks.vega = (p_up - p_down) / (2.0 * h_vol);
                greeks.vomma = (p_up - 2.0 * ref_option_price_data.option_price + p_down) / (h_vol * h_vol);
            }

            if (base_greek_cfg.mc.compute_rho) {
                std::cout << "\n\n  >>> Computing Rho...\n";
                const double r = base_greek_cfg.market.r;
                const double h_r = base_greek_cfg.mc.risk_free_rate_absolute_bump_size;

                KC::UInputs cfg_up = base_greek_cfg;
                cfg_up.market.r = r + h_r;
                double p_up = pricing_function(cfg_up, Mem).option_price;

                KC::UInputs cfg_down = base_greek_cfg;
                cfg_down.market.r = r - h_r;
                double p_down = pricing_function(cfg_down, Mem).option_price;

                greeks.rho = (p_up - p_down) / (2.0 * h_r);
            }

            if (base_greek_cfg.mc.compute_theta) {
                std::cout << "\n\n  >>> Computing Theta...\n";
                const double T = base_greek_cfg.time.t_end; // time to maturity
                const double h_T = base_greek_cfg.mc.time_absolute_bump_size;

                // Safety check: Ensure the option has more time left than the bump size
                if (T > h_T) {
                    KC::UInputs cfg_down = base_greek_cfg;

                    // Continuous Maturity Shift: Shrink T, but keep N_time_steps the same
                    // NOTE: To price the option from the perspective of tomorrow using a simulation that starts at $0$,
                    // we set the total simulation horizon ($t_{\text{end}}$) to the remaining time: $T - h_T$.
                    // In other words, we are standing at higher unit time (e.g. tomorrow), so, because 1 unit time has elapsed,
                    // the time remaining until maturity is now T−ht
                    cfg_down.time.t_end = T - h_T;

                    // Manually compress the continuous grid step to fit the new time, but keep N_time_steps the same
                    cfg_down.time.dt = cfg_down.time.t_end / base_greek_cfg.time.N_time_steps;

                    // Now run the pricing function with the compressed grid (the grid has the same shape)
                    double p_down = pricing_function(cfg_down, Mem).option_price;

                    // Annualized Theta: (Price with perspective of tomorrow - Price with perspective of today) / time elapsed
                    greeks.theta = (p_down - ref_option_price_data.option_price) / h_T;
                } else {
                    std::cout <<
                            "   [Warning] Option too close to expiry to compute finite-difference Theta. Theta set to zero.\n";
                    greeks.theta = 0.0;
                }
            }

            greeks.print_summary(); // print the summary of the greeks computed
        }

        // 4. Return combined results
        return MCResults{
            .MCConfig = config,
            .OptionPrice = ref_option_price_data,
            .Greeks = greeks
        };
    }
}
