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
#include <chrono>
#include <utility>

#include "../config/Config.hpp"

#include "./../memory/LSMMemory.hpp"

#include "../Typedefs.hpp"
#include "./MCEngine.hpp"
#include "../../IO/OutManager.hpp"

#include "../schemes/RandNGenerator.hpp"
#include "../schemes/markovian/MSolver.hpp"

#include "../options/OptionPricer.hpp"
#include "../options/LSMAmerican.hpp"


namespace quantkos::Engine {
    namespace KI = quantkos::Implemented;
    namespace KT = quantkos::Types;
    namespace KIO = quantkos::IO;
    namespace KIO = quantkos::IO;


    /**
     * @brief Orchestrator for compile-time specialized Forward-Backward Monte Carlo simulations (LSM).
     * * This class implements the Longstaff-Schwartz Method (LSM) for pricing American options
     * on execution spaces managed by Kokkos (e.g., CUDA, HIP, OpenMP). It avoids virtual function
     * overhead by resolving model policies, numerical schemes, and option execution rights at compile-time.
     * * Pricing is structured as a two-phase process:
     * * ### Phase 1: Forward Path Generation
     * Asset price paths are generated on the Device (GPU) in chunked batches to respect hard VRAM or RAM budgets.
     * As each batch is completed, its paths are synchronized and copied into a CPU-bound Master Matrix
     * (`HostMasterPathsView`) using transfers across the PCIe bus.
     * * ### Phase 2: Backward Induction & Least-Squares Regression
     * Starting at $t = T-1$ and moving backward to $t = 1$:
     * 1. A column-major time slice of asset prices is streamed from the Host Master Matrix to the Device.
     * 2. Paths that are In-The-Money (ITM) are regressed against future cash flows using weighted basis functions
     * (Monomial or Laguerre polynomials) via parallel reductions.
     * 3. The regression coefficients ($\beta$) are solved on the Host (CPU) using Cholesky decomposition.
     * 4. Expected continuation values are computed on the Device to determine optimal early exercise boundaries
     * and update the vector of cash flows.
     * * @tparam ModelPolicy Compile-time stochastic process selection (e.g., Heston, Bates; etc.).
     * @tparam SchemePolicy Compile-time SDE integration algorithm (e.g., Euler, Milstein, AndersonQE, etc.).
     * @tparam OptRight Compile-time option exercise right (Call or Put).
     */
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptRight OptRight>
    class ForwardBackwardMCRunner {
    public:
        /**
         * @brief Constructs a forward-backward runner instance bound to the master user configuration.
         * @param conf Configuration inputs containing runtime parameters.
         */
        explicit ForwardBackwardMCRunner(KC::UInputs conf) : config(std::move(conf)) {
            // Apply scaling to the local copy so that the master config is not effected
            config.apply_price_scaling();
        }

        /// @name Lifecycle Protocols
        ///@{
        // Delete copy operations (prevents accidental duplication)
        ForwardBackwardMCRunner(const ForwardBackwardMCRunner &) = delete; ///< Prohibits copy construction.
        ForwardBackwardMCRunner &operator=(const ForwardBackwardMCRunner &) = delete; ///< Prohibits copy assignment.

        // Default the move constructor and delete move assignment
        ForwardBackwardMCRunner(ForwardBackwardMCRunner &&) = default;

        ///< Default move constructor for seamless transfer.
        ForwardBackwardMCRunner &operator=(ForwardBackwardMCRunner &&) = delete; ///< Prohibits move assignment.

        // Default destructor
        ~ForwardBackwardMCRunner() = default; ///< Default destructor.
        ///@}


        /**
         * @brief Entry point to run the American pricing engine.
         * * Manages lifecycle resources, executes the forward path generation phase, coordinates
         * backward induction, and computes final expected values and statistical bands.
         * * @note The solver is forced to act like a European option during Phase 1 because paths are
         * simply generated forward without path-dependent early exercise logic during this stage.
         * * @return Immutable snapshot container of Monte Carlo outputs (`MCResults`).
         */
        MCResults get_option_price() const {
            // Initialize Helper Classes
            // IMPORTANT: We force the forward solver to act like a European option (simply evaluate the forward paths).
            BackwardLSMMemory Mem(config);
            RNGManager RNGen(config);
            MSolver<ModelPolicy, SchemePolicy, KI::OptType::European, OptRight> Solver(config);
            OptionPricer OPricer(config);
            KIO::OutputManager OWriter(config);
            ForwardMCProgressTracker MCTracker(config, Mem.BatchMem);

            // PHASE 1: Forward Batch Generation
            MCTracker.print_pre_execution_diagnostic();
            OWriter.print_paths_info_planned_outputs();
            run_forward_phase(Mem, RNGen, Solver, OWriter, MCTracker);

            // PHASE 2: Backward Induction (Longstaff-Schwartz regression)
            run_backward_phase(Mem);

            // 4. PHASE 3: Compute Final Option Price
            // Feed the optimized backward cashflows into the standard pricer
            OPricer.accumulate_batch_payoffs(Mem.h_best_future_outcomes, config.mc.N_Paths);
            OPricer.evaluate_option_price();
            OPricer.print_info_option_price();

            return MCResults{config, OPricer.get_option_price_data()};
        }

    private:
        KC::UInputs config; ///< Copy of the master config. By copying the configuration the original values cannot be modified.

        // ====================================================================
        // PHASE 1: GENERATE PATHS (Write to CPU RAM temp daata)
        // ====================================================================
        /**
         * @brief Generates paths batch-by-batch and transfers them to the CPU Master Matrix.
         * * @param Mem Backward memory orchestrator holding Host/Device buffers.
         * @param RNGen Monotonic random sequence generator pool.
         * @param Solver SDE integration solver specializing path dynamics.
         * @param OWriter Output files manager.
         * @param MCTracker Console execution diagnostics and progress tracking monitor.
         * @notes Bundles the Host-Device deep-copy logic in a localized callback lambda to decouple
         * SDE solver step-evolutions from the memory mapping strategy.
         */
        void run_forward_phase(BackwardLSMMemory &Mem,
                               const RNGManager &RNGen,
                               const MSolver<ModelPolicy, SchemePolicy, KI::OptType::European, OptRight> &Solver,
                               KIO::OutputManager &OWriter,
                               ForwardMCProgressTracker &MCTracker) const {
            std::cout << "  >>> Starting Phase 1: Forward Path Generation...\n";

            // Pass a lambda function that copies the simulated GPU batch directly
            auto copy_batch_to_master_matrix = [&](const int batch_idx, const int current_batch_size) {
                Mem.copy_current_mc_batch_to_master_mc_matrix(batch_idx, current_batch_size);
            };

            // Call the generic forward runner
            run_forward_all_mc_batches(Mem.BatchMem,
                                       RNGen,
                                       Solver,
                                       OWriter,
                                       MCTracker,
                                       copy_batch_to_master_matrix);
        }

        // ====================================================================
        // PHASE 2: BACKWARD INDUCTION (Read from CPU RAM)
        // ====================================================================
        /**
         * @brief Coordinates the backward induction time-marching regression loop (Uses Longstaff-Schwarz Algorithm).
         * * Iteratively steps backward from maturity. In each step, a time slice is
         * streamed to VRAM, path regressions are constructed, and optimal exercise cash
         * flows are updated in parallel on the GPU.
         * * @param Mem Backward memory orchestrator containing streaming views and regression buffers.
         */
        void run_backward_phase(BackwardLSMMemory &Mem) const {
            const int nT = config.time.N_time_steps;
            const KT::Real strike_price = config.options.StrikePrice;
            const KT::Real discount_factor = Kokkos::exp(-config.market.r * config.time.dt);

            using EngineLSM = LSM::LSMEngine<OptRight>;

            std::cout << "  >>> Starting Phase 2: Backward Induction (PCIe Streaming)...\n";
            BackwardLSMProgressTracker BTracker(config);
            BTracker.start_tracking();

            // Initialization (t = T-1)
            Mem.bring_host_prices_time_slice_to_device(nT - 1);
            EngineLSM::initialize_cashflows(Mem.d_prices_current_time,
                                            Mem.d_best_future_outcomes,
                                            strike_price);
            BTracker.update_progress();

            // THE BACKWARD TIME MARCH
            for (int t = nT - 2; t > 0; --t) {
                // A. Memory Orchestration: Stream CPU RAM -> GPU VRAM
                Mem.bring_host_prices_time_slice_to_device(t);

                // B. Math Orchestration: Regression
                const auto ls_coeffs = EngineLSM::perform_cross_paths_regression(Mem.d_prices_current_time,
                    Mem.d_best_future_outcomes,
                    discount_factor,
                    strike_price);

                // C. Math Orchestration: Early Exercise Evaluation
                EngineLSM::update_cashflows(Mem.d_prices_current_time,
                                            Mem.d_best_future_outcomes,
                                            ls_coeffs,
                                            discount_factor,
                                            strike_price);
                BTracker.update_progress();
            }

            // 3. FINALIZATION (t = 0)
            EngineLSM::apply_final_discount(Mem.d_best_future_outcomes, discount_factor);

            // Pull final results back to CPU
            Mem.bring_device_cash_flows_to_host();

            // Stop tracking the backward phase (safe since no prints in between)
            BTracker.finalize_tracking();
        }
    };
}
