#pragma once


#include <iostream>
#include <chrono>

#include "../config/Config.hpp"

#include "./../memory/LSMMemory.hpp"

#include "../Typedefs.hpp"
#include "./MCEngine.hpp"
#include "../../IO/OutManager.hpp"

#include "../schemes/RandNGenerator.hpp"
#include "../schemes/markovian/MSolver.hpp"

#include "../options/OptionPricer.hpp"
#include "../options/LSMAmerican.hpp"


namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;
    namespace KIO = KOps::IO;

    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptRight OptRight>
    class ForwardBackwardMCRunner {
    public:
        explicit ForwardBackwardMCRunner(const KC::UInputs &conf) : config(conf) {
        }

        // Delete copy operations (prevents accidental duplication)
        ForwardBackwardMCRunner(const ForwardBackwardMCRunner&) = delete;
        ForwardBackwardMCRunner& operator=(const ForwardBackwardMCRunner&) = delete;

        // Default the move constructor and delete move assignment
        ForwardBackwardMCRunner(ForwardBackwardMCRunner&&) = default;
        ForwardBackwardMCRunner& operator=(ForwardBackwardMCRunner&&) = delete;

        // Default destructor
        ~ForwardBackwardMCRunner() = default;

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
        const KC::UInputs& config;

        // ====================================================================
        // PHASE 1: GENERATE PATHS (Write to CPU RAM temp daata)
        // ====================================================================
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
        void run_backward_phase(BackwardLSMMemory &Mem) const {
            const int nT = config.time.N_time_steps;
            const KT::Real strike_price = config.options.StrikePrice;
            const KT::Real discount_factor = Kokkos::exp(-config.market.r * config.time.dt);

            using EngineLSM = LSM::LSMEngine<OptRight>;

            std::cout << "  >>> Starting Phase 2: Backward Induction (PCIe Streaming)...\n";

            // Initialization (t = T-1)
            Mem.bring_host_prices_time_slice_to_device(nT - 1);
            EngineLSM::initialize_cashflows(Mem.d_prices_current_time,
                                            Mem.d_best_future_outcomes,
                                            strike_price);

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
            }

            // 3. FINALIZATION (t = 0)
            EngineLSM::apply_final_discount(Mem.d_best_future_outcomes, discount_factor);

            // Pull final results back to CPU
            Mem.bring_device_cash_flows_to_host();
        }
    };
}
