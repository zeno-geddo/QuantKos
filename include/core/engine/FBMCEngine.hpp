// #pragma once
//
//
// #include <iostream>
// #include <chrono>
//
// #include "../config/Config.hpp"
//
// #include "./memory/LSMMemory.hpp"
//
// #include "../Typedefs.hpp"
// #include "./MCUtils.hpp"
// #include "../../IO/OutManager.hpp"
//
// #include "../schemes/RandNGenerator.hpp"
// #include "../schemes/markovian/MSolver.hpp"
//
// #include "../options/OptionPricer.hpp"
// #include "../options/Payoff.hpp"
//
// namespace KOps::Engine {
//     namespace KI = KOps::Implemented;
//     namespace KT = KOps::Types;
//     namespace KIO = KOps::IO;
//
//     template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptRight OptRight>
//     class ForwardBackwardMCRunner {
//     public:
//         explicit ForwardBackwardMCRunner(const KC::UInputs &conf) : config(conf) {}
//
//         MCUtils run_mc_simulation() const {
//             // 1. Initialize Helper Classes
//             // IMPORTANT: We force the forward solver to act like a European option.
//
//             LSMMemory Mem(config);
//             RNGManager RNGen(config);
//             MSolver<ModelPolicy, SchemePolicy, KI::OptType::European, OptRight> Solver(config);
//             OptionPricer OPricer(config);
//             KIO::OutputManager OWriter(config);
//             // MCProgressTracker MCTracker(config, Mem.BatchMem);
//
//             // Print Diagnostics
//             MCTracker.print_pre_execution_diagnostic();
//             OWriter.print_paths_info_planned_outputs();
//
//             // 2. PHASE 1: Forward Batch Generation
//             run_forward_phase(Mem, RNGen, Solver, OWriter, MCTracker);
//
//             // 3. PHASE 2: Backward Induction (Longstaff-Schwartz)
//             run_backward_phase(Mem);
//
//             // 4. PHASE 3: Compute Final Option Price
//             // Feed the optimized backward cashflows into the standard pricer
//             OPricer.accumulate_batch_payoffs(Mem.h_cash_flows, config.mc.N_Paths);
//             OPricer.evaluate_option_price();
//             OPricer.print_info_option_price();
//
//             return MCUtils{config, OPricer.get_option_price_data()};
//         }
//
//     private:
//         const KC::UInputs config;
//
//         // ====================================================================
//         // PHASE 1: GENERATE PATHS (Write to CPU RAM)
//         // ====================================================================
//         void run_forward_phase(LSMMemory &Mem,
//                                const RNGManager &RNGen,
//                                const MSolver<ModelPolicy, SchemePolicy, KI::OptType::European, OptRight> &Solver,
//                                KIO::OutputManager &OWriter,
//                                MCProgressTracker &MCTracker) const {
//
//             const int full_batch_size = Mem.BatchMem.n_sims_per_batch;
//             const int n_full_batches = Mem.BatchMem.n_full_batches();
//             const int n_sims_left_over = Mem.BatchMem.n_sims_left_over_after_full_batches();
//             const int n_total_batch_loops = Mem.BatchMem.total_batch_loops();
//
//             std::cout << "  >>> Starting Phase 1: Forward Path Generation...\n";
//             MCTracker.start_tracking();
//
//             for (int b = 0; b < n_total_batch_loops; ++b) {
//                 const int current_batch_size = (b < n_full_batches) ? full_batch_size : n_sims_left_over;
//                 const int path_start_idx = b * full_batch_size;
//
//                 // 1. Generate paths in GPU VRAM
//                 Solver.execute_batch(current_batch_size, Mem.BatchMem, RNGen);
//
//                 // 2. Pull paths to Host (CPU)
//                 Mem.BatchMem.deep_copy_to_host();
//
//                 // 3. Slot the small batch into the massive CPU master matrix
//                 auto h_sub_master = Kokkos::subview(Mem.h_master_paths,
//                                                     std::make_pair(path_start_idx, path_start_idx + current_batch_size),
//                                                     Kokkos::ALL());
//                 Kokkos::deep_copy(h_sub_master, Mem.BatchMem.h_batch_view);
//
//                 // Optional: Save to disk if user requested
//                 OWriter.save_paths_batch_if_needed(current_batch_size, Mem.BatchMem);
//
//                 MCTracker.update_progress(b + 1);
//             }
//             MCTracker.finalize_tracking();
//         }
//
//         // ====================================================================
//         // PHASE 2: BACKWARD INDUCTION (Read from CPU RAM)
//         // ====================================================================
//         void run_backward_phase(LSMMemory &Mem) const {
//             const int N = config.mc.N_Paths;
//             const int T = config.time.N_time_steps;
//             const KT::Real K = config.options.StrikePrice;
//             const KT::Real discount = Kokkos::exp(-config.model.heston.r * config.time.dt);
//
//             std::cout << "  >>> Starting Phase 2: Backward Induction (PCIe Streaming)...\n";
//
//             auto d_cf = Mem.d_cash_flows;
//             auto d_slice = Mem.d_time_slice;
//
//             // 1. Initialize cash flows at maturity (T-1)
//             auto h_terminal_slice = Kokkos::subview(Mem.h_master_paths, Kokkos::ALL(), T - 1);
//             Kokkos::deep_copy(d_slice, h_terminal_slice); // Stream to GPU
//
//             Kokkos::parallel_for("LSM_Init_Cashflows", N, KOKKOS_LAMBDA(const int i) {
//                 d_cf(i) = Payoff<OptRight>::evaluate_payoff(d_slice(i), K);
//             });
//             Kokkos::fence();
//
//             // 2. March backward through time
//             for (int t = T - 2; t > 0; --t) {
//
//                 // Stream current time slice from CPU to GPU
//                 auto h_column = Kokkos::subview(Mem.h_master_paths, Kokkos::ALL(), t);
//                 Kokkos::deep_copy(d_slice, h_column);
//
//                 // -----------------------------------------------------------------
//                 // [TODO]: Implement A^T A regression kernel here
//                 // Vector3D beta = compute_lsm_regression(d_slice, d_cf, discount);
//                 // -----------------------------------------------------------------
//
//                 // Evaluate early exercise boundary
//                 Kokkos::parallel_for("LSM_Update_Cashflows", N, KOKKOS_LAMBDA(const int i) {
//                     KT::Real S = d_slice(i);
//                     KT::Real intrinsic = Payoff<OptRight>::evaluate_payoff(S, K);
//
//                     // Discount the future cash flow to current step
//                     d_cf(i) *= discount;
//
//                     if (intrinsic > KT::real_zero) { // If In-The-Money
//                         // KT::Real continuation = beta[0] + beta[1]*S + beta[2]*S*S;
//                         KT::Real continuation = KT::real_zero; // Placeholder
//
//                         if (intrinsic > continuation) {
//                             d_cf(i) = intrinsic; // Overwrite future: Exercise now!
//                         }
//                     }
//                 });
//                 Kokkos::fence();
//             }
//
//             // 3. Final step: Pull the fully optimized cashflows back to the Host
//             Kokkos::deep_copy(Mem.h_cash_flows, d_cf);
//         }
//     };
// }
//
// ```