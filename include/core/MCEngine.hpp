#pragma once

#include <iostream>

#include "./SDESolver.hpp"
#include "./../IO/Config.hpp"
#include "./../IO/OutManager.hpp"
#include "./Typedefs.hpp"
#include "./MCMem.hpp"
#include "./RandNGenerator.hpp"





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
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KO = KOps::Out;


    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
    class MCRunner {
    public:
        explicit MCRunner(const KC::UInputs &conf) : config(conf) {
        }

        // ====================================================================
        // LEVEL 3: The Actual Simulation Engine (Fully Resolved at Compile Time)
        // ====================================================================

        void run_mc_simulation() {
            // NOTE : the total number of simulations are performed in batches to handle cases when not enough memory is available

            // 1. Initialize Helper Classes needed during the MC
            MCBatchMem BatchMem(config); // Handles the Memory
            RNGenerator RNGen(config); // Handles the Random number
            SDESolver<ModelPolicy, SchemePolicy> Solver(config); // Handles the Temporal integration
            KO::OutputManager OWriter(config.output); // Handles the outputs

            // 2. Print MC Info
            print_info_planned_mc(BatchMem);

            // 3. Run all batches
            run_all_mc_batches(BatchMem, RNGen, Solver, OWriter);

        }

    private:
        const KC::UInputs config;

        // ====================================================================
        // DIAGNOSTIC HELPER: Summarizes hardware, memory, and runtime math
        // ====================================================================
        void print_info_planned_mc(const MCBatchMem &BatchMem) const {
            std::cout << "    ========================================================\n";
            std::cout << "                  ENGINE EXECUTION SUMMARY                  \n";
            std::cout << "    ========================================================\n";
            std::cout << "      [Hardware Backend Framework]\n";
            std::cout << "        Active Execution Space   :  " << BatchMem.execution_space_name() << "\n";
            std::cout << "        Compute Precision Type   :  " << (sizeof(KT::Real) == 8 ? "64-bit Double" : "32-bit Float") << "\n";
            std::cout << "    --------------------------------------------------------\n";
            std::cout << "      [Simulation Matrix Framework]\n";
            std::cout << "        Total Targeted Paths        :  " << config.mc.N_Paths << "\n";
            std::cout << "        Number Time Steps Per Path  :  " << config.time.N_time_steps << "\n";
            std::cout << "        Time Step                   :  " << config.time.dt << "\n";
            std::cout << "    --------------------------------------------------------\n";
            std::cout << "      [Memory & Streaming Control]\n";
            std::cout << "        Allocated Paths Per Batch:  " << BatchMem.n_sims_per_batch << "\n";
            std::cout << "        Total Stream Loops       :  " << BatchMem.total_batch_loops() << "\n";
            std::cout << "        Hardware Memory / Batch  :  " << BatchMem.device_memory_mb() << " MB\n";
            std::cout << "        Unbatched Direct Footprint: " << BatchMem.total_paths_footprint_mb() << " MB\n";
            std::cout << "    ========================================================\n" << std::endl;
        }


        void run_all_mc_batches(MCBatchMem &BatchMem,
                                const RNGenerator &RNGen,
                                const SDESolver<ModelPolicy, SchemePolicy> &Solver,
                                KO::OutputManager &OWriter) const {

            const int full_batch_size = BatchMem.n_sims_per_batch;
            const int n_full_batches = BatchMem.n_full_batches();
            const int n_sims_left_over = BatchMem.n_sims_left_over_after_full_batches();
            const int total_batch_loops = BatchMem.total_batch_loops();

            for (int b = 0; b < total_batch_loops; ++b) {
                std::cout << "    -> Processing Batch " << b + 1 << "/" << total_batch_loops << "...\r" << std::flush;

                // Fire off the compute kernel
                int current_batch_size = (b < n_full_batches) ? full_batch_size : n_sims_left_over;
                Solver.execute_batch(current_batch_size, BatchMem, RNGen);

                // Synch the host and dev
                BatchMem.deep_copy_to_host(current_batch_size);

                //Save batch to disk (TO BE DONE)
                //OWriter.xxxx()
            }
            std::cout << std::endl;
        }
    };
}
