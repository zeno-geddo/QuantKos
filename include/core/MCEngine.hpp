#pragma once

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "SDESchemes.hpp"
#include "./../IO/Config.hpp"
#include "./../IO/OutManager.hpp"
#include "./Typedefs.hpp"
#include "./MCMem.hpp"
#include "./RandNGenerator.hpp"


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

            std::cout << "Starting Monte Carlo Simulation..." << std::endl;

            // 1. Initialize Helper Classes needed during the MC
            MCBatchMem BatchMem(config); // Handles the Memory
            RNGenerator RNGen(config); // Handles the Random number
            SDESchemes<ModelPolicy, SchemePolicy> Scheme(config); // Handles the Temporal integration
            KO::OutputManager OWriter(config.output); // Handles the outputs

            // 2. Run all batches
            run_all_mc_batches(BatchMem, RNGen, Scheme, OWriter);

            std::cout << "Simulation completed successfully!" << std::endl;
        }

    private:
        const KC::UInputs config;

        void run_all_mc_batches(MCBatchMem &BatchMem,
                                const RNGenerator &RNGen,
                                const SDESchemes<ModelPolicy, SchemePolicy> &Scheme,
                                KO::OutputManager &OWriter) {
            const int total_paths = config.mc.N_Paths;
            const int batch_size = BatchMem.n_sims_per_batch;

            const int n_full_batches = total_paths / batch_size;
            const int n_sims_left_over = total_paths % batch_size;
            const int total_batch_loops = n_full_batches + (n_sims_left_over > 0 ? 1 : 0);

            for (int b = 0; b < total_batch_loops; ++b) {
                std::cout << "    -> Processing Batch " << b + 1 << "/" << total_batch_loops << "...\r" << std::flush;

                // Fire off the compute kernel
                int current_batch_size = (b < n_full_batches) ? batch_size : n_sims_left_over;
                run_sims_of_current_batch(current_batch_size, BatchMem, RNGen, Scheme);

                // Synch the host and dev
                BatchMem.deep_copy_to_host(current_batch_size);

                //Save batch to disk (TO BE DONE)
                //OWriter.xxxx()
            }
            std::cout << std::endl;
        }

        void run_sims_of_current_batch(int n_active_paths,
                                       MCBatchMem &BatchMem,
                                       const RNGenerator &RNGen,
                                       const SDESchemes<ModelPolicy, SchemePolicy> &Scheme) {
            auto local_batch_view = BatchMem.d_batch_view;
            auto local_pool = RNGen.get_pool();

            const int n_t_steps = config.time.N_time_steps;
            const KT::Real S0 = config.init.S0;
            const KT::Real v0 = config.init.v0;

            // The boundary limit 'active_paths' completely protects the matrix boundaries safely
            Kokkos::parallel_for("EvolveSDEs", n_active_paths, KOKKOS_LAMBDA(const int n_p)
            {
                auto rn_generator = local_pool.get_state();

                KT::Real S = S0;
                KT::Real v = v0;

                for (int n_t = 0; n_t < n_t_steps; ++n_t) {
                    auto [next_S, next_v] = Scheme.evolve_step(S, v, rn_generator);

                    S = next_S;
                    v = next_v;

                    local_batch_view(n_p, n_t) = S; // Fast native layout indexing math!
                }
                local_pool.free_state(rn_generator);
            }
            )
            ;
            Kokkos::fence();
        }
    };
}
