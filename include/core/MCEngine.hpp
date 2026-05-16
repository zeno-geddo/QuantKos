#pragma once

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "SDESolver.hpp"
#include "./../IO/Config.hpp"
#include "./../IO/OutManager.hpp"


namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KO = KOps::Out;

    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
    class MCRunner {
    public:
        // Export these types for cleaner downstream integration or testing
        //using RNGPool      = Kokkos::Random_XorShift64_Pool<>;
        //using PathView     = Kokkos::View<double**, Kokkos::LayoutDefault>;
        //using HostPathView = PathView::HostMirror;


        explicit MCRunner(const KC::UInputs &conf) : config(conf) {
        }

        // ====================================================================
        // LEVEL 3: The Actual Simulation Engine (Fully Resolved at Compile Time)
        // ====================================================================

        void run_mc_simulation() {
            // NOTE : the total number of simulations are performed in batches to handle cases when not enough memory is available

            std::cout << "Starting Monte Carlo Simulation..." << std::endl;

            // 0. Initialize Helper Classes
            KO::OutputManager writer(config.output);
            SDESolver<ModelPolicy, SchemePolicy> solver(config);

            // 1. Allocate GPU View and CPU Mirror (Done ONLY ONCE)
            allocate_memory();

            // 2. Initialize Random Number Generator Pool
            initialize_rng_pool();

            // 3. Run all batches
            run_all_mc_batches();

            std::cout << "Simulation completed successfully!" << std::endl;
        }

    private:
        const KC::UInputs config;

        using DeviceBatchView = Kokkos::View<double **, Kokkos::LayoutDefault>;
        using HostBatchView = DeviceBatchView::HostMirror;

        DeviceBatchView d_batch_view;
        HostBatchView h_batch_view;
        int engine_batch_size;
        int num_steps;

        void allocate_memory() {

            // 1. Determine the layout strategy chosen by the config/auto-tuner
            if (config.mc.batch_size == -1) {
                engine_batch_size = config.mc.N_Paths;
            } else if (config.mc.batch_size == 0) {
                engine_batch_size = determine_optimal_batch_size();
            } else {
                engine_batch_size = config.mc.batch_size;
            }

            // 2. High-Bound Safety: Optimization clamp (No error needed)
            if (engine_batch_size > config.mc.N_Paths) {
                engine_batch_size = config.mc.N_Paths;
            }

            // 3. Low-Bound Safety: Throw a hard exception if the budget is unrunnable
            if (engine_batch_size < 1) {
                throw std::runtime_error(
                    "[Critical Error] The assigned VRAM/RAM hardware budget is too restrictive "
                    "to accommodate even a single Monte Carlo path timeline simulation. "
                    "Please increase your memory budget or increase your time step size (dt)."
                );
            }


            // Allocate the views using our newly stored class attributes
            num_steps = config.time.N_time_steps;
            std::cout << "  [Memory Allocation] Allocating reusable buffers ("
                  << engine_batch_size << " x " << num_steps << ")..." << std::endl;
            d_batch_view = DeviceBatchView("gpu_paths_batch_buffer", engine_batch_size, num_steps);
            h_batch_view = Kokkos::create_mirror_view(d_batch_view);
        }


        [[nodiscard]] int determine_optimal_batch_size() const {
            const double bytes_per_sde_path = config.time.N_time_steps * sizeof(KT::Real);

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
            const double gpu_budget_bytes = static_cast<double>(config.mc.Max_VRAM_MB) * 1024.0 * 1024.0;

            int calculated_paths = static_cast<int>(gpu_budget_bytes / bytes_per_sde_path);

            // Integer division to get multiples of 32 (1 warps or wavefronts)
            // This ensures that every single GPU warp launched is packed with active execution threads, maximizing your hardware saturation
            return (calculated_paths / 32) * 32;


#else
            const double cpu_budget_bytes = static_cast<double>(config.mc.Max_CPU_RAM_MB) * 1024.0 * 1024.0;

            return static_cast<int>(cpu_budget_bytes / bytes_per_sde_path); //Integet division

#endif
        }


        void initialize_rng_pool() {
        }

        void run_all_mc_batches() {
            // Loop over the batches
            run_sims_of_current_batch();
            save_batch_to_disk();
        }

        void run_sims_of_current_batch() {
            // Kokkos Loop over all sims of the batch
            // Call the fd scheme that evolves in time a single sim
        }

        void save_batch_to_disk() {
        }
    };
}
