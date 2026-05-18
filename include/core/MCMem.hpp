#pragma once

#include <Kokkos_Core.hpp>

#include "./Typedefs.hpp"
#include "./../IO/Config.hpp"


namespace KOps::Engine {
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;

    using DevView = Kokkos::View<KT::Real **>; // DefaultExecutionSpace by default
    using HostView = DevView::host_mirror_type;

    // ------------------------------------------------------------------------
    // STATE SDE (Requires 1 copy)
    // ------------------------------------------------------------------------
    class MCBatchMem {
    public:
        DevView d_batch_view; // Device Views (GPU)
        HostView h_batch_view; // Host Mirrors (CPU)
        int n_sims_per_batch;

        // Explicit to initialize it explicitly
        explicit MCBatchMem(const KC::UInputs &conf) : config(conf) {
            // Memory is allocated during the construction of the class
            allocate_batch_memory();
        }

        // Full-View Synchronizers
        void deep_copy_to_host() const { Kokkos::deep_copy(h_batch_view, d_batch_view); }
        void deep_copy_to_device() const { Kokkos::deep_copy(d_batch_view, h_batch_view); }

        // ABSTRACT OVERLOAD of Full-View Synchronizers: Slices and syncs only the active rows
        void deep_copy_to_host(int n_active_paths) const {
            // If asked for the full size, forward them to the faster full-view copy automatically!
            if (n_active_paths == n_sims_per_batch) {
                deep_copy_to_host();
                return;
            }

            // Targeted slice sync for remainder edge cases
            auto d_sub = Kokkos::subview(d_batch_view, Kokkos::pair<int, int>(0, n_active_paths), Kokkos::ALL);
            auto h_sub = Kokkos::subview(h_batch_view, Kokkos::pair<int, int>(0, n_active_paths), Kokkos::ALL);
            Kokkos::deep_copy(h_sub, d_sub);
        }

        void deep_copy_to_device(int n_active_paths) const {
            // If asked for the full size, forward them to the faster full-view copy automatically!
            if (n_active_paths == n_sims_per_batch) {
                deep_copy_to_device();
                return;
            }

            // Targeted slice sync for remainder edge cases
            auto d_sub = Kokkos::subview(d_batch_view, Kokkos::pair<int, int>(0, n_active_paths), Kokkos::ALL);
            auto h_sub = Kokkos::subview(h_batch_view, Kokkos::pair<int, int>(0, n_active_paths), Kokkos::ALL);
            Kokkos::deep_copy(d_sub, h_sub);
        }

        // Get allocated memory size
        [[nodiscard]] size_t device_memory_bytes() const {
            // span: distance between lowest and highest address, must be contiguous memory to work
            return d_batch_view.span() * sizeof(KT::Real);
        }

        [[nodiscard]] size_t host_memory_bytes() const {
            // Only count host memory if it's a true separate physical allocation (GPU builds)
            if (h_batch_view.data() != nullptr && h_batch_view.data() != d_batch_view.data()) {
                // span: distance between lowest and highest address, must be contiguous memory to work
                return h_batch_view.span() * sizeof(KT::Real);
            }
            return 0; // 0 duplicate bytes allocated if running natively on a host CPU
        }

    private:
        const KC::UInputs config;


        void allocate_batch_memory() {
            // 1. Determine the layout strategy chosen by the config/auto-tuner
            if (config.mc.batch_size == -1) {
                n_sims_per_batch = config.mc.N_Paths;
            } else if (config.mc.batch_size == 0) {
                n_sims_per_batch = determine_optimal_batch_size();
            } else {
                n_sims_per_batch = config.mc.batch_size;
            }

            // 2. High-Bound Safety: Optimization clamp (No error needed)
            if (n_sims_per_batch > config.mc.N_Paths) {
                n_sims_per_batch = config.mc.N_Paths;
            }

            // 3. Low-Bound Safety: Throw a hard exception if the budget is unrunnable
            if (n_sims_per_batch < 1) {
                throw std::runtime_error(
                    "[Critical Error] The assigned VRAM/RAM hardware budget is too restrictive "
                    "to accommodate even a single Monte Carlo path timeline simulation. "
                    "Please increase your memory budget or increase your time step size (dt)."
                );
            }


            // Allocate the views using our newly stored class attributes
            std::cout << "  [Memory Allocation] Allocating reusable buffers ("
                  << n_sims_per_batch << " x " << config.time.N_time_steps << ")..." << std::endl;
            d_batch_view = DevView("gpu_paths_batch_buffer", n_sims_per_batch, config.time.N_time_steps);
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
    };
}
