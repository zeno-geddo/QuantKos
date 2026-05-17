#pragma once

#include <string>
#include <Kokkos_Core.hpp>

#include "./Typedefs.hpp"
#include "./../IO/Config.hpp"


namespace KOps::Engine {
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;

    using DevView = Kokkos::View<KT::Real **>; // DefaultExecutionSpace by default
    using HostView = DevView::HostMirror;

    // ------------------------------------------------------------------------
    // STATE SDE (Requires 1 copy)
    // ------------------------------------------------------------------------
    class BatchState {
    public:
        DevView d_batch_view; // Device Views (GPU)
        HostView h_batch_view; // Host Mirrors (CPU)
        int engine_batch_size;
        int num_steps;


        explicit BatchState(const KC::UInputs &conf) : config(conf) {
            // Memory is allocated during the construction of the class
            allocate_batch_memory();
        }


        void deep_copy_to_host() const { Kokkos::deep_copy(h_batch_view, d_batch_view); }
        void deep_copy_to_device() const { Kokkos::deep_copy(d_batch_view, h_batch_view); }


        [[nodiscard]] size_t device_memory_bytes() const {
            size_t total = 0;

            auto add_dev_mem = [&total](const DevView &dev) {
                // use inline lambdas
                total += dev.required_allocation_size(dev.extent(0), dev.extent(1));
            };

            add_dev_mem(d_batch_view);

            return total;
        }

        [[nodiscard]] size_t host_memory_bytes() const {
            size_t total = 0;

            // Define a quick inline lambda to handle the shallow copy check
            auto add_host_mem = [&total](const DevView &dev, const HostView &host) {
                if (host.data() != nullptr && host.data() != dev.data()) {
                    total += host.required_allocation_size(host.extent(0), host.extent(1));
                }
            };

            add_host_mem(d_batch_view, h_batch_view);

            return total;
        }

    private:
        const KC::UInputs config;


        void allocate_batch_memory() {
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
            d_batch_view = DevView("gpu_paths_batch_buffer", engine_batch_size, num_steps);
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
