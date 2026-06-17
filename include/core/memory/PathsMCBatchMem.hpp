#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>


#include "../Typedefs.hpp"
#include "../config/Config.hpp"
#include "./MemoryTypes.hpp"


namespace KOps::Engine {
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;


    // ------------------------------------------------------------------------
    // Class managing the memory for markovian models, when the entire times grid is kept
    // but only a subsets of the total paths are kept to not saturate memory
    // ------------------------------------------------------------------------
    class PathsMCBatchMem {
    public:
        int n_sims_per_batch;

        // Device Views (N_sims_per_batch, TotN_T_steps)
        DevPathsView d_batch_view;
        DevPayoffView d_payoffs;

        // Host Views
        HostPathsView h_batch_view;
        HostPayoffView h_payoffs;



        // Explicit to initialize it explicitly
        explicit PathsMCBatchMem(const KC::UInputs &conf) : config(conf) {
            // Memory is allocated during the construction of the class
            allocate_batch_memory();
            std::cout << "  [MCBatchMem] Memory allocated correctly." << std::endl;

        }

        //-------------------------------------------
        // SYNCH
        //-------------------------------------------

        // Full-View Synchronizers
        void deep_copy_to_host() const {
            Kokkos::deep_copy(h_batch_view, d_batch_view);
            Kokkos::deep_copy(h_payoffs, d_payoffs);
        }
        void deep_copy_to_device() const {
            Kokkos::deep_copy(d_batch_view, h_batch_view);
            Kokkos::deep_copy(d_payoffs, h_payoffs);
        }

        //-------------------------------------------
        // MEMORY INFO
        //-------------------------------------------

        // Hardware Interrogation
        [[nodiscard]] std::string execution_space_name() const {
            return Kokkos::DefaultExecutionSpace::name();
        }

        // Get allocated memory size
        [[nodiscard]] size_t device_paths_memory_bytes() const {
            // span: distance between lowest and highest address, must be contiguous memory to work
            return d_batch_view.span() * sizeof(KT::Real);
        }

        [[nodiscard]] size_t device_payoffs_memory_bytes() const {
            // span: distance between lowest and highest address, must be contiguous memory to work
            return d_payoffs.span() * sizeof(KT::Real);
        }

        [[nodiscard]] size_t tot_device_memory_bytes() const {
            // span: distance between lowest and highest address, must be contiguous memory to work
            return device_paths_memory_bytes() + device_payoffs_memory_bytes();
        }

        [[nodiscard]] size_t host_paths_memory_bytes() const {
            // Only count host memory if it's a true separate physical allocation (GPU builds)
            if (h_batch_view.data() != nullptr && h_batch_view.data() != d_batch_view.data()) {
                // span: distance between lowest and highest address, must be contiguous memory to work
                return h_batch_view.span() * sizeof(KT::Real);
            }
            return 0; // 0 duplicate bytes allocated if running natively on a host CPU
        }

        [[nodiscard]] size_t host_payoffs_memory_bytes() const {
            // Only count host memory if it's a true separate physical allocation (GPU builds)
            if (h_payoffs.data() != nullptr && h_payoffs.data() != h_payoffs.data()) {
                // span: distance between lowest and highest address, must be contiguous memory to work
                return h_payoffs.span() * sizeof(KT::Real);
            }
            return 0; // 0 duplicate bytes allocated if running natively on a host CPU
        }

        [[nodiscard]] size_t tot_host_memory_bytes() const {
            return host_paths_memory_bytes() + host_payoffs_memory_bytes();
        }

        [[nodiscard]] double bytes_to_mb(size_t n_bytes) const {
            return static_cast<double>(n_bytes) / (1024.0 * 1024.0);
        }

        [[nodiscard]] double total_paths_footprint_mb() const {
            size_t total_bytes = static_cast<size_t>(config.mc.N_Paths) * static_cast<size_t>(config.time.N_time_steps) * sizeof(KT::Real);
            return static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        }

        [[nodiscard]] double total_payoffs_footprint_mb() const {
            size_t total_bytes = static_cast<size_t>(config.mc.N_Paths) * sizeof(KT::Real);
            return static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        }

        // Give host layout
        [[nodiscard]] bool is_host_row_major() const {
            // Check if the layout is Row-Major (C-Style)
            // Evaluates completely at compile-time. The compiler will optimize this
            // to a hardcoded 'return true;' or 'return false;' in the binary.
            return std::is_same_v<
                typename decltype(h_batch_view)::array_layout,
                Kokkos::LayoutRight // right indices are contiguous
            >;
        }

        [[nodiscard]] bool is_host_col_major() const {
            // Check if the layout is Col-Major (Fortran-Style)
            // Evaluates completely at compile-time. The compiler will optimize this
            // to a hardcoded 'return true;' or 'return false;' in the binary.
            return std::is_same_v<
                typename decltype(h_batch_view)::array_layout,
                Kokkos::LayoutLeft // left indices are contiguous
            >;
        }

        // Check if the memory block is physically contiguous
        [[nodiscard]] bool is_host_contiguous() const {
            // Dynamically checks the actual memory slice, catching non-contiguous subviews or LayoutStride edge cases.
            return h_batch_view.span_is_contiguous();
        }

        //-------------------------------------------
        // BATCH INFO
        //-------------------------------------------

        [[nodiscard]] int total_batch_loops() const {
            const int n_batches = n_full_batches();
            const int left_over   = n_sims_left_over_after_full_batches();
            return n_batches + (left_over > 0 ? 1 : 0);
        }

        [[nodiscard]] int n_full_batches() const {
            return config.mc.N_Paths / n_sims_per_batch;
        }

        [[nodiscard]] int n_sims_left_over_after_full_batches() const {
            return config.mc.N_Paths % n_sims_per_batch;
        }

        [[nodiscard]] int get_curr_batch_size(int batch_idx) const {
            const int n_full_batches = config.mc.N_Paths / n_sims_per_batch;
            if (batch_idx < n_full_batches) {
                return n_sims_per_batch;
            }
            return config.mc.N_Paths % n_sims_per_batch;
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


            // Allocate the PATHS views using our newly stored class attributes
            std::cout << "  [Memory Allocation] Allocating reusable buffers ("
                    << n_sims_per_batch << " x " << config.time.N_time_steps << ") for the paths of the MC batches..." << std::endl;
            d_batch_view = DevPathsView("gpu_paths_batch_buffer", n_sims_per_batch, config.time.N_time_steps);
            h_batch_view = Kokkos::create_mirror_view(d_batch_view);

            // Allocate the PAYOFFS views
            std::cout << "  [Memory Allocation] Allocating reusable buffers ("
                   << n_sims_per_batch << ") for the payoffs of the MC batches..." << std::endl;
            d_payoffs = DevPayoffView("gpu_payoffs_batch_buffer", n_sims_per_batch);
            h_payoffs = Kokkos::create_mirror_view(d_payoffs);
        }


        [[nodiscard]] int determine_optimal_batch_size() const {
            const double bytes_per_sde_path = config.time.N_time_steps * sizeof(KT::Real);

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
            const double gpu_budget_bytes = static_cast<double>(config.mc.Max_VRAM_MB) * 1024.0 * 1024.0;

            const int calculated_paths = static_cast<int>(gpu_budget_bytes / bytes_per_sde_path);

            // Integer division to get multiples of 32 (1 warps or wavefronts)
            // First, it ensures that every single GPU warp launched is packed with active execution threads (warp is 100% full)
            // Second, it aligns data boundaries with the physical architecture of the GPU chip
            // (data are read by 32 blocks, so you just make single memory transaction to read data for a single warp)
            return (calculated_paths / 32) * 32;

#else
            const double cpu_budget_bytes = static_cast<double>(config.mc.Max_CPU_RAM_MB) * 1024.0 * 1024.0;

            return static_cast<int>(cpu_budget_bytes / bytes_per_sde_path); //Integer division

#endif
        }
    };
}
