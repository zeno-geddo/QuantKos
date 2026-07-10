#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>


#include "../Typedefs.hpp"
#include "../config/Config.hpp"
#include "../config/ConfigFileEnums.hpp"
#include "./MemoryTypes.hpp"


namespace KOps::Engine {
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;
    namespace KI = KOps::Implemented;


    // ------------------------------------------------------------------------
    // Class managing the memory, when the entire times grid is kept
    // but only a subsets of the total paths are kept to not saturate memory
    // ------------------------------------------------------------------------
    /**
     * @brief Manages hardware memory allocation for batch-processed Monte Carlo simulations.
     * * This class implements a memory-constrained batching strategy to perform large-scale
     * path simulations without saturating the device memory. It calculates an optimal
     * batch size based on user-provided VRAM/RAM constraints and allocates contiguous
     * buffers for asset paths and their resulting payoffs.
     * * @note This class adheres to RAII principles. Copying is prohibited to prevent
     * memory corruption; move semantics are supported to allow for safe ownership transfer.
     * * @note This class utilizes Kokkos mirror views. The h_batch_view will point to the same memory as d_batch_view when executing on a CPU-only architecture, resulting in zero-cost synchronization
    */
    class PathsMCBatchMem {
    public:
        int n_sims_per_batch; ///< Number of paths processed in a single batch (kernel launch).

        /** @name Kokkos Device Views */
        ///@{
        DevBatchPathsView d_batch_view; ///< Device-side view for asset paths (N_sims_per_batch, TotN_T_steps).
        DevBatchPayoffView d_payoffs; ///< Device-side view for option payoffs (N_sims_per_batch).
        ///@}

        /** @name Kokkos Host Views */
        ///@{
        HostBatchPathsView h_batch_view; ///< Host-side mirror for asset paths synchronization.
        HostBatchPayoffView h_payoffs; ///< Host-side mirror for payoff data synchronization.
        ///@}

        /**
        * @brief Constructs the manager and allocates hardware-aligned memory buffers.
        * (Must be initialized explicitly)
        * @param conf reference to master configuration, containing also memory budget and path counts.
        */
        explicit PathsMCBatchMem(const KC::UInputs &conf) : config(conf) {
            // Memory is allocated during the construction of the class
            allocate_batch_memory();
            std::cout << "  [Batch Memory] Memory allocated correctly." << std::endl;
        }

        /**
         * @brief Copy constructor is explicitly deleted.
         * * Duplicate views tracking the same active GPU memory addresses would result in
         * multiple host-device synchronization conflicts and race conditions. To enforce a single
         * source of truth for the active batch memory, copying is prohibited.
         */
        PathsMCBatchMem(const PathsMCBatchMem &) = delete;

        /**
         * @brief Copy assignment operator is explicitly deleted.
         */
        PathsMCBatchMem &operator=(const PathsMCBatchMem &) = delete;

        /**
         * @brief Default move constructor to safely transfer memory ownership across execution scopes if needed.
         * * Transfers ownership of the underlying Kokkos allocation handles and view pointers from an expiring
         * instance to a newly initialized object. This operation is virtually instantaneous and occurs
         * without executing any expensive allocations or memory copies in CPU or GPU space.
        */
        PathsMCBatchMem(PathsMCBatchMem &&) = default;

        /**
         * @brief Move assignment operator is explicitly deleted.
         * * Re-assigning an already allocated batch memory layout dynamically after its initial setup is
         * prohibited to protect active CUDA/HIP streams or OpenMP parallel loops from reference changes mid-run.
         */
        PathsMCBatchMem &operator=(PathsMCBatchMem &&) = delete;

        /**
         * @brief Default destructor.
         * * Because this class utilizes reference-counted Kokkos Views, memory cleanup is automated.
         * When the internal reference counters of the device and host view allocations drop to zero
         * upon destruction, the underlying physical memory pools (VRAM / RAM) are safely reclaimed.
         */
        ~PathsMCBatchMem() = default;

        //-------------------------------------------
        // SYNCH
        //-------------------------------------------
        /** @name Synchronization Routines */
        ///@{
        /** @brief Copies current batch data from the device to the host. */
        void deep_copy_to_host() const {
            Kokkos::deep_copy(h_batch_view, d_batch_view);
            Kokkos::deep_copy(h_payoffs, d_payoffs);
        }

        /** @brief Copies modified batch data from the host back to the device. */
        void deep_copy_to_device() const {
            Kokkos::deep_copy(d_batch_view, h_batch_view);
            Kokkos::deep_copy(d_payoffs, h_payoffs);
        }

        ///@}

        //-------------------------------------------
        // MEMORY INFO
        //-------------------------------------------
        /** @name Memory Interrogation Utilities */
        ///@{

        /**
         * @brief Returns the name of the active default hardware execution space (e.g., Cuda, HIP, OpenMP).
         */
        [[nodiscard]] std::string execution_space_name() const {
            return Kokkos::DefaultExecutionSpace::name();
        }

        /**
         * @brief Calculates the total physical size in bytes of the device-side asset batch path View.
         */
        [[nodiscard]] size_t device_paths_memory_bytes() const {
            // span: distance between lowest and highest address, must be contiguous memory to work
            return d_batch_view.span() * sizeof(KT::Real);
        }

        /**
         * * @brief Calculates the total physical size in bytes of the device-side path batch payoff View.
         */
        [[nodiscard]] size_t device_payoffs_memory_bytes() const {
            // span: distance between lowest and highest address, must be contiguous memory to work
            return d_payoffs.span() * sizeof(KT::Real);
        }

        /**
         * @brief Returns the combined device-side VRAM footprint (Paths + Payoffs) allocated for this batch.
         */
        [[nodiscard]] size_t tot_device_memory_bytes() const {
            // span: distance between lowest and highest address, must be contiguous memory to work
            return device_paths_memory_bytes() + device_payoffs_memory_bytes();
        }

        /** @brief Calculates the host-side RAM size in bytes used for path mirroring.
         * @note Returns 0 if executing on CPU-only builds where device and host share the same pointers.
         */
        [[nodiscard]] size_t host_paths_memory_bytes() const {
            // Only count host memory if it's a true separate physical allocation (GPU builds)
            if (h_batch_view.data() != nullptr && h_batch_view.data() != d_batch_view.data()) {
                // span: distance between lowest and highest address, must be contiguous memory to work
                return h_batch_view.span() * sizeof(KT::Real);
            }
            return 0; // 0 duplicate bytes allocated if running natively on a host CPU
        }

        /** * @brief Calculates the host-side RAM size in bytes used for payoffs mirroring.
         * @note Returns 0 if executing on CPU-only builds where device and host share the same pointers.
         */
        [[nodiscard]] size_t host_payoffs_memory_bytes() const {
            // Only count host memory if it's a true separate physical allocation (GPU builds)
            if (h_payoffs.data() != nullptr && h_payoffs.data() != h_payoffs.data()) {
                // span: distance between lowest and highest address, must be contiguous memory to work
                return h_payoffs.span() * sizeof(KT::Real);
            }
            return 0; // 0 duplicate bytes allocated if running natively on a host CPU
        }

        /** * @brief Returns the combined host-side RAM footprint (Paths + Payoffs) allocated for mirroring.
         */
        [[nodiscard]] size_t tot_host_memory_bytes() const {
            return host_paths_memory_bytes() + host_payoffs_memory_bytes();
        }

        /** * @brief Utility helper to convert raw byte sizes into Megabytes (MB).
         */
        [[nodiscard]] double bytes_to_mb(size_t n_bytes) const {
            return static_cast<double>(n_bytes) / (1024.0 * 1024.0);
        }

        /** * @brief Estimates the full, unrolled paths footprint in MB if all simulation steps were kept in RAM at once.
         */
        [[nodiscard]] double total_paths_footprint_mb() const {
            const size_t total_bytes = static_cast<size_t>(config.mc.N_Paths) * static_cast<size_t>(config.time.
                                           N_time_steps)
                                       * sizeof(KT::Real);
            return static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        }

        /** * @brief Calculates the total footprint in MB needed to hold the final payoffs across all simulation paths.
         */
        [[nodiscard]] double total_payoffs_footprint_mb() const {
            size_t total_bytes = static_cast<size_t>(config.mc.N_Paths) * sizeof(KT::Real);
            return static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        }

        // Give host layout
        /** * @brief Compile-time check verifying if the Host View layout is Row-Major (LayoutRight, or C-Style).
         */
        [[nodiscard]] bool is_host_row_major() const {
            // Check if the layout is Row-Major (C-Style)
            // Evaluates completely at compile-time. The compiler will optimize this
            // to a hardcoded 'return true;' or 'return false;' in the binary.
            return std::is_same_v<
                typename decltype(h_batch_view)::array_layout,
                Kokkos::LayoutRight // right indices are contiguous
            >;
        }

        /** * @brief Compile-time check verifying if the Host View layout is Column-Major (LayoutLeft, or Fortran-Style).
         */
        [[nodiscard]] bool is_host_col_major() const {
            // Check if the layout is Col-Major (Fortran-Style)
            // Evaluates completely at compile-time. The compiler will optimize this
            // to a hardcoded 'return true;' or 'return false;' in the binary.
            return std::is_same_v<
                typename decltype(h_batch_view)::array_layout,
                Kokkos::LayoutLeft // left indices are contiguous
            >;
        }

        /** * @brief Dynamically audits if the host memory allocation slice is physically contiguous in memory.
         */
        [[nodiscard]] bool is_host_contiguous() const {
            // Dynamically checks the actual memory slice, catching non-contiguous subviews or LayoutStride edge cases.
            return h_batch_view.span_is_contiguous();
        }

        ///@}


        //-------------------------------------------
        // BATCH INFO
        //-------------------------------------------
        /** @name Batch Info */
        ///@{

        /** * @brief Returns the total count of batch iterations required to complete the entire simulation run.
         */
        [[nodiscard]] int total_batch_loops() const {
            const int n_batches = n_full_batches();
            const int left_over = n_sims_left_over_after_full_batches();
            return n_batches + (left_over > 0 ? 1 : 0);
        }

        /** * @brief Returns the total number of perfectly sized full batches to process during the entire MonteCarlo.
         */
        [[nodiscard]] int n_full_batches() const {
            return config.mc.N_Paths / n_sims_per_batch;
        }

        /** * @brief Returns the path remainder count that must be scheduled in a final, smaller partial batch.
         */
        [[nodiscard]] int n_sims_left_over_after_full_batches() const {
            return config.mc.N_Paths % n_sims_per_batch;
        }

        /** * @brief Resolves the exact path count for a target batch index, safely adjusting for smaller leftover batches.
         * @param batch_idx The active zero-indexed batch sequence counter.
         */
        [[nodiscard]] int get_curr_batch_size(const int batch_idx) const {
            const int n_full_batches = config.mc.N_Paths / n_sims_per_batch;
            if (batch_idx < n_full_batches) {
                return n_sims_per_batch;
            }
            return config.mc.N_Paths % n_sims_per_batch;
        }

        ///@}

    private:
        const KC::UInputs& config; ///< Local reference mapping back to the master input settings.

        /** * @brief Allocates and mirrors device/host views based on hardware budget.
        */
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
            std::cout << "  [Batch Memory] Allocating reusable buffers ("
                    << n_sims_per_batch << " x " << config.time.N_time_steps << ") for the paths of the MC batches..."
                    << std::endl;
            d_batch_view = DevBatchPathsView("gpu_paths_batch_buffer", n_sims_per_batch, config.time.N_time_steps);
            h_batch_view = Kokkos::create_mirror_view(d_batch_view);

            // Allocate the PAYOFFS views
            std::cout << "  [Batch Memory] Allocating reusable buffers ("
                    << n_sims_per_batch << ") for the payoffs of the MC batches..." << std::endl;
            d_payoffs = DevBatchPayoffView("gpu_payoffs_batch_buffer", n_sims_per_batch);
            h_payoffs = Kokkos::create_mirror_view(d_payoffs);
        }

        /** * @brief Computes optimal batch size based on available device VRAM or host RAM.
         * * Applies architectural alignment (multiple of 32) to ensure thread-warp
         * execution efficiency on GPU backends.
         */
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
            double cpu_budget_bytes = static_cast<double>(config.mc.Max_CPU_RAM_MB) * 1024.0 * 1024.0;

            // If you need to allocate a master matrix for backward paths on cpu, allocate this first
            if (config.options.opt_type == KI::OptType::American) {
                const double master_matrix_bytes = static_cast<double>(config.mc.N_Paths) * bytes_per_sde_path;
                cpu_budget_bytes -= master_matrix_bytes;

                // If the Master Matrix alone eats the entire budget, return 0.
                // The Low-Bound Safety check in allocate_batch_memory() will catch this and throw cleanly.
                if (cpu_budget_bytes <= 0.0) {
                    throw std::runtime_error(
                        "[Memory Capacity Error] The Longstaff-Schwartz algorithm requires CPU RAM "
                        "that, summed to the batches memory used for the forward computation, exceeds your assigned Max_CPU_RAM_MB budget.\n"
                        "  -> Master Matrix RAM   : " + std::to_string(master_matrix_bytes) + " MB\n"
                        "  -> Host Batch Buffers  : " + std::to_string(cpu_budget_bytes) + " MB\n"
                        "  -> Config Budget       : " + std::to_string(config.mc.Max_CPU_RAM_MB) + " MB\n"
                        "Possible Actions: \n"
                        "\t 1) Increase 'Max_CPU_RAM_MB' in your config,\n"
                        "\t 2) Reduce the the batch size (explicitly assign a moderate number of sims per batch),\n"
                        "\t 3) Reduce the number of paths/time steps.\n"
                    );
                }
            }

            return static_cast<int>(cpu_budget_bytes / bytes_per_sde_path); //Integer division

#endif
        }
    };
}
