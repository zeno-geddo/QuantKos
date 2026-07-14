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

#pragma once

#include <Kokkos_Core.hpp>
#include "../Typedefs.hpp"
#include "../config/Config.hpp"
#include "PathsMCBatchMem.hpp"

namespace KOps::Engine {
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;

    /**
     * @brief Manages memory for American option pricing when using the Longstaff-Schwartz algorithm.
     * * This class coordinates the large-scale storage of asset paths on the Host (CPU) and
     * provides high-speed streaming buffers to the Device (GPU) for backward regression steps.
     * * @note This manager enforces a heavy memory budget. It pre-allocates a "Master Matrix"
     * in Host RAM to accommodate the entire path history, while providing lightweight
     * streaming buffers for the regression process.
     * * @note This class adheres to RAII principles. Copying is prohibited to prevent
     * memory corruption; move semantics are supported to allow for safe ownership transfer.
     */
    class BackwardLSMMemory {
    public:
        // 1. The standard forward batch memory manager
        PathsMCBatchMem BatchMem; ///< Forward phase batch manager.

        // 2. The Massive Master Matrix (EXCLUSIVELY ON CPU RAM)
        using HostMasterPathsView = Kokkos::View<KT::Real **, Kokkos::LayoutLeft, Kokkos::HostSpace>;
        /** @brief The full asset path matrix (N_paths x T_steps) stored in CPU RAM.
        * @note Using LayoutLeft ensures that columns (time-slices) are contiguous in CPU RAM,
         allowing fast PCIe streaming transfers
        */
        HostMasterPathsView h_master_paths;

        // 3. Tiny GPU Buffers for the Backward Pass
        /** @name GPU Regression Buffers */
        ///@{
        Kokkos::View<KT::Real *, Kokkos::LayoutLeft> d_prices_current_time;
        /** @brief Best known future outcome (Current present value of the optimal future strategy)
         * @note At any given time step t during the loop, d_best_future_outcome(i) contains the amount of money you will make on path i if you hold the option from time t until the best possible future moment, discounted back to time t
         */
        Kokkos::View<KT::Real *, Kokkos::LayoutLeft> d_best_future_outcomes;
        ///@}

        // Host mirror to retrieve the final bets possible outcomes (cash flows)
        Kokkos::View<KT::Real *, Kokkos::LayoutLeft>::host_mirror_type h_best_future_outcomes;

        /**
         * @brief Initializes LSM memory and performs a budget-check against physical RAM.
         * @param conf The global configuration providing path count, time steps, and memory limits.
         * @throw std::runtime_error If the Master Matrix and batch buffers exceed the configured CPU RAM budget.
         */
        explicit BackwardLSMMemory(const KC::UInputs &conf) : BatchMem(conf) {
            const int N = conf.mc.N_Paths;
            const int T = conf.time.N_time_steps;

            // 1. Calculate the RAM needed for the Master Matrix
            const double required_master_mb = static_cast<double>(N) * T * sizeof(KT::Real) / (1024.0 * 1024.0);

            // 2. Add the RAM already claimed by the Host-side batch mirrors
            const double tot_host_batch_mb = BatchMem.bytes_to_mb(BatchMem.tot_host_memory_bytes());
            const double host_paths_batch_mb = BatchMem.bytes_to_mb(BatchMem.host_paths_memory_bytes());
            const double host_payoffs_batch_mb = BatchMem.bytes_to_mb(BatchMem.host_payoffs_memory_bytes());

            // 3. Check if there is enough memory
            const double total_required_cpu_mb = required_master_mb + tot_host_batch_mb;
            const double budget_ram_mb = static_cast<double>(conf.mc.Max_CPU_RAM_MB);

            if (total_required_cpu_mb > budget_ram_mb) {
                throw std::runtime_error(
                    "[Memory Capacity Error] The Longstaff-Schwartz algorithm requires CPU RAM "
                    "that, summed to the batches memory used for the forward computation, exceeds your assigned Max_CPU_RAM_MB budget.\n"
                    "  -> Master Matrix RAM         : " + std::to_string(required_master_mb) + " MB\n"
                    "  -> Host Paths Batch Buffers  : " + std::to_string(host_paths_batch_mb) + " MB\n"
                    "  -> Host Payoff BatchBuffers  : " + std::to_string(host_payoffs_batch_mb) + " MB\n"
                    "  -> Total Required            : " + std::to_string(total_required_cpu_mb) + " MB\n"
                    "  -> Config Budget             : " + std::to_string(budget_ram_mb) + " MB\n"
                    "Possible Actions: \n"
                    "\t 1) Increase 'Max_CPU_RAM_MB' in your config,\n"
                    "\t 2) Reduce the the batch size (explicitly assign a moderate number of sims per batch),\n"
                    "\t 3) Reduce the number of paths/time steps.\n"
                );
            }

            // 4. Allocate memory if there is enough
            std::cout << "  [LSM Memory] Allocating " << N << "x" << T << " Master Matrix in Host RAM..." << std::endl;
            h_master_paths = HostMasterPathsView("HostMasterMatrix", N, T);

            std::cout << "  [LSM Memory] Allocating Temporary Streaming Buffers for Regressions..." << std::endl;
            d_prices_current_time = Kokkos::View<KT::Real *, Kokkos::LayoutLeft>("Device_Time_Slice", N);
            d_best_future_outcomes = Kokkos::View<KT::Real *, Kokkos::LayoutLeft>("Device_Cash_Flows", N);
            h_best_future_outcomes = Kokkos::create_mirror_view(d_best_future_outcomes);
        }

        // Delete copies to prevent shared memory issues
        BackwardLSMMemory(const BackwardLSMMemory &) = delete;

        BackwardLSMMemory &operator=(const BackwardLSMMemory &) = delete;

        // Default move constructor to allow safe ownership transfer
        BackwardLSMMemory(BackwardLSMMemory &&) = default;

        BackwardLSMMemory &operator=(BackwardLSMMemory &&) = delete;

        // Default destructor
        ~BackwardLSMMemory() = default;

        /** @name Data Management Routines */
        ///@{
        /** * @brief Transfers a completed path batch from the GPU to the Master Matrix on the Host.
         * @param batch_idx The index of the current batch.
         * @param current_batch_size The number of paths in this specific batch.
         *  * @note Master matrix is always layout right, while batch-view depends on how the device is being used.
        */
        void copy_current_mc_batch_to_master_mc_matrix(const int batch_idx, const int current_batch_size) {
            // Aim : Extract the exact sub-block of the master matrix we want to fill and fill it
            // Note: master matrix is always layout right, while batch-view depends on how the device is being used
            // Note: subview only gives a window, it does not allocate any new memory

            // 0. Move data to host using PCIe BUS, it is the slowest copy (cannot copy directly from gpu to master matrix in cpu)
            BatchMem.deep_copy_to_host();

            // 1. Create Target: CPU Master Matrix subview
            const int path_start_idx = batch_idx * BatchMem.n_sims_per_batch;
            const int paths_end_idx = path_start_idx + current_batch_size;
            const auto target_paths_range = std::make_pair(path_start_idx, paths_end_idx);
            auto h_sub_master = Kokkos::subview(h_master_paths, // target memory to slice
                                                target_paths_range, // target first dimension range (the rows/paths)
                                                Kokkos::ALL() // target all second dimension (all cols/times)
            );

            // 2. Create Source: CPU Host Batch subview
            auto h_sub_batch = Kokkos::subview(BatchMem.h_batch_view,
                                               std::make_pair(0, current_batch_size),
                                               Kokkos::ALL()); // Handle the case of the last incomplete batch

            // 3. Perform the CPU-to-CPU copy, implicitly making a transpose if necessary
            // Note : This is much faster since it does not require PCIe BUS but is done in CPU RAM and L3 cache, they are much faster!
            // Doing the second copy should not be a disaster in terms of performance
            Kokkos::deep_copy(h_sub_master, h_sub_batch);
        }

        /** * @brief Streams a single time-slice (column) from the Master Matrix to the GPU.
        * @param time_step The specific simulation time step to stream.
        */
        void bring_host_prices_time_slice_to_device(const int time_step) const {
            // 1. Slice out the target column from the column-major master matrix.
            // Note: Kokkos::ALL() ensures we grab every simulated path for this time step.
            auto h_column_view = Kokkos::subview(h_master_paths, Kokkos::ALL(), time_step);

            // 2. Stream the contiguous memory slice across the PCIe bus to the GPU.
            Kokkos::deep_copy(d_prices_current_time, h_column_view);
        }

        /** @brief Synchronizes LSM best future outcomes (cash flows) from Device to Host. */
        void bring_device_cash_flows_to_host() {
            Kokkos::deep_copy(h_best_future_outcomes, d_best_future_outcomes);
        }
        ///@}
    };
}
