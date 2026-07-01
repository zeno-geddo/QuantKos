#pragma once

#pragma once

#include <Kokkos_Core.hpp>
#include "../Typedefs.hpp"
#include "../config/Config.hpp"
#include "PathsMCBatchMem.hpp"

namespace KOps::Engine {
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;

    class BackwardLSMMemory {
    public:
        // 1. The standard forward batch memory manager
        PathsMCBatchMem BatchMem;

        // 2. The Massive Master Matrix (EXCLUSIVELY ON CPU RAM)
        // Using LayoutLeft ensures that columns (time-slices) are contiguous in CPU RAM,
        // allowing fast PCIe streaming transfers.
        using HostMasterPathsView = Kokkos::View<KT::Real **, Kokkos::LayoutLeft, Kokkos::HostSpace>;
        HostMasterPathsView h_master_paths;

        // 3. Tiny GPU Buffers for the Backward Pass
        Kokkos::View<KT::Real *, Kokkos::LayoutLeft> d_prices_current_time;
        Kokkos::View<KT::Real *, Kokkos::LayoutLeft> d_best_future_outcomes;
        // Best known future outcome (Current present value of the optimal future strategy)
        // Note : At any given time step t during the loop, d_best_future_outcome(i) contains the amount of money you will make on path i if you hold the option from time t until the best possible future moment, discounted back to time t


        // Host mirror to retrieve the final cash flows
        Kokkos::View<KT::Real *, Kokkos::LayoutLeft>::host_mirror_type h_best_future_outcomes;

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
        BackwardLSMMemory(const BackwardLSMMemory&) = delete;
        BackwardLSMMemory& operator=(const BackwardLSMMemory&) = delete;

        // Default move constructor to allow safe ownership transfer
        BackwardLSMMemory(BackwardLSMMemory&&) = default;
        BackwardLSMMemory& operator=(BackwardLSMMemory&&) = delete;

        // Default destructor
        ~BackwardLSMMemory() = default;

        void copy_current_mc_batch_to_master_mc_matrix(const int batch_idx, const int current_batch_size) {
            // Aim : Extract the exact sub-block of the master matrix we want to fill and fill it
            // Note: master matrix is always layout right, while batchview depends on how the device is being used
            // Note: subview only gives a window, it does allocate any new memory

            // Move data to host using PCIe BUS, it is the slowest copy (cannot copy directly from gpu to cpu)
            BatchMem.deep_copy_to_host();

            // Create Target: CPU Master Matrix subview
            const int path_start_idx = batch_idx * BatchMem.n_sims_per_batch;
            const int paths_end_idx = path_start_idx + current_batch_size;
            const auto target_paths_range = std::make_pair(path_start_idx, paths_end_idx);
            auto h_sub_master = Kokkos::subview(h_master_paths, // target memory to slice
                                                target_paths_range, // target first dimension range (the rows/paths)
                                                Kokkos::ALL() // target all second dimension (all cols/times)
            );

            // Source: CPU Host Batch subview
            auto h_sub_batch = Kokkos::subview(BatchMem.h_batch_view,
                                               std::make_pair(0, current_batch_size),
                                               Kokkos::ALL()); // Handle the case of the last incomplete batch

            // Perform the CPU-to-CPU copy, implicitly making a transpose if necessary
            // Note : This is much faster since it does not requires PCIe BUS but is done in CPU RAM and L3 cache, they are much faster!
            // Doing the second copy should not be a disaster in terms of performance
            Kokkos::deep_copy(h_sub_master, h_sub_batch);
        }

        void bring_host_prices_time_slice_to_device(const int time_step) const {
            // 1. Slice out the target column from the column-major master matrix.
            // Note: Kokkos::ALL() ensures we grab every simulated path for this time step.
            auto h_column_view = Kokkos::subview(h_master_paths, Kokkos::ALL(), time_step);

            // 2. Stream the contiguous memory slice across the PCIe bus to the GPU.
            Kokkos::deep_copy(d_prices_current_time, h_column_view);
        }

        void bring_device_cash_flows_to_host() {
            Kokkos::deep_copy(h_best_future_outcomes, d_best_future_outcomes);
        }
    };
}
