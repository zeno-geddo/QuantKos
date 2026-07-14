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

// #pragma once
//
// #include <Kokkos_Core.hpp>
// #include <type_traits>
// #include <stdexcept>
// #include <iostream>
// #include <string>
//
// #include "../Typedefs.hpp"
// #include "../config/Config.hpp"
// #include "./MemoryTypes.hpp"
//
// namespace KOps::Engine {
//     namespace KT = KOps::Types;
//     namespace KC = KOps::Config;
//
//     class TimesMCBatchMem {
//     public:
//         int n_times_per_batch;
//
//         // Device Views (Shape: Tot_N_Paths x n_times_per_batch)
//         DevPathsView d_batch_view;
//         DevPayoffView d_payoffs; // Allocated for ALL paths since all paths run simultaneously
//
//         // Host Views
//         HostPathsView h_batch_view;
//         HostPayoffView h_payoffs;
//
//         explicit TimesMCBatchMem(const KC::UInputs &conf) : config(conf) {
//             allocate_batch_memory();
//             std::cout << "  [TimesMCBatchMem] Time-batched memory allocated successfully." << std::endl;
//         }
//
//         //-------------------------------------------
//         // SYNCHRONIZATION
//         //-------------------------------------------
//         void deep_copy_to_host() const {
//             Kokkos::deep_copy(h_batch_view, d_batch_view);
//             Kokkos::deep_copy(h_payoffs, d_payoffs);
//         }
//
//         void deep_copy_to_device() const {
//             Kokkos::deep_copy(d_batch_view, h_batch_view);
//             Kokkos::deep_copy(d_payoffs, h_payoffs);
//         }
//
//         //-------------------------------------------
//         // MEMORY INFO
//         //-------------------------------------------
//         [[nodiscard]] size_t device_paths_memory_bytes() const {
//             return d_batch_view.span() * sizeof(KT::Real);
//         }
//
//         [[nodiscard]] size_t device_payoffs_memory_bytes() const {
//             return d_payoffs.span() * sizeof(KT::Real);
//         }
//
//         [[nodiscard]] size_t tot_device_memory_bytes() const {
//             return device_paths_memory_bytes() + device_payoffs_memory_bytes();
//         }
//
//         [[nodiscard]] size_t host_paths_memory_bytes() const {
//             if (h_batch_view.data() != nullptr && h_batch_view.data() != d_batch_view.data()) {
//                 return h_batch_view.span() * sizeof(KT::Real);
//             }
//             return 0;
//         }
//
//         [[nodiscard]] size_t host_payoffs_memory_bytes() const {
//             // Fixed the self-comparison bug from original snippet
//             if (h_payoffs.data() != nullptr && h_payoffs.data() != d_payoffs.data()) {
//                 return h_payoffs.span() * sizeof(KT::Real);
//             }
//             return 0;
//         }
//
//         [[nodiscard]] double total_paths_footprint_mb() const {
//             size_t total_bytes = static_cast<size_t>(config.mc.N_Paths) * static_cast<size_t>(config.time.N_time_steps) * sizeof(KT::Real);
//             return static_cast<double>(total_bytes) / (1024.0 * 1024.0);
//         }
//
//         //-------------------------------------------
//         // BATCH INFO (Here mapping over Time Axis)
//         //-------------------------------------------
//         [[nodiscard]] int total_batch_loops() const {
//             const int n_batches = n_full_batches();
//             const int left_over   = n_times_left_over_after_full_batches();
//             return n_batches + (left_over > 0 ? 1 : 0);
//         }
//
//         [[nodiscard]] int n_full_batches() const {
//             return config.time.N_time_steps / n_times_per_batch;
//         }
//
//         [[nodiscard]] int n_times_left_over_after_full_batches() const {
//             return config.time.N_time_steps % n_times_per_batch;
//         }
//
//         [[nodiscard]] int get_curr_batch_size(int batch_idx) const {
//             if (batch_idx < n_full_batches()) {
//                 return n_times_per_batch;
//             }
//             return config.time.N_time_steps % n_times_per_batch;
//         }
//
//     private:
//         const KC::UInputs config;
//
//         void allocate_batch_memory() {
//             // Assuming you add a 'time_batch_size' to your config layer
//             // -1 means no batching (allocate all time steps at once)
//             if (config.time.time_batch_size == -1) {
//                 n_times_per_batch = config.time.N_time_steps;
//             } else if (config.time.time_batch_size == 0) {
//                 n_times_per_batch = determine_optimal_batch_size();
//             } else {
//                 n_times_per_batch = config.time.time_batch_size;
//             }
//
//             if (n_times_per_batch > config.time.N_time_steps) {
//                 n_times_per_batch = config.time.N_time_steps;
//             }
//
//             if (n_times_per_batch < 1) {
//                 throw std::runtime_error("[Critical Error] Memory budget too restrictive for a single time step.");
//             }
//
//             // Shape: (N_Paths, n_times_per_batch)
//             std::cout << "  [Memory Allocation] Allocating reusable buffers ("
//                       << config.mc.N_Paths << " x " << n_times_per_batch << ") for time steps..." << std::endl;
//
//             d_batch_view = DevPathsView("gpu_times_batch_buffer", config.mc.N_Paths, n_times_per_batch);
//             h_batch_view = Kokkos::create_mirror_view(d_batch_view);
//
//             // Payoffs allocation remains mapped to total paths
//             d_payoffs = DevPayoffView("gpu_payoffs_total_buffer", config.mc.N_Paths);
//             h_payoffs = Kokkos::create_mirror_view(d_payoffs);
//         }
//
//         [[nodiscard]] int determine_optimal_batch_size() const {
//             // Every single time step in the batch must hold all paths
//             const double bytes_per_time_step = config.mc.N_Paths * sizeof(KT::Real);
//
// #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
//             const double gpu_budget_bytes = static_cast<double>(config.mc.Max_VRAM_MB) * 1024.0 * 1024.0;
//             return static_cast<int>(gpu_budget_bytes / bytes_per_time_step);
// #else
//             const double cpu_budget_bytes = static_cast<double>(config.mc.Max_CPU_RAM_MB) * 1024.0 * 1024.0;
//             return static_cast<int>(cpu_budget_bytes / bytes_per_time_step);
// #endif
//         }
//     };
// }