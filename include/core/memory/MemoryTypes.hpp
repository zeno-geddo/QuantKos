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

#include <Kokkos_Core.hpp>

#include "../Typedefs.hpp"


namespace KOps::Engine {
    namespace KT = KOps::Types;

    /**
     * @brief 2D device View allocated on the default execution space to track asset price paths.
     * * Represents the price matrix ($N_{\text{paths_x_batch}} \times N_{\text{time\_steps}}$).
    * * ### Layout Adaptation by Hardware Space:
     * - **On GPUs (e.g., CUDA, HIP)**: Kokkos defaults this View to a Column-Major layout (@c LayoutLeft).
     * This ensures memory-coalesced writes as parallel thread lanes process different paths and write to
     * VRAM together in lockstep (warps).
     * - **On CPUs (e.g., OpenMP, Threads, Serial)**: Kokkos automatically switches the default to a Row-Major
     * layout (@c LayoutRight). This aligns sequential memory access with CPU cache lines (L1/L2/L3) and
     * enables compilers to perform auto-vectorization (SIMD), maximizing CPU-level cache locality.
     */
    using DevBatchPathsView = Kokkos::View<KT::Real **>; // DefaultExecutionSpace by default

    /**
     * @brief Host memory mirror matching the exact layout and dimensions of @c DevPathsView.
     * @note Kokkos's @c `host_mirror_type` explicitly forces the host view to share the exact same memory layout (e.g., @c LayoutLeft) as its device counterpart.
     * This layout symmetry is a strict hardware requirement. It allows the PCIe controller to perform high-speed,
     * contiguous Direct Memory Access (DMA) transfers during @c Kokkos::deep_copy() operations without requiring
     * expensive, on-the-fly layout transpositions on either the host or device processors.
     */
    using HostBatchPathsView = DevBatchPathsView::host_mirror_type;

    /**
     * @brief 1D device View allocated to store option payoffs at the batch level.
     * * Holds the calculated option payoff at maturity or early exercise boundaries.
     */
    using DevBatchPayoffView  = Kokkos::View<KT::Real *>;

    /**
     * @brief Pinned host memory mirror matching the dimensions of @c DevPayoffView.
     * * Used to safely pull completed payoff results from the device memory space back to the CPU
     * for statistical evaluation and reporting.
     */
    using HostBatchPayoffView = DevBatchPayoffView::host_mirror_type;

}