#pragma once

#pragma once

#include <Kokkos_Core.hpp>
#include "../../Typedefs.hpp"
#include "../../config/Config.hpp"
#include "PathsMCBatchMem.hpp"

namespace KOps::Engine {
    namespace KT = KOps::Types;
    namespace KC = KOps::Config;

    class LSMMemory {
    public:
        // 1. The standard forward batch memory manager
        PathsMCBatchMem BatchMem;

        // 2. The Massive Master Matrix (EXCLUSIVELY ON CPU RAM)
        // Using LayoutLeft ensures that columns (time-slices) are contiguous in CPU RAM,
        // allowing fast PCIe streaming transfers.
        using HostMasterPathsView = Kokkos::View<KT::Real**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
        HostMasterPathsView h_master_paths;

        // 3. Tiny GPU Buffers for the Backward Pass
        Kokkos::View<KT::Real*, Kokkos::LayoutLeft> d_time_slice;
        Kokkos::View<KT::Real*, Kokkos::LayoutLeft> d_cash_flows;

        // Host mirror to retrieve the final cash flows
        Kokkos::View<KT::Real*, Kokkos::LayoutLeft>::host_mirror_type h_cash_flows;

        explicit LSMMemory(const KC::UInputs &conf) : BatchMem(conf) {
            const int N = conf.mc.N_Paths;
            const int T = conf.time.N_time_steps;

            std::cout << "  [LSMMemory] Allocating " << N << "x" << T << " Master Matrix in Host RAM..." << std::endl;
            h_master_paths = HostMasterPathsView("HostMasterMatrix", N, T);

            std::cout << "  [LSMMemory] Allocating PCIe Streaming GPU Buffers..." << std::endl;
            d_time_slice = Kokkos::View<KT::Real*, Kokkos::LayoutLeft>("Device_Time_Slice", N);
            d_cash_flows = Kokkos::View<KT::Real*, Kokkos::LayoutLeft>("Device_Cash_Flows", N);
            h_cash_flows = Kokkos::create_mirror_view(d_cash_flows);
        }
    };
}