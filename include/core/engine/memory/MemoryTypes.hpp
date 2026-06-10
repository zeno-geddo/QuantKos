#pragma once

#include <Kokkos_Core.hpp>

#include "../../Typedefs.hpp"


namespace KOps::Engine {
    namespace KT = KOps::Types;

    using DevPathsView = Kokkos::View<KT::Real **>; // DefaultExecutionSpace by default
    using HostPathsView = DevPathsView::host_mirror_type; // Kokkos forces the host to have the same layout as the device?

    using DevPayoffView  = Kokkos::View<KT::Real *>;
    using HostPayoffView = DevPayoffView::host_mirror_type;

}