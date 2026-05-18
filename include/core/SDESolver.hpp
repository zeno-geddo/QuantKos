#pragma once
#include <Kokkos_Core.hpp>

#include "./../IO/ConfigEnums.hpp"
#include "SDESchemes.hpp"
#include "Typedefs.hpp"

namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;

    // Placed inside namespace KOps::Engine
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
    struct EvolveSDE {
        // 1. The Execution Context (The data the GPU needs)
        DevView local_batch_view;
        typename RNGenerator::PoolType local_pool;

        int n_t_steps;
        KT::Real S0;
        KT::Real v0;

        // The trivially copyable mathematical solver
        SDESchemes<ModelPolicy, SchemePolicy> scheme;

        // 2. The Execution Operator (Replaces KOKKOS_LAMBDA)
        KOKKOS_INLINE_FUNCTION
        void operator()(const int n_p) const {
            auto rn_generator = local_pool.get_state();

            KT::Real S = S0;
            KT::Real v = v0;

            for (int n_t = 0; n_t < n_t_steps; ++n_t) {
                auto [next_S, next_v] = scheme.evolve_step(S, v, rn_generator);

                S = next_S;
                v = next_v;

                local_batch_view(n_p, n_t) = S;
            }
            local_pool.free_state(rn_generator);
        }
    };
}