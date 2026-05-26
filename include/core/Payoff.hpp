#pragma once

#include "Typedefs.hpp"
#include "./../IO/ConfigEnums.hpp"

namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;


    // Payoff Evaluator (Resolved at Compile Time)
    template<KI::OptRight OptRight>
    struct Payoff;

    template<>
    struct Payoff<KI::OptRight::Call> {
        KOKKOS_INLINE_FUNCTION static KT::Real evaluate(const KT::Real S, const KT::Real K) {
            return S > K ? S - K : KT::real_zero;
        }
    };

    template<>
    struct Payoff<KI::OptRight::Put> {
        KOKKOS_INLINE_FUNCTION static KT::Real evaluate(const KT::Real S, const KT::Real K) {
            return K > S ? K - S : KT::real_zero;
        }
    };
}
