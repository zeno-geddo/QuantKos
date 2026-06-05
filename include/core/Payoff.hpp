#pragma once

#include "Typedefs.hpp"
#include "./../IO/ConfigEnums.hpp"

namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;


    // ========================================================================
    // Payoff Evaluator (Resolved at Compile Time)
    // ========================================================================
    template<KI::OptRight OptRight>
    struct Payoff;

    template<>
    struct Payoff<KI::OptRight::Call> {
        KOKKOS_INLINE_FUNCTION static KT::Real evaluate_payoff(const KT::Real S, const KT::Real K) {
            return S > K ? S - K : KT::real_zero;
        }
    };

    template<>
    struct Payoff<KI::OptRight::Put> {
        KOKKOS_INLINE_FUNCTION static KT::Real evaluate_payoff(const KT::Real S, const KT::Real K) {
            return K > S ? K - S : KT::real_zero;
        }
    };

    // ========================================================================
    // PayoffTracker : track reference price and compute payoff
    // ========================================================================
    template<KI::OptType OptType, KI::OptRight OptRight>
    struct PayoffTracker {
        // Registers allocated only if needed; otherwise optimized out by the compiler
        KT::Real S_sum;
        KT::Real S_max;
        KT::Real S_min;

        // Constructor handles initialization based on S0
        KOKKOS_INLINE_FUNCTION PayoffTracker(const KT::Real S0)
            : S_sum(KT::real_zero), S_max(S0), S_min(S0) {
        }

        // Tract the target price (To be called inside the time loop)
        KOKKOS_INLINE_FUNCTION void track_current_price(const KT::Real S) {
            if constexpr (OptType == KI::OptType::Asian) {
                S_sum += S;
            }
            // Group all options that care about the path CEILING
            else if constexpr ( //OptType == KI::OptType::LookbackMax ||
                OptType == KI::OptType::BarrierUpAndOut ||
                OptType == KI::OptType::BarrierUpAndIn) {
                S_max = Kokkos::fmax(S_max, S);
            }
            // Group all options that care about the path FLOOR
            else if constexpr ( //OptType == KI::OptType::LookbackMin ||
                OptType == KI::OptType::BarrierDownAndOut ||
                OptType == KI::OptType::BarrierDownAndIn) {
                S_min = Kokkos::fmin(S_min, S);
            }
        }

        // Final Payoff Resolver (To be called outside the time loop)
        KOKKOS_INLINE_FUNCTION KT::Real evaluate_final_payoff(const KT::Real S,
                                                              const KT::Real Strike,
                                                              const KT::Real BarrierPrice,
                                                              const int n_t_steps) const {
            KT::Real payoff = KT::real_zero;
            // STANDARD OPTIONS LOGIC
            if constexpr (OptType == KI::OptType::European) {
                payoff = Payoff<OptRight>::evaluate_payoff(S, Strike);
            } else if constexpr (OptType == KI::OptType::Asian) {
                const KT::Real avg_price = S_sum / static_cast<KT::Real>(n_t_steps);
                payoff = Payoff<OptRight>::evaluate_payoff(avg_price, Strike);
            }
            // BARRIER OPTIONS LOGIC
            else if constexpr (OptType == KI::OptType::BarrierUpAndOut) {
                if (S_max < BarrierPrice) {
                    // Survived
                    payoff = Payoff<OptRight>::evaluate_payoff(S, Strike);
                }
            } else if constexpr (OptType == KI::OptType::BarrierUpAndIn) {
                if (S_max >= BarrierPrice) {
                    // Activated
                    payoff = Payoff<OptRight>::evaluate_payoff(S, Strike);
                }
            } else if constexpr (OptType == KI::OptType::BarrierDownAndOut) {
                if (S_min > BarrierPrice) {
                    // Survived
                    payoff = Payoff<OptRight>::evaluate_payoff(S, Strike);
                }
            } else if constexpr (OptType == KI::OptType::BarrierDownAndIn) {
                if (S_min <= BarrierPrice) {
                    // Activated
                    payoff = Payoff<OptRight>::evaluate_payoff(S, Strike);
                }
            }

            return payoff;
        }
    };
}
