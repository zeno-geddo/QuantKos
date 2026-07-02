#pragma once

#include "../Typedefs.hpp"
#include "../config/ConfigFileEnums.hpp"

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
            else if constexpr (
                // Barrier types
                OptType == KI::OptType::BarrierUpAndOut ||
                OptType == KI::OptType::BarrierUpAndIn ||
                // Lookback types
                (OptType == KI::OptType::LookbackFixedStrike && OptRight == KI::OptRight::Call) ||
                (OptType == KI::OptType::LookbackFloatingStrike && OptRight == KI::OptRight::Put)
            ) {
                S_max = Kokkos::fmax(S_max, S);
            }
            // Group all options that care about the path FLOOR
            else if constexpr (
                // Barrier types
                OptType == KI::OptType::BarrierDownAndOut ||
                OptType == KI::OptType::BarrierDownAndIn ||
                // Lookback types
                (OptType == KI::OptType::LookbackFixedStrike && OptRight == KI::OptRight::Put) ||
                (OptType == KI::OptType::LookbackFloatingStrike && OptRight == KI::OptRight::Call)
            ) {
                S_min = Kokkos::fmin(S_min, S);
            }
            // Fallback, path-independent option not requiring tracking .
            else {
                // European,
                // DigitalCashOrNothing,
                // DigitalAssetOrNothing
            }
        }

        // Final Payoff Resolver (To be called outside the time loop)
        KOKKOS_INLINE_FUNCTION KT::Real evaluate_final_payoff(const KT::Real S,
                                                              const KT::Real Strike,
                                                              const KT::Real BarrierPrice,
                                                              const int n_t_steps) const {
            KT::Real payoff = KT::real_zero;
            // --------------------------------------------------------
            // STANDARD OPTIONS LOGIC
            // --------------------------------------------------------
            if constexpr (OptType == KI::OptType::European) {
                payoff = Payoff<OptRight>::evaluate_payoff(S, Strike);
            } else if constexpr (OptType == KI::OptType::Asian) {
                const KT::Real avg_price = S_sum / static_cast<KT::Real>(n_t_steps);
                payoff = Payoff<OptRight>::evaluate_payoff(avg_price, Strike);
            }
            // --------------------------------------------------------
            // BARRIER OPTIONS LOGIC
            // --------------------------------------------------------
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
            // --------------------------------------------------------
            // LOOKBACK OPTIONS LOGIC
            // --------------------------------------------------------
            else if constexpr (OptType == KI::OptType::LookbackFixedStrike) {
                // Fixed Strike: Substitute S_max or S_min as the Reference Price
                if constexpr (OptRight == KI::OptRight::Call) {
                    payoff = Payoff<OptRight>::evaluate_payoff(S_max, Strike);
                } else {
                    // Put
                    payoff = Payoff<OptRight>::evaluate_payoff(S_min, Strike);
                }
            } else if constexpr (OptType == KI::OptType::LookbackFloatingStrike) {
                // Floating Strike: Reference Price is S, Strike is substituted with S_min or S_max
                if constexpr (OptRight == KI::OptRight::Call) {
                    // Evaluates: max(S - S_min, 0)
                    payoff = Payoff<OptRight>::evaluate_payoff(S, S_min);
                } else {
                    // Put
                    // Evaluates: max(S_max - S, 0)
                    payoff = Payoff<OptRight>::evaluate_payoff(S, S_max);
                }
            }
            // --------------------------------------------------------
            // BINARY OPTIONS LOGIC
            // --------------------------------------------------------
            else if constexpr (OptType == KI::OptType::BinaryCashOrNothing) {
                // Normalized to pay exactly 1.0 unit of cash if In-The-Money
                constexpr  KT::Real cash_payout = 1.0;
                if constexpr (OptRight == KI::OptRight::Call) {
                    payoff = (S > Strike) ? cash_payout : KT::real_zero;
                } else { // Put
                    payoff = (S < Strike) ? cash_payout : KT::real_zero;
                }
            } else if constexpr (OptType == KI::OptType::BinaryAssetOrNothing) {
                // Pays the actual terminal asset price if In-The-Money
                if constexpr (OptRight == KI::OptRight::Call) {
                    payoff = (S > Strike) ? S : KT::real_zero;
                } else { // Put
                    payoff = (S < Strike) ? S : KT::real_zero;
                }
            }


            return payoff;
        }
    };
}
