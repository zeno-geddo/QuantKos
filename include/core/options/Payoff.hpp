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

#include "../Typedefs.hpp"
#include "../config/ConfigFileEnums.hpp"

namespace quantkos::Engine {
    namespace KI = quantkos::Implemented;
    namespace KT = quantkos::Types;


    // ========================================================================
    // Payoff Evaluator (Resolved at Compile Time)
    // ========================================================================
    /**
     * @brief Template trait for compile-time payoff evaluation.
     * * Provides static, device-compatible primitives to calculate terminal option values.
     * Specializations isolate execution paths to eliminate branch instructions within parallel loops.
     * @tparam OptRight The option exercise right contract policy (Call vs. Put).
     */
    template<KI::OptRight OptRight>
    struct Payoff;

    /**
     * @brief Compile-time specialization evaluating standard Call option contract payoffs.
     */
    template<>
    struct Payoff<KI::OptRight::Call> {
        /**
         * @brief Resolves the standard terminal call payoff function.
         * * Mathematically evaluates:
         * $$ Payoff = \max(S - K, 0) $$
         * @param S The final underlying asset spot price (or path reference price).
         * @param K The strike price of the option contract.
         * @return The intrinsic call value, or 0.0 if out-of-the-money.
         */
        KOKKOS_INLINE_FUNCTION static KT::Real evaluate_payoff(const KT::Real S, const KT::Real K) {
            return S > K ? S - K : KT::real_zero;
        }
    };

    /**
     * @brief Compile-time specialization evaluating standard Put option contract payoffs.
     */
    template<>
    struct Payoff<KI::OptRight::Put> {
        /**
         * @brief Resolves the standard terminal put payoff function.
         * * Mathematically evaluates:
         * $$ Payoff = \max(K - S, 0) $$
         * @param S The final underlying asset spot price (or path reference price).
         * @param K The strike price of the option contract.
         * @return The intrinsic put value, or 0.0 if out-of-the-money.
         */
        KOKKOS_INLINE_FUNCTION static KT::Real evaluate_payoff(const KT::Real S, const KT::Real K) {
            return K > S ? K - S : KT::real_zero;
        }
    };

    // ========================================================================
    // PayoffTracker : track reference price and compute payoff
    // ========================================================================
    /**
     * @brief Unified compile-time state accumulator and payoff resolver for path-dependent contracts.
     * * This structure is designed to handle multiple option types (European, Asian, Barrier variants,
     * Lookbacks, and Binaries) inside parallel GPU simulation loops.
     * * @tparam OptType Compile-time option structure classification index.
     * @tparam OptRight Compile-time option execution right (Call or Put).
     */
    template<KI::OptType OptType, KI::OptRight OptRight>
    struct PayoffTracker {
        // Registers allocated only if needed; otherwise optimized out by the compiler
        KT::Real S_sum; ///< Running summation of asset prices along the path (Used for Asian averages).
        KT::Real S_max; ///< Absolute running maximum asset price observed (Used for Upper Barriers / Max Lookbacks)
        KT::Real S_min; ///< Absolute running minimum asset price observed (Used for Lower Barriers / Min Lookbacks).

        // Constructor handles initialization based on S0
        /**
         * @brief Constructs the tracker state, initializing path metrics with the initial spot price.
         * @param S0 Initial spot price of the underlying asset.
         */
        KOKKOS_INLINE_FUNCTION PayoffTracker(const KT::Real S0)
            : S_sum(KT::real_zero), S_max(S0), S_min(S0) {
        }

        // Tract the target price (To be called inside the time loop)
        /**
         * @brief Tracks and updates path-dependent price metrics.
         * @param S The current asset spot price at the active time step.
         * @note This function must be called inside the inner time integration loop. It updates
         * running totals or extremes using optimized device-level functions like `Kokkos::fmax` and `Kokkos::fmin`.
         * @note Branch paths are resolved at compile-time based on the template parameters.
        */
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
        /**
         * @brief Resolves the final option payoff function based on accumulated price statistics.
         * * Evaluates the contract value at the end of the timeline according to its specific financial payoff rules:
         * - **Asian**: Based on the arithmetic path average $\bar{S} = \frac{1}{N} \sum S_t$.
         * - **Barrier**: Evaluates the payoff if the path activation or knock-out conditions are met.
         * - **Lookback (Fixed)**: Substitute extrema price into the standard payoff function.
         * - **Lookback (Floating)**: Evaluates floating strike parameters $\max(S_T - S_{\min}, 0)$ or $\max(S_{\max} - S_T, 0)$.
         * - **Binary**: Returns a fixed cash unit (1.0) or asset value if finishing in-the-money.
         * * @param S Terminal spot price of the underlying asset ($S_T$).
         * @param Strike Target fixed execution strike price ($K$).
         * @param BarrierPrice Trigger boundary price for activation or knock-out conditions ($H$).
         * @param n_t_steps Total discrete time integration steps executed along the timeline.
         * @return The resulting option payoff value.
         */
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
                constexpr KT::Real cash_payout = 1.0;
                if constexpr (OptRight == KI::OptRight::Call) {
                    payoff = (S > Strike) ? cash_payout : KT::real_zero;
                } else {
                    // Put
                    payoff = (S < Strike) ? cash_payout : KT::real_zero;
                }
            } else if constexpr (OptType == KI::OptType::BinaryAssetOrNothing) {
                // Pays the actual terminal asset price if In-The-Money
                if constexpr (OptRight == KI::OptRight::Call) {
                    payoff = (S > Strike) ? S : KT::real_zero;
                } else {
                    // Put
                    payoff = (S < Strike) ? S : KT::real_zero;
                }
            }


            return payoff;
        }
    };
}
