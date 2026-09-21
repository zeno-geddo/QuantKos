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

// File (YAML) → Parser → Validator → Config → Solver
#pragma once


namespace quantkos::Implemented {
    /**
    * @brief Defines the supported option exercise styles and payoff structures.
    */
    enum class OptType {
        // ------------------------------------------------
        // Standard Options (Terminal & Averages)
        // ------------------------------------------------
        /** @name Standard Vanilla Options */
        ///@{
        European, ///< Standard vanilla option exercisable only at maturity.
        Asian, ///< Payoff depends on the average price of the underlying.
        ///@}

        /** @name Barrier Options (Conditional Survival/Activation)*/
        ///@{
        BarrierUpAndOut, ///< Option becomes worthless if the asset rises above the barrier.
        BarrierDownAndOut, ///< Option becomes worthless if the asset falls below the barrier.
        BarrierUpAndIn, ///< Option activates only if the asset rises to touch the barrier.
        BarrierDownAndIn, ///< Option activates only if the asset falls to touch the barrier.
        ///@}

        /** @name Lookback Options */
        ///@{
        LookbackFloatingStrike, ///< Strike price is set to the historical minimum (Call) or maximum (Put).
        LookbackFixedStrike,    ///< Strike is fixed, payoff uses the historical maximum (Call) or minimum (Put).
        ///@}

        /** @name Binary Options */
        ///@{
        BinaryCashOrNothing, ///< Pays a fixed cash amount if the option finishes in-the-money.
        BinaryAssetOrNothing, ///< Pays the value of the terminal asset price if in-the-money.
        ///@}

        /** @name Early Exercise Options */
        ///@{
        American,
        ///@}
    };

    /**
    * @brief Represents the exercise right of the option contract.
    */
    enum class OptRight {
        /** @brief The right to purchase the underlying asset. */
        Call,
        /** @brief The right to sell the underlying asset. */
        Put
    };

    /** @brief Supported stochastic math models for asset dynamics.
     */
    enum class MathModel {
        Heston,
        Bates,
    };

    /** @brief Numerical discretization schemes for SDE paths. */
    enum class NumScheme {
        Euler, ///< Euler-Maruyama discretization.
        ImplicitMilstein, ///< First-order Implicit Milstein scheme.
        AndersonQE, ///< Quadratic Exponential (QE) scheme for Heston.
    };

    /** @brief Supported output data formats. */
    enum class IOFormat {
        BIN, ///< High-performance custom binary protocol.
        TXT, ///< Human-readable text format for debugging/plotting.
    };

    /** @brief Supported verbosity levels. */
    enum class VerbosityLevel : int {
        None     = 0, ///< Nothing is printed.
        Low      = 1, ///< Input message, results.
        Medium   = 2, ///< Input message, setup info, loop progression with timing, results.
        High     = 3, ///< Full diagnostic implemented.
    };

    // Helper to compare VerbosityLevel with integer thresholds
    constexpr bool operator>=(VerbosityLevel level, const int val) noexcept {
        return static_cast<int>(level) >= val;
    }

    constexpr bool operator>=(VerbosityLevel level, VerbosityLevel target) noexcept {
        return static_cast<int>(level) >= static_cast<int>(target);
    }

    /** @brief Supported basis for Longstaff-Schwarz Algorithm (for American Options). */
    enum class LSRegressionBasis {
        LaguerreP02, ///< Order 2 Laguerre polynomial.
        LaguerreP03, ///< Order 3 Laguerre polynomial.
        LaguerreP04, ///< Order 4 Laguerre polynomial.
    };

    /** @brief Supported Greeks. */
    enum class Greeks {
        Delta, ///< The sensitivity to underlying spot price and gamma, Rate of change of Delta. Computed using Central Difference on Spot price.
        Gamma,
        Vega, ///< The sensitivity to volatility. Computed using Central Difference on Initial Volatility.
        Vomma,
        Rho, ///< The sensitivity to interest rates. Computed using Central Difference on Risk-Free Rate.
        Theta, ///< The Sensitivity to final time.
        Vanna, ///< Cross-sensitivity between spot price and volatility (dDelta/dVol or dVega/dS)
        //Speed
    };
}
