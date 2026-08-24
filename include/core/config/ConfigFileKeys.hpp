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

#include <string_view>

// ------------------------------------------------------------------------
// Entries of the yaml input file
// ------------------------------------------------------------------------

/**
 * @brief Namespace containing string constants for YAML configuration keys.
 * Modifications here must be synchronized with the parser implementation.
 */
namespace quantkos::Keys {
    // ------------------------------------------------------------------------
    // Top-Level Blocks
    // ------------------------------------------------------------------------
    /** @name Top-Level Configuration Blocks of YAML File */
    ///@{
    static constexpr std::string_view Market = "Market"; ///< Key Market block.
    static constexpr std::string_view Options = "Options"; ///< Key Options block.
    static constexpr std::string_view Model = "Model"; ///< Key SDE model block.
    static constexpr std::string_view Numerics = "Numerics"; ///< Key Numerics block.
    static constexpr std::string_view Time = "Time"; ///< Key Time block.
    static constexpr std::string_view MC = "MC"; ///< Key Monte Carlo block.
    static constexpr std::string_view Output = "Output"; ///< Key Output block.
    ///@}

    // ------------------------------------------------------------------------
    // Initialization Parameters (Initial Conditions)
    // ------------------------------------------------------------------------
    /**
     * @brief Keys for initial market conditions (expected in the Market block).
     */
    namespace MarketParams {
        static constexpr std::string_view Price = "Price"; ///< Key Initial spot price (S0)
        static constexpr std::string_view Variance = "Variance"; ///< Key Initial variance (v0).
        static constexpr std::string_view r = "r"; ///< Key Annualized risk-free rate.
        static constexpr std::string_view q = "q"; ///< Key Annualized dividend yield.
    }

    // ------------------------------------------------------------------------
    // Option Parameters
    // ------------------------------------------------------------------------
    /**
     * @brief Keys for options types and parameters (expected in the options block).
     */
    namespace OptionsParams {
        static constexpr std::string_view OptionType = "OptionType";
        static constexpr std::string_view OptionRight = "OptionRight";
        static constexpr std::string_view StrikePrice = "StrikePrice";
        static constexpr std::string_view BarrierPrice = "BarrierPrice";
    }

    // ------------------------------------------------------------------------
    // Model Parameters
    // ------------------------------------------------------------------------
    /**
     * @brief Keys for SDE type and parameters (expected in the Model block).
     */
    namespace MathModelParams {
        // Possible Sub-Blocks
        static constexpr std::string_view IDHestonBlock = "Heston";
        static constexpr std::string_view IDBatesBlock = "Bates";

        // Heston Sub-block (present also in Bates Sub-block)
        static constexpr std::string_view k = "k"; ///< Key Mean reversion speed of the variance
        static constexpr std::string_view theta = "theta"; ///< Key Mean reversion level of the variance
        static constexpr std::string_view sigma = "sigma"; ///< Key Volatility of the variance
        static constexpr std::string_view rho = "rho"; ///< Key Correlation between price and varaince brownian motions

        // Bates Sub-block
        static constexpr std::string_view lambda_J = "lambda_J"; ///< Key Jump Intensity (λ)
        static constexpr std::string_view mu_J = "mu_J"; ///< Key Mean Jump Size (μJ)
        static constexpr std::string_view sigma_J = "sigma_J"; ///< Key Jump Volatility (σJ)


    }

    // ------------------------------------------------------------------------
    // Numerical Parameters
    // ------------------------------------------------------------------------
    /**
     * @brief Keys for Numerical Scheme type and parameters (expected in the Numerics block).
     */
    namespace NumSchemeParams {
        // Main Block
        static constexpr std::string_view Scheme = "Scheme"; ///< Key ID numerical scheme
    }

    // ------------------------------------------------------------------------
    // Time Integration Parameters
    // ------------------------------------------------------------------------
    /**
     * @brief Keys for time stepping parameters (expected in the Time block).
     */
    namespace TimeParams {
        static constexpr std::string_view T_End = "T_End"; ///< Key for option expiry time
        static constexpr std::string_view Inp_DT = "Inp_DT";
        static constexpr std::string_view DT = "DT";
        static constexpr std::string_view N_TSteps = "N_TSteps";
    }

    // ------------------------------------------------------------------------
    // MonteCarlo Parameters
    // ------------------------------------------------------------------------
    /**
     * @brief Keys for MonteCarlo parameters (expected in the MC block).
     */
    namespace MCParams {
        static constexpr std::string_view normalize_prices = "Normalize_prices"; ///< Key for bool for normalizing prices.
        static constexpr std::string_view N_Realizations = "N_Paths"; ///< Key for Total number of SDE realizations.
        static constexpr std::string_view Batch_Size = "Batch_Size"; ///< Key number of paths per kernel launch.
        static constexpr std::string_view RNG_Seed = "RNG_Seed"; ///< Key Initial seed for RNG.
        static constexpr std::string_view Max_VRAM_MB = "Max_VRAM_MB"; ///< Key for limit on GPU VRAM allocation.
        static constexpr std::string_view Max_CPU_RAM_MB = "Max_CPU_RAM_MB"; ///< Limit on CPU host memory allocation.
        // Payoffs
        static constexpr std::string_view analyze_risk_neutral_payoff_distribution = "analyze_risk_neutral_payoff_distribution"; ///< Key to analyze the distribution of the risk-neutral discounted payoffs
        // Greeks
        static constexpr std::string_view compute_delta_et_gamma = "compute_delta_et_gamma"; ///< Key for computing delta and gamma (greeks).
        static constexpr std::string_view compute_vega = "compute_vega"; ///< Key for computing vega (greeks).
        static constexpr std::string_view compute_rho = "compute_rho"; ///< Key for computing rho (greeks).
        static constexpr std::string_view compute_theta = "compute_theta"; ///< Key for computing theta (greeks).
        static constexpr std::string_view spot_price_relative_bump_size = "spot_price_relative_bump_size"; ///< Key to set delta and gamma bump size (greeks).
        static constexpr std::string_view volatility_absolute_bump_size = "volatility_absolute_bump_size"; ///< Key to set vega bump size (greeks).
        static constexpr std::string_view risk_free_rate_absolute_bump_size = "risk_free_rate_absolute_bump_size"; ///< Key to set rho bump size (greeks).
        static constexpr std::string_view time_absolute_bump_size = "time_absolute_bump_size"; ///< Key to set theta bump size (greeks).
    }

    // ------------------------------------------------------------------------
    // Output Parameters
    // ------------------------------------------------------------------------
    /**
     * @brief Keys for Outputs parameters (expected in the Outputs block).
     */
    namespace OutParams {
        static constexpr std::string_view out_dir = "out_dir"; ///< Key for Target directory for generated logs and paths.
        static constexpr std::string_view Name_Log_File = "Name_Log_File"; ///< Key for Filename for the simulated path storage.
        static constexpr std::string_view Name_Paths_Out_File = "Name_Paths_Out_File"; ///< Key for Filename for the runtime execution log.
        static constexpr std::string_view Format = "Format"; ///< Key for Output Data format.
    }
} // namespace Labes::Keys
