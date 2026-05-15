// File (YAML) → Parser → Validator → Config → Solver

#pragma once

#include <string_view>

// ------------------------------------------------------------------------
// Entries of the yaml input file
// ------------------------------------------------------------------------


namespace KOps::Keys {
    // ------------------------------------------------------------------------
    // Top-Level Blocks
    // ------------------------------------------------------------------------
    static constexpr std::string_view Model = "Model";
    static constexpr std::string_view Init = "Init";
    static constexpr std::string_view Numerics = "Numerics";
    static constexpr std::string_view Time = "Time";
    static constexpr std::string_view MC = "MC";
    static constexpr std::string_view Output = "Output";

    // ------------------------------------------------------------------------
    // Model Parameters
    // ------------------------------------------------------------------------
    namespace MathModelParams {
        // Possible Sub-Blocks
        static constexpr std::string_view HestonBlock = "Heston";
        static constexpr std::string_view BatesBlock = "Bates";

        // Heston Sub-block
        static constexpr std::string_view r = "r"; //
        static constexpr std::string_view q = "q"; //
        static constexpr std::string_view k = "k"; // Mean reversion speed of the variance
        static constexpr std::string_view theta = "theta"; // Mean reversion level of the variance
        static constexpr std::string_view sigma = "sigma"; // Volatility of the variance
        static constexpr std::string_view rho = "rho"; // Correlation between price and varaince brownian motions

        // Bates Sub-block
        // ...to be done ...

    }

    // ------------------------------------------------------------------------
    // Initialization Parameters (Initial Conditions)
    // ------------------------------------------------------------------------
    namespace InitParams {
        static constexpr std::string_view Price = "Price"; // Initial Price
        static constexpr std::string_view Variance = "Variance"; // Initial Variance
    }


    // ------------------------------------------------------------------------
    // Numerical Parameters
    // ------------------------------------------------------------------------
    namespace NumSchemeParams {
        // Main Block
        static constexpr std::string_view Scheme = "Scheme";
    }

    // ------------------------------------------------------------------------
    // Time Integration Parameters
    // ------------------------------------------------------------------------
    namespace TimeParams {
        static constexpr std::string_view T_End = "T_End";
        static constexpr std::string_view DT = "DT";
    }

    // ------------------------------------------------------------------------
    // MonteCarlo Parameters
    // ------------------------------------------------------------------------
    namespace MCParams {
        static constexpr std::string_view N_Realizations = "N_Paths";
    }


    // ------------------------------------------------------------------------
    // Output Parameters
    // ------------------------------------------------------------------------
    namespace OutParams {
        static constexpr std::string_view N_Paths_Out_Batches = "N_Paths_Out_Batches";
        static constexpr std::string_view Name_Log_File = "Name_Log_File";
        static constexpr std::string_view Name_Out_File = "Name_Out_File";
        static constexpr std::string_view Format = "Format";
    }
} // namespace Labes::Keys
