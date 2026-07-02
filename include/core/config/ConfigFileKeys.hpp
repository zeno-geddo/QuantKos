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
    static constexpr std::string_view Options = "Options";
    static constexpr std::string_view Model = "Model";
    static constexpr std::string_view Init = "Init";
    static constexpr std::string_view Numerics = "Numerics";
    static constexpr std::string_view Time = "Time";
    static constexpr std::string_view MC = "MC";
    static constexpr std::string_view Output = "Output";

    // ------------------------------------------------------------------------
    // Option Parameters
    // ------------------------------------------------------------------------
    namespace OptionsParams {
        static constexpr std::string_view OptionType = "OptionType";
        static constexpr std::string_view OptionRight = "OptionRight";
        static constexpr std::string_view StrikePrice = "StrikePrice";
        static constexpr std::string_view BarrierPrice = "BarrierPrice";
    }

    // ------------------------------------------------------------------------
    // Model Parameters
    // ------------------------------------------------------------------------
    namespace MathModelParams {
        // Possible Sub-Blocks
        static constexpr std::string_view IDHestonBlock = "Heston";
        static constexpr std::string_view IDBatesBlock = "Bates";

        // Heston Sub-block (present also in Bates Sub-block)
        static constexpr std::string_view r = "r"; // Risk-free interest rate
        static constexpr std::string_view q = "q"; // Continuous dividend yield
        static constexpr std::string_view k = "k"; // Mean reversion speed of the variance
        static constexpr std::string_view theta = "theta"; // Mean reversion level of the variance
        static constexpr std::string_view sigma = "sigma"; // Volatility of the variance
        static constexpr std::string_view rho = "rho"; // Correlation between price and varaince brownian motions

        // Bates Sub-block
        static constexpr std::string_view lambda_J = "lambda_J"; // Jump Intensity (λ)
        static constexpr std::string_view mu_J = "mu_J"; // Mean Jump Size (μJ)
        static constexpr std::string_view sigma_J = "sigma_J"; // Jump Volatility (σJ)


    }

    // ------------------------------------------------------------------------
    // Initialization Parameters (Initial Conditions)
    // ------------------------------------------------------------------------
    namespace InitParams {
        static constexpr std::string_view Price = "Price"; // Initial Price (Spot Price)
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
        static constexpr std::string_view Inp_DT = "Inp_DT";
        static constexpr std::string_view DT = "DT";
        static constexpr std::string_view N_TSteps = "N_TSteps";
    }

    // ------------------------------------------------------------------------
    // MonteCarlo Parameters
    // ------------------------------------------------------------------------
    namespace MCParams {
        static constexpr std::string_view N_Realizations = "N_Paths";
        static constexpr std::string_view Batch_Size = "Batch_Size";
        static constexpr std::string_view RNG_Seed = "RNG_Seed";
        static constexpr std::string_view Max_VRAM_MB = "Max_VRAM_MB";
        static constexpr std::string_view Max_CPU_RAM_MB = "Max_CPU_RAM_MB";
    }


    // ------------------------------------------------------------------------
    // Output Parameters
    // ------------------------------------------------------------------------
    namespace OutParams {
        static constexpr std::string_view out_dir = "out_dir";
        static constexpr std::string_view Name_Log_File = "Name_Log_File";
        static constexpr std::string_view Name_Paths_Out_File = "Name_Paths_Out_File";
        static constexpr std::string_view Format = "Format";
    }
} // namespace Labes::Keys
