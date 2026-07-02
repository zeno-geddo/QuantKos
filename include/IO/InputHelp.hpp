#pragma once

#include <iostream>
#include <string>
#include <filesystem>
#include <string_view>
#include <vector>
#include <sstream>
#include "./../core/config/ConfigFileKeys.hpp"
#include "./../core/config/ConfigFileKeysEnumMaps.hpp"

namespace KOps::HELP {
    namespace K = KOps::Keys;
    namespace KI = KOps::Implemented;

    inline void print_welcome_msg() {
        constexpr std::string_view indent = "    ";

        std::cout << "\n\n" << indent << "====================================================================\n"
                << indent << "                              KOptions                              \n"
                << indent << "====================================================================\n"
                << indent << "    Framework         : Parallel SDE Option Pricing Engine          \n"
                << indent << "    Compute Backend   : C++ and Kokkos (for Performance Portability)\n"
                << indent << "--------------------------------------------------------------------\n"
                << indent << "    Author            : Zeno GEDDO                                  \n"
                << indent << "    Version           : v0.1.0 (Beta)                               \n"
                << indent << "    Build Year        : 2026                                        \n"
                << indent << "    License           : ...........                                 \n"
                << indent << "    Contact           : zeno.geddo@gmail.com                        \n"
                << indent << "====================================================================\n"
                << std::endl;
    }

    inline void print_usage(std::string_view executable_path) {
        // Extract just the filename out of the absolute path wrapper
        std::filesystem::path prog_path(executable_path);
        std::string filename = prog_path.filename().string();

        std::cout << "    [Usage Guide]\n"
                << "      Execution Command Syntax:\n"
                << "        ./" << filename << " <path_to_config_file.yaml>\n\n"
                << "      Example Command Usage:\n"
                << "        ./" << filename << " ../config/heston_euler.yaml\n\n"
                << "    --------------------------------------------------------\n"
                << "    Note: The configuration input file must be a validated \n"
                << "          YAML/JSON specification containing comprehensive \n"
                << "          model parameters .\n"
                << "    ========================================================\n"
                << std::endl;
    }


    // ------------------------------------------------------------------------
    // Helper to generate a string of allowed options: "# [Opt1, Opt2, ...]"
    // ------------------------------------------------------------------------
    template<typename T>
    std::string get_allowed_options() {
        std::stringstream ss;
        ss << "# [";
        const auto &map = KI::StrEnumMap<T>::get();
        size_t i = 0;
        for (const auto &pair: map) {
            ss << pair.first;
            if (i < map.size() - 1) ss << ", "; // Avoid inserting a comma also after the last entry
            i++;
        }
        ss << "]";
        return ss.str();
    }

    inline void print_example_config() {
        std::cout << "\n--- Template Configuration File ---\n"
                << "# Copy this structure into your .yaml file, and chose one one of the implementation between '[' and ']' ...\n\n";

        // ---------------------------------------------------------
        // Option Section
        // ---------------------------------------------------------
        std::cout << K::Options << ":\n"
                << "  " << K::OptionsParams::OptionType << "   : European   #" << get_allowed_options<KI::OptType>() << "\n"
                << "  " << K::OptionsParams::OptionRight << "  : Call       #" << get_allowed_options<KI::OptRight>() << "\n"
                << "  " << K::OptionsParams::StrikePrice << "  : 150        # Strike Price\n";


        // ---------------------------------------------------------
        // Model Section
        // ---------------------------------------------------------
        std::cout << K::Model << ":\n"
                << "  [" << K::MathModelParams::IDHestonBlock << " :\n"
                << "    " << K::MathModelParams::r << ": 0.05     # Risk-free interest rate\n"
                << "    " << K::MathModelParams::q << ": 0.       # Continuous dividend yield\n"
                << "    " << K::MathModelParams::k << ": 2.       # Mean reversion speed of the variance (kappa)\n"
                << "    " << K::MathModelParams::theta << ": 0.04 # Long-term mean of the variance\n"
                << "    " << K::MathModelParams::sigma << ": 0.3  # Volatility of the variance (vol-of-vol)\n"
                << "    " << K::MathModelParams::rho <<
                ": 0.7    # Correlation between price and variance Brownian motions\n"
                << "  ],\n"
                << "  [" << K::MathModelParams::IDHestonBlock << " :\n"
                << "    " << K::MathModelParams::r << ": 0.05     # Risk-free interest rate\n"
                << "    " << K::MathModelParams::q << ": 0.       # Continuous dividend yield\n"
                << "    " << K::MathModelParams::k << ": 2.       # Mean reversion speed of the variance (kappa)\n"
                << "    " << K::MathModelParams::theta << ": 0.04 # Long-term mean of the variance\n"
                << "    " << K::MathModelParams::sigma << ": 0.3  # Volatility of the variance (vol-of-vol)\n"
                << "    " << K::MathModelParams::rho <<
                ": 0.7    # Correlation between price and variance Brownian motions\n"
                << "    " << K::MathModelParams::lambda_J << ": 0.1  # Merton Jump Intensity (λ)\n"
                << "    " << K::MathModelParams::mu_J << ": -0.1  # Merton Mean Jump Size (μJ)\n"
                << "    " << K::MathModelParams::sigma_J << ": 0.15  # Merton Jump Volatility (σJ)\n"
                << "  ]\n";

        // ---------------------------------------------------------
        // Initialization Section
        // ---------------------------------------------------------
        std::cout << K::Init << ":\n"
                << "  " << K::InitParams::Price << ": 100.         # Initial asset price (S0)\n "
                << "  " << K::InitParams::Variance << ": 0.04      # Initial variance (v0)\n";
        // ---------------------------------------------------------


        // ---------------------------------------------------------
        // Numerics Section
        // ---------------------------------------------------------
        std::cout << K::Numerics << ":\n"
                << "  " << K::NumSchemeParams::Scheme << ": Scheme " << get_allowed_options<KI::NumScheme>() << "\n";

        // ---------------------------------------------------------
        // Time Section
        // ---------------------------------------------------------
        std::cout << K::Time << ":\n"
                << "  " << K::TimeParams::T_End << ": 1.       # Time to maturity (in years)\n"
                << "  " << K::TimeParams::Inp_DT << ": 0.005   # Time step size (dt)\n";

        // ---------------------------------------------------------
        // MC Section
        // ---------------------------------------------------------
        std::cout << K::MC << ":\n"
                << "  " << K::MCParams::N_Realizations << ": 1000       # Number of Monte Carlo realizations/paths\n"
                << "  " << K::MCParams::Batch_Size <<
                ": 0              # Number of paths run in parallel before saving (0 means auto-computed based on hardware, -1 means all paths)\n"
                << "  " << K::MCParams::RNG_Seed <<
                ": 184467440737095  # Random Number Generator Seed (uint62_t, must be > 0)\n"
                << "  " << K::MCParams::Max_VRAM_MB << ": 256           # Max Available VRAM \n"
                << "  " << K::MCParams::Max_CPU_RAM_MB << ": 4000       # Max Available RAM \n";

        // Output Section
        // ---------------------------------------------------------
        std::cout << K::Output << ":\n"
                << "  " << K::OutParams::out_dir << ": \"install/outputs\"\n"
                << "  " << K::OutParams::Name_Paths_Out_File << ": \"KOptionsPaths.paths\"\n"
                << "  " << K::OutParams::Name_Log_File << ": \"KOptions.log\"\n"
                << "  " << K::OutParams::Format << ":  " << get_allowed_options<KI::IOFormat>() << "\n";
    }
}
