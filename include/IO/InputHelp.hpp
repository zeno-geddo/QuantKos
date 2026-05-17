#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "ConfigKeys.hpp"
#include "ConfigKeysEnumMaps.hpp"

namespace KOps::HELP {

    namespace K = KOps::Keys;
    namespace KI = KOps::Implemented;

    // ------------------------------------------------------------------------
    // Helper to generate a string of allowed options: "# [Opt1, Opt2, ...]"
    // ------------------------------------------------------------------------
    template <typename T>
    std::string get_allowed_options() {
        std::stringstream ss;
        ss << "# [";
        const auto& map = KI::StrEnumMap<T>::get();
        size_t i = 0;
        for (const auto& pair : map) {
            ss << pair.first;
            if (i < map.size() - 1) ss << ", "; // Avoid inserting a comma also after the last entry
            i++;
        }
        ss << "]";
        return ss.str();
    }

    inline void print_usage(const char* prog_name) {
        std::cout << "Usage: " << prog_name << " <config_file.yaml>\n\n";
    }

    inline void print_example_config() {
        std::cout << "--- Template Configuration File ---\n"
                  << "# Copy this structure into your .yaml file, and chose one one of the implementation between '[' and ']' ...\n\n";


        // ---------------------------------------------------------
        // Model Section
        // ---------------------------------------------------------
        std::cout << K::Model << ":\n"
                  << "  " << K::MathModelParams::IDHestonBlock <<" :\n"
                  << "    " << K::MathModelParams::r << ": 0.05     # Risk-free interest rate\n"
                  << "    " << K::MathModelParams::q << ": 0.       # Continuous dividend yield\n"
                  << "    " << K::MathModelParams::k << ": 2.       # Mean reversion speed of the variance (kappa)\n"
                  << "    " << K::MathModelParams::theta << ": 0.04 # Long-term mean of the variance\n"
                  << "    " << K::MathModelParams::sigma << ": 0.3  # Volatility of the variance (vol-of-vol)\n"
                  << "    " << K::MathModelParams::rho << ": 0.7    # Correlation between price and variance Brownian motions\n"

                  << "\n  # Parameters of Bates model\n"
                  << "  # ...to be done ...\n";

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
                  << "  " << K::MCParams::N_Realizations << ": 1000     # Number of Monte Carlo realizations/paths\n"
                  << "  " << K::MCParams::Batch_Size << ": 0            # Number of paths run in parallel before saving (0 means auto-computed based on hardware)\n"
                  << "  " << K::MCParams::Max_VRAM_MB << ": 256         # Max Available VRAM \n"
                  << "  " << K::MCParams::Max_CPU_RAM_MB << ": 4000     # Max Available RAM \n";

        // Output Section
        // ---------------------------------------------------------
        std::cout << K::Output << ":\n"
                  << "  " << K::OutParams::Name_Out_File << ": \"output\"\n"
                  << "  " << K::OutParams::Name_Log_File << ": \"sim.log\"\n"
                  << "  " << K::OutParams::Format << ":  " << get_allowed_options<KI::IOFormat>() << "\n";
    }
}