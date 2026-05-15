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
                  << "  # Parameters of Heston model\n"
                  << "  " << K::MathModelParams::r << ": 1.\n"
                  << "  " << K::MathModelParams::q << ": 1.\n"
                  << "  " << K::MathModelParams::k << ": 1.\n"
                  << "  " << K::MathModelParams::theta << ": 1.\n"
                  << "  " << K::MathModelParams::sigma << ": 1.\n"
                  << "  " << K::MathModelParams::rho << ": 1.\n"

                  << "\n  # Parameters of Bates model\n"
                  << "  # ...to be done ...\n";

        // ---------------------------------------------------------
        // Initialization Section
        // ---------------------------------------------------------
        std::cout << K::Init << ":\n"
                  << "  " << K::InitParams::Price << ": 100.\n "
                  << "  " << K::InitParams::Variance << ": 1.\n";
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
                  << "  " << K::TimeParams::T_End << ": 6000.\n"
                  << "  " << K::TimeParams::DT << ": 60.\n";

        // ---------------------------------------------------------
        // MC Section
        // ---------------------------------------------------------
        std::cout << K::MC << ":\n"
                  << "  " << K::MCParams::N_Realizations << ": 1000\n";

        // Output Section
        // ---------------------------------------------------------
        std::cout << K::Output << ":\n"
                  << "  " << K::OutParams::N_Paths_Out_Batches << ": 200\n"
                  << "  " << K::OutParams::Name_Out_File << ": \"output\"\n"
                  << "  " << K::OutParams::Name_Log_File << ": \"sim.log\"\n"
                  << "  " << K::OutParams::Format << ":  " << get_allowed_options<KI::IOFormat>() << "\n";
    }
}