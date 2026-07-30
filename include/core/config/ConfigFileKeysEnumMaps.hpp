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

#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include "ConfigFileEnums.hpp"


/**
 * @brief Namespace containing utilities for bidirectional mapping between input configuration strings and internal enumerations.
 * * Provides a type-safe interface to serialize enums for logging/output and deserialize
 * strings provided in configuration files into C++ enum members.
 *
 * File (YAML) → Parser → Current Validator → Config → Solver
 */
namespace KOps::Implemented {
    // --------------------------------------------------------
    // 1. The "Dictionary" Holders to Map Strings to Enums
    // --------------------------------------------------------
    /**
     * @brief Template trait to register string-to-enum mappings.
     * @tparam T The enum type to be mapped.
     */
    template<typename T> // (For any type T, a function get will be provided.)
    struct StrEnumMap {
        /** @brief Returns a reference to the static map for the requested type. */
        static const std::map<std::string, T> &get();
    };

    // --- Specialization: OptionType ---
    /** @brief Specialization Mapping string identifiers to the supported financial option payoff structures. */
    template<>
    inline const std::map<std::string, OptType> &StrEnumMap<OptType>::get() {
        static const std::map<std::string, OptType> m = {
            // Standard Options
            {"European",               OptType::European},
            {"Asian",                  OptType::Asian},

            // Barrier Options
            {"BarrierUpAndOut",        OptType::BarrierUpAndOut},
            {"BarrierDownAndOut",      OptType::BarrierDownAndOut},
            {"BarrierUpAndIn",         OptType::BarrierUpAndIn},
            {"BarrierDownAndIn",       OptType::BarrierDownAndIn},

            // Lookback Options
            {"LookbackFloatingStrike", OptType::LookbackFloatingStrike},
            {"LookbackFixedStrike",    OptType::LookbackFixedStrike},

            // Binary Options
            {"BinaryCashOrNothing",    OptType::BinaryCashOrNothing},
            {"BinaryAssetOrNothing",   OptType::BinaryAssetOrNothing},

            // Backward Path Dependent
            {"American",               OptType::American}
        };
        return m;
    }

    // --- Specialization: OptionRight ---
    /** @brief Specialization Mapping string identifiers to option exercise rights (Call/Put). */
    template<>
    inline const std::map<std::string, OptRight> &StrEnumMap<OptRight>::get() {
        static const std::map<std::string, OptRight> m = {
            {"Call", OptRight::Call},
            {"Put", OptRight::Put},
        };
        return m;
    }


    // --- Specialization: MathModel ---
    /** @brief Specialization Mapping string identifiers to available SDE models. */
    template<>
    inline const std::map<std::string, MathModel> &StrEnumMap<MathModel>::get() {
        static const std::map<std::string, MathModel> m = {
            {"Heston", MathModel::Heston},
            {"Bates", MathModel::Bates},
        };
        return m;
    }

    // --- Specialization: NumScheme ---
    /** @brief Specialization Mapping string identifiers to available Numerical Schemes. */
    template<>
    inline const std::map<std::string, NumScheme> &StrEnumMap<NumScheme>::get() {
        static const std::map<std::string, NumScheme> m = {
            {"Euler", NumScheme::Euler},
            {"ImplicitMilstein", NumScheme::ImplicitMilstein},
            {"AndersonQE", NumScheme::AndersonQE}
        };
        return m;
    }


    // --- Specialization: IOFormat ---
    /** @brief Specialization Mapping string identifiers to available Output Formats. */
    template<>
    inline const std::map<std::string, IOFormat> &StrEnumMap<IOFormat>::get() {
        static const std::map<std::string, IOFormat> m = {
            {"BIN", IOFormat::BIN},
            {"TXT", IOFormat::TXT}
        };
        return m;
    }

    // --------------------------------------------------------
    // 2. Enum -> String
    // --------------------------------------------------------
    /**
     * @brief Converts an enum value to its string representation.
     * @tparam T The enum type.
     * @param value The enum instance to convert.
     * @return The string name of the enum, or "Unknown" if no mapping exists.
     */
    template<typename T>
    std::string enum_to_string(T value) {
        const auto &m = StrEnumMap<T>::get();
        for (const auto &pair: m) {
            if (pair.second == value) {
                return pair.first;
            }
        }
        return "Unknown";
    }

    // --------------------------------------------------------
    // 3. Operator Overloads (allows 'std::cout << MyEnum')
    // --------------------------------------------------------
    inline std::ostream &operator<<(std::ostream &os, const MathModel e) { return os << enum_to_string(e); }
    inline std::ostream &operator<<(std::ostream &os, const NumScheme e) { return os << enum_to_string(e); }
    inline std::ostream &operator<<(std::ostream &os, const IOFormat e) { return os << enum_to_string(e); }

    // --------------------------------------------------------
    // 4. String -> Enum
    // --------------------------------------------------------
    /**
     * @brief Converts a string configuration value to its corresponding enum member.
     * * Validates that the input string exists in the registered mapping; otherwise,
     * throws a descriptive runtime error listing all allowed options.
     * * @tparam T The target enum type.
     * @param input The string value retrieved from the input file.
     * @param field_name Name of the configuration field (used for descriptive error messages).
     * @return The corresponding enum member.
     * @throw std::runtime_error If the input string is invalid.
     */
    template<typename T>
    T string_to_enum(const std::string &input, const std::string &field_name) {
        const auto &m = StrEnumMap<T>::get();
        auto it = m.find(input);

        if (it != m.end()) {
            return it->second;
        }

        // Error handling inside the utility
        std::string err = "Config Error: Invalid value '" + input + "' for " + field_name + ".\n";
        err += "Allowed options: [ ";
        for (const auto &pair: m) err += pair.first + " ";
        err += "]";
        throw std::runtime_error(err);
    }

} // namespace Labes
