// File (YAML) → Parser → Validator → Config → Solver

#pragma once

#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include "ConfigEnums.hpp"

namespace KOps::Implemented {
    // --------------------------------------------------------
    // 1. The "Dictionary" Holders to Map Strings to Enums
    // --------------------------------------------------------
    template<typename T> // (For any type T, a function get will be provided.)
    struct StrEnumMap {
        static const std::map<std::string, T> &get();
    };

    // --- Specialization: MathModel ---
    template<>
    inline const std::map<std::string, MathModel> &StrEnumMap<MathModel>::get() {
        static const std::map<std::string, MathModel> m = {
            {"Heston", MathModel::Heston},
            {"Bates", MathModel::Bates},
        };
        return m;
    }

    // --- Specialization: NumScheme ---
    template<>
    inline const std::map<std::string, NumScheme> &StrEnumMap<NumScheme>::get() {
        static const std::map<std::string, NumScheme> m = {
            {"Euler", NumScheme::Euler},
            {"Milstein", NumScheme::Milstein},
            {"AndersonQE", NumScheme::AndersonQE}
        };
        return m;
    }


    // --- Specialization: IOFormat ---
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
