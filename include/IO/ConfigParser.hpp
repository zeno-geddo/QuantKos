# pragma once

#include <string>
#include "Config.hpp"

namespace KOps::Config {

    class Parser {
    public:
        /**
         * @brief Parses a YAML configuration file.
         * @param filename Path to the .yaml file.
         * @return A fully populated Config object.
         * @throws std::runtime_error if file is missing or invalid.
         */
        static UInputs parse(const std::string& filename);
    };

} // namespace Labes