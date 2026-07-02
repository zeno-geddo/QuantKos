# pragma once

#include <string>
#include "./../core/config/Config.hpp"

namespace KOps::Config {

    class Parser {
    public:

        // Prevent anyone from ever creating a 'Parser' object
        Parser() = delete;

        /**
         * @brief Parses a YAML configuration file.
         * @param file Path to the .yaml file.
         * @return A fully populated Config object.
         * @throws std::runtime_error if file is missing or invalid.
         */
        static UInputs parse(const std::string& file);
    };

} // namespace Labes